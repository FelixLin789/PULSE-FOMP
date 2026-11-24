"""PULSE-v7: Hierarchical Hybrid (5:1 GDN-2 + Gated NoPE-MLA) & Pure-O(1) Rank-Expanded Engine.

Implements the PULSE-v7 architecture blueprint (Stage 0 + Stage 1 primitives, ready for Stages 2-4):
- `patch_fla_sm89_shared_memory`: unlocks large-tile FLA Triton kernels on 99 KiB Ada/L4 (`sm_89`).
- `GatedDeltaNet2Block`: channel-wise decay (alpha_t), decoupled key-erase (b_t in (0,2)) &
  value-write (w_t in (0,2)) with zero erasure floor, optional Triadic tensor expansion (E=1,2,4),
  and Canon-A/C/D residual convs.
- `GatedSWABlock`: W=128 sliding window with QK-Norm, FlexAttention block-sparse fusion,
  and NeurIPS 2025 Best Paper multiplicative Sigmoid output gate.
- `GatedNoPEMLABlock`: global NoPE Multi-Head Latent Attention (DeepSeek MLA low-rank KV latent +
  QK-Norm + multiplicative Sigmoid gate) for unbounded K > d_head retrieval with ~46x KV compression
  and zero RoPE extrapolation collapse.
- Pure-O(1) Stage-1 rank-wall explorers: `ToasterMemoryBlock` (2-layer chunk target-prop MLP memory)
  and `RAMNetMemoryBlock` (S=128 sparse Top-8 slot RAM).
- `PULSEv7Config` / `PULSEv7Model`: clean regional compilation (`compile_fast_paths`), Qwen3.8
  split-matrix `FairMuonAdamW` tagging (`_no_muon = True` on all gates/convs), and strictly zero-leak
  state cloning verified across 512 / 4096 / 8192.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import types
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.pulse_v4 import (
    _RAW_FLEX_ATTN,
    _SDPA_HAS_ENABLE_GQA,
    _FastRMSNorm,
    _build_swa_block_mask,
    _canon4_inductor,
)
from pulse.scalable_transformer import _apply_rope
from pulse.v3_blocks import (
    _get_flex_attention,
    _probe_fla_dplr_backend,
    _probe_fla_gdn_backend,
    _probe_fla_kda_backend,
)


def patch_fla_sm89_shared_memory() -> Dict[str, Any]:
    """Check installed `fla` for 102400 (100 KiB) shared-memory gate and lower DEFAULT to 101376 (99 KiB) on sm_89."""
    info: Dict[str, Any] = {"cuda": torch.cuda.is_available(), "patched": [], "checked_modules": []}
    if not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    max_smem = int(getattr(props, "shared_memory_per_block_optin", getattr(props, "shared_memory_per_block", 0)))
    info.update(device=props.name, compute_capability=f"{cc[0]}.{cc[1]}", max_smem_optin=max_smem)
    try:
        import fla.utils as fla_utils
    except Exception as exc:
        info["fla_import_error"] = str(exc)
        return info

    info["checked_modules"].append("fla.utils")
    check_fn = getattr(fla_utils, "check_shared_mem", None)
    if callable(check_fn):
        try:
            info["before_check_none"] = bool(check_fn("none"))
            info["before_check_ada"] = bool(check_fn("ada"))
        except Exception as exc:
            info["before_check_err"] = str(exc)

    if max_smem < 101376 or max_smem >= 102400:
        return info

    target = min(max_smem, 101376)
    backend_cls = getattr(fla_utils, "Backend", None)
    if backend_cls is not None and hasattr(backend_cls, "get_shared_memory"):
        orig_get = backend_cls.get_shared_memory
        if not getattr(orig_get, "_pulse_sm89_patched", False):
            def _patched_get_shared_memory(arch: str = "none") -> int:
                val = orig_get(arch)
                return target if val == 102400 else val

            _patched_get_shared_memory._pulse_sm89_patched = True  # type: ignore[attr-defined]
            backend_cls.get_shared_memory = staticmethod(_patched_get_shared_memory)
            info["patched"].append(f"fla.utils.Backend.get_shared_memory: 102400 -> {target}")

    for attr in dir(fla_utils):
        try:
            val = getattr(fla_utils, attr)
        except Exception:
            continue
        if isinstance(val, int) and val == 102400:
            setattr(fla_utils, attr, target)
            info["patched"].append(f"fla.utils.{attr}: 102400 -> {target}")
        elif isinstance(val, dict):
            for k, v in list(val.items()):
                if isinstance(v, int) and v == 102400:
                    val[k] = target
                    info["patched"].append(f"fla.utils.{attr}[{k!r}]: 102400 -> {target}")

    if callable(check_fn) and hasattr(check_fn, "cache_clear"):
        check_fn.cache_clear()
        try:
            info["after_check_none"] = bool(check_fn("none"))
            info["after_check_ada"] = bool(check_fn("ada"))
        except Exception as exc:
            info["after_check_err"] = str(exc)
    return info


def _mark_no_muon(module: nn.Module) -> None:
    for p in module.parameters():
        p._no_muon = True  # type: ignore[attr-defined]


class _Canon4(nn.Module):
    """Inductor-fusable 4-tap causal depthwise residual convolution (Canon-A/C/D, strictly O(1) 3-tok state)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.zeros(4, dim))
        self.weight._no_muon = True  # type: ignore[attr-defined]

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, 3, self.dim, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None, return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_len = x.shape[1]
        if state is None:
            y = _canon4_inductor(x, self.weight, apply_silu=False)
            if not return_state:
                return y, None
            if seq_len >= 3:
                new_st = x[:, -3:, :].detach().clone(memory_format=torch.contiguous_format)
            else:
                new_st = F.pad(x, (0, 0, 3 - seq_len, 0), value=0.0).detach().clone(
                    memory_format=torch.contiguous_format
                )
            return y, new_st

        x_cat = torch.cat([state.to(x.dtype), x], dim=1)  # [B, 3 + T, D]
        w = self.weight.to(x.dtype)
        y = (
            x * (w[3] + 1.0)
            + x_cat[:, 2 : 2 + seq_len, :] * w[2]
            + x_cat[:, 1 : 1 + seq_len, :] * w[1]
            + x_cat[:, 0:seq_len, :] * w[0]
        )
        new_st = (
            x_cat[:, -3:, :].detach().clone(memory_format=torch.contiguous_format) if return_state else None
        )
        return y, new_st


def _wy_chunk_gdn2_cpu(
    q: torch.Tensor,       # [B, T, H, Dk]
    k: torch.Tensor,       # [B, T, H, Dk]
    v_eff: torch.Tensor,   # [B, T, H, Dv]  (already multiplied by write gate w_t or beta_t * w_t)
    log_a: torch.Tensor,   # [B, T, H, Dk]  (channel-wise log decay <= 0)
    b_er: torch.Tensor,    # [B, T, H, Dk]  (channel-wise erase gate in [0, 2])
    S_init: torch.Tensor,  # [B, H, Dv, Dk]
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact channel-wise GDN-2 recurrence: S_t = S_{t-1} Diag(a_t) (I - (b_t*k_t) k_t^T) + v_eff k_t^T.

    Uses exact blockwise WY when log_a is scalar-broadcast across Dk, and exact channel-wise WY
    (via per-chunk lower-triangular solve) when log_a varies across channels Dk.
    """
    bsz, seq_len, n_heads, dk = q.shape
    dv = v_eff.shape[-1]
    c_sz = chunk_size
    rem = seq_len % c_sz
    pad_len = (c_sz - rem) if rem != 0 else 0
    with torch.amp.autocast(device_type=q.device.type, enabled=False):
        q_f = q.transpose(1, 2).float()      # [B, H, T, Dk]
        k_f = k.transpose(1, 2).float()
        v_f = v_eff.transpose(1, 2).float()  # [B, H, T, Dv]
        la_f = log_a.transpose(1, 2).float() # [B, H, T, Dk]
        bk_f = (b_er * k).transpose(1, 2).float()  # [B, H, T, Dk]
        if pad_len > 0:
            q_f = F.pad(q_f, (0, 0, 0, pad_len), value=0.0)
            k_f = F.pad(k_f, (0, 0, 0, pad_len), value=0.0)
            v_f = F.pad(v_f, (0, 0, 0, pad_len), value=0.0)
            la_f = F.pad(la_f, (0, 0, 0, pad_len), value=0.0)
            bk_f = F.pad(bk_f, (0, 0, 0, pad_len), value=0.0)

        n_chunks = q_f.shape[2] // c_sz
        q_c = q_f.view(bsz, n_heads, n_chunks, c_sz, dk)
        k_c = k_f.view(bsz, n_heads, n_chunks, c_sz, dk)
        v_c = v_f.view(bsz, n_heads, n_chunks, c_sz, dv)
        la_c = la_f.view(bsz, n_heads, n_chunks, c_sz, dk)
        bk_c = bk_f.view(bsz, n_heads, n_chunks, c_sz, dk)

        g_c = la_c.cumsum(dim=-2)  # [B, H, C, cs, Dk]
        exp_g = torch.exp(g_c.clamp(-30.0, 0.0))  # [B, H, C, cs, Dk]
        idx = torch.arange(c_sz, device=q.device)
        strict_lower = idx.unsqueeze(1) > idx.unsqueeze(0)   # [cs, cs]
        causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)   # [cs, cs]
        eye_c = torch.eye(c_sz, device=q.device, dtype=torch.float32)

        # Channel-wise exact decay gram matrix: A[i,j] = sum_d (bk_i,d * exp(g_i,d - g_j,d) * k_j,d)
        # Scale by chunk-mid shift to avoid exp overflow when g_i - g_j is computed via ratio!
        g_shift = g_c[..., c_sz // 2 : c_sz // 2 + 1, :]  # [B, H, C, 1, Dk]
        e_pos = torch.exp((g_c - g_shift).clamp(-30.0, 30.0))
        e_neg = torch.exp((g_shift - g_c).clamp(-30.0, 30.0))

        # L_mat[i, j] = delta_ij + 1_{i>j} * (bk_i * e_pos_i) . (k_j * e_neg_j)
        bk_scaled = bk_c * e_pos
        k_inv = k_c * e_neg
        L_raw = torch.matmul(bk_scaled, k_inv.transpose(-1, -2))
        L_mat = eye_c + torch.where(strict_lower, L_raw, torch.zeros_like(L_raw))

        rhs = torch.cat([exp_g * bk_c, v_c], dim=-1)  # [..., cs, Dk + Dv]
        W_all, U_all = torch.linalg.solve_triangular(L_mat, rhs, upper=False, unitriangular=True).split([dk, dv], dim=-1)

        q_e = q_c * e_pos
        M_raw = torch.matmul(q_e, k_inv.transpose(-1, -2))
        M_causal = torch.where(causal_mask, M_raw, torch.zeros_like(M_raw))
        q_init = q_c * exp_g

        g_end = g_c[..., -1:, :]  # [B, H, C, 1, Dk]
        k_to_end = torch.exp((g_end - g_c).clamp(-30.0, 0.0)) * k_c
        decay_end = torch.exp(g_end.clamp(-30.0, 0.0))  # [B, H, C, 1, Dk]

        S = S_init.float()  # [B, H, Dv, Dk]
        out_chunks = []
        for c_idx in range(n_chunks):
            delta_cur = U_all[:, :, c_idx] - torch.matmul(W_all[:, :, c_idx], S.transpose(-1, -2))
            out_chunks.append(
                torch.matmul(q_init[:, :, c_idx], S.transpose(-1, -2))
                + torch.matmul(M_causal[:, :, c_idx], delta_cur)
            )
            S = S * decay_end[:, :, c_idx] + torch.matmul(delta_cur.transpose(-1, -2), k_to_end[:, :, c_idx])
        o_all = torch.cat(out_chunks, dim=2)[:, :, :seq_len, :].transpose(1, 2)  # [B, T, H, Dv]
    return o_all.to(q.dtype), S


class _CanonSwiGLUMLP(nn.Module):
    """SwiGLU MLP with lightweight Canon-C (pre-MLP) & Canon-D (pre-activation) causal residual convs."""

    def __init__(self, d_model: int, d_ff: int, use_canon_cd: bool = True) -> None:
        super().__init__()
        self.use_canon_cd = use_canon_cd
        self.norm2 = _FastRMSNorm(d_model)
        self.canon_c = _Canon4(d_model) if use_canon_cd else None
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.canon_d = _Canon4(2 * d_ff) if use_canon_cd else None
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.d_ff = d_ff
        for m in (self.w_gate, self.w_up):
            nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(self.w_down.weight)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        if not self.use_canon_cd:
            return {}
        assert self.canon_c is not None and self.canon_d is not None
        return {
            "conv_c": self.canon_c.init_state(batch_size, device, dtype),
            "conv_d": self.canon_d.init_state(batch_size, device, dtype),
        }

    def forward(
        self, x: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None, return_state: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n2 = self.norm2(x)
        new_st: Dict[str, torch.Tensor] = {}
        if self.canon_c is not None:
            sc = state["conv_c"] if state is not None else None
            n2, nsc = self.canon_c(n2, state=sc, return_state=return_state)
            if return_state and nsc is not None:
                new_st["conv_c"] = nsc
        if self.canon_d is None:
            return x + self.w_down(F.silu(self.w_gate(n2)) * self.w_up(n2)), new_st
        gu = F.linear(n2, torch.cat([self.w_gate.weight, self.w_up.weight], dim=0))
        sd = state["conv_d"] if state is not None else None
        gu, nsd = self.canon_d(gu, state=sd, return_state=return_state)
        if return_state and nsd is not None:
            new_st["conv_d"] = nsd
        g, u = gu.chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u), new_st


class GatedDeltaNet2Block(nn.Module):
    """PULSE-v7 GDN-2 Block: Channel-Wise Erase/Write Decoupled DeltaNet + Optional Triadic E + Canon-A/C/D.

    Supports three backend-aligned recurrence modes for Stage 0 hardware calibration:
      - `gdn2_dplr`: full channel-wise alpha_t, b_t in (0,2), w_t in (0,2) via FLA `chunk_dplr_delta_rule`.
      - `gdn2_kda`:  channel-wise alpha_t, head-wise zero-floor beta_t, channel-wise w_t via FLA `chunk_kda`.
      - `gdn_scalar`: head-wise alpha_t, head-wise zero-floor beta_t, head-wise w_t via FLA `chunk_gated_delta_rule`.
    """

    def __init__(
        self,
        d_model: int,
        embed_dim: int,
        n_heads: int,
        d_ff: int,
        recurrence_mode: Literal["gdn2_dplr", "gdn2_kda", "gdn_scalar"] = "gdn2_dplr",
        triadic_e: int = 1,
        gate_rank: int = 0,
        chunk_size: int = 64,
        use_canon_acd: bool = True,
        use_canon_cd: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.recurrence_mode = recurrence_mode
        self.triadic_e = int(triadic_e)
        self.dk = self.head_dim * self.triadic_e
        self.chunk_size = chunk_size
        self.is_global_mla = False
        self.is_swa = False
        canon_cd = use_canon_acd if use_canon_cd is None else use_canon_cd

        self.norm1 = _FastRMSNorm(d_model)
        self.canon_a = _Canon4(d_model) if use_canon_acd else None

        # Clean unfused Muon projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.qkv_conv = _Canon4(3 * d_model)

        if self.triadic_e > 1:
            self.ek_proj = nn.Linear(d_model, n_heads * self.triadic_e, bias=True)
            self.eq_proj = nn.Linear(d_model, n_heads * self.triadic_e, bias=True)
            _mark_no_muon(self.ek_proj)
            _mark_no_muon(self.eq_proj)
        else:
            self.ek_proj = None
            self.eq_proj = None

        self.ve_proj = nn.Linear(embed_dim, d_model, bias=False)
        self.ve_lam = nn.Parameter(torch.tensor(0.1))

        # Gate projections — ALL explicitly tagged `_no_muon = True` (Qwen3.8 / GLM-5.3 recipe!)
        gr = gate_rank if gate_rank > 0 else max(32, d_model // 8)
        if recurrence_mode == "gdn_scalar":
            self.gates_proj = nn.Linear(d_model, 4 * n_heads, bias=True)
            _mark_no_muon(self.gates_proj)
        elif recurrence_mode == "gdn2_kda":
            self.a_down = nn.Linear(d_model, gr, bias=False)
            self.a_up = nn.Linear(gr, d_model, bias=True)
            self.w_down_g = nn.Linear(d_model, gr, bias=False)
            self.w_up_g = nn.Linear(gr, d_model, bias=True)
            self.eg_proj = nn.Linear(d_model, 2 * n_heads, bias=True)
            for m in (self.a_down, self.a_up, self.w_down_g, self.w_up_g, self.eg_proj):
                _mark_no_muon(m)
        else:  # gdn2_dplr (full channel-wise alpha_t, b_t, w_t!)
            self.a_down = nn.Linear(d_model, gr, bias=False)
            self.a_up = nn.Linear(gr, d_model, bias=True)
            self.b_down = nn.Linear(d_model, gr, bias=False)
            self.b_up = nn.Linear(gr, d_model, bias=True)
            self.w_down_g = nn.Linear(d_model, gr, bias=False)
            self.w_up_g = nn.Linear(gr, d_model, bias=True)
            self.g_proj = nn.Linear(d_model, n_heads, bias=True)
            for m in (self.a_down, self.a_up, self.b_down, self.b_up, self.w_down_g, self.w_up_g, self.g_proj):
                _mark_no_muon(m)

        self.head_norm = _FastRMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = _CanonSwiGLUMLP(d_model, d_ff, use_canon_cd=canon_cd)

        self._fla_probe: Optional[Dict[str, Any]] = None
        self._fla_calls: int = 0
        self._fla_fallback_reason: Optional[str] = None
        self._c_pre: Optional[Any] = None
        self._c_post: Optional[Any] = None
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj, self.ve_proj):
            nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(self.o_proj.weight)
        h, d = self.n_heads, self.head_dim
        with torch.no_grad():
            decay_bias = torch.linspace(-0.5, -4.5, h).unsqueeze(-1).expand(h, d).reshape(-1)
            if self.recurrence_mode == "gdn_scalar":
                nn.init.normal_(self.gates_proj.weight, 0.0, 0.02)
                self.gates_proj.bias[:h].copy_(torch.linspace(-0.5, -4.5, h))
                self.gates_proj.bias[h : 2 * h].fill_(-1.0)
                self.gates_proj.bias[2 * h : 3 * h].fill_(0.0)
                self.gates_proj.weight[3 * h : 4 * h].zero_()
                self.gates_proj.bias[3 * h : 4 * h].fill_(1.5)
            elif self.recurrence_mode == "gdn2_kda":
                for m in (self.a_down, self.a_up, self.w_down_g, self.w_up_g, self.eg_proj):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                self.a_up.bias.copy_(decay_bias)
                self.w_up_g.bias.zero_()
                self.eg_proj.bias[:h].fill_(-1.0)
                self.eg_proj.weight[h:].zero_()
                self.eg_proj.bias[h:].fill_(1.5)
            else:
                for m in (self.a_down, self.a_up, self.b_down, self.b_up, self.w_down_g, self.w_up_g):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                self.a_up.bias.copy_(decay_bias)
                self.b_up.bias.fill_(-1.0)
                self.w_up_g.bias.zero_()
                self.g_proj.weight.zero_()
                self.g_proj.bias.fill_(1.5)
            if self.ek_proj is not None and self.eq_proj is not None:
                nn.init.normal_(self.ek_proj.weight, 0.0, 0.02)
                nn.init.normal_(self.eq_proj.weight, 0.0, 0.02)
                nn.init.zeros_(self.ek_proj.bias)
                nn.init.zeros_(self.eq_proj.bias)

    def compile_regional(self, device: torch.device) -> None:
        if device.type != "cuda":
            return
        if self.recurrence_mode == "gdn_scalar":
            self._fla_probe = _probe_fla_gdn_backend(device)
        elif self.recurrence_mode == "gdn2_kda":
            self._fla_probe = _probe_fla_kda_backend(device)
        else:
            self._fla_probe = _probe_fla_dplr_backend(device)
        if self._c_pre is None:
            try:
                self._c_pre = torch.compile(self._pre_fast, dynamic=False)
                self._c_post = torch.compile(self._post_fast, dynamic=False)
            except Exception:
                self._c_pre = self._c_post = None

    def _triadic_expand(self, x_dir: torch.Tensor, e_raw: torch.Tensor) -> torch.Tensor:
        # x_dir: [B, T, H, d], e_raw: [B, T, H*E] -> normalized vec(x_dir (x) e_dir) in [B, T, H, d*E]
        bsz, seq_len, h, d = x_dir.shape
        e_dir = F.normalize(e_raw.view(bsz, seq_len, h, self.triadic_e), p=2, dim=-1, eps=1e-6)
        return (x_dir.unsqueeze(-1) * e_dir.unsqueeze(-2)).reshape(bsz, seq_len, h, d * self.triadic_e)

    def _pre_impl(
        self, x_in: torch.Tensor, e_tok: torch.Tensor, state: Optional[Dict[str, torch.Tensor]], return_state: bool
    ):
        bsz, seq_len, _ = x_in.shape
        n1 = self.norm1(x_in)
        new_ca = None
        if self.canon_a is not None:
            n1, new_ca = self.canon_a(n1, state=state["conv_a"] if state is not None else None, return_state=return_state)

        w_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        qkv_raw = F.linear(n1, w_qkv)
        qkv_c, new_qb = self.qkv_conv(qkv_raw, state=state["conv_qkv"] if state is not None else None, return_state=return_state)
        q, k, v = F.silu(qkv_c).chunk(3, dim=-1)

        q_v = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k_v = k.view(bsz, seq_len, self.n_heads, self.head_dim)
        q_dir = q_v * torch.rsqrt(q_v.float().pow(2).sum(dim=-1, keepdim=True) + 1e-6).to(x_in.dtype)
        k_dir = k_v * torch.rsqrt(k_v.float().pow(2).sum(dim=-1, keepdim=True) + 1e-6).to(x_in.dtype)
        if self.ek_proj is not None and self.eq_proj is not None:
            q_dir = self._triadic_expand(q_dir, self.eq_proj(n1))
            k_dir = self._triadic_expand(k_dir, self.ek_proj(n1))

        v = v.view(bsz, seq_len, self.n_heads, self.head_dim)
        v = v + self.ve_lam.to(x_in.dtype) * self.ve_proj(e_tok).view(bsz, seq_len, self.n_heads, self.head_dim)

        if self.recurrence_mode == "gdn_scalar":
            a_raw, e_raw, w_raw, g_raw = self.gates_proj(n1).chunk(4, dim=-1)
            log_a = -F.softplus(a_raw.float()).clamp(1e-6, 10.0)           # [B, T, H]
            beta = (2.0 * torch.sigmoid(e_raw)).to(x_in.dtype)             # [B, T, H]
            w_val = (2.0 * torch.sigmoid(w_raw)).to(x_in.dtype)            # [B, T, H]
            v_use = (w_val.unsqueeze(-1) * v).to(x_in.dtype)               # coupled: FLA multiplies by beta!
            g_h = torch.sigmoid(g_raw).unsqueeze(-1)
            return q_dir, k_dir, v_use, log_a, beta, g_h, new_ca, new_qb
        elif self.recurrence_mode == "gdn2_kda":
            a_raw = self.a_up(F.silu(self.a_down(n1))).view(bsz, seq_len, self.n_heads, self.head_dim)
            w_raw = self.w_up_g(F.silu(self.w_down_g(n1))).view(bsz, seq_len, self.n_heads, self.head_dim)
            e_raw, g_raw = self.eg_proj(n1).chunk(2, dim=-1)
            log_a = -F.softplus(a_raw.float()).clamp(1e-6, 10.0)           # [B, T, H, d]
            # Kimi Linear KDA contract: beta in (0, 1) prevents BF16 WY inverse NaN under safe_gate=True
            beta = torch.sigmoid(e_raw).to(x_in.dtype)                     # [B, T, H]
            w_val = (2.0 * torch.sigmoid(w_raw)).to(x_in.dtype)            # [B, T, H, d]
            v_use = (w_val * v).to(x_in.dtype)                             # coupled: KDA multiplies by beta!
            g_h = torch.sigmoid(g_raw).unsqueeze(-1)
            return q_dir, k_dir, v_use, log_a, beta, g_h, new_ca, new_qb
        else:  # gdn2_dplr: full decoupled channel-wise alpha_t, b_t, w_t!
            a_raw = self.a_up(F.silu(self.a_down(n1))).view(bsz, seq_len, self.n_heads, self.head_dim)
            b_raw = self.b_up(F.silu(self.b_down(n1))).view(bsz, seq_len, self.n_heads, self.head_dim)
            w_raw = self.w_up_g(F.silu(self.w_down_g(n1))).view(bsz, seq_len, self.n_heads, self.head_dim)
            g_raw = self.g_proj(n1)
            log_a = -F.softplus(a_raw.float()).clamp(1e-6, 10.0)           # [B, T, H, d]
            b_er = (2.0 * torch.sigmoid(b_raw)).to(x_in.dtype)             # [B, T, H, d]
            w_val = (2.0 * torch.sigmoid(w_raw)).to(x_in.dtype)            # [B, T, H, d]
            v_eff = (w_val * v).to(x_in.dtype)                             # decoupled direct write w_t * v_t!
            g_h = torch.sigmoid(g_raw).unsqueeze(-1)
            return q_dir, k_dir, v_eff, log_a, b_er, g_h, new_ca, new_qb

    def _pre_fast(self, x_in: torch.Tensor, e_tok: torch.Tensor):
        q, k, v_in, log_a, gate_er, g_h = self._pre_impl(x_in, e_tok, state=None, return_state=False)[:6]
        return q.contiguous(), k.contiguous(), v_in.contiguous(), log_a.contiguous(), gate_er.contiguous(), g_h

    def _post_fast(self, x_in: torch.Tensor, o_all: torch.Tensor, g_h: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x_in.shape
        o_m = (self.head_norm(o_all.to(x_in.dtype)) * g_h.to(x_in.dtype)).contiguous().view(bsz, seq_len, self.d_model)
        x = x_in + self.o_proj(o_m)
        y, _ = self.mlp(x, state=None, return_state=False)
        return y

    def _run_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v_in: torch.Tensor,
        log_a: torch.Tensor,
        gate_er: torch.Tensor,
        S_in: Optional[torch.Tensor],
        return_state: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, h, dk = q.shape
        dv = self.head_dim
        probe = self._fla_probe
        if probe is None and q.is_cuda:
            if self.recurrence_mode == "gdn_scalar":
                probe = _probe_fla_gdn_backend(q.device)
            elif self.recurrence_mode == "gdn2_kda":
                probe = _probe_fla_kda_backend(q.device)
            else:
                probe = _probe_fla_dplr_backend(q.device)
            self._fla_probe = probe

        # Fast CUDA FLA Triton dispatch (zero-copy when already in target dtype & contiguous)
        if probe and probe.get("available", False) and q.is_cuda:
            kd = torch.bfloat16 if (q.dtype == torch.float32 and torch.cuda.is_bf16_supported()) else q.dtype
            q_k = q if (q.dtype == kd and q.is_contiguous()) else q.to(kd).contiguous()
            k_k = k if (k.dtype == kd and k.is_contiguous()) else k.to(kd).contiguous()
            v_k = v_in if (v_in.dtype == kd and v_in.is_contiguous()) else v_in.to(kd).contiguous()
            init_st = None if S_in is None else (S_in.transpose(-1, -2).contiguous() if probe["state_is_kv"] else S_in.contiguous())
            try:
                if self.recurrence_mode == "gdn_scalar":
                    g_k = log_a if (log_a.dtype == torch.float32 and log_a.is_contiguous()) else log_a.float().contiguous()
                    b_k = gate_er if (gate_er.dtype == kd and gate_er.is_contiguous()) else gate_er.to(kd).contiguous()
                    kw: Dict[str, Any] = {
                        "q": q_k, "k": k_k, "v": v_k, "g": g_k, "beta": b_k,
                        "scale": 1.0, "initial_state": init_st, "output_final_state": return_state,
                    }
                    if probe.get("has_head_first", False):
                        kw["head_first"] = False
                    o_fla, st_fla = probe["fn"](**kw)[:2]
                    self._fla_calls += 1
                    sf = None if (not return_state or st_fla is None) else (st_fla.transpose(-1, -2) if probe["state_is_kv"] else st_fla)
                    return o_fla, sf
                elif self.recurrence_mode == "gdn2_kda" and self.triadic_e == 1:
                    g_t = (log_a if (log_a.dtype == torch.float32 and log_a.is_contiguous()) else log_a.float().contiguous()) if probe.get("g_fp32", True) else log_a.to(kd).contiguous()
                    b_k = gate_er if (gate_er.dtype == kd and gate_er.is_contiguous()) else gate_er.to(kd).contiguous()
                    kw = {
                        "q": q_k, "k": k_k, "v": v_k, "g": g_t, "beta": b_k,
                        "scale": 1.0, "initial_state": init_st, "output_final_state": return_state,
                    }
                    if probe.get("has_head_first", False):
                        kw["head_first"] = False
                    if probe.get("has_safe_gate", False):
                        kw["safe_gate"] = True
                    o_fla, st_fla = probe["fn"](**kw)[:2]
                    self._fla_calls += 1
                    sf = None if (not return_state or st_fla is None) else (st_fla.transpose(-1, -2) if probe["state_is_kv"] else st_fla)
                    return o_fla, sf
                elif self.recurrence_mode == "gdn2_dplr" and self.triadic_e == 1:
                    alpha = torch.exp(log_a.float()).to(kd)
                    mode_name = probe.get("mode", "post_decay_ab")
                    a_vec = -(alpha * gate_er.to(kd) * k_k) if mode_name.startswith("post_decay") else -(gate_er.to(kd) * k_k)
                    b_vec = k_k
                    if mode_name.endswith("_ba"):
                        a_vec, b_vec = b_vec, a_vec
                    gk_t = (log_a if (log_a.dtype == torch.float32 and log_a.is_contiguous()) else log_a.float().contiguous()) if probe.get("gk_fp32", True) else log_a.to(kd).contiguous()
                    kw = {
                        "q": q_k, "k": k_k, "v": v_k, "a": a_vec.contiguous(), "b": b_vec.contiguous(), "gk": gk_t,
                        "scale": 1.0, "initial_state": init_st, "output_final_state": return_state,
                    }
                    if probe.get("has_head_first", False):
                        kw["head_first"] = False
                    o_fla, st_fla = probe["fn"](**kw)[:2]
                    self._fla_calls += 1
                    sf = None if (not return_state or st_fla is None) else (st_fla.transpose(-1, -2) if probe["state_is_kv"] else st_fla)
                    return o_fla, sf
            except Exception as exc:
                self._fla_fallback_reason = f"{type(exc).__name__}: {exc}"[:240]

        # Exact algebraic fallback (also handles Triadic rectangular K = d*E natively!)
        S0 = S_in if S_in is not None else torch.zeros(bsz, h, dv, dk, device=q.device, dtype=torch.float32)
        if self.recurrence_mode == "gdn_scalar":
            la_exp = log_a.unsqueeze(-1).expand(bsz, seq_len, h, dk)
            b_exp = gate_er.unsqueeze(-1).expand(bsz, seq_len, h, dk)
            v_eff = v_in * gate_er.unsqueeze(-1)
        elif self.recurrence_mode == "gdn2_kda":
            la_exp = log_a.repeat(1, 1, 1, self.triadic_e) if self.triadic_e > 1 else log_a
            b_exp = gate_er.unsqueeze(-1).expand(bsz, seq_len, h, dk)
            v_eff = v_in * gate_er.unsqueeze(-1)
        else:
            la_exp = log_a.repeat(1, 1, 1, self.triadic_e) if self.triadic_e > 1 else log_a
            b_exp = gate_er.repeat(1, 1, 1, self.triadic_e) if self.triadic_e > 1 else gate_er
            v_eff = v_in
        o_all, sf = _wy_chunk_gdn2_cpu(q, k, v_eff, la_exp, b_exp, S0, chunk_size=min(self.chunk_size, 32))
        return o_all, (sf if return_state else None)

    def forward(
        self,
        x_in: torch.Tensor,
        e_tok: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        bsz, seq_len, _ = x_in.shape
        if state is None and not return_state and self._c_pre is not None and seq_len > 1:
            q, k, v_in, log_a, gate_er, g_h = self._c_pre(x_in, e_tok)
            o_all, _ = self._run_recurrence(q, k, v_in, log_a, gate_er, None, return_state=False)
            y = self._c_post(x_in, o_all, g_h)  # type: ignore[misc]
            return y, None

        q, k, v_in, log_a, gate_er, g_h, new_ca, new_qb = self._pre_impl(x_in, e_tok, state=state, return_state=return_state)
        S_in = state["S"] if state is not None else None
        o_all, S_fin = self._run_recurrence(q, k, v_in, log_a, gate_er, S_in, return_state=return_state)
        o_m = (self.head_norm(o_all.to(x_in.dtype)) * g_h.to(x_in.dtype)).contiguous().view(bsz, seq_len, self.d_model)
        x = x_in + self.o_proj(o_m)
        y, mlp_st = self.mlp(x, state=state, return_state=return_state)
        if not return_state:
            return y, None
        assert S_fin is not None
        out_st: Dict[str, torch.Tensor] = {
            "S": S_fin.detach().clone(memory_format=torch.contiguous_format),
            "conv_qkv": new_qb,
            **mlp_st,
        }
        if new_ca is not None:
            out_st["conv_a"] = new_ca
        return y, out_st


class GatedSWABlock(nn.Module):
    """Gated Sliding-Window Attention (W=128) with QK-Norm, FlexAttention, Multiplicative Sigmoid Gate & Canon-A/C/D."""

    def __init__(
        self,
        d_model: int,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        window_size: int = 128,
        use_canon_acd: bool = True,
        use_canon_cd: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.window_size = int(window_size)
        self.is_global_mla = False
        self.is_swa = True
        canon_cd = use_canon_acd if use_canon_cd is None else use_canon_cd

        self.norm1 = _FastRMSNorm(d_model)
        self.canon_a = _Canon4(d_model) if use_canon_acd else None
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.q_norm = _FastRMSNorm(self.head_dim)
        self.k_norm = _FastRMSNorm(self.head_dim)
        self.ve_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.ve_lam = nn.Parameter(torch.tensor(0.1))
        # NeurIPS 2025 Best Paper multiplicative Sigmoid gate after SDPA (explicitly AdamW!)
        self.out_gate = nn.Linear(d_model, n_heads, bias=True)
        _mark_no_muon(self.out_gate)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = _CanonSwiGLUMLP(d_model, d_ff, use_canon_cd=canon_cd)
        self._c_fast: Optional[Any] = None
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj, self.ve_proj):
            nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.out_gate.weight)
        nn.init.constant_(self.out_gate.bias, 1.5)

    def compile_regional(self, device: torch.device) -> None:
        if device.type == "cuda" and self._c_fast is None:
            try:
                self._c_fast = torch.compile(self._fast_impl, dynamic=False)
            except Exception:
                self._c_fast = None

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bm: Any, past_len: int) -> torch.Tensor:
        seq_len = q.shape[2]
        q, k = q.to(v.dtype), k.to(v.dtype)
        use_gqa = self.n_heads != self.n_kv_heads
        if bm is not None and past_len == 0:
            if _RAW_FLEX_ATTN is not None and torch.compiler.is_compiling():
                return _RAW_FLEX_ATTN(q, k, v, block_mask=bm, enable_gqa=use_gqa)
            flex_fn, _ = _get_flex_attention()
            if flex_fn is not None:
                return flex_fn(q, k, v, block_mask=bm, enable_gqa=use_gqa)
        if use_gqa and _SDPA_HAS_ENABLE_GQA:
            k_e, v_e = k, v
            sdpa_kw: Dict[str, Any] = {"enable_gqa": True}
        elif use_gqa:
            rep = self.n_heads // self.n_kv_heads
            k_e = k.repeat_interleave(rep, dim=1)
            v_e = v.repeat_interleave(rep, dim=1)
            sdpa_kw = {}
        else:
            k_e, v_e = k, v
            sdpa_kw = {}
        w = self.window_size
        if seq_len == 1:
            return F.scaled_dot_product_attention(q, k_e[:, :, -w:, :], v_e[:, :, -w:, :], is_causal=False, **sdpa_kw)
        if k_e.shape[2] <= w and past_len == 0:
            return F.scaled_dot_product_attention(q, k_e, v_e, is_causal=True, **sdpa_kw)
        outs = []
        for qs in range(0, seq_len, w):
            qe = min(qs + w, seq_len)
            gqs, gqe = past_len + qs, past_len + qe
            ks = max(0, gqs - (w - 1))
            q_pos = torch.arange(gqs, gqe, device=q.device).unsqueeze(1)
            k_pos = torch.arange(ks, gqe, device=q.device).unsqueeze(0)
            mask = (q_pos >= k_pos) & ((q_pos - k_pos) < w)
            outs.append(F.scaled_dot_product_attention(q[:, :, qs:qe, :], k_e[:, :, ks:gqe, :], v_e[:, :, ks:gqe, :], attn_mask=mask, **sdpa_kw))
        return torch.cat(outs, dim=2)

    def _fast_impl(self, x_in: torch.Tensor, e_tok: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, bm: Any) -> torch.Tensor:
        bsz, seq_len, _ = x_in.shape
        n1 = self.norm1(x_in)
        if self.canon_a is not None:
            n1, _ = self.canon_a(n1, state=None, return_state=False)
        q = _apply_rope(self.q_norm(self.q_proj(n1).view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2), cos, sin)
        k = _apply_rope(self.k_norm(self.k_proj(n1).view(bsz, seq_len, self.n_kv_heads, self.head_dim)).transpose(1, 2), cos, sin)
        v = (self.v_proj(n1) + self.ve_lam.to(x_in.dtype) * self.ve_proj(e_tok)).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        out = self._attend(q, k, v, bm, past_len=0)
        g_h = torch.sigmoid(self.out_gate(n1)).unsqueeze(-1)
        x = x_in + self.o_proj((out.transpose(1, 2) * g_h).reshape(bsz, seq_len, self.d_model))
        y, _ = self.mlp(x, state=None, return_state=False)
        return y

    def forward(
        self,
        x_in: torch.Tensor,
        e_tok: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bm: Any = None,
        state: Optional[Dict[str, Any]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        bsz, seq_len, _ = x_in.shape
        if state is None and not return_state and self._c_fast is not None and seq_len > 1:
            return self._c_fast(x_in, e_tok, cos, sin, bm), None

        n1 = self.norm1(x_in)
        new_ca = None
        if self.canon_a is not None:
            n1, new_ca = self.canon_a(n1, state=state["conv_a"] if state is not None else None, return_state=return_state)
        qkv = F.linear(n1, torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0))
        q, k, v = qkv.split([self.d_model, self.kv_dim, self.kv_dim], dim=-1)
        q = _apply_rope(self.q_norm(q.view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2), cos, sin)
        k = _apply_rope(self.k_norm(k.view(bsz, seq_len, self.n_kv_heads, self.head_dim)).transpose(1, 2), cos, sin)
        v = (v + self.ve_lam.to(x_in.dtype) * self.ve_proj(e_tok)).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if state is not None and "k" in state:
            past_len = int(state["k"].shape[2])
            k_full = torch.cat([state["k"].to(k.dtype), k], dim=2)
            v_full = torch.cat([state["v"].to(v.dtype), v], dim=2)
        else:
            k_full, v_full = k, v
        out = self._attend(q, k_full, v_full, bm if state is None else None, past_len=past_len)
        g_h = torch.sigmoid(self.out_gate(n1)).unsqueeze(-1)
        x = x_in + self.o_proj((out.transpose(1, 2) * g_h).reshape(bsz, seq_len, self.d_model))
        y, mlp_st = self.mlp(x, state=state, return_state=return_state)
        if not return_state:
            return y, None
        out_st: Dict[str, Any] = {
            "k": k_full[:, :, -self.window_size :, :].detach().clone(memory_format=torch.contiguous_format),
            "v": v_full[:, :, -self.window_size :, :].detach().clone(memory_format=torch.contiguous_format),
            "pos": (int(state["pos"]) if state is not None else 0) + seq_len,
            **mlp_st,
        }
        if new_ca is not None:
            out_st["conv_a"] = new_ca
        return y, out_st


class GatedNoPEMLABlock(nn.Module):
    """PULSE-v7B Global Gather-and-Aggregate Layer: NoPE + DeepSeek Low-Rank KV Latent + Multiplicative Sigmoid Gate.

    Caches ONLY compressed latent `c_kv` [B, T, d_latent] (e.g. 192 dim for 105M -> 3.00 MiB across 2 layers at 4K),
    zero RoPE (zero length-extrapolation collapse), QK-Norm, and Canon-A/C/D.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_latent: int = 192,
        use_canon_acd: bool = True,
        use_canon_cd: Optional[bool] = None,
        use_ssmax: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_latent = int(d_latent)
        self.is_global_mla = True
        self.is_swa = False
        self.use_ssmax = bool(use_ssmax)
        canon_cd = use_canon_acd if use_canon_cd is None else use_canon_cd

        self.norm1 = _FastRMSNorm(d_model)
        self.canon_a = _Canon4(d_model) if use_canon_acd else None
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down = nn.Linear(d_model, self.d_latent, bias=False)
        self.kv_norm = _FastRMSNorm(self.d_latent)
        self.k_up = nn.Linear(self.d_latent, d_model, bias=False)
        self.v_up = nn.Linear(self.d_latent, d_model, bias=False)
        self.q_norm = _FastRMSNorm(self.head_dim)
        self.k_norm = _FastRMSNorm(self.head_dim)
        if self.use_ssmax:
            # Scalable-Softmax (arXiv:2501.19399): s_h * ln(pos + 2) prevents 1/N NoPE dilution at 8K+
            self.ssmax_scale = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.25))
            self.ssmax_scale._no_muon = True  # type: ignore[attr-defined]
        else:
            self.ssmax_scale = None
        self.out_gate = nn.Linear(d_model, n_heads, bias=True)
        _mark_no_muon(self.out_gate)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = _CanonSwiGLUMLP(d_model, d_ff, use_canon_cd=canon_cd)
        self._c_fast: Optional[Any] = None
        self._init_weights()

    def _apply_ssmax(self, q: torch.Tensor, past_len: int) -> torch.Tensor:
        if self.ssmax_scale is None:
            return q
        seq_len = q.shape[2]
        pos = torch.arange(past_len, past_len + seq_len, device=q.device, dtype=torch.float32)
        s = self.ssmax_scale.float().abs().clamp(0.05, 2.0)
        return q * (s * torch.log(pos + 2.0).view(1, 1, seq_len, 1)).to(q.dtype)

    def _init_weights(self) -> None:
        for m in (self.q_proj, self.kv_down, self.k_up, self.v_up):
            nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.out_gate.weight)
        nn.init.constant_(self.out_gate.bias, 1.5)

    def compile_regional(self, device: torch.device) -> None:
        if device.type == "cuda" and self._c_fast is None:
            try:
                self._c_fast = torch.compile(self._fast_impl, dynamic=False)
            except Exception:
                self._c_fast = None

    def _fast_impl(self, x_in: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x_in.shape
        n1 = self.norm1(x_in)
        if self.canon_a is not None:
            n1, _ = self.canon_a(n1, state=None, return_state=False)
        q = self._apply_ssmax(self.q_norm(self.q_proj(n1).view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2), 0)
        c_kv = self.kv_norm(self.kv_down(n1))
        k = self.k_norm(self.k_up(c_kv).view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_up(c_kv).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, is_causal=True)
        g_h = torch.sigmoid(self.out_gate(n1)).unsqueeze(-1)
        x = x_in + self.o_proj((out.transpose(1, 2) * g_h).reshape(bsz, seq_len, self.d_model))
        y, _ = self.mlp(x, state=None, return_state=False)
        return y

    def forward(
        self,
        x_in: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        bsz, seq_len, _ = x_in.shape
        if state is None and not return_state and self._c_fast is not None and seq_len > 1:
            return self._c_fast(x_in), None

        n1 = self.norm1(x_in)
        new_ca = None
        if self.canon_a is not None:
            n1, new_ca = self.canon_a(n1, state=state["conv_a"] if state is not None else None, return_state=return_state)
        c_cur = self.kv_norm(self.kv_down(n1))
        if state is not None and "c_kv" in state:
            past_len = state["c_kv"].shape[1]
            c_all = torch.cat([state["c_kv"].to(c_cur.dtype), c_cur], dim=1)
        else:
            past_len = 0
            c_all = c_cur
        q = self._apply_ssmax(self.q_norm(self.q_proj(n1).view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2), past_len)
        total_len = c_all.shape[1]
        k = self.k_norm(self.k_up(c_all).view(bsz, total_len, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_up(c_all).view(bsz, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        if past_len == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))
        elif seq_len == 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            q_pos = torch.arange(past_len, past_len + seq_len, device=q.device).unsqueeze(1)
            k_pos = torch.arange(total_len, device=q.device).unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=(q_pos >= k_pos))
        g_h = torch.sigmoid(self.out_gate(n1)).unsqueeze(-1)
        x = x_in + self.o_proj((out.transpose(1, 2) * g_h).reshape(bsz, seq_len, self.d_model))
        y, mlp_st = self.mlp(x, state=state, return_state=return_state)
        if not return_state:
            return y, None
        out_st: Dict[str, torch.Tensor] = {
            "c_kv": c_all.detach().clone(memory_format=torch.contiguous_format),
            **mlp_st,
        }
        if new_ca is not None:
            out_st["conv_a"] = new_ca
        return y, out_st


# ==============================================================================
# Stage 1 Pure-O(1) Rank-Wall Explorer Primitives: Toaster & RAM-Net
# ==============================================================================

class ToasterMemoryBlock(nn.Module):
    """Toaster (Google Research, Nov 2026): Two-Layer Cascaded Nonlinear MLP Memory with Chunk Target Propagation.

    Maintains strictly O(1) cascaded states S1 [B, H, dh, dk] and S2 [B, H, dv, dh] separated by
    nonlinearity sigma(z) = L2Norm(SiLU(z)), both updated with exact chunk-level WY Delta rules.
    """

    def __init__(
        self,
        d_model: int,
        embed_dim: int,
        n_heads: int,
        d_ff: int,
        d_hidden: int = 128,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_hidden = int(d_hidden)
        self.chunk_size = int(chunk_size)

        self.norm1 = _FastRMSNorm(d_model)
        self.canon_a = _Canon4(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.qkv_conv = _Canon4(3 * d_model)
        self.ve_proj = nn.Linear(embed_dim, d_model, bias=False)
        self.ve_lam = nn.Parameter(torch.tensor(0.1))

        # Initial orthogonal basis S1_0 so chunk 0 produces well-spread nonlinear hidden codes u = sigma(S1 k)!
        self.s1_init = nn.Parameter(torch.empty(n_heads, self.d_hidden, self.head_dim))
        self.tp_eta = nn.Parameter(torch.full((n_heads,), 0.5))
        self.gates_proj = nn.Linear(d_model, 5 * n_heads, bias=True)  # a1, a2, b1, b2, g_out
        _mark_no_muon(self.gates_proj)

        self.head_norm = _FastRMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = _CanonSwiGLUMLP(d_model, d_ff, use_canon_cd=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj, self.ve_proj, self.gates_proj):
            nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(self.o_proj.weight)
        h = self.n_heads
        with torch.no_grad():
            for hi in range(h):
                q_mat = torch.empty(self.d_hidden, self.d_hidden)
                nn.init.orthogonal_(q_mat)
                self.s1_init[hi].copy_(q_mat[:, : self.head_dim] * math.sqrt(self.d_hidden / self.head_dim))
            self.gates_proj.bias[:h].fill_(-3.5)
            self.gates_proj.bias[h : 2 * h].fill_(-3.5)
            self.gates_proj.bias[2 * h : 3 * h].fill_(-1.5)
            self.gates_proj.bias[3 * h : 4 * h].fill_(-1.0)
            self.gates_proj.weight[4 * h :].zero_()
            self.gates_proj.bias[4 * h :].fill_(1.5)

    def forward(
        self, x_in: torch.Tensor, e_tok: torch.Tensor, return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        bsz, seq_len, _ = x_in.shape
        n1, _ = self.canon_a(self.norm1(x_in), state=None, return_state=False)
        qkv, _ = self.qkv_conv(F.linear(n1, torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)))
        q, k, v = F.silu(qkv).chunk(3, dim=-1)
        q = F.normalize(q.view(bsz, seq_len, self.n_heads, self.head_dim), p=2, dim=-1, eps=1e-6)
        k = F.normalize(k.view(bsz, seq_len, self.n_heads, self.head_dim), p=2, dim=-1, eps=1e-6)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim) + self.ve_lam * self.ve_proj(e_tok).view(
            bsz, seq_len, self.n_heads, self.head_dim
        )

        a1_r, a2_r, b1_r, b2_r, g_r = self.gates_proj(n1).chunk(5, dim=-1)
        la1 = -F.softplus(a1_r.float()).clamp(1e-6, 10.0)
        la2 = -F.softplus(a2_r.float()).clamp(1e-6, 10.0)
        b1 = (2.0 * torch.sigmoid(b1_r)).float()
        b2 = (2.0 * torch.sigmoid(b2_r)).float()
        g_h = torch.sigmoid(g_r).unsqueeze(-1)

        # Chunk-level cascaded target propagation over strictly O(1) states (S1, S2)!
        c_sz = self.chunk_size
        S1 = self.s1_init.unsqueeze(0).expand(bsz, -1, -1, -1).float().clone()  # [B, H, dh, dk]
        S2 = torch.zeros(bsz, self.n_heads, self.head_dim, self.d_hidden, device=x_in.device, dtype=torch.float32)
        eta = self.tp_eta.abs().clamp(0.05, 2.0).view(1, 1, self.n_heads, 1)
        outs = []
        with torch.amp.autocast(device_type=x_in.device.type, enabled=False):
            q_f, k_f, v_f = q.float(), k.float(), v.float()
            for s in range(0, seq_len, c_sz):
                e = min(s + c_sz, seq_len)
                qc, kc, vc = q_f[:, s:e], k_f[:, s:e], v_f[:, s:e]
                # Layer 1 nonlinear hidden codes using chunk-start S1
                hq = F.normalize(F.silu(torch.einsum("bhmd,bthd->bthm", S1, qc)), p=2, dim=-1, eps=1e-6)
                hk = F.normalize(F.silu(torch.einsum("bhmd,bthd->bthm", S1, kc)), p=2, dim=-1, eps=1e-6)
                # Layer 2 exact chunk-WY DeltaNet over d_hidden=128 nonlinear keys!
                la2_c = la2[:, s:e].unsqueeze(-1).expand(-1, -1, -1, self.d_hidden)
                b2_c = b2[:, s:e].unsqueeze(-1).expand(-1, -1, -1, self.d_hidden)
                oc, S2_new = _wy_chunk_gdn2_cpu(hq, hk, vc * b2[:, s:e].unsqueeze(-1), la2_c, b2_c, S2, chunk_size=c_sz)
                outs.append(oc)
                # Target propagation: nudge hidden target u_hat toward reducing Layer-2 residual, update S1!
                pred_v = torch.einsum("bhvm,bthm->bthv", S2, hk)
                grad_h = torch.einsum("bhvm,bthv->bthm", S2, vc - pred_v)
                u_tgt = F.normalize(hk + eta * grad_h, p=2, dim=-1, eps=1e-6)
                la1_c = la1[:, s:e].unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                b1_c = b1[:, s:e].unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                _, S1 = _wy_chunk_gdn2_cpu(kc, kc, u_tgt * b1[:, s:e].unsqueeze(-1), la1_c, b1_c, S1, chunk_size=c_sz)
                S2 = S2_new
        o_all = torch.cat(outs, dim=1)
        o_m = (self.head_norm(o_all.to(x_in.dtype)) * g_h.to(x_in.dtype)).contiguous().view(bsz, seq_len, self.d_model)
        x = x_in + self.o_proj(o_m)
        y, _ = self.mlp(x, state=None, return_state=False)
        st = (
            {
                "S1": S1.detach().clone(memory_format=torch.contiguous_format),
                "S2": S2.detach().clone(memory_format=torch.contiguous_format),
            }
            if return_state
            else None
        )
        return y, st


class RAMNetMemoryBlock(nn.Module):
    """RAM-Net (arXiv:2602): Fixed Slot Array M [B, H, S, dv] with Key/Query-Aligned Sparse Top-K Addressing."""

    def __init__(
        self,
        d_model: int,
        embed_dim: int,
        n_heads: int,
        d_ff: int,
        num_slots: int = 128,
        top_k: int = 8,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_slots = int(num_slots)
        self.top_k = int(top_k)
        self.chunk_size = int(chunk_size)

        self.norm1 = _FastRMSNorm(d_model)
        self.canon_a = _Canon4(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.qkv_conv = _Canon4(3 * d_model)
        self.ve_proj = nn.Linear(embed_dim, d_model, bias=False)
        self.ve_lam = nn.Parameter(torch.tensor(0.1))

        # Shared key/query-aligned slot address prototypes [H, S, d_head]
        self.addr_w = nn.Parameter(torch.empty(n_heads, self.num_slots, self.head_dim))
        self.addr_temp = nn.Parameter(torch.full((n_heads, 1), 10.0))
        self.gates_proj = nn.Linear(d_model, 3 * n_heads, bias=True)  # a, beta, g_out
        _mark_no_muon(self.gates_proj)

        self.head_norm = _FastRMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = _CanonSwiGLUMLP(d_model, d_ff, use_canon_cd=True)
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj, self.ve_proj, self.gates_proj):
            nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(self.o_proj.weight)
        h = self.n_heads
        with torch.no_grad():
            # Well-spread spherical address table (repeated orthogonal blocks + normalized random)
            nn.init.normal_(self.addr_w, 0.0, 1.0)
            self.addr_w.copy_(F.normalize(self.addr_w, p=2, dim=-1))
            self.gates_proj.bias[:h].fill_(-4.0)
            self.gates_proj.bias[h : 2 * h].fill_(-1.0)
            self.gates_proj.weight[2 * h :].zero_()
            self.gates_proj.bias[2 * h :].fill_(1.5)

    def _sparse_address(self, x_dir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_dir: [B, T, H, d] -> unit-L2 sparse Top-K address vector a in [B, T, H, S]
        w_a = F.normalize(self.addr_w, p=2, dim=-1, eps=1e-6)
        temp = self.addr_temp.abs().clamp(2.0, 25.0).view(1, 1, self.n_heads, 1)
        logits = torch.einsum("bthd,hsd->bths", x_dir, w_a) * temp
        probs = F.softmax(logits, dim=-1)
        top_idx = torch.topk(logits, self.top_k, dim=-1).indices
        mask = torch.zeros_like(probs).scatter_(-1, top_idx, 1.0)
        sparse = probs * mask
        # L2-normalize sparse address so delta rule M <- M(I - beta a a^T) + beta v a^T is exact projection!
        addr = F.normalize(sparse, p=2, dim=-1, eps=1e-6)
        return addr, probs

    def forward(
        self, x_in: torch.Tensor, e_tok: torch.Tensor, return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        bsz, seq_len, _ = x_in.shape
        n1, _ = self.canon_a(self.norm1(x_in), state=None, return_state=False)
        qkv, _ = self.qkv_conv(F.linear(n1, torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)))
        q, k, v = F.silu(qkv).chunk(3, dim=-1)
        q_dir = F.normalize(q.view(bsz, seq_len, self.n_heads, self.head_dim), p=2, dim=-1, eps=1e-6)
        k_dir = F.normalize(k.view(bsz, seq_len, self.n_heads, self.head_dim), p=2, dim=-1, eps=1e-6)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim) + self.ve_lam * self.ve_proj(e_tok).view(
            bsz, seq_len, self.n_heads, self.head_dim
        )

        a_r, e_r, g_r = self.gates_proj(n1).chunk(3, dim=-1)
        log_a = -F.softplus(a_r.float()).clamp(1e-6, 10.0)
        beta = (2.0 * torch.sigmoid(e_r)).to(x_in.dtype)
        g_h = torch.sigmoid(g_r).unsqueeze(-1)

        a_q, _ = self._sparse_address(q_dir)
        a_k, probs_k = self._sparse_address(k_dir)

        # Write-weighted slot load balancing
        wb = beta.detach().unsqueeze(-1)
        wb_sum = wb.sum((0, 1)).clamp_min(1e-4)
        f_s = ((a_k > 0).float() * wb).sum((0, 1)) / wb_sum
        P_s = (probs_k * wb).sum((0, 1)) / wb_sum
        self.last_aux_loss = (self.num_slots * (f_s * P_s).sum(-1)).mean()

        # Exact WY Delta rule on slot bank M [B, H, dv, S]!
        M0 = torch.zeros(bsz, self.n_heads, self.head_dim, self.num_slots, device=x_in.device, dtype=torch.float32)
        la_exp = log_a.unsqueeze(-1).expand(bsz, seq_len, self.n_heads, self.num_slots)
        b_exp = beta.unsqueeze(-1).expand(bsz, seq_len, self.n_heads, self.num_slots)
        o_all, M_fin = _wy_chunk_gdn2_cpu(a_q, a_k, v * beta.unsqueeze(-1), la_exp, b_exp, M0, chunk_size=self.chunk_size)

        o_m = (self.head_norm(o_all.to(x_in.dtype)) * g_h.to(x_in.dtype)).contiguous().view(bsz, seq_len, self.d_model)
        x = x_in + self.o_proj(o_m)
        y, _ = self.mlp(x, state=None, return_state=False)
        st = {"M_ram": M_fin.detach().clone(memory_format=torch.contiguous_format)} if return_state else None
        return y, st


# ==============================================================================
# PULSE-v7 Model Config & Regional-Compiled Backbone
# ==============================================================================

@dataclass
class PULSEv7Config:
    vocab_size: int = 50257
    embed_dim: int = 192
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 6
    # Layer kinds: "gdn2" (GDN-2 / Triadic), "swa" (Gated SWA-128), "mla" (Gated NoPE-MLA)
    layer_types: Tuple[str, ...] = ("swa", "gdn2", "swa", "gdn2", "swa", "mla")
    recurrence_mode: Literal["gdn2_dplr", "gdn2_kda", "gdn_scalar"] = "gdn2_kda"
    triadic_e: int = 1
    gate_rank: int = 64
    d_ff: int = 1440
    delta_chunk_size: int = 64
    swa_window_size: int = 128
    mla_d_latent: int = 128
    rope_theta: float = 10000.0
    logit_softcap: float = 30.0
    use_canon_acd: bool = True
    use_canon_cd: Optional[bool] = None
    use_ssmax: bool = True
    # Stage 2 LT2 Latent Recurrent Depth: Prelude [0:s] -> Looped Core [s:e] (T loops, Full BPTT) -> Coda [e:L]
    loop_core_range: Optional[Tuple[int, int]] = None
    loop_steps: int = 1
    loop_exit_eps: float = 0.05

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.n_layers:
            raise ValueError(f"len(layer_types)={len(self.layer_types)} != n_layers={self.n_layers}")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("Use even head_dim and divisible d_model / n_heads.")
        if self.loop_core_range is not None:
            s, e = self.loop_core_range
            if not (0 <= s < e <= self.n_layers):
                raise ValueError(f"Invalid loop_core_range={self.loop_core_range} for n_layers={self.n_layers}")

    @classmethod
    def micro_test(cls) -> PULSEv7Config:
        return cls(
            vocab_size=260, embed_dim=32, d_model=64, n_heads=4, n_kv_heads=2, n_layers=6,
            layer_types=("swa", "gdn2", "gdn2", "swa", "gdn2", "mla"),
            recurrence_mode="gdn2_dplr", gate_rank=16, d_ff=128, delta_chunk_size=8,
            swa_window_size=8, mla_d_latent=32, use_canon_acd=True, use_canon_cd=True,
        )

    @classmethod
    def colab_30m_v7b(cls, recurrence_mode: Literal["gdn2_dplr", "gdn2_kda", "gdn_scalar"] = "gdn2_kda") -> PULSEv7Config:
        """~28.8M params (STRICTLY <= 29.12M TF): 6L 5:1 hierarchical hybrid ([SWA-128, 4x GDN-2, Gated NoPE-MLA])."""
        return cls(
            vocab_size=50257, embed_dim=192, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6,
            layer_types=("swa", "gdn2", "gdn2", "gdn2", "gdn2", "mla"),
            recurrence_mode=recurrence_mode, gate_rank=64, d_ff=1312, delta_chunk_size=64,
            swa_window_size=128, mla_d_latent=128,
        )

    @classmethod
    def colab_30m_v7b_fast(
        cls,
        recurrence_mode: Literal["gdn2_dplr", "gdn2_kda", "gdn_scalar"] = "gdn_scalar",
        n_gdn: int = 2,
    ) -> PULSEv7Config:
        """L4-tuned lean 30M 5:1 hybrid (no 2*d_ff Canon-D HBM bottleneck, no vocab softcap tanh, STRICTLY <= 29.12M TF)."""
        if n_gdn == 2:
            lt = ("swa", "gdn2", "swa", "swa", "gdn2", "mla")
        elif n_gdn == 3:
            lt = ("swa", "gdn2", "gdn2", "swa", "gdn2", "mla")
        else:
            lt = ("swa", "gdn2", "gdn2", "gdn2", "gdn2", "mla")
        return cls(
            vocab_size=50257, embed_dim=192, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6,
            layer_types=lt, recurrence_mode=recurrence_mode, gate_rank=64, d_ff=1408,
            delta_chunk_size=64, swa_window_size=128, mla_d_latent=128,
            logit_softcap=0.0, use_canon_acd=True, use_canon_cd=False,
        )

    @classmethod
    def colab_30m_v7b_lt2(cls, loop_steps: int = 2, looped: bool = True) -> PULSEv7Config:
        """Stage 2 30M 6L equivalent (STRICTLY <= 29.12M TF): 2 Prelude + 2 O(1) Looped Core (T=2 Full BPTT) + 2 Coda."""
        return cls(
            vocab_size=50257, embed_dim=192, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6,
            layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"),
            recurrence_mode="gdn_scalar", gate_rank=64, d_ff=1408, delta_chunk_size=64,
            swa_window_size=128, mla_d_latent=128, logit_softcap=0.0,
            use_canon_acd=True, use_canon_cd=False, use_ssmax=True,
            loop_core_range=((2, 4) if looped else None),
            loop_steps=(loop_steps if looped else 1),
            loop_exit_eps=0.05,
        )

    @classmethod
    def colab_30m_v7a_o1(cls, recurrence_mode: Literal["gdn2_dplr", "gdn2_kda", "gdn_scalar"] = "gdn2_kda") -> PULSEv7Config:
        """~28.9M strictly O(1) baseline (STRICTLY <= 29.12M TF, zero AttnRes, zero beta floor)."""
        return cls(
            vocab_size=50257, embed_dim=192, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6,
            layer_types=("swa", "gdn2", "gdn2", "swa", "gdn2", "gdn2"),
            recurrence_mode=recurrence_mode, gate_rank=64, d_ff=1312, delta_chunk_size=64,
            swa_window_size=128, logit_softcap=0.0, use_canon_acd=True, use_canon_cd=False,
        )

    @classmethod
    def colab_100m_v7b(cls, recurrence_mode: Literal["gdn2_dplr", "gdn2_kda", "gdn_scalar"] = "gdn2_kda") -> PULSEv7Config:
        """~104.5M params (STRICTLY <= 104.84M TF): 12L 5:1 flagship ([SWA-128, 4x GDN-2, Gated NoPE-MLA] x 2)."""
        return cls(
            vocab_size=50257, embed_dim=384, d_model=768, n_heads=12, n_kv_heads=4, n_layers=12,
            layer_types=("swa", "gdn2", "gdn2", "gdn2", "gdn2", "mla") * 2,
            recurrence_mode=recurrence_mode, gate_rank=96, d_ff=1920, delta_chunk_size=64,
            swa_window_size=128, mla_d_latent=192,
        )

    @classmethod
    def colab_100m_v7b_fast(
        cls,
        recurrence_mode: Literal["gdn2_dplr", "gdn2_kda", "gdn_scalar"] = "gdn_scalar",
        n_gdn: int = 4,
    ) -> PULSEv7Config:
        """L4-tuned lean 105M 5:1 hybrid (no 3840-ch Canon-D, no vocab softcap tanh, STRICTLY <= 104.84M TF)."""
        if n_gdn == 4:
            lt = ("swa", "gdn2", "swa", "swa", "gdn2", "mla") * 2
        elif n_gdn == 6:
            lt = ("swa", "gdn2", "gdn2", "swa", "gdn2", "mla") * 2
        else:
            lt = ("swa", "gdn2", "gdn2", "gdn2", "gdn2", "mla") * 2
        return cls(
            vocab_size=50257, embed_dim=384, d_model=768, n_heads=12, n_kv_heads=4, n_layers=12,
            layer_types=lt, recurrence_mode=recurrence_mode, gate_rank=96, d_ff=2048,
            delta_chunk_size=64, swa_window_size=128, mla_d_latent=192,
            logit_softcap=0.0, use_canon_acd=True, use_canon_cd=False,
        )

    @classmethod
    def colab_100m_v7a_o1(cls, recurrence_mode: Literal["gdn2_dplr", "gdn2_kda", "gdn_scalar"] = "gdn2_kda") -> PULSEv7Config:
        """~104.5M strictly O(1) 12L (STRICTLY <= 104.84M TF, zero AttnRes, zero beta floor)."""
        return cls(
            vocab_size=50257, embed_dim=384, d_model=768, n_heads=12, n_kv_heads=4, n_layers=12,
            layer_types=("swa", "gdn2", "gdn2") * 4,
            recurrence_mode=recurrence_mode, gate_rank=96, d_ff=1920, delta_chunk_size=64,
            swa_window_size=128,
        )


class PULSEv7Model(nn.Module):
    """PULSE-v7 Autoregressive Model with Regional Compilation & Strictly Zero-Leak Working State."""

    def __init__(self, config: PULSEv7Config) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.emb_in = nn.Linear(config.embed_dim, config.d_model, bias=False)
        self.emb_norm = _FastRMSNorm(config.d_model)

        self.blocks = nn.ModuleList()
        canon_cd = config.use_canon_acd if config.use_canon_cd is None else config.use_canon_cd
        for ltype in config.layer_types:
            if ltype == "gdn2":
                self.blocks.append(
                    GatedDeltaNet2Block(
                        d_model=config.d_model, embed_dim=config.embed_dim, n_heads=config.n_heads,
                        d_ff=config.d_ff, recurrence_mode=config.recurrence_mode,
                        triadic_e=config.triadic_e, gate_rank=config.gate_rank,
                        chunk_size=config.delta_chunk_size, use_canon_acd=config.use_canon_acd,
                        use_canon_cd=canon_cd,
                    )
                )
            elif ltype == "swa":
                self.blocks.append(
                    GatedSWABlock(
                        d_model=config.d_model, embed_dim=config.embed_dim, n_heads=config.n_heads,
                        n_kv_heads=config.n_kv_heads, d_ff=config.d_ff,
                        window_size=config.swa_window_size, use_canon_acd=config.use_canon_acd,
                        use_canon_cd=canon_cd,
                    )
                )
            elif ltype == "mla":
                self.blocks.append(
                    GatedNoPEMLABlock(
                        d_model=config.d_model, n_heads=config.n_heads, d_ff=config.d_ff,
                        d_latent=config.mla_d_latent, use_canon_acd=config.use_canon_acd,
                        use_canon_cd=canon_cd, use_ssmax=config.use_ssmax,
                    )
                )
            else:
                raise ValueError(f"Unknown layer type: {ltype}")

        # Stage 2 LT2 Latent Recurrent Depth parameters (single-scalar reinjection w_init=0.99 + STARS damped residual)
        if config.loop_core_range is not None:
            self.loop_reinj_logit = nn.Parameter(torch.tensor(math.log(99.0)))
            self.loop_damp_logit = nn.Parameter(torch.tensor(math.log(0.7 / 0.3)))
            self.loop_reinj_logit._no_muon = True  # type: ignore[attr-defined]
            self.loop_damp_logit._no_muon = True  # type: ignore[attr-defined]
            self.loop_delta_norm = _FastRMSNorm(config.d_model)
        else:
            self.loop_reinj_logit = None
            self.loop_damp_logit = None
            self.loop_delta_norm = None

        self.norm_f = _FastRMSNorm(config.d_model)
        self.emb_out = nn.Linear(config.d_model, config.embed_dim, bias=False)
        half = self.head_dim // 2
        self.register_buffer(
            "_inv_freq",
            1.0 / (config.rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half)),
            persistent=False,
        )
        self._rope_cache: Dict[Tuple[int, int, str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._bm_cache: Dict[Tuple[int, str], Any] = {}
        self._c_emb: Optional[Any] = None
        self._c_ce: Optional[Any] = None
        self._c_logits: Optional[Any] = None
        self._c_fused_regions: Optional[Tuple[List[int], List[Any]]] = None
        self._c_loop_stages: Optional[Dict[str, Tuple[List[int], List[Any]]]] = None
        nn.init.normal_(self.tok_emb.weight, 0.0, 1.0 / math.sqrt(config.embed_dim))
        nn.init.normal_(self.emb_in.weight, 0.0, 1.0 / math.sqrt(config.embed_dim))
        nn.init.zeros_(self.emb_out.weight)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def io_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def _build_fused_region_fns(self) -> Tuple[List[int], List[Any]]:
        """Partition arbitrary layer_types into pure-PyTorch compiled regions split ONLY at FLA GDN kernels."""
        gdn_indices = [i for i, lt in enumerate(self.config.layer_types) if lt == "gdn2"]

        def _uniq(fn: Any, name: str) -> Any:
            return types.FunctionType(fn.__code__.replace(co_name=name), fn.__globals__, name, fn.__defaults__, fn.__closure__)

        if not gdn_indices:
            def whole_fn(input_ids: torch.Tensor, cos: Any, sin: Any, bm: Any) -> torch.Tensor:
                e_tok, x = self._emb_impl(input_ids)
                for blk in self.blocks:
                    x = blk._fast_impl(x) if blk.is_global_mla else blk._fast_impl(x, e_tok, cos, sin, bm)  # type: ignore[operator]
                return x
            return [], [_uniq(whole_fn, "_v7_region_all")]

        regions: List[Any] = []
        # Region 0: emb -> non-GDN blocks before gdn_indices[0] -> gdn[0]._pre_fast
        g0 = gdn_indices[0]
        pre_blks_0 = [self.blocks[j] for j in range(g0)]
        b_g0: GatedDeltaNet2Block = self.blocks[g0]  # type: ignore[assignment]

        def reg_entry(input_ids: torch.Tensor, cos: Any, sin: Any, bm: Any):
            e_tok, x = self._emb_impl(input_ids)
            for blk in pre_blks_0:
                x = blk._fast_impl(x) if blk.is_global_mla else blk._fast_impl(x, e_tok, cos, sin, bm)  # type: ignore[operator]
            q, k, v_in, log_a, gate_er, g_h = b_g0._pre_fast(x, e_tok)
            return e_tok, x, q, k, v_in, log_a, gate_er, g_h

        regions.append(_uniq(reg_entry, "_v7_region_entry"))

        # Middle regions: gdn[k]._post_fast -> intervening SWA/MLA blocks -> gdn[k+1]._pre_fast
        for k_idx in range(len(gdn_indices) - 1):
            gi = gdn_indices[k_idx]
            gj = gdn_indices[k_idx + 1]
            b_prev: GatedDeltaNet2Block = self.blocks[gi]  # type: ignore[assignment]
            mid_blks = [self.blocks[j] for j in range(gi + 1, gj)]
            b_next: GatedDeltaNet2Block = self.blocks[gj]  # type: ignore[assignment]

            def reg_mid(x_in, o_all, g_h, e_tok, cos, sin, bm, b_prev=b_prev, mid_blks=mid_blks, b_next=b_next):
                x = b_prev._post_fast(x_in, o_all, g_h)
                for blk in mid_blks:
                    x = blk._fast_impl(x) if blk.is_global_mla else blk._fast_impl(x, e_tok, cos, sin, bm)  # type: ignore[operator]
                q, k, v_in, log_a, gate_er, g_n = b_next._pre_fast(x, e_tok)
                return x, q, k, v_in, log_a, gate_er, g_n

            regions.append(_uniq(reg_mid, f"_v7_region_mid_{k_idx}"))

        # Exit region: gdn[-1]._post_fast -> trailing SWA/MLA blocks
        glast = gdn_indices[-1]
        b_last: GatedDeltaNet2Block = self.blocks[glast]  # type: ignore[assignment]
        tail_blks = [self.blocks[j] for j in range(glast + 1, self.config.n_layers)]

        def reg_exit(x_in, o_all, g_h, e_tok, cos, sin, bm):
            x = b_last._post_fast(x_in, o_all, g_h)
            for blk in tail_blks:
                x = blk._fast_impl(x) if blk.is_global_mla else blk._fast_impl(x, e_tok, cos, sin, bm)  # type: ignore[operator]
            return x

        regions.append(_uniq(reg_exit, "_v7_region_exit"))
        return gdn_indices, regions

    def _build_slice_region_fns(self, start: int, end: int, prefix: str) -> Tuple[List[int], List[Any]]:
        """Partition blocks[start:end] into pure-PyTorch compiled regions split ONLY at FLA GDN kernels."""
        gdn_indices = [i for i in range(start, end) if self.config.layer_types[i] == "gdn2"]

        def _uniq(fn: Any, name: str) -> Any:
            return types.FunctionType(fn.__code__.replace(co_name=name), fn.__globals__, name, fn.__defaults__, fn.__closure__)

        if not gdn_indices:
            slice_blks = [self.blocks[j] for j in range(start, end)]

            def whole_s(x: torch.Tensor, e_tok: torch.Tensor, cos: Any, sin: Any, bm: Any, blks=slice_blks) -> torch.Tensor:
                for blk in blks:
                    x = blk._fast_impl(x) if blk.is_global_mla else blk._fast_impl(x, e_tok, cos, sin, bm)  # type: ignore[operator]
                return x

            return [], [_uniq(whole_s, f"_v7_{prefix}_all")]

        regions: List[Any] = []
        g0 = gdn_indices[0]
        pre_blks = [self.blocks[j] for j in range(start, g0)]
        b_g0: GatedDeltaNet2Block = self.blocks[g0]  # type: ignore[assignment]

        def reg_entry(x: torch.Tensor, e_tok: torch.Tensor, cos: Any, sin: Any, bm: Any, pre=pre_blks, bg=b_g0):
            for blk in pre:
                x = blk._fast_impl(x) if blk.is_global_mla else blk._fast_impl(x, e_tok, cos, sin, bm)  # type: ignore[operator]
            q, k, v_in, log_a, gate_er, g_h = bg._pre_fast(x, e_tok)
            return x, q, k, v_in, log_a, gate_er, g_h

        regions.append(_uniq(reg_entry, f"_v7_{prefix}_entry"))

        for k_idx in range(len(gdn_indices) - 1):
            gi, gj = gdn_indices[k_idx], gdn_indices[k_idx + 1]
            b_prev: GatedDeltaNet2Block = self.blocks[gi]  # type: ignore[assignment]
            mid_blks = [self.blocks[j] for j in range(gi + 1, gj)]
            b_next: GatedDeltaNet2Block = self.blocks[gj]  # type: ignore[assignment]

            def reg_mid(x_in, o_all, g_h, e_tok, cos, sin, bm, bp=b_prev, mb=mid_blks, bn=b_next):
                x = bp._post_fast(x_in, o_all, g_h)
                for blk in mb:
                    x = blk._fast_impl(x) if blk.is_global_mla else blk._fast_impl(x, e_tok, cos, sin, bm)  # type: ignore[operator]
                q, k, v_in, log_a, gate_er, g_n = bn._pre_fast(x, e_tok)
                return x, q, k, v_in, log_a, gate_er, g_n

            regions.append(_uniq(reg_mid, f"_v7_{prefix}_mid_{k_idx}"))

        glast = gdn_indices[-1]
        b_last: GatedDeltaNet2Block = self.blocks[glast]  # type: ignore[assignment]
        tail_blks = [self.blocks[j] for j in range(glast + 1, end)]

        def reg_exit(x_in, o_all, g_h, e_tok, cos, sin, bm, bl=b_last, tb=tail_blks):
            x = bl._post_fast(x_in, o_all, g_h)
            for blk in tb:
                x = blk._fast_impl(x) if blk.is_global_mla else blk._fast_impl(x, e_tok, cos, sin, bm)  # type: ignore[operator]
            return x

        regions.append(_uniq(reg_exit, f"_v7_{prefix}_exit"))
        return gdn_indices, regions

    def _run_slice_regions(
        self, x: torch.Tensor, e_tok: torch.Tensor, cos: Any, sin: Any, bm: Any, gdn_indices: List[int], regions: List[Any]
    ) -> torch.Tensor:
        if not gdn_indices:
            return regions[0](x, e_tok, cos, sin, bm)
        x_cur, q, k, v_in, log_a, gate_er, g_h = regions[0](x, e_tok, cos, sin, bm)
        b0: GatedDeltaNet2Block = self.blocks[gdn_indices[0]]  # type: ignore[assignment]
        o_cur, _ = b0._run_recurrence(q, k, v_in, log_a, gate_er, None, return_state=False)
        for k_idx, gi_next in enumerate(gdn_indices[1:]):
            x_cur, q, k, v_in, log_a, gate_er, g_h = regions[k_idx + 1](x_cur, o_cur, g_h, e_tok, cos, sin, bm)
            bn: GatedDeltaNet2Block = self.blocks[gi_next]  # type: ignore[assignment]
            o_cur, _ = bn._run_recurrence(q, k, v_in, log_a, gate_er, None, return_state=False)
        return regions[-1](x_cur, o_cur, g_h, e_tok, cos, sin, bm)

    def _run_fused_regions(
        self, input_ids: torch.Tensor, gdn_indices: List[int], regions: List[Any]
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        need_swa = any(blk.is_swa for blk in self.blocks)  # type: ignore[attr-defined]
        dtype = self.tok_emb.weight.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype(input_ids.device.type)
        cos = sin = bm = None
        if need_swa:
            cos, sin = self._rope(0, seq_len, input_ids.device, dtype)
            bm = self._get_swa_mask(seq_len, input_ids.device)
        if not gdn_indices:
            return regions[0](input_ids, cos, sin, bm)
        e_tok, x_cur, q, k, v_in, log_a, gate_er, g_h = regions[0](input_ids, cos, sin, bm)
        b0: GatedDeltaNet2Block = self.blocks[gdn_indices[0]]  # type: ignore[assignment]
        o_cur, _ = b0._run_recurrence(q, k, v_in, log_a, gate_er, None, return_state=False)
        for k_idx, gi_next in enumerate(gdn_indices[1:]):
            x_cur, q, k, v_in, log_a, gate_er, g_h = regions[k_idx + 1](x_cur, o_cur, g_h, e_tok, cos, sin, bm)
            bn: GatedDeltaNet2Block = self.blocks[gi_next]  # type: ignore[assignment]
            o_cur, _ = bn._run_recurrence(q, k, v_in, log_a, gate_er, None, return_state=False)
        return regions[-1](x_cur, o_cur, g_h, e_tok, cos, sin, bm)

    def compile_fast_paths(self) -> None:
        """Arbitrary-depth regional fusion compilation (6L/12L/24L): fuses all non-FLA ops between GDN calls."""
        if not torch.cuda.is_available():
            return
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 64)
        dev = self.tok_emb.weight.device
        for blk in self.blocks:
            if isinstance(blk, GatedDeltaNet2Block):
                if blk.recurrence_mode == "gdn_scalar":
                    blk._fla_probe = _probe_fla_gdn_backend(dev)
                elif blk.recurrence_mode == "gdn2_kda":
                    blk._fla_probe = _probe_fla_kda_backend(dev)
                else:
                    blk._fla_probe = _probe_fla_dplr_backend(dev)
        if self.config.loop_core_range is not None:
            if self._c_loop_stages is None:
                try:
                    s, e = self.config.loop_core_range
                    self._c_loop_stages = {}
                    for pname, a, b in (("pre", 0, s), ("core", s, e), ("coda", e, self.config.n_layers)):
                        gi, fns = self._build_slice_region_fns(a, b, pname)
                        self._c_loop_stages[pname] = (gi, [torch.compile(fn, dynamic=False) for fn in fns])
                    self._c_emb = torch.compile(self._emb_impl, dynamic=False)
                except Exception:
                    self._c_loop_stages = None
        elif self._c_fused_regions is None:
            try:
                gdn_idx, fns = self._build_fused_region_fns()
                self._c_fused_regions = (gdn_idx, [torch.compile(fn, dynamic=False) for fn in fns])
            except Exception:
                self._c_fused_regions = None
        if self._c_ce is None:
            try:
                self._c_ce = torch.compile(self._ce_loss_impl, dynamic=False)
                self._c_logits = torch.compile(self._logits_impl, dynamic=False)
            except Exception:
                self._c_ce = self._c_logits = None

    @torch.no_grad()
    def diagnose_graph_breaks(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Run torch._dynamo.explain on each fused region to prove 0 hidden graph breaks on CUDA/CPU."""
        gdn_idx, fns = self._build_fused_region_fns()
        seq_len = input_ids.shape[1]
        dtype = torch.bfloat16 if input_ids.is_cuda else self.tok_emb.weight.dtype
        need_swa = any(blk.is_swa for blk in self.blocks)  # type: ignore[attr-defined]
        cos = sin = bm = None
        if need_swa:
            cos, sin = self._rope(0, seq_len, input_ids.device, dtype)
            bm = self._get_swa_mask(seq_len, input_ids.device)
        report: Dict[str, Any] = {"n_regions": len(fns), "n_gdn_splits": len(gdn_idx), "regions": {}}
        with torch.autocast(input_ids.device.type, dtype=torch.bfloat16, enabled=input_ids.is_cuda):
            exp0 = torch._dynamo.explain(fns[0])(input_ids, cos, sin, bm)
            report["regions"][fns[0].__name__] = {"graph_count": exp0.graph_count, "break_count": exp0.graph_break_count, "reasons": [str(r)[:180] for r in exp0.break_reasons]}
            if gdn_idx:
                e_tok, x_cur, q, k, v_in, log_a, gate_er, g_h = fns[0](input_ids, cos, sin, bm)
                o_cur, _ = self.blocks[gdn_idx[0]]._run_recurrence(q, k, v_in, log_a, gate_er, None, return_state=False)  # type: ignore[operator]
                for k_i, gi_next in enumerate(gdn_idx[1:]):
                    fn_m = fns[k_i + 1]
                    expm = torch._dynamo.explain(fn_m)(x_cur, o_cur, g_h, e_tok, cos, sin, bm)
                    report["regions"][fn_m.__name__] = {"graph_count": expm.graph_count, "break_count": expm.graph_break_count, "reasons": [str(r)[:180] for r in expm.break_reasons]}
                    x_cur, q, k, v_in, log_a, gate_er, g_h = fn_m(x_cur, o_cur, g_h, e_tok, cos, sin, bm)
                    o_cur, _ = self.blocks[gi_next]._run_recurrence(q, k, v_in, log_a, gate_er, None, return_state=False)  # type: ignore[operator]
                exp_e = torch._dynamo.explain(fns[-1])(x_cur, o_cur, g_h, e_tok, cos, sin, bm)
                report["regions"][fns[-1].__name__] = {"graph_count": exp_e.graph_count, "break_count": exp_e.graph_break_count, "reasons": [str(r)[:180] for r in exp_e.break_reasons]}
        report["total_breaks"] = sum(v["break_count"] for v in report["regions"].values())
        return report

    def _emb_impl(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e_raw = self.tok_emb(input_ids)
        x0 = self.emb_norm(self.emb_in(e_raw))
        return e_raw.to(x0.dtype), x0

    def _logits_impl(self, h_fin: torch.Tensor, w_emb: torch.Tensor) -> torch.Tensor:
        logits = F.linear(self.emb_out(self.norm_f(h_fin)), w_emb)
        if self.config.logit_softcap > 0.0:
            sc = self.config.logit_softcap
            logits = sc * torch.tanh(logits / sc)
        return logits

    def _ce_loss_impl(self, h_fin: torch.Tensor, w_emb: torch.Tensor, y_tgt: torch.Tensor) -> torch.Tensor:
        logits = self._logits_impl(h_fin, w_emb)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_tgt.reshape(-1))

    def _rope(self, start_pos: int, length: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (start_pos, length, str(device), dtype)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        half = self.head_dim // 2
        pos = torch.arange(start_pos, start_pos + length, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self._inv_freq.to(device))
        res = (freqs.cos().to(dtype).view(1, 1, length, half), freqs.sin().to(dtype).view(1, 1, length, half))
        if start_pos == 0 and length <= max(512, self.config.swa_window_size) and len(self._rope_cache) < 8:
            self._rope_cache[key] = res
        return res

    def _get_swa_mask(self, seq_len: int, device: torch.device) -> Any:
        # PyTorch 2.14 FlexAttention decode autotuner can pick BLOCK_M=256 > SPARSE_Q_BLOCK_SIZE=128 for L<512
        if device.type != "cuda" or seq_len < 512 or seq_len <= self.config.swa_window_size:
            return None
        key = (seq_len, str(device))
        bm = self._bm_cache.get(key)
        if bm is None:
            bm = _build_swa_block_mask(seq_len, self.config.swa_window_size, device)
            if seq_len <= 512 and len(self._bm_cache) < 4:
                self._bm_cache[key] = bm
        return bm

    def _run_block_range(
        self,
        start: int,
        end: int,
        x: torch.Tensor,
        e_tok: torch.Tensor,
        cos: Any,
        sin: Any,
        bm: Any,
        block_states: Optional[List[Optional[Dict[str, Any]]]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[Dict[str, Any]]]]:
        new_sts: List[Optional[Dict[str, Any]]] = []
        for i in range(start, end):
            blk = self.blocks[i]
            st_i = block_states[i] if block_states is not None else None
            if blk.is_global_mla:  # type: ignore[attr-defined]
                x, nst = blk(x, state=st_i, return_state=return_state)
            elif blk.is_swa:  # type: ignore[attr-defined]
                x, nst = blk(x, e_tok, cos, sin, bm=bm, state=st_i, return_state=return_state)
            else:
                x, nst = blk(x, e_tok, state=st_i, return_state=return_state)
            if return_state:
                new_sts.append(nst)
        return x, new_sts

    def _run_backbone(
        self,
        input_ids: torch.Tensor,
        block_states: Optional[List[Optional[Dict[str, Any]]]] = None,
        return_state: bool = False,
        n_loops: Optional[int] = None,
        return_first_exit: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[Dict[str, Any]]], Optional[torch.Tensor]]:
        seq_len = input_ids.shape[1]
        if self.config.loop_core_range is None:
            if block_states is None and not return_state and self._c_fused_regions is not None and seq_len > 1:
                gdn_idx, c_regs = self._c_fused_regions
                return self._run_fused_regions(input_ids, gdn_idx, c_regs), [], None
            emb_fn = self._c_emb if (self._c_emb is not None and seq_len > 1) else self._emb_impl
            e_tok, x = emb_fn(input_ids)
            start_pos = 0
            if block_states is not None:
                for st in block_states:
                    if st is not None and "pos" in st:
                        start_pos = int(st["pos"])
                        break
            need_swa = any(blk.is_swa for blk in self.blocks)  # type: ignore[attr-defined]
            cos = sin = bm = None
            if need_swa:
                cos, sin = self._rope(start_pos, seq_len, input_ids.device, x.dtype)
                bm = self._get_swa_mask(seq_len, input_ids.device) if start_pos == 0 else None
            x, new_sts = self._run_block_range(0, self.config.n_layers, x, e_tok, cos, sin, bm, block_states, return_state)
            return x, new_sts, None

        # Stage 2 LT2 Latent Recurrent Depth: Prelude -> Looped Core (T loops, Full BPTT) -> Coda
        s, e = self.config.loop_core_range
        loops = int(n_loops if n_loops is not None else self.config.loop_steps)
        assert self.loop_reinj_logit is not None and self.loop_damp_logit is not None and self.loop_delta_norm is not None

        emb_fn = self._c_emb if (self._c_emb is not None and seq_len > 1 and block_states is None) else self._emb_impl
        e_tok, x = emb_fn(input_ids)
        need_swa = any(blk.is_swa for blk in self.blocks)  # type: ignore[attr-defined]
        cos = sin = bm = None
        if need_swa:
            cos, sin = self._rope(0, seq_len, input_ids.device, x.dtype)
            bm = self._get_swa_mask(seq_len, input_ids.device)

        use_compiled = (block_states is None and not return_state and self._c_loop_stages is not None and seq_len > 1)
        if use_compiled:
            assert self._c_loop_stages is not None
            pre_g, pre_r = self._c_loop_stages["pre"]
            core_g, core_r = self._c_loop_stages["core"]
            coda_g, coda_r = self._c_loop_stages["coda"]
            x_pre = self._run_slice_regions(x, e_tok, cos, sin, bm, pre_g, pre_r)
        else:
            x_pre, pre_sts = self._run_block_range(0, s, x, e_tok, cos, sin, bm, block_states, return_state)

        w = torch.sigmoid(self.loop_reinj_logit).to(x_pre.dtype)
        gamma = torch.sigmoid(self.loop_damp_logit).to(x_pre.dtype)
        z = x_pre
        z_exit1: Optional[torch.Tensor] = None
        core_sts: List[Optional[Dict[str, Any]]] = []
        for r_idx in range(loops):
            u = w * z + (1.0 - w) * x_pre
            if use_compiled:
                y_core = self._run_slice_regions(u, e_tok, cos, sin, bm, core_g, core_r)
            else:
                y_core, core_sts = self._run_block_range(s, e, u, e_tok, cos, sin, bm, None, return_state=(return_state and r_idx == loops - 1))
            # STARS damped normalized residual feedback prevents r>4 limit-cycle oscillation!
            z = u + gamma * self.loop_delta_norm(y_core - u)
            if r_idx == 0 and return_first_exit and loops > 1:
                z_exit1 = z

        if use_compiled:
            h_fin = self._run_slice_regions(z, e_tok, cos, sin, bm, coda_g, coda_r)
            h_exit1 = self._run_slice_regions(z_exit1, e_tok, cos, sin, bm, coda_g, coda_r) if z_exit1 is not None else None
            return h_fin, [], h_exit1
        h_fin, coda_sts = self._run_block_range(e, self.config.n_layers, z, e_tok, cos, sin, bm, block_states, return_state)
        h_exit1 = None
        if z_exit1 is not None:
            h_exit1, _ = self._run_block_range(e, self.config.n_layers, z_exit1, e_tok, cos, sin, bm, None, False)
        return h_fin, (pre_sts + core_sts + coda_sts if return_state else []), h_exit1

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        want_exit = self.training and self.config.loop_core_range is not None and self.config.loop_steps > 1 and self.config.loop_exit_eps > 0.0
        h, _, h_exit1 = self._run_backbone(tokens[:, :-1], return_state=False, return_first_exit=want_exit)
        ce_fn = self._c_ce if self._c_ce is not None else self._ce_loss_impl
        loss_fin = ce_fn(h, self.tok_emb.weight, tokens[:, 1:])
        if h_exit1 is not None:
            eps = float(self.config.loop_exit_eps)
            loss_e1 = ce_fn(h_exit1, self.tok_emb.weight, tokens[:, 1:])
            return (1.0 - eps) * loss_fin + eps * loss_e1
        return loss_fin

    def query_logits(self, tokens: torch.Tensor, positions: torch.Tensor, n_loops: Optional[int] = None) -> torch.Tensor:
        h, _, _ = self._run_backbone(tokens, return_state=False, n_loops=n_loops)
        rows = torch.arange(tokens.size(0), device=tokens.device)[:, None]
        return self._logits_impl(h[rows, positions], self.tok_emb.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        working_state: Optional[Dict[str, Any]] = None,
        *,
        return_state: bool = True,
        last_logits_only: bool = False,
        n_loops: Optional[int] = None,
    ) -> Dict[str, Any]:
        block_states = working_state["blocks"] if working_state is not None else None
        h, sts, _ = self._run_backbone(input_ids, block_states=block_states, return_state=return_state, n_loops=n_loops)
        if last_logits_only:
            h = h[:, -1:, :]
        lg_fn = self._c_logits if (self._c_logits is not None and h.size(1) > 1) else self._logits_impl
        return {"logits": lg_fn(h, self.tok_emb.weight), "working_state": ({"blocks": sts} if return_state else None)}

    def forward_with_hooks(
        self,
        input_ids: torch.Tensor,
        *,
        mid_layer: Optional[int] = None,
        mid_hook: Optional[Any] = None,
        n_loops: Optional[int] = None,
        collect_gdn_signals: bool = False,
    ) -> Dict[str, Any]:
        """Stage 3 hookable forward: mid-layer Semantic RLS/EngramEdit injection & native GDN Delta-Binding signals."""
        seq_len = input_ids.shape[1]
        e_tok, x = self._emb_impl(input_ids)
        need_swa = any(blk.is_swa for blk in self.blocks)  # type: ignore[attr-defined]
        cos = sin = bm = None
        if need_swa:
            cos, sin = self._rope(0, seq_len, input_ids.device, x.dtype)
            bm = self._get_swa_mask(seq_len, input_ids.device)

        gdn_signals: List[torch.Tensor] = []

        def _run_blk(blk: nn.Module, cur_x: torch.Tensor) -> torch.Tensor:
            if blk.is_global_mla:  # type: ignore[attr-defined]
                y, _ = blk(cur_x, state=None, return_state=False)
                return y
            if blk.is_swa:  # type: ignore[attr-defined]
                y, _ = blk(cur_x, e_tok, cos, sin, bm=bm, state=None, return_state=False)
                return y
            assert isinstance(blk, GatedDeltaNet2Block)
            if not collect_gdn_signals:
                y, _ = blk(cur_x, e_tok, state=None, return_state=False)
                return y
            q, k, v_in, log_a, gate_er, g_h, _, _ = blk._pre_impl(cur_x, e_tok, state=None, return_state=False)
            o_all, _ = blk._run_recurrence(q, k, v_in, log_a, gate_er, None, return_state=False)
            # Fast FLA probe at q=k: S_t k_t = (1 - beta_t) S_{t-1} a_t k_t + beta_t v_t -> exact binding residual!
            o_at_k, _ = blk._run_recurrence(k.clone(), k, v_in, log_a, gate_er, None, return_state=False)
            vf, okf = v_in.float(), o_at_k.float()
            res_after = vf - okf  # [B, T, H, d]
            bf = gate_er.float()  # [B, T, H] or [B, T, H, d]
            if bf.ndim == 4:
                bf = bf.mean(dim=-1)
            af = torch.exp(log_a.float())
            if af.ndim == 4:
                af = af.mean(dim=-1)
            denom = (1.0 - bf).abs().clamp_min(0.1).unsqueeze(-1)
            delta_est = res_after / denom
            sig = torch.cat(
                [
                    (delta_est.pow(2).sum(dim=-1) + 1e-6).sqrt(),     # [B, T, H] ||delta_t|| (grad-safe)
                    (res_after.pow(2).sum(dim=-1) + 1e-6).sqrt(),     # [B, T, H] ||v_t - S_t k_t|| (grad-safe)
                    F.cosine_similarity(vf, okf, dim=-1, eps=1e-6),   # [B, T, H] cos(v_t, S_t k_t)
                    bf,                                               # [B, T, H] erase/write beta_t
                    af,                                               # [B, T, H] decay alpha_t
                ],
                dim=-1,
            )
            gdn_signals.append(sig)
            o_m = (blk.head_norm(o_all.to(cur_x.dtype)) * g_h.to(cur_x.dtype)).contiguous().view(cur_x.size(0), seq_len, blk.d_model)
            y, _ = blk.mlp(cur_x + blk.o_proj(o_m), state=None, return_state=False)
            return y

        h_mid: Optional[torch.Tensor] = None
        if self.config.loop_core_range is None:
            m_idx = (self.config.n_layers // 2) if mid_layer is None else int(mid_layer)
            for i, blk in enumerate(self.blocks):
                x = _run_blk(blk, x)
                if i + 1 == m_idx:
                    h_mid = x
                    if mid_hook is not None:
                        x = mid_hook(x, input_ids)
            h_fin = x
        else:
            s, e = self.config.loop_core_range
            loops = int(n_loops if n_loops is not None else self.config.loop_steps)
            assert self.loop_reinj_logit is not None and self.loop_damp_logit is not None and self.loop_delta_norm is not None
            for i in range(0, s):
                x = _run_blk(self.blocks[i], x)
            x_pre = x
            w = torch.sigmoid(self.loop_reinj_logit).to(x_pre.dtype)
            gamma = torch.sigmoid(self.loop_damp_logit).to(x_pre.dtype)
            z = x_pre
            for r_idx in range(loops):
                u = w * z + (1.0 - w) * x_pre
                y_core = u
                for i in range(s, e):
                    # Collect GDN signals only on final loop to keep fixed signal dim across r
                    prev_flag = collect_gdn_signals
                    if r_idx < loops - 1:
                        collect_gdn_signals = False
                    y_core = _run_blk(self.blocks[i], y_core)
                    collect_gdn_signals = prev_flag
                z = u + gamma * self.loop_delta_norm(y_core - u)
            h_mid = z
            if mid_hook is not None:
                z = mid_hook(z, input_ids)
            for i in range(e, self.config.n_layers):
                z = _run_blk(self.blocks[i], z)
            h_fin = z

        return {
            "logits": self._logits_impl(h_fin, self.tok_emb.weight),
            "h_mid": h_mid,
            "h_fin": h_fin,
            "gdn_signals": torch.cat(gdn_signals, dim=-1) if gdn_signals else None,
        }

    @staticmethod
    def storage_audit_bytes(state: Dict[str, Any]) -> Tuple[int, int]:
        """Return (logical_bytes, deduplicated_backing_storage_bytes) to enforce zero view-leak CI."""
        logical = 0
        allocs: Dict[int, int] = {}

        def _walk(obj: Any) -> None:
            nonlocal logical
            if isinstance(obj, torch.Tensor):
                logical += obj.nelement() * obj.element_size()
                st = obj.untyped_storage()
                allocs[st.data_ptr()] = st.nbytes()
            elif isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    _walk(v)

        _walk(state)
        return logical, sum(allocs.values())
