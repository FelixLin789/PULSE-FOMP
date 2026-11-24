"""PULSE Whiteboard Anvil Core (Week 1): Dual Strong Baselines & Automated Anti-Strawman Auditors.

Provides reusable, zero-bloat infrastructure for Week 1 (Anvil) through Week 4 (30M/100M Validation):
  1. `TFModernLM`: Full-strength 2026 Transformer strong baseline — split Q/K/V/O matrices (for clean
     split-matrix FairMuonAdamW), QK-Norm, Gated SDPA (NeurIPS 2025), NoPE/RoPE + Scalable-Softmax
     (SSMax), Canon-ACD 4-tap causal residual convolutions (Allen-Zhu Part 4.1), and SwiGLU MLP.
  2. `build_dual_strong_baselines`: Constructs strictly parameter-matched `TF-Modern` and `PULSE-v7B-Ref`
     (`PULSE <= TF`, ~3.0M for Week-1/2 mechanism lab, with SWA window W=32 << L_train to block shortcuts).
  3. Automated Anti-Strawman Auditors:
       - `audit_zero_hidden_params`: Verifies zero unregistered/hidden puzzle lookup tables & strict Muon split.
       - `estimate_exact_step_flops`: Full per-step FLOP accounting (projections, O(L^2 d)/SWA/GDN/MLA, LM head, loops).
       - `audit_linear_probe_hop_ladder`: Fits frozen linear probes per layer/loop step for hops 1..H (P1 Depo1).
       - `audit_gradient_sign_flip`: Mandatory ICML 2026 sign-flip verification for fast-weight plasticity (P4).
       - `audit_boundary_histogram`: High-entropy vs low-entropy boundary density & ratio drift audit (P5 / DC-ER).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW
from pulse.pulse_v7 import (
    PULSEv7Config,
    PULSEv7Model,
    _Canon4,
    _CanonSwiGLUMLP,
    _FastRMSNorm,
    _mark_no_muon,
)


@dataclass
class TFModernConfig:
    """Full-strength modern Transformer baseline (NoPE/RoPE + SSMax + QK-Norm + Gated SDPA + Canon-ACD + SwiGLU)."""

    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 704
    use_rope: bool = False  # Default NoPE + SSMax + Canon-ACD for clean 4x-8x length extrapolation
    rope_theta: float = 10000.0
    use_ssmax: bool = True
    use_canon_acd: bool = True
    use_gated_sdpa: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0 or (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("d_model must be divisible by n_heads with even head_dim.")

    @classmethod
    def lab_3m(cls, vocab_size: int = 256) -> TFModernConfig:
        """~3.6M-parameter TF-Modern (power-of-2 d=256, H=8, d_head=32) matched to 6L PULSE-v7B-Ref."""
        return cls(vocab_size=vocab_size, d_model=256, n_heads=8, n_layers=6, d_ff=448)

    @classmethod
    def scale_125m(cls, vocab_size: int = 50257) -> TFModernConfig:
        """123.93M TF-Modern strong baseline (12L d=768 H=12 d_ff=2048 NoPE+SSMax+QK-Norm+GatedSDPA+Canon-ACD)."""
        return cls(vocab_size=vocab_size, d_model=768, n_heads=12, n_layers=12, d_ff=2048)

    @classmethod
    def scale_350m(cls, vocab_size: int = 50257) -> TFModernConfig:
        """360.93M TF-Modern strong baseline (24L d=1024 H=16 d_ff=2816 NoPE+SSMax+QK-Norm+GatedSDPA+Canon-ACD)."""
        return cls(vocab_size=vocab_size, d_model=1024, n_heads=16, n_layers=24, d_ff=2816)

    @classmethod
    def micro_test(cls, vocab_size: int = 128) -> TFModernConfig:
        return cls(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=4, d_ff=160)


def _apply_rope_modern(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class ModernAttentionBlock(nn.Module):
    """Split-matrix Q/K/V/O attention with QK-Norm, Gated SDPA, SSMax, Canon-A, and Canon-CD SwiGLU."""

    def __init__(self, cfg: TFModernConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.norm1 = _FastRMSNorm(cfg.d_model)
        self.canon_a = _Canon4(cfg.d_model) if cfg.use_canon_acd else None

        # Strictly split 2D matrices so FairMuonAdamW orthogonalizes Q, K, V, O independently!
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.q_norm = _FastRMSNorm(self.head_dim)
        self.k_norm = _FastRMSNorm(self.head_dim)

        self.out_gate = nn.Linear(cfg.d_model, cfg.n_heads, bias=True) if cfg.use_gated_sdpa else None
        if self.out_gate is not None:
            _mark_no_muon(self.out_gate)

        self.ssmax_scale = nn.Parameter(torch.full((1, cfg.n_heads, 1, 1), 0.5)) if cfg.use_ssmax else None
        if self.ssmax_scale is not None:
            self.ssmax_scale._no_muon = True  # type: ignore[attr-defined]

        self.mlp = _CanonSwiGLUMLP(cfg.d_model, cfg.d_ff, use_canon_cd=cfg.use_canon_acd)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(self.o_proj.weight)
        if self.out_gate is not None:
            nn.init.zeros_(self.out_gate.weight)
            nn.init.constant_(self.out_gate.bias, 1.5)

    def forward(
        self,
        x_in: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x_in.shape
        n1 = self.norm1(x_in)
        if self.canon_a is not None:
            n1, _ = self.canon_a(n1, state=None, return_state=False)

        q = self.q_norm(self.q_proj(n1).view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(n1).view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(n1).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.cfg.use_rope and cos is not None and sin is not None:
            q = _apply_rope_modern(q, cos, sin)
            k = _apply_rope_modern(k, cos, sin)

        if self.ssmax_scale is not None:
            # Scalable-Softmax: q_pos scaled by |s_h| * log(pos + 2) prevents softmax dilution at 4x-8x length!
            pos_log = torch.log(torch.arange(2, seq_len + 2, device=x_in.device, dtype=torch.float32)).view(1, 1, seq_len, 1)
            q = (q.float() * (1.0 + self.ssmax_scale.float().abs() * pos_log)).to(v.dtype)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))  # [B, H, T, d_head]
        if self.out_gate is not None:
            g = torch.sigmoid(self.out_gate(n1)).transpose(1, 2).unsqueeze(-1).to(o.dtype)
            o = o * g
        o = o.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        x = x_in + self.o_proj(o)
        y, _ = self.mlp(x, state=None, return_state=False)
        return y


class TFModernLM(nn.Module):
    """Full-strength modern Transformer strong baseline with regional compilation & layer-state hooks."""

    def __init__(self, config: TFModernConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.tok_emb.weight._no_muon = True  # type: ignore[attr-defined]
        self.emb_norm = _FastRMSNorm(config.d_model)
        self.blocks = nn.ModuleList([ModernAttentionBlock(config) for _ in range(config.n_layers)])
        self.norm_f = _FastRMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Tied embeddings, zero hidden params
        nn.init.normal_(self.tok_emb.weight, 0.0, 0.02)
        self._compiled_blocks: Optional[List[Any]] = None

    def io_parameters(self) -> List[nn.Parameter]:
        seen = set()
        out: List[nn.Parameter] = []
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.io_parameters())

    def compile_fast_paths(self) -> None:
        if not torch.cuda.is_available() or self._compiled_blocks is not None:
            return
        self._compiled_blocks = [torch.compile(blk, dynamic=True) for blk in self.blocks]

    def _rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.use_rope:
            return None, None
        half = self.head_dim // 2
        inv = 1.0 / (self.config.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv)
        return freqs.cos().to(dtype).view(1, 1, seq_len, half), freqs.sin().to(dtype).view(1, 1, seq_len, half)

    def forward_anvil(
        self,
        input_ids: torch.Tensor,
        return_layer_states: bool = False,
        mid_delta_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        x = self.emb_norm(self.tok_emb(input_ids))
        cos, sin = self._rope(input_ids.shape[1], input_ids.device, x.dtype)
        states: List[torch.Tensor] = []
        runners = self._compiled_blocks if (self.training and self._compiled_blocks is not None and not return_layer_states and mid_delta_fn is None) else self.blocks
        for idx, blk in enumerate(runners):
            x = blk(x, cos, sin)
            if mid_delta_fn is not None:
                x = mid_delta_fn(x, idx)
            if return_layer_states:
                states.append(x)
        h_fin = self.norm_f(x)
        logits = F.linear(h_fin, self.tok_emb.weight)
        return {"logits": logits, "h_fin": h_fin, "layer_states": states}

    def forward(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        return self.forward_anvil(input_ids, return_layer_states=False)

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Memory-efficient BF16 autoregressive CE loss matching PULSEv8Model.compute_loss."""
        logits = self.forward(tokens[:, :-1])["logits"]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))


class PULSEv7BAnvilAdapter(nn.Module):
    """Thin zero-extra-parameter adapter exposing `forward_anvil` on `PULSEv7Model` identically to `TFModernLM`."""

    def __init__(self, model: PULSEv7Model) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        # Ensure small-vocab embeddings & embedding bridges never leak into Muon (<8192 vocab lab!)
        for mod in (model.tok_emb, model.emb_in, model.emb_out):
            _mark_no_muon(mod)
        for name, p in model.named_parameters():
            if "ssmax" in name:
                p._no_muon = True  # type: ignore[attr-defined]

    def io_parameters(self) -> List[nn.Parameter]:
        return self.model.io_parameters()

    def num_parameters(self) -> int:
        return self.model.num_parameters()

    def compile_fast_paths(self) -> None:
        self.model.compile_fast_paths()

    def forward_anvil(
        self,
        input_ids: torch.Tensor,
        return_layer_states: bool = False,
        mid_delta_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        # Use static-shape compiled fast path only during training; eval varies L (1x/2x/4x extrap, rollouts)
        if self.training and not return_layer_states and mid_delta_fn is None:
            out = self.model(input_ids, return_state=False)
            return {"logits": out["logits"], "h_fin": None, "layer_states": []}
        seq_len = input_ids.shape[1]
        e_tok, x = self.model._emb_impl(input_ids)
        need_swa = any(blk.is_swa for blk in self.model.blocks)  # type: ignore[attr-defined]
        cos = sin = bm = None
        if need_swa:
            cos, sin = self.model._rope(0, seq_len, input_ids.device, x.dtype)
            bm = self.model._get_swa_mask(seq_len, input_ids.device)
        states: List[torch.Tensor] = []
        for idx, blk in enumerate(self.model.blocks):
            if blk.is_global_mla:  # type: ignore[attr-defined]
                x, _ = blk(x, state=None, return_state=False)
            elif blk.is_swa:  # type: ignore[attr-defined]
                x, _ = blk(x, e_tok, cos, sin, bm=bm, state=None, return_state=False)
            else:
                x, _ = blk(x, e_tok, state=None, return_state=False)
            if mid_delta_fn is not None:
                x = mid_delta_fn(x, idx)
            if return_layer_states:
                states.append(x)
        h_fin = self.model.norm_f(x)
        logits = self.model._logits_impl(x, self.model.tok_emb.weight)
        return {"logits": logits, "h_fin": h_fin, "layer_states": states}

    def forward(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        return self.forward_anvil(input_ids, return_layer_states=False)

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Memory-efficient BF16 autoregressive CE loss matching PULSEv8Model.compute_loss."""
        logits = self.forward(tokens[:, :-1])["logits"]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))


def build_dual_strong_baselines(
    vocab_size: int,
    smoke: bool = False,
    seed: int = 41,
) -> Dict[str, nn.Module]:
    """Construct matched `tf_modern` and `pulse_v7b_ref` (strictly PULSE <= TF params, SWA W=32 << L_train)."""
    torch.manual_seed(seed)
    if smoke:
        tf_cfg = TFModernConfig.micro_test(vocab_size=vocab_size)
        p_cfg = PULSEv7Config(
            vocab_size=vocab_size, embed_dim=64, d_model=64, n_heads=4, n_kv_heads=2, n_layers=4,
            layer_types=("swa", "gdn2", "gdn2", "mla"),
            recurrence_mode="gdn_scalar", gate_rank=16, d_ff=144, delta_chunk_size=16,
            swa_window_size=16, mla_d_latent=32, logit_softcap=0.0,
            use_canon_acd=True, use_canon_cd=False, use_ssmax=True,
        )
    else:
        tf_cfg = TFModernConfig.lab_3m(vocab_size=vocab_size)
        # 6L 5:1 hybrid (SWA-32 << L_train to block local window cheating, GDN-scalar, NoPE-MLA + SSMax)
        p_cfg = PULSEv7Config(
            vocab_size=vocab_size, embed_dim=192, d_model=256, n_heads=8, n_kv_heads=4, n_layers=6,
            layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"),
            recurrence_mode="gdn_scalar", gate_rank=32, d_ff=416, delta_chunk_size=64,
            swa_window_size=32, mla_d_latent=64, logit_softcap=0.0,
            use_canon_acd=True, use_canon_cd=False, use_ssmax=True,
        )
    tf_m = TFModernLM(tf_cfg)
    torch.manual_seed(seed)
    p_m = PULSEv7BAnvilAdapter(PULSEv7Model(p_cfg))
    assert p_m.num_parameters() <= tf_m.num_parameters(), (
        f"Param parity violated: PULSE-v7B-Ref ({p_m.num_parameters():,}) > TF-Modern ({tf_m.num_parameters():,})"
    )
    return {"tf_modern": tf_m, "pulse_v7b_ref": p_m}


# ==============================================================================
# Automated Anti-Strawman Auditors (Section 3.1 & 3.2)
# ==============================================================================

def audit_zero_hidden_params(model: nn.Module, vocab_size: int, d_model: int) -> Dict[str, Any]:
    """Verify zero hidden lookup tables (HRM/TRM 430M puzzle-table ban) and strict FairMuonAdamW gate isolation."""
    params = model.io_parameters() if hasattr(model, "io_parameters") else list(model.parameters())
    total_params = sum(p.numel() for p in params)
    # Count non-RoPE floating buffers to catch hidden lookup tables disguised as buffers
    float_buf_elems = sum(
        b.numel() for name, b in model.named_buffers()
        if b.is_floating_point() and b.numel() > 16 and not name.endswith("_inv_freq")
    )
    max_allowed_emb = vocab_size * d_model
    oversized_tables = [
        (name, tuple(p.shape), p.numel())
        for name, p in model.named_parameters()
        if ("emb" in name or "table" in name or "puzzle" in name) and p.numel() > max_allowed_emb
    ]

    opt = FairMuonAdamW(params, lr=1e-3)
    muon_ids = {id(p) for p in opt.param_groups[0]["params"]}
    leaked_into_muon: List[str] = []
    for name, p in model.named_parameters():
        if id(p) not in muon_ids:
            continue
        bad_tags = ("emb", "norm", "canon", "qkv_conv", "out_gate", "gates_proj", "ssmax", "bias", "router")
        if any(tag in name for tag in bad_tags):
            leaked_into_muon.append(name)

    muon_params = sum(p.numel() for p in opt.param_groups[0]["params"])
    adam_params = sum(p.numel() for p in opt.param_groups[1]["params"])
    passed = (len(oversized_tables) == 0) and (len(leaked_into_muon) == 0) and (float_buf_elems == 0)
    return {
        "total_params": int(total_params),
        "muon_params": int(muon_params),
        "adam_params": int(adam_params),
        "float_buffer_elems": int(float_buf_elems),
        "oversized_tables": oversized_tables,
        "leaked_into_muon": leaked_into_muon,
        "pass_zero_hidden_audit": bool(passed),
    }


def estimate_exact_step_flops(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    loop_factor: float = 1.0,
    backward_mult: float = 2.0,
) -> Dict[str, Any]:
    """Full per-step FLOP accounting (NOT naive 6ND): sums linear projections, Attention/SWA/GDN/MLA, and LM head."""
    cfg = model.config  # type: ignore[attr-defined]
    d = int(cfg.d_model)
    n_l = int(cfg.n_layers)
    v = int(cfg.vocab_size)
    tokens = batch_size * seq_len

    # 1. Exact forward MACs * 2 from all nn.Linear layers (excl. lm_head handled explicitly below)
    proj_flops_per_tok = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in name:
            proj_flops_per_tok += 2 * m.in_features * m.out_features
    proj_fwd = tokens * proj_flops_per_tok * loop_factor

    # 2. Exact sequence-mixing FLOPs (causal factor 0.5 for softmax attention: QK^T + AV = 4 * 0.5 * L^2 * d = 2 L^2 d)
    mix_fwd = 0.0
    if isinstance(model, TFModernLM):
        mix_fwd = n_l * (2.0 * batch_size * seq_len * seq_len * d) * loop_factor
    else:
        hd = d // int(cfg.n_heads)
        swa_w = min(seq_len, int(cfg.swa_window_size))
        for lt in cfg.layer_types:
            if lt == "mla":
                mix_fwd += 2.0 * batch_size * seq_len * seq_len * d
            elif lt == "swa":
                mix_fwd += 2.0 * batch_size * seq_len * swa_w * d
            else:  # GDN linear recurrence: ~4 * B * L * H * d_h^2 (read + delta + write + chunk WY)
                mix_fwd += 6.0 * batch_size * seq_len * d * hd
        mix_fwd *= loop_factor

    # 3. LM Head matmul: 2 * B * L * d_emb * V
    d_emb = int(getattr(cfg, "embed_dim", d))
    head_fwd = 2.0 * tokens * d_emb * v

    fwd_total = proj_fwd + mix_fwd + head_fwd
    step_total = fwd_total * (1.0 + backward_mult)
    return {
        "fwd_gflops": round(fwd_total / 1e9, 4),
        "step_gflops": round(step_total / 1e9, 4),
        "mix_share_pct": round(mix_fwd / max(1.0, fwd_total) * 100.0, 2),
        "head_share_pct": round(head_fwd / max(1.0, fwd_total) * 100.0, 2),
    }


def audit_linear_probe_hop_ladder(
    layer_states_tr: List[torch.Tensor],   # List[L] of [N_tr, D]
    targets_tr: torch.Tensor,              # [N_tr, H] ground-truth intermediate node IDs for hops 1..H
    layer_states_te: List[torch.Tensor],   # List[L] of [N_te, D]
    targets_te: torch.Tensor,              # [N_te, H]
    n_classes: int,
    ridge: float = 1.0,
) -> Dict[str, Any]:
    """Closed-form Ridge linear probe per layer/loop for hops 1..H (P1 Inside-the-Loop algorithmic ladder)."""
    n_layers = len(layer_states_tr)
    n_hops = int(targets_tr.shape[1])
    acc_matrix: List[List[float]] = []

    for l_idx in range(n_layers):
        Xtr = layer_states_tr[l_idx].float()
        Xte = layer_states_te[l_idx].float()
        # Standardize & add bias column
        mu = Xtr.mean(dim=0, keepdim=True)
        sd = Xtr.std(dim=0, keepdim=True).clamp_min(1e-5)
        Xtr_n = torch.cat([(Xtr - mu) / sd, torch.ones(Xtr.shape[0], 1)], dim=1)
        Xte_n = torch.cat([(Xte - mu) / sd, torch.ones(Xte.shape[0], 1)], dim=1)
        d_p = Xtr_n.shape[1]
        gram = Xtr_n.T @ Xtr_n + ridge * torch.eye(d_p)
        row_accs: List[float] = []
        for h_idx in range(n_hops):
            Ytr = F.one_hot(targets_tr[:, h_idx].long(), num_classes=n_classes).float()
            W = torch.linalg.solve(gram, Xtr_n.T @ Ytr)
            preds = (Xte_n @ W).argmax(dim=-1)
            row_accs.append(round(float((preds == targets_te[:, h_idx]).float().mean().item() * 100.0), 2))
        acc_matrix.append(row_accs)

    # Ladder summary: highest hop index (>= 50% probe accuracy) reached at each layer
    frontier_by_layer = [
        max([0] + [h + 1 for h, a in enumerate(row) if a >= 50.0])
        for row in acc_matrix
    ]
    return {
        "n_layers_probed": n_layers,
        "n_hops": n_hops,
        "acc_pct_by_layer_hop": acc_matrix,
        "solved_hop_frontier_by_layer": frontier_by_layer,
        "monotonic_ladder_detected": bool(len(frontier_by_layer) >= 2 and frontier_by_layer[-1] > frontier_by_layer[0] and frontier_by_layer[-1] >= 4),
    }


def audit_gradient_sign_flip(
    acc_normal_pct: float,
    acc_flipped_pct: float,
    min_drop_pp: float = 50.0,
) -> Dict[str, Any]:
    """Mandatory ICML 2026 sign-flip verification (P4): negating fast-weight update MUST collapse accuracy."""
    drop_pp = acc_normal_pct - acc_flipped_pct
    return {
        "acc_normal_pct": round(acc_normal_pct, 2),
        "acc_sign_flipped_pct": round(acc_flipped_pct, 2),
        "drop_pp": round(drop_pp, 2),
        "pass_sign_flip_check": bool(drop_pp >= min_drop_pp),
    }


def audit_boundary_histogram(
    boundary_mask: np.ndarray,       # [N, T] bool/0-1 chunk boundaries
    high_entropy_mask: np.ndarray,   # [N, T] bool/0-1 ground-truth high-entropy turns
    target_ratio_bounds: Tuple[float, float] = (3.2, 4.8),
) -> Dict[str, Any]:
    """Audit dynamic chunking boundary alignment & ratio stability (P5 / Route 2 DC-ER)."""
    b = boundary_mask.astype(np.float64)
    hi = high_entropy_mask.astype(np.bool_)
    lo = ~hi
    total_rate = float(b.mean())
    comp_ratio = float(1.0 / max(1e-6, total_rate))
    hi_density = float(b[hi].mean()) if hi.any() else 0.0
    lo_density = float(b[lo].mean()) if lo.any() else 0.0
    selectivity = hi_density / max(1e-6, lo_density)
    return {
        "compression_ratio": round(comp_ratio, 3),
        "high_entropy_boundary_density": round(hi_density, 4),
        "low_entropy_boundary_density": round(lo_density, 4),
        "entropy_selectivity_ratio": round(selectivity, 3),
        "ratio_in_bounds": bool(target_ratio_bounds[0] <= comp_ratio <= target_ratio_bounds[1]),
        "pass_boundary_audit": bool(target_ratio_bounds[0] <= comp_ratio <= target_ratio_bounds[1] and selectivity >= 1.5),
    }
