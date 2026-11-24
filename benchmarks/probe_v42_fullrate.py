"""90-Second L4 Ablation: Whole-Block Compiled Full-Rate PULSE-v4.2 & Pillar-1 Compressed I/O vs TF-Compiled.

Tests exact root causes discovered from exp_pulse_v32_micro2k vs exp_pulse_v4_micro2k:
0. TF-Compiled (Baseline: 6L, embed=192, d_ff=1408, 29.12M, Fused AdamW)
1. PULSE-v4.2-6L (Full-Rate M=1, embed=192, 3 FastGDN + 3 GQA-FlexSWA-128, d_ff=1216, Canon-A/B/C, Whole-Block Compile)
2. PULSE-v4.2-Pillar1-8L-GDN (Full-Rate M=1, embed=128, 4 FastGDN + 4 GQA-FlexSWA-128, d_ff=960, Canon-A/B/C, Whole-Block Compile)
3. PULSE-v4.2-Pillar1-8L-KDA (Full-Rate M=1, embed=128, 4 KDA + 4 GQA-FlexSWA-128, d_ff=864, Canon-A/B/C, Whole-Block Compile)
4. PULSE-v4.2-Pillar1-6L-SpeedKing (Full-Rate M=1, embed=128, 3 FastGDN + 3 GQA-FlexSWA-128, d_ff=1536, Canon-A/B/C, Whole-Block Compile)
"""

from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pulse.experiment import FairMuonAdamW
from pulse.pulse_v3 import PULSEv3Config, PULSEv3Model
from pulse.pulse_v4 import PULSEv4Config, PULSEv4Model
from pulse.scalable_transformer import RMSNorm, ScalableTransformerConfig, ScalableTransformerLM, _apply_rope
from pulse.v3_blocks import (
    CanonLayer,
    _get_flex_attention,
    _probe_fla_gdn_backend,
    _probe_fla_kda_backend,
)


@torch.no_grad()
def neyman_pearson_free_energy(
    model: nn.Module, prompt_ids: torch.Tensor, cand_ids: torch.Tensor
) -> torch.Tensor:
    """Upgraded Dual-Channel Neyman-Pearson Log-Bayes-Factor + Peak Outlier Tail-Risk + PC Free Energy.

    Key mathematical fix over prior softplus-clamped PMI:
    1. Sums Exact Log-Bayes-Factor `(NLL(c|prompt) - 1.0 * NLL(c|empty))` in unclamped log-odds space
       BEFORE positive projection, preserving strong negative evidence when prompt entities echo!
    2. Adds sharp beta=3.5 Peak Tail-Risk (quadratic entropy-normalized surprisal + MAP log-odds gap)
       so a single 1-word entity swap spikes unmistakably without dilution!
    3. Adds 0-backprop Predictive Coding Core energy, then maps via strictly monotone positive softplus.
    """
    p_len = prompt_ids.shape[1]
    c_len = cand_ids.shape[1]
    full_ids = torch.cat([prompt_ids, cand_ids], dim=1)
    out = model(full_ids, use_fast_weights=False, max_think_steps=16, return_innovation=False, return_state=False)
    logits = out["logits"][:, p_len - 1 : p_len + c_len - 1, :].float()
    log_p = F.log_softmax(logits, dim=-1)
    probs = log_p.exp()
    nll_tok = -log_p.gather(dim=-1, index=cand_ids.unsqueeze(-1)).squeeze(-1)
    top1_nll = -log_p.max(dim=-1).values
    neg_lp = -log_p
    ent_tok = (probs * neg_lp).sum(dim=-1)
    std_tok = (probs * (neg_lp - ent_tok.unsqueeze(-1)).pow(2)).sum(dim=-1).clamp_min(0.25).sqrt()

    # Unconditional prior NLL(c | empty)
    bos_id = min(model.config.vocab_size - 1, 50256)
    bos = torch.full((cand_ids.shape[0], 1), bos_id, device=cand_ids.device, dtype=torch.long)
    uncond_in = torch.cat([bos, cand_ids[:, :-1]], dim=1)
    out_u = model(uncond_in, use_fast_weights=False, max_think_steps=1, return_innovation=False, return_state=False)
    nll_u = -F.log_softmax(out_u["logits"].float(), dim=-1).gather(dim=-1, index=cand_ids.unsqueeze(-1)).squeeze(-1)

    # Channel 1: Exact Neyman-Pearson Log-Bayes-Factor (alpha = 1.0, UNCLAMPED!)
    ll_ratio = nll_tok - nll_u  # [B, K]: negative if prompt predicts cand, positive if mismatch!
    lse_ctx = (torch.logsumexp(1.5 * ll_ratio, dim=-1) - math.log(max(c_len, 1))) / 1.5
    s_ctx = ll_ratio.mean(dim=-1) + 0.45 * lse_ctx

    # Channel 2: Sharp Pointwise Tail-Risk Outlier (catches 1-2 word Hard Entity Swap!)
    z_pos = ((nll_tok - ent_tok) / std_tok).clamp(0.0, 12.0)
    map_gap = (nll_tok - top1_nll).clamp(0.0, 20.0)
    tok_outlier = 0.55 * z_pos.pow(2) + 0.25 * map_gap
    s_peak = (torch.logsumexp(3.5 * tok_outlier, dim=-1) - math.log(max(c_len, 1))) / 3.5

    # Channel 3: Zero-Backprop Predictive Coding Core Concept Energy
    u_ctx = model._cue_to_causal_key(prompt_ids, use_fast_weights=False)  # type: ignore[operator]
    z_cand = model.encode_target_concept(cand_ids)  # type: ignore[operator]
    e_pred = model.causal_layer.energy(z_cand, u_ctx, use_fast_weights=False)  # type: ignore[operator]

    # Combine in log-odds space FIRST (zero information loss!), THEN strictly monotone positive energy map!
    raw_score = 1.00 * s_ctx + 0.50 * s_peak + 0.15 * e_pred
    return F.softplus(raw_score + 2.0, beta=1.2)


def audit_anti_hallucination(val_blocks: torch.Tensor, dev: torch.device) -> None:
    """Part 0: Audit Upgraded Neyman-Pearson Anti-Hallucination on 128 real held-out stories."""
    print("\n=== PART 0: NEYMAN-PEARSON DUAL-CHANNEL ANTI-HALLUCINATION AUDIT (128 REAL STORIES) ===", flush=True)
    ckpts = [
        ("PULSE-v3.2 (7.11 PPL ckpt)", "v3", Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI/runs/exp_pulse_v32_micro2k/checkpoints/pulse_latest.pt")),
        ("PULSE-v4   (8.39 PPL ckpt)", "v4", Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI/runs/exp_pulse_v4_micro2k/checkpoints/pulse_latest.pt")),
    ]
    for label, ver, path in ckpts:
        if not path.exists():
            continue
        m: nn.Module = (PULSEv3Model(PULSEv3Config.colab_30m()) if ver == "v3" else PULSEv4Model(PULSEv4Config.colab_30m())).to(dev)
        m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model_state_dict"])
        m.eval()
        k = m.config.chunk_size
        et_all, ef_all, eh_all = [], [], []
        for bi in range(8):  # 8 x 16 = 128 real held-out stories!
            g = torch.Generator(device="cpu").manual_seed(42 * 100_003 + 999_002 + bi)
            idx = torch.randint(0, val_blocks.shape[0], (16,), generator=g)
            b = val_blocks.index_select(0, idx).long().to(dev)
            split = (b.shape[1] // k - 1) * k
            pr = b[:, :split]
            tc = b[:, split : split + k]
            fc = torch.roll(tc, shifts=1, dims=0)
            hc = tc.clone()
            sp = k // 2
            donor = torch.roll(tc, shifts=8, dims=0)
            hc[:, sp] = donor[:, sp]
            hc[:, sp - 1] = donor[:, sp - 1]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                et_all.append(neyman_pearson_free_energy(m, pr, tc))
                ef_all.append(neyman_pearson_free_energy(m, pr, fc))
                eh_all.append(neyman_pearson_free_energy(m, pr, hc))
        et, ef, eh = torch.cat(et_all), torch.cat(ef_all), torch.cat(eh_all)
        auroc_c = float((ef.unsqueeze(1) > et.unsqueeze(0)).float().mean().item())
        auroc_h = float((eh.unsqueeze(1) > et.unsqueeze(0)).float().mean().item())
        pair_c = float((ef > et).float().mean().item()) * 100.0
        pair_h = float((eh > et).float().mean().item()) * 100.0
        print(
            f"[NP-AUDIT] {label} | E_true={et.mean():.3f} | E_cross={ef.mean():.3f} ({ef.mean()/et.mean():.1f}x) | "
            f"E_hard={eh.mean():.3f} ({eh.mean()/et.mean():.1f}x) | "
            f"Cross AUROC={auroc_c:.4f} (paired={pair_c:.1f}%) | Hard AUROC={auroc_h:.4f} (paired={pair_h:.1f}%)",
            flush=True,
        )
        del m
        gc.collect()
        torch.cuda.empty_cache()


def _swa_block_mask(seq_len: int, window_size: int, device: torch.device) -> Any:
    flex_fn, create_bm = _get_flex_attention()
    if flex_fn is None or create_bm is None:
        return None
    w = int(window_size)

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) < w)

    return create_bm(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


class WholeSWABlock(nn.Module):
    """Single-Graph Compiled GQA FlexAttention SWA-128 + Triton Canon-A/B/C + Qwen Gate + SwiGLU."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        window_size: int = 128,
        conv_kernel: int = 4,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.qkv_dim = d_model + 2 * self.kv_dim
        self.window_size = window_size

        self.skip_lex = nn.Parameter(torch.full((d_model,), 0.0 if is_first else 0.05))
        self.norm1 = RMSNorm(d_model)
        self.canon_a = CanonLayer(d_model, kernel_size=conv_kernel, activation="none")
        self.qkvg_proj = nn.Linear(d_model, self.qkv_dim + d_model, bias=False)
        self.canon_b = CanonLayer(self.qkv_dim, kernel_size=conv_kernel, activation="none")
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.canon_c = CanonLayer(d_model, kernel_size=conv_kernel, activation="none")
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self._flex_fn, _ = _get_flex_attention()

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_mask: Any,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        n1 = self.norm1(x + self.skip_lex.to(x.dtype) * x0)
        n1, _ = self.canon_a(n1, state=None, return_state=False)
        qkv_raw, gate_raw = self.qkvg_proj(n1).split([self.qkv_dim, self.d_model], dim=-1)
        qkv, _ = self.canon_b(qkv_raw, state=None, return_state=False)
        q, k, v = qkv.split([self.d_model, self.kv_dim, self.kv_dim], dim=-1)
        q = _apply_rope(q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        k = _apply_rope(k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2), cos, sin)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self._flex_fn is not None and block_mask is not None and x.is_cuda:
            out = self._flex_fn(
                q, k, v, block_mask=block_mask, enable_gqa=(self.n_heads != self.n_kv_heads)
            )
        else:
            if self.n_heads != self.n_kv_heads:
                rep = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out_m = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        x = x + self.o_proj(out_m * torch.sigmoid(gate_raw))
        n2, _ = self.canon_c(self.norm2(x), state=None, return_state=False)
        g, u = self.w_gate_up(n2).chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u)


class FusedDeltaBlock(nn.Module):
    """2-Graph + 1-FLA-Kernel Fused DeltaNet (FastGDN or KDA) + Triton Canon-A/B/C + ValueRes + SwiGLU."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_kda: bool = False,
        gate_rank: int = 64,
        conv_kernel: int = 4,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_kda = use_kda
        self.is_first = is_first

        self.skip_lex = nn.Parameter(torch.full((d_model,), 0.0 if is_first else 0.05))
        self.norm1 = RMSNorm(d_model)
        self.canon_a = CanonLayer(d_model, kernel_size=conv_kernel, activation="none")
        if use_kda:
            r = min(gate_rank, d_model)
            self._splits = (3 * d_model, d_model, r, r, n_heads)
            self.in_proj = nn.Linear(d_model, sum(self._splits), bias=False)
            self.alpha_up = nn.Linear(r, d_model, bias=True)
            self.write_up = nn.Linear(r, d_model, bias=True)
            self.erase_bias = nn.Parameter(torch.full((n_heads,), -1.0))
            with torch.no_grad():
                self.alpha_up.bias.copy_(torch.linspace(-0.5, -4.5, n_heads).repeat_interleave(self.head_dim))
                self.write_up.bias.fill_(-1.0)
        else:
            self._splits = (3 * d_model, d_model, n_heads, n_heads, n_heads)
            self.in_proj = nn.Linear(d_model, sum(self._splits), bias=False)
            self.decay_bias = nn.Parameter(torch.linspace(-0.5, -4.5, n_heads))
            self.erase_bias = nn.Parameter(torch.full((n_heads,), -1.0))
            self.write_bias = nn.Parameter(torch.full((n_heads,), -1.0))

        self.canon_b = CanonLayer(3 * d_model, kernel_size=conv_kernel, activation="silu")
        self.v_res_logit = nn.Parameter(torch.full((n_heads,), -2.0))
        self.head_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.canon_c = CanonLayer(d_model, kernel_size=conv_kernel, activation="none")
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

        self._c_pre: Optional[Any] = None
        self._c_pre_v: Optional[Any] = None
        self._c_post: Optional[Any] = None
        self._fla_fn: Optional[Any] = None

    def compile_fast(self, device: torch.device) -> None:
        probe = _probe_fla_kda_backend(device) if self.use_kda else _probe_fla_gdn_backend(device)
        if probe.get("available"):
            self._fla_fn = probe["fn"]
        if device.type == "cuda":
            self._c_pre = torch.compile(self._pre_impl, dynamic=True)
            self._c_pre_v = torch.compile(self._pre_vres_impl, dynamic=True)
            self._c_post = torch.compile(self._post_impl, dynamic=True)

    def _pre_common(self, x: torch.Tensor, x0: torch.Tensor, v0: Optional[torch.Tensor]):
        bsz, seq_len, _ = x.shape
        n1 = self.norm1(x + self.skip_lex.to(x.dtype) * x0)
        n1, _ = self.canon_a(n1, state=None, return_state=False)
        parts = self.in_proj(n1).split(self._splits, dim=-1)
        qkv, _ = self.canon_b(parts[0], state=None, return_state=False)
        gate_raw = parts[1]
        q, k, v = qkv.chunk(3, dim=-1)
        q = F.normalize(q.view(bsz, seq_len, self.n_heads, self.head_dim), p=2, dim=-1, eps=1e-6).to(x.dtype)
        k = F.normalize(k.view(bsz, seq_len, self.n_heads, self.head_dim), p=2, dim=-1, eps=1e-6).to(x.dtype)
        v_raw = v.view(bsz, seq_len, self.n_heads, self.head_dim)
        if v0 is not None:
            lam = torch.sigmoid(self.v_res_logit).to(x.dtype).view(1, 1, self.n_heads, 1)
            v_use = (1.0 - lam) * v_raw + lam * v0.to(x.dtype)
        else:
            v_use = v_raw

        if self.use_kda:
            a_lat, w_lat, e_raw = parts[2], parts[3], parts[4]
            log_a = -F.softplus(self.alpha_up(F.silu(a_lat)).view(bsz, seq_len, self.n_heads, self.head_dim).float()).clamp(1e-4, 10.0)
            beta = (0.05 + 1.90 * torch.sigmoid(e_raw + self.erase_bias)).to(x.dtype)
            w_val = 2.0 * torch.sigmoid(self.write_up(F.silu(w_lat)).view(bsz, seq_len, self.n_heads, self.head_dim))
            v_eff = ((w_val / beta.unsqueeze(-1)) * v_use).to(x.dtype)
        else:
            a_raw, e_raw, w_raw = parts[2], parts[3], parts[4]
            log_a = -F.softplus(a_raw.float() + self.decay_bias).clamp(1e-4, 10.0)
            beta = (0.05 + 1.90 * torch.sigmoid(e_raw + self.erase_bias)).to(x.dtype)
            w_val = (2.0 * torch.sigmoid(w_raw + self.write_bias)).to(x.dtype)
            v_eff = ((w_val / beta).unsqueeze(-1) * v_use).to(x.dtype)
        return q, k, v_eff, log_a, beta, gate_raw, v_raw

    def _pre_impl(self, x: torch.Tensor, x0: torch.Tensor):
        return self._pre_common(x, x0, None)

    def _pre_vres_impl(self, x: torch.Tensor, x0: torch.Tensor, v0: torch.Tensor):
        return self._pre_common(x, x0, v0)

    def _post_impl(self, x: torch.Tensor, o_all: torch.Tensor, gate_raw: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        o_m = self.head_norm(o_all.to(x.dtype)).contiguous().view(bsz, seq_len, self.d_model)
        x = x + self.o_proj(o_m * torch.sigmoid(gate_raw))
        n2, _ = self.canon_c(self.norm2(x), state=None, return_state=False)
        g, u = self.w_gate_up(n2).chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u)

    def forward(
        self, x: torch.Tensor, x0: torch.Tensor, v0: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if v0 is not None:
            fn = self._c_pre_v if self._c_pre_v is not None else self._pre_vres_impl
            q, k, v_eff, log_a, beta, gate_raw, v_raw = fn(x, x0, v0)
        else:
            fn = self._c_pre if self._c_pre is not None else self._pre_impl
            q, k, v_eff, log_a, beta, gate_raw, v_raw = fn(x, x0)

        if self._fla_fn is not None and x.is_cuda:
            kd = torch.bfloat16 if (q.dtype == torch.float32 and torch.cuda.is_bf16_supported()) else q.dtype
            o_all, _ = self._fla_fn(
                q=q.to(kd).contiguous(),
                k=k.to(kd).contiguous(),
                v=v_eff.to(kd).contiguous(),
                g=log_a.float().contiguous(),
                beta=beta.to(kd).contiguous(),
                scale=1.0,
                output_final_state=False,
            )[:2]
        else:
            # Tiny CPU fallback for local sanity verification
            o_all = v_eff

        post_fn = self._c_post if self._c_post is not None else self._post_impl
        return post_fn(x, o_all, gate_raw), v_raw


class ProbeV42LM(nn.Module):
    """Full-Rate M=1 Alternating FastGDN/KDA + Whole-Compiled GQA-FlexSWA-128 + Pillar-1 I/O."""

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        n_layers: int = 8,
        d_ff: int = 960,
        use_kda: bool = False,
        window_size: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.emb_in = nn.Linear(embed_dim, d_model, bias=False)

        # Alternating DeltaNet (even) -> GQA FlexSWA-128 (odd)
        self.blocks = nn.ModuleList()
        self.is_swa: List[bool] = []
        for i in range(n_layers):
            swa = (i % 2 == 1)
            self.is_swa.append(swa)
            if swa:
                self.blocks.append(
                    WholeSWABlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        d_ff=d_ff,
                        window_size=window_size,
                        is_first=(i == 0),
                    )
                )
            else:
                self.blocks.append(
                    FusedDeltaBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        use_kda=use_kda,
                        is_first=(i == 0),
                    )
                )
        self.norm_f = RMSNorm(d_model)
        self.emb_out = nn.Linear(d_model, embed_dim, bias=False)
        half = self.head_dim // 2
        self.register_buffer(
            "_inv_freq", 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half)), persistent=False
        )
        self._compiled_swa: List[Optional[Any]] = [None] * n_layers
        self._bm_cache: Dict[Tuple[int, str], Any] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, 0.0, 1.0 / math.sqrt(self.tok_emb.embedding_dim))
        nn.init.normal_(self.emb_in.weight, 0.0, 1.0 / math.sqrt(self.tok_emb.embedding_dim))
        nn.init.normal_(self.emb_out.weight, 0.0, 1.0 / math.sqrt(self.d_model))
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def compile_fast_paths(self) -> None:
        dev = self.tok_emb.weight.device
        for i, blk in enumerate(self.blocks):
            if self.is_swa[i]:
                if dev.type == "cuda":
                    self._compiled_swa[i] = torch.compile(blk, dynamic=True)
            else:
                blk.compile_fast(dev)  # type: ignore[operator]

    def _get_rope_and_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        half = self.head_dim // 2
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self._inv_freq.to(device))
        cos = freqs.cos().to(dtype).view(1, 1, seq_len, half)
        sin = freqs.sin().to(dtype).view(1, 1, seq_len, half)
        key = (seq_len, str(device))
        if key not in self._bm_cache and device.type == "cuda":
            self._bm_cache[key] = _swa_block_mask(seq_len, self.window_size, device)
        return cos, sin, self._bm_cache.get(key)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x0 = self.emb_in(self.tok_emb(input_ids))
        x = x0
        cos, sin, bm = self._get_rope_and_mask(input_ids.shape[1], input_ids.device, x.dtype)
        v0: Optional[torch.Tensor] = None
        for i, blk in enumerate(self.blocks):
            if self.is_swa[i]:
                fn = self._compiled_swa[i] if self._compiled_swa[i] is not None else blk
                x = fn(x, x0, cos, sin, bm)
            else:
                x, v_raw = blk(x, x0, None if i == 0 else v0)
                if i == 0:
                    v0 = v_raw
        return F.linear(self.emb_out(self.norm_f(x)), self.tok_emb.weight)

    def compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self.forward(batch[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))


def _benchmark_one(
    name: str,
    model: nn.Module,
    train_blocks: torch.Tensor,
    val_blocks: torch.Tensor,
    dev: torch.device,
    steps: int = 400,
    use_muon: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(42)
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= 29_120_192, f"{name} exceeded TF param budget: {n_params:,} > 29,120,192"
    if hasattr(model, "compile_fast_paths"):
        model.compile_fast_paths()  # type: ignore[operator]

    if use_muon:
        opt: torch.optim.Optimizer = FairMuonAdamW(
            list(model.parameters()), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=(dev.type == "cuda")
        )

    def sample_batch(blocks: torch.Tensor, idx: int, bs: int = 16) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(42 * 100_003 + idx)
        ii = torch.randint(0, blocks.shape[0], (bs,), generator=g)
        return blocks.index_select(0, ii).long().to(dev, non_blocking=True)

    @torch.no_grad()
    def eval_ppl() -> float:
        model.eval()
        losses = []
        for bi in range(12):
            b = sample_batch(val_blocks, 900_000 + bi, 16)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                losses.append(float(model.compute_loss(b).item()))  # type: ignore[operator]
        model.train()
        return float(math.exp(min(sum(losses) / len(losses), 20.0)))

    # 1. Measure T=512 training speed, VRAM & Val PPL @ mid and end!
    model.train()
    step_ms_list: List[float] = []
    ppl_mid = 0.0
    mid_step = steps // 2
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    warmup_s = int(steps * 0.05)
    for step in range(steps):
        if step < warmup_s:
            lr_m = (step + 1) / max(warmup_s, 1)
        else:
            prog = (step - warmup_s) / max(steps - warmup_s, 1)
            lr_m = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = 6e-4 * lr_m

        if step == 5 and dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for micro in range(2):
            b = sample_batch(train_blocks, step * 2 + micro, 16)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                (model.compute_loss(b) / 2.0).backward()  # type: ignore[operator]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if step >= 5:
            step_ms_list.append(dt)
        if step + 1 == mid_step:
            ppl_mid = eval_ppl()

    ppl_end = eval_ppl()
    med_512 = float(np.median(step_ms_list))
    vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if dev.type == "cuda" else 0.0

    # 2. Measure T=4096 training speed (2x2 = same 16,384 tokens/step!)
    b4k = torch.randint(4, 50257, (2, 4097), device=dev)
    model.train()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                (model.compute_loss(b4k) / 2.0).backward()  # type: ignore[operator]
        opt.step()
    runs_4k = []
    for _ in range(5):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                (model.compute_loss(b4k) / 2.0).backward()  # type: ignore[operator]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        runs_4k.append((time.perf_counter() - t0) * 1000.0)
    med_4k = float(np.median(runs_4k))

    # 3. Measure N=4096 & N=8192 warmed inference latency
    tok4k = torch.randint(4, 50257, (1, 4096), device=dev)
    tok8k = torch.randint(4, 50257, (1, 8192), device=dev)
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
        for _ in range(3):
            model(tok4k)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            model(tok4k)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        inf_4k = (time.perf_counter() - t0) * 1000.0 / 5.0

        for _ in range(2):
            model(tok8k)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            model(tok8k)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        inf_8k = (time.perf_counter() - t0) * 1000.0 / 3.0

    del model, opt, b4k, tok4k, tok8k
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[V4.2-ABLATION] {name:30s} | {n_params/1e6:5.2f}M | "
        f"T512={med_512:5.1f}ms ({vram_mb:4.0f}MB) | T4K={med_4k:5.1f}ms | "
        f"Inf4K={inf_4k:4.1f}ms Inf8K={inf_8k:4.1f}ms | PPL@{mid_step}={ppl_mid:5.2f} PPL@{steps}={ppl_end:5.2f}",
        flush=True,
    )
    return {"name": name, "params": n_params, "t512": med_512, "t4k": med_4k, "ppl_end": ppl_end}


def main() -> None:
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.cache_size_limit = 128
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reserve ~0.60M for PULSE's K=8 Core bridge + fast_ssm so full PULSE-v4.2 stays strictly <= 29.12M!
    specs = [
        ("0. TF-Compiled (AdamW)", False, lambda: ScalableTransformerLM(ScalableTransformerConfig.colab_30m())),
        ("1. V4.2-6L-e192 (3GDN+3SWA)", False, lambda: ProbeV42LM(embed_dim=192, n_layers=6, d_ff=1280, use_kda=False)),
        ("2. V4.2-P1-8L-GDN (e128)", False, lambda: ProbeV42LM(embed_dim=128, n_layers=8, d_ff=1024, use_kda=False)),
        ("3. V4.2-P1-8L-KDA (e128)", False, lambda: ProbeV42LM(embed_dim=128, n_layers=8, d_ff=960, use_kda=True)),
        ("4. V4.2-P1-6L-Speed (e128)", False, lambda: ProbeV42LM(embed_dim=128, n_layers=6, d_ff=1600, use_kda=False)),
        ("5. TF-Compiled (Muon)", True, lambda: ScalableTransformerLM(ScalableTransformerConfig.colab_30m())),
        ("6. V4.2-P1-8L-GDN (Muon)", True, lambda: ProbeV42LM(embed_dim=128, n_layers=8, d_ff=1024, use_kda=False)),
        ("7. V4.2-P1-6L-Speed (Muon)", True, lambda: ProbeV42LM(embed_dim=128, n_layers=6, d_ff=1600, use_kda=False)),
    ]

    if dev.type == "cpu":
        print("=== LOCAL CPU PARAM & SHAPE SANITY CHECK ===")
        tok = torch.randint(4, 200, (2, 33))
        for name, _, fn in specs[:5]:
            m = fn()
            n_p = sum(p.numel() for p in m.parameters())
            loss = m.compute_loss(tok)
            print(f"{name:30s} | params={n_p:,} ({n_p/1e6:.2f}M) | loss={loss.item():.3f}")
        return

    # Load cached pure TinyStories blocks from Drive
    cache_candidates = [
        Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI/runs/exp_pulse_v4_micro2k/data_cache"),
        Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI/runs/exp_pulse_v32_micro2k/data_cache"),
    ]
    cache_dir = next((p for p in cache_candidates if (p / "train_pure_tinystories_512.pt").exists()), None)
    assert cache_dir is not None, "Cached TinyStories not found in Drive runs!"
    train_blocks = torch.load(cache_dir / "train_pure_tinystories_512.pt", map_location="cpu", weights_only=True)
    val_blocks = torch.load(cache_dir / "val_pure_tinystories_512.pt", map_location="cpu", weights_only=True)

    # Part 0: Neyman-Pearson Anti-Hallucination Audit on 128 held-out real stories!
    audit_anti_hallucination(val_blocks, dev)

    print(f"\n=== PART 1 & 2: PULSE-v4.2 FULL-RATE + PILLAR-1 + SYMMETRICAL MUON 400-STEP SHOWDOWN ===", flush=True)
    for name, use_muon, fn in specs:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        _benchmark_one(name, fn(), train_blocks, val_blocks, dev, steps=400, use_muon=use_muon)


if __name__ == "__main__":
    main()
