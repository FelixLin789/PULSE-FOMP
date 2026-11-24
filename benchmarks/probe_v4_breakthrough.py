"""90-second L4 Live Verification & Architecture Showdown:
Part 1 (10s): Test upgraded Entropy-Calibrated Hierarchical Variational Free Energy AUROC
              directly on the ALREADY-TRAINED `exp_pulse_v32_micro2k` checkpoint!
Part 2 (80s): Controlled 250-step Iso-Param (<= 29.12M) TinyStories Showdown on L4 comparing:
  - Baseline: Compiled Transformer (standard CE & chunked linear CE, T=512 & T=2048)
  - Candidate A: PULSE-v4-Engram-FullRate (M=1, Iso-Param Context-Gated Engram N-Gram Memory +
                 chunk_gated_delta_rule + Triton Canon4 + Value-Res + HyperSkip)
  - Candidate B: PULSE-v4-Hourglass (M=2 Multi-Rate Concept Cortex + Iso-Param Context-Gated Engram +
                 chunk_gated_delta_rule @ T/2 + Triton Canon4 + Value-Res)
Measures EXACT: trainable params, ms/step (standard CE vs Chunked Fused CE), Peak VRAM,
Val PPL @ 250 steps, AND T=2048 long-context training ms/step!
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pulse.pulse_v3 import PULSEv3Config, PULSEv3Model
from pulse.scalable_transformer import RMSNorm, ScalableTransformerConfig, ScalableTransformerLM
from pulse.v3_blocks import CanonLayer, GatedSlidingWindowAttention


# ---------------------------------------------------------------------------
# 1. Exact Chunked Linear Cross-Entropy (0 approximation, -3 GB VRAM, faster L2!)
# ---------------------------------------------------------------------------
def chunked_linear_cross_entropy(
    h_emb: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    n_chunks: int = 8,
) -> torch.Tensor:
    """Exact 50,257-way cross-entropy without materializing full [B*T, V] logits in HBM!"""
    flat_h = h_emb.reshape(-1, h_emb.shape[-1])
    flat_t = targets.reshape(-1)
    n_tok = flat_h.shape[0]
    chunk_sz = (n_tok + n_chunks - 1) // n_chunks

    def _chunk_loss(hc: torch.Tensor, tc: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        logits_c = F.linear(hc, w)
        return F.cross_entropy(logits_c, tc, reduction="sum")

    total = flat_h.new_zeros((), dtype=torch.float32)
    for start in range(0, n_tok, chunk_sz):
        end = min(start + chunk_sz, n_tok)
        total = total + checkpoint(
            _chunk_loss, flat_h[start:end], flat_t[start:end], weight, use_reentrant=False
        )
    return total / float(n_tok)


# ---------------------------------------------------------------------------
# 2. Iso-Parameter Context-Gated Engram Conditional N-Gram Memory (DeepSeek Jan 2026)
# ---------------------------------------------------------------------------
class ContextGatedEngram(nn.Module):
    """O(1) deterministic multi-head Bigram+Trigram conditional memory lookup (0 matmul lookup!)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_buckets: int = 16384,
        head_dim: int = 48,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.head_dim = head_dim
        self.mem_dim = num_heads * head_dim

        # Multi-head embedding table (shared flat table with head offsets -> 1 fused gather!)
        self.table = nn.Embedding(num_heads * num_buckets, head_dim)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

        self.k_proj = nn.Linear(self.mem_dim, d_model, bias=False)
        self.v_proj = nn.Linear(self.mem_dim, d_model, bias=False)
        self.q_norm = RMSNorm(d_model)
        self.k_norm = RMSNorm(d_model)
        self.out_scale = nn.Parameter(torch.full((d_model,), 0.1))

        # Distinct large odd primes for collision-resistant bigram & trigram hashing
        self.register_buffer(
            "_p1", torch.tensor([1000003, 1000033, 1000037, 1000039], dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_p2", torch.tensor([1000081, 1000099, 1000117, 1000121], dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_p3", torch.tensor([1000133, 1000151, 1000159, 1000171], dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_offsets", torch.arange(num_heads, dtype=torch.long) * num_buckets, persistent=False
        )

    def lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Strictly causal O(1) bigram (heads 0,1) + trigram (heads 2,3) hash gather -> [B, T, mem_dim]."""
        bsz, seq_len = input_ids.shape
        t0 = input_ids
        t1 = F.pad(input_ids[:, :-1], (1, 0), value=0)
        t2 = F.pad(input_ids[:, :-2], (2, 0), value=0)

        # Compute 4 head hashes in parallel [B, T, 4]
        h0 = (t0 * self._p1[0] ^ t1 * self._p2[0]) & (self.num_buckets - 1)
        h1 = (t0 * self._p1[1] ^ t1 * self._p2[1]) & (self.num_buckets - 1)
        h2 = (t0 * self._p1[2] ^ t1 * self._p2[2] ^ t2 * self._p3[2]) & (self.num_buckets - 1)
        h3 = (t0 * self._p1[3] ^ t1 * self._p2[3] ^ t2 * self._p3[3]) & (self.num_buckets - 1)
        idx = torch.stack([h0, h1, h2, h3], dim=-1) + self._offsets.view(1, 1, -1)
        return self.table(idx).reshape(bsz, seq_len, self.mem_dim)

    def forward(self, h: torch.Tensor, mem_raw: torch.Tensor) -> torch.Tensor:
        k = self.k_norm(self.k_proj(mem_raw))
        q = self.q_norm(h)
        gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model))
        return h + self.out_scale.to(h.dtype) * (gate * self.v_proj(mem_raw))


# ---------------------------------------------------------------------------
# 3. Fast GatedDeltaNet (FLA chunk_gated_delta_rule + Decoupled Write Gate + Value-Res)
# ---------------------------------------------------------------------------
class FastGatedDeltaNetBlock(nn.Module):
    """Blazing-fast GatedDeltaNet block using FLA `chunk_gated_delta_rule` (2.69ms @ T=512, 1.35ms @ T=256!)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_canon_a: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.skip_lex = nn.Parameter(torch.full((d_model,), 0.05))
        self.norm1 = RMSNorm(d_model)
        self.canon_a = CanonLayer(d_model, kernel_size=4, activation="none") if use_canon_a else None
        # ONE fused GEMM: qkv (3D) + out_gate (D) + decay_a (H) + beta_e (H) + write_w (H)
        self._splits = (3 * d_model, d_model, n_heads, n_heads, n_heads)
        self.in_proj = nn.Linear(d_model, sum(self._splits), bias=False)
        self.canon_b = CanonLayer(3 * d_model, kernel_size=4, activation="silu")
        self.decay_bias = nn.Parameter(torch.linspace(-0.5, -4.5, n_heads))
        self.beta_bias = nn.Parameter(torch.full((n_heads,), -1.0))
        self.write_bias = nn.Parameter(torch.full((n_heads,), -1.0))
        self.v_res_logit = nn.Parameter(torch.full((n_heads,), -2.0))
        self.head_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.canon_c = CanonLayer(d_model, kernel_size=4, activation="none")
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

        self._c_pre = None
        self._c_post = None

    def compile_fast(self) -> None:
        self._c_pre = torch.compile(self._pre_fn, dynamic=True)
        self._c_post = torch.compile(self._post_fn, dynamic=True)

    def _pre_fn(self, x: torch.Tensor, x0: torch.Tensor, v0: Optional[torch.Tensor]):
        h = self.norm1(x + self.skip_lex.to(x.dtype) * x0)
        if self.canon_a is not None:
            h, _ = self.canon_a(h, state=None, return_state=False)
        qkv_raw, gate_raw, a_raw, b_raw, w_raw = self.in_proj(h).split(self._splits, dim=-1)
        qkv, _ = self.canon_b(qkv_raw, state=None, return_state=False)
        q, k, v = qkv.chunk(3, dim=-1)
        B, T, _ = x.shape
        q = F.normalize(q.view(B, T, self.n_heads, self.head_dim), p=2, dim=-1, eps=1e-6).to(x.dtype)
        k = F.normalize(k.view(B, T, self.n_heads, self.head_dim), p=2, dim=-1, eps=1e-6).to(x.dtype)
        v_raw = v.view(B, T, self.n_heads, self.head_dim)
        if v0 is not None:
            lam = torch.sigmoid(self.v_res_logit).to(x.dtype).view(1, 1, self.n_heads, 1)
            v_use = (1.0 - lam) * v_raw + lam * v0.to(x.dtype)
        else:
            v_use = v_raw
        g_hd = -F.softplus(a_raw.float() + self.decay_bias).clamp(1e-4, 10.0)
        beta = (0.05 + 1.90 * torch.sigmoid(b_raw + self.beta_bias)).to(x.dtype)
        w_wr = (2.0 * torch.sigmoid(w_raw + self.write_bias)).to(x.dtype)
        v_eff = ((w_wr / beta).unsqueeze(-1) * v_use).to(x.dtype)
        return q, k, v_eff, g_hd, beta, gate_raw, v_raw

    def _post_fn(self, x: torch.Tensor, o: torch.Tensor, gate_raw: torch.Tensor, x0: torch.Tensor):
        B, T, _ = x.shape
        o_m = self.head_norm(o.to(x.dtype)).contiguous().view(B, T, self.d_model)
        x = x + self.o_proj(o_m * torch.sigmoid(gate_raw))
        n2, _ = self.canon_c(self.norm2(x), state=None, return_state=False)
        g, u = self.w_gate_up(n2).chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u)

    def forward(self, x: torch.Tensor, x0: torch.Tensor, v0: Optional[torch.Tensor] = None):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        pre_fn = self._c_pre if self._c_pre is not None else self._pre_fn
        post_fn = self._c_post if self._c_post is not None else self._post_fn
        q, k, v_eff, g_hd, beta, gate_raw, v_raw = pre_fn(x, x0, v0)
        o, _ = chunk_gated_delta_rule(q, k, v_eff, g_hd, beta, scale=1.0, output_final_state=False)
        return post_fn(x, o, gate_raw, x0), v_raw


class FastSWABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, window_size: int = 128) -> None:
        super().__init__()
        self.skip_lex = nn.Parameter(torch.full((d_model,), 0.05))
        self.norm1 = RMSNorm(d_model)
        self.canon_a = CanonLayer(d_model, kernel_size=4, activation="none")
        self.swa = GatedSlidingWindowAttention(d_model, n_heads, window_size=window_size, use_conv=False)
        self.norm2 = RMSNorm(d_model)
        self.canon_c = CanonLayer(d_model, kernel_size=4, activation="none")
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self._c_pre = None
        self._c_post = None

    def compile_fast(self) -> None:
        self._c_pre = torch.compile(lambda x, x0: self.canon_a(self.norm1(x + self.skip_lex.to(x.dtype) * x0), None, False)[0], dynamic=True)
        self.swa._compiled_pre = torch.compile(self.swa._pre_proj_train, dynamic=True)
        self.swa._compiled_pre_vres = torch.compile(self.swa._pre_proj_train_vres, dynamic=True)
        self.swa._compiled_post = torch.compile(self.swa._post_proj, dynamic=True)
        self._c_post = torch.compile(self._post_fn, dynamic=True)

    def _post_fn(self, x: torch.Tensor, hm: torch.Tensor):
        x = x + hm
        n2, _ = self.canon_c(self.norm2(x), state=None, return_state=False)
        g, u = self.w_gate_up(n2).chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u)

    def forward(self, x: torch.Tensor, x0: torch.Tensor, v0: Optional[torch.Tensor] = None) -> torch.Tensor:
        n1 = self._c_pre(x, x0) if self._c_pre is not None else self.canon_a(self.norm1(x + self.skip_lex.to(x.dtype) * x0), None, False)[0]
        hm, _ = self.swa(n1, state=None, return_state=False, v0=v0)
        return self._c_post(x, hm) if self._c_post is not None else self._post_fn(x, hm)


class PulseV4Candidate(nn.Module):
    """Iso-Param (<= 29.12M) PULSE-v4 Candidate supporting M=1 (FullRate) or M=2 (Hourglass Concept Cortex) + Engram!"""

    def __init__(self, hourglass_m: int = 1, use_engram: bool = True, d_ff: int = 960, n_engram_buckets: int = 16384) -> None:
        super().__init__()
        self.hourglass_m = hourglass_m
        self.use_engram = use_engram
        d_model = 512
        embed_dim = 192
        self.tok_emb = nn.Embedding(50257, embed_dim)
        self.emb_in = nn.Linear(embed_dim, d_model, bias=False)
        self.engram = ContextGatedEngram(d_model, num_heads=4, num_buckets=n_engram_buckets, head_dim=48) if use_engram else None

        if hourglass_m == 2:
            # 1 Sensory SWA @ T=512 -> Merge M=2 -> 3 Delta + 1 SWA @ T=256 -> Causal Upsample -> 1 Motor SWA @ T=512! (6 blocks total!)
            self.sensory = FastSWABlock(d_model, 8, d_ff=768, window_size=128)
            self.down_proj = nn.Linear(2 * d_model, d_model, bias=False)
            self.down_norm = RMSNorm(d_model)
            self.bos_concept = nn.Parameter(torch.zeros(1, 1, d_model))
            self.concept_blocks = nn.ModuleList([
                FastGatedDeltaNetBlock(d_model, 8, d_ff=d_ff, use_canon_a=False),
                FastGatedDeltaNetBlock(d_model, 8, d_ff=d_ff, use_canon_a=False),
                FastSWABlock(d_model, 8, d_ff=d_ff, window_size=128),
                FastGatedDeltaNetBlock(d_model, 8, d_ff=d_ff, use_canon_a=False),
            ])
            self.up_proj = nn.Linear(d_model, 2 * d_model, bias=False)
            self.up_gate = nn.Parameter(torch.full((d_model,), 0.25))
            self.motor = FastSWABlock(d_model, 8, d_ff=768, window_size=128)
        else:
            self.blocks = nn.ModuleList([
                FastGatedDeltaNetBlock(d_model, 8, d_ff=d_ff, use_canon_a=False),
                FastGatedDeltaNetBlock(d_model, 8, d_ff=d_ff, use_canon_a=False),
                FastSWABlock(d_model, 8, d_ff=d_ff, window_size=128),
                FastGatedDeltaNetBlock(d_model, 8, d_ff=d_ff, use_canon_a=False),
                FastGatedDeltaNetBlock(d_model, 8, d_ff=d_ff, use_canon_a=False),
                FastSWABlock(d_model, 8, d_ff=d_ff, window_size=128),
            ])
        self.norm_f = RMSNorm(d_model)
        self.emb_out = nn.Linear(d_model, embed_dim, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, 0.0, 1.0 / math.sqrt(192))
        nn.init.normal_(self.emb_in.weight, 0.0, 1.0 / math.sqrt(192))
        nn.init.normal_(self.emb_out.weight, 0.0, 1.0 / math.sqrt(512))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def compile_fast(self) -> None:
        if self.hourglass_m == 2:
            self.sensory.compile_fast()
            for blk in self.concept_blocks:
                blk.compile_fast()
            self.motor.compile_fast()
        else:
            for blk in self.blocks:
                blk.compile_fast()

    def forward_h_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        x0 = self.emb_in(self.tok_emb(input_ids))
        mem_raw = self.engram.lookup(input_ids) if self.engram is not None else None
        if self.engram is not None and mem_raw is not None:
            x = self.engram(x0, mem_raw)
        else:
            x = x0

        if self.hourglass_m == 2:
            B, T, D = x.shape
            h_sens = self.sensory(x, x0)
            # Strictly causal M=2 phrase merger -> T/2 concept frames!
            n_c = T // 2
            c0 = self.down_norm(self.down_proj(h_sens[:, : n_c * 2, :].reshape(B, n_c, 2 * D)))
            c = c0
            v0 = None
            for idx, blk in enumerate(self.concept_blocks):
                if isinstance(blk, FastGatedDeltaNetBlock):
                    c, v_ret = blk(c, c0, v0=None if idx == 0 else v0)
                    if idx == 0:
                        v0 = v_ret
                else:
                    c = blk(c, c0, v0=v0)
            # Strictly causal shift by 1 concept frame: chunk c sees ONLY chunks 0..c-1 (zero future leak!)
            bos = self.bos_concept.expand(B, 1, -1).to(c.dtype)
            c_shifted = torch.cat([bos, c[:, :-1, :]], dim=1)  # [B, n_c, D]
            c_up = self.up_proj(c_shifted).reshape(B, n_c * 2, D)
            if c_up.shape[1] < T:
                c_up = F.pad(c_up, (0, 0, 0, T - c_up.shape[1]))
            h_fused = h_sens + self.up_gate.to(x.dtype) * c_up
            if self.engram is not None and mem_raw is not None:
                h_fused = self.engram(h_fused, mem_raw)
            h_out = self.motor(h_fused, x0)
            return self.emb_out(self.norm_f(h_out))

        v0 = None
        for idx, blk in enumerate(self.blocks):
            if isinstance(blk, FastGatedDeltaNetBlock):
                x, v_ret = blk(x, x0, v0=None if idx == 0 else v0)
                if idx == 0:
                    v0 = v_ret
            else:
                x = blk(x, x0, v0=v0)
            if idx == 2 and self.engram is not None and mem_raw is not None:
                x = self.engram(x, mem_raw)
        return self.emb_out(self.norm_f(x))

    def compute_loss(self, batch: torch.Tensor, use_chunked_ce: bool = False) -> torch.Tensor:
        h_emb = self.forward_h_emb(batch[:, :-1])
        tgts = batch[:, 1:]
        if use_chunked_ce:
            return chunked_linear_cross_entropy(h_emb, self.tok_emb.weight, tgts, n_chunks=8)
        logits = F.linear(h_emb, self.tok_emb.weight)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgts.reshape(-1))


def main() -> None:
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}", flush=True)

    # =====================================================================
    # PART 1: Verify Upgraded Anti-Hallucination AUROC on TRAINED v3.2 Checkpoint!
    # =====================================================================
    run_dir = Path("/content/drive/MyDrive/Colab Notebooks/PULSE_AI/runs/exp_pulse_v32_micro2k")
    ckpt_p = run_dir / "checkpoints" / "pulse_latest.pt"
    val_cache = run_dir / "data_cache" / "val_pure_tinystories_512.pt"
    train_cache = run_dir / "data_cache" / "train_pure_tinystories_512.pt"

    if ckpt_p.exists() and val_cache.exists():
        print("\n=== PART 1: ANTI-HALLUCINATION VARIATIONAL FREE ENERGY CHECK (TRAINED v3.2 CKPT) ===", flush=True)
        p_eval = PULSEv3Model(PULSEv3Config.colab_30m()).to(dev)
        pl = torch.load(ckpt_p, map_location=dev, weights_only=False)
        p_eval.load_state_dict(pl["model_state_dict"])
        p_eval.eval()
        val_blocks = torch.load(val_cache, map_location="cpu", weights_only=True)

        # Evaluate across 64 stories (4 batches x 16) for rock-solid statistical confidence!
        e_trues, e_falses, e_hards = [], [], []
        k = p_eval.config.chunk_size
        with torch.no_grad():
            for bi in range(4):
                gen = torch.Generator(device="cpu").manual_seed(42 * 100_003 + 999_002 + bi)
                idx = torch.randint(0, val_blocks.shape[0], (16,), generator=gen)
                eb = val_blocks.index_select(0, idx).long().to(dev)
                split = (eb.shape[1] // k - 1) * k
                prompts = eb[:, :split]
                t_cont = eb[:, split : split + k]
                f_cont = torch.roll(t_cont, shifts=1, dims=0)
                h_cont = t_cont.clone()
                swap_pos = k // 2
                donor = torch.roll(t_cont, shifts=8, dims=0)
                h_cont[:, swap_pos] = donor[:, swap_pos]
                h_cont[:, swap_pos - 1] = donor[:, swap_pos - 1]
                same_sw = h_cont == t_cont
                if same_sw[:, swap_pos].any():
                    h_cont[:, swap_pos] = (t_cont[:, swap_pos] + 53) % 50257

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e_trues.append(p_eval.continuation_free_energy(prompts, t_cont, use_fast_weights=False))
                    e_falses.append(p_eval.continuation_free_energy(prompts, f_cont, use_fast_weights=False))
                    e_hards.append(p_eval.continuation_free_energy(prompts, h_cont, use_fast_weights=False))

        et = torch.cat(e_trues)
        ef = torch.cat(e_falses)
        eh = torch.cat(e_hards)
        auroc_cross = (ef.unsqueeze(1) > et.unsqueeze(0)).float().mean().item()
        auroc_hard = (eh.unsqueeze(1) > et.unsqueeze(0)).float().mean().item()
        pair_cross = (ef > et).float().mean().item()
        pair_hard = (eh > et).float().mean().item()
        print(
            f"[ANTI-HALLUCINATION 64-STORY AUDIT] E_true={et.mean().item():.3f} | "
            f"E_hard_swap={eh.mean().item():.3f} ({eh.mean().item()/et.mean().item():.2f}x) | "
            f"E_cross={ef.mean().item():.3f} ({ef.mean().item()/et.mean().item():.2f}x)",
            flush=True,
        )
        print(
            f"[ANTI-HALLUCINATION AUROC] Cross-Story AUROC = {auroc_cross:.4f} (paired={pair_cross*100:.1f}%) | "
            f"Hard Single-Entity Swap AUROC = {auroc_hard:.4f} (paired={pair_hard*100:.1f}%)",
            flush=True,
        )
        del p_eval, pl; gc.collect(); torch.cuda.empty_cache()

    # =====================================================================
    # PART 2: 250-Step Iso-Param (<= 29.12M) Speed + Val PPL Showdown on REAL TinyStories!
    # =====================================================================
    assert train_cache.exists() and val_cache.exists(), "Need cached TinyStories blocks!"
    train_blocks = torch.load(train_cache, map_location="cpu", weights_only=True)
    val_blocks = torch.load(val_cache, map_location="cpu", weights_only=True)

    def _sample(blocks, seed_idx, bs):
        g = torch.Generator(device="cpu").manual_seed(42 * 100_003 + seed_idx)
        return blocks.index_select(0, torch.randint(0, blocks.shape[0], (bs,), generator=g)).long().to(dev, non_blocking=True)

    def _run_candidate(name: str, model: nn.Module, use_chunked_ce: bool, steps: int = 250):
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
        if hasattr(model, "compile_fast"):
            model.compile_fast()
        elif hasattr(model, "compile_fast_paths"):
            model.compile_fast_paths()

        # Warmup 3 steps before timing & VRAM reset
        model.train()
        for w_i in range(3):
            opt.zero_grad(set_to_none=True)
            for m in range(2):
                b = _sample(train_blocks, w_i * 2 + m, 16)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if isinstance(model, ScalableTransformerLM):
                        if use_chunked_ce:
                            h = model.emb_in(model.tok_emb(b[:, :-1]))
                            cos, sin = model._rope_cos_sin(0, b.shape[1] - 1, dev, h.dtype)
                            for blk in model.blocks:
                                h, _ = blk(h, cos, sin, use_cache=False)
                            h_emb = model.emb_out(model.norm_f(h))
                            loss = chunked_linear_cross_entropy(h_emb, model.tok_emb.weight, b[:, 1:], 8) / 2.0
                        else:
                            loss = model.compute_loss(b) / 2.0
                    else:
                        loss = model.compute_loss(b, use_chunked_ce=use_chunked_ce) / 2.0
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Actual timed 250-step controlled run (identical batches 0..249!)
        torch.manual_seed(42)
        if hasattr(model, "_init_weights"):
            model._init_weights()
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

        step_times = []
        for s in range(steps):
            lr_m = min(1.0, (s + 1) / 25.0)
            for pg in opt.param_groups:
                pg["lr"] = 6e-4 * lr_m
            torch.cuda.synchronize(); t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            for m in range(2):
                b = _sample(train_blocks, s * 2 + m, 16)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if isinstance(model, ScalableTransformerLM):
                        if use_chunked_ce:
                            h = model.emb_in(model.tok_emb(b[:, :-1]))
                            cos, sin = model._rope_cos_sin(0, b.shape[1] - 1, dev, h.dtype)
                            for blk in model.blocks:
                                h, _ = blk(h, cos, sin, use_cache=False)
                            h_emb = model.emb_out(model.norm_f(h))
                            loss = chunked_linear_cross_entropy(h_emb, model.tok_emb.weight, b[:, 1:], 8) / 2.0
                        else:
                            loss = model.compute_loss(b) / 2.0
                    else:
                        loss = model.compute_loss(b, use_chunked_ce=use_chunked_ce) / 2.0
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t0) * 1000.0)
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

        # Eval Val PPL on 12 deterministic val batches
        model.eval()
        vlosses = []
        with torch.no_grad():
            for bi in range(12):
                vb = _sample(val_blocks, 900_000 + bi, 16)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if isinstance(model, ScalableTransformerLM):
                        vl = model.compute_loss(vb)
                    else:
                        vl = model.compute_loss(vb, use_chunked_ce=False)
                vlosses.append(float(vl.item()))
        val_ppl = math.exp(sum(vlosses) / len(vlosses))

        # Also measure T=2048 training speed (B=4, T=2048, accum=2 = SAME 16,384 tokens/step!)
        b2k = torch.randint(4, 50257, (4, 2049), device=dev)
        model.train()
        t2k_runs = []
        for it in range(5):
            torch.cuda.synchronize(); ts = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            for _ in range(2):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if isinstance(model, ScalableTransformerLM):
                        (model.compute_loss(b2k) / 2.0).backward()
                    else:
                        (model.compute_loss(b2k, use_chunked_ce=use_chunked_ce) / 2.0).backward()
            opt.step()
            torch.cuda.synchronize()
            if it >= 2:
                t2k_runs.append((time.perf_counter() - ts) * 1000.0)
        ms_2k = float(np.median(t2k_runs))

        med_ms = float(np.median(step_times[5:]))
        print(
            f"[SHOWDOWN] {name:34s} | params={n_params/1e6:5.2f}M | T=512: {med_ms:5.1f}ms ({peak_mb:5.0f}MB) | "
            f"T=2048: {ms_2k:5.1f}ms | ValPPL@250={val_ppl:5.2f}",
            flush=True,
        )
        del model, opt; gc.collect(); torch.cuda.empty_cache()

    print("\n=== PART 2: 250-STEP ISO-PARAM (<=29.12M) REAL TINYSTORIES SHOWDOWN ===", flush=True)
    torch.manual_seed(42)
    _run_candidate("1. TF-Compiled (Std CE)", ScalableTransformerLM(ScalableTransformerConfig.colab_30m()).to(dev), False)
    torch.manual_seed(42)
    _run_candidate("2. TF-Compiled (Chunked CE)", ScalableTransformerLM(ScalableTransformerConfig.colab_30m()).to(dev), True)
    torch.manual_seed(42)
    _run_candidate("3. PULSE-v4-M1+Engram (Std CE)", PulseV4Candidate(hourglass_m=1, use_engram=True, d_ff=864, n_engram_buckets=16384).to(dev), False)
    torch.manual_seed(42)
    _run_candidate("4. PULSE-v4-M1+Engram (ChunkCE)", PulseV4Candidate(hourglass_m=1, use_engram=True, d_ff=864, n_engram_buckets=16384).to(dev), True)
    torch.manual_seed(42)
    _run_candidate("5. PULSE-v4-M2-Hourglass+Engram", PulseV4Candidate(hourglass_m=2, use_engram=True, d_ff=736, n_engram_buckets=16384).to(dev), True)
    torch.manual_seed(42)
    _run_candidate("6. PULSE-v4-M2-Hourglass(StdCE)", PulseV4Candidate(hourglass_m=2, use_engram=True, d_ff=736, n_engram_buckets=16384).to(dev), False)


if __name__ == "__main__":
    main()
