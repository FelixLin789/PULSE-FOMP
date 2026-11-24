"""FOMP-v3: Base-Model-Agnostic First-Order Mid-Layer Fast Weights for Open-Domain Multi-Token Editing.

Upgrades FOMP-CMS v2.2 from fixed 3-token synthetic rules to open-domain variable-length natural language:
  1. Zero-Invasion Backbone Attachment: hooks cleanly onto ANY frozen causal decoder LM
     (e.g. HuggingFace `Qwen2.5-1.5B` layers [10, 12, 14] OR our own `PULSEv8Model` layers [4, 6]).
     Guarantees exact 0.0000 bit-for-bit identity whenever `g_ext == 0` or `compiled is None`.
  2. Single-Pass Variable-Length Compilation (`100%` 1st-order, zero test-time backprop):
     - Semantic cue key `k_m in R^{key_dim}` extracted from the final cue token at `read_layer`
       using a shared projector (`Norm -> key_proj -> L2Norm`), guaranteeing exact Q/K alignment.
     - Entity-invariant multi-token value bank `V_m in R^{max_tgt_tokens x val_dim}` encoded from
       standalone target phrase tokens + learned target-position embeddings.
  3. High-Concurrency Top-K Sparse Routing + Vectorized Causal Multi-Token Latching:
     - Top-K (`K=2`) sparse selection over `M` episodes (`M=1..256+`) with hard ReLU threshold
       (`sims <= tau -> strictly 0.0`) + 2-layer SiLU `rel_gate`.
     - Vectorized causal lag-shift latching over `p in [0, max_tgt_tokens)`: when prompt end `t_0`
       matches episode `m*`, generation step `t_0 + p` automatically reads `V_{m*, p}` while
       `p < len(tgt_{m*})` and shuts off cleanly to `0.0` once the multi-token phrase finishes!
  4. Built-in ICML 2026 Sign-Flip causal verification switch (`sign_flip=True`).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.pulse_v4 import _FastRMSNorm
from pulse.pulse_v7 import _mark_no_muon


@dataclass
class FOMPv3Config:
    d_model: int = 1536
    read_layer: int = 14
    inject_layers: Tuple[int, ...] = (14, 17, 21, 24, 26)
    key_dim: int = 384
    ent_dim: int = 256
    val_dim: int = 512
    max_tgt_tokens: int = 8
    top_k: int = 2
    rel_hidden: int = 128
    init_tau: float = 0.50
    min_tau: float = 0.40
    init_ent_tau: float = 0.48
    min_ent_tau: float = 0.32
    key_ctx_window: int = 12
    ent_window: int = 18
    gate_kappa: float = 0.03


def _causal_window_sum(x: torch.Tensor, win: int) -> torch.Tensor:
    """Exact causal sliding-window sum over last `min(t + 1, win)` positions: `[B, T, D] -> [B, T, D]`."""
    t_len = x.shape[1]
    csum = x.float().cumsum(dim=1)
    if t_len > win:
        return csum - F.pad(csum[:, :-win, :], (0, 0, win, 0))
    return csum


def _extract_tail_window(h_all: torch.Tensor, last_idx: torch.Tensor, win: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract last `min(last_idx[i]+1, win)` causal token states `[B, T, d] -> ([B, win, d], [B, win])`."""
    bsz, _, d_m = h_all.shape
    w_h = torch.zeros(bsz, win, d_m, device=h_all.device, dtype=h_all.dtype)
    w_m = torch.zeros(bsz, win, device=h_all.device, dtype=torch.float32)
    for i in range(bsz):
        end = int(last_idx[i].item()) + 1
        start = max(0, end - win)
        ln = end - start
        w_h[i, :ln] = h_all[i, start:end]
        w_m[i, :ln] = 1.0
    return w_h, w_m


def _causal_window_mean(h: torch.Tensor, win: int) -> torch.Tensor:
    """Exact causal sliding-window mean over last `min(t + 1, win)` positions: `[B, T, d] -> [B, T, d]`."""
    t_len = h.shape[1]
    w_sum = _causal_window_sum(h, win)
    counts = torch.arange(1, t_len + 1, device=h.device, dtype=torch.float32).clamp(max=win).view(1, t_len, 1)
    return w_sum / counts


class FOMPv3Module(nn.Module):
    """FOMP-v3.4: Base-model-agnostic 1st-order fast-weight compiler with Dual-Lock 2-layer MLP + Bridge Entity routing."""

    def __init__(self, config: FOMPv3Config) -> None:
        super().__init__()
        self.config = config
        self.n_inject = len(config.inject_layers)
        self.inject_map = {l_idx: i for i, l_idx in enumerate(config.inject_layers)}

        self.norm = _FastRMSNorm(config.d_model)
        self.ctx_norm = _FastRMSNorm(config.d_model)
        # Lock 1: Dual-view shared Q/K relation+context 2-layer residual MLP ([h_last, h_ctx_win])
        self.key_proj = nn.Linear(2 * config.d_model, config.key_dim, bias=False)
        self.key_mlp = nn.Linear(config.key_dim, config.key_dim, bias=True)
        # Lock 2: Unsupervised salience-pooled entity + latent bridge-end ([u_salient, u_last]) 2-layer residual MLP
        self.ent_score = nn.Linear(config.d_model, 1, bias=True)
        self.ent_proj = nn.Linear(2 * config.d_model, config.ent_dim, bias=False)
        self.ent_mlp = nn.Linear(config.ent_dim, config.ent_dim, bias=True)
        # 2-layer relation/edit-context gate (separates edit-answer positions from general prose & mid-answer tokens)
        self.rel_gate = nn.Sequential(
            nn.Linear(config.d_model, config.rel_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(config.rel_hidden, 1, bias=True),
        )
        # Entity-invariant multi-token value compressor + per-injection-layer low-rank expander
        self.val_down = nn.Linear(config.d_model, config.val_dim, bias=False)
        self.tgt_pos_emb = nn.Parameter(torch.zeros(config.max_tgt_tokens, config.val_dim))
        self.val_ups = nn.ModuleList([nn.Linear(config.val_dim, config.d_model, bias=False) for _ in range(self.n_inject)])
        # Full-rank unembedding residual path (preserves exact 128K/152K-vocab proper nouns zero-shot,
        # gated strictly by Lock-1/Lock-2 similarity `(top_rel, top_ent)` so it can NEVER overfit to `\nAnswer:`!)
        self.emb_diag = nn.Parameter(torch.ones(config.d_model))
        self.direct_gate = nn.Linear(2, 1, bias=True)
        self.fam_tau = nn.Parameter(torch.tensor(float(config.init_tau)))
        self.ent_tau = nn.Parameter(torch.tensor(float(config.init_ent_tau)))
        self.delta_scale = nn.Parameter(torch.full((self.n_inject,), -1.2))
        self.emb_scale = nn.Parameter(torch.linspace(-2.5, 0.8, self.n_inject))

        for m in [
            self.key_proj, self.key_mlp, self.ent_score, self.ent_proj, self.ent_mlp,
            self.rel_gate, self.direct_gate, self.val_down, *self.val_ups,
        ]:
            _mark_no_muon(m)
        for p in (self.tgt_pos_emb, self.emb_diag, self.fam_tau, self.ent_tau, self.delta_scale, self.emb_scale):
            p._no_muon = True  # type: ignore[attr-defined]
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.key_proj.weight, 0.0, 0.02)
        nn.init.normal_(self.key_mlp.weight, 0.0, 0.01)
        nn.init.zeros_(self.key_mlp.bias)
        nn.init.normal_(self.ent_score.weight, 0.0, 0.02)
        nn.init.zeros_(self.ent_score.bias)
        nn.init.normal_(self.ent_proj.weight, 0.0, 0.02)
        nn.init.normal_(self.ent_mlp.weight, 0.0, 0.01)
        nn.init.zeros_(self.ent_mlp.bias)
        nn.init.normal_(self.val_down.weight, 0.0, 0.02)
        nn.init.normal_(self.tgt_pos_emb, 0.0, 0.02)
        nn.init.normal_(self.rel_gate[0].weight, 0.0, 0.02)  # type: ignore[arg-type]
        nn.init.zeros_(self.rel_gate[0].bias)                # type: ignore[arg-type]
        nn.init.normal_(self.rel_gate[2].weight, 0.0, 0.02)  # type: ignore[arg-type]
        nn.init.constant_(self.rel_gate[2].bias, 1.0)        # type: ignore[arg-type]
        with torch.no_grad():
            self.direct_gate.weight.copy_(torch.tensor([[4.0, 1.0]]))
            self.direct_gate.bias.fill_(1.5)
        for vu in self.val_ups:
            nn.init.normal_(vu.weight, 0.0, 0.01)

    def _match_dtype(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.key_proj.weight.dtype)

    def project_key(self, h_read: torch.Tensor, h_ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        u_last = self.norm(self._match_dtype(h_read))
        u_ctx = u_last if h_ctx is None else self.ctx_norm(self._match_dtype(h_ctx))
        feat = torch.cat([u_last, u_ctx], dim=-1)
        if self.training:
            feat = F.dropout(feat, p=0.10, training=True)
        z = self.key_proj(feat)
        z = z + self.key_mlp(F.silu(z))
        return F.normalize(z.float(), p=2, dim=-1, eps=1e-6)

    def _salient_pool_window(self, h_win: torch.Tensor, win_mask: torch.Tensor) -> torch.Tensor:
        """Exact log-sum-exp stable salience softmax pooling over fixed token window `[..., W, d]` -> `[..., d]`."""
        hn = self.norm(self._match_dtype(h_win))
        s = self.ent_score(hn).float().squeeze(-1)                                   # [..., W]
        s = s.masked_fill(win_mask <= 0, -1e9)
        w = F.softmax(s, dim=-1).unsqueeze(-1)                                       # [..., W, 1]
        return (w * hn.float()).sum(dim=-2).to(self.ent_proj.weight.dtype)

    def _causal_salient_pool(self, h_seq: torch.Tensor, win: int) -> torch.Tensor:
        """Exact O(T*W) numerically stable causal sliding-window salience softmax pooling `[B, T, d]` -> `[B, T, d]`."""
        bsz, t_len, _ = h_seq.shape
        hn = self.norm(self._match_dtype(h_seq))
        s = self.ent_score(hn).float().squeeze(-1)                                   # [B, T]
        s_pad = F.pad(s, (win - 1, 0), value=-1e9)                                   # [B, T + win - 1]
        s_win = s_pad.unfold(1, win, 1)                                              # [B, T, win] (oldest..newest)
        w = F.softmax(s_win, dim=-1)                                                 # [B, T, win]
        hn_f = hn.float()
        hn_pad = F.pad(hn_f, (0, 0, win - 1, 0), value=0.0)                          # [B, T + win - 1, d]
        out = torch.zeros_like(hn_f)
        for j in range(win):
            # Offset j corresponds to lag = (win - 1 - j): aligns with w[..., j]
            out = out + w[:, :, j : j + 1] * hn_pad[:, j : j + t_len, :]
        return out.to(self.ent_proj.weight.dtype)

    def _encode_entity_feat(self, u_sal: torch.Tensor, u_last: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([u_sal, u_last], dim=-1)
        if self.training:
            feat = F.dropout(feat, p=0.10, training=True)
        z = self.ent_proj(feat)
        z = z + self.ent_mlp(F.silu(z))
        return F.normalize(z.float(), p=2, dim=-1, eps=1e-6)

    def project_entity_key_from_window(
        self, h_win: torch.Tensor, win_mask: torch.Tensor, h_last: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project salience-pooled entity + latent bridge-end `[..., W, d]` -> L2-normalized `[..., ent_dim]`."""
        u_sal = self._salient_pool_window(h_win, win_mask)
        u_last = u_sal if h_last is None else self.norm(self._match_dtype(h_last))
        return self._encode_entity_feat(u_sal, u_last)

    def project_entity_key_seq(self, h_seq: torch.Tensor) -> torch.Tensor:
        """Causal per-position entity + bridge-end vector `[B, T, d]` -> L2-normalized `[B, T, ent_dim]`."""
        u_sal = self._causal_salient_pool(h_seq, self.config.ent_window)
        u_last = self.norm(self._match_dtype(h_seq))
        return self._encode_entity_feat(u_sal, u_last)

    def _hill_gate(self, sim: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        raw_g = F.relu(sim - tau) / (1.0 - tau)
        kappa = float(self.config.gate_kappa)
        r2 = raw_g.pow(2)
        return r2 * (1.0 + kappa) / (r2 + kappa)

    def compile_from_states(
        self,
        h_cue: torch.Tensor,      # [B, M, d] or [M, d] final cue token hidden state at `read_layer`
        h_tgt: torch.Tensor,      # [B, M, Kv, d] or [M, Kv, d] standalone target phrase hidden states at `read_layer`
        tgt_mask: torch.Tensor,   # [B, M, Kv] or [M, Kv] bool/float mask (1 for valid target tokens, 0 for pad)
        g_ext: float = 1.0,       # Hard external-evidence gate in {0, 1}
        e_tgt: Optional[torch.Tensor] = None,        # [B, M, Kv, d] or [M, Kv, d] exact output-embedding vectors
        h_cue_ctx: Optional[torch.Tensor] = None,    # [B, M, d] or [M, d] causal local-window mean ending at cue end
        h_cue_win: Optional[torch.Tensor] = None,    # [B, M, W_e, d] or [M, W_e, d] last W_e cue token states
        h_cue_wmask: Optional[torch.Tensor] = None,  # [B, M, W_e] or [M, W_e] valid mask for h_cue_win
    ) -> Dict[str, torch.Tensor]:
        """Compile M variable-length multi-token facts into a Dual-Lock memory bank in 1 forward pass."""
        if h_cue.ndim == 2:
            h_cue = h_cue.unsqueeze(0)
            h_tgt = h_tgt.unsqueeze(0)
            tgt_mask = tgt_mask.unsqueeze(0)
            if e_tgt is not None:
                e_tgt = e_tgt.unsqueeze(0)
            if h_cue_ctx is not None:
                h_cue_ctx = h_cue_ctx.unsqueeze(0)
            if h_cue_win is not None:
                h_cue_win = h_cue_win.unsqueeze(0)
                assert h_cue_wmask is not None
                h_cue_wmask = h_cue_wmask.unsqueeze(0)
        kv = self.config.max_tgt_tokens
        assert h_tgt.shape[2] == kv and tgt_mask.shape[2] == kv, (
            f"Expected max_tgt_tokens={kv}, got h_tgt={tuple(h_tgt.shape)}, mask={tuple(tgt_mask.shape)}"
        )
        k_fam = self.project_key(h_cue, h_cue_ctx)                                   # [B, M, key_dim]
        if h_cue_win is not None and h_cue_wmask is not None:
            k_ent = self.project_entity_key_from_window(h_cue_win, h_cue_wmask, h_last=h_cue)
        else:
            u_fb = self.norm(self._match_dtype(h_cue))
            k_ent = self._encode_entity_feat(u_fb, u_fb)

        v_raw = self.val_down(self.norm(self._match_dtype(h_tgt))).float() + self.tgt_pos_emb.float().view(1, 1, kv, -1)
        v_rms = v_raw.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        m_f = tgt_mask.float()
        v_bank = (v_raw / v_rms) * m_f.unsqueeze(-1)                                 # [B, M, Kv, val_dim]

        e_src = h_tgt if e_tgt is None else e_tgt
        e_f = e_src.float()
        e_rms = e_f.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        e_bank = (e_f / e_rms) * self.emb_diag.float().view(1, 1, 1, -1) * m_f.unsqueeze(-1)  # [B, M, Kv, d]

        gate = float(g_ext)
        if gate == 0.0:
            v_bank = torch.zeros_like(v_bank)
            e_bank = torch.zeros_like(e_bank)
        return {
            "k_fam": k_fam,
            "k_ent": k_ent,
            "v_bank": v_bank,
            "e_bank": e_bank,
            "tgt_mask": m_f,
            "g_ext": torch.tensor(gate, device=h_cue.device, dtype=torch.float32),
        }

    def route_and_latch(
        self,
        h_read: torch.Tensor,               # [B, T, d]
        compiled: Dict[str, torch.Tensor],
        val_drop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Dual-Lock (Relation x Salient-Entity AND-gate) Top-K routing + causal multi-token latching."""
        bsz, seq_len, d_m = h_read.shape
        k_fam = compiled["k_fam"]
        k_ent = compiled["k_ent"]
        v_bank = compiled["v_bank"]
        e_bank = compiled.get("e_bank")
        tgt_mask = compiled["tgt_mask"]
        if k_fam.shape[0] == 1 and bsz > 1:
            k_fam = k_fam.expand(bsz, -1, -1)
            k_ent = k_ent.expand(bsz, -1, -1)
            v_bank = v_bank.expand(bsz, -1, -1, -1)
            if e_bank is not None:
                e_bank = e_bank.expand(bsz, -1, -1, -1)
            tgt_mask = tgt_mask.expand(bsz, -1, -1)

        h_ctx = _causal_window_mean(h_read, self.config.key_ctx_window)
        hn = self.norm(self._match_dtype(h_read))
        qf_rel = self.project_key(h_read, h_ctx)                                     # [B, T, key_dim]
        qf_ent = self.project_entity_key_seq(h_read)                                 # [B, T, ent_dim]

        sims_rel = torch.einsum("btk,bmk->btm", qf_rel, k_fam)                       # [B, T, M]
        sims_ent = torch.einsum("btk,bmk->btm", qf_ent, k_ent)                       # [B, T, M]
        sims_joint = 0.5 * (sims_rel + sims_ent)                                     # [B, T, M]

        tau_rel = self.fam_tau.float().clamp(self.config.min_tau, 0.88)
        tau_ent = self.ent_tau.float().clamp(self.config.min_ent_tau, 0.90)
        # Factorized AND-Lock: BOTH Relation lock AND Salient-Entity lock must clear their thresholds!
        g_all = self._hill_gate(sims_rel, tau_rel) * self._hill_gate(sims_ent, tau_ent)

        # Rank Top-K by Dual-Lock open strength first so locked-out neighbors can NEVER steal a slot!
        k_top = min(self.config.top_k, k_fam.shape[1])
        _, top_idx = (g_all + 1e-3 * sims_joint).topk(k=k_top, dim=-1)               # [B, T, K]
        top_rel = sims_rel.gather(-1, top_idx)                                       # [B, T, K]
        top_ent = sims_ent.gather(-1, top_idx)                                       # [B, T, K]
        top_joint = sims_joint.gather(-1, top_idx)                                   # [B, T, K]
        shaped_g = g_all.gather(-1, top_idx)                                         # [B, T, K], strictly 0 if EITHER lock fails!

        rel_logits = self.rel_gate(hn).float().squeeze(-1)                           # [B, T]
        r_gate = torch.sigmoid(rel_logits).unsqueeze(-1)                             # [B, T, 1]
        d_in = torch.stack([top_rel - 0.5, top_ent - 0.5], dim=-1).to(self.direct_gate.weight.dtype)
        d_gate = torch.sigmoid(self.direct_gate(d_in).float().squeeze(-1))           # [B, T, K]
        g_top = r_gate * shaped_g                                                    # [B, T, K]
        p_logits = (top_joint * 15.0).masked_fill(shaped_g <= 0.0, -1e4)
        p_top = F.softmax(p_logits, dim=-1) if k_top > 1 else torch.ones_like(top_joint)
        w_top = g_top * p_top * top_joint.clamp_min(0.0)                             # [B, T, K]
        w_e_top = w_top * d_gate                                                     # [B, T, K]
        top_vals = top_joint

        if float(compiled["g_ext"].item()) == 0.0:
            latched_v = torch.zeros(bsz, seq_len, self.config.val_dim, device=h_read.device, dtype=torch.float32)
            latched_e = torch.zeros(bsz, seq_len, d_m, device=h_read.device, dtype=torch.float32)
            return {
                "latched_v": latched_v,
                "latched_e": latched_e,
                "max_sim": top_vals[..., 0],
                "max_gate": g_top[..., 0],
                "d_gate": d_gate[..., 0],
                "rel_logits": rel_logits,
            }

        # Vectorized causal lag-shift latching over p = 0 .. max_tgt_tokens - 1 (100% out-of-place for clean autograd):
        # Position t reads episode triggered at (t - p) using its p-th target slot (masked to 0 when p >= L_y)!
        kv = self.config.max_tgt_tokens
        b_ar = torch.arange(bsz, device=h_read.device).view(bsz, 1, 1)
        v_parts: List[torch.Tensor] = []
        e_parts: List[torch.Tensor] = []
        for lag in range(min(kv, seq_len)):
            t_len = seq_len - lag
            idx_l = top_idx[:, :t_len, :]                                            # [B, T-lag, K]
            m_l = tgt_mask[b_ar, idx_l, lag]                                         # [B, T-lag, K]
            w_l = (w_top[:, :t_len, :] * m_l).unsqueeze(-1)                          # [B, T-lag, K, 1]
            v_l = v_bank[b_ar, idx_l, lag, :]                                        # [B, T-lag, K, val_dim]
            c_v = (w_l * v_l).sum(dim=2)                                             # [B, T-lag, val_dim]
            v_parts.append(c_v if lag == 0 else F.pad(c_v, (0, 0, lag, 0)))
            if e_bank is not None:
                we_l = (w_e_top[:, :t_len, :] * m_l).unsqueeze(-1)                   # [B, T-lag, K, 1]
                e_l = e_bank[b_ar, idx_l, lag, :]                                    # [B, T-lag, K, d]
                c_e = (we_l * e_l).sum(dim=2)                                        # [B, T-lag, d]
                e_parts.append(c_e if lag == 0 else F.pad(c_e, (0, 0, lag, 0)))
        latched_v = torch.stack(v_parts, dim=0).sum(dim=0)
        if val_drop_mask is not None:
            latched_v = latched_v * val_drop_mask.to(device=latched_v.device, dtype=latched_v.dtype).view(bsz, 1, 1)
        latched_e = (
            torch.stack(e_parts, dim=0).sum(dim=0)
            if e_parts
            else torch.zeros(bsz, seq_len, d_m, device=h_read.device, dtype=torch.float32)
        )

        return {
            "latched_v": latched_v,
            "latched_e": latched_e,
            "max_sim": top_vals[..., 0],
            "max_gate": g_top[..., 0],
            "d_gate": d_gate[..., 0],
            "rel_logits": rel_logits,
        }

    def compute_layer_delta(
        self,
        latched_v: torch.Tensor,    # [B, T, val_dim]
        inj_idx: int,
        target_dtype: torch.dtype,
        sign_flip: bool = False,
        latched_e: Optional[torch.Tensor] = None,
        h_rms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s_v = F.softplus(self.delta_scale[inj_idx].float())
        # Exact weight-RMS normalization keeps val_path O(1) across ANY d_model (1536..8192) so it can NEVER drown e_bank!
        w_up = self.val_ups[inj_idx].weight.float()
        w_up_scale = w_up.pow(2).mean().add(1e-12).sqrt() * math.sqrt(self.config.val_dim)
        delta = s_v * F.linear(latched_v.float(), w_up / w_up_scale)
        if latched_e is not None:
            s_e = F.softplus(self.emb_scale[inj_idx].float())
            delta = delta + s_e * latched_e.float()
        if h_rms is not None:
            delta = delta * h_rms.float()
        sign = -1.0 if sign_flip else 1.0
        return (sign * delta).to(target_dtype)


def _get_decoder_layers(backbone: nn.Module) -> nn.ModuleList:
    """Resolve transformer/PULSE block list across HuggingFace Qwen2/Llama/GPT and PULSEv8Model."""
    if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        return backbone.model.layers  # type: ignore[return-value]
    if hasattr(backbone, "blocks"):
        return backbone.blocks  # type: ignore[return-value]
    if hasattr(backbone, "transformer") and hasattr(backbone.transformer, "h"):
        return backbone.transformer.h  # type: ignore[return-value]
    raise ValueError(f"Unsupported backbone module hierarchy: {type(backbone).__name__}")


def _get_out_embedding_weight(backbone: nn.Module) -> torch.Tensor:
    """Resolve output unembedding matrix `[V, d]` across HuggingFace causal LMs and PULSEv8Model."""
    if hasattr(backbone, "get_output_embeddings") and backbone.get_output_embeddings() is not None:
        return backbone.get_output_embeddings().weight  # type: ignore[return-value]
    if hasattr(backbone, "lm_head") and hasattr(backbone.lm_head, "weight"):
        return backbone.lm_head.weight  # type: ignore[return-value]
    if hasattr(backbone, "get_input_embeddings") and backbone.get_input_embeddings() is not None:
        return backbone.get_input_embeddings().weight  # type: ignore[return-value]
    if hasattr(backbone, "tok_emb") and hasattr(backbone.tok_emb, "weight"):
        return backbone.tok_emb.weight  # type: ignore[return-value]
    raise ValueError(f"Cannot locate output/token embedding weight on {type(backbone).__name__}")


def _extract_logits(out: Any) -> torch.Tensor:
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    return out


class _EarlyStopSignal(Exception):
    """Internal control-flow shortcut to stop frozen backbone execution immediately at `read_layer`."""

    def __init__(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden


class FOMPv3BackboneWrapper(nn.Module):
    """Non-invasive wrapper attaching `FOMPv3Module` to ANY frozen causal LM (Qwen2.5, Llama, PULSE-v8.1)."""

    def __init__(self, backbone: nn.Module, fomp: FOMPv3Module, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.fomp = fomp
        self.layers = _get_decoder_layers(backbone)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self._last_routing: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode_early_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ONLY frozen layers `0 .. read_layer-1` and capture input hidden state to `layers[read_layer]`."""
        read_l = self.fomp.config.read_layer
        captured: Dict[str, torch.Tensor] = {}

        def _pre_hook(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            h = args[0] if len(args) > 0 else kwargs["hidden_states"]
            captured["h"] = h.detach()
            raise _EarlyStopSignal(h.detach())

        handle = self.layers[read_l].register_forward_pre_hook(_pre_hook, with_kwargs=True)
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            try:
                if attention_mask is not None and hasattr(self.backbone, "model"):
                    self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                else:
                    self.backbone(input_ids)
            except _EarlyStopSignal:
                pass
        finally:
            handle.remove()
            if was_training:
                self.backbone.train()
        assert "h" in captured, f"Failed to capture read_layer={read_l} hidden state"
        return captured["h"]

    @torch.no_grad()
    def lookup_target_embeddings(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        """Fetch frozen output-embedding directions `[..., d]` for target token IDs (0-cost indexing)."""
        w_out = _get_out_embedding_weight(self.backbone)
        return F.embedding(tgt_ids.to(w_out.device), w_out).detach()

    def compile_token_episodes(
        self,
        cue_ids_list: Sequence[Sequence[int]],
        tgt_ids_list: Sequence[Sequence[int]],
        pad_id: int,
        device: torch.device,
        g_ext: float = 1.0,
        chunk_size: int = 128,
    ) -> Dict[str, torch.Tensor]:
        """Compile M variable-length `(cue_ids, tgt_ids)` episodes into a single `[1, M, ...]` memory bank (O(1) VRAM)."""
        m_cnt = len(cue_ids_list)
        assert m_cnt > 0 and len(tgt_ids_list) == m_cnt
        kv = self.fomp.config.max_tgt_tokens
        win_c = self.fomp.config.key_ctx_window
        win_e = self.fomp.config.ent_window

        # 1. Chunked encoding so M=1024+ compiles cleanly in constant VRAM without global max-len padding waste
        hc_parts, hctx_parts, hwin_parts, hwm_parts = [], [], [], []
        ht_parts, et_parts, tm_parts = [], [], []
        for s0 in range(0, m_cnt, chunk_size):
            c_sub = cue_ids_list[s0 : s0 + chunk_size]
            t_sub = tgt_ids_list[s0 : s0 + chunk_size]
            n_sub = len(c_sub)

            max_c = max(len(c) for c in c_sub)
            c_mat = torch.full((n_sub, max_c), pad_id, device=device, dtype=torch.long)
            c_mask = torch.zeros((n_sub, max_c), device=device, dtype=torch.long)
            c_last = torch.zeros((n_sub,), device=device, dtype=torch.long)
            for i, c in enumerate(c_sub):
                ln = len(c)
                c_mat[i, :ln] = torch.tensor(c, device=device, dtype=torch.long)
                c_mask[i, :ln] = 1
                c_last[i] = ln - 1
            h_c_all = self.encode_early_batch(c_mat, attention_mask=c_mask)
            h_c_ctx_all = _causal_window_mean(h_c_all, win_c)
            ar_s = torch.arange(n_sub, device=device)
            hc_parts.append(h_c_all[ar_s, c_last])
            hctx_parts.append(h_c_ctx_all[ar_s, c_last])
            hw_s, hwm_s = _extract_tail_window(h_c_all, c_last, win_e)
            hwin_parts.append(hw_s)
            hwm_parts.append(hwm_s)

            t_mat = torch.full((n_sub, kv), pad_id, device=device, dtype=torch.long)
            t_mask = torch.zeros((n_sub, kv), device=device, dtype=torch.long)
            for i, t in enumerate(t_sub):
                ln = min(kv, max(1, len(t)))
                t_mat[i, :ln] = torch.tensor(list(t[:ln]), device=device, dtype=torch.long)
                t_mask[i, :ln] = 1
            ht_parts.append(self.encode_early_batch(t_mat, attention_mask=t_mask))
            et_parts.append(self.lookup_target_embeddings(t_mat))
            tm_parts.append(t_mask)

        return self.fomp.compile_from_states(
            torch.cat(hc_parts, dim=0),
            torch.cat(ht_parts, dim=0),
            torch.cat(tm_parts, dim=0),
            g_ext=g_ext,
            e_tgt=torch.cat(et_parts, dim=0),
            h_cue_ctx=torch.cat(hctx_parts, dim=0),
            h_cue_win=torch.cat(hwin_parts, dim=0),
            h_cue_wmask=torch.cat(hwm_parts, dim=0),
        )

    @contextmanager
    def _attach_hooks(
        self,
        compiled: Optional[Dict[str, torch.Tensor]],
        sign_flip: bool = False,
        val_drop_mask: Optional[torch.Tensor] = None,
    ) -> Iterator[None]:
        if compiled is None or float(compiled["g_ext"].item()) == 0.0:
            self._last_routing = {}
            yield
            return

        handles = []
        state: Dict[str, Any] = {}
        read_l = self.fomp.config.read_layer
        inj_map = self.fomp.inject_map

        for l_idx in sorted(set([read_l, *self.fomp.config.inject_layers])):
            def _make_pre(layer_i: int):
                def _hook(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                    use_kw = len(args) == 0
                    h = kwargs["hidden_states"] if use_kw else args[0]
                    if layer_i == read_l:
                        rt = self.fomp.route_and_latch(h, compiled, val_drop_mask=val_drop_mask)
                        state["rt"] = rt
                        self._last_routing = rt
                    if layer_i in inj_map:
                        h_rms = h.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().detach()
                        delta = self.fomp.compute_layer_delta(
                            state["rt"]["latched_v"],
                            inj_map[layer_i],
                            h.dtype,
                            sign_flip=sign_flip,
                            latched_e=state["rt"]["latched_e"],
                            h_rms=h_rms,
                        )
                        h_new = h + delta
                        if use_kw:
                            new_kw = dict(kwargs)
                            new_kw["hidden_states"] = h_new
                            return args, new_kw
                        return (h_new, *args[1:]), kwargs
                    return args, kwargs
                return _hook

            handles.append(self.layers[l_idx].register_forward_pre_hook(_make_pre(l_idx), with_kwargs=True))

        try:
            yield
        finally:
            for hd in handles:
                hd.remove()

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compiled: Optional[Dict[str, torch.Tensor]] = None,
        sign_flip: bool = False,
        val_drop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        with self._attach_hooks(compiled=compiled, sign_flip=sign_flip, val_drop_mask=val_drop_mask):
            if attention_mask is not None and hasattr(self.backbone, "model"):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                out = self.backbone(input_ids)
        logits = _extract_logits(out)
        return {"logits": logits, "routing": self._last_routing}

    @torch.no_grad()
    def greedy_generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        device: torch.device,
        compiled: Optional[Dict[str, torch.Tensor]] = None,
        sign_flip: bool = False,
        eos_id: Optional[int] = None,
    ) -> List[int]:
        """Exact multi-token greedy decode with causal FOMP-v3 latching across generated tokens."""
        cur = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
        gen: List[int] = []
        for _ in range(max_new_tokens):
            lg = self.forward_logits(cur, compiled=compiled, sign_flip=sign_flip)["logits"][0, -1]
            nxt = int(lg.argmax(dim=-1).item())
            gen.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        return gen
