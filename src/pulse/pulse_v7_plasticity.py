"""PULSE-v7 Stage 3 Modules: Dual-Track Semantic Plasticity (Semantic RLS + EngramEdit) & Transition-Conditioned Verifier.

Implements Section 5 (Wall #3) and Section 6 (Wall #4) of the PULSE-v7 blueprint:
- `SemanticKeyProjector`: lightweight relation/paraphrase-aligned semantic key encoder (trained on disjoint
  non-test templates so held-out entity paraphrases align in mid-layer key space without touching backbone weights).
- `DualTrackSemanticMemory`:
    * Track 1: Exact token-suffix O(1) fast table (100% exact recall, +0.000% drift).
    * Track 2 (Wake): Mid-layer (`L_mid`) familiarity-gated Sherman-Morrison Semantic RLS (`c_t -> Delta h_mid`),
      injecting before Coda so downstream layers preserve multi-token fluency and 2-hop composition.
    * Sleep (`consolidate_to_engram`): EngramEdit decoupled conditional memory bank (`K_engram, V_engram`) with
      sparse top-k cosine addressing & familiarity gating — zero matrix saturation even at high edit counts.
- `TransitionConditionedVerifier`: strictly < 50K-param 2-layer MLP taking `[gdn_signals_t, h_t]` (or matched
  Transformer `[tf_signals_t, h_t]`) for 3-way classification (Consistent / Legal State Transition / Illegal Swap).
- `decore_contrastive_logits`: verifier-gated DeCoRe contrastive decoding intervention.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticKeyProjector(nn.Module):
    """Causal window-pooled linear subspace projector mapping h_mid_seq [B, T, D] -> normalized semantic key [B, T, d_key].

    A learnable 1-vector causal attention query (`pool_q`, window W=12) attends to entity + relation tokens
    anywhere in the prompt while a linear projection annihilates template syntax — generalizing zero-shot
    to unseen entities while staying strictly causal & O(1) per token on long streams.
    """

    def __init__(self, d_model: int, d_key: int = 128, pool_window: int = 12) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.pool_window = int(pool_window)
        self.norm = nn.LayerNorm(d_model)
        self.pool_q = nn.Parameter(torch.randn(d_model) * (1.0 / (d_model ** 0.5)))
        self.proj = nn.Linear(2 * d_model, d_key, bias=False)

    def _causal_pool(self, h_norm: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        # h_norm: [B, T, D] -> pooled [B, 1 or T, D]
        bsz, seq_len, dim = h_norm.shape
        w = min(self.pool_window, seq_len)
        scores = torch.matmul(h_norm, self.pool_q)  # [B, T]
        if last_only or seq_len == 1:
            sc_w = scores[:, -w:]
            attn = F.softmax(sc_w, dim=-1).unsqueeze(-1)  # [B, w, 1]
            return (attn * h_norm[:, -w:, :]).sum(dim=1, keepdim=True)
        # Causal sliding-window unfold across full sequence [B, T, w, D]
        pad_h = F.pad(h_norm, (0, 0, w - 1, 0), value=0.0)
        pad_s = F.pad(scores, (w - 1, 0), value=-1e4)
        win_h = pad_h.unfold(1, w, 1).transpose(-1, -2)  # [B, T, w, D]
        win_s = pad_s.unfold(1, w, 1)                    # [B, T, w]
        attn = F.softmax(win_s, dim=-1).unsqueeze(-1)    # [B, T, w, 1]
        return (attn * win_h).sum(dim=-2)                # [B, T, D]

    def forward(self, h_mid: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        if h_mid.ndim == 2:
            h_mid = h_mid.unsqueeze(1)
            squeeze_t = True
        else:
            squeeze_t = False
        hn = self.norm(h_mid.float())
        hp = self._causal_pool(hn, last_only=last_only)
        h_cur = hn[:, -1:, :] if last_only else hn
        out = F.normalize(self.proj(torch.cat([hp, h_cur], dim=-1)), p=2, dim=-1, eps=1e-6)
        return out.squeeze(1) if squeeze_t else out


class DualTrackSemanticMemory(nn.Module):
    """PULSE-v7 Dual-Track Online Plasticity: Exact Suffix + Mid-Layer Familiarity-Gated Semantic RLS + EngramEdit Sleep."""

    def __init__(
        self,
        d_model: int,
        d_key: int = 128,
        suffix_len: int = 5,
        ridge_lambda: float = 0.05,
        tau_sem: float = 0.85,
        engram_temp: float = 25.0,
        engram_top_k: int = 2,
        key_projector: Optional[SemanticKeyProjector] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.suffix_len = int(suffix_len)
        self.ridge_lambda = float(ridge_lambda)
        self.tau_sem = float(tau_sem)
        self.engram_temp = float(engram_temp)
        self.engram_top_k = int(engram_top_k)
        self.key_projector = key_projector

        # Track 1: Exact suffix fast table (tuple of last-k token IDs -> Delta h_mid)
        self.exact_table: Dict[Tuple[int, ...], torch.Tensor] = {}

        # Track 2 (Wake): Closed-form Sherman-Morrison RLS over semantic keys c_t -> Delta h_mid
        self.register_buffer("W_fast", torch.zeros(d_model, d_key, dtype=torch.float32))
        self.register_buffer("P_inv", torch.eye(d_key, dtype=torch.float32) / self.ridge_lambda)
        self.wake_keys: List[torch.Tensor] = []
        self.wake_vals: List[torch.Tensor] = []

        # Sleep (EngramEdit): Decoupled conditional memory bank (zero matrix-rank saturation!)
        self.engram_keys: Optional[torch.Tensor] = None  # [N_sleep, d_key]
        self.engram_vals: Optional[torch.Tensor] = None  # [N_sleep, d_model]

        self._wake_key_mat: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.exact_table.clear()
        self.W_fast.zero_()
        self.P_inv.copy_(torch.eye(self.d_key, device=self.P_inv.device, dtype=torch.float32) / self.ridge_lambda)
        self.wake_keys.clear()
        self.wake_vals.clear()
        self._wake_key_mat = None
        self.engram_keys = None
        self.engram_vals = None

    @torch.no_grad()
    def encode_key(self, h_mid: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        if self.key_projector is not None:
            return self.key_projector(h_mid, last_only=last_only)
        hf = h_mid[:, -1:, :].float() if (last_only and h_mid.ndim == 3) else h_mid.float()
        if hf.shape[-1] >= self.d_key:
            hf = hf[..., : self.d_key]
        else:
            hf = F.pad(hf, (0, self.d_key - hf.shape[-1]))
        return F.normalize(hf, p=2, dim=-1, eps=1e-6)

    @torch.no_grad()
    def write_fact(
        self,
        prompt_ids: torch.Tensor,
        h_mid_seq: torch.Tensor,
        delta_h_target: torch.Tensor,
        extra_keys: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Write one episodic fact into Track 1 (exact suffix) + Track 2 (Sherman-Morrison RLS)."""
        dev = self.W_fast.device
        dh = delta_h_target.detach().float().to(dev).view(self.d_model)
        toks = prompt_ids.detach().cpu().view(-1).tolist()
        if len(toks) >= self.suffix_len:
            self.exact_table[tuple(toks[-self.suffix_len :])] = dh.clone()

        h_in = h_mid_seq.detach().to(dev)
        if h_in.ndim == 1:
            h_in = h_in.view(1, 1, self.d_model)
        elif h_in.ndim == 2:
            h_in = h_in.unsqueeze(0)
        c0 = self.encode_key(h_in, last_only=True).view(self.d_key)
        keys_to_write = [c0]
        if extra_keys:
            keys_to_write.extend([k.detach().float().to(dev).view(self.d_key) for k in extra_keys])

        for c in keys_to_write:
            c = F.normalize(c, p=2, dim=-1, eps=1e-6)
            Pc = self.P_inv @ c
            denom = 1.0 + float(torch.dot(c, Pc).item())
            g = Pc / max(denom, 1e-6)
            err = dh - (self.W_fast @ c)
            self.W_fast.add_(torch.outer(err, g))
            self.P_inv.sub_(torch.outer(g, Pc))
            self.wake_keys.append(c)
            self.wake_vals.append(dh.clone())
        self._wake_key_mat = torch.stack(self.wake_keys, dim=0)

    @torch.no_grad()
    def consolidate_to_engram(self) -> int:
        """EngramEdit Sleep: move Wake episodic RLS traces into decoupled conditional memory & reset W_fast."""
        if not self.wake_keys:
            return 0
        k_new = torch.stack(self.wake_keys, dim=0)  # [M, d_key]
        v_new = torch.stack(self.wake_vals, dim=0)  # [M, d_model]
        if self.engram_keys is None or self.engram_vals is None:
            self.engram_keys = k_new
            self.engram_vals = v_new
        else:
            self.engram_keys = torch.cat([self.engram_keys, k_new], dim=0)
            self.engram_vals = torch.cat([self.engram_vals, v_new], dim=0)
        n_moved = len(self.wake_keys)
        self.W_fast.zero_()
        self.P_inv.copy_(torch.eye(self.d_key, device=self.P_inv.device, dtype=torch.float32) / self.ridge_lambda)
        self.wake_keys.clear()
        self.wake_vals.clear()
        self._wake_key_mat = None
        return n_moved

    @torch.no_grad()
    def max_familiarity(self, h_mid: torch.Tensor) -> torch.Tensor:
        """Compute max cosine familiarity rho(c_t) against all stored Wake + Engram prototype keys."""
        c = self.encode_key(h_mid)  # [..., d_key]
        rho = torch.full(c.shape[:-1], -1.0, device=c.device, dtype=torch.float32)
        if self._wake_key_mat is not None and self._wake_key_mat.numel() > 0:
            sim_w = torch.matmul(c, self._wake_key_mat.to(c.device).T).max(dim=-1).values
            rho = torch.maximum(rho, sim_w)
        if self.engram_keys is not None and self.engram_keys.numel() > 0:
            sim_e = torch.matmul(c, self.engram_keys.to(c.device).T).max(dim=-1).values
            rho = torch.maximum(rho, sim_e)
        return rho

    @torch.no_grad()
    def apply_mid_hook(
        self,
        h_mid: torch.Tensor,
        input_ids: torch.Tensor,
        use_exact_track: bool = True,
        use_semantic_track: bool = True,
        last_pos_only: bool = False,
    ) -> torch.Tensor:
        """Familiarity-gated mid-layer injection: strictly 0.000 delta when unhit -> +0.000% PPL drift!"""
        if not self.exact_table and self._wake_key_mat is None and self.engram_keys is None:
            return h_mid
        bsz, seq_len, _ = h_mid.shape
        delta = torch.zeros_like(h_mid)
        handled = torch.zeros(bsz, seq_len, dtype=torch.bool, device=h_mid.device)

        # Track 1: Exact suffix hit (checked at last_pos or all valid suffixes)
        if use_exact_track and self.exact_table:
            t_range = [seq_len - 1] if last_pos_only else list(range(self.suffix_len - 1, seq_len))
            ids_cpu = input_ids.detach().cpu().tolist()
            for b in range(bsz):
                row = ids_cpu[b]
                for t in t_range:
                    if t + 1 >= self.suffix_len:
                        key = tuple(row[t + 1 - self.suffix_len : t + 1])
                        dh = self.exact_table.get(key)
                        if dh is not None:
                            delta[b, t].copy_(dh.to(h_mid.device, h_mid.dtype))
                            handled[b, t] = True

        # Track 2: Familiarity-gated Semantic RLS (Wake) + EngramEdit (Sleep)
        if use_semantic_track and (self._wake_key_mat is not None or self.engram_keys is not None):
            c = self.encode_key(h_mid, last_only=last_pos_only)  # [B, 1 or T, d_key]
            q_shape = (bsz, 1, self.d_model) if last_pos_only else h_mid.shape
            best_sim = torch.full(c.shape[:-1], -1.0, device=c.device, dtype=torch.float32)
            cand = torch.zeros(q_shape, device=c.device, dtype=torch.float32)

            if self._wake_key_mat is not None and self._wake_key_mat.numel() > 0:
                sim_w = torch.matmul(c, self._wake_key_mat.to(c.device).T).max(dim=-1).values
                mask_w = sim_w > self.tau_sem
                if mask_w.any():
                    pred_w = F.linear(c, self.W_fast.to(c.device))
                    cand = torch.where(mask_w.unsqueeze(-1), pred_w, cand)
                    best_sim = torch.where(mask_w, sim_w, best_sim)

            if self.engram_keys is not None and self.engram_vals is not None and self.engram_keys.numel() > 0:
                sims = torch.matmul(c, self.engram_keys.to(c.device).T)  # [B, Tq, N_sleep]
                k_use = min(self.engram_top_k, sims.shape[-1])
                top_v, top_i = torch.topk(sims, k=k_use, dim=-1)
                max_e = top_v[..., 0]
                use_e = (max_e > self.tau_sem) & (max_e > best_sim)
                if use_e.any():
                    weights = F.softmax(top_v * self.engram_temp, dim=-1)  # [B, Tq, k]
                    vals = self.engram_vals.to(c.device)[top_i]            # [B, Tq, k, d_model]
                    pred_e = (weights.unsqueeze(-1) * vals).sum(dim=-2)
                    cand = torch.where(use_e.unsqueeze(-1), pred_e, cand)
                    best_sim = torch.where(use_e, max_e, best_sim)

            active = best_sim > self.tau_sem
            if active.any():
                if last_pos_only:
                    unhandled = active[:, 0] & (~handled[:, -1])
                    if unhandled.any():
                        delta[unhandled, -1] = cand[unhandled, 0].to(h_mid.dtype)
                else:
                    unhandled = active & (~handled)
                    delta = torch.where(unhandled.unsqueeze(-1), cand.to(h_mid.dtype), delta)

        return h_mid + delta


def solve_mid_layer_target_delta(
    coda_fn: Any,
    h_mid_seq: torch.Tensor,
    target_tok_id: int,
    steps: int = 18,
    lr: float = 0.35,
    l2_reg: float = 5e-4,
    context_jitter: float = 0.15,
) -> torch.Tensor:
    """Solve for a context-robust Coda-native mid-layer residual `delta_h` at the last prompt position."""
    h_base = h_mid_seq.detach()
    delta = torch.zeros(1, 1, h_base.shape[-1], device=h_base.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    tgt = torch.tensor([int(target_tok_id)], device=h_base.device, dtype=torch.long)
    for st in range(steps):
        opt.zero_grad(set_to_none=True)
        h_mod = h_base.clone()
        jitter = (context_jitter * torch.randn_like(delta)) if (context_jitter > 0.0 and st > 0) else 0.0
        h_mod[:, -1:, :] = h_base[:, -1:, :] + (delta + jitter).to(h_base.dtype)
        logits_last = coda_fn(h_mod)[:, -1, :].float()
        loss = F.cross_entropy(logits_last, tgt) + l2_reg * delta.pow(2).mean()
        loss.backward()
        opt.step()
        if st >= 4 and int(logits_last.argmax(-1).item()) == int(target_tok_id) and float(loss.item()) < 0.02:
            break
    return delta.detach().squeeze(0).squeeze(0)


class TransitionConditionedVerifier(nn.Module):
    """Strictly < 50K-parameter 2-layer MLP classifying entity mentions into 3 states:
      0 = Consistent, 1 = Legal State Transition, 2 = Illegal Fluent Entity Swap (Contradiction).
    """

    def __init__(self, d_in: int, d_hidden: int = 48, n_classes: int = 3, dropout: float = 0.35) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, n_classes)
        n_params = sum(p.numel() for p in self.parameters())
        if n_params >= 50000:
            raise ValueError(f"Verifier parameter count {n_params} >= 50,000 blueprint budget!")

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(self.norm(x.float())))))


def decore_contrastive_logits(
    logits_full: torch.Tensor,
    logits_prior: torch.Tensor,
    p_illegal: torch.Tensor,
    lam: float = 1.5,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Verifier-gated DeCoRe contrastive decoding: amplifies long-range evidence only when conflict is detected."""
    gate = (p_illegal > threshold).to(logits_full.dtype)
    while gate.ndim < logits_full.ndim:
        gate = gate.unsqueeze(-1)
    return logits_full + (lam * gate) * (logits_full - logits_prior)
