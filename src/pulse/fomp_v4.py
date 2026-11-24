"""FOMP-v4: Lifting the Two-Hop Curse via Read-Equivalent Bridge Plasticity & Gated Virtual-KV Memory.

Extends `FOMP-v3.6` (`src/pulse/fomp_v3.py`) with a 3-stage laminar circuit trained strictly on single-hop
CounterFact (`0%` MQuAKE and `0%` ZsRE training data):
  1. Stage A — Early Mid-Sentence Span Dual-Lock (`bridge_read_layer`, ~0.31 L) + Read-Equivalent Entity
     Replacement (`bridge_inject_layers`, e.g. L10, L13, L16, L19 on 32L Llama-3):
       Detects Hop-1 edited clauses embedded inside multi-hop questions (when outer relation != inner relation)
       and performs gated convex replacement `h_t <- (1 - a) h_t + a * h_entity(target_new) + delta_adapt`,
       allowing frozen downstream layers (L14..L31) to natively compute Hop-2 (`Route 1: Legible`).
  2. Stage B — Mid-Layer Direct Dual-Lock (`read_layer`, ~0.50 L, inherited verbatim from `FOMP-v3.6`):
       Handles all single-hop direct, paraphrase, and zero-shot ZsRE queries with weight-RMS-normalized
       `val_ups`, full-rank `e_bank`, and causal multi-token lag-shift latching.
  3. Stage C — Late Resolved-Bridge Entity Lock (`latent_read_layer`, ~0.625 L -> L23, L27, L31):
       Reads the frozen backbone's implicitly resolved Hop-1 entity representation and matches it into
       `direct.k_ent` space (self-supervised on `cf_train` `cue -> target_true` pairs, `0%` MQuAKE!) to catch
       Hop-2 edited multi-hop queries.
  4. Optional Pre-Compiled Virtual-KV Readout (`Route 2: Bridge-KV`, `mode="bridge_kv"`):
       Uses the exact same neural Stage A/B/C routers (`0` KV overhead on all single-hop & unrelated locality
       queries!), mounting pre-compiled RoPE-aligned `DynamicCache` KV tensors strictly when a multi-hop
       bridge gate (`g_span` or `g_latent`) fires while `g_direct == 0`.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.fomp_v3 import (
    FOMPv3Config,
    FOMPv3Module,
    _EarlyStopSignal,
    _causal_window_mean,
    _extract_logits,
    _extract_tail_window,
    _get_decoder_layers,
    _get_out_embedding_weight,
)
from pulse.pulse_v4 import _FastRMSNorm
from pulse.pulse_v7 import _mark_no_muon


@dataclass
class FOMPv4Config:
    d_model: int = 4096
    bridge_read_layer: int = 10
    bridge_inject_layers: Tuple[int, ...] = (10, 13, 16, 19)
    read_layer: int = 16
    inject_layers: Tuple[int, ...] = (16, 19, 23, 27, 31)
    latent_read_layer: int = 20
    key_dim: int = 512
    ent_dim: int = 384
    val_dim: int = 512
    bridge_rank: int = 256
    max_tgt_tokens: int = 8
    top_k: int = 2
    rel_hidden: int = 128
    span_window: int = 14
    init_span_rel_tau: float = 0.54
    min_span_rel_tau: float = 0.46
    init_span_ent_tau: float = 0.54
    min_span_ent_tau: float = 0.46
    init_lat_rel_tau: float = 0.46
    min_lat_rel_tau: float = 0.40
    init_lat_ent_tau: float = 0.44
    min_lat_ent_tau: float = 0.36
    init_outer_direct_tau: float = 0.36
    min_outer_direct_tau: float = 0.24
    gate_kappa: float = 0.03

    @classmethod
    def for_backbone(cls, d_model: int, n_layers: int) -> "FOMPv4Config":
        rl = max(2, int(round(n_layers * 0.50)))
        inj = tuple(sorted(set([rl, min(n_layers - 1, rl + 3), min(n_layers - 1, rl + 7), max(rl, n_layers - 5), max(rl, n_layers - 1)])))
        brl = max(1, int(round(n_layers * 0.31)))
        step_b = max(1, int(round(n_layers * 0.095)))
        binj = tuple(sorted(set([
            brl,
            min(n_layers - 2, brl + step_b),
            min(n_layers - 2, rl),
            min(n_layers - 2, rl + step_b),
        ])))
        lrl = min(n_layers - 2, max(rl + 1, int(round(n_layers * 0.625))))
        return cls(
            d_model=d_model,
            bridge_read_layer=brl,
            bridge_inject_layers=binj,
            read_layer=rl,
            inject_layers=inj,
            latent_read_layer=lrl,
            key_dim=(512 if d_model >= 3072 else 384),
            ent_dim=(384 if d_model >= 3072 else 256),
            val_dim=512,
            bridge_rank=(256 if d_model >= 3072 else 128),
        )

    def to_v3_config(self) -> FOMPv3Config:
        return FOMPv3Config(
            d_model=self.d_model,
            read_layer=self.read_layer,
            inject_layers=self.inject_layers,
            key_dim=self.key_dim,
            ent_dim=self.ent_dim,
            val_dim=self.val_dim,
            max_tgt_tokens=self.max_tgt_tokens,
            top_k=self.top_k,
            rel_hidden=self.rel_hidden,
            gate_kappa=self.gate_kappa,
        )


def strip_cue_copula(cue: str) -> str:
    """Strip trailing copula/auxiliary tokens so cue matches embedded noun-phrase bridges (`0` dataset heuristics)."""
    t = cue.strip()
    if t.lower().startswith("question:"):
        t = t[len("question:") :].strip()
    for suf in ("\nanswer:", "answer:"):
        if t.lower().endswith(suf):
            t = t[: -len(suf)].strip()
    t = t.rstrip("?").strip()
    toks = t.split()
    while len(toks) > 2 and toks[-1].lower() in {"is", "was", "are", "were", "be"}:
        toks.pop()
    return " ".join(toks)


def format_virtual_kv_fact(cue: str, target_new: str) -> str:
    """Format a concise factual statement for pre-compiled Virtual-KV caching."""
    c = cue.strip()
    y = target_new.strip()
    if c.lower().startswith("question:"):
        q = c[len("question:") :].strip()
        for suf in ("\nanswer:", "answer:"):
            if q.lower().endswith(suf):
                q = q[: -len(suf)].strip()
        if not q.endswith("?"):
            q += "?"
        return f"- {q} {y}."
    return f"- {c} {y}."


class FOMPv4Module(nn.Module):
    """3-Stage Laminar Plasticity Module combining Direct FOMP-v3.6, Legible Bridge Swap, and Latent Entity Routing."""

    def __init__(self, config: FOMPv4Config) -> None:
        super().__init__()
        self.config = config
        self.direct = FOMPv3Module(config.to_v3_config())

        self.n_bridge_inj = len(config.bridge_inject_layers)
        self.bridge_inj_map = {l_idx: i for i, l_idx in enumerate(config.bridge_inject_layers)}
        self.late_inj_indices = tuple(i for i, l_idx in enumerate(config.inject_layers) if l_idx >= config.latent_read_layer)

        # --- Stage A: Early Mid-Sentence Span Dual-Lock (Salience-Pooled Relation x Salience-Pooled Entity) ---
        self.span_norm = _FastRMSNorm(config.d_model)
        self.span_rel_score = nn.Linear(config.d_model, 1, bias=True)
        self.span_rel_proj = nn.Linear(2 * config.d_model, config.key_dim, bias=False)
        self.span_rel_mlp = nn.Linear(config.key_dim, config.key_dim, bias=True)

        self.span_ent_score = nn.Linear(config.d_model, 1, bias=True)
        self.span_ent_proj = nn.Linear(2 * config.d_model, config.ent_dim, bias=False)
        self.span_ent_mlp = nn.Linear(config.ent_dim, config.ent_dim, bias=True)

        self.span_rel_tau = nn.Parameter(torch.tensor(float(config.init_span_rel_tau)))
        self.span_ent_tau = nn.Parameter(torch.tensor(float(config.init_span_ent_tau)))

        # Read-Equivalent Bridge Replacement Adapter across `bridge_inject_layers`:
        # Zero-init `bridge_ups` so at step 0 it performs pure native entity-state convex replacement!
        self.bridge_down = nn.Linear(2 * config.d_model, config.bridge_rank, bias=False)
        self.bridge_ups = nn.ModuleList([nn.Linear(config.bridge_rank, config.d_model, bias=False) for _ in range(self.n_bridge_inj)])
        self.bridge_alpha_logit = nn.Parameter(torch.full((self.n_bridge_inj,), 1.4))  # sigmoid(1.4) ~= 0.80
        # Spread weights for `[t* - 2, t* - 1, t*]` so multi-subword bridge heads get clean entity erasure+write
        self.bridge_tok_spread = nn.Parameter(torch.tensor([-1.8, -0.6, 2.5]))

        # --- Stage C: Late Implicitly-Resolved Bridge Entity Projector (`latent_read_layer` -> `direct.k_ent` space) ---
        self.lat_norm = _FastRMSNorm(config.d_model)
        self.lat_tok_score = nn.Linear(config.d_model, 1, bias=True)
        self.lat_ent_proj = nn.Linear(2 * config.d_model, config.ent_dim, bias=False)
        self.lat_ent_mlp = nn.Linear(config.ent_dim, config.ent_dim, bias=True)
        self.lat_rel_tau = nn.Parameter(torch.tensor(float(config.init_lat_rel_tau)))
        self.lat_ent_tau = nn.Parameter(torch.tensor(float(config.init_lat_ent_tau)))
        self.lat_boost = nn.Parameter(torch.tensor(0.0))

        # --- Full-Prompt Outer-Relation Projector (`read_layer` -> `direct.k_fam` space) ---
        # Resolves 12-token tail blindness on nested-NP queries ("What is the X of the Y of Z?"):
        # (1) Arbitrates Stage B vs Stage A (`q_outer_rel . k_fam[m_span] < outer_direct_tau` => 2-hop bridge, suppress Direct!)
        # (2) Supplies prefix outer-relation matching to Stage C (`route_latent_hop2`).
        self.outer_norm = _FastRMSNorm(config.d_model)
        self.outer_tok_score = nn.Linear(config.d_model, 1, bias=True)
        self.outer_rel_proj = nn.Linear(3 * config.d_model, config.key_dim, bias=False)
        self.outer_rel_mlp = nn.Linear(config.key_dim, config.key_dim, bias=True)
        self.outer_direct_tau = nn.Parameter(torch.tensor(float(config.init_outer_direct_tau)))

        for m in [
            self.span_rel_score, self.span_rel_proj, self.span_rel_mlp,
            self.span_ent_score, self.span_ent_proj, self.span_ent_mlp,
            self.bridge_down, *self.bridge_ups,
            self.lat_tok_score, self.lat_ent_proj, self.lat_ent_mlp,
            self.outer_tok_score, self.outer_rel_proj, self.outer_rel_mlp,
        ]:
            _mark_no_muon(m)
        for p in (
            self.span_rel_tau, self.span_ent_tau, self.bridge_alpha_logit,
            self.bridge_tok_spread, self.lat_rel_tau, self.lat_ent_tau, self.lat_boost,
            self.outer_direct_tau,
        ):
            p._no_muon = True  # type: ignore[attr-defined]
        self._init_bridge_weights()

    def _init_bridge_weights(self) -> None:
        for lin in (self.span_rel_proj, self.span_ent_proj, self.lat_ent_proj, self.outer_rel_proj, self.bridge_down):
            nn.init.normal_(lin.weight, 0.0, 0.02)
        for mlp in (self.span_rel_mlp, self.span_ent_mlp, self.lat_ent_mlp):
            nn.init.normal_(mlp.weight, 0.0, 0.01)
            nn.init.zeros_(mlp.bias)
        # Zero-init `outer_rel_mlp` so at step 0 `q_outer_rel == direct.project_key` (100% OOD single-hop preservation!)
        nn.init.zeros_(self.outer_rel_mlp.weight)
        nn.init.zeros_(self.outer_rel_mlp.bias)
        for sc in (self.span_rel_score, self.span_ent_score, self.lat_tok_score, self.outer_tok_score):
            nn.init.normal_(sc.weight, 0.0, 0.02)
            nn.init.zeros_(sc.bias)
        for bu in self.bridge_ups:
            nn.init.zeros_(bu.weight)

    def bridge_parameters(self) -> List[nn.Parameter]:
        """Return strictly the new Stage A & Stage C parameters (allows freezing `self.direct` cleanly)."""
        direct_ids = {id(p) for p in self.direct.parameters()}
        return [p for p in self.parameters() if id(p) not in direct_ids]

    def _hill(self, sim: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        raw_g = F.relu(sim - tau) / (1.0 - tau)
        kappa = float(self.config.gate_kappa)
        r2 = raw_g.pow(2)
        return r2 * (1.0 + kappa) / (r2 + kappa)

    def _pool_window(self, h_win: torch.Tensor, win_mask: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
        hn = self.span_norm(h_win.to(scorer.weight.dtype))
        s = scorer(hn).float().squeeze(-1).masked_fill(win_mask <= 0, -1e9)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        return (w * hn.float()).sum(dim=-2).to(scorer.weight.dtype)

    def _causal_pool_seq(self, h_seq: torch.Tensor, win: int, scorer: nn.Linear) -> torch.Tensor:
        bsz, t_len, _ = h_seq.shape
        hn = self.span_norm(h_seq.to(scorer.weight.dtype))
        s = scorer(hn).float().squeeze(-1)
        s_pad = F.pad(s, (win - 1, 0), value=-1e9)
        w = F.softmax(s_pad.unfold(1, win, 1), dim=-1)
        hn_f = hn.float()
        hn_pad = F.pad(hn_f, (0, 0, win - 1, 0), value=0.0)
        out = torch.zeros_like(hn_f)
        for j in range(win):
            out = out + w[:, :, j : j + 1] * hn_pad[:, j : j + t_len, :]
        return out.to(scorer.weight.dtype)

    def project_span_keys_from_window(
        self, h_win: torch.Tensor, win_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode `[..., W, d]` cue window into copula-invariant `(k_span_rel, k_span_ent)`."""
        u_rel = self._pool_window(h_win, win_mask, self.span_rel_score)
        u_ent = self._pool_window(h_win, win_mask, self.span_ent_score)
        # Mean over valid window tokens provides smooth phrase context without anchoring to trailing 'is'
        denom = win_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        u_mean = (self.span_norm(h_win.to(u_rel.dtype)).float() * win_mask.float().unsqueeze(-1)).sum(dim=-2) / denom
        u_mean_c = u_mean.to(u_rel.dtype)

        fr = torch.cat([u_rel, u_mean_c], dim=-1)
        fe = torch.cat([u_ent, u_rel], dim=-1)
        if self.training:
            fr = F.dropout(fr, p=0.10, training=True)
            fe = F.dropout(fe, p=0.10, training=True)
        zr = self.span_rel_proj(fr)
        zr = zr + self.span_rel_mlp(F.silu(zr))
        ze = self.span_ent_proj(fe)
        ze = ze + self.span_ent_mlp(F.silu(ze))
        return F.normalize(zr.float(), p=2, dim=-1, eps=1e-6), F.normalize(ze.float(), p=2, dim=-1, eps=1e-6)

    def project_span_queries_seq(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal per-token span `(q_span_rel, q_span_ent)` across `[B, T, d]`."""
        win = self.config.span_window
        u_rel = self._causal_pool_seq(h_seq, win, self.span_rel_score)
        u_ent = self._causal_pool_seq(h_seq, win, self.span_ent_score)
        u_mean = _causal_window_mean(self.span_norm(h_seq.to(u_rel.dtype)), win).to(u_rel.dtype)

        fr = torch.cat([u_rel, u_mean], dim=-1)
        fe = torch.cat([u_ent, u_rel], dim=-1)
        if self.training:
            fr = F.dropout(fr, p=0.10, training=True)
            fe = F.dropout(fe, p=0.10, training=True)
        zr = self.span_rel_proj(fr)
        zr = zr + self.span_rel_mlp(F.silu(zr))
        ze = self.span_ent_proj(fe)
        ze = ze + self.span_ent_mlp(F.silu(ze))
        return F.normalize(zr.float(), p=2, dim=-1, eps=1e-6), F.normalize(ze.float(), p=2, dim=-1, eps=1e-6)

    def project_latent_entity_from_seq(
        self, h_lat_seq: torch.Tensor, seq_mask: Optional[torch.Tensor] = None, prompt_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pool implicitly resolved Hop-1 bridge entity from prompt body at `latent_read_layer` -> `[B, ent_dim]`."""
        if prompt_len is not None and prompt_len < h_lat_seq.shape[1]:
            h_lat_seq = h_lat_seq[:, :prompt_len, :]
            if seq_mask is not None:
                seq_mask = seq_mask[:, :prompt_len]
        hn = self.lat_norm(h_lat_seq.to(self.lat_ent_proj.weight.dtype))
        s = self.lat_tok_score(hn).float().squeeze(-1)
        if seq_mask is not None:
            s = s.masked_fill(seq_mask <= 0, -1e9)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        u_pool = (w * hn.float()).sum(dim=1).to(hn.dtype)
        if seq_mask is not None:
            last_idx = (seq_mask.long().sum(dim=1) - 1).clamp_min(0)
            u_last = hn[torch.arange(hn.shape[0], device=hn.device), last_idx]
        else:
            u_last = hn[:, -1]
        feat = torch.cat([u_pool, u_last], dim=-1)
        if self.training:
            feat = F.dropout(feat, p=0.10, training=True)
        z = self.lat_ent_proj(feat)
        z = z + self.lat_ent_mlp(F.silu(z))
        return F.normalize(z.float(), p=2, dim=-1, eps=1e-6)

    def project_outer_rel_from_seq(
        self, h_read_seq: torch.Tensor, seq_mask: Optional[torch.Tensor] = None, prompt_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Pool the full prompt's outer interrogative/clausal relation at `read_layer` -> `[B, key_dim]` in `direct.k_fam` space."""
        if prompt_len is not None and prompt_len < h_read_seq.shape[1]:
            h_read_seq = h_read_seq[:, :prompt_len, :]
            if seq_mask is not None:
                seq_mask = seq_mask[:, :prompt_len]
        bsz, t_len, _ = h_read_seq.shape
        hn = self.outer_norm(h_read_seq.to(self.outer_rel_proj.weight.dtype))
        s = self.outer_tok_score(hn).float().squeeze(-1)
        if seq_mask is not None:
            s = s.masked_fill(seq_mask <= 0, -1e9)
            lens = seq_mask.long().sum(dim=1).clamp_min(1)
        else:
            lens = torch.full((bsz,), t_len, device=h_read_seq.device, dtype=torch.long)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        u_pool = (w * hn.float()).sum(dim=1).to(hn.dtype)

        # First-half prefix mean explicitly captures leading outer interrogatives ("Question: What is the capital of...")
        half_lens = ((lens + 1) // 2).clamp_min(1)
        pos_idx = torch.arange(t_len, device=h_read_seq.device).unsqueeze(0)
        pref_m = (pos_idx < half_lens.unsqueeze(1)).float()
        if seq_mask is not None:
            pref_m = pref_m * (seq_mask > 0).float()
        u_pref = ((hn.float() * pref_m.unsqueeze(-1)).sum(dim=1) / pref_m.sum(dim=1, keepdim=True).clamp_min(1.0)).to(hn.dtype)

        last_idx = (lens - 1).clamp_min(0)
        ar = torch.arange(bsz, device=hn.device)
        u_last = hn[ar, last_idx]

        # Anchor on frozen `direct.project_key` at `last_idx` so 1-hop & unseen OOD ZsRE relations match `k_fam` natively,
        # while the learned prefix adapter swings `q_outer_rel` onto the outer relation whenever a 2-hop prefix is present!
        h_ctx = _causal_window_mean(h_read_seq, self.direct.config.key_ctx_window)
        q_tail = self.direct.project_key(h_read_seq[ar, last_idx], h_ctx[ar, last_idx])

        feat = torch.cat([u_pool, u_pref, u_last], dim=-1)
        if self.training:
            feat = F.dropout(feat, p=0.10, training=True)
        delta_out = self.outer_rel_mlp(F.silu(self.outer_rel_proj(feat))).float()
        return F.normalize(q_tail.float() + delta_out, p=2, dim=-1, eps=1e-6)

    def arbitrate_direct_vs_bridge(
        self,
        dir_rt: Dict[str, torch.Tensor],
        span_rt: Dict[str, torch.Tensor],
        h_read_seq: torch.Tensor,
        compiled: Dict[str, Any],
        seq_mask: Optional[torch.Tensor] = None,
        prompt_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decide whether `direct` owns the prompt (1-hop) or an embedded Hop-1 span overrides `direct` (2-hop)."""
        bsz, seq_len, _ = h_read_seq.shape
        p_end = (prompt_len - 1) if (prompt_len is not None and prompt_len <= seq_len) else (seq_len - 1)
        g_dir_end = dir_rt["max_gate"][:, p_end]
        g_sp = span_rt["peak_gate"]

        q_out = self.project_outer_rel_from_seq(h_read_seq, seq_mask=seq_mask, prompt_len=prompt_len)
        k_fam = compiled["k_fam"]
        if k_fam.shape[0] == 1 and bsz > 1:
            k_fam = k_fam.expand(bsz, -1, -1)
        ar = torch.arange(bsz, device=h_read_seq.device)
        k_fam_ms = k_fam[ar, span_rt["peak_m"]]
        s_outer_ms = (q_out * k_fam_ms).sum(dim=-1)
        tau_od = self.outer_direct_tau.float().clamp(self.config.min_outer_direct_tau, 0.95)

        # Stage A overrides Direct iff a mid-sentence/nested-NP span matched AND either Direct was silent at `p_end`
        # OR the full prompt's outer relation diverges from the matched span's single-hop relation!
        need_stage_a = (g_sp > 0.05) & ((g_dir_end <= 0.15) | (s_outer_ms < tau_od))
        direct_owns = (g_dir_end > 0.15) & (~need_stage_a)
        return {
            "direct_owns": direct_owns,
            "need_stage_a": need_stage_a,
            "q_outer_rel": q_out,
            "s_outer_ms": s_outer_ms,
            "g_dir_end": g_dir_end,
        }

    def compile_v4_states(
        self,
        direct_compiled: Dict[str, torch.Tensor],
        h_span_win: torch.Tensor,          # [B, M, W_s, d] or [M, W_s, d] at `bridge_read_layer`
        m_span_win: torch.Tensor,          # [B, M, W_s] or [M, W_s]
        h_tgt_bridge: torch.Tensor,        # [B, M, n_bridge_inj, d] or [M, n_bridge_inj, d] native entity states
        k_tgt_ent: Optional[torch.Tensor] = None,  # [B, M, ent_dim] target entity keys for 2-edit chaining
    ) -> Dict[str, Any]:
        if h_span_win.ndim == 3:
            h_span_win = h_span_win.unsqueeze(0)
            m_span_win = m_span_win.unsqueeze(0)
            h_tgt_bridge = h_tgt_bridge.unsqueeze(0)
            if k_tgt_ent is not None and k_tgt_ent.ndim == 2:
                k_tgt_ent = k_tgt_ent.unsqueeze(0)
        k_srel, k_sent = self.project_span_keys_from_window(h_span_win, m_span_win)
        g_ext = float(direct_compiled["g_ext"].item())
        if g_ext == 0.0:
            h_tgt_bridge = torch.zeros_like(h_tgt_bridge)
        out = dict(direct_compiled)
        out.update({
            "k_srel": k_srel,
            "k_sent": k_sent,
            "h_tgt_bridge": h_tgt_bridge.float(),
            "k_tgt_ent": k_tgt_ent,
        })
        return out

    def route_span_bridge(
        self,
        h_bridge_seq: torch.Tensor,          # [B, T, d] at `bridge_read_layer`
        compiled: Dict[str, Any],
        seq_mask: Optional[torch.Tensor] = None,
        tail_guard_tokens: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """Detect mid-sentence Hop-1 bridge clause `[B, T, d]` -> peak position `t*` and episode `m*`."""
        bsz, t_len, _ = h_bridge_seq.shape
        k_srel = compiled["k_srel"]
        k_sent = compiled["k_sent"]
        if k_srel.shape[0] == 1 and bsz > 1:
            k_srel = k_srel.expand(bsz, -1, -1)
            k_sent = k_sent.expand(bsz, -1, -1)

        q_srel, q_sent = self.project_span_queries_seq(h_bridge_seq)
        s_rel = torch.einsum("btk,bmk->btm", q_srel, k_srel)
        s_ent = torch.einsum("btk,bmk->btm", q_sent, k_sent)

        tau_sr = self.span_rel_tau.float().clamp(self.config.min_span_rel_tau, 0.90)
        tau_se = self.span_ent_tau.float().clamp(self.config.min_span_ent_tau, 0.90)
        g_raw = self._hill(s_rel, tau_sr) * self._hill(s_ent, tau_se)                 # [B, T, M]

        # Mask BOS (`t == 0`) and final answer-wrapper tail (`t >= L_i - tail_guard_tokens`) so only
        # genuine mid-sentence clauses can trigger Stage A bridge replacement!
        pos_mask = torch.ones((bsz, t_len), device=h_bridge_seq.device, dtype=torch.float32)
        pos_mask[:, 0] = 0.0
        if seq_mask is not None:
            lens = seq_mask.long().sum(dim=1)
            for i in range(bsz):
                li = int(lens[i].item())
                cutoff = max(1, li - tail_guard_tokens)
                pos_mask[i, cutoff:] = 0.0
        elif t_len > tail_guard_tokens + 1:
            pos_mask[:, t_len - tail_guard_tokens :] = 0.0

        if float(compiled["g_ext"].item()) == 0.0:
            pos_mask.zero_()
        g_masked = g_raw * pos_mask.unsqueeze(-1)                                    # [B, T, M]

        # Peak episode & token selection per sequence
        s_joint = 0.5 * (s_rel + s_ent)
        score_tm = g_masked + 1e-3 * s_joint * pos_mask.unsqueeze(-1)
        flat_idx = score_tm.reshape(bsz, -1).argmax(dim=-1)                          # [B]
        m_cnt = k_srel.shape[1]
        peak_t = flat_idx // m_cnt                                                   # [B]
        peak_m = flat_idx % m_cnt                                                    # [B]
        ar = torch.arange(bsz, device=h_bridge_seq.device)
        peak_gate = g_masked[ar, peak_t, peak_m]                                     # [B]
        peak_sim = s_joint[ar, peak_t, peak_m]                                       # [B]

        # Build localized 3-token soft envelope `[peak_t - 2, peak_t - 1, peak_t]` for smooth entity erasure+write
        spread = torch.sigmoid(self.bridge_tok_spread.float())                       # [3]
        tok_env = torch.zeros((bsz, t_len), device=h_bridge_seq.device, dtype=torch.float32)
        for off_i, lag in enumerate((2, 1, 0)):
            tp = (peak_t - lag).clamp_min(1)
            valid = ((peak_t - lag) >= 1).float()
            tok_env[ar, tp] = torch.maximum(tok_env[ar, tp], peak_gate * spread[off_i] * valid)

        return {
            "peak_t": peak_t,
            "peak_m": peak_m,
            "peak_gate": peak_gate,
            "peak_sim": peak_sim,
            "tok_env": tok_env,
            "g_masked": g_masked,
        }

    def compute_bridge_replacement(
        self,
        h_cur: torch.Tensor,               # [B, T, d]
        span_rt: Dict[str, torch.Tensor],
        compiled: Dict[str, Any],
        bridge_inj_idx: int,
        sign_flip: bool = False,
        active_mask: Optional[torch.Tensor] = None,  # [B] 1.0 when Stage A allowed
    ) -> torch.Tensor:
        """Read-Equivalent convex replacement: `(1 - a) h_t + a * RMSMatch(h_entity(y_m)) + delta_adapt`."""
        bsz, _, d_m = h_cur.shape
        hb = compiled["h_tgt_bridge"]                                                # [1 or B, M, n_b, d]
        if hb.shape[0] == 1 and bsz > 1:
            hb = hb.expand(bsz, -1, -1, -1)
        ar = torch.arange(bsz, device=h_cur.device)
        h_ent = hb[ar, span_rt["peak_m"], bridge_inj_idx, :]                         # [B, d] float32

        # Match local RMS of `h_cur` so the injected entity vector sits natively on the layer's manifold
        cur_rms = h_cur.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        ent_rms = h_ent.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().unsqueeze(1)
        h_ent_matched = h_ent.unsqueeze(1) * (cur_rms / ent_rms)                     # [B, T, d]

        # Optional learned low-rank refinement (`0` at initialization!)
        cat_in = torch.cat([
            self.span_norm(h_cur.to(self.bridge_down.weight.dtype)),
            self.span_norm(h_ent_matched.to(self.bridge_down.weight.dtype)),
        ], dim=-1)
        z_b = F.silu(self.bridge_down(cat_in))
        delta_adapt = self.bridge_ups[bridge_inj_idx](z_b).float() * cur_rms

        alpha = torch.sigmoid(self.bridge_alpha_logit[bridge_inj_idx].float())
        env = span_rt["tok_env"].unsqueeze(-1)                                       # [B, T, 1]
        if active_mask is not None:
            env = env * active_mask.float().view(bsz, 1, 1)
        w = (alpha * env).clamp(0.0, 1.0)
        target_state = h_ent_matched + delta_adapt
        if sign_flip:
            # Causal sign-flip verification pushes away from target entity representation
            h_out = h_cur.float() - w * target_state
        else:
            h_out = (1.0 - w) * h_cur.float() + w * target_state
        return h_out.to(h_cur.dtype)

    def route_latent_hop2(
        self,
        h_read_seq: torch.Tensor,          # [B, T, d] at `read_layer` (L16) for outer relation matching
        h_lat_seq: torch.Tensor,           # [B, T, d] at `latent_read_layer` (L20) for resolved Hop-1 entity
        compiled: Dict[str, Any],
        seq_mask: Optional[torch.Tensor] = None,
        prompt_len: Optional[int] = None,
        q_outer_rel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Match outer question relation (`L16` tail OR prefix outer-rel) x implicitly resolved Hop-1 bridge entity (`L20`)."""
        bsz, seq_len, _ = h_read_seq.shape
        p_end = (prompt_len - 1) if (prompt_len is not None and prompt_len <= seq_len) else (seq_len - 1)
        k_fam = compiled["k_fam"]
        k_ent = compiled["k_ent"]
        v_bank = compiled["v_bank"]
        e_bank = compiled["e_bank"]
        tgt_mask = compiled["tgt_mask"]
        if k_fam.shape[0] == 1 and bsz > 1:
            k_fam = k_fam.expand(bsz, -1, -1)
            k_ent = k_ent.expand(bsz, -1, -1)
            v_bank = v_bank.expand(bsz, -1, -1, -1)
            e_bank = e_bank.expand(bsz, -1, -1, -1)
            tgt_mask = tgt_mask.expand(bsz, -1, -1)

        h_ctx = _causal_window_mean(h_read_seq, self.direct.config.key_ctx_window)
        q_rel_all = self.direct.project_key(h_read_seq, h_ctx)                       # [B, T, key_dim]
        if q_outer_rel is None:
            q_outer_rel = self.project_outer_rel_from_seq(h_read_seq, seq_mask=seq_mask, prompt_len=prompt_len)
        q_lat_ent = self.project_latent_entity_from_seq(h_lat_seq, seq_mask=seq_mask, prompt_len=prompt_len)  # [B, ent_dim]

        sims_rel_tail = torch.einsum("btk,bmk->btm", q_rel_all, k_fam)               # [B, T, M]
        sims_rel_out = torch.einsum("bk,bmk->bm", q_outer_rel, k_fam).unsqueeze(1)   # [B, 1, M]
        sims_rel = torch.maximum(sims_rel_tail, sims_rel_out)                        # [B, T, M]
        sims_lent = torch.einsum("bk,bmk->bm", q_lat_ent, k_ent).unsqueeze(1).expand(-1, seq_len, -1)

        tau_lr = self.lat_rel_tau.float().clamp(self.config.min_lat_rel_tau, 0.88)
        tau_le = self.lat_ent_tau.float().clamp(self.config.min_lat_ent_tau, 0.90)
        g_all = self._hill(sims_rel, tau_lr) * self._hill(sims_lent, tau_le)
        if float(compiled["g_ext"].item()) == 0.0:
            g_all = torch.zeros_like(g_all)

        k_top = min(self.config.top_k, k_fam.shape[1])
        sims_joint = 0.5 * (sims_rel + sims_lent)
        _, top_idx = (g_all + 1e-3 * sims_joint).topk(k=k_top, dim=-1)
        shaped_g = g_all.gather(-1, top_idx)
        top_joint = sims_joint.gather(-1, top_idx)

        hn = self.direct.norm(self.direct._match_dtype(h_read_seq))
        r_gate = torch.sigmoid(self.direct.rel_gate(hn).float().squeeze(-1)).unsqueeze(-1)
        g_top = r_gate * shaped_g
        p_logits = (top_joint * 15.0).masked_fill(shaped_g <= 0.0, -1e4)
        p_top = F.softmax(p_logits, dim=-1) if k_top > 1 else torch.ones_like(top_joint)
        boost = 1.0 + 0.5 * torch.tanh(self.lat_boost.float())
        w_top = g_top * p_top * top_joint.clamp_min(0.0) * boost
        if prompt_len is not None and p_end > 0:
            # Stage C only triggers at `p_end` (`\nAnswer:`) and latches onto generated tokens `p_end .. p_end + kv - 1`
            trig_mask = torch.zeros((1, seq_len, 1), device=h_read_seq.device, dtype=w_top.dtype)
            trig_mask[:, p_end, :] = 1.0
            w_top = w_top * trig_mask

        kv = self.config.max_tgt_tokens
        b_ar = torch.arange(bsz, device=h_read_seq.device).view(bsz, 1, 1)
        v_parts: List[torch.Tensor] = []
        e_parts: List[torch.Tensor] = []
        for lag in range(min(kv, seq_len)):
            t_l = seq_len - lag
            idx_l = top_idx[:, :t_l, :]
            m_l = tgt_mask[b_ar, idx_l, lag]
            w_l = (w_top[:, :t_l, :] * m_l).unsqueeze(-1)
            c_v = (w_l * v_bank[b_ar, idx_l, lag, :]).sum(dim=2)
            c_e = (w_l * e_bank[b_ar, idx_l, lag, :]).sum(dim=2)
            v_parts.append(c_v if lag == 0 else F.pad(c_v, (0, 0, lag, 0)))
            e_parts.append(c_e if lag == 0 else F.pad(c_e, (0, 0, lag, 0)))

        return {
            "latched_v": torch.stack(v_parts, dim=0).sum(dim=0),
            "latched_e": torch.stack(e_parts, dim=0).sum(dim=0),
            "top_idx_end": top_idx[:, p_end, :],
            "max_gate_end": g_top[:, p_end, 0],
            "max_gate_seq": g_top[..., 0].max(dim=1).values,
        }


class FOMPv4BackboneWrapper(nn.Module):
    """Unified wrapper supporting BOTH `mode='legible'` (pure activation) and `mode='bridge_kv'` (Virtual-KV)."""

    def __init__(self, backbone: nn.Module, fomp: FOMPv4Module, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.fomp = fomp
        self.layers = _get_decoder_layers(backbone)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self._last_routing: Dict[str, Any] = {}

    @torch.no_grad()
    def encode_layers_batch(
        self,
        input_ids: torch.Tensor,
        layer_indices: Sequence[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """Run frozen backbone once up to `max(layer_indices)` and capture input hidden states at each requested layer."""
        req = sorted(set(int(x) for x in layer_indices))
        max_l = req[-1]
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        for li in req:
            def _mk(l_idx: int):
                def _pre(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
                    h = args[0] if len(args) > 0 else kwargs["hidden_states"]
                    captured[l_idx] = h.detach()
                    if l_idx == max_l:
                        raise _EarlyStopSignal(h.detach())
                return _pre
            handles.append(self.layers[li].register_forward_pre_hook(_mk(li), with_kwargs=True))

        was_tr = self.backbone.training
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
            for hd in handles:
                hd.remove()
            if was_tr:
                self.backbone.train()
        return captured

    @torch.no_grad()
    def lookup_target_embeddings(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        w_out = _get_out_embedding_weight(self.backbone)
        return F.embedding(tgt_ids.to(w_out.device), w_out).detach()

    @torch.no_grad()
    def _build_kv_snapshot(self, pref_ids: List[int], device: torch.device) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Pre-compile a single episode's RoPE-aligned KV tensors if the backbone supports `use_cache=True`."""
        if not hasattr(self.backbone, "model"):
            return None
        inp = torch.tensor([pref_ids], device=device, dtype=torch.long)
        out = self.backbone(input_ids=inp, use_cache=True)
        pkv = getattr(out, "past_key_values", None)
        if pkv is None:
            return None
        if hasattr(pkv, "layers"):
            return [(pkv.layers[i].keys.detach().clone(), pkv.layers[i].values.detach().clone()) for i in range(len(pkv.layers))]
        return [(pkv[i][0].detach().clone(), pkv[i][1].detach().clone()) for i in range(len(pkv))]

    def compile_episodes(
        self,
        cue_ids_list: Sequence[Sequence[int]],
        tgt_ids_list: Sequence[Sequence[int]],
        pad_id: int,
        device: torch.device,
        bos_id: Optional[int] = None,
        cue_span_ids_list: Optional[Sequence[Sequence[int]]] = None,
        fact_texts: Optional[Sequence[str]] = None,
        tokenizer: Optional[Any] = None,
        g_ext: float = 1.0,
        precompile_kv: bool = False,
        chunk_size: int = 128,
    ) -> Dict[str, Any]:
        """Single-pass compilation for BOTH `legible` (pure activation) and `bridge_kv` (pre-compiled Virtual-KV)."""
        cfg = self.fomp.config
        m_cnt = len(cue_ids_list)
        kv = cfg.max_tgt_tokens
        win_c = self.fomp.direct.config.key_ctx_window
        win_e = self.fomp.direct.config.ent_window
        win_s = cfg.span_window
        span_src_list = cue_span_ids_list if cue_span_ids_list is not None else cue_ids_list

        hc_p, hctx_p, hwin_p, hwm_p = [], [], [], []
        hsw_p, msw_p = [], []
        ht_p, et_p, tm_p, htb_p, kte_p = [], [], [], [], []

        cue_layers = [cfg.bridge_read_layer, cfg.read_layer]
        tgt_layers = sorted(set([cfg.read_layer, *cfg.bridge_inject_layers]))

        for s0 in range(0, m_cnt, chunk_size):
            c_sub = cue_ids_list[s0 : s0 + chunk_size]
            sp_sub = span_src_list[s0 : s0 + chunk_size]
            t_sub = tgt_ids_list[s0 : s0 + chunk_size]
            n_sub = len(c_sub)
            ar_s = torch.arange(n_sub, device=device)

            # 1. Direct cues -> `read_layer`
            mx_c = max(len(c) for c in c_sub)
            c_mat = torch.full((n_sub, mx_c), pad_id, device=device, dtype=torch.long)
            c_msk = torch.zeros((n_sub, mx_c), device=device, dtype=torch.long)
            c_last = torch.zeros((n_sub,), device=device, dtype=torch.long)
            for i, c in enumerate(c_sub):
                ln = len(c)
                c_mat[i, :ln] = torch.tensor(c, device=device, dtype=torch.long)
                c_msk[i, :ln] = 1
                c_last[i] = ln - 1
            cap_c = self.encode_layers_batch(c_mat, cue_layers, attention_mask=c_msk)
            h_rl = cap_c[cfg.read_layer]
            h_ctx = _causal_window_mean(h_rl, win_c)
            hc_p.append(h_rl[ar_s, c_last])
            hctx_p.append(h_ctx[ar_s, c_last])
            hw, hwm = _extract_tail_window(h_rl, c_last, win_e)
            hwin_p.append(hw)
            hwm_p.append(hwm)

            # 2. Copula-stripped span cues -> `bridge_read_layer`
            if cue_span_ids_list is None:
                sw, swm = _extract_tail_window(cap_c[cfg.bridge_read_layer], c_last, win_s)
            else:
                mx_sp = max(len(x) for x in sp_sub)
                sp_mat = torch.full((n_sub, mx_sp), pad_id, device=device, dtype=torch.long)
                sp_msk = torch.zeros((n_sub, mx_sp), device=device, dtype=torch.long)
                sp_last = torch.zeros((n_sub,), device=device, dtype=torch.long)
                for i, x in enumerate(sp_sub):
                    ln = len(x)
                    sp_mat[i, :ln] = torch.tensor(x, device=device, dtype=torch.long)
                    sp_msk[i, :ln] = 1
                    sp_last[i] = ln - 1
                cap_sp = self.encode_layers_batch(sp_mat, [cfg.bridge_read_layer], attention_mask=sp_msk)
                sw, swm = _extract_tail_window(cap_sp[cfg.bridge_read_layer], sp_last, win_s)
            hsw_p.append(sw)
            msw_p.append(swm)

            # 3. Target tokens (standalone for `v_bank` + BOS-prefixed for native entity hidden states across `bridge_inject_layers`)
            t_mat = torch.full((n_sub, kv), pad_id, device=device, dtype=torch.long)
            t_msk = torch.zeros((n_sub, kv), device=device, dtype=torch.long)
            tb_len = kv + (1 if bos_id is not None else 0)
            tb_mat = torch.full((n_sub, tb_len), pad_id, device=device, dtype=torch.long)
            tb_msk = torch.zeros((n_sub, tb_len), device=device, dtype=torch.long)
            tb_last = torch.zeros((n_sub,), device=device, dtype=torch.long)
            for i, t in enumerate(t_sub):
                ln = min(kv, max(1, len(t)))
                t_slice = list(t[:ln])
                t_mat[i, :ln] = torch.tensor(t_slice, device=device, dtype=torch.long)
                t_msk[i, :ln] = 1
                tb_seq = ([bos_id] + t_slice) if bos_id is not None else t_slice
                tb_mat[i, : len(tb_seq)] = torch.tensor(tb_seq, device=device, dtype=torch.long)
                tb_msk[i, : len(tb_seq)] = 1
                tb_last[i] = len(tb_seq) - 1

            ht_p.append(self.encode_layers_batch(t_mat, [cfg.read_layer], attention_mask=t_msk)[cfg.read_layer])
            et_p.append(self.lookup_target_embeddings(t_mat))
            tm_p.append(t_msk)

            cap_tb = self.encode_layers_batch(tb_mat, tgt_layers, attention_mask=tb_msk)
            b_states = torch.stack([cap_tb[li][ar_s, tb_last] for li in cfg.bridge_inject_layers], dim=1)
            htb_p.append(b_states)
            tw_e, twm_e = _extract_tail_window(cap_tb[cfg.read_layer], tb_last, win_e)
            with torch.no_grad():
                kte_p.append(self.fomp.direct.project_entity_key_from_window(tw_e, twm_e, h_last=cap_tb[cfg.read_layer][ar_s, tb_last]))

        direct_comp = self.fomp.direct.compile_from_states(
            torch.cat(hc_p, dim=0),
            torch.cat(ht_p, dim=0),
            torch.cat(tm_p, dim=0),
            g_ext=g_ext,
            e_tgt=torch.cat(et_p, dim=0),
            h_cue_ctx=torch.cat(hctx_p, dim=0),
            h_cue_win=torch.cat(hwin_p, dim=0),
            h_cue_wmask=torch.cat(hwm_p, dim=0),
        )
        comp = self.fomp.compile_v4_states(
            direct_comp,
            torch.cat(hsw_p, dim=0),
            torch.cat(msw_p, dim=0),
            torch.cat(htb_p, dim=0),
            k_tgt_ent=torch.cat(kte_p, dim=0),
        )

        # 4. Optional Virtual-KV pre-compilation for Route 2 (`bridge_kv`)
        if precompile_kv and fact_texts is not None and tokenizer is not None and g_ext > 0.0:
            header = "Updated facts (follow these strictly over prior knowledge):\n"
            kv_snaps: List[Optional[List[Tuple[torch.Tensor, torch.Tensor]]]] = []
            fact_line_ids: List[List[int]] = []
            pref_full_ids: List[List[int]] = []
            for ft in fact_texts:
                full_txt = header + ft + "\n\n"
                p_ids = ([bos_id] if bos_id is not None else []) + tokenizer.encode(full_txt, add_special_tokens=False)
                pref_full_ids.append(p_ids)
                fact_line_ids.append(tokenizer.encode(ft + "\n", add_special_tokens=False))
                kv_snaps.append(self._build_kv_snapshot(p_ids, device) if m_cnt <= 128 else None)
            comp["kv_snaps"] = kv_snaps
            comp["pref_full_ids"] = pref_full_ids
            comp["fact_line_ids"] = fact_line_ids
            comp["kv_header_ids"] = ([bos_id] if bos_id is not None else []) + tokenizer.encode(header, add_special_tokens=False)
            comp["kv_sep_ids"] = tokenizer.encode("\n", add_special_tokens=False)
            comp["bos_id"] = bos_id
        return comp

    @contextmanager
    def _bridge_only_hooks(
        self,
        compiled: Dict[str, Any],
        seq_mask: Optional[torch.Tensor] = None,
        sign_flip: bool = False,
        active_mask: Optional[torch.Tensor] = None,
        tail_guard_tokens: int = 3,
        prompt_len: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Attach Stage-A Read-Equivalent Bridge hooks across `bridge_read_layer` & `bridge_inject_layers`."""
        cfg = self.fomp.config
        state: Dict[str, Any] = {}
        handles = []
        hook_layers = sorted(set([cfg.bridge_read_layer, *cfg.bridge_inject_layers]))

        for li in hook_layers:
            def _mk(l_idx: int):
                def _pre(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                    use_kw = len(args) == 0
                    h = kwargs["hidden_states"] if use_kw else args[0]
                    if l_idx == cfg.bridge_read_layer:
                        h_br = h[:, :prompt_len, :] if (prompt_len is not None and prompt_len < h.shape[1]) else h
                        m_br = seq_mask[:, :prompt_len] if (seq_mask is not None and prompt_len is not None and prompt_len < h.shape[1]) else seq_mask
                        sp_rt = self.fomp.route_span_bridge(
                            h_br, compiled, seq_mask=m_br, tail_guard_tokens=tail_guard_tokens
                        )
                        if h_br.shape[1] < h.shape[1]:
                            sp_rt["tok_env"] = F.pad(sp_rt["tok_env"], (0, h.shape[1] - h_br.shape[1]), value=0.0)
                        state["span_rt"] = sp_rt
                    if l_idx in self.fomp.bridge_inj_map:
                        h_new = self.fomp.compute_bridge_replacement(
                            h,
                            state["span_rt"],
                            compiled,
                            self.fomp.bridge_inj_map[l_idx],
                            sign_flip=sign_flip,
                            active_mask=active_mask,
                        )
                        if use_kw:
                            nkw = dict(kwargs)
                            nkw["hidden_states"] = h_new
                            return args, nkw
                        return (h_new, *args[1:]), kwargs
                    return args, kwargs
                return _pre
            handles.append(self.layers[li].register_forward_pre_hook(_mk(li), with_kwargs=True))
        try:
            yield state
        finally:
            for hd in handles:
                hd.remove()

    def forward_bridge_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        compiled: Dict[str, Any],
        tail_guard_tokens: int = 3,
    ) -> Dict[str, Any]:
        """Differentiable Stage-A training pass for Self-Supervised Frozen-Teacher KL Distillation."""
        with self._bridge_only_hooks(compiled=compiled, seq_mask=attention_mask, tail_guard_tokens=tail_guard_tokens) as st:
            if hasattr(self.backbone, "model"):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                out = self.backbone(input_ids)
        return {"logits": _extract_logits(out), "span_rt": st["span_rt"]}

    def _forward_single_pass_probed(
        self,
        input_ids: torch.Tensor,
        compiled: Optional[Dict[str, Any]],
        attention_mask: Optional[torch.Tensor] = None,
        sign_flip: bool = False,
        allow_stage_a: bool = True,
        allow_stage_c: bool = True,
        prompt_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Single-pass execution handling Direct (`L16`), Case-2 Latent (`L20`), and Locality (`0` gates) in 1 pass,
        triggering a clean Stage-A re-pass strictly when an embedded Hop-1 span fires (`need_stage_a == True`)."""
        if compiled is None or float(compiled["g_ext"].item()) == 0.0:
            if attention_mask is not None and hasattr(self.backbone, "model"):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                out = self.backbone(input_ids)
            return {"logits": _extract_logits(out), "mode_used": "off", "extra_kv_tokens": 0}

        cfg = self.fomp.config
        state: Dict[str, Any] = {}
        handles = []
        all_layers = sorted(set([cfg.bridge_read_layer, cfg.read_layer, cfg.latent_read_layer, *cfg.inject_layers]))

        class _NeedStageARepass(Exception):
            pass

        for li in all_layers:
            def _mk(l_idx: int):
                def _pre(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                    use_kw = len(args) == 0
                    h = kwargs["hidden_states"] if use_kw else args[0]
                    if l_idx == cfg.bridge_read_layer:
                        h_br = h[:, :prompt_len, :] if (prompt_len is not None and prompt_len < h.shape[1]) else h
                        m_br = attention_mask[:, :prompt_len] if (attention_mask is not None and prompt_len is not None and prompt_len < h.shape[1]) else attention_mask
                        sp_rt = self.fomp.route_span_bridge(h_br, compiled, seq_mask=m_br, tail_guard_tokens=3)
                        if h_br.shape[1] < h.shape[1]:
                            sp_rt["tok_env"] = F.pad(sp_rt["tok_env"], (0, h.shape[1] - h_br.shape[1]), value=0.0)
                        state["span_rt"] = sp_rt
                    if l_idx == cfg.read_layer:
                        state["h_read"] = h
                        rt_d = self.fomp.direct.route_and_latch(h, compiled)
                        state["dir_rt"] = rt_d
                        arb = self.fomp.arbitrate_direct_vs_bridge(
                            rt_d, state["span_rt"], h, compiled, seq_mask=attention_mask, prompt_len=prompt_len
                        )
                        state["arb"] = arb
                        if allow_stage_a and bool(arb["need_stage_a"].any().item()):
                            state["active_a"] = arb["need_stage_a"].float()
                            raise _NeedStageARepass()
                    if l_idx == cfg.latent_read_layer and allow_stage_c:
                        state["lat_rt"] = self.fomp.route_latent_hop2(
                            state["h_read"], h, compiled, seq_mask=attention_mask, prompt_len=prompt_len,
                            q_outer_rel=state["arb"]["q_outer_rel"],
                        )

                    if l_idx in self.fomp.direct.inject_map:
                        inj_i = self.fomp.direct.inject_map[l_idx]
                        h_rms = h.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().detach()
                        sup_d = state["arb"]["direct_owns"].float().view(-1, 1, 1)
                        delta = self.fomp.direct.compute_layer_delta(
                            state["dir_rt"]["latched_v"] * sup_d,
                            inj_i,
                            h.dtype,
                            sign_flip=sign_flip,
                            latched_e=state["dir_rt"]["latched_e"] * sup_d,
                            h_rms=h_rms,
                        )
                        if allow_stage_c and "lat_rt" in state and inj_i in self.fomp.late_inj_indices:
                            sup_c = (~state["arb"]["direct_owns"]).float().view(-1, 1, 1)
                            if bool((sup_c > 0).any().item()):
                                delta_c = self.fomp.direct.compute_layer_delta(
                                    state["lat_rt"]["latched_v"] * sup_c,
                                    inj_i,
                                    h.dtype,
                                    sign_flip=sign_flip,
                                    latched_e=state["lat_rt"]["latched_e"] * sup_c,
                                    h_rms=h_rms,
                                )
                                delta = delta + delta_c
                        h_new = h + delta
                        if use_kw:
                            nkw = dict(kwargs)
                            nkw["hidden_states"] = h_new
                            return args, nkw
                        return (h_new, *args[1:]), kwargs
                    return args, kwargs
                return _pre
            handles.append(self.layers[li].register_forward_pre_hook(_mk(li), with_kwargs=True))

        repass = False
        try:
            try:
                if attention_mask is not None and hasattr(self.backbone, "model"):
                    out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                else:
                    out = self.backbone(input_ids)
            except _NeedStageARepass:
                repass = True
        finally:
            for hd in handles:
                hd.remove()

        if repass:
            with self._bridge_only_hooks(
                compiled=compiled, seq_mask=attention_mask, sign_flip=sign_flip,
                active_mask=state["active_a"], tail_guard_tokens=3, prompt_len=prompt_len,
            ) as st_a:
                if attention_mask is not None and hasattr(self.backbone, "model"):
                    out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                else:
                    out = self.backbone(input_ids)
            self._last_routing = {"mode": "stage_a_legible", "span_rt": st_a["span_rt"], "arb": state.get("arb")}
            return {"logits": _extract_logits(out), "mode_used": "stage_a_legible", "extra_kv_tokens": 0}

        self._last_routing = {
            "mode": "direct_or_stage_c",
            "dir_rt": state.get("dir_rt"),
            "span_rt": state.get("span_rt"),
            "lat_rt": state.get("lat_rt"),
            "arb": state.get("arb"),
        }
        return {"logits": _extract_logits(out), "mode_used": "direct_or_stage_c", "extra_kv_tokens": 0}

    @torch.no_grad()
    def select_virtual_kv_episodes(
        self,
        prompt_ids: Sequence[int],
        compiled: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[List[int], str]:
        """Probe Stage A/B/C gates once (`0..L20`): return `([], 'direct_hit' | 'none')` when single-hop or locality,
        or `(matched_episode_indices, reason)` strictly when a multi-hop bridge (`g_span` or `g_latent`) fires!"""
        cfg = self.fomp.config
        inp = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
        msk = torch.ones_like(inp)
        caps = self.encode_layers_batch(inp, [cfg.bridge_read_layer, cfg.read_layer, cfg.latent_read_layer], attention_mask=msk)

        dir_rt = self.fomp.direct.route_and_latch(caps[cfg.read_layer], compiled)
        span_rt = self.fomp.route_span_bridge(caps[cfg.bridge_read_layer], compiled, seq_mask=msk, tail_guard_tokens=3)
        arb = self.fomp.arbitrate_direct_vs_bridge(dir_rt, span_rt, caps[cfg.read_layer], compiled, seq_mask=msk)
        if bool(arb["direct_owns"][0].item()):
            return [], "direct_hit"

        picked: List[int] = []
        reason = "none"
        if float(span_rt["peak_gate"][0].item()) > 0.05:
            m1 = int(span_rt["peak_m"][0].item())
            picked.append(m1)
            reason = "span_hop1"
            # Check 2-edit chain: does `target_new` of `m1` match `k_ent` of another episode `m2` in the bank?
            if compiled.get("k_tgt_ent") is not None:
                kt = compiled["k_tgt_ent"][0, m1]                                    # [ent_dim]
                sims_chain = torch.mv(compiled["k_ent"][0], kt)
                sims_chain[m1] = -1.0
                tau_e = float(self.fomp.direct.ent_tau.float().clamp(self.fomp.direct.config.min_ent_tau, 0.90).item())
                best_v, best_i = sims_chain.max(dim=0)
                if float(best_v.item()) >= tau_e:
                    picked.append(int(best_i.item()))

        lat_rt = self.fomp.route_latent_hop2(
            caps[cfg.read_layer], caps[cfg.latent_read_layer], compiled, seq_mask=msk, q_outer_rel=arb["q_outer_rel"]
        )
        if float(lat_rt["max_gate_end"][0].item()) > 0.05:
            for idx_c in lat_rt["top_idx_end"][0].tolist():
                if int(idx_c) not in picked:
                    picked.append(int(idx_c))
                    if reason == "none":
                        reason = "latent_hop2"
                if len(picked) >= 2:
                    break
        return picked[:2], reason

    def _run_with_virtual_kv_ids(
        self,
        query_ids: Sequence[int],
        ep_indices: List[int],
        compiled: Dict[str, Any],
        device: torch.device,
        sign_flip: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """Evaluate logits on `query_ids` conditioned on pre-compiled Virtual-KV of `ep_indices`."""
        if sign_flip:
            # Under causal sign-flip verification, suppress Virtual-KV retrieval completely
            inp = torch.tensor([list(query_ids)], device=device, dtype=torch.long)
            out = self.backbone(inp)
            return _extract_logits(out), 0

        bos_id = compiled.get("bos_id")
        q_no_bos = list(query_ids[1:]) if (bos_id is not None and len(query_ids) > 1 and query_ids[0] == bos_id) else list(query_ids)

        if len(ep_indices) == 1 and compiled.get("kv_snaps") and compiled["kv_snaps"][ep_indices[0]] is not None:
            from transformers import DynamicCache

            snap = compiled["kv_snaps"][ep_indices[0]]
            dc = DynamicCache()
            for li, (k_t, v_t) in enumerate(snap):
                dc.update(k_t.to(device).clone(), v_t.to(device).clone(), li)
            q_t = torch.tensor([q_no_bos], device=device, dtype=torch.long)
            out = self.backbone(input_ids=q_t, past_key_values=dc, use_cache=True)
            extra = len(compiled["pref_full_ids"][ep_indices[0]]) - (1 if bos_id is not None else 0)
            return _extract_logits(out), max(0, extra)

        # Multi-episode chain or non-DynamicCache model (`_SmokeHFModel`): build combined prefix cleanly
        pref: List[int] = list(compiled["kv_header_ids"])
        for ei in ep_indices:
            pref.extend(compiled["fact_line_ids"][ei])
        pref.extend(compiled["kv_sep_ids"])
        full = pref + q_no_bos
        inp = torch.tensor([full], device=device, dtype=torch.long)
        out = self.backbone(inp)
        lg = _extract_logits(out)[:, len(pref) :, :]
        extra = len(pref) - (1 if bos_id is not None else 0)
        return lg, max(0, extra)

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compiled: Optional[Dict[str, Any]] = None,
        sign_flip: bool = False,
        mode: str = "legible",
    ) -> Dict[str, Any]:
        if mode == "bridge_kv" and compiled is not None and float(compiled["g_ext"].item()) > 0.0 and input_ids.shape[0] == 1 and "pref_full_ids" in compiled:
            eps, reason = self.select_virtual_kv_episodes(input_ids[0].tolist(), compiled, input_ids.device)
            if eps:
                lg, extra = self._run_with_virtual_kv_ids(input_ids[0].tolist(), eps, compiled, input_ids.device, sign_flip=sign_flip)
                return {"logits": lg, "mode_used": f"virtual_kv:{reason}", "extra_kv_tokens": extra}
        return self._forward_single_pass_probed(
            input_ids, compiled=compiled, attention_mask=attention_mask, sign_flip=sign_flip,
            allow_stage_a=(mode == "legible"), allow_stage_c=(mode == "legible"),
        )

    @torch.no_grad()
    def score_target_logprob(
        self,
        prompt_ids: Sequence[int],
        tgt_ids: Sequence[int],
        device: torch.device,
        amp: bool,
        compiled: Optional[Dict[str, Any]] = None,
        mode: str = "legible",
        sign_flip: bool = False,
    ) -> Tuple[float, int]:
        """Length-normalized target log-prob + virtual-KV token count (`0` for `legible` & all non-triggered queries)."""
        p_l, t_l = list(prompt_ids), list(tgt_ids)
        if mode == "bridge_kv" and compiled is not None and float(compiled["g_ext"].item()) > 0.0 and "pref_full_ids" in compiled:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                eps, _ = self.select_virtual_kv_episodes(p_l, compiled, device)
                if eps:
                    lg, extra = self._run_with_virtual_kv_ids(p_l + t_l, eps, compiled, device, sign_flip=sign_flip)
                    # `lg` aligns with `q_no_bos` (which removed leading BOS if present); index from the end!
                    t_len = len(t_l)
                    lp = F.log_softmax(lg[0, -t_len - 1 : -1].float(), dim=-1)
                    tgt_t = torch.tensor(t_l, device=device, dtype=torch.long)
                    return float(lp.gather(-1, tgt_t.unsqueeze(-1)).squeeze(-1).mean().item()), extra

        full = torch.tensor([p_l + t_l], device=device, dtype=torch.long)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            res = self._forward_single_pass_probed(
                full, compiled=compiled, sign_flip=sign_flip,
                allow_stage_a=(mode == "legible"), allow_stage_c=(mode == "legible"),
                prompt_len=len(p_l),
            )
            lg = res["logits"]
        s = len(p_l) - 1
        lp = F.log_softmax(lg[0, s : s + len(t_l)].float(), dim=-1)
        tgt_t = torch.tensor(t_l, device=device, dtype=torch.long)
        return float(lp.gather(-1, tgt_t.unsqueeze(-1)).squeeze(-1).mean().item()), 0

    @torch.no_grad()
    def greedy_generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        device: torch.device,
        amp: bool = False,
        compiled: Optional[Dict[str, Any]] = None,
        mode: str = "legible",
        sign_flip: bool = False,
        eos_id: Optional[int] = None,
    ) -> Tuple[List[int], int]:
        """Multi-token greedy decode returning `(generated_ids, virtual_kv_tokens_used)`."""
        p_l = list(prompt_ids)
        p_len = len(p_l)
        if mode == "bridge_kv" and compiled is not None and float(compiled["g_ext"].item()) > 0.0 and "pref_full_ids" in compiled:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                eps, _ = self.select_virtual_kv_episodes(p_l, compiled, device)
            if eps and not sign_flip:
                cur_q = list(p_l)
                gen: List[int] = []
                extra_used = 0
                for _ in range(max_new_tokens):
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg, extra_used = self._run_with_virtual_kv_ids(cur_q, eps, compiled, device, sign_flip=False)
                    nxt = int(lg[0, -1].argmax(dim=-1).item())
                    gen.append(nxt)
                    if eos_id is not None and nxt == eos_id:
                        break
                    cur_q.append(nxt)
                return gen, extra_used

        cur = torch.tensor([p_l], device=device, dtype=torch.long)
        gen = []
        for _ in range(max_new_tokens):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = self._forward_single_pass_probed(
                    cur, compiled=compiled, sign_flip=sign_flip,
                    allow_stage_a=(mode == "legible"), allow_stage_c=(mode == "legible"),
                    prompt_len=p_len,
                )["logits"][0, -1]
            nxt = int(lg.argmax(dim=-1).item())
            gen.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        return gen, 0
