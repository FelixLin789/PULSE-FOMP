"""FOMP-v4.2: Pure-Activation 3-Route Laminar Plasticity (`0` Extra Prompt/KV Tokens, `0%` MQuAKE/ZsRE Train).

Unifies three strictly disjoint pure-activation execution routes inside a single frozen causal LLM:
  1. **Route 1 — Single-Hop & Locality (`mh_prob <= mh_gate_tau`)**:
     Executes the **100% unmodified `FOMP-v3.6-ZeroMQ` (`self.direct`)** single-pass fast path (`0.000000`
     bit-for-bit logit diff vs `v3.6`, `0` extra tokens, `~0.11s` compile, zero single-hop regression!).
  2. **Route 2 — `Hop-1 Edited` Compositional Bridge (`mh_prob > mh_gate_tau` AND `g_span > span_gate_min`)**:
     Read-Equivalent convex mid-layer entity replacement across `bridge_inject_layers` (`L10, 13, 16, 19`),
     allowing upper layers (`L20..31`) to resolve the outer Hop-2 predicate natively (`0` extra tokens!).
  3. **Route 3 — `Hop-2 Edited` Logit-Lens Implicit Secondary Retrieval (`mh_prob > mh_gate_tau` AND `g_span <= span_gate_min`)**:
     Once the frozen backbone's middle-to-late layers (`latent_probe_layers`, e.g. `L16, 19, 22, 25`) latently
     resolve the unmentioned Hop-1 bridge entity at its mid-sentence token position, projects those hidden states
     through a lightweight residual Tuned Lens + the backbone's own frozen `final_norm + lm_head` (full 128K
     open-vocabulary space, ZERO unseen-entity projection bottleneck!), cross-locks against the prompt's outer
     relation (`max(q_rel_tail, q_hop2_outer) . k_fam[m]`), and injects the matched Hop-2 edit `m*` across all
     5 `v3.6` layers `(16, 19, 23, 27, 31)` at `\nAnswer:` (`0` extra tokens!).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fomp_v3 import (
    FOMPv3Config,
    FOMPv3Module,
    _EarlyStopSignal,
    _FastRMSNorm,
    _causal_window_mean,
    _extract_logits,
    _extract_tail_window,
    _get_decoder_layers,
    _get_out_embedding_weight,
    _mark_no_muon,
)

# Strictly closed-class English function words / QA boilerplate excluded from latent subject token candidates
_LATENT_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "by", "for", "to", "from", "with", "as", "into", "onto",
    "is", "was", "are", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did",
    "that", "this", "these", "those", "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    "and", "or", "but", "not", "no", "nor", "if", "then", "than", "so", "it", "its", "he", "him", "his",
    "she", "her", "hers", "they", "them", "their", "we", "our", "you", "your", "question", "answer", "s",
})


def _project_logit_lens(backbone: nn.Module, h: torch.Tensor) -> torch.Tensor:
    """Project hidden states `[B, T, d]` through frozen backbone's own final RMSNorm + `lm_head` -> `[B, T, V]`."""
    model_sub = getattr(backbone, "model", None)
    fnorm = getattr(model_sub, "norm", None) if model_sub is not None else getattr(backbone, "norm", None)
    if fnorm is not None:
        h = fnorm(h)
    lm_head = getattr(backbone, "lm_head", None)
    if lm_head is not None:
        return lm_head(h)
    w_out = _get_out_embedding_weight(backbone)
    return F.linear(h.to(w_out.dtype), w_out)


@dataclass
class FOMPv4Config:
    d_model: int = 4096
    bridge_read_layer: int = 10
    bridge_inject_layers: Tuple[int, ...] = (10, 13, 16, 19)
    read_layer: int = 16
    inject_layers: Tuple[int, ...] = (16, 19, 23, 27, 31)
    latent_probe_layers: Tuple[int, ...] = (16, 19, 22, 25)
    key_dim: int = 512
    ent_dim: int = 384
    val_dim: int = 512
    bridge_rank: int = 256
    lens_rank: int = 64
    mh_hidden: int = 256
    max_tgt_tokens: int = 8
    max_latent_subj_cands: int = 6
    top_k: int = 2
    rel_hidden: int = 128
    span_window: int = 14
    init_span_rel_tau: float = 0.56
    min_span_rel_tau: float = 0.48
    init_span_ent_tau: float = 0.56
    min_span_ent_tau: float = 0.48
    mh_gate_tau: float = 0.80
    span_gate_min: float = 0.20
    hop2_ll_tau: float = -4.2
    hop2_rel_tau: float = 0.28
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
        probes = tuple(sorted(set([
            rl,
            min(n_layers - 1, rl + step_b),
            min(n_layers - 1, rl + 2 * step_b),
            min(n_layers - 1, rl + 3 * step_b),
        ])))
        return cls(
            d_model=d_model,
            bridge_read_layer=brl,
            bridge_inject_layers=binj,
            read_layer=rl,
            inject_layers=inj,
            latent_probe_layers=probes,
            key_dim=(512 if d_model >= 3072 else 384),
            ent_dim=(384 if d_model >= 3072 else 256),
            val_dim=512,
            bridge_rank=(256 if d_model >= 3072 else 128),
            lens_rank=(64 if d_model >= 3072 else 32),
            mh_hidden=(256 if d_model >= 3072 else 128),
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
    """Format a concise factual statement for optional Virtual-KV ablation."""
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
    """Pure-Activation 3-Route Laminar Plasticity Module (`Single-Hop v3.6` + `Hop-1 Legible` + `Hop-2 Logit-Lens`)."""

    def __init__(self, config: FOMPv4Config) -> None:
        super().__init__()
        self.config = config
        self.direct = FOMPv3Module(config.to_v3_config())

        self.n_bridge_inj = len(config.bridge_inject_layers)
        self.bridge_inj_map = {l_idx: i for i, l_idx in enumerate(config.bridge_inject_layers)}
        self.n_probes = len(config.latent_probe_layers)

        # --- Route 2 (Stage A): Mid-Sentence / Nested-NP Span Dual-Lock + Read-Equivalent Entity Replacement ---
        self.span_norm = _FastRMSNorm(config.d_model)
        self.span_rel_score = nn.Linear(config.d_model, 1, bias=True)
        self.span_rel_proj = nn.Linear(2 * config.d_model, config.key_dim, bias=False)
        self.span_rel_mlp = nn.Linear(config.key_dim, config.key_dim, bias=True)

        self.span_ent_score = nn.Linear(config.d_model, 1, bias=True)
        self.span_ent_proj = nn.Linear(2 * config.d_model, config.ent_dim, bias=False)
        self.span_ent_mlp = nn.Linear(config.ent_dim, config.ent_dim, bias=True)

        self.span_rel_tau = nn.Parameter(torch.tensor(float(config.init_span_rel_tau)))
        self.span_ent_tau = nn.Parameter(torch.tensor(float(config.init_span_ent_tau)))

        self.bridge_down = nn.Linear(2 * config.d_model, config.bridge_rank, bias=False)
        self.bridge_ups = nn.ModuleList([nn.Linear(config.bridge_rank, config.d_model, bias=False) for _ in range(self.n_bridge_inj)])
        self.bridge_alpha_logit = nn.Parameter(torch.full((self.n_bridge_inj,), 1.4))
        self.bridge_tok_spread = nn.Parameter(torch.tensor([-1.8, -0.6, 2.5]))

        # --- Strict Compositional Multi-Hop Gate (`read_layer`) ---
        # Bias `-3.0` (`p_mh = 0.047 << 0.80`) guarantees step-0 bit-for-bit identity with `v3.6`!
        self.mh_norm = _FastRMSNorm(config.d_model)
        self.mh_tok_score = nn.Linear(config.d_model, 1, bias=True)
        self.mh_fc1 = nn.Linear(5 * config.d_model, config.mh_hidden, bias=True)
        self.mh_fc2 = nn.Linear(config.mh_hidden, 1, bias=True)

        # --- Route 3 (`Hop-2`): Low-Rank Residual Tuned-Lens Adapters + Outer-Relation Projector ---
        # Zero-init `lens_ups` => exact raw Logit Lens at step 0; self-supervised alignment to L31 on `cf_train`!
        self.lens_norms = nn.ModuleList([_FastRMSNorm(config.d_model) for _ in range(self.n_probes)])
        self.lens_downs = nn.ModuleList([nn.Linear(config.d_model, config.lens_rank, bias=False) for _ in range(self.n_probes)])
        self.lens_ups = nn.ModuleList([nn.Linear(config.lens_rank, config.d_model, bias=False) for _ in range(self.n_probes)])

        # Outer-relation projector ONLY invoked inside Route 3 (`mh_prob > 0.80 & g_span <= 0.20`), NEVER on single-hop!
        self.hop2_out_norm = _FastRMSNorm(config.d_model)
        self.hop2_out_score = nn.Linear(config.d_model, 1, bias=True)
        self.hop2_out_proj = nn.Linear(3 * config.d_model, config.key_dim, bias=False)
        self.hop2_out_mlp = nn.Linear(config.key_dim, config.key_dim, bias=True)

        # Calibrated Logit-Lens log-prob & outer-relation thresholds (registered as buffers so checkpoints persist them!)
        self.register_buffer("hop2_ll_tau_buf", torch.tensor(float(config.hop2_ll_tau), dtype=torch.float32))
        self.register_buffer("hop2_rel_tau_buf", torch.tensor(float(config.hop2_rel_tau), dtype=torch.float32))

        for m in [
            self.span_rel_score, self.span_rel_proj, self.span_rel_mlp,
            self.span_ent_score, self.span_ent_proj, self.span_ent_mlp,
            self.bridge_down, *self.bridge_ups,
            self.mh_tok_score, self.mh_fc1, self.mh_fc2,
            *self.lens_downs, *self.lens_ups,
            self.hop2_out_score, self.hop2_out_proj, self.hop2_out_mlp,
        ]:
            _mark_no_muon(m)
        for p in (self.span_rel_tau, self.span_ent_tau, self.bridge_alpha_logit, self.bridge_tok_spread):
            p._no_muon = True  # type: ignore[attr-defined]
        self._init_bridge_weights()

    def _init_bridge_weights(self) -> None:
        for lin in (self.span_rel_proj, self.span_ent_proj, self.bridge_down, self.hop2_out_proj, *self.lens_downs):
            nn.init.normal_(lin.weight, 0.0, 0.02)
        for mlp in (self.span_rel_mlp, self.span_ent_mlp):
            nn.init.normal_(mlp.weight, 0.0, 0.01)
            nn.init.zeros_(mlp.bias)
        nn.init.zeros_(self.hop2_out_mlp.weight)
        nn.init.zeros_(self.hop2_out_mlp.bias)
        for sc in (self.span_rel_score, self.span_ent_score, self.mh_tok_score, self.hop2_out_score):
            nn.init.normal_(sc.weight, 0.0, 0.02)
            nn.init.zeros_(sc.bias)
        nn.init.normal_(self.mh_fc1.weight, 0.0, 0.02)
        nn.init.zeros_(self.mh_fc1.bias)
        nn.init.zeros_(self.mh_fc2.weight)
        nn.init.constant_(self.mh_fc2.bias, -3.0)
        for up in (*self.bridge_ups, *self.lens_ups):
            nn.init.zeros_(up.weight)

    def bridge_parameters(self) -> List[nn.Parameter]:
        """Return strictly the Route 2/3 & multi-hop gate parameters (`self.direct` remains 100% frozen)."""
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
        u_rel = self._pool_window(h_win, win_mask, self.span_rel_score)
        u_ent = self._pool_window(h_win, win_mask, self.span_ent_score)
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

    def apply_tuned_lens(self, h: torch.Tensor, probe_idx: int) -> torch.Tensor:
        """Zero-init low-rank residual Tuned-Lens translator aligning mid-layer `h` to final `lm_head` space."""
        w_dt = self.lens_downs[probe_idx].weight.dtype
        hn = self.lens_norms[probe_idx](h.to(w_dt))
        delta = self.lens_ups[probe_idx](F.silu(self.lens_downs[probe_idx](hn)))
        return (h.float() + delta.float()).to(h.dtype)

    def project_hop2_outer_rel(
        self, h_read_seq: torch.Tensor, seq_mask: Optional[torch.Tensor] = None, prompt_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract outer question relation in `direct.k_fam` space (invoked ONLY inside Route 3 Hop-2!)."""
        if prompt_len is not None and prompt_len < h_read_seq.shape[1]:
            h_read_seq = h_read_seq[:, :prompt_len, :]
            if seq_mask is not None:
                seq_mask = seq_mask[:, :prompt_len]
        bsz, t_len, _ = h_read_seq.shape
        hn = self.hop2_out_norm(h_read_seq.to(self.hop2_out_proj.weight.dtype))
        s = self.hop2_out_score(hn).float().squeeze(-1)
        if seq_mask is not None:
            s = s.masked_fill(seq_mask <= 0, -1e9)
            lens = seq_mask.long().sum(dim=1).clamp_min(1)
        else:
            lens = torch.full((bsz,), t_len, device=h_read_seq.device, dtype=torch.long)
        w = F.softmax(s, dim=-1).unsqueeze(-1)
        u_pool = (w * hn.float()).sum(dim=1).to(hn.dtype)

        half_lens = ((lens + 1) // 2).clamp_min(1)
        pos_idx = torch.arange(t_len, device=h_read_seq.device).unsqueeze(0)
        pref_m = (pos_idx < half_lens.unsqueeze(1)).float()
        if seq_mask is not None:
            pref_m = pref_m * (seq_mask > 0).float()
        u_pref = ((hn.float() * pref_m.unsqueeze(-1)).sum(dim=1) / pref_m.sum(dim=1, keepdim=True).clamp_min(1.0)).to(hn.dtype)

        last_idx = (lens - 1).clamp_min(0)
        ar = torch.arange(bsz, device=hn.device)
        u_last = hn[ar, last_idx]
        h_ctx = _causal_window_mean(h_read_seq, self.direct.config.key_ctx_window)
        q_tail = self.direct.project_key(h_read_seq[ar, last_idx], h_ctx[ar, last_idx])

        feat = torch.cat([u_pool, u_pref, u_last], dim=-1)
        if self.training:
            feat = F.dropout(feat, p=0.10, training=True)
        delta_out = self.hop2_out_mlp(F.silu(self.hop2_out_proj(feat))).float()
        return F.normalize(q_tail.float() + delta_out, p=2, dim=-1, eps=1e-6)

    def compute_multihop_logits(
        self, h_read_seq: torch.Tensor, seq_mask: Optional[torch.Tensor] = None, prompt_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict compositional multi-hop log-odds `[B]` from 3-segment + salience features at `read_layer`."""
        if prompt_len is not None and prompt_len < h_read_seq.shape[1]:
            h_read_seq = h_read_seq[:, :prompt_len, :]
            if seq_mask is not None:
                seq_mask = seq_mask[:, :prompt_len]
        bsz, t_len, _ = h_read_seq.shape
        hn = self.mh_norm(h_read_seq.to(self.mh_fc1.weight.dtype))
        s = self.mh_tok_score(hn).float().squeeze(-1)
        if seq_mask is not None:
            s = s.masked_fill(seq_mask <= 0, -1e9)
            lens = seq_mask.long().sum(dim=1).clamp_min(1)
            valid_f = (seq_mask > 0).float()
        else:
            lens = torch.full((bsz,), t_len, device=h_read_seq.device, dtype=torch.long)
            valid_f = torch.ones((bsz, t_len), device=h_read_seq.device, dtype=torch.float32)

        w = F.softmax(s, dim=-1).unsqueeze(-1)
        u_pool = (w * hn.float()).sum(dim=1).to(hn.dtype)

        pos = torch.arange(t_len, device=h_read_seq.device).unsqueeze(0)
        c1 = (lens // 3).clamp_min(1).unsqueeze(1)
        c2 = ((2 * lens) // 3).clamp_min(1).unsqueeze(1)
        l_col = lens.unsqueeze(1)

        def _seg_mean(m_bool: torch.Tensor) -> torch.Tensor:
            mf = m_bool.float() * valid_f
            return ((hn.float() * mf.unsqueeze(-1)).sum(dim=1) / mf.sum(dim=1, keepdim=True).clamp_min(1.0)).to(hn.dtype)

        u_s1 = _seg_mean(pos < c1)
        u_s2 = _seg_mean((pos >= c1) & (pos < c2))
        u_s3 = _seg_mean((pos >= c2) & (pos < l_col))
        u_last = hn[torch.arange(bsz, device=hn.device), (lens - 1).clamp_min(0)]

        feat = torch.cat([u_pool, u_s1, u_s2, u_s3, u_last], dim=-1)
        h_mid = F.silu(self.mh_fc1(feat))
        if self.training:
            h_mid = F.dropout(h_mid, p=0.20, training=True)
        return self.mh_fc2(h_mid).float().squeeze(-1)

    def arbitrate_routes(
        self,
        span_rt: Dict[str, torch.Tensor],
        h_read_seq: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        prompt_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Strictly partition queries into:
          - `is_mh == False`             -> Route 1: 100% unmodified `FOMP-v3.6` (`0.000000` diff!)
          - `is_mh & g_sp > span_min`    -> Route 2: `Hop-1` Read-Equivalent Legible Bridge
          - `is_mh & g_sp <= span_min`   -> Route 3: `Hop-2` Logit-Lens Implicit Secondary Retrieval"""
        g_sp = span_rt["peak_gate"]
        mh_logits = self.compute_multihop_logits(h_read_seq, seq_mask=seq_mask, prompt_len=prompt_len)
        mh_prob = torch.sigmoid(mh_logits)
        is_mh = mh_prob > float(self.config.mh_gate_tau)
        need_hop1 = is_mh & (g_sp > float(self.config.span_gate_min))
        need_hop2_probe = is_mh & (~need_hop1)
        return {
            "is_mh": is_mh,
            "need_hop1": need_hop1,
            "need_hop2_probe": need_hop2_probe,
            "mh_prob": mh_prob,
            "mh_logits": mh_logits,
            "g_sp": g_sp,
        }

    def compile_v4_states(
        self,
        direct_compiled: Dict[str, torch.Tensor],
        h_span_win: torch.Tensor,
        m_span_win: torch.Tensor,
        h_tgt_bridge: torch.Tensor,
        k_tgt_ent: Optional[torch.Tensor] = None,
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
        h_bridge_seq: torch.Tensor,
        compiled: Dict[str, Any],
        seq_mask: Optional[torch.Tensor] = None,
        tail_guard_tokens: int = 3,
    ) -> Dict[str, torch.Tensor]:
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
        g_raw = self._hill(s_rel, tau_sr) * self._hill(s_ent, tau_se)

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
        g_masked = g_raw * pos_mask.unsqueeze(-1)

        s_joint = 0.5 * (s_rel + s_ent)
        score_tm = g_masked + 1e-3 * s_joint * pos_mask.unsqueeze(-1)
        flat_idx = score_tm.reshape(bsz, -1).argmax(dim=-1)
        m_cnt = k_srel.shape[1]
        peak_t = flat_idx // m_cnt
        peak_m = flat_idx % m_cnt
        ar = torch.arange(bsz, device=h_bridge_seq.device)
        peak_gate = g_masked[ar, peak_t, peak_m]
        peak_sim = s_joint[ar, peak_t, peak_m]

        spread = torch.sigmoid(self.bridge_tok_spread.float())
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
        h_cur: torch.Tensor,
        span_rt: Dict[str, torch.Tensor],
        compiled: Dict[str, Any],
        bridge_inj_idx: int,
        sign_flip: bool = False,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, _, _ = h_cur.shape
        hb = compiled["h_tgt_bridge"]
        if hb.shape[0] == 1 and bsz > 1:
            hb = hb.expand(bsz, -1, -1, -1)
        ar = torch.arange(bsz, device=h_cur.device)
        h_ent = hb[ar, span_rt["peak_m"], bridge_inj_idx, :]

        cur_rms = h_cur.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        ent_rms = h_ent.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().unsqueeze(1)
        h_ent_matched = h_ent.unsqueeze(1) * (cur_rms / ent_rms)

        cat_in = torch.cat([
            self.span_norm(h_cur.to(self.bridge_down.weight.dtype)),
            self.span_norm(h_ent_matched.to(self.bridge_down.weight.dtype)),
        ], dim=-1)
        z_b = F.silu(self.bridge_down(cat_in))
        delta_adapt = self.bridge_ups[bridge_inj_idx](z_b).float() * cur_rms

        alpha = torch.sigmoid(self.bridge_alpha_logit[bridge_inj_idx].float())
        env = span_rt["tok_env"].unsqueeze(-1)
        if active_mask is not None:
            env = env * active_mask.float().view(bsz, 1, 1)
        w = (alpha * env).clamp(0.0, 1.0)
        target_state = h_ent_matched + delta_adapt
        if sign_flip:
            h_out = h_cur.float() - w * target_state
        else:
            h_out = (1.0 - w) * h_cur.float() + w * target_state
        return h_out.to(h_cur.dtype)

    def build_forced_episode_latch(
        self,
        ep_idx: int,
        seq_len: int,
        p_end: int,
        compiled: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build exact `v3.6` `(latched_v, latched_e)` anchored at `p_end` for Hop-2 retrieved episode `ep_idx`."""
        kv = self.config.max_tgt_tokens
        v_ep = compiled["v_bank"][0, ep_idx]              # [Kv, val_dim]
        e_ep = compiled["e_bank"][0, ep_idx]              # [Kv, d]
        m_ep = compiled["tgt_mask"][0, ep_idx]            # [Kv]
        latched_v = torch.zeros((1, seq_len, self.config.val_dim), device=device, dtype=torch.float32)
        latched_e = torch.zeros((1, seq_len, self.config.d_model), device=device, dtype=torch.float32)
        for lag in range(kv):
            pos = p_end + lag
            if pos >= seq_len:
                break
            mf = m_ep[lag]
            if float(mf.item()) > 0.0:
                latched_v[0, pos] = v_ep[lag] * mf
                latched_e[0, pos] = e_ep[lag] * mf
        return latched_v, latched_e


def _extract_salient_subject_candidates(
    cue_ids_list: Sequence[Sequence[int]],
    salience_scores: List[torch.Tensor],
    bos_id: Optional[int],
    tokenizer: Optional[Any],
    max_cands: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    """Extract softmax-salient non-stopword cue subject tokens + BPE word-start/space variants (`0` subject cheat!)."""
    m_cnt = len(cue_ids_list)
    tid_mat = torch.zeros((m_cnt, max_cands), device=device, dtype=torch.long)
    msk_mat = torch.zeros((m_cnt, max_cands), device=device, dtype=torch.float32)
    str_lists: List[List[str]] = []

    for m, (c_ids, sal) in enumerate(zip(cue_ids_list, salience_scores)):
        j0 = 1 if (bos_id is not None and len(c_ids) > 1 and c_ids[0] == bos_id) else 0
        pieces = [tokenizer.decode([int(tid)]) if tokenizer is not None else str(int(tid)) for tid in c_ids]
        cleans = [p.strip(" .,?!:;\"'()[]\n\t").lower() for p in pieces]
        has_space_bpe = (tokenizer is not None) and any(p.startswith(" ") for p in pieces[j0:])

        sal_sub = sal[j0 : len(c_ids)].float()
        w_sub = F.softmax(sal_sub, dim=0) if sal_sub.numel() > 0 else torch.ones((1,), dtype=torch.float32)

        scored: List[Tuple[float, int]] = []
        fallback: List[Tuple[float, int]] = []
        for j in range(j0, len(c_ids)):
            wj = float(w_sub[j - j0].item()) if (j - j0) < w_sub.numel() else 0.0
            fallback.append((wj, j))
            if tokenizer is None or (len(cleans[j]) >= 2 and cleans[j] not in _LATENT_STOPWORDS):
                scored.append((wj, j))

        chosen_src = scored if scored else fallback
        chosen_src.sort(key=lambda x: x[0], reverse=True)
        # Keep top-1 always + at most 1 more token with salience >= 15% of peak (avoids polluting with relation words!)
        top_w = chosen_src[0][0] if chosen_src else 0.0
        kept_js = [chosen_src[0][1]] if chosen_src else []
        for wj, j in chosen_src[1:2]:
            if wj >= 0.15 * top_w:
                kept_js.append(j)

        cands_tids: List[int] = []
        cands_strs: List[str] = []
        seen_tids: set = set()

        def _add_pos(pos_j: int) -> None:
            tid = int(c_ids[pos_j])
            piece = pieces[pos_j]
            clean = cleans[pos_j]
            variants: List[int] = [tid]
            if tokenizer is not None and len(clean) >= 2:
                is_word_start = piece.startswith(" ") or (pos_j <= j0)
                if is_word_start:
                    raw_word = piece.strip(" .,?!:;\"'()[]\n\t")
                    for form in (" " + raw_word, raw_word):
                        enc = tokenizer.encode(form, add_special_tokens=False)
                        if enc:
                            variants.append(int(enc[0]))
            for vt in variants:
                if vt not in seen_tids and len(cands_tids) < max_cands:
                    seen_tids.add(vt)
                    cands_tids.append(vt)
                    cands_strs.append(clean)

        for j in kept_js:
            # For continuation BPE subwords (e.g. `"olini"` in `[" Muss", "olini"]`), add word-start token FIRST!
            if has_space_bpe and j > j0 and not pieces[j].startswith(" "):
                k = j - 1
                while k > j0 and not pieces[k].startswith(" "):
                    k -= 1
                if len(cleans[k]) >= 2 and cleans[k] not in _LATENT_STOPWORDS:
                    _add_pos(k)
            _add_pos(j)

        for s_i, vt in enumerate(cands_tids):
            tid_mat[m, s_i] = vt
            msk_mat[m, s_i] = 1.0
        str_lists.append(cands_strs)

    return tid_mat, msk_mat, str_lists


class FOMPv4BackboneWrapper(nn.Module):
    """Unified wrapper executing Pure-Activation 3-Route Plasticity (`Single-Hop v3.6` + `Hop-1 Legible` + `Hop-2 Logit-Lens`)."""

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
        target_layers: Sequence[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        t_set = set(target_layers)
        max_l = max(t_set)
        captured: Dict[int, torch.Tensor] = {}
        handles = []
        for li in sorted(t_set):
            def _mk(idx_l: int):
                def _pre(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
                    h = kwargs["hidden_states"] if len(args) == 0 else args[0]
                    captured[idx_l] = h.detach()
                    if idx_l == max_l:
                        raise _EarlyStopSignal(captured[idx_l])
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
    def capture_probes_and_final(
        self,
        input_ids: torch.Tensor,
        probe_layers: Sequence[int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Single frozen-backbone pass capturing all `probe_layers` inputs PLUS final pre-norm layer output."""
        captured: Dict[int, torch.Tensor] = {}
        final_box: List[torch.Tensor] = []
        handles = []
        for li in sorted(set(probe_layers)):
            def _mk(idx_l: int):
                def _pre(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
                    h = kwargs["hidden_states"] if len(args) == 0 else args[0]
                    captured[idx_l] = h.detach()
                return _pre
            handles.append(self.layers[li].register_forward_pre_hook(_mk(li), with_kwargs=True))

        def _post(_mod: nn.Module, _args: Tuple[Any, ...], output: Any) -> None:
            h_out = output[0] if isinstance(output, tuple) else output
            final_box.append(h_out.detach())

        handles.append(self.layers[-1].register_forward_hook(_post))
        was_tr = self.backbone.training
        self.backbone.eval()
        try:
            if attention_mask is not None and hasattr(self.backbone, "model"):
                self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                self.backbone(input_ids)
        finally:
            for hd in handles:
                hd.remove()
            if was_tr:
                self.backbone.train()
        return captured, final_box[0]

    @torch.no_grad()
    def lookup_target_embeddings(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        w_out = _get_out_embedding_weight(self.backbone)
        return F.embedding(tgt_ids.to(w_out.device), w_out).detach()

    @torch.no_grad()
    def _build_kv_snapshot(self, pref_ids: List[int], device: torch.device) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
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

    @torch.no_grad()
    def _get_or_build_kv_snapshot(
        self, ep_idx: int, compiled: Dict[str, Any], device: torch.device
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        snaps = compiled.get("kv_snaps")
        if snaps is not None and snaps[ep_idx] is not None:
            return snaps[ep_idx]
        lazy_map: Dict[int, Any] = compiled.setdefault("_lazy_kv_snaps", {})
        if ep_idx not in lazy_map and "pref_full_ids" in compiled:
            lazy_map[ep_idx] = self._build_kv_snapshot(compiled["pref_full_ids"][ep_idx], device)
        return lazy_map.get(ep_idx)

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
        """Single-pass compilation for all 3 routes (`v3.6 Direct` + `Hop-1 Legible` + `Hop-2 Logit-Lens`)."""
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
        sal_lists: List[torch.Tensor] = []

        cue_layers = [cfg.bridge_read_layer, cfg.read_layer]
        tgt_layers = sorted(set([cfg.read_layer, *cfg.bridge_inject_layers]))

        for s0 in range(0, m_cnt, chunk_size):
            c_sub = cue_ids_list[s0 : s0 + chunk_size]
            sp_sub = span_src_list[s0 : s0 + chunk_size]
            t_sub = tgt_ids_list[s0 : s0 + chunk_size]
            n_sub = len(c_sub)
            ar_s = torch.arange(n_sub, device=device)

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

            # Reuse `v3.6`'s trained entity salience scorer to rank subject tokens for Route 3 Logit-Lens retrieval!
            with torch.no_grad():
                hn_sal = self.fomp.direct.norm(self.fomp.direct._match_dtype(h_rl))
                sal_b = self.fomp.direct.ent_score(hn_sal).float().squeeze(-1)
                for i, c in enumerate(c_sub):
                    sal_lists.append(sal_b[i, : len(c)].cpu())

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

            off = 1 if bos_id is not None else 0
            t_in = torch.full((n_sub, kv + off), pad_id, device=device, dtype=torch.long)
            t_in_m = torch.zeros((n_sub, kv + off), device=device, dtype=torch.long)
            if bos_id is not None:
                t_in[:, 0] = bos_id
                t_in_m[:, 0] = 1
            t_ids_c = torch.full((n_sub, kv), pad_id, device=device, dtype=torch.long)
            t_m_c = torch.zeros((n_sub, kv), device=device, dtype=torch.float32)
            t_last = torch.zeros((n_sub,), device=device, dtype=torch.long)
            for i, t in enumerate(t_sub):
                t_cl = list(t)[:kv] if len(t) > 0 else [pad_id]
                ln = len(t_cl)
                tt = torch.tensor(t_cl, device=device, dtype=torch.long)
                t_in[i, off : off + ln] = tt
                t_in_m[i, off : off + ln] = 1
                t_ids_c[i, :ln] = tt
                t_m_c[i, :ln] = 1.0
                t_last[i] = off + ln - 1

            # Encode `t_ids_c` WITHOUT `bos_id` for `self.direct` so `v_bank` matches FOMP-v3.6 bit-for-bit!
            cap_t_dir = self.encode_layers_batch(t_ids_c, [cfg.read_layer], attention_mask=t_m_c.long())
            ht_p.append(cap_t_dir[cfg.read_layer])
            et_p.append(self.lookup_target_embeddings(t_ids_c))
            tm_p.append(t_m_c)

            cap_t = self.encode_layers_batch(t_in, tgt_layers, attention_mask=t_in_m)
            b_states = torch.stack([cap_t[li][ar_s, t_last] for li in cfg.bridge_inject_layers], dim=1)
            htb_p.append(b_states)

            tw_e, tw_m = _extract_tail_window(cap_t[cfg.read_layer], t_last, win_e)
            kte = self.fomp.direct.project_entity_key_from_window(tw_e, tw_m, h_last=cap_t[cfg.read_layer][ar_s, t_last])
            kte_p.append(kte)

        dir_comp = self.fomp.direct.compile_from_states(
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
            dir_comp,
            torch.cat(hsw_p, dim=0),
            torch.cat(msw_p, dim=0),
            torch.cat(htb_p, dim=0),
            k_tgt_ent=torch.cat(kte_p, dim=0),
        )
        tid_mat, msk_mat, str_lists = _extract_salient_subject_candidates(
            cue_ids_list, sal_lists, bos_id, tokenizer, cfg.max_latent_subj_cands, device
        )
        comp["latent_subj_tids"] = tid_mat
        comp["latent_subj_mask"] = msk_mat
        comp["latent_subj_strs"] = str_lists
        comp["tokenizer"] = tokenizer
        comp["bos_id"] = bos_id
        comp["_lazy_kv_snaps"] = {}

        if fact_texts is not None and tokenizer is not None:
            hdr = "Updated facts (follow these strictly over prior knowledge):\n"
            hdr_ids = ([bos_id] if bos_id is not None else []) + tokenizer.encode(hdr, add_special_tokens=False)
            sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)
            f_line_ids = [tokenizer.encode(ft, add_special_tokens=False) for ft in fact_texts]
            pref_full_ids = [hdr_ids + fl + sep_ids for fl in f_line_ids]
            comp["kv_header_ids"] = hdr_ids
            comp["kv_sep_ids"] = sep_ids
            comp["fact_line_ids"] = f_line_ids
            comp["pref_full_ids"] = pref_full_ids
            comp["kv_snaps"] = [self._build_kv_snapshot(p_f, device) for p_f in pref_full_ids] if (precompile_kv and float(g_ext) > 0.0) else None
        return comp

    @torch.no_grad()
    def route_hop2_logit_lens(
        self,
        prompt_ids_1d: Sequence[int],
        h_probes: Dict[int, torch.Tensor],
        h_read_prompt: torch.Tensor,
        compiled: Dict[str, Any],
    ) -> Tuple[Optional[int], float, float]:
        """Route 3: Implicit Secondary Retrieval in native 128K vocabulary space (`Logit Lens` x `Outer Relation Lock`).
        Returns `(matched_ep_idx_or_None, best_ll_logp, best_outer_rel_cos)`."""
        cfg = self.fomp.config
        if "latent_subj_tids" not in compiled or float(compiled["g_ext"].item()) == 0.0:
            return None, -100.0, 0.0

        tid_mat = compiled["latent_subj_tids"]            # [M, S]
        msk_mat = compiled["latent_subj_mask"].clone()    # [M, S]
        str_lists = compiled["latent_subj_strs"]
        tok = compiled.get("tokenizer")
        m_cnt, s_cnt = tid_mat.shape

        # Mask out any candidate token already present literally in the surface prompt -> 100% implicit bridge detection!
        q_tid_set = set(int(x) for x in prompt_ids_1d)
        q_word_set: set = set()
        if tok is not None:
            for qtid in q_tid_set:
                w_c = tok.decode([qtid]).strip(" .,?!:;\"'()[]\n\t").lower()
                if len(w_c) >= 2:
                    q_word_set.add(w_c)
        for m in range(m_cnt):
            for s_i, w_s in enumerate(str_lists[m]):
                if int(tid_mat[m, s_i].item()) in q_tid_set or (w_s and w_s in q_word_set):
                    msk_mat[m, s_i] = 0.0

        # Score candidate subject tokens across all mid-to-late probe layers & non-BOS prompt positions
        best_lp = torch.full((m_cnt, s_cnt), -100.0, device=tid_mat.device, dtype=torch.float32)
        flat_tids = tid_mat.reshape(-1)
        has_bos = compiled.get("bos_id") is not None
        for pi, l_idx in enumerate(cfg.latent_probe_layers):
            h_p = h_probes.get(l_idx)
            if h_p is None or h_p.shape[1] == 0:
                continue
            start_t = 1 if (has_bos and h_p.shape[1] > 1) else 0
            h_hat = self.fomp.apply_tuned_lens(h_p[:, start_t:, :], pi)    # [1, L', d]
            lg_v = _project_logit_lens(self.backbone, h_hat).float()       # [1, L', V]
            lp_v = F.log_softmax(lg_v, dim=-1)[0]                          # [L', V]
            gathered = lp_v[:, flat_tids].reshape(-1, m_cnt, s_cnt)        # [L', M, S]
            layer_max = gathered.max(dim=0).values                         # [M, S]
            best_lp = torch.maximum(best_lp, layer_max)

        best_lp = best_lp.masked_fill(msk_mat <= 0.0, -100.0)
        ll_score = best_lp.max(dim=-1).values                              # [M]

        # Lock 2: Outer Relation Match (`max(q_rel_tail, q_hop2_outer) . k_fam[m]`)
        k_fam = compiled["k_fam"][0]                                       # [M, key_dim]
        h_ctx = _causal_window_mean(h_read_prompt, self.fomp.direct.config.key_ctx_window)
        q_tail = self.fomp.direct.project_key(h_read_prompt[:, -1], h_ctx[:, -1])[0]
        q_out = self.fomp.project_hop2_outer_rel(h_read_prompt)[0]
        s_rel = torch.maximum(torch.mv(k_fam, q_tail), torch.mv(k_fam, q_out))  # [M]

        tau_ll = float(self.fomp.hop2_ll_tau_buf.item())
        tau_rel = float(self.fomp.hop2_rel_tau_buf.item())
        valid = (ll_score >= tau_ll) & (s_rel >= tau_rel)
        if not bool(valid.any().item()):
            best_m = int(ll_score.argmax().item())
            return None, float(ll_score[best_m].item()), float(s_rel[best_m].item())

        joint = (ll_score + 4.0 * s_rel).masked_fill(~valid, -1e9)
        pick_m = int(joint.argmax().item())
        return pick_m, float(ll_score[pick_m].item()), float(s_rel[pick_m].item())

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
                            h, state["span_rt"], compiled, self.fomp.bridge_inj_map[l_idx],
                            sign_flip=sign_flip, active_mask=active_mask,
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

    def _run_forced_hop2_episode(
        self,
        input_ids: torch.Tensor,
        ep_idx: int,
        compiled: Dict[str, Any],
        prompt_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        sign_flip: bool = False,
    ) -> torch.Tensor:
        """Run clean 5-layer `v3.6` injection locked strictly onto Route-3 retrieved Hop-2 episode `ep_idx`."""
        seq_len = input_ids.shape[1]
        p_end = min(prompt_len, seq_len) - 1
        lv, le = self.fomp.build_forced_episode_latch(ep_idx, seq_len, p_end, compiled, input_ids.device)
        handles = []
        for li in self.fomp.direct.config.inject_layers:
            inj_i = self.fomp.direct.inject_map[li]
            def _mk(ii: int):
                def _pre(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                    use_kw = len(args) == 0
                    h = kwargs["hidden_states"] if use_kw else args[0]
                    h_rms = h.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().detach()
                    delta = self.fomp.direct.compute_layer_delta(
                        lv, ii, h.dtype, sign_flip=sign_flip, latched_e=le, h_rms=h_rms,
                    )
                    h_new = h + delta
                    if use_kw:
                        nkw = dict(kwargs)
                        nkw["hidden_states"] = h_new
                        return args, nkw
                    return (h_new, *args[1:]), kwargs
                return _pre
            handles.append(self.layers[li].register_forward_pre_hook(_mk(inj_i), with_kwargs=True))
        try:
            if attention_mask is not None and hasattr(self.backbone, "model"):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                out = self.backbone(input_ids)
        finally:
            for hd in handles:
                hd.remove()
        return _extract_logits(out)

    def forward_bridge_train(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        compiled: Dict[str, Any],
        tail_guard_tokens: int = 3,
    ) -> Dict[str, Any]:
        with self._bridge_only_hooks(compiled=compiled, seq_mask=attention_mask, tail_guard_tokens=tail_guard_tokens) as st:
            if hasattr(self.backbone, "model"):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                out = self.backbone(input_ids)
        return {"logits": _extract_logits(out), "span_rt": st["span_rt"]}

    def _pick_chain_episodes(self, m1: int, compiled: Dict[str, Any]) -> List[int]:
        picked = [m1]
        if compiled.get("k_tgt_ent") is not None:
            kt = compiled["k_tgt_ent"][0, m1]
            sims_chain = torch.mv(compiled["k_ent"][0], kt)
            sims_chain[m1] = -1.0
            tau_e = float(self.fomp.direct.ent_tau.float().clamp(self.fomp.direct.config.min_ent_tau, 0.90).item())
            best_v, best_i = sims_chain.max(dim=0)
            if float(best_v.item()) >= tau_e:
                picked.append(int(best_i.item()))
        return picked

    def _forward_single_pass_probed(
        self,
        input_ids: torch.Tensor,
        compiled: Optional[Dict[str, Any]],
        attention_mask: Optional[torch.Tensor] = None,
        sign_flip: bool = False,
        mode: str = "legible",
        prompt_len: Optional[int] = None,
        return_vkv_cache: bool = False,
        forced_hop2_ep: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Unified 3-Route execution (`Single-Hop v3.6` -> 1 pass; `Hop-1 Legible` -> early switch at L16;
        `Hop-2 Logit-Lens` -> captures `latent_probe_layers` on candidate multi-hop and injects matched `m*`)."""
        if compiled is None or float(compiled["g_ext"].item()) == 0.0:
            if attention_mask is not None and hasattr(self.backbone, "model"):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                out = self.backbone(input_ids)
            return {"logits": _extract_logits(out), "mode_used": "off", "extra_kv_tokens": 0}

        p_len_eff = prompt_len if prompt_len is not None else input_ids.shape[1]
        if forced_hop2_ep is not None:
            if forced_hop2_ep < 0:
                if attention_mask is not None and hasattr(self.backbone, "model"):
                    out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                else:
                    out = self.backbone(input_ids)
                self._last_routing = {"mode": "hop2_miss_clean"}
                return {"logits": _extract_logits(out), "mode_used": "hop2_miss_clean", "extra_kv_tokens": 0}
            lg = self._run_forced_hop2_episode(
                input_ids, forced_hop2_ep, compiled, p_len_eff, attention_mask=attention_mask, sign_flip=sign_flip
            )
            self._last_routing = {"mode": "hop2_logit_lens", "hop2_ep": forced_hop2_ep}
            return {"logits": lg, "mode_used": "hop2_logit_lens", "extra_kv_tokens": 0, "hop2_ep": forced_hop2_ep}

        cfg = self.fomp.config
        state: Dict[str, Any] = {"h_probes": {}, "want_hop2": False}
        handles = []
        max_probe_l = max(cfg.latent_probe_layers)
        all_layers = sorted(set([cfg.bridge_read_layer, cfg.read_layer, *cfg.inject_layers, *cfg.latent_probe_layers]))

        class _NeedHop1Switch(Exception):
            pass

        class _NeedHop2Resolve(Exception):
            pass

        for li in all_layers:
            def _mk(l_idx: int):
                def _pre(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                    use_kw = len(args) == 0
                    h = kwargs["hidden_states"] if use_kw else args[0]
                    if l_idx == cfg.bridge_read_layer:
                        h_br = h[:, :p_len_eff, :] if p_len_eff < h.shape[1] else h
                        m_br = attention_mask[:, :p_len_eff] if (attention_mask is not None and p_len_eff < h.shape[1]) else attention_mask
                        sp_rt = self.fomp.route_span_bridge(h_br, compiled, seq_mask=m_br, tail_guard_tokens=3)
                        if h_br.shape[1] < h.shape[1]:
                            sp_rt["tok_env"] = F.pad(sp_rt["tok_env"], (0, h.shape[1] - h_br.shape[1]), value=0.0)
                        state["span_rt"] = sp_rt
                    if l_idx == cfg.read_layer:
                        arb = self.fomp.arbitrate_routes(
                            state["span_rt"], h, seq_mask=attention_mask, prompt_len=p_len_eff
                        )
                        state["arb"] = arb
                        can_vkv = (mode == "bridge_kv") and (input_ids.shape[0] == 1) and ("pref_full_ids" in compiled)
                        if (mode == "legible" or can_vkv) and bool(arb["need_hop1"].any().item()):
                            state["active_a"] = arb["need_hop1"].float()
                            raise _NeedHop1Switch()
                        if (mode == "legible") and (input_ids.shape[0] == 1) and bool(arb["need_hop2_probe"][0].item()):
                            state["want_hop2"] = True
                            state["h_read_prompt"] = h[:, :p_len_eff, :].detach()
                        state["dir_rt"] = self.fomp.direct.route_and_latch(h, compiled)

                    if state["want_hop2"] and l_idx in cfg.latent_probe_layers:
                        state["h_probes"][l_idx] = h[:, :p_len_eff, :].detach()
                        if l_idx == max_probe_l:
                            raise _NeedHop2Resolve()

                    if (not state["want_hop2"]) and (l_idx in self.fomp.direct.inject_map):
                        inj_i = self.fomp.direct.inject_map[l_idx]
                        h_rms = h.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().detach()
                        delta = self.fomp.direct.compute_layer_delta(
                            state["dir_rt"]["latched_v"], inj_i, h.dtype, sign_flip=sign_flip,
                            latched_e=state["dir_rt"]["latched_e"], h_rms=h_rms,
                        )
                        h_new = h + delta
                        if use_kw:
                            nkw = dict(kwargs)
                            nkw["hidden_states"] = h_new
                            return args, nkw
                        return (h_new, *args[1:]), kwargs
                    return args, kwargs
                return _pre
            handles.append(self.layers[li].register_forward_pre_hook(_mk(li), with_kwargs=True))

        switch_kind = ""
        try:
            try:
                if attention_mask is not None and hasattr(self.backbone, "model"):
                    out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                else:
                    out = self.backbone(input_ids)
            except _NeedHop1Switch:
                switch_kind = "hop1"
            except _NeedHop2Resolve:
                switch_kind = "hop2"
        finally:
            for hd in handles:
                hd.remove()

        if switch_kind == "hop1":
            if mode == "bridge_kv" and input_ids.shape[0] == 1 and "pref_full_ids" in compiled:
                m1 = int(state["span_rt"]["peak_m"][0].item())
                eps = self._pick_chain_episodes(m1, compiled)
                if return_vkv_cache:
                    lg, extra, pkv = self._run_with_virtual_kv_ids(
                        input_ids[0].tolist(), eps, compiled, input_ids.device, sign_flip=sign_flip, return_cache=True
                    )
                else:
                    lg, extra = self._run_with_virtual_kv_ids(
                        input_ids[0].tolist(), eps, compiled, input_ids.device, sign_flip=sign_flip
                    )
                    pkv = None
                self._last_routing = {"mode": "virtual_kv:span_hop1", "eps": eps, "arb": state.get("arb")}
                return {"logits": lg, "mode_used": "virtual_kv:span_hop1", "extra_kv_tokens": extra, "eps": eps, "pkv": pkv}

            with self._bridge_only_hooks(
                compiled=compiled, seq_mask=attention_mask, sign_flip=sign_flip,
                active_mask=state["active_a"], tail_guard_tokens=3, prompt_len=p_len_eff,
            ) as st_a:
                if attention_mask is not None and hasattr(self.backbone, "model"):
                    out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                else:
                    out = self.backbone(input_ids)
            self._last_routing = {"mode": "stage_a_legible", "span_rt": st_a["span_rt"], "arb": state.get("arb")}
            return {"logits": _extract_logits(out), "mode_used": "stage_a_legible", "extra_kv_tokens": 0}

        if switch_kind == "hop2":
            ep_hit, best_ll, best_rel = self.route_hop2_logit_lens(
                input_ids[0, :p_len_eff].tolist(), state["h_probes"], state["h_read_prompt"], compiled
            )
            if ep_hit is not None:
                lg = self._run_forced_hop2_episode(
                    input_ids, ep_hit, compiled, p_len_eff, attention_mask=attention_mask, sign_flip=sign_flip
                )
                self._last_routing = {
                    "mode": "hop2_logit_lens", "hop2_ep": ep_hit,
                    "hop2_ll": best_ll, "hop2_rel": best_rel, "arb": state.get("arb"),
                }
                return {"logits": lg, "mode_used": "hop2_logit_lens", "extra_kv_tokens": 0, "hop2_ep": ep_hit}
            # No Hop-2 edit matched -> clean unedited pass (0 spurious drift!)
            if attention_mask is not None and hasattr(self.backbone, "model"):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                out = self.backbone(input_ids)
            self._last_routing = {
                "mode": "hop2_miss_clean", "hop2_ll": best_ll, "hop2_rel": best_rel, "arb": state.get("arb"),
            }
            return {"logits": _extract_logits(out), "mode_used": "hop2_miss_clean", "extra_kv_tokens": 0}

        self._last_routing = {
            "mode": "direct_v36",
            "dir_rt": state.get("dir_rt"),
            "span_rt": state.get("span_rt"),
            "arb": state.get("arb"),
        }
        return {"logits": _extract_logits(out), "mode_used": "direct_v36", "extra_kv_tokens": 0}

    def _run_with_virtual_kv_ids(
        self,
        query_ids: Sequence[int],
        ep_indices: List[int],
        compiled: Dict[str, Any],
        device: torch.device,
        sign_flip: bool = False,
        return_cache: bool = False,
    ) -> Any:
        if sign_flip:
            inp = torch.tensor([list(query_ids)], device=device, dtype=torch.long)
            out = self.backbone(inp)
            return (_extract_logits(out), 0, None) if return_cache else (_extract_logits(out), 0)

        bos_id = compiled.get("bos_id")
        q_no_bos = list(query_ids[1:]) if (bos_id is not None and len(query_ids) > 1 and query_ids[0] == bos_id) else list(query_ids)

        if len(ep_indices) == 1:
            snap = self._get_or_build_kv_snapshot(ep_indices[0], compiled, device)
            if snap is not None:
                from transformers import DynamicCache

                dc = DynamicCache()
                for li, (k_t, v_t) in enumerate(snap):
                    dc.update(k_t.to(device).clone(), v_t.to(device).clone(), li)
                q_t = torch.tensor([q_no_bos], device=device, dtype=torch.long)
                out = self.backbone(input_ids=q_t, past_key_values=dc, use_cache=True)
                extra = len(compiled["pref_full_ids"][ep_indices[0]]) - (1 if bos_id is not None else 0)
                if return_cache:
                    return _extract_logits(out), max(0, extra), getattr(out, "past_key_values", None)
                return _extract_logits(out), max(0, extra)

        pref: List[int] = list(compiled["kv_header_ids"])
        for ei in ep_indices:
            pref.extend(compiled["fact_line_ids"][ei])
        pref.extend(compiled["kv_sep_ids"])
        full = pref + q_no_bos
        inp = torch.tensor([full], device=device, dtype=torch.long)
        if hasattr(self.backbone, "model") and return_cache:
            out = self.backbone(input_ids=inp, use_cache=True)
            lg = _extract_logits(out)[:, len(pref) :, :]
            extra = len(pref) - (1 if bos_id is not None else 0)
            return lg, max(0, extra), getattr(out, "past_key_values", None)
        out = self.backbone(inp)
        lg = _extract_logits(out)[:, len(pref) :, :]
        extra = len(pref) - (1 if bos_id is not None else 0)
        return (lg, max(0, extra), None) if return_cache else (lg, max(0, extra))

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compiled: Optional[Dict[str, Any]] = None,
        sign_flip: bool = False,
        mode: str = "legible",
    ) -> Dict[str, Any]:
        return self._forward_single_pass_probed(
            input_ids, compiled=compiled, attention_mask=attention_mask, sign_flip=sign_flip, mode=mode,
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
        p_l, t_l = list(prompt_ids), list(tgt_ids)
        full = torch.tensor([p_l + t_l], device=device, dtype=torch.long)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            res = self._forward_single_pass_probed(
                full, compiled=compiled, sign_flip=sign_flip, mode=mode, prompt_len=len(p_l),
            )
            lg = res["logits"]
            extra = int(res.get("extra_kv_tokens", 0))
        t_len = len(t_l)
        lp = F.log_softmax(lg[0, -t_len - 1 : -1].float(), dim=-1)
        tgt_t = torch.tensor(t_l, device=device, dtype=torch.long)
        return float(lp.gather(-1, tgt_t.unsqueeze(-1)).squeeze(-1).mean().item()), extra

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
        """Multi-token greedy decode across all 3 routes (`forced_hop2_ep` reused across multi-token steps!)."""
        p_l = list(prompt_ids)
        p_len = len(p_l)
        cur = torch.tensor([p_l], device=device, dtype=torch.long)
        gen: List[int] = []

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            res0 = self._forward_single_pass_probed(
                cur, compiled=compiled, sign_flip=sign_flip, mode=mode, prompt_len=p_len,
                return_vkv_cache=(mode == "bridge_kv" and not sign_flip and max_new_tokens > 1),
            )

        if res0["mode_used"].startswith("virtual_kv:") and not sign_flip and max_new_tokens > 1:
            eps = res0["eps"]
            extra_used = int(res0.get("extra_kv_tokens", 0))
            pkv = res0.get("pkv")
            nxt = int(res0["logits"][0, -1].argmax(dim=-1).item())
            gen.append(nxt)
            if eos_id is not None and nxt == eos_id:
                return gen, extra_used
            cur_q = list(p_l) + [nxt]
            for _ in range(1, max_new_tokens):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    if pkv is not None:
                        step_in = torch.tensor([[nxt]], device=device, dtype=torch.long)
                        out_s = self.backbone(input_ids=step_in, past_key_values=pkv, use_cache=True)
                        pkv = getattr(out_s, "past_key_values", None)
                        lg_s = _extract_logits(out_s)[0, -1]
                    else:
                        lg_all, _ = self._run_with_virtual_kv_ids(cur_q, eps, compiled, device, sign_flip=False)  # type: ignore[arg-type]
                        lg_s = lg_all[0, -1]
                nxt = int(lg_s.argmax(dim=-1).item())
                gen.append(nxt)
                if eos_id is not None and nxt == eos_id:
                    break
                cur_q.append(nxt)
            return gen, extra_used

        locked_hop2_ep = res0.get("hop2_ep")
        if locked_hop2_ep is None and res0["mode_used"] == "hop2_miss_clean":
            locked_hop2_ep = -1
        nxt = int(res0["logits"][0, -1].argmax(dim=-1).item())
        gen.append(nxt)
        extra_used = int(res0.get("extra_kv_tokens", 0))
        if eos_id is not None and nxt == eos_id:
            return gen, extra_used

        cur = torch.cat([cur, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        for _ in range(1, max_new_tokens):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = self._forward_single_pass_probed(
                    cur, compiled=compiled, sign_flip=sign_flip, mode=mode, prompt_len=p_len,
                    forced_hop2_ep=locked_hop2_ep,
                )["logits"][0, -1]
            nxt = int(lg.argmax(dim=-1).item())
            gen.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        return gen, extra_used
