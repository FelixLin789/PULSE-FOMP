"""Stage 2.5 Peer-Review Rigor Audit on `Llama-3-8B-Instruct` (Addresses All 3 Adversarial Critiques).

Directly resolves every methodological gap in Stage 2 with zero-strawman fairness and full statistical power:
  1. Same-Paradigm Retrieval/Memory Baselines (`IKE / Dense RAG + Prompt` via `BAAI/bge-small-en-v1.5`):
       - `ike_rag_always`: Top-2 retrieved facts always prepended (high recall / 2-hop reference)
       - `ike_rag_gated` : Top-2 retrieved facts gated by cosine threshold calibrated strictly on `cf_train`
                           (0% test leakage; direct apples-to-apples counterpart to FOMP's Dual-Lock gate!)
  2. Strict Zero-MQuAKE Ablation (`fomp_v36_zero_mq`):
       - Trains `FOMP-v3.6` with **0% MQuAKE** and **0% ZsRE** training data (`cf_train + cf_qa` only),
         eliminating the in-distribution vs training-free confound on MQuAKE multi-hop!
       - Also evaluates the existing `fomp_v36` checkpoint (with MQuAKE train) explicitly labeled as in-domain.
  3. High Statistical Power (`n = 200` CF [2x100], `n = 192` MQuAKE [3x64], `n = 200` ZsRE [2x100]):
       - Full greedy Gen-EM on EVERY test item (no 32-item cap!),
       - Exact per-item binary hit vectors stored in JSON,
       - Exact two-sided paired McNemar p-values + Wilson 95% confidence intervals + batch mean +/- std!

Resumable arm-by-arm with atomic JSON checkpointing and automatic Markdown report generation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from pulse.fomp_v3 import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module  # noqa: E402
from v7_stage0_stage1 import save_json, sync  # noqa: E402
from fomp_v3_open_edit import (  # noqa: E402
    EditRecord,
    _SmokeHFModel,
    _SmokeTokenizer,
    _load_real_datasets,
    _make_smoke_datasets,
    _parse_mquake_row,
    _train_fomp_v3,
)
from fomp_v34_stress_ood import (  # noqa: E402
    ZsRERecord,
    _load_s1_1024_unseen,
    _load_zsre_test,
    _make_smoke_zsre,
)
from fomp_v35_llama3_8b_sota import (  # noqa: E402
    _build_requests,
    _enc_prompt,
    _ensure_alphaedit_nullspace_gpu,
    _eval_ppl_any,
    _greedy_gen_any,
    _score_logprob_any,
)


# ==============================================================================
# Exact Statistics: Paired McNemar Test + Wilson 95% CI (Zero SciPy Dependency)
# ==============================================================================

def exact_mcnemar_p(hits_a: Sequence[int], hits_b: Sequence[int]) -> Dict[str, Any]:
    """Exact two-sided paired McNemar test on aligned per-item binary hit vectors."""
    assert len(hits_a) == len(hits_b), f"Length mismatch: {len(hits_a)} vs {len(hits_b)}"
    n = len(hits_a)
    n10 = sum(1 for a, b in zip(hits_a, hits_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(hits_a, hits_b) if a == 0 and b == 1)
    n_disc = n10 + n01
    if n_disc == 0:
        p = 1.0
    else:
        k = min(n10, n01)
        # Exact binomial two-sided tail with p=0.5 using log-space for large n_disc safety
        tail = 0.0
        log2 = math.log(2.0)
        for i in range(k + 1):
            log_prob = math.lgamma(n_disc + 1) - math.lgamma(i + 1) - math.lgamma(n_disc - i + 1) - n_disc * log2
            tail += math.exp(log_prob)
        p = min(1.0, 2.0 * tail)
    acc_a = (sum(hits_a) / n * 100.0) if n else 0.0
    acc_b = (sum(hits_b) / n * 100.0) if n else 0.0
    return {
        "n": n,
        "a_pct": round(acc_a, 2),
        "b_pct": round(acc_b, 2),
        "diff_pp": round(acc_a - acc_b, 2),
        "n10_a_wins": n10,
        "n01_b_wins": n01,
        "p_value": float(f"{p:.4g}"),
        "sig_005": bool(p < 0.05),
    }


def wilson_ci95(hits: Sequence[int]) -> Tuple[float, float, float]:
    """Return `(pct, ci_low_pct, ci_high_pct)` using Wilson score 95% interval."""
    n = len(hits)
    if n == 0:
        return 0.0, 0.0, 0.0
    p = sum(hits) / n
    z = 1.959963984540054
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return round(p * 100.0, 2), round(max(0.0, center - half) * 100.0, 2), round(min(1.0, center + half) * 100.0, 2)


def _metric_from_hits(hits: List[int]) -> Dict[str, Any]:
    pct, lo, hi = wilson_ci95(hits)
    return {"pct": pct, "ci95": [lo, hi], "n": len(hits), "hits": hits}


# ==============================================================================
# Expanded Zero-Leakage MQuAKE Loader (`n = 192`, 0% Overlap with Train!)
# ==============================================================================

def _load_mquake_expanded_unseen(
    tokenizer: Any,
    max_tgt_tokens: int,
    base_mq_test: List[EditRecord],
    exclude_subjs: Set[str],
    target_n: int = 192,
) -> List[EditRecord]:
    """Expand MQuAKE test set across CF3k + CF9k + T splits with 0% overlap with `exclude_subjs`."""
    from datasets import load_dataset

    def _valid_tgt(s: str) -> bool:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return 1 <= len(ids) <= max_tgt_tokens

    pool: List[EditRecord] = list(base_mq_test)
    used: Set[str] = {r.subject.lower() for r in pool}
    mq_dict = load_dataset("henryzhongsc/MQuAKE-Remastered")
    for split_name in ("CF3k", "CF9k", "CF6334", "T"):
        if len(pool) >= target_n or split_name not in mq_dict:
            continue
        for row in mq_dict[split_name]:
            if len(pool) >= target_n:
                break
            rec = _parse_mquake_row(row)
            if rec is None:
                continue
            sl = rec.subject.lower()
            if sl in exclude_subjs or sl in used:
                continue
            if _valid_tgt(rec.target_new) and _valid_tgt(rec.multihop_answer_new) and _valid_tgt(rec.multihop_answer_true):
                pool.append(rec)
                used.add(sl)
    assert used.isdisjoint(exclude_subjs), "Expanded MQuAKE test set leaked a training subject!"
    print(f"[Data] Expanded MQuAKE unseen test pool: {len(pool)} records (0% overlap with train).", flush=True)
    return pool


# ==============================================================================
# Same-Paradigm Dense Retriever Baseline: IKE (Top-K RAG + Prompt Injection)
# ==============================================================================

class DenseIKERetriever:
    """Zero-strawman Top-K Dense Retriever (`BAAI/bge-small-en-v1.5`, or hash-bag in `--smoke`)."""

    def __init__(self, model_id: str, device: torch.device, smoke: bool = False) -> None:
        self.model_id = model_id
        self.device = device
        self.smoke = smoke
        self.n_params = 0
        if not smoke:
            from transformers import AutoModel, AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.enc = AutoModel.from_pretrained(model_id).to(device).eval()
            for p in self.enc.parameters():
                p.requires_grad = False
            self.n_params = sum(p.numel() for p in self.enc.parameters())
            print(f"[IKE Retriever] Loaded {model_id} ({self.n_params/1e6:.2f}M params) on {device}.", flush=True)
        else:
            self.tok = None
            self.enc = None

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        if self.smoke:
            # Deterministic lexical bag-of-words embedding for smoke verification
            out = torch.zeros((len(texts), 64), dtype=torch.float32)
            for i, t in enumerate(texts):
                for w in t.lower().replace("?", " ").replace(":", " ").split():
                    out[i, sum(ord(c) * (j + 1) for j, c in enumerate(w)) % 64] += 1.0
            return F.normalize(out, p=2, dim=-1, eps=1e-6).to(self.device)

        embs: List[torch.Tensor] = []
        for s0 in range(0, len(texts), batch_size):
            sub = list(texts[s0 : s0 + batch_size])
            batch = self.tok(sub, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            out = self.enc(**batch)
            # BGE standard CLS pooling + L2 normalize
            vec = out.last_hidden_state[:, 0].float()
            embs.append(F.normalize(vec, p=2, dim=-1, eps=1e-6))
        return torch.cat(embs, dim=0)

    def calibrate_threshold(self, cf_train: List[EditRecord], max_n: int = 400) -> Tuple[float, Dict[str, float]]:
        """Calibrate IKE similarity threshold strictly on `cf_train` (0% test leakage!)."""
        sub = cf_train[: min(max_n, len(cf_train))]
        k_cues = self.encode([r.cue for r in sub])
        q_pos = self.encode([r.paraphrases[0] for r in sub])
        q_neg = self.encode([r.neighborhoods[0] for r in sub])
        # Max similarity across the bank (reflects real M-edit retrieval competition!)
        s_pos = (q_pos @ k_cues.T).max(dim=-1).values.cpu().numpy()
        s_neg = (q_neg @ k_cues.T).max(dim=-1).values.cpu().numpy()

        best_tau = 0.75
        best_score = -1.0
        best_stats: Dict[str, float] = {}
        for tau in np.linspace(0.50, 0.94, 45):
            pos_rec = float((s_pos >= tau).mean())
            neg_rej = float((s_neg < tau).mean())
            score = 0.5 * (pos_rec + neg_rej)
            if score > best_score:
                best_score = score
                best_tau = float(round(tau, 3))
                best_stats = {
                    "train_para_recall_pct": round(pos_rec * 100.0, 2),
                    "train_neigh_reject_pct": round(neg_rej * 100.0, 2),
                    "balanced_score": round(score * 100.0, 2),
                }
        print(f"[IKE Calib on cf_train (n={len(sub)})] best_tau={best_tau} | {best_stats}", flush=True)
        return best_tau, best_stats

    def build_bank(self, recs: Sequence[Any]) -> Tuple[Dict[str, Any], float]:
        sync(self.device)
        t0 = time.perf_counter()
        cues = [r.cue for r in recs]
        facts = [f"- {r.cue}{r.target_new}." for r in recs]
        keys = self.encode(cues)
        sync(self.device)
        return {"keys": keys, "facts": facts}, round(time.perf_counter() - t0, 4)

    def retrieve_prefix(
        self,
        query: str,
        bank: Dict[str, Any],
        top_k: int = 2,
        threshold: Optional[float] = None,
    ) -> Tuple[str, int]:
        q_emb = self.encode([query])[0]
        sims = bank["keys"] @ q_emb
        k = min(top_k, sims.shape[0])
        vals, idxs = sims.topk(k=k)
        picked: List[str] = []
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            if threshold is None or v >= threshold:
                picked.append(bank["facts"][idx])
        if not picked:
            return "", 0
        pref = "Updated facts (follow these strictly over prior knowledge):\n" + "\n".join(picked) + "\n\n"
        return pref, len(picked)


# ==============================================================================
# Per-Item Exact Evaluation Helpers (Full Gen-EM on ALL n Items + Hit Vectors!)
# ==============================================================================

def _eval_batch_cf_hits(
    model: nn.Module,
    tokenizer: Any,
    cf_batch: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    prose_txts: List[str],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    ike: Optional[DenseIKERetriever] = None,
    ike_bank: Optional[Dict[str, Any]] = None,
    ike_tau: Optional[float] = None,
) -> Dict[str, Any]:
    model.eval()
    extra_toks: List[int] = []

    def _mk_ids(prompt_s: str) -> List[int]:
        if ike is not None and ike_bank is not None:
            pref, _ = ike.retrieve_prefix(prompt_s, ike_bank, top_k=2, threshold=ike_tau)
            full_p = pref + prompt_s
            p_ids = _enc_prompt(tokenizer, full_p, prepend_bos)
            raw_ids = _enc_prompt(tokenizer, prompt_s, prepend_bos)
            extra_toks.append(max(0, len(p_ids) - len(raw_ids)))
            return p_ids
        extra_toks.append(0)
        return _enc_prompt(tokenizer, prompt_s, prepend_bos)

    def _run(pairs: List[Tuple[str, str, str]], want_new: bool) -> Dict[str, List[int]]:
        prob_hits: List[int] = []
        em_hits: List[int] = []
        for pr, t_new, t_orig in pairs:
            p_ids = _mk_ids(pr)
            tn_ids = tokenizer.encode(t_new, add_special_tokens=False)
            to_ids = tokenizer.encode(t_orig, add_special_tokens=False)
            lp_n = _score_logprob_any(model, p_ids, tn_ids, device, amp, wrap, compiled)
            lp_o = _score_logprob_any(model, p_ids, to_ids, device, amp, wrap, compiled)
            prob_hits.append(int((lp_n > lp_o) if want_new else (lp_o > lp_n)))

            want_s = (t_new if want_new else t_orig).strip().lower()
            want_len = len(tn_ids if want_new else to_ids)
            g_ids = _greedy_gen_any(model, p_ids, max(1, want_len + 1), device, amp, wrap, compiled)
            em_hits.append(int(tokenizer.decode(g_ids, skip_special_tokens=True).strip().lower().startswith(want_s)))
        return {"prob": prob_hits, "em": em_hits}

    sync(device)
    t0 = time.perf_counter()
    eff = _run([(r.cue, r.target_new, r.target_true) for r in cf_batch], want_new=True)
    par = _run([(r.paraphrases[0], r.target_new, r.target_true) for r in cf_batch], want_new=True)
    nei = _run([(r.neighborhoods[0], r.target_new, r.target_true) for r in cf_batch], want_new=False)
    sync(device)
    n_q = max(1, len(cf_batch) * 3)
    ms_q = round((time.perf_counter() - t0) * 1000.0 / n_q, 2)

    # General prose PPL: for IKE, also test whether retriever triggers on general WikiText prose!
    if ike is not None and ike_bank is not None:
        nlls: List[float] = []
        for ch, raw_t in zip(prose_chunks, prose_txts):
            pref, _ = ike.retrieve_prefix(raw_t[:200], ike_bank, top_k=2, threshold=ike_tau)
            p_pref = tokenizer.encode(pref, add_special_tokens=False) if pref else []
            ch_l = ch.tolist()
            full = torch.tensor([p_pref + ch_l], device=device, dtype=torch.long)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = _score_logits_for_ppl(model, full[:, :-1])
            start = len(p_pref)
            pred = lg[:, start:, :].reshape(-1, lg.shape[-1]).float()
            tgt = full[:, start + 1 :].reshape(-1)
            nlls.append(float(F.cross_entropy(pred, tgt).item()))
        ppl = round(math.exp(min(20.0, float(np.mean(nlls)))), 4)
    else:
        ppl = _eval_ppl_any(model, prose_chunks, device, amp, wrap, compiled)

    return {
        "eff_prob": eff["prob"], "eff_em": eff["em"],
        "par_prob": par["prob"], "par_em": par["em"],
        "nei_prob": nei["prob"], "nei_em": nei["em"],
        "ppl": ppl, "ms_per_query": ms_q,
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)) if extra_toks else 0.0, 2),
    }


@torch.no_grad()
def _score_logits_for_ppl(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=x, use_cache=False) if hasattr(model, "model") else model(x)
    return out["logits"] if isinstance(out, dict) else (out.logits if hasattr(out, "logits") else out[0])


def _eval_batch_mquake_hits(
    model: nn.Module,
    tokenizer: Any,
    mq_batch: List[EditRecord],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    ike: Optional[DenseIKERetriever] = None,
    ike_bank: Optional[Dict[str, Any]] = None,
    ike_tau: Optional[float] = None,
) -> Dict[str, Any]:
    model.eval()
    s_prob, s_em, m_prob, m_em = [], [], [], []
    extra_toks: List[int] = []

    def _mk_ids(prompt_s: str) -> List[int]:
        if ike is not None and ike_bank is not None:
            pref, _ = ike.retrieve_prefix(prompt_s, ike_bank, top_k=2, threshold=ike_tau)
            p_ids = _enc_prompt(tokenizer, pref + prompt_s, prepend_bos)
            raw_ids = _enc_prompt(tokenizer, prompt_s, prepend_bos)
            extra_toks.append(max(0, len(p_ids) - len(raw_ids)))
            return p_ids
        extra_toks.append(0)
        return _enc_prompt(tokenizer, prompt_s, prepend_bos)

    for r in mq_batch:
        c_ids = _mk_ids(r.cue)
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        s_prob.append(int(
            _score_logprob_any(model, c_ids, tn_ids, device, amp, wrap, compiled)
            > _score_logprob_any(model, c_ids, to_ids, device, amp, wrap, compiled)
        ))
        g_c = _greedy_gen_any(model, c_ids, max(1, len(tn_ids) + 1), device, amp, wrap, compiled)
        s_em.append(int(tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(r.target_new.strip().lower())))

        q_ids = _mk_ids(r.paraphrases[0])
        an_ids = tokenizer.encode(r.multihop_answer_new, add_special_tokens=False)
        ao_ids = tokenizer.encode(r.multihop_answer_true, add_special_tokens=False)
        m_prob.append(int(
            _score_logprob_any(model, q_ids, an_ids, device, amp, wrap, compiled)
            > _score_logprob_any(model, q_ids, ao_ids, device, amp, wrap, compiled)
        ))
        g_q = _greedy_gen_any(model, q_ids, max(1, len(an_ids) + 1), device, amp, wrap, compiled)
        m_em.append(int(tokenizer.decode(g_q, skip_special_tokens=True).strip().lower().startswith(r.multihop_answer_new.strip().lower())))

    return {
        "sh_prob": s_prob, "sh_em": s_em, "mh_prob": m_prob, "mh_em": m_em,
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)) if extra_toks else 0.0, 2),
    }


def _eval_batch_zsre_hits(
    model: nn.Module,
    tokenizer: Any,
    z_batch: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    loc_gen_len: int = 6,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    ike: Optional[DenseIKERetriever] = None,
    ike_bank: Optional[Dict[str, Any]] = None,
    ike_tau: Optional[float] = None,
) -> Dict[str, Any]:
    model.eval()
    r_prob, r_em, g_prob, g_em, loc_exact = [], [], [], [], []
    loc_tok_m = loc_tok_t = 0
    extra_toks: List[int] = []

    def _mk_ids(prompt_s: str) -> List[int]:
        if ike is not None and ike_bank is not None:
            pref, _ = ike.retrieve_prefix(prompt_s, ike_bank, top_k=2, threshold=ike_tau)
            p_ids = _enc_prompt(tokenizer, pref + prompt_s, prepend_bos)
            raw_ids = _enc_prompt(tokenizer, prompt_s, prepend_bos)
            extra_toks.append(max(0, len(p_ids) - len(raw_ids)))
            return p_ids
        extra_toks.append(0)
        return _enc_prompt(tokenizer, prompt_s, prepend_bos)

    for i, r in enumerate(z_batch):
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        tn_s = r.target_new.strip().lower()

        c_ids = _mk_ids(r.cue)
        r_prob.append(int(
            _score_logprob_any(model, c_ids, tn_ids, device, amp, wrap, compiled)
            > _score_logprob_any(model, c_ids, to_ids, device, amp, wrap, compiled)
        ))
        g_c = _greedy_gen_any(model, c_ids, max(1, len(tn_ids) + 1), device, amp, wrap, compiled)
        r_em.append(int(tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(tn_s)))

        p_ids = _mk_ids(r.paraphrase)
        g_prob.append(int(
            _score_logprob_any(model, p_ids, tn_ids, device, amp, wrap, compiled)
            > _score_logprob_any(model, p_ids, to_ids, device, amp, wrap, compiled)
        ))
        g_p = _greedy_gen_any(model, p_ids, max(1, len(tn_ids) + 1), device, amp, wrap, compiled)
        g_em.append(int(tokenizer.decode(g_p, skip_special_tokens=True).strip().lower().startswith(tn_s)))

        l_ids = _mk_ids(r.loc_prompt)
        g_l = _greedy_gen_any(model, l_ids, loc_gen_len, device, amp, wrap, compiled)
        b_l = base_loc_preds[i]
        ml = min(len(g_l), len(b_l))
        loc_tok_m += sum(1 for a, b in zip(g_l[:ml], b_l[:ml]) if a == b)
        loc_tok_t += max(1, ml)
        loc_exact.append(int(g_l[:ml] == b_l[:ml]))

    return {
        "rel_prob": r_prob, "rel_em": r_em,
        "gen_prob": g_prob, "gen_em": g_em,
        "loc_exact": loc_exact,
        "loc_tok_match_pct": round(100.0 * loc_tok_m / max(1, loc_tok_t), 2),
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)) if extra_toks else 0.0, 2),
    }


def _aggregate_cf(batches: List[Dict[str, Any]], edit_secs: List[float], base_ppl: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("eff_prob", "eff_em", "par_prob", "par_em", "nei_prob", "nei_em"):
        hits = [h for b in batches for h in b[k]]
        out[k] = _metric_from_hits(hits)
    out["edit_time_sec_per_batch"] = round(float(np.mean(edit_secs)), 4) if edit_secs else 0.0
    out["general_prose_ppl"] = round(float(np.mean([b["ppl"] for b in batches])), 4)
    out["ppl_drift_vs_base"] = round(out["general_prose_ppl"] - base_ppl, 5)
    out["ms_per_query"] = round(float(np.mean([b["ms_per_query"] for b in batches])), 2)
    out["avg_extra_prompt_tokens"] = round(float(np.mean([b["avg_extra_prompt_tokens"] for b in batches])), 2)
    out["n_batches"] = len(batches)
    return out


def _aggregate_mq(batches: List[Dict[str, Any]], edit_secs: List[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("sh_prob", "sh_em", "mh_prob", "mh_em"):
        hits = [h for b in batches for h in b[k]]
        out[k] = _metric_from_hits(hits)
        # Per-batch mean +/- std across disjoint batches
        b_pcts = [sum(b[k]) / max(1, len(b[k])) * 100.0 for b in batches]
        out[k]["batch_std_pp"] = round(float(np.std(b_pcts)), 2) if len(b_pcts) > 1 else 0.0
    out["edit_time_sec_per_batch"] = round(float(np.mean(edit_secs)), 4) if edit_secs else 0.0
    out["avg_extra_prompt_tokens"] = round(float(np.mean([b["avg_extra_prompt_tokens"] for b in batches])), 2)
    out["n_batches"] = len(batches)
    return out


def _aggregate_zsre(batches: List[Dict[str, Any]], edit_secs: List[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ("rel_prob", "rel_em", "gen_prob", "gen_em", "loc_exact"):
        hits = [h for b in batches for h in b[k]]
        out[k] = _metric_from_hits(hits)
    out["loc_tok_match_pct"] = round(float(np.mean([b["loc_tok_match_pct"] for b in batches])), 2)
    out["edit_time_sec_per_batch"] = round(float(np.mean(edit_secs)), 4) if edit_secs else 0.0
    out["avg_extra_prompt_tokens"] = round(float(np.mean([b["avg_extra_prompt_tokens"] for b in batches])), 2)
    out["n_batches"] = len(batches)
    return out


# ==============================================================================
# Markdown Report with Paired McNemar p-values & Wilson 95% CIs
# ==============================================================================

def _write_rigor_report(summary: Dict[str, Any], out_md: Path) -> None:
    arms = summary.get("arms", {})
    order = ["base", "ike_rag_always", "ike_rag_gated", "alphaedit", "fomp_v36_zero_mq", "fomp_v36"]
    lines = [
        f"# Stage 2.5 Peer-Review Rigor Audit on `{summary.get('backbone')}` (`{summary.get('device')}`)",
        "",
        f"- **Protocol**: Disjoint multi-batch evaluation (`CF n={summary['n_cf_total']}` [{summary['cf_batches']}x{summary['m_cf']}], "
        f"`MQuAKE n={summary['n_mq_total']}` [{summary['mq_batches']}x{summary['m_mq']}], "
        f"`ZsRE n={summary['n_zsre_total']}` [{summary['zsre_batches']}x{summary['m_zsre']}]); **full greedy Gen-EM on 100% of items**, "
        f"strictly 0% train/test entity overlap, **0% ZsRE train**, and dedicated **`fomp_v36_zero_mq` (0% MQuAKE train!)**.",
        f"- **Base Held-Out WikiText-2 PPL**: `{summary.get('base_prose_ppl')}`",
        "",
        f"## Table 1: CounterFact (`n = {summary['n_cf_total']}`, Full Gen-EM + Wilson 95% CI)",
        "",
        "| Method (Paradigm) | Edit/Index s | Extra Prompt Tok | Efficacy EM | Paraphrase EM [95% CI] | Retain Prob [95% CI] | PPL Drift | Query ms |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    labels = {
        "base": "Base (Unedited)",
        "ike_rag_always": "IKE-RAG Always (bge-small Top-2)",
        "ike_rag_gated": "IKE-RAG Gated (cf_train τ)",
        "alphaedit": "AlphaEdit (Weight-Edit, ICLR'25)",
        "fomp_v36_zero_mq": "FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)",
        "fomp_v36": "FOMP-v3.6 (with MQ train)",
    }
    for k in order:
        if k not in arms or "track1_cf" not in arms[k]:
            continue
        t = arms[k]["track1_cf"]
        et = f"{t['edit_time_sec_per_batch']}s" if k != "base" else "—"
        pe, pc = t["par_em"]["pct"], t["par_em"]["ci95"]
        ne, nc = t["nei_prob"]["pct"], t["nei_prob"]["ci95"]
        lines.append(
            f"| **{labels[k]}** | {et} | {t['avg_extra_prompt_tokens']} | {t['eff_em']['pct']}% | "
            f"{pe}% [{pc[0]}, {pc[1]}] | {ne}% [{nc[0]}, {nc[1]}] | {t['ppl_drift_vs_base']:+.4f} | {t['ms_per_query']} ms |"
        )

    lines += [
        "",
        f"## Table 2: MQuAKE-Remastered Multi-Hop (`n = {summary['n_mq_total']}` Across {summary['mq_batches']} Disjoint Batches)",
        "",
        "| Method | MQuAKE Train? | Edit/Index s | Extra Prompt Tok | Single-Hop EM [95% CI] | **2-Hop Multi-Hop EM [95% CI]** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k in order:
        if k not in arms or "track2_mq" not in arms[k]:
            continue
        t = arms[k]["track2_mq"]
        mq_tr_s = "706 rows (in-domain)" if k == "fomp_v36" else "**0% (Zero-Shot)**"
        et = f"{t['edit_time_sec_per_batch']}s" if k != "base" else "—"
        sh, sc = t["sh_em"]["pct"], t["sh_em"]["ci95"]
        mh, mc = t["mh_em"]["pct"], t["mh_em"]["ci95"]
        lines.append(
            f"| **{labels[k]}** | {mq_tr_s} | {et} | {t['avg_extra_prompt_tokens']} | "
            f"{sh}% [{sc[0]}, {sc[1]}] | **{mh}% [{mc[0]}, {mc[1]}]** |"
        )

    lines += [
        "",
        f"## Table 3: Zero-Shot ZsRE (`n = {summary['n_zsre_total']}`, 0% ZsRE Training Data for ALL Methods!)",
        "",
        "| Method | Edit/Index s | Extra Prompt Tok | Reliability EM [95% CI] | Generality EM [95% CI] | **Locality Exact [95% CI]** (Tok-Match) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k in order:
        if k not in arms or "track3_zsre" not in arms[k]:
            continue
        t = arms[k]["track3_zsre"]
        et = f"{t['edit_time_sec_per_batch']}s" if k != "base" else "—"
        re_, rc = t["rel_em"]["pct"], t["rel_em"]["ci95"]
        ge_, gc = t["gen_em"]["pct"], t["gen_em"]["ci95"]
        lo_, lc = t["loc_exact"]["pct"], t["loc_exact"]["ci95"]
        lines.append(
            f"| **{labels[k]}** | {et} | {t['avg_extra_prompt_tokens']} | {re_}% [{rc[0]}, {rc[1]}] | "
            f"{ge_}% [{gc[0]}, {gc[1]}] | **{lo_}% [{lc[0]}, {lc[1]}]** ({t['loc_tok_match_pct']}%) |"
        )

    # Paired McNemar exact significance matrix
    tests = summary.get("paired_mcnemar", [])
    if tests:
        lines += [
            "",
            "## Table 4: Exact Paired McNemar Significance Tests (Identical Per-Item Evaluation)",
            "",
            "| Comparison (`A` vs `B`) | Track / Metric | `A` % | `B` % | Diff (pp) | `n10 / n01` | Exact McNemar `p` | `p < 0.05`? |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for row in tests:
            lines.append(
                f"| `{row['a']}` vs `{row['b']}` | {row['metric']} | {row['a_pct']}% | {row['b_pct']}% | "
                f"{row['diff_pp']:+.2f} | {row['n10_a_wins']} / {row['n01_b_wins']} | `{row['p_value']}` | "
                f"{'YES' if row['sig_005'] else 'no'} |"
            )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_paired_tests(arms: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = [
        ("fomp_v36_zero_mq", "alphaedit"),
        ("fomp_v36_zero_mq", "ike_rag_gated"),
        ("fomp_v36_zero_mq", "ike_rag_always"),
        ("fomp_v36", "alphaedit"),
        ("fomp_v36", "ike_rag_gated"),
        ("fomp_v36", "ike_rag_always"),
        ("fomp_v36", "fomp_v36_zero_mq"),
    ]
    specs = [
        ("track1_cf", "par_em", "CF Paraphrase EM"),
        ("track1_cf", "nei_prob", "CF Retain Prob"),
        ("track2_mq", "sh_em", "MQuAKE Single-Hop EM"),
        ("track2_mq", "mh_em", "MQuAKE 2-Hop EM"),
        ("track3_zsre", "gen_em", "ZsRE Generality EM"),
        ("track3_zsre", "loc_exact", "ZsRE Locality Exact"),
    ]
    rows: List[Dict[str, Any]] = []
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        for trk, key, label in specs:
            if trk not in arms[a] or trk not in arms[b]:
                continue
            ha = arms[a][trk][key]["hits"]
            hb = arms[b][trk][key]["hits"]
            if len(ha) == len(hb) and len(ha) > 0:
                res = exact_mcnemar_p(ha, hb)
                res.update(a=a, b=b, metric=label)
                rows.append(res)
    return rows


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage2-dir", type=Path, default=None, help="Existing Stage 2 dir to reuse fomp_v36 ckpt & AlphaEdit P")
    ap.add_argument("--model-id", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--ike-model-id", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--easyedit-dir", type=Path, default=Path("/content/EasyEdit"))
    ap.add_argument("--arms", type=str, default="base,ike_rag_always,ike_rag_gated,fomp_v36,fomp_v36_zero_mq,alphaedit")
    ap.add_argument("--force-arms", type=str, default="")
    ap.add_argument("--fomp-train-steps", type=int, default=2000)
    ap.add_argument("--m-cf", type=int, default=100)
    ap.add_argument("--cf-batches", type=int, default=2)
    ap.add_argument("--m-mq", type=int, default=64)
    ap.add_argument("--mq-batches", type=int, default=3)
    ap.add_argument("--m-zsre", type=int, default=100)
    ap.add_argument("--zsre-batches", type=int, default=2)
    ap.add_argument("--alphaedit-cov-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--deadline", type=float, default=time.time() + 6000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "rigor_audit_summary.json"
    out_md = args.out / "rigor_audit_report.md"

    req_arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    force_arms = {a.strip().lower() for a in args.force_arms.split(",") if a.strip()}
    if not args.smoke and args.easyedit_dir.exists() and str(args.easyedit_dir) not in sys.path:
        sys.path.insert(0, str(args.easyedit_dir))

    if args.smoke:
        tokenizer: Any = _SmokeTokenizer()
        backbone: nn.Module = _SmokeHFModel(vocab=128, d=64, n_layers=8).to(device)
        fomp_cfg = FOMPv3Config(d_model=64, read_layer=4, inject_layers=(4, 7), key_dim=32, ent_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _make_smoke_datasets()
        cf_pool = cf_te[:12]
        mq_pool = (mq_te * 2)[:12]
        z_pool = _make_smoke_zsre(12)
        m_cf, cf_b = 6, 2
        m_mq, mq_b = 4, 3
        m_z, z_b = 6, 2
        loc_gen_len = 4
        prepend_bos = False
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[Setup] Loading {args.model_id} in BF16 ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        try:
            backbone = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).to(device)
        except TypeError:
            backbone = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
        d_m = int(backbone.config.hidden_size)
        n_l = int(backbone.config.num_hidden_layers)
        rl = max(2, int(round(n_l * 0.50)))
        inj = tuple(sorted(set([rl, min(n_l - 1, rl + 3), min(n_l - 1, rl + 7), max(rl, n_l - 5), max(rl, n_l - 1)])))
        fomp_cfg = FOMPv3Config(
            d_model=d_m, read_layer=rl, inject_layers=inj,
            key_dim=(512 if d_m >= 3072 else 384),
            ent_dim=(384 if d_m >= 3072 else 256),
            val_dim=512, max_tgt_tokens=8, top_k=2, rel_hidden=128,
        )
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _load_real_datasets(tokenizer, max_tgt_tokens=8, seed=args.seed)
        train_subjs = {r.subject.lower() for r in cf_tr + mq_tr}
        m_cf, cf_b = int(args.m_cf), int(args.cf_batches)
        m_mq, mq_b = int(args.m_mq), int(args.mq_batches)
        m_z, z_b = int(args.m_zsre), int(args.zsre_batches)
        cf_pool = _load_s1_1024_unseen(tokenizer, 8, cf_te, train_subjs, target_n=m_cf * cf_b)
        mq_pool = _load_mquake_expanded_unseen(tokenizer, 8, mq_te, train_subjs, target_n=m_mq * mq_b)
        z_pool = _load_zsre_test(tokenizer, 8, train_subjs, target_n=m_z * z_b)
        loc_gen_len = 6
        prepend_bos = getattr(tokenizer, "bos_token_id", None) is not None

    cf_batches = [cf_pool[i * m_cf : (i + 1) * m_cf] for i in range(cf_b) if len(cf_pool[i * m_cf : (i + 1) * m_cf]) > 0]
    mq_batches = [mq_pool[i * m_mq : (i + 1) * m_mq] for i in range(mq_b) if len(mq_pool[i * m_mq : (i + 1) * m_mq]) > 0]
    z_batches = [z_pool[i * m_z : (i + 1) * m_z] for i in range(z_b) if len(z_pool[i * m_z : (i + 1) * m_z]) > 0]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prose_len = 48 if args.smoke else 256
    valid_prose = [
        (torch.tensor(tokenizer.encode(t, add_special_tokens=False)[: prose_len + 1], dtype=torch.long), t)
        for t in prose_txts
        if len(tokenizer.encode(t, add_special_tokens=False)) >= (16 if args.smoke else 64)
    ]
    split_p = (len(valid_prose) // 2) if args.smoke else min(20, len(valid_prose) // 2)
    prose_tr_chunks = [c for c, _ in valid_prose[:split_p]]
    prose_ev_pairs = valid_prose[split_p : split_p + (4 if args.smoke else 16)]
    prose_ev_chunks = [c for c, _ in prose_ev_pairs]
    prose_ev_txts = [t for _, t in prose_ev_pairs]
    loc_pool = [t[:220].strip() for t in prose_txts[:split_p]] or ["The sun rises in the east every morning."]

    summary: Dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    summary.update(
        {
            "device": torch.cuda.get_device_name(0) if amp else "cpu",
            "backbone": "smoke_8L" if args.smoke else args.model_id,
            "m_cf": m_cf, "cf_batches": len(cf_batches), "n_cf_total": sum(len(b) for b in cf_batches),
            "m_mq": m_mq, "mq_batches": len(mq_batches), "n_mq_total": sum(len(b) for b in mq_batches),
            "m_zsre": m_z, "zsre_batches": len(z_batches), "n_zsre_total": sum(len(b) for b in z_batches),
            "prepend_bos": prepend_bos,
        }
    )
    summary.setdefault("arms", {})

    base_ppl = _eval_ppl_any(backbone, prose_ev_chunks, device, amp)
    summary["base_prose_ppl"] = base_ppl
    base_loc_batches: List[List[List[int]]] = []
    for zb in z_batches:
        preds: List[List[int]] = []
        for r in zb:
            l_ids = _enc_prompt(tokenizer, r.loc_prompt, prepend_bos)
            preds.append(_greedy_gen_any(backbone, l_ids, loc_gen_len, device, amp))
        base_loc_batches.append(preds)

    # --- Arm 1: Base ---
    if "base" in req_arms and ("base" not in summary["arms"] or "base" in force_arms):
        print("\n=== [Rigor Arm] BASE ===", flush=True)
        b_cf = [_eval_batch_cf_hits(backbone, tokenizer, cb, prose_ev_chunks, prose_ev_txts, device, amp, prepend_bos) for cb in cf_batches]
        b_mq = [_eval_batch_mquake_hits(backbone, tokenizer, mb, device, amp, prepend_bos) for mb in mq_batches]
        b_z = [_eval_batch_zsre_hits(backbone, tokenizer, zb, bl, device, amp, prepend_bos, loc_gen_len) for zb, bl in zip(z_batches, base_loc_batches)]
        summary["arms"]["base"] = {
            "track1_cf": _aggregate_cf(b_cf, [], base_ppl),
            "track2_mq": _aggregate_mq(b_mq, []),
            "track3_zsre": _aggregate_zsre(b_z, []),
        }
        summary["paired_mcnemar"] = _compute_paired_tests(summary["arms"])
        save_json(out_json, summary)
        _write_rigor_report(summary, out_md)

    # --- Arms 2 & 3: Same-Paradigm IKE / Dense RAG Baselines (Always & Train-Calibrated Gated!) ---
    need_ike = [a for a in ("ike_rag_always", "ike_rag_gated") if a in req_arms and (a not in summary["arms"] or a in force_arms)]
    if need_ike:
        ike = DenseIKERetriever(args.ike_model_id, device=device, smoke=args.smoke)
        ike_tau, ike_calib = ike.calibrate_threshold(cf_tr, max_n=(16 if args.smoke else 400))
        summary["ike_retriever"] = {"model_id": args.ike_model_id, "n_params": ike.n_params, "calibrated_tau": ike_tau, "calib_stats": ike_calib}
        for arm_name in need_ike:
            print(f"\n=== [Rigor Arm] {arm_name.upper()} (tau={None if arm_name == 'ike_rag_always' else ike_tau}) ===", flush=True)
            tau_use = None if arm_name == "ike_rag_always" else ike_tau
            cf_bs, cf_ts = [], []
            for cb in cf_batches:
                bk, ts = ike.build_bank(cb)
                cf_ts.append(ts)
                cf_bs.append(_eval_batch_cf_hits(backbone, tokenizer, cb, prose_ev_chunks, prose_ev_txts, device, amp, prepend_bos, ike=ike, ike_bank=bk, ike_tau=tau_use))
            mq_bs, mq_ts = [], []
            for mb in mq_batches:
                bk, ts = ike.build_bank(mb)
                mq_ts.append(ts)
                mq_bs.append(_eval_batch_mquake_hits(backbone, tokenizer, mb, device, amp, prepend_bos, ike=ike, ike_bank=bk, ike_tau=tau_use))
            z_bs, z_ts = [], []
            for zb, bl in zip(z_batches, base_loc_batches):
                bk, ts = ike.build_bank(zb)
                z_ts.append(ts)
                z_bs.append(_eval_batch_zsre_hits(backbone, tokenizer, zb, bl, device, amp, prepend_bos, loc_gen_len, ike=ike, ike_bank=bk, ike_tau=tau_use))
            summary["arms"][arm_name] = {
                "tau": tau_use,
                "track1_cf": _aggregate_cf(cf_bs, cf_ts, base_ppl),
                "track2_mq": _aggregate_mq(mq_bs, mq_ts),
                "track3_zsre": _aggregate_zsre(z_bs, z_ts),
            }
            print(f"[{arm_name.upper()}] CF_ParEM={summary['arms'][arm_name]['track1_cf']['par_em']['pct']}% "
                  f"CF_Ret={summary['arms'][arm_name]['track1_cf']['nei_prob']['pct']}% "
                  f"MQ_2HopEM={summary['arms'][arm_name]['track2_mq']['mh_em']['pct']}% "
                  f"ZsRE_GenEM={summary['arms'][arm_name]['track3_zsre']['gen_em']['pct']}% "
                  f"ZsRE_Loc={summary['arms'][arm_name]['track3_zsre']['loc_exact']['pct']}%", flush=True)
            summary["paired_mcnemar"] = _compute_paired_tests(summary["arms"])
            save_json(out_json, summary)
            _write_rigor_report(summary, out_md)
        del ike
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    # --- Arms 4 & 5: FOMP-v3.6 (In-Domain MQ) & FOMP-v3.6-ZeroMQ (0% MQuAKE & 0% ZsRE!) ---
    for arm_name in ("fomp_v36", "fomp_v36_zero_mq"):
        if arm_name not in req_arms or (arm_name in summary["arms"] and arm_name not in force_arms):
            continue
        if time.time() > args.deadline - 180:
            break
        zero_mq = arm_name == "fomp_v36_zero_mq"
        print(f"\n=== [Rigor Arm] {arm_name.upper()} (zero_mquake={zero_mq}) ===", flush=True)
        torch.manual_seed(args.seed)
        fomp = FOMPv3Module(fomp_cfg).to(device)
        wrap = FOMPv3BackboneWrapper(backbone, fomp, freeze_backbone=True)
        ckpt_path = args.out / f"{arm_name}_llama3_8b.pt"
        if arm_name == "fomp_v36" and not ckpt_path.exists() and args.stage2_dir is not None:
            reuse_p = args.stage2_dir / "fomp_v36_llama3_8b.pt"
            if reuse_p.exists() and not args.smoke:
                ckpt_path = reuse_p
        if arm_name in force_arms and ckpt_path.parent == args.out and ckpt_path.exists():
            ckpt_path.unlink()
        tr_meta = _train_fomp_v3(
            wrap, tokenizer, cf_tr, ([] if zero_mq else mq_tr), prose_tr_chunks, device, amp, args.smoke,
            steps=(8 if args.smoke else args.fomp_train_steps),
            ckpt_path=ckpt_path,
            deadline=args.deadline,
            prepend_bos=prepend_bos,
        )

        def _compile_b(recs_list: Sequence[Any]) -> Tuple[Dict[str, torch.Tensor], float]:
            sync(device)
            tc = time.perf_counter()
            cm = wrap.compile_token_episodes(
                [_enc_prompt(tokenizer, r.cue, prepend_bos) for r in recs_list],
                [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in recs_list],
                pad_id=pad_id, device=device, g_ext=1.0,
            )
            sync(device)
            return cm, round(time.perf_counter() - tc, 4)

        cf_bs, cf_ts = [], []
        for cb in cf_batches:
            cm, ts = _compile_b(cb)
            cf_ts.append(ts)
            cf_bs.append(_eval_batch_cf_hits(backbone, tokenizer, cb, prose_ev_chunks, prose_ev_txts, device, amp, prepend_bos, wrap=wrap, compiled=cm))
        mq_bs, mq_ts = [], []
        for mb in mq_batches:
            cm, ts = _compile_b(mb)
            mq_ts.append(ts)
            mq_bs.append(_eval_batch_mquake_hits(backbone, tokenizer, mb, device, amp, prepend_bos, wrap=wrap, compiled=cm))
        z_bs, z_ts = [], []
        for zb, bl in zip(z_batches, base_loc_batches):
            cm, ts = _compile_b(zb)
            z_ts.append(ts)
            z_bs.append(_eval_batch_zsre_hits(backbone, tokenizer, zb, bl, device, amp, prepend_bos, loc_gen_len, wrap=wrap, compiled=cm))

        summary["arms"][arm_name] = {
            "train_meta": tr_meta,
            "zero_mquake_train": zero_mq,
            "track1_cf": _aggregate_cf(cf_bs, cf_ts, base_ppl),
            "track2_mq": _aggregate_mq(mq_bs, mq_ts),
            "track3_zsre": _aggregate_zsre(z_bs, z_ts),
        }
        print(f"[{arm_name.upper()}] CF_ParEM={summary['arms'][arm_name]['track1_cf']['par_em']['pct']}% "
              f"CF_Ret={summary['arms'][arm_name]['track1_cf']['nei_prob']['pct']}% "
              f"MQ_2HopEM={summary['arms'][arm_name]['track2_mq']['mh_em']['pct']}% "
              f"ZsRE_GenEM={summary['arms'][arm_name]['track3_zsre']['gen_em']['pct']}% "
              f"ZsRE_Loc={summary['arms'][arm_name]['track3_zsre']['loc_exact']['pct']}%", flush=True)
        del wrap, fomp
        gc.collect()
        if amp:
            torch.cuda.empty_cache()
        summary["paired_mcnemar"] = _compute_paired_tests(summary["arms"])
        save_json(out_json, summary)
        _write_rigor_report(summary, out_md)

    # --- Arm 6: Official AlphaEdit (Reuses Cached 4.1GB Null-Space P from Stage 2!) ---
    if not args.smoke and "alphaedit" in req_arms and ("alphaedit" not in summary["arms"] or "alphaedit" in force_arms):
        if time.time() < args.deadline - 180:
            print("\n=== [Rigor Arm] ALPHAEDIT (Official EasyEdit, n=200/192/200) ===", flush=True)
            try:
                import easyeditor.models.alphaedit.AlphaEdit_main as ae_mod
                from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
                from easyeditor.util import nethook

                stats_dir = (args.stage2_dir / "easyedit_stats") if (args.stage2_dir and (args.stage2_dir / "easyedit_stats").exists()) else (args.out / "easyedit_stats")
                p_path = (args.stage2_dir / "llama3_8b_null_space_P.pt") if (args.stage2_dir and (args.stage2_dir / "llama3_8b_null_space_P.pt").exists()) else (args.out / "llama3_8b_null_space_P.pt")
                hp = AlphaEditHyperParams.from_hparams(str(args.easyedit_dir / "hparams/AlphaEdit/llama3-8b.yaml"))
                hp.model_name = args.model_id
                hp.device = 0 if device.type == "cuda" else -1
                hp.stats_dir = str(stats_dir)
                hp.P_loc = str(p_path)
                hp.mom2_dataset = "wikitext"
                hp.mom2_n_samples = int(args.alphaedit_cov_samples)
                _ensure_alphaedit_nullspace_gpu(backbone, tokenizer, hp, p_path, device)
                ae_mod.P_loaded = False
                ae_mod.cache_c_new = False

                def _ae_run_batch(recs_b: Sequence[Any], fn) -> Tuple[Dict[str, Any], float]:
                    tokenizer.padding_side = "right"
                    reqs = _build_requests(recs_b, loc_pool)
                    w_copy: Dict[str, torch.Tensor] = {}
                    try:
                        sync(device)
                        t0 = time.perf_counter()
                        _, w_copy = ae_mod.apply_AlphaEdit_to_model(
                            backbone, tokenizer, reqs, hp, copy=False, return_orig_weights=True, reset_cache=True
                        )
                        sync(device)
                        es = round(time.perf_counter() - t0, 2)
                        return fn(), es
                    finally:
                        with torch.no_grad():
                            for wn, ow in w_copy.items():
                                p = nethook.get_parameter(backbone, wn)
                                p.copy_(ow.to(device=p.device, dtype=p.dtype))
                        for p in backbone.parameters():
                            p.requires_grad = False
                        gc.collect()
                        torch.cuda.empty_cache()

                cf_bs, cf_ts = [], []
                for bi, cb in enumerate(cf_batches):
                    print(f"  [AlphaEdit CF batch {bi+1}/{len(cf_batches)}]...", flush=True)
                    r, ts = _ae_run_batch(cb, lambda cb=cb: _eval_batch_cf_hits(backbone, tokenizer, cb, prose_ev_chunks, prose_ev_txts, device, amp, prepend_bos))
                    cf_bs.append(r); cf_ts.append(ts)
                mq_bs, mq_ts = [], []
                for bi, mb in enumerate(mq_batches):
                    print(f"  [AlphaEdit MQuAKE batch {bi+1}/{len(mq_batches)}]...", flush=True)
                    r, ts = _ae_run_batch(mb, lambda mb=mb: _eval_batch_mquake_hits(backbone, tokenizer, mb, device, amp, prepend_bos))
                    mq_bs.append(r); mq_ts.append(ts)
                z_bs, z_ts = [], []
                for bi, (zb, bl) in enumerate(zip(z_batches, base_loc_batches)):
                    print(f"  [AlphaEdit ZsRE batch {bi+1}/{len(z_batches)}]...", flush=True)
                    r, ts = _ae_run_batch(zb, lambda zb=zb, bl=bl: _eval_batch_zsre_hits(backbone, tokenizer, zb, bl, device, amp, prepend_bos, loc_gen_len))
                    z_bs.append(r); z_ts.append(ts)

                summary["arms"]["alphaedit"] = {
                    "track1_cf": _aggregate_cf(cf_bs, cf_ts, base_ppl),
                    "track2_mq": _aggregate_mq(mq_bs, mq_ts),
                    "track3_zsre": _aggregate_zsre(z_bs, z_ts),
                }
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[ALPHAEDIT ERROR] {exc}\n{tb}", flush=True)
                summary["arms"]["alphaedit"] = {"error": f"{type(exc).__name__}: {exc}", "traceback": tb[-1500:]}
            summary["paired_mcnemar"] = _compute_paired_tests(summary["arms"])
            save_json(out_json, summary)
            _write_rigor_report(summary, out_md)

    summary["paired_mcnemar"] = _compute_paired_tests(summary["arms"])
    save_json(out_json, summary)
    _write_rigor_report(summary, out_md)
    print("\n=== STAGE 2.5 RIGOR AUDIT COMPLETE ->", out_md, "===", flush=True)


if __name__ == "__main__":
    main()
