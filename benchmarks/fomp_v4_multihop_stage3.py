"""Stage 3: Lifting the Two-Hop Curse (`FOMP-v4 Legible` & `FOMP-v4 Bridge-KV`) + Clean RAG & Oracle ICL Audit.

Trained with **STRICTLY 0% MQuAKE and 0% ZsRE data** (`cf_train + cf_qa` ONLY!) and evaluated on the exact
same disjoint high-power splits (`CF n=200` [2x100], `MQuAKE n=192` [3x64], `ZsRE n=200` [2x100]) as Stage 2.5,
reusing completed Stage 2.5 arms (`base`, `ike_rag_always`, `ike_rag_gated`, `fomp_v36`, `fomp_v36_zero_mq`,
`alphaedit`) atomically while adding 4 decisive new arms:
  1. `oracle_icl`          : Ground-truth edited fact prepended directly per item (100% retrieval prompt ceiling,
                             isolating retrieval accuracy vs. in-context prior-override accuracy per Claude #2).
  2. `ike_rag_gated_clean` : Strips `"Question: ...\\nAnswer:"` boilerplate before `bge-small` encoding AND
                             calibrates `tau` symmetrically on `cf_train + cf_qa` (0% test leakage), repairing
                             the format-shift RAG gate issue flagged in adversarial review.
  3. `fomp_v4_legible`     : Route 1 — Pure Activation 3-Stage Laminar Plasticity (`0` extra prompt/KV tokens,
                             `0%` MQuAKE/ZsRE train): Stage-A Read-Equivalent Bridge Replacement (`L10..L19`) +
                             Stage-B Direct FOMP-v3.6 (`L16..L31`) + Stage-C Resolved-Entity Hop-2 (`L20->L23..31`).
  4. `fomp_v4_bridge_kv`   : Route 2 — Hybrid Laminar Router + Pre-Compiled Virtual-KV Memory (`0%` MQuAKE/ZsRE
                             train): uses the exact same neural Stage A/B/C gates (`0` extra tokens on all
                             single-hop & locality queries!), mounting pre-compiled `DynamicCache` KV tensors
                             strictly on multi-hop bridge triggers.

Also decomposes `MQuAKE n=192` 2-Hop EM into `Hop-1 Edited (ans != target_new)` vs `Hop-2 Edited (ans == target_new)`
across ALL arms (retroactively indexing saved 192-item `hits` vectors from Stage 2.5!).
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from pulse.fomp_v3 import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module, _causal_window_mean, _extract_tail_window  # noqa: E402
from pulse.fomp_v4 import (  # noqa: E402
    FOMPv4BackboneWrapper,
    FOMPv4Config,
    FOMPv4Module,
    format_virtual_kv_fact,
    strip_cue_copula,
)
from v7_stage0_stage1 import save_json, sync  # noqa: E402
from fomp_v3_open_edit import (  # noqa: E402
    EditRecord,
    _SmokeHFModel,
    _SmokeTokenizer,
    _load_real_datasets,
    _make_smoke_datasets,
    _train_fomp_v3,
)
from fomp_v34_stress_ood import (  # noqa: E402
    ZsRERecord,
    _load_s1_1024_unseen,
    _load_zsre_test,
    _make_smoke_zsre,
)
from fomp_v35_llama3_8b_sota import (  # noqa: E402
    _enc_prompt,
    _eval_ppl_any,
    _greedy_gen_any,
    _score_logprob_any,
)
from fomp_v36_rigor_audit import (  # noqa: E402
    DenseIKERetriever,
    _aggregate_cf,
    _aggregate_mq,
    _aggregate_zsre,
    _eval_batch_cf_hits,
    _eval_batch_mquake_hits,
    _eval_batch_zsre_hits,
    _load_mquake_expanded_unseen,
    _metric_from_hits,
    _score_logits_for_ppl,
    exact_mcnemar_p,
)


# ==============================================================================
# Clean Wrapper-Stripped Dense IKE Retriever & Oracle ICL Helpers
# ==============================================================================

def strip_qa_wrapper(text: str) -> str:
    """Strip surface `'Question: ...\\nAnswer:'` wrapper so bi-encoder cosine reflects semantics, not template."""
    t = text.strip()
    if t.lower().startswith("question:"):
        t = t[len("question:") :].strip()
    for suf in ("\nanswer:", "answer:"):
        if t.lower().endswith(suf):
            t = t[: -len(suf)].strip()
    return t.rstrip("?").strip()


class CleanDenseIKERetriever(DenseIKERetriever):
    """Repaired full-strength Gated RAG: strips QA boilerplate before encoding & calibrates on `cf_train + cf_qa`."""

    def encode_clean(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        return self.encode([strip_qa_wrapper(t) for t in texts], batch_size=batch_size)

    def calibrate_clean_threshold(self, cf_train: List[EditRecord], max_n: int = 300) -> Tuple[float, Dict[str, float]]:
        sub = cf_train[: min(max_n, len(cf_train))]
        half = len(sub) // 2
        # Symmetric 50% declarative + 50% QA-wrapped mix matching FOMP-ZeroMQ's exact training format mix (`0%` ZsRE!)
        cues = [r.cue for r in sub[:half]] + [f"Question: {r.cue.rstrip('.')}?\nAnswer:" for r in sub[half:]]
        paras = [r.paraphrases[0] for r in sub[:half]] + [f"Question: {r.paraphrases[0].rstrip('.?')}?\nAnswer:" for r in sub[half:]]
        neis = [r.neighborhoods[0] for r in sub[:half]] + [f"Question: {r.neighborhoods[0].rstrip('.?')}?\nAnswer:" for r in sub[half:]]
        # Also include random cross-entity QA negatives (mirrors ZsRE unrelated locality queries using only `cf_train`!)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(sub)).tolist()
        cross_qa = [f"Question: {sub[j].paraphrases[0].rstrip('.?')}?\nAnswer:" for j in perm[: max(1, half)]]

        k_cues = self.encode_clean(cues)
        q_pos = self.encode_clean(paras)
        q_nei = self.encode_clean(neis)
        q_crs = self.encode_clean(cross_qa[: min(len(cross_qa), max(1, len(sub) // 4))])
        # Exclude self-match on `cross_qa` by zeroing diagonal where indices collide
        s_pos = (q_pos @ k_cues.T).max(dim=-1).values.cpu().numpy()
        s_nei = (q_nei @ k_cues.T).max(dim=-1).values.cpu().numpy()
        crs_mat = (q_crs @ k_cues.T).cpu().numpy()
        for ci, orig_j in enumerate(perm[: len(q_crs)]):
            crs_mat[ci, orig_j] = -1.0
        s_crs = crs_mat.max(axis=-1)

        best_tau = 0.78
        best_score = -1.0
        best_stats: Dict[str, float] = {}
        for tau in np.linspace(0.55, 0.94, 40):
            pos_rec = float((s_pos >= tau).mean())
            nei_rej = float((s_nei < tau).mean())
            crs_rej = float((s_crs < tau).mean())
            neg_rej = 0.5 * (nei_rej + crs_rej)
            score = 0.5 * (pos_rec + neg_rej)
            if score > best_score:
                best_score = score
                best_tau = float(round(tau, 3))
                best_stats = {
                    "train_para_recall_pct": round(pos_rec * 100.0, 2),
                    "train_neigh_reject_pct": round(nei_rej * 100.0, 2),
                    "train_cross_qa_reject_pct": round(crs_rej * 100.0, 2),
                    "balanced_score": round(score * 100.0, 2),
                }
        print(f"[Clean IKE Calib on cf_train+cf_qa (n={len(sub)})] best_tau={best_tau} | {best_stats}", flush=True)
        return best_tau, best_stats

    def build_bank(self, recs: Sequence[Any]) -> Tuple[Dict[str, Any], float]:
        sync(self.device)
        t0 = time.perf_counter()
        cues = [r.cue for r in recs]
        facts = [format_virtual_kv_fact(r.cue, r.target_new) for r in recs]
        keys = self.encode_clean(cues)
        sync(self.device)
        return {"keys": keys, "facts": facts}, round(time.perf_counter() - t0, 4)

    def retrieve_prefix(
        self,
        query: str,
        bank: Dict[str, Any],
        top_k: int = 2,
        threshold: Optional[float] = None,
    ) -> Tuple[str, int]:
        q_emb = self.encode_clean([query])[0]
        sims = bank["keys"] @ q_emb
        k = min(top_k, sims.shape[0])
        vals, idxs = sims.topk(k=k)
        picked = [bank["facts"][idx] for v, idx in zip(vals.tolist(), idxs.tolist()) if threshold is None or v >= threshold]
        if not picked:
            return "", 0
        pref = "Updated facts (follow these strictly over prior knowledge):\n" + "\n".join(picked) + "\n\n"
        return pref, len(picked)


def _oracle_prefix(cue: str, target_new: str) -> str:
    return "Updated facts (follow these strictly over prior knowledge):\n" + format_virtual_kv_fact(cue, target_new) + "\n\n"


def _eval_oracle_icl_cf(
    model: nn.Module,
    tokenizer: Any,
    cf_batch: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
) -> Dict[str, Any]:
    """Oracle ICL: exact paired ground-truth fact prepended per item (tests pure in-context following & neighbor bleed)."""
    model.eval()
    extra_toks: List[int] = []

    def _run_items(mode_q: str, want_new: bool) -> Dict[str, List[int]]:
        prob_hits, em_hits = [], []
        for r in cf_batch:
            raw_q = r.cue if mode_q == "eff" else (r.paraphrases[0] if mode_q == "par" else r.neighborhoods[0])
            pref = _oracle_prefix(r.cue, r.target_new)
            p_ids = _enc_prompt(tokenizer, pref + raw_q, prepend_bos)
            raw_ids = _enc_prompt(tokenizer, raw_q, prepend_bos)
            extra_toks.append(max(0, len(p_ids) - len(raw_ids)))

            tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
            to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
            lp_n = _score_logprob_any(model, p_ids, tn_ids, device, amp)
            lp_o = _score_logprob_any(model, p_ids, to_ids, device, amp)
            prob_hits.append(int((lp_n > lp_o) if want_new else (lp_o > lp_n)))

            want_s = (r.target_new if want_new else r.target_true).strip().lower()
            want_len = len(tn_ids if want_new else to_ids)
            g_ids = _greedy_gen_any(model, p_ids, max(1, want_len + 1), device, amp)
            em_hits.append(int(tokenizer.decode(g_ids, skip_special_tokens=True).strip().lower().startswith(want_s)))
        return {"prob": prob_hits, "em": em_hits}

    sync(device)
    t0 = time.perf_counter()
    eff = _run_items("eff", True)
    par = _run_items("par", True)
    nei = _run_items("nei", False)
    sync(device)
    ms_q = round((time.perf_counter() - t0) * 1000.0 / max(1, len(cf_batch) * 3), 2)
    ppl = _eval_ppl_any(model, prose_chunks, device, amp)
    return {
        "eff_prob": eff["prob"], "eff_em": eff["em"],
        "par_prob": par["prob"], "par_em": par["em"],
        "nei_prob": nei["prob"], "nei_em": nei["em"],
        "ppl": ppl, "ms_per_query": ms_q,
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)), 2),
    }


def _eval_oracle_icl_mq(
    model: nn.Module,
    tokenizer: Any,
    mq_batch: List[EditRecord],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
) -> Dict[str, Any]:
    model.eval()
    s_prob, s_em, m_prob, m_em = [], [], [], []
    extra_toks: List[int] = []
    for r in mq_batch:
        pref = _oracle_prefix(r.cue, r.target_new)
        c_ids = _enc_prompt(tokenizer, pref + r.cue, prepend_bos)
        raw_c = _enc_prompt(tokenizer, r.cue, prepend_bos)
        extra_toks.append(max(0, len(c_ids) - len(raw_c)))
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        s_prob.append(int(_score_logprob_any(model, c_ids, tn_ids, device, amp) > _score_logprob_any(model, c_ids, to_ids, device, amp)))
        g_c = _greedy_gen_any(model, c_ids, max(1, len(tn_ids) + 1), device, amp)
        s_em.append(int(tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(r.target_new.strip().lower())))

        q_ids = _enc_prompt(tokenizer, pref + r.paraphrases[0], prepend_bos)
        raw_q = _enc_prompt(tokenizer, r.paraphrases[0], prepend_bos)
        extra_toks.append(max(0, len(q_ids) - len(raw_q)))
        an_ids = tokenizer.encode(r.multihop_answer_new, add_special_tokens=False)
        ao_ids = tokenizer.encode(r.multihop_answer_true, add_special_tokens=False)
        m_prob.append(int(_score_logprob_any(model, q_ids, an_ids, device, amp) > _score_logprob_any(model, q_ids, ao_ids, device, amp)))
        g_q = _greedy_gen_any(model, q_ids, max(1, len(an_ids) + 1), device, amp)
        m_em.append(int(tokenizer.decode(g_q, skip_special_tokens=True).strip().lower().startswith(r.multihop_answer_new.strip().lower())))
    return {
        "sh_prob": s_prob, "sh_em": s_em, "mh_prob": m_prob, "mh_em": m_em,
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)), 2),
    }


def _eval_oracle_icl_zsre(
    model: nn.Module,
    tokenizer: Any,
    z_batch: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    loc_gen_len: int = 6,
) -> Dict[str, Any]:
    model.eval()
    r_prob, r_em, g_prob, g_em, loc_exact = [], [], [], [], []
    loc_tok_m = loc_tok_t = 0
    extra_toks: List[int] = []
    for i, r in enumerate(z_batch):
        pref = _oracle_prefix(r.cue, r.target_new)
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        tn_s = r.target_new.strip().lower()

        c_ids = _enc_prompt(tokenizer, pref + r.cue, prepend_bos)
        extra_toks.append(max(0, len(c_ids) - len(_enc_prompt(tokenizer, r.cue, prepend_bos))))
        r_prob.append(int(_score_logprob_any(model, c_ids, tn_ids, device, amp) > _score_logprob_any(model, c_ids, to_ids, device, amp)))
        g_c = _greedy_gen_any(model, c_ids, max(1, len(tn_ids) + 1), device, amp)
        r_em.append(int(tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(tn_s)))

        p_ids = _enc_prompt(tokenizer, pref + r.paraphrase, prepend_bos)
        extra_toks.append(max(0, len(p_ids) - len(_enc_prompt(tokenizer, r.paraphrase, prepend_bos))))
        g_prob.append(int(_score_logprob_any(model, p_ids, tn_ids, device, amp) > _score_logprob_any(model, p_ids, to_ids, device, amp)))
        g_p = _greedy_gen_any(model, p_ids, max(1, len(tn_ids) + 1), device, amp)
        g_em.append(int(tokenizer.decode(g_p, skip_special_tokens=True).strip().lower().startswith(tn_s)))

        # Locality with the single paired edit fact in-context (tests single-fact prompt interference vs FOMP!)
        l_ids = _enc_prompt(tokenizer, pref + r.loc_prompt, prepend_bos)
        extra_toks.append(max(0, len(l_ids) - len(_enc_prompt(tokenizer, r.loc_prompt, prepend_bos))))
        g_l = _greedy_gen_any(model, l_ids, loc_gen_len, device, amp)
        b_l = base_loc_preds[i]
        ml = min(len(g_l), len(b_l))
        loc_tok_m += sum(1 for a, b in zip(g_l[:ml], b_l[:ml]) if a == b)
        loc_tok_t += max(1, ml)
        loc_exact.append(int(g_l[:ml] == b_l[:ml]))
    return {
        "rel_prob": r_prob, "rel_em": r_em, "gen_prob": g_prob, "gen_em": g_em,
        "loc_exact": loc_exact,
        "loc_tok_match_pct": round(100.0 * loc_tok_m / max(1, loc_tok_t), 2),
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)), 2),
    }


# ==============================================================================
# Self-Supervised Zero-MQuAKE / Zero-ZsRE FOMP-v4 Training (`cf_train` ONLY!)
# ==============================================================================

_OUTER_TEMPLATES: Tuple[str, ...] = (
    "Question: Where is {NP} located?\nAnswer:",
    "Question: In which city was {NP} founded?\nAnswer:",
    "Question: What country is {NP} associated with?\nAnswer:",
    "Question: Who founded {NP}?\nAnswer:",
    "Question: On which continent is {NP} situated?\nAnswer:",
    "Question: What is {NP} best known for?\nAnswer:",
    "Question: Which language is primarily used in {NP}?\nAnswer:",
    "Question: Where is the headquarters of {NP}?\nAnswer:",
    "Question: Who is the leader of {NP}?\nAnswer:",
    "Question: In what year did {NP} begin?\nAnswer:",
    "Question: What is the capital or main hub of {NP}?\nAnswer:",
    "Question: Which organization owns or governs {NP}?\nAnswer:",
)


def _clause_to_embedded_np(raw_clause: str) -> str:
    """Turn a single-hop CounterFact cue/paraphrase into a grammatically valid embedded noun phrase."""
    t = raw_clause.strip().rstrip(".?!")
    toks = t.split()
    if not toks:
        return t
    if toks[-1].lower() in {"is", "was", "are", "were"} and len(toks) > 2:
        res = " ".join(toks[:-1])
        if res[:4].lower() == "the ":
            res = "the " + res[4:]
        return res
    low0 = toks[0].lower()
    if low0 in {"the", "a", "an"}:
        t = low0 + " " + " ".join(toks[1:])
    return f"what {t}"


def _replace_ci(text: str, sub: str, repl: str) -> Optional[str]:
    if not sub:
        return None
    idx = text.lower().find(sub.lower())
    if idx < 0:
        return None
    return text[:idx] + repl + text[idx + len(sub) :]


def _make_compositional_qa(outer_rec: EditRecord, inner_np: str) -> str:
    """Synthesize a natural 2-hop compositional question wrapping `inner_np` inside `outer_rec`'s relation (`0%` MQuAKE!)."""
    no_cop = strip_cue_copula(outer_rec.cue)
    rep_nc = _replace_ci(no_cop, outer_rec.subject, inner_np)
    if rep_nc is not None and no_cop.lower().startswith("the "):
        body = "the " + rep_nc[4:]
        return f"Question: What is {body}?\nAnswer:"
    if outer_rec.paraphrases:
        rep_p = _replace_ci(outer_rec.paraphrases[0].strip().rstrip(".?!"), outer_rec.subject, inner_np)
        if rep_p is not None:
            return f"Question: {rep_p}?\nAnswer:"
    if rep_nc is not None:
        return f"Question: What is {_clause_to_embedded_np(rep_nc)}?\nAnswer:"
    return _OUTER_TEMPLATES[0].replace("{NP}", inner_np)


def _train_fomp_v4_zero_mq(
    wrap_v4: FOMPv4BackboneWrapper,
    tokenizer: Any,
    cf_train: List[EditRecord],
    prose_tr_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    smoke: bool,
    v4_ckpt_path: Path,
    zero_mq_v3_ckpt: Optional[Path] = None,
    bridge_steps: int = 900,
    prepend_bos: bool = True,
    deadline: float = 9e18,
) -> Dict[str, Any]:
    """Train FOMP-v4 strictly on `cf_train` (`0%` MQuAKE, `0%` ZsRE!)."""
    if v4_ckpt_path.exists():
        ck = torch.load(v4_ckpt_path, map_location=device, weights_only=False)
        wrap_v4.fomp.load_state_dict(ck["fomp"])
        print(f"[FOMP-v4] Loaded cached ZeroMQ v4 checkpoint -> {v4_ckpt_path}", flush=True)
        return ck.get("train_meta", {"loaded_cached": True})

    t0 = time.perf_counter()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id if prepend_bos else None

    # --- Phase 0: Load or train Direct `FOMP-v3.6-ZeroMQ` core (`0%` MQuAKE & `0%` ZsRE) and freeze it! ---
    if zero_mq_v3_ckpt is not None and zero_mq_v3_ckpt.exists() and not smoke:
        ck3 = torch.load(zero_mq_v3_ckpt, map_location=device, weights_only=False)
        wrap_v4.fomp.direct.load_state_dict(ck3["fomp"])
        print(f"[FOMP-v4] Loaded frozen Direct core from {zero_mq_v3_ckpt} (100% single-hop & ZsRE locality preserved!)", flush=True)
        direct_meta = ck3.get("train_meta", {"reused_v36_zero_mq": True})
    else:
        tmp_v3_wrap = FOMPv3BackboneWrapper(wrap_v4.backbone, wrap_v4.fomp.direct, freeze_backbone=True)
        tmp_v3_p = v4_ckpt_path.parent / "_tmp_direct_zero_mq.pt"
        direct_meta = _train_fomp_v3(
            tmp_v3_wrap, tokenizer, cf_train, [], prose_tr_chunks, device, amp, smoke,
            steps=(6 if smoke else 1600), ckpt_path=tmp_v3_p, deadline=deadline, prepend_bos=prepend_bos,
        )
        del tmp_v3_wrap

    for p in wrap_v4.fomp.direct.parameters():
        p.requires_grad_(False)
    wrap_v4.fomp.direct.eval()

    # --- Phase 1: Pre-cache single-hop `cf_train` bridge & implicit-entity states (`0%` MQuAKE / `0%` ZsRE!) ---
    cfg = wrap_v4.fomp.config
    win_s = cfg.span_window
    win_e = wrap_v4.fomp.direct.config.ent_window
    n_pool = min(len(cf_train), 20 if smoke else 1600)
    sub_cf = cf_train[:n_pool]
    rng = np.random.default_rng(2026)

    cue_raw_ids: List[List[int]] = []
    cue_sp_ids: List[List[int]] = []
    np_pos_list: List[str] = []
    sh_q_ids: List[List[int]] = []
    pos_q_ids: List[List[int]] = []
    pos_sp_end: List[int] = []
    neg_q_ids: List[List[int]] = []
    neg_sp_end: List[int] = []
    teach_q_ids: List[List[int]] = []
    tgt_ent_ids: List[List[int]] = []
    exp_true_ids: List[List[int]] = []

    for i, r in enumerate(sub_cf):
        tmpl = _OUTER_TEMPLATES[i % len(_OUTER_TEMPLATES)]
        prefix_part, suffix_part = tmpl.split("{NP}")

        cue_raw_ids.append(_enc_prompt(tokenizer, r.cue, prepend_bos))
        c_str = strip_cue_copula(r.cue)
        cue_sp_ids.append(_enc_prompt(tokenizer, c_str, prepend_bos))

        use_para = (i % 2 == 1) and len(r.paraphrases) > 0
        np_pos = _clause_to_embedded_np(r.paraphrases[0] if use_para else r.cue)
        np_pos_list.append(np_pos)
        p0 = r.paraphrases[0] if r.paraphrases else r.cue
        sh_str = p0 if (i % 2 == 0) else f"Question: {p0.rstrip('.?')}?\nAnswer:"
        sh_q_ids.append(_enc_prompt(tokenizer, sh_str, prepend_bos))
        np_neg = _clause_to_embedded_np(r.neighborhoods[0])

        pre_pos = _enc_prompt(tokenizer, prefix_part + np_pos, prepend_bos)
        suf_ids = tokenizer.encode(suffix_part, add_special_tokens=False)
        pos_q_ids.append(pre_pos + suf_ids)
        pos_sp_end.append(len(pre_pos) - 1)

        pre_neg = _enc_prompt(tokenizer, prefix_part + np_neg, prepend_bos)
        neg_q_ids.append(pre_neg + suf_ids)
        neg_sp_end.append(len(pre_neg) - 1)

        teach_q_ids.append(_enc_prompt(tokenizer, prefix_part + r.target_new.strip() + suffix_part, prepend_bos))
        t_toks = tokenizer.encode(r.target_new, add_special_tokens=False)[: cfg.max_tgt_tokens] or [pad_id]
        tgt_ent_ids.append(([bos_id] + t_toks) if bos_id is not None else t_toks)
        exp_true_ids.append(_enc_prompt(tokenizer, f"{r.target_true.strip()} is associated with", prepend_bos))

    print(f"[FOMP-v4] Pre-caching self-supervised bridge & implicit-entity states on {n_pool} cf_train rows (0% MQuAKE)...", flush=True)
    bs_enc = 8 if smoke else 64

    def _enc_windows_at_layer(seqs: List[List[int]], ends: List[int], layer_i: int, win: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w_parts, m_parts = [], []
        for s0 in range(0, len(seqs), bs_enc):
            sub_s = seqs[s0 : s0 + bs_enc]
            sub_e = ends[s0 : s0 + bs_enc]
            mx = max(len(x) for x in sub_s)
            mat = torch.full((len(sub_s), mx), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(sub_s), mx), device=device, dtype=torch.long)
            e_t = torch.tensor(sub_e, device=device, dtype=torch.long)
            for j, x in enumerate(sub_s):
                mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
                msk[j, : len(x)] = 1
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                h_l = wrap_v4.encode_layers_batch(mat, [layer_i], attention_mask=msk)[layer_i]
            hw, hm = _extract_tail_window(h_l, e_t, win)
            w_parts.append(hw.cpu())
            m_parts.append(hm.cpu())
        return torch.cat(w_parts, dim=0), torch.cat(m_parts, dim=0)

    h_cue_sw, m_cue_sw = _enc_windows_at_layer(cue_sp_ids, [len(x) - 1 for x in cue_sp_ids], cfg.bridge_read_layer, win_s)
    h_pos_sw, m_pos_sw = _enc_windows_at_layer(pos_q_ids, pos_sp_end, cfg.bridge_read_layer, win_s)
    h_neg_sw, m_neg_sw = _enc_windows_at_layer(neg_q_ids, neg_sp_end, cfg.bridge_read_layer, win_s)

    # Pre-cache native entity hidden states `h_tgt_bridge` across `bridge_inject_layers`
    htb_parts = []
    for s0 in range(0, len(tgt_ent_ids), bs_enc):
        sub_t = tgt_ent_ids[s0 : s0 + bs_enc]
        mx = max(len(x) for x in sub_t)
        mat = torch.full((len(sub_t), mx), pad_id, device=device, dtype=torch.long)
        msk = torch.zeros((len(sub_t), mx), device=device, dtype=torch.long)
        last_t = torch.zeros((len(sub_t),), device=device, dtype=torch.long)
        for j, x in enumerate(sub_t):
            mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
            msk[j, : len(x)] = 1
            last_t[j] = len(x) - 1
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            caps_t = wrap_v4.encode_layers_batch(mat, cfg.bridge_inject_layers, attention_mask=msk)
        ar_j = torch.arange(len(sub_t), device=device)
        htb_parts.append(torch.stack([caps_t[li][ar_j, last_t] for li in cfg.bridge_inject_layers], dim=1).cpu())
    h_tgt_bridge_tr = torch.cat(htb_parts, dim=0)

    # Pre-cache Stage-C `L20` sequences (`W_lat = 28`) & frozen `direct.k_ent(target_true)`
    w_lat = 28
    h_lat_pos, m_lat_pos = _enc_windows_at_layer(pos_q_ids, [len(x) - 1 for x in pos_q_ids], cfg.latent_read_layer, w_lat)
    h_lat_neg, m_lat_neg = _enc_windows_at_layer(neg_q_ids, [len(x) - 1 for x in neg_q_ids], cfg.latent_read_layer, w_lat)

    kent_true_parts = []
    known_flags: List[bool] = []
    with torch.no_grad():
        for s0 in range(0, len(exp_true_ids), bs_enc):
            sub_e = exp_true_ids[s0 : s0 + bs_enc]
            mx = max(len(x) for x in sub_e)
            mat = torch.full((len(sub_e), mx), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(sub_e), mx), device=device, dtype=torch.long)
            last_e = torch.zeros((len(sub_e),), device=device, dtype=torch.long)
            for j, x in enumerate(sub_e):
                mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
                msk[j, : len(x)] = 1
                last_e[j] = len(x) - 1
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                h16 = wrap_v4.encode_layers_batch(mat, [cfg.read_layer], attention_mask=msk)[cfg.read_layer]
                ew, em = _extract_tail_window(h16, last_e, win_e)
                ar_j = torch.arange(len(sub_e), device=device)
                ke = wrap_v4.fomp.direct.project_entity_key_from_window(ew, em, h_last=h16[ar_j, last_e])
            kent_true_parts.append(ke.float().cpu())
        # Filter Stage-C pool to unique `target_true` strings where frozen backbone prefers `target_true` over `target_new`
        # (prevents false-negative InfoNCE collisions on repeated targets like 'French'/'Paris' and skips unknown facts!)
        for r in sub_cf:
            if smoke:
                known_flags.append(True)
            else:
                c_i = _enc_prompt(tokenizer, r.cue, prepend_bos)
                tn_i = tokenizer.encode(r.target_new, add_special_tokens=False)[:4] or [pad_id]
                tt_i = tokenizer.encode(r.target_true, add_special_tokens=False)[:4] or [pad_id]
                known_flags.append(
                    _score_logprob_any(wrap_v4.backbone, c_i, tt_i, device, amp)
                    >= _score_logprob_any(wrap_v4.backbone, c_i, tn_i, device, amp)
                )
    k_ent_true_tr = torch.cat(kent_true_parts, dim=0)

    seen_true: set = set()
    stage_c_pool: List[int] = []
    for idx_i, (r, is_kn) in enumerate(zip(sub_cf, known_flags)):
        key_t = r.target_true.strip().lower()
        if is_kn and key_t not in seen_true:
            seen_true.add(key_t)
            stage_c_pool.append(idx_i)
    if len(stage_c_pool) < 8:
        stage_c_pool = list(range(n_pool))
    print(f"[FOMP-v4] Stage-C unique known `target_true` pool: {len(stage_c_pool)}/{n_pool} rows.", flush=True)

    # Pre-cache frozen `direct.k_fam` + full-sequence `L16` states for 1-hop vs 2-hop outer-relation arbitration
    kfam_parts = []
    win_c = wrap_v4.fomp.direct.config.key_ctx_window
    with torch.no_grad():
        for s0 in range(0, len(cue_raw_ids), bs_enc):
            sub_c = cue_raw_ids[s0 : s0 + bs_enc]
            mx = max(len(x) for x in sub_c)
            mat = torch.full((len(sub_c), mx), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(sub_c), mx), device=device, dtype=torch.long)
            last_c = torch.zeros((len(sub_c),), device=device, dtype=torch.long)
            for j, x in enumerate(sub_c):
                mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
                msk[j, : len(x)] = 1
                last_c[j] = len(x) - 1
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                h16_c = wrap_v4.encode_layers_batch(mat, [cfg.read_layer], attention_mask=msk)[cfg.read_layer]
                h16_ctx = _causal_window_mean(h16_c, win_c)
                ar_j = torch.arange(len(sub_c), device=device)
                kf = wrap_v4.fomp.direct.project_key(h16_c[ar_j, last_c], h16_ctx[ar_j, last_c])
            kfam_parts.append(kf.float().cpu())
    k_fam_tr = torch.cat(kfam_parts, dim=0)                                      # [n_pool, key_dim]

    outer_j_list: List[int] = []
    comp_q_ids: List[List[int]] = []
    for i in range(n_pool):
        chosen_j = (i + max(1, n_pool // 2)) % n_pool
        for off in range(1, min(40, n_pool)):
            cand_j = (i + 17 + off * 7) % n_pool
            if cand_j != i and float((k_fam_tr[i] * k_fam_tr[cand_j]).sum().item()) < 0.25:
                chosen_j = cand_j
                break
        outer_j_list.append(chosen_j)
        comp_q_ids.append(_enc_prompt(tokenizer, _make_compositional_qa(sub_cf[chosen_j], np_pos_list[i]), prepend_bos))
    outer_j_arr = np.array(outer_j_list, dtype=np.int64)

    def _enc_full_seq_at_layer(seqs: List[List[int]], layer_i: int, max_len: int = 36) -> Tuple[torch.Tensor, torch.Tensor]:
        h_parts, m_parts = [], []
        for s0 in range(0, len(seqs), bs_enc):
            sub_s = [x[:max_len] for x in seqs[s0 : s0 + bs_enc]]
            mat = torch.full((len(sub_s), max_len), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(sub_s), max_len), device=device, dtype=torch.long)
            for j, x in enumerate(sub_s):
                mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
                msk[j, : len(x)] = 1
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                h_l = wrap_v4.encode_layers_batch(mat, [layer_i], attention_mask=msk)[layer_i]
            h_parts.append(h_l.cpu())
            m_parts.append(msk.float().cpu())
        return torch.cat(h_parts, dim=0), torch.cat(m_parts, dim=0)

    h16_sh, m16_sh = _enc_full_seq_at_layer(sh_q_ids, cfg.read_layer, max_len=36)
    h16_cp, m16_cp = _enc_full_seq_at_layer(comp_q_ids, cfg.read_layer, max_len=36)
    h16_tp, m16_tp = _enc_full_seq_at_layer(pos_q_ids, cfg.read_layer, max_len=36)

    # Dummy direct tensors (`[1, M, ...]`) for `compile_v4_states` during Stage-A end-to-end training
    dummy_direct = {
        "k_fam": torch.zeros((1, 1, cfg.key_dim), device=device),
        "k_ent": torch.zeros((1, 1, cfg.ent_dim), device=device),
        "v_bank": torch.zeros((1, 1, cfg.max_tgt_tokens, cfg.val_dim), device=device),
        "e_bank": torch.zeros((1, 1, cfg.max_tgt_tokens, cfg.d_model), device=device),
        "tgt_mask": torch.ones((1, 1, cfg.max_tgt_tokens), device=device),
        "g_ext": torch.tensor(1.0, device=device),
    }

    # --- Phase 2: Optimize Stage A & Stage C (`bridge_parameters` ONLY) ---
    train_params = wrap_v4.fomp.bridge_parameters()
    lr = 1.0e-3 if not smoke else 2.0e-3
    opt = torch.optim.AdamW(train_params, lr=lr, weight_decay=1e-4)
    steps = 8 if smoke else bridge_steps
    ema_start = max(2, int(steps * 0.65))
    ema_decay = 0.985
    ema_state: Optional[Dict[str, torch.Tensor]] = None

    nc_bs = min(16 if smoke else 128, n_pool)
    e2e_bs = min(4 if smoke else 6, n_pool)
    n_pos_e2e = max(2, e2e_bs - 2)
    last_kl = 0.0
    last_nce_a = 0.0
    last_nce_c = 0.0

    for st in range(1, steps + 1):
        if time.time() > deadline - 240:
            print(f"[FOMP-v4] Stopping bridge training early at step {st} due to deadline.", flush=True)
            break

        wrap_v4.fomp.train()
        wrap_v4.fomp.direct.eval()
        opt.zero_grad(set_to_none=True)

        idx_nc = rng.choice(n_pool, size=nc_bs, replace=False)
        lbl_nc = torch.arange(nc_bs, device=device)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            # 1. Stage-A Span Dual-Lock InfoNCE + 128x128 Cross-Negative Margin + Gate Calibration
            ks_r, ks_e = wrap_v4.fomp.project_span_keys_from_window(h_cue_sw[idx_nc].to(device), m_cue_sw[idx_nc].to(device))
            qs_r, qs_e = wrap_v4.fomp.project_span_keys_from_window(h_pos_sw[idx_nc].to(device), m_pos_sw[idx_nc].to(device))
            ns_r, ns_e = wrap_v4.fomp.project_span_keys_from_window(h_neg_sw[idx_nc].to(device), m_neg_sw[idx_nc].to(device))

            sr_pos = torch.matmul(ks_r, qs_r.T) / 0.07
            sr_all = torch.cat([sr_pos, torch.matmul(ks_r, ns_r.T) / 0.07], dim=-1)
            se_pos = torch.matmul(ks_e, qs_e.T) / 0.07
            se_all = torch.cat([se_pos, torch.matmul(ks_e, ns_e.T) / 0.07], dim=-1)
            loss_nce_a = 0.5 * (
                0.5 * (F.cross_entropy(sr_all, lbl_nc) + F.cross_entropy(sr_pos.T, lbl_nc))
                + 0.5 * (F.cross_entropy(se_all, lbl_nc) + F.cross_entropy(se_pos.T, lbl_nc))
            )
            cn_sr = torch.matmul(ks_r, ns_r.T)
            cn_se = torch.matmul(ks_e, ns_e.T)
            # Note: `ns_r` shares the SAME relation as `ks_r` (wrong entity!), so penalize off-diagonal relation & ALL entity pairs!
            eye_m = torch.eye(nc_bs, device=device, dtype=torch.bool)
            loss_marg_a = (
                2.0 * F.relu(cn_sr.masked_fill(eye_m, 0.0) - 0.25).pow(2).mean()
                + 2.0 * F.relu(cn_se.diagonal() - 0.16).pow(2).mean()
                + 4.0 * F.relu(cn_se - 0.18).pow(2).mean()
            )
            tau_sr = wrap_v4.fomp.span_rel_tau.float().clamp(cfg.min_span_rel_tau, 0.90)
            tau_se = wrap_v4.fomp.span_ent_tau.float().clamp(cfg.min_span_ent_tau, 0.90)
            g_sp_pos = wrap_v4.fomp._hill((ks_r * qs_r).sum(-1), tau_sr) * wrap_v4.fomp._hill((ks_e * qs_e).sum(-1), tau_se)
            g_sp_neg = wrap_v4.fomp._hill((ks_r * ns_r).sum(-1), tau_sr) * wrap_v4.fomp._hill((ks_e * ns_e).sum(-1), tau_se)
            loss_tau_a = F.relu(0.88 - g_sp_pos).pow(2).mean() + 6.0 * g_sp_neg.pow(2).mean()

            # 2. Stage-C Implicit Hop-1 (`L20`) -> Frozen `direct.k_ent(target_true)` InfoNCE + Margin (unique-target pool!)
            nc_c = min(nc_bs, len(stage_c_pool))
            idx_c = np.array(rng.choice(stage_c_pool, size=nc_c, replace=False), dtype=np.int64)
            lbl_c = torch.arange(nc_c, device=device)
            eye_c = torch.eye(nc_c, device=device, dtype=torch.bool)
            ql_pos = wrap_v4.fomp.project_latent_entity_from_seq(h_lat_pos[idx_c].to(device), m_lat_pos[idx_c].to(device))
            ql_neg = wrap_v4.fomp.project_latent_entity_from_seq(h_lat_neg[idx_c].to(device), m_lat_neg[idx_c].to(device))
            ke_tru = k_ent_true_tr[idx_c].to(device)

            sl_pos = torch.matmul(ql_pos, ke_tru.T) / 0.07
            sl_all = torch.cat([sl_pos, torch.matmul(ql_neg, ke_tru.T) / 0.07], dim=-1)
            loss_nce_c = 0.5 * (F.cross_entropy(sl_all, lbl_c) + F.cross_entropy(sl_pos.T, lbl_c))
            cn_sl = torch.matmul(ql_neg, ke_tru.T)
            off_sl = torch.matmul(ql_pos, ke_tru.T).masked_fill(eye_c, -1.0)
            loss_marg_c = (
                2.0 * F.relu(cn_sl.diagonal() - 0.18).pow(2).mean()
                + 4.0 * F.relu(cn_sl - 0.18).pow(2).mean()
                + 4.0 * F.relu(off_sl - 0.22).pow(2).mean()
            )
            tau_le = wrap_v4.fomp.lat_ent_tau.float().clamp(cfg.min_lat_ent_tau, 0.90)
            g_le_pos = wrap_v4.fomp._hill((ql_pos * ke_tru).sum(-1), tau_le)
            g_le_neg = wrap_v4.fomp._hill((ql_neg * ke_tru).sum(-1), tau_le)
            g_le_crs = wrap_v4.fomp._hill(off_sl, tau_le)
            loss_tau_c = F.relu(0.80 - g_le_pos).pow(2).mean() + 6.0 * g_le_neg.pow(2).mean() + 8.0 * g_le_crs.pow(2).mean()

            # 2b. Full-Prompt Outer-Relation Projector (`L16 -> direct.k_fam`) + 1-Hop vs 2-Hop Arbitration
            qo_sh = wrap_v4.fomp.project_outer_rel_from_seq(h16_sh[idx_nc].to(device), m16_sh[idx_nc].to(device))
            qo_cp = wrap_v4.fomp.project_outer_rel_from_seq(h16_cp[idx_nc].to(device), m16_cp[idx_nc].to(device))
            qo_tp = wrap_v4.fomp.project_outer_rel_from_seq(h16_tp[idx_nc].to(device), m16_tp[idx_nc].to(device))
            kf_in = k_fam_tr[idx_nc].to(device)
            kf_out = k_fam_tr[outer_j_arr[idx_nc]].to(device)

            # Mask off-diagonal entries that happen to share the exact same relation in CounterFact (`cos > 0.85`)
            same_in = ((kf_in @ kf_in.T) > 0.85) & (~eye_m)
            same_out = ((kf_out @ kf_out.T) > 0.85) & (~eye_m)
            lg_o_sh = (qo_sh @ kf_in.T).masked_fill(same_in, -1e4) / 0.08
            lg_o_cp = (qo_cp @ kf_out.T).masked_fill(same_out, -1e4) / 0.08
            loss_nce_out = 0.5 * (F.cross_entropy(lg_o_sh, lbl_nc) + F.cross_entropy(lg_o_cp, lbl_nc))

            s_sh_in = (qo_sh * kf_in).sum(-1)
            s_cp_in = (qo_cp * kf_in).sum(-1)
            s_tp_in = (qo_tp * kf_in).sum(-1)
            tau_od = wrap_v4.fomp.outer_direct_tau.float().clamp(cfg.min_outer_direct_tau, 0.75)
            loss_sep_out = (
                F.relu(0.76 - s_sh_in).pow(2).mean()
                + 3.0 * F.relu(s_cp_in - 0.14).pow(2).mean()
                + 3.0 * F.relu(s_tp_in - 0.14).pow(2).mean()
                + F.relu((tau_od + 0.16) - s_sh_in).pow(2).mean()
                + 2.0 * F.relu(s_cp_in - (tau_od - 0.14)).pow(2).mean()
                + 2.0 * F.relu(s_tp_in - (tau_od - 0.14)).pow(2).mean()
            )

            # 3. End-to-End Self-Supervised Frozen-Teacher Read-Equivalent KL Distillation (`every 2nd step`)
            loss_e2e = torch.tensor(0.0, device=device)
            if st % 2 == 1 or smoke:
                m_ep = min(4 if smoke else 16, n_pool)
                ch_e2e = rng.choice(n_pool, size=e2e_bs, replace=False).tolist()
                bank_idx = np.zeros((e2e_bs, m_ep), dtype=np.int64)
                for bi, ti in enumerate(ch_e2e):
                    bank_idx[bi, 0] = ti
                    if m_ep > 1:
                        others = [x for x in rng.choice(n_pool, size=m_ep, replace=False).tolist() if x != ti][: m_ep - 1]
                        bank_idx[bi, 1 : 1 + len(others)] = others

                stu_seqs: List[List[int]] = []
                tea_seqs: List[List[int]] = []
                for bi, ti in enumerate(ch_e2e):
                    if bi < n_pos_e2e:
                        stu_seqs.append(pos_q_ids[ti])
                        tea_seqs.append(teach_q_ids[ti])
                    elif bi == n_pos_e2e:
                        stu_seqs.append(neg_q_ids[ti])
                        tea_seqs.append(neg_q_ids[ti])
                    else:
                        # Single-hop unrelated QA negative (`cf_train` question on a different item)
                        other_i = int((ti + 7) % n_pool)
                        qa_unrel = _enc_prompt(tokenizer, f"Question: {sub_cf[other_i].paraphrases[0].rstrip('.?')}?\nAnswer:", prepend_bos)
                        stu_seqs.append(qa_unrel)
                        tea_seqs.append(qa_unrel)

                def _pad_b(lst: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
                    mx_l = max(len(x) for x in lst)
                    mt = torch.full((len(lst), mx_l), pad_id, device=device, dtype=torch.long)
                    mk = torch.zeros((len(lst), mx_l), device=device, dtype=torch.long)
                    ends = []
                    for bi, x in enumerate(lst):
                        mt[bi, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
                        mk[bi, : len(x)] = 1
                        ends.append(len(x) - 1)
                    return mt, mk, ends

                inp_stu, msk_stu, end_stu = _pad_b(stu_seqs)
                inp_tea, msk_tea, end_tea = _pad_b(tea_seqs)

                with torch.no_grad():
                    out_tea = wrap_v4.backbone(input_ids=inp_tea, attention_mask=msk_tea, use_cache=False) if hasattr(wrap_v4.backbone, "model") else wrap_v4.backbone(inp_tea)
                    lg_tea_all = out_tea.logits if hasattr(out_tea, "logits") else out_tea

                dd_b = {k: (v.expand(e2e_bs, m_ep, *v.shape[2:]) if v.ndim >= 3 else v) for k, v in dummy_direct.items()}
                comp_b = wrap_v4.fomp.compile_v4_states(
                    dd_b,
                    h_cue_sw[bank_idx].to(device),
                    m_cue_sw[bank_idx].to(device),
                    h_tgt_bridge_tr[bank_idx].to(device),
                )
                out_stu = wrap_v4.forward_bridge_train(inp_stu, msk_stu, comp_b, tail_guard_tokens=3)
                ar_e = torch.arange(e2e_bs, device=device)
                lg_s_end = out_stu["logits"][ar_e, torch.tensor(end_stu, device=device)].float()
                lg_t_end = lg_tea_all[ar_e, torch.tensor(end_tea, device=device)].float()

                loss_kl_bridge = F.kl_div(
                    F.log_softmax(lg_s_end, dim=-1),
                    F.softmax(lg_t_end, dim=-1),
                    reduction="batchmean",
                )
                neg_gates = out_stu["span_rt"]["peak_gate"][n_pos_e2e:]
                loss_e2e = loss_kl_bridge + 8.0 * neg_gates.pow(2).mean()
                last_kl = float(loss_kl_bridge.item())

            loss = (
                loss_e2e
                + 0.8 * (loss_nce_a + loss_nce_c + loss_nce_out)
                + 3.0 * (loss_marg_a + loss_marg_c)
                + 1.5 * (loss_tau_a + loss_tau_c + loss_sep_out)
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()

        if st >= ema_start:
            with torch.no_grad():
                if ema_state is None:
                    ema_state = {k: v.detach().float().clone() for k, v in wrap_v4.fomp.state_dict().items()}
                else:
                    for k, v in wrap_v4.fomp.state_dict().items():
                        ema_state[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)

        last_nce_a = float(loss_nce_a.item())
        last_nce_c = float(loss_nce_c.item())
        if st == 1 or st % (4 if smoke else 150) == 0 or st == steps:
            tsr = wrap_v4.fomp.span_rel_tau.clamp(cfg.min_span_rel_tau, 0.90).item()
            tse = wrap_v4.fomp.span_ent_tau.clamp(cfg.min_span_ent_tau, 0.90).item()
            tle = wrap_v4.fomp.lat_ent_tau.clamp(cfg.min_lat_ent_tau, 0.90).item()
            tod = wrap_v4.fomp.outer_direct_tau.clamp(cfg.min_outer_direct_tau, 0.75).item()
            al0 = torch.sigmoid(wrap_v4.fomp.bridge_alpha_logit[0]).item()
            print(
                f"  [FOMP-v4 ZeroMQ Bridge st={st:3d}/{steps}] kl_legible={last_kl:.3f} "
                f"nce_span={last_nce_a:.3f} nce_lat={last_nce_c:.3f} "
                f"tau_span=({tsr:.2f},{tse:.2f}) tau_lat_e={tle:.2f} tau_out={tod:.2f} alpha0={al0:.2f}",
                flush=True,
            )

    if ema_state is not None:
        wrap_v4.fomp.load_state_dict({k: v.to(dtype=wrap_v4.fomp.state_dict()[k].dtype, device=device) for k, v in ema_state.items()})

    sync(device)
    sec = round(time.perf_counter() - t0, 2)
    meta = {
        "zero_mquake_train": True,
        "zero_zsre_train": True,
        "bridge_steps": steps,
        "train_sec": sec,
        "total_params": sum(p.numel() for p in wrap_v4.fomp.parameters()),
        "bridge_only_params": sum(p.numel() for p in train_params),
        "final_kl_legible": round(last_kl, 4),
        "final_nce_span": round(last_nce_a, 4),
        "final_nce_lat": round(last_nce_c, 4),
        "learned_span_tau": [
            round(float(wrap_v4.fomp.span_rel_tau.clamp(cfg.min_span_rel_tau, 0.90).item()), 4),
            round(float(wrap_v4.fomp.span_ent_tau.clamp(cfg.min_span_ent_tau, 0.90).item()), 4),
        ],
        "learned_lat_ent_tau": round(float(wrap_v4.fomp.lat_ent_tau.clamp(cfg.min_lat_ent_tau, 0.90).item()), 4),
        "learned_outer_direct_tau": round(float(wrap_v4.fomp.outer_direct_tau.clamp(cfg.min_outer_direct_tau, 0.75).item()), 4),
        "learned_bridge_alpha": [round(float(x), 3) for x in torch.sigmoid(wrap_v4.fomp.bridge_alpha_logit).tolist()],
        "direct_core_meta": direct_meta,
    }
    tmp = v4_ckpt_path.with_suffix(".pt.tmp")
    torch.save({"fomp": wrap_v4.fomp.state_dict(), "config": wrap_v4.fomp.config.__dict__, "train_meta": meta}, tmp)
    os.replace(tmp, v4_ckpt_path)
    return meta


# ==============================================================================
# FOMP-v4 Evaluation Helpers (`mode = "legible"` & `mode = "bridge_kv"`)
# ==============================================================================

def _compile_v4_batch(
    wrap_v4: FOMPv4BackboneWrapper,
    tokenizer: Any,
    recs: Sequence[Any],
    pad_id: int,
    prepend_bos: bool,
    device: torch.device,
    precompile_kv: bool,
) -> Tuple[Dict[str, Any], float]:
    bos_id = tokenizer.bos_token_id if prepend_bos else None
    c_ids = [_enc_prompt(tokenizer, r.cue, prepend_bos) for r in recs]
    sp_ids = [_enc_prompt(tokenizer, strip_cue_copula(r.cue), prepend_bos) for r in recs]
    t_ids = [tokenizer.encode(r.target_new, add_special_tokens=False)[: wrap_v4.fomp.config.max_tgt_tokens] for r in recs]
    facts = [format_virtual_kv_fact(r.cue, r.target_new) for r in recs]
    sync(device)
    t0 = time.perf_counter()
    comp = wrap_v4.compile_episodes(
        c_ids, t_ids, pad_id=pad_id, device=device, bos_id=bos_id,
        cue_span_ids_list=sp_ids, fact_texts=facts, tokenizer=tokenizer,
        g_ext=1.0, precompile_kv=precompile_kv,
    )
    sync(device)
    return comp, round(time.perf_counter() - t0, 4)


def _eval_v4_cf(
    wrap_v4: FOMPv4BackboneWrapper,
    tokenizer: Any,
    cf_batch: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    compiled: Dict[str, Any],
    mode: str,
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
) -> Dict[str, Any]:
    wrap_v4.fomp.eval()
    extra_toks: List[int] = []

    def _run(pairs: List[Tuple[str, str, str]], want_new: bool) -> Dict[str, List[int]]:
        prob_hits, em_hits = [], []
        for pr, t_new, t_orig in pairs:
            p_ids = _enc_prompt(tokenizer, pr, prepend_bos)
            tn_ids = tokenizer.encode(t_new, add_special_tokens=False)
            to_ids = tokenizer.encode(t_orig, add_special_tokens=False)
            lp_n, ex1 = wrap_v4.score_target_logprob(p_ids, tn_ids, device, amp, compiled=compiled, mode=mode)
            lp_o, _ = wrap_v4.score_target_logprob(p_ids, to_ids, device, amp, compiled=compiled, mode=mode)
            prob_hits.append(int((lp_n > lp_o) if want_new else (lp_o > lp_n)))

            want_s = (t_new if want_new else t_orig).strip().lower()
            want_len = len(tn_ids if want_new else to_ids)
            g_ids, ex2 = wrap_v4.greedy_generate(p_ids, max(1, want_len + 1), device, amp=amp, compiled=compiled, mode=mode)
            extra_toks.append(max(ex1, ex2))
            em_hits.append(int(tokenizer.decode(g_ids, skip_special_tokens=True).strip().lower().startswith(want_s)))
        return {"prob": prob_hits, "em": em_hits}

    sync(device)
    t0 = time.perf_counter()
    eff = _run([(r.cue, r.target_new, r.target_true) for r in cf_batch], True)
    par = _run([(r.paraphrases[0], r.target_new, r.target_true) for r in cf_batch], True)
    nei = _run([(r.neighborhoods[0], r.target_new, r.target_true) for r in cf_batch], False)
    sync(device)
    ms_q = round((time.perf_counter() - t0) * 1000.0 / max(1, len(cf_batch) * 3), 2)

    nlls = []
    with torch.no_grad():
        for ch in prose_chunks:
            x = ch[:-1].unsqueeze(0).to(device)
            y = ch[1:].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = wrap_v4.forward_logits(x, compiled=compiled, mode=mode)["logits"][0].float()
            nlls.append(float(F.cross_entropy(lg[-len(y) :], y).item()))
    ppl = round(math.exp(min(20.0, float(np.mean(nlls)))), 4)

    return {
        "eff_prob": eff["prob"], "eff_em": eff["em"],
        "par_prob": par["prob"], "par_em": par["em"],
        "nei_prob": nei["prob"], "nei_em": nei["em"],
        "ppl": ppl, "ms_per_query": ms_q,
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)) if extra_toks else 0.0, 2),
    }


def _eval_v4_mq(
    wrap_v4: FOMPv4BackboneWrapper,
    tokenizer: Any,
    mq_batch: List[EditRecord],
    compiled: Dict[str, Any],
    mode: str,
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
) -> Dict[str, Any]:
    wrap_v4.fomp.eval()
    s_prob, s_em, m_prob, m_em = [], [], [], []
    extra_toks: List[int] = []
    for r in mq_batch:
        c_ids = _enc_prompt(tokenizer, r.cue, prepend_bos)
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        lp_cn, ex_c1 = wrap_v4.score_target_logprob(c_ids, tn_ids, device, amp, compiled=compiled, mode=mode)
        lp_co, _ = wrap_v4.score_target_logprob(c_ids, to_ids, device, amp, compiled=compiled, mode=mode)
        s_prob.append(int(lp_cn > lp_co))
        g_c, ex_c2 = wrap_v4.greedy_generate(c_ids, max(1, len(tn_ids) + 1), device, amp=amp, compiled=compiled, mode=mode)
        extra_toks.append(max(ex_c1, ex_c2))
        s_em.append(int(tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(r.target_new.strip().lower())))

        q_ids = _enc_prompt(tokenizer, r.paraphrases[0], prepend_bos)
        an_ids = tokenizer.encode(r.multihop_answer_new, add_special_tokens=False)
        ao_ids = tokenizer.encode(r.multihop_answer_true, add_special_tokens=False)
        lp_qn, ex_q1 = wrap_v4.score_target_logprob(q_ids, an_ids, device, amp, compiled=compiled, mode=mode)
        lp_qo, _ = wrap_v4.score_target_logprob(q_ids, ao_ids, device, amp, compiled=compiled, mode=mode)
        m_prob.append(int(lp_qn > lp_qo))
        g_q, ex_q2 = wrap_v4.greedy_generate(q_ids, max(1, len(an_ids) + 1), device, amp=amp, compiled=compiled, mode=mode)
        extra_toks.append(max(ex_q1, ex_q2))
        m_em.append(int(tokenizer.decode(g_q, skip_special_tokens=True).strip().lower().startswith(r.multihop_answer_new.strip().lower())))
    return {
        "sh_prob": s_prob, "sh_em": s_em, "mh_prob": m_prob, "mh_em": m_em,
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)) if extra_toks else 0.0, 2),
    }


def _eval_v4_zsre(
    wrap_v4: FOMPv4BackboneWrapper,
    tokenizer: Any,
    z_batch: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    compiled: Dict[str, Any],
    mode: str,
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    loc_gen_len: int = 6,
) -> Dict[str, Any]:
    wrap_v4.fomp.eval()
    r_prob, r_em, g_prob, g_em, loc_exact = [], [], [], [], []
    loc_tok_m = loc_tok_t = 0
    extra_toks: List[int] = []
    for i, r in enumerate(z_batch):
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        tn_s = r.target_new.strip().lower()

        c_ids = _enc_prompt(tokenizer, r.cue, prepend_bos)
        lp_rn, ex_r1 = wrap_v4.score_target_logprob(c_ids, tn_ids, device, amp, compiled=compiled, mode=mode)
        lp_ro, _ = wrap_v4.score_target_logprob(c_ids, to_ids, device, amp, compiled=compiled, mode=mode)
        r_prob.append(int(lp_rn > lp_ro))
        g_c, ex_r2 = wrap_v4.greedy_generate(c_ids, max(1, len(tn_ids) + 1), device, amp=amp, compiled=compiled, mode=mode)
        extra_toks.append(max(ex_r1, ex_r2))
        r_em.append(int(tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(tn_s)))

        p_ids = _enc_prompt(tokenizer, r.paraphrase, prepend_bos)
        lp_gn, ex_g1 = wrap_v4.score_target_logprob(p_ids, tn_ids, device, amp, compiled=compiled, mode=mode)
        lp_go, _ = wrap_v4.score_target_logprob(p_ids, to_ids, device, amp, compiled=compiled, mode=mode)
        g_prob.append(int(lp_gn > lp_go))
        g_p, ex_g2 = wrap_v4.greedy_generate(p_ids, max(1, len(tn_ids) + 1), device, amp=amp, compiled=compiled, mode=mode)
        extra_toks.append(max(ex_g1, ex_g2))
        g_em.append(int(tokenizer.decode(g_p, skip_special_tokens=True).strip().lower().startswith(tn_s)))

        l_ids = _enc_prompt(tokenizer, r.loc_prompt, prepend_bos)
        g_l, ex_l = wrap_v4.greedy_generate(l_ids, loc_gen_len, device, amp=amp, compiled=compiled, mode=mode)
        extra_toks.append(ex_l)
        b_l = base_loc_preds[i]
        ml = min(len(g_l), len(b_l))
        loc_tok_m += sum(1 for a, b in zip(g_l[:ml], b_l[:ml]) if a == b)
        loc_tok_t += max(1, ml)
        loc_exact.append(int(g_l[:ml] == b_l[:ml]))
    return {
        "rel_prob": r_prob, "rel_em": r_em, "gen_prob": g_prob, "gen_em": g_em,
        "loc_exact": loc_exact,
        "loc_tok_match_pct": round(100.0 * loc_tok_m / max(1, loc_tok_t), 2),
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)) if extra_toks else 0.0, 2),
    }


# ==============================================================================
# Hop-1 vs Hop-2 Breakdown & Complete Stage 3 Report Generator
# ==============================================================================

def _compute_hop_subtype_breakdown(arms: Dict[str, Any], mq_flat: List[EditRecord]) -> Dict[str, Any]:
    """Split `mh_em['hits']` into `hop1_edited (ans != target_new)` vs `hop2_edited (ans == target_new)`."""
    is_hop2 = [r.multihop_answer_new.strip().lower() == r.target_new.strip().lower() for r in mq_flat]
    out: Dict[str, Any] = {
        "n_hop1_edited": int(sum(1 for x in is_hop2 if not x)),
        "n_hop2_edited": int(sum(1 for x in is_hop2 if x)),
        "per_arm": {},
    }
    for name, a_dict in arms.items():
        if "track2_mq" not in a_dict or "mh_em" not in a_dict["track2_mq"]:
            continue
        hits = a_dict["track2_mq"]["mh_em"]["hits"]
        if len(hits) != len(is_hop2):
            continue
        h1_hits = [h for h, h2 in zip(hits, is_hop2) if not h2]
        h2_hits = [h for h, h2 in zip(hits, is_hop2) if h2]
        out["per_arm"][name] = {
            "hop1_edited_em": _metric_from_hits(h1_hits) if h1_hits else {"pct": 0.0, "ci95": [0.0, 0.0], "n": 0},
            "hop2_edited_em": _metric_from_hits(h2_hits) if h2_hits else {"pct": 0.0, "ci95": [0.0, 0.0], "n": 0},
        }
    return out


def _compute_stage3_mcnemar(arms: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs = [
        ("fomp_v4_legible", "fomp_v36_zero_mq"),
        ("fomp_v4_legible", "alphaedit"),
        ("fomp_v4_legible", "ike_rag_gated_clean"),
        ("fomp_v4_legible", "oracle_icl"),
        ("fomp_v4_bridge_kv", "fomp_v36_zero_mq"),
        ("fomp_v4_bridge_kv", "fomp_v4_legible"),
        ("fomp_v4_bridge_kv", "alphaedit"),
        ("fomp_v4_bridge_kv", "ike_rag_gated_clean"),
        ("fomp_v4_bridge_kv", "ike_rag_always"),
        ("fomp_v4_bridge_kv", "oracle_icl"),
        ("fomp_v36_zero_mq", "oracle_icl"),
        ("fomp_v36_zero_mq", "ike_rag_gated_clean"),
    ]
    metrics = [
        ("CF Paraphrase EM", lambda a: a["track1_cf"]["par_em"]["hits"]),
        ("CF Retain Prob", lambda a: a["track1_cf"]["nei_prob"]["hits"]),
        ("MQuAKE 2-Hop EM", lambda a: a["track2_mq"]["mh_em"]["hits"]),
        ("ZsRE Generality EM", lambda a: a["track3_zsre"]["gen_em"]["hits"]),
        ("ZsRE Locality Exact", lambda a: a["track3_zsre"]["loc_exact"]["hits"]),
    ]
    rows: List[Dict[str, Any]] = []
    for a_n, b_n in pairs:
        if a_n not in arms or b_n not in arms:
            continue
        if "track1_cf" not in arms[a_n] or "track1_cf" not in arms[b_n]:
            continue
        for m_label, getter in metrics:
            res = exact_mcnemar_p(getter(arms[a_n]), getter(arms[b_n]))
            res.update({"a": a_n, "b": b_n, "metric": m_label})
            rows.append(res)
    return rows


def _write_stage3_report(summary: Dict[str, Any], out_md: Path) -> None:
    arms = summary.get("arms", {})
    sub = summary.get("mquake_hop_subtypes", {})
    order = [
        "base",
        "oracle_icl",
        "ike_rag_always",
        "ike_rag_gated",
        "ike_rag_gated_clean",
        "alphaedit",
        "fomp_v36_zero_mq",
        "fomp_v36",
        "fomp_v4_legible",
        "fomp_v4_bridge_kv",
    ]
    labels = {
        "base": "Base (Unedited)",
        "oracle_icl": "Oracle ICL (100% GT Fact in Prompt)",
        "ike_rag_always": "IKE-RAG Always (bge-small Top-2)",
        "ike_rag_gated": "IKE-RAG Gated (raw cf_train τ)",
        "ike_rag_gated_clean": "IKE-RAG Gated Clean (QA-stripped + cf_qa τ)",
        "alphaedit": "AlphaEdit (Weight-Edit, ICLR'25)",
        "fomp_v36_zero_mq": "FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)",
        "fomp_v36": "FOMP-v3.6 (+706 MQ in-domain)",
        "fomp_v4_legible": "FOMP-v4 Legible (Route 1: Pure Act, 0% MQ/ZsRE)",
        "fomp_v4_bridge_kv": "FOMP-v4 Bridge-KV (Route 2: Pre-KV, 0% MQ/ZsRE)",
    }

    lines = [
        f"# Stage 3 Multi-Hop & Clean-Baseline Audit on `{summary.get('backbone')}` (`{summary.get('device')}`)",
        "",
        f"- **Zero-Leakage Guarantee**: Both **`FOMP-v4 Legible` (Route 1)** and **`FOMP-v4 Bridge-KV` (Route 2)** are trained with **strictly `0%` MQuAKE and `0%` ZsRE** (`cf_train` self-supervised only).",
        f"- **Disjoint Sample Sizes**: `CF n={summary['n_cf_total']}`, `MQuAKE n={summary['n_mq_total']}` (`Hop-1 edited = {sub.get('n_hop1_edited', 0)}`, `Hop-2 edited = {sub.get('n_hop2_edited', 0)}`), `ZsRE n={summary['n_zsre_total']}`.",
        "",
        "## Table 1: CounterFact (`n = 200`, Full Greedy Gen-EM)",
        "",
        "| Method | Edit/Index s | Extra Tok | Eff EM | Paraphrase EM [95% CI] | Retain Prob [95% CI] | PPL Drift | Query ms |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k in order:
        if k not in arms or "track1_cf" not in arms[k]:
            continue
        t = arms[k]["track1_cf"]
        et = f"{t['edit_time_sec_per_batch']}s" if k not in ("base", "oracle_icl") else "—"
        pe, pc = t["par_em"]["pct"], t["par_em"]["ci95"]
        ne, nc = t["nei_prob"]["pct"], t["nei_prob"]["ci95"]
        lines.append(
            f"| **{labels[k]}** | {et} | {t['avg_extra_prompt_tokens']} | {t['eff_em']['pct']}% | "
            f"{pe}% [{pc[0]}, {pc[1]}] | {ne}% [{nc[0]}, {nc[1]}] | {t['ppl_drift_vs_base']:+.4f} | {t['ms_per_query']} ms |"
        )

    lines += [
        "",
        "## Table 2: MQuAKE-Remastered Multi-Hop (`n = 192`, Overall + Hop-1 vs Hop-2 Breakdown)",
        "",
        "| Method | MQ Train | Extra Tok | 1-Hop EM | **2-Hop Overall EM [95% CI]** | Hop-1 Edited EM | Hop-2 Edited EM |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k in order:
        if k not in arms or "track2_mq" not in arms[k]:
            continue
        t = arms[k]["track2_mq"]
        mqt = "706 rows" if k == "fomp_v36" else "**0%**"
        me, mc = t["mh_em"]["pct"], t["mh_em"]["ci95"]
        pa = sub.get("per_arm", {}).get(k, {})
        h1s = f"{pa['hop1_edited_em']['pct']}%" if "hop1_edited_em" in pa else "—"
        h2s = f"{pa['hop2_edited_em']['pct']}%" if "hop2_edited_em" in pa else "—"
        lines.append(
            f"| **{labels[k]}** | {mqt} | {t['avg_extra_prompt_tokens']} | {t['sh_em']['pct']}% | "
            f"**{me}% [{mc[0]}, {mc[1]}]** | {h1s} | {h2s} |"
        )

    lines += [
        "",
        "## Table 3: Zero-Shot ZsRE (`n = 200`, 0% ZsRE Train for ALL Methods!)",
        "",
        "| Method | Edit/Index s | Extra Tok | Rel EM [95% CI] | **Gen EM [95% CI]** | **Locality Exact [95% CI]** (Tok) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k in order:
        if k not in arms or "track3_zsre" not in arms[k]:
            continue
        t = arms[k]["track3_zsre"]
        et = f"{t['edit_time_sec_per_batch']}s" if k not in ("base", "oracle_icl") else "—"
        re, rc = t["rel_em"]["pct"], t["rel_em"]["ci95"]
        ge, gc_i = t["gen_em"]["pct"], t["gen_em"]["ci95"]
        le, lc = t["loc_exact"]["pct"], t["loc_exact"]["ci95"]
        lines.append(
            f"| **{labels[k]}** | {et} | {t['avg_extra_prompt_tokens']} | {re}% [{rc[0]}, {rc[1]}] | "
            f"**{ge}% [{gc_i[0]}, {gc_i[1]}]** | **{le}% [{lc[0]}, {lc[1]}]** ({t['loc_tok_match_pct']}%) |"
        )

    lines += [
        "",
        "## Table 4: Exact Paired McNemar Significance Tests",
        "",
        "| Comparison (`A` vs `B`) | Metric | `A` % | `B` % | Diff (pp) | `n10 / n01` | Exact McNemar `p` | `p < 0.05`? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary.get("stage3_paired_mcnemar", []):
        lines.append(
            f"| `{r['a']}` vs `{r['b']}` | {r['metric']} | {r['a_pct']}% | {r['b_pct']}% | "
            f"{r['diff_pp']:+.2f} | {r['n10_a_wins']} / {r['n01_b_wins']} | `{r['p_value']}` | "
            f"{'YES' if r['sig_005'] else 'no'} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ==============================================================================
# Main Runner (Seamlessly Resumes from Stage 2.5 `rigor_audit_summary.json`!)
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage25-dir", type=Path, default=None, help="Directory holding Stage 2.5 summary & ZeroMQ ckpt")
    ap.add_argument("--model-id", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--ike-model-id", type=str, default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--arms", type=str, default="oracle_icl,ike_rag_gated_clean,fomp_v4_legible,fomp_v4_bridge_kv")
    ap.add_argument("--force-arms", type=str, default="")
    ap.add_argument("--m-cf", type=int, default=100)
    ap.add_argument("--cf-batches", type=int, default=2)
    ap.add_argument("--m-mq", type=int, default=64)
    ap.add_argument("--mq-batches", type=int, default=3)
    ap.add_argument("--m-zsre", type=int, default=100)
    ap.add_argument("--zsre-batches", type=int, default=2)
    ap.add_argument("--bridge-steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deadline", type=float, default=9e18)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "stage3_multihop_summary.json"
    out_md = args.out / "stage3_multihop_report.md"

    req_arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    force_arms = {a.strip().lower() for a in args.force_arms.split(",") if a.strip()}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.smoke) else "cpu")
    amp = device.type == "cuda"

    if args.smoke:
        tokenizer: Any = _SmokeTokenizer()
        backbone: nn.Module = _SmokeHFModel(vocab=128, d=64, n_layers=8).to(device)
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _make_smoke_datasets()
        # Make half of smoke MQuAKE test items Hop-1 (`ans != tn`) and half Hop-2 (`ans == tn`)!
        for j, mr in enumerate(mq_te):
            if j % 2 == 0:
                mr.multihop_answer_new = f" hop2ans{j}"
        cf_pool = cf_te
        mq_pool = mq_te * 2
        z_pool = _make_smoke_zsre(16)
        m_cf, cf_b = 6, 2
        m_mq, mq_b = 4, 3
        m_z, z_b = 6, 2
        loc_gen_len = 4
        prepend_bos = False
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[Stage 3 Setup] Loading {args.model_id} in BF16 ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        try:
            backbone = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).to(device)
        except TypeError:
            backbone = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
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
    mq_flat = [r for b in mq_batches for r in b]

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

    # Seed `summary` from existing `out_json` OR Stage 2.5 `rigor_audit_summary.json`!
    summary: Dict[str, Any] = {}
    if out_json.exists():
        summary = json.loads(out_json.read_text(encoding="utf-8"))
    elif args.stage25_dir is not None and (args.stage25_dir / "rigor_audit_summary.json").exists():
        summary = json.loads((args.stage25_dir / "rigor_audit_summary.json").read_text(encoding="utf-8"))
        print(f"[Stage 3] Imported {len(summary.get('arms', {}))} completed arms from Stage 2.5!", flush=True)

    summary.update({
        "device": torch.cuda.get_device_name(0) if amp else "cpu",
        "backbone": "smoke_8L" if args.smoke else args.model_id,
        "m_cf": m_cf, "cf_batches": len(cf_batches), "n_cf_total": sum(len(b) for b in cf_batches),
        "m_mq": m_mq, "mq_batches": len(mq_batches), "n_mq_total": sum(len(b) for b in mq_batches),
        "m_zsre": m_z, "zsre_batches": len(z_batches), "n_zsre_total": sum(len(b) for b in z_batches),
        "prepend_bos": prepend_bos,
    })
    summary.setdefault("arms", {})

    base_ppl = summary.get("base_prose_ppl") or _eval_ppl_any(backbone, prose_ev_chunks, device, amp)
    summary["base_prose_ppl"] = base_ppl
    base_loc_batches: List[List[List[int]]] = [
        [_greedy_gen_any(backbone, _enc_prompt(tokenizer, r.loc_prompt, prepend_bos), loc_gen_len, device, amp) for r in zb]
        for zb in z_batches
    ]

    if "base" not in summary["arms"] or "base" in force_arms:
        print("\n=== [Stage 3 Arm] BASE ===", flush=True)
        b_cf = [_eval_batch_cf_hits(backbone, tokenizer, cb, prose_ev_chunks, prose_ev_txts, device, amp, prepend_bos) for cb in cf_batches]
        b_mq = [_eval_batch_mquake_hits(backbone, tokenizer, mb, device, amp, prepend_bos) for mb in mq_batches]
        b_z = [_eval_batch_zsre_hits(backbone, tokenizer, zb, bl, device, amp, prepend_bos, loc_gen_len) for zb, bl in zip(z_batches, base_loc_batches)]
        summary["arms"]["base"] = {
            "track1_cf": _aggregate_cf(b_cf, [], base_ppl),
            "track2_mq": _aggregate_mq(b_mq, []),
            "track3_zsre": _aggregate_zsre(b_z, []),
        }

    # --- New Arm 1: `oracle_icl` (100% Ground-Truth Fact in Prompt Ceiling!) ---
    if "oracle_icl" in req_arms and ("oracle_icl" not in summary["arms"] or "oracle_icl" in force_arms):
        print("\n=== [Stage 3 Arm] ORACLE_ICL (100% Ground-Truth Retrieval Ceiling) ===", flush=True)
        oc_cf = [_eval_oracle_icl_cf(backbone, tokenizer, cb, prose_ev_chunks, device, amp, prepend_bos) for cb in cf_batches]
        oc_mq = [_eval_oracle_icl_mq(backbone, tokenizer, mb, device, amp, prepend_bos) for mb in mq_batches]
        oc_z = [_eval_oracle_icl_zsre(backbone, tokenizer, zb, bl, device, amp, prepend_bos, loc_gen_len) for zb, bl in zip(z_batches, base_loc_batches)]
        summary["arms"]["oracle_icl"] = {
            "track1_cf": _aggregate_cf(oc_cf, [], base_ppl),
            "track2_mq": _aggregate_mq(oc_mq, []),
            "track3_zsre": _aggregate_zsre(oc_z, []),
        }
        print(f"[ORACLE_ICL] CF_ParEM={summary['arms']['oracle_icl']['track1_cf']['par_em']['pct']}% "
              f"MQ_2HopEM={summary['arms']['oracle_icl']['track2_mq']['mh_em']['pct']}% "
              f"ZsRE_GenEM={summary['arms']['oracle_icl']['track3_zsre']['gen_em']['pct']}% "
              f"ZsRE_Loc={summary['arms']['oracle_icl']['track3_zsre']['loc_exact']['pct']}%", flush=True)
        summary["mquake_hop_subtypes"] = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)
        summary["stage3_paired_mcnemar"] = _compute_stage3_mcnemar(summary["arms"])
        save_json(out_json, summary)
        _write_stage3_report(summary, out_md)

    # --- New Arm 2: `ike_rag_gated_clean` (QA-Wrapper Stripped + Symmetric `cf_train + cf_qa` Calibration!) ---
    if "ike_rag_gated_clean" in req_arms and ("ike_rag_gated_clean" not in summary["arms"] or "ike_rag_gated_clean" in force_arms):
        print("\n=== [Stage 3 Arm] IKE_RAG_GATED_CLEAN (Wrapper-Stripped + cf_qa Calibrated) ===", flush=True)
        c_ike = CleanDenseIKERetriever(args.ike_model_id, device=device, smoke=args.smoke)
        c_tau, c_stats = c_ike.calibrate_clean_threshold(cf_tr, max_n=(16 if args.smoke else 300))
        summary["clean_ike_retriever"] = {"model_id": args.ike_model_id, "calibrated_tau": c_tau, "calib_stats": c_stats}

        cf_bs, cf_ts, mq_bs, mq_ts, z_bs, z_ts = [], [], [], [], [], []
        for cb in cf_batches:
            bk, ts = c_ike.build_bank(cb); cf_ts.append(ts)
            cf_bs.append(_eval_batch_cf_hits(backbone, tokenizer, cb, prose_ev_chunks, prose_ev_txts, device, amp, prepend_bos, ike=c_ike, ike_bank=bk, ike_tau=c_tau))
        for mb in mq_batches:
            bk, ts = c_ike.build_bank(mb); mq_ts.append(ts)
            mq_bs.append(_eval_batch_mquake_hits(backbone, tokenizer, mb, device, amp, prepend_bos, ike=c_ike, ike_bank=bk, ike_tau=c_tau))
        for zb, bl in zip(z_batches, base_loc_batches):
            bk, ts = c_ike.build_bank(zb); z_ts.append(ts)
            z_bs.append(_eval_batch_zsre_hits(backbone, tokenizer, zb, bl, device, amp, prepend_bos, loc_gen_len, ike=c_ike, ike_bank=bk, ike_tau=c_tau))

        summary["arms"]["ike_rag_gated_clean"] = {
            "tau": c_tau,
            "track1_cf": _aggregate_cf(cf_bs, cf_ts, base_ppl),
            "track2_mq": _aggregate_mq(mq_bs, mq_ts),
            "track3_zsre": _aggregate_zsre(z_bs, z_ts),
        }
        print(f"[IKE_RAG_GATED_CLEAN] CF_ParEM={summary['arms']['ike_rag_gated_clean']['track1_cf']['par_em']['pct']}% "
              f"CF_Ret={summary['arms']['ike_rag_gated_clean']['track1_cf']['nei_prob']['pct']}% "
              f"MQ_2HopEM={summary['arms']['ike_rag_gated_clean']['track2_mq']['mh_em']['pct']}% "
              f"ZsRE_GenEM={summary['arms']['ike_rag_gated_clean']['track3_zsre']['gen_em']['pct']}% "
              f"ZsRE_Loc={summary['arms']['ike_rag_gated_clean']['track3_zsre']['loc_exact']['pct']}%", flush=True)
        del c_ike
        gc.collect()
        if amp:
            torch.cuda.empty_cache()
        summary["mquake_hop_subtypes"] = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)
        summary["stage3_paired_mcnemar"] = _compute_stage3_mcnemar(summary["arms"])
        save_json(out_json, summary)
        _write_stage3_report(summary, out_md)

    # --- New Arms 3 & 4: `fomp_v4_legible` (Route 1) & `fomp_v4_bridge_kv` (Route 2) ---
    need_v4 = [a for a in ("fomp_v4_legible", "fomp_v4_bridge_kv") if a in req_arms and (a not in summary["arms"] or a in force_arms)]
    if need_v4:
        d_m = int(backbone.config.hidden_size)
        n_l = int(backbone.config.num_hidden_layers)
        v4_cfg = FOMPv4Config.for_backbone(d_m, n_l)
        torch.manual_seed(args.seed)
        fomp_v4 = FOMPv4Module(v4_cfg).to(device)
        wrap_v4 = FOMPv4BackboneWrapper(backbone, fomp_v4, freeze_backbone=True)

        v4_ckpt = args.out / "fomp_v4_zero_mq_llama3_8b.pt"
        if any(a in force_arms for a in need_v4) and v4_ckpt.exists():
            v4_ckpt.unlink()
        z_v3_ckpt = None
        if args.stage25_dir is not None and (args.stage25_dir / "fomp_v36_zero_mq_llama3_8b.pt").exists():
            z_v3_ckpt = args.stage25_dir / "fomp_v36_zero_mq_llama3_8b.pt"
        elif (args.out / "fomp_v36_zero_mq_llama3_8b.pt").exists():
            z_v3_ckpt = args.out / "fomp_v36_zero_mq_llama3_8b.pt"

        v4_meta = _train_fomp_v4_zero_mq(
            wrap_v4, tokenizer, cf_tr, prose_tr_chunks, device, amp, args.smoke,
            v4_ckpt_path=v4_ckpt, zero_mq_v3_ckpt=z_v3_ckpt,
            bridge_steps=int(args.bridge_steps), prepend_bos=prepend_bos, deadline=args.deadline,
        )

        for arm_name in need_v4:
            if time.time() > args.deadline - 120:
                break
            mode = "legible" if arm_name == "fomp_v4_legible" else "bridge_kv"
            use_kv = mode == "bridge_kv"
            print(f"\n=== [Stage 3 Arm] {arm_name.upper()} (mode={mode}, 0% MQuAKE & 0% ZsRE!) ===", flush=True)

            cf_bs, cf_ts, mq_bs, mq_ts, z_bs, z_ts = [], [], [], [], [], []
            for cb in cf_batches:
                cm, ts = _compile_v4_batch(wrap_v4, tokenizer, cb, pad_id, prepend_bos, device, precompile_kv=use_kv)
                cf_ts.append(ts)
                cf_bs.append(_eval_v4_cf(wrap_v4, tokenizer, cb, prose_ev_chunks, cm, mode, device, amp, prepend_bos))
            for mb in mq_batches:
                cm, ts = _compile_v4_batch(wrap_v4, tokenizer, mb, pad_id, prepend_bos, device, precompile_kv=use_kv)
                mq_ts.append(ts)
                mq_bs.append(_eval_v4_mq(wrap_v4, tokenizer, mb, cm, mode, device, amp, prepend_bos))
            for zb, bl in zip(z_batches, base_loc_batches):
                cm, ts = _compile_v4_batch(wrap_v4, tokenizer, zb, pad_id, prepend_bos, device, precompile_kv=use_kv)
                z_ts.append(ts)
                z_bs.append(_eval_v4_zsre(wrap_v4, tokenizer, zb, bl, cm, mode, device, amp, prepend_bos, loc_gen_len))

            summary["arms"][arm_name] = {
                "train_meta": v4_meta,
                "mode": mode,
                "zero_mquake_train": True,
                "track1_cf": _aggregate_cf(cf_bs, cf_ts, base_ppl),
                "track2_mq": _aggregate_mq(mq_bs, mq_ts),
                "track3_zsre": _aggregate_zsre(z_bs, z_ts),
            }
            sub_now = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)["per_arm"].get(arm_name, {})
            h1_p = sub_now.get("hop1_edited_em", {}).get("pct", 0.0)
            h2_p = sub_now.get("hop2_edited_em", {}).get("pct", 0.0)
            print(
                f"[{arm_name.upper()}] CF_ParEM={summary['arms'][arm_name]['track1_cf']['par_em']['pct']}% "
                f"CF_Ret={summary['arms'][arm_name]['track1_cf']['nei_prob']['pct']}% "
                f"MQ_2HopEM={summary['arms'][arm_name]['track2_mq']['mh_em']['pct']}% (Hop1={h1_p}%, Hop2={h2_p}%) "
                f"ZsRE_GenEM={summary['arms'][arm_name]['track3_zsre']['gen_em']['pct']}% "
                f"ZsRE_Loc={summary['arms'][arm_name]['track3_zsre']['loc_exact']['pct']}%",
                flush=True,
            )
            summary["mquake_hop_subtypes"] = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)
            summary["stage3_paired_mcnemar"] = _compute_stage3_mcnemar(summary["arms"])
            save_json(out_json, summary)
            _write_stage3_report(summary, out_md)

        del wrap_v4, fomp_v4
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    summary["mquake_hop_subtypes"] = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)
    summary["stage3_paired_mcnemar"] = _compute_stage3_mcnemar(summary["arms"])
    save_json(out_json, summary)
    _write_stage3_report(summary, out_md)
    print("\n=== STAGE 3 MULTI-HOP & CLEAN-BASELINE AUDIT COMPLETE ->", out_md, "===", flush=True)


if __name__ == "__main__":
    main()
