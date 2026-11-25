"""Stage 3.2: Pure-Activation 3-Route FOMP-v4.2 (`Single-Hop v3.6` + `Hop-1 Legible` + `Hop-2 Logit-Lens`) & 4-Shot IKE Audit.

Unifies all three mechanisms with **0 extra prompt/KV tokens** and **strictly 0% MQuAKE & 0% ZsRE train**:
  1. **Route 1 — Single-Hop & Locality (`mh_prob <= 0.80`)**:
     Executes **100% unmodified `FOMP-v3.6-ZeroMQ`** in 1 pass (`0.000000` logit diff vs `v3.6`, eliminating all
     `v4.0` single-hop & ZsRE regressions).
  2. **Route 2 — `Hop-1 Edited` Compositional Bridge (`mh_prob > 0.80` AND `g_span > 0.20`)**:
     Read-Equivalent convex mid-layer entity replacement (`L10, 13, 16, 19`, `0` extra tokens).
  3. **Route 3 — `Hop-2 Edited` Tuned Logit-Lens Implicit Secondary Retrieval (`mh_prob > 0.80` AND `g_span <= 0.20`)**:
     Probes `L16, 19, 22, 25` via zero-init low-rank residual Tuned-Lens + frozen `final_norm + lm_head` (128K open
     vocab, masking all surface query tokens), cross-locks with outer relation (`max(q_tail, q_hop2_outer) . k_fam`),
     and injects the matched Hop-2 edit across all 5 `v3.6` layers at `\nAnswer:` (`0` extra tokens).
  4. **Full-Strength Non-Strawman 4-Shot IKE Baselines (`oracle_ike_std`, `oracle_ike_chat`, `ike_rag_always_demo`,
     `ike_rag_gated_clean_demo`)**: Official 4-shot Copy/Update/Retain demonstrations (`cf_train[:4]`) in BOTH Zheng
     et al. (2023) continuation format and native Llama-3 chat template with pre-flight Oracle verification.
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

from pulse.fomp_v3 import (  # noqa: E402
    FOMPv3BackboneWrapper,
    FOMPv3Config,
    FOMPv3Module,
    _causal_window_mean,
    _extract_tail_window,
)
from pulse.fomp_v4 import (  # noqa: E402
    FOMPv4BackboneWrapper,
    FOMPv4Config,
    FOMPv4Module,
    _project_logit_lens,
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
    exact_mcnemar_p,
)


# ==============================================================================
# Clean Wrapper-Stripped Dense IKE Retriever & 4-Shot IKE / Chat Formatters
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


def format_ike_fact_line(cue: str, target_new: str) -> str:
    """Format a clean single-sentence fact statement `New Fact: ...` following Zheng et al. (EMNLP 2023)."""
    c = cue.strip()
    y = target_new.strip()
    if c.lower().startswith("question:"):
        q = strip_qa_wrapper(c)
        return f"{q}? {y}"
    return f"{c} {y}"


class CleanDenseIKERetriever(DenseIKERetriever):
    """Repaired full-strength Gated RAG: strips QA boilerplate before encoding & calibrates on `cf_train + cf_qa`."""

    def encode_clean(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        return self.encode([strip_qa_wrapper(t) for t in texts], batch_size=batch_size)

    def calibrate_clean_threshold(self, cf_train: List[EditRecord], max_n: int = 300) -> Tuple[float, Dict[str, float]]:
        sub = cf_train[: min(max_n, len(cf_train))]
        half = len(sub) // 2
        cues = [r.cue for r in sub[:half]] + [f"Question: {r.cue.rstrip('.')}?\nAnswer:" for r in sub[half:]]
        paras = [r.paraphrases[0] for r in sub[:half]] + [f"Question: {r.paraphrases[0].rstrip('.?')}?\nAnswer:" for r in sub[half:]]
        neis = [r.neighborhoods[0] for r in sub[:half]] + [f"Question: {r.neighborhoods[0].rstrip('.?')}?\nAnswer:" for r in sub[half:]]
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(sub)).tolist()
        cross_qa = [f"Question: {sub[j].paraphrases[0].rstrip('.?')}?\nAnswer:" for j in perm[: max(1, half)]]

        k_cues = self.encode_clean(cues)
        q_pos = self.encode_clean(paras)
        q_nei = self.encode_clean(neis)
        q_crs = self.encode_clean(cross_qa[: min(len(cross_qa), max(1, len(sub) // 4))])
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
        facts = [format_ike_fact_line(r.cue, r.target_new) for r in recs]
        keys = self.encode_clean(cues)
        sync(self.device)
        return {"keys": keys, "facts": facts}, round(time.perf_counter() - t0, 4)

    def retrieve_facts(
        self,
        query: str,
        bank: Dict[str, Any],
        top_k: int = 2,
        threshold: Optional[float] = None,
    ) -> List[str]:
        q_emb = self.encode_clean([query])[0]
        sims = bank["keys"] @ q_emb
        k = min(top_k, sims.shape[0])
        vals, idxs = sims.topk(k=k)
        return [bank["facts"][idx] for v, idx in zip(vals.tolist(), idxs.tolist()) if threshold is None or v >= threshold]


class IKEPromptFormatter:
    """4-Shot IKE Formatter (`Copy + Update + Retain` strictly from `cf_train[:4]`) supporting BOTH:
      - `fmt = 'std'`  : Zheng et al. (EMNLP 2023) `New Fact: ... / Prompt: ...` continuation format.
      - `fmt = 'chat'` : Native `Llama-3-8B-Instruct` chat template with 4-shot User/Assistant IKE turns."""

    def __init__(self, tokenizer: Any, cf_demos: Sequence[EditRecord], prepend_bos: bool) -> None:
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        d = list(cf_demos[:4])
        while len(d) < 4:
            d.append(d[-1])
        self.demos = d
        self.has_chat = bool(getattr(tokenizer, "chat_template", None))

        # Stop IDs for chat generation (`eos_token_id` + `<|eot_id|>`)
        stops: set = set()
        if getattr(tokenizer, "eos_token_id", None) is not None:
            e = tokenizer.eos_token_id
            stops.update(e if isinstance(e, list) else [int(e)])
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if isinstance(eot, int) and eot >= 0 and eot != getattr(tokenizer, "unk_token_id", -1):
                    stops.add(eot)
            except Exception:
                pass
        self.stop_ids = stops

        d0, d1, d2, d3 = self.demos
        self.cloze_hdr = "\n\n".join([
            f"New Fact: {format_ike_fact_line(d0.cue, d0.target_new)}\nPrompt: {d0.cue} {d0.target_new.strip()}",
            f"New Fact: {format_ike_fact_line(d1.cue, d1.target_new)}\nPrompt: {d1.paraphrases[0]} {d1.target_new.strip()}",
            f"New Fact: {format_ike_fact_line(d2.cue, d2.target_new)}\nPrompt: {d2.neighborhoods[0]} {d2.target_true.strip()}",
            f"New Fact: {format_ike_fact_line(d3.cue, d3.target_new)}\nPrompt: {d3.paraphrases[0]} {d3.target_new.strip()}",
        ])
        self.qa_hdr = "\n\n".join([
            f"New Fact: {format_ike_fact_line(d0.cue, d0.target_new)}\nQuestion: {d0.cue.rstrip('.?')}?\nAnswer: {d0.target_new.strip()}",
            f"New Fact: {format_ike_fact_line(d1.cue, d1.target_new)}\nQuestion: {d1.paraphrases[0].rstrip('.?')}?\nAnswer: {d1.target_new.strip()}",
            f"New Fact: {format_ike_fact_line(d2.cue, d2.target_new)}\nQuestion: {d2.neighborhoods[0].rstrip('.?')}?\nAnswer: {d2.target_true.strip()}",
            "New Fact: The country of citizenship of Alan Turing is France\nQuestion: What is the capital of the country of citizenship of Alan Turing?\nAnswer: Paris",
        ])
        self.chat_prefix_msgs = [
            {
                "role": "system",
                "content": (
                    "Complete the Prompt or answer the Question with ONLY the exact short target entity or phrase. "
                    "Output no extra words, punctuation, or explanation. Strictly follow New Fact when it applies to "
                    "the queried entity and relation (including multi-step questions); if the query asks about an "
                    "unrelated entity, answer with the original factual answer."
                ),
            },
            {"role": "user", "content": f"New Fact: {format_ike_fact_line(d0.cue, d0.target_new)}\nPrompt: {d0.cue}"},
            {"role": "assistant", "content": d0.target_new.strip()},
            {"role": "user", "content": f"New Fact: {format_ike_fact_line(d1.cue, d1.target_new)}\nPrompt: {d1.paraphrases[0]}"},
            {"role": "assistant", "content": d1.target_new.strip()},
            {"role": "user", "content": f"New Fact: {format_ike_fact_line(d2.cue, d2.target_new)}\nPrompt: {d2.neighborhoods[0]}"},
            {"role": "assistant", "content": d2.target_true.strip()},
            {"role": "user", "content": f"New Fact: {format_ike_fact_line(d3.cue, d3.target_new)}\nQuestion: {d3.paraphrases[0].rstrip('.?')}?"},
            {"role": "assistant", "content": d3.target_new.strip()},
            {"role": "user", "content": "New Fact: The country of citizenship of Alan Turing is France\nQuestion: What is the capital of the country of citizenship of Alan Turing?"},
            {"role": "assistant", "content": "Paris"},
        ]

    def encode_query(self, raw_q: str, facts: Sequence[str], fmt: str) -> Tuple[List[int], bool, int]:
        """Return `(prompt_ids, strip_leading_space_on_target, extra_tokens_over_raw)`.
        When `facts` is empty (Gated RAG rejected!), returns untouched `raw_q` (`0` extra tokens!)."""
        raw_ids = _enc_prompt(self.tokenizer, raw_q, self.prepend_bos)
        if not facts:
            return raw_ids, False, 0

        f_block = "\n".join(f"New Fact: {f}" for f in facts)
        is_qa = raw_q.lstrip().lower().startswith("question:")

        if fmt == "chat" and self.has_chat:
            user_q = f"Question: {strip_qa_wrapper(raw_q)}?" if is_qa else f"Prompt: {raw_q}"
            msgs = list(self.chat_prefix_msgs) + [{"role": "user", "content": f"{f_block}\n{user_q}"}]
            res = self.tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            if isinstance(res, dict) or hasattr(res, "keys"):
                p_ids = [int(x) for x in res["input_ids"]]
            else:
                p_ids = [int(x) for x in res]
            return p_ids, True, max(0, len(p_ids) - len(raw_ids))

        if is_qa:
            txt = f"{self.qa_hdr}\n\n{f_block}\n{raw_q}"
        else:
            txt = f"{self.cloze_hdr}\n\n{f_block}\nPrompt: {raw_q}"
        p_ids = _enc_prompt(self.tokenizer, txt, self.prepend_bos)
        return p_ids, False, max(0, len(p_ids) - len(raw_ids))

    def encode_target(self, target_str: str, strip_space: bool) -> List[int]:
        s = target_str.strip() if strip_space else target_str
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        return ids if ids else [0]


@torch.no_grad()
def _greedy_gen_ike(
    model: nn.Module,
    prompt_ids: List[int],
    max_new_tokens: int,
    device: torch.device,
    amp: bool,
    stop_ids: Optional[set] = None,
) -> List[int]:
    if hasattr(model, "model"):
        cur = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out0 = model(input_ids=cur, use_cache=True)
        pkv = getattr(out0, "past_key_values", None)
        if pkv is not None:
            nxt = int(out0.logits[0, -1].argmax(dim=-1).item())
            if stop_ids and nxt in stop_ids:
                return []
            gen = [nxt]
            for _ in range(1, max_new_tokens):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out = model(input_ids=torch.tensor([[gen[-1]]], device=device, dtype=torch.long), past_key_values=pkv, use_cache=True)
                    pkv = getattr(out, "past_key_values", None)
                    nxt = int(out.logits[0, -1].argmax(dim=-1).item())
                if stop_ids and nxt in stop_ids:
                    break
                gen.append(nxt)
            return gen
    return _greedy_gen_any(model, prompt_ids, max_new_tokens, device, amp)


def _eval_ike_arm_cf(
    model: nn.Module,
    fmt_helper: IKEPromptFormatter,
    cf_batch: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    fmt: str,
    device: torch.device,
    amp: bool,
    ike: Optional[CleanDenseIKERetriever] = None,
    ike_bank: Optional[Dict[str, Any]] = None,
    ike_tau: Optional[float] = None,
) -> Dict[str, Any]:
    """Unified CounterFact evaluator for `oracle_ike_*` (`ike=None`) and `ike_rag_*_demo` (`ike!=None`)."""
    model.eval()
    tok = fmt_helper.tokenizer
    extra_toks: List[int] = []
    stops = fmt_helper.stop_ids if fmt == "chat" else None

    def _run_items(mode_q: str, want_new: bool) -> Dict[str, List[int]]:
        prob_hits, em_hits = [], []
        for r in cf_batch:
            raw_q = r.cue if mode_q == "eff" else (r.paraphrases[0] if mode_q == "par" else r.neighborhoods[0])
            if ike is None:
                facts = [format_ike_fact_line(r.cue, r.target_new)]
            else:
                assert ike_bank is not None
                facts = ike.retrieve_facts(raw_q, ike_bank, top_k=2, threshold=ike_tau)
            p_ids, strip_sp, ex = fmt_helper.encode_query(raw_q, facts, fmt)
            extra_toks.append(ex)

            tn_ids = fmt_helper.encode_target(r.target_new, strip_sp)
            to_ids = fmt_helper.encode_target(r.target_true, strip_sp)
            lp_n = _score_logprob_any(model, p_ids, tn_ids, device, amp)
            lp_o = _score_logprob_any(model, p_ids, to_ids, device, amp)
            prob_hits.append(int((lp_n > lp_o) if want_new else (lp_o > lp_n)))

            want_s = (r.target_new if want_new else r.target_true).strip().lower()
            want_len = len(tn_ids if want_new else to_ids)
            g_ids = _greedy_gen_ike(model, p_ids, max(2, want_len + 2), device, amp, stop_ids=stops)
            em_hits.append(int(tok.decode(g_ids, skip_special_tokens=True).strip().lower().startswith(want_s)))
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


def _eval_ike_arm_mq(
    model: nn.Module,
    fmt_helper: IKEPromptFormatter,
    mq_batch: List[EditRecord],
    fmt: str,
    device: torch.device,
    amp: bool,
    ike: Optional[CleanDenseIKERetriever] = None,
    ike_bank: Optional[Dict[str, Any]] = None,
    ike_tau: Optional[float] = None,
) -> Dict[str, Any]:
    model.eval()
    tok = fmt_helper.tokenizer
    stops = fmt_helper.stop_ids if fmt == "chat" else None
    s_prob, s_em, m_prob, m_em = [], [], [], []
    extra_toks: List[int] = []
    for r in mq_batch:
        f_sh = [format_ike_fact_line(r.cue, r.target_new)] if ike is None else ike.retrieve_facts(r.cue, ike_bank, top_k=2, threshold=ike_tau)  # type: ignore[arg-type]
        c_ids, strip_c, ex_c = fmt_helper.encode_query(r.cue, f_sh, fmt)
        extra_toks.append(ex_c)
        tn_ids = fmt_helper.encode_target(r.target_new, strip_c)
        to_ids = fmt_helper.encode_target(r.target_true, strip_c)
        s_prob.append(int(_score_logprob_any(model, c_ids, tn_ids, device, amp) > _score_logprob_any(model, c_ids, to_ids, device, amp)))
        g_c = _greedy_gen_ike(model, c_ids, max(2, len(tn_ids) + 2), device, amp, stop_ids=stops)
        s_em.append(int(tok.decode(g_c, skip_special_tokens=True).strip().lower().startswith(r.target_new.strip().lower())))

        f_mh = [format_ike_fact_line(r.cue, r.target_new)] if ike is None else ike.retrieve_facts(r.paraphrases[0], ike_bank, top_k=2, threshold=ike_tau)  # type: ignore[arg-type]
        q_ids, strip_q, ex_q = fmt_helper.encode_query(r.paraphrases[0], f_mh, fmt)
        extra_toks.append(ex_q)
        an_ids = fmt_helper.encode_target(r.multihop_answer_new, strip_q)
        ao_ids = fmt_helper.encode_target(r.multihop_answer_true, strip_q)
        m_prob.append(int(_score_logprob_any(model, q_ids, an_ids, device, amp) > _score_logprob_any(model, q_ids, ao_ids, device, amp)))
        g_q = _greedy_gen_ike(model, q_ids, max(2, len(an_ids) + 2), device, amp, stop_ids=stops)
        m_em.append(int(tok.decode(g_q, skip_special_tokens=True).strip().lower().startswith(r.multihop_answer_new.strip().lower())))
    return {
        "sh_prob": s_prob, "sh_em": s_em, "mh_prob": m_prob, "mh_em": m_em,
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)), 2),
    }


def _eval_ike_arm_zsre(
    model: nn.Module,
    fmt_helper: IKEPromptFormatter,
    z_batch: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    base_chat_loc_strs: Optional[List[str]],
    fmt: str,
    device: torch.device,
    amp: bool,
    loc_gen_len: int = 6,
    ike: Optional[CleanDenseIKERetriever] = None,
    ike_bank: Optional[Dict[str, Any]] = None,
    ike_tau: Optional[float] = None,
) -> Dict[str, Any]:
    model.eval()
    tok = fmt_helper.tokenizer
    stops = fmt_helper.stop_ids if fmt == "chat" else None
    r_prob, r_em, g_prob, g_em, loc_exact = [], [], [], [], []
    loc_tok_m = loc_tok_t = 0
    extra_toks: List[int] = []
    for i, r in enumerate(z_batch):
        tn_s = r.target_new.strip().lower()

        f_r = [format_ike_fact_line(r.cue, r.target_new)] if ike is None else ike.retrieve_facts(r.cue, ike_bank, top_k=2, threshold=ike_tau)  # type: ignore[arg-type]
        c_ids, strip_r, ex_r = fmt_helper.encode_query(r.cue, f_r, fmt)
        extra_toks.append(ex_r)
        tn_r = fmt_helper.encode_target(r.target_new, strip_r)
        to_r = fmt_helper.encode_target(r.target_true, strip_r)
        r_prob.append(int(_score_logprob_any(model, c_ids, tn_r, device, amp) > _score_logprob_any(model, c_ids, to_r, device, amp)))
        g_c = _greedy_gen_ike(model, c_ids, max(2, len(tn_r) + 2), device, amp, stop_ids=stops if strip_r else None)
        r_em.append(int(tok.decode(g_c, skip_special_tokens=True).strip().lower().startswith(tn_s)))

        f_g = [format_ike_fact_line(r.cue, r.target_new)] if ike is None else ike.retrieve_facts(r.paraphrase, ike_bank, top_k=2, threshold=ike_tau)  # type: ignore[arg-type]
        p_ids, strip_g, ex_g = fmt_helper.encode_query(r.paraphrase, f_g, fmt)
        extra_toks.append(ex_g)
        tn_g = fmt_helper.encode_target(r.target_new, strip_g)
        to_g = fmt_helper.encode_target(r.target_true, strip_g)
        g_prob.append(int(_score_logprob_any(model, p_ids, tn_g, device, amp) > _score_logprob_any(model, p_ids, to_g, device, amp)))
        g_p = _greedy_gen_ike(model, p_ids, max(2, len(tn_g) + 2), device, amp, stop_ids=stops if strip_g else None)
        g_em.append(int(tok.decode(g_p, skip_special_tokens=True).strip().lower().startswith(tn_s)))

        f_l = [format_ike_fact_line(r.cue, r.target_new)] if ike is None else ike.retrieve_facts(r.loc_prompt, ike_bank, top_k=2, threshold=ike_tau)  # type: ignore[arg-type]
        l_ids, strip_l, ex_l = fmt_helper.encode_query(r.loc_prompt, f_l, fmt)
        extra_toks.append(ex_l)
        if strip_l and base_chat_loc_strs is not None:
            # Fair chat-template locality comparison: compare normalized entity answer vs unedited chat-mode Base!
            g_l = _greedy_gen_ike(model, l_ids, loc_gen_len + 2, device, amp, stop_ids=stops)
            pred_s = tok.decode(g_l, skip_special_tokens=True).strip().lower()
            ref_s = base_chat_loc_strs[i]
            hit = int(pred_s == ref_s or (len(ref_s) > 0 and pred_s.startswith(ref_s)))
            loc_exact.append(hit)
            loc_tok_m += hit
            loc_tok_t += 1
        else:
            g_l = _greedy_gen_ike(model, l_ids, loc_gen_len, device, amp, stop_ids=None)
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


def _preflight_verify_ike_formats(
    model: nn.Module,
    fmt_helper: IKEPromptFormatter,
    cf_train: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    smoke: bool,
) -> Tuple[str, Dict[str, Any]]:
    """Run held-out `cf_train` pre-flight check on BOTH `std` (Zheng et al. 4-shot) and `chat` (Llama-3 Chat 4-shot)."""
    val_slice = cf_train[4 : min(len(cf_train), 10 if smoke else 36)]
    stats: Dict[str, Any] = {}
    best_fmt = "std"
    best_score = -1.0
    for cand in ("std", "chat"):
        if cand == "chat" and not fmt_helper.has_chat:
            continue
        r_cf = _eval_ike_arm_cf(model, fmt_helper, val_slice, prose_chunks[:2], cand, device, amp, ike=None)
        eff_pct = round(100.0 * float(np.mean(r_cf["eff_em"])), 2)
        par_pct = round(100.0 * float(np.mean(r_cf["par_em"])), 2)
        ret_pct = round(100.0 * float(np.mean(r_cf["nei_prob"])), 2)
        combo = 0.4 * eff_pct + 0.4 * par_pct + 0.2 * ret_pct
        stats[cand] = {
            "preflight_n": len(val_slice),
            "eff_em_pct": eff_pct,
            "par_em_pct": par_pct,
            "retain_prob_pct": ret_pct,
            "avg_extra_tokens": r_cf["avg_extra_prompt_tokens"],
        }
        print(f"[IKE Preflight `{cand}` (n={len(val_slice)})] Oracle Eff_EM={eff_pct}% | Par_EM={par_pct}% | Retain={ret_pct}%", flush=True)
        if combo > best_score:
            best_score = combo
            best_fmt = cand
    print(f"[IKE Preflight Winner] Selected `{best_fmt}` for `ike_rag_*_demo` baselines!", flush=True)
    return best_fmt, stats


# ==============================================================================
# Self-Supervised Zero-MQuAKE / Zero-ZsRE FOMP-v4.1 Training (`cf_train` ONLY!)
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


def _make_compositional_qa(outer_rec: EditRecord, inner_np: str, use_para: bool = False) -> str:
    """Synthesize a natural 2-hop compositional question wrapping `inner_np` inside `outer_rec`'s relation (`0%` MQuAKE!)."""
    if use_para and outer_rec.paraphrases:
        rep_p = _replace_ci(outer_rec.paraphrases[0].strip().rstrip(".?!"), outer_rec.subject, inner_np)
        if rep_p is not None:
            return f"Question: {rep_p}?\nAnswer:"
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
    """Train FOMP-v4.2 (3-Route Pure Activation) strictly on `cf_train` (`0%` MQuAKE, `0%` ZsRE!)."""
    if v4_ckpt_path.exists():
        ck = torch.load(v4_ckpt_path, map_location=device, weights_only=False)
        wrap_v4.fomp.load_state_dict(ck["fomp"])
        print(f"[FOMP-v4.2] Loaded cached ZeroMQ v4.2 checkpoint -> {v4_ckpt_path}", flush=True)
        return ck.get("train_meta", {"loaded_cached": True})

    t0 = time.perf_counter()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id if prepend_bos else None

    # --- Phase 0: Load or train Direct `FOMP-v3.6-ZeroMQ` core (`0%` MQuAKE & `0%` ZsRE) and freeze it! ---
    if zero_mq_v3_ckpt is not None and zero_mq_v3_ckpt.exists() and not smoke:
        ck3 = torch.load(zero_mq_v3_ckpt, map_location=device, weights_only=False)
        wrap_v4.fomp.direct.load_state_dict(ck3["fomp"])
        print(f"[FOMP-v4.2] Loaded frozen Direct core from {zero_mq_v3_ckpt} (100% single-hop & ZsRE locality preserved!)", flush=True)
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

    # --- Phase 1: Pre-cache single-hop `cf_train` states for Route 2, MH Gate, and Route 3 Outer-Relation (`0%` MQ!) ---
    cfg = wrap_v4.fomp.config
    win_s = cfg.span_window
    n_pool = min(len(cf_train), 20 if smoke else 1600)
    sub_cf = cf_train[:n_pool]
    rng = np.random.default_rng(2026)

    cue_sp_ids: List[List[int]] = []
    np_pos_list: List[str] = []
    pos_q_ids: List[List[int]] = []
    pos_sp_end: List[int] = []
    neg_q_ids: List[List[int]] = []
    neg_sp_end: List[int] = []
    teach_q_ids: List[List[int]] = []
    tgt_ent_ids: List[List[int]] = []

    sh_cue_ids: List[List[int]] = []
    sh_par_ids: List[List[int]] = []
    sh_qa_np_ids: List[List[int]] = []
    sh_qa_par_ids: List[List[int]] = []

    for i, r in enumerate(sub_cf):
        tmpl = _OUTER_TEMPLATES[i % len(_OUTER_TEMPLATES)]
        prefix_part, suffix_part = tmpl.split("{NP}")

        c_str = strip_cue_copula(r.cue)
        cue_sp_ids.append(_enc_prompt(tokenizer, c_str, prepend_bos))

        use_para = (i % 2 == 1) and len(r.paraphrases) > 0
        np_pos = _clause_to_embedded_np(r.paraphrases[0] if use_para else r.cue)
        np_pos_list.append(np_pos)
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

        p0 = r.paraphrases[0] if r.paraphrases else r.cue
        sh_cue_ids.append(_enc_prompt(tokenizer, r.cue, prepend_bos))
        sh_par_ids.append(_enc_prompt(tokenizer, p0, prepend_bos))
        sh_qa_np_ids.append(_enc_prompt(tokenizer, f"Question: What is {_clause_to_embedded_np(r.cue)}?\nAnswer:", prepend_bos))
        sh_qa_par_ids.append(_enc_prompt(tokenizer, f"Question: {p0.rstrip('.?')}?\nAnswer:", prepend_bos))

    print(f"[FOMP-v4.2] Pre-caching Stage-A span, 1-hop/2-hop gate & Hop-2 outer-rel states on {n_pool} cf_train rows (0% MQuAKE)...", flush=True)
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

    def _enc_full_seq_at_layer(seqs: List[List[int]], layer_i: int, max_len: int = 38) -> Tuple[torch.Tensor, torch.Tensor]:
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

    h_cue_sw, m_cue_sw = _enc_windows_at_layer(cue_sp_ids, [len(x) - 1 for x in cue_sp_ids], cfg.bridge_read_layer, win_s)
    h_pos_sw, m_pos_sw = _enc_windows_at_layer(pos_q_ids, pos_sp_end, cfg.bridge_read_layer, win_s)
    h_neg_sw, m_neg_sw = _enc_windows_at_layer(neg_q_ids, neg_sp_end, cfg.bridge_read_layer, win_s)

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

    # Pre-cache `L16` sequences for the 4 single-hop (`y=0`) classes + compute frozen `k_fam_tr` for orthogonal 2-hop pairing
    h16_sh_cue, m16_sh_cue = _enc_full_seq_at_layer(sh_cue_ids, cfg.read_layer)
    h16_sh_par, m16_sh_par = _enc_full_seq_at_layer(sh_par_ids, cfg.read_layer)
    h16_sh_qnp, m16_sh_qnp = _enc_full_seq_at_layer(sh_qa_np_ids, cfg.read_layer)
    h16_sh_qpr, m16_sh_qpr = _enc_full_seq_at_layer(sh_qa_par_ids, cfg.read_layer)

    kf_parts = []
    win_c = wrap_v4.fomp.direct.config.key_ctx_window
    with torch.no_grad():
        for s0 in range(0, n_pool, bs_enc):
            hb = h16_sh_cue[s0 : s0 + bs_enc].to(device)
            mb = m16_sh_cue[s0 : s0 + bs_enc].to(device)
            li_b = (mb.long().sum(dim=1) - 1).clamp_min(0)
            ar_b = torch.arange(hb.shape[0], device=device)
            hctx_b = _causal_window_mean(hb, win_c)
            kf_parts.append(wrap_v4.fomp.direct.project_key(hb[ar_b, li_b], hctx_b[ar_b, li_b]).cpu())
    k_fam_tr = torch.cat(kf_parts, dim=0)  # [n_pool, key_dim]

    # Pair each inner NP with an orthogonal outer relation (`cos < 0.25`) so `project_hop2_outer_rel` MUST read outer rel!
    sim_kf = (k_fam_tr @ k_fam_tr.T).numpy()
    outer_j_arr = np.zeros((n_pool,), dtype=np.int64)
    base_off = max(1, n_pool // 3)
    for i in range(n_pool):
        pick = (i + base_off) % n_pool
        for step_k in range(1, min(40, n_pool)):
            cand = (i + base_off + step_k * 13) % n_pool
            if cand != i and sim_kf[i, cand] < 0.25:
                pick = cand
                break
        outer_j_arr[i] = pick

    comp_q_ids = [
        _enc_prompt(tokenizer, _make_compositional_qa(sub_cf[int(outer_j_arr[i])], np_pos_list[i], use_para=(i % 2 == 1)), prepend_bos)
        for i in range(n_pool)
    ]
    h16_mh_cmp, m16_mh_cmp = _enc_full_seq_at_layer(comp_q_ids, cfg.read_layer)
    h16_mh_tmp, m16_mh_tmp = _enc_full_seq_at_layer(pos_q_ids, cfg.read_layer)

    dummy_direct = {
        "k_fam": torch.zeros((1, 1, cfg.key_dim), device=device),
        "k_ent": torch.zeros((1, 1, cfg.ent_dim), device=device),
        "v_bank": torch.zeros((1, 1, cfg.max_tgt_tokens, cfg.val_dim), device=device),
        "e_bank": torch.zeros((1, 1, cfg.max_tgt_tokens, cfg.d_model), device=device),
        "tgt_mask": torch.ones((1, 1, cfg.max_tgt_tokens), device=device),
        "g_ext": torch.tensor(1.0, device=device),
    }

    n_tr = max(8, int(n_pool * 0.90)) if not smoke else n_pool
    val_idx = np.arange(n_tr, n_pool) if n_tr < n_pool else np.arange(0, min(8, n_pool))

    phase1_params = [p for n, p in wrap_v4.fomp.named_parameters() if not n.startswith(("direct.", "lens_"))]
    lens_params = [p for n, p in wrap_v4.fomp.named_parameters() if n.startswith("lens_")]
    lr = 1.0e-3 if not smoke else 2.0e-3
    opt = torch.optim.AdamW(phase1_params, lr=lr, weight_decay=1e-4)
    steps = 8 if smoke else bridge_steps
    ema_start = max(2, int(steps * 0.65))
    ema_decay = 0.985
    ema_state: Optional[Dict[str, torch.Tensor]] = None

    nc_bs = min(16 if smoke else 128, n_tr)
    e2e_bs = min(4 if smoke else 6, n_tr)
    n_pos_e2e = max(2, e2e_bs - 2)
    last_kl = 0.0
    last_nce_a = 0.0
    last_mh_bce = 0.0
    last_h2_rel = 0.0

    for st in range(1, steps + 1):
        if time.time() > deadline - 300:
            break

        wrap_v4.fomp.train()
        wrap_v4.fomp.direct.eval()
        opt.zero_grad(set_to_none=True)

        idx_nc = rng.choice(n_tr, size=nc_bs, replace=False)
        lbl_nc = torch.arange(nc_bs, device=device)
        eye_m = torch.eye(nc_bs, device=device, dtype=torch.bool)

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

            # 2. Strict Syntactic Multi-Hop Detector (4 single-hop neg classes vs 2 compositional multi-hop pos classes)
            lg_sh1 = wrap_v4.fomp.compute_multihop_logits(h16_sh_cue[idx_nc].to(device), m16_sh_cue[idx_nc].to(device))
            lg_sh2 = wrap_v4.fomp.compute_multihop_logits(h16_sh_par[idx_nc].to(device), m16_sh_par[idx_nc].to(device))
            lg_sh3 = wrap_v4.fomp.compute_multihop_logits(h16_sh_qnp[idx_nc].to(device), m16_sh_qnp[idx_nc].to(device))
            lg_sh4 = wrap_v4.fomp.compute_multihop_logits(h16_sh_qpr[idx_nc].to(device), m16_sh_qpr[idx_nc].to(device))
            lg_mh1 = wrap_v4.fomp.compute_multihop_logits(h16_mh_cmp[idx_nc].to(device), m16_mh_cmp[idx_nc].to(device))
            lg_mh2 = wrap_v4.fomp.compute_multihop_logits(h16_mh_tmp[idx_nc].to(device), m16_mh_tmp[idx_nc].to(device))

            lg_neg_all = torch.cat([lg_sh1, lg_sh2, lg_sh3, lg_sh4], dim=0)
            lg_pos_all = torch.cat([lg_mh1, lg_mh2], dim=0)
            loss_mh = (
                1.5 * F.binary_cross_entropy_with_logits(lg_neg_all, torch.zeros_like(lg_neg_all))
                + 1.0 * F.binary_cross_entropy_with_logits(lg_pos_all, torch.ones_like(lg_pos_all))
                + 0.5 * F.relu(lg_neg_all + 2.0).pow(2).mean()
                + 0.5 * F.relu(2.0 - lg_pos_all).pow(2).mean()
            )

            # 2b. Route-3 Outer-Relation Projector (`k_fam` space, relation-collision masked; NEVER invoked on single-hop!)
            k_out_cp = k_fam_tr[outer_j_arr[idx_nc]].to(device)
            k_out_sh = k_fam_tr[idx_nc].to(device)
            q_out_cp = wrap_v4.fomp.project_hop2_outer_rel(h16_mh_cmp[idx_nc].to(device), m16_mh_cmp[idx_nc].to(device))
            q_out_sh = wrap_v4.fomp.project_hop2_outer_rel(h16_sh_qpr[idx_nc].to(device), m16_sh_qpr[idx_nc].to(device))
            col_cp = (torch.matmul(k_out_cp, k_out_cp.T) > 0.70) & (~eye_m)
            col_sh = (torch.matmul(k_out_sh, k_out_sh.T) > 0.70) & (~eye_m)
            lg_ocp = (torch.matmul(q_out_cp, k_out_cp.T) / 0.08).masked_fill(col_cp, -1e4)
            lg_osh = (torch.matmul(q_out_sh, k_out_sh.T) / 0.08).masked_fill(col_sh, -1e4)
            loss_hop2_rel = 0.5 * (F.cross_entropy(lg_ocp, lbl_nc) + F.cross_entropy(lg_osh, lbl_nc))

            # 3. End-to-End Self-Supervised Frozen-Teacher Read-Equivalent KL Distillation (`every 2nd step`)
            loss_e2e = torch.tensor(0.0, device=device)
            if st % 2 == 1 or smoke:
                m_ep = min(4 if smoke else 16, n_tr)
                ch_e2e = rng.choice(n_tr, size=e2e_bs, replace=False).tolist()
                bank_idx = np.zeros((e2e_bs, m_ep), dtype=np.int64)
                for bi, ti in enumerate(ch_e2e):
                    bank_idx[bi, 0] = ti
                    if m_ep > 1:
                        others = [x for x in rng.choice(n_tr, size=m_ep, replace=False).tolist() if x != ti][: m_ep - 1]
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
                        other_i = int((ti + 7) % n_tr)
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

            loss = loss_e2e + 0.8 * loss_nce_a + 3.0 * loss_marg_a + 1.5 * loss_tau_a + 1.2 * loss_mh + 0.8 * loss_hop2_rel

        loss.backward()
        torch.nn.utils.clip_grad_norm_(phase1_params, 1.0)
        opt.step()

        if st >= ema_start:
            with torch.no_grad():
                if ema_state is None:
                    ema_state = {k: v.detach().float().clone() for k, v in wrap_v4.fomp.state_dict().items()}
                else:
                    for k, v in wrap_v4.fomp.state_dict().items():
                        ema_state[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)

        last_nce_a = float(loss_nce_a.item())
        last_mh_bce = float(loss_mh.item())
        last_h2_rel = float(loss_hop2_rel.item())
        if st == 1 or st % (4 if smoke else 150) == 0 or st == steps:
            tsr = wrap_v4.fomp.span_rel_tau.clamp(cfg.min_span_rel_tau, 0.90).item()
            tse = wrap_v4.fomp.span_ent_tau.clamp(cfg.min_span_ent_tau, 0.90).item()
            al0 = torch.sigmoid(wrap_v4.fomp.bridge_alpha_logit[0]).item()
            print(
                f"  [FOMP-v4.2 Bridge st={st:3d}/{steps}] kl_leg={last_kl:.3f} "
                f"nce_sp={last_nce_a:.3f} mh={last_mh_bce:.3f} h2_rel={last_h2_rel:.3f} "
                f"tau_sp=({tsr:.2f},{tse:.2f}) a0={al0:.2f}",
                flush=True,
            )

    if ema_state is not None:
        wrap_v4.fomp.load_state_dict({k: v.to(dtype=wrap_v4.fomp.state_dict()[k].dtype, device=device) for k, v in ema_state.items()})

    # --- Phase 2: Self-Supervised Residual Tuned-Lens Alignment (`sh_cue + prose` Belrose KL + known latent entity) ---
    print(f"[FOMP-v4.2 Phase 2] Training low-rank residual Tuned-Lens on cf_train+prose (0% MQuAKE)...", flush=True)
    n_probes = wrap_v4.fomp.n_probes
    gen_seqs = list(sh_cue_ids[: min(n_tr, 12 if smoke else 224)]) + [
        c.tolist()[:48] for c in prose_tr_chunks[: min(len(prose_tr_chunks), 4 if smoke else 32)]
    ]
    hp_toks: List[List[torch.Tensor]] = [[] for _ in range(n_probes)]
    hf_toks: List[torch.Tensor] = []
    for s0 in range(0, len(gen_seqs), bs_enc):
        sub_s = gen_seqs[s0 : s0 + bs_enc]
        mx = max(len(x) for x in sub_s)
        mat = torch.full((len(sub_s), mx), pad_id, device=device, dtype=torch.long)
        msk = torch.zeros((len(sub_s), mx), device=device, dtype=torch.long)
        for j, x in enumerate(sub_s):
            mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
            msk[j, : len(x)] = 1
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            caps_p, h_fin = wrap_v4.capture_probes_and_final(mat, cfg.latent_probe_layers, attention_mask=msk)
        for j, x in enumerate(sub_s):
            ln = len(x)
            st_pos = 1 if (bos_id is not None and ln > 1) else 0
            avail = list(range(st_pos, ln))
            if len(avail) > 8:
                picked = sorted(set(rng.choice(avail[:-1], size=7, replace=False).tolist() + [avail[-1]]))
            else:
                picked = avail
            pt = torch.tensor(picked, device=device, dtype=torch.long)
            for pi, li in enumerate(cfg.latent_probe_layers):
                hp_toks[pi].append(caps_p[li][j, pt].cpu())
            hf_toks.append(h_fin[j, pt].cpu())

    lens_hp_all = [torch.cat(hp_toks[pi], dim=0) for pi in range(n_probes)]  # each [N_tok, d]
    lens_hf_all = torch.cat(hf_toks, dim=0)                                  # [N_tok, d]
    n_lens_tok = lens_hf_all.shape[0]

    def _first_ent_tok_variants(ent_str: str) -> Tuple[List[int], str]:
        w0 = ent_str.strip().split()[0] if ent_str.strip() else ""
        w_clean = w0.strip(" .,?!:;\"'()[]\n\t").lower()
        tids: List[int] = []
        for form in (ent_str, " " + w0, w0):
            if not form.strip():
                continue
            enc = tokenizer.encode(form, add_special_tokens=False)
            if enc and int(enc[0]) not in tids:
                tids.append(int(enc[0]))
        return tids, w_clean

    # Identify rows where frozen backbone natively knows `target_true` over `target_new` (strictly `cf_train`!)
    cand_indices = list(range(0, min(n_tr, 16 if smoke else 512))) + [int(x) for x in val_idx.tolist()]
    cand_indices = sorted(set(cand_indices))
    known_flags: Dict[int, bool] = {}
    ent_var_map: Dict[int, Tuple[List[int], str]] = {}
    for s0 in range(0, len(cand_indices), bs_enc):
        sub_idx = cand_indices[s0 : s0 + bs_enc]
        sub_c = [sh_cue_ids[i] for i in sub_idx]
        mx = max(len(x) for x in sub_c)
        mat = torch.full((len(sub_c), mx), pad_id, device=device, dtype=torch.long)
        msk = torch.zeros((len(sub_c), mx), device=device, dtype=torch.long)
        ends = []
        for j, x in enumerate(sub_c):
            mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
            msk[j, : len(x)] = 1
            ends.append(len(x) - 1)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out_k = wrap_v4.backbone(input_ids=mat, attention_mask=msk, use_cache=False) if hasattr(wrap_v4.backbone, "model") else wrap_v4.backbone(mat)
            lg_k = (out_k.logits if hasattr(out_k, "logits") else out_k)[torch.arange(len(sub_c), device=device), torch.tensor(ends, device=device)].float()
            lp_k = F.log_softmax(lg_k, dim=-1)
        for j, idx_i in enumerate(sub_idx):
            t_tr_ids, w_tr = _first_ent_tok_variants(sub_cf[idx_i].target_true)
            t_nw_ids, _ = _first_ent_tok_variants(sub_cf[idx_i].target_new)
            ent_var_map[idx_i] = (t_tr_ids, w_tr)
            if not t_tr_ids or not t_nw_ids:
                known_flags[idx_i] = False
                continue
            # Ensure ground-truth latent entity token does NOT literally appear in `pos_q_ids[idx_i]` or `comp_q_ids[idx_i]`
            q_used = set(pos_q_ids[idx_i]) | set(comp_q_ids[idx_i])
            if any(tid in q_used for tid in t_tr_ids):
                known_flags[idx_i] = False
                continue
            known_flags[idx_i] = bool(float(lp_k[j, t_tr_ids[0]].item()) > float(lp_k[j, t_nw_ids[0]].item()))

    tr_known = [i for i in cand_indices if i < n_tr and known_flags.get(i, False)]
    val_known = [int(i) for i in val_idx.tolist() if known_flags.get(int(i), False)]
    if smoke and len(tr_known) < 4:
        tr_known = [i for i in range(min(8, n_tr)) if ent_var_map.get(i, ([], ""))[0]]
    if smoke and len(val_known) < 4:
        val_known = [int(i) for i in val_idx.tolist()[:8] if ent_var_map.get(int(i), ([], ""))[0]]

    # Cache 2 inner-NP boundary states per `tr_known` row on `pos_q_ids` for mild task-aligned Tuned-Lens boost
    lat_hp: Optional[List[torch.Tensor]] = None
    lat_tids: Optional[torch.Tensor] = None
    if len(tr_known) >= 4:
        lh_parts: List[List[torch.Tensor]] = [[] for _ in range(n_probes)]
        lt_list: List[int] = []
        for s0 in range(0, len(tr_known), bs_enc):
            sub_i = tr_known[s0 : s0 + bs_enc]
            sub_s = [pos_q_ids[i] for i in sub_i]
            mx = max(len(x) for x in sub_s)
            mat = torch.full((len(sub_s), mx), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(sub_s), mx), device=device, dtype=torch.long)
            for j, x in enumerate(sub_s):
                mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
                msk[j, : len(x)] = 1
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                caps_l = wrap_v4.encode_layers_batch(mat, cfg.latent_probe_layers, attention_mask=msk)
            for j, idx_i in enumerate(sub_i):
                ln = len(pos_q_ids[idx_i])
                p0_j = min(ln - 1, max(0, pos_sp_end[idx_i]))
                p1_j = min(ln - 1, p0_j + 1)
                p_idx = torch.tensor([p0_j, p1_j], device=device, dtype=torch.long)
                for pi, li in enumerate(cfg.latent_probe_layers):
                    lh_parts[pi].append(caps_l[li][j, p_idx].unsqueeze(0).cpu())
                lt_list.append(ent_var_map[idx_i][0][0])
        lat_hp = [torch.cat(lh_parts[pi], dim=0) for pi in range(n_probes)]  # each [N_tr_k, 2, d]
        lat_tids = torch.tensor(lt_list, dtype=torch.long)

    opt_lens = torch.optim.AdamW(lens_params, lr=1.0e-3, weight_decay=1e-2)
    lens_steps = 6 if smoke else 160
    bs_l = min(16 if smoke else 64, n_lens_tok)
    last_lens_kl = 0.0
    last_lens_ent = 0.0
    for lst in range(1, lens_steps + 1):
        wrap_v4.fomp.train()
        opt_lens.zero_grad(set_to_none=True)
        pick_l = rng.choice(n_lens_tok, size=bs_l, replace=False)
        hf_b = lens_hf_all[pick_l].to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            p_ref = F.softmax(_project_logit_lens(wrap_v4.backbone, hf_b).float(), dim=-1)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            kl_acc = torch.tensor(0.0, device=device)
            for pi in range(n_probes):
                hp_b = lens_hp_all[pi][pick_l].to(device)
                h_hat = wrap_v4.fomp.apply_tuned_lens(hp_b, pi)
                lp_p = F.log_softmax(_project_logit_lens(wrap_v4.backbone, h_hat).float(), dim=-1)
                kl_acc = kl_acc + F.kl_div(lp_p, p_ref, reduction="batchmean")
            loss_kl_l = kl_acc / float(n_probes)

            loss_lat_e = torch.tensor(0.0, device=device)
            if lat_hp is not None and lat_tids is not None:
                bs_e = min(8 if smoke else 32, len(tr_known))
                pe_idx = rng.choice(len(tr_known), size=bs_e, replace=False)
                t_b = lat_tids[pe_idx].to(device)  # [bs_e]
                lp_cands = []
                for pi in range(n_probes):
                    h_e = lat_hp[pi][pe_idx].to(device)  # [bs_e, 2, d]
                    h_hat_e = wrap_v4.fomp.apply_tuned_lens(h_e, pi)
                    lp_e = F.log_softmax(_project_logit_lens(wrap_v4.backbone, h_hat_e).float(), dim=-1)
                    lp_t = lp_e.gather(-1, t_b.view(bs_e, 1, 1).expand(bs_e, 2, 1)).squeeze(-1)  # [bs_e, 2]
                    lp_cands.append(lp_t)
                lp_stack = torch.cat(lp_cands, dim=-1)  # [bs_e, n_probes*2]
                soft_best = torch.logsumexp(lp_stack * 2.0, dim=-1) / 2.0
                loss_lat_e = (-soft_best).mean()
            loss_l_tot = loss_kl_l + 0.15 * loss_lat_e
        loss_l_tot.backward()
        torch.nn.utils.clip_grad_norm_(lens_params, 1.0)
        opt_lens.step()
        last_lens_kl = float(loss_kl_l.item())
        last_lens_ent = float(loss_lat_e.item())

    # --- Phase 3: Held-Out `val_known` 64-Edit-Bank Calibration (Monotonic Raw vs Tuned Lens + `(tau_ll, tau_rel)`) ---
    wrap_v4.fomp.eval()
    hop2_calib_meta: Dict[str, Any] = {"val_known": len(val_known), "lens_kept": "raw_fallback"}
    if len(val_known) >= 4:
        val_seqs = [comp_q_ids[i] for i in val_known]
        val_probes_list: List[Dict[int, torch.Tensor]] = []
        val_hread_list: List[torch.Tensor] = []
        val_qsets = [set(s) for s in val_seqs]
        for s0 in range(0, len(val_seqs), bs_enc):
            sub_s = val_seqs[s0 : s0 + bs_enc]
            mx = max(len(x) for x in sub_s)
            mat = torch.full((len(sub_s), mx), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(sub_s), mx), device=device, dtype=torch.long)
            for j, x in enumerate(sub_s):
                mat[j, : len(x)] = torch.tensor(x, device=device, dtype=torch.long)
                msk[j, : len(x)] = 1
            need_ls = sorted(set([cfg.read_layer, *cfg.latent_probe_layers]))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                caps_v = wrap_v4.encode_layers_batch(mat, need_ls, attention_mask=msk)
            for j, x in enumerate(sub_s):
                ln = len(x)
                val_probes_list.append({li: caps_v[li][j : j + 1, :ln, :].cpu() for li in cfg.latent_probe_layers})
                val_hread_list.append(caps_v[cfg.read_layer][j : j + 1, :ln, :].cpu())

        v_cnt = len(val_known)
        k_val_bank = torch.stack([k_fam_tr[int(outer_j_arr[i])] for i in val_known], dim=0).to(device)  # [V, key_dim]
        rel_mat = np.zeros((v_cnt, v_cnt), dtype=np.float32)
        with torch.no_grad():
            for vi in range(v_cnt):
                hr = val_hread_list[vi].to(device)
                hctx = _causal_window_mean(hr, win_c)
                qt = wrap_v4.fomp.direct.project_key(hr[:, -1], hctx[:, -1])[0]
                qo = wrap_v4.fomp.project_hop2_outer_rel(hr)[0]
                rel_mat[vi] = torch.maximum(torch.mv(k_val_bank, qt), torch.mv(k_val_bank, qo)).cpu().numpy()

        max_c = 4
        v_tids = torch.zeros((v_cnt, max_c), device=device, dtype=torch.long)
        v_tmsk = torch.zeros((v_cnt, max_c), device=device, dtype=torch.float32)
        v_words = []
        for vi, idx_i in enumerate(val_known):
            tids_i, w_i = ent_var_map[idx_i]
            v_words.append(w_i)
            for si, tid in enumerate(tids_i[:max_c]):
                v_tids[vi, si] = tid
                v_tmsk[vi, si] = 1.0
        flat_vtids = v_tids.reshape(-1)
        same_ent = np.array([[v_words[i] == v_words[j] for j in range(v_cnt)] for i in range(v_cnt)], dtype=bool)
        bank_k = min(64, v_cnt)

        @torch.no_grad()
        def _score_lens_and_calibrate() -> Dict[str, Any]:
            ll_mat = np.full((v_cnt, v_cnt), -100.0, dtype=np.float32)
            for vi in range(v_cnt):
                best_lp = torch.full((v_cnt, max_c), -100.0, device=device, dtype=torch.float32)
                for pi, li in enumerate(cfg.latent_probe_layers):
                    hp = val_probes_list[vi][li].to(device)
                    st_t = 1 if (bos_id is not None and hp.shape[1] > 1) else 0
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        h_hat = wrap_v4.fomp.apply_tuned_lens(hp[:, st_t:, :], pi)
                        lg_v = _project_logit_lens(wrap_v4.backbone, h_hat).float()
                    lp_v = F.log_softmax(lg_v, dim=-1)[0]
                    gathered = lp_v[:, flat_vtids].reshape(-1, v_cnt, max_c).max(dim=0).values
                    best_lp = torch.maximum(best_lp, gathered)
                msk_q = v_tmsk.clone()
                qs = val_qsets[vi]
                for vj in range(v_cnt):
                    for si in range(max_c):
                        if msk_q[vj, si] > 0 and int(v_tids[vj, si].item()) in qs:
                            msk_q[vj, si] = 0.0
                ll_mat[vi] = best_lp.masked_fill(msk_q <= 0.0, -100.0).max(dim=-1).values.cpu().numpy()

            best_obj = -1e9
            best_res = {
                "tp_top1_pct": 0.0, "clean_fp_pct": 0.0,
                "ll_tau": float(cfg.hop2_ll_tau), "rel_tau": float(cfg.hop2_rel_tau),
                "pos_ll_med": round(float(np.median(np.diag(ll_mat))), 3),
                "pos_rel_med": round(float(np.median(np.diag(rel_mat))), 3),
            }
            for t_ll in np.linspace(-6.0, -1.2, 25):
                for t_rl in (0.15, 0.22, 0.28, 0.35, 0.42, 0.50):
                    tp_hits = 0
                    fp_hits = 0
                    for vi in range(v_cnt):
                        dist_js = [j for j in range(v_cnt) if not same_ent[vi, j]][: max(1, bank_k - 1)]
                        d_pass = [j for j in dist_js if ll_mat[vi, j] >= t_ll and rel_mat[vi, j] >= t_rl]
                        if d_pass:
                            fp_hits += 1
                        if ll_mat[vi, vi] >= t_ll and rel_mat[vi, vi] >= t_rl:
                            my_sc = ll_mat[vi, vi] + 4.0 * rel_mat[vi, vi]
                            if not d_pass or my_sc > max(ll_mat[vi, j] + 4.0 * rel_mat[vi, j] for j in d_pass):
                                tp_hits += 1
                    tp_r = tp_hits / float(v_cnt)
                    fp_r = fp_hits / float(v_cnt)
                    if fp_r <= 0.05 or smoke:
                        obj = tp_r - 2.0 * fp_r
                        if obj > best_obj:
                            best_obj = obj
                            best_res = {
                                "tp_top1_pct": round(100.0 * tp_r, 2),
                                "clean_fp_pct": round(100.0 * fp_r, 2),
                                "ll_tau": round(float(t_ll), 3),
                                "rel_tau": round(float(t_rl), 3),
                                "pos_ll_med": round(float(np.median(np.diag(ll_mat))), 3),
                                "pos_rel_med": round(float(np.median(np.diag(rel_mat))), 3),
                            }
            return best_res

        tuned_res = _score_lens_and_calibrate()
        saved_ups = [u.weight.detach().clone() for u in wrap_v4.fomp.lens_ups]
        for u in wrap_v4.fomp.lens_ups:
            u.weight.data.zero_()
        raw_res = _score_lens_and_calibrate()
        if (tuned_res["tp_top1_pct"] - 2.0 * tuned_res["clean_fp_pct"]) >= (raw_res["tp_top1_pct"] - 2.0 * raw_res["clean_fp_pct"]):
            for u, w_s in zip(wrap_v4.fomp.lens_ups, saved_ups):
                u.weight.data.copy_(w_s)
            win_res = tuned_res
            kept_kind = "tuned_lens"
        else:
            win_res = raw_res
            kept_kind = "raw_logit_lens"

        if not smoke:
            wrap_v4.fomp.hop2_ll_tau_buf.fill_(win_res["ll_tau"])
            wrap_v4.fomp.hop2_rel_tau_buf.fill_(win_res["rel_tau"])
        hop2_calib_meta = {
            "val_known": v_cnt,
            "bank_k": bank_k,
            "lens_kept": kept_kind,
            "tuned_lens": tuned_res,
            "raw_logit_lens": raw_res,
            "selected_ll_tau": round(float(wrap_v4.fomp.hop2_ll_tau_buf.item()), 3),
            "selected_rel_tau": round(float(wrap_v4.fomp.hop2_rel_tau_buf.item()), 3),
            "lens_kl_final": round(last_lens_kl, 4),
            "lens_ent_final": round(last_lens_ent, 4),
        }
        print(
            f"[FOMP-v4.2 Hop-2 Calib (val_known={v_cnt}, bank={bank_k})] kept={kept_kind} | "
            f"raw_tp1={raw_res['tp_top1_pct']}% (fp={raw_res['clean_fp_pct']}%) | "
            f"tuned_tp1={tuned_res['tp_top1_pct']}% (fp={tuned_res['clean_fp_pct']}%) | "
            f"tau=({hop2_calib_meta['selected_ll_tau']}, {hop2_calib_meta['selected_rel_tau']}) | "
            f"pos_med=(ll={win_res['pos_ll_med']}, rel={win_res['pos_rel_med']})",
            flush=True,
        )

    # Evaluate held-out 1-hop false-trigger rate & 2-hop recall of `mh_prob > mh_gate_tau`!
    wrap_v4.fomp.eval()
    with torch.no_grad():
        p_sh_val = torch.cat([
            torch.sigmoid(wrap_v4.fomp.compute_multihop_logits(h16_sh_cue[val_idx].to(device), m16_sh_cue[val_idx].to(device))),
            torch.sigmoid(wrap_v4.fomp.compute_multihop_logits(h16_sh_par[val_idx].to(device), m16_sh_par[val_idx].to(device))),
            torch.sigmoid(wrap_v4.fomp.compute_multihop_logits(h16_sh_qnp[val_idx].to(device), m16_sh_qnp[val_idx].to(device))),
            torch.sigmoid(wrap_v4.fomp.compute_multihop_logits(h16_sh_qpr[val_idx].to(device), m16_sh_qpr[val_idx].to(device))),
        ])
        p_mh_val = torch.cat([
            torch.sigmoid(wrap_v4.fomp.compute_multihop_logits(h16_mh_cmp[val_idx].to(device), m16_mh_cmp[val_idx].to(device))),
            torch.sigmoid(wrap_v4.fomp.compute_multihop_logits(h16_mh_tmp[val_idx].to(device), m16_mh_tmp[val_idx].to(device))),
        ])
        fp_1hop_pct = round(100.0 * float((p_sh_val > cfg.mh_gate_tau).float().mean().item()), 2)
        rec_2hop_pct = round(100.0 * float((p_mh_val > cfg.mh_gate_tau).float().mean().item()), 2)
    print(f"[FOMP-v4.2 Val Check] 1-Hop False-Trigger Rate={fp_1hop_pct}% | 2-Hop Recall={rec_2hop_pct}% (tau={cfg.mh_gate_tau})", flush=True)

    sync(device)
    sec = round(time.perf_counter() - t0, 2)
    meta = {
        "zero_mquake_train": True,
        "zero_zsre_train": True,
        "bridge_steps": steps,
        "lens_steps": lens_steps,
        "train_sec": sec,
        "total_params": sum(p.numel() for p in wrap_v4.fomp.parameters()),
        "bridge_only_params": sum(p.numel() for p in wrap_v4.fomp.bridge_parameters()),
        "final_kl_legible": round(last_kl, 4),
        "final_nce_span": round(last_nce_a, 4),
        "final_hop2_rel_nce": round(last_h2_rel, 4),
        "val_1hop_false_trigger_pct": fp_1hop_pct,
        "val_2hop_recall_pct": rec_2hop_pct,
        "hop2_calib": hop2_calib_meta,
        "learned_span_tau": [
            round(float(wrap_v4.fomp.span_rel_tau.clamp(cfg.min_span_rel_tau, 0.90).item()), 4),
            round(float(wrap_v4.fomp.span_ent_tau.clamp(cfg.min_span_ent_tau, 0.90).item()), 4),
        ],
        "learned_bridge_alpha": [round(float(x), 3) for x in torch.sigmoid(wrap_v4.fomp.bridge_alpha_logit).tolist()],
        "direct_core_meta": direct_meta,
    }
    tmp = v4_ckpt_path.with_suffix(".pt.tmp")
    torch.save({"fomp": wrap_v4.fomp.state_dict(), "config": wrap_v4.fomp.config.__dict__, "train_meta": meta}, tmp)
    os.replace(tmp, v4_ckpt_path)
    return meta


# ==============================================================================
# FOMP-v4.1 Evaluation Helpers (`precompile_kv = False` Lazy DynamicCache!)
# ==============================================================================

def _compile_v4_batch(
    wrap_v4: FOMPv4BackboneWrapper,
    tokenizer: Any,
    recs: Sequence[Any],
    pad_id: int,
    prepend_bos: bool,
    device: torch.device,
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
        g_ext=1.0, precompile_kv=False,
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
    route_counts: Dict[str, int] = {}
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
        rt_mode = str(wrap_v4._last_routing.get("mode", "unknown"))
        route_counts[rt_mode] = route_counts.get(rt_mode, 0) + 1
        lp_qo, _ = wrap_v4.score_target_logprob(q_ids, ao_ids, device, amp, compiled=compiled, mode=mode)
        m_prob.append(int(lp_qn > lp_qo))
        g_q, ex_q2 = wrap_v4.greedy_generate(q_ids, max(1, len(an_ids) + 1), device, amp=amp, compiled=compiled, mode=mode)
        extra_toks.append(max(ex_q1, ex_q2))
        m_em.append(int(tokenizer.decode(g_q, skip_special_tokens=True).strip().lower().startswith(r.multihop_answer_new.strip().lower())))
    return {
        "sh_prob": s_prob, "sh_em": s_em, "mh_prob": m_prob, "mh_em": m_em,
        "avg_extra_prompt_tokens": round(float(np.mean(extra_toks)) if extra_toks else 0.0, 2),
        "mq_route_counts": route_counts,
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
# Hop-1 vs Hop-2 Breakdown & Complete Stage 3.1 Report Generator
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
        ("fomp_v42_legible", "fomp_v36_zero_mq"),
        ("fomp_v42_legible", "alphaedit"),
        ("fomp_v42_legible", "ike_rag_gated_clean_demo"),
        ("fomp_v42_legible", "ike_rag_always_demo"),
        ("fomp_v42_legible", "oracle_ike_std"),
        ("fomp_v42_legible", "oracle_ike_chat"),
        ("fomp_v42_bridge_kv", "fomp_v42_legible"),
        ("fomp_v36_zero_mq", "ike_rag_gated_clean_demo"),
        ("fomp_v36_zero_mq", "oracle_ike_std"),
    ]
    metrics = [
        ("CF Efficacy EM", lambda a: a["track1_cf"]["eff_em"]["hits"]),
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
    pf = summary.get("ike_preflight", {})
    order = [
        "base",
        "oracle_ike_std",
        "oracle_ike_chat",
        "ike_rag_always_demo",
        "ike_rag_gated_clean_demo",
        "alphaedit",
        "fomp_v36_zero_mq",
        "fomp_v36",
        "fomp_v42_legible",
        "fomp_v42_bridge_kv",
    ]
    labels = {
        "base": "Base (Unedited)",
        "oracle_ike_std": "Oracle IKE-4Shot (Zheng'23 Std)",
        "oracle_ike_chat": "Oracle IKE-4Shot (Llama-3 Chat)",
        "ike_rag_always_demo": "IKE-RAG Always (4-Shot Demo, Top-2)",
        "ike_rag_gated_clean_demo": "IKE-RAG Gated Clean (4-Shot Demo + QA-τ)",
        "alphaedit": "AlphaEdit (Weight-Edit, ICLR'25)",
        "fomp_v36_zero_mq": "FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)",
        "fomp_v36": "FOMP-v3.6 (+706 MQ in-domain)",
        "fomp_v42_legible": "FOMP-v4.2 Pure-Activation 3-Route (0 tok)",
        "fomp_v42_bridge_kv": "FOMP-v4.2 Bridge-KV Ablation (Lazy KV)",
    }

    lines = [
        f"# Stage 3.2 Pure-Activation 3-Route FOMP-v4.2 & 4-Shot IKE Audit on `{summary.get('backbone')}` (`{summary.get('device')}`)",
        "",
        f"- **Zero-Leakage & Pure-Activation Guarantee**: `FOMP-v4.2` (`0` extra prompt/KV tokens) and all 4-shot IKE demos use strictly `cf_train` (**`0%` MQuAKE, `0%` ZsRE**).",
        f"- **IKE Pre-Flight Sanity Check (`cf_train` held-out)**: `{json.dumps(pf)}`",
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
        et = f"{t['edit_time_sec_per_batch']}s" if not k.startswith(("base", "oracle_")) else "—"
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
        et = f"{t['edit_time_sec_per_batch']}s" if not k.startswith(("base", "oracle_")) else "—"
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
    ap.add_argument(
        "--arms",
        type=str,
        default="oracle_ike_std,oracle_ike_chat,ike_rag_always_demo,ike_rag_gated_clean_demo,fomp_v42_legible",
    )
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
    out_json = args.out / "stage32_multihop_summary.json"
    out_md = args.out / "stage32_multihop_report.md"

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

        print(f"[Stage 3.2 Setup] Loading {args.model_id} in BF16 ...", flush=True)
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

    summary: Dict[str, Any] = {}
    if out_json.exists():
        summary = json.loads(out_json.read_text(encoding="utf-8"))
    elif args.stage25_dir is not None and (args.stage25_dir / "rigor_audit_summary.json").exists():
        summary = json.loads((args.stage25_dir / "rigor_audit_summary.json").read_text(encoding="utf-8"))
        print(f"[Stage 3.2] Imported {len(summary.get('arms', {}))} completed arms from Stage 2.5!", flush=True)

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

    fmt_helper = IKEPromptFormatter(tokenizer, cf_tr[:4], prepend_bos)
    base_chat_loc_batches: Optional[List[List[str]]] = None
    if fmt_helper.has_chat:
        base_chat_loc_batches = []
        for zb in z_batches:
            b_strs = []
            for r in zb:
                msgs = list(fmt_helper.chat_prefix_msgs) + [{"role": "user", "content": f"Question: {strip_qa_wrapper(r.loc_prompt)}?"}]
                res = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
                l_ids = [int(x) for x in (res["input_ids"] if isinstance(res, dict) or hasattr(res, "keys") else res)]
                g_b = _greedy_gen_ike(backbone, l_ids, loc_gen_len + 2, device, amp, stop_ids=fmt_helper.stop_ids)
                b_strs.append(tokenizer.decode(g_b, skip_special_tokens=True).strip().lower())
            base_chat_loc_batches.append(b_strs)

    if "base" not in summary["arms"] or "base" in force_arms:
        print("\n=== [Stage 3.2 Arm] BASE ===", flush=True)
        b_cf = [_eval_batch_cf_hits(backbone, tokenizer, cb, prose_ev_chunks, prose_ev_txts, device, amp, prepend_bos) for cb in cf_batches]
        b_mq = [_eval_batch_mquake_hits(backbone, tokenizer, mb, device, amp, prepend_bos) for mb in mq_batches]
        b_z = [_eval_batch_zsre_hits(backbone, tokenizer, zb, bl, device, amp, prepend_bos, loc_gen_len) for zb, bl in zip(z_batches, base_loc_batches)]
        summary["arms"]["base"] = {
            "track1_cf": _aggregate_cf(b_cf, [], base_ppl),
            "track2_mq": _aggregate_mq(b_mq, []),
            "track3_zsre": _aggregate_zsre(b_z, []),
        }

    # --- Priority 1 Pre-Flight: Verify 4-shot IKE (`std` & `chat`) reaches ~100% Oracle Efficacy on `cf_train` ---
    best_ike_fmt, pf_stats = _preflight_verify_ike_formats(
        backbone, fmt_helper, cf_tr, prose_ev_chunks, device, amp, args.smoke
    )
    summary["ike_preflight"] = {"selected_rag_fmt": best_ike_fmt, "formats": pf_stats}

    for oc_name, oc_fmt in (("oracle_ike_std", "std"), ("oracle_ike_chat", "chat")):
        if oc_name in req_arms and (oc_name not in summary["arms"] or oc_name in force_arms):
            eff_fmt = "std" if (oc_fmt == "chat" and not fmt_helper.has_chat) else oc_fmt
            print(f"\n=== [Stage 3.2 Arm] {oc_name.upper()} (4-Shot IKE Demo, fmt={eff_fmt}) ===", flush=True)
            oc_cf = [_eval_ike_arm_cf(backbone, fmt_helper, cb, prose_ev_chunks, eff_fmt, device, amp) for cb in cf_batches]
            oc_mq = [_eval_ike_arm_mq(backbone, fmt_helper, mb, eff_fmt, device, amp) for mb in mq_batches]
            oc_z = [
                _eval_ike_arm_zsre(
                    backbone, fmt_helper, zb, bl,
                    (base_chat_loc_batches[bi] if base_chat_loc_batches is not None else None),
                    eff_fmt, device, amp, loc_gen_len,
                )
                for bi, (zb, bl) in enumerate(zip(z_batches, base_loc_batches))
            ]
            summary["arms"][oc_name] = {
                "ike_format": eff_fmt,
                "track1_cf": _aggregate_cf(oc_cf, [], base_ppl),
                "track2_mq": _aggregate_mq(oc_mq, []),
                "track3_zsre": _aggregate_zsre(oc_z, []),
            }
            print(
                f"[{oc_name.upper()}] CF_EffEM={summary['arms'][oc_name]['track1_cf']['eff_em']['pct']}% "
                f"CF_ParEM={summary['arms'][oc_name]['track1_cf']['par_em']['pct']}% "
                f"MQ_1Hop={summary['arms'][oc_name]['track2_mq']['sh_em']['pct']}% "
                f"MQ_2Hop={summary['arms'][oc_name]['track2_mq']['mh_em']['pct']}% "
                f"ZsRE_Gen={summary['arms'][oc_name]['track3_zsre']['gen_em']['pct']}%",
                flush=True,
            )
            summary["mquake_hop_subtypes"] = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)
            summary["stage3_paired_mcnemar"] = _compute_stage3_mcnemar(summary["arms"])
            save_json(out_json, summary)
            _write_stage3_report(summary, out_md)

    need_rag = [a for a in ("ike_rag_always_demo", "ike_rag_gated_clean_demo") if a in req_arms and (a not in summary["arms"] or a in force_arms)]
    if need_rag:
        c_ike = CleanDenseIKERetriever(args.ike_model_id, device=device, smoke=args.smoke)
        c_tau, c_stats = c_ike.calibrate_clean_threshold(cf_tr, max_n=(16 if args.smoke else 300))
        summary["clean_ike_retriever"] = {"model_id": args.ike_model_id, "calibrated_tau": c_tau, "calib_stats": c_stats, "prompt_fmt": best_ike_fmt}

        for r_arm in need_rag:
            use_tau = None if r_arm == "ike_rag_always_demo" else c_tau
            print(f"\n=== [Stage 3.2 Arm] {r_arm.upper()} (fmt={best_ike_fmt}, tau={use_tau}) ===", flush=True)
            cf_bs, cf_ts, mq_bs, mq_ts, z_bs, z_ts = [], [], [], [], [], []
            for cb in cf_batches:
                bk, ts = c_ike.build_bank(cb); cf_ts.append(ts)
                cf_bs.append(_eval_ike_arm_cf(backbone, fmt_helper, cb, prose_ev_chunks, best_ike_fmt, device, amp, ike=c_ike, ike_bank=bk, ike_tau=use_tau))
            for mb in mq_batches:
                bk, ts = c_ike.build_bank(mb); mq_ts.append(ts)
                mq_bs.append(_eval_ike_arm_mq(backbone, fmt_helper, mb, best_ike_fmt, device, amp, ike=c_ike, ike_bank=bk, ike_tau=use_tau))
            for bi, (zb, bl) in enumerate(zip(z_batches, base_loc_batches)):
                bk, ts = c_ike.build_bank(zb); z_ts.append(ts)
                z_bs.append(
                    _eval_ike_arm_zsre(
                        backbone, fmt_helper, zb, bl,
                        (base_chat_loc_batches[bi] if base_chat_loc_batches is not None else None),
                        best_ike_fmt, device, amp, loc_gen_len, ike=c_ike, ike_bank=bk, ike_tau=use_tau,
                    )
                )
            summary["arms"][r_arm] = {
                "tau": use_tau,
                "ike_format": best_ike_fmt,
                "track1_cf": _aggregate_cf(cf_bs, cf_ts, base_ppl),
                "track2_mq": _aggregate_mq(mq_bs, mq_ts),
                "track3_zsre": _aggregate_zsre(z_bs, z_ts),
            }
            print(
                f"[{r_arm.upper()}] CF_EffEM={summary['arms'][r_arm]['track1_cf']['eff_em']['pct']}% "
                f"CF_ParEM={summary['arms'][r_arm]['track1_cf']['par_em']['pct']}% "
                f"MQ_2Hop={summary['arms'][r_arm]['track2_mq']['mh_em']['pct']}% "
                f"ZsRE_Gen={summary['arms'][r_arm]['track3_zsre']['gen_em']['pct']}% "
                f"ZsRE_Loc={summary['arms'][r_arm]['track3_zsre']['loc_exact']['pct']}%",
                flush=True,
            )
            summary["mquake_hop_subtypes"] = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)
            summary["stage3_paired_mcnemar"] = _compute_stage3_mcnemar(summary["arms"])
            save_json(out_json, summary)
            _write_stage3_report(summary, out_md)
        del c_ike
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    # --- Flagship Pure-Activation 3-Route `fomp_v42_legible` (plus optional `fomp_v42_bridge_kv` ablation) ---
    need_v4 = [a for a in ("fomp_v42_legible", "fomp_v42_bridge_kv") if a in req_arms and (a not in summary["arms"] or a in force_arms)]
    if need_v4:
        d_m = int(backbone.config.hidden_size)
        n_l = int(backbone.config.num_hidden_layers)
        v4_cfg = FOMPv4Config.for_backbone(d_m, n_l)
        torch.manual_seed(args.seed)
        fomp_v4 = FOMPv4Module(v4_cfg).to(device)
        wrap_v4 = FOMPv4BackboneWrapper(backbone, fomp_v4, freeze_backbone=True)

        v4_ckpt = args.out / "fomp_v42_zero_mq_llama3_8b.pt"
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
            mode = "legible" if arm_name == "fomp_v42_legible" else "bridge_kv"
            print(f"\n=== [Stage 3.2 Arm] {arm_name.upper()} (mode={mode}, 0% MQuAKE & 0% ZsRE!) ===", flush=True)

            cf_bs, cf_ts, mq_bs, mq_ts, z_bs, z_ts = [], [], [], [], [], []
            mq_route_tot: Dict[str, int] = {}
            for cb in cf_batches:
                cm, ts = _compile_v4_batch(wrap_v4, tokenizer, cb, pad_id, prepend_bos, device)
                cf_ts.append(ts)
                cf_bs.append(_eval_v4_cf(wrap_v4, tokenizer, cb, prose_ev_chunks, cm, mode, device, amp, prepend_bos))
            for mb in mq_batches:
                cm, ts = _compile_v4_batch(wrap_v4, tokenizer, mb, pad_id, prepend_bos, device)
                mq_ts.append(ts)
                mq_res = _eval_v4_mq(wrap_v4, tokenizer, mb, cm, mode, device, amp, prepend_bos)
                for rk, rv in mq_res.get("mq_route_counts", {}).items():
                    mq_route_tot[rk] = mq_route_tot.get(rk, 0) + rv
                mq_bs.append(mq_res)
            for zb, bl in zip(z_batches, base_loc_batches):
                cm, ts = _compile_v4_batch(wrap_v4, tokenizer, zb, pad_id, prepend_bos, device)
                z_ts.append(ts)
                z_bs.append(_eval_v4_zsre(wrap_v4, tokenizer, zb, bl, cm, mode, device, amp, prepend_bos, loc_gen_len))

            summary["arms"][arm_name] = {
                "train_meta": v4_meta,
                "mode": mode,
                "zero_mquake_train": True,
                "mq_route_counts": mq_route_tot,
                "track1_cf": _aggregate_cf(cf_bs, cf_ts, base_ppl),
                "track2_mq": _aggregate_mq(mq_bs, mq_ts),
                "track3_zsre": _aggregate_zsre(z_bs, z_ts),
            }
            sub_now = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)["per_arm"].get(arm_name, {})
            h1_p = sub_now.get("hop1_edited_em", {}).get("pct", 0.0)
            h2_p = sub_now.get("hop2_edited_em", {}).get("pct", 0.0)
            print(
                f"[{arm_name.upper()}] CF_EffEM={summary['arms'][arm_name]['track1_cf']['eff_em']['pct']}% "
                f"CF_ParEM={summary['arms'][arm_name]['track1_cf']['par_em']['pct']}% "
                f"MQ_1Hop={summary['arms'][arm_name]['track2_mq']['sh_em']['pct']}% "
                f"MQ_2Hop={summary['arms'][arm_name]['track2_mq']['mh_em']['pct']}% (Hop1={h1_p}%, Hop2={h2_p}%) "
                f"MQ_Routes={mq_route_tot} "
                f"ZsRE_Gen={summary['arms'][arm_name]['track3_zsre']['gen_em']['pct']}% "
                f"ZsRE_Loc={summary['arms'][arm_name]['track3_zsre']['loc_exact']['pct']}%",
                flush=True,
            )
            summary["mquake_hop_subtypes"] = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)
            summary["stage3_paired_mcnemar"] = _compute_stage3_mcnemar(summary["arms"])
            save_json(out_json, summary)
            _write_stage3_report(summary, out_md)

    summary["mquake_hop_subtypes"] = _compute_hop_subtype_breakdown(summary["arms"], mq_flat)
    summary["stage3_paired_mcnemar"] = _compute_stage3_mcnemar(summary["arms"])
    save_json(out_json, summary)
    _write_stage3_report(summary, out_md)
    print(f"\n[DONE] Stage 3.2 Summary -> {out_json}\n[DONE] Stage 3.2 Report  -> {out_md}", flush=True)


if __name__ == "__main__":
    main()
