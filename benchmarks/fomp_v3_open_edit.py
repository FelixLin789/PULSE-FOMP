"""FOMP-v3 Open-Domain 4-Arm Head-to-Head Benchmark (CounterFact + MQuAKE-Remastered on Qwen2.5-1.5B).

Zero-strawman head-to-head comparison on the EXACT same frozen Qwen2.5-1.5B backbone & strictly entity-disjoint
held-out test splits from public ungated HuggingFace benchmarks (`azhx/counterfact` & `henryzhongsc/MQuAKE-Remastered`):
  - Arm 1: `Base` (unedited Qwen2.5-1.5B)
  - Arm 2: `ICL` (in-context prompt injection of M=1, 16, 64, 128 edited facts -> measures accuracy + token/latency tax)
  - Arm 3: `LoRA (r=8) + Replay` (gradient editing on layers 10..27 + WikiText-2 replay -> measures accuracy + forgetting)
  - Arm 4: `FOMP-v3` (Ours: 1-pass 1st-order mid-layer fast weights + Top-2 sparse routing + multi-token causal latching,
           evaluated at M=1, 16, 64, 128 + mandatory ICML 2026 Sign-Flip + g_ext=0 & loaded WikiText-2 PPL drift!)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import _cosine_lr_multiplier
from pulse.fomp_v3 import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module, _causal_window_mean, _extract_tail_window

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import save_json, sync  # noqa: E402


@dataclass
class EditRecord:
    source: str                     # "counterfact" or "mquake"
    subject: str
    cue: str                        # e.g. "The capital of France is"
    target_new: str                 # leading-space target phrase, e.g. " Lyon"
    target_true: str                # original phrase, e.g. " Paris"
    paraphrases: List[str]          # paraphrase cues (CounterFact) or 2-hop questions (MQuAKE)
    neighborhoods: List[str]        # same-relation unedited entity cues (locality test!)
    multihop_answer_new: str = ""   # for MQuAKE when 2-hop answer differs from edit target_new
    multihop_answer_true: str = ""


# ==============================================================================
# Dataset Loaders (Ungated CounterFact + MQuAKE-Remastered + WikiText-2)
# ==============================================================================

_FALLBACK_PROSE = [
    "Modern computing systems rely on hierarchical memory architectures that balance fast on-chip caches with larger external DRAM banks. "
    "Efficient algorithms minimize global memory round-trips by keeping active working states inside local registers and shared memory.",
    "The history of astronomy spans ancient celestial navigation, optical telescopes, radio interferometry, and spaceborne infrared observatories. "
    "Each generation of instruments expanded humanity's view from planetary orbits to the large-scale filamentary structure of the cosmos.",
    "Ocean currents transport thermal energy from equatorial regions toward the poles, regulating regional climates across continents. "
    "Changes in salinity and surface winds alter deep-water overturning circulation on decadal to millennial timescales.",
    "Compiler optimization transforms high-level source code into machine instructions by eliminating dead stores, fusing memory-bound loops, "
    "and scheduling vector operations to saturate hardware execution units without changing observable program semantics.",
]


def _parse_counterfact_row(row: Dict[str, Any]) -> Optional[EditRecord]:
    rr = row.get("requested_rewrite")
    if isinstance(rr, list) and rr:
        rr = rr[0]
    if not isinstance(rr, dict):
        return None
    subj = str(rr.get("subject") or "").strip()
    tmpl = str(rr.get("prompt") or "").strip()
    tn = rr.get("target_new")
    tt = rr.get("target_true")
    tn_s = str(tn["str"] if isinstance(tn, dict) else (tn or "")).strip()
    tt_s = str(tt["str"] if isinstance(tt, dict) else (tt or "")).strip()
    if not (subj and tmpl and tn_s and tt_s) or tn_s.lower() == tt_s.lower():
        return None
    cue = tmpl.format(subj) if "{}" in tmpl else f"{tmpl} {subj}"
    paras = [str(p).strip() for p in (row.get("paraphrase_prompts") or []) if isinstance(p, str) and len(p.strip()) > 5]
    neighs = [str(p).strip() for p in (row.get("neighborhood_prompts") or []) if isinstance(p, str) and len(p.strip()) > 5]
    if not paras or not neighs:
        return None
    return EditRecord(
        source="counterfact", subject=subj, cue=cue,
        target_new=" " + tn_s, target_true=" " + tt_s,
        paraphrases=paras[:3], neighborhoods=neighs[:3],
    )


def _parse_mquake_row(row: Dict[str, Any]) -> Optional[EditRecord]:
    rrs = row.get("requested_rewrite")
    if isinstance(rrs, dict):
        rrs = [rrs]
    if not isinstance(rrs, list) or len(rrs) != 1:
        return None
    rr = rrs[0]
    subj = str(rr.get("subject") or "").strip()
    tmpl = str(rr.get("prompt") or "").strip()
    tn = rr.get("target_new_str") if rr.get("target_new_str") is not None else rr.get("target_new")
    tt = rr.get("target_true_str") if rr.get("target_true_str") is not None else rr.get("target_true")
    tn_s = str(tn["str"] if isinstance(tn, dict) else (tn or "")).strip()
    tt_s = str(tt["str"] if isinstance(tt, dict) else (tt or "")).strip()
    ans_new = str(row.get("new_answer") or "").strip()
    ans_orig = str(row.get("answer") or "").strip()
    qs = [str(q).strip() for q in (row.get("questions") or []) if isinstance(q, str) and len(q.strip()) > 5]
    if not (subj and tmpl and tn_s and tt_s and ans_new and ans_orig and qs):
        return None
    if ans_new.lower() == ans_orig.lower():
        return None
    cue = tmpl.format(subj) if "{}" in tmpl else f"{tmpl} {subj}"
    q_prompts = [f"Question: {q}\nAnswer:" for q in qs[:3]]
    return EditRecord(
        source="mquake", subject=subj, cue=cue,
        target_new=" " + tn_s, target_true=" " + tt_s,
        paraphrases=q_prompts, neighborhoods=[],
        multihop_answer_new=" " + ans_new, multihop_answer_true=" " + ans_orig,
    )


def _load_real_datasets(
    tokenizer: Any,
    max_tgt_tokens: int,
    n_cf_train: int = 3200,
    n_cf_test: int = 128,
    n_mq_train: int = 900,
    n_mq_test: int = 64,
    seed: int = 41,
) -> Tuple[List[EditRecord], List[EditRecord], List[EditRecord], List[EditRecord], List[str]]:
    from datasets import load_dataset

    def _valid_tgt(s: str) -> bool:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return 1 <= len(ids) <= max_tgt_tokens

    print("[Data] Loading azhx/counterfact ...", flush=True)
    cf_ds = load_dataset("azhx/counterfact", split="train")
    cf_all: List[EditRecord] = []
    # Keep fixed 5184 parse cap so `cf_test` & `mq_test` remain 100% bit-for-bit identical across v3.2/v3.3/v3.4!
    cf_parse_cap = 5184 if n_cf_test == 128 else max(5184, (n_cf_train + n_cf_test) * 3)
    for row in cf_ds:
        rec = _parse_counterfact_row(row)
        if rec is not None and _valid_tgt(rec.target_new) and _valid_tgt(rec.target_true):
            cf_all.append(rec)
        if len(cf_all) >= cf_parse_cap:
            break

    print(f"[Data] Parsed {len(cf_all)} valid CounterFact records. Loading henryzhongsc/MQuAKE-Remastered ...", flush=True)
    mq_all: List[EditRecord] = []
    try:
        mq_dict = load_dataset("henryzhongsc/MQuAKE-Remastered")
        split_name = "CF3k" if "CF3k" in mq_dict else list(mq_dict.keys())[0]
        mq_split = mq_dict[split_name]
        for row in mq_split:
            rec = _parse_mquake_row(row)
            if (
                rec is not None
                and _valid_tgt(rec.target_new)
                and _valid_tgt(rec.multihop_answer_new)
                and _valid_tgt(rec.multihop_answer_true)
            ):
                mq_all.append(rec)
        if not mq_all and len(mq_split) > 0:
            print(f"[Data] Debug MQuAKE row[0] keys={list(mq_split[0].keys())} rr={mq_split[0].get('requested_rewrite')}", flush=True)
    except Exception as exc:
        print(f"[Data] Warning: MQuAKE load failed ({exc}); continuing with CounterFact only.", flush=True)
    print(f"[Data] Parsed {len(mq_all)} valid 1-edit MQuAKE multi-hop records.", flush=True)

    # Strictly subject-disjoint train / test split!
    rng = np.random.default_rng(seed)
    cf_idx = rng.permutation(len(cf_all)).tolist()
    cf_test: List[EditRecord] = []
    test_subjs: set = set()
    for idx in cf_idx:
        r = cf_all[idx]
        sl = r.subject.lower()
        if sl not in test_subjs:
            cf_test.append(r)
            test_subjs.add(sl)
        if len(cf_test) >= n_cf_test:
            break

    mq_test: List[EditRecord] = []
    if mq_all:
        for idx in rng.permutation(len(mq_all)).tolist():
            r = mq_all[idx]
            sl = r.subject.lower()
            if sl not in test_subjs:
                mq_test.append(r)
                test_subjs.add(sl)
            if len(mq_test) >= n_mq_test:
                break

    cf_train = [cf_all[i] for i in cf_idx if cf_all[i].subject.lower() not in test_subjs][:n_cf_train]
    mq_train = [r for r in mq_all if r.subject.lower() not in test_subjs][:n_mq_train]

    # Assert strict zero subject overlap!
    tr_subjs = {r.subject.lower() for r in cf_train + mq_train}
    assert tr_subjs.isdisjoint(test_subjs), "Entity leakage between train and test splits!"

    # General English text for Locality PPL & LoRA Replay (WikiText-2 ungated!)
    prose: List[str] = []
    try:
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        buf: List[str] = []
        for ex in wt:
            t = str(ex.get("text") or "").strip()
            if len(t) > 80 and not t.startswith("="):
                buf.append(t)
                if len(" ".join(buf)) > 1200:
                    prose.append(" ".join(buf))
                    buf = []
                if len(prose) >= 40:
                    break
    except Exception as exc:
        print(f"[Data] WikiText-2 fallback ({exc}); using built-in prose.", flush=True)
    if len(prose) < 8:
        prose = _FALLBACK_PROSE * 4

    print(
        f"[Data] Split summary (0% subject overlap): CF train={len(cf_train)}, CF test={len(cf_test)}, "
        f"MQuAKE train={len(mq_train)}, MQuAKE test={len(mq_test)}, prose_docs={len(prose)}",
        flush=True,
    )
    print(f"  Sample CF test[0]: cue={cf_test[0].cue!r} -> new={cf_test[0].target_new!r} (orig={cf_test[0].target_true!r})", flush=True)
    if mq_test:
        print(f"  Sample MQ test[0]: cue={mq_test[0].cue!r} -> new={mq_test[0].target_new!r} | Q={mq_test[0].paraphrases[0]!r} -> {mq_test[0].multihop_answer_new!r}", flush=True)
    return cf_train, cf_test, mq_train, mq_test, prose


# ==============================================================================
# Smoke-Mode Dummy HF Model & Tokenizer (Full 4-Arm CPU Coverage!)
# ==============================================================================

class _SmokeTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        toks = [4 + (abs(hash(w)) % 120) for w in text.strip().split()]
        return toks or [4]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"w{i}" for i in ids)


class _SmokeDecoderLayer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d, bias=False)
        self.self_attn.v_proj = nn.Linear(d, d, bias=False)
        self.self_attn.o_proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        a = self.self_attn.o_proj(torch.tanh(self.self_attn.q_proj(hidden_states) + self.self_attn.v_proj(hidden_states)))
        h = hidden_states + 0.1 * a
        h = h + 0.1 * self.mlp.down_proj(F.silu(h))
        return (h,)


class _SmokeHFModel(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 64, n_layers: int = 8) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": d, "num_hidden_layers": n_layers})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList([_SmokeDecoderLayer(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache: bool = False):
        h = self.model.embed_tokens(input_ids)
        for lyr in self.model.layers:
            h = lyr(h)[0]
        return type("Out", (), {"logits": self.lm_head(h)})()


def _make_smoke_datasets() -> Tuple[List[EditRecord], List[EditRecord], List[EditRecord], List[EditRecord], List[str]]:
    def _mk(prefix: str, n: int, src: str) -> List[EditRecord]:
        out = []
        for i in range(n):
            out.append(EditRecord(
                source=src, subject=f"{prefix}{i}",
                cue=f"The capital of {prefix}{i} is",
                target_new=f" new{i} city{i}", target_true=f" old{i}",
                paraphrases=[f"Which city is capital of {prefix}{i}?"],
                neighborhoods=[f"The capital of neigh{i} is"],
                multihop_answer_new=f" new{i} city{i}", multihop_answer_true=f" old{i}",
            ))
        return out
    return _mk("trcf", 24, "counterfact"), _mk("tecf", 16, "counterfact"), _mk("trmq", 12, "mquake"), _mk("temq", 8, "mquake"), _FALLBACK_PROSE * 2


# ==============================================================================
# Core Evaluation Helpers (Identical Metrics Across ALL 4 Arms!)
# ==============================================================================

@torch.no_grad()
def _seq_target_logprob(
    wrap: FOMPv3BackboneWrapper,
    prompt_ids: List[int],
    tgt_ids: List[int],
    device: torch.device,
    amp: bool,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    sign_flip: bool = False,
) -> float:
    full = torch.tensor([prompt_ids + tgt_ids], device=device, dtype=torch.long)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        lg = wrap.forward_logits(full, compiled=compiled, sign_flip=sign_flip)["logits"][0]
    p_len = len(prompt_ids)
    pred = lg[p_len - 1 : p_len + len(tgt_ids) - 1].float()
    tgt = torch.tensor(tgt_ids, device=device, dtype=torch.long)
    return float(-F.cross_entropy(pred, tgt, reduction="mean").item())


@torch.no_grad()
def _eval_prose_ppl(
    wrap: FOMPv3BackboneWrapper,
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
) -> float:
    nlls: List[float] = []
    for ch in prose_chunks:
        ch_d = ch.unsqueeze(0).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg = wrap.forward_logits(ch_d[:, :-1], compiled=compiled)["logits"]
            nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), ch_d[:, 1:].reshape(-1))
        nlls.append(float(nll.item()))
    return round(math.exp(min(20.0, float(np.mean(nlls)))), 4)


@torch.no_grad()
def _evaluate_edits(
    wrap: FOMPv3BackboneWrapper,
    tokenizer: Any,
    cf_subset: List[EditRecord],
    mq_subset: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    icl_prefix: str = "",
    sign_flip: bool = False,
    max_gen_eval: int = 32,
    eval_prose: bool = True,
) -> Dict[str, Any]:
    """Evaluate Efficacy, Paraphrase, MQuAKE 2-Hop, Neighborhood Locality, General PPL, and Query Latency."""
    wrap.backbone.eval()
    wrap.fomp.eval()
    pref_ids = tokenizer.encode(icl_prefix, add_special_tokens=False) if icl_prefix else []

    def _run_group(pairs: List[Tuple[str, str, str]], want_new: bool) -> Dict[str, float]:
        if not pairs:
            return {"prob_win_pct": 0.0, "gen_em_pct": 0.0, "n": 0}
        wins = gen_hits = gen_n = 0
        for idx, (prompt_s, t_new_s, t_orig_s) in enumerate(pairs):
            p_ids = pref_ids + tokenizer.encode(prompt_s, add_special_tokens=False)
            if len(p_ids) > 3800:
                p_ids = p_ids[-3800:]
            n_ids = tokenizer.encode(t_new_s, add_special_tokens=False)
            o_ids = tokenizer.encode(t_orig_s, add_special_tokens=False)
            lp_n = _seq_target_logprob(wrap, p_ids, n_ids, device, amp, compiled=compiled, sign_flip=sign_flip)
            lp_o = _seq_target_logprob(wrap, p_ids, o_ids, device, amp, compiled=compiled, sign_flip=sign_flip)
            ok_p = (lp_n > lp_o) if want_new else (lp_o > lp_n)
            wins += int(ok_p)
            if idx < max_gen_eval:
                gen_n += 1
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    g_ids = wrap.greedy_generate(
                        p_ids, max_new_tokens=max(4, len(n_ids) + 2), device=device,
                        compiled=compiled, sign_flip=sign_flip, eos_id=tokenizer.eos_token_id,
                    )
                g_txt = tokenizer.decode(g_ids, skip_special_tokens=True).strip().lower()
                expected = (t_new_s if want_new else t_orig_s).strip().lower()
                if g_txt.startswith(expected) or (len(expected) >= 3 and expected in g_txt[: len(expected) + 8]):
                    gen_hits += 1
        return {
            "prob_win_pct": round(wins / len(pairs) * 100.0, 2),
            "gen_em_pct": round(gen_hits / max(1, gen_n) * 100.0, 2),
            "n": len(pairs),
        }

    eff_pairs = [(r.cue, r.target_new, r.target_true) for r in cf_subset]
    par_pairs = [(r.paraphrases[0], r.target_new, r.target_true) for r in cf_subset if r.paraphrases]
    nei_pairs = [(r.neighborhoods[0], r.target_new, r.target_true) for r in cf_subset if r.neighborhoods]
    mq_pairs = [(r.paraphrases[0], r.multihop_answer_new, r.multihop_answer_true) for r in mq_subset if r.paraphrases]

    # Measure real query latency over up to 16 prompts
    ms_per_query = 0.0
    if eval_prose:
        lat_prompts = [pref_ids + tokenizer.encode(r.paraphrases[0], add_special_tokens=False) for r in cf_subset[: min(16, len(cf_subset))]]
        sync(device)
        t0 = time.perf_counter()
        for p_ids in lat_prompts:
            inp = torch.tensor([p_ids[-3800:]], device=device, dtype=torch.long)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                _ = wrap.forward_logits(inp, compiled=compiled, sign_flip=sign_flip)
        sync(device)
        ms_per_query = round((time.perf_counter() - t0) * 1000.0 / max(1, len(lat_prompts)), 2)

    return {
        "efficacy": _run_group(eff_pairs, want_new=True),
        "paraphrase": _run_group(par_pairs, want_new=True),
        "mquake_2hop": _run_group(mq_pairs, want_new=True),
        "neighborhood_retain": _run_group(nei_pairs, want_new=False),
        "general_prose_ppl": _eval_prose_ppl(wrap, prose_chunks, device, amp, compiled=compiled) if eval_prose else 0.0,
        "extra_prompt_tokens": len(pref_ids),
        "ms_per_query": ms_per_query,
    }


# ==============================================================================
# FOMP-v3 First-Order Training on Frozen Qwen2.5-1.5B (Pre-Cached Early States!)
# ==============================================================================

def _train_fomp_v3(
    wrap: FOMPv3BackboneWrapper,
    tokenizer: Any,
    cf_train: List[EditRecord],
    mq_train: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    smoke: bool,
    steps: int,
    ckpt_path: Path,
    deadline: float,
    prepend_bos: bool = False,
) -> Dict[str, Any]:
    if ckpt_path.exists() and not smoke:
        print(f"[FOMP-v3] Loading trained checkpoint -> {ckpt_path}", flush=True)
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        wrap.fomp.load_state_dict(st["fomp"])
        return st.get("train_meta", {"loaded_from_ckpt": True})

    # Format-Invariant Anti-Shortcut Augmentation (FOMP-v3.5):
    # Wrap a subset of `cf_train` (0% test overlap, 0% ZsRE!) into QA format paired with SAME-FORMAT QA
    # neighbor negatives, and pair MQuAKE QA positives with OTHER MQuAKE QA negatives so Dual-Lock
    # strictly ignores shared `?\nAnswer:` suffixes and routes purely by entity + semantic intent!
    def _wrap_qa(s: str) -> str:
        s = s.strip()
        if s and s[-1] not in "?.!":
            s = s + "?"
        return f"Question: {s}\nAnswer:"

    n_qa_aug = min(8 if smoke else 800, len(cf_train))
    cf_qa_aug: List[EditRecord] = []
    if n_qa_aug > 0:
        step_qa = max(1, len(cf_train) // n_qa_aug)
        for r in cf_train[::step_qa][:n_qa_aug]:
            cf_qa_aug.append(
                EditRecord(
                    source="cf_qa",
                    subject=r.subject,
                    cue=_wrap_qa(r.cue),
                    target_new=r.target_new,
                    target_true=r.target_true,
                    paraphrases=[_wrap_qa(p) for p in r.paraphrases],
                    neighborhoods=[_wrap_qa(n) for n in r.neighborhoods],
                )
            )

    all_tr = cf_train + cf_qa_aug + mq_train
    n_tr = len(all_tr)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    kv = wrap.fomp.config.max_tgt_tokens

    print(
        f"[FOMP-v3.5] Pre-caching frozen early states (L0..L{wrap.fomp.config.read_layer - 1}) for {n_tr} train records "
        f"(cf={len(cf_train)}, cf_qa={len(cf_qa_aug)}, mq={len(mq_train)})...",
        flush=True,
    )
    t_cache = time.perf_counter()
    bos_id = getattr(tokenizer, "bos_token_id", None) if prepend_bos else None

    def _enc_p(txt: str) -> List[int]:
        ids = tokenizer.encode(txt, add_special_tokens=False)
        return ([int(bos_id)] + ids) if (bos_id is not None and (not ids or ids[0] != int(bos_id))) else ids

    n_cf_tr = len(cf_train)
    n_qa_tr = len(cf_qa_aug)
    n_mq_tr = len(mq_train)
    c_ids_all = [_enc_p(r.cue) for r in all_tr]
    t_ids_all = [tokenizer.encode(r.target_new, add_special_tokens=False)[:kv] for r in all_tr]
    p0_ids_all = [_enc_p(r.paraphrases[0]) for r in all_tr]
    p1_ids_all = [
        _enc_p(r.paraphrases[1] if len(r.paraphrases) > 1 else r.paraphrases[0])
        for r in all_tr
    ]
    p_tgt_all = [
        tokenizer.encode(r.multihop_answer_new if (r.source == "mquake" and r.multihop_answer_new) else r.target_new, add_special_tokens=False)[:kv]
        for r in all_tr
    ]
    n_ids_all: List[List[int]] = []
    for i, r in enumerate(all_tr):
        if r.neighborhoods:
            n_ids_all.append(_enc_p(r.neighborhoods[0]))
        elif r.source == "mquake" and n_mq_tr > 1:
            # Same-format QA negative from another MQuAKE record (kills `?\nAnswer:` template shortcut!)
            mq_j = i - (n_cf_tr + n_qa_tr)
            other_mq = mq_train[(mq_j + 17) % n_mq_tr]
            n_ids_all.append(_enc_p(other_mq.paraphrases[0]))
        else:
            n_ids_all.append(_enc_p(all_tr[(i + 17) % n_tr].cue))

    def _batch_encode_last(seqs: List[List[int]], mbs: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out_last: List[torch.Tensor] = []
        out_ctx: List[torch.Tensor] = []
        out_win: List[torch.Tensor] = []
        out_wm: List[torch.Tensor] = []
        win_c = wrap.fomp.config.key_ctx_window
        win_e = wrap.fomp.config.ent_window
        for s0 in range(0, len(seqs), mbs):
            sub = seqs[s0 : s0 + mbs]
            mx = max(len(x) for x in sub)
            mat = torch.full((len(sub), mx), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(sub), mx), device=device, dtype=torch.long)
            lst = torch.zeros((len(sub),), device=device, dtype=torch.long)
            for i, s in enumerate(sub):
                mat[i, : len(s)] = torch.tensor(s, device=device, dtype=torch.long)
                msk[i, : len(s)] = 1
                lst[i] = len(s) - 1
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                h = wrap.encode_early_batch(mat, attention_mask=msk)
            h_ctx = _causal_window_mean(h, win_c)
            h_win, w_m = _extract_tail_window(h, lst, win_e)
            ar = torch.arange(len(sub), device=device)
            out_last.append(h[ar, lst].float().cpu())
            out_ctx.append(h_ctx[ar, lst].float().cpu())
            out_win.append(h_win.to(torch.float16).cpu())
            out_wm.append(w_m.cpu())
        return torch.cat(out_last, dim=0), torch.cat(out_ctx, dim=0), torch.cat(out_win, dim=0), torch.cat(out_wm, dim=0)

    def _batch_encode_tgts(seqs: List[List[int]], mbs: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_out, e_out, m_out = [], [], []
        for s0 in range(0, len(seqs), mbs):
            sub = seqs[s0 : s0 + mbs]
            mat = torch.full((len(sub), kv), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((len(sub), kv), device=device, dtype=torch.long)
            for i, s in enumerate(sub):
                ln = min(kv, max(1, len(s)))
                mat[i, :ln] = torch.tensor(s[:ln], device=device, dtype=torch.long)
                msk[i, :ln] = 1
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                h = wrap.encode_early_batch(mat, attention_mask=msk)
                e = wrap.lookup_target_embeddings(mat)
            h_out.append(h.float().cpu())
            e_out.append(e.float().cpu())
            m_out.append(msk.cpu())
        return torch.cat(h_out, dim=0), torch.cat(e_out, dim=0), torch.cat(m_out, dim=0)

    h_cue_tr, h_cue_ctx_tr, h_cue_win_tr, m_cue_win_tr = _batch_encode_last(c_ids_all)
    h_p0_tr, h_p0_ctx_tr, h_p0_win_tr, m_p0_win_tr = _batch_encode_last(p0_ids_all)
    h_p1_tr, h_p1_ctx_tr, h_p1_win_tr, m_p1_win_tr = _batch_encode_last(p1_ids_all)
    h_nei_tr, h_nei_ctx_tr, h_nei_win_tr, m_nei_win_tr = _batch_encode_last(n_ids_all)
    h_tgt_tr, e_tgt_tr, m_tgt_tr = _batch_encode_tgts(t_ids_all)

    # Open-Vocab Multi-Token Target-Swap Pool (FOMP-v3.6):
    # Real 1..3-word English spans sliced strictly from train WikiText-2 prose (0% test/ZsRE overlap!)
    # so `e_bank` + causal multi-token latching (`lag=0..kv-1`) generalize zero-shot to ALL 128K+ vocab tokens!
    rand_tgt_ids: List[List[int]] = []
    if prose_chunks:
        rng_span = np.random.default_rng(777)
        cap_rand = 16 if smoke else 512
        for ch in prose_chunks:
            words = [w for w in tokenizer.decode(ch.tolist(), skip_special_tokens=True).split() if (w.isalpha() or smoke) and len(w) >= 2]
            for wi in range(0, max(0, len(words) - 3), 3):
                n_w = int(rng_span.integers(1, 4))
                ids = tokenizer.encode(" " + " ".join(words[wi : wi + n_w]), add_special_tokens=False)[:kv]
                if 1 <= len(ids) <= kv:
                    rand_tgt_ids.append(ids)
                if len(rand_tgt_ids) >= cap_rand:
                    break
            if len(rand_tgt_ids) >= cap_rand:
                break
    h_rand_tr = e_rand_tr = m_rand_tr = None
    if rand_tgt_ids:
        h_rand_tr, e_rand_tr, m_rand_tr = _batch_encode_tgts(rand_tgt_ids)
    print(
        f"[FOMP-v3.6] Cached dual-lock multi-paraphrase states + salient windows + {len(rand_tgt_ids)} open-vocab spans "
        f"in {time.perf_counter() - t_cache:.1f}s.",
        flush=True,
    )

    lr = 8.0e-4 if wrap.fomp.config.d_model >= 3072 else 1.5e-3
    opt = torch.optim.AdamW(wrap.fomp.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    bs = 4 if smoke else 8
    w_steps = max(4, steps // 10)
    rng = np.random.default_rng(12345)
    ema_start = max(1, int(steps * 0.65))
    ema_decay = 0.985
    ema_state: Optional[Dict[str, torch.Tensor]] = None

    def _pad_seqs(prompts: List[List[int]], tgts: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        fulls = [p + t for p, t in zip(prompts, tgts)]
        mx = max(len(x) for x in fulls)
        inp = torch.full((len(fulls), mx), pad_id, device=device, dtype=torch.long)
        msk = torch.zeros((len(fulls), mx), device=device, dtype=torch.long)
        lbl = torch.full((len(fulls), mx), -100, device=device, dtype=torch.long)
        p_ends: List[int] = []
        for i, (p, t) in enumerate(zip(prompts, tgts)):
            seq = p + t
            inp[i, : len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
            msk[i, : len(seq)] = 1
            # Causal shift: position j predicts token j+1 -> target tokens live at j in [len(p)-1 .. len(p)+len(t)-2]
            for k_t, tok_id in enumerate(t):
                lbl[i, len(p) - 1 + k_t] = tok_id
            p_ends.append(len(p) - 1)
        return inp, msk, lbl, p_ends

    sync(device)
    t0 = time.perf_counter()
    last_l_ce = last_l_nc = 0.0
    for st in range(1, steps + 1):
        if time.time() > deadline - 180:
            print(f"[FOMP-v3] Stopping train early at step {st - 1} for deadline safety.", flush=True)
            break
        mult = _cosine_lr_multiplier(st - 1, steps, w_steps, 0.1)
        for pg in opt.param_groups:
            pg["lr"] = lr * mult

        # Sample bs target train edits: balanced Cloze CF + QA-wrapped CF + MQuAKE 2-hop positives!
        m_ep = (1, 4, 8)[st % 3] if smoke else (1, 8, 24, 64)[st % 4]
        m_ep = min(m_ep, n_tr)
        n_dir = 2
        if not smoke and n_cf_tr >= 4 and n_qa_tr >= 3 and n_mq_tr >= 2:
            # Indices [0:2]=direct (1 cloze + 1 QA), [2:6]=paraphrase (2 cloze + 2 QA), [6:8]=MQuAKE 2-hop!
            cf_p = rng.choice(n_cf_tr, size=3, replace=False).tolist()
            qa_p = (n_cf_tr + rng.choice(n_qa_tr, size=3, replace=False)).tolist()
            mq_p = (n_cf_tr + n_qa_tr + rng.choice(n_mq_tr, size=2, replace=False)).tolist()
            chosen = [cf_p[0], qa_p[0], cf_p[1], cf_p[2], qa_p[1], qa_p[2], mq_p[0], mq_p[1]]
        elif not smoke and n_cf_tr >= 5 and n_qa_tr >= 3:
            # 0% MQuAKE mode: keep identical 3/8 QA ratio (1 dir + 2 para QA, 1 dir + 4 para Cloze)
            cf_p = rng.choice(n_cf_tr, size=bs - 3, replace=False).tolist()
            qa_p = (n_cf_tr + rng.choice(n_qa_tr, size=3, replace=False)).tolist()
            chosen = [cf_p[0], qa_p[0], *cf_p[1:], qa_p[1], qa_p[2]]
        else:
            chosen = rng.choice(n_tr, size=bs, replace=False).tolist()
        bank_idx = np.zeros((bs, m_ep), dtype=np.int64)
        for bi, ti in enumerate(chosen):
            bank_idx[bi, 0] = ti
            if m_ep > 1:
                pool = rng.choice(n_tr, size=m_ep, replace=False)
                pool = [x for x in pool if x != ti][: m_ep - 1]
                bank_idx[bi, 1 : 1 + len(pool)] = pool

        hc_b = h_cue_tr[bank_idx].to(device)
        hc_ctx_b = h_cue_ctx_tr[bank_idx].to(device)
        hc_win_b = h_cue_win_tr[bank_idx].to(device)
        hc_wm_b = m_cue_win_tr[bank_idx].to(device)
        ht_b = h_tgt_tr[bank_idx].to(device)
        et_b = e_tgt_tr[bank_idx].to(device)
        mt_b = m_tgt_tr[bank_idx].to(device)

        # Positive prompts: 2 direct cues + (bs-2) multi-paraphrase / MQuAKE 2-hop queries!
        pos_p: List[List[int]] = []
        pos_t: List[List[int]] = []
        n_single_hop = min(6, bs) if (not smoke and n_mq_tr >= 2) else bs
        val_drop_mask = torch.ones((bs,), device=device, dtype=torch.float32)
        for bi, idx_i in enumerate(chosen):
            if bi < n_dir:
                pos_p.append(c_ids_all[idx_i])
                pos_t.append(t_ids_all[idx_i])
            else:
                use_p1 = bool(rng.integers(0, 2))
                pos_p.append(p1_ids_all[idx_i] if use_p1 else p0_ids_all[idx_i])
                pos_t.append(p_tgt_all[idx_i])
            # On single-hop items (Cloze + QA, NEVER MQuAKE 2-hop), apply:
            #   1) 40% Open-Vocab Multi-Token Target Swap -> eliminates CounterFact target-vocab memorization!
            #   2) 50% `val_path` Modality Dropout -> forces full-rank `e_bank` alone to nail 100% zero-shot greedy EM!
            if bi < n_single_hop:
                if h_rand_tr is not None and len(rand_tgt_ids) > 0 and rng.random() < 0.40:
                    r_idx = int(rng.integers(0, len(rand_tgt_ids)))
                    ht_b[bi, 0] = h_rand_tr[r_idx].to(device)
                    et_b[bi, 0] = e_rand_tr[r_idx].to(device)
                    mt_b[bi, 0] = m_rand_tr[r_idx].to(device)
                    pos_t[bi] = rand_tgt_ids[r_idx]
                if rng.random() < 0.50:
                    val_drop_mask[bi] = 0.0
        inp_pos, msk_pos, lbl_pos, pos_ends = _pad_seqs(pos_p, pos_t)

        # Negative prompts: diverse held-out-split train WikiText-2 prose snippets + same-relation neighborhoods!
        neg_p = [n_ids_all[i] for i in chosen]
        n_prose_neg = 1 if smoke else 2
        prose_sl_len = 24 if smoke else 48
        for pi in range(n_prose_neg):
            ch_p = prose_chunks[(st * 2 + pi) % len(prose_chunks)]
            max_off = max(0, len(ch_p) - prose_sl_len)
            off = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
            neg_p[pi] = ch_p[off : off + prose_sl_len].tolist()
        mx_n = max(len(x) for x in neg_p)
        inp_neg = torch.full((bs, mx_n), pad_id, device=device, dtype=torch.long)
        msk_neg = torch.zeros((bs, mx_n), device=device, dtype=torch.long)
        for i, s in enumerate(neg_p):
            inp_neg[i, : len(s)] = torch.tensor(s, device=device, dtype=torch.long)
            msk_neg[i, : len(s)] = 1

        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg_neg_base = wrap.forward_logits(inp_neg, attention_mask=msk_neg, compiled=None)["logits"]

        wrap.fomp.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            comp = wrap.fomp.compile_from_states(
                hc_b, ht_b, mt_b, g_ext=1.0, e_tgt=et_b,
                h_cue_ctx=hc_ctx_b, h_cue_win=hc_win_b, h_cue_wmask=hc_wm_b,
            )
            o_pos = wrap.forward_logits(inp_pos, attention_mask=msk_pos, compiled=comp, val_drop_mask=val_drop_mask)
            lg_pos = o_pos["logits"].float()
            # Label smoothing prevents overconfident +30 logit explosion that amplifies false-trigger leakage!
            loss_ce = F.cross_entropy(
                lg_pos.reshape(-1, lg_pos.shape[-1]), lbl_pos.reshape(-1),
                ignore_index=-100, label_smoothing=0.05,
            )

            o_neg = wrap.forward_logits(inp_neg, attention_mask=msk_neg, compiled=comp)

            # Exact full-vocab KL(p_base || p_edited) preservation on prose + same-relation neighborhoods!
            kl_ed: List[torch.Tensor] = []
            kl_bs: List[torch.Tensor] = []
            for bi in range(n_prose_neg):
                ln_p = len(neg_p[bi])
                idx_p = torch.linspace(0, ln_p - 1, steps=min(16, ln_p), device=device).long()
                kl_ed.append(o_neg["logits"][bi, idx_p].float())
                kl_bs.append(lg_neg_base[bi, idx_p].float())
            for bi in range(n_prose_neg, bs):
                ne = len(neg_p[bi]) - 1
                kl_ed.append(o_neg["logits"][bi, ne : ne + 1].float())
                kl_bs.append(lg_neg_base[bi, ne : ne + 1].float())
            loss_kl_neg = F.kl_div(
                F.log_softmax(torch.cat(kl_ed, dim=0), dim=-1),
                F.softmax(torch.cat(kl_bs, dim=0), dim=-1),
                reduction="batchmean",
            )

            # Gate supervision:
            #   - rel_gate separates query-end positions (pos_ends -> 1) from prose & mid-answer tokens (-> 0);
            #   - dedicated `ans_rl` penalty ensures multi-token generation (`lag>=1`) never double-triggers at `lag=0`!
            ar_b = torch.arange(bs, device=device)
            rl_pos_end = o_pos["routing"]["rel_logits"][ar_b, torch.tensor(pos_ends, device=device)]
            rl_prose_all = torch.cat([o_neg["routing"]["rel_logits"][pi, : len(neg_p[pi])] for pi in range(n_prose_neg)], dim=0)
            ans_rl: List[torch.Tensor] = []
            for bi, pe in enumerate(pos_ends):
                if pe + 1 < inp_pos.shape[1] and len(pos_t[bi]) > 1:
                    ans_rl.append(o_pos["routing"]["rel_logits"][bi, pe + 1 : pe + len(pos_t[bi])])
            loss_ans_gate = (
                F.binary_cross_entropy_with_logits(torch.cat(ans_rl, dim=0), torch.zeros_like(torch.cat(ans_rl, dim=0)))
                if ans_rl
                else torch.tensor(0.0, device=device)
            )
            loss_gate = (
                0.5 * (F.binary_cross_entropy_with_logits(rl_pos_end, torch.ones_like(rl_pos_end))
                       + F.binary_cross_entropy_with_logits(rl_prose_all, torch.zeros_like(rl_prose_all)))
                + 0.35 * loss_ans_gate
                + 5.0 * o_neg["routing"]["max_gate"].pow(2).mean()
            )

            # Lock 1 (Relation+Context MLP) + Lock 2 (Salient-Entity + Latent-Bridge MLP) batch-128 Hard-Negative InfoNCE!
            nc_n = min(128 if not smoke else 16, n_tr)
            nc_idx = rng.choice(n_tr, size=nc_n, replace=False)
            pick_p1 = torch.from_numpy(rng.integers(0, 2, size=nc_n) == 1)
            hp_last = torch.where(pick_p1.unsqueeze(-1), h_p1_tr[nc_idx], h_p0_tr[nc_idx]).to(device)
            hp_ctx = torch.where(pick_p1.unsqueeze(-1), h_p1_ctx_tr[nc_idx], h_p0_ctx_tr[nc_idx]).to(device)
            hp_win = torch.where(pick_p1.view(-1, 1, 1), h_p1_win_tr[nc_idx], h_p0_win_tr[nc_idx]).to(device)
            mp_win = torch.where(pick_p1.unsqueeze(-1), m_p1_win_tr[nc_idx], m_p0_win_tr[nc_idx]).to(device)

            hc_nc = h_cue_tr[nc_idx].to(device)
            hn_nc = h_nei_tr[nc_idx].to(device)
            zc = wrap.fomp.project_key(hc_nc, h_cue_ctx_tr[nc_idx].to(device))
            zp = wrap.fomp.project_key(hp_last, hp_ctx)
            zn = wrap.fomp.project_key(hn_nc, h_nei_ctx_tr[nc_idx].to(device))
            sim_cp = torch.matmul(zc, zp.transpose(-2, -1)) / 0.07
            sim_cpn = torch.cat([sim_cp, torch.matmul(zc, zn.transpose(-2, -1)) / 0.07], dim=-1)
            lbl_nc = torch.arange(nc_n, device=device)
            loss_nce_rel = 0.5 * (F.cross_entropy(sim_cpn, lbl_nc) + F.cross_entropy(sim_cp.transpose(-2, -1), lbl_nc))
            cn_rel = torch.matmul(zc, zn.transpose(-2, -1))
            loss_margin_rel = (
                F.relu(cn_rel.diagonal() - 0.25).pow(2).mean()
                + 2.0 * F.relu(cn_rel - 0.25).pow(2).mean()
            )

            # Lock 2 combines salient subject window + latent bridge-end (`h_last`);
            # penalize BOTH matched hard-neg pairs AND all 128x128 cross-neg pairs (kills InfoNCE constant-shift loophole!)
            ze_c = wrap.fomp.project_entity_key_from_window(h_cue_win_tr[nc_idx].to(device), m_cue_win_tr[nc_idx].to(device), h_last=hc_nc)
            ze_p = wrap.fomp.project_entity_key_from_window(hp_win, mp_win, h_last=hp_last)
            ze_n = wrap.fomp.project_entity_key_from_window(h_nei_win_tr[nc_idx].to(device), m_nei_win_tr[nc_idx].to(device), h_last=hn_nc)
            se_cp = torch.matmul(ze_c, ze_p.transpose(-2, -1)) / 0.07
            se_cpn = torch.cat([se_cp, torch.matmul(ze_c, ze_n.transpose(-2, -1)) / 0.07], dim=-1)
            loss_nce_ent = 0.5 * (F.cross_entropy(se_cpn, lbl_nc) + F.cross_entropy(se_cp.transpose(-2, -1), lbl_nc))
            cn_ent = torch.matmul(ze_c, ze_n.transpose(-2, -1))
            loss_margin_ent = (
                F.relu(cn_ent.diagonal() - 0.15).pow(2).mean()
                + 4.0 * F.relu(cn_ent - 0.15).pow(2).mean()
            )

            loss_nce = 0.5 * (loss_nce_rel + loss_nce_ent)
            loss_margin = loss_margin_rel + 2.0 * loss_margin_ent
            # Symmetric bounded-ceiling reg: keep val_path clean (`<=0.8`) while allowing zero-shot `e_bank` up to `2.5`!
            loss_reg = (
                0.05 * F.relu(F.softplus(wrap.fomp.delta_scale) - 0.8).pow(2).mean()
                + 0.02 * F.relu(F.softplus(wrap.fomp.emb_scale) - 2.5).pow(2).mean()
            )

            loss = loss_ce + 0.8 * loss_nce + 2.0 * loss_gate + 3.0 * loss_margin + 2.0 * loss_kl_neg + loss_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrap.fomp.parameters(), 1.0)
        opt.step()
        if st >= ema_start:
            with torch.no_grad():
                if ema_state is None:
                    ema_state = {k: v.detach().float().clone() for k, v in wrap.fomp.state_dict().items()}
                else:
                    for k, v in wrap.fomp.state_dict().items():
                        ema_state[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
        last_l_ce = float(loss_ce.item())
        last_l_nc = float(loss_nce.item())

        if st == 1 or (st == 50 and not smoke) or st % (5 if smoke else 200) == 0 or st == steps:
            with torch.no_grad():
                ce_dir = F.cross_entropy(lg_pos[:n_dir].reshape(-1, lg_pos.shape[-1]), lbl_pos[:n_dir].reshape(-1), ignore_index=-100).item()
                ce_par = F.cross_entropy(lg_pos[n_dir:].reshape(-1, lg_pos.shape[-1]), lbl_pos[n_dir:].reshape(-1), ignore_index=-100).item()
            tr_tau = wrap.fomp.fam_tau.clamp(wrap.fomp.config.min_tau, 0.88).item()
            te_tau = wrap.fomp.ent_tau.clamp(wrap.fomp.config.min_ent_tau, 0.90).item()
            v_end = F.softplus(wrap.fomp.delta_scale[-1]).item()
            e_end = F.softplus(wrap.fomp.emb_scale[-1]).item()
            print(
                f"  [FOMP-v3.6 Train st={st:4d}/{steps}] ce={last_l_ce:.3f} (dir={ce_dir:.3f} par={ce_par:.3f}) "
                f"nce={last_l_nc:.3f} kl_neg={loss_kl_neg.item():.4f} gate={loss_gate.item():.3f} "
                f"m_rel={loss_margin_rel.item():.4f} m_ent={loss_margin_ent.item():.4f} "
                f"tau=({tr_tau:.2f},{te_tau:.2f}) s=(v:{v_end:.2f},e:{e_end:.2f})",
                flush=True,
            )

    if ema_state is not None:
        wrap.fomp.load_state_dict({k: v.to(dtype=wrap.fomp.state_dict()[k].dtype, device=device) for k, v in ema_state.items()})

    sync(device)
    sec = round(time.perf_counter() - t0, 2)
    meta = {
        "train_steps": steps, "train_sec": sec, "fomp_params": sum(p.numel() for p in wrap.fomp.parameters()),
        "final_ce": round(last_l_ce, 4), "final_nce": round(last_l_nc, 4),
        "lr": lr, "ema_applied": ema_state is not None,
        "learned_tau_rel": round(float(wrap.fomp.fam_tau.clamp(wrap.fomp.config.min_tau, 0.88).item()), 4),
        "learned_tau_ent": round(float(wrap.fomp.ent_tau.clamp(wrap.fomp.config.min_ent_tau, 0.90).item()), 4),
        "learned_val_scale": [round(float(x), 3) for x in F.softplus(wrap.fomp.delta_scale).tolist()],
        "learned_emb_scale": [round(float(x), 3) for x in F.softplus(wrap.fomp.emb_scale).tolist()],
    }
    tmp = ckpt_path.with_suffix(".pt.tmp")
    torch.save({"fomp": wrap.fomp.state_dict(), "config": wrap.fomp.config.__dict__, "train_meta": meta}, tmp)
    os.replace(tmp, ckpt_path)
    return meta


# ==============================================================================
# LoRA + Replay Opponent (Exact Same Mid-to-Late Layers 10..27!)
# ==============================================================================

class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device, dtype=torch.float32) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))


def _run_lora_replay_arm(
    wrap: FOMPv3BackboneWrapper,
    tokenizer: Any,
    cf_subset: List[EditRecord],
    mq_subset: List[EditRecord],
    prose_tr_chunks: List[torch.Tensor],
    prose_ev_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    smoke: bool,
    steps: int = 35,
) -> Dict[str, Any]:
    """Attach rank-8 LoRA to layers[read_layer:], fine-tune on M test facts + WikiText-2 replay, eval, then restore."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    saved_modules: List[Tuple[nn.Module, str, nn.Linear]] = []
    lora_params: List[nn.Parameter] = []
    read_l = wrap.fomp.config.read_layer

    for lyr in wrap.layers[read_l:]:
        for parent_attr, child_names in (
            ("self_attn", ("q_proj", "v_proj", "o_proj")),
            (None, ("q_proj", "v_proj", "o_proj")),
            ("mlp", ("down_proj", "w_down")),
        ):
            parent = lyr if parent_attr is None else getattr(lyr, parent_attr, None)
            if parent is None:
                continue
            for cn in child_names:
                mod = getattr(parent, cn, None)
                if isinstance(mod, nn.Linear):
                    saved_modules.append((parent, cn, mod))
                    lora = _LoRALinear(mod, rank=8)
                    setattr(parent, cn, lora)
                    lora_params.extend([lora.A, lora.B])

    all_edits = cf_subset + mq_subset
    edit_seqs = [
        (tokenizer.encode(r.cue, add_special_tokens=False), tokenizer.encode(r.target_new, add_special_tokens=False))
        for r in all_edits
    ]
    opt = torch.optim.AdamW(lora_params, lr=3.0e-4, weight_decay=0.0)
    rng = np.random.default_rng(777 + len(all_edits))
    bs = min(8, len(edit_seqs))

    sync(device)
    t0 = time.perf_counter()
    try:
        for st in range(1, (8 if smoke else steps) + 1):
            idxs = rng.choice(len(edit_seqs), size=bs, replace=(len(edit_seqs) < bs))
            sub = [edit_seqs[int(i)] for i in idxs]
            mx = max(len(p) + len(t) for p, t in sub)
            inp = torch.full((bs, mx), pad_id, device=device, dtype=torch.long)
            msk = torch.zeros((bs, mx), device=device, dtype=torch.long)
            lbl = torch.full((bs, mx), -100, device=device, dtype=torch.long)
            for i, (p, t) in enumerate(sub):
                seq = p + t
                inp[i, : len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
                msk[i, : len(seq)] = 1
                for k_t, tid in enumerate(t):
                    lbl[i, len(p) - 1 + k_t] = tid

            rep = prose_tr_chunks[st % len(prose_tr_chunks)][: (48 if smoke else 256)].unsqueeze(0).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg_e = wrap.forward_logits(inp, attention_mask=msk, compiled=None)["logits"]
                l_edit = F.cross_entropy(lg_e.reshape(-1, lg_e.shape[-1]).float(), lbl.reshape(-1), ignore_index=-100)
                lg_r = wrap.forward_logits(rep[:, :-1], compiled=None)["logits"]
                l_rep = F.cross_entropy(lg_r.reshape(-1, lg_r.shape[-1]).float(), rep[:, 1:].reshape(-1))
                loss = l_edit + 0.5 * l_rep
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
        sync(device)
        edit_sec = round(time.perf_counter() - t0, 2)

        # Evaluate on the EXACT same `[:64]` CF + `[:32]` MQ held-out slice as Base, ICL, and FOMP-v3.2!
        res = _evaluate_edits(wrap, tokenizer, cf_subset[:64], mq_subset[:32], prose_ev_chunks, device, amp, compiled=None)
        res["edit_time_sec"] = edit_sec
        res["lora_params"] = sum(p.numel() for p in lora_params)
    finally:
        # Cleanly restore frozen base backbone!
        for parent, cn, orig in saved_modules:
            setattr(parent, cn, orig)
        del opt, lora_params
        gc.collect()
        if amp:
            torch.cuda.empty_cache()
    return res


def _eval_m1_episodes(
    wrap: FOMPv3BackboneWrapper,
    tokenizer: Any,
    cf_episodes: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    pad_id: int,
    max_tgt_tokens: int,
    mode: str,
) -> Dict[str, Any]:
    """Average M=1 single-edit metrics across K independent episodes (eliminates 1-sample noise!)."""
    groups = ("efficacy", "paraphrase", "neighborhood_retain")
    agg = {g: [0.0, 0.0, 0] for g in groups}
    comp_ms_list: List[float] = []
    prose_ppl = ms_q = 0.0
    extra_toks = 0
    for i, rec in enumerate(cf_episodes):
        comp_1 = None
        pref = ""
        if mode == "fomp":
            sync(device)
            t_c = time.perf_counter()
            comp_1 = wrap.compile_token_episodes(
                [tokenizer.encode(rec.cue, add_special_tokens=False)],
                [tokenizer.encode(rec.target_new, add_special_tokens=False)[:max_tgt_tokens]],
                pad_id=pad_id, device=device, g_ext=1.0,
            )
            sync(device)
            comp_ms_list.append((time.perf_counter() - t_c) * 1000.0)
        else:
            pref = f"Updated facts (always follow these overrides):\n- {rec.cue}{rec.target_new}.\n\n"
        ev = _evaluate_edits(
            wrap, tokenizer, [rec], [], prose_chunks, device, amp,
            compiled=comp_1, icl_prefix=pref, max_gen_eval=1, eval_prose=(i == 0),
        )
        if i == 0:
            prose_ppl, ms_q, extra_toks = ev["general_prose_ppl"], ev["ms_per_query"], ev["extra_prompt_tokens"]
        for g in groups:
            if ev[g]["n"] > 0:
                agg[g][0] += ev[g]["prob_win_pct"]
                agg[g][1] += ev[g]["gen_em_pct"]
                agg[g][2] += 1
    out: Dict[str, Any] = {
        g: {
            "prob_win_pct": round(agg[g][0] / max(1, agg[g][2]), 2),
            "gen_em_pct": round(agg[g][1] / max(1, agg[g][2]), 2),
            "n": agg[g][2],
        }
        for g in groups
    }
    out["mquake_2hop"] = {"prob_win_pct": 0.0, "gen_em_pct": 0.0, "n": 0}
    out["general_prose_ppl"] = prose_ppl
    out["extra_prompt_tokens"] = extra_toks
    out["ms_per_query"] = ms_q
    out["n_compiled_facts"] = 1
    if comp_ms_list:
        out["compile_ms"] = round(float(np.mean(comp_ms_list)), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--deadline", type=float, default=time.time() + 3600)
    parser.add_argument("--fomp-only", action="store_true", help="Run Base + FOMP-v3.4 only (skip ICL & LoRA baselines)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "fomp_v3_open_edit.json"

    if args.smoke:
        tokenizer: Any = _SmokeTokenizer()
        backbone: nn.Module = _SmokeHFModel(vocab=128, d=64, n_layers=8).to(device)
        fomp_cfg = FOMPv3Config(d_model=64, read_layer=4, inject_layers=(4, 6), key_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _make_smoke_datasets()
        m_grid = [1, 8, 16]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[Setup] Loading {args.model_id} in BF16 ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            backbone = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).to(device)
        except TypeError:
            backbone = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
        d_m = int(backbone.config.hidden_size)
        n_l = int(backbone.config.num_hidden_layers)
        rl = max(2, int(round(n_l * 0.50)))  # L14 for 28L Qwen2.5-1.5B: rich entity+relation & 1st-hop bridge!
        inj = tuple(sorted(set([rl, min(n_l - 2, rl + 3), min(n_l - 2, rl + 7), max(rl, n_l - 4), max(rl, n_l - 2)])))
        fomp_cfg = FOMPv3Config(
            d_model=d_m, read_layer=rl, inject_layers=inj,
            key_dim=384, val_dim=512, max_tgt_tokens=8, top_k=2, rel_hidden=128,
        )
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _load_real_datasets(tokenizer, max_tgt_tokens=8, seed=args.seed)
        m_grid = [1, 16, 64, 128]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prose_len = 48 if args.smoke else 256
    all_prose_chunks = [
        torch.tensor(tokenizer.encode(t, add_special_tokens=False)[: prose_len + 1], dtype=torch.long)
        for t in prose_txts
        if len(tokenizer.encode(t, add_special_tokens=False)) >= (16 if args.smoke else 64)
    ]
    # Strictly disjoint WikiText-2 train/replay vs held-out evaluation prose (0% prose overlap!)
    split_p = (len(all_prose_chunks) // 2) if args.smoke else min(20, len(all_prose_chunks) // 2)
    prose_tr_chunks = all_prose_chunks[:split_p]
    prose_ev_chunks = all_prose_chunks[split_p : split_p + (4 if args.smoke else 16)]
    assert prose_tr_chunks and prose_ev_chunks, "Need non-empty disjoint train and eval prose chunks!"

    torch.manual_seed(args.seed)
    fomp = FOMPv3Module(fomp_cfg).to(device)
    wrap = FOMPv3BackboneWrapper(backbone, fomp, freeze_backbone=True)

    # Warm up CUDA kernels before Arm 1 latency measurement
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        _ = wrap.forward_logits(prose_ev_chunks[0][:16].unsqueeze(0).to(device), compiled=None)
    sync(device)

    summary: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if amp else "cpu",
        "backbone": "smoke_8L" if args.smoke else args.model_id,
        "fomp_v3_config": fomp_cfg.__dict__,
        "fomp_v3_params": sum(p.numel() for p in fomp.parameters()),
        "dataset_sizes": {
            "cf_train": len(cf_tr), "cf_test": len(cf_te),
            "mq_train": len(mq_tr), "mq_test": len(mq_te),
            "prose_train": len(prose_tr_chunks), "prose_eval": len(prose_ev_chunks),
        },
    }

    # --- Arm 1: Base (Unedited Backbone) ---
    print("\n=== Arm 1: Base (Unedited) ===", flush=True)
    summary["arm1_base"] = _evaluate_edits(wrap, tokenizer, cf_te[:64], mq_te[:32], prose_ev_chunks, device, amp, compiled=None)
    base_ppl = summary["arm1_base"]["general_prose_ppl"]
    print("[Base]", json.dumps(summary["arm1_base"]), flush=True)
    save_json(out_json, summary)

    # --- Train FOMP-v3.4 on strictly disjoint train split ---
    print("\n=== Training FOMP-v3.4 (Dual-Lock 2-Layer MLP + Bridge-Entity + Multi-Paraphrase, 0% Overlap) ===", flush=True)
    summary["fomp_v3_train"] = _train_fomp_v3(
        wrap, tokenizer, cf_tr, mq_tr, prose_tr_chunks, device, amp, args.smoke,
        steps=(10 if args.smoke else args.train_steps), ckpt_path=(args.out / "fomp_v34.pt"), deadline=args.deadline,
    )
    save_json(out_json, summary)

    # --- Arm 4: FOMP-v3.4 across M = 1, 16, 64, 128 (+ g_ext=0 & ICML 2026 Sign-Flip!) ---
    print("\n=== Arm 4: FOMP-v3.4 (Ours, Dual-Lock MLP Single-Pass 1st-Order Compilation) ===", flush=True)
    fomp_res: Dict[str, Any] = {}
    m1_k = 4 if args.smoke else min(16, len(cf_te))
    for m_val in m_grid:
        if m_val == 1:
            ev = _eval_m1_episodes(wrap, tokenizer, cf_te[:m1_k], prose_ev_chunks, device, amp, pad_id, fomp_cfg.max_tgt_tokens, mode="fomp")
            ev["ppl_drift_vs_base"] = round(ev["general_prose_ppl"] - base_ppl, 5)
            fomp_res["M1"] = ev
            print(f"[FOMP-v3.4 M=1 (avg {m1_k}x)] compile={ev['compile_ms']}ms | {json.dumps(ev)}", flush=True)
            continue
        cf_sub = cf_te[: min(m_val, len(cf_te))]
        mq_n = min(max(8, m_val // 2), len(mq_te))
        mq_sub = mq_te[:mq_n]
        all_sub = cf_sub + mq_sub
        sync(device)
        t_c = time.perf_counter()
        comp_m = wrap.compile_token_episodes(
            [tokenizer.encode(r.cue, add_special_tokens=False) for r in all_sub],
            [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in all_sub],
            pad_id=pad_id, device=device, g_ext=1.0,
        )
        sync(device)
        comp_ms = round((time.perf_counter() - t_c) * 1000.0, 2)
        ev = _evaluate_edits(wrap, tokenizer, cf_sub[:64], mq_sub[:32], prose_ev_chunks, device, amp, compiled=comp_m)
        ev["n_compiled_facts"] = len(all_sub)
        ev["compile_ms"] = comp_ms
        ev["ppl_drift_vs_base"] = round(ev["general_prose_ppl"] - base_ppl, 5)
        if m_val == m_grid[-1] or m_val == 64:
            comp_g0 = wrap.compile_token_episodes(
                [tokenizer.encode(r.cue, add_special_tokens=False) for r in all_sub[:8]],
                [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in all_sub[:8]],
                pad_id=pad_id, device=device, g_ext=0.0,
            )
            ev["g_ext0_ppl_drift"] = round(abs(_eval_prose_ppl(wrap, prose_ev_chunks, device, amp, compiled=comp_g0) - base_ppl), 6)
            ev_sf = _evaluate_edits(wrap, tokenizer, cf_sub[:32], mq_sub[:16], prose_ev_chunks, device, amp, compiled=comp_m, sign_flip=True, max_gen_eval=16)
            ev["sign_flip"] = {"efficacy": ev_sf["efficacy"], "paraphrase": ev_sf["paraphrase"], "mquake_2hop": ev_sf["mquake_2hop"]}
        fomp_res[f"M{m_val}"] = ev
        print(f"[FOMP-v3.4 M={m_val}] compile={comp_ms}ms | {json.dumps(ev)}", flush=True)
    summary["arm4_fomp_v3"] = fomp_res
    save_json(out_json, summary)

    if args.fomp_only:
        print("\n=== FOMP-v3.4 FAST BENCHMARK COMPLETE (--fomp-only) ->", out_json, "===", flush=True)
        return

    # --- Arm 2: ICL (Prompt-Injecting All M Edited Facts!) ---
    print("\n=== Arm 2: ICL (In-Context Prompting) ===", flush=True)
    icl_res: Dict[str, Any] = {}
    for m_val in m_grid:
        if time.time() > args.deadline - 120:
            break
        if m_val == 1:
            ev = _eval_m1_episodes(wrap, tokenizer, cf_te[:m1_k], prose_ev_chunks, device, amp, pad_id, fomp_cfg.max_tgt_tokens, mode="icl")
            icl_res["M1"] = ev
            print(f"[ICL M=1 (avg {m1_k}x)] {json.dumps(ev)}", flush=True)
            continue
        cf_sub = cf_te[: min(m_val, len(cf_te))]
        mq_n = min(max(8, m_val // 2), len(mq_te))
        mq_sub = mq_te[:mq_n]
        lines = ["Updated facts (always follow these overrides):"] + [f"- {r.cue}{r.target_new}." for r in (cf_sub + mq_sub)]
        pref = "\n".join(lines) + "\n\n"
        ev = _evaluate_edits(wrap, tokenizer, cf_sub[:64], mq_sub[:32], prose_ev_chunks, device, amp, compiled=None, icl_prefix=pref)
        ev["n_compiled_facts"] = len(cf_sub) + len(mq_sub)
        icl_res[f"M{m_val}"] = ev
        print(f"[ICL M={m_val}] {json.dumps(ev)}", flush=True)
    summary["arm2_icl"] = icl_res
    save_json(out_json, summary)

    # --- Arm 3: LoRA (r=8) + Replay ---
    print("\n=== Arm 3: LoRA (r=8) + WikiText-2 Replay ===", flush=True)
    lora_res: Dict[str, Any] = {}
    for m_val in [m for m in m_grid if m >= 8]:
        if time.time() > args.deadline - 120:
            break
        cf_sub = cf_te[: min(m_val, len(cf_te))]
        mq_n = min(max(8, m_val // 2), len(mq_te))
        mq_sub = mq_te[:mq_n]
        ev = _run_lora_replay_arm(wrap, tokenizer, cf_sub, mq_sub, prose_tr_chunks, prose_ev_chunks, device, amp, args.smoke)
        ev["n_compiled_facts"] = len(cf_sub) + len(mq_sub)
        ev["ppl_drift_vs_base"] = round(ev["general_prose_ppl"] - base_ppl, 5)
        lora_res[f"M{m_val}"] = ev
        print(f"[LoRA+Replay M={m_val}] {json.dumps(ev)}", flush=True)
    summary["arm3_lora_replay"] = lora_res
    save_json(out_json, summary)

    print("\n=== FOMP-v3.3 4-ARM OPEN-DOMAIN BENCHMARK COMPLETE ->", out_json, "===", flush=True)


if __name__ == "__main__":
    main()
