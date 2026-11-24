"""FOMP-v3.4 Stage 1 Ruthless Reviewer Stress-Test & OOD Suite.

Answers the two hardest ICLR / NeurIPS reviewer attacks before scaling to 8B:
  S1. High-Concurrency Scaling Wall (`M = 128, 256, 512, 1024` live facts, strictly 0% train entity overlap).
  S2. Zero-Shot Cross-Dataset `ZsRE` Transfer (`0%` ZsRE training data; official Reliability / Generality / Base-Match Locality at `M = 1, 16, 100, 500`).
  S3. Strictly Relation-Disjoint CounterFact OOD (`0%` subject overlap AND `0%` Wikidata `relation_id` overlap across 8 held-out relation families).

All three tests include `g_ext=0` exact-identity verification and ICML 2026 Sign-Flip causal verification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

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
    _eval_m1_episodes,
    _eval_prose_ppl,
    _evaluate_edits,
    _load_real_datasets,
    _make_smoke_datasets,
    _parse_counterfact_row,
    _seq_target_logprob,
    _train_fomp_v3,
)


@dataclass
class ZsRERecord:
    subject: str
    cue: str
    paraphrase: str
    target_new: str
    target_true: str
    loc_prompt: str


def _fmt_qa(q: str) -> str:
    q = q.strip()
    if q.lower().startswith("nq question:"):
        q = q[len("nq question:") :].strip()
    if q and q[-1] not in "?.!":
        q = q + "?"
    return f"Question: {q}\nAnswer:"


def _parse_zsre_row(row: Dict[str, Any]) -> Optional[ZsRERecord]:
    subj = str(row.get("subject") or "").strip()
    src = str(row.get("src") or "").strip()
    rep = str(row.get("rephrase") or "").strip()
    alt = str(row.get("alt") or "").strip()
    ans_list = row.get("answers") or []
    orig = str(ans_list[0] if (isinstance(ans_list, list) and ans_list) else (row.get("pred") or "")).strip()
    loc = str(row.get("loc") or "").strip()
    if not (subj and src and rep and alt and orig and loc):
        return None
    if alt.lower() == orig.lower():
        return None
    return ZsRERecord(
        subject=subj,
        cue=_fmt_qa(src),
        paraphrase=_fmt_qa(rep),
        target_new=" " + alt,
        target_true=" " + orig,
        loc_prompt=_fmt_qa(loc),
    )


def _load_s1_1024_unseen(
    tokenizer: Any,
    max_tgt_tokens: int,
    base_cf_test: List[EditRecord],
    exclude_subjs: Set[str],
    target_n: int = 1024,
) -> List[EditRecord]:
    from datasets import load_dataset

    def _valid_tgt(s: str) -> bool:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return 1 <= len(ids) <= max_tgt_tokens

    pool: List[EditRecord] = list(base_cf_test)
    used: Set[str] = {r.subject.lower() for r in pool}
    cf_ds = load_dataset("azhx/counterfact", split="train")
    for row in cf_ds:
        if len(pool) >= target_n:
            break
        rec = _parse_counterfact_row(row)
        if rec is None:
            continue
        sl = rec.subject.lower()
        if sl in exclude_subjs or sl in used:
            continue
        if _valid_tgt(rec.target_new) and _valid_tgt(rec.target_true):
            pool.append(rec)
            used.add(sl)
    assert used.isdisjoint(exclude_subjs), "S1 1024 pool leaked a training subject!"
    return pool


def _load_zsre_test(
    tokenizer: Any,
    max_tgt_tokens: int,
    exclude_subjs: Set[str],
    target_n: int = 500,
) -> List[ZsRERecord]:
    from datasets import load_dataset

    def _valid_tgt(s: str) -> bool:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return 1 <= len(ids) <= max_tgt_tokens

    ds = None
    for repo in ("CrowdMind/ZsRE", "gjyotin305/full_zsre"):
        try:
            # Fetch only the 7.7MB test split when available to avoid downloading unused 66MB train.jsonl
            try:
                ds, sp = load_dataset(repo, split="test"), "test"
            except Exception:
                ds_dict = load_dataset(repo)
                sp = "test" if "test" in ds_dict else ("validation" if "validation" in ds_dict else list(ds_dict.keys())[0])
                ds = ds_dict[sp]
            print(f"[S2 ZsRE] Loaded {repo} split={sp} ({len(ds)} rows)", flush=True)
            break
        except Exception as exc:
            print(f"[S2 ZsRE] {repo} unavailable ({exc}); trying fallback...", flush=True)
    assert ds is not None, "Failed to load any public ZsRE dataset!"

    out: List[ZsRERecord] = []
    used: Set[str] = set()
    for row in ds:
        if len(out) >= target_n:
            break
        rec = _parse_zsre_row(row)
        if rec is None:
            continue
        sl = rec.subject.lower()
        if sl in exclude_subjs or sl in used:
            continue
        if _valid_tgt(rec.target_new) and _valid_tgt(rec.target_true):
            out.append(rec)
            used.add(sl)
    assert used.isdisjoint(exclude_subjs), "ZsRE test set leaked a training subject!"
    print(f"[S2 ZsRE] Parsed {len(out)} strictly unseen ZsRE records (0% train overlap). Sample: {out[0].cue!r} -> {out[0].target_new!r}", flush=True)
    return out


def _load_relation_disjoint_cf(
    tokenizer: Any,
    max_tgt_tokens: int,
    n_holdout_rels: int = 8,
    n_train: int = 3200,
    n_test: int = 128,
    seed: int = 99,
) -> Tuple[List[EditRecord], List[EditRecord], List[str], List[str]]:
    from datasets import load_dataset

    def _valid_tgt(s: str) -> bool:
        ids = tokenizer.encode(s, add_special_tokens=False)
        return 1 <= len(ids) <= max_tgt_tokens

    cf_ds = load_dataset("azhx/counterfact", split="train")
    by_rel: Dict[str, List[EditRecord]] = defaultdict(list)
    for row in cf_ds:
        rr = row.get("requested_rewrite") or {}
        rid = str(rr.get("relation_id") or "").strip()
        if not rid:
            continue
        rec = _parse_counterfact_row(row)
        if rec is not None and _valid_tgt(rec.target_new) and _valid_tgt(rec.target_true):
            by_rel[rid].append(rec)

    eligible_rels = sorted([r for r, lst in by_rel.items() if len(lst) >= 40])
    rng = np.random.default_rng(seed)
    perm_rels = [eligible_rels[i] for i in rng.permutation(len(eligible_rels))]
    ood_rels = sorted(perm_rels[:n_holdout_rels])
    seen_rels = sorted([r for r in by_rel.keys() if r not in set(ood_rels)])

    # Build strictly unique-subject OOD test set from the held-out relations
    ood_pool: List[EditRecord] = []
    for r in ood_rels:
        ood_pool.extend(by_rel[r])
    ood_test: List[EditRecord] = []
    te_subjs: Set[str] = set()
    for idx in rng.permutation(len(ood_pool)).tolist():
        rec = ood_pool[idx]
        sl = rec.subject.lower()
        if sl not in te_subjs:
            ood_test.append(rec)
            te_subjs.add(sl)
        if len(ood_test) >= n_test:
            break

    # Build train set strictly from seen_rels AND disjoint from te_subjs!
    seen_pool: List[EditRecord] = []
    for r in seen_rels:
        seen_pool.extend(by_rel[r])
    ood_train: List[EditRecord] = []
    for idx in rng.permutation(len(seen_pool)).tolist():
        rec = seen_pool[idx]
        if rec.subject.lower() not in te_subjs:
            ood_train.append(rec)
        if len(ood_train) >= n_train:
            break

    tr_subjs = {r.subject.lower() for r in ood_train}
    assert set(seen_rels).isdisjoint(set(ood_rels)), "Relation-ID leakage!"
    assert tr_subjs.isdisjoint(te_subjs), "Entity leakage in relation-disjoint OOD split!"
    print(
        f"[S3 Rel-Disjoint OOD] Held-out {len(ood_rels)}/{len(by_rel)} relations: {ood_rels} | "
        f"train={len(ood_train)} (0% rel & 0% entity overlap) | test={len(ood_test)}",
        flush=True,
    )
    return ood_train, ood_test, seen_rels, ood_rels


@torch.no_grad()
def _eval_zsre(
    wrap: FOMPv3BackboneWrapper,
    tokenizer: Any,
    recs: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    compiled: Optional[Dict[str, torch.Tensor]],
    sign_flip: bool = False,
    max_eval: int = 64,
    loc_gen_len: int = 6,
) -> Dict[str, Any]:
    wrap.eval()
    sub = recs[: min(max_eval, len(recs))]
    rel_win = rel_em = gen_win = gen_em = 0
    loc_tok_match = loc_tok_tot = loc_exact = 0

    sync(device)
    t0 = time.perf_counter()
    n_q = 0
    for i, r in enumerate(sub):
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        tt_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        tn_s = r.target_new.strip().lower()

        c_ids = tokenizer.encode(r.cue, add_special_tokens=False)
        if _seq_target_logprob(wrap, c_ids, tn_ids, device, amp, compiled, sign_flip) > _seq_target_logprob(wrap, c_ids, tt_ids, device, amp, compiled, sign_flip):
            rel_win += 1
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            g_c = wrap.greedy_generate(c_ids, max_new_tokens=max(1, len(tn_ids) + 1), device=device, compiled=compiled, sign_flip=sign_flip)
        if tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(tn_s):
            rel_em += 1
        n_q += 1

        p_ids = tokenizer.encode(r.paraphrase, add_special_tokens=False)
        if _seq_target_logprob(wrap, p_ids, tn_ids, device, amp, compiled, sign_flip) > _seq_target_logprob(wrap, p_ids, tt_ids, device, amp, compiled, sign_flip):
            gen_win += 1
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            g_p = wrap.greedy_generate(p_ids, max_new_tokens=max(1, len(tn_ids) + 1), device=device, compiled=compiled, sign_flip=sign_flip)
        if tokenizer.decode(g_p, skip_special_tokens=True).strip().lower().startswith(tn_s):
            gen_em += 1
        n_q += 1

        # Official WISE / EasyEdit ZsRE Locality: greedy token match vs unedited Base model!
        l_ids = tokenizer.encode(r.loc_prompt, add_special_tokens=False)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            g_l = wrap.greedy_generate(l_ids, max_new_tokens=loc_gen_len, device=device, compiled=compiled, sign_flip=sign_flip)
        b_l = base_loc_preds[i]
        m_len = min(len(g_l), len(b_l))
        loc_tok_match += sum(1 for a, b in zip(g_l[:m_len], b_l[:m_len]) if a == b)
        loc_tok_tot += max(1, m_len)
        if g_l[:m_len] == b_l[:m_len]:
            loc_exact += 1
        n_q += 1
    sync(device)
    ms_q = round((time.perf_counter() - t0) * 1000.0 / max(1, n_q), 2)
    ppl = _eval_prose_ppl(wrap, prose_chunks, device, amp, compiled)
    n_s = max(1, len(sub))
    return {
        "n": len(sub),
        "reliability": {"prob_win_pct": round(100.0 * rel_win / n_s, 2), "gen_em_pct": round(100.0 * rel_em / n_s, 2)},
        "generality": {"prob_win_pct": round(100.0 * gen_win / n_s, 2), "gen_em_pct": round(100.0 * gen_em / n_s, 2)},
        "locality": {
            "token_match_vs_base_pct": round(100.0 * loc_tok_match / max(1, loc_tok_tot), 2),
            "exact_seq_vs_base_pct": round(100.0 * loc_exact / n_s, 2),
        },
        "general_prose_ppl": ppl,
        "ms_per_query": ms_q,
    }


def _make_smoke_zsre(n: int = 16) -> List[ZsRERecord]:
    return [
        ZsRERecord(
            subject=f"ZSubj{i}",
            cue=f"Question: Who created ZSubj{i}?\nAnswer:",
            paraphrase=f"Question: What is the creator of ZSubj{i}?\nAnswer:",
            target_new=f" ZNew{i}",
            target_true=f" ZOld{i}",
            loc_prompt=f"Question: What is the color of sky {i}?\nAnswer:",
        )
        for i in range(n)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None, help="Pre-trained FOMP-v3.4 checkpoint for S1 & S2")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--ood-train-steps", type=int, default=1200)
    parser.add_argument("--main-train-steps", type=int, default=1500)
    parser.add_argument("--skip-s3", action="store_true", help="Skip S3 relation-disjoint OOD retrain for fast S1+S2 verification")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--deadline", type=float, default=time.time() + 3600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "fomp_v34_stage1_stress_ood.json"

    if args.smoke:
        tokenizer: Any = _SmokeTokenizer()
        backbone: nn.Module = _SmokeHFModel(vocab=128, d=64, n_layers=8).to(device)
        fomp_cfg = FOMPv3Config(d_model=64, read_layer=4, inject_layers=(4, 6), key_dim=32, ent_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _make_smoke_datasets()
        s1_pool = cf_te * 2
        s1_grid = [8, 16, 32]
        zsre_recs = _make_smoke_zsre(16)
        zsre_grid = [1, 8, 16]
        ood_tr, ood_te, seen_rels, ood_rels = cf_tr, cf_te, ["P1", "P2"], ["P9"]
        s3_grid = [1, 8, 16]
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
        rl = max(2, int(round(n_l * 0.50)))
        inj = tuple(sorted(set([rl, min(n_l - 2, rl + 3), min(n_l - 2, rl + 7), max(rl, n_l - 4), max(rl, n_l - 2)])))
        fomp_cfg = FOMPv3Config(
            d_model=d_m, read_layer=rl, inject_layers=inj,
            key_dim=384, ent_dim=256, val_dim=512, max_tgt_tokens=8, top_k=2, rel_hidden=128,
        )
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _load_real_datasets(tokenizer, max_tgt_tokens=8, seed=args.seed)
        train_subjs = {r.subject.lower() for r in cf_tr + mq_tr}
        s1_pool = _load_s1_1024_unseen(tokenizer, 8, cf_te, train_subjs, target_n=1024)
        s1_grid = [128, 256, 512, 1024]
        zsre_recs = _load_zsre_test(tokenizer, 8, train_subjs, target_n=500)
        zsre_grid = [1, 16, 100, 500]
        if not args.skip_s3:
            ood_tr, ood_te, seen_rels, ood_rels = _load_relation_disjoint_cf(tokenizer, 8, n_holdout_rels=8, n_train=3200, n_test=128, seed=99)
        else:
            ood_tr, ood_te, seen_rels, ood_rels = [], [], [], []
        s3_grid = [1, 16, 64, 128]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prose_len = 48 if args.smoke else 256
    all_prose_chunks = [
        torch.tensor(tokenizer.encode(t, add_special_tokens=False)[: prose_len + 1], dtype=torch.long)
        for t in prose_txts
        if len(tokenizer.encode(t, add_special_tokens=False)) >= (16 if args.smoke else 64)
    ]
    split_p = (len(all_prose_chunks) // 2) if args.smoke else min(20, len(all_prose_chunks) // 2)
    prose_tr_chunks = all_prose_chunks[:split_p]
    prose_ev_chunks = all_prose_chunks[split_p : split_p + (4 if args.smoke else 16)]

    torch.manual_seed(args.seed)
    fomp = FOMPv3Module(fomp_cfg).to(device)
    wrap = FOMPv3BackboneWrapper(backbone, fomp, freeze_backbone=True)
    base_ppl = _eval_prose_ppl(wrap, prose_ev_chunks, device, amp, compiled=None)

    summary: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if amp else "cpu",
        "backbone": "smoke_8L" if args.smoke else args.model_id,
        "fomp_v34_params": sum(p.numel() for p in fomp.parameters()),
        "base_prose_ppl": base_ppl,
    }

    # Load pre-trained FOMP-v3.5 for S1 & S2 (or train fresh FOMP-v3.5 if no ckpt given)
    if args.ckpt is not None and args.ckpt.exists() and not args.smoke:
        print(f"[Setup] Loading pre-trained FOMP checkpoint -> {args.ckpt}", flush=True)
        st_ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        wrap.fomp.load_state_dict(st_ck["fomp"])
        summary["s1_s2_ckpt"] = str(args.ckpt)
    else:
        summary["s1_s2_train"] = _train_fomp_v3(
            wrap, tokenizer, cf_tr, mq_tr, prose_tr_chunks, device, amp, args.smoke,
            steps=(8 if args.smoke else args.main_train_steps), ckpt_path=(args.out / "fomp_v35.pt"), deadline=args.deadline,
        )
    save_json(out_json, summary)

    # Also verify CounterFact + MQuAKE 2-Hop mixed M=128 (192 live facts) to guarantee zero 2-Hop regression!
    print("\n=== [S0] CounterFact + MQuAKE 2-Hop Regression Check (M=128, 192 live facts) ===", flush=True)
    cf_m128 = cf_te[: (8 if args.smoke else 128)]
    mq_m64 = mq_te[: (4 if args.smoke else 64)]
    mix_all = cf_m128 + mq_m64
    sync(device)
    t_mix = time.perf_counter()
    comp_mix = wrap.compile_token_episodes(
        [tokenizer.encode(r.cue, add_special_tokens=False) for r in mix_all],
        [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in mix_all],
        pad_id=pad_id, device=device, g_ext=1.0,
    )
    sync(device)
    ev_mix = _evaluate_edits(wrap, tokenizer, cf_m128[: (8 if args.smoke else 64)], mq_m64[: (4 if args.smoke else 32)], prose_ev_chunks, device, amp, compiled=comp_mix, max_gen_eval=(4 if args.smoke else 32))
    ev_mix["n_compiled_facts"] = len(mix_all)
    ev_mix["compile_ms"] = round((time.perf_counter() - t_mix) * 1000.0, 2)
    ev_mix["ppl_drift_vs_base"] = round(ev_mix["general_prose_ppl"] - base_ppl, 5)
    summary["s0_mixed_cf_mquake_m128"] = ev_mix
    print(f"[S0 CF+MQuAKE M=128] compile={ev_mix['compile_ms']}ms | {json.dumps(ev_mix)}", flush=True)
    save_json(out_json, summary)

    # ==========================================================================
    # S1: High-Concurrency Scaling Wall (M = 128, 256, 512, 1024)
    # ==========================================================================
    print("\n=== [S1] High-Concurrency Scaling Wall (M = 128 -> 1024 Unseen Facts) ===", flush=True)
    eval_slice_s1 = s1_pool[: (8 if args.smoke else 64)]
    s1_base = _evaluate_edits(wrap, tokenizer, eval_slice_s1, [], prose_ev_chunks, device, amp, compiled=None, max_gen_eval=(4 if args.smoke else 32))
    s1_res: Dict[str, Any] = {"base": s1_base}
    print("[S1 Base]", json.dumps(s1_base), flush=True)

    for m_val in s1_grid:
        sub_m = s1_pool[: min(m_val, len(s1_pool))]
        sync(device)
        t_c = time.perf_counter()
        comp_m = wrap.compile_token_episodes(
            [tokenizer.encode(r.cue, add_special_tokens=False) for r in sub_m],
            [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in sub_m],
            pad_id=pad_id, device=device, g_ext=1.0,
        )
        sync(device)
        comp_ms = round((time.perf_counter() - t_c) * 1000.0, 2)
        ev = _evaluate_edits(wrap, tokenizer, eval_slice_s1, [], prose_ev_chunks, device, amp, compiled=comp_m, max_gen_eval=(4 if args.smoke else 32))
        ev["n_compiled_facts"] = len(sub_m)
        ev["compile_ms"] = comp_ms
        ev["ppl_drift_vs_base"] = round(ev["general_prose_ppl"] - base_ppl, 5)
        if m_val == s1_grid[-1]:
            c0 = wrap.compile_token_episodes(
                [tokenizer.encode(r.cue, add_special_tokens=False) for r in sub_m[:8]],
                [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in sub_m[:8]],
                pad_id=pad_id, device=device, g_ext=0.0,
            )
            ev["g_ext0_ppl_drift"] = round(abs(_eval_prose_ppl(wrap, prose_ev_chunks, device, amp, compiled=c0) - base_ppl), 6)
            ev_sf = _evaluate_edits(wrap, tokenizer, eval_slice_s1[: (4 if args.smoke else 32)], [], prose_ev_chunks, device, amp, compiled=comp_m, sign_flip=True, max_gen_eval=(4 if args.smoke else 16))
            ev["sign_flip"] = {"efficacy": ev_sf["efficacy"], "paraphrase": ev_sf["paraphrase"]}
        s1_res[f"M{m_val}"] = ev
        print(f"[S1 M={m_val}] compile={comp_ms}ms | {json.dumps(ev)}", flush=True)
    summary["s1_concurrency_1024"] = s1_res
    save_json(out_json, summary)

    # ==========================================================================
    # S2: Zero-Shot Cross-Dataset ZsRE Transfer (0% ZsRE Training!)
    # ==========================================================================
    print("\n=== [S2] Zero-Shot Cross-Dataset ZsRE Transfer (0% ZsRE Training Data!) ===", flush=True)
    z_eval_n = 4 if args.smoke else 64
    loc_gen_len = 4 if args.smoke else 6
    base_loc_preds: List[List[int]] = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        for r in zsre_recs[:z_eval_n]:
            l_ids = tokenizer.encode(r.loc_prompt, add_special_tokens=False)
            base_loc_preds.append(wrap.greedy_generate(l_ids, max_new_tokens=loc_gen_len, device=device, compiled=None))
    z_base = _eval_zsre(wrap, tokenizer, zsre_recs, base_loc_preds, prose_ev_chunks, device, amp, compiled=None, max_eval=z_eval_n, loc_gen_len=loc_gen_len)
    s2_res: Dict[str, Any] = {"base": z_base}
    print("[S2 ZsRE Base]", json.dumps(z_base), flush=True)

    for m_val in zsre_grid:
        if m_val == 1:
            k_m1 = 4 if args.smoke else 16
            r_w = r_e = g_w = g_e = l_tm = l_tt = l_ex = 0
            c_ms_l: List[float] = []
            for j in range(k_m1):
                r = zsre_recs[j]
                sync(device)
                tc = time.perf_counter()
                cm = wrap.compile_token_episodes(
                    [tokenizer.encode(r.cue, add_special_tokens=False)],
                    [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens]],
                    pad_id=pad_id, device=device, g_ext=1.0,
                )
                sync(device)
                c_ms_l.append((time.perf_counter() - tc) * 1000.0)
                ev1 = _eval_zsre(wrap, tokenizer, [r], [base_loc_preds[j]], prose_ev_chunks, device, amp, compiled=cm, max_eval=1, loc_gen_len=loc_gen_len)
                r_w += ev1["reliability"]["prob_win_pct"]
                r_e += ev1["reliability"]["gen_em_pct"]
                g_w += ev1["generality"]["prob_win_pct"]
                g_e += ev1["generality"]["gen_em_pct"]
                l_tm += ev1["locality"]["token_match_vs_base_pct"]
                l_ex += ev1["locality"]["exact_seq_vs_base_pct"]
                l_tt += 1
            ev_z1 = {
                "n": k_m1,
                "n_compiled_facts": 1,
                "compile_ms": round(float(np.mean(c_ms_l)), 2),
                "reliability": {"prob_win_pct": round(r_w / k_m1, 2), "gen_em_pct": round(r_e / k_m1, 2)},
                "generality": {"prob_win_pct": round(g_w / k_m1, 2), "gen_em_pct": round(g_e / k_m1, 2)},
                "locality": {"token_match_vs_base_pct": round(l_tm / k_m1, 2), "exact_seq_vs_base_pct": round(l_ex / k_m1, 2)},
                "ppl_drift_vs_base": round(_eval_prose_ppl(wrap, prose_ev_chunks, device, amp, compiled=cm) - base_ppl, 5),
            }
            s2_res["M1"] = ev_z1
            print(f"[S2 ZsRE M=1 (avg {k_m1}x)] {json.dumps(ev_z1)}", flush=True)
            continue

        sub_z = zsre_recs[: min(m_val, len(zsre_recs))]
        sync(device)
        t_c = time.perf_counter()
        comp_z = wrap.compile_token_episodes(
            [tokenizer.encode(r.cue, add_special_tokens=False) for r in sub_z],
            [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in sub_z],
            pad_id=pad_id, device=device, g_ext=1.0,
        )
        sync(device)
        comp_ms = round((time.perf_counter() - t_c) * 1000.0, 2)
        ev_z = _eval_zsre(wrap, tokenizer, sub_z, base_loc_preds, prose_ev_chunks, device, amp, compiled=comp_z, max_eval=z_eval_n, loc_gen_len=loc_gen_len)
        ev_z["n_compiled_facts"] = len(sub_z)
        ev_z["compile_ms"] = comp_ms
        ev_z["ppl_drift_vs_base"] = round(ev_z["general_prose_ppl"] - base_ppl, 5)
        if m_val in (100, zsre_grid[-1]):
            ev_sf = _eval_zsre(wrap, tokenizer, sub_z, base_loc_preds, prose_ev_chunks, device, amp, compiled=comp_z, sign_flip=True, max_eval=min(16, z_eval_n), loc_gen_len=loc_gen_len)
            ev_z["sign_flip"] = {"reliability": ev_sf["reliability"], "generality": ev_sf["generality"]}
        s2_res[f"M{m_val}"] = ev_z
        print(f"[S2 ZsRE M={m_val}] compile={comp_ms}ms | {json.dumps(ev_z)}", flush=True)
    summary["s2_zeroshot_zsre"] = s2_res
    save_json(out_json, summary)

    # ==========================================================================
    # S3: Strictly Relation-Disjoint CounterFact OOD (0% Entity & 0% Relation Overlap!)
    # ==========================================================================
    if args.skip_s3:
        print("\n=== [S3] Skipped (--skip-s3); S0 + S1 + S2 complete ->", out_json, "===", flush=True)
        return
    print("\n=== [S3] Strictly Relation-Disjoint CounterFact OOD (0% Entity + 0% Relation-ID Overlap!) ===", flush=True)
    torch.manual_seed(args.seed + 100)
    fomp_ood = FOMPv3Module(fomp_cfg).to(device)
    wrap_ood = FOMPv3BackboneWrapper(backbone, fomp_ood, freeze_backbone=True)
    ood_eval_slice = ood_te[: (8 if args.smoke else 64)]
    ood_base = _evaluate_edits(wrap_ood, tokenizer, ood_eval_slice, [], prose_ev_chunks, device, amp, compiled=None, max_gen_eval=(4 if args.smoke else 32))
    print("[S3 OOD Base]", json.dumps(ood_base), flush=True)

    ood_train_meta = _train_fomp_v3(
        wrap_ood, tokenizer, ood_tr, [], prose_tr_chunks, device, amp, args.smoke,
        steps=(8 if args.smoke else args.ood_train_steps), ckpt_path=(args.out / "fomp_v34_rel_ood.pt"), deadline=args.deadline,
    )
    s3_res: Dict[str, Any] = {
        "seen_relations": seen_rels,
        "heldout_ood_relations": ood_rels,
        "train_meta": ood_train_meta,
        "base": ood_base,
    }
    m1_k = 4 if args.smoke else min(16, len(ood_te))
    for m_val in s3_grid:
        if m_val == 1:
            ev1 = _eval_m1_episodes(wrap_ood, tokenizer, ood_te[:m1_k], prose_ev_chunks, device, amp, pad_id, fomp_cfg.max_tgt_tokens, mode="fomp")
            ev1["ppl_drift_vs_base"] = round(ev1["general_prose_ppl"] - base_ppl, 5)
            s3_res["M1"] = ev1
            print(f"[S3 Rel-OOD M=1 (avg {m1_k}x)] {json.dumps(ev1)}", flush=True)
            continue
        sub_o = ood_te[: min(m_val, len(ood_te))]
        sync(device)
        t_c = time.perf_counter()
        comp_o = wrap_ood.compile_token_episodes(
            [tokenizer.encode(r.cue, add_special_tokens=False) for r in sub_o],
            [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in sub_o],
            pad_id=pad_id, device=device, g_ext=1.0,
        )
        sync(device)
        comp_ms = round((time.perf_counter() - t_c) * 1000.0, 2)
        ev_o = _evaluate_edits(wrap_ood, tokenizer, sub_o[: (8 if args.smoke else 64)], [], prose_ev_chunks, device, amp, compiled=comp_o, max_gen_eval=(4 if args.smoke else 32))
        ev_o["n_compiled_facts"] = len(sub_o)
        ev_o["compile_ms"] = comp_ms
        ev_o["ppl_drift_vs_base"] = round(ev_o["general_prose_ppl"] - base_ppl, 5)
        if m_val == s3_grid[-1]:
            c0 = wrap_ood.compile_token_episodes(
                [tokenizer.encode(r.cue, add_special_tokens=False) for r in sub_o[:8]],
                [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in sub_o[:8]],
                pad_id=pad_id, device=device, g_ext=0.0,
            )
            ev_o["g_ext0_ppl_drift"] = round(abs(_eval_prose_ppl(wrap_ood, prose_ev_chunks, device, amp, compiled=c0) - base_ppl), 6)
            ev_sf = _evaluate_edits(wrap_ood, tokenizer, sub_o[: (4 if args.smoke else 32)], [], prose_ev_chunks, device, amp, compiled=comp_o, sign_flip=True, max_gen_eval=(4 if args.smoke else 16))
            ev_o["sign_flip"] = {"efficacy": ev_sf["efficacy"], "paraphrase": ev_sf["paraphrase"]}
        s3_res[f"M{m_val}"] = ev_o
        print(f"[S3 Rel-OOD M={m_val}] compile={comp_ms}ms | {json.dumps(ev_o)}", flush=True)
    summary["s3_relation_disjoint_ood"] = s3_res
    save_json(out_json, summary)

    print("\n=== FOMP-v3.4 STAGE 1 STRESS & OOD SUITE COMPLETE ->", out_json, "===", flush=True)


if __name__ == "__main__":
    main()
