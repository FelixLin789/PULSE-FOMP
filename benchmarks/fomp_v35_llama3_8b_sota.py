"""Stage 2 Publication-Grade SOTA Showdown on `Llama-3-8B-Instruct`.

Strict zero-strawman, identical-backbone, identical-split, identical-metric comparison across 6 arms:
  1. `Base`      : Unedited `NousResearch/Meta-Llama-3-8B-Instruct`
  2. `FOMP-v3.5` : Ours (Dual-Lock 2-Layer MLP + Format-Invariant Anti-Shortcut + Single-Pass 1st-Order)
  3. `GRACE`     : Official `zjunlp/EasyEdit` GRACE codebook adapter
  4. `WISE`      : Official `zjunlp/EasyEdit` WISE dual-memory editor (NeurIPS 2024)
  5. `AlphaEdit` : Official `zjunlp/EasyEdit` AlphaEdit null-space editor (ICLR 2025 Outstanding Paper)
  6. `MEMIT`     : Official `zjunlp/EasyEdit` MEMIT mass-editing baseline

Evaluated across 3 standard tracks (0% overlap with FOMP train split; 0% ZsRE training data for FOMP!):
  - Track 1: CounterFact (`M = 100` for all 6 arms; plus `M = 500, 1000` scaling wall for FOMP-v3.5)
  - Track 2: MQuAKE-Remastered Multi-Hop (`M = 64` for all 6 arms)
  - Track 3: Zero-Shot ZsRE (`M = 100` for all 6 arms; official Reliability / Generality / Base-Match Locality)

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

from pulse.fomp_v3 import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module  # noqa: E402
from v7_stage0_stage1 import save_json, sync  # noqa: E402
from fomp_v3_open_edit import (  # noqa: E402
    EditRecord,
    _SmokeHFModel,
    _SmokeTokenizer,
    _eval_prose_ppl,
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


# ==============================================================================
# Unified BOS-Consistent Evaluation Helpers (100% Identical Across ALL 6 Arms!)
# ==============================================================================

def _enc_prompt(tokenizer: Any, text: str, prepend_bos: bool) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    bos_id = getattr(tokenizer, "bos_token_id", None) if prepend_bos else None
    if bos_id is not None and (not ids or ids[0] != int(bos_id)):
        return [int(bos_id)] + ids
    return ids


@torch.no_grad()
def _forward_logits_any(
    model: nn.Module,
    input_ids: torch.Tensor,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    sign_flip: bool = False,
) -> torch.Tensor:
    if wrap is not None and compiled is not None:
        return wrap.forward_logits(input_ids, compiled=compiled, sign_flip=sign_flip)["logits"]
    if hasattr(model, "model"):
        out = model(input_ids=input_ids, use_cache=False)
    else:
        out = model(input_ids)
    if isinstance(out, dict):
        return out["logits"]
    if hasattr(out, "logits"):
        return out.logits
    return out[0]


@torch.no_grad()
def _score_logprob_any(
    model: nn.Module,
    prompt_ids: List[int],
    tgt_ids: List[int],
    device: torch.device,
    amp: bool,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    sign_flip: bool = False,
) -> float:
    full = torch.tensor([prompt_ids + tgt_ids], device=device, dtype=torch.long)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        lg = _forward_logits_any(model, full, wrap=wrap, compiled=compiled, sign_flip=sign_flip)[0]
    p_len = len(prompt_ids)
    pred = lg[p_len - 1 : p_len + len(tgt_ids) - 1].float()
    tgt = torch.tensor(tgt_ids, device=device, dtype=torch.long)
    return float(-F.cross_entropy(pred, tgt, reduction="mean").item())


@torch.no_grad()
def _greedy_gen_any(
    model: nn.Module,
    prompt_ids: List[int],
    max_new_tokens: int,
    device: torch.device,
    amp: bool,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    sign_flip: bool = False,
) -> List[int]:
    if wrap is not None and compiled is not None:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            return wrap.greedy_generate(prompt_ids, max_new_tokens=max_new_tokens, device=device, compiled=compiled, sign_flip=sign_flip)
    cur = list(prompt_ids)
    gen: List[int] = []
    for _ in range(max_new_tokens):
        x = torch.tensor([cur], device=device, dtype=torch.long)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg = _forward_logits_any(model, x)
        nxt = int(lg[0, -1].argmax(dim=-1).item())
        gen.append(nxt)
        cur.append(nxt)
    return gen


@torch.no_grad()
def _eval_ppl_any(
    model: nn.Module,
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
) -> float:
    if wrap is not None and compiled is not None:
        return _eval_prose_ppl(wrap, prose_chunks, device, amp, compiled=compiled)
    nlls: List[float] = []
    for ch in prose_chunks:
        ch_d = ch.unsqueeze(0).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg = _forward_logits_any(model, ch_d[:, :-1])
            nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), ch_d[:, 1:].reshape(-1))
        nlls.append(float(nll.item()))
    return round(math.exp(min(20.0, float(np.mean(nlls)))), 4)


@torch.no_grad()
def _eval_track1_cf(
    model: nn.Module,
    tokenizer: Any,
    cf_recs: List[EditRecord],
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    sign_flip: bool = False,
    max_eval: int = 64,
    max_gen: int = 32,
) -> Dict[str, Any]:
    model.eval()
    sub = cf_recs[: min(max_eval, len(cf_recs))]

    def _run(pairs: List[Tuple[str, str, str]], want_new: bool) -> Dict[str, Any]:
        win = em = g_cnt = 0
        for idx, (pr, t_new, t_orig) in enumerate(pairs):
            p_ids = _enc_prompt(tokenizer, pr, prepend_bos)
            tn_ids = tokenizer.encode(t_new, add_special_tokens=False)
            to_ids = tokenizer.encode(t_orig, add_special_tokens=False)
            lp_n = _score_logprob_any(model, p_ids, tn_ids, device, amp, wrap, compiled, sign_flip)
            lp_o = _score_logprob_any(model, p_ids, to_ids, device, amp, wrap, compiled, sign_flip)
            if (lp_n > lp_o) if want_new else (lp_o > lp_n):
                win += 1
            if idx < max_gen:
                g_cnt += 1
                want_s = (t_new if want_new else t_orig).strip().lower()
                want_len = len(tn_ids if want_new else to_ids)
                g_ids = _greedy_gen_any(model, p_ids, max(1, want_len + 1), device, amp, wrap, compiled, sign_flip)
                if tokenizer.decode(g_ids, skip_special_tokens=True).strip().lower().startswith(want_s):
                    em += 1
        n = max(1, len(pairs))
        return {
            "prob_win_pct": round(100.0 * win / n, 2),
            "gen_em_pct": round(100.0 * em / max(1, g_cnt), 2),
            "n": len(pairs),
        }

    sync(device)
    t0 = time.perf_counter()
    eff = _run([(r.cue, r.target_new, r.target_true) for r in sub], want_new=True)
    par = _run([(r.paraphrases[0], r.target_new, r.target_true) for r in sub if r.paraphrases], want_new=True)
    nei = _run([(r.neighborhoods[0], r.target_new, r.target_true) for r in sub if r.neighborhoods], want_new=False)
    sync(device)
    ms_q = round((time.perf_counter() - t0) * 1000.0 / max(1, eff["n"] + par["n"] + nei["n"]), 2)
    ppl = _eval_ppl_any(model, prose_chunks, device, amp, wrap, compiled)
    return {"efficacy": eff, "paraphrase": par, "neighborhood_retain": nei, "general_prose_ppl": ppl, "ms_per_query": ms_q}


@torch.no_grad()
def _eval_track2_mquake(
    model: nn.Module,
    tokenizer: Any,
    mq_recs: List[EditRecord],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    sign_flip: bool = False,
    max_eval: int = 32,
) -> Dict[str, Any]:
    model.eval()
    sub = mq_recs[: min(max_eval, len(mq_recs))]
    s_win = s_em = m_win = m_em = 0
    for r in sub:
        # Single-hop direct check
        c_ids = _enc_prompt(tokenizer, r.cue, prepend_bos)
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        if _score_logprob_any(model, c_ids, tn_ids, device, amp, wrap, compiled, sign_flip) > _score_logprob_any(model, c_ids, to_ids, device, amp, wrap, compiled, sign_flip):
            s_win += 1
        g_c = _greedy_gen_any(model, c_ids, max(1, len(tn_ids) + 1), device, amp, wrap, compiled, sign_flip)
        if tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(r.target_new.strip().lower()):
            s_em += 1

        # 2-Hop multi-hop check
        q_ids = _enc_prompt(tokenizer, r.paraphrases[0], prepend_bos)
        an_ids = tokenizer.encode(r.multihop_answer_new, add_special_tokens=False)
        ao_ids = tokenizer.encode(r.multihop_answer_true, add_special_tokens=False)
        if _score_logprob_any(model, q_ids, an_ids, device, amp, wrap, compiled, sign_flip) > _score_logprob_any(model, q_ids, ao_ids, device, amp, wrap, compiled, sign_flip):
            m_win += 1
        g_q = _greedy_gen_any(model, q_ids, max(1, len(an_ids) + 1), device, amp, wrap, compiled, sign_flip)
        if tokenizer.decode(g_q, skip_special_tokens=True).strip().lower().startswith(r.multihop_answer_new.strip().lower()):
            m_em += 1
    n = max(1, len(sub))
    return {
        "n": len(sub),
        "single_hop_efficacy": {"prob_win_pct": round(100.0 * s_win / n, 2), "gen_em_pct": round(100.0 * s_em / n, 2)},
        "mquake_2hop": {"prob_win_pct": round(100.0 * m_win / n, 2), "gen_em_pct": round(100.0 * m_em / n, 2)},
    }


@torch.no_grad()
def _eval_track3_zsre(
    model: nn.Module,
    tokenizer: Any,
    z_recs: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    prose_chunks: List[torch.Tensor],
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    wrap: Optional[FOMPv3BackboneWrapper] = None,
    compiled: Optional[Dict[str, torch.Tensor]] = None,
    sign_flip: bool = False,
    max_eval: int = 64,
    loc_gen_len: int = 6,
) -> Dict[str, Any]:
    model.eval()
    sub = z_recs[: min(max_eval, len(z_recs))]
    r_win = r_em = g_win = g_em = l_tm = l_tt = l_ex = 0
    for i, r in enumerate(sub):
        tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
        to_ids = tokenizer.encode(r.target_true, add_special_tokens=False)
        tn_s = r.target_new.strip().lower()

        c_ids = _enc_prompt(tokenizer, r.cue, prepend_bos)
        if _score_logprob_any(model, c_ids, tn_ids, device, amp, wrap, compiled, sign_flip) > _score_logprob_any(model, c_ids, to_ids, device, amp, wrap, compiled, sign_flip):
            r_win += 1
        g_c = _greedy_gen_any(model, c_ids, max(1, len(tn_ids) + 1), device, amp, wrap, compiled, sign_flip)
        if tokenizer.decode(g_c, skip_special_tokens=True).strip().lower().startswith(tn_s):
            r_em += 1

        p_ids = _enc_prompt(tokenizer, r.paraphrase, prepend_bos)
        if _score_logprob_any(model, p_ids, tn_ids, device, amp, wrap, compiled, sign_flip) > _score_logprob_any(model, p_ids, to_ids, device, amp, wrap, compiled, sign_flip):
            g_win += 1
        g_p = _greedy_gen_any(model, p_ids, max(1, len(tn_ids) + 1), device, amp, wrap, compiled, sign_flip)
        if tokenizer.decode(g_p, skip_special_tokens=True).strip().lower().startswith(tn_s):
            g_em += 1

        l_ids = _enc_prompt(tokenizer, r.loc_prompt, prepend_bos)
        g_l = _greedy_gen_any(model, l_ids, loc_gen_len, device, amp, wrap, compiled, sign_flip)
        b_l = base_loc_preds[i]
        m_len = min(len(g_l), len(b_l))
        l_tm += sum(1 for a, b in zip(g_l[:m_len], b_l[:m_len]) if a == b)
        l_tt += max(1, m_len)
        if g_l[:m_len] == b_l[:m_len]:
            l_ex += 1
    n = max(1, len(sub))
    ppl = _eval_ppl_any(model, prose_chunks, device, amp, wrap, compiled)
    return {
        "n": len(sub),
        "reliability": {"prob_win_pct": round(100.0 * r_win / n, 2), "gen_em_pct": round(100.0 * r_em / n, 2)},
        "generality": {"prob_win_pct": round(100.0 * g_win / n, 2), "gen_em_pct": round(100.0 * g_em / n, 2)},
        "locality": {
            "token_match_vs_base_pct": round(100.0 * l_tm / max(1, l_tt), 2),
            "exact_seq_vs_base_pct": round(100.0 * l_ex / n, 2),
        },
        "general_prose_ppl": ppl,
    }


# ==============================================================================
# Official EasyEdit Baseline Adapters (GRACE, WISE, AlphaEdit, MEMIT)
# ==============================================================================

def _build_requests(recs: Sequence[Any], loc_pool: List[str]) -> List[Dict[str, str]]:
    reqs: List[Dict[str, str]] = []
    for i, r in enumerate(recs):
        # Ensure `subject` appears verbatim inside `prompt` (required by AlphaEdit / MEMIT)
        cue = r.cue
        subj = r.subject
        if subj not in cue:
            # Fallback to first capitalized span or full cue prefix if casing differs slightly
            low_idx = cue.lower().find(subj.lower())
            subj = cue[low_idx : low_idx + len(subj)] if low_idx >= 0 else cue.split()[0]
        reqs.append(
            {
                "prompt": cue,
                "subject": subj,
                "target_new": r.target_new.strip(),
                "loc_prompt": loc_pool[i % len(loc_pool)],
            }
        )
    return reqs


def _run_easyedit_grace(
    backbone: nn.Module,
    tokenizer: Any,
    model_id: str,
    easyedit_dir: Path,
    cf_m: List[EditRecord],
    mq_m: List[EditRecord],
    z_m: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    prose_ev: List[torch.Tensor],
    loc_pool: List[str],
    base_ppl: float,
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    max_eval: int,
    max_gen: int,
    loc_gen_len: int,
) -> Dict[str, Any]:
    from easyeditor.models.grace.GRACE import GRACE
    from easyeditor.models.grace.grace_hparams import GraceHyperParams
    from easyeditor.models.grace.utils import tokenize as grace_tokenize

    n_l = len(backbone.model.layers)
    hp = GraceHyperParams(
        alg_name="GRACE",
        model_name=model_id,
        device=0 if device.type == "cuda" else -1,
        inner_params=[f"model.layers[{max(0, n_l - 5)}].mlp.down_proj.weight"],
        edit_lr=1.0,
        n_iter=50,
        eps=1.0,
        dist_fn="euc",
        val_init="cold",
        val_train="sgd",
        val_reg="None",
        reg="early_stop",
        replacement="replace_last",
        eps_expand="coverage",
        num_pert=8,
        dropout=0.0,
    )
    hp.use_chat_template = False

    def _run_one(recs: Sequence[Any], eval_fn) -> Dict[str, Any]:
        tokenizer.padding_side = "right"
        editor = GRACE(model=backbone, config=hp, device=device)
        try:
            reqs = _build_requests(recs, loc_pool)
            sync(device)
            t0 = time.perf_counter()
            for req in reqs:
                toks = grace_tokenize(req, tokenizer=tokenizer, device=device, hparams=hp)
                editor.edit(config=hp, tokens=toks, edit_id=req["target_new"])
            sync(device)
            edit_sec = round(time.perf_counter() - t0, 2)
            # Switch GRACE codebook layer to inference mode at prompt-end position
            lyr = eval(f"backbone.{editor.layer}")
            lyr.training = False
            lyr.key_id = -1
            res = eval_fn()
            res["edit_time_sec"] = edit_sec
            res["n_edited_facts"] = len(reqs)
            if "general_prose_ppl" in res:
                res["ppl_drift_vs_base"] = round(res["general_prose_ppl"] - base_ppl, 5)
            return res
        finally:
            editor.reset_layer()
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
            for p in backbone.parameters():
                p.requires_grad = False
            torch.cuda.empty_cache()

    out: Dict[str, Any] = {}
    out["track1_counterfact"] = _run_one(
        cf_m, lambda: _eval_track1_cf(backbone, tokenizer, cf_m, prose_ev, device, amp, prepend_bos, max_eval=max_eval, max_gen=max_gen)
    )
    print("[GRACE Track1 CF]", json.dumps(out["track1_counterfact"]), flush=True)
    out["track2_mquake"] = _run_one(
        mq_m, lambda: _eval_track2_mquake(backbone, tokenizer, mq_m, device, amp, prepend_bos, max_eval=min(32, len(mq_m)))
    )
    print("[GRACE Track2 MQuAKE]", json.dumps(out["track2_mquake"]), flush=True)
    out["track3_zsre"] = _run_one(
        z_m, lambda: _eval_track3_zsre(backbone, tokenizer, z_m, base_loc_preds, prose_ev, device, amp, prepend_bos, max_eval=max_eval, loc_gen_len=loc_gen_len)
    )
    print("[GRACE Track3 ZsRE]", json.dumps(out["track3_zsre"]), flush=True)
    return out


def _run_easyedit_wise(
    backbone: nn.Module,
    tokenizer: Any,
    model_id: str,
    easyedit_dir: Path,
    cf_m: List[EditRecord],
    mq_m: List[EditRecord],
    z_m: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    prose_ev: List[torch.Tensor],
    loc_pool: List[str],
    base_ppl: float,
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    max_eval: int,
    max_gen: int,
    loc_gen_len: int,
) -> Dict[str, Any]:
    import easyeditor.models.wise.WISE as wise_mod
    from easyeditor.models.wise.WISE import WISE
    from easyeditor.models.wise.utils import get_context_templates as wise_get_ctx
    from easyeditor.models.wise.utils import tokenize as wise_tokenize
    from easyeditor.models.wise.wise_hparams import WISEHyperParams

    hp_path = easyedit_dir / "hparams/WISE/llama3-8b.yaml"
    hp = WISEHyperParams.from_hparams(str(hp_path))
    hp.model_name = model_id
    hp.device = 0 if device.type == "cuda" else -1
    hp.use_chat_template = False
    n_l = len(backbone.model.layers)
    hp.inner_params = [f"model.layers[{max(0, n_l - 3)}].mlp.down_proj.weight"]

    ctx_templates = wise_get_ctx(backbone, tokenizer, length_params=[[5, 5], [10, 5]], device=device)

    def _run_one(recs: Sequence[Any], eval_fn) -> Dict[str, Any]:
        tokenizer.padding_side = "right"
        wise_mod.edit_history.clear()
        wise_mod.merge_group_edit_history.clear()
        editor = WISE(model=backbone, config=hp, device=device)
        try:
            reqs = _build_requests(recs, loc_pool)
            sync(device)
            t0 = time.perf_counter()
            for req in reqs:
                toks, act_mask, deact_mask = wise_tokenize([req], tokenizer=tokenizer, device=device, context_templates=ctx_templates, hparams=hp)
                editor.edit(config=hp, tokens=toks, act_mask=act_mask, deact_mask=deact_mask)
            sync(device)
            edit_sec = round(time.perf_counter() - t0, 2)
            lyr = editor.get_adapter_layer()
            lyr.training = False
            lyr.editing = False
            res = eval_fn()
            res["edit_time_sec"] = edit_sec
            res["n_edited_facts"] = len(reqs)
            if "general_prose_ppl" in res:
                res["ppl_drift_vs_base"] = round(res["general_prose_ppl"] - base_ppl, 5)
            return res
        finally:
            editor.reset_layer()
            wise_mod.edit_history.clear()
            wise_mod.merge_group_edit_history.clear()
            for p in backbone.parameters():
                p.requires_grad = False
            torch.cuda.empty_cache()

    out: Dict[str, Any] = {}
    out["track1_counterfact"] = _run_one(
        cf_m, lambda: _eval_track1_cf(backbone, tokenizer, cf_m, prose_ev, device, amp, prepend_bos, max_eval=max_eval, max_gen=max_gen)
    )
    print("[WISE Track1 CF]", json.dumps(out["track1_counterfact"]), flush=True)
    out["track2_mquake"] = _run_one(
        mq_m, lambda: _eval_track2_mquake(backbone, tokenizer, mq_m, device, amp, prepend_bos, max_eval=min(32, len(mq_m)))
    )
    print("[WISE Track2 MQuAKE]", json.dumps(out["track2_mquake"]), flush=True)
    out["track3_zsre"] = _run_one(
        z_m, lambda: _eval_track3_zsre(backbone, tokenizer, z_m, base_loc_preds, prose_ev, device, amp, prepend_bos, max_eval=max_eval, loc_gen_len=loc_gen_len)
    )
    print("[WISE Track3 ZsRE]", json.dumps(out["track3_zsre"]), flush=True)
    return out


def _ensure_alphaedit_nullspace_gpu(
    backbone: nn.Module,
    tokenizer: Any,
    hp: Any,
    p_path: Path,
    device: torch.device,
) -> float:
    """Compute exact AlphaEdit null-space projector `P` on GPU via `eigh` (~500x faster than EasyEdit CPU SVD)."""
    if p_path.exists():
        print(f"[AlphaEdit] Reusing cached null-space projector -> {p_path}", flush=True)
        return 0.0
    from easyeditor.models.alphaedit.AlphaEdit_main import get_cov
    from easyeditor.util import nethook

    t0 = time.perf_counter()
    w_out = nethook.get_parameter(backbone, f"{hp.rewrite_module_tmp.format(hp.layers[-1])}.weight")
    d_in = int(w_out.shape[1])
    del w_out
    P_all = torch.zeros((len(hp.layers), d_in, d_in), dtype=torch.float32, device="cpu")
    for i, layer in enumerate(hp.layers):
        print(f"[AlphaEdit] Computing covariance + GPU null-space eigh for layer {layer} ({hp.mom2_dataset}, n={hp.mom2_n_samples})...", flush=True)
        cov = get_cov(
            backbone, tokenizer, hp.rewrite_module_tmp.format(layer),
            hp.mom2_dataset, hp.mom2_n_samples, hp.mom2_dtype,
            force_recompute=False, hparams=hp,
        )
        # Symmetric PSD covariance -> `eigh` eigenvalues/vectors are mathematically identical to `svd` and 500x faster on A100!
        cov_f = cov.to(device=device, dtype=torch.float32)
        cov_f = 0.5 * (cov_f + cov_f.T)
        evals, evecs = torch.linalg.eigh(cov_f)
        small_idx = (evals < float(hp.nullspace_threshold)).nonzero(as_tuple=True)[0]
        print(f"  layer {layer}: null-space rank = {len(small_idx)} / {d_in}", flush=True)
        U_null = evecs[:, small_idx]
        P_all[i] = (U_null @ U_null.T).cpu()
        del cov, cov_f, evals, evecs, U_null
        torch.cuda.empty_cache()
    p_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = p_path.with_suffix(".pt.tmp")
    torch.save(P_all, tmp)
    os.replace(tmp, p_path)
    sec = round(time.perf_counter() - t0, 2)
    print(f"[AlphaEdit] Saved GPU-computed null-space projector -> {p_path} in {sec}s.", flush=True)
    return sec


def _run_easyedit_alphaedit_or_memit(
    alg: str,
    backbone: nn.Module,
    tokenizer: Any,
    model_id: str,
    easyedit_dir: Path,
    stats_dir: Path,
    p_path: Path,
    cov_samples: int,
    cf_m: List[EditRecord],
    mq_m: List[EditRecord],
    z_m: List[ZsRERecord],
    base_loc_preds: List[List[int]],
    prose_ev: List[torch.Tensor],
    loc_pool: List[str],
    base_ppl: float,
    device: torch.device,
    amp: bool,
    prepend_bos: bool,
    max_eval: int,
    max_gen: int,
    loc_gen_len: int,
) -> Dict[str, Any]:
    from easyeditor.util import nethook

    tokenizer.padding_side = "right"
    if alg == "alphaedit":
        import easyeditor.models.alphaedit.AlphaEdit_main as ae_mod
        from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams

        hp = AlphaEditHyperParams.from_hparams(str(easyedit_dir / "hparams/AlphaEdit/llama3-8b.yaml"))
        hp.model_name = model_id
        hp.device = 0 if device.type == "cuda" else -1
        hp.stats_dir = str(stats_dir)
        hp.P_loc = str(p_path)
        hp.mom2_dataset = "wikitext"
        hp.mom2_n_samples = int(cov_samples)
        prep_sec = _ensure_alphaedit_nullspace_gpu(backbone, tokenizer, hp, p_path, device)
        ae_mod.P_loaded = False
        ae_mod.cache_c_new = False
        apply_fn = lambda reqs: ae_mod.apply_AlphaEdit_to_model(
            backbone, tokenizer, reqs, hp, copy=False, return_orig_weights=True, reset_cache=True
        )
    else:
        from easyeditor.models.memit.memit_hparams import MEMITHyperParams
        from easyeditor.models.memit.memit_main import apply_memit_to_model

        hp = MEMITHyperParams(
            alg_name="MEMIT",
            model_name=model_id,
            stats_dir=str(stats_dir),
            device=0 if device.type == "cuda" else -1,
            layers=[4, 5, 6, 7, 8],
            clamp_norm_factor=0.75,
            layer_selection="all",
            fact_token="subject_last",
            v_num_grad_steps=25,
            v_lr=0.1,
            v_loss_layer=len(backbone.model.layers) - 1,
            v_weight_decay=0.5,
            kl_factor=0.0625,
            mom2_adjustment=True,
            mom2_update_weight=15000,
            rewrite_module_tmp="model.layers.{}.mlp.down_proj",
            layer_module_tmp="model.layers.{}",
            mlp_module_tmp="model.layers.{}.mlp",
            attn_module_tmp="model.layers.{}.self_attn",
            ln_f_module="model.norm",
            lm_head_module="lm_head",
            mom2_dataset="wikitext",
            mom2_n_samples=int(cov_samples),
            mom2_dtype="float32",
        )
        prep_sec = 0.0
        apply_fn = lambda reqs: apply_memit_to_model(
            backbone, tokenizer, reqs, hp, copy=False, return_orig_weights=True
        )

    def _run_one(recs: Sequence[Any], eval_fn) -> Dict[str, Any]:
        tokenizer.padding_side = "right"
        reqs = _build_requests(recs, loc_pool)
        weights_copy: Dict[str, torch.Tensor] = {}
        try:
            sync(device)
            t0 = time.perf_counter()
            _, weights_copy = apply_fn(reqs)
            sync(device)
            edit_sec = round(time.perf_counter() - t0, 2)
            res = eval_fn()
            res["edit_time_sec"] = edit_sec
            res["n_edited_facts"] = len(reqs)
            if "general_prose_ppl" in res:
                res["ppl_drift_vs_base"] = round(res["general_prose_ppl"] - base_ppl, 5)
            return res
        finally:
            with torch.no_grad():
                for w_name, orig_w in weights_copy.items():
                    p = nethook.get_parameter(backbone, w_name)
                    p.copy_(orig_w.to(device=p.device, dtype=p.dtype))
            for p in backbone.parameters():
                p.requires_grad = False
            gc.collect()
            torch.cuda.empty_cache()

    out: Dict[str, Any] = {"prep_nullspace_sec": prep_sec}
    out["track1_counterfact"] = _run_one(
        cf_m, lambda: _eval_track1_cf(backbone, tokenizer, cf_m, prose_ev, device, amp, prepend_bos, max_eval=max_eval, max_gen=max_gen)
    )
    print(f"[{alg.upper()} Track1 CF]", json.dumps(out["track1_counterfact"]), flush=True)
    out["track2_mquake"] = _run_one(
        mq_m, lambda: _eval_track2_mquake(backbone, tokenizer, mq_m, device, amp, prepend_bos, max_eval=min(32, len(mq_m)))
    )
    print(f"[{alg.upper()} Track2 MQuAKE]", json.dumps(out["track2_mquake"]), flush=True)
    out["track3_zsre"] = _run_one(
        z_m, lambda: _eval_track3_zsre(backbone, tokenizer, z_m, base_loc_preds, prose_ev, device, amp, prepend_bos, max_eval=max_eval, loc_gen_len=loc_gen_len)
    )
    print(f"[{alg.upper()} Track3 ZsRE]", json.dumps(out["track3_zsre"]), flush=True)
    return out


# ==============================================================================
# Markdown Report Generator (Publication-Ready Tables!)
# ==============================================================================

def _write_markdown_report(summary: Dict[str, Any], out_md: Path) -> None:
    arms = summary.get("arms", {})
    m_edit = summary.get("m_edit", 100)
    lines = [
        f"# Stage 2 SOTA Showdown on `{summary.get('backbone')}` (`{summary.get('device')}`)",
        "",
        f"- **Protocol**: 100% identical backbone, identical BOS-consistent prompt tokenization, strictly 0% train/test entity overlap, **0% ZsRE training data for FOMP-v3.5**.",
        f"- **Base Held-Out WikiText-2 PPL**: `{summary.get('base_prose_ppl')}`",
        "",
        f"## Table 1: CounterFact (`M = {m_edit}` Concurrent Edits)",
        "",
        "| Method | Edit Time | Efficacy (Prob / EM) | Paraphrase (Prob / EM) | Neighborhood Retain | WikiText-2 PPL Drift | Query ms |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    order = ["base", "grace", "wise", "memit", "alphaedit", "fomp_v35", "fomp_v36"]
    for k in order:
        if k not in arms or "error" in arms[k]:
            continue
        t1 = arms[k].get("track1_counterfact") or {}
        if not t1:
            continue
        ef = t1["efficacy"]
        pa = t1["paraphrase"]
        ne = t1["neighborhood_retain"]
        et = f"{t1.get('edit_time_sec', 0.0)}s" if k != "base" else "—"
        dr = f"{t1.get('ppl_drift_vs_base', 0.0):+.4f}" if k != "base" else "0.0000"
        lines.append(
            f"| **{k.upper()}** | {et} | {ef['prob_win_pct']}% / {ef['gen_em_pct']}% | "
            f"{pa['prob_win_pct']}% / {pa['gen_em_pct']}% | {ne['prob_win_pct']}% | {dr} | {t1['ms_per_query']} ms |"
        )

    lines += [
        "",
        "## Table 2: MQuAKE-Remastered Multi-Hop Reasoning",
        "",
        "| Method | Edit Time | Single-Hop Efficacy (Prob / EM) | **2-Hop Multi-Hop (Prob / EM)** |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for k in order:
        if k not in arms or "error" in arms[k]:
            continue
        t2 = arms[k].get("track2_mquake") or {}
        if not t2:
            continue
        sh = t2["single_hop_efficacy"]
        mh = t2["mquake_2hop"]
        et = f"{t2.get('edit_time_sec', 0.0)}s" if k != "base" else "—"
        lines.append(f"| **{k.upper()}** | {et} | {sh['prob_win_pct']}% / {sh['gen_em_pct']}% | **{mh['prob_win_pct']}% / {mh['gen_em_pct']}%** |")

    lines += [
        "",
        f"## Table 3: Zero-Shot ZsRE (`M = {m_edit}` Edits — 0% ZsRE Training Data for FOMP-v3.5!)",
        "",
        "| Method | Edit Time | Reliability (Prob / EM) | Generality (Prob / EM) | **Locality (Tok-Match / Exact)** | WikiText-2 PPL Drift |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k in order:
        if k not in arms or "error" in arms[k]:
            continue
        t3 = arms[k].get("track3_zsre") or {}
        if not t3:
            continue
        re_ = t3["reliability"]
        ge_ = t3["generality"]
        lo_ = t3["locality"]
        et = f"{t3.get('edit_time_sec', 0.0)}s" if k != "base" else "—"
        dr = f"{t3.get('ppl_drift_vs_base', 0.0):+.4f}" if k != "base" else "0.0000"
        lines.append(
            f"| **{k.upper()}** | {et} | {re_['prob_win_pct']}% / {re_['gen_em_pct']}% | "
            f"{ge_['prob_win_pct']}% / {ge_['gen_em_pct']}% | **{lo_['token_match_vs_base_pct']}% / {lo_['exact_seq_vs_base_pct']}%** | {dr} |"
        )

    f_arm_name = "FOMP-v3.6" if "fomp_v36" in arms else "FOMP-v3.5"
    f_arm = arms.get("fomp_v36") or arms.get("fomp_v35") or {}
    if "track1_scaling_1000" in f_arm:
        lines += [
            "",
            f"## Table 4: {f_arm_name} High-Concurrency Scaling Wall (`M = 500, 1000`)",
            "",
            "| Concurrency `M` | Compile Time | Efficacy (Prob / EM) | Paraphrase (Prob / EM) | Neighborhood Retain | PPL Drift |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for mk, mv in f_arm["track1_scaling_1000"].items():
            lines.append(
                f"| `{mk}` | {mv['edit_time_sec']}s | {mv['efficacy']['prob_win_pct']}% / {mv['efficacy']['gen_em_pct']}% | "
                f"{mv['paraphrase']['prob_win_pct']}% / {mv['paraphrase']['gen_em_pct']}% | "
                f"{mv['neighborhood_retain']['prob_win_pct']}% | {mv['ppl_drift_vs_base']:+.4f} |"
            )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ==============================================================================
# Main Orchestrator (Resumable Arm-by-Arm)
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--easyedit-dir", type=Path, default=Path("/content/EasyEdit"))
    parser.add_argument("--arms", type=str, default="base,fomp_v36,grace,wise,alphaedit,memit")
    parser.add_argument("--force-arms", type=str, default="", help="Comma-separated arms to re-run even if already saved")
    parser.add_argument("--fomp-train-steps", type=int, default=2000)
    parser.add_argument("--m-edit", type=int, default=100)
    parser.add_argument("--alphaedit-cov-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--deadline", type=float, default=time.time() + 5400)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "llama3_8b_sota_summary.json"
    out_md = args.out / "llama3_8b_sota_report.md"

    req_arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    force_arms = {a.strip().lower() for a in args.force_arms.split(",") if a.strip()}
    if not args.smoke and args.easyedit_dir.exists() and str(args.easyedit_dir) not in sys.path:
        sys.path.insert(0, str(args.easyedit_dir))

    if args.smoke:
        tokenizer: Any = _SmokeTokenizer()
        backbone: nn.Module = _SmokeHFModel(vocab=128, d=64, n_layers=8).to(device)
        fomp_cfg = FOMPv3Config(d_model=64, read_layer=4, inject_layers=(4, 6), key_dim=32, ent_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _make_smoke_datasets()
        s1_pool = cf_te * 2
        z_recs = _make_smoke_zsre(16)
        m_edit, m_mq, max_eval, max_gen, loc_gen_len = 8, 8, 8, 4, 4
        prepend_bos = False
        scale_ms = [16]
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
        rl = max(2, int(round(n_l * 0.50)))  # L16 for 32L Llama-3-8B
        inj = tuple(sorted(set([rl, min(n_l - 1, rl + 3), min(n_l - 1, rl + 7), max(rl, n_l - 5), max(rl, n_l - 1)])))
        fomp_cfg = FOMPv3Config(
            d_model=d_m, read_layer=rl, inject_layers=inj,
            key_dim=(512 if d_m >= 3072 else 384),
            ent_dim=(384 if d_m >= 3072 else 256),
            val_dim=512, max_tgt_tokens=8, top_k=2, rel_hidden=128,
        )
        cf_tr, cf_te, mq_tr, mq_te, prose_txts = _load_real_datasets(tokenizer, max_tgt_tokens=8, seed=args.seed)
        train_subjs = {r.subject.lower() for r in cf_tr + mq_tr}
        s1_pool = _load_s1_1024_unseen(tokenizer, 8, cf_te, train_subjs, target_n=1000)
        z_recs = _load_zsre_test(tokenizer, 8, train_subjs, target_n=max(200, args.m_edit))
        m_edit, m_mq, max_eval, max_gen, loc_gen_len = int(args.m_edit), min(64, len(mq_te)), 64, 32, 6
        prepend_bos = getattr(tokenizer, "bos_token_id", None) is not None
        scale_ms = [500, 1000]

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
    loc_pool = [t[:220].strip() for t in prose_txts[:split_p]] or ["The sun rises in the east every morning."]

    cf_m = s1_pool[:m_edit]
    mq_m = mq_te[:m_mq]
    z_m = z_recs[:m_edit]

    summary: Dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    summary.update(
        {
            "device": torch.cuda.get_device_name(0) if amp else "cpu",
            "backbone": "smoke_8L" if args.smoke else args.model_id,
            "m_edit": m_edit,
            "m_mquake": m_mq,
            "prepend_bos": prepend_bos,
            "fomp_v35_config": fomp_cfg.__dict__,
        }
    )
    summary.setdefault("arms", {})

    # Precompute unedited Base greedy predictions on ZsRE locality prompts (official WISE/EasyEdit Locality reference)
    base_ppl = _eval_ppl_any(backbone, prose_ev_chunks, device, amp)
    summary["base_prose_ppl"] = base_ppl
    base_loc_preds: List[List[int]] = []
    for r in z_m[: min(max_eval, len(z_m))]:
        l_ids = _enc_prompt(tokenizer, r.loc_prompt, prepend_bos)
        base_loc_preds.append(_greedy_gen_any(backbone, l_ids, loc_gen_len, device, amp))

    # --- Arm 1: Base ---
    if "base" in req_arms and ("base" not in summary["arms"] or "base" in force_arms):
        print("\n=== [Arm 1/6] Base (Unedited Backbone) ===", flush=True)
        b_res = {
            "track1_counterfact": _eval_track1_cf(backbone, tokenizer, cf_m, prose_ev_chunks, device, amp, prepend_bos, max_eval=max_eval, max_gen=max_gen),
            "track2_mquake": _eval_track2_mquake(backbone, tokenizer, mq_m, device, amp, prepend_bos, max_eval=min(32, len(mq_m))),
            "track3_zsre": _eval_track3_zsre(backbone, tokenizer, z_m, base_loc_preds, prose_ev_chunks, device, amp, prepend_bos, max_eval=max_eval, loc_gen_len=loc_gen_len),
        }
        summary["arms"]["base"] = b_res
        print("[Base]", json.dumps(b_res), flush=True)
        save_json(out_json, summary)
        _write_markdown_report(summary, out_md)

    # --- Arm 2: FOMP-v3.6 / v3.5 (Ours) ---
    fomp_arm = "fomp_v36" if "fomp_v36" in req_arms else ("fomp_v35" if "fomp_v35" in req_arms else "")
    if fomp_arm and (fomp_arm not in summary["arms"] or fomp_arm in force_arms):
        print(f"\n=== [Arm 2/6] {fomp_arm.upper()} (Ours, Open-Vocab RMS-Normed Dual-Lock Single-Pass) ===", flush=True)
        torch.manual_seed(args.seed)
        fomp = FOMPv3Module(fomp_cfg).to(device)
        wrap = FOMPv3BackboneWrapper(backbone, fomp, freeze_backbone=True)
        fomp_ckpt = args.out / f"{fomp_arm}_llama3_8b.pt"
        if fomp_arm in force_arms and fomp_ckpt.exists():
            fomp_ckpt.unlink()
        tr_meta = _train_fomp_v3(
            wrap, tokenizer, cf_tr, mq_tr, prose_tr_chunks, device, amp, args.smoke,
            steps=(8 if args.smoke else args.fomp_train_steps),
            ckpt_path=fomp_ckpt,
            deadline=args.deadline,
            prepend_bos=prepend_bos,
        )

        def _compile_list(recs_list: Sequence[Any], g_ext: float = 1.0) -> Tuple[Dict[str, torch.Tensor], float]:
            sync(device)
            tc = time.perf_counter()
            cm = wrap.compile_token_episodes(
                [_enc_prompt(tokenizer, r.cue, prepend_bos) for r in recs_list],
                [tokenizer.encode(r.target_new, add_special_tokens=False)[: fomp_cfg.max_tgt_tokens] for r in recs_list],
                pad_id=pad_id, device=device, g_ext=g_ext,
            )
            sync(device)
            return cm, round(time.perf_counter() - tc, 4)

        def _log_diag(tag: str, recs_list: Sequence[Any], cm: Dict[str, torch.Tensor], n_show: int = 5) -> None:
            with torch.no_grad():
                for r in recs_list[:n_show]:
                    p_ids = _enc_prompt(tokenizer, r.cue, prepend_bos)
                    tn_ids = tokenizer.encode(r.target_new, add_special_tokens=False)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        rt = wrap.forward_logits(torch.tensor([p_ids], device=device), compiled=cm)["routing"]
                    g_ids = _greedy_gen_any(backbone, p_ids, max(1, len(tn_ids) + 1), device, amp, wrap=wrap, compiled=cm)
                    g_str = tokenizer.decode(g_ids, skip_special_tokens=True)
                    mg = float(rt["max_gate"][0, -1].item())
                    dg = float(rt["d_gate"][0, -1].item())
                    print(
                        f"  [{fomp_arm.upper()} Diag {tag}] want={r.target_new!r} ({len(tn_ids)}tok) | "
                        f"gen={g_str!r} | gate={mg:.3f} d_gate={dg:.3f}",
                        flush=True,
                    )

        f_res: Dict[str, Any] = {"train_meta": tr_meta}
        # Track 1: CounterFact M=m_edit
        c1, t1_s = _compile_list(cf_m, 1.0)
        e1 = _eval_track1_cf(backbone, tokenizer, cf_m, prose_ev_chunks, device, amp, prepend_bos, wrap=wrap, compiled=c1, max_eval=max_eval, max_gen=max_gen)
        e1["edit_time_sec"] = t1_s
        e1["n_edited_facts"] = len(cf_m)
        e1["ppl_drift_vs_base"] = round(e1["general_prose_ppl"] - base_ppl, 5)
        c0, _ = _compile_list(cf_m[:8], 0.0)
        e1["g_ext0_ppl_drift"] = round(abs(_eval_ppl_any(backbone, prose_ev_chunks, device, amp, wrap=wrap, compiled=c0) - base_ppl), 6)
        sf1 = _eval_track1_cf(backbone, tokenizer, cf_m, prose_ev_chunks, device, amp, prepend_bos, wrap=wrap, compiled=c1, sign_flip=True, max_eval=min(32, max_eval), max_gen=min(16, max_gen))
        e1["sign_flip"] = {"efficacy": sf1["efficacy"], "paraphrase": sf1["paraphrase"]}
        f_res["track1_counterfact"] = e1
        print(f"[{fomp_arm.upper()} Track1 CF]", json.dumps(e1), flush=True)

        # Track 1 Scaling Wall: M=500, 1000
        sc_res: Dict[str, Any] = {}
        for m_sc in scale_ms:
            sub_sc = s1_pool[: min(m_sc, len(s1_pool))]
            cm_s, ts_s = _compile_list(sub_sc, 1.0)
            es = _eval_track1_cf(backbone, tokenizer, cf_m, prose_ev_chunks, device, amp, prepend_bos, wrap=wrap, compiled=cm_s, max_eval=max_eval, max_gen=max_gen)
            es["edit_time_sec"] = ts_s
            es["n_edited_facts"] = len(sub_sc)
            es["ppl_drift_vs_base"] = round(es["general_prose_ppl"] - base_ppl, 5)
            sc_res[f"M{m_sc}"] = es
            print(f"[{fomp_arm.upper()} Scaling M={m_sc}]", json.dumps(es), flush=True)
        f_res["track1_scaling_1000"] = sc_res

        # Track 2: MQuAKE Multi-Hop
        c2, t2_s = _compile_list(mq_m, 1.0)
        _log_diag("MQuAKE", mq_m, c2)
        e2 = _eval_track2_mquake(backbone, tokenizer, mq_m, device, amp, prepend_bos, wrap=wrap, compiled=c2, max_eval=min(32, len(mq_m)))
        e2["edit_time_sec"] = t2_s
        e2["n_edited_facts"] = len(mq_m)
        sf2 = _eval_track2_mquake(backbone, tokenizer, mq_m, device, amp, prepend_bos, wrap=wrap, compiled=c2, sign_flip=True, max_eval=min(16, len(mq_m)))
        e2["sign_flip"] = sf2
        f_res["track2_mquake"] = e2
        print(f"[{fomp_arm.upper()} Track2 MQuAKE]", json.dumps(e2), flush=True)

        # Track 3: Zero-Shot ZsRE
        c3, t3_s = _compile_list(z_m, 1.0)
        _log_diag("ZsRE", z_m, c3)
        e3 = _eval_track3_zsre(backbone, tokenizer, z_m, base_loc_preds, prose_ev_chunks, device, amp, prepend_bos, wrap=wrap, compiled=c3, max_eval=max_eval, loc_gen_len=loc_gen_len)
        e3["edit_time_sec"] = t3_s
        e3["n_edited_facts"] = len(z_m)
        e3["ppl_drift_vs_base"] = round(e3["general_prose_ppl"] - base_ppl, 5)
        sf3 = _eval_track3_zsre(backbone, tokenizer, z_m, base_loc_preds, prose_ev_chunks, device, amp, prepend_bos, wrap=wrap, compiled=c3, sign_flip=True, max_eval=min(16, max_eval), loc_gen_len=loc_gen_len)
        e3["sign_flip"] = {"reliability": sf3["reliability"], "generality": sf3["generality"]}
        f_res["track3_zsre"] = e3
        print(f"[{fomp_arm.upper()} Track3 ZsRE]", json.dumps(e3), flush=True)

        summary["arms"][fomp_arm] = f_res
        del wrap, fomp, c1, c2, c3, c0
        gc.collect()
        torch.cuda.empty_cache()
        save_json(out_json, summary)
        _write_markdown_report(summary, out_md)

    # --- Arms 3..6: Official EasyEdit Baselines (GRACE, WISE, AlphaEdit, MEMIT) ---
    if not args.smoke:
        stats_dir = args.out / "easyedit_stats"
        p_path = args.out / "llama3_8b_null_space_P.pt"
        for arm_name in ("grace", "wise", "alphaedit", "memit"):
            if arm_name not in req_arms or (arm_name in summary["arms"] and arm_name not in force_arms):
                continue
            if time.time() > args.deadline - 180:
                print(f"[Deadline] Stopping before {arm_name} to save cleanly.", flush=True)
                break
            print(f"\n=== [Official EasyEdit Arm] {arm_name.upper()} ===", flush=True)
            try:
                if arm_name == "grace":
                    res = _run_easyedit_grace(
                        backbone, tokenizer, args.model_id, args.easyedit_dir, cf_m, mq_m, z_m,
                        base_loc_preds, prose_ev_chunks, loc_pool, base_ppl, device, amp, prepend_bos, max_eval, max_gen, loc_gen_len,
                    )
                elif arm_name == "wise":
                    res = _run_easyedit_wise(
                        backbone, tokenizer, args.model_id, args.easyedit_dir, cf_m, mq_m, z_m,
                        base_loc_preds, prose_ev_chunks, loc_pool, base_ppl, device, amp, prepend_bos, max_eval, max_gen, loc_gen_len,
                    )
                else:
                    res = _run_easyedit_alphaedit_or_memit(
                        arm_name, backbone, tokenizer, args.model_id, args.easyedit_dir, stats_dir, p_path, args.alphaedit_cov_samples,
                        cf_m, mq_m, z_m, base_loc_preds, prose_ev_chunks, loc_pool, base_ppl, device, amp, prepend_bos, max_eval, max_gen, loc_gen_len,
                    )
                summary["arms"][arm_name] = res
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[{arm_name.upper()} ERROR] {exc}\n{tb}", flush=True)
                summary["arms"][arm_name] = {"error": f"{type(exc).__name__}: {exc}", "traceback": tb[-1500:]}
            save_json(out_json, summary)
            _write_markdown_report(summary, out_md)
            gc.collect()
            torch.cuda.empty_cache()

    _write_markdown_report(summary, out_md)
    print("\n=== STAGE 2 LLAMA-3-8B SOTA SHOWDOWN COMPLETE ->", out_md, "===", flush=True)


if __name__ == "__main__":
    main()
