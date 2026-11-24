"""PULSE-v7 Stage 3b-v2: Fair Zero-Strawman Joint Entity-Binding Adaptation + 8x Extrapolation (L=512 -> 2048 -> 4096).

Why Stage 3b-v2 exists (documented honestly from Stage 3b):
  - Frozen 2.1M-token checkpoints (FineWeb PPL 467 / 536) have not yet acquired entity-ownership
    tracking circuits (both PULSE=0.477 and TF=0.526 sat at random 0.50).
  - Stage 3b-v2 applies IDENTICAL lightweight joint 3-way entity-binding adaptation to BOTH
    PULSE-v7B-30M and TF-30M (reusing Drive checkpoints from Stage 2, zero duplicate pretraining!):
      * Exact same <50K TransitionConditionedVerifier (37,795 params, d_in=752 -> 48 -> 3)
      * Exact same anti-shortcut 3-way trajectories (60+ disjoint train names, 30 train objs;
        strictly disjoint val & test names/objs; FineWeb-TRAIN vs FineWeb-VAL prose;
        50% post-transfer stale-owner/intruder contradictions; 50% Class-0 distractor transfers;
        randomized L_train in {192..512} and randomized event depths; held-out unseen template!)
      * Exact same AdamW optimizer, schedule, batch order, and 3-term loss:
            L = L_3way_verifier + 0.5 * L_owner_next_token + 1.0 * L_fineweb_replay
      * Zero-shot length extrapolation evaluation at L=512, L=512 (unseen template), L=2048 (4x),
        and L=4096 (8x!), plus GDN-signals-only ablation and Verifier-Gated DeCoRe at L=512 & L=4096.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import _cosine_lr_multiplier
from pulse.pulse_v7 import PULSEv7Model, patch_fla_sm89_shared_memory
from pulse.pulse_v7_plasticity import TransitionConditionedVerifier, decore_contrastive_logits
from pulse.scalable_transformer import ScalableTransformerLM

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import CachedFineWebReader, eval_lm_ppl, save_json, sync  # noqa: E402
from v7_stage2 import eval_lm_ppl_loops  # noqa: E402
from v7_stage3 import _TextEncoder, _load_or_init_30m_models, compute_auroc  # noqa: E402


# Generous pool of common English first names (~140); filtered dynamically to distinct single tokens.
_CANDIDATE_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Jack", "Karen",
    "Liam", "Maria", "Noah", "Paul", "Rachel", "Samuel", "Tina", "Victor", "Wendy", "Arthur",
    "Derek", "Hannah", "Isaac", "Julia", "Kevin", "Laura", "Martin", "Oscar", "Simon", "Adam",
    "Alex", "Amy", "Andrew", "Anna", "Ben", "Beth", "Brian", "Bruce", "Carl", "Carol",
    "Chris", "Claire", "Colin", "Craig", "Dan", "Daniel", "Dave", "Dean", "Diana", "Donald",
    "Donna", "Drew", "Dylan", "Eddie", "Edward", "Ellen", "Emily", "Eric", "Evan", "Gary",
    "George", "Glen", "Grant", "Greg", "Harry", "Helen", "Ian", "James", "Jane", "Jason",
    "Jean", "Jeff", "Jenny", "Jerry", "Jesse", "Jim", "Joan", "Joe", "John", "Jordan",
    "Joseph", "Josh", "Joy", "Judy", "Justin", "Kate", "Keith", "Kelly", "Ken", "Kim",
    "Kyle", "Larry", "Leo", "Linda", "Lisa", "Luke", "Mark", "Mary", "Matt", "Max",
    "Megan", "Michael", "Mike", "Nancy", "Nathan", "Neil", "Nick", "Nina", "Owen", "Pat",
    "Patrick", "Peter", "Phil", "Ray", "Richard", "Rick", "Rob", "Robert", "Robin", "Roger",
    "Ron", "Rose", "Roy", "Ryan", "Sam", "Sara", "Sarah", "Scott", "Sean", "Seth",
    "Shane", "Sharon", "Steve", "Susan", "Ted", "Terry", "Thomas", "Tim", "Todd", "Tom",
    "Tony", "Tyler", "Wayne", "Will", "William",
]

# Strictly word-disjoint object sets (zero shared adjectives OR nouns across Train / Val / Test!)
_TRAIN_OBJS = [f"{adj} {noun}" for adj in ("blue", "silver", "red", "brass", "jade", "iron") for noun in ("gem", "key", "coin", "lamp", "ring")]
_VAL_OBJS = [f"{adj} {noun}" for adj in ("gold", "oak") for noun in ("seal", "flute", "box", "crown")]
_TEST_OBJS = [f"{adj} {noun}" for adj in ("amber", "copper", "ivory") for noun in ("stone", "bell", "card", "orb")]

# 4 varied surface templates per event; index 3 is strictly HELD OUT for unseen-template test!
_ACQ_TPLS = [
    " {a} received the {obj} inside the hall.",
    " {a} collected the {obj} near the gate.",
    " {a} picked up the {obj} from the desk.",
    " {a} claimed the {obj} beside the tower.",
]
_TRF_TPLS = [
    " Later, {a} handed the {obj} directly to {b}.",
    " Afterward, {a} passed the {obj} over to {b}.",
    " Soon, {a} transferred the {obj} to {b}.",
    " Before dusk, {a} delivered the {obj} to {b}.",
]
# {obj} ALWAYS precedes {actor} so predicting true_owner at the comma is 100% causally well-defined!
_FIN_TPLS = [
    (" Later, to unlock the chest using the {obj},", " stepped forward."),
    (" Soon after, holding the {obj} firmly,", " entered the room."),
    (" Eventually, carrying the {obj} across the room,", " arrived."),
    (" At last, lifting the {obj} into the light,", " spoke."),
]


def _split_disjoint_names(enc: _TextEncoder, seed: int = 707) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    pool = enc.single_token_pool(_CANDIDATE_NAMES)
    if len(pool) < 36:
        raise RuntimeError(f"Need >= 36 single-token names, got {len(pool)}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool)).tolist()
    shuf = [pool[i] for i in idx]
    n_tr = max(18, int(len(shuf) * 0.60))
    n_va = max(6, int(len(shuf) * 0.15))
    tr_names = shuf[:n_tr]
    va_names = shuf[n_tr : n_tr + n_va]
    te_names = shuf[n_tr + n_va :]
    assert len(te_names) >= 12
    return tr_names, va_names, te_names


def _sample_one_trajectory(
    enc: _TextEncoder,
    d_arr: np.ndarray,
    name_pool: List[Tuple[str, int]],
    objs: List[str],
    cls: int,
    target_len: int,
    rng: np.random.Generator,
    tpl_indices: Tuple[int, ...],
) -> Dict[str, Any]:
    """Generate one anti-shortcut 3-way Entity-Binding trajectory of exact length `target_len`.

      Class 0 (Consistent):
        A acquires obj (+ 50% irrelevant D transfers obj2 to E) -> A uses obj.
      Class 1 (Legal State Transition):
        A acquires obj -> A transfers obj to B (+ 35% irrelevant D acquires obj2 after) -> B uses obj.
      Class 2 (Illegal Fluent Entity Swap):
        50% Sub-case 2a (Unbacked intruder): A acquires obj (+ 50% irrelevant D->E obj2) -> C uses obj.
        50% Sub-case 2b (Post-transfer contradiction): A acquires obj -> A transfers obj to B -> A (stale, 60%) or C (40%) uses obj!
    """
    picked = rng.choice(len(name_pool), size=5, replace=False).tolist()
    (a_nm, a_id), (b_nm, b_id), (c_nm, c_id), (d_nm, _), (e_nm, _) = [name_pool[i] for i in picked]
    o_idx = rng.choice(len(objs), size=2, replace=False).tolist()
    obj, obj2 = objs[o_idx[0]], objs[o_idx[1]]

    t_acq = int(rng.choice(tpl_indices))
    t_trf = int(rng.choice(tpl_indices))
    t_fin = int(rng.choice(tpl_indices))

    events: List[List[int]] = [enc.encode(_ACQ_TPLS[t_acq].format(a=a_nm, obj=obj))]
    subcase = "consistent"

    if cls == 0:
        true_owner_id, actor_nm, actor_id = a_id, a_nm, a_id
        if float(rng.random()) < 0.50:
            t_d = int(rng.choice(tpl_indices))
            events.append(enc.encode(_TRF_TPLS[t_d].format(a=d_nm, b=e_nm, obj=obj2)))
            subcase = "consistent_with_distractor_transfer"
    elif cls == 1:
        true_owner_id, actor_nm, actor_id = b_id, b_nm, b_id
        events.append(enc.encode(_TRF_TPLS[t_trf].format(a=a_nm, b=b_nm, obj=obj)))
        subcase = "legal_transition"
        if float(rng.random()) < 0.35:
            t_d = int(rng.choice(tpl_indices))
            events.append(enc.encode(_ACQ_TPLS[t_d].format(a=d_nm, obj=obj2)))
            subcase = "legal_transition_with_distractor"
    else:
        if float(rng.random()) < 0.50:
            # Sub-case 2a: Unbacked intruder (no transfer of obj; 50% distractor transfer of obj2)
            true_owner_id, actor_nm, actor_id = a_id, c_nm, c_id
            subcase = "illegal_intruder_no_transfer"
            if float(rng.random()) < 0.50:
                t_d = int(rng.choice(tpl_indices))
                events.append(enc.encode(_TRF_TPLS[t_d].format(a=d_nm, b=e_nm, obj=obj2)))
        else:
            # Sub-case 2b: Post-transfer contradiction (stale ex-owner A or third party C!)
            true_owner_id = b_id
            events.append(enc.encode(_TRF_TPLS[t_trf].format(a=a_nm, b=b_nm, obj=obj)))
            if float(rng.random()) < 0.60:
                actor_nm, actor_id = a_nm, a_id
                subcase = "illegal_stale_ex_owner"
            else:
                actor_nm, actor_id = c_nm, c_id
                subcase = "illegal_post_transfer_intruder"

    fin_pre_txt, fin_suf_txt = _FIN_TPLS[t_fin]
    fin_pre = enc.encode(fin_pre_txt.format(obj=obj))
    actor_toks = enc.encode(" " + actor_nm)
    assert len(actor_toks) == 1 and actor_toks[0] == actor_id
    fin_suf = enc.encode(fin_suf_txt)

    fixed_len = sum(len(ev) for ev in events) + len(fin_pre) + len(actor_toks) + len(fin_suf)
    rem = target_len - fixed_len
    if rem < len(events) + 1:
        raise ValueError(f"target_len={target_len} too small for fixed_len={fixed_len}")

    # Distribute FineWeb prose across (leading gap + inter-event gaps + final gap) randomly -> ZERO position shortcut!
    n_gaps = len(events) + 1
    g_lead = int(rng.integers(0, min(48, max(1, rem // 4)) + 1))
    rem_mid = rem - g_lead
    if n_gaps == 2:
        gaps = [g_lead, rem_mid]
    else:
        # Random Dirichlet-like split across remaining middle gaps with min 2 tokens each
        cuts = sorted(rng.integers(2, max(3, rem_mid - 2 * (n_gaps - 2)), size=n_gaps - 2).tolist())
        parts = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [rem_mid - cuts[-1]]
        # Ensure strictly non-negative and exact sum == rem_mid
        parts = [max(1, p) for p in parts]
        diff = sum(parts) - rem_mid
        parts[-1] -= diff
        if parts[-1] < 1:
            parts = [rem_mid // (n_gaps - 1)] * (n_gaps - 1)
            parts[-1] += rem_mid - sum(parts)
        gaps = [g_lead] + parts

    max_st = max(1, len(d_arr) - rem - 16)
    st = int(rng.integers(0, max_st))
    seq: List[int] = []
    cur = st
    for gi, g_len in enumerate(gaps):
        if g_len > 0:
            seq.extend(d_arr[cur : cur + g_len].tolist())
            cur += g_len
        if gi < len(events):
            seq.extend(events[gi])
    prefix_len = len(seq) + len(fin_pre)
    seq.extend(fin_pre)
    seq.extend(actor_toks)
    seq.extend(fin_suf)
    assert len(seq) == target_len, f"Length mismatch {len(seq)} != {target_len}"
    actor_pos = prefix_len
    assert seq[actor_pos] == actor_id
    return {
        "label": cls,
        "subcase": subcase,
        "seq": seq,
        "owner_pred_pos": actor_pos - 1,
        "actor_pos": actor_pos,
        "end_pos": len(seq) - 1,
        "final_clause_len": len(fin_pre),
        "true_owner_id": true_owner_id,
        "actor_id": actor_id,
    }


def _build_balanced_dataset(
    enc: _TextEncoder,
    distractor_tok: torch.Tensor,
    name_pool: List[Tuple[str, int]],
    objs: List[str],
    n_per_class: int,
    target_len: int,
    seed: int,
    tpl_indices: Tuple[int, ...] = (0, 1, 2),
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    d_arr = distractor_tok.cpu().numpy()
    items: List[Dict[str, Any]] = []
    for cls in (0, 1, 2):
        for _ in range(n_per_class):
            items.append(_sample_one_trajectory(enc, d_arr, name_pool, objs, cls, target_len, rng, tpl_indices))
    return items


def _pool_verifier_features(
    sig: torch.Tensor,       # [B, T, D_sig]
    h_fin: torch.Tensor,     # [B, T, d_model]
    items: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Length-scale invariant pooling: [span_max, span_mean - prefix_mean, h_fin_span_mean]."""
    feats = []
    sig_only = []
    hf = h_fin.float()
    for b, it in enumerate(items):
        s_pos, e_pos = it["actor_pos"], it["end_pos"]
        sp = sig[b, s_pos : e_pos + 1]
        pre = sig[b, : max(1, s_pos)]
        sp_max = sp.max(dim=0).values
        sp_rel = sp.mean(dim=0) - pre.mean(dim=0)
        s_vec = torch.cat([sp_max, sp_rel], dim=-1)
        h_vec = hf[b, s_pos : e_pos + 1].mean(dim=0)
        sig_only.append(s_vec)
        feats.append(torch.cat([s_vec, h_vec], dim=-1))
    return torch.stack(feats, dim=0), torch.stack(sig_only, dim=0)


def _forward_pulse_batch(
    model: PULSEv7Model,
    items: List[Dict[str, Any]],
    device: torch.device,
    amp: bool,
) -> Dict[str, torch.Tensor]:
    tk = torch.tensor([it["seq"] for it in items], device=device, dtype=torch.long)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        out = model.forward_with_hooks(tk, collect_gdn_signals=True)
    sig = out["gdn_signals"].float()  # [B, T, n_gdn * H * 5]
    feats, sig_only = _pool_verifier_features(sig, out["h_fin"], items)
    pred_pos = torch.tensor([it["owner_pred_pos"] for it in items], device=device, dtype=torch.long)
    owner_logits = out["logits"][torch.arange(len(items), device=device), pred_pos].float()
    # Naive ||delta_t||_2 scalar at verification span (no transition conditioning!)
    n_h = model.config.n_heads
    naive_scores = []
    for b, it in enumerate(items):
        sp = sig[b, it["actor_pos"] : it["end_pos"] + 1]
        naive_scores.append(sp.view(sp.shape[0], -1, 5 * n_h)[..., :n_h].mean())
    return {
        "feats": feats,
        "sig_only": sig_only,
        "owner_logits": owner_logits,
        "naive_delta": torch.stack(naive_scores, dim=0),
    }


def _forward_tf_batch(
    model: ScalableTransformerLM,
    items: List[Dict[str, Any]],
    device: torch.device,
    amp: bool,
    target_sig_dim: int,
) -> Dict[str, torch.Tensor]:
    """Differentiable TF forward extracting EXACT matched-dimension per-head/per-layer dynamics (zero zero-padding at 30M!)."""
    tk = torch.tensor([it["seq"] for it in items], device=device, dtype=torch.long)
    bsz, seq_len = tk.shape
    n_h = model.config.n_heads
    hd = model.head_dim
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        x = model.emb_in(model.tok_emb(tk))
        cos, sin = model._rope_cos_sin(0, seq_len, device, x.dtype)
        per_layer_sigs: List[torch.Tensor] = []
        for blk in model.blocks:
            x_next, _ = blk(x, cos, sin, past_kv=None, use_cache=False)
            xf, xnf = x.float(), x_next.float()
            dx = xnf - xf
            xf_h = xf.view(bsz, seq_len, n_h, hd)
            xnf_h = xnf.view(bsz, seq_len, n_h, hd)
            dx_h = dx.view(bsz, seq_len, n_h, hd)
            # 2*H + 4 = 20 real features/layer -> 6 * 20 = 120 exact match to PULSE gdn_signals!
            h_dn = (dx_h.pow(2).sum(dim=-1) + 1e-6).sqrt()                 # [B, T, H]
            h_cos = F.cosine_similarity(xf_h, xnf_h, dim=-1, eps=1e-6)     # [B, T, H]
            l_dn = (dx.pow(2).sum(dim=-1, keepdim=True) + 1e-6).sqrt()     # [B, T, 1]
            l_cos = F.cosine_similarity(xf, xnf, dim=-1, eps=1e-6).unsqueeze(-1)
            l_xn = (xf.pow(2).sum(dim=-1, keepdim=True) + 1e-6).sqrt()
            l_xnn = (xnf.pow(2).sum(dim=-1, keepdim=True) + 1e-6).sqrt()
            per_layer_sigs.append(torch.cat([h_dn, h_cos, l_dn, l_cos, l_xn, l_xnn], dim=-1))
            x = x_next
        h_fin = model.norm_f(x)
        h_emb = model.emb_out(h_fin)
        pred_pos = torch.tensor([it["owner_pred_pos"] for it in items], device=device, dtype=torch.long)
        # Compute vocab projection ONLY at owner_pred_pos to save VRAM on long sequences!
        h_emb_sel = h_emb[torch.arange(bsz, device=device), pred_pos]
        owner_logits = F.linear(h_emb_sel, model.tok_emb.weight).float()

    sig = torch.cat(per_layer_sigs, dim=-1)  # [B, T, n_layers*(2H+4)]
    if sig.shape[-1] < target_sig_dim:
        sig = F.pad(sig, (0, target_sig_dim - sig.shape[-1]))
    elif sig.shape[-1] > target_sig_dim:
        sig = sig[..., :target_sig_dim]
    feats, sig_only = _pool_verifier_features(sig, h_fin, items)
    return {"feats": feats, "sig_only": sig_only, "owner_logits": owner_logits}


def _metrics_from_probs(probs: np.ndarray, labels: np.ndarray, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = probs.argmax(axis=-1)
    p_ill = probs[:, 2]
    auroc_all = compute_auroc(p_ill, (labels == 2).astype(np.int64))
    m_12 = (labels == 1) | (labels == 2)
    auroc_12 = compute_auroc(p_ill[m_12], (labels[m_12] == 2).astype(np.int64))
    far_legal = float((preds[labels == 1] == 2).mean() * 100.0) if (labels == 1).any() else 0.0
    far_cons = float((preds[labels == 0] == 2).mean() * 100.0) if (labels == 0).any() else 0.0
    acc_3way = float((preds == labels).mean() * 100.0)
    stale_mask = np.array([it["subcase"] == "illegal_stale_ex_owner" for it in items], dtype=bool)
    stale_rec = float((preds[stale_mask] == 2).mean() * 100.0) if stale_mask.any() else 0.0
    intr_mask = np.array([it["subcase"] == "illegal_intruder_no_transfer" for it in items], dtype=bool)
    intr_rec = float((preds[intr_mask] == 2).mean() * 100.0) if intr_mask.any() else 0.0
    return {
        "auroc_illegal_vs_all": round(auroc_all, 4),
        "auroc_illegal_vs_legal_transition": round(auroc_12, 4),
        "legal_transition_false_alarm_pct": round(far_legal, 2),
        "consistent_false_alarm_pct": round(far_cons, 2),
        "acc_3way_pct": round(acc_3way, 2),
        "stale_ex_owner_recall_pct": round(stale_rec, 2),
        "unbacked_intruder_recall_pct": round(intr_rec, 2),
        "pass_gate": bool(auroc_all >= 0.85 and far_legal <= 10.0),
    }


@torch.no_grad()
def _eval_arm_dataset(
    arm_kind: str,
    model: nn.Module,
    verifier: TransitionConditionedVerifier,
    items: List[Dict[str, Any]],
    device: torch.device,
    amp: bool,
    target_sig_dim: int,
    eval_bs: int = 4,
) -> Dict[str, Any]:
    model.eval()
    verifier.eval()
    all_probs, all_naive, all_sig_only = [], [], []
    owner_pref_hits = owner_top1_hits = 0
    ver_losses = []
    for i in range(0, len(items), eval_bs):
        chunk = items[i : i + eval_bs]
        if arm_kind == "pulse":
            res = _forward_pulse_batch(model, chunk, device, amp)  # type: ignore[arg-type]
            all_naive.extend(res["naive_delta"].cpu().tolist())
        else:
            res = _forward_tf_batch(model, chunk, device, amp, target_sig_dim)  # type: ignore[arg-type]
        v_logits = verifier(res["feats"])
        y_ch = torch.tensor([it["label"] for it in chunk], device=device, dtype=torch.long)
        ver_losses.append(float(F.cross_entropy(v_logits, y_ch, reduction="sum").item()))
        all_probs.append(F.softmax(v_logits, dim=-1).cpu().numpy())
        all_sig_only.append(res["sig_only"].cpu())
        o_lg = res["owner_logits"].cpu()
        for bi, it in enumerate(chunk):
            if int(o_lg[bi].argmax(-1).item()) == it["true_owner_id"]:
                owner_top1_hits += 1
            if it["label"] == 2 and float(o_lg[bi, it["true_owner_id"]].item()) > float(o_lg[bi, it["actor_id"]].item()):
                owner_pref_hits += 1
    probs = np.concatenate(all_probs, axis=0)
    labels = np.array([it["label"] for it in items], dtype=np.int64)
    metrics = _metrics_from_probs(probs, labels, items)
    n_ill = int((labels == 2).sum())
    metrics["ver_loss"] = round(sum(ver_losses) / max(1, len(items)), 4)
    metrics["owner_top1_acc_pct"] = round(owner_top1_hits / max(1, len(items)) * 100.0, 2)
    metrics["illegal_base_owner_pref_pct"] = round(owner_pref_hits / max(1, n_ill) * 100.0, 2)
    out = {
        "metrics": metrics,
        "probs": probs,
        "labels": labels,
        "sig_only": torch.cat(all_sig_only, dim=0),
    }
    if all_naive:
        out["naive_delta"] = np.array(all_naive, dtype=np.float64)
    return out


@torch.no_grad()
def _eval_decore_intervention(
    arm_kind: str,
    model: nn.Module,
    items: List[Dict[str, Any]],
    p_illegal_arr: np.ndarray,
    device: torch.device,
    amp: bool,
    lam: float = 1.5,
) -> Dict[str, Any]:
    """Verifier-gated DeCoRe contrastive decoding at owner_pred_pos: full context vs final-clause-only prior!"""
    model.eval()
    base_pref = adj_pref = base_top1 = adj_top1 = clean_same = ill_tot = clean_tot = 0
    for idx_i, it in enumerate(items):
        p_ill = torch.tensor(float(p_illegal_arr[idx_i]), device=device)
        prompt_ids = it["seq"][: it["actor_pos"]]
        prior_ids = prompt_ids[-it["final_clause_len"] :]
        tk_full = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        tk_prior = torch.tensor([prior_ids], device=device, dtype=torch.long)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            if arm_kind == "pulse":
                lf = model(tk_full, return_state=False, last_logits_only=True)["logits"][0, -1].float()  # type: ignore[operator]
                lp = model(tk_prior, return_state=False, last_logits_only=True)["logits"][0, -1].float()  # type: ignore[operator]
            else:
                lf = model(tk_full)[0][0, -1].float()  # type: ignore[operator]
                lp = model(tk_prior)[0][0, -1].float()  # type: ignore[operator]
        l_adj = decore_contrastive_logits(lf, lp, p_ill, lam=lam, threshold=0.5)
        t_id, a_id = it["true_owner_id"], it["actor_id"]
        if it["label"] == 2:
            ill_tot += 1
            if lf[t_id] > lf[a_id]:
                base_pref += 1
            if l_adj[t_id] > l_adj[a_id]:
                adj_pref += 1
            if int(lf.argmax(-1).item()) == t_id:
                base_top1 += 1
            if int(l_adj.argmax(-1).item()) == t_id:
                adj_top1 += 1
        else:
            clean_tot += 1
            if int(lf.argmax(-1).item()) == int(l_adj.argmax(-1).item()):
                clean_same += 1
    return {
        "illegal_base_owner_pref_pct": round(base_pref / max(1, ill_tot) * 100.0, 2),
        "illegal_decore_owner_pref_pct": round(adj_pref / max(1, ill_tot) * 100.0, 2),
        "illegal_base_top1_owner_pct": round(base_top1 / max(1, ill_tot) * 100.0, 2),
        "illegal_decore_top1_owner_pct": round(adj_top1 / max(1, ill_tot) * 100.0, 2),
        "clean_fluency_exact_match_pct": round(clean_same / max(1, clean_tot) * 100.0, 2),
    }


def _train_gdn_only_ablation(
    pulse_model: PULSEv7Model,
    tr_items_cache: List[Dict[str, Any]],
    va_sig: torch.Tensor,
    va_y: np.ndarray,
    eval_sigs: Dict[str, Tuple[torch.Tensor, np.ndarray, List[Dict[str, Any]]]],
    device: torch.device,
    amp: bool,
    steps: int = 200,
) -> Dict[str, Any]:
    """Train a strictly <50K verifier ONLY on PULSE's 240-dim native GDN Delta-Binding signals (ZERO h_fin!)."""
    pulse_model.eval()
    tr_sigs, tr_ys = [], []
    with torch.no_grad():
        for i in range(0, len(tr_items_cache), 9):
            ch = tr_items_cache[i : i + 9]
            res = _forward_pulse_batch(pulse_model, ch, device, amp)
            tr_sigs.append(res["sig_only"].cpu())
            tr_ys.extend([it["label"] for it in ch])
    X_tr = torch.cat(tr_sigs, dim=0).to(device)
    y_tr = torch.tensor(tr_ys, device=device, dtype=torch.long)
    X_va = va_sig.to(device)
    y_va = torch.from_numpy(va_y).long().to(device)

    torch.manual_seed(99)
    ver = TransitionConditionedVerifier(d_in=X_tr.shape[1], d_hidden=48, n_classes=3, dropout=0.35).to(device)
    opt = torch.optim.AdamW(ver.parameters(), lr=3e-3, weight_decay=0.05)
    best_vl, best_sd = float("inf"), None
    for st in range(steps):
        ver.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ver(X_tr), y_tr)
        loss.backward()
        opt.step()
        if (st + 1) % 10 == 0:
            ver.eval()
            with torch.no_grad():
                vl = float(F.cross_entropy(ver(X_va), y_va).item())
            if vl < best_vl:
                best_vl = vl
                best_sd = {k: v.detach().clone() for k, v in ver.state_dict().items()}
    if best_sd is not None:
        ver.load_state_dict(best_sd)
    ver.eval()
    out: Dict[str, Any] = {"verifier_params": ver.num_parameters(), "d_in": int(X_tr.shape[1]), "evals": {}}
    with torch.no_grad():
        for ename, (Xe, ye, e_items) in eval_sigs.items():
            probs = F.softmax(ver(Xe.to(device)), dim=-1).cpu().numpy()
            out["evals"][ename] = _metrics_from_probs(probs, ye, e_items)
    return out


def _joint_adapt_and_eval_arm(
    arm_kind: str,
    model: nn.Module,
    enc: _TextEncoder,
    train_tok: torch.Tensor,
    val_tok: torch.Tensor,
    tr_names: List[Tuple[str, int]],
    va_items: List[Dict[str, Any]],
    test_suites: Dict[str, List[Dict[str, Any]]],
    device: torch.device,
    amp: bool,
    smoke: bool,
    target_sig_dim: int,
    steps: int,
) -> Dict[str, Any]:
    """Run IDENTICAL joint 3-way adaptation + zero-shot length-extrapolation evaluation on PULSE or TF."""
    for p in model.parameters():
        p.requires_grad_(True)

    ppl_len = 32 if smoke else 512
    base_ppl = (
        eval_lm_ppl_loops(model, val_tok, ppl_len, device, amp, n_loops=2)["ppl"]  # type: ignore[arg-type]
        if arm_kind == "pulse"
        else eval_lm_ppl(model, val_tok, ppl_len, device, amp)["ppl"]  # type: ignore[arg-type]
    )
    print(f"[Stage 3b-v2 | {arm_kind}] Baseline FineWeb Val PPL = {base_ppl:.2f}", flush=True)

    d_in = 2 * target_sig_dim + model.config.d_model  # type: ignore[attr-defined]
    torch.manual_seed(42)
    verifier = TransitionConditionedVerifier(d_in=d_in, d_hidden=48, n_classes=3, dropout=0.35).to(device)
    print(f"[Stage 3b-v2 | {arm_kind}] Verifier d_in={d_in}, params={verifier.num_parameters():,} (<50K)", flush=True)

    base_lr_bb = 3.0e-4
    base_lr_ver = 2.0e-3
    opt = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "lr": base_lr_bb, "weight_decay": 0.01},
            {"params": list(verifier.parameters()), "lr": base_lr_ver, "weight_decay": 0.05},
        ]
    )

    # Identical RNG seed & 2-stage length curriculum (<=512!) for BOTH PULSE and TF -> exact same training stream!
    rng_tr = np.random.default_rng(20263)
    d_tr_arr = train_tok.cpu().numpy()
    short_lengths = [72] if smoke else [192, 256]
    full_lengths = [72, 96] if smoke else [192, 256, 320, 384, 448, 512]
    replay_wins = train_tok.unfold(0, ppl_len + 1, ppl_len)
    n_rep_wins = len(replay_wins)
    rep_bs = 2 if smoke else 4
    eval_every = 8 if smoke else 75
    warmup_steps = max(4, steps // 16)
    curr_cut = max(1, steps // 5)

    best_val_loss = float("inf")
    best_step = 0
    best_m_sd: Optional[Dict[str, torch.Tensor]] = None
    best_v_sd: Optional[Dict[str, torch.Tensor]] = None
    val_curve: List[Dict[str, Any]] = []
    cached_tr_for_ablation: List[Dict[str, Any]] = []

    sync(device)
    t0 = time.perf_counter()
    for st in range(1, steps + 1):
        mult = _cosine_lr_multiplier(st - 1, warmup_steps, steps, min_ratio=0.10)
        opt.param_groups[0]["lr"] = base_lr_bb * mult
        opt.param_groups[1]["lr"] = base_lr_ver * mult

        # Strictly balanced batch: bs=9 (3 Class 0, 3 Class 1, 3 Class 2) at shared random length L_b!
        L_b = int(rng_tr.choice(short_lengths if st <= curr_cut else full_lengths))
        batch_items = [
            _sample_one_trajectory(enc, d_tr_arr, tr_names, _TRAIN_OBJS, cls, L_b, rng_tr, (0, 1, 2))
            for cls in (0, 0, 0, 1, 1, 1, 2, 2, 2)
        ]
        if arm_kind == "pulse" and len(cached_tr_for_ablation) < (54 if smoke else 270) and L_b >= (72 if smoke else 320):
            cached_tr_for_ablation.extend(batch_items)

        model.train()
        verifier.train()
        opt.zero_grad(set_to_none=True)

        if arm_kind == "pulse":
            res = _forward_pulse_batch(model, batch_items, device, amp)  # type: ignore[arg-type]
        else:
            res = _forward_tf_batch(model, batch_items, device, amp, target_sig_dim)  # type: ignore[arg-type]

        y_cls = torch.tensor([it["label"] for it in batch_items], device=device, dtype=torch.long)
        y_own = torch.tensor([it["true_owner_id"] for it in batch_items], device=device, dtype=torch.long)
        loss_ver = F.cross_entropy(verifier(res["feats"]), y_cls)
        loss_own = F.cross_entropy(res["owner_logits"], y_own)

        # FineWeb-TRAIN LM replay (weight 0.5 -> protects general language modeling without drowning binding gradients!)
        r_idx = rng_tr.integers(0, n_rep_wins, size=rep_bs)
        rw = replay_wins[r_idx].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            if arm_kind == "pulse":
                loss_rep = model.compute_loss(rw)  # type: ignore[operator]
            else:
                lg_r, _ = model(rw[:, :-1])  # type: ignore[operator]
                loss_rep = F.cross_entropy(lg_r.reshape(-1, lg_r.size(-1)).float(), rw[:, 1:].reshape(-1))

        total_loss = loss_ver + 0.5 * loss_own + 0.5 * loss_rep
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite loss in {arm_kind} at step {st}: {total_loss.item()}")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(verifier.parameters()), 1.0)
        opt.step()

        if st % eval_every == 0 or st == steps:
            va_out = _eval_arm_dataset(arm_kind, model, verifier, va_items, device, amp, target_sig_dim, eval_bs=3)
            vm = va_out["metrics"]
            val_curve.append({"step": st, "ver_loss": vm["ver_loss"], "auroc": vm["auroc_illegal_vs_all"], "far_legal": vm["legal_transition_false_alarm_pct"], "acc3": vm["acc_3way_pct"]})
            print(
                f"[Stage 3b-v2 | {arm_kind} step {st:3d}/{steps}] tr_ver={loss_ver.item():.3f} tr_own={loss_own.item():.3f} "
                f"tr_rep={loss_rep.item():.3f} | VAL loss={vm['ver_loss']:.3f} AUROC={vm['auroc_illegal_vs_all']:.4f} "
                f"FAR_legal={vm['legal_transition_false_alarm_pct']:.1f}% Acc3={vm['acc_3way_pct']:.1f}%",
                flush=True,
            )
            if vm["ver_loss"] < best_val_loss:
                best_val_loss = vm["ver_loss"]
                best_step = st
                best_m_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_v_sd = {k: v.detach().cpu().clone() for k, v in verifier.state_dict().items()}

    sync(device)
    train_sec = time.perf_counter() - t0
    if best_m_sd is not None and best_v_sd is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_m_sd.items()})
        verifier.load_state_dict({k: v.to(device) for k, v in best_v_sd.items()})

    # Post-adaptation FineWeb-Edu Val PPL drift check
    after_ppl = (
        eval_lm_ppl_loops(model, val_tok, ppl_len, device, amp, n_loops=2)["ppl"]  # type: ignore[arg-type]
        if arm_kind == "pulse"
        else eval_lm_ppl(model, val_tok, ppl_len, device, amp)["ppl"]  # type: ignore[arg-type]
    )
    drift_pct = (after_ppl / base_ppl - 1.0) * 100.0
    print(f"[Stage 3b-v2 | {arm_kind}] Restored best step={best_step} | FineWeb PPL {base_ppl:.2f} -> {after_ppl:.2f} ({drift_pct:+.2f}%) | train={train_sec:.1f}s", flush=True)

    # Full Held-Out Evaluation across ALL lengths + unseen template + DeCoRe at L_short & L_4k!
    eval_results: Dict[str, Any] = {}
    naive_results: Dict[str, Any] = {}
    decore_results: Dict[str, Any] = {}
    eval_sigs_for_ablation: Dict[str, Tuple[torch.Tensor, np.ndarray, List[Dict[str, Any]]]] = {}

    va_final = _eval_arm_dataset(arm_kind, model, verifier, va_items, device, amp, target_sig_dim, eval_bs=3)
    for ename, e_items in test_suites.items():
        eb = 1 if ("4096" in ename or "2048" in ename) else 3
        ev = _eval_arm_dataset(arm_kind, model, verifier, e_items, device, amp, target_sig_dim, eval_bs=eb)
        eval_results[ename] = ev["metrics"]
        eval_sigs_for_ablation[ename] = (ev["sig_only"], ev["labels"], e_items)
        if "naive_delta" in ev:
            n_sc, ye = ev["naive_delta"], ev["labels"]
            m12 = (ye == 1) | (ye == 2)
            thr = float(np.median(n_sc))
            naive_results[ename] = {
                "auroc_illegal_vs_all": round(compute_auroc(n_sc, (ye == 2).astype(np.int64)), 4),
                "auroc_illegal_vs_legal_transition": round(compute_auroc(n_sc[m12], (ye[m12] == 2).astype(np.int64)), 4),
                "legal_transition_false_alarm_at_med_pct": round(float((n_sc[ye == 1] > thr).mean() * 100.0), 2),
            }
        if ename in ("L=512", "L=4096", "L=96", "L=192"):
            decore_results[ename] = _eval_decore_intervention(arm_kind, model, e_items, ev["probs"][:, 2], device, amp, lam=1.5)
        print(f"[Stage 3b-v2 | {arm_kind} | {ename}] {json.dumps(ev['metrics'])}", flush=True)

    payload: Dict[str, Any] = {
        "verifier_params": verifier.num_parameters(),
        "best_val_step": best_step,
        "train_sec": round(train_sec, 1),
        "fineweb_val_ppl_before": round(base_ppl, 2),
        "fineweb_val_ppl_after": round(after_ppl, 2),
        "fineweb_val_drift_pct": round(drift_pct, 2),
        "val_curve": val_curve,
        "evals": eval_results,
        "decore_intervention": decore_results,
    }
    if arm_kind == "pulse":
        payload["naive_delta_norm"] = naive_results
        payload["gdn_signals_only_ablation"] = _train_gdn_only_ablation(
            model, cached_tr_for_ablation, va_final["sig_only"], va_final["labels"], eval_sigs_for_ablation, device, amp, steps=(60 if smoke else 200)  # type: ignore[arg-type]
        )
        print(f"[Stage 3b-v2 | pulse GDN-only ablation] {json.dumps(payload['gdn_signals_only_ablation']['evals'])}", flush=True)
    return payload


def run_stage3b_v2(
    out: Path,
    fineweb_dir: Optional[Path],
    ckpt_dir: Optional[Path],
    device: torch.device,
    smoke: bool,
    steps: Optional[int] = None,
) -> Dict[str, Any]:
    dest = out / "stage3b_v2_results.json"
    amp = device.type == "cuda"
    sm89 = patch_fla_sm89_shared_memory()

    pulse_m, tf_m = _load_or_init_30m_models(ckpt_dir, device, smoke=smoke)
    if smoke:
        train_tok = torch.randint(4, 260, (96 * 16 + 1,))
        val_tok = torch.randint(4, 260, (96 * 12 + 1,))
        val_dist_tok = torch.randint(4, 260, (96 * 16 + 1,))
    else:
        assert fineweb_dir is not None
        reader = CachedFineWebReader(fineweb_dir)
        # Disjoint FineWeb splits: TRAIN prose for train/val/replay; VAL[0..32K] for exact Stage 2b PPL; VAL[32K..163K] for Test distractors!
        train_tok = reader.read("train", 0, 512 * 256 + 1)
        val_tok = reader.read("val", 0, 512 * 64 + 1)
        val_dist_tok = reader.read("val", 512 * 64 + 1, 512 * 256)

    enc = _TextEncoder(vocab_size=pulse_m.config.vocab_size, smoke=smoke)
    tr_names, va_names, te_names = _split_disjoint_names(enc, seed=707)
    n_gdn = sum(1 for t in pulse_m.config.layer_types if t == "gdn2")
    target_sig_dim = n_gdn * pulse_m.config.n_heads * 5  # 3 * 8 * 5 = 120 (60 in micro smoke)
    print(f"[Stage 3b-v2] Disjoint single-token names: train={len(tr_names)}, val={len(va_names)}, test={len(te_names)} | sig_dim={target_sig_dim}", flush=True)

    # Construct fixed entity-disjoint Val & Held-out Test suites shared IDENTICALLY between PULSE and TF!
    if smoke:
        va_items = _build_balanced_dataset(enc, train_tok, va_names, _VAL_OBJS, n_per_class=4, target_len=96, seed=801, tpl_indices=(0, 1, 2))
        test_suites = {
            "L=96": _build_balanced_dataset(enc, val_dist_tok, te_names, _TEST_OBJS, n_per_class=4, target_len=96, seed=802, tpl_indices=(0, 1, 2)),
            "L=96_unseen_tpl": _build_balanced_dataset(enc, val_dist_tok, te_names, _TEST_OBJS, n_per_class=4, target_len=96, seed=803, tpl_indices=(3,)),
            "L=192": _build_balanced_dataset(enc, val_dist_tok, te_names, _TEST_OBJS, n_per_class=4, target_len=192, seed=804, tpl_indices=(0, 1, 2)),
        }
        n_steps = int(steps if steps is not None else 16)
    else:
        va_items = _build_balanced_dataset(enc, train_tok, va_names, _VAL_OBJS, n_per_class=30, target_len=512, seed=801, tpl_indices=(0, 1, 2))
        test_suites = {
            "L=512": _build_balanced_dataset(enc, val_dist_tok, te_names, _TEST_OBJS, n_per_class=60, target_len=512, seed=802, tpl_indices=(0, 1, 2)),
            "L=512_unseen_tpl": _build_balanced_dataset(enc, val_dist_tok, te_names, _TEST_OBJS, n_per_class=30, target_len=512, seed=803, tpl_indices=(3,)),
            "L=2048": _build_balanced_dataset(enc, val_dist_tok, te_names, _TEST_OBJS, n_per_class=45, target_len=2048, seed=804, tpl_indices=(0, 1, 2)),
            "L=4096": _build_balanced_dataset(enc, val_dist_tok, te_names, _TEST_OBJS, n_per_class=45, target_len=4096, seed=805, tpl_indices=(0, 1, 2)),
        }
        n_steps = int(steps if steps is not None else 750)

    payload: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sm89_patch": sm89,
        "protocol": {
            "steps": n_steps,
            "batch_size": 9,
            "train_names": len(tr_names),
            "val_names": len(va_names),
            "test_names": len(te_names),
            "train_objs": len(_TRAIN_OBJS),
            "val_objs": len(_VAL_OBJS),
            "test_objs": len(_TEST_OBJS),
            "sig_dim_per_tok": target_sig_dim,
            "anti_shortcut_checks": [
                "disjoint_names_and_disjoint_obj_words",
                "fineweb_train_vs_val_prose",
                "50pct_post_transfer_stale_owner_contradictions",
                "50pct_class0_distractor_transfers",
                "randomized_batch_length_and_event_depths",
                "held_out_unseen_surface_template_3",
            ],
        },
    }

    # 1) PULSE-v7B-30M joint adaptation + eval (free GPU memory before TF!)
    payload["pulse_v7b_30m"] = _joint_adapt_and_eval_arm(
        "pulse", pulse_m, enc, train_tok, val_tok, tr_names, va_items, test_suites, device, amp, smoke, target_sig_dim, n_steps
    )
    save_json(dest, payload)
    del pulse_m
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2) TF-30M IDENTICAL joint adaptation + eval
    payload["tf_30m_matched"] = _joint_adapt_and_eval_arm(
        "tf", tf_m, enc, train_tok, val_tok, tr_names, va_items, test_suites, device, amp, smoke, target_sig_dim, n_steps
    )
    save_json(dest, payload)
    print("\n=== STAGE 3b-v2 COMPLETED ->", dest, "===", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fineweb-dir", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    run_stage3b_v2(args.out, args.fineweb_dir, args.ckpt_dir, device, smoke=args.smoke, steps=args.steps)


if __name__ == "__main__":
    main()
