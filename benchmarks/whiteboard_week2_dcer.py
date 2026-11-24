"""PULSE Week-2 Stage A Fast Screening: Route 2 — DC-ER (Dynamic Concept-Space Equilibrium Reasoner).

Strictly obeys all Blueprint Section 4 (Route 2) & Anti-Strawman constraints:
  - Total params strictly <= TF-Modern (~3.69M), 0 hidden params (`audit_zero_hidden_params == True`).
  - Strictly causal dynamic chunking (unit-tested for zero future-token leakage on BOTH `boundary_mask` & `logits`).
  - DLCM Decoupled muP LR scaling (`concept_core` LR = `base_lr * d_tok / d_con = 0.80 * base_lr`).
  - Strict Iso-Time L4 comparison (after 3 untimed warmup steps so `torch.compile` never penalizes baselines!)
    against full-strength `tf_modern` and `pulse_v7b_ref` across three orthogonal probes (~11 min / ~0.9 CCU total):
      * T1 — `P5b Lano VarDensity CFG`: exact Week-1 variable-density CFG -> audits `audit_boundary_histogram`
             (target compression ratio in [3.2, 4.8], selectivity >= 1.5, zero ratio drift) & wall-clock ratio <= 1.45x TF.
      * T2 — `P5b-Deep Variable-Density Concept Stack`: 8-deep LIFO stack with adjacent-pair mod-4 composition
             hidden inside ~78% variable-length (3-5 tok) deterministic filler macros -> direct Iso-Time gap test!
      * T3 — `P1b Depo2-SC`: Single-Cycle multi-token node graph (5-7 tok/node, vocab=4, ZERO 2/4-cycle shortcuts!)
             -> direct Iso-Time multi-hop exact-match & test-time R=4 -> R=6 concept-loop scaling!
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.pulse_v7 import patch_fla_sm89_shared_memory
from pulse.whiteboard_anvil import (
    audit_boundary_histogram,
    audit_zero_hidden_params,
    build_dual_strong_baselines,
)
from pulse.whiteboard_routes import DCERConfig, DCERModel

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import save_json, sync  # noqa: E402
from whiteboard_probes import sample_p5b_lano_vardensity  # noqa: E402
from whiteboard_week2_v2 import _wsd_lr_multiplier  # noqa: E402


# ==============================================================================
# Probe T2: P5b-Deep — Variable-Density Concept-Stack CFG (True ~4x Concept Ratio)
# ==============================================================================

def sample_p5b_lano_deep_stack(
    batch_size: int,
    stack_depth: int = 8,
    seq_len: int = 160,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Depth-8 LIFO concept stack wrapped in variable-length (3-5 tok) deterministic macros!

    Tokens:
      - 0..3   : open-concept symbols c_i (4-way high entropy)
      - 4..7   : close-concept targets y_j = 4 + ((c_{K-1-j} + c_{K-2-j}) % 4) (LIFO pop + neighbor sum!)
      - 16..39 : variable-length (3, 4, 5, 3 tok) deterministic low-entropy filler macros (H=0, ~78% of tokens!)
      - 44     : PUSH_END boundary marker
      - 45     : PAD (vocab=48)
    """
    rng = np.random.default_rng(seed)
    open_macros = [
        [16, 17, 18],              # len 3
        [19, 20, 21, 22],          # len 4
        [23, 24, 25, 26, 27],      # len 5
        [28, 29, 30],              # len 3  (mean macro len = 3.75 -> concept ratio ~4.75 -> with PUSH_END ~4.4!)
    ]
    close_macros = [list(reversed(m)) for m in open_macros]
    t_mid, t_pad = 44, 45

    seqs = np.full((batch_size, seq_len + 1), t_pad, dtype=np.int64)
    close_mask = np.zeros((batch_size, seq_len + 1), dtype=np.bool_)
    macro_mask = np.zeros((batch_size, seq_len + 1), dtype=np.bool_)
    hi_in_mask = np.zeros((batch_size, seq_len + 1), dtype=np.bool_)

    for b in range(batch_size):
        cs = rng.integers(0, 4, size=stack_depth).tolist()
        toks: List[int] = []
        cm: List[bool] = []
        mm: List[bool] = []
        hm: List[bool] = []
        # Opening push phase
        for c in cs:
            toks.append(int(c))
            cm.append(False); mm.append(False); hm.append(True)
            m = open_macros[int(c)]
            toks.extend(m)
            cm.extend([False] * len(m)); mm.extend([True] * len(m)); hm.extend([False] * len(m))
        # Midpoint switch
        toks.append(t_mid)
        cm.append(False); mm.append(False); hm.append(True)
        # LIFO closing pop + adjacent-pair composition phase
        for j in range(stack_depth):
            c_top = cs[stack_depth - 1 - j]
            c_nxt = cs[stack_depth - 2 - j] if (stack_depth - 2 - j) >= 0 else 0
            y = (c_top + c_nxt) % 4
            toks.append(4 + y)
            cm.append(True); mm.append(False); hm.append(True)
            m = close_macros[y]
            toks.extend(m)
            cm.extend([False] * len(m)); mm.extend([True] * len(m)); hm.extend([False] * len(m))
        assert len(toks) <= seq_len + 1, f"DeepStack overflow {len(toks)} > {seq_len+1}"
        seqs[b, : len(toks)] = toks
        close_mask[b, : len(toks)] = cm
        macro_mask[b, : len(toks)] = mm
        hi_in_mask[b, : len(toks)] = hm

    return {
        "input_ids": torch.from_numpy(seqs[:, :-1]),
        "targets": torch.from_numpy(seqs[:, 1:]),
        "close_mask": torch.from_numpy(close_mask[:, 1:]),
        "macro_mask": torch.from_numpy(macro_mask[:, 1:]),
        "hi_in_mask": torch.from_numpy(hi_in_mask[:, :-1]),  # Strictly aligned with input_ids!
        "active_mask": torch.from_numpy(seqs[:, :-1] != t_pad),
        "pad_id": t_pad,
    }


# ==============================================================================
# Probe T3: P1b Depo2-SC — Single-Cycle Multi-Token Node Graph (ZERO 2/4-Cycles!)
# ==============================================================================

def sample_p1b_depo2_single_cycle(
    batch_size: int,
    n_nodes: int = 10,
    hops: Tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    seq_len: int = 224,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Single Hamiltonian cycle Depo2: 5-7 tok/node over vocab {0,1,2,3} (ZERO short-cycle cheat!).
    Tokens: 0..3=alphabet, 4=NODE_SEP, 5=EDGE_SEP, 6=QRY, 7..14=HOP_1..HOP_8, 15=PAD (vocab=16).
    """
    rng = np.random.default_rng(seed)
    node_sep, edge_sep, qry_tok, hop_base, pad_tok = 4, 5, 6, 7, 15
    seqs = np.full((batch_size, seq_len), pad_tok, dtype=np.int64)
    ans_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
    struct_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
    hop_ids = np.empty(batch_size, dtype=np.int64)

    for b in range(batch_size):
        codes: List[List[int]] = []
        seen = set()
        while len(codes) < n_nodes:
            clen = int(rng.integers(5, 8))
            c = tuple(rng.integers(0, 4, size=clen).tolist())
            if c not in seen:
                seen.add(c)
                codes.append(list(c))
        # Single full cycle of length n_nodes: pi^k(s) != s for ALL 1 <= k < n_nodes!
        cyc = rng.permutation(n_nodes)
        perm = np.empty(n_nodes, dtype=np.int64)
        for i in range(n_nodes):
            perm[cyc[i]] = cyc[(i + 1) % n_nodes]
        order = rng.permutation(n_nodes)
        toks: List[int] = []
        sm: List[bool] = []
        for u in order:
            toks.extend(codes[u]); sm.extend([True] + [False] * (len(codes[u]) - 1))
            toks.append(node_sep); sm.append(True)
            toks.extend(codes[perm[u]]); sm.extend([True] + [False] * (len(codes[perm[u]]) - 1))
            toks.append(edge_sep); sm.append(True)
        s = int(rng.integers(0, n_nodes))
        k = int(rng.choice(hops))
        cur = s
        for _ in range(k):
            cur = int(perm[cur])
        toks.append(qry_tok); sm.append(True)
        toks.extend(codes[s]); sm.extend([True] + [False] * (len(codes[s]) - 1))
        toks.append(hop_base + (k - 1)); sm.append(True)
        toks.append(node_sep); sm.append(True)
        ans_start = len(toks)
        ans_code = codes[cur] + [edge_sep]
        toks.extend(ans_code); sm.extend([True] + [False] * (len(codes[cur]) - 1) + [True])
        assert len(toks) <= seq_len, f"Depo2-SC overflow {len(toks)} > {seq_len}"
        seqs[b, : len(toks)] = toks
        struct_mask[b, : len(toks)] = sm
        ans_mask[b, ans_start - 1 : ans_start - 1 + len(ans_code)] = True
        hop_ids[b] = k

    return {
        "input_ids": torch.from_numpy(seqs[:, :-1]),
        "targets": torch.from_numpy(seqs[:, 1:]),
        "ans_mask": torch.from_numpy(ans_mask[:, :-1]),
        "struct_in_mask": torch.from_numpy(struct_mask[:, :-1]),
        "active_mask": torch.from_numpy(seqs[:, :-1] != pad_tok),
        "hops": torch.from_numpy(hop_ids),
        "pad_id": pad_tok,
    }


# ==============================================================================
# Shared Strict Iso-Time Trainer (3 untimed warmup steps -> exact equal wall-clock!)
# ==============================================================================

def _build_arm(arm: str, vocab_size: int, smoke: bool, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if arm in ("tf_modern", "pulse_v7b_ref"):
        return build_dual_strong_baselines(vocab_size, smoke=smoke, seed=seed)[arm]
    cfg = DCERConfig.micro_test(vocab_size) if smoke else DCERConfig.lab_3m(vocab_size)
    return DCERModel(cfg)


def _train_isotime(
    arm: str,
    vocab_size: int,
    step_loss_fn: Any,
    eval_fn: Any,
    device: torch.device,
    amp: bool,
    smoke: bool,
    seed: int,
    iso_sec: float,
    max_steps: int,
    base_lr: float,
    tag: str,
    post_eval_fn: Any = None,
) -> Dict[str, Any]:
    model = _build_arm(arm, vocab_size, smoke=smoke, seed=seed).to(device)
    if hasattr(model, "compile_fast_paths"):
        model.compile_fast_paths()
    if isinstance(model, DCERModel):
        opt = torch.optim.AdamW(model.decoupled_param_groups(base_lr, weight_decay=0.01), betas=(0.9, 0.98))
    else:
        opt = torch.optim.AdamW(model.io_parameters(), lr=base_lr, betas=(0.9, 0.98), weight_decay=0.01)

    # 3 untimed warmup steps so torch.compile / FLA autotuning NEVER eats into Iso-Time wall-clock!
    for wst in range(3):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = step_loss_fn(model, seed * 900000 + wst)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.io_parameters(), 1.0)
        opt.step()
    sync(device)

    w_steps = max(5, min(120, max_steps // 10))
    ratio_traj: List[float] = []
    t0 = time.perf_counter()
    steps_done = 0
    best_sc = -1e9
    best_det: Dict[str, Any] = {}
    next_eval_t = iso_sec * 0.35

    while steps_done < max_steps:
        elapsed = time.perf_counter() - t0
        if elapsed >= iso_sec and steps_done >= (10 if smoke else 50):
            break
        steps_done += 1
        frac_t = min(1.0, elapsed / max(1e-3, iso_sec))
        # Time-based WSD decay in final 15% of wall-clock budget (fair regardless of step count!)
        if steps_done <= w_steps:
            mult = max(0.05, float(steps_done) / w_steps)
        elif frac_t < 0.85:
            mult = 1.0
        else:
            t_d = min(1.0, (frac_t - 0.85) / 0.15)
            mult = 0.1 + 0.9 * 0.5 * (1.0 + float(np.cos(np.pi * t_d)))
        for pg in opt.param_groups:
            pg["lr"] = base_lr * pg.get("lr_mult", 1.0) * mult

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = step_loss_fn(model, seed * 100000 + steps_done)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.io_parameters(), 1.0)
        opt.step()

        if isinstance(model, DCERModel) and (steps_done % 25 == 0 or steps_done == 1):
            ratio_traj.append(round(model.last_compression_ratio, 3))

        now_e = time.perf_counter() - t0
        if now_e >= next_eval_t and not smoke:
            sync(device)
            ev_t0 = time.perf_counter()
            model.eval()
            sc, det = eval_fn(model)
            if sc >= best_sc:
                best_sc, best_det = sc, det
            # Pause Iso-Time clock during eval so eval latency never penalizes training budget!
            t0 += time.perf_counter() - ev_t0
            next_eval_t = (time.perf_counter() - t0) + iso_sec * 0.35

    sync(device)
    train_sec = time.perf_counter() - t0
    model.eval()
    sc_fin, det_fin = eval_fn(model)
    if sc_fin >= best_sc:
        best_sc, best_det = sc_fin, det_fin

    rec: Dict[str, Any] = {
        "arm": arm,
        "params": model.num_parameters(),
        "seed": seed,
        "steps": steps_done,
        "train_sec": round(train_sec, 2),
        "ms_per_step": round(train_sec * 1000.0 / max(1, steps_done), 2),
        "primary_score": round(best_sc, 2),
        "metrics_final": det_fin,
        "metrics_best": best_det,
    }
    if isinstance(model, DCERModel) and ratio_traj:
        rec["ratio_trajectory_summary"] = {
            "min": min(ratio_traj),
            "max": max(ratio_traj),
            "final": ratio_traj[-1],
            "stayed_in_kill_bounds_2p5_6p0": bool(min(ratio_traj) >= 2.5 and max(ratio_traj) <= 6.0),
        }
    if post_eval_fn is not None and isinstance(model, DCERModel):
        rec["test_time_R6_extrap"] = post_eval_fn(model)
    print(
        f"[{tag}:{arm}] steps={steps_done} ({train_sec:.1f}s, {rec['ms_per_step']}ms/st) "
        f"score={best_sc:.2f} {json.dumps(det_fin)}",
        flush=True,
    )
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


# ==============================================================================
# Stage A Runner: T1 (Lano CFG Audit) + T2 (Deep Concept Stack) + T3 (Depo2-SC)
# ==============================================================================

def run_dcer_stage_a(out: Path, device: torch.device, amp: bool, smoke: bool, seed: int, deadline: float) -> Dict[str, Any]:
    dest = out / "route2_dcer_stage_a.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    # Mandatory Zero-Hidden-Params & Strict <= TF Parity Audit across V=16 and V=48!
    audits = {}
    for v in (16, 48):
        dc_a = DCERModel(DCERConfig.micro_test(v) if smoke else DCERConfig.lab_3m(v))
        tf_a = build_dual_strong_baselines(v, smoke=smoke, seed=seed)["tf_modern"]
        au = audit_zero_hidden_params(dc_a, v, dc_a.config.d_tok)
        assert au["pass_zero_hidden_audit"] and (smoke or dc_a.num_parameters() <= tf_a.num_parameters()), (
            f"DC-ER audit/parity failed at V={v}: {dc_a.num_parameters()} vs TF {tf_a.num_parameters()} | {au}"
        )
        audits[f"V{v}"] = {"dcer_params": dc_a.num_parameters(), "tf_params": tf_a.num_parameters(), "audit": au}
        del dc_a, tf_a

    bs = 16 if smoke else 64
    arms = ("tf_modern", "pulse_v7b_ref", "dc_er")
    res: Dict[str, Any] = {"audits": audits, "t1_lano_vardensity": {}, "t2_lano_deep_stack": {}, "t3_depo2_single_cycle": {}}

    # --- T1: Exact Week-1 P5b Lano VarDensity CFG (Boundary Histogram & Speed Ratio Audit) ---
    seq_t1 = 96 if smoke else 192
    iso_t1 = 4.0 if smoke else 45.0
    max_st1 = 25 if smoke else 2000
    ev_t1 = sample_p5b_lano_vardensity(32 if smoke else 128, seq_len=seq_t1, seed=5201)

    def _loss_t1(m: nn.Module, s_sd: int) -> torch.Tensor:
        b = sample_p5b_lano_vardensity(bs, seq_len=seq_t1, seed=s_sd)
        o = m(b["input_ids"].to(device))
        loss = F.cross_entropy(o["logits"].float().reshape(-1, 48), b["targets"].to(device).reshape(-1))
        return loss + m.aux_loss if isinstance(m, DCERModel) else loss

    @torch.no_grad()
    def _eval_t1(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
        x_d, y_d = ev_t1["input_ids"].to(device), ev_t1["targets"].to(device)
        hi_tgt = ev_t1["hi_entropy_mask"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            o = m(x_d)
            lg = o["logits"].float()
        tok_nll = F.cross_entropy(lg.reshape(-1, 48), y_d.reshape(-1), reduction="none").view_as(y_d)
        lo_acc = float(lg[~hi_tgt].argmax(-1).eq(y_d[~hi_tgt]).float().mean().item() * 100.0)
        det: Dict[str, Any] = {
            "overall_nll": round(float(tok_nll.mean().item()), 4),
            "hi_entropy_nll": round(float(tok_nll[hi_tgt].mean().item()), 4),
            "lo_macro_acc": round(lo_acc, 2),
        }
        if isinstance(m, DCERModel):
            hi_in = (ev_t1["input_ids"] < 16).numpy()
            bm = o["boundary_mask"].cpu().numpy()
            det["boundary_audit"] = audit_boundary_histogram(bm[:, 1:], hi_in[:, 1:])
        return float(100.0 - tok_nll.mean().item() * 20.0), det

    for arm in arms:
        res["t1_lano_vardensity"][arm] = _train_isotime(
            arm, 48, _loss_t1, _eval_t1, device, amp, smoke, seed, iso_t1, max_st1, 1.5e-3, "T1-LanoCFG"
        )
    tf_ms1 = res["t1_lano_vardensity"]["tf_modern"]["ms_per_step"]
    res["t1_lano_vardensity"]["dcer_vs_tf_wallclock_ratio"] = round(res["t1_lano_vardensity"]["dc_er"]["ms_per_step"] / max(1e-3, tf_ms1), 3)
    save_json(dest, res)

    # --- T2: P5b-Deep Variable-Density Concept-Stack CFG (Strict Iso-Time Separation!) ---
    if time.time() < deadline - 120:
        d_st2 = 5 if smoke else 8
        seq_t2 = 112 if smoke else 160
        iso_t2 = 4.0 if smoke else 75.0
        max_st2 = 25 if smoke else 3000
        ev_t2 = sample_p5b_lano_deep_stack(32 if smoke else 256, stack_depth=d_st2, seq_len=seq_t2, seed=5301)

        def _loss_t2(m: nn.Module, s_sd: int) -> torch.Tensor:
            b = sample_p5b_lano_deep_stack(bs, stack_depth=d_st2, seq_len=seq_t2, seed=s_sd)
            kw = {"active_mask": b["active_mask"].to(device), "pad_id": b["pad_id"]} if isinstance(m, DCERModel) else {}
            o = m(b["input_ids"].to(device), **kw)
            loss = F.cross_entropy(o["logits"].float().reshape(-1, 48), b["targets"].to(device).reshape(-1), ignore_index=b["pad_id"])
            return loss + m.aux_loss if isinstance(m, DCERModel) else loss

        @torch.no_grad()
        def _eval_t2(m: nn.Module, override_loops: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
            x_d, y_d = ev_t2["input_ids"].to(device), ev_t2["targets"].to(device)
            cm, mm = ev_t2["close_mask"], ev_t2["macro_mask"]
            kw: Dict[str, Any] = {}
            if isinstance(m, DCERModel):
                kw = {"active_mask": ev_t2["active_mask"].to(device), "pad_id": ev_t2["pad_id"], "n_loops": override_loops}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                o = m.forward_anvil(x_d, **kw) if isinstance(m, DCERModel) else m(x_d)
                lg = o["logits"].float().cpu()
            preds = lg.argmax(-1)
            close_acc = float(preds[cm].eq(ev_t2["targets"][cm]).float().mean().item() * 100.0)
            macro_acc = float(preds[mm].eq(ev_t2["targets"][mm]).float().mean().item() * 100.0)
            full_em = float(np.mean([bool(preds[i][cm[i]].eq(ev_t2["targets"][i][cm[i]]).all().item()) for i in range(preds.shape[0])]) * 100.0)
            det: Dict[str, Any] = {
                "closing_concept_acc_pct": round(close_acc, 2),
                "closing_stack_full_em_pct": round(full_em, 2),
                "macro_acc_pct": round(macro_acc, 2),
            }
            if isinstance(m, DCERModel):
                am = ev_t2["active_mask"].numpy()[:, 1:]
                bm = o["boundary_mask"].cpu().numpy()[:, 1:]
                hi = ev_t2["hi_in_mask"].numpy()[:, 1:]
                det["boundary_audit"] = audit_boundary_histogram(bm[am], hi[am])
            return close_acc, det

        for arm in arms:
            res["t2_lano_deep_stack"][arm] = _train_isotime(
                arm, 48, _loss_t2, _eval_t2, device, amp, smoke, seed, iso_t2, max_st2, 1.5e-3, "T2-DeepStack",
                post_eval_fn=(lambda m: _eval_t2(m, override_loops=(5 if smoke else 6))[1]),
            )
        tf_ms2 = res["t2_lano_deep_stack"]["tf_modern"]["ms_per_step"]
        res["t2_lano_deep_stack"]["dcer_vs_tf_wallclock_ratio"] = round(res["t2_lano_deep_stack"]["dc_er"]["ms_per_step"] / max(1e-3, tf_ms2), 3)
        res["t2_lano_deep_stack"]["dcer_gap_vs_tf_pp"] = round(
            res["t2_lano_deep_stack"]["dc_er"]["primary_score"] - res["t2_lano_deep_stack"]["tf_modern"]["primary_score"], 2
        )
        res["t2_lano_deep_stack"]["dcer_gap_vs_pulse_pp"] = round(
            res["t2_lano_deep_stack"]["dc_er"]["primary_score"] - res["t2_lano_deep_stack"]["pulse_v7b_ref"]["primary_score"], 2
        )
        save_json(dest, res)

    # --- T3: P1b Depo2-SC Single-Cycle Multi-Token Graph (5-7 tok/node, vocab=4, Strict Iso-Time!) ---
    if time.time() < deadline - 120:
        n_nd = 7 if smoke else 10
        seq_t3 = 160 if smoke else 224
        iso_t3 = 4.0 if smoke else 80.0
        max_st3 = 25 if smoke else 3000
        ev_h_list = (1, 2, 4) if smoke else (1, 2, 4, 8)
        ev_t3 = {h: sample_p1b_depo2_single_cycle(24 if smoke else 128, n_nodes=n_nd, hops=(h,), seq_len=seq_t3, seed=5400 + h) for h in ev_h_list}

        def _loss_t3(m: nn.Module, s_sd: int) -> torch.Tensor:
            b = sample_p1b_depo2_single_cycle(bs, n_nodes=n_nd, hops=(1, 2, 3, 4, 6, 8), seq_len=seq_t3, seed=s_sd)
            kw = {"active_mask": b["active_mask"].to(device), "pad_id": b["pad_id"]} if isinstance(m, DCERModel) else {}
            o = m(b["input_ids"].to(device), **kw)
            lg = o["logits"].float()
            msk = b["ans_mask"].to(device)
            loss = F.cross_entropy(lg[msk], b["targets"].to(device)[msk])
            return loss + m.aux_loss if isinstance(m, DCERModel) else loss

        @torch.no_grad()
        def _eval_t3(m: nn.Module, override_loops: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
            det: Dict[str, Any] = {}
            ems: List[float] = []
            for h, eb in ev_t3.items():
                kw: Dict[str, Any] = {"active_mask": eb["active_mask"].to(device), "pad_id": eb["pad_id"], "n_loops": override_loops} if isinstance(m, DCERModel) else {}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    o = m.forward_anvil(eb["input_ids"].to(device), **kw) if isinstance(m, DCERModel) else m(eb["input_ids"].to(device))
                preds = o["logits"].float().argmax(-1).cpu()
                msk, tgt = eb["ans_mask"], eb["targets"]
                em = float(np.mean([bool((preds[i][msk[i]] == tgt[i][msk[i]]).all().item()) for i in range(preds.shape[0])]) * 100.0)
                det[f"hop{h}_em"] = round(em, 2)
                ems.append(em)
                if isinstance(m, DCERModel) and h == ev_h_list[-1]:
                    am = eb["active_mask"].numpy()[:, 1:]
                    bm = o["boundary_mask"].cpu().numpy()[:, 1:]
                    st_m = eb["struct_in_mask"].numpy()[:, 1:]
                    det["boundary_audit"] = audit_boundary_histogram(bm[am], st_m[am])
            mean_em = float(statistics.mean(ems))
            det["mean_em"] = round(mean_em, 2)
            return mean_em, det

        for arm in arms:
            res["t3_depo2_single_cycle"][arm] = _train_isotime(
                arm, 16, _loss_t3, _eval_t3, device, amp, smoke, seed, iso_t3, max_st3, 1.5e-3, "T3-Depo2SC",
                post_eval_fn=(lambda m: _eval_t3(m, override_loops=(5 if smoke else 6))[1]),
            )
        tf_ms3 = res["t3_depo2_single_cycle"]["tf_modern"]["ms_per_step"]
        res["t3_depo2_single_cycle"]["dcer_vs_tf_wallclock_ratio"] = round(res["t3_depo2_single_cycle"]["dc_er"]["ms_per_step"] / max(1e-3, tf_ms3), 3)
        res["t3_depo2_single_cycle"]["dcer_gap_vs_tf_pp"] = round(
            res["t3_depo2_single_cycle"]["dc_er"]["primary_score"] - res["t3_depo2_single_cycle"]["tf_modern"]["primary_score"], 2
        )
        res["t3_depo2_single_cycle"]["dcer_gap_vs_pulse_pp"] = round(
            res["t3_depo2_single_cycle"]["dc_er"]["primary_score"] - res["t3_depo2_single_cycle"]["pulse_v7b_ref"]["primary_score"], 2
        )
        save_json(dest, res)

    # Stage A Blueprint Gate & Kill-Switch Summary
    t1_au = res["t1_lano_vardensity"]["dc_er"]["metrics_final"].get("boundary_audit", {})
    t2_au = res["t2_lano_deep_stack"].get("dc_er", {}).get("metrics_final", {}).get("boundary_audit", {})
    ratios_wc = [
        res["t1_lano_vardensity"].get("dcer_vs_tf_wallclock_ratio", 9.9),
        res["t2_lano_deep_stack"].get("dcer_vs_tf_wallclock_ratio", 9.9),
        res["t3_depo2_single_cycle"].get("dcer_vs_tf_wallclock_ratio", 9.9),
    ]
    res["stage_a_verdict"] = {
        "max_wallclock_ratio_vs_tf": max(ratios_wc),
        "pass_wallclock_1p45x_gate": bool(max(ratios_wc) <= 1.45),
        "t1_boundary_audit_pass": bool(t1_au.get("pass_boundary_audit", False)),
        "t2_boundary_audit_pass": bool(t2_au.get("pass_boundary_audit", False)),
        "t2_deep_stack_gap_vs_tf_pp": res["t2_lano_deep_stack"].get("dcer_gap_vs_tf_pp", 0.0),
        "t2_deep_stack_gap_vs_pulse_pp": res["t2_lano_deep_stack"].get("dcer_gap_vs_pulse_pp", 0.0),
        "t3_depo2_gap_vs_tf_pp": res["t3_depo2_single_cycle"].get("dcer_gap_vs_tf_pp", 0.0),
        "t3_depo2_gap_vs_pulse_pp": res["t3_depo2_single_cycle"].get("dcer_gap_vs_pulse_pp", 0.0),
    }
    save_json(dest, res)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--deadline", type=float, default=time.time() + 3600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()

    summary: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sm89_patch": sm89,
        "seed": args.seed,
        "route2_dcer_stage_a": run_dcer_stage_a(args.out, device, amp, args.smoke, args.seed, args.deadline),
    }
    save_json(args.out / "week2_dcer_summary.json", summary)
    print("\n=== WEEK-2 ROUTE 2 (DC-ER) STAGE A COMPLETED ->", args.out / "week2_dcer_summary.json", "===", flush=True)


if __name__ == "__main__":
    main()
