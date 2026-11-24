"""PULSE Week-2 Stage A Fast Screening: Route 3 (FOMP-CMS) + Route 1 (APRM) on Colab NVIDIA L4.

Strictly obeys all Week-1 Anti-Strawman rules & Stage A Kill Switches:
  - Zero hidden params verified via `audit_zero_hidden_params`; total params strictly <= TF-Modern (~3.73M).
  - Part A — Route 3 (`FOMP-CMS`, ~4 min / ~0.35 CCU):
      * Exact Week-1 `P4RuleWorld(seed=404)` (test entities 48..63 100% unseen in BOTH pretraining & episodic training!).
      * 100% 1st-order autograd (zero 2nd-order HVP); evaluates BOTH M=1 per-episode AND M=16 simultaneous compilation!
      * Mandatory ICML 2026 Sign-Flip Check (-DeltaW_mid MUST collapse >= 50 pp on 1-hop & 2-hop!).
      * Pass Gate: 1-Pass 2-Hop >= 75% (vs Week-1 TF/PULSE 18.75%), Sign-Flip PASS, self-gen drift < 0.05 nats, bg drift < 0.5%.
  - Part B — Route 1 (`APRM`, ~14 min / ~1.15 CCU):
      * B1 (`P1a-SC`): Single-Cycle Derangement Depo1 (N=16, ZERO short-cycle shortcut!) with IDENTICAL 2-stage hop
        curriculum (1..2 -> 1..4 -> 1..8) on TF-Modern, PULSE-v7B-Ref, AND APRM + direct Gradient-Horizon Ablation
        (grad_horizon = -1 [full BPTT] vs 4 [last-4] vs 1 [1-step implicit]) + Linear-Probe Hop Ladder + OOD hop10/12
        test-time loop scaling (T=8 -> T=12)!
      * B2 (`P1c DAG Arithmetic`): Depth 5+ flat register evaluation vs Week-1 11.33%.
      * B3 (`P2 S5-Retrieval 4x`): Non-Abelian S_5 x K=64 retrieval vs Week-1 21.4% / 25.0%.
      * B4 (`P5a Star-Graph`): 5-branch lookahead planning vs Week-1 7.6% / 18.9%.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW, _cosine_lr_multiplier
from pulse.pulse_v7 import patch_fla_sm89_shared_memory
from pulse.whiteboard_anvil import (
    audit_gradient_sign_flip,
    audit_linear_probe_hop_ladder,
    audit_zero_hidden_params,
    build_dual_strong_baselines,
)
from pulse.whiteboard_routes import APRMConfig, APRMModel, FOMPCMSConfig, FOMPCMSModel

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import save_json, sync  # noqa: E402
from whiteboard_probes import (  # noqa: E402
    P4RuleWorld,
    sample_p1c_dag_arith,
    sample_p2_s5_retrieval,
    sample_p5a_star_graph,
)


def _make_opt(model: nn.Module, lr: float) -> FairMuonAdamW:
    return FairMuonAdamW(model.io_parameters(), lr=lr, muon_lr_mult=8.0, adam_lr_mult=1.5, weight_decay=0.01)


# ==============================================================================
# Single-Cycle Derangement Depo1 (ZERO 2/4-cycle return-to-start shortcut!)
# ==============================================================================

def sample_p1a_single_cycle_depo1(
    batch_size: int,
    n_nodes: int = 16,
    hops: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Single Hamiltonian cycle permutation of length n_nodes -> pi^k(s) != s for ALL 1 <= k < n_nodes!
    Tokens: 0..n_nodes-1 = nodes, n_nodes = EDGE_SEP, n_nodes+1 = QRY, n_nodes+2.. = HOP_1..HOP_12 (vocab=48).
    """
    rng = np.random.default_rng(seed)
    edge_sep, qry_tok, hop_base = n_nodes, n_nodes + 1, n_nodes + 2
    max_h = max(hops)
    seq_len = n_nodes * 3 + 3
    seqs = np.empty((batch_size, seq_len), dtype=np.int64)
    targets = np.empty(batch_size, dtype=np.int64)
    inter = np.empty((batch_size, max_h), dtype=np.int64)

    for b in range(batch_size):
        # Single full cycle of length n_nodes: zero short cycles!
        cyc = rng.permutation(n_nodes)
        perm = np.empty(n_nodes, dtype=np.int64)
        for i in range(n_nodes):
            perm[cyc[i]] = cyc[(i + 1) % n_nodes]
        edge_order = rng.permutation(n_nodes)
        pos = 0
        for u in edge_order:
            seqs[b, pos] = u
            seqs[b, pos + 1] = perm[u]
            seqs[b, pos + 2] = edge_sep
            pos += 3
        s = int(rng.integers(0, n_nodes))
        k = int(rng.choice(hops))
        cur = s
        for h in range(max_h):
            cur = int(perm[cur])
            inter[b, h] = cur
        seqs[b, pos] = qry_tok
        seqs[b, pos + 1] = s
        seqs[b, pos + 2] = hop_base + (k - 1)
        targets[b] = inter[b, k - 1]

    return {
        "input_ids": torch.from_numpy(seqs),
        "target": torch.from_numpy(targets),
        "inter_targets": torch.from_numpy(inter),
    }


# ==============================================================================
# Part A: Route 3 — FOMP-CMS Stage A Screen on Exact Week-1 P4RuleWorld
# ==============================================================================

def _sample_fomp_episodic_batch(
    world: P4RuleWorld,
    batch_size: int,
    n_eps_per_item: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Sample strictly train-entity-only episodes (0..47) -> zero-overlap ViewA/ViewB 1-hop & 1-Pass 2-hop + unrelated locality."""
    rng = np.random.default_rng(seed)
    ep_ids = np.empty((batch_size, n_eps_per_item, 3), dtype=np.int64)
    q_va = np.empty((batch_size, 2), dtype=np.int64)
    q_vb = np.empty((batch_size, 3), dtype=np.int64)
    q_2h = np.empty((batch_size, 3), dtype=np.int64)
    q_unrel = np.empty((batch_size, 3), dtype=np.int64)
    y_o1 = np.empty(batch_size, dtype=np.int64)
    y_2h = np.empty(batch_size, dtype=np.int64)
    y_unrel = np.empty(batch_size, dtype=np.int64)

    for b in range(batch_size):
        # Pick (n_eps + 1) distinct train_ents: first n_eps are compiled, last 1 is unrelated locality control!
        chosen = rng.choice(world.train_ents, size=n_eps_per_item + 1, replace=False)
        for m in range(n_eps_per_item):
            sm = int(chosen[m])
            old_o1 = int(world.r1_old[sm])
            cands = [e for e in range(world.n_ent) if int(world.r2_map[e]) != int(world.r2_map[old_o1])]
            o1_new = int(rng.choice(cands))
            ep_ids[b, m] = [world.t_rule, sm, o1_new]
            if m == 0:
                q_va[b] = [world.t_qr1, sm]
                q_vb[b] = [world.t_qr1] + world.view_b(sm)
                q_2h[b] = [world.t_q2h] + world.view_b(sm)
                y_o1[b] = o1_new
                y_2h[b] = int(world.r2_map[o1_new])
        su = int(chosen[-1])
        q_unrel[b] = [world.t_q2h] + world.view_b(su)
        y_unrel[b] = int(world.r2_map[int(world.r1_old[su])])

    return {
        "ep_ids": torch.from_numpy(ep_ids),
        "q_va": torch.from_numpy(q_va),
        "q_vb": torch.from_numpy(q_vb),
        "q_2h": torch.from_numpy(q_2h),
        "q_unrel": torch.from_numpy(q_unrel),
        "y_o1": torch.from_numpy(y_o1),
        "y_2h": torch.from_numpy(y_2h),
        "y_unrel": torch.from_numpy(y_unrel),
    }


@torch.no_grad()
def _eval_fomp_cms(
    model: FOMPCMSModel,
    world: P4RuleWorld,
    ev_bg: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_eps: List[Dict[str, Any]],
    device: torch.device,
    amp: bool,
) -> Dict[str, Any]:
    model.eval()
    ev_x, ev_y, ev_m = ev_bg
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        lg_bg0 = model(ev_x.to(device))["logits"].float()
    bg_acc_clean = float(lg_bg0[ev_m.to(device)].argmax(-1).cpu().eq(ev_y[ev_m]).float().mean().item() * 100.0)
    bg_nll_clean = float(F.cross_entropy(lg_bg0[ev_m.to(device)], ev_y.to(device)[ev_m.to(device)]).item())

    def _run_mode(simultaneous: bool) -> Dict[str, Any]:
        va_ok = vb_ok = h2_ok = 0
        va_fl = vb_fl = h2_fl = 0
        if simultaneous:
            ep_all = torch.tensor([[[world.t_rule, ep["s"], ep["o1_new"]] for ep in test_eps]], device=device, dtype=torch.long)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                comp_all = model.compile_episodes(ep_all, g_ext=1.0)
            # Check background locality while ALL 16 unseen test facts are loaded simultaneously!
            comp_bg = {k: (v.expand(ev_x.shape[0], *v.shape[1:]) if v.ndim >= 2 else v) for k, v in comp_all.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg_bg_loaded = model.forward_with_fast_weights(ev_x.to(device), compiled=comp_bg)["logits"].float()
            bg_acc_loaded = float(lg_bg_loaded[ev_m.to(device)].argmax(-1).cpu().eq(ev_y[ev_m]).float().mean().item() * 100.0)
            bg_drift_pp = bg_acc_clean - bg_acc_loaded
        else:
            comp_all = None
            bg_drift_pp = 0.0

        for ep in test_eps:
            if simultaneous:
                comp = comp_all
            else:
                ep_tk = torch.tensor([[[world.t_rule, ep["s"], ep["o1_new"]]]], device=device, dtype=torch.long)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    comp = model.compile_episodes(ep_tk, g_ext=1.0)

            def _q(seq_ids: List[int], sl: slice, tgt_off: int, flip: bool) -> bool:
                tk = torch.tensor([seq_ids], device=device, dtype=torch.long)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg = model.forward_with_fast_weights(tk, compiled=comp, sign_flip=flip)["logits"][0, -1, sl].float()
                return int(lg.argmax(-1).item()) == tgt_off

            va_ok += int(_q(ep["qry_r1_view_a"], slice(0, world.n_ent), ep["o1_new"], False))
            vb_ok += int(_q(ep["qry_r1_view_b"], slice(0, world.n_ent), ep["o1_new"], False))
            h2_ok += int(_q(ep["qry_2hop_view_b"], slice(world.prop_base, world.prop_base + world.n_props), ep["y_new"] - world.prop_base, False))
            va_fl += int(_q(ep["qry_r1_view_a"], slice(0, world.n_ent), ep["o1_new"], True))
            vb_fl += int(_q(ep["qry_r1_view_b"], slice(0, world.n_ent), ep["o1_new"], True))
            h2_fl += int(_q(ep["qry_2hop_view_b"], slice(world.prop_base, world.prop_base + world.n_props), ep["y_new"] - world.prop_base, True))

        n = len(test_eps)
        va_p, vb_p, h2_p = va_ok / n * 100.0, vb_ok / n * 100.0, h2_ok / n * 100.0
        out_m = {
            "view_a_direct_pct": round(va_p, 2),
            "view_b_zero_overlap_1hop_pct": round(vb_p, 2),
            "view_b_1pass_2hop_pct": round(h2_p, 2),
            "sign_flip_view_a": audit_gradient_sign_flip(va_p, va_fl / n * 100.0),
            "sign_flip_view_b": audit_gradient_sign_flip(vb_p, vb_fl / n * 100.0),
            "sign_flip_2hop": audit_gradient_sign_flip(h2_p, h2_fl / n * 100.0),
        }
        if simultaneous:
            out_m["bg_acc_drift_pp_with_16_loaded"] = round(bg_drift_pp, 2)
        return out_m

    # Verify hard external gate g_ext=0 on self-generated episodes produces strictly 0.0000 nats drift!
    sg_ep = torch.tensor([[[world.t_rule, int(test_eps[0]["s"]), int(test_eps[0]["o1_new"])]]], device=device, dtype=torch.long)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        comp_blocked = model.compile_episodes(sg_ep, g_ext=0.0)
        comp_blk_bg = {k: (v.expand(ev_x.shape[0], *v.shape[1:]) if v.ndim >= 2 else v) for k, v in comp_blocked.items()}
        lg_sg = model.forward_with_fast_weights(ev_x.to(device), compiled=comp_blk_bg)["logits"].float()
    sg_drift = float(F.cross_entropy(lg_sg[ev_m.to(device)], ev_y.to(device)[ev_m.to(device)]).item()) - bg_nll_clean

    m1 = _run_mode(simultaneous=False)
    m16 = _run_mode(simultaneous=True)
    return {
        "bg_acc_clean_pct": round(bg_acc_clean, 2),
        "M1_per_episode": m1,
        "M16_simultaneous": m16,
        "gated_selfgen_drift_nats": round(abs(sg_drift), 6),
        "pass_stage_a_gate": bool(
            m1["view_b_1pass_2hop_pct"] >= 75.0
            and m1["sign_flip_2hop"]["pass_sign_flip_check"]
            and abs(sg_drift) < 0.05
            and abs(m16["bg_acc_drift_pp_with_16_loaded"]) <= 1.0
        ),
    }


def run_route3_fomp_cms(out: Path, device: torch.device, amp: bool, smoke: bool, seeds: Tuple[int, ...]) -> Dict[str, Any]:
    dest = out / "route3_fomp_cms.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    world = P4RuleWorld(seed=404)
    ev_bg = world.sample_pretrain_batch(60 if smoke else 400, seed=4101)
    test_eps = world.build_test_episodes(seed=4201)
    bs_bg = 20 if smoke else 80
    bs_ep = 12 if smoke else 48
    steps = 20 if smoke else 1500
    lr = 1.0e-3

    # Verify strict <= TF-Modern parameter parity & zero hidden params!
    cfg = FOMPCMSConfig.micro_test(world.vocab_size) if smoke else FOMPCMSConfig.lab_3m(world.vocab_size)
    m_audit = FOMPCMSModel(cfg)
    tf_ref = build_dual_strong_baselines(world.vocab_size, smoke=smoke, seed=41)["tf_modern"]
    au = audit_zero_hidden_params(m_audit, world.vocab_size, cfg.d_model)
    assert au["pass_zero_hidden_audit"] and m_audit.num_parameters() <= tf_ref.num_parameters(), (
        f"FOMP-CMS audit/parity failed: {m_audit.num_parameters()} vs TF {tf_ref.num_parameters()} | {au}"
    )
    del m_audit, tf_ref

    seed_runs: List[Dict[str, Any]] = []
    for sd in seeds:
        torch.manual_seed(sd)
        model = FOMPCMSModel(cfg).to(device)
        opt = _make_opt(model, lr)
        w_steps = max(5, steps // 10)
        sync(device)
        t0 = time.perf_counter()
        for st in range(1, steps + 1):
            mult = _cosine_lr_multiplier(st - 1, w_steps, steps, min_ratio=0.1)
            for pg in opt.param_groups:
                pg["lr"] = lr * mult
            model.train()
            opt.zero_grad(set_to_none=True)

            # 1) Background world loss (ViewA<->ViewB, r2, train_ents old r1 & 2-hop)
            bx, by, bm = world.sample_pretrain_batch(bs_bg, seed=sd * 100000 + st)
            # 2) Episodic compilation loss ONLY on train_ents (0..47)! Mix M in (1, 4, 8) episodes!
            m_ep = (1, 4, 8)[st % 3]
            eb = _sample_fomp_episodic_batch(world, bs_ep, m_ep, seed=sd * 200000 + st)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg_bg = model(bx.to(device))["logits"].float()
                loss_bg = F.cross_entropy(lg_bg[bm.to(device)], by.to(device)[bm.to(device)])

                comp = model.compile_episodes(eb["ep_ids"].to(device), g_ext=1.0)
                o_va = model.forward_with_fast_weights(eb["q_va"].to(device), compiled=comp)
                o_vb = model.forward_with_fast_weights(eb["q_vb"].to(device), compiled=comp)
                o_2h = model.forward_with_fast_weights(eb["q_2h"].to(device), compiled=comp)
                o_un = model.forward_with_fast_weights(eb["q_unrel"].to(device), compiled=comp)

                y_o1 = eb["y_o1"].to(device)
                y_2h = eb["y_2h"].to(device)
                y_un = eb["y_unrel"].to(device)
                l_va = F.cross_entropy(o_va["logits"][:, -1, : world.n_ent].float(), y_o1)
                l_vb = F.cross_entropy(o_vb["logits"][:, -1, : world.n_ent].float(), y_o1)
                l_2h = F.cross_entropy(o_2h["logits"][:, -1, world.prop_base : world.prop_base + world.n_props].float(), y_2h)
                l_un = F.cross_entropy(o_un["logits"][:, -1, world.prop_base : world.prop_base + world.n_props].float(), y_un)

                # Contrastive familiarity margin: high similarity on target entity, low on unrelated entity!
                sim_pos = o_2h["max_fam_sim"][:, -1]
                sim_neg = o_un["max_fam_sim"][:, -1]
                l_fam = F.relu(0.75 - sim_pos).pow(2).mean() + F.relu(sim_neg - 0.15).pow(2).mean()
                loss = loss_bg + 0.5 * l_va + 0.75 * l_vb + 1.25 * l_2h + 0.5 * l_un + 0.5 * l_fam

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sync(device)
        sec = time.perf_counter() - t0
        ev = _eval_fomp_cms(model, world, ev_bg, test_eps, device, amp)
        ev["seed"] = sd
        ev["train_sec"] = round(sec, 2)
        ev["ms_per_step"] = round(sec * 1000.0 / steps, 2)
        seed_runs.append(ev)
        print(
            f"[Route3-FOMP-CMS | seed={sd}] ({sec:.1f}s, {ev['ms_per_step']}ms/st) "
            f"M1 1-hop={ev['M1_per_episode']['view_b_zero_overlap_1hop_pct']}% "
            f"2-hop={ev['M1_per_episode']['view_b_1pass_2hop_pct']}% "
            f"(flip={ev['M1_per_episode']['sign_flip_2hop']['acc_sign_flipped_pct']}%) | "
            f"M16 2-hop={ev['M16_simultaneous']['view_b_1pass_2hop_pct']}% "
            f"(bg_drift={ev['M16_simultaneous']['bg_acc_drift_pp_with_16_loaded']:+.2f}pp) | "
            f"PASS={ev['pass_stage_a_gate']}",
            flush=True,
        )
        del model, opt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    m1_2h = [r["M1_per_episode"]["view_b_1pass_2hop_pct"] for r in seed_runs]
    m16_2h = [r["M16_simultaneous"]["view_b_1pass_2hop_pct"] for r in seed_runs]
    payload = {
        "audit": au,
        "week1_baselines_ref": {"tf_modern_1pass_2hop_pct": 18.75, "pulse_v7b_ref_1pass_2hop_pct": 18.75},
        "mean_M1_1pass_2hop_pct": round(float(statistics.mean(m1_2h)), 2),
        "std_M1_1pass_2hop_pct": round(float(statistics.pstdev(m1_2h)), 2),
        "mean_M16_1pass_2hop_pct": round(float(statistics.mean(m16_2h)), 2),
        "all_seeds_pass_stage_a": all(r["pass_stage_a_gate"] for r in seed_runs),
        "seeds": seed_runs,
    }
    save_json(dest, payload)
    return payload


# ==============================================================================
# Part B: Route 1 — APRM Stage A Screen (P1a-SC + Grad Horizon + P1c + P2 + P5a)
# ==============================================================================

def _train_one_arm(
    model: nn.Module,
    step_loss_fn: Any,
    eval_fn: Any,
    device: torch.device,
    amp: bool,
    lr: float,
    steps: int,
    eval_every: int,
    tag: str,
    seed: int,
) -> Dict[str, Any]:
    model = model.to(device)
    if hasattr(model, "compile_fast_paths"):
        model.compile_fast_paths()
    opt = _make_opt(model, lr)
    w_steps = max(5, steps // 10)
    sync(device)
    t0 = time.perf_counter()
    best_sc, best_det = -1e9, {}
    consec_hi = 0
    done = 0
    for st in range(1, steps + 1):
        mult = _cosine_lr_multiplier(st - 1, w_steps, steps, min_ratio=0.1)
        for pg in opt.param_groups:
            pg["lr"] = lr * mult
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = step_loss_fn(model, st, steps, seed * 100000 + st)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done = st
        if st % eval_every == 0 or st == steps:
            model.eval()
            sc, det = eval_fn(model)
            if sc > best_sc:
                best_sc, best_det = sc, det
            if sc >= 99.5:
                consec_hi += 1
                if consec_hi >= 2:
                    break
            else:
                consec_hi = 0
    sync(device)
    sec = time.perf_counter() - t0
    model.eval()
    sc_f, det_f = eval_fn(model)
    if sc_f >= best_sc:
        best_sc, best_det = sc_f, det_f
    rec: Dict[str, Any] = {
        "seed": seed,
        "lr": lr,
        "steps": done,
        "train_sec": round(sec, 2),
        "ms_per_step": round(sec * 1000.0 / max(1, done), 2),
        "primary_score": round(best_sc, 2),
        "metrics": best_det,
    }
    if isinstance(model, APRMModel):
        rec["active_slot_entropy_pct"] = round(model.active_slot_entropy_pct, 2)
    print(
        f"[{tag} | seed={seed} lr={lr:.0e}] steps={done} ({sec:.1f}s, {rec['ms_per_step']}ms/st) "
        f"score={best_sc:.2f} {json.dumps(best_det)}"
        + (f" slot_ent={rec['active_slot_entropy_pct']}%" if "active_slot_entropy_pct" in rec else ""),
        flush=True,
    )
    return rec


def run_route1_aprm(
    out: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
    seeds: Tuple[int, ...],
    deadline: float,
) -> Dict[str, Any]:
    dest = out / "route1_aprm.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    bs = 16 if smoke else 64
    cfg_audit = APRMConfig.micro_test(160) if smoke else APRMConfig.lab_3m(160)
    m_audit = APRMModel(cfg_audit)
    tf_ref = build_dual_strong_baselines(160, smoke=smoke, seed=41)["tf_modern"]
    au = audit_zero_hidden_params(m_audit, 160, cfg_audit.d_model)
    assert au["pass_zero_hidden_audit"] and m_audit.num_parameters() <= tf_ref.num_parameters(), (
        f"APRM audit/parity failed: {m_audit.num_parameters()} vs TF {tf_ref.num_parameters()} | {au}"
    )
    del m_audit, tf_ref

    res: Dict[str, Any] = {"audit": au, "b1_p1a_single_cycle_depo1": {}, "b2_p1c_dag_arith": {}, "b3_p2_s5_retrieval": {}, "b4_p5a_star_graph": {}}

    # ==========================================================================
    # B1: P1a-SC Single-Cycle Depo1 (N=16, Fair 2-Stage Curriculum 1..2 -> 1..4 -> 1..8)
    #     + Direct Gradient-Horizon Ablation (Full BPTT vs Last-4 vs 1-Step Implicit)
    # ==========================================================================
    n_nodes = 12 if smoke else 16
    steps_b1 = 24 if smoke else 2000
    ev_every_b1 = max(6, steps_b1 // 5)
    ev_hops = (1, 2, 4, 6, 8, 10) if not smoke else (1, 2, 4, 8)
    ev_sets_b1 = {h: sample_p1a_single_cycle_depo1(32 if smoke else 256, n_nodes=n_nodes, hops=(h,), seed=6100 + h) for h in ev_hops}
    pr_tr = sample_p1a_single_cycle_depo1(48 if smoke else 384, n_nodes=n_nodes, hops=(8,), seed=6201)
    pr_te = sample_p1a_single_cycle_depo1(32 if smoke else 256, n_nodes=n_nodes, hops=(8,), seed=6202)

    def _b1_curriculum_hops(st: int, tot: int) -> Tuple[int, ...]:
        frac = st / max(1, tot)
        if frac <= 0.25:
            return (1, 2)
        if frac <= 0.50:
            return (1, 2, 3, 4)
        return (1, 2, 3, 4, 5, 6, 7, 8)

    def _loss_b1(m: nn.Module, st: int, tot: int, step_seed: int) -> torch.Tensor:
        b = sample_p1a_single_cycle_depo1(bs, n_nodes=n_nodes, hops=_b1_curriculum_hops(st, tot), seed=step_seed)
        if isinstance(m, APRMModel):
            lg = m(b["input_ids"].to(device), use_final_slot_deref=True)["logits"][:, -1, :n_nodes].float()
            return F.cross_entropy(lg, b["target"].to(device)) + m.aux_loss
        lg = m(b["input_ids"].to(device))["logits"][:, -1, :n_nodes].float()
        return F.cross_entropy(lg, b["target"].to(device))

    @torch.no_grad()
    def _eval_b1(m: nn.Module, override_loops: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        by_h = {}
        for h, eb in ev_sets_b1.items():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                if isinstance(m, APRMModel):
                    lg = m.forward_anvil(eb["input_ids"].to(device), n_loops=override_loops, use_final_slot_deref=True)["logits"][:, -1, :n_nodes].float()
                else:
                    lg = m(eb["input_ids"].to(device))["logits"][:, -1, :n_nodes].float()
            by_h[f"hop{h}"] = round(float(lg.argmax(-1).cpu().eq(eb["target"]).float().mean().item() * 100.0), 2)
        return by_h["hop8"], by_h

    def _make_b1_arm(name: str, sd: int) -> nn.Module:
        torch.manual_seed(sd)
        if name in ("tf_modern", "pulse_v7b_ref"):
            return build_dual_strong_baselines(48, smoke=smoke, seed=sd)[name]
        gh = {"aprm_full_bptt": -1, "aprm_trunc4_bptt": 4, "aprm_implicit_1step": 1}[name]
        loops = 4 if smoke else 8
        c = APRMConfig.micro_test(48, loop_steps=loops, grad_horizon=gh) if smoke else APRMConfig.lab_3m(48, loop_steps=loops, grad_horizon=gh)
        return APRMModel(c)

    b1_arms = ("tf_modern", "pulse_v7b_ref", "aprm_full_bptt", "aprm_trunc4_bptt", "aprm_implicit_1step")
    best_aprm_name, best_aprm_sc = "aprm_full_bptt", -1.0
    for arm_name in b1_arms:
        if time.time() >= deadline - 120:
            break
        m = _make_b1_arm(arm_name, seeds[0])
        rec = _train_one_arm(m, _loss_b1, _eval_b1, device, amp, lr=1.5e-3, steps=steps_b1, eval_every=ev_every_b1, tag=f"B1-Depo1SC:{arm_name}", seed=seeds[0])
        # Linear-Probe Hop Ladder across ALL layers / loop steps!
        with torch.no_grad():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                kw = {"use_final_slot_deref": True} if isinstance(m, APRMModel) else {}
                st_tr = [s[:, -1, :].float().cpu() for s in m.forward_anvil(pr_tr["input_ids"].to(device), return_layer_states=True, **kw)["layer_states"]]
                st_te = [s[:, -1, :].float().cpu() for s in m.forward_anvil(pr_te["input_ids"].to(device), return_layer_states=True, **kw)["layer_states"]]
        rec["hop_ladder_audit"] = audit_linear_probe_hop_ladder(st_tr, pr_tr["inter_targets"], st_te, pr_te["inter_targets"], n_classes=n_nodes)
        if isinstance(m, APRMModel):
            # Test-Time Loop Extrapolation (T=8 -> T=12 on unseen hop10!)
            _, det_t12 = _eval_b1(m, override_loops=(6 if smoke else 12))
            rec["test_time_T12_extrap"] = det_t12
            if rec["primary_score"] > best_aprm_sc:
                best_aprm_sc = rec["primary_score"]
                best_aprm_name = arm_name
        res["b1_p1a_single_cycle_depo1"][arm_name] = rec
        del m
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3-Seed Confirmation on winning APRM variant ONLY if seed 41 beat 50% Stage A Kill Switch!
    if len(seeds) > 1 and best_aprm_sc >= 50.0 and time.time() < deadline - 180:
        confirm_runs = [res["b1_p1a_single_cycle_depo1"][best_aprm_name]]
        for sd in seeds[1:]:
            m = _make_b1_arm(best_aprm_name, sd)
            confirm_runs.append(_train_one_arm(m, _loss_b1, _eval_b1, device, amp, lr=1.5e-3, steps=steps_b1, eval_every=ev_every_b1, tag=f"B1-Confirm:{best_aprm_name}", seed=sd))
            del m
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        scs = [r["primary_score"] for r in confirm_runs]
        res["b1_p1a_single_cycle_depo1"]["winner_3seed_confirm"] = {
            "winner_arm": best_aprm_name,
            "mean_hop8": round(float(statistics.mean(scs)), 2),
            "std_hop8": round(float(statistics.pstdev(scs)), 2),
            "runs": confirm_runs,
        }
    save_json(dest, res)

    win_gh = {"aprm_full_bptt": -1, "aprm_trunc4_bptt": 4, "aprm_implicit_1step": 1}[best_aprm_name]
    steps_other = 20 if smoke else 1600
    ev_other = max(5, steps_other // 4)

    # ==========================================================================
    # B2: P1c DAG Arithmetic (vs Week-1 TF/PULSE 11.33%)
    # ==========================================================================
    if time.time() < deadline - 120:
        ev_dag = sample_p1c_dag_arith(48 if smoke else 512, seed=1401)
        def _loss_b2(m: nn.Module, st: int, tot: int, step_seed: int) -> torch.Tensor:
            b = sample_p1c_dag_arith(bs, seed=step_seed)
            lg = m(b["input_ids"].to(device), use_final_slot_deref=True)["logits"][:, -1, 24:40].float()
            return F.cross_entropy(lg, (b["target"] - 24).to(device)) + m.aux_loss  # type: ignore[attr-defined]
        @torch.no_grad()
        def _eval_b2(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = m(ev_dag["input_ids"].to(device), use_final_slot_deref=True)["logits"][:, -1, 24:40].float()
            ok = lg.argmax(-1).cpu().eq(ev_dag["target"] - 24).float()
            dep = ev_dag["depths"]
            det = {"overall": round(float(ok.mean().item() * 100.0), 2)}
            for lo, hi, nm in ((1, 2, "d1_2"), (3, 4, "d3_4"), (5, 9, "d5plus")):
                md = (dep >= lo) & (dep <= hi)
                if md.any():
                    det[nm] = round(float(ok[md].mean().item() * 100.0), 2)
            return det["overall"], det
        torch.manual_seed(seeds[0])
        c2 = APRMConfig.micro_test(48, loop_steps=4, grad_horizon=win_gh) if smoke else APRMConfig.lab_3m(48, loop_steps=6, grad_horizon=win_gh)
        m2 = APRMModel(c2)
        res["b2_p1c_dag_arith"] = {
            "week1_baselines_ref": {"tf_modern": 11.33, "pulse_v7b_ref": 11.33},
            "aprm": _train_one_arm(m2, _loss_b2, _eval_b2, device, amp, lr=1.5e-3, steps=steps_other, eval_every=ev_other, tag="B2-DAG:aprm", seed=seeds[0]),
        }
        del m2
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_json(dest, res)

    # ==========================================================================
    # B3: P2 S5-Retrieval 4x Extrapolation (vs Week-1 TF 21.35% / PULSE 25.00%)
    # ==========================================================================
    if time.time() < deadline - 120:
        l_tr, k_tr, sw_tr = (96, 10, 12) if smoke else (192, 16, 24)
        ev_cfg = [(96, 10, 12, "ID_1x"), (192, 20, 24, "OOD_2x"), (384, 32, 48, "OOD_4x")] if smoke else [
            (192, 16, 24, "ID_1x_K16"), (384, 32, 48, "OOD_2x_K32"), (768, 64, 96, "OOD_4x_K64")
        ]
        ev_sets_p2 = {nm: sample_p2_s5_retrieval(24 if smoke else 128, seq_len=L, n_pairs=K, n_swaps=sw, seed=2100 + i) for i, (L, K, sw, nm) in enumerate(ev_cfg)}
        def _loss_b3(m: nn.Module, st: int, tot: int, step_seed: int) -> torch.Tensor:
            b = sample_p2_s5_retrieval(bs, seq_len=l_tr, n_pairs=k_tr, n_swaps=sw_tr, seed=step_seed)
            # Strict causal AR forward (use_final_slot_deref=False because 3 queries sit at pos -5, -3, -1!)
            lg = m(b["input_ids"].to(device), use_final_slot_deref=False)["logits"].float()
            ar = torch.arange(bs, device=device)
            return (
                F.cross_entropy(lg[ar, b["pos_s5"].to(device), :5], b["y_s5"].to(device))
                + F.cross_entropy(lg[ar, b["pos_kv"].to(device), 64:128], (b["y_kv"] - 64).to(device))
                + F.cross_entropy(lg[ar, b["pos_joint"].to(device), 64:128], (b["y_joint"] - 64).to(device))
                + m.aux_loss  # type: ignore[attr-defined]
            )
        @torch.no_grad()
        def _eval_b3(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            det = {}
            for nm, eb in ev_sets_p2.items():
                n_e = eb["input_ids"].shape[0]
                s5_h = kv_h = jt_h = 0
                for s0 in range(0, n_e, 32):
                    s1 = min(n_e, s0 + 32)
                    ar = torch.arange(s1 - s0, device=device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg = m(eb["input_ids"][s0:s1].to(device), use_final_slot_deref=False)["logits"].float()
                    s5_h += int(lg[ar, eb["pos_s5"][s0:s1].to(device), :5].argmax(-1).cpu().eq(eb["y_s5"][s0:s1]).sum().item())
                    kv_h += int(lg[ar, eb["pos_kv"][s0:s1].to(device), 64:128].argmax(-1).cpu().eq(eb["y_kv"][s0:s1] - 64).sum().item())
                    jt_h += int(lg[ar, eb["pos_joint"][s0:s1].to(device), 64:128].argmax(-1).cpu().eq(eb["y_joint"][s0:s1] - 64).sum().item())
                det[nm] = {"s5": round(s5_h / n_e * 100.0, 2), "kv": round(kv_h / n_e * 100.0, 2), "joint": round(jt_h / n_e * 100.0, 2)}
            return det[ev_cfg[-1][3]]["joint"], det
        torch.manual_seed(seeds[0])
        c3 = APRMConfig.micro_test(160, loop_steps=4, grad_horizon=win_gh) if smoke else APRMConfig.lab_3m(160, loop_steps=4, grad_horizon=win_gh)
        m3 = APRMModel(c3)
        res["b3_p2_s5_retrieval"] = {
            "week1_baselines_ref": {"tf_modern_joint4x": 21.35, "pulse_v7b_ref_joint4x": 25.00},
            "aprm": _train_one_arm(m3, _loss_b3, _eval_b3, device, amp, lr=2.0e-3, steps=steps_other, eval_every=ev_other, tag="B3-S5Ret:aprm", seed=seeds[0]),
        }
        del m3
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        save_json(dest, res)

    # ==========================================================================
    # B4: P5a Star-Graph Lookahead Planning (vs Week-1 TF 7.55% / PULSE 18.88%)
    # ==========================================================================
    if time.time() < deadline - 120:
        ev_star = sample_p5a_star_graph(48 if smoke else 256, degree=5, path_len=4, seed=5101)
        p_len, path_l = ev_star["prefix_len"], ev_star["path_len"]
        def _loss_b4(m: nn.Module, st: int, tot: int, step_seed: int) -> torch.Tensor:
            b = sample_p5a_star_graph(bs, degree=5, path_len=4, seed=step_seed)
            lg = m(b["input_ids"].to(device), use_final_slot_deref=False)["logits"][:, p_len - 1 :, :32].float()
            tgt = b["targets"][:, p_len - 1 :].to(device)
            return F.cross_entropy(lg.reshape(-1, 32), tgt.reshape(-1)) + m.aux_loss  # type: ignore[attr-defined]
        @torch.no_grad()
        def _eval_b4(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            x_in = ev_star["input_ids"].to(device)
            tgt = ev_star["targets"][:, p_len - 1 :]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = m(x_in, use_final_slot_deref=False)["logits"][:, p_len - 1 :, :32].float()
            step1 = float(lg[:, 0].argmax(-1).cpu().eq(tgt[:, 0]).float().mean().item() * 100.0)
            cur = x_in[:, :p_len].clone()
            for _ in range(path_l):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    nxt = m(cur, use_final_slot_deref=False)["logits"][:, -1, :32].argmax(-1, keepdim=True)
                cur = torch.cat([cur, nxt], dim=1)
            full_em = float(cur[:, p_len:].cpu().eq(tgt).all(dim=1).float().mean().item() * 100.0)
            return step1, {"first_step_myopia_acc": round(step1, 2), "full_path_em": round(full_em, 2)}
        torch.manual_seed(seeds[0])
        c4 = APRMConfig.micro_test(48, loop_steps=4, grad_horizon=win_gh) if smoke else APRMConfig.lab_3m(48, loop_steps=6, grad_horizon=win_gh)
        m4 = APRMModel(c4)
        res["b4_p5a_star_graph"] = {
            "week1_baselines_ref": {"tf_modern_step1": 7.55, "pulse_v7b_ref_step1": 18.88},
            "aprm": _train_one_arm(m4, _loss_b4, _eval_b4, device, amp, lr=1.5e-3, steps=steps_other, eval_every=ev_other, tag="B4-StarGraph:aprm", seed=seeds[0]),
        }
        del m4

    # Stage A Verdict Summary
    b1_best = res["b1_p1a_single_cycle_depo1"].get(best_aprm_name, {})
    res["stage_a_verdict"] = {
        "best_aprm_grad_horizon": best_aprm_name,
        "depo1_hop8_pct": b1_best.get("primary_score", 0.0),
        "slot_entropy_pct": b1_best.get("active_slot_entropy_pct", 0.0),
        "monotonic_hop_ladder": b1_best.get("hop_ladder_audit", {}).get("monotonic_ladder_detected", False),
        "pass_depo1_80_gate": bool(b1_best.get("primary_score", 0.0) >= 80.0 and b1_best.get("active_slot_entropy_pct", 0.0) >= 20.0),
        "kill_switch_triggered": bool(b1_best.get("primary_score", 0.0) < 50.0 or b1_best.get("active_slot_entropy_pct", 0.0) < 20.0),
    }
    save_json(dest, res)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--routes", type=str, default="route3,route1")
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--deadline", type=float, default=time.time() + 7200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()

    routes = [r.strip().lower() for r in args.routes.split(",") if r.strip()]
    seeds = (41,) if args.smoke else tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    summary: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sm89_patch": sm89,
        "routes": routes,
        "seeds": list(seeds),
    }
    if "route3" in routes:
        summary["route3_fomp_cms"] = run_route3_fomp_cms(args.out, device, amp, args.smoke, seeds)
        save_json(args.out / "week2_screen_summary.json", summary)
    if "route1" in routes and time.time() < args.deadline - 120:
        summary["route1_aprm"] = run_route1_aprm(args.out, device, amp, args.smoke, seeds, args.deadline)
        save_json(args.out / "week2_screen_summary.json", summary)

    print("\n=== WEEK-2 STAGE A SCREENING COMPLETED ->", args.out / "week2_screen_summary.json", "===", flush=True)


if __name__ == "__main__":
    main()
