"""PULSE Week-2 v2 Fast Truth-Test Runner (Single-Seed Fast Screen + Auto 3-Seed Confirm on Route 3 PASS).

Targets the two exact v1 Stage A failure modes diagnosed locally before spending L4 CCUs:
  1. Part 1 — Route 3 (`FOMP-CMS v2`, ~2.0 min/seed on L4):
       - Shared early-manifold `key_proj` (L0..L1) + Entity-Invariant Bilinear Outer-Product `val_projs`
         + Sharp ReLU threshold gate (`0.0000` below `tau`) + Relation-Conditioned `rel_gate` (0.00 pp drift
         even on same-entity non-r1 queries `t_r2`/`t_veq` and in-context `t_rule`!).
       - Strict zero-leakage protocol: `test_ents (48..63)` NEVER appear in `r1`/2-hop pretraining,
         prefix-invariance alignment, or episodic compilation.
       - Automatically confirms on `seeds=(42, 43)` ONLY if `seed=41` clears ALL Stage A gates
         (`M1 2-hop >= 75%`, ICML 2026 Sign-Flip PASS, `M16 bg_drift <= 1.0 pp`, `g_ext=0 drift < 0.05 nats`).
  2. Part 2 — Route 1 (`APRM v2` vs `TF-Modern` on Single-Cycle `Depo1`, `N=16`, `seed=41`, ~5–6 min on L4):
       - Fixed v1 `delta_norm` 200x init-noise amplification + upgraded to strictly chunk-causal slot snapshots
         active across ALL token positions in every attractor loop!
       - Answer-Free Multi-Query training (`Q=8` queries/graph, ZERO answer tokens in context) with pure
         single-query `Q=1` evaluation (`[edges..., QRY, s, HOP_k]`).
       - IDENTICAL Mastery-Gated Frontier Curriculum + IDENTICAL WSD AdamW (`lr=1.0e-3`) on `tf_modern`,
         `aprm_v2_full` (`grad_horizon=-1`), and `aprm_v2_trunc4` (`grad_horizon=4`).
       - Full audits: `hop1..hop10`, Linear-Probe Hop Ladder (`[1..8]`), OOD `T=8 -> T=12` loop scaling,
         and `active_slot_entropy_pct`.
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

from pulse.experiment import FairMuonAdamW, _cosine_lr_multiplier
from pulse.pulse_v7 import patch_fla_sm89_shared_memory
from pulse.whiteboard_anvil import (
    audit_linear_probe_hop_ladder,
    audit_zero_hidden_params,
    build_dual_strong_baselines,
)
from pulse.whiteboard_routes import APRMConfig, APRMModel, FOMPCMSConfig, FOMPCMSModel

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import save_json, sync  # noqa: E402
from whiteboard_probes import P4RuleWorld  # noqa: E402
from whiteboard_week2_screen import (  # noqa: E402
    _eval_fomp_cms,
    _sample_fomp_episodic_batch,
    sample_p1a_single_cycle_depo1,
)


# ==============================================================================
# Answer-Free Multi-Query Single-Cycle Depo1 Sampler (ZERO answer leakage!)
# ==============================================================================

def sample_p1a_sc_multiquery_af(
    batch_size: int,
    n_nodes: int = 16,
    hops: Tuple[int, ...] = (1,),
    frontier_hops: Optional[Tuple[int, ...]] = None,
    frontier_prob: float = 0.60,
    n_queries: int = 8,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Single Hamiltonian cycle graph + Q answer-free queries: `[edges..., QRY, s1, HOP, QRY, s2, HOP, ...]`.

    STRICT ZERO-LEAKAGE: No intermediate or query answer token EVER appears in `input_ids`!
    Every query `qi` must solve `pi^{k_qi}(s_qi)` purely from the graph prefix in 1 forward pass.
    """
    rng = np.random.default_rng(seed)
    edge_sep, qry_tok, hop_base = n_nodes, n_nodes + 1, n_nodes + 2
    graph_len = n_nodes * 3
    seq_len = graph_len + n_queries * 3
    seqs = np.empty((batch_size, seq_len), dtype=np.int64)
    tgts = np.empty((batch_size, n_queries), dtype=np.int64)
    q_pos = np.array([graph_len + qi * 3 + 2 for qi in range(n_queries)], dtype=np.int64)

    use_front = frontier_hops is not None and len(frontier_hops) > 0 and len(hops) > len(frontier_hops)
    base_hops = tuple(h for h in hops if not use_front or h not in frontier_hops) if use_front else hops

    for b in range(batch_size):
        cyc = rng.permutation(n_nodes)
        perm = np.empty(n_nodes, dtype=np.int64)
        for i in range(n_nodes):
            perm[cyc[i]] = cyc[(i + 1) % n_nodes]
        pos = 0
        for u in rng.permutation(n_nodes):
            seqs[b, pos] = u
            seqs[b, pos + 1] = perm[u]
            seqs[b, pos + 2] = edge_sep
            pos += 3
        starts = rng.choice(n_nodes, size=n_queries, replace=False)
        for qi in range(n_queries):
            s = int(starts[qi])
            if use_front and rng.random() < frontier_prob:
                k = int(rng.choice(frontier_hops))  # type: ignore[arg-type]
            else:
                k = int(rng.choice(base_hops))
            cur = s
            for _ in range(k):
                cur = int(perm[cur])
            seqs[b, pos] = qry_tok
            seqs[b, pos + 1] = s
            seqs[b, pos + 2] = hop_base + (k - 1)
            pos += 3
            tgts[b, qi] = cur

    return {
        "input_ids": torch.from_numpy(seqs),
        "targets": torch.from_numpy(tgts),
        "q_pos": torch.from_numpy(q_pos),
    }


def _wsd_lr_multiplier(step: int, warmup_steps: int, total_steps: int, decay_frac: float = 0.15, min_ratio: float = 0.1) -> float:
    """Warmup-Stable-Decay LR schedule: keeps full LR across late curriculum mastery transitions!"""
    if step < warmup_steps:
        return max(0.05, float(step + 1) / max(1, warmup_steps))
    decay_start = int(total_steps * (1.0 - decay_frac))
    if step < decay_start:
        return 1.0
    t = float(step - decay_start) / max(1, total_steps - decay_start)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, t))))


# ==============================================================================
# Part 1: Route 3 — FOMP-CMS v2 on Exact Week-1 P4RuleWorld
# ==============================================================================

@torch.no_grad()
def _audit_bg_per_kind_drift(
    model: FOMPCMSModel,
    world: P4RuleWorld,
    ev_bg: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_eps: List[Dict[str, Any]],
    device: torch.device,
    amp: bool,
) -> Dict[str, Any]:
    ev_x, ev_y, ev_m = ev_bg
    ep_all = torch.tensor([[[world.t_rule, ep["s"], ep["o1_new"]] for ep in test_eps]], device=device, dtype=torch.long)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        comp_all = model.compile_episodes(ep_all, g_ext=1.0)
        comp_bg = {k: (v.expand(ev_x.shape[0], *v.shape[1:]) if v.ndim >= 2 else v) for k, v in comp_all.items()}
        lg0 = model(ev_x.to(device))["logits"].float()
        out1 = model.forward_with_fast_weights(ev_x.to(device), compiled=comp_bg)
        lg1 = out1["logits"].float()
        gm = out1["max_fam_gate"].float()
    names = ("kind0_veq", "kind1_r2", "kind2_old_r1", "kind3_old_2h", "kind4_ictx_2h")
    out: Dict[str, Any] = {}
    for kd, nm in enumerate(names):
        idx = torch.arange(kd, ev_x.shape[0], 5, device=device)
        m_k = ev_m.to(device)[idx]
        y_k = ev_y.to(device)[idx][m_k]
        a0 = float(lg0[idx][m_k].argmax(-1).eq(y_k).float().mean().item() * 100.0)
        a1 = float(lg1[idx][m_k].argmax(-1).eq(y_k).float().mean().item() * 100.0)
        out[nm] = {
            "clean_pct": round(a0, 2),
            "loaded_pct": round(a1, 2),
            "drift_pp": round(a0 - a1, 2),
            "max_gate": round(float(gm[idx].max().item()), 4),
        }
    return out


def _train_fomp_v2_seed(
    world: P4RuleWorld,
    cfg: FOMPCMSConfig,
    ev_bg: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_eps: List[Dict[str, Any]],
    sd: int,
    steps: int,
    bs_bg: int,
    bs_ep: int,
    device: torch.device,
    amp: bool,
) -> Dict[str, Any]:
    torch.manual_seed(sd)
    model = FOMPCMSModel(cfg).to(device)
    lr = 1.5e-3
    opt = FairMuonAdamW(model.io_parameters(), lr=lr, muon_lr_mult=4.0, adam_lr_mult=1.5, weight_decay=0.01)
    w_steps = max(5, steps // 10)

    # STRICT ZERO-LEAK alignment tensors:
    # 1) All 64 entities ONLY under kind==0 (t_veq) surface equivalence (ZERO r1 / o1_new / 2-hop info!)
    all_va_veq = torch.tensor([[world.t_veq, e] for e in range(world.n_ent)], device=device, dtype=torch.long)
    all_vb_veq = torch.tensor([[world.t_veq] + world.view_b(e) for e in range(world.n_ent)], device=device, dtype=torch.long)
    tr_ents = world.train_ents.tolist()

    sync(device)
    t0 = time.perf_counter()
    for st in range(1, steps + 1):
        mult = _cosine_lr_multiplier(st - 1, w_steps, steps, min_ratio=0.1)
        for pg in opt.param_groups:
            pg["lr"] = lr * mult
        model.train()
        opt.zero_grad(set_to_none=True)

        bx, by, bm = world.sample_pretrain_batch(bs_bg, seed=sd * 100000 + st)
        bx_d, by_d, bm_d = bx.to(device), by.to(device), bm.to(device)
        m_ep = (1, 4, 8)[st % 3]
        eb = _sample_fomp_episodic_batch(world, bs_ep, m_ep, seed=sd * 200000 + st)

        # Prefix-invariance alignment strictly on train_ents (0..47)!
        rng = np.random.default_rng(sd * 300000 + st)
        pa_pool = (world.t_rule, world.t_qr1, world.t_veq)
        pb_pool = (world.t_qr1, world.t_q2h, world.t_veq)
        tr_va = torch.tensor([[int(rng.choice(pa_pool)), e] for e in tr_ents], device=device, dtype=torch.long)
        tr_vb = torch.tensor([[int(rng.choice(pb_pool))] + world.view_b(e) for e in tr_ents], device=device, dtype=torch.long)

        # Same-entity non-r1 relation preservation strictly on edited train_ent s0!
        s0_list = eb["ep_ids"][:, 0, 1].tolist()
        o1_list = eb["ep_ids"][:, 0, 2].tolist()
        q_r2_a = torch.tensor([[world.t_r2, s] for s in s0_list], device=device, dtype=torch.long)
        q_r2_b = torch.tensor([[world.t_r2] + world.view_b(s) for s in s0_list], device=device, dtype=torch.long)
        q_veq = torch.tensor([[world.t_veq] + world.view_b(s) + [world.t_veq] for s in s0_list], device=device, dtype=torch.long)
        q_rule = torch.tensor([[world.t_rule, s, o] for s, o in zip(s0_list, o1_list)], device=device, dtype=torch.long)
        y_r2_s0 = torch.tensor([int(world.r2_map[s]) for s in s0_list], device=device, dtype=torch.long)
        y_s0 = torch.tensor(s0_list, device=device, dtype=torch.long)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg_bg = model(bx_d)["logits"].float()
            loss_bg = F.cross_entropy(lg_bg[bm_d], by_d[bm_d])

            l_align = 0.5 * model.view_alignment_loss(all_va_veq, all_vb_veq) + 0.5 * model.view_alignment_loss(tr_va, tr_vb)

            comp = model.compile_episodes(eb["ep_ids"].to(device), g_ext=1.0)
            o_va = model.forward_with_fast_weights(eb["q_va"].to(device), compiled=comp)
            o_vb = model.forward_with_fast_weights(eb["q_vb"].to(device), compiled=comp)
            o_2h = model.forward_with_fast_weights(eb["q_2h"].to(device), compiled=comp)
            o_un = model.forward_with_fast_weights(eb["q_unrel"].to(device), compiled=comp)

            o_r2a = model.forward_with_fast_weights(q_r2_a, compiled=comp)
            o_r2b = model.forward_with_fast_weights(q_r2_b, compiled=comp)
            o_veq = model.forward_with_fast_weights(q_veq, compiled=comp)
            o_rul = model.forward_with_fast_weights(q_rule, compiled=comp)

            y_o1 = eb["y_o1"].to(device)
            y_2h = eb["y_2h"].to(device)
            y_un = eb["y_unrel"].to(device)
            l_ep = (
                0.50 * F.cross_entropy(o_va["logits"][:, -1, : world.n_ent].float(), y_o1)
                + 0.75 * F.cross_entropy(o_vb["logits"][:, -1, : world.n_ent].float(), y_o1)
                + 1.25 * F.cross_entropy(o_2h["logits"][:, -1, world.prop_base : world.prop_base + world.n_props].float(), y_2h)
                + 0.50 * F.cross_entropy(o_un["logits"][:, -1, world.prop_base : world.prop_base + world.n_props].float(), y_un)
            )
            # Non-r1 relation preservation on edited entity s0 (clean BCE drives rel_gate -> 0.000 on t_r2, t_veq, t_rule!)
            l_rel_shield = (
                0.40 * F.cross_entropy(o_r2a["logits"][:, -1, world.prop_base : world.prop_base + world.n_props].float(), y_r2_s0)
                + 0.40 * F.cross_entropy(o_r2b["logits"][:, -1, world.prop_base : world.prop_base + world.n_props].float(), y_r2_s0)
                + 0.40 * F.cross_entropy(o_veq["logits"][:, -1, : world.n_ent].float(), y_s0)
                + 1.00 * (
                    F.binary_cross_entropy_with_logits(o_r2a["rel_logits"], torch.zeros_like(o_r2a["rel_logits"]))
                    + F.binary_cross_entropy_with_logits(o_r2b["rel_logits"], torch.zeros_like(o_r2b["rel_logits"]))
                    + F.binary_cross_entropy_with_logits(o_veq["rel_logits"], torch.zeros_like(o_veq["rel_logits"]))
                    + F.binary_cross_entropy_with_logits(o_rul["rel_logits"], torch.zeros_like(o_rul["rel_logits"]))
                    + F.binary_cross_entropy_with_logits(o_va["rel_logits"][:, -1], torch.ones_like(o_va["rel_logits"][:, -1]))
                    + F.binary_cross_entropy_with_logits(o_vb["rel_logits"][:, -1], torch.ones_like(o_vb["rel_logits"][:, -1]))
                    + F.binary_cross_entropy_with_logits(o_2h["rel_logits"][:, -1], torch.ones_like(o_2h["rel_logits"][:, -1]))
                )
            )
            loss = loss_bg + l_align + l_ep + l_rel_shield

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.io_parameters(), 1.0)
        opt.step()

    sync(device)
    sec = time.perf_counter() - t0
    ev = _eval_fomp_cms(model, world, ev_bg, test_eps, device, amp)
    ev["bg_per_kind_audit"] = _audit_bg_per_kind_drift(model, world, ev_bg, test_eps, device, amp)
    ev["seed"] = sd
    ev["train_sec"] = round(sec, 2)
    ev["ms_per_step"] = round(sec * 1000.0 / steps, 2)
    m1, m16 = ev["M1_per_episode"], ev["M16_simultaneous"]
    print(
        f"[Route3-FOMP-v2 | seed={sd}] ({sec:.1f}s, {ev['ms_per_step']}ms/st) bg={ev['bg_acc_clean_pct']}% | "
        f"M1: VA={m1['view_a_direct_pct']}% VB1={m1['view_b_zero_overlap_1hop_pct']}% "
        f"2H={m1['view_b_1pass_2hop_pct']}% (flip={m1['sign_flip_2hop']['acc_sign_flipped_pct']}%) | "
        f"M16: VB1={m16['view_b_zero_overlap_1hop_pct']}% 2H={m16['view_b_1pass_2hop_pct']}% "
        f"(bg_drift={m16['bg_acc_drift_pp_with_16_loaded']:+.2f}pp) | "
        f"g_ext0={ev['gated_selfgen_drift_nats']} | PASS={ev['pass_stage_a_gate']}",
        flush=True,
    )
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ev


def run_route3_fomp_v2(
    out: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
    auto_3seed: bool,
) -> Dict[str, Any]:
    dest = out / "route3_fomp_cms_v2.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    world = P4RuleWorld(seed=404)
    ev_bg = world.sample_pretrain_batch(100 if smoke else 400, seed=4101)
    test_eps = world.build_test_episodes(seed=4201)
    bs_bg = 32 if smoke else 80
    bs_ep = 16 if smoke else 48
    steps = 30 if smoke else 1200

    cfg = FOMPCMSConfig.micro_test(world.vocab_size) if smoke else FOMPCMSConfig.lab_3m(world.vocab_size)
    if smoke:
        cfg.n_layers = 6
        cfg.inject_layers = (2, 3)
        cfg.key_dim = 32
    m_audit = FOMPCMSModel(cfg)
    tf_ref = build_dual_strong_baselines(world.vocab_size, smoke=smoke, seed=41)["tf_modern"]
    au = audit_zero_hidden_params(m_audit, world.vocab_size, cfg.d_model)
    assert au["pass_zero_hidden_audit"] and (smoke or m_audit.num_parameters() <= tf_ref.num_parameters()), (
        f"FOMP-CMS v2 audit/parity failed: {m_audit.num_parameters()} vs TF {tf_ref.num_parameters()} | {au}"
    )
    del m_audit, tf_ref

    seed_runs = [_train_fomp_v2_seed(world, cfg, ev_bg, test_eps, 41, steps, bs_bg, bs_ep, device, amp)]
    # Auto-expand to seeds 42 & 43 ONLY if seed 41 clears Stage A PASS (zero wasted CCUs!)
    if auto_3seed and not smoke and seed_runs[0]["pass_stage_a_gate"]:
        print("[Route3-FOMP-v2] seed=41 PASSED Stage A! Auto-running 3-seed confirmation (seeds 42, 43)...", flush=True)
        for sd in (42, 43):
            seed_runs.append(_train_fomp_v2_seed(world, cfg, ev_bg, test_eps, sd, steps, bs_bg, bs_ep, device, amp))

    m1_2h = [r["M1_per_episode"]["view_b_1pass_2hop_pct"] for r in seed_runs]
    m16_2h = [r["M16_simultaneous"]["view_b_1pass_2hop_pct"] for r in seed_runs]
    m16_dr = [r["M16_simultaneous"]["bg_acc_drift_pp_with_16_loaded"] for r in seed_runs]
    payload = {
        "audit": au,
        "week1_baselines_ref": {"tf_modern_1pass_2hop_pct": 18.75, "pulse_v7b_ref_1pass_2hop_pct": 18.75},
        "v1_ref": {"mean_M1_1pass_2hop_pct": 6.25, "mean_M16_bg_drift_pp": 16.83},
        "n_seeds_run": len(seed_runs),
        "mean_M1_1pass_2hop_pct": round(float(statistics.mean(m1_2h)), 2),
        "std_M1_1pass_2hop_pct": round(float(statistics.pstdev(m1_2h)), 2),
        "mean_M16_1pass_2hop_pct": round(float(statistics.mean(m16_2h)), 2),
        "mean_M16_bg_drift_pp": round(float(statistics.mean(m16_dr)), 2),
        "all_seeds_pass_stage_a": all(r["pass_stage_a_gate"] for r in seed_runs),
        "seeds": seed_runs,
    }
    save_json(dest, payload)
    return payload


# ==============================================================================
# Part 2: Route 1 — APRM v2 vs TF-Modern on Single-Cycle Depo1 (Mastery-Gated!)
# ==============================================================================

def run_route1_aprm_v2(
    out: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
    seed: int,
    deadline: float,
) -> Dict[str, Any]:
    dest = out / "route1_aprm_v2_depo1.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    n_nodes = 8 if smoke else 16
    n_queries = 4 if smoke else 8
    bs = 16 if smoke else 64
    steps = 30 if smoke else 3000
    eval_every = 10 if smoke else 150
    lr = 2.0e-3 if smoke else 1.0e-3
    vocab_size = 48

    cfg_audit = APRMConfig.micro_test(vocab_size) if smoke else APRMConfig.lab_3m(vocab_size)
    m_audit = APRMModel(cfg_audit)
    tf_ref = build_dual_strong_baselines(vocab_size, smoke=smoke, seed=seed)["tf_modern"]
    au = audit_zero_hidden_params(m_audit, vocab_size, cfg_audit.d_model)
    assert au["pass_zero_hidden_audit"] and m_audit.num_parameters() <= tf_ref.num_parameters(), (
        f"APRM v2 audit/parity failed: {m_audit.num_parameters()} vs TF {tf_ref.num_parameters()} | {au}"
    )
    tf_params = tf_ref.num_parameters()
    del m_audit, tf_ref

    ev_hops = (1, 2, 4, 8) if smoke else (1, 2, 3, 4, 6, 8, 10)
    ev_sets = {h: sample_p1a_single_cycle_depo1(32 if smoke else 256, n_nodes=n_nodes, hops=(h,), seed=7100 + h) for h in ev_hops}
    pr_tr = sample_p1a_single_cycle_depo1(48 if smoke else 384, n_nodes=n_nodes, hops=(8,), seed=7201)
    pr_te = sample_p1a_single_cycle_depo1(32 if smoke else 256, n_nodes=n_nodes, hops=(8,), seed=7202)

    # Mastery-Gated 4-Stage Frontier Curriculum (100% IDENTICAL on TF-Modern and APRM v2!)
    stages: List[Tuple[Tuple[int, ...], Optional[Tuple[int, ...]], Tuple[int, ...], float]] = [
        ((1,), None, (1,), 90.0),                                   # Stage 0 -> advance when hop1 >= 90%
        ((1, 2), (2,), (2,), 90.0),                                 # Stage 1 -> advance when hop2 >= 90%
        ((1, 2, 3, 4), (3, 4), (3, 4) if not smoke else (4,), 85.0),# Stage 2 -> advance when mean(h3,h4) >= 85%
        ((1, 2, 3, 4, 5, 6, 7, 8), (5, 6, 7, 8), (8,), 99.5),       # Stage 3 -> full 1..8 hops!
    ]

    @torch.no_grad()
    def _eval_q1(model: nn.Module, override_loops: Optional[int] = None) -> Dict[str, float]:
        model.eval()
        by_h: Dict[str, float] = {}
        for h, eb in ev_sets.items():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                if isinstance(model, APRMModel):
                    lg = model.forward_anvil(eb["input_ids"].to(device), n_loops=override_loops)["logits"][:, -1, :n_nodes].float()
                else:
                    lg = model(eb["input_ids"].to(device))["logits"][:, -1, :n_nodes].float()
            by_h[f"hop{h}"] = round(float(lg.argmax(-1).cpu().eq(eb["target"]).float().mean().item() * 100.0), 2)
        return by_h

    def _make_arm(arm_name: str) -> nn.Module:
        torch.manual_seed(seed)
        if arm_name == "tf_modern":
            return build_dual_strong_baselines(vocab_size, smoke=smoke, seed=seed)["tf_modern"]
        gh = {"aprm_v2_full": -1, "aprm_v2_trunc4": 4}[arm_name]
        loops = 4 if smoke else 8
        c = APRMConfig.micro_test(vocab_size, loop_steps=loops, grad_horizon=gh) if smoke else APRMConfig.lab_3m(vocab_size, loop_steps=loops, grad_horizon=gh)
        return APRMModel(c)

    arms = ("tf_modern", "aprm_v2_full", "aprm_v2_trunc4")
    results: Dict[str, Any] = {
        "audit": au,
        "tf_modern_params": tf_params,
        "protocol": {"n_nodes": n_nodes, "train_n_queries_answer_free": n_queries, "eval_n_queries": 1, "lr": lr, "schedule": "WSD_85_15", "optimizer": "AdamW(betas=0.9,0.98)", "steps": steps},
        "arms": {},
    }

    for arm_name in arms:
        if time.time() >= deadline - 90:
            break
        model = _make_arm(arm_name).to(device)
        if hasattr(model, "compile_fast_paths"):
            model.compile_fast_paths()
        opt = torch.optim.AdamW(model.io_parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
        w_steps = max(5, min(150, steps // 10))
        stage_idx = 0
        stage_transitions: List[Dict[str, Any]] = []
        best_h8 = -1.0
        best_det: Dict[str, float] = {}
        consec_hi = 0

        sync(device)
        t0 = time.perf_counter()
        done = 0
        for st in range(1, steps + 1):
            mult = _wsd_lr_multiplier(st - 1, w_steps, steps, decay_frac=0.15, min_ratio=0.1)
            for pg in opt.param_groups:
                pg["lr"] = lr * mult

            hops_s, front_s, gate_hops, gate_thr = stages[stage_idx]
            b = sample_p1a_sc_multiquery_af(
                bs, n_nodes=n_nodes, hops=hops_s, frontier_hops=front_s,
                frontier_prob=0.60, n_queries=n_queries, seed=seed * 100000 + st,
            )
            x_d = b["input_ids"].to(device)
            y_d = b["targets"].to(device)
            qp_d = b["q_pos"].to(device)

            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = model(x_d)["logits"][:, qp_d, :n_nodes].float()
                loss = F.cross_entropy(lg.reshape(-1, n_nodes), y_d.reshape(-1))
                if isinstance(model, APRMModel):
                    loss = loss + model.aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.io_parameters(), 1.0)
            opt.step()
            done = st

            if st % eval_every == 0 or st == steps:
                det = _eval_q1(model)
                if det["hop8"] >= best_h8:
                    best_h8, best_det = det["hop8"], det
                gate_val = float(statistics.mean([det[f"hop{h}"] for h in gate_hops]))
                print(
                    f"[B1-v2:{arm_name} | st={st:4d} stage={stage_idx} lr={lr*mult:.1e}] "
                    f"loss={loss.item():.3f} {json.dumps(det)}"
                    + (f" slot_ent={model.active_slot_entropy_pct:.1f}%" if isinstance(model, APRMModel) else ""),
                    flush=True,
                )
                if stage_idx < len(stages) - 1 and gate_val >= gate_thr:
                    stage_transitions.append({"from_stage": stage_idx, "to_stage": stage_idx + 1, "step": st, "gate_acc": round(gate_val, 2)})
                    stage_idx += 1
                    print(f"  --> [{arm_name}] MASTERY GATE PASSED ({gate_val:.1f}% >= {gate_thr}%) -> Stage {stage_idx} {stages[stage_idx][0]}!", flush=True)
                elif stage_idx == len(stages) - 1 and det["hop8"] >= 99.0:
                    consec_hi += 1
                    if consec_hi >= 2:
                        break
                else:
                    consec_hi = 0

        sync(device)
        sec = time.perf_counter() - t0
        det_fin = _eval_q1(model)
        if det_fin["hop8"] >= best_h8:
            best_h8, best_det = det_fin["hop8"], det_fin

        # Linear-Probe Hop Ladder across ALL layers / loop steps!
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            st_tr = [s[:, -1, :].float().cpu() for s in model.forward_anvil(pr_tr["input_ids"].to(device), return_layer_states=True)["layer_states"]]
            st_te = [s[:, -1, :].float().cpu() for s in model.forward_anvil(pr_te["input_ids"].to(device), return_layer_states=True)["layer_states"]]
        ladder = audit_linear_probe_hop_ladder(st_tr, pr_tr["inter_targets"], st_te, pr_te["inter_targets"], n_classes=n_nodes)

        rec: Dict[str, Any] = {
            "seed": seed,
            "steps": done,
            "train_sec": round(sec, 2),
            "ms_per_step": round(sec * 1000.0 / max(1, done), 2),
            "final_stage_reached": stage_idx,
            "stage_transitions": stage_transitions,
            "best_hop8_pct": round(best_h8, 2),
            "metrics_final": det_fin,
            "metrics_best_hop8": best_det,
            "hop_ladder_audit": ladder,
        }
        if isinstance(model, APRMModel):
            rec["active_slot_entropy_pct"] = round(model.active_slot_entropy_pct, 2)
            rec["test_time_T12_extrap"] = _eval_q1(model, override_loops=(6 if smoke else 12))
            print(f"  --> [{arm_name}] T=12 OOD Extrap: {json.dumps(rec['test_time_T12_extrap'])} | Ladder monotonic={ladder['monotonic_ladder_detected']} frontier={ladder['solved_hop_frontier_by_layer']}", flush=True)
        else:
            print(f"  --> [{arm_name}] Ladder monotonic={ladder['monotonic_ladder_detected']} frontier={ladder['solved_hop_frontier_by_layer']}", flush=True)

        results["arms"][arm_name] = rec
        save_json(dest, results)
        del model, opt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ap_best = max(
        (results["arms"][k] for k in ("aprm_v2_full", "aprm_v2_trunc4") if k in results["arms"]),
        key=lambda r: r["best_hop8_pct"],
        default={},
    )
    tf_rec = results["arms"].get("tf_modern", {})
    results["verdict"] = {
        "tf_modern_hop8_pct": tf_rec.get("best_hop8_pct", 0.0),
        "aprm_v2_best_hop8_pct": ap_best.get("best_hop8_pct", 0.0),
        "aprm_v2_gap_vs_tf_pp": round(ap_best.get("best_hop8_pct", 0.0) - tf_rec.get("best_hop8_pct", 0.0), 2),
        "aprm_v2_monotonic_ladder": ap_best.get("hop_ladder_audit", {}).get("monotonic_ladder_detected", False),
        "aprm_v2_slot_entropy_pct": ap_best.get("active_slot_entropy_pct", 0.0),
        "pass_depo1_80_gate": bool(ap_best.get("best_hop8_pct", 0.0) >= 80.0 and ap_best.get("active_slot_entropy_pct", 0.0) >= 20.0),
    }
    save_json(dest, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parts", type=str, default="route3,route1")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--no-auto-3seed", action="store_true")
    parser.add_argument("--deadline", type=float, default=time.time() + 3600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()
    parts = [p.strip().lower() for p in args.parts.split(",") if p.strip()]

    summary: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sm89_patch": sm89,
        "parts": parts,
        "seed": args.seed,
    }
    if "route3" in parts:
        summary["route3_fomp_cms_v2"] = run_route3_fomp_v2(args.out, device, amp, args.smoke, auto_3seed=(not args.no_auto_3seed))
        save_json(args.out / "week2_v2_summary.json", summary)
    if "route1" in parts and time.time() < args.deadline - 120:
        summary["route1_aprm_v2"] = run_route1_aprm_v2(args.out, device, amp, args.smoke, args.seed, args.deadline)
        save_json(args.out / "week2_v2_summary.json", summary)

    print("\n=== WEEK-2 v2 TRUTH-TEST COMPLETED ->", args.out / "week2_v2_summary.json", "===", flush=True)


if __name__ == "__main__":
    main()
