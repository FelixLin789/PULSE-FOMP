"""Unit & Anti-Shortcut Verification Suite for Week-1 PULSE Whiteboard Anvil."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from pulse.whiteboard_anvil import (
    audit_boundary_histogram,
    audit_gradient_sign_flip,
    audit_linear_probe_hop_ladder,
    audit_zero_hidden_params,
    build_dual_strong_baselines,
    estimate_exact_step_flops,
)

_BENCH = str(Path(__file__).resolve().parents[1] / "benchmarks")
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)
import whiteboard_probes as wp  # noqa: E402


def test_dual_strong_baselines_parity_bf16_and_zero_hidden_audit() -> None:
    # Full 3.7M lab configs: PULSE <= TF, 0 hidden params, 0 leaked gates/embs in Muon
    lab = build_dual_strong_baselines(256, smoke=False, seed=41)
    assert lab["pulse_v7b_ref"].num_parameters() <= lab["tf_modern"].num_parameters()
    for name, m in lab.items():
        au = audit_zero_hidden_params(m, 256, int(m.config.d_model))
        assert au["pass_zero_hidden_audit"], f"{name} failed audit: {au}"
        fl = estimate_exact_step_flops(m, 64, 192)
        assert fl["step_gflops"] > 0.0 and fl["mix_share_pct"] > 0.0

    # Micro BF16 autocast forward + backward + layer_states hook check
    micro = build_dual_strong_baselines(96, smoke=True, seed=41)
    tok = torch.randint(0, 96, (2, 32))
    for name, m in micro.items():
        m.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m.forward_anvil(tok, return_layer_states=True)
            loss = out["logits"].float().pow(2).mean() + out["h_fin"].float().pow(2).mean()
        assert len(out["layer_states"]) == m.config.n_layers
        loss.backward()
        grads = [p.grad for p in m.io_parameters() if p.grad is not None]
        assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads), f"Non-finite grad in {name}"


def test_all_five_probes_anti_shortcut_invariants() -> None:
    # P1a Depo1: inter_targets[:, k-1] == target exactly
    b1a = wp.sample_p1a_depo1(16, n_nodes=16, hops=(1, 2, 4, 8), seed=1)
    for i in range(16):
        k = int(b1a["hops"][i].item())
        assert int(b1a["inter_targets"][i, k - 1].item()) == int(b1a["target"][i].item())

    # P1b Depo2: answer mask non-empty and codes strictly in 4-symbol alphabet {0..3} + sep 5
    b1b = wp.sample_p1b_depo2(8, n_nodes=8, hops=(2, 4), seq_len=176, seed=2)
    assert b1b["ans_mask"].any(dim=1).all()
    ans_toks = b1b["targets"][b1b["ans_mask"]]
    assert set(ans_toks.tolist()).issubset({0, 1, 2, 3, 5})

    # P1c DAG Arithmetic: targets strictly in 24..39
    b1c = wp.sample_p1c_dag_arith(16, seed=3)
    assert ((b1c["target"] >= 24) & (b1c["target"] < 40)).all()

    # P2 S5-Retrieval: KV pairs end strictly >32 tokens before queries (blocks SWA-32 shortcut!)
    b2 = wp.sample_p2_s5_retrieval(8, seq_len=192, n_pairs=16, n_swaps=24, seed=4)
    assert int(b2["pos_joint"][0].item()) - (16 * 3) > 32
    assert ((b2["y_joint"] >= 64) & (b2["y_joint"] < 128)).all()

    # P3 Alias-Swap: forced counter-conventional never equals conventional attr for canonical aliases!
    b3 = wp.sample_p3_alias_swap(24, seq_len=256, force_counter_conv=True, force_mid_pos=True, seed=5)
    m_can = b3["is_canon"]
    assert m_can.any() and (~m_can).any()
    assert (b3["y_use"][m_can] != b3["conv_attr"][m_can]).all()

    # P4 No-Hash Rule-2Hop: View A (0..63) and View B (64..79) have ZERO overlapping tokens!
    w4 = wp.P4RuleWorld(seed=6)
    va_all = {t for e in range(w4.n_ent) for t in w4.view_a(e)}
    vb_all = {t for e in range(w4.n_ent) for t in w4.view_b(e)}
    assert va_all.isdisjoint(vb_all)
    eps = w4.build_test_episodes(seed=7)
    assert len(eps) == 16

    # P5a Star-Graph: greedy prefix_len aligns with path target
    b5a = wp.sample_p5a_star_graph(8, degree=5, path_len=4, seed=8)
    assert b5a["targets"][:, b5a["prefix_len"] - 1 :].shape == (8, 4)

    # P5b Lano VarDensity: high-entropy mask has both True and False turns
    b5b = wp.sample_p5b_lano_vardensity(8, seq_len=192, seed=9)
    hm = b5b["hi_entropy_mask"].numpy()
    assert hm.any() and (~hm).any()


def test_automated_auditors_hop_ladder_sign_flip_and_boundary_hist() -> None:
    # 1. Hop ladder auditor detects monotonic progress when layer l encodes hop l+1
    rng = torch.Generator().manual_seed(11)
    n_tr, n_te, d, n_cls = 160, 64, 24, 8
    y_tr = torch.randint(0, n_cls, (n_tr, 4), generator=rng)
    y_te = torch.randint(0, n_cls, (n_te, 4), generator=rng)
    proto = torch.randn(n_cls, d, generator=rng)
    tr_s, te_s = [], []
    for l_idx in range(4):
        # Layer l cleanly exposes hops 0..l
        xtr = sum(proto[y_tr[:, h]] for h in range(l_idx + 1)) + 0.05 * torch.randn(n_tr, d, generator=rng)
        xte = sum(proto[y_te[:, h]] for h in range(l_idx + 1)) + 0.05 * torch.randn(n_te, d, generator=rng)
        tr_s.append(xtr)
        te_s.append(xte)
    lad = audit_linear_probe_hop_ladder(tr_s, y_tr, te_s, y_te, n_classes=n_cls)
    assert lad["solved_hop_frontier_by_layer"][-1] == 4
    assert lad["monotonic_ladder_detected"] is True

    # 2. Gradient sign-flip check passes on >=50pp collapse, fails when flip barely changes accuracy
    assert audit_gradient_sign_flip(92.0, 10.0)["pass_sign_flip_check"] is True
    assert audit_gradient_sign_flip(85.0, 82.0)["pass_sign_flip_check"] is False

    # 3. Boundary histogram check distinguishes high-entropy turns from uniform stride-4
    hi = np.zeros((16, 120), dtype=np.bool_)
    hi[:, ::4] = False
    hi[:, 1::5] = True  # 20% high-entropy turns -> compression ratio 5.0
    hi[:, 0] = True     # pushes rate to ~20.8% -> ratio ~4.8
    unif = np.zeros_like(hi)
    unif[:, ::4] = True
    au_unif = audit_boundary_histogram(unif, hi)
    au_oracle = audit_boundary_histogram(hi, hi, target_ratio_bounds=(3.2, 5.2))
    assert au_unif["pass_boundary_audit"] is False
    assert au_oracle["pass_boundary_audit"] is True
