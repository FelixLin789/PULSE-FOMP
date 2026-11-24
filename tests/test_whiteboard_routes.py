"""Unit & Causality Verification Suite for Week-2 Whiteboard Routes (Route 3 FOMP-CMS & Route 1 APRM)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from pulse.whiteboard_anvil import audit_zero_hidden_params, build_dual_strong_baselines
from pulse.whiteboard_routes import APRMConfig, APRMModel, DCERConfig, DCERModel, FOMPCMSConfig, FOMPCMSModel

_BENCH = str(Path(__file__).resolve().parents[1] / "benchmarks")
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)
import whiteboard_week2_screen as w2  # noqa: E402


def test_routes_zero_hidden_params_and_strict_parity() -> None:
    for vocab in (16, 48, 160):
        tf = build_dual_strong_baselines(vocab, smoke=False, seed=41)["tf_modern"]
        dc = DCERModel(DCERConfig.lab_3m(vocab))
        assert dc.num_parameters() <= tf.num_parameters(), f"DC-ER ({dc.num_parameters()}) > TF ({tf.num_parameters()}) at V={vocab}"
        au_dc = audit_zero_hidden_params(dc, vocab, 256)
        assert au_dc["pass_zero_hidden_audit"], f"dc_er failed audit at V={vocab}: {au_dc}"
    tf = build_dual_strong_baselines(160, smoke=False, seed=41)["tf_modern"]
    fm = FOMPCMSModel(FOMPCMSConfig.lab_3m(160))
    ap = APRMModel(APRMConfig.lab_3m(160))
    assert fm.num_parameters() <= tf.num_parameters()
    assert ap.num_parameters() <= tf.num_parameters()
    for name, m in (("fomp_cms", fm), ("aprm", ap)):
        au = audit_zero_hidden_params(m, 160, 256)
        assert au["pass_zero_hidden_audit"], f"{name} failed audit: {au}"


def test_route3_fomp_cms_1st_order_compilation_g_ext_and_sign_flip() -> None:
    torch.manual_seed(17)
    cfg = FOMPCMSConfig.micro_test(112)
    m = FOMPCMSModel(cfg)
    ep = torch.tensor([[[96, 5, 12], [96, 17, 23]], [[96, 8, 31], [96, 41, 2]]], dtype=torch.long)
    qry = torch.tensor([[96, 5], [96, 8]], dtype=torch.long)  # Exact early-manifold key match on pos -1!

    # 1. Hard external gate g_ext=0 -> strictly identical to compiled=None!
    m.eval()
    with torch.no_grad():
        base = m.forward_with_fast_weights(qry, compiled=None)["logits"]
        comp_zero = m.compile_episodes(ep, g_ext=0.0)
        blocked = m.forward_with_fast_weights(qry, compiled=comp_zero)["logits"]
        torch.testing.assert_close(blocked, base, rtol=0.0, atol=0.0)

        # 2. Exact sign-flip negation at hyper layer output + sharp ReLU zero-leakage shield below tau!
        comp_one = m.compile_episodes(ep, g_ext=1.0)
        h_early = m.encode_early(qry)
        g_m, sims, _ = m.hyper.compute_familiarity(h_early, comp_one)
        assert g_m[:, -1, 0].min().item() > 0.70  # Exact match cos=1.0 * sigmoid(1.5) -> strong gate activation!
        hp = m.hyper.apply_layer_fast_weight(h_early, comp_one, 0, g_m=g_m, sims=sims, sign_flip=False)
        hn = m.hyper.apply_layer_fast_weight(h_early, comp_one, 0, g_m=g_m, sims=sims, sign_flip=True)
        torch.testing.assert_close(hp - h_early, -(hn - h_early), rtol=1e-5, atol=1e-5)

        # Strict zero-leakage check when all similarities <= tau:
        g_zero = torch.zeros_like(g_m)
        h_silent = m.hyper.apply_layer_fast_weight(h_early, comp_one, 0, g_m=g_zero, sims=sims, sign_flip=False)
        torch.testing.assert_close(h_silent, h_early, rtol=0.0, atol=0.0)

    # 3. 1st-order episodic backward + InfoNCE view alignment under BF16 autocast!
    m.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        comp = m.compile_episodes(ep, g_ext=1.0)
        out = m.forward_with_fast_weights(qry, compiled=comp)
        va = torch.tensor([[99, 5], [99, 8]], dtype=torch.long)
        vb = torch.tensor([[99, 64, 77], [99, 65, 72]], dtype=torch.long)
        loss = out["logits"].float().pow(2).mean() + m.view_alignment_loss(va, vb)
    loss.backward()
    assert m.hyper.key_proj.weight.grad is not None and m.hyper.key_proj.weight.grad.abs().sum().item() > 0.0
    assert m.hyper.val_projs[0].weight.grad is not None and m.hyper.val_projs[0].weight.grad.abs().sum().item() > 0.0
    assert all(torch.isfinite(p.grad).all() for p in m.io_parameters() if p.grad is not None)


def test_route1_aprm_strict_causality_and_grad_horizons_bf16() -> None:
    torch.manual_seed(23)
    cfg = APRMConfig.micro_test(96, loop_steps=4, grad_horizon=-1)
    m = APRMModel(cfg).eval()
    # Non-zero init on output projections so any future leakage would show up immediately!
    with torch.no_grad():
        m.pre_slot.o_proj.weight.normal_(std=0.05)
        m.core_lookup.o_proj.weight.normal_(std=0.05)
        m.core_cont.o_proj.weight.normal_(std=0.05)

    # 1. Strict AR causality check: mutating tokens t > 11 must NOT alter logits at t <= 11!
    x1 = torch.randint(0, 96, (2, 24))
    x2 = x1.clone()
    x2[:, 12:] = (x2[:, 12:] + 7) % 96
    with torch.no_grad():
        l1 = m.forward_anvil(x1, use_final_slot_deref=False)["logits"]
        l2 = m.forward_anvil(x2, use_final_slot_deref=False)["logits"]
    torch.testing.assert_close(l1[:, :12], l2[:, :12], rtol=1e-5, atol=1e-5)
    assert m.active_slot_entropy_pct > 20.0

    # 2. All 3 gradient horizons (-1 Full, 4 Truncated, 1 Implicit) produce finite non-zero Prelude+Core grads in BF16!
    for gh in (-1, 2, 1):
        m_g = APRMModel(APRMConfig.micro_test(96, loop_steps=4, grad_horizon=gh))
        m_g.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m_g.forward_anvil(x1, return_layer_states=True, use_final_slot_deref=True)
            loss = out["logits"][:, -1].float().pow(2).mean() + m_g.aux_loss
        assert len(out["layer_states"]) == 4 + 2  # [x_pre, loop0..3, h_fin]
        loss.backward()
        assert m_g.pre_slot.r_proj.weight.grad is not None and m_g.pre_slot.r_proj.weight.grad.abs().sum().item() > 0.0
        assert m_g.core_lookup.q_proj.weight.grad is not None and m_g.core_lookup.q_proj.weight.grad.abs().sum().item() > 0.0
        assert all(torch.isfinite(p.grad).all() for p in m_g.io_parameters() if p.grad is not None)


def test_single_cycle_depo1_has_zero_short_cycles() -> None:
    b = w2.sample_p1a_single_cycle_depo1(64, n_nodes=16, hops=(1, 2, 4, 8, 12), seed=77)
    starts = b["input_ids"][:, -2]  # [QRY, start, HOP_k]
    for h_idx in range(12):
        # In a 16-node single cycle, pi^k(s) can NEVER return to s for any k in 1..15!
        assert (b["inter_targets"][:, h_idx] != starts).all()


def test_multiquery_af_depo1_zero_answer_leak() -> None:
    import whiteboard_week2_v2 as w2v2
    b = w2v2.sample_p1a_sc_multiquery_af(32, n_nodes=16, hops=(1, 2, 3, 4), frontier_hops=(3, 4), n_queries=8, seed=88)
    x, y, qp = b["input_ids"], b["targets"], b["q_pos"]
    assert x.shape == (32, 16 * 3 + 8 * 3)
    assert qp.tolist() == [48 + qi * 3 + 2 for qi in range(8)]
    # Every query slot is strictly [QRY=17, start_node, HOP_k >= 18]: ZERO answer tokens inserted!
    for qi in range(8):
        assert (x[:, 48 + qi * 3] == 17).all()
        assert (x[:, 48 + qi * 3 + 2] >= 18).all()
        assert (y[:, qi] != x[:, 48 + qi * 3 + 1]).all()  # Single cycle -> target != start!


def test_route2_dcer_strict_causality_and_bf16_grads() -> None:
    torch.manual_seed(31)
    cfg = DCERConfig.micro_test(vocab_size=48, loop_steps=3, target_ratio=4.0)
    m = DCERModel(cfg).eval()
    with torch.no_grad():
        m.con_to_tok.weight.normal_(0.0, 0.05)
        for blk in m.concept_core:
            blk.o_proj.weight.normal_(0.0, 0.05)

    # 1. Strict AR causality: mutating tokens t >= 16 must NOT alter logits[:, :16] OR boundary_mask[:, :16]!
    x1 = torch.randint(0, 48, (2, 32))
    x2 = x1.clone()
    x2[:, 16:] = (x2[:, 16:] + 13) % 48
    with torch.no_grad():
        o1 = m.forward_anvil(x1)
        o2 = m.forward_anvil(x2)
    assert torch.equal(o1["boundary_mask"][:, :16], o2["boundary_mask"][:, :16])
    torch.testing.assert_close(o1["logits"][:, :16], o2["logits"][:, :16], rtol=1e-5, atol=1e-5)

    # 2. BF16 autocast end-to-end backward flows cleanly through Encoder + Router + Concept Core + Decoder!
    m.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m.forward_anvil(x1)
        loss = out["logits"].float().pow(2).mean() + m.aux_loss
    loss.backward()
    assert m.router.q_bnd.weight.grad is not None and m.router.q_bnd.weight.grad.abs().sum().item() > 0.0
    assert m.concept_core[0].q_proj.weight.grad is not None and m.concept_core[0].q_proj.weight.grad.abs().sum().item() > 0.0
    assert m.con_to_tok.weight.grad is not None and m.con_to_tok.weight.grad.abs().sum().item() > 0.0
    assert all(torch.isfinite(p.grad).all() for p in m.io_parameters() if p.grad is not None)
    assert abs(m.concept_lr_scale() - 0.80) < 1e-6


def test_dcer_probes_deep_stack_and_single_cycle_depo2() -> None:
    import whiteboard_week2_dcer as wd
    # 1. DeepStack: verify exact closing-pop targets exist (8 per item) & hi_in_mask aligns with input_ids!
    ds = wd.sample_p5b_lano_deep_stack(16, stack_depth=8, seq_len=160, seed=42)
    assert ds["close_mask"].sum(dim=1).eq(8).all()
    assert ds["targets"][ds["close_mask"]].min().item() >= 4 and ds["targets"][ds["close_mask"]].max().item() <= 7
    assert (ds["hi_in_mask"].sum(dim=1) == 17).all()  # 8 open + 1 PUSH_END + 8 close = 17 concept turns!

    # 2. Depo2-SC: verify single Hamiltonian cycle (for every hop 1..8, answer code != start code!)
    d2 = wd.sample_p1b_depo2_single_cycle(32, n_nodes=10, hops=(1, 2, 4, 8), seq_len=224, seed=43)
    for b in range(32):
        seq = d2["input_ids"][b].tolist()
        qry_i = seq.index(6)
        hop_i = next(i for i in range(qry_i + 1, len(seq)) if seq[i] >= 7)
        start_code = seq[qry_i + 1 : hop_i]
        ans_code = d2["targets"][b][d2["ans_mask"][b]].tolist()[:-1]  # strip trailing EDGE_SEP
        assert ans_code != start_code, f"Short-cycle cheat found in Depo2-SC: {ans_code} == {start_code}"
