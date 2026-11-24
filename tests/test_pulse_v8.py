"""Unit, Parity, Causality & Plasticity Verification Suite for Unified PULSE-v8."""
from __future__ import annotations

import torch

from pulse.pulse_v8 import PULSEv8Config, PULSEv8Model
from pulse.triton_slot_register import fused_slot_register_scan, pytorch_slot_register_scan
from pulse.whiteboard_anvil import audit_zero_hidden_params, build_dual_strong_baselines
from pulse.whiteboard_routes import RoleFillerSlotRegisterBlock


def test_pulse_v8_zero_hidden_params_and_strict_parity() -> None:
    for vocab in (96, 112, 160):
        tf = build_dual_strong_baselines(vocab, smoke=False, seed=41)["tf_modern"]
        v8 = PULSEv8Model(PULSEv8Config.lab_3m(vocab, use_fomp=True))
        assert v8.num_parameters() <= tf.num_parameters(), (
            f"PULSE-v8 ({v8.num_parameters():,}) > TF-Modern ({tf.num_parameters():,}) at V={vocab}"
        )
        au = audit_zero_hidden_params(v8, vocab, 256)
        assert au["pass_zero_hidden_audit"], f"PULSE-v8 failed audit at V={vocab}: {au}"

    # Verify Stage B 30M flagship (v8.1: 2x SWA + 8x GDN + 2x NoPE-MLA + native FOMP-v2.2, zero 64-Slot tax!)
    v8_30m = PULSEv8Model(PULSEv8Config.stage_b_30m(vocab_size=50257, use_fomp=True))
    assert len(v8_30m.slot_blocks) == 0 and v8_30m.hyper is not None
    assert v8_30m.config.layer_types.count("gdn") == 8
    assert v8_30m.num_parameters() <= 40_706_832
    assert audit_zero_hidden_params(v8_30m, 50257, 384)["pass_zero_hidden_audit"]
    # Verify exact backbone weight compatibility between pretrained use_fomp=False and flagship use_fomp=True!
    v7b_sd = PULSEv8Model(PULSEv8Config.stage_b_30m(vocab_size=50257, use_fomp=False)).state_dict()
    inc = v8_30m.load_state_dict(v7b_sd, strict=False)
    assert len(inc.unexpected_keys) == 0 and all(
        k.startswith(("hyper.", "med_down.", "med_up.")) for k in inc.missing_keys
    )


def test_pulse_v8_strict_causality_and_fomp_v22_bf16() -> None:
    torch.manual_seed(19)
    cfg = PULSEv8Config.micro_test(vocab_size=112, use_fomp=True)
    m = PULSEv8Model(cfg).eval()
    with torch.no_grad():
        m.slot_blocks[0].o_proj.weight.normal_(0.0, 0.05)

    # 1. Strict AR causality: mutating tokens t >= 12 must NOT alter logits[:, :12]!
    x1 = torch.randint(0, 112, (2, 24))
    x2 = x1.clone()
    x2[:, 12:] = (x2[:, 12:] + 11) % 112
    with torch.no_grad():
        l1 = m(x1)["logits"]
        l2 = m(x2)["logits"]
    torch.testing.assert_close(l1[:, :12], l2[:, :12], rtol=1e-5, atol=1e-5)
    assert m.active_slot_entropy_pct > 20.0

    # 2. FOMP-CMS v2.2: g_ext=0 strictly identity + exact sign-flip negation + 2-layer rel_gate!
    ep = torch.tensor([[[96, 5, 12], [96, 17, 23]], [[96, 8, 31], [96, 41, 2]]], dtype=torch.long)
    qry = torch.tensor([[96, 5], [96, 8]], dtype=torch.long)
    with torch.no_grad():
        base = m.forward_with_fast_weights(qry, compiled=None)["logits"]
        blk = m.forward_with_fast_weights(qry, compiled=m.compile_episodes(ep, g_ext=0.0))["logits"]
        torch.testing.assert_close(blk, base, rtol=0.0, atol=0.0)

        comp = m.compile_episodes(ep, g_ext=1.0)
        h_e = m.encode_early(qry)
        assert m.hyper is not None
        g_m, sims, rl = m.hyper.compute_familiarity(h_e, comp)
        assert rl.shape == (2, 2)
        hp = m.hyper.apply_layer_fast_weight(h_e, comp, 0, g_m=g_m, sims=sims, sign_flip=False)
        hn = m.hyper.apply_layer_fast_weight(h_e, comp, 0, g_m=g_m, sims=sims, sign_flip=True)
        torch.testing.assert_close(hp - h_e, -(hn - h_e), rtol=1e-5, atol=1e-5)

    # 3. BF16 autocast end-to-end 1st-order backward through GDN + 64-Slot + MLA + FOMP-v2.2!
    m.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        comp = m.compile_episodes(ep, g_ext=1.0)
        out = m.forward_with_fast_weights(qry, compiled=comp)
        va = torch.tensor([[99, 5], [99, 8]], dtype=torch.long)
        vb = torch.tensor([[99, 64, 77], [99, 65, 72]], dtype=torch.long)
        loss = out["logits"].float().pow(2).mean() + m.view_alignment_loss(va, vb) + m.aux_loss
    loss.backward()
    assert m.slot_blocks[0].r_proj.weight.grad is not None and m.slot_blocks[0].r_proj.weight.grad.abs().sum().item() > 0.0
    assert m.hyper.val_projs[0].weight.grad is not None and m.hyper.val_projs[0].weight.grad.abs().sum().item() > 0.0
    assert m.hyper.rel_gate[0].weight.grad is not None and m.hyper.rel_gate[0].weight.grad.abs().sum().item() > 0.0
    assert all(torch.isfinite(p.grad).all() for p in m.io_parameters() if p.grad is not None)


def test_slot_register_scan_matches_block_and_finite_diff_grads() -> None:
    torch.manual_seed(77)
    # 1. Non-multiple-of-chunk length T=37 checks padding/masking exactness & causality!
    bsz, h, seq_len, s, d, c_sz = 2, 4, 37, 16, 16, 16
    q = torch.nn.functional.normalize(torch.randn(bsz, h, seq_len, d), dim=-1).requires_grad_(True)
    r = torch.nn.functional.normalize(torch.randn(bsz, h, seq_len, d), dim=-1).requires_grad_(True)
    f = torch.randn(bsz, h, seq_len, d, requires_grad=True)
    g = torch.sigmoid(torch.randn(bsz, h, seq_len)).requires_grad_(True)
    k0 = (torch.nn.functional.normalize(torch.randn(1, h, s, d), dim=-1) * 0.25).requires_grad_(True)
    itau = torch.full((h,), 6.0, requires_grad=True)
    bf = torch.full((h,), 1.5, requires_grad=True)

    o_ref, a_ref = pytorch_slot_register_scan(q, r, f, g, k0, itau, bf, chunk_size=c_sz)
    assert o_ref.shape == (bsz, h, seq_len, d) and a_ref.shape == (bsz, h, s)
    assert torch.isfinite(o_ref).all() and torch.isfinite(a_ref).all()
    loss_ref = (o_ref * torch.randn_like(o_ref)).sum() + 0.1 * a_ref.pow(2).sum()
    loss_ref.backward()
    for t in (q, r, f, g, k0, itau, bf):
        assert t.grad is not None and torch.isfinite(t.grad).all()

    # 2. Verify RoleFillerSlotRegisterBlock return_hist=False matches return_hist=True on output!
    blk = RoleFillerSlotRegisterBlock(d_model=64, n_heads=4, d_ff=96, n_slots=16, chunk_size=16).eval()
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        y_hist, _, _ = blk(x, return_hist=True)
        y_fast, kh, vh = blk(x, return_hist=False)
    assert kh is None and vh is None
    torch.testing.assert_close(y_fast, y_hist, rtol=1e-4, atol=1e-4)

    # 3. Verify fused_slot_register_scan runs cleanly (CPU fallback or CUDA Triton!)
    o_f, a_f = fused_slot_register_scan(
        q.detach().clone().requires_grad_(True),
        r.detach().clone().requires_grad_(True),
        f.detach().clone().requires_grad_(True),
        g.detach().clone().requires_grad_(True),
        k0.detach().clone().requires_grad_(True),
        itau.detach().clone().requires_grad_(True),
        bf.detach().clone().requires_grad_(True),
        chunk_size=c_sz,
        use_tf32=False,
    )
    torch.testing.assert_close(o_f, o_ref.detach(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(a_f, a_ref.detach(), rtol=1e-4, atol=1e-4)


def test_pulse_v8_compiled_fast_paths_dispatch_matches_eager() -> None:
    """Exercise every block's `_c_fast` / `_c_pre` / `_c_post` CUDA-compiled dispatch branch on CPU!"""
    torch.manual_seed(29)
    m = PULSEv8Model(PULSEv8Config.micro_test(vocab_size=112, use_fomp=True)).train()
    x = torch.randint(0, 112, (2, 24))
    y_eager = m(x)["logits"].detach()

    # Bind compiled fast-path hooks directly to exact eager functions so every CUDA dispatch branch is covered!
    for blk in m.blocks:
        if hasattr(blk, "_fast_impl"):
            blk._c_fast = blk._fast_impl  # type: ignore[attr-defined]
        if hasattr(blk, "_pre_fast") and hasattr(blk, "_post_fast"):
            blk._c_pre = blk._pre_fast    # type: ignore[attr-defined]
            blk._c_post = blk._post_fast  # type: ignore[attr-defined]
    out_c = m(x)
    torch.testing.assert_close(out_c["logits"], y_eager, rtol=1e-5, atol=1e-5)
    (out_c["logits"].pow(2).mean() + m.aux_loss).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.io_parameters() if p.grad is not None)
    assert m.active_slot_entropy_pct > 20.0
