"""Unit & Compiler Verification Suite for PULSE-v7 (`src/pulse/pulse_v7.py`)."""
from __future__ import annotations

from dataclasses import replace

import torch

from pulse.experiment import FairMuonAdamW
from pulse.pulse_v7 import (
    PULSEv7Config,
    PULSEv7Model,
    RAMNetMemoryBlock,
    ToasterMemoryBlock,
    patch_fla_sm89_shared_memory,
)


def test_v7_all_recurrence_modes_streaming_parity_and_zero_leak_ci() -> None:
    for mode in ("gdn2_dplr", "gdn2_kda", "gdn_scalar"):
        torch.manual_seed(11)
        cfg = replace(PULSEv7Config.micro_test(), recurrence_mode=mode)
        model = PULSEv7Model(cfg).eval()
        with torch.no_grad():
            model.emb_out.weight.normal_(std=0.06)
            for blk in model.blocks:
                blk.o_proj.weight.normal_(std=0.03)
                blk.mlp.w_down.weight.normal_(std=0.03)
            tokens = torch.randint(4, cfg.vocab_size, (2, 35))
            whole = model(tokens, return_state=False)["logits"]
            out_full = model(tokens, return_state=True)
            torch.testing.assert_close(out_full["logits"], whole, rtol=1e-5, atol=1e-5)
            log_b, back_b = model.storage_audit_bytes(out_full["working_state"])
            assert log_b == back_b, f"View storage leak in {mode}: {log_b} != {back_b}"

            # Multi-chunk + single-token streaming equivalence
            state = None
            parts = []
            for chunk in torch.split(tokens, (13, 11, 10, 1), dim=1):
                out = model(chunk, state, return_state=True)
                parts.append(out["logits"])
                state = out["working_state"]
                lb, bb = model.storage_audit_bytes(state)
                assert lb == bb
            torch.testing.assert_close(torch.cat(parts, dim=1), whole, rtol=1e-4, atol=1e-4)


def test_v7a_strict_o1_backing_bytes_and_v7b_mla_compression() -> None:
    torch.manual_seed(19)
    cfg_a = replace(
        PULSEv7Config.micro_test(),
        layer_types=("swa", "gdn2", "gdn2", "swa", "gdn2", "gdn2"),
        recurrence_mode="gdn2_dplr",
    )
    m_a = PULSEv7Model(cfg_a).eval()
    bytes_by_len = {}
    with torch.no_grad():
        for slen in (32, 128, 256):
            st = m_a(torch.randint(4, 260, (1, slen)), return_state=True)["working_state"]
            lb, bb = m_a.storage_audit_bytes(st)
            assert lb == bb
            bytes_by_len[slen] = bb
    assert bytes_by_len[32] == bytes_by_len[128] == bytes_by_len[256], f"Non-O(1) v7A state: {bytes_by_len}"

    # Triadic E=4 also strictly O(1)
    cfg_tri = replace(cfg_a, triadic_e=4, recurrence_mode="gdn_scalar")
    m_tri = PULSEv7Model(cfg_tri).eval()
    with torch.no_grad():
        _, b1 = m_tri.storage_audit_bytes(m_tri(torch.randint(4, 260, (1, 32)), return_state=True)["working_state"])
        _, b2 = m_tri.storage_audit_bytes(m_tri(torch.randint(4, 260, (1, 128)), return_state=True)["working_state"])
        assert b1 == b2


def test_v7_qwen38_muon_gate_exclusion_and_fused_region_fullgraph() -> None:
    torch.manual_seed(29)
    cfg = PULSEv7Config(
        vocab_size=128, embed_dim=24, d_model=48, n_heads=4, n_kv_heads=2, n_layers=12,
        layer_types=("swa", "gdn2", "gdn2", "gdn2", "gdn2", "mla") * 2,
        recurrence_mode="gdn2_dplr", gate_rank=16, d_ff=96, delta_chunk_size=8,
        swa_window_size=8, mla_d_latent=24,
    )
    model = PULSEv7Model(cfg)
    with torch.no_grad():
        model.emb_out.weight.normal_(std=0.05)
        for blk in model.blocks:
            blk.o_proj.weight.normal_(std=0.03)
            blk.mlp.w_down.weight.normal_(std=0.03)

    # 1. Verify Qwen3.8 FairMuonAdamW recipe: ZERO gate or conv weights enter Muon!
    opt = FairMuonAdamW(model.io_parameters())
    muon_ids = {id(p) for p in opt.param_groups[0]["params"]}
    for name, p in model.named_parameters():
        if any(tag in name for tag in ("canon_", "qkv_conv", "out_gate", "gates_proj", "_down", "_up", "g_proj", "a_down", "a_up", "b_down", "b_up", "w_down_g", "w_up_g")):
            if "mlp.w_down" in name or "mlp.w_up" in name or "kv_down" in name or "k_up" in name or "v_up" in name:
                continue
            assert id(p) not in muon_ids, f"Gate/conv parameter leaked into Muon: {name}"

    # 2. Verify fused regions match reference per-block path + pass fullgraph aot_eager under recompile_limit=2!
    prev_lim = torch._dynamo.config.recompile_limit
    torch._dynamo.config.recompile_limit = 2
    try:
        gdn_idx, fns = model._build_fused_region_fns()
        assert len({fn.__code__ for fn in fns}) == len(fns)
        c_fns = [torch.compile(fn, backend="aot_eager", dynamic=False, fullgraph=True) for fn in fns]
        tok = torch.randint(4, 128, (2, 16))
        ref_h, _, _ = model._run_backbone(tok, return_state=False)
        reg_h = model._run_fused_regions(tok, gdn_idx, c_fns)
        torch.testing.assert_close(reg_h, ref_h, rtol=1e-5, atol=1e-5)
        reg_h.pow(2).mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    finally:
        torch._dynamo.config.recompile_limit = prev_lim


def test_stage1_toaster_and_ramnet_forward_backward_o1_state() -> None:
    torch.manual_seed(37)
    x = torch.randn(2, 24, 64, requires_grad=True)
    e = torch.randn(2, 24, 32)
    for blk in (
        ToasterMemoryBlock(d_model=64, embed_dim=32, n_heads=4, d_ff=128, d_hidden=32, chunk_size=8),
        RAMNetMemoryBlock(d_model=64, embed_dim=32, n_heads=4, d_ff=128, num_slots=32, top_k=4, chunk_size=8),
    ):
        y1, st1 = blk(x, e, return_state=True)
        assert y1.shape == x.shape and st1 is not None
        y1.pow(2).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        x.grad.zero_()
        with torch.no_grad():
            _, st2 = blk(torch.randn(2, 48, 64), torch.randn(2, 48, 32), return_state=True)
            b1 = sum(t.untyped_storage().nbytes() for t in st1.values())
            b2 = sum(t.untyped_storage().nbytes() for t in st2.values())
            assert b1 == b2
    assert isinstance(patch_fla_sm89_shared_memory(), dict)


def test_v7_stage2_lt2_full_bptt_and_stars_scaling() -> None:
    from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM

    # 1. Verify 30M LT2 param parity strictly <= matched 6L Transformer
    tf30_p = ScalableTransformerLM(ScalableTransformerConfig.colab_30m()).num_parameters()
    lt2_30_p = PULSEv7Model(PULSEv7Config.colab_30m_v7b_lt2(loop_steps=2, looped=True)).num_parameters()
    assert lt2_30_p <= tf30_p, f"LT2 30M {lt2_30_p} > TF {tf30_p}"

    # 2. Micro LT2: Full BPTT through T=2 loops + exit loss + STARS r=1..6 stability + compiled stages!
    torch.manual_seed(53)
    cfg = replace(
        PULSEv7Config.micro_test(),
        layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"),
        recurrence_mode="gdn_scalar", use_canon_cd=False, logit_softcap=0.0, use_ssmax=True,
        loop_core_range=(2, 4), loop_steps=2, loop_exit_eps=0.05,
    )
    model = PULSEv7Model(cfg)
    with torch.no_grad():
        model.emb_out.weight.normal_(std=0.05)
        for blk in model.blocks:
            blk.o_proj.weight.normal_(std=0.03)
            blk.mlp.w_down.weight.normal_(std=0.03)

    toks = torch.randint(4, cfg.vocab_size, (2, 17))
    loss = model.compute_loss(toks)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.loop_reinj_logit is not None and model.loop_reinj_logit.grad is not None and model.loop_reinj_logit.grad.abs().item() > 0.0
    assert model.loop_damp_logit is not None and model.loop_damp_logit.grad is not None and model.loop_damp_logit.grad.abs().item() > 0.0
    assert model.blocks[2].q_proj.weight.grad is not None and model.blocks[2].q_proj.weight.grad.abs().sum().item() > 0.0

    # STARS test-time scaling r=1, 2, 4, 6: finite & bounded
    model.eval()
    with torch.no_grad():
        norms = []
        for r in (1, 2, 4, 6):
            lg = model(toks[:, :-1], return_state=False, n_loops=r)["logits"]
            assert torch.isfinite(lg).all()
            norms.append(float(lg.float().norm().item()))
        assert max(norms) < 10.0 * min(norms) + 1e-3, f"STARS instability across r=1..6: {norms}"

    # Compiled LT2 stages match eager path identically under fullgraph aot_eager
    s, e = cfg.loop_core_range  # type: ignore[misc]
    stages = {}
    for pname, a, b in (("pre", 0, s), ("core", s, e), ("coda", e, cfg.n_layers)):
        gi, fns = model._build_slice_region_fns(a, b, pname)
        stages[pname] = (gi, [torch.compile(fn, backend="aot_eager", dynamic=False, fullgraph=True) for fn in fns])
    with torch.no_grad():
        ref = model(toks[:, :-1], return_state=False, n_loops=2)["logits"]
        model._c_loop_stages = stages
        comp = model(toks[:, :-1], return_state=False, n_loops=2)["logits"]
        torch.testing.assert_close(comp, ref, rtol=1e-4, atol=1e-4)


def test_v7_stage3_dual_track_plasticity_and_verifier() -> None:
    from pulse.pulse_v7_plasticity import (
        DualTrackSemanticMemory,
        SemanticKeyProjector,
        TransitionConditionedVerifier,
        decore_contrastive_logits,
    )

    torch.manual_seed(71)
    cfg = replace(
        PULSEv7Config.micro_test(),
        layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"),
        recurrence_mode="gdn_scalar", use_canon_cd=False, logit_softcap=0.0,
        loop_core_range=(2, 4), loop_steps=2,
    )
    model = PULSEv7Model(cfg).eval()
    toks = torch.randint(4, cfg.vocab_size, (2, 20))
    out = model.forward_with_hooks(toks, collect_gdn_signals=True)
    assert out["h_mid"] is not None and out["gdn_signals"] is not None
    # 3 GDN layers x 4 heads x 5 signals = 60 dims
    assert out["gdn_signals"].shape == (2, 20, 60)
    assert torch.isfinite(out["gdn_signals"]).all()

    # Joint differentiable backward through gdn_signals + h_fin + logits under BF16 autocast (Stage 3b-v2 guard)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_tr = model.forward_with_hooks(toks, collect_gdn_signals=True)
        loss_tr = out_tr["gdn_signals"].pow(2).mean() + out_tr["h_fin"].float().pow(2).mean() + out_tr["logits"].float().pow(2).mean()
    loss_tr.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    model.zero_grad(set_to_none=True)
    model.eval()

    # DualTrackSemanticMemory: exact recall + paraphrase recall + strictly 0.000 unhit drift + EngramEdit sleep
    proj = SemanticKeyProjector(cfg.d_model, d_key=32).eval()
    mem = DualTrackSemanticMemory(cfg.d_model, d_key=32, suffix_len=4, tau_sem=0.85, key_projector=proj)
    h_fact = torch.randn(1, 6, cfg.d_model)
    dh_tgt = torch.randn(cfg.d_model)
    p_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    mem.write_fact(p_ids, h_fact, dh_tgt)

    # Unrelated base sequence: familiarity <= tau_sem -> STRICTLY 0.000 delta!
    h_base = -h_fact.clone()
    out_base = mem.apply_mid_hook(h_base, torch.tensor([[20, 21, 22, 23, 24, 25]]))
    torch.testing.assert_close(out_base, h_base, rtol=0.0, atol=0.0)

    # Paraphrase query (different tokens, close semantic key -> semantic track hits!)
    h_para = h_fact + 0.02 * torch.randn_like(h_fact)
    out_para = mem.apply_mid_hook(h_para, torch.tensor([[30, 31, 32, 33, 34, 35]]), last_pos_only=True)
    rel_err = (out_para[0, -1] - h_para[0, -1] - dh_tgt).norm().item() / dh_tgt.norm().item()
    assert rel_err < 0.1, f"Semantic RLS relative error too high: {rel_err}"

    # Sleep consolidation to EngramEdit: W_fast cleared, engram bank recalls accurately!
    moved = mem.consolidate_to_engram()
    assert moved == 1 and mem.W_fast.abs().max().item() == 0.0
    out_sleep = mem.apply_mid_hook(h_para, torch.tensor([[30, 31, 32, 33, 34, 35]]), last_pos_only=True)
    torch.testing.assert_close(out_sleep[0, -1] - h_para[0, -1], dh_tgt, rtol=1e-4, atol=1e-4)

    # TransitionConditionedVerifier strictly < 50K params (even at d_model=512 + 240 signals = 752!)
    ver = TransitionConditionedVerifier(d_in=752, d_hidden=48, n_classes=3)
    assert ver.num_parameters() < 50000
    v_logits = ver(torch.randn(4, 752))
    assert v_logits.shape == (4, 3)

    # DeCoRe contrastive decoding untouched when p_illegal <= threshold, active when > threshold
    lf, lp = torch.randn(2, 10), torch.randn(2, 10)
    adj = decore_contrastive_logits(lf, lp, torch.tensor([0.1, 0.9]), lam=1.5, threshold=0.5)
    torch.testing.assert_close(adj[0], lf[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(adj[1], lf[1] + 1.5 * (lf[1] - lp[1]), rtol=1e-5, atol=1e-5)
