"""TDD Unit & Integration Tests for PULSE-v4 (`src/pulse/pulse_v4.py`)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW
from pulse.pulse_v4 import (
    PULSEv4Config,
    PULSEv4Model,
    _V5EchoicSWABlock,
    _V5FusedDeltaBlock,
    _V5MultiScaleCanon,
)
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM
from pulse.v3_blocks import ContextGatedEngram, FastGatedDeltaNet


def test_v5_multiscale_canon_echoic_swa_and_delta_exact_streaming_equivalence() -> None:
    torch.manual_seed(7)
    # 1. Exact Multi-Scale Canon (k=4 + dilated d=4, 12-token ring) batch vs streaming equivalence!
    mc = _V5MultiScaleCanon(dim=32).eval()
    x = torch.randn(2, 19, 32)
    with torch.no_grad():
        y_mc_par, st_mc_par = mc(x, state=None, return_state=True)
        st_mc = mc.init_state(2, x.device, x.dtype)
        ys_mc = []
        for t in range(19):
            yt, st_mc = mc(x[:, t : t + 1, :], state=st_mc, return_state=True)
            ys_mc.append(yt)
        assert torch.allclose(y_mc_par, torch.cat(ys_mc, dim=1), atol=1e-5, rtol=1e-5)
        assert st_mc_par is not None and torch.allclose(st_mc_par, st_mc, atol=1e-5)

    # 2. Exact V5 Echoic SWA + Fused Delta batch vs token-by-token streaming equivalence!
    swa = _V5EchoicSWABlock(d_model=32, embed_dim=16, n_heads=4, n_kv_heads=2, d_ff=64, window_size=6, conv_kernel=4).eval()
    gdn = _V5FusedDeltaBlock(d_model=32, embed_dim=16, n_heads=4, d_ff=64, chunk_size=4, conv_kernel=4).eval()
    # Non-zero init on output projections for non-trivial equivalence check
    torch.nn.init.normal_(swa.o_proj.weight, 0.0, 0.1)
    torch.nn.init.normal_(swa.w_down.weight, 0.0, 0.1)
    torch.nn.init.normal_(gdn.o_proj.weight, 0.0, 0.1)
    torch.nn.init.normal_(gdn.w_down.weight, 0.0, 0.1)

    x0 = torch.randn(2, 15, 32)
    e_tok = torch.randn(2, 15, 16)
    half = 4
    inv_f = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(15, dtype=torch.float32)
    freqs = torch.outer(pos, inv_f)
    cos = freqs.cos().view(1, 1, 15, half)
    sin = freqs.sin().view(1, 1, 15, half)

    with torch.no_grad():
        y_swa_par, _, st_swa_par, _ = swa([x0, x[:, :15, :]], e_tok, None, cos, sin, block_mask=None, state=None, return_state=True)
        st_s = swa.init_state(2, x.device, x.dtype)
        ys_s = []
        for t in range(15):
            yt, _, st_s, _ = swa(
                [x0[:, t : t + 1, :], x[:, t : t + 1, :]],
                e_tok[:, t : t + 1, :],
                None,
                cos[:, :, t : t + 1, :],
                sin[:, :, t : t + 1, :],
                block_mask=None,
                state=st_s,
                return_state=True,
            )
            ys_s.append(yt)
        assert torch.allclose(y_swa_par, torch.cat(ys_s, dim=1), atol=1e-4, rtol=1e-4)
        assert st_s is not None and st_s["k"].shape[2] == 6  # Strictly bounded O(1) ring buffer!

        y_gdn_par, st_gdn_par, _, _ = gdn([x0, x[:, :15, :]], e_tok, return_state=True)
        st_g = gdn.init_state(2, x.device, x.dtype)
        ys_g = []
        for t in range(15):
            yt, st_g, _ = gdn.step([x0[:, t : t + 1, :], x[:, t : t + 1, :]], e_tok[:, t : t + 1, :], st_g)
            ys_g.append(yt)
        assert torch.allclose(y_gdn_par, torch.cat(ys_g, dim=1), atol=1e-4, rtol=1e-4)
        assert st_gdn_par is not None and torch.allclose(st_gdn_par["S"], st_g["S"], atol=1e-4, rtol=1e-4)

    # 3. Verify Clean Muon Matrix Separation: zero `_no_muon` gate matrices enter Muon group!
    pm = PULSEv4Model(PULSEv4Config.micro_test())
    muon_opt = FairMuonAdamW(pm.io_parameters())
    for p in muon_opt.param_groups[0]["params"]:
        assert not getattr(p, "_no_muon", False), "Gate parameter leaked into Muon orthogonalization!"


def test_context_gated_engram_causality_and_streaming_exact_match() -> None:
    torch.manual_seed(0)
    eng = ContextGatedEngram(d_model=32, num_heads=4, num_buckets=256, head_dim=8)
    ids = torch.randint(4, 200, (2, 17))
    h = torch.randn(2, 17, 32)

    mem_par, prev_par = eng.lookup(ids)
    out_par = eng.inject(h, mem_par)
    assert out_par.shape == h.shape
    assert prev_par.shape == (2, 2)

    # Exact token-by-token streaming match!
    prev = eng.init_state(2, ids.device)
    outs = []
    for t in range(ids.shape[1]):
        m_t, prev = eng.lookup(ids[:, t : t + 1], prev_tokens=prev)
        outs.append(eng.inject(h[:, t : t + 1, :], m_t))
    out_seq = torch.cat(outs, dim=1)
    assert torch.allclose(out_par, out_seq, atol=1e-5, rtol=1e-5)
    assert torch.equal(prev_par, prev)

    # Strict causality: mutating future tokens cannot change past outputs!
    ids_mut = ids.clone()
    ids_mut[:, 9:] = (ids_mut[:, 9:] + 31) % 200
    mem_mut, _ = eng.lookup(ids_mut)
    assert torch.allclose(mem_par[:, :9, :], mem_mut[:, :9, :], atol=1e-6)


def test_fast_gated_deltanet_wy_vs_step_and_innovation_radar() -> None:
    torch.manual_seed(1)
    gdn = FastGatedDeltaNet(d_model=32, n_heads=4, chunk_size=4, conv_kernel=4)
    x = torch.randn(2, 14, 32, requires_grad=True)

    y_par, st_par, inn_par = gdn(x, return_innovation=True, return_state=True)
    st = gdn.init_state(2, device=x.device, dtype=x.dtype)
    ys, inns = [], []
    for t in range(x.shape[1]):
        yt, st, it = gdn.step(x[:, t : t + 1, :], st, return_innovation=True)
        ys.append(yt)
        inns.append(it)
    y_seq = torch.cat(ys, dim=1)
    inn_seq = torch.cat(inns, dim=1)
    assert torch.allclose(y_par, y_seq, atol=1e-4, rtol=1e-4)
    assert torch.allclose(st_par["S"], st["S"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(inn_par, inn_seq, atol=1e-4, rtol=1e-4)

    y_par.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_pulse_v4_zero_lag_causality_o1_state_streaming_and_iso_param_parity() -> None:
    torch.manual_seed(2)
    cfg = PULSEv4Config.micro_test()
    model = PULSEv4Model(cfg).eval()

    # 1. Strict Zero-Lag causality check across M=2 chunk boundary!
    x_a = torch.randint(4, cfg.vocab_size, (1, 16))
    x_b = x_a.clone()
    x_b[:, 8:] = (x_b[:, 8:] + 37) % cfg.vocab_size
    with torch.no_grad():
        la = model(x_a, use_fast_weights=False)["logits"]
        lb = model(x_b, use_fast_weights=False)["logits"]
    assert torch.allclose(la[:, :8, :], lb[:, :8, :], atol=1e-5), (
        f"Future leak detected: {(la[:, :8, :] - lb[:, :8, :]).abs().max().item()}"
    )

    # 2. Strictly O(1) working state across 4x sequence length!
    with torch.no_grad():
        out_s = model(torch.randint(4, cfg.vocab_size, (1, 16)), use_fast_weights=False)
        out_l = model(torch.randint(4, cfg.vocab_size, (1, 64)), use_fast_weights=False)
    assert model.working_state_bytes(out_s["working_state"]) == model.working_state_bytes(out_l["working_state"])

    # 3. Streaming generation works cleanly
    gen = model.generate(x_a[:, :8], max_new_tokens=10, temperature=0.8)
    assert gen.shape == (1, 18)

    # 4. Strict <= 1.0x trainable parameter parity at BOTH 30M and 100M!
    for p_cfg, t_cfg in [
        (PULSEv4Config.colab_30m(), ScalableTransformerConfig.colab_30m()),
        (PULSEv4Config.colab_100m(), ScalableTransformerConfig.colab_100m()),
    ]:
        pm = PULSEv4Model(p_cfg)
        tm = ScalableTransformerLM(t_cfg)
        p_tr = sum(p.numel() for p in pm.io_parameters())
        t_tr = tm.num_parameters()
        assert p_tr <= t_tr, f"PULSE-v4 trainable ({p_tr:,}) exceeded Transformer ({t_tr:,})!"
        assert p_tr >= int(t_tr * 0.96), f"PULSE-v4 trainable ({p_tr:,}) too far below Transformer ({t_tr:,})!"
        assert pm.core_param_count() > 500_000


def test_pulse_v4_training_zero_core_grads_rls_online_sleep_and_pmi_anti_hallucination() -> None:
    torch.manual_seed(3)
    cfg = PULSEv4Config.micro_test()
    model = PULSEv4Model(cfg)
    opt = torch.optim.AdamW(model.io_parameters(), lr=4e-3, weight_decay=0.05)

    k = cfg.semantic_clock
    story_a = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 10]])
    story_b = torch.tensor([[80, 81, 82, 83, 84, 85, 86, 87, 90, 91, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107, 80]])
    batch = torch.cat([story_a, story_b, story_a, story_b], dim=0)

    losses = []
    for _ in range(20):
        st = model.train_step_hybrid_local(batch, io_optimizer=opt, async_core=True)
        losses.append(st["ce_loss"])
        assert model.core_autograd_grad_count() == 0
    assert losses[-1] < losses[0] * 0.5

    model.eval()
    # Bayesian PMI Anti-Hallucination check
    with torch.no_grad():
        prompts = batch[:2, : k * 2]
        t_cont = batch[:2, k * 2 : k * 3]
        f_cont = torch.roll(t_cont, shifts=1, dims=0)
        e_true = model.continuation_free_energy(prompts, t_cont, use_fast_weights=False)
        e_false = model.continuation_free_energy(prompts, f_cont, use_fast_weights=False)
        assert e_false.mean().item() > e_true.mean().item() * 2.0

    # RLS Online Teaching -> 100% Concept Recall + Orthogonal Sleep -> W_fast = 0
    cues = torch.randint(130, 240, (6, k * 2))
    tgts = torch.randint(130, 240, (6, k))
    with torch.no_grad():
        tgt_bank = model.encode_target_concept(tgts)

        def acc(use_fast: bool) -> float:
            pred = model.predict_next_concept(cues, use_fast_weights=use_fast)
            sims = F.cosine_similarity(pred.unsqueeze(1), tgt_bank.unsqueeze(0), dim=-1)
            return float((sims.argmax(dim=-1) == torch.arange(6)).float().mean().item())

    model.teach_fact_online(cues, tgts)
    assert model.core_autograd_grad_count() == 0
    assert acc(use_fast=True) == 1.0
    # Verify exact closed-form LM Top-1 flip = 100.0%!
    with torch.no_grad():
        pred_top1 = model(cues, use_fast_weights=True)["logits"][:, -1, :].argmax(dim=-1)
        assert float((pred_top1 == tgts[:, 0]).float().mean().item()) == 1.0

    model.sleep_consolidate(num_replay_samples=24, clear_fast_weights=True)
    assert model.causal_layer.w_fast.norm().item() == 0.0
    assert acc(use_fast=False) >= 0.83
