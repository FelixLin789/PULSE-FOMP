"""TDD Unit Tests for PULSE-v3 Frontier Building Blocks (`src/pulse/v3_blocks.py`)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pulse.v3_blocks import (
    CanonLayer,
    GatedDeltaNet2,
    GatedSlidingWindowAttention,
    ManifoldHyperConnection,
    RLSHippocampalMemory,
    TriSignalContrastiveCritic,
    sinkhorn_knopp,
)


def test_canon_layer_causality_and_streaming_exact_match() -> None:
    torch.manual_seed(0)
    canon = CanonLayer(dim=32, kernel_size=4)
    x = torch.randn(2, 17, 32)

    y_par, st_par = canon(x)
    assert y_par.shape == x.shape
    assert st_par.shape == (2, 32, 3)

    # Exact token-by-token streaming match
    st = canon.init_state(2, device=x.device, dtype=x.dtype)
    ys = []
    for t in range(x.shape[1]):
        yt, st = canon.step(x[:, t : t + 1, :], st)
        ys.append(yt)
    y_seq = torch.cat(ys, dim=1)
    assert torch.allclose(y_par, y_seq, atol=1e-5, rtol=1e-5)
    assert torch.allclose(st_par, st, atol=1e-5, rtol=1e-5)

    # Strict causality: mutating future tokens cannot affect past outputs
    x_mut = x.clone()
    x_mut[:, 8:, :] += 10.0
    y_mut, _ = canon(x_mut)
    assert torch.allclose(y_par[:, :8, :], y_mut[:, :8, :], atol=1e-6)

    # Strictly O(1) state independent of sequence length
    _, st_long = canon(torch.randn(2, 64, 32))
    assert st_par.nelement() == st_long.nelement()


def test_mhc_sinkhorn_birkhoff_doubly_stochastic_and_deep_stability() -> None:
    torch.manual_seed(1)
    logits = torch.randn(4, 16, 4, 4, requires_grad=True)
    h_res = sinkhorn_knopp(logits, num_iters=10)

    assert (h_res >= 0.0).all()
    assert torch.allclose(h_res.sum(dim=-1), torch.ones(4, 16, 4), atol=1e-3)
    assert torch.allclose(h_res.sum(dim=-2), torch.ones(4, 16, 4), atol=1e-3)
    h_res.pow(2).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    # Stack 16 mHC layers with zero sublayer output -> preserves signal norm without explosion!
    layers = [ManifoldHyperConnection(d_model=32, n_streams=4, sinkhorn_iters=8) for _ in range(16)]
    x0 = torch.randn(2, 12, 32)
    streams = layers[0].expand_input(x0)  # [B, T, n, D]
    init_norm = streams.norm().item()
    for mhc in layers:
        sub_in, h_res_m, h_post = mhc.pre_mix(streams)
        assert sub_in.shape == (2, 12, 32)
        streams = mhc.post_mix(streams, torch.zeros_like(sub_in), h_res_m, h_post)
    final_norm = streams.norm().item()
    # Doubly stochastic contraction never explodes norm!
    assert final_norm <= init_norm * 1.05
    assert final_norm >= init_norm * 0.5


def test_gated_deltanet2_exact_wy_vs_step_and_innovation_radar() -> None:
    torch.manual_seed(2)
    gdn2 = GatedDeltaNet2(d_model=32, n_heads=4, chunk_size=4, gate_rank=16, conv_kernel=4)
    x = torch.randn(2, 14, 32, requires_grad=True)

    # 1. Parallel WY forward vs exact sequential step() match
    y_par, st_par = gdn2(x, return_innovation=False)
    assert y_par.shape == x.shape

    st = gdn2.init_state(2, device=x.device, dtype=x.dtype)
    ys = []
    innov_seq = []
    for t in range(x.shape[1]):
        yt, st, inn_t = gdn2.step(x[:, t : t + 1, :], st, return_innovation=True)
        ys.append(yt)
        innov_seq.append(inn_t)
    y_seq = torch.cat(ys, dim=1)
    innov_seq_t = torch.cat(innov_seq, dim=1)  # [B, T]

    assert torch.allclose(y_par, y_seq, atol=1e-4, rtol=1e-4), (
        f"Max diff y: {(y_par - y_seq).abs().max().item()}"
    )
    assert torch.allclose(st_par["S"], st["S"], atol=1e-4, rtol=1e-4), (
        f"Max diff S: {(st_par['S'] - st['S']).abs().max().item()}"
    )

    # 2. Exact Delta-Innovation Radar matches token-by-token innovation norms!
    y_par2, _, innov_par = gdn2(x.detach(), return_innovation=True)
    assert torch.allclose(y_par2, y_par.detach(), atol=1e-5)
    assert innov_par is not None and innov_par.shape == (2, 14)
    assert torch.allclose(innov_par, innov_seq_t, atol=1e-4, rtol=1e-4), (
        f"Max diff innovation: {(innov_par - innov_seq_t).abs().max().item()}"
    )

    # 3. Clean backward pass & strictly O(1) state bytes
    y_par.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    _, st_short = gdn2(torch.randn(1, 16, 32))
    _, st_long = gdn2(torch.randn(1, 80, 32))
    assert GatedDeltaNet2.state_bytes(st_short) == GatedDeltaNet2.state_bytes(st_long)


def test_gated_sliding_window_attention_exact_streaming_and_o1_ring() -> None:
    torch.manual_seed(3)
    swa = GatedSlidingWindowAttention(d_model=32, n_heads=4, window_size=6, conv_kernel=4)
    x = torch.randn(2, 19, 32, requires_grad=True)

    y_par, st_par = swa(x)
    assert y_par.shape == x.shape
    assert st_par["k"].shape[2] == 6  # Strictly capped to window_size!

    st = swa.init_state(2, device=x.device, dtype=x.dtype)
    ys = []
    for t in range(x.shape[1]):
        yt, st = swa.step(x[:, t : t + 1, :], st)
        ys.append(yt)
    y_seq = torch.cat(ys, dim=1)

    assert torch.allclose(y_par, y_seq, atol=1e-4, rtol=1e-4), (
        f"Max diff SWA: {(y_par - y_seq).abs().max().item()}"
    )
    assert torch.allclose(st_par["k"], st["k"], atol=1e-5)

    # Backward & O(1) state check
    y_par.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    _, st_long = swa(torch.randn(2, 48, 32))
    assert GatedSlidingWindowAttention.state_bytes(st_par) == GatedSlidingWindowAttention.state_bytes(st_long)


def test_rls_hippocampal_memory_100pct_online_zero_interference_and_orthogonal_sleep() -> None:
    torch.manual_seed(4)
    key_dim, val_dim, emb_dim = 48, 32, 24
    hipp = RLSHippocampalMemory(
        key_dim=key_dim,
        val_dim=val_dim,
        bypass_dim=emb_dim,
        max_episodic_keys=64,
        hit_threshold=0.85,
        ridge_lambda=1e-2,
    )
    # Verify 100% zero-backprop parameters/buffers
    assert all(not p.requires_grad for p in hipp.parameters())

    # Simulate pre-existing slow knowledge W_slow on 40 prior directions
    w_slow = torch.randn(val_dim, key_dim) * 0.2
    b_slow = torch.zeros(val_dim)
    prior_keys = F.normalize(torch.randn(40, key_dim), dim=-1)
    prior_targets = F.linear(prior_keys, w_slow, b_slow)

    # 16 correlated unseen facts to teach online in ONE single pass!
    base_dir = F.normalize(torch.randn(1, key_dim), dim=-1)
    new_keys = F.normalize(0.45 * base_dir + 0.55 * torch.randn(16, key_dim), dim=-1)
    new_vals = F.normalize(torch.randn(16, val_dim), dim=-1)
    new_bypass_targets = F.normalize(torch.randn(16, emb_dim), dim=-1)

    # Teach online (single RLS pass, 0 backprop!)
    stats = hipp.teach_rls_online(
        keys=new_keys,
        val_residuals=new_vals - F.linear(new_keys, w_slow, b_slow),
        bypass_deltas=new_bypass_targets,
    )
    assert stats["stored_keys"] == 16.0

    # 1. Online concept recall MUST be 100.0%!
    pred_online = F.linear(new_keys, w_slow, b_slow) + hipp.query_value_delta(new_keys)
    sims = F.cosine_similarity(pred_online.unsqueeze(1), new_vals.unsqueeze(0), dim=-1)
    acc_online = (sims.argmax(dim=-1) == torch.arange(16)).float().mean().item()
    assert acc_online == 1.0, f"Expected 100% RLS online recall, got {acc_online:.3f}"

    # 2. Confidence-gated bypass MUST return strictly 0.0 on unrelated queries (0.00% interference!)
    unrelated = F.normalize(torch.randn(20, key_dim), dim=-1)
    # Ensure unrelated queries are truly below threshold
    max_cos_unrel = F.cosine_similarity(unrelated.unsqueeze(1), new_keys.unsqueeze(0), dim=-1).max(dim=-1).values
    strictly_unhit = unrelated[max_cos_unrel < 0.70]
    bypass_unhit, conf_unhit = hipp.query_bypass(strictly_unhit)
    assert bypass_unhit.abs().max().item() == 0.0
    assert (conf_unhit < 0.85).all()

    # Matched cues activate bypass cleanly and retrieve exact bypass targets
    bypass_hit, conf_hit = hipp.query_bypass(new_keys)
    assert (conf_hit > 0.99).all()
    b_sims = F.cosine_similarity(bypass_hit.unsqueeze(1), new_bypass_targets.unsqueeze(0), dim=-1)
    assert (b_sims.argmax(dim=-1) == torch.arange(16)).all()

    # 3. Closed-form Orthogonal-Subspace Sleep Consolidation: W_fast -> W_slow, wipe W_fast = 0!
    importance = torch.ones_like(w_slow)
    sleep_stats = hipp.sleep_consolidate_orthogonal(
        w_slow=w_slow,
        b_slow=b_slow,
        importance_w=importance,
        replay_keys=prior_keys,
        clear_fast_weights=True,
    )
    assert hipp.w_fast.norm().item() == 0.0
    assert sleep_stats["consolidated_facts"] == 16.0

    # Post-sleep recall using ONLY W_slow (W_fast == 0!) MUST stay >= 93.75% (target 100%)
    pred_post = F.linear(new_keys, w_slow, b_slow)
    sims_post = F.cosine_similarity(pred_post.unsqueeze(1), new_vals.unsqueeze(0), dim=-1)
    acc_post = (sims_post.argmax(dim=-1) == torch.arange(16)).float().mean().item()
    assert acc_post >= 0.9375, f"Expected >= 93.75% post-sleep W_slow recall, got {acc_post:.3f}"

    # Old prior knowledge drift MUST be tiny!
    old_mse = (F.linear(prior_keys, w_slow, b_slow) - prior_targets).pow(2).mean().item()
    assert old_mse < 0.02, f"Old replay drift too large: {old_mse:.4f}"


def test_tri_signal_contrastive_critic_zero_backprop_hard_negatives_and_lse_peak() -> None:
    torch.manual_seed(5)
    critic = TriSignalContrastiveCritic(
        top_dim=32,
        concept_dim=24,
        tok_ctx_dim=32,
        tok_emb_dim=16,
        proj_dim=24,
    )
    assert all(not p.requires_grad for p in critic.parameters())

    # Synthetic coherent ground-truth mapping
    gt_concept = torch.randn(24, 32) * 0.4
    gt_tok = torch.randn(16, 32) * 0.4

    for _ in range(35):
        u_ctx = F.normalize(torch.randn(48, 32), dim=-1)
        z_true = F.normalize(F.linear(u_ctx, gt_concept) + 0.05 * torch.randn(48, 24), dim=-1)
        # Hard negative: keep 70% of true concept, perturb 30% (entity swap!)
        z_hard = F.normalize(0.65 * z_true + 0.75 * torch.roll(z_true, shifts=1, dims=0), dim=-1)

        h_tok = F.normalize(torch.randn(16, 8, 32), dim=-1)
        e_tok_true = F.normalize(F.linear(h_tok, gt_tok) + 0.05 * torch.randn(16, 8, 16), dim=-1)
        e_tok_hard = e_tok_true.clone()
        e_tok_hard[:, 3, :] = -e_tok_hard[:, 3, :]  # Single-word entity contradiction!

        delta_innov = torch.rand(16, 8) * 0.5 + 0.5
        critic.update_wake_sleep_local(
            u_ctx=u_ctx,
            z_true=z_true,
            z_hard_neg=z_hard,
            h_tok_ctx=h_tok,
            e_tok_true=e_tok_true,
            e_tok_hard_neg=e_tok_hard,
            delta_innov=delta_innov,
            lr=0.06,
        )

    # Strictly zero autograd gradients
    assert sum(1 for p in critic.parameters() if p.grad is not None) == 0

    # Evaluate on held-out single-token contradiction + Delta innovation spike
    u_eval = F.normalize(torch.randn(24, 32), dim=-1)
    z_eval_true = F.normalize(F.linear(u_eval, gt_concept), dim=-1)
    z_eval_bad = F.normalize(0.65 * z_eval_true + 0.75 * torch.roll(z_eval_true, shifts=1, dims=0), dim=-1)

    h_eval = F.normalize(torch.randn(24, 8, 32), dim=-1)
    e_eval_true = F.normalize(F.linear(h_eval, gt_tok), dim=-1)
    e_eval_bad = e_eval_true.clone()
    e_eval_bad[:, 4, :] = -e_eval_bad[:, 4, :]

    inn_true = torch.full((24, 8), 0.75)
    inn_bad = torch.full((24, 8), 0.75)
    inn_bad[:, 4] = 2.4  # Contradiction spike at token 4!

    score_true = critic.tri_signal_energy(u_eval, z_eval_true, h_eval, e_eval_true, inn_true)
    score_bad = critic.tri_signal_energy(u_eval, z_eval_bad, h_eval, e_eval_bad, inn_bad)
    auroc = (score_bad.unsqueeze(1) > score_true.unsqueeze(0)).float().mean().item()
    assert auroc >= 0.95, f"Expected Tri-Signal AUROC >= 0.95, got {auroc:.3f}"
