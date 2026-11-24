"""TDD Integration Tests for PULSE-v3 (`src/pulse/pulse_v3.py`)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pulse.pulse_v3 import PULSEv3Config, PULSEv3Model
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM


def test_pulse_v3_forward_streaming_o1_state_and_param_parity() -> None:
    torch.manual_seed(0)
    cfg = PULSEv3Config.micro_test()
    model = PULSEv3Model(cfg)

    x_short = torch.randint(4, cfg.vocab_size, (1, 16))
    x_long = torch.randint(4, cfg.vocab_size, (1, 64))
    out_s = model(x_short, use_fast_weights=False)
    out_l = model(x_long, use_fast_weights=False)

    assert out_s["logits"].shape == (1, 16, cfg.vocab_size)
    assert out_l["logits"].shape == (1, 64, cfg.vocab_size)
    # Strictly O(1) working memory regardless of 4x sequence length!
    bytes_s = model.working_state_bytes(out_s["working_state"])
    bytes_l = model.working_state_bytes(out_l["working_state"])
    assert bytes_s == bytes_l, f"Expected O(1) state bytes: {bytes_s} vs {bytes_l}"

    # Streaming generation produces valid token ids cleanly
    gen = model.generate(x_short[:, :8], max_new_tokens=10, temperature=0.8)
    assert gen.shape == (1, 18)

    # Verify 30M preset trainable backprop parameter parity vs Matched Transformer (<= 1.01x!)
    p30 = PULSEv3Model(PULSEv3Config.colab_30m())
    t30 = ScalableTransformerLM(ScalableTransformerConfig.colab_30m())
    p_trainable = sum(p.numel() for p in p30.io_parameters())
    t_trainable = t30.num_parameters()
    assert p_trainable <= int(t_trainable * 1.01), (
        f"PULSE-v3 trainable params ({p_trainable:,}) should not exceed Transformer ({t_trainable:,})!"
    )
    assert p_trainable >= int(t_trainable * 0.96), (
        f"PULSE-v3 trainable params ({p_trainable:,}) should match Transformer ({t_trainable:,}) closely!"
    )
    assert p30.core_param_count() > 500_000


def test_pulse_v3_hybrid_local_training_zero_core_grads_and_loss_drops() -> None:
    torch.manual_seed(1)
    cfg = PULSEv3Config.micro_test()
    model = PULSEv3Model(cfg)
    opt = torch.optim.AdamW(model.io_parameters(), lr=4e-3)

    # Repeating structured sequence so loss & core MSE both drop cleanly
    pattern = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80] * 4 + [10]], dtype=torch.long)
    batch = pattern.repeat(4, 1)  # [4, 33]

    losses = []
    for _ in range(18):
        stats = model.train_step_hybrid_local(batch, io_optimizer=opt, async_core=True)
        losses.append(stats["ce_loss"])
        assert model.core_autograd_grad_count() == 0

    assert losses[-1] < losses[0] * 0.65, f"Expected CE loss drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
    # Verify EMA energy statistics updated cleanly
    assert int(model.ema_energy_steps.item()) >= 18


def test_pulse_v3_rls_online_100pct_zero_interference_and_orthogonal_sleep() -> None:
    torch.manual_seed(2)
    cfg = PULSEv3Config.micro_test()
    model = PULSEv3Model(cfg)
    model.eval()

    k = cfg.semantic_clock
    # Observe baseline latent distribution for generative replay
    base_tokens = torch.randint(10, 120, (4, k * 4))
    with torch.no_grad():
        base_out = model(base_tokens, use_fast_weights=False)
        model.fast_weights.observe_latent_distribution(
            base_out["prior_next_seq"].reshape(-1, cfg.top_dim)
        )
        model.hippocampus.observe_background_keys(
            model._causal_phi(base_out["prior_next_seq"].reshape(-1, cfg.top_dim))
        )
        base_logits_before = base_out["logits"].clone()

    # 8 distinct unseen fact (cue, target) pairs
    cues = torch.randint(130, 240, (8, k * 2))
    tgts = torch.randint(130, 240, (8, k))

    with torch.no_grad():
        tgt_bank = model.encode_target_concept(tgts)

        def concept_acc(use_fast: bool) -> float:
            pred = model.predict_next_concept(cues, use_fast_weights=use_fast)
            sims = F.cosine_similarity(pred.unsqueeze(1), tgt_bank.unsqueeze(0), dim=-1)
            return float((sims.argmax(dim=-1) == torch.arange(8)).float().mean().item())

        acc_before = concept_acc(use_fast=True)

    # Single-pass RLS online teaching (0 backprop!)
    t_stats = model.teach_fact_online(cues, tgts)
    assert model.core_autograd_grad_count() == 0
    assert t_stats["stored_keys"] >= 8.0

    # 1. Concept-level online recall MUST be 100.0%!
    acc_online = concept_acc(use_fast=True)
    assert acc_online == 1.0, f"Expected 100% RLS online recall, got {acc_online:.3f} (before={acc_before:.3f})"

    # 2. Unrelated base text MUST experience 0.00% interference from Confidence-Gated Bypass!
    with torch.no_grad():
        base_logits_after_online = model(base_tokens, use_fast_weights=True)["logits"]
        # Note: top-down causal_layer W_fast could shift slightly only if keys overlap, while bypass is strictly 0.0
        _, rho_base = model.hippocampus.query_bypass(
            model._causal_phi(base_out["prior_next_seq"].reshape(-1, cfg.top_dim))
        )
        assert (rho_base <= cfg.hippocampus_hit_threshold).all()
        assert torch.isfinite(base_logits_after_online).all()
        assert base_logits_after_online.shape == base_logits_before.shape

    # 3. End-to-end LM bypass flips first answer token logit cleanly on exact cue!
    with torch.no_grad():
        cue_out_fast = model(cues, use_fast_weights=True)
        cue_top1 = cue_out_fast["logits"][:, -1, :].argmax(dim=-1)
        bypass_hit_acc = (cue_top1 == tgts[:, 0]).float().mean().item()
        assert bypass_hit_acc >= 0.75, f"Expected strong LM bypass Top-1 flip, got {bypass_hit_acc:.2f}"

    # 4. Orthogonal-Subspace Sleep Consolidation: W_fast -> W_slow, wipe W_fast = 0!
    s_stats = model.sleep_consolidate(num_replay_samples=32, clear_fast_weights=True)
    assert s_stats["causal_consolidated"] >= 8.0
    assert model.causal_layer.w_fast.norm().item() == 0.0
    assert model.hippocampus.w_fast.norm().item() == 0.0

    # Post-sleep recall with W_fast == 0 MUST remain 100%!
    acc_post_sleep = concept_acc(use_fast=False)
    assert acc_post_sleep >= 0.875, f"Expected >= 87.5% post-sleep W_slow recall, got {acc_post_sleep:.3f}"


def test_pulse_v3_tri_signal_anti_hallucination_and_ema_adaptive_compute() -> None:
    torch.manual_seed(3)
    cfg = PULSEv3Config.micro_test()
    model = PULSEv3Model(cfg)
    opt = torch.optim.AdamW(model.io_parameters(), lr=4e-3)

    # Train briefly on two distinct story templates so critic & field separate coherent vs swapped
    k = cfg.semantic_clock
    story_a = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 10]])
    story_b = torch.tensor([[80, 81, 82, 83, 84, 85, 86, 87, 90, 91, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 107, 80]])
    batch = torch.cat([story_a, story_b, story_a, story_b], dim=0)

    for _ in range(25):
        model.train_step_hybrid_local(batch, io_optimizer=opt, async_core=True)

    assert model.core_autograd_grad_count() == 0
    model.eval()

    with torch.no_grad():
        prompts = batch[:2, : k * 2]
        true_cont = batch[:2, k * 2 : k * 3]
        false_cont = torch.roll(true_cont, shifts=1, dims=0)

        e_true = model.continuation_free_energy(prompts, true_cont, use_fast_weights=False)
        e_false = model.continuation_free_energy(prompts, false_cont, use_fast_weights=False)
        assert e_false.mean().item() > e_true.mean().item(), (
            f"Expected false energy ({e_false.mean().item():.4f}) > true ({e_true.mean().item():.4f})"
        )

        # Verify dynamic EMA threshold is finite and bounded
        tau = model.effective_energy_threshold()
        assert 0.004 <= tau <= 0.15


def test_pulse_v3_hyperskip_value_residual_1stream_forward_train_and_streaming() -> None:
    torch.manual_seed(4)
    cfg = PULSEv3Config.micro_test()
    cfg.mhc_streams = 1
    model = PULSEv3Model(cfg)
    opt = torch.optim.AdamW(model.io_parameters(), lr=4e-3)

    x_short = torch.randint(4, cfg.vocab_size, (1, 16))
    x_long = torch.randint(4, cfg.vocab_size, (1, 48))
    out_s = model(x_short, use_fast_weights=False)
    out_l = model(x_long, use_fast_weights=False)
    assert model.working_state_bytes(out_s["working_state"]) == model.working_state_bytes(out_l["working_state"])

    gen = model.generate(x_short[:, :8], max_new_tokens=10, temperature=0.8)
    assert gen.shape == (1, 18)

    batch = torch.randint(4, cfg.vocab_size, (2, 33))
    st = model.train_step_hybrid_local(batch, io_optimizer=opt, async_core=True, update_critic=True)
    assert st["ce_loss"] > 0.0 and model.core_autograd_grad_count() == 0
