"""Tests for Pillar 3: Tri-Timescale Plastic Memory (Fast SSM, Fast Weights, Sleep Consolidation)."""

import torch

from pulse.config import PULSEConfig
from pulse.memory import (
    FastWorkingStateSSM,
    SleepConsolidator,
    SurpriseGatedFastWeights,
)
from pulse.predictive_field import BidirectionalPredictiveField


class TestFastWorkingStateSSM:
    def test_maintains_constant_o1_state_size_and_bounded_norm_over_long_horizon(self):
        torch.manual_seed(0)
        ssm = FastWorkingStateSSM(input_dim=32, state_dim=48)
        state = ssm.init_state(batch_size=2)
        initial_nbytes = state.nelement() * state.element_size()

        for _ in range(500):
            z = torch.randn(2, 32) * 3.0
            state, prior = ssm.step(z, state)

        final_nbytes = state.nelement() * state.element_size()
        assert final_nbytes == initial_nbytes  # Strictly O(1) memory across arbitrary horizon
        assert prior.shape == (2, 32)
        assert torch.isfinite(state).all()
        assert state.abs().max().item() <= 1.05  # Guaranteed bounded contraction

    def test_local_temporal_prediction_update_reduces_next_state_error_without_autograd(self):
        torch.manual_seed(3)
        ssm = FastWorkingStateSSM(input_dim=16, state_dim=24)
        # Repeating deterministic temporal pattern z_0 -> z_1 -> z_2
        seq = [torch.randn(4, 16) * 0.7 for _ in range(4)]

        def eval_mse() -> float:
            with torch.no_grad():
                h = ssm.init_state(batch_size=4)
                errs = []
                for t in range(len(seq) - 1):
                    h, prior = ssm.step(seq[t], h)
                    errs.append((seq[t + 1] - prior).pow(2).mean().item())
                return sum(errs) / len(errs)

        before = eval_mse()
        with torch.no_grad():
            for _ in range(40):
                h = ssm.init_state(batch_size=4)
                for t in range(len(seq) - 1):
                    h, _ = ssm.step(seq[t], h)
                    ssm.update_prior_local(h, seq[t + 1], lr=0.1)
        after = eval_mse()

        assert after < before * 0.2


class TestSurpriseGatedFastWeights:
    def test_surprise_gate_blocks_low_surprise_and_triggers_on_high_surprise(self):
        torch.manual_seed(5)
        cfg = PULSEConfig.micro()
        cfg.surprise_threshold = 0.4
        cfg.surprise_temperature = 0.05
        cfg.fast_weight_alpha = 0.8
        cfg.fast_weight_decay = 0.0

        field = BidirectionalPredictiveField(cfg)
        fast_mem = SurpriseGatedFastWeights(cfg)

        with torch.no_grad():
            # Low surprise: consistent top-down sample -> near-zero error -> W_fast barely changes
            z_top = torch.randn(4, cfg.top_dim) * 0.6
            states_low = field.generate_top_down(z_top)
            stats_low = fast_mem.update(field, states_low)
            norm_after_low = sum(layer.w_fast.norm().item() for layer in field.layers)

            # High surprise: novel rule mapping z_top -> brand new z_bottom
            z_bottom_novel = torch.randn(4, cfg.concept_dim) * 2.5
            states_high = [z_bottom_novel] + states_low[1:]
            stats_high = fast_mem.update(field, states_high)
            norm_after_high = sum(layer.w_fast.norm().item() for layer in field.layers)

        assert stats_low["max_gate"] < 0.05
        assert norm_after_low < 1e-2
        assert stats_high["max_gate"] > 0.9
        assert stats_high["layer_gates"][0] > 0.9
        assert stats_high["layer_gates"][1] < 0.05
        assert norm_after_high > norm_after_low * 10.0

    def test_online_fast_weights_internalize_new_rule_and_decay_naturally(self):
        torch.manual_seed(9)
        cfg = PULSEConfig.micro()
        cfg.surprise_threshold = 0.1
        cfg.surprise_temperature = 0.05
        cfg.fast_weight_alpha = 0.9
        cfg.fast_weight_decay = 0.1

        field = BidirectionalPredictiveField(cfg)
        fast_mem = SurpriseGatedFastWeights(cfg)

        with torch.no_grad():
            # Teach a brand-new unseen association: z_top_rule -> z_bottom_rule
            z_top_rule = torch.randn(2, cfg.top_dim)
            z_bottom_rule = torch.randn(2, cfg.concept_dim)

            # Construct consistent intermediate state for the rule
            mid_states = field.generate_top_down(z_top_rule)
            err_before = (z_bottom_rule - mid_states[0]).pow(2).mean().item()

            # Online inference-time plasticity (a few exposures, zero backprop)
            rule_states = [z_bottom_rule] + mid_states[1:]
            for _ in range(3):
                fast_mem.update(field, rule_states)

            # Recall purely from weights (zero context!)
            recalled = field.generate_top_down(z_top_rule)[0]
            err_after = (z_bottom_rule - recalled).pow(2).mean().item()

            # Verify natural decay shrinks W_fast when no surprise occurs
            norm_peak = field.layers[0].w_fast.norm().item()
            fast_mem.decay_only(field)
            norm_decayed = field.layers[0].w_fast.norm().item()

        assert err_after < err_before * 0.2
        assert norm_decayed < norm_peak

    def test_low_rank_constraint_caps_fast_weight_matrix_rank(self):
        torch.manual_seed(13)
        cfg = PULSEConfig.micro()
        cfg.fast_weight_rank = 4
        cfg.surprise_threshold = 0.05
        cfg.surprise_temperature = 0.02
        cfg.fast_weight_alpha = 0.8

        field = BidirectionalPredictiveField(cfg)
        fast_mem = SurpriseGatedFastWeights(cfg)

        with torch.no_grad():
            for _ in range(10):
                z_top = torch.randn(4, cfg.top_dim)
                states = field.generate_top_down(z_top)
                states[0] = torch.randn(4, cfg.concept_dim) * 2.0
                fast_mem.update(field, states)

            sv = torch.linalg.svdvals(field.layers[0].w_fast)
            # Singular values beyond rank 4 must be numerically zero
            assert sv[4:].max().item() < 1e-4


class TestSleepConsolidator:
    def test_sleep_consolidates_fast_weights_into_slow_while_preserving_old_knowledge(self):
        torch.manual_seed(21)
        cfg = PULSEConfig.micro()
        cfg.surprise_threshold = 0.05
        cfg.surprise_temperature = 0.02
        cfg.fast_weight_alpha = 0.9
        cfg.fast_weight_decay = 0.0
        cfg.consolidation_lr = 0.15
        cfg.consolidation_ewc_lambda = 0.2

        field = BidirectionalPredictiveField(cfg)
        fast_mem = SurpriseGatedFastWeights(cfg)
        consolidator = SleepConsolidator(cfg)

        with torch.no_grad():
            # 1. Old knowledge already encoded in W_slow
            z_top_old = torch.randn(6, cfg.top_dim) * 0.7
            z_bottom_old_truth = field.generate_top_down(z_top_old, use_fast_weights=False)[0]
            fast_mem.observe_latent_distribution(z_top_old)

            # 2. New knowledge taught online into W_fast during wakefulness
            z_top_new = torch.randn(3, cfg.top_dim) * 0.7
            states_new_base = field.generate_top_down(z_top_new, use_fast_weights=False)
            z_bottom_new_truth = torch.randn(3, cfg.concept_dim) * 1.2
            err_new_before_wake = (z_bottom_new_truth - states_new_base[0]).pow(2).mean().item()

            # Repeated wake exposures so episode gets recorded with high salience
            states_new = [z_bottom_new_truth] + states_new_base[1:]
            for _ in range(4):
                fast_mem.update(field, states_new, record_episode=True)

            # 3. Sleep consolidation: transfer W_fast -> W_slow via generative replay, then wipe W_fast
            sleep_stats = consolidator.consolidate(
                field,
                fast_mem,
                num_replay_samples=24,
                num_epochs=35,
                clear_fast_weights=True,
            )

            # Confirm W_fast is cleared after sleep
            assert all(layer.w_fast.norm().item() < 1e-6 for layer in field.layers)

            # 4. Evaluate BOTH new and old knowledge purely from W_slow!
            pred_new_slow = field.generate_top_down(z_top_new, use_fast_weights=False)[0]
            pred_old_slow = field.generate_top_down(z_top_old, use_fast_weights=False)[0]

            err_new_after_sleep = (z_bottom_new_truth - pred_new_slow).pow(2).mean().item()
            err_old_drift = (z_bottom_old_truth - pred_old_slow).pow(2).mean().item()

        assert sleep_stats["consolidated_episodes"] >= 1
        assert err_new_after_sleep < err_new_before_wake * 0.35
        assert err_old_drift < 0.25  # Old knowledge preserved without catastrophic forgetting

    def test_load_latent_state_normalizes_and_subsamples_large_batches_cleanly(self):
        cfg = PULSEConfig.micro()
        fast_mem = SurpriseGatedFastWeights(cfg)
        fast_mem.load_latent_state(
            latent_mean=torch.ones(cfg.top_dim, dtype=torch.bfloat16),
            latent_var=torch.ones(cfg.top_dim, dtype=torch.bfloat16),
            latent_count=10,
            prior_centroids=[torch.zeros(cfg.top_dim, dtype=torch.bfloat16)],
        )
        assert fast_mem._latent_mean is not None and fast_mem._latent_mean.dtype == torch.float32
        assert fast_mem._latent_mean.device.type == "cpu"
        fast_mem.observe_latent_distribution(torch.randn(256, cfg.top_dim))
        assert fast_mem._latent_count == 266
        assert len(fast_mem._prior_centroids) <= fast_mem.max_prior_centroids
