"""Tests for Pillar 2: Bidirectional Predictive Coding Field."""

import pytest
import torch

from pulse.config import PULSEConfig
from pulse.predictive_field import BidirectionalPredictiveField, PredictiveLayer


class TestPredictiveLayer:
    def test_top_down_prediction_uses_slow_plus_fast_weights(self):
        layer = PredictiveLayer(upper_dim=16, lower_dim=12, precision_min=0.1)
        z_upper = torch.randn(4, 16)
        z_lower = torch.randn(4, 12)

        pred_slow_only = layer.predict_down(z_upper)
        assert pred_slow_only.shape == (4, 12)

        # Inject non-zero fast weights and confirm prediction shifts accordingly
        with torch.no_grad():
            layer.w_fast.fill_(0.25)
        pred_with_fast = layer.predict_down(z_upper)
        assert not torch.allclose(pred_slow_only, pred_with_fast)

        err = layer.prediction_error(z_lower, z_upper)
        assert torch.allclose(err, z_lower - pred_with_fast)

    def test_analytical_local_gradients_match_autograd_exactly(self):
        torch.manual_seed(42)
        layer = PredictiveLayer(upper_dim=10, lower_dim=8, precision_min=0.1)
        with torch.no_grad():
            layer.w_fast.normal_(0.0, 0.1)

        z_upper = torch.randn(5, 10, requires_grad=True)
        z_lower = torch.randn(5, 8, requires_grad=True)

        # Autograd reference for batch-mean energy
        energy_per_sample = layer.energy(z_lower, z_upper)
        total_energy = energy_per_sample.sum()
        mean_energy = energy_per_sample.mean()

        grad_z_lower_ref, grad_z_upper_ref = torch.autograd.grad(
            total_energy, (z_lower, z_upper), retain_graph=True
        )
        grad_w_ref, grad_b_ref = torch.autograd.grad(
            mean_energy, (layer.w_slow, layer.b_slow)
        )

        # Pure analytical local gradients under torch.no_grad()
        with torch.no_grad():
            local_state_grads = layer.local_state_gradients(z_lower.detach(), z_upper.detach())
            local_weight_grads = layer.local_weight_gradients(z_lower.detach(), z_upper.detach())

        assert torch.allclose(local_state_grads["dE_dz_lower"], grad_z_lower_ref, atol=1e-5)
        assert torch.allclose(local_state_grads["dE_dz_upper"], grad_z_upper_ref, atol=1e-5)
        assert torch.allclose(local_weight_grads["dE_dw_slow"], grad_w_ref, atol=1e-5)
        assert torch.allclose(local_weight_grads["dE_db_slow"], grad_b_ref, atol=1e-5)


class TestBidirectionalPredictiveField:
    def test_relaxation_monotonically_decreases_free_energy_without_autograd(self):
        torch.manual_seed(7)
        cfg = PULSEConfig.micro()
        cfg.max_think_steps = 20
        cfg.energy_threshold = 1e-6
        field = BidirectionalPredictiveField(cfg)

        z_bottom = torch.randn(6, cfg.concept_dim)
        z_prior = torch.randn(6, cfg.top_dim)

        with torch.no_grad():
            result = field.relax(z_bottom, top_prior=z_prior)

        traj = result.energy_trajectory  # [num_recorded_steps, B]
        assert traj.shape[0] >= 2
        assert result.final_energy.shape == (6,)
        # Final energy must be strictly lower than initial energy for every sample
        assert (result.final_energy < result.initial_energy).all()
        # Monotonic non-increasing energy per sample (Lyapunov safeguard)
        diffs = traj[1:] - traj[:-1]
        assert (diffs <= 1e-6).all()

    def test_adaptive_compute_uses_fewer_steps_for_easy_consistent_inputs_than_hard_conflicts(self):
        torch.manual_seed(11)
        cfg = PULSEConfig.micro()
        cfg.max_think_steps = 25
        cfg.energy_threshold = 0.05
        field = BidirectionalPredictiveField(cfg)

        # Construct an "easy" sample already near an internal generative attractor
        with torch.no_grad():
            z_top_easy = torch.randn(4, cfg.top_dim) * 0.5
            states_easy = field.generate_top_down(z_top_easy)
            z_bottom_easy = states_easy[0]
            # Prime amortized initializer on this easy manifold region by overriding init
            easy_res = field.relax(
                z_bottom_easy,
                top_prior=z_top_easy,
                init_states=states_easy,
            )

            # Construct a "hard" conflicting sample: bottom sensory input contradicts top prior
            z_bottom_hard = -z_bottom_easy * 2.0 + torch.randn_like(z_bottom_easy)
            z_top_conflict = -z_top_easy * 2.0
            hard_res = field.relax(
                z_bottom_hard,
                top_prior=z_top_conflict,
            )

        assert easy_res.steps_taken.float().mean() <= 2.0
        assert hard_res.steps_taken.float().mean() > easy_res.steps_taken.float().mean() + 3.0

    def test_contradictory_hypothesis_produces_higher_equilibrium_energy_than_consistent(self):
        torch.manual_seed(19)
        cfg = PULSEConfig.micro()
        cfg.max_think_steps = 20
        field = BidirectionalPredictiveField(cfg)

        with torch.no_grad():
            z_top_true = torch.randn(8, cfg.top_dim) * 0.8
            z_bottom_true = field.generate_top_down(z_top_true)[0]

            # Consistent pair: generated by the world model itself
            consistent = field.evaluate_consistency(z_bottom_true, z_top_true)
            # Contradictory pair: opposite top hypothesis clamped against same bottom sensory state
            contradictory = field.evaluate_consistency(z_bottom_true, -z_top_true * 2.0)

        assert contradictory.final_energy.mean() > consistent.final_energy.mean() * 2.0

    def test_attractor_revision_moves_corrupted_hypothesis_toward_consistent_truth(self):
        torch.manual_seed(23)
        cfg = PULSEConfig.micro()
        cfg.max_think_steps = 30
        cfg.energy_threshold = 1e-5
        field = BidirectionalPredictiveField(cfg)

        with torch.no_grad():
            z_top_true = torch.randn(4, cfg.top_dim) * 0.6
            states_true = field.generate_top_down(z_top_true)
            z_bottom = states_true[0]

            # Start from a corrupted/hallucinated latent trajectory
            corrupted_states = [z_bottom.clone()] + [
                s + torch.randn_like(s) * 1.2 for s in states_true[1:]
            ]
            # Relax without clamping top -> energy minimization must revise internal hypothesis
            revised = field.relax(z_bottom, init_states=corrupted_states)

            # Bottom prediction from revised latent state must match sensory input far better
            pred_before = field.layers[0].predict_down(corrupted_states[1])
            pred_after = field.layers[0].predict_down(revised.states[1])
            err_before = (z_bottom - pred_before).pow(2).mean()
            err_after = (z_bottom - pred_after).pow(2).mean()

        assert err_after < err_before * 0.35
