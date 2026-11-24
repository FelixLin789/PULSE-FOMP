"""Tests for Pillar 4: Local Energy Learning (100% Backprop-Free Synaptic Plasticity)."""

import torch

from pulse.config import PULSEConfig
from pulse.local_learning import LocalIOUpdater, LocalSynapticOptimizer
from pulse.manifold import GatedConceptDecoder
from pulse.predictive_field import BidirectionalPredictiveField


class TestLocalSynapticOptimizer:
    def test_trains_nonlinear_mapping_strictly_without_autograd_or_backward(self):
        torch.manual_seed(0)
        cfg = PULSEConfig.micro()
        cfg.layer_dims = (16, 24, 12)
        cfg.max_think_steps = 20
        cfg.weight_lr = 0.03
        field = BidirectionalPredictiveField(cfg)
        opt = LocalSynapticOptimizer(field, cfg)

        # Synthetic nonlinear teacher mapping z_top -> z_bottom
        w_true_1 = torch.randn(20, 12) * 0.5
        w_true_2 = torch.randn(16, 20) * 0.5
        z_top = torch.randn(32, 12)
        z_bottom = torch.tanh(z_top @ w_true_1.T) @ w_true_2.T

        with torch.no_grad():
            pred_before = field.generate_top_down(z_top, use_fast_weights=False)[0]
            mse_before = (z_bottom - pred_before).pow(2).mean().item()

        # Train purely with local energy learning (asserting zero .grad created)
        energies = []
        for _ in range(60):
            stats = opt.train_supervised_pair(field, z_bottom=z_bottom, z_top_target=z_top)
            energies.append(stats["equilibrium_energy"])

        # Verify ZERO autograd gradients were ever attached to any core parameter
        for name, param in field.named_parameters():
            assert param.grad is None, f"Parameter {name} unexpectedly has .grad populated!"

        with torch.no_grad():
            pred_after = field.generate_top_down(z_top, use_fast_weights=False)[0]
            mse_after = (z_bottom - pred_after).pow(2).mean().item()

        assert energies[-1] < energies[0] * 0.35
        assert mse_after < mse_before * 0.25

    def test_asynchronous_shuffled_layer_updates_converge_without_global_sync(self):
        torch.manual_seed(7)
        cfg = PULSEConfig.micro()
        cfg.layer_dims = (12, 18, 18, 10)  # 3 predictive layers
        cfg.max_think_steps = 20
        cfg.weight_lr = 0.03
        field = BidirectionalPredictiveField(cfg)
        opt = LocalSynapticOptimizer(field, cfg)

        z_top = torch.randn(24, 10)
        z_bottom = torch.tanh(z_top @ (torch.randn(12, 10) * 0.6).T)

        with torch.no_grad():
            mse_before = (
                z_bottom - field.generate_top_down(z_top, use_fast_weights=False)[0]
            ).pow(2).mean().item()

        for _ in range(60):
            opt.train_supervised_pair(
                field,
                z_bottom=z_bottom,
                z_top_target=z_top,
                async_order=True,
            )

        with torch.no_grad():
            mse_after = (
                z_bottom - field.generate_top_down(z_top, use_fast_weights=False)[0]
            ).pow(2).mean().item()

        assert mse_after < mse_before * 0.3

    def test_amortized_bottom_up_improves_and_reduces_adaptive_thinking_steps(self):
        torch.manual_seed(13)
        cfg = PULSEConfig.micro()
        cfg.layer_dims = (16, 24, 16)
        cfg.max_think_steps = 25
        cfg.energy_threshold = 0.08
        cfg.weight_lr = 0.04
        field = BidirectionalPredictiveField(cfg)
        opt = LocalSynapticOptimizer(field, cfg)

        # Fixed set of recurring familiar patterns
        z_patterns = torch.randn(16, 16) * 0.8

        steps_before = field.relax(z_patterns).steps_taken.float().mean().item()
        for _ in range(50):
            opt.train_unsupervised_sensory(field, z_bottom=z_patterns)
        steps_after = field.relax(z_patterns).steps_taken.float().mean().item()

        assert steps_after < steps_before


class TestLocalIOUpdater:
    def test_updates_decoder_action_and_token_heads_purely_without_autograd(self):
        torch.manual_seed(19)
        dec = GatedConceptDecoder(concept_dim=24, output_dim=16, max_tokens=4)
        updater = LocalIOUpdater(lr=0.05)

        z = torch.randn(32, 24)
        # Distinct linearly-separable target classes and token patterns
        targets = torch.arange(32) % 16
        token_targets = torch.stack([(targets + k) % 16 for k in range(4)], dim=1)

        with torch.no_grad():
            out_before = dec(z, decode_steps=4)
            acc_action_before = (out_before["logits"].argmax(dim=-1) == targets).float().mean().item()
            acc_tok_before = (
                (out_before["token_logits"].argmax(dim=-1) == token_targets).float().mean().item()
            )

        for _ in range(80):
            updater.step_decoder_action_local(dec, z, targets)
            updater.step_decoder_tokens_local(dec, z, token_targets)

        for name, p in dec.named_parameters():
            assert p.grad is None, f"Decoder param {name} unexpectedly populated .grad!"

        with torch.no_grad():
            out_after = dec(z, decode_steps=4)
            acc_action_after = (out_after["logits"].argmax(dim=-1) == targets).float().mean().item()
            acc_tok_after = (
                (out_after["token_logits"].argmax(dim=-1) == token_targets).float().mean().item()
            )

        assert acc_action_after > max(acc_action_before + 0.4, 0.85)
        assert acc_tok_after > max(acc_tok_before + 0.4, 0.85)
