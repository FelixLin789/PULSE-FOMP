"""Tests for PULSE-v2 building blocks: Chunkwise WY Gated DeltaNet, O(1) Echoic Window, and Contrastive Energy Critic."""

import torch

from pulse.delta_resonance import (
    ChunkwiseGatedDeltaNet,
    ContrastiveEnergyCritic,
    ShortCausalConv,
    SlidingWindowEchoicAttention,
)


class TestShortCausalConv:
    def test_streaming_step_matches_parallel_forward_and_is_strictly_causal(self):
        torch.manual_seed(0)
        conv = ShortCausalConv(dim=16, kernel_size=4)
        x = torch.randn(2, 11, 16)

        y_par, state_par = conv(x)
        assert y_par.shape == x.shape
        assert state_par.shape == (2, 16, 3)

        # Verify strict causality: perturbing t >= 6 must not alter outputs for t < 6
        x_alt = x.clone()
        x_alt[:, 6:, :] += 5.0
        y_alt, _ = conv(x_alt)
        assert torch.allclose(y_par[:, :6, :], y_alt[:, :6, :], atol=1e-6)

        # Verify step-by-step streaming matches parallel forward
        st = conv.init_state(batch_size=2)
        outs = []
        for t in range(x.shape[1]):
            yt, st = conv.step(x[:, t : t + 1, :], st)
            outs.append(yt)
        y_seq = torch.cat(outs, dim=1)
        assert torch.allclose(y_par, y_seq, atol=1e-5)
        assert torch.allclose(state_par, st, atol=1e-5)


class TestChunkwiseGatedDeltaNet:
    def test_wy_chunkwise_parallel_forward_matches_exact_recurrent_delta_rule(self):
        torch.manual_seed(7)
        layer = ChunkwiseGatedDeltaNet(d_model=32, n_heads=4, chunk_size=8)
        x = torch.randn(2, 20, 32)  # Non-multiple of chunk_size to test padding cleanly

        y_par, state_par = layer(x)
        assert y_par.shape == x.shape
        assert state_par["S"].shape == (2, 4, 8, 8)  # [B, H, d_v, d_k] strictly O(1)!

        # Exact sequential token-by-token streaming reference
        st = layer.init_state(batch_size=2)
        outs = []
        for t in range(x.shape[1]):
            yt, st = layer.step(x[:, t : t + 1, :], st)
            outs.append(yt)
        y_seq = torch.cat(outs, dim=1)

        assert torch.allclose(y_par, y_seq, atol=1e-4)
        assert torch.allclose(state_par["S"], st["S"], atol=1e-4)

    def test_is_strictly_causal_and_maintains_o1_state_across_lengths(self):
        torch.manual_seed(11)
        layer = ChunkwiseGatedDeltaNet(d_model=32, n_heads=4, chunk_size=8)
        x_a = torch.randn(2, 24, 32)
        x_b = x_a.clone()
        x_b[:, 13:, :] += 3.0

        y_a, st_short = layer(x_a)
        y_b, _ = layer(x_b)
        assert torch.allclose(y_a[:, :13, :], y_b[:, :13, :], atol=1e-5)

        _, st_long = layer(torch.randn(2, 128, 32))
        assert st_short["S"].shape == st_long["S"].shape
        assert layer.state_bytes(st_short) == layer.state_bytes(st_long)


class TestSlidingWindowEchoicAttention:
    def test_ring_buffer_is_strictly_o1_and_causally_matches_streaming(self):
        torch.manual_seed(19)
        attn = SlidingWindowEchoicAttention(d_model=32, n_heads=4, window_size=8)
        x = torch.randn(2, 20, 32)

        y_par, st_20 = attn(x)
        _, st_64 = attn(torch.randn(2, 64, 32))

        # Ring KV buffer is capped at window_size=8 regardless of sequence length!
        assert st_20["k"].shape[2] == 8
        assert st_64["k"].shape[2] == 8
        assert attn.state_bytes(st_20) == attn.state_bytes(st_64)

        # Strict causality check
        x_alt = x.clone()
        x_alt[:, 10:, :] += 4.0
        y_alt, _ = attn(x_alt)
        assert torch.allclose(y_par[:, :10, :], y_alt[:, :10, :], atol=1e-5)

        # Streaming step-by-step matches parallel sliding-window forward
        st = attn.init_state(batch_size=2)
        outs = []
        for t in range(x.shape[1]):
            yt, st = attn.step(x[:, t : t + 1, :], st)
            outs.append(yt)
        y_seq = torch.cat(outs, dim=1)
        assert torch.allclose(y_par, y_seq, atol=1e-4)


class TestContrastiveEnergyCritic:
    def test_zero_backprop_wake_sleep_learning_achieves_high_auroc(self):
        torch.manual_seed(29)
        critic = ContrastiveEnergyCritic(top_dim=32, concept_dim=32, proj_dim=24)

        # Paired positive (context -> true continuation) vs mismatched negatives
        u_ctx = torch.randn(16, 32)
        w_map = torch.randn(32, 32) * 0.6
        z_true = torch.tanh(u_ctx @ w_map.T)
        z_false = torch.roll(z_true, shifts=1, dims=0)

        for _ in range(60):
            critic.update_wake_sleep_local(u_ctx, z_true, lr=0.05)

        # Strictly ZERO autograd gradients created
        for name, p in critic.named_parameters():
            assert p.grad is None, f"Critic parameter {name} unexpectedly populated .grad!"

        with torch.no_grad():
            e_true = critic.energy(u_ctx, z_true)
            e_false = critic.energy(u_ctx, z_false)
            auroc = (e_false.unsqueeze(1) > e_true.unsqueeze(0)).float().mean().item()

        assert auroc >= 0.95
        assert e_false.mean().item() > e_true.mean().item() * 2.0
