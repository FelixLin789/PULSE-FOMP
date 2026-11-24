"""Tests for PULSEConfig and Pillar 1: Continuous Concept Manifold (Latent World-Space)."""

import math
import pytest
import torch

from pulse.config import PULSEConfig
from pulse.manifold import (
    ConceptManifoldProjection,
    EMATargetEncoder,
    GatedConceptDecoder,
    TextConceptEncoder,
    VectorConceptEncoder,
)


class TestPULSEConfig:
    def test_presets_produce_valid_hierarchical_configs(self):
        for preset in (PULSEConfig.micro(), PULSEConfig.small(), PULSEConfig.base()):
            assert len(preset.layer_dims) >= 3
            assert preset.num_layers == len(preset.layer_dims) - 1
            assert preset.concept_dim == preset.layer_dims[0]
            assert preset.top_dim == preset.layer_dims[-1]
            assert preset.energy_threshold > 0
            assert preset.max_think_steps >= 2

    def test_invalid_layer_dims_raises_value_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            PULSEConfig(layer_dims=(64,))

    def test_invalid_hyperparameters_raise_value_error(self):
        with pytest.raises(ValueError, match="energy_threshold"):
            PULSEConfig(energy_threshold=-0.1)
        with pytest.raises(ValueError, match="fast_weight_decay"):
            PULSEConfig(fast_weight_decay=1.5)


class TestConceptManifoldProjection:
    def test_projects_arbitrary_scale_inputs_onto_bounded_manifold(self):
        proj = ConceptManifoldProjection(dim=32)
        huge = torch.randn(8, 32) * 1000.0
        tiny = torch.randn(8, 32) * 1e-4

        z_huge = proj(huge)
        z_tiny = proj(tiny)

        expected_radius = math.sqrt(32)
        assert z_huge.shape == (8, 32)
        assert torch.allclose(z_huge.norm(dim=-1), torch.full((8,), expected_radius), atol=1e-2)
        assert torch.allclose(z_tiny.norm(dim=-1), torch.full((8,), expected_radius), atol=1e-1)
        assert torch.isfinite(z_huge).all()
        assert torch.isfinite(z_tiny).all()

    def test_anti_collapse_energy_penalizes_collapsed_representations_more_than_diverse(self):
        proj = ConceptManifoldProjection(dim=16)
        torch.manual_seed(0)
        diverse = proj(torch.randn(64, 16))
        collapsed = proj(torch.ones(64, 16) + 1e-4 * torch.randn(64, 16))

        e_diverse = proj.anti_collapse_energy(diverse)
        e_collapsed = proj.anti_collapse_energy(collapsed)

        assert e_collapsed.item() > e_diverse.item() * 2.0


class TestMultimodalEncodersAndEMA:
    def test_text_concept_encoder_compresses_tokens_into_concept_frames(self):
        enc = TextConceptEncoder(vocab_size=128, concept_dim=32, chunk_size=4)
        tokens = torch.randint(0, 128, (3, 12))

        frames = enc.encode_chunks(tokens)
        utterance = enc.encode_utterance(tokens)

        assert frames.shape == (3, 3, 32)  # 12 tokens / chunk_size 4 = 3 concept frames
        assert utterance.shape == (3, 32)
        assert torch.isfinite(frames).all()

    def test_vector_and_text_encoders_share_unified_manifold_geometry(self):
        text_enc = TextConceptEncoder(vocab_size=64, concept_dim=24, chunk_size=4)
        vec_enc = VectorConceptEncoder(input_dim=10, concept_dim=24)

        z_text = text_enc.encode_utterance(torch.randint(0, 64, (4, 8)))
        z_vec = vec_enc(torch.randn(4, 10))

        assert z_text.shape == z_vec.shape == (4, 24)
        # Both live on the same radius manifold
        assert torch.allclose(z_text.norm(dim=-1).mean(), z_vec.norm(dim=-1).mean(), rtol=0.15)

    def test_ema_target_encoder_tracks_online_encoder_without_gradients(self):
        vec_enc = VectorConceptEncoder(input_dim=8, concept_dim=16)
        ema = EMATargetEncoder(vec_enc, decay=0.9)

        x = torch.randn(4, 8)
        y_before = ema(x)
        assert not y_before.requires_grad

        # Mutate online encoder weights
        with torch.no_grad():
            for p in vec_enc.parameters():
                p.add_(1.0)

        ema.update(vec_enc)
        y_after = ema(x)
        y_online = vec_enc(x).detach()

        # EMA moved toward online, but did not jump instantaneously
        dist_before_online = (y_before - y_online).norm()
        dist_after_online = (y_after - y_online).norm()
        assert dist_after_online < dist_before_online
        assert dist_after_online > 1e-4


class TestGatedConceptDecoder:
    def test_decodes_converged_latent_state_and_gates_by_free_energy(self):
        dec = GatedConceptDecoder(concept_dim=32, output_dim=64, max_tokens=6, energy_gate_threshold=0.2)
        z = torch.randn(4, 32)
        low_energy = torch.tensor([0.01, 0.05, 0.02, 0.03])
        high_energy = torch.tensor([1.5, 2.0, 1.2, 3.0])

        out_low = dec(z, energy=low_energy, decode_steps=6)
        out_high = dec(z, energy=high_energy, decode_steps=6)

        assert out_low["logits"].shape == (4, 64)
        assert out_low["token_logits"].shape == (4, 6, 64)
        assert (out_low["render_confidence"] > 0.7).all()
        assert (out_high["render_confidence"] < 0.3).all()
        assert out_low["should_render"].all()
        assert (~out_high["should_render"]).all()
