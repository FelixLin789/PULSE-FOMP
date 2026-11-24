"""Configuration schema and presets for the PULSE architecture."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class PULSEConfig:
    """Hyperparameters governing the four coupled PULSE mechanisms.

    1. Continuous Concept Manifold (Latent World-Space)
    2. Bidirectional Predictive Coding Field
    3. Tri-Timescale Plastic Memory (Fast SSM, Fast Weights, Slow Consolidation)
    4. Local Energy Learning (Zero Global Backprop)
    """

    # Pillar 1: Continuous Concept Manifold & I/O
    vocab_size: int = 256
    layer_dims: tuple[int, ...] = (64, 96, 128, 96)
    output_dim: int = 256
    chunk_size: int = 4
    max_decode_tokens: int = 16
    manifold_radius: float | None = None

    # Pillar 2: Bidirectional Predictive Coding Field
    max_think_steps: int = 24
    energy_threshold: float = 1e-2
    infer_lr: float = 0.15
    infer_momentum: float = 0.5
    infer_grad_clip: float = 5.0
    precision_min: float = 0.1
    prior_precision: float = 0.25
    energy_gate_threshold: float = 0.5

    # Pillar 3: Tri-Timescale Plastic Memory
    ssm_state_dim: int = 128
    fast_weight_alpha: float = 0.35
    surprise_threshold: float = 0.5
    surprise_temperature: float = 0.25
    fast_weight_decay: float = 0.02
    fast_weight_rank: int | None = None
    fast_weight_max_norm: float = 4.0
    replay_buffer_capacity: int = 128
    consolidation_ewc_lambda: float = 1.0
    consolidation_lr: float = 0.05

    # Pillar 4: Local Energy Learning
    weight_lr: float = 1e-2
    weight_momentum: float = 0.8
    weight_decay: float = 1e-4
    io_lr: float = 5e-3
    ema_decay: float = 0.99
    anti_collapse_weight: float = 0.05

    def __post_init__(self) -> None:
        if len(self.layer_dims) < 2:
            raise ValueError("layer_dims must contain at least 2 dimensions (sensory and latent).")
        if any(d <= 0 for d in self.layer_dims):
            raise ValueError("All layer_dims entries must be positive integers.")
        if self.energy_threshold <= 0:
            raise ValueError("energy_threshold must be strictly positive.")
        if self.max_think_steps < 1:
            raise ValueError("max_think_steps must be at least 1.")
        if not (0.0 <= self.fast_weight_decay < 1.0):
            raise ValueError("fast_weight_decay must be in [0.0, 1.0).")
        if self.ssm_state_dim <= 0:
            raise ValueError("ssm_state_dim must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.manifold_radius is None:
            self.manifold_radius = math.sqrt(self.layer_dims[0])

    @property
    def num_layers(self) -> int:
        """Number of predictive coding transformations between adjacent latent levels."""
        return len(self.layer_dims) - 1

    @property
    def concept_dim(self) -> int:
        """Bottom-level continuous concept manifold dimension (z^(0))."""
        return self.layer_dims[0]

    @property
    def top_dim(self) -> int:
        """Top-level abstract world-model latent dimension (z^(L))."""
        return self.layer_dims[-1]

    @classmethod
    def micro(cls) -> PULSEConfig:
        """Tiny preset ideal for fast unit tests and toy verification."""
        return cls(
            vocab_size=128,
            layer_dims=(32, 48, 48),
            output_dim=128,
            chunk_size=4,
            max_decode_tokens=8,
            ssm_state_dim=48,
            max_think_steps=16,
            energy_threshold=1e-2,
        )

    @classmethod
    def small(cls) -> PULSEConfig:
        """Compact preset for local CPU/GPU training, benchmarks, and interactive CLI."""
        return cls(
            vocab_size=256,
            layer_dims=(64, 96, 128, 96),
            output_dim=256,
            chunk_size=4,
            max_decode_tokens=16,
            ssm_state_dim=128,
            max_think_steps=24,
            energy_threshold=1e-2,
        )

    @classmethod
    def base(cls) -> PULSEConfig:
        """Research-grade preset for 8GB GPU workloads."""
        return cls(
            vocab_size=512,
            layer_dims=(128, 192, 256, 256, 192),
            output_dim=512,
            chunk_size=4,
            max_decode_tokens=32,
            ssm_state_dim=256,
            max_think_steps=32,
            energy_threshold=5e-3,
        )
