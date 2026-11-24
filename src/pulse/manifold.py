"""Pillar 1: Continuous Concept Manifold (Latent World-Space) and decoupled I/O."""

from __future__ import annotations

import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptManifoldProjection(nn.Module):
    """Projects continuous representations onto a bounded hyperspherical concept manifold.

    Architectural innovation over naive latent predictive coding:
    - Smooth hyperspherical projection (radius sqrt(dim) by default, i.e., unit per-element RMS)
      prevents norm runaway or collapse to zero.
    - Provides VICReg-style variance/covariance anti-collapse energy so latent world-space
      remains full-rank and informative without requiring surface pixel/token reconstruction.
    """

    def __init__(self, dim: int, radius: float | None = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.radius = float(radius if radius is not None else math.sqrt(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).clamp_min(self.eps)
        return (x / norm) * self.radius

    def anti_collapse_energy(self, z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
        """VICReg-inspired variance + off-diagonal covariance penalty preventing collapse."""
        flat = z.reshape(-1, z.shape[-1])
        batch = flat.shape[0]
        if batch < 2:
            return flat.new_tensor(0.0)

        std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(target_std - std).mean()

        centered = flat - flat.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(batch - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / self.dim
        return var_loss + cov_loss


class TextConceptEncoder(nn.Module):
    """Lightweight language-to-concept I/O encoder.

    Compresses raw discrete token spans of length `chunk_size` into unified continuous
    concept frames z_t^(0), decoupling internal reasoning steps from token count.
    """

    def __init__(
        self,
        vocab_size: int,
        concept_dim: int,
        chunk_size: int = 4,
        embed_dim: int | None = None,
        radius: float | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.concept_dim = concept_dim
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim or max(16, concept_dim // 2)

        self.token_embed = nn.Embedding(vocab_size, self.embed_dim)
        self.chunk_pos = nn.Parameter(torch.randn(1, 1, chunk_size, self.embed_dim) * 0.02)
        self.chunk_proj = nn.Sequential(
            nn.Linear(chunk_size * self.embed_dim, concept_dim),
            nn.Tanh(),
            nn.Linear(concept_dim, concept_dim),
        )
        self.feature_norm = nn.LayerNorm(concept_dim, elementwise_affine=False)
        self.pool_gate = nn.Linear(concept_dim, 1)
        self.manifold = ConceptManifoldProjection(concept_dim, radius=radius)

    def encode_chunks(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode [B, T] token IDs into [B, ceil(T / chunk_size), concept_dim] concept frames."""
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        bsz, seq_len = tokens.shape
        rem = seq_len % self.chunk_size
        if rem != 0:
            pad_len = self.chunk_size - rem
            tokens = F.pad(tokens, (0, pad_len), value=0)
            seq_len = tokens.shape[1]

        num_chunks = seq_len // self.chunk_size
        token_chunks = tokens.view(bsz, num_chunks, self.chunk_size)
        emb = self.token_embed(token_chunks) + self.chunk_pos
        flat_chunks = emb.reshape(bsz, num_chunks, self.chunk_size * self.embed_dim)
        concepts = self.chunk_proj(flat_chunks)

        # Length-extrapolatable orthogonal phase rotation binding each chunk to its position
        pos = torch.arange(num_chunks, device=tokens.device, dtype=concepts.dtype).unsqueeze(1)
        half = self.concept_dim // 2
        if half > 0:
            freqs = torch.exp(
                torch.arange(half, device=tokens.device, dtype=concepts.dtype)
                * (-math.log(10000.0) / max(half, 1))
            )
            angles = (pos + 1.0) * freqs.unsqueeze(0)  # [N_chunks, half]
            c_even = concepts[:, :, 0 : 2 * half : 2]
            c_odd = concepts[:, :, 1 : 2 * half : 2]
            cos_a = torch.cos(angles).unsqueeze(0)
            sin_a = torch.sin(angles).unsqueeze(0)
            rotated = concepts.clone()
            rotated[:, :, 0 : 2 * half : 2] = c_even * cos_a - c_odd * sin_a
            rotated[:, :, 1 : 2 * half : 2] = c_even * sin_a + c_odd * cos_a
            concepts = rotated

        return self.manifold(self.feature_norm(concepts))

    def encode_utterance(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode [B, T] token IDs into a high-contrast concept vector [B, concept_dim]."""
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        bsz, seq_len = tokens.shape
        rem = seq_len % self.chunk_size
        padded_tokens = (
            F.pad(tokens, (0, self.chunk_size - rem), value=0) if rem != 0 else tokens
        )
        num_chunks = padded_tokens.shape[1] // self.chunk_size
        # Mask out pure-padding chunks so shared padding never collapses distinct short utterances
        chunk_active = (padded_tokens.view(bsz, num_chunks, self.chunk_size) != 0).any(dim=-1)
        all_pad = ~chunk_active.any(dim=-1, keepdim=True)
        chunk_active = torch.where(all_pad, torch.ones_like(chunk_active), chunk_active)

        frames = self.encode_chunks(padded_tokens)
        gate_logits = self.pool_gate(frames).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~chunk_active, -1e9)
        weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
        mean_pooled = (frames * weights).sum(dim=1)

        # Signed max-salience pooling amplifies subtle character/token differences
        masked_frames = frames.masked_fill(~chunk_active.unsqueeze(-1), 0.0)
        max_idx = masked_frames.abs().argmax(dim=1, keepdim=True)
        salient_pooled = masked_frames.gather(1, max_idx).squeeze(1)

        combined = 0.5 * mean_pooled + 0.5 * salient_pooled
        return self.manifold(self.feature_norm(combined))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encode_utterance(tokens)


class VectorConceptEncoder(nn.Module):
    """Lightweight continuous/multimodal sensory adapter mapping raw vectors to concept space."""

    def __init__(self, input_dim: int, concept_dim: int, radius: float | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        hidden = max(concept_dim, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, concept_dim),
        )
        self.manifold = ConceptManifoldProjection(concept_dim, radius=radius)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.manifold(self.net(x))


class EMATargetEncoder(nn.Module):
    """JEPA-style Exponential Moving Average target encoder for non-collapsing causal targets."""

    def __init__(self, online_encoder: nn.Module, decay: float = 0.99) -> None:
        super().__init__()
        self.decay = decay
        self.encoder = copy.deepcopy(online_encoder)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        for p_ema, p_online in zip(self.encoder.parameters(), online_encoder.parameters()):
            p_ema.data.lerp_(p_online.data, 1.0 - self.decay)

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.encoder(*args, **kwargs)


class GatedConceptDecoder(nn.Module):
    """Lightweight energy-gated renderer translating converged latent states into language/actions.

    Rendering is explicitly decoupled from internal thinking: `render_confidence` and
    `should_render` reflect whether the Predictive Coding Field has converged below
    `energy_gate_threshold`.
    """

    def __init__(
        self,
        concept_dim: int,
        output_dim: int,
        max_tokens: int = 16,
        energy_gate_threshold: float = 0.5,
        gate_temperature: float = 0.15,
    ) -> None:
        super().__init__()
        self.concept_dim = concept_dim
        self.output_dim = output_dim
        self.max_tokens = max_tokens
        self.energy_gate_threshold = energy_gate_threshold
        self.gate_temperature = gate_temperature

        self.action_head = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.SiLU(),
            nn.Linear(concept_dim, output_dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, concept_dim) * 0.05)
        self.token_head = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.SiLU(),
            nn.Linear(concept_dim, output_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        energy: torch.Tensor | None = None,
        decode_steps: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.action_head(z)

        steps = min(decode_steps if decode_steps is not None else self.max_tokens, self.max_tokens)
        z_seq = z.unsqueeze(1) + self.pos_embed[:, :steps, :]
        token_logits = self.token_head(z_seq)

        if energy is None:
            energy = z.new_zeros(z.shape[0])
        render_confidence = torch.sigmoid(
            (self.energy_gate_threshold - energy) / self.gate_temperature
        )
        should_render = energy <= self.energy_gate_threshold

        return {
            "logits": logits,
            "token_logits": token_logits,
            "render_confidence": render_confidence,
            "should_render": should_render,
        }
