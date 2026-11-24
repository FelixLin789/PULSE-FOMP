"""Pillar 3: Tri-Timescale Plastic Memory (Fast SSM, Surprise-Gated Fast Weights, Sleep Consolidation)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.config import PULSEConfig
from pulse.predictive_field import BidirectionalPredictiveField, _activation_and_derivative


class FastWorkingStateSSM(nn.Module):
    """Timescale 1 (Millisecond / O(1) Memory): Selective Gated Continuous-Time State Space Cell.

    Replaces unbounded KV Cache with a fixed-size continuous working state h_t in R^(B x d_state).
    Because decay A_bar = exp(dt * (-exp(a_log))) lies strictly in (0, 1) and the nonlinear drive
    is bounded in [-1, 1], ||h_t||_inf remains strictly bounded over arbitrarily long horizons
    (millions of steps) with O(1) memory and O(N) compute.
    """

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        # Multi-timescale log-decay rates spanning fast to slow working memory
        init_rates = torch.linspace(-1.5, 0.5, state_dim)
        self.a_log = nn.Parameter(init_rates)
        self.dt_proj = nn.Linear(input_dim, state_dim)
        self.b_proj = nn.Linear(input_dim, state_dim)
        self.gate_proj = nn.Linear(input_dim + state_dim, state_dim)
        self.readout_norm = nn.LayerNorm(state_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(state_dim, input_dim)

        nn.init.normal_(self.out_proj.weight, std=1.0 / math.sqrt(state_dim))
        nn.init.zeros_(self.out_proj.bias)

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        target_device = device if device is not None else self.a_log.device
        target_dtype = dtype if dtype is not None else self.a_log.dtype
        return torch.zeros(batch_size, self.state_dim, device=target_device, dtype=target_dtype)

    def _readout_features(self, state: torch.Tensor) -> torch.Tensor:
        return self.readout_norm(state)

    def step(self, z_top: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance O(1) working memory by one concept frame and emit next-step causal prior."""
        dt = F.softplus(self.dt_proj(z_top))
        a_neg = -torch.exp(self.a_log).unsqueeze(0)
        a_bar = torch.exp(dt * a_neg)  # Strictly in (0, 1)

        drive = torch.tanh(self.b_proj(z_top))
        gate = torch.sigmoid(self.gate_proj(torch.cat([z_top, state], dim=-1)))
        new_state = a_bar * state + (1.0 - a_bar) * (gate * drive)
        prior_pred = self.out_proj(self._readout_features(new_state))
        return new_state, prior_pred

    def predict_prior(self, state: torch.Tensor) -> torch.Tensor:
        """Project current working memory state into a top-level prior z_prior^(L)."""
        return self.out_proj(self._readout_features(state))

    @torch.no_grad()
    def update_prior_local(
        self,
        state: torch.Tensor,
        z_next_target: torch.Tensor,
        lr: float = 0.05,
    ) -> float:
        """Backprop-free local normalized delta-rule update of temporal causal predictor."""
        feat = self._readout_features(state)
        pred = self.out_proj(feat)
        err = z_next_target - pred
        bsz = max(feat.shape[0], 1)
        feat_norm_sq = feat.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
        self.out_proj.weight.add_(lr * ((err / feat_norm_sq).T @ feat) / bsz)
        self.out_proj.bias.add_(lr * err.mean(dim=0))
        return err.pow(2).mean().item()


@dataclass
class EpisodicTrace:
    """High-salience inference episode buffered in O(1) capacity for Sleep Consolidation."""

    z_bottom: torch.Tensor
    z_top: torch.Tensor
    energy_drop: float
    hit_count: int = 1

    @property
    def salience(self) -> float:
        return self.energy_drop * (1.0 + math.log1p(self.hit_count))


class SurpriseGatedFastWeights:
    """Timescale 2 (Mid-Term Online Plasticity): Inference-time surprise-gated fast weights.

    Implements the blueprint plasticity law with normalized Delta-Rule orthogonalization:
        dW_fast = alpha * sigma((||eps_t|| - gamma) / T) * (u_t v_t^T) - lambda * W_fast
    where u_t = eps_t^(l) and v_t = phi(z_t^(l+1)) / (||phi(z_t^(l+1))||^2 + eps).

    Using prediction error eps_t as u_t turns Hebbian correlation into the self-stabilizing
    Widrow-Hoff Delta Rule: repeated exposure drives eps_t -> 0 without runaway norms and cleanly
    overwrites outdated rules without interference.
    """

    def __init__(self, config: PULSEConfig, max_prior_centroids: int = 32) -> None:
        self.config = config
        self.max_prior_centroids = max_prior_centroids
        self.episodes: List[EpisodicTrace] = []
        self._latent_mean: torch.Tensor | None = None
        self._latent_var: torch.Tensor | None = None
        self._latent_count: int = 0
        # Fixed O(1) online Gaussian-Mixture prior sketch for high-fidelity generative replay
        self._prior_centroids: List[torch.Tensor] = []

    def surprise_gate(self, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sample RMS surprise magnitude and smooth surprise gate sigma(||eps|| - gamma)."""
        rms_surprise = torch.sqrt(eps.pow(2).mean(dim=-1) + 1e-12)
        temp = max(self.config.surprise_temperature, 1e-4)
        gate = torch.sigmoid((rms_surprise - self.config.surprise_threshold) / temp)
        return rms_surprise, gate

    @torch.no_grad()
    def load_latent_state(
        self,
        latent_mean: torch.Tensor | None,
        latent_var: torch.Tensor | None,
        latent_count: int = 0,
        prior_centroids: List[torch.Tensor] | None = None,
    ) -> None:
        """Restore wake latent statistics onto CPU regardless of checkpoint map_location."""
        self._latent_mean = (
            latent_mean.detach().float().cpu() if latent_mean is not None else None
        )
        self._latent_var = (
            latent_var.detach().float().cpu() if latent_var is not None else None
        )
        self._latent_count = int(latent_count or 0)
        self._prior_centroids = [
            c.detach().float().cpu() for c in (prior_centroids or [])
        ]

    @torch.no_grad()
    def observe_latent_distribution(self, z_top: torch.Tensor, max_centroid_samples: int = 16) -> None:
        """Track running mean, variance, and O(1) GMM centroids of wake top-level concepts."""
        z_cpu = z_top.detach().float().cpu()
        batch_mean = z_cpu.mean(dim=0)
        batch_var = z_cpu.var(dim=0, unbiased=False).clamp_min(1e-3)
        if self._latent_mean is None or self._latent_var is None:
            self._latent_mean = batch_mean
            self._latent_var = batch_var
            self._latent_count = z_cpu.shape[0]
        else:
            # Guarantee CPU residence even if state was loaded via a GPU map_location
            if self._latent_mean.device != batch_mean.device:
                self._latent_mean = self._latent_mean.detach().float().cpu()
                self._latent_var = self._latent_var.detach().float().cpu()
            momentum = 0.1
            self._latent_mean = (1.0 - momentum) * self._latent_mean + momentum * batch_mean
            self._latent_var = (1.0 - momentum) * self._latent_var + momentum * batch_var
            self._latent_count += z_cpu.shape[0]

        if self._prior_centroids and self._prior_centroids[0].device != z_cpu.device:
            self._prior_centroids = [c.detach().float().cpu() for c in self._prior_centroids]

        # Subsample large GPU batches so online centroid tracking stays O(1) fast (~0.1ms/step)
        n_rows = z_cpu.shape[0]
        if n_rows > max_centroid_samples:
            stride = max(n_rows // max_centroid_samples, 1)
            z_sub = z_cpu[::stride][:max_centroid_samples]
        else:
            z_sub = z_cpu

        # Online competitive centroid update (bounded O(1) memory, zero raw data stored)
        for b in range(z_sub.shape[0]):
            vec = z_sub[b]
            if not self._prior_centroids:
                self._prior_centroids.append(vec.clone())
                continue
            stack = torch.stack(self._prior_centroids, dim=0)
            sims = F.cosine_similarity(stack, vec.unsqueeze(0), dim=-1)
            best_idx = int(sims.argmax().item())
            if float(sims[best_idx].item()) > 0.92 or len(self._prior_centroids) >= self.max_prior_centroids:
                self._prior_centroids[best_idx] = 0.8 * self._prior_centroids[best_idx] + 0.2 * vec
            else:
                self._prior_centroids.append(vec.clone())

    def sample_prior_latents(
        self,
        num_samples: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample synthetic top-level latents from the O(1) GMM wake prior for generative replay."""
        if self._prior_centroids:
            centers = torch.stack(self._prior_centroids, dim=0).to(device=device, dtype=dtype)
            if centers.shape[1] == dim:
                idx = torch.arange(num_samples, device=device) % centers.shape[0]
                chosen = centers.index_select(0, idx)
                noise_scale = 0.08 * torch.sqrt(
                    self._latent_var.to(device=device, dtype=dtype)
                    if self._latent_var is not None
                    else torch.ones(dim, device=device, dtype=dtype)
                )
                return chosen + noise_scale.unsqueeze(0) * torch.randn_like(chosen)
        if self._latent_mean is None or self._latent_var is None:
            return torch.randn(num_samples, dim, device=device, dtype=dtype) * 0.7
        mean = self._latent_mean.to(device=device, dtype=dtype)
        std = torch.sqrt(self._latent_var.to(device=device, dtype=dtype))
        return mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            num_samples, dim, device=device, dtype=dtype
        )

    @staticmethod
    def _project_low_rank(mat: torch.Tensor, rank: int) -> torch.Tensor:
        max_r = min(mat.shape[0], mat.shape[1])
        if rank >= max_r:
            return mat
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        return (u[:, :rank] * s[:rank].unsqueeze(0)) @ vh[:rank, :]

    @torch.no_grad()
    def decay_only(self, field: BidirectionalPredictiveField) -> None:
        """Apply natural exponential forgetting -lambda * W_fast across all layers."""
        keep = 1.0 - self.config.fast_weight_decay
        for layer in field.layers:
            layer.w_fast.mul_(keep)

    @torch.no_grad()
    def reset(self, field: BidirectionalPredictiveField, clear_episodes: bool = False) -> None:
        """Clear all fast weights (and optionally buffered episodes)."""
        for layer in field.layers:
            layer.w_fast.zero_()
        if clear_episodes:
            self.episodes.clear()

    @torch.no_grad()
    def update(
        self,
        field: BidirectionalPredictiveField,
        states: List[torch.Tensor],
        record_episode: bool = True,
    ) -> Dict[str, float]:
        """Perform one online inference-time fast-weight plasticity step across all layers."""
        energy_before, _ = field.compute_energy_and_errors(states, use_fast_weights=True)
        self.observe_latent_distribution(states[-1])

        gates_recorded: List[float] = []
        surprises_recorded: List[float] = []
        keep = 1.0 - self.config.fast_weight_decay
        alpha = self.config.fast_weight_alpha
        max_norm = self.config.fast_weight_max_norm
        rank = self.config.fast_weight_rank

        for idx, layer in enumerate(field.layers):
            z_lower = states[idx].detach()
            z_upper = states[idx + 1].detach()
            phi_u, _ = _activation_and_derivative(z_upper, layer.activation)
            eps = z_lower - layer.predict_down(z_upper, use_fast_weights=True)

            rms_surp, gate = self.surprise_gate(eps)
            gates_recorded.append(gate.mean().item())
            surprises_recorded.append(rms_surp.mean().item())

            # Normalized Delta-Rule outer product: sum_b g_b * eps_b * (phi_b / (||phi_b||^2 + eps))^T
            bsz = max(z_lower.shape[0], 1)
            phi_norm_sq = phi_u.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
            v_norm = phi_u / phi_norm_sq  # [B, upper_dim]
            u_gated = gate.unsqueeze(-1) * eps  # [B, lower_dim]
            delta_w = (alpha / bsz) * (u_gated.T @ v_norm)

            layer.w_fast.mul_(keep).add_(delta_w)

            if rank is not None:
                layer.w_fast.copy_(self._project_low_rank(layer.w_fast, rank))

            cur_norm = torch.linalg.matrix_norm(layer.w_fast, ord="fro")
            if cur_norm > max_norm:
                layer.w_fast.mul_(max_norm / cur_norm.clamp_min(1e-8))

        energy_after, _ = field.compute_energy_and_errors(states, use_fast_weights=True)
        energy_drop = (energy_before - energy_after).clamp_min(0.0)
        mean_gate = sum(gates_recorded) / max(len(gates_recorded), 1)
        max_gate = max(gates_recorded) if gates_recorded else 0.0
        mean_surprise = sum(surprises_recorded) / max(len(surprises_recorded), 1)

        if record_episode and max_gate > 0.5:
            self._record_salient_episodes(states[0], states[-1], energy_drop)

        return {
            "mean_gate": mean_gate,
            "max_gate": max_gate,
            "layer_gates": gates_recorded,
            "mean_surprise": mean_surprise,
            "mean_energy_drop": energy_drop.mean().item(),
            "num_episodes": float(len(self.episodes)),
        }

    def _record_salient_episodes(
        self,
        z_bottom: torch.Tensor,
        z_top: torch.Tensor,
        energy_drop: torch.Tensor,
    ) -> None:
        cap = self.config.replay_buffer_capacity
        for b in range(z_bottom.shape[0]):
            drop_val = float(energy_drop[b].item())
            if drop_val <= 1e-5:
                continue
            zb = z_bottom[b].detach().cpu().clone()
            zt = z_top[b].detach().cpu().clone()

            # Merge if a closely matching cue already exists (tracks activation frequency!)
            matched = False
            for ep in self.episodes:
                sim = F.cosine_similarity(ep.z_top.unsqueeze(0), zt.unsqueeze(0)).item()
                if sim > 0.94:
                    ep.hit_count += 1
                    ep.energy_drop = max(ep.energy_drop, drop_val)
                    ep.z_bottom = 0.5 * ep.z_bottom + 0.5 * zb
                    ep.z_top = 0.5 * ep.z_top + 0.5 * zt
                    matched = True
                    break
            if not matched:
                self.episodes.append(EpisodicTrace(z_bottom=zb, z_top=zt, energy_drop=drop_val))
                if len(self.episodes) > cap:
                    # Evict lowest-salience episode to keep strictly O(1) bounded memory
                    self.episodes.sort(key=lambda e: e.salience, reverse=True)
                    del self.episodes[cap:]


class SleepConsolidator:
    """Timescale 3 (Slow Core Consolidation): Offline Generative Replay Distillation.

    Filters high-salience (high-frequency + high-energy-drop) episodes from wakefulness,
    generates synthetic replay trajectories from W_slow to protect previously learned
    knowledge without storing raw datasets, and distills W_fast into W_slow via local
    importance-weighted updates.
    """

    def __init__(self, config: PULSEConfig) -> None:
        self.config = config

    @torch.no_grad()
    def consolidate(
        self,
        field: BidirectionalPredictiveField,
        fast_memory: SurpriseGatedFastWeights,
        top_k_episodes: int = 32,
        num_replay_samples: int = 32,
        num_epochs: int = 25,
        clear_fast_weights: bool = True,
    ) -> Dict[str, float]:
        """Distill high-salience W_fast trajectories into W_slow using generative replay."""
        if not fast_memory.episodes:
            return {
                "consolidated_episodes": 0.0,
                "new_loss_final": 0.0,
                "replay_loss_final": 0.0,
            }

        device = field.layers[0].w_slow.device
        dtype = field.layers[0].w_slow.dtype

        # 1. Select highest-salience episodes (frequent activation + large energy drop)
        selected = sorted(fast_memory.episodes, key=lambda e: e.salience, reverse=True)[
            :top_k_episodes
        ]
        z_bottom_new = torch.stack([e.z_bottom for e in selected], dim=0).to(
            device=device, dtype=dtype
        )
        z_top_new = torch.stack([e.z_top for e in selected], dim=0).to(
            device=device, dtype=dtype
        )

        # Build teacher trajectories reflecting current W_slow + W_fast knowledge
        teacher_new = field.generate_top_down(z_top_new, use_fast_weights=True)
        teacher_new[0] = z_bottom_new  # Ground bottom sensory target directly to episode truth

        # 2. Generative Replay of old slow-weight world model (no raw dataset required!)
        z_top_replay = fast_memory.sample_prior_latents(
            num_samples=num_replay_samples,
            dim=field.layer_dims[-1],
            device=device,
            dtype=dtype,
        )
        teacher_replay = field.generate_top_down(z_top_replay, use_fast_weights=False)

        # Anchor current slow weights for EWC-style synaptic importance protection
        anchors_w = [layer.w_slow.detach().clone() for layer in field.layers]
        anchors_b = [layer.b_slow.detach().clone() for layer in field.layers]

        lr = self.config.consolidation_lr
        ewc_lambda = self.config.consolidation_ewc_lambda
        new_loss_val = 0.0
        replay_loss_val = 0.0

        # 3. Local layer-wise distillation into W_slow (100% backprop-free!)
        for _ in range(num_epochs):
            epoch_new_losses = []
            epoch_rep_losses = []
            for idx, layer in enumerate(field.layers):
                # New-knowledge local teacher-student residual (student uses W_slow only)
                phi_new, _ = _activation_and_derivative(teacher_new[idx + 1], layer.activation)
                pred_new = F.linear(phi_new, layer.w_slow, layer.b_slow)
                err_new = teacher_new[idx] - pred_new
                epoch_new_losses.append(err_new.pow(2).mean().item())

                denom_new = phi_new.pow(2).sum(dim=-1).mean().clamp_min(1e-4)
                bsz_new = max(phi_new.shape[0], 1)
                grad_w_new = (err_new.T @ phi_new) / (bsz_new * denom_new)
                grad_b_new = err_new.mean(dim=0)

                # Old-knowledge generative replay preservation residual
                phi_rep, _ = _activation_and_derivative(teacher_replay[idx + 1], layer.activation)
                pred_rep = F.linear(phi_rep, layer.w_slow, layer.b_slow)
                err_rep = teacher_replay[idx] - pred_rep
                epoch_rep_losses.append(err_rep.pow(2).mean().item())

                denom_rep = phi_rep.pow(2).sum(dim=-1).mean().clamp_min(1e-4)
                bsz_rep = max(phi_rep.shape[0], 1)
                grad_w_rep = (err_rep.T @ phi_rep) / (bsz_rep * denom_rep)
                grad_b_rep = err_rep.mean(dim=0)

                # Synaptic importance anchor restoring force (anti-catastrophic forgetting)
                ewc_pull_w = ewc_lambda * layer.importance_w * (layer.w_slow - anchors_w[idx])
                ewc_pull_b = 0.1 * ewc_lambda * (layer.b_slow - anchors_b[idx])

                layer.w_slow.add_(lr * (grad_w_new + grad_w_rep - ewc_pull_w))
                layer.b_slow.add_(lr * (grad_b_new + grad_b_rep - ewc_pull_b))

                # Also locally align amortized bottom-up recognizer w_up to consolidated attractors
                phi_low_all, _ = _activation_and_derivative(
                    torch.cat([teacher_new[idx], teacher_replay[idx]], dim=0),
                    layer.activation,
                )
                z_up_all = torch.cat([teacher_new[idx + 1], teacher_replay[idx + 1]], dim=0)
                up_err = z_up_all - F.linear(phi_low_all, layer.w_up, layer.b_up)
                denom_up = phi_low_all.pow(2).sum(dim=-1).mean().clamp_min(1e-4)
                layer.w_up.add_(
                    0.5 * lr * (up_err.T @ phi_low_all) / (max(phi_low_all.shape[0], 1) * denom_up)
                )
                layer.b_up.add_(0.5 * lr * up_err.mean(dim=0))

            new_loss_val = sum(epoch_new_losses) / max(len(epoch_new_losses), 1)
            replay_loss_val = sum(epoch_rep_losses) / max(len(epoch_rep_losses), 1)

        # 4. Update diagonal synaptic importance omega_slow for future sleep cycles
        for idx, layer in enumerate(field.layers):
            phi_all, _ = _activation_and_derivative(
                torch.cat([teacher_new[idx + 1], teacher_replay[idx + 1]], dim=0),
                layer.activation,
            )
            # Squared pre-synaptic activation energy marks heavily-used consolidated synapses
            activity = phi_all.pow(2).mean(dim=0, keepdim=True).expand_as(layer.w_slow)
            layer.importance_w.mul_(0.85).add_(0.15 * activity)

        # 5. Cooldown fast weights to free short-term plasticity capacity for next wake phase
        if clear_fast_weights:
            fast_memory.reset(field, clear_episodes=True)
        else:
            for layer in field.layers:
                layer.w_fast.mul_(0.1)
            fast_memory.episodes.clear()

        return {
            "consolidated_episodes": float(len(selected)),
            "new_loss_final": new_loss_val,
            "replay_loss_final": replay_loss_val,
        }
