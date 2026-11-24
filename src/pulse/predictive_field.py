"""Pillar 2: Bidirectional Predictive Coding Field with Precision-Weighted Free Energy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.config import PULSEConfig


def _activation_and_derivative(x: torch.Tensor, kind: str = "wide_tanh") -> tuple[torch.Tensor, torch.Tensor]:
    """Compute smooth nonlinearity phi(x) and exact analytic derivative phi'(x).

    `wide_tanh` = 1.6 * tanh(x / 1.6) keeps phi'(x) near 1.0 across [-1.6, 1.6] for clean
    multi-layer error propagation while remaining strictly bounded (Lyapunov-safe, no blow-up).
    """
    if kind == "wide_tanh":
        scale = 1.6
        t = torch.tanh(x / scale)
        y = scale * t
        dy = 1.0 - t * t
        return y, dy
    if kind == "tanh":
        y = torch.tanh(x)
        dy = 1.0 - y * y
        return y, dy
    if kind == "silu":
        sig = torch.sigmoid(x)
        y = x * sig
        dy = sig * (1.0 + x * (1.0 - sig))
        return y, dy
    raise ValueError(f"Unsupported activation kind: {kind}")


class PredictiveLayer(nn.Module):
    """Single bidirectional predictive coding interface between level l+1 (upper) and level l (lower).

    Top-Down flow:  z_hat^(l) = W_eff^(l) @ phi(z^(l+1)) + b_slow^(l), where W_eff = W_slow + W_fast
    Bottom-Up flow: eps^(l)   = z^(l) - z_hat^(l)   (surprise / prediction error)

    All state gradients and slow-weight gradients are computed analytically from strictly local
    layer signals (z^(l), z^(l+1), eps^(l)), enabling zero-backprop local energy learning.
    """

    def __init__(
        self,
        upper_dim: int,
        lower_dim: int,
        precision_min: float = 0.1,
        activation: str = "wide_tanh",
    ) -> None:
        super().__init__()
        self.upper_dim = upper_dim
        self.lower_dim = lower_dim
        self.precision_min = precision_min
        self.activation = activation

        # Semi-orthogonal initialization (gain 1.15 compensates for tanh compression)
        w_init = torch.empty(lower_dim, upper_dim)
        nn.init.orthogonal_(w_init, gain=1.15)
        self.w_slow = nn.Parameter(w_init)
        self.b_slow = nn.Parameter(torch.zeros(lower_dim))

        # Plastic mid-term fast weights (updated online during inference, not frozen)
        self.register_buffer("w_fast", torch.zeros(lower_dim, upper_dim))
        # Diagonal synaptic importance for catastrophic-forgetting protection during sleep
        self.register_buffer("importance_w", torch.zeros(lower_dim, upper_dim))

        # Precision weighting: pi = softplus(log_precision) + precision_min, initialized to ~1.0
        target_sp = max(1.0 - precision_min, 1e-3)
        init_raw = math.log(math.expm1(target_sp))
        self.log_precision = nn.Parameter(torch.full((lower_dim,), init_raw))

        # Independent orthogonal amortized bottom-up recognizer: novel inputs start surprising,
        # familiar inputs become fast (1-2 steps) once learned locally!
        w_up_init = torch.empty(upper_dim, lower_dim)
        nn.init.orthogonal_(w_up_init)
        self.w_up = nn.Parameter(w_up_init * 0.7)
        self.b_up = nn.Parameter(torch.zeros(upper_dim))

    @property
    def w_eff(self) -> torch.Tensor:
        """Combined effective synaptic weights: W_slow + W_fast."""
        return self.w_slow + self.w_fast

    @property
    def precision(self) -> torch.Tensor:
        """Strictly positive per-dimension precision vector pi^(l)."""
        return F.softplus(self.log_precision) + self.precision_min

    def propose_up(self, z_lower: torch.Tensor) -> torch.Tensor:
        """Fast amortized bottom-up initialization z^(l) -> z_init^(l+1)."""
        phi_l, _ = _activation_and_derivative(z_lower, self.activation)
        return F.linear(phi_l, self.w_up, self.b_up)

    def predict_down(self, z_upper: torch.Tensor, use_fast_weights: bool = True) -> torch.Tensor:
        """Top-down generative prediction z_hat^(l) = f_theta(z^(l+1))."""
        phi_u, _ = _activation_and_derivative(z_upper, self.activation)
        weight = self.w_eff if use_fast_weights else self.w_slow
        return F.linear(phi_u, weight, self.b_slow)

    def prediction_error(
        self,
        z_lower: torch.Tensor,
        z_upper: torch.Tensor,
        use_fast_weights: bool = True,
    ) -> torch.Tensor:
        """Bottom-up prediction error (surprise signal): eps^(l) = z^(l) - z_hat^(l)."""
        return z_lower - self.predict_down(z_upper, use_fast_weights=use_fast_weights)

    def energy(
        self,
        z_lower: torch.Tensor,
        z_upper: torch.Tensor,
        use_fast_weights: bool = True,
    ) -> torch.Tensor:
        """Dimension-normalized precision-weighted free energy per sample: [B]."""
        eps = self.prediction_error(z_lower, z_upper, use_fast_weights=use_fast_weights)
        pi = self.precision
        return 0.5 * (pi * eps.pow(2)).mean(dim=-1)

    def local_state_gradients(
        self,
        z_lower: torch.Tensor,
        z_upper: torch.Tensor,
        use_fast_weights: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Exact analytical local gradients of layer energy w.r.t. z_lower and z_upper."""
        phi_u, dphi_u = _activation_and_derivative(z_upper, self.activation)
        weight = self.w_eff if use_fast_weights else self.w_slow
        pred = F.linear(phi_u, weight, self.b_slow)
        eps = z_lower - pred

        # w_l = pi / d_l because energy averages over lower_dim
        w_l = self.precision / self.lower_dim
        weighted_eps = w_l * eps  # [B, lower_dim]
        dE_dz_lower = weighted_eps
        dE_dz_upper = -dphi_u * (weighted_eps @ weight)

        # Strictly local Gauss-Newton diagonal curvature for dimension-invariant relaxation
        hess_diag_lower = w_l.unsqueeze(0).expand_as(z_lower)
        col_curv = (w_l.unsqueeze(1) * weight.pow(2)).sum(dim=0)  # [upper_dim]
        hess_diag_upper = (dphi_u.pow(2)) * col_curv.unsqueeze(0)
        return {
            "eps": eps,
            "weighted_eps": weighted_eps,
            "dE_dz_lower": dE_dz_lower,
            "dE_dz_upper": dE_dz_upper,
            "hess_diag_lower": hess_diag_lower,
            "hess_diag_upper": hess_diag_upper,
        }

    def local_weight_gradients(
        self,
        z_lower: torch.Tensor,
        z_upper: torch.Tensor,
        use_fast_weights: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Exact analytical local gradients of batch-mean energy w.r.t. slow weights & precision."""
        phi_u, _ = _activation_and_derivative(z_upper, self.activation)
        weight = self.w_eff if use_fast_weights else self.w_slow
        pred = F.linear(phi_u, weight, self.b_slow)
        eps = z_lower - pred

        bsz = max(z_lower.shape[0], 1)
        pi = self.precision
        weighted_eps = (pi / self.lower_dim) * eps  # [B, lower_dim]

        # Exact d(mean_b E_b)/dW_slow and d(mean_b E_b)/db_slow
        dE_dw_slow = -(weighted_eps.T @ phi_u) / bsz
        dE_db_slow = -weighted_eps.mean(dim=0)

        # Variational Gaussian precision gradient: d/d(rho) of 0.5/d * (pi * eps^2 - log(pi))
        mean_sq_err = eps.pow(2).mean(dim=0)
        dpi_drho = torch.sigmoid(self.log_precision)
        dE_dlog_precision = (0.5 / self.lower_dim) * (mean_sq_err - 1.0 / pi) * dpi_drho

        # Local amortized bottom-up proposal gradient: minimize 0.5 * ||z_upper - propose_up(z_lower)||^2
        phi_l, _ = _activation_and_derivative(z_lower, self.activation)
        up_err = z_upper - F.linear(phi_l, self.w_up, self.b_up)
        dE_dw_up = -(up_err.T @ phi_l) / (bsz * self.upper_dim)
        dE_db_up = -up_err.mean(dim=0) / self.upper_dim

        return {
            "eps": eps,
            "phi_upper": phi_u,
            "dE_dw_slow": dE_dw_slow,
            "dE_db_slow": dE_db_slow,
            "dE_dlog_precision": dE_dlog_precision,
            "dE_dw_up": dE_dw_up,
            "dE_db_up": dE_db_up,
        }


@dataclass
class RelaxationResult:
    """Structured output of bidirectional predictive coding relaxation."""

    states: List[torch.Tensor]
    errors: List[torch.Tensor]
    initial_energy: torch.Tensor
    final_energy: torch.Tensor
    energy_trajectory: torch.Tensor
    steps_taken: torch.Tensor
    converged: torch.Tensor


class BidirectionalPredictiveField(nn.Module):
    """Hierarchical Bidirectional Predictive Coding Field.

    Minimizes depth-invariant precision-weighted variational free energy:
        E(z_t) = (1/L) * sum_{l=0}^{L-1} E^(l)(z^(l), z^(l+1)) + E_prior(z^(L), z_prior^(L))

    Key capabilities:
    - Exact analytical local state evolution (zero global autograd graph).
    - Local Gauss-Newton diagonal preconditioning + Lyapunov monotonic safeguarding.
    - Native per-sample Adaptive Compute: early stops converged samples (E <= tau) while hard
      or contradictory samples continue internal latent relaxation.
    - Structural Anti-Hallucination: evaluates bidirectional consistency energy and revises
      corrupted hypotheses toward consistent attractors.
    """

    def __init__(self, config: PULSEConfig, activation: str = "wide_tanh") -> None:
        super().__init__()
        self.config = config
        self.layer_dims = config.layer_dims
        self.layers = nn.ModuleList(
            [
                PredictiveLayer(
                    upper_dim=config.layer_dims[i + 1],
                    lower_dim=config.layer_dims[i],
                    precision_min=config.precision_min,
                    activation=activation,
                )
                for i in range(config.num_layers)
            ]
        )

    def generate_top_down(
        self,
        z_top: torch.Tensor,
        use_fast_weights: bool = True,
    ) -> List[torch.Tensor]:
        """Pure top-down generative sweep from abstract top latent z^(L) down to z^(0)."""
        states: List[torch.Tensor | None] = [None] * len(self.layer_dims)
        states[-1] = z_top
        for idx in range(len(self.layers) - 1, -1, -1):
            states[idx] = self.layers[idx].predict_down(
                states[idx + 1], use_fast_weights=use_fast_weights  # type: ignore[arg-type]
            )
        return states  # type: ignore[return-value]

    @staticmethod
    def _project_ball(z: torch.Tensor, max_rms: float = 1.35) -> torch.Tensor:
        """Keep latent state inside the bounded continuous concept manifold (prevents saturation/blow-up)."""
        max_radius = max_rms * math.sqrt(z.shape[-1])
        norms = torch.linalg.vector_norm(z, ord=2, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (max_radius / norms).clamp_max(1.0)
        return z * scale

    def initialize_states(
        self,
        z_bottom: torch.Tensor,
        top_prior: torch.Tensor | None = None,
        clamp_top: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        """Amortized bottom-up initialization projected onto the bounded concept manifold."""
        states = [z_bottom.clone()]
        cur = z_bottom
        for layer in self.layers:
            cur = self._project_ball(layer.propose_up(cur))
            states.append(cur)

        if clamp_top is not None:
            states[-1] = clamp_top.clone()
        elif top_prior is not None:
            states[-1] = self._project_ball(0.5 * states[-1] + 0.5 * top_prior)
        return states

    def compute_energy_and_errors(
        self,
        states: Sequence[torch.Tensor],
        top_prior: torch.Tensor | None = None,
        use_fast_weights: bool = True,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute depth-invariant per-sample free energy [B] and per-layer prediction errors."""
        bsz = states[0].shape[0]
        total_energy = states[0].new_zeros(bsz)
        errors: List[torch.Tensor] = []
        inv_layers = 1.0 / max(len(self.layers), 1)

        for idx, layer in enumerate(self.layers):
            eps = layer.prediction_error(
                states[idx], states[idx + 1], use_fast_weights=use_fast_weights
            )
            errors.append(eps)
            total_energy = total_energy + inv_layers * 0.5 * (layer.precision * eps.pow(2)).mean(
                dim=-1
            )

        if top_prior is not None:
            prior_eps = states[-1] - top_prior
            total_energy = total_energy + 0.5 * self.config.prior_precision * prior_eps.pow(2).mean(
                dim=-1
            )
        return total_energy, errors

    def _compute_local_state_grads(
        self,
        states: Sequence[torch.Tensor],
        top_prior: torch.Tensor | None = None,
        clamp_bottom: bool = True,
        clamp_top: bool = False,
        use_fast_weights: bool = True,
    ) -> List[torch.Tensor]:
        """Assemble local Gauss-Newton preconditioned state descent directions from adjacent layers."""
        grads = [torch.zeros_like(s) for s in states]
        curv = [torch.zeros_like(s) for s in states]
        inv_layers = 1.0 / max(len(self.layers), 1)
        for idx, layer in enumerate(self.layers):
            local = layer.local_state_gradients(
                states[idx], states[idx + 1], use_fast_weights=use_fast_weights
            )
            grads[idx] = grads[idx] + inv_layers * local["dE_dz_lower"]
            grads[idx + 1] = grads[idx + 1] + inv_layers * local["dE_dz_upper"]
            curv[idx] = curv[idx] + inv_layers * local["hess_diag_lower"]
            curv[idx + 1] = curv[idx + 1] + inv_layers * local["hess_diag_upper"]

        if top_prior is not None:
            prior_w = self.config.prior_precision / self.layer_dims[-1]
            grads[-1] = grads[-1] + prior_w * (states[-1] - top_prior)
            curv[-1] = curv[-1] + prior_w

        # Local Levenberg-Marquardt preconditioning (strictly positive diagonal -> strict descent)
        precond: List[torch.Tensor] = []
        max_norm = self.config.infer_grad_clip
        for l_idx, (g, h) in enumerate(zip(grads, curv)):
            if (l_idx == 0 and clamp_bottom) or (l_idx == len(grads) - 1 and clamp_top):
                precond.append(torch.zeros_like(g))
                continue
            damping = 0.05 / self.layer_dims[l_idx]
            step = g / (h + damping)
            norms = torch.linalg.vector_norm(step, ord=2, dim=-1, keepdim=True).clamp_min(1e-8)
            scale = (max_norm / norms).clamp_max(1.0)
            precond.append(step * scale)
        return precond

    @torch.no_grad()
    def relax(
        self,
        z_bottom: torch.Tensor,
        top_prior: torch.Tensor | None = None,
        clamp_bottom: bool = True,
        clamp_top: torch.Tensor | None = None,
        init_states: Sequence[torch.Tensor] | None = None,
        max_steps: int | None = None,
        energy_threshold: float | None = None,
        use_fast_weights: bool = True,
    ) -> RelaxationResult:
        """Iteratively minimize variational free energy via local damped momentum dynamics.

        Supports per-sample adaptive compute early stopping and Lyapunov monotonic energy
        safeguarding completely under `torch.no_grad()`.
        """
        steps_limit = max_steps if max_steps is not None else self.config.max_think_steps
        threshold = (
            energy_threshold if energy_threshold is not None else self.config.energy_threshold
        )
        is_top_clamped = clamp_top is not None

        if init_states is not None:
            states = [s.detach().clone() for s in init_states]
            if clamp_bottom:
                states[0] = z_bottom.detach().clone()
            if clamp_top is not None:
                states[-1] = clamp_top.detach().clone()
        else:
            states = self.initialize_states(
                z_bottom.detach(),
                top_prior=top_prior.detach() if top_prior is not None else None,
                clamp_top=clamp_top.detach() if clamp_top is not None else None,
            )

        prior_detached = top_prior.detach() if top_prior is not None else None
        bsz = states[0].shape[0]
        device = states[0].device

        current_energy, errors = self.compute_energy_and_errors(
            states, top_prior=prior_detached, use_fast_weights=use_fast_weights
        )
        initial_energy = current_energy.clone()
        energy_history = [current_energy.clone()]

        velocities = [torch.zeros_like(s) for s in states]
        step_scales = torch.ones(bsz, 1, device=device, dtype=states[0].dtype)
        steps_taken = torch.zeros(bsz, device=device, dtype=torch.long)
        active = current_energy > threshold

        lr = self.config.infer_lr
        momentum = self.config.infer_momentum

        for _ in range(steps_limit):
            if not active.any():
                break

            grads = self._compute_local_state_grads(
                states,
                top_prior=prior_detached,
                clamp_bottom=clamp_bottom,
                clamp_top=is_top_clamped,
                use_fast_weights=use_fast_weights,
            )

            active_col = active.unsqueeze(-1)
            cand_velocities = [
                torch.where(
                    active_col,
                    momentum * vel - (lr * step_scales) * grad,
                    torch.zeros_like(vel),
                )
                for vel, grad in zip(velocities, grads)
            ]
            cand_states = []
            for l_idx, (s, dv) in enumerate(zip(states, cand_velocities)):
                updated = torch.where(active_col, s + dv, s)
                is_clamped = (l_idx == 0 and clamp_bottom) or (
                    l_idx == len(states) - 1 and is_top_clamped
                )
                if not is_clamped:
                    updated = self._project_ball(updated)
                cand_states.append(updated)

            cand_energy, _ = self.compute_energy_and_errors(
                cand_states, top_prior=prior_detached, use_fast_weights=use_fast_weights
            )

            improved = active & (cand_energy <= current_energy + 1e-7)
            overshot = active & (~improved)

            # Lyapunov fallback: if momentum overshot for any sample, try damped steepest descent
            if overshot.any():
                backtrack_lr = 0.5 * lr * step_scales
                bt_states = []
                for l_idx, (s, g, cs) in enumerate(zip(states, grads, cand_states)):
                    bt = torch.where(overshot.unsqueeze(-1), s - backtrack_lr * g, cs)
                    is_clamped = (l_idx == 0 and clamp_bottom) or (
                        l_idx == len(states) - 1 and is_top_clamped
                    )
                    if not is_clamped:
                        bt = self._project_ball(bt)
                    bt_states.append(bt)
                bt_velocities = [
                    torch.where(overshot.unsqueeze(-1), torch.zeros_like(cv), cv)
                    for cv in cand_velocities
                ]
                bt_energy, _ = self.compute_energy_and_errors(
                    bt_states, top_prior=prior_detached, use_fast_weights=use_fast_weights
                )
                bt_improved = overshot & (bt_energy <= current_energy + 1e-7)

                accept = improved | bt_improved
                accept_col = accept.unsqueeze(-1)
                chosen_states = [
                    torch.where(bt_improved.unsqueeze(-1), bts, cs)
                    for bts, cs in zip(bt_states, cand_states)
                ]
                chosen_velocities = [
                    torch.where(bt_improved.unsqueeze(-1), btv, cv)
                    for btv, cv in zip(bt_velocities, cand_velocities)
                ]
                chosen_energy = torch.where(bt_improved, bt_energy, cand_energy)

                states = [torch.where(accept_col, chs, s) for chs, s in zip(chosen_states, states)]
                velocities = [
                    torch.where(accept_col, chv, torch.zeros_like(v))
                    for chv, v in zip(chosen_velocities, velocities)
                ]
                current_energy = torch.where(accept, chosen_energy, current_energy)

                # Adapt per-sample step scale
                step_scales = torch.where(
                    improved.unsqueeze(-1),
                    (step_scales * 1.05).clamp_max(2.0),
                    step_scales,
                )
                step_scales = torch.where(
                    overshot.unsqueeze(-1),
                    (step_scales * 0.5).clamp_min(1e-3),
                    step_scales,
                )
            else:
                accept_col = improved.unsqueeze(-1)
                states = [torch.where(accept_col, cs, s) for cs, s in zip(cand_states, states)]
                velocities = [
                    torch.where(accept_col, cv, v) for cv, v in zip(cand_velocities, velocities)
                ]
                current_energy = torch.where(improved, cand_energy, current_energy)
                step_scales = torch.where(
                    improved.unsqueeze(-1),
                    (step_scales * 1.05).clamp_max(2.0),
                    step_scales,
                )

            steps_taken = steps_taken + active.long()
            energy_history.append(current_energy.clone())
            active = active & (current_energy > threshold)

        final_energy, errors = self.compute_energy_and_errors(
            states, top_prior=prior_detached, use_fast_weights=use_fast_weights
        )
        return RelaxationResult(
            states=states,
            errors=errors,
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_trajectory=torch.stack(energy_history, dim=0),
            steps_taken=steps_taken,
            converged=final_energy <= threshold,
        )

    @torch.no_grad()
    def evaluate_consistency(
        self,
        z_bottom: torch.Tensor,
        z_top_hypothesis: torch.Tensor,
        max_steps: int | None = None,
        use_fast_weights: bool = True,
    ) -> RelaxationResult:
        """Structural anti-hallucination check: clamp premise & hypothesis, measure residual energy."""
        return self.relax(
            z_bottom=z_bottom,
            clamp_bottom=True,
            clamp_top=z_top_hypothesis,
            max_steps=max_steps,
            energy_threshold=0.0,  # Fully relax internal constraints to get true equilibrium energy
            use_fast_weights=use_fast_weights,
        )
