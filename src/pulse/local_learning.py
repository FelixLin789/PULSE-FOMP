"""Pillar 4: Local Energy Learning — 100% Backprop-Free Synaptic Plasticity Engine."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from pulse.config import PULSEConfig
from pulse.manifold import GatedConceptDecoder
from pulse.predictive_field import BidirectionalPredictiveField


class _LocalAdamState:
    """Self-contained per-tensor AdamW state operating strictly inside torch.no_grad()."""

    def __init__(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> None:
        self.m = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.step = 0

    @torch.no_grad()
    def apply(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        beta1: float = 0.85,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        self.step += 1
        if self.m.device != param.device or self.m.dtype != param.dtype:
            self.m = self.m.to(device=param.device, dtype=param.dtype)
            self.v = self.v.to(device=param.device, dtype=param.dtype)

        self.m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        self.v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bc1 = 1.0 - beta1**self.step
        bc2 = 1.0 - beta2**self.step
        m_hat = self.m / bc1
        v_hat = self.v / bc2

        # Maximal Update Parameterization (muP) fan-in scaling: keeps output delta O(1)
        # across arbitrary layer widths (d=16 to d=4096) and prevents Adam overshoot.
        if param.ndim >= 2:
            fan_in = max(param.shape[1], 1)
            eff_lr = lr * min(1.0, 8.0 / fan_in)
        else:
            eff_lr = lr * 0.5

        if weight_decay > 0.0:
            param.mul_(1.0 - eff_lr * weight_decay)
        param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-eff_lr)


class LocalSynapticOptimizer:
    """Backprop-free layer-local optimizer for BidirectionalPredictiveField.

    Every PredictiveLayer owns independent local optimizer state and updates purely from
    its own adjacent state pair (z^(l), z^(l+1)) inside `@torch.no_grad()`.
    No global computation graph is constructed, no activations are retained for global
    backpropagation, and layers can update synchronously or in arbitrary asynchronous order.
    """

    def __init__(
        self,
        field: BidirectionalPredictiveField,
        config: PULSEConfig,
        beta1: float = 0.85,
        beta2: float = 0.99,
    ) -> None:
        self.config = config
        self.beta1 = beta1
        self.beta2 = beta2
        self._layer_states: List[Dict[str, _LocalAdamState]] = []
        for layer in field.layers:
            self._layer_states.append(
                {
                    "w_slow": _LocalAdamState(layer.w_slow.shape, layer.w_slow.device, layer.w_slow.dtype),
                    "b_slow": _LocalAdamState(layer.b_slow.shape, layer.b_slow.device, layer.b_slow.dtype),
                    "log_precision": _LocalAdamState(
                        layer.log_precision.shape,
                        layer.log_precision.device,
                        layer.log_precision.dtype,
                    ),
                    "w_up": _LocalAdamState(layer.w_up.shape, layer.w_up.device, layer.w_up.dtype),
                    "b_up": _LocalAdamState(layer.b_up.shape, layer.b_up.device, layer.b_up.dtype),
                }
            )

    @torch.no_grad()
    def step_layer(
        self,
        field: BidirectionalPredictiveField,
        layer_idx: int,
        z_lower: torch.Tensor,
        z_upper: torch.Tensor,
        lr: float | None = None,
        update_precision: bool = True,
        update_amortized: bool = True,
        use_fast_weights: bool = False,
    ) -> float:
        """Update a single PredictiveLayer purely from adjacent states (z^(l), z^(l+1))."""
        layer = field.layers[layer_idx]
        st = self._layer_states[layer_idx]
        step_lr = lr if lr is not None else self.config.weight_lr
        wd = self.config.weight_decay

        grads = layer.local_weight_gradients(
            z_lower.detach(), z_upper.detach(), use_fast_weights=use_fast_weights
        )

        # Strictly local synaptic orthogonality homeostasis (target gain^2=1.32 prevents collapse)
        w = layer.w_slow
        target_sq = 1.32
        if w.shape[0] <= w.shape[1]:
            eye = torch.eye(w.shape[0], device=w.device, dtype=w.dtype) * target_sq
            ortho_grad = ((w @ w.T - eye) @ w) / max(w.shape[0], 1)
        else:
            eye = torch.eye(w.shape[1], device=w.device, dtype=w.dtype) * target_sq
            ortho_grad = (w @ (w.T @ w - eye)) / max(w.shape[1], 1)
        total_w_grad = grads["dE_dw_slow"] + 0.015 * ortho_grad

        st["w_slow"].apply(
            layer.w_slow,
            total_w_grad,
            lr=step_lr,
            beta1=self.beta1,
            beta2=self.beta2,
            weight_decay=wd,
        )
        st["b_slow"].apply(
            layer.b_slow,
            grads["dE_db_slow"],
            lr=step_lr,
            beta1=self.beta1,
            beta2=self.beta2,
        )

        if update_precision:
            st["log_precision"].apply(
                layer.log_precision,
                grads["dE_dlog_precision"],
                lr=0.25 * step_lr,
                beta1=self.beta1,
                beta2=self.beta2,
            )
            layer.log_precision.clamp_(-1.5, 2.5)

        if update_amortized:
            # Amortized recognizer learns faster so familiar concepts converge in 1-2 steps
            st["w_up"].apply(
                layer.w_up,
                grads["dE_dw_up"],
                lr=2.0 * step_lr,
                beta1=self.beta1,
                beta2=self.beta2,
                weight_decay=wd,
            )
            st["b_up"].apply(
                layer.b_up,
                grads["dE_db_up"],
                lr=2.0 * step_lr,
                beta1=self.beta1,
                beta2=self.beta2,
            )

        # Track diagonal synaptic importance locally for Sleep Consolidation EWC
        grad_sq = grads["dE_dw_slow"].pow(2)
        norm_factor = grad_sq.mean().clamp_min(1e-8)
        layer.importance_w.mul_(0.95).add_(0.05 * (grad_sq / norm_factor))

        return grads["eps"].pow(2).mean().item()

    @torch.no_grad()
    def step_field(
        self,
        field: BidirectionalPredictiveField,
        settled_states: Sequence[torch.Tensor],
        lr: float | None = None,
        async_order: bool = False,
        layer_subset: Sequence[int] | None = None,
        update_precision: bool = True,
        update_amortized: bool = True,
        use_fast_weights: bool = False,
    ) -> Dict[str, float]:
        """Update all (or a subset of) layers independently using settled local states."""
        indices = list(layer_subset) if layer_subset is not None else list(range(len(field.layers)))
        if async_order and len(indices) > 1:
            perm = torch.randperm(len(indices)).tolist()
            indices = [indices[i] for i in perm]

        layer_mses: List[float] = []
        for idx in indices:
            mse = self.step_layer(
                field,
                idx,
                settled_states[idx],
                settled_states[idx + 1],
                lr=lr,
                update_precision=update_precision,
                update_amortized=update_amortized,
                use_fast_weights=use_fast_weights,
            )
            layer_mses.append(mse)

        return {
            "mean_layer_mse": sum(layer_mses) / max(len(layer_mses), 1),
            "layers_updated": float(len(indices)),
        }

    @torch.no_grad()
    def train_supervised_pair(
        self,
        field: BidirectionalPredictiveField,
        z_bottom: torch.Tensor,
        z_top_target: torch.Tensor,
        lr: float | None = None,
        async_order: bool = False,
        max_steps: int | None = None,
    ) -> Dict[str, float]:
        """Prospective-configuration local learning step clamping bottom input & top target."""
        relax = field.relax(
            z_bottom=z_bottom,
            clamp_bottom=True,
            clamp_top=z_top_target,
            max_steps=max_steps,
            energy_threshold=1e-4,
            use_fast_weights=False,
        )
        step_stats = self.step_field(
            field,
            relax.states,
            lr=lr,
            async_order=async_order,
            use_fast_weights=False,
        )
        return {
            "initial_energy": relax.initial_energy.mean().item(),
            "equilibrium_energy": relax.final_energy.mean().item(),
            "think_steps": relax.steps_taken.float().mean().item(),
            **step_stats,
        }

    @torch.no_grad()
    def train_unsupervised_sensory(
        self,
        field: BidirectionalPredictiveField,
        z_bottom: torch.Tensor,
        top_prior: torch.Tensor | None = None,
        lr: float | None = None,
        async_order: bool = False,
        max_steps: int | None = None,
    ) -> Dict[str, float]:
        """Unsupervised local predictive coding step: relax hidden explanation, then update locally."""
        relax = field.relax(
            z_bottom=z_bottom,
            top_prior=top_prior,
            clamp_bottom=True,
            clamp_top=None,
            max_steps=max_steps,
            use_fast_weights=False,
        )
        step_stats = self.step_field(
            field,
            relax.states,
            lr=lr,
            async_order=async_order,
            use_fast_weights=False,
        )
        return {
            "initial_energy": relax.initial_energy.mean().item(),
            "equilibrium_energy": relax.final_energy.mean().item(),
            "think_steps": relax.steps_taken.float().mean().item(),
            **step_stats,
        }


class LocalIOUpdater:
    """Backprop-free local plasticity for lightweight GatedConceptDecoder I/O renderers.

    Updates the 2-layer action and token render heads analytically inside `@torch.no_grad()`,
    ensuring zero global autograd graph ever connects I/O rendering back into the PULSE core.
    """

    def __init__(self, lr: float = 1e-2, weight_decay: float = 1e-4) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self._states: Dict[int, _LocalAdamState] = {}

    def _get_state(self, param: torch.Tensor) -> _LocalAdamState:
        key = id(param)
        if key not in self._states:
            self._states[key] = _LocalAdamState(param.shape, param.device, param.dtype)
        return self._states[key]

    @torch.no_grad()
    def _step_two_layer_ce(
        self,
        l0_weight: torch.Tensor,
        l0_bias: torch.Tensor,
        l2_weight: torch.Tensor,
        l2_bias: torch.Tensor,
        x: torch.Tensor,
        targets: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, torch.Tensor]:
        """Exact analytical cross-entropy local update for Linear -> SiLU -> Linear."""
        step_lr = lr if lr is not None else self.lr
        n_samples = max(x.shape[0], 1)
        num_classes = l2_weight.shape[0]

        pre = F.linear(x, l0_weight, l0_bias)
        sig = torch.sigmoid(pre)
        h = pre * sig
        dsilu = sig * (1.0 + pre * (1.0 - sig))

        logits = F.linear(h, l2_weight, l2_bias)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs, targets).item()

        probs = log_probs.exp()
        one_hot = F.one_hot(targets, num_classes=num_classes).to(dtype=probs.dtype)
        dlogits = (probs - one_hot) / n_samples

        dw2 = dlogits.T @ h
        db2 = dlogits.sum(dim=0)
        dh = (dlogits @ l2_weight) * dsilu
        dw0 = dh.T @ x
        db0 = dh.sum(dim=0)
        dx = dh @ l0_weight

        self._get_state(l2_weight).apply(l2_weight, dw2, lr=step_lr, weight_decay=self.weight_decay)
        self._get_state(l2_bias).apply(l2_bias, db2, lr=step_lr)
        self._get_state(l0_weight).apply(l0_weight, dw0, lr=step_lr, weight_decay=self.weight_decay)
        self._get_state(l0_bias).apply(l0_bias, db0, lr=step_lr)
        return loss, dx

    @torch.no_grad()
    def step_decoder_action_local(
        self,
        decoder: GatedConceptDecoder,
        z: torch.Tensor,
        targets: torch.Tensor,
        lr: float | None = None,
    ) -> float:
        """Backprop-free local update of GatedConceptDecoder.action_head."""
        loss, _ = self.step_decoder_action_local_with_grad(decoder, z, targets, lr=lr)
        return loss

    @torch.no_grad()
    def step_decoder_action_local_with_grad(
        self,
        decoder: GatedConceptDecoder,
        z: torch.Tensor,
        targets: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, torch.Tensor]:
        """Backprop-free local update returning (loss, dL/dz) for prospective configuration."""
        l0 = decoder.action_head[0]
        l2 = decoder.action_head[2]
        return self._step_two_layer_ce(
            l0.weight, l0.bias, l2.weight, l2.bias, z.detach(), targets, lr=lr
        )

    @torch.no_grad()
    def step_decoder_tokens_local(
        self,
        decoder: GatedConceptDecoder,
        z: torch.Tensor,
        token_targets: torch.Tensor,
        lr: float | None = None,
    ) -> float:
        """Backprop-free local update of GatedConceptDecoder.token_head and pos_embed."""
        loss, _ = self.step_decoder_tokens_local_with_grad(decoder, z, token_targets, lr=lr)
        return loss

    @torch.no_grad()
    def step_decoder_tokens_local_with_grad(
        self,
        decoder: GatedConceptDecoder,
        z: torch.Tensor,
        token_targets: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, torch.Tensor]:
        """Backprop-free local token head update returning (loss, dL/dz)."""
        step_lr = lr if lr is not None else self.lr
        bsz, seq_len = token_targets.shape
        pos = decoder.pos_embed[:, :seq_len, :]
        z_seq = z.detach().unsqueeze(1) + pos  # [B, S, D]

        l0 = decoder.token_head[0]
        l2 = decoder.token_head[2]
        loss, dx_flat = self._step_two_layer_ce(
            l0.weight,
            l0.bias,
            l2.weight,
            l2.bias,
            z_seq.reshape(bsz * seq_len, -1),
            token_targets.reshape(-1),
            lr=step_lr,
        )
        dx_seq = dx_flat.view(bsz, seq_len, -1)
        dpos = dx_seq.sum(dim=0, keepdim=True)
        pos_grad_full = torch.zeros_like(decoder.pos_embed)
        pos_grad_full[:, :seq_len, :] = dpos
        self._get_state(decoder.pos_embed).apply(decoder.pos_embed, pos_grad_full, lr=step_lr)
        return loss, dx_seq.sum(dim=1)
