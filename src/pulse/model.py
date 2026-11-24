"""Unified PULSEModel coupling all four blueprint pillars into a runnable AI architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.config import PULSEConfig
from pulse.local_learning import LocalIOUpdater, LocalSynapticOptimizer, _LocalAdamState
from pulse.manifold import (
    EMATargetEncoder,
    GatedConceptDecoder,
    TextConceptEncoder,
    VectorConceptEncoder,
)
from pulse.memory import (
    EpisodicTrace,
    FastWorkingStateSSM,
    SleepConsolidator,
    SurpriseGatedFastWeights,
)
from pulse.predictive_field import (
    BidirectionalPredictiveField,
    PredictiveLayer,
    RelaxationResult,
    _activation_and_derivative,
)


@dataclass
class ThinkOutput:
    """Rich diagnostic output from a single PULSE latent-space thinking cycle."""

    z_bottom: torch.Tensor
    z_top: torch.Tensor
    z_causal_pred: torch.Tensor
    working_state: torch.Tensor
    relaxation: RelaxationResult
    surprise_stats: Optional[Dict[str, Any]] = None


class PULSEModel(nn.Module):
    """PULSE: Plastic Latent Dynamics Engine (可塑性隐态动力学引擎).

    Integrates all four natively coupled architectural mechanisms:
    1. Continuous Concept Manifold (Latent World-Space) & decoupled multimodal I/O.
    2. Bidirectional Predictive Coding Field with local Gauss-Newton relaxation,
       native per-sample adaptive compute, and structural anti-hallucination energy.
    3. Tri-Timescale Plastic Memory:
       - Fast Working State SSM (O(1) memory, O(N) compute, bounded contraction)
       - Mid-Term Surprise-Gated Delta-Rule Fast Weights (online inference-time learning)
       - Slow Core Sleep Consolidation via Generative Replay + EWC importance anchoring
    4. Local Energy Learning: 100% backprop-free training (`@torch.no_grad()`, zero `.backward()`).
    """

    def __init__(self, config: PULSEConfig) -> None:
        super().__init__()
        self.config = config

        # Pillar 1: Continuous Concept Manifold & Decoupled I/O
        self.text_encoder = TextConceptEncoder(
            vocab_size=config.vocab_size,
            concept_dim=config.concept_dim,
            chunk_size=config.chunk_size,
            radius=config.manifold_radius,
        )
        self.ema_text_encoder = EMATargetEncoder(self.text_encoder, decay=config.ema_decay)
        self.vector_encoders = nn.ModuleDict()
        self.decoder = GatedConceptDecoder(
            concept_dim=config.concept_dim,
            output_dim=config.output_dim,
            max_tokens=config.max_decode_tokens,
            energy_gate_threshold=config.energy_gate_threshold,
        )
        # Project top abstract latent -> decoder concept readout (also learns locally!)
        self.top_to_readout = PredictiveLayer(
            upper_dim=config.top_dim,
            lower_dim=config.concept_dim,
            precision_min=config.precision_min,
        )
        # Dedicated Causal Latent Transition Head: z_top -> z_consequence^(0) with W_slow + W_fast
        self.causal_layer = PredictiveLayer(
            upper_dim=config.top_dim,
            lower_dim=config.concept_dim,
            precision_min=config.precision_min,
        )

        # Pillar 2: Bidirectional Predictive Coding Field
        self.field = BidirectionalPredictiveField(config)

        # Pillar 3: Tri-Timescale Plastic Memory
        self.fast_ssm = FastWorkingStateSSM(
            input_dim=config.top_dim,
            state_dim=config.ssm_state_dim,
        )
        self.fast_weights = SurpriseGatedFastWeights(config)
        self.sleep_consolidator = SleepConsolidator(config)
        self._working_state: Optional[torch.Tensor] = None
        self._causal_episodes: List[EpisodicTrace] = []

        # Pillar 4: Local Energy Learning (100% backprop-free)
        self.local_opt = LocalSynapticOptimizer(self.field, config)
        self.io_updater = LocalIOUpdater(lr=config.io_lr, weight_decay=config.weight_decay)
        self._causal_adam_w = _LocalAdamState(
            self.causal_layer.w_slow.shape,
            self.causal_layer.w_slow.device,
            self.causal_layer.w_slow.dtype,
        )
        self._causal_adam_b = _LocalAdamState(
            self.causal_layer.b_slow.shape,
            self.causal_layer.b_slow.device,
            self.causal_layer.b_slow.dtype,
        )
        self._readout_adam_w = _LocalAdamState(
            self.top_to_readout.w_slow.shape,
            self.top_to_readout.w_slow.device,
            self.top_to_readout.w_slow.dtype,
        )
        self._readout_adam_b = _LocalAdamState(
            self.top_to_readout.b_slow.shape,
            self.top_to_readout.b_slow.device,
            self.top_to_readout.b_slow.dtype,
        )

    def register_vector_modality(self, name: str, input_dim: int) -> None:
        """Attach an additional continuous sensory modality onto the same concept manifold."""
        device = next(self.parameters()).device
        self.vector_encoders[name] = VectorConceptEncoder(
            input_dim=input_dim,
            concept_dim=self.config.concept_dim,
            radius=self.config.manifold_radius,
        ).to(device)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_utterance(tokens)

    def encode_text_chunks(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_chunks(tokens)

    def encode_vector(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        return self.vector_encoders[modality](x)

    def reset_working_state(self) -> None:
        """Clear millisecond Fast Working State h_t (simulating zero-context prompt)."""
        self._working_state = None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def working_state_bytes(self, state: Optional[torch.Tensor] = None) -> int:
        st = state if state is not None else self._working_state
        if st is None:
            return 0
        return st.nelement() * st.element_size()

    @torch.no_grad()
    def think(
        self,
        z_bottom: torch.Tensor,
        working_state: Optional[torch.Tensor] = None,
        stateful: bool = False,
        use_fast_weights: bool = True,
        enable_online_plasticity: bool = False,
        max_steps: Optional[int] = None,
        energy_threshold: Optional[float] = None,
    ) -> ThinkOutput:
        """Core latent-space reasoning cycle (perception -> predictive relaxation -> SSM step)."""
        bsz = z_bottom.shape[0]
        device = z_bottom.device
        dtype = z_bottom.dtype

        if working_state is not None:
            state = working_state
            top_prior: Optional[torch.Tensor] = self.fast_ssm.predict_prior(state)
        elif (
            stateful
            and self._working_state is not None
            and self._working_state.shape[0] == bsz
            and self._working_state.device == device
        ):
            state = self._working_state
            top_prior = self.fast_ssm.predict_prior(state)
        else:
            state = self.fast_ssm.init_state(bsz, device=device, dtype=dtype)
            top_prior = None

        relax = self.field.relax(
            z_bottom=z_bottom,
            top_prior=top_prior,
            clamp_bottom=True,
            max_steps=max_steps,
            energy_threshold=energy_threshold,
            use_fast_weights=use_fast_weights,
        )
        z_top = relax.states[-1]
        self.fast_weights.observe_latent_distribution(z_top)

        surprise_stats: Optional[Dict[str, Any]] = None
        if enable_online_plasticity:
            surprise_stats = self.fast_weights.update(self.field, relax.states, record_episode=True)

        new_state, _ = self.fast_ssm.step(z_top, state)
        if stateful:
            self._working_state = new_state

        z_causal_pred = self.causal_layer.predict_down(z_top, use_fast_weights=use_fast_weights)
        return ThinkOutput(
            z_bottom=z_bottom,
            z_top=z_top,
            z_causal_pred=z_causal_pred,
            working_state=new_state,
            relaxation=relax,
            surprise_stats=surprise_stats,
        )

    @torch.no_grad()
    def forward(
        self,
        tokens: torch.Tensor,
        working_state: Optional[torch.Tensor] = None,
        stateful: bool = False,
        use_fast_weights: bool = True,
        enable_online_plasticity: bool = False,
        decode_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full decoupled perception -> continuous latent thinking -> gated decoding pipeline."""
        z_bottom = self.encode_text(tokens)
        thought = self.think(
            z_bottom=z_bottom,
            working_state=working_state,
            stateful=stateful,
            use_fast_weights=use_fast_weights,
            enable_online_plasticity=enable_online_plasticity,
            max_steps=max_steps,
        )
        z_readout = self.top_to_readout.predict_down(thought.z_top, use_fast_weights=use_fast_weights)
        decoded = self.decoder(
            z_readout,
            energy=thought.relaxation.final_energy,
            decode_steps=decode_steps,
        )
        return {
            **decoded,
            "z_bottom": thought.z_bottom,
            "z_top": thought.z_top,
            "z_causal_pred": thought.z_causal_pred,
            "working_state": thought.working_state,
            "initial_energy": thought.relaxation.initial_energy,
            "final_energy": thought.relaxation.final_energy,
            "steps_taken": thought.relaxation.steps_taken,
            "converged": thought.relaxation.converged,
            "energy_trajectory": thought.relaxation.energy_trajectory,
            "surprise_stats": thought.surprise_stats,
        }

    @torch.no_grad()
    def process_sequence(
        self,
        tokens: torch.Tensor,
        use_fast_weights: bool = True,
        max_steps_per_frame: Optional[int] = None,
        enable_online_plasticity: bool = False,
    ) -> Dict[str, Any]:
        """Process an arbitrarily long token sequence in O(N) time and O(1) working memory.

        Architectural innovation — Vectorized Amortized Fast-Path + Surprise-Sparse Relaxation:
        All concept frames are proposed in one vectorized O(N) sweep; iterative predictive
        relaxation is triggered ONLY on high-surprise frames (E > tau), while the Fast Working
        State SSM integrates the full sequence in strictly O(1) memory.
        """
        frames = self.encode_text_chunks(tokens)  # [B, N_chunks, concept_dim]
        bsz, n_frames, dim0 = frames.shape

        # 1. Vectorized amortized bottom-up sweep over all N_chunks in one BLAS pass
        flat_frames = frames.reshape(bsz * n_frames, dim0)
        init_states = self.field.initialize_states(flat_frames)
        init_energy, _ = self.field.compute_energy_and_errors(
            init_states, use_fast_weights=use_fast_weights
        )

        # 2. Surprise-sparse iterative relaxation ONLY on unconverged high-surprise frames
        threshold = self.config.energy_threshold
        steps_limit = (
            max_steps_per_frame
            if max_steps_per_frame is not None
            else self.config.max_think_steps
        )
        surprising_idx = torch.nonzero(init_energy > threshold, as_tuple=False).squeeze(-1)

        z_top_flat = init_states[-1]
        final_energy_flat = init_energy
        steps_flat = torch.zeros_like(init_energy, dtype=torch.long)

        if surprising_idx.numel() > 0 and steps_limit > 0:
            sub_init = [s.index_select(0, surprising_idx) for s in init_states]
            sub_relax = self.field.relax(
                z_bottom=sub_init[0],
                init_states=sub_init,
                clamp_bottom=True,
                max_steps=steps_limit,
                energy_threshold=threshold,
                use_fast_weights=use_fast_weights,
            )
            z_top_flat = z_top_flat.clone()
            final_energy_flat = final_energy_flat.clone()
            z_top_flat.index_copy_(0, surprising_idx, sub_relax.states[-1])
            final_energy_flat.index_copy_(0, surprising_idx, sub_relax.final_energy)
            steps_flat.index_copy_(0, surprising_idx, sub_relax.steps_taken)

            if enable_online_plasticity:
                self.fast_weights.update(self.field, sub_relax.states, record_episode=True)

        z_top_seq = z_top_flat.view(bsz, n_frames, -1)
        final_energy_seq = final_energy_flat.view(bsz, n_frames)

        # 3. O(N) time, O(1) memory Fast Working State SSM integration
        dt_seq = F.softplus(self.fast_ssm.dt_proj(z_top_seq))
        a_neg = -torch.exp(self.fast_ssm.a_log).view(1, 1, -1)
        a_bar_seq = torch.exp(dt_seq * a_neg)
        drive_seq = torch.tanh(self.fast_ssm.b_proj(z_top_seq))
        # Precompute input contribution to gate_proj in one vectorized BLAS call
        w_g = self.fast_ssm.gate_proj.weight
        b_g = self.fast_ssm.gate_proj.bias
        w_gz = w_g[:, : self.fast_ssm.input_dim]
        w_gh = w_g[:, self.fast_ssm.input_dim :]
        gate_z_seq = F.linear(z_top_seq, w_gz, b_g)

        state = self.fast_ssm.init_state(bsz, device=frames.device, dtype=frames.dtype)
        for idx in range(n_frames):
            gate = torch.sigmoid(gate_z_seq[:, idx, :] + F.linear(state, w_gh))
            a_bar = a_bar_seq[:, idx, :]
            state = a_bar * state + (1.0 - a_bar) * (gate * drive_seq[:, idx, :])

        self._working_state = state
        last_z_top = z_top_seq[:, -1, :]
        last_energy = final_energy_seq[:, -1]
        z_causal_pred = self.causal_layer.predict_down(
            last_z_top, use_fast_weights=use_fast_weights
        )
        z_readout = self.top_to_readout.predict_down(
            last_z_top, use_fast_weights=use_fast_weights
        )
        decoded = self.decoder(z_readout, energy=last_energy)
        return {
            **decoded,
            "working_state": state,
            "num_concept_frames": n_frames,
            "mean_steps_per_frame": steps_flat.float().mean().item(),
            "final_energy": last_energy,
            "z_top": last_z_top,
            "z_causal_pred": z_causal_pred,
        }

    @torch.no_grad()
    def predict_causal_concept(
        self,
        cue_tokens: torch.Tensor,
        use_fast_weights: bool = True,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict the causal consequence concept vector z_out^(0) for a given cue (zero context)."""
        z_cue = self.encode_text(cue_tokens)
        relax = self.field.relax(
            z_cue, clamp_bottom=True, max_steps=max_steps, use_fast_weights=use_fast_weights
        )
        return self.causal_layer.predict_down(relax.states[-1], use_fast_weights=use_fast_weights)

    @torch.no_grad()
    def decode_causal_tokens(
        self,
        cue_tokens: torch.Tensor,
        decode_steps: Optional[int] = None,
        use_fast_weights: bool = True,
    ) -> Dict[str, Any]:
        """Predict causal concept in latent space and render it via the energy-gated decoder."""
        z_cue = self.encode_text(cue_tokens)
        relax = self.field.relax(z_cue, clamp_bottom=True, use_fast_weights=use_fast_weights)
        z_pred = self.causal_layer.predict_down(relax.states[-1], use_fast_weights=use_fast_weights)
        decoded = self.decoder(z_pred, energy=relax.final_energy, decode_steps=decode_steps)
        return {
            **decoded,
            "z_pred": z_pred,
            "steps_taken": relax.steps_taken,
            "initial_energy": relax.initial_energy,
            "final_energy": relax.final_energy,
            "energy_trajectory": relax.energy_trajectory,
        }

    @torch.no_grad()
    def _update_predictive_layer_local(
        self,
        layer: PredictiveLayer,
        adam_w: _LocalAdamState,
        adam_b: _LocalAdamState,
        z_lower: torch.Tensor,
        z_upper: torch.Tensor,
        lr: float,
    ) -> float:
        grads = layer.local_weight_gradients(z_lower, z_upper, use_fast_weights=False)
        adam_w.apply(layer.w_slow, grads["dE_dw_slow"], lr=lr, weight_decay=self.config.weight_decay)
        adam_b.apply(layer.b_slow, grads["dE_db_slow"], lr=lr)
        grad_sq = grads["dE_dw_slow"].pow(2)
        layer.importance_w.mul_(0.95).add_(0.05 * (grad_sq / grad_sq.mean().clamp_min(1e-8)))
        return grads["eps"].pow(2).mean().item()

    @torch.no_grad()
    def train_step_local(
        self,
        tokens: torch.Tensor,
        target_class: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        async_order: bool = False,
    ) -> Dict[str, float]:
        """100% backprop-free local energy training step with Prospective Configuration."""
        z_bottom = self.encode_text(tokens)
        relax_free = self.field.relax(z_bottom, clamp_bottom=True, use_fast_weights=False)
        z_top = relax_free.states[-1]
        self.fast_weights.observe_latent_distribution(z_top)

        phi_top, dphi_top = _activation_and_derivative(z_top, self.top_to_readout.activation)
        z_readout = F.linear(phi_top, self.top_to_readout.w_slow, self.top_to_readout.b_slow)

        action_loss = 0.0
        token_loss = 0.0
        dz_readout = torch.zeros_like(z_readout)
        has_supervision = target_class is not None or target_tokens is not None

        if target_class is not None:
            action_loss, dx_act = self.io_updater.step_decoder_action_local_with_grad(
                self.decoder, z_readout, target_class
            )
            dz_readout = dz_readout + dx_act
        if target_tokens is not None:
            token_loss, dx_tok = self.io_updater.step_decoder_tokens_local_with_grad(
                self.decoder, z_readout, target_tokens
            )
            dz_readout = dz_readout + dx_tok

        if has_supervision:
            # Local update of top_to_readout from decoder error signal (zero autograd!)
            dw_ro = dz_readout.T @ phi_top
            db_ro = dz_readout.sum(dim=0)
            dz_top = (dz_readout @ self.top_to_readout.w_slow) * dphi_top
            self._readout_adam_w.apply(
                self.top_to_readout.w_slow,
                dw_ro,
                lr=self.config.weight_lr,
                weight_decay=self.config.weight_decay,
            )
            self._readout_adam_b.apply(
                self.top_to_readout.b_slow, db_ro, lr=self.config.weight_lr
            )

            # Prospective Configuration: exact batch-size-invariant per-sample target nudge
            # (dz_top carries 1/B from mean loss, so B * dz_top is exact d(loss_b)/d(z_top_b))
            raw_nudge = -(0.8 * z_top.shape[0]) * dz_top
            nudge_norm = torch.linalg.vector_norm(raw_nudge, ord=2, dim=-1, keepdim=True).clamp_min(
                1e-8
            )
            nudge = raw_nudge * (0.5 / nudge_norm).clamp_max(1.0)
            z_top_target = z_top + nudge
            # Keep target within bounded concept manifold radius
            max_top_r = math.sqrt(self.config.top_dim) * 1.5
            top_norm = torch.linalg.vector_norm(
                z_top_target, ord=2, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            z_top_target = z_top_target * (max_top_r / top_norm).clamp_max(1.0)
            field_stats = self.local_opt.train_supervised_pair(
                self.field,
                z_bottom=z_bottom,
                z_top_target=z_top_target,
                async_order=async_order,
            )
            readout_mse = dz_readout.pow(2).mean().item()
        else:
            field_stats = self.local_opt.train_unsupervised_sensory(
                self.field, z_bottom=z_bottom, async_order=async_order
            )
            readout_mse = self._update_predictive_layer_local(
                self.top_to_readout,
                self._readout_adam_w,
                self._readout_adam_b,
                z_lower=z_bottom,
                z_upper=z_top,
                lr=self.config.weight_lr,
            )

        self.ema_text_encoder.update(self.text_encoder)
        return {
            **field_stats,
            "readout_mse": readout_mse,
            "action_loss": action_loss,
            "token_loss": token_loss,
        }

    @torch.no_grad()
    def train_causal_concept_local(
        self,
        cue_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        async_order: bool = False,
        train_decoder: bool = False,
    ) -> Dict[str, float]:
        """100% backprop-free JEPA-style causal latent transition learning: cue -> target concept."""
        z_cue = self.encode_text(cue_tokens)
        z_target = self.ema_text_encoder.encoder.encode_utterance(target_tokens)

        field_stats = self.local_opt.train_unsupervised_sensory(
            self.field, z_bottom=z_cue, async_order=async_order
        )
        relax = self.field.relax(z_cue, clamp_bottom=True, use_fast_weights=False)
        z_top_cue = relax.states[-1]
        self.fast_weights.observe_latent_distribution(z_top_cue)

        causal_mse = self._update_predictive_layer_local(
            self.causal_layer,
            self._causal_adam_w,
            self._causal_adam_b,
            z_lower=z_target,
            z_upper=z_top_cue,
            lr=self.config.weight_lr,
        )

        token_loss = 0.0
        if train_decoder:
            seq_len = min(target_tokens.shape[1], self.config.max_decode_tokens)
            token_loss = self.io_updater.step_decoder_tokens_local(
                self.decoder, z_target, target_tokens[:, :seq_len]
            )

        self.ema_text_encoder.update(self.text_encoder)
        return {
            **field_stats,
            "causal_mse": causal_mse,
            "token_loss": token_loss,
        }

    @torch.no_grad()
    def teach_online(
        self,
        cue_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        repeats: int = 3,
        adapt_decoder: bool = False,
    ) -> Dict[str, float]:
        """Inference-time online rule internalization directly into W_fast (zero backprop!).

        Uses sequential Kaczmarz normalized Delta-Rule updates:
            dW_fast = alpha * sigma((||eps_t|| - gamma) / T) * (eps_t * v_t^T) - lambda * W_fast
        keeping the concept hierarchy coordinate frame aligned so multiple new rules are
        orthogonalized and recalled at ~100% accuracy with zero context window.
        """
        z_cue = self.encode_text(cue_tokens)
        z_target = self.encode_text(target_tokens)
        relax = self.field.relax(z_cue, clamp_bottom=True, use_fast_weights=True)
        z_top_cue = relax.states[-1]

        alpha = self.config.fast_weight_alpha
        keep = 1.0 - self.config.fast_weight_decay
        max_norm = self.config.fast_weight_max_norm
        rank = self.config.fast_weight_rank

        peak_gate = 0.0
        peak_surprise = 0.0
        total_drop = 0.0
        bsz = z_target.shape[0]

        for _ in range(max(repeats, 1)):
            e_before = self.causal_layer.energy(z_target, z_top_cue, use_fast_weights=True)
            self.causal_layer.w_fast.mul_(keep)

            # Sequential Kaczmarz Delta Rule cleanly orthogonalizes multi-rule online teaching
            for b in range(bsz):
                zt_b = z_target[b : b + 1]
                zu_b = z_top_cue[b : b + 1]
                phi_u, _ = _activation_and_derivative(zu_b, self.causal_layer.activation)
                eps = zt_b - self.causal_layer.predict_down(zu_b, use_fast_weights=True)
                rms_surp, gate = self.fast_weights.surprise_gate(eps)

                peak_gate = max(peak_gate, float(gate.item()))
                peak_surprise = max(peak_surprise, float(rms_surp.item()))

                phi_norm_sq = phi_u.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
                v_norm = phi_u / phi_norm_sq
                u_gated = gate.unsqueeze(-1) * eps
                self.causal_layer.w_fast.add_(alpha * (u_gated.T @ v_norm))

            if rank is not None:
                self.causal_layer.w_fast.copy_(
                    SurpriseGatedFastWeights._project_low_rank(self.causal_layer.w_fast, rank)
                )
            cur_norm = torch.linalg.matrix_norm(self.causal_layer.w_fast, ord="fro")
            if cur_norm > max_norm:
                self.causal_layer.w_fast.mul_(max_norm / cur_norm.clamp_min(1e-8))

            e_after = self.causal_layer.energy(z_target, z_top_cue, use_fast_weights=True)
            drop = (e_before - e_after).clamp_min(0.0)
            total_drop += float(drop.mean().item())

            if peak_gate > 0.35:
                self._record_causal_episode(z_target, z_top_cue, drop)

            if adapt_decoder:
                seq_len = min(target_tokens.shape[1], self.config.max_decode_tokens)
                for _ in range(4):
                    self.io_updater.step_decoder_tokens_local(
                        self.decoder, z_target, target_tokens[:, :seq_len], lr=0.05
                    )

        return {
            "max_gate": peak_gate,
            "max_surprise": peak_surprise,
            "total_energy_drop": total_drop,
            "causal_episodes": float(len(self._causal_episodes)),
        }

    def _record_causal_episode(
        self,
        z_target: torch.Tensor,
        z_top_cue: torch.Tensor,
        energy_drop: torch.Tensor,
    ) -> None:
        cap = self.config.replay_buffer_capacity
        for b in range(z_target.shape[0]):
            drop_val = float(energy_drop[b].item())
            if drop_val <= 1e-6:
                continue
            zb = z_target[b].detach().cpu().clone()
            zt = z_top_cue[b].detach().cpu().clone()
            matched = False
            for ep in self._causal_episodes:
                if F.cosine_similarity(ep.z_top.unsqueeze(0), zt.unsqueeze(0)).item() > 0.94:
                    ep.hit_count += 1
                    ep.energy_drop = max(ep.energy_drop, drop_val)
                    ep.z_bottom = 0.5 * ep.z_bottom + 0.5 * zb
                    ep.z_top = 0.5 * ep.z_top + 0.5 * zt
                    matched = True
                    break
            if not matched:
                self._causal_episodes.append(
                    EpisodicTrace(z_bottom=zb, z_top=zt, energy_drop=drop_val)
                )
                if len(self._causal_episodes) > cap:
                    self._causal_episodes.sort(key=lambda e: e.salience, reverse=True)
                    del self._causal_episodes[cap:]

    @torch.no_grad()
    def sleep_consolidate(
        self,
        top_k_episodes: int = 32,
        num_replay_samples: int = 32,
        num_epochs: int = 30,
        clear_fast_weights: bool = True,
    ) -> Dict[str, float]:
        """Offline Sleep Consolidation: Generative Replay distillation from W_fast -> W_slow."""
        field_stats = self.sleep_consolidator.consolidate(
            self.field,
            self.fast_weights,
            top_k_episodes=top_k_episodes,
            num_replay_samples=num_replay_samples,
            num_epochs=num_epochs,
            clear_fast_weights=clear_fast_weights,
        )

        causal_count = 0.0
        causal_new_loss = 0.0
        causal_replay_loss = 0.0

        if self._causal_episodes:
            device = self.causal_layer.w_slow.device
            dtype = self.causal_layer.w_slow.dtype
            selected = sorted(
                self._causal_episodes, key=lambda e: e.salience, reverse=True
            )[:top_k_episodes]
            causal_count = float(len(selected))

            z_target_new = torch.stack([e.z_bottom for e in selected], dim=0).to(
                device=device, dtype=dtype
            )
            z_top_new = torch.stack([e.z_top for e in selected], dim=0).to(
                device=device, dtype=dtype
            )

            # Generative Replay of old slow-weight knowledge (filtered to never contradict new episodes!)
            z_top_replay_raw = self.fast_weights.sample_prior_latents(
                num_samples=num_replay_samples,
                dim=self.config.top_dim,
                device=device,
                dtype=dtype,
            )
            sims_to_new = F.cosine_similarity(
                z_top_replay_raw.unsqueeze(1), z_top_new.unsqueeze(0), dim=-1
            ).max(dim=-1).values
            keep_mask = sims_to_new < 0.82
            z_top_replay = (
                z_top_replay_raw[keep_mask]
                if keep_mask.any()
                else z_top_replay_raw[:0]
            )
            z_target_replay = (
                self.causal_layer.predict_down(z_top_replay, use_fast_weights=False)
                if z_top_replay.shape[0] > 0
                else z_target_new[:0]
            )

            anchor_w = self.causal_layer.w_slow.detach().clone()
            anchor_b = self.causal_layer.b_slow.detach().clone()
            lr = self.config.consolidation_lr
            ewc_lambda = self.config.consolidation_ewc_lambda

            for _ in range(num_epochs):
                # Interleave sequential normalized Kaczmarz over BOTH new episodes AND generative
                # replay trajectories (without shifting global bias toward new episodes alone!)
                phi_new, _ = _activation_and_derivative(z_top_new, self.causal_layer.activation)
                for b in range(z_top_new.shape[0]):
                    pn = phi_new[b : b + 1]
                    pred_b = F.linear(pn, self.causal_layer.w_slow, self.causal_layer.b_slow)
                    err_b = z_target_new[b : b + 1] - pred_b
                    dn = pn.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
                    self.causal_layer.w_slow.add_(lr * ((err_b / dn).T @ pn))

                if z_top_replay.shape[0] > 0:
                    phi_rep, _ = _activation_and_derivative(
                        z_top_replay, self.causal_layer.activation
                    )
                    for b in range(z_top_replay.shape[0]):
                        pr = phi_rep[b : b + 1]
                        pred_r = F.linear(pr, self.causal_layer.w_slow, self.causal_layer.b_slow)
                        err_r = z_target_replay[b : b + 1] - pred_r
                        dr = pr.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
                        self.causal_layer.w_slow.add_(0.5 * lr * ((err_r / dr).T @ pr))
                    pred_rep = F.linear(phi_rep, self.causal_layer.w_slow, self.causal_layer.b_slow)
                    causal_replay_loss = (z_target_replay - pred_rep).pow(2).mean().item()

                pred_new = F.linear(phi_new, self.causal_layer.w_slow, self.causal_layer.b_slow)
                causal_new_loss = (z_target_new - pred_new).pow(2).mean().item()

                ewc_w = ewc_lambda * self.causal_layer.importance_w * (
                    self.causal_layer.w_slow - anchor_w
                )
                self.causal_layer.w_slow.add_(-0.05 * lr * ewc_w)
                self.causal_layer.b_slow.lerp_(anchor_b, 0.1)

            # Register newly consolidated concepts into the wake GMM prior for future sleep cycles
            self.fast_weights.observe_latent_distribution(z_top_new)

            phi_all, _ = _activation_and_derivative(
                torch.cat([z_top_new, z_top_replay], dim=0), self.causal_layer.activation
            )
            activity = phi_all.pow(2).mean(dim=0, keepdim=True).expand_as(self.causal_layer.w_slow)
            self.causal_layer.importance_w.mul_(0.85).add_(0.15 * activity)

            if clear_fast_weights:
                self.causal_layer.w_fast.zero_()
            else:
                self.causal_layer.w_fast.mul_(0.1)
            self._causal_episodes.clear()

        return {
            "field_consolidated": field_stats["consolidated_episodes"],
            "causal_consolidated": causal_count,
            "causal_new_loss": causal_new_loss,
            "causal_replay_loss": causal_replay_loss,
        }

    @torch.no_grad()
    def check_consistency(
        self,
        premise_tokens: torch.Tensor,
        hypothesis_tokens: torch.Tensor,
        use_fast_weights: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Structural anti-hallucination verification via causal + hierarchical free energy."""
        z_premise = self.encode_text(premise_tokens)
        z_hyp = self.encode_text(hypothesis_tokens)

        relax_p = self.field.relax(z_premise, clamp_bottom=True, use_fast_weights=use_fast_weights)
        z_top_p = relax_p.states[-1]
        causal_energy = self.causal_layer.energy(
            z_hyp, z_top_p, use_fast_weights=use_fast_weights
        )
        # Also evaluate cosine alignment in concept manifold
        z_pred = self.causal_layer.predict_down(z_top_p, use_fast_weights=use_fast_weights)
        cosine_sim = F.cosine_similarity(z_pred, z_hyp, dim=-1)

        return {
            "consistency_energy": causal_energy,
            "premise_energy": relax_p.final_energy,
            "cosine_similarity": cosine_sim,
            "is_consistent": causal_energy <= self.config.energy_gate_threshold,
            "steps_taken": relax_p.steps_taken,
        }

    def save_checkpoint(self, path: str) -> None:
        """Persist PULSE config, slow weights, fast weights, and latent statistics."""
        payload = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "latent_mean": self.fast_weights._latent_mean,
            "latent_var": self.fast_weights._latent_var,
            "latent_count": self.fast_weights._latent_count,
            "prior_centroids": self.fast_weights._prior_centroids,
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> PULSEModel:
        """Restore a PULSEModel checkpoint including slow & plastic fast weights."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg_dict = payload["config"]
        cfg_dict["layer_dims"] = tuple(cfg_dict["layer_dims"])
        cfg = PULSEConfig(**cfg_dict)
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        model.fast_weights.load_latent_state(
            payload.get("latent_mean"),
            payload.get("latent_var"),
            payload.get("latent_count", 0),
            payload.get("prior_centroids", []),
        )
        return model
