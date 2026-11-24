"""PULSE-v2: Hyper-Plastic Resonance Engine (超塑性隐态共振引擎).

Unifies five frontier & original post-Transformer breakthroughs under strict O(1) working memory
and O(N) sequence compute:
1. Chunkwise WY Gated DeltaNet Matrix Fast-Weight Layers (S_t in R^{H x d_v x d_k}, O(1) memory,
   exact Householder-like Delta Rule surgical erase & write, pure Tensor-Core GEMM speed).
2. Bounded Sliding-Window Echoic Cortex (W=64, O(1) capped ring buffer, O(N*W) block-banded)
   + Depthwise ShortCausalConv (k=4, O(1) conv state) eliminating chunk-boundary blind spots.
3. 100% Backprop-Free Bidirectional Predictive Coding Cortex (`requires_grad=False`, 0 `.grad`
   tensors) with Local Gauss-Newton relaxation, native adaptive compute, and cosine-decayed local
   plasticity injecting top-down semantic feedback directly into the upper resonance stream.
4. Wake-Sleep Local Contrastive Free-Energy Critic (0 backprop!) combining predictive MSE with
   positive/negative InfoNCE energy repulsion for structural anti-hallucination.
5. Tri-Timescale Plastic Memory: O(1) Matrix DeltaNet + O(log N) ParallelSelectiveSSM + online
   surprise-gated W_fast + O(1) GMM Generative Replay Sleep Consolidation (W_fast -> W_slow).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.config import PULSEConfig
from pulse.delta_resonance import (
    ChunkwiseGatedDeltaNet,
    ContrastiveEnergyCritic,
    SlidingWindowEchoicAttention,
)
from pulse.local_learning import LocalSynapticOptimizer, _LocalAdamState
from pulse.manifold import ConceptManifoldProjection
from pulse.memory import EpisodicTrace, SurpriseGatedFastWeights
from pulse.predictive_field import (
    BidirectionalPredictiveField,
    PredictiveLayer,
    _activation_and_derivative,
)
from pulse.pulse_lm import ParallelSelectiveSSM
from pulse.scalable_transformer import RMSNorm


@dataclass
class PULSEv2Config:
    """Configuration for PULSE-v2 (Hyper-Plastic Resonance Engine)."""

    vocab_size: int = 50257
    embed_dim: int = 192
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 960
    num_blocks: int = 6
    echoic_layers: Tuple[int, ...] = (2, 5)
    delta_chunk_size: int = 32
    echoic_window_size: int = 64
    conv_kernel: int = 4

    # Concept Cortex (100% Zero-Backprop Predictive Coding Core)
    chunk_size: int = 4
    layer_dims: Tuple[int, ...] = (320, 480, 576, 480, 320)
    ssm_state_dim: int = 512
    critic_proj_dim: int = 256

    # Predictive Coding & Adaptive Compute
    max_think_steps: int = 16
    train_think_steps: int = 3
    energy_threshold: float = 0.02
    infer_lr: float = 0.2
    infer_momentum: float = 0.45
    infer_grad_clip: float = 5.0
    precision_min: float = 0.1
    prior_precision: float = 0.2

    # Tri-Timescale Plastic Memory
    fast_weight_alpha: float = 0.85
    surprise_threshold: float = 0.15
    surprise_temperature: float = 0.05
    fast_weight_decay: float = 0.0
    fast_weight_rank: int | None = None
    fast_weight_max_norm: float = 36.0
    replay_buffer_capacity: int = 128
    consolidation_ewc_lambda: float = 0.25
    consolidation_lr: float = 0.12

    # Hybrid-Local Learning
    core_weight_lr: float = 0.05
    core_weight_decay: float = 1e-4
    anti_collapse_weight: float = 0.02
    ssm_align_weight: float = 0.10

    @property
    def concept_dim(self) -> int:
        return self.layer_dims[0]

    @property
    def top_dim(self) -> int:
        return self.layer_dims[-1]

    @property
    def mid_block_index(self) -> int:
        return self.num_blocks // 2

    def to_core_config(self) -> PULSEConfig:
        return PULSEConfig(
            vocab_size=self.vocab_size,
            layer_dims=self.layer_dims,
            output_dim=self.vocab_size,
            chunk_size=self.chunk_size,
            ssm_state_dim=self.ssm_state_dim,
            max_think_steps=self.max_think_steps,
            energy_threshold=self.energy_threshold,
            infer_lr=self.infer_lr,
            infer_momentum=self.infer_momentum,
            infer_grad_clip=self.infer_grad_clip,
            precision_min=self.precision_min,
            prior_precision=self.prior_precision,
            fast_weight_alpha=self.fast_weight_alpha,
            surprise_threshold=self.surprise_threshold,
            surprise_temperature=self.surprise_temperature,
            fast_weight_decay=self.fast_weight_decay,
            fast_weight_rank=self.fast_weight_rank,
            fast_weight_max_norm=self.fast_weight_max_norm,
            replay_buffer_capacity=self.replay_buffer_capacity,
            consolidation_ewc_lambda=self.consolidation_ewc_lambda,
            consolidation_lr=self.consolidation_lr,
            weight_lr=self.core_weight_lr,
            weight_decay=self.core_weight_decay,
        )

    @classmethod
    def micro_test(cls) -> PULSEv2Config:
        return cls(
            vocab_size=260,
            embed_dim=32,
            d_model=64,
            n_heads=4,
            d_ff=128,
            num_blocks=4,
            echoic_layers=(1, 3),
            delta_chunk_size=8,
            echoic_window_size=8,
            chunk_size=4,
            layer_dims=(48, 64, 64, 48),
            ssm_state_dim=64,
            critic_proj_dim=32,
            max_think_steps=12,
            train_think_steps=2,
            core_weight_lr=0.06,
        )

    @classmethod
    def colab_30m(cls) -> PULSEv2Config:
        """~29.3M-parameter PULSE-v2 matched closely (~1.006x) to ScalableTransformerLM.colab_30m()."""
        return cls(
            vocab_size=50257,
            embed_dim=192,
            d_model=512,
            n_heads=8,
            d_ff=896,
            num_blocks=6,
            echoic_layers=(2, 5),
            delta_chunk_size=32,
            echoic_window_size=64,
            chunk_size=4,
            layer_dims=(384, 576, 704, 576, 384),
            ssm_state_dim=512,
            critic_proj_dim=256,
            max_think_steps=16,
            train_think_steps=3,
            core_weight_lr=0.05,
        )


class _ResonanceBlock(nn.Module):
    """Single O(1)-memory token resonance block: (GatedDeltaNet or EchoicWindow) + SwiGLU."""

    def __init__(self, config: PULSEv2Config, is_echoic: bool) -> None:
        super().__init__()
        self.is_echoic = is_echoic
        self.norm1 = RMSNorm(config.d_model)
        if is_echoic:
            self.mixer: nn.Module = SlidingWindowEchoicAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                window_size=config.echoic_window_size,
                conv_kernel=config.conv_kernel,
            )
        else:
            self.mixer = ChunkwiseGatedDeltaNet(
                d_model=config.d_model,
                n_heads=config.n_heads,
                chunk_size=config.delta_chunk_size,
                conv_kernel=config.conv_kernel,
            )
        self.norm2 = RMSNorm(config.d_model)
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        return self.mixer.init_state(batch_size, device=device, dtype=dtype)  # type: ignore[operator]

    def forward(
        self, x: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h, new_state = self.mixer(self.norm1(x), state=state)
        x = x + h
        n2 = self.norm2(x)
        x = x + self.w_down(F.silu(self.w_gate(n2)) * self.w_up(n2))
        return x, new_state

    def step(
        self, x_step: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h, new_state = self.mixer.step(self.norm1(x_step), state=state)  # type: ignore[operator]
        x = x_step + h
        n2 = self.norm2(x)
        x = x + self.w_down(F.silu(self.w_gate(n2)) * self.w_up(n2))
        return x, new_state


class PULSEv2Model(nn.Module):
    """PULSE-v2: Hyper-Plastic Resonance Engine combining O(1) Matrix Gated DeltaNet +
    O(1) Sliding Echoic Cortex + 100% Zero-Backprop Bidirectional Predictive Coding Cortex +
    Wake-Sleep Contrastive Free-Energy Anti-Hallucination + Online W_fast & Sleep Consolidation."""

    def __init__(self, config: PULSEv2Config) -> None:
        super().__init__()
        self.config = config
        self.core_cfg = config.to_core_config()

        # --- Shared Factorized Tied Embeddings (100% matched to TransformerBaseline) ---
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.emb_in = nn.Linear(config.embed_dim, config.d_model, bias=False)

        # --- O(1)-Memory Token Resonance Stream (Lower + Upper Halves) ---
        echoic_set = set(config.echoic_layers)
        self.blocks = nn.ModuleList(
            [_ResonanceBlock(config, is_echoic=(i in echoic_set)) for i in range(config.num_blocks)]
        )
        self.norm_f = RMSNorm(config.d_model)
        self.emb_out = nn.Linear(config.d_model, config.embed_dim, bias=False)

        # --- Concept Manifold Bridge (Mid-Stream <-> Zero-Backprop Predictive Cortex) ---
        self.mid_to_concept = nn.Sequential(
            nn.Linear(config.d_model, config.concept_dim),
            nn.Tanh(),
            nn.Linear(config.concept_dim, config.concept_dim),
        )
        self.concept_norm = nn.LayerNorm(config.concept_dim, elementwise_affine=False)
        self.manifold = ConceptManifoldProjection(config.concept_dim)
        self.bos_concept_prior = nn.Parameter(torch.zeros(1, 1, config.concept_dim))

        # Top-down gated feedback bridge (zero-initialized -> smooth progressive coupling!)
        self.concept_to_stream = nn.Linear(config.concept_dim + config.top_dim, config.d_model, bias=False)
        self.top_down_norm = RMSNorm(config.d_model)
        self.top_down_gate = nn.Parameter(torch.zeros(1))

        # --- 100% ZERO-BACKPROP Bidirectional Predictive Coding Core ---
        self.field = BidirectionalPredictiveField(self.core_cfg)
        self.causal_layer = PredictiveLayer(
            upper_dim=config.top_dim,
            lower_dim=config.concept_dim,
            precision_min=config.precision_min,
        )
        self.contrastive_critic = ContrastiveEnergyCritic(
            top_dim=config.top_dim,
            concept_dim=config.concept_dim,
            proj_dim=config.critic_proj_dim,
        )
        for mod in (self.field, self.causal_layer, self.contrastive_critic):
            for p in mod.parameters():
                p.requires_grad_(False)

        # --- Pillar 3: Tri-Timescale Plastic Memory ---
        self.fast_ssm = ParallelSelectiveSSM(
            input_dim=config.top_dim,
            state_dim=config.ssm_state_dim,
        )
        self.fast_weights = SurpriseGatedFastWeights(self.core_cfg, max_prior_centroids=64)
        self._causal_episodes: List[EpisodicTrace] = []

        # --- Pillar 4: Backprop-Free Local Optimizers ---
        self.local_opt = LocalSynapticOptimizer(self.field, self.core_cfg)
        self._causal_adam_w: Optional[_LocalAdamState] = None
        self._causal_adam_b: Optional[_LocalAdamState] = None
        self._init_stream_weights()

    def _init_stream_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0 / math.sqrt(self.config.embed_dim))
        nn.init.normal_(self.emb_in.weight, mean=0.0, std=1.0 / math.sqrt(self.config.embed_dim))
        nn.init.normal_(self.emb_out.weight, mean=0.0, std=1.0 / math.sqrt(self.config.d_model))
        for blk in self.blocks:
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.concept_to_stream.weight, mean=0.0, std=0.02)

    def _ensure_causal_adam(self) -> None:
        if self._causal_adam_w is None:
            w = self.causal_layer.w_slow
            b = self.causal_layer.b_slow
            self._causal_adam_w = _LocalAdamState(w.shape, w.device, w.dtype)
            self._causal_adam_b = _LocalAdamState(b.shape, b.device, b.dtype)

    def io_parameters(self) -> List[nn.Parameter]:
        """Parameters outside the zero-backprop Predictive Coding & Contrastive Energy Core."""
        return [p for p in self.parameters() if p.requires_grad]

    def core_param_count(self) -> int:
        return sum(
            p.numel()
            for mod in (self.field, self.causal_layer, self.contrastive_critic)
            for p in mod.parameters()
        )

    def core_autograd_grad_count(self) -> int:
        return sum(
            1
            for mod in (self.field, self.causal_layer, self.contrastive_critic)
            for p in mod.parameters()
            if p.grad is not None
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def working_state_bytes(state: Dict[str, Any]) -> int:
        total = 0
        for val in state.values():
            if isinstance(val, torch.Tensor):
                total += val.nelement() * val.element_size()
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        total += sum(t.nelement() * t.element_size() for t in item.values())
                    elif isinstance(item, torch.Tensor):
                        total += item.nelement() * item.element_size()
        return total

    def _pool_mid_to_concepts(self, h_mid: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = h_mid.shape
        k = self.config.chunk_size
        n_chunks = seq_len // k
        if n_chunks == 0:
            return h_mid.new_zeros((bsz, 0, self.config.concept_dim))
        pooled = h_mid[:, : n_chunks * k, :].view(bsz, n_chunks, k, dim).mean(dim=2)
        return self.manifold(self.concept_norm(self.mid_to_concept(pooled)))

    @torch.no_grad()
    def _relax_concept_frames(
        self,
        z_frames: torch.Tensor,
        max_steps: int,
        use_fast_weights: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        bsz, n_chunks, d0 = z_frames.shape
        if n_chunks == 0:
            empty_top = z_frames.new_zeros((bsz, 0, self.config.top_dim))
            empty_e = z_frames.new_zeros((bsz, 0))
            return empty_top, empty_e, empty_e, z_frames.new_zeros((bsz, 0), dtype=torch.long), []

        with torch.amp.autocast(device_type=z_frames.device.type, enabled=False):
            z_flat = z_frames.detach().float().reshape(bsz * n_chunks, d0)
            init_states = self.field.initialize_states(z_flat)
            init_energy, _ = self.field.compute_energy_and_errors(
                init_states, use_fast_weights=use_fast_weights
            )

            threshold = self.config.energy_threshold
            surprising = torch.nonzero(init_energy > threshold, as_tuple=False).squeeze(-1)

            states_out = [s.clone() for s in init_states]
            final_energy = init_energy.clone()
            steps_taken = torch.zeros_like(init_energy, dtype=torch.long)

            if surprising.numel() > 0 and max_steps > 0:
                sub_init = [s.index_select(0, surprising) for s in init_states]
                sub_relax = self.field.relax(
                    z_bottom=sub_init[0],
                    init_states=sub_init,
                    clamp_bottom=True,
                    max_steps=max_steps,
                    energy_threshold=threshold,
                    use_fast_weights=use_fast_weights,
                )
                for l_idx in range(len(states_out)):
                    states_out[l_idx].index_copy_(0, surprising, sub_relax.states[l_idx])
                final_energy.index_copy_(0, surprising, sub_relax.final_energy)
                steps_taken.index_copy_(0, surprising, sub_relax.steps_taken)

            z_top_seq = states_out[-1].to(z_frames.dtype).view(bsz, n_chunks, self.config.top_dim)
            return (
                z_top_seq,
                init_energy.view(bsz, n_chunks),
                final_energy.view(bsz, n_chunks),
                steps_taken.view(bsz, n_chunks),
                states_out,
            )

    def _run_lower_stream(
        self,
        input_ids: torch.Tensor,
        block_states: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        x = self.emb_in(self.tok_emb(input_ids))
        mid = self.config.mid_block_index
        new_states: List[Dict[str, torch.Tensor]] = []
        for idx in range(mid):
            st = block_states[idx] if block_states is not None else None
            x, nst = self.blocks[idx](x, state=st)
            new_states.append(nst)
        return x, new_states

    def forward(
        self,
        input_ids: torch.Tensor,
        working_state: Optional[Dict[str, Any]] = None,
        use_fast_weights: bool = True,
        max_think_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Strictly causal O(N)-time, O(1)-working-memory forward pass."""
        bsz, seq_len = input_ids.shape
        k = self.config.chunk_size
        mid = self.config.mid_block_index
        steps_cap = max_think_steps if max_think_steps is not None else self.config.max_think_steps

        in_block_states = working_state["blocks"] if working_state is not None else None
        in_ssm_state = working_state["ssm"] if working_state is not None else None

        # 1. Lower Resonance Stream (DeltaNet + Echoic Window + ShortConv)
        h_mid, lower_states = self._run_lower_stream(input_ids, block_states=in_block_states)

        # 2. Mid-Stream Concept Pooling -> Zero-Backprop Predictive Coding Cortex
        z_chunks = self._pool_mid_to_concepts(h_mid)  # [B, floor(T/K), concept_dim]
        n_complete = z_chunks.shape[1]
        z_top_seq, init_e, final_e, steps_taken, settled_states = self._relax_concept_frames(
            z_chunks, max_steps=steps_cap, use_fast_weights=use_fast_weights
        )

        # 3. O(log N) Parallel Selective Concept SSM -> Strictly Causal Next-Chunk Forecast
        ssm_state, prior_next_seq, _ = self.fast_ssm.scan(z_top_seq, init_state=in_ssm_state)

        with torch.no_grad(), torch.amp.autocast(device_type=h_mid.device.type, enabled=False):
            if n_complete > 0:
                z_pred_next = self.causal_layer.predict_down(
                    prior_next_seq.detach().float(), use_fast_weights=use_fast_weights
                ).to(h_mid.dtype)
            else:
                z_pred_next = h_mid.new_zeros((bsz, 0, self.config.concept_dim))

        # 4. Strictly Causal Top-Down Semantic Feedback (chunk c tokens see ONLY chunks 0..c-1!)
        n_groups = math.ceil(seq_len / k)
        bos_c = self.bos_concept_prior.expand(bsz, 1, -1).to(h_mid.dtype)
        bos_t = h_mid.new_zeros((bsz, 1, self.config.top_dim))
        ctx_c = torch.cat([bos_c, z_pred_next.detach()], dim=1)[:, :n_groups, :]
        ctx_t = torch.cat([bos_t, prior_next_seq], dim=1)[:, :n_groups, :]
        if ctx_c.shape[1] < n_groups:
            pad_g = n_groups - ctx_c.shape[1]
            ctx_c = torch.cat([ctx_c, ctx_c[:, -1:, :].expand(bsz, pad_g, -1)], dim=1)
            ctx_t = torch.cat([ctx_t, ctx_t[:, -1:, :].expand(bsz, pad_g, -1)], dim=1)

        ctx_tok = torch.cat([ctx_c, ctx_t], dim=-1).repeat_interleave(k, dim=1)[:, :seq_len, :]
        top_down_mod = self.top_down_norm(self.concept_to_stream(ctx_tok))
        x = h_mid + torch.tanh(self.top_down_gate) * top_down_mod

        # 5. Upper Resonance Stream -> Tied Output Logits
        upper_states: List[Dict[str, torch.Tensor]] = []
        for idx in range(mid, self.config.num_blocks):
            st = in_block_states[idx] if in_block_states is not None else None
            x, nst = self.blocks[idx](x, state=st)
            upper_states.append(nst)

        h_emb = self.emb_out(self.norm_f(x))
        logits = F.linear(h_emb, self.tok_emb.weight)

        # Latest strictly-O(1) concept context for streaming continuation
        last_ctx_c = (z_pred_next[:, -1:, :] if n_complete > 0 else bos_c).detach().clone()
        last_ctx_t = (prior_next_seq[:, -1:, :] if n_complete > 0 else bos_t).detach().clone()

        new_working_state: Dict[str, Any] = {
            "blocks": lower_states + upper_states,
            "ssm": ssm_state.detach(),
            "last_ctx_c": last_ctx_c,
            "last_ctx_t": last_ctx_t,
        }
        return {
            "logits": logits,
            "working_state": new_working_state,
            "z_chunks": z_chunks,
            "z_top_seq": z_top_seq,
            "prior_next_seq": prior_next_seq,
            "z_pred_next": z_pred_next,
            "initial_energy": init_e,
            "final_energy": final_e,
            "steps_taken": steps_taken,
            "settled_states": settled_states,
        }

    def train_step_hybrid_local(
        self,
        input_ids: torch.Tensor,
        io_optimizer: torch.optim.Optimizer,
        async_core: bool = True,
        max_grad_norm: float = 1.0,
        use_amp_bf16: bool = False,
        core_lr_mult: float = 1.0,
    ) -> Dict[str, float]:
        """Hybrid-Local training step with cosine-scaled zero-backprop core & Wake-Sleep critic."""
        self.train()
        self._ensure_causal_adam()
        io_optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp_bf16 and input_ids.is_cuda
        x_in = input_ids[:, :-1]
        y_tgt = input_ids[:, 1:]

        with torch.amp.autocast(
            device_type="cuda" if input_ids.is_cuda else "cpu",
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            out = self.forward(
                x_in,
                use_fast_weights=False,
                max_think_steps=self.config.train_think_steps,
            )
            logits = out["logits"]
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_tgt.reshape(-1))

            # Pool ALL complete chunks of full input_ids for strictly causal JEPA target alignment
            with torch.no_grad():
                h_mid_full, _ = self._run_lower_stream(input_ids)
                z_all = self._pool_mid_to_concepts(h_mid_full).detach()

            z_chunks = out["z_chunks"]
            anti_collapse = (
                self.manifold.anti_collapse_energy(z_chunks.float())
                if z_chunks.shape[1] >= 1
                else ce_loss.new_tensor(0.0)
            )
            n_pairs = min(out["prior_next_seq"].shape[1], max(z_all.shape[1] - 1, 0))
            if n_pairs >= 1:
                z_target_chunks = z_all[:, 1 : n_pairs + 1, :]
                z_top_targets, _, _, _, _ = self._relax_concept_frames(
                    z_target_chunks,
                    max_steps=self.config.train_think_steps,
                    use_fast_weights=False,
                )
                ssm_pred_loss = F.mse_loss(out["prior_next_seq"][:, :n_pairs, :], z_top_targets.detach())
            else:
                ssm_pred_loss = ce_loss.new_tensor(0.0)

            stream_loss = (
                ce_loss
                + self.config.anti_collapse_weight * anti_collapse
                + self.config.ssm_align_weight * ssm_pred_loss
            )

        # Backprop ONLY touches stream/I/O params (core has requires_grad=False!)
        stream_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.io_parameters(), max_grad_norm)
        io_optimizer.step()

        # --- Pillar 4: 100% Backprop-Free Local Energy & Wake-Sleep Contrastive Learning ---
        core_lr = self.config.core_weight_lr * max(float(core_lr_mult), 0.05)
        with torch.no_grad():
            settled = out["settled_states"]
            core_mse = 0.0
            if settled:
                step_stats = self.local_opt.step_field(
                    self.field,
                    settled,
                    lr=core_lr,
                    async_order=async_core,
                    use_fast_weights=False,
                )
                core_mse = step_stats["mean_layer_mse"]
                self.fast_weights.observe_latent_distribution(settled[-1])

            causal_mse = 0.0
            contrastive_loss = 0.0
            if n_pairs >= 1:
                u_prev = out["prior_next_seq"][:, :n_pairs, :].detach().float().reshape(-1, self.config.top_dim)
                z_next = z_all[:, 1 : n_pairs + 1, :].detach().float().reshape(-1, self.config.concept_dim)

                grads = self.causal_layer.local_weight_gradients(
                    z_lower=z_next, z_upper=u_prev, use_fast_weights=False
                )
                assert self._causal_adam_w is not None and self._causal_adam_b is not None
                self._causal_adam_w.apply(
                    self.causal_layer.w_slow,
                    grads["dE_dw_slow"],
                    lr=core_lr,
                    weight_decay=self.config.core_weight_decay,
                )
                self._causal_adam_b.apply(self.causal_layer.b_slow, grads["dE_db_slow"], lr=core_lr)
                grad_sq = grads["dE_dw_slow"].pow(2)
                self.causal_layer.importance_w.mul_(0.95).add_(
                    0.05 * (grad_sq / grad_sq.mean().clamp_min(1e-8))
                )
                causal_mse = grads["eps"].pow(2).mean().item()
                self.fast_weights.observe_latent_distribution(u_prev)

                # Wake-Sleep Contrastive InfoNCE local plasticity for structural anti-hallucination!
                c_stats = self.contrastive_critic.update_wake_sleep_local(
                    u_prev, z_next, lr=core_lr
                )
                contrastive_loss = c_stats["contrastive_loss"]

        return {
            "ce_loss": float(ce_loss.item()),
            "ppl": float(math.exp(min(ce_loss.item(), 20.0))),
            "anti_collapse": float(anti_collapse.item()),
            "core_mse": float(core_mse),
            "causal_mse": float(causal_mse),
            "contrastive_loss": float(contrastive_loss),
            "mean_steps": float(out["steps_taken"].float().mean().item())
            if out["steps_taken"].numel() > 0
            else 0.0,
            "mean_energy": float(out["final_energy"].float().mean().item())
            if out["final_energy"].numel() > 0
            else 0.0,
        }

    @torch.no_grad()
    def encode_target_concept(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Encode a continuation chunk into the continuous concept manifold [B, concept_dim]."""
        h_mid, _ = self._run_lower_stream(target_ids)
        return self._pool_mid_to_concepts(h_mid)[:, 0, :].detach().float()

    @torch.no_grad()
    def _cue_to_causal_key(self, cue_ids: torch.Tensor, use_fast_weights: bool = True) -> torch.Tensor:
        h_mid, _ = self._run_lower_stream(cue_ids)
        z_chunks = self._pool_mid_to_concepts(h_mid)
        z_top_seq, _, _, _, _ = self._relax_concept_frames(
            z_chunks, max_steps=self.config.max_think_steps, use_fast_weights=use_fast_weights
        )
        _, prior_seq, _ = self.fast_ssm.scan(z_top_seq)
        return prior_seq[:, -1, :].float()

    @torch.no_grad()
    def predict_next_concept(self, cue_ids: torch.Tensor, use_fast_weights: bool = True) -> torch.Tensor:
        key = self._cue_to_causal_key(cue_ids, use_fast_weights=use_fast_weights)
        return self.causal_layer.predict_down(key, use_fast_weights=use_fast_weights)

    @torch.no_grad()
    def teach_fact_online(
        self,
        cue_ids: torch.Tensor,
        target_ids: torch.Tensor,
        repeats: int = 5,
    ) -> Dict[str, float]:
        """Inference-time online rule/fact learning directly into W_fast (0 backprop, 0 context!)."""
        key = self._cue_to_causal_key(cue_ids, use_fast_weights=True)
        z_target = self.encode_target_concept(target_ids)

        alpha = self.config.fast_weight_alpha
        keep = 1.0 - self.config.fast_weight_decay
        max_norm = max(self.config.fast_weight_max_norm, math.sqrt(self.config.concept_dim) * 1.6)
        rank = self.config.fast_weight_rank

        peak_gate = 0.0
        peak_surprise = 0.0
        total_drop = 0.0
        bsz = z_target.shape[0]

        for _ in range(max(repeats, 1)):
            e_before = self.causal_layer.energy(z_target, key, use_fast_weights=True)
            self.causal_layer.w_fast.mul_(keep)

            for b in range(bsz):
                zt_b = z_target[b : b + 1]
                k_b = key[b : b + 1]
                phi_u, _ = _activation_and_derivative(k_b, self.causal_layer.activation)
                eps = zt_b - self.causal_layer.predict_down(k_b, use_fast_weights=True)
                rms_surp, gate = self.fast_weights.surprise_gate(eps)

                peak_gate = max(peak_gate, float(gate.item()))
                peak_surprise = max(peak_surprise, float(rms_surp.item()))

                phi_norm_sq = phi_u.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
                v_norm = phi_u / phi_norm_sq
                self.causal_layer.w_fast.add_(alpha * ((gate.unsqueeze(-1) * eps).T @ v_norm))

            if rank is not None:
                self.causal_layer.w_fast.copy_(
                    SurpriseGatedFastWeights._project_low_rank(self.causal_layer.w_fast, rank)
                )
            cur_norm = torch.linalg.matrix_norm(self.causal_layer.w_fast, ord="fro")
            if cur_norm > max_norm:
                self.causal_layer.w_fast.mul_(max_norm / cur_norm.clamp_min(1e-8))

            e_after = self.causal_layer.energy(z_target, key, use_fast_weights=True)
            drop = (e_before - e_after).clamp_min(0.0)
            total_drop += float(drop.mean().item())

            if peak_gate > 0.35:
                for b in range(bsz):
                    dval = float(drop[b].item())
                    if dval <= 1e-6:
                        continue
                    self._causal_episodes.append(
                        EpisodicTrace(
                            z_bottom=z_target[b].detach().cpu().clone(),
                            z_top=key[b].detach().cpu().clone(),
                            energy_drop=dval,
                        )
                    )
                if len(self._causal_episodes) > self.config.replay_buffer_capacity:
                    self._causal_episodes.sort(key=lambda e: e.salience, reverse=True)
                    del self._causal_episodes[self.config.replay_buffer_capacity :]

        return {
            "max_gate": peak_gate,
            "max_surprise": peak_surprise,
            "total_energy_drop": total_drop,
            "buffered_episodes": float(len(self._causal_episodes)),
        }

    @torch.no_grad()
    def sleep_consolidate(
        self,
        top_k_episodes: int = 32,
        num_replay_samples: int = 40,
        num_epochs: int = 35,
        clear_fast_weights: bool = True,
    ) -> Dict[str, float]:
        """Offline Sleep Consolidation: O(1) GMM Generative Replay + EWC (W_fast -> W_slow)."""
        if not self._causal_episodes:
            return {"causal_consolidated": 0.0, "new_mse": 0.0, "replay_mse": 0.0}

        device = self.causal_layer.w_slow.device
        dtype = self.causal_layer.w_slow.dtype
        selected = sorted(self._causal_episodes, key=lambda e: e.salience, reverse=True)[:top_k_episodes]

        z_target_new = torch.stack([e.z_bottom for e in selected], dim=0).to(device=device, dtype=dtype)
        z_key_new = torch.stack([e.z_top for e in selected], dim=0).to(device=device, dtype=dtype)

        z_key_rep_raw = self.fast_weights.sample_prior_latents(
            num_samples=num_replay_samples,
            dim=self.config.top_dim,
            device=device,
            dtype=dtype,
        )
        sims = F.cosine_similarity(z_key_rep_raw.unsqueeze(1), z_key_new.unsqueeze(0), dim=-1).max(dim=-1).values
        keep_mask = sims < 0.82
        z_key_rep = z_key_rep_raw[keep_mask] if keep_mask.any() else z_key_rep_raw[:0]
        z_tgt_rep = (
            self.causal_layer.predict_down(z_key_rep, use_fast_weights=False)
            if z_key_rep.shape[0] > 0
            else z_target_new[:0]
        )

        anchor_w = self.causal_layer.w_slow.detach().clone()
        anchor_b = self.causal_layer.b_slow.detach().clone()
        lr = self.config.consolidation_lr
        ewc_lambda = self.config.consolidation_ewc_lambda
        new_mse = 0.0
        rep_mse = 0.0

        for _ in range(num_epochs):
            phi_new, _ = _activation_and_derivative(z_key_new, self.causal_layer.activation)
            for b in range(z_key_new.shape[0]):
                pn = phi_new[b : b + 1]
                err_b = z_target_new[b : b + 1] - F.linear(pn, self.causal_layer.w_slow, self.causal_layer.b_slow)
                dn = pn.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
                self.causal_layer.w_slow.add_(lr * ((err_b / dn).T @ pn))

            if z_key_rep.shape[0] > 0:
                phi_rep, _ = _activation_and_derivative(z_key_rep, self.causal_layer.activation)
                for b in range(z_key_rep.shape[0]):
                    pr = phi_rep[b : b + 1]
                    err_r = z_tgt_rep[b : b + 1] - F.linear(pr, self.causal_layer.w_slow, self.causal_layer.b_slow)
                    dr = pr.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
                    self.causal_layer.w_slow.add_(0.5 * lr * ((err_r / dr).T @ pr))
                rep_mse = (
                    z_tgt_rep - F.linear(phi_rep, self.causal_layer.w_slow, self.causal_layer.b_slow)
                ).pow(2).mean().item()

            new_mse = (
                z_target_new - F.linear(phi_new, self.causal_layer.w_slow, self.causal_layer.b_slow)
            ).pow(2).mean().item()
            ewc_w = ewc_lambda * self.causal_layer.importance_w * (self.causal_layer.w_slow - anchor_w)
            self.causal_layer.w_slow.add_(-0.05 * lr * ewc_w)
            self.causal_layer.b_slow.lerp_(anchor_b, 0.1)

        self.fast_weights.observe_latent_distribution(z_key_new)
        if clear_fast_weights:
            self.causal_layer.w_fast.zero_()
            for layer in self.field.layers:
                layer.w_fast.zero_()
        self._causal_episodes.clear()
        return {
            "causal_consolidated": float(len(selected)),
            "new_mse": new_mse,
            "replay_mse": rep_mse,
        }

    @torch.no_grad()
    def continuation_free_energy(
        self,
        prompt_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
        use_fast_weights: bool = True,
    ) -> torch.Tensor:
        """Combined Predictive + Wake-Sleep Contrastive Free Energy for structural anti-hallucination."""
        key = self._cue_to_causal_key(prompt_ids, use_fast_weights=use_fast_weights)
        # Contextualized continuation concept (evaluated right after prompt's lower stream!)
        full = torch.cat([prompt_ids, continuation_ids], dim=1)
        h_full, _ = self._run_lower_stream(full)
        h_cont = h_full[:, -continuation_ids.shape[1] :, :]
        z_cont = self._pool_mid_to_concepts(h_cont)[:, 0, :].float()

        e_pred = self.causal_layer.energy(z_cont, key, use_fast_weights=use_fast_weights)
        e_contrast = self.contrastive_critic.energy(key, z_cont)
        return 0.25 * e_pred + 0.75 * e_contrast

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 40,
        use_fast_weights: bool = True,
    ) -> torch.Tensor:
        """Strictly O(1)-working-memory streaming autoregressive generation."""
        self.eval()
        out = prompt_ids.clone()
        mid = self.config.mid_block_index
        k = self.config.chunk_size

        prefill = self.forward(out, use_fast_weights=use_fast_weights)
        logits = prefill["logits"][:, -1, :]
        st = prefill["working_state"]
        block_states: List[Dict[str, torch.Tensor]] = st["blocks"]
        ssm_state: torch.Tensor = st["ssm"]
        cur_ctx_c: torch.Tensor = st["last_ctx_c"]
        cur_ctx_t: torch.Tensor = st["last_ctx_t"]
        mid_buf: List[torch.Tensor] = []

        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k > 0 and top_k < scaled.shape[-1]:
                    vals, _ = torch.topk(scaled, top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < vals[:, -1:], float("-inf"))
                next_tok = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1)

            out = torch.cat([out, next_tok], dim=1)

            # Single-token O(1) Resonance Step!
            x = self.emb_in(self.tok_emb(next_tok))
            for idx in range(mid):
                x, block_states[idx] = self.blocks[idx].step(x, block_states[idx])
            mid_buf.append(x)

            ctx_vec = torch.cat([cur_ctx_c, cur_ctx_t], dim=-1)
            x = x + torch.tanh(self.top_down_gate) * self.top_down_norm(
                self.concept_to_stream(ctx_vec)
            )
            for idx in range(mid, self.config.num_blocks):
                x, block_states[idx] = self.blocks[idx].step(x, block_states[idx])
            logits = F.linear(self.emb_out(self.norm_f(x)), self.tok_emb.weight)[:, -1, :]

            # Every K tokens, run 1 O(1) Predictive Coding Cortex update!
            if len(mid_buf) == k:
                h_chunk = torch.cat(mid_buf, dim=1)
                mid_buf.clear()
                z_new = self._pool_mid_to_concepts(h_chunk)
                z_top_new, _, _, _, _ = self._relax_concept_frames(
                    z_new, max_steps=self.config.max_think_steps, use_fast_weights=use_fast_weights
                )
                ssm_state, cur_ctx_t = self.fast_ssm.step(z_top_new[:, 0, :], ssm_state)
                cur_ctx_t = cur_ctx_t.unsqueeze(1)
                cur_ctx_c = self.causal_layer.predict_down(
                    cur_ctx_t.float(), use_fast_weights=use_fast_weights
                ).to(x.dtype)

        return out
