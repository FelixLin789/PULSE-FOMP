"""PULSE v6: causal resonance backbone and a bounded, exact cue memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pulse.pulse_v4 import PULSEBackbone, PULSEBackboneConfig


@dataclass
class PULSEv6Config(PULSEBackboneConfig):
    # FineWeb-Edu 3-seed ablation: AttnRes hurt PPL (+0.045 nats) & speed (-16%);
    # zero beta_floor unlocks selective write/erase gating on noisy 4K/8K haystacks.
    use_attn_res: bool = False
    delta_beta_floor: float = 0.0
    memory_capacity: int = 128
    max_cue_tokens: int = 32
    memory_ridge: float = 1e-4

    def __post_init__(self) -> None:
        if self.memory_capacity < 0 or self.max_cue_tokens < 1 or self.memory_ridge <= 0:
            raise ValueError("Invalid cue memory capacity, cue length limit, or ridge.")
        if self.d_model % self.n_heads or self.d_model // self.n_heads % 2 or self.n_heads % self.n_kv_heads:
            raise ValueError("Use even head dimensions and an integral ratio of query to KV heads.")

    @classmethod
    def micro_test(cls) -> PULSEv6Config:
        return cls(vocab_size=260, embed_dim=32, d_model=64, n_heads=4, n_kv_heads=2,
                   n_layers=4, delta_layers=(0, 2), d_ff=128, delta_chunk_size=8, echoic_window_size=8)

    @classmethod
    def colab_30m(cls) -> PULSEv6Config:
        return cls()

    @classmethod
    def colab_100m(cls) -> PULSEv6Config:
        return cls(embed_dim=384, d_model=768, n_heads=12, n_kv_heads=4, n_layers=12,
                   delta_layers=(0, 3, 6, 9), d_ff=2192, delta_chunk_size=64)


class RLSCueMemory(nn.Module):
    """Exact suffix keys encode orthogonal one-hot features for diagonal RLS.

    Each column is a correction in the tied LM head's input space. An explicit
    write to an existing key replaces its latest target. New keys use FIFO
    replacement at capacity. All facts, weights and write cursors serialize.
    """

    def __init__(self, capacity: int, width: int, embed_dim: int, ridge: float) -> None:
        super().__init__()
        self.capacity = capacity
        self.width = width
        self.ridge = ridge
        self.register_buffer("cues", torch.full((capacity, width), -1, dtype=torch.long))
        self.register_buffer("lengths", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("precision", torch.full((capacity,), ridge))
        self.register_buffer("fast", torch.zeros(capacity, embed_dim))
        self.register_buffer("slow", torch.zeros_like(self.fast))
        self.register_buffer("pending", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("consolidated", torch.zeros_like(self.pending))
        self.register_buffer("used", torch.zeros((), dtype=torch.long))
        self.register_buffer("write_cursor", torch.zeros((), dtype=torch.long))
        self._size = 0

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._size = int(self.used.item())

    @torch.no_grad()
    def teach(self, cue: torch.Tensor, correction: torch.Tensor) -> None:
        length = cue.numel()
        exact = self.lengths[:self._size].eq(length)
        exact &= self.cues[:self._size, -length:].eq(cue).all(dim=-1)
        matches = exact.nonzero(as_tuple=True)[0]
        if matches.numel():
            slot = int(matches[0].item())
        else:
            slot = int(self.write_cursor.item())
            self.write_cursor.fill_((slot + 1) % self.capacity)
            self._size = min(self._size + 1, self.capacity)
            self.used.fill_(self._size)
            self.cues[slot].fill_(-1)
            self.cues[slot, -length:].copy_(cue)
            self.lengths[slot] = length
            self.fast[slot].zero_()
            self.slow[slot].zero_()
            self.consolidated[slot] = False

        # Sherman-Morrison for one active basis vector. Re-teaching resets its prior.
        self.precision[slot].fill_(self.ridge)
        gain = (self.precision[slot] + 1).reciprocal()
        residual = correction - self.slow[slot] - self.fast[slot]
        self.fast[slot].add_(residual * gain)
        self.precision[slot].add_(1)
        self.pending[slot] = True

    @torch.no_grad()
    def sleep(self) -> int:
        count = int(self.pending.sum().item())
        self.slow.add_(self.fast)
        self.fast.zero_()
        self.consolidated.logical_or_(self.pending)
        self.pending.zero_()
        self.precision.fill_(self.ridge)
        return count

    @torch.no_grad()
    def correction(self, tokens: torch.Tensor, count: int, use_fast_weights: bool) -> torch.Tensor | None:
        if self._size == 0:
            return None

        cues = self.cues[:self._size]
        lengths = self.lengths[:self._size]
        # Filter by the final token before checking full cues.
        eligible = tokens[:, -count:, None].eq(cues[:, -1])
        if not use_fast_weights:
            eligible &= self.consolidated[:self._size]
        batch, position, slot = eligible.nonzero(as_tuple=True)
        if slot.numel() == 0:
            return None

        windows = F.pad(tokens, (self.width - 1, 0), value=-1).unfold(1, self.width, 1)
        keys = cues[slot]
        exact = (windows[batch, position + tokens.size(1) - count].eq(keys) | keys.eq(-1)).all(dim=-1)
        batch, position, slot = batch[exact], position[exact], slot[exact]
        if slot.numel() == 0:
            return None

        # The longest matching suffix wins if multiple taught cues end here.
        base = self.capacity + 1
        priority = lengths[slot] * base + slot + 1
        winners = torch.zeros(tokens.size(0) * count, device=tokens.device, dtype=torch.long)
        winners.scatter_reduce_(0, batch * count + position, priority, reduce="amax", include_self=True)
        slots = winners.remainder(base) - 1
        missing = slots.lt(0)
        indices = slots.clamp_min(0)
        delta = self.slow[indices]
        if use_fast_weights:
            delta = delta + self.fast[indices]
        return delta.masked_fill(missing[:, None], 0).view(tokens.size(0), count, -1)


class PULSEv6Model(PULSEBackbone):
    """Autoregressive model with optional online edits and sleep consolidation."""

    def __init__(self, config: PULSEv6Config) -> None:
        super().__init__(config)
        self.config: PULSEv6Config
        self.memory = (
            RLSCueMemory(config.memory_capacity, config.max_cue_tokens, config.embed_dim, config.memory_ridge)
            if config.memory_capacity
            else None
        )
        self._init_io_weights()

    def load_v53_backbone(self, state_dict) -> None:
        """Load compatible language weights from v5.3. Legacy latent cue keys need re-teaching."""
        weights = self.state_dict()
        backbone_keys = [name for name in weights if not name.startswith("memory.")]
        missing = [name for name in backbone_keys if name not in state_dict]
        if missing:
            raise ValueError(f"Missing compatible backbone weights: {missing}")
        weights.update((name, state_dict[name]) for name in backbone_keys)
        self.load_state_dict(weights, strict=True)

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        _, hidden, _, _, _ = self._run_backbone(tokens[:, :-1], return_state=False)
        loss_fn = self._compiled_ce if self._compiled_ce is not None else self._ce_loss_impl
        return loss_fn(hidden, self.tok_emb.weight, tokens[:, 1:])

    def query_logits(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Use the plain language head at selected positions for associative recall."""
        _, hidden, _, _, _ = self._run_backbone(tokens, return_state=False)
        rows = torch.arange(tokens.size(0), device=tokens.device)[:, None]
        return self._logits_impl(hidden[rows, positions], self.tok_emb.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        working_state=None,
        *,
        return_state: bool = True,
        use_memory: bool = True,
        use_fast_weights: bool = True,
        last_logits_only: bool = False,
    ) -> dict:
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token per sequence.")
        block_states = working_state["blocks"] if working_state is not None else None
        _, hidden, _, blocks, _ = self._run_backbone(input_ids, block_states=block_states, return_state=return_state)
        if last_logits_only:
            hidden = hidden[:, -1:, :]

        tokens = input_ids
        state = {"blocks": blocks} if return_state else None
        if self.memory is not None:
            past = working_state.get("cue_tail") if working_state is not None else None
            if past is not None:
                tokens = torch.cat((past, input_ids), dim=1)
            if state is not None:
                size = min(self.config.max_cue_tokens - 1, tokens.size(1))
                state["cue_tail"] = tokens[:, tokens.size(1) - size:].detach().clone(
                    memory_format=torch.contiguous_format
                )
            if use_memory:
                correction = self.memory.correction(tokens, hidden.size(1), use_fast_weights)
                if correction is not None:
                    hidden = hidden + correction.to(hidden.dtype)

        score = self._compiled_logits if self._compiled_logits is not None and hidden.size(1) > 1 else self._logits_impl
        return {"logits": score(hidden, self.tok_emb.weight), "working_state": state}

    @torch.no_grad()
    def teach_fact_online(self, cue_ids: torch.Tensor, target_ids: torch.Tensor) -> dict:
        """Write a cue and its next-token target, reporting achieved greedy recall."""
        if self.memory is None:
            raise ValueError("This model has no cue memory.")
        if cue_ids.ndim != 2 or not 0 < cue_ids.size(1) <= self.config.max_cue_tokens:
            raise ValueError(f"Use 1 to {self.config.max_cue_tokens} tokens per cue.")
        if target_ids.shape not in ((cue_ids.size(0),), (cue_ids.size(0), 1)):
            raise ValueError("Provide one next-token target per cue.")
        targets = target_ids.reshape(-1)
        if bool(((targets < 0) | (targets >= self.config.vocab_size)).any()):
            raise ValueError("The target token is outside the language vocabulary.")
        with torch.autocast(cue_ids.device.type, enabled=False):
            _, hidden, _, _, _ = self._run_backbone(cue_ids, return_state=False)
            weight = self.tok_emb.weight.float()
            logits = F.linear(hidden[:, -1].float(), weight)
            for cue, target, base in zip(cue_ids, targets, logits, strict=True):
                # Project onto violated target-versus-competitor margin constraints.
                # A target embedding alone can also increase a competitor's score.
                correction = torch.zeros_like(weight[target])
                for _ in range(256):
                    revised = base + F.linear(correction, weight)
                    competitors = revised.clone()
                    competitors[target] = -torch.inf
                    rival = competitors.argmax()
                    deficit = revised[rival] - revised[target] + 4.
                    if float(deficit) <= 0:
                        break
                    direction = weight[target] - weight[rival]
                    norm = direction.square().sum()
                    if float(norm) < 1e-12:
                        break
                    correction.add_(direction * ((deficit + .5) / norm))
                # Compensate for the regularized diagonal RLS gain.
                self.memory.teach(cue, correction * (1. + self.config.memory_ridge))
            actual = self(cue_ids, return_state=False, last_logits_only=True)["logits"][:, 0]
        return {"stored_keys": self.memory._size, "recall": actual.argmax(-1).eq(targets).float().mean().item()}

    @torch.no_grad()
    def sleep_consolidate(self) -> dict:
        return {"consolidated_facts": self.memory.sleep() if self.memory is not None else 0}

    @torch.no_grad()
    def generate(
        self, prompt_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = .8,
        top_k: int = 40, use_memory: bool = True, use_fast_weights: bool = True,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return prompt_ids.clone()
        self.eval()
        start = prompt_ids.size(1)
        output = F.pad(prompt_ids, (0, max_new_tokens))
        current = prompt_ids
        state = None
        for i in range(max_new_tokens):
            out = self(current, state, last_logits_only=True, use_memory=use_memory, use_fast_weights=use_fast_weights)
            logits = out["logits"][:, 0, :]
            state = out["working_state"]
            if temperature <= 0:
                current = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if 0 < top_k < logits.size(-1):
                    threshold = logits.topk(top_k).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                current = torch.multinomial(logits.softmax(-1), 1)
            output[:, start + i:start + i + 1] = current
        return output
