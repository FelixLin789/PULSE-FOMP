"""Standard Causal Decoder-Only Transformer Baseline with KV Cache for fair benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for a standard causal Transformer baseline."""

    vocab_size: int = 260
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    max_seq_len: int = 4096
    output_dim: int | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.output_dim is None:
            self.output_dim = self.vocab_size


KVCacheLayer = Tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Standard O(N^2) Multi-Head Causal Self-Attention with linear-growth KV Cache."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCacheLayer] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCacheLayer]]:
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        total_len = k.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T_q, T_k] -> O(N^2)

        # Explicit causal mask when query length > 1
        if seq_len > 1:
            q_pos = torch.arange(total_len - seq_len, total_len, device=x.device).unsqueeze(1)
            k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)
            causal_mask = k_pos > q_pos
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out), present_kv


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCacheLayer] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCacheLayer]]:
        attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present_kv


class TransformerBaseline(nn.Module):
    """Standard fixed-depth causal Transformer trained via global backpropagation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.output_dim or config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        past_kvs: Optional[List[KVCacheLayer]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[KVCacheLayer]]]:
        bsz, seq_len = tokens.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None and len(past_kvs) > 0 else 0
        positions = torch.arange(past_len, past_len + seq_len, device=tokens.device)
        positions = positions.clamp_max(self.config.max_seq_len - 1)

        x = self.tok_emb(tokens) + self.pos_emb(positions).unsqueeze(0)
        new_kvs: List[KVCacheLayer] = []
        for idx, block in enumerate(self.blocks):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            x, present_kv = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache and present_kv is not None:
                new_kvs.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, (new_kvs if use_cache else None)

    def classify_last(self, tokens: torch.Tensor) -> torch.Tensor:
        """Standard fixed-depth classification readout from the final token representation."""
        logits, _ = self.forward(tokens, use_cache=False)
        return logits[:, -1, :]

    @staticmethod
    def kv_cache_bytes(kv_cache: Optional[List[KVCacheLayer]]) -> int:
        """Compute exact memory footprint in bytes of an active KV cache."""
        if not kv_cache:
            return 0
        total = 0
        for k, v in kv_cache:
            total += k.nelement() * k.element_size() + v.nelement() * v.element_size()
        return total

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
