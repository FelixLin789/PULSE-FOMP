"""Modern Llama-style Scalable Causal Transformer Baseline (RoPE, RMSNorm, SwiGLU, KV Cache).

Used for fair, matched-parameter (~30M) head-to-head scientific experiments against PULSE-LM.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ScalableTransformerConfig:
    """Configuration for a modern RoPE + RMSNorm + SwiGLU causal Transformer LM."""

    vocab_size: int = 50257
    embed_dim: int = 192
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1408
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

    @classmethod
    def micro_test(cls) -> ScalableTransformerConfig:
        return cls(
            vocab_size=260,
            embed_dim=32,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=160,
            max_seq_len=512,
        )

    @classmethod
    def colab_30m(cls) -> ScalableTransformerConfig:
        """~29.2M-parameter modern Transformer matched closely to PULSELMConfig.colab_30m()."""
        return cls(
            vocab_size=50257,
            embed_dim=192,
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=1408,
            max_seq_len=8192,
        )

    @classmethod
    def colab_100m(cls) -> ScalableTransformerConfig:
        """~101M-parameter modern Transformer matched to PULSEv3Config.colab_100m()."""
        return cls(
            vocab_size=50257,
            embed_dim=384,
            d_model=768,
            n_heads=12,
            n_layers=12,
            d_ff=2048,
            max_seq_len=8192,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D_head], cos/sin: [1, 1, T, D_head // 2]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


KVCacheEntry = Tuple[torch.Tensor, torch.Tensor]


class RoPECausalAttention(nn.Module):
    def __init__(self, config: ScalableTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[KVCacheEntry] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCacheEntry]]:
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        if past_kv is None and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if seq_len > 1:
                total_len = k.shape[2]
                q_pos = torch.arange(total_len - seq_len, total_len, device=x.device).unsqueeze(1)
                k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)
                scores = scores.masked_fill((k_pos > q_pos).unsqueeze(0).unsqueeze(0), float("-inf"))
            out = torch.matmul(F.softmax(scores, dim=-1), v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out), present_kv


class SwiGLUBlock(nn.Module):
    def __init__(self, config: ScalableTransformerConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = RoPECausalAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self._compiled_nocache: Optional[object] = None

    def _forward_nocache(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), cos, sin, past_kv=None, use_cache=False)
        x = x + h
        n2 = self.norm2(x)
        return x + self.w_down(F.silu(self.w_gate(n2)) * self.w_up(n2))

    def compile_block(self) -> None:
        if not torch.cuda.is_available() or self._compiled_nocache is not None:
            return
        try:
            self._compiled_nocache = torch.compile(self._forward_nocache, dynamic=True)
        except Exception:
            self._compiled_nocache = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[KVCacheEntry] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCacheEntry]]:
        if past_kv is None and not use_cache and self._compiled_nocache is not None:
            return self._compiled_nocache(x, cos, sin), None  # type: ignore[operator]
        h, present_kv = self.attn(self.norm1(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        x = x + h
        n2 = self.norm2(x)
        x = x + self.w_down(F.silu(self.w_gate(n2)) * self.w_up(n2))
        return x, present_kv


class ScalableTransformerLM(nn.Module):
    """Modern RoPE + SwiGLU causal Transformer LM with factorized tied embeddings and KV Cache."""

    def __init__(self, config: ScalableTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads

        # Factorized tied embedding shared identically with PULSELanguageModel for fair comparison
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.emb_in = nn.Linear(config.embed_dim, config.d_model, bias=False)
        self.blocks = nn.ModuleList([SwiGLUBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.emb_out = nn.Linear(config.d_model, config.embed_dim, bias=False)

        self._init_weights()

    def compile_fast_paths(self) -> None:
        """Symmetrically enable torch.compile on Transformer blocks on CUDA for fair benchmarking."""
        if not torch.cuda.is_available():
            return
        for blk in self.blocks:
            blk.compile_block()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_cos_sin(
        self, start_pos: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )
        positions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        return (
            freqs.cos().to(dtype).view(1, 1, seq_len, half),
            freqs.sin().to(dtype).view(1, 1, seq_len, half),
        )

    def encode(
        self,
        tokens: torch.Tensor,
        past_kvs: Optional[List[KVCacheEntry]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[KVCacheEntry]]]:
        bsz, seq_len = tokens.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None and len(past_kvs) > 0 else 0

        x = self.emb_in(self.tok_emb(tokens))
        cos, sin = self._rope_cos_sin(past_len, seq_len, x.device, x.dtype)

        new_kvs: List[KVCacheEntry] = []
        for idx, block in enumerate(self.blocks):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            x, present_kv = block(x, cos, sin, past_kv=layer_past, use_cache=use_cache)
            if use_cache and present_kv is not None:
                new_kvs.append(present_kv)

        h_emb = self.emb_out(self.norm_f(x))
        return h_emb, (new_kvs if use_cache else None)

    def forward(
        self,
        tokens: torch.Tensor,
        past_kvs: Optional[List[KVCacheEntry]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[KVCacheEntry]]]:
        h_emb, cache = self.encode(tokens, past_kvs=past_kvs, use_cache=use_cache)
        return F.linear(h_emb, self.tok_emb.weight), cache

    def query_logits(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encode(tokens)
        rows = torch.arange(tokens.size(0), device=tokens.device)[:, None]
        return F.linear(hidden[rows, positions], self.tok_emb.weight)

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(input_ids[:, :-1], use_cache=False)
        targets = input_ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        out = prompt_ids.clone()
        logits, kv = self.forward(out, use_cache=True)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            if temperature <= 0.0:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if top_k > 0 and top_k < scaled.shape[-1]:
                    vals, _ = torch.topk(scaled, top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < vals[:, -1:], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_tok], dim=1)
            logits, kv = self.forward(next_tok, past_kvs=kv, use_cache=True)
        return out

    @staticmethod
    def kv_cache_bytes(kv_cache: Optional[List[KVCacheEntry]]) -> int:
        if not kv_cache:
            return 0
        total = 0
        for k, v in kv_cache:
            total += k.nelement() * k.element_size() + v.nelement() * v.element_size()
        return total

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
