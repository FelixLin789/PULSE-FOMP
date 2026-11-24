"""PULSE-v2 Core Innovations:
1. ShortCausalConv — O(1)-state depthwise causal convolution for crisp local n-gram syntax.
2. ChunkwiseGatedDeltaNet — Exact WY-representation Tensor-Core Gated Delta-Rule Matrix Fast Weights
   with strictly O(1) memory state S in R^{H x d_v x d_k} and O(N) compute.
3. SlidingWindowEchoicAttention — Strictly O(1) ring-buffer local echoic cortex (W=64) with
   exact O(N * W) chunked sliding-window execution (never materializes an O(N^2) matrix!).
4. ContrastiveEnergyCritic — 100% backprop-free (`requires_grad=False`, `@torch.no_grad()`)
   Wake-Sleep Positive/Negative Contrastive Free-Energy discriminator for structural anti-hallucination.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.local_learning import _LocalAdamState
from pulse.scalable_transformer import RMSNorm, _apply_rope


class ShortCausalConv(nn.Module):
    """Depthwise causal 1D convolution with strictly O(1) streaming ring state [B, D, K-1]."""

    def __init__(self, dim: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            padding=0,
            bias=True,
        )

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        d = device if device is not None else self.conv.weight.device
        dt = dtype if dtype is not None else self.conv.weight.dtype
        return torch.zeros(batch_size, self.dim, self.kernel_size - 1, device=d, dtype=dt)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel causal conv over [B, T, D] returning (y, final_state[B, D, K-1])."""
        bsz, seq_len, _ = x.shape
        x_t = x.transpose(1, 2)  # [B, D, T]
        pad_state = (
            state
            if state is not None
            else self.init_state(bsz, device=x.device, dtype=x.dtype)
        )
        x_cat = torch.cat([pad_state, x_t], dim=-1)  # [B, D, K-1 + T]
        y = F.silu(self.conv(x_cat)).transpose(1, 2)
        new_state = x_cat[:, :, x_cat.size(-1) - self.kernel_size + 1 :].detach().clone(
            memory_format=torch.contiguous_format
        )
        return y, new_state

    def step(
        self,
        x_step: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact O(1) streaming single-token step: x_step [B, 1, D] -> (y [B, 1, D], new_state)."""
        return self.forward(x_step, state=state)


class ChunkwiseGatedDeltaNet(nn.Module):
    """Matrix-Valued Fast-Weight Working Memory with Exact WY Chunkwise Parallelism.

    Recurrence per head h (exact Householder-like Delta Rule with continuous-time gating):
        S_t = alpha_t * S_{t-1} * (I - beta_t * k_t k_t^T) + beta_t * v_t k_t^T
        o_t = S_t @ q_t

    - State S_t in R^{B x H x d_v x d_k} is strictly O(1) constant memory regardless of N.
    - Keys k_t are L2-normalized and beta_t in (0, 2), ensuring contractive stability & exact
      associative overwriting/erasure (solving state tracking that Mamba/linear attention fail).
    - Forward pass uses exact Bischof-Van Loan WY block factorization in chunks of C tokens,
      turning sequential updates into pure Tensor-Core GEMMs (`O(N)` compute).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        chunk_size: int = 32,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        self.short_conv = ShortCausalConv(d_model, kernel_size=conv_kernel)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Input-dependent decay alpha_t in (0, 1) and delta-rule write rate beta_t in (0, 2)
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=True)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_gate = nn.Linear(d_model, d_model, bias=False)
        self.head_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Initialize alpha bias slightly positive -> gentle retention; beta bias moderate
        nn.init.constant_(self.alpha_proj.bias, 2.0)
        nn.init.constant_(self.beta_proj.bias, -1.0)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        d = device if device is not None else self.qkv_proj.weight.device
        dt = dtype if dtype is not None else self.qkv_proj.weight.dtype
        return {
            "S": torch.zeros(
                batch_size, self.n_heads, self.head_dim, self.head_dim, device=d, dtype=torch.float32
            ),
            "conv": self.short_conv.init_state(batch_size, device=d, dtype=dt),
        }

    @staticmethod
    def state_bytes(state: Dict[str, torch.Tensor]) -> int:
        return sum(t.nelement() * t.element_size() for t in state.values())

    def _project_inputs(
        self, x_conv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x_conv.shape
        q, k, v = self.qkv_proj(x_conv).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # L2-normalize keys & queries (ensures Householder factor I - beta k k^T has bounded norm <= 1!)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        q = F.normalize(q, p=2, dim=-1, eps=1e-6) * math.sqrt(self.head_dim)

        # log(alpha_t) <= 0 strictly, beta_t in (0, 2) (true Householder delta eraser)
        log_alpha = F.logsigmoid(self.alpha_proj(x_conv)).transpose(1, 2)  # [B, H, T]
        beta = 2.0 * torch.sigmoid(self.beta_proj(x_conv)).transpose(1, 2)  # [B, H, T]
        return q, k, v, log_alpha, beta

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Exact WY Chunkwise Parallel Forward over [B, T, D]."""
        bsz, seq_len, _ = x.shape
        st = state if state is not None else self.init_state(bsz, device=x.device, dtype=x.dtype)

        x_conv, new_conv = self.short_conv(x, state=st["conv"])
        q, k, v, log_alpha, beta = self._project_inputs(x_conv)

        c_sz = self.chunk_size
        rem = seq_len % c_sz
        pad_len = (c_sz - rem) if rem != 0 else 0

        # Run WY triangular solves & state accumulation strictly in FP32 (TF32 on L4/A100;
        # avoids unsupported BF16 solve_triangular on CUDA and preserves exact numerical match!)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            q_f = q.float()
            k_f = k.float()
            v_f = v.float()
            la_f = log_alpha.float()
            b_f = beta.float()

            if pad_len > 0:
                q_f = F.pad(q_f, (0, 0, 0, pad_len), value=0.0)
                k_f = F.pad(k_f, (0, 0, 0, pad_len), value=0.0)
                v_f = F.pad(v_f, (0, 0, 0, pad_len), value=0.0)
                # Identity padding: log_alpha=0 (alpha=1) & beta=0 leaves S_final bit-exact!
                la_f = F.pad(la_f, (0, pad_len), value=0.0)
                b_f = F.pad(b_f, (0, pad_len), value=0.0)

            total_len = q_f.shape[2]
            n_chunks = total_len // c_sz

            q_c = q_f.view(bsz, self.n_heads, n_chunks, c_sz, self.head_dim)
            k_c = k_f.view(bsz, self.n_heads, n_chunks, c_sz, self.head_dim)
            v_c = v_f.view(bsz, self.n_heads, n_chunks, c_sz, self.head_dim)
            la_c = la_f.view(bsz, self.n_heads, n_chunks, c_sz)
            b_c = b_f.view(bsz, self.n_heads, n_chunks, c_sz)

            # Cumulative log-decay within every chunk: g_c[..., i] = sum_{m=0}^i log(alpha_m) <= 0
            g_c = la_c.cumsum(dim=-1)  # [B, H, N_c, C]
            g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)  # [B, H, N_c, C, C]

            idx = torch.arange(c_sz, device=x.device)
            strict_lower = idx.unsqueeze(1) > idx.unsqueeze(0)  # i > j
            causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)  # i >= j
            eye_c = torch.eye(c_sz, device=x.device, dtype=torch.float32)

            # Numerically safe exp(g_i - g_j) for i >= j (strictly in (0, 1])
            decay_strict = torch.where(
                strict_lower, torch.exp(g_diff.clamp(-30.0, 0.0)), torch.zeros_like(g_diff)
            )
            decay_causal = torch.where(
                causal_mask, torch.exp(g_diff.clamp(-30.0, 0.0)), torch.zeros_like(g_diff)
            )

            # Intra-chunk Gram matrix A_ij = beta_i * exp(g_i - g_j) * (k_i^T k_j) for i > j
            kk = torch.matmul(k_c, k_c.transpose(-1, -2))  # [B, H, N_c, C, C]
            A = (b_c.unsqueeze(-1) * decay_strict) * kk
            L_mat = eye_c + A  # Unit lower-triangular [B, H, N_c, C, C]

            # Solve (I + A) [W, U] = [beta * exp(g) * K, beta * V] in ONE fused batched triangular solve!
            exp_g = torch.exp(g_c.clamp(-30.0, 0.0))  # [B, H, N_c, C]
            rhs_w = (b_c * exp_g).unsqueeze(-1) * k_c
            rhs_u = b_c.unsqueeze(-1) * v_c
            rhs = torch.cat([rhs_w, rhs_u], dim=-1)  # [B, H, N_c, C, 2D]
            sol = torch.linalg.solve_triangular(L_mat, rhs, upper=False, unitriangular=True)
            W_all, U_all = sol.split(self.head_dim, dim=-1)

            # Precompute intra-chunk causal attention kernels QK * exp(g_i - g_j)
            qk_decay = torch.matmul(q_c, k_c.transpose(-1, -2)) * decay_causal  # [B, H, N_c, C, C]
            q_scaled = exp_g.unsqueeze(-1) * q_c  # [B, H, N_c, C, D]
            g_end = g_c[..., -1:]  # [B, H, N_c, 1]
            k_to_end = torch.exp((g_end - g_c).clamp(-30.0, 0.0)).unsqueeze(-1) * k_c
            decay_chunk = torch.exp(g_end.clamp(-30.0, 0.0)).unsqueeze(-1)  # [B, H, N_c, 1, 1]

            # Inter-chunk matrix state recurrence: ONLY N_c = T / C iterations (e.g. 8 for T=512!)
            S = st["S"].float()  # [B, H, D_v, D_k]
            out_chunks = []
            for c_idx in range(n_chunks):
                w_cur = W_all[:, :, c_idx]  # [B, H, C, D_k]
                u_cur = U_all[:, :, c_idx]  # [B, H, C, D_v]
                # Corrected values subtracting prior matrix memory S: V_new = U - W @ S^T
                v_new = u_cur - torch.matmul(w_cur, S.transpose(-1, -2))
                # Output = inter-chunk read + intra-chunk WY GEMM
                o_inter = torch.matmul(q_scaled[:, :, c_idx], S.transpose(-1, -2))
                o_intra = torch.matmul(qk_decay[:, :, c_idx], v_new)
                out_chunks.append(o_inter + o_intra)
                # Update O(1) matrix state S: S_c = exp(g_end)*S_{c-1} + V_new^T @ K_to_end
                S = decay_chunk[:, :, c_idx] * S + torch.matmul(
                    v_new.transpose(-1, -2), k_to_end[:, :, c_idx]
                )

            o_all = torch.cat(out_chunks, dim=2)[:, :, :seq_len, :]  # [B, H, T, D]
            o_norm = self.head_norm(o_all.to(x.dtype))

        o_merged = o_norm.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.o_proj(o_merged * F.silu(self.out_gate(x)))
        return y, {"S": S.detach(), "conv": new_conv}

    def step(
        self,
        x_step: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Exact O(1) single-token streaming Delta-Rule update matching chunkwise WY to < 1e-4."""
        bsz = x_step.shape[0]
        x_conv, new_conv = self.short_conv.step(x_step, state["conv"])
        q, k, v, log_alpha, beta = self._project_inputs(x_conv)

        with torch.amp.autocast(device_type=x_step.device.type, enabled=False):
            qt = q[:, :, 0, :].float()  # [B, H, D]
            kt = k[:, :, 0, :].float()
            vt = v[:, :, 0, :].float()
            at = torch.exp(log_alpha[:, :, 0].float()).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            bt = beta[:, :, 0].float().unsqueeze(-1)  # [B, H, 1]

            S = state["S"].float()  # [B, H, D_v, D_k]
            S_decayed = at * S
            # Read old association S_decayed @ k_t, compute error, erase & write rank-1 delta!
            old_v = torch.einsum("bhvd,bhd->bhv", S_decayed, kt)
            delta_v = bt * (vt - old_v)
            S_new = S_decayed + torch.einsum("bhv,bhd->bhvd", delta_v, kt)
            ot = torch.einsum("bhvd,bhd->bhv", S_new, qt).unsqueeze(2)
            o_norm = self.head_norm(ot.to(x_step.dtype))

        o_merged = o_norm.transpose(1, 2).contiguous().view(bsz, 1, self.d_model)
        y = self.o_proj(o_merged * F.silu(self.out_gate(x_step)))
        return y, {"S": S_new.detach(), "conv": new_conv}


class SlidingWindowEchoicAttention(nn.Module):
    """Strictly O(1) Ring-Buffer Local Echoic Cortex (W=64) with O(N * W) Chunked Execution.

    Never materializes an O(N^2) attention matrix and strictly caps its streaming KV ring
    buffer to `window_size` tokens regardless of whether N = 256 or N = 10,000,000.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 64,
        conv_kernel: int = 4,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.rope_theta = rope_theta

        self.short_conv = ShortCausalConv(d_model, kernel_size=conv_kernel)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        d = device if device is not None else self.qkv.weight.device
        dt = dtype if dtype is not None else self.qkv.weight.dtype
        return {
            "k": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=d, dtype=dt),
            "v": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=d, dtype=dt),
            "pos": torch.zeros(1, device=d, dtype=torch.long),
            "conv": self.short_conv.init_state(batch_size, device=d, dtype=dt),
        }

    @staticmethod
    def state_bytes(state: Dict[str, torch.Tensor]) -> int:
        return sum(t.nelement() * t.element_size() for t in state.values())

    def _rope(
        self, start_pos: int, length: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )
        pos = torch.arange(start_pos, start_pos + length, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        return (
            freqs.cos().to(dtype).view(1, 1, length, half),
            freqs.sin().to(dtype).view(1, 1, length, half),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Exact O(N * W) chunked sliding-window causal attention returning O(1) capped ring KV."""
        bsz, seq_len, _ = x.shape
        st = state if state is not None else self.init_state(bsz, device=x.device, dtype=x.dtype)
        start_pos = int(st["pos"].item())

        x_conv, new_conv = self.short_conv(x, state=st["conv"])
        q, k, v = self.qkv(x_conv).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(start_pos, seq_len, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        w = self.window_size
        k_full = torch.cat([st["k"], k], dim=2)
        v_full = torch.cat([st["v"], v], dim=2)
        past_len = st["k"].shape[2]

        # Exact O(N * W) block-banded execution (each block of W queries attends to at most 2W keys!)
        out_blocks = []
        for q_start in range(0, seq_len, w):
            q_end = min(q_start + w, seq_len)
            q_blk = q[:, :, q_start:q_end, :]
            k_end_abs = past_len + q_end
            k_start_abs = max(0, past_len + q_start - (w - 1))
            k_blk = k_full[:, :, k_start_abs:k_end_abs, :]
            v_blk = v_full[:, :, k_start_abs:k_end_abs, :]

            q_abs = torch.arange(past_len + q_start, past_len + q_end, device=x.device).unsqueeze(1)
            k_abs = torch.arange(k_start_abs, k_end_abs, device=x.device).unsqueeze(0)
            diff = q_abs - k_abs
            valid = (diff >= 0) & (diff < w)  # Strictly causal AND within sliding window W!
            attn_mask = torch.zeros(
                q_end - q_start, k_end_abs - k_start_abs, device=x.device, dtype=q.dtype
            ).masked_fill(~valid, float("-inf"))

            out_blk = F.scaled_dot_product_attention(q_blk, k_blk, v_blk, attn_mask=attn_mask)
            out_blocks.append(out_blk)

        out = torch.cat(out_blocks, dim=2).transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.o_proj(out)

        # Strictly cap ring KV buffer to window_size -> O(1) memory even at N = 10,000,000!
        new_state = {
                "k": k_full[:, :, -w:, :].detach().clone(memory_format=torch.contiguous_format),
                "v": v_full[:, :, -w:, :].detach().clone(memory_format=torch.contiguous_format),
            "pos": st["pos"] + seq_len,
            "conv": new_conv,
        }
        return y, new_state

    def step(
        self,
        x_step: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Exact O(1) streaming step over capped window_size ring buffer."""
        return self.forward(x_step, state=state)


class ContrastiveEnergyCritic(nn.Module):
    """100% Backprop-Free Wake-Sleep Contrastive Free-Energy Discriminator.

    All parameters have `requires_grad=False` and update purely inside `@torch.no_grad()` via
    Wake (positive truth energy descent) + Sleep/Dream (negative in-batch hallucination energy
    ascent) analytical InfoNCE local plasticity.
    """

    def __init__(
        self,
        top_dim: int,
        concept_dim: int,
        proj_dim: int = 256,
        temperature: float = 0.12,
    ) -> None:
        super().__init__()
        self.top_dim = top_dim
        self.concept_dim = concept_dim
        self.proj_dim = proj_dim
        self.temperature = temperature

        wu = torch.empty(proj_dim, top_dim)
        wz = torch.empty(proj_dim, concept_dim)
        nn.init.orthogonal_(wu)
        nn.init.orthogonal_(wz)
        self.w_u = nn.Parameter(wu, requires_grad=False)
        self.w_z = nn.Parameter(wz, requires_grad=False)

        self._adam_u: Optional[_LocalAdamState] = None
        self._adam_z: Optional[_LocalAdamState] = None

    def _ensure_adam(self) -> None:
        if self._adam_u is None:
            self._adam_u = _LocalAdamState(self.w_u.shape, self.w_u.device, self.w_u.dtype)
            self._adam_z = _LocalAdamState(self.w_z.shape, self.w_z.device, self.w_z.dtype)

    @torch.no_grad()
    def energy(self, u_ctx: torch.Tensor, z_cand: torch.Tensor) -> torch.Tensor:
        """Compatibility free energy in [0, 2]: 0 = coherent truth, ~1-2 = contradiction/hallucination."""
        with torch.amp.autocast(device_type=u_ctx.device.type, enabled=False):
            pu = F.normalize(F.linear(u_ctx.float(), self.w_u.float()), p=2, dim=-1, eps=1e-6)
            pz = F.normalize(F.linear(z_cand.float(), self.w_z.float()), p=2, dim=-1, eps=1e-6)
            return 1.0 - (pu * pz).sum(dim=-1)

    @torch.no_grad()
    def update_wake_sleep_local(
        self,
        u_ctx: torch.Tensor,
        z_true: torch.Tensor,
        lr: float = 0.04,
        max_samples: int = 256,
    ) -> Dict[str, float]:
        """Exact analytical Wake-Sleep Contrastive InfoNCE local plasticity (0 backprop!)."""
        self._ensure_adam()
        assert self._adam_u is not None and self._adam_z is not None

        with torch.amp.autocast(device_type=u_ctx.device.type, enabled=False):
            u = u_ctx.detach().float().reshape(-1, self.top_dim)
            z = z_true.detach().float().reshape(-1, self.concept_dim)
            n = u.shape[0]
            if n < 2:
                return {"contrastive_loss": 0.0, "e_pos": 0.0, "e_neg": 0.0}
            if n > max_samples:
                stride = max(n // max_samples, 1)
                u = u[::stride][:max_samples]
                z = z[::stride][:max_samples]
                n = u.shape[0]

            qu = F.linear(u, self.w_u.float())
            qz = F.linear(z, self.w_z.float())
            nu = torch.linalg.vector_norm(qu, ord=2, dim=-1, keepdim=True).clamp_min(1e-6)
            nz = torch.linalg.vector_norm(qz, ord=2, dim=-1, keepdim=True).clamp_min(1e-6)
            pu = qu / nu
            pz = qz / nz

            # Similarity matrix S[i, j] = cos(pu_i, pz_j) / T
            sim = pu @ pz.T  # [N, N]
            logits = sim / self.temperature
            eye = torch.eye(n, device=u.device, dtype=torch.float32)

            # Symmetric InfoNCE loss & exact analytical gradient (Wake = +diag, Sleep = -off-diag)
            p_row = F.softmax(logits, dim=1)
            p_col = F.softmax(logits, dim=0)
            loss_row = -F.log_softmax(logits, dim=1).diagonal().mean()
            loss_col = -F.log_softmax(logits, dim=0).diagonal().mean()
            c_loss = 0.5 * (loss_row + loss_col)

            d_sim = (0.5 * (p_row + p_col) - eye) / (n * self.temperature)
            d_pu = d_sim @ pz
            d_pz = d_sim.T @ pu

            # Project gradients onto unit hypersphere tangent planes
            d_qu = (d_pu - (d_pu * pu).sum(dim=-1, keepdim=True) * pu) / nu
            d_qz = (d_pz - (d_pz * pz).sum(dim=-1, keepdim=True) * pz) / nz

            dw_u = d_qu.T @ u
            dw_z = d_qz.T @ z

            self._adam_u.apply(self.w_u, dw_u, lr=lr, weight_decay=1e-4)
            self._adam_z.apply(self.w_z, dw_z, lr=lr, weight_decay=1e-4)

            e_pos = (1.0 - sim.diagonal()).mean().item()
            off_mask = ~eye.bool()
            e_neg = (1.0 - sim[off_mask]).mean().item()
            return {
                "contrastive_loss": float(c_loss.item()),
                "e_pos": float(e_pos),
                "e_neg": float(e_neg),
            }
