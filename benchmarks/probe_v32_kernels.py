"""20-second L4 verification & timing benchmark for PULSE-v3.2-Turbo kernels:
1. Custom fused Triton 4-tap CanonLayer (FW + BW exact gradcheck vs PyTorch + ms on 1536ch & 512ch)
2. FLA `causal_conv1d` availability & speed comparison
3. Compiled Semantic Core (1-step branchless relax + compiled Hillis-Steele SSM + top-down)
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from pulse.pulse_v3 import PULSEv3Config, PULSEv3Model


@triton.jit
def _canon4_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    T,
    D,
    stride_xb,
    stride_xt,
    stride_yb,
    stride_yt,
    APPLY_SILU: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)
    n_t_blocks = tl.cdiv(T, BLOCK_T)
    b = pid_bt // n_t_blocks
    t_blk = pid_bt % n_t_blocks

    offs_t = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_t = offs_t < T

    # Load 4 weights [BLOCK_D] in fp32 (stays in registers for whole tile!)
    w0 = tl.load(w_ptr + 0 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + 1 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + 2 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + 3 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32) + 1.0

    base_x = x_ptr + b * stride_xb
    # Load x[t], x[t-1], x[t-2], x[t-3] coalesced along D!
    x_t0 = tl.load(base_x + offs_t[:, None] * stride_xt + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    m1 = (offs_t >= 1) & mask_t
    x_t1 = tl.load(base_x + (offs_t - 1)[:, None] * stride_xt + offs_d[None, :], mask=m1[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    m2 = (offs_t >= 2) & mask_t
    x_t2 = tl.load(base_x + (offs_t - 2)[:, None] * stride_xt + offs_d[None, :], mask=m2[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    m3 = (offs_t >= 3) & mask_t
    x_t3 = tl.load(base_x + (offs_t - 3)[:, None] * stride_xt + offs_d[None, :], mask=m3[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc = x_t0 * w3[None, :] + x_t1 * w2[None, :] + x_t2 * w1[None, :] + x_t3 * w0[None, :]
    if APPLY_SILU:
        acc = acc * tl.sigmoid(acc)

    base_y = y_ptr + b * stride_yb
    tl.store(base_y + offs_t[:, None] * stride_yt + offs_d[None, :], acc, mask=mask_t[:, None] & mask_d[None, :])


@triton.jit
def _canon4_bwd_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    dx_ptr,
    dw_ptr,
    T,
    D,
    stride_xb,
    stride_xt,
    stride_dyb,
    stride_dyt,
    stride_dxb,
    stride_dxt,
    APPLY_SILU: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)
    n_t_blocks = tl.cdiv(T, BLOCK_T)
    b = pid_bt // n_t_blocks
    t_blk = pid_bt % n_t_blocks

    offs_t = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask_t = offs_t < T

    w0 = tl.load(w_ptr + 0 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + 1 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + 2 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + 3 * D + offs_d, mask=mask_d, other=0.0).to(tl.float32) + 1.0

    base_x = x_ptr + b * stride_xb
    base_dy = dy_ptr + b * stride_dyb

    # Compute du[t + s] for s in {0, 1, 2, 3} (needed for both dw and dx!)
    # Helper: for any shift s, du_s = dy[t+s] * dsilu(u[t+s])
    # First load x needed for u[t]..u[t+3] and dw
    x_m3 = tl.load(base_x + (offs_t - 3)[:, None] * stride_xt + offs_d[None, :], mask=((offs_t >= 3) & mask_t)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    x_m2 = tl.load(base_x + (offs_t - 2)[:, None] * stride_xt + offs_d[None, :], mask=((offs_t >= 2) & mask_t)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    x_m1 = tl.load(base_x + (offs_t - 1)[:, None] * stride_xt + offs_d[None, :], mask=((offs_t >= 1) & mask_t)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    x_0 = tl.load(base_x + offs_t[:, None] * stride_xt + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    dy_0 = tl.load(base_dy + offs_t[:, None] * stride_dyt + offs_d[None, :], mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    if APPLY_SILU:
        u0 = x_0 * w3[None, :] + x_m1 * w2[None, :] + x_m2 * w1[None, :] + x_m3 * w0[None, :]
        s0 = tl.sigmoid(u0)
        du_0 = dy_0 * (s0 * (1.0 + u0 * (1.0 - s0)))
    else:
        du_0 = dy_0

    # Accumulate dw[0..3] across BLOCK_T and atomic_add into FP32 dw buffer!
    tl.atomic_add(dw_ptr + 0 * D + offs_d, tl.sum(du_0 * x_m3, axis=0), mask=mask_d)
    tl.atomic_add(dw_ptr + 1 * D + offs_d, tl.sum(du_0 * x_m2, axis=0), mask=mask_d)
    tl.atomic_add(dw_ptr + 2 * D + offs_d, tl.sum(du_0 * x_m1, axis=0), mask=mask_d)
    tl.atomic_add(dw_ptr + 3 * D + offs_d, tl.sum(du_0 * x_0, axis=0), mask=mask_d)

    # Now compute du[t+1], du[t+2], du[t+3] for dx[t]
    t_p1 = offs_t + 1
    m_p1 = t_p1 < T
    dy_1 = tl.load(base_dy + t_p1[:, None] * stride_dyt + offs_d[None, :], mask=m_p1[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    t_p2 = offs_t + 2
    m_p2 = t_p2 < T
    dy_2 = tl.load(base_dy + t_p2[:, None] * stride_dyt + offs_d[None, :], mask=m_p2[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    t_p3 = offs_t + 3
    m_p3 = t_p3 < T
    dy_3 = tl.load(base_dy + t_p3[:, None] * stride_dyt + offs_d[None, :], mask=m_p3[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    if APPLY_SILU:
        x_p1 = tl.load(base_x + t_p1[:, None] * stride_xt + offs_d[None, :], mask=m_p1[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        x_p2 = tl.load(base_x + t_p2[:, None] * stride_xt + offs_d[None, :], mask=m_p2[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        x_p3 = tl.load(base_x + t_p3[:, None] * stride_xt + offs_d[None, :], mask=m_p3[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        u1 = x_p1 * w3[None, :] + x_0 * w2[None, :] + x_m1 * w1[None, :] + x_m2 * w0[None, :]
        s1 = tl.sigmoid(u1)
        du_1 = dy_1 * (s1 * (1.0 + u1 * (1.0 - s1)))

        u2 = x_p2 * w3[None, :] + x_p1 * w2[None, :] + x_0 * w1[None, :] + x_m1 * w0[None, :]
        s2 = tl.sigmoid(u2)
        du_2 = dy_2 * (s2 * (1.0 + u2 * (1.0 - s2)))

        u3 = x_p3 * w3[None, :] + x_p2 * w2[None, :] + x_p1 * w1[None, :] + x_0 * w0[None, :]
        s3 = tl.sigmoid(u3)
        du_3 = dy_3 * (s3 * (1.0 + u3 * (1.0 - s3)))
    else:
        du_1 = dy_1
        du_2 = dy_2
        du_3 = dy_3

    dx = du_0 * w3[None, :] + du_1 * w2[None, :] + du_2 * w1[None, :] + du_3 * w0[None, :]
    base_dx = dx_ptr + b * stride_dxb
    tl.store(base_dx + offs_t[:, None] * stride_dxt + offs_d[None, :], dx, mask=mask_t[:, None] & mask_d[None, :])


class _TritonCanon4Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w_kd: torch.Tensor, apply_silu: bool) -> torch.Tensor:
        x_c = x.contiguous()
        w_c = w_kd.contiguous()
        B, T, D = x_c.shape
        y = torch.empty_like(x_c)
        BLOCK_T = 32
        BLOCK_D = 64
        grid = (B * triton.cdiv(T, BLOCK_T), triton.cdiv(D, BLOCK_D))
        _canon4_fwd_kernel[grid](
            x_c, w_c, y, T, D,
            x_c.stride(0), x_c.stride(1),
            y.stride(0), y.stride(1),
            APPLY_SILU=apply_silu,
            BLOCK_T=BLOCK_T,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(x_c, w_c)
        ctx.apply_silu = apply_silu
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_c, w_c = ctx.saved_tensors
        dy_c = dy.contiguous()
        B, T, D = x_c.shape
        dx = torch.empty_like(x_c)
        dw = torch.zeros(4, D, device=x_c.device, dtype=torch.float32)
        BLOCK_T = 32
        BLOCK_D = 64
        grid = (B * triton.cdiv(T, BLOCK_T), triton.cdiv(D, BLOCK_D))
        _canon4_bwd_kernel[grid](
            x_c, w_c, dy_c, dx, dw, T, D,
            x_c.stride(0), x_c.stride(1),
            dy_c.stride(0), dy_c.stride(1),
            dx.stride(0), dx.stride(1),
            APPLY_SILU=ctx.apply_silu,
            BLOCK_T=BLOCK_T,
            BLOCK_D=BLOCK_D,
        )
        return dx, dw.to(w_c.dtype), None


def _time_cuda(fn, warmup: int = 4, iters: int = 12) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> None:
    dev = torch.device("cuda")
    torch.manual_seed(0)

    # 1. Exact numeric & gradient verification (both silu=False and silu=True!)
    for silu in (False, True):
        x_ref = torch.randn(2, 37, 128, device=dev, dtype=torch.float32, requires_grad=True)
        w_ref = torch.randn(4, 128, device=dev, dtype=torch.float32, requires_grad=True)
        x_pad = F.pad(x_ref, (0, 0, 3, 0))
        u_ref = x_ref * (1.0 + w_ref[3]) + x_pad[:, 2:39] * w_ref[2] + x_pad[:, 1:38] * w_ref[1] + x_pad[:, 0:37] * w_ref[0]
        y_ref = F.silu(u_ref) if silu else u_ref
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)

        x_tri = x_ref.detach().clone().requires_grad_(True)
        w_tri = w_ref.detach().clone().requires_grad_(True)
        y_tri = _TritonCanon4Fn.apply(x_tri, w_tri, silu)
        y_tri.backward(grad_out)

        print(
            f"[GRADCHECK silu={silu}] y_diff={(y_tri - y_ref).abs().max().item():.2e} "
            f"dx_diff={(x_tri.grad - x_ref.grad).abs().max().item():.2e} "
            f"dw_diff={(w_tri.grad - w_ref.grad).abs().max().item():.2e}",
            flush=True,
        )

    # 2. Speed test on [16, 512, 1536] (Canon-B) and [16, 512, 512] (Canon-A/C) in BF16!
    for D, silu, label in [(1536, True, "Canon-B 1536ch+SiLU"), (512, False, "Canon-A/C 512ch")]:
        x = torch.randn(16, 512, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(4, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        def _tri_fwbw():
            y = _TritonCanon4Fn.apply(x, w, silu)
            y.float().mean().backward()
        ms_tri = _time_cuda(_tri_fwbw)
        print(f"[TRITON CANON4] {label:22s}: {ms_tri:5.2f} ms (FW+BW)", flush=True)

    # 3. Check if FLA causal_conv1d exists & time it for comparison
    try:
        from fla.modules.convolution import causal_conv1d as fla_conv1d
        x = torch.randn(16, 512, 1536, device=dev, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(1536, 4, device=dev, dtype=torch.bfloat16, requires_grad=True)
        def _fla_fwbw():
            y = fla_conv1d(x, w, activation="silu")[0]
            y.float().mean().backward()
        ms_fla = _time_cuda(_fla_fwbw)
        print(f"[FLA CONV1D   ] Canon-B 1536ch+SiLU   : {ms_fla:5.2f} ms (FW+BW)", flush=True)
    except Exception as exc:
        print(f"[FLA CONV1D   ] unavailable: {exc}", flush=True)

    # 4. Compiled Semantic Core (B=16, T=512 FW+BW AND B=1, T=4096 Infer!)
    p30 = PULSEv3Model(PULSEv3Config.colab_30m()).to(dev)
    p30.compile_fast_paths()

    def _core_train_block(h_mid: torch.Tensor, streams: torch.Tensor):
        z = p30._pool_mid_to_concepts(h_mid)
        zt, _, _, _, _ = p30._relax_concept_frames(z, max_steps=1, use_fast_weights=False)
        _, prior, _ = p30.fast_ssm.scan(zt)
        z_pred = p30.causal_layer.predict_down(prior.detach().float(), use_fast_weights=False).to(h_mid.dtype)
        bsz, seq_len, _ = h_mid.shape
        k = p30.config.semantic_clock
        n_groups = (seq_len + k - 1) // k
        bos_c = p30.bos_concept_prior.expand(bsz, 1, -1).to(h_mid.dtype)
        bos_t = h_mid.new_zeros((bsz, 1, p30.config.top_dim))
        ctx_c = torch.cat([bos_c, z_pred.detach()], dim=1)[:, :n_groups, :]
        ctx_t = torch.cat([bos_t, prior], dim=1)[:, :n_groups, :]
        ctx_tok = torch.cat([ctx_c, ctx_t], dim=-1).repeat_interleave(k, dim=1)[:, :seq_len, :]
        td = p30.top_down_norm(p30.concept_to_stream(ctx_tok))
        return streams + torch.tanh(p30.top_down_gate) * td.unsqueeze(-2)

    c_core = torch.compile(_core_train_block, dynamic=True)
    h16 = torch.randn(16, 512, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
    s16 = torch.randn(16, 512, 2, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
    def _c_core_fwbw():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = c_core(h16, s16)
        y.float().mean().backward()
    ms_core_comp_train = _time_cuda(_c_core_fwbw)

    p30.eval()
    h4k = torch.randn(1, 4096, 512, device=dev, dtype=torch.bfloat16)
    s4k = torch.randn(1, 4096, 2, 512, device=dev, dtype=torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        ms_core_comp_4k = _time_cuda(lambda: c_core(h4k, s4k))

    print(f"[COMPILED CORE] Train 16x512 FW+BW    : {ms_core_comp_train:5.2f} ms (was 13.65 ms!)", flush=True)
    print(f"[COMPILED CORE] Infer 1x4096 FW       : {ms_core_comp_4k:5.2f} ms (was 8.16 ms!)", flush=True)


if __name__ == "__main__":
    main()
