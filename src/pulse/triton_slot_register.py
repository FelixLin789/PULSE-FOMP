"""L4 sm_89 99 KiB SMEM Fused Triton Kernel for On-Chip 64-Slot Role-Filler Register.

Keeps the entire slot register state `(K_reg, V_reg, usage)` (`S=64, D=32/64` -> `16-32 KiB << 99 KiB SMEM`)
resident on-chip across the full sequence length `T`, eliminating the ~80 ms/step Python chunk loop and
all intermediate HBM round-trips. Provides both:
  - `pytorch_slot_register_scan`: exact mathematical reference (also used cleanly on CPU).
  - `fused_slot_register_scan`: `torch.autograd.Function` backed by fused forward + backward Triton kernels
    (`num_stages=1` guarantees < 48 KiB SMEM on Ada `sm_89` NVIDIA L4).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU environments without Triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


def has_triton_slot_kernel() -> bool:
    return bool(_HAS_TRITON and torch.cuda.is_available())


def pytorch_slot_register_scan(
    q: torch.Tensor,          # [B, H, T, D] float32 (already L2-normalized)
    r: torch.Tensor,          # [B, H, T, D] float32 (already L2-normalized)
    f: torch.Tensor,          # [B, H, T, D] float32
    gamma: torch.Tensor,      # [B, H, T] float32 in (0, 1)
    k0: torch.Tensor,         # [1, H, S, D] float32
    inv_tau: torch.Tensor,    # [H] float32 > 0
    b_free: torch.Tensor,     # [H] float32 >= 0
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact PyTorch mathematical reference matching the fused Triton kernel to FP32 precision.

    Returns `(o_full [B, H, T, D], a_wr_mean_bh [B, H, S])`.
    """
    bsz, n_heads, seq_len, d = q.shape
    n_slots = k0.shape[2]
    device = q.device

    K_reg = k0.float().expand(bsz, n_heads, n_slots, d).clone()
    V_reg = torch.zeros(bsz, n_heads, n_slots, d, device=device, dtype=torch.float32)
    usage = torch.zeros(bsz, n_heads, n_slots, device=device, dtype=torch.float32)

    itau = inv_tau.float().view(1, n_heads, 1, 1)
    bf = b_free.float().view(1, n_heads, 1, 1)

    outs = []
    a_wr_sum = torch.zeros(bsz, n_heads, n_slots, device=device, dtype=torch.float32)

    for t0 in range(0, seq_len, chunk_size):
        t1 = min(seq_len, t0 + chunk_size)
        c_len = t1 - t0
        qc = q[:, :, t0:t1, :].float()
        rc = r[:, :, t0:t1, :].float()
        fc = f[:, :, t0:t1, :].float()
        gc = gamma[:, :, t0:t1].float().unsqueeze(-1)  # [B, H, C, 1]

        # 1. Strictly causal READ over prior register R_{<c} + intra-chunk causal bindings j <= t
        s_reg = torch.matmul(qc, K_reg.transpose(-2, -1)) * itau
        s_intra = torch.matmul(qc, rc.transpose(-2, -1)) * itau
        c_mask = torch.triu(torch.ones(c_len, c_len, device=device, dtype=torch.bool), diagonal=1)
        s_intra = s_intra + torch.log(gc.transpose(-2, -1).clamp_min(1e-4))
        s_intra = s_intra.masked_fill(c_mask.view(1, 1, c_len, c_len), float("-inf"))
        all_w = F.softmax(torch.cat([s_reg, s_intra], dim=-1), dim=-1)
        o_c = torch.matmul(all_w[..., :n_slots], V_reg) + torch.matmul(all_w[..., n_slots:], fc)
        outs.append(o_c)

        # 2. Competitive Role-Filler WRITE at chunk boundary
        m_free = (1.0 - usage).unsqueeze(2)
        sim_wr = torch.matmul(rc, K_reg.transpose(-2, -1)) * itau + bf * m_free
        a_wr = F.softmax(sim_wr, dim=-1)
        a_wr_sum = a_wr_sum + a_wr.sum(dim=2)

        w_tok = gc * a_wr
        w_sum = w_tok.sum(dim=2)                              # [B, H, S]
        w_slot = w_sum.clamp(0.0, 1.0)
        w_norm = w_tok / w_sum.unsqueeze(2).clamp_min(1e-6)
        r_agg = torch.einsum("bhcs,bhcd->bhsd", w_norm, rc)
        f_agg = torch.einsum("bhcs,bhcd->bhsd", w_norm, fc)

        ws = w_slot.unsqueeze(-1)
        K_raw = (1.0 - ws) * K_reg + ws * r_agg
        k_norm = torch.sqrt((K_raw * K_raw).sum(dim=-1, keepdim=True) + 1e-12)
        K_reg = K_raw / k_norm.clamp_min(1e-6)
        V_reg = (1.0 - ws) * V_reg + ws * f_agg
        usage = (usage + w_slot).clamp(0.0, 1.0)

    o_full = torch.cat(outs, dim=2)
    a_wr_mean_bh = a_wr_sum / float(max(1, seq_len))
    return o_full, a_wr_mean_bh


if _HAS_TRITON:

    @triton.jit
    def _slot_reg_fwd_kernel(
        Q_ptr, R_ptr, F_ptr, G_ptr, K0_ptr, InvTau_ptr, BFree_ptr,
        O_ptr, AMean_ptr, KHist_ptr, VHist_ptr, UHist_ptr,
        T, inv_T, N_CHUNKS, H,
        C: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
        INPUT_PREC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h_idx = pid % H

        inv_tau = tl.load(InvTau_ptr + h_idx).to(tl.float32)
        b_free = tl.load(BFree_ptr + h_idx).to(tl.float32)

        offs_s = tl.arange(0, S)
        offs_d = tl.arange(0, D)
        offs_c = tl.arange(0, C)

        # Load initial slot keys K0[h, S, D] -> stays resident on-chip in SRAM/registers across all chunks!
        k0_base = K0_ptr + h_idx * S * D
        K_reg = tl.load(k0_base + offs_s[:, None] * D + offs_d[None, :]).to(tl.float32)
        V_reg = tl.zeros((S, D), dtype=tl.float32)
        usage = tl.zeros((S,), dtype=tl.float32)
        a_wr_acc = tl.zeros((S,), dtype=tl.float32)

        bh_td = pid * T * D
        bh_t = pid * T
        bh_hist_sd = pid * N_CHUNKS * S * D
        bh_hist_s = pid * N_CHUNKS * S

        causal_c = offs_c[:, None] >= offs_c[None, :]

        for c_idx in range(0, N_CHUNKS):
            t0 = c_idx * C
            t_offs = t0 + offs_c
            valid_c = t_offs < T

            # Save chunk-start snapshots for exact O(1)-SRAM backward recompute
            tl.store(KHist_ptr + bh_hist_sd + c_idx * S * D + offs_s[:, None] * D + offs_d[None, :], K_reg)
            tl.store(VHist_ptr + bh_hist_sd + c_idx * S * D + offs_s[:, None] * D + offs_d[None, :], V_reg)
            tl.store(UHist_ptr + bh_hist_s + c_idx * S + offs_s, usage)

            # Load chunk tokens [C, D] and [C]
            tok_ptrs = bh_td + t_offs[:, None] * D + offs_d[None, :]
            mask_cd = valid_c[:, None]
            qc = tl.load(Q_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
            rc = tl.load(R_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
            fc = tl.load(F_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
            gc_raw = tl.load(G_ptr + bh_t + t_offs, mask=valid_c, other=0.0).to(tl.float32)
            gc = tl.where(valid_c, gc_raw, 0.0)

            # 1. Strictly causal 2-block online softmax READ (zero [C, S+C] concat buffer!)
            s_reg = tl.dot(qc, tl.trans(K_reg), input_precision=INPUT_PREC) * inv_tau             # [C, S]
            s_int = tl.dot(qc, tl.trans(rc), input_precision=INPUT_PREC) * inv_tau              # [C, C]
            log_gc = tl.log(tl.maximum(gc, 1e-4))
            mask_int = causal_c & valid_c[None, :]
            s_int = tl.where(mask_int, s_int + log_gc[None, :], -1.0e30)

            m_reg = tl.max(s_reg, axis=1)
            m_int = tl.max(s_int, axis=1)
            m_tot = tl.where(valid_c, tl.maximum(m_reg, m_int), 0.0)

            e_reg = tl.where(valid_c[:, None], tl.exp(s_reg - m_tot[:, None]), 0.0)
            e_int = tl.where(mask_int & valid_c[:, None], tl.exp(s_int - m_tot[:, None]), 0.0)
            l_tot = tl.where(valid_c, tl.sum(e_reg, axis=1) + tl.sum(e_int, axis=1), 1.0)
            w_reg = e_reg / l_tot[:, None]
            w_int = e_int / l_tot[:, None]

            o_c = tl.dot(w_reg, V_reg, input_precision=INPUT_PREC) + tl.dot(w_int, fc, input_precision=INPUT_PREC)
            tl.store(O_ptr + tok_ptrs, o_c, mask=mask_cd)

            # 2. Competitive Role-Filler WRITE at chunk boundary
            sim_wr = tl.dot(rc, tl.trans(K_reg), input_precision=INPUT_PREC) * inv_tau + b_free * (1.0 - usage)[None, :]
            m_wr = tl.max(sim_wr, axis=1)
            e_wr = tl.exp(sim_wr - m_wr[:, None])
            a_wr = e_wr / tl.sum(e_wr, axis=1)[:, None]
            a_wr_acc = a_wr_acc + tl.sum(tl.where(valid_c[:, None], a_wr, 0.0), axis=0)

            w_tok = gc[:, None] * a_wr                                                          # [C, S], 0 on padded rows!
            w_sum = tl.sum(w_tok, axis=0)                                                       # [S]
            w_slot = tl.minimum(w_sum, 1.0)                                                     # [S]
            inv_w_sum = 1.0 / tl.maximum(w_sum, 1e-6)
            w_norm = w_tok * inv_w_sum[None, :]                                                 # [C, S]

            r_agg = tl.dot(tl.trans(w_norm), rc, input_precision=INPUT_PREC)                    # [S, D]
            f_agg = tl.dot(tl.trans(w_norm), fc, input_precision=INPUT_PREC)                    # [S, D]

            K_raw = (1.0 - w_slot[:, None]) * K_reg + w_slot[:, None] * r_agg
            k_norm = tl.sqrt(tl.sum(K_raw * K_raw, axis=1) + 1e-12)
            K_reg = K_raw / tl.maximum(k_norm, 1e-6)[:, None]
            V_reg = (1.0 - w_slot[:, None]) * V_reg + w_slot[:, None] * f_agg
            usage = tl.minimum(usage + w_slot, 1.0)

        tl.store(AMean_ptr + pid * S + offs_s, a_wr_acc * inv_T)


    @triton.jit
    def _slot_reg_bwd_kernel(
        Q_ptr, R_ptr, F_ptr, G_ptr, InvTau_ptr, BFree_ptr,
        KHist_ptr, VHist_ptr, UHist_ptr,
        dO_ptr, dAMean_ptr,
        dQ_ptr, dR_ptr, dF_ptr, dG_ptr, dK0_bh_ptr, dV_bh_ptr, dInvTau_bh_ptr, dBFree_bh_ptr,
        T, inv_T, N_CHUNKS, H,
        C: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
        INPUT_PREC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h_idx = pid % H

        inv_tau = tl.load(InvTau_ptr + h_idx).to(tl.float32)
        b_free = tl.load(BFree_ptr + h_idx).to(tl.float32)

        offs_s = tl.arange(0, S)
        offs_d = tl.arange(0, D)
        offs_c = tl.arange(0, C)

        d_a_ent = tl.load(dAMean_ptr + pid * S + offs_s).to(tl.float32) * inv_T

        dU_next = tl.zeros((S,), dtype=tl.float32)
        d_inv_tau_acc = tl.zeros((1,), dtype=tl.float32)
        d_b_free_acc = tl.zeros((1,), dtype=tl.float32)

        bh_td = pid * T * D
        bh_t = pid * T
        bh_sd = pid * S * D + offs_s[:, None] * D + offs_d[None, :]
        bh_hist_sd = pid * N_CHUNKS * S * D
        bh_hist_s = pid * N_CHUNKS * S

        causal_c = offs_c[:, None] >= offs_c[None, :]

        # Reverse scan across chunks in 3 non-overlapping live-range stages so D=64 fits < 48 KiB SMEM on L4 sm_89!
        # `dK0_bh_ptr` and `dV_bh_ptr` are zero-initialized and stay L2-resident (192 KiB << 48 MiB L2 on L4).
        for rev_i in range(0, N_CHUNKS):
            c_idx = N_CHUNKS - 1 - rev_i
            t0 = c_idx * C
            t_offs = t0 + offs_c
            valid_c = t_offs < T
            mask_cd = valid_c[:, None]
            tok_ptrs = bh_td + t_offs[:, None] * D + offs_d[None, :]
            hist_sd = bh_hist_sd + c_idx * S * D + offs_s[:, None] * D + offs_d[None, :]

            # === Stage 1: WRITE softmax + WRITE-V (releases V_reg, fc, f_agg, d_f_agg before K-stage!) ===
            K_reg = tl.load(KHist_ptr + hist_sd).to(tl.float32)
            usage = tl.load(UHist_ptr + bh_hist_s + c_idx * S + offs_s).to(tl.float32)
            rc = tl.load(R_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
            gc_raw = tl.load(G_ptr + bh_t + t_offs, mask=valid_c, other=0.0).to(tl.float32)
            gc = tl.where(valid_c, gc_raw, 0.0)

            raw_wr = tl.dot(rc, tl.trans(K_reg), input_precision=INPUT_PREC)                    # [C, S]
            sim_wr = raw_wr * inv_tau + b_free * (1.0 - usage)[None, :]
            m_wr = tl.max(sim_wr, axis=1)
            e_wr = tl.exp(sim_wr - m_wr[:, None])
            a_wr = e_wr / tl.sum(e_wr, axis=1)[:, None]

            w_tok = gc[:, None] * a_wr
            w_sum = tl.sum(w_tok, axis=0)
            w_slot = tl.minimum(w_sum, 1.0)
            inv_w_sum = 1.0 / tl.maximum(w_sum, 1e-6)
            w_norm = w_tok * inv_w_sum[None, :]

            fc = tl.load(F_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
            V_reg = tl.load(VHist_ptr + hist_sd).to(tl.float32)
            dV_next = tl.load(dV_bh_ptr + bh_sd).to(tl.float32)
            f_agg = tl.dot(tl.trans(w_norm), fc, input_precision=INPUT_PREC)
            dV_prev = (1.0 - w_slot[:, None]) * dV_next
            tl.store(dV_bh_ptr + bh_sd, dV_prev)
            d_f_agg = w_slot[:, None] * dV_next
            d_w_slot_v = tl.sum(dV_next * (f_agg - V_reg), axis=1)
            d_fc_wr = tl.dot(w_norm, d_f_agg, input_precision=INPUT_PREC)
            tl.store(dF_ptr + tok_ptrs, d_fc_wr, mask=mask_cd)
            d_w_norm = tl.dot(fc, tl.trans(d_f_agg), input_precision=INPUT_PREC)

            # === Stage 2: WRITE-K + WRITE-U + WRITE softmax backward ===
            dK_next = tl.load(dK0_bh_ptr + bh_sd).to(tl.float32)
            r_agg = tl.dot(tl.trans(w_norm), rc, input_precision=INPUT_PREC)
            K_raw = (1.0 - w_slot[:, None]) * K_reg + w_slot[:, None] * r_agg
            k_norm = tl.sqrt(tl.sum(K_raw * K_raw, axis=1) + 1e-12)
            k_den = tl.maximum(k_norm, 1e-6)
            K_unit = K_raw / k_den[:, None]

            proj_k = tl.sum(dK_next * K_unit, axis=1)
            dK_raw = tl.where(
                (k_norm > 1e-6)[:, None],
                (dK_next - K_unit * proj_k[:, None]) / k_den[:, None],
                dK_next / 1e-6,
            )
            dK_prev = (1.0 - w_slot[:, None]) * dK_raw
            d_r_agg = w_slot[:, None] * dK_raw
            d_w_slot_k = tl.sum(dK_raw * (r_agg - K_reg), axis=1)

            d_u_pre = tl.where((usage + w_slot) < 1.0, dU_next, 0.0)
            d_w_slot = d_w_slot_k + d_w_slot_v + d_u_pre
            d_w_sum_clamp = tl.where(w_sum < 1.0, d_w_slot, 0.0)

            d_rc_wr = tl.dot(w_norm, d_r_agg, input_precision=INPUT_PREC)
            d_w_norm = d_w_norm + tl.dot(rc, tl.trans(d_r_agg), input_precision=INPUT_PREC)

            d_w_sum_norm = tl.where(w_sum > 1e-6, -tl.sum(d_w_norm * w_norm, axis=0) * inv_w_sum, 0.0)
            d_w_tok = d_w_norm * inv_w_sum[None, :] + (d_w_sum_clamp + d_w_sum_norm)[None, :]

            d_gc_wr = tl.sum(d_w_tok * a_wr, axis=1)
            d_a_wr = d_w_tok * gc[:, None] + tl.where(valid_c[:, None], d_a_ent[None, :], 0.0)
            d_sim_wr = a_wr * (d_a_wr - tl.sum(d_a_wr * a_wr, axis=1)[:, None])
            d_sim_wr = tl.where(valid_c[:, None], d_sim_wr, 0.0)

            d_rc_wr = d_rc_wr + tl.dot(d_sim_wr, K_reg, input_precision=INPUT_PREC) * inv_tau
            dK_prev = dK_prev + tl.dot(tl.trans(d_sim_wr), rc, input_precision=INPUT_PREC) * inv_tau
            dU_next = d_u_pre - tl.sum(d_sim_wr, axis=0) * b_free
            d_inv_tau_acc = d_inv_tau_acc + tl.sum(tl.sum(d_sim_wr * raw_wr, axis=1), axis=0)
            d_b_free_acc = d_b_free_acc + tl.sum(tl.sum(d_sim_wr * (1.0 - usage)[None, :], axis=1), axis=0)

            tl.store(dK0_bh_ptr + bh_sd, dK_prev)
            tl.store(dR_ptr + tok_ptrs, d_rc_wr, mask=mask_cd)
            tl.store(dG_ptr + bh_t + t_offs, d_gc_wr, mask=valid_c)
            tl.debug_barrier()

            # === Stage 3: READ recompute & backward (loads qc, d_oc ONLY here!) ===
            qc = tl.load(Q_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
            d_oc = tl.load(dO_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32)

            raw_reg = tl.dot(qc, tl.trans(K_reg), input_precision=INPUT_PREC)
            raw_int = tl.dot(qc, tl.trans(rc), input_precision=INPUT_PREC)
            log_gc = tl.log(tl.maximum(gc, 1e-4))
            mask_int = causal_c & valid_c[None, :]
            s_reg = raw_reg * inv_tau
            s_int = tl.where(mask_int, raw_int * inv_tau + log_gc[None, :], -1.0e30)

            m_tot = tl.where(valid_c, tl.maximum(tl.max(s_reg, axis=1), tl.max(s_int, axis=1)), 0.0)
            e_reg = tl.where(valid_c[:, None], tl.exp(s_reg - m_tot[:, None]), 0.0)
            e_int = tl.where(mask_int & valid_c[:, None], tl.exp(s_int - m_tot[:, None]), 0.0)
            l_tot = tl.where(valid_c, tl.sum(e_reg, axis=1) + tl.sum(e_int, axis=1), 1.0)
            w_reg = e_reg / l_tot[:, None]
            w_int = e_int / l_tot[:, None]

            V_reg_rd = tl.load(VHist_ptr + hist_sd).to(tl.float32)
            d_w_reg = tl.dot(d_oc, tl.trans(V_reg_rd), input_precision=INPUT_PREC)
            dV_tot = tl.load(dV_bh_ptr + bh_sd).to(tl.float32) + tl.dot(tl.trans(w_reg), d_oc, input_precision=INPUT_PREC)
            tl.store(dV_bh_ptr + bh_sd, dV_tot)

            fc_rd = tl.load(F_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
            d_w_int = tl.dot(d_oc, tl.trans(fc_rd), input_precision=INPUT_PREC)
            dF_tot = tl.load(dF_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32) + tl.dot(tl.trans(w_int), d_oc, input_precision=INPUT_PREC)
            tl.store(dF_ptr + tok_ptrs, dF_tot, mask=mask_cd)

            dot_w = tl.sum(d_w_reg * w_reg, axis=1) + tl.sum(d_w_int * w_int, axis=1)
            d_s_reg = w_reg * (d_w_reg - dot_w[:, None])
            d_s_int = w_int * (d_w_int - dot_w[:, None])

            d_qc = (
                tl.dot(d_s_reg, K_reg, input_precision=INPUT_PREC)
                + tl.dot(d_s_int, rc, input_precision=INPUT_PREC)
            ) * inv_tau
            tl.store(dQ_ptr + tok_ptrs, d_qc, mask=mask_cd)

            dK_tot = tl.load(dK0_bh_ptr + bh_sd).to(tl.float32) + tl.dot(tl.trans(d_s_reg), qc, input_precision=INPUT_PREC) * inv_tau
            tl.store(dK0_bh_ptr + bh_sd, dK_tot)

            dR_tot = tl.load(dR_ptr + tok_ptrs, mask=mask_cd, other=0.0).to(tl.float32) + tl.dot(tl.trans(d_s_int), qc, input_precision=INPUT_PREC) * inv_tau
            tl.store(dR_ptr + tok_ptrs, dR_tot, mask=mask_cd)

            d_gc_rd = tl.where(valid_c & (gc > 1e-4), tl.sum(d_s_int, axis=0) / tl.maximum(gc, 1e-4), 0.0)
            dG_tot = tl.load(dG_ptr + bh_t + t_offs, mask=valid_c, other=0.0).to(tl.float32) + d_gc_rd
            tl.store(dG_ptr + bh_t + t_offs, dG_tot, mask=valid_c)

            d_inv_tau_acc = (
                d_inv_tau_acc
                + tl.sum(tl.sum(d_s_reg * raw_reg, axis=1), axis=0)
                + tl.sum(tl.sum(d_s_int * raw_int, axis=1), axis=0)
            )
            tl.debug_barrier()

        # Store per-(b, h) scalar parameter gradients (`dK0_bh_ptr` already holds exact final dK0!)
        offs_1 = tl.arange(0, 1)
        tl.store(dInvTau_bh_ptr + pid + offs_1, d_inv_tau_acc)
        tl.store(dBFree_bh_ptr + pid + offs_1, d_b_free_acc)


class _FusedSlotRegisterFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        r: torch.Tensor,
        f: torch.Tensor,
        gamma: torch.Tensor,
        k0: torch.Tensor,
        inv_tau: torch.Tensor,
        b_free: torch.Tensor,
        chunk_size: int,
        input_prec: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_c = q.float().contiguous()
        r_c = r.float().contiguous()
        f_c = f.float().contiguous()
        g_c = gamma.float().contiguous()
        k0_c = k0.float().contiguous()
        itau_c = inv_tau.float().contiguous()
        bf_c = b_free.float().contiguous()

        bsz, n_heads, seq_len, d = q_c.shape
        n_slots = k0_c.shape[2]
        n_chunks = (seq_len + chunk_size - 1) // chunk_size

        o_full = torch.empty_like(q_c)
        a_mean = torch.empty((bsz, n_heads, n_slots), device=q_c.device, dtype=torch.float32)
        k_hist = torch.empty((bsz, n_heads, n_chunks, n_slots, d), device=q_c.device, dtype=torch.float32)
        v_hist = torch.empty((bsz, n_heads, n_chunks, n_slots, d), device=q_c.device, dtype=torch.float32)
        u_hist = torch.empty((bsz, n_heads, n_chunks, n_slots), device=q_c.device, dtype=torch.float32)

        grid = (bsz * n_heads,)
        inv_t = 1.0 / float(max(1, seq_len))
        nw_fwd = 4
        _slot_reg_fwd_kernel[grid](
            q_c, r_c, f_c, g_c, k0_c, itau_c, bf_c,
            o_full, a_mean, k_hist, v_hist, u_hist,
            seq_len, inv_t, n_chunks, n_heads,
            C=chunk_size, S=n_slots, D=d, INPUT_PREC=input_prec,
            num_warps=nw_fwd, num_stages=1,
        )
        ctx.save_for_backward(q_c, r_c, f_c, g_c, itau_c, bf_c, k_hist, v_hist, u_hist)
        ctx.chunk_size = chunk_size
        ctx.n_slots = n_slots
        ctx.input_prec = input_prec
        return o_full, a_mean

    @staticmethod
    def backward(ctx, d_o: torch.Tensor, d_a_mean: torch.Tensor):  # type: ignore[override]
        q_c, r_c, f_c, g_c, itau_c, bf_c, k_hist, v_hist, u_hist = ctx.saved_tensors
        bsz, n_heads, seq_len, d = q_c.shape
        n_slots = ctx.n_slots
        chunk_size = ctx.chunk_size
        n_chunks = k_hist.shape[2]

        d_oc = d_o.float().contiguous()
        d_am = d_a_mean.float().contiguous() if d_a_mean is not None else torch.zeros((bsz, n_heads, n_slots), device=q_c.device, dtype=torch.float32)

        d_q = torch.empty_like(q_c)
        d_r = torch.empty_like(r_c)
        d_f = torch.empty_like(f_c)
        d_g = torch.empty_like(g_c)
        d_k0_bh = torch.zeros((bsz, n_heads, n_slots, d), device=q_c.device, dtype=torch.float32)
        d_v_bh = torch.zeros((bsz, n_heads, n_slots, d), device=q_c.device, dtype=torch.float32)
        d_itau_bh = torch.empty((bsz, n_heads), device=q_c.device, dtype=torch.float32)
        d_bf_bh = torch.empty((bsz, n_heads), device=q_c.device, dtype=torch.float32)

        grid = (bsz * n_heads,)
        inv_t = 1.0 / float(max(1, seq_len))
        _slot_reg_bwd_kernel[grid](
            q_c, r_c, f_c, g_c, itau_c, bf_c,
            k_hist, v_hist, u_hist,
            d_oc, d_am,
            d_q, d_r, d_f, d_g, d_k0_bh, d_v_bh, d_itau_bh, d_bf_bh,
            seq_len, inv_t, n_chunks, n_heads,
            C=chunk_size, S=n_slots, D=d, INPUT_PREC=ctx.input_prec,
            num_warps=4, num_stages=1,
        )
        return (
            d_q,
            d_r,
            d_f,
            d_g,
            d_k0_bh.sum(dim=0, keepdim=True),
            d_itau_bh.sum(dim=0),
            d_bf_bh.sum(dim=0),
            None,
            None,
        )


def fused_slot_register_scan(
    q: torch.Tensor,
    r: torch.Tensor,
    f: torch.Tensor,
    gamma: torch.Tensor,
    k0: torch.Tensor,
    inv_tau: torch.Tensor,
    b_free: torch.Tensor,
    chunk_size: int = 32,
    use_tf32: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to L4 sm_89 99 KiB SMEM Fused Triton kernel when eligible, else exact PyTorch reference."""
    d = q.shape[-1]
    n_slots = k0.shape[2]
    can_triton = (
        _HAS_TRITON
        and q.is_cuda
        and chunk_size in (16, 32, 64)
        and n_slots in (16, 32, 64)
        and d in (16, 32, 64)
    )
    if not can_triton:
        return pytorch_slot_register_scan(q, r, f, gamma, k0, inv_tau, b_free, chunk_size=chunk_size)
    prec = "tf32" if use_tf32 else "ieee"
    return _FusedSlotRegisterFn.apply(q, r, f, gamma, k0, inv_tau, b_free, chunk_size, prec)  # type: ignore[return-value]
