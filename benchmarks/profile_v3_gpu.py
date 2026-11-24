"""25-second surgical CUDA autopsy & A/B profiler for PULSE-v3.1-Turbo vs Compiled Transformer."""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from pulse.pulse_v3 import PULSEv3Config, PULSEv3Model
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM
from pulse.v3_blocks import _get_cached_sliding_block_mask, _get_flex_attention, _probe_fla_kda_backend


def _time_cuda(fn, warmup: int = 4, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _fwbw(fn):
    def _run():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = fn()
        y.float().mean().backward()
    return _run


def main() -> None:
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}", flush=True)

    B, T, D, H, K = 16, 512, 512, 8, 64
    p30 = PULSEv3Model(PULSEv3Config.colab_30m()).to(dev)
    p30.compile_fast_paths()
    t30 = ScalableTransformerLM(ScalableTransformerConfig.colab_30m()).to(dev)
    t30.compile_fast_paths()
    _probe_fla_kda_backend(dev)

    x_hid = torch.randn(B, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    x_st = torch.randn(B, T, 2, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    toks = torch.randint(4, 50257, (B, T + 1), device=dev)
    cos, sin = t30._rope_cos_sin(0, T, dev, torch.bfloat16)

    delta_blk = p30.blocks[0]
    swa_blk = p30.blocks[2]
    tf_blk = t30.blocks[0]

    # 1. Whole block FW+BW (compiled)
    ms_tf_blk = _time_cuda(_fwbw(lambda: tf_blk(x_hid, cos, sin, use_cache=False)[0]))
    ms_delta_blk = _time_cuda(_fwbw(lambda: delta_blk(x_st, return_state=False)[0]))
    ms_swa_blk = _time_cuda(_fwbw(lambda: swa_blk(x_st, return_state=False)[0]))

    # 2. Sub-piece breakdown inside Delta block (compiled)
    ms_d_pre1 = _time_cuda(_fwbw(lambda: delta_blk._compiled_pre1(x_st)[0]))
    ms_d_mpre = _time_cuda(_fwbw(lambda: delta_blk.mixer._compiled_pre(x_hid)[2]))
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        q, k, v_kda, log_a, beta, gate_raw, _ = delta_blk.mixer._pre_proj_train(x_hid)
        h_res1 = torch.full((B, T, 1), 0.9, device=dev, dtype=torch.bfloat16)
        h_post1 = torch.ones((B, T, 2), device=dev, dtype=torch.bfloat16)
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v_kda = v_kda.detach().requires_grad_(True)
    log_a = log_a.detach().requires_grad_(True)
    beta = beta.detach().requires_grad_(True)
    gate_raw = gate_raw.detach().requires_grad_(True)
    o_dummy = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16, requires_grad=True)

    from fla.ops.kda import chunk_kda
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    g_hd = log_a.mean(dim=-1).detach().requires_grad_(True)

    ms_kda = _time_cuda(_fwbw(lambda: chunk_kda(q, k, v_kda, log_a, beta, scale=1.0, safe_gate=True, output_final_state=False)[0]))
    ms_gdn = _time_cuda(_fwbw(lambda: chunk_gated_delta_rule(q, k, v_kda, g_hd, beta, scale=1.0, output_final_state=False)[0]))
    ms_d_mpost = _time_cuda(_fwbw(lambda: delta_blk.mixer._compiled_post(o_dummy, gate_raw, torch.bfloat16)))
    ms_d_post1_mlp = _time_cuda(_fwbw(lambda: delta_blk._compiled_post1_mlp(x_st, x_hid, h_res1, h_post1)[0]))

    # 3. Sub-piece breakdown inside SWA-128 block (compiled)
    ms_s_mpre = _time_cuda(_fwbw(lambda: swa_blk.mixer._compiled_pre(x_hid)[0]))
    qs = torch.randn(B, H, T, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    ks = torch.randn(B, H, T, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    vs = torch.randn(B, H, T, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
    flex_fn, _ = _get_flex_attention()
    bmask = _get_cached_sliding_block_mask(T, 128, dev)
    ms_flex = _time_cuda(_fwbw(lambda: flex_fn(qs, ks, vs, block_mask=bmask)))
    ms_sdpa = _time_cuda(_fwbw(lambda: F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)))
    ms_s_mpost = _time_cuda(_fwbw(lambda: swa_blk.mixer._compiled_post(qs, gate_raw)))

    # 4. A/B Test: Merged 2-graph Delta block vs current 4-graph Delta block
    def _merged_pre(st):
        sub1, hr1, hp1 = delta_blk.mhc1.pre_mix(st)
        n1, _ = delta_blk.canon_a(delta_blk.norm1(sub1), state=None, return_state=False)
        q_, k_, v_, la_, be_, gr_, _ = delta_blk.mixer._pre_proj_train(n1)
        return q_, k_, v_, la_, be_, gr_, hr1, hp1

    def _merged_post(st, o_all, gr, hr1, hp1):
        hm = delta_blk.mixer._post_proj(o_all, gr, st.dtype)
        st2 = delta_blk.mhc1.post_mix(st, hm, hr1, hp1)
        sub2, hr2, hp2 = delta_blk.mhc2.pre_mix(st2)
        n2, _ = delta_blk.canon_c(delta_blk.norm2(sub2), state=None, return_state=False)
        g_, u_ = delta_blk.w_gate_up(n2).chunk(2, dim=-1)
        hf = delta_blk.w_down(F.silu(g_) * u_)
        return delta_blk.mhc2.post_mix(st2, hf, hr2, hp2)

    c_mpre = torch.compile(_merged_pre)
    c_mpost = torch.compile(_merged_post)

    def _run_merged_delta():
        q_, k_, v_, la_, be_, gr_, hr1_, hp1_ = c_mpre(x_st)
        o_, _ = chunk_kda(q_, k_, v_, la_, be_, scale=1.0, safe_gate=True, output_final_state=False)
        return c_mpost(x_st, o_, gr_, hr1_, hp1_)

    ms_delta_merged_kda = _time_cuda(_fwbw(_run_merged_delta))

    # 5. A/B Test: 1-stream residual (no mHC) + Canon + SwiGLU compiled vs mHC(n=2) + Canon + SwiGLU
    def _1stream_mlp(x_in):
        n2, _ = delta_blk.canon_c(delta_blk.norm2(x_in), state=None, return_state=False)
        g_, u_ = delta_blk.w_gate_up(n2).chunk(2, dim=-1)
        return x_in + delta_blk.w_down(F.silu(g_) * u_)

    c_1stream = torch.compile(_1stream_mlp)
    ms_1stream_mlp = _time_cuda(_fwbw(lambda: c_1stream(x_hid)))

    # 6. A/B Test: CanonLayer shift-sum vs F.conv1d on [B, T, 1536] under torch.compile
    x3 = torch.randn(B, T, 3 * D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    cb = delta_blk.mixer.canon_b
    c_canon_shift = torch.compile(lambda z: cb(z, state=None, return_state=False)[0])
    def _canon_conv1d(z):
        zt = z.transpose(1, 2)
        d = F.conv1d(zt, cb.conv.weight, padding=3, groups=cb.dim)[:, :, :z.shape[1]].transpose(1, 2)
        return F.silu(z + d)
    c_canon_conv = torch.compile(_canon_conv1d)
    ms_canon_shift = _time_cuda(_fwbw(lambda: c_canon_shift(x3)))
    ms_canon_conv = _time_cuda(_fwbw(lambda: c_canon_conv(x3)))

    # 7. Semantic Core + Top-Down + Local Update + LM Head
    def _sem_core_full():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = p30._compiled_pool(x_hid)
            zt, ie, fe, st_t, settled = p30._relax_concept_frames(z, max_steps=2, use_fast_weights=False)
            _, prior, _ = p30.fast_ssm.scan(zt)
            z_pred = p30.causal_layer.predict_down(prior.detach().float(), use_fast_weights=False).to(x_hid.dtype)
            bos_c = p30.bos_concept_prior.expand(B, 1, -1).to(x_hid.dtype)
            bos_t = x_hid.new_zeros((B, 1, p30.config.top_dim))
            ctx_c = torch.cat([bos_c, z_pred.detach()], dim=1)[:, :64, :]
            ctx_t = torch.cat([bos_t, prior], dim=1)[:, :64, :]
            ctx_tok = torch.cat([ctx_c, ctx_t], dim=-1).repeat_interleave(8, dim=1)
            td = p30.top_down_norm(p30.concept_to_stream(ctx_tok))
            return x_st + torch.tanh(p30.top_down_gate) * td.unsqueeze(-2)

    ms_sem_core = _time_cuda(_fwbw(_sem_core_full))

    # Local update x1 (field + causal + critic)
    p_opt = torch.optim.AdamW(p30.io_parameters(), lr=6e-4)
    p30.train_step_hybrid_local(toks, p_opt, use_amp_bf16=True, grad_accum_steps=1, update_radar_every=False, observe_prior=False)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        z_d = p30._pool_mid_to_concepts(x_hid)
        zt_d, _, _, _, settled_d = p30._relax_concept_frames(z_d, max_steps=2, use_fast_weights=False)
        _, prior_d, _ = p30.fast_ssm.scan(zt_d)
        u_prev = prior_d[:, :-1].float().reshape(-1, p30.config.top_dim)
        z_next = z_d[:, 1:].float().reshape(-1, p30.config.concept_dim)
        y_tgt = toks[:, 1:]
        e_true = p30.tok_emb(y_tgt).detach().float()
        e_hard = p30.tok_emb(torch.roll(y_tgt, 1, 0)).detach().float()

    def _local_update_all():
        p30.local_opt.step_field(p30.field, settled_d, lr=0.045, async_order=True, use_fast_weights=False)
        p30.contrastive_critic.update_wake_sleep_local(
            u_ctx=u_prev, z_true=z_next, z_hard_neg=torch.roll(z_next, 1, 0),
            h_tok_ctx=x_hid.detach().float(), e_tok_true=e_true, e_tok_hard_neg=e_hard, lr=0.045
        )
    ms_local_all = _time_cuda(_local_update_all)

    def _lm_head():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = F.linear(p30.emb_out(p30.norm_f(x_hid)), p30.tok_emb.weight)
            loss = F.cross_entropy(logits.reshape(-1, 50257), toks[:, 1:].reshape(-1))
        loss.backward()
    ms_lm_head = _time_cuda(_lm_head)

    print("\n================ PULSE-v3.1-TURBO MICROBATCH AUTOPSY (B=16, T=512) ================", flush=True)
    print(f"TF-Compiled 1 block            : {ms_tf_blk:6.2f} ms  -> x6 = {ms_tf_blk*6:6.1f} ms", flush=True)
    print(f"PULSE-v3.1 Delta blk (4-graph) : {ms_delta_blk:6.2f} ms  -> x4 = {ms_delta_blk*4:6.1f} ms", flush=True)
    print(f"  -- pre1 (mhc1+canon_a)       : {ms_d_pre1:6.2f} ms", flush=True)
    print(f"  -- mixer_pre (in+cb+gates)   : {ms_d_mpre:6.2f} ms", flush=True)
    print(f"  -- FLA chunk_kda (no state)  : {ms_kda:6.2f} ms  (vs chunk_gdn: {ms_gdn:5.2f} ms)", flush=True)
    print(f"  -- mixer_post (norm+g+o)     : {ms_d_mpost:6.2f} ms", flush=True)
    print(f"  -- post1_mlp (mhc+cc+SwiGLU) : {ms_d_post1_mlp:6.2f} ms  (vs 1-stream SwiGLU: {ms_1stream_mlp:5.2f} ms)", flush=True)
    print(f"  ** A/B Merged 2-graph Delta  : {ms_delta_merged_kda:6.2f} ms  (SAVES {(ms_delta_blk - ms_delta_merged_kda)*4:5.1f} ms/mb!)", flush=True)
    print(f"PULSE-v3.1 SWA-128 blk         : {ms_swa_blk:6.2f} ms  -> x2 = {ms_swa_blk*2:6.1f} ms", flush=True)
    print(f"  -- swa_pre={ms_s_mpre:5.2f}ms | FlexAttn={ms_flex:5.2f}ms (SDPA={ms_sdpa:5.2f}ms) | swa_post={ms_s_mpost:5.2f}ms", flush=True)
    print(f"A/B Canon-B (1536ch) shift-sum : {ms_canon_shift:6.2f} ms  vs F.conv1d: {ms_canon_conv:5.2f} ms", flush=True)
    print(f"Semantic Core + TopDown (FW+BW): {ms_sem_core:6.2f} ms", flush=True)
    print(f"Local Field + Critic (x1/step) : {ms_local_all:6.2f} ms", flush=True)
    print(f"LM Head + CE Loss              : {ms_lm_head:6.2f} ms", flush=True)
    print("===================================================================================", flush=True)

    # 8. N=4096 Inference Breakdown
    p30.eval(); t30.eval()
    tok4k = torch.randint(4, 50257, (1, 4096), device=dev)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        ms_tf_4k = _time_cuda(lambda: t30(tok4k, use_cache=True))
        ms_p_4k = _time_cuda(lambda: p30(tok4k, use_fast_weights=False, max_think_steps=1))
        x4k = p30.blocks[0].mhc1.expand_input(p30.emb_in(p30.tok_emb(tok4k)))
        ms_4k_d = _time_cuda(lambda: delta_blk(x4k, return_state=True))
        ms_4k_s = _time_cuda(lambda: swa_blk(x4k, return_state=True))
        h4k = x4k.mean(-2)
        def _core_4k():
            z = p30._compiled_pool(h4k)
            zt, _, _, _, _ = p30._relax_concept_frames(z, max_steps=1, use_fast_weights=False)
            return p30.fast_ssm.scan(zt)
        ms_4k_core = _time_cuda(_core_4k)
        ms_4k_lm = _time_cuda(lambda: F.linear(p30.emb_out(p30.norm_f(h4k)), p30.tok_emb.weight))
    print(f"\n[N=4096 AUTOPSY] TF={ms_tf_4k:5.2f}ms | PULSE-v3.1={ms_p_4k:5.2f}ms", flush=True)
    print(f"  1x Delta={ms_4k_d:5.2f}ms (x4={ms_4k_d*4:5.1f}ms) | 1x SWA={ms_4k_s:5.2f}ms (x2={ms_4k_s*2:5.1f}ms) | Core={ms_4k_core:5.2f}ms | LMHead={ms_4k_lm:5.2f}ms", flush=True)


if __name__ == "__main__":
    main()
