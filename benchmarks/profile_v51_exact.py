"""15-Second L4 Exact Breakdown + A/B Ablation Profiler for PULSE-v5.1 (318.9ms -> <200ms!)."""

import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pulse.experiment import FairMuonAdamW
from pulse.pulse_v4 import PULSEv4Config, PULSEv4Model, _canon4_inductor, _unit_rms
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM


def bench(name, fn, n_warm=3, n_it=8):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n_it):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[P51] {name:46s} | med={np.median(ts):6.2f} ms | peak={mb:5.0f} MB", flush=True)


def main():
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    b = torch.randint(4, 50257, (16, 513), device=dev)
    x_in, y_tgt = b[:, :-1], b[:, 1:]

    p5 = PULSEv4Model(PULSEv4Config.colab_30m()).to(dev)
    p5.compile_fast_paths()
    tf = ScalableTransformerLM(ScalableTransformerConfig.colab_30m()).to(dev)
    tf.compile_fast_paths()

    # 1. End-to-End 1-micro Fwd-Only vs Fwd+Bwd (NO core, NO opt)
    def p5_fwd():
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p5(x_in, use_fast_weights=False, max_think_steps=1, return_state=False, run_core=False)

    def tf_fwd():
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tf(x_in, use_cache=False)

    def p5_fb():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = p5(x_in, use_fast_weights=False, max_think_steps=1, return_state=False, run_core=False)
            loss = F.cross_entropy(out["logits"].reshape(-1, 50257), y_tgt.reshape(-1))
        loss.backward()
        p5.zero_grad(set_to_none=True)

    def tf_fb():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = tf.compute_loss(b)
        loss.backward()
        tf.zero_grad(set_to_none=True)

    bench("1a. PULSE-v5.1 1-micro FWD ONLY", p5_fwd)
    bench("1b. TF-Compiled 1-micro FWD ONLY", tf_fwd)
    bench("1c. PULSE-v5.1 1-micro FWD+BWD (no core)", p5_fb)
    bench("1d. TF-Compiled 1-micro FWD+BWD", tf_fb)

    # 2. Optimizer step cost: Batched Muon PULSE vs TF
    p_opt = FairMuonAdamW(p5.io_parameters(), lr=6e-4, weight_decay=0.1)
    t_opt = FairMuonAdamW(list(tf.parameters()), lr=6e-4, weight_decay=0.1)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        F.cross_entropy(p5(x_in, use_fast_weights=False, max_think_steps=1, return_state=False, run_core=False)["logits"].reshape(-1, 50257), y_tgt.reshape(-1)).backward()
        tf.compute_loss(b).backward()
    bench("2a. PULSE-v5.1 FairMuonAdamW.step()", lambda: p_opt.step())
    bench("2b. TF-Compiled FairMuonAdamW.step()", lambda: t_opt.step())
    p5.zero_grad(set_to_none=True)
    tf.zero_grad(set_to_none=True)

    # 3. Full train_step_hybrid_local WITH vs WITHOUT do_core
    p5._step_counter = 100  # skips core except %4==1
    def p5_step_nocore():
        p5._step_counter = 101  # after +1 becomes 102 (%4!=1) -> NO core
        p5.train_step_hybrid_local(b, p_opt, use_amp_bf16=True, grad_accum_steps=2, is_first_micro=True, is_last_micro=False)
        p5.train_step_hybrid_local(b, p_opt, use_amp_bf16=True, grad_accum_steps=2, is_first_micro=False, is_last_micro=True)

    def p5_step_withcore():
        p5._step_counter = 100  # after +1 becomes 101 (%4==1) -> WITH core
        p5.train_step_hybrid_local(b, p_opt, use_amp_bf16=True, grad_accum_steps=2, is_first_micro=True, is_last_micro=False)
        p5.train_step_hybrid_local(b, p_opt, use_amp_bf16=True, grad_accum_steps=2, is_first_micro=False, is_last_micro=True)

    bench("3a. Full 16x2 step (NO core)", p5_step_nocore)
    bench("3b. Full 16x2 step (WITH core)", p5_step_withcore)

    # 4. Per-block FWD+BWD breakdown across depth!
    xs = [torch.randn(16, 512, 512, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(6)]
    xhs = [_unit_rms(t.detach()).requires_grad_(True) for t in xs]
    e = torch.randn(16, 512, 192, device=dev, dtype=torch.bfloat16)
    v0 = torch.randn(16, 512, 4, 64, device=dev, dtype=torch.bfloat16)
    cos, sin = p5._rope(0, 512, dev, torch.bfloat16)

    def blk_bench(label, fn):
        def run():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                y = fn()
            y.sum().backward()
            for t in xs + xhs:
                t.grad = None
        bench(label, run)

    blk_bench("4a. DeltaBlock[0] (L=1) FWD+BWD", lambda: p5.blocks[0](xs[:1], e, return_state=False, history_hat=xhs[:1])[0])
    blk_bench("4b. DeltaBlock[3] (L=4) FWD+BWD", lambda: p5.blocks[3](xs[:4], e, return_state=False, history_hat=xhs[:4])[0])
    blk_bench("4c. SWABlock[1]   (L=2, no v0) FWD+BWD", lambda: p5.blocks[1](xs[:2], e, None, cos, sin, None, return_state=False, history_hat=xhs[:2])[0])
    blk_bench("4d. SWABlock[5]   (L=6, w/ v0) FWD+BWD", lambda: p5.blocks[5](xs[:6], e, v0, cos, sin, None, return_state=False, history_hat=xhs[:6])[0])
    blk_bench("4e. Transformer Block[0] FWD+BWD", lambda: tf.blocks[0](xs[0], cos, sin, None, False)[0])

    # 5. Micro-op A/B inside torch.compile: Canon-A/B implementations!
    x = torch.randn(16, 512, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
    qkv = torch.randn(16, 512, 1024, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w4_512 = nn.Parameter(torch.randn(4, 512, device=dev, dtype=torch.bfloat16) * 0.02)
    w3_512 = nn.Parameter(torch.randn(3, 512, device=dev, dtype=torch.bfloat16) * 0.02)
    w4_1024 = nn.Parameter(torch.randn(4, 1024, device=dev, dtype=torch.bfloat16) * 0.02)
    conv4_1024 = nn.Conv1d(1024, 1024, kernel_size=4, groups=1024, padding=0, bias=False).to(dev, torch.bfloat16)

    def ms_canon_pad(z, wl, wd):
        T = z.shape[1]
        zp = F.pad(z, (0, 0, 12, 0))
        return (
            z * (wl[3] + 1.0)
            + zp[:, 11:11+T] * wl[2] + zp[:, 10:10+T] * wl[1] + zp[:, 9:9+T] * wl[0]
            + zp[:, 8:8+T] * wd[0] + zp[:, 4:4+T] * wd[1] + zp[:, 0:T] * wd[2]
        )

    def c4_conv1d(z):
        return z + conv4_1024(F.pad(z.transpose(1, 2), (3, 0))).transpose(1, 2)

    c_ms_pad = torch.compile(ms_canon_pad, dynamic=True)
    c_c4_ind = torch.compile(lambda z, w: _canon4_inductor(z, w, False), dynamic=True)
    c_c4_conv = torch.compile(c4_conv1d, dynamic=True)
    c_c4_triton = torch.compile(lambda z, w: torch.ops.pulse.triton_canon4_v2_fwd(z, w, False), dynamic=True)

    bench("5a. MS-CanonA F.pad compiled (512)", lambda: (c_ms_pad(x, w4_512, w3_512).sum().backward(), setattr(x, "grad", None), setattr(w4_512, "grad", None), setattr(w3_512, "grad", None)))
    bench("5b. CanonB F.pad compiled (1024)", lambda: (c_c4_ind(qkv, w4_1024).sum().backward(), setattr(qkv, "grad", None), setattr(w4_1024, "grad", None)))
    bench("5c. CanonB Conv1d compiled (1024)", lambda: (c_c4_conv(qkv).sum().backward(), setattr(qkv, "grad", None), conv4_1024.zero_grad(set_to_none=True)))
    bench("5d. CanonB Triton custom_op (1024)", lambda: (c_c4_triton(qkv, w4_1024).sum().backward(), setattr(qkv, "grad", None), setattr(w4_1024, "grad", None)))

    # 6. Check AttnRes(L=6) cost alone inside torch.compile!
    q_ar = nn.Parameter(torch.randn(512, device=dev, dtype=torch.bfloat16) * 0.02)
    def ar_fn(h0, h1, h2, h3, h4, h5, u0, u1, u2, u3, u4, u5, q):
        hs = (h0, h1, h2, h3, h4, h5)
        us = (u0, u1, u2, u3, u4, u5)
        sc = torch.stack([(u * q).sum(dim=-1) for u in us], dim=-1)
        w = F.softmax(sc.float(), dim=-1).to(h0.dtype)
        out = w[..., 0:1] * hs[0]
        for i in range(1, 6):
            out = out + w[..., i:i+1] * hs[i]
        return out
    c_ar = torch.compile(ar_fn, dynamic=True)
    bench("6a. AttnRes(L=6) compiled FWD+BWD", lambda: (c_ar(*xs, *xhs, q_ar).sum().backward(), [setattr(t, "grad", None) for t in xs + xhs], setattr(q_ar, "grad", None)))

    # 7. Check ReLU^2(2176) vs SwiGLU(1408) inside torch.compile!
    w_up_r = nn.Linear(512, 2176, bias=False).to(dev, torch.bfloat16)
    w_dn_r = nn.Linear(2176, 512, bias=False).to(dev, torch.bfloat16)
    w_g_s = nn.Linear(512, 1408, bias=False).to(dev, torch.bfloat16)
    w_u_s = nn.Linear(512, 1408, bias=False).to(dev, torch.bfloat16)
    w_d_s = nn.Linear(1408, 512, bias=False).to(dev, torch.bfloat16)
    c_relu2 = torch.compile(lambda z: w_dn_r(F.relu(w_up_r(z)).square()), dynamic=True)
    c_swiglu = torch.compile(lambda z: w_d_s(F.silu(w_g_s(z)) * w_u_s(z)), dynamic=True)
    bench("7a. ReLU^2 (2176) compiled FWD+BWD", lambda: (c_relu2(x).sum().backward(), setattr(x, "grad", None), w_up_r.zero_grad(set_to_none=True), w_dn_r.zero_grad(set_to_none=True)))
    bench("7b. SwiGLU (1408) compiled FWD+BWD", lambda: (c_swiglu(x).sum().backward(), setattr(x, "grad", None), w_g_s.zero_grad(set_to_none=True), w_u_s.zero_grad(set_to_none=True), w_d_s.zero_grad(set_to_none=True)))


if __name__ == "__main__":
    main()
