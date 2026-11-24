"""15-Second L4 Micro-Profiler: Pinpoints EVERY millisecond of PULSE-v5 (557.8ms -> <195ms!)."""

import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pulse.pulse_v4 import PULSEv4Config, PULSEv4Model
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
    print(f"[PROF] {name:42s} | med={np.median(ts):6.2f} ms | peak={mb:6.0f} MB", flush=True)


def main():
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    b = torch.randint(4, 50257, (16, 513), device=dev)
    x_in, y_tgt = b[:, :-1], b[:, 1:]

    # 1. Profile LM Head + CE WITH vs WITHOUT eager logit_softcap!
    h_emb = torch.randn(16, 512, 192, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w_emb = torch.randn(50257, 192, device=dev, dtype=torch.bfloat16, requires_grad=True)

    def ce_plain():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = F.linear(h_emb, w_emb)
            loss = F.cross_entropy(logits.reshape(-1, 50257), y_tgt.reshape(-1))
        loss.backward()
        h_emb.grad = w_emb.grad = None

    def ce_softcap_eager():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = F.linear(h_emb, w_emb)
            logits = 30.0 * torch.tanh(logits / 30.0)
            loss = F.cross_entropy(logits.reshape(-1, 50257), y_tgt.reshape(-1))
        loss.backward()
        h_emb.grad = w_emb.grad = None

    def _softcap_ce(he, we, yt):
        logits = 30.0 * torch.tanh(F.linear(he, we) / 30.0)
        return F.cross_entropy(logits.reshape(-1, 50257), yt.reshape(-1))

    ce_sc_comp_fn = torch.compile(_softcap_ce)

    def ce_softcap_compiled():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = ce_sc_comp_fn(h_emb, w_emb, y_tgt)
        loss.backward()
        h_emb.grad = w_emb.grad = None

    bench("1a. LM Head + CE (plain, 1 micro)", ce_plain)
    bench("1b. LM Head + CE + EAGER softcap", ce_softcap_eager)
    bench("1c. LM Head + CE + COMPILED softcap", ce_softcap_compiled)

    # 2. Profile SDPA enable_gqa=True vs repeat_interleave vs MHA(8Q/8KV) inside torch.compile!
    q = torch.randn(16, 8, 512, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k4 = torch.randn(16, 4, 512, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v4 = torch.randn(16, 4, 512, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k8 = torch.randn(16, 8, 512, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v8 = torch.randn(16, 8, 512, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)

    sdpa_gqa = torch.compile(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
    sdpa_rep = torch.compile(lambda q, k, v: F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True))
    sdpa_mha = torch.compile(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True))

    bench("2a. SDPA compiled enable_gqa=True (8Q/4KV)", lambda: (sdpa_gqa(q, k4, v4).sum().backward(), setattr(q, "grad", None), setattr(k4, "grad", None), setattr(v4, "grad", None)))
    bench("2b. SDPA compiled repeat_interleave (8Q/4KV)", lambda: (sdpa_rep(q, k4, v4).sum().backward(), setattr(q, "grad", None), setattr(k4, "grad", None), setattr(v4, "grad", None)))
    bench("2c. SDPA compiled MHA (8Q/8KV)", lambda: (sdpa_mha(q, k8, v8).sum().backward(), setattr(q, "grad", None), setattr(k8, "grad", None), setattr(v8, "grad", None)))

    # 3. Profile individual PULSE-v5 blocks vs Transformer block (1 micro fwd+bwd)!
    p5 = PULSEv4Model(PULSEv4Config.colab_30m()).to(dev)
    p5.compile_fast_paths()
    tf = ScalableTransformerLM(ScalableTransformerConfig.colab_30m()).to(dev)
    tf.compile_fast_paths()

    x = torch.randn(16, 512, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
    e = torch.randn(16, 512, 192, device=dev, dtype=torch.bfloat16)
    cos, sin = p5._rope(0, 512, dev, torch.bfloat16)

    def run_delta_blk():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y, _, _, _ = p5.blocks[0]([x], e, return_state=False)
        y.sum().backward()
        x.grad = None

    def run_swa_blk():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y, _, _, _ = p5.blocks[1]([x, x], e, None, cos, sin, None, return_state=False)
        y.sum().backward()
        x.grad = None

    def run_tf_blk():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y, _ = tf.blocks[0](x, cos, sin, None, False)
        y.sum().backward()
        x.grad = None

    bench("3a. PULSE-v5 DeltaBlock (1 micro fwd+bwd)", run_delta_blk)
    bench("3b. PULSE-v5 SWABlock   (1 micro fwd+bwd)", run_swa_blk)
    bench("3c. Transformer Block   (1 micro fwd+bwd)", run_tf_blk)

    # 4. Profile full PULSE-v5 forward+backward WITHOUT softcap vs WITH softcap
    p5.config.logit_softcap = 0.0
    def run_p5_nocap():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = p5(x_in, use_fast_weights=False, max_think_steps=1, return_state=False, run_core=False)
            loss = F.cross_entropy(out["logits"].reshape(-1, 50257), y_tgt.reshape(-1))
        loss.backward()
        p5.zero_grad(set_to_none=True)

    bench("4a. Full PULSE-v5 (no softcap, 1 micro)", run_p5_nocap)
    p5.config.logit_softcap = 30.0
    bench("4b. Full PULSE-v5 (w/ softcap, 1 micro)", run_p5_nocap)
    p5.config.logit_softcap = 0.0

    # 5. Profile N=8192 inference: W=512 vs W=128 FlexAttention (plain LM head, no softcap)!
    tok8k = torch.randint(4, 50257, (1, 8192), device=dev)
    p5.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        bench("5a. PULSE-v5 Infer N=8192 (W=512, state=T)", lambda: p5(tok8k, use_fast_weights=False, max_think_steps=1, return_state=True))
        bench("5b. PULSE-v5 Infer N=8192 (W=512, state=F)", lambda: p5(tok8k, use_fast_weights=False, max_think_steps=1, return_state=False))
        p5.config.echoic_window_size = 128
        for blk in p5.blocks:
            if getattr(blk, "is_echoic", False):
                blk.window_size = 128
        p5._bm_cache.clear()
        bench("5c. PULSE-v5 Infer N=8192 (W=128, state=T)", lambda: p5(tok8k, use_fast_weights=False, max_think_steps=1, return_state=True))


if __name__ == "__main__":
    main()
