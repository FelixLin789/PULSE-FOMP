"""Week-3 PULSE-v8 L4 sm_89 Fused Triton 64-Slot Verification, Speed Benchmark (3M & 30M) & Gate-3 Check.

Executes three ruthless hardware & correctness verifications in ~3-6 min (~0.25-0.5 CCU) on NVIDIA L4:
  1. Check 1 — Strict Numerical Parity (Fused sm_89 Triton Fwd+Bwd vs Exact PyTorch Reference):
       * Tests 3M (D=32, T=256), non-aligned (D=32, T=137), and 30M Stage B (D=64, T=256) in BOTH IEEE & TF32.
       * Asserts exact match on output `O`, `a_mean`, and ALL 7 gradients `(dQ, dR, dF, dGamma, dK0, dInvTau, dBFree)`.
  2. Check 2 — Real L4 Speedup & Full-Model Wall-Clock Ratio vs Compiled `TF-Modern`:
       * Isolated 64-Slot Fwd+Bwd speedup (PyTorch eager vs Fused sm_89 Triton) at 3M L=256, 30M L=1024, 30M L=4096.
       * End-to-end Fwd+Bwd+Opt step time: `PULSE-v8` (compiled + Triton) vs matched `TF-Modern` (compiled)
         at 3M L=256, 30M L=1024, and 30M L=4096!
  3. Check 3 — Gate-3 `P4 No-Hash Rule-2Hop` on Unified `PULSE-v8` (`L0=SWA-32` + Fused Triton 64-Slot + FOMP-v2.2):
       * Verifies `L0=SWA` fixes GDN 2-token prefix blurring (`VA -> 95-100%`, `M1/M16 2-Hop >= 85%`, `bg_drift <= 1.0 pp`).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW
from pulse.pulse_v7 import patch_fla_sm89_shared_memory
from pulse.pulse_v8 import PULSEv8Config, PULSEv8Model
from pulse.triton_slot_register import (
    fused_slot_register_scan,
    has_triton_slot_kernel,
    pytorch_slot_register_scan,
)
from pulse.whiteboard_anvil import TFModernConfig, TFModernLM, audit_zero_hidden_params

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from pulse_v8_gate_check import run_gate3_p4_fomp_v22  # noqa: E402
from v7_stage0_stage1 import save_json, sync  # noqa: E402


def _check_numerical_parity(device: torch.device, smoke: bool) -> Dict[str, Any]:
    cases = [(2, 4, 37, 16, 16, 16)] if smoke else [
        (2, 8, 256, 64, 32, 32),   # 3M exact
        (2, 8, 137, 64, 32, 32),   # non-multiple-of-32 padding test
        (2, 6, 256, 64, 64, 32),   # 30M Stage B exact (D=64 -> 32 KiB SMEM!)
    ]
    report: Dict[str, Any] = {"triton_cuda_active": has_triton_slot_kernel(), "cases": []}
    all_ok = True
    for (B, H, T, S, D, C) in cases:
        for use_tf32 in ((False,) if smoke else (False, True)):
            # Match PyTorch's FP32 matmul mode to the Triton precision being audited (IEEE vs TF32)!
            if device.type == "cuda":
                torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
            torch.manual_seed(123)
            q0 = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1)
            r0 = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1)
            f0 = torch.randn(B, H, T, D, device=device)
            g0 = torch.sigmoid(torch.randn(B, H, T, device=device))
            k0 = F.normalize(torch.randn(1, H, S, D, device=device), dim=-1) * 0.25
            it0 = torch.full((H,), 6.5, device=device)
            bf0 = torch.full((H,), 1.5, device=device)
            go = torch.randn(B, H, T, D, device=device)
            ga = torch.randn(B, H, S, device=device) * 0.1

            # PyTorch reference
            t_ref = [x.detach().clone().requires_grad_(True) for x in (q0, r0, f0, g0, k0, it0, bf0)]
            o_ref, a_ref = pytorch_slot_register_scan(*t_ref, chunk_size=C)
            ((o_ref * go).sum() + (a_ref * ga).sum()).backward()

            # Fused Triton kernel (or fallback on CPU)
            t_tri = [x.detach().clone().requires_grad_(True) for x in (q0, r0, f0, g0, k0, it0, bf0)]
            o_tri, a_tri = fused_slot_register_scan(*t_tri, chunk_size=C, use_tf32=use_tf32)
            ((o_tri * go).sum() + (a_tri * ga).sum()).backward()

            diffs = {
                "O": float((o_tri - o_ref).abs().max().item()),
                "a_mean": float((a_tri - a_ref).abs().max().item()),
            }
            names = ("dQ", "dR", "dF", "dGamma", "dK0", "dInvTau", "dBFree")
            for nm, xr, xt in zip(names, t_ref, t_tri):
                assert xr.grad is not None and xt.grad is not None
                denom = max(1.0, float(xr.grad.abs().max().item()))
                diffs[nm] = float((xt.grad - xr.grad).abs().max().item() / denom)
            tol = 5e-3 if use_tf32 else 2e-4
            passed = all(v <= tol for v in diffs.values())
            all_ok = all_ok and passed
            rec = {
                "shape": f"B={B},H={H},T={T},S={S},D={D},C={C}",
                "prec": "tf32" if use_tf32 else "ieee",
                "max_rel_diffs": {k: round(v, 6) for k, v in diffs.items()},
                "tol": tol,
                "passed": passed,
            }
            report["cases"].append(rec)
            print(f"[Parity {rec['shape']} {rec['prec']}] PASS={passed} diffs={rec['max_rel_diffs']}", flush=True)
            assert passed, f"Triton numerical parity FAILED: {rec}"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    report["all_passed"] = all_ok
    return report


def _bench_slot_layer(B: int, H: int, T: int, S: int, D: int, C: int, device: torch.device, smoke: bool) -> Dict[str, Any]:
    q = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1)
    r = F.normalize(torch.randn(B, H, T, D, device=device), dim=-1)
    f = torch.randn(B, H, T, D, device=device)
    g = torch.sigmoid(torch.randn(B, H, T, device=device))
    k0 = F.normalize(torch.randn(1, H, S, D, device=device), dim=-1) * 0.25
    it = torch.full((H,), 6.5, device=device)
    bf = torch.full((H,), 1.5, device=device)
    n_w, n_m = (1, 3) if smoke else (5, 25)

    def _time_fn(fn) -> float:
        for _ in range(n_w):
            ts = [x.detach().clone().requires_grad_(True) for x in (q, r, f, g, k0, it, bf)]
            o, a = fn(*ts, chunk_size=C)
            (o.pow(2).mean() + 0.01 * a.pow(2).mean()).backward()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n_m):
            ts = [x.detach().clone().requires_grad_(True) for x in (q, r, f, g, k0, it, bf)]
            o, a = fn(*ts, chunk_size=C)
            (o.pow(2).mean() + 0.01 * a.pow(2).mean()).backward()
        sync(device)
        return (time.perf_counter() - t0) * 1000.0 / n_m

    ms_py = _time_fn(pytorch_slot_register_scan)
    ms_tr = _time_fn(fused_slot_register_scan)
    spd = ms_py / max(1e-6, ms_tr)
    rec = {
        "shape": f"B={B},H={H},T={T},S={S},D={D},C={C}",
        "pytorch_eager_ms": round(ms_py, 2),
        "triton_sm89_ms": round(ms_tr, 2),
        "speedup_x": round(spd, 2),
    }
    print(f"[Slot-Kernel {rec['shape']}] PyTorch={rec['pytorch_eager_ms']}ms -> Triton={rec['triton_sm89_ms']}ms ({rec['speedup_x']}x!)", flush=True)
    return rec


def _bench_full_model_step(
    label: str,
    model: nn.Module,
    vocab_size: int,
    bs: int,
    seq_len: int,
    device: torch.device,
    amp: bool,
    smoke: bool,
) -> Dict[str, Any]:
    model = model.to(device)
    if hasattr(model, "compile_fast_paths"):
        model.compile_fast_paths()
    opt = FairMuonAdamW(model.io_parameters(), lr=1.0e-3, muon_lr_mult=6.0, adam_lr_mult=1.5, weight_decay=0.01)
    x = torch.randint(0, vocab_size, (bs, seq_len), device=device)
    y = torch.randint(0, vocab_size, (bs, seq_len), device=device)
    n_w, n_m = (1, 3) if smoke else (5, 20)

    def _step() -> None:
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg = model(x)["logits"].float()
            loss = F.cross_entropy(lg.reshape(-1, vocab_size), y.reshape(-1))
            if hasattr(model, "aux_loss"):
                loss = loss + model.aux_loss  # type: ignore[attr-defined]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.io_parameters(), 1.0)
        opt.step()

    for _ in range(n_w):
        _step()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_m):
        _step()
    sync(device)
    ms = (time.perf_counter() - t0) * 1000.0 / n_m
    tok_s = (bs * seq_len) / (ms / 1000.0)
    params = model.num_parameters() if hasattr(model, "num_parameters") else sum(p.numel() for p in model.parameters())
    fla_calls = sum(int(getattr(b, "_fla_calls", 0)) for b in getattr(model, "blocks", []))
    rec = {
        "label": label, "params": int(params), "bs": bs, "seq_len": seq_len,
        "ms_per_step": round(ms, 2), "tok_per_sec": round(tok_s, 1), "fla_calls": fla_calls,
    }
    print(f"[Full-Model {label} | {params:,} params | B={bs} L={seq_len}] {rec['ms_per_step']} ms/st ({rec['tok_per_sec']:,.0f} tok/s, fla_calls={fla_calls})", flush=True)
    del model, opt, x, y
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-gate3", action="store_true")
    parser.add_argument("--no-auto-3seed", action="store_true")
    parser.add_argument("--deadline", type=float, default=time.time() + 1800)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()

    # Check 0: Verify updated L0=SWA PULSE-v8 zero-hidden-param & strict <= TF parity!
    v8_3m = PULSEv8Model(PULSEv8Config.micro_test(112) if args.smoke else PULSEv8Config.lab_3m(112))
    tf_3m = TFModernLM(TFModernConfig.micro_test(112) if args.smoke else TFModernConfig.lab_3m(112))
    au_3m = audit_zero_hidden_params(v8_3m, 112, v8_3m.config.d_model)
    assert au_3m["pass_zero_hidden_audit"] and (args.smoke or v8_3m.num_parameters() <= tf_3m.num_parameters()), (
        f"PULSE-v8 3M audit/parity failed: {v8_3m.num_parameters()} vs TF {tf_3m.num_parameters()} | {au_3m}"
    )
    del v8_3m, tf_3m

    out_json = args.out / "pulse_v8_triton_bench.json"
    summary: Dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8")) if (out_json.exists() and not args.smoke) else {}
    summary.update({
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sm89_patch": sm89,
        "audit_3m": au_3m,
    })

    # Check 1: Numerical Fwd+Bwd Parity (skip if already passed on same L4 run!)
    if not summary.get("check1_numerical_parity", {}).get("all_passed", False):
        summary["check1_numerical_parity"] = _check_numerical_parity(device, args.smoke)
        save_json(out_json, summary)
    else:
        print("[Check 1] Reusing already-verified 6/6 PASS parity results from", out_json, flush=True)

    # Check 2: Real L4 Speedup (Single Layer + Full 3M & 30M Models!)
    if "check2_slot_layer_speed" not in summary or args.smoke:
        slot_benches = [(4, 4, 64, 16, 16, 16)] if args.smoke else [
            (60, 8, 256, 64, 32, 32),   # 3M P3 shape
            (16, 6, 1024, 64, 64, 32),  # 30M Stage B 1K
            (4, 6, 4096, 64, 64, 32),   # 30M Stage B 4K
        ]
        summary["check2_slot_layer_speed"] = [_bench_slot_layer(*cfg_s, device=device, smoke=args.smoke) for cfg_s in slot_benches]
        save_json(out_json, summary)

    full_benches: List[Dict[str, Any]] = []
    if args.smoke:
        pairs = [("3M_micro_L64", 112, 8, 64, PULSEv8Config.micro_test(112), TFModernConfig.micro_test(112))]
    else:
        tf_30m_cfg = TFModernConfig(vocab_size=32768, d_model=384, n_heads=6, n_layers=12, d_ff=1024, use_rope=False, use_ssmax=True, use_canon_acd=True, use_gated_sdpa=True)
        v8_30m_cfg = PULSEv8Config.stage_b_30m(vocab_size=32768, use_fomp=True, use_slots=True)
        pairs = [
            ("3M_L256", 96, 60, 256, PULSEv8Config.lab_3m(96), TFModernConfig.lab_3m(96)),
            ("30M_L1024", 32768, 16, 1024, v8_30m_cfg, tf_30m_cfg),
            ("30M_L4096", 32768, 4, 4096, v8_30m_cfg, tf_30m_cfg),
        ]
    for tag, vocab, bs_m, sl_m, v8_c, tf_c in pairs:
        r_tf = _bench_full_model_step(f"TF-Modern-{tag}", TFModernLM(tf_c), vocab, bs_m, sl_m, device, amp, args.smoke)
        r_v8 = _bench_full_model_step(f"PULSE-v8-{tag}", PULSEv8Model(v8_c), vocab, bs_m, sl_m, device, amp, args.smoke)
        ratio = round(r_v8["ms_per_step"] / max(1e-3, r_tf["ms_per_step"]), 3)
        full_benches.append({"tag": tag, "tf_modern": r_tf, "pulse_v8": r_v8, "wallclock_ratio_vs_tf": ratio})
        print(f"  ==> [{tag}] PULSE-v8 / TF-Modern wallclock ratio = {ratio}x (params: {r_v8['params']:,} vs {r_tf['params']:,})", flush=True)
    summary["check2_full_model_speed"] = full_benches
    save_json(args.out / "pulse_v8_triton_bench.json", summary)

    # Check 3: Gate-3 P4 No-Hash Rule-2Hop with updated L0=SWA PULSE-v8!
    if not args.skip_gate3 and time.time() < args.deadline - 120:
        summary["check3_p4_l0swa_gate3"] = run_gate3_p4_fomp_v22(
            args.out, device, amp, args.smoke, auto_3seed=(not args.no_auto_3seed), deadline=args.deadline
        )
        save_json(args.out / "pulse_v8_triton_bench.json", summary)

    print("\n=== WEEK-3 TRITON 64-SLOT BENCH & GATE-3 COMPLETED ->", args.out / "pulse_v8_triton_bench.json", "===", flush=True)


if __name__ == "__main__":
    main()
