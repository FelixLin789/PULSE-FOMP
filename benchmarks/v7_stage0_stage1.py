"""PULSE-v7 Stage 0 (Base Cleanup & L4 Speed) + Stage 1 (Rank-Wall 4-Way Showdown) Runner.

Directly implements Section 7.3 of the PULSE-v7 blueprint on NVIDIA L4 (or CPU --smoke test):
  Stage 0 (~1.5 CCU):
    0.1 FLA sm_89 99 KiB shared-memory check/patch (102400 -> 101376) + backend probes
    0.2 Zero-leak O(1) / MLA-192 storage audit at 512 / 4096 / 8192 (BF16)
    0.3 105M compiled BF16 L4 speed calibration (16,384 tok/step) vs matched 104.84M Transformer
    0.4 30M FineWeb-Edu 3-seed (41, 42, 43) ablation reusing cached Drive shard (exact same slices!)
  Stage 1 (~2.0 CCU):
    2L Noisy-Haystack (d_head=32) 5-arm rank-wall showdown:
      TF-FullKV vs Triadic-GDN2(E=4) vs Toaster(dh=128) vs RAM-Net(S=128,Top-8) vs PULSE-v7B(GDN2+NoPE-MLA)
    Evaluated on L=384 K=4..64, L=4096 K=8..64, L=8192 K=8,32, Passkey 4K/8K, L4 step overhead & state bytes.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW, _cosine_lr_multiplier
from pulse.pulse_v6 import PULSEv6Config, PULSEv6Model
from pulse.pulse_v7 import (
    GatedDeltaNet2Block,
    GatedNoPEMLABlock,
    GatedSWABlock,
    PULSEv7Config,
    PULSEv7Model,
    RAMNetMemoryBlock,
    ToasterMemoryBlock,
    patch_fla_sm89_shared_memory,
    _FastRMSNorm,
)
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM
from pulse.v3_blocks import _probe_fla_dplr_backend, _probe_fla_gdn_backend, _probe_fla_kda_backend


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CachedFineWebReader:
    """Zero-download memmap reader for the already-cached FineWeb-Edu Drive shard (0000.{train,val}.u16)."""

    def __init__(self, shard_dir: Path) -> None:
        self.train_path = shard_dir / "0000.train.u16"
        self.val_path = shard_dir / "0000.val.u16"
        if not (self.train_path.exists() and self.val_path.exists()):
            raise FileNotFoundError(f"Cached FineWeb-Edu shards not found in {shard_dir}")
        self.maps = {
            "train": np.memmap(self.train_path, mode="r", dtype="<u2"),
            "val": np.memmap(self.val_path, mode="r", dtype="<u2"),
        }

    def read(self, split: str, start: int, count: int) -> torch.Tensor:
        arr = self.maps[split][start : start + count]
        if len(arr) != count:
            raise EOFError(f"Shard {split} too short for [{start}:{start + count}]")
        return torch.from_numpy(np.array(arr, dtype=np.int64))


def views(tokens: torch.Tensor, batch: int, length: int, accum: int, device: torch.device) -> torch.Tensor:
    return tokens.to(device).unfold(0, length + 1, length).reshape(accum, batch, length + 1)


def make_optim(model: nn.Module) -> FairMuonAdamW:
    params = model.io_parameters() if hasattr(model, "io_parameters") else list(model.parameters())
    return FairMuonAdamW(params, lr=6e-4, weight_decay=0.1)


def train_lm_step(model: nn.Module, opt: FairMuonAdamW, batches: torch.Tensor, step: int, total_steps: int, amp: bool) -> float:
    model.train()
    mult = _cosine_lr_multiplier(step, total_steps, max(1, int(0.03 * total_steps)), 0.1)
    for g in opt.param_groups:
        g["lr"] = 6e-4 * mult
    opt.zero_grad(set_to_none=True)
    losses = []
    for tokens in batches:
        with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=amp):
            loss = model.compute_loss(tokens)
        (loss / len(batches)).backward()
        losses.append(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=amp)
    opt.step()
    return float(torch.stack(losses).mean().item())


@torch.no_grad()
def eval_lm_ppl(model: nn.Module, val_tokens: torch.Tensor, length: int, device: torch.device, amp: bool) -> Dict[str, Any]:
    model.eval()
    windows = val_tokens.unfold(0, length + 1, length)
    rows: List[float] = []
    for chunk in windows.split(4):
        chunk = chunk.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(chunk[:, :-1], return_state=False) if isinstance(model, (PULSEv7Model, PULSEv6Model)) else model(chunk[:, :-1])
            logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), chunk[:, 1:].reshape(-1), reduction="none")
            rows.extend(nll.view(chunk.size(0), -1).mean(dim=1).cpu().tolist())
    return {"ppl": math.exp(float(np.mean(rows))), "mean_nll": float(np.mean(rows)), "window_nll": rows}


def boot_ci(diff: np.ndarray, n: int = 2000, seed: int = 1234) -> List[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ==============================================================================
# Stage 0: FLA sm_89 Patch + Zero-Leak CI + 105M Speed + 30M FineWeb 3-Seed
# ==============================================================================

def run_stage0_backend(out: Path, device: torch.device) -> Dict[str, Any]:
    dest = out / "stage0_backend.json"
    sm89 = patch_fla_sm89_shared_memory()
    probes = {}
    if device.type == "cuda":
        with torch.random.fork_rng(devices=[0]):
            for name, fn in (
                ("gdn_scalar", _probe_fla_gdn_backend),
                ("gdn2_kda", _probe_fla_kda_backend),
                ("gdn2_dplr", _probe_fla_dplr_backend),
            ):
                p = fn(device)
                probes[name] = {k: v for k, v in p.items() if k != "fn"}
    payload = {"sm89_patch": sm89, "fla_probes": probes, "torch": torch.__version__}
    save_json(dest, payload)
    print("[Stage 0.1] Backend & sm_89 patch:", json.dumps(payload, indent=2), flush=True)
    return payload


@torch.no_grad()
def run_stage0_storage(out: Path, device: torch.device, smoke: bool) -> Dict[str, Any]:
    dest = out / "stage0_storage.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    amp = device.type == "cuda"
    lengths = (32, 128, 256) if smoke else (512, 4096, 8192)
    cfgs = (
        {
            "v7a_o1": replace(PULSEv7Config.micro_test(), layer_types=("swa", "gdn2", "gdn2", "swa", "gdn2", "gdn2")),
            "v7b_mla": PULSEv7Config.micro_test(),
        }
        if smoke
        else {
            "v7a_o1_105m": PULSEv7Config.colab_100m_v7a_o1("gdn2_kda"),
            "v7b_5to1_mla_105m": PULSEv7Config.colab_100m_v7b("gdn2_kda"),
        }
    )
    rows = {}
    for name, cfg in cfgs.items():
        m = PULSEv7Model(cfg).to(device).eval()
        per_len = {}
        for slen in lengths:
            tok = torch.randint(4, cfg.vocab_size, (1, slen), device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                st = m(tok, return_state=True, last_logits_only=True)["working_state"]
            lb, bb = m.storage_audit_bytes(st)
            per_len[str(slen)] = {
                "logical_bytes": lb,
                "backing_bytes": bb,
                "mib": round(bb / (1024 * 1024), 4),
                "zero_view_leak": (lb == bb),
            }
            del st, tok
        vals = [per_len[str(s)]["backing_bytes"] for s in lengths]
        rows[name] = {
            "params": m.num_parameters(),
            "lengths": per_len,
            "strict_o1_constant": (len(set(vals)) == 1),
        }
        del m
        if amp:
            gc.collect()
            torch.cuda.empty_cache()
    save_json(dest, rows)
    print("[Stage 0.2] Zero-Leak Storage CI:", json.dumps(rows, indent=2), flush=True)
    return rows


def run_stage0_speed_105m(out: Path, device: torch.device, smoke: bool) -> Dict[str, Any]:
    dest = out / "stage0_speed_105m.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    amp = device.type == "cuda"
    batch, accum, length = (2, 2, 32) if smoke else (8, 4, 512)
    warmup, timed = (2, 4) if smoke else (16, 48)
    vocab = 260 if smoke else 50257

    if smoke:
        specs = [
            ("tf_micro", lambda: ScalableTransformerLM(ScalableTransformerConfig.micro_test())),
            ("v7b_scalar", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), recurrence_mode="gdn_scalar"))),
            ("v7b_kda", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), recurrence_mode="gdn2_kda"))),
            ("v7b_dplr", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), recurrence_mode="gdn2_dplr"))),
        ]
    else:
        specs = [
            ("tf_105m", lambda: ScalableTransformerLM(ScalableTransformerConfig.colab_100m())),
            ("v6_12l_mega_105m", lambda: PULSEv6Model(PULSEv6Config.colab_100m())),
            ("v7b_scalar_105m", lambda: PULSEv7Model(PULSEv7Config.colab_100m_v7b("gdn_scalar"))),
            ("v7b_kda_105m", lambda: PULSEv7Model(PULSEv7Config.colab_100m_v7b("gdn2_kda"))),
            ("v7b_dplr_105m", lambda: PULSEv7Model(PULSEv7Config.colab_100m_v7b("gdn2_dplr"))),
            ("v7a_o1_kda_105m", lambda: PULSEv7Model(PULSEv7Config.colab_100m_v7a_o1("gdn2_kda"))),
        ]

    rng_tok = torch.randint(4, vocab, (accum, batch, length + 1), device=device)
    rows: Dict[str, Any] = {}
    for name, ctor in specs:
        try:
            torch.manual_seed(42)
            model = ctor().to(device)
            if amp and hasattr(model, "compile_fast_paths"):
                model.compile_fast_paths()
            opt = make_optim(model)
            if amp:
                torch.cuda.reset_peak_memory_stats(device)
            times = []
            last_loss = 0.0
            for step in range(warmup + timed):
                sync(device)
                t0 = time.perf_counter()
                last_loss = train_lm_step(model, opt, rng_tok, step, 1000, amp)
                sync(device)
                if step >= warmup:
                    times.append((time.perf_counter() - t0) * 1000.0)
            med_ms = float(statistics.median(times))
            mean_ms = float(statistics.mean(times))
            stride = batch * accum * length
            peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if amp else 0.0
            fla_calls = sum(getattr(m, "_fla_calls", 0) for m in model.modules())
            fla_errs = sorted({getattr(m, "_fla_fallback_reason") for m in model.modules() if getattr(m, "_fla_fallback_reason", None)})
            rows[name] = {
                "params": model.num_parameters(),
                "median_ms_per_step": round(med_ms, 2),
                "mean_ms_per_step": round(mean_ms, 2),
                "tokens_per_sec": round(stride / (med_ms / 1000.0), 1),
                "peak_vram_mib": round(peak_mib, 1),
                "last_loss": round(last_loss, 4),
                "fla_calls": fla_calls,
                "fla_fallback_reasons": fla_errs,
            }
            print(f"[Stage 0.3] {name:18s} params={model.num_parameters():,}  med={med_ms:6.2f} ms/step  tok/s={rows[name]['tokens_per_sec']:8.1f}  VRAM={peak_mib:6.1f} MiB  fla={fla_calls}", flush=True)
            del model, opt
        except Exception as exc:
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
            print(f"[Stage 0.3] {name:18s} ERROR: {rows[name]['error']}", flush=True)
        if amp:
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()

    tf_key = "tf_micro" if smoke else "tf_105m"
    tf_ms = rows.get(tf_key, {}).get("median_ms_per_step")
    v7_candidates = {k: v["median_ms_per_step"] for k, v in rows.items() if k.startswith("v7") and "median_ms_per_step" in v}
    best_v7 = min(v7_candidates, key=v7_candidates.get) if v7_candidates else None
    summary = {
        "batch": batch, "accum": accum, "length": length, "tokens_per_step": batch * accum * length,
        "arms": rows,
        "best_v7_arm": best_v7,
        "best_v7_ms": v7_candidates[best_v7] if best_v7 else None,
        "tf_ms": tf_ms,
        "pass_555ms_redline": (v7_candidates[best_v7] <= 555.0) if (best_v7 and not smoke) else True,
        "faster_than_tf": (v7_candidates[best_v7] <= tf_ms) if (best_v7 and tf_ms) else False,
    }
    save_json(dest, summary)
    return summary


def run_stage0_speed_105m_v2(out: Path, device: torch.device, smoke: bool) -> Dict[str, Any]:
    """Stage 0.3b Speed Fix: Graph-break audit + lean L4 speed calibration at L=512 AND L=2048 (16,384 tok/step)."""
    dest = out / "stage0_speed_105m_v2.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    amp = device.type == "cuda"
    vocab = 260 if smoke else 50257
    warmup, timed = (2, 4) if smoke else (12, 36)

    # 1) Explicit torch._dynamo.explain graph-break audit on the lean flagship!
    gb_dest = out / "stage0_graph_breaks.json"
    if not gb_dest.exists():
        cfg_gb = replace(PULSEv7Config.micro_test(), use_canon_cd=False, logit_softcap=0.0) if smoke else PULSEv7Config.colab_100m_v7b_fast("gdn_scalar", 4)
        m_gb = PULSEv7Model(cfg_gb).to(device).eval()
        tok_gb = torch.randint(4, vocab, (2, 32 if smoke else 512), device=device)
        gb_rep = m_gb.diagnose_graph_breaks(tok_gb)
        save_json(gb_dest, gb_rep)
        print(f"[Stage 0.3b] Graph-break audit: regions={gb_rep['n_regions']}, total_breaks={gb_rep['total_breaks']}", flush=True)
        del m_gb, tok_gb
        if amp:
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()

    if smoke:
        regimes = [("L32", 2, 2, 32)]
        specs = [
            ("tf_micro", lambda: ScalableTransformerLM(ScalableTransformerConfig.micro_test())),
            ("v7b_fast_scalar", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), recurrence_mode="gdn_scalar", use_canon_cd=False, logit_softcap=0.0))),
            ("v7b_fast_kda", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), recurrence_mode="gdn2_kda", use_canon_cd=False, logit_softcap=0.0))),
        ]
    else:
        regimes = [("L512", 8, 4, 512), ("L2048", 2, 4, 2048)]
        specs = [
            ("tf_105m", lambda: ScalableTransformerLM(ScalableTransformerConfig.colab_100m())),
            ("v7b_fast4_scalar_105m", lambda: PULSEv7Model(PULSEv7Config.colab_100m_v7b_fast("gdn_scalar", 4))),
            ("v7b_fast6_scalar_105m", lambda: PULSEv7Model(PULSEv7Config.colab_100m_v7b_fast("gdn_scalar", 6))),
            ("v7b_fast8_scalar_105m", lambda: PULSEv7Model(PULSEv7Config.colab_100m_v7b_fast("gdn_scalar", 8))),
            ("v7b_fast4_kda_105m", lambda: PULSEv7Model(PULSEv7Config.colab_100m_v7b_fast("gdn2_kda", 4))),
        ]

    by_regime: Dict[str, Any] = {}
    for r_name, batch, accum, length in regimes:
        rng_tok = torch.randint(4, vocab, (accum, batch, length + 1), device=device)
        rows: Dict[str, Any] = {}
        for name, ctor in specs:
            try:
                torch.manual_seed(42)
                model = ctor().to(device)
                if amp and hasattr(model, "compile_fast_paths"):
                    model.compile_fast_paths()
                opt = make_optim(model)
                if amp:
                    torch.cuda.reset_peak_memory_stats(device)
                times = []
                last_loss = 0.0
                for step in range(warmup + timed):
                    sync(device)
                    t0 = time.perf_counter()
                    last_loss = train_lm_step(model, opt, rng_tok, step, 1000, amp)
                    sync(device)
                    if step >= warmup:
                        times.append((time.perf_counter() - t0) * 1000.0)
                med_ms = float(statistics.median(times))
                stride = batch * accum * length
                peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if amp else 0.0
                fla_calls = sum(getattr(m, "_fla_calls", 0) for m in model.modules())
                fla_errs = sorted({getattr(m, "_fla_fallback_reason") for m in model.modules() if getattr(m, "_fla_fallback_reason", None)})
                rows[name] = {
                    "params": model.num_parameters(),
                    "median_ms_per_step": round(med_ms, 2),
                    "tokens_per_sec": round(stride / (med_ms / 1000.0), 1),
                    "peak_vram_mib": round(peak_mib, 1),
                    "last_loss": round(last_loss, 4),
                    "fla_calls": fla_calls,
                    "fla_fallback_reasons": fla_errs,
                }
                print(f"[Stage 0.3b:{r_name}] {name:22s} params={model.num_parameters():,}  med={med_ms:6.2f} ms/step  tok/s={rows[name]['tokens_per_sec']:8.1f}  VRAM={peak_mib:6.1f} MiB  fla={fla_calls}", flush=True)
                del model, opt
            except Exception as exc:
                rows[name] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
                print(f"[Stage 0.3b:{r_name}] {name:22s} ERROR: {rows[name]['error']}", flush=True)
            if amp:
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
        tf_key = "tf_micro" if smoke else "tf_105m"
        tf_ms = rows.get(tf_key, {}).get("median_ms_per_step")
        v7_c = {k: v["median_ms_per_step"] for k, v in rows.items() if k.startswith("v7") and "median_ms_per_step" in v}
        best = min(v7_c, key=v7_c.get) if v7_c else None
        by_regime[r_name] = {
            "batch": batch, "accum": accum, "length": length, "tokens_per_step": batch * accum * length,
            "arms": rows, "best_v7_arm": best, "best_v7_ms": v7_c[best] if best else None, "tf_ms": tf_ms,
            "faster_than_tf": (v7_c[best] <= tf_ms) if (best and tf_ms) else False,
        }

    l512 = by_regime.get("L512", {})
    payload = {
        "regimes": by_regime,
        "pass_555ms_redline_L512": (l512.get("best_v7_ms", 9999.0) <= 555.0) if not smoke else True,
    }
    save_json(dest, payload)
    return payload


def run_stage0_fineweb_30m(
    out: Path, shard_dir: Optional[Path], device: torch.device, smoke: bool, deadline: float
) -> Dict[str, Any]:
    dest = out / "stage0_fineweb_30m.json"
    runs_dir = out / "stage0_30m_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda"

    if smoke:
        seeds = (41, 42)
        steps, length, batch, accum = 16, 32, 2, 2
        val_tok = torch.randint(4, 260, (length * 4 + 1,))
        reader = None
        arm_specs = [
            ("v7b_kda_30m", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), recurrence_mode="gdn2_kda"))),
            ("v7b_dplr_30m", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), recurrence_mode="gdn2_dplr"))),
            ("tf_30m", lambda: ScalableTransformerLM(ScalableTransformerConfig.micro_test())),
        ]
    else:
        assert shard_dir is not None
        reader = CachedFineWebReader(shard_dir)
        seeds = (41, 42, 43)
        steps, length, batch, accum = 512, 512, 4, 2  # EXACT same 2,097,152 tok/seed as small_ablation.json!
        val_tok = reader.read("val", 0, length * 64 + 1)
        arm_specs = [
            ("v7b_fast2_scalar_30m", lambda: PULSEv7Model(PULSEv7Config.colab_30m_v7b_fast("gdn_scalar", 2))),
            ("v7b_fast4_scalar_30m", lambda: PULSEv7Model(PULSEv7Config.colab_30m_v7b_fast("gdn_scalar", 4))),
            ("v7b_fast2_kda_30m", lambda: PULSEv7Model(PULSEv7Config.colab_30m_v7b_fast("gdn2_kda", 2))),
            ("v7a_o1_scalar_30m", lambda: PULSEv7Model(PULSEv7Config.colab_30m_v7a_o1("gdn_scalar"))),
            ("tf_30m", lambda: ScalableTransformerLM(ScalableTransformerConfig.colab_30m())),
        ]

    stride = length * batch * accum
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for s_idx, seed in enumerate(seeds):
        for arm_name, ctor in arm_specs:
            p = runs_dir / f"{seed}_{arm_name}.json"
            if p.exists():
                rec_loaded = json.loads(p.read_text(encoding="utf-8"))
                if "error" not in rec_loaded:
                    records.append(rec_loaded)
                    continue
            if time.time() + 45 >= deadline:
                break
            try:
                torch.manual_seed(seed)
                model = ctor().to(device)
                if amp and hasattr(model, "compile_fast_paths"):
                    model.compile_fast_paths()
                opt = make_optim(model)
                step_ms = []
                last_loss = 0.0
                for step in range(steps):
                    if reader is not None:
                        raw = reader.read("train", (s_idx * steps + step) * stride, stride + 1)
                    else:
                        g = torch.Generator().manual_seed(seed * 10000 + step)
                        raw = torch.randint(4, 260, (stride + 1,), generator=g)
                    b = views(raw, batch, length, accum, device)
                    sync(device)
                    t0 = time.perf_counter()
                    last_loss = train_lm_step(model, opt, b, step, steps, amp)
                    sync(device)
                    if not math.isfinite(last_loss):
                        raise FloatingPointError(f"{arm_name} seed={seed} step={step}: {last_loss}")
                    step_ms.append((time.perf_counter() - t0) * 1000.0)
                ev = eval_lm_ppl(model, val_tok, length, device, amp)
                rec = {
                    "seed": seed, "arm": arm_name, "params": model.num_parameters(), "steps": steps,
                    "tokens": steps * stride, "median_ms": round(float(statistics.median(step_ms[min(16, steps // 2):])), 2),
                    "train_last_loss": round(last_loss, 4), "val": ev,
                }
                save_json(p, rec)
                records.append(rec)
                print(f"[Stage 0.4] seed={seed} {arm_name:20s} ValPPL={ev['ppl']:7.2f}  med_ms={rec['median_ms']:6.2f}", flush=True)
                del model, opt
            except Exception as exc:
                err_rec = {"seed": seed, "arm": arm_name, "error": f"{type(exc).__name__}: {exc}"[:600]}
                save_json(p, err_rec)
                errors.append(err_rec)
                print(f"[Stage 0.4] seed={seed} {arm_name:20s} ERROR: {err_rec['error']}", flush=True)
            if amp:
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()

    summary_arms = {}
    for arm_name, _ in arm_specs:
        rs = [r for r in records if r["arm"] == arm_name]
        if not rs:
            continue
        ppls = [r["val"]["ppl"] for r in rs]
        summary_arms[arm_name] = {
            "params": rs[0]["params"],
            "seeds": [r["seed"] for r in rs],
            "ppls": [round(x, 2) for x in ppls],
            "mean_ppl": round(float(statistics.mean(ppls)), 2),
            "mean_median_ms": round(float(statistics.mean(r["median_ms"] for r in rs)), 2),
        }
        if arm_name != "tf_30m":
            tf_rs = {r["seed"]: r for r in records if r["arm"] == "tf_30m"}
            diffs = [
                np.subtract(r["val"]["window_nll"], tf_rs[r["seed"]]["val"]["window_nll"])
                for r in rs
                if r["seed"] in tf_rs
            ]
            if diffs:
                all_d = np.concatenate(diffs)
                summary_arms[arm_name]["delta_nll_vs_tf"] = round(float(all_d.mean()), 4)
                summary_arms[arm_name]["ci95_vs_tf"] = [round(x, 4) for x in boot_ci(all_d)]

    best_v7_ppl = min((v["mean_ppl"] for k, v in summary_arms.items() if k.startswith("v7")), default=9999.0)
    payload = {
        "complete": len(records) == len(seeds) * len(arm_specs),
        "reference_historical_v6_no_attnres_ppl": 483.79,
        "pass_480_ppl": best_v7_ppl <= 480.0,
        "summary": summary_arms,
        "errors": errors,
    }
    save_json(dest, payload)
    print("[Stage 0.4] 30M FineWeb-Edu Summary:", json.dumps(payload, indent=2), flush=True)
    return payload


# ==============================================================================
# Stage 1: 2L Noisy-Haystack Rank-Wall 4-Way Showdown (d_head = 32)
# ==============================================================================

def sample_noisy_mqar(length: int, pairs: int, batch_size: int, pool: int, seed: int):
    rng = np.random.default_rng(seed)
    prefix = min(length - 2 * pairs, max(3 * pairs, 3 * length // 4))
    write_slots = np.arange(0, prefix - 2, 3)
    read_slots = np.arange(prefix, length - 1, 2)
    tokens = rng.integers(4 + 2 * pool, 4 + 3 * pool, (batch_size, length), dtype=np.int64)
    qry = np.empty((batch_size, pairs), dtype=np.int64)
    tgt = np.empty_like(qry)
    gap = np.empty_like(qry)
    vpos = np.empty_like(qry)
    kt = np.empty_like(qry)
    vt = np.empty_like(qry)
    for i in range(batch_size):
        keys = rng.choice(pool, pairs, replace=False) + 4
        vals = rng.choice(pool, pairs, replace=False) + 4 + pool
        writes = rng.choice(write_slots, pairs, replace=False)
        reads = np.sort(rng.choice(read_slots, pairs, replace=False))
        order = rng.permutation(pairs)
        tokens[i, writes], tokens[i, writes + 1], tokens[i, writes + 2] = 1, keys, vals
        tokens[i, reads], tokens[i, reads + 1] = 2, keys[order]
        qry[i], tgt[i] = reads + 1, vals[order]
        gap[i] = reads + 1 - (writes[order] + 2)
        vpos[i], kt[i], vt[i] = writes + 2, keys, vals
    return (
        torch.from_numpy(tokens), torch.from_numpy(qry), torch.from_numpy(tgt),
        torch.from_numpy(vpos), torch.from_numpy(kt), torch.from_numpy(vt), gap,
    )


def sample_noisy_passkey(length: int, depth_frac: float, episodes: int, pool: int, seed: int):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(4 + 2 * pool, 4 + 3 * pool, (episodes, length), dtype=np.int64)
    keys = rng.integers(4, 4 + pool, size=episodes)
    vals = rng.integers(4 + pool, 4 + 2 * pool, size=episodes)
    wpos = max(0, min(length - 8, int(length * depth_frac)))
    tokens[:, wpos], tokens[:, wpos + 1], tokens[:, wpos + 2] = 1, keys, vals
    tokens[:, -2], tokens[:, -1] = 2, keys
    qry = torch.full((episodes, 1), length - 1, dtype=torch.long)
    tgt = torch.from_numpy(vals).unsqueeze(1).long()
    return torch.from_numpy(tokens).long(), qry, tgt, (length - 1) - (wpos + 2)


class Stage1LabModel(nn.Module):
    """Unified 2L (d_model=128, n_heads=4, d_head=32) lab wrapper for all 4 Stage-1 rank-wall candidates."""

    def __init__(self, kind: Literal["triadic_e4", "toaster_128", "ramnet_128", "v7b_nope_mla"], vocab: int = 256):
        super().__init__()
        self.kind = kind
        self.d_model = 128
        self.head_dim = 32
        self.tok_emb = nn.Embedding(vocab, 128)
        self.emb_norm = _FastRMSNorm(128)

        if kind == "v7b_nope_mla":
            # Exact Section 7.3 Stage 1 Route B 2L config: 1 GDN-2 + 1 Gated NoPE-MLA (d_latent=32)!
            self.layer0: nn.Module = GatedDeltaNet2Block(
                d_model=128, embed_dim=128, n_heads=4, d_ff=256,
                recurrence_mode="gdn2_dplr", triadic_e=1, gate_rank=32, chunk_size=16, use_canon_acd=True,
            )
            self.layer1: nn.Module = GatedNoPEMLABlock(d_model=128, n_heads=4, d_ff=256, d_latent=32, use_canon_acd=True)
        else:
            # Pure-O(1) Route A candidates: Layer 0 = Gated SWA-64 -> Layer 1 = Rank-Expanded O(1) Memory!
            self.layer0 = GatedSWABlock(
                d_model=128, embed_dim=128, n_heads=4, n_kv_heads=2, d_ff=256, window_size=64, use_canon_acd=True,
            )
            if kind == "triadic_e4":
                self.layer1 = GatedDeltaNet2Block(
                    d_model=128, embed_dim=128, n_heads=4, d_ff=256,
                    recurrence_mode="gdn_scalar", triadic_e=4, chunk_size=16, use_canon_acd=True,
                )
            elif kind == "toaster_128":
                self.layer1 = ToasterMemoryBlock(d_model=128, embed_dim=128, n_heads=4, d_ff=256, d_hidden=128, chunk_size=16)
            elif kind == "ramnet_128":
                self.layer1 = RAMNetMemoryBlock(d_model=128, embed_dim=128, n_heads=4, d_ff=256, num_slots=128, top_k=8, chunk_size=16)
            else:
                raise ValueError(kind)

        self.norm_f = _FastRMSNorm(128)
        self.emb_out = nn.Linear(128, 128, bias=False)
        half = self.head_dim // 2
        self.register_buffer("_inv_freq", 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half)), persistent=False)
        nn.init.normal_(self.tok_emb.weight, 0.0, 1.0 / math.sqrt(128))
        nn.init.zeros_(self.emb_out.weight)

    def _rope(self, length: int, device: torch.device, dtype: torch.dtype):
        pos = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self._inv_freq.to(device))
        half = self.head_dim // 2
        return freqs.cos().to(dtype).view(1, 1, length, half), freqs.sin().to(dtype).view(1, 1, length, half)

    def forward_hidden(self, tokens: torch.Tensor, return_state: bool = False):
        e_tok = self.tok_emb(tokens)
        x = self.emb_norm(e_tok)
        if isinstance(self.layer0, GatedSWABlock):
            cos, sin = self._rope(tokens.size(1), tokens.device, x.dtype)
            x, st0 = self.layer0(x, e_tok, cos, sin, bm=None, state=None, return_state=return_state)
        else:
            x, st0 = self.layer0(x, e_tok, state=None, return_state=return_state)
        if isinstance(self.layer1, GatedNoPEMLABlock):
            x, st1 = self.layer1(x, state=None, return_state=return_state)
        else:
            x, st1 = self.layer1(x, e_tok, return_state=return_state)
        h = self.emb_out(self.norm_f(x))
        return h, ([st0, st1] if return_state else None)

    def query_logits(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h, _ = self.forward_hidden(tokens, return_state=False)
        rows = torch.arange(tokens.size(0), device=tokens.device)[:, None]
        lg = F.linear(h[rows, positions], self.tok_emb.weight)
        return 30.0 * torch.tanh(lg / 30.0)

    def aux_loss(self) -> torch.Tensor:
        return getattr(self.layer1, "last_aux_loss", torch.tensor(0.0, device=self.tok_emb.weight.device))


def run_stage1_rank_wall(out: Path, device: torch.device, smoke: bool, deadline: float) -> Dict[str, Any]:
    dest = out / "stage1_rank_wall.json"
    runs_dir = out / "stage1_arms"
    runs_dir.mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda"
    vocab, pool = 256, 64
    max_steps = 40 if smoke else 1000
    batch_size = 16 if smoke else 64

    arm_specs = [
        ("tf_fullkv", lambda: ScalableTransformerLM(ScalableTransformerConfig(vocab_size=vocab, embed_dim=128, d_model=128, n_heads=4, n_layers=2, d_ff=272, max_seq_len=8192))),
        ("v7b_gdn2_nope_mla", lambda: Stage1LabModel("v7b_nope_mla", vocab=vocab)),
        ("v7a_triadic_e4", lambda: Stage1LabModel("triadic_e4", vocab=vocab)),
        ("v7a_toaster_128", lambda: Stage1LabModel("toaster_128", vocab=vocab)),
        ("v7a_ramnet_128_top8", lambda: Stage1LabModel("ramnet_128", vocab=vocab)),
    ]

    results: Dict[str, Any] = {}
    for name, ctor in arm_specs:
        arm_path = runs_dir / f"{name}.json"
        if arm_path.exists():
            loaded = json.loads(arm_path.read_text(encoding="utf-8"))
            if "error" not in loaded:
                results[name] = loaded
                continue
        if time.time() + 60 >= deadline:
            break
        try:
            torch.manual_seed(42)
            model = ctor().to(device)
            with torch.no_grad():
                nn.init.normal_(model.tok_emb.weight, 0.0, 1.0 / math.sqrt(128))
            opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
            stages = [(32, 4), (64, 6)] if smoke else [(32, 4), (64, 6), (128, 8), (256, 8)]
            st_idx = 0
            t0 = time.perf_counter()
            step_times_256: List[float] = []
            last_loss = 0.0

            for step in range(1, max_steps + 1):
                length, pairs = stages[st_idx]
                toks, qry, tgt, vpos, kt, vt, _ = sample_noisy_mqar(length, pairs, batch_size, pool, 80000 + step * 19)
                toks, qry, tgt, vpos, kt, vt = [x.to(device) for x in (toks, qry, tgt, vpos, kt, vt)]
                model.train()
                for pg in opt.param_groups:
                    pg["lr"] = 3e-3 * min(1.0, step / 50.0)
                sync(device)
                tk = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                pos_all = torch.cat([qry, vpos], dim=1)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    scores = model.query_logits(toks, pos_all)
                    ans_s, rec_s = scores.split(pairs, dim=1)
                    loss = (
                        F.cross_entropy(ans_s.reshape(-1, ans_s.shape[-1]), tgt.reshape(-1))
                        + 0.25 * F.cross_entropy(rec_s[:, :, 4 : 4 + pool].reshape(-1, pool), (kt - 4).reshape(-1))
                        + 0.25 * F.cross_entropy(rec_s[:, :, 4 + pool : 4 + 2 * pool].reshape(-1, pool), (vt - 4 - pool).reshape(-1))
                    )
                    if hasattr(model, "aux_loss"):
                        loss = loss + 0.01 * model.aux_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sync(device)
                dt_ms = (time.perf_counter() - tk) * 1000.0
                if length == 256 or (smoke and st_idx == len(stages) - 1):
                    step_times_256.append(dt_ms)
                last_loss = float(loss.item())

                if step % (20 if smoke else 50) == 0:
                    model.eval()
                    with torch.no_grad():
                        et, eq, eg, _, _, _, _ = sample_noisy_mqar(length, pairs, 24, pool, 1111)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                            acc_cur = float(model.query_logits(et.to(device), eq.to(device)).argmax(-1).cpu().eq(eg).float().mean().item())
                        et2, eq2, eg2, _, _, _, _ = sample_noisy_mqar(256 if not smoke else 64, 8 if not smoke else 6, 24, pool, 2222)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                            acc_dev = float(model.query_logits(et2.to(device), eq2.to(device)).argmax(-1).cpu().eq(eg2).float().mean().item())
                    print(f"[Stage 1:{name:19s}] step={step:3d} ({length:3d}/{pairs}) loss={last_loss:.3f} cur={acc_cur*100:5.1f}% dev={acc_dev*100:5.1f}% ({time.perf_counter()-t0:.1f}s)", flush=True)
                    if st_idx < len(stages) - 1 and acc_cur >= 0.94:
                        st_idx += 1
                    elif st_idx == len(stages) - 1 and acc_dev >= 0.99 and step >= (40 if smoke else 500):
                        break

            model.eval()
            rec: Dict[str, Any] = {
                "params": sum(p.numel() for p in model.parameters()),
                "train_steps": step,
                "train_sec": round(time.perf_counter() - t0, 1),
                "median_ms_at_256": round(float(statistics.median(step_times_256[-20:])), 2) if step_times_256 else None,
                "cap_L384": {}, "cap_L4096": {}, "mqar_8192": {}, "passkey": {}, "state_bytes": {},
            }

            with torch.no_grad():
                rec["fla_calls"] = sum(getattr(m, "_fla_calls", 0) for m in model.modules())
                rec["fla_fallback_reasons"] = sorted({getattr(m, "_fla_fallback_reason") for m in model.modules() if getattr(m, "_fla_fallback_reason", None)})
                for slen in ((32, 128) if smoke else (512, 4096, 8192)):
                    tok_s = torch.randint(4, vocab, (1, slen), device=device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        if hasattr(model, "forward_hidden"):
                            _, sts = model.forward_hidden(tok_s, return_state=True)
                        else:
                            _, sts = model(tok_s, use_cache=True)
                    lb, bb = PULSEv7Model.storage_audit_bytes(sts)
                    rec["state_bytes"][str(slen)] = {"logical": lb, "backing": bb}

                def _eval_grid(eval_L: int, K: int, episodes: int, seed: int) -> Dict[str, float]:
                    bs = 16 if eval_L <= 1024 else (4 if eval_L <= 4096 else 2)
                    hits_all, gaps_all = [], []
                    for s in range(0, episodes, bs):
                        cur = min(bs, episodes - s)
                        tk, qr, tg, _, _, _, gp = sample_noisy_mqar(eval_L, K, cur, pool, seed + s * 29)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                            pr = model.query_logits(tk.to(device), qr.to(device)).argmax(-1).cpu().numpy()
                        hits_all.append(pr == tg.numpy())
                        gaps_all.append(gp)
                    h_arr = np.concatenate(hits_all, axis=0)
                    g_arr = np.concatenate(gaps_all, axis=0)
                    out_m = g_arr >= 64
                    return {
                        "all_pct": round(float(h_arr.mean() * 100.0), 2),
                        "d_ge_64_pct": round(float(h_arr[out_m].mean() * 100.0), 2) if out_m.any() else 0.0,
                    }

                k_384 = (4, 16) if smoke else (4, 8, 16, 24, 32, 48, 64)
                for K in k_384:
                    rec["cap_L384"][str(K)] = _eval_grid(128 if smoke else 384, K, 8 if smoke else 24, 91000 + K)
                if not smoke:
                    for K in (8, 16, 32, 64):
                        rec["cap_L4096"][str(K)] = _eval_grid(4096, K, 8, 95000 + K)
                    for K in (8, 32):
                        rec["mqar_8192"][str(K)] = _eval_grid(8192, K, 6, 98000 + K)
                    for slen in (4096, 8192):
                        pk_row = {}
                        for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
                            tk, qr, tg, dist = sample_noisy_passkey(slen, frac, 16, pool, 70000 + slen + int(frac * 1000))
                            hits = 0
                            for s in range(0, 16, 4):
                                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                                    pr = model.query_logits(tk[s : s + 4].to(device), qr[s : s + 4].to(device)).argmax(-1).cpu()
                                hits += int(pr.eq(tg[s : s + 4]).sum().item())
                            pk_row[f"{int(frac*100)}%(d={dist})"] = round(hits / 16.0 * 100.0, 1)
                        pk_row["mean_pct"] = round(float(statistics.mean(pk_row.values())), 2)
                        rec["passkey"][str(slen)] = pk_row

            save_json(arm_path, rec)
            results[name] = rec
            print(f"[Stage 1 DONE] {name}: L384_K64={rec['cap_L384'].get('64')}, L4096_K64={rec['cap_L4096'].get('64')}, PK8K={rec['passkey'].get('8192', {}).get('mean_pct')}, ms256={rec['median_ms_at_256']}", flush=True)
            del model, opt
        except Exception as exc:
            err_rec = {"error": f"{type(exc).__name__}: {exc}"[:600]}
            save_json(arm_path, err_rec)
            results[name] = err_rec
            print(f"[Stage 1 ERROR] {name}: {err_rec['error']}", flush=True)
        if amp:
            gc.collect()
            torch.cuda.empty_cache()

    verdict = {
        "complete": len(results) == len(arm_specs),
        "arms": results,
    }
    if not smoke and "v7b_gdn2_nope_mla" in results and "error" not in results["v7b_gdn2_nope_mla"]:
        r_b = results["v7b_gdn2_nope_mla"]
        verdict["route_b_passed"] = (
            r_b["cap_L384"]["64"]["all_pct"] >= 94.0 and r_b["passkey"]["8192"]["mean_pct"] >= 98.0
        )
    save_json(dest, verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fineweb-dir", type=Path, default=None)
    parser.add_argument("--deadline", type=float, default=time.time() + 7200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-stage0", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    if not args.skip_stage0:
        summary["stage0_backend"] = run_stage0_backend(args.out, device)
        summary["stage0_storage"] = run_stage0_storage(args.out, device, smoke=args.smoke)
        summary["stage0_speed_105m"] = run_stage0_speed_105m(args.out, device, smoke=args.smoke)
        summary["stage0_speed_105m_v2"] = run_stage0_speed_105m_v2(args.out, device, smoke=args.smoke)
        summary["stage0_fineweb_30m"] = run_stage0_fineweb_30m(args.out, args.fineweb_dir, device, smoke=args.smoke, deadline=args.deadline - 120)
    if not args.skip_stage1 and time.time() < args.deadline - 120:
        summary["stage1_rank_wall"] = run_stage1_rank_wall(args.out, device, smoke=args.smoke, deadline=args.deadline - 60)

    save_json(args.out / "stage0_stage1_summary.json", summary)
    print("\n=== ALL COMPLETED ->", args.out / "stage0_stage1_summary.json", "===", flush=True)


if __name__ == "__main__":
    main()
