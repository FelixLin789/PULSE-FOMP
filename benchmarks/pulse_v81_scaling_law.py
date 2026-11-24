"""PULSE-v8.1 vs TF-Modern Phase 2 Scaling Law Benchmark (125M / 350M, Resumable H100/DDP + E1..E4).

Zero-strawman, strictly parameter-matched (`PULSE-v8.1 <= TF-Modern`) scaling law showdown on FineWeb-Edu:
  - Arms (exact same 2.5B FineWeb-Edu token stream, FairMuonAdamW, L=2048, 524K tok/step, BF16):
      1. `tf_modern` : 12L d=768 H=12 d_ff=2048 NoPE+SSMax+QK-Norm+GatedSDPA+Canon-ACD (123.93M; 16K state 576 MiB)
      2. `pulse_v81` : 12L d=768 H=12 KV=4 d_ff=1920 [2x SWA-256 + 8x FLA GDN + 2x NoPE-MLA-192]
                       (122.52M <= TF; 16K state 14.0 MiB = 41.1x smaller!)
  - Crash-proof resumability: atomic `ckpt_latest.pt` every 250M tokens + deterministic data cursor + DDP support.
  - Paper-grade 4-Suite Evaluation:
      - Suite E1: Scaling Curve (loss every 250M tok) + 2K->4K/8K/16K Extrapolation PPL + Paired 95% CI + 16K Latency/State
      - Suite E2: Real Public Zero-Shot Downstream Benchmarks (LAMBADA, HellaSwag, ARC-Easy, PIQA, WinoGrande)
      - Suite E3: Real-Prose Copy/Induction (2K/4K/8K) + Multi-Key Needle-in-FineWeb Haystack (K=16/32/64 up to 16K)
      - Suite E4: Full-Loop Native `FOMP-v3.2` on trained `PULSE-v8.1` vs `TF-Modern` (Base / ICL / LoRA+Replay) on CounterFact
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW, _cosine_lr_multiplier
from pulse.fomp_v3 import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module
from pulse.pulse_v7 import patch_fla_sm89_shared_memory
from pulse.pulse_v8 import PULSEv8Config, PULSEv8Model
from pulse.whiteboard_anvil import TFModernConfig, TFModernLM, audit_zero_hidden_params

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from fomp_v3_open_edit import (  # noqa: E402
    _eval_m1_episodes,
    _eval_prose_ppl,
    _evaluate_edits,
    _load_real_datasets,
    _make_smoke_datasets,
    _run_lora_replay_arm,
    _train_fomp_v3,
)
from v7_stage0_stage1 import boot_ci, save_json, sync  # noqa: E402


# ==============================================================================
# Distributed Helpers & Eager Eval Guard (Zero Dynamo Recompilation Stalls!)
# ==============================================================================

def _init_dist() -> Tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            return rank, world, local_rank, torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="gloo")
        return rank, world, local_rank, torch.device("cpu")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, dev


def _is_main(rank: int) -> bool:
    return rank == 0


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


@contextmanager
def _eager_eval(model: nn.Module) -> Iterator[None]:
    """Temporarily bypass fixed-shape compiled sub-paths during variable-length evals (zero dynamo stalls)."""
    saved: List[Tuple[nn.Module, str, Any]] = []
    for mod in _unwrap(model).modules():
        for attr in ("_c_fast", "_c_pre", "_c_post", "_compiled_blocks"):
            val = getattr(mod, attr, None)
            if val is not None:
                saved.append((mod, attr, val))
                setattr(mod, attr, None)
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        for mod, attr, val in saved:
            setattr(mod, attr, val)
        if was_training:
            model.train()


class _TiktokenAdapter:
    """Zero-dependency HuggingFace-compatible tokenizer wrapper around fast Rust `tiktoken` (`gpt2`)."""

    def __init__(self, smoke: bool = False) -> None:
        self.smoke = smoke
        if smoke:
            self.enc = None
            self.eos_token_id = 1
            self.pad_token_id = 0
            self.vocab_size = 256
        else:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
            self.eos_token_id = int(self.enc.eot_token)
            self.pad_token_id = int(self.enc.eot_token)
            self.vocab_size = int(self.enc.n_vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if self.smoke:
            toks = [4 + (abs(hash(w)) % 240) for w in text.strip().split()]
            return toks or [4]
        assert self.enc is not None
        ids = self.enc.encode_ordinary(text)
        return ids if ids else [self.eos_token_id]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if self.smoke:
            return " ".join(f"w{i}" for i in ids)
        assert self.enc is not None
        clean = [int(i) for i in ids if not (skip_special_tokens and int(i) == self.eos_token_id)]
        return self.enc.decode(clean)


# ==============================================================================
# Part 1: Fast Multithreaded FineWeb-Edu `uint16` Memmap Sharder (Zero RAM Bloat)
# ==============================================================================

def prepare_fineweb_shards(
    data_dir: Path,
    target_train_tokens: int,
    target_val_tokens: int = 20_000_000,
    smoke: bool = False,
) -> Path:
    """Stream & tokenize HuggingFaceFW/fineweb-edu (sample-10BT) into NVMe `uint16` memmaps once."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tr_path = data_dir / "train.u16"
    va_path = data_dir / "val.u16"
    meta_path = data_dir / "meta.json"

    if tr_path.exists() and va_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("train_tokens", 0) >= target_train_tokens and meta.get("val_tokens", 0) >= target_val_tokens:
            print(f"[Data] Reusing cached FineWeb-Edu shard: {data_dir} ({meta})", flush=True)
            return data_dir

    if smoke:
        rng = np.random.default_rng(42)
        tr_arr = rng.integers(4, 240, size=target_train_tokens, dtype="<u2")
        va_arr = rng.integers(4, 240, size=target_val_tokens, dtype="<u2")
        tr_arr.tofile(tr_path)
        va_arr.tofile(va_path)
        save_json(meta_path, {"train_tokens": int(len(tr_arr)), "val_tokens": int(len(va_arr)), "smoke": True})
        print(f"[Data] Generated smoke shard -> {data_dir} (train={len(tr_arr):,}, val={len(va_arr):,})", flush=True)
        return data_dir

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = int(enc.eot_token)
    n_threads = min(16, max(1, os.cpu_count() or 4))
    print(
        f"[Data] Streaming HuggingFaceFW/fineweb-edu (sample-10BT) -> {data_dir}\n"
        f"       Target: {target_train_tokens/1e9:.2f}B train + {target_val_tokens/1e6:.1f}M val (tiktoken threads={n_threads})",
        flush=True,
    )
    hf_ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    n_files = max(4, min(14, math.ceil((target_train_tokens + target_val_tokens) / 600_000_000) + 1))
    pq_urls = [f"{hf_ep}/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{i:03d}_00000.parquet" for i in range(n_files)]
    ds = load_dataset("parquet", data_files=pq_urls, split="train", streaming=True)

    va_tmp = data_dir / "val.u16.tmp"
    tr_tmp = data_dir / "train.u16.tmp"
    val_written = 0
    tr_written = 0
    t0 = time.perf_counter()
    next_log = 100_000_000
    text_batch: List[str] = []

    with va_tmp.open("wb") as vf, tr_tmp.open("wb") as tf:
        def _flush_batch(texts: List[str]) -> None:
            nonlocal val_written, tr_written, next_log
            encoded = enc.encode_ordinary_batch(texts, num_threads=n_threads)
            flat: List[int] = []
            for seq in encoded:
                flat.extend(seq)
                flat.append(eot)
            arr = np.asarray(flat, dtype="<u2")
            if val_written < target_val_tokens:
                need = target_val_tokens - val_written
                take = arr[:need]
                vf.write(take.tobytes())
                val_written += len(take)
                arr = arr[need:]
            if len(arr) > 0 and tr_written < target_train_tokens:
                need = target_train_tokens - tr_written
                take = arr[:need]
                tf.write(take.tobytes())
                tr_written += len(take)
            if tr_written >= next_log:
                dt = time.perf_counter() - t0
                rate = (val_written + tr_written) / max(1e-3, dt)
                rem = (target_train_tokens - tr_written) / max(1.0, rate)
                print(
                    f"  [Data] tokenized {tr_written/1e9:.3f}B / {target_train_tokens/1e9:.2f}B "
                    f"({rate/1e6:.2f}M tok/s, ETA {rem/60:.1f}m)",
                    flush=True,
                )
                next_log += 100_000_000

        for ex in ds:
            txt = ex.get("text")
            if isinstance(txt, str) and len(txt) >= 64:
                text_batch.append(txt)
                if len(text_batch) >= 512:
                    _flush_batch(text_batch)
                    text_batch.clear()
                    if tr_written >= target_train_tokens:
                        break
        if text_batch and tr_written < target_train_tokens:
            _flush_batch(text_batch)

    os.replace(va_tmp, va_path)
    os.replace(tr_tmp, tr_path)
    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu:sample-10BT",
        "vocab": "gpt2",
        "train_tokens": int(tr_written),
        "val_tokens": int(val_written),
        "prep_sec": round(time.perf_counter() - t0, 1),
    }
    save_json(meta_path, meta)
    print(f"[Data] Shard ready in {meta['prep_sec']:.1f}s -> {meta}", flush=True)
    return data_dir


class MemmapShardReader:
    """Zero-copy NVMe memmap token reader for deterministic, crash-resumable training."""

    def __init__(self, data_dir: Path) -> None:
        self.train = np.memmap(data_dir / "train.u16", mode="r", dtype="<u2")
        self.val = np.memmap(data_dir / "val.u16", mode="r", dtype="<u2")

    def read(self, split: str, start: int, count: int) -> torch.Tensor:
        mp = self.train if split == "train" else self.val
        n = len(mp)
        if start + count <= n:
            arr = np.array(mp[start : start + count], dtype=np.int64)
        else:
            # Clean wrap-around if multi-epoch
            s0 = start % n
            if s0 + count <= n:
                arr = np.array(mp[s0 : s0 + count], dtype=np.int64)
            else:
                p1 = np.array(mp[s0:], dtype=np.int64)
                p2 = np.array(mp[: count - len(p1)], dtype=np.int64)
                arr = np.concatenate([p1, p2], axis=0)
        return torch.from_numpy(arr)


# ==============================================================================
# Part 2: Strictly Parameter-Matched Model Factory & Analytical State Calculator
# ==============================================================================

def build_arm_model(arm: str, scale: str, smoke: bool, device: torch.device) -> nn.Module:
    if smoke:
        if arm == "tf_modern":
            cfg_tf = TFModernConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=6, d_ff=176)
            return TFModernLM(cfg_tf).to(device)
        cfg_p = PULSEv8Config(
            vocab_size=256, d_model=64, n_heads=4, n_kv_heads=2, d_ff=160,
            layer_types=("swa", "gdn", "gdn", "gdn", "gdn", "mla"),
            swa_window_size=32, mla_d_latent=16, use_fomp=False,
        )
        return PULSEv8Model(cfg_p).to(device)

    if scale == "125m":
        return (
            TFModernLM(TFModernConfig.scale_125m(vocab_size=50257)).to(device)
            if arm == "tf_modern"
            else PULSEv8Model(PULSEv8Config.scale_125m(vocab_size=50257, use_fomp=False)).to(device)
        )
    if scale == "350m":
        return (
            TFModernLM(TFModernConfig.scale_350m(vocab_size=50257)).to(device)
            if arm == "tf_modern"
            else PULSEv8Model(PULSEv8Config.scale_350m(vocab_size=50257, use_fomp=False)).to(device)
        )
    raise ValueError(f"Unsupported scale={scale!r}; expected '125m' or '350m'")


def state_mib_at_length(model: nn.Module, seq_len: int) -> float:
    """Exact BF16 inference KV/recurrent state footprint in MiB at sequence length `seq_len`."""
    m = _unwrap(model)
    cfg = m.config
    if isinstance(m, TFModernLM):
        # Full KV cache across all layers: n_layers * 2(K,V) * d_model * L * 2B
        return round(cfg.n_layers * 2 * cfg.d_model * seq_len * 2 / (1024 ** 2), 3)
    assert isinstance(m, PULSEv8Model)
    hd = cfg.d_model // cfg.n_heads
    n_swa = cfg.layer_types.count("swa")
    n_gdn = cfg.layer_types.count("gdn")
    n_mla = cfg.layer_types.count("mla")
    swa_b = n_swa * 2 * (cfg.n_kv_heads * hd) * min(seq_len, cfg.swa_window_size) * 2
    gdn_b = n_gdn * cfg.n_heads * hd * hd * 4  # FP32 recurrent state matrix S_t
    mla_b = n_mla * cfg.mla_d_latent * seq_len * 2  # Compressed c_kv latent (BF16)
    return round((swa_b + gdn_b + mla_b) / (1024 ** 2), 3)


# ==============================================================================
# Part 3: Resumable H100/DDP Pretraining Loop with 250M-Token Scaling Curves
# ==============================================================================

@torch.no_grad()
def eval_val_ppl(
    model: nn.Module,
    reader: MemmapShardReader,
    seq_len: int,
    n_windows: int,
    device: torch.device,
    amp: bool,
    batch_size: int = 2,
) -> Dict[str, Any]:
    need = n_windows * seq_len + 1
    toks = reader.read("val", 0, min(need, len(reader.val)))
    windows = toks.unfold(0, seq_len + 1, seq_len)[:n_windows]
    rows: List[float] = []
    with _eager_eval(model):
        for chunk in windows.split(batch_size):
            ch = chunk.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = _unwrap(model)(ch[:, :-1])["logits"]
                nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), ch[:, 1:].reshape(-1), reduction="none")
            rows.extend(nll.view(ch.size(0), -1).mean(dim=1).cpu().tolist())
    mean_nll = float(np.mean(rows))
    return {
        "ppl": round(math.exp(min(20.0, mean_nll)), 4),
        "bpb": round(mean_nll / math.log(2.0), 5),
        "mean_nll": round(mean_nll, 5),
        "n_windows": len(rows),
        "window_nll": [round(x, 5) for x in rows],
    }


def _save_ckpt_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_arm(
    arm: str,
    scale: str,
    reader: MemmapShardReader,
    out_dir: Path,
    target_tokens: int,
    seq_len: int,
    micro_batch: int,
    grad_accum: int,
    lr: float,
    eval_every_tokens: int,
    seed: int,
    rank: int,
    world: int,
    device: torch.device,
    amp: bool,
    smoke: bool,
    resume: bool,
    preflight_steps: int = 0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    ckpt_final = arm_dir / "ckpt_final.pt"
    ckpt_latest = arm_dir / "ckpt_latest.pt"

    torch.manual_seed(seed)
    if amp:
        torch.cuda.manual_seed_all(seed)
    base_model = build_arm_model(arm, scale, smoke, device)
    audit = audit_zero_hidden_params(base_model, base_model.config.vocab_size, base_model.config.d_model)
    assert audit["pass_zero_hidden_audit"], f"[{arm}] Zero-hidden-param audit failed: {audit}"
    n_params = base_model.num_parameters()

    if resume and preflight_steps == 0 and ckpt_final.exists():
        if _is_main(rank):
            print(f"[{arm}] Found completed final checkpoint -> {ckpt_final}; skipping pretraining.", flush=True)
        st = torch.load(ckpt_final, map_location=device, weights_only=False)
        base_model.load_state_dict(st["model"])
        return base_model, st["train_meta"]

    # Local micro-steps per rank (global_tokens_per_step invariant across 1..N GPUs!)
    assert grad_accum % world == 0, f"grad_accum ({grad_accum}) must be divisible by world_size ({world})"
    local_accum = grad_accum // world
    global_seqs = micro_batch * grad_accum
    global_step_tokens = global_seqs * seq_len
    total_steps = preflight_steps if preflight_steps > 0 else max(1, math.ceil(target_tokens / global_step_tokens))
    warmup_steps = max(1, int(0.03 * total_steps))
    eval_every_steps = max(1, round(eval_every_tokens / global_step_tokens))

    opt = FairMuonAdamW(base_model.io_parameters(), lr=lr, weight_decay=0.1)
    start_step = 0
    tokens_seen = 0
    prior_sec = 0.0
    scaling_curve: List[Dict[str, Any]] = []

    if resume and preflight_steps == 0 and ckpt_latest.exists():
        st = torch.load(ckpt_latest, map_location=device, weights_only=False)
        base_model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step = int(st["step"])
        tokens_seen = int(st["tokens_seen"])
        prior_sec = float(st["elapsed_sec"])
        scaling_curve = list(st.get("scaling_curve", []))
        if _is_main(rank):
            print(
                f"[{arm}] Resuming from step {start_step}/{total_steps} "
                f"({tokens_seen/1e9:.3f}B tokens, prior {prior_sec/60:.1f}m)",
                flush=True,
            )

    if amp and not smoke:
        base_model.compile_fast_paths()
    model: nn.Module = (
        nn.parallel.DistributedDataParallel(base_model, device_ids=[device.index] if amp else None)
        if world > 1
        else base_model
    )

    if _is_main(rank):
        print(
            f"\n=== [{arm}] scale={scale} params={n_params/1e6:.2f}M (audit_ok={audit['pass_zero_hidden_audit']}) | "
            f"target={target_tokens/1e9:.2f}B tok ({total_steps} steps x {global_step_tokens:,} tok/step, L={seq_len}) ===",
            flush=True,
        )

    if amp:
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    t_run0 = time.perf_counter()
    t_steady0 = t_run0
    steady_toks = 0
    t_win = t_run0
    win_toks = 0
    last_loss = 0.0
    log_every = 2 if (smoke or preflight_steps > 0) else max(10, min(50, total_steps // 40))

    for step in range(start_step, total_steps):
        model.train()
        mult = _cosine_lr_multiplier(step, total_steps, warmup_steps, 0.1)
        for pg in opt.param_groups:
            pg["lr"] = lr * mult

        # Deterministic per-step global token slice -> identical across arms & exact on --resume!
        raw = reader.read("train", step * global_step_tokens, global_step_tokens + 1)
        all_seqs = raw.unfold(0, seq_len + 1, seq_len)  # [global_seqs, L+1]
        local_n = micro_batch * local_accum
        my_seqs = all_seqs[rank * local_n : (rank + 1) * local_n].reshape(local_accum, micro_batch, seq_len + 1)

        opt.zero_grad(set_to_none=True)
        step_losses: List[torch.Tensor] = []
        for ai in range(local_accum):
            mb = my_seqs[ai].to(device, non_blocking=True)
            sync_ctx = model.no_sync() if (world > 1 and ai < local_accum - 1) else _null_ctx()
            with sync_ctx:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    loss = model.module.compute_loss(mb) if world > 1 else model.compute_loss(mb)
                (loss / local_accum).backward()
            step_losses.append(loss.detach())

        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0, foreach=amp)
        opt.step()

        last_loss = float(torch.stack(step_losses).mean().item())
        if not math.isfinite(last_loss):
            raise RuntimeError(f"[{arm}] Non-finite loss detected at step {step}: {last_loss}")
        tokens_seen += global_step_tokens
        win_toks += global_step_tokens
        if step == start_step:
            sync(device)
            t_steady0 = time.perf_counter()
        else:
            steady_toks += global_step_tokens

        step1 = step + 1
        if _is_main(rank) and (step1 == 1 or step1 % log_every == 0 or step1 == total_steps):
            sync(device)
            now = time.perf_counter()
            tps = win_toks / max(1e-3, now - t_win)
            elapsed_tot = prior_sec + (now - t_run0)
            eta_h = ((total_steps - step1) * global_step_tokens / max(1.0, tps)) / 3600.0
            print(
                f"  [{arm} st={step1:5d}/{total_steps}] tok={tokens_seen/1e9:.3f}B | "
                f"loss={last_loss:.4f} | lr={lr*mult:.2e} | {tps/1e3:6.1f}K tok/s | "
                f"elapsed={elapsed_tot/60:.1f}m | ETA={eta_h:.2f}h",
                flush=True,
            )
            t_win = now
            win_toks = 0

        # Periodic milestone validation + atomic resumable checkpoint (every ~250M tokens)
        if preflight_steps == 0 and (step1 % eval_every_steps == 0 or step1 == total_steps):
            if world > 1:
                dist.barrier()
            if _is_main(rank):
                sync(device)
                elapsed_tot = prior_sec + (time.perf_counter() - t_run0)
                e_id = eval_val_ppl(base_model, reader, seq_len, n_windows=(4 if smoke else 24), device=device, amp=amp, batch_size=2)
                e_2x = eval_val_ppl(base_model, reader, seq_len * 2, n_windows=(2 if smoke else 12), device=device, amp=amp, batch_size=1)
                pt = {
                    "step": step1,
                    "tokens_seen": tokens_seen,
                    "elapsed_sec": round(elapsed_tot, 1),
                    "train_loss": round(last_loss, 4),
                    f"val_ppl_{seq_len}": e_id["ppl"],
                    f"val_nll_{seq_len}": e_id["mean_nll"],
                    f"val_ppl_{seq_len*2}": e_2x["ppl"],
                }
                scaling_curve.append(pt)
                print(
                    f"  [MILESTONE {arm} @ {tokens_seen/1e9:.3f}B] "
                    f"val_{seq_len}={e_id['ppl']:.2f} | val_{seq_len*2}={e_2x['ppl']:.2f}",
                    flush=True,
                )
                _save_ckpt_atomic(
                    ckpt_latest,
                    {
                        "arm": arm, "scale": scale, "step": step1, "tokens_seen": tokens_seen,
                        "elapsed_sec": elapsed_tot, "scaling_curve": scaling_curve,
                        "model": base_model.state_dict(), "opt": opt.state_dict(),
                    },
                )
                save_json(arm_dir / "scaling_curve.json", scaling_curve)
            if world > 1:
                dist.barrier()

    sync(device)
    now_end = time.perf_counter()
    total_sec = prior_sec + (now_end - t_run0)
    peak_vram_gib = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2) if amp else 0.0
    # True steady-state tok/s excludes step-1 Triton/TileLang/Inductor JIT compile warmup!
    if steady_toks > 0:
        avg_tps = round(steady_toks / max(1e-3, now_end - t_steady0), 1)
    else:
        avg_tps = round(((total_steps - start_step) * global_step_tokens) / max(1e-3, now_end - t_run0), 1)
    train_meta = {
        "arm": arm,
        "scale": scale,
        "n_params": n_params,
        "non_emb_params": n_params - base_model.tok_emb.weight.numel(),
        "tokens_seen": tokens_seen,
        "total_steps": total_steps,
        "total_sec": round(total_sec, 1),
        "avg_tok_per_sec": avg_tps,
        "peak_vram_gib": peak_vram_gib,
        "final_train_loss": round(last_loss, 4),
        "scaling_curve": scaling_curve,
        "state_mib_2k": state_mib_at_length(base_model, 2048),
        "state_mib_16k": state_mib_at_length(base_model, 16384),
    }

    if preflight_steps == 0 and _is_main(rank):
        _save_ckpt_atomic(ckpt_final, {"arm": arm, "scale": scale, "model": base_model.state_dict(), "train_meta": train_meta})
    return base_model, train_meta


@contextmanager
def _null_ctx() -> Iterator[None]:
    yield


# ==============================================================================
# Part 4: Suites E1 (2K->16K Extrapolation), E2 (Zero-Shot), E3 (Retrieval), E4 (FOMP-v3.2)
# ==============================================================================

@torch.no_grad()
def eval_suite_e1(
    model: nn.Module,
    reader: MemmapShardReader,
    lengths: Sequence[int],
    device: torch.device,
    amp: bool,
    smoke: bool,
) -> Dict[str, Any]:
    """Suite E1: FineWeb-Edu Extrapolation PPL + Exact State MiB + Real Prefill Latency & Peak VRAM."""
    out: Dict[str, Any] = {}
    for ln in lengths:
        nw = 2 if smoke else (32 if ln <= 4096 else (16 if ln <= 8192 else 8))
        bs = 2 if ln <= 2048 else 1
        res = eval_val_ppl(model, reader, ln, n_windows=nw, device=device, amp=amp, batch_size=bs)
        # Measure real prefill latency & peak VRAM at length `ln`
        probe = reader.read("val", 0, ln).unsqueeze(0).to(device)
        with _eager_eval(model):
            if amp:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                _ = _unwrap(model)(probe)
            sync(device)
            t0 = time.perf_counter()
            n_rep = 2 if smoke else (5 if ln <= 8192 else 3)
            for _ in range(n_rep):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    _ = _unwrap(model)(probe)
            sync(device)
            prefill_ms = round((time.perf_counter() - t0) * 1000.0 / n_rep, 2)
            peak_mib = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1) if amp else 0.0
        res["state_mib"] = state_mib_at_length(model, ln)
        res["prefill_ms"] = prefill_ms
        res["prefill_peak_mib"] = peak_mib
        out[f"L{ln}"] = res
    return out


@torch.no_grad()
def _score_completion_nll(
    model: nn.Module,
    prompt_ids: List[int],
    cont_ids: List[int],
    device: torch.device,
    amp: bool,
    max_ctx: int = 2048,
) -> Tuple[float, bool]:
    """Return `(mean_nll, exact_greedy_match)` for `cont_ids` given `prompt_ids`."""
    if not prompt_ids:
        prompt_ids = [0]
    if not cont_ids:
        return 0.0, True
    if len(prompt_ids) + len(cont_ids) > max_ctx:
        prompt_ids = prompt_ids[-(max_ctx - len(cont_ids)) :]
    inp = torch.tensor([prompt_ids + cont_ids], device=device, dtype=torch.long)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        lg = _unwrap(model)(inp)["logits"][0]
    p_len = len(prompt_ids)
    pred = lg[p_len - 1 : p_len + len(cont_ids) - 1].float()
    tgt = torch.tensor(cont_ids, device=device, dtype=torch.long)
    nll = float(F.cross_entropy(pred, tgt, reduction="mean").item())
    greedy_ok = bool((pred.argmax(dim=-1) == tgt).all().item())
    return nll, greedy_ok


def _load_zeroshot_tasks(smoke: bool, max_samples: int = 1000, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Load real ungated zero-shot downstream benchmarks (exact same deterministic subset for both arms)."""
    if smoke:
        return {
            "lambada": [{"prompt": f"the quick brown fox {i}", "target": " jumps"} for i in range(6)],
            "hellaswag": [{"prompt": f"A person walks into a room {i}.", "choices": [" They sit down.", " Purple cloud."], "label": 0} for i in range(6)],
            "arc_easy": [{"prompt": "Question: Water freezes at?\nAnswer:", "choices": [" 0 C", " 100 C"], "label": 0} for i in range(6)],
            "piqa": [{"prompt": "Goal: Boil water\nSolution:", "choices": [" Use a kettle.", " Put in freezer."], "label": 0} for i in range(6)],
            "winogrande": [{"choices_ctx": [("The trophy fit in the suitcase because the trophy", " was small."), ("The trophy fit in the suitcase because the suitcase", " was small.")], "label": 0} for _ in range(6)],
        }

    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    tasks: Dict[str, List[Dict[str, Any]]] = {}

    def _sub(rows: List[Any]) -> List[Any]:
        if len(rows) <= max_samples:
            return rows
        idx = rng.choice(len(rows), size=max_samples, replace=False)
        return [rows[int(i)] for i in sorted(idx)]

    # 1. LAMBADA (OpenAI)
    try:
        ds = list(load_dataset("EleutherAI/lambada_openai", split="test"))
        items = []
        for r in _sub(ds):
            txt = str(r["text"]).strip()
            parts = txt.rsplit(" ", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                items.append({"prompt": parts[0], "target": " " + parts[1]})
        tasks["lambada"] = items
    except Exception as exc:
        print(f"[E2] Warning: LAMBADA load failed ({exc})", flush=True)

    # 2. HellaSwag
    try:
        ds = list(load_dataset("Rowan/hellaswag", split="validation"))
        items = []
        for r in _sub(ds):
            lbl = int(r["label"]) if str(r["label"]).isdigit() else -1
            if 0 <= lbl < len(r["endings"]):
                items.append({"prompt": str(r["ctx"]).strip(), "choices": [" " + str(e).strip() for e in r["endings"]], "label": lbl})
        tasks["hellaswag"] = items
    except Exception as exc:
        print(f"[E2] Warning: HellaSwag load failed ({exc})", flush=True)

    # 3. ARC-Easy
    try:
        ds = list(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"))
        items = []
        for r in _sub(ds):
            ch_labels = r["choices"]["label"]
            ch_texts = r["choices"]["text"]
            ak = str(r["answerKey"])
            if ak in ch_labels:
                items.append({
                    "prompt": f"Question: {r['question'].strip()}\nAnswer:",
                    "choices": [" " + str(t).strip() for t in ch_texts],
                    "label": ch_labels.index(ak),
                })
        tasks["arc_easy"] = items
    except Exception as exc:
        print(f"[E2] Warning: ARC-Easy load failed ({exc})", flush=True)

    # 4. PIQA
    for repo in ("baber/piqa", "ybisk/piqa"):
        try:
            ds = list(load_dataset(repo, split="validation"))
            items = []
            for r in _sub(ds):
                lbl = int(r["label"])
                if lbl in (0, 1):
                    items.append({
                        "prompt": f"Question: {r['goal'].strip()}\nAnswer:",
                        "choices": [" " + str(r["sol1"]).strip(), " " + str(r["sol2"]).strip()],
                        "label": lbl,
                    })
            tasks["piqa"] = items
            break
        except Exception:
            continue

    # 5. WinoGrande
    try:
        ds = list(load_dataset("allenai/winogrande", "winogrande_xl", split="validation"))
        items = []
        for r in _sub(ds):
            sent = str(r["sentence"])
            if "_" in sent and str(r["answer"]) in ("1", "2"):
                left, right = sent.split("_", 1)
                c1 = (left + str(r["option1"]), right)
                c2 = (left + str(r["option2"]), right)
                items.append({"choices_ctx": [c1, c2], "label": int(r["answer"]) - 1})
        tasks["winogrande"] = items
    except Exception as exc:
        print(f"[E2] Warning: WinoGrande load failed ({exc})", flush=True)

    print("[E2] Loaded zero-shot tasks:", {k: len(v) for k, v in tasks.items()}, flush=True)
    return tasks


@torch.no_grad()
def eval_suite_e2(
    model: nn.Module,
    tok: _TiktokenAdapter,
    tasks: Dict[str, List[Dict[str, Any]]],
    device: torch.device,
    amp: bool,
) -> Dict[str, Any]:
    """Suite E2: Zero-Shot Downstream Benchmarks (`acc_norm` + per-item hits for paired bootstrap)."""
    res: Dict[str, Any] = {}
    with _eager_eval(model):
        for name, items in tasks.items():
            if not items:
                continue
            hits: List[int] = []
            nlls: List[float] = []
            for ex in items:
                if name == "lambada":
                    nll, ok = _score_completion_nll(model, tok.encode(ex["prompt"]), tok.encode(ex["target"]), device, amp)
                    hits.append(int(ok))
                    nlls.append(nll)
                elif name == "winogrande":
                    scores = [
                        _score_completion_nll(model, tok.encode(pre), tok.encode(suf), device, amp)[0]
                        for pre, suf in ex["choices_ctx"]
                    ]
                    hits.append(int(int(np.argmin(scores)) == ex["label"]))
                else:
                    p_ids = tok.encode(ex["prompt"])
                    scores = [_score_completion_nll(model, p_ids, tok.encode(c), device, amp)[0] for c in ex["choices"]]
                    hits.append(int(int(np.argmin(scores)) == ex["label"]))
            entry: Dict[str, Any] = {"acc_pct": round(float(np.mean(hits)) * 100.0, 2), "n": len(hits), "hits": hits}
            if nlls:
                entry["ppl"] = round(math.exp(min(20.0, float(np.mean(nlls)))), 2)
            res[name] = entry
    return res


@torch.no_grad()
def eval_suite_e3(
    model: nn.Module,
    reader: MemmapShardReader,
    tok: _TiktokenAdapter,
    copy_lengths: Sequence[int],
    needle_lengths: Sequence[int],
    k_grid: Sequence[int],
    device: torch.device,
    amp: bool,
    smoke: bool,
) -> Dict[str, Any]:
    """Suite E3: Real-Text Repeated-Span Copy (2K..8K) + Multi-Key Needle-in-FineWeb Haystack (up to 16K)."""
    rng = np.random.default_rng(2026)
    copy_res: Dict[str, Any] = {}
    needle_res: Dict[str, Any] = {}

    with _eager_eval(model):
        # E3.1 Zero-shot repeated-span copy / induction inside held-out FineWeb-Edu
        span_len = 8 if smoke else 32
        n_trials = 4 if smoke else 16
        for ln in copy_lengths:
            hits = total = 0
            for ti in range(n_trials):
                bg = reader.read("val", 10_000 + ti * ln, ln).tolist()
                if len(bg) < ln:
                    break
                span = bg[ln // 6 : ln // 6 + span_len]
                # Place exact copy of span at the end, evaluate exact greedy continuation on 2nd half of span
                seq = bg[: ln - span_len] + span
                inp = torch.tensor([seq], device=device, dtype=torch.long)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg = _unwrap(model)(inp[:, :-1])["logits"][0]
                start_q = len(seq) - span_len + 4  # after 4-token trigger prefix
                preds = lg[start_q - 1 : len(seq) - 1].argmax(dim=-1).cpu().tolist()
                trues = seq[start_q:]
                hits += sum(int(a == b) for a, b in zip(preds, trues))
                total += len(trues)
            copy_res[f"L{ln}"] = {"copy_acc_pct": round(hits / max(1, total) * 100.0, 2), "n_tokens": total}

        # E3.2 Real-prose multi-key needle retrieval (K=16, 32, 64 inside FineWeb-Edu up to 16K!)
        n_q = 4 if smoke else 16
        for ln in needle_lengths:
            needle_res[f"L{ln}"] = {}
            for k_val in k_grid:
                hits = 0
                for ti in range(n_q):
                    bg = reader.read("val", 200_000 + ti * ln, ln).tolist()
                    keys = [f"K{ti:02d}X{j:03d}" for j in range(k_val)]
                    vals = [f"{int(rng.integers(1000, 9999))}" for _ in range(k_val)]
                    needles = [tok.encode(f"\nCode {k} is {v}.\n") for k, v in zip(keys, vals)]
                    q_idx = int(rng.integers(0, k_val))
                    q_ids = tok.encode(f"\nCode {keys[q_idx]} is ")
                    v_ids = tok.encode(vals[q_idx])
                    # Insert K needles evenly across FineWeb-Edu background so total length == ln
                    need_tot = sum(len(x) for x in needles) + len(q_ids) + len(v_ids)
                    bg_keep = max(k_val + 1, ln - need_tot)
                    seg = max(1, bg_keep // (k_val + 1))
                    seq: List[int] = []
                    pos = 0
                    for nd in needles:
                        seq.extend(bg[pos : pos + seg])
                        pos += seg
                        seq.extend(nd)
                    seq.extend(bg[pos : pos + seg])
                    prompt = (seq + q_ids)[-(ln - len(v_ids)) :]
                    _, ok = _score_completion_nll(model, prompt, v_ids, device, amp, max_ctx=ln)
                    hits += int(ok)
                needle_res[f"L{ln}"][f"K{k_val}"] = {"acc_pct": round(hits / max(1, n_q) * 100.0, 2), "n": n_q}

    return {"copy": copy_res, "needle": needle_res}


def eval_suite_e4_fomp_loop(
    pulse_model: nn.Module,
    tf_model: nn.Module,
    tok: _TiktokenAdapter,
    reader: MemmapShardReader,
    out_dir: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
) -> Dict[str, Any]:
    """Suite E4: Full-Loop Native FOMP-v3.3 on our trained PULSE-v8.1 vs TF-Modern (Base/ICL/LoRA+Replay)."""
    p_mod = _unwrap(pulse_model)
    t_mod = _unwrap(tf_model)
    d_m = int(p_mod.config.d_model)
    n_l = int(p_mod.config.n_layers)
    rl = max(2, n_l // 2)
    inj = tuple(sorted(set([rl, min(n_l - 1, rl + 2), max(rl, n_l - 3), max(rl, n_l - 2), n_l - 1])))
    fomp_cfg = FOMPv3Config(
        d_model=d_m, read_layer=rl, inject_layers=inj,
        key_dim=(32 if smoke else 256), ent_dim=(32 if smoke else 256), val_dim=(16 if smoke else 256),
        max_tgt_tokens=(4 if smoke else 8), top_k=2, rel_hidden=(16 if smoke else 128),
    )

    if smoke:
        cf_tr, cf_te, mq_tr, mq_te, _ = _make_smoke_datasets()
        m_grid = [1, 8]
    else:
        cf_tr, cf_te, mq_tr, mq_te, _ = _load_real_datasets(
            tok, max_tgt_tokens=fomp_cfg.max_tgt_tokens, n_cf_train=1200, n_cf_test=64, n_mq_train=300, n_mq_test=32, seed=41
        )
        m_grid = [1, 16, 64]

    # Disjoint FineWeb-Edu validation chunks (0% overlap between FOMP/LoRA train replay & held-out PPL eval!)
    ch_len = 48 if smoke else 256
    prose_tr = [reader.read("val", 500_000 + i * (ch_len + 1), ch_len + 1) for i in range(4 if smoke else 20)]
    prose_ev = [reader.read("val", 700_000 + i * (ch_len + 1), ch_len + 1) for i in range(4 if smoke else 16)]

    with _eager_eval(p_mod), _eager_eval(t_mod):
        fomp_p = FOMPv3Module(fomp_cfg).to(device)
        wrap_p = FOMPv3BackboneWrapper(p_mod, fomp_p, freeze_backbone=True)
        wrap_t = FOMPv3BackboneWrapper(t_mod, FOMPv3Module(fomp_cfg).to(device), freeze_backbone=True)

        p_base = _evaluate_edits(wrap_p, tok, cf_te[:64], mq_te[:32], prose_ev, device, amp, compiled=None, max_gen_eval=(4 if smoke else 24))
        t_base = _evaluate_edits(wrap_t, tok, cf_te[:64], mq_te[:32], prose_ev, device, amp, compiled=None, max_gen_eval=(4 if smoke else 24))
        p_base_ppl = p_base["general_prose_ppl"]
        t_base_ppl = t_base["general_prose_ppl"]

        train_meta = _train_fomp_v3(
            wrap_p, tok, cf_tr, mq_tr, prose_tr, device, amp, smoke,
            steps=(10 if smoke else 800), ckpt_path=(out_dir / "pulse_fomp_v34.pt"), deadline=time.time() + 3600,
        )

        pulse_fomp: Dict[str, Any] = {}
        tf_icl: Dict[str, Any] = {}
        tf_lora: Dict[str, Any] = {}
        m1_k = 4 if smoke else 16
        for m_val in m_grid:
            if m_val == 1:
                ev_p = _eval_m1_episodes(wrap_p, tok, cf_te[:m1_k], prose_ev, device, amp, tok.pad_token_id, fomp_cfg.max_tgt_tokens, mode="fomp")
                ev_p["ppl_drift_vs_base"] = round(ev_p["general_prose_ppl"] - p_base_ppl, 5)
                pulse_fomp["M1"] = ev_p
                tf_icl["M1"] = _eval_m1_episodes(wrap_t, tok, cf_te[:m1_k], prose_ev, device, amp, tok.pad_token_id, fomp_cfg.max_tgt_tokens, mode="icl")
                continue
            cf_sub = cf_te[: min(m_val, len(cf_te))]
            mq_sub = mq_te[: min(max(4, m_val // 2), len(mq_te))]
            all_sub = cf_sub + mq_sub

            # PULSE-v8.1 + FOMP-v3.2
            sync(device)
            t0 = time.perf_counter()
            comp = wrap_p.compile_token_episodes(
                [tok.encode(r.cue) for r in all_sub],
                [tok.encode(r.target_new)[: fomp_cfg.max_tgt_tokens] for r in all_sub],
                pad_id=tok.pad_token_id, device=device, g_ext=1.0,
            )
            sync(device)
            ev_p = _evaluate_edits(wrap_p, tok, cf_sub[:64], mq_sub[:32], prose_ev, device, amp, compiled=comp, max_gen_eval=(4 if smoke else 24))
            ev_p["compile_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            ev_p["ppl_drift_vs_base"] = round(ev_p["general_prose_ppl"] - p_base_ppl, 5)
            if m_val == m_grid[-1]:
                c0 = wrap_p.compile_token_episodes(
                    [tok.encode(r.cue) for r in all_sub[:4]],
                    [tok.encode(r.target_new)[: fomp_cfg.max_tgt_tokens] for r in all_sub[:4]],
                    pad_id=tok.pad_token_id, device=device, g_ext=0.0,
                )
                ev_p["g_ext0_ppl_drift"] = round(abs(_eval_prose_ppl(wrap_p, prose_ev, device, amp, compiled=c0) - p_base_ppl), 6)
                ev_sf = _evaluate_edits(wrap_p, tok, cf_sub[:24], mq_sub[:12], prose_ev, device, amp, compiled=comp, sign_flip=True, max_gen_eval=(4 if smoke else 16))
                ev_p["sign_flip"] = {"efficacy": ev_sf["efficacy"], "paraphrase": ev_sf["paraphrase"]}
            pulse_fomp[f"M{m_val}"] = ev_p

            # TF-Modern + ICL
            pref = "\n".join(["Updated facts:"] + [f"- {r.cue}{r.target_new}." for r in all_sub]) + "\n\n"
            tf_icl[f"M{m_val}"] = _evaluate_edits(wrap_t, tok, cf_sub[:64], mq_sub[:32], prose_ev, device, amp, compiled=None, icl_prefix=pref, max_gen_eval=(4 if smoke else 24))

            # TF-Modern + LoRA(r=8)+Replay
            ev_l = _run_lora_replay_arm(wrap_t, tok, cf_sub, mq_sub, prose_tr, prose_ev, device, amp, smoke)
            ev_l["ppl_drift_vs_base"] = round(ev_l["general_prose_ppl"] - t_base_ppl, 5)
            tf_lora[f"M{m_val}"] = ev_l

    return {
        "fomp_train": train_meta,
        "pulse_base": p_base,
        "tf_base": t_base,
        "pulse_fomp_v32": pulse_fomp,
        "tf_icl": tf_icl,
        "tf_lora_replay": tf_lora,
    }


def _write_markdown_report(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        f"# PULSE-v8.1 vs TF-Modern Phase 2 Scaling Law Report (`{summary['scale']}`, `{summary['target_tokens']/1e9:.2f}B` tokens)",
        "",
        f"- **GPU**: `{summary['device']}` (`world_size={summary['world_size']}`)",
        f"- **TF-Modern**: `{summary['train']['tf_modern']['n_params']/1e6:.2f}M` params | `{summary['train']['tf_modern']['avg_tok_per_sec']/1e3:.1f}K tok/s` | `16K state = {summary['train']['tf_modern']['state_mib_16k']} MiB`",
        f"- **PULSE-v8.1**: `{summary['train']['pulse_v81']['n_params']/1e6:.2f}M` params | `{summary['train']['pulse_v81']['avg_tok_per_sec']/1e3:.1f}K tok/s` | `16K state = {summary['train']['pulse_v81']['state_mib_16k']} MiB`",
        "",
        "## Suite E1: Long-Context Extrapolation (FineWeb-Edu Val PPL)",
        "| Context | TF-Modern PPL | PULSE-v8.1 PPL | Delta PPL | State Ratio | Prefill ms (TF vs PULSE) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    e1_tf = summary["eval"]["e1_extrapolation"]["tf_modern"]
    e1_p = summary["eval"]["e1_extrapolation"]["pulse_v81"]
    for lk in e1_tf:
        rt = e1_tf[lk]
        rp = e1_p[lk]
        ratio = rt["state_mib"] / max(1e-6, rp["state_mib"])
        lines.append(
            f"| `{lk}` | `{rt['ppl']}` | **`{rp['ppl']}`** | `{rp['ppl'] - rt['ppl']:+.3f}` | **`{ratio:.1f}x`** | `{rt['prefill_ms']}ms` vs `{rp['prefill_ms']}ms` |"
        )
    lines += ["", "## Suite E2: Real Public Zero-Shot Downstream Benchmarks", "| Task | TF-Modern Acc | PULSE-v8.1 Acc | Delta |", "| :--- | :---: | :---: | :---: |"]
    e2_tf = summary["eval"]["e2_zeroshot"]["tf_modern"]
    e2_p = summary["eval"]["e2_zeroshot"]["pulse_v81"]
    for tk in e2_tf:
        if tk in e2_p:
            lines.append(f"| `{tk}` | `{e2_tf[tk]['acc_pct']}%` | **`{e2_p[tk]['acc_pct']}%`** | `{e2_p[tk]['acc_pct'] - e2_tf[tk]['acc_pct']:+.2f} pp` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/fineweb_edu_2p5b"))
    parser.add_argument("--scale", choices=("125m", "350m"), default="125m")
    parser.add_argument("--tokens", type=int, default=2_500_000_000)
    parser.add_argument("--val-tokens", type=int, default=20_000_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--eval-every-tokens", type=int, default=250_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", type=str, default="pulse_v81,tf_modern")
    parser.add_argument("--prepare-data-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--skip-fomp", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    arm_order = [a.strip() for a in args.arms.split(",") if a.strip()]
    assert set(arm_order) == {"pulse_v81", "tf_modern"}, f"Invalid --arms={args.arms!r}"

    rank, world, _, device = _init_dist()
    amp = device.type == "cuda" and not args.smoke
    if amp:
        torch.set_float32_matmul_precision("high")
        patch_fla_sm89_shared_memory()

    if args.smoke:
        args.tokens = 16_384
        args.val_tokens = 32_768
        args.seq_len = 64
        args.micro_batch = 2
        args.grad_accum = 2
        args.eval_every_tokens = 8_192

    args.out.mkdir(parents=True, exist_ok=True)
    if _is_main(rank):
        prepare_fineweb_shards(args.data_dir, args.tokens, args.val_tokens, smoke=args.smoke)
    if world > 1:
        dist.barrier()
    if args.prepare_data_only:
        if world > 1:
            dist.destroy_process_group()
        # Avoid pyarrow C++ thread teardown GIL race after atomic meta.json flush
        sys.stdout.flush()
        os._exit(0)

    reader = MemmapShardReader(args.data_dir)
    if args.preflight:
        if _is_main(rank):
            print("\n=== PREFLIGHT: 12-Step Speed/VRAM Calibration + 16K OOD Sanity Check ===", flush=True)
        pf: Dict[str, Any] = {}
        for arm in arm_order:
            m, meta = train_arm(
                arm, args.scale, reader, args.out / "preflight", args.tokens, args.seq_len,
                args.micro_batch, args.grad_accum, args.lr, args.eval_every_tokens,
                args.seed, rank, world, device, amp, args.smoke, resume=False, preflight_steps=12,
            )
            if _is_main(rank):
                ln_test = 128 if args.smoke else 16384
                with torch.no_grad(), _eager_eval(m), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    probe = reader.read("val", 0, ln_test).unsqueeze(0).to(device)
                    lg = _unwrap(m)(probe)["logits"]
                    assert torch.isfinite(lg.float()).all(), f"[{arm}] Non-finite logits at L={ln_test}!"
                est_h = (args.tokens / max(1.0, meta["avg_tok_per_sec"])) / 3600.0
                meta["est_hours_for_target_tokens"] = round(est_h, 2)
                pf[arm] = meta
                print(
                    f"[PREFLIGHT {arm}] params={meta['n_params']/1e6:.2f}M | {meta['avg_tok_per_sec']/1e3:.1f}K tok/s | "
                    f"peak_vram={meta['peak_vram_gib']} GiB | L={ln_test} finite=OK | est={est_h:.2f}h",
                    flush=True,
                )
            del m
            gc.collect()
            if amp:
                torch.cuda.empty_cache()
        if _is_main(rank):
            assert pf["pulse_v81"]["n_params"] <= pf["tf_modern"]["n_params"], "PULSE params must be <= TF-Modern!"
            save_json(args.out / "preflight.json", pf)
            tot_h = pf["tf_modern"]["est_hours_for_target_tokens"] + pf["pulse_v81"]["est_hours_for_target_tokens"]
            print(f"\n=== PREFLIGHT PASS! Estimated total 2-arm pretrain time: {tot_h:.2f}h ===", flush=True)
        if world > 1:
            dist.destroy_process_group()
        return

    # Full 2-arm pretraining (exact same token stream, strictly parameter matched, PULSE-first by default!)
    models: Dict[str, nn.Module] = {}
    train_metas: Dict[str, Any] = {}
    for arm in arm_order:
        m, meta = train_arm(
            arm, args.scale, reader, args.out, args.tokens, args.seq_len,
            args.micro_batch, args.grad_accum, args.lr, args.eval_every_tokens,
            args.seed, rank, world, device, amp, args.smoke, resume=(not args.fresh),
        )
        # Keep on CPU between arms to guarantee peak VRAM headroom!
        models[arm] = _unwrap(m).cpu()
        train_metas[arm] = meta
        del m
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    if world > 1:
        dist.barrier()

    if _is_main(rank):
        assert train_metas["pulse_v81"]["n_params"] <= train_metas["tf_modern"]["n_params"]
        tok = _TiktokenAdapter(smoke=args.smoke)
        summary: Dict[str, Any] = {
            "device": torch.cuda.get_device_name(0) if amp else "cpu",
            "world_size": world,
            "scale": args.scale,
            "target_tokens": args.tokens,
            "seq_len": args.seq_len,
            "train": train_metas,
            "eval": {},
        }
        out_json = args.out / "scaling_law_summary.json"
        save_json(out_json, summary)

        # Suite E1: 2K -> 16K Long-Context Extrapolation
        print("\n=== Suite E1: Long-Context Extrapolation (2K -> 16K) ===", flush=True)
        e1_lens = (64, 128) if args.smoke else (2048, 4096, 8192, 16384)
        e1: Dict[str, Any] = {}
        for arm in ("tf_modern", "pulse_v81"):
            m = models[arm].to(device)
            e1[arm] = eval_suite_e1(m, reader, e1_lens, device, amp, args.smoke)
            m.cpu()
            if amp:
                torch.cuda.empty_cache()
        e1["paired_ci_nll_pulse_minus_tf"] = {
            f"L{ln}": boot_ci(np.array(e1["pulse_v81"][f"L{ln}"]["window_nll"]) - np.array(e1["tf_modern"][f"L{ln}"]["window_nll"]))
            for ln in e1_lens
        }
        summary["eval"]["e1_extrapolation"] = e1
        save_json(out_json, summary)
        print("[E1]", json.dumps({a: {k: {kk: vv for kk, vv in v.items() if kk != "window_nll"} for k, v in e1[a].items()} for a in ("tf_modern", "pulse_v81")}), flush=True)

        # Suite E2: Real Public Zero-Shot Downstream Benchmarks
        print("\n=== Suite E2: Zero-Shot Downstream Benchmarks (LAMBADA / HellaSwag / ARC-Easy / PIQA / WinoGrande) ===", flush=True)
        zs_tasks = _load_zeroshot_tasks(args.smoke, max_samples=(6 if args.smoke else 1000), seed=args.seed)
        e2: Dict[str, Any] = {}
        for arm in ("tf_modern", "pulse_v81"):
            m = models[arm].to(device)
            e2[arm] = eval_suite_e2(m, tok, zs_tasks, device, amp)
            m.cpu()
            if amp:
                torch.cuda.empty_cache()
        e2["paired_ci_acc_pp_pulse_minus_tf"] = {
            tk: [round(x * 100.0, 2) for x in boot_ci(np.array(e2["pulse_v81"][tk]["hits"]) - np.array(e2["tf_modern"][tk]["hits"]))]
            for tk in e2["tf_modern"]
            if tk in e2["pulse_v81"]
        }
        # Strip bulky raw hit vectors from saved JSON while keeping exact CI & summary!
        for arm in ("tf_modern", "pulse_v81"):
            for tk in e2[arm]:
                e2[arm][tk].pop("hits", None)
        summary["eval"]["e2_zeroshot"] = e2
        save_json(out_json, summary)
        print("[E2]", json.dumps(e2), flush=True)

        # Suite E3: Real-Text Copy + Multi-Key Needle Retrieval up to 16K
        print("\n=== Suite E3: Real-Prose Copy & Multi-Key Needle Retrieval (up to 16K) ===", flush=True)
        copy_lens = (64, 128) if args.smoke else (2048, 4096, 8192)
        ndl_lens = (128,) if args.smoke else (4096, 8192, 16384)
        k_grid = (2, 4) if args.smoke else (16, 32, 64)
        e3: Dict[str, Any] = {}
        for arm in ("tf_modern", "pulse_v81"):
            m = models[arm].to(device)
            e3[arm] = eval_suite_e3(m, reader, tok, copy_lens, ndl_lens, k_grid, device, amp, args.smoke)
            m.cpu()
            if amp:
                torch.cuda.empty_cache()
        summary["eval"]["e3_long_retrieval"] = e3
        save_json(out_json, summary)
        print("[E3]", json.dumps(e3), flush=True)

        # Suite E4: Full-Loop Native FOMP-v3.2 on trained PULSE-v8.1 vs TF-Modern
        if not args.skip_fomp:
            print("\n=== Suite E4: Native FOMP-v3.2 on Trained PULSE-v8.1 vs TF-Modern (ICL / LoRA+Replay) ===", flush=True)
            p_dev = models["pulse_v81"].to(device)
            t_dev = models["tf_modern"].to(device)
            summary["eval"]["e4_fomp_v32"] = eval_suite_e4_fomp_loop(p_dev, t_dev, tok, reader, args.out, device, amp, args.smoke)
            save_json(out_json, summary)
            print("[E4] Complete ->", json.dumps(summary["eval"]["e4_fomp_v32"]["pulse_fomp_v32"]), flush=True)

        rep_path = args.out / "scaling_law_report.md"
        _write_markdown_report(summary, rep_path)
        print(f"\n=== PHASE 2 SCALING LAW COMPLETE -> {out_json} & {rep_path} ===", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
