"""PULSE-v8 Week-3/4 Stage B: 30M-Class (100M-Token) Real-Corpus Iso-Time Showdown & 4K/Plasticity Suite.

Executes the final real-corpus falsification & scaling stage of the PULSE Next-Gen Blueprint on NVIDIA L4:
  Arms (Zero-Strawman, Strictly <= TF-Modern Params, FairMuonAdamW, Strict Iso-Time):
    1. `tf_modern_30m` : 12L d=384 H=6 d_ff=1024 NoPE + SSMax + Canon + Gated SDPA (40.71M total / 21.41M non-emb)
    2. `pulse_v8_30m`  : 12L d=384 H=6 d_ff=896 [2x SWA-128 + 6x FLA GDN + 2x sm_89 Triton 64-Slot (L2+L10) +
                         2x NoPE-MLA + native FOMP-v2.2] (39.57M total / 20.27M non-emb, strictly <= TF!)
    3. `pulse_v7b_30m` : Controlled ablation — identical 12L hybrid WITHOUT 64-Slot or FOMP (39.43M <= TF)

  Corpus:
    - 80% FineWeb-Edu (zero-download Drive memmap 740M train / 11.88M val, GPT-2 BPE V=50257)
    - 20% Real Code (one-time cached stream from ungated HF `bigcode/the-stack-smol` / `nampdn-ai/tiny-codes`,
      with clean fallback to 100% FineWeb-Edu if offline)

  Comprehensive Evaluation Suite on Trained 30M Checkpoints:
    - Eval B1: Real-Corpus Val PPL / BPB at L=1024 (ID) & L=4096 (4x OOD) + Paired Bootstrap 95% CI + 4K State MiB
    - Eval B2: Zero-Shot Real-Text Repeated-Span Copy / Induction at L=1024 & L=4096
    - Eval B3: Real-Prose L=4096 Multi-Key Retrieval (K=16 -> K=32 -> K=64 Rank Wall inside FineWeb-Edu!)
    - Eval B4: Real-Prose L=4096 3-Way Counter-Conventional Entity Alias-Swap (Consistent / Legal / Illegal Swap)
    - Eval B5: Real-Language Zero-Overlap Online Rule Editing (`PULSE-v8 FOMP-v2.2` vs `TF+LoRA+Replay` vs `TF+ICL`)
               with mandatory ICML 2026 Sign-Flip & FineWeb-Edu Val PPL Drift audits!
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW
from pulse.pulse_v7 import patch_fla_sm89_shared_memory
from pulse.pulse_v8 import PULSEv8Config, PULSEv8Model
from pulse.triton_slot_register import has_triton_slot_kernel
from pulse.whiteboard_anvil import TFModernConfig, TFModernLM, audit_zero_hidden_params

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import CachedFineWebReader, boot_ci, save_json, sync  # noqa: E402
from v7_stage3 import _TextEncoder, compute_auroc  # noqa: E402


# ==============================================================================
# Part 0: Deterministic 80% FineWeb-Edu + 20% Real Code Corpus Reader
# ==============================================================================

def _build_or_load_code_shard(
    cache_dir: Path,
    train_target: int = 22_000_000,
    val_target: int = 600_000,
    timeout_s: float = 180.0,
) -> Optional[Path]:
    """Stream & cache 20% real code GPT-2 BPE shard once; returns cache_dir or None on clean fallback."""
    tr_p = cache_dir / "code_25m.train.u16"
    va_p = cache_dir / "code_25m.val.u16"
    meta_p = cache_dir / "code_25m.json"
    if tr_p.exists() and va_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("train_tokens", 0) >= 15_000_000 and meta.get("val_tokens", 0) >= 200_000:
            print(f"[Data] Reusing cached code shard: {meta}", flush=True)
            return cache_dir
    try:
        import tiktoken
        from datasets import load_dataset
    except Exception as exc:
        print(f"[Data] Code stream deps unavailable ({exc}); falling back to 100% FineWeb-Edu.", flush=True)
        return None

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        ("bigcode/the-stack-smol", None, "train", "content"),
        ("nampdn-ai/tiny-codes", None, "train", "response"),
    ]
    t0 = time.perf_counter()
    for ds_name, cfg_name, split, field in sources:
        try:
            print(f"[Data] Streaming code shard from {ds_name} (target {train_target/1e6:.1f}M train + {val_target/1e3:.0f}K val)...", flush=True)
            ds = load_dataset(ds_name, cfg_name, split=split, streaming=True)
            val_buf: List[int] = []
            tr_chunks: List[np.ndarray] = []
            tr_buf: List[int] = []
            tr_tot = 0
            n_docs = 0
            for ex in ds:
                if time.perf_counter() - t0 > timeout_s:
                    print(f"[Data] Code stream hit {timeout_s:.0f}s cap at {tr_tot/1e6:.1f}M tokens.", flush=True)
                    break
                txt = ex.get(field) or ex.get("code") or ex.get("text") or ""
                if not isinstance(txt, str) or len(txt) < 64:
                    continue
                toks = enc.encode_ordinary(txt)
                toks.append(eot)
                n_docs += 1
                if len(val_buf) < val_target:
                    val_buf.extend(toks)
                else:
                    tr_buf.extend(toks)
                    if len(tr_buf) >= 250_000:
                        arr = np.asarray(tr_buf, dtype="<u2")
                        tr_chunks.append(arr)
                        tr_tot += len(arr)
                        tr_buf.clear()
                        if tr_tot >= train_target:
                            break
            if tr_buf:
                arr = np.asarray(tr_buf, dtype="<u2")
                tr_chunks.append(arr)
                tr_tot += len(arr)
            if tr_tot >= 8_000_000 and len(val_buf) >= 150_000:
                tr_all = np.concatenate(tr_chunks)
                va_all = np.asarray(val_buf[:val_target], dtype="<u2")
                tr_all.tofile(tr_p)
                va_all.tofile(va_p)
                meta = {
                    "source": ds_name,
                    "docs": n_docs,
                    "train_tokens": int(len(tr_all)),
                    "val_tokens": int(len(va_all)),
                    "build_sec": round(time.perf_counter() - t0, 1),
                }
                save_json(meta_p, meta)
                print(f"[Data] Cached code shard successfully: {meta}", flush=True)
                return cache_dir
        except Exception as exc:
            print(f"[Data] Source {ds_name} failed ({type(exc).__name__}: {exc}); trying next...", flush=True)
    print("[Data] All code stream sources unavailable; running cleanly on 100% FineWeb-Edu.", flush=True)
    return None


class MixedCorpusReader:
    """Deterministic 80% FineWeb-Edu + 20% Code (or 100% FineWeb-Edu / synthetic smoke) reader."""

    def __init__(
        self,
        fineweb_dir: Optional[Path],
        code_dir: Optional[Path],
        smoke: bool = False,
        vocab_size: int = 50257,
    ) -> None:
        self.smoke = smoke
        self.vocab_size = vocab_size
        self.fw = None if smoke else CachedFineWebReader(fineweb_dir)  # type: ignore[arg-type]
        self.code_train: Optional[np.memmap] = None
        self.code_val: Optional[np.memmap] = None
        if not smoke and code_dir is not None and (code_dir / "code_25m.train.u16").exists():
            self.code_train = np.memmap(code_dir / "code_25m.train.u16", mode="r", dtype="<u2")
            self.code_val = np.memmap(code_dir / "code_25m.val.u16", mode="r", dtype="<u2")
            self.corpus_mix = "80% FineWeb-Edu + 20% Code"
        elif smoke:
            self.corpus_mix = "Synthetic Smoke (CPU)"
        else:
            self.corpus_mix = "100% FineWeb-Edu"

    def read_train_step(self, step: int, batch: int, accum: int, seq_len: int) -> torch.Tensor:
        """Returns `[accum, batch, seq_len + 1]` int64 identical across all arms for a given `step`."""
        n_seq = batch * accum
        span = seq_len + 1
        if self.smoke:
            g = torch.Generator().manual_seed(777000 + step)
            return torch.randint(4, self.vocab_size, (accum, batch, span), generator=g)

        assert self.fw is not None
        fw_len = len(self.fw.maps["train"])
        if self.code_train is None:
            start = (step * n_seq * seq_len) % max(1, fw_len - n_seq * span - 1)
            toks = self.fw.read("train", start, n_seq * seq_len + 1)
            return toks.unfold(0, span, seq_len).reshape(accum, batch, span)

        # Deterministic exact 20% code mix (e.g. 3 or 4 code sequences out of 16 -> exact 20.0% across every 5 steps!)
        n_code = int(((step + 1) * n_seq) // 5 - (step * n_seq) // 5)
        n_fw = n_seq - n_code
        fw_off = ((step * n_seq - (step * n_seq) // 5) * seq_len) % max(1, fw_len - n_fw * span - 1)
        cd_len = len(self.code_train)
        cd_off = (((step * n_seq) // 5) * seq_len) % max(1, cd_len - max(1, n_code) * span - 1)

        fw_tok = self.fw.read("train", fw_off, n_fw * seq_len + 1).unfold(0, span, seq_len)
        if n_code > 0:
            cd_arr = np.array(self.code_train[cd_off : cd_off + n_code * seq_len + 1], dtype=np.int64)
            cd_tok = torch.from_numpy(cd_arr).unfold(0, span, seq_len)
            merged = torch.cat([fw_tok, cd_tok], dim=0)
        else:
            merged = fw_tok
        # Interleave deterministically so every microbatch has balanced mix
        perm = torch.arange(n_seq)
        if n_code > 0:
            idx = [(i * 5) % n_seq for i in range(n_seq)]
            if len(set(idx)) == n_seq:
                perm = torch.tensor(idx, dtype=torch.long)
        return merged[perm].reshape(accum, batch, span)

    def read_val(self, split: str, n_windows: int, seq_len: int, offset: int = 0) -> Optional[torch.Tensor]:
        """Returns `[n_windows, seq_len + 1]` int64 validation windows."""
        span = seq_len + 1
        need = n_windows * seq_len + 1
        if self.smoke:
            g = torch.Generator().manual_seed(999000 + offset + (1 if split == "code" else 0))
            return torch.randint(4, self.vocab_size, (n_windows, span), generator=g)
        if split == "fineweb":
            assert self.fw is not None
            arr = self.fw.read("val", offset, need)
            return arr.unfold(0, span, seq_len)
        if split == "code" and self.code_val is not None:
            max_w = max(0, (len(self.code_val) - offset - 1) // seq_len)
            use_w = min(n_windows, max_w)
            if use_w <= 0:
                return None
            arr = torch.from_numpy(np.array(self.code_val[offset : offset + use_w * seq_len + 1], dtype=np.int64))
            return arr.unfold(0, span, seq_len)
        return None


# ==============================================================================
# Model Factory & Exact 4K Storage Audit
# ==============================================================================

def build_stage_b_model(arm: str, vocab_size: int, smoke: bool) -> nn.Module:
    if smoke:
        if arm == "tf_modern_30m":
            return TFModernLM(TFModernConfig.micro_test(vocab_size=vocab_size))
        if arm == "pulse_v8_30m":
            return PULSEv8Model(PULSEv8Config.micro_test(vocab_size=vocab_size, use_fomp=True))
        if arm == "pulse_v7b_30m":
            return PULSEv8Model(PULSEv8Config(
                vocab_size=vocab_size, d_model=64, n_heads=4, n_kv_heads=2, d_ff=96,
                layer_types=("swa", "gdn", "gdn", "gdn", "gdn", "mla"),
                swa_window_size=16, mla_d_latent=32, use_fomp=False,
            ))
        raise ValueError(f"Unknown arm: {arm}")

    if arm == "tf_modern_30m":
        return TFModernLM(TFModernConfig(
            vocab_size=vocab_size, d_model=384, n_heads=6, n_layers=12, d_ff=1024,
            use_rope=False, use_ssmax=True, use_canon_acd=True, use_gated_sdpa=True,
        ))
    if arm == "pulse_v8_30m":
        return PULSEv8Model(PULSEv8Config.stage_b_30m(vocab_size=vocab_size, use_fomp=True))
    if arm == "pulse_v7b_30m":
        return PULSEv8Model(PULSEv8Config(
            vocab_size=vocab_size, d_model=384, n_heads=6, n_kv_heads=2, d_ff=896,
            layer_types=("swa", "gdn", "gdn", "gdn", "gdn", "mla", "swa", "gdn", "gdn", "gdn", "gdn", "mla"),
            swa_window_size=128, mla_d_latent=96, use_fomp=False,
        ))
    raise ValueError(f"Unknown arm: {arm}")


def working_state_bytes_bf16(model: nn.Module, seq_len: int) -> Dict[str, Any]:
    """Exact BF16 inference KV/working-state footprint (batch=1) at sequence length `seq_len`."""
    if isinstance(model, TFModernLM):
        c = model.config
        total = c.n_layers * 2 * seq_len * c.d_model * 2  # K + V in BF16 across all layers
        return {"seq_len": seq_len, "bytes": int(total), "mib": round(total / (1024 * 1024), 4)}
    assert isinstance(model, PULSEv8Model)
    c = model.config
    dh = c.d_model // c.n_heads
    kv_dim = c.n_kv_heads * dh
    total = 0
    for lt in c.layer_types:
        if lt == "swa":
            total += 2 * min(seq_len, c.swa_window_size) * kv_dim * 2
        elif lt == "gdn":
            total += c.n_heads * dh * dh * 2  # O(1) state matrix [H, dh, dh]
        elif lt == "slot":
            total += c.n_heads * (2 * c.n_slots * dh + c.n_slots) * 4  # O(1) FP32 (K_reg, V_reg, usage)
        elif lt == "mla":
            total += seq_len * c.mla_d_latent * 2  # Compressed NoPE-MLA latent c_kv
    return {"seq_len": seq_len, "bytes": int(total), "mib": round(total / (1024 * 1024), 4)}


def _iso_cosine_lr(base_lr: float, step: int, warmup_steps: int, train_sec: float, iso_sec: float, max_steps: int) -> float:
    """Strictly fair Iso-Time cosine schedule: reaches 0.1*base_lr smoothly whether capped by time or steps."""
    if step < max(1, warmup_steps):
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    p_step = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    p_time = train_sec / max(1.0, iso_sec)
    prog = min(1.0, max(0.0, max(p_step, p_time)))
    return base_lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_val_windows(model: nn.Module, windows: torch.Tensor, device: torch.device, amp: bool, micro_bs: int = 4) -> Dict[str, Any]:
    model.eval()
    rows: List[float] = []
    for chunk in windows.split(micro_bs):
        chunk = chunk.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg = model(chunk[:, :-1])["logits"]
            nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), chunk[:, 1:].reshape(-1), reduction="none")
            rows.extend(nll.float().view(chunk.shape[0], -1).mean(dim=1).cpu().tolist())
    mean_nll = float(np.mean(rows))
    return {
        "ppl": round(math.exp(min(20.0, mean_nll)), 4),
        "mean_nll": round(mean_nll, 5),
        "bpb_approx": round(mean_nll / math.log(2.0), 5),
        "n_windows": len(rows),
        "window_nll": [round(x, 5) for x in rows],
    }


# ==============================================================================
# Part 1: Strict Iso-Time 30M Pretraining (Atomic Checkpoints & Resume!)
# ==============================================================================

def run_pretrain_arm(
    arm: str,
    reader: MixedCorpusReader,
    out_dir: Path,
    vocab_size: int,
    device: torch.device,
    amp: bool,
    smoke: bool,
    iso_time_min: float,
    max_tokens: int,
    seed: int,
    deadline: float,
) -> Dict[str, Any]:
    res_path = out_dir / f"pretrain_{arm}.json"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = ckpt_dir / f"{arm}_final.pt"
    latest_ckpt = ckpt_dir / f"{arm}_latest.pt"

    if res_path.exists() and final_ckpt.exists():
        print(f"[Pretrain:{arm}] Reusing completed checkpoint & summary -> {res_path}", flush=True)
        return json.loads(res_path.read_text(encoding="utf-8"))

    seq_len = 64 if smoke else 1024
    # Auto-adapt microbatch to VRAM while keeping exact identical 16,384 tok/step across L4 (24GB) & A100/H100 (40/80GB)!
    vram_gb = (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) if (amp and not smoke) else 0.0
    if smoke:
        batch, accum = 4, 2
    elif vram_gb >= 36.0:
        batch, accum = 16, 1
    else:
        batch, accum = 8, 2
    toks_per_step = batch * accum * seq_len
    max_steps = 16 if smoke else max(1, math.ceil(max_tokens / toks_per_step))
    iso_sec = 30.0 if smoke else iso_time_min * 60.0
    warmup_steps = 3 if smoke else min(200, max(20, max_steps // 30))
    eval_every = 8 if smoke else 1000
    base_lr = 6.0e-4

    torch.manual_seed(seed)
    model = build_stage_b_model(arm, vocab_size, smoke).to(device)
    audit = audit_zero_hidden_params(model, vocab_size, model.config.d_model)
    assert audit["pass_zero_hidden_audit"], f"{arm} failed zero-hidden-param audit: {audit}"

    opt = FairMuonAdamW(model.io_parameters(), lr=base_lr, muon_lr_mult=10.0, adam_lr_mult=2.0, weight_decay=0.1)

    start_step = 0
    train_sec = 0.0
    curve: List[Dict[str, Any]] = []
    if latest_ckpt.exists():
        st_dict = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st_dict["model"])
        opt.load_state_dict(st_dict["opt"])
        start_step = int(st_dict["step"])
        train_sec = float(st_dict["train_sec"])
        curve = list(st_dict.get("curve", []))
        print(f"[Pretrain:{arm}] Resumed from step={start_step} ({start_step*toks_per_step/1e6:.2f}M tok, {train_sec/60:.2f}m)", flush=True)

    if amp and hasattr(model, "compile_fast_paths"):
        model.compile_fast_paths()

    quick_val = reader.read_val("fineweb", 4 if smoke else 16, seq_len, offset=0)
    assert quick_val is not None

    # 3 un-timed compilation warmup steps (identical for every arm, never alters saved weights!)
    w_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    w_opt = copy.deepcopy(opt.state_dict())
    for wst in range(2 if smoke else 3):
        wb = reader.read_train_step(900000 + wst, batch, accum, seq_len).to(device)
        model.train()
        opt.zero_grad(set_to_none=True)
        for mb in wb:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model.compute_loss(mb)  # type: ignore[operator]
            (loss / accum).backward()
        opt.step()
    sync(device)
    model.load_state_dict(w_state)
    opt.load_state_dict(w_opt)
    del w_state, w_opt

    if amp:
        torch.cuda.reset_peak_memory_stats(device)

    step = start_step
    last_loss = 0.0
    t_log = time.perf_counter()
    print(f"[Pretrain:{arm}] Starting: params={model.num_parameters():,} | B={batch}x{accum} L={seq_len} | max_steps={max_steps} ({max_tokens/1e6:.1f}M tok) | iso={iso_sec/60:.1f}m", flush=True)

    while step < max_steps and train_sec < iso_sec and time.time() < deadline - 180:
        cur_lr = _iso_cosine_lr(base_lr, step, warmup_steps, train_sec, iso_sec, max_steps)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        batches = reader.read_train_step(step, batch, accum, seq_len).to(device, non_blocking=amp)

        sync(device)
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for mb in batches:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model.compute_loss(mb)  # type: ignore[operator]
            (loss / accum).backward()
            loss_acc += float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(model.io_parameters(), 1.0, foreach=amp)
        opt.step()
        sync(device)
        train_sec += time.perf_counter() - t0
        last_loss = loss_acc / accum
        step += 1

        if step % (4 if smoke else 200) == 0 or step == max_steps or train_sec >= iso_sec:
            tok_s = (step * toks_per_step) / max(1e-3, train_sec)
            ent_s = f" slot_ent={model.active_slot_entropy_pct:.1f}%" if hasattr(model, "active_slot_entropy_pct") else ""
            print(
                f"  [{arm} st={step:4d}/{max_steps}] tok={step*toks_per_step/1e6:5.2f}M | t={train_sec/60:5.2f}m | "
                f"{tok_s:7,.0f} tok/s | loss={last_loss:.4f} | lr={cur_lr:.2e}{ent_s}",
                flush=True,
            )

        if step % eval_every == 0 or step == max_steps or train_sec >= iso_sec or (time.perf_counter() - t_log > 600):
            ev = eval_val_windows(model, quick_val, device, amp, micro_bs=4)
            pt = {
                "step": step,
                "tokens_M": round(step * toks_per_step / 1e6, 3),
                "train_sec": round(train_sec, 2),
                "train_loss": round(last_loss, 4),
                "quick_val_ppl": ev["ppl"],
                "quick_val_nll": ev["mean_nll"],
                "slot_ent_pct": round(model.active_slot_entropy_pct, 2) if hasattr(model, "active_slot_entropy_pct") else None,
            }
            curve.append(pt)
            print(f"  >>> [{arm} CKPT st={step}] {pt['tokens_M']}M tok | quick_val_ppl={pt['quick_val_ppl']} | t={train_sec/60:.2f}m", flush=True)
            tmp_ck = latest_ckpt.with_suffix(".pt.tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "train_sec": train_sec, "curve": curve}, tmp_ck)
            os.replace(tmp_ck, latest_ckpt)
            t_log = time.perf_counter()

    # Save final weights
    tmp_fin = final_ckpt.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "arm": arm, "step": step, "tokens": step * toks_per_step, "train_sec": train_sec}, tmp_fin)
    os.replace(tmp_fin, final_ckpt)
    if latest_ckpt.exists():
        latest_ckpt.unlink()

    fla_calls = sum(int(getattr(b, "_fla_calls", 0)) for b in getattr(model, "blocks", []))
    peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if amp else 0.0
    toks_seen = step * toks_per_step
    res = {
        "arm": arm,
        "corpus_mix": reader.corpus_mix,
        "params": int(model.num_parameters()),
        "non_emb_params": int(model.num_parameters() - vocab_size * model.config.d_model),
        "audit": audit,
        "steps_completed": int(step),
        "tokens_trained": int(toks_seen),
        "tokens_trained_M": round(toks_seen / 1e6, 3),
        "train_sec": round(train_sec, 2),
        "ms_per_step": round(train_sec * 1000.0 / max(1, step), 2),
        "tok_per_sec": round(toks_seen / max(1e-3, train_sec), 1),
        "peak_vram_mib": round(peak_mib, 1),
        "fla_calls": fla_calls,
        "triton_slot_active": bool(has_triton_slot_kernel() and getattr(model, "slot_blocks", [])),
        "final_train_loss": round(last_loss, 4),
        "curve": curve,
        "storage_1k": working_state_bytes_bf16(model, 1024),
        "storage_4k": working_state_bytes_bf16(model, 4096),
        "hit_deadline_early": bool(step < max_steps and train_sec < iso_sec),
    }
    save_json(res_path, res)

    del model, opt
    if amp:
        torch._dynamo.reset()
    gc.collect()
    if amp:
        torch.cuda.empty_cache()
    return res


# ==============================================================================
# Part 2: Comprehensive 30M Downstream Evaluation Suite (B1 - B5)
# ==============================================================================

def _load_pretrained(arm: str, ckpt_path: Path, vocab_size: int, smoke: bool, device: torch.device) -> nn.Module:
    model = build_stage_b_model(arm, vocab_size, smoke).to(device)
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    return model


def run_eval_b1_ppl_extrap(
    arms: List[str], reader: MixedCorpusReader, out_dir: Path, vocab_size: int, device: torch.device, amp: bool, smoke: bool
) -> Dict[str, Any]:
    dest = out_dir / "eval_b1_ppl_extrap.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    l_1x, l_4x = (64, 256) if smoke else (1024, 4096)
    n_1x, n_4x = (6, 4) if smoke else (64, 32)
    fw_1x = reader.read_val("fineweb", n_1x, l_1x, offset=16 * l_1x)
    fw_4x = reader.read_val("fineweb", n_4x, l_4x, offset=128 * l_1x)
    cd_1x = reader.read_val("code", n_1x // 2, l_1x, offset=0)
    cd_4x = reader.read_val("code", max(2, n_4x // 2), l_4x, offset=0)
    assert fw_1x is not None and fw_4x is not None

    per_arm: Dict[str, Any] = {}
    for arm in arms:
        m = _load_pretrained(arm, out_dir / "checkpoints" / f"{arm}_final.pt", vocab_size, smoke, device)
        r_fw1 = eval_val_windows(m, fw_1x, device, amp, micro_bs=4)
        ent1 = round(m.active_slot_entropy_pct, 2) if hasattr(m, "active_slot_entropy_pct") else None
        r_fw4 = eval_val_windows(m, fw_4x, device, amp, micro_bs=1)
        ent4 = round(m.active_slot_entropy_pct, 2) if hasattr(m, "active_slot_entropy_pct") else None
        rec: Dict[str, Any] = {
            "fineweb_1x": r_fw1,
            "fineweb_4x": r_fw4,
            "slot_ent_1x_pct": ent1,
            "slot_ent_4x_pct": ent4,
            "storage_1x": working_state_bytes_bf16(m, l_1x),
            "storage_4x": working_state_bytes_bf16(m, l_4x),
        }
        if cd_1x is not None:
            rec["code_1x"] = eval_val_windows(m, cd_1x, device, amp, micro_bs=4)
        if cd_4x is not None:
            rec["code_4x"] = eval_val_windows(m, cd_4x, device, amp, micro_bs=1)
        per_arm[arm] = rec
        print(
            f"[Eval B1:{arm}] FW 1K PPL={r_fw1['ppl']} | FW 4K PPL={r_fw4['ppl']} | "
            f"4K State={rec['storage_4x']['mib']} MiB | slot_ent_4K={ent4}%",
            flush=True,
        )
        del m
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    ci_cmp: Dict[str, Any] = {}
    if "pulse_v8_30m" in per_arm:
        for base in ("tf_modern_30m", "pulse_v7b_30m"):
            if base in per_arm:
                for key in ("fineweb_1x", "fineweb_4x", "code_1x", "code_4x"):
                    if key in per_arm["pulse_v8_30m"] and key in per_arm[base]:
                        d_nll = np.array(per_arm["pulse_v8_30m"][key]["window_nll"]) - np.array(per_arm[base][key]["window_nll"])
                        ci_cmp[f"v8_minus_{base}_{key}"] = {
                            "mean_delta_nll": round(float(d_nll.mean()), 5),
                            "ci95": [round(x, 5) for x in boot_ci(d_nll)],
                            "v8_better": bool(d_nll.mean() < 0.0),
                        }
    res = {"arms": per_arm, "paired_bootstrap_ci": ci_cmp}
    save_json(dest, res)
    return res


@torch.no_grad()
def run_eval_b2_zeroshot_copy(
    arms: List[str], reader: MixedCorpusReader, out_dir: Path, vocab_size: int, device: torch.device, amp: bool, smoke: bool
) -> Dict[str, Any]:
    """Zero-shot long-range repeated-span copy / induction inside real FineWeb-Edu documents (1K & 4K)."""
    dest = out_dir / "eval_b2_zeroshot_copy.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    cfgs = [("L1x", 64, 6, 8, 2, 16, 48) if smoke else ("L1024", 1024, 48, 16, 4, 80, 940),
            ("L4x", 256, 4, 8, 2, 16, 224) if smoke else ("L4096", 4096, 32, 16, 4, 120, 3900)]
    datasets: Dict[str, Tuple[torch.Tensor, int, int, int]] = {}
    for tag, sl, n_w, span_len, cue, src_p, dst_p in cfgs:
        base_w = reader.read_val("fineweb", n_w, sl, offset=300 * sl)
        assert base_w is not None
        x = base_w[:, :sl].clone()
        x[:, dst_p : dst_p + span_len] = x[:, src_p : src_p + span_len]
        datasets[tag] = (x, dst_p, cue, span_len)

    out_arms: Dict[str, Any] = {}
    for arm in arms:
        m = _load_pretrained(arm, out_dir / "checkpoints" / f"{arm}_final.pt", vocab_size, smoke, device).eval()
        arm_r: Dict[str, Any] = {}
        for tag, (x_all, dst_p, cue, span_len) in datasets.items():
            hits = tot = 0
            nlls: List[float] = []
            mbs = 1 if x_all.shape[1] >= 2048 else 4
            for chunk in x_all.split(mbs):
                chunk = chunk.to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg = m(chunk[:, :-1])["logits"]
                # Predict positions dst_p + cue .. dst_p + span_len - 1 (from logits at pos-1)
                pred_lg = lg[:, dst_p + cue - 1 : dst_p + span_len - 1, :].float()
                tgt = chunk[:, dst_p + cue : dst_p + span_len]
                hits += int((pred_lg.argmax(dim=-1) == tgt).sum().item())
                tot += int(tgt.numel())
                nlls.append(float(F.cross_entropy(pred_lg.reshape(-1, pred_lg.shape[-1]), tgt.reshape(-1)).item()))
            arm_r[tag] = {"copy_top1_acc_pct": round(hits / max(1, tot) * 100.0, 2), "copy_nll": round(float(np.mean(nlls)), 4)}
        out_arms[arm] = arm_r
        print(f"[Eval B2 Zero-Shot Copy:{arm}] {json.dumps(arm_r)}", flush=True)
        del m
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    save_json(dest, out_arms)
    return out_arms


def _get_word_pools(vocab_size: int, smoke: bool) -> Tuple[List[int], List[int], List[int], _TextEncoder]:
    enc = _TextEncoder(vocab_size=vocab_size, smoke=smoke)
    if smoke:
        return list(range(16, 48)), list(range(48, 80)), list(range(80, 112)), enc
    # Deterministic disjoint single-token English pools from GPT-2 BPE (zero overlap across all 3 pools!)
    keys_c = [
        "Alpha", "Bravo", "Delta", "Echo", "Fox", "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike",
        "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Victor", "Whiskey", "Yankee", "Zulu", "Atlas",
        "Orion", "Titan", "Nova", "Solar", "Lunar", "Comet", "Pulse", "Prism", "Spark", "Vortex", "Zenith",
        "Apex", "Beacon", "Cipher", "Dagger", "Ember", "Falcon", "Glacier", "Harbor", "Ivory", "Jasper", "Knight",
        "Lantern", "Marble", "Nebula", "Onyx", "Phoenix", "Quartz", "Raven", "Summit", "Tempest", "Umbra", "Valor",
        "Willow", "Anchor", "Bison", "Cedar", "Drift", "Eagle", "Flint", "Grove", "Hawk", "Iris", "Jade", "Kelp",
        "Lotus", "Maple", "North", "Olive", "Pine", "River", "Stone", "Thorn",
    ]
    vals_c = [
        "amber", "birch", "coral", "dusk", "elm", "fern", "gold", "hazel", "indigo", "jute", "kelp", "linen",
        "moss", "nickel", "ochre", "pearl", "quill", "ruby", "sage", "teal", "umber", "velvet", "wax", "zinc",
        "acorn", "berry", "clover", "daisy", "fennel", "ginger", "honey", "ivy", "lemon", "mint", "nutmeg", "oat",
        "peach", "plum", "reed", "thyme", "wheat", "yam", "apple", "bread", "chalk", "clay", "cloud", "coal",
        "copper", "cotton", "cream", "dew", "dust", "earth", "fiber", "flame", "foam", "frost", "glass", "grain",
        "grass", "gravel", "iron", "leaf", "lime", "mist", "mud", "oak", "oil", "paper", "rain", "salt", "sand",
        "silk", "silver", "slate", "smoke", "snow", "soil", "steam", "steel", "straw", "sugar", "water", "wood",
    ]
    alias_c = [
        "North", "South", "East", "West", "Red", "Blue", "Green", "Black", "White", "Gray", "Brown", "Pink",
        "Cyan", "Lime", "Mint", "Gold", "King", "Queen", "Duke", "Earl", "Lord", "Baron", "Chef", "Nurse",
        "Pilot", "Guard", "Judge", "Clerk", "Scout", "Smith", "Baker", "Mason", "Miner", "Diver", "Rider", "Walker",
        "Tiger", "Lion", "Bear", "Wolf", "Deer", "Seal", "Whale", "Shark", "Crane", "Swan", "Owl", "Crow",
        "Finch", "Robin", "Duck", "Goose", "Frog", "Toad", "Snake", "Lizard", "Camel", "Horse", "Sheep", "Goat",
        "Mouse", "Rat", "Otter", "Panda", "Koala", "Zebra", "Moose", "Bison", "Viper", "Cobra", "Gecko", "Snail",
        "Crab", "Clam", "Squid", "Trout", "Perch", "Bass", "Carp", "Pike",
    ]
    used: set = set()
    pools: List[List[int]] = []
    for cand in (keys_c, vals_c, alias_c):
        p: List[int] = []
        for w, tid in enc.single_token_pool(cand):
            if tid not in used and tid < vocab_size:
                used.add(tid)
                p.append(tid)
            if len(p) == 64:
                break
        # Fill any shortfall deterministically from safe mid-vocab tokens
        cur = 4000 + len(pools) * 200
        while len(p) < 64:
            if cur not in used:
                used.add(cur)
                p.append(cur)
            cur += 1
        pools.append(p)
    return pools[0], pools[1], pools[2], enc


def run_eval_b3_4k_multikey(
    arms: List[str], reader: MixedCorpusReader, out_dir: Path, vocab_size: int, device: torch.device, amp: bool, smoke: bool
) -> Dict[str, Any]:
    """Real-Prose L=4096 Multi-Key Retrieval (K=16 -> K=32 -> K=64 Rank Wall inside FineWeb-Edu Prose!)."""
    dest = out_dir / "eval_b3_4k_multikey.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    key_pool, val_pool, _, enc = _get_word_pools(vocab_size, smoke)
    n_pool = len(key_pool)
    val_ids_t = torch.tensor(val_pool, device=device, dtype=torch.long)
    tok_note = enc.encode(" Note:")
    tok_arrow = enc.encode(" =>")
    tok_recall = enc.encode(" Recall:")

    bg_pool_1k = reader.read_val("fineweb", 16 if smoke else 64, 64 if smoke else 1024, offset=500 * (64 if smoke else 1024))
    bg_pool_4k = reader.read_val("fineweb", 8 if smoke else 32, 256 if smoke else 4096, offset=700 * (64 if smoke else 1024))
    assert bg_pool_1k is not None and bg_pool_4k is not None

    def _sample_prose_kv(bs: int, seq_len: int, n_pairs: int, bg_src: torch.Tensor, seed_i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(seed_i)
        n_bg = bg_src.shape[0]
        x = torch.empty((bs, seq_len), dtype=torch.long)
        y = torch.empty((bs,), dtype=torch.long)
        q_sfx = tok_recall + [0] + tok_arrow
        q_len = len(q_sfx)
        usable = seq_len - q_len - 2
        stride = max(len(tok_note) + len(tok_arrow) + 3, usable // max(1, n_pairs))
        for i in range(bs):
            row = bg_src[int(rng.integers(0, n_bg)), :seq_len].clone()
            ks = rng.choice(n_pool, size=n_pairs, replace=False)
            vs = rng.integers(0, n_pool, size=n_pairs)
            for j in range(n_pairs):
                pos = min(usable - (len(tok_note) + len(tok_arrow) + 2), j * stride + int(rng.integers(0, max(1, stride // 3))))
                fact = tok_note + [key_pool[int(ks[j])]] + tok_arrow + [val_pool[int(vs[j])]]
                if pos + len(fact) <= usable:
                    row[pos : pos + len(fact)] = torch.tensor(fact, dtype=torch.long)
            qi = int(rng.integers(0, n_pairs))
            qry = tok_recall + [key_pool[int(ks[qi])]] + tok_arrow
            row[seq_len - len(qry) :] = torch.tensor(qry, dtype=torch.long)
            x[i] = row
            y[i] = int(vs[qi])
        return x, y

    l_tr, k_tr = (64, 8) if smoke else (1024, 16)
    steps = 12 if smoke else 350
    bs = 4 if smoke else 8
    eval_specs = (
        [("ID_1x_K8", 64, 8, bg_pool_1k), ("OOD_4x_K24", 256, 24, bg_pool_4k)]
        if smoke
        else [("ID_1K_K16", 1024, 16, bg_pool_1k), ("OOD_2K_K32", 2048, 32, bg_pool_4k), ("OOD_4K_K64", 4096, 64, bg_pool_4k)]
    )
    ev_batches = {nm: _sample_prose_kv(8 if smoke else 48, sl, kp, src, 33000 + i) for i, (nm, sl, kp, src) in enumerate(eval_specs)}

    out_arms: Dict[str, Any] = {}
    for arm in arms:
        m = _load_pretrained(arm, out_dir / "checkpoints" / f"{arm}_final.pt", vocab_size, smoke, device)
        opt = FairMuonAdamW(m.io_parameters(), lr=3.0e-4, muon_lr_mult=6.0, adam_lr_mult=1.5, weight_decay=0.01)
        for st in range(steps):
            xb, yb = _sample_prose_kv(bs, l_tr, k_tr, bg_pool_1k, 44000 + st)
            xb, yb = xb.to(device), yb.to(device)
            m.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = m(xb)["logits"][:, -1, :].index_select(-1, val_ids_t).float()
                loss = F.cross_entropy(lg, yb)
                if hasattr(m, "aux_loss"):
                    loss = loss + m.aux_loss  # type: ignore[attr-defined]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.io_parameters(), 1.0)
            opt.step()

        m.eval()
        arm_r: Dict[str, Any] = {}
        with torch.no_grad():
            for nm, (xe, ye) in ev_batches.items():
                hits = 0
                mbs = 1 if xe.shape[1] >= 2048 else 4
                for s0 in range(0, xe.shape[0], mbs):
                    s1 = min(xe.shape[0], s0 + mbs)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg = m(xe[s0:s1].to(device))["logits"][:, -1, :].index_select(-1, val_ids_t).float()
                    hits += int((lg.argmax(dim=-1).cpu() == ye[s0:s1]).sum().item())
                arm_r[nm] = round(hits / xe.shape[0] * 100.0, 2)
        if hasattr(m, "active_slot_entropy_pct"):
            arm_r["slot_ent_pct"] = round(m.active_slot_entropy_pct, 2)
        out_arms[arm] = arm_r
        print(f"[Eval B3 4K Multi-Key:{arm}] {json.dumps(arm_r)}", flush=True)
        del m, opt
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    save_json(dest, out_arms)
    return out_arms


def run_eval_b4_4k_alias_swap(
    arms: List[str], reader: MixedCorpusReader, out_dir: Path, vocab_size: int, device: torch.device, amp: bool, smoke: bool
) -> Dict[str, Any]:
    """Real-Prose L=4096 3-Way Counter-Conventional Entity Alias-Swap (Consistent / Legal Transfer / Illegal Swap)."""
    dest = out_dir / "eval_b4_4k_alias_swap.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    key_pool, val_pool, _, enc = _get_word_pools(vocab_size, smoke)
    n_ent = 16
    ents, roles = key_pool[:n_ent], val_pool[:n_ent]
    role_ids_t = torch.tensor(roles, device=device, dtype=torch.long)

    t_upd = enc.encode(" Update:")
    t_now = enc.encode(" holds")
    t_trf = enc.encode(" Transfer:")
    t_to = enc.encode(" gave it to")
    t_chk = enc.encode(" Claim:")
    t_cfm = enc.encode(" Confirm:")

    bg_1k = reader.read_val("fineweb", 16 if smoke else 64, 64 if smoke else 1024, offset=850 * (64 if smoke else 1024))
    bg_4k = reader.read_val("fineweb", 8 if smoke else 32, 256 if smoke else 4096, offset=950 * (64 if smoke else 1024))
    assert bg_1k is not None and bg_4k is not None

    def _sample_alias(bs: int, seq_len: int, conv_prob: float, bg_src: torch.Tensor, seed_i: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(seed_i)
        n_bg = bg_src.shape[0]
        x = torch.empty((bs, seq_len), dtype=torch.long)
        pos_cfm = torch.empty((bs,), dtype=torch.long)
        pos_chk = torch.empty((bs,), dtype=torch.long)
        y_role = torch.empty((bs,), dtype=torch.long)
        y_def = torch.empty((bs,), dtype=torch.long)
        y_cls = torch.empty((bs,), dtype=torch.long)

        for i in range(bs):
            row = bg_src[int(rng.integers(0, n_bg)), :seq_len].clone()
            e1 = int(rng.integers(0, n_ent))
            e2 = (e1 + 1 + int(rng.integers(0, n_ent - 1))) % n_ent
            e_bad = (e2 + 1 + int(rng.integers(0, n_ent - 2))) % n_ent
            if e_bad == e1:
                e_bad = (e1 + 2) % n_ent
            is_conv = bool(rng.random() < conv_prob)
            r_true = e1 if is_conv else (e1 + 1 + int(rng.integers(0, n_ent - 1))) % n_ent
            cls = int(rng.integers(0, 3))  # 0=consistent, 1=legal transfer, 2=illegal fluent swap

            # 1) Mid-document binding (~35% depth)
            f1 = t_upd + [ents[e1]] + t_now + [roles[r_true]]
            p1 = max(2, int(seq_len * 0.35))
            row[p1 : p1 + len(f1)] = torch.tensor(f1, dtype=torch.long)

            # 2) Optional Legal Transfer (~65% depth)
            holder = e1
            if cls == 1:
                f2 = t_trf + [ents[e1]] + t_to + [ents[e2]]
                p2 = max(p1 + len(f1) + 2, int(seq_len * 0.65))
                row[p2 : p2 + len(f2)] = torch.tensor(f2, dtype=torch.long)
                holder = e2

            # 3) Tail: Claim (for 3-way verifier) + Confirm (for counter-conventional binding prediction!)
            claimed_e = holder if cls in (0, 1) else e_bad
            f_chk = t_chk + [ents[claimed_e]] + t_now + [roles[r_true]]
            f_cfm = t_cfm + [ents[holder]] + t_now
            tail = f_chk + f_cfm
            pt = seq_len - len(tail)
            row[pt:] = torch.tensor(tail, dtype=torch.long)
            x[i] = row
            pos_chk[i] = pt + len(f_chk) - 1
            pos_cfm[i] = seq_len - 1
            y_role[i] = r_true
            y_def[i] = holder  # conventional default for holder
            y_cls[i] = cls
        return {"x": x, "pos_chk": pos_chk, "pos_cfm": pos_cfm, "y_role": y_role, "y_def": y_def, "y_cls": y_cls}

    l_tr = 64 if smoke else 1024
    steps = 12 if smoke else 350
    bs = 4 if smoke else 8
    ev_1k = _sample_alias(12 if smoke else 60, 64 if smoke else 1024, 0.0, bg_1k, 55001)
    ev_4k = _sample_alias(12 if smoke else 48, 256 if smoke else 4096, 0.0, bg_4k, 55002)

    out_arms: Dict[str, Any] = {}
    for arm in arms:
        m = _load_pretrained(arm, out_dir / "checkpoints" / f"{arm}_final.pt", vocab_size, smoke, device)
        d_m = m.config.d_model
        ver = nn.Sequential(nn.Linear(d_m, 96), nn.SiLU(), nn.Linear(96, 3)).to(device)
        opt = FairMuonAdamW(list(m.io_parameters()) + list(ver.parameters()), lr=3.0e-4, muon_lr_mult=6.0, adam_lr_mult=1.5, weight_decay=0.01)

        for st in range(steps):
            b = _sample_alias(bs, l_tr, 0.90, bg_1k, 66000 + st)
            ar = torch.arange(bs, device=device)
            m.train(); ver.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = m(b["x"].to(device))
                lg_r = out["logits"][ar, b["pos_cfm"].to(device)].index_select(-1, role_ids_t).float()
                lg_v = ver(out["h_fin"][ar, b["pos_chk"].to(device)].float())
                loss = F.cross_entropy(lg_r, b["y_role"].to(device)) + 0.5 * F.cross_entropy(lg_v, b["y_cls"].to(device))
                if hasattr(m, "aux_loss"):
                    loss = loss + m.aux_loss  # type: ignore[attr-defined]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(m.io_parameters()) + list(ver.parameters()), 1.0)
            opt.step()

        @torch.no_grad()
        def _eval_set(eb: Dict[str, torch.Tensor]) -> Dict[str, Any]:
            m.eval(); ver.eval()
            n_e = eb["x"].shape[0]
            mbs = 1 if eb["x"].shape[1] >= 2048 else 4
            hits = 0
            biases: List[float] = []
            scores: List[float] = []
            labels: List[int] = []
            leg_far = leg_tot = 0
            for s0 in range(0, n_e, mbs):
                s1 = min(n_e, s0 + mbs)
                ar = torch.arange(s1 - s0, device=device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out = m(eb["x"][s0:s1].to(device))
                    lg_r = out["logits"][ar, eb["pos_cfm"][s0:s1].to(device)].index_select(-1, role_ids_t).float()
                    lg_v = ver(out["h_fin"][ar, eb["pos_chk"][s0:s1].to(device)].float())
                yr = eb["y_role"][s0:s1].to(device)
                yd = eb["y_def"][s0:s1].to(device)
                yc = eb["y_cls"][s0:s1]
                hits += int((lg_r.argmax(dim=-1) == yr).sum().item())
                lp = F.log_softmax(lg_r, dim=-1)
                biases.extend((lp[ar, yd] - lp[ar, yr]).cpu().tolist())
                pr = F.softmax(lg_v, dim=-1).cpu()
                scores.extend(pr[:, 2].tolist())
                labels.extend((yc == 2).long().tolist())
                lm = (yc == 1)
                if lm.any():
                    leg_tot += int(lm.sum().item())
                    leg_far += int((pr[lm].argmax(dim=-1) == 2).sum().item())
            return {
                "cc_mid_acc_pct": round(hits / n_e * 100.0, 2),
                "prior_bias_nats": round(float(np.mean(biases)), 4),
                "auroc_3way": round(compute_auroc(np.asarray(scores), np.asarray(labels)), 4),
                "legal_far_pct": round(leg_far / max(1, leg_tot) * 100.0, 2),
            }

        arm_r = {"L_1K": _eval_set(ev_1k), "L_4K": _eval_set(ev_4k), "verifier_params": sum(p.numel() for p in ver.parameters())}
        out_arms[arm] = arm_r
        print(f"[Eval B4 4K Alias-Swap:{arm}] 1K={json.dumps(arm_r['L_1K'])} | 4K={json.dumps(arm_r['L_4K'])}", flush=True)
        del m, ver, opt
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    save_json(dest, out_arms)
    return out_arms


class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))


def run_eval_b5_online_plasticity(
    arms: List[str], reader: MixedCorpusReader, out_dir: Path, vocab_size: int, device: torch.device, amp: bool, smoke: bool
) -> Dict[str, Any]:
    """Real-Language Zero-Overlap Online Rule Editing: PULSE-v8 FOMP-v2.2 vs TF+LoRA+Replay vs TF+ICL."""
    dest = out_dir / "eval_b5_online_plasticity.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))

    key_pool, val_pool, alias_pool, enc = _get_word_pools(vocab_size, smoke)
    n_ent = 32 if smoke else 64
    n_tr = 24 if smoke else 48
    n_te = n_ent - n_tr
    ents = key_pool[:n_ent]
    aliases = alias_pool[:n_ent]
    vals = val_pool[:n_ent]
    val_ids_t = torch.tensor(vals, device=device, dtype=torch.long)

    # Fixed 1-token relation markers (zero lexical overlap between ViewA and ViewB!)
    t_rule = enc.encode(" Rule")[0] if not smoke else 10
    t_qr1 = enc.encode(" Code")[0] if not smoke else 11
    t_q2h = enc.encode(" Hop")[0] if not smoke else 12
    t_veq = enc.encode(" Alias")[0] if not smoke else 13
    t_r2 = enc.encode(" Map")[0] if not smoke else 14

    rng_w = np.random.default_rng(202611)
    r2_map = rng_w.permutation(n_ent).tolist()
    # Unseen test rules: strictly held-out during all pretraining/calibration!
    test_rules = [(n_tr + i, int(rng_w.integers(0, n_ent))) for i in range(n_te)]
    fw_val = reader.read_val("fineweb", 4 if smoke else 16, 64 if smoke else 512, offset=400 * (64 if smoke else 512))
    assert fw_val is not None

    def _ep_tok(e_idx: int, y_idx: int) -> List[int]:
        return [t_rule, ents[e_idx], vals[y_idx]]

    def _va_qry(e_idx: int) -> List[int]:
        return [t_qr1, ents[e_idx]]

    def _vb_qry(e_idx: int) -> List[int]:
        return [t_qr1, aliases[e_idx]]

    def _2h_qry(e_idx: int) -> List[int]:
        return [t_q2h, aliases[e_idx]]

    def _sample_bg(bs_b: int, sd_i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(sd_i)
        xs, ys = [], []
        for _ in range(bs_b):
            kind = int(rng.integers(0, 2))
            e = int(rng.integers(0, n_ent))
            if kind == 0:
                xs.append([t_veq, aliases[e], t_veq]); ys.append(ents[e])
            else:
                xs.append([t_r2, vals[e], t_r2]); ys.append(vals[r2_map[e]])
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    res: Dict[str, Any] = {"n_train_ents": n_tr, "n_test_ents": n_te, "chance_pct": round(100.0 / n_ent, 2)}

    # --- Arm A: PULSE-v8-30M Native FOMP-CMS v2.2 (with FineWeb replay + ICML 2026 Sign-Flip + PPL drift!) ---
    if "pulse_v8_30m" in arms:
        m = _load_pretrained("pulse_v8_30m", out_dir / "checkpoints" / "pulse_v8_30m_final.pt", vocab_size, smoke, device)
        fw_clean = eval_val_windows(m, fw_val, device, amp, micro_bs=4)["ppl"]
        opt = FairMuonAdamW(m.io_parameters(), lr=4.0e-4, muon_lr_mult=4.0, adam_lr_mult=1.5, weight_decay=0.01)
        steps_f = 20 if smoke else 600
        bs_ep = 8 if smoke else 32
        all_va = torch.tensor([[t_veq, ents[e]] for e in range(n_ent)], device=device, dtype=torch.long)
        all_vb = torch.tensor([[t_veq, aliases[e]] for e in range(n_ent)], device=device, dtype=torch.long)

        for st in range(1, steps_f + 1):
            rng = np.random.default_rng(770000 + st)
            m_ep = (1, 4, 8)[st % 3]
            ep_ids = np.zeros((bs_ep, m_ep, 3), dtype=np.int64)
            q_va = np.zeros((bs_ep, 2), dtype=np.int64)
            q_vb = np.zeros((bs_ep, 2), dtype=np.int64)
            q_2h = np.zeros((bs_ep, 2), dtype=np.int64)
            q_un = np.zeros((bs_ep, 2), dtype=np.int64)
            y_1h = np.zeros((bs_ep,), dtype=np.int64)
            y_2h = np.zeros((bs_ep,), dtype=np.int64)
            for i in range(bs_ep):
                es = rng.choice(n_tr, size=m_ep + 1, replace=False)
                vs = rng.integers(0, n_ent, size=m_ep)
                for j in range(m_ep):
                    ep_ids[i, j] = _ep_tok(int(es[j]), int(vs[j]))
                s0, v0 = int(es[0]), int(vs[0])
                q_va[i] = _va_qry(s0); q_vb[i] = _vb_qry(s0); q_2h[i] = _2h_qry(s0)
                q_un[i] = _va_qry(int(es[-1]))
                y_1h[i] = v0; y_2h[i] = r2_map[v0]

            bx, by = _sample_bg(bs_ep, 880000 + st)
            fw_mb = reader.read_train_step(500000 + st, 2 if smoke else 2, 1, 64 if smoke else 256)[0].to(device)

            m.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss_fw = m.compute_loss(fw_mb)
                comp = m.compile_episodes(torch.from_numpy(ep_ids).to(device), g_ext=1.0)
                comp3 = {k: (v.repeat(3, *([1] * (v.ndim - 1))) if v.ndim >= 2 else v) for k, v in comp.items()}
                q_cat = torch.cat([torch.from_numpy(q_va), torch.from_numpy(q_vb), torch.from_numpy(q_2h)], dim=0).to(device)
                o_cat = m.forward_with_fast_weights(q_cat, compiled=comp3)
                lg_cat = o_cat["logits"][:, -1, :].index_select(-1, val_ids_t).float()
                l_va, l_vb, l_2h = lg_cat.split(bs_ep, dim=0)
                y1_t = torch.from_numpy(y_1h).to(device); y2_t = torch.from_numpy(y_2h).to(device)
                o_un = m.forward_with_fast_weights(torch.from_numpy(q_un).to(device), compiled=comp)
                o_bg = m.forward_with_fast_weights(bx.to(device), compiled=None)
                o_bg_ld = m.forward_with_fast_weights(bx.to(device), compiled=comp)
                comp_fw = {k: (v[: fw_mb.shape[0]] if v.ndim >= 2 else v) for k, v in comp.items()}
                o_fw_ld = m.forward_with_fast_weights(fw_mb[:, :16], compiled=comp_fw)
                l_bg = F.cross_entropy(o_bg["logits"][:, -1, :].float(), by.to(device))
                l_align = m.view_alignment_loss(all_va, all_vb)
                # Explicit positive/negative gate supervision (protect BOTH background rules AND real FineWeb prose!)
                rl_pos = o_cat["rel_logits"][:, -1]
                rl_neg = torch.cat([o_bg_ld["rel_logits"][:, -1], o_fw_ld["rel_logits"].reshape(-1)], dim=0)
                l_gate = (
                    0.5 * (F.binary_cross_entropy_with_logits(rl_pos, torch.ones_like(rl_pos))
                           + F.binary_cross_entropy_with_logits(rl_neg, torch.zeros_like(rl_neg)))
                    + 2.0 * (
                        o_un["max_fam_gate"][:, -1].pow(2).mean()
                        + o_bg_ld["max_fam_gate"][:, -1].pow(2).mean()
                        + o_fw_ld["max_fam_gate"].pow(2).mean()
                    )
                )
                loss = (
                    0.5 * loss_fw + l_bg + 0.5 * l_align + l_gate
                    + F.cross_entropy(l_va, y1_t) + 1.5 * F.cross_entropy(l_vb, y1_t) + 2.5 * F.cross_entropy(l_2h, y2_t)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.io_parameters(), 1.0)
            opt.step()

        # Evaluate strictly unseen test rules (M=1 & M=16 simultaneous + Sign-Flip + FineWeb Val PPL drift!)
        m.eval()
        with torch.no_grad():
            fw_post_clean = eval_val_windows(m, fw_val, device, amp, micro_bs=4)["ppl"]
            ep_all = torch.tensor([[[*_ep_tok(e, v)] for e, v in test_rules]], device=device, dtype=torch.long)  # [1, M, 3]
            comp_m16 = m.compile_episodes(ep_all, g_ext=1.0)
            comp_g0 = m.compile_episodes(ep_all, g_ext=0.0)

            # FineWeb Val PPL with g_ext=0 (exact 0.0000!) and with M=16 loaded (identical per-window averaging!)
            def _eval_fw_with_comp(comp_d: Dict[str, torch.Tensor]) -> float:
                rows_w: List[float] = []
                for chunk in fw_val.split(4):
                    chunk = chunk.to(device)
                    c_rep = {k: (v.expand(chunk.shape[0], *([-1] * (v.ndim - 1))) if v.ndim >= 2 else v) for k, v in comp_d.items()}
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg = m.forward_with_fast_weights(chunk[:, :-1], compiled=c_rep)["logits"]
                        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), chunk[:, 1:].reshape(-1), reduction="none")
                        rows_w.extend(nll.float().view(chunk.shape[0], -1).mean(dim=1).cpu().tolist())
                return round(math.exp(min(20.0, float(np.mean(rows_w)))), 4)

            fw_g0 = _eval_fw_with_comp(comp_g0)
            fw_m16 = _eval_fw_with_comp(comp_m16)

            def _eval_rules(simultaneous: bool, sign_flip: bool) -> Dict[str, float]:
                h_va = h_vb = h_2h = 0
                for e_i, v_i in test_rules:
                    c_use = comp_m16 if simultaneous else m.compile_episodes(torch.tensor([[[*_ep_tok(e_i, v_i)]]], device=device), g_ext=1.0)
                    for nm, q_toks, tgt in (("va", _va_qry(e_i), v_i), ("vb", _vb_qry(e_i), v_i), ("2h", _2h_qry(e_i), r2_map[v_i])):
                        q_t = torch.tensor([q_toks], device=device, dtype=torch.long)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                            lg = m.forward_with_fast_weights(q_t, compiled=c_use, sign_flip=sign_flip)["logits"][0, -1].index_select(-1, val_ids_t).float()
                        ok = int(int(lg.argmax().item()) == tgt)
                        if nm == "va": h_va += ok
                        elif nm == "vb": h_vb += ok
                        else: h_2h += ok
                return {
                    "va_pct": round(h_va / n_te * 100.0, 2),
                    "vb_1hop_pct": round(h_vb / n_te * 100.0, 2),
                    "vb_1pass_2hop_pct": round(h_2h / n_te * 100.0, 2),
                }

            m1_n = _eval_rules(False, False); m1_f = _eval_rules(False, True)
            m16_n = _eval_rules(True, False); m16_f = _eval_rules(True, True)

        res["pulse_v8_fomp_v22"] = {
            "fineweb_ppl_pre_calib": fw_clean,
            "fineweb_ppl_clean": fw_post_clean,
            "fineweb_ppl_g_ext0": fw_g0,
            "g_ext0_drift_ppl": round(abs(fw_g0 - fw_post_clean), 6),
            "fineweb_ppl_m16_loaded": fw_m16,
            "m16_fineweb_drift_ppl": round(fw_m16 - fw_post_clean, 4),
            "M1_normal": m1_n,
            "M1_sign_flip": m1_f,
            "M16_normal": m16_n,
            "M16_sign_flip": m16_f,
            "pass_sign_flip": bool(m16_n["vb_1pass_2hop_pct"] - m16_f["vb_1pass_2hop_pct"] >= 30.0),
        }
        print(f"[Eval B5 PULSE-v8 FOMP-v2.2] {json.dumps(res['pulse_v8_fomp_v22'])}", flush=True)
        del m, opt
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    # --- Arm B: Zero-Strawman Opponents on Pretrained TF-Modern-30M (LoRA+Replay & ICL) ---
    if "tf_modern_30m" in arms:
        tf = _load_pretrained("tf_modern_30m", out_dir / "checkpoints" / "tf_modern_30m_final.pt", vocab_size, smoke, device)
        # First teach TF-Modern the EXACT SAME background knowledge (ViewA==ViewB + R2 + train rules + FineWeb replay)!
        opt_tf = FairMuonAdamW(tf.io_parameters(), lr=4.0e-4, muon_lr_mult=4.0, adam_lr_mult=1.5, weight_decay=0.01)
        steps_tf = 20 if smoke else 400
        bs_ep = 8 if smoke else 32
        for st in range(1, steps_tf + 1):
            rng = np.random.default_rng(910000 + st)
            bx, by = _sample_bg(bs_ep, 920000 + st)
            # Also teach ICL format on train entities ONLY so TF+ICL is a genuinely strong zero-strawman opponent!
            icl_x, icl_y = [], []
            for _ in range(bs_ep // 2):
                es = rng.choice(n_tr, size=4, replace=False)
                vs = rng.integers(0, n_ent, size=4)
                pref: List[int] = []
                for e_j, v_j in zip(es, vs):
                    pref.extend(_ep_tok(int(e_j), int(v_j)))
                s0, v0 = int(es[0]), int(vs[0])
                q_t = _2h_qry(s0) if (st % 2 == 0) else _vb_qry(s0)
                icl_x.append(pref + q_t)
                icl_y.append(r2_map[v0] if (st % 2 == 0) else v0)
            fw_mb = reader.read_train_step(500000 + st, 2, 1, 64 if smoke else 256)[0].to(device)
            tf.train()
            opt_tf.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                l_fw = tf.compute_loss(fw_mb)
                l_bg = F.cross_entropy(tf(bx.to(device))["logits"][:, -1, :].float(), by.to(device))
                l_icl = F.cross_entropy(
                    tf(torch.tensor(icl_x, device=device))["logits"][:, -1, :].index_select(-1, val_ids_t).float(),
                    torch.tensor(icl_y, device=device),
                )
                loss = 0.5 * l_fw + l_bg + 1.5 * l_icl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tf.io_parameters(), 1.0)
            opt_tf.step()

        tf_clean_ppl = eval_val_windows(tf, fw_val, device, amp, micro_bs=4)["ppl"]

        # Opponent B1: TF-Modern + ICL (prepend all M=16 test rules into prompt context!)
        tf.eval()
        pref_16: List[int] = []
        for e_i, v_i in test_rules:
            pref_16.extend(_ep_tok(e_i, v_i))
        with torch.no_grad():
            h_va = h_vb = h_2h = 0
            for e_i, v_i in test_rules:
                for nm, q_toks, tgt in (("va", _va_qry(e_i), v_i), ("vb", _vb_qry(e_i), v_i), ("2h", _2h_qry(e_i), r2_map[v_i])):
                    inp = torch.tensor([pref_16 + q_toks], device=device, dtype=torch.long)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg = tf(inp)["logits"][0, -1].index_select(-1, val_ids_t).float()
                    ok = int(int(lg.argmax().item()) == tgt)
                    if nm == "va": h_va += ok
                    elif nm == "vb": h_vb += ok
                    else: h_2h += ok
        res["tf_modern_icl_m16"] = {
            "extra_prompt_tokens": len(pref_16),
            "va_pct": round(h_va / n_te * 100.0, 2),
            "vb_1hop_pct": round(h_vb / n_te * 100.0, 2),
            "vb_1pass_2hop_pct": round(h_2h / n_te * 100.0, 2),
        }

        # Opponent B2: TF-Modern + LoRA (rank 8) + FineWeb Replay (30 gradient steps on the 16 unseen test facts!)
        lora_params: List[nn.Parameter] = []
        for blk in tf.blocks:
            blk.o_proj = _LoRALinear(blk.o_proj, rank=8)  # type: ignore[assignment]
            blk.mlp.w_down = _LoRALinear(blk.mlp.w_down, rank=8)  # type: ignore[assignment]
            lora_params.extend([blk.o_proj.A, blk.o_proj.B, blk.mlp.w_down.A, blk.mlp.w_down.B])  # type: ignore[attr-defined]
        opt_l = torch.optim.AdamW(lora_params, lr=2.0e-3, weight_decay=0.0)
        ep_t = torch.tensor([_ep_tok(e_i, v_i) for e_i, v_i in test_rules], device=device, dtype=torch.long)
        for st in range(10 if smoke else 30):
            fw_mb = reader.read_train_step(600000 + st, 2, 1, 64 if smoke else 256)[0].to(device)
            tf.train()
            opt_l.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg_ep = tf(ep_t[:, :-1])["logits"][:, -1, :].float()
                loss = F.cross_entropy(lg_ep, ep_t[:, -1]) + 0.5 * tf.compute_loss(fw_mb)
            loss.backward()
            opt_l.step()

        tf_lora_ppl = eval_val_windows(tf, fw_val, device, amp, micro_bs=4)["ppl"]
        tf.eval()
        with torch.no_grad():
            h_va = h_vb = h_2h = 0
            for e_i, v_i in test_rules:
                for nm, q_toks, tgt in (("va", _va_qry(e_i), v_i), ("vb", _vb_qry(e_i), v_i), ("2h", _2h_qry(e_i), r2_map[v_i])):
                    q_t = torch.tensor([q_toks], device=device, dtype=torch.long)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg = tf(q_t)["logits"][0, -1].index_select(-1, val_ids_t).float()
                    ok = int(int(lg.argmax().item()) == tgt)
                    if nm == "va": h_va += ok
                    elif nm == "vb": h_vb += ok
                    else: h_2h += ok
        res["tf_modern_lora_replay_m16"] = {
            "fineweb_ppl_clean": tf_clean_ppl,
            "fineweb_ppl_after_lora": tf_lora_ppl,
            "fineweb_drift_ppl": round(tf_lora_ppl - tf_clean_ppl, 4),
            "va_pct": round(h_va / n_te * 100.0, 2),
            "vb_1hop_pct": round(h_vb / n_te * 100.0, 2),
            "vb_1pass_2hop_pct": round(h_2h / n_te * 100.0, 2),
        }
        print(f"[Eval B5 TF+ICL] {json.dumps(res['tf_modern_icl_m16'])} | [TF+LoRA+Replay] {json.dumps(res['tf_modern_lora_replay_m16'])}", flush=True)
        del tf, opt_tf, opt_l
        gc.collect()
        if amp:
            torch.cuda.empty_cache()

    save_json(dest, res)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fineweb-dir", type=Path, default=None)
    parser.add_argument("--iso-time-min", type=float, default=25.0)
    parser.add_argument("--max-tokens", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--no-code-mix", action="store_true")
    parser.add_argument("--no-v7b-ablation", action="store_true")
    parser.add_argument("--deadline", type=float, default=time.time() + 7200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()

    vocab_size = 128 if args.smoke else 50257
    code_dir = None if (args.smoke or args.no_code_mix) else _build_or_load_code_shard(args.out / "data_cache")
    reader = MixedCorpusReader(args.fineweb_dir, code_dir, smoke=args.smoke, vocab_size=vocab_size)

    arms = ["tf_modern_30m", "pulse_v8_30m"]
    if not args.no_v7b_ablation:
        arms.append("pulse_v7b_30m")

    summary: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if amp else "cpu",
        "sm89_patch": sm89,
        "corpus_mix": reader.corpus_mix,
        "vocab_size": vocab_size,
        "iso_time_min": args.iso_time_min,
        "max_tokens": args.max_tokens,
        "pretrain": {},
    }

    # Part 1: Strict Iso-Time Pretraining (atomic per-arm & per-1000-step resume!)
    trained_arms: List[str] = []
    for arm in arms:
        if time.time() > args.deadline - 240 and not (args.out / f"pretrain_{arm}.json").exists():
            print(f"[Deadline] Skipping unstarted arm {arm} to preserve time for evaluation.", flush=True)
            continue
        summary["pretrain"][arm] = run_pretrain_arm(
            arm=arm, reader=reader, out_dir=args.out, vocab_size=vocab_size, device=device, amp=amp,
            smoke=args.smoke, iso_time_min=args.iso_time_min, max_tokens=args.max_tokens,
            seed=args.seed, deadline=args.deadline,
        )
        trained_arms.append(arm)
        save_json(args.out / "stage_b_30m_summary.json", summary)

    # Verify strict param parity PULSE-v8 <= TF-Modern!
    if "tf_modern_30m" in summary["pretrain"] and "pulse_v8_30m" in summary["pretrain"] and not args.smoke:
        assert summary["pretrain"]["pulse_v8_30m"]["params"] <= summary["pretrain"]["tf_modern_30m"]["params"]

    # Part 2: Comprehensive Evaluation Suite (B1 - B5)
    if trained_arms and time.time() < args.deadline - 60:
        summary["eval_b1_ppl_extrap"] = run_eval_b1_ppl_extrap(trained_arms, reader, args.out, vocab_size, device, amp, args.smoke)
        save_json(args.out / "stage_b_30m_summary.json", summary)
    if trained_arms and time.time() < args.deadline - 60:
        summary["eval_b2_zeroshot_copy"] = run_eval_b2_zeroshot_copy(trained_arms, reader, args.out, vocab_size, device, amp, args.smoke)
        save_json(args.out / "stage_b_30m_summary.json", summary)
    if trained_arms and time.time() < args.deadline - 90:
        summary["eval_b3_4k_multikey"] = run_eval_b3_4k_multikey(trained_arms, reader, args.out, vocab_size, device, amp, args.smoke)
        save_json(args.out / "stage_b_30m_summary.json", summary)
    if trained_arms and time.time() < args.deadline - 90:
        summary["eval_b4_4k_alias_swap"] = run_eval_b4_4k_alias_swap(trained_arms, reader, args.out, vocab_size, device, amp, args.smoke)
        save_json(args.out / "stage_b_30m_summary.json", summary)
    if trained_arms and time.time() < args.deadline - 90:
        summary["eval_b5_online_plasticity"] = run_eval_b5_online_plasticity(trained_arms, reader, args.out, vocab_size, device, amp, args.smoke)
        save_json(args.out / "stage_b_30m_summary.json", summary)

    print("\n=== PULSE-v8 STAGE B 30M REAL-CORPUS SHOWDOWN COMPLETE ->", args.out / "stage_b_30m_summary.json", "===", flush=True)


if __name__ == "__main__":
    main()
