"""PULSE-v7 Stage 1b (SSMax 8K Passkey) + Stage 2 (LT2 2-Loop Full BPTT Latent Recurrent Depth).

Implements Section 4 & Section 7.3 (Stage 2) of the PULSE-v7 blueprint:
  - Stage 1b (~2 min): Verify 2-line Scalable-Softmax (`s_h * ln(pos + 2)`) inside `GatedNoPEMLABlock`
    lifts 2L `v7b_gdn2_nope_mla` 8K Passkey from 81.2% -> >= 98% while holding L=4096 K=64 >= 98%.
  - Stage 2a (~8 min): Break the TC^0 depth wall with 6L parameter-matched models (PULSE <= TF):
      * `tf_6l` (fixed-depth RoPE-SwiGLU Transformer)
      * `v7b_nonlooped_6l` (6L PULSE-v7B, T=1)
      * `v7b_lt2_looped_6l` (2 Prelude + 2 O(1) Looped Core [T=2 Full BPTT, w_init=0.99, damped residual, eps=0.05] + 2 Coda)
    Benchmarked on:
      1. Swap-S5 (non-Abelian S_5 permutation tracking, train L<=32 -> test L=32,64,128,256, r=1,2,4,6)
      2. Parity (cumulative mod-2 XOR tracking, train L<=40 -> test L=40,80,128,256, r=1,2,4,6)
      3. Multi-Hop Pointer Chasing (1-pass latent reasoning without CoT, train 1..4 hops -> test 2,4,6,8 hops, r=1,2,4,6)
  - Stage 2b (~12 min): 30M FineWeb-Edu 3-seed (41, 42, 43) Iso-FLOP (step 384 vs 512) & Iso-Param (step 512)
    comparison + STARS r=1,2,4,6 validation stability + saves BF16 seed-41 checkpoints for Stage 3!
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
from pulse.pulse_v7 import (
    PULSEv7Config,
    PULSEv7Model,
    patch_fla_sm89_shared_memory,
)
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM

# Reuse Stage 0/1 helpers cleanly without duplicating logic
import sys

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import (  # noqa: E402
    CachedFineWebReader,
    Stage1LabModel,
    boot_ci,
    make_optim,
    sample_noisy_mqar,
    sample_noisy_passkey,
    save_json,
    sync,
    train_lm_step,
    views,
)


@torch.no_grad()
def eval_lm_ppl_loops(
    model: nn.Module, val_tokens: torch.Tensor, length: int, device: torch.device, amp: bool, n_loops: Optional[int] = None
) -> Dict[str, Any]:
    model.eval()
    windows = val_tokens.unfold(0, length + 1, length)
    rows: List[float] = []
    for chunk in windows.split(4):
        chunk = chunk.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            if isinstance(model, PULSEv7Model):
                out = model(chunk[:, :-1], return_state=False, n_loops=n_loops)
                logits = out["logits"]
            else:
                out = model(chunk[:, :-1])
                logits = out[0] if isinstance(out, tuple) else out
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), chunk[:, 1:].reshape(-1), reduction="none")
            rows.extend(nll.view(chunk.size(0), -1).mean(dim=1).cpu().tolist())
    return {"ppl": math.exp(float(np.mean(rows))), "mean_nll": float(np.mean(rows)), "window_nll": rows}


# ==============================================================================
# Stage 1b: Scalable-Softmax (SSMax) 8K Passkey Verification on 2L PULSE-v7B
# ==============================================================================

def run_stage1b_ssmax(out: Path, device: torch.device, smoke: bool) -> Dict[str, Any]:
    dest = out / "stage1b_ssmax.json"
    if dest.exists():
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        if "error" not in loaded:
            return loaded

    amp = device.type == "cuda"
    vocab, pool = 256, 64
    max_steps = 40 if smoke else 1000
    batch_size = 16 if smoke else 64
    stages = [(32, 4), (64, 6)] if smoke else [(32, 4), (64, 6), (128, 8), (256, 8)]

    torch.manual_seed(42)
    # GatedNoPEMLABlock now defaults to use_ssmax=True!
    model = Stage1LabModel("v7b_nope_mla", vocab=vocab).to(device)
    with torch.no_grad():
        nn.init.normal_(model.tok_emb.weight, 0.0, 1.0 / math.sqrt(128))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    st_idx = 0
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        length, pairs = stages[st_idx]
        toks, qry, tgt, vpos, kt, vt, _ = sample_noisy_mqar(length, pairs, batch_size, pool, 80000 + step * 19)
        toks, qry, tgt, vpos, kt, vt = [x.to(device) for x in (toks, qry, tgt, vpos, kt, vt)]
        model.train()
        for pg in opt.param_groups:
            pg["lr"] = 3e-3 * min(1.0, step / 50.0)
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % (20 if smoke else 50) == 0:
            model.eval()
            with torch.no_grad():
                et, eq, eg, _, _, _, _ = sample_noisy_mqar(length, pairs, 24, pool, 1111)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    acc_cur = float(model.query_logits(et.to(device), eq.to(device)).argmax(-1).cpu().eq(eg).float().mean().item())
                et2, eq2, eg2, _, _, _, _ = sample_noisy_mqar(256 if not smoke else 64, 8 if not smoke else 6, 24, pool, 2222)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    acc_dev = float(model.query_logits(et2.to(device), eq2.to(device)).argmax(-1).cpu().eq(eg2).float().mean().item())
            print(f"[Stage 1b:v7b_ssmax] step={step:3d} ({length:3d}/{pairs}) loss={loss.item():.3f} cur={acc_cur*100:5.1f}% dev={acc_dev*100:5.1f}% ({time.perf_counter()-t0:.1f}s)", flush=True)
            if st_idx < len(stages) - 1 and acc_cur >= 0.94:
                st_idx += 1
            elif st_idx == len(stages) - 1 and acc_dev >= 0.99 and step >= (40 if smoke else 500):
                break

    model.eval()
    rec: Dict[str, Any] = {
        "params": sum(p.numel() for p in model.parameters()),
        "train_steps": step,
        "train_sec": round(time.perf_counter() - t0, 1),
        "ssmax_scales": [round(float(x), 4) for x in model.layer1.ssmax_scale.detach().abs().flatten().cpu().tolist()],  # type: ignore[union-attr]
        "cap_L384": {}, "cap_L4096": {}, "mqar_8192": {}, "passkey": {},
    }
    with torch.no_grad():
        def _ev(eval_L: int, K: int, eps: int, seed: int) -> float:
            bs = 16 if eval_L <= 1024 else (4 if eval_L <= 4096 else 2)
            hits = []
            for s in range(0, eps, bs):
                cur = min(bs, eps - s)
                tk, qr, tg, _, _, _, _ = sample_noisy_mqar(eval_L, K, cur, pool, seed + s * 29)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    pr = model.query_logits(tk.to(device), qr.to(device)).argmax(-1).cpu().numpy()
                hits.append(pr == tg.numpy())
            return round(float(np.concatenate(hits, axis=0).mean() * 100.0), 2)

        for K in ((4, 16) if smoke else (16, 32, 64)):
            rec["cap_L384"][str(K)] = _ev(128 if smoke else 384, K, 8 if smoke else 24, 91000 + K)
        if not smoke:
            for K in (8, 32, 64):
                rec["cap_L4096"][str(K)] = _ev(4096, K, 8, 95000 + K)
            for K in (8, 32):
                rec["mqar_8192"][str(K)] = _ev(8192, K, 6, 98000 + K)
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
            rec["pass_route_b_gate"] = (rec["cap_L384"]["64"] >= 94.0 and rec["passkey"]["8192"]["mean_pct"] >= 98.0)

    save_json(dest, rec)
    print("[Stage 1b DONE] SSMax NoPE-MLA:", json.dumps(rec, indent=2), flush=True)
    del model, opt
    if amp:
        gc.collect()
        torch.cuda.empty_cache()
    return rec


# ==============================================================================
# Stage 2a: Algorithmic State Tracking (Swap-S5, Parity) & Multi-Hop Reasoning
# ==============================================================================

_S5_SWAPS: List[Tuple[int, int]] = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # 10 transpositions


def sample_swap_s5(length: int, batch_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Non-Abelian S_5 ball-cup tracking: token 0 = init cup c_0 in [0..4], tokens 1..L = swap op in [5..15].
    Target at every step t=0..L is current cup index in [0..4]."""
    rng = np.random.default_rng(seed)
    tokens = np.empty((batch_size, length + 1), dtype=np.int64)
    targets = np.empty((batch_size, length + 1), dtype=np.int64)
    init_c = rng.integers(0, 5, size=batch_size)
    tokens[:, 0] = init_c
    targets[:, 0] = init_c
    cur = init_c.copy()
    ops = rng.integers(0, 11, size=(batch_size, length))  # 0..9 = 10 swaps, 10 = noop
    tokens[:, 1:] = ops + 5
    for t in range(length):
        op_t = ops[:, t]
        for idx, (a, b) in enumerate(_S5_SWAPS):
            m = op_t == idx
            if m.any():
                is_a = m & (cur == a)
                is_b = m & (cur == b)
                cur[is_a] = b
                cur[is_b] = a
        targets[:, t + 1] = cur
    return torch.from_numpy(tokens), torch.from_numpy(targets)


def sample_parity(length: int, batch_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential cumulative mod-2 XOR state tracking. Tokens in {2, 3}, per-step targets in {0, 1}."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(batch_size, length), dtype=np.int64)
    tokens = bits + 2
    targets = np.cumsum(bits, axis=1) % 2
    return torch.from_numpy(tokens), torch.from_numpy(targets)


def sample_multihop(n_nodes: int, hops: int, batch_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1-pass latent multi-hop pointer chasing (NO CoT!).
    Context: shuffled directed cycle edges [u, EDGE_TOK, v] for all n_nodes nodes, then [QRY_TOK, start, HOPS_BASE+k].
    Predicts f^k(start) at the final token position."""
    rng = np.random.default_rng(seed)
    edge_tok = n_nodes + 1
    qry_tok = n_nodes + 2
    hop_base = n_nodes + 3
    seq_len = 3 * n_nodes + 3
    tokens = np.empty((batch_size, seq_len), dtype=np.int64)
    targets = np.empty((batch_size, 1), dtype=np.int64)
    for b in range(batch_size):
        perm = rng.permutation(n_nodes)
        # Random permutation graph f(u) = perm[u]
        order = rng.permutation(n_nodes)
        pos = 0
        for u in order:
            tokens[b, pos] = u
            tokens[b, pos + 1] = edge_tok
            tokens[b, pos + 2] = perm[u]
            pos += 3
        start = int(rng.integers(0, n_nodes))
        cur = start
        for _ in range(hops):
            cur = int(perm[cur])
        tokens[b, pos] = qry_tok
        tokens[b, pos + 1] = start
        tokens[b, pos + 2] = hop_base + min(hops, 12)
        targets[b, 0] = cur
    qpos = torch.full((batch_size, 1), seq_len - 1, dtype=torch.long)
    return torch.from_numpy(tokens), qpos, torch.from_numpy(targets)


def _make_stage2a_arms(vocab: int = 64, smoke: bool = False) -> List[Tuple[str, Any]]:
    dm, nh, nkv, df_tf, df_v7, dl = (64, 4, 2, 160, 144, 24) if smoke else (192, 6, 3, 512, 448, 48)
    return [
        (
            "tf_6l",
            lambda: ScalableTransformerLM(
                ScalableTransformerConfig(vocab_size=vocab, embed_dim=dm, d_model=dm, n_heads=nh, n_layers=6, d_ff=df_tf, max_seq_len=512)
            ),
        ),
        (
            "v7b_nonlooped_6l",
            lambda: PULSEv7Model(
                PULSEv7Config(
                    vocab_size=vocab, embed_dim=dm, d_model=dm, n_heads=nh, n_kv_heads=nkv, n_layers=6,
                    layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"),
                    recurrence_mode="gdn_scalar", gate_rank=32, d_ff=df_v7, delta_chunk_size=16,
                    swa_window_size=64, mla_d_latent=dl, logit_softcap=0.0,
                    use_canon_acd=True, use_canon_cd=False, use_ssmax=True,
                    loop_core_range=None, loop_steps=1,
                )
            ),
        ),
        (
            "v7b_lt2_looped_6l",
            lambda: PULSEv7Model(
                PULSEv7Config(
                    vocab_size=vocab, embed_dim=dm, d_model=dm, n_heads=nh, n_kv_heads=nkv, n_layers=6,
                    layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"),
                    recurrence_mode="gdn_scalar", gate_rank=32, d_ff=df_v7, delta_chunk_size=16,
                    swa_window_size=64, mla_d_latent=dl, logit_softcap=0.0,
                    use_canon_acd=True, use_canon_cd=False, use_ssmax=True,
                    loop_core_range=(2, 4), loop_steps=2, loop_exit_eps=0.05,
                )
            ),
        ),
    ]


def _forward_seq_logits(model: nn.Module, tokens: torch.Tensor, n_loops: Optional[int] = None, want_exit1: bool = False):
    if isinstance(model, PULSEv7Model):
        h, _, h_e1 = model._run_backbone(tokens, return_state=False, n_loops=n_loops, return_first_exit=want_exit1)
        lg = model._logits_impl(h, model.tok_emb.weight)
        lg_e1 = model._logits_impl(h_e1, model.tok_emb.weight) if h_e1 is not None else None
        return lg, lg_e1
    out = model(tokens)
    lg = out[0] if isinstance(out, tuple) else out
    return lg, None


def run_stage2a_algorithmic(out: Path, device: torch.device, smoke: bool, deadline: float) -> Dict[str, Any]:
    dest = out / "stage2a_algorithmic.json"
    runs_dir = out / "stage2a_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda"
    arms = _make_stage2a_arms(vocab=64, smoke=smoke)
    bs = 16 if smoke else 64

    tasks_cfg = [
        ("swap_s5", 30 if smoke else 900, 32, (32, 64) if smoke else (32, 64, 128, 256), 5),
        ("parity", 30 if smoke else 900, 40, (40, 80) if smoke else (40, 80, 128, 256), 2),
        ("multihop", 30 if smoke else 1000, 16, (2, 4) if smoke else (2, 4, 6, 8), 16),
    ]

    all_res: Dict[str, Any] = {}
    for task_name, max_steps, train_param, eval_grid, n_classes in tasks_cfg:
        task_out: Dict[str, Any] = {}
        for arm_name, ctor in arms:
            p = runs_dir / f"{task_name}_{arm_name}.json"
            if p.exists():
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if "error" not in loaded:
                    task_out[arm_name] = loaded
                    continue
            if time.time() + 45 >= deadline:
                break
            try:
                torch.manual_seed(42)
                model = ctor().to(device)
                is_looped = isinstance(model, PULSEv7Model) and model.config.loop_core_range is not None
                opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
                t0 = time.perf_counter()
                last_loss = 0.0
                for step in range(1, max_steps + 1):
                    model.train()
                    mult = _cosine_lr_multiplier(step - 1, max_steps, max(10, max_steps // 20), 0.1)
                    for pg in opt.param_groups:
                        pg["lr"] = 1.5e-3 * mult
                    opt.zero_grad(set_to_none=True)

                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        if task_name == "swap_s5":
                            cur_L = int(np.random.default_rng(step).integers(8, train_param + 1))
                            tk, tg = sample_swap_s5(cur_L, bs, 10000 + step * 17)
                            tk, tg = tk.to(device), tg.to(device)
                            lg, lg1 = _forward_seq_logits(model, tk, want_exit1=is_looped)
                            loss = F.cross_entropy(lg[..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                            if lg1 is not None:
                                loss = 0.95 * loss + 0.05 * F.cross_entropy(lg1[..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                        elif task_name == "parity":
                            cur_L = int(np.random.default_rng(step).integers(8, train_param + 1))
                            tk, tg = sample_parity(cur_L, bs, 20000 + step * 17)
                            tk, tg = tk.to(device), tg.to(device)
                            lg, lg1 = _forward_seq_logits(model, tk, want_exit1=is_looped)
                            loss = F.cross_entropy(lg[..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                            if lg1 is not None:
                                loss = 0.95 * loss + 0.05 * F.cross_entropy(lg1[..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                        else:  # multihop (train 1..4 hops!)
                            h_k = int(np.random.default_rng(step).integers(1, 5))
                            tk, qp, tg = sample_multihop(train_param, h_k, bs, 30000 + step * 17)
                            tk, qp, tg = tk.to(device), qp.to(device), tg.to(device)
                            lg, lg1 = _forward_seq_logits(model, tk, want_exit1=is_looped)
                            rows_idx = torch.arange(bs, device=device)[:, None]
                            q_lg = lg[rows_idx, qp][..., :n_classes]
                            loss = F.cross_entropy(q_lg.reshape(-1, n_classes), tg.reshape(-1))
                            if lg1 is not None:
                                q_lg1 = lg1[rows_idx, qp][..., :n_classes]
                                loss = 0.95 * loss + 0.05 * F.cross_entropy(q_lg1.reshape(-1, n_classes), tg.reshape(-1))

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    last_loss = float(loss.item())

                model.eval()
                loop_sweep = (1, 2, 4, 6) if is_looped else (1,)
                eval_by_loop: Dict[str, Any] = {}
                with torch.no_grad():
                    for r in loop_sweep:
                        row_r: Dict[str, Any] = {}
                        for gval in eval_grid:
                            ev_bs = 32 if smoke else 128
                            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                                if task_name == "swap_s5":
                                    tk, tg = sample_swap_s5(gval, ev_bs, 91000 + gval)
                                    lg, _ = _forward_seq_logits(model, tk.to(device), n_loops=(r if is_looped else None))
                                    pred = lg[..., :n_classes].argmax(-1).cpu()
                                    row_r[f"L={gval}"] = {
                                        "seq_acc_pct": round(float(pred.eq(tg).float().mean().item() * 100.0), 2),
                                        "final_acc_pct": round(float(pred[:, -1].eq(tg[:, -1]).float().mean().item() * 100.0), 2),
                                    }
                                elif task_name == "parity":
                                    tk, tg = sample_parity(gval, ev_bs, 92000 + gval)
                                    lg, _ = _forward_seq_logits(model, tk.to(device), n_loops=(r if is_looped else None))
                                    pred = lg[..., :n_classes].argmax(-1).cpu()
                                    row_r[f"L={gval}"] = {
                                        "seq_acc_pct": round(float(pred.eq(tg).float().mean().item() * 100.0), 2),
                                        "final_acc_pct": round(float(pred[:, -1].eq(tg[:, -1]).float().mean().item() * 100.0), 2),
                                    }
                                else:
                                    tk, qp, tg = sample_multihop(train_param, gval, ev_bs, 93000 + gval)
                                    lg, _ = _forward_seq_logits(model, tk.to(device), n_loops=(r if is_looped else None))
                                    rows_idx = torch.arange(ev_bs, device=device)[:, None]
                                    pred = lg[rows_idx, qp.to(device)][..., :n_classes].argmax(-1).cpu()
                                    row_r[f"hops={gval}"] = round(float(pred.eq(tg).float().mean().item() * 100.0), 2)
                        eval_by_loop[f"r={r}"] = row_r

                rec: Dict[str, Any] = {
                    "task": task_name, "arm": arm_name, "params": sum(p.numel() for p in model.parameters()),
                    "train_steps": max_steps, "train_sec": round(time.perf_counter() - t0, 1),
                    "last_loss": round(last_loss, 4), "eval_by_loop": eval_by_loop,
                }
                if is_looped and model.loop_reinj_logit is not None and model.loop_damp_logit is not None:
                    rec["learned_w_reinj"] = round(float(torch.sigmoid(model.loop_reinj_logit).item()), 4)
                    rec["learned_gamma_damp"] = round(float(torch.sigmoid(model.loop_damp_logit).item()), 4)
                save_json(p, rec)
                task_out[arm_name] = rec
                print(f"[Stage 2a:{task_name:8s}] {arm_name:18s} params={rec['params']:,} sec={rec['train_sec']:5.1f}s -> {json.dumps(eval_by_loop)}", flush=True)
                del model, opt
            except Exception as exc:
                err_rec = {"task": task_name, "arm": arm_name, "error": f"{type(exc).__name__}: {exc}"[:600]}
                save_json(p, err_rec)
                task_out[arm_name] = err_rec
                print(f"[Stage 2a:{task_name:8s}] {arm_name:18s} ERROR: {err_rec['error']}", flush=True)
            if amp:
                gc.collect()
                torch.cuda.empty_cache()
        all_res[task_name] = task_out

    save_json(dest, all_res)
    return all_res


# ==============================================================================
# Stage 2b: 30M FineWeb-Edu 3-Seed Iso-FLOP & Iso-Param + STARS r=1..6 Stability
# ==============================================================================

def run_stage2b_fineweb_30m(
    out: Path, shard_dir: Optional[Path], stage01_dir: Optional[Path], device: torch.device, smoke: bool, deadline: float
) -> Dict[str, Any]:
    dest = out / "stage2b_fineweb_30m.json"
    runs_dir = out / "stage2b_30m_runs"
    ckpt_dir = out / "checkpoints"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda"

    if smoke:
        seeds = (41, 42)
        steps, iso_flop_step, length, batch, accum = 16, 12, 32, 2, 2
        val_tok = torch.randint(4, 260, (length * 4 + 1,))
        reader = None
        arm_specs = [
            ("v7b_nonlooped_30m", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"), recurrence_mode="gdn_scalar", use_canon_cd=False, logit_softcap=0.0, loop_core_range=None, loop_steps=1))),
            ("v7b_lt2_looped_30m", lambda: PULSEv7Model(replace(PULSEv7Config.micro_test(), layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"), recurrence_mode="gdn_scalar", use_canon_cd=False, logit_softcap=0.0, loop_core_range=(2, 4), loop_steps=2, loop_exit_eps=0.05))),
            ("tf_30m", lambda: ScalableTransformerLM(ScalableTransformerConfig.micro_test())),
        ]
    else:
        assert shard_dir is not None
        reader = CachedFineWebReader(shard_dir)
        seeds = (41, 42, 43)
        # 6L non-looped x 512 steps == 8 eff-layer LT2 x 384 steps -> EXACT ISO-FLOP at step 384, EXACT ISO-PARAM at step 512!
        steps, iso_flop_step, length, batch, accum = 512, 384, 512, 4, 2
        val_tok = reader.read("val", 0, length * 64 + 1)
        arm_specs = [
            ("v7b_nonlooped_30m", lambda: PULSEv7Model(PULSEv7Config.colab_30m_v7b_lt2(loop_steps=1, looped=False))),
            ("v7b_lt2_looped_30m", lambda: PULSEv7Model(PULSEv7Config.colab_30m_v7b_lt2(loop_steps=2, looped=True))),
            ("tf_30m", lambda: ScalableTransformerLM(ScalableTransformerConfig.colab_30m())),
        ]

    stride = length * batch * accum
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for s_idx, seed in enumerate(seeds):
        for arm_name, ctor in arm_specs:
            p = runs_dir / f"{seed}_{arm_name}.json"
            ckpt_path = ckpt_dir / f"seed{seed}_{arm_name}.pt"
            # Reuse tf_30m seeds 42/43 from Stage 0.4 if available (seed 41 trains & saves checkpoint for Stage 3!)
            if not p.exists() and arm_name == "tf_30m" and seed != 41 and stage01_dir is not None:
                prev_tf = stage01_dir / "stage0_30m_runs" / f"{seed}_tf_30m.json"
                if prev_tf.exists():
                    prev_rec = json.loads(prev_tf.read_text(encoding="utf-8"))
                    if "error" not in prev_rec:
                        save_json(p, prev_rec)
            if p.exists():
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if "error" not in loaded:
                    records.append(loaded)
                    continue
            if time.time() + 60 >= deadline:
                break
            try:
                torch.manual_seed(seed)
                model = ctor().to(device)
                is_looped = isinstance(model, PULSEv7Model) and model.config.loop_core_range is not None
                if amp and hasattr(model, "compile_fast_paths"):
                    model.compile_fast_paths()
                opt = make_optim(model)
                step_ms: List[float] = []
                last_loss = 0.0
                val_iso_flop = None
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
                    if is_looped and (step + 1) == iso_flop_step:
                        val_iso_flop = eval_lm_ppl_loops(model, val_tok, length, device, amp, n_loops=2)
                        print(f"[Stage 2b] seed={seed} {arm_name:20s} IsoFLOP(step={iso_flop_step}) ValPPL={val_iso_flop['ppl']:7.2f}", flush=True)

                ev = eval_lm_ppl_loops(model, val_tok, length, device, amp, n_loops=(2 if is_looped else None))
                stars_sweep = None
                if is_looped:
                    stars_sweep = {}
                    for r in (1, 2, 4, 6):
                        er = eval_lm_ppl_loops(model, val_tok, length, device, amp, n_loops=r)
                        stars_sweep[f"r={r}"] = round(er["ppl"], 2)

                # Save seed-41 checkpoints in BF16 for direct reuse in Stage 3 (Semantic RLS + Delta Verifier)!
                if seed == 41 and not smoke:
                    sd = {k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}
                    torch.save({"arm": arm_name, "seed": seed, "state_dict": sd, "val_ppl": ev["ppl"]}, ckpt_path)

                rec: Dict[str, Any] = {
                    "seed": seed, "arm": arm_name, "params": model.num_parameters(), "steps": steps,
                    "tokens": steps * stride, "median_ms": round(float(statistics.median(step_ms[min(16, steps // 2):])), 2),
                    "train_last_loss": round(last_loss, 4), "val": ev,
                }
                if val_iso_flop is not None:
                    rec["val_iso_flop"] = {"step": iso_flop_step, "ppl": round(val_iso_flop["ppl"], 2), "mean_nll": round(val_iso_flop["mean_nll"], 4)}
                if stars_sweep is not None:
                    rec["stars_loop_ppl"] = stars_sweep
                    assert model.loop_reinj_logit is not None and model.loop_damp_logit is not None
                    rec["learned_w_reinj"] = round(float(torch.sigmoid(model.loop_reinj_logit).item()), 4)
                    rec["learned_gamma_damp"] = round(float(torch.sigmoid(model.loop_damp_logit).item()), 4)
                save_json(p, rec)
                records.append(rec)
                print(f"[Stage 2b] seed={seed} {arm_name:20s} ValPPL={ev['ppl']:7.2f}  med_ms={rec['median_ms']:6.2f}  stars={stars_sweep}", flush=True)
                del model, opt
            except Exception as exc:
                err_rec = {"seed": seed, "arm": arm_name, "error": f"{type(exc).__name__}: {exc}"[:600]}
                save_json(p, err_rec)
                errors.append(err_rec)
                print(f"[Stage 2b] seed={seed} {arm_name:20s} ERROR: {err_rec['error']}", flush=True)
            if amp:
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()

    summary_arms: Dict[str, Any] = {}
    for arm_name, _ in arm_specs:
        rs = [r for r in records if r["arm"] == arm_name]
        if not rs:
            continue
        ppls = [r["val"]["ppl"] for r in rs]
        row: Dict[str, Any] = {
            "params": rs[0]["params"],
            "seeds": [r["seed"] for r in rs],
            "ppls": [round(x, 2) for x in ppls],
            "mean_ppl": round(float(statistics.mean(ppls)), 2),
            "mean_median_ms": round(float(statistics.mean(r["median_ms"] for r in rs)), 2),
        }
        if "val_iso_flop" in rs[0]:
            row["mean_iso_flop_ppl"] = round(float(statistics.mean(r["val_iso_flop"]["ppl"] for r in rs)), 2)
        if "stars_loop_ppl" in rs[0]:
            row["mean_stars_loop_ppl"] = {
                k: round(float(statistics.mean(r["stars_loop_ppl"][k] for r in rs)), 2)
                for k in rs[0]["stars_loop_ppl"]
            }
        for ref_name in ("v7b_nonlooped_30m", "tf_30m"):
            if arm_name != ref_name:
                ref_rs = {r["seed"]: r for r in records if r["arm"] == ref_name}
                diffs = [
                    np.subtract(r["val"]["window_nll"], ref_rs[r["seed"]]["val"]["window_nll"])
                    for r in rs
                    if r["seed"] in ref_rs
                ]
                if diffs:
                    all_d = np.concatenate(diffs)
                    row[f"delta_nll_vs_{ref_name}"] = round(float(all_d.mean()), 4)
                    row[f"ci95_vs_{ref_name}"] = [round(x, 4) for x in boot_ci(all_d)]
        summary_arms[arm_name] = row

    payload = {"complete": len(records) == len(seeds) * len(arm_specs), "summary": summary_arms, "errors": errors}
    save_json(dest, payload)
    print("[Stage 2b] 30M LT2 Summary:", json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fineweb-dir", type=Path, default=None)
    parser.add_argument("--stage01-dir", type=Path, default=None)
    parser.add_argument("--deadline", type=float, default=time.time() + 7200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-stage1b", action="store_true")
    parser.add_argument("--skip-stage2a", action="store_true")
    parser.add_argument("--skip-stage2b", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()

    summary: Dict[str, Any] = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "sm89_patch": sm89}
    if not args.skip_stage1b:
        summary["stage1b_ssmax"] = run_stage1b_ssmax(args.out, device, smoke=args.smoke)
    if not args.skip_stage2a and time.time() < args.deadline - 120:
        summary["stage2a_algorithmic"] = run_stage2a_algorithmic(args.out, device, smoke=args.smoke, deadline=args.deadline - 120)
    if not args.skip_stage2b and time.time() < args.deadline - 120:
        summary["stage2b_fineweb_30m"] = run_stage2b_fineweb_30m(args.out, args.fineweb_dir, args.stage01_dir, device, smoke=args.smoke, deadline=args.deadline - 60)

    save_json(args.out / "stage2_summary.json", summary)
    print("\n=== STAGE 2 COMPLETED ->", args.out / "stage2_summary.json", "===", flush=True)


if __name__ == "__main__":
    main()
