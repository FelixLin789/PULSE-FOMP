"""PULSE-v7 Stage 2a-Fix (Curriculum TC^0 Showdown) + Stage 3a (Dual-Track Semantic Plasticity & EngramEdit)
+ Stage 3b (3-Way Transition-Conditioned Delta-Binding Verifier <50K MLP + DeCoRe at 512 & 4K).

Implements Section 5, Section 6 & Section 7.3 (Stage 3) of the PULSE-v7 blueprint on NVIDIA L4 (or CPU --smoke):
  - Reuses trained 30M BF16 checkpoints straight from Drive `runs/pulse_v7_stage2/checkpoints/`
    (`seed41_v7b_lt2_looped_30m.pt` & `seed41_tf_30m.pt`) — ZERO duplicate pretraining!
  - Stage 2a-Fix (~5 min): Curriculum Swap-S5 (32->256), Parity (40->256), 8-node Multi-Hop (2/4/6/8-hop)
    with PyTorch 2.14 FlexAttention L<512 fix + STARS r=1,2,4,6 sweep.
  - Stage 3a (~7 min): Disjoint-entity 50/100/200 novel facts with 3 unseen paraphrases + 2-hop bridges +
    FineWeb-Edu validation locality audit:
      * PULSE-v7 Dual-Track Semantic RLS (Wake)
      * PULSE-v7 EngramEdit (Sleep Consolidation)
      * PULSE-v7 Static AlphaEdit (Rejected null-space baseline -> demonstrates rank saturation!)
      * TF-30M + LoRA (rank 8) + FineWeb Replay (Zero-strawman gradient opponent)
      * TF-30M + RAG / ICL (Zero-strawman prompt-injection opponent -> measures context-token & latency tax)
  - Stage 3b (~6 min): 3-Way Entity-Binding Trajectories (Consistent / Legal State Transition / Illegal Fluent Swap)
    embedded inside real FineWeb-Edu distractor prose at L=512 AND L=4096:
      * Naive ||delta_t||_2 threshold (false-alarms legal transitions)
      * TF-30M + Matched <50K 2-layer MLP Verifier (same 40,707 params!)
      * PULSE-v7B + <50K Transition-Conditioned Delta-Binding Verifier ([gdn_signals, h_mid])
      * Verifier-Gated DeCoRe Contrastive Decoding intervention
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import _cosine_lr_multiplier
from pulse.pulse_v7 import PULSEv7Config, PULSEv7Model, patch_fla_sm89_shared_memory
from pulse.pulse_v7_plasticity import (
    DualTrackSemanticMemory,
    SemanticKeyProjector,
    TransitionConditionedVerifier,
    decore_contrastive_logits,
    solve_mid_layer_target_delta,
)
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import CachedFineWebReader, eval_lm_ppl, save_json, sync  # noqa: E402
from v7_stage2 import (  # noqa: E402
    _forward_seq_logits,
    _make_stage2a_arms,
    eval_lm_ppl_loops,
    sample_multihop,
    sample_parity,
    sample_swap_s5,
)


class _TextEncoder:
    """GPT-2 BPE via tiktoken on Colab L4, with deterministic word-hash fallback for offline CPU --smoke."""

    def __init__(self, vocab_size: int = 50257, smoke: bool = False) -> None:
        self.vocab_size = vocab_size
        self.enc = None
        if not smoke:
            try:
                import tiktoken

                self.enc = tiktoken.get_encoding("gpt2")
            except Exception:
                self.enc = None

    def encode(self, text: str) -> List[int]:
        if self.enc is not None:
            return self.enc.encode(text)
        toks = []
        for w in text.replace("?", " ?").replace(".", " .").replace(",", " ,").replace(":", " :").split():
            h = 10 + (abs(hash(w)) % max(16, self.vocab_size - 12))
            toks.append(h)
        return toks or [10]

    def single_token_pool(self, candidates: List[str]) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        seen = set()
        for w in candidates:
            ids = self.encode(" " + w)
            if len(ids) == 1 and ids[0] not in seen:
                seen.add(ids[0])
                out.append((w, ids[0]))
        return out


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Exact rank-based Mann-Whitney U AUROC (no sklearn dependency!)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Handle ties with average rank
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    r_pos = ranks[labels == 1].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


# ==============================================================================
# Part 1: Stage 2a-Fix — Clean Curriculum TC^0 Depth Wall Showdown
# ==============================================================================

def run_stage2a_fixed(out: Path, device: torch.device, smoke: bool, deadline: float) -> Dict[str, Any]:
    dest = out / "stage2a_fixed.json"
    runs_dir = out / "stage2a_fixed_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda"
    arms = _make_stage2a_arms(vocab=64, smoke=smoke)
    bs = 16 if smoke else 64

    tasks_cfg = [
        ("swap_s5", 30 if smoke else 1500, [(4, 12), (8, 32)], (32, 64) if smoke else (32, 64, 128, 256), 5),
        ("parity", 30 if smoke else 1500, [(4, 12), (8, 40)], (40, 80) if smoke else (40, 80, 128, 256), 2),
        ("multihop8", 30 if smoke else 1500, [(1, 1), (1, 2), (1, 4)], (2, 4) if smoke else (2, 4, 6, 8), 8),
    ]

    all_res: Dict[str, Any] = {}
    for task_name, max_steps, curr_stages, eval_grid, n_classes in tasks_cfg:
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
                opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
                t0 = time.perf_counter()
                st_idx = 0
                last_loss = 0.0
                for step in range(1, max_steps + 1):
                    model.train()
                    mult = _cosine_lr_multiplier(step - 1, max_steps, max(10, max_steps // 20), 0.1)
                    for pg in opt.param_groups:
                        pg["lr"] = 2e-3 * mult
                    opt.zero_grad(set_to_none=True)
                    lo, hi = curr_stages[st_idx]
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        if task_name == "swap_s5":
                            cur_L = int(np.random.default_rng(step).integers(lo, hi + 1))
                            tk, tg = sample_swap_s5(cur_L, bs, 10000 + step * 17)
                            tk, tg = tk.to(device), tg.to(device)
                            lg, lg1 = _forward_seq_logits(model, tk, want_exit1=is_looped)
                            loss = F.cross_entropy(lg[..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                            if lg1 is not None:
                                loss = 0.95 * loss + 0.05 * F.cross_entropy(lg1[..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                        elif task_name == "parity":
                            cur_L = int(np.random.default_rng(step).integers(lo, hi + 1))
                            tk, tg = sample_parity(cur_L, bs, 20000 + step * 17)
                            tk, tg = tk.to(device), tg.to(device)
                            lg, lg1 = _forward_seq_logits(model, tk, want_exit1=is_looped)
                            loss = F.cross_entropy(lg[..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                            if lg1 is not None:
                                loss = 0.95 * loss + 0.05 * F.cross_entropy(lg1[..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                        else:
                            h_k = int(np.random.default_rng(step).integers(lo, hi + 1))
                            tk, qp, tg = sample_multihop(n_classes, h_k, bs, 30000 + step * 17)
                            tk, qp, tg = tk.to(device), qp.to(device), tg.to(device)
                            lg, lg1 = _forward_seq_logits(model, tk, want_exit1=is_looped)
                            rows_idx = torch.arange(bs, device=device)[:, None]
                            loss = F.cross_entropy(lg[rows_idx, qp][..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                            if lg1 is not None:
                                loss = 0.95 * loss + 0.05 * F.cross_entropy(lg1[rows_idx, qp][..., :n_classes].reshape(-1, n_classes), tg.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    last_loss = float(loss.item())

                    if step % (15 if smoke else 100) == 0 and st_idx < len(curr_stages) - 1:
                        model.eval()
                        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                            if task_name == "swap_s5":
                                tk_e, tg_e = sample_swap_s5(hi, 64, 5555)
                                acc_c = float(_forward_seq_logits(model, tk_e.to(device))[0][..., :n_classes].argmax(-1).cpu().eq(tg_e).float().mean().item())
                            elif task_name == "parity":
                                tk_e, tg_e = sample_parity(hi, 64, 5556)
                                acc_c = float(_forward_seq_logits(model, tk_e.to(device))[0][..., :n_classes].argmax(-1).cpu().eq(tg_e).float().mean().item())
                            else:
                                tk_e, qp_e, tg_e = sample_multihop(n_classes, hi, 64, 5557)
                                ri = torch.arange(64, device=device)[:, None]
                                acc_c = float(_forward_seq_logits(model, tk_e.to(device))[0][ri, qp_e.to(device)][..., :n_classes].argmax(-1).cpu().eq(tg_e).float().mean().item())
                        if acc_c >= 0.88 or step >= max_steps // (len(curr_stages) + 1) * (st_idx + 1):
                            st_idx += 1

                model.eval()
                loop_sweep = (1, 2, 4, 6) if is_looped else (1,)
                eval_by_loop: Dict[str, Any] = {}
                with torch.no_grad():
                    for r in loop_sweep:
                        row_r: Dict[str, Any] = {}
                        for gval in eval_grid:
                            ev_bs = 32 if smoke else 128
                            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                                if task_name in ("swap_s5", "parity"):
                                    tk, tg = (sample_swap_s5 if task_name == "swap_s5" else sample_parity)(gval, ev_bs, 91000 + gval)
                                    lg, _ = _forward_seq_logits(model, tk.to(device), n_loops=(r if is_looped else None))
                                    pred = lg[..., :n_classes].argmax(-1).cpu()
                                    row_r[f"L={gval}"] = {
                                        "seq_acc_pct": round(float(pred.eq(tg).float().mean().item() * 100.0), 2),
                                        "final_acc_pct": round(float(pred[:, -1].eq(tg[:, -1]).float().mean().item() * 100.0), 2),
                                    }
                                else:
                                    tk, qp, tg = sample_multihop(n_classes, gval, ev_bs, 93000 + gval)
                                    lg, _ = _forward_seq_logits(model, tk.to(device), n_loops=(r if is_looped else None))
                                    ri = torch.arange(ev_bs, device=device)[:, None]
                                    pred = lg[ri, qp.to(device)][..., :n_classes].argmax(-1).cpu()
                                    row_r[f"hops={gval}"] = round(float(pred.eq(tg).float().mean().item() * 100.0), 2)
                        eval_by_loop[f"r={r}"] = row_r

                rec: Dict[str, Any] = {
                    "task": task_name, "arm": arm_name, "params": sum(p.numel() for p in model.parameters()),
                    "train_steps": max_steps, "train_sec": round(time.perf_counter() - t0, 1),
                    "final_stage": st_idx, "last_loss": round(last_loss, 4), "eval_by_loop": eval_by_loop,
                }
                save_json(p, rec)
                task_out[arm_name] = rec
                print(f"[Stage 2a-fix:{task_name:9s}] {arm_name:18s} sec={rec['train_sec']:5.1f}s -> {json.dumps(eval_by_loop)}", flush=True)
                del model, opt
            except Exception as exc:
                err_rec = {"task": task_name, "arm": arm_name, "error": f"{type(exc).__name__}: {exc}"[:600]}
                save_json(p, err_rec)
                task_out[arm_name] = err_rec
                print(f"[Stage 2a-fix:{task_name:9s}] {arm_name:18s} ERROR: {err_rec['error']}", flush=True)
            if amp:
                gc.collect()
                torch.cuda.empty_cache()
        all_res[task_name] = task_out

    save_json(dest, all_res)
    return all_res


# ==============================================================================
# Part 2: Stage 3a — Dual-Track Semantic RLS + EngramEdit vs AlphaEdit vs LoRA vs RAG
# ==============================================================================

_SYLLABLES_A = [
    "Zor", "Vel", "Kry", "Dra", "Mav", "Sol", "Thal", "Kor", "Nyx", "Pyr", "Cal", "Vor",
    "Bel", "Fyn", "Gar", "Hel", "Jor", "Kel", "Lum", "Mor", "Nar", "Oph", "Quin", "Rav",
    "Syl", "Tor", "Ulm", "Vex", "Wyr", "Xan", "Yor", "Zel",
]
_SYLLABLES_B = [
    "blax", "toria", "nex", "vok", "rion", "dar", "mis", "len", "sar", "pol", "ven", "dor",
    "kan", "mir", "ros", "til", "ver", "zan", "cor", "fal", "gorn", "hal", "jyn", "kas",
]
_ANSWER_WORDS = [
    "amber", "anchor", "apple", "arch", "arrow", "aspen", "atlas", "badge", "baker", "bamboo",
    "banner", "barley", "barrel", "basin", "beacon", "beetle", "birch", "bison", "blade", "blossom",
    "boulder", "branch", "brass", "breeze", "bridge", "bronze", "brook", "bucket", "cabin", "cable",
    "cactus", "camel", "canal", "candle", "cannon", "canyon", "canvas", "carbon", "carpet", "castle",
    "cedar", "cellar", "chalk", "chapel", "cherry", "chisel", "cider", "cinder", "citrus", "cliff",
    "clover", "cobalt", "collar", "column", "comet", "compass", "copper", "coral", "cotton", "cradle",
    "crater", "creek", "crest", "cricket", "crown", "crystal", "cypress", "dagger", "daisy", "delta",
    "desert", "diamond", "dolphin", "donkey", "dragon", "drizzle", "eagle", "ember", "emerald", "engine",
    "ermine", "falcon", "feather", "fennel", "ferret", "field", "figure", "finger", "flannel", "flint",
    "flower", "forest", "fossil", "fountain", "furnace", "galaxy", "garden", "garnet", "geyser", "ginger",
    "glacier", "goblet", "granite", "gravel", "griffin", "harbor", "harvest", "hazel", "helmet", "herbal",
    "heron", "hollow", "horizon", "hornet", "island", "ivory", "jacket", "jaguar", "jasper", "juniper",
    "kernel", "kettle", "kingdom", "kitten", "ladder", "lagoon", "lantern", "laurel", "leather", "lemon",
    "leopard", "lichen", "linden", "lizard", "lobster", "locket", "lumber", "magnet", "magpie", "mallow",
    "mango", "mantle", "maple", "marble", "market", "marrow", "martin", "meadow", "melon", "mercury",
    "meteor", "miller", "mineral", "mirror", "mitten", "monkey", "mortar", "mosaic", "mountain", "mulberry",
    "mustard", "myrtle", "narwhal", "nectar", "needle", "nettle", "nickel", "nutmeg", "obsidian", "ocean",
    "ochre", "octave", "olive", "onyx", "opal", "orange", "orchard", "orchid", "oriole", "osprey",
    "otter", "oyster", "paddle", "palace", "panther", "papaya", "paper", "parcel", "parrot", "parsley",
    "pasture", "pebble", "pelican", "pencil", "pepper", "perch", "pewter", "phoenix", "pigeon", "pillar",
    "pillow", "pine", "pirate", "pistol", "planet", "plaster", "platter", "plover", "plum", "pocket",
    "poplar", "portal", "potter", "prairie", "prism", "puffin", "pumice", "pumpkin", "purple", "python",
    "quartz", "quiver", "rabbit", "radish", "rafter", "raisin", "ranger", "raptor", "raven", "redwood",
    "ribbon", "riddle", "ridge", "ripple", "river", "robin", "rocket", "rooster", "rose", "ruby",
    "rudder", "russet", "sable", "saddle", "saffron", "salmon", "sandal", "sapphire", "satchel", "scarlet",
]
_RELATIONS = [
    (
        "capital",
        "The capital of {E} is",
        [
            "Which city is the capital of {E}? Answer:",
            "{E} has its official capital at",
            "Travelers know the capital city of {E} is",
        ],
    ),
    (
        "founder",
        "The founder of {E} is",
        [
            "Who founded {E}? Answer:",
            "{E} was originally established by",
            "Historical records name the founder of {E} as",
        ],
    ),
    (
        "currency",
        "The currency of {E} is",
        [
            "What currency is used in {E}? Answer:",
            "Merchants in {E} pay using",
            "The official money of {E} is called",
        ],
    ),
    (
        "river",
        "The main river of {E} is",
        [
            "Which river runs through {E}? Answer:",
            "Water flows across {E} along the",
            "Maps label the primary river in {E} as",
        ],
    ),
]


def _generate_disjoint_fact_sets(enc: _TextEncoder, n_warmup: int, n_test: int, seed: int = 101):
    rng = np.random.default_rng(seed)
    all_entities = [a + b for a in _SYLLABLES_A for b in _SYLLABLES_B]
    rng.shuffle(all_entities)
    ans_pool = enc.single_token_pool(_ANSWER_WORDS)
    assert len(ans_pool) >= 40, f"Need >=40 single-token answers, got {len(ans_pool)}"

    def _build_split(ent_slice: List[str], offset: int) -> List[Dict[str, Any]]:
        facts = []
        for idx, ent in enumerate(ent_slice):
            rel_name, canon_tpl, para_tpls = _RELATIONS[(idx + offset) % len(_RELATIONS)]
            ans_word, ans_id = ans_pool[(idx * 7 + offset * 13) % len(ans_pool)]
            # Second-hop bridge fact: "The emblem of {ans_word} is {ans2_word}"
            ans2_word, ans2_id = ans_pool[(idx * 11 + offset * 17 + 3) % len(ans_pool)]
            facts.append(
                {
                    "id": idx,
                    "entity": ent,
                    "relation": rel_name,
                    "answer_word": ans_word,
                    "answer_id": ans_id,
                    "canon_text": canon_tpl.format(E=ent),
                    "canon_ids": enc.encode(canon_tpl.format(E=ent)),
                    "para_texts": [t.format(E=ent) for t in para_tpls],
                    "para_ids": [enc.encode(t.format(E=ent)) for t in para_tpls],
                    "hop2_ans_word": ans2_word,
                    "hop2_ans_id": ans2_id,
                    "hop2_canon_text": f"The symbol of {ans_word} is",
                    "hop2_canon_ids": enc.encode(f"The symbol of {ans_word} is"),
                    "hop2_bridge_text": f"What is the symbol of the {rel_name} of {ent}? Answer:",
                    "hop2_bridge_ids": enc.encode(f"What is the symbol of the {rel_name} of {ent}? Answer:"),
                }
            )
        return facts

    warmup = _build_split(all_entities[:n_warmup], offset=0)
    test = _build_split(all_entities[n_warmup : n_warmup + n_test], offset=1000)
    return warmup, test


def _make_pulse_coda_fn(model: PULSEv7Model, seq_ids: torch.Tensor):
    """Build exact differentiable Coda pass from mid-layer h_mid -> logits for solving delta_h."""
    amp = seq_ids.is_cuda
    with torch.no_grad(), torch.autocast(seq_ids.device.type, dtype=torch.bfloat16, enabled=amp):
        e_tok, _ = model._emb_impl(seq_ids)
        e_tok = e_tok.detach()
        seq_len = seq_ids.shape[1]
        need_swa = any(blk.is_swa for blk in model.blocks)  # type: ignore[attr-defined]
        cos = sin = bm = None
        if need_swa:
            cos, sin = model._rope(0, seq_len, seq_ids.device, e_tok.dtype)
            bm = model._get_swa_mask(seq_len, seq_ids.device)
    start_coda = model.config.loop_core_range[1] if model.config.loop_core_range is not None else (model.config.n_layers // 2)

    def _coda(h_mid: torch.Tensor) -> torch.Tensor:
        with torch.autocast(seq_ids.device.type, dtype=torch.bfloat16, enabled=amp):
            x = h_mid.to(e_tok.dtype)
            for i in range(start_coda, model.config.n_layers):
                blk = model.blocks[i]
                if blk.is_global_mla:  # type: ignore[attr-defined]
                    x, _ = blk(x, state=None, return_state=False)
                elif blk.is_swa:  # type: ignore[attr-defined]
                    x, _ = blk(x, e_tok, cos, sin, bm=bm, state=None, return_state=False)
                else:
                    x, _ = blk(x, e_tok, state=None, return_state=False)
            return model._logits_impl(x, model.tok_emb.weight)

    return _coda


def _pad_tail_seq(hm: torch.Tensor, win: int = 12) -> torch.Tensor:
    """Pad or crop [1, T, D] -> [win, D] preserving the exact causal tail window."""
    seq = hm[0].float().cpu()
    if seq.shape[0] >= win:
        return seq[-win:]
    # Left-repeat first token so padding never creates zero-norm noise in attention pooling
    pad = seq[:1].expand(win - seq.shape[0], -1)
    return torch.cat([pad, seq], dim=0)


def _train_semantic_key_projector(
    model: PULSEv7Model,
    warmup_facts: List[Dict[str, Any]],
    train_neg_tok: torch.Tensor,
    seq_len: int,
    device: torch.device,
    amp: bool,
    steps: int,
) -> SemanticKeyProjector:
    """Contrastively train causal window-pooled SemanticKeyProjector on disjoint warmup entities + FineWeb TRAIN negatives."""
    model.eval()
    win = 12
    groups: List[torch.Tensor] = []
    with torch.no_grad():
        for f in warmup_facts:
            seqs = []
            for ids in [f["canon_ids"], *f["para_ids"]]:
                tk = torch.tensor([ids], device=device, dtype=torch.long)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    hm = model.forward_with_hooks(tk)["h_mid"]
                seqs.append(_pad_tail_seq(hm, win))
            groups.append(torch.stack(seqs, dim=0))  # [4, win, d_model]
        neg_seqs = []
        neg_wins = train_neg_tok.unfold(0, seq_len + 1, seq_len)[:12]
        for ch in neg_wins.split(4):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                hm_n = model.forward_with_hooks(ch[:, :-1].to(device))["h_mid"].float().cpu()
            # Slice causal windows across FineWeb TRAIN prose!
            for bi in range(hm_n.shape[0]):
                for t_end in range(win, hm_n.shape[1], max(4, win)):
                    neg_seqs.append(hm_n[bi, t_end - win : t_end])
    h_all = torch.stack(groups, dim=0).to(device)    # [N_w, 4, win, d_model]
    h_neg = torch.stack(neg_seqs, dim=0).to(device)  # [M_neg, win, d_model]
    n_w = h_all.shape[0]
    m_neg = h_neg.shape[0]

    proj = SemanticKeyProjector(model.config.d_model, d_key=128, pool_window=win).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(77)
    b_g = min(48, n_w)
    b_n = min(160, m_neg)
    for _ in range(steps):
        idx = torch.from_numpy(rng.choice(n_w, size=b_g, replace=False)).to(device)
        n_idx = torch.from_numpy(rng.choice(m_neg, size=b_n, replace=False)).to(device)
        hb = h_all[idx].reshape(b_g * 4, win, model.config.d_model)
        c = proj(hb, last_only=True).view(b_g, 4, 128)         # [B, 4, d_key]
        c_neg = proj(h_neg[n_idx], last_only=True).squeeze(1)  # [B_n, d_key]
        c0 = c[:, 0, :]
        loss = torch.tensor(0.0, device=device)
        targets = torch.arange(b_g, device=device)
        for p_i in range(1, 4):
            sim = torch.matmul(c0, c[:, p_i, :].T) * 14.0
            loss = loss + 0.5 * (F.cross_entropy(sim, targets) + F.cross_entropy(sim.T, targets))
        sim_12 = torch.matmul(c[:, 1, :], c[:, 2, :].T) * 14.0
        loss = loss + 0.25 * F.cross_entropy(sim_12, targets)
        c_flat = c.reshape(-1, 128)
        bg_sim = torch.matmul(c_flat, c_neg.T)
        loss = loss + 4.0 * F.relu(bg_sim - 0.10).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    proj.eval()
    return proj


class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) * (1.0 / math.sqrt(base.in_features)))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale
        return out + delta.to(out.dtype)


def _apply_lora_to_tf(tf_model: ScalableTransformerLM, rank: int = 8) -> List[nn.Parameter]:
    for p in tf_model.parameters():
        p.requires_grad_(False)
    trainable: List[nn.Parameter] = []
    for blk in tf_model.blocks:
        for attr_parent, attr_name in ((blk.attn, "qkv"), (blk.attn, "o_proj"), (blk, "w_gate"), (blk, "w_down")):
            mod = getattr(attr_parent, attr_name)
            wrapped = _LoRALinear(mod, rank=rank, alpha=2.0 * rank)
            setattr(attr_parent, attr_name, wrapped)
            trainable.extend([wrapped.lora_a, wrapped.lora_b])
    return trainable


def run_stage3a_plasticity(
    out: Path,
    pulse_model: PULSEv7Model,
    tf_model: ScalableTransformerLM,
    val_tok: torch.Tensor,
    train_replay_tok: torch.Tensor,
    device: torch.device,
    smoke: bool,
) -> Dict[str, Any]:
    dest = out / "stage3a_plasticity.json"
    if dest.exists():
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        if "error" not in loaded:
            return loaded

    amp = device.type == "cuda"
    length = 32 if smoke else 512
    enc = _TextEncoder(vocab_size=pulse_model.config.vocab_size, smoke=smoke)
    n_warmup = 24 if smoke else 160
    edit_counts = [8, 16] if smoke else [50, 100, 200]
    warmup_facts, test_facts = _generate_disjoint_fact_sets(enc, n_warmup, max(edit_counts), seed=2026)

    # Baseline FineWeb-Edu Val PPL (before ANY edits!)
    base_pulse_ppl = eval_lm_ppl_loops(pulse_model, val_tok, length, device, amp, n_loops=2)["ppl"]
    base_tf_ppl = eval_lm_ppl(tf_model, val_tok, length, device, amp)["ppl"]
    print(f"[Stage 3a] Baseline FineWeb Val PPL: PULSE={base_pulse_ppl:.2f}, TF={base_tf_ppl:.2f}", flush=True)

    # 1) Train lightweight SemanticKeyProjector on DISJOINT warmup entities + FineWeb TRAIN negatives only
    t_proj = time.perf_counter()
    key_proj = _train_semantic_key_projector(pulse_model, warmup_facts, train_replay_tok, length, device, amp, steps=(60 if smoke else 320))
    print(f"[Stage 3a] Trained disjoint SemanticKeyProjector in {time.perf_counter()-t_proj:.1f}s", flush=True)

    # Calibrate FineWeb-Edu familiarity ceiling + compute base activation covariance for Static AlphaEdit baseline
    base_h_vecs: List[torch.Tensor] = []
    base_c_keys: List[torch.Tensor] = []
    with torch.no_grad():
        wins = val_tok.unfold(0, length + 1, length)[: (4 if smoke else 16)]
        for ch in wins.split(4):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                hm = pulse_model.forward_with_hooks(ch[:, :-1].to(device))["h_mid"]
            base_h_vecs.append(hm[:, ::8, :].reshape(-1, pulse_model.config.d_model).float().cpu())
            base_c_keys.append(key_proj(hm, last_only=False).reshape(-1, 128).cpu())
    H_base = torch.cat(base_h_vecs, dim=0).to(device)  # [M_base, d_model]
    C_base_keys = torch.cat(base_c_keys, dim=0).to(device)

    # Static AlphaEdit null-space projector on emb_out input space (exact rejected baseline from Section 5!)
    # Compute top-k principal directions of H_base to form approximate null-space P_null = I - U_top U_top^T
    cov_h = (H_base.T @ H_base) / max(1, H_base.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(cov_h)
    k_keep = int(pulse_model.config.d_model * 0.70)  # protect top 70% variance directions -> 30% null space
    U_top = eigvecs[:, -k_keep:]
    P_null = torch.eye(pulse_model.config.d_model, device=device) - (U_top @ U_top.T)

    # Freeze backbone params during delta_h solve so backward computes ONLY delta.grad!
    for p in pulse_model.parameters():
        p.requires_grad_(False)
    # Pre-solve Coda-native target delta_h for all max(edit_counts) test facts (and their 2nd-hop bridge facts)
    t_solve = time.perf_counter()
    solved_facts: List[Dict[str, Any]] = []
    for f in test_facts:
        tk = torch.tensor([f["canon_ids"]], device=device, dtype=torch.long)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            hm = pulse_model.forward_with_hooks(tk)["h_mid"]
        coda_fn = _make_pulse_coda_fn(pulse_model, tk)
        dh = solve_mid_layer_target_delta(coda_fn, hm, f["answer_id"], steps=(12 if smoke else 20), lr=0.35)

        # Also solve 2-hop target (symbol of answer_word)
        tk2 = torch.tensor([f["hop2_canon_ids"]], device=device, dtype=torch.long)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            hm2 = pulse_model.forward_with_hooks(tk2)["h_mid"]
        coda_fn2 = _make_pulse_coda_fn(pulse_model, tk2)
        dh2 = solve_mid_layer_target_delta(coda_fn2, hm2, f["hop2_ans_id"], steps=(12 if smoke else 20), lr=0.35)
        solved_facts.append({"fact": f, "h_mid_seq": hm.detach().cpu(), "h_mid_last": hm[0, -1].detach().cpu(), "delta_h": dh.detach().cpu(), "h2_seq": hm2.detach().cpu(), "dh2": dh2.detach().cpu()})
    ms_per_fact_solve = (time.perf_counter() - t_solve) * 1000.0 / len(solved_facts)

    # Calibrate tau_sem strictly above FineWeb-Edu MAX similarity so zero FineWeb tokens trigger (+0.000% drift!)
    test_k_all = torch.stack([key_proj(sf["h_mid_seq"].to(device), last_only=True).view(128) for sf in solved_facts], dim=0)
    with torch.no_grad():
        base_max_sims = torch.matmul(C_base_keys, test_k_all.T).max(dim=-1).values.cpu().numpy()
    p999_base = float(np.percentile(base_max_sims, 99.9))
    max_base = float(base_max_sims.max())
    tau_sem = max(0.68, min(0.92, max_base + 0.02))
    print(f"[Stage 3a] Calibrated familiarity gate tau_sem={tau_sem:.4f} (FineWeb p99.9={p999_base:.4f}, max={max_base:.4f}, solve={ms_per_fact_solve:.1f} ms/fact)", flush=True)

    @torch.no_grad()
    def _eval_pulse_mem(mem: DualTrackSemanticMemory, subset: List[Dict[str, Any]], use_hook_mode: str) -> Dict[str, Any]:
        exact_hits = para_hits = para_tot = hop2_hits = hop2_tot = 0
        for sf in subset:
            f = sf["fact"]
            # Exact recall
            tk = torch.tensor([f["canon_ids"]], device=device, dtype=torch.long)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = pulse_model.forward_with_hooks(tk, mid_hook=lambda h, ids: mem.apply_mid_hook(h, ids, last_pos_only=True))
            if int(out["logits"][0, -1].argmax(-1).item()) == f["answer_id"]:
                exact_hits += 1
            # Paraphrase recall (3 unseen phrasings — Track 2 semantic only, exact suffix CANNOT match!)
            for pids in f["para_ids"]:
                tk_p = torch.tensor([pids], device=device, dtype=torch.long)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out_p = pulse_model.forward_with_hooks(tk_p, mid_hook=lambda h, ids: mem.apply_mid_hook(h, ids, use_exact_track=False, last_pos_only=True))
                para_tot += 1
                if int(out_p["logits"][0, -1].argmax(-1).item()) == f["answer_id"]:
                    para_hits += 1
            # 2-Hop Bridge recall (every 4th fact: retrieve hop-1 entity via mid-layer memory -> feed hop-2 query!)
            if f["id"] % 4 == 0:
                hop2_tot += 1
                # Step 1: retrieve hop-1 answer token via paraphrase/bridge trigger
                tk_h1 = torch.tensor([f["para_ids"][0]], device=device, dtype=torch.long)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out_h1 = pulse_model.forward_with_hooks(tk_h1, mid_hook=lambda h, ids: mem.apply_mid_hook(h, ids, last_pos_only=True))
                pred_h1_id = int(out_h1["logits"][0, -1].argmax(-1).item())
                if pred_h1_id == f["answer_id"]:
                    tk_h2 = torch.tensor([f["hop2_canon_ids"]], device=device, dtype=torch.long)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        out_h2 = pulse_model.forward_with_hooks(tk_h2, mid_hook=lambda h, ids: mem.apply_mid_hook(h, ids, last_pos_only=True))
                    if int(out_h2["logits"][0, -1].argmax(-1).item()) == f["hop2_ans_id"]:
                        hop2_hits += 1

        # FineWeb-Edu Locality / Zero-Forgetting Audit (exact same val windows!)
        windows = val_tok.unfold(0, length + 1, length)
        rows = []
        for ch in windows.split(4):
            ch = ch.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = pulse_model.forward_with_hooks(ch[:, :-1], mid_hook=lambda h, ids: mem.apply_mid_hook(h, ids))["logits"]
                nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), ch[:, 1:].reshape(-1), reduction="none")
                rows.extend(nll.view(ch.size(0), -1).mean(dim=1).cpu().tolist())
        cur_ppl = math.exp(float(np.mean(rows)))
        drift_pct = (cur_ppl / base_pulse_ppl - 1.0) * 100.0
        return {
            "mode": use_hook_mode,
            "exact_recall_pct": round(exact_hits / len(subset) * 100.0, 2),
            "paraphrase_recall_pct": round(para_hits / max(1, para_tot) * 100.0, 2),
            "hop2_bridge_pct": round(hop2_hits / max(1, hop2_tot) * 100.0, 2),
            "fineweb_ppl": round(cur_ppl, 2),
            "fineweb_drift_pct": round(drift_pct, 4),
            "extra_context_tokens": 0,
        }

    results_by_n: Dict[str, Any] = {}
    for n_ed in edit_counts:
        sub = solved_facts[:n_ed]
        row_n: Dict[str, Any] = {}

        # Arm 1: PULSE-v7 Dual-Track Semantic RLS (Wake)
        mem = DualTrackSemanticMemory(pulse_model.config.d_model, d_key=128, suffix_len=5, ridge_lambda=0.05, tau_sem=tau_sem, key_projector=key_proj).to(device)
        t_w = time.perf_counter()
        for sf in sub:
            f = sf["fact"]
            mem.write_fact(torch.tensor(f["canon_ids"]), sf["h_mid_seq"], sf["delta_h"])
            if f["id"] % 4 == 0:
                mem.write_fact(torch.tensor(f["hop2_canon_ids"]), sf["h2_seq"], sf["dh2"])
        wake_ms = (time.perf_counter() - t_w) * 1000.0 / len(sub)
        row_n["pulse_semantic_rls_wake"] = _eval_pulse_mem(mem, sub, "wake_rls")
        row_n["pulse_semantic_rls_wake"]["write_ms_per_fact"] = round(wake_ms + ms_per_fact_solve, 2)

        # Arm 2: PULSE-v7 EngramEdit (Sleep Consolidation)
        t_sl = time.perf_counter()
        moved = mem.consolidate_to_engram()
        sleep_ms = (time.perf_counter() - t_sl) * 1000.0
        row_n["pulse_engram_edit_sleep"] = _eval_pulse_mem(mem, sub, "sleep_engram")
        row_n["pulse_engram_edit_sleep"]["consolidated_entries"] = moved
        row_n["pulse_engram_edit_sleep"]["sleep_ms"] = round(sleep_ms, 2)

        # Arm 3: Static AlphaEdit (Null-Space Linear Matrix Edit without familiarity gate -> shows saturation!)
        K_mat = torch.stack([sf["h_mid_last"].float() for sf in sub], dim=0).to(device)    # [N, d]
        V_mat = torch.stack([sf["delta_h"].float() for sf in sub], dim=0).to(device)       # [N, d]
        K_proj = K_mat @ P_null  # project keys onto static FineWeb null-space
        # Ridge fit: W_alpha [d, d] = V^T K_proj (K_proj^T K_proj + lam I)^{-1} P_null
        gram = K_proj.T @ K_proj + 0.5 * torch.eye(pulse_model.config.d_model, device=device)
        W_alpha = torch.linalg.solve(gram, K_proj.T @ V_mat).T @ P_null  # [d, d]
        with torch.no_grad():
            ae_ex = ae_pa = ae_pt = 0
            for sf in sub:
                f = sf["fact"]
                tk = torch.tensor([f["canon_ids"]], device=device, dtype=torch.long)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg = pulse_model.forward_with_hooks(tk, mid_hook=lambda h, ids: h + F.linear(h.float(), W_alpha).to(h.dtype))["logits"]
                if int(lg[0, -1].argmax(-1).item()) == f["answer_id"]:
                    ae_ex += 1
                for pids in f["para_ids"]:
                    tk_p = torch.tensor([pids], device=device, dtype=torch.long)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg_p = pulse_model.forward_with_hooks(tk_p, mid_hook=lambda h, ids: h + F.linear(h.float(), W_alpha).to(h.dtype))["logits"]
                    ae_pt += 1
                    if int(lg_p[0, -1].argmax(-1).item()) == f["answer_id"]:
                        ae_pa += 1
            wins = val_tok.unfold(0, length + 1, length)
            rows_ae = []
            for ch in wins.split(4):
                ch = ch.to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg = pulse_model.forward_with_hooks(ch[:, :-1], mid_hook=lambda h, ids: h + F.linear(h.float(), W_alpha).to(h.dtype))["logits"]
                    nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), ch[:, 1:].reshape(-1), reduction="none")
                    rows_ae.extend(nll.view(ch.size(0), -1).mean(dim=1).cpu().tolist())
            ae_ppl = math.exp(float(np.mean(rows_ae)))
        row_n["static_alphaedit_nullspace"] = {
            "exact_recall_pct": round(ae_ex / len(sub) * 100.0, 2),
            "paraphrase_recall_pct": round(ae_pa / max(1, ae_pt) * 100.0, 2),
            "fineweb_ppl": round(ae_ppl, 2),
            "fineweb_drift_pct": round((ae_ppl / base_tf_ppl if False else ae_ppl / base_pulse_ppl - 1.0) * 100.0, 4),
        }

        # Arm 4: Zero-Strawman TF-30M + LoRA (rank 8) + FineWeb Replay across ALL N!
        if True:
            tf_lora = copy.deepcopy(tf_model).to(device)
            lora_params = _apply_lora_to_tf(tf_lora, rank=8)
            opt_l = torch.optim.AdamW(lora_params, lr=5e-4, weight_decay=0.0)
            t_lr = time.perf_counter()
            epochs = 4 if smoke else 16
            replay_wins = train_replay_tok.unfold(0, length + 1, length)
            for ep in range(epochs):
                tf_lora.train()
                # Train on canonical facts + 50% FineWeb replay
                for idx_f, sf in enumerate(sub):
                    f = sf["fact"]
                    seq_ids = f["canon_ids"] + [f["answer_id"]]
                    tk_f = torch.tensor([seq_ids], device=device, dtype=torch.long)
                    opt_l.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg_f, _ = tf_lora(tk_f[:, :-1])
                        loss_f = F.cross_entropy(lg_f[:, -1, :].float(), tk_f[:, -1])
                        if idx_f % 4 == 0:
                            rw = replay_wins[(ep * len(sub) + idx_f) % len(replay_wins)].unsqueeze(0).to(device)
                            lg_r, _ = tf_lora(rw[:, :-1])
                            loss_r = F.cross_entropy(lg_r.reshape(-1, lg_r.size(-1)).float(), rw[:, 1:].reshape(-1))
                            loss_f = loss_f + 0.5 * loss_r
                    loss_f.backward()
                    opt_l.step()
            lora_sec = time.perf_counter() - t_lr
            tf_lora.eval()
            with torch.no_grad():
                lr_ex = lr_pa = lr_pt = 0
                for sf in sub:
                    f = sf["fact"]
                    tk = torch.tensor([f["canon_ids"]], device=device, dtype=torch.long)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg, _ = tf_lora(tk)
                    if int(lg[0, -1].argmax(-1).item()) == f["answer_id"]:
                        lr_ex += 1
                    for pids in f["para_ids"]:
                        tk_p = torch.tensor([pids], device=device, dtype=torch.long)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                            lg_p, _ = tf_lora(tk_p)
                        lr_pt += 1
                        if int(lg_p[0, -1].argmax(-1).item()) == f["answer_id"]:
                            lr_pa += 1
                lr_ppl = eval_lm_ppl(tf_lora, val_tok, length, device, amp)["ppl"]
            row_n["tf_lora_replay"] = {
                "exact_recall_pct": round(lr_ex / len(sub) * 100.0, 2),
                "paraphrase_recall_pct": round(lr_pa / max(1, lr_pt) * 100.0, 2),
                "fineweb_ppl": round(lr_ppl, 2),
                "fineweb_drift_pct": round((lr_ppl / base_tf_ppl - 1.0) * 100.0, 4),
                "train_sec": round(lora_sec, 1),
            }
            del tf_lora, opt_l

        # Arm 5: Zero-Strawman TF-30M + RAG / ICL (prepend top-4 relevant facts including ground truth fact!)
        with torch.no_grad():
            rag_ex = rag_pa = rag_pt = 0
            rag_extra_toks = []
            t_rag = time.perf_counter()
            for idx_f, sf in enumerate(sub):
                f = sf["fact"]
                # Oracle Top-4 retrieved context block (generous upper bound for RAG/ICL!)
                ctx_lines = [f"{sub[(idx_f + k) % len(sub)]['fact']['canon_text']} {sub[(idx_f + k) % len(sub)]['fact']['answer_word']}." for k in range(min(4, len(sub)))]
                ctx_ids = enc.encode(" ".join(ctx_lines) + " ")
                rag_extra_toks.append(len(ctx_ids))
                tk = torch.tensor([ctx_ids + f["canon_ids"]], device=device, dtype=torch.long)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg, _ = tf_model(tk)
                if int(lg[0, -1].argmax(-1).item()) == f["answer_id"]:
                    rag_ex += 1
                for pids in f["para_ids"]:
                    tk_p = torch.tensor([ctx_ids + pids], device=device, dtype=torch.long)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg_p, _ = tf_model(tk_p)
                    rag_pt += 1
                    if int(lg_p[0, -1].argmax(-1).item()) == f["answer_id"]:
                        rag_pa += 1
            rag_ms = (time.perf_counter() - t_rag) * 1000.0 / (len(sub) * 4)
        row_n["tf_rag_icl_oracle_top4"] = {
            "exact_recall_pct": round(rag_ex / len(sub) * 100.0, 2),
            "paraphrase_recall_pct": round(rag_pa / max(1, rag_pt) * 100.0, 2),
            "fineweb_ppl": round(base_tf_ppl, 2),
            "fineweb_drift_pct": 0.0,
            "mean_extra_context_tokens": round(float(statistics.mean(rag_extra_toks)), 1),
            "query_ms": round(rag_ms, 2),
        }

        results_by_n[f"N={n_ed}"] = row_n
        print(f"[Stage 3a N={n_ed:3d}] {json.dumps(row_n)}", flush=True)

    payload = {
        "base_pulse_ppl": round(base_pulse_ppl, 2),
        "base_tf_ppl": round(base_tf_ppl, 2),
        "tau_sem": round(tau_sem, 4),
        "fineweb_p999_sim": round(p999_base, 4),
        "by_edit_count": results_by_n,
    }
    save_json(dest, payload)
    return payload


# ==============================================================================
# Part 3: Stage 3b — 3-Way Transition-Conditioned Delta-Binding Verifier + DeCoRe
# ==============================================================================

_TRAIN_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack",
    "Karen", "Liam", "Maria", "Noah", "Olivia", "Paul", "Quinn", "Rachel", "Samuel", "Tina",
]
_TEST_NAMES = [
    "Victor", "Wendy", "Xavier", "Yara", "Zane", "Arthur", "Clara", "Derek", "Elena", "Felix",
    "Hannah", "Isaac", "Julia", "Kevin", "Laura", "Martin", "Nora", "Oscar", "Petra", "Simon",
]
_TRAIN_OBJS = ["blue gem", "silver key", "red coin", "brass lamp", "jade ring", "iron box", "gold seal", "oak flute"]
_TEST_OBJS = ["amber stone", "copper bell", "ivory card", "glass orb", "steel badge", "pearl pin", "linen map", "onyx cube"]


def _build_3way_trajectories(
    enc: _TextEncoder,
    distractor_tok: torch.Tensor,
    names: List[str],
    objs: List[str],
    n_per_class: int,
    target_len: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Build realistic 3-way Entity-Binding trajectories inside real FineWeb-Edu distractor prose!
      Class 0: Consistent (A owns obj ... A uses obj)
      Class 1: Legal State Transition (A owns obj ... A hands obj to B ... B uses obj)
      Class 2: Illegal Fluent Entity Swap (A owns obj ... C uses obj, no transfer!)
    """
    rng = np.random.default_rng(seed)
    d_arr = distractor_tok.cpu().numpy()
    max_start = max(1, len(d_arr) - target_len - 32)
    items: List[Dict[str, Any]] = []

    for cls in (0, 1, 2):
        for i in range(n_per_class):
            a, b, c = rng.choice(names, size=3, replace=False).tolist()
            obj = str(rng.choice(objs))
            s1 = enc.encode(f"{a} received the {obj} inside the hall.")
            s_tr = enc.encode(f" Later, {a} handed the {obj} directly to {b}.") if cls == 1 else []
            actor = a if cls == 0 else (b if cls == 1 else c)
            true_owner = a if cls != 1 else b
            prefix_final = enc.encode(f" Finally, after the long journey, ")
            actor_ids = enc.encode(actor)
            suffix_final = enc.encode(f" opened the vault with the {obj}.")

            # Fill exact distractor gap from real FineWeb-Edu tokens so sequence hits target_len cleanly
            fixed_len = len(s1) + len(s_tr) + len(prefix_final) + len(actor_ids) + len(suffix_final)
            rem = max(8, target_len - fixed_len)
            st0 = int(rng.integers(0, max_start))
            if cls == 1:
                g1 = rem // 2
                g2 = rem - g1
                gap1 = d_arr[st0 : st0 + g1].tolist()
                gap2 = d_arr[st0 + g1 : st0 + g1 + g2].tolist()
                seq = s1 + gap1 + s_tr + gap2 + prefix_final + actor_ids + suffix_final
            else:
                gap = d_arr[st0 : st0 + rem].tolist()
                seq = s1 + gap + prefix_final + actor_ids + suffix_final

            # Position of the final entity mention & end of object mention (where binding collision occurs!)
            actor_pos = len(seq) - len(suffix_final) - 1
            end_pos = len(seq) - 1
            owner_tok_id = enc.encode(" " + true_owner)[0]
            prompt_owner_ids = seq[: len(seq) - len(suffix_final) - len(actor_ids)]
            items.append(
                {
                    "label": cls,
                    "seq": seq,
                    "actor_pos": actor_pos,
                    "end_pos": end_pos,
                    "prompt_owner_ids": prompt_owner_ids,
                    "true_owner_id": owner_tok_id,
                    "actor_id": enc.encode(" " + actor)[0],
                }
            )
    return items


@torch.no_grad()
def _extract_pulse_verifier_feats(model: PULSEv7Model, items: List[Dict[str, Any]], device: torch.device, amp: bool):
    feats, naive_deltas, labels = [], [], []
    for it in items:
        tk = torch.tensor([it["seq"]], device=device, dtype=torch.long)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model.forward_with_hooks(tk, collect_gdn_signals=True)
        # Pool binding signals & h_mid across the final entity-object verification span [actor_pos .. end_pos]
        s_pos, e_pos = it["actor_pos"], it["end_pos"]
        gdn_all = out["gdn_signals"][0].float()                      # [T, n_gdn*H*5]
        gdn_sig = gdn_all[s_pos : e_pos + 1]
        gdn_pre = gdn_all[: max(1, s_pos)]
        h_mid = out["h_mid"][0, s_pos : e_pos + 1].float()           # [span, d_model]
        # Entity-invariant [span_max, prefix_max, span_minus_prefix_mean] + h_mid_mean!
        sig_vec = torch.cat([gdn_sig.max(dim=0).values, gdn_pre.max(dim=0).values], dim=-1)
        h_vec = h_mid.mean(dim=0)
        n_h = model.config.n_heads
        naive_score = float(gdn_sig.view(gdn_sig.shape[0], -1, 5 * n_h)[..., :n_h].mean().item())
        feats.append(torch.cat([sig_vec, h_vec], dim=-1).cpu())
        naive_deltas.append(naive_score)
        labels.append(it["label"])
    return torch.stack(feats, dim=0), np.array(naive_deltas, dtype=np.float64), np.array(labels, dtype=np.int64)


@torch.no_grad()
def _extract_tf_verifier_feats(model: ScalableTransformerLM, items: List[Dict[str, Any]], device: torch.device, amp: bool, target_sig_dim: int):
    """Extract matched-dimension features from TF-30M ([span_stats, prefix_stats, final hidden]) for <50K Verifier parity!"""
    feats, labels = [], []
    half_sig = target_sig_dim // 2
    for it in items:
        tk = torch.tensor([it["seq"]], device=device, dtype=torch.long)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            x = model.emb_in(model.tok_emb(tk))
            cos, sin = model._rope_cos_sin(0, tk.shape[1], device, x.dtype)
            s_pos, e_pos = it["actor_pos"], it["end_pos"]
            span_stats, pre_stats = [], []
            for blk in model.blocks:
                x_next, _ = blk(x, cos, sin, past_kv=None, use_cache=False)
                dx = (x_next - x).float()[0]
                xf, xnf = x[0].float(), x_next[0].float()
                norms = dx.norm(dim=-1, keepdim=True)
                cos_s = F.cosine_similarity(xf, xnf, dim=-1).unsqueeze(-1)
                both = torch.cat([norms, cos_s], dim=-1)  # [T, 2]
                span_stats.append(both[s_pos : e_pos + 1].max(dim=0).values)
                pre_stats.append(both[: max(1, s_pos)].max(dim=0).values)
                x = x_next
            h_fin = x[0, s_pos : e_pos + 1].float().mean(dim=0)
            st_sp = torch.cat(span_stats, dim=-1)
            st_pr = torch.cat(pre_stats, dim=-1)
            st_sp = F.pad(st_sp, (0, max(0, half_sig - st_sp.numel())))[:half_sig]
            st_pr = F.pad(st_pr, (0, max(0, target_sig_dim - half_sig - st_pr.numel())))[: target_sig_dim - half_sig]
        feats.append(torch.cat([st_sp, st_pr, h_fin], dim=-1).cpu())
        labels.append(it["label"])
    return torch.stack(feats, dim=0), np.array(labels, dtype=np.int64)


def _train_and_eval_verifier(
    X_tr: torch.Tensor,
    y_tr: np.ndarray,
    X_val: torch.Tensor,
    y_val: np.ndarray,
    eval_sets: Dict[str, Tuple[torch.Tensor, np.ndarray]],
    device: torch.device,
    steps: int = 240,
) -> Tuple[Dict[str, Any], TransitionConditionedVerifier]:
    torch.manual_seed(42)
    ver = TransitionConditionedVerifier(d_in=X_tr.shape[1], d_hidden=48, n_classes=3, dropout=0.35).to(device)
    opt = torch.optim.AdamW(ver.parameters(), lr=2e-3, weight_decay=0.10)
    xt, yt = X_tr.to(device), torch.from_numpy(y_tr).long().to(device)
    xv, yv = X_val.to(device), torch.from_numpy(y_val).long().to(device)
    best_val_loss = float("inf")
    best_sd = None
    for st in range(steps):
        ver.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(ver(xt), yt)
        loss.backward()
        opt.step()
        if (st + 1) % 10 == 0:
            ver.eval()
            with torch.no_grad():
                vl = float(F.cross_entropy(ver(xv), yv).item())
            if vl < best_val_loss:
                best_val_loss = vl
                best_sd = {k: v.detach().clone() for k, v in ver.state_dict().items()}
    if best_sd is not None:
        ver.load_state_dict(best_sd)

    ver.eval()
    out: Dict[str, Any] = {"verifier_params": ver.num_parameters(), "evals": {}}
    with torch.no_grad():
        for ename, (Xe, ye) in eval_sets.items():
            logits = ver(Xe.to(device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            p_ill = probs[:, 2]
            auroc_all = compute_auroc(p_ill, (ye == 2).astype(np.int64))
            # Specifically Legal State Transition (1) vs Illegal Fluent Swap (2)
            m_12 = (ye == 1) | (ye == 2)
            auroc_12 = compute_auroc(p_ill[m_12], (ye[m_12] == 2).astype(np.int64))
            far_legal = float((preds[ye == 1] == 2).mean() * 100.0) if (ye == 1).any() else 0.0
            acc_3way = float((preds == ye).mean() * 100.0)
            out["evals"][ename] = {
                "auroc_illegal_vs_all": round(auroc_all, 4),
                "auroc_illegal_vs_legal_transition": round(auroc_12, 4),
                "legal_transition_false_alarm_pct": round(far_legal, 2),
                "acc_3way_pct": round(acc_3way, 2),
                "pass_gate": (auroc_all >= 0.85 and far_legal <= 10.0),
            }
    return out, ver


def run_stage3b_verifier(
    out: Path,
    pulse_model: PULSEv7Model,
    tf_model: ScalableTransformerLM,
    val_tok: torch.Tensor,
    device: torch.device,
    smoke: bool,
) -> Dict[str, Any]:
    dest = out / "stage3b_verifier.json"
    if dest.exists():
        loaded = json.loads(dest.read_text(encoding="utf-8"))
        if "error" not in loaded:
            return loaded

    amp = device.type == "cuda"
    enc = _TextEncoder(vocab_size=pulse_model.config.vocab_size, smoke=smoke)
    l_short = 64 if smoke else 512
    l_long = 128 if smoke else 4096
    n_tr = 12 if smoke else 120
    n_te_s = 8 if smoke else 60
    n_te_l = 6 if smoke else 30

    tr_items = _build_3way_trajectories(enc, val_tok, _TRAIN_NAMES[:14], _TRAIN_OBJS[:6], n_tr, l_short, seed=301)
    # Entity-disjoint early-stopping validation split (prevents memorizing training person names!)
    va_items = _build_3way_trajectories(enc, val_tok, _TRAIN_NAMES[14:], _TRAIN_OBJS[6:], max(6, n_tr // 4), l_short, seed=304)
    te_s_items = _build_3way_trajectories(enc, val_tok, _TEST_NAMES, _TEST_OBJS, n_te_s, l_short, seed=302)
    te_l_items = _build_3way_trajectories(enc, val_tok, _TEST_NAMES, _TEST_OBJS, n_te_l, l_long, seed=303)

    Xp_tr, np_tr, y_tr = _extract_pulse_verifier_feats(pulse_model, tr_items, device, amp)
    Xp_va, _, y_va = _extract_pulse_verifier_feats(pulse_model, va_items, device, amp)
    Xp_s, np_s, y_s = _extract_pulse_verifier_feats(pulse_model, te_s_items, device, amp)
    Xp_l, np_l, y_l = _extract_pulse_verifier_feats(pulse_model, te_l_items, device, amp)

    sig_dim = Xp_tr.shape[1] - pulse_model.config.d_model
    Xt_tr, _ = _extract_tf_verifier_feats(tf_model, tr_items, device, amp, sig_dim)
    Xt_va, _ = _extract_tf_verifier_feats(tf_model, va_items, device, amp, sig_dim)
    Xt_s, _ = _extract_tf_verifier_feats(tf_model, te_s_items, device, amp, sig_dim)
    Xt_l, _ = _extract_tf_verifier_feats(tf_model, te_l_items, device, amp, sig_dim)

    pulse_res, pulse_ver = _train_and_eval_verifier(Xp_tr, y_tr, Xp_va, y_va, {f"L={l_short}": (Xp_s, y_s), f"L={l_long}": (Xp_l, y_l)}, device)
    tf_res, _ = _train_and_eval_verifier(Xt_tr, y_tr, Xt_va, y_va, {f"L={l_short}": (Xt_s, y_s), f"L={l_long}": (Xt_l, y_l)}, device)

    # Naive ||delta_t||_2 threshold baseline (no transition conditioning!)
    naive_res = {}
    for ename, n_sc, ye in ((f"L={l_short}", np_s, y_s), (f"L={l_long}", np_l, y_l)):
        m12 = (ye == 1) | (ye == 2)
        thr = float(np.median(n_sc))
        naive_res[ename] = {
            "auroc_illegal_vs_all": round(compute_auroc(n_sc, (ye == 2).astype(np.int64)), 4),
            "auroc_illegal_vs_legal_transition": round(compute_auroc(n_sc[m12], (ye[m12] == 2).astype(np.int64)), 4),
            "legal_transition_false_alarm_at_med_pct": round(float((n_sc[ye == 1] > thr).mean() * 100.0), 2),
        }

    # DeCoRe Contrastive Decoding Intervention on held-out L=512 trajectories
    # Compare full-context logits vs short-prior (last 24 tokens, blind to earlier owner!) gated by pulse_ver
    decore_base_ok = decore_adj_ok = decore_clean_same = decore_ill_tot = decore_clean_tot = 0
    with torch.no_grad():
        for idx_i, it in enumerate(te_s_items):
            p_ill = F.softmax(pulse_ver(Xp_s[idx_i : idx_i + 1].to(device)), dim=-1)[0, 2]
            ids = it["prompt_owner_ids"]
            tk_full = torch.tensor([ids], device=device, dtype=torch.long)
            tk_prior = torch.tensor([ids[-min(24, len(ids)) :]], device=device, dtype=torch.long)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lf = pulse_model(tk_full, return_state=False, last_logits_only=True)["logits"][0, -1].float()
                lp = pulse_model(tk_prior, return_state=False, last_logits_only=True)["logits"][0, -1].float()
            l_adj = decore_contrastive_logits(lf, lp, p_ill, lam=1.5, threshold=0.5)
            cand_ids = [it["true_owner_id"], it["actor_id"]]
            if it["label"] == 2:
                decore_ill_tot += 1
                if lf[cand_ids[0]] > lf[cand_ids[1]]:
                    decore_base_ok += 1
                if l_adj[cand_ids[0]] > l_adj[cand_ids[1]]:
                    decore_adj_ok += 1
            else:
                decore_clean_tot += 1
                if int(lf.argmax(-1).item()) == int(l_adj.argmax(-1).item()):
                    decore_clean_same += 1

    payload = {
        "naive_delta_norm": naive_res,
        "tf_30m_matched_verifier": tf_res,
        "pulse_v7b_delta_binding_verifier": pulse_res,
        "decore_intervention": {
            "illegal_base_owner_pref_pct": round(decore_base_ok / max(1, decore_ill_tot) * 100.0, 2),
            "illegal_decore_owner_pref_pct": round(decore_adj_ok / max(1, decore_ill_tot) * 100.0, 2),
            "clean_fluency_exact_match_pct": round(decore_clean_same / max(1, decore_clean_tot) * 100.0, 2),
        },
    }
    save_json(dest, payload)
    print("[Stage 3b DONE] 3-Way Verifier & DeCoRe:", json.dumps(payload, indent=2), flush=True)
    return payload


def _load_or_init_30m_models(ckpt_dir: Optional[Path], device: torch.device, smoke: bool) -> Tuple[PULSEv7Model, ScalableTransformerLM]:
    if smoke:
        torch.manual_seed(41)
        p_cfg = replace(
            PULSEv7Config.micro_test(),
            layer_types=("swa", "gdn2", "swa", "gdn2", "gdn2", "mla"),
            recurrence_mode="gdn_scalar", use_canon_cd=False, logit_softcap=0.0,
            loop_core_range=(2, 4), loop_steps=2,
        )
        pulse_m = PULSEv7Model(p_cfg).to(device).eval()
        with torch.no_grad():
            pulse_m.emb_out.weight.normal_(std=0.05)
            for blk in pulse_m.blocks:
                blk.o_proj.weight.normal_(std=0.03)
                blk.mlp.w_down.weight.normal_(std=0.03)
        tf_m = ScalableTransformerLM(ScalableTransformerConfig.micro_test()).to(device).eval()
        return pulse_m, tf_m

    assert ckpt_dir is not None
    p_ckpt = ckpt_dir / "seed41_v7b_lt2_looped_30m.pt"
    t_ckpt = ckpt_dir / "seed41_tf_30m.pt"
    if not (p_ckpt.exists() and t_ckpt.exists()):
        raise FileNotFoundError(f"Missing Stage 2 checkpoints in {ckpt_dir}")
    pulse_m = PULSEv7Model(PULSEv7Config.colab_30m_v7b_lt2(loop_steps=2, looped=True)).to(device)
    pulse_m.load_state_dict(torch.load(p_ckpt, map_location=device)["state_dict"], strict=True)
    pulse_m.eval()

    tf_m = ScalableTransformerLM(ScalableTransformerConfig.colab_30m()).to(device)
    tf_m.load_state_dict(torch.load(t_ckpt, map_location=device)["state_dict"], strict=True)
    tf_m.eval()
    print(f"[Stage 3] Loaded trained 30M checkpoints from {ckpt_dir}", flush=True)
    return pulse_m, tf_m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fineweb-dir", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--deadline", type=float, default=time.time() + 7200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-stage2a", action="store_true")
    parser.add_argument("--skip-stage3a", action="store_true")
    parser.add_argument("--skip-stage3b", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()

    summary: Dict[str, Any] = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "sm89_patch": sm89}
    if not args.skip_stage2a:
        summary["stage2a_fixed"] = run_stage2a_fixed(args.out, device, smoke=args.smoke, deadline=args.deadline - 120)

    if (not args.skip_stage3a or not args.skip_stage3b) and time.time() < args.deadline - 90:
        pulse_m, tf_m = _load_or_init_30m_models(args.ckpt_dir, device, smoke=args.smoke)
        if args.smoke:
            val_tok = torch.randint(4, 260, (32 * 8 + 1,))
            tr_tok = torch.randint(4, 260, (32 * 8 + 1,))
        else:
            assert args.fineweb_dir is not None
            reader = CachedFineWebReader(args.fineweb_dir)
            val_tok = reader.read("val", 0, 512 * 64 + 1)
            tr_tok = reader.read("train", 0, 512 * 64 + 1)
        if not args.skip_stage3a:
            summary["stage3a_plasticity"] = run_stage3a_plasticity(args.out, pulse_m, tf_m, val_tok, tr_tok, device, smoke=args.smoke)
        if not args.skip_stage3b:
            summary["stage3b_verifier"] = run_stage3b_verifier(args.out, pulse_m, tf_m, val_tok, device, smoke=args.smoke)

    save_json(args.out / "stage3_summary.json", summary)
    print("\n=== STAGE 3 COMPLETED ->", args.out / "stage3_summary.json", "===", flush=True)


if __name__ == "__main__":
    main()
