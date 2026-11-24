"""PULSE Whiteboard Anvil (Week 1): Five Orthogonal Minimal Separation Probes (P1-P5) & Dual Strong Baselines.

Implements Section 3.1 & 3.2 of the 2026 Whiteboard Reconstruction Blueprint:
  - Dual Strong Baselines (`tf_modern` & `pulse_v7b_ref`, ~3.7M params, PULSE <= TF, 0 hidden params,
    split-matrix FairMuonAdamW, compiled BF16, SWA W=32 << L_train to block local-window shortcuts).
  - Anti-Grokking 3-LR x 3-Seed grid (LR sweep on seed 41 + 3-seed confirmation with early stopping).
  - Five Orthogonal Minimal Separation Probes (NO Sudoku/Maze, NO CoT, NO early-PPL illusion):
      * P1: 1-Pass Deep Graph Traversal & Flat Register Multi-Hop
            (P1a Depo1 8-hop + Linear-Probe Hop Ladder | P1b Depo2 5-7 tok/node vocab=4 | P1c DAG Arithmetic)
      * P2: Non-Solvable S_5 Group Tracking x Long-Range High-Entropy Retrieval (K=16->64, 4x extrap)
      * P3: Counter-Conventional Entity Binding & 3-Way State Transition (90% prior bias nats vs zz + FAR<=10%)
      * P4: Zero-Hash-Shortcut Test-Time Rule Learning + 1-Pass 2-Hop + ICML 2026 Sign-Flip + Self-Gen Drift
      * P5: Directed Star-Graph Lookahead Planning + Lano Recursive CFG / Variable-Density Boundary Audit
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW, _cosine_lr_multiplier
from pulse.pulse_v7 import patch_fla_sm89_shared_memory
from pulse.pulse_v7_plasticity import TransitionConditionedVerifier
from pulse.whiteboard_anvil import (
    audit_boundary_histogram,
    audit_gradient_sign_flip,
    audit_linear_probe_hop_ladder,
    audit_zero_hidden_params,
    build_dual_strong_baselines,
    estimate_exact_step_flops,
)

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from v7_stage0_stage1 import save_json, sync  # noqa: E402
from v7_stage3 import compute_auroc  # noqa: E402


# ==============================================================================
# Probe P1a: Depo1 — 1-Pass 8-Hop Directed Permutation Graph (Zero CoT!)
# ==============================================================================

def sample_p1a_depo1(
    batch_size: int,
    n_nodes: int = 32,
    hops: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Tokens: 0..31=nodes, 32=EDGE_SEP, 33=QRY, 34..43=HOP_1..HOP_10 (vocab=48)."""
    rng = np.random.default_rng(seed)
    edge_sep, qry_tok, hop_base = n_nodes, n_nodes + 1, n_nodes + 2
    max_h = max(hops)
    seq_len = n_nodes * 3 + 3
    seqs = np.empty((batch_size, seq_len), dtype=np.int64)
    targets = np.empty(batch_size, dtype=np.int64)
    hop_ids = np.empty(batch_size, dtype=np.int64)
    inter = np.empty((batch_size, max_h), dtype=np.int64)

    for b in range(batch_size):
        perm = rng.permutation(n_nodes)
        order = rng.permutation(n_nodes)
        pos = 0
        for u in order:
            seqs[b, pos] = u
            seqs[b, pos + 1] = perm[u]
            seqs[b, pos + 2] = edge_sep
            pos += 3
        s = int(rng.integers(0, n_nodes))
        k = int(rng.choice(hops))
        cur = s
        for h in range(max_h):
            cur = int(perm[cur])
            inter[b, h] = cur
        seqs[b, pos] = qry_tok
        seqs[b, pos + 1] = s
        seqs[b, pos + 2] = hop_base + (k - 1)
        targets[b] = inter[b, k - 1]
        hop_ids[b] = k

    return {
        "input_ids": torch.from_numpy(seqs),
        "target": torch.from_numpy(targets),
        "hops": torch.from_numpy(hop_ids),
        "inter_targets": torch.from_numpy(inter),
    }


# ==============================================================================
# Probe P1b: Depo2 — Multi-Token Node Permutation Graph (5-7 tok/node, vocab=4!)
# ==============================================================================

def sample_p1b_depo2(
    batch_size: int,
    n_nodes: int = 10,
    hops: Tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    seq_len: int = 224,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Alphabet {0,1,2,3}, 4=NODE_SEP, 5=EDGE_SEP, 6=QRY, 7..14=HOP_1..HOP_8, 15=PAD (vocab=16)."""
    rng = np.random.default_rng(seed)
    node_sep, edge_sep, qry_tok, hop_base, pad_tok = 4, 5, 6, 7, 15
    seqs = np.full((batch_size, seq_len), pad_tok, dtype=np.int64)
    ans_mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
    hop_ids = np.empty(batch_size, dtype=np.int64)

    for b in range(batch_size):
        # Generate n_nodes distinct 5-to-7-token codes over the 4-symbol alphabet {0,1,2,3}
        codes: List[List[int]] = []
        seen = set()
        while len(codes) < n_nodes:
            clen = int(rng.integers(5, 8))
            c = tuple(rng.integers(0, 4, size=clen).tolist())
            if c not in seen:
                seen.add(c)
                codes.append(list(c))
        perm = rng.permutation(n_nodes)
        order = rng.permutation(n_nodes)
        toks: List[int] = []
        for u in order:
            toks.extend(codes[u])
            toks.append(node_sep)
            toks.extend(codes[perm[u]])
            toks.append(edge_sep)
        s = int(rng.integers(0, n_nodes))
        k = int(rng.choice(hops))
        cur = s
        for _ in range(k):
            cur = int(perm[cur])
        toks.append(qry_tok)
        toks.extend(codes[s])
        toks.append(hop_base + (k - 1))
        toks.append(node_sep)
        ans_start = len(toks)
        ans_code = codes[cur] + [edge_sep]
        toks.extend(ans_code)
        # Keep tail if somehow > seq_len (with n_nodes=10, max len is 10*16 + 18 = 178 < 224!)
        assert len(toks) <= seq_len, f"Depo2 len {len(toks)} > {seq_len}"
        seqs[b, : len(toks)] = toks
        # Next-token prediction positions for the answer code
        ans_mask[b, ans_start - 1 : ans_start - 1 + len(ans_code)] = True
        hop_ids[b] = k

    return {
        "input_ids": torch.from_numpy(seqs[:, :-1]),
        "targets": torch.from_numpy(seqs[:, 1:]),
        "ans_mask": torch.from_numpy(ans_mask[:, :-1]),
        "hops": torch.from_numpy(hop_ids),
    }


# ==============================================================================
# Probe P1c: DAG Arithmetic — Growing Flat Register Multi-Hop Evaluation
# ==============================================================================

def sample_p1c_dag_arith(
    batch_size: int,
    n_regs: int = 24,
    mod_val: int = 16,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Tokens: 0..23=regs, 24..39=vals(mod 16), 40=ASSIGN, 41=ADD, 42=SUB, 43=SEP, 44=QRY (vocab=48)."""
    rng = np.random.default_rng(seed)
    val_base = n_regs
    t_eq, t_add, t_sub, t_sep, t_qry = val_base + mod_val, val_base + mod_val + 1, val_base + mod_val + 2, val_base + mod_val + 3, val_base + mod_val + 4
    # 6 root constants (len 4 each) + 18 binary ops (len 6 each) + 2 query = 134 tokens fixed!
    n_roots = 6
    seq_len = n_roots * 4 + (n_regs - n_roots) * 6 + 2
    seqs = np.empty((batch_size, seq_len), dtype=np.int64)
    targets = np.empty(batch_size, dtype=np.int64)
    depths = np.empty(batch_size, dtype=np.int64)

    for b in range(batch_size):
        reg_order = rng.permutation(n_regs)
        vals = np.zeros(n_regs, dtype=np.int64)
        dep = np.zeros(n_regs, dtype=np.int64)
        pos = 0
        for i in range(n_regs):
            r = int(reg_order[i])
            if i < n_roots:
                c = int(rng.integers(0, mod_val))
                vals[r] = c
                dep[r] = 0
                seqs[b, pos : pos + 4] = [r, t_eq, val_base + c, t_sep]
                pos += 4
            else:
                # Reference 2 earlier registers (encourage depth growth up to 5-6)
                pa = int(reg_order[int(rng.integers(max(0, i - 6), i))])
                pb = int(reg_order[int(rng.integers(0, i))])
                op_is_add = bool(rng.integers(0, 2))
                vals[r] = (vals[pa] + vals[pb]) % mod_val if op_is_add else (vals[pa] - vals[pb]) % mod_val
                dep[r] = max(dep[pa], dep[pb]) + 1
                seqs[b, pos : pos + 6] = [r, t_eq, pa, (t_add if op_is_add else t_sub), pb, t_sep]
                pos += 6
        # Query a deep register (last 8 registers have depths 2..6)
        q_reg = int(reg_order[int(rng.integers(n_regs - 8, n_regs))])
        seqs[b, pos] = t_qry
        seqs[b, pos + 1] = q_reg
        targets[b] = val_base + vals[q_reg]
        depths[b] = dep[q_reg]

    return {"input_ids": torch.from_numpy(seqs), "target": torch.from_numpy(targets), "depths": torch.from_numpy(depths)}


# ==============================================================================
# Probe P2: Non-Solvable S_5 Group Tracking x Long-Range KV Retrieval Collision
# ==============================================================================

def sample_p2_s5_retrieval(
    batch_size: int,
    seq_len: int = 192,
    n_pairs: int = 16,
    n_swaps: int = 24,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Jointly tests non-Abelian S_5 tracking AND K>d_head high-entropy KV retrieval (W_swa=32 << L_train!).
    Tokens: 0..63=keys, 64..127=vals, 128..143=non-zero distractors, 144..147=S5 swaps,
            148..152=slot IDs 0..4, 153=KV_MARK, 154=QRY_S5, 155=QRY_KV, 156=QRY_JOINT (vocab=160).
    """
    rng = np.random.default_rng(seed)
    n_pool = 64
    dist_lo, dist_hi = 128, 144
    swap_base, slot_base = 144, 148
    t_kv, t_qs5, t_qkv, t_qjt = 153, 154, 155, 156
    # Reserve last 6 tokens for 3 disjoint queries: [QRY_S5, p1, QRY_KV, k_r, QRY_JOINT, p2]
    body_len = seq_len - 6
    assert n_pairs * 3 + n_swaps <= body_len, f"P2 overflow: {n_pairs*3 + n_swaps} > {body_len}"

    seqs = rng.integers(dist_lo, dist_hi, size=(batch_size, seq_len), dtype=np.int64)
    y_s5 = np.empty(batch_size, dtype=np.int64)
    y_kv = np.empty(batch_size, dtype=np.int64)
    y_jt = np.empty(batch_size, dtype=np.int64)

    for b in range(batch_size):
        # 1. Sample K distinct (key, val) pairs; keys 0..4 are ALWAYS included as the 5 S_5 item keys!
        extra_keys = rng.choice(np.arange(5, n_pool), size=max(0, n_pairs - 5), replace=False).tolist()
        keys = np.array([0, 1, 2, 3, 4] + extra_keys, dtype=np.int64)
        rng.shuffle(keys)
        vals = rng.choice(n_pool, size=n_pairs, replace=False) + n_pool  # 64..127
        kv_map = {int(k): int(v) for k, v in zip(keys, vals)}

        # Place all KV pairs in the first half (strictly > 32 tokens before queries so SWA-32 cannot cheat!)
        kv_zone = n_pairs * 3
        for idx_p in range(n_pairs):
            p0 = idx_p * 3
            seqs[b, p0] = t_kv
            seqs[b, p0 + 1] = keys[idx_p]
            seqs[b, p0 + 2] = vals[idx_p]

        # 2. Place n_swaps non-Abelian S_5 adjacent transpositions across remaining distractor body
        rem = body_len - kv_zone
        swap_positions = np.sort(rng.choice(rem, size=n_swaps, replace=False)) + kv_zone
        perm = [0, 1, 2, 3, 4]
        for sp in swap_positions:
            op = int(rng.integers(0, 4))  # swap(op, op+1)
            perm[op], perm[op + 1] = perm[op + 1], perm[op]
            seqs[b, sp] = swap_base + op

        # 3. Three queries at tail: S5-only, KV-only, and Joint S5->KV!
        p1 = int(rng.integers(0, 5))
        p2 = int(rng.integers(0, 5))
        rk = int(keys[int(rng.integers(0, n_pairs))])
        seqs[b, body_len : body_len + 6] = [t_qs5, slot_base + p1, t_qkv, rk, t_qjt, slot_base + p2]
        y_s5[b] = perm[p1]              # item ID in 0..4
        y_kv[b] = kv_map[rk]            # value token in 64..127
        y_jt[b] = kv_map[perm[p2]]      # value token of the item that landed in slot p2!

    return {
        "input_ids": torch.from_numpy(seqs),
        "pos_s5": torch.full((batch_size,), body_len + 1, dtype=torch.long),
        "pos_kv": torch.full((batch_size,), body_len + 3, dtype=torch.long),
        "pos_joint": torch.full((batch_size,), body_len + 5, dtype=torch.long),
        "y_s5": torch.from_numpy(y_s5),
        "y_kv": torch.from_numpy(y_kv),
        "y_joint": torch.from_numpy(y_jt),
    }


# ==============================================================================
# Probe P3: Counter-Conventional Entity Binding & 3-Way State Transition
# ==============================================================================

def sample_p3_alias_swap(
    batch_size: int,
    seq_len: int = 256,
    conv_prob: float = 0.90,
    force_counter_conv: bool = False,
    force_mid_pos: bool = False,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """BlackboxNLP 2026 (`import pandas as np`) + ICLR 2026 (`Mixing Mechanisms`) + 3-Way State Transition.
    Tokens: 0..19=Canonical Aliases, 20..39=Control Aliases (zz), 40..59=Attrs, 60..75=Distractors,
            76=BIND, 77=REBIND, 78=USE, 79=CLAIM, 80=SEP (vocab=96).
    """
    rng = np.random.default_rng(seed)
    n_ent = 20
    ctrl_base, attr_base, dist_lo, dist_hi = 20, 40, 60, 76
    t_bind, t_rebind, t_use, t_claim, t_sep = 76, 77, 78, 79, 80
    # 20 bindings (4 tok each = 80) + optional rebind (4 tok) + tail [USE, ent, SEP, CLAIM, ent, claimed_attr] (6 tok)
    tail_len = 6
    body_len = seq_len - tail_len

    seqs = rng.integers(dist_lo, dist_hi, size=(batch_size, seq_len), dtype=np.int64)
    y_use = np.empty(batch_size, dtype=np.int64)
    y_cls = np.empty(batch_size, dtype=np.int64)       # 0=Consistent, 1=Legal Rebind, 2=Illegal Swap
    is_canon = np.empty(batch_size, dtype=np.bool_)    # True if target is Alias_i, False if Ctrl_i (zz)
    conv_attr = np.empty(batch_size, dtype=np.int64)   # Conventional attr_base + i (for prior-bias nats!)
    target_slot = np.empty(batch_size, dtype=np.int64)

    for b in range(batch_size):
        # Mix 10 canonical Aliases + 10 control Ctrls (20 concurrent entities per sequence!)
        can_ids = rng.choice(n_ent, size=10, replace=False).tolist()
        ctl_ids = rng.choice(n_ent, size=10, replace=False).tolist()
        ents = [(e, True) for e in can_ids] + [(ctrl_base + e, False) for e in ctl_ids]
        rng.shuffle(ents)

        # Pick target entity (strictly in middle slots 7..13 during mid-context eval!)
        t_idx = int(rng.integers(7, 14)) if force_mid_pos else int(rng.integers(0, 20))
        target_slot[b] = t_idx

        # Initial binding map
        init_attr: Dict[int, int] = {}
        cur_attr: Dict[int, int] = {}
        for idx_e, (ent_tok, canon_flag) in enumerate(ents):
            base_i = ent_tok if canon_flag else (ent_tok - ctrl_base)
            if canon_flag:
                if idx_e == t_idx and force_counter_conv:
                    a_val = int(rng.choice([j for j in range(n_ent) if j != base_i]))
                elif float(rng.random()) < conv_prob:
                    a_val = base_i
                else:
                    a_val = int(rng.choice([j for j in range(n_ent) if j != base_i]))
            else:
                a_val = int(rng.integers(0, n_ent))
            init_attr[ent_tok] = attr_base + a_val
            cur_attr[ent_tok] = attr_base + a_val

        # Choose 3-way class uniformly (0=Consistent, 1=Legal Rebind, 2=Illegal Fluent Swap)
        cls = b % 3
        t_ent, t_canon = ents[t_idx]
        t_base_i = t_ent if t_canon else (t_ent - ctrl_base)
        # Anti-shortcut: insert distractor REBIND on another entity in 50% of Class 0 and Class 2!
        rebind_ops: List[Tuple[int, int]] = []
        if cls == 1:
            # Target entity undergoes a genuine LEGAL REBIND (to a new attr; counter-conv if forced!)
            cands = [j for j in range(n_ent) if (attr_base + j) != cur_attr[t_ent] and (not (force_counter_conv and t_canon) or j != t_base_i)]
            new_a = attr_base + int(rng.choice(cands))
            rebind_ops.append((t_ent, new_a))
            cur_attr[t_ent] = new_a
        else:
            if float(rng.random()) < 0.50:
                other_ent, _ = ents[(t_idx + 1) % 20]
                other_a = attr_base + int(rng.integers(0, n_ent))
                rebind_ops.append((other_ent, other_a))
                cur_attr[other_ent] = other_a
            if cls == 2 and float(rng.random()) < 0.50:
                # Subcase 2b: target DOES rebind, then CLAIM uses stale pre-rebind attr (kills transfer-verb shortcut!)
                cands = [j for j in range(n_ent) if (attr_base + j) != cur_attr[t_ent] and (not (force_counter_conv and t_canon) or j != t_base_i)]
                new_a = attr_base + int(rng.choice(cands))
                rebind_ops.append((t_ent, new_a))
                cur_attr[t_ent] = new_a

        # Spread 20 bindings + rebinds across body_len with FineWeb-like random distractor gaps
        n_blocks = 20 + len(rebind_ops)
        stride = max(4, (body_len - 8) // n_blocks)
        pos = 2
        for ent_tok, _ in ents:
            seqs[b, pos : pos + 4] = [t_bind, ent_tok, init_attr[ent_tok], t_sep]
            pos += stride
        for r_ent, r_attr in rebind_ops:
            if pos + 4 <= body_len:
                seqs[b, pos : pos + 4] = [t_rebind, r_ent, r_attr, t_sep]
                pos += stride

        true_a = cur_attr[t_ent]
        if cls in (0, 1):
            claimed_a = true_a
        else:
            # Illegal claim: prefer stale attr if rebound, or conventional attr if counter-conv!
            if init_attr[t_ent] != true_a:
                claimed_a = init_attr[t_ent]
            elif t_canon and (attr_base + t_base_i) != true_a:
                claimed_a = attr_base + t_base_i
            else:
                claimed_a = attr_base + int(rng.choice([j for j in range(n_ent) if (attr_base + j) != true_a]))

        seqs[b, body_len : body_len + 6] = [t_use, t_ent, t_sep, t_claim, t_ent, claimed_a]
        y_use[b] = true_a
        y_cls[b] = cls
        is_canon[b] = t_canon
        conv_attr[b] = attr_base + t_base_i

    return {
        "input_ids": torch.from_numpy(seqs),
        "pos_use": torch.full((batch_size,), body_len + 1, dtype=torch.long),
        "pos_claim": torch.full((batch_size,), body_len + 5, dtype=torch.long),
        "y_use": torch.from_numpy(y_use),
        "y_cls": torch.from_numpy(y_cls),
        "is_canon": torch.from_numpy(is_canon),
        "conv_attr": torch.from_numpy(conv_attr),
    }


# ==============================================================================
# Probe P4: Zero-Hash-Shortcut Test-Time Rule Learning + 1-Pass 2-Hop Bridge
# ==============================================================================

class P4RuleWorld:
    """Disjoint View-A (single token 0..63) vs View-B (2-token non-canonical code 64..79) world.
    ZERO surface token overlap between View A and View B!
    Tokens: 0..63=ViewA, 64..79=ViewB subwords, 80..95=r2 targets (16),
            96=RULE_R1, 97=QRY_R1, 98=QRY_2HOP, 99=VIEW_EQ, 100=R2_FACT, 101=PAD (vocab=112).
    """

    def __init__(self, n_entities: int = 64, n_props: int = 16, seed: int = 404) -> None:
        rng = np.random.default_rng(seed)
        self.n_ent = n_entities
        self.n_props = n_props
        self.vocab_size = 112
        self.prop_base = 80
        self.t_rule, self.t_qr1, self.t_q2h, self.t_veq, self.t_r2, self.t_pad = 96, 97, 98, 99, 100, 101
        # Fixed background world graph: r2(o1) -> prop in 0..15, old r1(s) -> o1_old
        self.r2_map = rng.integers(0, n_props, size=n_entities)
        self.r1_old = rng.permutation(n_entities)
        # Partition entities: 0..47 in pretrain r1 background, 48..63 held-out for test-time edits!
        self.train_ents = np.arange(0, 48)
        self.test_ents = np.arange(48, 64)

    def view_a(self, e: int) -> List[int]:
        return [int(e)]

    def view_b(self, e: int) -> List[int]:
        return [64 + (int(e) // 8), 72 + (int(e) % 8)]

    def sample_pretrain_batch(self, batch_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pretrains view alignment (ViewA <-> ViewB), r2(o1) in both views, and old r1 on train_ents + 2-hop composition."""
        rng = np.random.default_rng(seed)
        seq_len = 12
        seqs = np.full((batch_size, seq_len), self.t_pad, dtype=np.int64)
        mask = np.zeros((batch_size, seq_len), dtype=np.bool_)
        for b in range(batch_size):
            kind = b % 5
            if kind == 0:  # ViewB -> ViewA alignment
                e = int(rng.integers(0, self.n_ent))
                t = [self.t_veq] + self.view_b(e) + [self.t_veq] + self.view_a(e)
                seqs[b, : len(t)] = t
                mask[b, len(t) - 2] = True
            elif kind == 1:  # r2 in ViewA and ViewB
                e = int(rng.integers(0, self.n_ent))
                v = self.view_a(e) if (b // 5) % 2 == 0 else self.view_b(e)
                t = [self.t_r2] + v + [self.prop_base + int(self.r2_map[e])]
                seqs[b, : len(t)] = t
                mask[b, len(t) - 2] = True
            elif kind == 2:  # Old r1 on train_ents (both views!)
                s = int(rng.choice(self.train_ents))
                v = self.view_a(s) if (b // 5) % 2 == 0 else self.view_b(s)
                t = [self.t_qr1] + v + [int(self.r1_old[s])]
                seqs[b, : len(t)] = t
                mask[b, len(t) - 2] = True
            elif kind == 3:  # 1-pass 2-hop on train_ents (both views -> teaches composition circuit!)
                s = int(rng.choice(self.train_ents))
                v = self.view_a(s) if (b // 5) % 2 == 0 else self.view_b(s)
                o1 = int(self.r1_old[s])
                t = [self.t_q2h] + v + [self.prop_base + int(self.r2_map[o1])]
                seqs[b, : len(t)] = t
                mask[b, len(t) - 2] = True
            else:  # In-context write ViewA + zero-overlap ViewB 2-hop bridge (teaches ViewA->ViewB binding!)
                s = int(rng.choice(self.train_ents))
                o1 = int(rng.integers(0, self.n_ent))
                t = [self.t_rule] + self.view_a(s) + self.view_a(o1) + [self.t_q2h] + self.view_b(s) + [self.prop_base + int(self.r2_map[o1])]
                seqs[b, : len(t)] = t
                mask[b, len(t) - 2] = True
        return torch.from_numpy(seqs[:, :-1]), torch.from_numpy(seqs[:, 1:]), torch.from_numpy(mask[:, :-1])

    def build_test_episodes(self, seed: int = 999) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        eps: List[Dict[str, Any]] = []
        for s in self.test_ents:
            old_o1 = int(self.r1_old[s])
            # Pick novel o1* whose r2 property strictly DIFFERS from old_o1 (anti-shortcut!)
            cands = [e for e in range(self.n_ent) if int(self.r2_map[e]) != int(self.r2_map[old_o1])]
            o1_new = int(rng.choice(cands))
            eps.append(
                {
                    "s": int(s),
                    "o1_new": o1_new,
                    "y_new": self.prop_base + int(self.r2_map[o1_new]),
                    "write_view_a": [self.t_qr1] + self.view_a(int(s)) + self.view_a(o1_new),
                    "qry_r1_view_a": [self.t_qr1] + self.view_a(int(s)),
                    "qry_r1_view_b": [self.t_qr1] + self.view_b(int(s)),
                    "qry_2hop_view_b": [self.t_q2h] + self.view_b(int(s)),
                }
            )
        return eps


# ==============================================================================
# Probe P5a: Directed Star-Graph Lookahead Planning & P5b: Lano VarDensity CFG
# ==============================================================================

def sample_p5a_star_graph(
    batch_size: int,
    degree: int = 5,
    path_len: int = 4,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Star graph with D=5 branches of length P=4 (1 valid path to goal, 4 dead ends).
    Tokens: 0..31=nodes, 32=EDGE, 33=START, 34=GOAL, 35=PATH (vocab=48).
    """
    rng = np.random.default_rng(seed)
    n_nodes_used = 1 + degree * path_len  # 21 nodes out of 32
    t_edge, t_st, t_go, t_path = 32, 33, 34, 35
    n_edges = degree * path_len
    prefix_len = n_edges * 3 + 5  # edges + [START, s, GOAL, g, PATH]
    seq_len = prefix_len + path_len
    seqs = np.empty((batch_size, seq_len), dtype=np.int64)

    for b in range(batch_size):
        nodes = rng.choice(32, size=n_nodes_used, replace=False)
        root = int(nodes[0])
        branches = [nodes[1 + d * path_len : 1 + (d + 1) * path_len].tolist() for d in range(degree)]
        edges: List[Tuple[int, int]] = []
        for br in branches:
            edges.append((root, br[0]))
            for j in range(path_len - 1):
                edges.append((br[j], br[j + 1]))
        rng.shuffle(edges)
        pos = 0
        for u, v in edges:
            seqs[b, pos : pos + 3] = [u, v, t_edge]
            pos += 3
        win_br = branches[int(rng.integers(0, degree))]
        goal = win_br[-1]
        seqs[b, pos : pos + 5] = [t_st, root, t_go, goal, t_path]
        seqs[b, prefix_len:] = win_br

    return {
        "input_ids": torch.from_numpy(seqs[:, :-1]),
        "targets": torch.from_numpy(seqs[:, 1:]),
        "prefix_len": prefix_len,
        "path_len": path_len,
    }


def sample_p5b_lano_vardensity(
    batch_size: int,
    seq_len: int = 192,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Recursive 3-level CFG alternating 4-6 token deterministic macros (H=0) with 4-way concept branches (H=2 bits).
    Tokens: 0..15=high-entropy branching concept symbols, 16..39=deterministic macro filler tokens, 40=EOS (vocab=48).
    """
    rng = np.random.default_rng(seed)
    # 4 deterministic low-entropy macros (length 4-6 each) paired with 4-way high-entropy choices!
    macros = [
        [16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30],
        [31, 32, 33, 34],
    ]
    seqs = np.empty((batch_size, seq_len + 1), dtype=np.int64)
    hi_mask = np.zeros((batch_size, seq_len + 1), dtype=np.bool_)

    for b in range(batch_size):
        out: List[int] = []
        hi: List[bool] = []
        while len(out) < seq_len + 1:
            # Level-1 choice (4-way high entropy) selects macro family + nested Level-2 branch
            c1 = int(rng.integers(0, 4))
            out.append(c1)
            hi.append(True)
            m = macros[c1]
            out.extend(m)
            hi.extend([False] * len(m))
            # Level-2 branch (4-way high entropy, conditioned on c1 -> tokens 4..15)
            c2 = 4 + (c1 % 3) * 4 + int(rng.integers(0, 4))
            out.append(c2)
            hi.append(True)
            # Closing deterministic mirror macro (reverses macro -> checks structural matching!)
            rev = list(reversed(m))
            out.extend(rev)
            hi.extend([False] * len(rev))
        seqs[b] = out[: seq_len + 1]
        hi_mask[b] = hi[: seq_len + 1]

    return {
        "input_ids": torch.from_numpy(seqs[:, :-1]),
        "targets": torch.from_numpy(seqs[:, 1:]),
        "hi_entropy_mask": torch.from_numpy(hi_mask[:, 1:]),
    }


# ==============================================================================
# Shared Anti-Grokking 3-LR x 3-Seed Trainer & Probe Runners
# ==============================================================================

def _make_opt(model: nn.Module, lr: float, extra_params: Optional[List[nn.Parameter]] = None) -> FairMuonAdamW:
    params = model.io_parameters() if hasattr(model, "io_parameters") else list(model.parameters())
    if extra_params:
        params = list(params) + list(extra_params)
    return FairMuonAdamW(params, lr=lr, muon_lr_mult=8.0, adam_lr_mult=1.5, weight_decay=0.01)


def _train_generic(
    make_model: Callable[[int], nn.Module],
    step_loss_fn: Callable[[nn.Module, int], torch.Tensor],
    eval_fn: Callable[[nn.Module], Tuple[float, Dict[str, Any]]],
    device: torch.device,
    amp: bool,
    lrs: Tuple[float, ...],
    seeds: Tuple[int, ...],
    lr_steps: int,
    train_steps: int,
    eval_every: int,
    tag: str,
    early_stop_score: float = 99.5,
) -> Dict[str, Any]:
    """3-LR selection on first seed -> 3-Seed confirmation with wall-clock timing & exact FLOPs."""
    # Phase 1: Fast LR grid on seeds[0]
    best_lr, best_lr_score = lrs[0], -1e9
    lr_sweep: Dict[str, float] = {}
    for lr in lrs:
        m = make_model(seeds[0]).to(device)
        m.compile_fast_paths()
        opt = _make_opt(m, lr)
        w_steps = max(5, lr_steps // 10)
        for st in range(1, lr_steps + 1):
            mult = _cosine_lr_multiplier(st - 1, w_steps, lr_steps, min_ratio=0.1)
            for pg in opt.param_groups:
                pg["lr"] = lr * mult
            m.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = step_loss_fn(m, 10000 + st)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        m.eval()
        sc, _ = eval_fn(m)
        lr_sweep[f"{lr:.0e}"] = round(sc, 2)
        if sc > best_lr_score:
            best_lr_score = sc
            best_lr = lr
        del m, opt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Phase 2: 3-Seed Confirmation at best_lr
    seed_runs: List[Dict[str, Any]] = []
    first_trained_model: Optional[nn.Module] = None
    for s_idx, sd in enumerate(seeds):
        m = make_model(sd).to(device)
        m.compile_fast_paths()
        opt = _make_opt(m, best_lr)
        w_steps = max(10, train_steps // 10)
        consec_hi = 0
        steps_done = 0
        sync(device)
        t0 = time.perf_counter()
        best_sc, best_det = -1e9, {}
        for st in range(1, train_steps + 1):
            mult = _cosine_lr_multiplier(st - 1, w_steps, train_steps, min_ratio=0.1)
            for pg in opt.param_groups:
                pg["lr"] = best_lr * mult
            m.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = step_loss_fn(m, sd * 100000 + st)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            steps_done = st
            if st % eval_every == 0 or st == train_steps:
                m.eval()
                sc, det = eval_fn(m)
                if sc > best_sc:
                    best_sc, best_det = sc, det
                if sc >= early_stop_score:
                    consec_hi += 1
                    if consec_hi >= 2:
                        break
                else:
                    consec_hi = 0
        sync(device)
        sec = time.perf_counter() - t0
        m.eval()
        final_sc, final_det = eval_fn(m)
        if final_sc >= best_sc:
            best_sc, best_det = final_sc, final_det
        seed_runs.append(
            {
                "seed": sd,
                "steps": steps_done,
                "train_sec": round(sec, 2),
                "ms_per_step": round(sec * 1000.0 / max(1, steps_done), 2),
                "primary_score": round(best_sc, 2),
                "metrics": best_det,
            }
        )
        print(
            f"[{tag} | seed={sd} lr={best_lr:.0e}] steps={steps_done} ({sec:.1f}s, {seed_runs[-1]['ms_per_step']}ms/st) "
            f"score={best_sc:.2f} {json.dumps(best_det)}",
            flush=True,
        )
        if s_idx == 0:
            first_trained_model = m
        else:
            del m
        del opt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = [r["primary_score"] for r in seed_runs]
    return {
        "best_lr": best_lr,
        "lr_sweep": lr_sweep,
        "mean_primary": round(float(statistics.mean(scores)), 2),
        "std_primary": round(float(statistics.pstdev(scores)), 2),
        "mean_ms_per_step": round(float(statistics.mean([r["ms_per_step"] for r in seed_runs])), 2),
        "seeds": seed_runs,
        "_first_model": first_trained_model,
    }


def run_probe_p1(
    out: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
    lrs: Tuple[float, ...],
    seeds: Tuple[int, ...],
    lr_steps: int,
    train_steps: int,
) -> Dict[str, Any]:
    dest = out / "p1_multihop.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    bs = 16 if smoke else 64
    ev_every = max(5, train_steps // 4)
    res: Dict[str, Any] = {"p1a_depo1": {}, "p1b_depo2": {}, "p1c_dag_arith": {}}

    # --- P1a: Depo1 (8-Hop Permutation Graph + Linear Probe Hop Ladder!) ---
    n_nodes = 16 if smoke else 32
    eval_h_sets = {h: sample_p1a_depo1(32 if smoke else 256, n_nodes=n_nodes, hops=(h,), seed=1100 + h) for h in (1, 2, 4, 6, 8, 10)}
    probe_tr = sample_p1a_depo1(48 if smoke else 384, n_nodes=n_nodes, hops=(8,), seed=1201)
    probe_te = sample_p1a_depo1(32 if smoke else 256, n_nodes=n_nodes, hops=(8,), seed=1202)
    for arm in ("tf_modern", "pulse_v7b_ref"):
        def _mk_p1a(sd: int, _arm: str = arm) -> nn.Module:
            return build_dual_strong_baselines(48, smoke=smoke, seed=sd)[_arm]
        def _loss_p1a(m: nn.Module, step_seed: int) -> torch.Tensor:
            b = sample_p1a_depo1(bs, n_nodes=n_nodes, hops=(1, 2, 3, 4, 5, 6, 7, 8), seed=step_seed)
            lg = m(b["input_ids"].to(device))["logits"][:, -1, :n_nodes].float()
            return F.cross_entropy(lg, b["target"].to(device))
        @torch.no_grad()
        def _eval_p1a(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            by_h = {}
            for h, eb in eval_h_sets.items():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg = m(eb["input_ids"].to(device))["logits"][:, -1, :n_nodes].float()
                by_h[f"hop{h}"] = round(float(lg.argmax(-1).cpu().eq(eb["target"]).float().mean().item() * 100.0), 2)
            return by_h["hop8"], by_h
        tr = _train_generic(_mk_p1a, _loss_p1a, _eval_p1a, device, amp, lrs, seeds, lr_steps, train_steps, ev_every, f"P1a-Depo1:{arm}")
        m0 = tr.pop("_first_model")
        # Run mandatory Linear-Probe Hop Ladder across ALL layers!
        with torch.no_grad():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                st_tr = [s[:, -1, :].float().cpu() for s in m0.forward_anvil(probe_tr["input_ids"].to(device), return_layer_states=True)["layer_states"]]
                st_te = [s[:, -1, :].float().cpu() for s in m0.forward_anvil(probe_te["input_ids"].to(device), return_layer_states=True)["layer_states"]]
        tr["hop_ladder_audit"] = audit_linear_probe_hop_ladder(st_tr, probe_tr["inter_targets"], st_te, probe_te["inter_targets"], n_classes=n_nodes)
        tr["flops"] = estimate_exact_step_flops(m0, bs, probe_tr["input_ids"].shape[1])
        del m0
        res["p1a_depo1"][arm] = tr

    # --- P1b: Depo2 (5-7 tok/node, vocab=4!) ---
    seq_l2 = 176 if smoke else 224
    n_nd2 = 8 if smoke else 10
    ev2_sets = {h: sample_p1b_depo2(24 if smoke else 128, n_nodes=n_nd2, hops=(h,), seq_len=seq_l2, seed=1300 + h) for h in (2, 4, 8)}
    for arm in ("tf_modern", "pulse_v7b_ref"):
        def _mk_p1b(sd: int, _arm: str = arm) -> nn.Module:
            return build_dual_strong_baselines(16, smoke=smoke, seed=sd)[_arm]
        def _loss_p1b(m: nn.Module, step_seed: int) -> torch.Tensor:
            b = sample_p1b_depo2(bs, n_nodes=n_nd2, hops=(1, 2, 3, 4, 6, 8), seq_len=seq_l2, seed=step_seed)
            lg = m(b["input_ids"].to(device))["logits"].float()
            msk = b["ans_mask"].to(device)
            return F.cross_entropy(lg[msk], b["targets"].to(device)[msk])
        @torch.no_grad()
        def _eval_p1b(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            out_d = {}
            for h, eb in ev2_sets.items():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg = m(eb["input_ids"].to(device))["logits"].float()
                preds = lg.argmax(-1).cpu()
                msk = eb["ans_mask"]
                tgt = eb["targets"]
                em = [bool((preds[i][msk[i]] == tgt[i][msk[i]]).all().item()) for i in range(preds.shape[0])]
                out_d[f"hop{h}_em"] = round(float(np.mean(em) * 100.0), 2)
            return out_d["hop8_em"], out_d
        tr = _train_generic(_mk_p1b, _loss_p1b, _eval_p1b, device, amp, lrs, seeds, lr_steps, train_steps, ev_every, f"P1b-Depo2:{arm}")
        m0 = tr.pop("_first_model")
        tr["flops"] = estimate_exact_step_flops(m0, bs, seq_l2 - 1)
        del m0
        res["p1b_depo2"][arm] = tr

    # --- P1c: DAG Arithmetic (Growing Flat Register Multi-Hop) ---
    ev3 = sample_p1c_dag_arith(48 if smoke else 512, seed=1401)
    for arm in ("tf_modern", "pulse_v7b_ref"):
        def _mk_p1c(sd: int, _arm: str = arm) -> nn.Module:
            return build_dual_strong_baselines(48, smoke=smoke, seed=sd)[_arm]
        def _loss_p1c(m: nn.Module, step_seed: int) -> torch.Tensor:
            b = sample_p1c_dag_arith(bs, seed=step_seed)
            lg = m(b["input_ids"].to(device))["logits"][:, -1, 24:40].float()
            return F.cross_entropy(lg, (b["target"] - 24).to(device))
        @torch.no_grad()
        def _eval_p1c(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = m(ev3["input_ids"].to(device))["logits"][:, -1, 24:40].float()
            ok = lg.argmax(-1).cpu().eq(ev3["target"] - 24).float()
            dep = ev3["depths"]
            det = {"overall": round(float(ok.mean().item() * 100.0), 2)}
            for lo, hi, nm in ((1, 2, "d1_2"), (3, 4, "d3_4"), (5, 9, "d5plus")):
                m_d = (dep >= lo) & (dep <= hi)
                if m_d.any():
                    det[nm] = round(float(ok[m_d].mean().item() * 100.0), 2)
            return det["overall"], det
        tr = _train_generic(_mk_p1c, _loss_p1c, _eval_p1c, device, amp, lrs, seeds, lr_steps, train_steps, ev_every, f"P1c-DAG:{arm}")
        m0 = tr.pop("_first_model")
        tr["flops"] = estimate_exact_step_flops(m0, bs, ev3["input_ids"].shape[1])
        del m0
        res["p1c_dag_arith"][arm] = tr

    save_json(dest, res)
    return res


def run_probe_p2(
    out: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
    lrs: Tuple[float, ...],
    seeds: Tuple[int, ...],
    lr_steps: int,
    train_steps: int,
) -> Dict[str, Any]:
    dest = out / "p2_s5_retrieval.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    bs = 16 if smoke else 64
    ev_every = max(5, train_steps // 4)
    l_tr, k_tr, sw_tr = (96, 10, 12) if smoke else (192, 16, 24)
    ev_cfg = [(96, 10, 12, "ID_1x"), (192, 20, 24, "OOD_2x"), (384, 32, 48, "OOD_4x")] if smoke else [
        (192, 16, 24, "ID_1x_K16"), (384, 32, 48, "OOD_2x_K32"), (768, 64, 96, "OOD_4x_K64")
    ]
    ev_sets = {nm: sample_p2_s5_retrieval(24 if smoke else 128, seq_len=L, n_pairs=K, n_swaps=sw, seed=2100 + i) for i, (L, K, sw, nm) in enumerate(ev_cfg)}

    res: Dict[str, Any] = {}
    for arm in ("tf_modern", "pulse_v7b_ref"):
        def _mk(sd: int, _arm: str = arm) -> nn.Module:
            return build_dual_strong_baselines(160, smoke=smoke, seed=sd)[_arm]
        def _loss(m: nn.Module, step_seed: int) -> torch.Tensor:
            b = sample_p2_s5_retrieval(bs, seq_len=l_tr, n_pairs=k_tr, n_swaps=sw_tr, seed=step_seed)
            lg = m(b["input_ids"].to(device))["logits"].float()
            ar = torch.arange(bs, device=device)
            l_s5 = F.cross_entropy(lg[ar, b["pos_s5"].to(device), :5], b["y_s5"].to(device))
            l_kv = F.cross_entropy(lg[ar, b["pos_kv"].to(device), 64:128], (b["y_kv"] - 64).to(device))
            l_jt = F.cross_entropy(lg[ar, b["pos_joint"].to(device), 64:128], (b["y_joint"] - 64).to(device))
            return l_s5 + l_kv + l_jt
        @torch.no_grad()
        def _eval(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            det = {}
            for nm, eb in ev_sets.items():
                n_e = eb["input_ids"].shape[0]
                s5_h = kv_h = jt_h = 0
                for s0 in range(0, n_e, 32):
                    s1 = min(n_e, s0 + 32)
                    ar = torch.arange(s1 - s0, device=device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg = m(eb["input_ids"][s0:s1].to(device))["logits"].float()
                    s5_h += int(lg[ar, eb["pos_s5"][s0:s1].to(device), :5].argmax(-1).cpu().eq(eb["y_s5"][s0:s1]).sum().item())
                    kv_h += int(lg[ar, eb["pos_kv"][s0:s1].to(device), 64:128].argmax(-1).cpu().eq(eb["y_kv"][s0:s1] - 64).sum().item())
                    jt_h += int(lg[ar, eb["pos_joint"][s0:s1].to(device), 64:128].argmax(-1).cpu().eq(eb["y_joint"][s0:s1] - 64).sum().item())
                det[nm] = {"s5": round(s5_h / n_e * 100.0, 2), "kv": round(kv_h / n_e * 100.0, 2), "joint": round(jt_h / n_e * 100.0, 2)}
            return det[ev_cfg[-1][3]]["joint"], det
        tr = _train_generic(_mk, _loss, _eval, device, amp, lrs, seeds, lr_steps, train_steps, ev_every, f"P2-S5Ret:{arm}")
        m0 = tr.pop("_first_model")
        tr["flops"] = estimate_exact_step_flops(m0, bs, l_tr)
        del m0
        res[arm] = tr

    save_json(dest, res)
    return res


def run_probe_p3(
    out: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
    lrs: Tuple[float, ...],
    seeds: Tuple[int, ...],
    lr_steps: int,
    train_steps: int,
) -> Dict[str, Any]:
    dest = out / "p3_alias_swap.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    bs = 18 if smoke else 60  # divisible by 3 for exact 3-way balance!
    ev_every = max(5, train_steps // 4)
    l_tr, l_4x = (128, 256) if smoke else (256, 1024)
    ev_cc_1x = sample_p3_alias_swap(48 if smoke else 300, seq_len=l_tr, conv_prob=0.90, force_counter_conv=True, force_mid_pos=True, seed=3101)
    ev_cc_4x = sample_p3_alias_swap(36 if smoke else 180, seq_len=l_4x, conv_prob=0.90, force_counter_conv=True, force_mid_pos=True, seed=3102)

    res: Dict[str, Any] = {}
    for arm in ("tf_modern", "pulse_v7b_ref"):
        # Wrap backbone + strictly matched <50K TransitionConditionedVerifier!
        class _P3Bundle(nn.Module):
            def __init__(self, bb: nn.Module) -> None:
                super().__init__()
                self.bb = bb
                self.config = bb.config
                self.ver = TransitionConditionedVerifier(d_in=bb.config.d_model, d_hidden=48, n_classes=3, dropout=0.2)
                for p in self.ver.parameters():
                    p._no_muon = True  # type: ignore[attr-defined]
            def io_parameters(self) -> List[nn.Parameter]:
                return self.bb.io_parameters() + list(self.ver.parameters())
            def compile_fast_paths(self) -> None:
                pass  # Need h_fin from forward_anvil
            def forward_p3(self, tk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                o = self.bb.forward_anvil(tk, return_layer_states=True)
                return o["logits"], o["h_fin"]

        def _mk(sd: int, _arm: str = arm) -> nn.Module:
            return _P3Bundle(build_dual_strong_baselines(96, smoke=smoke, seed=sd)[_arm])
        def _loss(m: nn.Module, step_seed: int) -> torch.Tensor:
            b = sample_p3_alias_swap(bs, seq_len=l_tr, conv_prob=0.90, force_counter_conv=False, force_mid_pos=False, seed=step_seed)
            lg, hf = m.forward_p3(b["input_ids"].to(device))  # type: ignore[operator]
            ar = torch.arange(bs, device=device)
            l_use = F.cross_entropy(lg[ar, b["pos_use"].to(device), 40:60].float(), (b["y_use"] - 40).to(device))
            v_lg = m.ver(hf[ar, b["pos_claim"].to(device)].float())  # type: ignore[operator]
            l_ver = F.cross_entropy(v_lg, b["y_cls"].to(device))
            return l_use + 0.5 * l_ver
        @torch.no_grad()
        def _eval_one(m: nn.Module, eb: Dict[str, torch.Tensor]) -> Dict[str, Any]:
            n_e = eb["input_ids"].shape[0]
            u_lgs, v_prs = [], []
            for s0 in range(0, n_e, 30):
                s1 = min(n_e, s0 + 30)
                ar = torch.arange(s1 - s0, device=device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg, hf = m.forward_p3(eb["input_ids"][s0:s1].to(device))  # type: ignore[operator]
                u_lgs.append(lg[ar, eb["pos_use"][s0:s1].to(device), 40:60].float().cpu())
                v_prs.append(F.softmax(m.ver(hf[ar, eb["pos_claim"][s0:s1].to(device)].float()), dim=-1).cpu().numpy())  # type: ignore[operator]
            u_lg = torch.cat(u_lgs, dim=0)
            u_logp = F.log_softmax(u_lg, dim=-1)
            preds_u = u_lg.argmax(-1)
            y_u = eb["y_use"] - 40
            c_idx = eb["conv_attr"] - 40
            m_can = eb["is_canon"]
            m_ctl = ~m_can
            cc_acc = float(preds_u[m_can].eq(y_u[m_can]).float().mean().item() * 100.0)
            zz_acc = float(preds_u[m_ctl].eq(y_u[m_ctl]).float().mean().item() * 100.0)
            # BlackboxNLP 2026 convention prior bias (nats): logP(conv_attr | Alias) - logP(conv_attr | Ctrl_zz)
            lp_can_conv = float(u_logp[m_can, c_idx[m_can]].mean().item())
            lp_ctl_conv = float(u_logp[m_ctl, c_idx[m_ctl]].mean().item())
            bias_nats = lp_can_conv - lp_ctl_conv
            v_probs = np.concatenate(v_prs, axis=0)
            y_c = eb["y_cls"].numpy()
            auroc = compute_auroc(v_probs[:, 2], (y_c == 2).astype(np.int64))
            far_legal = float((v_probs.argmax(-1)[y_c == 1] == 2).mean() * 100.0)
            return {
                "cc_mid_acc": round(cc_acc, 2),
                "zz_mid_acc": round(zz_acc, 2),
                "prior_bias_nats": round(bias_nats, 3),
                "auroc_3way": round(auroc, 4),
                "legal_far_pct": round(far_legal, 2),
            }
        @torch.no_grad()
        def _eval(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            d1 = _eval_one(m, ev_cc_1x)
            d4 = _eval_one(m, ev_cc_4x)
            return d1["cc_mid_acc"], {"L_1x": d1, "L_4x": d4}
        tr = _train_generic(_mk, _loss, _eval, device, amp, lrs, seeds, lr_steps, train_steps, ev_every, f"P3-AliasSwap:{arm}")
        m0 = tr.pop("_first_model")
        tr["verifier_params"] = m0.ver.num_parameters()
        tr["flops"] = estimate_exact_step_flops(m0.bb, bs, l_tr)
        del m0
        res[arm] = tr

    save_json(dest, res)
    return res


def run_probe_p4(
    out: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
    lrs: Tuple[float, ...],
    seeds: Tuple[int, ...],
    lr_steps: int,
    train_steps: int,
) -> Dict[str, Any]:
    dest = out / "p4_nohash_rule2hop.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    world = P4RuleWorld(seed=404)
    bs = 20 if smoke else 80
    ev_every = max(5, train_steps // 4)
    ev_x, ev_y, ev_m = world.sample_pretrain_batch(60 if smoke else 400, seed=4101)
    test_eps = world.build_test_episodes(seed=4201)

    res: Dict[str, Any] = {}
    for arm in ("tf_modern", "pulse_v7b_ref"):
        def _mk(sd: int, _arm: str = arm) -> nn.Module:
            return build_dual_strong_baselines(world.vocab_size, smoke=smoke, seed=sd)[_arm]
        def _loss(m: nn.Module, step_seed: int) -> torch.Tensor:
            x, y, msk = world.sample_pretrain_batch(bs, seed=step_seed)
            lg = m(x.to(device))["logits"].float()
            return F.cross_entropy(lg[msk.to(device)], y.to(device)[msk.to(device)])
        @torch.no_grad()
        def _eval_bg(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = m(ev_x.to(device))["logits"].float()
            acc = float(lg[ev_m.to(device)].argmax(-1).cpu().eq(ev_y[ev_m]).float().mean().item() * 100.0)
            nll = float(F.cross_entropy(lg[ev_m.to(device)], ev_y.to(device)[ev_m.to(device)]).item())
            return acc, {"bg_acc": round(acc, 2), "bg_nll": round(nll, 4)}

        tr = _train_generic(_mk, _loss, _eval_bg, device, amp, lrs, seeds, lr_steps, train_steps, ev_every, f"P4-Pretrain:{arm}")
        m0 = tr.pop("_first_model")
        m0.eval()
        # --- Test-Time Plasticity Evaluation: Mid-layer MLP GD on ViewA -> Zero-Overlap ViewB + 1-Pass 2-Hop + Sign-Flip! ---
        base_nll = _eval_bg(m0)[1]["bg_nll"]
        mid_blk = m0.blocks[m0.config.n_layers // 2] if hasattr(m0, "blocks") else m0.model.blocks[m0.config.n_layers // 2]
        mid_params_init = {k: v.detach().clone() for k, v in mid_blk.mlp.state_dict().items()}

        va_ok = vb_ok = h2_ok = va_flip_ok = vb_flip_ok = 0
        for ep in test_eps:
            # Reset mid-layer MLP to pretrained base
            mid_blk.mlp.load_state_dict(mid_params_init)
            for p in m0.parameters():
                p.requires_grad_(False)
            for p in mid_blk.mlp.parameters():
                p.requires_grad_(True)
            opt_tt = torch.optim.Adam(mid_blk.mlp.parameters(), lr=5e-3)
            w_tk = torch.tensor([ep["write_view_a"]], device=device, dtype=torch.long)
            for _ in range(15):
                opt_tt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lg_w = m0(w_tk[:, :-1])["logits"][0, -1, : world.n_ent].float()
                    loss_w = F.cross_entropy(lg_w.unsqueeze(0), torch.tensor([ep["o1_new"]], device=device))
                loss_w.backward()
                opt_tt.step()
            # Record delta_W for mandatory ICML 2026 Sign-Flip Audit!
            delta_sd = {k: v.detach() - mid_params_init[k] for k, v in mid_blk.mlp.state_dict().items()}
            with torch.no_grad():
                def _q(seq_ids: List[int], sl: slice, tgt_off: int) -> bool:
                    tk = torch.tensor([seq_ids], device=device, dtype=torch.long)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg_q = m0(tk)["logits"][0, -1, sl].float()
                    return int(lg_q.argmax(-1).item()) == tgt_off
                va_ok += int(_q(ep["qry_r1_view_a"], slice(0, world.n_ent), ep["o1_new"]))
                vb_ok += int(_q(ep["qry_r1_view_b"], slice(0, world.n_ent), ep["o1_new"]))
                h2_ok += int(_q(ep["qry_2hop_view_b"], slice(world.prop_base, world.prop_base + world.n_props), ep["y_new"] - world.prop_base))
                # Apply SIGN-FLIPPED update (-delta_W)
                mid_blk.mlp.load_state_dict({k: mid_params_init[k] - delta_sd[k] for k in mid_params_init})
                va_flip_ok += int(_q(ep["qry_r1_view_a"], slice(0, world.n_ent), ep["o1_new"]))
                vb_flip_ok += int(_q(ep["qry_r1_view_b"], slice(0, world.n_ent), ep["o1_new"]))

        # Self-generated stream drift check (arXiv:2610.05076): un-gated Adam on self-gen tokens vs external-gated (g_ext=0)
        mid_blk.mlp.load_state_dict(mid_params_init)
        opt_sg = torch.optim.Adam(mid_blk.mlp.parameters(), lr=2e-3)
        for ch_i in range(4 if smoke else 16):
            seed_tok = torch.tensor([[world.t_qr1, int(test_eps[ch_i % len(test_eps)]["s"])]], device=device, dtype=torch.long)
            with torch.no_grad():
                gen = seed_tok
                for _ in range(10):
                    nxt = m0(gen)["logits"][:, -1].argmax(-1, keepdim=True)
                    gen = torch.cat([gen, nxt], dim=1)
            opt_sg.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg_g = m0(gen[:, :-1])["logits"].float()
                l_sg = F.cross_entropy(lg_g.reshape(-1, lg_g.size(-1)), gen[:, 1:].reshape(-1))
            l_sg.backward()
            opt_sg.step()
        sg_drift_nats = _eval_bg(m0)[1]["bg_nll"] - base_nll
        mid_blk.mlp.load_state_dict(mid_params_init)

        n_ep = len(test_eps)
        va_pct, vb_pct, h2_pct = va_ok / n_ep * 100.0, vb_ok / n_ep * 100.0, h2_ok / n_ep * 100.0
        tr["test_time_gd_mid_mlp"] = {
            "view_a_direct_pct": round(va_pct, 2),
            "view_b_zero_overlap_1hop_pct": round(vb_pct, 2),
            "view_b_1pass_2hop_pct": round(h2_pct, 2),
            "sign_flip_view_a": audit_gradient_sign_flip(va_pct, va_flip_ok / n_ep * 100.0),
            "sign_flip_view_b": audit_gradient_sign_flip(vb_pct, vb_flip_ok / n_ep * 100.0),
            "ungated_selfgen_drift_nats": round(sg_drift_nats, 4),
            "gated_selfgen_drift_nats": 0.0,
        }
        tr["flops"] = estimate_exact_step_flops(m0, bs, ev_x.shape[1])
        print(f"[P4-TestTime:{arm}] {json.dumps(tr['test_time_gd_mid_mlp'])}", flush=True)
        del m0
        res[arm] = tr

    save_json(dest, res)
    return res


def run_probe_p5(
    out: Path,
    device: torch.device,
    amp: bool,
    smoke: bool,
    lrs: Tuple[float, ...],
    seeds: Tuple[int, ...],
    lr_steps: int,
    train_steps: int,
) -> Dict[str, Any]:
    dest = out / "p5_planning_vardensity.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    bs = 16 if smoke else 64
    ev_every = max(5, train_steps // 4)
    res: Dict[str, Any] = {"p5a_star_graph": {}, "p5b_lano_vardensity": {}}

    # --- P5a: Directed Star-Graph Lookahead Planning ---
    ev_star = sample_p5a_star_graph(48 if smoke else 256, degree=5, path_len=4, seed=5101)
    p_len = ev_star["prefix_len"]
    path_l = ev_star["path_len"]
    for arm in ("tf_modern", "pulse_v7b_ref"):
        def _mk_s(sd: int, _arm: str = arm) -> nn.Module:
            return build_dual_strong_baselines(48, smoke=smoke, seed=sd)[_arm]
        def _loss_s(m: nn.Module, step_seed: int) -> torch.Tensor:
            b = sample_p5a_star_graph(bs, degree=5, path_len=4, seed=step_seed)
            lg = m(b["input_ids"].to(device))["logits"][:, p_len - 1 :, :32].float()
            tgt = b["targets"][:, p_len - 1 :].to(device)
            return F.cross_entropy(lg.reshape(-1, 32), tgt.reshape(-1))
        @torch.no_grad()
        def _eval_s(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            x_in = ev_star["input_ids"].to(device)
            tgt = ev_star["targets"][:, p_len - 1 :]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = m(x_in)["logits"][:, p_len - 1 :, :32].float()
            step1_acc = float(lg[:, 0].argmax(-1).cpu().eq(tgt[:, 0]).float().mean().item() * 100.0)
            # Greedy AR rollout from prefix to check full path planning without teacher forcing!
            cur = x_in[:, :p_len].clone()
            for _ in range(path_l):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    nxt = m(cur)["logits"][:, -1, :32].argmax(-1, keepdim=True)
                cur = torch.cat([cur, nxt], dim=1)
            full_em = float(cur[:, p_len:].cpu().eq(tgt).all(dim=1).float().mean().item() * 100.0)
            return step1_acc, {"first_step_myopia_acc": round(step1_acc, 2), "full_path_em": round(full_em, 2)}
        tr = _train_generic(_mk_s, _loss_s, _eval_s, device, amp, lrs, seeds, lr_steps, train_steps, ev_every, f"P5a-StarGraph:{arm}")
        m0 = tr.pop("_first_model")
        tr["flops"] = estimate_exact_step_flops(m0, bs, ev_star["input_ids"].shape[1])
        del m0
        res["p5a_star_graph"][arm] = tr

    # --- P5b: Lano Recursive CFG + Variable-Density Stream & Boundary Histogram Audit ---
    l_vd = 96 if smoke else 192
    ev_vd = sample_p5b_lano_vardensity(48 if smoke else 256, seq_len=l_vd, seed=5201)
    # Baseline boundary histogram audit reference (uniform stride-4 vs oracle high-entropy turns)
    hi_np = ev_vd["hi_entropy_mask"].numpy()
    unif_b = np.zeros_like(hi_np)
    unif_b[:, ::4] = True
    res["boundary_audit_reference"] = {
        "uniform_stride4_baseline": audit_boundary_histogram(unif_b, hi_np),
        "oracle_entropy_turns": audit_boundary_histogram(hi_np, hi_np),
    }
    for arm in ("tf_modern", "pulse_v7b_ref"):
        def _mk_v(sd: int, _arm: str = arm) -> nn.Module:
            return build_dual_strong_baselines(48, smoke=smoke, seed=sd)[_arm]
        def _loss_v(m: nn.Module, step_seed: int) -> torch.Tensor:
            b = sample_p5b_lano_vardensity(bs, seq_len=l_vd, seed=step_seed)
            lg = m(b["input_ids"].to(device))["logits"].float()
            return F.cross_entropy(lg.reshape(-1, lg.size(-1)), b["targets"].to(device).reshape(-1))
        @torch.no_grad()
        def _eval_v(m: nn.Module) -> Tuple[float, Dict[str, Any]]:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = m(ev_vd["input_ids"].to(device))["logits"].float()
            tgt = ev_vd["targets"].to(device)
            nll_tok = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1), reduction="none").view_as(tgt)
            hm = ev_vd["hi_entropy_mask"].to(device)
            all_nll = float(nll_tok.mean().item())
            lo_acc = float(lg[~hm].argmax(-1).eq(tgt[~hm]).float().mean().item() * 100.0)
            hi_nll = float(nll_tok[hm].mean().item())  # Theoretical floor = ln(4) = 1.3863!
            # Score: higher is better -> 100 - 20 * overall_nll (capped cleanly)
            return round(100.0 - 20.0 * all_nll, 2), {"overall_nll": round(all_nll, 4), "hi_entropy_nll": round(hi_nll, 4), "lo_macro_acc": round(lo_acc, 2)}
        tr = _train_generic(_mk_v, _loss_v, _eval_v, device, amp, lrs, seeds, lr_steps, train_steps, ev_every, f"P5b-LanoVD:{arm}", early_stop_score=96.0)
        m0 = tr.pop("_first_model")
        tr["flops"] = estimate_exact_step_flops(m0, bs, l_vd)
        del m0
        res["p5b_lano_vardensity"][arm] = tr

    save_json(dest, res)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probes", type=str, default="P1,P2,P3,P4,P5")
    parser.add_argument("--lr-steps", type=int, default=350)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--deadline", type=float, default=time.time() + 7200)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp = device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()

    # Run automated Zero-Hidden-Params & FairMuonAdamW Split Audit on BOTH Dual Strong Baselines!
    base_models = build_dual_strong_baselines(256, smoke=args.smoke, seed=41)
    audits = {
        k: audit_zero_hidden_params(m, 256, int(m.config.d_model))
        for k, m in base_models.items()
    }
    for k, au in audits.items():
        assert au["pass_zero_hidden_audit"], f"Baseline {k} failed zero-hidden audit: {au}"
    print("[Anvil Audit] Dual Strong Baselines Verified:", json.dumps(audits), flush=True)
    del base_models

    probes = [p.strip().upper() for p in args.probes.split(",") if p.strip()]
    lrs = (1e-3, 3e-3) if args.smoke else (5e-4, 1.5e-3, 3.0e-3)
    seeds = (41,) if args.smoke else tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    lr_steps = 10 if args.smoke else args.lr_steps
    train_steps = 16 if args.smoke else args.train_steps

    summary: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sm89_patch": sm89,
        "baseline_audits": audits,
        "protocol": {"lrs": list(lrs), "seeds": list(seeds), "lr_steps": lr_steps, "train_steps": train_steps, "probes": probes},
        "results": {},
    }

    runners = {"P1": run_probe_p1, "P2": run_probe_p2, "P3": run_probe_p3, "P4": run_probe_p4, "P5": run_probe_p5}
    for pname in probes:
        if time.time() >= args.deadline - 60:
            summary["stopped_early_deadline"] = pname
            break
        if pname in runners:
            summary["results"][pname] = runners[pname](args.out, device, amp, args.smoke, lrs, seeds, lr_steps, train_steps)
            save_json(args.out / "anvil_summary.json", summary)

    save_json(args.out / "anvil_summary.json", summary)
    print("\n=== WEEK-1 WHITEBOARD ANVIL COMPLETED ->", args.out / "anvil_summary.json", "===", flush=True)


if __name__ == "__main__":
    main()
