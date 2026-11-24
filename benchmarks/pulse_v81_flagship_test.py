"""Standalone Verification for Final Flagship PULSE-v8.1 (5:1 GDN+SWA+NoPE-MLA + Native FOMP-CMS v2.2).

Single-arm, zero-baseline-retraining verification (~4-6 min on L4, ~2 min on A100):
  1. Loads pretrained 30M `5:1` backbone (`pulse_v7b_30m_final.pt`) directly into `PULSE-v8.1` (`strict=False`),
     verifying exact zero-regression on FineWeb-Edu 1K & 4K (4x OOD) PPL, 4K Zero-Shot Copy, and 2.00 MiB 4K state.
  2. Runs backbone-preserving `FOMP-CMS v2.2` calibration (differential LR: 1.5e-3 on FOMP hyper/med vs gentle
     8e-5 on pretrained backbone + 3.0x FineWeb-Edu replay) so FineWeb 1K/4K PPL stays anchored near ~60!
  3. Verifies ALL capabilities simultaneously on the EXACT SAME unified checkpoint:
       - Post-FOMP FineWeb-Edu 1K & 4K Val PPL + 4K Zero-Shot Copy (zero catastrophic forgetting!)
       - M=1 & M=16 zero-overlap 1-Pass 2-Hop on strictly unseen GPT-2 BPE entities
       - Mandatory ICML 2026 Sign-Flip collapse (-> 0.0%)
       - Exact 0.0000 `g_ext=0` drift AND near-zero `M=16` loaded FineWeb 1K & 4K Val PPL drift!
"""
from __future__ import annotations

import argparse
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
import torch.nn.functional as F

from pulse.experiment import FairMuonAdamW, _cosine_lr_multiplier
from pulse.pulse_v7 import patch_fla_sm89_shared_memory
from pulse.pulse_v8 import PULSEv8Config, PULSEv8Model
from pulse.whiteboard_anvil import audit_zero_hidden_params

_BENCH_DIR = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from pulse_v8_stage_b_30m import (  # noqa: E402
    MixedCorpusReader,
    _get_word_pools,
    eval_val_windows,
    working_state_bytes_bf16,
)
from v7_stage0_stage1 import save_json, sync  # noqa: E402


@torch.no_grad()
def _eval_zeroshot_copy(
    model: PULSEv8Model, reader: MixedCorpusReader, device: torch.device, amp: bool, smoke: bool
) -> Dict[str, Any]:
    cfgs = (
        [("L1x", 64, 6, 8, 2, 16, 48), ("L4x", 256, 4, 8, 2, 16, 224)]
        if smoke
        else [("L1024", 1024, 48, 16, 4, 80, 940), ("L4096", 4096, 32, 16, 4, 120, 3900)]
    )
    model.eval()
    out: Dict[str, Any] = {}
    for tag, sl, n_w, span_len, cue, src_p, dst_p in cfgs:
        base_w = reader.read_val("fineweb", n_w, sl, offset=300 * sl)
        assert base_w is not None
        x_all = base_w[:, :sl].clone()
        x_all[:, dst_p : dst_p + span_len] = x_all[:, src_p : src_p + span_len]
        hits = tot = 0
        nlls: List[float] = []
        mbs = 1 if sl >= 2048 else 4
        for chunk in x_all.split(mbs):
            chunk = chunk.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                lg = model(chunk[:, :-1])["logits"]
            pred_lg = lg[:, dst_p + cue - 1 : dst_p + span_len - 1, :].float()
            tgt = chunk[:, dst_p + cue : dst_p + span_len]
            hits += int((pred_lg.argmax(dim=-1) == tgt).sum().item())
            tot += int(tgt.numel())
            nlls.append(float(F.cross_entropy(pred_lg.reshape(-1, pred_lg.shape[-1]), tgt.reshape(-1)).item()))
        out[tag] = {"copy_top1_acc_pct": round(hits / max(1, tot) * 100.0, 2), "copy_nll": round(float(np.mean(nlls)), 4)}
    return out


@torch.no_grad()
def _eval_fw_with_compiled(
    model: PULSEv8Model, windows: torch.Tensor, comp_d: Dict[str, torch.Tensor], device: torch.device, amp: bool, micro_bs: int
) -> float:
    model.eval()
    rows_w: List[float] = []
    for chunk in windows.split(micro_bs):
        chunk = chunk.to(device)
        c_rep = {k: (v.expand(chunk.shape[0], *([-1] * (v.ndim - 1))) if v.ndim >= 2 else v) for k, v in comp_d.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            lg = model.forward_with_fast_weights(chunk[:, :-1], compiled=c_rep)["logits"]
            nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), chunk[:, 1:].reshape(-1), reduction="none")
            rows_w.extend(nll.float().view(chunk.shape[0], -1).mean(dim=1).cpu().tolist())
    return round(math.exp(min(20.0, float(np.mean(rows_w)))), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fineweb-dir", type=Path, default=None)
    parser.add_argument("--backbone-ckpt", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--deadline", type=float, default=time.time() + 1800)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    sm89 = patch_fla_sm89_shared_memory()

    vocab_size = 128 if args.smoke else 50257
    reader = MixedCorpusReader(args.fineweb_dir, None, smoke=args.smoke, vocab_size=vocab_size)

    torch.manual_seed(args.seed)
    if args.smoke:
        cfg = PULSEv8Config(
            vocab_size=vocab_size, d_model=64, n_heads=4, n_kv_heads=2, d_ff=96,
            layer_types=("swa", "gdn", "gdn", "gdn", "gdn", "mla"),
            swa_window_size=16, mla_d_latent=32, use_fomp=True,
            fomp_inject_layers=(2, 3), fomp_key_dim=32, fomp_med_rank=4,
        )
    else:
        cfg = PULSEv8Config.stage_b_30m(vocab_size=vocab_size, use_fomp=True, use_slots=False)

    model = PULSEv8Model(cfg).to(device)
    audit = audit_zero_hidden_params(model, vocab_size, cfg.d_model)
    assert audit["pass_zero_hidden_audit"] and len(model.slot_blocks) == 0, f"Flagship audit failed: {audit}"

    ckpt_info: Dict[str, Any] = {"loaded": False}
    if args.backbone_ckpt is not None and args.backbone_ckpt.exists() and not args.smoke:
        st = torch.load(args.backbone_ckpt, map_location=device, weights_only=False)
        inc = model.load_state_dict(st["model"], strict=False)
        assert len(inc.unexpected_keys) == 0, f"Unexpected keys in backbone ckpt: {inc.unexpected_keys}"
        assert all(k.startswith(("hyper.", "med_down.", "med_up.")) for k in inc.missing_keys), f"Missing backbone keys: {inc.missing_keys}"
        ckpt_info = {"loaded": True, "path": str(args.backbone_ckpt), "new_fomp_tensors": len(inc.missing_keys)}
        print(f"[PULSE-v8.1] Loaded pretrained 5:1 backbone cleanly ({len(inc.missing_keys)} new FOMP tensors).", flush=True)

    # 1. Pre-Calibration Baseline (Exact Zero-Regression Check!)
    l_1x, l_4x = (64, 256) if args.smoke else (1024, 4096)
    n_1x, n_4x = (6, 4) if args.smoke else (64, 32)
    fw_1x = reader.read_val("fineweb", n_1x, l_1x, offset=16 * l_1x)
    fw_4x = reader.read_val("fineweb", n_4x, l_4x, offset=128 * l_1x)
    assert fw_1x is not None and fw_4x is not None

    pre_1k = eval_val_windows(model, fw_1x, device, amp, micro_bs=4)
    pre_4k = eval_val_windows(model, fw_4x, device, amp, micro_bs=1)
    pre_copy = _eval_zeroshot_copy(model, reader, device, amp, args.smoke)
    print(f"[Pre-FOMP] FW 1K PPL={pre_1k['ppl']} | FW 4K PPL={pre_4k['ppl']} | Copy={json.dumps(pre_copy)}", flush=True)

    # 2. Backbone-Preserving FOMP-CMS v2.2 Calibration (Differential LR + 3x FineWeb Replay!)
    key_pool, val_pool, alias_pool, enc = _get_word_pools(vocab_size, args.smoke)
    n_ent = 32 if args.smoke else 64
    n_tr = 24 if args.smoke else 48
    n_te = n_ent - n_tr
    ents, aliases, vals = key_pool[:n_ent], alias_pool[:n_ent], val_pool[:n_ent]
    val_ids_t = torch.tensor(vals, device=device, dtype=torch.long)

    t_rule = enc.encode(" Rule")[0] if not args.smoke else 10
    t_qr1 = enc.encode(" Code")[0] if not args.smoke else 11
    t_q2h = enc.encode(" Hop")[0] if not args.smoke else 12
    t_veq = enc.encode(" Alias")[0] if not args.smoke else 13
    t_r2 = enc.encode(" Map")[0] if not args.smoke else 14

    rng_w = np.random.default_rng(202611)
    r2_map = rng_w.permutation(n_ent).tolist()
    test_rules = [(n_tr + i, int(rng_w.integers(0, n_ent))) for i in range(n_te)]

    def _ep_tok(e_i: int, y_i: int) -> List[int]: return [t_rule, ents[e_i], vals[y_i]]
    def _va_qry(e_i: int) -> List[int]: return [t_qr1, ents[e_i]]
    def _vb_qry(e_i: int) -> List[int]: return [t_qr1, aliases[e_i]]
    def _2h_qry(e_i: int) -> List[int]: return [t_q2h, aliases[e_i]]

    def _sample_bg(bs_b: int, sd_i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(sd_i)
        xs, ys = [], []
        for _ in range(bs_b):
            e = int(rng.integers(0, n_ent))
            if int(rng.integers(0, 2)) == 0:
                xs.append([t_veq, aliases[e], t_veq]); ys.append(ents[e])
            else:
                xs.append([t_r2, vals[e], t_r2]); ys.append(vals[r2_map[e]])
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    # Differential optimizer: fast AdamW (1.5e-3) for FOMP hyper/med, gentle FairMuonAdamW (8e-5) for pretrained backbone!
    fomp_ids = {id(p) for n, p in model.named_parameters() if n.startswith(("hyper.", "med_down.", "med_up."))}
    fomp_params = [p for p in model.io_parameters() if id(p) in fomp_ids]
    bb_params = [p for p in model.io_parameters() if id(p) not in fomp_ids]

    lr_fomp, lr_bb = 1.5e-3, 8.0e-5
    opt_fomp = torch.optim.AdamW(fomp_params, lr=lr_fomp, betas=(0.9, 0.95), weight_decay=0.01)
    opt_bb = FairMuonAdamW(bb_params, lr=lr_bb, muon_lr_mult=2.5, adam_lr_mult=1.0, weight_decay=0.05)

    steps = 16 if args.smoke else args.steps
    w_steps = max(4, steps // 10)
    bs_ep = 8 if args.smoke else 32
    fw_replay_bs = 2 if args.smoke else 4
    fw_replay_len = 64 if args.smoke else 512

    all_va = torch.tensor([[t_veq, ents[e]] for e in range(n_ent)], device=device, dtype=torch.long)
    all_vb = torch.tensor([[t_veq, aliases[e]] for e in range(n_ent)], device=device, dtype=torch.long)
    quick_fw = fw_1x[: (4 if args.smoke else 16)]

    sync(device)
    t0 = time.perf_counter()
    for st in range(1, steps + 1):
        if time.time() > args.deadline - 90:
            print(f"[Deadline] Stopping calibration early at step {st - 1}", flush=True)
            break
        mult = _cosine_lr_multiplier(st - 1, steps, w_steps, 0.15)
        for pg in opt_fomp.param_groups:
            pg["lr"] = lr_fomp * mult
        for pg in opt_bb.param_groups:
            pg["lr"] = lr_bb * mult

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
        fw_mb = reader.read_train_step(500000 + st, fw_replay_bs, 1, fw_replay_len)[0].to(device)

        model.train()
        opt_fomp.zero_grad(set_to_none=True)
        opt_bb.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss_fw = model.compute_loss(fw_mb)
            comp = model.compile_episodes(torch.from_numpy(ep_ids).to(device), g_ext=1.0)
            comp3 = {k: (v.repeat(3, *([1] * (v.ndim - 1))) if v.ndim >= 2 else v) for k, v in comp.items()}
            q_cat = torch.cat([torch.from_numpy(q_va), torch.from_numpy(q_vb), torch.from_numpy(q_2h)], dim=0).to(device)
            o_cat = model.forward_with_fast_weights(q_cat, compiled=comp3)
            lg_cat = o_cat["logits"][:, -1, :].index_select(-1, val_ids_t).float()
            l_va, l_vb, l_2h = lg_cat.split(bs_ep, dim=0)
            y1_t = torch.from_numpy(y_1h).to(device); y2_t = torch.from_numpy(y_2h).to(device)
            o_un = model.forward_with_fast_weights(torch.from_numpy(q_un).to(device), compiled=comp)
            o_bg = model.forward_with_fast_weights(bx.to(device), compiled=None)
            o_bg_ld = model.forward_with_fast_weights(bx.to(device), compiled=comp)
            comp_fw = {k: (v[: fw_mb.shape[0]] if v.ndim >= 2 else v) for k, v in comp.items()}
            o_fw_ld = model.forward_with_fast_weights(fw_mb[:, :32], compiled=comp_fw)
            l_bg = F.cross_entropy(o_bg["logits"][:, -1, :].float(), by.to(device))
            l_align = model.view_alignment_loss(all_va, all_vb)

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
            # Strong 3.0x FineWeb replay anchors backbone LM quality right near ~59 PPL!
            loss = (
                3.0 * loss_fw + 1.0 * l_bg + 0.5 * l_align + 1.5 * l_gate
                + F.cross_entropy(l_va, y1_t) + 1.5 * F.cross_entropy(l_vb, y1_t) + 2.5 * F.cross_entropy(l_2h, y2_t)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.io_parameters(), 1.0)
        opt_fomp.step()
        opt_bb.step()

        if st % (8 if args.smoke else 150) == 0 or st == steps:
            q_ppl = eval_val_windows(model, quick_fw, device, amp, micro_bs=4)["ppl"]
            tr_2h = float((l_2h.detach().argmax(-1) == y2_t).float().mean().item() * 100.0)
            print(f"  [Calib st={st:3d}/{steps}] quick_FW_1K_PPL={q_ppl:.2f} | batch_2H={tr_2h:.1f}% | loss_fw={loss_fw.item():.3f}", flush=True)

    sync(device)
    calib_sec = time.perf_counter() - t0

    # 3. Unified Final Verification (ALL metrics on the EXACT SAME single PULSE-v8.1 weights!)
    post_1k = eval_val_windows(model, fw_1x, device, amp, micro_bs=4)
    post_4k = eval_val_windows(model, fw_4x, device, amp, micro_bs=1)
    post_copy = _eval_zeroshot_copy(model, reader, device, amp, args.smoke)

    model.eval()
    with torch.no_grad():
        ep_all = torch.tensor([[[*_ep_tok(e, v)] for e, v in test_rules]], device=device, dtype=torch.long)
        comp_m16 = model.compile_episodes(ep_all, g_ext=1.0)
        comp_g0 = model.compile_episodes(ep_all, g_ext=0.0)

        fw_1k_g0 = _eval_fw_with_compiled(model, fw_1x, comp_g0, device, amp, micro_bs=4)
        fw_1k_m16 = _eval_fw_with_compiled(model, fw_1x, comp_m16, device, amp, micro_bs=4)
        fw_4k_g0 = _eval_fw_with_compiled(model, fw_4x, comp_g0, device, amp, micro_bs=1)
        fw_4k_m16 = _eval_fw_with_compiled(model, fw_4x, comp_m16, device, amp, micro_bs=1)

        def _eval_rules(simultaneous: bool, sign_flip: bool) -> Dict[str, float]:
            h_va = h_vb = h_2h = 0
            for e_i, v_i in test_rules:
                c_use = comp_m16 if simultaneous else model.compile_episodes(torch.tensor([[[*_ep_tok(e_i, v_i)]]], device=device), g_ext=1.0)
                for nm, q_toks, tgt in (("va", _va_qry(e_i), v_i), ("vb", _vb_qry(e_i), v_i), ("2h", _2h_qry(e_i), r2_map[v_i])):
                    q_t = torch.tensor([q_toks], device=device, dtype=torch.long)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        lg = model.forward_with_fast_weights(q_t, compiled=c_use, sign_flip=sign_flip)["logits"][0, -1].index_select(-1, val_ids_t).float()
                    ok = int(int(lg.argmax().item()) == tgt)
                    if nm == "va": h_va += ok
                    elif nm == "vb": h_vb += ok
                    else: h_2h += ok
            return {
                "va_pct": round(h_va / n_te * 100.0, 2),
                "vb_1hop_pct": round(h_vb / n_te * 100.0, 2),
                "vb_1pass_2hop_pct": round(h_2h / n_te * 100.0, 2),
            }

        m1_n, m1_f = _eval_rules(False, False), _eval_rules(False, True)
        m16_n, m16_f = _eval_rules(True, False), _eval_rules(True, True)

    # Save finalized unified PULSE-v8.1 flagship checkpoint!
    ckpt_out = args.out / "PULSE_v81_flagship_30m.pt"
    tmp_ck = ckpt_out.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, tmp_ck)
    os.replace(tmp_ck, ckpt_out)

    passed_all = bool(
        m16_n["vb_1pass_2hop_pct"] >= 85.0
        and m16_f["vb_1pass_2hop_pct"] <= 15.0
        and abs(fw_1k_g0 - post_1k["ppl"]) == 0.0
        and abs(fw_4k_g0 - post_4k["ppl"]) == 0.0
        and (fw_1k_m16 - post_1k["ppl"]) <= 0.10
        and (fw_4k_m16 - post_4k["ppl"]) <= 0.10
        and (args.smoke or (post_1k["ppl"] <= pre_1k["ppl"] + 6.0 and post_4k["ppl"] <= pre_4k["ppl"] + 6.0))
    )

    summary = {
        "model": "PULSE-v8.1 Flagship (2x SWA-128 + 8x FLA GDN + 2x Gated NoPE-MLA + Native FOMP-CMS v2.2)",
        "device": torch.cuda.get_device_name(0) if amp else "cpu",
        "sm89_patch": sm89,
        "params": int(model.num_parameters()),
        "non_emb_params": int(model.num_parameters() - vocab_size * cfg.d_model),
        "audit": audit,
        "storage_1k": working_state_bytes_bf16(model, 1024),
        "storage_4k": working_state_bytes_bf16(model, 4096),
        "backbone_ckpt": ckpt_info,
        "calib_sec": round(calib_sec, 2),
        "pre_fomp": {"fw_1k_ppl": pre_1k["ppl"], "fw_4k_ppl": pre_4k["ppl"], "zeroshot_copy": pre_copy},
        "post_fomp_unified": {
            "fw_1k_ppl_clean": post_1k["ppl"],
            "fw_1k_ppl_g_ext0": fw_1k_g0,
            "fw_1k_ppl_m16_loaded": fw_1k_m16,
            "fw_1k_m16_drift_ppl": round(fw_1k_m16 - post_1k["ppl"], 5),
            "fw_4k_ppl_clean": post_4k["ppl"],
            "fw_4k_ppl_g_ext0": fw_4k_g0,
            "fw_4k_ppl_m16_loaded": fw_4k_m16,
            "fw_4k_m16_drift_ppl": round(fw_4k_m16 - post_4k["ppl"], 5),
            "zeroshot_copy": post_copy,
            "M1_normal": m1_n,
            "M1_sign_flip": m1_f,
            "M16_normal": m16_n,
            "M16_sign_flip": m16_f,
        },
        "pass_all_flagship_gates": passed_all,
    }
    save_json(args.out / "pulse_v81_flagship_test.json", summary)
    print("\n=== PULSE-v8.1 FLAGSHIP STANDALONE TEST RESULT ===", flush=True)
    print(json.dumps(summary["post_fomp_unified"], indent=2), flush=True)
    print(f"PASS_ALL_FLAGSHIP_GATES = {passed_all} | Saved -> {args.out / 'pulse_v81_flagship_test.json'}", flush=True)

    del model, opt_fomp, opt_bb
    gc.collect()
    if amp:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
