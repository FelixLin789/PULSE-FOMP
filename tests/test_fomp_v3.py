"""Unit & Invariant Verification Suite for FOMP-v3 (Open-Domain Multi-Token Mid-Layer Fast Weights)."""
from __future__ import annotations

import torch
import torch.nn as nn

from pulse.fomp_v3 import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module
from pulse.pulse_v8 import PULSEv8Config, PULSEv8Model


class _DummyHFDecoderLayer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return (hidden_states + 0.1 * torch.tanh(self.proj(hidden_states)),)


class _DummyHFModel(nn.Module):
    """Minimal HuggingFace Qwen2/Llama-style model structure (`model.layers`, `lm_head`, `logits`)."""

    def __init__(self, vocab: int = 128, d: int = 64, n_layers: int = 8) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList([_DummyHFDecoderLayer(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache: bool = False):
        h = self.model.embed_tokens(input_ids)
        for lyr in self.model.layers:
            h = lyr(h)[0]
        return {"logits": self.lm_head(h)}


def test_fomp_v3_zero_drift_sign_flip_and_multitoken_latching() -> None:
    torch.manual_seed(42)
    bb = _DummyHFModel(vocab=128, d=64, n_layers=8).eval()
    cfg = FOMPv3Config(d_model=64, read_layer=3, inject_layers=(3, 5), key_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
    fomp = FOMPv3Module(cfg).eval()
    wrap = FOMPv3BackboneWrapper(bb, fomp, freeze_backbone=True)

    # All backbone weights strictly frozen!
    assert all(not p.requires_grad for p in bb.parameters())
    assert all(p.requires_grad for p in fomp.parameters())

    # Variable-length cues + multi-token targets (lengths 1, 2, 3, 4)
    cues = [[5, 11, 22], [7, 13, 19, 25, 31], [9, 14], [3, 8, 12, 18]]
    tgts = [[40], [41, 42], [43, 44, 45], [46, 47, 48, 49]]
    dev = torch.device("cpu")

    # 1. g_ext=0 strictly bit-for-bit identical to unedited backbone!
    q_ids = torch.tensor([[7, 13, 19, 25, 31, 41, 42, 99]], dtype=torch.long)
    comp_0 = wrap.compile_token_episodes(cues, tgts, pad_id=0, device=dev, g_ext=0.0)
    with torch.no_grad():
        lg_base = wrap.forward_logits(q_ids, compiled=None)["logits"]
        lg_g0 = wrap.forward_logits(q_ids, compiled=comp_0)["logits"]
    torch.testing.assert_close(lg_g0, lg_base, rtol=0.0, atol=0.0)

    # 2. Multi-token causal latching: episode 1 has target length 2 ([41, 42]);
    #    query matches at pos 4 (token 31) -> pos 4 (lag 0) & pos 5 (lag 1) get non-zero latched_v,
    #    pos 6 (lag 2 >= L_y=2) automatically shuts off cleanly!
    comp_1 = wrap.compile_token_episodes(cues, tgts, pad_id=0, device=dev, g_ext=1.0)
    with torch.no_grad():
        out_p = wrap.forward_logits(q_ids, compiled=comp_1, sign_flip=False)
        out_n = wrap.forward_logits(q_ids, compiled=comp_1, sign_flip=True)
    lv = out_p["routing"]["latched_v"][0]  # [T, val_dim]
    assert lv[4].abs().sum().item() > 0.0, "Prompt-end pos 4 (lag 0) should latch!"
    assert lv[5].abs().sum().item() > 0.0, "Answer token pos 5 (lag 1) should latch!"
    assert lv[6].abs().sum().item() < 1e-6, "Post-phrase pos 6 (lag 2 >= L_y=2) must auto-shutoff!"

    # 3. Exact ICML 2026 Sign-Flip anti-symmetry on single-layer delta (including full-rank emb_skip + h_rms)!
    h_rms = torch.full((1, q_ids.shape[1], 1), 12.5)
    d_pos = fomp.compute_layer_delta(
        out_p["routing"]["latched_v"], 0, torch.float32, sign_flip=False,
        latched_e=out_p["routing"]["latched_e"], h_rms=h_rms,
    )
    d_neg = fomp.compute_layer_delta(
        out_p["routing"]["latched_v"], 0, torch.float32, sign_flip=True,
        latched_e=out_p["routing"]["latched_e"], h_rms=h_rms,
    )
    torch.testing.assert_close(d_pos, -d_neg, rtol=1e-6, atol=1e-6)
    assert out_p["routing"]["latched_e"][0, 4].abs().sum().item() > 0.0
    assert out_p["routing"]["latched_e"][0, 6].abs().sum().item() < 1e-6
    assert not torch.allclose(out_p["logits"][:, 4:6], out_n["logits"][:, 4:6])

    # 4. Greedy multi-token decode runs cleanly
    gen = wrap.greedy_generate([7, 13, 19, 25, 31], max_new_tokens=4, device=dev, compiled=comp_1)
    assert len(gen) == 4


def test_fomp_v3_bf16_backbone_with_fp32_fomp_outside_autocast() -> None:
    """Reproduces & prevents the BF16-backbone + FP32-FOMP outside-autocast dtype crash."""
    torch.manual_seed(7)
    bb = _DummyHFModel(vocab=128, d=64, n_layers=8).to(torch.bfloat16).eval()
    cfg = FOMPv3Config(d_model=64, read_layer=4, inject_layers=(4, 6), key_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
    fomp = FOMPv3Module(cfg).float().eval()
    wrap = FOMPv3BackboneWrapper(bb, fomp, freeze_backbone=True)
    dev = torch.device("cpu")

    # Outside torch.autocast: compile, forward_logits, and greedy_generate must NEVER throw dtype mismatch!
    comp = wrap.compile_token_episodes([[5, 11, 22], [7, 13, 19]], [[40, 41], [42]], pad_id=0, device=dev, g_ext=1.0)
    q_ids = torch.tensor([[5, 11, 22, 40, 41]], dtype=torch.long)
    with torch.no_grad():
        out = wrap.forward_logits(q_ids, compiled=comp)
    assert out["logits"].dtype == torch.bfloat16
    assert torch.isfinite(out["logits"].float()).all()
    gen = wrap.greedy_generate([5, 11, 22], max_new_tokens=3, device=dev, compiled=comp)
    assert len(gen) == 3


def test_fomp_v3_high_concurrency_m128_and_pulse_v8_attachment() -> None:
    torch.manual_seed(99)
    # Attach cleanly to our own PULSEv8Model (base-model agnosticism check!)
    pv8 = PULSEv8Model(PULSEv8Config.micro_test(vocab_size=112, use_fomp=False))
    cfg = FOMPv3Config(d_model=64, read_layer=2, inject_layers=(2, 3), key_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
    fomp = FOMPv3Module(cfg)
    wrap = FOMPv3BackboneWrapper(pv8, fomp, freeze_backbone=True)

    # Compile M=128 variable-length multi-token facts in one batch + 1st-order BF16 backward!
    cues = [torch.randint(4, 112, (int(3 + (i % 5)),)).tolist() for i in range(128)]
    tgts = [torch.randint(4, 112, (int(1 + (i % 4)),)).tolist() for i in range(128)]
    dev = torch.device("cpu")
    comp = wrap.compile_token_episodes(cues, tgts, pad_id=0, device=dev, g_ext=1.0)
    assert comp["k_fam"].shape == (1, 128, 32)
    assert comp["k_ent"].shape == (1, 128, cfg.ent_dim)
    assert comp["v_bank"].shape == (1, 128, 4, 16)
    assert comp["e_bank"].shape == (1, 128, 4, 64)

    q = torch.tensor([cues[7] + tgts[7]], dtype=torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = wrap.forward_logits(q, compiled=comp)
        loss = out["logits"].float().pow(2).mean() + out["routing"]["rel_logits"].pow(2).mean()
    loss.backward()
    assert fomp.key_proj.weight.grad is not None and fomp.key_proj.weight.grad.abs().sum().item() > 0.0
    assert fomp.key_mlp.weight.grad is not None and fomp.key_mlp.weight.grad.abs().sum().item() > 0.0
    assert fomp.ent_score.weight.grad is not None and fomp.ent_score.weight.grad.abs().sum().item() > 0.0
    assert fomp.ent_proj.weight.grad is not None and fomp.ent_proj.weight.grad.abs().sum().item() > 0.0
    assert fomp.ent_mlp.weight.grad is not None and fomp.ent_mlp.weight.grad.abs().sum().item() > 0.0
    assert fomp.val_down.weight.grad is not None and fomp.val_down.weight.grad.abs().sum().item() > 0.0
    assert fomp.val_ups[0].weight.grad is not None and fomp.val_ups[0].weight.grad.abs().sum().item() > 0.0
    assert fomp.emb_diag.grad is not None and fomp.emb_diag.grad.abs().sum().item() > 0.0
    assert fomp.emb_scale.grad is not None and fomp.emb_scale.grad.abs().sum().item() > 0.0
    assert fomp.direct_gate.weight.grad is not None and fomp.direct_gate.weight.grad.abs().sum().item() > 0.0
    assert fomp.ctx_norm.weight.grad is not None and fomp.ctx_norm.weight.grad.abs().sum().item() > 0.0
    assert all(p.grad is None for p in pv8.parameters()), "Frozen backbone must receive ZERO parameter grads!"


def test_dual_lock_salient_pool_equivalence_and_and_gate() -> None:
    """Verifies exact <1e-6 equivalence between causal & windowed salience pooling + strict Dual-Lock AND-gate."""
    from pulse.fomp_v3 import _extract_tail_window

    torch.manual_seed(123)
    cfg = FOMPv3Config(d_model=64, read_layer=3, inject_layers=(3,), key_dim=32, ent_dim=32, val_dim=16, max_tgt_tokens=2, top_k=2, ent_window=8)
    fomp = FOMPv3Module(cfg).eval()
    # Amplify salience scores to test extreme dynamic range stability (zero float32 cancellation!)
    with torch.no_grad():
        fomp.ent_score.weight.mul_(40.0)

    h = torch.randn(2, 24, 64)
    # 1. Exact equivalence at both short prefix (t=3 < win=8) and full window (t=19 >= win=8)
    causal_pool = fomp._causal_salient_pool(h, win=8)
    causal_ent = fomp.project_entity_key_seq(h)
    for t_idx in (3, 7, 19, 23):
        last_i = torch.full((2,), t_idx, dtype=torch.long)
        hw, wm = _extract_tail_window(h, last_i, win=8)
        win_pool = fomp._salient_pool_window(hw, wm)
        torch.testing.assert_close(causal_pool[:, t_idx, :], win_pool, rtol=1e-5, atol=1e-5)
        win_ent = fomp.project_entity_key_from_window(hw, wm, h_last=h[:, t_idx, :])
        torch.testing.assert_close(causal_ent[:, t_idx, :], win_ent, rtol=1e-5, atol=1e-5)

    # 2. Strict causality: mutating tokens t>=12 must have ZERO effect on t<12
    h2 = h.clone()
    h2[:, 12:, :] = torch.randn_like(h2[:, 12:, :]) * 10.0
    torch.testing.assert_close(fomp._causal_salient_pool(h, 8)[:, :12], fomp._causal_salient_pool(h2, 8)[:, :12], rtol=0.0, atol=0.0)

    # 3. Strict AND-Lock: if relation matches (sim_rel=1.0) but entity fails (sim_ent <= min_ent_tau),
    #    gate MUST be strictly 0.0 AND a locked-out decoy must NEVER steal a Top-K slot from a valid dual-lock fact!
    q_h = h[:1, :6, :]
    qf_rel = fomp.project_key(q_h, fomp._causal_salient_pool(q_h, 8))  # shape [1, 6, 32]
    from pulse.fomp_v3 import _causal_window_mean
    qf_rel_true = fomp.project_key(q_h, _causal_window_mean(q_h, cfg.key_ctx_window))[:, 5, :]
    qf_ent_true = fomp.project_entity_key_seq(q_h)[:, 5, :]

    # Episode 0: decoy with exact relation match (cos_rel=1.0) but orthogonal entity (cos_ent=0.0 -> LOCKED!)
    ent_orth = torch.randn_like(qf_ent_true)
    ent_orth = ent_orth - (ent_orth * qf_ent_true).sum(dim=-1, keepdim=True) * qf_ent_true
    ent_orth = torch.nn.functional.normalize(ent_orth, p=2, dim=-1)
    comp_locked = {
        "k_fam": qf_rel_true.unsqueeze(1),
        "k_ent": ent_orth.unsqueeze(1),
        "v_bank": torch.ones(1, 1, 2, 16),
        "e_bank": torch.ones(1, 1, 2, 64),
        "tgt_mask": torch.ones(1, 1, 2),
        "g_ext": torch.tensor(1.0),
    }
    rt_locked = fomp.route_and_latch(q_h, comp_locked)
    assert rt_locked["max_gate"][0, 5].item() == 0.0, "Entity lock failure MUST zero gate even when cos_rel == 1.0!"
    assert rt_locked["latched_v"][0, 5].abs().sum().item() == 0.0

    # Episode 0 & 1 = locked decoys (cos_rel=1.0, cos_ent=0.0), Episode 2 = valid dual-lock match (cos_rel=0.8, cos_ent=0.8)
    k_fam_3 = torch.stack([qf_rel_true, qf_rel_true, qf_rel_true], dim=1)
    k_ent_3 = torch.stack([ent_orth, ent_orth, qf_ent_true], dim=1)
    v_3 = torch.zeros(1, 3, 2, 16)
    v_3[:, 2, :, :] = 1.0
    comp_anti_theft = {
        "k_fam": k_fam_3,
        "k_ent": k_ent_3,
        "v_bank": v_3,
        "e_bank": torch.zeros(1, 3, 2, 64),
        "tgt_mask": torch.ones(1, 3, 2),
        "g_ext": torch.tensor(1.0),
    }
    rt_win = fomp.route_and_latch(q_h, comp_anti_theft)
    assert rt_win["latched_v"][0, 5].abs().sum().item() > 0.0, "Valid Dual-Lock fact MUST beat locked-out decoys in Top-K!"


def test_causal_window_mean_strict_causality() -> None:
    from pulse.fomp_v3 import _causal_window_mean

    torch.manual_seed(0)
    x1 = torch.randn(2, 20, 16)
    x2 = x1.clone()
    x2[:, 10:, :] = torch.randn_like(x2[:, 10:, :])
    y1 = _causal_window_mean(x1, win=6)
    y2 = _causal_window_mean(x2, win=6)
    # Strictly causal: altering t>=10 must have ZERO effect on t<10!
    torch.testing.assert_close(y1[:, :10], y2[:, :10], rtol=0.0, atol=0.0)
    # Exact window match at t=14 (average of x1[:, 9:15])
    torch.testing.assert_close(y1[:, 14], x1[:, 9:15].mean(dim=1), rtol=1e-6, atol=1e-6)
    # Exact prefix match at t=3 (average of x1[:, 0:4])
    torch.testing.assert_close(y1[:, 3], x1[:, 0:4].mean(dim=1), rtol=1e-6, atol=1e-6)


def test_fomp_v36_val_rms_invariance_and_val_dropout() -> None:
    """Verifies exact weight-RMS invariance of val_path (prevents 8B norm blowup) & val_drop_mask behavior."""
    torch.manual_seed(321)
    bb = _DummyHFModel(vocab=128, d=64, n_layers=8).eval()
    cfg = FOMPv3Config(d_model=64, read_layer=3, inject_layers=(3, 7), key_dim=32, ent_dim=32, val_dim=16, max_tgt_tokens=4, top_k=2, rel_hidden=16)
    fomp = FOMPv3Module(cfg).eval()
    wrap = FOMPv3BackboneWrapper(bb, fomp, freeze_backbone=True)
    dev = torch.device("cpu")

    cues = [[5, 11, 22], [7, 13, 19]]
    tgts = [[40, 41, 42], [43, 44]]
    comp = wrap.compile_token_episodes(cues, tgts, pad_id=0, device=dev, g_ext=1.0)
    q = torch.tensor([[5, 11, 22, 40, 41]], dtype=torch.long)

    # 1. Exact weight-RMS scale invariance: multiplying val_ups[0].weight by 100x must NOT blow up delta!
    with torch.no_grad():
        out_before = wrap.forward_logits(q, compiled=comp)["logits"]
        fomp.val_ups[0].weight.mul_(100.0)
        out_after = wrap.forward_logits(q, compiled=comp)["logits"]
    torch.testing.assert_close(out_after, out_before, rtol=1e-4, atol=1e-4)

    # 2. val_drop_mask=0 zeroes latched_v while preserving full-rank latched_e and valid d_gate!
    with torch.no_grad():
        out_drop = wrap.forward_logits(q, compiled=comp, val_drop_mask=torch.zeros(1))
    assert out_drop["routing"]["latched_v"].abs().sum().item() == 0.0
    assert out_drop["routing"]["latched_e"][0, 2].abs().sum().item() > 0.0
    assert 0.0 < out_drop["routing"]["d_gate"][0, 2].item() <= 1.0


def test_mquake_remastered_flat_schema_parser() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from fomp_v3_open_edit import _parse_mquake_row

    row = {
        "questions": ["On which continent is the country located, whose citizen is Emma Bunton?"],
        "answer": "Europe",
        "new_answer": "Asia",
        "requested_rewrite": [{
            "prompt": "{} is located in the continent of",
            "subject": "United Kingdom",
            "target_new_str": "Asia",
            "target_true_str": "Europe",
        }],
    }
    rec = _parse_mquake_row(row)
    assert rec is not None
    assert rec.cue == "United Kingdom is located in the continent of"
    assert rec.target_new == " Asia" and rec.multihop_answer_new == " Asia"
