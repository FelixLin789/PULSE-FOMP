"""Unit & invariance tests for FOMP-v4.1 (Strict Multi-Hop Bridge + Exact v3.6 Single-Hop + Lazy DynamicCache) & 4-Shot IKE."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from pulse.fomp_v3 import FOMPv3BackboneWrapper  # noqa: E402
from pulse.fomp_v4 import (  # noqa: E402
    FOMPv4BackboneWrapper,
    FOMPv4Config,
    FOMPv4Module,
    format_virtual_kv_fact,
    strip_cue_copula,
)
from fomp_v4_multihop_stage3 import (  # noqa: E402
    CleanDenseIKERetriever,
    IKEPromptFormatter,
    _clause_to_embedded_np,
    _compute_hop_subtype_breakdown,
    _make_compositional_qa,
    format_ike_fact_line,
    strip_qa_wrapper,
)
from fomp_v3_open_edit import EditRecord, _SmokeTokenizer, _make_smoke_datasets  # noqa: E402


def _tiny_llama(n_layers: int = 8, d: int = 64) -> LlamaForCausalLM:
    torch.manual_seed(7)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=d,
        intermediate_size=128,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return LlamaForCausalLM(cfg).eval()


def test_fomp_v41_zero_drift_exact_v36_single_hop_and_lazy_dynamic_cache() -> None:
    device = torch.device("cpu")
    backbone = _tiny_llama(n_layers=8, d=64)
    v4_cfg = FOMPv4Config.for_backbone(64, 8)
    fomp = FOMPv4Module(v4_cfg).eval()
    wrap = FOMPv4BackboneWrapper(backbone, fomp, freeze_backbone=True)
    v3_wrap = FOMPv3BackboneWrapper(backbone, fomp.direct, freeze_backbone=True)
    tok = _SmokeTokenizer()

    cue_ids = [[1, 10, 11, 12, 13], [1, 20, 21, 22, 23]]
    tgt_ids = [[30, 31], [40, 41, 42]]
    facts = ["- EntityA is produced by NewCorp.", "- OrgB was founded in NewCity."]

    # 1. Exact 0.000000 drift when `g_ext == 0.0` in BOTH modes
    comp_off = wrap.compile_episodes(
        cue_ids, tgt_ids, pad_id=0, device=device, bos_id=1,
        fact_texts=facts, tokenizer=tok, g_ext=0.0, precompile_kv=False,
    )
    q = torch.tensor([[1, 50, 51, 52, 53, 54, 55, 56]])
    with torch.no_grad():
        base_lg = backbone(q).logits
        lg_leg = wrap.forward_logits(q, compiled=comp_off, mode="legible")["logits"]
        lg_bkv = wrap.forward_logits(q, compiled=comp_off, mode="bridge_kv")["logits"]
    assert (base_lg - lg_leg).abs().max().item() == 0.0
    assert (base_lg - lg_bkv).abs().max().item() == 0.0

    # 2. Exact bit-for-bit equivalence (`0.000000` diff!) with `FOMP-v3.6` on ALL single-hop queries (`mh_prob <= 0.80`)!
    comp_on = wrap.compile_episodes(
        cue_ids, tgt_ids, pad_id=0, device=device, bos_id=1,
        fact_texts=facts, tokenizer=tok, g_ext=1.0, precompile_kv=False,
    )
    assert comp_on["kv_snaps"] is None and len(comp_on["_lazy_kv_snaps"]) == 0, "Expected lazy KV cache at compile time"
    comp_v3 = v3_wrap.compile_token_episodes(cue_ids, tgt_ids, pad_id=0, device=device, g_ext=1.0)
    sh_q = torch.tensor([cue_ids[0]])
    with torch.no_grad():
        lg_v3 = v3_wrap.forward_logits(sh_q, compiled=comp_v3)["logits"]
        res_v4_leg = wrap.forward_logits(sh_q, compiled=comp_on, mode="legible")
        res_v4_bkv = wrap.forward_logits(sh_q, compiled=comp_on, mode="bridge_kv")
    assert res_v4_leg["mode_used"] == "direct_v36" and res_v4_bkv["mode_used"] == "direct_v36"
    assert (lg_v3 - res_v4_leg["logits"]).abs().max().item() == 0.0
    assert (lg_v3 - res_v4_bkv["logits"]).abs().max().item() == 0.0
    assert len(comp_on["_lazy_kv_snaps"]) == 0, "Single-hop query must NOT allocate any DynamicCache snapshot!"

    # 3. Verify lazy `_run_with_virtual_kv_ids` + incremental `past_key_values` matches full concatenation to < 1e-5!
    unrel_ids = [1, 90, 91, 92, 93, 94, 95, 96, 97]
    with torch.no_grad():
        lg_vkv, ex_tok = wrap._run_with_virtual_kv_ids(unrel_ids, [0], comp_on, device, sign_flip=False)
        full_concat = torch.tensor([comp_on["pref_full_ids"][0] + unrel_ids[1:]])
        lg_ref = backbone(full_concat).logits[:, -len(unrel_ids[1:]) :, :]
    assert ex_tok > 0 and 0 in comp_on["_lazy_kv_snaps"]
    assert (lg_vkv - lg_ref).abs().max().item() < 1e-5


def test_fomp_v41_strict_multihop_switch_and_incremental_vkv_decode() -> None:
    device = torch.device("cpu")
    backbone = _tiny_llama(n_layers=8, d=64)
    v4_cfg = FOMPv4Config.for_backbone(64, 8)
    v4_cfg.init_span_rel_tau = 0.20
    v4_cfg.min_span_rel_tau = 0.15
    v4_cfg.init_span_ent_tau = 0.20
    v4_cfg.min_span_ent_tau = 0.15
    fomp = FOMPv4Module(v4_cfg).eval()
    wrap = FOMPv4BackboneWrapper(backbone, fomp, freeze_backbone=True)
    tok = _SmokeTokenizer()

    cue = [1, 14, 15, 16, 17]
    tgt = [61, 62]
    comp = wrap.compile_episodes(
        [cue], [tgt], pad_id=0, device=device, bos_id=1,
        fact_texts=["- Cue14 is Target61."], tokenizer=tok, g_ext=1.0, precompile_kv=False,
    )
    mh_q = [1, 14, 15, 16, 17, 80, 81, 82, 83]
    mh_t = torch.tensor([mh_q])

    # When `mh_fc2.bias = -3.0` (`mh_prob = 0.047 <= 0.80`), query stays strictly on `direct_v36`
    with torch.no_grad():
        res_off = wrap.forward_logits(mh_t, compiled=comp, mode="legible")
    assert res_off["mode_used"] == "direct_v36"

    # When `mh_detector` fires (`mh_fc2.bias = +3.0` -> `mh_prob = 0.95 > 0.80`) AND `g_span > 0.20`:
    # - `legible` triggers `stage_a_legible` (`0` extra tokens)
    # - `bridge_kv` triggers `virtual_kv:span_hop1` and decodes multi-token output identically to full concat!
    fomp.mh_fc2.bias.data.fill_(3.0)
    with torch.no_grad():
        base_lg = backbone(mh_t).logits
        res_a = wrap.forward_logits(mh_t, compiled=comp, mode="legible")
        res_flip = wrap.forward_logits(mh_t, compiled=comp, mode="legible", sign_flip=True)
    assert res_a["mode_used"] == "stage_a_legible"
    assert (res_a["logits"] - base_lg).abs().max().item() > 1e-4
    assert (res_a["logits"] - res_flip["logits"]).abs().max().item() > 1e-4

    gen_vkv, ex_vkv = wrap.greedy_generate(mh_q, max_new_tokens=3, device=device, compiled=comp, mode="bridge_kv")
    # Compare incremental `DynamicCache` 3-token generation vs brute-force token-by-token full concatenation!
    ref_cur = list(comp["pref_full_ids"][0] + mh_q[1:])
    ref_gen = []
    with torch.no_grad():
        for _ in range(3):
            nxt = int(backbone(torch.tensor([ref_cur])).logits[0, -1].argmax().item())
            ref_gen.append(nxt)
            ref_cur.append(nxt)
    assert ex_vkv > 0 and gen_vkv == ref_gen


def test_fomp_v42_hop2_logit_lens_implicit_retrieval_and_surface_mask() -> None:
    """Verify Route 3 (`hop2_logit_lens`) detects unmentioned latent subjects, blocks surface-present ones, and reuses lock."""
    device = torch.device("cpu")
    backbone = _tiny_llama(n_layers=8, d=64)
    v4_cfg = FOMPv4Config.for_backbone(64, 8)
    fomp = FOMPv4Module(v4_cfg).eval()
    # Non-zero direct injection gain so forced Hop-2 injection measurably shifts logits vs base!
    for up in fomp.direct.val_ups:
        torch.nn.init.normal_(up.weight, 0.0, 0.05)
    fomp.direct.emb_scale.data.fill_(1.0)
    wrap = FOMPv4BackboneWrapper(backbone, fomp, freeze_backbone=True)
    tok = _SmokeTokenizer()

    # Episode 0 has cue `[1, 14, 15, 16, 17]`; multi-hop query `[1, 90..97]` shares ZERO non-BOS tokens with Episode 0!
    cue = [1, 14, 15, 16, 17]
    tgt = [61, 62]
    comp = wrap.compile_episodes([cue], [tgt], pad_id=0, device=device, bos_id=1, tokenizer=tok, g_ext=1.0)
    # Force multi-hop gate ON (`mh_prob = 0.95 > 0.80`) and span gate strictly OFF (`span_rel_tau = 0.99 -> g_span = 0.0`)
    fomp.mh_fc2.bias.data.fill_(3.0)
    fomp.span_rel_tau.data.fill_(0.99)
    fomp.hop2_ll_tau_buf.fill_(-10.0)
    fomp.hop2_rel_tau_buf.fill_(-1.0)

    q_implicit = [1, 90, 91, 92, 93, 94, 95, 96]
    q_imp_t = torch.tensor([q_implicit])
    with torch.no_grad():
        base_lg = backbone(q_imp_t).logits
        res_h2 = wrap.forward_logits(q_imp_t, compiled=comp, mode="legible")
        res_h2_flip = wrap.forward_logits(q_imp_t, compiled=comp, mode="legible", sign_flip=True)
    assert res_h2["mode_used"] == "hop2_logit_lens" and res_h2["hop2_ep"] == 0 and res_h2["extra_kv_tokens"] == 0
    assert (res_h2["logits"] - base_lg).abs().max().item() > 1e-4
    assert (res_h2["logits"] - res_h2_flip["logits"]).abs().max().item() > 1e-4

    # Multi-token greedy decode reuses `forced_hop2_ep == 0` with `0` extra KV tokens!
    gen_h2, ex_h2 = wrap.greedy_generate(q_implicit, max_new_tokens=3, device=device, compiled=comp, mode="legible")
    assert len(gen_h2) == 3 and ex_h2 == 0 and wrap._last_routing["mode"] == "hop2_logit_lens"

    # Surface-mask check: if ALL salient candidate tokens of Episode 0 already appear in the prompt, Route 3 MUST block!
    cand_tids = [int(x) for x, m in zip(comp["latent_subj_tids"][0].tolist(), comp["latent_subj_mask"][0].tolist()) if m > 0]
    q_surface = [1, 90, 91] + cand_tids + [95, 96]
    with torch.no_grad():
        res_surf = wrap.forward_logits(torch.tensor([q_surface]), compiled=comp, mode="legible")
    assert res_surf["mode_used"] == "hop2_miss_clean" and res_surf["extra_kv_tokens"] == 0


def test_ike_4shot_formatter_and_hop_subtype_split() -> None:
    assert strip_qa_wrapper("Question: What university did Watts Humphrey attend?\nAnswer:") == "What university did Watts Humphrey attend"
    assert strip_cue_copula("The original broadcaster of The Westerner is") == "The original broadcaster of The Westerner"
    assert format_virtual_kv_fact("Question: Who founded Apple?\nAnswer:", " Steve Jobs") == "- Who founded Apple? Steve Jobs."
    assert format_ike_fact_line("Honda CBR is produced by", " Porsche") == "Honda CBR is produced by Porsche"
    assert _clause_to_embedded_np("The mother tongue of X is") == "the mother tongue of X"
    outer_r = EditRecord("cf", "France", "The capital city of France is", " Lyon", " Paris", [], [], "", "")
    assert _make_compositional_qa(outer_r, "the country of citizenship of Alan Turing") == (
        "Question: What is the capital city of the country of citizenship of Alan Turing?\nAnswer:"
    )

    cf_tr, _, _, _, _ = _make_smoke_datasets()
    tok = _SmokeTokenizer()
    fmt_h = IKEPromptFormatter(tok, cf_tr[:4], prepend_bos=False)
    ids_empty, sp_e, ex_e = fmt_h.encode_query(cf_tr[5].cue, [], "std")
    assert ex_e == 0 and not sp_e and len(ids_empty) > 0
    ids_std, sp_s, ex_s = fmt_h.encode_query(cf_tr[5].cue, ["Fact A is B"], "std")
    assert ex_s > 20 and not sp_s and len(ids_std) > len(ids_empty)

    c_ike = CleanDenseIKERetriever("smoke", device=torch.device("cpu"), smoke=True)
    tau, stats = c_ike.calibrate_clean_threshold(cf_tr, max_n=12)
    assert 0.55 <= tau <= 0.94 and "train_cross_qa_reject_pct" in stats

    recs = [
        EditRecord("mquake", "S1", "c1", " BBC", " NBC", ["q1"], [], " London", " NYC"),
        EditRecord("mquake", "S2", "c2", " Ottawa", " NYC", ["q2"], [], " Ottawa", " NYC"),
    ]
    fake_arms = {"a": {"track2_mq": {"mh_em": {"hits": [1, 0]}}}}
    sub = _compute_hop_subtype_breakdown(fake_arms, recs)
    assert sub["n_hop1_edited"] == 1 and sub["n_hop2_edited"] == 1
    assert sub["per_arm"]["a"]["hop1_edited_em"]["pct"] == 100.0
    assert sub["per_arm"]["a"]["hop2_edited_em"]["pct"] == 0.0


def test_stage32_smoke_cli_end_to_end(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks/fomp_v4_multihop_stage3.py"),
        "--smoke",
        "--out",
        str(tmp_path / "s32_smoke"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Smoke failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert (tmp_path / "s32_smoke/stage32_multihop_summary.json").exists()
    assert (tmp_path / "s32_smoke/stage32_multihop_report.md").exists()
    rep = (tmp_path / "s32_smoke/stage32_multihop_report.md").read_text(encoding="utf-8")
    assert "FOMP-v4.2 Pure-Activation 3-Route" in rep
