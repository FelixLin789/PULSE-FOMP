"""Unit & invariance tests for FOMP-v4 (`Legible` Read-Equivalent Bridge + `Bridge-KV` Virtual-KV) & Stage 3."""
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

from pulse.fomp_v4 import (  # noqa: E402
    FOMPv4BackboneWrapper,
    FOMPv4Config,
    FOMPv4Module,
    format_virtual_kv_fact,
    strip_cue_copula,
)
from fomp_v4_multihop_stage3 import (  # noqa: E402
    CleanDenseIKERetriever,
    _clause_to_embedded_np,
    _compute_hop_subtype_breakdown,
    _make_compositional_qa,
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


def test_fomp_v4_zero_drift_and_dynamic_cache_virtual_kv() -> None:
    device = torch.device("cpu")
    backbone = _tiny_llama(n_layers=8, d=64)
    v4_cfg = FOMPv4Config.for_backbone(64, 8)
    fomp = FOMPv4Module(v4_cfg).eval()
    wrap = FOMPv4BackboneWrapper(backbone, fomp, freeze_backbone=True)
    tok = _SmokeTokenizer()

    cue_ids = [[1, 10, 11, 12, 13], [1, 20, 21, 22, 23]]
    tgt_ids = [[30, 31], [40, 41, 42]]
    facts = ["- EntityA is produced by NewCorp.", "- OrgB was founded in NewCity."]

    # 1. Exact 0.000000 drift when `g_ext == 0.0` in BOTH modes
    comp_off = wrap.compile_episodes(
        cue_ids, tgt_ids, pad_id=0, device=device, bos_id=1,
        fact_texts=facts, tokenizer=tok, g_ext=0.0, precompile_kv=True,
    )
    q = torch.tensor([[1, 50, 51, 52, 53, 54, 55, 56]])
    with torch.no_grad():
        base_lg = backbone(q).logits
        lg_leg = wrap.forward_logits(q, compiled=comp_off, mode="legible")["logits"]
        lg_bkv = wrap.forward_logits(q, compiled=comp_off, mode="bridge_kv")["logits"]
    assert (base_lg - lg_leg).abs().max().item() == 0.0
    assert (base_lg - lg_bkv).abs().max().item() == 0.0

    # 2. Unrelated query (`all gates == 0`) uses 0 extra KV tokens and preserves exact Base logits!
    comp_on = wrap.compile_episodes(
        cue_ids, tgt_ids, pad_id=0, device=device, bos_id=1,
        fact_texts=facts, tokenizer=tok, g_ext=1.0, precompile_kv=True,
    )
    assert comp_on["kv_snaps"][0] is not None, "Expected pre-compiled DynamicCache snapshot on LlamaForCausalLM"
    # Calibrated thresholds above untrained BOS anisotropy ensure unrelated query gates are strictly 0
    fomp.span_rel_tau.data.fill_(0.85)
    fomp.lat_ent_tau.data.fill_(0.85)
    fomp.direct.fam_tau.data.fill_(0.85)
    unrel_ids = [1, 90, 91, 92, 93, 94, 95, 96, 97]
    gen_u, ex_u = wrap.greedy_generate(unrel_ids, max_new_tokens=3, device=device, compiled=comp_on, mode="bridge_kv")
    assert ex_u == 0 and len(gen_u) == 3

    # 3. Verify `_run_with_virtual_kv_ids` matches full prefix+query concatenation via `DynamicCache`
    with torch.no_grad():
        lg_vkv, ex_tok = wrap._run_with_virtual_kv_ids(unrel_ids, [0], comp_on, device, sign_flip=False)
        full_concat = torch.tensor([comp_on["pref_full_ids"][0] + unrel_ids[1:]])
        lg_ref = backbone(full_concat).logits[:, -len(unrel_ids[1:]) :, :]
    assert ex_tok > 0
    assert (lg_vkv - lg_ref).abs().max().item() < 1e-5


def test_fomp_v4_stage_a_legible_replacement_and_multitoken_direct_guard() -> None:
    device = torch.device("cpu")
    backbone = _tiny_llama(n_layers=8, d=64)
    v4_cfg = FOMPv4Config.for_backbone(64, 8)
    # Lower span thresholds so embedded cue triggers Stage A deterministically when Direct is off
    v4_cfg.init_span_rel_tau = 0.20
    v4_cfg.min_span_rel_tau = 0.15
    v4_cfg.init_span_ent_tau = 0.20
    v4_cfg.min_span_ent_tau = 0.15
    fomp = FOMPv4Module(v4_cfg).eval()
    wrap = FOMPv4BackboneWrapper(backbone, fomp, freeze_backbone=True)

    cue = [1, 14, 15, 16, 17]
    tgt = [61, 62]
    comp = wrap.compile_episodes([cue], [tgt], pad_id=0, device=device, bos_id=1, g_ext=1.0)

    # 1. Selective Direct (`fam_tau = 0.85`): exact `cue` has `cos=1.0 > 0.85`, while `mh_q` has `g_dir_end == 0`
    # and exact embedded span match at `t=4` (`cos=1.0`) -> routes to `stage_a_legible`!
    fomp.direct.fam_tau.data.fill_(0.85)
    mh_q = torch.tensor([[1, 14, 15, 16, 17, 80, 81, 82, 83]])
    with torch.no_grad():
        base_lg = backbone(mh_q).logits
        res_a = wrap.forward_logits(mh_q, compiled=comp, mode="legible")
        res_flip = wrap.forward_logits(mh_q, compiled=comp, mode="legible", sign_flip=True)
    assert res_a["mode_used"] == "stage_a_legible"
    assert (res_a["logits"] - base_lg).abs().max().item() > 1e-4
    assert (res_a["logits"] - res_flip["logits"]).abs().max().item() > 1e-4

    # 2. Nested-NP Arbitration (`fam_tau = 0.20` so Direct tail window tries to fire, while `s_outer_ms` on `mh_q`
    # `< outer_direct_tau = 0.90` whereas exact `cue` has `s_outer_ms == 1.00 >= 0.90`): Stage A overrides Direct on `mh_q`!
    fomp.direct.fam_tau.data.fill_(0.20)
    fomp.direct.ent_tau.data.fill_(0.20)
    fomp.outer_direct_tau.data.fill_(0.90)
    with torch.no_grad():
        res_override = wrap.forward_logits(mh_q, compiled=comp, mode="legible")
    assert res_override["mode_used"] == "stage_a_legible"

    # 3. Direct single-hop query (`cue` itself) is tail-guarded (`t >= L - 3`) and stays strictly in `direct_or_stage_c`!
    gen_d, ex_d = wrap.greedy_generate(cue, max_new_tokens=3, device=device, compiled=comp, mode="legible")
    assert len(gen_d) == 3 and ex_d == 0
    assert wrap._last_routing["mode"] == "direct_or_stage_c"


def test_clean_ike_wrapper_stripping_and_hop_subtype_split() -> None:
    assert strip_qa_wrapper("Question: What university did Watts Humphrey attend?\nAnswer:") == "What university did Watts Humphrey attend"
    assert strip_cue_copula("The original broadcaster of The Westerner is") == "The original broadcaster of The Westerner"
    assert format_virtual_kv_fact("Question: Who founded Apple?\nAnswer:", " Steve Jobs") == "- Who founded Apple? Steve Jobs."
    assert _clause_to_embedded_np("The mother tongue of X is") == "the mother tongue of X"
    assert _clause_to_embedded_np("Honda CBR is produced by").startswith("what ")
    outer_r = EditRecord("cf", "France", "The capital city of France is", " Lyon", " Paris", [], [], "", "")
    assert _make_compositional_qa(outer_r, "the country of citizenship of Alan Turing") == (
        "Question: What is the capital city of the country of citizenship of Alan Turing?\nAnswer:"
    )

    cf_tr, _, _, _, _ = _make_smoke_datasets()
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


def test_stage3_smoke_cli_end_to_end(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "benchmarks/fomp_v4_multihop_stage3.py"),
        "--smoke",
        "--out",
        str(tmp_path / "s3_smoke"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"Smoke failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert (tmp_path / "s3_smoke/stage3_multihop_summary.json").exists()
    assert (tmp_path / "s3_smoke/stage3_multihop_report.md").exists()
