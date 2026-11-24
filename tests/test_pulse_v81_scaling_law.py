"""Unit & End-to-End Smoke Tests for Phase 2 Scaling Law (`PULSE-v8.1` vs `TF-Modern`)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from pulse.pulse_v8 import PULSEv8Config, PULSEv8Model
from pulse.whiteboard_anvil import TFModernConfig, TFModernLM, audit_zero_hidden_params

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from pulse_v81_scaling_law import state_mib_at_length  # noqa: E402


def test_125m_config_param_parity_and_40x_state_reduction() -> None:
    tf_cfg = TFModernConfig.scale_125m(vocab_size=50257)
    p_cfg = PULSEv8Config.scale_125m(vocab_size=50257, use_fomp=False)
    tf = TFModernLM(tf_cfg)
    p = PULSEv8Model(p_cfg)

    assert audit_zero_hidden_params(tf, tf_cfg.vocab_size, tf_cfg.d_model)["pass_zero_hidden_audit"]
    assert audit_zero_hidden_params(p, p_cfg.vocab_size, p_cfg.d_model)["pass_zero_hidden_audit"]
    assert p.num_parameters() <= tf.num_parameters(), (
        f"PULSE-v8.1-125M ({p.num_parameters():,}) must be <= TF-Modern-125M ({tf.num_parameters():,})"
    )
    s_tf_16k = state_mib_at_length(tf, 16384)
    s_p_16k = state_mib_at_length(p, 16384)
    assert s_tf_16k / s_p_16k >= 40.0, f"Expected >=40x 16K state reduction, got {s_tf_16k:.1f} vs {s_p_16k:.1f}"


def test_scaling_law_smoke_preflight_train_resume_and_all_suites(tmp_path: Path) -> None:
    out_dir = tmp_path / "run_smoke"
    data_dir = tmp_path / "data_smoke"
    script = ROOT / "benchmarks/pulse_v81_scaling_law.py"
    env = {"PYTHONPATH": str(ROOT / "src")}

    # 1. Preflight check
    r_pf = subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir), "--data-dir", str(data_dir), "--smoke", "--preflight"],
        capture_output=True, text=True, check=True, env={**__import__("os").environ, **env},
    )
    assert "PREFLIGHT PASS" in r_pf.stdout
    assert (out_dir / "preflight.json").exists()

    # 2. Full 2-arm smoke training + Suites E1, E2, E3, E4 + report
    subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir), "--data-dir", str(data_dir), "--smoke"],
        capture_output=True, text=True, check=True, env={**__import__("os").environ, **env},
    )
    summary_p = out_dir / "scaling_law_summary.json"
    report_p = out_dir / "scaling_law_report.md"
    assert summary_p.exists() and report_p.exists()
    data = json.loads(summary_p.read_text(encoding="utf-8"))
    assert data["train"]["pulse_v81"]["n_params"] <= data["train"]["tf_modern"]["n_params"]
    assert "e1_extrapolation" in data["eval"]
    assert "e2_zeroshot" in data["eval"]
    assert "e3_long_retrieval" in data["eval"]
    assert "e4_fomp_v32" in data["eval"]
    assert data["eval"]["e4_fomp_v32"]["pulse_fomp_v32"]["M8"]["g_ext0_ppl_drift"] == 0.0

    # 3. Resume idempotency check (re-running loads final checkpoints without retraining!)
    r_res = subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir), "--data-dir", str(data_dir), "--smoke", "--skip-fomp"],
        capture_output=True, text=True, check=True, env={**__import__("os").environ, **env},
    )
    assert "skipping pretraining" in r_res.stdout
