"""Unit tests for Stage 2.5 Peer-Review Rigor Audit (McNemar, Wilson CI, Dense IKE-RAG)."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from fomp_v3_open_edit import _make_smoke_datasets  # noqa: E402
from fomp_v36_rigor_audit import DenseIKERetriever, exact_mcnemar_p, wilson_ci95  # noqa: E402


def test_exact_mcnemar_and_wilson_ci() -> None:
    # 10 discordant wins for A, 0 for B -> exact two-sided p = 2 * (0.5 ** 10) = 0.001953125
    a = [1] * 10 + [1] * 5 + [0] * 5
    b = [0] * 10 + [1] * 5 + [0] * 5
    r = exact_mcnemar_p(a, b)
    assert r["n"] == 20 and r["n10_a_wins"] == 10 and r["n01_b_wins"] == 0
    assert abs(r["p_value"] - (2.0 * (0.5 ** 10))) < 1e-6
    assert r["sig_005"] is True

    # Identical predictions -> p = 1.0
    assert exact_mcnemar_p(a, a)["p_value"] == 1.0
    # Balanced discordance (5 vs 5) -> p = 1.0
    r_tie = exact_mcnemar_p([1] * 5 + [0] * 5, [0] * 5 + [1] * 5)
    assert abs(r_tie["p_value"] - 1.0) < 1e-9 and r_tie["sig_005"] is False

    pct, lo, hi = wilson_ci95([1] * 160 + [0] * 40)
    assert pct == 80.0 and 73.0 < lo < 80.0 < hi < 86.0


def test_dense_ike_retriever_gating_and_calibration() -> None:
    cf_tr, cf_te, _, _, _ = _make_smoke_datasets()
    ike = DenseIKERetriever("smoke-bge", device=torch.device("cpu"), smoke=True)
    tau, stats = ike.calibrate_threshold(cf_tr, max_n=16)
    assert 0.50 <= tau <= 0.94 and "balanced_score" in stats

    bank, sec = ike.build_bank(cf_te[:8])
    assert sec >= 0.0 and bank["keys"].shape == (8, 64)

    # Always-on returns Top-2 facts; impossible threshold 1.1 returns empty prefix (0 extra tokens!)
    pref_on, k_on = ike.retrieve_prefix(cf_te[0].cue, bank, top_k=2, threshold=None)
    assert k_on == 2 and cf_te[0].cue in pref_on
    pref_off, k_off = ike.retrieve_prefix("completely unrelated query xyz", bank, top_k=2, threshold=1.1)
    assert k_off == 0 and pref_off == ""
