"""Generate publication-grade figures directly from archived Stage 2.5 & Stage 3 H800 outputs."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import math

ROOT = Path(__file__).resolve().parents[1]


def wilson_ci_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return (round(max(0.0, center - half) * 100.0, 2), round(min(1.0, center + half) * 100.0, 2))

RIGOR_JSON = ROOT / "experiment_outputs/fomp_v36_release/rigor_audit_summary.json"
STAGE3_MD = ROOT / "experiment_outputs/fomp_v4_stage3/stage3_multihop_report.md"
STAGE3_JSON = ROOT / "experiment_outputs/fomp_v4_stage3/stage3_multihop_summary.json"
OUT_DIR = ROOT / "docs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _err(metric: dict) -> tuple[float, float, float]:
    val = float(metric["pct"])
    lo, hi = float(metric["ci95"][0]), float(metric["ci95"][1])
    return val, max(0.0, val - lo), max(0.0, hi - val)


def ensure_stage3_json() -> dict:
    """Build or load `stage3_multihop_summary.json` from `stage3_multihop_report.md`."""
    if STAGE3_JSON.exists() and STAGE3_JSON.stat().st_size > 100_000:
        return json.loads(STAGE3_JSON.read_text(encoding="utf-8"))

    md = STAGE3_MD.read_text(encoding="utf-8")
    name_to_key = {
        "Base (Unedited)": "base",
        "Oracle ICL (100% GT Fact in Prompt)": "oracle_icl",
        "IKE-RAG Always (bge-small Top-2)": "ike_rag_always",
        "IKE-RAG Gated (raw cf_train τ)": "ike_rag_gated",
        "IKE-RAG Gated Clean (QA-stripped + cf_qa τ)": "ike_rag_gated_clean",
        "AlphaEdit (Weight-Edit, ICLR'25)": "alphaedit",
        "FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)": "fomp_v36_zero_mq",
        "FOMP-v3.6 (+706 MQ in-domain)": "fomp_v36",
        "FOMP-v4 Legible (Route 1: Pure Act, 0% MQ/ZsRE)": "fomp_v4_legible",
        "FOMP-v4 Bridge-KV (Route 2: Pre-KV, 0% MQ/ZsRE)": "fomp_v4_bridge_kv",
    }

    def parse_ci(cell: str, n: int) -> dict:
        c = cell.replace("*", "").strip()
        m = re.match(r"([0-9.]+)%\s*\[([0-9.]+),\s*([0-9.]+)\]", c)
        if m:
            pct = float(m.group(1))
            return {"pct": pct, "ci95": [float(m.group(2)), float(m.group(3))], "n": n}
        m2 = re.match(r"([0-9.]+)%", c)
        assert m2 is not None, f"Cannot parse percentage cell: {cell}"
        pct = float(m2.group(1))
        k = int(round(pct * n / 100.0))
        return {"pct": pct, "ci95": list(wilson_ci_95(k, n)), "n": n}

    arms = {v: {"track1_cf": {}, "track2_mq": {}, "track3_zsre": {}} for v in name_to_key.values()}
    sections = md.split("## Table ")

    for line in sections[1].splitlines():
        if not line.startswith("| **"):
            continue
        cols = [x.strip() for x in line.strip("|").split("|")]
        k = name_to_key[cols[0].replace("**", "").strip()]
        arms[k]["track1_cf"] = {
            "edit_time_sec_per_batch": 0.0 if cols[1] == "—" else float(cols[1].rstrip("s")),
            "avg_extra_prompt_tokens": float(cols[2]),
            "eff_em": parse_ci(cols[3], 200),
            "par_em": parse_ci(cols[4], 200),
            "nei_prob": parse_ci(cols[5], 200),
            "ppl_drift": float(cols[6]),
            "ms_per_query": float(cols[7].replace("ms", "").strip()),
        }

    hop_sub: dict = {"n_hop1_edited": 127, "n_hop2_edited": 65, "per_arm": {}}
    for line in sections[2].splitlines():
        if not line.startswith("| **"):
            continue
        cols = [x.strip() for x in line.strip("|").split("|")]
        k = name_to_key[cols[0].replace("**", "").strip()]
        arms[k]["track2_mq"] = {
            "mq_train": cols[1].replace("**", "").strip(),
            "avg_extra_prompt_tokens": float(cols[2]),
            "sh_em": parse_ci(cols[3], 192),
            "mh_em": parse_ci(cols[4], 192),
        }
        hop_sub["per_arm"][k] = {
            "hop1_edited_em": parse_ci(cols[5], 127),
            "hop2_edited_em": parse_ci(cols[6], 65),
        }

    for line in sections[3].splitlines():
        if not line.startswith("| **"):
            continue
        cols = [x.strip() for x in line.strip("|").split("|")]
        k = name_to_key[cols[0].replace("**", "").strip()]
        loc_cell, tok_part = cols[5].split("(")
        arms[k]["track3_zsre"] = {
            "edit_time_sec_per_batch": 0.0 if cols[1] == "—" else float(cols[1].rstrip("s")),
            "avg_extra_prompt_tokens": float(cols[2]),
            "rel_em": parse_ci(cols[3], 200),
            "gen_em": parse_ci(cols[4], 200),
            "loc_exact": parse_ci(loc_cell, 200),
            "loc_tok_pct": float(tok_part.rstrip(")%")),
        }

    mcnemar = []
    for line in sections[4].splitlines():
        if not line.startswith("| `"):
            continue
        cols = [x.strip() for x in line.strip("|").split("|")]
        a, b = [x.strip("` ") for x in cols[0].split(" vs ")]
        pa = float(cols[2].rstrip("%"))
        pb = float(cols[3].rstrip("%"))
        diff_pp = float(cols[4])
        n10, n01 = [int(x.strip()) for x in cols[5].split("/")]
        mcnemar.append({
            "arm_a": a,
            "arm_b": b,
            "metric": cols[1],
            "pct_a": pa,
            "pct_b": pb,
            "diff_pp": diff_pp,
            "n10": n10,
            "n01": n01,
            "exact_p_two_sided": float(cols[6].strip("`")),
            "significant_005": cols[7] == "YES",
        })

    out = {
        "stage": "Stage 3 Multi-Hop & Clean-Baseline Audit (FOMP-v4 Legible + Bridge-KV + Clean RAG + Oracle ICL)",
        "hardware": "NVIDIA H800 PCIe (80GB)",
        "model_id": "Meta-Llama-3-8B-Instruct (BF16)",
        "zero_mquake_train_v4": True,
        "zero_zsre_train_all": True,
        "sample_sizes": {
            "counterfact": 200,
            "mquake": 192,
            "mquake_hop1_edited": 127,
            "mquake_hop2_edited": 65,
            "zsre": 200,
        },
        "arms": arms,
        "mquake_hop_subtypes": hop_sub,
        "paired_mcnemar_tests": mcnemar,
    }
    STAGE3_JSON.parent.mkdir(parents=True, exist_ok=True)
    STAGE3_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def plot_rigor_bars(data: dict) -> Path:
    arms = ["ike_rag_always", "ike_rag_gated", "alphaedit", "fomp_v36_zero_mq", "fomp_v36"]
    labels = [
        "IKE-RAG\nAlways\n(+31~49 tok)",
        "IKE-RAG\nGated (τ=.72)\n(+14~49 tok)",
        "AlphaEdit\n(ICLR'25)\n(160s/100)",
        "FOMP-v3.6\nZeroMQ\n(0% MQ/ZsRE)",
        "FOMP-v3.6\n(+706 MQ)\n(0% ZsRE)",
    ]
    colors_a = ["#94a3b8", "#64748b", "#f59e0b", "#0ea5e9", "#2563eb"]
    colors_b = ["#cbd5e1", "#94a3b8", "#fcd34d", "#7dd3fc", "#93c5fd"]

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 220,
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), constrained_layout=True)

    panels = [
        (
            "A. CounterFact (n = 200, Full Gen-EM)",
            "track1_cf",
            ("par_em", "Paraphrase EM"),
            ("nei_prob", "Neighborhood Retain"),
            "FOMP vs AlphaEdit Par-EM: +20.5~21.0 pp (p < 5e-7)",
        ),
        (
            "B. Zero-Shot ZsRE (n = 200, 0% ZsRE Train)",
            "track3_zsre",
            ("gen_em", "Generality EM"),
            ("loc_exact", "Locality Exact (6-tok)"),
            "ZeroMQ Locality 100% vs AE 51% (p=6e-30) & IKE 10.5% (p=3e-54)",
        ),
        (
            "C. MQuAKE-Remastered (n = 192, 3x64)",
            "track2_mq",
            ("sh_em", "Single-Hop EM"),
            ("mh_em", "2-Hop Multi-Hop EM"),
            "Stage 2.5 Gap Motivating Stage 3 (FOMP-v4 -> 25.0% / 28.7%)",
        ),
    ]

    x = np.arange(len(arms))
    w = 0.36
    for ax, (title, track, (k1, n1), (k2, n2), subtitle) in zip(axes, panels):
        v1, l1, h1 = zip(*[_err(data["arms"][a][track][k1]) for a in arms])
        v2, l2, h2 = zip(*[_err(data["arms"][a][track][k2]) for a in arms])

        b1 = ax.bar(x - w / 2, v1, w, yerr=[l1, h1], capsize=3, color=colors_a, edgecolor="#0f172a", linewidth=0.7, label=n1, error_kw={"elinewidth": 1.0, "ecolor": "#334155"})
        b2 = ax.bar(x + w / 2, v2, w, yerr=[l2, h2], capsize=3, color=colors_b, edgecolor="#0f172a", linewidth=0.7, hatch="//", label=n2, error_kw={"elinewidth": 1.0, "ecolor": "#334155"})

        for rect, val, hi in zip(b1, v1, h1):
            ax.text(rect.get_x() + rect.get_width() / 2, min(107.5, val + hi + 1.8), f"{val:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#0f172a")
        for rect, val, hi in zip(b2, v2, h2):
            ax.text(rect.get_x() + rect.get_width() / 2, min(107.5, val + hi + 1.8), f"{val:.1f}", ha="center", va="bottom", fontsize=7.4, color="#334155")

        ax.set_title(title, pad=14)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.6, color="#475569", style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.6)
        ax.set_ylim(0, 114)
        ax.set_ylabel("Accuracy (%) with Wilson 95% CI")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(frameon=False, loc="upper left", fontsize=8.0)

    fig.suptitle(
        "Stage 2.5 Baseline Rigor Audit on Llama-3-8B-Instruct (NVIDIA H800, 100% Full Greedy Gen-EM)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    out = OUT_DIR / "stage25_rigor_audit_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_stage3_audit(s3: dict) -> Path:
    """3-panel Stage 3 Multi-Hop & Clean-Baseline Audit figure."""
    plt.rcParams.update({
        "font.size": 9.2,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 220,
    })
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18.2, 5.3), constrained_layout=True, gridspec_kw={"width_ratios": [1.28, 1.0, 1.05]}
    )

    # Panel A: MQuAKE 2-Hop Overall vs Hop-1 Edited (n=127) & Hop-2 Edited (n=65), 0% MQuAKE Train!
    mh_arms = [
        "fomp_v36_zero_mq",
        "alphaedit",
        "fomp_v4_legible",
        "fomp_v4_bridge_kv",
        "ike_rag_gated_clean",
        "ike_rag_always",
        "oracle_icl",
    ]
    mh_labels = [
        "FOMP-v3.6\nZeroMQ\n(0 tok)",
        "AlphaEdit\n(ICLR'25)\n(0 tok)",
        "FOMP-v4\nLegible\n(0 tok)",
        "FOMP-v4\nBridge-KV\n(+9.2 tok)",
        "Clean Gated\nIKE-RAG\n(+28.1 tok)",
        "IKE-RAG\nAlways\n(+36.1 tok)",
        "Oracle ICL\n(100% GT\nPrompt)",
    ]
    x1 = np.arange(len(mh_arms))
    w1 = 0.27
    ov = [s3["arms"][a]["track2_mq"]["mh_em"]["pct"] for a in mh_arms]
    h1 = [s3["mquake_hop_subtypes"]["per_arm"][a]["hop1_edited_em"]["pct"] for a in mh_arms]
    h2 = [s3["mquake_hop_subtypes"]["per_arm"][a]["hop2_edited_em"]["pct"] for a in mh_arms]

    b_ov = ax1.bar(x1 - w1, ov, w1, color="#1d4ed8", edgecolor="#0f172a", linewidth=0.65, label="Overall 2-Hop EM (n=192)")
    b_h1 = ax1.bar(x1, h1, w1, color="#10b981", edgecolor="#0f172a", linewidth=0.65, label="Hop-1 Edited EM (n=127)")
    b_h2 = ax1.bar(x1 + w1, h2, w1, color="#f59e0b", edgecolor="#0f172a", linewidth=0.65, hatch="//", label="Hop-2 Edited EM (n=65)")
    for bars, fw in ((b_ov, "bold"), (b_h1, "bold"), (b_h2, "normal")):
        for r in bars:
            v = r.get_height()
            ax1.text(r.get_x() + r.get_width() / 2, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8, fontweight=fw, color="#0f172a")
    ax1.set_title("A. MQuAKE 2-Hop Breakdown (All 0% MQuAKE Train)", pad=14)
    ax1.text(0.5, 1.01, "Hop-1: v4-Legible 37.8% (=RAG) & Bridge-KV 42.5% (>RAG); Hop-2 exposes latent limit", transform=ax1.transAxes, ha="center", va="bottom", fontsize=7.5, color="#475569", style="italic")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(mh_labels, fontsize=6.9)
    ax1.set_ylim(0, 86)
    ax1.set_ylabel("Exact Match (%)")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(frameon=False, loc="upper left", fontsize=7.8)

    # Panel B: Clean Baselines vs FOMP on CF Paraphrase EM & Zero-Shot ZsRE Gen/Loc
    cb_arms = [
        "oracle_icl",
        "ike_rag_gated_clean",
        "alphaedit",
        "fomp_v36_zero_mq",
        "fomp_v4_bridge_kv",
    ]
    cb_labels = [
        "Oracle ICL\n(100% GT\nPrompt)",
        "Clean Gated\nIKE-RAG\n(QA-stripped)",
        "AlphaEdit\n(ICLR'25\nWeight-Edit)",
        "FOMP-v3.6\nZeroMQ\n(0 tok)",
        "FOMP-v4\nBridge-KV\n(+8~14 tok)",
    ]
    x2 = np.arange(len(cb_arms))
    w2 = 0.26
    cf_p = [s3["arms"][a]["track1_cf"]["par_em"]["pct"] for a in cb_arms]
    zs_g = [s3["arms"][a]["track3_zsre"]["gen_em"]["pct"] for a in cb_arms]
    zs_l = [s3["arms"][a]["track3_zsre"]["loc_exact"]["pct"] for a in cb_arms]

    c1 = ax2.bar(x2 - w2, cf_p, w2, color="#0284c7", edgecolor="#0f172a", linewidth=0.65, label="CF Paraphrase EM (n=200)")
    c2 = ax2.bar(x2, zs_g, w2, color="#6366f1", edgecolor="#0f172a", linewidth=0.65, label="Zero-Shot ZsRE Gen EM (n=200)")
    c3 = ax2.bar(x2 + w2, zs_l, w2, color="#94a3b8", edgecolor="#0f172a", linewidth=0.65, hatch="//", label="ZsRE Exact Locality (n=200)")
    for bars, fw in ((c1, "bold"), (c2, "bold"), (c3, "normal")):
        for r in bars:
            v = r.get_height()
            ax2.text(r.get_x() + r.get_width() / 2, v + 1.1, f"{v:.1f}", ha="center", va="bottom", fontsize=7.0, fontweight=fw, color="#0f172a")
    ax2.set_title("B. Clean Gated RAG & Oracle ICL vs. FOMP", pad=14)
    ax2.text(0.5, 1.01, "Even vs 100% Oracle ICL (60.0%) & Clean RAG (55.5%), FOMP wins CF Para by +20~28 pp (p<5e-6)", transform=ax2.transAxes, ha="center", va="bottom", fontsize=7.4, color="#475569", style="italic")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cb_labels, fontsize=7.5)
    ax2.set_ylim(0, 122)
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.88, edgecolor="none", loc="upper left", fontsize=7.5)

    # Panel C: 4-Track Zero-Shot Macro Average (CF Para + MQ 2-Hop + ZsRE Gen + ZsRE Loc) / 4
    rank_arms = [
        ("fomp_v4_bridge_kv", "FOMP-v4 Bridge-KV (Ours, 0% MQ/ZsRE)", "#059669"),
        ("fomp_v36_zero_mq", "FOMP-v3.6-ZeroMQ (Ours, 0 tok)", "#0284c7"),
        ("ike_rag_gated_clean", "IKE-RAG Gated Clean (Repaired)", "#475569"),
        ("fomp_v4_legible", "FOMP-v4 Legible (Ours, Pure Act 0 tok)", "#0ea5e9"),
        ("oracle_icl", "Oracle ICL (100% GT Fact in Prompt)", "#8b5cf6"),
        ("alphaedit", "AlphaEdit (ICLR'25 Weight-Edit)", "#d97706"),
        ("ike_rag_always", "IKE-RAG Always (bge-small Top-2)", "#94a3b8"),
    ]
    macros = []
    for k, lbl, col in rank_arms:
        m_val = 0.25 * (
            s3["arms"][k]["track1_cf"]["par_em"]["pct"]
            + s3["arms"][k]["track2_mq"]["mh_em"]["pct"]
            + s3["arms"][k]["track3_zsre"]["gen_em"]["pct"]
            + s3["arms"][k]["track3_zsre"]["loc_exact"]["pct"]
        )
        macros.append((lbl, m_val, col))
    y3 = np.arange(len(macros))[::-1]
    vals3 = [m[1] for m in macros]
    cols3 = [m[2] for m in macros]
    lbls3 = [m[0] for m in macros]
    hb = ax3.barh(y3, vals3, height=0.62, color=cols3, edgecolor="#0f172a", linewidth=0.7)
    for r, v in zip(hb, vals3):
        ax3.text(v + 1.0, r.get_y() + r.get_height() / 2, f"{v:.1f}%", va="center", ha="left", fontsize=8.3, fontweight="bold", color="#0f172a")
    ax3.set_yticks(y3)
    ax3.set_yticklabels(lbls3, fontsize=7.9)
    ax3.set_xlim(0, 88)
    ax3.set_xlabel("4-Metric Macro Average (%)  [CF Para + MQ 2-Hop + ZsRE Gen + ZsRE Loc]")
    ax3.set_title("C. Overall Zero-Leakage Macro Ranking", pad=14)
    ax3.text(0.5, 1.01, "Strictly 0% MQuAKE & 0% ZsRE training across all 7 methods", transform=ax3.transAxes, ha="center", va="bottom", fontsize=7.6, color="#475569", style="italic")
    ax3.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(
        "Stage 3 Multi-Hop & Clean-Baseline Audit on Llama-3-8B-Instruct (NVIDIA H800, 0% MQuAKE & 0% ZsRE Train)",
        fontsize=12.6,
        fontweight="bold",
        y=1.03,
    )
    out = OUT_DIR / "stage3_multihop_and_clean_audit.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pareto_and_speed(s3: dict) -> Path:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 220,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9), constrained_layout=True)

    meta = [
        ("ike_rag_always", "IKE-RAG Always\n(+49 tok)", "#94a3b8", "s", (12, -8)),
        ("ike_rag_gated_clean", "Clean Gated RAG\n(+20.5 tok)", "#475569", "D", (-128, 12)),
        ("alphaedit", "AlphaEdit (ICLR'25)\n(160s edit, 0 tok)", "#d97706", "^", (-55, -42)),
        ("fomp_v36_zero_mq", "FOMP-v3.6-ZeroMQ\n(0.09s, 0 tok)", "#0284c7", "o", (-145, 6)),
        ("fomp_v4_legible", "FOMP-v4 Legible\n(0.15s, 0 tok)", "#0ea5e9", "P", (-162, -52)),
        ("fomp_v4_bridge_kv", "FOMP-v4 Bridge-KV\n(2.2s, +9~14 tok)", "#059669", "*", (-162, -6)),
    ]

    for arm, label, color, marker, offset in meta:
        loc = s3["arms"][arm]["track3_zsre"]["loc_exact"]["pct"]
        mh = s3["arms"][arm]["track2_mq"]["mh_em"]["pct"]
        par = s3["arms"][arm]["track1_cf"]["par_em"]["pct"]
        sz = 250 if marker == "*" else 125
        ax1.scatter(loc, mh, s=sz, color=color, marker=marker, edgecolor="#0f172a", linewidth=0.9, zorder=5)
        ax1.annotate(
            f"{label}\n(CF Para {par:.1f}%)",
            (loc, mh),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.8,
            color="#0f172a",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.8},
        )

    ax1.annotate(
        "",
        xy=(100.0, 28.65),
        xytext=(100.0, 2.60),
        arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 2.0, "ls": "--"},
    )
    ax1.text(97.5, 14.5, "+26.1 pp\n(p=8e-13)", fontsize=7.8, fontweight="bold", color="#059669", ha="right")

    ax1.set_xlim(-2, 108)
    ax1.set_ylim(-2, 46)
    ax1.set_xlabel("Zero-Shot ZsRE Exact Locality (%)  [Higher = Safer]")
    ax1.set_ylabel("MQuAKE 2-Hop Multi-Hop EM (%)  [Higher = Deeper]")
    ax1.set_title("A. Locality vs. Multi-Hop Frontier (0% MQ & 0% ZsRE)")
    ax1.grid(True, linestyle="--", alpha=0.3)

    speed_offsets = {
        "ike_rag_always": (8, 4),
        "ike_rag_gated_clean": (8, -22),
        "alphaedit": (-118, 10),
        "fomp_v36_zero_mq": (10, 4),
        "fomp_v4_legible": (10, -24),
        "fomp_v4_bridge_kv": (10, -10),
    }
    for arm, label, color, marker, _ in meta:
        sec = s3["arms"][arm]["track1_cf"]["edit_time_sec_per_batch"]
        par = s3["arms"][arm]["track1_cf"]["par_em"]["pct"]
        sz = 250 if marker == "*" else 125
        ax2.scatter(sec, par, s=sz, color=color, marker=marker, edgecolor="#0f172a", linewidth=0.9, zorder=5)
        ax2.annotate(
            f"{label.splitlines()[0]}\n({sec:.2f}s, {par:.1f}%)",
            (sec, par),
            xytext=speed_offsets[arm],
            textcoords="offset points",
            fontsize=7.8,
            color="#0f172a",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.8},
        )
    ax2.set_xscale("log")
    ax2.set_xlim(0.008, 450)
    ax2.set_ylim(45, 94)
    ax2.set_xlabel("Edit / Index Compile Time for M = 100 Facts (seconds, log scale)")
    ax2.set_ylabel("CounterFact Paraphrase Gen-EM (%)")
    ax2.set_title("B. Edit Speed vs. Paraphrase Generalization (Llama-3-8B)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.25)

    out = OUT_DIR / "locality_multihop_pareto_and_speed.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    s25 = json.loads(RIGOR_JSON.read_text(encoding="utf-8"))
    s3 = ensure_stage3_json()
    p1 = plot_rigor_bars(s25)
    p2 = plot_stage3_audit(s3)
    p3 = plot_pareto_and_speed(s3)
    print(f"Generated:\n  {STAGE3_JSON}\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
