"""Generate publication-grade figures directly from archived raw Stage 2.5 JSONs."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RIGOR_JSON = ROOT / "experiment_outputs/fomp_v36_release/rigor_audit_summary.json"
OUT_DIR = ROOT / "docs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _err(metric: dict) -> tuple[float, float, float]:
    val = float(metric["pct"])
    lo, hi = float(metric["ci95"][0]), float(metric["ci95"][1])
    return val, max(0.0, val - lo), max(0.0, hi - val)


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
            "Honest Boundary: IKE-RAG dominates 2-Hop (38.0% vs 15.1% / 2.6%)",
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
        "Stage 2.5 Peer-Review Rigor Audit on Llama-3-8B-Instruct (NVIDIA H800, 100% Full Greedy Gen-EM)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    out = OUT_DIR / "stage25_rigor_audit_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pareto_and_speed(data: dict) -> Path:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 220,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    meta = [
        ("ike_rag_always", "IKE-RAG Always\n(+49 tok)", "#64748b", "s", (16, 4)),
        ("ike_rag_gated", "IKE-RAG Gated (τ=0.72)\n(+49 tok)", "#475569", "D", (16, -34)),
        ("alphaedit", "AlphaEdit (ICLR'25)\n(160s edit, 0 tok)", "#d97706", "^", (-55, -52)),
        ("fomp_v36", "FOMP-v3.6 (+706 MQ)\n(0.10s, 0 tok)", "#2563eb", "o", (-88, 36)),
        ("fomp_v36_zero_mq", "FOMP-v3.6-ZeroMQ\n(0% MQ/ZsRE, 0.10s)", "#0284c7", "*", (-132, 14)),
    ]

    # Panel A: Locality vs Multi-Hop Pareto Frontier
    for arm, label, color, marker, offset in meta:
        loc = data["arms"][arm]["track3_zsre"]["loc_exact"]["pct"]
        mh = data["arms"][arm]["track2_mq"]["mh_em"]["pct"]
        gen = data["arms"][arm]["track3_zsre"]["gen_em"]["pct"]
        sz = 230 if marker == "*" else 120
        ax1.scatter(loc, mh, s=sz, color=color, marker=marker, edgecolor="#0f172a", linewidth=0.9, zorder=5)
        ax1.annotate(
            f"{label}\n(ZsRE Gen {gen:.1f}%)",
            (loc, mh),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.0,
            color="#0f172a",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.8},
        )

    # Draw Pareto frontier connecting dominant points
    p_x = [data["arms"][a]["track3_zsre"]["loc_exact"]["pct"] for a in ("ike_rag_always", "alphaedit", "fomp_v36", "fomp_v36_zero_mq")]
    p_y = [data["arms"][a]["track2_mq"]["mh_em"]["pct"] for a in ("ike_rag_always", "alphaedit", "fomp_v36", "fomp_v36_zero_mq")]
    ax1.plot(p_x, p_y, linestyle="--", color="#94a3b8", linewidth=1.2, zorder=2, label="Empirical Locality vs 2-Hop Frontier")
    ax1.set_xlim(-2, 108)
    ax1.set_ylim(-2, 46)
    ax1.set_xlabel("Zero-Shot ZsRE Exact Locality (%)  [Higher = Safer]")
    ax1.set_ylabel("MQuAKE 2-Hop Multi-Hop EM (%)  [Higher = Deeper]")
    ax1.set_title("A. Locality vs. Multi-Hop Pareto Frontier (Llama-3-8B)")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(frameon=False, loc="lower left", fontsize=8.0)

    # Panel B: Speed (log10 s) vs CounterFact Paraphrase EM
    speed_offsets = {
        "ike_rag_always": (8, -4),
        "ike_rag_gated": (8, -6),
        "alphaedit": (-118, 12),
        "fomp_v36": (10, -2),
        "fomp_v36_zero_mq": (10, -24),
    }
    for arm, label, color, marker, _ in meta:
        sec = data["arms"][arm]["track1_cf"]["edit_time_sec_per_batch"]
        par = data["arms"][arm]["track1_cf"]["par_em"]["pct"]
        sz = 230 if marker == "*" else 120
        ax2.scatter(sec, par, s=sz, color=color, marker=marker, edgecolor="#0f172a", linewidth=0.9, zorder=5)
        ax2.annotate(
            f"{label.splitlines()[0]}\n({sec:.2f}s, {par:.1f}%)",
            (sec, par),
            xytext=speed_offsets[arm],
            textcoords="offset points",
            fontsize=8.0,
            color="#0f172a",
            arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.8},
        )
    ax2.set_xscale("log")
    ax2.set_xlim(0.008, 450)
    ax2.set_ylim(30, 96)
    ax2.set_xlabel("Edit / Index Compile Time for M = 100 Facts (seconds, log scale)")
    ax2.set_ylabel("CounterFact Paraphrase Gen-EM (%)")
    ax2.set_title("B. Edit Speed vs. Paraphrase Generalization (~2,000x Speedup)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.25)

    out = OUT_DIR / "locality_multihop_pareto_and_speed.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    data = json.loads(RIGOR_JSON.read_text(encoding="utf-8"))
    p1 = plot_rigor_bars(data)
    p2 = plot_pareto_and_speed(data)
    print(f"Generated:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
