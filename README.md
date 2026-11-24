# FOMP-v3.6 & PULSE — Instant Zero-Gradient Mid-Layer Plasticity for Frozen LLMs

**FOMP-v3.6** (*First-Order Mid-Layer Plasticity*) is a `20.88M`-parameter (**`0.26%`** overhead on an 8B model) plug-and-play fast-weight adapter that compiles **100 to 1,000+ multi-token factual edits into any frozen causal LLM in a single forward pass (`~0.08–0.10s` per 100 edits on H800)** — with **zero test-time gradients**, **zero base weight modification**, **zero extra prompt tokens**, and **exact `0.000000` drift when deactivated (`g_ext = 0`)**.

---

## Flagship Stage 2.5 Peer-Review Rigor Audit (`Llama-3-8B-Instruct`, NVIDIA H800)

Evaluated head-to-head on identical frozen BF16 `Meta-Llama-3-8B-Instruct` weights across **disjoint multi-batch splits** (**`n = 200` CounterFact [2×100]**, **`n = 192` MQuAKE [3×64]**, **`n = 200` Zero-Shot ZsRE [2×100]**) using **100% full greedy Gen-EM**, **Wilson 95% CIs**, and **exact paired McNemar $p$-values** against both unmodified official [`zjunlp/EasyEdit`](https://github.com/zjunlp/EasyEdit) **AlphaEdit** *(ICLR 2025 Outstanding Paper)* and same-paradigm dense retrieval **`IKE-RAG`** (`BAAI/bge-small-en-v1.5`, Always & `cf_train`-calibrated Gated `τ=0.72`):

| Method (`Llama-3-8B-Instruct`, H800) | Edit/Index s *(M=100)* | Extra Prompt Tok | CF Efficacy EM | **CF Paraphrase EM** *(n=200)* | CF Retain *(Base=88.5%)* | **Zero-Shot ZsRE Gen EM** *(n=200)* | **ZsRE Locality Exact** *(n=200)* | MQuAKE 2-Hop EM *(n=192)* | PPL Drift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base** *(unedited)* | — | `0.0` | 0.5% | 0.0% | 88.5% | 1.0% | 100.0% | 0.5% | `+0.0000` |
| **IKE-RAG Always** *(`bge-small` Top-2)* | `0.015 s` | `+31~49` | 86.0% | 57.5% | 73.0% | 77.5% | **7.5%** | **`38.0%`** | `-0.0591` |
| **IKE-RAG Gated** *(`cf_train τ=0.72`)* | `0.019 s` | `+14~49` | 69.0% | 39.5% | 85.0% | 77.5% | **10.5%** | `32.8%` | `+0.0000` |
| **AlphaEdit** *(ICLR'25, EasyEdit)* | `159.9 s` | `0.0` | **100.0%** | 63.0% | 85.5% | 83.0% | **51.0%** | `17.7%` | `+0.0411` |
| **FOMP-v3.6-ZeroMQ** *(`0%` MQ & `0%` ZsRE)* | **`0.086 s`** *(1,868× AE)* | **`0.0`** | **100.0%** | **`83.5%`** *(p=2.8e-7)* | **`86.5%`** | **`95.0%`** *(p=7.0e-5)* | **`100.0%`** *(p=6.3e-30)* | `2.6%` | **`+0.0004`** |
| **FOMP-v3.6** *(+706 MQ, `0%` ZsRE)* | **`0.076 s`** *(2,101× AE)* | **`0.0`** | **100.0%** | **`84.0%`** *(p=4.3e-7)* | `85.0%` | **`98.0%`** *(p=6.9e-8)* | **`92.5%`** *(p=2.3e-21)* | `15.1%` | `+0.0109` |

![Stage 2.5 Peer-Review Rigor Audit](docs/figures/stage25_rigor_audit_bars.png)

### Key Scientific Takeaways (Zero-Strawman Summary)
1. **Dominant Single-Hop, Paraphrase & Zero-Shot Format-Shift Locality**: Even with **0% MQuAKE and 0% ZsRE training data**, `FOMP-v3.6-ZeroMQ` compiles **`1,655×–1,868×` faster** than `AlphaEdit`, beats both `AlphaEdit` (`+20.5 pp`, `p=2.79e-7`) and `IKE-RAG` (`+26.0/+44.0 pp`, `p<=3.13e-8`) on CounterFact Paraphrase EM, beats both on Zero-Shot ZsRE Generality EM (`95.0%` vs `83.0%`/`77.5%`, `p<=6.96e-5`), and achieves **`100.0%` (`200/200`) exact ZsRE Locality** where `AlphaEdit` drops to `51.0%` (`p=6.31e-30`) and `IKE-RAG Gated` collapses to `10.5%` (`p=2.61e-54`) due to `"Question: ...\nAnswer:"` cosine inflation.
2. **Honest Multi-Hop Boundary (`MQuAKE n=192`)**: Explicit context-injection **`IKE-RAG`** decisively wins 2-hop multi-hop reasoning (`38.02%` Always / `32.81%` Gated vs `17.71%` `AlphaEdit`, `15.10%` in-domain `FOMP-v3.6`, and `2.60%` zero-shot `FOMP-v3.6-ZeroMQ`). Single-pass mid-layer activation editing excels at ultra-fast, zero-prompt-tax, surgical factual updates, while compositional multi-hop inference remains the domain of explicit in-context reasoning.
3. **High Concurrency & Causal Guardrails (Stage 2 A100)**: Scales to **`M = 1,000` concurrent edits in `1.30s`** (`100%` Eff, `84.4%` Para EM, `+0.0458` PPL), passes exact **`g_ext = 0` (`0.000000` drift)** and **Sign-Flip causal destruction (`0.0% / 0.0%`)** checks, and adds `~35–48%` forward query latency (`105 ms` vs `71–78 ms` on H800) with `0` extra prompt tokens.

![Locality vs Multi-Hop Pareto Frontier and Speedup](docs/figures/locality_multihop_pareto_and_speed.png)

---

## Cross-Architecture Generality (3 Distinct Frozen Backbone Families)

FOMP drops cleanly onto any causal LM architecture without modifying backbone weights:

| Backbone Family | Architecture | Concurrent Edits (`M`) | Adapter Overhead | Efficacy (Prob / EM) | Paraphrase (Prob / **EM**) | Retain *(vs Base)* | PPL Drift | Key Baseline Reference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Meta-Llama-3-8B-Instruct`** | `8B` GQA Transformer | `100` *(2×100, H800)* | `20.88M` (`0.26%`) | 100.0% / **100.0%** | 87.0% / **83.5%** *(ZeroMQ)* | **86.5%** *(Base 88.5%)* | `+0.0004` | `AlphaEdit` `63.0%` EM · `IKE-RAG` `57.5%` EM |
| **`Qwen/Qwen2.5-1.5B`** | `1.5B` Dense Transformer | `128` *(L4)* / `1,024` | `8.0M` (`0.52%`) | 100.0% / **100.0%** | 98.4% / **87.5%** (`90.6%` @1K) | **81.3%** *(Base 81.3%)* | `-0.0009` | `ICL (+2152 tok)` `56.3%` · `LoRA` `12.5%` (`+2.55` PPL) |
| **`PULSE-v8.1-125M`** | `122.5M` Hybrid (`SWA+GDN+MLA`) | `64` *(H800)* | `2.20M` (`1.80%`) | 100.0% / **100.0%** | 93.8% / **95.8%** | **60.9%** *(Base 60.9%)* | `-0.0033` | `TF+ICL (+1177 tok)` `16.7%` · `TF+LoRA` `6.3%` (`+3.30` PPL) |

*Note on `PULSE-v8.1-125M`*: Included as a matched-compute (`2.50B` FineWeb-Edu tokens, single scale/seed) open hybrid linear-attention base model (`19.60` vs `20.39` 2K val PPL vs matched `TF-Modern-125M`, `14.0 MiB` vs `576.0 MiB` 16K state) specifically to verify FOMP's architecture agnosticism beyond standard Transformers; see [`docs/FOMP_V36_TECHNICAL_REPORT.md`](docs/FOMP_V36_TECHNICAL_REPORT.md#b-hybrid-linear-attention-backbone--pulse-v81-125m-250b-fineweb-edu-nvidia-h800) for complete E1–E4 disclosures.

---

## 10-Line Quick Start (Instant Zero-Gradient Editing)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pulse import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module

ckpt = torch.load("checkpoints/fomp_v36_llama3_8b.pt", map_location="cuda", weights_only=False)
tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
backbone = AutoModelForCausalLM.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16).cuda().eval()

fomp = FOMPv3Module(FOMPv3Config(**ckpt["config"])).cuda().eval()
fomp.load_state_dict(ckpt["fomp"])
wrap = FOMPv3BackboneWrapper(backbone, fomp, freeze_backbone=True)

# Compile new facts in ~2 ms (1 forward pass, 0 backprop, 0 base weight changes!)
bos = [tok.bos_token_id]
bank = wrap.compile_token_episodes(
    [bos + tok.encode("Question: What university did Watts Humphrey attend?\nAnswer:", add_special_tokens=False)],
    [tok.encode(" University of Michigan", add_special_tokens=False)],
    pad_id=tok.eos_token_id, device=torch.device("cuda"), g_ext=1.0,
)
gen = wrap.greedy_generate(
    bos + tok.encode("Question: Which university was Watts Humphrey educated at?\nAnswer:", add_special_tokens=False),
    max_new_tokens=6, device=torch.device("cuda"), compiled=bank,
)
print(tok.decode(gen, skip_special_tokens=True))  # -> " University of Michigan"
```

---

## CLI Demo, Tests & Reproduction

```bash
# 1. Run the complete 9-test FOMP-v3.6 + Stage 2.5 statistics & invariance suite
PYTHONPATH=src:benchmarks pytest tests/test_fomp_v3.py tests/test_fomp_v36_rigor_audit.py -v

# 2. Try the interactive FOMP-v3.6 editing demo (zero-GPU --mock mode, or real Llama-3-8B on GPU)
PYTHONPATH=src python demo/fomp_v36_demo.py --mock
PYTHONPATH=src python demo/fomp_v36_demo.py --ckpt checkpoints/fomp_v36_llama3_8b.pt

# 3. Run the Stage 2.5 Rigor Audit smoke test locally on CPU (all 5 CPU-compatible arms + McNemar)
PYTHONPATH=src:benchmarks python benchmarks/fomp_v36_rigor_audit.py --smoke --out scratch_smoke

# 4. Regenerate publication figures directly from archived raw JSONs
python scripts/plot_release_figures.py
```

* **Full Technical Report & Complete Statistical Tables**: [`docs/FOMP_V36_TECHNICAL_REPORT.md`](docs/FOMP_V36_TECHNICAL_REPORT.md)
* **One-Click Colab Notebooks**: [`colab/FOMP_v36_Stage25_Rigor_Audit.ipynb`](colab/FOMP_v36_Stage25_Rigor_Audit.ipynb) · [`colab/FOMP_v35_Llama3_8B_SOTA_Showdown.ipynb`](colab/FOMP_v35_Llama3_8B_SOTA_Showdown.ipynb)
* **Pre-Trained 8B Adapter Checkpoint**: [Download `fomp_v36_llama3_8b.pt` (`v3.6` Release)](https://github.com/FelixLin789/PULSE-FOMP/releases/tag/v3.6) (`79.7 MB`, `20,878,481` params, `sha256=71b036e83ebbc62f...`)

---

## Repository Structure

```text
├── src/pulse/
│   ├── fomp_v3.py                               # FOMP-v3.6 core: Dual-Lock MLP, RMS-normed values, causal latching, wrapper
│   ├── pulse_v8.py / pulse_v7.py                # PULSE-v8.1 hybrid SWA + GatedDeltaNet + NoPE-MLA base model
│   └── ...                                      # Historical PULSE prototypes & matched Transformer baselines
├── checkpoints/
│   └── README.md                                # Download & SHA-256 verification for fomp_v36_llama3_8b.pt (Release v3.6)
├── demo/
│   └── fomp_v36_demo.py                         # Interactive FOMP-v3.6 live editor CLI (teach / ask / toggle / flip / list)
├── benchmarks/
│   ├── fomp_v36_rigor_audit.py                  # Stage 2.5 Peer-Review Rigor Audit (6 Arms, n=200/192/200, McNemar + Wilson CI)
│   ├── fomp_v35_llama3_8b_sota.py               # Stage 2 Llama-3-8B Showdown (vs EasyEdit AlphaEdit / MEMIT / GRACE / WISE)
│   ├── fomp_v34_stress_ood.py                   # Stage 1 Stress & OOD Suite (M=1,024, Zero-Shot ZsRE, Relation-Disjoint OOD)
│   ├── fomp_v3_open_edit.py                     # Qwen2.5-1.5B 4-Arm Benchmark & FOMP-v3.6 Trainer
│   └── pulse_v81_scaling_law.py                 # Phase 2 125M / 2.50B FineWeb-Edu Hybrid Base + E1–E4 FOMP Suite
├── scripts/
│   ├── plot_release_figures.py                  # Generates docs/figures/*.png strictly from archived raw JSONs
│   └── run_h800_stage25.sh                      # One-command AutoDL H800 Stage 2.5 runner
├── experiment_outputs/
│   ├── fomp_v36_release/                        # Raw Stage 2.5 (H800) & Stage 2 (A100) JSONs and Markdown reports
│   └── phase2_pulse_v81_125m_2p5b/              # Raw Phase 2 125M / 2.50B FineWeb-Edu JSON & report
├── colab/
│   ├── FOMP_v36_Stage25_Rigor_Audit.ipynb       # Stage 2.5 Colab A100/H100 reproduction notebook
│   └── FOMP_v35_Llama3_8B_SOTA_Showdown.ipynb   # Stage 2 Colab A100 reproduction notebook
├── docs/
│   ├── FOMP_V36_TECHNICAL_REPORT.md             # Full paper-grade report, math, Stage 2.5 tables, and honest limitations
│   └── figures/                                 # Publication figures generated from raw JSONs
└── tests/
    ├── test_fomp_v3.py                          # FOMP-v3.6 unit & mathematical invariance tests (7 tests)
    └── test_fomp_v36_rigor_audit.py             # Stage 2.5 McNemar, Wilson CI & IKE-RAG tests (2 tests)
```
