# PULSE & FOMP-v3.6 — Instant Zero-Gradient Knowledge Editing & Plastic Sequence Architecture

**FOMP-v3.6** (*First-Order Mid-Layer Plasticity*) is a `20.88M`-parameter (**`0.26%`** overhead) plug-and-play fast-weight adapter that compiles **100 to 1,000+ multi-token factual edits into any frozen causal LLM in a single forward pass** — with **zero test-time gradients**, **zero base weight modification**, **zero extra prompt tokens**, and **exact `0.000000` drift when switched off (`g_ext = 0`)**.

---

## Flagship Results on `Llama-3-8B-Instruct` (`NVIDIA A100`, Identical Backbone & Splits)

Head-to-head against unmodified official [`zjunlp/EasyEdit`](https://github.com/zjunlp/EasyEdit) baselines (**AlphaEdit** ICLR 2025 Outstanding Paper, **MEMIT**, **GRACE**) across **CounterFact (`M=100` & `M=1,000`)**, **MQuAKE-Remastered Multi-Hop (`M=64`)**, and **Zero-Shot ZsRE (`M=100`, `0%` ZsRE training data for FOMP!)**:

| Method (`Llama-3-8B-Instruct`) | Edit Time (`M=100`) | CF Efficacy<br>(Prob / EM) | CF Paraphrase<br>(Prob / **EM**) | CF Retain<br>*(Base=87.5%)* | MQuAKE 2-Hop<br>(Prob / **EM**) | Zero-Shot ZsRE<br>Rel / **Gen EM** | **ZsRE Locality**<br>*(Tok / Exact)* | PPL Drift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base** *(unedited)* | — | 12.5% / 3.1% | 9.4% / 0.0% | 87.5% | 21.9% / 3.1% | 1.6% / 1.6% | 100% / 100% | `0.0000` |
| **GRACE** *(EasyEdit, EM)* | 255.3 s | *90.6% EM* | 0.0% EM | **87.5%** | 21.9% / 3.1% | 39.1% / 1.6% | **100% / 100%** | `+0.0000` |
| **MEMIT** *(EasyEdit)* | 275.9 s | 95.3% / 93.8% | 87.5% / 50.0% | 85.9% | 31.3% / 9.4% | 96.9% / 73.4% | 70.1% / 57.8% | `+0.0094` |
| **AlphaEdit** *(ICLR'25)* | 285.2 s *(+479s `P`)* | **100% / 100%** | **92.2%** / 56.3% | 82.8% | **40.6%** / 12.5% | 98.4% / 82.8% | 66.7% / 53.1% | `+0.0185` |
| **FOMP-v3.6 (Ours, `M=100`)** | **`0.18 s`** *(**1,594×** faster)* | **100% / 100%** | 87.5% / **81.3%** | **85.9%** | 34.4% / **31.3%** *(**2.5×** AE)* | **100% / 95.3%** | **97.1% / 96.9%** | **`+0.0000`** |
| **FOMP-v3.6 (`M=1,000`)** | **`1.30 s`** | **100% / 100%** | 85.9% / **84.4%** | 79.7% | — | — | — | `+0.0458` |

* **Zero-Strawman Disclosure**: Official `WISE` (`llama3-8b.yaml`, `use_chat_template=False`) diverged (`+30.59` PPL on CF) and is excluded from main claims; `GRACE` is reported by `Gen-EM` due to `key_id=-1` evaluation. Full raw JSONs and causal Sign-Flip (`0.0% / 0.0%`) checks live in [`experiment_outputs/fomp_v36_release/`](experiment_outputs/fomp_v36_release/) and [`docs/FOMP_V36_TECHNICAL_REPORT.md`](docs/FOMP_V36_TECHNICAL_REPORT.md).

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

```powershell
# 1. Run the FOMP-v3.6 unit & invariance test suite (7/7 tests)
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m pytest tests/test_fomp_v3.py -v

# 2. Try the interactive FOMP-v3.6 editing demo (zero-GPU --mock mode, or real Llama-3-8B on GPU)
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe demo/fomp_v36_demo.py --mock
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe demo/fomp_v36_demo.py --ckpt checkpoints/fomp_v36_llama3_8b.pt

# 3. Run the Stage 2 Llama-3-8B benchmark smoke test locally on CPU
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe benchmarks/fomp_v35_llama3_8b_sota.py --smoke --out scratch_smoke
```

* **One-Click Colab A100 Notebook**: [`colab/FOMP_v35_Llama3_8B_SOTA_Showdown.ipynb`](colab/FOMP_v35_Llama3_8B_SOTA_Showdown.ipynb)
* **Full Technical Report & Math Derivations**: [`docs/FOMP_V36_TECHNICAL_REPORT.md`](docs/FOMP_V36_TECHNICAL_REPORT.md)
* **Pre-Trained 8B Adapter Checkpoint**: [Download `fomp_v36_llama3_8b.pt` (`v3.6` Release)](https://github.com/FelixLin789/PULSE-FOMP/releases/tag/v3.6) (`79.7 MB`, `sha256=71b036e83ebbc62f...`, see [`checkpoints/README.md`](checkpoints/README.md))

---

## Repository Structure

```text
├── src/pulse/
│   ├── fomp_v3.py                    # FOMP-v3.6 core: Dual-Lock MLP, RMS-normed values, causal latching, wrapper
│   ├── pulse_v8.py / pulse_v7.py     # PULSE-v8.1 hybrid GatedDeltaNet + SWA sequence backbone
│   └── ...                           # Earlier PULSE prototypes (v2–v6, local energy learning, baselines)
├── checkpoints/
│   └── fomp_v36_llama3_8b.pt         # Verified Llama-3-8B-Instruct FOMP-v3.6 adapter weights (20.88M params)
├── demo/
│   ├── fomp_v36_demo.py              # Interactive FOMP-v3.6 live editor CLI (teach / ask / toggle / flip / list)
│   └── pulse_cli.py                  # Historical PULSE prototype lab CLI
├── benchmarks/
│   ├── fomp_v35_llama3_8b_sota.py    # Stage 2 6-Arm Llama-3-8B SOTA Showdown (vs EasyEdit AlphaEdit/MEMIT/GRACE/WISE)
│   ├── fomp_v34_stress_ood.py        # Stage 1 Stress & OOD Suite (M=1,024, Zero-Shot ZsRE, Relation-Disjoint OOD)
│   ├── fomp_v3_open_edit.py          # Qwen2.5-1.5B 4-Arm Benchmark (Base / ICL / LoRA+Replay / FOMP) + Trainer
│   └── pulse_v81_scaling_law.py      # Phase 2 125M / 2.5B FineWeb-Edu Scaling Law & E1..E4 Suite
├── experiment_outputs/
│   └── fomp_v36_release/             # Archived raw JSON & Markdown reports from A100 & L4 GPU runs
├── colab/
│   └── FOMP_v35_Llama3_8B_SOTA_Showdown.ipynb  # Self-contained Colab A100 reproduction notebook
├── docs/
│   ├── FOMP_V36_TECHNICAL_REPORT.md  # Full FOMP-v3.6 paper-grade report, math, tables, and ablations
│   └── ARCHITECTURE.md               # Historical PULSE v1–v6 architecture notes
└── tests/
    └── test_fomp_v3.py               # Unit & mathematical invariance test suite for FOMP-v3.6
```
