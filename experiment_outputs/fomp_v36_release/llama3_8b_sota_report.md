# Stage 2 SOTA Showdown on `NousResearch/Meta-Llama-3-8B-Instruct` (`NVIDIA A100-SXM4-40GB`)

- **Protocol**: 100% identical backbone, identical BOS-consistent prompt tokenization, strictly 0% train/test entity overlap, **0% ZsRE training data for FOMP-v3.5**.
- **Base Held-Out WikiText-2 PPL**: `9.3951`

## Table 1: CounterFact (`M = 100` Concurrent Edits)

| Method | Edit Time | Efficacy (Prob / EM) | Paraphrase (Prob / EM) | Neighborhood Retain | WikiText-2 PPL Drift | Query ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASE** | — | 12.5% / 3.12% | 9.38% / 0.0% | 87.5% | 0.0000 | 107.88 ms |
| **GRACE** | 255.26s | 21.88% / 90.62% | 9.38% / 0.0% | 87.5% | +0.0000 | 115.69 ms |
| **WISE** | 111.6s | 12.5% / 3.12% | 9.38% / 0.0% | 87.5% | +30.5905 | 113.16 ms |
| **MEMIT** | 275.88s | 95.31% / 93.75% | 87.5% / 50.0% | 85.94% | +0.0094 | 114.29 ms |
| **ALPHAEDIT** | 285.24s | 100.0% / 100.0% | 92.19% / 56.25% | 82.81% | +0.0185 | 112.45 ms |
| **FOMP_V35** | 0.2806s | 100.0% / 100.0% | 93.75% / 87.5% | 85.94% | +0.0000 | 148.71 ms |
| **FOMP_V36** | 0.1789s | 100.0% / 100.0% | 87.5% / 81.25% | 85.94% | +0.0000 | 144.78 ms |

## Table 2: MQuAKE-Remastered Multi-Hop Reasoning

| Method | Edit Time | Single-Hop Efficacy (Prob / EM) | **2-Hop Multi-Hop (Prob / EM)** |
| :--- | :---: | :---: | :---: |
| **BASE** | — | 9.38% / 0.0% | **21.88% / 3.12%** |
| **GRACE** | 162.67s | 28.12% / 40.62% | **21.88% / 3.12%** |
| **WISE** | 103.01s | 12.5% / 0.0% | **21.88% / 3.12%** |
| **MEMIT** | 176.16s | 87.5% / 84.38% | **31.25% / 9.38%** |
| **ALPHAEDIT** | 179.41s | 100.0% / 96.88% | **40.62% / 12.5%** |
| **FOMP_V35** | 0.0942s | 90.62% / 31.25% | **28.12% / 15.62%** |
| **FOMP_V36** | 0.0939s | 100.0% / 87.5% | **34.38% / 31.25%** |

## Table 3: Zero-Shot ZsRE (`M = 100` Edits — 0% ZsRE Training Data for FOMP-v3.5!)

| Method | Edit Time | Reliability (Prob / EM) | Generality (Prob / EM) | **Locality (Tok-Match / Exact)** | WikiText-2 PPL Drift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BASE** | — | 25.0% / 1.56% | 25.0% / 1.56% | **100.0% / 100.0%** | 0.0000 |
| **GRACE** | 253.67s | 40.62% / 39.06% | 25.0% / 1.56% | **100.0% / 100.0%** | +0.0000 |
| **WISE** | 154.45s | 40.62% / 15.62% | 29.69% / 4.69% | **95.83% / 95.31%** | +2.3062 |
| **MEMIT** | 280.01s | 98.44% / 96.88% | 95.31% / 73.44% | **70.05% / 57.81%** | -0.0005 |
| **ALPHAEDIT** | 281.18s | 100.0% / 98.44% | 96.88% / 82.81% | **66.67% / 53.12%** | +0.0264 |
| **FOMP_V35** | 0.181s | 100.0% / 23.44% | 96.88% / 21.88% | **82.81% / 79.69%** | +0.0000 |
| **FOMP_V36** | 0.1792s | 100.0% / 100.0% | 98.44% / 95.31% | **97.14% / 96.88%** | +0.0007 |

## Table 4: FOMP-v3.6 High-Concurrency Scaling Wall (`M = 500, 1000`)

| Concurrency `M` | Compile Time | Efficacy (Prob / EM) | Paraphrase (Prob / EM) | Neighborhood Retain | PPL Drift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `M500` | 0.6524s | 100.0% / 100.0% | 85.94% / 81.25% | 82.81% | +0.0571 |
| `M1000` | 1.2977s | 100.0% / 100.0% | 85.94% / 84.38% | 79.69% | +0.0458 |
