# Stage 2.5 Peer-Review Rigor Audit on `models/Meta-Llama-3-8B-Instruct` (`NVIDIA H800 PCIe`)

- **Protocol**: Disjoint multi-batch evaluation (`CF n=200` [2x100], `MQuAKE n=192` [3x64], `ZsRE n=200` [2x100]); **full greedy Gen-EM on 100% of items**, strictly 0% train/test entity overlap, **0% ZsRE train**, and dedicated **`fomp_v36_zero_mq` (0% MQuAKE train!)**.
- **Base Held-Out WikiText-2 PPL**: `9.3939`

## Table 1: CounterFact (`n = 200`, Full Gen-EM + Wilson 95% CI)

| Method (Paradigm) | Edit/Index s | Extra Prompt Tok | Efficacy EM | Paraphrase EM [95% CI] | Retain Prob [95% CI] | PPL Drift | Query ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Unedited)** | — | 0.0 | 0.5% | 0.0% [0.0, 1.88] | 88.5% [83.34, 92.21] | +0.0000 | 70.94 ms |
| **IKE-RAG Always (bge-small Top-2)** | 0.0153s | 31.33 | 86.0% | 57.5% [50.57, 64.15] | 73.0% [66.46, 78.68] | -0.0591 | 78.02 ms |
| **IKE-RAG Gated (cf_train τ)** | 0.0186s | 13.54 | 69.0% | 39.5% [32.98, 46.41] | 85.0% [79.39, 89.29] | +0.0000 | 77.97 ms |
| **AlphaEdit (Weight-Edit, ICLR'25)** | 159.91s | 0.0 | 100.0% | 63.0% [56.12, 69.39] | 85.5% [79.95, 89.71] | +0.0411 | 71.62 ms |
| **FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)** | 0.0856s | 0.0 | 100.0% | 83.5% [77.73, 88.0] | 86.5% [81.07, 90.55] | +0.0004 | 104.97 ms |
| **FOMP-v3.6 (with MQ train)** | 0.0761s | 0.0 | 100.0% | 84.0% [78.29, 88.43] | 85.0% [79.39, 89.29] | +0.0109 | 105.5 ms |

## Table 2: MQuAKE-Remastered Multi-Hop (`n = 192` Across 3 Disjoint Batches)

| Method | MQuAKE Train? | Edit/Index s | Extra Prompt Tok | Single-Hop EM [95% CI] | **2-Hop Multi-Hop EM [95% CI]** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base (Unedited)** | **0% (Zero-Shot)** | — | 0.0 | 0.0% [0.0, 1.96] | **0.52% [0.09, 2.89]** |
| **IKE-RAG Always (bge-small Top-2)** | **0% (Zero-Shot)** | 0.0086s | 36.09 | 56.77% [49.7, 63.58] | **38.02% [31.45, 45.06]** |
| **IKE-RAG Gated (cf_train τ)** | **0% (Zero-Shot)** | 0.0086s | 19.89 | 40.62% [33.93, 47.69] | **32.81% [26.57, 39.73]** |
| **AlphaEdit (Weight-Edit, ICLR'25)** | **0% (Zero-Shot)** | 104.38s | 0.0 | 100.0% [98.04, 100.0] | **17.71% [12.96, 23.73]** |
| **FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)** | **0% (Zero-Shot)** | 0.0519s | 0.0 | 93.75% [89.39, 96.39] | **2.6% [1.12, 5.95]** |
| **FOMP-v3.6 (with MQ train)** | 706 rows (in-domain) | 0.0516s | 0.0 | 89.06% [83.86, 92.73] | **15.1% [10.73, 20.85]** |

## Table 3: Zero-Shot ZsRE (`n = 200`, 0% ZsRE Training Data for ALL Methods!)

| Method | Edit/Index s | Extra Prompt Tok | Reliability EM [95% CI] | Generality EM [95% CI] | **Locality Exact [95% CI]** (Tok-Match) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base (Unedited)** | — | 0.0 | 1.5% [0.51, 4.32] | 1.0% [0.27, 3.57] | **100.0% [98.12, 100.0]** (100.0%) |
| **IKE-RAG Always (bge-small Top-2)** | 0.0181s | 49.36 | 78.5% [72.3, 83.63] | 77.5% [71.23, 82.74] | **7.5% [4.6, 12.0]** (22.0%) |
| **IKE-RAG Gated (cf_train τ)** | 0.0179s | 48.58 | 78.5% [72.3, 83.63] | 77.5% [71.23, 82.74] | **10.5% [6.97, 15.52]** (24.09%) |
| **AlphaEdit (Weight-Edit, ICLR'25)** | 162.825s | 0.0 | 98.0% [94.97, 99.22] | 83.0% [77.18, 87.57] | **51.0% [44.12, 57.84]** (63.33%) |
| **FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)** | 0.0984s | 0.0 | 99.5% [97.22, 99.91] | 95.0% [91.04, 97.26] | **100.0% [98.12, 100.0]** (100.0%) |
| **FOMP-v3.6 (with MQ train)** | 0.1033s | 0.0 | 98.5% [95.68, 99.49] | 98.0% [94.97, 99.22] | **92.5% [88.0, 95.4]** (93.66%) |

## Table 4: Exact Paired McNemar Significance Tests (Identical Per-Item Evaluation)

| Comparison (`A` vs `B`) | Track / Metric | `A` % | `B` % | Diff (pp) | `n10 / n01` | Exact McNemar `p` | `p < 0.05`? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fomp_v36_zero_mq` vs `alphaedit` | CF Paraphrase EM | 83.5% | 63.0% | +20.50 | 53 / 12 | `2.786e-07` | YES |
| `fomp_v36_zero_mq` vs `alphaedit` | CF Retain Prob | 86.5% | 85.5% | +1.00 | 6 / 4 | `0.7539` | no |
| `fomp_v36_zero_mq` vs `alphaedit` | MQuAKE Single-Hop EM | 93.75% | 100.0% | -6.25 | 0 / 12 | `0.0004883` | YES |
| `fomp_v36_zero_mq` vs `alphaedit` | MQuAKE 2-Hop EM | 2.6% | 17.71% | -15.10 | 1 / 30 | `2.98e-08` | YES |
| `fomp_v36_zero_mq` vs `alphaedit` | ZsRE Generality EM | 95.0% | 83.0% | +12.00 | 30 / 6 | `6.96e-05` | YES |
| `fomp_v36_zero_mq` vs `alphaedit` | ZsRE Locality Exact | 100.0% | 51.0% | +49.00 | 98 / 0 | `6.311e-30` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated` | CF Paraphrase EM | 83.5% | 39.5% | +44.00 | 101 / 13 | `4.774e-18` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated` | CF Retain Prob | 86.5% | 85.0% | +1.50 | 7 / 4 | `0.5488` | no |
| `fomp_v36_zero_mq` vs `ike_rag_gated` | MQuAKE Single-Hop EM | 93.75% | 40.62% | +53.12 | 107 / 5 | `5.416e-26` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated` | MQuAKE 2-Hop EM | 2.6% | 32.81% | -30.21 | 0 / 58 | `6.939e-18` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated` | ZsRE Generality EM | 95.0% | 77.5% | +17.50 | 42 / 7 | `3.625e-07` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated` | ZsRE Locality Exact | 100.0% | 10.5% | +89.50 | 179 / 0 | `2.61e-54` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_always` | CF Paraphrase EM | 83.5% | 57.5% | +26.00 | 71 / 19 | `3.127e-08` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_always` | CF Retain Prob | 86.5% | 73.0% | +13.50 | 36 / 9 | `6.575e-05` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_always` | MQuAKE Single-Hop EM | 93.75% | 56.77% | +36.98 | 78 / 7 | `2.796e-16` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_always` | MQuAKE 2-Hop EM | 2.6% | 38.02% | -35.42 | 1 / 69 | `1.203e-19` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_always` | ZsRE Generality EM | 95.0% | 77.5% | +17.50 | 42 / 7 | `3.625e-07` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_always` | ZsRE Locality Exact | 100.0% | 7.5% | +92.50 | 185 / 0 | `4.078e-56` | YES |
| `fomp_v36` vs `alphaedit` | CF Paraphrase EM | 84.0% | 63.0% | +21.00 | 56 / 14 | `4.303e-07` | YES |
| `fomp_v36` vs `alphaedit` | CF Retain Prob | 85.0% | 85.5% | -0.50 | 8 / 9 | `1.0` | no |
| `fomp_v36` vs `alphaedit` | MQuAKE Single-Hop EM | 89.06% | 100.0% | -10.94 | 0 / 21 | `9.537e-07` | YES |
| `fomp_v36` vs `alphaedit` | MQuAKE 2-Hop EM | 15.1% | 17.71% | -2.60 | 24 / 29 | `0.5831` | no |
| `fomp_v36` vs `alphaedit` | ZsRE Generality EM | 98.0% | 83.0% | +15.00 | 32 / 2 | `6.938e-08` | YES |
| `fomp_v36` vs `alphaedit` | ZsRE Locality Exact | 92.5% | 51.0% | +41.50 | 87 / 4 | `2.26e-21` | YES |
| `fomp_v36` vs `ike_rag_gated` | CF Paraphrase EM | 84.0% | 39.5% | +44.50 | 102 / 13 | `2.688e-18` | YES |
| `fomp_v36` vs `ike_rag_gated` | CF Retain Prob | 85.0% | 85.0% | +0.00 | 8 / 8 | `1.0` | no |
| `fomp_v36` vs `ike_rag_gated` | MQuAKE Single-Hop EM | 89.06% | 40.62% | +48.44 | 104 / 11 | `3.822e-20` | YES |
| `fomp_v36` vs `ike_rag_gated` | MQuAKE 2-Hop EM | 15.1% | 32.81% | -17.71 | 20 / 54 | `9.613e-05` | YES |
| `fomp_v36` vs `ike_rag_gated` | ZsRE Generality EM | 98.0% | 77.5% | +20.50 | 45 / 4 | `8.225e-10` | YES |
| `fomp_v36` vs `ike_rag_gated` | ZsRE Locality Exact | 92.5% | 10.5% | +82.00 | 166 / 2 | `7.589e-47` | YES |
| `fomp_v36` vs `ike_rag_always` | CF Paraphrase EM | 84.0% | 57.5% | +26.50 | 70 / 17 | `8.351e-09` | YES |
| `fomp_v36` vs `ike_rag_always` | CF Retain Prob | 85.0% | 73.0% | +12.00 | 37 / 13 | `0.0009362` | YES |
| `fomp_v36` vs `ike_rag_always` | MQuAKE Single-Hop EM | 89.06% | 56.77% | +32.29 | 75 / 13 | `9.326e-12` | YES |
| `fomp_v36` vs `ike_rag_always` | MQuAKE 2-Hop EM | 15.1% | 38.02% | -22.92 | 22 / 66 | `2.879e-06` | YES |
| `fomp_v36` vs `ike_rag_always` | ZsRE Generality EM | 98.0% | 77.5% | +20.50 | 45 / 4 | `8.225e-10` | YES |
| `fomp_v36` vs `ike_rag_always` | ZsRE Locality Exact | 92.5% | 7.5% | +85.00 | 171 / 1 | `5.78e-50` | YES |
| `fomp_v36` vs `fomp_v36_zero_mq` | CF Paraphrase EM | 84.0% | 83.5% | +0.50 | 14 / 13 | `1.0` | no |
| `fomp_v36` vs `fomp_v36_zero_mq` | CF Retain Prob | 85.0% | 86.5% | -1.50 | 2 / 5 | `0.4531` | no |
| `fomp_v36` vs `fomp_v36_zero_mq` | MQuAKE Single-Hop EM | 89.06% | 93.75% | -4.69 | 4 / 13 | `0.04904` | YES |
| `fomp_v36` vs `fomp_v36_zero_mq` | MQuAKE 2-Hop EM | 15.1% | 2.6% | +12.50 | 26 / 2 | `3.032e-06` | YES |
| `fomp_v36` vs `fomp_v36_zero_mq` | ZsRE Generality EM | 98.0% | 95.0% | +3.00 | 8 / 2 | `0.1094` | no |
| `fomp_v36` vs `fomp_v36_zero_mq` | ZsRE Locality Exact | 92.5% | 100.0% | -7.50 | 0 / 15 | `6.104e-05` | YES |
