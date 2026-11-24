# Stage 3 Multi-Hop & Clean-Baseline Audit on `models/Meta-Llama-3-8B-Instruct` (`NVIDIA H800 PCIe`)

- **Zero-Leakage Guarantee**: Both **`FOMP-v4 Legible` (Route 1)** and **`FOMP-v4 Bridge-KV` (Route 2)** are trained with **strictly `0%` MQuAKE and `0%` ZsRE** (`cf_train` self-supervised only).
- **Disjoint Sample Sizes**: `CF n=200`, `MQuAKE n=192` (`Hop-1 edited = 127`, `Hop-2 edited = 65`), `ZsRE n=200`.

## Table 1: CounterFact (`n = 200`, Full Greedy Gen-EM)

| Method | Edit/Index s | Extra Tok | Eff EM | Paraphrase EM [95% CI] | Retain Prob [95% CI] | PPL Drift | Query ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Unedited)** | — | 0.0 | 0.5% | 0.0% [0.0, 1.88] | 88.5% [83.34, 92.21] | +0.0000 | 70.94 ms |
| **Oracle ICL (100% GT Fact in Prompt)** | — | 20.61 | 69.5% | 60.0% [53.08, 66.54] | 62.0% [55.11, 68.44] | +0.0000 | 74.34 ms |
| **IKE-RAG Always (bge-small Top-2)** | 0.0153s | 31.33 | 86.0% | 57.5% [50.57, 64.15] | 73.0% [66.46, 78.68] | -0.0591 | 78.02 ms |
| **IKE-RAG Gated (raw cf_train τ)** | 0.0186s | 13.54 | 69.0% | 39.5% [32.98, 46.41] | 85.0% [79.39, 89.29] | +0.0000 | 77.97 ms |
| **IKE-RAG Gated Clean (QA-stripped + cf_qa τ)** | 0.0153s | 16.98 | 73.0% | 55.5% [48.57, 62.22] | 84.0% [78.29, 88.43] | +0.0000 | 79.47 ms |
| **AlphaEdit (Weight-Edit, ICLR'25)** | 159.91s | 0.0 | 100.0% | 63.0% [56.12, 69.39] | 85.5% [79.95, 89.71] | +0.0411 | 71.62 ms |
| **FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)** | 0.0856s | 0.0 | 100.0% | 83.5% [77.73, 88.0] | 86.5% [81.07, 90.55] | +0.0004 | 104.97 ms |
| **FOMP-v3.6 (+706 MQ in-domain)** | 0.0761s | 0.0 | 100.0% | 84.0% [78.29, 88.43] | 85.0% [79.39, 89.29] | +0.0109 | 105.5 ms |
| **FOMP-v4 Legible (Route 1: Pure Act, 0% MQ/ZsRE)** | 0.1457s | 0.0 | 83.0% | 76.5% [70.16, 81.84] | 87.5% [82.2, 91.39] | +0.1212 | 151.5 ms |
| **FOMP-v4 Bridge-KV (Route 2: Pre-KV, 0% MQ/ZsRE)** | 2.2329s | 8.4 | 83.5% | 80.0% [73.91, 84.95] | 88.0% [82.77, 91.8] | -0.0468 | 182.12 ms |

## Table 2: MQuAKE-Remastered Multi-Hop (`n = 192`, Overall + Hop-1 vs Hop-2 Breakdown)

| Method | MQ Train | Extra Tok | 1-Hop EM | **2-Hop Overall EM [95% CI]** | Hop-1 Edited EM | Hop-2 Edited EM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base (Unedited)** | **0%** | 0.0 | 0.0% | **0.52% [0.09, 2.89]** | 0.79% | 0.0% |
| **Oracle ICL (100% GT Fact in Prompt)** | **0%** | 23.27 | 36.98% | **59.38% [52.31, 66.07]** | 51.97% | 73.85% |
| **IKE-RAG Always (bge-small Top-2)** | **0%** | 36.09 | 56.77% | **38.02% [31.45, 45.06]** | 37.8% | 38.46% |
| **IKE-RAG Gated (raw cf_train τ)** | **0%** | 19.89 | 40.62% | **32.81% [26.57, 39.73]** | 36.22% | 26.15% |
| **IKE-RAG Gated Clean (QA-stripped + cf_qa τ)** | **0%** | 28.1 | 50.52% | **33.33% [27.05, 40.27]** | 31.5% | 36.92% |
| **AlphaEdit (Weight-Edit, ICLR'25)** | **0%** | 0.0 | 100.0% | **17.71% [12.96, 23.73]** | 22.05% | 9.23% |
| **FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)** | **0%** | 0.0 | 93.75% | **2.6% [1.12, 5.95]** | 2.36% | 3.08% |
| **FOMP-v3.6 (+706 MQ in-domain)** | 706 rows | 0.0 | 89.06% | **15.1% [10.73, 20.85]** | 14.96% | 15.38% |
| **FOMP-v4 Legible (Route 1: Pure Act, 0% MQ/ZsRE)** | **0%** | 0.0 | 83.33% | **25.0% [19.41, 31.57]** | 37.8% | 0.0% |
| **FOMP-v4 Bridge-KV (Route 2: Pre-KV, 0% MQ/ZsRE)** | **0%** | 9.21 | 83.33% | **28.65% [22.72, 35.41]** | 42.52% | 1.54% |

## Table 3: Zero-Shot ZsRE (`n = 200`, 0% ZsRE Train for ALL Methods!)

| Method | Edit/Index s | Extra Tok | Rel EM [95% CI] | **Gen EM [95% CI]** | **Locality Exact [95% CI]** (Tok) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base (Unedited)** | — | 0.0 | 1.5% [0.51, 4.32] | **1.0% [0.27, 3.57]** | **100.0% [98.12, 100.0]** (100.0%) |
| **Oracle ICL (100% GT Fact in Prompt)** | — | 25.73 | 92.5% [88.0, 95.4] | **86.5% [81.07, 90.55]** | **14.5% [10.29, 20.05]** (30.16%) |
| **IKE-RAG Always (bge-small Top-2)** | 0.0181s | 49.36 | 78.5% [72.3, 83.63] | **77.5% [71.23, 82.74]** | **7.5% [4.6, 12.0]** (22.0%) |
| **IKE-RAG Gated (raw cf_train τ)** | 0.0179s | 48.58 | 78.5% [72.3, 83.63] | **77.5% [71.23, 82.74]** | **10.5% [6.97, 15.52]** (24.09%) |
| **IKE-RAG Gated Clean (QA-stripped + cf_qa τ)** | 0.0187s | 20.5 | 92.5% [88.0, 95.4] | **88.0% [82.77, 91.8]** | **99.0% [96.43, 99.73]** (99.41%) |
| **AlphaEdit (Weight-Edit, ICLR'25)** | 162.825s | 0.0 | 98.0% [94.97, 99.22] | **83.0% [77.18, 87.57]** | **51.0% [44.12, 57.84]** (63.33%) |
| **FOMP-v3.6-ZeroMQ (0% MQ/ZsRE)** | 0.0984s | 0.0 | 99.5% [97.22, 99.91] | **95.0% [91.04, 97.26]** | **100.0% [98.12, 100.0]** (100.0%) |
| **FOMP-v3.6 (+706 MQ in-domain)** | 0.1033s | 0.0 | 98.5% [95.68, 99.49] | **98.0% [94.97, 99.22]** | **92.5% [88.0, 95.4]** (93.66%) |
| **FOMP-v4 Legible (Route 1: Pure Act, 0% MQ/ZsRE)** | 0.1709s | 0.0 | 50.5% [43.63, 57.35] | **48.0% [41.18, 54.9]** | **100.0% [98.12, 100.0]** (100.0%) |
| **FOMP-v4 Bridge-KV (Route 2: Pre-KV, 0% MQ/ZsRE)** | 2.2464s | 14.02 | 92.5% [88.0, 95.4] | **87.5% [82.2, 91.39]** | **100.0% [98.12, 100.0]** (100.0%) |

## Table 4: Exact Paired McNemar Significance Tests

| Comparison (`A` vs `B`) | Metric | `A` % | `B` % | Diff (pp) | `n10 / n01` | Exact McNemar `p` | `p < 0.05`? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fomp_v4_legible` vs `fomp_v36_zero_mq` | CF Paraphrase EM | 76.5% | 83.5% | -7.00 | 23 / 37 | `0.09246` | no |
| `fomp_v4_legible` vs `fomp_v36_zero_mq` | CF Retain Prob | 87.5% | 86.5% | +1.00 | 20 / 18 | `0.8714` | no |
| `fomp_v4_legible` vs `fomp_v36_zero_mq` | MQuAKE 2-Hop EM | 25.0% | 2.6% | +22.40 | 46 / 3 | `6.981e-11` | YES |
| `fomp_v4_legible` vs `fomp_v36_zero_mq` | ZsRE Generality EM | 48.0% | 95.0% | -47.00 | 3 / 97 | `2.631e-25` | YES |
| `fomp_v4_legible` vs `fomp_v36_zero_mq` | ZsRE Locality Exact | 100.0% | 100.0% | +0.00 | 0 / 0 | `1.0` | no |
| `fomp_v4_legible` vs `alphaedit` | CF Paraphrase EM | 76.5% | 63.0% | +13.50 | 58 / 31 | `0.005545` | YES |
| `fomp_v4_legible` vs `alphaedit` | CF Retain Prob | 87.5% | 85.5% | +2.00 | 19 / 15 | `0.6076` | no |
| `fomp_v4_legible` vs `alphaedit` | MQuAKE 2-Hop EM | 25.0% | 17.71% | +7.29 | 35 / 21 | `0.08143` | no |
| `fomp_v4_legible` vs `alphaedit` | ZsRE Generality EM | 48.0% | 83.0% | -35.00 | 13 / 83 | `1.202e-13` | YES |
| `fomp_v4_legible` vs `alphaedit` | ZsRE Locality Exact | 100.0% | 51.0% | +49.00 | 98 / 0 | `6.311e-30` | YES |
| `fomp_v4_legible` vs `ike_rag_gated_clean` | CF Paraphrase EM | 76.5% | 55.5% | +21.00 | 62 / 20 | `3.715e-06` | YES |
| `fomp_v4_legible` vs `ike_rag_gated_clean` | CF Retain Prob | 87.5% | 84.0% | +3.50 | 12 / 5 | `0.1435` | no |
| `fomp_v4_legible` vs `ike_rag_gated_clean` | MQuAKE 2-Hop EM | 25.0% | 33.33% | -8.33 | 17 / 33 | `0.03284` | YES |
| `fomp_v4_legible` vs `ike_rag_gated_clean` | ZsRE Generality EM | 48.0% | 88.0% | -40.00 | 5 / 85 | `7.533e-20` | YES |
| `fomp_v4_legible` vs `ike_rag_gated_clean` | ZsRE Locality Exact | 100.0% | 99.0% | +1.00 | 2 / 0 | `0.5` | no |
| `fomp_v4_legible` vs `oracle_icl` | CF Paraphrase EM | 76.5% | 60.0% | +16.50 | 57 / 24 | `0.0003173` | YES |
| `fomp_v4_legible` vs `oracle_icl` | CF Retain Prob | 87.5% | 62.0% | +25.50 | 55 / 4 | `1.698e-12` | YES |
| `fomp_v4_legible` vs `oracle_icl` | MQuAKE 2-Hop EM | 25.0% | 59.38% | -34.38 | 4 / 70 | `1.29e-16` | YES |
| `fomp_v4_legible` vs `oracle_icl` | ZsRE Generality EM | 48.0% | 86.5% | -38.50 | 6 / 83 | `2.02e-18` | YES |
| `fomp_v4_legible` vs `oracle_icl` | ZsRE Locality Exact | 100.0% | 14.5% | +85.50 | 171 / 0 | `6.682e-52` | YES |
| `fomp_v4_bridge_kv` vs `fomp_v36_zero_mq` | CF Paraphrase EM | 80.0% | 83.5% | -3.50 | 26 / 33 | `0.435` | no |
| `fomp_v4_bridge_kv` vs `fomp_v36_zero_mq` | CF Retain Prob | 88.0% | 86.5% | +1.50 | 22 / 19 | `0.7552` | no |
| `fomp_v4_bridge_kv` vs `fomp_v36_zero_mq` | MQuAKE 2-Hop EM | 28.65% | 2.6% | +26.04 | 53 / 3 | `8.137e-13` | YES |
| `fomp_v4_bridge_kv` vs `fomp_v36_zero_mq` | ZsRE Generality EM | 87.5% | 95.0% | -7.50 | 4 / 19 | `0.002599` | YES |
| `fomp_v4_bridge_kv` vs `fomp_v36_zero_mq` | ZsRE Locality Exact | 100.0% | 100.0% | +0.00 | 0 / 0 | `1.0` | no |
| `fomp_v4_bridge_kv` vs `fomp_v4_legible` | CF Paraphrase EM | 80.0% | 76.5% | +3.50 | 7 / 0 | `0.01563` | YES |
| `fomp_v4_bridge_kv` vs `fomp_v4_legible` | CF Retain Prob | 88.0% | 87.5% | +0.50 | 5 / 4 | `1.0` | no |
| `fomp_v4_bridge_kv` vs `fomp_v4_legible` | MQuAKE 2-Hop EM | 28.65% | 25.0% | +3.65 | 11 / 4 | `0.1185` | no |
| `fomp_v4_bridge_kv` vs `fomp_v4_legible` | ZsRE Generality EM | 87.5% | 48.0% | +39.50 | 82 / 3 | `5.295e-21` | YES |
| `fomp_v4_bridge_kv` vs `fomp_v4_legible` | ZsRE Locality Exact | 100.0% | 100.0% | +0.00 | 0 / 0 | `1.0` | no |
| `fomp_v4_bridge_kv` vs `alphaedit` | CF Paraphrase EM | 80.0% | 63.0% | +17.00 | 60 / 26 | `0.000317` | YES |
| `fomp_v4_bridge_kv` vs `alphaedit` | CF Retain Prob | 88.0% | 85.5% | +2.50 | 20 / 15 | `0.4996` | no |
| `fomp_v4_bridge_kv` vs `alphaedit` | MQuAKE 2-Hop EM | 28.65% | 17.71% | +10.94 | 40 / 19 | `0.008641` | YES |
| `fomp_v4_bridge_kv` vs `alphaedit` | ZsRE Generality EM | 87.5% | 83.0% | +4.50 | 27 / 18 | `0.2327` | no |
| `fomp_v4_bridge_kv` vs `alphaedit` | ZsRE Locality Exact | 100.0% | 51.0% | +49.00 | 98 / 0 | `6.311e-30` | YES |
| `fomp_v4_bridge_kv` vs `ike_rag_gated_clean` | CF Paraphrase EM | 80.0% | 55.5% | +24.50 | 64 / 15 | `2.303e-08` | YES |
| `fomp_v4_bridge_kv` vs `ike_rag_gated_clean` | CF Retain Prob | 88.0% | 84.0% | +4.00 | 16 / 8 | `0.1516` | no |
| `fomp_v4_bridge_kv` vs `ike_rag_gated_clean` | MQuAKE 2-Hop EM | 28.65% | 33.33% | -4.69 | 17 / 26 | `0.2221` | no |
| `fomp_v4_bridge_kv` vs `ike_rag_gated_clean` | ZsRE Generality EM | 87.5% | 88.0% | -0.50 | 3 / 4 | `1.0` | no |
| `fomp_v4_bridge_kv` vs `ike_rag_gated_clean` | ZsRE Locality Exact | 100.0% | 99.0% | +1.00 | 2 / 0 | `0.5` | no |
| `fomp_v4_bridge_kv` vs `ike_rag_always` | CF Paraphrase EM | 80.0% | 57.5% | +22.50 | 63 / 18 | `5.204e-07` | YES |
| `fomp_v4_bridge_kv` vs `ike_rag_always` | CF Retain Prob | 88.0% | 73.0% | +15.00 | 43 / 13 | `7.333e-05` | YES |
| `fomp_v4_bridge_kv` vs `ike_rag_always` | MQuAKE 2-Hop EM | 28.65% | 38.02% | -9.37 | 29 / 47 | `0.05045` | no |
| `fomp_v4_bridge_kv` vs `ike_rag_always` | ZsRE Generality EM | 87.5% | 77.5% | +10.00 | 27 / 7 | `0.0008214` | YES |
| `fomp_v4_bridge_kv` vs `ike_rag_always` | ZsRE Locality Exact | 100.0% | 7.5% | +92.50 | 185 / 0 | `4.078e-56` | YES |
| `fomp_v4_bridge_kv` vs `oracle_icl` | CF Paraphrase EM | 80.0% | 60.0% | +20.00 | 58 / 18 | `4.713e-06` | YES |
| `fomp_v4_bridge_kv` vs `oracle_icl` | CF Retain Prob | 88.0% | 62.0% | +26.00 | 55 / 3 | `2.26e-13` | YES |
| `fomp_v4_bridge_kv` vs `oracle_icl` | MQuAKE 2-Hop EM | 28.65% | 59.38% | -30.73 | 0 / 59 | `3.469e-18` | YES |
| `fomp_v4_bridge_kv` vs `oracle_icl` | ZsRE Generality EM | 87.5% | 86.5% | +1.00 | 3 / 1 | `0.625` | no |
| `fomp_v4_bridge_kv` vs `oracle_icl` | ZsRE Locality Exact | 100.0% | 14.5% | +85.50 | 171 / 0 | `6.682e-52` | YES |
| `fomp_v36_zero_mq` vs `oracle_icl` | CF Paraphrase EM | 83.5% | 60.0% | +23.50 | 70 / 23 | `1.103e-06` | YES |
| `fomp_v36_zero_mq` vs `oracle_icl` | CF Retain Prob | 86.5% | 62.0% | +24.50 | 65 / 16 | `3.641e-08` | YES |
| `fomp_v36_zero_mq` vs `oracle_icl` | MQuAKE 2-Hop EM | 2.6% | 59.38% | -56.77 | 1 / 110 | `8.628e-32` | YES |
| `fomp_v36_zero_mq` vs `oracle_icl` | ZsRE Generality EM | 95.0% | 86.5% | +8.50 | 22 / 5 | `0.001514` | YES |
| `fomp_v36_zero_mq` vs `oracle_icl` | ZsRE Locality Exact | 100.0% | 14.5% | +85.50 | 171 / 0 | `6.682e-52` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated_clean` | CF Paraphrase EM | 83.5% | 55.5% | +28.00 | 79 / 23 | `2.282e-08` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated_clean` | CF Retain Prob | 86.5% | 84.0% | +2.50 | 26 / 21 | `0.5601` | no |
| `fomp_v36_zero_mq` vs `ike_rag_gated_clean` | MQuAKE 2-Hop EM | 2.6% | 33.33% | -30.73 | 1 / 60 | `5.378e-17` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated_clean` | ZsRE Generality EM | 95.0% | 88.0% | +7.00 | 20 / 6 | `0.009355` | YES |
| `fomp_v36_zero_mq` vs `ike_rag_gated_clean` | ZsRE Locality Exact | 100.0% | 99.0% | +1.00 | 2 / 0 | `0.5` | no |
