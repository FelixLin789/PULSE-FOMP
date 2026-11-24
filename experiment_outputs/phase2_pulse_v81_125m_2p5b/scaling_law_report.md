# PULSE-v8.1 vs TF-Modern Phase 2 Scaling Law Report (`125m`, `2.50B` tokens)

- **GPU**: `NVIDIA H800 PCIe` (`world_size=1`)
- **TF-Modern**: `123.93M` params | `122.6K tok/s` | `16K state = 576.0 MiB`
- **PULSE-v8.1**: `122.52M` params | `132.3K tok/s` | `16K state = 14.0 MiB`

## Suite E1: Long-Context Extrapolation (FineWeb-Edu Val PPL)
| Context | TF-Modern PPL | PULSE-v8.1 PPL | Delta PPL | State Ratio | Prefill ms (TF vs PULSE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `L2048` | `21.065` | **`20.2556`** | `-0.809` | **`20.6x`** | `12.93ms` vs `90.9ms` |
| `L4096` | `25.3663` | **`19.9675`** | `-5.399` | **`28.8x`** | `23.7ms` vs `157.97ms` |
| `L8192` | `65.7479` | **`19.8164`** | `-45.931` | **`36.0x`** | `48.92ms` vs `294.75ms` |
| `L16384` | `181.2032` | **`19.8`** | `-161.403` | **`41.1x`** | `107.99ms` vs `582.2ms` |

## Suite E2: Real Public Zero-Shot Downstream Benchmarks
| Task | TF-Modern Acc | PULSE-v8.1 Acc | Delta |
| :--- | :---: | :---: | :---: |
| `lambada` | `24.6%` | **`26.9%`** | `+2.30 pp` |
| `hellaswag` | `28.0%` | **`28.3%`** | `+0.30 pp` |
