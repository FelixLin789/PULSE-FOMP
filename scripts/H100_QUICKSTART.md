# PULSE-v8.1 Phase 2 H100 Scaling Law Quickstart (`125M × 2.5B` Tokens)

## 1. Recommended Rented GPU Spec (RunPod / Vast.ai / Lambda)
- **GPU**: `1× NVIDIA H100 80GB SXM5` (or `H100 PCIe` / `A100 80GB`)
- **Container Image**: PyTorch `2.4+` / `2.5+` with CUDA `12.4+` (Ubuntu 22.04)
- **Container / Volume Disk**: `>= 50 GB` (`5.0 GB` FineWeb-Edu `2.5B` shard + checkpoints + HF eval cache)
- **Expected Wall-Clock & Cost (`1× H100 80GB`)**:
  - Data shard prep (`2.5B` FineWeb-Edu): `~4–6 min`
  - Preflight check (speed, VRAM, `16K` finite check): `~2 min`
  - `TF-Modern-125M` + `PULSE-v8.1-125M` (`2.5B` tokens each): `~5.5–7 hours`
  - Suites `E1..E4` (`16K` extrapolation + 5 zero-shot benchmarks + `16K` needle + `FOMP-v3.2` loop): `~20–30 min`
  - **Total**: `~6.5–7.5 hours` (`~$14–$20` on RunPod/Vast.ai)

## 2. Upload & Launch Inside `tmux` (Crash-Proof!)
After uploading `pulse_h100_phase2.tar.gz` to your H100 box (e.g., `/workspace`):

```bash
mkdir -p /workspace/pulse_phase2 && cd /workspace/pulse_phase2
tar -xzf /workspace/pulse_h100_phase2.tar.gz

# Launch inside tmux so closing SSH NEVER kills training!
tmux new -s pulse "bash scripts/run_h100_scaling_law.sh"
```

- Detach tmux safely: press `Ctrl+B`, then `D`.
- Reattach anytime: `tmux attach -t pulse`.
- Live tail log: `tail -f runs/phase2_125m_2p5b/run.log`.

## 3. Crash / Spot Interruption Recovery (Zero Lost Work)
Checkpoints (`ckpt_latest.pt`) are saved atomically every `250M` tokens (`~15 min`). If the pod restarts, just re-run the exact same command:
```bash
tmux new -s pulse "bash scripts/run_h100_scaling_law.sh"
```
It automatically reuses `data/fineweb_edu_2p5b/` and resumes from the exact step & token cursor.

## 4. Multi-GPU DDP or Smaller-VRAM Overrides
```bash
# 2x RTX 4090 (24GB each, DDP):
NPROC=2 MICRO_BATCH=4 GRAD_ACCUM=64 bash scripts/run_h100_scaling_law.sh

# Fast 1.0B token probe first (can resume to 2.5B later):
TOKENS=1000000000 bash scripts/run_h100_scaling_law.sh
```

## 5. Files to Download When Finished
- `runs/phase2_125m_2p5b/scaling_law_report.md`
- `runs/phase2_125m_2p5b/scaling_law_summary.json`
- `runs/phase2_125m_2p5b/pulse_v81/ckpt_final.pt` (optional trained weights)
