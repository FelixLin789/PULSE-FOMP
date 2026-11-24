#!/usr/bin/env bash
# One-Command Rented H100/A100/4090 Phase 2 Scaling Law Runner (PULSE-v8.1 vs TF-Modern)
# Recommended inside tmux: `tmux new -s pulse "bash scripts/run_h100_scaling_law.sh"`
set -euo pipefail

SCALE="${SCALE:-125m}"
TOKENS="${TOKENS:-2500000000}"
SEQ_LEN="${SEQ_LEN:-2048}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
NPROC="${NPROC:-1}"
ARMS="${ARMS:-pulse_v81,tf_modern}"
SKIP_PIP="${SKIP_PIP:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
DATA_DIR="${DATA_DIR:-data/fineweb_edu_2p5b}"
OUT_DIR="${OUT_DIR:-runs/phase2_${SCALE}_2p5b}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run.log"

echo "=== [1/5] Hardware & Environment Check ===" | tee -a "${LOG}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a "${LOG}"
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count(), '| CC:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'none')" | tee -a "${LOG}"

if [ "${SKIP_PIP}" != "1" ]; then
  echo "=== [2/5] Installing Required Packages (FLA Triton + tiktoken + datasets) ===" | tee -a "${LOG}"
  python3 -m pip install -q --upgrade pip
  python3 -m pip install -q tiktoken datasets transformers einops pytest
  python3 -c "import fla" 2>/dev/null || python3 -m pip install -q flash-linear-attention
fi

if [ "${SKIP_TESTS}" != "1" ]; then
  echo "=== [3/5] Unit & Invariant Self-Check ===" | tee -a "${LOG}"
  python3 -m pytest tests/test_fomp_v3.py tests/test_pulse_v8.py tests/test_pulse_v81_scaling_law.py -q | tee -a "${LOG}"
fi

echo "=== [4/5] Preparing 2.5B FineWeb-Edu Memmap Shard (One-Time, Resumable) ===" | tee -a "${LOG}"
python3 -u benchmarks/pulse_v81_scaling_law.py \
  --out "${OUT_DIR}" --data-dir "${DATA_DIR}" --scale "${SCALE}" \
  --tokens "${TOKENS}" --prepare-data-only 2>&1 | tee -a "${LOG}"

if [ "${SKIP_PREFLIGHT}" != "1" ]; then
  echo "=== [4.5/5] 2-Minute Preflight (Speed, Peak VRAM, 16K Finite Check) ===" | tee -a "${LOG}"
  python3 -u benchmarks/pulse_v81_scaling_law.py \
    --out "${OUT_DIR}" --data-dir "${DATA_DIR}" --scale "${SCALE}" \
    --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --arms "${ARMS}" \
    --micro-batch "${MICRO_BATCH}" --grad-accum "${GRAD_ACCUM}" \
    --preflight 2>&1 | tee -a "${LOG}"
fi

echo "=== [5/5] Launching Full Resumable 2-Arm Showdown + E1..E4 Suites ===" | tee -a "${LOG}"
if [ "${NPROC}" -gt 1 ]; then
  LAUNCHER=(torchrun --standalone --nproc_per_node="${NPROC}")
else
  LAUNCHER=(python3 -u)
fi

"${LAUNCHER[@]}" benchmarks/pulse_v81_scaling_law.py \
  --out "${OUT_DIR}" --data-dir "${DATA_DIR}" --scale "${SCALE}" \
  --tokens "${TOKENS}" --seq-len "${SEQ_LEN}" --arms "${ARMS}" \
  --micro-batch "${MICRO_BATCH}" --grad-accum "${GRAD_ACCUM}" \
  2>&1 | tee -a "${LOG}"

echo "=== ALL DONE! Key outputs: ${OUT_DIR}/scaling_law_report.md & ${OUT_DIR}/scaling_law_summary.json ===" | tee -a "${LOG}"
