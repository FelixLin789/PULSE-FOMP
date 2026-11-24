#!/usr/bin/env bash
# One-Command AutoDL H800 Stage 2.5 Peer-Review Rigor Audit Runner
# Offline EasyEdit + pre-trained fomp_v36_llama3_8b.pt + hf-mirror (0 GitHub dependency!)
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export PYTHONPATH="$(pwd)/src:$(pwd)/EasyEdit:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ARMS="${ARMS:-base,ike_rag_always,ike_rag_gated,fomp_v36,fomp_v36_zero_mq,alphaedit}"
OUT_DIR="${OUT_DIR:-runs/fomp_v36_rigor_audit}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/rigor_audit.log"

echo "=== [1/4] Hardware & Disk Check ===" | tee -a "${LOG}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a "${LOG}"
df -h /root/autodl-tmp | tee -a "${LOG}"

echo "=== [2/4] Installing Lean Dependencies (Tsinghua PyPI, keeping system CUDA PyTorch) ===" | tee -a "${LOG}"
python3 -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
  pyyaml hydra-core omegaconf higher nltk sentencepiece accelerate datasets transformers einops pytest

echo "=== [3/4] Verifying Official EasyEdit AlphaEdit Import & 9/9 Unit Tests ===" | tee -a "${LOG}"
python3 -c "
from easyeditor.models.alphaedit.AlphaEdit_main import apply_AlphaEdit_to_model
from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams
print('Official EasyEdit AlphaEdit imported cleanly!')
" | tee -a "${LOG}"
python3 -m pytest tests/test_fomp_v3.py tests/test_fomp_v36_rigor_audit.py -q | tee -a "${LOG}"

echo "=== [4/4] Launching Stage 2.5 Rigor Audit (6 Arms, n=200/192/200, Full Gen-EM + McNemar) ===" | tee -a "${LOG}"
DEADLINE="$(python3 -c 'import time; print(int(time.time() + 7200))')"
python3 -u benchmarks/fomp_v36_rigor_audit.py \
  --out "${OUT_DIR}" \
  --stage2-dir "$(pwd)/stage2_cache" \
  --easyedit-dir "$(pwd)/EasyEdit" \
  --arms "${ARMS}" \
  --deadline "${DEADLINE}" \
  2>&1 | tee -a "${LOG}"

echo "=== STAGE 2.5 REPORT ===" | tee -a "${LOG}"
cat "${OUT_DIR}/rigor_audit_report.md"
