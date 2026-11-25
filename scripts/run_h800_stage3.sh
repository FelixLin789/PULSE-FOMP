#!/usr/bin/env bash
# Zero-Download AutoDL H800 Stage 3.2 Runner (Pure-Activation 3-Route FOMP-v4.2 + 4-Shot IKE Demo & Chat Audit)
set -euo pipefail

cd /root/autodl-tmp/stage25
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export PYTHONPATH="$PWD/src:$PWD/benchmarks:$PWD/EasyEdit:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OUT_DIR="${OUT_DIR:-runs/fomp_v42_stage32}"
STAGE25_DIR="${STAGE25_DIR:-runs/fomp_v36_rigor_audit}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/stage32.log"

echo "=== [1/3] Running All Unit & Invariance Tests (v3 + v3.6 + v4.2) ===" | tee -a "${LOG}"
python3 -m pytest tests/test_fomp_v3.py tests/test_fomp_v36_rigor_audit.py tests/test_fomp_v4.py -q | tee -a "${LOG}"

echo "=== [2/3] Launching Stage 3.2 (4-Shot IKE Std & Chat + Clean Demo RAG + Pure-Activation 3-Route FOMP-v4.2, 0% MQuAKE!) ===" | tee -a "${LOG}"
DEADLINE="$(python3 -c 'import time; print(int(time.time() + 10800))')"
python3 -u benchmarks/fomp_v4_multihop_stage3.py \
  --out "${OUT_DIR}" \
  --stage25-dir "${STAGE25_DIR}" \
  --model-id "models/Meta-Llama-3-8B-Instruct" \
  --ike-model-id "models/bge-small-en-v1.5" \
  --arms "oracle_ike_std,oracle_ike_chat,ike_rag_always_demo,ike_rag_gated_clean_demo,fomp_v42_legible" \
  --deadline "${DEADLINE}" \
  2>&1 | tee -a "${LOG}"

echo "=== [3/3] STAGE 3.2 COMPLETE REPORT ===" | tee -a "${LOG}"
cp -f "${OUT_DIR}/stage32_multihop_report.md" "${OUT_DIR}/stage32_multihop_summary.json" /root/autodl-tmp/
cat "${OUT_DIR}/stage32_multihop_report.md"

if [ "${AUTO_SHUTDOWN:-1}" = "1" ]; then
  echo "[Auto-Shutdown] Stage 3.2 finished cleanly! Outputs copied to /root/autodl-tmp/; shutting down H800 in 30s..." | tee -a "${LOG}"
  sleep 30
  shutdown -h now
fi
