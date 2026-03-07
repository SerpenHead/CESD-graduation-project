#!/usr/bin/env bash
# Linux one-click entrypoint for AutoDL.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${DATA_ROOT}" results figures

MODE="${1:-quick}"
MODEL="${2:-llava}"
DECODER="${3:-cesd}"

echo "[run.sh] mode=${MODE} model=${MODEL} decoder=${DECODER}"
echo "[run.sh] project=${PROJECT_ROOT}"
echo "[run.sh] data_root=${DATA_ROOT}"
echo "[run.sh] hf_cache=${HF_HOME}"

case "${MODE}" in
  setup)
    bash scripts/setup_autodl.sh
    ;;
  check)
    python scripts/check_runtime.py \
      --model "${MODEL}" \
      --data_root "${DATA_ROOT}" \
      --output "results/runtime_check_${MODEL}.json"
    ;;
  quick)
    python scripts/run_eval_pope.py \
      --model "${MODEL}" \
      --decoder "${DECODER}" \
      --num_samples 10 \
      --splits random \
      --coco_root "${DATA_ROOT}/mscoco/val2014" \
      --data_path "${DATA_ROOT}/pope"
    ;;
  pope)
    python scripts/run_eval_pope.py \
      --model "${MODEL}" \
      --decoder "${DECODER}" \
      --seed 42 \
      --coco_root "${DATA_ROOT}/mscoco/val2014" \
      --data_path "${DATA_ROOT}/pope"
    ;;
  chair)
    python scripts/run_eval_chair.py \
      --model "${MODEL}" \
      --decoder "${DECODER}" \
      --seed 42 \
      --data_path "${DATA_ROOT}/mscoco"
    ;;
  mme)
    python scripts/run_eval_mme.py \
      --model "${MODEL}" \
      --decoder "${DECODER}" \
      --seed 42 \
      --data_path "${DATA_ROOT}/mme"
    ;;
  ablation)
    python scripts/run_ablation.py \
      --model "${MODEL}" \
      --mode ablation \
      --seed 42 \
      --num_samples 100 \
      --coco_root "${DATA_ROOT}/mscoco/val2014" \
      --data_path "${DATA_ROOT}/pope"
    ;;
  tps)
    python scripts/run_ablation.py \
      --model "${MODEL}" \
      --mode tps \
      --seed 42 \
      --coco_root "${DATA_ROOT}/mscoco/val2014"
    ;;
  *)
    echo "Usage: bash run.sh [setup|check|quick|pope|chair|mme|ablation|tps] [model] [decoder]"
    exit 1
    ;;
esac
