#!/usr/bin/env bash
# Fast cold-start bootstrap for a fresh dual-4090 machine.
# Usage:
#   bash scripts/bootstrap_fresh_machine.sh model
#   bash scripts/bootstrap_fresh_machine.sh data
#   bash scripts/bootstrap_fresh_machine.sh all

set -euo pipefail

MODE="${1:-all}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
export MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${DATA_ROOT}" "${MODEL_ROOT}" results figures

if [ ! -e data ]; then
  ln -s "${DATA_ROOT}" data
fi

download_model() {
  pip install -q -r requirements.txt hf_transfer
  python - <<'PY'
from huggingface_hub import snapshot_download
import os

model_root = os.environ.get("MODEL_ROOT", "/root/autodl-tmp/models")
os.makedirs(model_root, exist_ok=True)

jobs = [
    ("llava-hf/llava-v1.6-vicuna-7b-hf", os.path.join(model_root, "llava-1.6-vicuna-7b")),
    ("Qwen/Qwen2-VL-7B-Instruct", os.path.join(model_root, "qwen2-vl-7b-instruct")),
]
for repo_id, local_dir in jobs:
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"[model] skip existing: {local_dir}")
        continue
    print(f"[model] downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "flax_model*"],
        resume_download=True,
    )
print("[model] done")
PY
}

download_data() {
  mkdir -p "${DATA_ROOT}/pope" "${DATA_ROOT}/mscoco" "${DATA_ROOT}/mscoco/annotations" "${DATA_ROOT}/mme"

  # POPE
  python - <<'PY'
import os, json, urllib.request

pope_dir = os.path.join(os.environ.get("DATA_ROOT", "/root/autodl-tmp/data"), "pope")
os.makedirs(pope_dir, exist_ok=True)
repo = "AoiDragon/POPE"
for split in ["random", "popular", "adversarial"]:
    fname = f"coco_pope_{split}.json"
    dst = os.path.join(pope_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        print(f"[data] skip existing {fname}")
        continue
    api = f"https://api.github.com/repos/{repo}/contents/output/coco/{fname}"
    req = urllib.request.Request(api, headers={"User-Agent": "CESD-bootstrap"})
    meta = json.loads(urllib.request.urlopen(req, timeout=30).read())
    urllib.request.urlretrieve(meta["download_url"], dst)
    print(f"[data] downloaded {fname}")
PY

  # COCO val2014 + annotations
  COCO_ZIP="${DATA_ROOT}/val2014.zip"
  ANN_ZIP="${DATA_ROOT}/annotations_trainval2014.zip"
  if command -v aria2c >/dev/null 2>&1; then
    DL="aria2c -x16 -s16 -k1M -c"
  else
    DL="wget -c --progress=bar:force"
  fi

  if [ ! -f "${DATA_ROOT}/mscoco/val2014/COCO_val2014_000000000042.jpg" ]; then
    ${DL} "http://images.cocodataset.org/zips/val2014.zip" -o "${COCO_ZIP##*/}" -d "${DATA_ROOT}"
    unzip -q -o "${COCO_ZIP}" -d "${DATA_ROOT}/mscoco/"
  fi

  if [ ! -f "${DATA_ROOT}/mscoco/annotations/instances_val2014.json" ]; then
    ${DL} "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" -o "${ANN_ZIP##*/}" -d "${DATA_ROOT}"
    unzip -q -o "${ANN_ZIP}" -d "${DATA_ROOT}/mscoco/"
  fi

  echo "[data] done"
}

case "${MODE}" in
  model)
    download_model
    ;;
  data)
    download_data
    ;;
  all)
    download_model
    download_data
    ;;
  *)
    echo "Usage: bash scripts/bootstrap_fresh_machine.sh [model|data|all]"
    exit 1
    ;;
esac

echo "[bootstrap] completed."
