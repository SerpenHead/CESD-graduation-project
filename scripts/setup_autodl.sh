#!/bin/bash
# AutoDL 环境一键初始化脚本
# 在 AutoDL 实例的终端中执行: bash scripts/setup_autodl.sh
# 适用于 PyTorch 2.x 镜像 (Python 3.10, CUDA 11.8+)

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
mkdir -p "$DATA_ROOT" "/root/autodl-tmp/models"

echo "=== [1/5] 安装 Python 依赖 ==="
pip install -q --upgrade pip

# 确保 PyTorch >= 2.4（transformers 4.45+ 要求）
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "0")
if python -c "from packaging.version import Version; exit(0 if Version('${TORCH_VER}') >= Version('2.4.0') else 1)" 2>/dev/null; then
    echo "  PyTorch ${TORCH_VER} OK"
else
    echo "  PyTorch ${TORCH_VER} < 2.4, upgrading..."
    pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
fi

pip install -q -r requirements.txt
# spaCy 英语模型（CHAIR 词形还原备用）
python -m nltk.downloader -q punkt averaged_perceptron_tagger

echo "=== [2/5] 创建数据目录 ==="
mkdir -p "$DATA_ROOT/pope" "$DATA_ROOT/mscoco/val2014" "$DATA_ROOT/mscoco/annotations" "$DATA_ROOT/mme" results figures
if [ ! -e data ]; then
    ln -s "$DATA_ROOT" data
fi

echo "=== [3/5] 下载 POPE 数据集 ==="
POPE_DIR="$DATA_ROOT/pope"
for SPLIT in random popular adversarial; do
    DEST="${POPE_DIR}/coco_pope_${SPLIT}.json"
    if [ -f "$DEST" ] && [ -s "$DEST" ]; then
        echo "  coco_pope_${SPLIT}.json already exists, skipping."
        continue
    fi
    echo "  Downloading coco_pope_${SPLIT}.json ..."
    # 优先使用 HF 镜像，失败则 fallback 到 GitHub raw
    HF_URL="${HF_ENDPOINT}/datasets/lmms-lab/POPE/resolve/main/coco/coco_pope_${SPLIT}.json"
    GH_URL="https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_${SPLIT}.json"
    wget -q --timeout=30 -O "$DEST" "$HF_URL" 2>/dev/null || \
    wget -q --timeout=30 -O "$DEST" "$GH_URL" 2>/dev/null || \
    { rm -f "$DEST"; echo "  WARNING: Failed to download ${SPLIT}. Please download manually."; }
    # 检查下载的文件是否有效（非空且是 JSON）
    if [ -f "$DEST" ] && ! head -c1 "$DEST" | grep -q '\['; then
        echo "  WARNING: Downloaded ${SPLIT} appears invalid, removing."
        rm -f "$DEST"
    fi
done

echo "=== [4/5] 下载 COCO val2014 (约 6.6 GB) ==="
COCO_ZIP="$DATA_ROOT/val2014.zip"
COCO_ANN_ZIP="$DATA_ROOT/annotations_trainval2014.zip"
if [ ! -f "$DATA_ROOT/mscoco/val2014/COCO_val2014_000000000042.jpg" ]; then
    if [ ! -f "$COCO_ZIP" ] || ! unzip -tq "$COCO_ZIP" >/dev/null 2>&1; then
        echo "  Downloading COCO val2014 images (supports resume with -c)..."
        rm -f "$COCO_ZIP"
        wget -c --progress=bar:force "http://images.cocodataset.org/zips/val2014.zip" -O "$COCO_ZIP"
    fi
    echo "  Extracting val2014..."
    unzip -q -o "$COCO_ZIP" -d "$DATA_ROOT/mscoco/"
fi
if [ ! -f "$DATA_ROOT/mscoco/annotations/instances_val2014.json" ]; then
    if [ ! -f "$COCO_ANN_ZIP" ] || ! unzip -tq "$COCO_ANN_ZIP" >/dev/null 2>&1; then
        echo "  Downloading COCO annotations..."
        rm -f "$COCO_ANN_ZIP"
        wget -c --progress=bar:force "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" -O "$COCO_ANN_ZIP"
    fi
    echo "  Extracting annotations..."
    unzip -q -o "$COCO_ANN_ZIP" -d "$DATA_ROOT/mscoco/"
fi

echo "=== [5/5] 下载模型 (HuggingFace Hub) ==="
python - <<'EOF'
from huggingface_hub import snapshot_download
import os

# LLaVA-1.6-vicuna-7b
local_llava = "/root/autodl-tmp/models/llava-1.6-vicuna-7b"
if not os.path.exists(local_llava):
    print("Downloading LLaVA-1.6-vicuna-7b ...")
    snapshot_download(
        repo_id="llava-hf/llava-v1.6-vicuna-7b-hf",
        local_dir=local_llava,
        ignore_patterns=["*.msgpack", "flax_model*"],
    )
    print("Done.")
else:
    print(f"Model already at {local_llava}")
EOF

echo ""
echo "=== 环境准备完成 ==="
echo ""
echo "快速验证命令:"
echo "  python scripts/run_eval_pope.py --model llava --decoder greedy --num_samples 10 --splits random"
echo ""
echo "完整评估命令:"
echo "  python scripts/run_eval_pope.py  --model llava --decoder cesd   --seed 42"
echo "  python scripts/run_eval_chair.py --model llava --decoder cesd   --seed 42"
echo "  python scripts/run_ablation.py   --mode ablation  --seed 42 --num_samples 100"
echo "  python scripts/run_ablation.py   --mode tps       --seed 42"
