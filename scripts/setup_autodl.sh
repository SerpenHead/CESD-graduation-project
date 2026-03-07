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
python - "$POPE_DIR" <<'PYEOF'
import sys, os, json, urllib.request

pope_dir = sys.argv[1]
os.makedirs(pope_dir, exist_ok=True)

# POPE JSON files live in AoiDragon/POPE on GitHub (output/coco/)
REPO = "AoiDragon/POPE"

for split in ["random", "popular", "adversarial"]:
    fname = f"coco_pope_{split}.json"
    dst = os.path.join(pope_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print(f"  {fname} already exists, skipping.")
        continue
    print(f"  Downloading {fname} ...")
    ok = False

    # Method 1: GitHub API -> download_url (works well in China)
    api_url = f"https://api.github.com/repos/{REPO}/contents/output/coco/{fname}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "CESD-bot"})
        resp = urllib.request.urlopen(req, timeout=30)
        meta = json.loads(resp.read())
        dl_url = meta["download_url"]
        urllib.request.urlretrieve(dl_url, dst)
        ok = os.path.exists(dst) and os.path.getsize(dst) > 1000
        if ok:
            print(f"  -> {dst} ({os.path.getsize(dst)} bytes) [GitHub API]")
    except Exception as e:
        print(f"  GitHub API failed: {e}")

    # Method 2: raw.githubusercontent.com direct
    if not ok:
        raw_url = f"https://raw.githubusercontent.com/{REPO}/main/output/coco/{fname}"
        try:
            urllib.request.urlretrieve(raw_url, dst)
            ok = os.path.exists(dst) and os.path.getsize(dst) > 1000
            if ok:
                print(f"  -> {dst} ({os.path.getsize(dst)} bytes) [GitHub raw]")
        except Exception as e:
            print(f"  GitHub raw failed: {e}")

    if not ok:
        if os.path.exists(dst):
            os.remove(dst)
        print(f"  WARNING: Could not download {fname}.")
        print(f"  Please download manually and place at {dst}")

print("  POPE download step done.")
PYEOF

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
