# CESD 在 AutoDL Linux 运行说明

本文档用于 Ubuntu/AutoDL GPU 服务器部署与运行。

## 1. 环境要求

- Ubuntu 20.04/22.04
- Python 3.10
- CUDA 11.8 或 12.x
- 建议 GPU:
  - RTX 4090 (24GB) 或
  - A100 (40GB/80GB)

## 2. 拉取代码

```bash
cd /root
git clone https://github.com/SerpenHead/CESD-graduation-project.git
cd CESD-graduation-project
```

## 3. 一键安装与数据准备

```bash
# 可选：使用国内 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 可选：将数据放在数据盘（AutoDL 推荐）
export DATA_ROOT=/root/autodl-tmp/data

bash run.sh setup
```

该步骤会：
- 安装 `requirements.txt` 依赖
- 下载 POPE 数据
- 下载 COCO val2014 与标注
- 下载 LLaVA-1.6 模型到 `/root/autodl-tmp/models/llava-1.6-vicuna-7b`
- 在项目目录创建 `data -> ${DATA_ROOT}` 软链接（若不存在）

## 4. 运行前自检（推荐）

```bash
# 完整自检（包含模型加载冒烟）
bash run.sh check llava

# 若只检查 CUDA/路径，不加载模型
python scripts/check_runtime.py --model llava --data_root "${DATA_ROOT}" --skip_model
```

自检会覆盖：
- torch + cuda 可用性
- HuggingFace 模型加载
- 数据集路径完整性
- DataLoader 多进程可用性（num_workers=2）
- OOM 风险提示

## 5. 快速验证

```bash
# 10 样本快速验证
bash run.sh quick llava cesd
```

## 6. 正式实验

```bash
# POPE
bash run.sh pope llava greedy
bash run.sh pope llava itad
bash run.sh pope llava cesd

# CHAIR
bash run.sh chair llava greedy
bash run.sh chair llava cesd

# MME
bash run.sh mme llava cesd

# 消融 + TPS
bash run.sh ablation llava
bash run.sh tps llava
```

## 7. 常见问题

### 7.1 CUDA 不可用

- 检查 `nvidia-smi` 是否正常
- 确认 AutoDL 镜像是 GPU 版本（非 CPU 镜像）

### 7.2 模型下载失败

- 设置 `HF_ENDPOINT=https://hf-mirror.com`
- 检查模型路径与 `configs/models/llava.yaml` 的 `model_path` 是否一致

### 7.3 OOM（显存不足）

- 先运行 `bash run.sh quick llava greedy`
- 降低生成长度（`max_new_tokens`）
- 优先运行 Greedy/DoLa，再运行 CESD（CESD 每步 token 有双前向）

## 8. 输出目录

- 评测结果：`results/*.json`
- 自检报告：`results/runtime_check_*.json`
- 图表输出：`figures/`
