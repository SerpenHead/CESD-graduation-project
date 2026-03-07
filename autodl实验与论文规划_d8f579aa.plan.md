---
name: AutoDL实验与论文规划
overview: 从 AutoDL 租用 GPU 开始，按阶段完成：环境部署 -> 快速验证 -> 正式实验 -> 数据分析 -> 论文撰写，预计 GPU 总用时约 10-15 小时。
todos:
  - id: fix-paths
    content: 修改 setup_autodl.sh 和 llava.yaml 中的路径，使模型/数据存放到 AutoDL 数据盘（/root/autodl-tmp/），添加 HF 镜像环境变量
    status: pending
  - id: batch-script
    content: 创建 scripts/run_all_experiments.sh 批量实验脚本，串联 POPE + CHAIR + 消融实验，支持 nohup 挂起
    status: pending
  - id: autodl-setup
    content: 在 AutoDL 上克隆仓库、运行 setup_autodl.sh 完成环境初始化
    status: pending
  - id: quick-verify
    content: 用 num_samples=10 快速验证 greedy 和 cesd 流程跑通
    status: pending
  - id: pope-main
    content: 运行 POPE 主实验：5种解码方法 × 3种 split × 500 样本
    status: pending
  - id: chair-main
    content: 运行 CHAIR 主实验：4种解码方法 × 500 样本
    status: pending
  - id: ablation
    content: 运行消融实验 + 参数扫描 + TPS 效率对比
    status: pending
  - id: collect-results
    content: 下载 results/ 目录到本地，生成论文图表
    status: pending
  - id: thesis-write
    content: 基于真实实验数据撰写论文第4章（实验）、第3章（方法补充）、第5章（结论）
    status: pending
isProject: false
---

# AutoDL 实验部署与论文完成全流程规划

## 阶段零：AutoDL 租用与基础配置（约 30 分钟，无需 GPU 费用）

### GPU 选择建议

- **推荐**：RTX 4090 (24GB)，约 2 元/小时，足够跑 7B 模型 fp16
- **备选**：A100 (40/80GB)，约 5 元/小时，更快但更贵
- **镜像**：选择 PyTorch 2.x + Python 3.10 + CUDA 11.8/12.x 官方镜像

### 数据盘 vs 系统盘

- AutoDL 的 `/root/autodl-tmp/` 是**数据盘**（不限容量，关机不丢失），**模型和数据集放这里**
- `/root/` 是系统盘（30GB 限制），**代码放这里**

### 上手步骤

```bash
# 1. 克隆代码到系统盘
cd /root
git clone https://github.com/SerpenHead/CESD-graduation-project.git
cd CESD-graduation-project

# 2. 修改 setup_autodl.sh 中的模型下载路径
#    将模型下载到数据盘以避免系统盘空间不足
#    修改脚本中 local_llava 为:
#      local_llava = "/root/autodl-tmp/models/llava-1.6-vicuna-7b"

# 3. 同时修改 configs/models/llava.yaml 取消注释并指向数据盘:
#    model_path: /root/autodl-tmp/models/llava-1.6-vicuna-7b

# 4. 建议将 data 也放到数据盘（COCO val2014 约 6.6GB）
#    ln -s /root/autodl-tmp/data /root/CESD-graduation-project/data
```

**重要**：当前 [setup_autodl.sh](scripts/setup_autodl.sh) 将模型下载到项目内的 `models/` 目录，但 [llava.yaml](configs/models/llava.yaml) 注释中的路径是 `/root/autodl-tmp/models/llava-1.6-vicuna-7b`。需要让两者保持一致，否则模型加载时会从 HuggingFace Hub 在线下载（非常慢或失败）。具体做法：

- 修改 `setup_autodl.sh` 第 59 行的 `local_llava` 为 `/root/autodl-tmp/models/llava-1.6-vicuna-7b`
- 取消 `configs/models/llava.yaml` 第 7 行的注释，设为 `model_path: /root/autodl-tmp/models/llava-1.6-vicuna-7b`

---

## 阶段一：环境安装与数据下载（约 1-2 小时，GPU 空闲）

```bash
# 安装依赖 + 下载 POPE + 下载 COCO + 下载模型
bash scripts/setup_autodl.sh
```

**注意事项**：

- COCO val2014 下载约 6.6GB，AutoDL 国内网络下 cocodataset.org 可能较慢。备选方案：使用国内镜像或 `autodl-tmp` 中的数据集市场
- HuggingFace 模型下载在国内可能需要设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 如果 POPE JSON 下载失败，可以手动从 GitHub 仓库 `RUCAIBox/POPE` 下载三个 JSON 文件放到 `data/pope/`

---

## 阶段二：快速验证（约 10 分钟，开始计 GPU 费用）

**目标**：确认整条流水线（模型加载 -> 输入处理 -> 解码 -> 评估 -> 结果保存）能跑通，不追求结果质量。

```bash
# 验证 Greedy（最简单，最快）
python scripts/run_eval_pope.py \
  --model llava --decoder greedy \
  --num_samples 10 --splits random

# 验证 CESD（核心算法）
python scripts/run_eval_pope.py \
  --model llava --decoder cesd \
  --num_samples 10 --splits random

# 验证 CHAIR
python scripts/run_eval_chair.py \
  --model llava --decoder greedy \
  --num_samples 5
```

**预期结果**：每个命令应在 5 分钟内完成，输出 JSON 到 `results/` 目录。如果报错，需要先调试修复。

**常见问题预案**：

- `CUDA out of memory`：检查是否有其他进程占用 GPU（`nvidia-smi`）
- `FileNotFoundError`：数据路径不对，检查 `data/pope/` 和 `data/mscoco/val2014/` 是否存在
- `ImportError`：依赖缺失，重新 `pip install -r requirements.txt`
- 模型加载失败：确认 `llava.yaml` 中 `model_path` 指向正确的本地路径

---

## 阶段三：正式实验（约 6-8 小时 GPU 时间）

验证通过后，按以下顺序执行正式实验。建议用 `nohup` 或 `tmux` 挂起长任务，防止终端断开丢失进度。

### 3.1 POPE 主实验（约 4-5 小时）

论文表格需要：LLaVA-1.6 × {Greedy, Beam, DoLa, iTaD, CESD} × {random, popular, adversarial}

```bash
# 使用 tmux 防止断连
tmux new -s exp

# 依次运行（每个约 20-60 分钟）
python scripts/run_eval_pope.py --model llava --decoder greedy --seed 42 --measure_tps
python scripts/run_eval_pope.py --model llava --decoder beam   --seed 42
python scripts/run_eval_pope.py --model llava --decoder dola   --seed 42
python scripts/run_eval_pope.py --model llava --decoder itad   --seed 42 --measure_tps
python scripts/run_eval_pope.py --model llava --decoder cesd   --seed 42 --measure_tps
```

每个命令默认跑 500 条 × 3 个 split。预计：

- Greedy/Beam: ~20 分钟
- DoLa: ~30 分钟
- iTaD: ~50 分钟（2x forward）
- CESD: ~60-90 分钟（2x forward + 稀疏化计算）

### 3.2 CHAIR 主实验（约 2-3 小时）

```bash
python scripts/run_eval_chair.py --model llava --decoder greedy --seed 42
python scripts/run_eval_chair.py --model llava --decoder dola   --seed 42
python scripts/run_eval_chair.py --model llava --decoder itad   --seed 42
python scripts/run_eval_chair.py --model llava --decoder cesd   --seed 42
```

### 3.3 消融实验（约 1-2 小时）

```bash
# 组件消融（移除动态层选择 / 移除稀疏化 / 两者都移除）
python scripts/run_ablation.py --mode ablation --seed 42 --num_samples 100

# alpha 参数扫描
python scripts/run_ablation.py --mode alpha --seed 42 --num_samples 100

# 稀疏化比例扫描
python scripts/run_ablation.py --mode sparsify --seed 42 --num_samples 100

# TPS 效率对比
python scripts/run_ablation.py --mode tps --seed 42
```

### 3.4（可选）Qwen2-VL 泛化性验证（额外 4-6 小时）

如果时间和预算允许，在 Qwen2-VL 上重复 POPE 主实验：

```bash
python scripts/run_eval_pope.py --model qwen2_vl --decoder greedy --seed 42
python scripts/run_eval_pope.py --model qwen2_vl --decoder cesd   --seed 42
```

---

## 阶段四：数据收集与图表生成（GPU 可关闭，本地完成）

实验完成后，将 `results/` 目录下的所有 JSON 文件下载到本地（`git` 不追踪 `results/`，需用 `scp` 或 AutoDL 文件下载）。

### 需要生成的论文图表

1. **POPE 主结果表格**（表4.1）：5种方法 × 3种split × 4个指标（Acc/P/R/F1）
2. **CHAIR 结果表格**（表4.2）：4种方法 × CHAIR_s / CHAIR_i
3. **消融实验表格**（表4.3）：Full CESD vs 去掉动态层选 vs 去掉稀疏化 vs 两者都去掉
4. **alpha 参数敏感性曲线**（图4.x）：x轴 alpha, y轴 F1/Accuracy
5. **sparsify_ratio 参数敏感性曲线**（图4.x）：x轴 ratio, y轴 F1/Accuracy
6. **TPS 效率对比条形图**（图4.x）：各方法 tokens/sec
7. **CESD 框架图**（图3.x）：算法流程示意（可手绘或用 mermaid/tikz）

这些图表可以在拿到 JSON 数据后，用 `src/analysis/plotting.py` 或 Jupyter notebook 生成。

---

## 阶段五：论文撰写（本地完成）

### 写作顺序建议

1. **第4章 实验**（先填表格和数据，再写分析段落）
  - 4.1 实验设置（模型、数据集、评估指标、基线方法 — 基本已有框架）
  - 4.2 POPE 结果分析
  - 4.3 CHAIR 结果分析
  - 4.4 消融实验
  - 4.5 参数敏感性分析
  - 4.6 效率分析
2. **第3章 方法**（结合实验数据补充动机和公式推导）
3. **第5章 结论**（基于真实实验数据总结）
4. **摘要 + 引言**（最后写）

---

## 预算估算


| 项目           | GPU 时间      | 单价（4090） | 费用        |
| ------------ | ----------- | -------- | --------- |
| 环境配置+数据下载    | 1h（可不开 GPU） | 0        | 0 元       |
| 快速验证         | 0.5h        | 2元/h     | 1 元       |
| POPE 主实验     | 5h          | 2元/h     | 10 元      |
| CHAIR 主实验    | 2.5h        | 2元/h     | 5 元       |
| 消融实验         | 1.5h        | 2元/h     | 3 元       |
| 调试+重跑 buffer | 3h          | 2元/h     | 6 元       |
| **合计**       | **~13h**    |          | **~25 元** |


如果加上 Qwen2-VL 实验，额外约 15 元，总计约 40 元。

---

## 需要代码修改的地方

在去 AutoDL 之前，建议先在本地做以下代码调整：

1. **修改 `setup_autodl.sh`**：将模型下载路径改为 `/root/autodl-tmp/models/`，数据下载路径改为 `/root/autodl-tmp/data/`，并在脚本中创建软链接到项目目录
2. **取消注释 `configs/models/llava.yaml` 的 `model_path`**：指向 `/root/autodl-tmp/models/llava-1.6-vicuna-7b`
3. **添加 HuggingFace 镜像环境变量**：在 `setup_autodl.sh` 开头添加 `export HF_ENDPOINT=https://hf-mirror.com`
4. **添加批量实验脚本**：创建一个 `scripts/run_all_experiments.sh` 串联所有实验命令，方便 `nohup` 一键挂起

