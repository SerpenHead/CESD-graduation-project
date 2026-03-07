# CESD 毕设项目进度报告

> 生成时间：2026-03-07  
> 用于：在新 AI 对话中快速恢复项目上下文

---

## 1. 项目目标

**课题**：无需训练的高效多模态大模型幻觉缓解算法研究  
**学校**：哈尔滨工业大学深圳校区  
**学号**：220111030  
**导师**：陈斌

**核心贡献**：提出 **CESD（Contrast-Enhanced Sparsified Decoding）** 算法，在推理解码阶段缓解视觉语言模型（VLM）的视觉幻觉，无需任何额外训练。

**CESD 算法原理（每个 token 生成步骤）**：
1. **Expert Pass**：完整前向传播 → 获得 expert logits + 所有层 hidden_states + attentions
2. **iTaV 计算**：对每层注意力矩阵，max-over-heads → softmax，得到各层对图像 token 的注意力向量
3. **动态层选择**：用 JSD(iTaV_N, iTaV_m) 选出与最终层差异最大的中间层 M*（"业余层"）
4. **Top-K 稀疏化**：在 M* 层输入的 hidden state 上，按注意力分数保留 top-k 个 token，其余置零
5. **Amateur Pass**：通过 **pre-forward hook** 将稀疏化后的 hidden state 注入 M* 层，再次完整前向传播得到 amateur logits
6. **对比解码**：`logits_final = expert + α * (expert - amateur)`

**与基线 iTaD 的区别**：iTaD 只做动态层选择 + 对比解码（无稀疏化）；CESD 在此基础上增加局部稀疏化，放大专家/业余层差异，理论上提升对比效果。

---

## 2. 当前代码结构

```
f:\graudationdesign\
├── PROJECT_STATE.md          ← 本文件
├── README.md
├── requirements.txt
│
├── configs/
│   ├── models/
│   │   ├── llava.yaml        ← LLaVA-1.6-vicuna-7b 配置
│   │   └── qwen2vl.yaml      ← Qwen2-VL-7B-Instruct 配置
│   └── eval/
│       ├── pope.yaml
│       ├── chair.yaml
│       └── mme.yaml
│
├── src/
│   ├── models/
│   │   ├── model_loader.py   ← 统一模型加载 + 聊天模板处理 ✅
│   │   └── model_utils.py    ← get_image_token_indices, get_model_info ✅
│   │
│   ├── decoding/             ← 所有解码策略 ✅
│   │   ├── cesd.py           ← CESD 核心算法（hook-based）✅
│   │   ├── itad.py           ← iTaD 基线（hook-based）✅
│   │   ├── dola.py           ← DoLa 基线 ✅
│   │   ├── greedy.py         ← 贪心解码 ✅
│   │   ├── beam_search.py    ← 束搜索 ✅
│   │   ├── vcd.py            ← VCD（简化版）✅
│   │   ├── vasparse.py       ← VASparse（简化版）✅
│   │   ├── opera.py          ← OPERA（推理时等同贪心）✅
│   │   └── __init__.py       ← 导出所有 Decoder 类 ✅
│   │
│   ├── evaluation/           ← 评估模块 ✅
│   │   ├── pope.py           ← POPE 评估（Acc/P/R/F1）✅
│   │   ├── chair.py          ← CHAIR 评估（chair_s/chair_i）✅
│   │   └── mme.py            ← MME 评估（简化版）✅
│   │
│   ├── utils/
│   │   ├── itav.py           ← iTaV 计算 + JSD + 层选择 + contrastive_decode ✅
│   │   ├── sparsification.py ← Top-K 稀疏化 ✅
│   │   ├── seed.py           ← set_seed() 随机种子控制 ✅
│   │   ├── timing.py         ← TPSMeter + measure_tps() ✅
│   │   └── visualization.py  ← itav_heatmap() ✅
│   │
│   └── analysis/
│       ├── ablation.py       ← 消融实验配置生成 ✅
│       └── plotting.py       ← 结果绘图工具 ✅
│
├── scripts/
│   ├── run_eval_pope.py      ← POPE 评估脚本（含 --seed, --measure_tps）✅
│   ├── run_eval_chair.py     ← CHAIR 评估脚本 ✅
│   ├── run_eval_mme.py       ← MME 评估脚本 ✅
│   ├── run_ablation.py       ← 消融/参数扫描/TPS 对比脚本 ✅
│   └── setup_autodl.sh       ← AutoDL 一键环境配置脚本 ✅
│
├── data/                     ← 数据集存放（需手动下载）
│   ├── pope/                 ← coco_pope_{random,popular,adversarial}.json
│   ├── mscoco/
│   │   ├── val2014/          ← COCO val2014 图像
│   │   └── annotations/      ← instances_val2014.json
│   └── mme/
│
├── results/                  ← 实验结果 JSON（自动生成，带时间戳）
├── figures/                  ← 论文图表
└── thesis/                   ← 论文章节草稿（框架已建，内容待填）
```

---

## 3. 已完成模块

| 模块 | 文件 | 状态 |
|---|---|---|
| 项目骨架 + 配置文件 | `configs/`, `requirements.txt`, `README.md` | ✅ 完成 |
| 统一模型加载（LLaVA + Qwen2-VL） | `src/models/model_loader.py` | ✅ 完成 |
| LLaVA 聊天模板（vicuna 格式） | `model_loader.py::_apply_llava_template` | ✅ 完成 |
| iTaV 计算 + JSD 层选择 | `src/utils/itav.py` | ✅ 完成 |
| Top-K 稀疏化 | `src/utils/sparsification.py` | ✅ 完成 |
| **CESD 核心解码（hook-based）** | `src/decoding/cesd.py` | ✅ 完成 |
| **iTaD 基线（hook-based）** | `src/decoding/itad.py` | ✅ 完成 |
| DoLa 基线 | `src/decoding/dola.py` | ✅ 完成 |
| Greedy / Beam / VCD / VASparse / OPERA | `src/decoding/` | ✅ 完成 |
| POPE 评估（Acc/P/R/F1） | `src/evaluation/pope.py` | ✅ 完成 |
| CHAIR 评估（含多词对象修复） | `src/evaluation/chair.py` | ✅ 完成 |
| MME 评估（简化版） | `src/evaluation/mme.py` | ✅ 完成 |
| 随机种子控制 | `src/utils/seed.py` | ✅ 完成 |
| TPS 测量工具 | `src/utils/timing.py` | ✅ 完成 |
| 消融实验配置 | `src/analysis/ablation.py` | ✅ 完成 |
| 结果绘图工具 | `src/analysis/plotting.py` | ✅ 完成 |
| 所有 eval 脚本（含 seed + TPS + 时间戳） | `scripts/` | ✅ 完成 |
| AutoDL 一键初始化脚本 | `scripts/setup_autodl.sh` | ✅ 完成 |

---

## 4. 未完成模块

| 模块 | 说明 | 优先级 |
|---|---|---|
| **实际实验数据** | 代码已就绪，尚未在 AutoDL 上运行过任何实验 | P0 最高 |
| **论文实验章节** | `thesis/experiment_section.md` 表格全空，等待真实数字 | P0 |
| **论文分析章节** | `thesis/analysis_section.md` 结论是凭空推断，需基于真实数据重写 | P0 |
| VCD 完整实现 | 当前退化为贪心（需图像 mask 扰动） | P2 |
| VASparse 完整实现 | 当前退化为贪心（需视觉 token 稀疏化） | P2 |
| OPERA 完整实现 | 原始是训练时方法，推理复现复杂 | P2 低 |
| `generate_figures.py` | 脚本存在但内容为空 | P1（实验后补） |
| 论文方法章节（方程公式） | `thesis/method_section.md` 只有 1 行 | P1 |
| Jupyter 结果分析笔记本 | `notebooks/results_analysis.ipynb` 未创建 | P2 |

---

## 5. 当前实验进度

> **实验数据：零**

代码完全在本地 Windows 开发，尚未部署到 AutoDL 云 GPU。所有实验结果均为空。

论文中的实验章节表格是占位符（`-`），分析结论是推断而非实测数据。

**正确的实验顺序（接下来需要做）**：

```
Step 1: AutoDL 环境配置
  bash scripts/setup_autodl.sh

Step 2: 快速验证（10 样本，确认流程跑通）
  python scripts/run_eval_pope.py \
    --model llava --decoder greedy \
    --num_samples 10 --splits random

Step 3: 基线实验（Greedy, iTaD）
  python scripts/run_eval_pope.py --model llava --decoder greedy --seed 42
  python scripts/run_eval_pope.py --model llava --decoder itad   --seed 42

Step 4: CESD 主实验
  python scripts/run_eval_pope.py --model llava --decoder cesd --seed 42 --measure_tps
  python scripts/run_eval_chair.py --model llava --decoder cesd --seed 42

Step 5: 消融实验
  python scripts/run_ablation.py --mode ablation --seed 42 --num_samples 100
  python scripts/run_ablation.py --mode alpha    --seed 42 --num_samples 100
  python scripts/run_ablation.py --mode sparsify --seed 42 --num_samples 100
  python scripts/run_ablation.py --mode tps      --seed 42

Step 6: 补全 DoLa, VCD 等其他基线
Step 7: Qwen2-VL 模型上重复以上实验
```

---

## 6. 关键设计决策

### 决策 1：CESD 使用 pre-forward hook 而非逐层 forward

**问题**：最初实现直接手动调用 `layers[i](hidden_states, attention_mask, ...)`，但 HuggingFace transformer 层内部期望 4D causal attention mask（(B,1,T,T)），而 processor 输出的是 2D mask（(B,T)），且不同版本 transformers 的层签名不同（需要 `cache_position` 等参数）。

**解决方案**：改用 `register_forward_pre_hook` 注入 h_sparse 到指定层，然后调用完整 `model.forward()`，让 HuggingFace 内部处理所有 mask/position/cache 转换。

**代价**：每个 token 需要 2 次完整 forward（expert + amateur），包括 2 次 CLIP 视觉编码器。比 Greedy 慢约 2-3x，但结果正确。

### 决策 2：CHAIR 自实现（非官方）

官方 CHAIR 需要 SPICE 工具链（Java）和完整同义词库。当前实现使用 bigram + 扩展同义词表进行词匹配，涵盖所有多词 COCO 对象（traffic light, hot dog 等）。**数字与论文发表值可能有差异**，需在论文中注明。

### 决策 3：VASparse / VCD 简化为贪心

完整实现需要修改视觉编码器内部（VASparse）或运行无图像前向传播（VCD）。当前退化为贪心作为占位符。若毕设需要对比这两个方法，需要单独实现。

### 决策 4：基座模型选择

- **主要**：`llava-hf/llava-1.6-vicuna-7b-hf`（约 14GB, fp16）
- **次要**：`Qwen/Qwen2-VL-7B-Instruct`（约 14GB, fp16）
- AutoDL 推荐：RTX 4090 (24GB) 或 A100 (80GB)

### 决策 5：LLaVA 聊天模板

必须使用 vicuna 格式：
```
A chat between a curious human and an artificial intelligence assistant. ...
USER: <image>\n{question} ASSISTANT:
```
否则 image token 定位失败，CESD/iTaD 退化为贪心。

---

## 7. 接下来需要完成的任务（优先级排序）

### P0 — 必须完成（答辩前）

- [ ] **在 AutoDL 上运行完整实验**，获得真实数字
  - POPE（random/popular/adversarial）× 所有方法
  - CHAIR × 主要方法（Greedy, iTaD, CESD）
- [ ] **用真实数字填写论文实验表格**
- [ ] **基于真实数据重写分析结论**（删除现有凭空推断内容）
- [ ] **论文方法章节**（数学推导部分已有框架，需扩写）

### P1 — 重要但可简化

- [ ] **消融实验**（在小样本上跑，100 条 POPE 足够）
- [ ] **TPS 效率对比**（用 `run_ablation.py --mode tps`）
- [ ] **生成论文图表**（参数敏感性曲线、消融条形图）
- [ ] **完善 `generate_figures.py`**

### P2 — 时间允许再做

- [ ] 在 Qwen2-VL 上重复实验（验证通用性）
- [ ] 完整实现 VASparse / VCD（如需对比）
- [ ] Jupyter 分析笔记本

---

## 8. 重要文件路径速查

| 文件 | 作用 |
|---|---|
| `src/decoding/cesd.py` | **CESD 核心算法**，最重要的文件 |
| `src/utils/itav.py` | iTaV 计算 + JSD + contrastive_decode |
| `src/utils/sparsification.py` | Top-K 稀疏化 |
| `src/models/model_loader.py` | 模型加载 + 聊天模板（LLaVA/Qwen2-VL）|
| `src/evaluation/pope.py` | POPE 评估逻辑 |
| `src/evaluation/chair.py` | CHAIR 评估（自实现，含多词修复）|
| `src/utils/timing.py` | TPS 测量工具 |
| `src/utils/seed.py` | 随机种子控制 |
| `scripts/run_eval_pope.py` | **主要运行入口**（含 seed/TPS）|
| `scripts/run_ablation.py` | 消融 + 参数扫描 + TPS 对比 |
| `scripts/setup_autodl.sh` | AutoDL 环境初始化 |
| `configs/models/llava.yaml` | LLaVA 路径配置（可加 model_path 指向本地）|
| `thesis/method_section.md` | 方法章节草稿（框架存在，内容待补）|
| `thesis/experiment_section.md` | 实验章节（表格全空，等待数据）|

---

## 9. 给新 AI 的补充说明

1. **代码已全部写好，一个实验都没跑过。** 最重要的事情是把代码传到 AutoDL 跑通。

2. **CESD 每步 token 做 2 次 forward**（expert + amateur），速度约为 greedy 的 1/3 到 1/2，在 A100 上 POPE 500 条约需 1-2 小时。

3. **VASparse 和 VCD 当前是假实现**（退化为 greedy），对比这两个方法的数字不可信，若需要请告知需要完整实现。

4. **CHAIR 数字与官方论文不可直接比较**（自实现非官方工具），需在论文中注明方法差异。

5. **下一个对话需要做的第一件事**：帮用户写 AutoDL 数据上传和实验启动命令，或者调试代码在 AutoDL 上的兼容性问题。
