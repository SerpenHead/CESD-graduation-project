# CESD: 对比度增强稀疏化解码

无需训练的高效多模态大模型幻觉缓解算法研究

## 项目简介

本项目实现了 **CESD (Contrast-Enhanced Sparsified Decoding)** 算法，用于缓解视觉语言模型（VLM）的幻觉问题。该方法结合了 iTaD 的动态层选择与 VASparse 的稀疏化思想，在解码阶段通过"专家-业余"对比解码抑制幻觉，无需额外训练。

## 环境配置

```bash
# 推荐使用 AutoDL 云 GPU (A100/RTX 4090)
pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger
```

## 项目结构

```
├── configs/          # 实验配置
├── src/              # 核心代码
│   ├── models/       # 模型加载
│   ├── decoding/     # 解码策略 (CESD, iTaD, DoLa, ...)
│   ├── evaluation/   # 评估模块 (POPE, CHAIR, MME)
│   ├── utils/        # 工具函数
│   └── analysis/     # 结果分析
├── scripts/          # 运行脚本
├── results/          # 实验结果
└── figures/          # 论文图表
```

## 快速开始

```bash
# POPE 评估 (最快出结果)
python scripts/run_eval_pope.py --model llava --decoder cesd

# CHAIR 评估
python scripts/run_eval_chair.py --model llava --decoder cesd

# 消融实验
python scripts/run_ablation.py --model llava
```

## 支持的模型

- LLaVA-1.6-vicuna-7b
- Qwen2-VL-7B-Instruct

## 支持的解码策略

- Greedy / Beam Search
- DoLa, VCD, iTaD, VASparse, OPERA
- **CESD (ours)**

## 引用

```bibtex
@mastersthesis{xu2026cesd,
  title={无需训练的高效多模态大模型幻觉缓解算法研究},
  author={徐景元},
  school={哈尔滨工业大学深圳校区},
  year={2026}
}
```
