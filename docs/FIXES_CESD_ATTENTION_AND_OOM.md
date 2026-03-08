# CESD 实验结果异常与 OOM 问题：问题总结与修复说明

> 文档记录时间：2026-03-08  
> 用于留存「实验结果与 Greedy 几乎一致」的原因分析、已做修复，以及 OOM 应对方案。

---

## 1. 现象：CESD/iTaD 与 Greedy 结果几乎一致

在 POPE 与 CHAIR 上，使用 CESD 或 iTaD 解码时，得到的指标与 Greedy 解码非常接近甚至完全相同，例如：

- **POPE**：Greedy / iTaD / CESD 在 random、popular 上数值完全一致；adversarial 上仅差 0.2% 左右。
- **CHAIR**：CESD 与 Greedy 的 CHAIR_s 相同，CHAIR_i 仅略优（约 0.1797 vs 0.1811）。

这容易让人怀疑实验设计有误。经排查，**根因在于实现/配置，而非实验设计**。

---

## 2. 根因：默认 SDPA 下未返回有效注意力，CESD/iTaD 全程退化为 Greedy

### 2.1 技术背景

- **SDPA（Scaled Dot Product Attention）**：PyTorch/Transformers 默认使用的融合注意力实现，**不支持** `output_attentions=True` 返回每层注意力权重。
- **CESD / iTaD**：依赖 `output_attentions=True` 和 `output_hidden_states=True` 计算 iTaV、选层、稀疏化与对比解码。若拿不到有效注意力，代码会 **fallback 到 Greedy**（直接取 expert logits 的 argmax）。

### 2.2 实际行为

- 使用 **默认配置** 加载 LLaVA（未指定 `attn_implementation`）时，模型使用 **SDPA**。
- 调用 `model(..., output_attentions=True)` 时：
  - 控制台会出现警告：`sdpa attention does not support output_attentions=True. Please set your attention to eager`。
  - 在当前使用的 Transformers 版本中，**返回的 `attentions` 不是 `None`，而是长度为 0 的元组**（`len(attentions) == 0`）。
- CESD/iTaD 中的判断原先仅写为 `if attentions is None or ...`，**未考虑空元组**。若未做后续修复，会继续用 `attentions[0]` 等索引，导致异常；在增加对 `len(attentions)==0` 的 fallback 后，**每一步都会走 fallback 分支**，即 **等价于全程 Greedy 解码**。

因此：**在此前的实验配置下，CESD 和 iTaD 实际上并没有执行对比解码，只是以 Greedy 方式解码，所以结果与 Greedy 一致。**

---

## 3. 已做修复

### 3.1 评测时对 CESD/iTaD 使用 eager 注意力

- **`src/models/model_loader.py`**  
  - 增加参数 `attn_implementation`，并传入 `from_pretrained(..., **kwargs)`。  
  - 当调用方传入 `attn_implementation="eager"` 时，模型使用可返回注意力权重的实现。

- **`scripts/run_eval_pope.py`**、**`scripts/run_eval_chair.py`**  
  - 当 `decoder in ("cesd", "itad")` 时，调用  
    `load_model(..., attn_implementation="eager")`。  
  - 保证跑 CESD/iTaD 时一定能拿到有效 `attentions`，对比解码真正生效。

### 3.2 解码器内对「无有效注意力」的 fallback

- **`src/decoding/cesd.py`**  
  - 在 expert 步后，除原有 `attentions is None` 等判断外，增加：  
    - `len(attentions)==0`（SDPA 返回空元组）；  
    - `len(hidden_states)<=1`（层输出不足）。  
  - 满足任一条件即走 fallback（`next_token = expert_logits.argmax(...)`），避免索引越界或误用无效数据。

- **`src/decoding/itad.py`**  
  - 同样增加对 `len(expert_out.attentions)==0` 的 fallback，逻辑与 CESD 一致。

这样即便将来某次误用默认 SDPA 跑 CESD/iTaD，也会安全退化为 Greedy，而不会崩溃。

### 3.3 验证脚本与统计

- **`scripts/verify_attention_output.py`**  
  - 用默认配置加载模型，做一次 `forward(..., output_attentions=True)`，检查 `attentions` 是否为 `None` 或 `len(attentions)==0`。  
  - 可选：用默认模型跑若干条 POPE，统计 CESD 的 **contrastive 步数** 与 **fallback 步数**。  
  - 若默认配置下 fallback 比例接近 100%，即可确认「之前结果一致是因为 CESD 未真正运行」。

- **`src/decoding/cesd.py`、`src/decoding/itad.py`**  
  - 增加 `_stats` 字典（`contrastive` / `fallback` 计数）与 **`get_and_reset_stats()`**，便于在验证或调试时统计每轮生成中有多少步是对比解码、多少步是 fallback。

### 3.4 文档与使用说明

- **`LINUX_RUN.md`**  
  - 新增「CESD/iTaD 注意力验证」小节，说明如何运行 `verify_attention_output.py` 以及如何理解「attentions 层数: 0」「fallback 比例: 100%」的含义。  
  - 说明评测脚本已对 cesd/itad 自动使用 `attn_implementation="eager"`。

---

## 4. 对既有实验结果的含义

- **在修复前**、使用默认配置（SDPA）跑出的 **CESD / iTaD 结果**：  
  实际等价于 **Greedy 解码**，不能代表 CESD/iTaD 的真实效果。  
- **若论文或报告要使用 CESD/iTaD 的数值**，需要 **在修复后、使用当前代码与配置重新跑一遍** POPE/CHAIR（以及其它相关评测），以保证使用的是「真正打开对比解码」的结果。

---

## 5. OOM（显存不足）问题与应对

### 5.1 原因

使用 `attn_implementation="eager"` 后：

- 每次前向都会计算并保存 **每层** 的注意力矩阵 `(B, num_heads, T, T)`，显存占用明显高于 SDPA。
- CESD 每个 token 做 **两次** 前向（expert + amateur），且 expert 步带有 `output_hidden_states=True` 和 `output_attentions=True`，显存峰值进一步增大。

在 24GB 显存（如 RTX 4090）上，部分样本（序列较长或显存碎片化）可能触发 **CUDA out of memory**。

### 5.2 已建议的应对（可按需实施）

1. **环境变量**（减轻碎片）：  
   `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后再运行评测。

2. **POPE 评测**：  
   - 将 `src/evaluation/pope.py` 中调用 `decode_fn` 时的 `max_new_tokens=16` 改为 `8`（或更小），以缩短序列、降低 attention 显存。
   - 在 `for item in tqdm(...)` 循环内，每处理完一个样本后调用 `torch.cuda.empty_cache()`，减轻长时间运行后的碎片。

3. **CESD 解码器**：  
   - 在 expert 步中，用完 `hidden_states` / `attentions` 得到 `h_sparse`、`m_star` 后，在调用 `_run_amateur_forward` 前对不再引用的大张量执行 `del`，必要时可配合 `torch.cuda.empty_cache()`。

4. **硬件**：  
   - 使用显存更大的 GPU（如 A100 40GB）可显著降低 OOM 概率。

若 OOM 仅发生在极少数样本，评测脚本会捕获异常、将该样本预测为默认值（如 "no"）并继续，最终仍会生成结果文件，但报告中可注明存在少量 OOM 样本及处理方式。

---

## 6. 重跑 CESD 评测的命令（修复后）

在项目根目录、且已设置 `DATA_ROOT`（如 `/root/autodl-tmp/data`）时：

```bash
# POPE（三个 split，各 500 条）
bash run.sh pope llava cesd

# CHAIR（500 张图）
bash run.sh chair llava cesd
```

如需先缓解 OOM，可先执行：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

再运行上述命令。

---

## 7. 小结

| 问题 | 原因 | 修复/结论 |
|------|------|-----------|
| CESD/iTaD 与 Greedy 结果几乎一致 | 默认 SDPA 不返回有效 attentions，解码器每一步 fallback 到 Greedy | 对 cesd/itad 使用 `attn_implementation="eager"` 加载；解码器内对空 attentions 做 fallback；用验证脚本确认 |
| 此前 CESD 结果是否可用 | 修复前跑出的 CESD/iTaD 结果实为 Greedy | 需用当前代码与配置重跑 POPE/CHAIR 等，才能得到真实 CESD 结果 |
| 重跑时 OOM | eager 注意力 + 双前向导致显存峰值升高 | 环境变量、减小 max_new_tokens、每样本 empty_cache、必要时更大显存 GPU |

本文档与对应代码修改一并留存，便于后续复现与排查。
