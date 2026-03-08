# 离开 AutoDL / 换机前检查清单

> 选择**新机重下**数据与模型时，按此清单操作即可。

---

## 一、已在 Git 里（新机 clone 即有）

- **代码**：`src/`、`scripts/`、`configs/`、`run.sh` 等
- **文档**：`README.md`、`LINUX_RUN.md`、`docs/FIXES_CESD_ATTENTION_AND_OOM.md`、本清单
- **实验结果**（已提交）：POPE/CHAIR 的 greedy、cesd、itad 的 `results/*.json`
- **依赖**：`requirements.txt`

数据与模型**不入库**，新机上一律重新下载。

---

## 二、离机前（AutoDL 上）——1 分钟自检

- [ ] 重要改动已提交并 push：`git status` 无未提交内容（或已 `git add` / `commit` / `push`）
- [ ] 需要保留的 `results/*.json` 已在仓库里（当前主要结果已提交）

**无需**在 AutoDL 上备份 data/ 或模型，直接关机即可。

---

## 三、新机重下——操作步骤

在新 GPU 机器上按顺序执行即可。

### 1. 克隆仓库

```bash
git clone https://github.com/SerpenHead/CESD-graduation-project.git
cd CESD-graduation-project
```

### 2. 设置数据与缓存路径（按你新机习惯改）

```bash
# 数据与模型将下载到该目录下（请改成你新机的路径）
export DATA_ROOT=/你的数据目录
export HF_HOME="${HF_HOME:-$DATA_ROOT/../hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
```

例如：`export DATA_ROOT=$HOME/data`，则数据会在 `~/data/`，HF 缓存在 `~/hf-cache/`。

### 3. 国内网络可设 HuggingFace 镜像（可选）

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 4. 一键安装 + 下载数据与模型

```bash
bash run.sh setup
```

会完成：安装 `requirements.txt`、下载 POPE、COCO val2014 与标注、下载 LLaVA-1.6 模型，并在项目下创建 `data` 软链到 `$DATA_ROOT`。

### 5. 模型路径（新机若没有 /root/autodl-tmp）

当前 `scripts/setup_autodl.sh` 会把模型下到 **`/root/autodl-tmp/models/llava-1.6-vicuna-7b`**。新机若没有该目录或不想用该路径，请**先**编辑再跑 setup：

- 打开 `scripts/setup_autodl.sh`
- 第 11 行：`mkdir -p` 的模型目录改成你新机路径（例如 `"$DATA_ROOT/../models"`）
- 第 117 行：`local_llava = "..."` 改成新机上的**绝对路径**（例如 `"/home/你/数据/models/llava-1.6-vicuna-7b"`），并先 `mkdir -p` 该目录的父级

然后执行 `bash run.sh setup`。最后把 **`configs/models/llava.yaml`** 里的 `model_path` 改成与上面一致的**绝对路径**（例如 `/你的目录/models/llava-1.6-vicuna-7b`）。

### 6. NLTK 数据（CHAIR 需要）

```bash
python -m nltk.downloader punkt averaged_perceptron_tagger
```

### 7. 验证

```bash
bash run.sh check llava
bash run.sh quick llava cesd
```

无报错且 quick 跑完即说明环境与数据、模型均正常。

### 8. 后续正式跑

```bash
# POPE
bash run.sh pope llava cesd

# CHAIR
bash run.sh chair llava cesd
```

若遇 OOM，见 `docs/FIXES_CESD_ATTENTION_AND_OOM.md` 中的显存应对。

---

## 四、小结

| 阶段     | 动作 |
|----------|------|
| **离机前** | 确认 git 已 push，无需备份 data/ 模型 |
| **新机**   | clone → 设 `DATA_ROOT`（及可选镜像）→ `bash run.sh setup` → 改 `llava.yaml` 的 `model_path`（如需）→ nltk 下载 → check + quick 验证 → 正式实验 |

按上述做即可在新机用「新机重下」方式继续工作。
