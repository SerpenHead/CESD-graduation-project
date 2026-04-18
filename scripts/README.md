# Scripts Guide

This folder is intentionally trimmed to only keep scripts used by the cloud experiment workflow.

## Kept Scripts

1. `bootstrap_fresh_machine.sh`
- Purpose: one-shot cloud bootstrap (models + datasets).
- Used in plan: `T1`.
- Example:
```bash
bash scripts/bootstrap_fresh_machine.sh all
```

2. `run_matrix.py`
- Purpose: run full benchmark matrix (`pope/chair/mme`) across model + decoder combinations.
- Used in plan: `T2`, `T3`, `T5`.
- Example:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_matrix.py \
  --models llava \
  --benchmarks pope chair mme \
  --decoders greedy beam dola itad vasparse vcd opera cesd \
  --num_samples 500 --seed 42 --results_root results/matrix --resume --skip_existing
```

3. `run_eval_pope.py`
- Purpose: single-run POPE evaluation entry.
- Called by: `run_matrix.py`.

4. `run_eval_chair.py`
- Purpose: single-run CHAIR evaluation entry.
- Called by: `run_matrix.py`.

5. `run_eval_mme.py`
- Purpose: single-run MME evaluation entry.
- Called by: `run_matrix.py`.

6. `run_ablation.py`
- Purpose: CESD ablation/sensitivity/TPS experiments.
- Used in plan: `T4`.
- Example:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py \
  --model llava --mode ablation --num_samples 100 --output_dir results/ablation/llava
```

7. `aggregate_results.py`
- Purpose: aggregate matrix outputs to summary JSON.
- Used in plan: `T6`.
- Example:
```bash
python scripts/aggregate_results.py \
  --results_root results/matrix \
  --output results/matrix_summary.json
```

8. `run_cloud_batches.sh`
- Purpose: convenience wrapper for batch modes (`smoke/main/ablation/repro`).
- Optional helper when you want one command per phase.

## Typical Cloud Execution Order

1. Bootstrap:
```bash
bash scripts/bootstrap_fresh_machine.sh all
```

2. Smoke matrix:
```bash
bash scripts/run_cloud_batches.sh smoke
```

3. Main matrix:
```bash
bash scripts/run_cloud_batches.sh main
```

4. Ablation:
```bash
bash scripts/run_cloud_batches.sh ablation
```

5. Reproducibility (multi-seed):
```bash
bash scripts/run_cloud_batches.sh repro
```

## Notes

- `run_matrix.py` writes one JSON per benchmark/model/decoder/seed.
- `run_ablation.py` writes one JSON per config plus a timestamped summary.
- `aggregate_results.py` reads the latest `seed_*.json` under each decoder directory.

## TMUX Commands (Cloud)

Save logs and run long jobs in detachable sessions:

```bash
cd /root/autodl-tmp/CESD-graduation-project
mkdir -p logs
```

1. Bootstrap once:
```bash
tmux new -d -s bootstrap 'cd /root/autodl-tmp/CESD-graduation-project && bash scripts/bootstrap_fresh_machine.sh all 2>&1 | tee logs/bootstrap.log'
```

2. Smoke tests on dual GPUs:
```bash
tmux new -d -s smoke_llava 'cd /root/autodl-tmp/CESD-graduation-project && CUDA_VISIBLE_DEVICES=0 python scripts/run_matrix.py --models llava --benchmarks pope chair mme --decoders greedy beam dola itad vasparse vcd opera cesd --num_samples 10 --seed 42 --results_root results/matrix_smoke 2>&1 | tee logs/smoke_llava.log'
tmux new -d -s smoke_qwen  'cd /root/autodl-tmp/CESD-graduation-project && CUDA_VISIBLE_DEVICES=1 python scripts/run_matrix.py --models qwen2_vl --benchmarks pope chair mme --decoders greedy beam dola itad vasparse vcd opera cesd --num_samples 10 --seed 42 --results_root results/matrix_smoke 2>&1 | tee logs/smoke_qwen.log'
```

3. Main full matrix on dual GPUs:
```bash
tmux new -d -s main_llava 'cd /root/autodl-tmp/CESD-graduation-project && CUDA_VISIBLE_DEVICES=0 python scripts/run_matrix.py --models llava --benchmarks pope chair mme --decoders greedy beam dola itad vasparse vcd opera cesd --num_samples 500 --seed 42 --results_root results/matrix --resume --skip_existing 2>&1 | tee logs/main_llava.log'
tmux new -d -s main_qwen  'cd /root/autodl-tmp/CESD-graduation-project && CUDA_VISIBLE_DEVICES=1 python scripts/run_matrix.py --models qwen2_vl --benchmarks pope chair mme --decoders greedy beam dola itad vasparse vcd opera cesd --num_samples 500 --seed 42 --results_root results/matrix --resume --skip_existing 2>&1 | tee logs/main_qwen.log'
```

4. Ablation batch:
```bash
tmux new -d -s ablation 'cd /root/autodl-tmp/CESD-graduation-project && bash scripts/run_cloud_batches.sh ablation 2>&1 | tee logs/ablation.log'
```

5. Aggregate outputs:
```bash
tmux new -d -s aggregate 'cd /root/autodl-tmp/CESD-graduation-project && python scripts/aggregate_results.py --results_root results/matrix --output results/matrix_summary.json 2>&1 | tee logs/aggregate.log'
```

Common session management:

```bash
tmux ls
tmux attach -t main_llava
# detach: Ctrl+b, then d
tmux kill-session -t main_llava
```
