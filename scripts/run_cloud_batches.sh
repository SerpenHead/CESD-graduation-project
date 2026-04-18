#!/usr/bin/env bash
# Cloud execution helper for dual 4090 setup.
# Usage:
#   bash scripts/run_cloud_batches.sh smoke
#   bash scripts/run_cloud_batches.sh main
#   bash scripts/run_cloud_batches.sh ablation
#   bash scripts/run_cloud_batches.sh repro

set -euo pipefail

MODE="${1:-smoke}"

case "${MODE}" in
  smoke)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_matrix.py \
      --models llava \
      --benchmarks pope chair mme \
      --decoders greedy beam dola itad vasparse vcd opera cesd \
      --num_samples 10 --seed 42 --results_root results/matrix_smoke

    CUDA_VISIBLE_DEVICES=1 python scripts/run_matrix.py \
      --models qwen2_vl \
      --benchmarks pope chair mme \
      --decoders greedy beam dola itad vasparse vcd opera cesd \
      --num_samples 10 --seed 42 --results_root results/matrix_smoke
    ;;

  main)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_matrix.py \
      --models llava \
      --benchmarks pope chair mme \
      --decoders greedy beam dola itad vasparse vcd opera cesd \
      --num_samples 500 --seed 42 --results_root results/matrix --resume --skip_existing

    CUDA_VISIBLE_DEVICES=1 python scripts/run_matrix.py \
      --models qwen2_vl \
      --benchmarks pope chair mme \
      --decoders greedy beam dola itad vasparse vcd opera cesd \
      --num_samples 500 --seed 42 --results_root results/matrix --resume --skip_existing

    python scripts/aggregate_results.py --results_root results/matrix --output results/matrix_summary.json
    ;;

  ablation)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py --model llava --mode ablation --num_samples 100 --output_dir results/ablation/llava
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py --model llava --mode alpha --num_samples 100 --output_dir results/ablation/llava
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py --model llava --mode sparsify --num_samples 100 --output_dir results/ablation/llava
    CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py --model llava --mode opera --num_samples 100 --output_dir results/ablation/llava

    CUDA_VISIBLE_DEVICES=1 python scripts/run_ablation.py --model qwen2_vl --mode ablation --num_samples 100 --output_dir results/ablation/qwen2_vl
    CUDA_VISIBLE_DEVICES=1 python scripts/run_ablation.py --model qwen2_vl --mode alpha --num_samples 100 --output_dir results/ablation/qwen2_vl
    CUDA_VISIBLE_DEVICES=1 python scripts/run_ablation.py --model qwen2_vl --mode sparsify --num_samples 100 --output_dir results/ablation/qwen2_vl
    CUDA_VISIBLE_DEVICES=1 python scripts/run_ablation.py --model qwen2_vl --mode opera --num_samples 100 --output_dir results/ablation/qwen2_vl
    ;;

  repro)
    for s in 13 42 123; do
      CUDA_VISIBLE_DEVICES=0 python scripts/run_matrix.py \
        --models llava \
        --benchmarks pope chair mme \
        --decoders greedy itad opera cesd \
        --num_samples 500 --seed "${s}" --results_root results/matrix_repro --resume --skip_existing
    done

    for s in 13 42 123; do
      CUDA_VISIBLE_DEVICES=1 python scripts/run_matrix.py \
        --models qwen2_vl \
        --benchmarks pope chair mme \
        --decoders greedy itad opera cesd \
        --num_samples 500 --seed "${s}" --results_root results/matrix_repro --resume --skip_existing
    done

    python scripts/aggregate_results.py --results_root results/matrix_repro --output results/matrix_repro_summary.json
    ;;

  *)
    echo "Usage: bash scripts/run_cloud_batches.sh [smoke|main|ablation|repro]"
    exit 1
    ;;
esac
