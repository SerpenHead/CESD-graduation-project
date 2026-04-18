#!/usr/bin/env python3
"""Unified experiment matrix runner for POPE / CHAIR / MME."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent

ALL_BENCHMARKS = ["pope", "chair", "mme"]
ALL_MODELS = ["llava", "qwen2_vl"]
ALL_DECODERS = ["greedy", "beam", "dola", "itad", "vasparse", "vcd", "opera", "cesd"]


def parse_model_device_map(raw: str) -> Dict[str, str]:
    """
    Parse "llava:0,qwen2_vl:1" -> {"llava": "0", "qwen2_vl": "1"}.
    """
    out: Dict[str, str] = {}
    if not raw.strip():
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid model-device mapping: {part}")
        model, dev = part.split(":", 1)
        out[model.strip()] = dev.strip()
    return out


def build_command(
    python_bin: str,
    benchmark: str,
    model: str,
    decoder: str,
    seed: int,
    num_samples: int,
    data_root: Path,
    out_path: Path,
) -> List[str]:
    if benchmark == "pope":
        return [
            python_bin,
            str(ROOT / "scripts" / "run_eval_pope.py"),
            "--model", model,
            "--decoder", decoder,
            "--seed", str(seed),
            "--num_samples", str(num_samples),
            "--splits", "random", "popular", "adversarial",
            "--data_path", str(data_root / "pope"),
            "--coco_root", str(data_root / "mscoco" / "val2014"),
            "--output", str(out_path),
        ]
    if benchmark == "chair":
        return [
            python_bin,
            str(ROOT / "scripts" / "run_eval_chair.py"),
            "--model", model,
            "--decoder", decoder,
            "--seed", str(seed),
            "--num_samples", str(num_samples),
            "--data_path", str(data_root / "mscoco"),
            "--output", str(out_path),
        ]
    if benchmark == "mme":
        return [
            python_bin,
            str(ROOT / "scripts" / "run_eval_mme.py"),
            "--model", model,
            "--decoder", decoder,
            "--seed", str(seed),
            "--num_samples", str(num_samples),
            "--data_path", str(data_root / "mme"),
            "--output", str(out_path),
        ]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def main():
    parser = argparse.ArgumentParser(description="Run full experiment matrix")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--decoders", nargs="+", default=ALL_DECODERS, choices=ALL_DECODERS)
    parser.add_argument("--benchmarks", nargs="+", default=ALL_BENCHMARKS, choices=ALL_BENCHMARKS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--results_root", default="results/matrix")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES for all runs")
    parser.add_argument(
        "--model_device_map",
        default="",
        help="Per-model device map, e.g. llava:0,qwen2_vl:1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    model_device_map = parse_model_device_map(args.model_device_map)
    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    total = len(args.models) * len(args.decoders) * len(args.benchmarks)
    done = 0
    skipped = 0

    print(f"[matrix] total jobs = {total}")
    for model in args.models:
        for benchmark in args.benchmarks:
            for decoder in args.decoders:
                out_path = results_root / benchmark / model / decoder / f"seed_{args.seed}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_path.exists() and (args.skip_existing or args.resume):
                    skipped += 1
                    print(f"[matrix] skip existing: {out_path}")
                    continue

                cmd = build_command(
                    python_bin=args.python_bin,
                    benchmark=benchmark,
                    model=model,
                    decoder=decoder,
                    seed=args.seed,
                    num_samples=args.num_samples,
                    data_root=data_root,
                    out_path=out_path,
                )

                env = os.environ.copy()
                model_dev = model_device_map.get(model)
                if model_dev is not None:
                    env["CUDA_VISIBLE_DEVICES"] = model_dev
                elif args.device is not None:
                    env["CUDA_VISIBLE_DEVICES"] = args.device

                print(f"[matrix] run: model={model} benchmark={benchmark} decoder={decoder}")
                print(f"[matrix] cmd: {' '.join(cmd)}")
                if args.dry_run:
                    continue

                subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
                done += 1

    print(f"[matrix] done={done} skipped={skipped} total={total}")


if __name__ == "__main__":
    main()
