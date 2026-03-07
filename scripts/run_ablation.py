#!/usr/bin/env python3
"""
Run CESD ablation study and parameter sensitivity experiments.

Modes:
  ablation  – remove dynamic layer selection / sparsification independently
  alpha     – sweep alpha (contrast strength)
  sparsify  – sweep sparsify_ratio (fraction of tokens kept)
  tps       – measure TPS for all decoders on one image

Example:
    python scripts/run_ablation.py --mode ablation --seed 42 --num_samples 100
    python scripts/run_ablation.py --mode alpha     --seed 42 --num_samples 50
    python scripts/run_ablation.py --mode tps       --seed 42
"""

import argparse
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.timing import measure_tps
from src.utils.runtime import get_inference_device, move_inputs_to_device
from src.models.model_loader import load_model, get_model_config, prepare_inputs
from src.evaluation.pope import POPEEvaluator
from src.decoding import (
    GreedyDecoder, BeamSearchDecoder, CESDDecoder,
    ITaDDecoder, DoLaDecoder, VASparseDecoder, OPERADecoder,
)
from src.analysis.ablation import (
    get_ablation_configs,
    get_alpha_sweep_configs,
    get_sparsify_sweep_configs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="llava")
    parser.add_argument("--mode",        default="ablation",
                        choices=["ablation", "alpha", "sparsify", "tps"])
    parser.add_argument("--data_path",   default="data/pope")
    parser.add_argument("--coco_root",   default="data/mscoco/val2014")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="POPE samples per config (keep small for speed)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  default="results/ablation")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation] Loading model: {args.model}")
    model, processor = load_model(args.model)
    config     = get_model_config(args.model)
    model_type = config.get("model_type", args.model)

    # ── TPS mode: no POPE needed ─────────────────────────────────────────────
    if args.mode == "tps":
        imgs = sorted(Path(args.coco_root).glob("*.jpg"))
        if not imgs:
            print("No images found in coco_root for TPS measurement.")
            return
        inputs = prepare_inputs(processor, str(imgs[0]), "Is there a cat?", model_type)
        inputs = move_inputs_to_device(inputs, get_inference_device())

        all_decoders = {
            "Greedy":   GreedyDecoder(),
            "Beam-5":   BeamSearchDecoder(beam_size=5),
            "DoLa":     DoLaDecoder(alpha=0.1),
            "iTaD":     ITaDDecoder(alpha=0.5, model_type=model_type),
            "VASparse": VASparseDecoder(keep_ratio=0.5),
            "OPERA":    OPERADecoder(),
            "CESD":     CESDDecoder(alpha=0.5, sparsify_ratio=0.2, model_type=model_type),
        }

        tps_results = {}
        for name, dec in all_decoders.items():
            print(f"  Measuring TPS for {name}...")
            info = measure_tps(dec, model, inputs, max_new_tokens=32,
                               n_warmup=1, n_runs=3)
            tps_results[name] = info
            print(f"    → {info['tps_mean']:.2f} ± {info['tps_std']:.2f} tok/s")

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"tps_{args.model}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "seed": args.seed,
                       "tps": tps_results}, f, indent=2)
        print(f"Saved → {out_path}")
        return

    # ── POPE-based ablation/sweep modes ──────────────────────────────────────
    if args.mode == "ablation":
        configs = get_ablation_configs()
    elif args.mode == "alpha":
        configs = get_alpha_sweep_configs()
    else:
        configs = get_sparsify_sweep_configs()

    evaluator = POPEEvaluator(
        data_path=args.data_path,
        coco_root=args.coco_root,
        splits=["random"],
        num_samples=args.num_samples,
    )

    all_results = {}
    for cfg in configs:
        name    = cfg.get("name", "config")
        cfg_kw  = {k: v for k, v in cfg.items() if k != "name"}
        decoder = CESDDecoder(model_type=model_type, **cfg_kw)

        print(f"\n[{name}] {cfg_kw}")

        def make_fn(d):
            def fn(m, **kw):
                return d(m, **kw)
            return fn

        try:
            res = evaluator.evaluate(
                model, processor, make_fn(decoder),
                model_type=model_type, splits=["random"],
            )
            all_results[name] = res["random"]
            with open(out_dir / f"{name}.json", "w") as f:
                json.dump({"config_name": name, "config": cfg_kw,
                           "seed": args.seed, "results": res}, f, indent=2)
        except Exception as e:
            print(f"  Error: {e}")
            all_results[name] = {}

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"{args.mode}_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({"model": args.model, "mode": args.mode,
                   "seed": args.seed, "results": all_results}, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Ablation Summary ({args.mode}):")
    for name, r in all_results.items():
        f1  = r.get("f1", 0)
        acc = r.get("accuracy", 0)
        print(f"  {name:30s}  Acc={acc:.4f}  F1={f1:.4f}")
    print(f"Saved → {summary_path}")


if __name__ == "__main__":
    main()
