#!/usr/bin/env python3
"""
Run POPE evaluation for all decoders.

Example:
    python scripts/run_eval_pope.py --model llava --decoder cesd --seed 42
    python scripts/run_eval_pope.py --model llava --decoder greedy --num_samples 500
"""

import argparse
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.seed import set_seed
from src.utils.runtime import get_inference_device, move_inputs_to_device
from src.models.model_loader import load_model, get_model_config, prepare_inputs
from src.evaluation.pope import POPEEvaluator
from src.decoding import (
    GreedyDecoder, BeamSearchDecoder,
    CESDDecoder, ITaDDecoder,
    DoLaDecoder, VASparseDecoder, OPERADecoder,
)


def build_decoder(name: str, model_type: str):
    return {
        "greedy":    GreedyDecoder(),
        "beam":      BeamSearchDecoder(beam_size=5),
        "dola":      DoLaDecoder(alpha=0.1),
        "itad":      ITaDDecoder(alpha=0.5, model_type=model_type),
        "vasparse":  VASparseDecoder(keep_ratio=0.5),
        "opera":     OPERADecoder(),
        "cesd":      CESDDecoder(alpha=0.5, sparsify_ratio=0.2, model_type=model_type),
    }[name]


def main():
    parser = argparse.ArgumentParser(description="POPE evaluation")
    parser.add_argument("--model",       default="llava",
                        choices=["llava", "qwen2_vl"])
    parser.add_argument("--decoder",     default="cesd",
                        choices=["greedy", "beam", "dola", "itad",
                                 "vasparse", "opera", "cesd"])
    parser.add_argument("--data_path",   default="data/pope")
    parser.add_argument("--coco_root",   default="data/mscoco/val2014")
    parser.add_argument("--splits",      nargs="+",
                        default=["random", "popular", "adversarial"])
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--measure_tps", action="store_true",
                        help="Measure tokens-per-second on first sample")
    parser.add_argument("--output",      default=None,
                        help="Output JSON path (auto-generated if not set)")
    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    set_seed(args.seed)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"[POPE] Loading model: {args.model}")
    model, processor = load_model(args.model)
    config      = get_model_config(args.model)
    model_type  = config.get("model_type", args.model)
    decoder     = build_decoder(args.decoder, model_type)

    # ── Optional TPS measurement (one image, 3 runs) ─────────────────────────
    tps_info = {}
    if args.measure_tps:
        sample_imgs = sorted(Path(args.coco_root).glob("*.jpg"))
        if sample_imgs:
            img_path = sample_imgs[0]
            inputs = prepare_inputs(processor, str(img_path), "Is there a cat?", model_type)
            inputs = move_inputs_to_device(inputs, get_inference_device())
            from src.utils.timing import measure_tps
            tps_info = measure_tps(decoder, model, inputs,
                                   max_new_tokens=16, n_warmup=1, n_runs=3)
            print(f"[TPS] {args.decoder}: {tps_info['tps_mean']:.2f} ± "
                  f"{tps_info['tps_std']:.2f} tok/s")

    # ── POPE evaluation ───────────────────────────────────────────────────────
    evaluator = POPEEvaluator(
        data_path=args.data_path,
        coco_root=args.coco_root,
        splits=args.splits,
        num_samples=args.num_samples,
    )

    def decode_fn(m, **kw):
        return decoder(m, **kw)

    results = evaluator.evaluate(
        model, processor, decode_fn,
        model_type=model_type,
        splits=args.splits,
    )

    # ── Save results ─────────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output or f"results/pope_{args.model}_{args.decoder}_{ts}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":      args.model,
        "decoder":    args.decoder,
        "seed":       args.seed,
        "num_samples": args.num_samples,
        "tps":        tps_info,
        "results":    results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nPOPE Results:")
    for split, m in results.items():
        print(f"  {split:12s}  Acc={m['accuracy']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
