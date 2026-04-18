#!/usr/bin/env python3
"""
Run CHAIR evaluation.

Example:
    python scripts/run_eval_chair.py --model llava --decoder cesd --seed 42
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
from src.utils.timing import measure_tps
from src.models.model_loader import load_model, get_model_config, prepare_inputs
from src.evaluation.chair import CHAIREvaluator
from src.decoding import (
    GreedyDecoder, BeamSearchDecoder, DoLaDecoder, ITaDDecoder,
    VASparseDecoder, VCDDecoder, OPERADecoder, CESDDecoder,
)


def build_decoder(name: str, model_type: str):
    return {
        "greedy": GreedyDecoder(),
        "beam": BeamSearchDecoder(beam_size=5),
        "dola": DoLaDecoder(alpha=0.1),
        "itad": ITaDDecoder(alpha=0.5, model_type=model_type),
        "vasparse": VASparseDecoder(keep_ratio=0.5, model_type=model_type),
        "vcd": VCDDecoder(alpha=0.5, noise_std=0.05),
        "opera": OPERADecoder(model_type=model_type),
        "cesd": CESDDecoder(alpha=0.5, sparsify_ratio=0.2, model_type=model_type),
    }[name]


def decode_stats(decoder) -> dict:
    if not hasattr(decoder, "get_and_reset_stats"):
        return {"contrastive": 0, "fallback": 0, "fallback_ratio": 0.0}
    stats = decoder.get_and_reset_stats()
    c = int(stats.get("contrastive", 0))
    f = int(stats.get("fallback", 0))
    total = c + f
    stats["fallback_ratio"] = (float(f) / float(total)) if total > 0 else 0.0
    return stats


def main():
    parser = argparse.ArgumentParser(description="CHAIR evaluation")
    parser.add_argument("--model",       default="llava",
                        choices=["llava", "qwen2_vl"])
    parser.add_argument("--decoder",     default="cesd",
                        choices=["greedy", "beam", "dola", "itad", "vasparse", "vcd", "opera", "cesd"])
    parser.add_argument("--data_path",   default="data/mscoco")
    parser.add_argument("--annot_path",  default=None,
                        help="Path to instances_val2014.json (auto-detected if not set)")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--measure_tps", action="store_true",
                        help="Measure tokens-per-second on first image")
    parser.add_argument("--prompt",      default="Describe this image in detail.",
                        help="Caption instruction sent to the model")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    need_attn = args.decoder in ("cesd", "itad", "opera", "vasparse")
    print(f"[CHAIR] Loading model: {args.model}" + (" (attn_implementation=eager)" if need_attn else ""))
    model, processor = load_model(args.model, attn_implementation="eager" if need_attn else None)
    config      = get_model_config(args.model)
    model_type  = config.get("model_type", args.model)

    decoder = build_decoder(args.decoder, model_type)

    tps_info = {}
    if args.measure_tps:
        sample_imgs = sorted((Path(args.data_path) / "val2014").glob("*.jpg"))
        if sample_imgs:
            inputs = prepare_inputs(processor, str(sample_imgs[0]), args.prompt, model_type)
            inputs = move_inputs_to_device(inputs, get_inference_device())
            tps_info = measure_tps(decoder, model, inputs, max_new_tokens=32, n_warmup=1, n_runs=3)
            print(f"[TPS] {args.decoder}: {tps_info['tps_mean']:.2f} ± {tps_info['tps_std']:.2f} tok/s")
            if hasattr(decoder, "get_and_reset_stats"):
                decoder.get_and_reset_stats()

    evaluator = CHAIREvaluator(
        data_path=args.data_path,
        annot_path=args.annot_path,
        num_samples=args.num_samples,
    )

    def decode_fn(m, **kw):
        return decoder(m, **kw)

    results = evaluator.evaluate(
        model, processor, decode_fn,
        model_type=model_type,
        prompt=args.prompt,
    )
    dec_stats = decode_stats(decoder)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output or f"results/chair_{args.model}_{args.decoder}_{ts}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":      args.model,
        "decoder":    args.decoder,
        "seed":       args.seed,
        "num_samples": args.num_samples,
        "tps":        tps_info,
        "decode_stats": dec_stats,
        "results":    results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nCHAIR Results:")
    print(f"  CHAIR_s = {results['chair_s']:.4f}  "
          f"CHAIR_i = {results['chair_i']:.4f}  "
          f"(n={results.get('n_evaluated', '?')})")
    if dec_stats:
        print(
            f"  decode_stats  contrastive={dec_stats.get('contrastive', 0)}  "
            f"fallback={dec_stats.get('fallback', 0)}  "
            f"fallback_ratio={dec_stats.get('fallback_ratio', 0.0):.4f}"
        )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
