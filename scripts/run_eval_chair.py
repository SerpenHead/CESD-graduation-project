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
from src.models.model_loader import load_model, get_model_config
from src.evaluation.chair import CHAIREvaluator
from src.decoding import GreedyDecoder, CESDDecoder, ITaDDecoder, DoLaDecoder


def main():
    parser = argparse.ArgumentParser(description="CHAIR evaluation")
    parser.add_argument("--model",       default="llava",
                        choices=["llava", "qwen2_vl"])
    parser.add_argument("--decoder",     default="cesd",
                        choices=["greedy", "dola", "itad", "cesd"])
    parser.add_argument("--data_path",   default="data/mscoco")
    parser.add_argument("--annot_path",  default=None,
                        help="Path to instances_val2014.json (auto-detected if not set)")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--prompt",      default="Describe this image in detail.",
                        help="Caption instruction sent to the model")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    need_attn = args.decoder in ("cesd", "itad")
    print(f"[CHAIR] Loading model: {args.model}" + (" (attn_implementation=eager for CESD/iTaD)" if need_attn else ""))
    model, processor = load_model(args.model, attn_implementation="eager" if need_attn else None)
    config      = get_model_config(args.model)
    model_type  = config.get("model_type", args.model)

    decoders = {
        "greedy": GreedyDecoder(),
        "dola":   DoLaDecoder(alpha=0.1),
        "itad":   ITaDDecoder(alpha=0.5, model_type=model_type),
        "cesd":   CESDDecoder(alpha=0.5, sparsify_ratio=0.2, model_type=model_type),
    }
    decoder = decoders[args.decoder]

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

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output or f"results/chair_{args.model}_{args.decoder}_{ts}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":      args.model,
        "decoder":    args.decoder,
        "seed":       args.seed,
        "num_samples": args.num_samples,
        "results":    results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nCHAIR Results:")
    print(f"  CHAIR_s = {results['chair_s']:.4f}  "
          f"CHAIR_i = {results['chair_i']:.4f}  "
          f"(n={results.get('n_evaluated', '?')})")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
