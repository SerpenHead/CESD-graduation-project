#!/usr/bin/env python3
"""
Run MME evaluation.

Example:
    python scripts/run_eval_mme.py --model llava --decoder cesd --seed 42
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
from src.evaluation.mme import MMEEvaluator
from src.decoding import GreedyDecoder, CESDDecoder


def main():
    parser = argparse.ArgumentParser(description="MME evaluation")
    parser.add_argument("--model",   default="llava", choices=["llava", "qwen2_vl"])
    parser.add_argument("--decoder", default="cesd",  choices=["greedy", "cesd"])
    parser.add_argument("--data_path", default="data/mme")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    model, processor = load_model(args.model)
    config     = get_model_config(args.model)
    model_type = config.get("model_type", args.model)

    decoder = (
        CESDDecoder(alpha=0.5, sparsify_ratio=0.2, model_type=model_type)
        if args.decoder == "cesd"
        else GreedyDecoder()
    )

    evaluator = MMEEvaluator(data_path=args.data_path)
    results = evaluator.evaluate(model, processor, decoder, model_type=model_type)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output or f"results/mme_{args.model}_{args.decoder}_{ts}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {"model": args.model, "decoder": args.decoder,
               "seed": args.seed, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"MME: Perception={results.get('perception', 0):.4f}  "
          f"Cognition={results.get('cognition', 0):.4f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
