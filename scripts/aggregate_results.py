#!/usr/bin/env python3
"""Aggregate matrix JSON outputs into a single summary JSON."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_payload(benchmark: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    results = payload.get("results", {})
    if benchmark == "pope" and isinstance(results, dict):
        f1_vals = []
        for split in ("random", "popular", "adversarial"):
            if split in results and isinstance(results[split], dict):
                out[f"{split}_f1"] = float(results[split].get("f1", 0.0))
                out[f"{split}_acc"] = float(results[split].get("accuracy", 0.0))
                f1_vals.append(float(results[split].get("f1", 0.0)))
        out["avg_f1"] = float(sum(f1_vals) / len(f1_vals)) if f1_vals else 0.0
    elif benchmark == "chair" and isinstance(results, dict):
        out["chair_s"] = float(results.get("chair_s", 0.0))
        out["chair_i"] = float(results.get("chair_i", 0.0))
        out["n_evaluated"] = int(results.get("n_evaluated", 0))
    elif benchmark == "mme" and isinstance(results, dict):
        out["perception"] = float(results.get("perception", 0.0))
        out["cognition"] = float(results.get("cognition", 0.0))
    out["tps_mean"] = float(payload.get("tps", {}).get("tps_mean", 0.0))
    ds = payload.get("decode_stats", {})
    out["fallback_ratio"] = float(ds.get("fallback_ratio", 0.0))
    out["contrastive_steps"] = int(ds.get("contrastive", 0))
    out["fallback_steps"] = int(ds.get("fallback", 0))
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate run_matrix outputs")
    parser.add_argument("--results_root", default="results/matrix")
    parser.add_argument("--output", default="results/matrix_summary.json")
    args = parser.parse_args()

    root = Path(args.results_root)
    summary: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    if not root.exists():
        raise FileNotFoundError(f"results_root not found: {root}")

    for bench_dir in root.iterdir():
        if not bench_dir.is_dir():
            continue
        benchmark = bench_dir.name
        summary.setdefault(benchmark, {})
        for model_dir in bench_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            summary[benchmark].setdefault(model, {})
            for decoder_dir in model_dir.iterdir():
                if not decoder_dir.is_dir():
                    continue
                decoder = decoder_dir.name
                files = sorted(decoder_dir.glob("seed_*.json"))
                if not files:
                    continue
                payload = load_json(files[-1])
                summary[benchmark][model][decoder] = summarize_payload(benchmark, payload)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary -> {out_path}")


if __name__ == "__main__":
    main()
