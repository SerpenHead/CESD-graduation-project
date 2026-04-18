#!/usr/bin/env python3
"""Generate paper figures and tables from results."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.plotting import (
    plot_param_sensitivity,
    plot_ablation_bars,
    plot_main_results_table,
)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_file(directory: Path, pattern: str):
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _extract_results(payload):
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        return payload["results"]
    if isinstance(payload, dict):
        return payload
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--ablation_file", default="results/ablation/ablation_summary.json")
    args = parser.parse_args()

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    ablation_dir = results_dir / "ablation"

    # Ablation bar chart
    ablation_path = Path(args.ablation_file)
    if not ablation_path.exists():
        latest_ablation = _latest_file(ablation_dir, "ablation_summary_*.json")
        if latest_ablation is not None:
            ablation_path = latest_ablation
    if ablation_path.exists():
        ablation_payload = _load_json(ablation_path)
        ablation = _extract_results(ablation_payload)
        f1_results = {k: v.get("f1", 0) for k, v in ablation.items() if isinstance(v, dict)}
        if f1_results:
            plot_ablation_bars(f1_results, str(fig_dir / "ablation_bars.png"), "CESD Ablation Study")

    # Param sensitivity (if we have sweep results)
    for sweep_name in ("alpha", "sparsify"):
        summary_path = _latest_file(ablation_dir, f"{sweep_name}_summary_*.json")
        if summary_path is None:
            continue
        sweep_payload = _load_json(summary_path)
        sweep_results = _extract_results(sweep_payload)
        results = {
            name: metrics.get("f1", 0)
            for name, metrics in sweep_results.items()
            if isinstance(metrics, dict)
        }
        if results:
            plot_param_sensitivity(
                results, sweep_name, str(fig_dir / f"{sweep_name}_sensitivity.png")
            )

    # Main results table
    pope_path = results_dir / "pope_main.json"
    if pope_path.exists():
        with open(pope_path) as f:
            data = json.load(f)
        if "decoders" in data:
            plot_main_results_table(
                data["decoders"],
                str(fig_dir / "table_pope_results.md"),
            )

    # Placeholder CESD framework diagram (mermaid-style description)
    framework_md = """# Figure 1: CESD Algorithm Framework

```mermaid
flowchart TD
    A[Input: image + text tokens] --> B[Full forward pass]
    B --> C[Expert Logits]
    B --> D[Compute iTaV per layer]
    D --> E[JSD: select contrastive layer M*]
    E --> F[Sparsify layer M* input]
    F --> G[Amateur Logits]
    C --> H[Contrastive Decode]
    G --> H
    H --> I[Output token]
```
"""
    with open(fig_dir / "figure1_cesd_framework.md", "w") as f:
        f.write(framework_md)

    print(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
