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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--ablation_file", default="results/ablation/ablation_summary.json")
    args = parser.parse_args()

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Ablation bar chart
    if Path(args.ablation_file).exists():
        with open(args.ablation_file) as f:
            ablation = json.load(f)
        f1_results = {k: v.get("f1", 0) for k, v in ablation.items()}
        plot_ablation_bars(f1_results, str(fig_dir / "ablation_bars.png"), "CESD Ablation Study")

    # Param sensitivity (if we have sweep results)
    for sweep_name, pattern in [
        ("alpha", "alpha_sweep*.json"),
        ("sparsify", "sparsify_sweep*.json"),
    ]:
        sweep_files = list(Path(args.results_dir).glob(pattern))
        if sweep_files:
            results = {}
            for p in sweep_files:
                with open(p) as f:
                    d = json.load(f)
                results[d.get("name", p.stem)] = d.get("f1", 0)
            plot_param_sensitivity(
                results, sweep_name, str(fig_dir / f"{sweep_name}_sensitivity.png")
            )

    # Main results table
    pope_path = Path(args.results_dir) / "pope_main.json"
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
