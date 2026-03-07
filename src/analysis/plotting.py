"""Plotting utilities for CESD paper figures."""

from pathlib import Path
from typing import Dict, List, Optional
import json


def plot_param_sensitivity(
    results: Dict[str, float],
    param_name: str,
    save_path: str,
    title: Optional[str] = None,
):
    """Plot parameter sensitivity curve."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    names = list(results.keys())
    vals = []
    for n in names:
        if "alpha" in n:
            try:
                v = float(n.split("alpha")[-1].replace("-", "").split("_")[0])
            except ValueError:
                v = 0.5
        elif "sparse" in n:
            try:
                v = float(n.split("sparse")[-1].replace("-", "").split("_")[0])
            except ValueError:
                v = 0.2
        else:
            v = 0
        vals.append(v)
    scores = [results[n] for n in names]
    order = np.argsort(vals)
    x = [vals[i] for i in order]
    y = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, "o-", linewidth=2, markersize=8)
    ax.set_xlabel(param_name)
    ax.set_ylabel("F1 / CHAIR")
    ax.set_title(title or f"{param_name} Sensitivity")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_bars(
    results: Dict[str, float],
    save_path: str,
    title: str = "Ablation Study",
):
    """Bar chart for ablation results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = list(results.keys())
    scores = list(results.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), scores, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_main_results_table(
    results: Dict[str, Dict[str, float]],
    save_path: str,
):
    """Save main results as formatted table (for LaTeX or markdown)."""
    lines = ["| Method | " + " | ".join(list(list(results.values())[0].keys())) + " |"]
    lines.append("|" + "---|" * (len(list(results.values())[0]) + 1) + "|")
    for method, metrics in results.items():
        row = "| " + method + " | " + " | ".join(f"{v:.4f}" for v in metrics.values()) + " |"
        lines.append(row)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
