"""
Ablation study and parameter sensitivity analysis for CESD.

- Ablation: remove dynamic layer selection, remove sparsification
- Param sensitivity: alpha (contrast strength), sparsify_ratio (k)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import itertools


def get_ablation_configs() -> List[Dict[str, Any]]:
    """CESD ablation configurations."""
    return [
        {"name": "CESD-full", "use_dynamic_layer": True, "use_sparsification": True, "alpha": 0.5, "sparsify_ratio": 0.2},
        {"name": "CESD-no-dynamic", "use_dynamic_layer": False, "use_sparsification": True, "alpha": 0.5, "sparsify_ratio": 0.2},
        {"name": "CESD-no-sparse", "use_dynamic_layer": True, "use_sparsification": False, "alpha": 0.5, "sparsify_ratio": 0.2},
        {"name": "CESD-no-both", "use_dynamic_layer": False, "use_sparsification": False, "alpha": 0.5, "sparsify_ratio": 0.2},
    ]


def get_alpha_sweep_configs(alphas: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """Alpha (contrast strength) sensitivity."""
    alphas = alphas or [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    return [
        {"name": f"CESD-alpha{a}", "alpha": a, "sparsify_ratio": 0.2}
        for a in alphas
    ]


def get_sparsify_sweep_configs(ratios: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """Sparsify ratio sensitivity."""
    ratios = ratios or [0.1, 0.2, 0.3, 0.5]
    return [
        {"name": f"CESD-sparse{r}", "alpha": 0.5, "sparsify_ratio": r}
        for r in ratios
    ]


def aggregate_ablation_results(
    result_files: List[Path],
    metric_key: str = "f1",
) -> Dict[str, float]:
    """Aggregate ablation results from JSON files."""
    agg = {}
    for p in result_files:
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        name = d.get("config_name", p.stem)
        if "results" in d:
            r = d["results"]
            if isinstance(r, dict) and "random" in r:
                agg[name] = r["random"].get(metric_key, 0)
            elif isinstance(r, dict):
                agg[name] = r.get(metric_key, 0)
            else:
                agg[name] = r
        else:
            agg[name] = d.get(metric_key, 0)
    return agg
