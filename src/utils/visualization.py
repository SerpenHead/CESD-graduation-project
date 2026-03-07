"""Visualization utilities for CESD analysis and paper figures."""

from typing import List, Optional
import numpy as np
import torch


def itav_heatmap(
    itavs: List[torch.Tensor],
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot iTaV evolution across layers as heatmap.

    Args:
        itavs: List of (num_image_tokens,) or (batch, num_image_tokens)
        layer_names: Optional layer labels
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    if isinstance(itavs[0], torch.Tensor):
        data = torch.stack([itavs[i].mean(0) if itavs[i].dim() > 1 else itavs[i] for i in range(len(itavs))])
        data = data.cpu().numpy()
    else:
        data = np.array(itavs)

    if data.ndim == 3:
        data = data.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, ax=ax, cmap="viridis", cbar=True)
    ax.set_xlabel("Image Token Index")
    ax.set_ylabel("Layer")
    if layer_names:
        ax.set_yticklabels(layer_names, rotation=0)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
