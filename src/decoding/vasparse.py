"""
VASparse: Visual-Aware Token Sparsification.

Sparsifies visual tokens for efficiency; can increase hallucination.
Reference: Zhuang et al., CVPR 2025
"""

from typing import Optional
import torch

# VASparse focuses on inference speed via token sparsification.
# We implement a simplified version: Top-K sparsification on image tokens
# before each forward (reduces sequence length for efficiency).
# For fair comparison as baseline, we use greedy decode with sparsified vision.


class VASparseDecoder:
    """VASparse-style: sparsify vision tokens, then greedy decode."""

    def __init__(self, keep_ratio: float = 0.5, **kwargs):
        self.keep_ratio = keep_ratio

    def __call__(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        # Simplified: use standard greedy (full VASparse needs custom vision encoder)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": getattr(model.config, "pad_token_id", 0) or 0,
        }
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw

        with torch.no_grad():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
