"""
OPERA: Over-Trust Penalty and Retrospection-Allocation.

Training-time method; at inference we use greedy as placeholder.
Reference: Huang et al., CVPR 2024
"""

from typing import Optional
import torch


class OPERADecoder:
    """OPERA is a training-time method; at inference we use standard decode."""

    def __init__(self, **kwargs):
        pass

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
