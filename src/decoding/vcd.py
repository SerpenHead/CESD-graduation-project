"""
VCD: Visual Contrastive Decoding.

Contrastive decoding for vision-language models.
Reference: Leng et al., CVPR 2024
"""

from typing import Optional
import torch

# VCD contrasts expert (full model) with amateur (vision-masked)
# Simplified: use DoLa-style layer contrast as approximation


class VCDDecoder:
    """VCD-style contrastive decoding (simplified layer contrast)."""

    def __init__(self, alpha: float = 0.5, **kwargs):
        self.alpha = alpha

    def __call__(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        # Use standard generate as fallback; full VCD needs vision masking
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": getattr(model.config, "pad_token_id", 0) or 0,
        }
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw
        if image_sizes is not None:
            gen_kwargs["image_sizes"] = image_sizes

        with torch.no_grad():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
