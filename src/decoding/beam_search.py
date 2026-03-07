"""Beam search decoding for VLM baseline."""

from typing import Optional
import torch


class BeamSearchDecoder:
    """Beam search decoding."""

    def __init__(self, beam_size: int = 5, **kwargs):
        self.beam_size = beam_size

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
            "num_beams": self.beam_size,
            "pad_token_id": getattr(model.config, "pad_token_id", 0) or 0,
        }
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        return outputs[:, : input_ids.shape[1] + max_new_tokens]
