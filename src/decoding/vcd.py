"""
VCD: Visual Contrastive Decoding.

Inference-time implementation:
  - expert logits: original visual input
  - amateur logits: visually perturbed input
  - final logits: expert + alpha * (expert - amateur)

Reference: Leng et al., CVPR 2024
"""

from typing import Optional, Dict
import torch

try:
    from ..utils.itav import contrastive_decode
    from .cesd import _eos_reached
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.itav import contrastive_decode
    from src.decoding.cesd import _eos_reached


def _perturb_pixel_values(pixel_values: Optional[torch.Tensor], noise_std: float) -> Optional[torch.Tensor]:
    if pixel_values is None:
        return None
    if noise_std <= 0:
        return pixel_values
    noise = torch.randn_like(pixel_values) * noise_std
    return pixel_values + noise


class VCDDecoder:
    """VCD decoding via visual perturbation contrast."""

    def __init__(self, alpha: float = 0.5, noise_std: float = 0.05, **kwargs):
        self.alpha = alpha
        self.noise_std = noise_std
        self._stats = {"contrastive": 0, "fallback": 0}

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
        device = next(model.parameters()).device
        eos_token_id = getattr(model.config, "eos_token_id", None)
        batch_size = input_ids.shape[0]

        expert_kwargs: Dict[str, torch.Tensor] = {}
        if pixel_values is not None:
            expert_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            expert_kwargs["image_grid_thw"] = image_grid_thw
        if image_sizes is not None:
            expert_kwargs["image_sizes"] = image_sizes

        amateur_kwargs = dict(expert_kwargs)
        amateur_kwargs["pixel_values"] = _perturb_pixel_values(pixel_values, self.noise_std)

        generated = input_ids.clone()
        cur_mask = attention_mask.clone() if attention_mask is not None else None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                expert_out = model(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    **expert_kwargs,
                )
            expert_logits = expert_out.logits[:, -1, :]

            if pixel_values is None:
                self._stats["fallback"] += 1
                next_tok = expert_logits.argmax(dim=-1, keepdim=True)
            else:
                amateur_logits = None
                try:
                    with torch.no_grad():
                        amateur_out = model(
                            input_ids=generated,
                            attention_mask=cur_mask,
                            output_hidden_states=False,
                            output_attentions=False,
                            **amateur_kwargs,
                        )
                    amateur_logits = amateur_out.logits[:, -1, :]
                except Exception:
                    amateur_logits = None

                if amateur_logits is None or amateur_logits.shape != expert_logits.shape:
                    self._stats["fallback"] += 1
                    next_tok = expert_logits.argmax(dim=-1, keepdim=True)
                else:
                    self._stats["contrastive"] += 1
                    logits = contrastive_decode(expert_logits, amateur_logits, self.alpha)
                    next_tok = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tok], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((batch_size, 1), device=device, dtype=cur_mask.dtype)],
                    dim=1,
                )
            if _eos_reached(next_tok, eos_token_id):
                break

        return generated

    def get_and_reset_stats(self) -> dict:
        s = dict(self._stats)
        self._stats["contrastive"] = 0
        self._stats["fallback"] = 0
        return s
