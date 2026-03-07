"""
DoLa: Decoding by Contrasting Layers.

Uses contrastive decoding between late and early layers of the LLM.
Reference: Chuang et al., 2023 (arXiv:2309.03883)
"""

from typing import Optional
import torch

try:
    from ..utils.itav import contrastive_decode
    from .cesd import _get_transformer_layers, _eos_reached
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.itav import contrastive_decode
    from src.decoding.cesd import _get_transformer_layers, _eos_reached


class DoLaDecoder:
    """
    DoLa: contrast mature (late) layer logits vs premature (early) layer logits.

    Args:
        alpha:           Contrast strength.
        mature_layer:    Late layer index (-1 = last layer).
        premature_layer: Early layer index (-5 = 5th from last).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        mature_layer: int = -1,
        premature_layer: int = -5,
        **kwargs,
    ):
        self.alpha = alpha
        self.mature_layer = mature_layer
        self.premature_layer = premature_layer

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
        layers = _get_transformer_layers(model)
        lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
        eos_token_id = getattr(model.config, "eos_token_id", None)

        # Fallback to greedy if we cannot access layers
        if layers is None or lm_head is None:
            gen_kwargs: dict = {
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

        n = len(layers)
        mature_idx = n + self.mature_layer if self.mature_layer < 0 else self.mature_layer
        premature_idx = n + self.premature_layer if self.premature_layer < 0 else self.premature_layer
        mature_idx = max(0, min(mature_idx, n - 1))
        premature_idx = max(0, min(premature_idx, n - 1))

        gen_kwargs = {}
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw

        batch_size = input_ids.shape[0]
        device = next(model.parameters()).device
        generated = input_ids.clone()
        cur_mask = attention_mask.clone() if attention_mask is not None else None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = model(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    output_hidden_states=True,
                    **gen_kwargs,
                )

            hidden = out.hidden_states
            if hidden is None or len(hidden) <= max(mature_idx, premature_idx) + 1:
                next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                # hidden[k+1] = output of layer k
                logits_mature = lm_head(hidden[mature_idx + 1][:, -1, :])
                logits_premature = lm_head(hidden[premature_idx + 1][:, -1, :])
                logits = contrastive_decode(logits_mature, logits_premature, self.alpha)
                next_tok = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tok], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat([
                    cur_mask,
                    torch.ones((batch_size, 1), device=device, dtype=cur_mask.dtype),
                ], dim=1)
            if _eos_reached(next_tok, eos_token_id):
                break

        return generated
