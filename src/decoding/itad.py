"""
iTaD: Image Token Attention-Guided Decoding (baseline).

Same dynamic layer selection as CESD but WITHOUT sparsification.
Amateur logits come from running the model forward with a hook that injects
the raw intermediate hidden state at layer M*.

Reference: Xu et al., NAACL 2025
"""

from typing import Optional
import torch

try:
    from ..utils.itav import compute_itav, select_contrastive_layer, contrastive_decode
    from ..models.model_utils import get_image_token_indices, get_model_info
    from .cesd import _get_transformer_layers, _eos_reached
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.itav import compute_itav, select_contrastive_layer, contrastive_decode
    from src.models.model_utils import get_image_token_indices, get_model_info
    from src.decoding.cesd import _get_transformer_layers, _eos_reached


def _run_layer_hook_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    gen_kwargs: dict,
    layers,
    m_star: int,
    h_inject: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Inject h_inject at layer m_star via pre-hook; run full model forward."""
    fired = [False]

    def _pre_hook(module, args):
        if not fired[0]:
            fired[0] = True
            if isinstance(args, tuple) and len(args) > 0:
                return (h_inject.to(args[0].dtype),) + args[1:]
        return args

    handle = layers[m_star].register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                **gen_kwargs,
            )
        return out.logits[:, -1, :]
    except Exception as e:
        print(f"[iTaD] Amateur forward failed: {e}")
        return None
    finally:
        handle.remove()


class ITaDDecoder:
    """iTaD baseline: dynamic layer selection + contrastive decode, no sparsification."""

    def __init__(self, alpha: float = 0.5, model_type: str = "llava", **kwargs):
        self.alpha = alpha
        self.model_info = get_model_info(model_type)
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
        image_token_id = self.model_info.get("image_token_id", 32000)
        eos_token_id = getattr(model.config, "eos_token_id", None)
        layers = _get_transformer_layers(model)

        gen_kwargs: dict = {}
        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            gen_kwargs["image_grid_thw"] = image_grid_thw
        if image_sizes is not None:
            gen_kwargs["image_sizes"] = image_sizes

        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        cur_mask = attention_mask.clone() if attention_mask is not None else None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                expert_out = model(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    **gen_kwargs,
                )

            expert_logits = expert_out.logits[:, -1, :]

            if (expert_out.hidden_states is None or expert_out.attentions is None or layers is None or
                    len(expert_out.attentions) == 0):
                self._stats["fallback"] += 1
                next_tok = expert_logits.argmax(dim=-1, keepdim=True)
            else:
                v_s, v_e = get_image_token_indices(input_ids[0], image_token_id, device)
                if v_s < 0:
                    self._stats["fallback"] += 1
                    next_tok = expert_logits.argmax(dim=-1, keepdim=True)
                else:
                    itavs = compute_itav(expert_out.attentions, v_s, v_e)
                    if not itavs:
                        self._stats["fallback"] += 1
                        next_tok = expert_logits.argmax(dim=-1, keepdim=True)
                    else:
                        final_idx = len(itavs) - 1
                        m_star = select_contrastive_layer(itavs, final_idx)
                        h_inject = expert_out.hidden_states[m_star]  # raw, no sparsification

                        amateur_logits = _run_layer_hook_forward(
                            model, generated, cur_mask, gen_kwargs,
                            layers, m_star, h_inject,
                        )
                        if amateur_logits is not None and amateur_logits.shape == expert_logits.shape:
                            self._stats["contrastive"] += 1
                            logits = contrastive_decode(expert_logits, amateur_logits, self.alpha)
                            next_tok = logits.argmax(dim=-1, keepdim=True)
                        else:
                            self._stats["fallback"] += 1
                            next_tok = expert_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tok], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat([
                    cur_mask,
                    torch.ones((batch_size, 1), device=device, dtype=cur_mask.dtype),
                ], dim=1)

            if _eos_reached(next_tok, eos_token_id):
                break

        return generated

    def get_and_reset_stats(self) -> dict:
        """返回本轮生成中 contrastive/fallback 步数并清零，用于验证是否真正跑了对比解码。"""
        s = dict(self._stats)
        self._stats["contrastive"] = 0
        self._stats["fallback"] = 0
        return s
