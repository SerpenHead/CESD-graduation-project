"""
VASparse: Visual-Aware Token Sparsification decoding baseline.

Inference-time implementation:
  - select a decoder layer
  - sparsify hidden-state tokens using attention-based Top-K
  - inject sparsified state and decode with greedy token choice

Reference: Zhuang et al., CVPR 2025
"""

from typing import Optional, Dict
import torch

try:
    from ..utils.sparsification import top_k_sparsify
    from ..models.model_utils import get_image_token_indices, resolve_image_token_id
    from .cesd import _get_transformer_layers, _eos_reached, _run_amateur_forward
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.sparsification import top_k_sparsify
    from src.models.model_utils import get_image_token_indices, resolve_image_token_id
    from src.decoding.cesd import _get_transformer_layers, _eos_reached, _run_amateur_forward


class VASparseDecoder:
    """VASparse-style decode with attention-guided hidden-state sparsification."""

    def __init__(
        self,
        keep_ratio: float = 0.5,
        sparse_layer: int = -4,
        keep_image_tokens: bool = True,
        model_type: str = "llava",
        **kwargs,
    ):
        self.keep_ratio = keep_ratio
        self.sparse_layer = sparse_layer
        self.keep_image_tokens = keep_image_tokens
        self.model_type = model_type
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
        layers = _get_transformer_layers(model)
        image_token_id = resolve_image_token_id(model=model, model_type=self.model_type)
        v_s, v_e = get_image_token_indices(input_ids[0], image_token_id, device)

        gen_kwargs: Dict[str, torch.Tensor] = {}
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
                out = model(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    **gen_kwargs,
                )
            expert_logits = out.logits[:, -1, :]

            if (
                layers is None
                or out.hidden_states is None
                or out.attentions is None
                or len(out.hidden_states) <= 1
                or len(out.attentions) == 0
            ):
                self._stats["fallback"] += 1
                next_tok = expert_logits.argmax(dim=-1, keepdim=True)
            else:
                n_layers = len(out.attentions)
                idx = self.sparse_layer if self.sparse_layer >= 0 else n_layers + self.sparse_layer
                idx = max(0, min(idx, n_layers - 1))

                h_input = out.hidden_states[idx]  # input to layer idx
                attn_m = out.attentions[idx]
                seq_len = h_input.shape[1]
                k = max(1, int(seq_len * self.keep_ratio))

                h_sparse = top_k_sparsify(
                    h_input,
                    attn_m,
                    k=k,
                    keep_image_tokens=self.keep_image_tokens,
                    image_token_start=v_s,
                    image_token_end=v_e,
                )
                sparse_logits = _run_amateur_forward(
                    model=model,
                    input_ids=generated,
                    attention_mask=cur_mask,
                    gen_kwargs=gen_kwargs,
                    layers=layers,
                    m_star=idx,
                    h_sparse=h_sparse,
                )
                if sparse_logits is None or sparse_logits.shape != expert_logits.shape:
                    self._stats["fallback"] += 1
                    next_tok = expert_logits.argmax(dim=-1, keepdim=True)
                else:
                    self._stats["contrastive"] += 1
                    next_tok = sparse_logits.argmax(dim=-1, keepdim=True)

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
