"""
OPERA: Over-Trust Penalty and Retrospection-Allocation decoding.

Inference-time implementation adapted for VLM generation:
  1) Run expert forward and compute image-attention confidence.
  2) If confidence is below threshold, apply over-trust penalty on top candidates.
  3) Retrospectively select token by one-step lookahead with attention-aware score.

Reference: Huang et al., CVPR 2024
"""

from typing import Optional, Dict, List
import torch

try:
    from ..decoding.cesd import _get_transformer_layers, _eos_reached
    from ..models.model_utils import get_image_token_indices, resolve_image_token_id
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.decoding.cesd import _get_transformer_layers, _eos_reached
    from src.models.model_utils import get_image_token_indices, resolve_image_token_id


def _vision_confidence(
    attentions: Optional[List[torch.Tensor]],
    image_start: int,
    image_end: int,
) -> Optional[torch.Tensor]:
    """
    Estimate current step confidence on visual grounding from last-layer attentions.

    Returns:
        (B,) tensor in [0, 1], or None if unavailable.
    """
    if attentions is None or len(attentions) == 0:
        return None
    if image_start < 0 or image_end < image_start:
        return None

    attn = attentions[-1]  # (B, H, T, T)
    if attn.dim() != 4:
        return None
    img_attn = attn[:, :, -1, image_start: image_end + 1]  # (B, H, V)
    if img_attn.numel() == 0:
        return None
    # Conservative confidence: head-wise max then mean over image tokens.
    return img_attn.max(dim=1)[0].mean(dim=-1).float()


class OPERADecoder:
    """OPERA decoding with confidence penalty + retrospective token allocation."""

    def __init__(
        self,
        threshold: float = 0.10,
        num_attn_candidates: int = 5,
        penalty_weights: float = 1.0,
        scale_factor: float = 50.0,
        lookahead_weight: float = 2.0,
        model_type: str = "llava",
        **kwargs,
    ):
        self.threshold = threshold
        self.num_attn_candidates = max(1, int(num_attn_candidates))
        self.penalty_weights = penalty_weights
        self.scale_factor = scale_factor
        self.lookahead_weight = lookahead_weight
        self.model_type = model_type
        self._stats = {"contrastive": 0, "fallback": 0, "penalized": 0}

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
        layers = _get_transformer_layers(model)
        eos_token_id = getattr(model.config, "eos_token_id", None)
        image_token_id = resolve_image_token_id(model=model, model_type=self.model_type)

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
        image_start, image_end = get_image_token_indices(input_ids[0], image_token_id, device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = model(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    output_hidden_states=False,
                    output_attentions=True,
                    **gen_kwargs,
                )
            logits = out.logits[:, -1, :]  # (B, V)

            if layers is None or out.attentions is None or len(out.attentions) == 0 or image_start < 0:
                self._stats["fallback"] += 1
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                confidence = _vision_confidence(out.attentions, image_start, image_end)
                if confidence is None:
                    self._stats["fallback"] += 1
                    next_tok = logits.argmax(dim=-1, keepdim=True)
                else:
                    k = min(self.num_attn_candidates, logits.shape[-1])
                    top_vals, top_ids = torch.topk(logits, k=k, dim=-1)
                    adjusted = logits.clone()

                    # Over-trust penalty when visual confidence is below threshold.
                    penalty_applied = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
                    for b in range(batch_size):
                        if confidence[b].item() >= self.threshold:
                            continue
                        penalty_applied[b] = True
                        penalty = self.scale_factor * (self.threshold - confidence[b].item())
                        for rank in range(k):
                            token_id = int(top_ids[b, rank].item())
                            rank_w = self.penalty_weights / float(rank + 1)
                            adjusted[b, token_id] = adjusted[b, token_id] - penalty * rank_w

                    if penalty_applied.any():
                        self._stats["penalized"] += int(penalty_applied.sum().item())

                    # Retrospection-allocation via one-step lookahead on top candidates.
                    selected = adjusted.argmax(dim=-1, keepdim=True)
                    for b in range(batch_size):
                        if not penalty_applied[b]:
                            continue

                        best_token = int(top_ids[b, 0].item())
                        best_score = -1e30
                        for rank in range(k):
                            cand = int(top_ids[b, rank].item())
                            cand_score = float(adjusted[b, cand].item())

                            trial_input = torch.cat(
                                [generated[b: b + 1], torch.tensor([[cand]], device=device, dtype=generated.dtype)],
                                dim=1,
                            )
                            if cur_mask is not None:
                                trial_mask = torch.cat(
                                    [cur_mask[b: b + 1], torch.ones((1, 1), device=device, dtype=cur_mask.dtype)],
                                    dim=1,
                                )
                            else:
                                trial_mask = None

                            look_conf = 0.0
                            try:
                                with torch.no_grad():
                                    trial_out = model(
                                        input_ids=trial_input,
                                        attention_mask=trial_mask,
                                        output_hidden_states=False,
                                        output_attentions=True,
                                        **gen_kwargs,
                                    )
                                trial_vis = _vision_confidence(trial_out.attentions, image_start, image_end)
                                if trial_vis is not None:
                                    look_conf = float(trial_vis[0].item())
                            except Exception:
                                # Ignore lookahead failures and keep adjusted score only.
                                pass

                            final_score = cand_score + self.lookahead_weight * look_conf
                            if final_score > best_score:
                                best_score = final_score
                                best_token = cand
                        selected[b, 0] = best_token

                    next_tok = selected
                    self._stats["contrastive"] += 1

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
        self._stats["penalized"] = 0
        return s
