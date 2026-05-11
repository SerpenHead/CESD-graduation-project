"""
CESD: Contrast-Enhanced Sparsified Decoding.

Core idea:
  1. Expert pass  – full model forward  → expert logits + all hidden_states + attentions
  2. Select M*    – layer with max JSD(iTaV_N, iTaV_m)
  3. Sparsify     – Top-K on hidden_states[M*] (input to layer M*)
  4. Amateur pass – replay only layer M* on sparsified input to obtain amateur logits
  5. Contrast     – logits_final = expert + alpha * (expert - amateur)
"""

from typing import Optional, List
import torch

try:
    from ..utils.itav import compute_itav, select_contrastive_layer, contrastive_decode
    from ..utils.layer_ops import get_lm_head, replay_single_layer, project_hidden_to_logits
    from ..utils.sparsification import top_k_sparsify
    from ..models.model_utils import get_image_token_indices, get_model_info, resolve_image_token_id
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.itav import compute_itav, select_contrastive_layer, contrastive_decode
    from src.utils.layer_ops import get_lm_head, replay_single_layer, project_hidden_to_logits
    from src.utils.sparsification import top_k_sparsify
    from src.models.model_utils import get_image_token_indices, get_model_info, resolve_image_token_id


def _get_transformer_layers(model) -> Optional[torch.nn.ModuleList]:
    """Extract the list of transformer decoder layers from a VLM."""
    # LLaVA-Next: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Some seq2seq models
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    # GPT-style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return None


def _eos_reached(next_token: torch.Tensor, eos_token_id) -> bool:
    """Check EOS robustly, handling None, int, and list eos_token_id."""
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (list, tuple)):
        return any((next_token == e).all().item() for e in eos_token_id)
    return (next_token == eos_token_id).all().item()


class CESDDecoder:
    """
    CESD: Contrast-Enhanced Sparsified Decoding.

    Args:
        alpha:             Contrast strength (default 0.5)
        sparsify_ratio:    Fraction of tokens to keep in Top-K (default 0.2 = 20%)
        model_type:        "llava" | "qwen2_vl" (controls image token id lookup)
        use_dynamic_layer: If False, fix amateur layer at N//2 (ablation)
        use_sparsification:If False, replay the selected layer without Top-K masking
    """

    def __init__(
        self,
        alpha: float = 0.5,
        sparsify_ratio: float = 0.2,
        model_type: str = "llava",
        use_dynamic_layer: bool = True,
        use_sparsification: bool = True,
        **kwargs,
    ):
        self.alpha = alpha
        self.sparsify_ratio = sparsify_ratio
        self.model_type = model_type
        self.use_dynamic_layer = use_dynamic_layer
        self.use_sparsification = use_sparsification
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
        image_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        image_token_id = image_token_id or resolve_image_token_id(
            model=model,
            model_type=self.model_type,
            fallback=self.model_info.get("image_token_id", 32000),
        )
        eos_token_id = getattr(model.config, "eos_token_id", None)
        layers = _get_transformer_layers(model)
        lm_head = get_lm_head(model)

        # Build static kwargs for model.forward() (vision inputs)
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
            # ── Expert pass ─────────────────────────────────────────────────────
            with torch.no_grad():
                expert_out = model(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    **gen_kwargs,
                )

            expert_logits = expert_out.logits[:, -1, :]  # (B, V)
            hidden_states = expert_out.hidden_states      # tuple: len = N_layers + 1
            attentions = expert_out.attentions            # tuple: len = N_layers

            # Fallback to greedy if internals not available (SDPA 可能返回 attentions=() 空元组)
            if (hidden_states is None or attentions is None or layers is None or lm_head is None or
                    (attentions is not None and len(attentions) == 0) or
                    (hidden_states is not None and len(hidden_states) <= 1)):
                self._stats["fallback"] += 1
                next_token = expert_logits.argmax(dim=-1, keepdim=True)
            else:
                # Locate image tokens in the ORIGINAL prompt (unchanged positions)
                v_s, v_e = get_image_token_indices(input_ids[0], image_token_id, device)
                if v_s < 0:
                    self._stats["fallback"] += 1
                    next_token = expert_logits.argmax(dim=-1, keepdim=True)
                else:
                    # ── Compute iTaV, select amateur layer ────────────────────
                    itavs = compute_itav(attentions, v_s, v_e)
                    n_layers = len(attentions)
                    final_idx = n_layers - 1

                    if self.use_dynamic_layer and itavs:
                        m_star = select_contrastive_layer(
                            itavs, final_idx,
                            candidate_layers=list(range(final_idx)),
                        )
                    else:
                        m_star = max(0, final_idx // 2)

                    # hidden_states[m_star] = output of layer (m_star-1) = input to layer m_star
                    h_input = hidden_states[m_star]      # (B, T, D)
                    attn_m = attentions[m_star]          # (B, H, T, T)

                    # ── Sparsify ──────────────────────────────────────────────
                    if self.use_sparsification:
                        seq_len = h_input.shape[1]
                        k = max(1, int(seq_len * self.sparsify_ratio))
                        h_sparse = top_k_sparsify(h_input, attn_m, k=k)
                    else:
                        h_sparse = h_input

                    # ── Amateur pass (single-layer replay) ────────────────────
                    amateur_hidden = replay_single_layer(
                        model=model,
                        layers=layers,
                        layer_idx=m_star,
                        hidden_in=h_sparse,
                        attention_mask=cur_mask,
                        model_type=self.model_type,
                    )
                    amateur_logits = None
                    if amateur_hidden is not None:
                        amateur_logits = project_hidden_to_logits(model, amateur_hidden[:, -1, :])

                    # ── Contrastive decode ────────────────────────────────────
                    if amateur_logits is not None and amateur_logits.shape == expert_logits.shape:
                        self._stats["contrastive"] += 1
                        logits = contrastive_decode(expert_logits, amateur_logits, self.alpha)
                        next_token = logits.argmax(dim=-1, keepdim=True)
                    else:
                        self._stats["fallback"] += 1
                        next_token = expert_logits.argmax(dim=-1, keepdim=True)

            # ── Append token, extend mask ────────────────────────────────────
            generated = torch.cat([generated, next_token], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat([
                    cur_mask,
                    torch.ones((batch_size, 1), device=device, dtype=cur_mask.dtype),
                ], dim=1)

            if _eos_reached(next_token, eos_token_id):
                break

        return generated

    def get_and_reset_stats(self) -> dict:
        """返回本轮生成中 contrastive/fallback 步数并清零，用于验证是否真正跑了对比解码。"""
        s = dict(self._stats)
        self._stats["contrastive"] = 0
        self._stats["fallback"] = 0
        return s
