"""Shared helpers for intermediate-layer projection and single-layer replay."""

from __future__ import annotations

import inspect
from typing import Optional, Sequence

import torch


def get_lm_head(model) -> Optional[torch.nn.Module]:
    """Return the model's output projection head when available."""
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head
    if hasattr(model, "get_output_embeddings"):
        return model.get_output_embeddings()
    return None


def get_final_norm(model) -> Optional[torch.nn.Module]:
    """Return the final normalization layer applied before lm_head when available."""
    candidates = [
        ("model", "norm"),
        ("model", "decoder", "norm"),
        ("transformer", "ln_f"),
        ("transformer", "norm"),
    ]
    for path in candidates:
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and cur is not None:
            return cur
    return None


def project_hidden_to_logits(
    model,
    hidden_states: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Project hidden states to logits using the model's final norm + lm_head path."""
    lm_head = get_lm_head(model)
    if lm_head is None or hidden_states is None:
        return None

    final_norm = get_final_norm(model)
    hidden = hidden_states
    if final_norm is not None:
        hidden = final_norm(hidden)
    return lm_head(hidden)


def get_layer_output_from_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    layer_idx: int,
) -> Optional[torch.Tensor]:
    """
    Map decoder-layer index to HuggingFace hidden_states index.

    hidden_states[0] is the embedding output, hidden_states[k+1] is the
    output of decoder layer k.
    """
    target_idx = layer_idx + 1
    if hidden_states is None or target_idx < 0 or target_idx >= len(hidden_states):
        return None
    return hidden_states[target_idx]


def project_intermediate_logits(
    model,
    hidden_states: Sequence[torch.Tensor],
    layer_idx: int,
) -> Optional[torch.Tensor]:
    """Project the selected intermediate layer output to vocabulary logits."""
    layer_out = get_layer_output_from_hidden_states(hidden_states, layer_idx)
    if layer_out is None:
        return None
    return project_hidden_to_logits(model, layer_out[:, -1, :])


def _make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build monotonic position ids from a 2D attention mask."""
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def _make_causal_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive 4D causal mask accepted by decoder layers."""
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    min_value = torch.finfo(dtype).min

    causal = torch.full((seq_len, seq_len), min_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()

    key_mask = attention_mask[:, None, None, :].to(torch.bool)
    causal = causal.masked_fill(~key_mask, min_value)
    return causal


def _compute_position_embeddings(
    model,
    hidden_in: torch.Tensor,
    position_ids: torch.Tensor,
):
    """Best-effort rotary position embedding computation for replay."""
    candidates = []
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        candidates.append(model.model.rotary_emb)
    if hasattr(model, "rotary_emb"):
        candidates.append(model.rotary_emb)

    for rotary in candidates:
        attempts = [
            (hidden_in, position_ids),
            (hidden_in, position_ids, hidden_in.shape[1]),
        ]
        for args in attempts:
            try:
                out = rotary(*args)
                if out is not None:
                    return out
            except Exception:
                continue
    return None


def _extract_hidden_from_layer_output(layer_out) -> Optional[torch.Tensor]:
    """Normalize decoder layer outputs to the hidden-state tensor."""
    if torch.is_tensor(layer_out):
        return layer_out
    if isinstance(layer_out, (tuple, list)) and layer_out:
        if torch.is_tensor(layer_out[0]):
            return layer_out[0]
    return None


def replay_single_layer(
    model,
    layers,
    layer_idx: int,
    hidden_in: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    model_type: str,
) -> Optional[torch.Tensor]:
    """
    Replay only the selected decoder layer on a full-sequence hidden-state input.

    This helper is intentionally conservative: it only targets currently supported
    `llava` and `qwen2_vl` model families and returns None if the layer interface
    cannot be satisfied safely.
    """
    if model_type not in {"llava", "qwen2_vl"}:
        return None
    if layers is None or layer_idx < 0 or layer_idx >= len(layers):
        return None
    if hidden_in is None or hidden_in.dim() != 3:
        return None

    layer = layers[layer_idx]
    try:
        signature = inspect.signature(layer.forward)
    except (TypeError, ValueError):
        return None

    batch_size, seq_len, _ = hidden_in.shape
    device = hidden_in.device
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
    elif attention_mask.dim() != 2:
        return None

    position_ids = _make_position_ids(attention_mask)
    cache_position = torch.arange(seq_len, device=device)

    kwargs = {}
    params = signature.parameters
    if "attention_mask" in params:
        kwargs["attention_mask"] = _make_causal_attention_mask(attention_mask, hidden_in.dtype)
    if "position_ids" in params:
        kwargs["position_ids"] = position_ids
    if "cache_position" in params:
        kwargs["cache_position"] = cache_position
    if "past_key_value" in params:
        kwargs["past_key_value"] = None
    if "output_attentions" in params:
        kwargs["output_attentions"] = False
    if "use_cache" in params:
        kwargs["use_cache"] = False
    if "position_embeddings" in params:
        position_embeddings = _compute_position_embeddings(model, hidden_in, position_ids)
        if position_embeddings is None:
            return None
        kwargs["position_embeddings"] = position_embeddings

    try:
        layer_out = layer(hidden_in, **kwargs)
    except Exception:
        return None

    hidden_out = _extract_hidden_from_layer_output(layer_out)
    if hidden_out is None or hidden_out.shape[:2] != hidden_in.shape[:2]:
        return None
    return hidden_out
