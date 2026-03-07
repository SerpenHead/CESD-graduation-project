"""
Token sparsification for CESD: Vanilla Top-K and vision-agnostic masking.

Applied to hidden states at the selected amateur layer input to simulate
information loss and amplify contrast with expert layer.
"""

from typing import Optional, Tuple
import torch


def top_k_sparsify(
    hidden_states: torch.Tensor,
    attention_weights: torch.Tensor,
    k: int,
    keep_image_tokens: bool = False,
    image_token_start: int = -1,
    image_token_end: int = -1,
) -> torch.Tensor:
    """
    Vanilla Top-K sparsification: keep tokens with highest attention scores,
    mask the rest to zero.

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        attention_weights: (batch, num_heads, seq_len, seq_len) - use last query's attention
                           or (batch, seq_len) - aggregated attention scores
        k: Number of tokens to keep (Top-K)
        keep_image_tokens: If True, always keep image tokens and only sparsify others
        image_token_start, image_token_end: Image token range when keep_image_tokens=True

    Returns:
        Sparsified hidden_states: (batch, seq_len, hidden_dim)
    """
    batch, seq_len, hidden_dim = hidden_states.shape

    if attention_weights.dim() == 4:
        # (B, H, T, T) -> use last query position's attention to all keys
        attn_scores = attention_weights[:, :, -1, :].max(dim=1)[0]  # (B, T)
    else:
        attn_scores = attention_weights  # (B, T)

    mask = torch.ones(batch, seq_len, device=hidden_states.device, dtype=torch.bool)

    if keep_image_tokens and image_token_start >= 0 and image_token_end >= 0:
        # Always keep image tokens
        mask[:, image_token_start : image_token_end + 1] = True
        # Sparsify only non-image tokens
        non_img_indices = list(range(image_token_start)) + list(range(image_token_end + 1, seq_len))
        if not non_img_indices:
            return hidden_states
        non_img_scores = attn_scores[:, non_img_indices]
        _, top_idx = torch.topk(non_img_scores, min(k, len(non_img_indices)), dim=-1)
        for b in range(batch):
            keep_idx = [non_img_indices[i] for i in top_idx[b].tolist()]
            mask[b, :] = False
            mask[b, keep_idx] = True
            mask[b, image_token_start : image_token_end + 1] = True
    else:
        # Vanilla Top-K over all tokens
        _, top_idx = torch.topk(attn_scores, min(k, seq_len), dim=-1)
        mask = torch.zeros(batch, seq_len, device=hidden_states.device, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

    mask = mask.unsqueeze(-1).expand_as(hidden_states)
    return hidden_states * mask.float()


def top_k_sparsify_by_ratio(
    hidden_states: torch.Tensor,
    attention_weights: torch.Tensor,
    keep_ratio: float,
    **kwargs,
) -> torch.Tensor:
    """
    Top-K sparsification where k = ceil(seq_len * keep_ratio).

    Args:
        keep_ratio: Fraction of tokens to keep (e.g., 0.2 = keep 20%)
    """
    seq_len = hidden_states.shape[1]
    k = max(1, int(seq_len * keep_ratio))
    return top_k_sparsify(hidden_states, attention_weights, k=k, **kwargs)
