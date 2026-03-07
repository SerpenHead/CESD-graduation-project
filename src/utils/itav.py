"""
Image Token Attention Vector (iTaV) computation and JSD-based layer selection.

iTaV: For each layer, aggregate attention from current query to image tokens
      using max-over-heads, then softmax over image tokens.
JSD:  Jensen-Shannon Divergence for comparing iTaV distributions across layers.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


def compute_itav(
    attentions: Tuple[torch.Tensor, ...],
    image_token_start: int,
    image_token_end: int,
    layer_indices: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """
    Compute Image Token Attention Vector (iTaV) for each layer.

    iTaV^n_t = softmax([max_h a^n_{t,j,h}  for j in [v_s, v_e]])

    Args:
        attentions:         Tuple of per-layer tensors (B, H, T, T)
        image_token_start:  Inclusive start of image token range
        image_token_end:    Inclusive end of image token range
        layer_indices:      Which layers to compute; default all

    Returns:
        List of (B, num_image_tokens) normalised iTaV tensors
    """
    if image_token_start < 0 or image_token_end < 0:
        return []
    if image_token_end < image_token_start:
        return []

    itavs = []
    layers = layer_indices if layer_indices is not None else range(len(attentions))

    for n in layers:
        attn = attentions[n]                             # (B, H, T, T)
        img_attn = attn[:, :, -1, image_token_start: image_token_end + 1]  # (B, H, V)
        max_over_heads = img_attn.max(dim=1)[0]          # (B, V)
        itav = F.softmax(max_over_heads.float(), dim=-1) # (B, V), float32 for stability
        itavs.append(itav)

    return itavs


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Jensen-Shannon Divergence (batch).
    JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M),  M = 0.5*(P+Q)

    Args:
        p, q: (B, D) probability distributions

    Returns:
        (B,) scalar JSD per batch element
    """
    p = p.float() + eps
    q = q.float() + eps
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def select_contrastive_layer(
    itavs: List[torch.Tensor],
    final_layer_idx: int,
    candidate_layers: Optional[List[int]] = None,
) -> int:
    """
    Select the intermediate layer M* with maximum JSD distance from final-layer iTaV.

    Args:
        itavs:            List of iTaV tensors indexed by layer
        final_layer_idx:  Index of the expert (final) layer
        candidate_layers: Candidate amateur layer indices; default all except final

    Returns:
        Index of selected amateur layer
    """
    if len(itavs) <= 1:
        return 0

    final_itav = itavs[final_layer_idx]
    candidates = candidate_layers or [i for i in range(len(itavs)) if i != final_layer_idx]
    if not candidates:
        return max(0, final_layer_idx - 1)

    best_layer = candidates[0]
    best_jsd = -1.0
    for m in candidates:
        if m == final_layer_idx:
            continue
        d = jsd(final_itav, itavs[m]).mean().item()
        if d > best_jsd:
            best_jsd = d
            best_layer = m

    return best_layer


def contrastive_decode(
    expert_logits: torch.Tensor,
    amateur_logits: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Expert-amateur contrastive decoding.
    logits_final = logits_expert + alpha * (logits_expert - logits_amateur)

    Args:
        expert_logits:  (B, V) full-information logits
        amateur_logits: (B, V) degraded logits from sparsified amateur layer
        alpha:          Contrast strength

    Returns:
        (B, V) adjusted logits
    """
    return expert_logits + alpha * (expert_logits - amateur_logits)
