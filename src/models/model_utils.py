"""Model utilities for VLM inference and decoding."""

from typing import Optional, Tuple, List, Dict, Any
import torch
from PIL import Image


def get_image_token_indices(
    input_ids: torch.Tensor,
    image_token_id: int,
    device: torch.device,
) -> Tuple[int, int]:
    """
    Get the start and end indices of image tokens in the input sequence.
    Returns (start_idx, end_idx) inclusive.
    """
    indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        return -1, -1
    return int(indices[0].item()), int(indices[-1].item())


def prepare_prompt_for_caption(prompt_template: str = None) -> str:
    """Default prompt for image captioning (CHAIR evaluation)."""
    if prompt_template is None:
        return "Describe this image in detail."
    return prompt_template


def prepare_prompt_for_vqa(question: str, prompt_template: str = None) -> str:
    """Prepare prompt for VQA (POPE yes/no questions)."""
    if prompt_template is None:
        return question
    return prompt_template.format(question=question)


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get model-specific info (image token id, layer structure, etc.)."""
    info = {
        "llava": {
            "image_token_id": 32000,
            "image_placeholder": "<image>",
            "num_layers": 32,
        },
        "qwen2_vl": {
            "image_token_id": 151643,  # <|vision_start|> or similar
            "image_placeholder": "<|vision_start|><|image_pad|><|vision_end|>",
            "num_layers": 28,
        },
    }
    return info.get(model_type, info["llava"])
