"""Model utilities for VLM inference and decoding."""

from typing import Optional, Tuple, List, Dict, Any, Sequence, Union
import torch
from PIL import Image


def get_image_token_indices(
    input_ids: torch.Tensor,
    image_token_id: Union[int, Sequence[int]],
    device: torch.device,
) -> Tuple[int, int]:
    """
    Get the start and end indices of image tokens in the input sequence.
    Returns (start_idx, end_idx) inclusive.
    """
    if isinstance(image_token_id, (list, tuple, set)):
        ids = [int(x) for x in image_token_id]
    else:
        ids = [int(image_token_id)]
    if not ids:
        return -1, -1

    if len(ids) == 1:
        token_mask = input_ids == ids[0]
    else:
        id_tensor = torch.tensor(ids, dtype=input_ids.dtype, device=device)
        token_mask = torch.isin(input_ids, id_tensor)
    indices = token_mask.nonzero(as_tuple=True)[0]
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


def resolve_image_token_id(
    model: Optional[Any] = None,
    model_type: str = "llava",
    fallback: Optional[int] = None,
) -> int:
    """
    Resolve image token id from model config at runtime.

    Priority:
      1. model.config known image-token fields
      2. model.generation_config known fields
      3. fallback arg
      4. get_model_info(model_type)["image_token_id"]
    """
    default_id = int(fallback if fallback is not None else get_model_info(model_type).get("image_token_id", 32000))
    if model is None:
        return default_id

    def _pick(obj: Any) -> Optional[int]:
        if obj is None:
            return None
        for key in (
            "image_token_id",
            "vision_token_id",
            "image_pad_token_id",
            "img_token_id",
        ):
            val = getattr(obj, key, None)
            if isinstance(val, int):
                return int(val)
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], int):
                return int(val[0])
        return None

    token_id = _pick(getattr(model, "config", None))
    if token_id is not None:
        return token_id
    token_id = _pick(getattr(model, "generation_config", None))
    if token_id is not None:
        return token_id
    return default_id
