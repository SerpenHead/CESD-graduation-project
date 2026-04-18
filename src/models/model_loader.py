"""Unified model loading for LLaVA-1.6 and Qwen2-VL-7B."""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import torch
import yaml
from PIL import Image


def get_model_config(model_name: str = "llava") -> Dict[str, Any]:
    """Load model config from YAML."""
    config_dir = Path(__file__).parent.parent.parent / "configs" / "models"
    config_path = config_dir / f"{model_name}.yaml"
    if not config_path.exists():
        config_path = config_dir / "llava.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(
    model_name: str = "llava",
    model_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
    **kwargs,
) -> Tuple[Any, Any]:
    """
    Load VLM model and processor.

    attn_implementation: "eager" 时强制使用 eager attention，保证 output_attentions=True
        能返回注意力权重（CESD/iTaD 需要）。默认 None 使用 SDPA，可能不返回 attentions。

    Returns:
        (model, processor) tuple with model.eval() already called.
    """
    from transformers import AutoProcessor, AutoModelForCausalLM

    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    config = get_model_config(model_name)
    configured_local_path = config.get("model_path")
    configured_hub_name = config.get("model_name")

    if model_path:
        hf_path = model_path
    elif configured_local_path and Path(configured_local_path).exists():
        hf_path = configured_local_path
    else:
        hf_path = configured_hub_name
        if configured_local_path and not Path(configured_local_path).exists():
            print(
                f"[model_loader] Local model path not found: {configured_local_path}. "
                "Falling back to model_name from config."
            )
    if hf_path is None:
        hf_path = {
            "llava": "llava-hf/llava-v1.6-vicuna-7b-hf",
            "qwen2_vl": "Qwen/Qwen2-VL-7B-Instruct",
        }.get(model_name, "llava-hf/llava-v1.6-vicuna-7b-hf")

    model_type = config.get("model_type", model_name)
    dtype = dtype or (torch.float16 if config.get("dtype") == "float16" else torch.bfloat16)

    try:
        if model_type == "llava":
            from transformers import LlavaNextForConditionalGeneration
            model = LlavaNextForConditionalGeneration.from_pretrained(
                hf_path, torch_dtype=dtype, device_map=device_map,
                trust_remote_code=trust_remote_code, **kwargs,
            )
        elif model_type == "qwen2_vl":
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                hf_path, torch_dtype=dtype, device_map=device_map,
                trust_remote_code=trust_remote_code, **kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_path, torch_dtype=dtype, device_map=device_map,
                trust_remote_code=trust_remote_code, **kwargs,
            )
    except Exception as e:
        print(f"[model_loader] Specific class failed ({e}), trying AutoModelForCausalLM.")
        model = AutoModelForCausalLM.from_pretrained(
            hf_path, torch_dtype=dtype, device_map=device_map,
            trust_remote_code=trust_remote_code, **kwargs,
        )

    processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=trust_remote_code)
    model.eval()
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Chat-template helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_llava_template(processor: Any, text: str) -> str:
    """
    Apply the vicuna/llava-1.6 chat template to a user question.

    Priority:
      1. processor.apply_chat_template (HF >= 4.39 processors support this)
      2. Fallback: manual vicuna template
    """
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
    ]
    try:
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    except (AttributeError, Exception):
        # Vicuna chat template used by llava-1.6-vicuna-7b-hf
        return (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
            f"USER: <image>\n{text} ASSISTANT:"
        )


def _apply_qwen2vl_template(processor: Any, text: str, image: Any) -> Dict[str, Any]:
    """Apply Qwen2-VL chat template and process inputs."""
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}
    ]
    try:
        # Newer transformers: apply_chat_template with return_dict=True
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        from qwen_vl_utils import process_vision_info
        text_proc = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        return processor(
            text=[text_proc],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Unified input preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_inputs(
    processor: Any,
    image: Union[Image.Image, str],
    text: str,
    model_type: str = "llava",
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Prepare model inputs with correct chat template for each model family.

    Args:
        processor:    HuggingFace processor
        image:        PIL Image or file path
        text:         User question / prompt (plain text, no template)
        model_type:   "llava" | "qwen2_vl"

    Returns:
        Dict with input_ids, attention_mask, pixel_values (and image_grid_thw for Qwen2-VL)
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    if model_type == "llava":
        formatted_text = _apply_llava_template(processor, text)
        inputs = processor(
            text=formatted_text,
            images=image,
            return_tensors=return_tensors,
            padding=True,
        )

    elif model_type == "qwen2_vl":
        inputs = _apply_qwen2vl_template(processor, text, image)

    else:
        # Generic fallback
        inputs = processor(
            text=text,
            images=image,
            return_tensors=return_tensors,
            padding=True,
        )

    return inputs
