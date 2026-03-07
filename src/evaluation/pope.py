"""
POPE: Polling-based Object Probing Evaluation.

Evaluates object hallucination via yes/no questions.
Metrics: Accuracy, Precision, Recall, F1.
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.utils.runtime import get_inference_device, move_inputs_to_device


def load_pope_data(
    data_path: str,
    split: str = "random",
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load POPE dataset.

    Expected format (JSON lines or JSON list):
    [{"image": "path_or_id", "text": "Is there a X?", "answer": "yes"/"no"}, ...]

    Or CSV: image,text,answer
    """
    data_path = Path(data_path)
    split_path = data_path / split
    json_path = split_path / f"coco_pope_{split}.json"
    if not json_path.exists():
        json_path = data_path / f"coco_pope_{split}.json"
    if not json_path.exists():
        json_path = data_path / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"POPE data not found. Please download from "
            "https://github.com/RUCAIBox/POPE and place in {data_path}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if num_samples:
        data = data[:num_samples]
    return data


def parse_answer(text: str) -> str:
    """Extract yes/no from model output."""
    text = text.strip().lower()
    if not text:
        return "no"
    # Take first word or first yes/no
    words = text.split()
    for w in words[:5]:
        if w in ("yes", "no"):
            return w
    if "yes" in text[:20]:
        return "yes"
    if "no" in text[:20]:
        return "no"
    return "no"


def compute_pope_metrics(
    preds: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall, F1."""
    preds_bin = [1 if p.lower() == "yes" else 0 for p in preds]
    labels_bin = [1 if l.lower() == "yes" else 0 for l in labels]

    acc = accuracy_score(labels_bin, preds_bin)
    prec = precision_score(labels_bin, preds_bin, zero_division=0)
    rec = recall_score(labels_bin, preds_bin, zero_division=0)
    f1 = f1_score(labels_bin, preds_bin, zero_division=0)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


class POPEEvaluator:
    """POPE benchmark evaluator."""

    def __init__(
        self,
        data_path: str = "data/pope",
        coco_root: Optional[str] = None,
        splits: Optional[List[str]] = None,
        num_samples: Optional[int] = 500,
    ):
        self.data_path = Path(data_path)
        self.coco_root = Path(coco_root or os.environ.get("COCO_ROOT", "data/mscoco/val2014"))
        self.splits = splits or ["random", "popular", "adversarial"]
        self.num_samples = num_samples

    def evaluate(
        self,
        model,
        processor,
        decode_fn: Callable,
        model_type: str = "llava",
        splits: Optional[List[str]] = None,
        **decode_kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run POPE evaluation.

        Args:
            model, processor: VLM and processor
            decode_fn: Function (model, inputs) -> generated_ids
            model_type: "llava" or "qwen2_vl"
            splits: Which splits to evaluate

        Returns:
            {split: {accuracy, precision, recall, f1}}
        """
        try:
            from src.models.model_loader import prepare_inputs
        except ImportError:
            from models.model_loader import prepare_inputs

        splits = splits or self.splits
        results = {}
        device = get_inference_device()

        for split in splits:
            try:
                data = load_pope_data(
                    str(self.data_path), split=split, num_samples=self.num_samples
                )
            except FileNotFoundError as e:
                print(f"Skip {split}: {e}")
                results[split] = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
                continue

            preds, labels = [], []
            for item in tqdm(data, desc=f"POPE {split}"):
                img_key = item.get("image", item.get("image_path", ""))
                if isinstance(img_key, (int, float)):
                    img_key = str(int(img_key))
                image_path = Path(str(img_key))
                if not image_path.is_absolute():
                    image_path = self.coco_root / image_path
                if (not image_path.exists()) and image_path.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                    try:
                        image_path = self.coco_root / f"COCO_val2014_{int(img_key):012d}.jpg"
                    except (ValueError, TypeError):
                        image_path = self.coco_root / image_path
                text = item.get("text", item.get("question", ""))
                label = item.get("answer", item.get("label", "no")).lower()
                if label not in ("yes", "no"):
                    label = "yes" if label == "1" else "no"

                try:
                    inputs = prepare_inputs(processor, str(image_path), text, model_type)
                    inputs = move_inputs_to_device(inputs, device)
                    gen_ids = decode_fn(
                        model,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        max_new_tokens=16,
                        **decode_kwargs,
                    )
                    out_text = processor.decode(
                        gen_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    pred = parse_answer(out_text)
                except Exception as e:
                    pred = "no"
                    print(f"Error on {str(image_path)}: {e}")
                preds.append(pred)
                labels.append(label)

            results[split] = compute_pope_metrics(preds, labels)

        return results
