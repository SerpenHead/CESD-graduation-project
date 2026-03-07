"""
MME: Multimodal Model Evaluation benchmark.

Perception + Cognition tasks. Simplified evaluator that runs a subset.
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from tqdm import tqdm


def load_mme_data(data_path: str, task: Optional[str] = None) -> List[Dict]:
    """Load MME task data. Format: [{image, question, answer}, ...]"""
    path = Path(data_path)
    if task:
        fpath = path / f"{task}.json"
    else:
        fpath = path / "mme.json"
    if not fpath.exists():
        return []
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_mme_answer(text: str) -> str:
    """Extract answer from model output for MME (often single letter or short)."""
    text = text.strip().upper()
    if not text:
        return ""
    for c in text[:50]:
        if c in "ABCD":
            return c
    return text[:1] if text else ""


class MMEEvaluator:
    """MME benchmark evaluator (simplified)."""

    def __init__(self, data_path: str = "data/mme", num_samples: Optional[int] = None):
        self.data_path = Path(data_path)
        self.num_samples = num_samples

    def evaluate(
        self,
        model,
        processor,
        decode_fn: Callable,
        model_type: str = "llava",
        tasks: Optional[List[str]] = None,
        **decode_kwargs,
    ) -> Dict[str, Any]:
        """Run MME evaluation. Returns perception/cognition scores."""
        try:
            from src.models.model_loader import prepare_inputs
        except ImportError:
            from models.model_loader import prepare_inputs

        tasks = tasks or ["existence", "count", "position", "color"]
        results = {}
        for task in tasks:
            data = load_mme_data(str(self.data_path), task)
            if not data:
                results[task] = {"accuracy": 0, "num": 0}
                continue
            if self.num_samples:
                data = data[: self.num_samples]
            correct = 0
            for item in tqdm(data, desc=f"MME {task}"):
                img_path = item.get("image", "")
                if not os.path.isabs(img_path):
                    img_path = str(self.data_path / img_path)
                q = item.get("question", item.get("text", ""))
                gt = str(item.get("answer", item.get("label", ""))).strip().upper()
                try:
                    inputs = prepare_inputs(processor, img_path, q, model_type)
                    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                    gen_ids = decode_fn(
                        model,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        max_new_tokens=32,
                        **decode_kwargs,
                    )
                    out = processor.decode(
                        gen_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    pred = parse_mme_answer(out)
                    if pred and gt and pred[0] == gt[0]:
                        correct += 1
                except Exception:
                    pass
            results[task] = {"accuracy": correct / len(data) if data else 0, "num": len(data)}

        perception = [results.get(t, {}).get("accuracy", 0) for t in ["existence", "count", "position", "color"]]
        cognition = [results.get(t, {}).get("accuracy", 0) for t in ["commonsense", "numerical", "text", "symbol"]]
        return {
            "perception": sum(perception) / len(perception) if perception else 0,
            "cognition": sum(cognition) / len(cognition) if cognition else 0,
            "tasks": results,
        }
