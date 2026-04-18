"""
MME: Multimodal Model Evaluation benchmark.

Perception + Cognition tasks. Simplified evaluator that runs a subset.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from tqdm import tqdm

from src.utils.runtime import get_inference_device, move_inputs_to_device

PERCEPTION_TASKS = ["existence", "count", "position", "color"]
COGNITION_TASKS = ["commonsense", "numerical", "text", "symbol"]


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

        tasks = tasks or (PERCEPTION_TASKS + COGNITION_TASKS)
        results = {}
        device = get_inference_device()
        for task in tasks:
            data = load_mme_data(str(self.data_path), task)
            if not data:
                results[task] = {"accuracy": 0, "num": 0, "num_failed": 0}
                continue
            if self.num_samples:
                data = data[: self.num_samples]
            correct = 0
            failed = 0
            for item in tqdm(data, desc=f"MME {task}"):
                img_path = Path(item.get("image", ""))
                if not img_path.is_absolute():
                    img_path = self.data_path / img_path
                q = item.get("question", item.get("text", ""))
                gt = str(item.get("answer", item.get("label", ""))).strip().upper()
                try:
                    inputs = prepare_inputs(processor, str(img_path), q, model_type)
                    inputs = move_inputs_to_device(inputs, device)
                    gen_ids = decode_fn(
                        model,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        image_sizes=inputs.get("image_sizes"),
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
                except Exception as e:
                    failed += 1
                    if failed <= 3:
                        print(f"[MME:{task}] sample failed ({img_path}): {e}")
            if failed == len(data):
                print(f"[MME:{task}] WARNING: all samples failed; check data path/model inputs.")
            results[task] = {
                "accuracy": correct / len(data) if data else 0,
                "num": len(data),
                "num_failed": failed,
            }

        def _group_mean(task_names: List[str]) -> float:
            group_scores = [
                results[t]["accuracy"]
                for t in task_names
                if t in results and results[t].get("num", 0) > 0
            ]
            return sum(group_scores) / len(group_scores) if group_scores else 0.0

        return {
            "perception": _group_mean(PERCEPTION_TASKS),
            "cognition": _group_mean(COGNITION_TASKS),
            "tasks": results,
        }
