#!/usr/bin/env python3
"""
AutoDL/Linux runtime sanity checks for CESD project.

Checks:
1) torch + cuda visibility
2) huggingface model loading (optional)
3) dataset path availability
4) DataLoader multiprocessing
5) rough OOM risk hints
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.model_loader import load_model, get_model_config, prepare_inputs


def _check_torch_cuda() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        idx = 0
        free_b, total_b = torch.cuda.mem_get_info(idx)
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_free_gb": round(free_b / (1024 ** 3), 2),
                "cuda_total_gb": round(total_b / (1024 ** 3), 2),
                "inference_device": str(torch.device("cuda")),
            }
        )
    else:
        info["inference_device"] = str(torch.device("cpu"))
    return info


def _check_dataset_paths(data_root: Path) -> Dict[str, Any]:
    checks = {
        "pope_random_json": data_root / "pope" / "coco_pope_random.json",
        "pope_popular_json": data_root / "pope" / "coco_pope_popular.json",
        "pope_adversarial_json": data_root / "pope" / "coco_pope_adversarial.json",
        "coco_val2014_dir": data_root / "mscoco" / "val2014",
        "coco_instances": data_root / "mscoco" / "annotations" / "instances_val2014.json",
    }
    out: Dict[str, Any] = {}
    for k, p in checks.items():
        out[k] = {"path": str(p), "exists": p.exists()}
    return out


def _check_dataloader_multiprocessing(num_workers: int = 2) -> Dict[str, Any]:
    from torch.utils.data import Dataset, DataLoader

    class _ToyDataset(Dataset):
        def __len__(self) -> int:
            return 128

        def __getitem__(self, idx: int) -> Dict[str, int]:
            return {"x": idx}

    result: Dict[str, Any] = {"num_workers": num_workers, "ok": False, "error": ""}
    try:
        loader = DataLoader(_ToyDataset(), batch_size=16, num_workers=num_workers, shuffle=False)
        n_batches = 0
        for _ in loader:
            n_batches += 1
            if n_batches >= 2:
                break
        result["ok"] = True
        result["tested_batches"] = n_batches
    except Exception as e:
        result["error"] = str(e)
    return result


def _check_hf_model_load(model_name: str, skip_model: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"model": model_name, "skipped": skip_model}
    if skip_model:
        return out

    try:
        model, processor = load_model(model_name)
        cfg = get_model_config(model_name)
        model_type = cfg.get("model_type", model_name)

        # Use an in-memory image to validate processor + input pipeline.
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        inputs = prepare_inputs(processor, img, "Is there a cat in this image?", model_type=model_type)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # Run one token generation for smoke test.
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                image_sizes=inputs.get("image_sizes"),
                max_new_tokens=1,
                do_sample=False,
            )

        out.update(
            {
                "ok": True,
                "model_type": model_type,
                "model_path_or_name": cfg.get("model_path") or cfg.get("model_name"),
            }
        )
    except Exception as e:
        out.update({"ok": False, "error": str(e)})
    return out


def _oom_risk_hint(cuda_info: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    hint: Dict[str, Any] = {"model": model_name}
    if not cuda_info.get("cuda_available", False):
        hint["risk"] = "HIGH"
        hint["reason"] = "CUDA unavailable. VLM inference on CPU is impractical."
        return hint

    total = float(cuda_info.get("cuda_total_gb", 0.0))
    free = float(cuda_info.get("cuda_free_gb", 0.0))
    # 7B VLM fp16 + KV cache + overhead; CESD double forward per token.
    if total < 20:
        level = "HIGH"
    elif total < 24:
        level = "MEDIUM"
    else:
        level = "LOW"

    suggestions = [
        "优先使用 24GB+ GPU（4090/A100）",
        "先用 --num_samples 10 做冒烟测试",
        "OOM 时优先降低 max_new_tokens（16/32）",
        "必要时仅先跑 Greedy/iTaD，再跑 CESD",
    ]
    if free < 8:
        suggestions.insert(0, "当前空闲显存较低，先清理占用进程（nvidia-smi）")

    hint.update({"risk": level, "cuda_total_gb": total, "cuda_free_gb": free, "suggestions": suggestions})
    return hint


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Linux/AutoDL runtime readiness")
    parser.add_argument("--model", default="llava", choices=["llava", "qwen2_vl"])
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument("--skip_model", action="store_true", help="Skip HF model loading smoke test")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report: Dict[str, Any] = {}
    report["torch_cuda"] = _check_torch_cuda()
    report["dataset_paths"] = _check_dataset_paths(Path(args.data_root))
    report["dataloader_multiprocessing"] = _check_dataloader_multiprocessing(num_workers=2)
    report["hf_model_loading"] = _check_hf_model_load(args.model, args.skip_model)
    report["oom_risk"] = _oom_risk_hint(report["torch_cuda"], args.model)

    print("\n=== Runtime Check Report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report -> {out_path}")


if __name__ == "__main__":
    main()
