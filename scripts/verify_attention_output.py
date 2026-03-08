#!/usr/bin/env python3
"""
验证 CESD/iTaD 是否真正使用了对比解码，还是因 attentions 为 None 退化为 Greedy。

1) 用默认配置加载模型，做一次 forward(output_attentions=True)，检查 attentions 是否为 None。
2) 用 attn_implementation="eager" 加载模型，再做一次 forward，确认能拿到 attentions。
3) 可选：用默认模型跑 2 条 POPE，统计 CESD 的 contrastive vs fallback 步数。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_pope_one(data_path: Path, coco_root: Path, split: str = "random"):
    """加载 POPE 一条数据用于单次 forward."""
    import json
    json_path = data_path / f"coco_pope_{split}.json"
    if not json_path.exists():
        json_path = data_path / split / f"coco_pope_{split}.json"
    if not json_path.exists():
        return None, None, None
    with open(json_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        data = json.loads(raw)
    else:
        data = [json.loads(line) for line in raw.split("\n") if line.strip()]
    if not data:
        return None, None, None
    item = data[0]
    img_key = item.get("image", item.get("image_path", ""))
    if isinstance(img_key, (int, float)):
        img_key = str(int(img_key))
    image_path = Path(img_key)
    if not image_path.is_absolute():
        image_path = coco_root / image_path
    if not image_path.exists():
        image_path = coco_root / f"COCO_val2014_{int(img_key):012d}.jpg"
    text = item.get("text", item.get("question", "Is there a cat?"))
    return str(image_path), text, image_path.exists()


def main():
    parser = argparse.ArgumentParser(description="Verify attention output for CESD/iTaD")
    parser.add_argument("--model", default="llava", choices=["llava", "qwen2_vl"])
    parser.add_argument("--data_path", default="data/pope")
    parser.add_argument("--coco_root", default="data/mscoco/val2014")
    parser.add_argument("--run_cesd_samples", type=int, default=0,
                        help="Run N POPE samples with CESD (default model) and print stats")
    parser.add_argument("--skip_eager_test", action="store_true",
                        help="Skip loading model with eager (saves GPU memory, avoid OOM)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    coco_root = Path(args.coco_root)

    from src.models.model_loader import load_model, get_model_config, prepare_inputs
    from src.utils.runtime import get_inference_device, move_inputs_to_device
    from src.decoding import CESDDecoder

    config = get_model_config(args.model)
    model_type = config.get("model_type", args.model)

    # ── 1) 默认加载，单次 forward 检查 attentions ─────────────────────────
    print("=" * 60)
    print("[1] 使用默认配置加载模型 (SDPA)，做一次 forward(output_attentions=True)")
    print("=" * 60)
    model_default, processor = load_model(args.model)
    device = get_inference_device()

    img_path, prompt, ok = load_pope_one(data_path, coco_root)
    if not ok or not img_path:
        print("  警告: 未找到 POPE 图片，使用随机输入做最小测试")
        import torch
        # 最小输入：仅验证 forward 是否返回 attentions
        batch = 1
        seq_len = 10
        input_ids = torch.randint(0, 32000, (batch, seq_len), device=device)
        attention_mask = torch.ones(batch, seq_len, device=device)
        # LLaVA 需要 pixel_values，没有的话可能报错；这里仅做 attentions 检查
        try:
            out = model_default(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )
        except Exception as e:
            print(f"  默认模型 forward 失败 (可能缺 pixel_values): {e}")
            out = None
        if out is not None:
            attn_ok = out.attentions is not None and len(out.attentions) > 0
            hid_ok = out.hidden_states is not None and len(out.hidden_states) > 0
            print(f"  attentions is None: {out.attentions is None}  (若 True 则 CESD/iTaD 会退化为 Greedy)")
            print(f"  hidden_states is None: {out.hidden_states is None}")
            if out.attentions is not None:
                print(f"  attentions 层数: {len(out.attentions)}")
    else:
        inputs = prepare_inputs(processor, img_path, prompt, model_type)
        if inputs is None:
            print("  prepare_inputs 返回 None，跳过")
        else:
            inputs = move_inputs_to_device(inputs, device)
            with __import__("torch").no_grad():
                out = model_default(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    image_sizes=inputs.get("image_sizes"),
                    output_hidden_states=True,
                    output_attentions=True,
                )
            attn_none = out.attentions is None
            hid_none = out.hidden_states is None
            print(f"  attentions is None: {attn_none}  (若 True 则 CESD/iTaD 会退化为 Greedy)")
            print(f"  hidden_states is None: {hid_none}")
            if out.attentions is not None:
                print(f"  attentions 层数: {len(out.attentions)}")

    # ── 2) 使用 attn_implementation="eager" 加载，再测一次（可选，省显存可 --skip_eager_test）────
    if not args.skip_eager_test:
        print()
        print("=" * 60)
        print("[2] 使用 attn_implementation='eager' 加载模型，再做一次 forward")
        print("=" * 60)
        del model_default
        import torch as _torch
        _torch.cuda.empty_cache()
        model_eager, _ = load_model(args.model, attn_implementation="eager")
        if ok and img_path:
            inputs = prepare_inputs(processor, img_path, prompt, model_type)
            if inputs is not None:
                inputs = move_inputs_to_device(inputs, device)
                with _torch.no_grad():
                    out_e = model_eager(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        image_sizes=inputs.get("image_sizes"),
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                print(f"  attentions is None: {out_e.attentions is None}")
                print(f"  hidden_states is None: {out_e.hidden_states is None}")
                if out_e.attentions is not None:
                    print(f"  attentions 层数: {len(out_e.attentions)}")
        del model_eager
        _torch.cuda.empty_cache()
    else:
        print("\n[2] 已跳过 (--skip_eager_test)。用 CESD/iTaD 评测时请以 attn_implementation='eager' 加载模型。")

    # ── 3) 可选：用默认模型跑几条 POPE + CESD，看 contrastive vs fallback 统计 ─────────────────
    if args.run_cesd_samples > 0 and ok and img_path:
        print()
        print("=" * 60)
        print(f"[3] 用默认模型跑 {args.run_cesd_samples} 条 POPE，CESD 步数统计")
        print("=" * 60)
        if args.skip_eager_test and "model_default" in dir() and model_default is not None:
            model_default2, processor2 = model_default, processor
        else:
            model_default2, processor2 = load_model(args.model)  # 故意不用 eager，验证是否会 fallback
        decoder = CESDDecoder(alpha=0.5, sparsify_ratio=0.2, model_type=model_type)
        json_path = data_path / "coco_pope_random.json"
        if not json_path.exists():
            json_path = data_path / "random" / "coco_pope_random.json"
        with open(json_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.split("\n") if l.strip()]
        n = min(args.run_cesd_samples, len(data))
        for i in range(n):
            item = data[i]
            img_key = item.get("image", item.get("image_path", ""))
            if isinstance(img_key, (int, float)):
                img_key = str(int(img_key))
            # POPE 可能是数字或 "COCO_val2014_000000310196.jpg"
            if isinstance(img_key, str) and img_key.endswith((".jpg", ".jpeg", ".png")):
                image_path = coco_root / img_key
            else:
                try:
                    image_path = coco_root / f"COCO_val2014_{int(img_key):012d}.jpg"
                except ValueError:
                    image_path = coco_root / str(img_key)
            if not image_path.exists():
                image_path = coco_root / str(img_key)
            text = item.get("text", "Is there a cat?")
            inputs = prepare_inputs(processor2, str(image_path), text, model_type)
            if inputs is None:
                continue
            inputs = move_inputs_to_device(inputs, device)
            decoder(
                model_default2,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                image_sizes=inputs.get("image_sizes"),
                max_new_tokens=16,
            )
        stats = decoder.get_and_reset_stats()
        print(f"  contrastive 步数: {stats['contrastive']}")
        print(f"  fallback 步数: {stats['fallback']}")
        total = stats["contrastive"] + stats["fallback"]
        if total > 0:
            pct = 100.0 * stats["fallback"] / total
            print(f"  fallback 比例: {pct:.1f}%  (若接近 100% 说明默认配置下 CESD 实际在当 Greedy 用)")

    print()
    print("验证结束。若 [1] 中 attentions is None 为 True，请在跑 CESD/iTaD 时用 attn_implementation='eager' 加载模型。")


if __name__ == "__main__":
    main()
