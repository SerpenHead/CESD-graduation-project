"""Inference timing utilities: measure Tokens Per Second (TPS)."""

import time
import torch
from typing import Optional


class TPSMeter:
    """
    Context-manager and callable to measure tokens-per-second.

    Usage:
        meter = TPSMeter()
        with meter:
            output_ids = decoder(model, input_ids, ...)
        print(f"TPS: {meter.tps:.2f}")
    """

    def __init__(self):
        self._start: float = 0.0
        self._end: float = 0.0
        self.new_tokens: int = 0
        self.elapsed_s: float = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._end = time.perf_counter()
        self.elapsed_s = self._end - self._start

    def record(self, input_len: int, output_len: int):
        """Record the number of newly generated tokens."""
        self.new_tokens = output_len - input_len

    @property
    def tps(self) -> float:
        """Tokens per second (0 if not recorded yet)."""
        if self.elapsed_s <= 0 or self.new_tokens <= 0:
            return 0.0
        return self.new_tokens / self.elapsed_s


def measure_tps(
    decoder,
    model,
    inputs: dict,
    max_new_tokens: int = 128,
    n_warmup: int = 1,
    n_runs: int = 3,
    **decode_kwargs,
) -> dict:
    """
    Measure average TPS for a decoder over multiple runs.

    Args:
        decoder:        A decoder callable (GreedyDecoder, CESDDecoder, …)
        model:          VLM model
        inputs:         Dict from prepare_inputs (input_ids, pixel_values, …)
        max_new_tokens: Max tokens to generate per run
        n_warmup:       Warm-up runs (not counted)
        n_runs:         Timed runs

    Returns:
        dict with keys: tps_mean, tps_std, elapsed_mean_s
    """
    import numpy as np

    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
    pix = inputs.get("pixel_values")
    if pix is not None:
        pix = pix.to(device)
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        grid = grid.to(device)
    img_sizes = inputs.get("image_sizes")
    if img_sizes is not None:
        img_sizes = img_sizes.to(device)

    prompt_len = input_ids.shape[1]

    tps_list = []
    elapsed_list = []
    for i in range(n_warmup + n_runs):
        meter = TPSMeter()
        with meter:
            gen_ids = decoder(
                model,
                input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=pix,
                image_grid_thw=grid,
                image_sizes=img_sizes,
                max_new_tokens=max_new_tokens,
                **decode_kwargs,
            )
        meter.record(prompt_len, gen_ids.shape[1])
        if i >= n_warmup:
            tps_list.append(meter.tps)
            elapsed_list.append(meter.elapsed_s)

    return {
        "tps_mean": float(np.mean(tps_list)),
        "tps_std": float(np.std(tps_list)),
        "elapsed_mean_s": float(np.mean(elapsed_list)),
    }
