#!/usr/bin/env python3
"""
Gerçek bellek profili — Faz 3
=============================
PyTorch allocator + psutil ile KV vs BulkState karşılaştırması.

Kullanım:
  python3 bulk_memory_profile.py --prompt-lens 512,2048,8192 --device mps
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import BaseKVGenerator, TinyLlamaWithBulk, create_hybrid, load_tinyllama
from bulk_memory_utils import estimate_bulk_state_bytes, estimate_kv_cache_bytes


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return 0.0


def _torch_alloc_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e6
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


def _reset_mem(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    if device.type == "mps":
        torch.mps.empty_cache()


def profile_prefill(
    label: str,
    fn,
    device: torch.device,
) -> dict:
    _reset_mem(device)
    rss_before = _rss_mb()
    alloc_before = _torch_alloc_mb(device)
    fn()
    alloc_after = _torch_alloc_mb(device)
    rss_after = _rss_mb()
    peak = alloc_after
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e6
    return {
        "label": label,
        "torch_alloc_delta_mb": round(alloc_after - alloc_before, 3),
        "torch_peak_mb": round(peak, 3),
        "rss_delta_mb": round(rss_after - rss_before, 3),
    }


def make_prompt(tokenizer, target: int) -> torch.Tensor:
    text = "The history of science spans centuries of discovery. "
    while len(tokenizer.encode(text)) < target:
        text += text
    return torch.tensor([tokenizer.encode(text)[:target]], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lens", default="512,2048,8192")
    parser.add_argument("--adapter", default="checkpoints/bulk_adapter/adapter.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_memory_profile_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)
    lens = [int(x.strip()) for x in args.prompt_lens.split(",") if x.strip()]

    print("=" * 70)
    print(f"BulkTrigger Bellek Profili | {device_summary(device)}")
    print("=" * 70)

    base, tokenizer, model_path = load_tinyllama(device, dtype)
    H = base.config.hidden_size
    n_layers = base.config.num_hidden_layers

    adapter_path = Path(args.adapter)
    if adapter_path.exists():
        hybrid, _, _ = create_hybrid(
            device, dtype, adapter_path, base_model=base,
            k_short=8, trigger_stride=4,
        )
    else:
        hybrid = TinyLlamaWithBulk(base, freeze_base=True, k_short=8, trigger_stride=4).to(device)

    kv_gen = BaseKVGenerator(base)
    rows = []

    print(f"\n{'Prompt':>8} {'KV teorik':>10} {'Bulk KB':>8} {'KV ΔMB':>8} {'Bulk ΔMB':>9}")
    for plen in lens:
        ids = make_prompt(tokenizer, plen).to(device)

        kv_prof = profile_prefill(
            "base_kv",
            lambda: kv_gen.prefill(ids),
            device,
        )
        bulk_prof = profile_prefill(
            "bulk",
            lambda: hybrid.prefill(ids, fast=True),
            device,
        )

        kv_theory = estimate_kv_cache_bytes(plen, n_layers, H) / 1e6
        bulk_theory_kb = estimate_bulk_state_bytes(1, H) / 1024

        print(
            f"{plen:>8} {kv_theory:>8.2f}MB {bulk_theory_kb:>6.1f}KB "
            f"{kv_prof['torch_alloc_delta_mb']:>7.2f} {bulk_prof['torch_alloc_delta_mb']:>8.2f}"
        )
        rows.append({
            "prompt_len": plen,
            "kv_theory_mb": round(kv_theory, 4),
            "bulk_state_kb": round(bulk_theory_kb, 2),
            "base_kv_profile": kv_prof,
            "bulk_profile": bulk_prof,
        })

    results = {
        "device": str(device),
        "model": model_path,
        "hidden_size": H,
        "n_layers": n_layers,
        "profiles": rows,
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
