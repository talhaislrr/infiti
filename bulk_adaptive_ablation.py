#!/usr/bin/env python3
"""
Adaptif trigger A/B test — Faz 2b
=================================
Sabit stride vs embedding/loss sürpriz tetikleyici.

Kullanım:
  python3 bulk_adaptive_ablation.py --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import TinyLlamaWithBulk, load_tinyllama
from bulk_state import BulkStateManager


def count_triggers(
    hidden: torch.Tensor,
    k_short: int,
    trigger_stride: int,
    adaptive: bool,
    surprise_threshold: float,
) -> dict:
    import torch.nn.functional as F
    from bulk_state import BulkStateTensors, embedding_surprise

    B, T, H = hidden.shape
    padded = F.pad(hidden, (0, 0, k_short - 1, 0))
    windows = padded.unfold(1, k_short, 1).permute(0, 1, 3, 2).contiguous()

    mgr = BulkStateManager(
        H, k_short=k_short, adaptive_trigger=adaptive,
        surprise_threshold=surprise_threshold,
    )
    surprise = embedding_surprise(hidden) if adaptive else None

    short_runs = 0
    state = BulkStateTensors.zeros(B, H, hidden.device, hidden.dtype)
    last_short = None
    for t in range(T):
        sur_t = surprise[:, t] if surprise is not None else None
        if mgr._should_run_short_encoder(t, trigger_stride, last_short, sur_t):
            short_runs += 1
            state = mgr.update(windows[:, t], state)
            last_short = state.short
        else:
            state = mgr._advance_without_short(state, last_short)

    return {
        "seq_len": T,
        "short_encoder_runs": short_runs,
        "trigger_rate": round(short_runs / max(T, 1), 4),
        "adaptive": adaptive,
        "trigger_stride": trigger_stride,
    }


@torch.no_grad()
def bench_prefill(hybrid: TinyLlamaWithBulk, ids: torch.Tensor) -> float:
    t0 = time.perf_counter()
    hybrid.prefill(ids, fast=True)
    return (time.perf_counter() - t0) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--trigger-stride", type=int, default=4)
    parser.add_argument("--surprise-threshold", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_adaptive_ablation_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)
    print("=" * 70)
    print(f"Adaptif Trigger A/B | {device_summary(device)}")
    print("=" * 70)

    base, tokenizer, model_path = load_tinyllama(device, dtype)
    text = "Science and history intertwine across epochs. "
    while len(tokenizer.encode(text)) < args.prompt_tokens:
        text += text
    ids = torch.tensor([tokenizer.encode(text)[: args.prompt_tokens]], dtype=torch.long).to(device)

    with torch.no_grad():
        hidden = base.model(ids).last_hidden_state.float()

    fixed_stats = count_triggers(
        hidden, k_short=8, trigger_stride=args.trigger_stride,
        adaptive=False, surprise_threshold=args.surprise_threshold,
    )
    adapt_stats = count_triggers(
        hidden, k_short=8, trigger_stride=args.trigger_stride,
        adaptive=True, surprise_threshold=args.surprise_threshold,
    )

    configs = [
        ("fixed_stride", False),
        ("adaptive", True),
    ]
    bench_rows = []
    for name, adaptive in configs:
        h = TinyLlamaWithBulk(
            base, k_short=8, trigger_stride=args.trigger_stride,
            adaptive_trigger=adaptive, surprise_threshold=args.surprise_threshold,
            freeze_base=True,
        ).to(device)
        ms = bench_prefill(h, ids)
        bench_rows.append({"name": name, "adaptive": adaptive, "prefill_ms": round(ms, 2)})
        print(f"  {name:14} prefill={ms:.1f}ms")

    print(f"\nFixed short-encoder: {fixed_stats['short_encoder_runs']}/{fixed_stats['seq_len']} "
          f"({fixed_stats['trigger_rate']:.1%})")
    print(f"Adapt short-encoder: {adapt_stats['short_encoder_runs']}/{adapt_stats['seq_len']} "
          f"({adapt_stats['trigger_rate']:.1%})")

    results = {
        "device": str(device),
        "model": model_path,
        "prompt_tokens": args.prompt_tokens,
        "fixed_stride": fixed_stats,
        "adaptive": adapt_stats,
        "prefill_benchmark": bench_rows,
        "compute_saved_pct": round(
            (1 - adapt_stats["short_encoder_runs"] / max(fixed_stats["short_encoder_runs"], 1)) * 100, 1
        ),
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
