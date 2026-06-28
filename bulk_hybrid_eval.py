#!/usr/bin/env python3
"""
BulkTrigger hibrit değerlendirme — TinyLlama gerekmez (v2 PoC)
==============================================================
Adaptif trigger, uzun menzil recall, bellek tahmini.

Kullanım:
  python3 bulk_hybrid_eval.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bulk_memory_utils import estimate_bulk_state_bytes, estimate_kv_cache_bytes
from bulk_trigger_arch import BulkTriggerLM, StandardDecoderLM, count_params
from bulk_trigger_v2 import BulkTriggerLMv2
from bulk_trigger_benchmark_v2 import (
    LongRangeRecallDataset,
    PatternDataset,
    train_epoch,
    eval_acc,
    eval_recall_at_pos,
    bench_bulk_v1,
    bench_bulk_v2,
)
from bulk_device import device_summary, pick_device


def compare_memory(d_model: int, n_layers: int, seq_lens: list[int]) -> list[dict]:
    rows = []
    bulk_b = estimate_bulk_state_bytes(n_layers, d_model)
    for sl in seq_lens:
        kv = estimate_kv_cache_bytes(sl, n_layers, d_model)
        rows.append({
            "seq_len": sl,
            "bulk_bytes": bulk_b,
            "bulk_kb": round(bulk_b / 1024, 2),
            "kv_bytes": kv,
            "kv_mb": round(kv / 1e6, 4),
            "ratio": round(kv / max(bulk_b, 1), 1),
        })
    return rows


def run_adaptive_ablation(device, quick: bool) -> dict:
    vocab, d, gap = 32, 128, 48 if quick else 64
    epochs = 12 if quick else 20
    tl = DataLoader(LongRangeRecallDataset(vocab, gap, 400), 32, shuffle=True)
    te = LongRangeRecallDataset(vocab, gap, 150)

    configs = [
        ("stride8", {"trigger_stride": 8, "adaptive_trigger": False}),
        ("stride4", {"trigger_stride": 4, "adaptive_trigger": False}),
        ("adaptive", {"trigger_stride": 8, "adaptive_trigger": True, "surprise_threshold": 0.5}),
    ]
    results = {}
    crit = nn.CrossEntropyLoss()
    for name, kw in configs:
        m = BulkTriggerLMv2(vocab, d_model=d, n_layers=2, k_short=8, long_interval=64, **kw).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
        t0 = time.time()
        for _ in range(epochs):
            train_epoch(m, tl, opt, crit, device)
        recall = eval_recall_at_pos(m, te, device)
        results[name] = {
            "recall": round(recall, 4),
            "params": count_params(m),
            "train_sec": round(time.time() - t0, 1),
            **kw,
        }
        print(f"  {name}: recall={recall:.4f}  ({results[name]['train_sec']}s)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_hybrid_eval_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    d, n_layers = 128, 2
    seq_lens = [512, 2048, 8192, 32768] if not args.quick else [512, 4096, 16384]

    print("=" * 70)
    print(f"BulkTrigger Hibrit Eval | {device_summary(device)}")
    print("=" * 70)

    print("\n[1] Bellek: BulkState vs KV-cache")
    mem = compare_memory(d, n_layers, seq_lens)
    for r in mem:
        print(f"  seq={r['seq_len']:6d}  bulk={r['bulk_kb']:6.1f} KB  KV={r['kv_mb']:.2f} MB  ratio={r['ratio']:.0f}x")

    print("\n[2] Adaptif trigger ablation (v2)")
    adaptive = run_adaptive_ablation(device, args.quick)

    print("\n[3] Latency O(1) check (v2)")
    m = BulkTriggerLMv2(32, d_model=128, n_layers=2).to(device)
    lat = bench_bulk_v2(m, device, [64, 256, 512], repeats=5)
    o1 = lat[-1]["ms_per_tok"] < lat[0]["ms_per_tok"] * 2.5
    print(f"  {lat}  O1={'OK' if o1 else 'FAIL'}")

    out = {
        "device": str(device),
        "memory_comparison": mem,
        "adaptive_ablation": adaptive,
        "latency_v2": lat,
        "o1_verified": o1,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
