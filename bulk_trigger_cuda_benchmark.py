#!/usr/bin/env python3
"""
BulkTrigger CUDA fp16 inference benchmark
========================================
Kullanım:
  python3 bulk_trigger_cuda_benchmark.py
  python3 bulk_trigger_cuda_benchmark.py --seq-lens 128 512 2048 4096
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import torch

from bulk_trigger_arch import BulkTriggerLM, StandardDecoderLM, estimate_kv_cache_bytes
from bulk_trigger_v2 import BulkTriggerLMv2


def bench_v2_fp16(model, device, seq_len: int, repeats: int = 20) -> dict:
    model.eval().half()
    vocab = model.head.out_features
    times = []
    for _ in range(repeats):
        hist: deque = deque(maxlen=model.k_short)
        states = model.init_bulk_states(1, device, torch.float16)
        pos = 0
        tok = torch.randint(1, vocab, (1,), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(seq_len):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, emb, states = model.generate_step(tok, hist, states, pos)
            hist.append(emb)
            pos += 1
            tok = logits.argmax(-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / seq_len * 1000)
    mem_kb = model.k_short * model.d_model * 2 / 1024
    return {
        "seq_len": seq_len,
        "ms_per_tok": round(sum(times) / len(times), 3),
        "bulk_mem_kb": round(mem_kb, 3),
        "dtype": "fp16",
    }


def bench_std_fp16(model, device, seq_len: int, repeats: int = 5) -> dict:
    model.eval().half()
    vocab = model.head.out_features
    times = []
    for _ in range(repeats):
        ids = torch.randint(1, vocab, (1, min(seq_len, 128)), device=device, dtype=torch.long)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps = max(1, seq_len // ids.size(1))
        for _ in range(steps):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model.generate_step_full(ids)
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_len = ids.size(1) - min(seq_len, 128)
        times.append((time.perf_counter() - t0) / max(gen_len, 1) * 1000)
    kv_mb = estimate_kv_cache_bytes(seq_len, len(model.layers), model.d_model) / 1e6 / 2
    return {
        "seq_len": seq_len,
        "ms_per_tok": round(sum(times) / len(times), 3),
        "kv_mb_fp16": round(kv_mb, 4),
        "dtype": "fp16",
        "note": "KV-cache'siz full-recompute baseline",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--output", default="bulk_trigger_cuda_results.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA yok — CPU fp32 fallback")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    vocab, d = 256, 256
    v2 = BulkTriggerLMv2(vocab, d_model=d, k_short=8, n_layers=4, max_len=8192).to(device)
    std = StandardDecoderLM(vocab, d_model=d, n_layers=4, max_len=8192).to(device)

    results = {"device": str(device), "v2": [], "standard": []}
    for slen in args.seq_lens:
        try:
            r2 = bench_v2_fp16(v2, device, slen)
            results["v2"].append(r2)
            print(f"v2  seq={slen:5d}  {r2['ms_per_tok']:.3f} ms/tok  mem={r2['bulk_mem_kb']:.1f} KB")
        except RuntimeError as e:
            print(f"v2  seq={slen} OOM/fail: {e}")
            results["v2"].append({"seq_len": slen, "error": str(e)})

        if slen <= 2048:
            try:
                rs = bench_std_fp16(std, device, slen)
                results["standard"].append(rs)
                print(f"std seq={slen:5d}  {rs['ms_per_tok']:.3f} ms/tok  KV={rs['kv_mb_fp16']:.2f} MB")
            except RuntimeError as e:
                results["standard"].append({"seq_len": slen, "error": str(e)})

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
