#!/usr/bin/env python3
"""
BulkTrigger hibrit inference — KV-cache + BulkState (Faz 2)
============================================================
Kullanım:
  python3 bulk_hybrid_infer.py --benchmark --device mps
  python3 bulk_hybrid_infer.py --benchmark --scale-prompt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import BaseKVGenerator, TinyLlamaWithBulk, create_hybrid, load_tinyllama
from bulk_memory_utils import estimate_bulk_state_bytes, estimate_kv_cache_bytes


@torch.no_grad()
def bench_recompute(model, input_ids, new_tokens: int) -> float:
    """Full recompute her adım (O(n²)) — eski yöntem."""
    ids = input_ids.clone()
    t0 = time.perf_counter()
    for _ in range(new_tokens):
        if isinstance(model, TinyLlamaWithBulk):
            logits = model(input_ids=ids).logits[:, -1, :]
        else:
            logits = model(input_ids=ids).logits[:, -1, :]
        ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
    return (time.perf_counter() - t0) / new_tokens * 1000


@torch.no_grad()
def bench_base_kv(gen: BaseKVGenerator, input_ids, new_tokens: int) -> dict:
    t0 = time.perf_counter()
    logits = gen.prefill(input_ids)
    prefill_ms = (time.perf_counter() - t0) * 1000
    tok = logits.argmax(-1, keepdim=True)
    t1 = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = gen.decode_step(tok.squeeze(1))
        tok = logits.argmax(-1, keepdim=True)
    decode_ms = (time.perf_counter() - t1) / max(new_tokens - 1, 1) * 1000
    return {"prefill_ms": round(prefill_ms, 2), "decode_ms_per_tok": round(decode_ms, 2)}


@torch.no_grad()
def bench_bulk_kv(hybrid: TinyLlamaWithBulk, input_ids, new_tokens: int, fast: bool = True) -> dict:
    t0 = time.perf_counter()
    logits = hybrid.prefill(input_ids, fast=fast)
    prefill_ms = (time.perf_counter() - t0) * 1000
    tok = logits.argmax(-1, keepdim=True)
    t1 = time.perf_counter()
    for _ in range(new_tokens - 1):
        logits = hybrid.decode_step(tok.squeeze(1))
        tok = logits.argmax(-1, keepdim=True)
    decode_ms = (time.perf_counter() - t1) / max(new_tokens - 1, 1) * 1000
    return {"prefill_ms": round(prefill_ms, 2), "decode_ms_per_tok": round(decode_ms, 2)}


def make_long_prompt(tokenizer, base_text: str, target_tokens: int) -> torch.Tensor:
    text = base_text
    while len(tokenizer.encode(text)) < target_tokens:
        text += " " + base_text
    ids = tokenizer.encode(text, return_tensors="pt")
    return ids[:, :target_tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--scale-prompt", action="store_true", help="Uzun prompt ile O(1) kanıtı")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--adapter", default="checkpoints/bulk_adapter/adapter.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_hybrid_infer_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)
    print("=" * 70)
    print(f"BulkTrigger Infer Faz 2 (KV+Bulk) | {device_summary(device)}")
    print("=" * 70)

    base, tokenizer, model_path = load_tinyllama(device, dtype)
    adapter_file = Path(args.adapter)
    adapter_kw = dict(k_short=8, trigger_stride=4, adaptive_trigger=False)

    if adapter_file.exists():
        hybrid, _, _ = create_hybrid(
            device, dtype, adapter_file, base_model=base, **adapter_kw
        )
    else:
        hybrid = TinyLlamaWithBulk(base, freeze_base=True, **adapter_kw).to(device)

    kv_base = BaseKVGenerator(base)
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)

    print(f"\nPrompt: {args.prompt!r} ({input_ids.size(1)} token)")

    results = {
        "device": str(device),
        "model": model_path,
        "adapter": str(adapter_file) if adapter_file.exists() else None,
        "prompt_tokens": input_ids.size(1),
        "new_tokens": args.new_tokens,
    }

    if args.benchmark or args.scale_prompt:
        prompt_lens = [input_ids.size(1)]
        if args.scale_prompt:
            prompt_lens = [32, 128, 512, 1024]

        print(f"\n{'Prompt':>8} {'Recomp/tok':>11} {'KV dec':>9} {'Bulk dec':>9} {'KV pre':>9} {'Bulk pre':>9} {'Fast pre':>9}")
        scale_rows = []
        decode_base, decode_bulk = [], []
        for plen in prompt_lens:
            ids = (
                make_long_prompt(tokenizer, args.prompt, plen).to(device)
                if plen != input_ids.size(1)
                else input_ids
            )
            ms_rec = bench_recompute(base, ids, args.new_tokens)
            b_kv = bench_base_kv(kv_base, ids, args.new_tokens)
            b_bulk = bench_bulk_kv(hybrid, ids, args.new_tokens, fast=True)
            b_slow = bench_bulk_kv(hybrid, ids, min(args.new_tokens, 8), fast=False) if plen <= 512 else None
            decode_base.append(b_kv["decode_ms_per_tok"])
            decode_bulk.append(b_bulk["decode_ms_per_tok"])
            fast_pre = b_bulk["prefill_ms"]
            slow_pre = b_slow["prefill_ms"] if b_slow else None
            speedup = f"{slow_pre / max(fast_pre, 0.01):.1f}x" if slow_pre else "—"
            print(
                f"{plen:>8} {ms_rec:>9.1f}ms {b_kv['decode_ms_per_tok']:>7.1f}ms "
                f"{b_bulk['decode_ms_per_tok']:>7.1f}ms {b_kv['prefill_ms']:>7.0f}ms "
                f"{b_bulk['prefill_ms']:>7.0f}ms {speedup:>9}"
            )
            scale_rows.append({
                "prompt_len": plen,
                "ms_recompute_per_tok": round(ms_rec, 2),
                **{f"base_{k}": v for k, v in b_kv.items()},
                **{f"bulk_{k}": v for k, v in b_bulk.items()},
                "bulk_slow_prefill_ms": slow_pre,
                "prefill_speedup": round(slow_pre / max(fast_pre, 0.01), 2) if slow_pre else None,
            })

        H = base.config.hidden_size
        n_layers = base.config.num_hidden_layers
        results["benchmark"] = scale_rows
        results["memory"] = {
            "bulk_state_kb": round(estimate_bulk_state_bytes(1, H) / 1024, 2),
            "kv_at_max_prompt_mb": round(
                estimate_kv_cache_bytes(max(prompt_lens), n_layers, H) / 1e6, 3
            ),
        }
        rec_short = scale_rows[0]["ms_recompute_per_tok"]
        rec_long = scale_rows[-1]["ms_recompute_per_tok"]
        dec_base_s, dec_base_l = decode_base[0], decode_base[-1]
        dec_bulk_s, dec_bulk_l = decode_bulk[0], decode_bulk[-1]

        def _scale_ratio(d_short, d_long):
            return round((d_long / max(d_short, 1e-6)) / max(rec_long / max(rec_short, 1e-6), 1e-6), 3)

        results["scaling_vs_recompute"] = {
            "base": _scale_ratio(dec_base_s, dec_base_l),
            "bulk": _scale_ratio(dec_bulk_s, dec_bulk_l),
        }
        results["decode_flat_2x"] = {
            "base": dec_base_l < dec_base_s * 2.0,
            "bulk": dec_bulk_l < dec_bulk_s * 2.0,
        }
        results["bulk_beats_base_at_max"] = dec_bulk_l < dec_base_l

        print(f"\nDecode ölçeklenmesi / recompute:")
        print(f"  Base:  {results['scaling_vs_recompute']['base']}×  "
              f"(düz 2×: {'✓' if results['decode_flat_2x']['base'] else '✗'})")
        print(f"  Bulk:  {results['scaling_vs_recompute']['bulk']}×  "
              f"(düz 2×: {'✓' if results['decode_flat_2x']['bulk'] else '✗'})")
        print(f"  Bulk < Base @ max prompt: {'✓' if results['bulk_beats_base_at_max'] else '✗'}")
        print(f"BulkState: {results['memory']['bulk_state_kb']} KB (sabit)")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
