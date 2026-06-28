#!/usr/bin/env python3
"""
Faz 3 — KV'siz kuyruk benchmark (bellek güvenli, sıralı yükleme)
================================================================
Modeller aynı anda RAM'de tutulmaz.

Kullanım:
  python3 bulk_kvfree_benchmark.py --device mps --prompt-lens 512,2048
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import BaseKVGenerator, TinyLlamaWithBulk, create_hybrid, load_tinyllama
from bulk_layer_swap import (
    TinyLlamaKVFreeTail,
    estimate_bulk_tail_bytes,
    estimate_early_kv_bytes,
    load_bulk_swap_checkpoint,
)
from bulk_memory_utils import estimate_bulk_state_bytes, estimate_kv_cache_bytes


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _unload(*objs) -> None:
    del objs
    gc.collect()


def make_prompt(tokenizer, text: str, target: int) -> torch.Tensor:
    while len(tokenizer.encode(text)) < target:
        text += " " + text
    return torch.tensor([tokenizer.encode(text)[:target]], dtype=torch.long)


@torch.no_grad()
def bench_recompute(model, ids: torch.Tensor, new_tokens: int) -> float:
    cur = ids.clone()
    t0 = time.perf_counter()
    for _ in range(new_tokens):
        out = model(input_ids=cur)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    return (time.perf_counter() - t0) / new_tokens * 1000


@torch.no_grad()
def bench_base_kv(base, ids: torch.Tensor, new_tokens: int) -> dict:
    gen = BaseKVGenerator(base)
    t0 = time.perf_counter()
    lg = gen.prefill(ids)
    prefill_ms = (time.perf_counter() - t0) * 1000
    tk = lg.argmax(-1, keepdim=True)
    t1 = time.perf_counter()
    for _ in range(new_tokens - 1):
        lg = gen.decode_step(tk.squeeze(1))
        tk = lg.argmax(-1, keepdim=True)
    decode_ms = (time.perf_counter() - t1) / max(new_tokens - 1, 1) * 1000
    return {"prefill_ms": round(prefill_ms, 2), "decode_ms_per_tok": round(decode_ms, 2)}


@torch.no_grad()
def bench_hybrid(hybrid: TinyLlamaWithBulk, ids: torch.Tensor, new_tokens: int) -> dict:
    t0 = time.perf_counter()
    lg = hybrid.prefill(ids, fast=True)
    prefill_ms = (time.perf_counter() - t0) * 1000
    tk = lg.argmax(-1, keepdim=True)
    t1 = time.perf_counter()
    for _ in range(new_tokens - 1):
        lg = hybrid.decode_step(tk.squeeze(1))
        tk = lg.argmax(-1, keepdim=True)
    decode_ms = (time.perf_counter() - t1) / max(new_tokens - 1, 1) * 1000
    return {"prefill_ms": round(prefill_ms, 2), "decode_ms_per_tok": round(decode_ms, 2)}


@torch.no_grad()
def bench_kvfree(model: TinyLlamaKVFreeTail, ids: torch.Tensor, new_tokens: int) -> dict:
    t0 = time.perf_counter()
    lg = model.prefill(ids)
    prefill_ms = (time.perf_counter() - t0) * 1000
    tk = lg.argmax(-1, keepdim=True)
    t1 = time.perf_counter()
    for _ in range(new_tokens - 1):
        lg = model.decode_step(tk.squeeze(1))
        tk = lg.argmax(-1, keepdim=True)
    decode_ms = (time.perf_counter() - t1) / max(new_tokens - 1, 1) * 1000
    return {"prefill_ms": round(prefill_ms, 2), "decode_ms_per_tok": round(decode_ms, 2)}


def scaling_ratio(decode_short: float, decode_long: float, rec_short: float, rec_long: float) -> float:
    return round(
        (decode_long / max(decode_short, 1e-6)) / max(rec_long / max(rec_short, 1e-6), 1e-6),
        3,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lens", default="512,2048")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--swap-layers", type=int, default=4)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--d-bulk", type=int, default=256)
    parser.add_argument("--skip-hybrid", action="store_true", help="Hybrid karşılaştırmayı atla (RAM)")
    parser.add_argument("--adapter", default="checkpoints/bulk_adapter/adapter.pt")
    parser.add_argument("--swap-checkpoint", default="checkpoints/bulk_swap/bulk_v2.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_kvfree_benchmark_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)
    lens = [int(x.strip()) for x in args.prompt_lens.split(",") if x.strip()]

    print("=" * 72)
    print(f"Faz 3 KVFree Benchmark (sıralı) | {device_summary(device)}")
    print(f"swap={args.swap_layers} window={args.sliding_window} d_bulk={args.d_bulk}")
    print("=" * 72)

    _, tokenizer, model_path = load_tinyllama(device, dtype)
    text = "The history of science spans centuries of discovery and invention. "
    prompts = {plen: make_prompt(tokenizer, text, plen).to(device) for plen in lens}

    rows = []
    dec_base, dec_hybrid, dec_kvfree, rec_ms = [], [], [], []

    # --- Base ---
    print("\n[1/3] Base KV...")
    base, _, _ = load_tinyllama(device, dtype)
    H = base.config.hidden_size
    n_layers = base.config.num_hidden_layers
    n_early = n_layers - args.swap_layers
    print(f"\n{'Prompt':>8} {'Rec/tok':>8} {'Base dec':>9} {'Hybrid dec':>11} {'KVFree dec':>11}")
    for plen in lens:
        ids = prompts[plen]
        _free(device)
        ms_rec = bench_recompute(base, ids, args.new_tokens)
        b_base = bench_base_kv(base, ids, args.new_tokens)
        dec_base.append(b_base["decode_ms_per_tok"])
        rec_ms.append(ms_rec)
        rows.append({"prompt_len": plen, "recompute_ms_per_tok": round(ms_rec, 2), "base": b_base})
        print(f"{plen:>8} {ms_rec:>7.1f}ms {b_base['decode_ms_per_tok']:>8.1f}ms", end="")
        if args.skip_hybrid:
            dec_hybrid.append(0.0)
            print(f" {'—':>11}", end="")
        dec_kvfree.append(0.0)
        print(f" {'…':>11}")
    del base
    _free(device)

    # --- Hybrid ---
    if not args.skip_hybrid:
        print("\n[2/3] Hybrid adapter...")
        base_h, _, _ = load_tinyllama(device, dtype)
        adapter_path = Path(args.adapter)
        if adapter_path.exists():
            hybrid, _, _ = create_hybrid(device, dtype, adapter_path, base_model=base_h)
        else:
            hybrid = TinyLlamaWithBulk(base_h, freeze_base=True).to(device)
        for i, plen in enumerate(lens):
            b_hybrid = bench_hybrid(hybrid, prompts[plen], args.new_tokens)
            dec_hybrid[i] = b_hybrid["decode_ms_per_tok"]
            rows[i]["hybrid"] = b_hybrid
        del hybrid, base_h
        _free(device)

    # --- KVFree ---
    print("\n[3/3] KVFree compact tail...")
    base_k, _, _ = load_tinyllama(device, dtype)
    kvfree = TinyLlamaKVFreeTail(
        base_k, n_swap_layers=args.swap_layers,
        sliding_window=args.sliding_window, compact=True, d_bulk=args.d_bulk,
    ).to(device)
    swap_ckpt = Path(args.swap_checkpoint)
    if swap_ckpt.exists():
        try:
            load_bulk_swap_checkpoint(kvfree, str(swap_ckpt), device)
            print(f"  checkpoint: {swap_ckpt}")
        except RuntimeError as e:
            print(f"  ⚠ checkpoint uyumsuz ({e}), rastgele ağırlık")
    for i, plen in enumerate(lens):
        b_kvfree = bench_kvfree(kvfree, prompts[plen], args.new_tokens)
        dec_kvfree[i] = b_kvfree["decode_ms_per_tok"]
        rows[i]["kvfree"] = b_kvfree
    del kvfree, base_k
    _free(device)

    # Tablo tamamla
    print(f"\n{'Prompt':>8} {'Rec/tok':>8} {'Base dec':>9} {'Hybrid dec':>11} {'KVFree dec':>11}")
    for i, plen in enumerate(lens):
        h = f"{dec_hybrid[i]:>10.1f}ms" if not args.skip_hybrid else f"{'—':>11}"
        print(f"{plen:>8} {rec_ms[i]:>7.1f}ms {dec_base[i]:>8.1f}ms {h} {dec_kvfree[i]:>10.1f}ms")

    eff_len = args.sliding_window if args.sliding_window > 0 else max(lens)
    d_tail = args.d_bulk
    mem = {
        "full_kv_mb_at_max": round(estimate_kv_cache_bytes(max(lens), n_layers, H) / 1e6, 2),
        "early_kv_mb_bounded": round(
            estimate_early_kv_bytes(eff_len, n_early, H, dtype_bytes=2) / 1e6, 2,
        ),
        "bulk_tail_kb": round(
            estimate_bulk_tail_bytes(args.swap_layers, H, d_bulk=d_tail) / 1024, 2,
        ),
        "hybrid_bulk_kb": round(estimate_bulk_state_bytes(1, H) / 1024, 2),
        "kvfree_total_active_mb": round(
            (estimate_early_kv_bytes(eff_len, n_early, H, dtype_bytes=2)
             + estimate_bulk_tail_bytes(args.swap_layers, H, d_bulk=d_tail)) / 1e6, 3,
        ),
    }

    scale = {
        "base": scaling_ratio(dec_base[0], dec_base[-1], rec_ms[0], rec_ms[-1]),
        "kvfree": scaling_ratio(dec_kvfree[0], dec_kvfree[-1], rec_ms[0], rec_ms[-1]),
    }
    if not args.skip_hybrid:
        scale["hybrid"] = scaling_ratio(dec_hybrid[0], dec_hybrid[-1], rec_ms[0], rec_ms[-1])

    kvfree_flat = dec_kvfree[-1] < dec_kvfree[0] * 2.5
    kvfree_vs_base = dec_kvfree[-1] < dec_base[-1]

    print(f"\n--- Bellek (teorik @ max {max(lens)}) ---")
    print(f"  Full KV:              {mem['full_kv_mb_at_max']} MB")
    print(f"  KVFree erken (≤{eff_len}): {mem['early_kv_mb_bounded']} MB")
    print(f"  KVFree bulk kuyruk:   {mem['bulk_tail_kb']} KB")
    print(f"  KVFree toplam:        {mem['kvfree_total_active_mb']} MB")
    print(f"\nKVFree decode düz: {'✓' if kvfree_flat else '✗'}  |  KVFree < Base: {'✓' if kvfree_vs_base else '✗'}")

    results = {
        "device": str(device),
        "model": model_path,
        "compact": True,
        "d_bulk": args.d_bulk,
        "n_swap": args.swap_layers,
        "sliding_window": args.sliding_window,
        "rows": rows,
        "memory": mem,
        "scaling": scale,
        "kvfree_decode_flat": kvfree_flat,
        "kvfree_beats_base_at_max": kvfree_vs_base,
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
