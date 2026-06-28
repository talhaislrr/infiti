#!/usr/bin/env python3
"""
KVFree'nin Base'in yapamadığı şeyi kanıtla
============================================
Base: KV-cache uzunlukla O(n) büyür → bellek + decode yavaşlar.
KVFree (sliding_window): erken KV ≤W token, bulk state sabit → düz maliyet.

Demo bölümleri:
  1) Decode ölçekleme: 512 vs 2048 prompt — Base yavaşlar, KVFree düz kalır
  2) Çok turlu oturum: her tur +chunk token — Base KV büyür, KVFree sınırlı
  3) Edge bütçe simülasyonu: sabit MB bütçede Base tur N'de FAIL, KVFree PASS

Kullanım (Colab L4):
  %cd /content/infiti
  !python3 bulk_kvfree_advantage.py --device cuda \\
      --swap-checkpoint /content/drive/MyDrive/infiti/bulk_v2.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import BaseKVGenerator, load_tinyllama
from bulk_layer_swap import (
    TinyLlamaKVFreeTail,
    estimate_bulk_tail_bytes,
    estimate_early_kv_bytes,
    load_bulk_swap_checkpoint,
)
from bulk_memory_utils import estimate_kv_cache_bytes


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


def _alloc_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1e6
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


def kv_seq_len(past: Any) -> int:
    if past is None:
        return 0
    if hasattr(past, "key_cache") and past.key_cache:
        return int(past.key_cache[0].size(-2))
    if isinstance(past, (list, tuple)) and past and past[0] is not None:
        return int(past[0][0].size(-2))
    return 0


def make_chunk(tokenizer, target_tokens: int, seed: str) -> torch.Tensor:
    text = seed + " "
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += seed + " "
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return torch.tensor([ids], dtype=torch.long)


@torch.inference_mode()
def measure_decode_ms(gen, n_steps: int = 16) -> float:
    t0 = time.perf_counter()
    tok = torch.tensor([1], dtype=torch.long)
    for _ in range(n_steps):
        lg = gen.decode_step(tok)
        tok = lg.argmax(-1)
        if tok.dim() == 0:
            tok = tok.unsqueeze(0)
    return (time.perf_counter() - t0) / n_steps * 1000


@torch.inference_mode()
def run_scaling_proof(
    base,
    kvfree: TinyLlamaKVFreeTail,
    tokenizer,
    device: torch.device,
    short_len: int,
    long_len: int,
    decode_steps: int,
) -> dict:
    text = "Scientific discovery advances through careful observation and experiment. "
    short_ids = make_chunk(tokenizer, short_len, text)[:, :short_len].to(device)
    long_ids = make_chunk(tokenizer, long_len, text)[:, :long_len].to(device)

    base_gen = BaseKVGenerator(base)
    _free(device)
    _reset_peak(device)
    base_gen.prefill(short_ids)
    base_short = measure_decode_ms(base_gen, decode_steps)

    _free(device)
    _reset_peak(device)
    base_gen.prefill(long_ids)
    base_long = measure_decode_ms(base_gen, decode_steps)
    base_ratio = base_long / max(base_short, 1e-6)

    kvfree.prefill(short_ids)
    kv_short = measure_decode_ms(kvfree, decode_steps)

    _free(device)
    _reset_peak(device)
    kvfree.prefill(long_ids)
    kv_long = measure_decode_ms(kvfree, decode_steps)
    kv_ratio = kv_long / max(kv_short, 1e-6)

    return {
        "short_len": short_len,
        "long_len": long_len,
        "base_decode_ms": {"short": round(base_short, 2), "long": round(base_long, 2)},
        "kvfree_decode_ms": {"short": round(kv_short, 2), "long": round(kv_long, 2)},
        "base_slowdown_ratio": round(base_ratio, 3),
        "kvfree_slowdown_ratio": round(kv_ratio, 3),
        "kvfree_flat_while_base_slows": kv_ratio < 1.25 and base_ratio > 1.15,
    }


def _feed_and_reply(gen, chunk: torch.Tensor, device: torch.device, reply_tokens: int, first: bool) -> None:
    if first:
        gen.prefill(chunk)
    else:
        for tid in chunk[0].tolist():
            gen.decode_step(torch.tensor([tid], device=device))
    tok = torch.tensor([1], device=device)
    for _ in range(reply_tokens):
        lg = gen.decode_step(tok)
        tok = lg.argmax(-1)
        if tok.dim() == 0:
            tok = tok.unsqueeze(0)


@torch.inference_mode()
def run_multiturn_proof(
    base,
    kvfree: TinyLlamaKVFreeTail,
    tokenizer,
    device: torch.device,
    n_turns: int,
    chunk_tokens: int,
    reply_tokens: int,
    n_early: int,
    n_layers: int,
    H: int,
    sliding_window: int,
    d_bulk: int,
    n_swap: int,
) -> dict:
    seed = "Multi turn session memory test chunk number "
    turns: list[dict] = []

    base_gen = BaseKVGenerator(base)

    for turn in range(1, n_turns + 1):
        chunk = make_chunk(tokenizer, chunk_tokens, seed + str(turn))
        chunk = chunk[:, :chunk_tokens].to(device)
        first = turn == 1

        _reset_peak(device)
        _feed_and_reply(base_gen, chunk, device, reply_tokens, first)
        base_kv_len = kv_seq_len(base_gen._past_key_values)
        base_dec = measure_decode_ms(base_gen, 8)
        base_theory_kv_mb = estimate_kv_cache_bytes(base_kv_len, n_layers, H, dtype_bytes=2) / 1e6
        base_peak = _peak_mb(device)

        _reset_peak(device)
        _feed_and_reply(kvfree, chunk, device, reply_tokens, first)
        cache = kvfree._cache
        kv_early_len = kv_seq_len(cache.early_past)
        kv_dec = measure_decode_ms(kvfree, 8)
        kv_theory_mb = (
            estimate_early_kv_bytes(kv_early_len, n_early, H, dtype_bytes=2)
            + estimate_bulk_tail_bytes(n_swap, H, d_bulk=d_bulk)
        ) / 1e6
        kv_peak = _peak_mb(device)

        turns.append({
            "turn": turn,
            "session_tokens_approx": turn * (chunk_tokens + reply_tokens),
            "base": {
                "kv_seq_len": base_kv_len,
                "theory_kv_mb": round(base_theory_kv_mb, 2),
                "decode_ms": round(base_dec, 2),
                "peak_alloc_mb": round(base_peak, 2),
            },
            "kvfree": {
                "early_kv_len": kv_early_len,
                "theory_active_mb": round(kv_theory_mb, 3),
                "decode_ms": round(kv_dec, 2),
                "peak_alloc_mb": round(kv_peak, 2),
                "early_capped": kv_early_len <= sliding_window,
            },
        })

    first = turns[0]
    last = turns[-1]
    base_kv_growth = last["base"]["kv_seq_len"] / max(first["base"]["kv_seq_len"], 1)
    kvfree_kv_flat = last["kvfree"]["early_kv_len"] <= sliding_window
    base_decode_growth = last["base"]["decode_ms"] / max(first["base"]["decode_ms"], 1e-6)
    kvfree_decode_flat = last["kvfree"]["decode_ms"] < first["kvfree"]["decode_ms"] * 1.35

    return {
        "n_turns": n_turns,
        "chunk_tokens": chunk_tokens,
        "reply_tokens": reply_tokens,
        "sliding_window": sliding_window,
        "turns": turns,
        "base_kv_grows": base_kv_growth > 2.0,
        "kvfree_kv_bounded": kvfree_kv_flat,
        "base_decode_slows": base_decode_growth > 1.15,
        "kvfree_decode_flat": kvfree_decode_flat,
    }


def run_budget_proof(
    turns: list[dict],
    budget_mb: float,
    n_layers: int,
    H: int,
    n_early: int,
    sliding_window: int,
    n_swap: int,
    d_bulk: int,
) -> dict:
    base_fail_turn = None
    kvfree_fail_turn = None
    for row in turns:
        t = row["turn"]
        if base_fail_turn is None and row["base"]["theory_kv_mb"] > budget_mb:
            base_fail_turn = t
        kv_mb = (
            estimate_early_kv_bytes(sliding_window, n_early, H, dtype_bytes=2)
            + estimate_bulk_tail_bytes(n_swap, H, d_bulk=d_bulk)
        ) / 1e6
        if kvfree_fail_turn is None and kv_mb > budget_mb:
            kvfree_fail_turn = t

    return {
        "budget_mb": budget_mb,
        "base_fail_at_turn": base_fail_turn,
        "kvfree_fail_at_turn": kvfree_fail_turn,
        "base_cannot_sustain": base_fail_turn is not None and kvfree_fail_turn is None,
        "note": (
            f"{budget_mb} MB KV bütçesinde Base tur {base_fail_turn}'de taşar, "
            f"KVFree ≤{sliding_window} erken KV ile geçer."
            if base_fail_turn and not kvfree_fail_turn
            else f"Bütçe {budget_mb} MB — her iki mod da geçer (bütçeyi artırın)."
        ),
    }


def print_verdict(scaling: dict, multiturn: dict, budget: dict) -> None:
    print("\n" + "=" * 72)
    print("SONUÇ — Base'in yapamadığı, KVFree'nin yaptığı")
    print("=" * 72)

    if scaling["kvfree_flat_while_base_slows"]:
        print(
            f"✓ Decode ölçekleme: Base {scaling['base_slowdown_ratio']:.2f}× yavaşlar "
            f"({scaling['short_len']}→{scaling['long_len']} tok), "
            f"KVFree {scaling['kvfree_slowdown_ratio']:.2f}× (düz)."
        )
    else:
        print(
            f"⚠ Decode: Base ratio={scaling['base_slowdown_ratio']:.2f}, "
            f"KVFree ratio={scaling['kvfree_slowdown_ratio']:.2f}"
        )

    if multiturn["base_kv_grows"] and multiturn["kvfree_kv_bounded"]:
        last = multiturn["turns"][-1]
        print(
            f"✓ Çok turlu oturum: Base KV {last['base']['kv_seq_len']} tok "
            f"({last['base']['theory_kv_mb']} MB teori), "
            f"KVFree erken KV {last['kvfree']['early_kv_len']} tok "
            f"(≤{multiturn['sliding_window']}, {last['kvfree']['theory_active_mb']} MB)."
        )

    if budget["base_cannot_sustain"]:
        print(f"✓ Edge bütçe ({budget['budget_mb']} MB): {budget['note']}")
    elif budget["base_fail_at_turn"]:
        print(f"⚠ Bütçe: {budget['note']}")
    else:
        print(
            f"ℹ Bütçe sim ({budget['budget_mb']} MB): Base tur {budget['base_fail_at_turn']} — "
            "daha düşük bütçe veya daha fazla tur deneyin."
        )

    proved = (
        scaling["kvfree_flat_while_base_slows"]
        and multiturn["base_kv_grows"]
        and multiturn["kvfree_kv_bounded"]
    )
    print(f"\nKanıt durumu: {'GEÇERLİ ✓' if proved else 'KISMİ — Colab CUDA ile tekrarlayın'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KVFree avantaj kanıtı (Base vs KVFree)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--swap-layers", type=int, default=4)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--d-bulk", type=int, default=256)
    parser.add_argument("--swap-checkpoint", default="checkpoints/bulk_swap/bulk_v2.pt")
    parser.add_argument("--short-len", type=int, default=512)
    parser.add_argument("--long-len", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--chunk-tokens", type=int, default=280)
    parser.add_argument("--reply-tokens", type=int, default=40)
    parser.add_argument("--mem-budget-mb", type=float, default=120.0,
                        help="Edge KV bütçe simülasyonu (MB)")
    parser.add_argument("--output", default="bulk_kvfree_advantage_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)
    max_ctx = 2048
    args.long_len = min(args.long_len, max_ctx)
    args.short_len = min(args.short_len, max_ctx)

    print("=" * 72)
    print(f"KVFree Avantaj Kanıtı | {device_summary(device)}")
    print(f"window={args.sliding_window} swap={args.swap_layers} budget={args.mem_budget_mb} MB")
    print("=" * 72)

    base, tokenizer, model_path = load_tinyllama(device, dtype)
    H = base.config.hidden_size
    n_layers = base.config.num_hidden_layers
    n_early = n_layers - args.swap_layers

    kvfree = TinyLlamaKVFreeTail(
        base,
        n_swap_layers=args.swap_layers,
        sliding_window=args.sliding_window,
        compact=True,
        d_bulk=args.d_bulk,
    ).to(device)
    ckpt = Path(args.swap_checkpoint)
    if ckpt.exists():
        load_bulk_swap_checkpoint(kvfree, str(ckpt), device)
        print(f"Checkpoint: {ckpt}")
    else:
        print(f"⚠ Checkpoint yok ({ckpt}) — rastgele bulk ağırlık")

    print("\n[1/3] Decode ölçekleme (512 vs 2048)...")
    scaling = run_scaling_proof(
        base, kvfree, tokenizer, device,
        args.short_len, args.long_len, args.decode_steps,
    )
    print(
        f"  Base:  {scaling['base_decode_ms']['short']}ms → {scaling['base_decode_ms']['long']}ms "
        f"({scaling['base_slowdown_ratio']:.2f}×)"
    )
    print(
        f"  KVFree: {scaling['kvfree_decode_ms']['short']}ms → {scaling['kvfree_decode_ms']['long']}ms "
        f"({scaling['kvfree_slowdown_ratio']:.2f}×)"
    )

    print(f"\n[2/3] Çok turlu oturum ({args.turns} tur × {args.chunk_tokens}+{args.reply_tokens} tok)...")
    multiturn = run_multiturn_proof(
        base, kvfree, tokenizer, device,
        args.turns, args.chunk_tokens, args.reply_tokens,
        n_early, n_layers, H,
        args.sliding_window, args.d_bulk, args.swap_layers,
    )
    print(f"  {'Tur':>4} {'Base KV':>8} {'Base dec':>9} {'KVFree KV':>10} {'KVFree dec':>11}")
    for row in multiturn["turns"]:
        b, k = row["base"], row["kvfree"]
        print(
            f"  {row['turn']:>4} {b['kv_seq_len']:>8} {b['decode_ms']:>8.1f}ms "
            f"{k['early_kv_len']:>10} {k['decode_ms']:>10.1f}ms"
        )

    print(f"\n[3/3] Edge bütçe simülasyonu ({args.mem_budget_mb} MB KV)...")
    budget = run_budget_proof(
        multiturn["turns"], args.mem_budget_mb,
        n_layers, H, n_early, args.sliding_window, args.swap_layers, args.d_bulk,
    )
    print(f"  {budget['note']}")

    print_verdict(scaling, multiturn, budget)

    results = {
        "device": str(device),
        "model": model_path,
        "sliding_window": args.sliding_window,
        "mem_budget_mb": args.mem_budget_mb,
        "scaling": scaling,
        "multiturn": multiturn,
        "budget": budget,
        "proved": (
            scaling["kvfree_flat_while_base_slows"]
            and multiturn["base_kv_grows"]
            and multiturn["kvfree_kv_bounded"]
        ),
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nKaydedildi: {args.output}")

    del base, kvfree
    _free(device)


if __name__ == "__main__":
    main()
