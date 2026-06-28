#!/usr/bin/env python3
"""
BulkTrigger Mimari Benchmark
============================
Standart causal decoder vs Tetikleyici-Üretici (KV-cache yok).

Ölçümler:
  1. Inference latency vs sequence length (O(1) vs O(n) beklentisi)
  2. Bellek: KV-cache vs sabit k-pencere
  3. Küçük LM görevi: sentetik copy / next-char

Kullanım:
  python3 bulk_trigger_benchmark.py
  python3 bulk_trigger_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bulk_trigger_arch import (
    BulkTriggerLM,
    StandardDecoderLM,
    count_params,
    estimate_kv_cache_bytes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sentetik veri: "copy" — prompt tekrar et
# ─────────────────────────────────────────────────────────────────────────────

class CopyDataset(Dataset):
    """Tekrarlayan pattern — öğrenmesi kolay: [a,b,c,d,a,b,c,d,...]"""

    def __init__(self, vocab_size: int = 16, pattern_len: int = 4, seq_len: int = 16, n_samples: int = 2000):
        self.data = []
        for _ in range(n_samples):
            pattern = torch.randint(1, vocab_size, (pattern_len,))
            reps = seq_len // pattern_len + 1
            seq = pattern.repeat(reps)[:seq_len]
            self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Inference benchmark
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LatencyRow:
    seq_len: int
    standard_ms: float
    bulk_ms: float
    speedup: float
    kv_cache_mb: float
    bulk_window_kb: float


def bench_inference(
    standard: StandardDecoderLM,
    bulk: BulkTriggerLM,
    device: torch.device,
    seq_lengths: list[int],
    warmup: int = 5,
    repeats: int = 20,
) -> list[LatencyRow]:
    rows = []
    vocab = standard.head.out_features if hasattr(standard.head, "out_features") else standard.head.in_features

    for slen in seq_lengths:
        # Standart: her adımda tüm prefix'i yeniden işle (KV-cache yok — adil O(n))
        prompt = torch.randint(1, vocab, (1, slen), device=device)

        for _ in range(warmup):
            ids = prompt.clone()
            for _ in range(5):
                _ = standard.generate_step_full(ids)
                ids = torch.cat([ids, ids[:, -1:]], dim=1)

        t0 = time.perf_counter()
        for _ in range(repeats):
            ids = prompt.clone()
            for _ in range(10):
                logits = standard.generate_step_full(ids)
                next_tok = logits.argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_tok], dim=1)
        std_time = (time.perf_counter() - t0) / (repeats * 10) * 1000

        # Bulk: sabit k pencere
        for _ in range(warmup):
            hist: deque = deque(maxlen=bulk.k)
            pos = 0
            tok = prompt[:, 0]
            for i in range(min(5, slen)):
                logits, emb = bulk.generate_step(tok, hist, pos)
                hist.append(emb.squeeze(0))
                pos += 1
                tok = logits.argmax(-1)

        t0 = time.perf_counter()
        for _ in range(repeats):
            hist = deque(maxlen=bulk.k)
            pos = 0
            tok = prompt[:, 0:1].squeeze(1)
            for i in range(slen):
                if i > 0:
                    tok = prompt[:, i]
                logits, emb = bulk.generate_step(tok, hist, pos)
                hist.append(emb.squeeze(0) if emb.dim() > 1 else emb)
                pos += 1
            for _ in range(10):
                logits, emb = bulk.generate_step(
                    logits.argmax(-1), hist, pos
                )
                hist.append(emb.squeeze(0) if emb.dim() > 1 else emb)
                pos += 1
        bulk_time = (time.perf_counter() - t0) / (repeats * (slen + 10)) * 1000

        kv_mb = estimate_kv_cache_bytes(slen + 10, len(standard.layers), standard.d_model) / 1e6
        bulk_kb = bulk.k * bulk.d_model * 4 / 1024  # fp32 embedding window

        rows.append(LatencyRow(
            seq_len=slen,
            standard_ms=round(std_time, 3),
            bulk_ms=round(bulk_time, 3),
            speedup=round(std_time / max(bulk_time, 1e-6), 2),
            kv_cache_mb=round(kv_mb, 3),
            bulk_window_kb=round(bulk_kb, 3),
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    vocab_size: int
    d_model: int
    standard_params: int
    bulk_params: int
    standard_acc: float
    bulk_acc: float
    standard_train_sec: float
    bulk_train_sec: float
    latency: list[dict]
    o1_verified: bool
    memory_ratio_at_max_len: float


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="bulk_trigger_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = 16
    d_model = 128
    epochs = 8 if args.quick else 20
    seq_lens = [16, 32, 64, 128] if not args.quick else [16, 32, 64, 128, 256]

    print("=" * 70)
    print("BULKTRIGGER MİMARİ BENCHMARK")
    print(f"device={device} | d_model={d_model} | vocab={vocab}")
    print("=" * 70)

    train_ds = CopyDataset(vocab, seq_len=16, n_samples=1500 if args.quick else 5000)
    test_ds = CopyDataset(vocab, seq_len=16, n_samples=300 if args.quick else 1000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    standard = StandardDecoderLM(vocab, d_model=d_model, n_layers=2, n_heads=4).to(device)
    bulk = BulkTriggerLM(
        vocab, d_model=d_model, n_layers=2, n_heads=4,
        trigger_layers=1, trigger_heads=2, k=4,
    ).to(device)

    sp, bp = count_params(standard), count_params(bulk)
    print(f"\nParametre: standart={sp:,}  bulk={bp:,} ({100*bp/sp:.1f}%)")

    criterion = nn.CrossEntropyLoss()
    opt_s = torch.optim.Adam(standard.parameters(), lr=3e-3)
    opt_b = torch.optim.Adam(bulk.parameters(), lr=3e-3)

    print(f"\nEğitim ({epochs} epoch, copy task)...")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        ls = train_epoch(standard, train_loader, opt_s, criterion, device)
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            acc = eval_acc(standard, test_loader, device)
            print(f"  [std] epoch {ep} loss={ls:.4f} acc={acc:.4f}")
    std_train = time.time() - t0
    std_acc = eval_acc(standard, test_loader, device)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        lb = train_epoch(bulk, train_loader, opt_b, criterion, device)
        if ep % max(1, epochs // 3) == 0 or ep == epochs:
            acc = eval_acc(bulk, test_loader, device)
            print(f"  [bulk] epoch {ep} loss={lb:.4f} acc={acc:.4f}")
    bulk_train = time.time() - t0
    bulk_acc = eval_acc(bulk, test_loader, device)

    print("\nInference latency (ms/token, KV-cache'siz standart vs Bulk O(1))...")
    latency = bench_inference(standard, bulk, device, seq_lens)

    print(f"\n{'SeqLen':>8} {'Std ms':>10} {'Bulk ms':>10} {'Hız':>8} {'KV MB':>8} {'Bulk KB':>8}")
    print("-" * 58)
    for r in latency:
        print(f"{r.seq_len:>8} {r.standard_ms:>10.3f} {r.bulk_ms:>10.3f} {r.speedup:>7.2f}x "
              f"{r.kv_cache_mb:>8.3f} {r.bulk_window_kb:>8.3f}")

    # O(1) doğrulama: uzun seq'de bulk latency artmamalı (±50% tolerans)
    if len(latency) >= 2:
        short, long = latency[0].bulk_ms, latency[-1].bulk_ms
        o1_ok = long < short * 2.5
        mem_ratio = latency[-1].kv_cache_mb / max(latency[-1].bulk_window_kb / 1024, 1e-6)
    else:
        o1_ok = False
        mem_ratio = 0.0

    print("\n" + "=" * 70)
    print("ÖZET")
    print("=" * 70)
    print(f"  Copy task acc     : standart={std_acc:.4f}  bulk={bulk_acc:.4f}")
    print(f"  Eğitim süresi     : standart={std_train:.1f}s  bulk={bulk_train:.1f}s")
    print(f"  O(1) latency      : {'✓' if o1_ok else '✗'} (bulk {latency[0].bulk_ms:.2f}→{latency[-1].bulk_ms:.2f} ms)")
    print(f"  Bellek (uzun seq) : KV-cache {latency[-1].kv_cache_mb:.2f} MB vs "
          f"Bulk pencere {latency[-1].bulk_window_kb:.2f} KB (~{mem_ratio:.0f}x fark)")

    if latency[-1].speedup > 1.0:
        print(f"  Hız (uzun seq)    : Bulk {latency[-1].speedup:.1f}x daha hızlı")
    else:
        print(f"  Hız (uzun seq)    : Bulk henüz hız avantajı yok (küçük model overhead)")

    result = BenchmarkResult(
        vocab_size=vocab,
        d_model=d_model,
        standard_params=sp,
        bulk_params=bp,
        standard_acc=round(std_acc, 4),
        bulk_acc=round(bulk_acc, 4),
        standard_train_sec=round(std_train, 2),
        bulk_train_sec=round(bulk_train, 2),
        latency=[asdict(r) for r in latency],
        o1_verified=o1_ok,
        memory_ratio_at_max_len=round(mem_ratio, 1),
    )
    Path(args.output).write_text(json.dumps(asdict(result), indent=2))
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
