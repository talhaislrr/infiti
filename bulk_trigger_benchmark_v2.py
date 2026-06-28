#!/usr/bin/env python3
"""
BulkTrigger v2 Benchmark — BulkState vs v1 (kausal) + uzun bağlam stres testi
==============================================================================
Kullanım:
  python3 bulk_trigger_benchmark_v2.py --quick
  python3 bulk_trigger_benchmark_v2.py --cuda
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bulk_trigger_arch import BulkTriggerLM, StandardDecoderLM, count_params, estimate_kv_cache_bytes
from bulk_trigger_v2 import BulkTriggerLMv2


# ── Veri setleri ─────────────────────────────────────────────────────────────

class PatternDataset(Dataset):
    def __init__(self, vocab: int, pattern_len: int, seq_len: int, n: int):
        self.data = []
        for _ in range(n):
            p = torch.randint(1, vocab, (pattern_len,))
            s = p.repeat(seq_len // pattern_len + 2)[:seq_len]
            self.data.append(s)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s = self.data[i]
        return s[:-1], s[1:]


class LongRangeRecallDataset(Dataset):
    """[marker, filler×gap, marker] — kritik pozisyon: marker'ı hatırla."""

    def __init__(self, vocab: int, gap: int, n: int):
        self.gap = gap
        self.data = []
        for _ in range(n):
            marker = torch.randint(1, vocab, (1,))
            filler = torch.randint(1, vocab, (gap,))
            self.data.append(torch.cat([marker, filler, marker]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s = self.data[i]
        return s[:-1], s[1:]

    @property
    def recall_pos(self) -> int:
        return self.gap


def train_epoch(model, loader, opt, crit, device, amp=False):
    model.train()
    total, n = 0.0, 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    ok, tot = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ok += (model(x).argmax(-1) == y).sum().item()
        tot += y.numel()
    return ok / max(tot, 1)


@torch.no_grad()
def eval_recall_at_pos(model, dataset: LongRangeRecallDataset, device) -> float:
    model.eval()
    pos = dataset.recall_pos
    ok, tot = 0, 0
    for i in range(len(dataset)):
        x, y = dataset[i]
        x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
        pred = model(x).argmax(-1)
        ok += (pred[0, pos] == y[0, pos]).item()
        tot += 1
    return ok / max(tot, 1)


def bench_bulk_v2(model: BulkTriggerLMv2, device, seq_lens, repeats=10):
    rows = []
    vocab = model.head.out_features
    for slen in seq_lens:
        t0 = time.perf_counter()
        for _ in range(repeats):
            hist: deque = deque(maxlen=model.k_short)
            states = model.init_bulk_states(1, device, torch.float32)
            pos = 0
            tok = torch.randint(1, vocab, (1,), device=device)
            for _ in range(slen + 10):
                logits, emb, states = model.generate_step(tok, hist, states, pos)
                hist.append(emb)
                pos += 1
                tok = logits.argmax(-1).clamp(0, vocab - 1)
        ms = (time.perf_counter() - t0) / (repeats * (slen + 10)) * 1000
        mem_kb = model.k_short * model.d_model * 4 / 1024
        rows.append({"seq_len": slen, "ms_per_tok": round(ms, 3), "bulk_mem_kb": mem_kb})
    return rows


def bench_bulk_v1(model: BulkTriggerLM, device, seq_lens, repeats=10):
    rows = []
    vocab = model.head.out_features
    for slen in seq_lens:
        t0 = time.perf_counter()
        for _ in range(repeats):
            hist: deque = deque()
            pos = 0
            tok = torch.randint(1, vocab, (1,), device=device)
            for _ in range(slen + 10):
                logits, emb = model.generate_step(tok, hist, pos)
                hist.append(emb)
                pos += 1
                tok = logits.argmax(-1).clamp(0, vocab - 1)
        ms = (time.perf_counter() - t0) / (repeats * (slen + 10)) * 1000
        rows.append({"seq_len": slen, "ms_per_tok": round(ms, 3)})
    return rows


def bench_standard_nocache(model, device, seq_lens, repeats=10):
    rows = []
    vocab = model.head.out_features
    max_len = model.max_len
    for slen in seq_lens:
        t0 = time.perf_counter()
        for _ in range(repeats):
            ids = torch.randint(1, vocab, (1, min(slen, max_len - 1)), device=device)
            n_steps = 10
            for _ in range(n_steps):
                logits = model.generate_step_full(ids)
                next_tok = logits.argmax(-1, keepdim=True).clamp(0, vocab - 1)
                ids = torch.cat([ids, next_tok], dim=1)
                if ids.size(1) >= max_len:
                    ids = ids[:, -(max_len - 1) :]
        ms = (time.perf_counter() - t0) / (repeats * n_steps) * 1000
        kv = estimate_kv_cache_bytes(min(slen, max_len), len(model.layers), model.d_model) / 1e6
        rows.append({"seq_len": slen, "ms_per_tok": round(ms, 3), "kv_mb": round(kv, 4)})
    return rows


def train_model(model, train_loader, test_loader, device, epochs, lr, amp=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        train_epoch(model, train_loader, opt, crit, device, amp=amp)
    return eval_acc(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", default="bulk_trigger_v2_results.json")
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    vocab, d_model = 32, 128
    gap = 48 if args.quick else 96
    epochs_recall = 20 if args.quick else 40
    epochs_pattern = 20 if args.quick else 40

    print("=" * 70)
    print(f"BulkTrigger v2 BENCHMARK (kausal v1) | device={device}")
    print("=" * 70)

    # ── Uzun menzil hatırlama ───────────────────────────────────────────────
    print(f"\n[1] Long-range recall (gap={gap}, eval @ pos={gap})")
    lr_train = LongRangeRecallDataset(vocab, gap, 600 if args.quick else 2000)
    lr_test = LongRangeRecallDataset(vocab, gap, 200 if args.quick else 500)

    v1 = BulkTriggerLM(vocab, d_model=d_model, k=4, n_layers=2).to(device)
    v2 = BulkTriggerLMv2(
        vocab, d_model=d_model, k_short=8,
        medium_interval=16, long_interval=64 if args.quick else 128,
        trigger_stride=4 if args.quick else 8,
        n_layers=2,
    ).to(device)

    tl = DataLoader(lr_train, 32, shuffle=True)
    te = DataLoader(lr_test, 64)

    train_model(v1, tl, te, device, epochs_recall, lr=5e-3, amp=use_cuda)
    train_model(v2, tl, te, device, epochs_recall, lr=5e-3, amp=use_cuda)

    v1_recall = eval_recall_at_pos(v1, lr_test, device)
    v2_recall = eval_recall_at_pos(v2, lr_test, device)
    v1_full = eval_acc(v1, te, device)
    v2_full = eval_acc(v2, te, device)
    print(f"  Recall@v1={v1_recall:.4f}  v2={v2_recall:.4f}")
    print(f"  Full-seq  v1={v1_full:.4f}  v2={v2_full:.4f}")

    # ── Pattern copy (ayrı modeller) ─────────────────────────────────────────
    print("\n[2] Pattern copy")
    v1p = BulkTriggerLM(vocab, d_model=d_model, k=4, n_layers=2).to(device)
    v2p = BulkTriggerLMv2(
        vocab, d_model=d_model, k_short=8,
        medium_interval=16, long_interval=64 if args.quick else 128,
        trigger_stride=4, n_layers=2,
    ).to(device)
    pt = DataLoader(PatternDataset(vocab, 4, 24, 500), 64, shuffle=True)
    pe = DataLoader(PatternDataset(vocab, 4, 24, 100), 64)
    v1_pat = train_model(v1p, pt, pe, device, epochs_pattern, lr=3e-3, amp=use_cuda)
    v2_pat = train_model(v2p, pt, pe, device, epochs_pattern, lr=3e-3, amp=use_cuda)
    print(f"  Pattern v1={v1_pat:.4f}  v2={v2_pat:.4f}")

    # ── Latency ──────────────────────────────────────────────────────────────
    print("\n[3] Inference latency (ms/token)")
    std = StandardDecoderLM(vocab, d_model=d_model, n_layers=2, max_len=2048).to(device)
    seq_lens = [32, 128, 256, 512] if args.quick else [32, 64, 128, 256, 512, 1024]
    lat_v1 = bench_bulk_v1(v1p, device, seq_lens)
    lat_v2 = bench_bulk_v2(v2p, device, seq_lens)
    lat_std = bench_standard_nocache(std, device, seq_lens)

    o1 = lat_v2[-1]["ms_per_tok"] < lat_v2[0]["ms_per_tok"] * 2.5

    print(f"{'Seq':>6} {'v1':>8} {'v2':>8} {'std':>8} {'KV MB':>8}")
    for i, slen in enumerate(seq_lens):
        print(
            f"{slen:>6} {lat_v1[i]['ms_per_tok']:>8.3f} {lat_v2[i]['ms_per_tok']:>8.3f} "
            f"{lat_std[i]['ms_per_tok']:>8.3f} {lat_std[i]['kv_mb']:>8.3f}"
        )

    out = {
        "device": str(device),
        "gap": gap,
        "params_v1": count_params(v1),
        "params_v2": count_params(v2),
        "v1_recall_acc": round(v1_recall, 4),
        "v2_recall_acc": round(v2_recall, 4),
        "v1_full_acc": round(v1_full, 4),
        "v2_full_acc": round(v2_full, 4),
        "v1_pattern_acc": round(v1_pat, 4),
        "v2_pattern_acc": round(v2_pat, 4),
        "latency_v1": lat_v1,
        "latency_v2": lat_v2,
        "latency_standard": lat_std,
        "o1_v2_verified": o1,
        "recall_winner": "v2" if v2_recall > v1_recall else "v1",
        "note": "v1 kausal cross-attn düzeltmesi uygulandı (pozisyon sızıntısı giderildi)",
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nRecall kazanan: {out['recall_winner']}")
    print(f"O(1) v2: {'✓' if o1 else '✗'}")
    print(f"Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
