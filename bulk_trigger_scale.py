#!/usr/bin/env python3
"""
BulkTrigger Ölçeklendirme — WikiText-2 + TinyLlama hibrit adapter
=================================================================
Kullanım:
  python3 bulk_trigger_scale.py --quick
  python3 bulk_trigger_scale.py --tinyllama --quick
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import TinyLlamaWithBulk, load_tinyllama
from bulk_trigger_v2 import BulkTriggerLMv2


# ── Char-level WikiText ───────────────────────────────────────────────────────

def build_char_vocab(texts: list[str]) -> dict[str, int]:
    chars = sorted(set("".join(texts)))
    vocab = {c: i + 1 for i, c in enumerate(chars)}
    vocab["<pad>"] = 0
    return vocab


class CharWikiDataset(Dataset):
    def __init__(self, texts: list[str], vocab: dict[str, int], seq_len: int):
        self.samples = []
        for t in texts:
            if len(t) < seq_len + 1:
                continue
            for i in range(0, len(t) - seq_len, seq_len // 2):
                chunk = t[i : i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    ids = torch.tensor([vocab.get(c, 0) for c in chunk])
                    self.samples.append((ids[:-1], ids[1:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


@torch.no_grad()
def perplexity(model, loader, device) -> float:
    model.eval()
    total_nll, n = 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_nll += crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += y.numel()
    return math.exp(total_nll / max(n, 1))


def run_wikitext_v2(device, quick: bool) -> dict:
    from datasets import load_dataset

    print("\n── WikiText-2 + BulkTriggerLMv2 ──")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t.strip()][: 50 if quick else 500]
    vocab = build_char_vocab(texts)
    seq_len = 64 if quick else 128
    data = CharWikiDataset(texts, vocab, seq_len)
    if len(data) < 10:
        return {"error": "yetersiz veri"}
    loader = DataLoader(data, batch_size=16 if quick else 32, shuffle=True)
    val_loader = DataLoader(data, batch_size=32)

    model = BulkTriggerLMv2(
        len(vocab), d_model=256 if quick else 384,
        n_layers=3, k_short=8, medium_interval=16, long_interval=128,
        trigger_stride=8, max_len=512,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    epochs = 2 if quick else 5

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x).reshape(-1, len(vocab)), y.reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item()
        ppl = perplexity(model, val_loader, device)
        print(f"  epoch {ep} loss={total/len(loader):.3f} ppl={ppl:.1f}")

    return {
        "vocab_size": len(vocab),
        "d_model": model.d_model,
        "params": sum(p.numel() for p in model.parameters()),
        "perplexity": round(perplexity(model, val_loader, device), 2),
        "train_sec": round(time.time() - t0, 1),
    }


def run_tinyllama_bulk(device, dtype, quick: bool) -> dict:
    print("\n── TinyLlama + BulkMemoryAdapter (forward test) ──")
    try:
        base, tokenizer, model_name = load_tinyllama(device, dtype)
    except ImportError:
        return {"error": "pip install transformers"}

    wrapped = TinyLlamaWithBulk(base, k_short=8, trigger_stride=4).to(device)
    text = "BulkTrigger hierarchical memory enables efficient long-context language modeling."
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = wrapped(input_ids=inputs["input_ids"])
    elapsed = time.perf_counter() - t0

    return {
        "model": model_name,
        "adapter_params": sum(p.numel() for p in wrapped.adapter.parameters()),
        "forward_sec": round(elapsed, 3),
        "logits_shape": list(out.logits.shape),
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tinyllama", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_trigger_scale_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device)
    print("=" * 70)
    print(f"BulkTrigger SCALE | {device_summary(device)}")
    print("=" * 70)

    results = {"device": str(device), "wikitext_v2": run_wikitext_v2(device, args.quick)}

    if args.tinyllama:
        results["tinyllama"] = run_tinyllama_bulk(device, dtype, args.quick)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nKaydedildi: {args.output}")
    print("\nTam eğitim için: python3 bulk_trigger_hybrid_train.py --quick")


if __name__ == "__main__":
    main()
