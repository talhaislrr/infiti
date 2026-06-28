#!/usr/bin/env python3
"""
BulkTrigger Faz 1 — TinyLlama + BulkMemoryAdapter eğitimi
==========================================================
Mac MPS / CUDA / CPU. Base frozen, sadece adapter eğitilir.

Kullanım:
  python3 bulk_trigger_hybrid_train.py --quick
  python3 bulk_trigger_hybrid_train.py --epochs 3 --seq-len 256
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import TinyLlamaWithBulk, load_tinyllama, save_adapter


class WikiTokenDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, seq_len: int):
        self.samples: list[torch.Tensor] = []
        buf: list[int] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
        for i in range(0, len(buf) - seq_len, seq_len // 2):
            chunk = buf[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return s[:-1], s[1:]


@torch.no_grad()
def eval_ppl(model, loader, device) -> float:
    model.eval()
    total_nll, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(input_ids=x, labels=y)
        if out.loss is not None:
            total_nll += out.loss.item() * y.numel()
            n += y.numel()
    return math.exp(total_nll / max(n, 1))


def train_adapter(
    hybrid: TinyLlamaWithBulk,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[float]:
    hybrid.adapter.train()
    for b in hybrid._swap_adapters:
        b.train()
    opt = torch.optim.AdamW(hybrid.trainable_parameters(), lr=lr, weight_decay=0.01)
    losses: list[float] = []
    for ep in range(epochs):
        ep_loss, steps = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = hybrid(input_ids=x, labels=y)
            loss = out.loss
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(hybrid.trainable_parameters()), 1.0)
            opt.step()
            ep_loss += loss.item()
            steps += 1
        losses.append(ep_loss / max(steps, 1))
        print(f"  epoch {ep + 1}/{epochs}  train_loss={losses[-1]:.4f}")
    return losses


@torch.no_grad()
def eval_base_ppl(base, loader, device) -> float:
    base.eval()
    total_nll, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = base(input_ids=x).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        total_nll += loss.item() * y.numel()
        n += y.numel()
    return math.exp(total_nll / max(n, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--trigger-stride", type=int, default=4)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--swap-layers", type=int, default=0, help="Faz 3: son N katman Bulk swap")
    parser.add_argument("--device", default=None, help="mps|cuda|cpu")
    parser.add_argument("--resume", default=None, help="checkpoints/bulk_adapter/adapter.pt")
    parser.add_argument("--output", default="bulk_hybrid_train_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=True)
    epochs = args.epochs or (1 if args.quick else 3)
    seq_len = args.seq_len or (128 if args.quick else 256)
    batch_size = args.batch_size or (2 if args.quick else 4)

    print("=" * 70)
    print("BulkTrigger Faz 1 — TinyLlama + BulkAdapter")
    print(device_summary(device))
    print("=" * 70)

    from datasets import load_dataset

    print("\n[1] Veri yükleniyor (WikiText-2)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    n_texts = 30 if args.quick else 300
    texts = [t for t in ds["text"] if t.strip()][:n_texts]
    val_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    val_texts = [t for t in val_ds["text"] if t.strip()][: max(10, n_texts // 5)]

    print(f"[2] TinyLlama yükleniyor ({dtype})...")
    t0 = time.time()
    base, tokenizer, model_name = load_tinyllama(device, dtype)

    train_data = WikiTokenDataset(texts, tokenizer, seq_len)
    val_data = WikiTokenDataset(val_texts, tokenizer, seq_len)
    if len(train_data) < 2:
        raise RuntimeError("Yetersiz eğitim verisi — seq-len küçült veya quick kapat")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    print(f"  train={len(train_data)} val={len(val_data)} seq={seq_len} batch={batch_size}")

    print("\n[3] Base perplexity (adapter yok)...")
    base_ppl = eval_base_ppl(base, val_loader, device)
    print(f"  base PPL = {base_ppl:.2f}")

    print("\n[4] BulkAdapter eğitimi...")
    hybrid = TinyLlamaWithBulk(
        base,
        k_short=8,
        medium_interval=16,
        long_interval=128,
        trigger_stride=args.trigger_stride,
        adaptive_trigger=args.adaptive,
        surprise_threshold=1.0,
        freeze_base=True,
        n_swap_layers=args.swap_layers,
    ).to(device)

    adapt_params = sum(p.numel() for p in hybrid.trainable_parameters())
    print(f"  adapter params = {adapt_params:,} (swap_layers={args.swap_layers})")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "top_adapter" in ckpt:
            hybrid.adapter.load_state_dict(ckpt["top_adapter"])
            for bulk, state in zip(hybrid._swap_adapters, ckpt.get("swap_adapters", [])):
                bulk.load_state_dict(state)
        else:
            hybrid.adapter.load_state_dict(ckpt)
        print(f"  resumed from {args.resume}")

    train_t0 = time.time()
    train_losses = train_adapter(hybrid, train_loader, device, epochs, args.lr)
    train_sec = time.time() - train_t0

    print("\n[5] Değerlendirme...")
    bulk_ppl = eval_ppl(hybrid, val_loader, device)
    print(f"  bulk PPL  = {bulk_ppl:.2f}")
    print(f"  delta     = {bulk_ppl - base_ppl:+.2f}")

    adapter_path = Path("checkpoints/bulk_adapter/adapter.pt")
    save_adapter(hybrid, adapter_path)
    print(f"\n  adapter kaydedildi: {adapter_path}")

    results = {
        "device": str(device),
        "model": model_name,
        "adapter_params": adapt_params,
        "seq_len": seq_len,
        "epochs": epochs,
        "trigger_stride": args.trigger_stride,
        "adaptive_trigger": args.adaptive,
        "swap_layers": args.swap_layers,
        "base_ppl": round(base_ppl, 3),
        "bulk_ppl": round(bulk_ppl, 3),
        "ppl_delta": round(bulk_ppl - base_ppl, 3),
        "train_losses": [round(x, 4) for x in train_losses],
        "train_sec": round(train_sec, 1),
        "load_sec": round(time.time() - t0 - train_sec, 1),
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSonuç: base={base_ppl:.2f} → bulk={bulk_ppl:.2f} ({train_sec:.0f}s)")
    print(f"Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
