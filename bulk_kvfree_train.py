#!/usr/bin/env python3
"""
Faz 3 — KVFree kuyruk katman eğitimi (compact + recall)
======================================================
  python3 bulk_kvfree_train.py --quick
  python3 bulk_kvfree_train.py --recall --epochs 3 --device mps
  python3 bulk_kvfree_train.py --mix --epochs 3 --resume checkpoints/bulk_swap/bulk_v2.pt
  python3 bulk_kvfree_train.py --conv --epochs 3 --device mps   # chat recall
  python3 bulk_kvfree_train.py --multifact --n-facts 3 --epochs 3 --device mps
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import load_tinyllama
from bulk_layer_swap import TinyLlamaKVFreeTail, count_swap_params, load_bulk_swap_checkpoint, save_bulk_swap_checkpoint
from bulk_recall_data import (
    RecallTokenDataset,
    build_recall_document,
    FILLER_PARAGRAPH,
    MultifactRecallTokenDataset,
    ConversationalRecallDataset,
    MixedRecallWikiDataset,
)
from bulk_trigger_hybrid_train import WikiTokenDataset, eval_base_ppl


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


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


@torch.no_grad()
def eval_recall_hit(
    model: TinyLlamaKVFreeTail,
    tokenizer,
    device,
    n_cases: int = 8,
    doc_tokens: int = 640,
    sliding_window: int = 512,
) -> dict:
    """Sentetik recall — fact başta, soru sonda."""
    from bulk_recall_data import FACT_KEYS, FACT_VALUES

    model.eval()
    hits = 0
    cases = []
    rng = random.Random(0)

    for i in range(n_cases):
        fk = FACT_KEYS[i % len(FACT_KEYS)]
        fv = FACT_VALUES[rng.randint(0, len(FACT_VALUES) - 1)]
        prompt = build_recall_document(
            fk, fv, FILLER_PARAGRAPH, doc_tokens, tokenizer,
            include_answer=False,
        )
        prompt = prompt.rstrip() + "\n"
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].to(device)

        model.sliding_window = sliding_window
        new_ids = model.generate_cached(ids, max_new_tokens=24)
        ans = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        hit = fv.lower() in ans.lower()
        hits += int(hit)
        cases.append({"fact": fv, "answer": ans[:120], "hit": hit})

    return {"hit_rate": hits / max(n_cases, 1), "hits": hits, "n": n_cases, "cases": cases}


def train_swap(
    model: TinyLlamaKVFreeTail,
    loader,
    device: torch.device,
    epochs: int,
    lr: float,
    accum_steps: int = 1,
    log_every: int = 5,
) -> list[float]:
    for layer in model.bulk_layers:
        layer.bulk.train()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.01)
    n_batches = len(loader)
    losses: list[float] = []
    for ep in range(epochs):
        ep_loss, steps = 0.0, 0
        opt.zero_grad()
        t_ep = time.time()
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            out = model(input_ids=x, labels=y)
            assert out.loss is not None
            (out.loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
                opt.step()
                opt.zero_grad()
            ep_loss += out.loss.item()
            steps += 1
            if step % log_every == 0:
                avg = ep_loss / steps
                elapsed = time.time() - t_ep
                print(
                    f"  ep{ep + 1}/{epochs}  step {step + 1}/{n_batches}  "
                    f"loss={out.loss.item():.4f}  avg={avg:.4f}  {elapsed:.0f}s",
                    flush=True,
                )
                _free(device)
        losses.append(ep_loss / max(steps, 1))
        print(f"  → epoch {ep + 1} done  avg_loss={losses[-1]:.4f}  {time.time() - t_ep:.0f}s")
        save_bulk_swap_checkpoint(model, str(Path(f"checkpoints/bulk_swap/bulk_v2_ep{ep + 1}.pt")))
        _free(device)
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--recall", action="store_true", help="Sadece sentetik recall verisi")
    parser.add_argument("--mix", action="store_true", help="WikiText + recall karışımı")
    parser.add_argument("--conv", action="store_true", help="Chat formatında conversational recall")
    parser.add_argument("--multifact", action="store_true", help="Multi-fact recall eğitimi")
    parser.add_argument("--n-facts", type=int, default=3, help="multifact modunda fact sayısı")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--d-bulk", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--swap-layers", type=int, default=4)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--resume", default=None, help="Mevcut swap checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_kvfree_train_results.json")
    args = parser.parse_args()

    compact = not (args.full or args.no_compact)
    device = pick_device(args.device)
    dtype = pick_dtype(device, train=True)
    epochs = args.epochs or (1 if args.quick else 3)
    # conv/multifact: seq_len 128 varsayılan (MPS için makul), recall: 256
    seq_len = args.seq_len or (
        128 if (args.conv or args.multifact) else
        256 if (args.recall or args.mix) else
        128
    )
    batch_size = args.batch_size or 1
    use_recall = args.recall or args.mix

    if args.conv:
        mode = "conv"
    elif args.multifact:
        mode = "multifact"
    elif args.recall:
        mode = "recall"
    elif args.mix:
        mode = "mix"
    else:
        mode = "wiki"

    print("=" * 70)
    print(f"Faz 3 KVFree Swap | {device_summary(device)}")
    print(f"  mode={mode} compact={compact} seq={seq_len} batch={batch_size}")
    print("=" * 70)

    base, tokenizer, model_name = load_tinyllama(device, dtype)

    from datasets import load_dataset

    n_texts = 30 if args.quick else 300

    if args.conv:
        n_samples = 30 if args.quick else 150
        train_ds: Dataset = ConversationalRecallDataset(
            tokenizer, n_samples=n_samples, seq_len=seq_len,
        )
        val_ds = ConversationalRecallDataset(
            tokenizer, n_samples=max(10, n_samples // 5), seq_len=seq_len, seed=99,
        )
    elif args.multifact:
        n_samples = 30 if args.quick else 200
        train_ds = MultifactRecallTokenDataset(
            tokenizer, n_samples=n_samples, seq_len=seq_len, n_facts=args.n_facts,
        )
        val_ds = MultifactRecallTokenDataset(
            tokenizer, n_samples=max(16, n_samples // 5), seq_len=seq_len,
            n_facts=args.n_facts, seed=99,
        )
    elif use_recall:
        n_recall = 50 if args.quick else 400
        recall_train = RecallTokenDataset(
            tokenizer, n_samples=n_recall, seq_len=seq_len, gap_range=(64, min(seq_len - 32, 480)),
        )
        recall_val = RecallTokenDataset(tokenizer, n_samples=max(16, n_recall // 5), seq_len=seq_len, seed=99)
        if args.recall:
            train_ds = recall_train
            val_ds = recall_val
        else:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"] if t.strip()][:n_texts]
            val_w = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            val_texts = [t for t in val_w["text"] if t.strip()][: max(10, n_texts // 5)]
            wiki_train = WikiTokenDataset(texts, tokenizer, seq_len)
            wiki_val = WikiTokenDataset(val_texts, tokenizer, seq_len)
            train_ds = MixedRecallWikiDataset(wiki_train, recall_train, recall_ratio=0.5)
            val_ds = MixedRecallWikiDataset(wiki_val, recall_val, recall_ratio=0.5)
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if t.strip()][:n_texts]
        val_ds_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        val_texts = [t for t in val_ds_raw["text"] if t.strip()][: max(10, n_texts // 5)]
        train_ds = WikiTokenDataset(texts, tokenizer, seq_len)
        val_ds = WikiTokenDataset(val_texts, tokenizer, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"\nTrain samples: {len(train_ds)}  |  Val: {len(val_ds)}")
    print("Base PPL...")
    base_ppl = eval_base_ppl(base, val_loader, device)
    print(f"  base PPL = {base_ppl:.2f}")

    model = TinyLlamaKVFreeTail(
        base, n_swap_layers=args.swap_layers, sliding_window=args.sliding_window,
        compact=compact, d_bulk=args.d_bulk,
    ).to(device)

    resume = args.resume or ("checkpoints/bulk_swap/bulk_v2.pt" if (args.mix or args.conv or args.multifact) else None)
    if resume and Path(resume).exists():
        load_bulk_swap_checkpoint(model, resume, device)
        print(f"  resumed: {resume}")

    n_params = count_swap_params(model)
    print(f"  swap params = {n_params:,}")

    t0 = time.time()
    losses = train_swap(model, train_loader, device, epochs, args.lr, args.accum_steps)
    swap_ppl = eval_ppl(model, val_loader, device)
    train_sec = time.time() - t0

    ckpt = Path("checkpoints/bulk_swap/bulk_v2.pt")
    save_bulk_swap_checkpoint(model, str(ckpt))

    recall_eval = {}
    if use_recall or args.multifact:
        print("\nRecall eval (sliding_window=512)...")
        recall_eval["window_512"] = eval_recall_hit(model, tokenizer, device, sliding_window=512)
        print(f"  hit={recall_eval['window_512']['hits']}/{recall_eval['window_512']['n']}")
        print("Recall eval (sliding_window=0)...")
        recall_eval["window_0"] = eval_recall_hit(model, tokenizer, device, sliding_window=0)
        print(f"  hit={recall_eval['window_0']['hits']}/{recall_eval['window_0']['n']}")

    print(f"\n  KVFree PPL = {swap_ppl:.2f}  (delta {swap_ppl - base_ppl:+.2f})")
    print(f"  checkpoint: {ckpt}")

    results = {
        "model": model_name,
        "train_mode": mode,
        "compact": compact,
        "d_bulk": args.d_bulk,
        "swap_layers": args.swap_layers,
        "sliding_window": args.sliding_window,
        "swap_params": n_params,
        "seq_len": seq_len,
        "epochs": epochs,
        "n_facts": args.n_facts if args.multifact else None,
        "base_ppl": round(base_ppl, 3),
        "kvfree_ppl": round(swap_ppl, 3),
        "train_losses": [round(x, 4) for x in losses],
        "train_sec": round(train_sec, 1),
        "recall_eval": recall_eval,
    }
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
