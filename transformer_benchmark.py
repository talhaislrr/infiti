"""
HFP × Transformer Benchmark (v2)
================================
Encoder Transformer + HFPConfig hiperparametreleri.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader, Dataset

from hfp_config import HFPConfig, HFPStiffTransientScheduler, HFPZenonQuantizationScheduler, _batch_grad_norm
from hfp_principles import StiffTransientEarlyStopping
from hfp_transformer import FFNMode, HFPTransformerClassifier, count_ffn_params, count_trainable_params


CATEGORIES = [
    "rec.sport.baseball",
    "sci.space",
    "comp.graphics",
    "talk.politics.misc",
]


@dataclass
class TxConfig:
    label: str
    ffn_mode: FFNMode
    max_epochs: int = 15
    hfp_config: HFPConfig | None = None
    use_generic_stop: bool = False


@dataclass
class TxResult:
    label: str
    seed: int
    params: int
    ffn_params: int
    epochs: int
    test_acc: float
    elapsed_sec: float
    final_rate: float


class Vocab:
    def __init__(self, max_size: int = 8000):
        self.max_size = max_size
        self.token2id: dict[str, int] = {"<pad>": 0, "<unk>": 1}

    def build(self, texts: list[str]):
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        for word, _ in counter.most_common(self.max_size - 2):
            self.token2id[word] = len(self.token2id)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.token2id.get(t, 1) for t in tokenize(text)][:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    @property
    def size(self) -> int:
        return len(self.token2id)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 1]


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab, max_len: int):
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = torch.tensor(self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long)
        return ids, torch.tensor(self.labels[idx], dtype=torch.long)


def load_data(max_train: int | None = None, max_test: int | None = None):
    train = fetch_20newsgroups(
        subset="train", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )
    test = fetch_20newsgroups(
        subset="test", categories=CATEGORIES, remove=("headers", "footers", "quotes")
    )
    train_texts, train_labels = list(train.data), list(train.target)
    test_texts, test_labels = list(test.data), list(test.target)
    if max_train:
        train_texts, train_labels = train_texts[:max_train], train_labels[:max_train]
    if max_test:
        test_texts, test_labels = test_texts[:max_test], test_labels[:max_test]
    split = int(len(train_texts) * 0.9)
    val_texts, val_labels = train_texts[split:], train_labels[split:]
    train_texts, train_labels = train_texts[:split], train_labels[:split]
    vocab = Vocab()
    vocab.build(train_texts)
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vocab


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


def run_experiment(cfg: TxConfig, seed: int, device, loaders, vocab) -> TxResult:
    set_seed(seed)
    train_l, val_l, test_l = loaders
    hfp = cfg.hfp_config or HFPConfig(
        initial_lr=3e-4,
        max_epochs=cfg.max_epochs,
        use_bulk=False,
        use_stiff=False,
        use_zenon=False,
    )
    hfp.max_epochs = cfg.max_epochs

    mode = cfg.ffn_mode
    bulk_rank = hfp.bulk_rank if mode == FFNMode.BULK else 128

    model = HFPTransformerClassifier(
        vocab_size=vocab.size,
        num_classes=len(CATEGORIES),
        mode=mode,
        bulk_rank=bulk_rank,
    ).to(device)

    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=hfp.initial_lr, weight_decay=0.01)
    stiff = HFPStiffTransientScheduler(opt, hfp) if hfp.use_stiff else None
    generic_stop = StiffTransientEarlyStopping(k=3, stiffness_threshold=0.001, min_epochs=5)
    zenon = HFPZenonQuantizationScheduler(hfp, len(train_l) * hfp.max_epochs) if hfp.use_zenon else None

    final_rate = 0.0
    t0 = time.time()
    epochs_run = 0

    for epoch in range(1, hfp.max_epochs + 1):
        model.train()
        for x, y in train_l:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if zenon:
                zenon.step()
            ctx = zenon.training_context(device.type) if zenon else contextlib.nullcontext()
            with ctx:
                logits = model(x)
                loss = crit(logits, y)
                if mode == FFNMode.BULK:
                    loss = loss + model.bulk_reg_loss()
            loss.backward()
            if zenon:
                zenon.record_grad_norm(_batch_grad_norm(model))
            opt.step()

        val_loss, _ = evaluate(model, val_l, device)
        epochs_run = epoch
        stop = False
        if stiff:
            _, stop = stiff.step(epoch, val_loss)
            final_rate = stiff.loss_rate_history[-1] if stiff.loss_rate_history else 0.0
        elif cfg.use_generic_stop:
            stop = generic_stop.step(epoch, val_loss)
        if stop:
            break

    _, test_acc = evaluate(model, test_l, device)
    return TxResult(
        label=cfg.label,
        seed=seed,
        params=count_trainable_params(model),
        ffn_params=count_ffn_params(model),
        epochs=epochs_run,
        test_acc=test_acc,
        elapsed_sec=round(time.time() - t0, 2),
        final_rate=round(final_rate, 6),
    )


def aggregate(runs: list[TxResult]) -> dict:
    def stat(vals):
        return {"mean": round(mean(vals), 4), "std": round(stdev(vals), 4) if len(vals) > 1 else 0.0}

    return {
        "label": runs[0].label,
        "params": runs[0].params,
        "ffn_params": runs[0].ffn_params,
        "test_acc": stat([r.test_acc for r in runs]),
        "elapsed_sec": stat([r.elapsed_sec for r in runs]),
        "epochs": stat([float(r.epochs) for r in runs]),
        "runs": [{"seed": r.seed, "test_acc": r.test_acc, "epochs": r.epochs} for r in runs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="transformer_results.json")
    args = parser.parse_args()

    max_train = max_test = None
    if args.quick:
        args.seeds = [42]
        args.epochs = 8
        max_train, max_test = 800, 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t, train_y, val_t, val_y, test_t, test_y, vocab = load_data(max_train, max_test)

    def loaders(seed):
        set_seed(seed)
        mk = lambda t, y, sh=False: DataLoader(NewsDataset(t, y, vocab, 128), batch_size=32, shuffle=sh)
        return mk(train_t, train_y, True), mk(val_t, val_y), mk(test_t, test_y)

    hfp_stiff = HFPConfig(
        initial_lr=3e-4, max_epochs=args.epochs,
        bulk_rank=128, use_bulk=True, use_stiff=True, use_zenon=False,
        stiffness_p=1.0, stiffness_threshold=0.001,
    )
    hfp_full = HFPConfig.efficient()
    hfp_full.initial_lr = 3e-4
    hfp_full.max_epochs = args.epochs

    configs = [
        TxConfig("A) Standart Transformer FFN", FFNMode.STANDARD, args.epochs),
        TxConfig("B) BulkLinear + generic stop", FFNMode.BULK, args.epochs, use_generic_stop=True),
        TxConfig("C) BulkLinear + HFP Stiff (v2)", FFNMode.BULK, args.epochs, hfp_config=hfp_stiff),
        TxConfig("D) BulkLinear + Stiff + Zenon (v2)", FFNMode.BULK, args.epochs, hfp_config=hfp_full),
    ]

    print("=" * 70)
    print("HFP × TRANSFORMER v2")
    print(f"Cihaz: {device} | Seeds: {args.seeds} | Epochs: {args.epochs}")
    print("=" * 70)

    results = []
    for cfg in configs:
        print(f"\n▶ {cfg.label}")
        runs = [run_experiment(cfg, s, device, loaders(s), vocab) for s in args.seeds]
        for r in runs:
            print(f"  seed={r.seed} acc={r.test_acc*100:.2f}% ep={r.epochs} {r.elapsed_sec}s")
        results.append(aggregate(runs))

    with open(args.output, "w") as f:
        json.dump({"meta": {"version": "hfp_v2"}, "results": results}, f, indent=2)
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
