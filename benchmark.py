"""
HFP v2 Benchmark — Esnek hiperparametre mimarisi
==================================================
Fiziksel sabit YOK. HFPConfig ile test edilebilir prensipler.

Koşular:
  A) Standart MLP (baseline)
  B) HFP default config (hfp_config.json)
  C) HFP fast preset (stiff odaklı)
  D) HFP efficient preset (rank + zenon)
  E) HFP sadece bulk (stiff/zenon kapalı)
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from statistics import mean, stdev

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hfp_config import HFPConfig, build_hfp_model, train_with_hfp


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_mnist_loaders(batch_size: int = 128, data_dir: str = "./data", seed: int = 42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    n_val = 6000
    train, val = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=512),
        DataLoader(test, batch_size=512),
    )


def run_config(label: str, config: HFPConfig, seed: int, device, data_dir: str) -> dict:
    set_seed(seed)
    train_l, val_l, test_l = make_mnist_loaders(seed=seed, data_dir=data_dir)
    config.max_epochs = config.max_epochs  # preserve

    if not config.use_bulk and label.startswith("A"):
        from hfp_principles import StandardMLP
        model = StandardMLP(784, 512, 10)
    else:
        model = build_hfp_model(784, 512, 10, config)

    result = train_with_hfp(model, config, train_l, val_l, test_l, device)
    print(
        f"  seed={seed} acc={result.test_acc*100:.2f}% ep={result.epochs_run} "
        f"{result.elapsed_sec}s rank={config.bulk_rank} p={config.stiffness_p}"
    )
    return {
        "label": label,
        "seed": seed,
        "config": asdict(config),
        "test_acc": result.test_acc,
        "test_loss": result.test_loss,
        "params": result.params,
        "epochs": result.epochs_run,
        "elapsed_sec": result.elapsed_sec,
        "inference_ms": result.inference_ms,
    }


def aggregate(runs: list[dict]) -> dict:
    accs = [r["test_acc"] for r in runs]
    times = [r["elapsed_sec"] for r in runs]
    epochs = [r["epochs"] for r in runs]

    def stat(v):
        return {"mean": round(mean(v), 4), "std": round(stdev(v), 4) if len(v) > 1 else 0.0}

    return {
        "label": runs[0]["label"],
        "params": runs[0]["params"],
        "config": runs[0]["config"],
        "test_acc": stat(accs),
        "elapsed_sec": stat(times),
        "epochs": stat([float(e) for e in epochs]),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", type=str, default="hfp_config.json")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = HFPConfig(use_bulk=False, use_stiff=False, use_zenon=False, max_epochs=args.epochs)
    default_cfg = HFPConfig.from_json(args.config)
    default_cfg.max_epochs = args.epochs
    fast_cfg = HFPConfig.fast()
    fast_cfg.max_epochs = args.epochs
    efficient_cfg = HFPConfig.efficient()
    efficient_cfg.max_epochs = args.epochs
    bulk_only = HFPConfig(use_bulk=True, use_stiff=False, use_zenon=False, max_epochs=args.epochs)

    experiments = [
        ("A) Standart MLP (baseline)", baseline),
        ("B) HFP default config", default_cfg),
        ("C) HFP fast preset", fast_cfg),
        ("D) HFP efficient preset", efficient_cfg),
        ("E) HFP bulk only", bulk_only),
    ]

    print("=" * 70)
    print("HFP v2 BENCHMARK — Esnek hiperparametre mimarisi")
    print(f"Fiziksel η̃ sabiti YOK | stiffness_p ve threshold test edilebilir")
    print(f"Cihaz: {device} | Seeds: {args.seeds} | Epochs: {args.epochs}")
    print("=" * 70)

    all_results = []
    for label, cfg in experiments:
        print(f"\n▶ {label}")
        runs = [run_config(label, cfg, s, device, args.data_dir) for s in args.seeds]
        agg = aggregate(runs)
        all_results.append(agg)
        acc = agg["test_acc"]
        print(f"  → acc={acc['mean']*100:.2f}%±{acc['std']*100:.2f} | {agg['elapsed_sec']['mean']:.1f}s")

    print("\n" + "=" * 90)
    print(f"{'Konfigürasyon':<35} {'Params':>8} {'Acc':>12} {'Epoch':>7} {'Süre':>7}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['label']:<35} {r['params']:>8,} "
            f"{r['test_acc']['mean']*100:>6.2f}±{r['test_acc']['std']*100:>4.2f}% "
            f"{r['epochs']['mean']:>7.1f} {r['elapsed_sec']['mean']:>6.1f}s"
        )
    print("=" * 90)

    out = {"meta": {"version": "hfp_v2", "seeds": args.seeds, "no_physics_constants": True}, "results": all_results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
