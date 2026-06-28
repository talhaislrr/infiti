"""
HFP Teori vs Generic Mühendislik — Adil Karşılaştırma
======================================================
Paper sabitleri (η̃=0.407) kullanan teorik stack ile
önceki generic implementasyonu ve standart baseline'ı karşılaştırır.

Koşular:
  A) Standart MLP + Adam                    (endüstri baseline)
  B) Generic BulkLinear + generic early stop (önceki "HFP" isimli kod)
  C) HFP Teori: ProjectedMLP + Cubic Stiff   (Paper I+II, Zeno yok)
  D) HFP Teori: Tam stack + Zeno 1/N         (Paper II §3.6)
  E) LoRA-style düşük rank (AI endüstri ref.)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hfp_principles import BulkMLP, StandardMLP, StiffTransientEarlyStopping
from hfp_theory import (
    ETA_TILDE,
    THETA_HAAR,
    HFPProjectedMLP,
    StiffTransientCubic,
    ZenoLeakageRegularizer,
    StandardZenoRegularizer,
    bulk_projection_rank,
    simulate_zeno_scaling,
)


@dataclass
class TheoryConfig:
    label: str
    mode: str  # standard | generic | theory | theory_zeno | lora
    max_epochs: int = 30
    rank: int = 128


@dataclass
class TheoryResult:
    label: str
    seed: int
    params: int
    rank: int
    epochs: int
    test_acc: float
    test_loss: float
    elapsed_sec: float
    final_theta: float
    final_flow: float
    val_acc: list[float] = field(default_factory=list)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(batch_size: int = 128, data_dir: str = "./data", seed: int = 42):
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


def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, correct / n


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class LoRAMLP(nn.Module):
    """Endüstri referansı: rank-r LoRA-style FFN."""

    def __init__(self, in_dim, hidden, out_dim, rank=128, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.fc1 = nn.Linear(in_dim, hidden)
        self.lora1_A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora1_B = nn.Parameter(torch.zeros(hidden, rank))
        self.fc2 = nn.Linear(hidden, hidden)
        self.lora2_A = nn.Parameter(torch.randn(rank, hidden) * 0.01)
        self.lora2_B = nn.Parameter(torch.zeros(hidden, rank))
        self.fc3 = nn.Linear(hidden, out_dim)

    def _lora(self, x, W, A, B):
        return x @ W.T + self.scale * (x @ A.T @ B.T)

    def forward(self, x):
        h = torch.relu(self._lora(x, self.fc1.weight, self.lora1_A, self.lora1_B) + self.fc1.bias)
        h = torch.relu(self._lora(h, self.fc2.weight, self.lora2_A, self.lora2_B) + self.fc2.bias)
        return self.fc3(h)


def build_model(cfg: TheoryConfig):
    if cfg.mode == "standard":
        return StandardMLP(784, 512, 10)
    if cfg.mode == "generic":
        return BulkMLP(784, 512, 10, rank=cfg.rank)
    if cfg.mode in ("theory", "theory_zeno"):
        return HFPProjectedMLP(784, 512, 10)
    if cfg.mode == "lora":
        return LoRAMLP(784, 512, 10, rank=cfg.rank)
    raise ValueError(cfg.mode)


def run(cfg: TheoryConfig, seed: int, device: torch.device, data_dir: str) -> TheoryResult:
    set_seed(seed)
    train_l, val_l, test_l = make_loaders(seed=seed, data_dir=data_dir)
    model = build_model(cfg).to(device)
    params = count_params(model)
    rank = getattr(model, "projection_rank", cfg.rank) if hasattr(model, "projection_rank") else cfg.rank

    crit = nn.CrossEntropyLoss()
    base_lr = 1e-3
    opt = optim.Adam(model.parameters(), lr=base_lr)

    generic_stop = StiffTransientEarlyStopping(k=3, stiffness_threshold=0.001, min_epochs=5)
    cubic_stop = StiffTransientCubic(theta_min=0.02, min_epochs=3)
    zeno_hfp = ZenoLeakageRegularizer(measurement_interval=50)

    val_accs: list[float] = []
    final_theta, final_flow = 1.0, ETA_TILDE
    t0 = time.time()
    epochs_run = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for x, y in train_l:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)

            if cfg.mode == "generic":
                loss = loss + model.bulk_regularization_loss()
            elif cfg.mode in ("theory", "theory_zeno"):
                loss = loss + model.haar_loss()

            loss.backward()

            if cfg.mode == "theory_zeno":
                zeno_hfp.set_theta_0(cubic_stop.theta_history[-1] if cubic_stop.theta_history else 1.0)
                zeno_hfp.apply_leakage(model)

            opt.step()

        val_loss, val_acc = evaluate(model, val_l, device)
        val_accs.append(val_acc)
        epochs_run = epoch

        stop = False
        if cfg.mode == "generic":
            stop = generic_stop.step(epoch, val_loss)
        elif cfg.mode in ("theory", "theory_zeno"):
            stop, final_theta, final_flow = cubic_stop.step(epoch, val_loss)
            new_lr = cubic_stop.learning_rate(base_lr, final_theta)
            for pg in opt.param_groups:
                pg["lr"] = new_lr

        if stop:
            break

    test_loss, test_acc = evaluate(model, test_l, device)
    return TheoryResult(
        label=cfg.label,
        seed=seed,
        params=params,
        rank=rank,
        epochs=epochs_run,
        test_acc=test_acc,
        test_loss=test_loss,
        elapsed_sec=round(time.time() - t0, 2),
        final_theta=round(final_theta, 5),
        final_flow=round(final_flow, 6),
        val_acc=val_accs,
    )


def aggregate(runs: list[TheoryResult]) -> dict:
    accs = [r.test_acc for r in runs]
    times = [r.elapsed_sec for r in runs]
    epochs = [r.epochs for r in runs]

    def s(v):
        return {"mean": round(mean(v), 4), "std": round(stdev(v), 4) if len(v) > 1 else 0.0}

    return {
        "label": runs[0].label,
        "mode": runs[0].label,
        "params": runs[0].params,
        "rank": runs[0].rank,
        "test_acc": s(accs),
        "elapsed_sec": s(times),
        "epochs": s([float(e) for e in epochs]),
        "eta_tilde": ETA_TILDE,
        "theta_haar_rad": THETA_HAAR,
        "runs": [{"seed": r.seed, "test_acc": r.test_acc, "epochs": r.epochs,
                  "elapsed_sec": r.elapsed_sec, "final_theta": r.final_theta} for r in runs],
    }


def print_verdict(results: list[dict], baseline: dict):
    print("\n" + "═" * 80)
    print("HFP TEORİ DEĞERLENDİRMESİ — AI DÜNYASINDA İŞ YAPAR MI?")
    print("═" * 80)

    theory = next((r for r in results if "Teori" in r["label"] and "Zeno" not in r["label"]), None)
    theory_zeno = next((r for r in results if "Zeno" in r["label"]), None)
    generic = next((r for r in results if "Generic" in r["label"]), None)
    lora = next((r for r in results if "LoRA" in r["label"]), None)

    b_acc = baseline["test_acc"]["mean"]
    print(f"\nBaseline acc: {b_acc*100:.2f}%")

    for r in results:
        acc = r["test_acc"]["mean"]
        delta = (acc - b_acc) * 100
        ps = r["params"]
        t = r["elapsed_sec"]["mean"]
        sign = "+" if delta >= 0 else ""
        print(f"  {r['label'][:50]}")
        print(f"    acc={acc*100:.2f}% ({sign}{delta:.2f}pp) | params={ps:,} | {t:.1f}s | rank={r['rank']}")

    print("\n── Teori vs Generic ──")
    if theory and generic:
        t_better = theory["test_acc"]["mean"] >= generic["test_acc"]["mean"] - 0.005
        t_faster = theory["elapsed_sec"]["mean"] <= generic["elapsed_sec"]["mean"]
        print(f"  Teori acc ≥ Generic?     {'EVET' if t_better else 'HAYIR'}")
        print(f"  Teori daha hızlı?        {'EVET' if t_faster else 'HAYIR'}")
        print(f"  Teoriden türetilen rank:  {theory['rank']} (sin²({THETA_HAAR})≈{math.sin(THETA_HAAR)**2:.3f})")

    print("\n── Teori vs LoRA (AI endüstri) ──")
    if theory and lora:
        lora_wins = lora["test_acc"]["mean"] > theory["test_acc"]["mean"]
        print(f"  LoRA daha iyi acc?       {'EVET' if lora_wins else 'HAYIR'}")
        print(f"  LoRA params: {lora['params']:,} vs Teori: {theory['params']:,}")

    print("\n── Zeno 1/N vs 1/N² (Paper II Tablo 1 simülasyonu) ──")
    zeno = simulate_zeno_scaling()
    for row in zeno["zeno_table"]:
        print(f"  N={row['N']:>6} | HFP/Std oranı={row['ratio_hfp_over_std']:.1f} | beklenen≈{row['expected_ratio_approx']:.1f}")

    print("\n── SONUÇ ──")
    if theory:
        acc_ok = theory["test_acc"]["mean"] >= b_acc * 0.995
        param_ok = theory["params"] < baseline["params"] * 0.6
        time_ok = theory["elapsed_sec"]["mean"] < baseline["elapsed_sec"]["mean"] * 0.7
        wins = sum([acc_ok, param_ok, time_ok])
        if wins >= 2:
            print("  HFP TEORİSİ bu testte generic'ten FARKLI ve KULLANILABILIR sonuç verdi.")
        elif theory["test_acc"]["mean"] >= generic["test_acc"]["mean"] - 0.003:
            print("  HFP TEORİSİ generic ile EŞDEĞER — teorik sabitler ek avantaj sağlamadı.")
        else:
            print("  HFP TEORİSİ generic/endüstri yöntemlerinin GERİSİNDE kaldı.")
        print(f"  AI dünyasında pratik değer: {'ORTA-YÜKSEK' if wins>=2 else 'DÜŞÜK-ORTA' if wins==1 else 'DÜŞÜK'}")
        print("  Not: η̃=0.407 ve Haar projeksiyonu fizik/kuantum için tasarlandı;")
        print("       ML'de kanıt için LLM fine-tune + Zeno qubit deneyi gerekir.")
    print("═" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output", default="theory_results.json")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        TheoryConfig("A) Standart MLP (baseline)", "standard", args.epochs),
        TheoryConfig("B) Generic BulkLinear + generic stop", "generic", args.epochs, rank=128),
        TheoryConfig("C) HFP Teori: Haar Proj + Cubic Stiff", "theory", args.epochs),
        TheoryConfig("D) HFP Teori: Tam stack + Zeno 1/N", "theory_zeno", args.epochs),
        TheoryConfig("E) LoRA-style (AI endüstri ref.)", "lora", args.epochs, rank=128),
    ]

    print("=" * 70)
    print("HFP TEORİ BENCHMARK")
    print(f"η̃={ETA_TILDE} | θ_Haar={THETA_HAAR} rad | rank teorik={bulk_projection_rank(512,512)}")
    print(f"Cihaz: {device} | Seeds: {args.seeds}")
    print("=" * 70)

    all_agg = []
    for cfg in configs:
        print(f"\n▶ {cfg.label}")
        runs = [run(cfg, s, device, args.data_dir) for s in args.seeds]
        for r in runs:
            print(f"  seed={r.seed} acc={r.test_acc*100:.2f}% ep={r.epochs} θ={r.final_theta} {r.elapsed_sec}s")
        agg = aggregate(runs)
        all_agg.append(agg)

    print_verdict(all_agg, all_agg[0])

    out = {"meta": {"eta_tilde": ETA_TILDE, "theta_haar": THETA_HAAR, "seeds": args.seeds},
           "zeno_simulation": simulate_zeno_scaling(), "results": all_agg}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
