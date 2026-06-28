"""
HFP Benchmark - Görselleştirme (v1 + v2 uyumlu)
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#F44336"]


def load_results(path: str = "results.json") -> dict:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"results": data}
    if "results" in data and "ablation" not in data:
        return {"results": data["results"], "meta": data.get("meta", {})}
    return data


def plot_v2_bar(results: list[dict], save_path: str = "hfp_benchmark.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HFP v2 Benchmark", fontsize=14, fontweight="bold")

    labels = [r["label"][:28] for r in results]
    accs = [r["test_acc"]["mean"] * 100 for r in results]
    stds = [r["test_acc"]["std"] * 100 for r in results]
    times = [r["elapsed_sec"]["mean"] for r in results]
    params = [r["params"] for r in results]

    x = np.arange(len(labels))
    axes[0].bar(x, accs, yerr=stds, capsize=4, color=COLORS[: len(labels)], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Doğruluk")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, times, color=COLORS[: len(labels)], alpha=0.6, label="Süre (s)")
    ax2 = axes[1].twinx()
    ax2.plot(x, [p / 1000 for p in params], "o--", color="#333", label="Param (K)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Süre (s)")
    ax2.set_ylabel("Param (K)")
    axes[1].set_title("Süre & Parametre")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Grafik: {save_path}")


def print_analysis(results: list[dict]):
    if not results:
        return
    baseline = results[0]
    print("\n── Özet ──")
    for r in results:
        acc = r["test_acc"]["mean"] * 100
        delta = acc - baseline["test_acc"]["mean"] * 100
        print(
            f"  {r['label'][:40]:40} acc={acc:5.2f}% ({delta:+.2f}pp) "
            f"ep={r['epochs']['mean']:.1f} t={r['elapsed_sec']['mean']:.1f}s"
        )


if __name__ == "__main__":
    data = load_results()
    results = data.get("results") or data.get("ablation", [])
    plot_v2_bar(results)
    print_analysis(results)
