"""
LoRA Ablation — Baseline vs HFP Stiff + Rank Sweep
==================================================
Zenon kapalı. Her varyant eğitilir ve HF benchmark ile ölçülür.

Kullanım:
  python3 lora_ablation.py              # tam: rank 32/64/128, 10 epoch
  python3 lora_ablation.py --quick      # smoke: rank 64, 3 epoch
  python3 lora_ablation.py --train-only
  python3 lora_ablation.py --bench-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hfp_lora_finetune import load_lora_settings, train_lora

import torch


def build_variants(settings: dict, quick: bool) -> list[dict]:
    ranks = settings["quick_ranks"] if quick else settings["ranks"]
    variants = []
    for rank in ranks:
        variants.append({"name": f"baseline_r{rank}", "mode": "baseline", "rank": rank})
        variants.append({"name": f"hfp_stiff_r{rank}", "mode": "hfp_stiff", "rank": rank})
    return variants


def run_training(variants: list[dict], settings: dict, quick: bool, model: str, train_data: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for i, v in enumerate(variants, 1):
        out_dir = f"./adapters/ablation/{v['name']}"
        print(f"\n{'='*70}")
        print(f"[{i}/{len(variants)}] Eğitim: {v['name']}")
        print(f"{'='*70}")

        train_lora(
            mode=v["mode"],
            model_id=model,
            train_path=train_data,
            output_dir=out_dir,
            rank=v["rank"],
            settings=settings,
            quick=quick,
            device=device,
        )

        meta_path = Path(out_dir) / "train_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        results.append({"variant": v["name"], **v, **meta})

    return results


def run_benchmark(variants: list[dict], model: str, prompts: str, output: str, quick: bool):
    adapter_map = {"base": None}
    for v in variants:
        adapter_map[v["name"]] = f"./adapters/ablation/{v['name']}"

    modes = ["base"] + [v["name"] for v in variants]
    cmd = [
        sys.executable, "hf_lora_benchmark.py",
        "--model", model,
        "--prompts", prompts,
        "--output", output,
        "--max-tokens", "32",
        "--modes", *modes,
    ]
    for name, path in adapter_map.items():
        if name == "base":
            continue
        cmd.extend(["--adapter", f"{name}={path}"])

    if quick:
        cmd.append("--quick")

    print(f"\n{'='*70}")
    print("Benchmark başlıyor...")
    print(f"{'='*70}")
    subprocess.run(cmd, check=True)


def print_ablation_summary(train_results: list[dict], bench_path: Path):
    if not bench_path.exists():
        return

    bench = json.loads(bench_path.read_text())
    summaries = {s["mode"]: s for s in bench["summaries"]}
    train_by_name = {t["variant"]: t for t in train_results}

    print("\n" + "=" * 95)
    print(f"{'Varyant':<22} {'Rank':>5} {'Epoch':>6} {'ValLoss':>8} {'Train s':>8} {'Acc':>7} {'Tok/s':>7}")
    print("-" * 95)

    for v in train_results:
        name = v["variant"]
        s = summaries.get(name, {})
        acc = s.get("accuracy_pct", 0)
        tps = s.get("avg_tokens_per_sec", 0)
        print(
            f"{name:<22} {v.get('lora_rank', '?'):>5} "
            f"{v.get('epochs_run', '?'):>6} {v.get('best_val_loss', 0):>8.4f} "
            f"{v.get('elapsed_sec', 0):>8.1f} {acc:>6.1f}% {tps:>7.1f}"
        )

    base = summaries.get("base", {})
    print("-" * 95)
    if base:
        print(f"{'base (no adapter)':<22} {'—':>5} {'—':>6} {'—':>8} {'—':>8} {base.get('accuracy_pct',0):>6.1f}% {base.get('avg_tokens_per_sec',0):>7.1f}")

    # En iyi HFP vs en iyi baseline
    bl = [s for k, s in summaries.items() if k.startswith("baseline_")]
    hfp = [s for k, s in summaries.items() if k.startswith("hfp_stiff_")]
    if bl and hfp:
        best_bl = max(bl, key=lambda x: (x.get("accuracy", 0), -x.get("avg_latency_sec", 999)))
        best_hfp = max(hfp, key=lambda x: (x.get("accuracy", 0), -x.get("avg_latency_sec", 999)))
        print(f"\n  En iyi baseline : {best_bl['mode']} → {best_bl['accuracy_pct']}%")
        print(f"  En iyi HFP stiff: {best_hfp['mode']} → {best_hfp['accuracy_pct']}%")
        diff = best_hfp["accuracy"] - best_bl["accuracy"]
        print(f"  HFP kazancı      : {diff*100:+.1f} pp")

        bl_train = train_by_name.get(best_bl["mode"], {})
        hfp_train = train_by_name.get(best_hfp["mode"], {})
        if bl_train and hfp_train:
            time_ratio = hfp_train.get("elapsed_sec", 1) / max(bl_train.get("elapsed_sec", 1), 0.01)
            print(f"  Eğitim süresi    : HFP/baseline = {time_ratio:.2f}x")

    print("=" * 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-data", default="train_data.jsonl")
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--config", default="hfp_lora_config.json")
    parser.add_argument("--output", default="lora_ablation_results.json")
    args = parser.parse_args()

    settings = load_lora_settings(args.config)
    variants = build_variants(settings, args.quick)

    train_results = []
    if not args.bench_only:
        train_results = run_training(variants, settings, args.quick, args.model, args.train_data)
        with open("lora_ablation_train.json", "w") as f:
            json.dump(train_results, f, indent=2)

    if not args.train_only:
        run_benchmark(variants, args.model, args.prompts, args.output, args.quick)

    if train_results:
        print_ablation_summary(train_results, Path(args.output))
    elif Path(args.output).exists():
        print_ablation_summary([], Path(args.output))

    print(f"\nSonuçlar: {args.output}")


if __name__ == "__main__":
    main()
