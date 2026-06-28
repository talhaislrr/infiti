"""
Birleşik Karşılaştırma — Ollama vs HF (base + fine-tune)
========================================================
Aynı 20 prompt, aynı skorlama, yan yana rapor.

Modeller:
  ollama:qwen2.5:0.5b     — Ollama quantize (Q4)
  hf:base                 — HF ham model
  hf:baseline_r64         — En iyi LoRA
  hf:hfp_stiff_r32        — En iyi HFP (opsiyonel)

Kullanım:
  python3 unified_compare.py
  python3 unified_compare.py --skip-ollama
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch

from hf_lora_benchmark import generate_one, load_model
from ollama_benchmark import (
    DEFAULT_OLLAMA,
    check_ollama,
    load_prompts,
    ollama_post,
    score_response,
)


DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BASELINE = "./adapters/ablation/baseline_r64"
DEFAULT_HFP = "./adapters/ablation/hfp_stiff_r32"


@dataclass
class UnifiedRecord:
    backend: str  # ollama | hf
    label: str
    prompt_id: str
    response: str
    correct: bool
    output_tokens: int
    latency_sec: float
    tokens_per_sec: float


def ollama_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }
    t0 = time.perf_counter()
    out = ollama_post(base_url, "/api/chat", payload)
    wall = time.perf_counter() - t0

    msg = out.get("message", {})
    response = msg.get("content", "")
    out_tok = out.get("eval_count", 0)
    eval_ns = out.get("eval_duration", 0)
    tps = (out_tok / (eval_ns / 1e9)) if eval_ns > 0 else 0.0

    return {
        "response": response.strip().split("\n")[0].strip(),
        "output_tokens": out_tok,
        "latency_sec": round(wall, 3),
        "tokens_per_sec": round(tps, 2),
    }


def run_ollama_model(
    label: str,
    model: str,
    prompts,
    base_url: str,
    max_tokens: int,
) -> list[UnifiedRecord]:
    print(f"\n▶ {label} (Ollama /api/chat)")
    records = []
    for i, item in enumerate(prompts, 1):
        try:
            gen = ollama_chat(base_url, model, item.prompt, max_tokens)
        except (urllib.error.HTTPError, RuntimeError) as e:
            print(f"  [{i}] {item.id} HATA: {e}")
            continue
        correct, _ = score_response(item, gen["response"])
        rec = UnifiedRecord(
            backend="ollama",
            label=label,
            prompt_id=item.id,
            response=gen["response"][:200],
            correct=correct,
            output_tokens=gen["output_tokens"],
            latency_sec=gen["latency_sec"],
            tokens_per_sec=gen["tokens_per_sec"],
        )
        records.append(rec)
        mark = "✓" if correct else "✗"
        print(f"  [{i}/{len(prompts)}] {item.id} {mark} «{gen['response'][:40]}»")
    return records


def run_hf_model(
    label: str,
    model_id: str,
    adapter_dir: str | None,
    prompts,
    device: torch.device,
    max_tokens: int,
) -> list[UnifiedRecord]:
    mode = "base" if adapter_dir is None else label
    print(f"\n▶ {label} (HF+PEFT)")
    model, tokenizer = load_model(mode, model_id, adapter_dir, device)
    records = []

    for i, item in enumerate(prompts, 1):
        response, n_out, elapsed, tps = generate_one(
            model, tokenizer, item.prompt, device, max_tokens
        )
        correct, _ = score_response(item, response)
        rec = UnifiedRecord(
            backend="hf",
            label=label,
            prompt_id=item.id,
            response=response[:200],
            correct=correct,
            output_tokens=n_out,
            latency_sec=round(elapsed, 3),
            tokens_per_sec=round(tps, 2),
        )
        records.append(rec)
        mark = "✓" if correct else "✗"
        print(f"  [{i}/{len(prompts)}] {item.id} {mark} «{response[:40]}»")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def aggregate(records: list[UnifiedRecord], label: str) -> dict:
    subset = [r for r in records if r.label == label]
    if not subset:
        return {"label": label, "n": 0}
    acc = mean(1.0 if r.correct else 0.0 for r in subset)
    return {
        "label": label,
        "backend": subset[0].backend,
        "n": len(subset),
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "avg_output_tokens": round(mean(r.output_tokens for r in subset), 2),
        "avg_tokens_per_sec": round(mean(r.tokens_per_sec for r in subset), 2),
        "avg_latency_sec": round(mean(r.latency_sec for r in subset), 3),
    }


def print_table(summaries: list[dict]):
    print("\n" + "=" * 92)
    print(f"{'Model':<28} {'Backend':<8} {'Acc':>7} {'OutTok':>7} {'Tok/s':>8} {'Latency':>9}")
    print("-" * 92)
    for s in summaries:
        if s["n"] == 0:
            continue
        print(
            f"{s['label']:<28} {s['backend']:<8} {s['accuracy_pct']:>6.1f}% "
            f"{s['avg_output_tokens']:>7.1f} {s['avg_tokens_per_sec']:>8.1f} "
            f"{s['avg_latency_sec']:>8.2f}s"
        )
    print("=" * 92)
    print("\nNot: Ollama=Q4 quantize | HF=fp32 CPU | Aynı prompt metni, HF chat template uygular.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--output", default="unified_compare_results.json")
    parser.add_argument("--ollama-model", default="qwen2.5:0.5b")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--baseline-adapter", default=DEFAULT_BASELINE)
    parser.add_argument("--hfp-adapter", default=DEFAULT_HFP)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-hfp", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("BİRLEŞİK KARŞILAŞTIRMA — Ollama vs HF")
    print(f"Prompt: {len(prompts)} | max_tokens={args.max_tokens} | device={device}")
    print("=" * 70)

    all_records: list[UnifiedRecord] = []
    labels: list[str] = []

    if not args.skip_ollama:
        if not check_ollama(args.ollama_url):
            print("UYARI: Ollama yok, atlanıyor.")
        else:
            label = f"ollama:{args.ollama_model}"
            labels.append(label)
            all_records.extend(
                run_ollama_model(label, args.ollama_model, prompts, args.ollama_url, args.max_tokens)
            )

    hf_runs = [
        ("hf:base", None),
        ("hf:baseline_r64", args.baseline_adapter),
    ]
    if not args.skip_hfp:
        hf_runs.append(("hf:hfp_stiff_r32", args.hfp_adapter))

    for label, adapter in hf_runs:
        if adapter and not Path(adapter).exists():
            print(f"UYARI: {adapter} yok, {label} atlanıyor")
            continue
        labels.append(label)
        all_records.extend(
            run_hf_model(label, args.hf_model, adapter, prompts, device, args.max_tokens)
        )

    summaries = [aggregate(all_records, lb) for lb in labels]
    print_table(summaries)

    out = {
        "meta": {
            "n_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "scoring": "numeric_first_token",
            "caveat": "Ollama Q4 quantize; HF fp32; farklı runtime",
        },
        "summaries": summaries,
        "records": [asdict(r) for r in all_records],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
