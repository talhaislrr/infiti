"""
HuggingFace LoRA Benchmark — Base vs Baseline vs HFP
=====================================================
Ollama ADAPTER, Qwen2.5 safetensors'ı henüz desteklemiyor.
Bu script aynı prompt setiyle doğrudan HF+PEFT üzerinde karşılaştırır.

Kullanım:
  python3 hf_lora_benchmark.py
  python3 hf_lora_benchmark.py --adapter baseline_r64=./adapters/ablation/baseline_r64
  python3 hf_lora_benchmark.py --modes base baseline_r64 hfp_stiff_r64
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ollama_benchmark import load_prompts, score_response


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class HFRunRecord:
    mode: str
    prompt_id: str
    response: str
    correct: bool
    output_tokens: int
    latency_sec: float
    tokens_per_sec: float


def parse_adapter_args(adapter_args: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in adapter_args:
        if "=" not in item:
            raise ValueError(f"Adapter formatı: name=path (aldı: {item})")
        name, path = item.split("=", 1)
        mapping[name.strip()] = path.strip()
    return mapping


def load_model(mode: str, model_id: str, adapter_dir: str | None, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if device.type == "cuda":
        kwargs["device_map"] = "auto"

    if mode == "base":
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device.type == "cpu":
            model = model.to(device)
        return model, tokenizer

    if adapter_dir is None or not Path(adapter_dir).exists():
        raise FileNotFoundError(f"Adapter bulunamadı: {adapter_dir}")

    base = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir)
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _eos_token_ids(tokenizer) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    for tok in ("", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, int, float, float]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    n_in = inputs.input_ids.shape[1]
    eos_ids = _eos_token_ids(tokenizer)

    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
        repetition_penalty=1.1,
    )
    elapsed = time.perf_counter() - t0

    n_out = out.shape[1] - n_in
    response = tokenizer.decode(out[0, n_in:], skip_special_tokens=True)
    # İlk satır / cümle — uzun tekrarları kes
    response = response.split("\n")[0].strip()
    tps = n_out / elapsed if elapsed > 0 else 0.0
    return response, n_out, elapsed, tps


def run_mode(
    mode: str,
    prompts,
    model_id: str,
    adapter_dir: str | None,
    device: torch.device,
    max_new_tokens: int,
) -> list[HFRunRecord]:
    print(f"\n▶ Mod: {mode}")
    model, tokenizer = load_model(mode, model_id, adapter_dir, device)
    records: list[HFRunRecord] = []

    for i, item in enumerate(prompts, 1):
        response, n_out, elapsed, tps = generate_one(
            model, tokenizer, item.prompt, device, max_new_tokens
        )
        correct, _ = score_response(item, response)
        rec = HFRunRecord(
            mode=mode,
            prompt_id=item.id,
            response=response[:300],
            correct=correct,
            output_tokens=n_out,
            latency_sec=round(elapsed, 3),
            tokens_per_sec=round(tps, 2),
        )
        records.append(rec)
        mark = "✓" if correct else "✗"
        preview = response[:40].replace("\n", " ")
        print(f"  [{i}/{len(prompts)}] {item.id} {mark} «{preview}» {n_out}tok {tps:.1f}tok/s")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def aggregate(records: list[HFRunRecord], mode: str) -> dict:
    subset = [r for r in records if r.mode == mode]
    if not subset:
        return {"mode": mode, "n": 0}
    acc = mean(1.0 if r.correct else 0.0 for r in subset)
    return {
        "mode": mode,
        "n": len(subset),
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "avg_output_tokens": round(mean(r.output_tokens for r in subset), 2),
        "avg_tokens_per_sec": round(mean(r.tokens_per_sec for r in subset), 2),
        "avg_latency_sec": round(mean(r.latency_sec for r in subset), 3),
    }


def print_summary(summaries: list[dict]):
    print("\n" + "=" * 80)
    print(f"{'Mod':<22} {'Acc':>8} {'OutTok':>8} {'Tok/s':>8} {'Latency':>10}")
    print("-" * 80)
    for s in summaries:
        if s["n"] == 0:
            continue
        print(
            f"{s['mode']:<22} {s['accuracy_pct']:>7.1f}% "
            f"{s['avg_output_tokens']:>8.1f} {s['avg_tokens_per_sec']:>8.1f} "
            f"{s['avg_latency_sec']:>9.2f}s"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["base", "baseline", "hfp"])
    parser.add_argument("--adapter", action="append", default=[], help="name=path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--baseline-adapter", default="./adapters/baseline")
    parser.add_argument("--hfp-adapter", default="./adapters/hfp")
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--output", default="hf_lora_results.json")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if args.quick:
        prompts = prompts[:5]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter_map: dict[str, str | None] = {"base": None}
    adapter_map.update(parse_adapter_args(args.adapter))

    # Geriye uyumluluk
    legacy = {"baseline": args.baseline_adapter, "hfp": args.hfp_adapter}
    for mode in args.modes:
        if mode in legacy and mode not in adapter_map:
            adapter_map[mode] = legacy[mode]

    print("=" * 70)
    print("HF LoRA BENCHMARK")
    print(f"Model: {args.model} | Cihaz: {device} | Prompt: {len(prompts)}")
    print("=" * 70)

    all_records: list[HFRunRecord] = []
    for mode in args.modes:
        path = adapter_map.get(mode)
        if mode != "base" and path is None:
            print(f"UYARI: '{mode}' için adapter yok, atlanıyor")
            continue
        if mode != "base" and not Path(path).exists():
            print(f"UYARI: Adapter bulunamadı ({path}), atlanıyor")
            continue
        all_records.extend(
            run_mode(mode, prompts, args.model, path, device, args.max_tokens)
        )

    summaries = [aggregate(all_records, m) for m in args.modes if any(r.mode == m for r in all_records)]
    print_summary(summaries)

    out = {
        "meta": {"model": args.model, "device": str(device), "modes": args.modes, "scoring": "numeric_first_token"},
        "summaries": summaries,
        "records": [asdict(r) for r in all_records],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
