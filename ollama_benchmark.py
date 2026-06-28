"""
Ollama LLM Benchmark — Baseline vs HFP model karşılaştırması
==============================================================
Aynı prompt seti ile doğruluk, token kullanımı, hız ölçer.

Kullanım:
  ollama pull qwen2.5:0.5b
  python3 hfp_lora_finetune.py --mode hfp --quick    # HFP adapter üret
  python3 export_ollama.py --name hfp-qwen
  python3 ollama_benchmark.py --models qwen2.5:0.5b hfp-qwen
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_OLLAMA = "http://localhost:11434"


@dataclass
class PromptItem:
    id: str
    prompt: str
    gold: str
    match: str  # numeric | contains | exact


@dataclass
class RunRecord:
    model: str
    prompt_id: str
    response: str
    correct: bool
    score: float
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    eval_duration_ns: int
    total_duration_ns: int
    tokens_per_sec: float
    latency_sec: float


def load_prompts(path: str) -> list[PromptItem]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(PromptItem(d["id"], d["prompt"], d["gold"], d.get("match", "contains")))
    return items


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))


def score_response(item: PromptItem, response: str) -> tuple[bool, float]:
    resp = normalize(response)
    gold = normalize(item.gold)

    if item.match == "exact":
        ok = resp == gold
        return ok, 1.0 if ok else 0.0

    if item.match == "numeric":
        nums = extract_numbers(response)
        gold_nums = extract_numbers(item.gold)
        if not gold_nums:
            ok = gold in resp
            return ok, 1.0 if ok else 0.0
        if not nums:
            return False, 0.0
        # İlk sayı cevap kabul edilir (soru metnindeki sayılar yanlış pozitif üretmesin)
        ok = nums[0] == gold_nums[0]
        return ok, 1.0 if ok else 0.0

    # contains
    ok = gold in resp
    return ok, 1.0 if ok else 0.0


def ollama_get(base_url: str, endpoint: str) -> dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def ollama_post(base_url: str, endpoint: str, payload: dict) -> dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def check_ollama(base_url: str) -> bool:
    try:
        ollama_get(base_url, "/api/tags")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 405:
            return False
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def list_models(base_url: str) -> list[str]:
    data = ollama_get(base_url, "/api/tags")
    return [m["name"] for m in data.get("models", [])]


def generate(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    t0 = time.perf_counter()
    try:
        out = ollama_post(base_url, "/api/generate", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"Ollama hata ({model}): {body}") from e
    wall = time.perf_counter() - t0

    prompt_tok = out.get("prompt_eval_count", 0)
    out_tok = out.get("eval_count", 0)
    eval_ns = out.get("eval_duration", 0)
    tps = (out_tok / (eval_ns / 1e9)) if eval_ns > 0 else 0.0

    return {
        "response": out.get("response", ""),
        "prompt_tokens": prompt_tok,
        "output_tokens": out_tok,
        "total_tokens": prompt_tok + out_tok,
        "eval_duration_ns": eval_ns,
        "total_duration_ns": out.get("total_duration", 0),
        "tokens_per_sec": round(tps, 2),
        "latency_sec": round(wall, 3),
    }


def run_benchmark(
    models: list[str],
    prompts: list[PromptItem],
    base_url: str,
    max_tokens: int,
    temperature: float,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for model in models:
        print(f"\n▶ Model: {model}")
        for i, item in enumerate(prompts, 1):
            try:
                gen = generate(base_url, model, item.prompt, max_tokens, temperature)
            except RuntimeError as e:
                print(f"  [{i}/{len(prompts)}] {item.id} HATA: {e}")
                continue

            correct, score = score_response(item, gen["response"])
            rec = RunRecord(
                model=model,
                prompt_id=item.id,
                response=gen["response"][:500],
                correct=correct,
                score=score,
                prompt_tokens=gen["prompt_tokens"],
                output_tokens=gen["output_tokens"],
                total_tokens=gen["total_tokens"],
                eval_duration_ns=gen["eval_duration_ns"],
                total_duration_ns=gen["total_duration_ns"],
                tokens_per_sec=gen["tokens_per_sec"],
                latency_sec=gen["latency_sec"],
            )
            records.append(rec)
            mark = "✓" if correct else "✗"
            print(
                f"  [{i}/{len(prompts)}] {item.id} {mark} "
                f"out={gen['output_tokens']}tok {gen['tokens_per_sec']:.1f}tok/s"
            )
    return records


def aggregate(records: list[RunRecord], model: str) -> dict:
    subset = [r for r in records if r.model == model]
    if not subset:
        return {"model": model, "n": 0}

    acc = sum(r.correct for r in subset) / len(subset)
    return {
        "model": model,
        "n": len(subset),
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "avg_output_tokens": round(statistics.mean(r.output_tokens for r in subset), 2),
        "avg_total_tokens": round(statistics.mean(r.total_tokens for r in subset), 2),
        "avg_tokens_per_sec": round(statistics.mean(r.tokens_per_sec for r in subset), 2),
        "avg_latency_sec": round(statistics.mean(r.latency_sec for r in subset), 3),
        "total_output_tokens": sum(r.output_tokens for r in subset),
    }


def print_summary(summaries: list[dict]):
    if not summaries or summaries[0]["n"] == 0:
        print("Sonuç yok.")
        return

    baseline = summaries[0]
    print("\n" + "=" * 88)
    print(f"{'Model':<25} {'Acc':>8} {'OutTok':>8} {'Tok/s':>8} {'Latency':>10}")
    print("-" * 88)
    for s in summaries:
        print(
            f"{s['model']:<25} {s['accuracy_pct']:>7.1f}% "
            f"{s['avg_output_tokens']:>8.1f} {s['avg_tokens_per_sec']:>8.1f} "
            f"{s['avg_latency_sec']:>9.2f}s"
        )
    print("=" * 88)

    if len(summaries) >= 2:
        hfp = summaries[1]
        print("\n── HFP vs Baseline ──")
        acc_d = (hfp["accuracy"] - baseline["accuracy"]) * 100
        tok_d = hfp["avg_output_tokens"] - baseline["avg_output_tokens"]
        spd_r = hfp["avg_tokens_per_sec"] / max(baseline["avg_tokens_per_sec"], 0.01)
        print(f"  Doğruluk farkı:     {acc_d:+.1f} pp")
        print(f"  Ort. çıktı token:   {tok_d:+.1f}")
        print(f"  Hız oranı (tok/s):  {spd_r:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Ollama LLM benchmark")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:0.5b", "hfp-qwen"])
    parser.add_argument("--prompts", default="prompts.jsonl")
    parser.add_argument("--output", default="ollama_results.json")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if not Path(args.prompts).exists():
        raise SystemExit(f"Prompt dosyası bulunamadı: {args.prompts}")

    if not check_ollama(args.ollama_url):
        raise SystemExit(
            f"Ollama API'ye ulaşılamıyor ({args.ollama_url}).\n"
            f"  Ollama uygulamasını açın veya: ollama list\n"
            f"  Port meşgulse zaten çalışıyordur — 'ollama serve' tekrar çalıştırmayın."
        )

    available = list_models(args.ollama_url)
    print(f"Yüklü modeller: {', '.join(available[:5])}{'...' if len(available) > 5 else ''}")
    for m in args.models:
        if not any(m == a or a.startswith(m + ":") or m.startswith(a.split(":")[0]) for a in available):
            print(f"UYARI: '{m}' yüklü modellerde yok. ollama pull {m}")

    prompts = load_prompts(args.prompts)
    print("=" * 70)
    print("OLLAMA LLM BENCHMARK")
    print(f"Modeller: {args.models}")
    print(f"Prompt sayısı: {len(prompts)} | max_tokens={args.max_tokens}")
    print("=" * 70)

    records = run_benchmark(
        args.models, prompts, args.ollama_url, args.max_tokens, args.temperature
    )
    summaries = [aggregate(records, m) for m in args.models]

    print_summary(summaries)

    out = {
        "meta": {
            "models": args.models,
            "n_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "summaries": summaries,
        "records": [asdict(r) for r in records],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi: {args.output}")


if __name__ == "__main__":
    main()
