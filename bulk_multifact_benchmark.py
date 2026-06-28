#!/usr/bin/env python3
"""
BulkState Multi-Fact Recall Benchmark
======================================
n_facts × doc_len grid → her kombinasyonda hit_rate ölçülür.

Kullanım:
  python3 bulk_multifact_benchmark.py --device mps
  python3 bulk_multifact_benchmark.py --device mps --n-facts 1 2 5 --doc-lens 512 2048
  python3 bulk_multifact_benchmark.py --quick --device mps
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from bulk_device import pick_device, pick_dtype
from bulk_hybrid import load_tinyllama
from bulk_layer_swap import TinyLlamaKVFreeTail, load_bulk_swap_checkpoint
from bulk_recall_data import (
    FACT_KEYS,
    FACT_VALUES,
    FILLER_PARAGRAPH,
    build_multifact_document,
)


def _build_model(device, dtype, checkpoint: str | None) -> tuple[TinyLlamaKVFreeTail, object]:
    base, tokenizer, _ = load_tinyllama(device, dtype)
    model = TinyLlamaKVFreeTail(
        base, n_swap_layers=4, sliding_window=512, compact=True, d_bulk=256,
    ).to(device)
    ckpt = checkpoint or "checkpoints/bulk_swap/bulk_v2.pt"
    if Path(ckpt).exists():
        load_bulk_swap_checkpoint(model, ckpt, device)
        print(f"  checkpoint: {ckpt}")
    else:
        print(f"  UYARI: checkpoint yok ({ckpt}) — rastgele ağırlıklar!")
    model.eval()
    return model, tokenizer


def _build_single_question_prompt(
    keys: list[str],
    values: list[str],
    filler: str,
    doc_len: int,
    tokenizer,
    query_key: str,
) -> str:
    """N fact'i başa yaz, dolguyu ekle, sona sadece 1 soru koy."""
    from bulk_recall_data import MAX_PROMPT_TOKENS

    target = min(doc_len, MAX_PROMPT_TOKENS - 32)

    header_lines = ["=== CONFIDENTIAL DOSSIER ==="]
    for k, v in zip(keys, values):
        header_lines.append(f"The {k} is {v}. This fact must be remembered.")
        header_lines.append(f"Reference code: {v.upper().replace(' ', '-')}")
    header = "\n".join(header_lines) + "\n\n"

    question = f"\nQuestion: What is the {query_key}?\nAnswer:"

    body = ""
    para = filler.strip()
    draft = header + question
    while len(tokenizer.encode(draft)) < target:
        body += para + "\n\n"
        draft = header + body + question

    ids = tokenizer.encode(draft)
    if len(ids) <= MAX_PROMPT_TOKENS:
        return draft

    header_ids = tokenizer.encode(header)
    tail_ids = tokenizer.encode(question)
    budget = MAX_PROMPT_TOKENS - len(header_ids) - len(tail_ids)
    body_ids = tokenizer.encode(body)
    if len(body_ids) > budget:
        body_ids = body_ids[:budget]
    return tokenizer.decode(header_ids + body_ids + tail_ids)


@torch.inference_mode()
def eval_multifact(
    model: TinyLlamaKVFreeTail,
    tokenizer,
    device: torch.device,
    n_facts: int,
    doc_len: int,
    n_cases: int,
    sliding_window: int,
    seed: int = 0,
) -> dict:
    """Tek (n_facts, doc_len) kombinasyonu için hit oranı.

    Her fact ayrı ayrı sorgulanır — tek generate'te tüm soruları
    cevaplamaya zorlamaz, gerçek recall kapasitesini ölçer.
    """
    rng = random.Random(seed)
    total_questions = 0
    total_hits = 0
    cases = []

    old_sw = model.sliding_window
    old_pb = getattr(model, "pure_bulk", False)
    model.sliding_window = sliding_window

    for _ in range(n_cases):
        keys = rng.sample(FACT_KEYS * ((n_facts // len(FACT_KEYS)) + 2), n_facts)
        values = [FACT_VALUES[rng.randint(0, len(FACT_VALUES) - 1)] for _ in range(n_facts)]

        hits_this = 0
        answers = []
        for qi, (qk, qv) in enumerate(zip(keys, values)):
            prompt = _build_single_question_prompt(
                keys, values, FILLER_PARAGRAPH, doc_len, tokenizer, query_key=qk,
            )
            ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048,
            )["input_ids"].to(device)

            new_ids = model.generate_cached(ids, max_new_tokens=16)
            generated = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip().lower()
            hit = qv.lower() in generated
            hits_this += int(hit)
            answers.append({"key": qk, "expected": qv, "hit": hit, "tail": generated[-80:]})

        total_hits += hits_this
        total_questions += n_facts
        cases.append({"values": values, "hits": hits_this, "answers": answers})

    model.sliding_window = old_sw
    model.pure_bulk = old_pb
    hit_rate = total_hits / max(total_questions, 1)
    return {
        "n_facts": n_facts,
        "doc_len": doc_len,
        "n_cases": n_cases,
        "total_questions": total_questions,
        "total_hits": total_hits,
        "hit_rate": round(hit_rate, 4),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-facts", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--doc-lens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--n-cases", type=int, default=8)
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--pure-bulk", action="store_true",
                        help="Decode sırasında erken KV tamamen kapat — saf BulkState testi")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--quick", action="store_true", help="n_facts=[1,2], doc_lens=[512], n_cases=4")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_multifact_benchmark_results.json")
    args = parser.parse_args()

    if args.quick:
        args.n_facts = [1, 2]
        args.doc_lens = [512]
        args.n_cases = 4

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)

    print("=" * 65)
    print(f"Multi-Fact Recall Benchmark | {device}")
    print(f"  n_facts={args.n_facts}  doc_lens={args.doc_lens}  n_cases={args.n_cases}")
    print("=" * 65)

    model, tokenizer = _build_model(device, dtype, args.checkpoint)

    if args.pure_bulk:
        model.pure_bulk = True
        print("  MOD: pure_bulk — erken KV devre dışı, saf BulkState testi")

    results = []
    t0 = time.time()

    for nf in args.n_facts:
        for dl in args.doc_lens:
            print(f"\nn_facts={nf}  doc_len={dl} ...", end=" ", flush=True)
            r = eval_multifact(
                model, tokenizer, device,
                n_facts=nf, doc_len=dl,
                n_cases=args.n_cases,
                sliding_window=args.sliding_window,
            )
            print(f"hit_rate={r['hit_rate']:.2%}  ({r['total_hits']}/{r['total_questions']})")
            results.append(r)

    elapsed = time.time() - t0

    # Özet tablo
    print("\n" + "=" * 65)
    print(f"{'n_facts':>8}  {'doc_len':>8}  {'hit_rate':>10}  hits/total")
    print("-" * 65)
    for r in results:
        print(
            f"{r['n_facts']:>8}  {r['doc_len']:>8}  {r['hit_rate']:>9.1%}  "
            f"{r['total_hits']}/{r['total_questions']}"
        )
    print(f"\nToplam süre: {elapsed:.1f}s")

    payload = {
        "device": str(device),
        "sliding_window": args.sliding_window,
        "pure_bulk": args.pure_bulk,
        "elapsed_sec": round(elapsed, 1),
        "results": results,
    }
    # cases çok büyük olabilir — özet için kaldır
    for r in payload["results"]:
        r.pop("cases", None)

    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
