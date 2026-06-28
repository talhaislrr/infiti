#!/usr/bin/env python3
"""
Uzun menzil recall demo — Faz 2b
================================
Sentetik kitap metninde başa gömülü gerçeği, prompt sonunda sorar.
Base KV vs Bulk KV+State karşılaştırması.

Kullanım:
  python3 bulk_longrange_demo.py --device mps
  python3 bulk_longrange_demo.py --doc-tokens 2048 --gap-tokens 1500
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from bulk_device import device_summary, pick_device, pick_dtype
from bulk_hybrid import BaseKVGenerator, TinyLlamaWithBulk, create_hybrid, create_kvfree, load_tinyllama
from bulk_recall_data import build_recall_document, FILLER_PARAGRAPH, MAX_PROMPT_TOKENS
from bulk_layer_swap import TinyLlamaKVFreeTail, load_bulk_swap_checkpoint


def build_synthetic_book(
    fact_key: str,
    fact_value: str,
    filler_paragraph: str,
    target_tokens: int,
    tokenizer,
    max_len: int = MAX_PROMPT_TOKENS,
) -> str:
    return build_recall_document(
        fact_key, fact_value, filler_paragraph, target_tokens, tokenizer,
        max_len=max_len, include_answer=False,
    )


def extract_answer(text: str) -> str:
    m = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip()[:200] if m else text.strip()[:200]


def score_answer(answer: str, expected: str) -> dict:
    ans = answer.lower()
    exp = expected.lower()
    hit = exp in ans or any(w in ans for w in exp.split() if len(w) > 3)
    return {"expected": expected, "answer": answer, "hit": hit}


@torch.no_grad()
def generate_answer(gen, tokenizer, prompt: str, max_new: int, device) -> str:
    ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS,
    )["input_ids"].to(device)

    if isinstance(gen, (TinyLlamaWithBulk, TinyLlamaKVFreeTail)):
        new_ids = gen.generate_cached(ids, max_new_tokens=max_new)
    elif isinstance(gen, BaseKVGenerator):
        new_ids = gen.generate(ids, max_new)
    elif hasattr(gen, "generate_cached"):
        new_ids = gen.generate_cached(ids, max_new_tokens=max_new)
    else:
        raise TypeError(f"Bilinmeyen generator: {type(gen)}")

    text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    if not text:
        return ""
    return extract_answer("Answer: " + text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fact-key", default="secret project codename")
    parser.add_argument("--fact-value", default="Nebula Seven")
    parser.add_argument("--doc-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt", type=int, default=2048)
    parser.add_argument("--gap-tokens", type=int, default=512, help="Dolgu minimum (yaklaşık)")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--adapter", default="checkpoints/bulk_adapter/adapter.pt")
    parser.add_argument("--kvfree", action="store_true", help="Faz 3 KVFree tail modu")
    parser.add_argument("--swap-checkpoint", default="checkpoints/bulk_swap/bulk_v2.pt")
    parser.add_argument("--swap-layers", type=int, default=4)
    parser.add_argument("--sliding-window", type=int, default=512,
                        help="Erken KV penceresi; 0= kırpma yok (recall test)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="bulk_longrange_demo_results.json")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)
    print("=" * 70)
    print(f"BulkTrigger Uzun Menzil Demo | {device_summary(device)}")
    print("=" * 70)

    filler = FILLER_PARAGRAPH * 3

    base, tokenizer, model_path = load_tinyllama(device, dtype)
    prompt = build_synthetic_book(
        args.fact_key, args.fact_value, filler,
        max(args.doc_tokens, args.gap_tokens + 64), tokenizer,
        max_len=args.max_prompt,
    )
    n_tok = len(tokenizer.encode(prompt, truncation=True, max_length=args.max_prompt))
    print(f"\nPrompt: {n_tok} token (max {args.max_prompt}) | fact={args.fact_value!r}")

    adapter_path = Path(args.adapter)
    adapter_kw = dict(k_short=8, trigger_stride=4, adaptive_trigger=False)

    if args.kvfree:
        base_clean, _, _ = load_tinyllama(device, dtype)
        base_gen = BaseKVGenerator(base_clean)
        base_kv, _, _ = load_tinyllama(device, dtype)
        bulk_model = TinyLlamaKVFreeTail(
            base_kv, n_swap_layers=args.swap_layers, sliding_window=args.sliding_window,
            compact=True, d_bulk=256,
        ).to(device)
        if Path(args.swap_checkpoint).exists():
            load_bulk_swap_checkpoint(bulk_model, args.swap_checkpoint, device)
            print(f"  KVFree checkpoint: {args.swap_checkpoint}")
    elif adapter_path.exists():
        base_gen = BaseKVGenerator(base)
        bulk_model, _, _ = create_hybrid(device, dtype, adapter_path, base_model=base, **adapter_kw)
    else:
        base_gen = BaseKVGenerator(base)
        bulk_model = TinyLlamaWithBulk(base, freeze_base=True, **adapter_kw).to(device)

    print("\n[Base KV-cache]")
    base_ans = generate_answer(base_gen, tokenizer, prompt, args.max_new, device)
    base_score = score_answer(base_ans, args.fact_value)
    print(f"  → {base_ans!r}  hit={base_score['hit']}")

    label = "KVFree tail" if args.kvfree else "Bulk KV+State"
    print(f"\n[{label}]")
    bulk_ans = generate_answer(bulk_model, tokenizer, prompt, args.max_new, device)
    bulk_score = score_answer(bulk_ans, args.fact_value)
    print(f"  → {bulk_ans!r}  hit={bulk_score['hit']}")

    results = {
        "device": str(device),
        "model": model_path,
        "mode": "kvfree" if args.kvfree else "hybrid",
        "prompt_tokens": n_tok,
        "fact_key": args.fact_key,
        "fact_value": args.fact_value,
        "base": base_score,
        "bulk": bulk_score,
        "bulk_wins": bulk_score["hit"] and not base_score["hit"],
    }
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nKaydedildi: {args.output}")
    if results["bulk_wins"]:
        print("✓ Bulk, base'e göre daha iyi recall")
    elif bulk_score["hit"] == base_score["hit"]:
        print("= Her iki yöntem aynı sonuç")
    else:
        print("✗ Base daha iyi (adapter eğitimi gerekebilir)")


if __name__ == "__main__":
    main()
