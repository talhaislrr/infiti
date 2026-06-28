#!/usr/bin/env python3
"""
BulkTrigger QA Demo CLI — Faz 3
================================
Metin veya PDF dosyasından bağlam yükler, soru sorar.

Kullanım:
  python3 bulk_demo.py --file book.txt --question "Ana karakter kim?"
  python3 bulk_demo.py --file report.pdf --question "Özet nedir?" --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bulk_device import pick_device, pick_dtype
from bulk_hybrid import TinyLlamaWithBulk, create_hybrid, load_tinyllama


def load_document(path: Path, max_chars: int = 120_000) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise SystemExit(
                "PDF için: pip install pypdf\n"
                "veya metni .txt olarak kaydedin."
            ) from e
        reader = PdfReader(str(path))
        parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(parts)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars].strip()


def build_qa_prompt(context: str, question: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def main():
    parser = argparse.ArgumentParser(description="BulkTrigger belge QA demo")
    parser.add_argument("--file", required=True, help="Metin veya PDF dosyası")
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--max-context-chars", type=int, default=80_000)
    parser.add_argument("--adapter", default="checkpoints/bulk_adapter/adapter.pt")
    parser.add_argument("--swap-layers", type=int, default=0, help="Faz 3 katman swap (eski wrapper)")
    parser.add_argument("--kvfree", action="store_true", help="Faz 3: gerçek KV'siz kuyruk swap")
    parser.add_argument("--sliding-window", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    doc_path = Path(args.file)
    if not doc_path.exists():
        raise SystemExit(f"Dosya bulunamadı: {doc_path}")

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)

    print(f"Yükleniyor: {doc_path.name} ({device})")
    context = load_document(doc_path, args.max_context_chars)
    prompt = build_qa_prompt(context, args.question)

    base, tokenizer, _ = load_tinyllama(device, dtype)

    if args.kvfree:
        from bulk_hybrid import create_kvfree
        from bulk_layer_swap import load_bulk_swap_checkpoint
        model, _, _ = create_kvfree(
            device, dtype, base_model=base,
            n_swap_layers=max(args.swap_layers, 4),
            sliding_window=args.sliding_window,
        )
        swap_ckpt = Path("checkpoints/bulk_swap/bulk_v2.pt")
        if swap_ckpt.exists():
            load_bulk_swap_checkpoint(model, str(swap_ckpt), device)
        print(f"Mod: KVFree tail (swap={model.n_swap}, window={args.sliding_window})")
    else:
        adapter_path = Path(args.adapter)
        adapter_kw = dict(k_short=8, trigger_stride=4, adaptive_trigger=False)
        if adapter_path.exists():
            model, _, _ = create_hybrid(
                device, dtype, adapter_path, base_model=base,
                n_swap_layers=args.swap_layers, **adapter_kw,
            )
        else:
            from bulk_hybrid import TinyLlamaWithBulk
            model = TinyLlamaWithBulk(
                base, freeze_base=True, n_swap_layers=args.swap_layers, **adapter_kw,
            ).to(device)
        print(f"Mod: Hybrid adapter (swap_layers={args.swap_layers})")

    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].to(device)
    n_tok = ids.size(1)
    print(f"Prompt: {n_tok} token")

    with torch.inference_mode():
        new_ids = model.generate_cached(ids, args.max_new)

    answer = tokenizer.decode(new_ids[0], skip_special_tokens=True)
    print("\n--- Cevap ---")
    print(answer)
    print("-------------")


if __name__ == "__main__":
    main()
