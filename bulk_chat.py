#!/usr/bin/env python3
"""
BulkTrigger interaktif sohbet
=============================
KVFree / hibrit modda multi-turn state persistence:
  Tur 1 → prefill
  Tur 2+ → sadece yeni suffix tokenları decode_step ile cache'e eklenir
  /reset → history + model cache sıfırlanır

Kullanım:
  python3 bulk_chat.py --kvfree --device mps
  python3 bulk_chat.py --hybrid --device mps
  python3 bulk_chat.py --base
  python3 bulk_chat.py --kvfree --no-persist   # eski davranış (full re-prefill)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Protocol

import torch

from bulk_device import pick_device, pick_dtype
from bulk_hybrid import (
    BaseKVGenerator,
    TinyLlamaWithBulk,
    create_hybrid,
    create_kvfree,
    load_tinyllama,
)
from bulk_layer_swap import TinyLlamaKVFreeTail, load_bulk_swap_checkpoint


SYSTEM = "You are a helpful assistant. Reply in one or two short sentences."
DEFAULT_MAX_CONTEXT = 4096


class CachedGenerator(Protocol):
    def prefill(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...
    def decode_step(self, token_id: torch.Tensor) -> torch.Tensor: ...


def build_messages(history: list[tuple[str, str]], user_msg: str) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM}]
    for u, a in history:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def format_chat(tokenizer, history: list[tuple[str, str]], user_msg: str) -> str:
    msgs = build_messages(history, user_msg)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    parts = [f"<|system|>\n{SYSTEM}</s>"]
    for u, a in history:
        parts.append(f"<|user|>\n{u}</s>")
        parts.append(f"<|assistant|>\n{a}</s>")
    parts.append(f"<|user|>\n{user_msg}</s>")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def format_turn_suffix(tokenizer, user_msg: str, first_turn: bool) -> str:
    """Yeni tur suffix — geçmişi yeniden tokenize etmez (cache uyumu)."""
    if first_turn:
        return format_chat(tokenizer, [], user_msg)
    if getattr(tokenizer, "chat_template", None):
        chunk = [
            {"role": "user", "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            chunk, tokenize=False, add_generation_prompt=True,
        )
    return f"<|user|>\n{user_msg}</s>\n<|assistant|>\n"


def tokenize_text(tokenizer, text: str, device: torch.device, max_len: int) -> list[int]:
    ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )["input_ids"][0].tolist()
    return ids


def trim_session_left(session_ids: list[int], max_len: int) -> list[int]:
    if len(session_ids) <= max_len:
        return session_ids
    return session_ids[-max_len:]


@torch.inference_mode()
def feed_tokens(model: CachedGenerator, token_ids: list[int], device: torch.device) -> torch.Tensor:
    logits: Optional[torch.Tensor] = None
    for tid in token_ids:
        t = torch.tensor([tid], device=device)
        logits = model.decode_step(t)
    assert logits is not None
    return logits


@torch.inference_mode()
def generate_from_logits(
    model: CachedGenerator,
    logits: torch.Tensor,
    tokenizer,
    device: torch.device,
    max_new: int,
    session_ids: list[int],
) -> tuple[str, list[int]]:
    eos = tokenizer.eos_token_id
    tokens: list[int] = []
    next_tok = logits.argmax(-1, keepdim=True)

    for _ in range(max_new):
        tid = int(next_tok.item())
        if tid == eos:
            session_ids.append(tid)
            break
        tokens.append(tid)
        session_ids.append(tid)
        logits = model.decode_step(next_tok.squeeze(1))
        next_tok = logits.argmax(-1, keepdim=True)

    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    text = text.split("</s>")[0].strip()
    if len(text) > 400:
        text = text[:400].rsplit(".", 1)[0] + "."
    return text, tokens


class ChatSession:
    """Multi-turn KV/BulkState cache — suffix incremental veya full re-prefill fallback."""

    def __init__(
        self,
        model: CachedGenerator,
        tokenizer,
        device: torch.device,
        max_new: int,
        max_context: int = DEFAULT_MAX_CONTEXT,
        persist: bool = True,
        use_chat_template: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new = max_new
        self.max_context = max_context
        self.persist = persist
        self.use_chat_template = use_chat_template
        self.session_ids: list[int] = []
        self.turn_count = 0
        self.last_mode = "idle"

    def reset(self) -> None:
        self.session_ids.clear()
        self.turn_count = 0
        self.last_mode = "reset"
        if isinstance(self.model, TinyLlamaKVFreeTail):
            self.model._cache = None
        elif isinstance(self.model, TinyLlamaWithBulk):
            self.model._past_key_values = None
            self.model._state = None
            self.model._hidden_hist.clear()
        elif isinstance(self.model, BaseKVGenerator):
            self.model._past_key_values = None

    @property
    def cache_pos(self) -> int:
        if isinstance(self.model, TinyLlamaKVFreeTail) and self.model._cache is not None:
            return self.model._cache.pos
        return len(self.session_ids)

    def _full_reprefill(self, full_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor([full_ids], device=self.device)
        if isinstance(self.model, TinyLlamaWithBulk):
            logits = self.model.prefill(ids, fast=True)
        else:
            logits = self.model.prefill(ids)
        self.session_ids = full_ids.copy()
        self.last_mode = "full_prefill"
        return logits

    @torch.inference_mode()
    def reply(self, history: list[tuple[str, str]], user_msg: str) -> tuple[str, dict]:
        if not self.persist:
            full_text = format_chat(self.tokenizer, history, user_msg)
            full_ids = tokenize_text(self.tokenizer, full_text, self.device, self.max_context)
            logits = self._full_reprefill(full_ids)
            text, _ = generate_from_logits(
                self.model, logits, self.tokenizer, self.device, self.max_new, [],
            )
            meta = {"prompt_tokens": len(full_ids), "mode": "no_persist", "cache_pos": 0}
            return text, meta

        first_turn = self.turn_count == 0 or not self.session_ids

        if self.use_chat_template and not first_turn:
            full_text = format_chat(self.tokenizer, history, user_msg)
            full_ids = tokenize_text(self.tokenizer, full_text, self.device, self.max_context)
            prefix = self.session_ids
            if len(prefix) <= len(full_ids) and full_ids[: len(prefix)] == prefix:
                delta = full_ids[len(prefix):]
                if not delta:
                    logits = self._full_reprefill(full_ids)
                    new_tokens = len(full_ids)
                else:
                    logits = feed_tokens(self.model, delta, self.device)
                    self.session_ids.extend(delta)
                    self.last_mode = "incremental"
                    new_tokens = len(delta)
            else:
                logits = self._full_reprefill(full_ids)
                new_tokens = len(full_ids)
        else:
            suffix = format_turn_suffix(self.tokenizer, user_msg, first_turn)
            delta = tokenize_text(self.tokenizer, suffix, self.device, self.max_context)

            if first_turn:
                ids = torch.tensor([delta], device=self.device)
                if isinstance(self.model, TinyLlamaWithBulk):
                    logits = self.model.prefill(ids, fast=True)
                else:
                    logits = self.model.prefill(ids)
                self.session_ids = delta.copy()
                self.last_mode = "prefill"
                new_tokens = len(delta)
            else:
                if not delta:
                    raise RuntimeError("Boş suffix — tokenize hatası")
                logits = feed_tokens(self.model, delta, self.device)
                self.session_ids.extend(delta)
                self.last_mode = "incremental"
                new_tokens = len(delta)

        if len(self.session_ids) > self.max_context:
            self.session_ids = trim_session_left(self.session_ids, self.max_context)
            full_ids = self.session_ids.copy()
            logits = self._full_reprefill(full_ids)
            self.last_mode = "context_trim_reprefill"
            new_tokens = len(full_ids)

        gen_ids_before = len(self.session_ids)
        text, _ = generate_from_logits(
            self.model, logits, self.tokenizer, self.device, self.max_new, self.session_ids,
        )
        self.turn_count += 1
        meta = {
            "prompt_tokens": gen_ids_before,
            "new_tokens": new_tokens,
            "generated": len(self.session_ids) - gen_ids_before,
            "session_tokens": len(self.session_ids),
            "cache_pos": self.cache_pos,
            "mode": self.last_mode,
        }
        return text, meta


def load_model(mode: str, device, dtype):
    base, tokenizer, path = load_tinyllama(device, dtype)

    if mode == "base":
        print(f"Mod: standart TinyLlama (KV-cache persist)\n  {path}")
        return BaseKVGenerator(base), tokenizer, "base"

    if mode == "hybrid":
        adapter = Path("checkpoints/bulk_adapter/adapter.pt")
        if adapter.exists():
            model, _, _ = create_hybrid(device, dtype, adapter, base_model=base)
        else:
            model = TinyLlamaWithBulk(base, freeze_base=True).to(device)
        print(f"Mod: Hybrid adapter (state persist)\n  {path}")
        return model, tokenizer, "hybrid"

    model, _, _ = create_kvfree(
        device, dtype, base_model=base,
        n_swap_layers=4, sliding_window=512, compact=True, d_bulk=256,
    )
    ckpt = Path("checkpoints/bulk_swap/bulk_v2.pt")
    if ckpt.exists():
        load_bulk_swap_checkpoint(model, str(ckpt), device)
        print(f"Mod: KVFree compact + persist (checkpoint yüklü)\n  {path}")
    else:
        print(f"Mod: KVFree persist (checkpoint yok — rastgele swap!)\n  {path}")
    return model, tokenizer, "kvfree"


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--kvfree", action="store_true", help="KVFree compact (varsayılan)")
    g.add_argument("--hybrid", action="store_true")
    g.add_argument("--base", action="store_true")
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument("--no-persist", action="store_true", help="Her turda full re-prefill")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    mode = "kvfree"
    if args.hybrid:
        mode = "hybrid"
    elif args.base:
        mode = "base"

    device = pick_device(args.device)
    dtype = pick_dtype(device, train=False)

    print("=" * 60)
    print("BulkTrigger Chat")
    print("  /quit — çıkış  |  /reset — cache sıfırla  |  /status — state bilgisi")
    if args.no_persist:
        print("  (--no-persist: her tur full re-prefill)")
    else:
        print("  Multi-turn state persistence: AÇIK")
    print("=" * 60)

    model, tokenizer, _ = load_model(mode, device, dtype)
    use_template = bool(getattr(tokenizer, "chat_template", None))
    session = ChatSession(
        model,
        tokenizer,
        device,
        args.max_new,
        max_context=args.max_context,
        persist=not args.no_persist,
        use_chat_template=use_template,
    )
    history: list[tuple[str, str]] = []

    while True:
        try:
            user = input("\nSen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz.")
            break

        if not user:
            continue
        if user.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Görüşürüz.")
            break
        if user.lower() == "/reset":
            history.clear()
            session.reset()
            print("(Konuşma + model cache sıfırlandı)")
            continue
        if user.lower() == "/status":
            print(
                f"  session_tokens={len(session.session_ids)}  "
                f"cache_pos={session.cache_pos}  "
                f"turns={session.turn_count}  "
                f"last_mode={session.last_mode}"
            )
            continue

        try:
            reply, meta = session.reply(history, user)
        except Exception as e:
            print(f"Hata: {e}")
            session.reset()
            print("  (cache sıfırlandı — tekrar deneyin)")
            continue

        mode_label = meta.get("mode", "?")
        print(
            f"  [{meta['session_tokens']} tok session | +{meta.get('new_tokens', '?')} yeni | "
            f"{mode_label}]",
            flush=True,
        )
        print(f"Asistan: {reply}")
        history.append((user, reply))


if __name__ == "__main__":
    main()
