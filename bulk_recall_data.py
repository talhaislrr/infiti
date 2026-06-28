"""
Sentetik uzun menzil recall verisi — eğitim + demo ortak.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch.utils.data import Dataset

MAX_PROMPT_TOKENS = 2048

FACT_KEYS = [
    "secret project codename",
    "hidden laboratory location",
    "agent codename",
    "mission target city",
    "encryption passphrase",
]

FACT_VALUES = [
    "Nebula Seven",
    "Crystal Harbor",
    "Silver Fox",
    "Prague",
    "Omega Key",
    "Blue Horizon",
    "Red Comet",
    "Iron Gate",
]

FILLER_PARAGRAPH = (
    "In the distant kingdom of Aldoria, scholars debated philosophy and trade routes. "
    "Merchants crossed the Silver Sea while astronomers mapped constellations. "
    "None of this relates to the confidential dossier header above."
)


def build_recall_document(
    fact_key: str,
    fact_value: str,
    filler: str,
    target_tokens: int,
    tokenizer,
    max_len: int = MAX_PROMPT_TOKENS,
    include_answer: bool = True,
) -> str:
    """Başa fact, araya dolgu, sona soru (+ cevap eğitim için)."""
    target_tokens = min(target_tokens, max_len - 16)
    header = (
        f"=== CONFIDENTIAL DOSSIER ===\n"
        f"The {fact_key} is {fact_value}. This fact must be remembered.\n"
        f"Reference code: {fact_value.upper().replace(' ', '-')}\n\n"
    )
    question = f"\nQuestion: What is the {fact_key}?\n"
    answer = f"Answer: {fact_value}" if include_answer else "Answer:"

    body = ""
    para = filler.strip()
    draft = header + question + answer
    while len(tokenizer.encode(draft)) < target_tokens:
        body += para + "\n\n"
        draft = header + body + question + answer

    ids = tokenizer.encode(draft)
    if len(ids) <= max_len:
        return draft

    header_ids = tokenizer.encode(header)
    tail_ids = tokenizer.encode(question + answer)
    budget = max_len - len(header_ids) - len(tail_ids)
    if budget < 8:
        return tokenizer.decode((header_ids + tail_ids)[:max_len])

    body_ids = tokenizer.encode(body)
    if len(body_ids) > budget:
        body_ids = body_ids[:budget]
    return tokenizer.decode(header_ids + body_ids + tail_ids)


class RecallTokenDataset(Dataset):
    """Sentetik fact→dolgu→soru→cevap LM örnekleri."""

    def __init__(
        self,
        tokenizer,
        n_samples: int = 200,
        seq_len: int = 512,
        gap_range: tuple[int, int] = (128, 480),
        max_len: int = MAX_PROMPT_TOKENS,
        seed: Optional[int] = 42,
    ):
        rng = random.Random(seed)
        self.samples: list[torch.Tensor] = []

        for i in range(n_samples):
            fact_key = FACT_KEYS[i % len(FACT_KEYS)]
            fact_value = FACT_VALUES[rng.randint(0, len(FACT_VALUES) - 1)]
            gap = rng.randint(gap_range[0], gap_range[1])
            target = min(seq_len + 1, gap + 80)

            text = build_recall_document(
                fact_key, fact_value, FILLER_PARAGRAPH, target, tokenizer, max_len=max_len,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)

            if len(ids) < seq_len + 1:
                pad = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
                ids = ids + [pad] * (seq_len + 1 - len(ids))
            else:
                ids = ids[: seq_len + 1]

            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[i]
        return s[:-1], s[1:]


class MixedRecallWikiDataset(Dataset):
    """WikiText + recall karışımı."""

    def __init__(self, wiki_ds: Dataset, recall_ds: Dataset, recall_ratio: float = 0.5):
        self.wiki = wiki_ds
        self.recall = recall_ds
        self.recall_ratio = recall_ratio
        self._len = max(len(wiki_ds), len(recall_ds))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.recall_ratio:
            return self.recall[i % len(self.recall)]
        return self.wiki[i % len(self.wiki)]


# ---------------------------------------------------------------------------
# Multi-fact support
# ---------------------------------------------------------------------------

def build_multifact_document(
    fact_keys: list[str],
    fact_values: list[str],
    filler: str,
    target_tokens: int,
    tokenizer,
    max_len: int = MAX_PROMPT_TOKENS,
    include_answers: bool = True,
) -> str:
    """Başa N fact, araya dolgu, sona N soru (+ cevaplar eğitim için)."""
    assert len(fact_keys) == len(fact_values)
    target_tokens = min(target_tokens, max_len - 32)

    header_lines = ["=== CONFIDENTIAL DOSSIER ==="]
    for k, v in zip(fact_keys, fact_values):
        header_lines.append(f"The {k} is {v}. This fact must be remembered.")
        header_lines.append(f"Reference code: {v.upper().replace(' ', '-')}")
    header = "\n".join(header_lines) + "\n\n"

    qa_lines = []
    for k, v in zip(fact_keys, fact_values):
        qa_lines.append(f"Question: What is the {k}?")
        qa_lines.append(f"Answer: {v}" if include_answers else "Answer:")
    qa = "\n".join(qa_lines)

    body = ""
    para = filler.strip()
    draft = header + qa
    while len(tokenizer.encode(draft)) < target_tokens:
        body += para + "\n\n"
        draft = header + body + qa

    ids = tokenizer.encode(draft)
    if len(ids) <= max_len:
        return draft

    header_ids = tokenizer.encode(header)
    tail_ids = tokenizer.encode(qa)
    budget = max_len - len(header_ids) - len(tail_ids)
    if budget < 8:
        return tokenizer.decode((header_ids + tail_ids)[:max_len])
    body_ids = tokenizer.encode(body)
    if len(body_ids) > budget:
        body_ids = body_ids[:budget]
    return tokenizer.decode(header_ids + body_ids + tail_ids)


class MultifactRecallTokenDataset(Dataset):
    """N fact → dolgu → N soru → N cevap LM örnekleri.

    n_facts: sabit veya (min, max) tuple → her örnekte rastgele N.
    """

    def __init__(
        self,
        tokenizer,
        n_samples: int = 200,
        seq_len: int = 512,
        n_facts: int | tuple[int, int] = 2,
        gap_range: tuple[int, int] = (128, 480),
        max_len: int = MAX_PROMPT_TOKENS,
        seed: Optional[int] = 42,
    ):
        rng = random.Random(seed)
        self.samples: list[torch.Tensor] = []

        n_facts_range = (n_facts, n_facts) if isinstance(n_facts, int) else n_facts

        for i in range(n_samples):
            nf = rng.randint(*n_facts_range)
            keys = rng.sample(FACT_KEYS * ((nf // len(FACT_KEYS)) + 1), nf)
            values = [FACT_VALUES[rng.randint(0, len(FACT_VALUES) - 1)] for _ in range(nf)]

            gap = rng.randint(gap_range[0], gap_range[1])
            target = min(seq_len + 1, gap + 80)

            text = build_multifact_document(
                keys, values, FILLER_PARAGRAPH, target, tokenizer, max_len=max_len,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            pad = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
            if len(ids) < seq_len + 1:
                ids = ids + [pad] * (seq_len + 1 - len(ids))
            else:
                ids = ids[: seq_len + 1]

            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[i]
        return s[:-1], s[1:]


# ---------------------------------------------------------------------------
# Conversational recall (chat format — isim / fact hatırlama)
# ---------------------------------------------------------------------------

CONV_FACT_PAIRS: list[tuple[str, str]] = [
    ("My name is {v}.", "What is my name?", "Your name is {v}."),
    ("I work at {v}.", "Where do I work?", "You work at {v}."),
    ("My favorite color is {v}.", "What is my favorite color?", "Your favorite color is {v}."),
    ("I live in {v}.", "What city do I live in?", "You live in {v}."),
    ("My secret code is {v}.", "What is my secret code?", "Your secret code is {v}."),
]

CONV_VALUES: list[str] = [
    "Talha", "Alice", "Mia", "Chen", "Zara",
    "TechCorp", "BlueLab", "StarBase",
    "red", "blue", "green",
    "Istanbul", "Tokyo", "Berlin", "Paris",
    "ALPHA-7", "OMEGA-9", "DELTA-3",
]

CONV_FILLER_TURNS: list[tuple[str, str]] = [
    ("What is 2 + 2?", "It is 4."),
    ("Tell me a short joke.", "Why don't scientists trust atoms? Because they make up everything."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How are you today?", "I am doing well, thank you!"),
    ("What is the speed of light?", "Approximately 299,792 km/s."),
]


def build_conversation_recall(
    intro: str,
    question: str,
    answer: str,
    filler_turns: list[tuple[str, str]],
    tokenizer,
    target_tokens: int,
    max_len: int = MAX_PROMPT_TOKENS,
    include_answer: bool = True,
) -> str:
    """Chat formatında fact → filler konuşmalar → soru → cevap."""
    eos = "</s>"

    def fmt_turn(u: str, a: str) -> str:
        return f"<|user|>\n{u}{eos}\n<|assistant|>\n{a}{eos}\n"

    header = f"<|system|>\nYou are a helpful assistant.{eos}\n"
    intro_turn = fmt_turn(intro, "Got it, I will remember that.")
    qa_block = f"<|user|>\n{question}{eos}\n<|assistant|>\n{answer if include_answer else ''}"

    body = header + intro_turn
    for u, a in filler_turns:
        candidate = body + fmt_turn(u, a) + qa_block
        if len(tokenizer.encode(candidate)) >= target_tokens:
            break
        body += fmt_turn(u, a)

    result = body + qa_block
    ids = tokenizer.encode(result)
    if len(ids) > max_len:
        result = tokenizer.decode(ids[:max_len])
    return result


class ConversationalRecallDataset(Dataset):
    """Chat format: intro → filler → soru → cevap.

    Modeli gerçek konuşma bağlamında fact hatırlamaya eğitir.
    """

    def __init__(
        self,
        tokenizer,
        n_samples: int = 200,
        seq_len: int = 256,
        n_filler_turns: int | tuple[int, int] = (2, 5),
        seed: Optional[int] = 42,
        max_len: int = MAX_PROMPT_TOKENS,
    ):
        rng = random.Random(seed)
        self.samples: list[torch.Tensor] = []

        n_filler_range = (
            (n_filler_turns, n_filler_turns)
            if isinstance(n_filler_turns, int)
            else n_filler_turns
        )

        for i in range(n_samples):
            pattern = CONV_FACT_PAIRS[i % len(CONV_FACT_PAIRS)]
            val = CONV_VALUES[rng.randint(0, len(CONV_VALUES) - 1)]
            intro = pattern[0].format(v=val)
            question = pattern[1]
            answer = pattern[2].format(v=val)

            nf = rng.randint(*n_filler_range)
            filler = rng.choices(CONV_FILLER_TURNS, k=nf)

            text = build_conversation_recall(
                intro, question, answer, filler, tokenizer,
                target_tokens=seq_len, max_len=max_len,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            pad = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
            if len(ids) < seq_len + 1:
                ids = ids + [pad] * (seq_len + 1 - len(ids))
            else:
                ids = ids[: seq_len + 1]

            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[i]
        return s[:-1], s[1:]
