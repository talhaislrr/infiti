"""Bulk bellek ve sürpriz skoru yardımcıları."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def kv_seq_len(past: Any) -> int:
    """KV cache uzunluğu — transformers 4.x tuple ve 5.x DynamicCache uyumu."""
    if past is None:
        return 0
    if hasattr(past, "get_seq_length"):
        try:
            n = int(past.get_seq_length())
            if n > 0:
                return n
        except (TypeError, ValueError, AttributeError):
            pass
    if hasattr(past, "key_cache") and past.key_cache:
        for k in past.key_cache:
            if k is not None and k.numel() > 0:
                return int(k.size(-2))
    if isinstance(past, (list, tuple)) and past and past[0] is not None:
        return int(past[0][0].size(-2))
    return 0


def token_loss_surprise(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Pozisyon başına cross-entropy sürprizi [B, T].
    labels: [B, T] — logits ile aynı uzunlukta hedef token id.
    """
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)
    flat_labels = labels.reshape(B * T)
    per_tok = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=ignore_index)
    return per_tok.view(B, T)


def estimate_bulk_state_bytes(
    n_layers: int,
    d_model: int,
    batch: int = 1,
    dtype_bytes: int = 4,
) -> int:
    """Katman başına 3 vektör (short/medium/long) × d_model."""
    return n_layers * batch * 3 * d_model * dtype_bytes


def estimate_kv_cache_bytes(
    seq_len: int,
    n_layers: int,
    d_model: int,
    batch: int = 1,
    dtype_bytes: int = 4,
) -> int:
    return 2 * n_layers * batch * seq_len * d_model * dtype_bytes
