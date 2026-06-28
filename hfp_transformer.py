"""
HFP × Transformer — Encoder tabanlı metin sınıflandırıcı
=========================================================
Standart Transformer FFN; BulkLinear ile HFP bulk projeksiyonu.
Eğitim: hfp_config.HFPStiffTransientScheduler + HFPZenonQuantizationScheduler
"""

from __future__ import annotations

import math
from enum import Enum

import torch
import torch.nn as nn

from hfp_principles import BulkLinear


class FFNMode(str, Enum):
    STANDARD = "standard"
    BULK = "bulk"


def build_ffn_linear(
    in_features: int,
    out_features: int,
    mode: FFNMode,
    bulk_rank: int = 128,
) -> nn.Module:
    if mode == FFNMode.STANDARD:
        return nn.Linear(in_features, out_features)
    if mode == FFNMode.BULK:
        return BulkLinear(in_features, out_features, rank=bulk_rank)
    raise ValueError(mode)


class TransformerFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        mode: FFNMode,
        bulk_rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.fc1 = build_ffn_linear(d_model, d_ff, mode, bulk_rank)
        self.fc2 = build_ffn_linear(d_ff, d_model, mode, bulk_rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def bulk_reg_loss(self) -> torch.Tensor:
        if self.mode != FFNMode.BULK:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in (self.fc1, self.fc2):
            if isinstance(layer, BulkLinear):
                loss = loss + layer.regularization_loss()
        return loss


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        mode: FFNMode,
        bulk_rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = TransformerFFN(d_model, d_ff, mode, bulk_rank, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class HFPTransformerClassifier(nn.Module):
    """Küçük encoder-only Transformer — LLM mimarisinin çekirdeği."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_len: int = 128,
        mode: FFNMode = FFNMode.STANDARD,
        bulk_rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.bulk_rank = bulk_rank
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, mode, bulk_rank, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, seq = input_ids.shape
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(b, -1)
        positions = positions.clamp(max=self.max_len - 1)

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        pad_mask = input_ids == 0
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)

        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)

    def bulk_reg_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total = total + layer.ffn.bulk_reg_loss()
        return total


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_ffn_params(model: HFPTransformerClassifier) -> int:
    return sum(p.numel() for layer in model.layers for p in layer.ffn.parameters())
