"""
BulkTrigger mimarisi — Tetikleyici / Üretici ayrışımı
======================================================
Token'lar geçmişi taşımaz; Trigger son k token'dan bulk_query üretir.
Generator yalnızca bulk_query + mevcut token ile çalışır (KV-cache yok).

Bileşenler:
  TriggerNetwork       → son k token → bulk_query [B, d_model]
  GeneratorBlock       → cross-attn(bulk_query) + FFN
  BulkTriggerDecoderLayer
  BulkTriggerLM        → küçük dil modeli
  StandardDecoderLM    → karşılaştırma baseline (causal self-attn)
"""

from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriggerNetwork(nn.Module):
    """Son k token'ı işleyip bağlam özet vektörü üretir."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        k: int = 4,
        dim_ff: int | None = None,
    ):
        super().__init__()
        self.k = k
        dim_ff = dim_ff or d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        window: [B, w, d_model]  (w <= k, solda pad olabilir)
        returns bulk_query: [B, d_model]
        """
        h = self.encoder(window)
        # Son gerçek pozisyonun özeti: mean pool
        bulk = h.mean(dim=1)
        return self.out_norm(bulk)


class GeneratorBlock(nn.Module):
    """Cross-attention: Query=mevcut token, K/V=bulk_query (tek vektör)."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        dim_ff = dim_ff or d_model * 4
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, bulk_query: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        bulk_query: [B, d_model] | [B, S, d_model] | [B, T, d_model]
        [B, T, d_model] geçirilirse pozisyonlar arası sızıntı olur; katman içi
        reshape ile her pozisyon yalnızca kendi bulk vektörüne bakar.
        """
        if bulk_query.dim() == 3 and bulk_query.size(1) != 1:
            B, T, D = x.shape
            if bulk_query.size(1) == T:
                # Eski API: [B, T, d] → pozisyon başına izole cross-attn
                x_flat = x.reshape(B * T, 1, D)
                kv = bulk_query.reshape(B * T, 1, D)
                out = self._attn_ffn(x_flat, kv)
                return out.reshape(B, T, D)
        if bulk_query.dim() == 2:
            kv = bulk_query.unsqueeze(1)
        else:
            kv = bulk_query
        return self._attn_ffn(x, kv)

    def _attn_ffn(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(x, kv, kv, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class BulkTriggerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        trigger_layers: int = 2,
        trigger_heads: int = 4,
        k: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k
        self.trigger = TriggerNetwork(d_model, trigger_layers, trigger_heads, k)
        self.generator = GeneratorBlock(d_model, n_heads, dropout=dropout)

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        """Her pozisyon için son k embedding → [B, T, k, d]"""
        B, T, D = x.shape
        padded = F.pad(x, (0, 0, self.k - 1, 0))
        # unfold: [B, T, D, k] → permute → [B, T, k, D]
        w = padded.unfold(dimension=1, size=self.k, step=1)
        return w.permute(0, 1, 3, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        windows = self._windows(x)
        flat = windows.reshape(B * T, self.k, D)
        bulk = self.trigger(flat)  # [B*T, D]
        x_flat = x.reshape(B * T, 1, D)
        out = self.generator(x_flat, bulk.unsqueeze(1))
        return out.reshape(B, T, D)


class BulkTriggerLM(nn.Module):
    """Tam BulkTrigger dil modeli."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        trigger_layers: int = 2,
        trigger_heads: int = 4,
        k: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            BulkTriggerDecoderLayer(d_model, n_heads, trigger_layers, trigger_heads, k, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos_embed(pos))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    def _build_window(self, history_embeds: deque, emb: torch.Tensor) -> torch.Tensor:
        B, D = emb.shape
        hist: list[torch.Tensor] = []
        for h in list(history_embeds)[-(self.k - 1) :]:
            if h.dim() == 1:
                h = h.unsqueeze(0)
            hist.append(h)
        hist.append(emb)
        window = torch.stack(hist, dim=1)
        if window.size(1) < self.k:
            pad = window[:, :1, :].expand(B, self.k - window.size(1), D)
            window = torch.cat([pad, window], dim=1)
        return window[:, -self.k :, :]

    @torch.inference_mode()
    def generate_step(
        self,
        token_id: torch.Tensor,
        history_embeds: deque,
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tek token inference — KV-cache yok.
        history_embeds: son k embedding deque ([B,d] veya [d] tensors)
        """
        emb = self.embed(token_id) + self.pos_embed(
            torch.tensor([min(pos, self.max_len - 1)], device=token_id.device)
        )
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        window = self._build_window(history_embeds, emb)

        x = emb.unsqueeze(1)
        for layer in self.layers:
            bq = layer.trigger(window)
            x = layer.generator(x, bq.unsqueeze(1))
        x = self.norm(x)
        logits = self.head(x[:, -1])
        return logits, emb


class StandardDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        dim_ff = dim_ff or d_model * 4
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = x.size(1)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class StandardDecoderLM(nn.Module):
    """Klasik causal Transformer decoder (karşılaştırma)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            StandardDecoderLayer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos_embed(pos))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

    @torch.inference_mode()
    def generate_step_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        """KV-cache yok — her adımda tüm geçmişi yeniden işle (adil karşılaştırma)."""
        return self.forward(input_ids)[:, -1]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def estimate_kv_cache_bytes(seq_len: int, n_layers: int, d_model: int, batch: int = 1) -> int:
    """Standart MHA KV cache: 2 * n_layers * batch * seq * d_model * 4 byte (fp32)"""
    return 2 * n_layers * batch * seq_len * d_model * 4
