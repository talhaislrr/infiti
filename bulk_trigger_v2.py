"""
BulkTrigger v2 — BulkState hiyerarşik bellek
=============================================
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bulk_state import BulkStateManager, BulkStateTensors, embedding_surprise
from bulk_trigger_arch import GeneratorBlock, StandardDecoderLM, count_params, estimate_kv_cache_bytes


class BulkTriggerDecoderLayerV2(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        k_short: int = 8,
        medium_interval: int = 16,
        long_interval: int = 128,
        trigger_stride: int = 8,
        adaptive_trigger: bool = False,
        surprise_threshold: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k_short = k_short
        self.trigger_stride = trigger_stride
        self.adaptive_trigger = adaptive_trigger
        self.bulk_mgr = BulkStateManager(
            d_model,
            k_short=k_short,
            medium_interval=medium_interval,
            long_interval=long_interval,
            adaptive_trigger=adaptive_trigger,
            surprise_threshold=surprise_threshold,
        )
        self.generator = GeneratorBlock(d_model, n_heads, dropout=dropout)

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        padded = F.pad(x, (0, 0, self.k_short - 1, 0))
        w = padded.unfold(dimension=1, size=self.k_short, step=1)
        return w.permute(0, 1, 3, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = self._windows(x)
        surprise = embedding_surprise(x) if self.adaptive_trigger else None
        bulk_kv = self.bulk_mgr.evolve_sequence(
            windows, trigger_stride=self.trigger_stride, surprise=surprise,
        )
        B, T, S, D = bulk_kv.shape
        # [B, T, 1, d] Q  ×  [B, T, 3, d] KV  → batch as B*T
        x_flat = x.reshape(B * T, 1, D)
        kv_flat = bulk_kv.reshape(B * T, S, D)
        out = self.generator(x_flat, kv_flat)
        return out.reshape(B, T, D)

    def forward_step(
        self,
        x: torch.Tensor,
        window: torch.Tensor,
        state: BulkStateTensors,
        surprise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        """Tek token inference — opsiyonel sürpriz ile long pin."""
        new_state = self.bulk_mgr.update(window, state, surprise=surprise)
        out = self.generator(x, new_state.as_kv())
        return out, new_state

    def consolidate_step(
        self,
        window: torch.Tensor,
        state: BulkStateTensors,
        surprise: Optional[torch.Tensor] = None,
    ) -> BulkStateTensors:
        """KV crop öncesi yalnızca BulkState güncelle."""
        return self.bulk_mgr.consolidate(window, state, surprise=surprise)

    def forward_with_state(
        self,
        x: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        """Batch prefill + final BulkState."""
        windows = self._windows(x)
        bulk_kv, state = self.bulk_mgr.evolve_sequence_with_state(
            windows, trigger_stride=self.trigger_stride, surprise=surprise,
        )
        B, T, S, D = bulk_kv.shape
        x_flat = x.reshape(B * T, 1, D)
        kv_flat = bulk_kv.reshape(B * T, S, D)
        out = self.generator(x_flat, kv_flat)
        return out.reshape(B, T, D), state


class BulkTriggerLMv2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        k_short: int = 8,
        medium_interval: int = 16,
        long_interval: int = 128,
        trigger_stride: int = 8,
        adaptive_trigger: bool = False,
        surprise_threshold: float = 1.0,
        max_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.k_short = k_short
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            BulkTriggerDecoderLayerV2(
                d_model, n_heads, k_short, medium_interval, long_interval,
                trigger_stride, adaptive_trigger, surprise_threshold, dropout,
            )
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
        return self.head(self.norm(x))

    def _build_window(self, embed_history: deque, emb: torch.Tensor) -> torch.Tensor:
        """Son k embedding → [B, k_short, d_model]."""
        B, D = emb.shape
        hist: list[torch.Tensor] = []
        for h in list(embed_history)[-(self.k_short - 1) :]:
            if h.dim() == 1:
                h = h.unsqueeze(0)
            hist.append(h)
        hist.append(emb)
        window = torch.stack(hist, dim=1)
        if window.size(1) < self.k_short:
            pad = window[:, :1, :].expand(B, self.k_short - window.size(1), D)
            window = torch.cat([pad, window], dim=1)
        return window[:, -self.k_short :, :]

    @torch.inference_mode()
    def generate_step(
        self,
        token_id: torch.Tensor,
        embed_history: deque,
        bulk_states: list[BulkStateTensors],
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[BulkStateTensors]]:
        device = token_id.device
        emb = self.embed(token_id) + self.pos_embed(
            torch.tensor([min(pos, self.max_len - 1)], device=device)
        )
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        window = self._build_window(embed_history, emb)

        x = emb.unsqueeze(1)
        new_states = []
        for layer, st in zip(self.layers, bulk_states):
            x, st = layer.forward_step(x, window, st)
            new_states.append(st)

        x = self.norm(x)
        logits = self.head(x[:, -1])
        return logits, emb, new_states

    def init_bulk_states(self, batch: int, device, dtype) -> list[BulkStateTensors]:
        return [
            BulkStateTensors.zeros(batch, self.d_model, device, dtype)
            for _ in self.layers
        ]
