"""
BulkState — Hiyerarşik biriken bellek (BulkTrigger v2)
=====================================================
short  : son k token (hızlı güncelleme)
medium : her medium_interval token'da GRU ile birikim
long   : her long_interval token'da gated fusion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BulkStateTensors:
    short: torch.Tensor   # [B, d]
    medium: torch.Tensor  # [B, d]
    long: torch.Tensor    # [B, d]
    step: int = 0

    def as_kv(self) -> torch.Tensor:
        """Cross-attention K/V: [B, 3, d]"""
        return torch.stack([self.short, self.medium, self.long], dim=1)

    @classmethod
    def zeros(cls, batch: int, d_model: int, device, dtype) -> "BulkStateTensors":
        z = torch.zeros(batch, d_model, device=device, dtype=dtype)
        return cls(short=z.clone(), medium=z.clone(), long=z.clone(), step=0)


def embedding_surprise(x: torch.Tensor) -> torch.Tensor:
    """Pozisyon başına sürpriz skoru: ardışık embedding farkının normu [B, T]."""
    B, T, _ = x.shape
    if T <= 1:
        return torch.zeros(B, T, device=x.device, dtype=x.dtype)
    diff = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
    z = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
    return torch.cat([z, diff], dim=1)


class BulkStateManager(nn.Module):
    """
    Kısa / orta / uzun ölçekli bulk belleği günceller.
    Inference: state adım adım birikir.
    Eğitim: trigger_stride ile short encoder seyrek çalıştırılabilir.
    adaptive_trigger=True ise yüksek sürpriz skorunda da tetiklenir.
    """

    def __init__(
        self,
        d_model: int,
        k_short: int = 8,
        medium_interval: int = 16,
        long_interval: int = 128,
        trigger_layers: int = 1,
        trigger_heads: int = 2,
        adaptive_trigger: bool = False,
        surprise_threshold: float = 1.0,
    ):
        super().__init__()
        self.k_short = k_short
        self.medium_interval = medium_interval
        self.long_interval = long_interval
        self.adaptive_trigger = adaptive_trigger
        self.surprise_threshold = surprise_threshold

        from bulk_trigger_arch import TriggerNetwork

        self.short_encoder = TriggerNetwork(
            d_model, n_layers=trigger_layers, n_heads=trigger_heads, k=k_short
        )
        self.medium_gru = nn.GRUCell(d_model, d_model)
        self.long_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.long_proj = nn.Linear(d_model, d_model)
        self.norm_short = nn.LayerNorm(d_model)
        self.norm_medium = nn.LayerNorm(d_model)
        self.norm_long = nn.LayerNorm(d_model)

    def _advance_without_short(
        self, state: BulkStateTensors, last_short_enc: torch.Tensor
    ) -> BulkStateTensors:
        step = state.step + 1
        medium, long = state.medium, state.long
        if step % self.medium_interval == 0 or step == 1:
            medium = self.norm_medium(self.medium_gru(last_short_enc, medium))
        if step % self.long_interval == 0 or step == 1:
            gate = self.long_gate(torch.cat([long, medium], dim=-1))
            long = self.norm_long(gate * long + (1.0 - gate) * self.long_proj(medium))
        return BulkStateTensors(
            short=last_short_enc, medium=medium, long=long, step=step
        )

    def _should_run_short_encoder(
        self,
        t: int,
        trigger_stride: int,
        last_short_enc: Optional[torch.Tensor],
        surprise_t: Optional[torch.Tensor],
    ) -> bool:
        if last_short_enc is None:
            return True
        if t % trigger_stride == 0:
            return True
        if self.adaptive_trigger and surprise_t is not None:
            return bool((surprise_t > self.surprise_threshold).any())
        return False

    def update(
        self,
        window: torch.Tensor,
        state: BulkStateTensors,
        force_medium: bool = False,
        force_long: bool = False,
    ) -> BulkStateTensors:
        """window: [B, k_short, d_model]"""
        short = self.norm_short(self.short_encoder(window))
        step = state.step + 1
        medium = state.medium
        long = state.long

        if force_medium or step % self.medium_interval == 0 or step == 1:
            medium = self.norm_medium(self.medium_gru(short, medium))

        if force_long or step % self.long_interval == 0 or step == 1:
            gate = self.long_gate(torch.cat([long, medium], dim=-1))
            long = self.norm_long(gate * long + (1.0 - gate) * self.long_proj(medium))

        return BulkStateTensors(short=short, medium=medium, long=long, step=step)

    def evolve_sequence(
        self,
        windows: torch.Tensor,
        trigger_stride: int = 1,
        surprise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        windows: [B, T, k, d] → [B, T, 3, d]
        surprise: [B, T] opsiyonel sürpriz skoru (adaptif tetik)
        """
        B, T, _, _ = windows.shape
        device, dtype = windows.device, windows.dtype
        state = BulkStateTensors.zeros(B, windows.size(-1), device, dtype)
        kv_list: list[torch.Tensor] = []
        last_short_enc: Optional[torch.Tensor] = None

        for t in range(T):
            sur_t = surprise[:, t] if surprise is not None else None
            if self._should_run_short_encoder(t, trigger_stride, last_short_enc, sur_t):
                state = self.update(windows[:, t], state)
                last_short_enc = state.short
            else:
                assert last_short_enc is not None
                state = self._advance_without_short(state, last_short_enc)
            kv_list.append(state.as_kv())

        return torch.stack(kv_list, dim=1)

    def evolve_sequence_with_state(
        self,
        windows: torch.Tensor,
        trigger_stride: int = 1,
        surprise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        """evolve_sequence + final BulkState (prefill sync için)."""
        B, T, _, _ = windows.shape
        device, dtype = windows.device, windows.dtype
        state = BulkStateTensors.zeros(B, windows.size(-1), device, dtype)
        kv_list: list[torch.Tensor] = []
        last_short_enc: Optional[torch.Tensor] = None

        for t in range(T):
            sur_t = surprise[:, t] if surprise is not None else None
            if self._should_run_short_encoder(t, trigger_stride, last_short_enc, sur_t):
                state = self.update(windows[:, t], state)
                last_short_enc = state.short
            else:
                assert last_short_enc is not None
                state = self._advance_without_short(state, last_short_enc)
            kv_list.append(state.as_kv())

        return torch.stack(kv_list, dim=1), state
