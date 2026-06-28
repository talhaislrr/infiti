"""
Yama 3: AdaptivePrecision
-------------------------
schedule_points (oran): [0.7, 0.9] → fp16 autocast, sonra int8 dynamic quant
"""

from __future__ import annotations

import contextlib
import copy
import logging
import warnings
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PrecisionMode = Literal["fp32", "fp16", "int8"]


class AdaptivePrecision:
    """
    Eğitim adımına göre hassasiyet modu.
    int8 geçişi: quantize_dynamic uygular (eğitim devamı sınırlı olabilir).
    """

    def __init__(
        self,
        total_steps: int,
        schedule_points: list[float] | None = None,
        grad_threshold: float = 1e-5,
        quantize_modules: tuple[type, ...] = (nn.Linear,),
    ):
        if schedule_points is None:
            schedule_points = [0.7, 0.9]
        if len(schedule_points) < 2:
            raise ValueError("schedule_points en az 2 oran içermeli, örn. [0.7, 0.9]")

        self.total_steps = max(1, total_steps)
        self.schedule_points = schedule_points
        self.grad_threshold = grad_threshold
        self.quantize_modules = quantize_modules

        self.fp16_step = int(self.total_steps * schedule_points[0])
        self.int8_step = int(self.total_steps * schedule_points[1])

        self.current_step = 0
        self.mode: PrecisionMode = "fp32"
        self.mode_history: list[PrecisionMode] = []
        self._last_grad_norm = 0.0
        self._int8_applied = False
        self._pending_mode: PrecisionMode | None = None

    def record_grad_norm(self, norm: float) -> None:
        self._last_grad_norm = norm

    def _can_transition(self) -> bool:
        if self._last_grad_norm > self.grad_threshold:
            logger.debug(
                "AdaptivePrecision: grad_norm=%.2e > threshold=%.2e — geçiş ertelendi",
                self._last_grad_norm,
                self.grad_threshold,
            )
            return False
        return True

    def _set_mode(self, mode: PrecisionMode) -> None:
        if mode == "int8" and not self._int8_applied:
            warnings.warn(
                "AdaptivePrecision: int8 modu — inference odaklı; eğitim kısıtlı olabilir.",
                stacklevel=3,
            )
            self._int8_applied = True
        self.mode = mode

    def step(self) -> PrecisionMode:
        self.current_step += 1
        n = self.current_step

        target: PrecisionMode = self.mode
        if n >= self.int8_step:
            target = "int8"
        elif n >= self.fp16_step:
            target = "fp16"

        if target != self.mode:
            if self._can_transition():
                self._set_mode(target)
                self._pending_mode = None
            else:
                self._pending_mode = target
        elif self._pending_mode is not None and self._can_transition():
            self._set_mode(self._pending_mode)
            self._pending_mode = None

        self.mode_history.append(self.mode)
        return self.mode

    def training_context(self, device_type: str = "cuda"):
        if self.mode == "fp16" and device_type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def maybe_quantize_model(self, model: nn.Module) -> nn.Module:
        """int8 modunda modeli dinamik nicele (CPU inference/eğitim sonu)."""
        if self.mode != "int8":
            return model
        if self._int8_applied:
            m = copy.deepcopy(model)
            m.eval()
            m.cpu()
            return torch.quantization.quantize_dynamic(m, self.quantize_modules, dtype=torch.qint8)
        return model

    def reset(self) -> None:
        self.current_step = 0
        self.mode = "fp32"
        self.mode_history.clear()
        self._last_grad_norm = 0.0
        self._int8_applied = False
        self._pending_mode = None
