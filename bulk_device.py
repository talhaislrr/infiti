"""Cihaz ve dtype seçimi — Mac MPS / CUDA / CPU."""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, train: bool = True) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    # MPS eğitimde fp32 daha kararlı
    return torch.float32


def device_summary(device: torch.device) -> str:
    parts = [str(device)]
    if device.type == "mps":
        parts.append("Apple Metal")
    elif device.type == "cuda":
        parts.append(torch.cuda.get_device_name(0))
    return " | ".join(parts)
