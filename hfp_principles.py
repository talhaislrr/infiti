"""
HFP (Holographic Field Processing) Prensipleri - Saf Yazılım Mimarisi
======================================================================
Üç temel geometrik prensip:
  1. Bulk Projection     → BulkLinear (W = U V^T)
  2. Stiff Transient     → StiffTransientEarlyStopping
  3. Zeno Sızıntısı      → ZenonQuantizationScheduler
"""

from __future__ import annotations

import contextlib
import copy
import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# 1. BULK PROJECTION LAYER
#    W = U V^T  →  düşük rankta parametre tasarrufu + implicit regularization
# ─────────────────────────────────────────────────────────────────────────────

class BulkLinear(nn.Module):
    """
    Standart nn.Linear yerine W = U @ V.T + bias faktörizasyonu kullanan katman.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        reg_lambda: float = 1e-4,
    ):
        super().__init__()
        rank = min(rank, in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.reg_lambda = reg_lambda

        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.V @ self.U.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def effective_weight(self) -> torch.Tensor:
        return self.U @ self.V.T

    def regularization_loss(self) -> torch.Tensor:
        """Düşük rank yapısını korumak için Frobenius norm cezası."""
        frob = self.U.pow(2).sum() + self.V.pow(2).sum()
        return self.reg_lambda * frob

    @property
    def effective_rank(self) -> float:
        with torch.no_grad():
            s = torch.linalg.svdvals(self.effective_weight())
            return (s.sum() / (s.max() + 1e-12)).item()


LowRankLinear = BulkLinear


class BulkMLP(nn.Module):
    """BulkLinear katmanlardan oluşan MLP."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        rank: int,
        reg_lambda: float = 1e-4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            BulkLinear(in_dim, hidden_dim, rank, reg_lambda=reg_lambda),
            nn.ReLU(),
            BulkLinear(hidden_dim, hidden_dim, rank, reg_lambda=reg_lambda),
            nn.ReLU(),
            BulkLinear(hidden_dim, out_dim, rank, reg_lambda=reg_lambda),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def bulk_regularization_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, BulkLinear):
                total = total + module.regularization_loss()
        return total


LowRankMLP = BulkMLP


class StandardMLP(nn.Module):
    """Karşılaştırma için tam-rank standart MLP."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. STIFF TRANSIENT EARLY STOPPING
#    |dL_val/dt| / N  →  plato tespiti ve erken durdurma
# ─────────────────────────────────────────────────────────────────────────────

class StiffTransientEarlyStopping:
    """
    Validation loss değişim hızını izler; stiff transient platosuna girildiğinde
    eğitimi durdurur.

    Kriter: |L[-1] - L[-k]| / k / N  <  stiffness_threshold
    """

    def __init__(
        self,
        k: int = 3,
        stiffness_threshold: float = 0.001,
        min_epochs: int = 5,
        min_delta: float = 1e-4,
    ):
        self.k = k
        self.stiffness_threshold = stiffness_threshold
        self.min_epochs = min_epochs
        self.min_delta = min_delta

        self.val_history: list[float] = []
        self.stiffness_history: list[float] = []
        self.stopped_epoch: int | None = None
        self.best_val_loss = float("inf")

    def step(self, epoch: int, val_loss: float) -> bool:
        """
        Yeni validation loss kaydet.
        True dönerse eğitim durdurulmalı.
        """
        self.val_history.append(val_loss)

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss

        if epoch < self.min_epochs or len(self.val_history) < self.k:
            self.stiffness_history.append(float("inf"))
            return False

        recent = self.val_history[-self.k :]
        delta = abs(recent[-1] - recent[0]) / self.k
        scaled = delta / (epoch + 1)
        self.stiffness_history.append(scaled)

        if scaled < self.stiffness_threshold:
            self.stopped_epoch = epoch
            return True
        return False

    def reset(self):
        self.val_history.clear()
        self.stiffness_history.clear()
        self.stopped_epoch = None
        self.best_val_loss = float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ZENON QUANTIZATION SCHEDULER
#    1/N yasasına göre fp32 → fp16 → int8 geçişi
# ─────────────────────────────────────────────────────────────────────────────

class ZenonQuantizationScheduler:
    """
    Eğitim adımına göre hassasiyet seviyesini kademeli düşürür.

    schedule_points : [fp32→fp16 adımı, fp16→int8 adımı]
    Zenon ölçeklemesi: geçişler step/(step+1) ile yumuşatılır.
    """

    PRECISION_ORDER = ("fp32", "fp16", "int8")

    def __init__(
        self,
        schedule_points: list[int] | None = None,
        total_steps: int = 10_000,
    ):
        if schedule_points is None:
            schedule_points = [total_steps // 3, 2 * total_steps // 3]
        self.schedule_points = sorted(schedule_points)
        self.total_steps = total_steps
        self.current_step = 0
        self.precision = "fp32"
        self.precision_history: list[str] = ["fp32"]

    def _zenon_scale(self, step: int) -> float:
        return step / (step + 1.0)

    def step(self) -> str:
        """Bir eğitim adımı ilerlet; mevcut hassasiyeti döndür."""
        self.current_step += 1
        n = self.current_step

        if len(self.schedule_points) >= 2 and n >= self.schedule_points[1]:
            threshold = self.schedule_points[1] * self._zenon_scale(n)
            if n >= threshold:
                self.precision = "int8"
        elif len(self.schedule_points) >= 1 and n >= self.schedule_points[0]:
            threshold = self.schedule_points[0] * self._zenon_scale(n)
            if n >= threshold:
                self.precision = "fp16"

        self.precision_history.append(self.precision)
        return self.precision

    def training_context(self, device_type: str = "cpu"):
        if self.precision == "fp16":
            return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        return contextlib.nullcontext()

    def reset(self):
        self.current_step = 0
        self.precision = "fp32"
        self.precision_history = ["fp32"]


def materialize_bulk_mlp(model: BulkMLP) -> StandardMLP:
    """BulkMLP'yi tam ağırlıklı StandardMLP'ye dönüştür (int8 inference için)."""
    bulk_linears = [m for m in model.net if isinstance(m, BulkLinear)]
    in_dim = bulk_linears[0].in_features
    hidden = bulk_linears[0].out_features
    out_dim = bulk_linears[-1].out_features

    std = StandardMLP(in_dim, hidden, out_dim)
    std_linears = [m for m in std.net if isinstance(m, nn.Linear)]

    for bulk, linear in zip(bulk_linears, std_linears):
        with torch.no_grad():
            linear.weight.copy_(bulk.effective_weight())
            if bulk.bias is not None and linear.bias is not None:
                linear.bias.copy_(bulk.bias)

    return std


def apply_int8_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Eğitim sonrası inference için dinamik int8 quantizasyon."""
    if isinstance(model, BulkMLP):
        model = materialize_bulk_mlp(model)

    quant_model = copy.deepcopy(model)
    quant_model.eval()
    quant_model.cpu()
    return torch.quantization.quantize_dynamic(
        quant_model,
        {nn.Linear},
        dtype=torch.qint8,
    )


def measure_inference_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    warmup: int = 20,
    repeats: int = 100,
) -> float:
    """Ortalama inference süresi (ms/batch)."""
    model.eval()
    sample_input = sample_input[: min(sample_input.shape[0], 128)]

    with torch.no_grad():
        for _ in range(warmup):
            model(sample_input)

        if sample_input.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                model(sample_input)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / repeats

        import time

        t0 = time.perf_counter()
        for _ in range(repeats):
            model(sample_input)
        elapsed = time.perf_counter() - t0
        return (elapsed / repeats) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Geriye dönük uyumluluk (eski benchmark referansları)
# ─────────────────────────────────────────────────────────────────────────────

class PowerLawScheduler:
    """Karşılaştırma için kuvvet yasası LR planlayıcısı."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        eta_0: float,
        alpha: float = 0.1,
        p: float = 0.75,
        patience: int = 5,
        min_delta: float = 1e-4,
    ):
        self.optimizer = optimizer
        self.eta_0 = eta_0
        self.alpha = alpha
        self.p = p
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float("inf")
        self._stagnation_count = 0
        self._activated = False
        self._t = 0
        self.lr_history: list[float] = [eta_0]

    def step(self, val_loss: float) -> float:
        if not self._activated:
            improved = val_loss < self._best_loss - self.min_delta
            if improved:
                self._best_loss = val_loss
                self._stagnation_count = 0
            else:
                self._stagnation_count += 1
            if self._stagnation_count >= self.patience:
                self._activated = True
                self._t = 0

        if self._activated:
            self._t += 1
            new_lr = self.eta_0 / (1.0 + self.alpha * self._t) ** self.p
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
        else:
            new_lr = self.eta_0

        self.lr_history.append(new_lr)
        return new_lr


class GeometricNoiseOptimizer:
    """Gradyanlara 1/N ölçekli gürültü ekleyen optimizer sarmalayıcısı."""

    def __init__(
        self,
        base_optimizer: optim.Optimizer,
        sigma_0: float = 0.1,
        clip_norm: float = 1.0,
        q: float = 0.5,
    ):
        self.optimizer = base_optimizer
        self.sigma_0 = sigma_0
        self.clip_norm = clip_norm
        self.q = q
        self._N = 0
        self.noise_history: list[float] = []

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._N += 1
        total_noise = 0.0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad_norm = param.grad.norm(2).item()
                clipped = min(grad_norm, self.clip_norm)
                sigma = self.sigma_0 * clipped / ((1.0 + self._N) ** self.q)
                param.grad.data.add_(torch.randn_like(param.grad) * sigma)
                total_noise += sigma
        self.noise_history.append(total_noise)
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()
