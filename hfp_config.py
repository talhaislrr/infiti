"""
HFP Tasarım Prensipleri — Esnek, test edilebilir hiperparametre mimarisi
========================================================================
Fiziksel sabitler (η̃=0.407 vb.) kodda YOK. HFP'nin ilkesel davranışları
(kuvvet yasası plato, adaptif niceleme, bulk projeksiyon) hiperparametre
olarak ifade edilir ve Optuna ile optimize edilebilir.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hfp_principles import BulkMLP, StandardMLP, apply_int8_dynamic_quantization, measure_inference_latency


@dataclass
class HFPConfig:
    """Tüm HFP prensiplerini tek yapılandırma nesnesinde birleştirir."""

    bulk_rank: int = 128
    bulk_reg_lambda: float = 1e-4
    stiffness_p: float = 1.0
    stiffness_threshold: float = 0.001
    stiffness_k: int = 3
    stiffness_min_epochs: int = 5
    zenon_schedule_points: list[float] = field(default_factory=lambda: [0.7, 0.9])
    zenon_grad_threshold: float = 1e-5
    initial_lr: float = 1e-3
    max_epochs: int = 30
    use_bulk: bool = True
    use_stiff: bool = True
    use_zenon: bool = False
    measure_int8: bool = False
    early_stop: bool = True

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "HFPConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def default(cls) -> "HFPConfig":
        return cls()

    @classmethod
    def fast(cls) -> "HFPConfig":
        """Hızlı eğitim: agresif stiff, zenon kapalı."""
        return cls(
            bulk_rank=128,
            stiffness_p=1.5,
            stiffness_threshold=0.002,
            use_zenon=False,
            early_stop=True,
        )

    @classmethod
    def efficient(cls) -> "HFPConfig":
        """Maliyet odaklı: düşük rank + stiff + zenon."""
        return cls(
            bulk_rank=64,
            stiffness_p=1.0,
            stiffness_threshold=0.001,
            zenon_schedule_points=[0.6, 0.85],
            zenon_grad_threshold=1e-4,
            use_zenon=True,
        )


class HFPStiffTransientScheduler:
    """
    Stiff transient prensibi — kuvvet yasası LR + plato erken durdurma.

    Plato tespiti: son k epoch'ta |dL/dt| / epoch < threshold
    LR düşürme (aktif olunca): lr = lr₀ / (1 + stiffness_p · t)^stiffness_p
    Erken dur: plato + LR zaten düşürülmüşse
    """

    def __init__(self, optimizer: optim.Optimizer, config: HFPConfig):
        self.optimizer = optimizer
        self.config = config
        self.base_lr = config.initial_lr
        self.val_history: list[float] = []
        self.lr_history: list[float] = [config.initial_lr]
        self.loss_rate_history: list[float] = []
        self._plateau_active = False
        self._plateau_steps = 0
        self.best_val_loss = float("inf")
        self.stopped_epoch: int | None = None

    def _loss_change_rate(self, epoch: int) -> float:
        k = self.config.stiffness_k
        if len(self.val_history) < k:
            return float("inf")
        recent = self.val_history[-k:]
        delta = abs(recent[-1] - recent[0]) / k
        return delta / (epoch + 1)

    def step(self, epoch: int, val_loss: float) -> tuple[float, bool]:
        """
        Returns: (current_lr, should_stop)
        """
        self.val_history.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        rate = self._loss_change_rate(epoch)
        self.loss_rate_history.append(rate)

        if epoch >= self.config.stiffness_min_epochs and rate < self.config.stiffness_threshold:
            self._plateau_active = True

        if self._plateau_active:
            self._plateau_steps += 1
            p = self.config.stiffness_p
            new_lr = self.base_lr / (1.0 + p * self._plateau_steps) ** p
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr
        else:
            new_lr = self.base_lr

        self.lr_history.append(new_lr)

        should_stop = False
        if (
            self.config.early_stop
            and self._plateau_active
            and epoch >= self.config.stiffness_min_epochs
            and rate < self.config.stiffness_threshold
        ):
            should_stop = True
            self.stopped_epoch = epoch

        return new_lr, should_stop


class HFPZenonQuantizationScheduler:
    """
    Zenon prensibi — eğitim ilerledikçe adaptif hassasiyet düşürme.

    schedule_points: toplam adımın yüzdesi [0.7, 0.9] → fp16, int8
    Gradyan büyüklüğü zenon_grad_threshold üstündeyse geçiş geciktirilir.
    """

    PRECISIONS = ("fp32", "fp16", "int8")

    def __init__(self, config: HFPConfig, total_steps: int):
        self.config = config
        self.total_steps = max(total_steps, 1)
        self.current_step = 0
        self.precision = "fp32"
        self.precision_history: list[str] = []
        self._fp16_ready = False
        self._int8_ready = False
        self._pending_grad_norm = 0.0

    def record_grad_norm(self, grad_norm: float):
        self._pending_grad_norm = grad_norm

    def _target_precision(self) -> str:
        progress = self.current_step / self.total_steps
        pts = sorted(self.config.zenon_schedule_points)
        if len(pts) >= 2 and progress >= pts[1]:
            return "int8"
        if len(pts) >= 1 and progress >= pts[0]:
            return "fp16"
        return "fp32"

    def step(self) -> str:
        self.current_step += 1
        target = self._target_precision()
        grad_ok = self._pending_grad_norm <= self.config.zenon_grad_threshold or self._pending_grad_norm == 0.0

        if target == "fp16" and not grad_ok:
            target = "fp32"
        elif target == "int8" and not grad_ok:
            target = "fp16" if self._fp16_ready else "fp32"

        if target in ("fp16", "int8"):
            self._fp16_ready = True
        if target == "int8":
            self._int8_ready = True

        self.precision = target
        self.precision_history.append(target)
        return self.precision

    def training_context(self, device_type: str = "cpu"):
        if self.precision == "fp16":
            return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        return contextlib.nullcontext()


def build_hfp_model(
    in_dim: int,
    hidden: int,
    out_dim: int,
    config: HFPConfig,
) -> nn.Module:
    if config.use_bulk:
        return BulkMLP(in_dim, hidden, out_dim, rank=config.bulk_rank, reg_lambda=config.bulk_reg_lambda)
    return StandardMLP(in_dim, hidden, out_dim)


def _batch_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total ** 0.5


@dataclass
class HFPTrainResult:
    test_acc: float
    test_loss: float
    val_acc: float
    val_loss: float
    params: int
    epochs_run: int
    elapsed_sec: float
    inference_ms: float
    int8_inference_ms: float | None
    lr_history: list[float]
    precision_history: list[str]
    loss_rate_history: list[float]
    config: dict[str, Any]


def train_with_hfp(
    model: nn.Module,
    config: HFPConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None = None,
    device: torch.device | None = None,
) -> HFPTrainResult:
    """
    HFPConfig ile modeli baştan sona eğit.
    BulkLinear + StiffTransient + Zenon entegrasyonu tek fonksiyonda.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)

    total_steps = len(train_loader) * config.max_epochs
    stiff = HFPStiffTransientScheduler(optimizer, config) if config.use_stiff else None
    zenon = HFPZenonQuantizationScheduler(config, total_steps) if config.use_zenon else None

    lr_history: list[float] = []
    precision_history: list[str] = []
    loss_rate_history: list[float] = []
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_run = 0

    t0 = time.time()
    device_type = device.type

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            ctx = contextlib.nullcontext()
            if zenon is not None:
                precision_history.append(zenon.step())
                ctx = zenon.training_context(device_type)

            with ctx:
                out = model(x)
                loss = criterion(out, y)
                if isinstance(model, BulkMLP):
                    loss = loss + model.bulk_regularization_loss()

            loss.backward()

            if zenon is not None:
                zenon.record_grad_norm(_batch_grad_norm(model))

            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        epochs_run = epoch

        should_stop = False
        if stiff is not None:
            lr, should_stop = stiff.step(epoch, val_loss)
            lr_history.append(lr)
            if stiff.loss_rate_history:
                loss_rate_history.append(stiff.loss_rate_history[-1])
        else:
            lr_history.append(config.initial_lr)

        if should_stop:
            break

    if test_loader is None:
        test_loss, test_acc = val_loss, val_acc
    else:
        test_loss, test_acc = evaluate(model, test_loader, device)

    elapsed = time.time() - t0

    sample = next(iter(test_loader or val_loader))[0].to(device)
    inference_ms = measure_inference_latency(model, sample)

    int8_ms = None
    if config.measure_int8:
        try:
            int8_model = apply_int8_dynamic_quantization(model)
            int8_ms = measure_inference_latency(int8_model, sample.cpu())
        except Exception:
            pass

    return HFPTrainResult(
        test_acc=test_acc,
        test_loss=test_loss,
        val_acc=best_val_acc,
        val_loss=best_val_loss,
        params=params,
        epochs_run=epochs_run,
        elapsed_sec=round(elapsed, 2),
        inference_ms=round(inference_ms, 3),
        int8_inference_ms=round(int8_ms, 3) if int8_ms else None,
        lr_history=lr_history,
        precision_history=precision_history,
        loss_rate_history=loss_rate_history,
        config=asdict(config),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += crit(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


def optuna_objective(
    trial: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 15,
) -> float:
    """
    Optuna trial → HFP hiperparametre araması.
    Amaç: validation accuracy maksimize (Optuna minimize eder → negatif döndür).
    """
    config = HFPConfig(
        bulk_rank=trial.suggest_categorical("bulk_rank", [32, 64, 128, 256]),
        stiffness_p=trial.suggest_float("stiffness_p", 0.5, 2.0),
        stiffness_threshold=trial.suggest_float("stiffness_threshold", 1e-4, 5e-3, log=True),
        zenon_schedule_points=[
            trial.suggest_float("zenon_p1", 0.5, 0.8),
            trial.suggest_float("zenon_p2", 0.8, 0.95),
        ],
        zenon_grad_threshold=trial.suggest_float("zenon_grad_threshold", 1e-6, 1e-3, log=True),
        initial_lr=trial.suggest_float("initial_lr", 1e-4, 3e-3, log=True),
        max_epochs=max_epochs,
        use_bulk=True,
        use_stiff=True,
        use_zenon=trial.suggest_categorical("use_zenon", [True, False]),
    )

    model = build_hfp_model(784, 512, 10, config)
    result = train_with_hfp(model, config, train_loader, val_loader, test_loader, device)
    return -result.val_acc
