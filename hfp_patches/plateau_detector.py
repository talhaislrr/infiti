"""
Yama 2: PlateauDetector
-----------------------
Val-loss platosu: |L_now - L_past| / (epoch_diff ** p) < threshold → dur
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field


@dataclass
class PlateauDetector:
    k: int = 5
    p: float = 1.0
    threshold: float = 1e-4
    min_epochs: int = 3

    val_history: list[float] = field(default_factory=list)
    rate_history: list[float] = field(default_factory=list)
    stopped_epoch: int | None = None
    best_val_loss: float = float("inf")

    def reset(self) -> None:
        self.val_history.clear()
        self.rate_history.clear()
        self.stopped_epoch = None
        self.best_val_loss = float("inf")

    def loss_change_rate(self, current_loss: float, past_loss: float, epoch_diff: int) -> float:
        if epoch_diff <= 0:
            return float("inf")
        return abs(current_loss - past_loss) / (epoch_diff ** self.p)

    def step(self, epoch: int, val_loss: float) -> bool:
        """
        True → eğitimi durdur.
        """
        self.val_history.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        if epoch < self.min_epochs or len(self.val_history) <= self.k:
            self.rate_history.append(float("inf"))
            return False

        past_loss = self.val_history[-1 - self.k]
        rate = self.loss_change_rate(val_loss, past_loss, self.k)
        self.rate_history.append(rate)

        if rate < self.threshold:
            self.stopped_epoch = epoch
            return True
        return False


# PyTorch Lightning (opsiyonel)
try:
    from pytorch_lightning.callbacks import Callback

    class PlateauDetectorCallback(Callback):
        """Lightning entegrasyonu."""

        def __init__(self, k: int = 5, p: float = 1.0, threshold: float = 1e-4, min_epochs: int = 3):
            super().__init__()
            self.detector = PlateauDetector(k=k, p=p, threshold=threshold, min_epochs=min_epochs)

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            metrics = trainer.callback_metrics
            val_loss = metrics.get("val_loss") or metrics.get("validation_loss")
            if val_loss is None:
                return
            val = float(val_loss)
            epoch = trainer.current_epoch + 1
            if self.detector.step(epoch, val):
                trainer.should_stop = True

except ImportError:

    class PlateauDetectorCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PlateauDetectorCallback için: pip install pytorch-lightning")
