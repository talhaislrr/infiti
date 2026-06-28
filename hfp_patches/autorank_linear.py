"""
Yama 1: AutoRankLinear
----------------------
W = U @ V.T + bias  |  rank = min(in, out) // rank_divisor
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn


def default_rank(
    in_features: int,
    out_features: int,
    rank_divisor: int = 8,
    min_rank: int = 4,
) -> int:
    """rank = min(in,out)//divisor, küçük katmanlarda min_rank tabanı."""
    raw = min(in_features, out_features) // rank_divisor
    return max(min_rank, raw, 1)


def count_linear_params(in_features: int, out_features: int, rank: int, bias: bool = True) -> int:
    n = rank * (in_features + out_features)
    if bias:
        n += out_features
    return n


class AutoRankLinear(nn.Module):
    """Düşük ranklı Linear: W ≈ U @ V.T"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int | None = None,
        bias: bool = True,
        rank_divisor: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank_divisor = rank_divisor
        self.rank = rank or default_rank(in_features, out_features, rank_divisor)
        self.rank = min(self.rank, in_features, out_features)

        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        self.V = nn.Parameter(torch.empty(in_features, self.rank))
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

    @property
    def num_params(self) -> int:
        return count_linear_params(self.in_features, self.out_features, self.rank, self.bias is not None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int | None = None, rank_divisor: int = 8) -> "AutoRankLinear":
        """SVD ile mevcut Linear ağırlıklarını düşük ranka projekte et."""
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"nn.Linear bekleniyor, alındı: {type(linear)}")

        layer = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            rank_divisor=rank_divisor,
        )

        W = linear.weight.data.detach().float()
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(layer.rank, S.numel())
            sqrt_s = S[:r].sqrt()
            layer.U[:, :r].copy_(U[:, :r] * sqrt_s.unsqueeze(0))
            layer.V[:, :r].copy_(Vh[:r, :].T * sqrt_s.unsqueeze(0))
            if layer.rank > r:
                layer.U[:, r:].zero_()
                layer.V[:, r:].zero_()
            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias)

        return layer

    def reconstruction_error(self, target_weight: torch.Tensor) -> float:
        with torch.no_grad():
            err = (self.effective_weight() - target_weight).pow(2).mean().sqrt()
        return err.item()


def replace_linears(
    module: nn.Module,
    rank_divisor: int = 8,
    copy_weights: bool = True,
) -> dict[str, int]:
    """
    Modül ağacındaki nn.Linear katmanlarını AutoRankLinear ile değiştir (yerinde).

    copy_weights=True  → SVD ile mevcut ağırlıkları aktar (fine-tune / sıkıştırma)
    copy_weights=False → taze init (sıfırdan eğitim)
    """
    stats: dict[str, int] = {}

    def _walk(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                old_p = child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
                if copy_weights:
                    new_layer = AutoRankLinear.from_linear(child, rank_divisor=rank_divisor)
                else:
                    new_layer = AutoRankLinear(
                        child.in_features, child.out_features, rank_divisor=rank_divisor,
                        bias=child.bias is not None,
                    )
                new_p = new_layer.num_params
                setattr(parent, name, new_layer)
                if old_p > 0:
                    stats[path] = round(100 * (1 - new_p / old_p))
            else:
                _walk(child, path)

    _walk(module)
    return stats


def iter_linears(module: nn.Module) -> Iterator[nn.Linear | AutoRankLinear]:
    for m in module.modules():
        if isinstance(m, (nn.Linear, AutoRankLinear)):
            yield m
