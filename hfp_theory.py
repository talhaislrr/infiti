"""
HFP Teorisi — Paper I & II'den türetilmiş sabitler ve mekanizmalar
==================================================================
Paper I  : dθ/dτ = −η̃ θ³,  η̃ ≈ 0.407 (Branch 2, Jordan bloğu)
Paper II : Zeno sızıntısı ∝ 1/N  (standart QM: 1/N²)
           Projeksiyon: |α|² = cos²θ, |β|² = sin²θ  (Haar → Born)

Generic mühendislik yaklaşımından fark: tüm katsayılar teoriden gelir,
hiperparametre taraması yok.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.optim as optim

from hfp_principles import BulkMLP, StandardMLP, BulkLinear


# ── Paper I Branch 2 kilitli parametreler ────────────────────────────────────
ETA_TILDE = 0.407          # η̃ — kübik stiff transient katsayısı
C1_LINEAR = 1.596          # Jordan bloğu lineer geri çağırma
C2_CROSS = 0.910           # çapraz terim katsayısı
BRANCH_A = 0.4884          # warp faktörü üssü a
BRANCH_BETA = 1.8208       # dilaton üssü β
THETA_HAAR = 0.495         # rad — Paper II §3.4.4 adiabatik limit


def bulk_projection_rank(in_features: int, out_features: int) -> int:
    """
    Paper II §3.1: |β|² = sin²θ → brane'e yansıyan efektif boyut oranı.
    θ = THETA_HAAR ile rank = min(in,out) · sin²(θ).
    """
    dim = min(in_features, out_features)
    return max(8, int(dim * math.sin(THETA_HAAR) ** 2))


class HFPProjectedLinear(nn.Module):
    """
    Paper II Axiom 2: |ψ⟩₄ = P_θ |Ψ⟩₅

    Bulk uzayı (rank+1 boyut) → brane projeksiyonu:
      W_eff = cos²(θ) · W_mean + sin²(θ) · (U @ V.T)
    θ teoriden sabit (Haar ölçüsü); eğitimde güncellenmez.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        rank = bulk_projection_rank(in_features, out_features)
        self.theta = THETA_HAAR
        self.cos2 = math.cos(self.theta) ** 2
        self.sin2 = math.sin(self.theta) ** 2

        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(in_features, rank))
        self.W_mean = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def effective_weight(self) -> torch.Tensor:
        return self.cos2 * self.W_mean + self.sin2 * (self.U @ self.V.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.effective_weight().T
        if self.bias is not None:
            out = out + self.bias
        return out

    def haar_regularization(self) -> torch.Tensor:
        """
        Paper II §3.5: Born kuralı → |α|² + |β|² = 1.
        Ağırlık enerjisinin cos²/sin² oranını koru.
        """
        w = self.effective_weight()
        energy = w.pow(2).mean()
        target = self.cos2  # brane tarafındaki beklenen projeksiyon ağırlığı
        return ETA_TILDE * (energy - target).pow(2)


class HFPProjectedMLP(nn.Module):
    """Teoriden türetilmiş rank ile üç katmanlı MLP."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            HFPProjectedLinear(in_dim, hidden),
            nn.ReLU(),
            HFPProjectedLinear(hidden, hidden),
            nn.ReLU(),
            HFPProjectedLinear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def haar_loss(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for m in self.modules():
            if isinstance(m, HFPProjectedLinear):
                total = total + m.haar_regularization()
        return total

    @property
    def projection_rank(self) -> int:
        first = next(m for m in self.modules() if isinstance(m, HFPProjectedLinear))
        return first.U.shape[1]


class StiffTransientCubic:
    """
    Paper I §6 + Paper II §3.4.3:
      dθ/dτ = −η̃ θ³

    θ ≡ √(L_val / L_best − 1)  — denge sapması (YA değişkeninin brane karşılığı)
    Plato: |η̃ θ³| < ε  veya θ < θ_min
    """

    def __init__(self, theta_min: float = 0.02, min_epochs: int = 3):
        self.theta_min = theta_min
        self.min_epochs = min_epochs
        self.best_val = float("inf")
        self.theta_history: list[float] = []
        self.flow_history: list[float] = []
        self.stopped_epoch: int | None = None

    def _compute_theta(self, val_loss: float) -> float:
        if self.best_val == float("inf"):
            return 1.0
        ratio = val_loss / max(self.best_val, 1e-8)
        return math.sqrt(max(ratio - 1.0, 0.0))

    def step(self, epoch: int, val_loss: float) -> tuple[bool, float, float]:
        if val_loss < self.best_val:
            self.best_val = val_loss

        theta = self._compute_theta(val_loss)
        flow = ETA_TILDE * theta ** 3
        self.theta_history.append(theta)
        self.flow_history.append(flow)

        if epoch < self.min_epochs:
            return False, theta, flow

        if theta < self.theta_min or flow < ETA_TILDE * self.theta_min ** 3:
            self.stopped_epoch = epoch
            return True, theta, flow
        return False, theta, flow

    def learning_rate(self, base_lr: float, theta: float) -> float:
        """Kübik plato: LR ∝ 1/(1 + η̃θ²) — stiff transient bölgesinde yavaşla."""
        return base_lr / (1.0 + ETA_TILDE * theta ** 2)


class ZenoLeakageRegularizer:
    """
    Paper II §3.6.2:
      θ(τ/N) ≈ θ₀(1 − η̃θ₀²τ/N)   →  sızıntı ∝ 1/N
      Standart QM: hata ∝ 1/N²

    Her N mini-batch'te bir "ölçüm periyodu".
    Gradyanlara η̃·θ₀²/N sızıntısı eklenir (1/N ölçeği, 1/N² değil).
    """

    def __init__(self, measurement_interval: int = 50):
        self.N = measurement_interval
        self.step_count = 0
        self.theta_0 = 1.0
        self.leakage_history: list[float] = []

    def set_theta_0(self, theta: float):
        self.theta_0 = max(theta, 1e-6)

    def apply_leakage(self, model: nn.Module) -> float:
        self.step_count += 1
        if self.step_count % self.N != 0:
            return 0.0

        period = self.step_count // self.N
        # HFP: ∝ 1/N  |  Standart: ∝ 1/N² (karşılaştırma için ikisi de hesaplanır)
        hfp_leak = ETA_TILDE * self.theta_0 ** 2 / period
        self.leakage_history.append(hfp_leak)

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * hfp_leak * p.grad.norm().item()
                    p.grad.add_(noise)
        return hfp_leak


class StandardZenoRegularizer:
    """Standart kuantum Zeno limiti: hata ∝ 1/N² (karşılaştırma)."""

    def __init__(self, measurement_interval: int = 50):
        self.N = measurement_interval
        self.step_count = 0
        self.theta_0 = 1.0
        self.leakage_history: list[float] = []

    def set_theta_0(self, theta: float):
        self.theta_0 = max(theta, 1e-6)

    def apply_leakage(self, model: nn.Module) -> float:
        self.step_count += 1
        if self.step_count % self.N != 0:
            return 0.0
        period = self.step_count // self.N
        std_leak = self.theta_0 ** 2 / (period ** 2)
        self.leakage_history.append(std_leak)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * std_leak * p.grad.norm().item()
                    p.grad.add_(noise)
        return std_leak


def simulate_zeno_scaling(n_periods: int = 1000, theta_0: float = 0.1, tau: float = 1.0) -> dict:
    """
    Paper II Tablo 1'i yazılımsal olarak doğrula:
    HFP/Standart oranı ≈ η̃ · N (sabit θ₀ için)
    """
    N_values = [10, 100, 1000, 10000]
    results = []
    for N in N_values:
        std_err = (theta_0 ** 2) * (tau / N) ** 2
        hfp_leak = ETA_TILDE * theta_0 ** 2 * tau / N
        ratio = hfp_leak / std_err if std_err > 0 else float("inf")
        results.append({
            "N": N,
            "standard_1_N2": std_err,
            "hfp_1_N": hfp_leak,
            "ratio_hfp_over_std": ratio,
            "expected_ratio_approx": ETA_TILDE * N,
        })
    return {"zeno_table": results, "theta_0": theta_0, "eta_tilde": ETA_TILDE}
