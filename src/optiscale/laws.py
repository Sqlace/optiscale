"""Chinchilla and optimizer-aware scaling law primitives.

Classic Chinchilla (Hoffmann et al., 2022):
    L(N, D) = E + A / N^α + B / D^β
    C ≈ 6 N D

Optimizer-aware extension (inspired by Volkova et al., 2026):
    shared exponents α, β with optimizer-specific rescaling ρ_N, ρ_D
    relative to AdamW reference:
        N_eff = ρ_N * N,  D_eff = ρ_D * D
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Hoffmann et al. Approach 3 reference coefficients (natural log loss scale).
CHINCHILLA_DEFAULT = {
    "E": 1.69,
    "A": 406.4,
    "B": 410.7,
    "alpha": 0.34,
    "beta": 0.28,
}

# Approximate ρ relative to AdamW. Tunable defaults for planning — not fitted
# on a single public multi-B sweep. Override freely for your own fits.
OPTIMIZER_RHOS: dict[str, dict[str, float]] = {
    "adamw": {"rho_n": 1.0, "rho_d": 1.0, "label": "AdamW"},
    "muon": {"rho_n": 1.35, "rho_d": 1.25, "label": "Muon"},
    "normuon": {"rho_n": 1.45, "rho_d": 1.30, "label": "NorMuon"},
    "aurora": {"rho_n": 1.40, "rho_d": 1.28, "label": "Aurora"},
    "soap": {"rho_n": 1.20, "rho_d": 1.15, "label": "SOAP"},
    "shampoo": {"rho_n": 1.18, "rho_d": 1.12, "label": "Shampoo"},
    "lion": {"rho_n": 1.05, "rho_d": 1.02, "label": "Lion"},
}


@dataclass(frozen=True)
class ChinchillaParams:
    E: float = CHINCHILLA_DEFAULT["E"]
    A: float = CHINCHILLA_DEFAULT["A"]
    B: float = CHINCHILLA_DEFAULT["B"]
    alpha: float = CHINCHILLA_DEFAULT["alpha"]
    beta: float = CHINCHILLA_DEFAULT["beta"]

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizerRho:
    name: str
    rho_n: float = 1.0
    rho_d: float = 1.0
    label: str | None = None

    @classmethod
    def from_name(cls, name: str) -> "OptimizerRho":
        key = name.lower().strip()
        if key not in OPTIMIZER_RHOS:
            known = ", ".join(sorted(OPTIMIZER_RHOS))
            raise KeyError(f"Unknown optimizer {name!r}. Known: {known}")
        meta = OPTIMIZER_RHOS[key]
        return cls(name=key, rho_n=meta["rho_n"], rho_d=meta["rho_d"], label=meta["label"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rho_n": self.rho_n,
            "rho_d": self.rho_d,
            "label": self.label or self.name,
        }


def list_optimizers() -> list[dict[str, Any]]:
    return [OptimizerRho.from_name(k).as_dict() for k in sorted(OPTIMIZER_RHOS)]


def flops(n: float, d: float) -> float:
    """Approximate training FLOPs C ≈ 6ND."""
    return 6.0 * float(n) * float(d)


def effective_params(
    n: float,
    d: float,
    rho: OptimizerRho | None = None,
) -> tuple[float, float]:
    rho = rho or OptimizerRho.from_name("adamw")
    return float(n) * rho.rho_n, float(d) * rho.rho_d


def chinchilla_loss(
    n: float | np.ndarray,
    d: float | np.ndarray,
    params: ChinchillaParams | None = None,
    rho: OptimizerRho | None = None,
) -> float | np.ndarray:
    """Predict loss under classic or optimizer-aware Chinchilla."""
    params = params or ChinchillaParams()
    rho = rho or OptimizerRho.from_name("adamw")
    n_eff = np.asarray(n, dtype=float) * rho.rho_n
    d_eff = np.asarray(d, dtype=float) * rho.rho_d
    loss = params.E + params.A / np.power(n_eff, params.alpha) + params.B / np.power(
        d_eff, params.beta
    )
    if np.isscalar(n) and np.isscalar(d):
        return float(loss)
    return loss


def tokens_per_param(n: float, d: float) -> float:
    return float(d) / float(n)


def format_params(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:.0f}"


def format_flops(c: float) -> str:
    if c >= 1e24:
        return f"{c / 1e24:.3f}e24"
    if c >= 1e21:
        return f"{c / 1e21:.3f} ZFLOPs"
    if c >= 1e18:
        return f"{c / 1e18:.3f} EFLOPs"
    return f"{c:.3e}"
