"""Fitting Chinchilla laws and shared-exponent optimizer-aware models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .laws import ChinchillaParams, OptimizerRho, chinchilla_loss


@dataclass
class FitResult:
    params: ChinchillaParams
    rmse: float
    n_points: int
    method: str
    optimizer: str | None = None
    rho_n: float = 1.0
    rho_d: float = 1.0
    ci: dict[str, tuple[float, float]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.as_dict(),
            "rmse": self.rmse,
            "n_points": self.n_points,
            "method": self.method,
            "optimizer": self.optimizer,
            "rho_n": self.rho_n,
            "rho_d": self.rho_d,
            "ci": self.ci,
            "extras": self.extras,
        }


def _huber(residual: np.ndarray, delta: float = 0.01) -> np.ndarray:
    abs_r = np.abs(residual)
    return np.where(abs_r <= delta, 0.5 * residual**2, delta * (abs_r - 0.5 * delta))


def simulate_isoflop_data(
    params: ChinchillaParams | None = None,
    compute_budgets: np.ndarray | None = None,
    n_points_per_budget: int = 12,
    noise_std: float = 0.0,
    optimizer: str = "adamw",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Synthetic IsoFLOP dataset for tutorials and unit tests."""
    params = params or ChinchillaParams()
    rho = OptimizerRho.from_name(optimizer)
    rng = np.random.default_rng(seed)
    if compute_budgets is None:
        compute_budgets = np.geomspace(1e19, 1e22, 6)
    ns, ds, ls, cs, opts = [], [], [], [], []
    for c in compute_budgets:
        # Spread N around the ridge
        from .allocate import allocate_for_budget

        center = allocate_for_budget(float(c), optimizer=optimizer, params=params)["N"]
        n_grid = np.geomspace(center / 8.0, center * 8.0, n_points_per_budget)
        d_grid = float(c) / (6.0 * n_grid)
        loss = np.asarray(chinchilla_loss(n_grid, d_grid, params, rho), dtype=float)
        if noise_std > 0:
            loss = loss + rng.normal(0.0, noise_std, size=loss.shape)
        ns.append(n_grid)
        ds.append(d_grid)
        ls.append(loss)
        cs.append(np.full_like(n_grid, c))
        opts.extend([optimizer] * len(n_grid))
    return {
        "N": np.concatenate(ns),
        "D": np.concatenate(ds),
        "L": np.concatenate(ls),
        "C": np.concatenate(cs),
        "optimizer": np.array(opts),
    }


def fit_chinchilla(
    n: np.ndarray,
    d: np.ndarray,
    loss: np.ndarray,
    method: str = "lbfgs",
    huber_delta: float = 0.01,
    bootstrap: int = 0,
    seed: int = 0,
    optimizer: str | None = None,
) -> FitResult:
    """Fit classic 5-parameter Chinchilla law to (N, D, L)."""
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    loss = np.asarray(loss, dtype=float)
    mask = np.isfinite(n) & np.isfinite(d) & np.isfinite(loss) & (n > 0) & (d > 0)
    n, d, loss = n[mask], d[mask], loss[mask]

    def pack(x: np.ndarray) -> ChinchillaParams:
        # Optimize in log-space for A,B,E positivity-ish
        return ChinchillaParams(
            E=float(np.exp(x[0])),
            A=float(np.exp(x[1])),
            B=float(np.exp(x[2])),
            alpha=float(np.clip(x[3], 0.05, 1.5)),
            beta=float(np.clip(x[4], 0.05, 1.5)),
        )

    def objective(x: np.ndarray) -> float:
        p = pack(x)
        pred = np.asarray(chinchilla_loss(n, d, p), dtype=float)
        return float(np.mean(_huber(pred - loss, huber_delta)))

    x0 = np.array(
        [
            np.log(1.69),
            np.log(406.4),
            np.log(410.7),
            0.34,
            0.28,
        ],
        dtype=float,
    )
    if method == "grid":
        # Coarse grid on exponents then LBFGS
        best = None
        best_val = np.inf
        for a in np.linspace(0.2, 0.5, 7):
            for b in np.linspace(0.2, 0.5, 7):
                x = x0.copy()
                x[3], x[4] = a, b
                val = objective(x)
                if val < best_val:
                    best_val, best = val, x
        x0 = best if best is not None else x0

    result = minimize(objective, x0, method="L-BFGS-B")
    fitted = pack(result.x)
    pred = np.asarray(chinchilla_loss(n, d, fitted), dtype=float)
    rmse = float(np.sqrt(np.mean((pred - loss) ** 2)))

    ci = None
    if bootstrap > 0 and len(n) >= 8:
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(n), size=len(n))
            sub = fit_chinchilla(n[idx], d[idx], loss[idx], method="lbfgs", bootstrap=0)
            samples.append(sub.params.as_dict())
        ci = {}
        for key in ["E", "A", "B", "alpha", "beta"]:
            vals = np.array([s[key] for s in samples])
            ci[key] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return FitResult(
        params=fitted,
        rmse=rmse,
        n_points=len(n),
        method=method,
        optimizer=optimizer,
        ci=ci,
        extras={"success": bool(result.success), "fun": float(result.fun)},
    )


def fit_shared_exponents(
    runs: dict[str, dict[str, np.ndarray]],
    reference: str = "adamw",
    huber_delta: float = 0.01,
    bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit shared α,β with per-optimizer ρ_N, ρ_D (AdamW fixed at 1).

    runs: {optimizer: {"N", "D", "L"}}
    """
    if reference not in runs:
        raise KeyError(f"reference optimizer {reference!r} missing from runs")

    opt_names = list(runs.keys())
    # Parameters: logE, logA, logB, alpha, beta, then for each non-ref: logρN, logρD
    non_ref = [o for o in opt_names if o != reference]
    n_extra = 2 * len(non_ref)

    def unpack(x: np.ndarray) -> tuple[ChinchillaParams, dict[str, OptimizerRho]]:
        params = ChinchillaParams(
            E=float(np.exp(x[0])),
            A=float(np.exp(x[1])),
            B=float(np.exp(x[2])),
            alpha=float(np.clip(x[3], 0.05, 1.5)),
            beta=float(np.clip(x[4], 0.05, 1.5)),
        )
        rhos = {reference: OptimizerRho(reference, 1.0, 1.0)}
        for i, name in enumerate(non_ref):
            rhos[name] = OptimizerRho(
                name,
                rho_n=float(np.exp(x[5 + 2 * i])),
                rho_d=float(np.exp(x[5 + 2 * i + 1])),
            )
        return params, rhos

    def objective(x: np.ndarray) -> float:
        params, rhos = unpack(x)
        total = 0.0
        count = 0
        for name, data in runs.items():
            pred = np.asarray(
                chinchilla_loss(data["N"], data["D"], params, rhos[name]),
                dtype=float,
            )
            total += float(np.sum(_huber(pred - np.asarray(data["L"], dtype=float), huber_delta)))
            count += len(data["L"])
        return total / max(count, 1)

    x0 = np.zeros(5 + n_extra, dtype=float)
    x0[0:5] = [np.log(1.69), np.log(406.4), np.log(410.7), 0.34, 0.28]
    for i, name in enumerate(non_ref):
        try:
            prior = OptimizerRho.from_name(name)
            x0[5 + 2 * i] = np.log(prior.rho_n)
            x0[5 + 2 * i + 1] = np.log(prior.rho_d)
        except KeyError:
            x0[5 + 2 * i] = 0.0
            x0[5 + 2 * i + 1] = 0.0

    result = minimize(objective, x0, method="L-BFGS-B")
    params, rhos = unpack(result.x)

    per_opt_rmse = {}
    for name, data in runs.items():
        pred = np.asarray(chinchilla_loss(data["N"], data["D"], params, rhos[name]), dtype=float)
        per_opt_rmse[name] = float(np.sqrt(np.mean((pred - data["L"]) ** 2)))

    out: dict[str, Any] = {
        "params": params.as_dict(),
        "rhos": {k: v.as_dict() for k, v in rhos.items()},
        "per_optimizer_rmse": per_opt_rmse,
        "mean_rmse": float(np.mean(list(per_opt_rmse.values()))),
        "success": bool(result.success),
        "method": "shared_exponents",
    }

    n_total = sum(len(data["L"]) for data in runs.values())
    if bootstrap > 0 and n_total >= 8:
        rng = np.random.default_rng(seed)
        param_samples: list[dict[str, float]] = []
        rho_samples: dict[str, list[tuple[float, float]]] = {k: [] for k in rhos}
        for _ in range(bootstrap):
            boot_runs: dict[str, dict[str, np.ndarray]] = {}
            for name, data in runs.items():
                n_pts = len(data["L"])
                idx = rng.integers(0, n_pts, size=n_pts)
                boot_runs[name] = {
                    "N": np.asarray(data["N"], dtype=float)[idx],
                    "D": np.asarray(data["D"], dtype=float)[idx],
                    "L": np.asarray(data["L"], dtype=float)[idx],
                }
            sub = fit_shared_exponents(
                boot_runs, reference=reference, huber_delta=huber_delta, bootstrap=0
            )
            param_samples.append(sub["params"])
            for name, meta in sub["rhos"].items():
                rho_samples[name].append((meta["rho_n"], meta["rho_d"]))
        ci: dict[str, Any] = {}
        for key in ["E", "A", "B", "alpha", "beta"]:
            vals = np.array([s[key] for s in param_samples])
            ci[key] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        rho_ci = {}
        for name, pairs in rho_samples.items():
            rn = np.array([p[0] for p in pairs])
            rd = np.array([p[1] for p in pairs])
            rho_ci[name] = {
                "rho_n": (float(np.percentile(rn, 2.5)), float(np.percentile(rn, 97.5))),
                "rho_d": (float(np.percentile(rd, 2.5)), float(np.percentile(rd, 97.5))),
            }
        out["ci"] = ci
        out["rho_ci"] = rho_ci

    return out


def compare_fit_strategies(
    runs: dict[str, dict[str, np.ndarray]],
    reference: str = "adamw",
    bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Side-by-side: separate per-optimizer fits vs shared-exponent robust fit."""
    separate = {}
    for name, data in runs.items():
        separate[name] = fit_chinchilla(
            data["N"],
            data["D"],
            data["L"],
            optimizer=name,
            bootstrap=bootstrap,
            seed=seed,
        ).as_dict()
    shared = fit_shared_exponents(
        runs, reference=reference, bootstrap=bootstrap, seed=seed
    )
    # Parameter variance across separate fits (ill-conditioning signal)
    alphas = [separate[k]["params"]["alpha"] for k in separate]
    betas = [separate[k]["params"]["beta"] for k in separate]
    return {
        "separate": separate,
        "shared": shared,
        "separate_alpha_std": float(np.std(alphas)),
        "separate_beta_std": float(np.std(betas)),
        "note": (
            "High separate_*_std indicates ill-conditioned per-optimizer Chinchilla fits; "
            "shared-exponent ρ model is typically more stable (Volkova et al., 2026)."
        ),
    }
