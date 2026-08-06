"""Allocation solvers: budget → (N*, D*), inverse targets, isoFLOP curves."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from .laws import (
    ChinchillaParams,
    OptimizerRho,
    chinchilla_loss,
    flops,
    format_flops,
    format_params,
    tokens_per_param,
)


def _optimal_nd_closed_form(
    compute: float,
    params: ChinchillaParams,
    rho: OptimizerRho,
) -> tuple[float, float]:
    """Closed-form compute-optimal N*, D* under C = 6ND with optimizer ρ.

    Minimize L = E + A/(ρ_N N)^α + B/(ρ_D D)^β subject to C = 6 N D.
    Solution:
        N* = G · (ρ_D^β / ρ_N^α)^{1/(α+β)} · (C/6)^{β/(α+β)}
        D* = C / (6 N*)
    with G = ((A α)/(B β))^{1/(α+β)}.
    """
    a, b = params.alpha, params.beta
    g_star = ((params.A * a) / (params.B * b)) ** (1.0 / (a + b))
    rho_pref = (rho.rho_d**b / rho.rho_n**a) ** (1.0 / (a + b))
    n = g_star * rho_pref * (compute / 6.0) ** (b / (a + b))
    d = compute / (6.0 * n)
    return float(n), float(d)


def allocate_for_budget(
    compute: float,
    optimizer: str = "adamw",
    params: ChinchillaParams | None = None,
    rho_n: float | None = None,
    rho_d: float | None = None,
    active_frac: float | None = None,
) -> dict[str, Any]:
    """Map FLOP budget C → optimal (N*, D*) for an optimizer.

    If ``active_frac`` is set (MoE), FLOPs obey C ≈ 6 · (active_frac · N) · D
    while loss depends on total params N. Equivalent to dense allocate at
    effective compute C / active_frac.
    """
    if compute <= 0:
        raise ValueError("compute must be positive")
    params = params or ChinchillaParams()
    rho = _resolve_rho(optimizer, rho_n, rho_d)
    frac = 1.0 if active_frac is None else float(active_frac)
    if not 0.0 < frac <= 1.0:
        raise ValueError("active_frac must be in (0, 1]")
    # Dense ridge on effective FLOPs C/frac → N_total*, D*
    n, d = _optimal_nd_closed_form(compute / frac, params, rho)
    loss = float(chinchilla_loss(n, d, params, rho))
    out = {
        "optimizer": rho.name,
        "label": rho.label or rho.name,
        "rho_n": rho.rho_n,
        "rho_d": rho.rho_d,
        "compute": compute,
        "N": n,
        "D": d,
        "tokens_per_param": tokens_per_param(n, d),
        "loss": loss,
        "N_human": format_params(n),
        "D_human": format_params(d),
        "compute_human": format_flops(compute),
    }
    if active_frac is not None:
        out["active_frac"] = frac
        out["N_active"] = n * frac
        out["N_active_human"] = format_params(n * frac)
    return out


def _resolve_rho(
    optimizer: str,
    rho_n: float | None = None,
    rho_d: float | None = None,
) -> OptimizerRho:
    rho = OptimizerRho.from_name(optimizer)
    if rho_n is not None or rho_d is not None:
        rho = OptimizerRho(
            name=rho.name,
            rho_n=rho_n if rho_n is not None else rho.rho_n,
            rho_d=rho_d if rho_d is not None else rho.rho_d,
            label=rho.label,
        )
    return rho


def allocate_fixed_n(
    n: float,
    compute: float | None = None,
    target_loss: float | None = None,
    optimizer: str = "adamw",
    params: ChinchillaParams | None = None,
    rho_n: float | None = None,
    rho_d: float | None = None,
) -> dict[str, Any]:
    """Fix model size N; solve for D given compute or target loss."""
    params = params or ChinchillaParams()
    rho = _resolve_rho(optimizer, rho_n, rho_d)
    if compute is not None:
        d = compute / (6.0 * n)
    elif target_loss is not None:
        # E + A/(ρN N)^α + B/(ρD D)^β = L  →  solve for D
        n_eff = n * rho.rho_n
        residual = target_loss - params.E - params.A / (n_eff**params.alpha)
        if residual <= 0:
            raise ValueError("target_loss unreachable for this N (below model floor)")
        d_eff = (params.B / residual) ** (1.0 / params.beta)
        d = d_eff / rho.rho_d
        compute = flops(n, d)
    else:
        raise ValueError("Provide compute or target_loss")
    loss = float(chinchilla_loss(n, d, params, rho))
    return {
        "optimizer": rho.name,
        "N": float(n),
        "D": float(d),
        "compute": float(compute),
        "loss": loss,
        "tokens_per_param": tokens_per_param(n, d),
        "rho_n": rho.rho_n,
        "rho_d": rho.rho_d,
        "N_human": format_params(n),
        "D_human": format_params(d),
        "compute_human": format_flops(float(compute)),
    }


def allocate_fixed_ratio(
    compute: float,
    tokens_per_param_ratio: float,
    optimizer: str = "adamw",
    params: ChinchillaParams | None = None,
    rho_n: float | None = None,
    rho_d: float | None = None,
) -> dict[str, Any]:
    """Allocate under fixed D/N = r (e.g. overtrain at 100 tok/param)."""
    if tokens_per_param_ratio <= 0:
        raise ValueError("ratio must be positive")
    # C = 6 N (r N) = 6 r N² → N = sqrt(C / (6r))
    n = np.sqrt(compute / (6.0 * tokens_per_param_ratio))
    d = tokens_per_param_ratio * n
    params = params or ChinchillaParams()
    rho = _resolve_rho(optimizer, rho_n, rho_d)
    loss = float(chinchilla_loss(n, d, params, rho))
    return {
        "optimizer": rho.name,
        "N": float(n),
        "D": float(d),
        "compute": float(compute),
        "loss": loss,
        "tokens_per_param": float(tokens_per_param_ratio),
        "constraint": "fixed_ratio",
        "rho_n": rho.rho_n,
        "rho_d": rho.rho_d,
        "N_human": format_params(float(n)),
        "D_human": format_params(float(d)),
        "compute_human": format_flops(compute),
    }


def compute_for_target_loss(
    target_loss: float,
    optimizer: str = "adamw",
    params: ChinchillaParams | None = None,
    rho_n: float | None = None,
    rho_d: float | None = None,
    c_min: float = 1e18,
    c_max: float = 1e28,
) -> dict[str, Any]:
    """Minimum compute to reach a target loss on the compute-optimal ridge."""
    params = params or ChinchillaParams()
    if target_loss <= params.E:
        raise ValueError(
            f"target_loss={target_loss} is at or below irreducible loss E={params.E}"
        )

    def objective(log_c: float) -> float:
        c = float(np.exp(log_c))
        alloc = allocate_for_budget(
            c, optimizer=optimizer, params=params, rho_n=rho_n, rho_d=rho_d
        )
        return (alloc["loss"] - target_loss) ** 2

    result = minimize_scalar(
        objective,
        bounds=(np.log(c_min), np.log(c_max)),
        method="bounded",
    )
    compute = float(np.exp(result.x))
    alloc = allocate_for_budget(
        compute, optimizer=optimizer, params=params, rho_n=rho_n, rho_d=rho_d
    )
    alloc["target_loss"] = target_loss
    alloc["solved"] = bool(result.success) and abs(alloc["loss"] - target_loss) < 1e-3
    return alloc


def isoflop_curve(
    compute: float,
    optimizer: str = "adamw",
    params: ChinchillaParams | None = None,
    n_points: int = 64,
    n_span: float = 30.0,
) -> dict[str, Any]:
    """Sweep N on an isoFLOP slice; mark the local loss minimum."""
    params = params or ChinchillaParams()
    rho = OptimizerRho.from_name(optimizer)
    opt = allocate_for_budget(compute, optimizer=optimizer, params=params)
    n_center = opt["N"]
    n_grid = np.geomspace(n_center / n_span, n_center * n_span, n_points)
    d_grid = compute / (6.0 * n_grid)
    losses = np.asarray(chinchilla_loss(n_grid, d_grid, params, rho), dtype=float)
    i_min = int(np.argmin(losses))
    return {
        "optimizer": rho.name,
        "compute": compute,
        "N": n_grid.tolist(),
        "D": d_grid.tolist(),
        "loss": losses.tolist(),
        "N_star": float(n_grid[i_min]),
        "D_star": float(d_grid[i_min]),
        "loss_star": float(losses[i_min]),
        "closed_form": opt,
    }


def _rho_kwargs(
    name: str,
    rho_overrides: dict[str, OptimizerRho | dict[str, Any]] | None,
) -> dict[str, float]:
    if not rho_overrides or name not in rho_overrides:
        return {}
    meta = rho_overrides[name]
    if isinstance(meta, OptimizerRho):
        return {"rho_n": meta.rho_n, "rho_d": meta.rho_d}
    return {
        "rho_n": float(meta.get("rho_n", 1.0)),
        "rho_d": float(meta.get("rho_d", 1.0)),
    }


def compare_optimizers(
    compute: float,
    optimizers: list[str] | None = None,
    params: ChinchillaParams | None = None,
    rho_overrides: dict[str, OptimizerRho | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    optimizers = optimizers or ["adamw", "muon", "normuon", "soap"]
    rows = [
        allocate_for_budget(compute, opt, params=params, **_rho_kwargs(opt, rho_overrides))
        for opt in optimizers
    ]
    base = next(r for r in rows if r["optimizer"] == "adamw") if "adamw" in optimizers else rows[0]
    for row in rows:
        row["delta_N_vs_ref"] = row["N"] - base["N"]
        row["delta_D_vs_ref"] = row["D"] - base["D"]
        row["N_ratio_vs_ref"] = row["N"] / base["N"]
        row["loss_delta_vs_ref"] = row["loss"] - base["loss"]
        row["compute_savings_vs_adamw_same_loss"] = None
    # Same-loss savings vs AdamW
    adamw_kw = _rho_kwargs("adamw", rho_overrides)
    adamw_loss = allocate_for_budget(compute, "adamw", params=params, **adamw_kw)["loss"]
    for row in rows:
        if row["optimizer"] == "adamw":
            row["compute_savings_vs_adamw_same_loss"] = 0.0
            continue
        needed = compute_for_target_loss(
            adamw_loss,
            optimizer=row["optimizer"],
            params=params,
            **_rho_kwargs(row["optimizer"], rho_overrides),
        )
        row["compute_to_match_adamw_loss"] = needed["compute"]
        row["compute_savings_vs_adamw_same_loss"] = 1.0 - needed["compute"] / compute
    return rows
