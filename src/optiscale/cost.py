"""GPU catalog, wall-clock and dollar cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .allocate import allocate_for_budget, compare_optimizers, compute_for_target_loss
from .laws import ChinchillaParams, OptimizerRho, format_flops, format_params

PLANNING_PRIOR_DISCLAIMER = (
    "ρ are planning priors unless --fit (or params/rho_overrides) was provided."
)


@dataclass(frozen=True)
class GPUSpec:
    name: str
    peak_tflops_bf16: float
    dollars_per_hour: float
    default_mfu: float = 0.40
    memory_gb: float = 80.0


GPU_CATALOG: dict[str, GPUSpec] = {
    "h100": GPUSpec("H100 SXM", 989.0, 4.00, 0.40, 80.0),
    "h100_pcie": GPUSpec("H100 PCIe", 756.0, 3.50, 0.38, 80.0),
    "a100": GPUSpec("A100 80GB", 312.0, 2.20, 0.45, 80.0),
    "a100_40": GPUSpec("A100 40GB", 312.0, 1.80, 0.42, 40.0),
    "l40s": GPUSpec("L40S", 362.0, 1.60, 0.35, 48.0),
    "rtx4090": GPUSpec("RTX 4090", 330.0, 0.80, 0.30, 24.0),
    "v100": GPUSpec("V100", 125.0, 0.90, 0.40, 32.0),
    "tpu_v5e": GPUSpec("TPU v5e", 197.0, 1.20, 0.50, 16.0),
}


def list_gpus() -> list[dict[str, Any]]:
    return [
        {
            "id": key,
            "name": g.name,
            "peak_tflops_bf16": g.peak_tflops_bf16,
            "dollars_per_hour": g.dollars_per_hour,
            "default_mfu": g.default_mfu,
            "memory_gb": g.memory_gb,
        }
        for key, g in GPU_CATALOG.items()
    ]


def get_gpu(gpu_id: str) -> GPUSpec:
    key = gpu_id.lower().strip()
    if key not in GPU_CATALOG:
        raise KeyError(f"Unknown GPU {gpu_id!r}. Known: {', '.join(GPU_CATALOG)}")
    return GPU_CATALOG[key]


def effective_flops_per_second(gpu: GPUSpec, count: int, mfu: float | None = None) -> float:
    mfu = gpu.default_mfu if mfu is None else mfu
    return gpu.peak_tflops_bf16 * 1e12 * mfu * count


def flops_to_wallclock(
    compute: float,
    gpu_id: str = "h100",
    count: int = 8,
    mfu: float | None = None,
) -> dict[str, Any]:
    gpu = get_gpu(gpu_id)
    mfu = gpu.default_mfu if mfu is None else mfu
    flops_s = effective_flops_per_second(gpu, count, mfu)
    seconds = compute / flops_s
    hours = seconds / 3600.0
    days = hours / 24.0
    cost = hours * gpu.dollars_per_hour * count
    return {
        "compute": compute,
        "compute_human": format_flops(compute),
        "gpu": gpu.name,
        "gpu_id": gpu_id,
        "count": count,
        "mfu": mfu,
        "seconds": seconds,
        "hours": hours,
        "days": days,
        "cost_usd": cost,
        "throughput_flops_s": flops_s,
    }


def budget_from_gpus(
    gpu_id: str,
    count: int,
    hours: float,
    mfu: float | None = None,
) -> dict[str, Any]:
    gpu = get_gpu(gpu_id)
    mfu = gpu.default_mfu if mfu is None else mfu
    compute = effective_flops_per_second(gpu, count, mfu) * hours * 3600.0
    cost = hours * gpu.dollars_per_hour * count
    return {
        "compute": compute,
        "compute_human": format_flops(compute),
        "gpu": gpu.name,
        "gpu_id": gpu_id,
        "count": count,
        "hours": hours,
        "mfu": mfu,
        "cost_usd": cost,
    }


def _rho_kw(
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


def cost_report(
    compute: float | None = None,
    budget_usd: float | None = None,
    gpu_id: str = "h100",
    count: int = 8,
    mfu: float | None = None,
    optimizers: list[str] | None = None,
    params: ChinchillaParams | None = None,
    rho_overrides: dict[str, OptimizerRho | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full cost + multi-optimizer allocation report.

    Pass fitted ``params`` / ``rho_overrides`` (e.g. from ``params_from_fit``)
    so allocations honor fitted ρ instead of planning priors.
    """
    gpu = get_gpu(gpu_id)
    mfu = gpu.default_mfu if mfu is None else mfu
    if compute is None:
        if budget_usd is None:
            raise ValueError("Provide compute or budget_usd")
        hours = budget_usd / (gpu.dollars_per_hour * count)
        compute = budget_from_gpus(gpu_id, count, hours, mfu)["compute"]
    wall = flops_to_wallclock(compute, gpu_id, count, mfu)
    rows = compare_optimizers(
        compute,
        optimizers=optimizers,
        params=params,
        rho_overrides=rho_overrides,
    )
    for row in rows:
        row["N_human"] = format_params(row["N"])
        row["D_human"] = format_params(row["D"])
    adamw = allocate_for_budget(
        compute, "adamw", params=params, **_rho_kw("adamw", rho_overrides)
    )
    fitted = bool(params is not None or rho_overrides)
    savings = []
    for row in rows:
        if row["optimizer"] == "adamw":
            continue
        match = compute_for_target_loss(
            adamw["loss"],
            optimizer=row["optimizer"],
            params=params,
            **_rho_kw(row["optimizer"], rho_overrides),
        )
        match_wall = flops_to_wallclock(match["compute"], gpu_id, count, mfu)
        savings.append(
            {
                "optimizer": row["optimizer"],
                "label": row["label"],
                "compute_to_match_adamw_loss": match["compute"],
                "compute_to_match_human": format_flops(match["compute"]),
                "fraction_of_budget": match["compute"] / compute,
                "usd_to_match": match_wall["cost_usd"],
                "usd_saved_vs_full_budget": wall["cost_usd"] - match_wall["cost_usd"],
                "pct_compute_saved": 100.0 * (1.0 - match["compute"] / compute),
            }
        )
    return {
        "wallclock": wall,
        "allocations": rows,
        "savings_vs_adamw_loss": savings,
        "reference_adamw_loss": adamw["loss"],
        "used_fitted_rhos": fitted,
        "rho_disclaimer": (
            "ρ from fit.json / overrides."
            if fitted
            else PLANNING_PRIOR_DISCLAIMER
        ),
    }
