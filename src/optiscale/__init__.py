"""OptiScale: optimizer-aware LLM scaling laws."""

from .allocate import (
    allocate_for_budget,
    allocate_fixed_n,
    allocate_fixed_ratio,
    compute_for_target_loss,
    isoflop_curve,
)
from .cost import GPU_CATALOG, budget_from_gpus, cost_report, flops_to_wallclock
from .fit import FitResult, compare_fit_strategies, fit_chinchilla, fit_shared_exponents
from .io import (
    load_fit_json,
    load_runs,
    params_from_fit,
    save_allocation_csv,
    save_fit_json,
    save_markdown_report,
)
from .laws import (
    OPTIMIZER_RHOS,
    ChinchillaParams,
    OptimizerRho,
    chinchilla_loss,
    effective_params,
    list_optimizers,
)

__all__ = [
    "OPTIMIZER_RHOS",
    "ChinchillaParams",
    "FitResult",
    "GPU_CATALOG",
    "OptimizerRho",
    "allocate_fixed_n",
    "allocate_fixed_ratio",
    "allocate_for_budget",
    "budget_from_gpus",
    "chinchilla_loss",
    "compare_fit_strategies",
    "compute_for_target_loss",
    "cost_report",
    "effective_params",
    "fit_chinchilla",
    "fit_shared_exponents",
    "flops_to_wallclock",
    "isoflop_curve",
    "list_optimizers",
    "load_fit_json",
    "load_runs",
    "params_from_fit",
    "save_allocation_csv",
    "save_fit_json",
    "save_markdown_report",
]

__version__ = "0.1.0"
