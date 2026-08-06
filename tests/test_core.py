"""Core OptiScale tests."""

from __future__ import annotations

import numpy as np
import pytest

from optiscale.allocate import (
    allocate_fixed_n,
    allocate_fixed_ratio,
    allocate_for_budget,
    compare_optimizers,
    compute_for_target_loss,
    isoflop_curve,
)
from optiscale.cost import budget_from_gpus, cost_report, flops_to_wallclock
from optiscale.fit import compare_fit_strategies, fit_chinchilla, simulate_isoflop_data
from optiscale.laws import ChinchillaParams, OptimizerRho, chinchilla_loss, flops


def test_flops_identity():
    assert flops(1e9, 20e9) == pytest.approx(6 * 1e9 * 20e9)


def test_allocate_respects_budget():
    alloc = allocate_for_budget(1e24, "adamw")
    assert flops(alloc["N"], alloc["D"]) == pytest.approx(1e24, rel=1e-6)
    assert alloc["tokens_per_param"] > 0


def test_muon_uses_fewer_params_than_adamw_same_budget():
    # Higher ρ_N means each physical param counts more → fewer physical params needed.
    adamw = allocate_for_budget(1e23, "adamw")
    muon = allocate_for_budget(1e23, "muon")
    assert muon["N"] < adamw["N"]
    assert muon["loss"] < adamw["loss"]


def test_fixed_ratio_overtrain():
    alloc = allocate_fixed_ratio(1e22, tokens_per_param_ratio=100.0)
    assert alloc["tokens_per_param"] == pytest.approx(100.0)
    assert flops(alloc["N"], alloc["D"]) == pytest.approx(1e22, rel=1e-6)


def test_fixed_n():
    alloc = allocate_fixed_n(7e9, compute=6 * 7e9 * 140e9)
    assert alloc["D"] == pytest.approx(140e9, rel=1e-6)


def test_isoflop_min_near_closed_form():
    curve = isoflop_curve(1e21, "adamw", n_points=80)
    assert abs(np.log(curve["N_star"]) - np.log(curve["closed_form"]["N"])) < 0.15


def test_target_loss_solver():
    base = allocate_for_budget(1e21, "adamw")
    solved = compute_for_target_loss(base["loss"], "adamw")
    assert solved["compute"] == pytest.approx(1e21, rel=0.05)


def test_cost_and_gpu_budget():
    b = budget_from_gpus("h100", 8, 24.0)
    assert b["compute"] > 0
    w = flops_to_wallclock(b["compute"], "h100", 8)
    assert w["hours"] == pytest.approx(24.0, rel=1e-3)
    report = cost_report(compute=1e23, gpu_id="a100", count=16)
    assert "allocations" in report
    assert len(report["savings_vs_adamw_loss"]) >= 1


def test_compare_optimizers():
    rows = compare_optimizers(1e22)
    names = {r["optimizer"] for r in rows}
    assert "adamw" in names and "muon" in names


def test_synthetic_fit_recovers_exponents():
    data = simulate_isoflop_data(noise_std=0.0, n_points_per_budget=16)
    fit = fit_chinchilla(data["N"], data["D"], data["L"], method="grid")
    assert fit.params.alpha == pytest.approx(0.34, abs=0.08)
    assert fit.params.beta == pytest.approx(0.28, abs=0.08)
    assert fit.rmse < 0.05


def test_shared_vs_separate_fit():
    true = ChinchillaParams()
    adamw = simulate_isoflop_data(params=true, optimizer="adamw", noise_std=0.005, seed=0)
    muon = simulate_isoflop_data(params=true, optimizer="muon", noise_std=0.005, seed=1)
    runs = {
        "adamw": {"N": adamw["N"], "D": adamw["D"], "L": adamw["L"]},
        "muon": {"N": muon["N"], "D": muon["D"], "L": muon["L"]},
    }
    cmp = compare_fit_strategies(runs)
    assert "shared" in cmp and "separate" in cmp
    assert cmp["shared"]["rhos"]["adamw"]["rho_n"] == pytest.approx(1.0)
    assert cmp["shared"]["rhos"]["muon"]["rho_n"] > 1.0


def test_optimizer_rho_unknown():
    with pytest.raises(KeyError):
        OptimizerRho.from_name("not-an-opt")


def test_loss_decreases_with_more_data():
    n = 1e9
    l1 = chinchilla_loss(n, 1e10)
    l2 = chinchilla_loss(n, 1e11)
    assert l2 < l1
