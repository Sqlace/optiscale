"""Core OptiScale tests."""

from __future__ import annotations

import json
from pathlib import Path

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
from optiscale.cli import main
from optiscale.cost import budget_from_gpus, cost_report, flops_to_wallclock
from optiscale.fit import compare_fit_strategies, fit_chinchilla, simulate_isoflop_data
from optiscale.io import load_fit_json, params_from_fit, save_fit_json
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
    for opt in ("adamw", "muon"):
        curve = isoflop_curve(1e21, opt, n_points=120)
        assert abs(np.log(curve["N_star"]) - np.log(curve["closed_form"]["N"])) < 0.05


def test_rho_closed_form_matches_grid_when_alpha_ne_beta():
    """Correct ρ pref: (ρ_D^β / ρ_N^α)^{1/(α+β)}, not sqrt(ρ_D/ρ_N)."""
    params = ChinchillaParams()  # α=0.34 ≠ β=0.28
    compute = 1e24
    closed = allocate_for_budget(compute, "muon", params=params)
    n_grid = np.geomspace(closed["N"] / 5, closed["N"] * 5, 400)
    d_grid = compute / (6.0 * n_grid)
    rho = OptimizerRho.from_name("muon")
    losses = np.asarray(chinchilla_loss(n_grid, d_grid, params, rho), dtype=float)
    i = int(np.argmin(losses))
    assert closed["N"] == pytest.approx(float(n_grid[i]), rel=1e-2)
    assert closed["loss"] == pytest.approx(float(losses[i]), rel=1e-5)


def test_foc_identity_at_n_star_varied_rho():
    """First-order condition: α A / (ρ_N N*)^α = β B / (ρ_D D*)^β."""
    params = ChinchillaParams()
    compute = 1e23
    for rho_n, rho_d in [(1.0, 1.0), (1.35, 1.25), (1.8, 0.9), (0.9, 1.6)]:
        alloc = allocate_for_budget(compute, "adamw", params=params, rho_n=rho_n, rho_d=rho_d)
        n, d = alloc["N"], alloc["D"]
        lhs = params.alpha * params.A / (rho_n * n) ** params.alpha
        rhs = params.beta * params.B / (rho_d * d) ** params.beta
        assert lhs == pytest.approx(rhs, rel=1e-6)


def test_target_loss_solver():
    base = allocate_for_budget(1e21, "adamw")
    solved = compute_for_target_loss(base["loss"], "adamw")
    assert solved["compute"] == pytest.approx(1e21, rel=0.05)
    assert solved["solved"] is True


def test_target_loss_rejects_below_E():
    with pytest.raises(ValueError):
        compute_for_target_loss(1.0, "adamw")  # E=1.69


def test_cost_and_gpu_budget():
    b = budget_from_gpus("h100", 8, 24.0)
    assert b["compute"] > 0
    w = flops_to_wallclock(b["compute"], "h100", 8)
    assert w["hours"] == pytest.approx(24.0, rel=1e-3)
    report = cost_report(compute=1e23, gpu_id="a100", count=16)
    assert "allocations" in report
    assert len(report["savings_vs_adamw_loss"]) >= 1
    assert "ρ" in report["rho_disclaimer"] or "rho" in report["rho_disclaimer"].lower()


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


def test_fit_json_roundtrip(tmp_path: Path):
    true = ChinchillaParams()
    adamw = simulate_isoflop_data(params=true, optimizer="adamw", noise_std=0.0, seed=0)
    muon = simulate_isoflop_data(params=true, optimizer="muon", noise_std=0.0, seed=1)
    runs = {
        "adamw": {"N": adamw["N"], "D": adamw["D"], "L": adamw["L"]},
        "muon": {"N": muon["N"], "D": muon["D"], "L": muon["L"]},
    }
    cmp = compare_fit_strategies(runs)
    path = tmp_path / "fit.json"
    save_fit_json(cmp, path)
    loaded = load_fit_json(path)
    params, rhos = params_from_fit(loaded)
    assert params.alpha == pytest.approx(cmp["shared"]["params"]["alpha"], rel=1e-9)
    assert rhos["muon"].rho_n == pytest.approx(cmp["shared"]["rhos"]["muon"]["rho_n"], rel=1e-9)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema"] == "optiscale.fit.v1"


def test_cost_report_with_fit_applies_rho():
    """Fitted ρ overrides must change allocations vs planning priors."""
    prior = cost_report(compute=1e23, optimizers=["adamw", "muon"])
    fake_rhos = {
        "adamw": OptimizerRho("adamw", 1.0, 1.0, "AdamW"),
        "muon": OptimizerRho("muon", 2.0, 1.5, "Muon"),
    }
    fitted = cost_report(
        compute=1e23,
        optimizers=["adamw", "muon"],
        rho_overrides=fake_rhos,
    )
    prior_muon = next(r for r in prior["allocations"] if r["optimizer"] == "muon")
    fit_muon = next(r for r in fitted["allocations"] if r["optimizer"] == "muon")
    assert fit_muon["rho_n"] == pytest.approx(2.0)
    assert fit_muon["N"] != pytest.approx(prior_muon["N"], rel=1e-3)
    assert fitted["used_fitted_rhos"] is True


def test_compare_budget_usd_with_fit_applies_rho(tmp_path: Path, capsys):
    true = ChinchillaParams()
    adamw = simulate_isoflop_data(params=true, optimizer="adamw", noise_std=0.0, seed=0)
    muon = simulate_isoflop_data(params=true, optimizer="muon", noise_std=0.0, seed=1)
    runs = {
        "adamw": {"N": adamw["N"], "D": adamw["D"], "L": adamw["L"]},
        "muon": {"N": muon["N"], "D": muon["D"], "L": muon["L"]},
    }
    cmp = compare_fit_strategies(runs)
    fit_path = tmp_path / "fit.json"
    save_fit_json(cmp, fit_path)
    md_path = tmp_path / "out.md"
    rc = main(
        [
            "compare",
            "--budget-usd",
            "100000",
            "--gpu",
            "h100",
            "--count",
            "8",
            "--optimizers",
            "adamw,muon",
            "--fit",
            str(fit_path),
            "--md",
            str(md_path),
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    muon_row = next(r for r in out if r["optimizer"] == "muon")
    assert muon_row["rho_n"] == pytest.approx(cmp["shared"]["rhos"]["muon"]["rho_n"], rel=1e-6)
    md = md_path.read_text(encoding="utf-8")
    assert "fit" in md.lower() or "ρ" in md


def test_optimizer_rho_unknown():
    with pytest.raises(KeyError):
        OptimizerRho.from_name("not-an-opt")


def test_loss_decreases_with_more_data():
    n = 1e9
    l1 = chinchilla_loss(n, 1e10)
    l2 = chinchilla_loss(n, 1e11)
    assert l2 < l1


def test_cli_allocate_modes_smoke(capsys):
    assert main(["allocate", "--flops", "1e22", "--optimizer", "muon"]) == 0
    assert main(["allocate", "--fixed-n", "1e9", "--flops", "1e21"]) == 0
    assert main(["allocate", "--ratio", "100", "--flops", "1e21"]) == 0
    assert main(["allocate", "--target-loss", "2.8", "--optimizer", "adamw"]) == 0
    out = capsys.readouterr().out
    assert "N" in out
