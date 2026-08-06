"""Command-line interface for OptiScale."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .allocate import (
    allocate_fixed_n,
    allocate_fixed_ratio,
    allocate_for_budget,
    compute_for_target_loss,
    isoflop_curve,
)
from .cost import budget_from_gpus, cost_report, list_gpus
from .fit import compare_fit_strategies, simulate_isoflop_data
from .io import (
    export_python_snippet,
    load_fit_json,
    load_runs,
    params_from_fit,
    save_allocation_csv,
    save_fit_json,
    save_markdown_report,
)
from .laws import list_optimizers


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=float))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optiscale",
        description="Optimizer-aware LLM scaling laws: allocate, fit, cost, compare.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("allocate", help="Compute-optimal N*, D* for a FLOP budget")
    a.add_argument("--flops", type=float, default=None, help="FLOP budget C (required except --target-loss)")
    mode = a.add_mutually_exclusive_group()
    mode.add_argument(
        "--fixed-n",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Fix model size N; needs --flops (or use allocate_fixed_n with compute)",
    )
    mode.add_argument(
        "--ratio",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Fix tokens/param ratio; requires --flops",
    )
    mode.add_argument(
        "--target-loss",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Inverse: min compute for target loss (--flops optional)",
    )
    a.add_argument("--optimizer", default="adamw")
    a.add_argument("--rho-n", type=float, default=None)
    a.add_argument("--rho-d", type=float, default=None)
    a.add_argument("--fit", default=None, help="fit.json from `optiscale fit --out`")
    a.add_argument(
        "--active-frac",
        type=float,
        default=None,
        help="MoE active parameter fraction in (0,1]; C≈6·frac·N·D",
    )

    c = sub.add_parser("compare", help="Compare optimizers at a budget")
    c.add_argument("--flops", type=float, default=None)
    c.add_argument("--budget-usd", type=float, default=None)
    c.add_argument("--optimizers", default="adamw,muon,normuon,soap")
    c.add_argument("--gpu", default="h100")
    c.add_argument("--count", type=int, default=8)
    c.add_argument("--mfu", type=float, default=None)
    c.add_argument("--csv", default=None)
    c.add_argument("--md", default=None)
    c.add_argument("--fit", default=None, help="fit.json with fitted ρ overrides")
    c.add_argument("--active-frac", type=float, default=None, help="MoE active fraction")

    f = sub.add_parser("fit", help="Fit laws from CSV/JSON or synthetic demo")
    f.add_argument("--data", default=None, help="Path to CSV/JSON runs")
    f.add_argument("--synthetic", action="store_true")
    f.add_argument("--noise", type=float, default=0.01)
    f.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="Bootstrap resamples for CI on separate + shared fits",
    )
    f.add_argument("--out", default=None, help="Write fit.json for allocate --fit")

    r = sub.add_parser("report", help="Full cost + savings report")
    r.add_argument("--flops", type=float, default=None)
    r.add_argument("--budget-usd", type=float, default=None)
    r.add_argument("--gpu", default="h100")
    r.add_argument("--count", type=int, default=8)
    r.add_argument("--mfu", type=float, default=None)
    r.add_argument("--md", default="optiscale-report.md")
    r.add_argument("--fit", default=None, help="fit.json with fitted ρ overrides")

    i = sub.add_parser("isoflop", help="IsoFLOP loss curve JSON")
    i.add_argument("--flops", type=float, required=True)
    i.add_argument("--optimizer", default="adamw")
    i.add_argument("--points", type=int, default=64)

    g = sub.add_parser("gpus", help="List GPU catalog")
    o = sub.add_parser("optimizers", help="List optimizer ρ priors")
    s = sub.add_parser("snippet", help="Print Python snippet")
    s.add_argument("--flops", type=float, default=1e24)
    s.add_argument("--optimizer", default="muon")

    b = sub.add_parser("from-gpus", help="Convert GPU×hours to FLOP budget")
    b.add_argument("--gpu", default="h100")
    b.add_argument("--count", type=int, default=8)
    b.add_argument("--hours", type=float, required=True)
    b.add_argument("--mfu", type=float, default=None)

    return p


def _load_fit_kwargs(path: str | None, optimizer: str, rho_n: float | None, rho_d: float | None):
    """Return (params, rho_n, rho_d) applying fit.json when present."""
    params = None
    if path:
        fit = load_fit_json(path)
        params, rhos = params_from_fit(fit)
        if optimizer in rhos and rho_n is None and rho_d is None:
            rho_n = rhos[optimizer].rho_n
            rho_d = rhos[optimizer].rho_d
    return params, rho_n, rho_d


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "allocate":
        params, rho_n, rho_d = _load_fit_kwargs(
            args.fit, args.optimizer, args.rho_n, args.rho_d
        )
        kw: dict[str, Any] = {
            "optimizer": args.optimizer,
            "params": params,
            "rho_n": rho_n,
            "rho_d": rho_d,
        }
        if args.fixed_n is not None:
            if args.flops is None:
                print("--fixed-n requires --flops", file=sys.stderr)
                return 2
            _print(allocate_fixed_n(args.fixed_n, compute=args.flops, **kw))
        elif args.ratio is not None:
            if args.flops is None:
                print("--ratio requires --flops", file=sys.stderr)
                return 2
            _print(
                allocate_fixed_ratio(
                    args.flops, tokens_per_param_ratio=args.ratio, **kw
                )
            )
        elif args.target_loss is not None:
            _print(compute_for_target_loss(args.target_loss, **kw))
        else:
            if args.flops is None:
                print("Provide --flops (or --target-loss / --fixed-n / --ratio)", file=sys.stderr)
                return 2
            if args.active_frac is not None:
                kw["active_frac"] = args.active_frac
            _print(allocate_for_budget(args.flops, **kw))
    elif args.cmd == "compare":
        opts = [x.strip() for x in args.optimizers.split(",") if x.strip()]
        fit_params = None
        fit_rhos = None
        if getattr(args, "fit", None):
            fit = load_fit_json(args.fit)
            fit_params, fit_rhos = params_from_fit(fit)
        rows = []
        compute = args.flops
        if compute is None:
            if args.budget_usd is None:
                print("Provide --flops or --budget-usd", file=sys.stderr)
                return 2
            report = cost_report(
                budget_usd=args.budget_usd,
                gpu_id=args.gpu,
                count=args.count,
                mfu=args.mfu,
                optimizers=opts,
                params=fit_params,
                rho_overrides=fit_rhos,
            )
            compute = report["wallclock"]["compute"]
            rows = report["allocations"]
        else:
            for name in opts:
                kw: dict[str, Any] = {"params": fit_params} if fit_params else {}
                if fit_rhos and name in fit_rhos:
                    kw["rho_n"] = fit_rhos[name].rho_n
                    kw["rho_d"] = fit_rhos[name].rho_d
                if args.active_frac is not None:
                    kw["active_frac"] = args.active_frac
                rows.append(allocate_for_budget(compute, optimizer=name, **kw))
            base = next((r for r in rows if r["optimizer"] == "adamw"), rows[0])
            for row in rows:
                row["N_ratio_vs_ref"] = row["N"] / base["N"]
                row["delta_N_vs_ref"] = row["N"] - base["N"]
        report_kw: dict[str, Any] = {
            "gpu_id": args.gpu,
            "count": args.count,
            "mfu": args.mfu,
            "optimizers": opts,
            "params": fit_params,
            "rho_overrides": fit_rhos,
        }
        if args.csv:
            save_allocation_csv(rows, args.csv)
        if args.md:
            report = cost_report(compute=compute, **report_kw)
            save_markdown_report(report, args.md)
        _print(rows)
    elif args.cmd == "fit":
        if args.synthetic or not args.data:
            adamw = simulate_isoflop_data(optimizer="adamw", noise_std=args.noise, seed=0)
            muon = simulate_isoflop_data(optimizer="muon", noise_std=args.noise, seed=1)
            runs = {
                "adamw": {"N": adamw["N"], "D": adamw["D"], "L": adamw["L"]},
                "muon": {"N": muon["N"], "D": muon["D"], "L": muon["L"]},
            }
        else:
            runs = load_runs(args.data)
        result = compare_fit_strategies(runs, bootstrap=args.bootstrap)
        if args.out:
            save_fit_json(result, args.out)
            result = {**result, "wrote": args.out}
        _print(result)
    elif args.cmd == "report":
        fit_params = None
        fit_rhos = None
        if getattr(args, "fit", None):
            fit = load_fit_json(args.fit)
            fit_params, fit_rhos = params_from_fit(fit)
        report = cost_report(
            compute=args.flops,
            budget_usd=args.budget_usd,
            gpu_id=args.gpu,
            count=args.count,
            mfu=args.mfu,
            params=fit_params,
            rho_overrides=fit_rhos,
        )
        save_markdown_report(report, args.md)
        _print({"wrote": args.md, "wallclock": report["wallclock"], "savings": report["savings_vs_adamw_loss"]})
    elif args.cmd == "isoflop":
        _print(isoflop_curve(args.flops, optimizer=args.optimizer, n_points=args.points))
    elif args.cmd == "gpus":
        _print(list_gpus())
    elif args.cmd == "optimizers":
        _print(list_optimizers())
    elif args.cmd == "snippet":
        print(export_python_snippet(args.flops, args.optimizer))
    elif args.cmd == "from-gpus":
        _print(budget_from_gpus(args.gpu, args.count, args.hours, args.mfu))
    else:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
