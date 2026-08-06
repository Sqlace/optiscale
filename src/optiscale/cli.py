"""Command-line interface for OptiScale."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .allocate import allocate_for_budget, compare_optimizers, isoflop_curve
from .cost import budget_from_gpus, cost_report, list_gpus
from .fit import compare_fit_strategies, simulate_isoflop_data
from .io import export_python_snippet, load_runs, save_allocation_csv, save_markdown_report
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
    a.add_argument("--flops", type=float, required=True)
    a.add_argument("--optimizer", default="adamw")
    a.add_argument("--rho-n", type=float, default=None)
    a.add_argument("--rho-d", type=float, default=None)

    c = sub.add_parser("compare", help="Compare optimizers at a budget")
    c.add_argument("--flops", type=float, default=None)
    c.add_argument("--budget-usd", type=float, default=None)
    c.add_argument("--optimizers", default="adamw,muon,normuon,soap")
    c.add_argument("--gpu", default="h100")
    c.add_argument("--count", type=int, default=8)
    c.add_argument("--mfu", type=float, default=None)
    c.add_argument("--csv", default=None)
    c.add_argument("--md", default=None)

    f = sub.add_parser("fit", help="Fit laws from CSV/JSON or synthetic demo")
    f.add_argument("--data", default=None, help="Path to CSV/JSON runs")
    f.add_argument("--synthetic", action="store_true")
    f.add_argument("--noise", type=float, default=0.01)

    r = sub.add_parser("report", help="Full cost + savings report")
    r.add_argument("--flops", type=float, default=None)
    r.add_argument("--budget-usd", type=float, default=None)
    r.add_argument("--gpu", default="h100")
    r.add_argument("--count", type=int, default=8)
    r.add_argument("--mfu", type=float, default=None)
    r.add_argument("--md", default="optiscale-report.md")

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "allocate":
        _print(
            allocate_for_budget(
                args.flops,
                optimizer=args.optimizer,
                rho_n=args.rho_n,
                rho_d=args.rho_d,
            )
        )
    elif args.cmd == "compare":
        opts = [x.strip() for x in args.optimizers.split(",") if x.strip()]
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
            )
            compute = report["wallclock"]["compute"]
            rows = report["allocations"]
        else:
            rows = compare_optimizers(compute, optimizers=opts)
        if args.csv:
            save_allocation_csv(rows, args.csv)
        if args.md:
            report = cost_report(
                compute=compute,
                gpu_id=args.gpu,
                count=args.count,
                mfu=args.mfu,
                optimizers=opts,
            )
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
        _print(compare_fit_strategies(runs))
    elif args.cmd == "report":
        report = cost_report(
            compute=args.flops,
            budget_usd=args.budget_usd,
            gpu_id=args.gpu,
            count=args.count,
            mfu=args.mfu,
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
