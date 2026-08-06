# OptiScale

**Chinchilla assumes AdamW.** Change the optimizer — redo the budget.

OptiScale is an optimizer-aware LLM scaling lab: allocate compute-optimal \(N^*\) and \(D^*\), cost jobs on real GPUs, compare Muon / NorMuon / SOAP / … against AdamW, and fit shared-exponent \(\rho_N,\rho_D\) models.

> Part of the **Spectral Training Stack** with [SpectOptim](https://github.com/Sqlace/spectoptim) and [OrthoLab](https://github.com/Sqlace/ortholab).

## Why

Hoffmann et al. (2022) Chinchilla laws are calibrated under AdamW. Volkova et al. (2026) show that fitting separate Chinchilla laws per optimizer is ill-conditioned; a robust alternative keeps shared exponents and introduces optimizer-specific rescaling factors \(\rho_N,\rho_D\) relative to AdamW.

Existing calculators ignore this. OptiScale puts allocate → cost → compare → fit in one product.

## Quick start

```bash
pip install -e ".[dev]"
optiscale allocate --flops 1e24 --optimizer muon
optiscale compare --budget-usd 1000000 --gpu h100 --count 64 --md report.md
optiscale fit --synthetic
optiscale report --flops 1e24 --gpu a100 --count 16
```

### Fit → compare recipe

```bash
# Fit shared-exponent ρ with bootstrap CI, persist for reuse
optiscale fit --synthetic --bootstrap 200 --out fit.json

# Allocations / cost report honor fitted ρ (not planning priors)
optiscale compare --flops 1e24 --fit fit.json --md report.md
optiscale compare --budget-usd 1000000 --gpu h100 --count 64 --fit fit.json
optiscale allocate --flops 1e24 --optimizer muon --fit fit.json
```

### Allocate modes

| Flag | Behavior |
|------|----------|
| `--flops C` | Default: compute-optimal \(N^*, D^*\) via `allocate_for_budget` |
| `--fixed-n N --flops C` | Fix model size; solve \(D = C/(6N)\) |
| `--ratio R --flops C` | Fix tokens/param \(D/N = R\) (e.g. overtrain) |
| `--target-loss L` | Inverse: min compute on the ridge for loss \(L\) (`--flops` optional) |
| `--rho-n` / `--rho-d` | Override catalog ρ |
| `--fit fit.json` | Load fitted params + ρ from `fit --out` |

### Fit / compare flags

| Flag | Behavior |
|------|----------|
| `fit --bootstrap N` | Bootstrap CI on separate Chinchilla + shared-exponent fits |
| `fit --out fit.json` | Persist for `allocate` / `compare` / `report --fit` |
| `compare --fit fit.json` | Re-allocate with fitted ρ (also applies to `--budget-usd` and `--md`) |

```python
from optiscale import allocate_for_budget, cost_report, compare_fit_strategies
from optiscale.fit import simulate_isoflop_data

alloc = allocate_for_budget(1e24, optimizer="muon")
print(alloc["N_human"], alloc["D_human"], alloc["loss"])

report = cost_report(compute=1e24, gpu_id="h100", count=8)
print(report["savings_vs_adamw_loss"])
```

## Web lab

```bash
cd web && npm install && npm run dev
```

Practitioner dashboard: FLOP / GPU-hour budgets, presets ($1M, 8×H100×30d, overtrain), editable ρ, target-loss inverse, isoFLOP curves, ρ-sensitivity heatmap, multi-optimizer table, shareable URL state, Markdown/CSV export.

Research tab: CSV upload + synthetic IsoFLOP demo + **Apply ρ to Practitioner**. Full L-BFGS shared-exponent fit lives in the Python CLI.

## Features (v0.1)

| Area | Capabilities |
|------|----------------|
| Laws | Classic Chinchilla + optimizer-aware \(\rho_N,\rho_D\) |
| Allocate | Budget→(N*,D*), fixed N, fixed tok/param, target-loss inverse |
| IsoFLOP | Loss curves + closed-form star marker |
| Cost | GPU catalog (H100 / H100 PCIe, A100 80/40, L40S, …), MFU, $/day |
| Compare | Multi-optimizer overlay + compute-savings to match AdamW loss |
| Fit | L-BFGS / grid+refine, Huber, bootstrap CI, shared-exponent vs separate |
| I/O | CSV/JSON runs, Markdown reports, Python snippets |
| Web | Dark/light, presets, URL state, export, ρ heatmap |

## Default ρ priors (planning only)

| Optimizer | ρ_N | ρ_D |
|-----------|-----|-----|
| AdamW | 1.00 | 1.00 |
| Muon | 1.35 | 1.25 |
| NorMuon | 1.45 | 1.30 |
| Aurora | 1.40 | 1.28 |
| SOAP | 1.20 | 1.15 |

Override with `--rho-n` / `--rho-d` or fit your own runs. These are **not** a substitute for your IsoFLOP campaigns.

## Citations

- Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022.
- Volkova et al., *Towards Robust Scaling Laws for Optimizers*, arXiv:2602.07712, 2026.
- Jordan et al., Muon optimizer; NorMuon / Aurora follow-ups in the spectral optimizer line.

## License

MIT
