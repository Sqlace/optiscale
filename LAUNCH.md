# Spectral Training Stack — Launch Kit

Three public repos under [@Sqlace](https://github.com/Sqlace):

| Repo | Hook | URL |
|------|------|-----|
| **OptiScale** | Chinchilla assumes AdamW — change the optimizer, redo the budget | https://github.com/Sqlace/optiscale |
| **SpectOptim** | Replace AdamW in ~20 lines; Muon / NorMuon / Aurora + telemetry | https://github.com/Sqlace/spectoptim |
| **OrthoLab** | Watch neuron death in 20 seconds — spectral geometry studio | https://github.com/Sqlace/ortholab |

## Show HN draft

**Title:** Show HN: OptiScale – Chinchilla scaling laws that know about Muon

**Body:**

Chinchilla (2022) is calibrated under AdamW. Newer spectral optimizers (Muon, NorMuon, SOAP, …) change effective parameter/data efficiency, so the same FLOP budget implies a different compute-optimal (N*, D*).

OptiScale is a small lab for that:

- Allocate N*/D* with optimizer-specific ρ_N, ρ_D
- Cost on H100/A100 catalogs (MFU, $)
- Compare savings to match AdamW loss
- Fit shared-exponent vs separate Chinchilla fits (Volkova-style claim)
- Web UI + `pip install` CLI

Sibling tools:

- SpectOptim — one PyTorch API for Muon-family optimizers + neuron-death telemetry
- OrthoLab — browser lab for Newton–Schulz / polar geometry

Links: https://github.com/Sqlace/optiscale · https://github.com/Sqlace/spectoptim · https://github.com/Sqlace/ortholab

## Twitter / X thread (short)

1/ Chinchilla assumes AdamW. If you train with Muon, your “optimal” model size is wrong.

2/ OptiScale reallocates N* and D* with optimizer ρ factors, costs the run on real GPUs, and shows how much compute you save to hit the same loss.

3/ SpectOptim: drop-in Muon / NorMuon / Aurora with auto AdamW routing for embeds.

4/ OrthoLab: interactive Newton–Schulz + neuron-death panels. Zero backend.

## Local demos (GIF stand-ins)

```bash
# OptiScale
cd ~/Projects/optiscale && pip install -e . && optiscale compare --flops 1e24 --md report.md
cd web && npm run dev

# SpectOptim
cd ~/Projects/spectoptim && pip install -e . && python bench/microbench.py

# OrthoLab
cd ~/Projects/ortholab && npm run dev
```

Record a 15–20s screen capture of: OptiScale optimizer dropdown jumping N*; OrthoLab Orthogonalize spectrum; SpectOptim diagnose() output.
