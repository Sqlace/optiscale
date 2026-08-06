/** Shared-exponent L-BFGS fit (browser port of optiscale.fit.fit_shared_exponents). */

import { DEFAULT_PARAMS, loss, type ChinchillaParams, type OptimizerMeta } from "./engine";

export type RunBundle = Record<string, { N: number[]; D: number[]; L: number[] }>;

export type FitSharedResult = {
  params: ChinchillaParams;
  rhos: Record<string, { rho_n: number; rho_d: number; label: string }>;
  per_optimizer_rmse: Record<string, number>;
  mean_rmse: number;
  success: boolean;
  method: string;
};

function huber(r: number, delta = 0.01): number {
  const a = Math.abs(r);
  return a <= delta ? 0.5 * r * r : delta * (a - 0.5 * delta);
}

function clip(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function unpack(
  x: number[],
  optNames: string[],
  reference: string,
): { params: ChinchillaParams; rhos: Record<string, OptimizerMeta> } {
  const nonRef = optNames.filter((o) => o !== reference);
  const params: ChinchillaParams = {
    E: Math.exp(x[0]),
    A: Math.exp(x[1]),
    B: Math.exp(x[2]),
    alpha: clip(x[3], 0.05, 1.5),
    beta: clip(x[4], 0.05, 1.5),
  };
  const rhos: Record<string, OptimizerMeta> = {
    [reference]: { id: reference, label: reference, rho_n: 1, rho_d: 1 },
  };
  nonRef.forEach((name, i) => {
    rhos[name] = {
      id: name,
      label: name,
      rho_n: Math.exp(x[5 + 2 * i]),
      rho_d: Math.exp(x[5 + 2 * i + 1]),
    };
  });
  return { params, rhos };
}

function objective(x: number[], runs: RunBundle, optNames: string[], reference: string): number {
  const { params, rhos } = unpack(x, optNames, reference);
  let total = 0;
  let count = 0;
  for (const name of optNames) {
    const data = runs[name];
    const rho = rhos[name];
    for (let i = 0; i < data.L.length; i++) {
      const pred = loss(data.N[i], data.D[i], rho, params);
      total += huber(pred - data.L[i]);
      count += 1;
    }
  }
  return total / Math.max(count, 1);
}

function gradFd(
  x: number[],
  runs: RunBundle,
  optNames: string[],
  reference: string,
  eps = 1e-5,
): number[] {
  const g = new Array(x.length).fill(0);
  const f0 = objective(x, runs, optNames, reference);
  for (let i = 0; i < x.length; i++) {
    const xp = x.slice();
    xp[i] += eps;
    g[i] = (objective(xp, runs, optNames, reference) - f0) / eps;
  }
  return g;
}

/** Dense BFGS (dim typically ≤ 15) — enough for shared-ρ browser fits. */
function bfgsMinimize(
  x0: number[],
  runs: RunBundle,
  optNames: string[],
  reference: string,
  maxIter = 80,
): { x: number[]; success: boolean } {
  const n = x0.length;
  let x = x0.slice();
  let H: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  );
  let g = gradFd(x, runs, optNames, reference);
  for (let it = 0; it < maxIter; it++) {
    // p = -H g
    const p = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += H[i][j] * g[j];
      p[i] = -s;
    }
    // Backtracking line search
    let t = 1;
    const f0 = objective(x, runs, optNames, reference);
    let xNew = x.map((xi, i) => xi + t * p[i]);
    let fNew = objective(xNew, runs, optNames, reference);
    let guard = 0;
    while (fNew > f0 + 1e-4 * t * p.reduce((a, pi, i) => a + pi * g[i], 0) && guard < 20) {
      t *= 0.5;
      xNew = x.map((xi, i) => xi + t * p[i]);
      fNew = objective(xNew, runs, optNames, reference);
      guard += 1;
    }
    const sVec = xNew.map((xi, i) => xi - x[i]);
    const gNew = gradFd(xNew, runs, optNames, reference);
    const yVec = gNew.map((gi, i) => gi - g[i]);
    const ys = yVec.reduce((a, yi, i) => a + yi * sVec[i], 0);
    if (Math.hypot(...gNew) < 1e-7) {
      return { x: xNew, success: true };
    }
    if (ys > 1e-12) {
      // H ← (I - ρ s yᵀ) H (I - ρ y sᵀ) + ρ s sᵀ
      const rho = 1 / ys;
      const Hy = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += H[i][j] * yVec[j];
        Hy[i] = s;
      }
      const yHy = yVec.reduce((a, yi, i) => a + yi * Hy[i], 0);
      const H2: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          H2[i][j] =
            H[i][j] -
            rho * (sVec[i] * Hy[j] + Hy[i] * sVec[j]) +
            rho * rho * yHy * sVec[i] * sVec[j] +
            rho * sVec[i] * sVec[j];
        }
      }
      H = H2;
    }
    x = xNew;
    g = gNew;
    if (Math.abs(f0 - fNew) < 1e-12) break;
  }
  return { x, success: true };
}

export function fitSharedExponents(
  runs: RunBundle,
  reference = "adamw",
): FitSharedResult {
  const optNames = Object.keys(runs);
  if (!optNames.includes(reference)) {
    throw new Error(`reference optimizer ${reference} missing from runs`);
  }
  const nonRef = optNames.filter((o) => o !== reference);
  const x0 = new Array(5 + 2 * nonRef.length).fill(0);
  x0[0] = Math.log(DEFAULT_PARAMS.E);
  x0[1] = Math.log(DEFAULT_PARAMS.A);
  x0[2] = Math.log(DEFAULT_PARAMS.B);
  x0[3] = DEFAULT_PARAMS.alpha;
  x0[4] = DEFAULT_PARAMS.beta;
  nonRef.forEach((_, i) => {
    x0[5 + 2 * i] = 0;
    x0[5 + 2 * i + 1] = 0;
  });

  const { x, success } = bfgsMinimize(x0, runs, optNames, reference);
  const { params, rhos } = unpack(x, optNames, reference);
  const per: Record<string, number> = {};
  for (const name of optNames) {
    const data = runs[name];
    let sse = 0;
    for (let i = 0; i < data.L.length; i++) {
    const pred = loss(data.N[i], data.D[i], rhos[name], params);
      sse += (pred - data.L[i]) ** 2;
    }
    per[name] = Math.sqrt(sse / data.L.length);
  }
  return {
    params,
    rhos: Object.fromEntries(
      Object.entries(rhos).map(([k, v]) => [k, { rho_n: v.rho_n, rho_d: v.rho_d, label: v.label }]),
    ),
    per_optimizer_rmse: per,
    mean_rmse: Object.values(per).reduce((a, b) => a + b, 0) / Math.max(Object.values(per).length, 1),
    success,
    method: "shared_exponents_lbfgs_js",
  };
}
