/** Browser port of OptiScale core math (mirrors Python package). */

export type ChinchillaParams = {
  E: number;
  A: number;
  B: number;
  alpha: number;
  beta: number;
};

export const DEFAULT_PARAMS: ChinchillaParams = {
  E: 1.69,
  A: 406.4,
  B: 410.7,
  alpha: 0.34,
  beta: 0.28,
};

export type OptimizerMeta = {
  id: string;
  label: string;
  rho_n: number;
  rho_d: number;
};

export const OPTIMIZERS: OptimizerMeta[] = [
  { id: "adamw", label: "AdamW", rho_n: 1.0, rho_d: 1.0 },
  { id: "muon", label: "Muon", rho_n: 1.35, rho_d: 1.25 },
  { id: "normuon", label: "NorMuon", rho_n: 1.45, rho_d: 1.3 },
  { id: "aurora", label: "Aurora", rho_n: 1.4, rho_d: 1.28 },
  { id: "soap", label: "SOAP", rho_n: 1.2, rho_d: 1.15 },
  { id: "shampoo", label: "Shampoo", rho_n: 1.18, rho_d: 1.12 },
  { id: "lion", label: "Lion", rho_n: 1.05, rho_d: 1.02 },
];

export type GPUSpec = {
  id: string;
  name: string;
  peakTflops: number;
  usdPerHour: number;
  defaultMfu: number;
  memoryGb: number;
};

/** Mirrors Python `cost.GPU_CATALOG`. */
export const GPUS: GPUSpec[] = [
  { id: "h100", name: "H100 SXM", peakTflops: 989, usdPerHour: 4.0, defaultMfu: 0.4, memoryGb: 80 },
  { id: "h100_pcie", name: "H100 PCIe", peakTflops: 756, usdPerHour: 3.5, defaultMfu: 0.38, memoryGb: 80 },
  { id: "a100", name: "A100 80GB", peakTflops: 312, usdPerHour: 2.2, defaultMfu: 0.45, memoryGb: 80 },
  { id: "a100_40", name: "A100 40GB", peakTflops: 312, usdPerHour: 1.8, defaultMfu: 0.42, memoryGb: 40 },
  { id: "l40s", name: "L40S", peakTflops: 362, usdPerHour: 1.6, defaultMfu: 0.35, memoryGb: 48 },
  { id: "rtx4090", name: "RTX 4090", peakTflops: 330, usdPerHour: 0.8, defaultMfu: 0.3, memoryGb: 24 },
  { id: "v100", name: "V100", peakTflops: 125, usdPerHour: 0.9, defaultMfu: 0.4, memoryGb: 32 },
  { id: "tpu_v5e", name: "TPU v5e", peakTflops: 197, usdPerHour: 1.2, defaultMfu: 0.5, memoryGb: 16 },
];

export const PRESETS = [
  { id: "1m", label: "$1M pretrain", kind: "usd" as const, value: 1_000_000, gpu: "h100", count: 64 },
  { id: "8xh100-30d", label: "8×H100 × 30 days", kind: "gpuHours" as const, hours: 24 * 30, gpu: "h100", count: 8 },
  { id: "70b", label: "Chinchilla 70B-ish", kind: "flops" as const, value: 6 * 70e9 * 1.4e12 },
  { id: "1e24", label: "10²⁴ FLOPs", kind: "flops" as const, value: 1e24 },
  { id: "overtrain", label: "Overtrain @ 100 tok/param", kind: "flops" as const, value: 1e23, ratio: 100 },
];

export function getOptimizer(id: string): OptimizerMeta {
  const o = OPTIMIZERS.find((x) => x.id === id);
  if (!o) throw new Error(`Unknown optimizer ${id}`);
  return o;
}

export function getGpu(id: string): GPUSpec {
  const g = GPUS.find((x) => x.id === id);
  if (!g) throw new Error(`Unknown GPU ${id}`);
  return g;
}

export function withRho(opt: OptimizerMeta, rho_n?: number, rho_d?: number): OptimizerMeta {
  return {
    ...opt,
    rho_n: rho_n ?? opt.rho_n,
    rho_d: rho_d ?? opt.rho_d,
  };
}

export function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
  return n.toFixed(0);
}

export function formatFlops(c: number): string {
  if (c >= 1e24) return `${(c / 1e24).toFixed(3)}×10²⁴`;
  if (c >= 1e21) return `${(c / 1e21).toFixed(2)} ZFLOP`;
  if (c >= 1e18) return `${(c / 1e18).toFixed(2)} EFLOP`;
  return c.toExponential(2);
}

export function loss(
  n: number,
  d: number,
  opt: OptimizerMeta,
  p: ChinchillaParams = DEFAULT_PARAMS,
): number {
  const ne = n * opt.rho_n;
  const de = d * opt.rho_d;
  return p.E + p.A / ne ** p.alpha + p.B / de ** p.beta;
}

export function allocate(
  compute: number,
  opt: OptimizerMeta,
  p: ChinchillaParams = DEFAULT_PARAMS,
): { N: number; D: number; loss: number; tokensPerParam: number } {
  // Minimize L=E+A/(ρN N)^α+B/(ρD D)^β s.t. C=6ND
  const a = p.alpha;
  const b = p.beta;
  const gStar = ((p.A * a) / (p.B * b)) ** (1 / (a + b));
  const rhoPref = (opt.rho_d ** b / opt.rho_n ** a) ** (1 / (a + b));
  const n = gStar * rhoPref * (compute / 6) ** (b / (a + b));
  const d = compute / (6 * n);
  return { N: n, D: d, loss: loss(n, d, opt, p), tokensPerParam: d / n };
}

export function allocateFixedRatio(
  compute: number,
  ratio: number,
  opt: OptimizerMeta,
  p: ChinchillaParams = DEFAULT_PARAMS,
) {
  const n = Math.sqrt(compute / (6 * ratio));
  const d = ratio * n;
  return { N: n, D: d, loss: loss(n, d, opt, p), tokensPerParam: ratio };
}

export function isoflopCurve(
  compute: number,
  opt: OptimizerMeta,
  points = 64,
  span = 30,
  p: ChinchillaParams = DEFAULT_PARAMS,
) {
  const center = allocate(compute, opt, p).N;
  const ns: number[] = [];
  const ds: number[] = [];
  const ls: number[] = [];
  for (let i = 0; i < points; i++) {
    const t = i / (points - 1);
    const n = (center / span) * Math.pow(span * span, t);
    const d = compute / (6 * n);
    ns.push(n);
    ds.push(d);
    ls.push(loss(n, d, opt, p));
  }
  let iMin = 0;
  for (let i = 1; i < ls.length; i++) if (ls[i] < ls[iMin]) iMin = i;
  return { N: ns, D: ds, loss: ls, iMin, star: { N: ns[iMin], D: ds[iMin], loss: ls[iMin] } };
}

export function flopsFromGpus(gpu: GPUSpec, count: number, hours: number, mfu?: number) {
  const m = mfu ?? gpu.defaultMfu;
  return gpu.peakTflops * 1e12 * m * count * hours * 3600;
}

export function wallclock(compute: number, gpu: GPUSpec, count: number, mfu?: number) {
  const m = mfu ?? gpu.defaultMfu;
  const flopsS = gpu.peakTflops * 1e12 * m * count;
  const hours = compute / flopsS / 3600;
  return {
    hours,
    days: hours / 24,
    cost: hours * gpu.usdPerHour * count,
    mfu: m,
  };
}

/** Approximate inverse: min compute to hit target loss on ridge. */
export function computeForLoss(
  target: number,
  opt: OptimizerMeta,
  p: ChinchillaParams = DEFAULT_PARAMS,
): number {
  let lo = Math.log(1e18);
  let hi = Math.log(1e28);
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    const c = Math.exp(mid);
    const l = allocate(c, opt, p).loss;
    if (l > target) lo = mid;
    else hi = mid;
  }
  return Math.exp(0.5 * (lo + hi));
}

export function compareAtBudget(
  compute: number,
  ids: string[] = OPTIMIZERS.map((o) => o.id),
  optOverride?: (id: string) => OptimizerMeta,
) {
  const adamw = optOverride?.("adamw") ?? getOptimizer("adamw");
  const base = allocate(compute, adamw);
  return ids.map((id) => {
    const opt = optOverride?.(id) ?? getOptimizer(id);
    const alloc = allocate(compute, opt);
    const matchCompute = computeForLoss(base.loss, opt);
    return {
      ...opt,
      ...alloc,
      nRatio: alloc.N / base.N,
      savings: 1 - matchCompute / compute,
      matchCompute,
    };
  });
}

/** ρ_N × ρ_D → relative ΔN* vs AdamW at fixed budget (sensitivity heatmap). */
export function rhoSensitivityGrid(
  compute: number,
  rhoNRange: [number, number] = [0.8, 2.0],
  rhoDRange: [number, number] = [0.8, 2.0],
  steps = 24,
  p: ChinchillaParams = DEFAULT_PARAMS,
): { rhoN: number[]; rhoD: number[]; deltaN: number[][]; loss: number[][] } {
  const adamw = allocate(compute, getOptimizer("adamw"), p);
  const rhoN: number[] = [];
  const rhoD: number[] = [];
  for (let i = 0; i < steps; i++) {
    const t = i / (steps - 1);
    rhoN.push(rhoNRange[0] + t * (rhoNRange[1] - rhoNRange[0]));
    rhoD.push(rhoDRange[0] + t * (rhoDRange[1] - rhoDRange[0]));
  }
  const deltaN: number[][] = [];
  const lossGrid: number[][] = [];
  for (let j = 0; j < steps; j++) {
    const rowN: number[] = [];
    const rowL: number[] = [];
    for (let i = 0; i < steps; i++) {
      const opt: OptimizerMeta = {
        id: "custom",
        label: "custom",
        rho_n: rhoN[i],
        rho_d: rhoD[j],
      };
      const alloc = allocate(compute, opt, p);
      rowN.push((alloc.N - adamw.N) / adamw.N);
      rowL.push(alloc.loss);
    }
    deltaN.push(rowN);
    lossGrid.push(rowL);
  }
  return { rhoN, rhoD, deltaN, loss: lossGrid };
}
