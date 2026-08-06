import "./styles.css";
import {
  OPTIMIZERS,
  GPUS,
  PRESETS,
  allocate,
  allocateFixedRatio,
  compareAtBudget,
  flopsFromGpus,
  formatFlops,
  formatParams,
  getGpu,
  getOptimizer,
  isoflopCurve,
  wallclock,
  type OptimizerMeta,
} from "./engine";

type State = {
  theme: "dark" | "light";
  tab: "practice" | "research" | "formula";
  flops: number;
  optimizer: string;
  gpu: string;
  count: number;
  mfu: number;
  ratioMode: boolean;
  ratio: number;
  selected: string[];
};

const STORAGE_KEY = "optiscale-scenario-v1";

function defaultState(): State {
  return {
    theme: "dark",
    tab: "practice",
    flops: 1e24,
    optimizer: "muon",
    gpu: "h100",
    count: 8,
    mfu: 0.4,
    ratioMode: false,
    ratio: 100,
    selected: ["adamw", "muon", "normuon", "soap"],
  };
}

function loadState(): State {
  const params = new URLSearchParams(location.search);
  const base = defaultState();
  if (params.get("C")) base.flops = Number(params.get("C"));
  if (params.get("opt")) base.optimizer = params.get("opt")!;
  if (params.get("gpu")) base.gpu = params.get("gpu")!;
  if (params.get("n")) base.count = Number(params.get("n"));
  if (params.get("mfu")) base.mfu = Number(params.get("mfu"));
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) Object.assign(base, JSON.parse(raw));
  } catch {
    /* ignore */
  }
  // URL wins for shareable links
  if (params.get("C")) base.flops = Number(params.get("C"));
  if (params.get("opt")) base.optimizer = params.get("opt")!;
  return base;
}

let state = loadState();

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  const q = new URLSearchParams({
    C: String(state.flops),
    opt: state.optimizer,
    gpu: state.gpu,
    n: String(state.count),
    mfu: String(state.mfu),
  });
  history.replaceState(null, "", `?${q.toString()}`);
}

function el<T extends HTMLElement>(sel: string): T {
  const n = document.querySelector(sel);
  if (!n) throw new Error(`Missing ${sel}`);
  return n as T;
}

function renderShell() {
  document.documentElement.dataset.theme = state.theme;
  document.querySelector("#app")!.innerHTML = `
    <header class="hero">
      <div class="claim">Chinchilla assumes AdamW — change the optimizer, redo the budget</div>
      <h1 class="brand">Opti<span>Scale</span></h1>
      <p class="tagline">
        Optimizer-aware scaling lab: allocate compute-optimal N* and D*, cost it on real GPUs,
        compare Muon / NorMuon / SOAP against AdamW, and fit shared-exponent ρ models.
      </p>
    </header>
    <div class="toolbar">
      <div class="tabs" role="tablist">
        <button class="tab ${state.tab === "practice" ? "active" : ""}" data-tab="practice">Practitioner</button>
        <button class="tab ${state.tab === "research" ? "active" : ""}" data-tab="research">Research fit</button>
        <button class="tab ${state.tab === "formula" ? "active" : ""}" data-tab="formula">Formulas</button>
      </div>
      <button class="icon-btn" id="themeBtn">${state.theme === "dark" ? "Light" : "Dark"}</button>
      <button class="icon-btn" id="shareBtn">Copy share link</button>
      <button class="icon-btn" id="exportMd">Export Markdown</button>
      <button class="icon-btn" id="exportCsv">Export CSV</button>
    </div>
    <div id="view"></div>
    <footer class="footer">
      Part of the <strong>Spectral Training Stack</strong> with
      <a href="https://github.com/spectral-training/spectoptim">SpectOptim</a> and
      <a href="https://github.com/spectral-training/ortholab">OrthoLab</a>.
      ρ priors are planning defaults — fit your own runs for production.
      Inspired by Hoffmann et al. (2022) and Volkova et al. (2026).
      <code class="mono">pip install optiscale</code>
    </footer>
  `;
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.tab = (btn as HTMLElement).dataset.tab as State["tab"];
      saveState();
      render();
    });
  });
  el("#themeBtn").onclick = () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    saveState();
    render();
  };
  el("#shareBtn").onclick = async () => {
    saveState();
    await navigator.clipboard.writeText(location.href);
    el("#shareBtn").textContent = "Copied";
    setTimeout(() => (el("#shareBtn").textContent = "Copy share link"), 1200);
  };
  el("#exportMd").onclick = exportMarkdown;
  el("#exportCsv").onclick = exportCsv;
}

function renderPractice() {
  const gpu = getGpu(state.gpu);
  const opt = getOptimizer(state.optimizer);
  const alloc = state.ratioMode
    ? allocateFixedRatio(state.flops, state.ratio, opt)
    : allocate(state.flops, opt);
  const wall = wallclock(state.flops, gpu, state.count, state.mfu);
  const rows = compareAtBudget(state.flops, state.selected);

  el("#view").innerHTML = `
    <div class="grid">
      <aside class="panel">
        <h2>Budget</h2>
        <div class="row" id="presets"></div>
        <label>FLOP budget C
          <input id="flopsInput" type="text" value="${state.flops.toExponential(3)}" />
        </label>
        <label>Or GPU-hours → FLOPs
          <div class="row">
            <select id="gpuSel">${GPUS.map((g) => `<option value="${g.id}" ${g.id === state.gpu ? "selected" : ""}>${g.name}</option>`).join("")}</select>
          </div>
        </label>
        <label>GPU count
          <input id="countInput" type="number" min="1" value="${state.count}" />
        </label>
        <label>Hours
          <input id="hoursInput" type="number" min="0.1" step="0.1" value="${(wall.hours).toFixed(1)}" />
        </label>
        <label>MFU ${(state.mfu * 100).toFixed(0)}%
          <input id="mfuInput" type="range" min="0.15" max="0.7" step="0.01" value="${state.mfu}" />
        </label>
        <label>Optimizer
          <select id="optSel">${OPTIMIZERS.map((o) => `<option value="${o.id}" ${o.id === state.optimizer ? "selected" : ""}>${o.label} (ρN=${o.rho_n}, ρD=${o.rho_d})</option>`).join("")}</select>
        </label>
        <label><input id="ratioMode" type="checkbox" ${state.ratioMode ? "checked" : ""}/> Fix tokens/param ratio</label>
        <label class="${state.ratioMode ? "" : "hidden"}">Tokens / param
          <input id="ratioInput" type="number" min="1" value="${state.ratio}" />
        </label>
        <p class="muted">Presets and URL state are saved locally. Override ρ in the Research tab after fitting.</p>
      </aside>
      <section class="panel">
        <h2>Allocation — ${opt.label}</h2>
        <div class="cards">
          <div class="card"><div class="k">N*</div><div class="v accent">${formatParams(alloc.N)}</div></div>
          <div class="card"><div class="k">D*</div><div class="v">${formatParams(alloc.D)}</div></div>
          <div class="card"><div class="k">tok / param</div><div class="v">${alloc.tokensPerParam.toFixed(1)}</div></div>
          <div class="card"><div class="k">Pred. loss</div><div class="v">${alloc.loss.toFixed(4)}</div></div>
          <div class="card"><div class="k">Wall-clock</div><div class="v">${wall.days.toFixed(2)} d</div></div>
          <div class="card"><div class="k">Est. $</div><div class="v">$${wall.cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div></div>
        </div>
        <h2>IsoFLOP curve</h2>
        <canvas id="isoCanvas" width="900" height="360"></canvas>
        <h2>Multi-optimizer compare</h2>
        <div class="row" id="optChips"></div>
        <table>
          <thead>
            <tr><th>Opt</th><th>N*</th><th>D*</th><th>Loss</th><th>N / AdamW</th><th>Compute saved†</th></tr>
          </thead>
          <tbody>
            ${rows
              .map(
                (r) => `<tr>
              <td>${r.label}</td>
              <td class="mono">${formatParams(r.N)}</td>
              <td class="mono">${formatParams(r.D)}</td>
              <td class="mono">${r.loss.toFixed(4)}</td>
              <td class="mono">${r.nRatio.toFixed(3)}</td>
              <td class="mono ${r.savings > 0 ? "delta-pos" : ""}">${(r.savings * 100).toFixed(1)}%</td>
            </tr>`,
              )
              .join("")}
          </tbody>
        </table>
        <p class="muted">† Compute saved to match AdamW's predicted loss at this budget (same target L).</p>
        <div class="export-row">
          <button class="icon-btn" id="copyPy">Copy Python snippet</button>
        </div>
        <pre class="muted mono" id="pySnippet"></pre>
      </section>
    </div>
  `;

  const presets = el<HTMLDivElement>("#presets");
  PRESETS.forEach((p) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = p.label;
    b.onclick = () => applyPreset(p.id);
    presets.appendChild(b);
  });

  const chips = el<HTMLDivElement>("#optChips");
  OPTIMIZERS.forEach((o) => {
    const b = document.createElement("button");
    b.className = `chip ${state.selected.includes(o.id) ? "active" : ""}`;
    b.textContent = o.label;
    b.onclick = () => {
      if (state.selected.includes(o.id)) state.selected = state.selected.filter((x) => x !== o.id);
      else state.selected = [...state.selected, o.id];
      if (!state.selected.length) state.selected = ["adamw"];
      saveState();
      render();
    };
    chips.appendChild(b);
  });

  el<HTMLInputElement>("#flopsInput").onchange = (e) => {
    state.flops = Number((e.target as HTMLInputElement).value);
    saveState();
    render();
  };
  el<HTMLSelectElement>("#gpuSel").onchange = (e) => {
    state.gpu = (e.target as HTMLSelectElement).value;
    const g = getGpu(state.gpu);
    state.mfu = g.defaultMfu;
    saveState();
    render();
  };
  el<HTMLInputElement>("#countInput").onchange = (e) => {
    state.count = Math.max(1, Number((e.target as HTMLInputElement).value));
    saveState();
    render();
  };
  el<HTMLInputElement>("#hoursInput").onchange = (e) => {
    const hours = Number((e.target as HTMLInputElement).value);
    state.flops = flopsFromGpus(getGpu(state.gpu), state.count, hours, state.mfu);
    saveState();
    render();
  };
  el<HTMLInputElement>("#mfuInput").oninput = (e) => {
    state.mfu = Number((e.target as HTMLInputElement).value);
    saveState();
    render();
  };
  el<HTMLSelectElement>("#optSel").onchange = (e) => {
    state.optimizer = (e.target as HTMLSelectElement).value;
    saveState();
    render();
  };
  el<HTMLInputElement>("#ratioMode").onchange = (e) => {
    state.ratioMode = (e.target as HTMLInputElement).checked;
    saveState();
    render();
  };
  const ratioInput = document.querySelector("#ratioInput") as HTMLInputElement | null;
  if (ratioInput) {
    ratioInput.onchange = (e) => {
      state.ratio = Number((e.target as HTMLInputElement).value);
      saveState();
      render();
    };
  }
  el("#copyPy").onclick = async () => {
    const snip = `from optiscale import allocate_for_budget, cost_report
alloc = allocate_for_budget(${state.flops.toExponential()}, optimizer="${state.optimizer}")
print(alloc)
print(cost_report(compute=${state.flops.toExponential()}, gpu_id="${state.gpu}", count=${state.count}))
`;
    el("#pySnippet").textContent = snip;
    await navigator.clipboard.writeText(snip);
  };

  drawIsoflop(opt);
}

function drawIsoflop(opt: OptimizerMeta) {
  const canvas = el<HTMLCanvasElement>("#isoCanvas");
  const ctx = canvas.getContext("2d")!;
  const curve = isoflopCurve(state.flops, opt);
  const adamwCurve = isoflopCurve(state.flops, getOptimizer("adamw"));
  const w = canvas.width;
  const h = canvas.height;
  const pad = 48;
  ctx.clearRect(0, 0, w, h);
  const allL = [...curve.loss, ...adamwCurve.loss];
  const minL = Math.min(...allL);
  const maxL = Math.max(...allL);
  const minN = Math.min(...curve.N);
  const maxN = Math.max(...curve.N);

  const xOf = (n: number) => pad + ((Math.log(n) - Math.log(minN)) / (Math.log(maxN) - Math.log(minN))) * (w - 2 * pad);
  const yOf = (l: number) => h - pad - ((l - minL) / (maxL - minL || 1)) * (h - 2 * pad);

  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--line");
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();

  const draw = (c: typeof curve, color: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    c.N.forEach((n, i) => {
      const x = xOf(n);
      const y = yOf(c.loss[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const s = c.star;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(xOf(s.N), yOf(s.loss), 5, 0, Math.PI * 2);
    ctx.fill();
  };

  draw(adamwCurve, getComputedStyle(document.documentElement).getPropertyValue("--adamw").trim() || "#7aa2c8");
  draw(curve, getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#3dcf8e");

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
  ctx.font = "12px IBM Plex Mono";
  ctx.fillText("N (log)", w / 2 - 20, h - 14);
  ctx.save();
  ctx.translate(14, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Loss", 0, 0);
  ctx.restore();
  ctx.fillText("AdamW", w - pad - 120, pad + 8);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  ctx.fillText(opt.label, w - pad - 120, pad + 24);
}

function applyPreset(id: string) {
  const p = PRESETS.find((x) => x.id === id);
  if (!p) return;
  if (p.kind === "flops") {
    state.flops = p.value;
    state.ratioMode = Boolean(p.ratio);
    if (p.ratio) state.ratio = p.ratio;
  } else if (p.kind === "usd") {
    state.gpu = p.gpu;
    state.count = p.count;
    const g = getGpu(state.gpu);
    state.mfu = g.defaultMfu;
    const hours = p.value / (g.usdPerHour * state.count);
    state.flops = flopsFromGpus(g, state.count, hours, state.mfu);
  } else if (p.kind === "gpuHours") {
    state.gpu = p.gpu;
    state.count = p.count;
    state.mfu = getGpu(state.gpu).defaultMfu;
    state.flops = flopsFromGpus(getGpu(state.gpu), state.count, p.hours, state.mfu);
  }
  saveState();
  render();
}

function renderResearch() {
  el("#view").innerHTML = `
    <div class="panel">
      <h2>Shared-exponent vs separate fits</h2>
      <p class="muted">
        Paste CSV with columns <code class="mono">N,D,L,optimizer</code> or run a synthetic IsoFLOP demo.
        Separate per-optimizer Chinchilla fits are often ill-conditioned; the shared α,β + ρ model is more stable.
      </p>
      <textarea id="csvIn" rows="8" placeholder="N,D,L,optimizer&#10;1e8,2e9,3.2,adamw&#10;..."></textarea>
      <div class="row" style="margin-top:0.75rem">
        <button class="icon-btn" id="synthBtn">Load synthetic demo</button>
        <button class="icon-btn" id="fitBtn">Compare fit strategies</button>
      </div>
      <div id="fitOut" class="muted">Results appear here.</div>
    </div>
  `;
  el("#synthBtn").onclick = () => {
    const lines = ["N,D,L,optimizer"];
    for (const opt of ["adamw", "muon"]) {
      const o = getOptimizer(opt);
      const budgets = [1e19, 1e20, 1e21, 1e22];
      for (const c of budgets) {
        const center = allocate(c, o).N;
        for (let i = 0; i < 8; i++) {
          const n = center / 6 * Math.pow(36, i / 7);
          const d = c / (6 * n);
          const ne = n * o.rho_n;
          const de = d * o.rho_d;
          const L =
            1.69 + 406.4 / ne ** 0.34 + 410.7 / de ** 0.28 + (Math.random() - 0.5) * 0.02;
          lines.push(`${n},${d},${L},${opt}`);
        }
      }
    }
    el<HTMLTextAreaElement>("#csvIn").value = lines.join("\n");
  };
  el("#fitBtn").onclick = () => {
    const text = el<HTMLTextAreaElement>("#csvIn").value.trim();
    if (!text) {
      el("#fitOut").textContent = "Paste CSV or load synthetic demo first.";
      return;
    }
    const rows = text
      .split(/\r?\n/)
      .slice(1)
      .map((line) => line.split(","))
      .filter((p) => p.length >= 4)
      .map((p) => ({ N: Number(p[0]), D: Number(p[1]), L: Number(p[2]), optimizer: p[3].trim().toLowerCase() }));
    const byOpt: Record<string, { N: number[]; D: number[]; L: number[] }> = {};
    for (const r of rows) {
      byOpt[r.optimizer] ??= { N: [], D: [], L: [] };
      byOpt[r.optimizer].N.push(r.N);
      byOpt[r.optimizer].D.push(r.D);
      byOpt[r.optimizer].L.push(r.L);
    }
    // Lightweight browser summary: mean residuals under default shared ρ vs AdamW-only
    const summary: string[] = ["Browser fit summary (full L-BFGS lives in Python `optiscale fit`):", ""];
    for (const [name, data] of Object.entries(byOpt)) {
      const o = OPTIMIZERS.find((x) => x.id === name) ?? { id: name, label: name, rho_n: 1, rho_d: 1 };
      let sse = 0;
      for (let i = 0; i < data.N.length; i++) {
        const ne = data.N[i] * o.rho_n;
        const de = data.D[i] * o.rho_d;
        const pred = 1.69 + 406.4 / ne ** 0.34 + 410.7 / de ** 0.28;
        sse += (pred - data.L[i]) ** 2;
      }
      const rmse = Math.sqrt(sse / data.N.length);
      summary.push(`${o.label}: n=${data.N.length}  RMSE under prior ρ = ${rmse.toFixed(4)}  (ρN=${o.rho_n}, ρD=${o.rho_d})`);
    }
    summary.push("", "Run for full shared-exponent fit + bootstrap CI:");
    summary.push("  pip install -e .");
    summary.push("  optiscale fit --synthetic");
    summary.push("  optiscale fit --data runs.csv");
    el("#fitOut").textContent = summary.join("\n");
  };
}

function renderFormula() {
  el("#view").innerHTML = `
    <div class="panel">
      <h2>Formulas</h2>
      <div class="formula">L(N, D) = E + A / N<sup>α</sup> + B / D<sup>β</sup></div>
      <div class="formula">C ≈ 6 N D</div>
      <div class="formula">N<sub>eff</sub> = ρ<sub>N</sub> N,&nbsp; D<sub>eff</sub> = ρ<sub>D</sub> D &nbsp;(optimizer-aware)</div>
      <p>
        Classic Chinchilla (Hoffmann et al., 2022) calibrates α, β, A, B, E under AdamW.
        Volkova et al. (2026) show separate per-optimizer fits are ill-conditioned; a robust law
        keeps shared exponents and introduces interpretable ρ<sub>N</sub>, ρ<sub>D</sub> relative to AdamW.
      </p>
      <p class="muted">
        Default coefficients: E=1.69, A=406.4, B=410.7, α=0.34, β=0.28.
        OptiScale ρ priors (Muon 1.35/1.25, NorMuon 1.45/1.30, …) are planning estimates — replace via fit.
      </p>
      <h2>Python</h2>
      <pre class="mono muted">pip install optiscale
optiscale allocate --flops 1e24 --optimizer muon
optiscale compare --budget-usd 1000000 --gpu h100 --count 64
optiscale report --flops 1e24 --md report.md</pre>
    </div>
  `;
}

function exportMarkdown() {
  const rows = compareAtBudget(state.flops, state.selected);
  const gpu = getGpu(state.gpu);
  const wall = wallclock(state.flops, gpu, state.count, state.mfu);
  const md = [
    "# OptiScale Report",
    "",
    `- Compute: ${formatFlops(state.flops)}`,
    `- GPU: ${gpu.name} × ${state.count} @ MFU ${state.mfu}`,
    `- Wall-clock: ${wall.days.toFixed(2)} days`,
    `- Cost: $${wall.cost.toFixed(0)}`,
    "",
    "| Optimizer | N* | D* | Loss | N/AdamW | Compute saved |",
    "|---|---:|---:|---:|---:|---:|",
    ...rows.map(
      (r) =>
        `| ${r.label} | ${formatParams(r.N)} | ${formatParams(r.D)} | ${r.loss.toFixed(4)} | ${r.nRatio.toFixed(3)} | ${(r.savings * 100).toFixed(1)}% |`,
    ),
    "",
  ].join("\n");
  download("optiscale-report.md", md);
}

function exportCsv() {
  const rows = compareAtBudget(state.flops, state.selected);
  const lines = [
    "optimizer,N,D,loss,tokens_per_param,n_ratio,compute_saved",
    ...rows.map(
      (r) =>
        `${r.id},${r.N},${r.D},${r.loss},${r.tokensPerParam},${r.nRatio},${r.savings}`,
    ),
  ];
  download("optiscale-compare.csv", lines.join("\n"));
}

function download(name: string, text: string) {
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function render() {
  renderShell();
  if (state.tab === "practice") renderPractice();
  else if (state.tab === "research") renderResearch();
  else renderFormula();
}

render();
