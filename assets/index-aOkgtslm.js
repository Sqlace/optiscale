(function(){const a=document.createElement("link").relList;if(a&&a.supports&&a.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const n of i)if(n.type==="childList")for(const l of n.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&r(l)}).observe(document,{childList:!0,subtree:!0});function t(i){const n={};return i.integrity&&(n.integrity=i.integrity),i.referrerPolicy&&(n.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?n.credentials="include":i.crossOrigin==="anonymous"?n.credentials="omit":n.credentials="same-origin",n}function r(i){if(i.ep)return;i.ep=!0;const n=t(i);fetch(i.href,n)}})();const w={E:1.69,A:406.4,B:410.7,alpha:.34,beta:.28},N=[{id:"adamw",label:"AdamW",rho_n:1,rho_d:1},{id:"muon",label:"Muon",rho_n:1.35,rho_d:1.25},{id:"normuon",label:"NorMuon",rho_n:1.45,rho_d:1.3},{id:"aurora",label:"Aurora",rho_n:1.4,rho_d:1.28},{id:"soap",label:"SOAP",rho_n:1.2,rho_d:1.15},{id:"shampoo",label:"Shampoo",rho_n:1.18,rho_d:1.12},{id:"lion",label:"Lion",rho_n:1.05,rho_d:1.02}],z=[{id:"h100",name:"H100 SXM",peakTflops:989,usdPerHour:4,defaultMfu:.4},{id:"a100",name:"A100 80GB",peakTflops:312,usdPerHour:2.2,defaultMfu:.45},{id:"l40s",name:"L40S",peakTflops:362,usdPerHour:1.6,defaultMfu:.35},{id:"rtx4090",name:"RTX 4090",peakTflops:330,usdPerHour:.8,defaultMfu:.3},{id:"v100",name:"V100",peakTflops:125,usdPerHour:.9,defaultMfu:.4}],A=[{id:"1m",label:"$1M pretrain",kind:"usd",value:1e6,gpu:"h100",count:64},{id:"8xh100-30d",label:"8×H100 × 30 days",kind:"gpuHours",hours:24*30,gpu:"h100",count:8},{id:"70b",label:"Chinchilla 70B-ish",kind:"flops",value:6*7e10*14e11},{id:"1e24",label:"10²⁴ FLOPs",kind:"flops",value:1e24},{id:"overtrain",label:"Overtrain @ 100 tok/param",kind:"flops",value:1e23,ratio:100}];function M(o){const a=N.find(t=>t.id===o);if(!a)throw new Error(`Unknown optimizer ${o}`);return a}function $(o){const a=z.find(t=>t.id===o);if(!a)throw new Error(`Unknown GPU ${o}`);return a}function x(o){return o>=1e12?`${(o/1e12).toFixed(2)}T`:o>=1e9?`${(o/1e9).toFixed(2)}B`:o>=1e6?`${(o/1e6).toFixed(2)}M`:o>=1e3?`${(o/1e3).toFixed(2)}K`:o.toFixed(0)}function U(o){return o>=1e24?`${(o/1e24).toFixed(3)}×10²⁴`:o>=1e21?`${(o/1e21).toFixed(2)} ZFLOP`:o>=1e18?`${(o/1e18).toFixed(2)} EFLOP`:o.toExponential(2)}function S(o,a,t,r=w){const i=o*t.rho_n,n=a*t.rho_d;return r.E+r.A/i**r.alpha+r.B/n**r.beta}function y(o,a,t=w){const r=t.alpha,i=t.beta;let l=(t.A*r/(t.B*i))**(1/(r+i))*(o/6)**(i/(r+i)),c=o/6/l,s=l/a.rho_n,d=c/a.rho_d;const p=Math.sqrt(o/(6*s*d));return s*=p,d*=p,{N:s,D:d,loss:S(s,d,a,t),tokensPerParam:d/s}}function W(o,a,t,r=w){const i=Math.sqrt(o/(6*a)),n=a*i;return{N:i,D:n,loss:S(i,n,t,r),tokensPerParam:a}}function T(o,a,t=64,r=30,i=w){const n=y(o,a,i).N,l=[],c=[],s=[];for(let p=0;p<t;p++){const h=p/(t-1),b=n/r*Math.pow(r*r,h),v=o/(6*b);l.push(b),c.push(v),s.push(S(b,v,a,i))}let d=0;for(let p=1;p<s.length;p++)s[p]<s[d]&&(d=p);return{N:l,D:c,loss:s,iMin:d,star:{N:l[d],D:c[d],loss:s[d]}}}function P(o,a,t,r){const i=r??o.defaultMfu;return o.peakTflops*1e12*i*a*t*3600}function I(o,a,t,r){const i=r??a.defaultMfu,n=a.peakTflops*1e12*i*t,l=o/n/3600;return{hours:l,days:l/24,cost:l*a.usdPerHour*t,mfu:i}}function j(o,a,t=w){let r=Math.log(1e18),i=Math.log(1e28);for(let n=0;n<60;n++){const l=.5*(r+i),c=Math.exp(l);y(c,a,t).loss>o?r=l:i=l}return Math.exp(.5*(r+i))}function k(o,a=N.map(t=>t.id)){const t=M("adamw"),r=y(o,t);return a.map(i=>{const n=M(i),l=y(o,n),c=j(r.loss,n);return{...n,...l,nRatio:l.N/r.N,savings:1-c/o,matchCompute:c}})}const B="optiscale-scenario-v1";function G(){return{theme:"dark",tab:"practice",flops:1e24,optimizer:"muon",gpu:"h100",count:8,mfu:.4,ratioMode:!1,ratio:100,selected:["adamw","muon","normuon","soap"]}}function V(){const o=new URLSearchParams(location.search),a=G();o.get("C")&&(a.flops=Number(o.get("C"))),o.get("opt")&&(a.optimizer=o.get("opt")),o.get("gpu")&&(a.gpu=o.get("gpu")),o.get("n")&&(a.count=Number(o.get("n"))),o.get("mfu")&&(a.mfu=Number(o.get("mfu")));try{const t=localStorage.getItem(B);t&&Object.assign(a,JSON.parse(t))}catch{}return o.get("C")&&(a.flops=Number(o.get("C"))),o.get("opt")&&(a.optimizer=o.get("opt")),a}let e=V();function m(){localStorage.setItem(B,JSON.stringify(e));const o=new URLSearchParams({C:String(e.flops),opt:e.optimizer,gpu:e.gpu,n:String(e.count),mfu:String(e.mfu)});history.replaceState(null,"",`?${o.toString()}`)}function u(o){const a=document.querySelector(o);if(!a)throw new Error(`Missing ${o}`);return a}function q(){document.documentElement.dataset.theme=e.theme,document.querySelector("#app").innerHTML=`
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
        <button class="tab ${e.tab==="practice"?"active":""}" data-tab="practice">Practitioner</button>
        <button class="tab ${e.tab==="research"?"active":""}" data-tab="research">Research fit</button>
        <button class="tab ${e.tab==="formula"?"active":""}" data-tab="formula">Formulas</button>
      </div>
      <button class="icon-btn" id="themeBtn">${e.theme==="dark"?"Light":"Dark"}</button>
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
  `,document.querySelectorAll(".tab").forEach(o=>{o.addEventListener("click",()=>{e.tab=o.dataset.tab,m(),f()})}),u("#themeBtn").onclick=()=>{e.theme=e.theme==="dark"?"light":"dark",m(),f()},u("#shareBtn").onclick=async()=>{m(),await navigator.clipboard.writeText(location.href),u("#shareBtn").textContent="Copied",setTimeout(()=>u("#shareBtn").textContent="Copy share link",1200)},u("#exportMd").onclick=Q,u("#exportCsv").onclick=tt}function K(){const o=$(e.gpu),a=M(e.optimizer),t=e.ratioMode?W(e.flops,e.ratio,a):y(e.flops,a),r=I(e.flops,o,e.count,e.mfu),i=k(e.flops,e.selected);u("#view").innerHTML=`
    <div class="grid">
      <aside class="panel">
        <h2>Budget</h2>
        <div class="row" id="presets"></div>
        <label>FLOP budget C
          <input id="flopsInput" type="text" value="${e.flops.toExponential(3)}" />
        </label>
        <label>Or GPU-hours → FLOPs
          <div class="row">
            <select id="gpuSel">${z.map(s=>`<option value="${s.id}" ${s.id===e.gpu?"selected":""}>${s.name}</option>`).join("")}</select>
          </div>
        </label>
        <label>GPU count
          <input id="countInput" type="number" min="1" value="${e.count}" />
        </label>
        <label>Hours
          <input id="hoursInput" type="number" min="0.1" step="0.1" value="${r.hours.toFixed(1)}" />
        </label>
        <label>MFU ${(e.mfu*100).toFixed(0)}%
          <input id="mfuInput" type="range" min="0.15" max="0.7" step="0.01" value="${e.mfu}" />
        </label>
        <label>Optimizer
          <select id="optSel">${N.map(s=>`<option value="${s.id}" ${s.id===e.optimizer?"selected":""}>${s.label} (ρN=${s.rho_n}, ρD=${s.rho_d})</option>`).join("")}</select>
        </label>
        <label><input id="ratioMode" type="checkbox" ${e.ratioMode?"checked":""}/> Fix tokens/param ratio</label>
        <label class="${e.ratioMode?"":"hidden"}">Tokens / param
          <input id="ratioInput" type="number" min="1" value="${e.ratio}" />
        </label>
        <p class="muted">Presets and URL state are saved locally. Override ρ in the Research tab after fitting.</p>
      </aside>
      <section class="panel">
        <h2>Allocation — ${a.label}</h2>
        <div class="cards">
          <div class="card"><div class="k">N*</div><div class="v accent">${x(t.N)}</div></div>
          <div class="card"><div class="k">D*</div><div class="v">${x(t.D)}</div></div>
          <div class="card"><div class="k">tok / param</div><div class="v">${t.tokensPerParam.toFixed(1)}</div></div>
          <div class="card"><div class="k">Pred. loss</div><div class="v">${t.loss.toFixed(4)}</div></div>
          <div class="card"><div class="k">Wall-clock</div><div class="v">${r.days.toFixed(2)} d</div></div>
          <div class="card"><div class="k">Est. $</div><div class="v">$${r.cost.toLocaleString(void 0,{maximumFractionDigits:0})}</div></div>
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
            ${i.map(s=>`<tr>
              <td>${s.label}</td>
              <td class="mono">${x(s.N)}</td>
              <td class="mono">${x(s.D)}</td>
              <td class="mono">${s.loss.toFixed(4)}</td>
              <td class="mono">${s.nRatio.toFixed(3)}</td>
              <td class="mono ${s.savings>0?"delta-pos":""}">${(s.savings*100).toFixed(1)}%</td>
            </tr>`).join("")}
          </tbody>
        </table>
        <p class="muted">† Compute saved to match AdamW's predicted loss at this budget (same target L).</p>
        <div class="export-row">
          <button class="icon-btn" id="copyPy">Copy Python snippet</button>
        </div>
        <pre class="muted mono" id="pySnippet"></pre>
      </section>
    </div>
  `;const n=u("#presets");A.forEach(s=>{const d=document.createElement("button");d.className="chip",d.textContent=s.label,d.onclick=()=>X(s.id),n.appendChild(d)});const l=u("#optChips");N.forEach(s=>{const d=document.createElement("button");d.className=`chip ${e.selected.includes(s.id)?"active":""}`,d.textContent=s.label,d.onclick=()=>{e.selected.includes(s.id)?e.selected=e.selected.filter(p=>p!==s.id):e.selected=[...e.selected,s.id],e.selected.length||(e.selected=["adamw"]),m(),f()},l.appendChild(d)}),u("#flopsInput").onchange=s=>{e.flops=Number(s.target.value),m(),f()},u("#gpuSel").onchange=s=>{e.gpu=s.target.value;const d=$(e.gpu);e.mfu=d.defaultMfu,m(),f()},u("#countInput").onchange=s=>{e.count=Math.max(1,Number(s.target.value)),m(),f()},u("#hoursInput").onchange=s=>{const d=Number(s.target.value);e.flops=P($(e.gpu),e.count,d,e.mfu),m(),f()},u("#mfuInput").oninput=s=>{e.mfu=Number(s.target.value),m(),f()},u("#optSel").onchange=s=>{e.optimizer=s.target.value,m(),f()},u("#ratioMode").onchange=s=>{e.ratioMode=s.target.checked,m(),f()};const c=document.querySelector("#ratioInput");c&&(c.onchange=s=>{e.ratio=Number(s.target.value),m(),f()}),u("#copyPy").onclick=async()=>{const s=`from optiscale import allocate_for_budget, cost_report
alloc = allocate_for_budget(${e.flops.toExponential()}, optimizer="${e.optimizer}")
print(alloc)
print(cost_report(compute=${e.flops.toExponential()}, gpu_id="${e.gpu}", count=${e.count}))
`;u("#pySnippet").textContent=s,await navigator.clipboard.writeText(s)},J(a)}function J(o){const a=u("#isoCanvas"),t=a.getContext("2d"),r=T(e.flops,o),i=T(e.flops,M("adamw")),n=a.width,l=a.height,c=48;t.clearRect(0,0,n,l);const s=[...r.loss,...i.loss],d=Math.min(...s),p=Math.max(...s),h=Math.min(...r.N),b=Math.max(...r.N),v=g=>c+(Math.log(g)-Math.log(h))/(Math.log(b)-Math.log(h))*(n-2*c),C=g=>l-c-(g-d)/(p-d||1)*(l-2*c);t.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--line"),t.beginPath(),t.moveTo(c,c),t.lineTo(c,l-c),t.lineTo(n-c,l-c),t.stroke();const L=(g,F)=>{t.strokeStyle=F,t.lineWidth=2,t.beginPath(),g.N.forEach((H,E)=>{const O=v(H),D=C(g.loss[E]);E===0?t.moveTo(O,D):t.lineTo(O,D)}),t.stroke();const _=g.star;t.fillStyle=F,t.beginPath(),t.arc(v(_.N),C(_.loss),5,0,Math.PI*2),t.fill()};L(i,getComputedStyle(document.documentElement).getPropertyValue("--adamw").trim()||"#7aa2c8"),L(r,getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()||"#3dcf8e"),t.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted").trim(),t.font="12px IBM Plex Mono",t.fillText("N (log)",n/2-20,l-14),t.save(),t.translate(14,l/2),t.rotate(-Math.PI/2),t.fillText("Loss",0,0),t.restore(),t.fillText("AdamW",n-c-120,c+8),t.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--accent").trim(),t.fillText(o.label,n-c-120,c+24)}function X(o){const a=A.find(t=>t.id===o);if(a){if(a.kind==="flops")e.flops=a.value,e.ratioMode=!!a.ratio,a.ratio&&(e.ratio=a.ratio);else if(a.kind==="usd"){e.gpu=a.gpu,e.count=a.count;const t=$(e.gpu);e.mfu=t.defaultMfu;const r=a.value/(t.usdPerHour*e.count);e.flops=P(t,e.count,r,e.mfu)}else a.kind==="gpuHours"&&(e.gpu=a.gpu,e.count=a.count,e.mfu=$(e.gpu).defaultMfu,e.flops=P($(e.gpu),e.count,a.hours,e.mfu));m(),f()}}function Z(){u("#view").innerHTML=`
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
  `,u("#synthBtn").onclick=()=>{const o=["N,D,L,optimizer"];for(const a of["adamw","muon"]){const t=M(a),r=[1e19,1e20,1e21,1e22];for(const i of r){const n=y(i,t).N;for(let l=0;l<8;l++){const c=n/6*Math.pow(36,l/7),s=i/(6*c),d=c*t.rho_n,p=s*t.rho_d,h=1.69+406.4/d**.34+410.7/p**.28+(Math.random()-.5)*.02;o.push(`${c},${s},${h},${a}`)}}}u("#csvIn").value=o.join(`
`)},u("#fitBtn").onclick=()=>{var i;const o=u("#csvIn").value.trim();if(!o){u("#fitOut").textContent="Paste CSV or load synthetic demo first.";return}const a=o.split(/\r?\n/).slice(1).map(n=>n.split(",")).filter(n=>n.length>=4).map(n=>({N:Number(n[0]),D:Number(n[1]),L:Number(n[2]),optimizer:n[3].trim().toLowerCase()})),t={};for(const n of a)t[i=n.optimizer]??(t[i]={N:[],D:[],L:[]}),t[n.optimizer].N.push(n.N),t[n.optimizer].D.push(n.D),t[n.optimizer].L.push(n.L);const r=["Browser fit summary (full L-BFGS lives in Python `optiscale fit`):",""];for(const[n,l]of Object.entries(t)){const c=N.find(p=>p.id===n)??{label:n,rho_n:1,rho_d:1};let s=0;for(let p=0;p<l.N.length;p++){const h=l.N[p]*c.rho_n,b=l.D[p]*c.rho_d,v=1.69+406.4/h**.34+410.7/b**.28;s+=(v-l.L[p])**2}const d=Math.sqrt(s/l.N.length);r.push(`${c.label}: n=${l.N.length}  RMSE under prior ρ = ${d.toFixed(4)}  (ρN=${c.rho_n}, ρD=${c.rho_d})`)}r.push("","Run for full shared-exponent fit + bootstrap CI:"),r.push("  pip install -e ."),r.push("  optiscale fit --synthetic"),r.push("  optiscale fit --data runs.csv"),u("#fitOut").textContent=r.join(`
`)}}function Y(){u("#view").innerHTML=`
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
  `}function Q(){const o=k(e.flops,e.selected),a=$(e.gpu),t=I(e.flops,a,e.count,e.mfu),r=["# OptiScale Report","",`- Compute: ${U(e.flops)}`,`- GPU: ${a.name} × ${e.count} @ MFU ${e.mfu}`,`- Wall-clock: ${t.days.toFixed(2)} days`,`- Cost: $${t.cost.toFixed(0)}`,"","| Optimizer | N* | D* | Loss | N/AdamW | Compute saved |","|---|---:|---:|---:|---:|---:|",...o.map(i=>`| ${i.label} | ${x(i.N)} | ${x(i.D)} | ${i.loss.toFixed(4)} | ${i.nRatio.toFixed(3)} | ${(i.savings*100).toFixed(1)}% |`),""].join(`
`);R("optiscale-report.md",r)}function tt(){const a=["optimizer,N,D,loss,tokens_per_param,n_ratio,compute_saved",...k(e.flops,e.selected).map(t=>`${t.id},${t.N},${t.D},${t.loss},${t.tokensPerParam},${t.nRatio},${t.savings}`)];R("optiscale-compare.csv",a.join(`
`))}function R(o,a){const t=new Blob([a],{type:"text/plain"}),r=document.createElement("a");r.href=URL.createObjectURL(t),r.download=o,r.click(),URL.revokeObjectURL(r.href)}function f(){q(),e.tab==="practice"?K():e.tab==="research"?Z():Y()}f();
