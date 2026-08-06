(function(){const o=document.createElement("link").relList;if(o&&o.supports&&o.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))t(a);new MutationObserver(a=>{for(const r of a)if(r.type==="childList")for(const c of r.addedNodes)c.tagName==="LINK"&&c.rel==="modulepreload"&&t(c)}).observe(document,{childList:!0,subtree:!0});function s(a){const r={};return a.integrity&&(r.integrity=a.integrity),a.referrerPolicy&&(r.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?r.credentials="include":a.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function t(a){if(a.ep)return;a.ep=!0;const r=s(a);fetch(a.href,r)}})();const k={E:1.69,A:406.4,B:410.7,alpha:.34,beta:.28},R=[{id:"adamw",label:"AdamW",rho_n:1,rho_d:1},{id:"muon",label:"Muon",rho_n:1.35,rho_d:1.25},{id:"normuon",label:"NorMuon",rho_n:1.45,rho_d:1.3},{id:"aurora",label:"Aurora",rho_n:1.4,rho_d:1.28},{id:"soap",label:"SOAP",rho_n:1.2,rho_d:1.15},{id:"shampoo",label:"Shampoo",rho_n:1.18,rho_d:1.12},{id:"lion",label:"Lion",rho_n:1.05,rho_d:1.02}],W=[{id:"h100",name:"H100 SXM",peakTflops:989,usdPerHour:4,defaultMfu:.4,memoryGb:80},{id:"h100_pcie",name:"H100 PCIe",peakTflops:756,usdPerHour:3.5,defaultMfu:.38,memoryGb:80},{id:"a100",name:"A100 80GB",peakTflops:312,usdPerHour:2.2,defaultMfu:.45,memoryGb:80},{id:"a100_40",name:"A100 40GB",peakTflops:312,usdPerHour:1.8,defaultMfu:.42,memoryGb:40},{id:"l40s",name:"L40S",peakTflops:362,usdPerHour:1.6,defaultMfu:.35,memoryGb:48},{id:"rtx4090",name:"RTX 4090",peakTflops:330,usdPerHour:.8,defaultMfu:.3,memoryGb:24},{id:"v100",name:"V100",peakTflops:125,usdPerHour:.9,defaultMfu:.4,memoryGb:32},{id:"tpu_v5e",name:"TPU v5e",peakTflops:197,usdPerHour:1.2,defaultMfu:.5,memoryGb:16}],V=[{id:"1m",label:"$1M pretrain",kind:"usd",value:1e6,gpu:"h100",count:64},{id:"8xh100-30d",label:"8×H100 × 30 days",kind:"gpuHours",hours:24*30,gpu:"h100",count:8},{id:"70b",label:"Chinchilla 70B-ish",kind:"flops",value:6*7e10*14e11},{id:"1e24",label:"10²⁴ FLOPs",kind:"flops",value:1e24},{id:"overtrain",label:"Overtrain @ 100 tok/param",kind:"flops",value:1e23,ratio:100}];function O(e){const o=R.find(s=>s.id===e);if(!o)throw new Error(`Unknown optimizer ${e}`);return o}function E(e){const o=W.find(s=>s.id===e);if(!o)throw new Error(`Unknown GPU ${e}`);return o}function Q(e,o,s){return{...e,rho_n:o??e.rho_n,rho_d:s??e.rho_d}}function T(e){return e>=1e12?`${(e/1e12).toFixed(2)}T`:e>=1e9?`${(e/1e9).toFixed(2)}B`:e>=1e6?`${(e/1e6).toFixed(2)}M`:e>=1e3?`${(e/1e3).toFixed(2)}K`:e.toFixed(0)}function q(e){return e>=1e24?`${(e/1e24).toFixed(3)}×10²⁴`:e>=1e21?`${(e/1e21).toFixed(2)} ZFLOP`:e>=1e18?`${(e/1e18).toFixed(2)} EFLOP`:e.toExponential(2)}function A(e,o,s,t=k){const a=e*s.rho_n,r=o*s.rho_d;return t.E+t.A/a**t.alpha+t.B/r**t.beta}function F(e,o,s=k,t){const r=s.alpha,c=s.beta,p=(s.A*r/(s.B*c))**(1/(r+c)),u=(o.rho_d**c/o.rho_n**r)**(1/(r+c)),h=p*u*(e/1/6)**(c/(r+c)),i=e/1/(6*h);return{N:h,D:i,loss:A(h,i,o,s),tokensPerParam:i/h}}function tt(e,o,s,t=k){const a=Math.sqrt(e/(6*o)),r=o*a;return{N:a,D:r,loss:A(a,r,s,t),tokensPerParam:o}}function G(e,o,s=64,t=30,a=k){const r=F(e,o,a).N,c=[],p=[],u=[];for(let i=0;i<s;i++){const d=i/(s-1),f=r/t*Math.pow(t*t,d),l=e/(6*f);c.push(f),p.push(l),u.push(A(f,l,o,a))}let h=0;for(let i=1;i<u.length;i++)u[i]<u[h]&&(h=i);return{N:c,D:p,loss:u,iMin:h,star:{N:c[h],D:p[h],loss:u[h]}}}function j(e,o,s,t){const a=t??e.defaultMfu;return e.peakTflops*1e12*a*o*s*3600}function J(e,o,s,t){const a=t??o.defaultMfu,r=o.peakTflops*1e12*a*s,c=e/r/3600;return{hours:c,days:c/24,cost:c*o.usdPerHour*s,mfu:a}}function K(e,o,s=k){let t=Math.log(1e18),a=Math.log(1e28);for(let r=0;r<60;r++){const c=.5*(t+a),p=Math.exp(c);F(p,o,s).loss>e?t=c:a=c}return Math.exp(.5*(t+a))}function z(e,o=R.map(t=>t.id),s){const t=O("adamw"),a=F(e,t);return o.map(r=>{const c=O(r),p=F(e,c),u=K(a.loss,c);return{...c,...p,nRatio:p.N/a.N,savings:1-u/e,matchCompute:u}})}function et(e,o=[.8,2],s=[.8,2],t=24,a=k){const r=F(e,O("adamw"),a),c=[],p=[];for(let i=0;i<t;i++){const d=i/(t-1);c.push(o[0]+d*(o[1]-o[0])),p.push(s[0]+d*(s[1]-s[0]))}const u=[],h=[];for(let i=0;i<t;i++){const d=[],f=[];for(let l=0;l<t;l++){const v={rho_n:c[l],rho_d:p[i]},w=F(e,v,a);d.push((w.N-r.N)/r.N),f.push(w.loss)}u.push(d),h.push(f)}return{rhoN:c,rhoD:p,deltaN:u,loss:h}}function ot(e,o,s=40,t=16,a=k){const r=F(e,o,a),c=[];for(let h=0;h<s;h++){const i=h/(s-1);c.push(r.N/t*Math.pow(t*t,i))}const p=c.map(h=>e/(6*h)),u=[];for(let h=0;h<s;h++){const i=[];for(let d=0;d<s;d++)i.push(A(c[d],p[h],o,a));u.push(i)}return{N:c,D:p,loss:u,star:r}}function st(e,o=2,s=4,t=1.5){return e*o*s*t/1e9}function nt(e,o,s){const t=st(e),a=o.memoryGb*s;return{needGb:t,haveGb:a,fits:t<=a*.9,util:t/a}}function at(e,o=.01){const s=Math.abs(e);return s<=o?.5*e*e:o*(s-.5*o)}function H(e,o,s){return Math.min(s,Math.max(o,e))}function X(e,o,s){const t=o.filter(c=>c!==s),a={E:Math.exp(e[0]),A:Math.exp(e[1]),B:Math.exp(e[2]),alpha:H(e[3],.05,1.5),beta:H(e[4],.05,1.5)},r={[s]:{id:s,label:s,rho_n:1,rho_d:1}};return t.forEach((c,p)=>{r[c]={id:c,label:c,rho_n:Math.exp(e[5+2*p]),rho_d:Math.exp(e[5+2*p+1])}}),{params:a,rhos:r}}function B(e,o,s,t){const{params:a,rhos:r}=X(e,s,t);let c=0,p=0;for(const u of s){const h=o[u],i=r[u];for(let d=0;d<h.L.length;d++){const f=A(h.N[d],h.D[d],i,a);c+=at(f-h.L[d]),p+=1}}return c/Math.max(p,1)}function U(e,o,s,t,a=1e-5){const r=new Array(e.length).fill(0),c=B(e,o,s,t);for(let p=0;p<e.length;p++){const u=e.slice();u[p]+=a,r[p]=(B(u,o,s,t)-c)/a}return r}function it(e,o,s,t,a=80){const r=e.length;let c=e.slice(),p=Array.from({length:r},(h,i)=>Array.from({length:r},(d,f)=>i===f?1:0)),u=U(c,o,s,t);for(let h=0;h<a;h++){const i=new Array(r).fill(0);for(let b=0;b<r;b++){let g=0;for(let $=0;$<r;$++)g+=p[b][$]*u[$];i[b]=-g}let d=1;const f=B(c,o,s,t);let l=c.map((b,g)=>b+d*i[g]),v=B(l,o,s,t),w=0;for(;v>f+1e-4*d*i.reduce((b,g,$)=>b+g*u[$],0)&&w<20;)d*=.5,l=c.map((b,g)=>b+d*i[g]),v=B(l,o,s,t),w+=1;const M=l.map((b,g)=>b-c[g]),y=U(l,o,s,t),x=y.map((b,g)=>b-u[g]),C=x.reduce((b,g,$)=>b+g*M[$],0);if(Math.hypot(...y)<1e-7)return{x:l,success:!0};if(C>1e-12){const b=1/C,g=new Array(r).fill(0);for(let N=0;N<r;N++){let S=0;for(let I=0;I<r;I++)S+=p[N][I]*x[I];g[N]=S}const $=x.reduce((N,S,I)=>N+S*g[I],0),P=Array.from({length:r},()=>new Array(r).fill(0));for(let N=0;N<r;N++)for(let S=0;S<r;S++)P[N][S]=p[N][S]-b*(M[N]*g[S]+g[N]*M[S])+b*b*$*M[N]*M[S]+b*M[N]*M[S];p=P}if(c=l,u=y,Math.abs(f-v)<1e-12)break}return{x:c,success:!0}}function rt(e,o="adamw"){const s=Object.keys(e);if(!s.includes(o))throw new Error(`reference optimizer ${o} missing from runs`);const t=s.filter(i=>i!==o),a=new Array(5+2*t.length).fill(0);a[0]=Math.log(k.E),a[1]=Math.log(k.A),a[2]=Math.log(k.B),a[3]=k.alpha,a[4]=k.beta,t.forEach((i,d)=>{a[5+2*d]=0,a[5+2*d+1]=0});const{x:r,success:c}=it(a,e,s,o),{params:p,rhos:u}=X(r,s,o),h={};for(const i of s){const d=e[i];let f=0;for(let l=0;l<d.L.length;l++){const v=A(d.N[l],d.D[l],u[i],p);f+=(v-d.L[l])**2}h[i]=Math.sqrt(f/d.L.length)}return{params:p,rhos:Object.fromEntries(Object.entries(u).map(([i,d])=>[i,{rho_n:d.rho_n,rho_d:d.rho_d,label:d.label}])),per_optimizer_rmse:h,mean_rmse:Object.values(h).reduce((i,d)=>i+d,0)/Math.max(Object.values(h).length,1),success:c,method:"shared_exponents_lbfgs_js"}}const Z="optiscale-scenario-v2";function lt(){const e=O("muon");return{theme:"dark",tab:"practice",flops:1e24,optimizer:"muon",gpu:"h100",count:8,mfu:.4,ratioMode:!1,ratio:100,selected:["adamw","muon","normuon","soap"],rhoN:e.rho_n,rhoD:e.rho_d,targetLoss:null,targetLossMode:!1}}function ct(){const e=new URLSearchParams(location.search),o=lt();e.get("C")&&(o.flops=Number(e.get("C"))),e.get("opt")&&(o.optimizer=e.get("opt")),e.get("gpu")&&(o.gpu=e.get("gpu")),e.get("n")&&(o.count=Number(e.get("n"))),e.get("mfu")&&(o.mfu=Number(e.get("mfu")));try{const s=localStorage.getItem(Z);s&&Object.assign(o,JSON.parse(s))}catch{}return e.get("C")&&(o.flops=Number(e.get("C"))),e.get("opt")&&(o.optimizer=e.get("opt")),o}let n=ct(),D={};function dt(){const e=O(n.optimizer);return Q(e,n.rhoN,n.rhoD)}function L(){localStorage.setItem(Z,JSON.stringify(n));const e=new URLSearchParams({C:String(n.flops),opt:n.optimizer,gpu:n.gpu,n:String(n.count),mfu:String(n.mfu)});history.replaceState(null,"",`?${e.toString()}`)}function m(e){const o=document.querySelector(e);if(!o)throw new Error(`Missing ${e}`);return o}function ut(){document.documentElement.dataset.theme=n.theme,document.querySelector("#app").innerHTML=`
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
        <button class="tab ${n.tab==="practice"?"active":""}" data-tab="practice">Practitioner</button>
        <button class="tab ${n.tab==="research"?"active":""}" data-tab="research">Research fit</button>
        <button class="tab ${n.tab==="formula"?"active":""}" data-tab="formula">Formulas</button>
      </div>
      <button class="icon-btn" id="themeBtn">${n.theme==="dark"?"Light":"Dark"}</button>
      <button class="icon-btn" id="shareBtn">Copy share link</button>
      <button class="icon-btn" id="exportMd">Export Markdown</button>
      <button class="icon-btn" id="exportCsv">Export CSV</button>
    </div>
    <div id="view"></div>
    <footer class="footer">
      Part of the <strong>Spectral Training Stack</strong> with
      <a href="https://github.com/Sqlace/spectoptim">SpectOptim</a> and
      <a href="https://github.com/Sqlace/ortholab">OrthoLab</a>.
      ρ priors are planning defaults — fit your own runs for production.
      Inspired by Hoffmann et al. (2022) and Volkova et al. (2026).
      <code class="mono">pip install optiscale</code>
    </footer>
  `,document.querySelectorAll(".tab").forEach(e=>{e.addEventListener("click",()=>{n.tab=e.dataset.tab,L(),_()})}),m("#themeBtn").onclick=()=>{n.theme=n.theme==="dark"?"light":"dark",L(),_()},m("#shareBtn").onclick=async()=>{L(),await navigator.clipboard.writeText(location.href),m("#shareBtn").textContent="Copied",setTimeout(()=>m("#shareBtn").textContent="Copy share link",1200)},m("#exportMd").onclick=yt,m("#exportCsv").onclick=$t}function pt(){const e=E(n.gpu),o=dt();let s=n.flops,t,a="";n.targetLossMode&&n.targetLoss!=null&&n.targetLoss>1.69?(s=K(n.targetLoss,o),t=F(s,o),a=`Inverse solve: C* ≈ ${q(s)} to reach L=${n.targetLoss}`):n.ratioMode?t=tt(n.flops,n.ratio,o):t=F(n.flops,o);const r=J(s,e,n.count,n.mfu),c=nt(t.N,e,n.count),p=z(s,n.selected);m("#view").innerHTML=`
    <div class="grid">
      <aside class="panel">
        <h2>Budget</h2>
        <div class="row" id="presets"></div>
        <label>FLOP budget C
          <input id="flopsInput" type="text" value="${s.toExponential(3)}" ${n.targetLossMode?"disabled":""} />
        </label>
        <label>Or GPU-hours → FLOPs
          <div class="row">
            <select id="gpuSel">${W.map(l=>`<option value="${l.id}" ${l.id===n.gpu?"selected":""}>${l.name}</option>`).join("")}</select>
          </div>
        </label>
        <label>GPU count
          <input id="countInput" type="number" min="1" value="${n.count}" />
        </label>
        <label>Hours
          <input id="hoursInput" type="number" min="0.1" step="0.1" value="${r.hours.toFixed(1)}" ${n.targetLossMode?"disabled":""} />
        </label>
        <label>MFU ${(n.mfu*100).toFixed(0)}%
          <input id="mfuInput" type="range" min="0.15" max="0.7" step="0.01" value="${n.mfu}" />
        </label>
        <label>Optimizer
          <select id="optSel">${R.map(l=>`<option value="${l.id}" ${l.id===n.optimizer?"selected":""}>${l.label}</option>`).join("")}</select>
        </label>
        <label>Custom ρ<sub>N</sub>
          <input id="rhoNInput" type="number" min="0.1" max="5" step="0.01" value="${n.rhoN}" />
        </label>
        <label>Custom ρ<sub>D</sub>
          <input id="rhoDInput" type="number" min="0.1" max="5" step="0.01" value="${n.rhoD}" />
        </label>
        <label><input id="ratioMode" type="checkbox" ${n.ratioMode?"checked":""} ${n.targetLossMode?"disabled":""}/> Fix tokens/param ratio</label>
        <label class="${n.ratioMode&&!n.targetLossMode?"":"hidden"}">Tokens / param
          <input id="ratioInput" type="number" min="1" value="${n.ratio}" />
        </label>
        <label><input id="targetLossMode" type="checkbox" ${n.targetLossMode?"checked":""}/> Target-loss inverse</label>
        <label class="${n.targetLossMode?"":"hidden"}">Target loss L
          <input id="targetLossInput" type="number" min="1.7" max="10" step="0.01" value="${n.targetLoss??2.5}" />
        </label>
        <p class="muted">ρ fields override catalog priors for the selected optimizer. Use Research → Apply ρ to seed from a fit.</p>
      </aside>
      <section class="panel">
        <h2>Allocation — ${o.label} (ρN=${o.rho_n}, ρD=${o.rho_d})</h2>
        ${a?`<p class="muted">${a}</p>`:""}
        <div class="cards">
          <div class="card"><div class="k">N*</div><div class="v accent">${T(t.N)}</div></div>
          <div class="card"><div class="k">D*</div><div class="v">${T(t.D)}</div></div>
          <div class="card"><div class="k">tok / param</div><div class="v">${t.tokensPerParam.toFixed(1)}</div></div>
          <div class="card"><div class="k">Pred. loss</div><div class="v">${t.loss.toFixed(4)}</div></div>
          <div class="card"><div class="k">Wall-clock</div><div class="v">${r.days.toFixed(2)} d</div></div>
          <div class="card"><div class="k">Est. $</div><div class="v">$${r.cost.toLocaleString(void 0,{maximumFractionDigits:0})}</div></div>
          <div class="card"><div class="k">Mem fit</div><div class="v ${c.fits?"delta-pos":"delta-neg"}">${c.fits?"OK":"TIGHT"} · ${c.needGb.toFixed(0)}/${c.haveGb.toFixed(0)} GB</div></div>
        </div>
        <h2>IsoFLOP curve</h2>
        <canvas id="isoCanvas" width="900" height="360"></canvas>
        <h2>N–D loss landscape</h2>
        <p class="muted">L(N,D) plane; star marks compute-optimal (N*,D*) on the C=6ND ridge.</p>
        <canvas id="landCanvas" width="480" height="420"></canvas>
        <h2>ρ sensitivity — ΔN* vs AdamW</h2>
        <p class="muted">Heatmap of relative ΔN* = (N*(ρ) − N*_AdamW) / N*_AdamW at the current FLOP budget.</p>
        <canvas id="heatCanvas" width="480" height="420"></canvas>
        <h2>Multi-optimizer compare</h2>
        <div class="row" id="optChips"></div>
        <table>
          <thead>
            <tr><th>Opt</th><th>N*</th><th>D*</th><th>Loss</th><th>N / AdamW</th><th>Compute saved†</th></tr>
          </thead>
          <tbody>
            ${p.map(l=>`<tr>
              <td>${l.label}</td>
              <td class="mono">${T(l.N)}</td>
              <td class="mono">${T(l.D)}</td>
              <td class="mono">${l.loss.toFixed(4)}</td>
              <td class="mono">${l.nRatio.toFixed(3)}</td>
              <td class="mono ${l.savings>0?"delta-pos":""}">${(l.savings*100).toFixed(1)}%</td>
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
  `;const u=m("#presets");V.forEach(l=>{const v=document.createElement("button");v.className="chip",v.textContent=l.label,v.onclick=()=>bt(l.id),u.appendChild(v)});const h=m("#optChips");R.forEach(l=>{const v=document.createElement("button");v.className=`chip ${n.selected.includes(l.id)?"active":""}`,v.textContent=l.label,v.onclick=()=>{n.selected.includes(l.id)?n.selected=n.selected.filter(w=>w!==l.id):n.selected=[...n.selected,l.id],n.selected.length||(n.selected=["adamw"]),L(),_()},h.appendChild(v)}),m("#flopsInput").onchange=l=>{n.flops=Number(l.target.value),L(),_()},m("#gpuSel").onchange=l=>{n.gpu=l.target.value;const v=E(n.gpu);n.mfu=v.defaultMfu,L(),_()},m("#countInput").onchange=l=>{n.count=Math.max(1,Number(l.target.value)),L(),_()},m("#hoursInput").onchange=l=>{const v=Number(l.target.value);n.flops=j(E(n.gpu),n.count,v,n.mfu),L(),_()},m("#mfuInput").oninput=l=>{n.mfu=Number(l.target.value),L(),_()},m("#optSel").onchange=l=>{n.optimizer=l.target.value;const v=O(n.optimizer);n.rhoN=v.rho_n,n.rhoD=v.rho_d,L(),_()};const i=l=>{const w=l.target.id;n.rhoN=Number(m("#rhoNInput").value),n.rhoD=Number(m("#rhoDInput").value),L(),_();const M=document.getElementById(w);if(M){M.focus();const y=M.value.length;M.setSelectionRange(y,y)}};m("#rhoNInput").oninput=i,m("#rhoDInput").oninput=i,m("#ratioMode").onchange=l=>{n.ratioMode=l.target.checked,n.ratioMode&&(n.targetLossMode=!1),L(),_()};const d=document.querySelector("#ratioInput");d&&(d.onchange=l=>{n.ratio=Number(l.target.value),L(),_()}),m("#targetLossMode").onchange=l=>{n.targetLossMode=l.target.checked,n.targetLossMode&&(n.ratioMode=!1,n.targetLoss==null&&(n.targetLoss=2.5)),L(),_()};const f=document.querySelector("#targetLossInput");f&&(f.onchange=l=>{n.targetLoss=Number(l.target.value),L(),_()}),m("#copyPy").onclick=async()=>{const l=`from optiscale import allocate_for_budget, cost_report
alloc = allocate_for_budget(${n.flops.toExponential()}, optimizer="${n.optimizer}", rho_n=${n.rhoN}, rho_d=${n.rhoD})
print(alloc)
print(cost_report(compute=${n.flops.toExponential()}, gpu_id="${n.gpu}", count=${n.count}))
`;m("#pySnippet").textContent=l,await navigator.clipboard.writeText(l)},ht(o,s),ft(o,s),mt(s)}function ht(e,o=n.flops){const s=m("#isoCanvas"),t=s.getContext("2d"),a=G(o,e),r=G(o,O("adamw")),c=s.width,p=s.height,u=48;t.clearRect(0,0,c,p);const h=[...a.loss,...r.loss],i=Math.min(...h),d=Math.max(...h),f=Math.min(...a.N),l=Math.max(...a.N),v=y=>u+(Math.log(y)-Math.log(f))/(Math.log(l)-Math.log(f))*(c-2*u),w=y=>p-u-(y-i)/(d-i||1)*(p-2*u);t.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--line"),t.beginPath(),t.moveTo(u,u),t.lineTo(u,p-u),t.lineTo(c-u,p-u),t.stroke();const M=(y,x)=>{t.strokeStyle=x,t.lineWidth=2,t.beginPath(),y.N.forEach((b,g)=>{const $=v(b),P=w(y.loss[g]);g===0?t.moveTo($,P):t.lineTo($,P)}),t.stroke();const C=y.star;t.fillStyle=x,t.beginPath(),t.arc(v(C.N),w(C.loss),5,0,Math.PI*2),t.fill()};M(r,getComputedStyle(document.documentElement).getPropertyValue("--adamw").trim()||"#7aa2c8"),M(a,getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()||"#3dcf8e"),t.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted").trim(),t.font="12px IBM Plex Mono",t.fillText("N (log)",c/2-20,p-14),t.save(),t.translate(14,p/2),t.rotate(-Math.PI/2),t.fillText("Loss",0,0),t.restore(),t.fillText("AdamW",c-u-120,u+8),t.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--accent").trim(),t.fillText(e.label,c-u-120,u+24)}function ft(e,o=n.flops){const s=m("#landCanvas");if(!s)return;const t=s.getContext("2d"),a=ot(o,e,36,12),r=56,c=48,p=24,u=24,h=s.width,i=s.height,d=a.N.length,f=(h-r-u)/d,l=(i-p-c)/d;t.clearRect(0,0,h,i);let v=1/0,w=-1/0;for(const b of a.loss)for(const g of b)g<v&&(v=g),g>w&&(w=g);const M=w-v||1;for(let b=0;b<d;b++)for(let g=0;g<d;g++){const $=(a.loss[b][g]-v)/M,P=Math.round(30+$*200),N=Math.round(180-$*120),S=Math.round(120+$*40);t.fillStyle=`rgb(${P},${N},${S})`,t.fillRect(r+g*f,p+b*l,Math.ceil(f),Math.ceil(l))}let y=0,x=0,C=1/0;for(let b=0;b<d;b++)for(let g=0;g<d;g++){const $=Math.abs(Math.log(a.N[b])-Math.log(a.star.N)),P=Math.abs(Math.log(a.D[g])-Math.log(a.star.D));$+P<C&&(C=$+P,y=b,x=g)}t.strokeStyle="#f3ebe1",t.lineWidth=2,t.beginPath(),t.arc(r+(y+.5)*f,p+(x+.5)*l,6,0,Math.PI*2),t.stroke(),t.fillStyle="#9a8f82",t.font="12px IBM Plex Mono",t.fillText("log N →",h/2-20,i-14),t.fillText("log D ↓",8,i/2),t.fillText(`★ ${e.label}`,r,16)}function mt(e=n.flops){const o=m("#heatCanvas"),s=o.getContext("2d"),t=et(e,[.8,2],[.8,2],28),a=56,r=48,c=24,p=24,u=o.width,h=o.height,i=(u-a-p)/t.rhoN.length,d=(h-c-r)/t.rhoD.length;s.clearRect(0,0,u,h);let f=1/0,l=-1/0;for(const y of t.deltaN)for(const x of y)x<f&&(f=x),x>l&&(l=x);const v=l-f||1;for(let y=0;y<t.rhoD.length;y++)for(let x=0;x<t.rhoN.length;x++){const b=(t.deltaN[y][x]-f)/v,g=Math.round(40+b*180),$=Math.round(80+(1-Math.abs(b-.5)*2)*120),P=Math.round(160-b*120);s.fillStyle=`rgb(${g},${$},${P})`;const N=h-r-(y+1)*d;s.fillRect(a+x*i,N,i+.5,d+.5)}const w=(n.rhoN-.8)/(2-.8),M=(n.rhoD-.8)/(2-.8);s.strokeStyle="#fff",s.lineWidth=2,s.beginPath(),s.arc(a+w*(u-a-p),h-r-M*(h-c-r),6,0,Math.PI*2),s.stroke(),s.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted").trim()||"#888",s.font="11px IBM Plex Mono",s.fillText("ρ_N →",a+(u-a-p)/2-16,h-14),s.save(),s.translate(14,h/2),s.rotate(-Math.PI/2),s.fillText("ρ_D →",0,0),s.restore(),s.fillText(`ΔN* [${(f*100).toFixed(0)}%, ${(l*100).toFixed(0)}%]`,a,16)}function bt(e){const o=V.find(s=>s.id===e);if(o){if(n.targetLossMode=!1,o.kind==="flops")n.flops=o.value,n.ratioMode=!!o.ratio,o.ratio&&(n.ratio=o.ratio);else if(o.kind==="usd"){n.gpu=o.gpu,n.count=o.count;const s=E(n.gpu);n.mfu=s.defaultMfu;const t=o.value/(s.usdPerHour*n.count);n.flops=j(s,n.count,t,n.mfu)}else o.kind==="gpuHours"&&(n.gpu=o.gpu,n.count=o.count,n.mfu=E(n.gpu).defaultMfu,n.flops=j(E(n.gpu),n.count,o.hours,n.mfu));L(),_()}}function gt(){const e=Object.keys(D);m("#view").innerHTML=`
    <div class="panel">
      <h2>Shared-exponent vs separate fits</h2>
      <p class="muted">
        Paste CSV with columns <code class="mono">N,D,L,optimizer</code> or run a synthetic IsoFLOP demo.
        Separate per-optimizer Chinchilla fits are often ill-conditioned; the shared α,β + ρ model is more stable.
      </p>
      <textarea id="csvIn" rows="8" placeholder="N,D,L,optimizer&#10;1e8,2e9,3.2,adamw&#10;..."></textarea>
      <div class="row" style="margin-top:0.75rem">
        <button class="icon-btn" id="synthBtn">Load synthetic demo</button>
        <button class="icon-btn" id="fitBtn">Run browser L-BFGS fit</button>
        <button class="icon-btn" id="dlFitBtn" ${e.length?"":"disabled"}>Download fit.json</button>
        <button class="icon-btn" id="applyRhoBtn" ${e.length?"":"disabled"}>Apply ρ to Practitioner</button>
        <select id="applyOptSel" ${e.length?"":"disabled"}>
          ${e.length?e.map(o=>`<option value="${o}">${D[o].label} (ρN=${D[o].rho_n.toFixed(2)})</option>`).join(""):'<option value="">Run fit first</option>'}
        </select>
      </div>
      <div id="fitOut" class="muted">Results appear here.</div>
    </div>
  `,m("#synthBtn").onclick=()=>{const o=["N,D,L,optimizer"];for(const s of["adamw","muon"]){const t=O(s),a=[1e19,1e20,1e21,1e22];for(const r of a){const c=F(r,t).N;for(let p=0;p<8;p++){const u=c/6*Math.pow(36,p/7),h=r/(6*u),i=u*t.rho_n,d=h*t.rho_d,f=1.69+406.4/i**.34+410.7/d**.28+(Math.random()-.5)*.02;o.push(`${u},${h},${f},${s}`)}}}m("#csvIn").value=o.join(`
`)},m("#fitBtn").onclick=()=>{var h;const o=m("#csvIn").value.trim();if(!o){m("#fitOut").textContent="Paste CSV or load synthetic demo first.";return}const s=o.split(/\r?\n/).slice(1).map(i=>i.split(",")).filter(i=>i.length>=4).map(i=>({N:Number(i[0]),D:Number(i[1]),L:Number(i[2]),optimizer:i[3].trim().toLowerCase()})),t={};for(const i of s)t[h=i.optimizer]??(t[h]={N:[],D:[],L:[]}),t[i.optimizer].N.push(i.N),t[i.optimizer].D.push(i.D),t[i.optimizer].L.push(i.L);D={};let a;try{a=rt(t)}catch(i){m("#fitOut").textContent=`Fit failed: ${i.message}`;return}const r=[`Browser L-BFGS shared-exponent fit (success=${a.success})`,`mean RMSE=${a.mean_rmse.toFixed(4)}`,`params: E=${a.params.E.toFixed(3)} A=${a.params.A.toFixed(1)} B=${a.params.B.toFixed(1)} α=${a.params.alpha.toFixed(3)} β=${a.params.beta.toFixed(3)}`,""];for(const[i,d]of Object.entries(a.rhos)){const f=a.per_optimizer_rmse[i]??NaN;r.push(`${d.label}: ρN=${d.rho_n.toFixed(3)} ρD=${d.rho_d.toFixed(3)}  RMSE=${f.toFixed(4)}`),D[i]={rho_n:d.rho_n,rho_d:d.rho_d,label:d.label}}r.push("","Download fit.json or Apply ρ to Practitioner."),window.__lastFitJson=a,m("#fitOut").textContent=r.join(`
`);const c=m("#applyRhoBtn"),p=m("#applyOptSel");c.disabled=!1,p.disabled=!1,p.innerHTML=Object.keys(D).map(i=>`<option value="${i}">${D[i].label} (ρN=${D[i].rho_n.toFixed(2)})</option>`).join("");const u=m("#dlFitBtn");u.disabled=!1,u.onclick=()=>{const i=window.__lastFitJson;if(!i)return;const d=new Blob([JSON.stringify(i,null,2)],{type:"application/json"}),f=document.createElement("a");f.href=URL.createObjectURL(d),f.download="fit.json",f.click()}},m("#applyRhoBtn").onclick=()=>{const o=m("#applyOptSel").value,s=D[o];s&&(R.some(t=>t.id===o)&&(n.optimizer=o),n.rhoN=s.rho_n,n.rhoD=s.rho_d,n.tab="practice",L(),_())}}function vt(){m("#view").innerHTML=`
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
optiscale allocate --fixed-n 7e9 --flops 1e23
optiscale allocate --ratio 100 --flops 1e23
optiscale allocate --target-loss 2.5 --optimizer muon
optiscale fit --synthetic --bootstrap 200 --out fit.json
optiscale compare --budget-usd 1000000 --gpu h100 --count 64 --fit fit.json --md report.md
optiscale report --flops 1e24 --md report.md</pre>
    </div>
  `}function yt(){const e=z(n.flops,n.selected),o=E(n.gpu),s=J(n.flops,o,n.count,n.mfu),t=["# OptiScale Report","",`- Compute: ${q(n.flops)}`,`- GPU: ${o.name} × ${n.count} @ MFU ${n.mfu}`,`- Wall-clock: ${s.days.toFixed(2)} days`,`- Cost: $${s.cost.toFixed(0)}`,`- Practitioner ρ: ρN=${n.rhoN}, ρD=${n.rhoD}`,"","| Optimizer | N* | D* | Loss | N/AdamW | Compute saved |","|---|---:|---:|---:|---:|---:|",...e.map(a=>`| ${a.label} | ${T(a.N)} | ${T(a.D)} | ${a.loss.toFixed(4)} | ${a.nRatio.toFixed(3)} | ${(a.savings*100).toFixed(1)}% |`),""].join(`
`);Y("optiscale-report.md",t)}function $t(){const o=["optimizer,N,D,loss,tokens_per_param,n_ratio,compute_saved",...z(n.flops,n.selected).map(s=>`${s.id},${s.N},${s.D},${s.loss},${s.tokensPerParam},${s.nRatio},${s.savings}`)];Y("optiscale-compare.csv",o.join(`
`))}function Y(e,o){const s=new Blob([o],{type:"text/plain"}),t=document.createElement("a");t.href=URL.createObjectURL(s),t.download=e,t.click(),URL.revokeObjectURL(t.href)}function _(){ut(),n.tab==="practice"?pt():n.tab==="research"?gt():vt()}_();
