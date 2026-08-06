(function(){const o=document.createElement("link").relList;if(o&&o.supports&&o.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))a(n);new MutationObserver(n=>{for(const l of n)if(l.type==="childList")for(const c of l.addedNodes)c.tagName==="LINK"&&c.rel==="modulepreload"&&a(c)}).observe(document,{childList:!0,subtree:!0});function s(n){const l={};return n.integrity&&(l.integrity=n.integrity),n.referrerPolicy&&(l.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?l.credentials="include":n.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function a(n){if(n.ep)return;n.ep=!0;const l=s(n);fetch(n.href,l)}})();const P={E:1.69,A:406.4,B:410.7,alpha:.34,beta:.28},T=[{id:"adamw",label:"AdamW",rho_n:1,rho_d:1},{id:"muon",label:"Muon",rho_n:1.35,rho_d:1.25},{id:"normuon",label:"NorMuon",rho_n:1.45,rho_d:1.3},{id:"aurora",label:"Aurora",rho_n:1.4,rho_d:1.28},{id:"soap",label:"SOAP",rho_n:1.2,rho_d:1.15},{id:"shampoo",label:"Shampoo",rho_n:1.18,rho_d:1.12},{id:"lion",label:"Lion",rho_n:1.05,rho_d:1.02}],W=[{id:"h100",name:"H100 SXM",peakTflops:989,usdPerHour:4,defaultMfu:.4,memoryGb:80},{id:"h100_pcie",name:"H100 PCIe",peakTflops:756,usdPerHour:3.5,defaultMfu:.38,memoryGb:80},{id:"a100",name:"A100 80GB",peakTflops:312,usdPerHour:2.2,defaultMfu:.45,memoryGb:80},{id:"a100_40",name:"A100 40GB",peakTflops:312,usdPerHour:1.8,defaultMfu:.42,memoryGb:40},{id:"l40s",name:"L40S",peakTflops:362,usdPerHour:1.6,defaultMfu:.35,memoryGb:48},{id:"rtx4090",name:"RTX 4090",peakTflops:330,usdPerHour:.8,defaultMfu:.3,memoryGb:24},{id:"v100",name:"V100",peakTflops:125,usdPerHour:.9,defaultMfu:.4,memoryGb:32},{id:"tpu_v5e",name:"TPU v5e",peakTflops:197,usdPerHour:1.2,defaultMfu:.5,memoryGb:16}],V=[{id:"1m",label:"$1M pretrain",kind:"usd",value:1e6,gpu:"h100",count:64},{id:"8xh100-30d",label:"8×H100 × 30 days",kind:"gpuHours",hours:24*30,gpu:"h100",count:8},{id:"70b",label:"Chinchilla 70B-ish",kind:"flops",value:6*7e10*14e11},{id:"1e24",label:"10²⁴ FLOPs",kind:"flops",value:1e24},{id:"overtrain",label:"Overtrain @ 100 tok/param",kind:"flops",value:1e23,ratio:100}];function C(e){const o=T.find(s=>s.id===e);if(!o)throw new Error(`Unknown optimizer ${e}`);return o}function D(e){const o=W.find(s=>s.id===e);if(!o)throw new Error(`Unknown GPU ${e}`);return o}function Q(e,o,s){return{...e,rho_n:o??e.rho_n,rho_d:s??e.rho_d}}function I(e){return e>=1e12?`${(e/1e12).toFixed(2)}T`:e>=1e9?`${(e/1e9).toFixed(2)}B`:e>=1e6?`${(e/1e6).toFixed(2)}M`:e>=1e3?`${(e/1e3).toFixed(2)}K`:e.toFixed(0)}function q(e){return e>=1e24?`${(e/1e24).toFixed(3)}×10²⁴`:e>=1e21?`${(e/1e21).toFixed(2)} ZFLOP`:e>=1e18?`${(e/1e18).toFixed(2)} EFLOP`:e.toExponential(2)}function R(e,o,s,a=P){const n=e*s.rho_n,l=o*s.rho_d;return a.E+a.A/n**a.alpha+a.B/l**a.beta}function F(e,o,s=P,a){const l=s.alpha,c=s.beta,u=(s.A*l/(s.B*c))**(1/(l+c)),d=(o.rho_d**c/o.rho_n**l)**(1/(l+c)),h=u*d*(e/1/6)**(c/(l+c)),i=e/1/(6*h);return{N:h,D:i,loss:R(h,i,o,s),tokensPerParam:i/h}}function tt(e,o,s,a=P){const n=Math.sqrt(e/(6*o)),l=o*n;return{N:n,D:l,loss:R(n,l,s,a),tokensPerParam:o}}function H(e,o,s=64,a=30,n=P){const l=F(e,o,n).N,c=[],u=[],d=[];for(let i=0;i<s;i++){const p=i/(s-1),r=l/a*Math.pow(a*a,p),m=e/(6*r);c.push(r),u.push(m),d.push(R(r,m,o,n))}let h=0;for(let i=1;i<d.length;i++)d[i]<d[h]&&(h=i);return{N:c,D:u,loss:d,iMin:h,star:{N:c[h],D:u[h],loss:d[h]}}}function z(e,o,s,a){const n=a??e.defaultMfu;return e.peakTflops*1e12*n*o*s*3600}function J(e,o,s,a){const n=a??o.defaultMfu,l=o.peakTflops*1e12*n*s,c=e/l/3600;return{hours:c,days:c/24,cost:c*o.usdPerHour*s,mfu:n}}function K(e,o,s=P){let a=Math.log(1e18),n=Math.log(1e28);for(let l=0;l<60;l++){const c=.5*(a+n),u=Math.exp(c);F(u,o,s).loss>e?a=c:n=c}return Math.exp(.5*(a+n))}function j(e,o=T.map(a=>a.id),s){const a=C("adamw"),n=F(e,a);return o.map(l=>{const c=C(l),u=F(e,c),d=K(n.loss,c);return{...c,...u,nRatio:u.N/n.N,savings:1-d/e,matchCompute:d}})}function et(e,o=[.8,2],s=[.8,2],a=24,n=P){const l=F(e,C("adamw"),n),c=[],u=[];for(let i=0;i<a;i++){const p=i/(a-1);c.push(o[0]+p*(o[1]-o[0])),u.push(s[0]+p*(s[1]-s[0]))}const d=[],h=[];for(let i=0;i<a;i++){const p=[],r=[];for(let m=0;m<a;m++){const x={rho_n:c[m],rho_d:u[i]},w=F(e,x,n);p.push((w.N-l.N)/l.N),r.push(w.loss)}d.push(p),h.push(r)}return{rhoN:c,rhoD:u,deltaN:d,loss:h}}function ot(e,o=.01){const s=Math.abs(e);return s<=o?.5*e*e:o*(s-.5*o)}function G(e,o,s){return Math.min(s,Math.max(o,e))}function X(e,o,s){const a=o.filter(c=>c!==s),n={E:Math.exp(e[0]),A:Math.exp(e[1]),B:Math.exp(e[2]),alpha:G(e[3],.05,1.5),beta:G(e[4],.05,1.5)},l={[s]:{id:s,label:s,rho_n:1,rho_d:1}};return a.forEach((c,u)=>{l[c]={id:c,label:c,rho_n:Math.exp(e[5+2*u]),rho_d:Math.exp(e[5+2*u+1])}}),{params:n,rhos:l}}function B(e,o,s,a){const{params:n,rhos:l}=X(e,s,a);let c=0,u=0;for(const d of s){const h=o[d],i=l[d];for(let p=0;p<h.L.length;p++){const r=R(h.N[p],h.D[p],i,n);c+=ot(r-h.L[p]),u+=1}}return c/Math.max(u,1)}function U(e,o,s,a,n=1e-5){const l=new Array(e.length).fill(0),c=B(e,o,s,a);for(let u=0;u<e.length;u++){const d=e.slice();d[u]+=n,l[u]=(B(d,o,s,a)-c)/n}return l}function st(e,o,s,a,n=80){const l=e.length;let c=e.slice(),u=Array.from({length:l},(h,i)=>Array.from({length:l},(p,r)=>i===r?1:0)),d=U(c,o,s,a);for(let h=0;h<n;h++){const i=new Array(l).fill(0);for(let b=0;b<l;b++){let g=0;for(let _=0;_<l;_++)g+=u[b][_]*d[_];i[b]=-g}let p=1;const r=B(c,o,s,a);let m=c.map((b,g)=>b+p*i[g]),x=B(m,o,s,a),w=0;for(;x>r+1e-4*p*i.reduce((b,g,_)=>b+g*d[_],0)&&w<20;)p*=.5,m=c.map((b,g)=>b+p*i[g]),x=B(m,o,s,a),w+=1;const L=m.map((b,g)=>b-c[g]),v=U(m,o,s,a),N=v.map((b,g)=>b-d[g]),E=N.reduce((b,g,_)=>b+g*L[_],0);if(Math.hypot(...v)<1e-7)return{x:m,success:!0};if(E>1e-12){const b=1/E,g=new Array(l).fill(0);for(let y=0;y<l;y++){let S=0;for(let A=0;A<l;A++)S+=u[y][A]*N[A];g[y]=S}const _=N.reduce((y,S,A)=>y+S*g[A],0),O=Array.from({length:l},()=>new Array(l).fill(0));for(let y=0;y<l;y++)for(let S=0;S<l;S++)O[y][S]=u[y][S]-b*(L[y]*g[S]+g[y]*L[S])+b*b*_*L[y]*L[S]+b*L[y]*L[S];u=O}if(c=m,d=v,Math.abs(r-x)<1e-12)break}return{x:c,success:!0}}function at(e,o="adamw"){const s=Object.keys(e);if(!s.includes(o))throw new Error(`reference optimizer ${o} missing from runs`);const a=s.filter(i=>i!==o),n=new Array(5+2*a.length).fill(0);n[0]=Math.log(P.E),n[1]=Math.log(P.A),n[2]=Math.log(P.B),n[3]=P.alpha,n[4]=P.beta,a.forEach((i,p)=>{n[5+2*p]=0,n[5+2*p+1]=0});const{x:l,success:c}=st(n,e,s,o),{params:u,rhos:d}=X(l,s,o),h={};for(const i of s){const p=e[i];let r=0;for(let m=0;m<p.L.length;m++){const x=R(p.N[m],p.D[m],d[i],u);r+=(x-p.L[m])**2}h[i]=Math.sqrt(r/p.L.length)}return{params:u,rhos:Object.fromEntries(Object.entries(d).map(([i,p])=>[i,{rho_n:p.rho_n,rho_d:p.rho_d,label:p.label}])),per_optimizer_rmse:h,mean_rmse:Object.values(h).reduce((i,p)=>i+p,0)/Math.max(Object.values(h).length,1),success:c,method:"shared_exponents_lbfgs_js"}}const Z="optiscale-scenario-v2";function nt(){const e=C("muon");return{theme:"dark",tab:"practice",flops:1e24,optimizer:"muon",gpu:"h100",count:8,mfu:.4,ratioMode:!1,ratio:100,selected:["adamw","muon","normuon","soap"],rhoN:e.rho_n,rhoD:e.rho_d,targetLoss:null,targetLossMode:!1}}function rt(){const e=new URLSearchParams(location.search),o=nt();e.get("C")&&(o.flops=Number(e.get("C"))),e.get("opt")&&(o.optimizer=e.get("opt")),e.get("gpu")&&(o.gpu=e.get("gpu")),e.get("n")&&(o.count=Number(e.get("n"))),e.get("mfu")&&(o.mfu=Number(e.get("mfu")));try{const s=localStorage.getItem(Z);s&&Object.assign(o,JSON.parse(s))}catch{}return e.get("C")&&(o.flops=Number(e.get("C"))),e.get("opt")&&(o.optimizer=e.get("opt")),o}let t=rt(),k={};function it(){const e=C(t.optimizer);return Q(e,t.rhoN,t.rhoD)}function $(){localStorage.setItem(Z,JSON.stringify(t));const e=new URLSearchParams({C:String(t.flops),opt:t.optimizer,gpu:t.gpu,n:String(t.count),mfu:String(t.mfu)});history.replaceState(null,"",`?${e.toString()}`)}function f(e){const o=document.querySelector(e);if(!o)throw new Error(`Missing ${e}`);return o}function lt(){document.documentElement.dataset.theme=t.theme,document.querySelector("#app").innerHTML=`
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
        <button class="tab ${t.tab==="practice"?"active":""}" data-tab="practice">Practitioner</button>
        <button class="tab ${t.tab==="research"?"active":""}" data-tab="research">Research fit</button>
        <button class="tab ${t.tab==="formula"?"active":""}" data-tab="formula">Formulas</button>
      </div>
      <button class="icon-btn" id="themeBtn">${t.theme==="dark"?"Light":"Dark"}</button>
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
  `,document.querySelectorAll(".tab").forEach(e=>{e.addEventListener("click",()=>{t.tab=e.dataset.tab,$(),M()})}),f("#themeBtn").onclick=()=>{t.theme=t.theme==="dark"?"light":"dark",$(),M()},f("#shareBtn").onclick=async()=>{$(),await navigator.clipboard.writeText(location.href),f("#shareBtn").textContent="Copied",setTimeout(()=>f("#shareBtn").textContent="Copy share link",1200)},f("#exportMd").onclick=ft,f("#exportCsv").onclick=bt}function ct(){const e=D(t.gpu),o=it();let s=t.flops,a,n="";t.targetLossMode&&t.targetLoss!=null&&t.targetLoss>1.69?(s=K(t.targetLoss,o),a=F(s,o),n=`Inverse solve: C* ≈ ${q(s)} to reach L=${t.targetLoss}`):t.ratioMode?a=tt(t.flops,t.ratio,o):a=F(t.flops,o);const l=J(s,e,t.count,t.mfu),c=j(s,t.selected);f("#view").innerHTML=`
    <div class="grid">
      <aside class="panel">
        <h2>Budget</h2>
        <div class="row" id="presets"></div>
        <label>FLOP budget C
          <input id="flopsInput" type="text" value="${s.toExponential(3)}" ${t.targetLossMode?"disabled":""} />
        </label>
        <label>Or GPU-hours → FLOPs
          <div class="row">
            <select id="gpuSel">${W.map(r=>`<option value="${r.id}" ${r.id===t.gpu?"selected":""}>${r.name}</option>`).join("")}</select>
          </div>
        </label>
        <label>GPU count
          <input id="countInput" type="number" min="1" value="${t.count}" />
        </label>
        <label>Hours
          <input id="hoursInput" type="number" min="0.1" step="0.1" value="${l.hours.toFixed(1)}" ${t.targetLossMode?"disabled":""} />
        </label>
        <label>MFU ${(t.mfu*100).toFixed(0)}%
          <input id="mfuInput" type="range" min="0.15" max="0.7" step="0.01" value="${t.mfu}" />
        </label>
        <label>Optimizer
          <select id="optSel">${T.map(r=>`<option value="${r.id}" ${r.id===t.optimizer?"selected":""}>${r.label}</option>`).join("")}</select>
        </label>
        <label>Custom ρ<sub>N</sub>
          <input id="rhoNInput" type="number" min="0.1" max="5" step="0.01" value="${t.rhoN}" />
        </label>
        <label>Custom ρ<sub>D</sub>
          <input id="rhoDInput" type="number" min="0.1" max="5" step="0.01" value="${t.rhoD}" />
        </label>
        <label><input id="ratioMode" type="checkbox" ${t.ratioMode?"checked":""} ${t.targetLossMode?"disabled":""}/> Fix tokens/param ratio</label>
        <label class="${t.ratioMode&&!t.targetLossMode?"":"hidden"}">Tokens / param
          <input id="ratioInput" type="number" min="1" value="${t.ratio}" />
        </label>
        <label><input id="targetLossMode" type="checkbox" ${t.targetLossMode?"checked":""}/> Target-loss inverse</label>
        <label class="${t.targetLossMode?"":"hidden"}">Target loss L
          <input id="targetLossInput" type="number" min="1.7" max="10" step="0.01" value="${t.targetLoss??2.5}" />
        </label>
        <p class="muted">ρ fields override catalog priors for the selected optimizer. Use Research → Apply ρ to seed from a fit.</p>
      </aside>
      <section class="panel">
        <h2>Allocation — ${o.label} (ρN=${o.rho_n}, ρD=${o.rho_d})</h2>
        ${n?`<p class="muted">${n}</p>`:""}
        <div class="cards">
          <div class="card"><div class="k">N*</div><div class="v accent">${I(a.N)}</div></div>
          <div class="card"><div class="k">D*</div><div class="v">${I(a.D)}</div></div>
          <div class="card"><div class="k">tok / param</div><div class="v">${a.tokensPerParam.toFixed(1)}</div></div>
          <div class="card"><div class="k">Pred. loss</div><div class="v">${a.loss.toFixed(4)}</div></div>
          <div class="card"><div class="k">Wall-clock</div><div class="v">${l.days.toFixed(2)} d</div></div>
          <div class="card"><div class="k">Est. $</div><div class="v">$${l.cost.toLocaleString(void 0,{maximumFractionDigits:0})}</div></div>
        </div>
        <h2>IsoFLOP curve</h2>
        <canvas id="isoCanvas" width="900" height="360"></canvas>
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
            ${c.map(r=>`<tr>
              <td>${r.label}</td>
              <td class="mono">${I(r.N)}</td>
              <td class="mono">${I(r.D)}</td>
              <td class="mono">${r.loss.toFixed(4)}</td>
              <td class="mono">${r.nRatio.toFixed(3)}</td>
              <td class="mono ${r.savings>0?"delta-pos":""}">${(r.savings*100).toFixed(1)}%</td>
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
  `;const u=f("#presets");V.forEach(r=>{const m=document.createElement("button");m.className="chip",m.textContent=r.label,m.onclick=()=>pt(r.id),u.appendChild(m)});const d=f("#optChips");T.forEach(r=>{const m=document.createElement("button");m.className=`chip ${t.selected.includes(r.id)?"active":""}`,m.textContent=r.label,m.onclick=()=>{t.selected.includes(r.id)?t.selected=t.selected.filter(x=>x!==r.id):t.selected=[...t.selected,r.id],t.selected.length||(t.selected=["adamw"]),$(),M()},d.appendChild(m)}),f("#flopsInput").onchange=r=>{t.flops=Number(r.target.value),$(),M()},f("#gpuSel").onchange=r=>{t.gpu=r.target.value;const m=D(t.gpu);t.mfu=m.defaultMfu,$(),M()},f("#countInput").onchange=r=>{t.count=Math.max(1,Number(r.target.value)),$(),M()},f("#hoursInput").onchange=r=>{const m=Number(r.target.value);t.flops=z(D(t.gpu),t.count,m,t.mfu),$(),M()},f("#mfuInput").oninput=r=>{t.mfu=Number(r.target.value),$(),M()},f("#optSel").onchange=r=>{t.optimizer=r.target.value;const m=C(t.optimizer);t.rhoN=m.rho_n,t.rhoD=m.rho_d,$(),M()};const h=r=>{const x=r.target.id;t.rhoN=Number(f("#rhoNInput").value),t.rhoD=Number(f("#rhoDInput").value),$(),M();const w=document.getElementById(x);if(w){w.focus();const L=w.value.length;w.setSelectionRange(L,L)}};f("#rhoNInput").oninput=h,f("#rhoDInput").oninput=h,f("#ratioMode").onchange=r=>{t.ratioMode=r.target.checked,t.ratioMode&&(t.targetLossMode=!1),$(),M()};const i=document.querySelector("#ratioInput");i&&(i.onchange=r=>{t.ratio=Number(r.target.value),$(),M()}),f("#targetLossMode").onchange=r=>{t.targetLossMode=r.target.checked,t.targetLossMode&&(t.ratioMode=!1,t.targetLoss==null&&(t.targetLoss=2.5)),$(),M()};const p=document.querySelector("#targetLossInput");p&&(p.onchange=r=>{t.targetLoss=Number(r.target.value),$(),M()}),f("#copyPy").onclick=async()=>{const r=`from optiscale import allocate_for_budget, cost_report
alloc = allocate_for_budget(${t.flops.toExponential()}, optimizer="${t.optimizer}", rho_n=${t.rhoN}, rho_d=${t.rhoD})
print(alloc)
print(cost_report(compute=${t.flops.toExponential()}, gpu_id="${t.gpu}", count=${t.count}))
`;f("#pySnippet").textContent=r,await navigator.clipboard.writeText(r)},dt(o,s),ut(s)}function dt(e,o=t.flops){const s=f("#isoCanvas"),a=s.getContext("2d"),n=H(o,e),l=H(o,C("adamw")),c=s.width,u=s.height,d=48;a.clearRect(0,0,c,u);const h=[...n.loss,...l.loss],i=Math.min(...h),p=Math.max(...h),r=Math.min(...n.N),m=Math.max(...n.N),x=v=>d+(Math.log(v)-Math.log(r))/(Math.log(m)-Math.log(r))*(c-2*d),w=v=>u-d-(v-i)/(p-i||1)*(u-2*d);a.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--line"),a.beginPath(),a.moveTo(d,d),a.lineTo(d,u-d),a.lineTo(c-d,u-d),a.stroke();const L=(v,N)=>{a.strokeStyle=N,a.lineWidth=2,a.beginPath(),v.N.forEach((b,g)=>{const _=x(b),O=w(v.loss[g]);g===0?a.moveTo(_,O):a.lineTo(_,O)}),a.stroke();const E=v.star;a.fillStyle=N,a.beginPath(),a.arc(x(E.N),w(E.loss),5,0,Math.PI*2),a.fill()};L(l,getComputedStyle(document.documentElement).getPropertyValue("--adamw").trim()||"#7aa2c8"),L(n,getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()||"#3dcf8e"),a.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted").trim(),a.font="12px IBM Plex Mono",a.fillText("N (log)",c/2-20,u-14),a.save(),a.translate(14,u/2),a.rotate(-Math.PI/2),a.fillText("Loss",0,0),a.restore(),a.fillText("AdamW",c-d-120,d+8),a.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--accent").trim(),a.fillText(e.label,c-d-120,d+24)}function ut(e=t.flops){const o=f("#heatCanvas"),s=o.getContext("2d"),a=et(e,[.8,2],[.8,2],28),n=56,l=48,c=24,u=24,d=o.width,h=o.height,i=(d-n-u)/a.rhoN.length,p=(h-c-l)/a.rhoD.length;s.clearRect(0,0,d,h);let r=1/0,m=-1/0;for(const v of a.deltaN)for(const N of v)N<r&&(r=N),N>m&&(m=N);const x=m-r||1;for(let v=0;v<a.rhoD.length;v++)for(let N=0;N<a.rhoN.length;N++){const b=(a.deltaN[v][N]-r)/x,g=Math.round(40+b*180),_=Math.round(80+(1-Math.abs(b-.5)*2)*120),O=Math.round(160-b*120);s.fillStyle=`rgb(${g},${_},${O})`;const y=h-l-(v+1)*p;s.fillRect(n+N*i,y,i+.5,p+.5)}const w=(t.rhoN-.8)/(2-.8),L=(t.rhoD-.8)/(2-.8);s.strokeStyle="#fff",s.lineWidth=2,s.beginPath(),s.arc(n+w*(d-n-u),h-l-L*(h-c-l),6,0,Math.PI*2),s.stroke(),s.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted").trim()||"#888",s.font="11px IBM Plex Mono",s.fillText("ρ_N →",n+(d-n-u)/2-16,h-14),s.save(),s.translate(14,h/2),s.rotate(-Math.PI/2),s.fillText("ρ_D →",0,0),s.restore(),s.fillText(`ΔN* [${(r*100).toFixed(0)}%, ${(m*100).toFixed(0)}%]`,n,16)}function pt(e){const o=V.find(s=>s.id===e);if(o){if(t.targetLossMode=!1,o.kind==="flops")t.flops=o.value,t.ratioMode=!!o.ratio,o.ratio&&(t.ratio=o.ratio);else if(o.kind==="usd"){t.gpu=o.gpu,t.count=o.count;const s=D(t.gpu);t.mfu=s.defaultMfu;const a=o.value/(s.usdPerHour*t.count);t.flops=z(s,t.count,a,t.mfu)}else o.kind==="gpuHours"&&(t.gpu=o.gpu,t.count=o.count,t.mfu=D(t.gpu).defaultMfu,t.flops=z(D(t.gpu),t.count,o.hours,t.mfu));$(),M()}}function ht(){const e=Object.keys(k);f("#view").innerHTML=`
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
          ${e.length?e.map(o=>`<option value="${o}">${k[o].label} (ρN=${k[o].rho_n.toFixed(2)})</option>`).join(""):'<option value="">Run fit first</option>'}
        </select>
      </div>
      <div id="fitOut" class="muted">Results appear here.</div>
    </div>
  `,f("#synthBtn").onclick=()=>{const o=["N,D,L,optimizer"];for(const s of["adamw","muon"]){const a=C(s),n=[1e19,1e20,1e21,1e22];for(const l of n){const c=F(l,a).N;for(let u=0;u<8;u++){const d=c/6*Math.pow(36,u/7),h=l/(6*d),i=d*a.rho_n,p=h*a.rho_d,r=1.69+406.4/i**.34+410.7/p**.28+(Math.random()-.5)*.02;o.push(`${d},${h},${r},${s}`)}}}f("#csvIn").value=o.join(`
`)},f("#fitBtn").onclick=()=>{var h;const o=f("#csvIn").value.trim();if(!o){f("#fitOut").textContent="Paste CSV or load synthetic demo first.";return}const s=o.split(/\r?\n/).slice(1).map(i=>i.split(",")).filter(i=>i.length>=4).map(i=>({N:Number(i[0]),D:Number(i[1]),L:Number(i[2]),optimizer:i[3].trim().toLowerCase()})),a={};for(const i of s)a[h=i.optimizer]??(a[h]={N:[],D:[],L:[]}),a[i.optimizer].N.push(i.N),a[i.optimizer].D.push(i.D),a[i.optimizer].L.push(i.L);k={};let n;try{n=at(a)}catch(i){f("#fitOut").textContent=`Fit failed: ${i.message}`;return}const l=[`Browser L-BFGS shared-exponent fit (success=${n.success})`,`mean RMSE=${n.mean_rmse.toFixed(4)}`,`params: E=${n.params.E.toFixed(3)} A=${n.params.A.toFixed(1)} B=${n.params.B.toFixed(1)} α=${n.params.alpha.toFixed(3)} β=${n.params.beta.toFixed(3)}`,""];for(const[i,p]of Object.entries(n.rhos)){const r=n.per_optimizer_rmse[i]??NaN;l.push(`${p.label}: ρN=${p.rho_n.toFixed(3)} ρD=${p.rho_d.toFixed(3)}  RMSE=${r.toFixed(4)}`),k[i]={rho_n:p.rho_n,rho_d:p.rho_d,label:p.label}}l.push("","Download fit.json or Apply ρ to Practitioner."),window.__lastFitJson=n,f("#fitOut").textContent=l.join(`
`);const c=f("#applyRhoBtn"),u=f("#applyOptSel");c.disabled=!1,u.disabled=!1,u.innerHTML=Object.keys(k).map(i=>`<option value="${i}">${k[i].label} (ρN=${k[i].rho_n.toFixed(2)})</option>`).join("");const d=f("#dlFitBtn");d.disabled=!1,d.onclick=()=>{const i=window.__lastFitJson;if(!i)return;const p=new Blob([JSON.stringify(i,null,2)],{type:"application/json"}),r=document.createElement("a");r.href=URL.createObjectURL(p),r.download="fit.json",r.click()}},f("#applyRhoBtn").onclick=()=>{const o=f("#applyOptSel").value,s=k[o];s&&(T.some(a=>a.id===o)&&(t.optimizer=o),t.rhoN=s.rho_n,t.rhoD=s.rho_d,t.tab="practice",$(),M())}}function mt(){f("#view").innerHTML=`
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
  `}function ft(){const e=j(t.flops,t.selected),o=D(t.gpu),s=J(t.flops,o,t.count,t.mfu),a=["# OptiScale Report","",`- Compute: ${q(t.flops)}`,`- GPU: ${o.name} × ${t.count} @ MFU ${t.mfu}`,`- Wall-clock: ${s.days.toFixed(2)} days`,`- Cost: $${s.cost.toFixed(0)}`,`- Practitioner ρ: ρN=${t.rhoN}, ρD=${t.rhoD}`,"","| Optimizer | N* | D* | Loss | N/AdamW | Compute saved |","|---|---:|---:|---:|---:|---:|",...e.map(n=>`| ${n.label} | ${I(n.N)} | ${I(n.D)} | ${n.loss.toFixed(4)} | ${n.nRatio.toFixed(3)} | ${(n.savings*100).toFixed(1)}% |`),""].join(`
`);Y("optiscale-report.md",a)}function bt(){const o=["optimizer,N,D,loss,tokens_per_param,n_ratio,compute_saved",...j(t.flops,t.selected).map(s=>`${s.id},${s.N},${s.D},${s.loss},${s.tokensPerParam},${s.nRatio},${s.savings}`)];Y("optiscale-compare.csv",o.join(`
`))}function Y(e,o){const s=new Blob([o],{type:"text/plain"}),a=document.createElement("a");a.href=URL.createObjectURL(s),a.download=e,a.click(),URL.revokeObjectURL(a.href)}function M(){lt(),t.tab==="practice"?ct():t.tab==="research"?ht():mt()}M();
