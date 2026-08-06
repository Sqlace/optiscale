(function(){const o=document.createElement("link").relList;if(o&&o.supports&&o.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))e(n);new MutationObserver(n=>{for(const l of n)if(l.type==="childList")for(const c of l.addedNodes)c.tagName==="LINK"&&c.rel==="modulepreload"&&e(c)}).observe(document,{childList:!0,subtree:!0});function a(n){const l={};return n.integrity&&(l.integrity=n.integrity),n.referrerPolicy&&(l.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?l.credentials="include":n.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function e(n){if(n.ep)return;n.ep=!0;const l=a(n);fetch(n.href,l)}})();const _={E:1.69,A:406.4,B:410.7,alpha:.34,beta:.28},k=[{id:"adamw",label:"AdamW",rho_n:1,rho_d:1},{id:"muon",label:"Muon",rho_n:1.35,rho_d:1.25},{id:"normuon",label:"NorMuon",rho_n:1.45,rho_d:1.3},{id:"aurora",label:"Aurora",rho_n:1.4,rho_d:1.28},{id:"soap",label:"SOAP",rho_n:1.2,rho_d:1.15},{id:"shampoo",label:"Shampoo",rho_n:1.18,rho_d:1.12},{id:"lion",label:"Lion",rho_n:1.05,rho_d:1.02}],A=[{id:"h100",name:"H100 SXM",peakTflops:989,usdPerHour:4,defaultMfu:.4,memoryGb:80},{id:"h100_pcie",name:"H100 PCIe",peakTflops:756,usdPerHour:3.5,defaultMfu:.38,memoryGb:80},{id:"a100",name:"A100 80GB",peakTflops:312,usdPerHour:2.2,defaultMfu:.45,memoryGb:80},{id:"a100_40",name:"A100 40GB",peakTflops:312,usdPerHour:1.8,defaultMfu:.42,memoryGb:40},{id:"l40s",name:"L40S",peakTflops:362,usdPerHour:1.6,defaultMfu:.35,memoryGb:48},{id:"rtx4090",name:"RTX 4090",peakTflops:330,usdPerHour:.8,defaultMfu:.3,memoryGb:24},{id:"v100",name:"V100",peakTflops:125,usdPerHour:.9,defaultMfu:.4,memoryGb:32},{id:"tpu_v5e",name:"TPU v5e",peakTflops:197,usdPerHour:1.2,defaultMfu:.5,memoryGb:16}],R=[{id:"1m",label:"$1M pretrain",kind:"usd",value:1e6,gpu:"h100",count:64},{id:"8xh100-30d",label:"8×H100 × 30 days",kind:"gpuHours",hours:24*30,gpu:"h100",count:8},{id:"70b",label:"Chinchilla 70B-ish",kind:"flops",value:6*7e10*14e11},{id:"1e24",label:"10²⁴ FLOPs",kind:"flops",value:1e24},{id:"overtrain",label:"Overtrain @ 100 tok/param",kind:"flops",value:1e23,ratio:100}];function P(s){const o=k.find(a=>a.id===s);if(!o)throw new Error(`Unknown optimizer ${s}`);return o}function w(s){const o=A.find(a=>a.id===s);if(!o)throw new Error(`Unknown GPU ${s}`);return o}function q(s,o,a){return{...s,rho_n:o??s.rho_n,rho_d:a??s.rho_d}}function S(s){return s>=1e12?`${(s/1e12).toFixed(2)}T`:s>=1e9?`${(s/1e9).toFixed(2)}B`:s>=1e6?`${(s/1e6).toFixed(2)}M`:s>=1e3?`${(s/1e3).toFixed(2)}K`:s.toFixed(0)}function H(s){return s>=1e24?`${(s/1e24).toFixed(3)}×10²⁴`:s>=1e21?`${(s/1e21).toFixed(2)} ZFLOP`:s>=1e18?`${(s/1e18).toFixed(2)} EFLOP`:s.toExponential(2)}function E(s,o,a,e=_){const n=s*a.rho_n,l=o*a.rho_d;return e.E+e.A/n**e.alpha+e.B/l**e.beta}function L(s,o,a=_){const e=a.alpha,n=a.beta,l=(a.A*e/(a.B*n))**(1/(e+n)),c=(o.rho_d**n/o.rho_n**e)**(1/(e+n)),u=l*c*(s/6)**(n/(e+n)),r=s/(6*u);return{N:u,D:r,loss:E(u,r,o,a),tokensPerParam:r/u}}function K(s,o,a,e=_){const n=Math.sqrt(s/(6*o)),l=o*n;return{N:n,D:l,loss:E(n,l,a,e),tokensPerParam:o}}function z(s,o,a=64,e=30,n=_){const l=L(s,o,n).N,c=[],u=[],r=[];for(let m=0;m<a;m++){const f=m/(a-1),i=l/e*Math.pow(e*e,f),d=s/(6*i);c.push(i),u.push(d),r.push(E(i,d,o,n))}let h=0;for(let m=1;m<r.length;m++)r[m]<r[h]&&(h=m);return{N:c,D:u,loss:r,iMin:h,star:{N:c[h],D:u[h],loss:r[h]}}}function T(s,o,a,e){const n=e??s.defaultMfu;return s.peakTflops*1e12*n*o*a*3600}function G(s,o,a,e){const n=e??o.defaultMfu,l=o.peakTflops*1e12*n*a,c=s/l/3600;return{hours:c,days:c/24,cost:c*o.usdPerHour*a,mfu:n}}function j(s,o,a=_){let e=Math.log(1e18),n=Math.log(1e28);for(let l=0;l<60;l++){const c=.5*(e+n),u=Math.exp(c);L(u,o,a).loss>s?e=c:n=c}return Math.exp(.5*(e+n))}function B(s,o=k.map(e=>e.id),a){const e=P("adamw"),n=L(s,e);return o.map(l=>{const c=P(l),u=L(s,c),r=j(n.loss,c);return{...c,...u,nRatio:u.N/n.N,savings:1-r/s,matchCompute:r}})}function J(s,o=[.8,2],a=[.8,2],e=24,n=_){const l=L(s,P("adamw"),n),c=[],u=[];for(let m=0;m<e;m++){const f=m/(e-1);c.push(o[0]+f*(o[1]-o[0])),u.push(a[0]+f*(a[1]-a[0]))}const r=[],h=[];for(let m=0;m<e;m++){const f=[],i=[];for(let d=0;d<e;d++){const y={rho_n:c[d],rho_d:u[m]},v=L(s,y,n);f.push((v.N-l.N)/l.N),i.push(v.loss)}r.push(f),h.push(i)}return{rhoN:c,rhoD:u,deltaN:r,loss:h}}const U="optiscale-scenario-v2";function X(){const s=P("muon");return{theme:"dark",tab:"practice",flops:1e24,optimizer:"muon",gpu:"h100",count:8,mfu:.4,ratioMode:!1,ratio:100,selected:["adamw","muon","normuon","soap"],rhoN:s.rho_n,rhoD:s.rho_d,targetLoss:null,targetLossMode:!1}}function Z(){const s=new URLSearchParams(location.search),o=X();s.get("C")&&(o.flops=Number(s.get("C"))),s.get("opt")&&(o.optimizer=s.get("opt")),s.get("gpu")&&(o.gpu=s.get("gpu")),s.get("n")&&(o.count=Number(s.get("n"))),s.get("mfu")&&(o.mfu=Number(s.get("mfu")));try{const a=localStorage.getItem(U);a&&Object.assign(o,JSON.parse(a))}catch{}return s.get("C")&&(o.flops=Number(s.get("C"))),s.get("opt")&&(o.optimizer=s.get("opt")),o}let t=Z(),M={};function Y(){const s=P(t.optimizer);return q(s,t.rhoN,t.rhoD)}function b(){localStorage.setItem(U,JSON.stringify(t));const s=new URLSearchParams({C:String(t.flops),opt:t.optimizer,gpu:t.gpu,n:String(t.count),mfu:String(t.mfu)});history.replaceState(null,"",`?${s.toString()}`)}function p(s){const o=document.querySelector(s);if(!o)throw new Error(`Missing ${s}`);return o}function Q(){document.documentElement.dataset.theme=t.theme,document.querySelector("#app").innerHTML=`
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
  `,document.querySelectorAll(".tab").forEach(s=>{s.addEventListener("click",()=>{t.tab=s.dataset.tab,b(),g()})}),p("#themeBtn").onclick=()=>{t.theme=t.theme==="dark"?"light":"dark",b(),g()},p("#shareBtn").onclick=async()=>{b(),await navigator.clipboard.writeText(location.href),p("#shareBtn").textContent="Copied",setTimeout(()=>p("#shareBtn").textContent="Copy share link",1200)},p("#exportMd").onclick=it,p("#exportCsv").onclick=rt}function tt(){const s=w(t.gpu),o=Y();let a=t.flops,e,n="";t.targetLossMode&&t.targetLoss!=null&&t.targetLoss>1.69?(a=j(t.targetLoss,o),e=L(a,o),n=`Inverse solve: C* ≈ ${H(a)} to reach L=${t.targetLoss}`):t.ratioMode?e=K(t.flops,t.ratio,o):e=L(t.flops,o);const l=G(a,s,t.count,t.mfu),c=B(a,t.selected);p("#view").innerHTML=`
    <div class="grid">
      <aside class="panel">
        <h2>Budget</h2>
        <div class="row" id="presets"></div>
        <label>FLOP budget C
          <input id="flopsInput" type="text" value="${a.toExponential(3)}" ${t.targetLossMode?"disabled":""} />
        </label>
        <label>Or GPU-hours → FLOPs
          <div class="row">
            <select id="gpuSel">${A.map(i=>`<option value="${i.id}" ${i.id===t.gpu?"selected":""}>${i.name}</option>`).join("")}</select>
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
          <select id="optSel">${k.map(i=>`<option value="${i.id}" ${i.id===t.optimizer?"selected":""}>${i.label}</option>`).join("")}</select>
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
          <div class="card"><div class="k">N*</div><div class="v accent">${S(e.N)}</div></div>
          <div class="card"><div class="k">D*</div><div class="v">${S(e.D)}</div></div>
          <div class="card"><div class="k">tok / param</div><div class="v">${e.tokensPerParam.toFixed(1)}</div></div>
          <div class="card"><div class="k">Pred. loss</div><div class="v">${e.loss.toFixed(4)}</div></div>
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
            ${c.map(i=>`<tr>
              <td>${i.label}</td>
              <td class="mono">${S(i.N)}</td>
              <td class="mono">${S(i.D)}</td>
              <td class="mono">${i.loss.toFixed(4)}</td>
              <td class="mono">${i.nRatio.toFixed(3)}</td>
              <td class="mono ${i.savings>0?"delta-pos":""}">${(i.savings*100).toFixed(1)}%</td>
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
  `;const u=p("#presets");R.forEach(i=>{const d=document.createElement("button");d.className="chip",d.textContent=i.label,d.onclick=()=>st(i.id),u.appendChild(d)});const r=p("#optChips");k.forEach(i=>{const d=document.createElement("button");d.className=`chip ${t.selected.includes(i.id)?"active":""}`,d.textContent=i.label,d.onclick=()=>{t.selected.includes(i.id)?t.selected=t.selected.filter(y=>y!==i.id):t.selected=[...t.selected,i.id],t.selected.length||(t.selected=["adamw"]),b(),g()},r.appendChild(d)}),p("#flopsInput").onchange=i=>{t.flops=Number(i.target.value),b(),g()},p("#gpuSel").onchange=i=>{t.gpu=i.target.value;const d=w(t.gpu);t.mfu=d.defaultMfu,b(),g()},p("#countInput").onchange=i=>{t.count=Math.max(1,Number(i.target.value)),b(),g()},p("#hoursInput").onchange=i=>{const d=Number(i.target.value);t.flops=T(w(t.gpu),t.count,d,t.mfu),b(),g()},p("#mfuInput").oninput=i=>{t.mfu=Number(i.target.value),b(),g()},p("#optSel").onchange=i=>{t.optimizer=i.target.value;const d=P(t.optimizer);t.rhoN=d.rho_n,t.rhoD=d.rho_d,b(),g()};const h=i=>{const y=i.target.id;t.rhoN=Number(p("#rhoNInput").value),t.rhoD=Number(p("#rhoDInput").value),b(),g();const v=document.getElementById(y);if(v){v.focus();const x=v.value.length;v.setSelectionRange(x,x)}};p("#rhoNInput").oninput=h,p("#rhoDInput").oninput=h,p("#ratioMode").onchange=i=>{t.ratioMode=i.target.checked,t.ratioMode&&(t.targetLossMode=!1),b(),g()};const m=document.querySelector("#ratioInput");m&&(m.onchange=i=>{t.ratio=Number(i.target.value),b(),g()}),p("#targetLossMode").onchange=i=>{t.targetLossMode=i.target.checked,t.targetLossMode&&(t.ratioMode=!1,t.targetLoss==null&&(t.targetLoss=2.5)),b(),g()};const f=document.querySelector("#targetLossInput");f&&(f.onchange=i=>{t.targetLoss=Number(i.target.value),b(),g()}),p("#copyPy").onclick=async()=>{const i=`from optiscale import allocate_for_budget, cost_report
alloc = allocate_for_budget(${t.flops.toExponential()}, optimizer="${t.optimizer}", rho_n=${t.rhoN}, rho_d=${t.rhoD})
print(alloc)
print(cost_report(compute=${t.flops.toExponential()}, gpu_id="${t.gpu}", count=${t.count}))
`;p("#pySnippet").textContent=i,await navigator.clipboard.writeText(i)},et(o,a),ot(a)}function et(s,o=t.flops){const a=p("#isoCanvas"),e=a.getContext("2d"),n=z(o,s),l=z(o,P("adamw")),c=a.width,u=a.height,r=48;e.clearRect(0,0,c,u);const h=[...n.loss,...l.loss],m=Math.min(...h),f=Math.max(...h),i=Math.min(...n.N),d=Math.max(...n.N),y=$=>r+(Math.log($)-Math.log(i))/(Math.log(d)-Math.log(i))*(c-2*r),v=$=>u-r-($-m)/(f-m||1)*(u-2*r);e.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--line"),e.beginPath(),e.moveTo(r,r),e.lineTo(r,u-r),e.lineTo(c-r,u-r),e.stroke();const x=($,N)=>{e.strokeStyle=N,e.lineWidth=2,e.beginPath(),$.N.forEach((C,D)=>{const F=y(C),O=v($.loss[D]);D===0?e.moveTo(F,O):e.lineTo(F,O)}),e.stroke();const I=$.star;e.fillStyle=N,e.beginPath(),e.arc(y(I.N),v(I.loss),5,0,Math.PI*2),e.fill()};x(l,getComputedStyle(document.documentElement).getPropertyValue("--adamw").trim()||"#7aa2c8"),x(n,getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()||"#3dcf8e"),e.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted").trim(),e.font="12px IBM Plex Mono",e.fillText("N (log)",c/2-20,u-14),e.save(),e.translate(14,u/2),e.rotate(-Math.PI/2),e.fillText("Loss",0,0),e.restore(),e.fillText("AdamW",c-r-120,r+8),e.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--accent").trim(),e.fillText(s.label,c-r-120,r+24)}function ot(s=t.flops){const o=p("#heatCanvas"),a=o.getContext("2d"),e=J(s,[.8,2],[.8,2],28),n=56,l=48,c=24,u=24,r=o.width,h=o.height,m=(r-n-u)/e.rhoN.length,f=(h-c-l)/e.rhoD.length;a.clearRect(0,0,r,h);let i=1/0,d=-1/0;for(const $ of e.deltaN)for(const N of $)N<i&&(i=N),N>d&&(d=N);const y=d-i||1;for(let $=0;$<e.rhoD.length;$++)for(let N=0;N<e.rhoN.length;N++){const C=(e.deltaN[$][N]-i)/y,D=Math.round(40+C*180),F=Math.round(80+(1-Math.abs(C-.5)*2)*120),O=Math.round(160-C*120);a.fillStyle=`rgb(${D},${F},${O})`;const V=h-l-($+1)*f;a.fillRect(n+N*m,V,m+.5,f+.5)}const v=(t.rhoN-.8)/(2-.8),x=(t.rhoD-.8)/(2-.8);a.strokeStyle="#fff",a.lineWidth=2,a.beginPath(),a.arc(n+v*(r-n-u),h-l-x*(h-c-l),6,0,Math.PI*2),a.stroke(),a.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted").trim()||"#888",a.font="11px IBM Plex Mono",a.fillText("ρ_N →",n+(r-n-u)/2-16,h-14),a.save(),a.translate(14,h/2),a.rotate(-Math.PI/2),a.fillText("ρ_D →",0,0),a.restore(),a.fillText(`ΔN* [${(i*100).toFixed(0)}%, ${(d*100).toFixed(0)}%]`,n,16)}function st(s){const o=R.find(a=>a.id===s);if(o){if(t.targetLossMode=!1,o.kind==="flops")t.flops=o.value,t.ratioMode=!!o.ratio,o.ratio&&(t.ratio=o.ratio);else if(o.kind==="usd"){t.gpu=o.gpu,t.count=o.count;const a=w(t.gpu);t.mfu=a.defaultMfu;const e=o.value/(a.usdPerHour*t.count);t.flops=T(a,t.count,e,t.mfu)}else o.kind==="gpuHours"&&(t.gpu=o.gpu,t.count=o.count,t.mfu=w(t.gpu).defaultMfu,t.flops=T(w(t.gpu),t.count,o.hours,t.mfu));b(),g()}}function at(){const s=Object.keys(M);p("#view").innerHTML=`
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
        <button class="icon-btn" id="applyRhoBtn" ${s.length?"":"disabled"}>Apply ρ to Practitioner</button>
        <select id="applyOptSel" ${s.length?"":"disabled"}>
          ${s.length?s.map(o=>`<option value="${o}">${M[o].label} (ρN=${M[o].rho_n.toFixed(2)})</option>`).join(""):'<option value="">Run fit first</option>'}
        </select>
      </div>
      <div id="fitOut" class="muted">Results appear here.</div>
    </div>
  `,p("#synthBtn").onclick=()=>{const o=["N,D,L,optimizer"];for(const a of["adamw","muon"]){const e=P(a),n=[1e19,1e20,1e21,1e22];for(const l of n){const c=L(l,e).N;for(let u=0;u<8;u++){const r=c/6*Math.pow(36,u/7),h=l/(6*r),m=r*e.rho_n,f=h*e.rho_d,i=1.69+406.4/m**.34+410.7/f**.28+(Math.random()-.5)*.02;o.push(`${r},${h},${i},${a}`)}}}p("#csvIn").value=o.join(`
`)},p("#fitBtn").onclick=()=>{var u;const o=p("#csvIn").value.trim();if(!o){p("#fitOut").textContent="Paste CSV or load synthetic demo first.";return}const a=o.split(/\r?\n/).slice(1).map(r=>r.split(",")).filter(r=>r.length>=4).map(r=>({N:Number(r[0]),D:Number(r[1]),L:Number(r[2]),optimizer:r[3].trim().toLowerCase()})),e={};for(const r of a)e[u=r.optimizer]??(e[u]={N:[],D:[],L:[]}),e[r.optimizer].N.push(r.N),e[r.optimizer].D.push(r.D),e[r.optimizer].L.push(r.L);M={};const n=["Browser fit summary (full L-BFGS lives in Python `optiscale fit`):",""];for(const[r,h]of Object.entries(e)){const m=k.find(d=>d.id===r)??{label:r,rho_n:1,rho_d:1};let f=0;for(let d=0;d<h.N.length;d++){const y=h.N[d]*m.rho_n,v=h.D[d]*m.rho_d,x=1.69+406.4/y**.34+410.7/v**.28;f+=(x-h.L[d])**2}const i=Math.sqrt(f/h.N.length);n.push(`${m.label}: n=${h.N.length}  RMSE under prior ρ = ${i.toFixed(4)}  (ρN=${m.rho_n}, ρD=${m.rho_d})`),M[r]={rho_n:m.rho_n,rho_d:m.rho_d,label:m.label}}n.push("","Run for full shared-exponent fit + bootstrap CI:"),n.push("  pip install -e ."),n.push("  optiscale fit --synthetic --bootstrap 200 --out fit.json"),n.push("  optiscale compare --flops 1e24 --fit fit.json"),p("#fitOut").textContent=n.join(`
`);const l=p("#applyRhoBtn"),c=p("#applyOptSel");l.disabled=!1,c.disabled=!1,c.innerHTML=Object.keys(M).map(r=>`<option value="${r}">${M[r].label} (ρN=${M[r].rho_n.toFixed(2)})</option>`).join("")},p("#applyRhoBtn").onclick=()=>{const o=p("#applyOptSel").value,a=M[o];a&&(k.some(e=>e.id===o)&&(t.optimizer=o),t.rhoN=a.rho_n,t.rhoD=a.rho_d,t.tab="practice",b(),g())}}function nt(){p("#view").innerHTML=`
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
  `}function it(){const s=B(t.flops,t.selected),o=w(t.gpu),a=G(t.flops,o,t.count,t.mfu),e=["# OptiScale Report","",`- Compute: ${H(t.flops)}`,`- GPU: ${o.name} × ${t.count} @ MFU ${t.mfu}`,`- Wall-clock: ${a.days.toFixed(2)} days`,`- Cost: $${a.cost.toFixed(0)}`,`- Practitioner ρ: ρN=${t.rhoN}, ρD=${t.rhoD}`,"","| Optimizer | N* | D* | Loss | N/AdamW | Compute saved |","|---|---:|---:|---:|---:|---:|",...s.map(n=>`| ${n.label} | ${S(n.N)} | ${S(n.D)} | ${n.loss.toFixed(4)} | ${n.nRatio.toFixed(3)} | ${(n.savings*100).toFixed(1)}% |`),""].join(`
`);W("optiscale-report.md",e)}function rt(){const o=["optimizer,N,D,loss,tokens_per_param,n_ratio,compute_saved",...B(t.flops,t.selected).map(a=>`${a.id},${a.N},${a.D},${a.loss},${a.tokensPerParam},${a.nRatio},${a.savings}`)];W("optiscale-compare.csv",o.join(`
`))}function W(s,o){const a=new Blob([o],{type:"text/plain"}),e=document.createElement("a");e.href=URL.createObjectURL(a),e.download=s,e.click(),URL.revokeObjectURL(e.href)}function g(){Q(),t.tab==="practice"?tt():t.tab==="research"?at():nt()}g();
