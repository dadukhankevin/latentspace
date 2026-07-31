"""The shared design system for every latentspace page — the hub and
each run's dashboard wear the same skin: dark glass, one accent
family, live JSON polling (no full-page refresh), labeled multi-series
fitness charts with hover readout. Self-contained: inline CSS/JS, no
external assets, served by stdlib HTTP."""

CSS = """
:root{
  --bg:#0a0d12; --panel:#10151d; --panel2:#0d1219; --line:#1c2431;
  --ink:#dbe4ee; --ink2:#8494a8; --ink3:#5a6a7e;
  --accent:#5cc8ff; --good:#7de3a0; --warn:#ffb454; --bad:#ff6b6b;
  --glow:0 0 18px rgba(92,200,255,.25);
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:
  radial-gradient(1200px 500px at 70% -10%, rgba(92,200,255,.06), transparent),
  var(--bg);
  color:var(--ink);
  font:14px/1.45 ui-monospace,'SF Mono',Menlo,monospace;
  padding:22px 26px;min-height:100vh}
a{color:var(--accent);text-decoration:none}
h1{font-size:15px;letter-spacing:.06em;font-weight:600}
.small{font-size:11px;color:var(--ink2)}
.hdr{display:flex;align-items:baseline;gap:14px;margin-bottom:16px;
  border-bottom:1px solid var(--line);padding-bottom:12px}
.hdr .sub{color:var(--ink2);font-size:12px}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:11px;
  padding:2px 10px;border-radius:999px;border:1px solid var(--line);
  color:var(--ink2)}
.pill.live{color:var(--good);border-color:rgba(125,227,160,.35)}
.pill.live .dot{width:7px;height:7px;border-radius:50%;
  background:var(--good);box-shadow:0 0 8px var(--good);
  animation:pulse 1.6s ease-in-out infinite}
.pill .dot{width:7px;height:7px;border-radius:50%;background:var(--ink3)}
@keyframes pulse{50%{opacity:.35}}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.tile{background:linear-gradient(180deg,var(--panel),var(--panel2));
  border:1px solid var(--line);border-radius:10px;padding:10px 16px;
  min-width:130px}
.tile .v{font-size:20px;font-weight:600;letter-spacing:.02em}
.tile .k{font-size:10px;color:var(--ink2);text-transform:uppercase;
  letter-spacing:.12em;margin-top:2px}
.tile.hot .v{color:var(--accent);text-shadow:var(--glow)}
.panel{background:linear-gradient(180deg,var(--panel),var(--panel2));
  border:1px solid var(--line);border-radius:12px;padding:14px 16px;
  margin:12px 0}
.panel h2{font-size:11px;color:var(--ink2);text-transform:uppercase;
  letter-spacing:.14em;margin-bottom:10px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th{font-size:10px;color:var(--ink3);text-transform:uppercase;
  letter-spacing:.1em;text-align:left;padding:4px 10px 6px 0;
  border-bottom:1px solid var(--line)}
td{padding:5px 10px 5px 0;border-bottom:1px solid
  rgba(28,36,49,.55);vertical-align:top}
tr:hover td{background:rgba(92,200,255,.03)}
.mono2{color:var(--ink2)} .dim{color:var(--ink3)}
.scorebar{position:relative;display:inline-block;width:90px;height:6px;
  background:#151c26;border-radius:3px;overflow:hidden;margin-right:8px;
  vertical-align:middle}
.scorebar i{position:absolute;left:0;top:0;bottom:0;
  background:linear-gradient(90deg,#2c6f9e,var(--accent));
  border-radius:3px}
.badge{font-size:10px;padding:1px 7px;border-radius:5px;
  border:1px solid var(--line);color:var(--ink2)}
.badge.found{color:var(--good);border-color:rgba(125,227,160,.3)}
.badge.refound{color:var(--warn);border-color:rgba(255,180,84,.3)}
.badge.bad{color:var(--bad);border-color:rgba(255,107,107,.3)}
.events{max-height:260px;overflow-y:auto;font-size:12px}
.events div{padding:2.5px 0;border-bottom:1px solid rgba(28,36,49,.4)}
.events .t{color:var(--ink3);margin-right:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));
  gap:14px}
.card{background:linear-gradient(180deg,var(--panel),var(--panel2));
  border:1px solid var(--line);border-radius:12px;padding:12px 14px;
  transition:border-color .2s}
.card:hover{border-color:rgba(92,200,255,.4)}
.card .name{font-weight:600;font-size:13.5px}
.card .meta{font-size:11px;color:var(--ink2);margin:3px 0 8px}
.chartwrap{position:relative}
.tip{position:absolute;pointer-events:none;background:#0d1420ee;
  border:1px solid var(--line);border-radius:8px;padding:6px 10px;
  font-size:11px;display:none;z-index:9;white-space:nowrap}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:#1c2431;border-radius:4px}
"""

CHART_JS = """
const PALETTE=['#5cc8ff','#ffb454','#8be98b','#d2a8ff','#7ee7e7','#ffd27a'];
function esc(s){return String(s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function fmt(v){if(v===null||v===undefined)return '—';
  const a=Math.abs(v);return a!==0&&(a<1e-3||a>=1e5)?
  v.toExponential(3):(+v.toFixed(5)).toString();}
function drawChart(el,series,points,opts){
  opts=opts||{}; const W=opts.w||el.clientWidth||700,H=opts.h||240;
  const mini=!!opts.mini, ML=mini?6:74, MR=mini?6:14,
        MT=mini?6:10, MB=mini?6:24;
  const names=Object.keys(series).filter(k=>series[k].length>=2).sort();
  if(!names.length){el.innerHTML=
    '<div class="dim" style="padding:30px 10px">awaiting data…</div>';return;}
  let xs=[],ys=[];
  names.forEach(n=>series[n].forEach(p=>{xs.push(p[0]);ys.push(p[1]);}));
  (points||[]).forEach(p=>ys.push(p[1]));
  const xmax=Math.max(...xs)*1.04+1e-9;
  let lo=Math.min(...ys),hi=Math.max(...ys);
  const pad=Math.max((hi-lo)*.14,1e-9); lo-=pad; hi+=pad;
  const X=x=>ML+(W-ML-MR)*x/xmax, Y=y=>MT+(H-MT-MB)*(hi-y)/(hi-lo);
  let s='<svg width="'+W+'" height="'+H+'">';
  s+='<defs><filter id="gl"><feDropShadow dx="0" dy="0" '+
     'stdDeviation="2.2" flood-color="#5cc8ff" flood-opacity="0.5"/>'+
     '</filter></defs>';
  if(!mini){for(let i=0;i<=3;i++){const gy=lo+pad+(hi-lo-2*pad)*i/3;
    s+='<line x1="'+ML+'" y1="'+Y(gy)+'" x2="'+(W-MR)+'" y2="'+Y(gy)+
       '" stroke="#1c2431" stroke-width="1"/>'+
       '<text x="'+(ML-8)+'" y="'+(Y(gy)+3)+'" font-size="10" '+
       'fill="#5a6a7e" text-anchor="end">'+fmt(gy)+'</text>';}}
  names.forEach((n,i)=>{
    const c=PALETTE[i%PALETTE.length],cv=series[n];
    let d='M '+X(cv[0][0])+' '+Y(cv[0][1]);
    for(let j=1;j<cv.length;j++)
      d+=' L '+X(cv[j][0])+' '+Y(cv[j-1][1])+' L '+X(cv[j][0])+' '+Y(cv[j][1]);
    if(names.length===1&&!mini){
      s+='<path d="'+d+' L '+X(cv[cv.length-1][0])+' '+Y(lo)+' L '+
         X(cv[0][0])+' '+Y(lo)+' Z" fill="'+c+'" opacity="0.06"/>';}
    s+='<path d="'+d+'" fill="none" stroke="'+c+'" stroke-width="'+
       (mini?1.6:2.2)+'"'+(mini?'':' filter="url(#gl)"')+'/>';
    if(!mini){const last=cv[cv.length-1];
      s+='<circle cx="'+X(last[0])+'" cy="'+Y(last[1])+'" r="3.5" fill="'+
         c+'"/>'+'<text x="'+(X(last[0])-8)+'" y="'+(Y(last[1])-9)+
         '" font-size="11" fill="'+c+'" text-anchor="end">'+esc(n)+' '+
         fmt(last[1])+'</text>';}});
  if(!mini)(points||[]).forEach(p=>{
    s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="2.6" fill="#3a4a5e"/>';});
  if(!mini)s+='<text x="'+ML+'" y="'+(H-6)+'" font-size="10" '+
    'fill="#5a6a7e">'+esc(opts.xlabel||'evaluations')+
    ' → &nbsp;·&nbsp; best so far ↑</text>';
  s+='</svg>';
  el.innerHTML=s;
  if(mini)return;
  const tip=el.parentElement.querySelector('.tip'); if(!tip)return;
  el.onmousemove=e=>{
    const r=el.getBoundingClientRect(),mx=e.clientX-r.left;
    const tx=(mx-ML)/(W-ML-MR)*xmax; let best=null;
    names.forEach((n,i)=>{const cv=series[n];
      let p=cv[0]; for(const q of cv){if(q[0]<=tx)p=q; else break;}
      if(!best||Math.abs(p[0]-tx)<Math.abs(best.p[0]-tx))
        best={n:n,p:p,c:PALETTE[i%PALETTE.length]};});
    if(!best){tip.style.display='none';return;}
    tip.style.display='block';
    tip.style.left=Math.min(mx+14,W-160)+'px';
    tip.style.top='14px';
    tip.innerHTML='<span style="color:'+best.c+'">●</span> '+esc(best.n)+
      '<br>x '+best.p[0]+' · '+fmt(best.p[1]);};
  el.onmouseleave=()=>{tip.style.display='none';};
}
"""


def page(title, body, boot_js):
    """Full HTML page: shared skin + a boot script that polls JSON and
    re-renders. body holds the static shells the JS fills."""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
{body}
<script>{CHART_JS}
{boot_js}</script></body></html>"""
