"""
sv_explorer.py
Generates a self-contained interactive HTML file where you can drag a vertical
frequency cursor on the SV spectrum and see the mode shape update live on the right.

Usage:
    python sv_explorer.py
    python sv_explorer.py --clean-csv "Harness Bridge Data Clean.csv" --out sv_explorer.html
"""
import pandas as pd
import numpy as np
from scipy.signal import csd
import json
import os
import argparse


def compute_sv_data(csv_path, nperseg=4096, fs=128.0):
    print("Loading sensor data...")
    df = pd.read_csv(csv_path, skiprows=25)
    sensor_cols = [col for col in df.columns if col not in ['Time', 'ParsedTime']]
    df = df.dropna(subset=sensor_cols)
    n_channels = len(sensor_cols)

    print(f"Computing cross-spectral density matrix ({n_channels}x{n_channels})...")
    f, _ = csd(df[sensor_cols[0]].values, df[sensor_cols[0]].values, fs=fs, nperseg=nperseg)
    n_freqs = len(f)

    G = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for i in range(n_channels):
        for j in range(n_channels):
            _, temp = csd(df[sensor_cols[i]].values, df[sensor_cols[j]].values, fs=fs, nperseg=nperseg)
            G[:, i, j] = temp

    print(f"Computing SVD at {n_freqs} frequency bins...")
    sv = np.zeros((n_freqs, n_channels))
    shapes = np.zeros((n_freqs, n_channels))

    for k in range(n_freqs):
        if np.any(np.isnan(G[k])) or np.any(np.isinf(G[k])):
            continue
        try:
            U, S, VH = np.linalg.svd(G[k])
            sv[k] = S
            u = U[:, 0]
            max_idx = np.argmax(np.abs(u))
            u_rot = u * np.exp(-1j * np.angle(u[max_idx]))
            real_u = np.real(u_rot)
            norm = np.max(np.abs(real_u))
            shapes[k] = real_u / norm if norm > 0 else real_u
        except np.linalg.LinAlgError:
            continue

    return f, sv, shapes, n_channels


def generate_explorer(csv_path, out_path, zero_joints=None, fixed_joints=None, nperseg=4096):
    if zero_joints is None:
        zero_joints = [4, 8]
    if fixed_joints is None:
        fixed_joints = [1, 11]

    f, sv, shapes, n_channels = compute_sv_data(csv_path, nperseg=nperseg)
    fs = 128.0

    sensor_joints = [2, 3, 5, 6, 7, 9, 10]
    spacings = [1.82, 1.82, 1.82, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25, 1.25]
    jp = [0.0]
    for s in spacings:
        jp.append(round(jp[-1] + s, 4))

    all_supports = sorted(set(zero_joints + fixed_joints))

    # Knot table: each joint has position, isSupport flag, sensorIdx (or null)
    knots = []
    for j in range(1, 12):
        pos = jp[j - 1]
        is_sup = j in all_supports
        s_idx = sensor_joints.index(j) if j in sensor_joints else None
        knots.append({'j': j, 'pos': pos, 'isSupport': is_sup, 'sensorIdx': s_idx})

    n_sv = min(3, n_channels)
    payload = {
        'freqs': [round(float(x), 4) for x in f],
        'sv': [[float(sv[k, i]) for i in range(n_sv)] for k in range(len(f))],
        'shapes': [[round(float(v), 5) for v in shapes[k]] for k in range(len(f))],
        'knots': knots,
        'fs': fs,
        'totalSpan': jp[-1],
        'sensorJoints': sensor_joints,
        'nSV': n_sv,
    }

    data_json = json.dumps(payload, separators=(',', ':'))
    html = build_html(data_json)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f"\nInteractive explorer saved to: {out_path}")
    print("  Open this file in your browser to explore mode shapes!")


def build_html(data_json):
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Harness Bridge — SV Mode Shape Explorer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{
  background:#080c14;color:#e2e8f0;font-family:'Inter',sans-serif;
  height:100vh;display:flex;flex-direction:column;overflow:hidden;
  user-select:none;
}
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 24px;
  background:linear-gradient(180deg,rgba(99,102,241,.12) 0%,rgba(0,0,0,0) 100%);
  border-bottom:1px solid rgba(255,255,255,.07);
  flex-shrink:0;
}
.title{
  font-size:16px;font-weight:700;letter-spacing:-.3px;
  background:linear-gradient(135deg,#60a5fa,#a78bfa 60%,#f472b6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.freq-badge{
  display:flex;flex-direction:column;align-items:center;
  background:rgba(249,115,22,.1);border:1px solid rgba(249,115,22,.3);
  border-radius:10px;padding:6px 18px;
}
.freq-input{
  font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:600;
  color:#fb923c;line-height:1;text-shadow:0 0 24px rgba(251,146,60,.5);
  background:transparent;border:none;outline:none;text-align:center;
  width:140px;
}
.freq-input::-webkit-outer-spin-button,
.freq-input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.freq-input[type=number] { -moz-appearance: textfield; }
.freq-label{font-size:10px;color:#78716c;text-transform:uppercase;letter-spacing:1px;margin-top:2px;}
.hint{font-size:12px;color:#475569;text-align:right;max-width:160px;}

.main{display:flex;flex:1;overflow:hidden;}

/* ─── SV Panel ─── */
.sv-panel{
  flex:0 0 60%;display:flex;flex-direction:column;
  padding:16px 20px;
  border-right:1px solid rgba(255,255,255,.06);
}
.panel-label{
  font-size:10px;font-weight:600;letter-spacing:2px;
  text-transform:uppercase;color:#64748b;margin-bottom:10px;
}
.canvas-wrap{
  flex:1;position:relative;cursor:crosshair;
  border-radius:10px;overflow:hidden;
  background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.05);
}
canvas{position:absolute;top:0;left:0;width:100%;height:100%;}

.sv-legend{
  display:flex;gap:18px;margin-top:10px;
}
.leg-item{display:flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8;}
.leg-swatch{width:22px;height:3px;border-radius:2px;}

/* ─── Mode Shape Panel ─── */
.mode-panel{
  flex:0 0 40%;display:flex;flex-direction:column;
  padding:16px 20px;
}
.mode-stats{
  display:flex;gap:10px;margin-bottom:14px;flex-shrink:0;
}
.stat-card{
  flex:1;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
  border-radius:8px;padding:8px 12px;
}
.stat-card .s-label{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.8px;}
.stat-card .s-val{
  font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:500;
  color:#e2e8f0;margin-top:2px;
}
.mode-nearest{
  background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);
  border-radius:8px;padding:8px 12px;margin-bottom:14px;
  font-size:12px;color:#a5b4fc;flex-shrink:0;
}
svg#bridge-svg{
  flex:1;width:100%;
  border-radius:10px;
  background:rgba(255,255,255,.02);
  border:1px solid rgba(255,255,255,.05);
}
</style>
</head>
<body>
<header>
  <div class="title">🌉 Harness Bridge · SV Mode Shape Explorer</div>
  <div class="freq-badge">
    <input type="number" step="0.01" min="0" class="freq-input" id="freq-input" value="0.00">
    <div class="freq-label">Frequency (Hz)</div>
  </div>
  <div class="hint">Drag/click SV plot or use<br>Left/Right arrows</div>
</header>

<div class="main">
  <div class="sv-panel">
    <div class="panel-label">Singular Value Spectrum · Drag cursor to explore</div>
    <div class="canvas-wrap" id="canvas-wrap">
      <canvas id="sv-canvas"></canvas>
    </div>
    <div class="sv-legend">
      <div class="leg-item"><div class="leg-swatch" style="background:#38bdf8"></div>SV 1</div>
      <div class="leg-item"><div class="leg-swatch" style="background:#34d399"></div>SV 2</div>
      <div class="leg-item"><div class="leg-swatch" style="background:#a78bfa"></div>SV 3</div>
      <div class="leg-item">
        <div class="leg-swatch" style="background:#f97316;height:1px;border-top:2px dashed #f97316"></div>
        Cursor
      </div>
    </div>
  </div>

  <div class="mode-panel">
    <div class="panel-label">Mode Shape at Cursor Frequency</div>
    <div class="mode-stats">
      <div class="stat-card">
        <div class="s-label">SV₁ Amplitude</div>
        <div class="s-val" id="sv1-val">—</div>
      </div>
      <div class="stat-card">
        <div class="s-label">SV₂/SV₁ Ratio</div>
        <div class="s-val" id="sv-ratio">—</div>
      </div>
      <div class="stat-card">
        <div class="s-label">SV₃/SV₁ Ratio</div>
        <div class="s-val" id="sv3-ratio">—</div>
      </div>
    </div>
    <div class="mode-nearest" id="nearest-info">Move cursor over the SV plot to begin</div>
    <svg id="bridge-svg" viewBox="0 0 600 260"></svg>
  </div>
</div>

<script>
/** =========================================================
 *  DATA (embedded by Python)
 * ========================================================= */
const DATA = """ + data_json + """;

/** =========================================================
 *  Cubic spline (natural BC)
 * ========================================================= */
class CubicSpline {
  constructor(xs, ys) {
    this.xs = xs; this.ys = ys;
    const n = xs.length;
    const h=[], alpha=[], l=new Array(n).fill(0),
          mu=new Array(n).fill(0), z=new Array(n).fill(0);
    this.c=new Array(n).fill(0);
    this.b=new Array(n-1).fill(0);
    this.d=new Array(n-1).fill(0);
    for(let i=0;i<n-1;i++) h[i]=xs[i+1]-xs[i];
    for(let i=1;i<n-1;i++)
      alpha[i]=3/h[i]*(ys[i+1]-ys[i])-3/h[i-1]*(ys[i]-ys[i-1]);
    l[0]=1; mu[0]=0; z[0]=0;
    for(let i=1;i<n-1;i++){
      l[i]=2*(xs[i+1]-xs[i-1])-h[i-1]*mu[i-1];
      mu[i]=h[i]/l[i];
      z[i]=(alpha[i]-h[i-1]*z[i-1])/l[i];
    }
    l[n-1]=1; z[n-1]=0; this.c[n-1]=0;
    for(let j=n-2;j>=0;j--){
      this.c[j]=z[j]-mu[j]*this.c[j+1];
      this.b[j]=(ys[j+1]-ys[j])/h[j]-h[j]*(this.c[j+1]+2*this.c[j])/3;
      this.d[j]=(this.c[j+1]-this.c[j])/(3*h[j]);
    }
  }
  eval(x){
    let i=0;
    for(let j=0;j<this.xs.length-1;j++) if(x>=this.xs[j]) i=j;
    i=Math.min(i,this.xs.length-2);
    const dx=x-this.xs[i];
    return this.ys[i]+this.b[i]*dx+this.c[i]*dx*dx+this.d[i]*dx*dx*dx;
  }
  evalDense(xmin,xmax,n){
    const pts=[];
    for(let k=0;k<n;k++){
      const x=xmin+(xmax-xmin)*k/(n-1);
      pts.push({x, y:this.eval(x)});
    }
    return pts;
  }
}

/** =========================================================
 *  State
 * ========================================================= */
const state = {
  cursorFreq: DATA.freqs[Math.floor(DATA.freqs.length/4)],
  dragging: false,
  cursorIdx: 0,
};

/** =========================================================
 *  Canvas SV Plot
 * ========================================================= */
const wrap  = document.getElementById('canvas-wrap');
const canvas = document.getElementById('sv-canvas');
const ctx    = canvas.getContext('2d');

const PAD = { l:52, r:16, t:12, b:36 };
const SV_COLORS = ['#38bdf8','#34d399','#a78bfa'];

let W=0, H=0;

function resize(){
  const r = wrap.getBoundingClientRect();
  canvas.width  = r.width  * devicePixelRatio;
  canvas.height = r.height * devicePixelRatio;
  W = canvas.width; H = canvas.height;
  drawSV();
}

function freqToX(f){
  const span = DATA.fs/2;
  return PAD.l*devicePixelRatio + (f/span)*(W - (PAD.l+PAD.r)*devicePixelRatio);
}
function xToFreq(px){
  const span = DATA.fs/2;
  return (px - PAD.l*devicePixelRatio) / (W-(PAD.l+PAD.r)*devicePixelRatio) * span;
}

// Log scale Y: we pre-compute min/max of log10(sv)
function computeLogBounds(){
  let mn=Infinity, mx=-Infinity;
  for(const row of DATA.sv){
    for(let i=0;i<DATA.nSV;i++){
      const v=row[i];
      if(v>0){ const lv=Math.log10(v); if(lv<mn)mn=lv; if(lv>mx)mx=lv; }
    }
  }
  // nice round bounds
  mn = Math.floor(mn) - 0.5;
  mx = Math.ceil(mx)  + 0.5;
  return {mn, mx};
}
const {mn:LOG_MIN, mx:LOG_MAX} = computeLogBounds();

function logToY(v){
  if(v<=0) return H - PAD.b*devicePixelRatio;
  const lv = Math.log10(v);
  const t  = (lv - LOG_MIN)/(LOG_MAX - LOG_MIN);
  return H - PAD.b*devicePixelRatio - t*(H-(PAD.t+PAD.b)*devicePixelRatio);
}

function drawSV(){
  ctx.clearRect(0,0,W,H);
  const dpr = devicePixelRatio;

  // Background grid
  ctx.strokeStyle='rgba(255,255,255,.06)';
  ctx.lineWidth=1;
  // horizontal log grid lines
  for(let dec=Math.ceil(LOG_MIN); dec<=Math.floor(LOG_MAX); dec++){
    const y = logToY(Math.pow(10,dec));
    ctx.beginPath(); ctx.moveTo(PAD.l*dpr,y); ctx.lineTo(W-PAD.r*dpr,y); ctx.stroke();
    ctx.fillStyle='#475569'; ctx.font=`${9*dpr}px Inter`;
    ctx.textAlign='right';
    ctx.fillText('10'+superscript(dec), (PAD.l-4)*dpr, y+3*dpr);
  }
  // Vertical frequency grid lines (every 10 Hz)
  for(let fv=0;fv<=DATA.fs/2;fv+=10){
    const x=freqToX(fv);
    ctx.beginPath(); ctx.moveTo(x,PAD.t*dpr); ctx.lineTo(x,H-PAD.b*dpr); ctx.stroke();
    if(fv>0){
      ctx.fillStyle='#475569'; ctx.font=`${9*dpr}px Inter`;
      ctx.textAlign='center';
      ctx.fillText(fv+'Hz', x, H-(PAD.b-12)*dpr);
    }
  }

  // Axes
  ctx.strokeStyle='rgba(255,255,255,.2)'; ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.moveTo(PAD.l*dpr, PAD.t*dpr);
  ctx.lineTo(PAD.l*dpr, H-PAD.b*dpr);
  ctx.lineTo(W-PAD.r*dpr, H-PAD.b*dpr);
  ctx.stroke();

  // SV curves
  for(let sv=0;sv<DATA.nSV;sv++){
    ctx.beginPath();
    ctx.strokeStyle=SV_COLORS[sv];
    ctx.lineWidth=(sv===0?2:1.2)*dpr;
    ctx.globalAlpha=sv===0?1:0.6;
    let first=true;
    for(let k=0;k<DATA.freqs.length;k++){
      const x=freqToX(DATA.freqs[k]);
      const y=logToY(DATA.sv[k][sv]);
      if(first){ctx.moveTo(x,y);first=false;}else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.globalAlpha=1;
  }

  // Cursor line
  const curX = freqToX(state.cursorFreq);
  ctx.setLineDash([6*dpr,4*dpr]);
  ctx.strokeStyle='#f97316';
  ctx.lineWidth=1.8*dpr;
  ctx.beginPath();
  ctx.moveTo(curX, PAD.t*dpr);
  ctx.lineTo(curX, H-PAD.b*dpr);
  ctx.stroke();
  ctx.setLineDash([]);

  // Cursor freq label
  ctx.fillStyle='#f97316';
  ctx.font=`${10*dpr}px JetBrains Mono`;
  ctx.textAlign='left';
  const labelX = curX+4*dpr > W-60*dpr ? curX-50*dpr : curX+4*dpr;
  ctx.fillText(state.cursorFreq.toFixed(2)+'Hz', labelX, PAD.t*dpr+14*dpr);

  // SV₁ dot on cursor
  const curSV1 = getCurrentSV1();
  ctx.fillStyle='#f97316';
  ctx.beginPath();
  ctx.arc(curX, logToY(curSV1), 4*dpr, 0, Math.PI*2);
  ctx.fill();
}

function superscript(n){
  const map={'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵',
             '6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
  return String(n).split('').map(c=>map[c]||c).join('');
}

function getCurrentSV1(){
  const idx = getNearestFreqIdx(state.cursorFreq);
  return DATA.sv[idx][0];
}

function getNearestFreqIdx(freq){
  let best=0, bestDist=Infinity;
  for(let k=0;k<DATA.freqs.length;k++){
    const d=Math.abs(DATA.freqs[k]-freq);
    if(d<bestDist){bestDist=d;best=k;}
  }
  return best;
}

// Mouse events
function canvasMouseEvent(e){
  const r = canvas.getBoundingClientRect();
  const px = (e.clientX - r.left) * devicePixelRatio;
  const freq = Math.max(0, Math.min(DATA.fs/2, xToFreq(px)));
  state.cursorFreq = freq;
  updateDisplay();
  drawSV();
}

canvas.addEventListener('mousedown', e=>{state.dragging=true; canvasMouseEvent(e);});
canvas.addEventListener('mousemove', e=>{if(state.dragging||true) canvasMouseEvent(e);});
canvas.addEventListener('mouseup',   ()=>state.dragging=false);
canvas.addEventListener('mouseleave',()=>state.dragging=false);

/** =========================================================
 *  Bridge Mode Shape SVG
 * ========================================================= */
const bridgeSVG = document.getElementById('bridge-svg');
const VW=600, VH=260;
const BX=40, BY=130, BW=520;  // bridge baseline in SVG coords

function spanToSVGx(span){
  return BX + (span / DATA.totalSpan) * BW;
}
function ampToSVGy(amp){
  // amp in [-1,1] → SVG y. Positive amp goes UP (smaller y)
  const scale=80;
  return BY - amp*scale;
}

function buildModeShapePath(pts){
  if(pts.length===0) return '';
  let d=`M ${pts[0].svgx.toFixed(1)} ${pts[0].svgy.toFixed(1)}`;
  for(let i=1;i<pts.length;i++)
    d+=` L ${pts[i].svgx.toFixed(1)} ${pts[i].svgy.toFixed(1)}`;
  return d;
}

function renderBridgeSVG(freqIdx){
  const shape = DATA.shapes[freqIdx];

  // Build knot arrays for cubic spline
  const xs=[], ys=[];
  for(const k of DATA.knots){
    if(k.isSupport || k.sensorIdx !== null){
      xs.push(k.pos);
      if(k.isSupport){
        ys.push(0);
      } else {
        ys.push(shape[k.sensorIdx]);
      }
    }
  }

  // Dense spline
  const spline = new CubicSpline(xs, ys);
  const N=300;
  const dense=[];
  for(let i=0;i<N;i++){
    const span = DATA.totalSpan*i/(N-1);
    dense.push({span, amp:spline.eval(span)});
  }

  const svgPts = dense.map(p=>({
    svgx:spanToSVGx(p.span),
    svgy:ampToSVGy(p.amp)
  }));

  // Build SVG content
  let html='';

  // Gradient defs
  html+=`<defs>
    <linearGradient id="shapeGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#60a5fa" stop-opacity="0.35"/>
      <stop offset="50%" stop-color="#60a5fa" stop-opacity="0.05"/>
      <stop offset="100%" stop-color="#60a5fa" stop-opacity="0.35"/>
    </linearGradient>
    <linearGradient id="shapeGradNeg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f472b6" stop-opacity="0.05"/>
      <stop offset="100%" stop-color="#f472b6" stop-opacity="0.3"/>
    </linearGradient>
  </defs>`;

  // Bridge deck baseline
  html+=`<line x1="${BX}" y1="${BY}" x2="${BX+BW}" y2="${BY}"
    stroke="rgba(255,255,255,0.15)" stroke-width="1.5" stroke-dasharray="4,4"/>`;

  // Zero line label
  html+=`<text x="${BX-6}" y="${BY+4}" font-size="9" fill="#475569"
    font-family="Inter" text-anchor="end">0</text>`;

  // Mode shape fill (area from path back to baseline)
  const closedPath = buildModeShapePath(svgPts)
    + ` L ${svgPts[svgPts.length-1].svgx.toFixed(1)} ${BY}`
    + ` L ${svgPts[0].svgx.toFixed(1)} ${BY} Z`;
  html+=`<path d="${closedPath}" fill="url(#shapeGrad)" opacity="0.8"/>`;

  // Mode shape curve
  html+=`<path d="${buildModeShapePath(svgPts)}"
    fill="none" stroke="#60a5fa" stroke-width="2.5"
    stroke-linejoin="round" stroke-linecap="round"/>`;

  // Support markers
  for(const k of DATA.knots){
    if(!k.isSupport) continue;
    const sx=spanToSVGx(k.pos);
    const isFixed = [1,11].includes(k.j);  // simplified - could pass fixed vs zero
    if(isFixed){
      // Triangle (fixed)
      const ts=10;
      html+=`<polygon points="${sx},${BY} ${sx-ts},${BY+ts*1.5} ${sx+ts},${BY+ts*1.5}"
        fill="#94a3b8" opacity="0.9"/>`;
      html+=`<line x1="${sx-ts}" y1="${BY+ts*1.5}" x2="${sx+ts}" y2="${BY+ts*1.5}"
        stroke="#94a3b8" stroke-width="2"/>`;
    } else {
      // Diamond (zero/roller)
      const ts=7;
      html+=`<polygon points="${sx},${BY-ts} ${sx+ts},${BY} ${sx},${BY+ts} ${sx-ts},${BY}"
        fill="#64748b" opacity="0.9"/>`;
    }
    html+=`<text x="${sx}" y="${BY+28}" font-size="9" fill="#64748b"
      font-family="Inter" text-anchor="middle">J${k.j}</text>`;
  }

  // Sensor dots and labels
  for(const k of DATA.knots){
    if(k.sensorIdx===null || k.isSupport) continue;
    const sx=spanToSVGx(k.pos);
    const amp=shape[k.sensorIdx];
    const sy=ampToSVGy(amp);
    // Dot
    html+=`<circle cx="${sx}" cy="${sy}" r="5"
      fill="#f87171" stroke="#fff" stroke-width="1.5" opacity="0.95"/>`;
    // Label
    const labelY = amp>=0 ? sy-12 : sy+18;
    html+=`<text x="${sx}" y="${labelY}" font-size="9" fill="#fca5a5"
      font-family="Inter" text-anchor="middle" font-weight="600">J${k.j}</text>`;
    const valY = amp>=0 ? sy-22 : sy+28;
    html+=`<text x="${sx}" y="${valY}" font-size="8" fill="#94a3b8"
      font-family="JetBrains Mono" text-anchor="middle">${amp.toFixed(3)}</text>`;
  }

  // Bridge joint numbers at baseline
  for(const k of DATA.knots){
    if(k.isSupport) continue;
    const sx=spanToSVGx(k.pos);
    html+=`<circle cx="${sx}" cy="${BY}" r="2.5" fill="rgba(255,255,255,0.2)"/>`;
  }

  // Title
  html+=`<text x="${BX+BW/2}" y="18" font-size="11" fill="#94a3b8"
    font-family="Inter" text-anchor="middle" font-weight="500">
    Mode Shape — Cubic Spline Interpolation
  </text>`;

  // Span label
  html+=`<text x="${BX+BW/2}" y="${VH-4}" font-size="9" fill="#475569"
    font-family="Inter" text-anchor="middle">
    Bridge Span (m) — Total: ${DATA.totalSpan.toFixed(2)} m
  </text>`;

  // Distance ticks
  for(const k of DATA.knots){
    const sx=spanToSVGx(k.pos);
    html+=`<line x1="${sx}" y1="${BY-3}" x2="${sx}" y2="${BY+3}"
      stroke="rgba(255,255,255,0.15)" stroke-width="1"/>`;
  }

  // Amplitude axis labels
  html+=`<text x="${BX-6}" y="${ampToSVGy(1)+4}" font-size="8" fill="#475569"
    font-family="Inter" text-anchor="end">+1</text>`;
  html+=`<text x="${BX-6}" y="${ampToSVGy(-1)+4}" font-size="8" fill="#475569"
    font-family="Inter" text-anchor="end">-1</text>`;

  bridgeSVG.innerHTML = html;
}

/** =========================================================
 *  Update info panels
 * ========================================================= */
function updateDisplay(){
  const freq=state.cursorFreq;
  const idx=getNearestFreqIdx(freq);
  const svRow=DATA.sv[idx];

  const freqInput = document.getElementById('freq-input');
  if (document.activeElement !== freqInput) {
    freqInput.value = freq.toFixed(2);
  }
  document.getElementById('sv1-val').textContent=svRow[0].toExponential(3);

  const ratio2 = DATA.nSV>=2 ? (svRow[1]/svRow[0]).toFixed(3) : '—';
  const ratio3 = DATA.nSV>=3 ? (svRow[2]/svRow[0]).toFixed(3) : '—';
  document.getElementById('sv-ratio').textContent=ratio2;
  document.getElementById('sv3-ratio').textContent=ratio3;

  // Nearest identified peak? (SV local max near cursor)
  document.getElementById('nearest-info').textContent=
    `Freq index: ${idx} / ${DATA.freqs.length-1}   |   Nearest freq bin: ${DATA.freqs[idx].toFixed(3)} Hz`;

  renderBridgeSVG(idx);
}

/** =========================================================
 *  Init
 * ========================================================= */
const freqInput = document.getElementById('freq-input');
freqInput.addEventListener('change', (e) => {
  let val = parseFloat(e.target.value);
  if (!isNaN(val)) {
    state.cursorFreq = Math.max(0, Math.min(DATA.fs/2, val));
    updateDisplay();
    drawSV();
  }
});
freqInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.target.blur();
  }
});

window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    const idx = getNearestFreqIdx(state.cursorFreq);
    let newIdx = idx;
    if (e.key === 'ArrowLeft') newIdx = Math.max(0, idx - 1);
    if (e.key === 'ArrowRight') newIdx = Math.min(DATA.freqs.length - 1, idx + 1);
    if (newIdx !== idx) {
      state.cursorFreq = DATA.freqs[newIdx];
      updateDisplay();
      drawSV();
      e.preventDefault();
    }
  }
});

window.addEventListener('resize', ()=>{ resize(); });
const ro = new ResizeObserver(resize);
ro.observe(wrap);

// Initial render
resize();
updateDisplay();
</script>
</body>
</html>
""".replace("/** =========================================================\n *  DATA (embedded by Python)\n * ========================================================= */\nconst DATA = ;", "const DATA = " + "PLACEHOLDER" + ";")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive SV Mode Shape Explorer HTML")
    parser.add_argument("--clean-csv", default="Harness Bridge Data Clean.csv")
    parser.add_argument("--zero-points", nargs="*", type=int, default=[4, 8])
    parser.add_argument("--fixed-points", nargs="*", type=int, default=[1, 11])
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--out", default="sv_explorer.html")
    args = parser.parse_args()

    if not os.path.exists(args.clean_csv):
        print(f"Error: '{args.clean_csv}' not found.")
        return

    generate_explorer(args.clean_csv, args.out, args.zero_points, args.fixed_points, args.nperseg)


if __name__ == "__main__":
    main()
