"""
windowed_sv_explorer.py
Generates an interactive HTML SV Mode Shape Explorer dividing a dataset into time windows.
"""

import pandas as pd
import numpy as np
from scipy.signal import csd
import json
import os
import argparse

def compute_sv_data_from_df(df_chunk, nperseg=4096, fs=128.0):
    sensor_cols = [col for col in df_chunk.columns if col not in ['Time', 'ParsedTime']]
    n_channels = len(sensor_cols)

    f, _ = csd(df_chunk[sensor_cols[0]].values, df_chunk[sensor_cols[0]].values, fs=fs, nperseg=nperseg)
    n_freqs = len(f)

    G = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for i in range(n_channels):
        for j in range(n_channels):
            _, temp = csd(df_chunk[sensor_cols[i]].values, df_chunk[sensor_cols[j]].values, fs=fs, nperseg=nperseg)
            G[:, i, j] = temp

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

def generate_windowed_explorer(csv_path, out_path, window_sec, zero_joints=None, fixed_joints=None, nperseg=4096):
    if zero_joints is None:
        zero_joints = [4, 8]
    if fixed_joints is None:
        fixed_joints = [1, 11]

    fs = 128.0
    chunk_size = int(window_sec * fs)

    print(f"Loading sensor data from {csv_path}...")
    df = pd.read_csv(csv_path, skiprows=25)
    sensor_cols = [col for col in df.columns if col not in ['Time', 'ParsedTime']]
    
    datasets = []
    global_f = None
    
    total_rows = len(df)
    n_chunks = total_rows // chunk_size
    if total_rows % chunk_size > 0:
        n_chunks += 1
        
    print(f"Dividing into {n_chunks} windows of ~{window_sec} seconds ({chunk_size} samples each).")
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        df_chunk = df.iloc[start_idx:end_idx]
        
        if len(df_chunk) < nperseg:
            print(f"Skipping window {i+1} as it has fewer samples ({len(df_chunk)}) than nperseg ({nperseg}).")
            continue
            
        start_sec = start_idx / fs
        end_sec = end_idx / fs
        name = f"Window {i+1} ({start_sec:.0f}s - {end_sec:.0f}s)"
        print(f"Computing SV for {name} ({len(df_chunk)} samples)...")
        
        f, sv, shapes, n_channels = compute_sv_data_from_df(df_chunk, nperseg=nperseg, fs=fs)
        if global_f is None:
            global_f = f
            
        n_sv = min(3, n_channels)
        datasets.append({
            'id': i,
            'name': name,
            'sv': [[float(sv[k, j]) for j in range(n_sv)] for k in range(len(f))],
            'shapes': [[round(float(v), 5) for v in shapes[k]] for k in range(len(f))],
            'nSV': n_sv
        })

    sensor_joints = [2, 3, 5, 6, 7, 9, 10]
    spacings = [1.82, 1.82, 1.82, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25, 1.25]
    jp = [0.0]
    for s in spacings:
        jp.append(round(jp[-1] + s, 4))

    all_supports = sorted(set(zero_joints + fixed_joints))

    knots = []
    for j in range(1, 12):
        pos = jp[j - 1]
        is_sup = j in all_supports
        s_idx = sensor_joints.index(j) if j in sensor_joints else None
        knots.append({'j': j, 'pos': pos, 'isSupport': is_sup, 'sensorIdx': s_idx})

    payload = {
        'freqs': [round(float(x), 4) for x in global_f],
        'datasets': datasets,
        'knots': knots,
        'fs': fs,
        'totalSpan': jp[-1],
        'sensorJoints': sensor_joints,
    }

    data_json = json.dumps(payload, separators=(',', ':'))
    html = build_html(data_json)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f"\nInteractive windowed explorer saved to: {out_path}")
    print("  Open this file in your browser to explore mode shapes!")

def build_html(data_json):
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Harness Bridge — Windowed SV Explorer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #080c14; color: #e2e8f0; font-family: 'Inter', sans-serif;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
  user-select: none;
}
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 24px;
  background: linear-gradient(180deg, rgba(99,102,241,.12) 0%, rgba(0,0,0,0) 100%);
  border-bottom: 1px solid rgba(255,255,255,.07);
  flex-shrink: 0;
}
.title {
  font-size: 16px; font-weight: 700; letter-spacing: -.3px;
  background: linear-gradient(135deg, #60a5fa, #a78bfa 60%, #f472b6);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.freq-badge {
  display: flex; flex-direction: column; align-items: center;
  background: rgba(249,115,22,.1); border: 1px solid rgba(249,115,22,.3);
  border-radius: 10px; padding: 6px 18px;
}
.freq-input {
  font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 600;
  color: #fb923c; line-height: 1; text-shadow: 0 0 24px rgba(251,146,60,.5);
  background: transparent; border: none; outline: none; text-align: center;
  width: 140px;
}
.freq-input::-webkit-outer-spin-button,
.freq-input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.freq-input[type=number] { -moz-appearance: textfield; }
.freq-label { font-size: 10px; color: #78716c; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }
.hint { font-size: 12px; color: #475569; text-align: right; max-width: 160px; }

.main { display: flex; flex: 1; overflow: hidden; }

/* SV Panel */
.sv-panel {
  flex: 0 0 50%; display: flex; flex-direction: column;
  padding: 16px 20px;
  border-right: 1px solid rgba(255,255,255,.06);
}
.panel-label {
  font-size: 10px; font-weight: 600; letter-spacing: 2px;
  text-transform: uppercase; color: #64748b; margin-bottom: 10px;
}
.canvas-wrap {
  flex: 1; position: relative; cursor: crosshair;
  border-radius: 10px; overflow: hidden;
  background: rgba(255,255,255,.02);
  border: 1px solid rgba(255,255,255,.05);
}
canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

.dataset-toggles {
  display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;
}

/* Mode Shape Panel */
.mode-panel {
  flex: 0 0 50%; display: flex; flex-direction: column;
  padding: 16px 20px; overflow-y: auto;
}
.mode-stats {
  display: flex; gap: 10px; margin-bottom: 14px; flex-shrink: 0;
}
.stat-card {
  flex: 1; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  border-radius: 8px; padding: 8px 12px;
}
.stat-card .s-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: .8px; }
.stat-card .s-val {
  font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 500;
  color: #e2e8f0; margin-top: 2px;
}
.mode-nearest {
  background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.2);
  border-radius: 8px; padding: 8px 12px; margin-bottom: 14px;
  font-size: 12px; color: #a5b4fc; flex-shrink: 0;
}

/* Scrollbar styles for right panel */
.mode-panel::-webkit-scrollbar { width: 8px; }
.mode-panel::-webkit-scrollbar-track { background: transparent; }
.mode-panel::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
.mode-panel::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
<script>
window.onerror = function(msg, url, lineNo, columnNo, error) {
  const errDiv = document.createElement('div');
  errDiv.style = "position:fixed; top:10px; right:10px; background:rgba(255,0,0,0.9); color:white; padding:10px; z-index:9999; max-width:400px; word-wrap:break-word;";
  errDiv.innerHTML = "<b>JS Error:</b> " + msg + "<br>Line: " + lineNo + " Col: " + columnNo + "<br><pre>" + (error && error.stack ? error.stack : "") + "</pre>";
  document.body.appendChild(errDiv);
  return false;
};

window.onerror = function(msg, url, lineNo, columnNo, error) {
  var errDiv = document.createElement('div');
  errDiv.style.position = 'fixed';
  errDiv.style.top = '0';
  errDiv.style.left = '0';
  errDiv.style.width = '100%';
  errDiv.style.background = 'red';
  errDiv.style.color = 'white';
  errDiv.style.padding = '20px';
  errDiv.style.zIndex = '999999';
  errDiv.style.fontFamily = 'monospace';
  errDiv.style.fontSize = '16px';
  errDiv.innerHTML = '<b>JS Error:</b> ' + msg + '<br>Line: ' + lineNo + '<br>Col: ' + columnNo;
  if(error && error.stack) errDiv.innerHTML += '<br><pre>' + error.stack + '</pre>';
  document.body.appendChild(errDiv);
  return false;
};
</script>
</head>
<body>
<header>
  <div class="title">🌉 Harness Bridge · Windowed SV Explorer</div>
  <div class="freq-badge">
    <input type="number" step="0.01" min="0" class="freq-input" id="freq-input" value="0.00">
    <div class="freq-label">Frequency (Hz)</div>
  </div>
  <div class="hint">Drag/click SV plot or use<br>Left/Right arrows</div>
</header>

<div class="main">
  <div class="sv-panel">
    <div class="panel-label">Singular Value Spectrum</div>
    <div class="dataset-toggles" id="dataset-toggles" style="margin-bottom: 12px; margin-top: 0;"></div>
    <div class="canvas-wrap" id="canvas-wrap">
      <canvas id="sv-canvas"></canvas>
    </div>
  </div>

  <div class="mode-panel">
    <div class="mode-nearest" id="nearest-info">Move cursor over the SV plot to begin</div>
    <div id="mode-panels"></div>
  </div>
</div>

<script>
window.onerror = function(msg, url, lineNo, columnNo, error) {
  const errDiv = document.createElement('div');
  errDiv.style = "position:fixed; top:10px; right:10px; background:rgba(255,0,0,0.9); color:white; padding:10px; z-index:9999; max-width:400px; word-wrap:break-word;";
  errDiv.innerHTML = "<b>JS Error:</b> " + msg + "<br>Line: " + lineNo + " Col: " + columnNo + "<br><pre>" + (error && error.stack ? error.stack : "") + "</pre>";
  document.body.appendChild(errDiv);
  return false;
};

const DATA = """ + data_json + """;

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

const COLORS = ['#38bdf8', '#34d399', '#a78bfa', '#f472b6', '#facc15', '#f87171', '#818cf8'];

const state = {
  cursorFreq: DATA.freqs[Math.floor(DATA.freqs.length/4)],
  dragging: false,
  activeDatasets: DATA.datasets.map(d => d.id) // all active initially
};

function initUI() {
  const togglesDiv = document.getElementById('dataset-toggles');
  DATA.datasets.forEach(ds => {
    const color = COLORS[ds.id % COLORS.length];
    togglesDiv.innerHTML += `
      <label style="color:${color}; cursor: pointer; display: flex; align-items: center; gap: 5px; font-size: 13px; background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px;">
        <input type="checkbox" checked value="${ds.id}" onchange="toggleDataset(${ds.id})"> 
        ${ds.name}
      </label>
    `;
  });
  buildModePanels();
}

window.toggleDataset = function(id) {
  const idx = state.activeDatasets.indexOf(id);
  if(idx > -1) state.activeDatasets.splice(idx, 1);
  else state.activeDatasets.push(id);
  state.activeDatasets.sort();
  buildModePanels();
  updateDisplay();
  drawSV();
}

function buildModePanels() {
  const container = document.getElementById('mode-panels');
  let html = '';
  state.activeDatasets.forEach(id => {
    const ds = DATA.datasets.find(d => d.id === id);
    const color = COLORS[id % COLORS.length];
    html += `
      <div class="dataset-mode-panel" style="margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; background: rgba(0,0,0,0.2);">
        <div style="color:${color}; font-weight:600; font-size: 14px; margin-bottom: 10px;">${ds.name}</div>
        <div class="mode-stats">
          <div class="stat-card">
            <div class="s-label">SV₁ Amplitude</div>
            <div class="s-val" id="sv1-val-${id}">—</div>
          </div>
          <div class="stat-card">
            <div class="s-label">SV₂/SV₁ Ratio</div>
            <div class="s-val" id="sv-ratio-${id}">—</div>
          </div>
          <div class="stat-card">
            <div class="s-label">SV₃/SV₁ Ratio</div>
            <div class="s-val" id="sv3-ratio-${id}">—</div>
          </div>
        </div>
        <svg id="bridge-svg-${id}" viewBox="0 0 600 260" style="width:100%; border-radius:10px; background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.05);"></svg>
      </div>
    `;
  });
  container.innerHTML = html;
}

const wrap  = document.getElementById('canvas-wrap');
const canvas = document.getElementById('sv-canvas');
const ctx    = canvas.getContext('2d');

const PAD = { l:52, r:16, t:12, b:36 };

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

function computeLogBounds(){
  let mn=Infinity, mx=-Infinity;
  for(const ds of DATA.datasets){
    for(const row of ds.sv){
      for(let i=0;i<ds.nSV;i++){
        const v=row[i];
        if(v>0){ const lv=Math.log10(v); if(lv<mn)mn=lv; if(lv>mx)mx=lv; }
      }
    }
  }
  if(mn === Infinity) { mn = -5; mx = 0; }
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
  for(let dec=Math.ceil(LOG_MIN); dec<=Math.floor(LOG_MAX); dec++){
    const y = logToY(Math.pow(10,dec));
    ctx.beginPath(); ctx.moveTo(PAD.l*dpr,y); ctx.lineTo(W-PAD.r*dpr,y); ctx.stroke();
    ctx.fillStyle='#475569'; ctx.font=`${9*dpr}px Inter`;
    ctx.textAlign='right';
    ctx.fillText('10'+superscript(dec), (PAD.l-4)*dpr, y+3*dpr);
  }
  for(let fv=0;fv<=DATA.fs/2;fv+=10){
    const x=freqToX(fv);
    ctx.beginPath(); ctx.moveTo(x,PAD.t*dpr); ctx.lineTo(x,H-PAD.b*dpr); ctx.stroke();
    if(fv>0){
      ctx.fillStyle='#475569'; ctx.font=`${9*dpr}px Inter`;
      ctx.textAlign='center';
      ctx.fillText(fv+'Hz', x, H-(PAD.b-12)*dpr);
    }
  }

  ctx.strokeStyle='rgba(255,255,255,.2)'; ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.moveTo(PAD.l*dpr, PAD.t*dpr);
  ctx.lineTo(PAD.l*dpr, H-PAD.b*dpr);
  ctx.lineTo(W-PAD.r*dpr, H-PAD.b*dpr);
  ctx.stroke();

  // SV curves for active datasets
  state.activeDatasets.forEach(id => {
    const ds = DATA.datasets.find(d => d.id === id);
    const color = COLORS[id % COLORS.length];
    
    for(let sv=0;sv<ds.nSV;sv++){
      ctx.beginPath();
      ctx.strokeStyle=color;
      ctx.lineWidth=(sv===0?2:1.2)*dpr;
      ctx.globalAlpha=sv===0?1:0.3;
      let first=true;
      for(let k=0;k<DATA.freqs.length;k++){
        const x=freqToX(DATA.freqs[k]);
        const y=logToY(ds.sv[k][sv]);
        if(first){ctx.moveTo(x,y);first=false;}else ctx.lineTo(x,y);
      }
      ctx.stroke();
    }
  });
  ctx.globalAlpha=1;

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

  ctx.fillStyle='#f97316';
  ctx.font=`${10*dpr}px JetBrains Mono`;
  ctx.textAlign='left';
  const labelX = curX+4*dpr > W-60*dpr ? curX-50*dpr : curX+4*dpr;
  ctx.fillText(state.cursorFreq.toFixed(2)+'Hz', labelX, PAD.t*dpr+14*dpr);

  // SV₁ dots on cursor for active datasets
  const idx = getNearestFreqIdx(state.cursorFreq);
  state.activeDatasets.forEach(id => {
    const ds = DATA.datasets.find(d => d.id === id);
    const curSV1 = ds.sv[idx][0];
    ctx.fillStyle = COLORS[id % COLORS.length];
    ctx.beginPath();
    ctx.arc(curX, logToY(curSV1), 4*dpr, 0, Math.PI*2);
    ctx.fill();
  });
}

function superscript(n){
  const map={'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵',
             '6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻'};
  return String(n).split('').map(c=>map[c]||c).join('');
}

function getNearestFreqIdx(freq){
  let best=0, bestDist=Infinity;
  for(let k=0;k<DATA.freqs.length;k++){
    const d=Math.abs(DATA.freqs[k]-freq);
    if(d<bestDist){bestDist=d;best=k;}
  }
  return best;
}

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


const VW=600, VH=260;
const BX=40, BY=130, BW=520;  // bridge baseline in SVG coords

function spanToSVGx(span){ return BX + (span / DATA.totalSpan) * BW; }
function ampToSVGy(amp){ return BY - amp*80; }

function buildModeShapePath(pts){
  if(pts.length===0) return '';
  let d=`M ${pts[0].svgx.toFixed(1)} ${pts[0].svgy.toFixed(1)}`;
  for(let i=1;i<pts.length;i++) d+=` L ${pts[i].svgx.toFixed(1)} ${pts[i].svgy.toFixed(1)}`;
  return d;
}

function renderBridgeSVG(id, freqIdx, ds){
  const shape = ds.shapes[freqIdx];
  const bridgeSVG = document.getElementById(`bridge-svg-${id}`);
  if(!bridgeSVG) return;

  const color = COLORS[id % COLORS.length];

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

  let html='';
  html+=`<defs>
    <linearGradient id="shapeGrad-${id}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${color}" stop-opacity="0.35"/>
      <stop offset="50%" stop-color="${color}" stop-opacity="0.05"/>
      <stop offset="100%" stop-color="${color}" stop-opacity="0.35"/>
    </linearGradient>
  </defs>`;

  html+=`<line x1="${BX}" y1="${BY}" x2="${BX+BW}" y2="${BY}" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" stroke-dasharray="4,4"/>`;
  html+=`<text x="${BX-6}" y="${BY+4}" font-size="9" fill="#475569" font-family="Inter" text-anchor="end">0</text>`;

  const closedPath = buildModeShapePath(svgPts) + ` L ${svgPts[svgPts.length-1].svgx.toFixed(1)} ${BY} L ${svgPts[0].svgx.toFixed(1)} ${BY} Z`;
  html+=`<path d="${closedPath}" fill="url(#shapeGrad-${id})" opacity="0.8"/>`;
  html+=`<path d="${buildModeShapePath(svgPts)}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;

  for(const k of DATA.knots){
    if(!k.isSupport) continue;
    const sx=spanToSVGx(k.pos);
    const isFixed = [1,11].includes(k.j);
    if(isFixed){
      const ts=10;
      html+=`<polygon points="${sx},${BY} ${sx-ts},${BY+ts*1.5} ${sx+ts},${BY+ts*1.5}" fill="#94a3b8" opacity="0.9"/>`;
      html+=`<line x1="${sx-ts}" y1="${BY+ts*1.5}" x2="${sx+ts}" y2="${BY+ts*1.5}" stroke="#94a3b8" stroke-width="2"/>`;
    } else {
      const ts=7;
      html+=`<polygon points="${sx},${BY-ts} ${sx+ts},${BY} ${sx},${BY+ts} ${sx-ts},${BY}" fill="#64748b" opacity="0.9"/>`;
    }
    html+=`<text x="${sx}" y="${BY+28}" font-size="9" fill="#64748b" font-family="Inter" text-anchor="middle">J${k.j}</text>`;
  }

  for(const k of DATA.knots){
    if(k.sensorIdx===null || k.isSupport) continue;
    const sx=spanToSVGx(k.pos);
    const amp=shape[k.sensorIdx];
    const sy=ampToSVGy(amp);
    html+=`<circle cx="${sx}" cy="${sy}" r="5" fill="#f87171" stroke="#fff" stroke-width="1.5" opacity="0.95"/>`;
    const labelY = amp>=0 ? sy-12 : sy+18;
    html+=`<text x="${sx}" y="${labelY}" font-size="9" fill="#fca5a5" font-family="Inter" text-anchor="middle" font-weight="600">J${k.j}</text>`;
    const valY = amp>=0 ? sy-22 : sy+28;
    html+=`<text x="${sx}" y="${valY}" font-size="8" fill="#94a3b8" font-family="JetBrains Mono" text-anchor="middle">${amp.toFixed(3)}</text>`;
  }

  for(const k of DATA.knots){
    if(k.isSupport) continue;
    const sx=spanToSVGx(k.pos);
    html+=`<circle cx="${sx}" cy="${BY}" r="2.5" fill="rgba(255,255,255,0.2)"/>`;
  }

  html+=`<text x="${BX+BW/2}" y="18" font-size="11" fill="#94a3b8" font-family="Inter" text-anchor="middle" font-weight="500">Mode Shape — Cubic Spline Interpolation</text>`;
  html+=`<text x="${BX+BW/2}" y="${VH-4}" font-size="9" fill="#475569" font-family="Inter" text-anchor="middle">Bridge Span (m) — Total: ${DATA.totalSpan.toFixed(2)} m</text>`;

  for(const k of DATA.knots){
    const sx=spanToSVGx(k.pos);
    html+=`<line x1="${sx}" y1="${BY-3}" x2="${sx}" y2="${BY+3}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>`;
  }

  html+=`<text x="${BX-6}" y="${ampToSVGy(1)+4}" font-size="8" fill="#475569" font-family="Inter" text-anchor="end">+1</text>`;
  html+=`<text x="${BX-6}" y="${ampToSVGy(-1)+4}" font-size="8" fill="#475569" font-family="Inter" text-anchor="end">-1</text>`;

  bridgeSVG.innerHTML = html;
}

function updateDisplay(){
  const freq=state.cursorFreq;
  const idx=getNearestFreqIdx(freq);

  const freqInput = document.getElementById('freq-input');
  if (document.activeElement !== freqInput) {
    freqInput.value = freq.toFixed(2);
  }

  document.getElementById('nearest-info').textContent=
    `Freq index: ${idx} / ${DATA.freqs.length-1}   |   Nearest freq bin: ${DATA.freqs[idx].toFixed(3)} Hz`;

  state.activeDatasets.forEach(id => {
    const ds = DATA.datasets.find(d => d.id === id);
    const svRow = ds.sv[idx];
    const el_sv1 = document.getElementById(`sv1-val-${id}`);
    if(el_sv1) el_sv1.textContent = svRow[0].toExponential(3);
    const ratio2 = ds.nSV>=2 ? (svRow[1]/svRow[0]).toFixed(3) : '—';
    const ratio3 = ds.nSV>=3 ? (svRow[2]/svRow[0]).toFixed(3) : '—';
    const el_sv2 = document.getElementById(`sv-ratio-${id}`);
    if(el_sv2) el_sv2.textContent = ratio2;
    const el_sv3 = document.getElementById(`sv3-ratio-${id}`);
    if(el_sv3) el_sv3.textContent = ratio3;

    renderBridgeSVG(id, idx, ds);
  });
}

const globalFreqInput = document.getElementById('freq-input');
globalFreqInput.addEventListener('change', (e) => {
  let val = parseFloat(e.target.value);
  if (!isNaN(val)) {
    state.cursorFreq = Math.max(0, Math.min(DATA.fs/2, val));
    updateDisplay();
    drawSV();
  }
});
globalFreqInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') e.target.blur();
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

initUI();
resize();
updateDisplay();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate interactive Windowed SV Explorer HTML")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--window-sec", type=float, default=3600.0, help="Window size in seconds")
    parser.add_argument("--zero-points", nargs="*", type=int, default=[4, 8])
    parser.add_argument("--fixed-points", nargs="*", type=int, default=[1, 11])
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--out", default="windowed_sv_explorer.html")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: '{args.csv}' not found.")
        return

    generate_windowed_explorer(args.csv, args.out, args.window_sec, args.zero_points, args.fixed_points, args.nperseg)

if __name__ == "__main__":
    main()
