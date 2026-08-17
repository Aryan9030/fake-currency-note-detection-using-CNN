/* ==========================================================================
   GRIDNET — LSTM electricity consumption forecaster
   script.js · vanilla JavaScript, no dependencies

   Contents
     1.  small utilities
     2.  synthetic load generator (daily / weekly / weather / noise)
     3.  LSTM: forward pass, backpropagation through time, Adam
     4.  dataset assembly + scaling
     5.  canvas charts (load, loss, hidden-state heatmap)
     6.  rendering (metrics, forecast sheet)
     7.  training orchestration
     8.  wiring
   ========================================================================== */

'use strict';

/* ------------------------------------------------------------------ 1. utils */

const $ = (id) => document.getElementById(id);

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rand) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randArr(n, s, rng) {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = (rng() * 2 - 1) * s;
  return a;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const WDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function fmt(v, d = 1) {
  if (!isFinite(v)) return '—';
  return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}

function stamp(d) {
  return `${WDAYS[d.getDay()]} ${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

function hourLabel(d) {
  return `${String(d.getHours()).padStart(2, '0')}:00`;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------- 2. synthetic load series */

const PROFILES = {
  residential: {
    label: 'Residential feeder',
    unit: 'kW',
    base: 4.8,
    comfort: 20,
    coolK: 0.05,
    heatK: 0.07,
    shape: [
      0.62, 0.55, 0.5, 0.47, 0.46, 0.5, 0.6, 0.78, 0.95, 1.02, 1.0, 0.98,
      1.0, 1.02, 1.05, 1.1, 1.22, 1.42, 1.5, 1.44, 1.28, 1.1, 0.92, 0.75,
    ],
    weekly: [1.0, 1.0, 1.0, 1.0, 1.02, 1.08, 1.12], // Sun .. Sat
    seed: 11,
  },
  commercial: {
    label: 'Commercial building',
    unit: 'kW',
    base: 212,
    comfort: 21,
    coolK: 0.032,
    heatK: 0.02,
    shape: [
      0.35, 0.3, 0.28, 0.28, 0.3, 0.4, 0.6, 0.9, 1.15, 1.25, 1.28, 1.25,
      1.22, 1.24, 1.26, 1.22, 1.12, 0.98, 0.82, 0.65, 0.52, 0.45, 0.4, 0.37,
    ],
    weekly: [0.26, 1.05, 1.06, 1.06, 1.05, 1.02, 0.28],
    seed: 47,
  },
  industrial: {
    label: 'Industrial plant',
    unit: 'kW',
    base: 790,
    comfort: 24,
    coolK: 0.008,
    heatK: 0.004,
    shape: [
      0.85, 0.84, 0.83, 0.83, 0.84, 0.86, 0.9, 0.96, 1.05, 1.12, 1.15, 1.16,
      1.15, 1.14, 1.15, 1.16, 1.14, 1.1, 1.04, 0.98, 0.94, 0.9, 0.88, 0.86,
    ],
    weekly: [0.28, 1.05, 1.06, 1.05, 1.05, 1.04, 0.32],
    seed: 83,
  },
};

/**
 * Builds an hourly consumption series: base level × daily shape × weekly
 * factor, plus a slow growth trend, an autoregressive outdoor-temperature
 * signal with cooling / heating sensitivity, occasional equipment spikes and
 * gaussian metering noise.
 */
function generateLoad(cfg, rand) {
  const p = PROFILES[cfg.profile];
  const n = cfg.days * 24;
  const y = new Float64Array(n);
  const dates = new Array(n);
  const start = Date.UTC(2024, 0, 1, 0, 0, 0);
  let temp = 13;

  for (let i = 0; i < n; i++) {
    const d = new Date(start + i * 3600000);
    dates[i] = d;
    const hour = d.getUTCHours();
    const dow = d.getUTCDay();
    const day = i / 24;

    // outdoor temperature: seasonal cycle + AR(1) wobble + heat waves.
    // The cycle is centred on the middle of the record so the train slice
    // covers the same operating range as the test slice (no level shift the
    // network has never seen).
    const seasonal = 16 + 8 * Math.cos((2 * Math.PI * (day - cfg.days / 2)) / 365);
    temp = 0.982 * temp + 0.018 * seasonal + gauss(rand) * 0.85;
    if (rand() < 0.0012) temp += 5 + rand() * 4; // brief heat wave

    let load = p.base * p.shape[hour] * p.weekly[dow] * (1 + (0.03 * day) / cfg.days);

    const cool = Math.max(0, temp - p.comfort);
    const heat = Math.max(0, p.comfort - 6 - temp);
    load += p.base * (p.coolK * cool + p.heatK * heat);

    if (rand() < 0.004) load *= 1 + 0.22 * rand(); // equipment inrush
    load *= 1 + gauss(rand) * cfg.noise;

    y[i] = Math.max(p.base * 0.05, load);
  }

  return { y, dates };
}

/* --------------------------------------------------------------- 3. the LSTM */

class LSTM {
  /**
   * Single-layer LSTM.
   *   D  input features (1 here: normalised load)
   *   H  hidden units
   * Weight layout: gates are stacked in blocks of H rows, in the order
   *   [ input i | forget f | output o | candidate g ]
   */
  constructor(D, H) {
    this.D = D;
    this.H = H;

    const rng = mulberry32(20240101);
    const sX = Math.sqrt(1 / D);
    const sH = Math.sqrt(1 / H);

    this.Wx = randArr(4 * H * D, sX, rng); // 4H x D
    this.Wh = randArr(4 * H * H, sH, rng); // 4H x H
    this.b = new Float64Array(4 * H);
    this.Wy = randArr(H, sH, rng); // H -> 1
    this.by = new Float64Array(1);

    for (let j = 0; j < H; j++) this.b[H + j] = 1.0; // forget-gate bias init

    this.clip = 5;
    this.t = 0;
    this.blocks = [this.Wx, this.Wh, this.b, this.Wy, this.by].map((w) => ({
      w,
      g: new Float64Array(w.length),
      m: new Float64Array(w.length),
      v: new Float64Array(w.length),
    }));
    this.grads = this.blocks.map((b) => b.g);
    this.zeroH = new Float64Array(H);
  }

  paramCount() {
    return this.blocks.reduce((s, b) => s + b.w.length, 0);
  }

  zeroGrad() {
    for (const b of this.blocks) b.g.fill(0);
  }

  scaleGrad(k) {
    for (const b of this.blocks) {
      const g = b.g;
      for (let i = 0; i < g.length; i++) g[i] *= k;
    }
  }

  /** Forward pass over a sequence. ys[t] is the prediction for step t+1. */
  forward(xs) {
    const T = xs.length;
    const H = this.H;
    const D = this.D;
    const Wx = this.Wx;
    const Wh = this.Wh;
    const b = this.b;
    const Wy = this.Wy;

    const h = [];
    const c = [];
    const ig = [];
    const fg = [];
    const og = [];
    const gg = [];
    const tc = [];
    const ys = new Float64Array(T);

    let hp = this.zeroH;
    let cp = this.zeroH;

    for (let t = 0; t < T; t++) {
      const x = xs[t];
      const ht = new Float64Array(H);
      const ct = new Float64Array(H);
      const it = new Float64Array(H);
      const ft = new Float64Array(H);
      const ot = new Float64Array(H);
      const gt = new Float64Array(H);
      const tct = new Float64Array(H);
      let y = this.by[0];

      for (let j = 0; j < H; j++) {
        let ai = b[j];
        let af = b[H + j];
        let ao = b[2 * H + j];
        let ag = b[3 * H + j];

        for (let d = 0; d < D; d++) {
          const xv = x[d];
          ai += Wx[j * D + d] * xv;
          af += Wx[(H + j) * D + d] * xv;
          ao += Wx[(2 * H + j) * D + d] * xv;
          ag += Wx[(3 * H + j) * D + d] * xv;
        }

        const jH = j * H;
        const h1 = (H + j) * H;
        const h2 = (2 * H + j) * H;
        const h3 = (3 * H + j) * H;
        for (let k = 0; k < H; k++) {
          const hv = hp[k];
          ai += Wh[jH + k] * hv;
          af += Wh[h1 + k] * hv;
          ao += Wh[h2 + k] * hv;
          ag += Wh[h3 + k] * hv;
        }

        const iv = 1 / (1 + Math.exp(-ai));
        const fv = 1 / (1 + Math.exp(-af));
        const ov = 1 / (1 + Math.exp(-ao));
        const gv = Math.tanh(ag);

        const cv = fv * cp[j] + iv * gv;
        const tcv = Math.tanh(cv);

        it[j] = iv;
        ft[j] = fv;
        ot[j] = ov;
        gt[j] = gv;
        ct[j] = cv;
        tct[j] = tcv;
        ht[j] = ov * tcv;

        y += Wy[j] * ht[j];
      }

      ys[t] = y;
      h.push(ht);
      c.push(ct);
      ig.push(it);
      fg.push(ft);
      og.push(ot);
      gg.push(gt);
      tc.push(tct);
      hp = ht;
      cp = ct;
    }

    return { xs, h, c, ig, fg, og, gg, tc, ys };
  }

  /** Backpropagation through time. dY[t] = dL/d ys[t]. */
  backward(fw, dY) {
    const xs = fw.xs;
    const h = fw.h;
    const c = fw.c;
    const ig = fw.ig;
    const fg = fw.fg;
    const og = fw.og;
    const gg = fw.gg;
    const tc = fw.tc;

    const T = xs.length;
    const H = this.H;
    const D = this.D;
    const Wh = this.Wh;
    const Wy = this.Wy;
    const gWx = this.grads[0];
    const gWh = this.grads[1];
    const gb = this.grads[2];
    const gWy = this.grads[3];
    const gby = this.grads[4];
    const zero = this.zeroH;

    let dhNext = zero;
    let dcNext = zero;

    for (let t = T - 1; t >= 0; t--) {
      const ht = h[t];
      const ct = c[t];
      const it = ig[t];
      const ft = fg[t];
      const ot = og[t];
      const gt = gg[t];
      const tct = tc[t];
      const cm1 = t > 0 ? c[t - 1] : zero;
      const hp1 = t > 0 ? h[t - 1] : zero;
      const x = xs[t];
      const dy = dY[t];

      gby[0] += dy;

      const dhNew = new Float64Array(H);
      const dcNew = new Float64Array(H);

      for (let j = 0; j < H; j++) {
        const dh = dy * Wy[j] + dhNext[j];
        const dcv = dh * ot[j] * (1 - tct[j] * tct[j]) + dcNext[j];
        const dao = dh * tct[j];
        const daf = dcv * cm1[j];
        const dai = dcv * gt[j];
        const dag = dcv * it[j];

        const gi = dai * it[j] * (1 - it[j]);
        const gf = daf * ft[j] * (1 - ft[j]);
        const go = dao * ot[j] * (1 - ot[j]);
        const gz = dag * (1 - gt[j] * gt[j]);

        gb[j] += gi;
        gb[H + j] += gf;
        gb[2 * H + j] += go;
        gb[3 * H + j] += gz;

        gWy[j] += dy * ht[j];

        for (let d = 0; d < D; d++) {
          const xv = x[d];
          gWx[j * D + d] += gi * xv;
          gWx[(H + j) * D + d] += gf * xv;
          gWx[(2 * H + j) * D + d] += go * xv;
          gWx[(3 * H + j) * D + d] += gz * xv;
        }

        const jH = j * H;
        const h1 = (H + j) * H;
        const h2 = (2 * H + j) * H;
        const h3 = (3 * H + j) * H;
        for (let k = 0; k < H; k++) {
          const hv = hp1[k];
          gWh[jH + k] += gi * hv;
          gWh[h1 + k] += gf * hv;
          gWh[h2 + k] += go * hv;
          gWh[h3 + k] += gz * hv;
          dhNew[k] += Wh[jH + k] * gi + Wh[h1 + k] * gf + Wh[h2 + k] * go + Wh[h3 + k] * gz;
        }

        dcNew[j] = dcv * ft[j];
      }

      dhNext = dhNew;
      dcNext = dcNew;
    }
  }

  /** Adam with global-norm gradient clipping. */
  adamStep(lr) {
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.t);
    const c2 = 1 - Math.pow(b2, this.t);

    let sq = 0;
    for (const B of this.blocks) {
      const g = B.g;
      for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
    }
    const sc = sq > 0 ? Math.min(1, this.clip / Math.sqrt(sq)) : 1;

    for (const B of this.blocks) {
      const w = B.w;
      const g = B.g;
      const m = B.m;
      const v = B.v;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] * sc;
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        w[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    }
  }

  snapshot() {
    return this.blocks.map((B) => Float64Array.from(B.w));
  }

  restore(snap) {
    if (!snap) return;
    this.blocks.forEach((B, i) => B.w.set(snap[i]));
  }
}

/* ------------------------------------------------------ 4. dataset assembly */

const state = {
  cfg: null,
  y: null,
  dates: null,
  scaled: null,
  min: 0,
  max: 1,
  split: 0,
  model: null,
  running: false,
  abort: false,
  fit: null,
  forecast: null,
  band: null,
  heat: null,
  history: [],
  metrics: null,
  viewSpan: 72,
  hover: -1,
  geo: null,
  elapsed: 0,
};

function readCfg() {
  const profile = $('selProfile').value;
  return {
    profile,
    hidden: +$('selHidden').value,
    window: +$('selWindow').value,
    steps: +$('selEpochs').value,
    lr: +$('selLr').value,
    horizon: +$('selHorizon').value,
    noise: +$('rngNoise').value / 100,
    days: +$('rngDays').value,
    seed: PROFILES[profile].seed * 1000 + +$('rngDays').value * 13 + +$('rngNoise').value * 7,
  };
}

const denorm = (v) => state.min + v * (state.max - state.min);

function buildAll() {
  state.cfg = readCfg();
  const rand = mulberry32(state.cfg.seed >>> 0);
  const gen = generateLoad(state.cfg, rand);

  state.y = gen.y;
  state.dates = gen.dates;
  state.split = Math.floor(gen.y.length * 0.8);

  // scaler fitted on the TRAIN slice only — no leakage into the test set
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < state.split; i++) {
    if (gen.y[i] < mn) mn = gen.y[i];
    if (gen.y[i] > mx) mx = gen.y[i];
  }
  state.min = mn;
  state.max = mx;
  const span = mx - mn || 1;

  state.scaled = new Float64Array(gen.y.length);
  for (let i = 0; i < gen.y.length; i++) state.scaled[i] = (gen.y[i] - mn) / span;

  state.model = new LSTM(1, state.cfg.hidden);
  state.fit = null;
  state.forecast = null;
  state.band = null;
  state.heat = null;
  state.history = [];
  state.metrics = null;
  state.hover = -1;

  updateSpec();
  renderAll();
}

/** One-step fit over a range, with a warm-up so the cell state is settled. */
function computeFitRange(i0) {
  const T = state.cfg.window;
  const s = Math.max(0, i0 - T);
  const xs = [];
  for (let i = s; i < state.y.length; i++) xs.push(Float64Array.of(state.scaled[i]));
  const fw = state.model.forward(xs);
  const fit = new Array(state.y.length).fill(null);
  for (let i = s + 1; i < state.y.length; i++) fit[i] = denorm(fw.ys[i - 1 - s]);
  return fit;
}

/** Recursive multi-step forecast from index `anchor` (inclusive). */
function forecastFrom(anchor, horizon) {
  const T = state.cfg.window;
  const window = [];
  for (let i = anchor - T + 1; i <= anchor; i++) window.push(Float64Array.of(state.scaled[i]));

  const out = [];
  for (let k = 0; k < horizon; k++) {
    const fw = state.model.forward(window);
    const nxt = fw.ys[T - 1];
    out.push(nxt);
    window.push(Float64Array.of(nxt));
    window.shift();
  }
  return out;
}

/* ---------------------------------------------------------- 5. canvas layer */

function setupCanvas(cv) {
  const dpr = Math.min(2.5, window.devicePixelRatio || 1);
  const r = cv.getBoundingClientRect();
  const w = Math.max(20, Math.round(r.width));
  const h = Math.max(20, Math.round(r.height));
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

const COL = {
  actual: 'rgba(148,163,184,0.95)',
  fit: '#22d3ee',
  fcst: '#fbbf24',
  pink: '#f472b6',
  grid: 'rgba(255,255,255,0.055)',
  axis: 'rgba(255,255,255,0.32)',
};

/* ---- main load chart -------------------------------------------------- */

function drawMain() {
  const cv = $('chartMain');
  const { ctx, w, h } = setupCanvas(cv);
  const S = state;
  if (!S.y) return;

  const L = 58;
  const R = 16;
  const Tp = 18;
  const B = 28;

  const N = S.y.length;
  const span = S.viewSpan === 0 ? N : Math.min(S.viewSpan, N);
  const i0 = N - span;
  const H = S.forecast ? S.forecast.length : 0;
  const nX = span + H;

  let ymin = Infinity;
  let ymax = -Infinity;
  const showFit = $('chkFit').checked && S.fit;
  const showFcst = $('chkFcst').checked && S.forecast;

  for (let i = i0; i < N; i++) {
    const v = S.y[i];
    if (v < ymin) ymin = v;
    if (v > ymax) ymax = v;
    if (showFit && S.fit[i] != null) {
      if (S.fit[i] < ymin) ymin = S.fit[i];
      if (S.fit[i] > ymax) ymax = S.fit[i];
    }
  }
  if (showFcst) {
    for (let k = 0; k < H; k++) {
      const v = denorm(S.forecast[k]);
      const b = $('chkBand').checked && S.band ? S.band[k] : 0;
      if (v - b < ymin) ymin = v - b;
      if (v + b > ymax) ymax = v + b;
    }
  }
  const pad = (ymax - ymin) * 0.12 || 1;
  ymin -= pad;
  ymax += pad;

  const X = (i) => L + ((i - i0) / Math.max(1, nX - 1)) * (w - L - R);
  const Y = (v) => Tp + (1 - (v - ymin) / (ymax - ymin)) * (h - Tp - B);

  // grid + y labels
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const dec = ymax > 500 ? 0 : 1;
  for (let g = 0; g <= 4; g++) {
    const v = ymin + ((ymax - ymin) * g) / 4;
    const yy = Math.round(Y(v)) + 0.5;
    ctx.strokeStyle = COL.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(L, yy);
    ctx.lineTo(w - R, yy);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.34)';
    ctx.fillText(fmt(v, dec), L - 9, yy);
  }

  // x labels
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const ticks = Math.max(3, Math.min(7, Math.floor((w - L - R) / 110)));
  for (let g = 0; g <= ticks; g++) {
    const idx = Math.round(i0 + ((nX - 1) * g) / ticks);
    const d = idx < N ? S.dates[idx] : new Date(S.dates[N - 1].getTime() + (idx - N + 1) * 3600000);
    ctx.fillStyle = 'rgba(255,255,255,0.28)';
    ctx.fillText(`${stamp(d)} ${hourLabel(d)}`, X(idx), h - B + 8);
  }

  // train/test split marker
  if (S.split > i0 && S.split < N) {
    const xs = Math.round(X(S.split)) + 0.5;
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = 'rgba(244,114,182,0.55)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xs, Tp);
    ctx.lineTo(xs, h - B);
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = 'rgba(244,114,182,0.9)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText('TEST', xs + 5, Tp + 2);
  }

  const clipSave = () => {
    ctx.save();
    ctx.beginPath();
    ctx.rect(L, Tp, w - L - R, h - Tp - B);
    ctx.clip();
  };

  // actual series
  if ($('chkActual').checked) {
    clipSave();
    ctx.beginPath();
    ctx.moveTo(X(i0), Y(S.y[i0]));
    for (let i = i0 + 1; i < N; i++) ctx.lineTo(X(i), Y(S.y[i]));

    const grad = ctx.createLinearGradient(0, Tp, 0, h - B);
    grad.addColorStop(0, 'rgba(148,163,184,0.20)');
    grad.addColorStop(1, 'rgba(148,163,184,0)');
    ctx.lineTo(X(N - 1), h - B);
    ctx.lineTo(X(i0), h - B);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(X(i0), Y(S.y[i0]));
    for (let i = i0 + 1; i < N; i++) ctx.lineTo(X(i), Y(S.y[i]));
    ctx.strokeStyle = COL.actual;
    ctx.lineWidth = 1.25;
    ctx.stroke();
    ctx.restore();
  }

  // lstm fit
  if (showFit) {
    clipSave();
    ctx.beginPath();
    let started = false;
    for (let i = i0; i < N; i++) {
      if (S.fit[i] == null) continue;
      if (!started) {
        ctx.moveTo(X(i), Y(S.fit[i]));
        started = true;
      } else ctx.lineTo(X(i), Y(S.fit[i]));
    }
    ctx.strokeStyle = COL.fit;
    ctx.lineWidth = 1.7;
    ctx.shadowColor = 'rgba(34,211,238,0.6)';
    ctx.shadowBlur = 7;
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.restore();
  }

  // forecast + band
  if (showFcst) {
    clipSave();
    const base = denorm(S.forecast[0]);
    ctx.beginPath();
    ctx.moveTo(X(N - 1), Y(S.y[N - 1]));
    ctx.lineTo(X(N), Y(base));
    for (let k = 0; k < H; k++) {
      ctx.lineTo(X(N + k), Y(denorm(S.forecast[k]) + ($('chkBand').checked && S.band ? S.band[k] : 0)));
    }
    for (let k = H - 1; k >= 0; k--) {
      ctx.lineTo(X(N + k), Y(denorm(S.forecast[k]) - ($('chkBand').checked && S.band ? S.band[k] : 0)));
    }
    ctx.lineTo(X(N), Y(base));
    ctx.closePath();
    ctx.fillStyle = 'rgba(251,191,36,0.13)';
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(X(N - 1), Y(S.y[N - 1]));
    for (let k = 0; k < H; k++) ctx.lineTo(X(N + k), Y(denorm(S.forecast[k])));
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = COL.fcst;
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(251,191,36,0.7)';
    ctx.shadowBlur = 9;
    ctx.stroke();
    ctx.restore();

    // anchor point
    ctx.beginPath();
    ctx.arc(X(N - 1), Y(S.y[N - 1]), 3.2, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
    const lastV = denorm(S.forecast[H - 1]);
    ctx.beginPath();
    ctx.arc(X(N + H - 1), Y(lastV), 3.6, 0, Math.PI * 2);
    ctx.fillStyle = COL.fcst;
    ctx.fill();
    ctx.restore();
  }

  // hover crosshair
  if (S.hover >= i0 && S.hover < N + H) {
    const xh = Math.round(X(S.hover)) + 0.5;
    ctx.strokeStyle = 'rgba(255,255,255,0.28)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xh, Tp);
    ctx.lineTo(xh, h - B);
    ctx.stroke();

    const isF = S.hover >= N;
    const v = isF ? denorm(S.forecast[S.hover - N]) : S.y[S.hover];
    const yv = Y(v);
    ctx.beginPath();
    ctx.arc(xh, yv, 4, 0, Math.PI * 2);
    ctx.fillStyle = isF ? COL.fcst : COL.fit;
    ctx.fill();
    ctx.strokeStyle = 'rgba(7,11,20,0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // frame
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.strokeRect(L + 0.5, Tp + 0.5, w - L - R - 1, h - Tp - B - 1);

  S.geo = { L, R, T: Tp, B, i0, nX, w, h, ymin, ymax, N };
}

function onChartMove(ev) {
  const S = state;
  if (!S.geo || !S.y) return;
  const cv = $('chartMain');
  const r = cv.getBoundingClientRect();
  const mx = ev.clientX - r.left;
  const { L, R, i0, nX, w, N } = S.geo;
  const frac = (mx - L) / Math.max(1, w - L - R);
  const idx = Math.round(i0 + frac * (nX - 1));
  const hi = N + (S.forecast ? S.forecast.length : 0);
  if (idx < i0 || idx >= hi) {
    S.hover = -1;
    $('chartTip').classList.remove('is-on');
    drawMain();
    return;
  }
  const isF = idx >= N;
  if (isF && !$('chkFcst').checked) {
    state.hover = -1;
    $('chartTip').classList.remove('is-on');
    drawMain();
    return;
  }
  S.hover = idx;
  drawMain();

  const d = isF
    ? new Date(S.dates[N - 1].getTime() + (idx - N + 1) * 3600000)
    : S.dates[idx];
  const val = isF ? denorm(S.forecast[idx - N]) : S.y[idx];
  const unit = PROFILES[S.cfg.profile].unit;
  const tip = $('chartTip');
  const rows = [
    `<span class="k">${stamp(d)} · ${hourLabel(d)}${isF ? ' · forecast' : ''}</span>`,
    `actual <b>${isF ? '—' : fmt(S.y[idx], 1) + ' ' + unit}</b>`,
  ];
  if (!isF && S.fit && S.fit[idx] != null) rows.push(`lstm&nbsp;&nbsp;&nbsp;<b>${fmt(S.fit[idx], 1)} ${unit}</b>`);
  if (isF) rows.push(`pred&nbsp;&nbsp;&nbsp;<b>${fmt(val, 1)} ${unit}</b>`);
  tip.innerHTML = rows.join('<br>');
  tip.classList.add('is-on');
}

/* ---- loss curve -------------------------------------------------------- */

function drawLoss() {
  const cv = $('chartLoss');
  const { ctx, w, h } = setupCanvas(cv);
  const hist = state.history;

  const L = 44;
  const R = 12;
  const Tp = 14;
  const B = 22;

  ctx.font = '9.5px "JetBrains Mono", monospace';
  ctx.strokeStyle = COL.grid;
  ctx.lineWidth = 1;

  if (hist.length < 2) {
    ctx.fillStyle = 'rgba(255,255,255,0.22)';
    ctx.textAlign = 'center';
    ctx.fillText('AWAITING TRAINING', w / 2, h / 2);
    return;
  }

  let lo = Infinity;
  let hi = -Infinity;
  for (const p of hist) {
    for (const v of [p.train, p.val]) {
      if (v > 0) {
        const lg = Math.log10(v);
        if (lg < lo) lo = lg;
        if (lg > hi) hi = lg;
      }
    }
  }
  if (!isFinite(lo)) {
    lo = -4;
    hi = 0;
  }
  const pad = Math.max(0.15, (hi - lo) * 0.12);
  lo -= pad;
  hi += pad;

  const s0 = hist[0].step;
  const s1 = hist[hist.length - 1].step;
  const X = (s) => L + ((s - s0) / Math.max(1, s1 - s0)) * (w - L - R);
  const Y = (v) => Tp + (1 - (Math.log10(Math.max(v, 1e-8)) - lo) / (hi - lo)) * (h - Tp - B);

  // decade gridlines
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
    const yy = Math.round(Y(Math.pow(10, e))) + 0.5;
    ctx.strokeStyle = COL.grid;
    ctx.beginPath();
    ctx.moveTo(L, yy);
    ctx.lineTo(w - R, yy);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.fillText(`1e${e}`, L - 6, yy);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(L, Tp, w - L - R, h - Tp - B);
  ctx.clip();

  const line = (key, color, width) => {
    ctx.beginPath();
    hist.forEach((p, i) => {
      if (!isFinite(p[key]) || p[key] <= 0) return;
      const x = X(p.step);
      const y = Y(p[key]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  };

  line('train', 'rgba(34,211,238,0.95)', 1.6);
  line('val', 'rgba(251,191,36,0.95)', 1.6);
  ctx.restore();

  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillStyle = 'rgba(34,211,238,0.85)';
  ctx.fillText('— train', L + 6, Tp + 4);
  ctx.fillStyle = 'rgba(251,191,36,0.85)';
  ctx.fillText('— validation', L + 6, Tp + 16);
  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.fillText(`step ${s1}`, w - R, h - B + 6);

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.strokeRect(L + 0.5, Tp + 0.5, w - L - R - 1, h - Tp - B - 1);
}

/* ---- hidden-state heatmap ---------------------------------------------- */

function cmap(v) {
  const stops = [
    [12, 18, 48],
    [14, 116, 174],
    [34, 211, 238],
    [163, 230, 53],
    [251, 191, 36],
    [244, 63, 94],
  ];
  const t = Math.max(0, Math.min(1, (v + 1) / 2)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const f = t - i;
  const a = stops[i];
  const b = stops[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(
    a[2] + (b[2] - a[2]) * f
  )})`;
}

function drawHeat() {
  const cv = $('chartHeat');
  const { ctx, w, h } = setupCanvas(cv);
  const M = state.heat;

  const L = 8;
  const R = 46;
  const Tp = 10;
  const B = 20;

  if (!M || !M.length) {
    ctx.font = '9.5px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.22)';
    ctx.textAlign = 'center';
    ctx.fillText('NO HIDDEN STATE YET', w / 2, h / 2);
    return;
  }

  const T = M.length;
  const H = M[0].length;
  const cw = (w - L - R) / T;
  const ch = (h - Tp - B) / H;

  for (let t = 0; t < T; t++) {
    const row = M[t];
    for (let j = 0; j < H; j++) {
      ctx.fillStyle = cmap(row[j]);
      ctx.fillRect(L + t * cw, Tp + j * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }

  // colour scale
  const bx = w - R + 12;
  const bw = 10;
  const bh = h - Tp - B;
  for (let i = 0; i < bh; i++) {
    ctx.fillStyle = cmap(1 - (2 * i) / bh);
    ctx.fillRect(bx, Tp + i, bw, 1);
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 1;
  ctx.strokeRect(bx + 0.5, Tp + 0.5, bw, bh);
  ctx.font = '8.5px "JetBrains Mono", monospace';
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText('+1', bx + bw + 4, Tp + 5);
  ctx.fillText('0', bx + bw + 4, Tp + bh / 2);
  ctx.fillText('−1', bx + bw + 4, Tp + bh - 5);

  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.fillText(`${H} units ↑`, L, h - B + 6);
  ctx.textAlign = 'right';
  ctx.fillText(`last ${T} h →`, w - R, h - B + 6);
}

/* ------------------------------------------------------------ 6. rendering */

function setStatus(mode, text) {
  const pill = $('statusPill');
  pill.className = 'pill' + (mode ? ` is-${mode}` : '');
  $('statusText').textContent = text;
}

function updateSpec() {
  const S = state;
  const unit = PROFILES[S.cfg.profile].unit;
  $('chartUnit').textContent = unit;
  $('paramCount').textContent = S.model.paramCount().toLocaleString('en-US');
  $('bpttDepth').textContent = `${S.cfg.window} steps`;
  $('splitInfo').textContent = `${S.split.toLocaleString('en-US')} / ${(S.y.length - S.split).toLocaleString('en-US')}`;
  $('hstatPoints').textContent = S.y.length.toLocaleString('en-US');
  $('hstatParams').textContent = S.model.paramCount().toLocaleString('en-US');
  $('legendNote').textContent = `${PROFILES[S.cfg.profile].label} · 80/20 split`;
  $('footStamp').textContent = `${PROFILES[S.cfg.profile].label} · ${S.cfg.days} days`;
}

function renderMetrics() {
  const S = state;
  const m = S.metrics;
  if (!m) {
    ['mRmse', 'mMae', 'mMape', 'mR2', 'mMulti'].forEach((id) => ($(id).textContent = '—'));
    $('hstatR2').textContent = '—';
    $('hstatBeat').textContent = '—';
    return;
  }
  const unit = PROFILES[S.cfg.profile].unit;
  $('mRmse').textContent = fmt(m.rmse, 1);
  $('mRmseSub').textContent = `${unit} · 1-step`;
  $('mMae').textContent = fmt(m.mae, 1);
  $('mMaeSub').textContent = `${unit} · 1-step`;
  $('mMape').textContent = `${fmt(m.mape, 2)}%`;
  $('mR2').textContent = fmt(m.r2, 3);
  $('mMulti').textContent = fmt(m.multi, 1);
  $('mMultiSub').textContent = `${unit} · ${S.cfg.horizon} h recursive`;

  const beat = (m.baseRmse - m.rmse) / m.baseRmse * 100;
  $('hstatR2').textContent = fmt(m.r2, 3);
  $('hstatBeat').textContent = `${beat >= 0 ? '+' : ''}${fmt(beat, 0)}%`;
  $('hstatBeat').style.color = beat >= 0 ? 'var(--lime)' : 'var(--pink)';
}

function renderTable() {
  const S = state;
  const body = $('fcBody');
  if (!S.forecast) {
    body.innerHTML = '<tr class="empty"><td colspan="6">No forecast yet — hit “Train the network”.</td></tr>';
    return;
  }
  const unit = PROFILES[S.cfg.profile].unit;
  const N = S.y.length;
  const vals = S.forecast.map(denorm);
  const peak = Math.max(...vals);
  const rows = [];

  vals.forEach((v, k) => {
    const d = new Date(S.dates[N - 1].getTime() + (k + 1) * 3600000);
    const prevIdx = N - 1 + k + 1 - 24;
    const prev = prevIdx >= 0 ? S.y[prevIdx] : null;
    const delta = prev == null ? null : v - prev;
    const pct = prev == null || prev === 0 ? null : (delta / prev) * 100;
    const share = (v / peak) * 100;
    rows.push(
      `<tr>
        <td class="mono dim">${String(k + 1).padStart(2, '0')}</td>
        <td class="mono">${stamp(d)} <span class="dim">${hourLabel(d)}</span></td>
        <td class="r v">${fmt(v, 1)} <span class="dim">${unit}</span></td>
        <td class="r ${delta == null ? '' : delta >= 0 ? 'up' : 'down'}">${
          delta == null ? '—' : `${delta >= 0 ? '+' : '−'}${fmt(Math.abs(delta), 1)} (${pct >= 0 ? '+' : '−'}${fmt(Math.abs(pct), 1)}%)`
        }</td>
        <td class="r">${fmt(share, 0)}%</td>
        <td class="r"><span class="bar-cell"><span class="mini-bar"><i style="width:${share.toFixed(1)}%"></i></span></span></td>
      </tr>`
    );
  });

  body.innerHTML = rows.join('');
  $('fcMeta').textContent = `${S.cfg.horizon} h recursive`;
}

function renderAll() {
  drawMain();
  drawLoss();
  drawHeat();
  renderMetrics();
  renderTable();
  $('chartEmpty').classList.toggle('is-on', !state.fit);
}

/* ------------------------------------------------- 7. training orchestration */

const BATCH = 6;

function makeSampler() {
  const T = state.cfg.window;
  const hi = state.split - T - 1;
  return () => Math.max(0, Math.floor(Math.random() * (hi + 1)));
}

function trainStep(lr) {
  const m = state.model;
  const T = state.cfg.window;
  const sc = state.scaled;
  const sample = makeSampler();

  m.zeroGrad();
  let loss = 0;
  let cnt = 0;

  for (let b = 0; b < BATCH; b++) {
    const s = sample();
    const xs = new Array(T);
    const tgt = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      xs[t] = Float64Array.of(sc[s + t]);
      tgt[t] = sc[s + t + 1];
    }
    const fw = m.forward(xs);
    const dY = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      const e = fw.ys[t] - tgt[t];
      dY[t] = e;
      loss += e * e;
      cnt++;
    }
    m.backward(fw, dY);
  }

  m.scaleGrad(1 / BATCH);
  m.adamStep(lr);
  return loss / cnt;
}

function valLoss() {
  const m = state.model;
  const T = state.cfg.window;
  const sc = state.scaled;
  const N = sc.length;
  let loss = 0;
  let cnt = 0;
  for (let s = state.split - T; s <= N - T - 1; s++) {
    const xs = new Array(T);
    for (let t = 0; t < T; t++) xs[t] = Float64Array.of(sc[s + t]);
    const fw = m.forward(xs);
    for (let t = 0; t < T; t++) {
      const e = fw.ys[t] - sc[s + t + 1];
      loss += e * e;
      cnt++;
    }
  }
  return cnt ? loss / cnt : NaN;
}

function captureHeat() {
  const T = state.cfg.window;
  const N = state.scaled.length;
  const xs = [];
  for (let i = N - T; i < N; i++) xs.push(Float64Array.of(state.scaled[i]));
  const fw = state.model.forward(xs);
  state.heat = fw.h;
}

function evaluate() {
  const S = state;
  const N = S.y.length;
  const unit = PROFILES[S.cfg.profile].unit;

  // full one-step fit across the whole series
  S.fit = computeFitRange(0);

  // test-set metrics + persistence baseline
  let se = 0;
  let ae = 0;
  let ape = 0;
  let bse = 0;
  let n = 0;
  const resid = [];
  let mean = 0;
  for (let i = S.split; i < N; i++) mean += S.y[i];
  mean /= N - S.split;

  for (let i = S.split; i < N; i++) {
    const p = S.fit[i];
    const a = S.y[i];
    const e = p - a;
    se += e * e;
    ae += Math.abs(e);
    ape += Math.abs(e) / a;
    resid.push(e);
    const be = S.y[i - 1] - a;
    bse += be * be;
    n++;
  }

  const sst = (() => {
    let s = 0;
    for (let i = S.split; i < N; i++) s += (S.y[i] - mean) ** 2;
    return s;
  })();

  const rmse = Math.sqrt(se / n);
  const sigma = Math.sqrt(se / n);

  // recursive multi-step evaluation on the last in-sample anchor
  const horizon = S.cfg.horizon;
  const anchor = N - 1 - horizon;
  const multi = forecastFrom(anchor, horizon);
  let mse2 = 0;
  for (let k = 0; k < horizon; k++) {
    const e = denorm(multi[k]) - S.y[anchor + 1 + k];
    mse2 += e * e;
  }

  // final forecast from the very end of the series + widening uncertainty band
  S.forecast = forecastFrom(N - 1, horizon);
  S.band = S.forecast.map((_, k) => 2 * sigma * Math.sqrt(k + 1));

  S.metrics = {
    rmse,
    mae: ae / n,
    mape: (ape / n) * 100,
    r2: 1 - se / sst,
    baseRmse: Math.sqrt(bse / n),
    multi: Math.sqrt(mse2 / horizon),
    unit,
  };

  captureHeat();
}

async function runTraining() {
  if (state.running) {
    state.abort = true;
    $('btnTrainLabel').textContent = 'Stopping…';
    return;
  }

  buildAll();
  const cfg = state.cfg;
  state.running = true;
  state.abort = false;

  const btn = $('btnTrain');
  btn.classList.add('is-stopping');
  $('btnTrainLabel').textContent = 'Stop training';
  setStatus('training', 'training');

  const total = cfg.steps;
  const t0 = performance.now();
  let best = Infinity;
  let snap = null;
  let ema = null;
  const patience = Math.max(15, Math.floor((total / 15) * 0.4));
  let sinceBest = 0;

  for (let step = 1; step <= total; step++) {
    if (state.abort) break;

    const decay = 0.5 * (1 + Math.cos((Math.PI * (step - 1)) / total));
    const lr = cfg.lr * (0.05 + 0.95 * decay);
    const tl = trainStep(lr);
    ema = ema == null ? tl : ema * 0.9 + tl * 0.1;

    if (step % 15 === 0 || step === 1) {
      const vl = valLoss();
      state.history.push({ step, train: ema, val: vl });
      if (vl < best - 1e-6) {
        best = vl;
        snap = state.model.snapshot();
        sinceBest = 0;
      } else sinceBest++;
    }

    if (step % 4 === 0) {
      const pct = Math.round((step / total) * 100);
      $('progressBar').style.width = `${pct}%`;
      $('progressPct').textContent = `${pct}%`;
      $('statEpoch').textContent = step.toLocaleString('en-US');
      $('statLoss').textContent = ema == null ? '—' : ema.toFixed(5);
      $('statVal').textContent = isFinite(best) ? best.toFixed(5) : '—';
      $('statBest').textContent = isFinite(best) ? best.toFixed(5) : '—';
      $('statLr').textContent = lr.toFixed(4);
      const el = (performance.now() - t0) / 1000;
      state.elapsed = el;
      $('statTime').textContent = `${el.toFixed(1)} s`;
      $('progressMeta').textContent = `batch ${BATCH} × window ${cfg.window} · adam`;
      drawLoss();
    }

    if (step % 60 === 0) {
      const span = state.viewSpan === 0 ? state.y.length : Math.min(state.viewSpan, state.y.length);
      state.fit = computeFitRange(state.y.length - span);
      drawMain();
    }

    if (sinceBest > patience) {
      $('progressMeta').textContent = 'early stop — validation plateaued';
      break;
    }

    if (step % 6 === 0) await sleep(0);
  }

  if (snap) state.model.restore(snap);

  evaluate();
  state.running = false;

  btn.classList.remove('is-stopping');
  $('btnTrainLabel').textContent = 'Retrain';
  $('progressBar').style.width = '100%';
  $('progressPct').textContent = '100%';
  $('statTime').textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;

  if (state.abort) {
    setStatus('done', 'stopped');
    $('progressMeta').textContent = 'stopped early — best checkpoint restored';
  } else {
    setStatus('done', 'trained');
    $('progressMeta').textContent = `done · ${total} steps · best val ${isFinite(best) ? best.toFixed(5) : '—'}`;
  }

  renderAll();
}

/* ------------------------------------------------------------- 8. wiring */

function wire() {
  $('btnTrain').addEventListener('click', runTraining);

  $('btnReset').addEventListener('click', () => {
    state.abort = true;
    setTimeout(() => {
      buildAll();
      setStatus('', 'engine idle');
      $('btnTrainLabel').textContent = 'Train the network';
      $('progressBar').style.width = '0%';
      $('progressPct').textContent = '0%';
      $('progressMeta').textContent = 'awaiting start';
      ['statEpoch', 'statLoss', 'statVal', 'statBest', 'statLr'].forEach((id) => ($(id).textContent = '—'));
      $('statTime').textContent = '0.0 s';
    }, 60);
  });

  $('rngNoise').addEventListener('input', (e) => {
    $('noiseVal').textContent = `${(+e.target.value).toFixed(1)}%`;
  });

  $('rngDays').addEventListener('input', (e) => {
    $('daysVal').textContent = `${e.target.value} days`;
  });

  ['selProfile', 'selHidden', 'selWindow', 'selEpochs', 'selLr', 'selHorizon', 'rngNoise', 'rngDays'].forEach((id) => {
    $(id).addEventListener('change', () => {
      if (state.running) return;
      buildAll();
      setStatus('', 'engine idle');
      $('btnTrainLabel').textContent = 'Train the network';
    });
  });

  ['chkActual', 'chkFit', 'chkFcst', 'chkBand'].forEach((id) => $(id).addEventListener('change', drawMain));

  $('zoomGroup').addEventListener('click', (e) => {
    const b = e.target.closest('.zoom');
    if (!b) return;
    document.querySelectorAll('.zoom').forEach((z) => z.classList.remove('active'));
    b.classList.add('active');
    state.viewSpan = +b.dataset.span;
    if (state.fit) state.fit = computeFitRange(state.y.length - (state.viewSpan === 0 ? state.y.length : Math.min(state.viewSpan, state.y.length)));
    drawMain();
  });

  const cv = $('chartMain');
  cv.addEventListener('mousemove', onChartMove);
  cv.addEventListener('mouseleave', () => {
    state.hover = -1;
    $('chartTip').classList.remove('is-on');
    drawMain();
  });

  let rt;
  window.addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(renderAll, 140);
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      runTraining();
    }
  });
}

function boot() {
  wire();
  buildAll();
  setStatus('', 'engine idle');
  setTimeout(runTraining, 700);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
