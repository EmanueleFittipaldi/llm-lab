'use strict';
/* LLM Lab - shared helpers: DOM, theme-aware canvas, charts, 3D camera, math, quiz. */
const LAB = { modules: [], redraws: new Set(), uid: 0, tests: {} };

/* ---------- DOM ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
const uid = (p = 'c') => p + (++LAB.uid);
const f2 = (x, d = 2) => { const s = Number(x).toFixed(d); return /^-0\.?0*$/.test(s) ? s.slice(1) : s; };
const nfmt = v => { const a = Math.abs(v); if (a === 0) return '0'; if (a >= 10000 || a < 0.01) return v.toExponential(0).replace('e+', 'e'); return String(+v.toFixed(a < 1 ? 2 : a < 100 ? 1 : 0)); };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const sum = a => a.reduce((s, x) => s + x, 0);
const range = n => Array.from({ length: n }, (_, i) => i);
const human = n => { const a = Math.abs(n); if (a >= 1e12) return (n / 1e12).toFixed(2) + ' T'; if (a >= 1e9) return (n / 1e9).toFixed(2) + ' B'; if (a >= 1e6) return (n / 1e6).toFixed(1) + ' M'; if (a >= 1e3) return (n / 1e3).toFixed(1) + ' k'; return String(Math.round(n)); };

/* ---------- theme ---------- */
function theme() {
  const s = getComputedStyle(document.documentElement), g = n => (s.getPropertyValue(n) || '').trim() || '#888888';
  return { bg: g('--bg'), surface: g('--surface'), surface2: g('--surface-2'), ink: g('--ink'), muted: g('--muted'), line: g('--line'),
    accent: g('--accent'), amber: g('--amber'), teal: g('--teal'), rose: g('--rose') };
}
const MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace", SANS = "'Archivo', 'Helvetica Neue', Arial, sans-serif";
function onTheme(fn) { LAB.redraws.add(fn); }
function fireTheme() { LAB.redraws.forEach(fn => { try { fn(); } catch (e) { console.error(e); } }); }
if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', fireTheme);
new MutationObserver(fireTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

const hex = c => { const m = /^#?([0-9a-f]{6})$/i.exec(c.trim()); const n = m ? parseInt(m[1], 16) : 0x888888; return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
const rgba = (c, a) => { const [r, g, b] = hex(c); return `rgba(${r},${g},${b},${clamp(a, 0, 1)})`; };
const mixc = (c1, c2, t) => { const a = hex(c1), b = hex(c2); return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`; };

/* ---------- canvas ---------- */
function setupCanvas(canvas, draw) {
  const ctx = canvas.getContext('2d');
  const o = { canvas, ctx, w: 0, h: 0 };
  o.fit = () => {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    o.w = r.width; o.h = r.height;
    canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  o.redraw = () => {
    if (!o.w) o.fit();
    if (!o.w) return;
    ctx.clearRect(0, 0, o.w, o.h);
    draw(ctx, o.w, o.h, theme());
  };
  o.visible = () => canvas.offsetParent !== null || canvas.getClientRects().length > 0;
  if (window.ResizeObserver) new ResizeObserver(() => { o.fit(); o.redraw(); }).observe(canvas);
  else window.addEventListener('resize', () => { o.fit(); o.redraw(); });
  onTheme(o.redraw);
  return o;
}
function canvasEl(h = 280, cls = '') { const c = el('canvas', { class: 'cv ' + cls }); c.style.height = h + 'px'; return c; }

function arrow(ctx, x0, y0, x1, y1, color, w = 2, head = 8) {
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * head * 0.6, y1 - Math.sin(a) * head * 0.6); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(a - 0.4), y1 - head * Math.sin(a - 0.4));
  ctx.lineTo(x1 - head * Math.cos(a + 0.4), y1 - head * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill(); ctx.restore();
}
function text(ctx, s, x, y, o = {}) {
  ctx.save(); ctx.font = `${o.weight || 500} ${o.size || 12}px ${o.sans ? SANS : MONO}`; ctx.fillStyle = o.color || '#888';
  ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'alphabetic'; ctx.fillText(s, x, y); ctx.restore();
}

/* generic line chart. series: [{xs, ys, color, dash, lw, label}] */
function chart(ctx, w, h, t, series, o = {}) {
  const pad = { l: o.padL ?? 46, r: o.padR ?? 14, t: o.padT ?? 14, b: o.padB ?? 32 };
  const x0 = pad.l, x1 = w - pad.r, y0 = h - pad.b, y1 = pad.t;
  const ally = series.flatMap(s => s.ys.filter(Number.isFinite)), allx = series.flatMap(s => s.xs);
  let xmin = o.xmin ?? Math.min(...allx), xmax = o.xmax ?? Math.max(...allx);
  let ymin = o.ymin ?? Math.min(...ally), ymax = o.ymax ?? Math.max(...ally);
  if (!Number.isFinite(ymin)) { ymin = 0; ymax = 1; }
  if (xmax === xmin) xmax = xmin + 1; if (ymax === ymin) ymax = ymin + 1;
  const X = v => x0 + (v - xmin) / (xmax - xmin) * (x1 - x0), Y = v => y0 - (v - ymin) / (ymax - ymin) * (y0 - y1);
  ctx.save(); ctx.font = `11px ${MONO}`; ctx.lineWidth = 1;
  const nx = o.nx ?? 5, ny = o.ny ?? 4, yf = o.yfmt || nfmt, xf = o.xfmt || nfmt;
  for (let i = 0; i <= ny; i++) {
    const v = ymin + (ymax - ymin) * i / ny, y = Y(v);
    ctx.strokeStyle = rgba(t.line, 0.7); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.fillStyle = t.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(yf(v), x0 - 6, y);
  }
  for (let i = 0; i <= nx; i++) {
    const v = xmin + (xmax - xmin) * i / nx, x = X(v);
    ctx.fillStyle = t.muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(xf(v), x, y0 + 6);
  }
  ctx.strokeStyle = t.muted; ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
  if (o.xlabel) text(ctx, o.xlabel, (x0 + x1) / 2, h - 3, { color: t.muted, align: 'center', size: 11 });
  if (o.ylabel) { ctx.save(); ctx.translate(11, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); text(ctx, o.ylabel, 0, 0, { color: t.muted, align: 'center', size: 11 }); ctx.restore(); }
  ctx.beginPath(); ctx.rect(x0, y1 - 2, x1 - x0, y0 - y1 + 4); ctx.clip();
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.lw || 2; ctx.setLineDash(s.dash || []); ctx.beginPath();
    let on = false;
    s.xs.forEach((xv, i) => { const yv = s.ys[i]; if (!Number.isFinite(yv)) { on = false; return; } const px = X(xv), py = Y(yv); if (on) ctx.lineTo(px, py); else ctx.moveTo(px, py); on = true; });
    ctx.stroke(); ctx.setLineDash([]);
    if (s.dots) s.xs.forEach((xv, i) => { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(xv), Y(s.ys[i]), 3, 0, 7); ctx.fill(); });
  }
  ctx.restore();
  let lx = x0 + 10;
  series.forEach(s => {
    if (!s.label) return;
    ctx.save(); ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []); ctx.beginPath(); ctx.moveTo(lx, y1 + 9); ctx.lineTo(lx + 18, y1 + 9); ctx.stroke(); ctx.restore();
    text(ctx, s.label, lx + 23, y1 + 13, { color: t.ink, size: 11 }); lx += 32 + s.label.length * 6.6;
  });
  return { X, Y, x0, x1, y0, y1 };
}

/* heatmap with optional labels and numbers */
function heat(ctx, M, x, y, cw, ch, o = {}) {
  const R = M.length, C = M[0].length, flat = M.flat().filter(Number.isFinite);
  const vmin = o.vmin ?? Math.min(...flat), vmax = o.vmax ?? Math.max(...flat), t = o.t;
  const m = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1;
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const v = M[i][j];
    ctx.fillStyle = t.surface2; ctx.fillRect(x + j * cw, y + i * ch, cw - 1, ch - 1);
    if (!Number.isFinite(v)) { ctx.strokeStyle = t.line; ctx.beginPath(); ctx.moveTo(x + j * cw + 3, y + i * ch + 3); ctx.lineTo(x + (j + 1) * cw - 4, y + (i + 1) * ch - 4); ctx.stroke(); if (o.num && cw > 26 && v === -Infinity) text(ctx, '−∞', x + j * cw + cw / 2, y + i * ch + ch / 2 + 4, { align: 'center', size: 11, color: t.muted }); continue; }
    let u;
    if (o.diverging) { u = v / m; ctx.fillStyle = rgba(u >= 0 ? t.accent : t.rose, Math.abs(u) * 0.95); }
    else { u = (v - vmin) / ((vmax - vmin) || 1); ctx.fillStyle = rgba(o.color || t.accent, u * 0.95); }
    ctx.fillRect(x + j * cw, y + i * ch, cw - 1, ch - 1);
    if (o.num && cw > 26) text(ctx, (o.numFmt || (q => f2(q, 2)))(v), x + j * cw + cw / 2, y + i * ch + ch / 2 + 4, { align: 'center', size: Math.min(11, cw / 3.2), color: Math.abs(u) > 0.55 ? (o.diverging || o.color ? '#fff' : '#fff') : t.ink });
  }
  if (o.rows) o.rows.forEach((s, i) => text(ctx, s, x - 6, y + i * ch + ch / 2 + 4, { align: 'right', color: o.hi === i ? t.accent : t.muted, size: 11, weight: o.hi === i ? 700 : 500 }));
  if (o.cols) o.cols.forEach((s, j) => text(ctx, s, x + j * cw + cw / 2, y - 6, { align: 'center', color: t.muted, size: 11 }));
}

/* ---------- 3D camera (drag to orbit) ---------- */
class Cam3D {
  constructor(cv, redraw, o = {}) {
    this.yaw = o.yaw ?? 0.7; this.pitch = o.pitch ?? 0.5; this.redraw = redraw;
    let drag = null; cv.style.touchAction = 'none'; cv.style.cursor = 'grab';
    cv.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, moved: false }; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; });
    cv.addEventListener('pointermove', e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
      this.yaw += dx * 0.01; this.pitch = clamp(this.pitch + dy * 0.01, -1.5, 1.5); drag.x = e.clientX; drag.y = e.clientY; redraw();
    });
    const end = () => { this.wasDrag = drag && drag.moved; drag = null; cv.style.cursor = 'grab'; };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
  }
  project(x, y, z, w, h, s) {
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const x1 = x * cy + z * sy, z1 = -x * sy + z * cy, y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
    const k = 1 / (1 - z2 * 0.16), sc = s * k;
    return [w / 2 + x1 * sc, h / 2 - y1 * sc, z2];
  }
}

/* ---------- random ---------- */
function rng(seed) {
  let a = seed >>> 0;
  const r = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  r.gauss = () => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  return r;
}

/* ---------- linear algebra (arrays of arrays) ---------- */
const MX = {
  zeros: (r, c) => Array.from({ length: r }, () => new Array(c).fill(0)),
  mul(A, B) { const n = A.length, k = B.length, m = B[0].length, C = MX.zeros(n, m); for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; if (a) for (let j = 0; j < m; j++) C[i][j] += a * B[p][j]; } return C; },
  T: A => A[0].map((_, j) => A.map(r => r[j])),
  map: (A, f) => A.map(r => r.map(f)),
  add: (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j])),
};
function softmax(z, T = 1) {
  const m = Math.max(...z); if (!Number.isFinite(m)) return z.map(() => 1 / z.length);
  const e = z.map(x => Math.exp((x - m) / T)), s = sum(e); return e.map(x => x / s);
}
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = a => Math.sqrt(dot(a, a));
const cosine = (a, b) => dot(a, b) / ((norm(a) * norm(b)) || 1);
const entropy = p => -sum(p.map(x => x > 0 ? x * Math.log(x) : 0));
/* single-head scaled dot-product attention. mask: true = causal */
function attention(X, Wq, Wk, Wv, causal = false) {
  const Q = MX.mul(X, Wq), K = MX.mul(X, Wk), V = MX.mul(X, Wv), d = Wq[0].length, T = X.length;
  const S = MX.mul(Q, MX.T(K)).map(r => r.map(v => v / Math.sqrt(d)));
  const A = S.map((r, i) => softmax(r.map((v, j) => causal && j > i ? -Infinity : v)));
  return { Q, K, V, S, A, Z: MX.mul(A, V) };
}
function pca(X, k = 3) {
  const n = X.length, d = X[0].length, mu = range(d).map(j => sum(X.map(r => r[j])) / n);
  let C = X.map(r => r.map((v, j) => v - mu[j]));
  const cov = MX.mul(MX.T(C), C).map(r => r.map(v => v / n)), comps = [], r = rng(7);
  for (let c = 0; c < k; c++) {
    let v = range(d).map(() => r() - 0.5);
    for (let it = 0; it < 200; it++) { const w = cov.map(row => dot(row, v)); const nw = norm(w) || 1; v = w.map(x => x / nw); }
    const lam = dot(cov.map(row => dot(row, v)), v); comps.push(v);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] -= lam * v[i] * v[j];
  }
  return { mu, comps, proj: C.map(row => comps.map(v => dot(row, v))) };
}

/* ---------- controls ---------- */
function slider(o) {
  const { label, min, max, step = 1, value, fmt = v => String(v), onInput } = o, id = o.id || uid('s');
  const inp = el('input', { type: 'range', min, max, step, value, id }), out = el('output', { text: fmt(+value), for: id });
  const root = el('label', { class: 'ctl', for: id }, el('span', { class: 'ctl-l', text: label }), inp, out);
  inp.addEventListener('input', () => { out.textContent = fmt(+inp.value); if (onInput) onInput(+inp.value); });
  return { el: root, input: inp, get: () => +inp.value, set: v => { inp.value = v; out.textContent = fmt(+inp.value); } };
}
function button(label, onclick, cls = '') { return el('button', { class: 'btn ' + cls, type: 'button', onclick, text: label }); }
function select(opts, value, onchange, label) {
  const id = uid('sel'), s = el('select', { id }, opts.map(([v, t]) => el('option', { value: v, text: t })));
  s.value = value; s.addEventListener('change', () => onchange(s.value));
  return label ? el('label', { class: 'ctl', for: id }, el('span', { class: 'ctl-l', text: label }), s) : s;
}
function checkbox(label, checked, onchange) {
  const id = uid('chk'), i = el('input', { type: 'checkbox', id }); i.checked = checked; i.addEventListener('change', () => onchange(i.checked));
  return el('label', { class: 'chk', for: id }, i, label);
}
function lab(title, ...kids) { return el('section', { class: 'lab' }, el('div', { class: 'lab-title', text: title }), ...kids); }
function controls(...kids) { return el('div', { class: 'controls' }, ...kids); }
function math(html) { return el('div', { class: 'eq', html }); }
function details(kind, title, tag, html) {
  const d = el('details', { class: kind }, el('summary', {}, title, tag ? el('small', { text: tag }) : null), el('div', { class: 'dbody', html }));
  return d;
}
const frac = (n, d) => `<span class="frac"><span>${n}</span><span>${d}</span></span>`;

function quiz(items) {
  const root = el('div', { class: 'quiz' });
  items.forEach((it, i) => {
    const q = el('div', { class: 'q' }, el('p', { html: `<b>${i + 1}.</b> ${it.q}` }));
    const why = el('p', { class: 'why', html: it.why, hidden: true });
    const opts = it.opts.map((t, j) => {
      const b = el('button', { class: 'opt', type: 'button', html: t });
      b.addEventListener('click', () => {
        if (q.dataset.done) return; q.dataset.done = '1';
        opts.forEach((o, k) => { o.disabled = true; if (k === it.ans) o.classList.add('right'); else if (k === j) o.classList.add('wrong'); });
        why.hidden = false;
      });
      return b;
    });
    q.append(...opts, why); root.append(q);
  });
  return root;
}

/* ---------- animation loop helper (pauses when canvas hidden) ---------- */
function loop(cv, fn) {
  let id = 0, on = false;
  const tick = () => { if (!on) return; if (cv.offsetParent === null) { id = requestAnimationFrame(tick); return; } fn(); id = requestAnimationFrame(tick); };
  return { start() { if (!on) { on = true; id = requestAnimationFrame(tick); } }, stop() { on = false; cancelAnimationFrame(id); }, get running() { return on; } };
}

function registerModule(m) { LAB.modules.push(m); LAB.modules.sort((a, b) => a.n - b.n); }
