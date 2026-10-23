'use strict';
/* Module 16 (capstone) - a complete decoder-only transformer, with hand-written backpropagation, trained in the page.
   Pre-LayerNorm blocks, multi-head causal attention, GELU MLP, learned positions, tied input/output embedding. */

const GC = Math.sqrt(2 / Math.PI);
const geluF = u => 0.5 * u * (1 + Math.tanh(GC * (u + 0.044715 * u * u * u)));
const geluD = u => { const t = Math.tanh(GC * (u + 0.044715 * u * u * u)); return 0.5 * (1 + t) + 0.5 * u * (1 - t * t) * GC * (1 + 3 * 0.044715 * u * u); };

class TinyGPT {
  constructor(V, cfg = {}, seed = 11) {
    const D = cfg.D || 16, H = cfg.H || 2, L = cfg.L || 1, T = cfg.T || 16, F = cfg.F || 3 * D;
    Object.assign(this, { V, D, H, L, T, F, dh: D / H }); const r = rng(seed), rnd = (n, s) => Float64Array.from({ length: n }, () => r.gauss() * s), ones = n => new Float64Array(n).fill(1), zeros = n => new Float64Array(n);
    const sd = 1 / Math.sqrt(D), so = sd / Math.sqrt(2 * L);
    this.E = rnd(V * D, 0.15); this.Pe = rnd(T * D, 0.1); this.gf = ones(D); this.bf = zeros(D);
    this.layers = Array.from({ length: L }, () => ({ g1: ones(D), n1: zeros(D), Wq: rnd(D * D, sd), Wk: rnd(D * D, sd), Wv: rnd(D * D, sd), Wo: rnd(D * D, so), g2: ones(D), n2: zeros(D), W1: rnd(D * F, sd), c1: zeros(F), W2: rnd(F * D, so / Math.sqrt(F / D)), c2: zeros(D) }));
    const zl = o => Object.fromEntries(Object.entries(o).map(([k, a]) => [k, new Float64Array(a.length)]));
    this.gE = new Float64Array(this.E.length); this.gPe = new Float64Array(this.Pe.length); this.ggf = zeros(D); this.gbf = zeros(D); this.gl = this.layers.map(zl);
    this.slots = [[this.E, this.gE], [this.Pe, this.gPe], [this.gf, this.ggf], [this.bf, this.gbf]];
    this.layers.forEach((o, l) => Object.keys(o).forEach(k => this.slots.push([o[k], this.gl[l][k]])));
    this.M = this.slots.map(([p]) => new Float64Array(p.length)); this.S = this.slots.map(([p]) => new Float64Array(p.length)); this.t = 0;
  }
  get nParams() { return this.slots.reduce((s, [p]) => s + p.length, 0); }
  /* ---- forward: x = Int32Array of n <= T tokens; returns { logits (n x V), cache } ---- */
  forward(x) {
    const { V, D, H, L, F, dh, E, Pe } = this, n = x.length, sc = 1 / Math.sqrt(dh), cache = { x, n, layers: [] };
    let h = new Float64Array(n * D);
    for (let t = 0; t < n; t++) for (let c = 0; c < D; c++) h[t * D + c] = E[x[t] * D + c] + Pe[t * D + c];
    for (let l = 0; l < L; l++) {
      const P = this.layers[l], ca = { hin: h };
      const ln1 = TinyGPT.ln(h, n, D, P.g1, P.n1); ca.a = ln1.y; ca.s1 = ln1;
      const q = TinyGPT.mm(ln1.y, n, D, P.Wq, D), k = TinyGPT.mm(ln1.y, n, D, P.Wk, D), v = TinyGPT.mm(ln1.y, n, D, P.Wv, D);
      const p = new Float64Array(H * n * n), o = new Float64Array(n * D);
      for (let hh = 0; hh < H; hh++) for (let i = 0; i < n; i++) {
        const row = (hh * n + i) * n; let m = -Infinity;
        for (let j = 0; j <= i; j++) { let s = 0; for (let c = 0; c < dh; c++) s += q[i * D + hh * dh + c] * k[j * D + hh * dh + c]; s *= sc; p[row + j] = s; if (s > m) m = s; }
        let z = 0; for (let j = 0; j <= i; j++) { p[row + j] = Math.exp(p[row + j] - m); z += p[row + j]; }
        for (let j = 0; j <= i; j++) p[row + j] /= z;
        for (let j = 0; j <= i; j++) { const w = p[row + j]; for (let c = 0; c < dh; c++) o[i * D + hh * dh + c] += w * v[j * D + hh * dh + c]; }
      }
      Object.assign(ca, { q, k, v, p, o });
      const ao = TinyGPT.mm(o, n, D, P.Wo, D), h1 = new Float64Array(n * D); for (let i = 0; i < n * D; i++) h1[i] = h[i] + ao[i];
      ca.h1 = h1; const ln2 = TinyGPT.ln(h1, n, D, P.g2, P.n2); ca.b = ln2.y; ca.s2 = ln2;
      const u = TinyGPT.mm(ln2.y, n, D, P.W1, F); for (let t = 0; t < n; t++) for (let c = 0; c < F; c++) u[t * F + c] += P.c1[c];
      const g = u.map(geluF), m2 = TinyGPT.mm(g, n, F, P.W2, D), h2 = new Float64Array(n * D);
      for (let t = 0; t < n; t++) for (let c = 0; c < D; c++) h2[t * D + c] = h1[t * D + c] + m2[t * D + c] + P.c2[c];
      ca.u = u; ca.g = g; cache.layers.push(ca); h = h2;
    }
    const lf = TinyGPT.ln(h, n, D, this.gf, this.bf); cache.hL = h; cache.lf = lf;
    const logits = new Float64Array(n * V);
    for (let t = 0; t < n; t++) for (let vv = 0; vv < V; vv++) { let s = 0; for (let c = 0; c < D; c++) s += lf.y[t * D + c] * E[vv * D + c]; logits[t * V + vv] = s; }
    return { logits, cache };
  }
  static mm(A, n, k, B, m) { const C = new Float64Array(n * m); for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i * k + p]; if (a !== 0) { const bo = p * m, co = i * m; for (let j = 0; j < m; j++) C[co + j] += a * B[bo + j]; } } return C; }
  static ln(x, n, D, g, b) {
    const y = new Float64Array(n * D), xh = new Float64Array(n * D), rs = new Float64Array(n);
    for (let t = 0; t < n; t++) { let mu = 0; for (let c = 0; c < D; c++) mu += x[t * D + c]; mu /= D; let va = 0; for (let c = 0; c < D; c++) { const d = x[t * D + c] - mu; va += d * d; } va /= D; const r = 1 / Math.sqrt(va + 1e-5); rs[t] = r; for (let c = 0; c < D; c++) { xh[t * D + c] = (x[t * D + c] - mu) * r; y[t * D + c] = xh[t * D + c] * g[c] + b[c]; } }
    return { y, xh, rs };
  }
  static lnBack(dy, st, n, D, g, dg, db) {
    const dx = new Float64Array(n * D);
    for (let t = 0; t < n; t++) {
      let m1 = 0, m2 = 0;
      for (let c = 0; c < D; c++) { const dxh = dy[t * D + c] * g[c]; m1 += dxh; m2 += dxh * st.xh[t * D + c]; dg[c] += dy[t * D + c] * st.xh[t * D + c]; db[c] += dy[t * D + c]; }
      m1 /= D; m2 /= D;
      for (let c = 0; c < D; c++) dx[t * D + c] = st.rs[t] * (dy[t * D + c] * g[c] - m1 - st.xh[t * D + c] * m2);
    }
    return dx;
  }
  /* C = A x B (A n x k, B k x m): accumulate dB += A^T dC and return dA = dC B^T */
  static mmBack(A, n, k, B, m, dC, dB) {
    const dA = new Float64Array(n * k);
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i * k + p]; let s = 0; const bo = p * m, co = i * m; for (let j = 0; j < m; j++) { const d = dC[co + j]; dB[bo + j] += a * d; s += d * B[bo + j]; } dA[i * k + p] = s; }
    return dA;
  }
  /* mean cross-entropy over a batch of windows (each of T+1 tokens: inputs 0..T-1, targets 1..T); fills gradients */
  lossAndGrad(batch, wantGrad = true) {
    const { V, D, H, L, F, dh, E } = this, sc = 1 / Math.sqrt(dh); let loss = 0, cnt = 0;
    if (wantGrad) { this.slots.forEach(([, g]) => g.fill(0)); }
    for (const win of batch) {
      const x = win.subarray(0, win.length - 1), n = x.length, { logits, cache } = this.forward(x), dl = new Float64Array(n * V);
      for (let t = 0; t < n; t++) {
        let m = -Infinity; for (let v = 0; v < V; v++) if (logits[t * V + v] > m) m = logits[t * V + v];
        let z = 0; for (let v = 0; v < V; v++) { dl[t * V + v] = Math.exp(logits[t * V + v] - m); z += dl[t * V + v]; }
        for (let v = 0; v < V; v++) dl[t * V + v] /= z;
        loss -= Math.log(dl[t * V + win[t + 1]] + 1e-12); cnt++;
        dl[t * V + win[t + 1]] -= 1;
      }
      if (!wantGrad) continue;
      const scale = 1 / (batch.length * n); for (let i = 0; i < dl.length; i++) dl[i] *= scale;
      // logits = lf.y E^T
      const lfy = cache.lf.y, dlf = new Float64Array(n * D);
      for (let t = 0; t < n; t++) for (let v = 0; v < V; v++) { const d = dl[t * V + v]; if (d === 0) continue; for (let c = 0; c < D; c++) { this.gE[v * D + c] += d * lfy[t * D + c]; dlf[t * D + c] += d * E[v * D + c]; } }
      let dh_ = TinyGPT.lnBack(dlf, cache.lf, n, D, this.gf, this.ggf, this.gbf);
      for (let l = L - 1; l >= 0; l--) {
        const P = this.layers[l], G = this.gl[l], ca = cache.layers[l];
        // MLP branch: h2 = h1 + g W2 + c2
        for (let t = 0; t < n; t++) for (let c = 0; c < D; c++) G.c2[c] += dh_[t * D + c];
        const dg = TinyGPT.mmBack(ca.g, n, F, P.W2, D, dh_, G.W2), du = new Float64Array(n * F);
        for (let i = 0; i < n * F; i++) du[i] = dg[i] * geluD(ca.u[i]);
        for (let t = 0; t < n; t++) for (let c = 0; c < F; c++) G.c1[c] += du[t * F + c];
        const db = TinyGPT.mmBack(ca.b, n, D, P.W1, F, du, G.W1), dh1 = TinyGPT.lnBack(db, ca.s2, n, D, P.g2, G.g2, G.n2);
        for (let i = 0; i < n * D; i++) dh1[i] += dh_[i];
        // attention branch: h1 = hin + o Wo
        const dO = TinyGPT.mmBack(ca.o, n, D, P.Wo, D, dh1, G.Wo), dq = new Float64Array(n * D), dk = new Float64Array(n * D), dv = new Float64Array(n * D);
        for (let hh = 0; hh < H; hh++) for (let i = 0; i < n; i++) {
          const row = (hh * n + i) * n, dp = new Float64Array(i + 1); let dot = 0;
          for (let j = 0; j <= i; j++) { let s = 0; for (let c = 0; c < dh; c++) { s += dO[i * D + hh * dh + c] * ca.v[j * D + hh * dh + c]; dv[j * D + hh * dh + c] += ca.p[row + j] * dO[i * D + hh * dh + c]; } dp[j] = s; dot += ca.p[row + j] * s; }
          for (let j = 0; j <= i; j++) { const ds = ca.p[row + j] * (dp[j] - dot) * sc; for (let c = 0; c < dh; c++) { dq[i * D + hh * dh + c] += ds * ca.k[j * D + hh * dh + c]; dk[j * D + hh * dh + c] += ds * ca.q[i * D + hh * dh + c]; } }
        }
        const da1 = TinyGPT.mmBack(ca.a, n, D, P.Wq, D, dq, G.Wq), da2 = TinyGPT.mmBack(ca.a, n, D, P.Wk, D, dk, G.Wk), da3 = TinyGPT.mmBack(ca.a, n, D, P.Wv, D, dv, G.Wv);
        for (let i = 0; i < n * D; i++) da1[i] += da2[i] + da3[i];
        const dhin = TinyGPT.lnBack(da1, ca.s1, n, D, P.g1, G.g1, G.n1);
        for (let i = 0; i < n * D; i++) dhin[i] += dh1[i];
        dh_ = dhin;
      }
      for (let t = 0; t < n; t++) for (let c = 0; c < D; c++) { this.gE[x[t] * D + c] += dh_[t * D + c]; this.gPe[t * D + c] += dh_[t * D + c]; }
    }
    return loss / cnt;
  }
  adam(lr, wd = 0) {
    this.t++; const b1 = 0.9, b2 = 0.99, c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    this.slots.forEach(([p, g], i) => { const M = this.M[i], S = this.S[i]; for (let j = 0; j < p.length; j++) { M[j] = b1 * M[j] + (1 - b1) * g[j]; S[j] = b2 * S[j] + (1 - b2) * g[j] * g[j]; p[j] -= lr * ((M[j] / c1) / (Math.sqrt(S[j] / c2) + 1e-8) + wd * p[j]); } });
  }
  evalLoss(data, from, to, maxWin = 40) {
    const T = this.T, wins = []; const span = to - from - T - 1; if (span <= 0) return NaN;
    const step = Math.max(1, Math.floor(span / maxWin)); for (let s = from; s + T + 1 <= to && wins.length < maxWin; s += step) wins.push(data.subarray(s, s + T + 1));
    return this.lossAndGrad(wins, false);
  }
  nextDist(ctx, temp = 1) {
    const x = Int32Array.from(ctx.slice(-this.T)), { logits } = this.forward(x), V = this.V, o = (x.length - 1) * V, z = Array.from(logits.subarray(o, o + V));
    return softmax(z, Math.max(temp, 1e-3));
  }
}
LAB.tests.gpt = { TinyGPT, geluF, geluD };

registerModule({
  id: 'm16', n: 16, title: 'Il transformer completo', eyebrow: 'capstone', heading: 'Un transformer vero, che si addestra qui',
  lede: 'Tutti i pezzi dei moduli precedenti, insieme: embedding, posizioni, LayerNorm, attention multi-head con maschera causale, MLP con GELU, residual, softmax, cross-entropy, backpropagation scritta a mano e Adam. Un GPT in miniatura da qualche migliaio di parametri, che impara a scrivere a caratteri.',
  build(root) {
    root.append(el('div', { html: `
      <p>Questo è il modello di tutto il corso, nella forma di un <b>decoder-only transformer</b> in stile GPT-2, ridotto al minimo: per ogni posizione <i>t</i> del testo, leggere i caratteri fino a <i>t</i> e prevedere il carattere <i>t</i>+1. Tutto il codice (forward <i>e</i> gradienti di ogni parametro) è scritto a mano in JavaScript, nella pagina, senza librerie; i test automatici confrontano ogni gradiente con le differenze finite.</p>
      <p>Il testo è lo stesso del modulo 8 (circa 1900 caratteri di prosa sul gatto Micio): pochissimo, quindi il modello <b>imparerà</b> molto ma <b>memorizzerà</b> anche. Serve a vedere i meccanismi, non a produrre testi belli.</p>` }));

    const chars = [...new Set(CORPUS)].sort(), data = Int32Array.from([...CORPUS].map(c => chars.indexOf(c))), V = chars.length, nTr = Math.floor(data.length * 0.9);
    let cfg = { D: 16, H: 2, L: 1, T: 16 }, net = null, running = false, steps = 0, hist = { tr: [], va: [], x: [] }, seed = 21;
    const MAXSTEPS = 4000;
    const sel = (opts, v, cb, label) => select(opts, v, cb, label);
    const dSel = sel([['16', 'd = 16'], ['24', 'd = 24'], ['32', 'd = 32']], '16', v => { cfg.D = +v; if (cfg.D % cfg.H) cfg.H = 1; resetNet(); }, 'dimensione d');
    const hSel = sel([['1', '1 head'], ['2', '2 head'], ['4', '4 head']], '2', v => { cfg.H = +v; resetNet(); }, 'head');
    const lSel = sel([['1', '1 layer'], ['2', '2 layer']], '1', v => { cfg.L = +v; resetNet(); }, 'layer');
    const sLR = slider({ label: 'learning rate', min: 0.002, max: 0.03, step: 0.001, value: 0.01, fmt: v => v.toFixed(3) });
    const cvL = canvasEl(230), out = el('div', { class: 'readout' });
    const cl = setupCanvas(cvL, (ctx, w, h, t) => {
      if (!hist.x.length) { text(ctx, 'premi "Allena" per vedere la loss scendere', 20, h / 2, { color: t.muted, size: 12 }); return; }
      const c = chart(ctx, w, h, t, [{ xs: hist.x, ys: hist.tr, color: t.accent, lw: 2.5, label: 'train' }, { xs: hist.x, ys: hist.va, color: t.amber, lw: 2.5, label: 'validazione (10% del testo mai visto)' }, { xs: [0, Math.max(100, hist.x[hist.x.length - 1])], ys: [Math.log(V), Math.log(V)], color: t.muted, dash: [5, 4], lw: 1.5 }], { xlabel: 'passi', ylabel: 'cross-entropy (nat)', ymin: 0, nx: 5 });
      text(ctx, 'ln V = ' + f2(Math.log(V), 2) + ' (modello che tira a caso)', c.x1 - 6, c.Y(Math.log(V)) - 5, { align: 'right', color: t.muted, size: 10 });
    });
    function resetNet() {
      running = false; steps = 0; hist = { tr: [], va: [], x: [] }; cfg.T = 16; net = new TinyGPT(V, cfg, 11); seed = 21;
      summary(); cl.redraw(); if (attUpd) attUpd(); if (genOut) genOut.textContent = '';
    }
    function summary(extra = '') {
      out.innerHTML = `parametri: <b>${net.nParams.toLocaleString('it-IT')}</b> (d=${cfg.D}, ${cfg.H} head, ${cfg.L} layer, contesto ${cfg.T} caratteri, vocabolario ${V}) · passi: <b>${steps}</b>${hist.tr.length ? ` · loss train <b>${f2(hist.tr[hist.tr.length - 1], 2)}</b>, validazione <b>${f2(hist.va[hist.va.length - 1], 2)}</b>` : ''}${extra}`;
    }
    const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
    function trainFrame() {
      if (!running) return;
      const t0 = Date.now(), T = net.T;
      while (Date.now() - t0 < 14 && steps < MAXSTEPS) {
        const batch = []; for (let b = 0; b < 8; b++) { const s = Math.floor(rnd() * (nTr - T - 1)); batch.push(data.subarray(s, s + T + 1)); }
        net.lossAndGrad(batch); net.adam(sLR.get()); steps++;
        if (steps % 25 === 0) { hist.x.push(steps); hist.tr.push(net.evalLoss(data, 0, nTr)); hist.va.push(net.evalLoss(data, nTr, data.length)); }
      }
      summary(steps >= MAXSTEPS ? ' · fine (limite di passi)' : ''); cl.redraw(); if (attUpd && steps % 50 === 0) attUpd();
      if (steps >= MAXSTEPS) { running = false; if (attUpd) attUpd(); return; }
      requestAnimationFrame(trainFrame);
    }
    const goBtn = button('Allena / riprendi', () => { if (steps >= MAXSTEPS) resetNet(); if (!running) { running = true; requestAnimationFrame(trainFrame); } }, 'primary');
    const stopBtn = button('Pausa', () => { running = false; if (attUpd) attUpd(); });

    /* attention view */
    const sample = el('input', { type: 'text', value: CORPUS.slice(0, 16), maxlength: '16', 'aria-label': 'testo da analizzare', style: 'font:500 14px var(--mono);padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);width:min(100%,250px)' });
    let aL = 0, aH = 0; const cvA = canvasEl(330), attOut = el('div', { class: 'readout' });
    const layerSel = sel([['0', 'layer 1'], ['1', 'layer 2']], '0', v => { aL = Math.min(+v, cfg.L - 1); attUpd(); }, 'layer');
    const headSel = sel([['0', 'head 1'], ['1', 'head 2'], ['2', 'head 3'], ['3', 'head 4']], '0', v => { aH = Math.min(+v, cfg.H - 1); attUpd(); }, 'head');
    const toIdx = s => Int32Array.from([...s].map(c => chars.indexOf(c)).filter(i => i >= 0).slice(0, 16));
    const ca = setupCanvas(cvA, (ctx, w, h, t) => {
      const x = toIdx(sample.value); if (x.length < 2) { text(ctx, 'scrivi almeno 2 caratteri noti', 20, h / 2, { color: t.muted, size: 12 }); return; }
      const { cache } = net.forward(x), n = x.length, L0 = Math.min(aL, cfg.L - 1), H0 = Math.min(aH, cfg.H - 1), p = cache.layers[L0].p;
      const M = range(n).map(i => range(n).map(j => j <= i ? p[(H0 * n + i) * n + j] : -Infinity)), cell = Math.min(24, (w - 120) / n, (h - 60) / n), x0 = 56, y0 = 36;
      heat(ctx, M, x0, y0, cell, cell, { t, vmin: 0, vmax: 1, rows: [...x].map(i => chars[i] === ' ' ? '␣' : chars[i]), cols: [...x].map(i => chars[i] === ' ' ? '␣' : chars[i]), num: cell > 26 });
      text(ctx, 'riga = carattere che "guarda" · colonna = carattere guardato · blu intenso = più attenzione · le celle barrate sono nel futuro (maschera causale)', 8, 14, { color: t.muted, size: 10 });
      const avg = (hd, off) => { const pp = cache.layers[L0].p; let s = 0, c = 0; for (let i = off; i < n; i++) { s += pp[(hd * n + i) * n + i - off]; c++; } return s / c; };
      attOut.innerHTML = `attenzione media di ogni head del layer ${L0 + 1} verso: <b>il carattere precedente</b> ${range(cfg.H).map(hd => 'h' + (hd + 1) + ' ' + f2(avg(hd, 1), 2)).join(' · ')} · <b>se stesso</b> ${range(cfg.H).map(hd => 'h' + (hd + 1) + ' ' + f2(avg(hd, 0), 2)).join(' · ')} (a caso: circa ${f2(range(n).reduce((s, i) => s + 1 / (i + 1), 0) / n, 2)})`;
    });
    function attUpd() { ca.redraw(); }
    sample.addEventListener('input', () => attUpd());

    /* generation */
    const sT = slider({ label: 'temperatura', min: 0.2, max: 1.5, step: 0.05, value: 0.8, fmt: v => v.toFixed(2) });
    const prompt = el('input', { type: 'text', value: 'il gatto ', 'aria-label': 'inizio del testo', style: 'font:500 14px var(--mono);padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);width:min(100%,250px)' });
    const genOut = el('div', { class: 'readout', style: 'white-space:pre-wrap;min-height:5em;border:1px dashed var(--line);border-radius:8px;padding:10px 12px;background:var(--surface)' });
    function generate() {
      let ctxs = [...prompt.value].map(c => chars.indexOf(c)).filter(i => i >= 0); if (!ctxs.length) ctxs = [chars.indexOf(' ')];
      const T0 = sT.get(); let s = prompt.value;
      for (let k = 0; k < 140; k++) { const p = net.nextDist(ctxs, T0), u = Math.random(); let a = 0, ix = p.length - 1; for (let i = 0; i < p.length; i++) { a += p[i]; if (u <= a) { ix = i; break; } } ctxs.push(ix); s += chars[ix]; }
      genOut.textContent = s;
    }

    root.append(lab('Laboratorio · addestra il transformer', controls(dSel, hSel, lSel, sLR.el), controls(goBtn, stopBtn, button('Reinizializza', () => resetNet())), cvL, out,
      el('p', { class: 'muted', html: 'Ogni passo usa 8 finestre da 16 caratteri (128 previsioni) e un passo di Adam. Il tempo per passo dipende dalle dimensioni: con d = 32 e 2 layer è parecchie volte più lento di quello predefinito. Cambiare una dimensione riparte da zero. Nel modello predefinito la validazione tocca il minimo dopo qualche centinaio di passi e poi <b>risale</b> mentre il training continua a scendere: con ~1900 caratteri il modello comincia presto a memorizzare, e un modello più grande lo fa ancora prima. È il fenomeno del modulo 8, qui con un transformer; fermarsi al minimo (<i>early stopping</i>) è la cura più semplice.' })));
    root.append(lab('Laboratorio · che cosa guarda l\'attention (con il modello come è ora)', controls(layerSel, headSel, sample), cvA, attOut,
      el('p', { class: 'muted', html: 'Prima dell\'addestramento le righe sono quasi uniformi (ogni carattere guarda a caso i precedenti). Nel modello predefinito, dopo 4000 passi, una delle due head dedica circa metà della sua attenzione al carattere precedente, contro il 20% circa che avrebbe a caso (le cifre esatte cambiano con la configurazione e dal punto di partenza: il pannello qui sopra le ricalcola). Con un modello così piccolo e un testo così corto i pattern non sono netti come nelle head dei modelli grandi: è ciò che il modello ha imparato <i>qui</i>, non una dimostrazione generale.' })));
    root.append(lab('Laboratorio · generazione', controls(sT.el), el('div', { class: 'controls' }, prompt, button('Genera 140 caratteri', generate, 'primary')), genOut,
      el('p', { class: 'muted', html: 'Il modello produce un carattere alla volta, campionando dalla distribuzione prevista (modulo 7) e rimettendo il carattere in ingresso (modulo 9, senza KV cache per semplicità). Il testo ha l\'aspetto dell\'italiano (sillabe e parole quasi plausibili); dopo molti passi ripete interi frammenti del corpus, segno di memorizzazione. Non aspettarti frasi sensate.' })));
    resetNet(); attUpd();

    root.append(el('h2', { text: 'Il modello, riga per riga' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Embedding + posizione</b> (moduli 2, 3): <var>h</var><sub>t</sub> = <var>E</var>[<var>x</var><sub>t</sub>] + <var>P</var>[<var>t</var>]. Le posizioni sono qui appresi, come in GPT-2.</li>
      <li><b>Blocco</b> (moduli 4, 5, 6), ripetuto <var>L</var> volte: <var>h</var> ← <var>h</var> + Attn(LN(<var>h</var>)), poi <var>h</var> ← <var>h</var> + MLP(LN(<var>h</var>)). LayerNorm <i>prima</i> di ogni sotto-blocco (pre-LN), connessioni residual, attention causale multi-head, MLP con GELU di larghezza 3<var>d</var>.</li>
      <li><b>Uscita</b> (modulo 7): LN finale e logit = <var>f</var><sub>t</sub> · <var>E</var><sup>T</sup>: la stessa matrice dell'embedding riusata come unembedding (<i>weight tying</i>).</li>
      <li><b>Loss e training</b> (modulo 8): cross-entropy media su tutte le posizioni, gradienti per backpropagation, Adam.</li>
      <li><b>Inferenza</b> (moduli 7, 9): campionamento con temperatura, un carattere alla volta.</li>
      <li>Quello che <b>manca</b> rispetto a un LLM vero: tokenizer BPE (qui sono caratteri), milioni di volte più parametri e dati, ottimizzazione su GPU, post-training (modulo 10), e tutto il resto del corso su come si scala e si usa.</li>` }));

    root.append(details('math', 'La matematica: la backpropagation attraverso il blocco', 'formule', `
      <p>I gradienti si ottengono applicando la regola della catena a ritroso lungo lo stesso grafo del forward. I passaggi non banali:</p>
      <p><b>Matrici.</b> Per <var>C</var> = <var>A</var><var>B</var>: ∂ℒ/∂<var>A</var> = (∂ℒ/∂<var>C</var>) <var>B</var><sup>T</sup> e ∂ℒ/∂<var>B</var> = <var>A</var><sup>T</sup> (∂ℒ/∂<var>C</var>).</p>
      <p><b>Softmax dell'attention.</b> Con <var>p</var> = softmax(<var>s</var>) per riga e <var>dp</var> = ∂ℒ/∂<var>p</var>:</p>
      <span class="eq"><var>ds</var><sub>j</sub> = <var>p</var><sub>j</sub> ( <var>dp</var><sub>j</sub> − Σ<sub>j′</sub> <var>p</var><sub>j′</sub> <var>dp</var><sub>j′</sub> )</span>
      <p>e poi, con <var>s</var><sub>ij</sub> = <var>q</var><sub>i</sub>·<var>k</var><sub>j</sub>/√<var>d</var><sub>h</sub>: <var>dq</var><sub>i</sub> = Σ<sub>j</sub> <var>ds</var><sub>ij</sub> <var>k</var><sub>j</sub>/√<var>d</var><sub>h</sub>, <var>dk</var><sub>j</sub> = Σ<sub>i</sub> <var>ds</var><sub>ij</sub> <var>q</var><sub>i</sub>/√<var>d</var><sub>h</sub>, e per i value <var>dv</var><sub>j</sub> = Σ<sub>i</sub> <var>p</var><sub>ij</sub> <var>do</var><sub>i</sub>. La maschera causale si ottiene sommando solo su <var>j</var> ≤ <var>i</var>.</p>
      <p><b>LayerNorm.</b> Con <var>x̂</var> = (<var>x</var> − μ)/σ e <var>y</var> = γ<var>x̂</var> + β, ponendo <var>d</var><var>x̂</var> = <var>dy</var>·γ:</p>
      <span class="eq"><var>dx</var> = ${frac('1', 'σ')} ( <var>d</var><var>x̂</var> − media(<var>d</var><var>x̂</var>) − <var>x̂</var> · media(<var>d</var><var>x̂</var> ⊙ <var>x̂</var>) )</span>
      <p><b>Residual.</b> Per <var>h</var>′ = <var>h</var> + <var>F</var>(<var>h</var>): ∂ℒ/∂<var>h</var> = ∂ℒ/∂<var>h</var>′ + (∂<var>F</var>/∂<var>h</var>)<sup>T</sup> ∂ℒ/∂<var>h</var>′. Il termine identità è la "autostrada" del gradiente del modulo 6.</p>
      <p><b>Weight tying.</b> Poiché <var>E</var> è usata in ingresso e in uscita, il suo gradiente è la <i>somma</i> dei due contributi (il codice li accumula nello stesso array).</p>
      <p><b>Parametri.</b> Con vocabolario <var>V</var>, contesto <var>T</var>, dimensione <var>d</var>, MLP <var>F</var>, <var>L</var> layer: <var>V</var><var>d</var> + <var>T</var><var>d</var> + 2<var>d</var> + <var>L</var>(4<var>d</var>² + 2<var>d</var><var>F</var> + <var>F</var> + 5<var>d</var>) (due LayerNorm da 2<var>d</var>, bias <var>c</var><sub>1</sub>, <var>c</var><sub>2</sub>).</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'A cosa serve la maschera causale nel training?', opts: ['A velocizzare il calcolo', 'A impedire che la previsione del carattere t+1 usi i caratteri successivi, così tutte le posizioni si addestrano in parallelo senza barare', 'A ridurre il vocabolario'], ans: 1, why: 'Senza maschera il modello potrebbe copiare la risposta dall\'ingresso. Con la maschera, un solo passaggio dà una previsione valida per ogni posizione.' },
      { q: 'Perché il gradiente di E nel weight tying è una somma?', opts: ['Perché E è molto grande', 'Perché E influisce sulla loss in due punti (ingresso e uscita) e i contributi si sommano', 'Perché Adam lo richiede'], ans: 1, why: 'Per la regola della catena, un parametro usato in più posti riceve la somma dei gradienti di ogni uso.' },
      { q: 'La loss di validazione smette di scendere mentre quella di training continua a calare. Cosa succede?', opts: ['Il modello sta imparando meglio', 'Il modello sta memorizzando il testo di training (overfitting)', 'Il learning rate è troppo basso'], ans: 1, why: 'Il modello si adatta a dettagli del training che non generalizzano: con un corpus di ~1900 caratteri succede presto.' },
    ]));
  },
});
