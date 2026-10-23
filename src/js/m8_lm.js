'use strict';
/* A small but real neural language model (character level, Bengio-style MLP) with hand-written backpropagation and Adam. */

const CORPUS = "il gatto dorme sul divano e il sole entra dalla finestra. la mattina il gatto si sveglia, si stiracchia e va in cucina a cercare il latte. in cucina c'è una ciotola piena d'acqua e un piatto con il pesce fresco. il gatto mangia con calma, poi esce in giardino. nel giardino ci sono i fiori, l'erba e un vecchio albero. il gatto sale sull'albero e guarda il mare lontano. il mare è calmo e blu, le barche tornano al porto con il pesce. i pescatori scaricano le casse e le portano al mercato. al mercato la gente compra il pesce, il pane e la frutta. una bambina compra una mela rossa e la mangia sulla strada. la strada porta alla piazza, dove c'è una fontana antica. nella piazza i vecchi parlano del tempo e i bambini giocano con la palla. la sera il vento diventa freddo e il gatto torna a casa. la bambina accende la luce, prepara la cena e racconta al gatto la sua giornata. il gatto ascolta e fa le fusa. fuori la luna illumina la strada e la città dorme. domani il sole tornerà e il gatto si sveglierà di nuovo sul divano. la vita nella piccola città è lenta e tranquilla. ogni giorno le stesse voci, gli stessi profumi, lo stesso mare. il gatto conosce ogni angolo del giardino e ogni suono della casa. quando piove resta vicino alla finestra e guarda le gocce scendere sul vetro. quando c'è il sole si distende sul muro caldo e chiude gli occhi. la bambina cresce e ogni anno il gatto diventa un po' più vecchio, ma la sera si siede sempre sulle sue ginocchia. in inverno la neve copre i tetti e il porto è silenzioso. le barche restano ferme e i pescatori riparano le reti nelle loro case. in primavera tornano i fiori e le rondini volano sopra la piazza. d'estate la gente va al mare, mangia il gelato e dorme sotto gli ombrelloni. in autunno si raccolgono le mele e si prepara il vino nuovo. così passano le stagioni e la città cambia colore, ma il mare resta sempre lo stesso.";

class CharLM {
  constructor(V, C, Em, H, seed = 7) {
    Object.assign(this, { V, C, Em, H }); const r = rng(seed), init = (n, s) => Float64Array.from({ length: n }, () => r.gauss() * s);
    this.E = init(V * Em, 0.3); this.W1 = init(C * Em * H, 1 / Math.sqrt(C * Em)); this.b1 = new Float64Array(H);
    this.W2 = init(H * V, 0.5 / Math.sqrt(H)); this.b2 = new Float64Array(V);
    this.P = [this.E, this.W1, this.b1, this.W2, this.b2];
    this.G = this.P.map(p => new Float64Array(p.length)); this.M = this.P.map(p => new Float64Array(p.length)); this.S = this.P.map(p => new Float64Array(p.length)); this.t = 0;
    this.x = new Float64Array(C * Em); this.a = new Float64Array(H); this.z = new Float64Array(V);
  }
  get nParams() { return this.P.reduce((s, p) => s + p.length, 0); }
  forward(ctx) {
    const { C, Em, H, V, E, W1, b1, W2, b2, x, a, z } = this;
    for (let c = 0; c < C; c++) for (let k = 0; k < Em; k++) x[c * Em + k] = E[ctx[c] * Em + k];
    for (let j = 0; j < H; j++) a[j] = b1[j];
    for (let i = 0; i < C * Em; i++) { const xi = x[i], row = i * H; for (let j = 0; j < H; j++) a[j] += xi * W1[row + j]; }
    for (let j = 0; j < H; j++) a[j] = Math.tanh(a[j]);
    for (let v = 0; v < V; v++) z[v] = b2[v];
    for (let j = 0; j < H; j++) { const aj = a[j], row = j * V; for (let v = 0; v < V; v++) z[v] += aj * W2[row + v]; }
    let m = -Infinity; for (let v = 0; v < V; v++) if (z[v] > m) m = z[v];
    const p = new Float64Array(V); let s = 0; for (let v = 0; v < V; v++) { p[v] = Math.exp(z[v] - m); s += p[v]; }
    for (let v = 0; v < V; v++) p[v] /= s; return p;
  }
  /* mean cross-entropy over a batch; fills this.G with gradients */
  lossAndGrad(ctxs, ys) {
    const { C, Em, H, V, W1, W2 } = this, [gE, gW1, gb1, gW2, gb2] = this.G, B = ys.length, dz = new Float64Array(H), dx = new Float64Array(C * Em), dl = new Float64Array(V);
    this.G.forEach(g => g.fill(0)); let loss = 0;
    for (let n = 0; n < B; n++) {
      const ctx = ctxs[n], y = ys[n], p = this.forward(ctx); loss -= Math.log(p[y] + 1e-12);
      for (let v = 0; v < V; v++) dl[v] = (p[v] - (v === y ? 1 : 0)) / B;
      for (let v = 0; v < V; v++) gb2[v] += dl[v];
      dz.fill(0);
      for (let j = 0; j < H; j++) { const aj = this.a[j], row = j * V; let da = 0; for (let v = 0; v < V; v++) { gW2[row + v] += aj * dl[v]; da += W2[row + v] * dl[v]; } dz[j] = da * (1 - aj * aj); }
      for (let j = 0; j < H; j++) gb1[j] += dz[j];
      dx.fill(0);
      for (let i = 0; i < C * Em; i++) { const xi = this.x[i], row = i * H; let s = 0; for (let j = 0; j < H; j++) { gW1[row + j] += xi * dz[j]; s += W1[row + j] * dz[j]; } dx[i] = s; }
      for (let c = 0; c < C; c++) for (let k = 0; k < Em; k++) gE[ctx[c] * Em + k] += dx[c * Em + k];
    }
    return loss / B;
  }
  adam(lr, b1 = 0.9, b2 = 0.999) {
    this.t++; const c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t;
    this.P.forEach((p, q) => { const g = this.G[q], m = this.M[q], s = this.S[q]; for (let i = 0; i < p.length; i++) { m[i] = b1 * m[i] + (1 - b1) * g[i]; s[i] = b2 * s[i] + (1 - b2) * g[i] * g[i]; p[i] -= lr * (m[i] / c1) / (Math.sqrt(s[i] / c2) + 1e-8); } });
  }
  evalLoss(data, lo, hi, maxN = 400) {
    const C = this.C, n = hi - lo - C; if (n <= 0) return NaN; const stride = Math.max(1, Math.floor(n / maxN)); let s = 0, k = 0;
    for (let i = lo + C; i < hi; i += stride) { s -= Math.log(this.forward(data.subarray(i - C, i))[data[i]] + 1e-12); k++; }
    return s / k;
  }
  generate(ctx0, n, temp, seed = 123) {
    const r = rng(seed), out = [], ctx = ctx0.slice();
    for (let i = 0; i < n; i++) {
      const p = this.forward(ctx), q = softmax(Array.from(p, v => Math.log(v + 1e-12)), temp); let u = r(), k = 0, acc = q[0];
      while (u > acc && k < q.length - 1) { k++; acc += q[k]; }
      out.push(k); ctx.shift(); ctx.push(k);
    }
    return out;
  }
}
LAB.tests.CharLM = CharLM; LAB.tests.CORPUS = CORPUS;

function buildLMLab(root) {
  const chars = [...new Set(CORPUS)].sort(), V = chars.length, stoi = new Map(chars.map((c, i) => [c, i]));
  const data = Int32Array.from([...CORPUS].map(c => stoi.get(c))), nTrain = Math.floor(data.length * 0.88), spaceId = stoi.get(' ');
  const showCh = c => c === ' ' ? '␣' : c;
  /* reference: what a pure bigram table would reach on the training part (in-sample conditional entropy) */
  const cnt = new Map(), ctot = new Map();
  for (let i = 1; i < nTrain; i++) { const k = data[i - 1] * 100 + data[i]; cnt.set(k, (cnt.get(k) || 0) + 1); ctot.set(data[i - 1], (ctot.get(data[i - 1]) || 0) + 1); }
  let hb = 0; cnt.forEach((c, k) => { hb -= c / (nTrain - 1) * Math.log(c / ctot.get(Math.floor(k / 100))); });
  const cfg = { C: 3, Em: 8, H: 64, lr: 0.01, spf: 10 };
  let lm, step, ema, hist, rb, lastViz = 0;
  const encode = s => [...s].map(c => stoi.has(c) ? stoi.get(c) : spaceId);
  const ctxOf = s => { const e = encode(s); while (e.length < cfg.C) e.unshift(spaceId); return e.slice(-cfg.C); };

  const promptIn = el('input', { type: 'text', id: 'lm-prompt', value: 'il gatto ' }); promptIn.style.width = '100%';
  const sT = slider({ label: 'temperatura', min: 0.3, max: 1.5, step: 0.1, value: 0.8, fmt: v => v.toFixed(1), onInput: () => samples() });
  const sC = slider({ label: 'contesto (caratteri)', min: 1, max: 6, value: cfg.C, onInput: v => { cfg.C = v; reset(); } });
  const sE = slider({ label: 'dim. embedding', min: 2, max: 16, value: cfg.Em, onInput: v => { cfg.Em = v; reset(); } });
  const sH = slider({ label: 'neuroni nascosti', min: 8, max: 128, step: 8, value: cfg.H, onInput: v => { cfg.H = v; reset(); } });
  const sLR = slider({ label: 'learning rate', min: 0.002, max: 0.05, step: 0.002, value: cfg.lr, fmt: v => v.toFixed(3), onInput: v => { cfg.lr = v; } });
  const sSp = slider({ label: 'passi per frame', min: 1, max: 60, value: cfg.spf, onInput: v => { cfg.spf = v; } });
  const lossCv = canvasEl(260), embCv = canvasEl(280);
  const stat = el('div', { class: 'readout' }), gen = el('div', { class: 'readout' }), topk = el('div', { class: 'readout' });
  gen.style.cssText = 'white-space:pre-wrap;min-height:7em;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px';

  const text_ = (ctx, s, x, y, t) => text(ctx, s, x, y, { align: 'center', color: t.muted, size: 13 });
  const lc = setupCanvas(lossCv, (ctx, w, h, t) => {
    if (!hist || hist.x.length < 2) { text_(ctx, 'premi "Allena" per iniziare', w / 2, h / 2, t); return; }
    const hi = Math.log(V) * 1.05, S = [
      { xs: [0, Math.max(step, 1)], ys: [Math.log(V), Math.log(V)], color: t.muted, dash: [3, 5], lw: 1, label: 'caso (ln V)' },
      { xs: [0, Math.max(step, 1)], ys: [hb, hb], color: t.teal, dash: [6, 4], lw: 1.5, label: 'tabella bigram (in-sample)' },
      { xs: hist.x, ys: hist.tr, color: t.accent, lw: 2.5, label: 'train' }, { xs: hist.x, ys: hist.va, color: t.amber, lw: 2.5, label: 'validazione' }];
    chart(ctx, w, h, t, S, { xlabel: 'passi di training', ylabel: 'cross-entropy (nat)', ymin: 0, ymax: hi, xmin: 0, xmax: Math.max(step, 50) });
  });
  const ec = setupCanvas(embCv, (ctx, w, h, t) => {
    if (!lm) return; const rows = range(V).map(i => Array.from(lm.E.subarray(i * lm.Em, (i + 1) * lm.Em))), P = pca(rows, 2).proj, m = Math.max(...P.flat().map(Math.abs)) || 1, cx = w / 2, cy = h / 2, s = Math.min(w, h) * 0.42 / m;
    ctx.strokeStyle = t.line; ctx.beginPath(); ctx.moveTo(8, cy); ctx.lineTo(w - 8, cy); ctx.moveTo(cx, 8); ctx.lineTo(cx, h - 8); ctx.stroke();
    chars.forEach((c, i) => {
      const cls = c === ' ' ? t.teal : /[aeiouàèéìòù]/.test(c) ? t.accent : /[a-z]/.test(c) ? t.amber : t.rose;
      text(ctx, showCh(c), cx + P[i][0] * s, cy - P[i][1] * s + 5, { align: 'center', color: cls, size: 15, weight: 700 });
    });
    text(ctx, 'embedding dei caratteri (PCA 2D) · blu vocali · arancio consonanti · verde spazio · rosa punteggiatura', 8, h - 6, { color: t.muted, size: 10 });
  });

  function reset() {
    lm = new CharLM(V, cfg.C, cfg.Em, cfg.H, 7); step = 0; lastViz = 0; ema = null; rb = rng(99); hist = { x: [], tr: [], va: [] };
    record(); samples(); status(); lc.redraw(); ec.redraw();
  }
  function record() {
    const tr = lm.evalLoss(data, 0, nTrain), va = lm.evalLoss(data, nTrain, data.length); hist.x.push(step); hist.tr.push(tr); hist.va.push(va); return [tr, va];
  }
  function samples() {
    if (!lm) return;
    const out = lm.generate(ctxOf(promptIn.value), 220, sT.get()); gen.textContent = promptIn.value + out.map(i => chars[i]).join('');
    const p = lm.forward(ctxOf(promptIn.value)), top = range(V).sort((a, b) => p[b] - p[a]).slice(0, 5);
    topk.innerHTML = `Dopo "<b>${promptIn.value.slice(-cfg.C).replace(/ /g, '␣')}</b>" il modello prevede: ` + top.map(i => `<b>${showCh(chars[i])}</b> ${f2(p[i] * 100, 0)}%`).join(' · ');
  }
  function status() {
    const tr = hist.tr[hist.tr.length - 1], va = hist.va[hist.va.length - 1];
    stat.innerHTML = `passo <b>${step}</b> · parametri <b>${lm.nParams.toLocaleString('it-IT')}</b> · vocabolario ${V} caratteri · loss train <b>${f2(tr, 3)}</b> · validazione <b>${f2(va, 3)}</b> (perplessità ${f2(Math.exp(va), 1)}) · caso = ln ${V} = ${f2(Math.log(V), 3)}`;
  }
  function trainSteps(n) {
    for (let k = 0; k < n; k++) {
      const ctxs = [], ys = [];
      for (let b = 0; b < 32; b++) { const i = cfg.C + Math.floor(rb() * (nTrain - cfg.C)); ctxs.push(data.subarray(i - cfg.C, i)); ys.push(data[i]); }
      const l = lm.lossAndGrad(ctxs, ys); lm.adam(cfg.lr); step++; ema = ema == null ? l : 0.95 * ema + 0.05 * l;
    }
    record(); status(); lc.redraw();
    if (step - lastViz >= 20 || n > 20) { lastViz = step; samples(); ec.redraw(); }
  }
  const lp = loop(lossCv, () => { trainSteps(cfg.spf); if (step >= 20000) { lp.stop(); playBtn.textContent = 'Allena'; } });
  const playBtn = button('Allena', () => { if (lp.running) { lp.stop(); playBtn.textContent = 'Allena'; } else { lp.start(); playBtn.textContent = 'Pausa'; } }, 'primary');
  promptIn.addEventListener('input', samples);

  root.append(lab('Laboratorio · un language model che si allena davvero, qui nel browser',
    el('p', { html: `Il testo è un breve racconto in italiano (${CORPUS.length} caratteri, 88% per il training e 12% per la validazione). Il modello vede gli ultimi <var>C</var> caratteri, li trasforma in embedding, li passa a un MLP e produce una distribuzione sul carattere successivo: la stessa struttura "contesto → logit → softmax → cross-entropy" di un LLM, con un'MLP al posto dei transformer block. Tutto (forward, backpropagation, Adam) è codice scritto a mano in questa pagina.` }),
    controls(sC.el, sE.el, sH.el), controls(sLR.el, sSp.el), controls(playBtn, button('+100 passi', () => trainSteps(100)), button('Reset', () => { lp.stop(); playBtn.textContent = 'Allena'; reset(); })),
    stat, lossCv,
    el('p', { html: '<b>Come leggerlo.</b> All\'inizio la loss è ≈ ln V: il modello non sa nulla. In poche centinaia di passi la curva blu (training) scende sotto la linea verde, la <i>tabella bigram</i> calcolata sullo stesso training set, perché il modello vede più di un carattere di contesto. Ma il testo è minuscolo: la validazione (arancione) tocca il minimo molto presto e poi <b>risale</b> mentre il training continua a calare. Il modello sta <b>memorizzando</b> il testo (overfitting). Prova ad aumentare il contesto: memorizza ancora più in fretta; con contesto = 1 non può memorizzare e le due curve restano vicine, ma alte. Gli LLM veri evitano questo con corpora di migliaia di miliardi di token.' }),
    el('div', { class: 'grid2' },
      el('div', {}, el('div', { class: 'lab-title', style: 'margin-bottom:8px', text: 'Testo generato (aggiornato durante il training)' }), promptIn, controls(sT.el), gen, topk),
      el('div', {}, el('div', { class: 'lab-title', style: 'margin-bottom:8px', text: 'Spazio degli embedding appreso' }), embCv))));
  reset();
  return { lm: () => lm };
}
