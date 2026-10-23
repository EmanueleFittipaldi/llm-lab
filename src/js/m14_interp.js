'use strict';
/* Module 14 - interpretability: a toy model of superposition (trained in the browser) and linear probes on a real small LM */

/* Toy model of superposition (after Elhage et al. 2022): n sparse features, m < n hidden dims.
   h = W x,  x^ = ReLU(W^T h + b),  loss = sum_i I_i (x_i - x^_i)^2 with importance I_i = decay^i. */
class SuperToy {
  constructor(n, m, S, decay, seed = 1) {
    Object.assign(this, { n, m, S, decay }); this.r = rng(seed);
    this.W = Float64Array.from({ length: m * n }, () => this.r.gauss() * 0.5); this.b = new Float64Array(n);
    this.I = Float64Array.from({ length: n }, (_, i) => Math.pow(decay, i));
    this.gW = new Float64Array(m * n); this.gb = new Float64Array(n);
    this.mW = new Float64Array(m * n); this.vW = new Float64Array(m * n); this.mb = new Float64Array(n); this.vb = new Float64Array(n); this.t = 0;
  }
  sample(B) { const { n, S, r } = this; return Array.from({ length: B }, () => Float64Array.from({ length: n }, () => r() < S ? 0 : r())); }
  lossGrad(X) {
    const { n, m, W, b, I, gW, gb } = this, B = X.length, h = new Float64Array(m), dz = new Float64Array(n), dh = new Float64Array(m);
    gW.fill(0); gb.fill(0); let loss = 0;
    for (const x of X) {
      h.fill(0); for (let k = 0; k < m; k++) for (let j = 0; j < n; j++) h[k] += W[k * n + j] * x[j];
      dh.fill(0);
      for (let i = 0; i < n; i++) {
        let z = b[i]; for (let k = 0; k < m; k++) z += W[k * n + i] * h[k];
        const xh = z > 0 ? z : 0, e = x[i] - xh; loss += I[i] * e * e;
        dz[i] = z > 0 ? -2 * I[i] * e / B : 0; gb[i] += dz[i];
        for (let k = 0; k < m; k++) { gW[k * n + i] += h[k] * dz[i]; dh[k] += W[k * n + i] * dz[i]; }
      }
      for (let k = 0; k < m; k++) for (let j = 0; j < n; j++) gW[k * n + j] += dh[k] * x[j];
    }
    return loss / B;
  }
  step(B = 256, lr = 0.03) {
    const loss = this.lossGrad(this.sample(B)); this.t++; const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    const upd = (p, g, M, V) => { for (let i = 0; i < p.length; i++) { M[i] = b1 * M[i] + (1 - b1) * g[i]; V[i] = b2 * V[i] + (1 - b2) * g[i] * g[i]; p[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8); } };
    upd(this.W, this.gW, this.mW, this.vW); upd(this.b, this.gb, this.mb, this.vb); return loss;
  }
  norms() { const { n, m, W } = this; return Array.from({ length: n }, (_, i) => Math.hypot(...Array.from({ length: m }, (_, k) => W[k * n + i]))); }
  gram() { const { n, m, W } = this; return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => { let s = 0; for (let k = 0; k < m; k++) s += W[k * n + i] * W[k * n + j]; return s; })); }
}

/* linear probes on the hidden layer of the character-level MLP from module 8 */
const isVowelCh = c => 'aeiouàèìù'.includes(c), isPunctCh = c => ' ,.\''.includes(c);
function trainCharLM(data, V, steps, seed = 99, model = null) {
  const lm = model || new CharLM(V, 3, 8, 64, 7); let q = seed; const rr = () => (q = (q * 1664525 + 1013904223) % 4294967296) / 4294967296;
  for (let s = 0; s < steps; s++) { const cx = [], yy = []; for (let b = 0; b < 32; b++) { const i = 3 + Math.floor(rr() * (data.length - 3)); cx.push(data.subarray(i - 3, i)); yy.push(data[i]); } lm.lossAndGrad(cx, yy); lm.adam(0.01); }
  return lm;
}
function probeCollect(model, data, chars, pos, n, seed, prop) {
  let q = seed; const r = () => (q = (q * 1664525 + 1013904223) % 4294967296) / 4294967296, X = [], y = [];
  for (let k = 0; k < n; k++) { const i = 3 + Math.floor(r() * (data.length - 3)), cx = data.subarray(i - 3, i); model.forward(cx); X.push(Array.from(model.a)); y.push(prop(chars[cx[3 + pos]]) ? 1 : 0); }
  return { X, y };
}
function probeFit(X, y, steps = 150, lr = 0.3, l2 = 0.001) {
  const d = X[0].length; let w = new Array(d).fill(0), b = 0;
  for (let s = 0; s < steps; s++) {
    const g = new Array(d).fill(0); let gb = 0;
    for (let i = 0; i < X.length; i++) { let z = b; for (let k = 0; k < d; k++) z += X[i][k] * w[k]; const e = 1 / (1 + Math.exp(-z)) - y[i]; for (let k = 0; k < d; k++) g[k] += e * X[i][k]; gb += e; }
    for (let k = 0; k < d; k++) w[k] -= lr * (g[k] / X.length + l2 * w[k]); b -= lr * gb / X.length;
  }
  return { w, b };
}
const probeAcc = (m, X, y) => X.reduce((s, x, i) => { let z = m.b; for (let k = 0; k < x.length; k++) z += x[k] * m.w[k]; return s + ((z > 0 ? 1 : 0) === y[i] ? 1 : 0); }, 0) / X.length;
function shuffled(a, seed) { let q = seed; const r = () => (q = (q * 1664525 + 1013904223) % 4294967296) / 4294967296, b = a.slice(); for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; }
function probeRun(trained, fresh, data, chars, pos, prop, nTrain = 1500, nTest = 800) {
  const out = {};
  for (const [nm, model] of [['trained', trained], ['untrained', fresh]]) {
    const tr = probeCollect(model, data, chars, pos, nTrain, 5, prop), te = probeCollect(model, data, chars, pos, nTest, 6, prop), m = probeFit(tr.X, tr.y);
    out[nm] = probeAcc(m, te.X, te.y);
    if (nm === 'trained') {
      const mean = a => sum(a) / a.length; out.majority = Math.max(mean(te.y), 1 - mean(te.y));
      out.control = probeAcc(probeFit(tr.X, shuffled(tr.y, 3)), te.X, shuffled(te.y, 4));
    }
  }
  return out;
}
LAB.tests.interp = { SuperToy, trainCharLM, probeRun, isVowelCh, isPunctCh };

registerModule({
  id: 'm14', n: 14, title: 'Interpretabilità', eyebrow: 'guardare dentro', heading: 'Interpretabilità: che cosa c\'è nei numeri',
  lede: 'Sappiamo scrivere le equazioni di un LLM, ma non sappiamo dire in generale che cosa calcola. L\'interpretabilità prova a leggere i pesi e le attivazioni come si leggerebbe un programma. Qui vediamo due idee reali, su modelli piccoli che si addestrano nella pagina.',
  build(root) {
    root.append(el('div', { html: `
      <p>Un vettore di attivazioni a 4096 dimensioni non ha dimensioni etichettate: nessuna coordinata significa "è una domanda" o "parla di Parigi". Eppure il modello <i>sembra</i> rappresentare molti più concetti di quante coordinate abbia. Come è possibile? La spiegazione più discussa è la <b>superposition</b>: i concetti sono <b>direzioni</b> nello spazio, non coordinate, e se i concetti compaiono di rado (sono <i>sparsi</i>) ne possono stare molti di più delle dimensioni, a costo di una piccola interferenza.</p>` }));

    /* ---- superposition ---- */
    const NF = 5, MF = 2; let toy = null, hist = [], running = false, stepsDone = 0;
    const sS = slider({ label: 'sparsità S (probabilità che una feature sia a zero)', min: 0, max: 0.99, step: 0.01, value: 0.9, fmt: v => v.toFixed(2), onInput: () => reset() });
    const sD = slider({ label: 'calo di importanza', min: 0.5, max: 1, step: 0.05, value: 0.7, fmt: v => v.toFixed(2), onInput: () => reset() });
    const cvS = canvasEl(300), cvN = canvasEl(200), out = el('div', { class: 'readout' });
    const FC = t => [t.accent, t.amber, t.teal, t.rose, '#8b6fb0'];
    const cS = setupCanvas(cvS, (ctx, w, h, t) => {
      if (!toy) return;
      const cols = FC(t), R = Math.min(h, w * 0.55) * 0.42, cx = Math.min(w * 0.3, R + 20), cy = h / 2;
      ctx.strokeStyle = t.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx - R - 8, cy); ctx.lineTo(cx + R + 8, cy); ctx.moveTo(cx, cy - R - 8); ctx.lineTo(cx, cy + R + 8); ctx.stroke();
      const ns = toy.norms();
      for (let i = 0; i < NF; i++) {
        const x = toy.W[i], y = toy.W[NF + i], len = ns[i], alive = len > 0.35;
        if (!alive) { ctx.fillStyle = rgba(cols[i], 0.5); ctx.beginPath(); ctx.arc(cx + x * R, cy - y * R, 3, 0, 7); ctx.fill(); continue; }
        arrow(ctx, cx, cy, cx + x * R, cy - y * R, cols[i], 3, 9);
        text(ctx, 'f' + (i + 1), cx + x * R * 1.12 + (x >= 0 ? 4 : -4), cy - y * R * 1.12 + 4, { align: x >= 0 ? 'left' : 'right', color: cols[i], size: 13, weight: 700 });
      }
      text(ctx, 'ogni freccia: dove una feature viene scritta nel piano 2D (vettore W[:, i])', 8, 14, { color: t.muted, size: 10 });
      const gx = cx + R + 60, gw = Math.max(80, w - gx - 14), M = toy.gram(), cell = Math.min(34, gw / NF, (h - 70) / NF);
      if (gw > 90) {
        text(ctx, 'interferenza WᵀW', gx, 30, { color: t.muted, size: 11 });
        heat(ctx, M, gx + 22, 48, cell, cell, { t, diverging: true, vmin: -1.2, vmax: 1.2, num: true, rows: ['f1', 'f2', 'f3', 'f4', 'f5'], cols: ['f1', 'f2', 'f3', 'f4', 'f5'], numFmt: v => f2(v, 1) });
        text(ctx, 'diagonale = lunghezza² · fuori diagonale = interferenza', gx, 48 + NF * cell + 18, { color: t.muted, size: 10 });
      }
    });
    const cN = setupCanvas(cvN, (ctx, w, h, t) => {
      if (!hist.length) { text(ctx, 'premi "Allena" per vedere la loss', 20, h / 2, { color: t.muted, size: 12 }); return; }
      const xs = hist.map((_, i) => (i + 1) * 25);
      chart(ctx, w, h, t, [{ xs, ys: hist, color: t.accent, lw: 2.5, label: 'loss (media su un batch)' }], { xlabel: 'passi', ymin: 0, nx: 5 });
    });
    function summary() {
      const ns = toy.norms(), alive = ns.filter(v => v > 0.35).length;
      out.innerHTML = `passi: <b>${stepsDone}</b> · feature rappresentate (lunghezza > 0,35): <b>${alive}</b> su ${NF} in uno spazio a <b>${MF}</b> dimensioni · lunghezze: ${ns.map((v, i) => 'f' + (i + 1) + ' ' + f2(v, 2)).join(' · ')}`;
    }
    function reset() { running = false; stepsDone = 0; hist = []; toy = new SuperToy(NF, MF, sS.get(), sD.get(), 1); summary(); cS.redraw(); cN.redraw(); }
    function frame() {
      if (!running) return;
      let last = 0; for (let k = 0; k < 25; k++) { last = toy.step(256); stepsDone++; if (stepsDone % 25 === 0) hist.push(last); }
      summary(); cS.redraw(); cN.redraw();
      if (stepsDone >= 2000) { running = false; return; }
      requestAnimationFrame(frame);
    }
    const trainBtn = button('Allena (2000 passi)', () => { if (stepsDone >= 2000) reset(); if (!running) { running = true; requestAnimationFrame(frame); } }, 'primary');
    root.append(lab('Laboratorio · cinque feature, due dimensioni (si addestra davvero, con Adam)',
      el('p', { html: 'Il modello deve comprimere 5 feature in 2 numeri (<var>h</var> = <var>W</var><var>x</var>) e poi ricostruirle (<var>x̂</var> = ReLU(<var>W</var><sup>T</sup><var>h</var> + <var>b</var>)). Ogni feature è a zero con probabilità <var>S</var>, altrimenti vale un numero casuale in [0, 1]. La feature 1 è la più importante, la 5 la meno. Cambiando un cursore il modello riparte da capo.' }),
      controls(sS.el, sD.el), controls(trainBtn, button('Azzera', () => reset())), cvS, out, cvN,
      el('p', { class: 'muted', html: 'Con <b>S = 0</b> (feature sempre presenti) il modello tiene solo le due più importanti, ortogonali, e ignora le altre. Alzando <b>S</b> compaiono una terza, una quarta e, con <b>S</b> molto alta, tutte e cinque le feature, disposte come i vertici di un pentagono: non sono più ortogonali, quindi si disturbano a vicenda (la matrice WᵀW ha valori fuori diagonale non nulli), ma tanto raramente sono attive insieme che il guadagno supera il costo. Il bias negativo e la ReLU filtrano parte dell\'interferenza.' })));
    reset();

    /* ---- probes ---- */
    const PROP = { vowel: ['è una vocale', isVowelCh], punct: ['è spazio o punteggiatura', isPunctCh] };
    let propK = 'vowel', result = null, busy = false;
    const psel = select(Object.entries(PROP).map(([k, v]) => [k, 'il carattere ' + v[0]]), propK, v => { propK = v; }, 'proprietà da cercare');
    const cvP = canvasEl(250), outP = el('div', { class: 'readout' });
    const cP = setupCanvas(cvP, (ctx, w, h, t) => {
      if (!result) { text(ctx, 'premi il pulsante: allena il modello del modulo 8 per 800 passi e addestra le sonde', 16, h / 2, { color: t.muted, size: 12 }); return; }
      const left = 44, top = 24, bot = 40, groups = [-1, -2, -3], series = [['majority', 'classe più frequente', t.muted], ['control', 'controllo: etichette mescolate', t.rose], ['untrained', 'rete NON addestrata', t.amber], ['trained', 'rete addestrata', t.accent]];
      const gw = (w - left - 10) / 3, bw = gw / (series.length + 1), Y = v => h - bot - v * (h - bot - top);
      for (let i = 0; i <= 4; i++) { ctx.strokeStyle = rgba(t.line, 0.7); ctx.beginPath(); ctx.moveTo(left, Y(i / 4)); ctx.lineTo(w - 6, Y(i / 4)); ctx.stroke(); text(ctx, f2(i / 4, 2), left - 5, Y(i / 4) + 4, { align: 'right', color: t.muted, size: 10 }); }
      groups.forEach((p, gi) => {
        series.forEach(([k, , col], si) => { const v = result[p][k], x = left + gi * gw + bw * (si + 0.5); ctx.fillStyle = col; ctx.fillRect(x, Y(v), bw * 0.85, h - bot - Y(v)); text(ctx, f2(v, 2), x + bw * 0.42, Y(v) - 4, { align: 'center', color: t.ink, size: 10 }); });
        text(ctx, p === -1 ? 'carattere precedente (posizione −1)' : 'posizione ' + p, left + gi * gw + gw / 2, h - bot + 16, { align: 'center', color: t.ink, size: 11 });
      });
      let lx = left; series.forEach(([, nm, col]) => { ctx.fillStyle = col; ctx.fillRect(lx, 6, 10, 10); text(ctx, nm, lx + 14, 15, { color: t.muted, size: 10 }); lx += 30 + nm.length * 5.6; });
    });
    function runProbe() {
      if (busy) return; busy = true; outP.textContent = 'calcolo in corso (addestramento + sonde)…';
      requestAnimationFrame(() => requestAnimationFrame(() => {
        try {
          const chars = [...new Set(CORPUS)].sort(), data = Int32Array.from([...CORPUS].map(c => chars.indexOf(c))), V = chars.length;
          const trained = trainCharLM(data, V, 800), fresh = new CharLM(V, 3, 8, 64, 7); result = {};
          [-1, -2, -3].forEach(p => { result[p] = probeRun(trained, fresh, data, chars, p, PROP[propK][1]); });
          const r1 = result[-1];
          outP.innerHTML = `proprietà: <b>il carattere ${PROP[propK][0]}</b>. Posizione −1: rete addestrata <b>${f2(r1.trained * 100, 0)}%</b>, non addestrata ${f2(r1.untrained * 100, 0)}%, controllo ${f2(r1.control * 100, 0)}% (classe più frequente: ${f2(r1.majority * 100, 0)}%).`;
        } catch (e) { outP.textContent = 'errore: ' + e.message; }
        busy = false; cP.redraw();
      }));
    }
    root.append(lab('Laboratorio · sonde lineari (linear probes) su un modello vero',
      el('p', { html: 'Prendiamo il language model a caratteri del modulo 8, lo addestriamo per 800 passi, e per migliaia di contesti da 3 caratteri leggiamo le 64 attivazioni dello strato nascosto. Poi addestriamo un classificatore <b>lineare</b> che, da quelle attivazioni, prova a indovinare una proprietà di uno dei caratteri in ingresso. Se ci riesce, l\'informazione è presente in modo linearmente leggibile.' }),
      controls(psel, button('Allena il modello e sonda', runProbe, 'primary')), cvP, outP,
      el('p', { class: 'muted', html: 'Come si legge: la <b>classe più frequente</b> è la soglia da battere; il <b>controllo</b> (stesse sonde su etichette mescolate) dice se la sonda sta solo memorizzando. La rete addestrata batte la non addestrata, ma <i>anche la rete a pesi casuali</i> fa molto meglio del caso: una proiezione casuale conserva già molta informazione sull\'ingresso. Per questo una sonda riuscita non basta a dire che il modello <i>usa</i> quella informazione.' })));
    cP.redraw();

    root.append(el('h2', { text: 'Gli strumenti dell\'interpretabilità' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Probing</b>: sonde lineari sulle attivazioni (sopra). Mostra che un'informazione è <i>presente</i>, non che è <i>usata</i>.</li>
      <li><b>Intervento causale</b> (<i>activation patching</i>, ablation): si sostituisce o si azzera un'attivazione e si guarda come cambia l'uscita. È la prova che una componente conta davvero, e il metodo con cui si è mostrato il circuito delle induction head (modulo 5).</li>
      <li><b>Logit lens</b>: si applica l'unembedding agli stati intermedi del residual stream per vedere quale token "starebbe prevedendo" il modello dopo ogni layer.</li>
      <li><b>Sparse autoencoder (SAE)</b>: si addestra un autoencoder con molte più unità che dimensioni e con penalità di sparsità sulle attivazioni di un layer, per scomporre la superposition in direzioni più interpretabili ("feature"). Il laboratorio qui sopra mostra proprio il fenomeno che gli SAE cercano di invertire.</li>
      <li><b>Circuiti</b>: si cerca il sottoinsieme di componenti (head, MLP) che implementa un comportamento specifico, per esempio copiare un nome visto prima.</li>
      <li><b>Limiti</b>: le feature trovate sono spesso interpretabili solo in parte, i metodi sono costosi su modelli grandi e una spiegazione plausibile non è per forza fedele a ciò che il modello calcola davvero.</li>` }));

    root.append(details('math', 'La matematica: perché la superposition può convenire', 'formule', `
      <p>Con <var>x̂</var> = ReLU(<var>W</var><sup>T</sup><var>W</var> <var>x</var> + <var>b</var>) l'uscita per la feature <var>i</var> è, senza ReLU, <var>x̂</var><sub>i</sub> = ‖<var>w</var><sub>i</sub>‖² <var>x</var><sub>i</sub> + Σ<sub>j≠i</sub> (<var>w</var><sub>i</sub>·<var>w</var><sub>j</sub>) <var>x</var><sub>j</sub> + <var>b</var><sub>i</sub>, dove <var>w</var><sub>i</sub> è la colonna <i>i</i> di <var>W</var>.</p>
      <p>Il primo termine è il segnale (ideale: ‖<var>w</var><sub>i</sub>‖ = 1), il secondo è l'<b>interferenza</b> dalle altre feature: se <var>w</var><sub>i</sub> e <var>w</var><sub>j</sub> non sono ortogonali, quando <var>x</var><sub>j</sub> è attiva "sporca" l'uscita <i>i</i>.</p>
      <p>Se ogni feature è attiva con probabilità 1 − <var>S</var>, la probabilità che due feature siano attive insieme è (1 − <var>S</var>)². Con <var>S</var> alta le collisioni sono rare e l'errore atteso dovuto all'interferenza cresce solo con questo fattore piccolo, mentre il guadagno di rappresentare una feature in più è lineare. Il confronto tra i due termini decide quante feature entrano.</p>
      <p>Con <var>m</var> = 2 e <var>n</var> = 5 simmetriche, la configurazione che minimizza l'interferenza massima è il pentagono regolare (vettori a 72°), esattamente la figura che emerge. In dimensioni alte il numero di direzioni quasi ortogonali cresce esponenzialmente con <var>m</var> (fenomeno di concentrazione della misura, imparentato col lemma di Johnson-Lindenstrauss), il che rende plausibile che un modello reale conservi molte più feature che dimensioni.</p>
      <p><b>Sonda lineare.</b> Con attivazioni <var>a</var> ∈ ℝ<sup>64</sup> e proprietà binaria <var>y</var> si addestra una regressione logistica <var>p</var>(<var>y</var>=1 | <var>a</var>) = σ(<var>v</var>·<var>a</var> + <var>c</var>) minimizzando la cross-entropy; l'accuratezza su dati mai visti misura la leggibilità <i>lineare</i> di <var>y</var>.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Che cosa significa che un modello usa "superposition"?', opts: ['Che ogni neurone rappresenta un solo concetto', 'Che rappresenta più feature che dimensioni, usando direzioni non ortogonali', 'Che ha più layer del necessario'], ans: 1, why: 'Le feature sono direzioni; se sono sparse, molte possono condividere uno spazio più piccolo, a costo di una certa interferenza.' },
      { q: 'Una sonda lineare legge una proprietà con alta accuratezza. Cosa si può concludere?', opts: ['Che il modello usa quella proprietà per decidere', 'Che l\'informazione è presente nelle attivazioni in forma linearmente leggibile', 'Che il modello capisce quel concetto'], ans: 1, why: 'Il probing dice che l\'informazione c\'è. Per sapere se conta serve un intervento causale, e un controllo con una rete a pesi casuali.' },
      { q: 'Perché con sparsità bassa il modello tiene solo due feature ortogonali?', opts: ['Perché le altre sono inutili', 'Perché se le feature sono quasi sempre attive l\'interferenza costa troppo, conviene rappresentare bene le più importanti', 'Perché il bias lo impone'], ans: 1, why: 'Con S bassa le collisioni sono frequenti e l\'interferenza pesa, quindi la soluzione ortogonale vince.' },
    ]));
  },
});
