'use strict';
/* Module 8 - training: cross-entropy, backpropagation, optimizers on a loss landscape, a real tiny LM */
function softmaxCE(z, y) { const p = softmax(z); return { p, loss: -Math.log(p[y] + 1e-300), grad: p.map((v, i) => v - (i === y ? 1 : 0)) }; }
function backprop(P) {
  const u = P.w1 * P.x + P.b1, h = Math.max(0, u), y = P.w2 * h + P.b2, L = (y - P.t) ** 2;
  const dy = 2 * (y - P.t), dw2 = dy * h, db2 = dy, dh = dy * P.w2, du = u > 0 ? dh : 0, dw1 = du * P.x, db1 = du, dx = du * P.w1;
  return { u, h, y, L, dy, dh, du, g: { w1: dw1, b1: db1, w2: dw2, b2: db2, x: dx } };
}
const SURF = {
  valle: { name: 'Valle stretta (Rosenbrock)', a: [-1.6, 1.6], b: [-0.6, 2.6], start: [-1.3, 2.1], lr: [0.01, 0.005, 0.05], mins: [[1, 1]], f: (a, b) => (1 - a) ** 2 + 8 * (b - a * a) ** 2, g: (a, b) => [-2 * (1 - a) - 32 * a * (b - a * a), 16 * (b - a * a)] },
  conca: { name: 'Conca molto allungata', a: [-3, 3], b: [-1, 1], start: [-2.6, 0.8], lr: [0.004, 0.002, 0.05], mins: [[0, 0]], f: (a, b) => 0.5 * (a * a + 400 * b * b), g: (a, b) => [a, 400 * b] },
  himmel: { name: 'Quattro minimi (Himmelblau)', a: [-5, 5], b: [-5, 5], start: [0, 0.2], lr: [0.05, 0.02, 0.1], mins: [[3, 2], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]], f: (a, b) => ((a * a + b - 11) ** 2 + (a + b * b - 7) ** 2) / 30, g: (a, b) => { const u = a * a + b - 11, v = a + b * b - 7; return [(4 * a * u + 2 * v) / 30, (2 * u + 4 * b * v) / 30]; } },
};
class Opt {
  constructor(kind) { this.kind = kind; }
  reset(s) { this.p = s.slice(); this.v = [0, 0]; this.m = [0, 0]; this.s = [0, 0]; this.t = 0; this.path = [s.slice()]; this.div = false; }
  step(gfn, lr) {
    if (this.div) return; const g = gfn(this.p[0], this.p[1]); this.t++;
    for (let i = 0; i < 2; i++) {
      if (this.kind === 'sgd') this.p[i] -= lr * g[i];
      else if (this.kind === 'momentum') { this.v[i] = 0.9 * this.v[i] + g[i]; this.p[i] -= lr * this.v[i]; }
      else { this.m[i] = 0.9 * this.m[i] + 0.1 * g[i]; this.s[i] = 0.999 * this.s[i] + 0.001 * g[i] * g[i]; this.p[i] -= lr * (this.m[i] / (1 - 0.9 ** this.t)) / (Math.sqrt(this.s[i] / (1 - 0.999 ** this.t)) + 1e-8); }
    }
    if (!Number.isFinite(this.p[0] + this.p[1]) || Math.abs(this.p[0]) > 50 || Math.abs(this.p[1]) > 50) { this.div = true; return; }
    this.path.push(this.p.slice()); if (this.path.length > 3000) this.path.shift();
  }
}
Object.assign(LAB.tests, { softmaxCE, backprop, SURF, Opt });

registerModule({
  id: 'm8', n: 8, title: 'Training', eyebrow: 'come si trovano i parametri', heading: 'Addestrare: scendere lungo la loss',
  lede: 'Tutto ciò che hai visto finora ha parametri casuali. L\'addestramento li aggiusta, miliardi di volte, per rendere più probabile il token che nel testo vero viene davvero dopo.',
  build(root) {
    root.append(el('div', { html: `
      <p>L'obiettivo è sorprendentemente semplice: per ogni posizione di ogni testo, il modello produce una distribuzione sul token successivo e si misura <b>quanta probabilità ha assegnato al token vero</b>. La <b>loss</b> è il logaritmo negativo di quella probabilità, mediato su tutte le posizioni (<i>cross-entropy</i>).
      Poi si calcola, per ogni parametro, <b>in che direzione</b> farlo muovere per ridurre la loss (<i>gradiente</i>, ottenuto con la <i>backpropagation</i>) e si fa un piccolo passo (<i>ottimizzatore</i>). Si ripete su miliardi di esempi.</p>
      <p>Un trucco rende tutto efficiente: grazie alla maschera causale, una singola passata su una sequenza di <var>T</var> token fornisce <var>T</var> previsioni (una per posizione) e quindi <var>T</var> termini di loss, tutti calcolati in parallelo. Si chiama <i>teacher forcing</i>.</p>` }));

    /* ---------- 8a cross-entropy ---------- */
    root.append(el('h2', { text: '1. La loss e il suo gradiente' }));
    root.append(el('p', { html: 'Se il modello assegna probabilità <var>p</var> al token giusto, la loss è −ln <var>p</var>: 0 se <var>p</var> = 1, ≈ 0,69 se <var>p</var> = ½, cresce senza limite se <var>p</var> → 0. Il gradiente rispetto ai logit ha una forma elegantissima: <b>(probabilità − one-hot del token giusto)</b>. Qui sotto i logit sono parametri liberi, così puoi vedere un vero passo di discesa.' }));
    const names = ['Roma', 'Milano', 'la', 'una', 'Torino', 'Napoli'], z0 = [1.0, 3.0, 2.0, 0.5, 2.5, 0.0];
    let z = z0.slice(), yT = 0, hist = [], ce = softmaxCE(z, yT);
    const cvCE = canvasEl(260), cvLoss = canvasEl(150), ceOut = el('div', { class: 'readout' });
    const sEta = slider({ label: 'learning rate η', min: 0.1, max: 2, step: 0.1, value: 0.5, fmt: v => v.toFixed(1) });
    const bars = setupCanvas(cvCE, (ctx, w, h, t) => {
      const n = z.length, left = 40, bw = (w - left - 10) / n, y1 = h * 0.5 - 14, yb = h * 0.76, hb = h * 0.14;
      text(ctx, 'probabilità p', 6, 14, { color: t.muted, size: 11 }); text(ctx, 'gradiente ∂L/∂z = p − y', 6, h * 0.5 + 12, { color: t.muted, size: 11 });
      ctx.strokeStyle = t.line; ctx.beginPath(); ctx.moveTo(left, y1); ctx.lineTo(w - 6, y1); ctx.moveTo(left, yb); ctx.lineTo(w - 6, yb); ctx.stroke();
      names.forEach((nm, i) => {
        const x = left + i * bw + bw * 0.16, ww = bw * 0.68, ph = ce.p[i] * (y1 - 24), g = ce.grad[i];
        ctx.fillStyle = i === yT ? t.amber : t.accent; ctx.fillRect(x, y1 - ph, ww, ph);
        text(ctx, f2(ce.p[i], 2), x + ww / 2, y1 - ph - 4, { align: 'center', color: t.ink, size: 11 });
        ctx.fillStyle = g > 0 ? t.rose : t.teal; ctx.fillRect(x, yb, ww, -g * hb);
        text(ctx, (g > 0 ? '+' : '') + f2(g, 2), x + ww / 2, g > 0 ? yb - g * hb - 4 : yb - g * hb + 12, { align: 'center', color: g > 0 ? t.rose : t.teal, size: 11 });
        text(ctx, nm + (i === yT ? ' ✓' : ''), x + ww / 2, h - 8, { align: 'center', color: i === yT ? t.amber : t.muted, size: 12, sans: true, weight: i === yT ? 800 : 500 });
      });
    });
    const lossC = setupCanvas(cvLoss, (ctx, w, h, t) => { chart(ctx, w, h, t, [{ xs: hist.map((_, i) => i), ys: hist, color: t.accent, lw: 2.5, dots: hist.length < 30 }], { xlabel: 'passi', ylabel: 'loss', ymin: 0, ny: 3, xmin: 0, xmax: Math.max(hist.length - 1, 5), nx: 5 }); });
    function ceUpd() {
      ce = softmaxCE(z, yT); if (!hist.length) hist.push(ce.loss);
      ceOut.innerHTML = `token giusto: <b>${names[yT]}</b> · p = <b>${f2(ce.p[yT], 3)}</b> · loss = −ln p = <b>${f2(ce.loss, 3)}</b> · perplessità e<sup>loss</sup> = ${f2(Math.exp(ce.loss), 2)}<br>` +
        `<span class="muted">rosa = "abbassa questo logit", verde = "alzalo". Il token giusto ha sempre gradiente negativo (−(1−p)); gli altri sono spinti giù in proporzione alla loro probabilità.</span>`;
      bars.redraw(); lossC.redraw();
    }
    const ceStep = n => { for (let k = 0; k < n; k++) { z = z.map((v, i) => v - sEta.get() * ce.grad[i]); ce = softmaxCE(z, yT); hist.push(ce.loss); } ceUpd(); };
    cvCE.addEventListener('click', ev => { const rc = cvCE.getBoundingClientRect(), i = Math.floor((ev.clientX - rc.left - 40) / ((rc.width - 50) / names.length)); if (i >= 0 && i < names.length) { yT = i; hist = []; ceUpd(); } });
    root.append(lab('Laboratorio · softmax + cross-entropy (clicca una barra per cambiare il token giusto)', controls(sEta.el, button('1 passo', () => ceStep(1), 'primary'), button('10 passi', () => ceStep(10)), button('Reset', () => { z = z0.slice(); hist = []; ceUpd(); })), cvCE, ceOut, cvLoss)); ceUpd();

    /* ---------- 8b backprop ---------- */
    root.append(el('h2', { text: '2. La backpropagation, a mano' }));
    root.append(el('p', { html: 'Una rete è una catena di operazioni semplici. La <b>regola della catena</b> dice che per sapere come la loss cambia rispetto a un nodo basta moltiplicare, andando all\'indietro, le derivate locali. Qui una "rete" con un solo neurone: <var>u</var> = <var>w</var><sub>1</sub><var>x</var> + <var>b</var><sub>1</sub>, <var>h</var> = ReLU(<var>u</var>), <var>y</var> = <var>w</var><sub>2</sub><var>h</var> + <var>b</var><sub>2</sub>, <var>L</var> = (<var>y</var> − <var>t</var>)². Ogni scheda mostra il valore in avanti (blu) e il gradiente ∂<var>L</var>/∂nodo all\'indietro (rosa).' }));
    const BP = { x: 1.5, t: 1.0, w1: 0.8, b1: -0.3, w2: 1.2, b2: 0.1 };
    const bpSl = Object.fromEntries(['x', 't', 'w1', 'b1', 'w2', 'b2'].map(k => [k, slider({ label: k === 'w1' ? 'w₁' : k === 'w2' ? 'w₂' : k === 'b1' ? 'b₁' : k === 'b2' ? 'b₂' : k, min: -2, max: 2, step: 0.05, value: BP[k], fmt: v => v.toFixed(2), onInput: v => { BP[k] = v; bpUpd(); } })]));
    const gBox = el('div', { class: 'graph' }), bpOut = el('div', { class: 'readout' }), sEta2 = slider({ label: 'η', min: 0.01, max: 0.3, step: 0.01, value: 0.1, fmt: v => v.toFixed(2) });
    let bpHist = [];
    function bpUpd(rec) {
      const r = backprop(BP); if (rec || !bpHist.length) bpHist.push(r.L);
      const eps = 1e-6; let err = 0;
      ['w1', 'b1', 'w2', 'b2'].forEach(k => { const a = { ...BP, [k]: BP[k] + eps }, b = { ...BP, [k]: BP[k] - eps }; err = Math.max(err, Math.abs((backprop(a).L - backprop(b).L) / (2 * eps) - r.g[k])); });
      const card = (name, f, g) => el('div', { class: 'gn' }, el('b', { text: name }), el('div', { class: 'f', text: 'valore ' + f2(f, 3) }), g == null ? null : el('div', { class: 'g', text: '∂L/∂ = ' + f2(g, 3) }));
      gBox.replaceChildren(card('x', BP.x, r.g.x), card('u = w₁x + b₁', r.u, r.du), card('h = ReLU(u)', r.h, r.dh), card('y = w₂h + b₂', r.y, r.dy), card('L = (y − t)²', r.L, 1));
      bpOut.innerHTML = `gradienti sui parametri: ∂L/∂w₂ = <b>${f2(r.g.w2, 3)}</b> · ∂L/∂b₂ = <b>${f2(r.g.b2, 3)}</b> · ∂L/∂w₁ = <b>${f2(r.g.w1, 3)}</b> · ∂L/∂b₁ = <b>${f2(r.g.b1, 3)}</b><br>` +
        `verifica numerica (differenze finite): errore massimo <b>${err.toExponential(1)}</b>` + (r.u <= 0 ? '<br><span style="color:var(--rose)">u ≤ 0: il neurone ReLU è spento, il gradiente verso w₁ e b₁ è esattamente 0 (neurone "morto" per questo input).</span>' : '') +
        `<br>loss nei passi recenti: ${bpHist.slice(-8).map(v => f2(v, 3)).join(' → ')}`;
    }
    const bpStep = () => { const r = backprop(BP); ['w1', 'b1', 'w2', 'b2'].forEach(k => { BP[k] -= sEta2.get() * r.g[k]; bpSl[k].set(BP[k]); }); bpUpd(true); };
    root.append(lab('Laboratorio · il grafo di calcolo', controls(...['x', 't'].map(k => bpSl[k].el)), controls(...['w1', 'b1', 'w2', 'b2'].map(k => bpSl[k].el)), gBox, bpOut, controls(sEta2.el, button('Passo di discesa', bpStep, 'primary'), button('10 passi', () => { for (let i = 0; i < 10; i++) bpStep(); }), button('Azzera storia', () => { bpHist = []; bpUpd(); }))));
    bpUpd();
    root.append(el('p', { html: 'In un LLM succede la stessa cosa con miliardi di nodi: si fa una passata <i>in avanti</i> salvando i valori intermedi, poi una passata <i>all\'indietro</i> che riutilizza la regola della catena. Il costo è circa il doppio della passata in avanti, ed è il motivo per cui il training costa ~6<var>N</var> FLOP per token contro ~2<var>N</var> dell\'inferenza.' }));

    /* ---------- 8c landscape ---------- */
    root.append(el('h2', { text: '3. Il paesaggio della loss e gli ottimizzatori' }));
    root.append(el('p', { html: 'Con due soli parametri (<var>a</var>, <var>b</var>) la loss è una superficie che possiamo vedere. Un LLM ha miliardi di parametri: stessa idea, ma in miliardi di dimensioni. Tre ottimizzatori partono dallo stesso punto: <b style="color:var(--amber)">SGD</b> segue il gradiente, <b style="color:var(--rose)">Momentum</b> accumula velocità, <b style="color:var(--teal)">Adam</b> riscala il passo di ogni parametro in base alla storia dei suoi gradienti. Trascina per ruotare, clicca sulla mappa piatta per scegliere un nuovo punto di partenza.' }));
    let surfKey = 'valle', S = SURF[surfKey], gmax = 1;
    const opts = [new Opt('sgd'), new Opt('momentum'), new Opt('adam')], ocol = ['amber', 'rose', 'teal'], onames = ['SGD', 'Momentum', 'Adam'];
    const view = f => Math.log1p(f) / Math.log1p(gmax);
    const setSurf = k => { surfKey = k; S = SURF[k]; if (typeof lrS !== 'undefined') lrS.forEach((sl, i) => sl.set(S.lr[i])); gmax = 0; for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) gmax = Math.max(gmax, S.f(S.a[0] + (S.a[1] - S.a[0]) * i / 40, S.b[0] + (S.b[1] - S.b[0]) * j / 40)); resetOpt(S.start); };
    const resetOpt = st => { opts.forEach(o => o.reset(st)); land3.redraw && refreshLand(); };
    const cv3 = canvasEl(380), cvMap = canvasEl(300), legend3 = el('div', { class: 'readout' });
    const cam = new Cam3D(cv3, () => land3.redraw(), { yaw: 0.9, pitch: 0.75 });
    const toXZ = (a, b) => [(a - S.a[0]) / (S.a[1] - S.a[0]) * 2 - 1, (b - S.b[0]) / (S.b[1] - S.b[0]) * 2 - 1];
    const land3 = setupCanvas(cv3, (ctx, w, h, t) => {
      if (!opts[0].path) return;
      const N = 30, sc = Math.min(w, h) * 0.4, quads = [], Y = (a, b) => (view(S.f(a, b)) - 0.5) * 1.1;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const pts = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([p, q]) => { const a = S.a[0] + (S.a[1] - S.a[0]) * p / N, b = S.b[0] + (S.b[1] - S.b[0]) * q / N, [X, Z] = toXZ(a, b); return cam.project(X, Y(a, b), Z, w, h, sc); });
        const a = S.a[0] + (S.a[1] - S.a[0]) * (i + 0.5) / N, b = S.b[0] + (S.b[1] - S.b[0]) * (j + 0.5) / N;
        quads.push({ pts, d: (pts[0][2] + pts[2][2]) / 2, v: view(S.f(a, b)) });
      }
      quads.sort((p, q) => p.d - q.d);
      quads.forEach(({ pts, v }) => { ctx.fillStyle = rgba(t.accent, 0.06 + 0.5 * v); ctx.strokeStyle = rgba(t.ink, 0.14); ctx.lineWidth = 0.6; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let k = 1; k < 4; k++) ctx.lineTo(pts[k][0], pts[k][1]); ctx.closePath(); ctx.fill(); ctx.stroke(); });
      const P3 = ([a, b]) => { const [X, Z] = toXZ(a, b); return cam.project(X, Y(a, b) + 0.02, Z, w, h, sc); };
      S.mins.forEach(m => { const q = P3(m); ctx.strokeStyle = t.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(q[0] - 6, q[1]); ctx.lineTo(q[0] + 6, q[1]); ctx.moveTo(q[0], q[1] - 6); ctx.lineTo(q[0], q[1] + 6); ctx.stroke(); });
      opts.forEach((o, k) => {
        const c = t[ocol[k]]; ctx.strokeStyle = c; ctx.lineWidth = 2.5; ctx.beginPath();
        o.path.forEach((p, i) => { const q = P3(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); ctx.stroke();
        const q = P3(o.p); ctx.fillStyle = c; ctx.beginPath(); ctx.arc(q[0], q[1], 5.5, 0, 7); ctx.fill(); ctx.strokeStyle = t.bg; ctx.lineWidth = 1.5; ctx.stroke();
      });
      text(ctx, 'altezza = ln(1 + loss) · croce = minimo · trascina per ruotare', 8, h - 8, { color: t.muted, size: 10 });
    });
    const mapC = setupCanvas(cvMap, (ctx, w, h, t) => {
      if (!opts[0].path) return;
      const N = 56, cw = w / N, ch = h / N;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const a = S.a[0] + (S.a[1] - S.a[0]) * (i + 0.5) / N, b = S.b[1] - (S.b[1] - S.b[0]) * (j + 0.5) / N, v = view(S.f(a, b)), band = Math.floor(v * 14);
        ctx.fillStyle = rgba(t.accent, 0.05 + 0.45 * v + (band % 2 ? 0.07 : 0)); ctx.fillRect(i * cw, j * ch, cw + 0.5, ch + 0.5);
      }
      const P = ([a, b]) => [(a - S.a[0]) / (S.a[1] - S.a[0]) * w, h - (b - S.b[0]) / (S.b[1] - S.b[0]) * h];
      S.mins.forEach(m => { const q = P(m); ctx.strokeStyle = t.ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(q[0] - 6, q[1]); ctx.lineTo(q[0] + 6, q[1]); ctx.moveTo(q[0], q[1] - 6); ctx.lineTo(q[0], q[1] + 6); ctx.stroke(); });
      opts.forEach((o, k) => { const c = t[ocol[k]]; ctx.strokeStyle = c; ctx.lineWidth = 2.5; ctx.beginPath(); o.path.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }); ctx.stroke(); const q = P(o.p); ctx.fillStyle = c; ctx.beginPath(); ctx.arc(q[0], q[1], 5, 0, 7); ctx.fill(); });
      const s0 = P(S.start); ctx.strokeStyle = t.ink; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(s0[0], s0[1], 8, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      text(ctx, 'mappa dall\'alto · clicca per scegliere la partenza', 8, h - 8, { color: t.ink, size: 10 });
    });
    cvMap.addEventListener('click', ev => {
      const rc = cvMap.getBoundingClientRect(), a = S.a[0] + (S.a[1] - S.a[0]) * (ev.clientX - rc.left) / rc.width, b = S.b[0] + (S.b[1] - S.b[0]) * (1 - (ev.clientY - rc.top) / rc.height);
      S.start = [a, b]; resetOpt(S.start);
    });
    const lrS = [
      slider({ label: 'lr SGD', min: 0.001, max: 0.06, step: 0.001, value: 0.01, fmt: v => v.toFixed(3) }),
      slider({ label: 'lr Momentum', min: 0.001, max: 0.03, step: 0.001, value: 0.005, fmt: v => v.toFixed(3) }),
      slider({ label: 'lr Adam', min: 0.005, max: 0.2, step: 0.005, value: 0.05, fmt: v => v.toFixed(3) })];
    const sSp3 = slider({ label: 'velocità', min: 1, max: 12, value: 3 });
    const stepAll = () => opts.forEach((o, k) => o.step(S.g, lrS[k].get()));
    function refreshLand() {
      legend3.innerHTML = opts.map((o, k) => { const f = S.f(o.p[0], o.p[1]); return `<span style="color:var(--${ocol[k]})"><b>${onames[k]}</b></span> passi ${o.t} · loss <b>${o.div ? '—' : f2(f, 4)}</b>${o.div ? ' <span style="color:var(--rose)">(divergente!)</span>' : f < 1e-3 ? ' ✓ al minimo' : ''}`; }).join('<br>');
      land3.redraw(); mapC.redraw();
    }
    const lp3 = loop(cv3, () => { for (let k = 0; k < sSp3.get(); k++) stepAll(); refreshLand(); });
    const play3 = button('Avvia', () => { if (lp3.running) { lp3.stop(); play3.textContent = 'Avvia'; } else { lp3.start(); play3.textContent = 'Pausa'; } }, 'primary');
    root.append(lab('Laboratorio · tre ottimizzatori sulla stessa superficie',
      controls(select(Object.keys(SURF).map(k => [k, SURF[k].name]), surfKey, v => { setSurf(v); }, 'superficie'), sSp3.el),
      controls(...lrS.map(x => x.el)),
      controls(play3, button('1 passo', () => { stepAll(); refreshLand(); }), button('Ricomincia', () => resetOpt(S.start))),
      el('div', { class: 'grid2' }, cv3, cvMap), legend3,
      el('p', { html: '<b>Da provare.</b> Ogni ottimizzatore ha il proprio learning rate (i valori iniziali sono scelti per ciascuno: non sono confrontabili tra loro, perché in SGD il passo dipende dalla scala del gradiente, in Adam è circa il learning rate stesso). Sulla <i>valle stretta</i> SGD avanza lentamente lungo il fondo, Momentum accumula velocità e lo supera, Adam arriva per primo. Sulla <i>conca molto allungata</i> (curvatura 1 in una direzione, 400 nell\'altra) SGD è limitato dalla direzione più ripida: se porti il suo learning rate sopra 2/400 = 0,005 <b>diverge</b>; Adam normalizza ogni coordinata e procede veloce in entrambe. Su <i>Himmelblau</i> ci sono quattro minimi: cambiando il punto di partenza (clic sulla mappa) cambia la valle in cui si finisce. Su problemi così piccoli tutti arrivano comunque: la differenza si vede su problemi mal condizionati e rumorosi come quelli reali.' })));
    setSurf('valle');

    /* ---------- 8d: real LM ---------- */
    root.append(el('h2', { text: '4. Un vero language model, addestrato qui' }));
    root.append(el('p', { html: 'Mettiamo insieme i pezzi: dati, previsione del token successivo, cross-entropy, backpropagation, Adam. Non è un transformer (nel prossimo aggiornamento del laboratorio lo sarà), ma il ciclo di training è identico a quello degli LLM.' }));
    buildLMLab(root);

    /* ---------- 8e: schedule ---------- */
    root.append(el('h2', { text: '5. Il training su larga scala' }));
    root.append(el('p', { html: 'Con miliardi di parametri e migliaia di miliardi di token, il ciclo è lo stesso ma attorniato da molti accorgimenti. Uno dei più importanti riguarda il <b>learning rate</b>: si parte da zero (<i>warmup</i>, per non destabilizzare un modello ancora casuale), si sale al massimo, poi si scende lentamente, spesso con un coseno, fino a una frazione del picco.' }));
    const sW = slider({ label: 'warmup (passi)', min: 0, max: 3000, step: 100, value: 500, onInput: () => sch.redraw() }), sTot = slider({ label: 'passi totali', min: 2000, max: 50000, step: 1000, value: 20000, fmt: human, onInput: () => sch.redraw() }), sPk = slider({ label: 'picco', min: 1, max: 10, step: 0.5, value: 3, fmt: v => v.toFixed(1) + 'e-4', onInput: () => sch.redraw() });
    const schCv = canvasEl(210);
    const sch = setupCanvas(schCv, (ctx, w, h, t) => {
      const W = sW.get(), T = sTot.get(), pk = sPk.get(), xs = range(201).map(i => i * T / 200), lrf = s => s < W ? pk * s / Math.max(W, 1) : pk * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * (s - W) / Math.max(T - W, 1))));
      chart(ctx, w, h, t, [{ xs, ys: xs.map(lrf), color: t.accent, lw: 3, label: 'learning rate (×10⁻⁴)' }], { xlabel: 'passo di training', ymin: 0, ymax: 10.5, xmin: 0, xmax: T, nx: 5 });
    });
    root.append(lab('Laboratorio · schedule del learning rate (warmup + decadimento a coseno, minimo al 10% del picco)', controls(sW.el, sTot.el, sPk.el), schCv));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Batch enormi.</b> Ogni passo media il gradiente su centinaia di migliaia o milioni di token, distribuiti su migliaia di GPU (parallelismo sui dati, sui tensori, sulle pipeline).</li>
      <li><b>AdamW.</b> Adam con <i>weight decay</i> separato dalla stima del gradiente; <b>gradient clipping</b> sulla norma globale (tipicamente 1,0) per evitare picchi.</li>
      <li><b>Precisione mista.</b> Calcoli in bf16/fp8 con copie master dei pesi in fp32.</li>
      <li><b>Scaling laws.</b> La loss scende come una legge di potenza in parametri, dati e calcolo. Una regola emersa dagli studi "Chinchilla" suggerisce circa 20 token di training per parametro per usare bene un budget di calcolo; i modelli recenti si addestrano spesso su molti più token per renderli più economici da usare.</li>
      <li><b>Dati.</b> Il testo viene filtrato, deduplicato e mescolato. La qualità dei dati conta quanto l'architettura.</li>` }));
    root.append(el('div', { class: 'note', html: '<b>E dopo?</b>Il modello appena pre-addestrato sa continuare testi, non sa ancora fare l\'assistente. Seguono tipicamente <i>supervised fine-tuning</i> su conversazioni di esempio e una fase di ottimizzazione sulle preferenze (RLHF, DPO) o su ricompense verificabili (per i modelli di ragionamento). Sono i prossimi moduli del laboratorio.' }));

    root.append(details('math', 'La matematica: cross-entropy, gradiente, backprop, Adam', 'formule', `
      <p><b>Loss.</b> Per una sequenza <var>x</var><sub>1..T</sub> e parametri <var>θ</var>:</p>
      <span class="eq"><var>L</var>(<var>θ</var>) = −${frac('1', '<var>T</var>')} Σ<sub><var>t</var>=1</sub><sup><var>T</var></sup> ln <var>p</var><sub><var>θ</var></sub>(<var>x</var><sub><var>t</var></sub> | <var>x</var><sub>&lt;<var>t</var></sub>)</span>
      <p>Minimizzarla equivale a massimizzare la verosimiglianza dei dati. Il suo esponenziale è la <b>perplessità</b>: il numero medio di scelte equiprobabili fra cui il modello è "indeciso".</p>
      <p><b>Gradiente sui logit.</b> Con <var>p</var> = softmax(<var>z</var>) e target <var>y</var>: <span class="m"><var>L</var> = −<var>z</var><sub><var>y</var></sub> + ln Σ<sub><var>j</var></sub> e<sup><var>z</var><sub><var>j</var></sub></sup></span>, quindi</p>
      <span class="eq">${frac('∂<var>L</var>', '∂<var>z</var><sub><var>k</var></sub>')} = <var>p</var><sub><var>k</var></sub> − [<var>k</var> = <var>y</var>]</span>
      <p><b>Backpropagation.</b> Per uno strato <var>a</var><sub>l</sub> = <var>σ</var>(<var>z</var><sub>l</sub>), <var>z</var><sub>l</sub> = <var>W</var><sub>l</sub><var>a</var><sub>l−1</sub> + <var>b</var><sub>l</sub>, definendo <var>δ</var><sub>l</sub> = ∂<var>L</var>/∂<var>z</var><sub>l</sub>:</p>
      <span class="eq"><var>δ</var><sub>l</sub> = (<var>W</var><sub>l+1</sub><sup>T</sup> <var>δ</var><sub>l+1</sub>) ⊙ <var>σ</var>′(<var>z</var><sub>l</sub>), &nbsp;&nbsp; ∂<var>L</var>/∂<var>W</var><sub>l</sub> = <var>δ</var><sub>l</sub> <var>a</var><sub>l−1</sub><sup>T</sup>, &nbsp;&nbsp; ∂<var>L</var>/∂<var>b</var><sub>l</sub> = <var>δ</var><sub>l</sub></span>
      <p>È esattamente quanto fa il codice della sezione 4. Nel grafo scalare della sezione 2: ∂<var>L</var>/∂<var>y</var> = 2(<var>y</var>−<var>t</var>), ∂<var>L</var>/∂<var>w</var><sub>2</sub> = ∂<var>L</var>/∂<var>y</var> · <var>h</var>, ∂<var>L</var>/∂<var>u</var> = ∂<var>L</var>/∂<var>h</var> · [<var>u</var> &gt; 0], ∂<var>L</var>/∂<var>w</var><sub>1</sub> = ∂<var>L</var>/∂<var>u</var> · <var>x</var>.</p>
      <p><b>Discesa del gradiente.</b> <span class="m"><var>θ</var> ← <var>θ</var> − <var>η</var>∇<var>L</var></span>. Su una quadratica con curvatura massima <var>λ</var> converge solo se <var>η</var> &lt; 2/<var>λ</var>: oltre, diverge (provalo sulla conca allungata).</p>
      <p><b>Momentum.</b> <span class="m"><var>v</var> ← β<var>v</var> + <var>g</var>, &nbsp; <var>θ</var> ← <var>θ</var> − <var>ηv</var></span>, con β ≈ 0,9. <b>Adam</b>:</p>
      <span class="eq"><var>m</var> ← β<sub>1</sub><var>m</var> + (1−β<sub>1</sub>)<var>g</var>, &nbsp; <var>s</var> ← β<sub>2</sub><var>s</var> + (1−β<sub>2</sub>)<var>g</var>², &nbsp; <var>θ</var> ← <var>θ</var> − <var>η</var> ${frac('<var>m</var> / (1−β<sub>1</sub><sup><var>t</var></sup>)', '√(<var>s</var> / (1−β<sub>2</sub><sup><var>t</var></sup>)) + ε')}</span>
      <p>Dividere per √<var>s</var> dà a ogni parametro un passo di taglia circa <var>η</var> indipendentemente dalla scala del suo gradiente: per questo Adam è robusto in problemi dove parametri diversi hanno curvature molto diverse.</p>`));
    root.append(details('refresh', 'Ripasso: derivata, gradiente, regola della catena', 'analisi', `
      <p>La derivata di <var>f</var>(<var>x</var>) dice di quanto cambia <var>f</var> per un piccolo spostamento di <var>x</var>. Il <b>gradiente</b> ∇<var>f</var> di una funzione di molte variabili raccoglie tutte le derivate parziali: punta nella direzione di massima crescita, quindi −∇<var>f</var> è la direzione di massima discesa.</p>
      <p><b>Regola della catena.</b> Se <var>y</var> = <var>g</var>(<var>x</var>) e <var>L</var> = <var>h</var>(<var>y</var>), allora d<var>L</var>/d<var>x</var> = d<var>L</var>/d<var>y</var> · d<var>y</var>/d<var>x</var>. Esempio: <var>L</var> = (2<var>x</var> − 1)²: <var>y</var> = 2<var>x</var> − 1, d<var>L</var>/d<var>y</var> = 2<var>y</var>, d<var>y</var>/d<var>x</var> = 2, quindi d<var>L</var>/d<var>x</var> = 4(2<var>x</var> − 1). La backpropagation applica questa regola nodo per nodo, partendo dalla loss.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Se il modello assegna probabilità 0,5 al token giusto, la loss (in nat) vale…', opts: ['0,5', '≈ 0,69', '≈ 2,3'], ans: 1, why: '−ln 0,5 = ln 2 ≈ 0,693.' },
      { q: 'Il gradiente della cross-entropy rispetto al logit di un token sbagliato è…', opts: ['sempre 0', 'uguale alla probabilità che il modello gli assegna (positivo: spinge il logit verso il basso)', 'uguale a 1 − p'], ans: 1, why: '∂L/∂z_k = p_k per i token sbagliati e p_y − 1 per quello giusto.' },
      { q: 'Perché Adam può essere più robusto di SGD semplice?', opts: ['Usa meno memoria', 'Normalizza il passo di ogni parametro con una stima della scala dei suoi gradienti', 'Non usa il gradiente'], ans: 1, why: 'La divisione per √s rende il passo ≈ η per ogni parametro, qualunque sia la scala del suo gradiente.' },
      { q: 'Nel language model della sezione 4, la loss di training continua a scendere ma quella di validazione risale. Come si chiama il fenomeno?', opts: ['Underfitting', 'Overfitting (memorizzazione)', 'Vanishing gradient'], ans: 1, why: 'Il modello ha pochi dati: impara a memoria il training set e generalizza peggio.' },
      { q: 'Perché si fa warmup del learning rate?', opts: ['Per far partire più in fretta il training', 'Per evitare passi troppo grandi su un modello appena inizializzato e ancora instabile', 'Per ridurre la memoria'], ans: 1, why: 'All\'inizio i gradienti sono molto rumorosi e le statistiche di Adam non sono ancora affidabili: partire piano evita divergenze.' },
    ]));
  },
});
