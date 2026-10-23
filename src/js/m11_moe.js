'use strict';
/* Module 11 - Mixture of Experts: top-k router, load balancing, capacity, active vs total parameters */

/* toy token set: 4 clusters of unequal size in a 2D "embedding" plane; cluster c has a preferred expert (c mod E) */
const MOE_CL = [[-1.5, 1.1, 108], [1.5, 1.3, 72], [-1.1, -1.5, 36], [1.7, -1.1, 24]];
function moeData(seed = 3) {
  const r = rng(seed), X = [], lab = [];
  MOE_CL.forEach(([cx, cy, n], c) => { for (let i = 0; i < n; i++) { X.push([cx + 0.5 * r.gauss(), cy + 0.5 * r.gauss()]); lab.push(c); } });
  return { X, lab };
}
const moeInit = (E, seed = 5) => { const r = rng(seed); return range(E).map(() => [r.gauss() * 0.9, r.gauss() * 0.9]); };
/* router: logits z = W x, p = softmax(z); top-k experts, gate weights = softmax restricted to the selected logits (as in Mixtral) */
function moeRoute(W, X, k) {
  return X.map(x => {
    const z = W.map(w => w[0] * x[0] + w[1] * x[1]), p = softmax(z), idx = range(W.length).sort((a, b) => z[b] - z[a]).slice(0, k), g = softmax(idx.map(i => z[i]));
    return { z, p, idx, g };
  });
}
function moeLoad(R, E) { const f = new Array(E).fill(0); let tot = 0; R.forEach(r => r.idx.forEach(i => { f[i]++; tot++; })); return f.map(v => v / tot); }
const moeMeanP = (R, E) => range(E).map(j => sum(R.map(r => r.p[j])) / R.length);
/* Switch-style auxiliary loss: E * sum_i f_i P_i  (= 1 when load and probabilities are uniform) */
function moeAux(R, E) { const f = moeLoad(R, E), P = moeMeanP(R, E); return E * sum(range(E).map(i => f[i] * P[i])); }
const moeCE = (R, lab, E) => -sum(R.map((r, n) => Math.log(r.p[lab[n] % E] + 1e-12))) / R.length;
/* gradient wrt W of  CE + lam * aux, treating the (non-differentiable) assignment fractions f as constants */
function moeGrad(W, X, lab, k, lam) {
  const E = W.length, R = moeRoute(W, X, k), f = moeLoad(R, E), N = X.length, G = W.map(() => [0, 0]);
  R.forEach((r, n) => {
    const fp = sum(range(E).map(i => f[i] * r.p[i]));
    for (let j = 0; j < E; j++) {
      const dz = (r.p[j] - (j === lab[n] % E ? 1 : 0)) / N + lam * (E / N) * r.p[j] * (f[j] - fp);
      G[j][0] += dz * X[n][0]; G[j][1] += dz * X[n][1];
    }
  });
  return G;
}
function moeTrain(W, X, lab, k, lam, steps, lr = 0.5) {
  let V = W.map(w => w.slice());
  for (let s = 0; s < steps; s++) { const G = moeGrad(V, X, lab, k, lam); V = V.map((w, j) => [w[0] - lr * G[j][0], w[1] - lr * G[j][1]]); }
  return V;
}
/* capacity: each expert accepts at most ceil(cf * N * k / E) assignments, in token order; the rest are dropped */
function moeDropped(R, E, k, cf) {
  const cap = Math.ceil(cf * R.length * k / E), cnt = new Array(E).fill(0); let drop = 0;
  R.forEach(r => r.idx.forEach(i => { if (cnt[i] >= cap) drop++; else cnt[i]++; }));
  return { drop: drop / (R.length * k), cap };
}
/* SwiGLU transformer with MoE FFN: parameters stored vs parameters used per token */
function moeParams(o) {
  const { d, L, hid, E, k, heads, kv, V } = o, dh = d / heads, attn = 2 * d * d + 2 * d * kv * dh, expert = 3 * d * hid, router = E > 1 ? d * E : 0, norms = 2 * d;
  const emb = 2 * V * d, shared = L * (attn + router + norms) + emb + d;
  return { total: shared + L * E * expert, active: shared + L * k * expert, shared, expertAll: L * E * expert, expertActive: L * k * expert };
}
/* expected fraction of experts touched by a batch of B tokens when routing is uniform at random */
const moeTouched = (E, k, B) => 1 - Math.pow(1 - k / E, B);
LAB.tests.moe = { moeData, moeInit, moeRoute, moeLoad, moeAux, moeCE, moeGrad, moeTrain, moeDropped, moeParams, moeTouched };

registerModule({
  id: 'm11', n: 11, title: 'Mixture of Experts', eyebrow: 'parametri tanti, calcolo poco', heading: 'Mixture of Experts: molti blocchi, ne usi pochi',
  lede: 'Nel Transformer denso ogni token attraversa tutti i parametri. In un modello MoE il blocco MLP viene replicato in tanti "esperti" e un piccolo router sceglie, token per token, quali usarne. Il modello ha molti parametri ma ne attiva pochi.',
  build(root) {
    root.append(el('div', { html: `
      <p>Il modulo 6 diceva che l'MLP contiene la maggior parte dei parametri del blocco. Idea del MoE: sostituire <i>un</i> MLP con <var>E</var> MLP indipendenti (gli <b>esperti</b>) e aggiungere un <b>router</b>, un piccolo strato lineare che dato il vettore del token produce un punteggio per ciascun esperto. Si tengono i <var>k</var> migliori (di solito 1 o 2, in alcuni modelli di più), le loro uscite si combinano pesate dai punteggi.</p>
      <p>Attenzione al nome: gli esperti <b>non</b> sono "esperto di matematica", "esperto di legge". La specializzazione che emerge è spesso più sottile e in parte sintattica; qui nel laboratorio la rendiamo evidente con punti in un piano per poter <i>vedere</i> come si comporta il router.</p>` }));

    const DATA = moeData();
    let E = 4, K = 1, W = moeInit(E), cf = 1.25;
    const COLS = t => [t.accent, t.amber, t.teal, t.rose, '#8b6fb0', '#b0743b', '#4a8f8f', '#9c9c3a'];
    const sE = slider({ label: 'esperti E', min: 2, max: 8, value: 4, onInput: v => { E = v; if (K > E) { K = E; sK.set(K); } W = moeInit(E); upd(); } });
    const sK = slider({ label: 'esperti attivi k', min: 1, max: 4, value: 1, onInput: v => { K = Math.min(v, E); upd(); } });
    const sL = slider({ label: 'peso loss di bilanciamento λ', min: 0, max: 16, step: 0.25, value: 0, fmt: v => v.toFixed(2), onInput: () => { } });
    const sC = slider({ label: 'capacity factor', min: 1, max: 2, step: 0.05, value: 1.25, fmt: v => v.toFixed(2), onInput: v => { cf = v; upd(); } });
    const cv = canvasEl(340), cvl = canvasEl(200), out = el('div', { class: 'readout' }); let hover = null;
    const cp = setupCanvas(cv, (ctx, w, h, t) => {
      const side = Math.min(w, h), ox = (w - side) / 2, S = side / 6, tx = x => ox + (x + 3) * S, ty = y => (3 - y) * S, cols = COLS(t);
      const n = 36, cell = side / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const x = -3 + (j + 0.5) * 6 / n, y = 3 - (i + 0.5) * 6 / n, z = W.map(wj => wj[0] * x + wj[1] * y), b = z.indexOf(Math.max(...z));
        ctx.fillStyle = rgba(cols[b], 0.13); ctx.fillRect(ox + j * cell, i * cell, cell + 0.5, cell + 0.5);
      }
      ctx.strokeStyle = t.line; ctx.strokeRect(ox, 0, side, side);
      const R = moeRoute(W, DATA.X, K);
      DATA.X.forEach((x, i) => {
        ctx.fillStyle = cols[R[i].idx[0]]; ctx.beginPath(); ctx.arc(tx(x[0]), ty(x[1]), 3, 0, 7); ctx.fill();
        if (K > 1) { ctx.strokeStyle = cols[R[i].idx[1]]; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(tx(x[0]), ty(x[1]), 4.8, 0, 7); ctx.stroke(); }
      });
      W.forEach((wj, j) => { const m = Math.hypot(wj[0], wj[1]) || 1, L = Math.min(2.4, m) * S; arrow(ctx, tx(0), ty(0), tx(0) + wj[0] / m * L, ty(0) - wj[1] / m * L, cols[j], 2.5, 8); text(ctx, 'E' + (j + 1), tx(0) + wj[0] / m * (L + 12), ty(0) - wj[1] / m * (L + 12) + 4, { align: 'center', color: cols[j], size: 12, weight: 700 }); });
      text(ctx, 'colore del punto = esperto con punteggio più alto · anello = secondo esperto (se k ≥ 2) · frecce = vettori del router', ox + 6, h - 6, { color: t.muted, size: 10 });
    });
    const cl = setupCanvas(cvl, (ctx, w, h, t) => {
      const R = moeRoute(W, DATA.X, K), f = moeLoad(R, E), P = moeMeanP(R, E), cols = COLS(t), left = 44, bot = 30, top = 22, bw = (w - left - 8) / E, mx = Math.max(0.5, ...f) * 1.1, Y = v => h - bot - v / mx * (h - bot - top);
      for (let i = 0; i <= 4; i++) { const v = mx * i / 4; ctx.strokeStyle = rgba(t.line, 0.7); ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(w - 6, Y(v)); ctx.stroke(); text(ctx, f2(v, 2), left - 5, Y(v) + 4, { align: 'right', color: t.muted, size: 10 }); }
      const cap = cf / E; // capacity expressed as a fraction of all assignments
      f.forEach((v, j) => {
        const x = left + j * bw + bw * 0.12, ww = bw * 0.76;
        ctx.fillStyle = cols[j]; ctx.fillRect(x, Y(v), ww / 2, h - bot - Y(v));
        ctx.fillStyle = rgba(t.muted, 0.5); ctx.fillRect(x + ww / 2, Y(P[j]), ww / 2, h - bot - Y(P[j]));
        text(ctx, 'E' + (j + 1), x + ww / 2, h - bot + 14, { align: 'center', color: t.ink, size: 11, weight: 700 });
      });
      ctx.strokeStyle = t.ink; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(left, Y(1 / E)); ctx.lineTo(w - 6, Y(1 / E)); ctx.stroke();
      ctx.strokeStyle = t.rose; ctx.beginPath(); ctx.moveTo(left, Y(cap)); ctx.lineTo(w - 6, Y(cap)); ctx.stroke(); ctx.setLineDash([]);
      text(ctx, 'colorata: quota di assegnamenti fᵢ · grigia: probabilità media Pᵢ · tratteggio nero: carico ideale 1/E · rosa: capacità', left, 12, { color: t.muted, size: 10 });
    });
    function upd() {
      const R = moeRoute(W, DATA.X, K), f = moeLoad(R, E), ce = moeCE(R, DATA.lab, E), aux = moeAux(R, E), dr = moeDropped(R, E, K, cf), imb = Math.max(...f) * E;
      out.innerHTML = `loss del "compito" (cross-entropy verso l'esperto giusto per ogni cluster): <b>${f2(ce, 2)}</b> · loss di bilanciamento <b>${f2(aux, 2)}</b> (ideale = 1) · esperto più carico: <b>${f2(imb, 2)}×</b> la quota ideale · assegnamenti scartati per capacità (${dr.cap} token per esperto): <b>${f2(dr.drop * 100, 1)}%</b>`;
      cp.redraw(); cl.redraw();
    }
    const train = n => { W = moeTrain(W, DATA.X, DATA.lab, K, sL.get(), n); upd(); };
    root.append(lab('Laboratorio · il router, il carico e la loss di bilanciamento',
      el('p', { html: 'Ogni punto è un token (240, in 4 gruppi di dimensioni diverse: 45%, 30%, 15%, 10%). Il router è una matrice <var>W</var> (un vettore per esperto): il punteggio di un token <var>x</var> per l\'esperto <var>j</var> è <var>W</var><sub>j</sub>·<var>x</var>. Nel "compito" giocattolo ogni gruppo ha un esperto giusto (gruppo 1 → E1, gruppo 2 → E2, …). Premi <b>Allena</b> per muovere i vettori del router con la discesa del gradiente.' }),
      controls(sE.el, sK.el), controls(sL.el, sC.el),
      controls(button('Allena 50 passi', () => train(50), 'primary'), button('Allena 300 passi', () => train(300)), button('Reinizializza router', () => { W = moeInit(E); upd(); })),
      cv, out, cvl,
      el('p', { class: 'muted', html: 'Con λ = 0 il router impara il compito ma eredita lo squilibrio dei dati: l\'esperto del gruppo grande è sovraccarico e, con capacity bassa, scarta token. Con <var>k</var> = 1, λ ≈ 8 e qualche centinaio di passi la loss di bilanciamento riporta i carichi vicino a 1/<var>E</var>, ma a un prezzo: parte dei token finisce a un esperto "sbagliato" e la loss del compito sale. Con λ molto alto il carico oscilla da un passo all\'altro, perché <var>f</var><sub>i</sub> è discreto e cambia a salti. Con <var>k</var> = 2 il carico è già più uniforme perché il secondo esperto distribuisce i token. Nei modelli veri si cerca il compromesso con un λ piccolo (dell\'ordine di 10<sup>-2</sup>) perché il compito è molto più ricco di questo.' })));
    upd();

    /* ---- Lab 2: parameters stored vs used ---- */
    const PRE = {
      mix: { name: 'Mixtral 8×7B (8 esperti, k=2)', d: 4096, L: 32, hid: 14336, E: 8, k: 2, heads: 32, kv: 8, V: 32000 },
      dense: { name: 'Denso equivalente (1 esperto)', d: 4096, L: 32, hid: 14336, E: 1, k: 1, heads: 32, kv: 8, V: 32000 },
      many: { name: 'Più esperti, più piccoli (64, k=6)', d: 4096, L: 32, hid: 1792, E: 64, k: 6, heads: 32, kv: 8, V: 32000 },
    };
    let pk = 'mix';
    const psel = select(Object.entries(PRE).map(([k, v]) => [k, v.name]), pk, v => { pk = v; upd2(); }, 'configurazione');
    const cv2 = canvasEl(170), out2 = el('div', { class: 'readout' });
    const c2 = setupCanvas(cv2, (ctx, w, h, t) => {
      const q = moeParams(PRE[pk]), mx = Math.max(...Object.values(PRE).map(o => moeParams(o).total)), left = 120, s = (w - left - 14) / mx;
      const rows = [['parametri totali', [[q.shared, rgba(t.muted, 0.6)], [q.expertAll, t.accent]]], ['usati per token', [[q.shared, rgba(t.muted, 0.6)], [q.expertActive, t.amber]]]];
      rows.forEach(([name, segs], i) => {
        const y = 28 + i * 54; text(ctx, name, 6, y + 22, { color: t.ink, size: 12, sans: true, weight: 700 }); let x = left;
        segs.forEach(([v, c]) => { ctx.fillStyle = c; ctx.fillRect(x, y, v * s, 32); x += v * s; });
        text(ctx, human(segs.reduce((a, b) => a + b[0], 0)), x + 6, y + 21, { color: t.ink, size: 12, weight: 700 });
      });
      ctx.fillStyle = rgba(t.muted, 0.6); ctx.fillRect(left, h - 18, 10, 10); text(ctx, 'embedding + attention + router (sempre usati)', left + 14, h - 9, { color: t.muted, size: 10 });
      ctx.fillStyle = t.accent; ctx.fillRect(left + 270, h - 18, 10, 10); text(ctx, 'esperti', left + 284, h - 9, { color: t.muted, size: 10 });
    });
    function upd2() {
      const q = moeParams(PRE[pk]);
      out2.innerHTML = `totali <b>${human(q.total)}</b> · attivi per token <b>${human(q.active)}</b> (${f2(q.active / q.total * 100, 0)}%) · pesi da tenere in memoria (fp16): <b>${f2(q.total * 2 / 1e9, 1)} GB</b> · calcolo per token ≈ ${f2(2 * q.active / 1e9, 1)} GFLOP`;
      c2.redraw();
    }
    root.append(lab('Laboratorio · parametri totali contro parametri attivi', controls(psel), cv2, out2,
      el('p', { class: 'muted', html: 'Conti fatti con le formule del modulo 0 (attention con GQA, MLP SwiGLU, embedding non condivisi). Il Mixtral qui è l\'architettura pubblicata; la terza configurazione è un esempio costruito per illustrare un MoE con esperti piccoli, non un modello reale.' })));
    upd2();

    /* ---- Lab 3: batching erodes the MoE advantage at decode ---- */
    const sE3 = slider({ label: 'esperti E', min: 2, max: 128, value: 8, onInput: () => upd3() });
    const sK3 = slider({ label: 'esperti attivi k', min: 1, max: 8, value: 2, onInput: () => upd3() });
    const cv3 = canvasEl(220), out3 = el('div', { class: 'readout' });
    const c3 = setupCanvas(cv3, (ctx, w, h, t) => {
      const e = sE3.get(), k = Math.min(sK3.get(), e), Bs = range(128).map(i => i + 1);
      chart(ctx, w, h, t, [{ xs: Bs, ys: Bs.map(b => moeTouched(e, k, b)), color: t.accent, lw: 2.5, label: 'frazione di esperti letti dalla memoria' }, { xs: [1, 128], ys: [k / e, k / e], color: t.amber, dash: [5, 4], lw: 1.8, label: 'frazione usata da un solo token (k/E)' }], { xlabel: 'token decodificati insieme (batch)', ymin: 0, ymax: 1.02, nx: 4, yfmt: v => f2(v, 1) });
    });
    function upd3() {
      const e = sE3.get(), k = Math.min(sK3.get(), e);
      out3.innerHTML = `batch 1: <b>${f2(moeTouched(e, k, 1) * 100, 0)}%</b> degli esperti · batch 8: <b>${f2(moeTouched(e, k, 8) * 100, 0)}%</b> · batch 32: <b>${f2(moeTouched(e, k, 32) * 100, 0)}%</b>`;
      c3.redraw();
    }
    root.append(lab('Laboratorio · MoE e batching in inferenza', controls(sE3.el, sK3.el), cv3, out3,
      el('p', { class: 'muted', html: 'Nel modulo 9 il decode era limitato dalla lettura dei pesi. Un MoE usa pochi esperti per token, quindi con batch 1 legge solo una piccola parte dei pesi. Ma con molti token insieme i router (supponendo scelte indipendenti e uniformi) toccano quasi tutti gli esperti: il vantaggio di velocità sparisce e restano il costo di memoria e la comunicazione tra GPU. In pratica le scelte non sono uniformi, ma il senso della curva resta.' })));
    upd3();

    root.append(el('h2', { text: 'I punti da portarsi via' }));
    root.append(el('ul', { class: 'prose', html: `
      <li>Un MoE <b>scollega</b> capacità (parametri totali) e costo per token (parametri attivi): più conoscenza immagazzinata a parità di calcolo.</li>
      <li>Il prezzo è la memoria (tutti gli esperti devono stare da qualche parte), la comunicazione (i token vanno inviati alle GPU che ospitano gli esperti scelti) e l'instabilità del training.</li>
      <li><b>Load balancing</b>: senza vincoli il router tende a favorire pochi esperti, che migliorano più in fretta e attirano ancora più token (<i>rich get richer</i>). Si aggiungono loss ausiliarie, oppure bias di bilanciamento aggiornati durante il training.</li>
      <li><b>Capacity</b>: ogni esperto accetta un numero massimo di token per batch; l'eccesso viene scartato (passa solo per la connessione residua) o reindirizzato.</li>
      <li>Varianti: esperti condivisi sempre attivi più esperti instradati, tanti esperti piccoli (fine-grained), routing "expert choice" (sono gli esperti a scegliere i token).</li>` }));

    root.append(details('math', 'La matematica: router, gate e loss ausiliaria', 'formule', `
      <p><b>Router e gate.</b> Con <var>x</var> ∈ ℝ<sup><var>d</var></sup> l'uscita dell'attention e <var>W</var><sub>r</sub> ∈ ℝ<sup><var>E</var>×<var>d</var></sup>: <var>z</var> = <var>W</var><sub>r</sub><var>x</var>, e sia <var>S</var> l'insieme degli indici dei <var>k</var> punteggi più alti. Nello stile di Mixtral i pesi di combinazione sono la softmax sui soli punteggi selezionati:</p>
      <span class="eq"><var>y</var> = Σ<sub><var>i</var>∈<var>S</var></sub> <var>g</var><sub>i</sub> · <var>FFN</var><sub>i</sub>(<var>x</var>), &nbsp; <var>g</var><sub>i</sub> = ${frac('exp(<var>z</var><sub>i</sub>)', 'Σ<sub><var>j</var>∈<var>S</var></sub> exp(<var>z</var><sub>j</sub>)')}</span>
      <p>La selezione top-<var>k</var> non è differenziabile, ma i pesi <var>g</var><sub>i</sub> sì: il router riceve gradiente attraverso di essi.</p>
      <p><b>Loss di bilanciamento (stile Switch Transformer).</b> Su un batch di <var>N</var> token, sia <var>f</var><sub>i</sub> la frazione di assegnamenti ricevuti dall'esperto <var>i</var> (dato discreto) e <var>P</var><sub>i</sub> = (1/<var>N</var>) Σ<sub>n</sub> <var>p</var><sub>n,i</sub> la probabilità media del router per <var>i</var> (differenziabile):</p>
      <span class="eq">ℒ<sub>aux</sub> = <var>E</var> · Σ<sub>i=1..E</sub> <var>f</var><sub>i</sub> <var>P</var><sub>i</sub></span>
      <p>Vale 1 quando carico e probabilità sono uniformi (<var>f</var><sub>i</sub> = <var>P</var><sub>i</sub> = 1/<var>E</var>) ed è maggiore di 1 quando l'esperto più usato è anche quello con le probabilità più alte. Il gradiente rispetto ai punteggi, con <var>f</var> trattato come costante, è</p>
      <span class="eq">${frac('∂ℒ<sub>aux</sub>', '∂<var>z</var><sub>n,j</sub>')} = ${frac('<var>E</var>', '<var>N</var>')} <var>p</var><sub>n,j</sub> ( <var>f</var><sub>j</sub> − Σ<sub>i</sub> <var>f</var><sub>i</sub> <var>p</var><sub>n,i</sub> )</span>
      <p>cioè per un token si abbassa il punteggio degli esperti con carico superiore alla media pesata e si alza quello degli altri. Il laboratorio usa questa espressione (verificata con differenze finite).</p>
      <p><b>Parametri.</b> Per layer: attention <var>A</var> (condivisa), router <var>d</var>·<var>E</var>, e <var>E</var> esperti da 3<var>d</var>·<var>h</var> (SwiGLU). Totali ≈ <var>L</var>(<var>A</var> + <var>E</var>·3<var>dh</var>) + embedding; attivi ≈ <var>L</var>(<var>A</var> + <var>k</var>·3<var>dh</var>) + embedding.</p>
      <p><b>Batching.</b> Con scelte indipendenti e uniformi, la probabilità che un dato esperto non sia scelto da nessun token di un batch di <var>B</var> è (1 − <var>k</var>/<var>E</var>)<sup><var>B</var></sup>, quindi la frazione attesa di esperti letti è 1 − (1 − <var>k</var>/<var>E</var>)<sup><var>B</var></sup>.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Un MoE con 8 esperti e k = 2 ha, rispetto a un denso con lo stesso MLP, circa…', opts: ['8 volte i parametri totali ma gli stessi attivi per un MLP doppio', '8 volte i parametri dell\'MLP e solo 2 attivi per token', 'gli stessi parametri e il doppio di calcolo'], ans: 1, why: 'Si memorizzano 8 MLP, per ogni token se ne attraversano 2. L\'attention e gli embedding sono invece condivisi.' },
      { q: 'Perché serve una loss di bilanciamento?', opts: ['Per rendere il router differenziabile', 'Perché il router tende a concentrare i token su pochi esperti, che così si allenano di più e attirano altri token', 'Per ridurre il vocabolario'], ans: 1, why: 'È un circolo vizioso (rich get richer): senza correttivi molti esperti restano sottoutilizzati e l\'hardware lavora in modo sbilanciato.' },
      { q: 'Perché con batch grandi il MoE perde parte del vantaggio in decode?', opts: ['Perché i token diversi attivano esperti diversi e quasi tutti gli esperti vengono letti dalla memoria', 'Perché la softmax diventa più lenta', 'Perché k aumenta'], ans: 0, why: 'La lettura dei pesi, che domina il decode, non è più ridotta a k/E ma si avvicina all\'intero modello.' },
    ]));
  },
});
