'use strict';
/* Module 7 - from logits to a token: temperature, top-k, top-p */
function sampleDist(z, T, k, p) {
  const n = z.length, p0 = softmax(z, Math.max(T, 1e-6)), order = range(n).sort((a, b) => p0[b] - p0[a]), keep = new Array(n).fill(false);
  let cum = 0;
  for (let r = 0; r < n; r++) { if (r >= k || (r > 0 && cum >= p)) break; keep[order[r]] = true; cum += p0[order[r]]; }
  const z1 = sum(p0.map((v, i) => keep[i] ? v : 0)), p1 = p0.map((v, i) => keep[i] ? v / z1 : 0);
  return { p0, p1, keep, order };
}
LAB.tests.sampleDist = sampleDist;

registerModule({
  id: 'm7', n: 7, title: 'Sampling', eyebrow: 'dai logit al token', heading: 'Da una distribuzione a una scelta',
  lede: 'Il modello non "dice" una parola: produce una distribuzione di probabilità su tutto il vocabolario. Come si sceglie da quella distribuzione cambia radicalmente il carattere del testo.',
  build(root) {
    root.append(el('div', { html: `
      <p>Dopo l'ultimo blocco, il vettore finale <var>h</var> della <b>ultima posizione</b> viene confrontato con ogni riga della matrice di embedding (<i>unembedding</i>): il prodotto scalare di <var>h</var> con la riga del token <var>v</var> è il suo <b>logit</b> <var>z</var><sub>v</sub>, un numero reale qualsiasi.
      La <b>softmax</b> li trasforma in probabilità. Poi si estrae. Ogni scelta qui sotto è una "manopola" usata davvero nelle API dei modelli.</p>` }));

    const PROMPTS = {
      'La capitale d\'Italia è': [['Roma', 9.2], ['Milano', 6.1], ['la', 5.0], ['una', 4.6], ['Torino', 4.1], ['Napoli', 3.9], ['stata', 3.8], ['il', 3.4], ['Firenze', 3.2], ['Venezia', 3.0], ['Bologna', 2.9], ['Parigi', 2.2]],
      'Il gatto dorme sul': [['divano', 8.0], ['letto', 7.3], ['tappeto', 6.1], ['pavimento', 5.5], ['cuscino', 4.9], ['tavolo', 4.4], ['balcone', 4.1], ['tetto', 3.8], ['computer', 3.0], ['sole', 2.0], ['mare', 1.1], ['frigorifero', 0.6]],
      'C\'era una volta': [['un', 7.4], ['una', 7.1], [',', 5.2], ['in', 4.8], ['il', 4.3], ['la', 3.9], ['tanto', 3.4], ['quando', 2.8], ['e', 2.2], ['oggi', 1.5], ['un po\'', 1.2], ['robot', 0.8]],
    };
    let pk = 'Il gatto dorme sul', D = null, counts = null, last = -1;
    const sT = slider({ label: 'temperatura T', min: 0.05, max: 3, step: 0.05, value: 1, fmt: v => v.toFixed(2), onInput: () => upd() });
    const sK = slider({ label: 'top-k', min: 1, max: 12, value: 12, onInput: () => upd() });
    const sP = slider({ label: 'top-p', min: 0.05, max: 1, step: 0.05, value: 1, fmt: v => v.toFixed(2), onInput: () => upd() });
    const cv = canvasEl(300), tcv = canvasEl(210), out = el('div', { class: 'readout' }), picked = el('div', { class: 'readout' });
    const names = () => PROMPTS[pk].map(x => x[0]), logits = () => PROMPTS[pk].map(x => x[1]);

    const bc = setupCanvas(cv, (ctx, w, h, t) => {
      if (!D) return;
      const n = D.p0.length, left = 40, bot = 62, top = 18, bw = (w - left - 10) / n, mx = Math.max(0.05, ...D.p0) * 1.12, Y = v => h - bot - v / mx * (h - bot - top);
      for (let i = 0; i <= 4; i++) { const v = mx * i / 4, y = Y(v); ctx.strokeStyle = rgba(t.line, 0.7); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - 6, y); ctx.stroke(); text(ctx, f2(v, v < 0.1 ? 3 : 2), left - 5, y + 4, { align: 'right', color: t.muted, size: 10 }); }
      const total = counts ? sum(counts) : 0;
      names().forEach((nm, i) => {
        const x = left + i * bw + bw * 0.14, ww = bw * 0.72, kept = D.keep[i];
        ctx.fillStyle = rgba(t.muted, 0.28); ctx.fillRect(x, Y(D.p0[i]), ww, h - bot - Y(D.p0[i]));
        if (kept) { ctx.fillStyle = t.accent; ctx.fillRect(x, Y(D.p1[i]), ww, h - bot - Y(D.p1[i])); }
        else { ctx.strokeStyle = t.rose; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, h - bot - 2); ctx.lineTo(x + ww, h - bot - 14); ctx.moveTo(x, h - bot - 14); ctx.lineTo(x + ww, h - bot - 2); ctx.stroke(); }
        if (total > 0) { const f = counts[i] / total; ctx.strokeStyle = t.amber; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x - 1, Y(f)); ctx.lineTo(x + ww + 1, Y(f)); ctx.stroke(); }
        if (i === last) { ctx.fillStyle = t.amber; ctx.beginPath(); ctx.moveTo(x + ww / 2, Y(Math.max(D.p0[i], D.p1[i])) - 4); ctx.lineTo(x + ww / 2 - 6, Y(Math.max(D.p0[i], D.p1[i])) - 14); ctx.lineTo(x + ww / 2 + 6, Y(Math.max(D.p0[i], D.p1[i])) - 14); ctx.fill(); }
        ctx.save(); ctx.translate(x + ww / 2, h - bot + 10); ctx.rotate(-0.6); text(ctx, nm, 0, 0, { align: 'right', color: kept ? t.ink : t.muted, size: 11, sans: true, weight: kept ? 700 : 500 }); ctx.restore();
      });
      const lg = [['probabilità dopo la temperatura', rgba(t.muted, 0.5)], ['probabilità finale (dopo top-k / top-p)', t.accent], ['frequenza osservata', t.amber]];
      let lx = left; lg.forEach(([s, c]) => { ctx.fillStyle = c; ctx.fillRect(lx, 4, 10, 10); text(ctx, s, lx + 14, 13, { color: t.muted, size: 10 }); lx += 28 + s.length * 6; });
    });

    const tc = setupCanvas(tcv, (ctx, w, h, t) => {
      const z = logits(), Ts = range(60).map(i => 0.05 + i * 0.05), pt = Ts.map(T => Math.max(...softmax(z, T))), H = Ts.map(T => entropy(softmax(z, T)) / Math.log(z.length));
      const c = chart(ctx, w, h, t, [{ xs: Ts, ys: pt, color: t.accent, lw: 2.5, label: 'p(token più probabile)' }, { xs: Ts, ys: H, color: t.amber, lw: 2.5, label: 'entropia normalizzata' }], { xlabel: 'temperatura T', ymin: 0, ymax: 1, xmin: 0, xmax: 3, nx: 6 });
      ctx.strokeStyle = t.rose; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(c.X(sT.get()), c.y1); ctx.lineTo(c.X(sT.get()), c.y0); ctx.stroke(); ctx.setLineDash([]);
    });

    function upd(resetCounts = true) {
      if (resetCounts) { counts = new Array(12).fill(0); last = -1; picked.textContent = ''; }
      D = sampleDist(logits(), sT.get(), sK.get(), sP.get());
      const H = entropy(D.p1) / Math.LN2, kept = D.keep.filter(Boolean).length, order = D.order;
      out.innerHTML = `token possibili dopo i filtri: <b>${kept}</b> su 12 · entropia <b>${f2(H)} bit</b> (≈ ${f2(Math.pow(2, H), 1)} scelte equiprobabili) · più probabile: <b>${names()[order[0]]}</b> ${f2(D.p1[order[0]] * 100, 1)}%`;
      bc.redraw(); tc.redraw();
    }
    function draw(n) {
      const cs = []; let acc = 0; D.p1.forEach(p => { acc += p; cs.push(acc); });
      for (let s = 0; s < n; s++) { const u = Math.random() * acc; let i = cs.findIndex(c => u <= c); if (i < 0) i = cs.length - 1; counts[i]++; last = i; }
      const seq = names()[last];
      picked.innerHTML = n === 1 ? `Token estratto: <b>${seq}</b>. Frase: "${pk} <b>${seq}</b>…"` : `${sum(counts)} estrazioni accumulate: la frequenza (tratto arancione) si avvicina alla probabilità finale.`;
      bc.redraw();
    }
    const sel = select(Object.keys(PROMPTS).map(k => [k, k]), pk, v => { pk = v; upd(); }, 'contesto');
    root.append(lab('Laboratorio · distribuzione del prossimo token (logit scelti a mano, 12 candidati)',
      controls(sel), controls(sT.el, sK.el, sP.el),
      controls(button('Estrai 1 token', () => draw(1), 'primary'), button('Estrai 500', () => draw(500)), button('Azzera', () => upd()), button('Greedy: T → 0', () => { sT.set(0.05); sK.set(1); sP.set(1); upd(); })),
      cv, out, picked,
      el('p', { class: 'muted', html: 'Un vocabolario vero ha decine di migliaia di candidati; qui ce ne sono 12 per leggibilità. Il filtro top-p tiene i token più probabili finché la loro somma raggiunge <var>p</var>.' })));
    root.append(lab('Laboratorio · che cosa fa davvero la temperatura', tcv,
      el('p', { html: 'A <var>T</var> → 0 tutta la probabilità va al token più probabile (e l\'entropia a 0): decisioni deterministiche. A <var>T</var> grande la distribuzione tende all\'uniforme: ogni token è ugualmente plausibile. A <var>T</var> = 1 si campiona dalla distribuzione "così com\'è" prodotta dal modello. La linea rosa è il valore scelto sopra.' })));
    upd();

    root.append(el('h2', { text: 'Le strategie' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Greedy</b>: si prende sempre il più probabile. Deterministico, ma tende a ripetizioni e testi piatti.</li>
      <li><b>Temperatura</b>: ridimensiona i logit prima della softmax. Bassa per compiti con una risposta giusta (codice, estrazione di dati), più alta per scrittura creativa.</li>
      <li><b>Top-k</b>: si scartano tutti tranne i <var>k</var> più probabili. Semplice ma cieco: a volte 5 candidati sono pochi, a volte troppi.</li>
      <li><b>Top-p (nucleus)</b>: si tiene il più piccolo insieme di token la cui probabilità cumulata raggiunge <var>p</var>. Si adatta: pochi token quando il modello è sicuro, molti quando è incerto.</li>
      <li><b>Altre</b>: min-p (soglia relativa al token più probabile), penalità di ripetizione/frequenza, beam search (si tengono più continuazioni in parallelo, usata più in traduzione che in chat).</li>` }));
    root.append(el('p', { html: 'Un dettaglio che spiega molto: anche a <var>T</var> = 0 le risposte di un LLM in produzione possono variare leggermente, perché le operazioni in virgola mobile su GPU non sono sempre eseguite nello stesso ordine e due logit quasi uguali possono invertirsi.' }));

    root.append(details('math', 'La matematica: logit, softmax con temperatura, entropia', 'formule', `
      <p><b>Dal vettore ai logit.</b> Con <var>h</var> ∈ ℝ<sup><var>d</var></sup> uscita della norma finale e <var>E</var> ∈ ℝ<sup><var>V</var>×<var>d</var></sup> (spesso la stessa matrice dell'embedding):</p>
      <span class="eq"><var>z</var> = <var>E h</var> (+ <var>b</var>), &nbsp;&nbsp; <var>z</var><sub>v</sub> = <var>E</var><sub>v</sub>·<var>h</var></span>
      <p><b>Softmax con temperatura.</b></p>
      <span class="eq"><var>p</var><sub>v</sub>(<var>T</var>) = ${frac('exp(<var>z</var><sub>v</sub> / <var>T</var>)', 'Σ<sub>u</sub> exp(<var>z</var><sub>u</sub> / <var>T</var>)')}</span>
      <p>È la distribuzione di Boltzmann con "energia" −<var>z</var> e temperatura <var>T</var>. Per <var>T</var> → 0 diventa un one-hot sull'argmax; per <var>T</var> → ∞ diventa uniforme. Le <i>differenze</i> di logit sono log-odds: <var>z</var><sub>a</sub> − <var>z</var><sub>b</sub> = <var>T</var>·ln(<var>p</var><sub>a</sub>/<var>p</var><sub>b</sub>).</p>
      <p><b>Entropia</b> (in bit): <span class="m"><var>H</var> = −Σ<sub>v</sub> <var>p</var><sub>v</sub> log<sub>2</sub> <var>p</var><sub>v</sub></span>. 2<sup><var>H</var></sup> è il numero di scelte equiprobabili equivalenti (la <i>perplessità</i> della distribuzione).</p>
      <p><b>Top-p.</b> Ordina i token per probabilità decrescente e sia <var>S</var><sub>p</sub> il più piccolo insieme con Σ<sub><var>v</var>∈<var>S</var></sub> <var>p</var><sub>v</sub> ≥ <var>p</var>. Si pone <var>p</var>′<sub>v</sub> = <var>p</var><sub>v</sub> / Σ<sub><var>u</var>∈<var>S</var></sub><var>p</var><sub>u</sub> per <var>v</var> ∈ <var>S</var>, 0 altrove, e si estrae da <var>p</var>′.</p>
      <p><b>Stabilità numerica.</b> Si calcola softmax(<var>z</var>) come exp(<var>z</var> − max <var>z</var>)/Σ…: stesso risultato, nessun overflow. In training si usa direttamente il <i>log-sum-exp</i>.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Abbassando la temperatura da 1 a 0,2, la distribuzione…', opts: ['diventa più piatta', 'diventa più concentrata sui token già più probabili', 'resta uguale ma cambia l\'ordine'], ans: 1, why: 'Dividere i logit per un T piccolo amplifica le differenze: il massimo prende quasi tutta la probabilità.' },
      { q: 'Qual è il vantaggio del top-p rispetto al top-k?', opts: ['È più veloce', 'Il numero di candidati si adatta all\'incertezza del modello', 'Garantisce risposte corrette'], ans: 1, why: 'Se il modello è molto sicuro, il nucleo contiene pochi token; se è incerto, ne contiene molti. Il top-k ne tiene sempre lo stesso numero.' },
      { q: 'I logit sono…', opts: ['probabilità comprese tra 0 e 1', 'numeri reali non normalizzati, prima della softmax', 'indici dei token'], ans: 1, why: 'Sono punteggi arbitrari; solo le loro differenze contano per la softmax.' },
    ]));
  },
});
