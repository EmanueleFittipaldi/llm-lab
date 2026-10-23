'use strict';
/* Module 13 - quantization: symmetric round-to-nearest with per-tensor or per-group scales */
/* group = number of consecutive weights sharing one scale (Infinity = one scale for the whole tensor) */
function quantize(w, bits, group) {
  const n = w.length, qmax = Math.pow(2, bits - 1) - 1, g = Math.min(group, n), deq = new Float64Array(n), scales = [];
  for (let s = 0; s < n; s += g) {
    const e = Math.min(n, s + g); let m = 0; for (let i = s; i < e; i++) m = Math.max(m, Math.abs(w[i]));
    const sc = m / qmax || 1; scales.push(sc);
    for (let i = s; i < e; i++) deq[i] = Math.max(-qmax, Math.min(qmax, Math.round(w[i] / sc))) * sc;
  }
  return { deq, scales, qmax };
}
const quantRMSE = (w, d) => Math.sqrt(w.reduce((s, v, i) => s + (v - d[i]) * (v - d[i]), 0) / w.length);
function quantWeights(n, sigma, nOut, mag, seed = 4) {
  const r = rng(seed), w = Float64Array.from({ length: n }, () => r.gauss() * sigma);
  for (let k = 0; k < nOut; k++) w[Math.floor((k + 0.5) * n / Math.max(1, nOut))] = (k % 2 ? -1 : 1) * sigma * mag;
  return w;
}
/* bits per weight actually stored: the payload plus one fp16 scale per group */
const quantBitsPerWeight = (bits, group) => bits + (isFinite(group) ? 16 / group : 0);
LAB.tests.quant = { quantize, quantRMSE, quantWeights, quantBitsPerWeight };

registerModule({
  id: 'm13', n: 13, title: 'Quantizzazione', eyebrow: 'meno bit per ogni peso', heading: 'Quantizzazione: far stare il modello nella memoria',
  lede: 'Un peso in fp16 occupa 16 bit. Ma ne servono davvero tutti? Memorizzare i pesi con 8 o 4 bit riduce la memoria di 2–4× (e, nel decode, che è limitato dalla lettura dei pesi, aumenta i token al secondo) in cambio di un piccolo errore di arrotondamento. Il trucco è dove si sceglie la scala.',
  build(root) {
    root.append(el('div', { html: `
      <p>La <b>quantizzazione</b> simmetrica più semplice sceglie una scala <var>s</var> e memorizza ogni peso come intero: <var>q</var> = round(<var>w</var>/<var>s</var>), limitato a [−<var>q</var><sub>max</sub>, <var>q</var><sub>max</sub>] con <var>q</var><sub>max</sub> = 2<sup>bit−1</sup> − 1. Per usarlo si ricostruisce <var>ŵ</var> = <var>s</var>·<var>q</var>. Il peso torna "arrotondato" al multiplo di <var>s</var> più vicino.</p>
      <p>Di solito <var>s</var> = max|<var>w</var>| / <var>q</var><sub>max</sub>, cosicché il peso più grande occupa tutta la scala. Qui nasce il problema: <b>un solo outlier</b> allarga la scala per tutti gli altri, che finiscono tutti sugli stessi pochi livelli. La soluzione standard: una scala per ogni <b>gruppo</b> di pesi (per esempio 32 o 128 consecutivi).</p>` }));

    const N = 2048, SIG = 0.02;
    const sBits = slider({ label: 'bit per peso', min: 2, max: 8, value: 4, onInput: () => upd() });
    let grpV = Infinity; const gsel = select([['Infinity', 'una scala per tutto il tensore'], ['256', 'gruppi da 256'], ['64', 'gruppi da 64'], ['32', 'gruppi da 32']], 'Infinity', v => { grpV = +v; upd(); }, 'granularità');
    const sOut = slider({ label: 'outlier', min: 0, max: 8, value: 2, onInput: () => upd() });
    const sMag = slider({ label: 'grandezza outlier (× σ)', min: 5, max: 60, value: 30, onInput: () => upd() });
    const cv = canvasEl(260), cvE = canvasEl(220), out = el('div', { class: 'readout' });
    let W = null, Q = null;
    const c1 = setupCanvas(cv, (ctx, w, h, t) => {
      if (!W) return;
      const left = 40, right = 10, top = 22, bot = 34, pw = w - left - right, ph = h - top - bot, lim = Math.max(...W.map(Math.abs)) * 1.05, nb = 90, X = v => left + (v + lim) / (2 * lim) * pw;
      const hist = new Array(nb).fill(0); W.forEach(v => { hist[Math.min(nb - 1, Math.floor((v + lim) / (2 * lim) * nb))]++; });
      const hm = Math.max(...hist), Yb = c => top + ph - Math.sqrt(c / hm) * ph;
      hist.forEach((c, i) => { ctx.fillStyle = rgba(t.muted, 0.45); ctx.fillRect(left + i * pw / nb, Yb(c), pw / nb - 0.5, top + ph - Yb(c)); });
      // quantisation levels of the first group
      const q0 = Q.qmax, sc = Q.scales[0];
      ctx.strokeStyle = rgba(t.accent, 0.8); ctx.lineWidth = 1;
      for (let k = -q0; k <= q0; k++) { const x = X(k * sc); if (x < left || x > left + pw) continue; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke(); }
      ctx.strokeStyle = t.line; ctx.strokeRect(left, top, pw, ph);
      text(ctx, 'istogramma dei pesi (radice quadrata dell\'altezza)  ·  righe blu = livelli rappresentabili dalla scala del primo gruppo', left, 13, { color: t.muted, size: 10 });
      [-lim, 0, lim].forEach(v => text(ctx, f2(v, 2), X(v), top + ph + 14, { align: 'center', color: t.muted, size: 10 }));
      text(ctx, `livelli distinti usati dai pesi entro 3σ: ${distinctLevels()}`, left, h - 6, { color: t.ink, size: 11 });
    });
    function distinctLevels() {
      const s = new Set(); W.forEach((v, i) => { if (Math.abs(v) < 3 * SIG) s.add(Math.round(Q.deq[i] / Q.scales[Math.floor(i / Math.min(grpV, N))])); }); return s.size;
    }
    const c2 = setupCanvas(cvE, (ctx, w, h, t) => {
      const bs = [2, 3, 4, 5, 6, 7, 8], mk = (grp) => bs.map(b => Math.log10(quantRMSE(W, quantize(W, b, grp).deq) / SIG));
      const c = chart(ctx, w, h, t, [
        { xs: bs, ys: mk(Infinity), color: t.rose, lw: 2.5, dots: true, label: 'una scala per tensore' }, { xs: bs, ys: mk(64), color: t.accent, lw: 2.5, dots: true, label: 'gruppi da 64' }, { xs: bs, ys: mk(32), color: t.teal, lw: 2.5, dots: true, label: 'gruppi da 32' }],
        { xlabel: 'bit per peso', ylabel: 'log₁₀ (errore RMS / σ)', nx: 6, ny: 4 });
      ctx.strokeStyle = t.ink; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(c.X(sBits.get()), c.y1); ctx.lineTo(c.X(sBits.get()), c.y0); ctx.stroke(); ctx.setLineDash([]);
    });
    function upd() {
      W = quantWeights(N, SIG, sOut.get(), sMag.get()); const grp = grpV; Q = quantize(W, sBits.get(), grp);
      const e = quantRMSE(W, Q.deq), sqnr = 20 * Math.log10(Math.sqrt(W.reduce((s, v) => s + v * v, 0) / N) / e);
      out.innerHTML = `scala primo gruppo: <b>${f2(Q.scales[0], 5)}</b> (σ dei pesi = ${SIG}) · errore RMS <b>${f2(e, 5)}</b> = ${f2(e / SIG * 100, 1)}% di σ · rapporto segnale/rumore <b>${f2(sqnr, 1)} dB</b> · bit effettivi per peso, scale incluse: <b>${f2(quantBitsPerWeight(sBits.get(), grp), 2)}</b>`;
      c1.redraw(); c2.redraw();
    }
    root.append(lab('Laboratorio · scala, outlier e granularità (quantizzatore vero su 2048 pesi gaussiani)',
      controls(sBits.el, gsel), controls(sOut.el, sMag.el), cv, out, cvE,
      el('p', { class: 'muted', html: 'Prova: 4 bit, nessun outlier → quasi tutti i pesi cadono su pochi livelli ma bene distribuiti. Poi aggiungi 2 outlier da 30σ con una sola scala: la scala si allarga di molte volte e i pesi centrali collassano su 1–3 livelli. Passando a gruppi da 32 solo i gruppi che contengono l\'outlier soffrono. Costo dei gruppi: una scala fp16 ogni 32 pesi aggiunge 0,5 bit per peso.' })));
    upd();

    /* ---- memory ---- */
    const MOD = [['Llama 3 8B', 8.03e9], ['Llama 3 70B', 70.6e9], ['Modello da 405B', 405e9]];
    const cvM = canvasEl(250), outM = el('div', { class: 'readout' });
    const cM = setupCanvas(cvM, (ctx, w, h, t) => {
      const fm = [['fp16', 16, Infinity], ['int8 (gruppi da 128)', 8, 128], ['4 bit (gruppi da 64)', 4, 64]], left = 150, right = 70, rowH = 22, top = 22, mx = 405e9 * 2 / 1e9, s = (w - left - right) / mx;
      let y = top; text(ctx, 'memoria dei soli pesi (GB) · linea: GPU da 80 GB', 4, 12, { color: t.muted, size: 10 });
      MOD.forEach(([nm, n]) => { text(ctx, nm, 4, y + 12, { color: t.ink, size: 12, sans: true, weight: 700 }); y += 16;
        fm.forEach(([fn, b, g], i) => { const gb = n * quantBitsPerWeight(b, g) / 8 / 1e9; ctx.fillStyle = [t.rose, t.accent, t.teal][i]; ctx.fillRect(left, y, gb * s, 15); text(ctx, fn, left - 6, y + 12, { align: 'right', color: t.muted, size: 10 }); text(ctx, f2(gb, 0) + ' GB', left + gb * s + 5, y + 12, { color: t.ink, size: 10 }); y += rowH - 4; });
        y += 6; });
      ctx.strokeStyle = t.amber; ctx.lineWidth = 2; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(left + 80 * s, top + 10); ctx.lineTo(left + 80 * s, h - 6); ctx.stroke(); ctx.setLineDash([]);
    });
    root.append(lab('Laboratorio · quanta memoria occupano i pesi', cvM, outM,
      el('p', { class: 'muted', html: 'Conto semplice: parametri × bit effettivi ÷ 8. Non include KV cache, attivazioni né overhead (modulo 9). Un modello da 70B in fp16 (≈141 GB) non entra in una GPU da 80 GB; a 4 bit con gruppi da 64 (≈ 4,25 bit effettivi) sì, ma con poco spazio per il contesto.' })));
    outM.innerHTML = 'Llama 3 70B: fp16 ≈ <b>141 GB</b> → 4 bit con gruppi da 64 ≈ <b>37 GB</b>. L\'8B in fp16 ≈ <b>16 GB</b> → 4 bit ≈ <b>4,3 GB</b>: abbastanza piccolo per un portatile.';
    cM.redraw();

    root.append(el('h2', { text: 'Dove si applica e come si fa meglio' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Post-training quantization (PTQ)</b>: si quantizza un modello già addestrato. Il round-to-nearest che vedi qui è la base; metodi come <b>GPTQ</b> o <b>AWQ</b> scelgono gli arrotondamenti guardando l'effetto sull'<i>uscita</i> (con dati di calibrazione), non solo l'errore sul peso.</li>
      <li><b>Quantization-aware training (QAT)</b>: si simula la quantizzazione durante il training perché il modello si adatti.</li>
      <li><b>Attivazioni e KV cache</b> si possono quantizzare, ma le attivazioni hanno outlier più estremi dei pesi (alcuni canali sono molto più grandi degli altri) e sono più delicate.</li>
      <li>Si risparmia memoria <i>e</i> banda: nel decode (modulo 9) il tempo dipende dai byte letti, quindi meno bit significa più token/s, se i kernel dequantizzano in modo efficiente.</li>
      <li>La perdita di qualità cresce al diminuire dei bit; dipende da modello e metodo e va <b>misurata</b> sul compito che interessa. In questo laboratorio non misuriamo la qualità di un modello vero: mostriamo soltanto l'errore sui pesi.</li>` }));

    root.append(details('math', 'La matematica: errore di quantizzazione', 'formule', `
      <p>Con scala <var>s</var>, l'errore di arrotondamento di un peso che non satura è <var>e</var> = <var>ŵ</var> − <var>w</var> ∈ [−<var>s</var>/2, <var>s</var>/2]. Se <var>w</var> è abbastanza "liscio" rispetto a <var>s</var>, l'errore è circa uniforme e</p>
      <span class="eq">E[<var>e</var>²] = ${frac('<var>s</var>²', '12')}, &nbsp; RMS = ${frac('<var>s</var>', '√12')} ≈ 0,29 <var>s</var></span>
      <p>(il test automatico verifica questa relazione sul quantizzatore del laboratorio). Poiché <var>s</var> = max|<var>w</var>|/<var>q</var><sub>max</sub> e <var>q</var><sub>max</sub> ≈ 2<sup>bit−1</sup>, ogni bit in meno <b>raddoppia</b> l'errore: circa 6 dB di rapporto segnale/rumore per bit.</p>
      <p><b>Effetto di un outlier.</b> Se un peso vale <var>m</var>σ, la scala per-tensore diventa <var>s</var> ≈ <var>m</var>σ/<var>q</var><sub>max</sub> invece di ≈ 3σ/<var>q</var><sub>max</sub>: l'errore sui pesi normali cresce di circa <var>m</var>/3 volte. Con gruppi di dimensione <var>g</var> l'effetto resta confinato nel gruppo.</p>
      <p><b>Errore sull'uscita.</b> Per <var>y</var> = Σ<sub>i</sub> <var>w</var><sub>i</sub><var>x</var><sub>i</sub> con errori indipendenti sui pesi, Var(<var>ŷ</var> − <var>y</var>) = Σ<sub>i</sub> Var(<var>e</var><sub>i</sub>) <var>x</var><sub>i</sub>²: gli errori si sommano in potenza, non in ampiezza, ed è per questo che l'uscita è spesso più robusta di quanto l'errore sul singolo peso lasci pensare.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Perché una scala per tutto il tensore soffre con gli outlier?', opts: ['Perché l\'outlier si perde', 'Perché la scala si allarga per includere l\'outlier e tutti gli altri pesi usano pochissimi livelli', 'Perché gli outlier sono sempre negativi'], ans: 1, why: 'La scala è max|w|/qmax: un valore enorme rende i passi di quantizzazione grandi rispetto ai pesi tipici.' },
      { q: 'Quanto cambia l\'errore RMS dei pesi togliendo un bit (a parità di scala per il valore massimo)?', opts: ['Si dimezza', 'Raddoppia', 'Resta uguale'], ans: 1, why: 'Il numero di livelli si dimezza, il passo s raddoppia e l\'errore RMS (≈ 0,29 s) con lui.' },
      { q: 'Perché quantizzare i pesi velocizza il decode?', opts: ['Perché si fanno meno FLOP', 'Perché si leggono meno byte dalla memoria a ogni token e il decode è limitato dalla lettura dei pesi', 'Perché la cache sparisce'], ans: 1, why: 'Modulo 9: nel decode il tempo è dominato dalla banda di memoria, non dal calcolo.' },
    ]));
  },
});
