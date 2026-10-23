'use strict';
/* Module 12 - scaling laws: L(N, D) = E + A/N^a + B/D^b, compute-optimal allocation, overtraining for cheap inference */
const SCALE_FITS = {
  rep: { name: 'replica (Besiroglu et al. 2024)', E: 1.8172, A: 482.01, B: 2085.43, a: 0.3478, b: 0.3658 },
  orig: { name: 'originale (Hoffmann et al. 2022)', E: 1.6934, A: 406.4, B: 410.7, a: 0.3392, b: 0.2849 },
};
const scLoss = (N, D, f) => f.E + f.A / Math.pow(N, f.a) + f.B / Math.pow(D, f.b);
/* minimise L(N, C/6N) over N: setting the derivative to zero gives N^(a+b) = (a A / b B) (C/6)^b */
const scNopt = (C, f) => Math.pow(f.a * f.A / (f.b * f.B), 1 / (f.a + f.b)) * Math.pow(C / 6, f.b / (f.a + f.b));
/* tokens needed for a model of N parameters to reach loss Lt (Infinity if Lt is below what N can ever reach) */
function scDforLoss(N, Lt, f) { const r = Lt - f.E - f.A / Math.pow(N, f.a); return r <= 0 ? Infinity : Math.pow(f.B / r, 1 / f.b); }
const scCost = (N, D, I) => 6 * N * D + 2 * N * I; // training FLOPs + serving I tokens
function scBestN(Lt, I, f) {
  let best = null;
  for (let e = 7; e <= 13; e += 0.02) { const N = Math.pow(10, e), D = scDforLoss(N, Lt, f); if (!isFinite(D)) continue; const c = scCost(N, D, I); if (!best || c < best.c) best = { N, D, c, e }; }
  return best;
}
LAB.tests.scale = { SCALE_FITS, scLoss, scNopt, scDforLoss, scCost, scBestN };

registerModule({
  id: 'm12', n: 12, title: 'Scaling laws', eyebrow: 'quanto grande, quanto a lungo', heading: 'Scaling laws: come si spende un budget di calcolo',
  lede: 'La loss di un modello non è casuale: segue leggi di potenza regolari in parametri, dati e calcolo. Questo permette di decidere prima di spendere milioni, quanto grande fare il modello e per quanti token addestrarlo.',
  build(root) {
    root.append(el('div', { html: `
      <p>Empiricamente la loss sul test di un modello ben addestrato si comporta come</p>
      <span class="eq"><var>L</var>(<var>N</var>, <var>D</var>) = <var>E</var> + ${frac('<var>A</var>', '<var>N</var><sup>α</sup>')} + ${frac('<var>B</var>', '<var>D</var><sup>β</sup>')}</span>
      <p>con <var>N</var> parametri e <var>D</var> token di training. <var>E</var> è la loss "irriducibile" (l'entropia del testo stesso), gli altri due termini dicono quanto manca per avere troppo pochi parametri o troppi pochi dati. Il costo di training è circa <b>6<var>ND</var> FLOP</b>: 2 FLOP per parametro nel passaggio in avanti e 4 nell'indietro, per ogni token.</p>
      <p>Le costanti vengono da un <b>fit</b> su molte esecuzioni di training: non sono leggi di natura, e il laboratorio mostra due fit diversi per la stessa formula, per ricordarlo.</p>` }));

    let fk = 'rep';
    const fit = () => SCALE_FITS[fk];
    const fsel = select(Object.entries(SCALE_FITS).map(([k, v]) => [k, v.name]), fk, v => { fk = v; upd(); }, 'costanti');
    const sC = slider({ label: 'budget di calcolo log₁₀ C (FLOP)', min: 19, max: 26, step: 0.1, value: 23.76, fmt: v => '10^' + v.toFixed(1), onInput: () => upd() });
    const cv = canvasEl(330), cvU = canvasEl(220), out = el('div', { class: 'readout' });
    const XN = [7.5, 12.5], YD = [8.5, 14.5];
    const c1 = setupCanvas(cv, (ctx, w, h, t) => {
      const f = fit(), left = 50, right = 10, top = 10, bot = 38, pw = w - left - right, ph = h - top - bot, nx = 60, ny = 40;
      const X = v => left + (v - XN[0]) / (XN[1] - XN[0]) * pw, Y = v => top + (1 - (v - YD[0]) / (YD[1] - YD[0])) * ph;
      let lo = Infinity, hi = -Infinity; const Lm = [];
      for (let i = 0; i < ny; i++) { Lm.push([]); for (let j = 0; j < nx; j++) { const l = scLoss(Math.pow(10, XN[0] + (j + 0.5) / nx * (XN[1] - XN[0])), Math.pow(10, YD[0] + (1 - (i + 0.5) / ny) * (YD[1] - YD[0])), f); Lm[i].push(l); lo = Math.min(lo, l); hi = Math.max(hi, l); } }
      for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) { const u = 1 - (Lm[i][j] - lo) / (hi - lo); ctx.fillStyle = rgba(t.accent, 0.08 + 0.85 * Math.pow(u, 1.5)); ctx.fillRect(left + j * pw / nx, top + i * ph / ny, pw / nx + 0.5, ph / ny + 0.5); }
      ctx.strokeStyle = t.muted; ctx.strokeRect(left, top, pw, ph);
      for (let e = 8; e <= 12; e++) text(ctx, '10^' + e, X(e), top + ph + 14, { align: 'center', color: t.muted, size: 10 });
      for (let e = 9; e <= 14; e++) text(ctx, '10^' + e, left - 5, Y(e) + 4, { align: 'right', color: t.muted, size: 10 });
      text(ctx, 'parametri N', left + pw / 2, h - 5, { align: 'center', color: t.muted, size: 11 });
      ctx.save(); ctx.translate(11, top + ph / 2); ctx.rotate(-Math.PI / 2); text(ctx, 'token di training D', 0, 0, { align: 'center', color: t.muted, size: 11 }); ctx.restore();
      const lc = sC.get(), C = Math.pow(10, lc);
      ctx.strokeStyle = t.amber; ctx.lineWidth = 2.5; ctx.beginPath(); let on = false;
      for (let e = XN[0]; e <= XN[1]; e += 0.05) { const ld = lc - Math.log10(6) - e; if (ld < YD[0] || ld > YD[1]) { on = false; continue; } if (on) ctx.lineTo(X(e), Y(ld)); else ctx.moveTo(X(e), Y(ld)); on = true; }
      ctx.stroke();
      const N = scNopt(C, f), D = C / 6 / N, px = X(Math.log10(N)), py = Y(Math.log10(D));
      if (px > left && px < left + pw && py > top && py < top + ph) { ctx.fillStyle = t.rose; ctx.beginPath(); ctx.arc(px, py, 6, 0, 7); ctx.fill(); text(ctx, 'ottimo', px + 9, py - 8, { color: t.rose, size: 11, weight: 700 }); }
      [['Chinchilla 70B', 70e9, 1.4e12], ['Gopher 280B', 280e9, 300e9]].forEach(([nm, n, d]) => { const x = X(Math.log10(n)), y = Y(Math.log10(d)); ctx.strokeStyle = t.ink; ctx.lineWidth = 1.5; ctx.strokeRect(x - 4, y - 4, 8, 8); text(ctx, nm, x + 8, y + 4, { color: t.ink, size: 10 }); });
      text(ctx, 'più scuro = loss più bassa · arancione: tutte le combinazioni con lo stesso calcolo C = 6ND', left + 6, top + 14, { color: t.ink, size: 10 });
    });
    const c2 = setupCanvas(cvU, (ctx, w, h, t) => {
      const f = fit(), C = Math.pow(10, sC.get()), es = range(121).map(i => 7.5 + i * 0.05), Ls = es.map(e => scLoss(Math.pow(10, e), C / 6 / Math.pow(10, e), f));
      const best = scNopt(C, f), c = chart(ctx, w, h, t, [{ xs: es, ys: Ls.map(l => Math.min(l, 4)), color: t.accent, lw: 2.5, label: 'loss a calcolo fisso' }], { xlabel: 'log₁₀ N  (a parità di calcolo: più N significa meno token)', ylabel: 'loss', nx: 5, xfmt: v => v.toFixed(1) });
      ctx.strokeStyle = t.rose; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(c.X(Math.log10(best)), c.y1); ctx.lineTo(c.X(Math.log10(best)), c.y0); ctx.stroke(); ctx.setLineDash([]);
    });
    function upd() {
      const f = fit(), C = Math.pow(10, sC.get()), N = scNopt(C, f), D = C / 6 / N;
      out.innerHTML = `ottimo a calcolo fisso: <b>${human(N)}</b> parametri e <b>${human(D)}</b> token (<b>${f2(D / N, 0)}</b> token per parametro) · loss prevista <b>${f2(scLoss(N, D, f), 3)}</b> · Chinchilla 70B/1,4T: ${f2(scLoss(70e9, 1.4e12, f), 3)}, Gopher 280B/300B: ${f2(scLoss(280e9, 300e9, f), 3)}`;
      c1.redraw(); c2.redraw();
    }
    root.append(lab('Laboratorio · il paesaggio della loss e l\'ottimo a calcolo fisso',
      controls(fsel, sC.el), cv, out, cvU,
      el('p', { class: 'muted', html: 'Il quadrato vuoto sotto la linea arancione non è raggiungibile col budget scelto. Lungo la linea c\'è un compromesso a U: modelli troppo piccoli hanno pochi parametri, modelli troppo grandi vedono pochi token. Con le costanti della <b>replica</b> l\'ottimo è vicino a 20 token per parametro (la regola quotata di Chinchilla). Con le costanti <b>originali</b> lo stesso calcolo dà circa 60: i fit risentono dei dati e del metodo, e una replica ha mostrato problemi nel fit originale. Il messaggio affidabile è la forma, non la cifra esatta.' })));
    upd();

    /* ---- overtraining ---- */
    const sT = slider({ label: 'loss obiettivo', min: 1.9, max: 2.6, step: 0.01, value: 2.1, fmt: v => v.toFixed(2), onInput: () => upd2() });
    const sI = slider({ label: 'token serviti in vita (log₁₀ I)', min: 9, max: 15, step: 0.25, value: 13, fmt: v => '10^' + v.toFixed(2), onInput: () => upd2() });
    const cv2 = canvasEl(240), out2 = el('div', { class: 'readout' });
    const c3 = setupCanvas(cv2, (ctx, w, h, t) => {
      const f = fit(), Lt = sT.get(), I = Math.pow(10, sI.get()), es = range(121).map(i => 8 + i * 0.04);
      const tot = es.map(e => { const N = Math.pow(10, e), D = scDforLoss(N, Lt, f); return isFinite(D) ? Math.log10(scCost(N, D, I)) : NaN; });
      const trn = es.map(e => { const N = Math.pow(10, e), D = scDforLoss(N, Lt, f); return isFinite(D) ? Math.log10(scCost(N, D, 0)) : NaN; });
      const c = chart(ctx, w, h, t, [{ xs: es, ys: trn, color: t.muted, lw: 2, dash: [5, 4], label: 'solo training' }, { xs: es, ys: tot, color: t.accent, lw: 2.5, label: 'training + servizio' }], { xlabel: 'log₁₀ N (parametri del modello)', ylabel: 'log₁₀ FLOP', nx: 5, xfmt: v => v.toFixed(1) });
      const b0 = scBestN(Lt, 0, f), b1 = scBestN(Lt, I, f);
      [[b0, t.muted], [b1, t.rose]].forEach(([b, col]) => { if (!b) return; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(c.X(b.e), c.Y(Math.log10(scCost(b.N, b.D, col === t.muted ? 0 : I))), 5, 0, 7); ctx.fill(); });
    });
    function upd2() {
      const f = fit(), Lt = sT.get(), I = Math.pow(10, sI.get()), b0 = scBestN(Lt, 0, f), b1 = scBestN(Lt, I, f);
      if (!b0 || !b1) { out2.textContent = 'Loss obiettivo non raggiungibile con queste costanti.'; c3.redraw(); return; }
      const costIfTrainOpt = scCost(b0.N, b0.D, I);
      out2.innerHTML = `ottimo solo-training: <b>${human(b0.N)}</b> param × <b>${human(b0.D)}</b> token (${f2(b0.D / b0.N, 0)} tok/param) · con ${human(I)} token serviti conviene <b>${human(b1.N)}</b> param × <b>${human(b1.D)}</b> token (<b>${f2(b1.D / b1.N, 0)}</b> tok/param) · costo totale <b>${f2(b1.c / costIfTrainOpt * 100, 0)}%</b> di quello del modello "ottimo per il training"`;
      c3.redraw();
    }
    root.append(lab('Laboratorio · se il modello verrà usato molto, conviene "sovra-addestrarlo"',
      el('p', { html: 'Chinchilla minimizza il costo del <i>training</i>. Ma se il modello servirà trilioni di token, ogni parametro costa 2 FLOP per ogni token servito. Fissiamo la loss da raggiungere e cerchiamo la dimensione che minimizza <b>training + inferenza</b>, cioè 6<var>ND</var> + 2<var>N</var>·<var>I</var>.' }),
      controls(sT.el, sI.el), cv2, out2,
      el('p', { class: 'muted', html: 'Con <var>I</var> piccolo il minimo coincide con l\'ottimo di Chinchilla; man mano che <var>I</var> cresce il minimo si sposta su modelli più piccoli addestrati per molti più token. È uno dei motivi per cui i modelli open recenti vedono centinaia o migliaia di token per parametro. La stessa spiegazione, in forma di calcolo, è nell\'articolo "Beyond Chinchilla-Optimal" (Sardana et al.).' })));
    upd2();

    root.append(el('h2', { text: 'Cosa ricordare' }));
    root.append(el('ul', { class: 'prose', html: `
      <li>La loss scende come potenza di <var>N</var> e <var>D</var>, con un limite <var>E</var> che nessuna scala supera. Ogni raddoppio dà un guadagno prevedibile ma decrescente.</li>
      <li>A calcolo fisso conviene far crescere parametri e dati <b>insieme</b>; la proporzione dipende dal fit.</li>
      <li>Le scaling laws dicono la <i>loss media</i> sul testo, non quando compariranno singole capacità: alcune abilità sembrano comparire all'improvviso, ma in parte dipende da come le si misura (metriche tutto-o-niente).</li>
      <li>I dati sono un limite quanto il calcolo: i testi di qualità a disposizione non sono infiniti. Si ricorre a più epoche, dati sintetici, filtraggio.</li>
      <li>Le leggi valgono entro il range misurato; estrapolare di molti ordini di grandezza è un rischio.</li>` }));

    root.append(details('math', 'La matematica: l\'ottimo in forma chiusa', 'formule', `
      <p>Con <var>D</var> = <var>C</var>/(6<var>N</var>) la loss lungo la linea a calcolo costante è</p>
      <span class="eq"><var>L</var>(<var>N</var>) = <var>E</var> + <var>A</var> <var>N</var><sup>−α</sup> + <var>B</var> (<var>C</var>/6)<sup>−β</sup> <var>N</var><sup>β</sup></span>
      <p>Derivando e ponendo a zero: −α<var>A</var> <var>N</var><sup>−α−1</sup> + β<var>B</var> (<var>C</var>/6)<sup>−β</sup> <var>N</var><sup>β−1</sup> = 0, da cui</p>
      <span class="eq"><var>N</var><sub>opt</sub> = ( ${frac('α<var>A</var>', 'β<var>B</var>')} )<sup>1/(α+β)</sup> (<var>C</var>/6)<sup>β/(α+β)</sup>, &nbsp; <var>D</var><sub>opt</sub> = ${frac('<var>C</var>', '6 <var>N</var><sub>opt</sub>')}</span>
      <p>Quindi <var>N</var><sub>opt</sub> ∝ <var>C</var><sup>β/(α+β)</sup> e <var>D</var><sub>opt</sub> ∝ <var>C</var><sup>α/(α+β)</sup>. Con α ≈ β i due crescono alla stessa velocità (≈ √<var>C</var>).</p>
      <p><b>Dati necessari per una loss obiettivo.</b> Risolvendo <var>L</var>(<var>N</var>, <var>D</var>) = <var>L</var><sub>t</sub> per <var>D</var>: <var>D</var> = ( <var>B</var> / (<var>L</var><sub>t</sub> − <var>E</var> − <var>A</var><var>N</var><sup>−α</sup>) )<sup>1/β</sup>. Esiste solo se <var>L</var><sub>t</sub> &gt; <var>E</var> + <var>A</var><var>N</var><sup>−α</sup>: un modello piccolo ha un pavimento che nessun dato può abbassare.</p>
      <p><b>Con inferenza.</b> Si minimizza 6<var>N</var><var>D</var>(<var>N</var>) + 2<var>N</var><var>I</var> lungo la curva <var>D</var>(<var>N</var>) a loss costante; il laboratorio lo fa numericamente.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Perché il costo di training è circa 6ND FLOP?', opts: ['6 sono i layer tipici', 'Circa 2 FLOP per parametro nel forward e circa 4 nel backward, per ogni token', 'Perché si fanno 6 epoche'], ans: 1, why: 'Il passaggio in avanti ha una moltiplicazione-somma (2 FLOP) per parametro; l\'indietro calcola due prodotti (gradiente rispetto ai pesi e agli input): circa il doppio.' },
      { q: 'A parità di calcolo, un modello enorme addestrato su pochissimi token è…', opts: ['sempre il migliore', 'quasi sempre subottimo: il termine dei dati B/D^β domina', 'equivalente a uno piccolo'], ans: 1, why: 'Il compromesso a U: oltre l\'ottimo, ogni parametro in più toglie token e il guadagno sui parametri non compensa la perdita sui dati.' },
      { q: 'Perché un modello destinato a molto traffico si addestra su più token dell\'ottimo di Chinchilla?', opts: ['Per far salire la loss', 'Perché il costo di inferenza cresce con N, quindi conviene un modello più piccolo addestrato più a lungo', 'Perché la KV cache lo richiede'], ans: 1, why: 'Minimizzare training + inferenza sposta l\'ottimo verso N più piccoli e D più grandi.' },
    ]));
  },
});
