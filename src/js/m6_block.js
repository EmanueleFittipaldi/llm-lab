'use strict';
/* Module 6 - the rest of the transformer block: MLP, residual stream, normalization */
const erf = x => { const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)); };
const ACT = {
  relu: { name: 'ReLU', f: z => z > 0 ? z : 0, d: z => z > 0 ? 1 : 0 },
  gelu: { name: 'GELU', f: z => z * 0.5 * (1 + erf(z / Math.SQRT2)), d: z => 0.5 * (1 + erf(z / Math.SQRT2)) + z * Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI) },
  silu: { name: 'SiLU', f: z => z / (1 + Math.exp(-z)), d: z => { const s = 1 / (1 + Math.exp(-z)); return s * (1 + z * (1 - s)); } },
};
/* one-hidden-layer network fitted to a 1-D function with Adam (a miniature of the transformer MLP) */
class Fit1D {
  constructor(H, act, target, seed = 1) {
    const r = rng(seed); this.H = H; this.act = ACT[act]; this.t = 0;
    this.xs = range(65).map(i => -2 + 4 * i / 64); this.ts = this.xs.map(target);
    const w = range(H).map(i => (i % 2 ? 1 : -1) * (0.8 + 0.6 * r())), kink = i => -1.8 + 3.6 * (i + 0.5) / H;
    this.th = Float64Array.from([...w, ...w.map((wi, i) => -wi * kink(i)), ...range(H).map(() => r.gauss() * 0.3), 0]);
    this.m = new Float64Array(this.th.length); this.v = new Float64Array(this.th.length);
  }
  pred(x) { const { H, th } = this; let y = th[3 * H]; for (let i = 0; i < H; i++) y += th[2 * H + i] * this.act.f(th[i] * x + th[H + i]); return y; }
  parts(x) { const { H, th } = this; return range(H).map(i => th[2 * H + i] * this.act.f(th[i] * x + th[H + i])); }
  loss() { return sum(this.xs.map((x, k) => (this.pred(x) - this.ts[k]) ** 2)) / this.xs.length; }
  grad() {
    const { H, th } = this, g = new Float64Array(th.length), N = this.xs.length;
    this.xs.forEach((x, k) => {
      const dy = 2 * (this.pred(x) - this.ts[k]) / N; g[3 * H] += dy;
      for (let i = 0; i < H; i++) { const z = th[i] * x + th[H + i], da = this.act.d(z); g[2 * H + i] += dy * this.act.f(z); g[i] += dy * th[2 * H + i] * da * x; g[H + i] += dy * th[2 * H + i] * da; }
    });
    return g;
  }
  step(lr = 0.03) {
    const g = this.grad(), b1 = 0.9, b2 = 0.999; this.t++;
    for (let i = 0; i < g.length; i++) {
      this.m[i] = b1 * this.m[i] + (1 - b1) * g[i]; this.v[i] = b2 * this.v[i] + (1 - b2) * g[i] * g[i];
      this.th[i] -= lr * (this.m[i] / (1 - b1 ** this.t)) / (Math.sqrt(this.v[i] / (1 - b2 ** this.t)) + 1e-8);
    }
  }
}
/* mean log10 of the gradient norm through L layers, with and without a skip connection */
function gradFlow(L, s, trials = 12, d = 16, seed = 1) {
  const plain = new Array(L + 1).fill(0), res = new Array(L + 1).fill(0);
  for (let tr = 0; tr < trials; tr++) {
    const r = rng(seed * 1000 + tr); let g = range(d).map(() => r.gauss()); const n0 = norm(g); g = g.map(v => v / n0);
    let gp = g.slice(), gr = g.slice();
    for (let l = 1; l <= L; l++) {
      const W = range(d).map(() => range(d).map(() => r.gauss() * s / Math.sqrt(d)));
      gp = W.map(row => dot(row, gp)); gr = gr.map((v, i) => v + dot(W[i], gr));
      plain[l] += Math.log10(norm(gp) + 1e-300) / trials; res[l] += Math.log10(norm(gr)) / trials;
    }
  }
  return { plain, res };
}
function normLayer(x, gamma, beta, kind = 'layer', eps = 1e-5) {
  const n = x.length;
  if (kind === 'rms') { const r = Math.sqrt(sum(x.map(v => v * v)) / n + eps); return x.map(v => v / r * gamma); }
  const mu = sum(x) / n, va = sum(x.map(v => (v - mu) ** 2)) / n; return x.map(v => (v - mu) / Math.sqrt(va + eps) * gamma + beta);
}
Object.assign(LAB.tests, { Fit1D, ACT, gradFlow, normLayer });

registerModule({
  id: 'm6', n: 6, layer: true, title: 'MLP, residual, norm', eyebrow: 'il resto del blocco', heading: 'Cosa c\'è nel blocco oltre all\'attention',
  lede: 'L\'attention sposta informazione tra token. L\'MLP la elabora, il residual stream la conserva, la normalizzazione tiene i numeri in un intervallo sano. Insieme rendono possibile impilare decine di layer.',
  build(root) {
    root.append(el('div', { html: `
      <p>Un blocco transformer (nella variante <i>pre-norm</i>, la più diffusa oggi) fa due cose in sequenza, ciascuna avvolta nello stesso schema "normalizza, trasforma, <b>somma all'ingresso</b>":</p>` }));
    const parts = [['x', 'ingresso [T, d]'], ['LayerNorm', 'normalizza ogni token'], ['Attention', 'mescola tra token'], ['⊕', 'somma al residual'], ['LayerNorm', 'normalizza'], ['MLP', 'elabora ogni token'], ['⊕', 'somma al residual'], ['x′', 'uscita [T, d]']];
    root.append(lab('Schema · un blocco', el('div', { class: 'flow' }, parts.map(([a, b]) => el('div', { class: 'stg', style: 'cursor:default' }, el('b', { text: a }), el('span', { text: b })))),
      math('<var>x</var>′ = <var>x</var> + <span class="fn">Attn</span>(<span class="fn">LN</span>(<var>x</var>)), &nbsp;&nbsp; <var>x</var>″ = <var>x</var>′ + <span class="fn">MLP</span>(<span class="fn">LN</span>(<var>x</var>′))')));
    root.append(el('p', { html: 'Il flusso <var>x</var> che attraversa tutto il blocco (e tutti i blocchi) si chiama <b>residual stream</b>: una specie di bus condiviso su cui ogni attention e ogni MLP leggono e scrivono <i>correzioni additive</i>. Il token entra come embedding e esce come vettore da cui si leggono i logit.' }));

    /* ---- MLP fit lab ---- */
    root.append(el('h2', { text: 'L\'MLP: elaborare token per token' }));
    root.append(el('p', { html: 'L\'MLP agisce su ogni posizione <b>separatamente</b>: espande il vettore da <var>d</var> a <var>d</var><sub>ff</sub> (di solito 4<var>d</var>), applica una non-linearità e riproietta a <var>d</var>. Senza non-linearità, due matrici in fila sarebbero equivalenti a una sola. Per vedere che cosa significa, qui sotto una versione ridotta (1 ingresso, 1 uscita) impara a riprodurre una curva: <b>ogni neurone ReLU è una "piega"</b> e la rete somma pieghe.' }));
    const targets = { 'sin(2x)': x => Math.sin(2 * x), '|x|': x => Math.abs(x), 'gradino': x => x > 0 ? 1 : 0, 'x³ − x': x => x * x * x - x };
    let tgt = 'sin(2x)', act = 'relu', fit = null, steps = 0;
    const cv = canvasEl(280), info = el('div', { class: 'readout' });
    const sH = slider({ label: 'neuroni', min: 1, max: 16, value: 5, onInput: () => reset() }), sLR = slider({ label: 'learning rate', min: 0.005, max: 0.1, step: 0.005, value: 0.03, fmt: v => v.toFixed(3) });
    let showParts = true;
    const c = setupCanvas(cv, (ctx, w, h, t) => {
      if (!fit) return;
      const S = [{ xs: fit.xs, ys: fit.ts, color: t.muted, dash: [6, 4], lw: 2, label: 'funzione da imparare' }];
      if (showParts) { const P = fit.xs.map(x => fit.parts(x)); for (let i = 0; i < fit.H; i++) S.push({ xs: fit.xs, ys: P.map(p => p[i]), color: rgba(t.teal, 0.45), lw: 1 }); }
      S.push({ xs: fit.xs, ys: fit.xs.map(x => fit.pred(x)), color: t.accent, lw: 3, label: 'rete' });
      const all = fit.ts.concat(fit.xs.map(x => fit.pred(x))), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.15 + 0.2;
      chart(ctx, w, h, t, S, { xmin: -2, xmax: 2, ymin: Math.max(Math.min(lo - pad, -0.5), -8), ymax: Math.min(Math.max(hi + pad, 1.2), 8), xlabel: 'ingresso x', ylabel: 'uscita', nx: 4 });
    });
    function reset() { fit = new Fit1D(sH.get(), act, targets[tgt], 3); steps = 0; lp.stop(); playBtn.textContent = 'Allena'; refresh(); }
    function refresh() { info.innerHTML = `passi <b>${steps}</b> · loss (MSE) <b>${fit.loss().toFixed(4)}</b> · parametri <b>${3 * fit.H + 1}</b> (pesi in, bias, pesi out, bias finale)`; c.redraw(); }
    const lp = loop(cv, () => { for (let k = 0; k < 8; k++) { fit.step(sLR.get()); steps++; } refresh(); if (steps >= 4000) { lp.stop(); playBtn.textContent = 'Allena'; } });
    const playBtn = button('Allena', () => { if (lp.running) { lp.stop(); playBtn.textContent = 'Allena'; } else { lp.start(); playBtn.textContent = 'Pausa'; } }, 'primary');
    root.append(lab('Laboratorio · una MLP che impara una curva',
      controls(select(Object.keys(targets).map(k => [k, k]), tgt, v => { tgt = v; reset(); }, 'funzione'), select(Object.keys(ACT).map(k => [k, ACT[k].name]), act, v => { act = v; reset(); }, 'attivazione'), sH.el, sLR.el),
      controls(playBtn, button('Reset', reset), checkbox('mostra il contributo di ogni neurone', true, v => { showParts = v; c.redraw(); })), cv, info,
      el('p', { html: 'Con 1-2 neuroni la rete può solo tracciare una spezzata grossolana; con 8-16 segue bene una curva liscia. Prova il <b>gradino</b>: una funzione discontinua si approssima con pendenze molto ripide. Confronta ReLU con GELU/SiLU, la variante "smussata" usata nei LLM.' })));
    reset();
    root.append(el('p', { html: `Nel modello vero i neuroni sono migliaia per layer e lavorano su vettori di <var>d</var> numeri, ma l'idea è la stessa. Le ricerche di interpretabilità suggeriscono di leggere l'MLP come una <b>memoria chiave-valore</b>: ogni neurone si accende quando riconosce un certo pattern nell'ingresso (la "chiave") e scrive nel residual stream una certa direzione (il "valore"). Dato che l'MLP pesa circa 2/3 dei parametri di un blocco, è plausibile che molte conoscenze fattuali risiedano lì, mentre l'attention instrada l'informazione.` }));

    /* ---- residual lab ---- */
    root.append(el('h2', { text: 'Il residual: perché si possono impilare tanti layer' }));
    root.append(el('p', { html: `Senza la somma <var>x</var> + …, il segnale (e il gradiente nel training) attraversa ogni layer moltiplicandosi per la sua matrice. Se ogni matrice "amplifica" di un fattore <var>s</var> &lt; 1, dopo <var>L</var> layer resta <var>s</var><sup><var>L</var></sup>: il gradiente svanisce. Con il residual, la derivata di ogni layer è <b>1 + qualcosa</b>, quindi esiste sempre un percorso diretto con guadagno 1. Qui sotto un esperimento numerico con matrici casuali (16 dimensioni).` }));
    const sL = slider({ label: 'layer', min: 2, max: 80, value: 40, onInput: () => rf() }), sS = slider({ label: 's (guadagno per layer)', min: 0.05, max: 1.2, step: 0.05, value: 0.5, fmt: v => v.toFixed(2), onInput: () => rf() });
    let seed = 1; const rcv = canvasEl(240), rOut = el('div', { class: 'readout' });
    let GF = null;
    const rc = setupCanvas(rcv, (ctx, w, h, t) => {
      if (!GF) return; const xs = range(GF.plain.length);
      chart(ctx, w, h, t, [{ xs, ys: GF.plain, color: t.amber, lw: 3, label: 'senza residual' }, { xs, ys: GF.res, color: t.accent, lw: 3, label: 'con residual' }, { xs, ys: xs.map(() => 0), color: t.muted, dash: [4, 4], lw: 1 }],
        { xlabel: 'profondità (layer attraversati)', ylabel: 'log₁₀ ‖gradiente‖', xmin: 0, xmax: sL.get(), yfmt: v => (v > 0 ? '+' : '') + Math.round(v), nx: 4 });
    });
    function rf() {
      const L = sL.get(), s = sS.get(); GF = gradFlow(L, s, 12, 16, seed);
      rOut.innerHTML = `Dopo ${L} layer: senza residual <b>10<sup>${f2(GF.plain[L], 1)}</sup></b> · con residual <b>10<sup>${f2(GF.res[L], 1)}</sup></b>` +
        (GF.res[L] > 6 ? '<br><span style="color:var(--rose)">Con guadagno grande anche il residual esplode: per questo servono normalizzazione e inizializzazioni scalate con la profondità.</span>' : '');
      rc.redraw();
    }
    root.append(lab('Laboratorio · flusso del gradiente in profondità', controls(sL.el, sS.el, button('Nuovo campione casuale', () => { seed++; rf(); })), rcv, rOut)); rf();

    /* ---- layernorm lab ---- */
    root.append(el('h2', { text: 'La normalizzazione' }));
    root.append(el('p', { html: 'Ogni layer somma contributi al residual stream, quindi la scala dei numeri tende a crescere e a variare molto da token a token. <b>LayerNorm</b> prende il vettore di <b>un solo token</b>, lo ricentra (media 0) e lo riscala (deviazione standard 1), poi applica un guadagno <var>γ</var> e un offset <var>β</var> appresi. <b>RMSNorm</b> (Llama e altri) salta la sottrazione della media. Muovi i valori e osserva.' }));
    let xv = [2.2, -0.5, 1.4, 3.1, -1.2, 0.4, 2.7, 0.9], kind = 'layer';
    const sl = xv.map((v, i) => slider({ label: 'x' + i, min: -6, max: 6, step: 0.1, value: v, fmt: q => q.toFixed(1), onInput: q => { xv[i] = q; ln(); } }));
    const sg = slider({ label: 'γ', min: 0.2, max: 3, step: 0.1, value: 1, fmt: v => v.toFixed(1), onInput: () => ln() }), sbeta = slider({ label: 'β', min: -2, max: 2, step: 0.1, value: 0, fmt: v => v.toFixed(1), onInput: () => ln() });
    const lcv = canvasEl(230), lOut = el('div', { class: 'readout' });
    let LN = null;
    const lc = setupCanvas(lcv, (ctx, w, h, t) => {
      if (!LN) return; const n = xv.length, half = (w - 20) / 2, m = Math.max(6, ...xv.map(Math.abs), ...LN.y.map(Math.abs)), y0 = h / 2 + 4, sc = (h / 2 - 28) / m;
      [[xv, 'ingresso x', t.amber, 6], [LN.y, kind === 'layer' ? 'uscita LayerNorm' : 'uscita RMSNorm', t.accent, 6]].forEach(([arr, name, col], k) => {
        const x0 = 10 + k * (half + 10), bw = (half - 20) / n;
        ctx.strokeStyle = t.muted; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + half - 10, y0); ctx.stroke();
        arr.forEach((v, i) => { ctx.fillStyle = col; ctx.fillRect(x0 + i * bw + 3, y0, bw - 6, -v * sc); text(ctx, String(i), x0 + i * bw + bw / 2, h - 6, { align: 'center', color: t.muted, size: 10 }); });
        const mu = sum(arr) / n; ctx.strokeStyle = t.rose; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x0, y0 - mu * sc); ctx.lineTo(x0 + half - 10, y0 - mu * sc); ctx.stroke(); ctx.setLineDash([]);
        text(ctx, name, x0, 16, { color: col, size: 12, sans: true, weight: 700 });
      });
      text(ctx, 'linea rosa = media', w - 8, 16, { color: t.rose, size: 10, align: 'right' });
    });
    function ln() {
      const g = sg.get(), b = sbeta.get(), y = normLayer(xv, g, b, kind), st = a => { const mu = sum(a) / a.length; return [mu, Math.sqrt(sum(a.map(v => (v - mu) ** 2)) / a.length)]; };
      LN = { y }; const [m1, s1] = st(xv), [m2, s2] = st(y);
      lOut.innerHTML = `ingresso: media <b>${f2(m1)}</b>, dev. std <b>${f2(s1)}</b> · uscita: media <b>${f2(m2)}</b>, dev. std <b>${f2(s2)}</b>` + (kind === 'layer' ? ` (attese β = ${f2(b)} e γ = ${f2(g)})` : ` (RMSNorm non toglie la media; la scala RMS vale γ = ${f2(g)})`);
      lc.redraw();
    }
    const scaleBtn = button('Moltiplica x per 3', () => { xv = xv.map(v => clamp(v * 3, -6, 6)); sl.forEach((s, i) => s.set(xv[i])); ln(); });
    root.append(lab('Laboratorio · LayerNorm e RMSNorm', controls(select([['layer', 'LayerNorm'], ['rms', 'RMSNorm']], kind, v => { kind = v; ln(); }, 'tipo'), sg.el, sbeta.el, scaleBtn, button('Reset', () => { xv = [2.2, -0.5, 1.4, 3.1, -1.2, 0.4, 2.7, 0.9]; sl.forEach((s, i) => s.set(xv[i])); ln(); })),
      el('div', { class: 'controls' }, sl.map(s => s.el)), lcv, lOut,
      el('p', { html: 'Prova "Moltiplica x per 3" (i valori sono limitati a ±6 per la grafica): l\'uscita non cambia. La normalizzazione rende il blocco <b>insensibile alla scala</b> del residual stream, che può così crescere con la profondità senza far saltare i layer successivi.' })));
    ln();

    root.append(details('math', 'La matematica: MLP, normalizzazione e residual', 'formule', `
      <p><b>MLP classica.</b> <span class="m"><span class="fn">MLP</span>(<var>x</var>) = <var>W</var><sub>2</sub> <var>σ</var>(<var>W</var><sub>1</sub><var>x</var> + <var>b</var><sub>1</sub>) + <var>b</var><sub>2</sub></span>, con <var>W</var><sub>1</sub> ∈ ℝ<sup><var>d</var><sub>ff</sub>×<var>d</var></sup>, <var>W</var><sub>2</sub> ∈ ℝ<sup><var>d</var>×<var>d</var><sub>ff</sub></sup>. Equivale a Σ<sub>i</sub> <var>σ</var>(<var>k</var><sub>i</sub>·<var>x</var> + <var>b</var><sub>i</sub>) <var>v</var><sub>i</sub>, dove <var>k</var><sub>i</sub> è la riga <var>i</var> di <var>W</var><sub>1</sub> e <var>v</var><sub>i</sub> la colonna <var>i</var> di <var>W</var><sub>2</sub>.</p>
      <p><b>Attivazioni.</b> ReLU(<var>z</var>) = max(0, <var>z</var>). GELU(<var>z</var>) = <var>z</var>·Φ(<var>z</var>), con Φ la funzione di ripartizione della normale. SiLU(<var>z</var>) = <var>z</var>·σ(<var>z</var>) con σ la sigmoide. <b>SwiGLU</b> (Llama, ecc.): <span class="m"><span class="fn">FFN</span>(<var>x</var>) = (<span class="fn">SiLU</span>(<var>xW</var><sub>g</sub>) ⊙ <var>xW</var><sub>u</sub>) <var>W</var><sub>d</sub></span>: una porta moltiplicativa che richiede tre matrici invece di due (per questo <var>d</var><sub>ff</sub> è di solito ~8<var>d</var>/3 invece di 4<var>d</var>, a parità di parametri).</p>
      <p><b>LayerNorm.</b> Per un vettore <var>x</var> ∈ ℝ<sup><var>d</var></sup>:</p>
      <span class="eq"><span class="fn">LN</span>(<var>x</var>) = <var>γ</var> ⊙ ${frac('<var>x</var> − <var>μ</var>', '√(<var>σ</var>² + ε)')} + <var>β</var>, &nbsp; <var>μ</var> = ${frac('1', '<var>d</var>')}Σ<sub>i</sub><var>x</var><sub>i</sub>, &nbsp; <var>σ</var>² = ${frac('1', '<var>d</var>')}Σ<sub>i</sub>(<var>x</var><sub>i</sub> − <var>μ</var>)²</span>
      <p><b>RMSNorm:</b> <span class="m"><var>x</var><sub>i</sub> / √(${frac('1', '<var>d</var>')}Σ<var>x</var><sub>j</sub>² + ε) · <var>γ</var><sub>i</sub></span>. Un po' più economica e, empiricamente, buona quanto LayerNorm.</p>
      <p><b>Pre-norm vs post-norm.</b> Il transformer originale normalizzava <i>dopo</i> la somma (<span class="m"><span class="fn">LN</span>(<var>x</var> + <span class="fn">F</span>(<var>x</var>))</span>). Oggi si normalizza <i>prima</i> del sotto-blocco, lasciando il residual stream "pulito": il gradiente ha un percorso identità fino all'ingresso e il training è molto più stabile a grandi profondità.</p>
      <p><b>Perché il residual aiuta.</b> Se <var>x</var><sub>l+1</sub> = <var>x</var><sub>l</sub> + <var>F</var><sub>l</sub>(<var>x</var><sub>l</sub>), allora ∂<var>x</var><sub>l+1</sub>/∂<var>x</var><sub>l</sub> = <var>I</var> + ∂<var>F</var><sub>l</sub>/∂<var>x</var><sub>l</sub>. Il prodotto sui layer, ∏(<var>I</var> + <var>J</var><sub>l</sub>), contiene sempre il termine identità: il gradiente verso i layer iniziali non può annullarsi del tutto.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'L\'MLP del transformer agisce…', opts: ['su tutti i token insieme, mescolandoli', 'su ogni posizione indipendentemente, con gli stessi pesi', 'solo sull\'ultimo token'], ans: 1, why: 'L\'MLP è "position-wise": nessuno scambio tra token (quello lo fa l\'attention), stessi pesi per tutte le posizioni.' },
      { q: 'Perché una rete con soli layer lineari in fila sarebbe inutile?', opts: ['Perché è troppo lenta', 'Perché il prodotto di matrici è una sola matrice: manca la non-linearità', 'Perché non ha bias'], ans: 1, why: 'W2(W1x) = (W2W1)x: senza una funzione non lineare in mezzo la profondità non aggiunge capacità.' },
      { q: 'LayerNorm normalizza…', opts: ['ogni token usando le statistiche dei suoi d valori', 'ogni canale usando le statistiche di tutti i token del batch', 'i pesi dopo ogni aggiornamento'], ans: 0, why: 'Calcola media e varianza lungo la dimensione d del singolo token: non dipende dal batch né dagli altri token.' },
      { q: 'Perché il residual aiuta il training di reti profonde?', opts: ['Riduce il numero di parametri', 'Garantisce un percorso con guadagno 1 per il gradiente', 'Elimina la necessità dell\'MLP'], ans: 1, why: 'La derivata di ogni layer contiene la matrice identità: il gradiente può fluire fino ai layer iniziali.' },
    ]));
  },
});
