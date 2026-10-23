'use strict';
/* Module 3 - positions: permutation blindness, sinusoidal encoding, RoPE */
const sinPE = (pos, d) => { const o = new Array(d); for (let i = 0; i < d / 2; i++) { const w = Math.pow(10000, -2 * i / d); o[2 * i] = Math.sin(pos * w); o[2 * i + 1] = Math.cos(pos * w); } return o; };
LAB.tests.sinPE = sinPE;

registerModule({
  id: 'm3', n: 3, title: 'Posizioni', eyebrow: 'dove sta ogni token', heading: 'L\'attention non sa contare',
  lede: 'Senza un\'informazione esplicita sull\'ordine, "il gatto mangia il pesce" e "il pesce mangia il gatto" sono per il modello lo stesso insieme di parole. Le codifiche posizionali risolvono il problema.',
  build(root) {
    root.append(el('div', { html: `
      <p>L'attention (modulo 4) confronta ogni token con tutti gli altri tramite prodotti scalari: il risultato per un token dipende da <i>quali</i> altri token ci sono, non da <i>dove</i> sono. Si dice che l'operazione è <b>equivariante alle permutazioni</b>.
      Provalo qui sotto: ho preso un piccolo blocco di self-attention con pesi casuali, ho calcolato l'uscita per la parola "gatto", poi ho rimescolato le parole e ripetuto il calcolo.</p>` }));

    /* ---- Lab A ---- */
    const d = 6, r = rng(3), TOK = ['il', 'gatto', 'mangia', 'pesce'];
    const emb = TOK.map(() => range(d).map(() => r.gauss() * 0.8));
    const mk = () => range(d).map(() => range(d).map(() => r.gauss() * 0.55));
    const Wq = mk(), Wk = mk(), Wv = mk();
    let usePE = false, perm = [0, 1, 2, 3], outs = null;
    const out = el('div', { class: 'readout' }), order = el('div', { class: 'chips' }), cv = canvasEl(230);
    function run() {
      const go = p => {
        const X = p.map((tk, pos) => emb[tk].map((v, j) => v + (usePE ? sinPE(pos, d)[j] : 0)));
        return attention(X, Wq, Wk, Wv, false).Z[p.indexOf(1)];
      };
      const a = go([0, 1, 2, 3]), b = go(perm); outs = { a, b };
      const diff = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
      order.replaceChildren(el('span', { class: 'muted', text: 'ordine attuale:' }), ...perm.map(i => el('span', { class: 'chip', text: TOK[i] })));
      out.innerHTML = `Differenza massima tra le due uscite di "gatto": <b>${diff.toExponential(2)}</b> ` + (diff < 1e-9 ? '→ <b>identiche</b>: il modello non ha modo di sapere che le parole sono state spostate.' : '→ <b>diverse</b>: ora l\'ordine influenza il risultato.');
      c.redraw();
    }
    const c = setupCanvas(cv, (ctx, w, h, t) => {
      if (!outs) return;
      const m = Math.max(1, ...outs.a.map(Math.abs), ...outs.b.map(Math.abs)), x0 = 40, bw = (w - x0 - 10) / d, y0 = h / 2 + 8, sc = (h / 2 - 34) / m;
      ctx.strokeStyle = t.muted; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(w - 6, y0); ctx.stroke();
      for (let j = 0; j < d; j++) {
        const x = x0 + j * bw + 6;
        ctx.fillStyle = t.accent; ctx.fillRect(x, y0, bw * 0.38, -outs.a[j] * sc);
        ctx.fillStyle = t.amber; ctx.fillRect(x + bw * 0.4, y0, bw * 0.38, -outs.b[j] * sc);
        text(ctx, 'd' + j, x + bw * 0.4, h - 6, { color: t.muted, align: 'center', size: 11 });
      }
      text(ctx, 'ordine originale', 44, 16, { color: t.accent, size: 12, sans: true, weight: 700 });
      text(ctx, 'ordine rimescolato', 170, 16, { color: t.amber, size: 12, sans: true, weight: 700 });
      text(ctx, 'uscita', 4, y0 - 6, { color: t.muted, size: 10 });
    });
    const shuffle = button('Mescola le parole', () => { const rr = Math.random; let p; do { p = [0, 1, 2, 3].sort(() => rr() - 0.5); } while (p.every((v, i) => v === i)); perm = p; run(); }, 'primary');
    const chk = checkbox('aggiungi codifica posizionale sinusoidale', false, v => { usePE = v; run(); });
    root.append(lab('Laboratorio · ordine invisibile', controls(shuffle, button('Ripristina', () => { perm = [0, 1, 2, 3]; run(); }), chk), order, cv, out,
      el('p', { class: 'muted', html: 'Barre blu: vettore di uscita di "gatto" nell\'ordine originale. Barre arancioni: lo stesso dopo il rimescolamento. (Qui l\'attention guarda tutti i token; nei modelli causali la maschera lascia filtrare un po\' di informazione posizionale, ma esplicita e robusta serve comunque.)' })));
    run();

    root.append(el('h2', { text: 'Tre soluzioni' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Posizioni apprese</b> (GPT-2): una seconda tabella <var>P</var> di forma <var>T</var><sub>max</sub>×<var>d</var>; si somma la riga della posizione all'embedding. Semplice, ma non si estende oltre la lunghezza vista in training.</li>
      <li><b>Sinusoidale</b> (transformer originale): vettori fissi, costruiti con seni e coseni a frequenze diverse. Nessun parametro. Qui sotto.</li>
      <li><b>RoPE</b> (Llama, Qwen, Mistral e molti altri): non si somma nulla; si <i>ruotano</i> query e key in funzione della posizione, così che il loro prodotto scalare dipenda solo dalla distanza relativa. Più in basso.</li>` }));

    /* ---- Lab B: sinusoidal ---- */
    const ch = canvasEl(300), cd = canvasEl(220), cw = canvasEl(220);
    const sN = slider({ label: 'posizioni', min: 8, max: 128, step: 8, value: 64, onInput: () => redrawB() });
    const sD = slider({ label: 'd', min: 8, max: 128, step: 8, value: 64, onInput: () => redrawB() });
    const hm = setupCanvas(ch, (ctx, w, h, t) => {
      const N = sN.get(), D = sD.get(), left = 36, top = 18, cwid = (w - left - 6) / D, chei = (h - top - 6) / N;
      const M = range(N).map(p => sinPE(p, D));
      heat(ctx, M, left, top, cwid, chei, { t, diverging: true, vmin: -1, vmax: 1 });
      text(ctx, 'dimensione →  (frequenza decresce da sinistra a destra)', left, 11, { color: t.muted, size: 11 });
      ctx.save(); ctx.translate(11, h / 2 + 30); ctx.rotate(-Math.PI / 2); text(ctx, 'posizione ↓', 0, 0, { color: t.muted, size: 11 }); ctx.restore();
    });
    const dc = setupCanvas(cd, (ctx, w, h, t) => {
      const D = sD.get(), ks = range(81), f = (p, k) => dot(sinPE(p, D), sinPE(p + k, D)) / D;
      chart(ctx, w, h, t, [{ xs: ks, ys: ks.map(k => f(0, k)), color: t.accent, lw: 3, label: 'p = 0' }, { xs: ks, ys: ks.map(k => f(37, k)), color: t.amber, dash: [6, 4], label: 'p = 37' }],
        { xlabel: 'distanza k tra le due posizioni', ylabel: 'PE(p)·PE(p+k) / d', ymin: -0.5, ymax: 1, ny: 3, xmin: 0, xmax: 80 });
    });
    const wc = setupCanvas(cw, (ctx, w, h, t) => {
      const D = sD.get(), xs = range(129), cols = [t.accent, t.amber, t.teal, t.rose];
      const S = [0, 1, 2, 3].map(j => { const i = Math.round(j * (D / 2) / 4); return { i, xs, ys: xs.map(p => Math.sin(p / Math.pow(10000, 2 * i / D))), color: cols[j], label: 'i=' + i }; });
      chart(ctx, w, h, t, S, { xlabel: 'posizione', ylabel: 'sin(p·ωᵢ)', ymin: -1, ymax: 1, ny: 2, xmin: 0, xmax: 128 });
    });
    function redrawB() { hm.redraw(); dc.redraw(); wc.redraw(); }
    root.append(lab('Laboratorio · codifica sinusoidale', controls(sN.el, sD.el), ch,
      el('div', { class: 'grid2' }, el('div', {}, el('p', { html: '<b>Onde a frequenze diverse.</b> Le prime dimensioni oscillano veloci, le ultime lentissime: come le cifre di un contatore, che permettono di leggere la posizione a scale diverse.' }), cw),
        el('div', {}, el('p', { html: '<b>Distanza relativa.</b> Il prodotto scalare tra due posizioni dipende solo dalla loro distanza <var>k</var>: le curve per <var>p</var> = 0 e <var>p</var> = 37 sono sovrapposte.' }), cd))));
    redrawB();

    /* ---- Lab C: RoPE ---- */
    const cr = canvasEl(300);
    const sm = slider({ label: 'posizione di q (m)', min: 0, max: 30, value: 5, onInput: () => rp() });
    const sn = slider({ label: 'posizione di k (n)', min: 0, max: 30, value: 2, onInput: () => rp() });
    const sth = slider({ label: 'θ (rad/posizione)', min: 0.05, max: 1.0, step: 0.05, value: 0.4, fmt: v => v.toFixed(2), onInput: () => rp() });
    const sq = slider({ label: 'angolo di q', min: 0, max: 360, step: 5, value: 70, fmt: v => v + '°', onInput: () => rp() });
    const sk = slider({ label: 'angolo di k', min: 0, max: 360, step: 5, value: 20, fmt: v => v + '°', onInput: () => rp() });
    const tbl = el('div', { class: 'scroll-x' }), ro = el('div', { class: 'readout' });
    const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
    const state = () => { const th = sth.get(), q = [1.0 * Math.cos(sq.get() * Math.PI / 180), 1.0 * Math.sin(sq.get() * Math.PI / 180)], k = [0.8 * Math.cos(sk.get() * Math.PI / 180), 0.8 * Math.sin(sk.get() * Math.PI / 180)]; return { th, q, k, m: sm.get(), n: sn.get() }; };
    const rc = setupCanvas(cr, (ctx, w, h, t) => {
      const { th, q, k, m, n } = state(), cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.4;
      ctx.strokeStyle = t.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.moveTo(cx - R - 8, cy); ctx.lineTo(cx + R + 8, cy); ctx.moveTo(cx, cy - R - 8); ctx.lineTo(cx, cy + R + 8); ctx.stroke();
      const P = v => [cx + v[0] * R, cy - v[1] * R], qm = rot(q, m * th), kn = rot(k, n * th);
      ctx.setLineDash([4, 4]); [[q, t.accent], [k, t.amber]].forEach(([v, c]) => { const p = P(v); ctx.strokeStyle = rgba(c, 0.5); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p[0], p[1]); ctx.stroke(); }); ctx.setLineDash([]);
      let p = P(qm); arrow(ctx, cx, cy, p[0], p[1], t.accent, 3, 10); text(ctx, `q rotata (m=${m})`, p[0] + 6, p[1] - 6, { color: t.accent, size: 12, sans: true, weight: 700 });
      p = P(kn); arrow(ctx, cx, cy, p[0], p[1], t.amber, 3, 10); text(ctx, `k ruotata (n=${n})`, p[0] + 6, p[1] + 14, { color: t.amber, size: 12, sans: true, weight: 700 });
      const a1 = -Math.atan2(qm[1], qm[0]), a2 = -Math.atan2(kn[1], kn[0]);
      ctx.strokeStyle = t.rose; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, R * 0.28, Math.min(a1, a2), Math.max(a1, a2)); ctx.stroke();
    });
    function rp() {
      const { th, q, k, m, n } = state(), sc = (mm, nn) => dot(rot(q, mm * th), rot(k, nn * th));
      ro.innerHTML = `q<sub>m</sub>·k<sub>n</sub> = <b>${f2(sc(m, n), 4)}</b> &nbsp; (distanza relativa n − m = ${n - m})`;
      const pairs = [[m, n], [m + 7, n + 7], [m + 23, n + 23]];
      tbl.replaceChildren(el('table', { class: 'tbl' }, el('tr', {}, ['m', 'n', 'n − m', 'q_m · k_n'].map(x => el('th', { text: x }))), ...pairs.map(([a, b]) => el('tr', {}, [a, b, b - a, f2(sc(a, b), 6)].map(x => el('td', { text: x }))))));
      rc.redraw();
    }
    root.append(lab('Laboratorio · RoPE (una coppia di dimensioni)', controls(sm.el, sn.el, sth.el), controls(sq.el, sk.el), cr, ro, tbl,
      el('p', { html: 'Sposta insieme <var>m</var> e <var>n</var> mantenendo la loro differenza: le due frecce ruotano insieme e l\'angolo tra loro (arco rosa) non cambia, quindi il prodotto scalare resta uguale. Nel modello vero le <var>d</var> dimensioni sono divise in <var>d</var>/2 coppie, ciascuna ruotata con la propria frequenza <var>θ</var><sub>i</sub> = 10000<sup>−2i/d</sup>.' })));
    rp();

    root.append(details('math', 'La matematica: sinusoidale e RoPE', 'formule', `
      <p><b>Sinusoidale.</b> Per posizione <var>p</var> e indice di coppia <var>i</var> (0 ≤ <var>i</var> &lt; <var>d</var>/2), con <var>ω</var><sub>i</sub> = 10000<sup>−2<var>i</var>/<var>d</var></sup>:</p>
      <span class="eq"><var>PE</var>(<var>p</var>, 2<var>i</var>) = sin(<var>p</var>·<var>ω</var><sub>i</sub>) &nbsp;&nbsp; <var>PE</var>(<var>p</var>, 2<var>i</var>+1) = cos(<var>p</var>·<var>ω</var><sub>i</sub>)</span>
      <p>Poiché sin <var>a</var> sin <var>b</var> + cos <var>a</var> cos <var>b</var> = cos(<var>a</var> − <var>b</var>), il prodotto scalare è
      <span class="m"><var>PE</var>(<var>p</var>)·<var>PE</var>(<var>p</var>+<var>k</var>) = Σ<sub>i</sub> cos(<var>k</var>·<var>ω</var><sub>i</sub>)</span>: dipende solo da <var>k</var>. Inoltre <var>PE</var>(<var>p</var>+<var>k</var>) è una trasformazione lineare di <var>PE</var>(<var>p</var>) (una rotazione per ogni coppia), quindi il modello può imparare a "guardare <var>k</var> posti indietro".</p>
      <p><b>RoPE.</b> Si raggruppano le dimensioni di <var>q</var> e <var>k</var> in coppie e si applica a ogni coppia, alla posizione <var>m</var>, la rotazione 2D:</p>
      <span class="eq"><var>R</var><sub><var>mθ</var></sub> = ⎡ cos <var>mθ</var> &nbsp; −sin <var>mθ</var> ⎤ &nbsp; ⎣ sin <var>mθ</var> &nbsp;&nbsp; cos <var>mθ</var> ⎦</span>
      <p>Poiché <var>R</var><sub>a</sub><sup>T</sup><var>R</var><sub>b</sub> = <var>R</var><sub>b−a</sub>:</p>
      <span class="eq">(<var>R</var><sub><var>mθ</var></sub><var>q</var>)<sup>T</sup>(<var>R</var><sub><var>nθ</var></sub><var>k</var>) = <var>q</var><sup>T</sup> <var>R</var><sub>(<var>n</var>−<var>m</var>)<var>θ</var></sub> <var>k</var></span>
      <p>La posizione assoluta sparisce: resta solo la distanza <var>n</var> − <var>m</var>. Nessun parametro aggiunto, e il segnale posizionale agisce direttamente nel punteggio di attention. Le frequenze basse (<var>i</var> grande) ruotano lentamente e permettono di distinguere distanze grandi; per estendere il contesto oltre quello di training si riscalano le frequenze (<i>position interpolation</i>, YaRN e simili).</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Perché serve una codifica posizionale?', opts: ['Per ridurre i parametri', 'Perché l\'attention da sola non distingue l\'ordine dei token', 'Per velocizzare il training'], ans: 1, why: 'Mescolando i token l\'uscita per ciascuno non cambia: senza informazione esplicita l\'ordine è invisibile.' },
      { q: 'Quale proprietà ha RoPE?', opts: ['Il punteggio q·k dipende solo dalla distanza relativa', 'Aggiunge un vettore appreso per ogni posizione', 'Elimina la softmax'], ans: 0, why: 'La rotazione di q per m e di k per n produce un prodotto scalare funzione di n − m.' },
      { q: 'Un modello con posizioni apprese fino a 1024 riceve 2000 token. Cosa succede?', opts: ['Funziona normalmente', 'Non ha una riga di P per le posizioni oltre 1024', 'Si riscalano automaticamente'], ans: 1, why: 'La tabella P ha 1024 righe: oltre quel limite non esiste un vettore per la posizione.' },
    ]));
  },
});
