'use strict';
/* Module 5 - multi-head attention, head patterns, induction heads */

/* induction circuit: head A moves "previous token" info, head B matches it and copies the following token */
function inductionCircuit(tokens, beta) {
  const T = tokens.length, vocab = [...new Set(tokens)];
  const A1 = tokens.map((_, i) => range(T).map(j => (i === 0 ? j === 0 : j === i - 1) ? 1 : 0));
  const A2 = tokens.map((_, i) => {
    const sc = range(T).map(j => {
      if (j > i) return -Infinity;
      return (j >= 1 && tokens[j - 1] === tokens[i]) ? beta : 0;
    });
    return softmax(sc);
  });
  const P = A2.map(row => vocab.map(v => sum(row.map((a, j) => tokens[j] === v ? a : 0))));
  const matched = tokens.map((_, i) => range(i + 1).some(j => j >= 1 && j !== i && tokens[j - 1] === tokens[i]));
  return { vocab, A1, A2, P, matched };
}
LAB.tests.induction = inductionCircuit;

registerModule({
  id: 'm5', n: 5, layer: true, title: 'Multi-head', eyebrow: 'molte attention in parallelo', heading: 'Più head, più modi di guardare',
  lede: 'Una sola attention può imparare un solo tipo di relazione per volta. Con più head in parallelo, ognuna in un sottospazio più piccolo, il modello ne tiene aperte molte insieme, allo stesso costo.',
  build(root) {
    root.append(el('div', { html: `
      <p>In una frase come "il gatto che ha mangiato il pesce dorme", un token può aver bisogno di sapere contemporaneamente: <i>chi è il mio soggetto</i>, <i>qual è la parola subito prima</i>, <i>dove è già comparsa questa parola</i>.
      Una singola distribuzione di attention non può puntare a tutto insieme. La soluzione è dividere i <var>d</var> canali in <var>h</var> gruppi da <var>d</var><sub>head</sub> = <var>d</var>/<var>h</var>: ogni <b>head</b> ha le sue <var>W</var><sub>Q</sub>, <var>W</var><sub>K</sub>, <var>W</var><sub>V</sub>, calcola la propria attention e produce un'uscita di <var>d</var><sub>head</sub> numeri.
      Le uscite si concatenano (tornando a <var>d</var> canali) e una matrice finale <var>W</var><sub>O</sub> le rimescola.</p>` }));

    /* ---- Lab A: shapes ---- */
    const sd = slider({ label: 'd_model', min: 64, max: 1024, step: 64, value: 512, onInput: () => shp() });
    const sh = slider({ label: 'head (h)', min: 1, max: 16, value: 8, onInput: () => shp() });
    const sT = slider({ label: 'T (token)', min: 128, max: 8192, step: 128, value: 2048, onInput: () => shp() });
    const shapeBox = el('div', { class: 'flow' }), shapeOut = el('div', { class: 'readout' });
    function shp() {
      const d = sd.get(), h = sh.get(), T = sT.get(), dh = d / h, ok = Number.isInteger(dh);
      const st = [['Input X', `[T, d] = [${T}, ${d}]`], ['Q, K, V per head', `h × [T, d_head] = ${h} × [${T}, ${ok ? dh : '?'}]`], ['Attention per head', `h × [T, T] = ${h} × [${T}, ${T}]`], ['Uscite delle head', `h × [T, ${ok ? dh : '?'}]`], ['Concat', `[T, h·d_head] = [${T}, ${ok ? d : '?'}]`], ['× W_O', `[T, d] = [${T}, ${d}]`]];
      shapeBox.replaceChildren(...st.map(([a, b]) => el('div', { class: 'stg', style: 'cursor:default' }, el('b', { text: a }), el('span', { text: b }))));
      shapeOut.innerHTML = ok
        ? `d_head = ${d}/${h} = <b>${dh}</b> · parametri attention per layer: <b>4d² = ${human(4 * d * d)}</b> (non dipendono da h) · entrate delle matrici di attention: <b>h·T² = ${human(h * T * T)}</b> per layer (in fp16 ≈ ${(h * T * T * 2 / 1e9).toFixed(2)} GB se materializzate)`
        : '<span style="color:var(--rose)">d_model deve essere divisibile per h: scegli un altro numero di head.</span>';
    }
    root.append(lab('Laboratorio · forme e costi', controls(sd.el, sh.el, sT.el), shapeBox, shapeOut)); shp();
    root.append(el('p', { html: 'Raddoppiare le head dimezza <var>d</var><sub>head</sub> e lascia invariati parametri e FLOP. Cambia solo quanta memoria occupano le matrici di attention (e, in generazione, la <b>KV cache</b>: per questo i modelli recenti condividono K e V tra più head con <i>grouped-query attention</i>).' }));

    /* ---- Lab B: head patterns ---- */
    const SENT = ['il', 'gatto', 'che', 'ha', 'mangiato', 'il', 'pesce', 'dorme'], N = SENT.length;
    const headMat = fn => SENT.map((_, i) => {
      const row = new Array(N).fill(0); let used = 0;
      for (const [j, w] of fn(i)) { row[j] += w; used += w; }
      const left = Math.max(0, 1 - used); for (let j = 0; j <= i; j++) row[j] += left / (i + 1);
      return row;
    });
    const SUBJ = { 3: [[1, 0.85]], 4: [[1, 0.45], [3, 0.4]], 6: [[4, 0.85]], 7: [[1, 0.85]] };
    const HEADS = [
      { name: 'Token precedente', color: 'accent', M: headMat(i => [[i === 0 ? 0 : i - 1, 0.85]]), why: 'Ogni token guarda quello subito prima. Serve a copiare informazione "di vicinato": è un mattone di molti circuiti.' },
      { name: 'Stessa parola', color: 'amber', M: headMat(i => { const js = range(i).filter(j => SENT[j] === SENT[i]); return js.length ? [[js[js.length - 1], 0.85]] : [[i, 0.6]]; }), why: 'Il secondo "il" guarda il primo. Utile per riconoscere ripetizioni.' },
      { name: 'Verbo → soggetto', color: 'teal', M: headMat(i => SUBJ[i] || [[i, 0.6]]), why: 'I verbi cercano il loro soggetto, "pesce" cerca il suo verbo. È il tipo di relazione sintattica che le head sembrano poter rappresentare.' },
      { name: 'Attention sink', color: 'rose', M: headMat(i => [[0, i === 0 ? 0.9 : 0.7]]), why: 'Quasi tutto il peso va al primo token, che funge da "parcheggio" quando nessuna informazione è rilevante. Fenomeno osservato nei modelli reali.' },
    ];
    let hsel = 2, qsel = 7;
    const arcCv = canvasEl(230), matCv = canvasEl(190), headInfo = el('p', { class: 'readout' });
    const arcs = setupCanvas(arcCv, (ctx, w, h, t) => {
      const H = HEADS[hsel], col = t[H.color], xs = SENT.map((_, i) => 40 + i * (w - 80) / (N - 1)), base = h - 52;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < i; j++) {
          const wgt = H.M[i][j]; if (wgt < 0.12) continue;
          ctx.strokeStyle = rgba(col, i === qsel ? Math.min(1, 0.25 + wgt) : 0.16 + 0.12 * wgt); ctx.lineWidth = 1 + wgt * (i === qsel ? 6 : 3);
          ctx.beginPath(); ctx.moveTo(xs[i], base); ctx.quadraticCurveTo((xs[i] + xs[j]) / 2, base - 24 - 20 * (i - j), xs[j], base); ctx.stroke();
        }
        const self = H.M[i][i]; if (self > 0.12) { ctx.strokeStyle = rgba(col, i === qsel ? 0.9 : 0.25); ctx.lineWidth = 1 + 3 * self; ctx.beginPath(); ctx.arc(xs[i], base - 8, 7, 0, 7); ctx.stroke(); }
        ctx.fillStyle = i === qsel ? t.ink : t.muted; ctx.beginPath(); ctx.arc(xs[i], base + 4, 3.5, 0, 7); ctx.fill();
        text(ctx, SENT[i], xs[i], base + 24, { align: 'center', color: i === qsel ? t.ink : t.muted, size: 12, weight: i === qsel ? 700 : 500, sans: true });
      }
      text(ctx, `head "${H.name}" · archi = dal token (query) a quelli che guarda (key) · clicca una parola per evidenziarla`, 10, 16, { color: t.muted, size: 11 });
    });
    arcCv.addEventListener('click', ev => {
      const rc = arcCv.getBoundingClientRect(), x = ev.clientX - rc.left; let best = 0, bd = 1e9;
      SENT.forEach((_, i) => { const d = Math.abs(x - (40 + i * (rc.width - 80) / (N - 1))); if (d < bd) { bd = d; best = i; } });
      qsel = best; arcs.redraw();
    });
    const mats = setupCanvas(matCv, (ctx, w, h, t) => {
      const gap = 14, S = Math.min((w - gap * 3 - 24) / 4, h - 46), cell = S / N;
      HEADS.forEach((H, k) => {
        const x = 12 + k * (S + gap), y = 10;
        heat(ctx, H.M, x, y, cell, cell, { t, color: t[H.color], vmin: 0, vmax: 1 });
        if (k === hsel) { ctx.strokeStyle = t.ink; ctx.lineWidth = 2; ctx.strokeRect(x - 2, y - 2, S + 3, S + 3); }
        text(ctx, H.name, x + S / 2, y + S + 18, { align: 'center', color: k === hsel ? t.ink : t.muted, size: 11, weight: k === hsel ? 700 : 500, sans: true });
      });
    });
    matCv.addEventListener('click', ev => {
      const rc = matCv.getBoundingClientRect(), x = ev.clientX - rc.left, gap = 14, S = Math.min((rc.width - gap * 3 - 24) / 4, rc.height - 46);
      const k = Math.floor((x - 12 + gap / 2) / (S + gap)); if (k >= 0 && k < HEADS.length) { hsel = k; upd(); }
    });
    const headBtns = HEADS.map((H, k) => button(H.name, () => { hsel = k; upd(); }));
    function upd() { headBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(k === hsel))); headInfo.innerHTML = `<b>${HEADS[hsel].name}.</b> ${HEADS[hsel].why}`; arcs.redraw(); mats.redraw(); }
    root.append(lab('Laboratorio · quattro head sulla stessa frase', controls(...headBtns), arcCv, matCv, headInfo,
      el('div', { class: 'note', html: '<b>Onestà</b>Questi quattro pattern sono <b>scritti a mano</b> per mostrare che cosa una head <i>può</i> fare. In un modello vero le head nascono dall\'addestramento, sono spesso meno nette, e molte svolgono compiti che non sappiamo nominare. Lo studio di quali head fanno cosa è un filone attivo di ricerca (interpretabilità meccanicistica).' })));
    upd();

    /* ---- Lab C: induction ---- */
    root.append(el('h2', { text: 'Un circuito vero: le induction head' }));
    root.append(el('p', { html: `Due head in due layer diversi, <b>composte</b>, riescono a fare qualcosa di sorprendente: se nel testo compare "… <i>A B</i> … <i>A</i>", prevedono che il prossimo token sia <i>B</i>. È copia-incolla guidato dal contesto, ed è considerato uno dei meccanismi alla base dell'<i>in-context learning</i> (Olsson e colleghi, Anthropic, 2022).` }));
    root.append(el('ol', { class: 'prose', html: `
      <li><b>Head 1 (previous-token head)</b>, layer precedente: ogni posizione <var>j</var> guarda <var>j</var> − 1 e scrive nel proprio residual "il token prima di me era <i>X</i>".</li>
      <li><b>Head 2 (induction head)</b>, layer successivo: la query della posizione <var>i</var> è "io sono <i>A</i>"; la key della posizione <var>j</var> contiene "il token prima di me era <i>X</i>". Il punteggio è alto quando <i>X</i> = <i>A</i>: cioè in tutti i punti che <b>seguono</b> una precedente occorrenza di <i>A</i>. Il value copia il token a <var>j</var>: proprio <i>B</i>.</li>` }));
    const defTxt = 'mario rossi lavora a milano . anna bianchi vive a roma . mario';
    const inp = el('input', { type: 'text', id: 'ind-in', value: defTxt }); inp.style.width = '100%';
    const sb = slider({ label: 'nitidezza β', min: 0, max: 12, step: 0.5, value: 8, onInput: () => ind() });
    const indCv = canvasEl(300), indOut = el('div', { class: 'readout' });
    let IC = null, toks = [], isel = 0;
    function ind() {
      toks = inp.value.toLowerCase().split(/\s+/).filter(Boolean).slice(0, 16);
      if (toks.length < 2) { indOut.textContent = 'Scrivi almeno due token separati da spazi.'; IC = null; indC.redraw(); return; }
      IC = inductionCircuit(toks, sb.get()); isel = clamp(isel, 0, toks.length - 1);
      const i = isel, pr = IC.P[i], order = range(IC.vocab.length).sort((a, b) => pr[b] - pr[a]).slice(0, 4);
      indOut.innerHTML = `Posizione <b>${i} "${toks[i]}"</b>: ` + (IC.matched[i]
        ? `la head 2 trova ${range(i + 1).filter(j => j >= 1 && j !== i && toks[j - 1] === toks[i]).length} occorrenza/e precedente/i di "${toks[i]}" e copia ciò che la seguiva → previsione: ` + order.map(k => `<b>${IC.vocab[k]}</b> ${f2(pr[k] * 100, 0)}%`).join(' · ')
        : `<span class="muted">"${toks[i]}" non è mai comparso prima: nessuna corrispondenza, l'attention si distribuisce quasi uniformemente e la head non ha una previsione utile.</span>`);
      indC.redraw();
    }
    const indC = setupCanvas(indCv, (ctx, w, h, t) => {
      if (!IC) return;
      const T = toks.length, gap = 40, S = Math.min((w - gap - 20) / 2, h - 80), cell = Math.min(S / T, 30), xs = [30 + 22, 30 + S + gap + 22];
      const left = [Math.min(46, 30), 30];
      [[IC.A1, 'Head 1 · token precedente', t.accent], [IC.A2, 'Head 2 · induction', t.amber]].forEach(([M, name, c], k) => {
        const x = k === 0 ? 40 : 40 + T * cell + gap + 40, y = 46;
        heat(ctx, M.map(r => r.map(v => v)), x, y, cell, cell, { t, color: c, vmin: 0, vmax: 1, rows: toks, cols: toks.map((s, j) => j % 1 === 0 ? s.slice(0, 4) : ''), hi: isel });
        text(ctx, name, x, 22, { color: c, size: 12, sans: true, weight: 700 });
        ctx.strokeStyle = t.ink; ctx.lineWidth = 1.5; ctx.strokeRect(x - 1, y + isel * cell - 1, T * cell + 1, cell + 1);
      });
      text(ctx, 'riga = posizione che guarda · colonna = posizione guardata · clicca una riga per cambiare posizione', 10, h - 10, { color: t.muted, size: 10 });
    });
    indCv.addEventListener('click', ev => {
      if (!toks.length) return; const rc = indCv.getBoundingClientRect(), y = ev.clientY - rc.top, S = Math.min((rc.width - 40 - 20) / 2, rc.height - 80), cell = Math.min(S / toks.length, 30), i = Math.floor((y - 46) / cell);
      if (i >= 0 && i < toks.length) { isel = i; ind(); }
    });
    inp.addEventListener('input', () => { isel = Math.max(0, inp.value.trim().split(/\s+/).length - 1); ind(); });
    root.append(lab('Laboratorio · induction head', el('p', { text: 'Modifica la sequenza (token separati da spazi). Posizionati sull\'ultimo token per vedere la previsione per "copia-incolla".' }), inp, controls(sb.el, button('Esempio con lettere', () => { inp.value = 'a b c d e a b c d e a b'; isel = 11; ind(); })), indCv, indOut));
    isel = defTxt.split(' ').length - 1; ind();

    root.append(details('math', 'La matematica: multi-head attention', 'formule', `
      <span class="eq"><span class="fn">head</span><sub><var>h</var></sub> = <span class="fn">Attention</span>(<var>XW</var><sub>Q</sub><sup>(<var>h</var>)</sup>, <var>XW</var><sub>K</sub><sup>(<var>h</var>)</sup>, <var>XW</var><sub>V</sub><sup>(<var>h</var>)</sup>) &nbsp; ∈ ℝ<sup><var>T</var>×<var>d</var><sub>head</sub></sup></span>
      <span class="eq"><span class="fn">MultiHead</span>(<var>X</var>) = <span class="fn">Concat</span>(<span class="fn">head</span><sub>1</sub>, …, <span class="fn">head</span><sub><var>h</var></sub>) <var>W</var><sub>O</sub> = Σ<sub><var>h</var></sub> <span class="fn">head</span><sub><var>h</var></sub> <var>W</var><sub>O</sub><sup>(<var>h</var>)</sup></span>
      <p>La seconda uguaglianza (<var>W</var><sub>O</sub> spezzata per righe in blocchi <var>d</var><sub>head</sub>×<var>d</var>) dice che <b>ogni head scrive nel residual stream una propria correzione additiva</b>: il contributo di una head può essere studiato da solo.</p>
      <p><b>Circuito di induzione.</b> Nel secondo layer la key a <var>j</var> è funzione di <var>x</var><sub>j−1</sub> grazie alla head 1 (<i>K-composition</i>). Se <i>e</i>(·) è l'identità del token, il punteggio ideale è <var>s</var><sub>ij</sub> = β·[<i>e</i>(<var>t</var><sub>i</sub>) = <i>e</i>(<var>t</var><sub>j−1</sub>)], e la previsione è <span class="m"><var>P</var>(<var>v</var> | <var>i</var>) = Σ<sub>j≤i</sub> <var>a</var><sub>ij</sub> [<var>t</var><sub>j</sub> = <var>v</var>]</span>. Con β grande la softmax tende a un one-hot sulle posizioni che seguono una ripetizione.</p>
      <p><b>Varianti.</b> <i>Multi-Query</i> (una sola K/V per tutte le head) e <i>Grouped-Query</i> (K/V condivise per gruppi di head) riducono la KV cache di un fattore <var>h</var>/<var>g</var> con piccola perdita di qualità.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Se raddoppio il numero di head mantenendo <var>d</var> fisso, i parametri dell\'attention…', opts: ['raddoppiano', 'restano uguali', 'si dimezzano'], ans: 1, why: 'Le matrici restano d×d in totale: cambia solo come i canali vengono suddivisi (d_head = d/h).' },
      { q: 'Cosa deve fare la <i>previous-token head</i> perché l\'induction head funzioni?', opts: ['Copiare in ogni posizione l\'identità del token precedente', 'Annullare la maschera causale', 'Moltiplicare i punteggi per √d'], ans: 0, why: 'Così la key di ogni posizione contiene "il token prima di me era X", e la seconda head può cercare le posizioni che seguono una certa occorrenza.' },
      { q: 'Perché una head potrebbe fare "attention sink" sul primo token?', opts: ['Perché è il più importante', 'Perché la softmax obbliga a distribuire peso totale 1 anche quando nulla è rilevante: serve un posto dove "parcheggiarlo"', 'È un errore di implementazione'], ans: 1, why: 'I pesi devono sommare a 1; se nessuna posizione è interessante, un token neutro assorbe il peso senza trasmettere informazione utile.' },
    ]));
  },
});
