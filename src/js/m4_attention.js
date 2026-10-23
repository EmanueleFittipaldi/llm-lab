'use strict';
/* Module 4 - self-attention, step by step, with editable 2x2 weights */
registerModule({
  id: 'm4', n: 4, layer: true, title: 'Self-attention', eyebrow: 'i token si parlano', heading: 'Self-attention: chi guarda chi',
  lede: 'È il meccanismo che permette a ogni token di raccogliere informazione dagli altri. Tutto si riduce a prodotti scalari, una softmax e una media pesata.',
  build(root) {
    root.append(el('div', { html: `
      <p>Immagina che ogni token faccia tre cose con il proprio vettore <var>x</var>:</p>
      <ul class="prose">
        <li>si costruisce una <b>query</b> <var>q</var> = <var>x</var><var>W</var><sub>Q</sub>: "che tipo di informazione sto cercando?"</li>
        <li>si costruisce una <b>key</b> <var>k</var> = <var>x</var><var>W</var><sub>K</sub>: "che tipo di informazione offro a chi mi cerca?"</li>
        <li>si costruisce un <b>value</b> <var>v</var> = <var>x</var><var>W</var><sub>V</sub>: "se qualcuno mi guarda, questo è il contenuto che passo."</li>
      </ul>
      <p>Il punteggio tra il token <var>i</var> e il token <var>j</var> è il prodotto scalare <var>q</var><sub>i</sub>·<var>k</var><sub>j</sub>. Una softmax sulle <var>j</var> trasforma i punteggi in pesi che sommano a 1,
      e l'uscita del token <var>i</var> è la <b>media dei value pesata con quei pesi</b>. Qui sotto tutto è in 2 dimensioni, così puoi vedere le frecce.</p>` }));

    const TOK = ['il', 'gatto', 'mangia', 'il', 'pesce'];
    const X = [[0.3, 0], [1.0, 0.1], [0.1, 1.0], [0.3, 0], [0.9, 0]];
    const W = { q: [[0, 2], [2, 0]], k: [[1, 0], [0, 1]], v: [[1, 0], [0, 1]] };
    const presets = {
      'Verbo cerca sostantivi (e viceversa)': { q: [[0, 2], [2, 0]], k: [[1, 0], [0, 1]], v: [[1, 0], [0, 1]] },
      'I simili si guardano': { q: [[1.5, 0], [0, 1.5]], k: [[1.5, 0], [0, 1.5]], v: [[1, 0], [0, 1]] },
      'Uniforme (W_Q = 0)': { q: [[0, 0], [0, 0]], k: [[1, 0], [0, 1]], v: [[1, 0], [0, 1]] },
      'Casuale': null,
    };
    let causal = false, stage = 3, qi = 2, cur = null;
    const inputs = { q: [], k: [], v: [] };
    const matrixBox = key => {
      const grid = el('div', { class: 'mgrid' });
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        const inp = el('input', { type: 'number', step: '0.1', value: String(W[key][i][j]), id: uid('w'), 'aria-label': `W${key.toUpperCase()}[${i}][${j}]` });
        inp.addEventListener('input', () => { const v = parseFloat(inp.value); if (Number.isFinite(v)) { W[key][i][j] = v; update(); } });
        inputs[key].push(inp); grid.append(inp);
      }
      return el('div', { class: 'mat' }, el('span', { class: 'mname', html: `W<sub>${key.toUpperCase()}</sub>` }), grid);
    };
    const syncInputs = () => ['q', 'k', 'v'].forEach(k => inputs[k].forEach((inp, n) => { inp.value = String(+W[k][n >> 1][n & 1].toFixed(2)); }));
    const applyPreset = name => {
      let p = presets[name];
      if (!p) { const r = rng(Math.floor(Math.random() * 1e6)); const m = () => range(2).map(() => range(2).map(() => +(r.gauss() * 1.2).toFixed(1))); p = { q: m(), k: m(), v: m() }; }
      ['q', 'k', 'v'].forEach(k => { W[k] = p[k].map(r => r.slice()); }); syncInputs(); update();
    };

    const qkvBox = el('div', { class: 'scroll-x' });
    const heatCv = canvasEl(260), geoCv = canvasEl(320), readout = el('div', { class: 'readout' });
    const stageNames = ['1 · Q Kᵀ', '2 · ÷ √d_k', '3 · maschera', '4 · softmax'];
    const stageBtns = stageNames.map((n, i) => el('button', { class: 'btn', type: 'button', 'aria-pressed': String(i === stage), text: n, onclick: () => { stage = i; stageBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(k === i))); hm.redraw(); } }));
    const maskChk = checkbox('maschera causale (nessun sguardo al futuro)', false, v => { causal = v; update(); });

    function compute() {
      const r = attention(X, W.q, W.k, W.v, causal);
      const S0 = MX.mul(r.Q, MX.T(r.K)), S2 = r.S.map((row, i) => row.map((v, j) => causal && j > i ? -Infinity : v));
      cur = { ...r, S0, S1: r.S, S2 };
    }
    function update() {
      compute();
      const fmt = v => `(${f2(v[0])}, ${f2(v[1])})`;
      qkvBox.replaceChildren(el('table', { class: 'tbl' }, el('tr', {}, ['token', 'x', 'q = xW_Q', 'k = xW_K', 'v = xW_V'].map(h => el('th', { text: h }))),
        ...TOK.map((t, i) => el('tr', { class: i === qi ? 'sel pick' : 'pick', onclick: () => { qi = i; update(); } }, [`${i} ${t}`, fmt(X[i]), fmt(cur.Q[i]), fmt(cur.K[i]), fmt(cur.V[i])].map(x => el('td', { text: x }))))));
      const a = cur.A[qi], z = cur.Z[qi];
      readout.innerHTML = `Query selezionata: <b>${qi} "${TOK[qi]}"</b> (clicca una riga della tabella o della matrice)<br>` +
        `pesi: ${TOK.map((t, j) => `${t}<sub>${j}</sub> <b>${f2(a[j])}</b>`).join(' · ')}<br>` +
        `uscita z = Σ<sub>j</sub> a<sub>j</sub> v<sub>j</sub> = <b>${fmt(z)}</b>`;
      hm.redraw(); gc.redraw();
    }

    const hm = setupCanvas(heatCv, (ctx, w, h, t) => {
      if (!cur) return;
      const T = TOK.length, left = 74, top = 30, cw = Math.min(78, (w - left - 8) / T), ch = Math.min(44, (h - top - 8) / T);
      const M = [cur.S0, cur.S1, cur.S2, cur.A][stage];
      heat(ctx, M, left, top, cw, ch, { t, diverging: stage < 3, vmin: stage === 3 ? 0 : undefined, vmax: stage === 3 ? 1 : undefined, num: true, rows: TOK.map((x, i) => `${x} ${i}`), cols: TOK.map((x, j) => `${x} ${j}`), hi: qi });
      ctx.strokeStyle = t.amber; ctx.lineWidth = 2; ctx.strokeRect(left - 1, top + qi * ch - 1, cw * T + 1, ch + 1);
      text(ctx, 'query i ↓   key j →', 6, 14, { color: t.muted, size: 11 });
      const lab = ['punteggi grezzi', 'punteggi scalati', 'punteggi mascherati', 'pesi di attention A (ogni riga somma a 1)'][stage];
      text(ctx, lab, left + cw * T + 10 > w ? left : left + 110, 14, { color: t.ink, size: 12, sans: true, weight: 700 });
    });
    heatCv.addEventListener('click', ev => {
      const rc = heatCv.getBoundingClientRect(), y = ev.clientY - rc.top, ch = Math.min(44, (rc.height - 38) / TOK.length), i = Math.floor((y - 30) / ch);
      if (i >= 0 && i < TOK.length) { qi = i; update(); }
    });

    const gc = setupCanvas(geoCv, (ctx, w, h, t) => {
      if (!cur) return;
      const maxn = Math.max(...[...cur.Q, ...cur.K, ...cur.V, ...cur.Z].map(norm), 1e-6), R = Math.min(w / 2, h) * 0.38 / maxn;
      [0, 1].forEach(p => {
        const cx = w * (0.25 + 0.5 * p), cy = h / 2 + 8, P = v => [cx + v[0] * R, cy - v[1] * R];
        ctx.strokeStyle = t.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx - maxn * R, cy); ctx.lineTo(cx + maxn * R, cy); ctx.moveTo(cx, cy - maxn * R); ctx.lineTo(cx, cy + maxn * R); ctx.arc(cx, cy, maxn * R, 0, 7); ctx.stroke();
        text(ctx, p ? 'value e uscita' : 'query e key', cx, 16, { align: 'center', color: t.ink, size: 12, sans: true, weight: 700 });
        if (p === 0) {
          const q = cur.Q[qi], nq = norm(q) || 1, u = [q[0] / nq, q[1] / nq];
          ctx.setLineDash([5, 4]); ctx.strokeStyle = rgba(t.accent, 0.55); ctx.beginPath(); const a = P([-u[0] * maxn, -u[1] * maxn]), b = P([u[0] * maxn, u[1] * maxn]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]);
          cur.K.forEach((k, j) => {
            const tip = P(k), pr = dot(k, u), pp = P([u[0] * pr, u[1] * pr]);
            ctx.strokeStyle = rgba(t.amber, 0.45); ctx.beginPath(); ctx.moveTo(tip[0], tip[1]); ctx.lineTo(pp[0], pp[1]); ctx.stroke();
            ctx.fillStyle = t.amber; ctx.beginPath(); ctx.arc(pp[0], pp[1], 3, 0, 7); ctx.fill();
            arrow(ctx, cx, cy, tip[0], tip[1], rgba(t.amber, 0.9), 2, 8);
            text(ctx, `k${j}`, tip[0] + 5, tip[1] - 4, { color: t.amber, size: 11, weight: 700 });
          });
          const tq = P(q); arrow(ctx, cx, cy, tq[0], tq[1], t.accent, 3.5, 11); text(ctx, `q${qi} (${TOK[qi]})`, tq[0] + 6, tq[1] + 14, { color: t.accent, size: 12, sans: true, weight: 700 });
        } else {
          const A = cur.A[qi];
          cur.V.forEach((v, j) => { const tip = P(v); arrow(ctx, cx, cy, tip[0], tip[1], rgba(t.teal, 0.2 + 0.8 * A[j]), 1.5 + 3 * A[j], 8); if (A[j] > 0.07) text(ctx, `v${j}`, tip[0] + 5, tip[1] - 4, { color: t.teal, size: 11, weight: 700 }); });
          const z = P(cur.Z[qi]); arrow(ctx, cx, cy, z[0], z[1], t.rose, 4, 12); text(ctx, 'z', z[0] + 7, z[1] + 12, { color: t.rose, size: 13, sans: true, weight: 700 });
        }
      });
      text(ctx, 'punti ambra sulla retta di q: proiezione di ogni k = punteggio (non scalato)', 8, h - 8, { color: t.muted, size: 10 });
    });

    root.append(lab('Laboratorio · 1. i pesi e i vettori',
      el('p', { html: 'Cinque token, ciascuno con un embedding di 2 numeri (<var>x</var><sub>0</sub> ≈ "è un sostantivo", <var>x</var><sub>1</sub> ≈ "è un verbo"; scelti a mano). Le tre matrici 2×2 sono i pesi che il modello imparerebbe: modificali o scegli un preset.' }),
      controls(...Object.keys(presets).map(n => button(n, () => applyPreset(n)))),
      el('div', { class: 'mats' }, matrixBox('q'), matrixBox('k'), matrixBox('v')), qkvBox));
    root.append(lab('Laboratorio · 2. dalla matrice dei punteggi ai pesi', controls(...stageBtns, maskChk), heatCv, readout,
      el('p', { html: 'Con il preset "Verbo cerca sostantivi", la riga di <b>mangia</b> punta su gatto e pesce, mentre la riga di <b>gatto</b> punta su mangia. Nota anche che i due "il" hanno righe identiche: stesso embedding, nessuna posizione (modulo 3).' })));
    root.append(lab('Laboratorio · 3. la geometria', geoCv,
      el('p', { html: 'A sinistra: la freccia blu è la query selezionata, le frecce ambra sono le key. Il punto ambra sulla retta tratteggiata è la proiezione di <var>k</var><sub>j</sub> su <var>q</var>: più è lontano nella direzione di <var>q</var>, più alto è il punteggio. A destra: i value sono pesati dai pesi di attention (più sono spessi, più contano) e la loro somma è l\'uscita <var>z</var>.' })));
    update();

    /* ---- sqrt(d) lab ---- */
    root.append(el('h2', { text: 'Perché dividere per √d' }));
    root.append(el('p', { html: 'Se le componenti di <var>q</var> e <var>k</var> hanno varianza 1, il loro prodotto scalare ha varianza <var>d</var><sub>k</sub>: la sua deviazione standard cresce come √<var>d</var><sub>k</sub>. Con <var>d</var><sub>k</sub> = 128 i punteggi oscillano facilmente di ±30, e la softmax diventa quasi un one-hot: un peso vale ~1, tutti gli altri ~0, e <b>il gradiente svanisce</b>. Dividere per √<var>d</var><sub>k</sub> riporta la deviazione standard a 1.' }));
    const dkS = slider({ label: 'd_k', min: 1, max: 512, value: 64, onInput: () => sq(), id: 'dk-slider' }), sqCv = canvasEl(230), sqOut = el('div', { class: 'readout' });
    let hist = null;
    function sq() {
      const d = dkS.get(), r = rng(5), N = 1500, raw = [], T = 8; let pmaxR = 0, pmaxS = 0, jR = 0, jS = 0;
      for (let n = 0; n < N; n++) { let s = 0; for (let i = 0; i < d; i++) s += r.gauss() * r.gauss(); raw.push(s); }
      for (let n = 0; n < 400; n++) {
        const sc = range(T).map(() => { let s = 0; for (let i = 0; i < d; i++) s += r.gauss() * r.gauss(); return s; });
        const pr = softmax(sc), ps = softmax(sc.map(v => v / Math.sqrt(d)));
        pmaxR += Math.max(...pr) / 400; pmaxS += Math.max(...ps) / 400; jR += sum(pr.map(p => p * (1 - p))) / 400; jS += sum(ps.map(p => p * (1 - p))) / 400;
      }
      const sd = Math.sqrt(sum(raw.map(v => v * v)) / N);
      hist = { raw, d, sd };
      sqOut.innerHTML = `Deviazione standard dei punteggi: senza scala <b>${f2(sd, 1)}</b> (teoria √d = ${f2(Math.sqrt(d), 1)}) · con scala <b>${f2(sd / Math.sqrt(d))}</b><br>` +
        `Con 8 key: peso massimo medio senza scala <b>${f2(pmaxR)}</b>, con scala <b>${f2(pmaxS)}</b> · segnale di gradiente Σp(1−p): <b>${f2(jR)}</b> contro <b>${f2(jS)}</b>`;
      sc2.redraw();
    }
    const sc2 = setupCanvas(sqCv, (ctx, w, h, t) => {
      if (!hist) return;
      const { raw, d } = hist, lim = 3.2 * Math.sqrt(d), B = 41, bins = new Array(B).fill(0), bs = new Array(B).fill(0), xs = range(B).map(i => -lim + (i + 0.5) * 2 * lim / B);
      raw.forEach(v => { const k = Math.floor((v + lim) / (2 * lim) * B); if (k >= 0 && k < B) bins[k]++; const u = v / Math.sqrt(d), k2 = Math.floor((u + lim) / (2 * lim) * B); if (k2 >= 0 && k2 < B) bs[k2]++; });
      const ys = bins.map(v => v / raw.length), ys2 = bs.map(v => v / raw.length);
      chart(ctx, w, h, t, [{ xs, ys, color: t.amber, lw: 2.5, label: 'q·k grezzo' }, { xs, ys: ys2, color: t.accent, lw: 2.5, label: 'q·k / √d' }], { xlabel: 'valore del punteggio', ylabel: 'frazione dei campioni', ymin: 0, xmin: -lim, xmax: lim, nx: 6 });
    });
    root.append(lab('Laboratorio · il peso della dimensione (simulazione con numeri casuali N(0,1))', controls(dkS.el), sqCv, sqOut)); sq();

    root.append(el('h2', { text: 'La maschera causale' }));
    root.append(el('p', { html: 'In un modello che genera testo, la posizione <var>i</var> non può vedere le posizioni <var>j</var> &gt; <var>i</var>: nel training userebbe la risposta come input. Si somma ai punteggi una matrice <var>M</var> con 0 sotto e sulla diagonale e −∞ sopra: dopo la softmax i pesi del futuro sono esattamente 0. Attiva la casella nel laboratorio 2 e osserva lo stadio "maschera". Grazie a questo trucco una sola passata calcola in parallelo la previsione per <b>tutte</b> le posizioni di una sequenza.' }));

    root.append(details('math', 'La matematica: self-attention completa', 'formule', `
      <p>Sia <var>X</var> ∈ ℝ<sup><var>T</var>×<var>d</var></sup> la matrice con un token per riga. Con <var>W</var><sub>Q</sub>, <var>W</var><sub>K</sub> ∈ ℝ<sup><var>d</var>×<var>d</var><sub>k</sub></sup> e <var>W</var><sub>V</sub> ∈ ℝ<sup><var>d</var>×<var>d</var><sub>v</sub></sup>:</p>
      <span class="eq"><var>Q</var> = <var>XW</var><sub>Q</sub>, &nbsp; <var>K</var> = <var>XW</var><sub>K</sub>, &nbsp; <var>V</var> = <var>XW</var><sub>V</sub></span>
      <span class="eq"><span class="fn">Attention</span>(<var>Q</var>,<var>K</var>,<var>V</var>) = <span class="fn">softmax</span>( ${frac('<var>QK</var><sup>T</sup>', '√<var>d</var><sub>k</sub>')} + <var>M</var> ) <var>V</var></span>
      <p>Forme: <var>QK</var><sup>T</sup> è <var>T</var>×<var>T</var>, la softmax agisce <b>per riga</b>, il risultato è <var>T</var>×<var>d</var><sub>v</sub>. Per il token <var>i</var>:</p>
      <span class="eq"><var>z</var><sub>i</sub> = Σ<sub>j≤i</sub> <var>a</var><sub>ij</sub> <var>v</var><sub>j</sub>, &nbsp;&nbsp; <var>a</var><sub>ij</sub> = ${frac('exp(<var>q</var><sub>i</sub>·<var>k</var><sub>j</sub> / √<var>d</var><sub>k</sub>)', 'Σ<sub>l≤i</sub> exp(<var>q</var><sub>i</sub>·<var>k</var><sub>l</sub> / √<var>d</var><sub>k</sub>)')}</span>
      <p>I pesi <var>a</var><sub>ij</sub> sono non negativi e sommano a 1: <var>z</var><sub>i</sub> sta nell'<i>involucro convesso</i> dei value. L'attention non inventa nulla, <b>mescola</b>; la trasformazione non lineare arriva dall'MLP.</p>
      <p><b>Solo il prodotto conta.</b> Il punteggio è <var>x</var><sub>i</sub> <var>W</var><sub>Q</sub><var>W</var><sub>K</sub><sup>T</sup> <var>x</var><sub>j</sub><sup>T</sup>: una forma bilineare di rango al più <var>d</var><sub>k</sub>. Analogamente <var>W</var><sub>V</sub> e la proiezione di uscita <var>W</var><sub>O</sub> agiscono insieme come <var>W</var><sub>V</sub><var>W</var><sub>O</sub>.</p>
      <p><b>Varianza dei punteggi.</b> Se le componenti di <var>q</var>, <var>k</var> sono indipendenti con media 0 e varianza 1: Var(<var>q</var>·<var>k</var>) = Σ<sub>l</sub> Var(<var>q</var><sub>l</sub><var>k</sub><sub>l</sub>) = <var>d</var><sub>k</sub>.</p>
      <p><b>Gradiente della softmax.</b> Per <var>a</var> = softmax(<var>s</var>): ∂<var>a</var><sub>j</sub>/∂<var>s</var><sub>l</sub> = <var>a</var><sub>j</sub>(δ<sub>jl</sub> − <var>a</var><sub>l</sub>). Se un <var>a</var><sub>j</sub> → 1 (o 0), tutti i termini tendono a 0: saturazione.</p>
      <p><b>Costo.</b> Tempo <i>O</i>(<var>T</var>²<var>d</var>) e, se si materializza <var>A</var>, memoria <i>O</i>(<var>T</var>²) per head: è il motivo per cui il contesto lungo è costoso. FlashAttention calcola lo stesso risultato a blocchi senza costruire mai l'intera matrice.</p>`));
    root.append(details('refresh', 'Ripasso: la softmax', 'funzioni', `
      <p>La softmax trasforma un vettore di numeri reali <var>s</var> in una distribuzione di probabilità: <span class="m">softmax(<var>s</var>)<sub>j</sub> = e<sup><var>s</var><sub>j</sub></sup> / Σ<sub>l</sub> e<sup><var>s</var><sub>l</sub></sup></span>. L'esponenziale rende tutto positivo ed enfatizza le differenze; la divisione normalizza.</p>
      <p>Esempio: softmax([2, 1, 0]) = [e², e¹, e⁰] / (e²+e¹+e⁰) = [7,39; 2,72; 1] / 11,11 ≈ [0,665; 0,245; 0,090]. Aggiungere una costante a tutti i punteggi non cambia il risultato: per questo nel codice si sottrae il massimo (stabilità numerica).</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Perché si divide per √<var>d</var><sub>k</sub> prima della softmax?', opts: ['Per rendere i pesi indipendenti dalla lunghezza della frase', 'Per mantenere i punteggi su una scala ≈ 1 ed evitare softmax saturate con gradienti nulli', 'Per far sommare i pesi a 1'], ans: 1, why: 'La varianza di q·k cresce con d_k. Senza scala la softmax diventa quasi one-hot; la normalizzazione a somma 1 la fa già la softmax stessa.' },
      { q: 'Cosa fa la maschera causale?', opts: ['Azzera i value dei token passati', 'Mette −∞ nei punteggi verso il futuro, così i loro pesi dopo la softmax valgono 0', 'Riduce la dimensione di K'], ans: 1, why: 'exp(−∞) = 0: la posizione i ignora le posizioni successive, e si può addestrare tutta la sequenza in parallelo.' },
      { q: 'L\'uscita di un token è…', opts: ['il suo value moltiplicato per sé stesso', 'una media pesata dei value di tutti i token visibili', 'la query più la key'], ans: 1, why: 'z_i = Σ_j a_ij v_j, con pesi a_ij ≥ 0 che sommano a 1.' },
      { q: 'Come cresce il costo dell\'attention con la lunghezza <var>T</var>?', opts: ['Linearmente', 'Quadraticamente', 'Logaritmicamente'], ans: 1, why: 'Ogni token si confronta con tutti gli altri: T×T punteggi.' },
    ]));
  },
});
