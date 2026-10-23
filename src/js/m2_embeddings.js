'use strict';
/* Module 2 - embeddings: lookup table, geometry, cosine similarity, analogies */
(function () {
  const DIMS = ['animato', 'umano', 'genere (M+ / F−)', 'nobiltà', 'taglia', 'commestibile', 'luogo', 'azione', 'area (IT+ / FR−)'];
  const CATS = [['persone', 'accent'], ['animali', 'amber'], ['cibo', 'teal'], ['luoghi', 'rose'], ['azioni', 'muted']];
  const RAW = [
    ['re', 0, [1, 1, 1, 1, .3, 0, 0, 0, 0]], ['regina', 0, [1, 1, -1, 1, .3, 0, 0, 0, 0]],
    ['principe', 0, [1, 1, 1, .7, 0, 0, 0, 0, 0]], ['principessa', 0, [1, 1, -1, .7, 0, 0, 0, 0, 0]],
    ['uomo', 0, [1, 1, 1, 0, .2, 0, 0, 0, 0]], ['donna', 0, [1, 1, -1, 0, .1, 0, 0, 0, 0]],
    ['ragazzo', 0, [1, 1, 1, 0, -.3, 0, 0, 0, 0]], ['ragazza', 0, [1, 1, -1, 0, -.3, 0, 0, 0, 0]],
    ['gatto', 1, [1, 0, .1, 0, -.6, 0, 0, 0, 0]], ['cane', 1, [1, 0, .1, 0, -.2, 0, 0, 0, 0]],
    ['leone', 1, [1, 0, .6, .3, .6, 0, 0, 0, 0]], ['tigre', 1, [1, 0, .2, .1, .5, 0, 0, 0, 0]], ['pesce', 1, [1, 0, 0, 0, -.5, .4, 0, 0, 0]],
    ['mela', 2, [0, 0, 0, 0, -.6, 1, 0, 0, 0]], ['pane', 2, [0, 0, 0, 0, -.2, 1, 0, 0, 0]], ['pizza', 2, [0, 0, 0, 0, .1, 1, .2, 0, .5]],
    ['Roma', 3, [0, 0, 0, 0, .4, 0, 1, 0, 1]], ['Parigi', 3, [0, 0, 0, 0, .4, 0, 1, 0, -1]],
    ['Italia', 3, [0, 0, 0, 0, 1, 0, .8, 0, 1]], ['Francia', 3, [0, 0, 0, 0, 1, 0, .8, 0, -1]],
    ['correre', 4, [.5, 0, 0, 0, 0, 0, 0, 1, 0]], ['camminare', 4, [.5, 0, 0, 0, -.3, 0, 0, .8, 0]],
    ['mangiare', 4, [.5, 0, 0, 0, 0, .4, 0, 1, 0]], ['dormire', 4, [.5, 0, 0, 0, -.4, 0, 0, .5, 0]],
  ];
  const r = rng(11);
  const WORDS = RAW.map(w => w[0]), CAT = RAW.map(w => w[1]), E = RAW.map(w => w[2].map(v => v + (r() - 0.5) * 0.12));
  LAB.tests.emb = { WORDS, E };

  registerModule({
    id: 'm2', n: 2, title: 'Embedding', eyebrow: 'dagli interi ai vettori', heading: 'Ogni token è un punto in uno spazio',
    lede: 'L\'embedding è una tabella: l\'indice del token seleziona una riga. Il bello è che, dopo l\'addestramento, la geometria di quello spazio riflette il significato.',
    build(root) {
      root.append(el('div', { html: `
        <p>Un intero come 4827 non dice nulla al modello: non è "più vicino" a 4828 di quanto lo sia a 12. Serve una rappresentazione in cui <b>la vicinanza abbia senso</b>.
        Si assegna a ogni token un vettore di <var>d</var> numeri reali (da poche centinaia a diverse migliaia). Tutti questi vettori, impilati, formano la matrice di embedding <var>E</var> di forma <var>V</var>×<var>d</var>:
        ed <b>è una normale matrice di parametri</b>, inizializzata a caso e migliorata dalla backpropagation come ogni altro peso.</p>
        <p>Matematicamente "cercare l'embedding" equivale a moltiplicare il vettore one-hot del token per <var>E</var>; in pratica si legge semplicemente la riga corrispondente.</p>` }));
      root.append(el('div', { class: 'note', html: '<b>Onestà</b>I 24 vettori qui sotto sono <b>costruiti a mano</b> con 9 dimensioni interpretabili (più un po\' di rumore), solo per rendere visibile la geometria. In un modello vero le dimensioni non hanno nomi, sono migliaia, e le direzioni significative non coincidono con gli assi. Nel modulo 8 vedrai embedding veri, appresi dai dati.' }));

      let sel = WORDS.indexOf('re'), mode = 'pca', axes = [0, 2, 4], star = null;
      const cv = canvasEl(400), mat = canvasEl(330);
      let proj = [];
      function computeProj() {
        let P;
        if (mode === 'pca') P = pca(E, 3).proj; else P = E.map(row => axes.map(a => row[a]));
        const m = Math.max(...P.flat().map(Math.abs)) || 1; proj = P.map(p => p.map(v => v / m));
        if (star) star.p = (mode === 'pca' ? (() => { const pc = pca(E, 3); return pc.comps.map(c => dot(star.v.map((v, j) => v - pc.mu[j]), c)); })() : axes.map(a => star.v[a])).map(v => v / m);
      }
      const cam = new Cam3D(cv, () => c3.redraw(), { yaw: 0.6, pitch: 0.35 });
      let screen = [];
      const c3 = setupCanvas(cv, (ctx, w, h, t) => {
        if (!proj.length) return;
        const s = Math.min(w, h) * 0.38, col = i => t[CATS[CAT[i]][1]];
        const O = cam.project(0, 0, 0, w, h, s);
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((a, k) => {
          const q = cam.project(a[0] * 1.05, a[1] * 1.05, a[2] * 1.05, w, h, s), q2 = cam.project(-a[0] * 1.05, -a[1] * 1.05, -a[2] * 1.05, w, h, s);
          ctx.strokeStyle = rgba(t.line, 0.9); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(q2[0], q2[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
          text(ctx, mode === 'pca' ? 'PC' + (k + 1) : DIMS[axes[k]], q[0], q[1] - 6, { color: t.muted, size: 10, align: 'center' });
        });
        screen = proj.map((p, i) => ({ i, q: cam.project(p[0], p[1], p[2], w, h, s) }));
        if (sel >= 0) {
          screen.slice().sort((a, b) => a.q[2] - b.q[2]);
          const nn = neighbors(E[sel], [sel], 3);
          nn.forEach(n => { const a = screen[sel].q, b = screen[n.i].q; ctx.strokeStyle = rgba(t.accent, 0.5); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]); });
        }
        [...screen].sort((a, b) => a.q[2] - b.q[2]).forEach(({ i, q }) => {
          const depth = clamp(0.6 + q[2] * 0.25, 0.35, 1), rad = i === sel ? 7 : 4.5;
          ctx.globalAlpha = depth; ctx.fillStyle = col(i); ctx.beginPath(); ctx.arc(q[0], q[1], rad, 0, 7); ctx.fill();
          if (i === sel) { ctx.strokeStyle = t.ink; ctx.lineWidth = 2; ctx.stroke(); }
          text(ctx, WORDS[i], q[0] + 8, q[1] + 4, { color: i === sel ? t.ink : t.ink, size: i === sel ? 13 : 11, weight: i === sel ? 700 : 500, sans: true });
          ctx.globalAlpha = 1;
        });
        if (star && star.p) {
          const q = cam.project(star.p[0], star.p[1], star.p[2], w, h, s);
          ctx.strokeStyle = t.rose; ctx.lineWidth = 2; ctx.beginPath();
          for (let k = 0; k < 5; k++) { const a = -Math.PI / 2 + k * 4 * Math.PI / 5, x = q[0] + Math.cos(a) * 11, y = q[1] + Math.sin(a) * 11; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
          ctx.closePath(); ctx.stroke(); text(ctx, 'risultato', q[0] + 14, q[1] - 10, { color: t.rose, size: 11, weight: 700, sans: true });
        }
      });
      cv.addEventListener('click', ev => {
        if (cam.wasDrag) { cam.wasDrag = false; return; }
        const rc = cv.getBoundingClientRect(), x = ev.clientX - rc.left, y = ev.clientY - rc.top;
        let best = -1, bd = 18;
        screen.forEach(({ i, q }) => { const d = Math.hypot(q[0] - x, q[1] - y); if (d < bd) { bd = d; best = i; } });
        if (best >= 0) { sel = best; refresh(); }
      });

      function neighbors(v, excl, k) {
        return WORDS.map((w, i) => ({ i, w, c: cosine(v, E[i]) })).filter(o => !excl.includes(o.i)).sort((a, b) => b.c - a.c).slice(0, k);
      }

      /* matrix lookup */
      const cm = setupCanvas(mat, (ctx, w, h, t) => {
        const n = WORDS.length, left = 92, top = 28, ch = Math.min(15, (h - top - 6) / n), cw = Math.min(34, (w - left - 8) / DIMS.length);
        heat(ctx, E, left, top, cw, ch, { t, diverging: true, vmin: -1, vmax: 1, rows: WORDS, hi: sel, cols: DIMS.map((_, j) => 'd' + j) });
        ctx.strokeStyle = t.amber; ctx.lineWidth = 2; ctx.strokeRect(left - 1, top + sel * ch - 1, cw * DIMS.length + 1, ch + 1);
        text(ctx, 'E  (V × d)', 6, 14, { color: t.muted, size: 11 });
      });

      /* panels */
      const info = el('div', { class: 'readout' }), cmpA = select(WORDS.map((w, i) => [i, w]), WORDS.indexOf('gatto'), () => refresh(), 'confronta con');
      const cmpOut = el('div', { class: 'readout' });
      const modeSel = select([['pca', 'PCA (3 componenti)'], ['axes', 'Assi scelti da te']], 'pca', v => { mode = v; axesBox.hidden = v === 'pca'; computeProj(); c3.redraw(); }, 'proiezione');
      const axSel = [0, 1, 2].map(k => select(DIMS.map((d, j) => [j, d]), axes[k], v => { axes[k] = +v; computeProj(); c3.redraw(); }));
      const axesBox = el('div', { class: 'controls', hidden: true }, ['x', 'y', 'z'].map((n, k) => el('label', { class: 'ctl' }, el('span', { class: 'ctl-l', text: n }), axSel[k])));
      const legend = el('div', { class: 'legend' }, CATS.map(([n, c]) => el('span', { html: `<i style="background:var(--${c})"></i>${n}` })));

      function refresh() {
        const v = E[sel], nn = neighbors(v, [sel], 4);
        info.innerHTML = `<b>${WORDS[sel]}</b> · id ${sel} · riga di E<br>` + DIMS.map((d, j) => `${d.padEnd(18, ' ').replace(/ /g, '&nbsp;')} <span style="color:${v[j] >= 0 ? 'var(--accent)' : 'var(--rose)'}">${f2(v[j])}</span>`).join('<br>') +
          `<br><br>Più simili (coseno): ` + nn.map(n => `${n.w} <b>${f2(n.c)}</b>`).join(' · ');
        const o = +cmpA.querySelector('select').value, a = E[sel], b = E[o], cs = cosine(a, b);
        cmpOut.innerHTML = `${WORDS[sel]} · ${WORDS[o]}<br>prodotto scalare = <b>${f2(dot(a, b))}</b><br>coseno = <b>${f2(cs)}</b> → angolo <b>${f2(Math.acos(clamp(cs, -1, 1)) * 180 / Math.PI, 0)}°</b>`;
        c3.redraw(); cm.redraw();
      }
      root.append(lab('Laboratorio · lo spazio degli embedding (trascina per ruotare, clicca un punto)',
        controls(modeSel), axesBox, cv, legend,
        el('div', { class: 'grid2' }, el('div', {}, info), el('div', {}, controls(cmpA), cmpOut))));
      root.append(lab('Laboratorio · la tabella E: cercare = leggere una riga', mat,
        el('p', { html: 'La riga evidenziata è il vettore del token selezionato. Blu = valore positivo, rosa = negativo. Moltiplicare il one-hot <code>[0,…,1,…,0]</code> per <var>E</var> restituisce esattamente questa riga.' })));

      /* analogy */
      const names = WORDS.map((w, i) => [i, w]);
      const aS = select(names, WORDS.indexOf('re'), () => ana(), 'a'), bS = select(names, WORDS.indexOf('uomo'), () => ana(), '− b'), cS = select(names, WORDS.indexOf('donna'), () => ana(), '+ c');
      const anaOut = el('div', { class: 'readout' });
      function ana() {
        const [a, b, c] = [aS, bS, cS].map(s => +s.querySelector('select').value), v = E[a].map((x, j) => x - E[b][j] + E[c][j]);
        const nn = neighbors(v, [a, b, c], 3); star = { v };
        anaOut.innerHTML = `${WORDS[a]} − ${WORDS[b]} + ${WORDS[c]} ≈ <b>${nn[0].w}</b> (coseno ${f2(nn[0].c)})<br><span class="muted">poi: ${nn.slice(1).map(n => `${n.w} ${f2(n.c)}`).join(' · ')}</span>`;
        computeProj(); c3.redraw();
      }
      root.append(lab('Laboratorio · aritmetica sugli embedding', controls(aS, bS, cS), anaOut,
        el('p', { html: 'Prova <code>re − uomo + donna</code>, <code>Roma − Italia + Francia</code>, <code>gatto − animali + …</code>. La stella rosa nello spazio sopra mostra dove cade il risultato. Nei modelli veri questo funziona solo in modo approssimato e per alcune relazioni, ma l\'idea che relazioni diverse corrispondano a <i>direzioni</i> è reale.' })));
      computeProj(); refresh(); ana();

      root.append(el('h2', { text: 'Che cosa impara davvero la tabella' }));
      root.append(el('p', { html: `Nessuno decide che "genere" sia una direzione. L'addestramento aggiusta le righe di <var>E</var> per rendere più probabile il token successivo corretto: parole usate in contesti simili finiscono per avere righe simili, perché devono produrre previsioni simili.
        La struttura del significato è un sottoprodotto. Nel prossimo modulo vedremo come ai vettori si aggiunge l'informazione sulla posizione, e poi come l'attention li mescola: dopo i primi layer il vettore di un token non rappresenta più la parola isolata, ma la parola <i>in quel contesto</i>.` }));
      root.append(details('math', 'La matematica: coseno, prodotto scalare e lookup', 'formule', `
        <p>Il <b>prodotto scalare</b> di due vettori è <span class="m"><var>a</var>·<var>b</var> = Σ<sub><var>i</var></sub> <var>a</var><sub><var>i</var></sub><var>b</var><sub><var>i</var></sub> = ‖<var>a</var>‖‖<var>b</var>‖ cos θ</span>. Il <b>coseno</b> lo normalizza e misura solo l'angolo:</p>
        <span class="eq">cos θ = ${frac('<var>a</var>·<var>b</var>', '‖<var>a</var>‖ ‖<var>b</var>‖')}</span>
        <p>Vale 1 per vettori allineati, 0 per ortogonali, −1 per opposti. Nell'attention si usa il prodotto scalare <i>non normalizzato</i>: la lunghezza dei vettori conta, e il modello può usarla per esprimere "quanto sono sicuro".</p>
        <p><b>Lookup come prodotto matriciale.</b> Se <var>x</var> ∈ {0,1}<sup><var>V</var></sup> è il one-hot del token <var>i</var>, allora <span class="m"><var>x</var><sup>T</sup><var>E</var> = <var>E</var><sub><var>i</var>,:</sub></span>. Nella backpropagation il gradiente della loss rispetto a <var>E</var> è diverso da zero solo nelle righe dei token presenti nel batch.</p>
        <p><b>Dimensione.</b> Con <var>d</var> grande, vettori casuali sono quasi ortogonali (il coseno tipico è ~ 1/√<var>d</var>): c'è "spazio" per rappresentare molte più feature che dimensioni, a patto di tollerare piccole interferenze. È l'ipotesi di <i>superposition</i> studiata nell'interpretabilità.</p>`));
      root.append(details('refresh', 'Ripasso: prodotto scalare e matrici', 'algebra lineare', `
        <p>Un vettore è una lista di numeri; geometricamente è una freccia. Il prodotto scalare moltiplica le componenti a coppie e somma: [1, 2]·[3, 4] = 1·3 + 2·4 = 11.</p>
        <p>Una matrice <var>A</var> (<var>n</var>×<var>k</var>) per una matrice <var>B</var> (<var>k</var>×<var>m</var>) dà una matrice <var>n</var>×<var>m</var>, dove l'elemento (<var>i</var>,<var>j</var>) è il prodotto scalare tra la riga <var>i</var> di <var>A</var> e la colonna <var>j</var> di <var>B</var>. La dimensione interna <var>k</var> deve coincidere. Quasi tutto in un LLM è questa operazione.</p>`));
      root.append(el('h2', { text: 'Verifica rapida' }));
      root.append(quiz([
        { q: 'Quanti parametri ha la matrice di embedding con vocabolario 50 000 e <var>d</var> = 768?', opts: ['768', '≈ 38,4 milioni', '≈ 50 mila'], ans: 1, why: '50 000 × 768 = 38,4 milioni: una riga di 768 numeri per ogni token del vocabolario.' },
        { q: 'Due vettori con coseno vicino a 1 sono…', opts: ['quasi ortogonali', 'quasi allineati (direzione simile)', 'sempre della stessa lunghezza'], ans: 1, why: 'Il coseno misura solo l\'angolo. Due vettori possono avere direzione identica e lunghezze molto diverse.' },
        { q: 'Chi decide che una direzione dello spazio rappresenti "genere"?', opts: ['Gli ingegneri, a mano', 'L\'addestramento: emerge perché utile per prevedere il testo', 'Il tokenizer'], ans: 1, why: 'Le righe di E sono parametri come gli altri. Le regolarità del linguaggio spingono le parole simili a vettori simili.' },
      ]));
    },
  });
})();
