'use strict';
/* Module 1 - Byte-Pair Encoding, trained live on an editable corpus */

function bpeWords(text) {
  const out = [], re = /(\s*)([\p{L}\p{N}]+|[^\s\p{L}\p{N}])/gu;
  let m, first = true;
  while ((m = re.exec(text.toLowerCase())) !== null) {
    const spaced = m[1].length > 0 || (first && /[\p{L}\p{N}]/u.test(m[2]));
    out.push((spaced ? '▁' : '') + m[2]); first = false;
  }
  return out;
}
function bpePairCounts(seqs) {
  const pc = new Map();
  for (const s of seqs) for (let j = 0; j < s.syms.length - 1; j++) {
    const k = s.syms[j] + '\u0001' + s.syms[j + 1]; pc.set(k, (pc.get(k) || 0) + s.f);
  }
  return pc;
}
function bpeTop(seqs, n = 6) {
  return [...bpePairCounts(seqs)].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, n).map(([k, c]) => [...k.split('\u0001'), c]);
}
function bpeTrain(text, nMerges) {
  const freq = new Map();
  for (const w of bpeWords(text)) freq.set(w, (freq.get(w) || 0) + 1);
  const seqs = [...freq].map(([w, f]) => ({ w, syms: [...w], f }));
  const base = [...new Set(seqs.flatMap(s => s.syms))].sort();
  const merges = [], counts = [seqs.reduce((a, s) => a + s.f * s.syms.length, 0)];
  for (let i = 0; i < nMerges; i++) {
    const top = bpeTop(seqs, 1)[0];
    if (!top) break;
    const [a, b, c] = top; merges.push([a, b, c]);
    for (const s of seqs) {
      const o = []; let j = 0;
      while (j < s.syms.length) {
        if (j < s.syms.length - 1 && s.syms[j] === a && s.syms[j + 1] === b) { o.push(a + b); j += 2; } else { o.push(s.syms[j]); j++; }
      }
      s.syms = o;
    }
    counts.push(seqs.reduce((acc, s) => acc + s.f * s.syms.length, 0));
  }
  return { base, merges, seqs, counts };
}
function bpeEncodeWord(word, merges) {
  let syms = [...word];
  for (const [a, b] of merges) {
    const o = []; let j = 0;
    while (j < syms.length) {
      if (j < syms.length - 1 && syms[j] === a && syms[j + 1] === b) { o.push(a + b); j += 2; } else { o.push(syms[j]); j++; }
    }
    syms = o;
  }
  return syms;
}
const bpeEncode = (text, merges) => bpeWords(text).flatMap(w => bpeEncodeWord(w, merges));
LAB.tests.bpe = { bpeWords, bpeTrain, bpeEncode, bpeEncodeWord, bpeTop };

registerModule({
  id: 'm1', n: 1, title: 'Tokenizzazione', eyebrow: 'dal testo agli interi', heading: 'Il testo diventa una sequenza di pezzi',
  lede: 'Un modello non legge lettere né parole: legge token. La scelta dei pezzi è un compromesso tra vocabolario piccolo e sequenze corte, e si può imparare dai dati.',
  build(root) {
    root.append(el('div', { html: `
      <p>Se usassimo i caratteri avremmo un vocabolario minuscolo ma sequenze lunghissime (e l'attention costa quanto il <i>quadrato</i> della lunghezza).
      Se usassimo le parole intere il vocabolario esploderebbe e ogni parola nuova, refuso o nome proprio sarebbe sconosciuto.
      La soluzione di quasi tutti gli LLM è il <b>Byte-Pair Encoding (BPE)</b>: si parte dai caratteri e si <b>fondono ripetutamente la coppia di simboli adiacenti più frequente</b>.
      Le sequenze comuni diventano pezzi singoli, quelle rare restano spezzate in pezzi più piccoli. Nessuna parola è mai davvero sconosciuta.</p>
      <p>Qui sotto puoi addestrare un BPE su un testo tuo. Il simbolo <code>▁</code> segna l'inizio di una parola (cioè uno spazio che la precede).</p>` }));

    const defaultText = 'il gatto mangia il pesce. il gatto dorme sul divano. la gatta mangia il pesce fresco. il pesce nuota nel mare. il mare è calmo e il gatto dorme. i gatti mangiano il pesce e dormono sul divano.';
    const ta = el('textarea', { id: 'bpe-corpus', 'aria-label': 'Testo di addestramento del tokenizer' }); ta.value = defaultText;
    const segs = el('div', { class: 'scroll-x' }), pairsBox = el('div', { class: 'scroll-x' }), mergesBox = el('div', { class: 'readout' }), stats = el('div', { class: 'readout' });
    mergesBox.style.cssText = 'max-height:210px;overflow:auto';
    const encIn = el('input', { type: 'text', id: 'bpe-enc', value: 'il gattino mangia il pesce' }); encIn.style.width = '100%';
    const encOut = el('div', { class: 'chips' }), encStat = el('div', { class: 'readout' });
    const cv = canvasEl(200);
    let R = null, hist = [];

    const nm = slider({ label: 'merge', min: 0, max: 80, value: 12, onInput: () => render() });
    const reset = button('Reset testo', () => { ta.value = defaultText; recompute(); });
    const plus = button('+1 merge', () => { nm.set(Math.min(80, nm.get() + 1)); render(); }, 'primary');

    const chart1 = setupCanvas(cv, (ctx, w, h, t) => {
      if (!hist.length) return;
      const xs = hist.map((_, i) => i);
      const c = chart(ctx, w, h, t, [{ xs, ys: hist, color: t.accent, lw: 2 }], { xlabel: 'numero di merge (vocabolario = caratteri + merge)', ylabel: 'token nel testo', xmin: 0, xmax: Math.max(hist.length - 1, 1), ymin: 0, nx: 6, ny: 4 });
      const k = Math.min(nm.get(), hist.length - 1);
      ctx.fillStyle = t.amber; ctx.beginPath(); ctx.arc(c.X(k), c.Y(hist[k]), 5, 0, 7); ctx.fill();
    });

    function recompute() { R = bpeTrain(ta.value, 80); hist = R.counts; render(); }
    function render() {
      if (!R) return;
      const n = Math.min(nm.get(), R.merges.length);
      const cur = bpeTrain(ta.value, n);
      const vocab = [...cur.base, ...cur.merges.map(m => m[0] + m[1])];
      const ids = new Map(vocab.map((v, i) => [v, i]));
      // segmentation of most frequent words
      const rows = [...cur.seqs].sort((a, b) => b.f - a.f || (a.w < b.w ? -1 : 1)).slice(0, 10);
      segs.replaceChildren(el('table', { class: 'tbl' }, el('tr', {}, el('th', { text: 'parola' }), el('th', { text: 'freq.' }), el('th', { text: 'token correnti' })),
        ...rows.map(s => el('tr', {}, el('td', { text: s.w }), el('td', { text: s.f }), el('td', { style: 'text-align:left' }, el('div', { class: 'chips' }, s.syms.map((x, i) => el('span', { class: 'chip c' + (i % 4), text: x }))))))));
      const tp = bpeTop(cur.seqs, 6), mx = tp.length ? tp[0][2] : 1;
      pairsBox.replaceChildren(el('table', { class: 'tbl' }, el('tr', {}, el('th', { text: 'prossima coppia candidata' }), el('th', { text: 'occorrenze' }), el('th', {})),
        ...tp.map((p, i) => el('tr', { class: i === 0 ? 'sel' : '' }, el('td', { text: `${p[0]} + ${p[1]}` }), el('td', { text: p[2] }), el('td', { style: 'width:90px' }, el('div', { class: 'bar', style: `width:${p[2] / mx * 100}%;background:var(--amber)` }))))));
      mergesBox.innerHTML = cur.merges.length ? cur.merges.map((m, i) => `${String(i + 1).padStart(2, ' ')}. ${m[0]} + ${m[1]} → <b>${m[0] + m[1]}</b> <span class="muted">(×${m[2]})</span>`).join('<br>') : '<span class="muted">Nessun merge: il vocabolario è fatto di soli caratteri.</span>';
      const chars = cur.seqs.reduce((a, s) => a + s.f * [...s.w].length, 0), toks = cur.counts[cur.counts.length - 1];
      stats.innerHTML = `Vocabolario: <b>${vocab.length}</b> token (${cur.base.length} caratteri + ${cur.merges.length} merge)<br>Testo: ${chars} simboli → <b>${toks}</b> token · compressione <b>${(chars / toks).toFixed(2)}×</b>`;
      // encoder
      const toksE = bpeEncode(encIn.value, cur.merges);
      encOut.replaceChildren(...toksE.map((x, i) => el('span', { class: 'chip c' + (i % 4), title: 'id ' + (ids.has(x) ? ids.get(x) : 'unk'), text: ids.has(x) ? `${x} #${ids.get(x)}` : `${x} #<unk>` })));
      const unk = toksE.filter(x => !ids.has(x)).length, nch = [...encIn.value].length;
      encStat.innerHTML = `${nch} caratteri → <b>${toksE.length}</b> token${unk ? ` · <span style="color:var(--rose)">${unk} sconosciuti (carattere mai visto: i tokenizer byte-level non hanno questo problema)</span>` : ''}`;
      chart1.redraw();
    }
    ta.addEventListener('input', recompute); encIn.addEventListener('input', render);

    root.append(lab('Laboratorio · addestra un BPE', ta, controls(nm.el, plus, reset), stats,
      el('div', { class: 'grid2' }, el('div', {}, segs), el('div', {}, pairsBox, el('div', { class: 'lab-title', style: 'margin:14px 0 6px', text: 'Merge appresi, in ordine' }), mergesBox))));
    root.append(lab('Laboratorio · usa il tokenizer appena addestrato',
      el('p', { text: 'Scrivi una frase, anche con parole mai viste: vedrai come viene spezzata. I numeri dopo # sono gli id che il modello riceverebbe.' }), encIn, encOut, encStat));
    root.append(lab('Laboratorio · rendimenti decrescenti', el('p', { text: 'Ogni merge riduce la lunghezza del testo, ma sempre meno: i primi catturano le coppie molto frequenti, gli ultimi quasi nulla. Il punto arancione è la configurazione scelta sopra.' }), cv));
    recompute();

    root.append(el('h2', { text: 'Conseguenze pratiche' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Vocabolario e costo.</b> Un vocabolario più grande aggiunge parametri (<var>V</var>·<var>d</var> nell'embedding) ma accorcia le sequenze, e il costo dell'attention scala con <var>T</var>². Gli LLM attuali usano da circa 32 mila a oltre 200 mila token.</li>
      <li><b>Byte-level.</b> GPT-2 e successori applicano il BPE ai <i>byte</i> UTF-8, non ai caratteri: i 256 byte sono sempre nel vocabolario di base, quindi qualunque stringa è codificabile.</li>
      <li><b>Stranezze spiegate.</b> Un modello fatica a contare le lettere di "strawberry" perché non vede lettere: vede due o tre pezzi opachi. Allo stesso modo l'aritmetica è più difficile quando i numeri sono spezzati in modo irregolare, e le lingue poco rappresentate nel corpus del tokenizer costano più token a parità di significato.</li>
      <li><b>Token speciali.</b> Il vocabolario contiene anche simboli di controllo (inizio/fine testo, turni di conversazione, tool). Per il modello sono token come gli altri.</li>` }));
    root.append(details('math', 'La matematica: l\'algoritmo BPE', 'algoritmo', `
      <p>Sia <var>C</var> il corpus, ridotto a un multiinsieme di parole con frequenza <var>f</var>(<var>w</var>), ognuna scritta come sequenza di simboli. Si ripete:</p>
      <ol class="prose"><li>per ogni coppia di simboli adiacenti (<var>a</var>, <var>b</var>) calcola <span class="eq" style="display:inline;padding:2px 8px"><var>n</var>(<var>a</var>,<var>b</var>) = Σ<sub><var>w</var></sub> <var>f</var>(<var>w</var>)·#(<var>a</var><var>b</var> in <var>w</var>)</span></li>
      <li>scegli (<var>a</var>*, <var>b</var>*) = argmax <var>n</var>(<var>a</var>,<var>b</var>) e aggiungi il nuovo simbolo <var>a</var>*<var>b</var>* al vocabolario;</li>
      <li>sostituisci ovunque la coppia con il nuovo simbolo.</li></ol>
      <p>Dopo <var>K</var> merge il vocabolario ha |<i>base</i>| + <var>K</var> elementi. Per <b>codificare</b> un testo nuovo si applicano gli stessi merge nello stesso ordine: l'ordine è parte del modello. L'algoritmo è greedy e deterministico; non ottimizza una funzione di costo globale.</p>
      <p>Esistono alternative (Unigram LM, WordPiece) che scelgono i pezzi con criteri probabilistici diversi, ma l'idea generale è la stessa: un dizionario di pezzi appreso dai dati.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Perché vocabolari più grandi accorciano le sequenze?', opts: ['Perché ogni token può rappresentare più caratteri', 'Perché il modello diventa più veloce', 'Perché si eliminano i token rari'], ans: 0, why: 'Più merge significano pezzi più lunghi in media, quindi meno token per lo stesso testo.' },
      { q: 'Il tokenizer viene addestrato insieme al modello con la backpropagation?', opts: ['Sì, con la stessa loss', 'No: si addestra prima, separatamente, e poi resta fisso'], ans: 1, why: 'Il BPE è un algoritmo di conteggio, non è differenziabile. Si costruisce una volta sul corpus e il modello impara embedding per quei pezzi.' },
      { q: 'Il modello riceve in ingresso…', opts: ['le lettere della parola', 'indici interi dei token', 'il significato della parola'], ans: 1, why: 'Solo interi. Il significato dovrà emergere dagli embedding appresi (modulo 2).' },
    ]));
  },
});
