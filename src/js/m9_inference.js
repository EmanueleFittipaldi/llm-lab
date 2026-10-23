'use strict';
/* Module 9 - inference: prefill vs decode, KV cache, memory, throughput, speculative decoding */

/* One attention head generating n positions. With useCache the K/V rows are computed once and appended;
   without it every step recomputes K and V (and Q) for the whole prefix. `proj` counts token-row projections. */
function kvDecode(Xs, Wq, Wk, Wv, useCache) {
  const n = Xs.length, d = Wq[0].length, outs = []; let proj = 0, Kc = [], Vc = [];
  for (let t = 0; t < n; t++) {
    if (useCache) {
      const x = [Xs[t]], q = MX.mul(x, Wq)[0];
      Kc.push(MX.mul(x, Wk)[0]); Vc.push(MX.mul(x, Wv)[0]); proj += 1;
      const a = softmax(Kc.map(k => dot(q, k) / Math.sqrt(d)));
      outs.push(range(d).map(j => sum(a.map((w, i) => w * Vc[i][j]))));
    } else {
      const r = attention(Xs.slice(0, t + 1), Wq, Wk, Wv, true); proj += t + 1;
      outs.push(r.Z[t]);
    }
  }
  return { outs, proj };
}
const kvBytesPerToken = (L, nkv, dh, bytes) => 2 * L * nkv * dh * bytes;
/* simplified roofline for one decode step with batch B: weights read once per step, each sequence reads its own KV cache */
function decodeStep(o) {
  const mem = (o.W + o.B * o.kvSeq) / o.bw, cmp = o.B * 2 * o.N / o.flops, t = Math.max(mem, cmp);
  return { t, tps: o.B / t, perUser: 1 / t, bound: mem >= cmp ? 'memoria' : 'calcolo', mem, cmp };
}
const specGain = (alpha, k) => alpha >= 1 ? k + 1 : (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
LAB.tests.infer = { kvDecode, kvBytesPerToken, decodeStep, specGain };

registerModule({
  id: 'm9', n: 9, title: 'KV cache e inferenza', eyebrow: 'far girare il modello davvero', heading: 'Inferenza: prefill, decode e KV cache',
  lede: 'Generare testo è un ciclo: un token alla volta, ogni token dipende da tutti i precedenti. Senza un trucco, il costo crescerebbe col quadrato della lunghezza. Il trucco si chiama KV cache, ed è anche il motivo per cui il contesto lungo costa memoria.',
  build(root) {
    root.append(el('div', { html: `
      <p>Durante il training il modello vede la sequenza intera e calcola tutte le posizioni <b>in parallelo</b> (la maschera causale impedisce di guardare avanti). In <b>inferenza</b> invece il token <var>t</var>+1 non esiste finché non hai estratto il token <var>t</var>: la generazione è sequenziale.</p>
      <p>Guarda la formula dell'attention: la <b>query</b> della posizione nuova si confronta con le <b>key</b> di tutte le posizioni precedenti e mescola i loro <b>value</b>. Ma key e value di una posizione passata <i>non cambiano mai</i>: dipendono solo dal suo input e dai pesi. Si possono quindi calcolare una volta sola e <b>tenere in memoria</b>. Questa è la KV cache.</p>` }));

    /* ---- Lab 1: prefill / decode, with and without cache ---- */
    const TOK = ['Il', 'gatto', 'dorme', 'sul', 'divano', 'e', 'sogna', 'un', 'grande', 'pesce', 'fresco', '.'], P = 5, N = TOK.length;
    const sStep = slider({ label: 'passo (0 = prefill)', min: 0, max: N - P, value: 2, onInput: () => { upd1(); } });
    const cv1 = canvasEl(250), out1 = el('div', { class: 'readout' }), cvC = canvasEl(200);
    const projAt = (cache, k) => { // cumulative token-row projections after step k
      let s = 0; for (let j = 0; j <= k; j++) s += cache ? (j === 0 ? P : 1) : P + j; return s;
    };
    const c1 = setupCanvas(cv1, (ctx, w, h, t) => {
      const k = sStep.get(), left = 96, cw = (w - left - 8) / N, rows = [['senza cache', false, 54], ['con KV cache', true, 142]];
      rows.forEach(([name, cache, y]) => {
        text(ctx, name, 4, y + 24, { color: t.ink, size: 12, sans: true, weight: 700 });
        for (let i = 0; i < N; i++) {
          const exists = i < P + k, x = left + i * cw;
          let state = 'none'; // none | new | cached
          if (exists) {
            const processedNow = cache ? (k === 0 ? i < P : i === P + k - 1) : true;
            state = processedNow ? 'new' : 'cached';
          }
          ctx.lineWidth = 1.5;
          if (state === 'none') { ctx.setLineDash([3, 3]); ctx.strokeStyle = t.line; ctx.strokeRect(x + 2, y, cw - 4, 38); ctx.setLineDash([]); }
          else if (state === 'new') { ctx.fillStyle = rgba(t.accent, 0.9); ctx.fillRect(x + 2, y, cw - 4, 38); }
          else { ctx.fillStyle = rgba(t.teal, 0.28); ctx.fillRect(x + 2, y, cw - 4, 38); ctx.strokeStyle = t.teal; ctx.strokeRect(x + 2, y, cw - 4, 38); }
          text(ctx, TOK[i], x + cw / 2, y + 23, { align: 'center', size: cw < 56 ? 10 : 11, color: state === 'new' ? '#fff' : (state === 'none' ? t.muted : t.ink), sans: true, weight: 600 });
          if (i < P) text(ctx, 'prompt', x + cw / 2, y - 5, { align: 'center', size: 9, color: t.amber });
        }
        text(ctx, `proiezioni K,V calcolate finora: ${projAt(cache, k)}`, left, y + 58, { color: t.muted, size: 11 });
      });
      ctx.fillStyle = rgba(t.accent, 0.9); ctx.fillRect(left, h - 22, 10, 10); text(ctx, 'calcolato in questo passo', left + 14, h - 13, { color: t.muted, size: 10 });
      ctx.fillStyle = rgba(t.teal, 0.4); ctx.fillRect(left + 168, h - 22, 10, 10); text(ctx, 'riletto dalla cache', left + 182, h - 13, { color: t.muted, size: 10 });
    });
    const cc = setupCanvas(cvC, (ctx, w, h, t) => {
      const ns = range(100).map(i => i + 1);
      chart(ctx, w, h, t, [
        { xs: ns, ys: ns.map(n => n * (n + 1) / 2), color: t.rose, lw: 2.5, label: 'senza cache: n(n+1)/2' },
        { xs: ns, ys: ns, color: t.accent, lw: 2.5, label: 'con cache: n' }], { xlabel: 'token generati n', ylabel: 'proiezioni K,V', ymin: 0, nx: 5 });
    });
    function upd1() {
      const k = sStep.get(), a = projAt(false, k), b = projAt(true, k);
      out1.innerHTML = k === 0
        ? `<b>Prefill</b>: i ${P} token del prompt sono processati insieme in un solo passaggio parallelo (alto uso di calcolo). Per ora non c'è differenza: ${a} proiezioni.`
        : `<b>Decode, passo ${k}</b>: senza cache ${a} proiezioni totali, con cache ${b} (${f2(a / b, 1)}× in meno). Ogni passo con cache aggiunge una sola riga a K e a V.`;
      c1.redraw();
    }
    root.append(lab('Laboratorio · prefill e decode, con e senza cache', controls(sStep.el), cv1, out1,
      el('p', { class: 'muted', html: 'Il numero di proiezioni è solo una parte del costo (c\'è anche l\'attention stessa, che resta lineare in <var>t</var> per ogni nuovo token), ma mostra bene la differenza: senza cache il lavoro totale cresce col <b>quadrato</b> dei token generati, con cache <b>linearmente</b>.' }),
      cvC));
    upd1();

    /* ---- Lab 2: KV memory ---- */
    const MODELS = {
      llama2_7b: { name: 'Llama-2 7B (MHA, 32 head KV)', L: 32, nkv: 32, dh: 128, N: 6.74e9 },
      llama3_8b: { name: 'Llama-3 8B (GQA, 8 head KV)', L: 32, nkv: 8, dh: 128, N: 8.03e9 },
      llama3_70b: { name: 'Llama-3 70B (GQA, 8 head KV)', L: 80, nkv: 8, dh: 128, N: 70.6e9 },
    };
    let mk = 'llama2_7b'; const DT = { fp16: 2, int8: 1, int4: 0.5 };
    const msel = select(Object.entries(MODELS).map(([k, m]) => [k, m.name]), mk, v => { mk = v; upd2(); }, 'modello');
    let dtk = 'fp16'; const dsel = select([['fp16', 'KV in fp16 (2 byte)'], ['int8', 'KV in int8 (1 byte)'], ['int4', 'KV in 4 bit (0,5 byte)']], 'fp16', v => { dtk = v; upd2(); }, 'precisione KV');
    const sCtx = slider({ label: 'contesto (token)', min: 10, max: 20, step: 0.5, value: 15, fmt: v => human(Math.pow(2, v)).replace(' ', '') , onInput: () => upd2() });
    const sB = slider({ label: 'sequenze in parallelo', min: 1, max: 64, value: 1, onInput: () => upd2() });
    const cv2 = canvasEl(250), out2 = el('div', { class: 'readout' });
    const gb = b => b / 1e9;
    const c2 = setupCanvas(cv2, (ctx, w, h, t) => {
      const B = sB.get(), by = DT[dtk], xs = range(40).map(i => Math.pow(2, 10 + i * 0.25)), cols = [t.rose, t.accent, t.teal];
      const series = Object.values(MODELS).map((m, i) => ({ xs, ys: xs.map(x => gb(kvBytesPerToken(m.L, m.nkv, m.dh, by) * x * B)), color: cols[i], lw: 2.5, label: m.name.split(' (')[0] + (m.nkv === 32 ? ' (MHA)' : ' (GQA)') }));
      series.push({ xs: [xs[0], xs[xs.length - 1]], ys: [80, 80], color: t.amber, dash: [6, 4], lw: 1.8, label: '80 GB (una GPU)' });
      chart(ctx, w, h, t, series, { xlabel: 'token di contesto', ylabel: 'KV cache (GB)', ymin: 0, ymax: 160, xmin: 0, nx: 4, xfmt: v => human(v).replace(' ', '') });
    });
    function upd2() {
      const m = MODELS[mk], by = DT[dtk], ctxN = Math.pow(2, sCtx.get()), B = sB.get(), per = kvBytesPerToken(m.L, m.nkv, m.dh, by), tot = per * ctxN * B;
      const wts = m.N * 2;
      out2.innerHTML = `per token: <b>${f2(per / 1024, 0)} KiB</b> (2 · ${m.L} layer · ${m.nkv} head KV · ${m.dh} dim · ${by} byte) · totale a ${human(ctxN)} token × ${B}: <b>${f2(gb(tot), 1)} GB</b> · pesi del modello in fp16: ${f2(gb(wts), 1)} GB → la cache pesa il <b>${f2(tot / wts * 100, 0)}%</b> dei pesi`;
      c2.redraw();
    }
    root.append(lab('Laboratorio · quanta memoria occupa la KV cache', controls(msel, dsel), controls(sCtx.el, sB.el), cv2, out2,
      el('p', { class: 'muted', html: 'Le curve usano precisione e numero di sequenze scelti sopra. Con 32 head KV (multi-head attention classica) un contesto di 128k token per una sola sequenza supera da solo una GPU da 80 GB. Per questo i modelli moderni usano <b>GQA</b> (Grouped-Query Attention): molte head di query condividono poche head di key/value, e la cache si riduce di 4× (qui 32→8).' })));
    upd2();

    /* ---- Lab 3: decode throughput ---- */
    const HW = { h100: { name: 'GPU datacenter (≈3,3 TB/s, ≈1000 TFLOPS)', bw: 3.35e12, fl: 9.9e14 }, rtx: { name: 'GPU consumer (≈1 TB/s, ≈80 TFLOPS)', bw: 1.0e12, fl: 8e13 }, lap: { name: 'Laptop (≈0,15 TB/s, ≈10 TFLOPS)', bw: 1.5e11, fl: 1e13 } };
    let hk = 'h100';
    const hsel = select(Object.entries(HW).map(([k, v]) => [k, v.name]), hk, v => { hk = v; upd3(); }, 'hardware (ordini di grandezza)');
    const sKV = slider({ label: 'contesto per sequenza', min: 10, max: 17, step: 0.5, value: 12, fmt: v => human(Math.pow(2, v)).replace(' ', ''), onInput: () => upd3() });
    const cv3 = canvasEl(250), out3 = el('div', { class: 'readout' });
    const stepO = B => { const m = MODELS[mk], hw = HW[hk]; return decodeStep({ B, N: m.N, W: m.N * 2, kvSeq: kvBytesPerToken(m.L, m.nkv, m.dh, 2) * Math.pow(2, sKV.get()), bw: hw.bw, flops: hw.fl }); };
    const c3 = setupCanvas(cv3, (ctx, w, h, t) => {
      const Bs = range(256).map(i => i + 1), r = Bs.map(stepO);
      const c = chart(ctx, w, h, t, [
        { xs: Bs, ys: r.map(x => x.tps), color: t.accent, lw: 2.5, label: 'token/s totali (somma su tutte le sequenze)' }
      ], { xlabel: 'sequenze in parallelo (batch)', ylabel: 'token/s', ymin: 0, nx: 4 });
      const ridge = Bs.findIndex(b => stepO(b).bound === 'calcolo');
      if (ridge > 0) { ctx.strokeStyle = t.rose; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(c.X(Bs[ridge]), c.y1); ctx.lineTo(c.X(Bs[ridge]), c.y0); ctx.stroke(); ctx.setLineDash([]); text(ctx, 'da qui limitati dal calcolo', c.X(Bs[ridge]) + 5, c.y0 - 8, { color: t.rose, size: 10 }); }
    });
    function upd3() {
      const one = stepO(1), r64 = stepO(64);
      out3.innerHTML = `batch 1: <b>${f2(one.tps, 0)} token/s</b> (limite: <b>${one.bound}</b>; leggere tutti i pesi richiede ${f2(one.mem * 1000, 1)} ms, il calcolo solo ${f2(one.cmp * 1000, 3)} ms) · batch 64: <b>${f2(r64.tps, 0)} token/s totali</b>, ${f2(r64.perUser, 1)} token/s a utente`;
      c3.redraw();
    }
    root.append(lab('Laboratorio · perché il decode è limitato dalla memoria', controls(hsel, sKV.el), cv3, out3,
      el('p', { class: 'muted', html: 'Modello semplificato: il tempo di un passo è il massimo tra (byte da leggere ÷ banda) e (FLOP ÷ potenza di calcolo). Usa il modello scelto sopra con pesi in fp16 e ignora overhead, comunicazione tra GPU e kernel non ottimali. Le cifre hardware sono <b>ordini di grandezza</b> da specifiche pubbliche, non misure.' })));
    upd3();

    /* ---- Lab 4: speculative decoding ---- */
    const sA = slider({ label: 'tasso di accettazione α', min: 0.2, max: 0.95, step: 0.01, value: 0.8, fmt: v => v.toFixed(2), onInput: () => upd4() });
    const sKd = slider({ label: 'token proposti k', min: 1, max: 12, value: 4, onInput: () => upd4() });
    const sC = slider({ label: 'costo draft / target', min: 0.01, max: 0.3, step: 0.01, value: 0.05, fmt: v => v.toFixed(2), onInput: () => upd4() });
    const cv4 = canvasEl(220), out4 = el('div', { class: 'readout' });
    const c4 = setupCanvas(cv4, (ctx, w, h, t) => {
      const a = sA.get(), cst = sC.get(), ks = range(12).map(i => i + 1), g = ks.map(k => specGain(a, k) / (1 + k * cst));
      const c = chart(ctx, w, h, t, [{ xs: ks, ys: g, color: t.accent, lw: 2.5, dots: true, label: 'speedup atteso' }, { xs: [1, 12], ys: [1, 1], color: t.muted, dash: [4, 4], lw: 1.5 }], { xlabel: 'token proposti k', ylabel: 'speedup ×', ymin: 0, nx: 11 });
      ctx.strokeStyle = t.rose; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(c.X(sKd.get()), c.y1); ctx.lineTo(c.X(sKd.get()), c.y0); ctx.stroke(); ctx.setLineDash([]);
    });
    function upd4() {
      const a = sA.get(), k = sKd.get(), cst = sC.get(), e = specGain(a, k);
      out4.innerHTML = `token attesi per ogni passo del modello grande: <b>${f2(e, 2)}</b> · speedup atteso: <b>${f2(e / (1 + k * cst), 2)}×</b>`;
      c4.redraw();
    }
    root.append(lab('Laboratorio · speculative decoding', controls(sA.el, sKd.el, sC.el), cv4, out4,
      el('p', { class: 'muted', html: 'Un modello piccolo ("draft") propone <var>k</var> token; il modello grande li verifica <b>tutti insieme in un solo passaggio</b> (come un prefill) e ne accetta un prefisso. Con la regola di accettazione/rifiuto corretta la distribuzione dei token prodotti è <b>esattamente</b> quella del modello grande. Dopo un certo <var>k</var> conviene smettere di proporre: più token proposti significa più probabilità che uno venga rifiutato e butti via il resto.' })));
    upd4();

    root.append(el('h2', { text: 'I punti da portarsi via' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Prefill</b>: processa tutto il prompt in parallelo; è <i>compute-bound</i> (molto calcolo per byte letto). Determina il "time to first token".</li>
      <li><b>Decode</b>: un token per passo; a ogni passo bisogna leggere tutti i pesi dalla memoria per fare pochissimo calcolo, quindi è <i>memory-bound</i>. Determina i token/s percepiti.</li>
      <li><b>Batching</b>: servire più utenti insieme ammortizza la lettura dei pesi, finché non si diventa compute-bound (o la KV cache riempie la memoria).</li>
      <li><b>KV cache</b>: toglie il lavoro ridondante ma costa memoria proporzionale a contesto × batch. Tecniche per ridurla: GQA/MQA, quantizzazione della cache, finestre scorrevoli, paging (PagedAttention).</li>
      <li><b>Speculative decoding</b>: usa un modello piccolo per proporre e uno grande per verificare, per guadagnare velocità senza cambiare l'output.</li>` }));

    root.append(details('math', 'La matematica: costo, memoria e roofline', 'formule', `
      <p><b>Equivalenza.</b> Per la posizione <var>t</var>, l'uscita dell'attention è</p>
      <span class="eq"><var>o</var><sub>t</sub> = Σ<sub>i≤t</sub> softmax<sub>i</sub>(<var>q</var><sub>t</sub>·<var>k</var><sub>i</sub> / √<var>d</var>) <var>v</var><sub>i</sub>, &nbsp; <var>k</var><sub>i</sub> = <var>x</var><sub>i</sub><var>W</var><sub>K</sub>, <var>v</var><sub>i</sub> = <var>x</var><sub>i</sub><var>W</var><sub>V</sub></span>
      <p>Poiché <var>k</var><sub>i</sub> e <var>v</var><sub>i</sub> non dipendono da <var>t</var>, ricalcolarli o rileggerli dalla cache dà lo stesso risultato (nel laboratorio questo è verificato numericamente fino all'errore di arrotondamento).</p>
      <p><b>Costo delle proiezioni.</b> Senza cache, generare <var>n</var> token richiede Σ<sub>t=1..n</sub> <var>t</var> = <var>n</var>(<var>n</var>+1)/2 proiezioni di riga, cioè O(<var>n</var><sup>2</sup>); con cache <var>n</var>.</p>
      <p><b>Memoria.</b> Per token si salvano <var>K</var> e <var>V</var> in ogni layer:</p>
      <span class="eq">byte/token = 2 · <var>L</var> · <var>n</var><sub>kv</sub> · <var>d</var><sub>head</sub> · <var>b</var></span>
      <p>Esempio Llama-2 7B in fp16: 2·32·32·128·2 = 524 288 byte = 0,5 MiB per token, quindi 4096 token ≈ 2 GiB per sequenza.</p>
      <p><b>Roofline semplificata.</b> Un passo di decode con batch <var>B</var> impiega</p>
      <span class="eq"><var>t</var><sub>step</sub> = max( ${frac('<var>W</var> + <var>B</var>·<var>S</var><sub>KV</sub>', 'BW')} , ${frac('2<var>N</var><var>B</var>', 'FLOPS')} )</span>
      <p>dove <var>W</var> sono i byte dei pesi, <var>S</var><sub>KV</sub> i byte di cache per sequenza, <var>N</var> i parametri (ogni parametro fa circa 2 FLOP per token). Il batch critico oltre cui si è compute-bound è <var>B</var>* ≈ <var>W</var>·FLOPS / (2<var>N</var>·BW) = (byte per parametro)·FLOPS/(2·BW), cioè l'<i>intensità aritmetica</i> della macchina.</p>
      <p><b>Speculative decoding.</b> Se ogni token proposto è accettato con probabilità <var>α</var> (indipendente) e se ne propongono <var>k</var>, il numero atteso di token ottenuti per passo del modello grande è</p>
      <span class="eq"><var>E</var> = Σ<sub>j=0..k</sub> <var>α</var><sup>j</sup> = ${frac('1 − <var>α</var><sup>k+1</sup>', '1 − <var>α</var>')}</span>
      <p>(il +1 è il token che il modello grande produce comunque in caso di rifiuto o dopo l'ultimo accettato). Con costo del draft <var>c</var> (relativo al target) lo speedup è <var>E</var>/(1 + <var>k c</var>). L'indipendenza di <var>α</var> è un'ipotesi: in pratica l'accettazione dipende dal contesto.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Perché con la KV cache non si salvano anche le query?', opts: ['Sono troppo grandi', 'La query di una posizione passata non serve mai più: serve solo quella del token nuovo', 'Le query cambiano a ogni passo'], ans: 1, why: 'Ogni passo calcola l\'uscita solo per la posizione nuova, che usa la sua query contro key e value di tutti i precedenti. Le query vecchie non vengono più usate.' },
      { q: 'Perché il decode di un singolo utente è tipicamente memory-bound?', opts: ['Perché la softmax è lenta', 'Perché per ogni token bisogna leggere tutti i pesi dalla memoria e farci pochissimi calcoli', 'Perché i token sono pochi'], ans: 1, why: 'Con batch 1 si fanno circa 2 FLOP per parametro letto: l\'intensità aritmetica è bassissima rispetto a quella che la GPU potrebbe sostenere.' },
      { q: 'Cosa riduce la GQA?', opts: ['Il numero di layer', 'La dimensione della KV cache, condividendo key/value tra gruppi di head di query', 'Il vocabolario'], ans: 1, why: 'Meno head KV significa meno byte per token in cache, a parità di head di query.' },
      { q: 'Il speculative decoding cambia la distribuzione dell\'output?', opts: ['Sì, è un\'approssimazione', 'No, con la regola di accettazione corretta è esatta', 'Solo a temperatura 0'], ans: 1, why: 'I token proposti vengono accettati con una probabilità che corregge la differenza tra draft e target; in caso di rifiuto si ricampiona dalla distribuzione residua.' },
    ]));
  },
});
