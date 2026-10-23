'use strict';
/* Module 0 - the pipeline and the parameter budget */
function paramCount(P) {
  const emb = P.V * P.d, pos = P.ctx * P.d, unemb = P.tie ? 0 : P.V * P.d;
  const attn = P.L * 4 * P.d * P.d, mlp = P.L * (P.glu === 'swiglu' ? 3 : 2) * P.d * P.hid, norms = P.L * 4 * P.d + 2 * P.d;
  return { emb, pos, unemb, attn, mlp, norms, N: emb + pos + unemb + attn + mlp + norms };
}
LAB.tests.paramCount = paramCount;
registerModule({
  id: 'm0', n: 0, title: 'La pipeline', eyebrow: 'panoramica', heading: 'Una funzione che prevede il token successivo',
  lede: 'Un LLM prende una sequenza di numeri interi e restituisce una distribuzione di probabilità sul prossimo intero. Tutto il resto è il modo in cui questa funzione è costruita e il modo in cui i suoi parametri vengono trovati.',
  build(root) {
    root.append(el('div', { html: `
      <p>Ogni modulo di questo laboratorio zooma su un pezzo di una catena che, vista da lontano, è sorprendentemente corta:
      il testo diventa <b>token</b> (interi), i token diventano <b>vettori</b> (embedding), una pila di identici <b>blocchi transformer</b> li rimescola,
      e alla fine un vettore per posizione viene trasformato in una <b>distribuzione di probabilità</b> sul vocabolario.
      Si sceglie un token, lo si aggiunge alla sequenza e si ripete: questo ciclo si chiama <i>generazione autoregressiva</i>.</p>
      <p>Prima di entrare nei dettagli, qui sotto trovi la mappa completa con le <b>forme dei tensori</b> in ogni punto.
      Capire le forme è metà della battaglia: quando una formula ti sembra oscura, chiediti sempre quanto è grande ogni oggetto.</p>` }));

    /* ---- Lab A: shapes ---- */
    const st = { T: 6, d: 768, L: 12, V: 50257 };
    const stages = [
      { k: 'Testo', s: () => '"il gatto dorme"', d: 'Una stringa di caratteri. Il modello non la vede mai direttamente: lavora solo con numeri.' },
      { k: 'Tokenizer', s: () => `[T] = [${st.T}]`, d: 'Spezza il testo in pezzi (<i>token</i>) presi da un vocabolario fisso e restituisce l\'indice di ogni pezzo. È un passaggio deterministico, senza apprendimento (modulo 1).' },
      { k: 'Embedding', s: () => `[T, d] = [${st.T}, ${st.d}]`, d: 'Ogni indice seleziona una riga di una matrice <var>E</var> di forma <var>V</var>×<var>d</var>. Da qui in poi ogni token è un vettore (modulo 2).' },
      { k: 'Posizioni', s: () => `[T, d] = [${st.T}, ${st.d}]`, d: 'Si inietta l\'informazione sull\'ordine, perché l\'attention da sola non distingue "gatto mangia pesce" da "pesce mangia gatto" (modulo 3). Nei modelli moderni avviene dentro l\'attention (RoPE).' },
      { k: 'Blocchi transformer', s: () => `[T, d] → [T, d]`, d: 'Ogni blocco contiene attention (i token si scambiano informazione) e MLP (ogni token elabora per conto suo), con residual e normalizzazione. <b>La forma non cambia mai</b>: è questo che permette di impilare quanti blocchi si vuole (moduli 4-6).' },
      { k: 'Logits', s: () => `[T, V] = [${st.T}, ${human(st.V)}]`, d: 'Una norma finale e poi una proiezione lineare da <var>d</var> a <var>V</var>: un punteggio (<i>logit</i>) per ogni token del vocabolario, in ogni posizione. Spesso la matrice è la stessa dell\'embedding, trasposta (<i>weight tying</i>).' },
      { k: 'Softmax', s: () => `[V] = [${human(st.V)}]`, d: 'Per generare serve solo l\'<b>ultima</b> posizione. La softmax trasforma i logit in probabilità che sommano a 1 (modulo 7).' },
      { k: 'Sampling', s: () => '1 intero', d: 'Si estrae un token (greedy, temperatura, top-k, top-p...). Viene accodato alla sequenza e il ciclo riparte con <var>T</var>+1. In training invece si confrontano le probabilità con il token vero (modulo 8).' },
    ];
    let sel = 2;
    const info = el('p', { class: 'readout' }), flow = el('div', { class: 'flow' });
    const renderFlow = () => {
      flow.replaceChildren(...stages.map((s, i) => {
        const b = el('button', { class: 'stg', type: 'button', 'aria-pressed': String(i === sel), onclick: () => { sel = i; renderFlow(); } },
          el('b', { text: i === 4 ? `Blocchi × ${st.L}` : s.k }), el('span', { text: s.s() }));
        return b;
      }));
      info.innerHTML = `<b>${stages[sel].k}</b> · ${stages[sel].d}`;
    };
    const cT = slider({ label: 'T (token)', min: 1, max: 64, value: st.T, onInput: v => { st.T = v; renderFlow(); } });
    const cd = slider({ label: 'd (modello)', min: 64, max: 4096, step: 64, value: st.d, onInput: v => { st.d = v; renderFlow(); } });
    const cL = slider({ label: 'L (layer)', min: 1, max: 96, value: st.L, onInput: v => { st.L = v; renderFlow(); } });
    const cV = slider({ label: 'V (vocab)', min: 1000, max: 200000, step: 1000, value: st.V, fmt: human, onInput: v => { st.V = v; renderFlow(); } });
    renderFlow();
    root.append(lab('Laboratorio · il viaggio di una sequenza', controls(cT.el, cd.el, cL.el, cV.el), flow, info));

    /* ---- Lab B: parameter calculator ---- */
    root.append(el('h2', { text: 'Dove vivono i parametri' }));
    root.append(el('p', { html: 'Un modello "da 7 miliardi" ha 7 miliardi di numeri da imparare. Quasi tutti stanno in <b>quattro</b> famiglie di matrici. Muovi i parametri architetturali per vedere come si ripartiscono, oppure parti da un preset.' }));
    const P = { d: 768, L: 12, h: 12, hid: 3072, V: 50257, ctx: 1024, glu: 'classic', tie: true, bytes: 2 };
    const bars = el('div', { class: 'stackbar' }), legend = el('div', { class: 'legend' }), out = el('div', { class: 'readout' });
    const sd = slider({ label: 'd_model', min: 64, max: 8192, step: 64, value: P.d, onInput: v => { P.d = v; calc(); } });
    const sL = slider({ label: 'layer', min: 1, max: 96, value: P.L, onInput: v => { P.L = v; calc(); } });
    const sh = slider({ label: 'head', min: 1, max: 64, value: P.h, onInput: v => { P.h = v; calc(); } });
    const sf = slider({ label: 'ffn hidden', min: 64, max: 32768, step: 64, value: P.hid, onInput: v => { P.hid = v; calc(); } });
    const sv = slider({ label: 'vocab', min: 1000, max: 260000, step: 1000, value: P.V, fmt: human, onInput: v => { P.V = v; calc(); } });
    const sc = slider({ label: 'posizioni apprese', min: 0, max: 8192, step: 128, value: P.ctx, onInput: v => { P.ctx = v; calc(); } });
    const gluSel = select([['classic', 'MLP classica (2 matrici)'], ['swiglu', 'SwiGLU (3 matrici)']], P.glu, v => { P.glu = v; calc(); }, 'MLP');
    const byteSel = select([['4', 'fp32'], ['2', 'fp16 / bf16'], ['1', 'int8'], ['0.5', 'int4']], '2', v => { P.bytes = +v; calc(); }, 'precisione');
    const tie = checkbox('weight tying (embedding = unembedding)', P.tie, v => { P.tie = v; calc(); });
    const presets = {
      'GPT-2 small': { d: 768, L: 12, h: 12, hid: 3072, V: 50257, ctx: 1024, glu: 'classic', tie: true },
      'GPT-2 XL': { d: 1600, L: 48, h: 25, hid: 6400, V: 50257, ctx: 1024, glu: 'classic', tie: true },
      'Stile Llama 7B': { d: 4096, L: 32, h: 32, hid: 11008, V: 32000, ctx: 0, glu: 'swiglu', tie: false },
      'Mini didattico': { d: 128, L: 4, h: 4, hid: 512, V: 8000, ctx: 256, glu: 'classic', tie: true },
    };
    const setP = p => { Object.assign(P, p); sd.set(P.d); sL.set(P.L); sh.set(P.h); sf.set(P.hid); sv.set(P.V); sc.set(P.ctx); gluSel.querySelector('select').value = P.glu; tie.querySelector('input').checked = P.tie; calc(); };
    function calc() {
      const { emb, pos, unemb, attn, mlp, norms, N } = paramCount(P);
      const parts = [['Embedding' + (P.tie ? '' : ' + unembedding'), emb + unemb + pos, 'var(--amber)'], ['Attention (Wq, Wk, Wv, Wo)', attn, 'var(--accent)'], ['MLP', mlp, 'var(--teal)'], ['Norm', norms, 'var(--rose)']];
      bars.replaceChildren(...parts.map(([n, v, c]) => el('div', { style: `width:${Math.max(0.4, v / N * 100)}%;background:${c}`, title: n })));
      legend.replaceChildren(...parts.map(([n, v, c]) => el('span', { html: `<i style="background:${c}"></i>${n}: ${human(v)} (${(v / N * 100).toFixed(1)}%)` })));
      const dh = P.d / P.h, ok = Number.isInteger(dh);
      const D = 20 * N, C = 6 * N * D;
      out.innerHTML = `Parametri totali <b>${human(N)}</b> (${N.toLocaleString('it-IT')})<br>` +
        `Dimensione per head <b>d_head = ${ok ? dh : '— (d non divisibile per h)'}</b><br>` +
        `Memoria dei soli pesi a ${P.bytes * 8} bit: <b>${(N * P.bytes / 1e9).toFixed(2)} GB</b><br>` +
        `Inferenza ≈ 2·N = <b>${human(2 * N)} FLOP</b> per token<br>` +
        `Training con D = 20·N token: ≈ 6·N·D = <b>${(C).toExponential(1)} FLOP</b>`;
    }
    const presetRow = controls(...Object.keys(presets).map(k => button(k, () => setP(presets[k]))));
    root.append(lab('Laboratorio · calcolatore di parametri', presetRow,
      controls(sd.el, sL.el, sh.el, sf.el), controls(sv.el, sc.el, gluSel, byteSel), tie, bars, legend, out));
    calc();
    root.append(el('p', { html: 'Nota che <b>il numero di head non cambia il numero di parametri</b>: cambia solo come i <var>d</var> canali vengono suddivisi (modulo 5). E che l\'MLP pesa circa il doppio dell\'attention: i "fatti" del modello vivono soprattutto lì.' }));

    root.append(details('math', 'La matematica: contare i parametri', 'formule', `
      <p>Per un transformer decoder-only con <var>L</var> layer, dimensione <var>d</var>, vocabolario <var>V</var> e MLP con dimensione nascosta <var>d</var><sub>ff</sub>:</p>
      <span class="eq"><var>N</var> ≈ <var>V</var>·<var>d</var> &nbsp;+&nbsp; <var>L</var>·( 4<var>d</var>² &nbsp;+&nbsp; 2<var>d</var>·<var>d</var><sub>ff</sub> ) &nbsp;+&nbsp; (<var>V</var>·<var>d</var> se non c'è weight tying)</span>
      <p>Il termine <b>4<var>d</var>²</b> sono le quattro matrici <var>W</var><sub>Q</sub>, <var>W</var><sub>K</sub>, <var>W</var><sub>V</sub>, <var>W</var><sub>O</sub> (ognuna <var>d</var>×<var>d</var>). Con <var>d</var><sub>ff</sub> = 4<var>d</var> l'MLP vale 8<var>d</var>², quindi ogni layer pesa circa <b>12<var>d</var>²</b>.
      Per GPT-2 small: 12·768² ≈ 7,08 M per layer, ×12 layer ≈ 85 M, più 38,6 M di embedding: circa 124 M.</p>
      <p>Per un conto più preciso mancano bias, qualche vettore di normalizzazione e, nei modelli moderni, le matrici K e V ridotte del <i>grouped-query attention</i>.</p>
      <p>Regola empirica dei costi: l'inferenza costa circa <b>2<var>N</var></b> FLOP per token (ogni parametro partecipa a una moltiplicazione e a una somma), il training circa <b>6<var>N</var></b> FLOP per token (forward + backward).</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Quale parte della pipeline <b>non</b> cambia la forma <var>[T, d]</var> dell\'input?', opts: ['Il blocco transformer (attention + MLP)', 'La proiezione finale sui logit', 'Il tokenizer'], ans: 0, why: 'Il blocco mappa [T, d] in [T, d], per questo si possono impilare L blocchi. La proiezione finale porta a [T, V]; il tokenizer produce [T] interi.' },
      { q: 'Raddoppiando <var>d</var> e lasciando il resto invariato, i parametri dei blocchi circa…', opts: ['raddoppiano', 'quadruplicano', 'restano uguali'], ans: 1, why: 'I blocchi sono dominati da matrici d×d e d×d_ff (con d_ff proporzionale a d): crescono come d².' },
      { q: 'In generazione, quale posizione dei logit serve per scegliere il token successivo?', opts: ['Tutte', 'Solo l\'ultima', 'Solo la prima'], ans: 1, why: 'Le altre posizioni servono in training (ognuna predice il proprio successivo, in parallelo). In generazione conta solo l\'ultima.' },
    ]));
  },
});
