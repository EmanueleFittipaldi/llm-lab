'use strict';
/* Module 15 - RAG, chunking, tool use and the agent loop. Everything runs for real in the page except the
   "planner" of the agent, which is a hand-written rule set standing in for the LLM. */

/* a fictional knowledge base (a bike-sharing app) so that the model could not already know the answers */
const KB = [
  "Per sbloccare una bici apri l'app, inquadra il codice QR sul manubrio e attendi il suono di conferma. Il lucchetto si apre in pochi secondi.",
  "Una corsa costa 1,50 € ogni 30 minuti. Dopo la prima mezz'ora si paga 1,50 € per ogni mezz'ora successiva.",
  "L'abbonamento annuale costa 36 € e comprende tutte le corse fino a 45 minuti. Si rinnova automaticamente ogni anno.",
  "Per disdire l'abbonamento vai nel menu Profilo, scegli Gestisci abbonamento e premi Annulla rinnovo. La disdetta vale dal ciclo successivo.",
  "Se la bici ha un guasto, premi Segnala problema nell'app e scegli il tipo di guasto: freni, catena, ruote o altro. La corsa in corso viene chiusa gratuitamente.",
  "Le bici si restituiscono in qualunque stazione con almeno uno stallo libero. Spingi la bici nello stallo finché senti il clic e attendi la conferma sull'app.",
  "Se non trovi stalli liberi, l'app indica la stazione più vicina e concede 10 minuti gratuiti per raggiungerla.",
  "Le biciclette elettriche sono contrassegnate da un adesivo verde e costano 3 € ogni 30 minuti. La batteria garantisce circa 60 chilometri di autonomia.",
  "Il servizio è attivo tutti i giorni dalle 6:00 alle 24:00. Di notte le stazioni restano aperte ma le bici non si sbloccano.",
  "In caso di furto o smarrimento di una bici durante la corsa, chiama subito il numero verde dell'assistenza e presenta denuncia entro 48 ore.",
  "Per ricaricare il credito usa carta o portafoglio digitale dal menu Pagamenti. Il credito minimo per sbloccare una bici è di 5 €.",
  "I minori di 14 anni non possono usare il servizio. Dai 14 ai 17 anni serve l'autorizzazione di un genitore caricata nell'app.",
  "Il casco non è fornito ma è consigliato. È vietato trasportare passeggeri sulla bici, anche bambini.",
  "Il numero verde dell'assistenza è attivo dalle 8:00 alle 20:00. Fuori orario puoi scrivere dalla sezione Aiuto dell'app.",
];
/* labelled queries: [text, index of the passage that answers it, a phrase that must survive chunking] */
const QUERIES = [
  ['Come sblocco la bici?', 0, 'codice QR'], ['Quanto costa una corsa?', 1, '1,50 € ogni 30 minuti'], ['Come disdico l\'abbonamento?', 3, 'Annulla rinnovo'],
  ['Cosa faccio se i freni non funzionano?', 4, 'Segnala problema'], ['Dove restituisco la bici?', 5, 'uno stallo libero'], ['Quanto dura la batteria delle bici elettriche?', 7, '60 chilometri'],
  ['A che ora chiude il servizio?', 8, 'dalle 6:00 alle 24:00'], ['Posso usare il servizio a 16 anni?', 11, 'autorizzazione di un genitore'],
];
const HARD_QUERY = 'Come faccio a smettere di pagare ogni anno?';
const STOPW = new Set('il lo la i gli le un uno una di a da in con su per tra fra e o ma se che non è sono si l d c all dell dal nel sul del della dei delle al alla alle nell sull ogni anche come cosa dove quanto quando posso faccio fare ha ho più'.split(' '));
/* crude Italian "stemming": keep the first 5 letters. Good enough to match abbonamento/abbonamenti, restituisco/restituiscono */
const ragTokens = s => s.toLowerCase().split(/[^a-zàèéìòù0-9]+/).filter(t => t && !STOPW.has(t)).map(t => t.length > 5 ? t.slice(0, 5) : t);
function ragStats(docs) {
  const D = docs.map(ragTokens), N = D.length, avg = D.reduce((a, d) => a + d.length, 0) / N, df = {};
  D.forEach(d => new Set(d).forEach(t => { df[t] = (df[t] || 0) + 1; }));
  return { D, N, avg, df };
}
function bm25Scores(docs, q, k1 = 1.5, b = 0.75) {
  const { D, N, avg, df } = ragStats(docs), qt = [...new Set(ragTokens(q))];
  return D.map(d => { const tf = {}; d.forEach(t => { tf[t] = (tf[t] || 0) + 1; }); return qt.reduce((s, t) => { if (!tf[t]) return s; const idf = Math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)); return s + idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * d.length / avg)); }, 0); });
}
function tfidfScores(docs, q) {
  const { D, N, df } = ragStats(docs), idf = t => Math.log(N / (df[t] || 1)) + 1, vec = ts => { const v = {}; ts.forEach(t => { v[t] = (v[t] || 0) + idf(t); }); return v; };
  const qv = vec(ragTokens(q)), qn = Math.sqrt(sum(Object.values(qv).map(x => x * x))) || 1;
  return D.map(d => { const dv = vec(d), dn = Math.sqrt(sum(Object.values(dv).map(x => x * x))) || 1; return sum(Object.keys(qv).map(t => (dv[t] || 0) * qv[t])) / (dn * qn); });
}
const ragRank = s => s.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
function ragChunks(text, size, overlap) { const out = []; for (let s = 0; s < text.length; s += Math.max(1, size - overlap)) { out.push([s, text.slice(s, s + size)]); if (s + size >= text.length) break; } return out; }
const ragRecall = (fn, k) => QUERIES.filter(([q, g]) => ragRank(fn(KB, q)).slice(0, k).some(x => x[1] === g)).length / QUERIES.length;

/* a safe arithmetic evaluator (recursive descent; no eval). Accepts decimal comma or point. */
function safeCalc(expr) {
  const s = expr.replace(/,/g, '.').replace(/[x×]/g, '*').replace(/÷/g, '/').replace(/\s+/g, ''); let i = 0;
  const peek = () => s[i], num = () => { const m = /^\d+(\.\d+)?/.exec(s.slice(i)); if (!m) throw new Error('numero atteso in posizione ' + i); i += m[0].length; return parseFloat(m[0]); };
  function atom() { if (peek() === '(') { i++; const v = add(); if (peek() !== ')') throw new Error('manca )'); i++; return v; } if (peek() === '-') { i++; return -atom(); } return num(); }
  function mul() { let v = atom(); while (peek() === '*' || peek() === '/') { const o = s[i++], r = atom(); if (o === '/' && r === 0) throw new Error('divisione per zero'); v = o === '*' ? v * r : v / r; } return v; }
  function add() { let v = mul(); while (peek() === '+' || peek() === '-') { const o = s[i++], r = mul(); v = o === '+' ? v + r : v - r; } return v; }
  const v = add(); if (i !== s.length) throw new Error('carattere inatteso "' + s[i] + '"'); return v;
}
/* the agent: a loop of (thought, action, observation) until it can answer. The planner below is a set of rules, NOT an LLM. */
function agentRun(question) {
  const steps = [], tool = { search: q => { const r = ragRank(bm25Scores(KB, q)); return { top: r[0], text: KB[r[0][1]], idx: r[0][1] }; }, calc: e => safeCalc(e) };
  const q = question.trim(), norm = q.replace(/^(quanto fa|calcola|quanto fa)\s*/i, '').replace(/[?]/g, '').trim();
  if (/^[\d\s+\-*/().,x×÷]+$/.test(norm) && /\d/.test(norm)) {
    steps.push({ thought: 'La domanda è un\'espressione aritmetica: serve la calcolatrice, non la ricerca.', action: ['calc', norm] });
    try { const v = tool.calc(norm); steps[0].obs = String(+v.toFixed(6)); steps.push({ answer: `Il risultato è ${+v.toFixed(6)}.` }); } catch (e) { steps[0].obs = 'errore: ' + e.message; steps.push({ answer: 'Non riesco a calcolarlo: ' + e.message + '.' }); }
    return steps;
  }
  const nRides = /(\d+)\s+(corse|corsa)/i.exec(q), mins = /(\d+)\s*minuti/i.exec(q), ele = /elettric/i.test(q);
  if (/(quanto)\s+(pago|costano|spendo)/i.test(q) && nRides) {
    const sq = ele ? 'quanto costa una corsa con bici elettrica' : 'quanto costa una corsa';
    steps.push({ thought: 'Mi serve il prezzo: lo cerco nella knowledge base, poi faccio il conto.', action: ['search', sq] });
    const r = tool.search(sq); steps[0].obs = `[${r.idx + 1}] ${r.text}`;
    const m = /(\d+(?:,\d+)?)\s*€ ogni (\d+) minuti/.exec(r.text);
    if (!m) { steps.push({ answer: 'Non ho trovato il prezzo nei documenti.' }); return steps; }
    const price = parseFloat(m[1].replace(',', '.')), block = +m[2], blocks = mins ? Math.ceil(+mins[1] / block) : 1, expr = `${nRides[1]} * ${blocks} * ${price}`;
    steps.push({ thought: `Il documento dice ${m[1]} € ogni ${block} minuti. ${mins ? `Una corsa da ${mins[1]} minuti vale ${blocks} blocchi.` : 'Assumo un blocco per corsa.'} Calcolo ${expr}.`, action: ['calc', expr] });
    const v = tool.calc(expr); steps[1].obs = String(+v.toFixed(2));
    steps.push({ answer: `${nRides[1]} corse${mins ? ' da ' + mins[1] + ' minuti' : ''}${ele ? ' in bici elettrica' : ''}: ${(+v.toFixed(2)).toString().replace('.', ',')} €.` });
    return steps;
  }
  steps.push({ thought: 'È una domanda sul servizio: cerco nella knowledge base.', action: ['search', q] });
  const r = tool.search(q); steps[0].obs = `[${r.idx + 1}] ${r.text} (punteggio ${f2(r.top[0], 2)})`;
  if (r.top[0] < 1.5) steps.push({ thought: 'Il punteggio è troppo basso: i documenti non parlano di questo.', answer: 'Non ho trovato nulla di pertinente nei documenti, quindi non rispondo.' });
  else steps.push({ thought: 'Il passaggio trovato risponde alla domanda.', answer: r.text.split('. ')[0].replace(/\.?$/, '.') });
  return steps;
}
LAB.tests.rag = { KB, QUERIES, HARD_QUERY, ragTokens, bm25Scores, tfidfScores, ragRank, ragChunks, ragRecall, safeCalc, agentRun };

registerModule({
  id: 'm15', n: 15, title: 'RAG, tool e agenti', eyebrow: 'dare al modello occhi e mani', heading: 'RAG, tool use e agenti',
  lede: 'Un LLM sa solo ciò che ha nei pesi e ciò che gli metti nel contesto. Il RAG gli mette nel contesto i documenti giusti; il tool use gli permette di chiamare programmi (una calcolatrice, una ricerca); un agente ripete il ciclo finché il lavoro è finito.',
  build(root) {
    root.append(el('div', { html: `
      <p>I limiti sono due: la conoscenza nei pesi è <b>congelata</b> alla data del training e non contiene i tuoi documenti privati; e il modello <b>calcola male</b> e non può agire sul mondo. Entrambi si aggirano <i>fuori</i> dal modello:</p>
      <p><b>RAG</b> (<i>retrieval-augmented generation</i>): data una domanda, si <b>cerca</b> in una base di documenti, si prendono i passaggi migliori e li si incolla nel prompt prima della domanda. Il modello risponde leggendo quei passaggi. Qui sotto una piccola base di conoscenza (14 paragrafi di un servizio di bike sharing <i>inventato</i>, così nessun modello li conosce già) e un motore di ricerca vero scritto nella pagina.</p>` }));

    /* ---- Lab 1: retrieval ---- */
    let method = 'bm25', qtext = QUERIES[2][0];
    const qin = el('input', { type: 'text', class: 'txt', value: qtext, 'aria-label': 'domanda', style: 'min-width:min(100%,360px);font:500 14px var(--mono);padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)' });
    const qsel = select([...QUERIES.map(q => [q[0], q[0]]), [HARD_QUERY, HARD_QUERY + '  (difficile)']], qtext, v => { qin.value = v; qtext = v; retr(); }, 'esempi');
    const msel = select([['bm25', 'BM25 (parole chiave, con normalizzazione)'], ['tfidf', 'TF-IDF + coseno']], method, v => { method = v; retr(); }, 'metodo');
    const list = el('div', { class: 'passages' }), promptBox = el('div', { class: 'readout', style: 'white-space:pre-wrap;border:1px dashed var(--line);border-radius:8px;padding:10px 12px;background:var(--surface)' }), evalOut = el('div', { class: 'readout' });
    const scoreFn = () => method === 'bm25' ? bm25Scores : tfidfScores;
    function retr() {
      qtext = qin.value; const sc = scoreFn()(KB, qtext), r = ragRank(sc), mx = r[0][0] || 1, gold = (QUERIES.find(q => q[0] === qtext) || [])[1];
      list.innerHTML = '';
      r.slice(0, 5).forEach(([s, i], k) => list.append(el('div', { style: `display:flex;gap:10px;align-items:flex-start;padding:6px 0;border-top:1px solid var(--line);${k < 3 ? '' : 'opacity:0.6'}` },
        el('div', { style: 'flex:none;width:120px' }, el('div', { class: 'readout', text: `#${k + 1} · par. ${i + 1}${gold === i ? ' ✓' : ''}` }), el('div', { style: `height:6px;border-radius:3px;margin-top:4px;background:var(--accent);width:${Math.max(2, s / mx * 100)}%` }), el('div', { class: 'readout', style: 'font-size:11px;color:var(--muted)', text: f2(s, 3) })),
        el('div', { style: 'font-size:0.93em', text: KB[i] }))));
      const ctxs = r.slice(0, 3).filter(x => x[0] > 0).map(x => '[' + (x[1] + 1) + '] ' + KB[x[1]]);
      promptBox.textContent = `Rispondi usando solo i documenti qui sotto. Se non bastano, dì che non lo sai.\n\n${ctxs.join('\n') || '(nessun documento trovato)'}\n\nDomanda: ${qtext}\nRisposta:`;
      const words = promptBox.textContent.split(/\s+/).length;
      evalOut.innerHTML = `prompt assemblato: ~<b>${Math.round(words * 1.5)}</b> token (stima) · sui ${QUERIES.length} esempi etichettati questo metodo trova il paragrafo giusto al primo posto in <b>${Math.round(ragRecall(scoreFn(), 1) * 8)}/8</b> casi, tra i primi 3 in <b>${Math.round(ragRecall(scoreFn(), 3) * 8)}/8</b>`;
    }
    qin.addEventListener('input', () => retr());
    root.append(lab('Laboratorio · retrieval (BM25 e TF-IDF veri, su 14 paragrafi)', controls(qsel, msel), el('div', { class: 'controls' }, qin), list, el('div', { class: 'lab-title', text: 'Il prompt che arriva all\'LLM (top-3 paragrafi)' }), promptBox, evalOut,
      el('p', { class: 'muted', html: 'Scrivi una tua domanda o scegli un esempio. Prova la domanda <b>difficile</b>: "smettere di pagare" e "disdire l\'abbonamento" significano la stessa cosa ma non hanno parole in comune, e la ricerca per parole chiave la sbaglia. Questo è il caso in cui servono i <b>modelli di embedding</b> (modulo 2): trasformano testi con significato simile in vettori vicini, anche senza parole condivise. Nei sistemi veri si combinano i due (ricerca ibrida).' })));
    retr();

    /* ---- Lab 2: chunking ---- */
    const DOC = KB.join(' ');
    const sSize = slider({ label: 'dimensione chunk (caratteri)', min: 60, max: 1200, step: 20, value: 200, onInput: () => chunkUpd() });
    const sOv = slider({ label: 'sovrapposizione', min: 0, max: 200, step: 20, value: 40, onInput: () => chunkUpd() });
    const cvC = canvasEl(230), outC = el('div', { class: 'readout' });
    const sweep = ov => { const sizes = range(30).map(i => 60 + i * 40); return { sizes, hit: sizes.map(sz => { const C = ragChunks(DOC, sz, Math.min(ov, sz - 20)).map(c => c[1]); return QUERIES.filter(([q, , key]) => { const s = bm25Scores(C, q); return C[s.indexOf(Math.max(...s))].includes(key); }).length / QUERIES.length; }), sent: sizes.map(sz => Math.min(1, sz / DOC.length)) }; };
    const cc = setupCanvas(cvC, (ctx, w, h, t) => {
      const sw = sweep(sOv.get()), c = chart(ctx, w, h, t, [{ xs: sw.sizes, ys: sw.hit, color: t.accent, lw: 2.5, label: 'risposte trovate nel chunk al primo posto' }, { xs: sw.sizes, ys: sw.sent, color: t.amber, lw: 2, dash: [5, 4], label: 'frazione del documento inviata al modello' }], { xlabel: 'dimensione del chunk (caratteri)', ymin: 0, ymax: 1.02, nx: 5 });
      ctx.strokeStyle = t.rose; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(c.X(sSize.get()), c.y1); ctx.lineTo(c.X(sSize.get()), c.y0); ctx.stroke(); ctx.setLineDash([]);
    });
    function chunkUpd() {
      const sz = sSize.get(), ov = Math.min(sOv.get(), sz - 20), C = ragChunks(DOC, sz, ov).map(c => c[1]);
      const hit = QUERIES.filter(([q, , key]) => { const s = bm25Scores(C, q); return C[s.indexOf(Math.max(...s))].includes(key); }).length, intact = QUERIES.filter(([, , key]) => C.some(c => c.includes(key))).length;
      outC.innerHTML = `<b>${C.length}</b> chunk da ~${sz} caratteri (sovrapposizione ${ov}) · la frase-chiave sopravvive intera in almeno un chunk per <b>${intact}/8</b> domande · il chunk al primo posto contiene la risposta per <b>${hit}/8</b> · il modello riceve ~${sz} caratteri (il ${f2(Math.min(1, sz / DOC.length) * 100, 0)}% dell'intero documento)`;
      cc.redraw();
    }
    root.append(lab('Laboratorio · chunking: a che dimensione si tagliano i documenti',
      el('p', { html: 'In pratica i documenti sono lunghi e si <b>spezzano in chunk</b> prima di indicizzarli. Qui l\'intera base di 14 paragrafi è un unico testo da tagliare in finestre di dimensione fissa. Misura: per ognuna delle 8 domande, il chunk migliore secondo BM25 contiene la frase che dà la risposta?' }),
      controls(sSize.el, sOv.el), cvC, outC,
      el('p', { class: 'muted', html: 'Chunk piccoli tagliano le frasi a metà (la risposta si spezza) e perdono il contesto; chunk grandi trovano più facilmente la risposta ma ne mandano molta di più al modello (costo, rumore, e il modello si distrae). La curva è ruvida perché sono solo 8 domande: è un esempio, non una legge. Nella pratica si taglia su confini naturali (paragrafi, titoli) invece che a caratteri fissi.' })));
    chunkUpd();

    /* ---- Lab 3: agent loop ---- */
    const aIn = el('input', { type: 'text', value: 'Quanto pago per 3 corse da 60 minuti con bici elettrica?', 'aria-label': 'domanda per l\'agente', style: 'min-width:min(100%,380px);font:500 14px var(--mono);padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink)' });
    const trace = el('div', { class: 'trace' });
    const EX = [['Quanto fa (17 + 5) * 3?', 'calcolo semplice'], ["Come disdico l'abbonamento?", 'domanda sui documenti'], ['Quanto pago per 4 corse da 30 minuti?', 'ricerca + calcolo'], ['Quanto pago per 3 corse da 60 minuti con bici elettrica?', 'ricerca + calcolo (bici elettrica)'], ['Come sblocco la bici?', 'la ricerca sbaglia paragrafo'], ['Chi ha vinto il campionato di calcio?', 'fuori dai documenti']];
    function runAgent() {
      const st = agentRun(aIn.value); trace.innerHTML = '';
      st.forEach((s, i) => {
        const row = (k, v, col) => el('div', { style: 'display:flex;gap:10px;padding:3px 0' }, el('b', { style: `flex:none;width:92px;color:${col};font:600 11px var(--display);text-transform:uppercase;letter-spacing:0.08em;padding-top:2px`, text: k }), el('div', { class: 'readout', style: 'font-weight:400', text: v }));
        const box = el('div', { style: 'border-left:3px solid var(--line);padding:2px 0 2px 10px;margin:6px 0' });
        if (s.thought) box.append(row('pensiero', s.thought, 'var(--muted)'));
        if (s.action) box.append(row('azione', `${s.action[0]}(${JSON.stringify(s.action[1])})`, 'var(--accent)'));
        if (s.obs !== undefined) box.append(row('osservazione', s.obs, 'var(--teal)'));
        if (s.answer) box.append(row('risposta', s.answer, 'var(--amber)'));
        trace.append(el('div', { class: 'readout', style: 'font-size:11px;color:var(--muted)', text: s.answer ? 'passo finale' : 'passo ' + (i + 1) }), box);
      });
    }
    aIn.addEventListener('keydown', e => { if (e.key === 'Enter') runAgent(); });
    root.append(lab('Laboratorio · un agente con due tool (ricerca nei documenti e calcolatrice)',
      el('p', { html: 'Il ciclo è quello di un agente vero: <b>pensiero → azione (chiamata a un tool) → osservazione → …</b> fino a una risposta. La ricerca è il BM25 di sopra; la calcolatrice è un parser aritmetico vero (niente <code>eval</code>). Ma attenzione: il "cervello" che sceglie azioni e argomenti è qui un <b>insieme di regole scritte a mano</b>, non un LLM. In un agente vero è il modello a produrre, per esempio in JSON, il nome del tool e i suoi argomenti (<i>function calling</i>); la struttura del ciclo è identica.' }),
      controls(select(EX, EX[3][0], v => { aIn.value = v; runAgent(); }, 'esempi')), el('div', { class: 'controls' }, aIn, button('Esegui', runAgent, 'primary')), trace,
      el('p', { class: 'muted', html: 'Prova una domanda fuori dai documenti: l\'agente si rifiuta invece di inventare. Prova anche "Come sblocco la bici?": la ricerca mette davanti il paragrafo sbagliato (ne parla al secondo posto) e l\'agente risponde male, perché si fida di ciò che trova. Il rifiuto qui è una soglia sul punteggio di ricerca; in un LLM vero dipende dal prompt (per esempio "se i documenti non bastano, dì che non lo sai") e non è mai garantito.' })));
    runAgent();

    root.append(el('h2', { text: 'Dove si rompono questi sistemi' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>Retrieval sbagliato</b>: se il passaggio giusto non arriva nel prompt, il modello non può usarlo, e può inventare una risposta plausibile. Il RAG si valuta in due parti: il retrieval (recall@k, come sopra) e la risposta.</li>
      <li><b>Contesto lungo non è memoria perfetta</b>: più testo si inserisce, più il modello può ignorare dettagli, soprattutto a metà del prompt.</li>
      <li><b>Prompt injection</b>: un documento recuperato o la risposta di un tool può contenere istruzioni ("ignora le regole precedenti…") che il modello legge come se venissero dall'utente. Nei sistemi con tool ad alto impatto (email, pagamenti, file) il testo non fidato va trattato come <i>dati</i>, con permessi minimi e conferma umana per le azioni irreversibili.</li>
      <li><b>Errori che si accumulano</b>: ogni passo di un agente può sbagliare, e i passi dipendono dai precedenti. Servono limiti di passi, controlli sull'esito dei tool e log leggibili.</li>
      <li><b>Costo e latenza</b>: ogni giro del ciclo è una chiamata al modello con tutto il contesto accumulato (qui la KV cache del modulo 9 conta molto).</li>` }));

    root.append(details('math', 'La matematica: TF-IDF, BM25 e coseno', 'formule', `
      <p><b>IDF.</b> Un termine raro è più informativo di uno comune. Con <var>N</var> documenti e <var>n</var><sub>t</sub> che contengono il termine <var>t</var>: idf(<var>t</var>) = ln(1 + (<var>N</var> − <var>n</var><sub>t</sub> + 0,5)/(<var>n</var><sub>t</sub> + 0,5)) (versione usata da BM25).</p>
      <p><b>BM25.</b> Per un documento <var>d</var> di lunghezza |<var>d</var>| e una query <var>q</var>:</p>
      <span class="eq">score(<var>d</var>, <var>q</var>) = Σ<sub><var>t</var>∈<var>q</var></sub> idf(<var>t</var>) · ${frac('<var>f</var><sub>t,d</sub> (<var>k</var><sub>1</sub> + 1)', '<var>f</var><sub>t,d</sub> + <var>k</var><sub>1</sub>(1 − <var>b</var> + <var>b</var> |<var>d</var>|/avgdl)')}</span>
      <p>La frequenza <var>f</var><sub>t,d</sub> ha rendimenti decrescenti (saturazione controllata da <var>k</var><sub>1</sub>) e i documenti lunghi sono penalizzati (<var>b</var>): una parola ripetuta 10 volte non vale 10 volte una sola.</p>
      <p><b>TF-IDF + coseno.</b> Ogni testo diventa un vettore sparso con pesi tf·idf; la similarità è il coseno (modulo 2): cos(<var>a</var>, <var>b</var>) = <var>a</var>·<var>b</var> / (‖<var>a</var>‖‖<var>b</var>‖). Il coseno normalizza per la lunghezza del vettore, quindi un testo corto e mirato può battere uno lungo.</p>
      <p><b>Retrieval denso.</b> Al posto dei conteggi di parole si usa un encoder che mappa testo → vettore di <var>d</var> dimensioni addestrato perché testi correlati abbiano coseno alto. La ricerca è un <i>nearest neighbor</i> su milioni di vettori, per cui si usano indici approssimati.</p>
      <p><b>Calcolatrice.</b> Il parser è a discesa ricorsiva con la grammatica <i>somma → prodotto (± prodotto)*</i>, <i>prodotto → atomo (× o ÷ atomo)*</i>, <i>atomo → numero | (somma) | −atomo</i>: la precedenza degli operatori sta nella struttura della grammatica.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Perché il RAG non richiede di riaddestrare il modello?', opts: ['Perché modifica i pesi in silenzio', 'Perché le informazioni nuove entrano nel contesto (prompt), non nei pesi', 'Perché usa un modello più grande'], ans: 1, why: 'Il modello legge i passaggi recuperati come qualunque altro testo del prompt; aggiornare i documenti basta ad aggiornare le risposte.' },
      { q: 'Una query e un documento dicono la stessa cosa con parole diverse. Quale approccio ha più probabilità di trovarlo?', opts: ['BM25', 'Retrieval con embedding', 'TF-IDF'], ans: 1, why: 'BM25 e TF-IDF confrontano parole; gli embedding confrontano significati.' },
      { q: 'Perché una calcolatrice come tool è meglio che far fare il conto al modello?', opts: ['È più economica', 'Il modello genera la risposta un token alla volta e sulle operazioni lunghe sbaglia; un programma è esatto', 'Perché il modello non sa leggere i numeri'], ans: 1, why: 'Un LLM approssima aritmetica con pattern statistici; un tool deterministico dà il risultato corretto, e al modello resta il compito di decidere quando usarlo.' },
      { q: 'Cosa rende pericolosa la prompt injection in un agente con tool?', opts: ['Rallenta il modello', 'Un testo non fidato può convincere il modello a eseguire azioni con i permessi dell\'agente', 'Aumenta il vocabolario'], ans: 1, why: 'Il modello non distingue in modo affidabile istruzioni e dati: serve limitare i permessi e chiedere conferma per le azioni importanti.' },
    ]));
  },
});
