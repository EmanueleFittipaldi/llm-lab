'use strict';
/* Module 10 - post-training: SFT loss masking, reward model (Bradley-Terry), KL-regularised policy, DPO */
const sigmoid = x => 1 / (1 + Math.exp(-x));
/* 12 toy candidate answers to one prompt. Hand-defined features: q = true quality, len = length (0..1), refLogit = base-model preference */
const CANDS = [
  { n: 'sbagliata, secca', q: 0.20, len: 0.10, z: 1.2 }, { n: 'vaga, breve', q: 0.35, len: 0.20, z: 1.0 },
  { n: 'ok, breve', q: 0.60, len: 0.30, z: 1.4 }, { n: 'corretta, concisa', q: 0.75, len: 0.35, z: 1.5 },
  { n: 'ottima, media', q: 0.90, len: 0.50, z: 0.8 }, { n: 'ottima, dettagliata', q: 0.85, len: 0.70, z: 0.5 },
  { n: 'buona, lunga', q: 0.70, len: 0.60, z: 0.9 }, { n: 'prolissa', q: 0.50, len: 0.80, z: 0.4 },
  { n: 'molto prolissa', q: 0.45, len: 0.95, z: 0.2 }, { n: 'divagante', q: 0.30, len: 0.90, z: 0.3 },
  { n: 'riempitivo enorme', q: 0.55, len: 1.00, z: 0.1 }, { n: 'fuori tema, media', q: 0.15, len: 0.50, z: 0.7 },
];
const REF = softmax(CANDS.map(c => c.z));
/* labelers prefer a over b with probability sigmoid(4 (u_a - u_b)), u = q + bias * len: annotators like long answers */
function makePairs(bias, N, seed) {
  const r = rng(seed), out = [];
  while (out.length < N) {
    const i = Math.floor(r() * CANDS.length), j = Math.floor(r() * CANDS.length); if (i === j) continue;
    const ui = CANDS[i].q + bias * CANDS[i].len, uj = CANDS[j].q + bias * CANDS[j].len;
    out.push(r() < sigmoid(4 * (ui - uj)) ? [i, j] : [j, i]); // [winner, loser]
  }
  return out;
}
/* reward model r(y) = w . (q_hat, len): here the "features" are given; in reality they are learned from text */
const feat = c => [c.q, c.len];
function btLoss(w, P) { return P.reduce((s, [a, b]) => { const m = dot(w, feat(CANDS[a])) - dot(w, feat(CANDS[b])); return s - Math.log(sigmoid(m)); }, 0) / P.length; }
function btGrad(w, P) {
  const g = [0, 0];
  P.forEach(([a, b]) => { const fa = feat(CANDS[a]), fb = feat(CANDS[b]), m = dot(w, fa) - dot(w, fb), c = -sigmoid(-m); g[0] += c * (fa[0] - fb[0]); g[1] += c * (fa[1] - fb[1]); });
  return g.map(x => x / P.length);
}
function fitRM(P, steps = 800, lr = 0.5) { let w = [0, 0]; for (let s = 0; s < steps; s++) { const g = btGrad(w, P); w = [w[0] - lr * g[0], w[1] - lr * g[1]]; } return w; }
const rewards = w => CANDS.map(c => dot(w, feat(c)));
const klDiv = (p, q) => sum(p.map((x, i) => x > 0 ? x * Math.log(x / q[i]) : 0));
/* closed-form optimum of  max E_pi[r] - beta KL(pi || ref):  pi* ∝ ref · exp(r / beta) */
const tilt = (r, beta) => softmax(CANDS.map((c, i) => c.z + r[i] / beta));
const trueQ = p => sum(p.map((x, i) => x * CANDS[i].q));
/* DPO on free logits theta (tabular policy): loss = -log sigmoid(beta [(log pi_w - log ref_w) - (log pi_l - log ref_l)]) */
function dpoLoss(th, P, beta) {
  const lp = softmax(th).map(Math.log), lr = REF.map(Math.log);
  return P.reduce((s, [a, b]) => s - Math.log(sigmoid(beta * ((lp[a] - lr[a]) - (lp[b] - lr[b])))), 0) / P.length;
}
function dpoGrad(th, P, beta) { // the softmax normaliser cancels in the difference, so d loss / d theta_j = -beta sigmoid(-m) (1[j=w] - 1[j=l])
  const lp = softmax(th).map(Math.log), lr = REF.map(Math.log), g = new Array(th.length).fill(0);
  P.forEach(([a, b]) => { const m = beta * ((lp[a] - lr[a]) - (lp[b] - lr[b])), c = -sigmoid(-m) * beta; g[a] += c; g[b] -= c; });
  return g.map(x => x / P.length);
}
function dpoTrain(P, beta, steps, lr = 0.9, every = 10) {
  let th = CANDS.map(c => c.z); const snaps = [th.slice()];
  for (let s = 1; s <= steps; s++) { const g = dpoGrad(th, P, beta); th = th.map((t, i) => t - lr * g[i]); if (s % every === 0) snaps.push(th.slice()); }
  return snaps;
}
LAB.tests.post = { CANDS, REF, makePairs, btLoss, btGrad, fitRM, rewards, klDiv, tilt, trueQ, dpoLoss, dpoGrad, dpoTrain, sigmoid };

registerModule({
  id: 'm10', n: 10, title: 'SFT, RLHF e DPO', eyebrow: 'dopo il pre-training', heading: 'Da modello di testo a assistente',
  lede: 'Dopo il pre-training il modello sa continuare un testo, non sa ancora "rispondere bene". Il post-training lo plasma in tre mosse: imitare esempi buoni (SFT), imparare cosa preferiscono le persone (reward model), ottimizzare verso quelle preferenze senza allontanarsi troppo dal modello di partenza (RLHF o DPO).',
  build(root) {
    root.append(el('div', { html: `
      <p><b>1. SFT</b> (<i>supervised fine-tuning</i>): si continua l'addestramento con la stessa cross-entropy del modulo 8, ma su conversazioni scritte o selezionate da persone.</p>
      <p><b>2. Preferenze</b>: per un prompt si generano più risposte e una persona (o un modello) dice quale è migliore. Si ottengono coppie <i>(vincitrice, perdente)</i>.</p>
      <p><b>3. Ottimizzazione sulle preferenze</b>: con <b>RLHF</b> si addestra un <i>reward model</i> sulle coppie e poi si ottimizza la policy con reinforcement learning (tipicamente PPO), penalizzando la distanza dal modello di partenza. Con <b>DPO</b> si salta il reward model e si ottimizza direttamente sulle coppie.</p>` }));

    /* ---- Lab A: SFT loss mask ---- */
    const CHAT = [['sys', 'Sei'], ['sys', 'un'], ['sys', 'assistente'], ['sys', 'utile.'], ['usr', 'Qual'], ['usr', 'è'], ['usr', 'la'], ['usr', 'capitale'], ['usr', "d'Italia?"], ['ast', 'La'], ['ast', 'capitale'], ['ast', "d'Italia"], ['ast', 'è'], ['ast', 'Roma.'], ['ast', '<fine>']];
    let masked = true; const chips = el('div', { class: 'chips' }), out1 = el('div', { class: 'readout' });
    const role = { sys: ['sistema', 'c2'], usr: ['utente', 'c1'], ast: ['assistente', 'c0'] };
    function drawChat() {
      chips.innerHTML = '';
      CHAT.forEach(([r, w], i) => {
        const inLoss = !masked || r === 'ast';
        chips.append(el('span', { class: 'chip ' + role[r][1], style: `opacity:${inLoss ? 1 : 0.38};${inLoss ? 'font-weight:700' : 'text-decoration:line-through'}`, title: role[r][0] + (inLoss ? ' · nella loss' : ' · escluso dalla loss'), text: w }));
      });
      const n = CHAT.filter(([r]) => !masked || r === 'ast').length;
      out1.innerHTML = `token che contribuiscono alla loss: <b>${n}</b> su ${CHAT.length} · loss = −(1/${n}) Σ log p(token | precedenti)${masked ? ' sulla sola risposta' : ' su tutto, compreso il prompt'}`;
    }
    root.append(lab('Laboratorio · SFT: su quali token si calcola la loss', controls(checkbox('maschera sistema e utente (loss solo sulla risposta)', true, v => { masked = v; drawChat(); })), chips, out1,
      el('p', { class: 'muted', html: 'La legenda: <span class="chip c2">sistema</span> <span class="chip c1">utente</span> <span class="chip c0">assistente</span>. Il modello vede sempre tutto il contesto, ma si addestra a <i>produrre</i> solo la parte dell\'assistente. Imparare a riprodurre il prompt dell\'utente non serve (e può insegnare cose sbagliate). Il token finale segnala quando fermarsi.' })));
    drawChat();

    /* ---- Lab B: reward model + KL-regularised optimisation ---- */
    const sBias = slider({ label: 'preferenza dei labeler per risposte lunghe', min: 0, max: 3, step: 0.25, value: 2, fmt: v => v.toFixed(2), onInput: () => refit() });
    const sN = slider({ label: 'coppie di preferenza', min: 40, max: 600, step: 20, value: 300, onInput: () => refit() });
    const sBeta = slider({ label: 'log₁₀ β (forza della penalità KL)', min: -1.5, max: 0.7, step: 0.05, value: 0.0, fmt: v => 'β = ' + Math.pow(10, v).toFixed(2), onInput: () => upd() });
    const cvP = canvasEl(300), cvB = canvasEl(230), out2 = el('div', { class: 'readout' }); let RM = null;
    const cP = setupCanvas(cvP, (ctx, w, h, t) => {
      if (!RM) return;
      const p = tilt(RM.r, Math.pow(10, sBeta.get())), n = CANDS.length, left = 40, bot = 92, top = 20, bw = (w - left - 8) / n, mx = Math.max(0.05, ...p, ...REF) * 1.15, Y = v => h - bot - v / mx * (h - bot - top);
      for (let i = 0; i <= 4; i++) { const v = mx * i / 4; ctx.strokeStyle = rgba(t.line, 0.7); ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(w - 6, Y(v)); ctx.stroke(); text(ctx, f2(v, 2), left - 5, Y(v) + 4, { align: 'right', color: t.muted, size: 10 }); }
      CANDS.forEach((c, i) => {
        const x = left + i * bw + bw * 0.12, ww = bw * 0.76;
        ctx.fillStyle = rgba(t.muted, 0.35); ctx.fillRect(x, Y(REF[i]), ww / 2, h - bot - Y(REF[i]));
        ctx.fillStyle = t.accent; ctx.fillRect(x + ww / 2, Y(p[i]), ww / 2, h - bot - Y(p[i]));
        // true quality marker
        ctx.fillStyle = t.teal; ctx.beginPath(); ctx.arc(x + ww / 2, h - bot + 8, 3 + c.q * 7, 0, 7); ctx.fill();
        ctx.save(); ctx.translate(x + ww / 2, h - bot + 24); ctx.rotate(-0.8); text(ctx, c.n, 0, 0, { align: 'right', color: t.ink, size: 10, sans: true, weight: 600 }); ctx.restore();
      });
      const lg = [['modello di partenza', rgba(t.muted, 0.6)], ['policy ottimizzata', t.accent], ['qualità vera (cerchio più grande = migliore)', t.teal]];
      let lx = left; lg.forEach(([s, c]) => { ctx.fillStyle = c; ctx.fillRect(lx, 4, 10, 10); text(ctx, s, lx + 14, 13, { color: t.muted, size: 10 }); lx += 28 + s.length * 6; });
    });
    const cB = setupCanvas(cvB, (ctx, w, h, t) => {
      if (!RM) return;
      const lbs = range(61).map(i => -1.5 + i * (2.2 / 60)), rs = RM.r, rmin = Math.min(...rs), rmax = Math.max(...rs);
      const ps = lbs.map(l => tilt(rs, Math.pow(10, l))), xs = lbs;
      const c = chart(ctx, w, h, t, [
        { xs, ys: ps.map(p => (sum(p.map((x, i) => x * rs[i])) - rmin) / (rmax - rmin)), color: t.amber, lw: 2.5, label: 'reward del modello di reward (0–1)' },
        { xs, ys: ps.map(trueQ), color: t.teal, lw: 2.5, label: 'qualità vera attesa' },
        { xs, ys: ps.map(p => klDiv(p, REF) / 3.5), color: t.rose, lw: 1.8, dash: [5, 4], label: 'KL dal modello di partenza (÷3,5)' }], { xlabel: 'log₁₀ β  (a sinistra: ottimizzazione più aggressiva)', ymin: 0, ymax: 1.05, nx: 4, xfmt: v => v.toFixed(1) });
      ctx.strokeStyle = t.ink; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(c.X(sBeta.get()), c.y1); ctx.lineTo(c.X(sBeta.get()), c.y0); ctx.stroke(); ctx.setLineDash([]);
    });
    function refit() {
      const P = makePairs(sBias.get(), sN.get(), 11), w = fitRM(P), r = rewards(w);
      RM = { P, w, r, acc: P.filter(([a, b]) => r[a] > r[b]).length / P.length }; upd();
    }
    function upd() {
      if (!RM) return;
      const beta = Math.pow(10, sBeta.get()), p = tilt(RM.r, beta);
      out2.innerHTML = `reward model appreso: r = <b>${f2(RM.w[0], 2)}</b>·qualità + <b>${f2(RM.w[1], 2)}</b>·lunghezza (accuratezza sulle coppie: ${f2(RM.acc * 100, 0)}%) · qualità vera attesa: <b>${f2(trueQ(REF), 2)}</b> → <b>${f2(trueQ(p), 2)}</b> · KL = <b>${f2(klDiv(p, REF), 2)}</b> nat`;
      cP.redraw(); cB.redraw();
    }
    root.append(lab('Laboratorio · reward model e ottimizzazione con penalità KL (reward hacking)',
      el('p', { html: 'Dodici risposte candidate a uno stesso prompt, con qualità e lunghezza <b>definite a mano</b> (un giocattolo, non un vero modello). I labeler scelgono la risposta migliore con rumore, ma sono più indulgenti verso risposte lunghe. Il reward model vede solo le coppie scelte e ne impara due pesi. Poi si trova la policy che massimizza il reward appreso restando vicina al modello di partenza.' }),
      controls(sBias.el, sN.el), controls(sBeta.el), cvP, out2, cvB,
      el('p', { class: 'muted', html: 'Senza preferenza per la lunghezza, ridurre β migliora quasi sempre la qualità vera. Con una preferenza forte il reward model impara che "lungo = buono": a β piccolo la policy va dove il <i>reward</i> è alto anche se la qualità vera no (legge di Goodhart). Il controllo del KL (β grande) non elimina il problema, lo limita.' })));
    refit();

    /* ---- Lab C: DPO ---- */
    const sBd = slider({ label: 'β di DPO', min: 0.05, max: 2, step: 0.05, value: 0.5, fmt: v => v.toFixed(2), onInput: () => dpoUpd(true) });
    const sSt = slider({ label: 'passi di ottimizzazione', min: 0, max: 200, step: 1, value: 40, fmt: v => String(v * 10), onInput: () => dpoUpd(false) });
    const cvD = canvasEl(260), cvL = canvasEl(260), out3 = el('div', { class: 'readout' }); let snaps = null;
    const cD = setupCanvas(cvD, (ctx, w, h, t) => {
      if (!snaps || !RM) return;
      const th = snaps[Math.min(sSt.get(), snaps.length - 1)], p = softmax(th), n = CANDS.length, left = 40, bot = 92, top = 20, bw = (w - left - 8) / n, mx = Math.max(0.05, ...p, ...REF) * 1.15, Y = v => h - bot - v / mx * (h - bot - top);
      for (let i = 0; i <= 4; i++) { const v = mx * i / 4; ctx.strokeStyle = rgba(t.line, 0.7); ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(w - 6, Y(v)); ctx.stroke(); text(ctx, f2(v, 2), left - 5, Y(v) + 4, { align: 'right', color: t.muted, size: 10 }); }
      CANDS.forEach((c, i) => {
        const x = left + i * bw + bw * 0.12, ww = bw * 0.76;
        ctx.fillStyle = rgba(t.muted, 0.35); ctx.fillRect(x, Y(REF[i]), ww / 2, h - bot - Y(REF[i]));
        ctx.fillStyle = t.accent; ctx.fillRect(x + ww / 2, Y(p[i]), ww / 2, h - bot - Y(p[i]));
        ctx.save(); ctx.translate(x + ww / 2, h - bot + 12); ctx.rotate(-0.8); text(ctx, c.n, 0, 0, { align: 'right', color: t.ink, size: 10, sans: true, weight: 600 }); ctx.restore();
      });
      text(ctx, 'grigio: modello di partenza · blu: policy dopo DPO sulle stesse coppie del laboratorio precedente', left, 12, { color: t.muted, size: 10 });
    });
    const cL = setupCanvas(cvL, (ctx, w, h, t) => {
      const beta = sBd.get(), N = 56, g = [];
      const lo = -6, hi = 6, M = range(N).map(i => range(N).map(j => { const dl = lo + (hi - lo) * (i + 0.5) / N, dw = lo + (hi - lo) * (j + 0.5) / N; return -Math.log(sigmoid(beta * (dw - dl))); }));
      const side = Math.min(w - 90, h - 40), x0 = 56, y0 = 8, cs = side / N;
      let vmax = 0; M.forEach(r => r.forEach(v => { if (v > vmax) vmax = v; }));
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { ctx.fillStyle = rgba(t.accent, Math.min(1, M[i][j] / vmax) * 0.95); ctx.fillRect(x0 + j * cs, y0 + (N - 1 - i) * cs, cs + 0.5, cs + 0.5); }
      ctx.strokeStyle = t.rose; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0, y0 + side); ctx.lineTo(x0 + side, y0); ctx.stroke();
      text(ctx, 'Δ vincente = log π(y⁺)/π_ref(y⁺)  →', x0, y0 + side + 20, { color: t.muted, size: 11 });
      ctx.save(); ctx.translate(14, y0 + side); ctx.rotate(-Math.PI / 2); text(ctx, 'Δ perdente = log π(y⁻)/π_ref(y⁻)  →', 0, 0, { color: t.muted, size: 11 }); ctx.restore();
      text(ctx, 'retta rosa: margine 0 (loss = ln 2)', x0 + side + 8, y0 + 14, { color: t.rose, size: 10 });
      text(ctx, 'loss alta: in alto a sinistra', x0 + side + 8, y0 + 30, { color: t.muted, size: 10 }); text(ctx, 'loss bassa: in basso a destra', x0 + side + 8, y0 + 44, { color: t.muted, size: 10 });
    });
    function dpoUpd(recompute) {
      if (!RM) return;
      if (recompute || !snaps) snaps = dpoTrain(RM.P, sBd.get(), 2000, 0.9, 10);
      const th = snaps[Math.min(sSt.get(), snaps.length - 1)], p = softmax(th);
      out3.innerHTML = `dopo <b>${sSt.get() * 10}</b> passi: qualità vera attesa <b>${f2(trueQ(p), 2)}</b> (partenza ${f2(trueQ(REF), 2)}) · KL dal partenza <b>${f2(klDiv(p, REF), 2)}</b> nat · loss DPO sulle coppie <b>${f2(dpoLoss(th, RM.P, sBd.get()), 3)}</b> (a θ₀: ${f2(Math.LN2, 3)})`;
      cD.redraw(); cL.redraw();
    }
    root.append(lab('Laboratorio · DPO: ottimizzare direttamente sulle coppie',
      el('p', { html: 'Qui non c\'è reward model: i logit della policy sono parametri liberi (tabellari) e la loss DPO li sposta in modo che la risposta vincente guadagni probabilità <i>rispetto al modello di partenza</i> più della perdente. Le coppie sono le stesse del laboratorio precedente (stessa preferenza per la lunghezza).' }),
      controls(sBd.el, sSt.el), cvD, out3, cvL,
      el('p', { class: 'muted', html: 'Nella mappa conta solo il <b>margine</b> Δ vincente − Δ perdente: il gradiente spinge a separare le due risposte, e spinge di più quando il modello sbaglia l\'ordine (margine negativo). Nota: la tabella è libera di imparare qualunque cosa sulle 12 risposte, anche il rumore; con poche coppie e molti passi può inseguire le coppie fortuite.' })));
    dpoUpd(true);

    root.append(el('h2', { text: 'Che cosa abbiamo visto' }));
    root.append(el('ul', { class: 'prose', html: `
      <li><b>SFT</b> insegna il formato e il tono di un assistente; la loss è mascherata sui token che il modello deve produrre.</li>
      <li><b>Reward model</b>: un modello addestrato sulle preferenze umane. È una <i>proxy</i> imperfetta di ciò che si vuole davvero, e ottimizzare troppo una proxy la rompe.</li>
      <li><b>Penalità KL</b>: il termine −β·KL(π‖π<sub>ref</sub>) tiene la policy vicina al modello di partenza; senza, la policy troverebbe i difetti del reward model.</li>
      <li><b>RLHF con PPO</b> è un processo a più parti (policy, reference, reward, critic) e instabile da regolare. <b>DPO</b> sfrutta una identità che lega policy ottimale e reward (vedi sotto) per ridurlo a una loss supervisionata sulle coppie.</li>
      <li>Varianti moderne: reward verificabili (es. test di codice o risultati matematici) al posto di preferenze umane, <i>RLAIF</i> (giudice automatico), GRPO (niente critic, vantaggio relativo dentro un gruppo di risposte).</li>` }));

    root.append(details('math', 'La matematica: Bradley-Terry, KL, DPO', 'formule', `
      <p><b>Reward model (Bradley-Terry).</b> Con <var>r</var>(<var>x</var>,<var>y</var>) punteggio scalare, la probabilità che <var>y</var><sub>w</sub> sia preferita a <var>y</var><sub>l</sub> è</p>
      <span class="eq"><var>P</var>(<var>y</var><sub>w</sub> ≻ <var>y</var><sub>l</sub>) = σ(<var>r</var>(<var>y</var><sub>w</sub>) − <var>r</var>(<var>y</var><sub>l</sub>)), &nbsp; ℒ<sub>RM</sub> = −E log σ(<var>r</var>(<var>y</var><sub>w</sub>) − <var>r</var>(<var>y</var><sub>l</sub>))</span>
      <p>Conta solo la <i>differenza</i> di reward: il reward è definito a meno di una costante.</p>
      <p><b>Obiettivo RLHF.</b> Si cerca una policy <var>π</var> che massimizza il reward medio con una penalità sulla divergenza dal modello di riferimento:</p>
      <span class="eq">max<sub>π</sub> E<sub>y∼π</sub>[<var>r</var>(<var>y</var>)] − <var>β</var> KL(<var>π</var> ‖ <var>π</var><sub>ref</sub>)</span>
      <p>Questo problema ha soluzione esatta (è una distribuzione di Gibbs, come la softmax con temperatura del modulo 7):</p>
      <span class="eq"><var>π</var>*(<var>y</var>) = ${frac('<var>π</var><sub>ref</sub>(<var>y</var>) exp(<var>r</var>(<var>y</var>)/<var>β</var>)', '<var>Z</var>')}</span>
      <p>Il grafico "reward hacking" usa proprio questa formula chiusa (nel RLHF vero si approssima con PPO su un modello enorme).</p>
      <p><b>Da qui a DPO.</b> Invertendo la formula: <var>r</var>(<var>y</var>) = <var>β</var> log(<var>π</var>*(<var>y</var>)/<var>π</var><sub>ref</sub>(<var>y</var>)) + <var>β</var> log <var>Z</var>. Sostituendo nella loss di Bradley-Terry la costante <var>Z</var> si cancella (dipende solo dal prompt) e resta</p>
      <span class="eq">ℒ<sub>DPO</sub> = −E log σ( <var>β</var> [ log ${frac('<var>π</var>(<var>y</var><sub>w</sub>)', '<var>π</var><sub>ref</sub>(<var>y</var><sub>w</sub>)')} − log ${frac('<var>π</var>(<var>y</var><sub>l</sub>)', '<var>π</var><sub>ref</sub>(<var>y</var><sub>l</sub>)')} ] )</span>
      <p><b>Gradiente.</b> Con margine <var>m</var> = <var>β</var>(Δ<sub>w</sub> − Δ<sub>l</sub>): ∂ℒ/∂<var>m</var> = −σ(−<var>m</var>), cioè la correzione è tanto maggiore quanto più il modello ordina male la coppia. Con policy tabellare sui logit <var>θ</var> la normalizzazione della softmax si cancella nella differenza e ∂ℒ/∂<var>θ</var><sub>j</sub> = −<var>β</var> σ(−<var>m</var>) (𝟙[<var>j</var>=w] − 𝟙[<var>j</var>=l]): il codice del laboratorio usa esattamente questa espressione (verificata con differenze finite).</p>
      <p><b>Attenzione.</b> L'equivalenza vale per il caso ideale (policy abbastanza espressiva, ottimizzazione perfetta). In pratica DPO e RLHF danno risultati diversi, e DPO può abbassare la probabilità anche delle risposte vincenti, perché la loss guarda solo il margine.</p>`));
    root.append(el('h2', { text: 'Verifica rapida' }));
    root.append(quiz([
      { q: 'Perché nella SFT si maschera il prompt dalla loss?', opts: ['Per risparmiare memoria', 'Perché il compito è imparare a produrre la risposta, non a riprodurre la domanda', 'Perché il prompt non entra nel contesto'], ans: 1, why: 'Il prompt resta nel contesto (il modello lo legge), ma la loss si calcola sui soli token che deve generare.' },
      { q: 'A che serve il termine KL nell\'obiettivo RLHF?', opts: ['A rendere la risposta più lunga', 'A impedire che la policy si allontani troppo dal modello di partenza sfruttando i difetti del reward model', 'A velocizzare il training'], ans: 1, why: 'Senza penalità la policy cerca output che massimizzano la proxy anche se non sono buoni (reward hacking) e perde fluidità e diversità.' },
      { q: 'Perché il reward model confronta coppie invece di dare un voto assoluto?', opts: ['Perché è più facile e più coerente per le persone dire quale tra due risposte è migliore', 'Perché i voti non sono numeri', 'Perché serve meno dati in ogni caso'], ans: 0, why: 'I giudizi relativi sono più affidabili di quelli assoluti, e Bradley-Terry li trasforma in un punteggio scalare.' },
      { q: 'Che cosa elimina DPO rispetto a RLHF con PPO?', opts: ['Il modello di riferimento', 'Il reward model esplicito e la fase di reinforcement learning', 'Le coppie di preferenza'], ans: 1, why: 'Serve ancora il riferimento e servono ancora le coppie, ma la policy si ottimizza con una loss supervisionata diretta.' },
    ]));
  },
});
