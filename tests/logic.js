'use strict';
/* Numerical / algorithmic checks of the pure logic behind the labs. */
const fs = require('fs'), path = require('path'), vm = require('vm');
const { makeEnv } = require('./fakedom');
const root = path.join(__dirname, '..');
const order = fs.readFileSync(path.join(root, 'build.py'), 'utf8').match(/order = \[([^\]]*)\]/s)[1].match(/"([^"]+)"/g).map(s => s.replace(/"/g, ''));
const env = makeEnv(), ctx = vm.createContext(env.win);
for (const f of order) { const p = path.join(root, 'src', 'js', f); if (fs.existsSync(p)) vm.runInContext(fs.readFileSync(p, 'utf8'), ctx, { filename: f }); }
const run = c => vm.runInContext(c, ctx);
let bad = 0;
const ok = (name, cond, extra = '') => { if (!cond) bad++; console.log((cond ? 'ok   ' : 'FAIL ') + name + (extra ? '  ' + extra : '')); };
const close = (a, b, e = 1e-9) => Math.abs(a - b) <= e;

// --- tokenizer
const B = run('LAB.tests.bpe');
let r = B.bpeTrain('aaab aaab', 3);
ok('BPE first merge is the most frequent pair (a,a)', r.merges[0][0] === 'a' && r.merges[0][1] === 'a' && r.merges[0][2] === 4);
const text = 'il gatto mangia il pesce. il gatto dorme sul divano. la gatta mangia il pesce fresco.';
r = B.bpeTrain(text, 25);
ok('BPE encode reproduces training segmentation', r.seqs.every(s => JSON.stringify(B.bpeEncodeWord(s.w, r.merges)) === JSON.stringify(s.syms)));
ok('BPE token count is non-increasing', r.counts.every((c, i) => i === 0 || c <= r.counts[i - 1]), r.counts.slice(0, 6).join(','));
ok('BPE round trip (concatenation recovers text)', B.bpeEncode('il gatto', r.merges).join('').replace(/▁/g, ' ').trim() === 'il gatto');

// --- parameter counts
const pc = run('LAB.tests.paramCount')({ d: 768, L: 12, h: 12, hid: 3072, V: 50257, ctx: 1024, glu: 'classic', tie: true });
ok('GPT-2 small ≈ 124M parameters', Math.abs(pc.N - 124.4e6) / 124.4e6 < 0.01, (pc.N / 1e6).toFixed(2) + 'M');
const pl = run('LAB.tests.paramCount')({ d: 4096, L: 32, h: 32, hid: 11008, V: 32000, ctx: 0, glu: 'swiglu', tie: false });
ok('Llama-style 7B ≈ 6.7B parameters', Math.abs(pl.N - 6.74e9) / 6.74e9 < 0.01, (pl.N / 1e9).toFixed(3) + 'B');

// --- embeddings
const { WORDS, E } = run('LAB.tests.emb');
const nearest = (v, ex) => WORDS.map((w, i) => [w, i]).filter(([w, i]) => !ex.includes(w)).map(([w, i]) => [w, run('cosine')(v, E[i])]).sort((a, b) => b[1] - a[1])[0][0];
const vec = (a, b, c) => E[WORDS.indexOf(a)].map((x, j) => x - E[WORDS.indexOf(b)][j] + E[WORDS.indexOf(c)][j]);
ok('re − uomo + donna ≈ regina', nearest(vec('re', 'uomo', 'donna'), ['re', 'uomo', 'donna']) === 'regina');
ok('Roma − Italia + Francia ≈ Parigi', nearest(vec('Roma', 'Italia', 'Francia'), ['Roma', 'Italia', 'Francia']) === 'Parigi');

// --- positional encoding
const sinPE = run('LAB.tests.sinPE'), dot = run('dot');
ok('PE(p)·PE(p+k) depends only on k', [0, 5, 40].every(k => close(dot(sinPE(0, 64), sinPE(k, 64)), dot(sinPE(37, 64), sinPE(37 + k, 64)), 1e-9)));

// --- attention
const out = run(`(() => {
  const r = rng(1), mk = () => range(4).map(() => range(4).map(() => r.gauss()));
  const X = range(5).map(() => range(4).map(() => r.gauss())), Wq = mk(), Wk = mk(), Wv = mk();
  const a = attention(X, Wq, Wk, Wv, false), c = attention(X, Wq, Wk, Wv, true);
  const perm = [3, 0, 4, 1, 2], b = attention(perm.map(i => X[i]), Wq, Wk, Wv, false);
  return {
    rows: a.A.every(row => Math.abs(sum(row) - 1) < 1e-12), causal: c.A.every((row, i) => row.every((v, j) => j <= i || v === 0)),
    equiv: perm.every((src, k) => b.Z[k].every((v, j) => Math.abs(v - a.Z[src][j]) < 1e-9)),
  };
})()`);
ok('attention rows sum to 1', out.rows); ok('causal mask zeroes the future', out.causal); ok('attention is permutation-equivariant', out.equiv);

// --- PCA
const pv = run(`(() => { const p = pca(LAB.tests.emb.E, 3).proj, v = k => { const m = sum(p.map(r => r[k])) / p.length; return sum(p.map(r => (r[k] - m) ** 2)); }; return [v(0), v(1), v(2)]; })()`);
ok('PCA components ordered by variance', pv[0] >= pv[1] && pv[1] >= pv[2], pv.map(x => x.toFixed(2)).join(' ≥ '));


// --- induction circuit
const ic = run('LAB.tests.induction')('mario rossi lavora a milano . anna bianchi vive a roma . mario'.split(' '), 8);
const last = ic.P[ic.P.length - 1], best = ic.vocab[last.indexOf(Math.max(...last))];
ok('induction head predicts "rossi" after a repeated "mario" (copy from context)', best === 'rossi' && Math.max(...last) > 0.95, best + ' ' + Math.max(...last).toFixed(3));
ok('induction: rows of head-1 and head-2 sum to 1', ic.A1.every(r => close(r.reduce((a, b) => a + b, 0), 1)) && ic.A2.every(r => close(r.reduce((a, b) => a + b, 0), 1)));


// --- MLP block pieces
const T6 = run('LAB.tests');
['relu', 'gelu', 'silu'].forEach(k => {
  const a = T6.ACT[k], h = 1e-5; const bad = [-2.3, -0.7, 0.4, 1.9].some(z => Math.abs((a.f(z + h) - a.f(z - h)) / (2 * h) - a.d(z)) > 1e-4);
  ok(`${k}: analytic derivative matches finite differences`, !bad);
});
const fit = run('new LAB.tests.Fit1D(8, "relu", x => Math.sin(2 * x), 3)'); const l0 = fit.loss(); for (let i = 0; i < 1500; i++) fit.step(0.03);
ok('Fit1D (8 ReLU neurons) learns sin(2x)', fit.loss() < 0.02 && fit.loss() < l0 / 10, l0.toFixed(3) + ' -> ' + fit.loss().toFixed(4));
const fg = T6.gradFlow(40, 0.5, 12, 16, 1);
ok('gradient vanishes without residual (s=0.5, 40 layers)', fg.plain[40] < -8, fg.plain[40].toFixed(1));
ok('gradient survives with residual (s=0.5, 40 layers)', fg.res[40] > -1 && fg.res[40] < 6, fg.res[40].toFixed(1));
const xs = [2.2, -0.5, 1.4, 3.1, -1.2, 0.4, 2.7, 0.9], y1 = T6.normLayer(xs, 1.5, 0.3, 'layer'), y2 = T6.normLayer(xs.map(v => v * 7), 1.5, 0.3, 'layer');
const mean = a => a.reduce((p, q) => p + q, 0) / a.length;
ok('LayerNorm output has mean β and std γ', close(mean(y1), 0.3, 1e-9) && close(Math.sqrt(mean(y1.map(v => (v - 0.3) ** 2))), 1.5, 1e-3));
ok('LayerNorm is scale-invariant', y1.every((v, i) => close(v, y2[i], 1e-4)));


// --- sampling
const sd1 = run('LAB.tests.sampleDist')([3, 2, 1, 0], 1, 4, 1), sdk = run('LAB.tests.sampleDist')([3, 2, 1, 0], 1, 1, 1), sdp = run('LAB.tests.sampleDist')([3, 2, 1, 0], 1, 4, 0.8), sdt = run('LAB.tests.sampleDist')([3, 2, 1, 0], 0.05, 4, 1);
ok('sampling: probabilities sum to 1', close(sd1.p1.reduce((a, b) => a + b, 0), 1));
ok('sampling: top-k=1 keeps only the argmax', sdk.keep.filter(Boolean).length === 1 && sdk.keep[0]);
ok('sampling: top-p keeps the smallest set reaching p', sdp.keep.filter(Boolean).length === 2, sdp.p0.slice(0, 2).map(v => v.toFixed(3)).join('+'));
ok('sampling: T -> 0 concentrates on the argmax', sdt.p1[0] > 0.999);


// --- training: gradients, optimizers, the tiny LM
const TT = run('LAB.tests');
const ce0 = TT.softmaxCE([1, 3, 2, 0.5], 2); let ceErr = 0;
[0, 1, 2, 3].forEach(k => { const e = 1e-6, a = [1, 3, 2, 0.5], b = [1, 3, 2, 0.5]; a[k] += e; b[k] -= e; ceErr = Math.max(ceErr, Math.abs((TT.softmaxCE(a, 2).loss - TT.softmaxCE(b, 2).loss) / (2 * e) - ce0.grad[k])); });
ok('softmax cross-entropy gradient = p − onehot (finite differences)', ceErr < 1e-6, ceErr.toExponential(1));
const P0 = { x: 1.5, t: 1, w1: 0.8, b1: -0.3, w2: 1.2, b2: 0.1 }, bp = TT.backprop(P0); let bpErr = 0;
['w1', 'b1', 'w2', 'b2', 'x'].forEach(k => { const e = 1e-6; bpErr = Math.max(bpErr, Math.abs((TT.backprop({ ...P0, [k]: P0[k] + e }).L - TT.backprop({ ...P0, [k]: P0[k] - e }).L) / (2 * e) - bp.g[k])); });
ok('scalar backprop matches finite differences', bpErr < 1e-6, bpErr.toExponential(1));
Object.entries(TT.SURF).forEach(([k, S]) => { let er = 0; [[0.3, -0.4], [-1.1, 1.7], [1.5, 0.2]].forEach(([a, b]) => { const e = 1e-6, g = S.g(a, b); er = Math.max(er, Math.abs((S.f(a + e, b) - S.f(a - e, b)) / (2 * e) - g[0]), Math.abs((S.f(a, b + e) - S.f(a, b - e)) / (2 * e) - g[1])); }); ok(`surface "${k}": analytic gradient matches finite differences`, er < 1e-4, er.toExponential(1)); });
const runOpt = (kind, key, lr, n) => { const S = TT.SURF[key], o = run(`new LAB.tests.Opt('${kind}')`); o.reset(S.start); for (let i = 0; i < n; i++) o.step(S.g, lr); return { f: S.f(o.p[0], o.p[1]), div: o.div }; };

ok('valley (tuned lrs, 400 steps): Momentum and Adam are far ahead of SGD', runOpt('momentum', 'valle', 0.005, 400).f < 0.01 * runOpt('sgd', 'valle', 0.01, 400).f && runOpt('adam', 'valle', 0.05, 400).f < 0.01 * runOpt('sgd', 'valle', 0.01, 400).f, ['sgd:0.01', 'momentum:0.005', 'adam:0.05'].map(x => { const [k, l] = x.split(':'); return k + ' ' + runOpt(k, 'valle', +l, 400).f.toExponential(1); }).join('  '));
ok('elongated bowl: Adam converges while SGD is still far (150 steps)', runOpt('adam', 'conca', 0.05, 150).f < 1e-3 && runOpt('sgd', 'conca', 0.004, 150).f > 0.1);
ok('SGD diverges on the elongated bowl when lr > 2/λ', runOpt('sgd', 'conca', 0.05, 200).div);
ok('SGD is stable below the limit (lr 0.004 < 2/400)', !runOpt('sgd', 'conca', 0.004, 3000).div && runOpt('sgd', 'conca', 0.004, 3000).f < 1e-6);
// tiny LM: analytic gradients vs finite differences, and learning
const lmT = run('new LAB.tests.CharLM(6, 2, 3, 5, 3)'); const ctxs = [[0, 1], [2, 3], [4, 5], [1, 1]], ys = [2, 4, 0, 5];
lmT.lossAndGrad(ctxs, ys); const gcopy = lmT.G.map(g => Float64Array.from(g)); let lmErr = 0;
lmT.P.forEach((p, q) => { for (const i of [0, Math.floor(p.length / 2), p.length - 1]) { const e = 1e-6, o = p[i]; p[i] = o + e; const a = lmT.lossAndGrad(ctxs, ys); p[i] = o - e; const b = lmT.lossAndGrad(ctxs, ys); p[i] = o; lmErr = Math.max(lmErr, Math.abs((a - b) / (2 * e) - gcopy[q][i])); } });
ok('CharLM hand-written backprop matches finite differences', lmErr < 1e-6, lmErr.toExponential(1));
const corpus = run('LAB.tests.CORPUS'), chars = [...new Set(corpus)].sort(), data = Int32Array.from([...corpus].map(c => chars.indexOf(c))), V = chars.length;
const lm = run(`new LAB.tests.CharLM(${V}, 3, 8, 64, 7)`), lmL0 = lm.evalLoss(data, 0, data.length);
ok('untrained LM loss ≈ ln V', Math.abs(lmL0 - Math.log(V)) < 0.35, lmL0.toFixed(2) + ' vs ' + Math.log(V).toFixed(2));
let seed = 99; const rr = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296; const nT = Math.floor(data.length * 0.88);
for (let s = 0; s < 500; s++) { const cx = [], yy = []; for (let b = 0; b < 32; b++) { const i = 3 + Math.floor(rr() * (nT - 3)); cx.push(data.subarray(i - 3, i)); yy.push(data[i]); } lm.lossAndGrad(cx, yy); lm.adam(0.01); }
const l1 = lm.evalLoss(data, 0, nT), v1 = lm.evalLoss(data, nT, data.length);
ok('LM learns (train loss falls well below ln V after 500 steps)', l1 < 0.6 * Math.log(V), `${lmL0.toFixed(2)} -> train ${l1.toFixed(2)}, val ${v1.toFixed(2)}`);

// --- module 9: inference
const INF = run('LAB.tests.infer');
{
  const r0 = run('rng')(5), d = 4, n = 9, mk = (a, b) => Array.from({ length: a }, () => Array.from({ length: b }, () => r0() - 0.5));
  const Xs = mk(n, d), Wq = mk(d, d), Wk = mk(d, d), Wv = mk(d, d);
  const A = INF.kvDecode(Xs, Wq, Wk, Wv, false), Bc = INF.kvDecode(Xs, Wq, Wk, Wv, true);
  let md = 0; A.outs.forEach((o, i) => o.forEach((v, j) => { md = Math.max(md, Math.abs(v - Bc.outs[i][j])); }));
  ok('KV cache gives the same outputs as full recomputation', md < 1e-12, md.toExponential(1));
  ok('projection counts: n(n+1)/2 without cache, n with cache', A.proj === n * (n + 1) / 2 && Bc.proj === n, `${A.proj} vs ${Bc.proj}`);
}
ok('Llama-2 7B KV cache = 0.5 MiB/token (fp16)', INF.kvBytesPerToken(32, 32, 128, 2) === 524288);
ok('GQA 32->8 KV heads cuts the cache 4x', INF.kvBytesPerToken(32, 32, 128, 2) / INF.kvBytesPerToken(32, 8, 128, 2) === 4);
{
  const o = { N: 7e9, W: 14e9, kvSeq: 0, bw: 3.35e12, flops: 9.9e14 };
  const s1 = INF.decodeStep({ ...o, B: 1 }), s2 = INF.decodeStep({ ...o, B: 2 }), sBig = INF.decodeStep({ ...o, B: 2000 });
  ok('decode at batch 1 is memory-bound; throughput ~ BW / weight bytes', s1.bound === 'memoria' && close(s1.tps, 3.35e12 / 14e9, 1e-6), s1.tps.toFixed(0) + ' tok/s');
  ok('doubling batch doubles throughput while memory-bound', close(s2.tps / s1.tps, 2, 1e-9));
  ok('very large batch becomes compute-bound', sBig.bound === 'calcolo');
}
{
  let seedS = 3; const rs = () => (seedS = (seedS * 1664525 + 1013904223) % 4294967296) / 4294967296, alpha = 0.7, k = 4; let tot = 0; const runs = 200000;
  for (let i = 0; i < runs; i++) { let a = 0; while (a < k && rs() < alpha) a++; tot += a + 1; }
  ok('speculative decoding expected tokens: closed form matches simulation', Math.abs(tot / runs - INF.specGain(alpha, k)) < 0.02, (tot / runs).toFixed(3) + ' vs ' + INF.specGain(alpha, k).toFixed(3));
  ok('specGain(alpha, 1) = 1 + alpha', close(INF.specGain(0.6, 1), 1.6));
}

// --- module 10: post-training
const PT = run('LAB.tests.post');
{
  const P = PT.makePairs(1, 200, 5), w0 = [0.3, -0.2], g = PT.btGrad(w0, P); let e = 0;
  for (let k = 0; k < 2; k++) { const h = 1e-6, a = w0.slice(), b = w0.slice(); a[k] += h; b[k] -= h; e = Math.max(e, Math.abs((PT.btLoss(a, P) - PT.btLoss(b, P)) / (2 * h) - g[k])); }
  ok('Bradley-Terry gradient matches finite differences', e < 1e-7, e.toExponential(1));
  const th0 = PT.CANDS.map(c => c.z), gd = PT.dpoGrad(th0.map((x, i) => x + 0.1 * Math.sin(i)), P, 0.7); let e2 = 0; const thB = th0.map((x, i) => x + 0.1 * Math.sin(i));
  for (const j of [0, 3, 7, 11]) { const h = 1e-6, a = thB.slice(), b = thB.slice(); a[j] += h; b[j] -= h; e2 = Math.max(e2, Math.abs((PT.dpoLoss(a, P, 0.7) - PT.dpoLoss(b, P, 0.7)) / (2 * h) - gd[j])); }
  ok('DPO gradient (softmax normaliser cancels) matches finite differences', e2 < 1e-7, e2.toExponential(1));
  ok('DPO loss at the reference policy is ln 2', close(PT.dpoLoss(th0, P, 0.5), Math.LN2, 1e-12));
  const snaps = PT.dpoTrain(P, 0.5, 400, 0.9, 10), l0 = PT.dpoLoss(snaps[0], P, 0.5), l1 = PT.dpoLoss(snaps[snaps.length - 1], P, 0.5);
  ok('DPO training lowers the DPO loss', l1 < l0 - 0.05, l0.toFixed(3) + ' -> ' + l1.toFixed(3));
  const pOpt = z => { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b); return e.map(v => v / s); };
  const r = PT.rewards(PT.fitRM(P)), pt = PT.tilt(r, 0.5), obj = p => p.reduce((s, x, i) => s + x * r[i], 0) - 0.5 * PT.klDiv(p, PT.REF);
  ok('KL-tilted policy is normalised', close(pt.reduce((a, b) => a + b), 1, 1e-12));
  ok('KL-tilted policy beats nearby perturbations on E[r] - beta KL', PT.CANDS.every((c, i) => { const q = pt.map((x, j) => x * (j === i ? 1.05 : 1)); const s = q.reduce((a, b) => a + b); return obj(q.map(x => x / s)) <= obj(pt) + 1e-12; }));
  const KLs = [4, 2, 1, 0.5, 0.25, 0.1].map(b => PT.klDiv(PT.tilt(r, b), PT.REF));
  ok('KL from reference grows as beta shrinks', KLs.every((v, i) => i === 0 || v > KLs[i - 1]), KLs.map(v => v.toFixed(2)).join(' < '));
  const qAt = (bias, beta) => { const rr = PT.rewards(PT.fitRM(PT.makePairs(bias, 300, 11))); return PT.trueQ(PT.tilt(rr, beta)); };
  const betas = [5, 2, 1, 0.5, 0.25, 0.1, 0.05], q0 = betas.map(b => qAt(0, b)), q2 = betas.map(b => qAt(2, b));
  ok('no length bias: true quality improves monotonically as beta shrinks', q0.every((v, i) => i === 0 || v >= q0[i - 1] - 1e-9), q0.map(v => v.toFixed(2)).join(' '));
  ok('strong length bias: Goodhart, true quality peaks then falls', Math.max(...q2) - q2[q2.length - 1] > 0.03 && q2.indexOf(Math.max(...q2)) < q2.length - 1, q2.map(v => v.toFixed(2)).join(' '));
}

// --- module 11: mixture of experts
const MO = run('LAB.tests.moe'), range = n => Array.from({ length: n }, (_, i) => i);
{
  const D = MO.moeData(), E4 = 4;
  // gradient of CE + lam*aux wrt W (f held constant) vs finite differences with f frozen
  const W0 = MO.moeInit(E4), lam = 1.7, G = MO.moeGrad(W0, D.X, D.lab, 2, lam), R0 = MO.moeRoute(W0, D.X, 2), f0 = MO.moeLoad(R0, E4);
  const frozen = W => { const R = MO.moeRoute(W, D.X, 2), P = range(E4).map(j => R.reduce((s, r) => s + r.p[j], 0) / R.length); return MO.moeCE(R, D.lab, E4) + lam * E4 * range(E4).reduce((s, i) => s + f0[i] * P[i], 0); };
  let e = 0; for (let j = 0; j < E4; j++) for (let c = 0; c < 2; c++) { const h = 1e-6, a = W0.map(w => w.slice()), b = W0.map(w => w.slice()); a[j][c] += h; b[j][c] -= h; e = Math.max(e, Math.abs((frozen(a) - frozen(b)) / (2 * h) - G[j][c])); }
  ok('router gradient (CE + balance loss, f frozen) matches finite differences', e < 1e-7, e.toExponential(1));
  const un = MO.moeAux(MO.moeRoute([[0, 0], [0, 0], [0, 0], [0, 0]], D.X, 1).map((r, i) => ({ ...r, idx: [i % 4] })), 4);
  ok('aux loss = 1 for perfectly uniform load and probabilities', close(un, 1, 1e-9), un.toFixed(6));
  const Wa = MO.moeTrain(MO.moeInit(4), D.X, D.lab, 1, 0, 300), Ra = MO.moeRoute(Wa, D.X, 1), fa = MO.moeLoad(Ra, 4), ceA = MO.moeCE(Ra, D.lab, 4);
  const Wb = MO.moeTrain(MO.moeInit(4), D.X, D.lab, 1, 8, 300), Rb = MO.moeRoute(Wb, D.X, 1), fb = MO.moeLoad(Rb, 4), ceB = MO.moeCE(Rb, D.lab, 4);
  ok('lam = 0: router learns the task, load follows the data imbalance', ceA < 0.4 && Math.max(...fa) * 4 > 1.5, `CE ${ceA.toFixed(2)}, max load ${(Math.max(...fa) * 4).toFixed(2)}x`);
  ok('k=1, lam = 8: load is more balanced, task loss is higher', Math.max(...fb) < Math.max(...fa) - 0.1 && ceB > ceA + 0.1, `max load ${(Math.max(...fb) * 4).toFixed(2)}x, CE ${ceB.toFixed(2)}, aux ${MO.moeAux(Rb, 4).toFixed(2)} vs ${MO.moeAux(Ra, 4).toFixed(2)}`);
  const Rk = MO.moeRoute(Wa, D.X, 2); ok('top-k gate weights sum to 1 and indices are distinct', Rk.every(r => close(r.g[0] + r.g[1], 1, 1e-12) && r.idx[0] !== r.idx[1]));
  const lo = MO.moeDropped(Ra, 4, 1, 1.0), hi = MO.moeDropped(Ra, 4, 1, 2.0);
  ok('more capacity drops fewer assignments', hi.drop <= lo.drop && lo.drop > 0, `${(lo.drop * 100).toFixed(1)}% -> ${(hi.drop * 100).toFixed(1)}%`);
}
{
  const mix = MO.moeParams({ d: 4096, L: 32, hid: 14336, E: 8, k: 2, heads: 32, kv: 8, V: 32000 });
  ok('Mixtral 8x7B: ≈46.7B total, ≈12.9B active', Math.abs(mix.total / 46.7e9 - 1) < 0.01 && Math.abs(mix.active / 12.9e9 - 1) < 0.01, `${(mix.total / 1e9).toFixed(2)}B / ${(mix.active / 1e9).toFixed(2)}B`);
  const dn = MO.moeParams({ d: 4096, L: 32, hid: 14336, E: 1, k: 1, heads: 32, kv: 8, V: 32000 });
  ok('dense (E=1, k=1) has total = active', dn.total === dn.active);
  let seedM = 9; const rm = () => (seedM = (seedM * 1664525 + 1013904223) % 4294967296) / 4294967296, Ee = 8, kk = 2, B = 5, runs = 40000; let touched = 0;
  for (let i = 0; i < runs; i++) { const s = new Set(); for (let b = 0; b < B; b++) { const pool = range(Ee); for (let j = 0; j < kk; j++) s.add(pool.splice(Math.floor(rm() * pool.length), 1)[0]); } touched += s.size / Ee; }
  ok('expected touched-expert fraction matches simulation', Math.abs(touched / runs - MO.moeTouched(Ee, kk, B)) < 0.01, (touched / runs).toFixed(3) + ' vs ' + MO.moeTouched(Ee, kk, B).toFixed(3));
}

// --- module 12: scaling laws
const SC = run('LAB.tests.scale');
{
  const f = SC.SCALE_FITS.rep, o = SC.SCALE_FITS.orig, C = 1e22;
  let bestE = 0, bestL = Infinity; for (let e = 8; e <= 12; e += 0.001) { const N = Math.pow(10, e), L = SC.scLoss(N, C / 6 / N, f); if (L < bestL) { bestL = L; bestE = e; } }
  ok('closed-form N_opt matches a brute-force search along the iso-FLOP line', Math.abs(Math.log10(SC.scNopt(C, f)) - bestE) < 0.01, `${bestE.toFixed(3)} vs ${Math.log10(SC.scNopt(C, f)).toFixed(3)}`);
  const Nc = SC.scNopt(5.76e23, f), Dc = 5.76e23 / 6 / Nc;
  ok('replication fit reproduces the ~70B / ~1.4T Chinchilla optimum at 5.76e23 FLOP', Math.abs(Nc / 70e9 - 1) < 0.1 && Math.abs(Dc / 1.4e12 - 1) < 0.1 && Math.abs(Dc / Nc - 20) < 3, `${(Nc / 1e9).toFixed(0)}B x ${(Dc / 1e12).toFixed(2)}T`);
  const No = SC.scNopt(5.76e23, o), Do = 5.76e23 / 6 / No; ok('original fit implies a much higher tokens/param ratio (~59)', Do / No > 50 && Do / No < 70, (Do / No).toFixed(1));
  ok('Chinchilla 70B/1.4T beats Gopher 280B/300B at equal compute (replica fit)', SC.scLoss(70e9, 1.4e12, f) < SC.scLoss(280e9, 300e9, f));
  const Lt = 2.1, b0 = SC.scBestN(Lt, 0, f), bI = SC.scBestN(Lt, 1e13, f);
  const Cs = b0.c, Nopt = SC.scNopt(Cs, f);
  ok('with no inference, the cost-optimal N for a target loss equals the compute-optimal N', Math.abs(Math.log10(b0.N) - Math.log10(Nopt)) < 0.03, `${(b0.N / 1e9).toFixed(2)}B vs ${(Nopt / 1e9).toFixed(2)}B`);
  ok('heavy inference shifts the optimum to a smaller model trained on more tokens', bI.N < b0.N && bI.D / bI.N > b0.D / b0.N, `${(b0.N / 1e9).toFixed(1)}B -> ${(bI.N / 1e9).toFixed(1)}B`);
  ok('a small model has a loss floor that no data can lower', !isFinite(SC.scDforLoss(1e7, 2.1, f)) && isFinite(SC.scDforLoss(1e10, 2.1, f)));
  ok('L(N, D(N, Lt)) = Lt', close(SC.scLoss(3e9, SC.scDforLoss(3e9, 2.2, f), f), 2.2, 1e-9));
}

// --- module 13: quantization
const QT = run('LAB.tests.quant');
{
  const w = QT.quantWeights(2048, 0.02, 0, 1), q8 = QT.quantize(w, 8, 32), meanS = q8.scales.reduce((a, b) => a + b) / q8.scales.length;
  ok('rounding error stays within half a step', w.every((v, i) => Math.abs(v - q8.deq[i]) <= q8.scales[Math.floor(i / 32)] / 2 + 1e-12));
  ok('RMS error ≈ s/sqrt(12) (uniform rounding noise)', Math.abs(QT.quantRMSE(w, q8.deq) / (meanS / Math.sqrt(12)) - 1) < 0.06, (QT.quantRMSE(w, q8.deq) / (meanS / Math.sqrt(12))).toFixed(3));
  const e6 = QT.quantRMSE(w, QT.quantize(w, 6, 32).deq), e7 = QT.quantRMSE(w, QT.quantize(w, 7, 32).deq);
  ok('one bit less roughly doubles the error', e6 / e7 > 1.8 && e6 / e7 < 2.2, (e6 / e7).toFixed(2));
  const wo = QT.quantWeights(2048, 0.02, 2, 30), eT = QT.quantRMSE(wo, QT.quantize(wo, 4, Infinity).deq), eG = QT.quantRMSE(wo, QT.quantize(wo, 4, 32).deq), eT0 = QT.quantRMSE(w, QT.quantize(w, 4, Infinity).deq);
  ok('outliers: per-tensor scale hurts much more than per-group', eT > 3 * eG && eT > 3 * eT0, `per-tensor ${(eT / 0.02).toFixed(3)}σ vs group-32 ${(eG / 0.02).toFixed(3)}σ (no outliers: ${(eT0 / 0.02).toFixed(3)}σ)`);
  ok('effective bits per weight include the fp16 scales', QT.quantBitsPerWeight(4, 64) === 4.25 && QT.quantBitsPerWeight(4, Infinity) === 4);
}

// --- module 14: interpretability
const IT = run('LAB.tests.interp');
{
  const toy = run('new LAB.tests.interp.SuperToy(5, 2, 0.5, 0.7, 3)'), Xb = toy.sample(16); toy.lossGrad(Xb);
  const gW = Float64Array.from(toy.gW), gb = Float64Array.from(toy.gb); let e = 0;
  const L = () => toy.lossGrad(Xb);
  for (const [arr, g] of [[toy.W, gW], [toy.b, gb]]) for (const i of [0, 3, arr.length - 1]) { const h = 1e-6, o = arr[i]; arr[i] = o + h; const a = L(); arr[i] = o - h; const b = L(); arr[i] = o; e = Math.max(e, Math.abs((a - b) / (2 * h) - g[i])); }
  ok('superposition toy: hand-written gradients match finite differences', e < 1e-6, e.toExponential(1));
  const trainToy = S => { const t = run(`new LAB.tests.interp.SuperToy(5, 2, ${S}, 0.7, 1)`); for (let s = 0; s < 2000; s++) t.step(256); return t; };
  const alive = t => t.norms().filter(v => v > 0.35).length;
  const t0 = trainToy(0), t9 = trainToy(0.9), t97 = trainToy(0.97);
  ok('S = 0: only the 2 most important features are kept (and they are orthogonal)', alive(t0) === 2 && t0.norms()[0] > 0.8 && t0.norms()[1] > 0.8 && Math.abs(t0.gram()[0][1]) < 0.15, t0.norms().map(v => v.toFixed(2)).join(' '));
  ok('S = 0.9: more than 2 features are kept', alive(t9) >= 3, t9.norms().map(v => v.toFixed(2)).join(' '));
  const g97 = t97.gram(); ok('S = 0.97: all 5 features are represented, with non-zero interference', alive(t97) === 5 && Math.max(...g97.flatMap((r, i) => r.filter((_, j) => j !== i).map(Math.abs))) > 0.3, t97.norms().map(v => v.toFixed(2)).join(' '));
  const ang = Array.from({ length: 5 }, (_, i) => Math.atan2(t97.W[5 + i], t97.W[i]) * 180 / Math.PI).sort((a, b) => a - b), gaps = ang.map((a, i) => ((ang[(i + 1) % 5] - a) + 360) % 360);
  ok('S = 0.97: the five vectors form a near-regular pentagon (gaps ≈ 72°)', gaps.every(g => Math.abs(g - 72) < 22), gaps.map(g => g.toFixed(0)).join(' '));
}
{
  const corpus = run('LAB.tests.CORPUS'), chars = [...new Set(corpus)].sort(), data = Int32Array.from([...corpus].map(c => chars.indexOf(c))), V = chars.length;
  const trained = IT.trainCharLM(data, V, 800), fresh = run(`new LAB.tests.CharLM(${V}, 3, 8, 64, 7)`);
  const r1 = IT.probeRun(trained, fresh, data, chars, -1, IT.isVowelCh);
  ok('probe: trained net decodes "previous char is a vowel" almost perfectly', r1.trained > 0.93, JSON.stringify(Object.fromEntries(Object.entries(r1).map(([k, v]) => [k, +v.toFixed(2)]))));
  ok('probe: a randomly initialised net is already well above the baseline (information survives random projections)', r1.untrained > r1.majority + 0.15 && r1.trained > r1.untrained + 0.03);
  ok('probe: shuffled-label control stays near the majority baseline', Math.abs(r1.control - r1.majority) < 0.08);
}

// --- module 15: retrieval and agents
const RG = run('LAB.tests.rag');
{
  const A = e => RG.safeCalc(e);
  ok('calculator: precedence, parentheses, decimals, unary minus', A('2+3*4') === 14 && A('(2+3)*4') === 20 && close(A('1,5*4'), 6) && A('-3+5') === 2 && A('10/4') === 2.5 && A('2*(3+(4-1))*2') === 24);
  const bad2 = ['2+', '(1+2', '1/0', '2**3', 'abc', ''].every(e => { try { A(e); return false; } catch (err) { return true; } });
  ok('calculator rejects malformed input instead of guessing', bad2);
  const r1b = RG.ragRecall(RG.bm25Scores, 1), r3b = RG.ragRecall(RG.bm25Scores, 3), r1t = RG.ragRecall(RG.tfidfScores, 1), r3t = RG.ragRecall(RG.tfidfScores, 3);
  ok('BM25 recall@3 is 8/8 and recall@1 is 6/8 on the labelled queries', r3b === 1 && r1b === 0.75, `bm25 ${r1b * 8}/8, ${r3b * 8}/8 · tfidf ${r1t * 8}/8, ${r3t * 8}/8`);
  const hs = RG.ragRank(RG.bm25Scores(RG.KB, RG.HARD_QUERY)), goldRank = hs.findIndex(x => x[1] === 3);
  ok('lexical retrieval fails on the paraphrased query (gold paragraph not first)', goldRank > 0, 'gold rank ' + (goldRank + 1));
  const ch = RG.ragChunks('abcdefghij', 4, 1); ok('chunking: window 4, overlap 1', JSON.stringify(ch.map(c => c[1])) === JSON.stringify(['abcd', 'defg', 'ghij']));
  const run1 = RG.agentRun('Quanto fa (17 + 5) * 3?'); ok('agent: arithmetic goes to the calculator', run1[0].action[0] === 'calc' && run1[0].obs === '66');
  const run2 = RG.agentRun('Quanto pago per 4 corse da 30 minuti?'); ok('agent: search then calc gives 6 €', run2[0].action[0] === 'search' && run2[1].action[0] === 'calc' && run2[1].obs === '6', run2[run2.length - 1].answer);
  const run3 = RG.agentRun('Quanto pago per 3 corse da 60 minuti con bici elettrica?'); ok('agent: electric bike price (3 € per 30 min) -> 18 €', run3[1].obs === '18', run3[run3.length - 1].answer);
  const run4 = RG.agentRun('Chi ha vinto il campionato di calcio?'); ok('agent: refuses when nothing relevant is found', /non rispondo|Non ho trovato/.test(run4[run4.length - 1].answer));
  const run5 = RG.agentRun("Come disdico l'abbonamento?"); ok('agent: answers from the right paragraph', /Annulla rinnovo/.test(run5[0].obs) && /disdire/.test(run5[run5.length - 1].answer));
}

console.log(bad ? `\n${bad} check(s) failed` : '\nall logic checks passed');
process.exit(bad ? 1 : 0);
