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

console.log(bad ? `\n${bad} check(s) failed` : '\nall logic checks passed');
process.exit(bad ? 1 : 0);
