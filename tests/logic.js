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

console.log(bad ? `\n${bad} check(s) failed` : '\nall logic checks passed');
process.exit(bad ? 1 : 0);
