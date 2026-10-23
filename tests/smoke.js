'use strict';
/* Smoke test: execute the whole app in a fake DOM, visit every module, poke every control, pump animations. */
const fs = require('fs'), path = require('path'), vm = require('vm');
const { makeEnv, FNode } = require('./fakedom');
const root = path.join(__dirname, '..');
const order = fs.readFileSync(path.join(root, 'build.py'), 'utf8').match(/order = \[([^\]]*)\]/s)[1].match(/"([^"]+)"/g).map(s => s.replace(/"/g, ''));
const env = makeEnv();
const ctx = vm.createContext(env.win);
let failures = 0;
const fail = (where, e) => { failures++; console.log('FAIL', where, '\n  ', ((e && e.message) || '') + '\n   ' + ((e && e.stack) || e).toString().split('\n').slice(0, 5).join('\n   ')); };

for (const f of order) {
  const p = path.join(root, 'src', 'js', f);
  if (!fs.existsSync(p)) continue;
  try { vm.runInContext(fs.readFileSync(p, 'utf8'), ctx, { filename: f }); } catch (e) { fail('load ' + f, e); }
}
const run = code => vm.runInContext(code, ctx);
const mods = run('LAB.modules.map(m => m.id)');
console.log('modules registered:', mods.join(', '));

function poke(node, where) {
  node.walk(n => {
    for (const [type, fns] of Object.entries(n.listeners)) {
      const tag = n.tagName, attrs = n.attrs;
      const fire = () => fns.forEach(fn => { try { fn({ type, target: n, clientX: 100, clientY: 100, pointerId: 1, preventDefault() {} }); } catch (e) { fail(`${where} <${tag} ${attrs.id || attrs.class || ''}> ${type}`, e); } });
      if (type === 'input' && attrs.type === 'range') {
        const lo = +attrs.min, hi = +attrs.max, st = +(attrs.step || 1);
        for (const v of [hi, lo, Math.round((lo + hi) / 2 / st) * st, lo + st]) { n.value = String(v); fire(); }
      } else if (type === 'change' && tag === 'SELECT') {
        n.children.forEach(o => { n.value = o.attrs.value; fire(); });
      } else if (type === 'change' && attrs.type === 'checkbox') { n.checked = !n.checked; fire(); n.checked = !n.checked; fire(); }
      else if (type === 'click') { if (!/play|avvia|auto/i.test(n.textContent || '')) fire(); else fire(); }
      else if (type === 'input') { fire(); }
    }
  });
}
function pumpFrames(n = 60) { for (let i = 0; i < n; i++) { const cbs = env.raf.splice(0); cbs.forEach(cb => { try { cb(performance.now()); } catch (e) { fail('raf', e); } }); } }

for (const id of mods) {
  try { run(`LAB.show(${JSON.stringify(id)})`); } catch (e) { fail('show ' + id, e); continue; }
  env.ro.forEach(cb => { try { cb([]); } catch (e) { fail('redraw ' + id, e); } });
  const page = env.stage.querySelectorAll('article').find(a => a.id === id) || env.stage.children.find(c => c.id === id);
  if (!page) { fail('page ' + id, 'not found'); continue; }
  poke(page, id);
  pumpFrames(80);
  env.ro.forEach(cb => { try { cb([]); } catch (e) { fail('redraw2 ' + id, e); } });
  console.log('visited', id);
}
// theme change redraw
try { run('fireTheme()'); } catch (e) { fail('fireTheme', e); }
console.log(failures ? `\n${failures} failure(s)` : '\nsmoke test passed');
process.exit(failures ? 1 : 0);
