'use strict';
/* App shell: builds the rail (the model stack as navigation) and shows one module at a time. */
(function () {
  const rail = $('#rail'), stage = $('#stage'), mods = LAB.modules;
  const seen = new Set();
  try { JSON.parse(localStorage.getItem('llmlab.seen') || '[]').forEach(x => seen.add(x)); } catch (e) { /* storage may be unavailable */ }

  const nodes = {}, pages = {};
  const mkNode = m => {
    const b = el('button', { class: 'node', type: 'button', 'data-id': m.id, onclick: () => show(m.id) },
      el('span', { class: 'idx', text: String(m.n) }), el('span', { text: m.title }), el('span', { class: 'seen' }));
    nodes[m.id] = b; return b;
  };

  rail.append(
    el('div', {}, el('div', { class: 'brand', html: 'LLM<span>·</span>Lab' }), el('div', { class: 'tag', text: 'come funziona un Large Language Model, un pezzo alla volta' })),
  );
  const list = el('div', { class: 'stack' });
  const first = mods.filter(m => !m.layer), inLayer = mods.filter(m => m.layer), after = mods.filter(m => !m.layer && m.n > Math.max(...inLayer.map(x => x.n)));
  const before = first.filter(m => m.n < Math.min(...inLayer.map(x => x.n)));
  before.forEach(m => list.append(mkNode(m)));
  list.append(el('div', { class: 'layers-label', text: '× N layer (ripetuto)' }));
  const lay = el('div', { class: 'layers' }); inLayer.forEach(m => lay.append(mkNode(m))); list.append(lay);
  after.forEach(m => list.append(mkNode(m)));
  rail.append(list);
  rail.append(el('div', { class: 'roadmap' }, el('b', { text: 'Prossimi moduli' }),
    el('ul', {}, ['Interpretabilità', 'RAG, tool use, agenti', 'Transformer vero addestrato nel browser'].map(t => el('li', { text: t })))));
  const th = el('div', { class: 'theme-row' });
  [['auto', 'Auto'], ['light', 'Chiaro'], ['dark', 'Scuro']].forEach(([v, t]) => th.append(button(t, () => {
    if (v === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', v);
  })));
  rail.append(th);

  mods.forEach((m, i) => {
    const page = el('article', { class: 'module', id: m.id, hidden: true, 'aria-label': m.title });
    page.append(el('div', { class: 'eyebrow', text: `Modulo ${m.n} · ${m.eyebrow}` }), el('h1', { text: m.heading || m.title }), el('p', { class: 'lede', html: m.lede }));
    m.build(page);
    const prev = mods[i - 1], next = mods[i + 1];
    page.append(el('div', { class: 'next' },
      prev ? button('← ' + prev.title, () => show(prev.id)) : el('span'),
      next ? button(next.title + ' →', () => show(next.id), 'primary') : el('span')));
    pages[m.id] = page; stage.append(page);
  });

  function show(id, fromHash) {
    if (!pages[id]) id = mods[0].id;
    mods.forEach(m => { pages[m.id].hidden = m.id !== id; nodes[m.id].setAttribute('aria-current', String(m.id === id)); });
    seen.add(id); nodes[id].classList.add('visited');
    try { localStorage.setItem('llmlab.seen', JSON.stringify([...seen])); } catch (e) { /* ignore */ }
    try { if (!fromHash) history.replaceState(null, '', '#' + id); } catch (e) { /* ignore */ }
    if (!fromHash) window.scrollTo({ top: 0 });
    window.dispatchEvent(new Event('resize'));
  }
  seen.forEach(id => nodes[id] && nodes[id].classList.add('visited'));
  window.addEventListener('hashchange', () => show(location.hash.slice(1), true));
  show(location.hash.slice(1) || mods[0].id, true);
  LAB.show = show;
})();
