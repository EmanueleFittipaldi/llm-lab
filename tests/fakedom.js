'use strict';
/* A tiny fake DOM, just enough to execute the app's JavaScript in plain Node (no browser, no dependencies).
   It catches runtime errors (typos, undefined variables, bad math in draw code); it does not test real layout. */
class FNode {
  constructor(tag, type = 1) {
    this.tagName = (tag || '').toUpperCase(); this.nodeType = type; this.children = []; this.attrs = {}; this.listeners = {};
    this.style = {}; this.dataset = {}; this.value = ''; this.checked = false; this.hidden = false; this.disabled = false; this.className = '';
    this.textContent = ''; this.parent = null; this._html = '';
    const set = new Set(); const self = this;
    this.classList = { add: c => { set.add(c); self.className = [...set].join(' '); }, remove: c => { set.delete(c); self.className = [...set].join(' '); }, contains: c => set.has(c) || self.className.split(/\s+/).includes(c), toggle: c => set.has(c) ? set.delete(c) : set.add(c) };
  }
  set innerHTML(v) { this._html = v; this.children = []; } get innerHTML() { return this._html; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'value') this.value = String(v); if (k === 'class') this.className = String(v); if (k === 'hidden') this.hidden = true; if (k === 'checked') this.checked = true; if (k === 'id') this.id = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  removeAttribute(k) { delete this.attrs[k]; }
  append(...k) { for (const c of k) { const n = typeof c === 'string' ? new FNode('#text', 3) : c; if (n.parent && n.parent !== this) { /* moved */ } n.parent = this; this.children.push(n); } }
  appendChild(c) { this.append(c); return c; }
  replaceChildren(...k) { this.children = []; this.append(...k); }
  addEventListener(t, fn) { (this.listeners[t] ||= []).push(fn); }
  removeEventListener() {} dispatchEvent() { return true; }
  walk(fn) { fn(this); this.children.forEach(c => c.walk && c.walk(fn)); }
  querySelectorAll(sel) { const out = []; this.walk(n => { if (n !== this && match(n, sel)) out.push(n); }); return out; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  closest() { return this; }
  getBoundingClientRect() { return { width: 600, height: parseFloat(this.style.height) || 300, left: 0, top: 0, right: 600, bottom: 300 }; }
  getClientRects() { return [1]; }
  get offsetParent() { return this.parent || { }; }
  getContext() { return ctxStub; }
  setPointerCapture() {} focus() {} scrollIntoView() {}
  get options() { return this.children; }
}
function match(n, sel) {
  if (sel.startsWith('#')) return n.id === sel.slice(1);
  if (sel.startsWith('.')) return n.className.split(/\s+/).includes(sel.slice(1));
  return n.tagName === sel.toUpperCase();
}
const ctxStub = new Proxy({ canvas: {} }, {
  get(t, k) { if (k in t) return t[k]; if (k === 'measureText') return () => ({ width: 10 }); if (k === 'createLinearGradient' || k === 'createRadialGradient') return () => ({ addColorStop() {} }); return () => undefined; },
  set(t, k, v) { t[k] = v; return true; },
});

function makeEnv() {
  const root = new FNode('html'), body = new FNode('body'), rail = new FNode('nav'), stage = new FNode('main');
  rail.id = 'rail'; stage.id = 'stage'; body.append(rail, stage);
  const ro = [], raf = [], winL = {}, errors = [];
  const document = {
    documentElement: root, body, createElement: t => new FNode(t), createTextNode: s => { const n = new FNode('#text', 3); n.textContent = String(s); return n; },
    querySelector: s => s === '#rail' ? rail : s === '#stage' ? stage : null, querySelectorAll: () => [],
  };
  const win = {
    document, devicePixelRatio: 1, innerWidth: 1000, innerHeight: 800, matchMedia: () => ({ matches: false, addEventListener() {} }),
    addEventListener: (t, f) => (winL[t] ||= []).push(f), dispatchEvent: e => (winL[e.type] || []).forEach(f => f(e)),
    getComputedStyle: () => ({ getPropertyValue: () => '#3a6bd6' }), scrollTo() {}, requestAnimationFrame: f => { raf.push(f); return raf.length; }, cancelAnimationFrame() {},
    ResizeObserver: class { constructor(cb) { this.cb = cb; ro.push(cb); } observe() { Promise.resolve().then(() => this.cb([])); } },
    MutationObserver: class { observe() {} },
    Event: class { constructor(t) { this.type = t; } },
    localStorage: { getItem: () => null, setItem() {} }, history: { replaceState() {} }, location: { hash: '' }, performance: { now: () => Date.now() },
    console, Math, JSON, Date, Number, String, Array, Object, Map, Set, Float64Array, Float32Array, Int32Array, Uint8Array, Promise, Symbol, Proxy, RegExp, Error, TypeError, setTimeout, clearTimeout,
    CSS: { supports: () => true },
  };
  win.window = win; win.self = win; win.globalThis = win;
  return { win, document, rail, stage, ro, raf, errors, FNode };
}
module.exports = { makeEnv, FNode };
