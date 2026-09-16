// Lightweight box editor over detection results: select, move, resize, add (drag on empty space), delete.
// Mutates result.detections in place so every export picks up the corrections.
import { classColor, rgbCss } from './labels.js';

const HANDLE = 8; // CSS px

export class BoxEditor {
  /**
   * @param {object} o
   * @param {HTMLCanvasElement} o.canvas
   * @param {() => object|null} o.result  current result (must have `detections` with x/y/w/h)
   * @param {() => void} o.render         redraws the base image + overlays; editor draws on top afterwards
   * @param {() => void} o.onChange       called after any edit (to refresh lists / stats)
   * @param {() => string[]} o.labels     class names for the active model
   */
  constructor({ canvas, result, render, onChange, labels }) {
    this.canvas = canvas;
    this.getResult = result;
    this.rerender = render;
    this.onChange = onChange;
    this.getLabels = labels;
    this.active = false;
    this.selected = -1;
    this.drag = null; // { mode: 'move'|'resize'|'create', handle, start, orig }
    this.lastCls = 0;
    this.onDown = this.onDown.bind(this);
    this.onMove = this.onMove.bind(this);
    this.onUp = this.onUp.bind(this);
    this.onKey = this.onKey.bind(this);
  }

  enable(on) {
    this.active = on;
    this.selected = -1;
    this.drag = null;
    const c = this.canvas;
    const m = on ? 'addEventListener' : 'removeEventListener';
    c[m]('pointerdown', this.onDown);
    c[m]('pointermove', this.onMove);
    c[m]('pointerup', this.onUp);
    window[m]('keydown', this.onKey);
    c.style.cursor = on ? 'crosshair' : '';
    this.rerender();
  }

  scale() {
    const r = this.canvas.getBoundingClientRect();
    return this.canvas.width / r.width; // canvas px per CSS px
  }

  point(e) {
    const r = this.canvas.getBoundingClientRect();
    const s = this.scale();
    return { x: (e.clientX - r.left) * s, y: (e.clientY - r.top) * s };
  }

  boxes() {
    return this.getResult()?.detections?.filter((d) => d.x !== undefined) || [];
  }

  /** Which of the 8 handles of the selected box is under p (or null). */
  handleAt(p) {
    const d = this.boxes()[this.selected];
    if (!d) return null;
    const h = HANDLE * this.scale();
    const xs = { l: d.x, c: d.x + d.w / 2, r: d.x + d.w };
    const ys = { t: d.y, m: d.y + d.h / 2, b: d.y + d.h };
    for (const [hy, y] of Object.entries(ys)) for (const [hx, x] of Object.entries(xs)) {
      if (hx === 'c' && hy === 'm') continue;
      if (Math.abs(p.x - x) <= h && Math.abs(p.y - y) <= h) return hx + hy;
    }
    return null;
  }

  hit(p) {
    const boxes = this.boxes();
    let best = -1;
    let area = Infinity;
    boxes.forEach((d, i) => {
      if (p.x >= d.x && p.x <= d.x + d.w && p.y >= d.y && p.y <= d.y + d.h && d.w * d.h < area) {
        best = i;
        area = d.w * d.h;
      }
    });
    return best;
  }

  onDown(e) {
    if (!this.active || e.button !== 0) return;
    const p = this.point(e);
    const handle = this.handleAt(p);
    const boxes = this.boxes();
    if (handle) {
      this.drag = { mode: 'resize', handle, start: p, orig: { ...boxes[this.selected] } };
    } else {
      const i = this.hit(p);
      if (i >= 0) {
        this.selected = i;
        this.drag = { mode: 'move', start: p, orig: { ...boxes[i] } };
      } else {
        this.selected = -1;
        this.drag = { mode: 'create', start: p };
      }
    }
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {}
    e.preventDefault();
    this.rerender();
  }

  onMove(e) {
    if (!this.active) return;
    const p = this.point(e);
    if (!this.drag) {
      const h = this.handleAt(p);
      this.canvas.style.cursor = h ? cursorFor(h) : this.hit(p) >= 0 ? 'move' : 'crosshair';
      return;
    }
    const boxes = this.boxes();
    const W = this.canvas.width;
    const H = this.canvas.height;
    const dx = p.x - this.drag.start.x;
    const dy = p.y - this.drag.start.y;
    if (this.drag.mode === 'move') {
      const d = boxes[this.selected];
      const o = this.drag.orig;
      d.x = clamp(o.x + dx, 0, W - d.w);
      d.y = clamp(o.y + dy, 0, H - d.h);
    } else if (this.drag.mode === 'resize') {
      const d = boxes[this.selected];
      const o = this.drag.orig;
      const h = this.drag.handle;
      let x1 = o.x;
      let y1 = o.y;
      let x2 = o.x + o.w;
      let y2 = o.y + o.h;
      if (h[0] === 'l') x1 = clamp(o.x + dx, 0, x2 - 4);
      if (h[0] === 'r') x2 = clamp(o.x + o.w + dx, x1 + 4, W);
      if (h[1] === 't') y1 = clamp(o.y + dy, 0, y2 - 4);
      if (h[1] === 'b') y2 = clamp(o.y + o.h + dy, y1 + 4, H);
      d.x = x1;
      d.y = y1;
      d.w = x2 - x1;
      d.h = y2 - y1;
    } else if (this.drag.mode === 'create') {
      this.drag.rect = { x: Math.min(p.x, this.drag.start.x), y: Math.min(p.y, this.drag.start.y), w: Math.abs(dx), h: Math.abs(dy) };
    }
    this.rerender();
  }

  onUp() {
    if (!this.active || !this.drag) return;
    const d = this.drag;
    this.drag = null;
    if (d.mode === 'create' && d.rect && d.rect.w > 6 && d.rect.h > 6) {
      const labels = this.getLabels();
      const cls = Math.min(this.lastCls, labels.length - 1);
      const r = this.getResult();
      r.detections.push({ ...d.rect, cls, name: labels[cls], label: labels[cls], score: 1, manual: true });
      this.selected = r.detections.length - 1;
    }
    this.onChange();
    this.rerender();
  }

  onKey(e) {
    if (!this.active) return;
    if ((e.key === 'Delete' || e.key === 'Backspace') && this.selected >= 0 && !isTyping(e)) {
      this.deleteSelected();
      e.preventDefault();
    } else if (e.key === 'Escape') {
      this.selected = -1;
      this.rerender();
    }
  }

  deleteSelected() {
    const r = this.getResult();
    if (!r || this.selected < 0) return;
    const target = this.boxes()[this.selected];
    r.detections.splice(r.detections.indexOf(target), 1);
    if (r.masks) r.masks = null; // masks no longer line up with the instance list
    this.selected = -1;
    this.onChange();
    this.rerender();
  }

  setClass(cls) {
    const d = this.boxes()[this.selected];
    const labels = this.getLabels();
    this.lastCls = cls;
    if (!d) return;
    d.cls = cls;
    d.name = labels[cls];
    d.label = labels[cls];
    this.onChange();
    this.rerender();
  }

  /** Draw selection handles (call after the normal overlay). */
  draw(ctx) {
    if (!this.active) return;
    const s = this.scale();
    const d = this.boxes()[this.selected];
    if (d) {
      ctx.save();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2 * s;
      ctx.setLineDash([6 * s, 4 * s]);
      ctx.strokeRect(d.x, d.y, d.w, d.h);
      ctx.setLineDash([]);
      ctx.fillStyle = rgbCss(classColor(d.cls));
      const h = HANDLE * s;
      for (const x of [d.x, d.x + d.w / 2, d.x + d.w]) for (const y of [d.y, d.y + d.h / 2, d.y + d.h]) {
        if (x === d.x + d.w / 2 && y === d.y + d.h / 2) continue;
        ctx.fillRect(x - h / 2, y - h / 2, h, h);
        ctx.strokeRect(x - h / 2, y - h / 2, h, h);
      }
      ctx.restore();
    }
    if (this.drag?.mode === 'create' && this.drag.rect) {
      const r = this.drag.rect;
      ctx.save();
      ctx.strokeStyle = '#64e0c8';
      ctx.lineWidth = 2 * s;
      ctx.setLineDash([6 * s, 4 * s]);
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.restore();
    }
  }
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function cursorFor(h) {
  return { lt: 'nwse-resize', rb: 'nwse-resize', rt: 'nesw-resize', lb: 'nesw-resize', ct: 'ns-resize', cb: 'ns-resize', lm: 'ew-resize', rm: 'ew-resize' }[h] || 'move';
}

function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
}
