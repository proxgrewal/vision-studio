// Main-thread rendering of worker results onto the output canvas, plus legend summaries.
import { classColor, rgbCss } from './labels.js';

export const COLORMAPS = {
  inferno: [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]],
  magma: [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]],
  viridis: [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]],
  turbo: [[48, 18, 59], [70, 107, 227], [39, 179, 232], [30, 226, 172], [122, 250, 82], [201, 240, 52], [252, 190, 33], [246, 105, 12], [201, 43, 4], [122, 4, 3]],
  gray: [[0, 0, 0], [255, 255, 255]],
};

function buildLut(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(t));
    const f = t - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * f;
  }
  return lut;
}
const LUTS = Object.fromEntries(Object.entries(COLORMAPS).map(([k, v]) => [k, buildLut(v)]));

const scratch = document.createElement('canvas');
const sctx = scratch.getContext('2d');

function luminance([r, g, b]) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/* ---------- YOLO detect / segment ---------- */
export function drawYolo(ctx, result, W, H, { showLabels = true, maskOpacity = 0.5 } = {}) {
  const { detections, masks, letterbox: lb } = result;
  if (masks && masks.count) {
    const { width: pw, height: ph, data } = masks;
    const plane = pw * ph;
    const scale = result.inputSize / pw;
    scratch.width = pw;
    scratch.height = ph;
    const img = sctx.createImageData(pw, ph);
    const px = img.data;
    ctx.save();
    ctx.globalAlpha = maskOpacity;
    ctx.imageSmoothingEnabled = true;
    for (let n = 0; n < masks.count; n++) {
      const col = classColor(detections[n].cls);
      px.fill(0);
      for (let i = 0, base = n * plane; i < plane; i++) {
        if (data[base + i]) {
          const o = i * 4;
          px[o] = col[0];
          px[o + 1] = col[1];
          px[o + 2] = col[2];
          px[o + 3] = 255;
        }
      }
      sctx.putImageData(img, 0, 0);
      ctx.drawImage(scratch, lb.dw / scale, lb.dh / scale, lb.newW / scale, lb.newH / scale, 0, 0, W, H);
    }
    ctx.restore();
  }
  const { lw, fontPx } = setupStroke(ctx, W, H);
  for (const d of detections) {
    const color = classColor(d.cls);
    ctx.strokeStyle = rgbCss(color);
    ctx.strokeRect(d.x, d.y, d.w, d.h);
    if (showLabels) drawTag(ctx, d.label + ' ' + (d.score * 100).toFixed(0) + '%', d.x - lw / 2, d.y, color, fontPx);
  }
}

function setupStroke(ctx, W, H) {
  const lw = Math.max(1.5, Math.min(W, H) / 300);
  const fontPx = Math.max(12, Math.round(Math.min(W, H) / 40));
  ctx.font = '600 ' + fontPx + 'px system-ui, sans-serif';
  ctx.textBaseline = 'top';
  ctx.lineWidth = lw;
  ctx.lineJoin = 'round';
  return { lw, fontPx };
}

/** Filled label chip whose bottom-left corner sits at (x, y), flipped below when it would clip the top. */
function drawTag(ctx, text, x, y, color, fontPx) {
  const tw = ctx.measureText(text).width + fontPx * 0.6;
  const th = fontPx * 1.4;
  const ty = y - th >= 0 ? y - th : y;
  ctx.fillStyle = rgbCss(color);
  ctx.fillRect(x, ty, tw, th);
  ctx.fillStyle = luminance(color) > 140 ? '#111' : '#fff';
  ctx.fillText(text, x + fontPx * 0.3, ty + fontPx * 0.2);
}

/* ---------- YOLO oriented boxes ---------- */
export function obbCorners({ cx, cy, w, h, angle }) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const hw = w / 2;
  const hh = h / 2;
  return [
    [cx - hw * cos + hh * sin, cy - hw * sin - hh * cos],
    [cx + hw * cos + hh * sin, cy + hw * sin - hh * cos],
    [cx + hw * cos - hh * sin, cy + hw * sin + hh * cos],
    [cx - hw * cos - hh * sin, cy - hw * sin + hh * cos],
  ];
}

export function drawObb(ctx, result, W, H, { showLabels = true, maskOpacity = 0.5 } = {}) {
  const { lw, fontPx } = setupStroke(ctx, W, H);
  for (const d of result.detections) {
    const color = classColor(d.cls);
    const pts = obbCorners(d);
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fillStyle = rgbCss(color, maskOpacity * 0.35);
    ctx.fill();
    ctx.strokeStyle = rgbCss(color);
    ctx.stroke();
    if (showLabels && result.detections.length <= 60) {
      const top = pts.reduce((a, p) => (p[1] < a[1] ? p : a));
      drawTag(ctx, d.label + ' ' + (d.score * 100).toFixed(0) + '%', top[0] - lw / 2, top[1], color, fontPx);
    }
  }
}

/* ---------- classification ---------- */
export function drawClassify(ctx, result, W, H) {
  const top = result.classes[0];
  if (!top) return;
  const fontPx = Math.max(14, Math.round(Math.min(W, H) / 22));
  ctx.font = '700 ' + fontPx + 'px system-ui, sans-serif';
  ctx.textBaseline = 'top';
  const text = top.label + '  ' + (top.score * 100).toFixed(1) + '%';
  const pad = fontPx * 0.5;
  const tw = ctx.measureText(text).width + pad * 2;
  const th = fontPx * 1.5;
  ctx.fillStyle = 'rgba(8,9,14,0.78)';
  ctx.fillRect(pad, H - th - pad, tw, th);
  ctx.fillStyle = '#64e0c8';
  ctx.fillText(text, pad * 2, H - th - pad + fontPx * 0.25);
}

export function summaryClassify(result) {
  return result.classes.map((c, i) => ({ label: c.label, color: i === 0 ? [100, 224, 200] : null, value: (c.score * 100).toFixed(1) + '%' }));
}

export function summaryYolo(result) {
  const counts = new Map();
  for (const d of result.detections) counts.set(d.cls, (counts.get(d.cls) || 0) + 1);
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([cls, n]) => ({ label: result.detections.find((d) => d.cls === cls).label, color: classColor(cls), value: '×' + n }));
}

/* ---------- semantic ---------- */
export function drawSemantic(ctx, result, W, H, { maskOpacity = 0.6 } = {}) {
  const { labelMap, width, height } = result;
  scratch.width = width;
  scratch.height = height;
  const img = sctx.createImageData(width, height);
  const px = img.data;
  const palette = [];
  for (let i = 0, o = 0; i < labelMap.length; i++, o += 4) {
    const cls = labelMap[i];
    const col = palette[cls] || (palette[cls] = classColor(cls));
    px[o] = col[0];
    px[o + 1] = col[1];
    px[o + 2] = col[2];
    px[o + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  ctx.save();
  ctx.globalAlpha = maskOpacity;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(scratch, 0, 0, width, height, 0, 0, W, H);
  ctx.restore();
}

export function summarySemantic(result) {
  const total = result.labelMap.length;
  const rows = [];
  for (let c = 0; c < result.counts.length; c++) {
    if (!result.counts[c]) continue;
    rows.push({ label: result.labels?.[c] ?? 'class ' + c, color: classColor(c), pct: result.counts[c] / total });
  }
  rows.sort((a, b) => b.pct - a.pct);
  return rows.filter((r) => r.pct >= 0.002).map((r) => ({ label: r.label, color: r.color, value: (r.pct * 100).toFixed(1) + '%' }));
}

/* ---------- depth ---------- */
export function drawDepth(ctx, result, W, H, { colormap = 'inferno', blend = 1, invert = false } = {}) {
  const { depth, width, height } = result;
  const lut = LUTS[colormap] || LUTS.inferno;
  scratch.width = width;
  scratch.height = height;
  const img = sctx.createImageData(width, height);
  const px = img.data;
  for (let i = 0, o = 0; i < depth.length; i++, o += 4) {
    const k = (invert ? 255 - depth[i] : depth[i]) * 3;
    px[o] = lut[k];
    px[o + 1] = lut[k + 1];
    px[o + 2] = lut[k + 2];
    px[o + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  ctx.save();
  ctx.globalAlpha = blend;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(scratch, 0, 0, width, height, 0, 0, W, H);
  ctx.restore();
}

export function summaryDepth(result) {
  return [
    { label: 'Model input', value: result.width + '×' + result.height },
    { label: 'Relative depth range', value: result.min.toFixed(2) + ' – ' + result.max.toFixed(2) },
  ];
}

export const RENDERERS = {
  detect: { draw: drawYolo, summary: summaryYolo },
  segment: { draw: drawYolo, summary: summaryYolo },
  obb: { draw: drawObb, summary: summaryYolo },
  classify: { draw: drawClassify, summary: summaryClassify },
  semantic: { draw: drawSemantic, summary: summarySemantic },
  depth: { draw: drawDepth, summary: summaryDepth },
};
