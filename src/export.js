// Result exporters: COCO JSON, YOLO txt labels, per-instance mask PNGs, 16-bit depth PNG, class-id PNG.
import { ZipWriter } from './zip.js';
import { obbCorners } from './render.js';

export function downloadBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

export function canvasBlob(canvas, type = 'image/png') {
  return new Promise((resolve) => canvas.toBlob(resolve, type));
}

const round = (v, p = 2) => Math.round(v * 10 ** p) / 10 ** p;

/** COCO-style JSON for one image (detect / segment / pose / obb). */
export function toCoco(result, W, H, imageName, labels) {
  const categories = labels.map((name, id) => ({ id, name }));
  const annotations = result.detections.map((d, i) => {
    const a = { id: i + 1, image_id: 1, category_id: d.cls, score: round(d.score, 4), iscrowd: 0 };
    if (d.id) a.track_id = d.id;
    if (result.kind === 'obb') {
      const c = obbCorners(d).map(([x, y]) => [round(x), round(y)]);
      a.segmentation = [c.flat()];
      a.obb = { cx: round(d.cx), cy: round(d.cy), w: round(d.w), h: round(d.h), angle: round(d.angle, 4) };
      const xs = c.map((p) => p[0]);
      const ys = c.map((p) => p[1]);
      a.bbox = [round(Math.min(...xs)), round(Math.min(...ys)), round(Math.max(...xs) - Math.min(...xs)), round(Math.max(...ys) - Math.min(...ys))];
    } else {
      a.bbox = [round(d.x), round(d.y), round(d.w), round(d.h)];
    }
    a.area = round(a.bbox[2] * a.bbox[3]);
    if (d.kpts) {
      a.keypoints = Array.from(d.kpts).map((v, j) => (j % 3 === 2 ? (v >= 0.5 ? 2 : 0) : round(v)));
      a.num_keypoints = a.keypoints.filter((_, j) => j % 3 === 2 && a.keypoints[j] > 0).length;
    }
    return a;
  });
  return JSON.stringify(
    {
      info: { description: 'Vision Studio export', date_created: new Date().toISOString() },
      images: [{ id: 1, file_name: imageName, width: W, height: H }],
      categories,
      annotations,
    },
    null,
    2,
  );
}

/** Ultralytics-format label file: normalised `cls cx cy w h` (+ keypoints, or 4 corners for OBB). */
export function toYoloTxt(result, W, H) {
  const f = (v) => v.toFixed(6);
  return result.detections
    .map((d) => {
      if (result.kind === 'obb') {
        return d.cls + ' ' + obbCorners(d).map(([x, y]) => f(x / W) + ' ' + f(y / H)).join(' ');
      }
      let line = d.cls + ' ' + f((d.x + d.w / 2) / W) + ' ' + f((d.y + d.h / 2) / H) + ' ' + f(d.w / W) + ' ' + f(d.h / H);
      if (d.kpts) {
        for (let k = 0; k < d.kpts.length; k += 3) line += ' ' + f(d.kpts[k] / W) + ' ' + f(d.kpts[k + 1] / H) + ' ' + (d.kpts[k + 2] >= 0.5 ? 2 : 0);
      }
      return line;
    })
    .join('\n');
}

/** One full-resolution binary PNG per instance mask, zipped. */
export async function masksZip(result, W, H) {
  const { masks, detections, letterbox: lb, inputSize } = result;
  const zip = new ZipWriter();
  const plane = masks.width * masks.height;
  const small = document.createElement('canvas');
  small.width = masks.width;
  small.height = masks.height;
  const sctx = small.getContext('2d');
  const full = document.createElement('canvas');
  full.width = W;
  full.height = H;
  const fctx = full.getContext('2d');
  const scale = inputSize / masks.width;
  for (let n = 0; n < masks.count; n++) {
    const img = sctx.createImageData(masks.width, masks.height);
    for (let i = 0; i < plane; i++) {
      const v = masks.data[n * plane + i];
      img.data.set([v, v, v, 255], i * 4);
    }
    sctx.putImageData(img, 0, 0);
    fctx.fillStyle = '#000';
    fctx.fillRect(0, 0, W, H);
    fctx.imageSmoothingEnabled = false;
    fctx.drawImage(small, lb.dw / scale, lb.dh / scale, lb.newW / scale, lb.newH / scale, 0, 0, W, H);
    const d = detections[n];
    await zip.add(String(n + 1).padStart(3, '0') + '_' + d.label.replace(/\s+/g, '_') + '_' + (d.score * 100).toFixed(0) + '.png', await canvasBlob(full));
  }
  await zip.add('instances.json', JSON.stringify(detections.map((d, i) => ({ mask: i + 1, label: d.label, cls: d.cls, score: d.score, bbox: [d.x, d.y, d.w, d.h] })), null, 2));
  return zip.blob();
}

/** Semantic label map as an 8-bit PNG where every pixel value is the ADE20K class id. */
export async function labelMapPng(result) {
  const c = document.createElement('canvas');
  c.width = result.width;
  c.height = result.height;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(result.width, result.height);
  for (let i = 0; i < result.labelMap.length; i++) {
    const v = result.labelMap[i];
    img.data.set([v, v, v, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  return canvasBlob(c);
}

/* ---------- 16-bit PNG (depth) ---------- */
const CRC_TABLE = new Int32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  CRC_TABLE[n] = c;
}
function crc32(bytes) {
  let c = -1;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}
function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  out.set(new TextEncoder().encode(type), 4);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** Encode a 16-bit greyscale PNG (big-endian samples) using the browser's deflate. */
export async function encodePng16(width, height, samples) {
  const raw = new Uint8Array(height * (1 + width * 2));
  for (let y = 0, p = 0, i = 0; y < height; y++) {
    raw[p++] = 0; // filter: none
    for (let x = 0; x < width; x++, i++) {
      raw[p++] = samples[i] >> 8;
      raw[p++] = samples[i] & 0xff;
    }
  }
  const cs = new CompressionStream('deflate');
  const writer = cs.writable.getWriter();
  writer.write(raw);
  writer.close();
  const idat = new Uint8Array(await new Response(cs.readable).arrayBuffer());
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = 16; // bit depth
  ihdr[9] = 0; // greyscale
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', new Uint8Array(0))], { type: 'image/png' });
}

/** Depth Anything relative inverse depth → 16-bit PNG (0 = farthest, 65535 = nearest). */
export function depthPng16(result) {
  const { raw, width, height, min, max } = result;
  const range = max - min || 1;
  const samples = new Uint16Array(raw.length);
  for (let i = 0; i < raw.length; i++) samples[i] = Math.round(((raw[i] - min) / range) * 65535);
  return encodePng16(width, height, samples);
}
