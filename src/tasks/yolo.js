import { centerCrop, letterbox } from '../image.js';
import { COCO_CLASSES, DOTA_CLASSES } from '../labels.js';
import { createSession, fetchModel, loadOrt } from '../runtime.js';
import { readOnnxMeta } from '../onnxmeta.js';

/** Built-in Ultralytics YOLO11 ONNX exports in /models. */
export const YOLO_KINDS = {
  detect: { kind: 'detect', size: 640, names: COCO_CLASSES, files: { n: 'yolo11n.onnx', s: 'yolo11s.onnx', m: 'yolo11m.onnx' } },
  segment: { kind: 'segment', size: 640, names: COCO_CLASSES, files: { n: 'yolo11n-seg.onnx', s: 'yolo11s-seg.onnx', m: 'yolo11m-seg.onnx' } },
  pose: { kind: 'pose', size: 640, names: ['person'], kptShape: [17, 3], files: { n: 'yolo11n-pose.onnx', s: 'yolo11s-pose.onnx' } },
  obb: { kind: 'obb', size: 1024, names: DOTA_CLASSES, files: { n: 'yolo11n-obb.onnx', s: 'yolo11s-obb.onnx' } },
  classify: { kind: 'classify', size: 224, names: null, namesFile: 'imagenet-names.json', files: { n: 'yolo11n-cls.onnx', s: 'yolo11s-cls.onnx' } },
};

/** Model files live in /models next to /src by default; library users can point elsewhere (e.g. the hosted copy). */
let MODEL_BASE = new URL('../../models/', import.meta.url).href;
export function setModelBase(url) {
  MODEL_BASE = url.endsWith('/') ? url : url + '/';
}
export const modelUrl = (file) => MODEL_BASE + file;

/**
 * YOLO11 on ONNX Runtime Web (any Ultralytics export: detect / segment / pose / obb / classify).
 *  detect   : output0 [1, 4+nc, A]
 *  segment  : output0 [1, 4+nc+32, A] + output1 [1, 32, H/4, W/4] mask protos
 *  pose     : output0 [1, 4+nc+K*3, A]  (keypoints x, y, visibility)
 *  obb      : output0 [1, 4+nc+1, A]     (last channel = rotation angle in radians)
 *  classify : output0 [1, nc] softmax probabilities
 */
export class YoloTask {
  /** @param cfg one of YOLO_KINDS, or a custom config { kind, size, names, kptShape?, buffer, label } */
  constructor(cfg) {
    this.cfg = { ...cfg };
    this.kind = cfg.kind;
    this.session = null;
    this.backend = null;
    this.requested = null;
    this.variant = null;
    this.labels = cfg.names;
    this.tracker = null;
  }

  pickVariant(backend, variant) {
    if (this.cfg.buffer) return 'custom';
    return this.cfg.files[variant] ? variant : 'n';
  }

  async load(backend, variant, onProgress) {
    const v = this.pickVariant(backend, variant);
    if (!this.labels && this.cfg.namesFile) this.labels = await fetch(modelUrl(this.cfg.namesFile)).then((r) => r.json());
    if (this.session && this.variant === v && this.requested === backend) return;
    const buffer = this.cfg.buffer || (await fetchModel(modelUrl(this.cfg.files[v]), onProgress));
    const { session, backend: used } = await createSession(buffer, backend);
    this.session = session;
    this.backend = used;
    this.requested = backend;
    this.variant = v;
    await this.warmup();
  }

  /** Additional session inputs (YOLO-World feeds text embeddings here). */
  async extraInputs() {
    return {};
  }

  async warmup() {
    const ort = await loadOrt();
    const s = this.cfg.size;
    const dummy = new ort.Tensor('float32', new Float32Array(3 * s * s), [1, 3, s, s]);
    await this.session.run({ [this.session.inputNames[0]]: dummy, ...(await this.extraInputs()) });
  }

  async run(source, srcW, srcH, opts = {}) {
    if (this.kind === 'classify') return this.runClassify(source, srcW, srcH, opts);
    return this.runDetect(source, srcW, srcH, opts);
  }

  async runClassify(source, srcW, srcH, { topK = 5 } = {}) {
    const ort = await loadOrt();
    const size = this.cfg.size;
    const t0 = performance.now();
    const input = new ort.Tensor('float32', centerCrop(source, size, srcW, srcH), [1, 3, size, size]);
    const t1 = performance.now();
    const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
    const t2 = performance.now();
    const probs = outputs[this.session.outputNames[0]].data;
    const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, topK);
    const t3 = performance.now();
    return {
      kind: 'classify',
      classes: idx.map((i) => ({ cls: i, label: (this.labels?.[i] || 'class ' + i).replace(/_/g, ' '), score: probs[i] })),
      timing: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 },
    };
  }

  async runDetect(source, srcW, srcH, { conf = 0.25, iou = 0.45, maxDet = 300, track = false } = {}) {
    const ort = await loadOrt();
    const size = this.cfg.size;
    const names = this.labels;
    const nc = names.length;
    const isObb = this.kind === 'obb';
    const isPose = this.kind === 'pose';
    const kptN = isPose ? this.cfg.kptShape[0] : 0;
    const kptDim = isPose ? this.cfg.kptShape[1] : 0;
    const t0 = performance.now();
    const lb = letterbox(source, size, srcW, srcH);
    const input = new ort.Tensor('float32', lb.tensor, [1, 3, size, size]);
    const extra = await this.extraInputs();
    const t1 = performance.now();
    const outputs = await this.session.run({ [this.session.inputNames[0]]: input, ...extra });
    const t2 = performance.now();

    const out0 = outputs[this.session.outputNames[0]];
    const [, channels, anchors] = out0.dims;
    const data = out0.data;
    const maskDim = this.kind === 'segment' ? channels - 4 - nc : 0;

    // Decode candidates above the confidence threshold.
    const cands = [];
    for (let a = 0; a < anchors; a++) {
      let best = 0;
      let bestCls = -1;
      for (let c = 0; c < nc; c++) {
        const s = data[(4 + c) * anchors + a];
        if (s > best) {
          best = s;
          bestCls = c;
        }
      }
      if (best < conf) continue;
      const cx = data[a];
      const cy = data[anchors + a];
      const w = data[2 * anchors + a];
      const h = data[3 * anchors + a];
      const angle = isObb ? data[(channels - 1) * anchors + a] : 0;
      cands.push({ cx, cy, w, h, angle, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls: bestCls, anchor: a });
    }

    const kept = nms(cands, iou, maxDet, isObb);

    // Map boxes (and keypoints) from letterbox space back to source pixels.
    const detections = kept.map((d) => {
      const base = { score: d.score, cls: d.cls, name: names[d.cls], label: names[d.cls] };
      if (isObb) {
        return { ...base, cx: (d.cx - lb.dw) / lb.ratio, cy: (d.cy - lb.dh) / lb.ratio, w: d.w / lb.ratio, h: d.h / lb.ratio, angle: d.angle };
      }
      const x = clamp((d.x1 - lb.dw) / lb.ratio, 0, srcW);
      const y = clamp((d.y1 - lb.dh) / lb.ratio, 0, srcH);
      const x2 = clamp((d.x2 - lb.dw) / lb.ratio, 0, srcW);
      const y2 = clamp((d.y2 - lb.dh) / lb.ratio, 0, srcH);
      const det = { ...base, x, y, w: x2 - x, h: y2 - y };
      if (isPose) {
        const kpts = new Float32Array(kptN * 3);
        for (let k = 0; k < kptN; k++) {
          const off = (4 + nc + k * kptDim) * anchors + d.anchor;
          kpts[k * 3] = (data[off] - lb.dw) / lb.ratio;
          kpts[k * 3 + 1] = (data[off + anchors] - lb.dh) / lb.ratio;
          kpts[k * 3 + 2] = kptDim > 2 ? data[off + 2 * anchors] : 1;
        }
        det.kpts = kpts;
      }
      return det;
    });

    if (track) {
      if (!this.tracker) this.tracker = new Tracker();
      this.tracker.update(detections, isObb);
    } else {
      this.tracker = null;
    }

    let masks = null;
    if (maskDim > 0 && detections.length) {
      const protos = outputs[this.session.outputNames[1]];
      masks = buildMasks(kept, data, anchors, nc, maskDim, protos, size);
    }
    const t3 = performance.now();

    return {
      kind: this.kind,
      detections,
      masks,
      inputSize: size,
      kptShape: this.cfg.kptShape || null,
      letterbox: { ratio: lb.ratio, dw: lb.dw, dh: lb.dh, newW: lb.newW, newH: lb.newH },
      timing: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 },
      transfer: masks ? [masks.data.buffer] : [],
    };
  }

  resetTracker() {
    this.tracker = null;
  }
}

/** Build a YoloTask from a user-supplied Ultralytics ONNX export by reading its embedded metadata. */
export function customYoloTask(buffer, label) {
  const meta = readOnnxMeta(buffer);
  const task = (meta.task || 'detect').toLowerCase();
  const kind = { detect: 'detect', segment: 'segment', pose: 'pose', obb: 'obb', classify: 'classify' }[task];
  if (!kind) throw new Error('Unsupported task "' + task + '" – expected an Ultralytics detect/segment/pose/obb/classify export');
  if (meta.end2end === 'True') throw new Error('End-to-end (NMS-fused) exports are not supported; export with nms=False');
  let names = null;
  if (meta.names) {
    const obj = parsePyDict(meta.names);
    names = Object.keys(obj).sort((a, b) => a - b).map((k) => obj[k]);
  }
  let size = meta.inputShape?.[3] || 640;
  if (!size && meta.imgsz) size = JSON.parse(meta.imgsz)[1];
  if (kind === 'classify' && !meta.inputShape?.[3]) size = 224;
  const kptShape = meta.kpt_shape ? JSON.parse(meta.kpt_shape) : kind === 'pose' ? [17, 3] : undefined;
  if (!names) throw new Error('No class names in ONNX metadata – export with Ultralytics ≥ 8.1');
  return new YoloTask({ kind, size, names, kptShape, buffer, label, meta });
}

/** Parse Ultralytics' python-dict `names` string, e.g. "{0: 'person', 1: 'bike'}". */
function parsePyDict(str) {
  const out = {};
  const re = /(\d+):\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")/g;
  let m;
  while ((m = re.exec(str))) out[m[1]] = (m[2] ?? m[3]).replace(/\\(.)/g, '$1');
  return out;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Greedy per-class NMS: only boxes of the same class suppress each other. */
function nms(cands, iouThr, maxDet, rotated) {
  cands.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of cands) {
    let suppressed = false;
    for (const k of kept) {
      if (k.cls !== c.cls) continue;
      if ((rotated ? probiou(c, k) : boxIou(c, k)) > iouThr) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push(c);
      if (kept.length >= maxDet) break;
    }
  }
  return kept;
}

export function boxIou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

/** Gaussian-distribution covariance of a rotated box (Ultralytics `_get_covariance_matrix`). */
function covariance(o) {
  const a = (o.w * o.w) / 12;
  const b = (o.h * o.h) / 12;
  const cos = Math.cos(o.angle);
  const sin = Math.sin(o.angle);
  return [a * cos * cos + b * sin * sin, a * sin * sin + b * cos * cos, (a - b) * cos * sin];
}

/** Probabilistic IoU between two oriented boxes (Ultralytics `probiou`, Bhattacharyya distance). */
export function probiou(o1, o2, eps = 1e-7) {
  const [a1, b1, c1] = covariance(o1);
  const [a2, b2, c2] = covariance(o2);
  const dx = o1.cx - o2.cx;
  const dy = o1.cy - o2.cy;
  const denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps;
  const t1 = (((a1 + a2) * dy * dy + (b1 + b2) * dx * dx) / denom) * 0.25;
  const t2 = (((c1 + c2) * -dx * dy) / denom) * 0.5;
  const det1 = Math.max(0, a1 * b1 - c1 * c1);
  const det2 = Math.max(0, a2 * b2 - c2 * c2);
  const t3 = Math.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / (4 * Math.sqrt(det1 * det2) + eps) + eps) * 0.5;
  const bd = Math.min(100, Math.max(eps, t1 + t2 + t3));
  const hd = Math.sqrt(1 - Math.exp(-bd) + eps);
  return 1 - hd;
}

/**
 * mask = sigmoid(coeffs · protos) > 0.5, cropped to the box.
 * Returns one binary plane per instance packed as [N, ph, pw] Uint8 (0/255).
 */
function buildMasks(kept, data, anchors, nc, maskDim, protos, inputSize) {
  const [, pc, ph, pw] = protos.dims;
  const P = protos.data;
  const plane = ph * pw;
  const scale = inputSize / pw;
  const out = new Uint8Array(kept.length * plane);
  const coeffs = new Float32Array(maskDim);
  for (let n = 0; n < kept.length; n++) {
    const d = kept[n];
    for (let m = 0; m < maskDim; m++) coeffs[m] = data[(4 + nc + m) * anchors + d.anchor];
    const bx1 = Math.max(0, Math.floor(d.x1 / scale));
    const by1 = Math.max(0, Math.floor(d.y1 / scale));
    const bx2 = Math.min(pw, Math.ceil(d.x2 / scale));
    const by2 = Math.min(ph, Math.ceil(d.y2 / scale));
    const base = n * plane;
    for (let y = by1; y < by2; y++) {
      for (let x = bx1; x < bx2; x++) {
        const idx = y * pw + x;
        let v = 0;
        for (let m = 0; m < pc; m++) v += coeffs[m] * P[m * plane + idx];
        if (v > 0) out[base + idx] = 255; // sigmoid(v) > 0.5  ⇔  v > 0
      }
    }
  }
  return { data: out, count: kept.length, width: pw, height: ph };
}

/* ---------------- tracking ---------------- */

/**
 * ByteTrack-style multi-object tracker (IoU association in two passes: high-score
 * detections first, then low-score ones against still-unmatched tracks). Constant-velocity
 * prediction, no Kalman filter – plenty for browser frame rates. Assigns `id` to each detection.
 */
class Tracker {
  constructor({ highThr = 0.5, matchThr = 0.3, maxLost = 30, minHits = 2 } = {}) {
    this.highThr = highThr;
    this.matchThr = matchThr;
    this.maxLost = maxLost;
    this.minHits = minHits;
    this.tracks = [];
    this.nextId = 1;
  }

  update(dets, rotated) {
    const toBox = (d) => (rotated ? { x1: d.cx - d.w / 2, y1: d.cy - d.h / 2, x2: d.cx + d.w / 2, y2: d.cy + d.h / 2 } : { x1: d.x, y1: d.y, x2: d.x + d.w, y2: d.y + d.h });
    // Predict.
    for (const t of this.tracks) {
      t.box = { x1: t.box.x1 + t.vx, y1: t.box.y1 + t.vy, x2: t.box.x2 + t.vx, y2: t.box.y2 + t.vy };
      t.lost++;
    }
    const boxes = dets.map(toBox);
    const high = [];
    const low = [];
    dets.forEach((d, i) => (d.score >= this.highThr ? high : low).push(i));
    const unmatchedTracks = new Set(this.tracks.map((_, i) => i));
    const assign = (detIdx) => {
      const pairs = [];
      for (const ti of unmatchedTracks) for (const di of detIdx) {
        if (this.tracks[ti].cls !== dets[di].cls) continue;
        const iou = boxIou(this.tracks[ti].box, boxes[di]);
        if (iou >= this.matchThr) pairs.push([iou, ti, di]);
      }
      pairs.sort((a, b) => b[0] - a[0]);
      const usedDet = new Set();
      for (const [, ti, di] of pairs) {
        if (!unmatchedTracks.has(ti) || usedDet.has(di)) continue;
        this.match(this.tracks[ti], dets[di], boxes[di]);
        unmatchedTracks.delete(ti);
        usedDet.add(di);
      }
      return detIdx.filter((di) => !usedDet.has(di));
    };
    const leftHigh = assign(high);
    assign(low);
    // New tracks from unmatched high-confidence detections.
    for (const di of leftHigh) {
      this.tracks.push({ id: this.nextId++, cls: dets[di].cls, box: boxes[di], vx: 0, vy: 0, lost: 0, hits: 1, trail: [center(boxes[di])] });
      dets[di].id = this.tracks[this.tracks.length - 1].id;
      dets[di].trail = this.tracks[this.tracks.length - 1].trail;
    }
    this.tracks = this.tracks.filter((t) => t.lost <= this.maxLost);
    // Only expose ids of tracks that have been seen at least minHits times (suppresses flicker).
    for (const d of dets) {
      const t = this.tracks.find((t) => t.id === d.id);
      if (t && t.hits < this.minHits) {
        delete d.id;
        delete d.trail;
      }
    }
  }

  match(t, det, box) {
    const c0 = center(t.box);
    const c1 = center(box);
    t.vx = 0.7 * t.vx + 0.3 * (c1[0] - c0[0]);
    t.vy = 0.7 * t.vy + 0.3 * (c1[1] - c0[1]);
    t.box = box;
    t.lost = 0;
    t.hits++;
    t.trail.push(c1);
    if (t.trail.length > 40) t.trail.shift();
    det.id = t.id;
    det.trail = t.trail;
  }
}

function center(b) {
  return [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2];
}
