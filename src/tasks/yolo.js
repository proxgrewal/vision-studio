import { centerCrop, letterbox } from '../image.js';
import { COCO_CLASSES, DOTA_CLASSES } from '../labels.js';
import { createSession, fetchModel, loadOrt } from '../runtime.js';

/** Per-head configuration for the Ultralytics YOLO11 ONNX exports in /models. */
const KINDS = {
  detect: { files: { n: 'models/yolo11n.onnx', s: 'models/yolo11s.onnx' }, size: 640, names: COCO_CLASSES },
  segment: { files: { n: 'models/yolo11n-seg.onnx', s: 'models/yolo11s-seg.onnx' }, size: 640, names: COCO_CLASSES },
  obb: { files: { n: 'models/yolo11n-obb.onnx', s: 'models/yolo11s-obb.onnx' }, size: 1024, names: DOTA_CLASSES },
  classify: { files: { n: 'models/yolo11n-cls.onnx', s: 'models/yolo11s-cls.onnx' }, size: 224, names: null, namesFile: 'models/imagenet-names.json' },
};

/**
 * YOLO11 on ONNX Runtime Web.
 *  detect   : output0 [1, 4+80, 8400]
 *  segment  : output0 [1, 4+80+32, 8400] + output1 [1, 32, 160, 160] mask protos
 *  obb      : output0 [1, 4+15+1, 21504]  (last channel = rotation angle in radians)
 *  classify : output0 [1, 1000] softmax probabilities
 */
export class YoloTask {
  constructor(kind, baseUrl) {
    this.kind = kind;
    this.cfg = KINDS[kind];
    this.baseUrl = baseUrl;
    this.session = null;
    this.backend = null;
    this.requested = null;
    this.variant = null;
    this.labels = this.cfg.names;
  }

  pickVariant(backend, variant) {
    return variant === 's' ? 's' : 'n';
  }

  async load(backend, variant, onProgress) {
    const v = this.pickVariant(backend, variant);
    if (!this.labels) this.labels = await fetch(new URL(this.cfg.namesFile, this.baseUrl)).then((r) => r.json());
    if (this.session && this.variant === v && this.requested === backend) return;
    const buffer = await fetchModel(new URL(this.cfg.files[v], this.baseUrl).href, onProgress);
    const { session, backend: used } = await createSession(buffer, backend);
    this.session = session;
    this.backend = used;
    this.requested = backend;
    this.variant = v;
    await this.warmup();
  }

  async warmup() {
    const ort = await loadOrt();
    const s = this.cfg.size;
    const dummy = new ort.Tensor('float32', new Float32Array(3 * s * s), [1, 3, s, s]);
    await this.session.run({ [this.session.inputNames[0]]: dummy });
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
      classes: idx.map((i) => ({ cls: i, label: (this.labels[i] || 'class ' + i).replace(/_/g, ' '), score: probs[i] })),
      timing: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 },
    };
  }

  async runDetect(source, srcW, srcH, { conf = 0.25, iou = 0.45, maxDet = 300 } = {}) {
    const ort = await loadOrt();
    const size = this.cfg.size;
    const names = this.labels;
    const nc = names.length;
    const isObb = this.kind === 'obb';
    const t0 = performance.now();
    const lb = letterbox(source, size, srcW, srcH);
    const input = new ort.Tensor('float32', lb.tensor, [1, 3, size, size]);
    const t1 = performance.now();
    const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
    const t2 = performance.now();

    const out0 = outputs[this.session.outputNames[0]];
    const [, channels, anchors] = out0.dims;
    const data = out0.data;
    const maskDim = isObb ? 0 : channels - 4 - nc;

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

    // Map boxes from letterbox space back to source pixels.
    const detections = kept.map((d) => {
      const base = { score: d.score, cls: d.cls, label: names[d.cls] };
      if (isObb) {
        return { ...base, cx: (d.cx - lb.dw) / lb.ratio, cy: (d.cy - lb.dh) / lb.ratio, w: d.w / lb.ratio, h: d.h / lb.ratio, angle: d.angle };
      }
      const x = clamp((d.x1 - lb.dw) / lb.ratio, 0, srcW);
      const y = clamp((d.y1 - lb.dh) / lb.ratio, 0, srcH);
      const x2 = clamp((d.x2 - lb.dw) / lb.ratio, 0, srcW);
      const y2 = clamp((d.y2 - lb.dh) / lb.ratio, 0, srcH);
      return { ...base, x, y, w: x2 - x, h: y2 - y };
    });

    let masks = null;
    if (this.kind === 'segment' && maskDim > 0 && detections.length) {
      const protos = outputs[this.session.outputNames[1]];
      masks = buildMasks(kept, data, anchors, nc, maskDim, protos, size);
    }
    const t3 = performance.now();

    return {
      detections,
      masks,
      inputSize: size,
      letterbox: { ratio: lb.ratio, dw: lb.dw, dh: lb.dh, newW: lb.newW, newH: lb.newH },
      timing: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 },
      transfer: masks ? [masks.data.buffer] : [],
    };
  }
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

function boxIou(a, b) {
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
function probiou(o1, o2, eps = 1e-7) {
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
