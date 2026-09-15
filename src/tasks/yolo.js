import { letterbox } from '../image.js';
import { COCO_CLASSES } from '../labels.js';
import { createSession, fetchModel, loadOrt } from '../runtime.js';

export const INPUT = 640;
const NUM_CLASSES = COCO_CLASSES.length;

const MODEL_FILES = {
  detect: { n: 'models/yolo11n.onnx', s: 'models/yolo11s.onnx' },
  segment: { n: 'models/yolo11n-seg.onnx', s: 'models/yolo11s-seg.onnx' },
};

/**
 * YOLO11 detection / instance segmentation on ONNX Runtime Web.
 * Output layout: [1, 4 + 80 (+32 mask coeffs), 8400] plus [1, 32, 160, 160] protos for -seg.
 */
export class YoloTask {
  constructor(kind, baseUrl) {
    this.kind = kind; // 'detect' | 'segment'
    this.baseUrl = baseUrl;
    this.session = null;
    this.backend = null;
    this.requested = null;
    this.variant = null;
  }

  pickVariant(backend, variant) {
    return variant === 's' ? 's' : 'n';
  }

  async load(backend, variant, onProgress) {
    const v = this.pickVariant(backend, variant);
    if (this.session && this.variant === v && this.requested === backend) return;
    const buffer = await fetchModel(new URL(MODEL_FILES[this.kind][v], this.baseUrl).href, onProgress);
    const { session, backend: used } = await createSession(buffer, backend);
    this.session = session;
    this.backend = used;
    this.requested = backend;
    this.variant = v;
    await this.warmup();
  }

  async warmup() {
    const ort = await loadOrt();
    const dummy = new ort.Tensor('float32', new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]);
    await this.session.run({ [this.session.inputNames[0]]: dummy });
  }

  async run(source, srcW, srcH, { conf = 0.25, iou = 0.45, maxDet = 100 } = {}) {
    const ort = await loadOrt();
    const t0 = performance.now();
    const lb = letterbox(source, INPUT, srcW, srcH);
    const input = new ort.Tensor('float32', lb.tensor, [1, 3, INPUT, INPUT]);
    const t1 = performance.now();
    const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
    const t2 = performance.now();

    const out0 = outputs[this.session.outputNames[0]];
    const [, channels, anchors] = out0.dims;
    const data = out0.data;
    const maskDim = channels - 4 - NUM_CLASSES;

    // Decode candidates above the confidence threshold.
    const cands = [];
    for (let a = 0; a < anchors; a++) {
      let best = 0;
      let bestCls = -1;
      for (let c = 0; c < NUM_CLASSES; c++) {
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
      cands.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls: bestCls, anchor: a });
    }

    const kept = nms(cands, iou, maxDet);

    // Map boxes from letterbox space back to source pixels.
    const detections = kept.map((d) => {
      const x = clamp((d.x1 - lb.dw) / lb.ratio, 0, srcW);
      const y = clamp((d.y1 - lb.dh) / lb.ratio, 0, srcH);
      const x2 = clamp((d.x2 - lb.dw) / lb.ratio, 0, srcW);
      const y2 = clamp((d.y2 - lb.dh) / lb.ratio, 0, srcH);
      return { x, y, w: x2 - x, h: y2 - y, score: d.score, cls: d.cls, label: COCO_CLASSES[d.cls] };
    });

    let masks = null;
    if (this.kind === 'segment' && maskDim > 0 && detections.length) {
      const protos = outputs[this.session.outputNames[1]];
      masks = buildMasks(kept, data, anchors, maskDim, protos);
    }
    const t3 = performance.now();

    return {
      detections,
      masks,
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
function nms(cands, iouThr, maxDet) {
  cands.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of cands) {
    const area = (c.x2 - c.x1) * (c.y2 - c.y1);
    let suppressed = false;
    for (const k of kept) {
      if (k.cls !== c.cls) continue;
      const ix = Math.max(0, Math.min(c.x2, k.x2) - Math.max(c.x1, k.x1));
      const iy = Math.max(0, Math.min(c.y2, k.y2) - Math.max(c.y1, k.y1));
      const inter = ix * iy;
      const union = area + (k.x2 - k.x1) * (k.y2 - k.y1) - inter;
      if (inter / union > iouThr) {
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

/**
 * mask = sigmoid(coeffs · protos) > 0.5, cropped to the box.
 * Returns one binary plane per instance packed as [N, ph, pw] Uint8 (0/255).
 */
function buildMasks(kept, data, anchors, maskDim, protos) {
  const [, pc, ph, pw] = protos.dims;
  const P = protos.data;
  const plane = ph * pw;
  const scale = INPUT / pw;
  const out = new Uint8Array(kept.length * plane);
  const coeffs = new Float32Array(maskDim);
  for (let n = 0; n < kept.length; n++) {
    const d = kept[n];
    for (let m = 0; m < maskDim; m++) coeffs[m] = data[(4 + NUM_CLASSES + m) * anchors + d.anchor];
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
