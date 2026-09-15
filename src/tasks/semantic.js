import { resizeNormalize } from '../image.js';
import { createSession, fetchModel, loadOrt } from '../runtime.js';

const HF = 'https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/';
const MODEL_FILES = {
  fp16: HF + 'onnx/model_fp16.onnx',
  fp32: HF + 'onnx/model.onnx',
  int8: HF + 'onnx/model_quantized.onnx',
};
const INPUT = 512;

/** SegFormer-B0 fine-tuned on ADE20K (150 scene classes) – per-pixel semantic labels. */
export class SemanticTask {
  constructor() {
    this.session = null;
    this.backend = null;
    this.requested = null;
    this.variant = null;
    this.labels = null;
  }

  pickVariant(backend, variant) {
    if (variant && variant !== 'auto') return variant;
    return backend === 'webgpu' ? 'fp16' : 'fp32';
  }

  async load(backend, variant, onProgress) {
    const v = this.pickVariant(backend, variant);
    if (!this.labels) {
      const cfg = await fetch(HF + 'config.json').then((r) => r.json());
      this.labels = Object.entries(cfg.id2label)
        .sort((a, b) => Number(a[0]) - Number(b[0]))
        .map(([, name]) => name.trim());
    }
    if (this.session && this.variant === v && this.requested === backend) return;
    const buffer = await fetchModel(MODEL_FILES[v], onProgress);
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

  async run(source, srcW, srcH, { fast = false } = {}) {
    const ort = await loadOrt();
    const t0 = performance.now();
    const data = resizeNormalize(source, INPUT, INPUT, srcW, srcH);
    const input = new ort.Tensor('float32', data, [1, 3, INPUT, INPUT]);
    const t1 = performance.now();
    const out = await this.session.run({ [this.session.inputNames[0]]: input });
    const t2 = performance.now();
    const logits = out[this.session.outputNames[0]];
    const [, C, lh, lw] = logits.dims;
    const up = fast ? 1 : 4; // bilinear upsample factor before argmax (128 → 512)
    const { labelMap, width, height } = argmaxUpsampled(logits.data, C, lh, lw, up);

    const counts = new Uint32Array(C);
    for (let i = 0; i < labelMap.length; i++) counts[labelMap[i]]++;
    const t3 = performance.now();
    return {
      labelMap,
      width,
      height,
      counts,
      labels: this.labels,
      timing: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 },
      transfer: [labelMap.buffer, counts.buffer],
    };
  }
}

/**
 * Bilinearly upsample logits by `up` and take the per-pixel argmax.
 * Only the argmax classes of the 4 neighbouring low-res pixels are candidates,
 * which is ~40× cheaper than scanning all 150 classes and visually identical.
 */
function argmaxUpsampled(L, C, lh, lw, up) {
  const plane = lh * lw;
  const low = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    let best = -Infinity;
    let bc = 0;
    for (let c = 0; c < C; c++) {
      const v = L[c * plane + i];
      if (v > best) {
        best = v;
        bc = c;
      }
    }
    low[i] = bc;
  }
  if (up === 1) return { labelMap: low, width: lw, height: lh };

  const width = lw * up;
  const height = lh * up;
  const labelMap = new Uint8Array(width * height);
  const cand = new Uint8Array(4);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(lh - 1, Math.max(0, (y + 0.5) / up - 0.5));
    const y0 = Math.floor(sy);
    const y1 = Math.min(lh - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = Math.min(lw - 1, Math.max(0, (x + 0.5) / up - 0.5));
      const x0 = Math.floor(sx);
      const x1 = Math.min(lw - 1, x0 + 1);
      const fx = sx - x0;
      const i00 = y0 * lw + x0;
      const i01 = y0 * lw + x1;
      const i10 = y1 * lw + x0;
      const i11 = y1 * lw + x1;
      cand[0] = low[i00];
      cand[1] = low[i01];
      cand[2] = low[i10];
      cand[3] = low[i11];
      if (cand[0] === cand[1] && cand[0] === cand[2] && cand[0] === cand[3]) {
        labelMap[y * width + x] = cand[0];
        continue;
      }
      const w00 = (1 - fy) * (1 - fx);
      const w01 = (1 - fy) * fx;
      const w10 = fy * (1 - fx);
      const w11 = fy * fx;
      let best = -Infinity;
      let bc = cand[0];
      for (let k = 0; k < 4; k++) {
        const c = cand[k];
        if (k > 0 && (c === cand[0] || (k > 1 && c === cand[1]) || (k > 2 && c === cand[2]))) continue;
        const base = c * plane;
        const v = w00 * L[base + i00] + w01 * L[base + i01] + w10 * L[base + i10] + w11 * L[base + i11];
        if (v > best) {
          best = v;
          bc = c;
        }
      }
      labelMap[y * width + x] = bc;
    }
  }
  return { labelMap, width, height };
}
