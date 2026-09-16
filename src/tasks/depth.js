import { resizeNormalize } from '../image.js';
import { createSession, fetchModel, loadOrt } from '../runtime.js';

const HF = (size) => 'https://huggingface.co/onnx-community/depth-anything-v2-' + size + '/resolve/main/onnx/';
const FILE = { fp16: 'model_fp16.onnx', fp32: 'model.onnx', int8: 'model_quantized.onnx' };
const BASE = 518; // shorter side, multiple of 14 (DPT processor: keep_aspect_ratio, ensure_multiple_of=14)
const MAX_LONG = 1036; // cap the longer side to keep inference time sane

/** Depth Anything V2 (small) – relative inverse depth map. */
export class DepthTask {
  constructor() {
    this.session = null;
    this.backend = null;
    this.requested = null;
    this.variant = null;
  }

  /** variant = "[small|base]:[auto|fp16|fp32|int8]" (either part may be omitted). */
  pickVariant(backend, variant) {
    const [a, b] = String(variant || 'auto').split(':');
    const size = ['small', 'base'].includes(a) ? a : 'small';
    let precision = ['small', 'base'].includes(a) ? b : a;
    if (!precision || precision === 'auto') precision = backend === 'webgpu' ? 'fp16' : 'int8';
    return size + ':' + precision;
  }

  async load(backend, variant, onProgress) {
    const v = this.pickVariant(backend, variant);
    if (this.session && this.variant === v && this.requested === backend) return;
    const [size, precision] = v.split(':');
    const buffer = await fetchModel(HF(size) + FILE[precision], onProgress);
    const { session, backend: used } = await createSession(buffer, backend);
    this.session = session;
    this.backend = used;
    this.requested = backend;
    this.variant = v;
    await this.warmup();
  }

  async warmup() {
    const ort = await loadOrt();
    const dummy = new ort.Tensor('float32', new Float32Array(3 * BASE * BASE), [1, 3, BASE, BASE]);
    await this.session.run({ [this.session.inputNames[0]]: dummy });
  }

  /** Match the DPT processor: shorter side → 518, aspect kept, both dims a multiple of 14. */
  inputSize(srcW, srcH, fast) {
    let scale = (fast ? 364 : BASE) / Math.min(srcW, srcH);
    if (Math.max(srcW, srcH) * scale > MAX_LONG) scale = MAX_LONG / Math.max(srcW, srcH);
    const r14 = (v) => Math.max(14, Math.round((v * scale) / 14) * 14);
    return { w: r14(srcW), h: r14(srcH) };
  }

  async run(source, srcW, srcH, { fast = false } = {}) {
    const ort = await loadOrt();
    const t0 = performance.now();
    const { w, h } = this.inputSize(srcW, srcH, fast);
    const data = resizeNormalize(source, w, h, srcW, srcH);
    const input = new ort.Tensor('float32', data, [1, 3, h, w]);
    const t1 = performance.now();
    const out = await this.session.run({ [this.session.inputNames[0]]: input });
    const t2 = performance.now();
    const depth = out[this.session.outputNames[0]];
    const [, oh, ow] = depth.dims;
    const d = depth.data;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < d.length; i++) {
      if (d[i] < min) min = d[i];
      if (d[i] > max) max = d[i];
    }
    // Quantise to 8 bits here so the UI only has to apply a colormap.
    const range = max - min || 1;
    const norm = new Uint8Array(d.length);
    for (let i = 0; i < d.length; i++) norm[i] = ((d[i] - min) / range) * 255;
    const raw = new Float32Array(d); // own copy so it can be transferred (used by 16-bit export & 3D view)
    const t3 = performance.now();
    return {
      kind: 'depth',
      depth: norm,
      raw,
      width: ow,
      height: oh,
      min,
      max,
      timing: { pre: t1 - t0, infer: t2 - t1, post: t3 - t2, total: t3 - t0 },
      transfer: [norm.buffer, raw.buffer],
    };
  }
}
