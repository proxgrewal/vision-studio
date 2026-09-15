import { createSession, fetchModel, loadOrt } from '../runtime.js';

const HF = 'https://huggingface.co/Xenova/slimsam-77-uniform/resolve/main/onnx/';
const FILES = {
  fp32: { enc: HF + 'vision_encoder.onnx', dec: HF + 'prompt_encoder_mask_decoder.onnx' },
  fp16: { enc: HF + 'vision_encoder_fp16.onnx', dec: HF + 'prompt_encoder_mask_decoder.onnx' },
  int8: { enc: HF + 'vision_encoder_quantized.onnx', dec: HF + 'prompt_encoder_mask_decoder_quantized.onnx' },
};
const SIZE = 1024; // longest edge, then zero-padded to 1024×1024
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const scratch = new OffscreenCanvas(SIZE, SIZE);
const sctx = scratch.getContext('2d', { willReadFrequently: true });

/**
 * SlimSAM (Segment Anything, 77% pruned ViT-B): encode an image once, then decode a mask per click set.
 * The encoder output stays in the worker; only 256×256 masks travel back to the page.
 */
export class SamTask {
  constructor() {
    this.kind = 'sam';
    this.encoder = null;
    this.decoder = null;
    this.backend = null;
    this.requested = null;
    this.variant = null;
    this.embedding = null; // { image_embeddings, image_positional_embeddings, scale, newW, newH, srcW, srcH }
    this.labels = null;
  }

  /** The worker treats a task as loaded when it has a `session`. */
  get session() {
    return this.encoder;
  }

  pickVariant(backend, variant) {
    if (variant && variant !== 'auto') return variant;
    return backend === 'webgpu' ? 'fp16' : 'int8';
  }

  async load(backend, variant, onProgress) {
    const v = this.pickVariant(backend, variant);
    if (this.encoder && this.variant === v && this.requested === backend) return;
    const files = FILES[v];
    const encBuf = await fetchModel(files.enc, onProgress);
    const decBuf = await fetchModel(files.dec, onProgress);
    const enc = await createSession(encBuf, backend);
    // The mask decoder is tiny and full of dynamic shapes; it is faster and more robust on WASM.
    const dec = await createSession(decBuf, 'wasm');
    this.encoder = enc.session;
    this.decoder = dec.session;
    this.backend = enc.backend;
    this.requested = backend;
    this.variant = v;
    this.embedding = null;
  }

  /** Standard task entry point: encodes the image so subsequent `prompt` calls are cheap. */
  async run(source, srcW, srcH) {
    const ort = await loadOrt();
    const t0 = performance.now();
    const scale = SIZE / Math.max(srcW, srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    sctx.clearRect(0, 0, SIZE, SIZE);
    sctx.drawImage(source, 0, 0, srcW, srcH, 0, 0, newW, newH);
    const { data } = sctx.getImageData(0, 0, SIZE, SIZE);
    const plane = SIZE * SIZE;
    const tensor = new Float32Array(3 * plane); // padding stays 0 (= normalised pad value)
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const i = y * SIZE + x;
        const p = i * 4;
        tensor[i] = (data[p] / 255 - MEAN[0]) / STD[0];
        tensor[i + plane] = (data[p + 1] / 255 - MEAN[1]) / STD[1];
        tensor[i + 2 * plane] = (data[p + 2] / 255 - MEAN[2]) / STD[2];
      }
    }
    const t1 = performance.now();
    const out = await this.encoder.run({ pixel_values: new ort.Tensor('float32', tensor, [1, 3, SIZE, SIZE]) });
    const t2 = performance.now();
    this.embedding = { emb: out.image_embeddings, pos: out.image_positional_embeddings, scale, newW, newH, srcW, srcH };
    return { kind: 'sam', embedded: true, points: [], masks: [], timing: { pre: t1 - t0, infer: t2 - t1, post: 0, total: t2 - t0 } };
  }

  /**
   * Decode a mask for a set of clicks (in source-image pixels; label 1 = foreground, 0 = background).
   * Returns the best of SAM's 3 candidate masks as a 256×256 Uint8 plane covering the padded 1024 square.
   */
  async prompt(points) {
    if (!this.embedding) throw new Error('Encode an image first');
    const ort = await loadOrt();
    const e = this.embedding;
    const t0 = performance.now();
    const n = points.length;
    const coords = new Float32Array(n * 2);
    const labels = new BigInt64Array(n);
    points.forEach((p, i) => {
      coords[i * 2] = p.x * e.scale;
      coords[i * 2 + 1] = p.y * e.scale;
      labels[i] = BigInt(p.label);
    });
    const out = await this.decoder.run({
      input_points: new ort.Tensor('float32', coords, [1, 1, n, 2]),
      input_labels: new ort.Tensor('int64', labels, [1, 1, n]),
      image_embeddings: e.emb,
      image_positional_embeddings: e.pos,
    });
    const scores = out.iou_scores.data;
    let best = 0;
    for (let i = 1; i < 3; i++) if (scores[i] > scores[best]) best = i;
    const [, , , mh, mw] = out.pred_masks.dims;
    const plane = mh * mw;
    const logits = out.pred_masks.data.subarray(best * plane, (best + 1) * plane);
    const mask = new Uint8Array(plane);
    for (let i = 0; i < plane; i++) if (logits[i] > 0) mask[i] = 255;
    const t1 = performance.now();
    return {
      mask,
      width: mw,
      height: mh,
      iou: scores[best],
      // Region of the 256×256 mask that maps onto the source image (rest is padding).
      cropW: (e.newW / SIZE) * mw,
      cropH: (e.newH / SIZE) * mh,
      timing: { infer: t1 - t0, total: t1 - t0 },
      transfer: [mask.buffer],
    };
  }
}
