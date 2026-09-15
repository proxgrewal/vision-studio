import { YoloTask } from './yolo.js';
import { createSession, fetchModel, loadOrt } from '../runtime.js';

const MAX_CLASSES = 32; // the exported head has 32 class slots; unused slots get zero embeddings
const modelUrl = (file) => new URL('../../models/' + file, import.meta.url).href;
const CLIP_HF = 'https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/';
const CLIP_TEXT = { q4f16: CLIP_HF + 'onnx/text_model_q4f16.onnx', fp16: CLIP_HF + 'onnx/text_model_fp16.onnx' };
const TRANSFORMERS_JS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';

/**
 * YOLO-World v2 (S) open-vocabulary detection. The ONNX export takes the CLIP text embeddings
 * of the prompts as a second input, so any phrase works: common nouns come from a precomputed
 * 1.7k-word vocabulary; anything else is encoded on demand with the CLIP ViT-B/32 text tower.
 */
export class WorldTask extends YoloTask {
  constructor() {
    super({ kind: 'world', size: 640, names: [], files: { s: 'yolov8s-worldv2.onnx' } });
    this.vocab = null; // { names: string[], index: Map<string, number>, embeddings: Float32Array (count×512) }
    this.textEncoder = null;
    this.tokenizer = null;
    this.cache = new Map(); // prompt → Float32Array(512)
    this.txtFeats = null;
    this.prompts = [];
  }

  pickVariant() {
    return 's';
  }

  async load(backend, variant, onProgress) {
    if (!this.vocab) {
      const [meta, bin] = await Promise.all([fetch(modelUrl('world-vocab.json')).then((r) => r.json()), fetch(modelUrl('world-vocab.bin')).then((r) => r.arrayBuffer())]);
      const f16 = new Uint16Array(bin);
      const embeddings = new Float32Array(f16.length);
      for (let i = 0; i < f16.length; i++) embeddings[i] = f16ToF32(f16[i]);
      this.vocab = { names: meta.names, index: new Map(meta.names.map((n, i) => [n, i])), embeddings, dim: meta.dim };
    }
    if (!this.prompts.length) await this.setPrompts(['person', 'car', 'dog', 'cat', 'bottle']);
    await super.load(backend, 's', onProgress);
  }

  /** Any request that needs the CLIP text encoder loads it (and the tokenizer) lazily. */
  async ensureTextEncoder(onProgress) {
    if (this.textEncoder) return;
    const [{ AutoTokenizer, env }, buffer] = await Promise.all([import(TRANSFORMERS_JS), fetchModel(CLIP_TEXT.q4f16, onProgress)]);
    env.allowLocalModels = false;
    this.tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32');
    this.textEncoder = (await createSession(buffer, 'wasm')).session; // small, runs in ~50 ms on CPU
  }

  async embed(prompts, onProgress) {
    const out = new Array(prompts.length);
    const missing = [];
    prompts.forEach((p, i) => {
      const key = normalize(p);
      const vi = this.vocab.index.get(key);
      if (this.cache.has(key)) out[i] = this.cache.get(key);
      else if (vi !== undefined) out[i] = this.vocab.embeddings.subarray(vi * this.vocab.dim, (vi + 1) * this.vocab.dim);
      else missing.push(i);
    });
    if (missing.length) {
      await this.ensureTextEncoder(onProgress);
      const ort = await loadOrt();
      const texts = missing.map((i) => normalize(prompts[i]));
      const enc = this.tokenizer(texts, { padding: 'max_length', max_length: 77, truncation: true });
      const ids = new ort.Tensor('int64', BigInt64Array.from(enc.input_ids.data, (v) => BigInt(v)), enc.input_ids.dims);
      const res = await this.textEncoder.run({ input_ids: ids });
      const emb = res[this.textEncoder.outputNames[0]];
      const dim = emb.dims[1];
      missing.forEach((pi, k) => {
        const v = new Float32Array(emb.data.subarray(k * dim, (k + 1) * dim));
        let n = 0;
        for (const x of v) n += x * x;
        n = Math.sqrt(n) || 1;
        for (let j = 0; j < dim; j++) v[j] /= n;
        this.cache.set(normalize(prompts[pi]), v);
        out[pi] = v;
      });
    }
    return out;
  }

  /** Set the classes to detect (≤ 32 free-text prompts). Returns which prompts needed the text encoder. */
  async setPrompts(prompts, onProgress) {
    const clean = [...new Set(prompts.map((p) => p.trim()).filter(Boolean))].slice(0, MAX_CLASSES);
    if (!clean.length) throw new Error('Enter at least one prompt');
    const embs = await this.embed(clean, onProgress);
    const feats = new Float32Array(MAX_CLASSES * 512);
    embs.forEach((e, i) => feats.set(e, i * 512));
    this.txtFeats = feats;
    this.prompts = clean;
    this.labels = clean;
    this.resetTracker();
    return { prompts: clean, fromVocab: clean.filter((p) => this.vocab.index.has(normalize(p))) };
  }

  async extraInputs() {
    const ort = await loadOrt();
    return { txt_feats: new ort.Tensor('float32', this.txtFeats || new Float32Array(MAX_CLASSES * 512), [1, MAX_CLASSES, 512]) };
  }
}

function normalize(s) {
  return s.trim().toLowerCase().replace(/\s+/g, ' ');
}

function f16ToF32(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + m / 1024);
}
