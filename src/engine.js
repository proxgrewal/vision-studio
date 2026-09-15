// Promise-based client for the inference worker.
export class Engine {
  constructor() {
    this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.seq = 0;
    this.worker.onmessage = (e) => {
      const { id, type, ...rest } = e.data;
      const p = this.pending.get(id);
      if (!p) return;
      if (type === 'progress') {
        p.onProgress?.(rest);
        return;
      }
      this.pending.delete(id);
      if (type === 'error') p.reject(new Error(rest.message));
      else p.resolve(rest);
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(e.message || 'worker crashed'));
      this.pending.clear();
    };
  }

  call(msg, { transfer = [], onProgress } = {}) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, ...msg }, transfer);
    });
  }

  /** Resolves with the auto-detected backend ('webgpu' | 'wasm'). */
  init() {
    return this.call({ type: 'init' }).then((r) => r.backend);
  }

  load(task, backend, variant, onProgress) {
    return this.call({ type: 'load', task, backend, variant }, { onProgress });
  }

  /** `source` is anything createImageBitmap accepts (img, canvas, video). */
  async run(task, source, opts) {
    const bitmap = await createImageBitmap(source);
    const { result } = await this.call({ type: 'run', task, bitmap, opts }, { transfer: [bitmap] });
    return result;
  }

  /** Load a user-supplied Ultralytics ONNX export (ArrayBuffer is transferred to the worker). */
  loadCustom(buffer, label, backend, onProgress) {
    return this.call({ type: 'loadCustom', buffer, label, backend }, { transfer: [buffer], onProgress });
  }

  /** SAM: decode a mask for click points [{x, y, label}] on the image encoded by the last run('sam', …). */
  samPrompt(points) {
    return this.call({ type: 'samPrompt', points }).then((r) => r.result);
  }

  resetTracker() {
    return this.call({ type: 'resetTracker' });
  }

  clearCache() {
    return this.call({ type: 'clearCache' });
  }
}
