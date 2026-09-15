// ONNX Runtime Web bootstrap: backend selection, cached model downloads, session creation.
const ORT_VERSION = '1.30.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ortPromise = null;
export async function loadOrt() {
  if (!ortPromise) {
    ortPromise = import(`${ORT_CDN}ort.webgpu.min.mjs`).then((ort) => {
      ort.env.wasm.wasmPaths = ORT_CDN;
      // Multi-threading needs cross-origin isolation, which static hosts rarely provide.
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 2) - 1)) : 1;
      return ort;
    });
  }
  return ortPromise;
}

export async function detectBackend() {
  if (!('gpu' in navigator)) return 'wasm';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? 'webgpu' : 'wasm';
  } catch {
    return 'wasm';
  }
}

const CACHE_NAME = 'vision-models-v1';

/** Fetch a model, streaming progress, and persist it in the Cache API for instant reloads. */
export async function fetchModel(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      onProgress?.({ loaded: 1, total: 1, cached: true });
      return await hit.arrayBuffer();
    }
  } catch {
    /* Cache API unavailable (private mode etc.) – fall through to a plain fetch. */
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download ${url} (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const chunks = [];
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.({ loaded, total, cached: false });
  }
  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    buffer.set(c, offset);
    offset += c.byteLength;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(buffer, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch {
      /* quota exceeded – ignore, the model still works for this session */
    }
  }
  return buffer.buffer;
}

/** Create a session on the preferred backend, falling back to WASM if WebGPU fails. */
export async function createSession(buffer, backend) {
  const ort = await loadOrt();
  const providers = backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  try {
    return { session: await ort.InferenceSession.create(buffer, { executionProviders: providers }), backend };
  } catch (err) {
    if (backend !== 'webgpu') throw err;
    console.warn('WebGPU session failed, falling back to WASM:', err);
    return { session: await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] }), backend: 'wasm' };
  }
}

export async function clearModelCache() {
  try {
    await caches.delete(CACHE_NAME);
  } catch {
    /* ignore */
  }
}
