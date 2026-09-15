// Inference worker: owns ONNX Runtime and every model session so the UI thread never blocks.
import { clearModelCache, detectBackend } from './runtime.js';
import { YoloTask } from './tasks/yolo.js';
import { DepthTask } from './tasks/depth.js';
import { SemanticTask } from './tasks/semantic.js';

const tasks = {};
const loading = {}; // task id → in-flight load promise, so concurrent requests share one download

function getTask(id, baseUrl) {
  if (!tasks[id]) {
    if (id === 'detect' || id === 'segment') tasks[id] = new YoloTask(id, baseUrl);
    else if (id === 'depth') tasks[id] = new DepthTask();
    else if (id === 'semantic') tasks[id] = new SemanticTask();
    else throw new Error('Unknown task ' + id);
  }
  return tasks[id];
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  const reply = (msg, transfer = []) => self.postMessage({ id, ...msg }, transfer);
  try {
    if (type === 'init') {
      reply({ type: 'ready', backend: await detectBackend() });
    } else if (type === 'load') {
      const { task, backend, variant, baseUrl } = e.data;
      const impl = getTask(task, baseUrl);
      const key = task + '|' + backend + '|' + variant;
      if (!loading[key]) {
        loading[key] = impl
          .load(backend, variant, (p) => self.postMessage({ id, type: 'progress', ...p }))
          .finally(() => delete loading[key]);
      }
      await loading[key];
      reply({ type: 'loaded', backend: impl.backend, variant: impl.variant, labels: impl.labels || null });
    } else if (type === 'run') {
      const { task, bitmap, opts } = e.data;
      const impl = getTask(task);
      if (!impl.session) throw new Error('Model not loaded');
      const result = await impl.run(bitmap, bitmap.width, bitmap.height, opts);
      bitmap.close();
      const { transfer = [], ...rest } = result;
      reply({ type: 'result', result: rest }, transfer);
    } else if (type === 'clearCache') {
      await clearModelCache();
      reply({ type: 'cleared' });
    }
  } catch (err) {
    reply({ type: 'error', message: err?.message || String(err) });
  }
};
