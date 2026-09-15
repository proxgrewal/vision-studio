// Inference worker: owns ONNX Runtime and every model session so the UI thread never blocks.
import { clearModelCache, detectBackend } from './runtime.js';
import { YOLO_KINDS, YoloTask, customYoloTask } from './tasks/yolo.js';
import { DepthTask } from './tasks/depth.js';
import { SemanticTask } from './tasks/semantic.js';
import { SamTask } from './tasks/sam.js';

const tasks = {};
const loading = {}; // task id → in-flight load promise, so concurrent requests share one download

function getTask(id) {
  if (!tasks[id]) {
    if (YOLO_KINDS[id]) tasks[id] = new YoloTask(YOLO_KINDS[id]);
    else if (id === 'depth') tasks[id] = new DepthTask();
    else if (id === 'semantic') tasks[id] = new SemanticTask();
    else if (id === 'sam') tasks[id] = new SamTask();
    else throw new Error('Unknown task ' + id);
  }
  return tasks[id];
}

function describe(impl) {
  return { backend: impl.backend, variant: impl.variant, labels: impl.labels || null, kind: impl.kind || null, size: impl.cfg?.size || null };
}

self.onmessage = async (e) => {
  const { id, type } = e.data;
  const reply = (msg, transfer = []) => self.postMessage({ id, ...msg }, transfer);
  try {
    if (type === 'init') {
      reply({ type: 'ready', backend: await detectBackend() });
    } else if (type === 'load') {
      const { task, backend, variant } = e.data;
      const impl = getTask(task);
      const key = task + '|' + backend + '|' + variant;
      if (!loading[key]) {
        loading[key] = impl
          .load(backend, variant, (p) => self.postMessage({ id, type: 'progress', ...p }))
          .finally(() => delete loading[key]);
      }
      await loading[key];
      reply({ type: 'loaded', ...describe(impl) });
    } else if (type === 'loadCustom') {
      // A user-supplied Ultralytics ONNX export; replaces any previous custom model.
      const { buffer, label, backend } = e.data;
      const impl = customYoloTask(buffer, label);
      tasks.custom = impl;
      await impl.load(backend, 'custom', (p) => self.postMessage({ id, type: 'progress', ...p }));
      reply({ type: 'loaded', ...describe(impl), label, meta: impl.cfg.meta });
    } else if (type === 'run') {
      const { task, bitmap, opts } = e.data;
      const impl = tasks[task];
      if (!impl?.session) throw new Error('Model not loaded');
      const result = await impl.run(bitmap, bitmap.width, bitmap.height, opts);
      bitmap.close();
      const { transfer = [], ...rest } = result;
      reply({ type: 'result', result: rest }, transfer);
    } else if (type === 'samPrompt') {
      const impl = tasks.sam;
      if (!impl?.embedding) throw new Error('Encode an image first');
      const { transfer = [], ...rest } = await impl.prompt(e.data.points);
      reply({ type: 'result', result: rest }, transfer);
    } else if (type === 'resetTracker') {
      for (const t of Object.values(tasks)) t.resetTracker?.();
      reply({ type: 'ok' });
    } else if (type === 'clearCache') {
      await clearModelCache();
      reply({ type: 'cleared' });
    }
  } catch (err) {
    reply({ type: 'error', message: err?.message || String(err) });
  }
};
