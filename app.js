import { Engine } from './src/engine.js';
import { rgbCss } from './src/labels.js';
import { RENDERERS } from './src/render.js';

const TASKS = {
  detect: { name: 'YOLO11 detect', desc: 'YOLO11 object detection — 80 COCO classes with bounding boxes.' },
  segment: { name: 'YOLO11 segment', desc: 'YOLO11 instance segmentation — a pixel mask for every detected object.' },
  obb: { name: 'YOLO11 OBB', desc: 'YOLO11 oriented bounding boxes — rotated boxes for aerial imagery (15 DOTA classes: planes, ships, vehicles…).' },
  classify: { name: 'YOLO11 classify', desc: 'YOLO11 image classification — top-5 ImageNet labels (1000 classes) for the whole image.' },
  semantic: { name: 'SegFormer-B0 ADE20K', desc: 'SegFormer semantic segmentation — labels every pixel with one of 150 ADE20K scene classes.' },
  depth: { name: 'Depth Anything V2 S', desc: 'Depth Anything V2 — monocular relative depth for any image.' },
};

const YOLO_TASKS = new Set(['detect', 'segment', 'obb', 'classify']);

const engine = new Engine();

const SAMPLES = [
  { file: 'samples/bus.jpg', alt: 'Bus and pedestrians' },
  { file: 'samples/street.jpg', alt: 'Street with stop sign' },
  { file: 'samples/boats.jpg', alt: 'Marina from above (aerial)' },
  { file: 'samples/living-room.jpg', alt: 'Living room' },
  { file: 'samples/cats.jpg', alt: 'Two cats' },
  { file: 'samples/river.jpg', alt: 'River under a bridge' },
];

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const video = $('video');
const dropzone = $('dropzone');
const stage = $('stage');

const state = {
  task: 'detect',
  detectedBackend: 'wasm',
  size: 'n',
  source: null, // { kind: 'image', el, w, h } | { kind: 'video' }
  result: null,
  busy: false,
  rerun: false,
  live: false,
  stream: null,
  fpsTimes: [],
  loadedKey: null,
};

/* ---------- settings ---------- */
function settings() {
  return {
    conf: Number($('conf').value),
    iou: Number($('iou').value),
    showLabels: $('show-labels').checked,
    maskOpacity: Number($('mask-opacity').value),
    colormap: $('colormap').value,
    blend: Number($('depth-blend').value),
    invert: $('depth-invert').checked,
    backend: $('backend').value,
    precision: $('precision').value,
  };
}

function backendChoice() {
  const o = settings().backend;
  return o === 'auto' ? state.detectedBackend : o;
}

/* ---------- status ---------- */
function setStatus(text, { error = false, progress = null } = {}) {
  const el = $('status-text');
  el.textContent = text;
  el.classList.toggle('error', error);
  if (progress !== null) $('progress-bar').style.width = Math.round(progress * 100) + '%';
}

function fmtMB(bytes) {
  return (bytes / 1048576).toFixed(1) + ' MB';
}

/* ---------- model loading ---------- */
async function ensureModel() {
  const { name } = TASKS[state.task];
  const backend = backendChoice();
  const isYolo = YOLO_TASKS.has(state.task);
  const variant = isYolo ? state.size : settings().precision;
  const key = state.task + '|' + backend + '|' + variant;
  if (state.loadedKey === key) return;
  setStatus('Loading ' + name + '…', { progress: 0 });
  const info = await engine.load(state.task, backend, variant, ({ loaded, total, cached }) => {
    if (cached) setStatus(name + ': loaded from cache, compiling…', { progress: 1 });
    else if (total) setStatus('Downloading ' + name + ' ' + fmtMB(loaded) + ' / ' + fmtMB(total) + (loaded >= total ? ' — compiling…' : ''), { progress: loaded / total });
    else setStatus('Downloading ' + name + ' ' + fmtMB(loaded), { progress: 0.5 });
  });
  state.loadedKey = key;
  setStatus(name + ' (' + info.variant + ') ready on ' + info.backend.toUpperCase(), { progress: 1 });
  $('backend-badge').textContent = info.backend.toUpperCase();
  $('backend-badge').className = 'badge ' + info.backend;
  $('stat-model').textContent = name + ' · ' + info.variant + ' · ' + info.backend;
}

/* ---------- inference & rendering ---------- */
const frame = document.createElement('canvas'); // latest webcam frame, so overlay and image stay in sync

function captureFrame() {
  if (!video.videoWidth) return;
  if (frame.width !== video.videoWidth || frame.height !== video.videoHeight) {
    frame.width = video.videoWidth;
    frame.height = video.videoHeight;
  }
  frame.getContext('2d').drawImage(video, 0, 0);
}

function sourceDims() {
  const s = state.source;
  if (!s) return null;
  if (s.kind === 'video') return { el: frame, w: frame.width, h: frame.height };
  return { el: s.el, w: s.w, h: s.h };
}

function drawBase(el, w, h) {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.drawImage(el, 0, 0, w, h);
}

function render() {
  const d = sourceDims();
  if (!d || !d.w) return;
  drawBase(d.el, d.w, d.h);
  if (state.result && state.result.task === state.task) {
    RENDERERS[state.task].draw(ctx, state.result, d.w, d.h, settings());
  }
}

function renderOriginal() {
  const d = sourceDims();
  if (d && d.w) drawBase(d.el, d.w, d.h);
}

async function infer() {
  if (state.busy) {
    state.rerun = true;
    return;
  }
  if (state.live) captureFrame();
  const d = sourceDims();
  if (!d || !d.w) return;
  state.busy = true;
  try {
    await ensureModel();
    const s = settings();
    const task = state.task;
    const result = await engine.run(task, d.el, { conf: s.conf, iou: s.iou, fast: state.live });
    if (task !== state.task) return; // user switched task mid-flight; a fresh run is queued
    result.task = task;
    state.result = result;
    render();
    updateStats(result);
    updateResults(result);
  } catch (err) {
    console.error(err);
    setStatus('Error: ' + (err?.message || err), { error: true });
  } finally {
    state.busy = false;
    if (state.rerun) {
      state.rerun = false;
      infer();
    }
  }
}

function updateStats(result) {
  $('stat-infer').textContent = result.timing.infer.toFixed(0);
  $('stat-total').textContent = result.timing.total.toFixed(0);
  const now = performance.now();
  state.fpsTimes.push(now);
  state.fpsTimes = state.fpsTimes.filter((t) => now - t < 2000);
  $('stat-fps').textContent = state.live && state.fpsTimes.length > 1 ? ((state.fpsTimes.length - 1) / ((now - state.fpsTimes[0]) / 1000)).toFixed(1) : (1000 / result.timing.total).toFixed(1);
}

function updateResults(result) {
  const list = $('results-list');
  const rows = RENDERERS[state.task].summary(result);
  const titles = { detect: 'Detections', segment: 'Instances', obb: 'Oriented boxes', classify: 'Top-5 ImageNet classes', semantic: 'Classes (share of image)', depth: 'Depth' };
  $('results-title').textContent = titles[state.task] + (result.detections ? ' · ' + result.detections.length : '');
  list.replaceChildren(
    ...(rows.length
      ? rows.map((r) => {
          const li = document.createElement('li');
          if (r.color) {
            const sw = document.createElement('span');
            sw.className = 'sw';
            sw.style.background = rgbCss(r.color);
            li.append(sw);
          }
          const label = document.createElement('span');
          label.textContent = r.label;
          const v = document.createElement('span');
          v.className = 'v';
          v.textContent = r.value;
          li.append(label, v);
          return li;
        })
      : [Object.assign(document.createElement('li'), { className: 'empty', textContent: state.task === 'obb' ? 'No aerial objects found — OBB is trained on satellite / drone imagery (try the marina sample).' : 'Nothing above the confidence threshold — try lowering it.' })]),
  );
}

/* ---------- sources ---------- */
function showCanvas(show) {
  canvas.classList.toggle('empty', !show);
  dropzone.classList.toggle('hidden', show);
  $('btn-download').disabled = !show;
  $('btn-compare').disabled = !show;
}

async function loadImageFile(fileOrUrl, label) {
  stopWebcam();
  const url = typeof fileOrUrl === 'string' ? fileOrUrl : URL.createObjectURL(fileOrUrl);
  try {
    const img = await loadImage(url);
    // Downscale very large photos: the models never need more than ~1600px and canvases get slow.
    const MAX = 1600;
    let { naturalWidth: w, naturalHeight: h } = img;
    let el = img;
    if (Math.max(w, h) > MAX) {
      const s = MAX / Math.max(w, h);
      const c = document.createElement('canvas');
      c.width = Math.round(w * s);
      c.height = Math.round(h * s);
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      el = c;
      w = c.width;
      h = c.height;
    }
    state.source = { kind: 'image', el, w, h, label };
    state.result = null;
    showCanvas(true);
    render();
    setStatus('Image ' + w + '×' + h + ' loaded');
    infer();
  } catch (err) {
    setStatus('Could not load image: ' + (err?.message || err), { error: true });
  } finally {
    if (typeof fileOrUrl !== 'string') URL.revokeObjectURL(url);
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('unsupported or unreachable image'));
    img.src = url;
  });
}

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    state.source = { kind: 'video' };
    state.result = null;
    state.live = true;
    state.fpsTimes = [];
    $('btn-webcam').textContent = 'Stop webcam';
    $('btn-webcam').classList.add('live');
    showCanvas(true);
    setStatus('Webcam ' + video.videoWidth + '×' + video.videoHeight);
    liveLoop();
  } catch (err) {
    setStatus('Webcam unavailable: ' + (err?.message || err), { error: true });
  }
}

function stopWebcam() {
  if (!state.live) return;
  state.live = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  $('btn-webcam').textContent = 'Start webcam';
  $('btn-webcam').classList.remove('live');
  if (state.source?.kind === 'video') {
    // Freeze the last frame so the user keeps something to look at.
    const frame = document.createElement('canvas');
    frame.width = canvas.width;
    frame.height = canvas.height;
    frame.getContext('2d').drawImage(canvas, 0, 0);
    state.source = { kind: 'image', el: frame, w: frame.width, h: frame.height, label: 'webcam frame' };
    state.result = null;
  }
}

async function liveLoop() {
  while (state.live) {
    if (!state.busy && video.readyState >= 2) await infer();
    await new Promise((r) => requestAnimationFrame(r));
  }
}

/* ---------- task switching ---------- */
function setTask(task) {
  state.task = task;
  document.querySelectorAll('#task-tabs button').forEach((b) => {
    const on = b.dataset.task === task;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on);
  });
  $('task-desc').textContent = TASKS[task].desc;
  $('settings-yolo').hidden = !YOLO_TASKS.has(task);
  $('settings-nms').hidden = task === 'classify';
  $('settings-mask').hidden = !(task === 'segment' || task === 'semantic' || task === 'obb');
  $('settings-depth').hidden = task !== 'depth';
  state.result = null;
  $('results-list').replaceChildren();
  $('results-title').textContent = 'Results';
  render();
  if (state.source) infer();
  else ensureModel().catch((e) => setStatus('Error: ' + e.message, { error: true }));
}

/* ---------- wiring ---------- */
function wire() {
  $('task-tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-task]');
    if (b) setTask(b.dataset.task);
  });
  $('model-size').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-size]');
    if (!b) return;
    state.size = b.dataset.size;
    document.querySelectorAll('#model-size button').forEach((x) => x.classList.toggle('active', x === b));
    state.result = null;
    state.source ? infer() : ensureModel();
  });

  // Sliders that change decoding need a re-run; pure rendering options only need a redraw.
  const bindOut = (id, outId, fmt = (v) => v) => {
    const el = $(id);
    const out = $(outId);
    el.addEventListener('input', () => (out.textContent = fmt(el.value)));
  };
  bindOut('conf', 'conf-out', (v) => Number(v).toFixed(2));
  bindOut('iou', 'iou-out', (v) => Number(v).toFixed(2));
  bindOut('mask-opacity', 'mask-opacity-out', (v) => Number(v).toFixed(2));
  bindOut('depth-blend', 'depth-blend-out', (v) => Number(v).toFixed(2));
  for (const id of ['conf', 'iou']) $(id).addEventListener('change', () => infer());
  for (const id of ['show-labels', 'mask-opacity', 'colormap', 'depth-blend', 'depth-invert']) $(id).addEventListener('input', render);
  for (const id of ['backend', 'precision']) {
    $(id).addEventListener('change', () => {
      state.result = null;
      state.source ? infer() : ensureModel();
    });
  }

  $('btn-upload').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (f) loadImageFile(f, f.name);
    e.target.value = '';
  });
  $('btn-webcam').addEventListener('click', () => (state.live ? (stopWebcam(), render()) : startWebcam()));

  $('btn-clear-cache').addEventListener('click', async () => {
    await engine.clearCache();
    setStatus('Model cache cleared — models will download again on next use.');
  });

  $('btn-download').addEventListener('click', () => {
    canvas.toBlob((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'vision-studio-' + state.task + '.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }, 'image/png');
  });
  const cmp = $('btn-compare');
  const down = () => renderOriginal();
  const up = () => render();
  cmp.addEventListener('pointerdown', down);
  cmp.addEventListener('pointerup', up);
  cmp.addEventListener('pointerleave', up);
  cmp.addEventListener('keydown', (e) => e.key === ' ' && down());
  cmp.addEventListener('keyup', (e) => e.key === ' ' && up());

  // Drag & drop / paste anywhere.
  const stop = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  for (const evt of ['dragenter', 'dragover']) document.addEventListener(evt, (e) => (stop(e), stage.classList.add('dragover')));
  for (const evt of ['dragleave', 'drop']) document.addEventListener(evt, (e) => (stop(e), stage.classList.remove('dragover')));
  document.addEventListener('drop', (e) => {
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/'));
    if (f) loadImageFile(f, f.name);
  });
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith('image/'));
    if (item) loadImageFile(item.getAsFile(), 'pasted image');
  });

  // Sample thumbnails.
  const samples = $('samples');
  for (const s of SAMPLES) {
    const b = document.createElement('button');
    b.title = s.alt;
    const img = document.createElement('img');
    img.src = s.file;
    img.alt = s.alt;
    img.loading = 'lazy';
    b.append(img);
    b.addEventListener('click', () => {
      samples.querySelectorAll('button').forEach((x) => x.classList.toggle('active', x === b));
      loadImageFile(s.file, s.alt);
    });
    samples.append(b);
  }
  window.addEventListener('beforeunload', stopWebcam);
}

async function main() {
  wire();
  showCanvas(false);
  try {
    state.detectedBackend = await engine.init();
  } catch (err) {
    setStatus('Failed to start the inference worker: ' + err.message, { error: true });
    return;
  }
  const badge = $('backend-badge');
  badge.textContent = state.detectedBackend.toUpperCase();
  badge.className = 'badge ' + state.detectedBackend;
  if (state.detectedBackend === 'wasm') setStatus('WebGPU not available — running on CPU (WASM). Chrome/Edge 113+ is much faster.');
  setTask('detect');
  // Kick off with the first sample so visitors see results immediately.
  $('samples').querySelector('button')?.click();
}

main();
