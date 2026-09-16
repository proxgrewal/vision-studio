import { Engine } from './src/engine.js';
import { COCO_CLASSES, DOTA_CLASSES, rgbCss } from './src/labels.js';
import { RENDERERS } from './src/render.js';
import { canvasBlob, depthPng16, downloadBlob, labelMapPng, masksZip, samMasksZip, toCoco, toYoloTxt } from './src/export.js';
import { ZipWriter } from './src/zip.js';
import { BoxEditor } from './src/editor.js';

const TASKS = {
  detect: { name: 'YOLO11 detect', desc: 'YOLO11 object detection — 80 COCO classes with bounding boxes.', sizes: ['n', 's', 'm'] },
  segment: { name: 'YOLO11 segment', desc: 'YOLO11 instance segmentation — a pixel mask for every detected object.', sizes: ['n', 's', 'm'] },
  pose: { name: 'YOLO11 pose', desc: 'YOLO11 pose estimation — 17 COCO keypoints per person with a skeleton overlay.', sizes: ['n', 's'] },
  obb: { name: 'YOLO11 OBB', desc: 'YOLO11 oriented bounding boxes — rotated boxes for aerial imagery (15 DOTA classes: planes, ships, vehicles…).', sizes: ['n', 's'] },
  classify: { name: 'YOLO11 classify', desc: 'YOLO11 image classification — top-5 ImageNet labels (1000 classes) for the whole image.', sizes: ['n', 's'] },
  semantic: { name: 'SegFormer-B0 ADE20K', desc: 'SegFormer semantic segmentation — labels every pixel with one of 150 ADE20K scene classes.', sizes: [] },
  depth: { name: 'Depth Anything V2 S', desc: 'Depth Anything V2 — monocular relative depth for any image.', sizes: [] },
  world: { name: 'YOLO-World v2 S', desc: 'Open-vocabulary detection — type any object names and YOLO-World finds them (CLIP text embeddings, 32 prompts max).', sizes: [] },
  sam: { name: 'SlimSAM', desc: 'Segment Anything (SlimSAM) — click any object to get its mask; shift+click to exclude.', sizes: [] },
  custom: { name: 'Custom model', desc: 'Your own Ultralytics ONNX export.', sizes: [] },
};
const YOLO_TASKS = new Set(['detect', 'segment', 'pose', 'obb', 'classify', 'custom', 'world']);
const BOX_KINDS = new Set(['detect', 'segment', 'pose', 'world']);

const SAMPLES = [
  { file: 'samples/bus.jpg', alt: 'Bus and pedestrians' },
  { file: 'samples/street.jpg', alt: 'Street with stop sign' },
  { file: 'samples/boats.jpg', alt: 'Marina from above (aerial)' },
  { file: 'samples/living-room.jpg', alt: 'Living room' },
  { file: 'samples/cats.jpg', alt: 'Two cats' },
  { file: 'samples/river.jpg', alt: 'River under a bridge' },
];

/** Community Ultralytics exports loadable straight from the Hub (must carry ONNX metadata). */
const MODEL_ZOO = [];

const engine = new Engine();

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const video = $('video');
const dropzone = $('dropzone');
const stage = $('stage');
const frame = document.createElement('canvas'); // latest video frame, so overlay and image stay in sync

const state = {
  task: 'detect',
  detectedBackend: 'wasm',
  size: 'n',
  source: null, // { kind: 'image' | 'video' | 'stream', el, w, h, label }
  result: null,
  busy: false,
  rerun: false,
  live: false,
  stream: null,
  fpsTimes: [],
  custom: null, // { label, kind, names, size }
  countLine: null,
  counts: null,
  recorder: null,
  drawingLine: null,
  gridMode: false,
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
    track: $('track').checked,
    countLine: $('count-line').checked,
    distance: $('distance').checked,
    finegrained: $('finegrained').checked,
  };
}

function backendChoice() {
  const o = settings().backend;
  return o === 'auto' ? state.detectedBackend : o;
}

const isMoving = () => state.source?.kind === 'video' || state.source?.kind === 'stream';
const resultKind = () => (state.task === 'custom' ? state.custom?.kind : state.task);

/* ---------- status ---------- */
function setStatus(text, { error = false, progress = null } = {}) {
  const el = $('status-text');
  el.textContent = text;
  el.classList.toggle('error', error);
  if (progress !== null) $('progress-bar').style.width = Math.round(progress * 100) + '%';
}

const fmtMB = (bytes) => (bytes / 1048576).toFixed(1) + ' MB';

function progressReporter(name) {
  return ({ loaded, total, cached }) => {
    if (cached) setStatus(name + ': loaded from cache, compiling…', { progress: 1 });
    else if (total) setStatus('Downloading ' + name + ' ' + fmtMB(loaded) + ' / ' + fmtMB(total) + (loaded >= total ? ' — compiling…' : ''), { progress: loaded / total });
    else setStatus('Downloading ' + name + ' ' + fmtMB(loaded), { progress: 0.5 });
  };
}

/* ---------- model loading ---------- */
const loaded = new Map(); // task → key of the variant currently loaded in the worker

async function ensureModelFor(task) {
  if (task === 'custom') {
    if (!state.custom) throw new Error('No custom model loaded');
    return;
  }
  const backend = backendChoice();
  const variant = YOLO_TASKS.has(task) ? (TASKS[task].sizes.includes(state.size) ? state.size : 'n') : settings().precision;
  const key = backend + '|' + variant;
  if (loaded.get(task) === key) return;
  const name = TASKS[task].name;
  setStatus('Loading ' + name + '…', { progress: 0 });
  const info = await engine.load(task, backend, variant, progressReporter(name));
  loaded.set(task, key);
  setStatus(name + ' (' + info.variant + ') ready on ' + info.backend.toUpperCase(), { progress: 1 });
  setBadge(info.backend);
  if (task === state.task) $('stat-model').textContent = name + ' · ' + info.variant + ' · ' + info.backend;
}

function setBadge(backend) {
  $('backend-badge').textContent = backend.toUpperCase();
  $('backend-badge').className = 'badge ' + backend;
}

/* ---------- sources ---------- */
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
  if (s.kind === 'image') return { el: s.el, w: s.w, h: s.h };
  return { el: frame, w: frame.width, h: frame.height };
}

function drawBase(el, w, h) {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.drawImage(el, 0, 0, w, h);
}

/* ---------- rendering ---------- */
function render() {
  const d = sourceDims();
  if (!d || !d.w) return;
  drawBase(d.el, d.w, d.h);
  const r = state.result;
  if (r && r.task === state.task && RENDERERS[r.kind]) RENDERERS[r.kind].draw(ctx, r, d.w, d.h, settings());
  drawCountLine();
  editor.draw(ctx);
}

function currentLabels() {
  if (state.task === 'custom') return state.custom.names;
  if (state.task === 'world') return [...new Set([...(state.result?.detections || []).map((x) => x.name), ...$('world-prompts').value.split(/[,\n]/).map((p) => p.trim()).filter(Boolean)])];
  if (state.task === 'obb') return DOTA_CLASSES;
  if (state.task === 'pose') return ['person'];
  return COCO_CLASSES;
}

const editor = new BoxEditor({
  canvas,
  result: () => state.result,
  render: () => render(),
  onChange: () => {
    if (state.result) updateResults(state.result);
    syncEditorClass();
  },
  labels: currentLabels,
});

function syncEditorClass() {
  const sel = $('editor-class');
  const labels = currentLabels();
  if (sel.options.length !== labels.length || sel.options[0]?.textContent !== labels[0]) {
    sel.replaceChildren(...labels.map((l, i) => Object.assign(document.createElement('option'), { value: i, textContent: l })));
  }
  const d = editor.boxes()[editor.selected];
  if (d) sel.value = d.cls;
}

function toggleEditor(on) {
  const canEdit = on && state.result?.detections && BOX_KINDS.has(state.result.kind) && !isMoving();
  if (on && !canEdit) return setStatus('Editing needs a detection result on a still image', { error: true });
  editor.enable(canEdit);
  $('editor-bar').hidden = !canEdit;
  $('btn-edit').classList.toggle('on', canEdit);
  if (canEdit) syncEditorClass();
}

function renderOriginal() {
  const d = sourceDims();
  if (d && d.w) drawBase(d.el, d.w, d.h);
}

function drawCountLine() {
  const line = state.drawingLine || state.countLine;
  if (!line || !settings().countLine) return;
  ctx.save();
  ctx.strokeStyle = '#ffd166';
  ctx.lineWidth = Math.max(2, canvas.width / 300);
  ctx.setLineDash([12, 8]);
  ctx.beginPath();
  ctx.moveTo(line.x1, line.y1);
  ctx.lineTo(line.x2, line.y2);
  ctx.stroke();
  ctx.restore();
}

/* ---------- inference ---------- */
async function infer() {
  if (editor.active) return; // keep manual edits until the user leaves edit mode
  if (state.busy) {
    state.rerun = true;
    return;
  }
  if (isMoving()) captureFrame();
  const d = sourceDims();
  if (!d || !d.w) return;
  state.busy = true;
  const task = state.task;
  try {
    await ensureModelFor(task);
    const s = settings();
    const moving = isMoving();
    const result = await engine.run(task, d.el, { conf: s.conf, iou: s.iou, fast: moving, track: s.track && moving });
    if (task !== state.task) return; // user switched task mid-flight; a fresh run is queued
    await enrich(result, d, s, moving);
    result.task = task;
    state.result = result;
    if (result.kind === 'sam') setStatus('Image encoded in ' + result.timing.infer.toFixed(0) + ' ms — click an object to segment it');
    if (s.countLine && state.countLine && result.detections) updateCounts(result);
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

/** Optional second-model passes: per-box distance from Depth Anything, fine-grained labels from the classifier. */
async function enrich(result, d, s, moving) {
  if (!result.detections?.length || !BOX_KINDS.has(result.kind)) return;
  if (s.distance) {
    await ensureModelFor('depth');
    const depth = await engine.run('depth', d.el, { fast: true });
    const sx = depth.width / d.w;
    const sy = depth.height / d.h;
    for (const det of result.detections) {
      const x0 = Math.max(0, Math.floor(det.x * sx));
      const y0 = Math.max(0, Math.floor(det.y * sy));
      const x1 = Math.min(depth.width, Math.ceil((det.x + det.w) * sx));
      const y1 = Math.min(depth.height, Math.ceil((det.y + det.h) * sy));
      const vals = [];
      for (let y = y0; y < y1; y += 2) for (let x = x0; x < x1; x += 2) vals.push(depth.depth[y * depth.width + x]);
      vals.sort((a, b) => a - b);
      const med = vals.length ? vals[vals.length >> 1] / 255 : 0;
      det.nearness = med;
      det.label = det.label + ' · ' + (med > 0.66 ? 'near' : med > 0.33 ? 'mid' : 'far') + ' ' + Math.round(med * 100) + '%';
    }
    result.detections.sort((a, b) => b.nearness - a.nearness);
  }
  if (s.finegrained && !moving) {
    await ensureModelFor('classify');
    const crop = document.createElement('canvas');
    for (const det of result.detections.slice(0, 12)) {
      if (det.w < 16 || det.h < 16) continue;
      crop.width = Math.round(det.w);
      crop.height = Math.round(det.h);
      crop.getContext('2d').drawImage(d.el, det.x, det.y, det.w, det.h, 0, 0, crop.width, crop.height);
      const c = await engine.run('classify', crop, { topK: 1 });
      const top = c.classes[0];
      if (top && top.score > 0.2) det.label = det.label + ' (' + top.label + ')';
    }
  }
}

function updateStats(result) {
  $('stat-infer').textContent = result.timing.infer.toFixed(0);
  $('stat-total').textContent = result.timing.total.toFixed(0);
  const now = performance.now();
  state.fpsTimes.push(now);
  state.fpsTimes = state.fpsTimes.filter((t) => now - t < 2000);
  const fps = isMoving() && state.fpsTimes.length > 1 ? (state.fpsTimes.length - 1) / ((now - state.fpsTimes[0]) / 1000) : 1000 / result.timing.total;
  $('stat-fps').textContent = fps.toFixed(1);
}

function updateResults(result) {
  const list = $('results-list');
  const renderer = RENDERERS[result.kind];
  const rows = renderer ? renderer.summary(result) : [];
  const titles = { detect: 'Detections', segment: 'Instances', pose: 'People', obb: 'Oriented boxes', classify: 'Top-5 classes', semantic: 'Classes (share of image)', depth: 'Depth', sam: 'Segment Anything objects', world: 'Open-vocabulary detections' };
  $('results-title').textContent = (titles[result.kind] || 'Results') + (result.detections ? ' · ' + result.detections.length : '');
  const empty = result.kind === 'obb' ? 'No aerial objects found — OBB is trained on satellite / drone imagery (try the marina sample).' : 'Nothing above the confidence threshold — try lowering it.';
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
      : [Object.assign(document.createElement('li'), { className: 'empty', textContent: empty })]),
  );
}

/* ---------- line counting ---------- */
function updateCounts(result) {
  const L = state.countLine;
  const c = state.counts || (state.counts = { a: 0, b: 0, byLabel: {}, side: new Map() });
  const side = (x, y) => Math.sign((L.x2 - L.x1) * (y - L.y1) - (L.y2 - L.y1) * (x - L.x1));
  for (const d of result.detections) {
    if (!d.id) continue;
    const cx = d.cx ?? d.x + d.w / 2;
    const cy = d.cy ?? d.y + d.h / 2;
    const s = side(cx, cy);
    const prev = c.side.get(d.id);
    if (prev !== undefined && prev !== 0 && s !== 0 && prev !== s) {
      const dir = s > 0 ? 'a' : 'b';
      c[dir]++;
      const base = d.name || d.label;
      c.byLabel[base] = c.byLabel[base] || { a: 0, b: 0 };
      c.byLabel[base][dir]++;
    }
    if (s !== 0) c.side.set(d.id, s);
  }
  const el = $('counter');
  el.hidden = false;
  const rows = Object.entries(c.byLabel).map(([k, v]) => k + ': ' + v.a + ' ↓ / ' + v.b + ' ↑').join('<br>');
  el.innerHTML = '<b>' + (c.a + c.b) + '</b> crossings · <b>' + c.a + '</b> ↓ · <b>' + c.b + '</b> ↑' + (rows ? '<br>' + rows : '');
}

function resetCounts() {
  state.counts = null;
  $('counter').hidden = true;
  engine.resetTracker();
}

function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * canvas.width, y: ((e.clientY - r.top) / r.height) * canvas.height };
}

/* ---------- loading sources ---------- */
function showCanvas(show) {
  canvas.classList.toggle('empty', !show);
  dropzone.classList.toggle('hidden', show);
  $('btn-download').disabled = !show;
  $('btn-compare').disabled = !show;
  $('btn-grid').disabled = !show;
}

function newSource(source) {
  if (editor.active) toggleEditor(false);
  stopStream();
  stopVideo();
  state.source = source;
  state.result = null;
  state.fpsTimes = [];
  resetCounts();
  $('video-controls').hidden = source.kind !== 'video';
  showCanvas(true);
  exitGrid();
}

async function loadFile(file, label = file.name) {
  if (file.type.startsWith('video/')) return loadVideoFile(file);
  const url = URL.createObjectURL(file);
  try {
    await loadImageUrl(url, label);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function loadImageUrl(url, label) {
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
    newSource({ kind: 'image', el, w, h, label });
    render();
    setStatus('Image ' + w + '×' + h + ' loaded');
    infer();
  } catch (err) {
    setStatus('Could not load image: ' + (err?.message || err), { error: true });
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

/* video files */
async function loadVideoFile(file) {
  const url = URL.createObjectURL(file);
  newSource({ kind: 'video', label: file.name, url });
  video.srcObject = null;
  video.src = url;
  video.loop = $('video-loop').checked;
  try {
    await new Promise((res, rej) => {
      video.onloadedmetadata = res;
      video.onerror = () => rej(new Error('unsupported video format'));
    });
  } catch (e) {
    return setStatus(e.message, { error: true });
  }
  await video.play().catch(() => {});
  $('btn-play').textContent = 'Pause';
  setStatus('Video ' + video.videoWidth + '×' + video.videoHeight + ' · ' + fmtTime(video.duration));
  state.live = true;
  liveLoop();
}

function stopVideo() {
  if (state.source?.kind !== 'video') return;
  state.live = false;
  video.pause();
  video.removeAttribute('src');
  video.load();
  URL.revokeObjectURL(state.source.url);
  stopRecording();
}

const fmtTime = (t) => (isFinite(t) ? Math.floor(t / 60) + ':' + String(Math.floor(t % 60)).padStart(2, '0') : '0:00');

/* webcam & screen */
async function startStream(kind) {
  try {
    let stream;
    if (kind === 'screen') {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: false });
    } else {
      const [w, h] = $('cam-res').value.split('x').map(Number);
      const deviceId = $('camera-select').value;
      stream = await navigator.mediaDevices.getUserMedia({
        video: { ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'environment' }), width: { ideal: w }, height: { ideal: h } },
        audio: false,
      });
      listCameras();
    }
    newSource({ kind: 'stream', label: kind });
    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    stream.getVideoTracks()[0].addEventListener('ended', () => (stopStream(), render()));
    state.live = true;
    $('btn-webcam').classList.toggle('live', kind === 'webcam');
    $('btn-screen').classList.toggle('live', kind === 'screen');
    $('btn-webcam').textContent = kind === 'webcam' ? 'Stop' : 'Webcam';
    $('btn-screen').textContent = kind === 'screen' ? 'Stop' : 'Screen';
    setStatus((kind === 'screen' ? 'Screen ' : 'Webcam ') + video.videoWidth + '×' + video.videoHeight);
    liveLoop();
  } catch (err) {
    setStatus('Capture unavailable: ' + (err?.message || err), { error: true });
  }
}

function stopStream() {
  if (state.source?.kind !== 'stream') return;
  state.live = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  $('btn-webcam').classList.remove('live');
  $('btn-screen').classList.remove('live');
  $('btn-webcam').textContent = 'Webcam';
  $('btn-screen').textContent = 'Screen';
  stopRecording();
  // Freeze the last frame so the user keeps something to look at.
  const still = document.createElement('canvas');
  still.width = frame.width || 1;
  still.height = frame.height || 1;
  still.getContext('2d').drawImage(frame, 0, 0);
  state.source = { kind: 'image', el: still, w: still.width, h: still.height, label: 'captured frame' };
  state.result = null;
}

async function listCameras() {
  try {
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    const sel = $('camera-select');
    const cur = sel.value;
    sel.replaceChildren(...cams.map((c, i) => Object.assign(document.createElement('option'), { value: c.deviceId, textContent: c.label || 'Camera ' + (i + 1) })));
    if (cur) sel.value = cur;
    $('camera-row').hidden = cams.length < 2;
  } catch {
    /* no permission yet */
  }
}

async function liveLoop() {
  while (state.live) {
    if (!state.busy && video.readyState >= 2 && !video.paused && state.task !== 'sam') await infer();
    if (state.source?.kind === 'video') syncScrub();
    await new Promise((r) => requestAnimationFrame(r));
  }
}

function syncScrub() {
  if (!video.duration) return;
  $('scrub').value = Math.round((video.currentTime / video.duration) * 1000);
  $('video-time').textContent = fmtTime(video.currentTime) + ' / ' + fmtTime(video.duration);
}

/* recording */
function toggleRecording() {
  if (state.recorder) return stopRecording();
  const stream = canvas.captureStream(30);
  const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find((m) => MediaRecorder.isTypeSupported(m));
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 6e6 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => downloadBlob(new Blob(chunks, { type: mime }), 'vision-studio-' + state.task + (mime.includes('mp4') ? '.mp4' : '.webm'));
  rec.start(500);
  state.recorder = rec;
  $('btn-record').textContent = '■ Stop';
  $('btn-record').classList.add('rec');
}

function stopRecording() {
  if (!state.recorder) return;
  state.recorder.stop();
  state.recorder = null;
  $('btn-record').textContent = '● Record';
  $('btn-record').classList.remove('rec');
}

/* ---------- custom models ---------- */
async function loadCustomModel(buffer, label) {
  try {
    setStatus('Loading ' + label + '…', { progress: 0 });
    const info = await engine.loadCustom(buffer, label, backendChoice(), progressReporter(label));
    state.custom = { label, kind: info.kind, names: info.labels, size: info.size };
    loaded.set('custom', 'x');
    const tab = document.querySelector('#task-tabs button[data-task="custom"]');
    tab.hidden = false;
    tab.textContent = label.replace(/\.onnx$/i, '').slice(0, 14);
    $('custom-info').textContent = info.kind + ' · ' + info.labels.length + ' classes · ' + info.size + ' px' + (info.meta?.version ? ' · ultralytics ' + info.meta.version : '');
    TASKS.custom.name = label;
    TASKS.custom.desc = 'Custom ' + info.kind + ' model "' + label + '" — ' + info.labels.length + ' classes: ' + info.labels.slice(0, 8).join(', ') + (info.labels.length > 8 ? '…' : '');
    setStatus(label + ' ready on ' + info.backend.toUpperCase(), { progress: 1 });
    $('stat-model').textContent = label + ' · ' + info.backend;
    setTask('custom');
  } catch (err) {
    setStatus('Custom model failed: ' + (err?.message || err), { error: true });
  }
}

async function loadCustomFromUrl(url) {
  try {
    setStatus('Downloading ' + url, { progress: 0 });
    const res = await fetch(url);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const total = Number(res.headers.get('content-length')) || 0;
    const chunks = [];
    let got = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.byteLength;
      if (total) setStatus('Downloading ' + fmtMB(got) + ' / ' + fmtMB(total), { progress: got / total });
    }
    const buf = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) {
      buf.set(c, o);
      o += c.byteLength;
    }
    await loadCustomModel(buf.buffer, url.split('/').pop().split('?')[0]);
  } catch (err) {
    setStatus('Could not fetch model (CORS?): ' + (err?.message || err), { error: true });
  }
}

async function samDecode() {
  const r = state.result;
  if (!r) return;
  if (!r.points.length) {
    r.preview = null;
    render();
    updateResults(r);
    return;
  }
  state.busy = true;
  try {
    r.preview = await engine.samPrompt(r.points);
    $('stat-infer').textContent = r.preview.timing.infer.toFixed(0);
    $('stat-total').textContent = r.preview.timing.total.toFixed(0);
    render();
    updateResults(r);
  } catch (err) {
    setStatus('SAM error: ' + err.message, { error: true });
  } finally {
    state.busy = false;
  }
}

/* ---------- task switching ---------- */
function setTask(task) {
  if (task === 'custom' && !state.custom) return;
  if (editor.active) toggleEditor(false);
  state.task = task;
  document.querySelectorAll('#task-tabs button').forEach((b) => {
    const on = b.dataset.task === task;
    b.classList.toggle('active', on);
    b.setAttribute('aria-selected', on);
  });
  $('task-desc').textContent = TASKS[task].desc;
  const kind = resultKind();
  const boxes = BOX_KINDS.has(kind) || kind === 'obb';
  $('settings-yolo').hidden = !YOLO_TASKS.has(task);
  $('model-size').hidden = task === 'custom';
  document.querySelectorAll('#model-size button').forEach((b) => (b.hidden = !TASKS[task].sizes?.includes(b.dataset.size)));
  $('settings-nms').hidden = kind === 'classify';
  $('row-count').hidden = !boxes;
  $('row-distance').hidden = !BOX_KINDS.has(kind);
  $('row-finegrained').hidden = !BOX_KINDS.has(kind);
  $('settings-mask').hidden = !(kind === 'segment' || kind === 'semantic' || kind === 'obb');
  $('settings-depth').hidden = task !== 'depth';
  $('settings-sam').hidden = task !== 'sam';
  $('settings-world').hidden = task !== 'world';
  $('model-size').hidden = task === 'custom' || task === 'world';
  $('settings-mask').hidden = $('settings-mask').hidden && task !== 'sam';
  stage.classList.toggle('drawline', task === 'sam' || settings().countLine);
  state.result = null;
  $('results-list').replaceChildren();
  $('results-title').textContent = 'Results';
  exitGrid();
  render();
  updateUrl();
  if (state.source) infer();
  else ensureModelFor(task).catch((e) => setStatus('Error: ' + e.message, { error: true }));
}

/* ---------- "all tasks" grid ---------- */
async function runGrid() {
  const d = sourceDims();
  if (!d || state.busy) return;
  const grid = $('grid-view');
  grid.hidden = false;
  state.gridMode = true;
  grid.replaceChildren();
  const order = ['detect', 'segment', 'pose', 'obb', 'classify', 'semantic', 'depth'];
  const cells = order.map((t) => {
    const fig = document.createElement('figure');
    const c = document.createElement('canvas');
    c.width = d.w;
    c.height = d.h;
    c.getContext('2d').drawImage(d.el, 0, 0);
    const cap = document.createElement('figcaption');
    cap.innerHTML = '<b>' + TASKS[t].name + '</b><span>queued…</span>';
    fig.append(c, cap);
    grid.append(fig);
    return { t, c, cap };
  });
  state.busy = true;
  try {
    for (const { t, c, cap } of cells) {
      if (!state.gridMode) break;
      cap.lastChild.textContent = 'loading…';
      await ensureModelFor(t);
      cap.lastChild.textContent = 'running…';
      const s = settings();
      const r = await engine.run(t, d.el, { conf: s.conf, iou: s.iou });
      const cctx = c.getContext('2d');
      cctx.drawImage(d.el, 0, 0);
      RENDERERS[r.kind].draw(cctx, r, d.w, d.h, s);
      cap.lastChild.textContent = r.timing.infer.toFixed(0) + ' ms' + (r.detections ? ' · ' + r.detections.length + ' objects' : r.classes ? ' · ' + r.classes[0].label : '');
    }
  } catch (err) {
    setStatus('Error: ' + err.message, { error: true });
  } finally {
    state.busy = false;
  }
}

function exitGrid() {
  state.gridMode = false;
  $('grid-view').hidden = true;
}

/* ---------- exports ---------- */
async function doExport(kind) {
  const r = state.result;
  const d = sourceDims();
  const base = 'vision-studio-' + state.task;
  const need = (ok, msg) => {
    if (!ok) setStatus(msg, { error: true });
    return ok;
  };
  switch (kind) {
    case 'png':
      if (need(d, 'Nothing to export')) downloadBlob(await canvasBlob(canvas), base + '.png');
      break;
    case 'json':
      if (need(r, 'Run a model first')) {
        const { masks, depth, raw, labelMap, ...rest } = r;
        downloadBlob(new Blob([JSON.stringify({ ...rest, image: { width: d.w, height: d.h, name: state.source.label } }, null, 2)], { type: 'application/json' }), base + '.json');
      }
      break;
    case 'coco':
      if (need(r?.detections, 'COCO export needs a detection / segmentation / pose / OBB result')) {
        const labels = state.task === 'custom' ? state.custom.names : r.kind === 'obb' ? DOTA_CLASSES : r.kind === 'pose' ? ['person'] : r.kind === 'world' ? [...new Set(r.detections.map((d) => d.name))] : COCO_CLASSES;
        downloadBlob(new Blob([toCoco(r, d.w, d.h, state.source.label || 'image', labels)], { type: 'application/json' }), base + '-coco.json');
      }
      break;
    case 'yolo':
      if (need(r?.detections, 'YOLO export needs a detection result')) downloadBlob(new Blob([toYoloTxt(r, d.w, d.h)], { type: 'text/plain' }), base + '.txt');
      break;
    case 'masks':
      if (r?.kind === 'sam') {
        if (need(r.objects?.length || r.preview, 'Click an object first')) downloadBlob(await samMasksZip(r, d.w, d.h), base + '-masks.zip');
      } else if (need(r?.masks?.count, 'Run instance segmentation first')) downloadBlob(await masksZip(r, d.w, d.h), base + '-masks.zip');
      break;
    case 'depth16':
      if (need(r?.raw, 'Run depth estimation first')) downloadBlob(await depthPng16(r), base + '-depth16.png');
      break;
    case 'labelmap':
      if (need(r?.labelMap, 'Run semantic segmentation first')) downloadBlob(await labelMapPng(r), base + '-labels.png');
      break;
    case 'link':
      await navigator.clipboard.writeText(location.href);
      setStatus('Share link copied: ' + location.href);
      break;
  }
}

/* ---------- batch ---------- */
async function runBatch(files) {
  if (!files.length) return;
  const zip = new ZipWriter();
  const rows = ['file,width,height,count,labels,infer_ms'];
  const off = document.createElement('canvas');
  const octx = off.getContext('2d');
  const s = settings();
  const task = state.task;
  await ensureModelFor(task);
  let i = 0;
  for (const f of files) {
    i++;
    setStatus('Batch ' + i + '/' + files.length + ': ' + f.name, { progress: i / files.length });
    let img;
    try {
      img = await createImageBitmap(f);
    } catch {
      rows.push(f.name + ',,,error,,');
      continue;
    }
    const r = await engine.run(task, img, { conf: s.conf, iou: s.iou });
    off.width = img.width;
    off.height = img.height;
    octx.drawImage(img, 0, 0);
    RENDERERS[r.kind]?.draw(octx, r, img.width, img.height, s);
    const stem = f.name.replace(/\.[^.]+$/, '');
    await zip.add('images/' + stem + '.png', await canvasBlob(off));
    const { masks, depth, raw, labelMap, ...rest } = r;
    await zip.add('results/' + stem + '.json', JSON.stringify(rest, null, 2));
    if (r.detections) await zip.add('labels/' + stem + '.txt', toYoloTxt(r, img.width, img.height));
    const labels = r.detections ? [...new Set(r.detections.map((d) => d.label))].join(' ') : r.classes ? r.classes[0].label : '';
    rows.push([f.name, img.width, img.height, r.detections?.length ?? '', '"' + labels + '"', r.timing.infer.toFixed(0)].join(','));
    img.close();
  }
  await zip.add('summary.csv', rows.join('\n'));
  downloadBlob(zip.blob(), 'vision-studio-batch-' + task + '.zip');
  setStatus('Batch done: ' + files.length + ' images', { progress: 1 });
}

/* ---------- URL state ---------- */
function applyUrl() {
  const p = new URLSearchParams(location.search);
  if (p.get('task') && TASKS[p.get('task')] && p.get('task') !== 'custom') state.task = p.get('task');
  if (p.get('size')) state.size = p.get('size');
  if (p.get('conf')) $('conf').value = p.get('conf');
  if (p.get('iou')) $('iou').value = p.get('iou');
  if (p.get('backend')) $('backend').value = p.get('backend');
  if (p.get('track')) $('track').checked = p.get('track') === '1';
  if (p.get('colormap')) $('colormap').value = p.get('colormap');
  $('conf-out').textContent = Number($('conf').value).toFixed(2);
  $('iou-out').textContent = Number($('iou').value).toFixed(2);
  document.querySelectorAll('#model-size button').forEach((b) => b.classList.toggle('active', b.dataset.size === state.size));
  return p.get('sample');
}

function updateUrl() {
  const s = settings();
  const p = new URLSearchParams();
  if (state.task !== 'custom') p.set('task', state.task);
  if (state.size !== 'n') p.set('size', state.size);
  if (s.conf !== 0.25) p.set('conf', s.conf);
  if (s.iou !== 0.45) p.set('iou', s.iou);
  if (s.backend !== 'auto') p.set('backend', s.backend);
  if (s.track) p.set('track', '1');
  if (state.task === 'depth' && s.colormap !== 'inferno') p.set('colormap', s.colormap);
  const sample = SAMPLES.find((x) => x.alt === state.source?.label);
  if (sample) p.set('sample', sample.file.split('/').pop().replace('.jpg', ''));
  history.replaceState(null, '', location.pathname + (p.toString() ? '?' + p : ''));
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
    updateUrl();
    state.source ? infer() : ensureModelFor(state.task);
  });

  const bindOut = (id, outId) => $(id).addEventListener('input', () => ($(outId).textContent = Number($(id).value).toFixed(2)));
  bindOut('conf', 'conf-out');
  bindOut('iou', 'iou-out');
  bindOut('mask-opacity', 'mask-opacity-out');
  bindOut('depth-blend', 'depth-blend-out');
  for (const id of ['conf', 'iou', 'distance', 'finegrained']) $(id).addEventListener('change', () => (updateUrl(), infer()));
  for (const id of ['show-labels', 'mask-opacity', 'colormap', 'depth-blend', 'depth-invert']) $(id).addEventListener('input', () => (updateUrl(), render()));
  $('track').addEventListener('change', () => (resetCounts(), updateUrl(), infer()));
  $('count-line').addEventListener('change', () => {
    stage.classList.toggle('drawline', $('count-line').checked);
    if ($('count-line').checked) {
      $('track').checked = true;
      setStatus('Drag a line across the video to count objects crossing it');
    }
    resetCounts();
    render();
  });
  for (const id of ['backend', 'precision']) {
    $(id).addEventListener('change', () => {
      state.result = null;
      loaded.clear();
      updateUrl();
      state.source ? infer() : ensureModelFor(state.task);
    });
  }

  // Inputs.
  $('btn-upload').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    if (f) loadFile(f);
    e.target.value = '';
  });
  $('btn-batch').addEventListener('click', () => $('batch-input').click());
  $('batch-input').addEventListener('change', (e) => {
    runBatch([...e.target.files]);
    e.target.value = '';
  });
  $('btn-webcam').addEventListener('click', () => (state.source?.kind === 'stream' && state.source.label === 'webcam' ? (stopStream(), render()) : startStream('webcam')));
  $('btn-screen').addEventListener('click', () => (state.source?.kind === 'stream' && state.source.label === 'screen' ? (stopStream(), render()) : startStream('screen')));
  $('camera-select').addEventListener('change', () => state.source?.kind === 'stream' && startStream('webcam'));

  // Video controls.
  $('btn-play').addEventListener('click', () => {
    if (video.paused) {
      video.play();
      $('btn-play').textContent = 'Pause';
    } else {
      video.pause();
      $('btn-play').textContent = 'Play';
    }
  });
  $('scrub').addEventListener('input', () => {
    if (!video.duration) return;
    video.currentTime = ($('scrub').value / 1000) * video.duration;
  });
  video.addEventListener('seeked', () => {
    if (state.source?.kind === 'video' && video.paused) infer();
    syncScrub();
  });
  video.addEventListener('ended', () => ($('btn-play').textContent = 'Play'));
  video.addEventListener('pause', () => state.task === 'sam' && state.source?.kind === 'video' && infer());
  $('video-loop').addEventListener('change', () => (video.loop = $('video-loop').checked));
  $('btn-record').addEventListener('click', toggleRecording);

  // Custom models.
  $('btn-custom-file').addEventListener('click', () => $('custom-input').click());
  $('custom-input').addEventListener('change', async (e) => {
    const f = e.target.files?.[0];
    if (f) loadCustomModel(await f.arrayBuffer(), f.name);
    e.target.value = '';
  });
  $('btn-custom-url').addEventListener('click', () => {
    const url = prompt('URL of an Ultralytics ONNX export (must allow CORS, e.g. a Hugging Face "resolve" link):');
    if (url) loadCustomFromUrl(url.trim());
  });
  const zoo = $('zoo-select');
  for (const m of MODEL_ZOO) zoo.append(Object.assign(document.createElement('option'), { value: m.url, textContent: m.name }));
  zoo.parentElement.hidden = MODEL_ZOO.length === 0;
  zoo.addEventListener('change', () => zoo.value && loadCustomFromUrl(zoo.value));

  // Exports & tools.
  $('export-buttons').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-export]');
    if (b) doExport(b.dataset.export).catch((err) => setStatus('Export failed: ' + err.message, { error: true }));
  });
  $('btn-clear-cache').addEventListener('click', async () => {
    await engine.clearCache();
    loaded.clear();
    setStatus('Model cache cleared — models will download again on next use.');
  });
  $('btn-download').addEventListener('click', () => doExport('png'));
  $('btn-grid').addEventListener('click', () => (state.gridMode ? exitGrid() : runGrid()));
  $('btn-edit').addEventListener('click', () => toggleEditor(!editor.active));
  $('editor-done').addEventListener('click', () => toggleEditor(false));
  $('editor-delete').addEventListener('click', () => editor.deleteSelected());
  $('editor-class').addEventListener('change', (e) => editor.setClass(Number(e.target.value)));
  canvas.addEventListener('pointerup', () => editor.active && syncEditorClass());
  $('btn-3d').addEventListener('click', async () => {
    if (!state.result?.raw) return setStatus('Run depth first', { error: true });
    const { openPointCloud } = await import('./src/pointcloud.js');
    const d = sourceDims();
    openPointCloud(state.result, d.el, d.w, d.h);
  });
  $('btn-theme').addEventListener('click', () => {
    const light = document.documentElement.dataset.theme !== 'light';
    document.documentElement.dataset.theme = light ? 'light' : 'dark';
    try {
      localStorage.setItem('theme', light ? 'light' : 'dark');
    } catch {}
  });

  // Hold-to-compare.
  const cmp = $('btn-compare');
  cmp.addEventListener('pointerdown', renderOriginal);
  cmp.addEventListener('pointerup', render);
  cmp.addEventListener('pointerleave', render);

  // Open-vocabulary prompts.
  const applyPrompts = async () => {
    const prompts = $('world-prompts').value.split(/[,\n]/).map((p) => p.trim()).filter(Boolean);
    if (!prompts.length) return;
    try {
      await ensureModelFor('world');
      setStatus('Encoding prompts…');
      const info = await engine.setPrompts(prompts, progressReporter('CLIP text encoder'));
      const custom = info.prompts.length - info.fromVocab.length;
      setStatus('Detecting ' + info.prompts.length + ' classes' + (custom ? ' (' + custom + ' encoded with CLIP)' : ''), { progress: 1 });
      state.result = null;
      infer();
    } catch (err) {
      setStatus('Prompt error: ' + err.message, { error: true });
    }
  };
  $('world-apply').addEventListener('click', applyPrompts);
  $('world-prompts').addEventListener('keydown', (e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), applyPrompts()));
  $('world-coco').addEventListener('click', () => (($('world-prompts').value = COCO_CLASSES.join(', ')), applyPrompts()));

  // Segment Anything clicks.
  canvas.addEventListener('click', async (e) => {
    if (state.task !== 'sam' || !state.result?.embedded || state.busy) return;
    const p = canvasPoint(e);
    state.result.points.push({ x: p.x, y: p.y, label: e.shiftKey ? 0 : 1 });
    await samDecode();
  });
  $('sam-keep').addEventListener('click', () => {
    const r = state.result;
    if (!r?.preview) return;
    r.objects = r.objects || [];
    r.objects.push(r.preview);
    r.preview = null;
    r.points = [];
    render();
    updateResults(r);
  });
  $('sam-undo').addEventListener('click', async () => {
    const r = state.result;
    if (!r?.points?.length) return;
    r.points.pop();
    await samDecode();
  });
  $('sam-clear').addEventListener('click', () => {
    const r = state.result;
    if (!r) return;
    r.points = [];
    r.preview = null;
    r.objects = [];
    render();
    updateResults(r);
  });
  $('sam-export').addEventListener('click', async () => {
    const r = state.result;
    const d = sourceDims();
    if (!r?.objects?.length && !r?.preview) return setStatus('Click an object first', { error: true });
    downloadBlob(await samMasksZip(r, d.w, d.h), 'vision-studio-sam-masks.zip');
  });

  // Count-line drawing on the canvas.
  canvas.addEventListener('pointerdown', (e) => {
    if (!settings().countLine || state.task === 'sam' || editor.active) return;
    const p = canvasPoint(e);
    state.drawingLine = { x1: p.x, y1: p.y, x2: p.x, y2: p.y };
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {}
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!state.drawingLine) return;
    const p = canvasPoint(e);
    state.drawingLine.x2 = p.x;
    state.drawingLine.y2 = p.y;
    render();
  });
  canvas.addEventListener('pointerup', () => {
    if (!state.drawingLine) return;
    const l = state.drawingLine;
    state.drawingLine = null;
    if (Math.hypot(l.x2 - l.x1, l.y2 - l.y1) > 10) {
      state.countLine = l;
      resetCounts();
    }
    render();
  });

  // Drag & drop / paste anywhere.
  const stop = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  for (const evt of ['dragenter', 'dragover']) document.addEventListener(evt, (e) => (stop(e), stage.classList.add('dragover')));
  for (const evt of ['dragleave', 'drop']) document.addEventListener(evt, (e) => (stop(e), stage.classList.remove('dragover')));
  document.addEventListener('drop', async (e) => {
    const files = [...(e.dataTransfer?.files || [])];
    const onnx = files.find((x) => /\.onnx$/i.test(x.name));
    if (onnx) return loadCustomModel(await onnx.arrayBuffer(), onnx.name);
    const media = files.filter((x) => x.type.startsWith('image/') || x.type.startsWith('video/'));
    if (media.length > 1 && media.every((x) => x.type.startsWith('image/'))) return runBatch(media);
    if (media[0]) loadFile(media[0]);
  });
  document.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith('image/'));
    if (item) loadFile(item.getAsFile(), 'pasted image');
  });

  // Sample thumbnails.
  const samples = $('samples');
  for (const s of SAMPLES) {
    const b = document.createElement('button');
    b.title = s.alt;
    b.dataset.file = s.file;
    const img = document.createElement('img');
    img.src = s.file;
    img.alt = s.alt;
    img.loading = 'lazy';
    b.append(img);
    b.addEventListener('click', () => {
      samples.querySelectorAll('button').forEach((x) => x.classList.toggle('active', x === b));
      loadImageUrl(s.file, s.alt).then(updateUrl);
    });
    samples.append(b);
  }
  window.addEventListener('beforeunload', () => (stopStream(), stopRecording()));
}

/* ---------- public API (window.vision) ---------- */
window.vision = {
  engine,
  tasks: Object.keys(TASKS).filter((t) => t !== 'custom'),
  /** Run a task on any image source (img, canvas, video, ImageBitmap, Blob). Returns the raw result object. */
  async run(task, source, opts = {}) {
    await ensureModelFor(task);
    const src = source instanceof Blob ? await createImageBitmap(source) : source;
    return engine.run(task, src, { conf: 0.25, iou: 0.45, ...opts });
  },
  load: (task) => ensureModelFor(task),
  loadCustom: (buffer, label) => loadCustomModel(buffer, label),
  draw: (canvasEl, result, opts = {}) => RENDERERS[result.kind]?.draw(canvasEl.getContext('2d'), result, canvasEl.width, canvasEl.height, opts),
  get result() {
    return state.result;
  },
};

async function main() {
  try {
    const t = localStorage.getItem('theme');
    if (t) document.documentElement.dataset.theme = t;
  } catch {}
  wire();
  showCanvas(false);
  const initialSample = applyUrl();
  try {
    state.detectedBackend = await engine.init();
  } catch (err) {
    setStatus('Failed to start the inference worker: ' + err.message, { error: true });
    return;
  }
  setBadge(state.detectedBackend);
  if (state.detectedBackend === 'wasm') setStatus('WebGPU not available — running on CPU (WASM). Chrome/Edge 113+ is much faster.');
  listCameras();
  setTask(state.task);
  const p = new URLSearchParams(location.search);
  if (p.get('model')) loadCustomFromUrl(p.get('model'));
  const buttons = [...$('samples').querySelectorAll('button')];
  (buttons.find((b) => initialSample && b.dataset.file.includes('/' + initialSample + '.')) || buttons[0])?.click();
}

main();
