# Vision Studio

**Live: https://proxgrewal.github.io/vision-studio/**

Open a URL, drop in an image or video (or turn on your webcam) and run modern computer-vision models **entirely in the browser** — nothing is uploaded anywhere. WebGPU when available, multi-threaded WASM otherwise, everything inside a Web Worker so the page never freezes.

| Task | Model | Notes |
| --- | --- | --- |
| Object detection | YOLO11 n / s / m (80 COCO classes) | ~15–40 ms on WebGPU |
| Instance segmentation | YOLO11 n / s / m-seg | mask protos decoded in JS |
| Pose estimation | YOLO11 n / s-pose | 17 COCO keypoints + skeleton |
| Oriented boxes | YOLO11 n / s-obb (DOTA aerial, 1024 px) | probIoU rotated NMS |
| Classification | YOLO11 n / s-cls (ImageNet-1k) | top-5 |
| **Open-vocabulary detection** | YOLO-World v2 S + CLIP ViT-B/32 text tower | type any phrase; 1,670 common nouns precomputed, others encoded in-browser |
| **Click-to-segment** | SlimSAM (Segment Anything, 77 % pruned) | encode once, ~130 ms per click |
| Semantic segmentation | SegFormer-B0 fine-tuned on ADE20K (150 classes) | candidate-class upsampling trick |
| Depth estimation | Depth Anything V2 Small / Base | 5 colormaps, 16-bit export, 3D point cloud |
| **Your own model** | any Ultralytics `.onnx` export | task, classes, input size read from ONNX metadata |

## Features

**Sources** — image upload, drag & drop, paste, sample images, **video files** (scrub / play / loop), **webcam** (camera + resolution picker), **screen capture**, and **batch processing** of many images into a ZIP.

**Video tools** — ByteTrack-style multi-object **tracking** (IDs + trails), **line-crossing counter** (draw a line, count in/out per class), **record** the annotated output to WebM.

**Combinations** — per-box **distance** from Depth Anything, **fine-grained labels** from the classifier inside each box, an **"All tasks"** grid that runs every model on one image.

**Outputs** — PNG, JSON, **COCO JSON**, **YOLO txt labels**, per-instance **mask PNGs (ZIP)**, **16-bit depth PNG**, class-id map, shareable URLs (`?task=obb&sample=boats&conf=0.4`). A built-in **box editor** lets you move / resize / add / relabel / delete detections before exporting — a tiny annotation tool.

**Platform** — installable **PWA** that works offline once models are cached, light/dark theme, a **benchmark page** (`bench.html`) that measures every model on your device, and a JavaScript API (`window.vision`, or the npm-style library below).

## Run locally

Any static server works; the service worker (multi-threaded WASM, offline shell) needs `localhost` or HTTPS.

```bash
git clone https://github.com/proxgrewal/vision-studio
cd vision-studio
python -m http.server 8000     # → http://localhost:8000
```

No build step: plain ES modules, ONNX Runtime Web and Three.js come from jsDelivr, Depth Anything / SegFormer / SAM / CLIP stream from the Hugging Face Hub on first use and are cached with the Cache API.

## Use it as a library

The engine is framework-free and can be dropped into any page (the YOLO weights are served from GitHub Pages with CORS enabled):

```js
import { Engine, RENDERERS, toCoco } from 'https://proxgrewal.github.io/vision-studio/src/index.js';

const engine = new Engine({ modelBase: 'https://proxgrewal.github.io/vision-studio/models/' });
const backend = await engine.init();                       // 'webgpu' | 'wasm'
await engine.load('detect', backend, 'n', ({ loaded, total }) => console.log(loaded / total));

const result = await engine.run('detect', imageElement, { conf: 0.25, iou: 0.45, track: false });
// result.detections → [{ x, y, w, h, score, cls, name, label }, …]
RENDERERS[result.kind].draw(canvas.getContext('2d'), result, canvas.width, canvas.height, { showLabels: true });
console.log(toCoco(result, canvas.width, canvas.height, 'photo.jpg', ['person', …]));

await engine.load('world', backend);                       // open-vocabulary
await engine.setPrompts(['red backpack', 'traffic light']);
await engine.load('sam', backend);                         // segment anything
await engine.run('sam', imageElement);                     // encode once
const mask = await engine.samPrompt([{ x: 120, y: 80, label: 1 }]);
```

`package.json` is set up for `npm publish` (`vision-studio-web`); `src/index.js` is the entry point and `src/worker.js` the worker. On the hosted page the same API is available as `window.vision.run(task, source, opts)`.

## How it works

```
index.html / app.js       UI: sources, video/webcam loop, tracking counter, editor, exports, grid, PWA
bench.html                per-device benchmark of every model
src/engine.js             promise wrapper around the worker (load / run / setPrompts / samPrompt)
src/worker.js             owns ONNX Runtime + every model session; nothing here blocks the page
src/runtime.js            ORT loader, WebGPU/WASM selection, Cache API model store with progress
src/tasks/yolo.js         letterbox → YOLO11 → decode, per-class NMS (box IoU / probIoU), mask protos,
                          keypoints, ByteTrack-style tracker, custom-model factory (reads ONNX metadata)
src/tasks/world.js        YOLO-World: prompts → CLIP embeddings (vocab table or text encoder) → txt_feats input
src/tasks/sam.js          SlimSAM encoder (once) + prompt decoder (per click)
src/tasks/semantic.js     SegFormer → bilinear-upsampled argmax (4 candidate classes per pixel)
src/tasks/depth.js        Depth Anything V2 → normalised relative depth (+ raw floats)
src/render.js             boxes, rotated boxes, skeletons, masks, colormaps, legends
src/export.js, zip.js     COCO / YOLO / mask ZIP / 16-bit PNG writers (no dependencies)
src/editor.js             box editor
src/pointcloud.js         Three.js depth point cloud
src/onnxmeta.js           minimal protobuf reader for ONNX metadata_props + input shape
coi-sw.js                 COOP/COEP headers for multi-threaded WASM + offline app shell
models/                   YOLO11 / YOLO-World ONNX exports, CLIP vocabulary embeddings, ImageNet names
```

### Exporting the models

```bash
pip install ultralytics onnx onnxslim
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True     # detect
yolo export model=yolo11n-seg.pt ...  yolo11n-pose.pt ...  yolo11n-cls.pt imgsz=224 ...  yolo11n-obb.pt imgsz=1024 ...
```

YOLO-World is exported with a small wrapper so the CLIP text embeddings become a runtime input (`txt_feats`, 32 slots) instead of being baked in; the vocabulary table in `models/world-vocab.bin` holds fp16 CLIP embeddings for COCO + LVIS + Objects365 + Open Images class names. Custom models work as long as they are standard Ultralytics exports (`nms=False`) — the class names, task and image size are read from the ONNX `metadata_props`.

## License

Code is released under **AGPL-3.0** (see `LICENSE`), matching the bundled [Ultralytics YOLO11 / YOLO-World](https://github.com/ultralytics/ultralytics) weights. Depth Anything V2 Small is Apache-2.0; SlimSAM and CLIP are Apache-2.0 / MIT; SegFormer is under the NVIDIA Source Code License (non-commercial research use). Sample images: Ultralytics assets and COCO val2017.
