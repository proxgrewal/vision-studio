# Vision Studio

Open a URL, drop in an image (or turn on your webcam) and run modern computer-vision models **entirely in the browser** — nothing is uploaded anywhere.

| Task | Model | Runs on |
| --- | --- | --- |
| Object detection | YOLO11 n / s (80 COCO classes) | ONNX Runtime Web |
| Instance segmentation | YOLO11 n-seg / s-seg | ONNX Runtime Web |
| Oriented bounding boxes | YOLO11 n-obb / s-obb (15 DOTA aerial classes, 1024 px) | ONNX Runtime Web |
| Image classification | YOLO11 n-cls / s-cls (ImageNet-1k, top-5) | ONNX Runtime Web |
| Semantic segmentation | SegFormer-B0 fine-tuned on ADE20K (150 classes) | ONNX Runtime Web |
| Depth estimation | Depth Anything V2 Small | ONNX Runtime Web |

WebGPU is used when available (Chrome / Edge 113+, recent Firefox & Safari); otherwise it falls back to multi-threaded WASM on the CPU. Models are cached in the browser after the first download.

## Features

- Upload, drag & drop, paste, sample images, or **live webcam** with FPS counter
- Confidence / IoU sliders, model size (Nano vs Small), overlay opacity, depth colormaps
- Hold **Original** to compare, **Download** the rendered result
- Inference runs in a **Web Worker**, so the UI never freezes — even on CPU
- Zero build step: plain ES modules, deployable to any static host

## Run locally

Any static server works. The service worker that enables multi-threaded WASM needs `localhost` or HTTPS.

```bash
python -m http.server 8000
# → http://localhost:8000
```

## How it works

```
index.html / app.js          UI, sources (image / webcam), rendering of results
src/engine.js                 promise wrapper around the worker
src/worker.js                 owns ONNX Runtime + model sessions (never blocks the page)
src/tasks/yolo.js             letterbox → YOLO11 → decode + per-class NMS (box IoU, or probIoU for OBB),
                              mask protos for -seg, centre-crop + top-k for -cls
src/tasks/semantic.js         SegFormer → bilinear-upsampled argmax (candidate-class trick)
src/tasks/depth.js            Depth Anything V2 → normalised relative depth
src/render.js                 boxes, masks, colormaps, legends (main thread)
src/runtime.js                ORT loader, backend detection, Cache API model store
coi-sw.js                     adds COOP/COEP headers so WASM can use threads on static hosts
models/                       YOLO11 ONNX exports (imgsz 640, opset 17)
```

The YOLO ONNX files were exported with `yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True`
(`imgsz=1024` for -obb, `imgsz=224` for -cls).
Depth Anything and SegFormer are fetched on demand from the Hugging Face Hub
([onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small),
[Xenova/segformer-b0-finetuned-ade-512-512](https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512)).

## License

Code in this repository is released under **AGPL-3.0** (see `LICENSE`), matching the license of the bundled
[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) weights. Depth Anything V2 Small is Apache-2.0;
SegFormer is released under the NVIDIA Source Code License (non-commercial research use).
Sample images: Ultralytics assets and COCO val2017 (CC-BY / Flickr).
