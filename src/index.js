// Public library entry point.
//
//   import { Engine, RENDERERS } from 'vision-studio-web';
//   const engine = new Engine({ modelBase: 'https://proxgrewal.github.io/vision-studio/models/' });
//   const backend = await engine.init();                 // 'webgpu' | 'wasm'
//   await engine.load('detect', backend, 'n');
//   const result = await engine.run('detect', imageElement, { conf: 0.25, iou: 0.45 });
//   RENDERERS[result.kind].draw(canvas.getContext('2d'), result, canvas.width, canvas.height);
//
// Every model runs inside a Web Worker on ONNX Runtime Web (WebGPU → multi-threaded WASM fallback).
export { Engine } from './engine.js';
export { RENDERERS, drawYolo, drawObb, drawClassify, drawSemantic, drawDepth, drawSam, obbCorners, COLORMAPS } from './render.js';
export { toCoco, toYoloTxt, masksZip, samMasksZip, labelMapPng, depthPng16, encodePng16, downloadBlob } from './export.js';
export { ZipWriter } from './zip.js';
export { BoxEditor } from './editor.js';
export { readOnnxMeta } from './onnxmeta.js';
export { COCO_CLASSES, DOTA_CLASSES, classColor, rgbCss } from './labels.js';
export { openPointCloud, closePointCloud } from './pointcloud.js';

/** Task ids understood by Engine.load / Engine.run. */
export const TASKS = ['detect', 'segment', 'pose', 'obb', 'classify', 'world', 'semantic', 'depth', 'sam'];
