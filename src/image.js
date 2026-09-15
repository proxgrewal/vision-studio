// Canvas-based preprocessing helpers shared by every task (runs inside the worker).

const scratch = new OffscreenCanvas(1, 1);
const sctx = scratch.getContext('2d', { willReadFrequently: true });

/**
 * Letterbox `source` into a size×size square (gray padding) and return
 * NCHW float32 RGB in [0,1] plus the geometry needed to map boxes back.
 */
export function letterbox(source, size, srcW, srcH) {
  const r = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  const dw = Math.floor((size - newW) / 2);
  const dh = Math.floor((size - newH) / 2);

  scratch.width = size;
  scratch.height = size;
  sctx.fillStyle = 'rgb(114,114,114)';
  sctx.fillRect(0, 0, size, size);
  sctx.drawImage(source, 0, 0, srcW, srcH, dw, dh, newW, newH);

  const { data } = sctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    tensor[i] = data[p] / 255;
    tensor[i + plane] = data[p + 1] / 255;
    tensor[i + 2 * plane] = data[p + 2] / 255;
  }
  return { tensor, ratio: r, dw, dh, newW, newH };
}

/**
 * Plain resize to width×height, then ImageNet mean/std normalisation.
 * Returns NCHW float32.
 */
export function resizeNormalize(source, width, height, srcW, srcH, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) {
  scratch.width = width;
  scratch.height = height;
  sctx.drawImage(source, 0, 0, srcW, srcH, 0, 0, width, height);
  const { data } = sctx.getImageData(0, 0, width, height);
  const plane = width * height;
  const tensor = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    tensor[i] = (data[p] / 255 - mean[0]) / std[0];
    tensor[i + plane] = (data[p + 1] / 255 - mean[1]) / std[1];
    tensor[i + 2 * plane] = (data[p + 2] / 255 - mean[2]) / std[2];
  }
  return tensor;
}

/**
 * Ultralytics classification transform: shorter side → size, centre crop, RGB in [0,1].
 * Returns NCHW float32.
 */
export function centerCrop(source, size, srcW, srcH) {
  const side = Math.min(srcW, srcH);
  const sx = (srcW - side) / 2;
  const sy = (srcH - side) / 2;
  scratch.width = size;
  scratch.height = size;
  sctx.drawImage(source, sx, sy, side, side, 0, 0, size, size);
  const { data } = sctx.getImageData(0, 0, size, size);
  const plane = size * size;
  const tensor = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    tensor[i] = data[p] / 255;
    tensor[i + plane] = data[p + 1] / 255;
    tensor[i + 2 * plane] = data[p + 2] / 255;
  }
  return tensor;
}

export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
