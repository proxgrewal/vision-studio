export const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush',
];

/** DOTA v1 aerial classes used by YOLO11-obb. */
export const DOTA_CLASSES = [
  'plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field',
  'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool',
];

// 20 hand-picked, high-contrast colours for the most common classes; golden-angle hues beyond that.
const BASE_PALETTE = [
  '#ff3838', '#48f90a', '#00c2ff', '#ffb21d', '#cb38ff', '#00d4bb', '#ff701f', '#6473ff', '#ff95c8', '#92cc17',
  '#2c99a8', '#ff9d97', '#8438ff', '#3ddb86', '#cfd231', '#0018ec', '#ff37c7', '#1a9334', '#344593', '#520085',
].map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);

/** Deterministic, well-separated colour for class index `i` as [r,g,b]. */
export function classColor(i) {
  if (i < BASE_PALETTE.length) return BASE_PALETTE[i];
  const h = (i * 137.508) % 360; // golden-angle hue spacing
  const s = 0.72 + 0.18 * ((i * 7) % 3) / 2;
  const l = 0.5 + 0.1 * ((i * 11) % 3) / 2;
  return hslToRgb(h / 360, s, l);
}

export function rgbCss([r, g, b], a = 1) {
  return a === 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${a})`;
}

function hslToRgb(h, s, l) {
  const f = (n) => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}
