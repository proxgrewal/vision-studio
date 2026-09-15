// 3D point-cloud view of a Depth Anything result (Three.js, loaded on demand from the CDN via the import map).
let overlay = null;
let stopAnim = null;

export async function openPointCloud(result, imageEl, W, H) {
  closePointCloud();
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');

  overlay = document.createElement('div');
  overlay.className = 'pc-overlay';
  overlay.innerHTML =
    '<div class="pc-bar"><span>Drag to orbit · scroll to zoom · right-drag to pan</span>' +
    '<label>Depth scale <input type="range" min="0.2" max="3" step="0.1" value="1" id="pc-scale"></label>' +
    '<label>Point size <input type="range" min="0.5" max="6" step="0.5" value="2" id="pc-size"></label>' +
    '<button class="tool" id="pc-close">Close ✕</button></div>';
  const glCanvas = document.createElement('canvas');
  overlay.prepend(glCanvas);
  document.body.append(overlay);

  const { raw, width, height, min, max } = result;
  // Subsample so we never push more than ~400k points to the GPU.
  const step = Math.max(1, Math.ceil(Math.sqrt((width * height) / 400000)));
  const cols = Math.floor(width / step);
  const rows = Math.floor(height / step);
  const n = cols * rows;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);

  // Sample image colours at depth-map resolution.
  const cc = document.createElement('canvas');
  cc.width = width;
  cc.height = height;
  const cctx = cc.getContext('2d', { willReadFrequently: true });
  cctx.drawImage(imageEl, 0, 0, W, H, 0, 0, width, height);
  const px = cctx.getImageData(0, 0, width, height).data;

  const range = max - min || 1;
  const f = 0.5 * width / Math.tan((60 * Math.PI) / 360); // assume a 60° horizontal FOV
  const zs = new Float32Array(n);
  let i = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++, i++) {
      const x = c * step;
      const y = r * step;
      const disp = (raw[y * width + x] - min) / range; // 1 = nearest
      const z = 1 / (0.15 + disp * 0.85); // metric-ish depth in [1, 6.7]
      zs[i] = z;
      positions[i * 3] = ((x - width / 2) / f) * z;
      positions[i * 3 + 1] = -((y - height / 2) / f) * z;
      positions[i * 3 + 2] = -z;
      const o = (y * width + x) * 4;
      colors[i * 3] = px[o] / 255;
      colors[i * 3 + 1] = px[o + 1] / 255;
      colors[i * 3 + 2] = px[o + 2] / 255;
    }
  }
  const base = positions.slice();

  const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0c12);
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
  camera.position.set(0, 0, 0.6);
  const controls = new OrbitControls(camera, glCanvas);
  controls.target.set(0, 0, -2.5);
  controls.enableDamping = true;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({ size: 2, vertexColors: true, sizeAttenuation: false });
  scene.add(new THREE.Points(geom, mat));

  const resize = () => {
    const w = overlay.clientWidth;
    const h = overlay.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  window.addEventListener('resize', resize);

  overlay.querySelector('#pc-scale').addEventListener('input', (e) => {
    const s = Number(e.target.value);
    const p = geom.attributes.position.array;
    for (let k = 0; k < n; k++) {
      const z = 1 + (zs[k] - 1) * s;
      const ratio = z / zs[k];
      p[k * 3] = base[k * 3] * ratio;
      p[k * 3 + 1] = base[k * 3 + 1] * ratio;
      p[k * 3 + 2] = -z;
    }
    geom.attributes.position.needsUpdate = true;
  });
  overlay.querySelector('#pc-size').addEventListener('input', (e) => (mat.size = Number(e.target.value)));
  overlay.querySelector('#pc-close').addEventListener('click', closePointCloud);
  const onKey = (e) => e.key === 'Escape' && closePointCloud();
  window.addEventListener('keydown', onKey);

  let raf = 0;
  const loop = () => {
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(loop);
  };
  loop();
  stopAnim = () => {
    cancelAnimationFrame(raf);
    window.removeEventListener('resize', resize);
    window.removeEventListener('keydown', onKey);
    controls.dispose();
    geom.dispose();
    mat.dispose();
    renderer.dispose();
  };
}

export function closePointCloud() {
  stopAnim?.();
  stopAnim = null;
  overlay?.remove();
  overlay = null;
}
