// Service worker with two jobs:
//  1. Add the COOP/COEP headers static hosts (GitHub Pages) cannot send, which enables
//     SharedArrayBuffer and therefore multi-threaded WASM inference for users without WebGPU.
//  2. Cache the app shell (HTML/JS/CSS/samples/CDN scripts) so the site works offline as a PWA.
//     Model files are cached separately by the app via the Cache API.
const SHELL = 'vision-shell-v2';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) =>
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith('vision-shell-') && k !== SHELL).map((k) => caches.delete(k)))).then(() => self.clients.claim())),
);

const isolate = (res) => {
  if (res.status === 0 || res.type === 'opaque') return res;
  const headers = new Headers(res.headers);
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
};

const cacheable = (req) => {
  if (req.method !== 'GET') return false;
  const u = new URL(req.url);
  if (u.origin === location.origin) return !u.pathname.includes('/models/');
  return u.hostname === 'cdn.jsdelivr.net'; // runtime + wasm + three.js; HF model files are handled by the app
};

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  if (!cacheable(req)) {
    e.respondWith(fetch(req).then(isolate));
    return;
  }
  // Network first with revalidation (so deploys show up immediately), cache as fallback for offline use.
  const sameOrigin = req.url.startsWith(location.origin);
  e.respondWith(
    fetch(sameOrigin ? new Request(req, { cache: 'no-cache' }) : req)
      .then((res) => {
        if (res.ok) caches.open(SHELL).then((c) => c.put(req, res.clone())).catch(() => {});
        return isolate(res);
      })
      .catch(async () => {
        const hit = await caches.match(req, { ignoreSearch: req.url.startsWith(location.origin) });
        return hit ? isolate(hit) : Response.error();
      }),
  );
});
