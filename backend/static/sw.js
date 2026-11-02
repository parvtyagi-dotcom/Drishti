// Service worker for the Post Siren app.
//
// It exists to make the siren installable to a phone's home screen - browsers
// require a service worker with a fetch handler before they offer "Install".
//
// It deliberately does NOT cache the live data paths. A siren showing a cached
// alert, or a cached "all clear", would be actively dangerous: the operator
// would be looking at history while believing it is current. Only the shell
// (page, icons, manifest) is cached, so the app still opens without network
// and can show that it is disconnected.

const SHELL_CACHE = "bw-siren-shell-v1";
const SHELL_ASSETS = [
  "/api/live/siren",
  "/api/live/siren/icon-192.png",
  "/api/live/siren/icon-512.png",
  "/api/live/siren/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== SHELL_CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  const isShell = SHELL_ASSETS.includes(url.pathname);

  if (!isShell) return; // live data always goes straight to the network

  // Network-first for the shell too, so a redeployed page is picked up; the
  // cache is only the offline fallback.
  event.respondWith(
    fetch(request)
      .then((response) => {
        const copy = response.clone();
        caches.open(SHELL_CACHE).then((cache) => cache.put(request, copy));
        return response;
      })
      .catch(() => caches.match(request)),
  );
});
