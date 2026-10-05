/* Pose worker: MediaPipe Tasks PoseLandmarker on the CPU, off the main thread.
 *
 * Why a worker: (1) the page already holds the legacy MediaPipe pose engine, which defines a global `Module`; a second MediaPipe wasm module
 * in the same window collides with it ("Module.arguments has been replaced ..."). A worker has its own globals. (2) Inference (~85 ms/frame
 * on a mid-range phone CPU) must not block the interface.
 * Why the CPU delegate: some GPU drivers (e.g. PowerVR BXM-8-256) crash MediaPipe's WebGL pipeline or return no poses; the CPU path gives
 * the same model's results. (It still needs WebGL, via OffscreenCanvas, to hand frames over; init fails cleanly if that is missing.)
 *
 * Protocol: {type:'init', tasks, model} -> {type:'ready'} | {type:'error', message}
 *           {type:'frame', id, ts, bitmap} -> {type:'result', id, landmarks, world} | {type:'error', id, message}
 */
var landmarker = null;

function loadBundle(url) {
  // The package ships an ES module. A classic worker is needed (the library loads its wasm loader with importScripts), so the
  // module's final `export { a as B, ... }` clause is turned into an assignment and the text is evaluated in the worker's global scope.
  return fetch(url).then(function (r) {
    if (!r.ok) throw new Error('bundle HTTP ' + r.status);
    return r.text();
  }).then(function (code) {
    var m = code.match(/export\s*\{([^}]*)\}\s*;?\s*(?:\/\/[^\n]*)?\s*$/);
    if (!m) throw new Error('unexpected bundle format');
    var map = m[1].split(',').map(function (p) {
      var parts = p.trim().split(/\s+as\s+/);
      return (parts[1] || parts[0]) + ':' + parts[0];
    }).join(',');
    (0, eval)(code.slice(0, m.index) + 'self.__mpTasks = {' + map + '};');
    return self.__mpTasks;
  });
}

function toList(points) {
  var out = new Array(points.length);
  for (var i = 0; i < points.length; i++) {
    var k = points[i];
    out[i] = { x: k.x, y: k.y, z: k.z || 0, visibility: k.visibility || 0 };
  }
  return out;
}

self.onmessage = function (ev) {
  var m = ev.data;
  if (m.type === 'init') {
    loadBundle(m.tasks + '/vision_bundle.mjs').then(function (lib) {
      return lib.FilesetResolver.forVisionTasks(m.tasks + '/wasm').then(function (fileset) {
        return lib.PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: m.model, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numPoses: 1
        });
      });
    }).then(function (lm) {
      landmarker = lm;
      // Frames are handed over through WebGL; without it the first detection throws. Find out now, not mid-session.
      landmarker.detectForVideo(new OffscreenCanvas(64, 64), performance.now());
      postMessage({ type: 'ready' });
    }).catch(function (e) {
      postMessage({ type: 'error', message: String((e && e.message) || e).slice(0, 300) });
    });
  } else if (m.type === 'frame') {
    try {
      var res = landmarker.detectForVideo(m.bitmap, m.ts);
      if (m.bitmap && m.bitmap.close) m.bitmap.close();
      var lm = res.landmarks && res.landmarks[0];
      var wl = res.worldLandmarks && res.worldLandmarks[0];
      postMessage({ type: 'result', id: m.id, landmarks: lm && lm.length >= 33 ? toList(lm) : null, world: wl && wl.length >= 33 ? toList(wl) : null });
    } catch (e) {
      postMessage({ type: 'error', id: m.id, message: String((e && e.message) || e).slice(0, 300) });
    }
  } else if (m.type === 'close') {
    try { if (landmarker) landmarker.close(); } catch (e) { /* ignore */ }
    landmarker = null;
    self.close();
  }
};
