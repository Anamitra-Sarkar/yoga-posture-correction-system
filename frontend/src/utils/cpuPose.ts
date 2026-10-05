// Compatibility pose engine: BlazePose on TensorFlow.js with the WebAssembly backend (no WebGL, no GPU).
//
// Why it exists: the normal engine (MediaPipe) needs a WebGL context. Some phones cannot give a browser one: e.g. phones with the
// IMG BXM-8-256 GPU (MediaTek Dimensity 7020/7025/930 ...) have driver problems that make Chrome refuse or lose WebGL contexts.
// On those phones this engine runs on the CPU instead: slower (a few frames per second) but it works, and it runs entirely on the device.
//
// It exposes the small part of MediaPipe's legacy `Pose` API that the app uses (setOptions / onResults / initialize / send / close) and
// returns results in the same shape (normalised image landmarks + metric world landmarks), so nothing downstream changes.
// It is only ever created when the fast engine is unavailable; devices where WebGL works never load any of this code's assets.

const CDN = "https://cdn.jsdelivr.net/npm/";
const TF = "4.22.0";
const PD = "2.1.3";
const WASM_BASE = `${CDN}@tensorflow/tfjs-backend-wasm@${TF}/dist/`;
const SCRIPTS = [
  `${CDN}@tensorflow/tfjs-core@${TF}/dist/tf-core.min.js`,
  `${CDN}@tensorflow/tfjs-converter@${TF}/dist/tf-converter.min.js`,
  `${CDN}@tensorflow/tfjs-backend-wasm@${TF}/dist/tf-backend-wasm.min.js`,
  `${CDN}@tensorflow-models/pose-detection@${PD}/dist/pose-detection.min.js`,
];
const MODEL_ROOT = "https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/";
const DETECTOR = `${MODEL_ROOT}detector/1`;
const LANDMARK_LITE = `${MODEL_ROOT}landmark/lite/2`;

// ---- which engine this device uses (remembered, so a phone that needed the compatibility engine goes straight to it) ----
const MODE_KEY = "asana.engine";
const MODE_TTL_MS = 3 * 24 * 60 * 60 * 1000;   // after 3 days the fast engine is tried again (browsers and drivers get updates)
export type EngineMode = "gpu" | "cpu";
export const readEngineMode = (): EngineMode => {
  try {
    const raw = window.localStorage.getItem(MODE_KEY);
    if (!raw) return "gpu";
    const { mode, at } = JSON.parse(raw);
    return mode === "cpu" && Date.now() - Number(at) < MODE_TTL_MS ? "cpu" : "gpu";
  } catch { return "gpu"; }
};
export const saveEngineMode = (mode: EngineMode) => {
  try {
    if (mode === "cpu") window.localStorage.setItem(MODE_KEY, JSON.stringify({ mode, at: Date.now() }));
    else window.localStorage.removeItem(MODE_KEY);
  } catch { /* private mode: the choice just is not remembered */ }
};
export const cpuEngineSupported = (): boolean => typeof WebAssembly !== "undefined";

// ---- script loading (each file once) ----
const scriptPromises = new Map<string, Promise<void>>();
const loadScript = (src: string): Promise<void> => {
  let p = scriptPromises.get(src);
  if (!p) {
    p = new Promise<void>((resolve, reject) => {
      const el = document.createElement("script");
      el.src = src;
      el.async = true;
      el.onload = () => resolve();
      el.onerror = () => { scriptPromises.delete(src); reject(new Error(`Could not load ${src}`)); };
      document.head.appendChild(el);
    });
    scriptPromises.set(src, p);
  }
  return p;
};

export class CpuPose {
  private detector: any = null;
  private loading: Promise<void> | null = null;
  private callback: ((results: any) => void) | null = null;
  private closed = false;

  setOptions(_options?: unknown) { /* the compatibility engine has a single (lite) model */ }
  onResults(callback: (results: any) => void) { this.callback = callback; }

  initialize(): Promise<void> {
    if (!this.loading) {
      this.loading = this.load().catch((e) => { this.loading = null; throw e; });   // a failed load can be retried
    }
    return this.loading;
  }

  private async load() {
    const w = window as any;
    if (!w.tf || !w.poseDetection) {
      // tf-core must be first; the rest depend on it
      for (const src of SCRIPTS) await loadScript(src);
    }
    w.tf.wasm.setWasmPaths(WASM_BASE);
    await w.tf.setBackend("wasm");
    await w.tf.ready();
    this.detector = await w.poseDetection.createDetector(w.poseDetection.SupportedModels.BlazePose, {
      runtime: "tfjs",
      modelType: "lite",
      enableSmoothing: true,
      detectorModelUrl: DETECTOR,
      landmarkModelUrl: LANDMARK_LITE,
    });
  }

  async send({ image }: { image: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement }) {
    await this.initialize();
    if (this.closed || !this.detector) return;
    const poses = await this.detector.estimatePoses(image, { flipHorizontal: false });
    if (this.closed) return;
    const w = (image as HTMLVideoElement).videoWidth || (image as HTMLImageElement).naturalWidth || (image as HTMLCanvasElement).width || 1;
    const h = (image as HTMLVideoElement).videoHeight || (image as HTMLImageElement).naturalHeight || (image as HTMLCanvasElement).height || 1;
    const pose = poses && poses[0];
    if (!pose || !pose.keypoints || pose.keypoints.length < 33) { this.callback?.({ image }); return; }   // nobody in view: same as MediaPipe
    const poseLandmarks = pose.keypoints.map((k: any) => ({ x: k.x / w, y: k.y / h, z: (k.z ?? 0) / w, visibility: k.score ?? 0 }));
    const poseWorldLandmarks = pose.keypoints3D && pose.keypoints3D.length >= 33
      ? pose.keypoints3D.map((k: any) => ({ x: k.x, y: k.y, z: k.z, visibility: k.score ?? 0 }))
      : undefined;
    this.callback?.({ poseLandmarks, poseWorldLandmarks, image });
  }

  close() {
    this.closed = true;
    try { this.detector?.dispose?.(); } catch { /* ignore */ }
    this.detector = null;
  }
}

// ---- offline readiness: put the engine's files in the browser cache during the first visit ----
// The service worker (next.config.js) serves these URLs from the same cache, cache-first, so later sessions work without internet.
export const ENGINE_CACHE = "asana-engine-v1";
const MP = `${CDN}@mediapipe/pose/`;

const putInCache = async (cache: Cache, url: string): Promise<Response | null> => {
  try {
    const hit = await cache.match(url);
    if (hit) return hit;
    const res = await fetch(url, { mode: "cors", credentials: "omit" });
    if (!res.ok) return null;
    await cache.put(url, res.clone());
    return res;
  } catch { return null; }
};

const modelFiles = async (cache: Cache, root: string): Promise<string[]> => {
  const manifestUrl = `${root}/model.json?tfjs-format=file`;
  const res = await putInCache(cache, manifestUrl);
  if (!res) return [];
  try {
    const json = await res.clone().json();
    const shards: string[] = (json.weightsManifest || []).flatMap((m: any) => m.paths || []);
    return shards.map((p) => `${root}/${p}?tfjs-format=file`);
  } catch { return []; }
};

/** Prefetch everything the chosen engine needs. Returns true when every file is in the cache. */
export async function warmEngineCache(kind: EngineMode, coarsePointer: boolean, shouldStop: () => boolean = () => false): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  const cache = await caches.open(ENGINE_CACHE);
  let urls: string[];
  if (kind === "gpu") {
    urls = [
      `${MP}pose_solution_packed_assets_loader.js`, `${MP}pose_solution_simd_wasm_bin.js`, `${MP}pose_solution_simd_wasm_bin.wasm`,
      `${MP}pose_solution_packed_assets.data`, `${MP}pose_web.binarypb`, `${MP}pose_landmark_lite.tflite`,
      ...(coarsePointer ? [] : [`${MP}pose_landmark_full.tflite`]),
    ];
  } else {
    urls = [...SCRIPTS, `${WASM_BASE}tfjs-backend-wasm-simd.wasm`, `${WASM_BASE}tfjs-backend-wasm.wasm`];
    urls.push(...(await modelFiles(cache, DETECTOR)), ...(await modelFiles(cache, LANDMARK_LITE)));
  }
  let ok = true;
  for (const u of urls) {          // one at a time: this runs in the background and must never compete with the camera
    if (shouldStop()) return false;
    if (!(await putInCache(cache, u))) ok = false;
  }
  return ok;
}
