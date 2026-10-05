// Compatibility pose engine for phones whose GPU driver cannot run MediaPipe's WebGL pipeline.
//
// Measured on a POCO M7 Pro 5G (Dimensity 7025 Ultra, PowerVR BXM-8-256, Chrome 154): WebGL contexts are created fine, but the old
// MediaPipe engine aborts inside the driver ("GlScalerCalculator ... frame_unifs_[i] != -1") and MediaPipe Tasks' GPU delegate silently
// returns no poses. The CPU delegate of MediaPipe Tasks works there (~80 ms/frame, the same models), so that is the first fallback.
// If the browser has no WebGL at all, the second fallback is BlazePose on TensorFlow.js' WebAssembly backend (needs no GL, a bit less accurate).
//
// The Tasks engine runs in a Web Worker (public/engine/pose-worker.js): its own globals (the page's legacy MediaPipe engine owns a global
// `Module` that a second MediaPipe wasm module would collide with) and no blocking of the interface.
// Everything runs on the device. Results are mapped to the shape of MediaPipe's legacy `Pose` results (normalised image landmarks +
// metric world landmarks), and the class exposes the part of that API the app uses, so nothing downstream changes. It is only created
// when the fast engine failed on this device.

const CDN = "https://cdn.jsdelivr.net/npm/";
// -- MediaPipe Tasks (CPU delegate)
const TASKS = `${CDN}@mediapipe/tasks-vision@0.10.21`;
const TASKS_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
// -- TensorFlow.js WASM (last resort, no WebGL needed)
const TF = "4.22.0";
const PD = "2.1.3";
const WASM_BASE = `${CDN}@tensorflow/tfjs-backend-wasm@${TF}/dist/`;
const TF_SCRIPTS = [
  `${CDN}@tensorflow/tfjs-core@${TF}/dist/tf-core.min.js`,
  `${CDN}@tensorflow/tfjs-converter@${TF}/dist/tf-converter.min.js`,
  `${CDN}@tensorflow/tfjs-backend-wasm@${TF}/dist/tf-backend-wasm.min.js`,
  `${CDN}@tensorflow-models/pose-detection@${PD}/dist/pose-detection.min.js`,
];
const MODEL_ROOT = "https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/";
const TF_DETECTOR = `${MODEL_ROOT}detector/1`;
const TF_LANDMARK_LITE = `${MODEL_ROOT}landmark/lite/2`;

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
    else { window.localStorage.removeItem(MODE_KEY); window.localStorage.removeItem(TFJS_KEY); }
  } catch { /* private mode: the choice just is not remembered */ }
};
export const cpuEngineSupported = (): boolean => typeof WebAssembly !== "undefined";

// remembers that this device needed the no-WebGL engine, so its files are also kept ready for offline use
const TFJS_KEY = "asana.engine.tfjs";
const usedTfjs = (): boolean => { try { return window.localStorage.getItem(TFJS_KEY) === "1"; } catch { return false; } };

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
  private worker: Worker | null = null;  // MediaPipe Tasks PoseLandmarker (CPU), in a worker
  private pending = new Map<number, { resolve: (r: any) => void; reject: (e: Error) => void }>();
  private seq = 0;
  private detector: any = null;         // TF.js BlazePose (WASM)
  private loading: Promise<void> | null = null;
  private callback: ((results: any) => void) | null = null;
  private closed = false;

  setOptions(_options?: unknown) { /* the compatibility engines use one lite model */ }
  onResults(callback: (results: any) => void) { this.callback = callback; }

  initialize(): Promise<void> {
    if (!this.loading) {
      this.loading = this.load().catch((e) => { this.loading = null; throw e; });   // a failed load can be retried
    }
    return this.loading;
  }

  private async load() {
    try {
      await this.loadTasks();
      return;
    } catch (e) {
      console.warn("MediaPipe Tasks (CPU) unavailable, using TensorFlow.js WASM:", e);
      this.stopWorker();
    }
    await this.loadTfjs();
    try { window.localStorage.setItem(TFJS_KEY, "1"); } catch { /* ignore */ }
  }

  private async loadTasks() {
    const worker = new Worker("/engine/pose-worker.js");
    this.worker = worker;
    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(() => reject(new Error("pose worker timed out")), 60000);
      worker.onerror = (ev) => { window.clearTimeout(timer); reject(new Error(ev.message || "pose worker failed")); };
      worker.onmessage = (ev) => {
        const d = ev.data;
        if (d.type === "ready") { window.clearTimeout(timer); resolve(); return; }
        if (d.type === "error" && d.id === undefined) { window.clearTimeout(timer); reject(new Error(d.message)); return; }
        const waiter = this.pending.get(d.id);                 // frame answers
        if (!waiter) return;
        this.pending.delete(d.id);
        if (d.type === "error") waiter.reject(new Error(d.message)); else waiter.resolve(d);
      };
      worker.postMessage({ type: "init", tasks: TASKS, model: TASKS_MODEL });
    });
  }

  private stopWorker() {
    const w = this.worker;
    this.worker = null;
    if (w) { try { w.postMessage({ type: "close" }); } catch { /* ignore */ } w.terminate(); }
    this.pending.forEach((p) => p.reject(new Error("engine closed")));
    this.pending.clear();
  }

  private async loadTfjs() {
    const w = window as any;
    if (!w.tf || !w.poseDetection) {
      for (const src of TF_SCRIPTS) await loadScript(src);   // tf-core must come first
    }
    w.tf.wasm.setWasmPaths(WASM_BASE);
    await w.tf.setBackend("wasm");
    await w.tf.ready();
    this.detector = await w.poseDetection.createDetector(w.poseDetection.SupportedModels.BlazePose, {
      runtime: "tfjs",
      modelType: "lite",
      enableSmoothing: true,
      detectorModelUrl: TF_DETECTOR,
      landmarkModelUrl: TF_LANDMARK_LITE,
    });
  }

  async send({ image }: { image: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement }) {
    await this.initialize();
    if (this.closed) return;
    if (this.worker) {
      const el = image as HTMLVideoElement;
      const w0 = el.videoWidth || (image as HTMLCanvasElement).width || 0, h0 = el.videoHeight || (image as HTMLCanvasElement).height || 0;
      if (!w0 || !h0) return;
      const scale = Math.min(1, 640 / Math.max(w0, h0));   // the model works on a ~256 px crop: no need to ship a full-size frame
      const bitmap = await createImageBitmap(image as ImageBitmapSource, { resizeWidth: Math.round(w0 * scale), resizeHeight: Math.round(h0 * scale), resizeQuality: "low" });
      const id = ++this.seq;
      const answer = await new Promise<any>((resolve, reject) => {
        const timer = window.setTimeout(() => { this.pending.delete(id); reject(new Error("pose worker did not answer")); }, 10000);
        this.pending.set(id, { resolve: (r) => { window.clearTimeout(timer); resolve(r); }, reject: (e) => { window.clearTimeout(timer); reject(e); } });
        this.worker!.postMessage({ type: "frame", id, ts: performance.now(), bitmap }, [bitmap]);
      });
      if (this.closed) return;
      if (!answer.landmarks) { this.callback?.({ image }); return; }   // nobody in view: same as MediaPipe
      this.callback?.({ poseLandmarks: answer.landmarks, poseWorldLandmarks: answer.world || undefined, image });
      return;
    }
    if (!this.detector) return;
    const poses = await this.detector.estimatePoses(image, { flipHorizontal: false });
    if (this.closed) return;
    const w = (image as HTMLVideoElement).videoWidth || (image as HTMLImageElement).naturalWidth || (image as HTMLCanvasElement).width || 1;
    const h = (image as HTMLVideoElement).videoHeight || (image as HTMLImageElement).naturalHeight || (image as HTMLCanvasElement).height || 1;
    const pose = poses && poses[0];
    if (!pose || !pose.keypoints || pose.keypoints.length < 33) { this.callback?.({ image }); return; }
    const poseLandmarks = pose.keypoints.map((k: any) => ({ x: k.x / w, y: k.y / h, z: (k.z ?? 0) / w, visibility: k.score ?? 0 }));
    const poseWorldLandmarks = pose.keypoints3D && pose.keypoints3D.length >= 33
      ? pose.keypoints3D.map((k: any) => ({ x: k.x, y: k.y, z: k.z, visibility: k.score ?? 0 }))
      : undefined;
    this.callback?.({ poseLandmarks, poseWorldLandmarks, image });
  }

  close() {
    this.closed = true;
    this.stopWorker();
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
  const res = await putInCache(cache, `${root}/model.json?tfjs-format=file`);
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
    urls = [`${TASKS}/vision_bundle.mjs`, `${TASKS}/wasm/vision_wasm_internal.js`, `${TASKS}/wasm/vision_wasm_internal.wasm`, TASKS_MODEL];
    if (usedTfjs()) {
      urls.push(...TF_SCRIPTS, `${WASM_BASE}tfjs-backend-wasm-simd.wasm`, `${WASM_BASE}tfjs-backend-wasm.wasm`);
      urls.push(...(await modelFiles(cache, TF_DETECTOR)), ...(await modelFiles(cache, TF_LANDMARK_LITE)));
    }
  }
  let ok = true;
  for (const u of urls) {          // one at a time: this runs in the background and must never compete with the camera
    if (shouldStop()) return false;
    if (!(await putInCache(cache, u))) ok = false;
  }
  return ok;
}
