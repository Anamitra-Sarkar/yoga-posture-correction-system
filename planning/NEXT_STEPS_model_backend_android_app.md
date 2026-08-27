# AsanaAI: Model/Backend Improvement + Native Android App

## ⚠️ RESUME NOTES (read this first)

Work on this plan paused here: **Lightning AI banned the user's account**, killing
the VM/Studio that Phases A and B depend on (SSH now refused, "Studio is not
running"). Before resuming Phase A/B work, next session needs to:

1. **Set up a replacement compute environment.** User's explicit instruction:
   *"use modal api key from now .. lightning ai banned me .. use modal L4 gpu or
   whatever u need .. using kaggle would be better for free gpu .. but anyways"* —
   i.e. prefer **Modal** (paid, has an API key available) as the primary path, but
   **Kaggle's free GPU/CPU quota** is explicitly flagged as a good option too
   (already used successfully earlier this session for CPU-parallel landmark
   extraction before the protobuf/TensorFlow conflict was hit — see if that's
   avoidable this time by not needing mediapipe on Kaggle's stock image, or by
   pinning deps more carefully).
2. **Check STGCN training status.** Last confirmed: epoch 113/120, val acc ~86%,
   climbing normally, on the (now-dead) Lightning VM. Every epoch that improved
   validation loss uploaded its checkpoint to `Arko007/yoga-posture-models` as
   `stgcn_sequence_model_v2.pth` immediately (see `train_stgcn_gpu.py`'s per-epoch
   HF upload logic) — so **the last-good checkpoint should have survived** even
   though the live training session died mid-run. First step on the new compute
   environment: download that checkpoint, confirm which epoch it's actually from,
   and decide whether to resume training from there (if the script/data supports
   clean resume) or accept it as final and move to evaluation/promotion.
3. **MLP training already fully completed** (40/40 epochs, best val loss 0.3868,
   90.87% val pose acc) before the VM died — not affected by this interruption.
4. All local files (training scripts, rules_classifier.py, geometry.py, the 35-image
   real-world test harness, download_test_images_vm.py, sweep_2d_vs_3d.py, etc.)
   are safe on the local machine / GitHub repo, not lost — only the remote compute
   session and anything that lived solely on that VM's disk (e.g. any raw video/
   landmark files not yet re-uploaded anywhere) is gone and would need re-fetching
   from the original HF dataset source if still needed.

## Context

The deadline moved out by a day, giving real headroom for improvement rather than
firefighting. Three things prompted this plan: (1) an invigilator was impressed and
suggested "make it an app," which the user wants interpreted as *both* keeping the
existing Vercel-hosted Next.js PWA and shipping a real installable Android app; (2)
the user wants the model/backend genuinely improved, not just patched — more pose
coverage, better accuracy; (3) the user explicitly flagged MediaPipe's z-depth
unreliability (the root cause diagnosed earlier this session, Section XI-F of the
paper) and wants it properly addressed or replaced, not just worked around by
dropping a dimension forever.

Current state going into this: MLP finished all 40 epochs (best val loss 0.3868,
90.87% val pose acc) but real-world accuracy never moved off ~0-6% regardless of
epoch — confirms this is a feature-representation ceiling, not an undertraining
problem. STGCN is still training (120 epochs total, currently mid-run, climbing
normally). Live production currently exposes only 3 poses (warrior_2, mountain_pose,
cobra_pose) via the MLP+rules hybrid in `backend/app/routers/pose.py`; plank,
tree_pose, and chair_pose are disabled (`DISABLED_POSES` in
`backend/app/utils/rules_classifier.py`) after real-world testing showed them
unreliable. Warrior I and Downward Dog have templates/rules but were never enabled
for live selection.

## Phase A — Address the MediaPipe z-depth root cause

**A1. Add `pose_world_landmarks` as a second independent signal (low-risk, proven).**
Already validated this session at 42.9% real-world accuracy (comparable to the
2D-zeroed approach's 45.7%) but retains genuine depth information the 2D approach
discards. Concretely:
- Add a second angle-extraction path using MediaPipe's `pose_world_landmarks` (metric-scale,
  separately calibrated) alongside the existing 2D-zeroed path in
  `backend/app/utils/geometry.py` / `frontend/src/utils/geometry.ts`.
- Extend the hybrid in `backend/app/routers/pose.py` from a 2-way (MLP vs. 2D-rules)
  comparison to a 3-way vote (MLP, 2D-rules, world-landmarks-rules), using majority
  agreement or a confidence-ordered fallback chain when the three disagree.
- Since world-landmarks isn't available client-side (frontend only sends computed
  angles today, not raw landmarks), this requires the frontend to also send raw
  landmark coordinates (or the backend's Gradio demo path, which already has raw
  MediaPipe access server-side, becomes the natural place to prototype this first).
- Re-verify against the existing 35-image real-world set plus a freshly-sourced set
  for the poses being re-enabled in Phase B, using the same Wikimedia-Commons
  downloader pattern already built (`download_test_images_vm.py`) and the same
  40-trial randomized threshold sweep methodology for rigor.

**A2. Investigate a lightweight 2D→3D lifting model (stretch goal, clearly scoped).**
This is the "real" fix the paper's Future Work section names. Framed as an
experiment with a hard go/no-go gate, not a guaranteed deliverable:
- Look for a small, pretrained, CPU-feasible 2D-to-3D lifter (e.g. a compact
  Martinez-et-al.-style MLP lifter, a few MB, millisecond CPU inference) that maps
  MediaPipe's reliable 2D (x,y) joint positions to calibrated 3D positions
  independent of MediaPipe's own unreliable z.
- If a usable pretrained checkpoint exists (ONNX or PyTorch, compatible license):
  integrate it as a third candidate feature pipeline, re-run the same 35-image +
  40-trial sweep methodology against it.
- **Go/no-go**: adopt only if it measurably beats both the current 2D-rules (45.7%)
  and world-landmarks (42.9%) baselines on real-world data. If no suitable
  pretrained model can be found without a from-scratch training project (which
  would need 3D mocap data we don't currently have access to), stop here and
  document the investigation's outcome in the paper rather than open-ending it.

## Phase B — Expand pose vocabulary using the proven methodology

For each of `chair_pose`, `tree_pose`, `plank` (currently disabled) and `warrior_1`,
`downward_dog` (never enabled), repeat the exact rigorous process already validated
for warrior_2/cobra/mountain this session:
1. Source 8-10 real-world images per pose via the existing Wikimedia Commons
   downloader (`download_test_images_vm.py`, already handles false-positive
   filtering) — run on the Lightning AI VM per the "don't download to local" rule.
2. Extract both 2D-zeroed and world-landmarks-based angles (Phase A1's dual path).
3. Empirically tune/verify thresholds against real angle data (not assumed values),
   using the same `dump_2d_angles.py`-style inspection and targeted, evidence-based
   fixes (as done for the chair/warrior symmetry fix) rather than guessing.
4. Validate via the 40-trial randomized threshold sweep (`sweep_2d_vs_3d.py` pattern)
   for statistical confidence, not a single lucky run.
5. Only remove from `DISABLED_POSES` / add to `POSE_SELECTOR_OPTIONS` in
   `frontend/src/pages/index.tsx` once real-world accuracy clears a real bar
   (roughly matching the ~40%+ already established for the live 3), with the same
   honest reporting discipline used throughout — if a pose still can't be made
   reliable, it stays disabled and that's recorded as a finding, not hidden.

**STGCN**: let the current 120-epoch run finish (already climbing normally, ~86%
val acc). Once done, evaluate for promotion to the live `stgcn_sequence_model.pth`
using the same real-video temporal-trace test already built
(`eval_stgcn_temporal_trace.py`) before ever overwriting the production checkpoint.

## Phase C — Native Android app via Capacitor

**Approach: Capacitor in server mode**, pointing the Android WebView at the live
Vercel URL rather than bundling a static export. This app has no Next.js API-route
dependency (`frontend/src/utils/api.ts` always resolves to the external HF Space
URL), is fundamentally online-first anyway (camera analysis needs a live backend
every session), and server mode means routine JS/CSS/logic fixes just need a normal
`git push` — no APK rebuild/resubmit cycle. `next-pwa`'s existing service worker
still gets to run inside the WebView for app-shell resilience, at no extra cost.

**Setup (all commands from `frontend/`):**
```
npm install @capacitor/core @capacitor/android
npm install -D @capacitor/cli
npm install @capacitor/camera @capacitor/splash-screen @capacitor/status-bar @capacitor/app
npx cap init "AsanaAI" "com.asanaai.app" --web-dir=".next"
# set server.url to the production Vercel domain in the generated capacitor.config.ts
npx cap add android
npx cap sync android
```
This creates only new files/directories (`capacitor.config.ts`, `android/`) — zero
changes required to `next.config.js`, `frontend/src/`, or the Vercel build pipeline.
Add `android/app/build/`, `android/.gradle/`, `android/local.properties` to
`.gitignore`; commit the rest of `android/` (build.gradle, AndroidManifest.xml,
MainActivity.java) so the project is reproducible.

**Camera/MediaPipe in the WebView — the one real risk area, budget explicit test
time rather than assuming it works:**
- Add `<uses-permission android:name="android.permission.CAMERA" />` +
  `<uses-feature android:name="android.hardware.camera" ... />` to
  `android/app/src/main/AndroidManifest.xml`.
- `@capacitor/camera` must be installed even though the app uses raw
  `getUserMedia`/`<video>`/`<canvas>`, not Capacitor's native Camera API — its
  presence is what wires `WebChromeClient.onPermissionRequest` so in-page
  `getUserMedia()` calls actually trigger Android's runtime permission dialog.
  This is the standard "camera works in Chrome, black screen in the APK" gotcha.
- MediaPipe's CDN `<script>` tags (jsdelivr) should load unchanged in server mode —
  standard cross-origin HTTPS script loading, same `runtimeCaching` cache-first rule
  already covers it. Verify WASM execution and cold-load time on a **real physical
  device** (camera passthrough on emulators is unreliable and won't validate real
  MediaPipe performance).
- Verify the pinch-to-zoom gesture logic (`index.tsx` ~565-686) isn't fighting the
  WebView's own pinch-to-zoom-the-page gesture — check the viewport meta tag has
  `user-scalable=no` or equivalent.

**Build & test:** Android Studio is the practical path (SDK/emulator/signing setup,
live logcat) — `npx cap open android`, then run on a real device via USB debugging
for camera testing. A bare command-line debug APK is also possible
(`cd android && ./gradlew assembleDebug`, then `adb install`) for quick sideload
testing without Android Studio.

**"Stylish and responsive" polish (scoped for a student project, not enterprise):**
status bar theming to match `manifest.json`'s `theme_color` via `@capacitor/status-bar`;
branded splash screen + app icon generated from the existing
`public/icons/icon-512.png` via `@capacitor/assets`; safe-area CSS padding if the
existing responsive styles don't already cover notches; sensible hardware
back-button handling via `@capacitor/app`'s `backButton` listener (default WebView
behavior can otherwise instantly exit the app); explicit portrait orientation lock
for the camera UI. Explicitly out of scope: custom native transitions, Play Store
signing/publishing (unless separately requested), adaptive-icon layering.

**Estimate:** ~2-3 focused days total (init+first build: 0.5-1 day; camera/MediaPipe
verification on a real device: 0.5-1 day; polish pass: 0.5-1 day) — appropriately
sized as an add-on, which is the reason to choose Capacitor over a React Native
rewrite here.

## Verification

- Every pose re-enablement gated on the same real-world test harness already built
  this session (VM-hosted downloader + rules classifier + 40-trial sweep), not on
  offline validation-split accuracy alone — this is the one methodological
  non-negotiable carried over from the whole "fake accuracy" conversation earlier.
- Backend changes tested against the live HF Space exactly as done throughout this
  session (direct curl tests with real angle data) before being called done.
- Android build verified by producing and installing a debug APK (or emulator run)
  confirming camera permission + MediaPipe load + backend connectivity all work
  inside the WebView, not just that the build compiles.
- No changes to the currently-live, working 3-pose hybrid deployment until its
  replacement is independently verified — same non-destructive discipline as the
  rest of this session.
