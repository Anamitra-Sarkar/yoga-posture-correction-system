# Incoming Agent Handoff — AsanaAI Android Companion

## Current Baseline

The active mobile project is located at `/home/ubuntu/asana-ai-mobile`. The latest stable checkpoint before this handoff is **`97fb5cdb`**. It is an Expo SDK 54 / React Native Android companion designed to share the yoga project’s existing backend while remaining a distinct, native mobile product. The original repository clone used for reference is located at `/home/ubuntu/yoga-repo-push`; its Android source had previously been copied to `mobile/` and pushed to the selected GitHub repository.

The product deliberately avoids web embedding. The Practice screen uses `expo-camera`, on-device MediaPipe landmark extraction in `components/pose-detector-worker.tsx`, a native SVG skeleton in `components/pose-skeleton.tsx`, and direct shared-backend calls from `lib/yoga-api.ts`. The camera feed has an optional immersive mode, but normal Practice is designed to remain compact and stable on tall phones.

## Verified Native Features

| Area | Current implementation | Key files |
|---|---|---|
| Navigation | Today, Practice, and Settings have clear selected-tab states; compact-screen safe-area spacing was corrected. | `app/(tabs)/_layout.tsx` |
| Practice camera | Camera permissions, front/rear switching, timed snapshots, native skeleton overlay, optional immersive camera mode, haptics, and native speech are implemented. | `app/(tabs)/practice.tsx` |
| Pose catalogue | The selector is intentionally limited to six web-app-supported asanas: Warrior II, Cobra, Mountain, Tree, Plank, and Downward Dog. | `lib/asana.ts`, `components/pose-picker.tsx` |
| Guidance | Each supported pose has native reference steps. | `components/pose-guide.tsx`, `lib/asana.ts` |
| Local data | Preferences and completed-session summaries persist through AsyncStorage. | `lib/preferences.ts`, `lib/practice-history.ts` |
| Diagnostics | Settings includes a camera/service readiness probe. Practice explicitly distinguishes pose mismatch, unidentifiable transition, incomplete framing, offline service, and camera permission states. | `lib/device-readiness.ts`, `lib/coaching-diagnostics.ts` |

## Image-Pipeline Findings

Run and preserve the reproducible verification assets before changing recognition behavior:

| Expected pose | Landmark result | Backend result | Required current app behavior |
|---|---|---|---|
| Warrior II | Complete 33/33 landmarks | `warrior_2`, score `1.000` | Confirm target and provide normal coaching. |
| Cobra (two references) | Complete 33/33 landmarks | Both returned `lunge_pose`, scores `0.265` and `0.686` | State that the coach read Lunge Pose rather than Cobra; do not claim Cobra recognition. |
| Plank (primary) | Complete on a repeated run | `transition/unknown`, score `0.000` | Explain that the body is in frame but the target is not identifiable yet. |
| Plank (alternate) | No complete landmark set | No analysis request | Ask the user to step back and show head, hands, and feet. |

The test artifacts are `test-assets/pose-pipeline/mediapipe-harness.html`, `scripts/serve-pose-fixtures.cjs`, and `scripts/run-fixture-backend-check.cjs`. The full record is `image_pipeline_verification.md`. The harness uses the same MediaPipe configuration and backend sequence as the native worker: **landmark detection → occlusion recovery → 15-angle extraction → frame analysis**.

> Do not turn a selected target into a successful result merely because it is selected. The current diagnostic layer was intentionally added to make uncertain or mismatched model output explicit and safe.

## Known Blocker

The local yoga repository checkout contains the FastAPI entry point but not its imported `services` and `routers` implementation. The deployed backend can be reached and returns frame-analysis responses, but its classifier label map and training source are not available in the supplied checkout. Therefore, the Cobra and Plank issue is an **observed deployed-model coverage limitation**, not a confirmed root cause. Do not claim retraining is complete unless the actual model/data source is acquired.

## Required Continuation Sequence

1. Obtain the missing backend classifier/model source or a documented model artifact and label map.
2. Confirm whether `cobra_pose` and `plank` are trained inference labels. If absent, collect curated full-body camera frames and retrain/calibrate the real classifier rather than changing only UI labels.
3. Re-run the fixture harness after every model change. Preserve Warrior II as a regression-positive control and retain Cobra/Plank mismatch cases as safety regression tests.
4. Test on a physical Android device in realistic light and distance. Validate front/rear camera, full-screen exit, selected tab visibility, camera permission denial, speech feedback, no-person recovery, mismatch copy, and saved session summaries.
5. Keep Android UI native. Do not reintroduce a WebView, copy web-app branding language into the mobile UI, or remove the safety diagnostics to hide classifier uncertainty.

## Validation Commands

```bash
cd /home/ubuntu/asana-ai-mobile
pnpm test
pnpm check
pnpm lint
CI=1 npx expo-doctor
```

At the handoff baseline, the suite reports **16 passing tests**, TypeScript passes, lint passes, and Expo Doctor reports **18/18 checks passed**. The ESLint process may emit a non-blocking module-type performance warning for `eslint.config.js`; it has not caused a validation failure.
