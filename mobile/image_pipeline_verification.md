# Representative Asana Image Verification

## Scope

This verification exercised the same MediaPipe Pose configuration embedded in the Android landmark worker and the same shared backend sequence used by the Practice screen: **landmark detection → occlusion recovery → 15-angle extraction → frame analysis**. The fixtures were full-body reference images for three poses from the app’s supported catalogue.

| Fixture | Reference source | Landmark result | Shared-backend result | Android behavior confirmed |
|---|---|---:|---|---|
| Warrior II | Yoga Journal reference image | 33/33 landmarks; 33 at ≥0.60 visibility | `warrior_2`, score `1.000`, no recovered joints | Target is recognized; the UI shows **Warrior II**, confirms the target, and provides safe coaching. |
| Cobra | YouAligned reference image | 33/33 landmarks; 33 at ≥0.60 visibility | `lunge_pose`, score `0.265`, no recovered joints | Mismatch behavior is correct and conservative: the UI identifies the returned pose, marks it as different from the selected Cobra target, and shows a pause-and-adjust cue rather than presenting a false success. |
| Plank | Yoga Collective reference image | No complete landmark set | No backend request sent | No-person behavior is correct: the Practice screen clears stale landmarks and directs the user to include their full body in frame. |

## Interpretation

The image pass validates a complete positive path for **Warrior II**, including the production backend response. It also validates two important safe failure paths: a pose-classification mismatch for Cobra and incomplete landmark detection for the selected Plank reference. These outcomes are deliberately surfaced as mismatch/no-person states in the Android client; the app does not fabricate a pose match or score.

> The fixture result is not a substitute for physical-device testing. It verifies the worker configuration and backend payload sequence with still images; final recognition quality still depends on camera distance, light, framing, and movement on the phone.

## Reproduction Assets

The temporary local fixture harness lives under `test-assets/pose-pipeline/`. It uses the app’s MediaPipe configuration and captures exact landmark rows for the deterministic backend runner at `scripts/run-fixture-backend-check.cjs`.

## Expanded Cobra and Plank Calibration

The second pass added one more Cobra reference and one more Plank reference. Both Cobra references produced complete landmarks, yet both were classified as `lunge_pose`; the first scored `0.265` and the second `0.686`. The primary Plank reference produced complete landmarks on the repeated run but the backend returned `transition/unknown` with a `0.000` score. The alternate Plank reference produced no complete landmark set.

| Expected pose | Fixture outcome | Safe Android response |
|---|---|---|
| Cobra | Complete landmarks, consistently returned as `lunge_pose` | The app now says which pose was read and that it differs from Cobra, then directs the user to the pose guide. |
| Plank | One complete result returned `transition/unknown`; one alternate reference had no complete landmarks | The app distinguishes an in-frame but unidentifiable pose from an incomplete frame and gives a specific recovery cue for each. |

The selected repository checkout contains the FastAPI entry point but not the imported classifier/router source needed to inspect or retrain the deployed label map. The expanded remote tests therefore establish an **observed coverage limitation** for Cobra and Plank, rather than attributing a model-training cause that the available source cannot verify.
