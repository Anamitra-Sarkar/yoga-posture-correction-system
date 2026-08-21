# AsanaAI Mobile

This directory contains the **native Expo/React Native Android companion** for the yoga posture correction system. It is a separate mobile product, not a WebView wrapper. The app uses the same hosted posture-analysis backend as the existing web product while preserving the original web application unchanged.

## Included experience

The mobile app provides a camera-first yoga practice flow with native front/rear camera switching, optional immersive camera mode, MediaPipe landmark extraction, an SVG skeleton overlay, direct backend analysis, speech guidance, haptics, local session history, settings, and device-readiness checks.

Only these supported target poses are selectable: **Warrior II, Cobra, Mountain, Tree, Plank, and Downward Dog**. The app deliberately does not invent recognition success: mismatch, transition/unknown, incomplete framing, permission, and network states are shown explicitly.

## Local development

Install dependencies and start the development server from this directory:

```bash
pnpm install
pnpm dev
```

To open the project on a connected Android device or emulator, run:

```bash
pnpm android
```

## Validation

Run the following checks before a checkpoint or release candidate:

```bash
pnpm test
pnpm check
pnpm lint
CI=1 npx expo-doctor
```

For documented landmark-to-backend fixture verification, see `image_pipeline_verification.md` and the files under `test-assets/pose-pipeline/`.

## Continuation notes

Read `AGENT_HANDOFF.md` and `todo.md` before changing the native coaching or classifier workflow. In particular, Cobra and Plank safety diagnostics reflect verified backend classifier coverage limits and must not be converted into fabricated success states.
