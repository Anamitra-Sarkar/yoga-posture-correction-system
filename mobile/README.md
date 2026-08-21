# AsanaAI Mobile

This directory contains the Expo-based Android companion for the existing AsanaAI web application. The original web client and FastAPI service remain unchanged in their existing locations.

## What it includes

The mobile app provides a calm, camera-first yoga practice experience with local pose selection, camera permissions, local language and voice preferences, practice-history storage, and a native shell for the existing live web coach. It does not fabricate posture scores when the live service is unavailable; the embedded coach provides the source of real-time analysis.

## Run locally

Install dependencies and start the Expo development server from this directory:

```bash
pnpm install
pnpm dev
```

To open the native project on an Android device or emulator, use:

```bash
pnpm android
```

## Validation

The mobile project provides the following checks:

```bash
pnpm test
pnpm check
pnpm lint
```

The app is configured as **AsanaAI** with Android package identifier `space.manus.asana.ai.mobile`. Publish/build it through the managed mobile project workflow to generate an APK rather than building an APK directly inside a constrained development environment.
