# AsanaAI Mobile — Interface Design Plan

## Product Direction

AsanaAI Mobile is a **camera-first yoga practice companion** for a person who wants a clear, quiet session rather than a dense analytics dashboard. The Android experience will complement—not replace—the existing web application. Its visual language will be considered and athletic: warm paper background, charcoal typography, muted olive for progress, and restrained terracotta for action. It will avoid gradients, glowing metric cards, robotic illustrations, fake charts, and generic “AI assistant” imagery.

The app is designed for one-handed use in **portrait 9:16**. Key actions stay in the lower third of the screen; secondary controls live in sheets or a simple settings screen. Motion supports orientation and confirmation, never decoration: press feedback scales to 0.97 over 100ms; sheets and feedback panels transition in 180–240ms; reduced-motion preferences disable non-essential transitions.

## Color and Type System

| Token | Color | Intended use |
|---|---:|---|
| Paper | `#F8F7F2` | Primary screen background |
| Ink | `#1F2621` | Headlines, key labels, iconography |
| Moss | `#5C6B4E` | Primary action, selected states, encouraging progress |
| Terracotta | `#B86145` | Important action, alignment warning, selected pose accent |
| Sage | `#DDE2D8` | Soft selected/preview surfaces |
| Clay | `#E9DED4` | Warm secondary panels and dividers |
| Mist | `#6B736D` | Supporting text and disabled states |
| White | `#FFFFFF` | Raised sheets and legible camera overlays |

Typography uses the platform system face with an editorial hierarchy. Screen titles are semibold with generous tracking and short line lengths. Metadata uses a smaller medium weight, never all caps except for compact section labels. The tone is direct, physical, and compassionate: “Set your stance,” “Camera ready,” and “Take a breath” rather than technical model language.

## Screen List and Layout

| Screen | Primary content and functionality | Layout details |
|---|---|---|
| **Today** | A concise practice invitation, selected pose, recent session status, and entry into practice. | Header has wordmark and muted settings affordance. A large, photo-free practice tile anchors the screen; its bottom-aligned action begins a session. Two compact rows show selected pose and connection state. |
| **Practice Camera** | Live camera preview, target-pose name, start/stop session control, guidance, connection and permission states. | Preview fills the upper 58–62% of the screen with a subtle rounded lower edge. The controls sit in a paper-colored dock at the bottom with a single large thumb-reachable action. Guidance arrives as a calm caption strip above the dock. |
| **Pose Picker** | Available target poses and a concise cue for each. | A native bottom sheet contains a vertical list with an icon, name, and one short alignment cue. The active pose is marked with a moss indicator, not a large colored card. |
| **Session Summary** | Real values received from the posture service: correctness score, detected pose, joint guidance, and restart action. | One large score block, followed by plain-language feedback rows. No fabricated charts or default scores; unavailable data has an explicit state. |
| **Settings** | Voice guidance, language selection, API connection information, and permission guidance. | Grouped native-style rows with toggles and chevrons. Controls persist locally using AsyncStorage. |
| **Connection / Permission Sheet** | Camera permission denial, no network, service unavailable, or no-person-detected state. | A compact bottom sheet that explains what happened and offers a clear recovery action. It never leaves the user on a blank preview. |

## Primary User Flows

1. **Start a practice:** The user opens Today, confirms or changes the target pose, taps “Begin practice,” grants camera permission if needed, and arrives in the live camera view. The primary control becomes “End session.”
2. **Receive posture guidance:** The camera capture sends an image/frame only after the user begins a practice. The app shows “Checking alignment” while a request is active, then renders only the returned detected pose, correctness score, and guidance. A service error becomes an explicit recovery message rather than simulated analysis.
3. **Choose a pose:** The user taps the pose row on Today or Practice, chooses from the bottom-sheet list, and returns to the current screen with the new cue applied.
4. **Review a session:** The user ends a practice and sees the last confirmed result. They can restart with the same pose or return to Today.
5. **Adjust preferences:** The user opens Settings, changes voice/language/API preferences, receives haptic confirmation where supported, and the app saves those choices locally.

## Interaction and Accessibility Rules

All primary actions have text labels in addition to icons. Touch targets are at least 44×44 points. Camera, connection, and analysis states use text plus color so no status depends solely on hue. The camera screen uses `keep-awake` while an active practice is running and returns to normal when the user exits. The Android back action dismisses sheets first, then returns from Practice to Today without losing a completed result.
