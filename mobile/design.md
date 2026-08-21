# AsanaAI Mobile — Interface Design Plan

## Product Direction

AsanaAI Mobile is a **camera-first yoga practice companion** for a person who wants a clear, quiet session rather than a dense analytics dashboard. It is a fully native Android experience: the camera preview, controls, landmark overlay, pose feedback, and session interface are rendered natively while its analysis requests use the shared posture-service contract. It does not embed another product interface. Its visual language will be considered and athletic: warm paper background, charcoal typography, muted olive for progress, and restrained terracotta for action. It will avoid gradients, glowing metric cards, robotic illustrations, fake charts, and generic “AI assistant” imagery.

The app is designed for one-handed use in **portrait 9:16** while remaining safe on compact Android phones, tall 20:9 displays, tablets, display cut-outs, and three-button or gesture navigation. Key actions stay in the lower third of the screen; secondary controls live in sheets or a simple settings screen. The camera stage calculates its height from the actual available window, preserves the device’s preview ratio with `cover` rendering, and uses safe-area insets rather than fixed bottom offsets. Motion supports orientation and confirmation, never decoration: press feedback scales to 0.97 over 100ms; sheets and feedback panels transition in 180–240ms; reduced-motion preferences disable non-essential transitions.

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
| **Practice Camera** | Native live preview, target-pose name, camera switch, start/stop session control, real analysis result, skeleton overlay, and permission/error states. | Preview uses a responsive stage with an unobscured 4:3 training frame and safe-area-aware floating controls. The rear/front switch lives at the preview top edge. Only the backend-returned 33-point skeleton is drawn; absent landmark data shows a clear framing guide instead of a synthetic skeleton. |
| **Pose Picker** | The six supported targets only: Virabhadrasana II, Bhujangasana, Tadasana, Vrikshasana, Phalakasana, and Adho Mukha Svanasana. | A native bottom sheet contains a vertical list with the exact pose names, level, one short alignment cue, and a measured-joint summary. The active pose is marked with a moss indicator, not a large colored card. |
| **Session Summary** | Real values received from the posture service: correctness score, detected pose, joint guidance, and restart action. | One large score block, followed by plain-language feedback rows. No fabricated charts or default scores; unavailable data has an explicit state. |
| **Settings** | Voice guidance, language selection, API connection information, and permission guidance. | Grouped native-style rows with toggles and chevrons. Controls persist locally using AsyncStorage. |
| **Connection / Permission Sheet** | Camera permission denial, no network, service unavailable, or no-person-detected state. | A compact bottom sheet that explains what happened and offers a clear recovery action. It never leaves the user on a blank preview. |

## Primary User Flows

1. **Start a practice:** The user opens Today, confirms or changes the target pose, taps “Begin practice,” grants camera permission if needed, and arrives in the live camera view. The primary control becomes “End session.”
2. **Receive posture guidance:** The user starts native analysis after framing their body. The app sends a compressed camera frame to the posture service, shows “Checking alignment” while a request is active, and draws the returned landmark skeleton, detected pose, correctness score, joint deviations, occlusion state, and safe correction. A service error becomes an explicit recovery message rather than simulated analysis.
3. **Choose a pose:** The user taps the pose row on Today or Practice, chooses from the bottom-sheet list, and returns to the current screen with the new cue applied.
4. **Review a session:** The user ends a practice and sees the last confirmed result. They can restart with the same pose or return to Today.
5. **Adjust preferences:** The user opens Settings, changes voice/language/API preferences, receives haptic confirmation where supported, and the app saves those choices locally.

## Interaction and Accessibility Rules

All primary actions have text labels in addition to icons. Touch targets are at least 44×44 points. Camera, connection, and analysis states use text plus color so no status depends solely on hue. The camera screen uses `keep-awake` while an active practice is running and returns to normal when the user exits. The Android back action dismisses sheets first, then returns from Practice to Today without losing a completed result. Camera switching has a single immediate haptic acknowledgement and is disabled only while a native snapshot request is in flight.
