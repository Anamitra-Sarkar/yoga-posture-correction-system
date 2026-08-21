# Web App Parity Audit

## Browser Evidence

The live production web application was inspected at `https://yoga-posture-correction-system.vercel.app/`.

- Initial screen evidence: `/home/ubuntu/screenshots/yoga-posture-correct_2026-08-21_04-58-18_6032.webp`
- Pose-selector follow-up: `/home/ubuntu/screenshots/yoga-posture-correct_2026-08-21_04-58-32_6116.webp`

## Supported Target Asanas

The native selector must contain **only** the six browser-observed web-app targets below.

| Backend / app identifier | Displayed Sanskrit name | Common name |
|---|---|---|
| `warrior_2` | Virabhadrasana II | Warrior II |
| `cobra` | Bhujangasana | Cobra |
| `mountain` | Tadasana | Mountain |
| `tree` | Vrikshasana | Tree |
| `plank` | Phalakasana | Plank |
| `downward_dog` | Adho Mukha Svanasana | Downward Dog |

## Observed Web-App Features to Recreate Natively

The browser shows a target-pose selector; pose guide with reference image, level, cues, and measured joints; start-video camera control; camera stream area; posture score; detected pose; static/sequence flow; visibility/fusing status; real-time feedback hub; system-status message; language selection; voice toggle; session timer; and new-session control.

The inactive camera state is intentionally explicit. Before analysis it displays **Camera Off**, a **Start Video** action, and non-final placeholder metrics. Native UX must instead show an honest waiting state until a backend response returns. The visible browser canvas is the landmark/skeleton render surface; the Android client will render an equivalent native SVG overlay from backend landmark data without embedding the web page.
