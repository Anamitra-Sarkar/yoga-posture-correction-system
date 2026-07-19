import os
import numpy as np
import gradio as gr

import torch

from app.main import app as api
from app.config import settings
from app.services.hf_loader import get_mlp_model
from app.utils.geometry import FEATURE_NAMES, extract_angles_from_landmarks
from app.utils.rules_classifier import classify_pose, score_pose

# ---------------------------------------------------------------------------
# Companion Gradio UI. HF Spaces on the Docker SDK cannot use ZeroGPU (free
# dynamically-allocated GPU quota); the Gradio SDK can. Mounting Gradio onto
# the existing FastAPI `app` (imported above, unchanged) keeps every /api/*
# route the production Next.js frontend already depends on working exactly
# as before -- this file only adds a themed demo page for anyone who opens
# the raw Space URL directly, it does not replace the API.
# ---------------------------------------------------------------------------

THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Lora:wght@600&display=swap');

:root {
    --color-primary: #4f67f4;
    --color-primary-2: #06b6d4;
    --color-success: #059669;
    --color-warning: #d97706;
    --color-error: #dc2626;
    --bg-base: #f7f6f3;
    --bg-surface: #ffffff;
    --color-text: #1a1614;
    --color-text-muted: #3d3530;
}

.gradio-container {
    background: var(--bg-base) !important;
    font-family: 'DM Sans', 'Helvetica Neue', system-ui, sans-serif !important;
    color: var(--color-text) !important;
}

#asana-header {
    background: linear-gradient(135deg, #4f67f4, #06b6d4);
    border-radius: 18px;
    padding: 28px 32px;
    color: #ffffff !important;
    margin-bottom: 20px;
}
#asana-header h1 {
    font-family: 'Lora', Georgia, serif !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin: 0 0 6px 0 !important;
}
#asana-header p { color: rgba(255,255,255,0.92) !important; margin: 0 !important; }

.asana-card {
    background: var(--bg-surface) !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05) !important;
}

#cta-row a.cta-button {
    display: inline-block;
    background: linear-gradient(135deg, #4f67f4, #06b6d4);
    color: #ffffff !important;
    padding: 10px 20px;
    border-radius: 9999px;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9rem;
}
"""

VERCEL_URL = "https://yoga-posture-correction-system.vercel.app"


def analyse_demo_image(image: np.ndarray):
    """
    Runs MediaPipe Pose server-side on a single snapshot (webcam or upload),
    computes the same 15 joint-angle features the production client computes
    in-browser, and classifies it with the exact same MLP + rule-engine
    sanity-check pipeline the live /api/analyse_frame route serves -- so this
    demo reflects the real behavior, not a mock.
    """
    if image is None:
        return "Capture or upload a photo of a yoga pose to see a live analysis.", {}

    try:
        import mediapipe as mp
    except Exception as e:
        return f"MediaPipe unavailable in this environment ({e}).", {}

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        results = pose.process(image)

    if not results.pose_landmarks:
        return "No person detected in the frame. Step back so your full body is visible.", {}

    points = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    angles = extract_angles_from_landmarks(points)
    angles_dict = {FEATURE_NAMES[i]: angles[i] for i in range(15)}

    model, classes = get_mlp_model()
    x_tensor = torch.tensor([angles], dtype=torch.float32).to(settings.DEVICE)
    with torch.no_grad():
        pose_logits, correctness_logit, deviations_pred = model(x_tensor)
        mlp_pose = classes[pose_logits.argmax(1).item()]
        mlp_correctness = torch.sigmoid(correctness_logit).item()
        devs_deg = (deviations_pred[0].cpu().numpy() * 180.0)
        mlp_devs = {FEATURE_NAMES[i]: float(devs_deg[i]) for i in range(15)}

    # Same MLP + rule-engine sanity-check pattern as /api/analyse_frame.
    rule_pose = classify_pose(angles_dict)
    if rule_pose == mlp_pose:
        predicted_pose, correctness, devs = mlp_pose, mlp_correctness, mlp_devs
    else:
        predicted_pose = rule_pose
        correctness, devs = score_pose(rule_pose, angles_dict)

    devs_dict = {name: round(min(180.0, max(0.0, float(val))), 1) for name, val in devs.items()}
    summary = (
        f"**Detected pose:** {predicted_pose.replace('_', ' ').title()}\n\n"
        f"**Correctness score:** {correctness * 100:.1f}%"
    )
    return summary, devs_dict


# Deliberately not named `demo`: HF's Gradio SDK runtime auto-launches any
# top-level variable called `demo` on its own, which raced with our explicit
# uvicorn.run() below and crashed the Space with "address already in use".
with gr.Blocks(title=settings.TITLE) as blocks_ui:
    gr.HTML(
        """
        <div id="asana-header">
            <h1>&#129497; AsanaAI &mdash; Smart Yoga Coach</h1>
            <p>Real-time yoga posture correction &mdash; 3-head ResMLP pose classifier + ST-GCN flow model,
            served from this Space. This page is a lightweight demo of the same backend the full app uses.</p>
        </div>
        """
    )

    with gr.Row(elem_classes=["asana-card"]):
        with gr.Column():
            cam_input = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Snapshot for pose analysis",
            )
            analyse_btn = gr.Button("Analyse Pose", variant="primary")
        with gr.Column():
            result_md = gr.Markdown("Capture or upload a photo of a yoga pose to see a live analysis.")
            deviation_json = gr.JSON(label="Per-joint deviation (degrees from target)")

    analyse_btn.click(fn=analyse_demo_image, inputs=cam_input, outputs=[result_md, deviation_json])

    gr.HTML(
        f"""
        <div id="cta-row" style="text-align:center; margin-top: 24px;">
            <a class="cta-button" href="{VERCEL_URL}" target="_blank">
                Open the full real-time app &rarr;
            </a>
        </div>
        """
    )

# ssr_mode explicitly off: HF Spaces sets GRADIO_SSR_MODE=True by default,
# which spawns a Node SSR subprocess that raced our own uvicorn bind on the
# same port and crashed the Space with "address already in use". Not needed
# for this internal demo page anyway.
app = gr.mount_gradio_app(api, blocks_ui, path="/", css=THEME_CSS, ssr_mode=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
