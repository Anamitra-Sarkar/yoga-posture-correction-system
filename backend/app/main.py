import torch
from fastapi import FastAPI
from app.config import settings
from app.services.hf_loader import initialize_models
from app.routers import pose, correction

# Enforce thread limits on startup
torch.set_num_threads(settings.CPU_THREADS)
torch.set_num_interop_threads(settings.CPU_THREADS)

app = FastAPI(
    title=settings.TITLE,
    version=settings.VERSION,
    description="Modular, production-grade API for Smart Yoga Posture Correction (P05)"
)

# Startup event: Pre-cache models to prevent runtime latency spikes
@app.on_event("startup")
def startup_event():
    print("Startup: Fetching and caching weights from Hugging Face Hub...", flush=True)
    initialize_models()

# Include routers
app.include_router(pose.router, prefix="/api", tags=["Pose Analysis"])
app.include_router(correction.router, prefix="/api", tags=["Postural Corrections"])

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "device": str(settings.DEVICE),
        "app_title": settings.TITLE
    }
