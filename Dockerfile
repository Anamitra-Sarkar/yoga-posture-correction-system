# NOT the deployment path for the Hugging Face Space.
#
# The Space runs sdk: gradio with app_file: app.py, declared in README.md's
# frontmatter -- confirmed against the HF API, which reports sdk "gradio" and
# does not even have this Dockerfile in the Space repo. That app.py serves the
# Gradio demo page AND mounts the same FastAPI routes, which is why /health and
# /api/* work there.
#
# This file therefore builds something different from what is deployed: it
# copies only backend/app and runs uvicorn directly, with no Gradio UI. Kept
# because it is still a valid way to run the API by itself locally or on any
# container host:
#
#     docker build -t asanaai-api . && docker run -p 7860:7860 asanaai-api
#
# If you change backend behaviour, the Space picks it up from app.py and
# backend/app via the GitHub Actions sync -- not from here.

FROM python:3.10-slim

# Install system dependencies needed for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirement files and install
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application package files
COPY backend/app /app/app

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Command to run uvicorn targeting the app module inside main
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
