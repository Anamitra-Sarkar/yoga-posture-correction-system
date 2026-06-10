FROM python:3.9-slim

# Install system dependencies needed for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy application files
COPY app.py /app/app.py

# Install python dependencies (CPU-only torch keeps footprint minimal to prevent OOMs)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    pandas \
    scikit-learn \
    fastapi \
    uvicorn \
    pydantic \
    huggingface_hub \
    requests

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Command to run the uvicorn API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
