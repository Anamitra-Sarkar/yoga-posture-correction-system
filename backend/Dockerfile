FROM python:3.9-slim

# Install system dependencies needed for OpenCV, MediaPipe, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirement files and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application package files
COPY app /app/app

# Expose port (default port for Hugging Face Spaces is 7860)
EXPOSE 7860

# Command to run uvicorn on port 7860 targeting the app module inside main
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
