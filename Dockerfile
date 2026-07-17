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
