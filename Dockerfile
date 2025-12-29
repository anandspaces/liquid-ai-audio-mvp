FROM python:3.11-slim-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    TOKENIZERS_PARALLELISM=false

# Install system dependencies (including audio libraries)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Set working directory
WORKDIR /app

# Install CPU-only PyTorch FIRST
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
RUN pip install --no-cache-dir \
    transformers>=4.55.0 \
    accelerate \
    sentencepiece \
    protobuf \
    fastapi \
    uvicorn[standard] \
    requests \
    python-multipart \
    soundfile \
    librosa \
    pydub \
    einops

# Copy the local model directory (with custom code)
COPY LFM2-Audio-1.5B /models/LFM2-Audio-1.5B

# Copy application code
COPY *.py ./

# Expose port
EXPOSE 8091

# Run with proper signal handling
CMD ["python", "-u", "api_server_audio.py"]