# CUDA 11.8 runtime; adjust if your plan uses a different CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: ffmpeg for yt-dlp postproc, libsndfile for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl ca-certificates libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && apt-get install -y python3 python3-pip && \
    python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install GPU PyTorch (CUDA 11.8 wheels)
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.0+cu118 torchaudio==2.1.0+cu118 torchvision==0.16.0+cu118

# Install WhisperX from GitHub (more reliable than PyPI)
RUN pip install git+https://github.com/m-bain/whisperx.git

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
