# Switch to widely available CUDA 11.8 runtime (compatible wheel exists: cu118)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ca-certificates \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upgrade pip first and set wheels cache dir
ENV PIP_NO_CACHE_DIR=1
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PaddlePaddle GPU build for CUDA 11.8 per docs
# https://www.paddleocr.ai/latest/en/version3.x/installation.html
RUN python3 -m pip install --no-cache-dir paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Copy requirements and install remaining deps
COPY requirements.txt /workspace/requirements.txt
# Install requirements first to leverage Docker layer cache; then copy app
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

# Copy application files
COPY app.py /workspace/app.py
COPY rp_handler.py /workspace/rp_handler.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Pre-download PaddleOCR models to avoid cold-start download in serverless
RUN python3 - << 'PY'
from paddleocr import PaddleOCR
import numpy as np

# Preload popular manga languages to bake weights into the image
langs = ["japan", "korean", "ch", "chinese_cht", "en"]
dummy = np.zeros((64, 64, 3), dtype=np.uint8)

for lang in langs:
    try:
        print(f"[preload] Initializing PaddleOCR for lang={lang} ...")
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        # Trigger lazy downloads (det/rec/cls) with a tiny run
        _ = ocr.ocr(dummy, cls=True)
        print(f"[preload] OK: {lang}")
    except Exception as e:
        print(f"[preload] WARN: {lang} -> {e}")
PY

EXPOSE 7861

CMD ["/workspace/start.sh"]


