FROM nvidia/cuda:12.6.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        wget \
        ca-certificates \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upgrade pip first
RUN python3 -m pip install --upgrade pip

# Install PaddlePaddle GPU build (CUDA 12.6) per docs
# https://www.paddleocr.ai/latest/en/version3.x/installation.html
RUN python3 -m pip install --no-cache-dir paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Copy requirements and install remaining deps
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

# Copy application files
COPY app.py /workspace/app.py
COPY rp_handler.py /workspace/rp_handler.py
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 7861

CMD ["/workspace/start.sh"]


