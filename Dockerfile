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

# Pre-fetch PaddleOCR weights (det/rec/cls) to avoid cold-start downloads
RUN bash -euxo pipefail -c '\
  ROOT=/root/.paddleocr/whl; \
  mkdir -p "$ROOT/det/ml/Multilingual_PP-OCRv3_det_infer"; \
  cd "$ROOT/det/ml/Multilingual_PP-OCRv3_det_infer"; \
  wget -q --retry-connrefused --tries=3 --timeout=30 \
    https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar \
    && tar -xf Multilingual_PP-OCRv3_det_infer.tar || true; \
  for LANG in korean en japan ch chinese_cht; do \
    case "$LANG" in \
      korean) REC_URL=https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar ;; \
      en) REC_URL=https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar ;; \
      japan) REC_URL=https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar ;; \
      ch) REC_URL=https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar ;; \
      chinese_cht) REC_URL=https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese_cht/chinese_cht_PP-OCRv4_rec_infer.tar ;; \
    esac; \
    mkdir -p "$ROOT/rec/$LANG/${LANG}_PP-OCRv4_rec_infer"; \
    cd "$ROOT/rec/$LANG/${LANG}_PP-OCRv4_rec_infer"; \
    wget -q --retry-connrefused --tries=3 --timeout=30 "$REC_URL" \
      && tar -xf "$(basename "$REC_URL")" || true; \
  done; \
  mkdir -p "$ROOT/cls/ch_ppocr_mobile_v2.0_cls_infer"; \
  cd "$ROOT/cls/ch_ppocr_mobile_v2.0_cls_infer"; \
  wget -q --retry-connrefused --tries=3 --timeout=30 \
    https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar \
    && tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar || true \
'

EXPOSE 7861

CMD ["/workspace/start.sh"]


