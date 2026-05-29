FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Disable transformers flash-attention auto-detection — avoids compatibility
    # crashes on GPUs where FA is unavailable or partially supported.
    TRANSFORMERS_NO_FLASH_ATTENTION=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# HF_TOKEN and MODEL_ID are injected via RunPod endpoint env, not baked here.
# RUNPOD_VOLUME_PATH is set automatically by RunPod when a Network Volume is mounted.
CMD ["python", "-u", "handler.py"]
