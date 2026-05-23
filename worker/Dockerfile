FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/hf-cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# HF_TOKEN and MODEL_ID are injected at RunPod endpoint config, not baked here
CMD ["python", "-u", "handler.py"]
