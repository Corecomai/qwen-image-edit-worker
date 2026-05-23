FROM runpod/pytorch:1.0.2-cu1281-torch260-ubuntu2204

WORKDIR /app

# Install system deps once in a single layer to keep image size down
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# HF_TOKEN and MODEL_ID are injected at RunPod endpoint config, not baked here
CMD ["python", "-u", "handler.py"]
