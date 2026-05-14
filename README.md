# Qwen Image Edit — Mode A (Pure Serverless)

**Model: Qwen/Qwen-Image-Edit-2511** | Image editing via text instruction  
**Target cost: 7.9 paisa/image** | RunPod RTX 3090 Community | Scale to zero when idle

---

## What This Does

Takes one or more input images + a text instruction and returns an edited image.

```
input.png + "make it look like a watercolor painting" → output.png
img1.png + img2.png + "put both subjects in a park" → output.png
```

---

## Step 1 — HuggingFace Setup

```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF read token
huggingface-cli whoami         # verify
```

---

## Step 2 — Build & Push Docker Image

```bash
cd worker/
docker build -t your-dockerhub/qwen-image-edit:latest .
docker push your-dockerhub/qwen-image-edit:latest
```

---

## Step 3 — Deploy on RunPod Serverless

1. Go to [runpod.io](https://runpod.io) → **Serverless** → **New Endpoint**
2. Select **Custom Source** → paste your Docker image URL
3. GPU: **RTX 3090 (Community Cloud)**
4. Set:
   - **Min Workers = 0** (scale to zero — pay ₹0 when idle)
   - **Max Workers = 3** (auto-scales for burst)
   - **Container Disk = 40 GB** (model cache)
5. Environment Variables:
   ```
   HF_TOKEN=<your token>
   MODEL_ID=Qwen/Qwen-Image-Edit-2511
   ```
6. Deploy → copy the **Endpoint ID**

---

## Step 4 — Configure Local Environment

```bash
cp .env.example .env
# Fill in RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, HF_TOKEN
pip install requests Pillow
```

---

## Step 5 — Edit an Image

```bash
source .env   # or: export $(cat .env | xargs)

# Single image edit
python client.py \
  --image photo.png \
  --prompt "make it look like a Studio Ghibli painting" \
  --out result.png

# Multi-image edit
python client.py \
  --image person1.png \
  --image person2.png \
  --prompt "place both people on a beach at sunset" \
  --out result.png
```

First call after idle: ~2 min cold start (model loading into VRAM).  
Subsequent calls: ~5 sec/image.

---

## Step 6 — Keep-Alive (Eliminate Cold Starts)

### Option A: Run as daemon
```bash
pip install Pillow   # needed for keepalive's dummy image
python keepalive.py
```

### Option B: System cron (recommended)
```bash
crontab -e
# Add this line:
*/4 * * * * cd /Users/shubhammohape/Documents/QwenAISetup && source .env && python keepalive.py --once >> /tmp/keepalive.log 2>&1
```

**Cost: ~₹10.5/month** to stay warm 24/7.

---

## API Parameters

| Parameter | Default | Description |
|---|---|---|
| `image` | required | Base64-encoded input image (single) |
| `images` | — | List of base64-encoded images (multi-image) |
| `prompt` | required | Text instruction for editing |
| `steps` | 40 | Inference steps (lower = faster, lower quality) |
| `cfg_scale` | 4.0 | How closely to follow the image consistency |
| `guidance_scale` | 1.0 | How closely to follow the text prompt |
| `negative_prompt` | `" "` | What to avoid in the output |
| `seed` | -1 | -1 for random, any int for reproducibility |

---

## Cost Reference

| Daily Volume | Monthly Cost | Cost/Image |
|---|---|---|
| 100/day | ₹235 | 7.9 paisa |
| 500/day | ₹1,175 | 7.9 paisa |
| 1,000/day | ₹2,350 | 7.9 paisa |
| 5,000/day | ₹11,757 | 7.9 paisa |
| 10,000/day | ₹23,514 | 7.9 paisa |

Keep-alive adds ~₹10.5/month flat.
