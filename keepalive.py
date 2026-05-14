"""
Keep-alive pinger for RunPod Serverless endpoint.
Sends a minimal request every 4 minutes to hold 1 worker warm.

Cost: ~5 sec × ₹0.000189/sec × 360 pings/day = ₹10.5/month
Run as a background process or system cron job:
    * * * * * cd /Users/shubhammohape/Documents/QwenAISetup && source .env && python keepalive.py --once >> /tmp/keepalive.log 2>&1

Or run as a long-running daemon:
    python keepalive.py
"""

import argparse
import base64
import io
import os
import time
from datetime import datetime

import requests
from PIL import Image

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

# 4 minutes keeps the worker warm without paying for a 24/7 worker
INTERVAL_SECONDS = 4 * 60


def _tiny_image_b64() -> str:
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Built once at startup — reused for every ping
PING_PAYLOAD = {
    "input": {
        "prompt": "ping",
        "image_base64": _tiny_image_b64(),
        "seed": 42,
        "width": 64,
        "height": 64,
    }
}


def ping() -> bool:
    try:
        resp = requests.post(f"{BASE_URL}/runsync", json=PING_PAYLOAD, headers=HEADERS, timeout=120)
        resp.raise_for_status()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] ping OK — status {resp.json().get('status')}")
        return True
    except Exception as exc:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] ping FAILED — {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="send one ping and exit (for cron)")
    args = parser.parse_args()

    if args.once:
        ping()
    else:
        print(f"Keep-alive daemon started (interval={INTERVAL_SECONDS}s)")
        while True:
            ping()
            time.sleep(INTERVAL_SECONDS)
