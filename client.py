"""
RunPod client for wlsdml1114/qwen_image_edit template (Qwen-Image-Edit-2511 via ComfyUI).

Usage:
    # Single image
    python client.py --image photo.png --prompt "make it watercolor" --out result.png

    # Two images
    python client.py --image img1.png --image img2.png --prompt "merge both subjects on a beach" --out result.png

    # Three images
    python client.py --image a.png --image b.png --image c.png --prompt "..." --out result.png
"""

import argparse
import base64
import os
import random
import time

import requests

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}


def _encode(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def edit(
    images: list[str],
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
) -> bytes:
    if not 1 <= len(images) <= 3:
        raise ValueError("Provide 1, 2, or 3 images")

    if seed == -1:
        seed = random.randint(0, 2**31)

    payload: dict = {
        "input": {
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
        }
    }

    keys = ["image_base64", "image_base64_2", "image_base64_3"]
    for key, path in zip(keys, images):
        payload["input"][key] = _encode(path)

    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]

    for _ in range(120):
        time.sleep(3)
        r = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()

        if data["status"] == "COMPLETED":
            return base64.b64decode(data["output"]["image"])
        if data["status"] == "FAILED":
            raise RuntimeError(f"Job failed: {data.get('error')}")

    raise TimeoutError("Job did not complete within 6 minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", required=True, metavar="PATH",
                        help="Input image (repeat up to 3 times for multi-image editing)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="output.png")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    print(f"Editing {len(args.image)} image(s): {args.prompt!r}")
    t0 = time.time()
    image_bytes = edit(args.image, args.prompt, args.width, args.height, args.seed)
    elapsed = time.time() - t0

    with open(args.out, "wb") as f:
        f.write(image_bytes)

    print(f"Saved to {args.out} in {elapsed:.1f}s")
