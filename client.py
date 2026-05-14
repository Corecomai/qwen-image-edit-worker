"""
RunPod Serverless client for Qwen-Image-Edit-2511.
Usage:
    python client.py --image input.png --prompt "make it look like a watercolor painting" --out output.png
    python client.py --image img1.png --image img2.png --prompt "put both subjects in a park" --out output.png
"""

import argparse
import base64
import os
import time

import requests

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def edit(
    images: list[str],
    prompt: str,
    steps: int = 40,
    cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
    negative_prompt: str = " ",
    seed: int = -1,
) -> bytes:
    encoded = [_encode_image(p) for p in images]

    payload = {
        "input": {
            "image" if len(encoded) == 1 else "images": encoded[0] if len(encoded) == 1 else encoded,
            "prompt": prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "seed": seed,
        }
    }

    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]

    for _ in range(120):
        time.sleep(3)
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=10)
        status_resp.raise_for_status()
        data = status_resp.json()

        if data["status"] == "COMPLETED":
            return base64.b64decode(data["output"]["image"])
        if data["status"] == "FAILED":
            raise RuntimeError(f"Job failed: {data.get('error')}")

    raise TimeoutError("Job did not complete within 6 minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", required=True, metavar="PATH",
                        help="Input image path. Repeat for multi-image editing.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="output.png")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    print(f"Editing {len(args.image)} image(s): {args.prompt!r}")
    t0 = time.time()
    image_bytes = edit(
        args.image, args.prompt, args.steps,
        args.cfg_scale, args.guidance_scale, args.negative_prompt, args.seed,
    )
    elapsed = time.time() - t0

    with open(args.out, "wb") as f:
        f.write(image_bytes)

    print(f"Saved to {args.out} in {elapsed:.1f}s")
