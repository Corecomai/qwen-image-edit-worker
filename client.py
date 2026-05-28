"""
RunPod client for Qwen-Image-Edit-2511 + Lightning LoRA serverless endpoint.

Usage:
    # Basic edit
    python client.py --image photo.png --prompt "make it watercolor"

    # High quality, landscape
    python client.py --image photo.png --prompt "..." --quality high --aspect-ratio 16:9

    # XL portrait
    python client.py --image photo.png --prompt "..." --size xl --aspect-ratio 9:16

    # Multi-image
    python client.py --image a.png --image b.png --prompt "merge both subjects on a beach"

Quality presets: fast (4 steps) | balanced (8, default) | high (16) | ultra (20)
Size presets:    small (512) | medium (768) | large (1024, default) | xl (1280)
Aspect ratios:   1:1 | 16:9 | 9:16 | 4:3 | 3:4 | 3:2 | 2:3  (or explicit --width/--height)
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
    quality: str = "balanced",
    aspect_ratio: str | None = None,
    size: str | None = None,
    width: int | None = None,
    height: int | None = None,
    output_format: str = "png",
    seed: int = -1,
) -> tuple[bytes, dict]:
    if not 1 <= len(images) <= 3:
        raise ValueError("Provide 1, 2, or 3 images")

    if seed == -1:
        seed = random.randint(0, 2**31)

    inp: dict = {
        "prompt": prompt,
        "seed": seed,
        "quality": quality,
        "output_format": output_format,
    }
    if aspect_ratio:
        inp["aspect_ratio"] = aspect_ratio
    if size:
        inp["size"] = size
    if width:
        inp["width"] = width
    if height:
        inp["height"] = height

    payload: dict = {"input": inp}

    if len(images) == 1:
        payload["input"]["image"] = _encode(images[0])
    else:
        payload["input"]["images"] = [_encode(p) for p in images]

    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]

    for _ in range(120):
        time.sleep(3)
        r = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()

        if data["status"] == "COMPLETED":
            output = data["output"]
            return base64.b64decode(output["image"]), output
        if data["status"] == "FAILED":
            raise RuntimeError(f"Job failed: {data.get('error')}")

    raise TimeoutError("Job did not complete within 6 minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="append", required=True, metavar="PATH",
                        help="Input image (repeat up to 3×)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="output.png")
    parser.add_argument("--quality", default="balanced",
                        choices=["fast", "balanced", "high", "ultra"],
                        help="fast=4steps | balanced=8 | high=16 | ultra=20")
    parser.add_argument("--aspect-ratio", default=None,
                        help="e.g. 16:9  9:16  4:3  1:1")
    parser.add_argument("--size", default=None,
                        choices=["small", "medium", "large", "xl"],
                        help="Base resolution: small=512 medium=768 large=1024 xl=1280")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--output-format", default="png", choices=["png", "jpeg"])
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    print(f"Editing {len(args.image)} image(s): {args.prompt!r}  "
          f"[quality={args.quality}, aspect={args.aspect_ratio or 'auto'}, size={args.size or 'large'}]")
    t0 = time.time()
    image_bytes, meta = edit(
        args.image, args.prompt,
        quality=args.quality,
        aspect_ratio=args.aspect_ratio,
        size=args.size,
        width=args.width,
        height=args.height,
        output_format=args.output_format,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    out_path = args.out if args.out != "output.png" else f"output.{args.output_format}"
    with open(out_path, "wb") as f:
        f.write(image_bytes)

    print(f"Saved to {out_path} in {elapsed:.1f}s  "
          f"[{meta.get('width')}×{meta.get('height')}, seed={meta.get('seed')}, steps={meta.get('steps')}]")
