import os
import base64
import io
import torch
import runpod
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipe = None


def load_model():
    global pipe

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    ).to("cuda")

    print(f"Model loaded: {model_id}")


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def handler(job):
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}

    # Accept "image" (single b64) or "images" (list of b64)
    raw = job_input.get("images") or ([job_input["image"]] if "image" in job_input else None)
    if not raw:
        return {"error": "image or images is required"}

    pil_images = [_b64_to_pil(img) for img in raw]
    image_arg = pil_images[0] if len(pil_images) == 1 else pil_images

    steps = int(job_input.get("steps", 40))
    cfg_scale = float(job_input.get("cfg_scale", 4.0))
    guidance_scale = float(job_input.get("guidance_scale", 1.0))
    negative_prompt = job_input.get("negative_prompt", " ")
    seed = job_input.get("seed", -1)

    generator = None
    if seed != -1:
        generator = torch.Generator("cuda").manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            image=image_arg,
            prompt=prompt,
            true_cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=generator,
        )

    buf = io.BytesIO()
    result.images[0].save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image": img_b64,
        "format": "png",
        "seed": seed,
    }


# Pipeline loads once at worker startup — cold start pays this cost once.
load_model()

runpod.serverless.start({"handler": handler})
