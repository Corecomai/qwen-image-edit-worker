import os
import base64
import io
import torch
import runpod
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipe = None

LIGHTNING_LORA = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_WEIGHT = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def load_model():
    global pipe

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA capability:{torch.cuda.get_device_capability(0)}", flush=True)
        print(f"VRAM total:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    print(f"PyTorch:        {torch.__version__}", flush=True)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )

    # Load Lightning LoRA — reduces inference from 40 steps to 4 (~10x speedup)
    pipe.load_lora_weights(
        LIGHTNING_LORA,
        weight_name=LIGHTNING_WEIGHT,
        adapter_name="lightning",
    )
    pipe.set_adapters(["lightning"], adapter_weights=[1.0])
    # Fuse LoRA into base weights before CPU offload — prevents device mismatch
    # when offload moves components between CPU and CUDA during inference
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    pipe.enable_model_cpu_offload()

    print(f"Model loaded: {model_id} + Lightning LoRA (4-step)", flush=True)


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def handler(job):
    try:
        job_input = job.get("input", {})

        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        raw = job_input.get("images") or ([job_input["image"]] if "image" in job_input else None)
        if not raw:
            return {"error": "image or images is required"}

        pil_images = [_b64_to_pil(img) for img in raw]
        image_arg = pil_images[0] if len(pil_images) == 1 else pil_images

        # Default 4 steps for Lightning LoRA; caller can override up to 8 for quality
        steps = int(job_input.get("steps", 4))
        cfg_scale = float(job_input.get("cfg_scale", 4.0))
        guidance_scale = float(job_input.get("guidance_scale", 1.0))
        negative_prompt = job_input.get("negative_prompt", " ")
        seed = job_input.get("seed", -1)
        width = job_input.get("width")
        height = job_input.get("height")

        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator("cpu").manual_seed(seed)

        kwargs = dict(
            image=image_arg,
            prompt=prompt,
            true_cfg_scale=cfg_scale,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=generator,
        )
        if width:
            kwargs["width"] = int(width)
        if height:
            kwargs["height"] = int(height)

        with torch.inference_mode():
            result = pipe(**kwargs)

        buf = io.BytesIO()
        result.images[0].save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image": img_b64,
            "format": "png",
            "seed": seed,
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"OOM: {str(e)} — try smaller image or fewer steps"}
    except Exception as e:
        return {"error": str(e)}


load_model()
runpod.serverless.start({"handler": handler})
