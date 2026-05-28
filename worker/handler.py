import math
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

    # BF16 full model needs ~80GB — runs full-GPU on A100 SXM 80GB
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
    is_fp8 = "fp8" in model_id.lower() or "FP8" in model_id

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device:    {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA capability:{torch.cuda.get_device_capability(0)}", flush=True)
        print(f"VRAM total:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    print(f"PyTorch:        {torch.__version__}", flush=True)
    print(f"Model variant:  {'FP8 (RTX 3090/4090)' if is_fp8 else 'BF16 (A100/H100)'}", flush=True)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    # FP8 model is ~12GB — full GPU on anything >=16GB (RTX 3090/4090)
    # BF16 model is ~80GB — only full GPU on A100 80GB+
    full_gpu_threshold = 16 if is_fp8 else 60
    use_full_gpu = vram_gb >= full_gpu_threshold

    # Load Lightning LoRA — reduces inference from 40 steps to 4 (~10x speedup)
    pipe.load_lora_weights(
        LIGHTNING_LORA,
        weight_name=LIGHTNING_WEIGHT,
        adapter_name="lightning",
    )
    pipe.set_adapters(["lightning"], adapter_weights=[1.0])

    if not is_fp8:
        # Fuse LoRA before CPU offload to prevent device mismatch (BF16 model only).
        # Quantized (FP8) models cannot fuse float LoRA deltas into integer weights.
        pipe.fuse_lora()
        pipe.unload_lora_weights()

    if use_full_gpu:
        pipe.to("cuda")
        print(f"Offload: none (full GPU, {vram_gb:.0f}GB)", flush=True)
        # torch.compile speeds up repeated inference — ~20% faster after warmup
        pipe.transformer = torch.compile(pipe.transformer, mode="default")
        print("torch.compile: enabled on transformer", flush=True)
    elif is_fp8:
        # FP8 model components (~12GB total) fit individually on any 16GB+ GPU
        pipe.enable_model_cpu_offload()
        print(f"Offload: model (component-level, {vram_gb:.0f}GB)", flush=True)
    else:
        # BF16 transformer alone is ~50GB — too large for model-level CPU offload
        # on any GPU < 80GB. Sequential offload moves layer-by-layer instead.
        pipe.enable_sequential_cpu_offload()
        print(f"Offload: sequential (layer-level, {vram_gb:.0f}GB)", flush=True)

    print(f"Model loaded: {model_id} + Lightning LoRA (4-step)", flush=True)


_QUALITY_PRESETS = {
    "fast":     {"steps": 4,  "cfg_scale": 3.5},
    "balanced": {"steps": 8,  "cfg_scale": 4.0},
    "high":     {"steps": 16, "cfg_scale": 4.5},
    "ultra":    {"steps": 20, "cfg_scale": 5.0},
}

_SIZE_PRESETS = {
    "small":  512,
    "medium": 768,
    "large":  1024,
    "xl":     1280,
}


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _resolve_dimensions(
    aspect_ratio: str | None,
    size: str | None,
    width: int | None,
    height: int | None,
) -> tuple[int | None, int | None]:
    # Explicit width/height always wins
    if width and height:
        return int(width), int(height)

    base_px = _SIZE_PRESETS.get(size or "large", 1024)

    if aspect_ratio:
        try:
            a_str, b_str = aspect_ratio.split(":")
            a, b = int(a_str), int(b_str)
        except ValueError:
            raise ValueError(f"aspect_ratio must be 'W:H' format, got: {aspect_ratio!r}")

        # Keep total pixels constant regardless of ratio, round to nearest 64
        total = base_px * base_px
        h = max(64, round(math.sqrt(total * b / a) / 64) * 64)
        w = max(64, round(math.sqrt(total * a / b) / 64) * 64)
        return w, h

    # size preset, square
    return base_px, base_px


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

        # Quality preset sets step/cfg defaults; explicit overrides win
        quality = job_input.get("quality", "balanced")
        if quality not in _QUALITY_PRESETS:
            return {"error": f"quality must be one of {list(_QUALITY_PRESETS)}"}
        preset = _QUALITY_PRESETS[quality]

        steps = int(job_input.get("steps", preset["steps"]))
        cfg_scale = float(job_input.get("cfg_scale", preset["cfg_scale"]))
        guidance_scale = float(job_input.get("guidance_scale", 1.0))
        negative_prompt = job_input.get("negative_prompt", " ")
        seed = job_input.get("seed", -1)
        output_format = job_input.get("output_format", "png").lower()
        if output_format not in ("png", "jpeg"):
            return {"error": "output_format must be 'png' or 'jpeg'"}

        width, height = _resolve_dimensions(
            aspect_ratio=job_input.get("aspect_ratio"),
            size=job_input.get("size"),
            width=job_input.get("width"),
            height=job_input.get("height"),
        )

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
            width=width,
            height=height,
        )

        with torch.inference_mode():
            result = pipe(**kwargs)

        buf = io.BytesIO()
        save_fmt = "JPEG" if output_format == "jpeg" else "PNG"
        save_kwargs = {"quality": 92} if save_fmt == "JPEG" else {}
        result.images[0].save(buf, format=save_fmt, **save_kwargs)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image": img_b64,
            "format": output_format,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "quality": quality,
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"error": f"OOM: {str(e)} — try smaller size or lower quality preset"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


load_model()
runpod.serverless.start({"handler": handler})
