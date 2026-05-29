from __future__ import annotations

import base64
import fcntl
import io
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from threading import Lock

import runpod
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from huggingface_hub import snapshot_download
from PIL import Image

# ---------------------------------------------------------------------------
# Storage — model files live on a RunPod Network Volume so they survive across
# cold starts. Falls back to /workspace/model-storage when no volume is mounted.
# ---------------------------------------------------------------------------
STORAGE_ROOT = Path(
    os.getenv("RUNPOD_VOLUME_PATH", os.getenv("MODEL_STORAGE_PATH", "/workspace/model-storage"))
)
_HF_ROOT   = STORAGE_ROOT / "huggingface"
_MODEL_ROOT = STORAGE_ROOT / "models"
_LOCK_ROOT  = STORAGE_ROOT / "locks"
_TMP_ROOT   = STORAGE_ROOT / "tmp"


def _configure_storage() -> None:
    for path, env_key in [
        (_HF_ROOT,           "HF_HOME"),
        (_HF_ROOT / "hub",   "HF_HUB_CACHE"),
        (_HF_ROOT / "assets","HF_ASSETS_CACHE"),
        (_TMP_ROOT,          "TMPDIR"),
    ]:
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_key, str(path))
    _MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    _LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(_TMP_ROOT)


_configure_storage()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIGHTNING_LORA   = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_WEIGHT = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

pipe       = None
_pipe_lock = Lock()

# Lightning LoRA is CFG-distilled: true_cfg_scale MUST be 1.0 (not 4.0).
# Using true_cfg_scale > 1 with a distilled LoRA produces silhouettes/illustrations.
_QUALITY_PRESETS = {
    "fast":     {"steps": 4, "cfg_scale": 1.0},
    "balanced": {"steps": 4, "cfg_scale": 1.0},
    "high":     {"steps": 4, "cfg_scale": 1.0},
    "ultra":    {"steps": 4, "cfg_scale": 1.0},
}

# Default portrait 832x1216 — 1024x1024 square has documented quality degradation.
_SIZE_PRESETS = {
    "small":  512,
    "medium": 704,
    "large":  896,
    "xl":     1152,
}
_DEFAULT_WIDTH  = 832
_DEFAULT_HEIGHT = 1216


# ---------------------------------------------------------------------------
# Model download with volume caching + file lock (safe for concurrent workers)
# ---------------------------------------------------------------------------

def _model_cache_dir(model_id: str) -> Path:
    return _MODEL_ROOT / model_id.replace("/", "--")


def _download_model(model_id: str) -> Path:
    local_dir = _model_cache_dir(model_id)
    lock_path  = _LOCK_ROOT / f"{local_dir.name}.lock"

    with open(lock_path, "w") as lock_file:
        print(f"Acquiring model lock: {lock_path}", flush=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            ready_marker = local_dir / ".snapshot-complete"
            model_index  = local_dir / "model_index.json"

            if ready_marker.exists() and model_index.exists():
                free_gb = shutil.disk_usage(STORAGE_ROOT).free / 1e9
                print(f"Volume cache hit: {local_dir}  (disk free: {free_gb:.0f}GB)", flush=True)
                return local_dir

            free_gb = shutil.disk_usage(STORAGE_ROOT).free / 1e9
            print(f"Downloading {model_id} → {local_dir}  (disk free: {free_gb:.0f}GB)", flush=True)

            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                max_workers=int(os.getenv("HF_DOWNLOAD_MAX_WORKERS", "4")),
                token=os.environ.get("HF_TOKEN") or None,
            )

            if not model_index.exists():
                raise RuntimeError(
                    f"Download finished but model_index.json missing in {local_dir}"
                )

            ready_marker.touch()
            free_gb = shutil.disk_usage(STORAGE_ROOT).free / 1e9
            print(f"Model cached at {local_dir}  (disk free: {free_gb:.0f}GB)", flush=True)
            return local_dir

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> None:
    global pipe

    # Fast path — no lock needed once loaded
    if pipe is not None:
        return

    with _pipe_lock:
        if pipe is not None:
            return

        model_id = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
        is_fp8   = "fp8" in model_id.lower() or "FP8" in model_id

        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"CUDA device:    {torch.cuda.get_device_name(0)}", flush=True)
            print(f"CUDA capability:{torch.cuda.get_device_capability(0)}", flush=True)
            print(f"VRAM total:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
        print(f"PyTorch:        {torch.__version__}", flush=True)
        print(f"Storage root:   {STORAGE_ROOT}", flush=True)
        print(f"Model variant:  {'FP8' if is_fp8 else 'BF16'}", flush=True)

        model_dir = _download_model(model_id)

        print(f"Loading pipeline from {model_dir} ...", flush=True)
        p = QwenImageEditPlusPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        # Lightning LoRA requires exponential time-shifting scheduler.
        # The base model's default linear scheduler produces degraded quality at 4 steps.
        p.scheduler = FlowMatchEulerDiscreteScheduler(
            base_image_seq_len=256,
            base_shift=math.log(3),
            max_image_seq_len=8192,
            num_train_timesteps=1000,
            shift=1.0,
            time_shift_type="exponential",
            use_dynamic_shifting=True,
        )
        print("Scheduler: FlowMatchEulerDiscrete (exponential, Lightning)", flush=True)

        vram_gb           = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        full_gpu_threshold = 16 if is_fp8 else 60
        use_full_gpu       = vram_gb >= full_gpu_threshold

        # Load Lightning LoRA — reduces inference from ~40 steps to 4 (~10x speedup).
        # HF hub caches the LoRA weights in HF_HOME on the volume after first download.
        p.load_lora_weights(LIGHTNING_LORA, weight_name=LIGHTNING_WEIGHT, adapter_name="lightning")
        p.set_adapters(["lightning"], adapter_weights=[1.0])

        if not is_fp8:
            # Fuse LoRA before CPU offload to prevent device mismatch (BF16 only).
            # FP8/quantized models cannot fuse float LoRA deltas into integer weights.
            p.fuse_lora()
            p.unload_lora_weights()

        # VAE tiling reduces peak VRAM during the decode step on large images.
        if getattr(p, "vae", None) is not None:
            p.vae.enable_tiling()
            print("VAE tiling: enabled", flush=True)

        if use_full_gpu:
            p.to("cuda")
            print(f"Offload: none (full GPU, {vram_gb:.0f}GB)", flush=True)
            # torch.compile speeds up repeated inference ~20% after warmup
            p.transformer = torch.compile(p.transformer, mode="default")
            print("torch.compile: enabled on transformer", flush=True)
        elif is_fp8:
            p.enable_model_cpu_offload()
            print(f"Offload: model (component-level, {vram_gb:.0f}GB)", flush=True)
        else:
            # BF16 transformer (~50GB) too large for component-level offload on <80GB GPU.
            # Sequential offload moves layer-by-layer instead.
            p.enable_sequential_cpu_offload()
            print(f"Offload: sequential (layer-level, {vram_gb:.0f}GB)", flush=True)

        print(f"Model ready: {model_id} + Lightning LoRA (4-step)", flush=True)
        pipe = p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _resolve_dimensions(
    aspect_ratio: str | None,
    size: str | None,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    if width and height:
        return int(width), int(height)

    base_px = _SIZE_PRESETS.get(size or "large", 896)

    if not aspect_ratio and not size:
        return _DEFAULT_WIDTH, _DEFAULT_HEIGHT

    if aspect_ratio:
        try:
            a_str, b_str = aspect_ratio.split(":")
            a, b = int(a_str), int(b_str)
        except ValueError:
            raise ValueError(f"aspect_ratio must be 'W:H' format, got: {aspect_ratio!r}")

        total = base_px * base_px
        h = max(64, round(math.sqrt(total * b / a) / 64) * 64)
        w = max(64, round(math.sqrt(total * a / b) / 64) * 64)
        return w, h

    return base_px, base_px


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    t_start = time.time()
    job_id  = job.get("id", "?")

    try:
        job_input = job.get("input", {})

        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        raw = job_input.get("images") or ([job_input["image"]] if "image" in job_input else None)
        if not raw:
            return {"error": "image or images is required"}

        pil_images = [_b64_to_pil(img) for img in raw]
        image_arg  = pil_images[0] if len(pil_images) == 1 else pil_images
        img_count  = len(pil_images)
        src_size   = f"{pil_images[0].width}x{pil_images[0].height}"

        quality = job_input.get("quality", "balanced")
        if quality not in _QUALITY_PRESETS:
            return {"error": f"quality must be one of {list(_QUALITY_PRESETS)}"}
        preset = _QUALITY_PRESETS[quality]

        steps           = int(job_input.get("steps", preset["steps"]))
        cfg_scale       = float(job_input.get("cfg_scale", preset["cfg_scale"]))
        negative_prompt = job_input.get("negative_prompt", " ")  # single space, not empty
        seed            = job_input.get("seed", -1)
        output_format   = job_input.get("output_format", "png").lower()
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

        vram_free = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
        print(
            f"[{job_id}] START  prompt={prompt[:60]!r}  images={img_count}({src_size})"
            f"  steps={steps}  cfg={cfg_scale}  out={width}x{height}  seed={seed}"
            f"  vram_free={vram_free:.1f}GB",
            flush=True,
        )

        t_infer = time.time()
        with torch.inference_mode():
            result = pipe(
                image=image_arg,
                prompt=prompt,
                true_cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                generator=generator,
                width=width,
                height=height,
            )
        infer_s = time.time() - t_infer

        buf = io.BytesIO()
        save_fmt    = "JPEG" if output_format == "jpeg" else "PNG"
        save_kwargs = {"quality": 92} if save_fmt == "JPEG" else {}
        result.images[0].save(buf, format=save_fmt, **save_kwargs)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_kb  = len(buf.getvalue()) / 1024

        total_s = time.time() - t_start
        print(
            f"[{job_id}] DONE   infer={infer_s:.1f}s  total={total_s:.1f}s"
            f"  output_size={img_kb:.0f}KB",
            flush=True,
        )

        return {
            "image":          img_b64,
            "format":         output_format,
            "seed":           seed,
            "width":          width,
            "height":         height,
            "steps":          steps,
            "quality":        quality,
            "infer_seconds":  round(infer_s, 1),
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[{job_id}] OOM  vram_total={vram_total:.0f}GB  error={e}", flush=True)
        return {"error": f"OOM: {str(e)} — try smaller size or lower quality preset"}
    except ValueError as e:
        print(f"[{job_id}] ValueError: {e}", flush=True)
        return {"error": str(e)}
    except Exception as e:
        print(f"[{job_id}] ERROR: {e}", flush=True)
        return {"error": str(e)}


load_model()
runpod.serverless.start({"handler": handler})
