"""Depth Anything V2 inference: RGB image/GeoTIFF -> relative depth."""

import os
import time
from pathlib import Path

import numpy as np

from backend import config

# HF_HOME must be set before importing transformers/huggingface_hub.
os.environ["HF_HOME"] = str(config.HF_HOME_DIR)

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForDepthEstimation, DPTImageProcessorPil


def load_engine():
    """Load the DAv2 processor, model, and selected device."""
    device = config.select_device()

    processor = DPTImageProcessorPil.from_pretrained(config.ACTIVE_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(
        config.ACTIVE_MODEL_ID,
        use_safetensors=True,
    )

    model.to(device)
    model.eval()

    return processor, model, device


def check_pixel_budget(width: int, height: int) -> None:
    """Reject excessively large inputs before decoding/inference."""
    if width * height > config.MAX_INPUT_PIXELS:
        raise ValueError(
            f"input {width}x{height} exceeds MAX_INPUT_PIXELS"
        )


def normalize_depth(arr: np.ndarray) -> np.ndarray:
    """Normalize depth to float32 [0, 1]."""
    arr = arr.astype(np.float32)

    if not np.isfinite(arr).all():
        return np.zeros_like(arr, dtype=np.float32)

    lo = float(arr.min())
    hi = float(arr.max())

    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)

    return np.clip(
        (arr - lo) / (hi - lo),
        0.0,
        1.0,
    ).astype(np.float32)


def infer_with(image: Image.Image, processor, model, device) -> np.ndarray:
    """Run DAv2 inference and restore output to source dimensions."""
    src_w, src_h = image.size
    check_pixel_budget(src_w, src_h)

    rgb = image.convert("RGB")

    inputs = processor(
        images=rgb,
        return_tensors="pt",
        do_resize=True,
        keep_aspect_ratio=False,
        size={
            "height": config.MODEL_INPUT_SIZE,
            "width": config.MODEL_INPUT_SIZE,
        },
    )

    actual_shape = inputs["pixel_values"].shape[-2:]

    if actual_shape != (
        config.MODEL_INPUT_SIZE,
        config.MODEL_INPUT_SIZE,
    ):
        raise ValueError(
            f"expected model input "
            f"{config.MODEL_INPUT_SIZE}x{config.MODEL_INPUT_SIZE}, "
            f"got {tuple(actual_shape)}"
        )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    predicted_depth = outputs.predicted_depth

    depth_4d = predicted_depth.reshape(
        1,
        1,
        *predicted_depth.shape[-2:],
    )

    resized = F.interpolate(
        depth_4d,
        size=(src_h, src_w),
        mode="bicubic",
        align_corners=False,
    )

    depth = (
        resized
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return normalize_depth(depth)


# Load exactly once when this module is imported.
_PROCESSOR, _MODEL, _DEVICE = load_engine()

DEVICE = _DEVICE


def run_inference(image: Image.Image) -> np.ndarray:
    """Public DAv2 inference API."""
    return infer_with(
        image,
        _PROCESSOR,
        _MODEL,
        _DEVICE,
    )


def load_geotiff_rgb(path: Path) -> Image.Image:
    """Read a 3-band GeoTIFF and return it as an RGB PIL image."""
    import rasterio

    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(
                f"Expected at least 3 bands in {path}, "
                f"found {src.count}"
            )

        data = src.read([1, 2, 3])

    # GeoTIFF is bands-first: (3, H, W)
    # PIL expects: (H, W, 3)
    rgb = np.moveaxis(data, 0, -1)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB")


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    """Save normalized relative depth as grayscale PNG."""
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth, got {depth.shape}")

    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    image = Image.fromarray(
        np.clip(depth * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_geotiff(path: Path, output_dir: Path) -> dict:
    """Run DAv2 on a GeoTIFF and save diagnostic outputs."""
    start = time.perf_counter()

    image = load_geotiff_rgb(path)
    depth = run_inference(image)

    elapsed = time.perf_counter() - start

    output_dir.mkdir(parents=True, exist_ok=True)

    depth_path = output_dir / f"{path.stem}_depth.png"
    npy_path = output_dir / f"{path.stem}_depth.npy"

    save_depth_png(depth, depth_path)
    np.save(npy_path, depth)

    return {
        "source": str(path),
        "depth_png": str(depth_path),
        "depth_npy": str(npy_path),
        "source_size": [image.width, image.height],
        "depth_shape": list(depth.shape),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "device": str(DEVICE),
        "seconds": elapsed,
    }
