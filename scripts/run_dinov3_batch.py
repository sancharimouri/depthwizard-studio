#!/usr/bin/env python3
"""Step 1.2: run Meta's official frozen DINOv3 satellite (SAT493M) backbone +
official CHMv2 DPT depth-probing head on the Sentinel-2 benchmark manifest.

Uses facebookresearch/dinov3's own `dinov3_vitl16_chmv2` hub entrypoint
(backbone_weights=SAT493M, depther_weights=CHMV2) -- Meta's provided
monocular height-estimation probe for the satellite-pretrained backbone --
unmodified, mirroring how scripts/run_dav2_batch.py reuses the existing
DAv2 pipeline. No training, no custom readout.

Backbone weights are Meta-license-gated (manual HF approval / signed CDN
URL). Pass a local .pth path or signed URL via --backbone-weights once you
have one; otherwise this will fail with an HTTP 403 from fbaipublicfiles.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINOV3_REPO = PROJECT_ROOT / "external" / "dinov3"
TORCH_HUB_CACHE = PROJECT_ROOT / "models" / "hub"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DAv2's own pipeline (backend/depth/depth_engine.py) resizes to a fixed
# 518x518 for the model forward pass, then bicubic-upsamples the output
# back to source resolution -- full source-resolution ViT-L attention on
# CPU is computationally intractable (>25min/tile, unbounded for 32 tiles).
# Matching that same input-resolution handling here keeps both backbones'
# procedures parallel rather than introducing an asymmetry.
DINOV3_INPUT_SIZE = 518


def parse_args():
    p = argparse.ArgumentParser(description="Run frozen DINOv3 SAT493M+CHMv2 on the benchmark manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--hf-safetensors",
        default=str(PROJECT_ROOT / "models" / "hub" / "hf_dinov3_sat493m" / "model.safetensors"),
        help="Path to the SAT493M backbone downloaded from the gated "
        "facebook/dinov3-vitl16-pretrain-sat493m HF repo (requires approved "
        "access). Converted in-memory to the original repo's state-dict "
        "format; see scripts/lib_dinov3_sat493m_loader.py.",
    )
    return p.parse_args()


def load_model(hf_safetensors: str):
    torch.hub.set_dir(str(TORCH_HUB_CACHE))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from lib_dinov3_sat493m_loader import load_dinov3_chmv2_depther

    return load_dinov3_chmv2_depther(Path(hf_safetensors))


def load_rgb(path: Path) -> Image.Image:
    import rasterio

    with rasterio.open(path) as src:
        data = src.read([1, 2, 3])
    rgb = np.moveaxis(data, 0, -1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((DINOV3_INPUT_SIZE, DINOV3_INPUT_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


@torch.no_grad()
def run_inference(model, image: Image.Image) -> np.ndarray:
    src_w, src_h = image.size
    x = preprocess(image)

    out = model(x)
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    if out.shape[-2:] != (src_h, src_w):
        out = F.interpolate(out.unsqueeze(1), size=(src_h, src_w), mode="bicubic", align_corners=False).squeeze(1)

    return out.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DINOv3 SAT493M + CHMv2 depther (frozen, no training)...")
    model = load_model(args.hf_safetensors)
    print("Loaded.\n")

    successes, skipped, errors = 0, 0, 0
    runtimes = []

    for i, row in manifest.iterrows():
        tile_id = row["tile_id"]
        rgb_path = PROJECT_ROOT / row["rgb_path"]
        out_path = output_dir / f"{tile_id}_depth.npy"

        print(f"[{i + 1}/{len(manifest)}] {tile_id}")
        if out_path.exists():
            print("  Already exists — skipping.")
            skipped += 1
            continue

        try:
            start = time.perf_counter()
            image = load_rgb(rgb_path)
            depth = run_inference(model, image)
            elapsed = time.perf_counter() - start
            np.save(out_path, depth)
            successes += 1
            runtimes.append(elapsed)
            print(f"  Done: {elapsed:.2f}s  range=[{depth.min():.3f}, {depth.max():.3f}]")
        except Exception as exc:
            errors += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print(f"New runs: {successes}  Skipped: {skipped}  Errors: {errors}")
    if runtimes:
        print(f"Mean runtime: {np.mean(runtimes):.2f}s  Median: {np.median(runtimes):.2f}s")


if __name__ == "__main__":
    main()
