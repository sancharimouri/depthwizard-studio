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


def parse_args():
    p = argparse.ArgumentParser(description="Run frozen DINOv3 SAT493M+CHMv2 on the benchmark manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--backbone-weights",
        default=None,
        help="Local .pth path or signed URL for the gated SAT493M backbone. "
        "Omit to use Meta's public (gated) fbaipublicfiles URL, which will "
        "403 without prior access.",
    )
    return p.parse_args()


def load_model(backbone_weights: str | None):
    torch.hub.set_dir(str(TORCH_HUB_CACHE))
    kwargs = {}
    if backbone_weights:
        kwargs["backbone_weights"] = backbone_weights
    model = torch.hub.load(str(DINOV3_REPO), "dinov3_vitl16_chmv2", source="local", pretrained=True, **kwargs)
    model.eval()
    return model


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
    model = load_model(args.backbone_weights)
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
