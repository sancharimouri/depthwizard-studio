#!/usr/bin/env python3
"""VHR domain-transfer sanity check for Method 6.

Runs the UNCHANGED, DFC2019-trained Method 6 model (full DAv2-Small
fine-tune, twin mean/log-variance head; checkpoint from
train_method6_full_dfc2019.py, no retraining here) on real Maxar Open Data
VHR satellite crops (Sikkim, India-Floods-Oct-2023 event, ~0.305m GSD --
essentially the same resolution as DFC2019 training) to isolate the
sensor-domain question (aerial-vs-satellite) from the resolution question
(Sentinel-2's 10m, already tested and failed separately).

Not a training run, not an accuracy claim -- there is no LiDAR/DEM ground
truth for this crop wired up here. The question is narrower: does the
model produce plausible, non-degenerate output (a coherent height field
correlated with visible structure) or degenerate output (constant,
saturated, or structurally nonsensical) on real satellite imagery it has
never seen a single pixel of.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path("/Users/anweshasaha/projects/DepthWizard2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_method6_finetune_twinhead import (  # noqa: E402
    TwinHeadDav2, pad_to, get_device, IMAGENET_MEAN, IMAGENET_STD,
)

CKPT = PROJECT_ROOT / "data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019.pt"
OUT_DIR = PROJECT_ROOT / "data/maxar_sanity/method6_inference"

CROPS = {
    "crop_a_offnadir13": ("data/maxar_sanity/45_120220122200_2022-03-07_10300100CE8D0400.tif", 8704, 8704),
    "crop_b_nearnadir2": ("data/maxar_sanity/45_120220031201_2022-03-07_10300100CF621C00.tif", 15000, 9000),
    "crop_c_offnadir26_finestgsd": ("data/maxar_sanity/45_120220211230_2022-03-14_1040010073381800.tif", 8704, 8704),
}
CROP_SIZE = 1512  # ~460m at 0.305m GSD; padded up to a multiple of 14 below


def load_crop(path: Path, cy: int, cx: int, size: int) -> np.ndarray:
    with rasterio.open(path) as src:
        win = rasterio.windows.Window(cx - size // 2, cy - size // 2, size, size)
        arr = src.read([1, 2, 3], window=win).astype(np.float32)
    return arr


def main():
    device = get_device()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = TwinHeadDav2(height_scale=ckpt["height_scale"], **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint, height_scale={ckpt['height_scale']:.2f}m")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pad_size = ((CROP_SIZE + 13) // 14) * 14

    for name, (rel_path, cy, cx) in CROPS.items():
        rgb = load_crop(PROJECT_ROOT / rel_path, cy, cx, CROP_SIZE)
        rgb_t = torch.from_numpy(rgb / 255.0)
        rgb_t = (rgb_t - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        rgb_t = pad_to(rgb_t, pad_size)[None].to(device)

        with torch.no_grad():
            mu, log_var = model(rgb_t)
        mu = mu[0, 0, :CROP_SIZE, :CROP_SIZE].cpu().numpy()
        sigma = torch.exp(0.5 * log_var)[0, 0, :CROP_SIZE, :CROP_SIZE].cpu().numpy()

        rgb_gray = rgb.mean(axis=0)
        corr = float(np.corrcoef(mu.ravel(), rgb_gray.ravel())[0, 1])

        # Local structure check: does mu vary spatially at all, or collapse to a constant?
        grad_mag = np.sqrt(F.conv2d(torch.from_numpy(mu)[None, None],
                                    torch.tensor([[[[-1., 1.]]]]), padding=(0, 1))[0, 0, :, :-1] ** 2 +
                           F.conv2d(torch.from_numpy(mu)[None, None],
                                    torch.tensor([[[[-1.], [1.]]]]), padding=(1, 0))[0, 0, :-1, :] ** 2).mean().item()

        np.save(OUT_DIR / f"{name}_mu.npy", mu)
        np.save(OUT_DIR / f"{name}_sigma.npy", sigma)
        np.save(OUT_DIR / f"{name}_rgb.npy", rgb.astype(np.uint8))

        print(f"\n{name}:")
        print(f"  mu:    min={mu.min():.2f}m  max={mu.max():.2f}m  mean={mu.mean():.2f}m  std={mu.std():.2f}m")
        print(f"  sigma: min={sigma.min():.2f}m  max={sigma.max():.2f}m  mean={sigma.mean():.2f}m")
        print(f"  spatial gradient magnitude (mean |d(mu)|/pixel): {grad_mag:.4f}m")
        print(f"  corr(mu, grayscale RGB): {corr:+.3f}")
        pct = np.percentile(mu, [1, 5, 25, 50, 75, 95, 99])
        print(f"  percentiles [1,5,25,50,75,95,99]: {[round(p,1) for p in pct]}")


if __name__ == "__main__":
    main()
