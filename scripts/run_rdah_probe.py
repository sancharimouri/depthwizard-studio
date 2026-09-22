#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDAH_DIR = PROJECT_ROOT / "external" / "RDAH-Net"
CHECKPOINT = RDAH_DIR / "104best_model.pth"


def load_rdah_module():
    sys.path.insert(0, str(RDAH_DIR))
    spec = importlib.util.spec_from_file_location(
        "rdah_test", RDAH_DIR / "test.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load external/RDAH-Net/test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()

    if arr.shape[0] < 3:
        raise ValueError(f"RGB TIFF must have >=3 bands, got {arr.shape}")

    arr = arr[:3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo = np.nanpercentile(arr, 1)
        hi = np.nanpercentile(arr, 99)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    arr = np.transpose(arr, (1, 2, 0))
    return arr


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    # Match RDAH inference preprocessing:
    # ToTensor -> [0,1] -> ImageNet normalization.
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype)[:, None, None]
    x = (x - mean) / std
    return x.unsqueeze(0)


def preprocess_depth(depth: np.ndarray) -> torch.Tensor:
    # RDAH's ToTensor leaves the single-channel depth/relative-depth values
    # in their source numeric scale; it does NOT ImageNet-normalize depth.
    x = torch.from_numpy(depth.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default="JAX_004_006")
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=PROJECT_ROOT / "data/dfc2019/raw/RGB/Track1-RGB",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=PROJECT_ROOT / "data/dfc2019/experiments/dav2_baseline/depth",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rgb_path = args.rgb_dir / f"{args.tile}_RGB.tif"
    depth_path = args.depth_dir / f"{args.tile}_depth.npy"

    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)
    if not depth_path.exists():
        raise FileNotFoundError(depth_path)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    print(f"Tile:       {args.tile}")
    print(f"RGB:        {rgb_path}")
    print(f"DAv2:       {depth_path}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device:     {args.device}")

    rgb = load_rgb(rgb_path)
    depth = np.load(depth_path).astype(np.float32)

    print("\nInput statistics")
    print(f"  RGB shape:   {rgb.shape}")
    print(f"  RGB dtype:   {rgb.dtype}")
    print(f"  DAv2 shape:  {depth.shape}")
    print(f"  DAv2 dtype:  {depth.dtype}")
    print(f"  DAv2 min:    {np.nanmin(depth):.6f}")
    print(f"  DAv2 max:    {np.nanmax(depth):.6f}")
    print(f"  DAv2 mean:   {np.nanmean(depth):.6f}")
    print(f"  DAv2 std:    {np.nanstd(depth):.6f}")

    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"Shape mismatch: RGB={rgb.shape[:2]}, DAv2={depth.shape}."
        )

    module = load_rdah_module()
    model = module.HeightPredTransformer()

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("\nCheckpoint load")
    print(f"  epoch:       {checkpoint.get('epoch')}")
    print(f"  loss:        {checkpoint.get('loss')}")
    print(f"  missing:     {len(missing)}")
    print(f"  unexpected:  {len(unexpected)}")
    if missing:
        print("  first missing:", missing[:10])
    if unexpected:
        print("  first unexpected:", unexpected[:10])

    device = torch.device(args.device)
    model = model.to(device).eval()

    rgb_tensor = preprocess_rgb(rgb).to(device)
    depth_tensor = preprocess_depth(depth).to(device)

    print("\nModel inputs")
    print(f"  depth: {tuple(depth_tensor.shape)}")
    print(f"  rgb:   {tuple(rgb_tensor.shape)}")

    with torch.no_grad():
        prediction = model(depth_tensor, rgb_tensor)

    prediction = prediction.detach().cpu().float().squeeze().numpy()

    print("\nPrediction")
    print(f"  shape: {prediction.shape}")
    print(f"  min:   {np.nanmin(prediction):.6f}")
    print(f"  max:   {np.nanmax(prediction):.6f}")
    print(f"  mean:  {np.nanmean(prediction):.6f}")
    print(f"  std:   {np.nanstd(prediction):.6f}")
    print(f"  nan:   {np.isnan(prediction).sum()}")


if __name__ == "__main__":
    main()

