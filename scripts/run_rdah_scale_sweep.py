
#!/usr/bin/env python3
"""
Controlled RDAH input-domain sweep.

This script tests the pretrained RDAH checkpoint with the same RGB input and
different transformations of the cached DAv2 relative-depth map.

It does NOT modify any .npy files.

Default tile:
    JAX_149_006

Tests:
    baseline
    x0.001, x0.01, x0.1, x1, x10, x100, x1000
    invert (1 - depth)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDAH_DIR = PROJECT_ROOT / "external" / "RDAH-Net"
CHECKPOINT = RDAH_DIR / "104best_model.pth"
RGB_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "RGB" / "Track1-RGB"
DEPTH_DIR = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "dav2_baseline"
    / "depth"
)


def load_rdah_model() -> torch.nn.Module:
    """
    Import the official RDAH model definition and load the checkpoint.

    RDAH test.py imports `loaddata` as a top-level module, so the repository
    directory must be added to sys.path before executing test.py.
    """
    if str(RDAH_DIR) not in sys.path:
        sys.path.insert(0, str(RDAH_DIR))

    test_py = RDAH_DIR / "test.py"

    spec = importlib.util.spec_from_file_location("rdah_test", test_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {test_py}")

    module = importlib.util.module_from_spec(spec)

    # Make the RDAH directory visible to imports such as:
    #     import loaddata
    spec.loader.exec_module(module)

    model = module.HeightPredTransformer()

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state = checkpoint["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

    model.eval()
    return model


def load_rgb(path: Path) -> torch.Tensor:
    """Load RGB and reproduce the RDAH RGB test preprocessing."""
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")

    # Match RDAH test preprocessing:
    # RGB -> [0,1] -> ImageNet normalization.
    rgb_tensor = transforms.ToTensor()(image)
    rgb_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(rgb_tensor)

    return rgb_tensor.unsqueeze(0)


def load_depth(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)

    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D DAv2 map, got {depth.shape}")

    if not np.isfinite(depth).all():
        raise ValueError("DAv2 map contains NaN/Inf")

    return depth


def run_prediction(
    model: torch.nn.Module,
    depth_np: np.ndarray,
    rgb_tensor: torch.Tensor,
) -> dict[str, float | int]:
    depth_tensor = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        prediction = model(depth_tensor, rgb_tensor)

    pred = prediction.squeeze().cpu().numpy()

    return {
        "in_min": float(depth_np.min()),
        "in_max": float(depth_np.max()),
        "in_mean": float(depth_np.mean()),
        "in_std": float(depth_np.std()),
        "out_min": float(pred.min()),
        "out_max": float(pred.max()),
        "out_mean": float(pred.mean()),
        "out_std": float(pred.std()),
        "nan": int(np.isnan(pred).sum()),
    }


def print_row(name: str, stats: dict[str, float | int]) -> None:
    print(
        f"{name:>10} | "
        f"in [{stats['in_min']:9.4f}, {stats['in_max']:9.4f}] "
        f"mean={stats['in_mean']:9.4f} std={stats['in_std']:9.4f} | "
        f"out [{stats['out_min']:9.4f}, {stats['out_max']:9.4f}] "
        f"mean={stats['out_mean']:9.4f} std={stats['out_std']:9.4f} "
        f"nan={stats['nan']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default="JAX_149_006")
    args = parser.parse_args()

    tile = args.tile
    rgb_path = RGB_DIR / f"{tile}_RGB.tif"
    depth_path = DEPTH_DIR / f"{tile}_depth.npy"

    for path in (rgb_path, depth_path, CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Tile:       {tile}")
    print(f"RGB:        {rgb_path}")
    print(f"DAv2:       {depth_path}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Device:     cpu")
    print()

    depth = load_depth(depth_path)
    rgb = load_rgb(rgb_path)

    print("Original DAv2")
    print(f"  shape: {depth.shape}")
    print(f"  min:   {depth.min():.6f}")
    print(f"  max:   {depth.max():.6f}")
    print(f"  mean:  {depth.mean():.6f}")
    print(f"  std:   {depth.std():.6f}")
    print()

    model = load_rdah_model()

    print("Checkpoint loaded: OK")
    print()
    print(
        "      test | "
        "input distribution                          | "
        "output distribution"
    )
    print("-" * 124)

    tests: list[tuple[str, np.ndarray]] = [
        ("baseline", depth),
        ("x0.001", depth * 0.001),
        ("x0.01", depth * 0.01),
        ("x0.1", depth * 0.1),
        ("x1", depth),
        ("x10", depth * 10.0),
        ("x100", depth * 100.0),
        ("x1000", depth * 1000.0),
        ("invert", 1.0 - depth),
    ]

    for name, candidate in tests:
        stats = run_prediction(
            model,
            candidate.astype(np.float32, copy=False),
            rgb,
        )
        print_row(name, stats)


if __name__ == "__main__":
    main()
