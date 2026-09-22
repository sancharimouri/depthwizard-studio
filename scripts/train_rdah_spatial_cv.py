
#!/usr/bin/env python3
"""
RDAH-Net fine-tuning on DepthWizard2 DFC2019 data.

Experiment: RDAH-FT-1
- Official RDAH architecture from external/RDAH-Net/test.py
- Pretrained checkpoint: external/RDAH-Net/104best_model.pth
- Input: DAv2 [0,1] .npy + RGB GeoTIFF
- Target: DFC2019 AGL GeoTIFF
- Loss: official RDAH MaskedLoss (SmoothL1)
- No augmentation
- Batch size: 1
- Default LR: 1e-5
- Spatial CV: 4 tile-level geographic quadrants

Why tile-level spatial quadrants instead of masking quadrants inside a tile:
RDAH is designed around a 1024x1024 input and its decoder returns a 2x spatial
upsampling relative to the encoder input. We therefore keep every training
sample at the native 1024x1024 resolution and hold out geographically separate
groups of tiles, avoiding spatial leakage from the test region.

Use:
    python scripts/train_rdah_spatial_cv.py --fold 0 --epochs 1
for a smoke test.

Then full:
    python scripts/train_rdah_spatial_cv.py --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDAH_DIR = PROJECT_ROOT / "external" / "RDAH-Net"
CHECKPOINT = RDAH_DIR / "104best_model.pth"

RGB_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "RGB" / "Track1-RGB"
GT_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "Truth" / "Track1-Truth"
DEPTH_DIR = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "dav2_baseline" / "depth"

MANIFEST = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "dav2_baseline" / "manifest.csv"
OUT_DIR = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "rdah"


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def import_rdah_model_class():
    if str(RDAH_DIR) not in sys.path:
        sys.path.insert(0, str(RDAH_DIR))

    test_py = RDAH_DIR / "test.py"
    spec = importlib.util.spec_from_file_location("rdah_test", test_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {test_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HeightPredTransformer, module.MaskedLoss


def load_pretrained(device: torch.device):
    HeightPredTransformer, _ = import_rdah_model_class()

    model = HeightPredTransformer()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    # Exact pretrained RDAH checkpoint loading.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint


def load_manifest_tiles() -> list[str]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Expected the existing 50-tile manifest at {MANIFEST}. "
            "Do not silently substitute another tile set."
        )

    tiles: list[str] = []

    with MANIFEST.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Manifest has no header: {MANIFEST}")

        # Find a likely tile/name column.
        candidates = ["tile", "tile_id", "name", "id", "rgb"]
        key = next((c for c in candidates if c in reader.fieldnames), None)

        if key is None:
            raise RuntimeError(
                f"Could not identify tile column in {MANIFEST}. "
                f"Columns: {reader.fieldnames}"
            )

        for row in reader:
            value = str(row[key]).strip()
            if not value:
                continue
            # Accept either bare tile names or RGB filenames.
            value = Path(value).stem
            value = re.sub(r"_RGB$", "", value)
            tiles.append(value)

    tiles = sorted(set(tiles))

    if not tiles:
        raise RuntimeError(f"No tiles found in {MANIFEST}")

    return tiles


def verify_tile_files(tiles: list[str]) -> list[str]:
    usable: list[str] = []

    for tile in tiles:
        rgb = RGB_DIR / f"{tile}_RGB.tif"
        depth = DEPTH_DIR / f"{tile}_depth.npy"
        gt = GT_DIR / f"{tile}_AGL.tif"

        missing = [str(p) for p in (rgb, depth, gt) if not p.exists()]
        if missing:
            print(f"SKIP {tile}: missing {', '.join(missing)}")
            continue

        usable.append(tile)

    if not usable:
        raise RuntimeError("No manifest tiles have all RGB, DAv2 and AGL files.")

    return usable


def tile_coords(tile: str) -> tuple[int, int]:
    """
    DFC tile IDs look like JAX_149_006 or OMA_248_029.
    The two numeric components are used as geographic row/column proxies.
    """
    m = re.search(r"_(\d+)_(\d+)$", tile)
    if not m:
        raise ValueError(f"Cannot parse spatial coordinates from tile id: {tile}")
    return int(m.group(1)), int(m.group(2))


def make_spatial_folds(tiles: list[str], k: int = 4) -> dict[int, list[str]]:
    if k != 4:
        raise ValueError("This experiment is defined for exactly 4 spatial folds.")

    coords = np.array([tile_coords(t) for t in tiles], dtype=np.float64)

    row_mid = float(np.median(coords[:, 0]))
    col_mid = float(np.median(coords[:, 1]))

    folds = {0: [], 1: [], 2: [], 3: []}

    for tile, (row, col) in zip(tiles, coords):
        row_hi = row >= row_mid
        col_hi = col >= col_mid

        # 0=low-row/low-col, 1=low-row/high-col,
        # 2=high-row/low-col, 3=high-row/high-col
        fold = (2 if row_hi else 0) + (1 if col_hi else 0)
        folds[fold].append(tile)

    return folds


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).transpose(1, 2, 0)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    image = Image.fromarray(arr, mode="RGB")
    tensor = transforms.ToTensor()(image)

    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(tensor)

    return tensor


def load_depth_tensor(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)

    if arr.shape != (1024, 1024):
        raise ValueError(f"{path}: expected 1024x1024, got {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: contains NaN/Inf")

    # Preserve the project's established DAv2 [0,1] representation.
    if arr.min() < -1e-6 or arr.max() > 1.000001:
        raise ValueError(
            f"{path}: expected DAv2 values in [0,1], "
            f"got min={arr.min()} max={arr.max()}"
        )

    return torch.from_numpy(arr).unsqueeze(0)


def load_target(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)

    if arr.shape != (1024, 1024):
        raise ValueError(f"{path}: expected 1024x1024, got {arr.shape}")

    # Match RDAH's original data treatment:
    # invalid AGL = -9999 -> invalid mask; valid target cast to uint16.
    valid = np.isfinite(arr) & (arr != -9999.0)
    # Preserve signed/float AGL values. Do NOT cast to uint16:
    # DFC2019 contains valid negative AGL values, and uint16 would wrap them
    # into huge positive heights and corrupt the training target.
    target = torch.from_numpy(np.where(valid, arr, 0.0).astype(np.float32)).unsqueeze(0)
    mask = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)

    return target, mask


def load_sample(tile: str):
    rgb = load_rgb_tensor(RGB_DIR / f"{tile}_RGB.tif")
    depth = load_depth_tensor(DEPTH_DIR / f"{tile}_depth.npy")
    target, mask = load_target(GT_DIR / f"{tile}_AGL.tif")

    # Original RDAH loader also excludes pure black RGB pixels.
    rgb_raw = None
    with rasterio.open(RGB_DIR / f"{tile}_RGB.tif") as src:
        raw = src.read([1, 2, 3]).transpose(1, 2, 0)
        rgb_raw = raw

    non_black = ~np.all(rgb_raw == 0, axis=2)
    mask = mask * torch.from_numpy(non_black.astype(np.float32)).unsqueeze(0)

    return depth, rgb, target, mask


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Faithful functional equivalent of RDAH-Net's MaskedLoss:
    SmoothL1(reduction='none'), valid mask, mean over valid pixels,
    plus a large NaN penalty.
    """
    safe_target = torch.where(torch.isnan(target), pred, target)
    valid = (~torch.isnan(safe_target)) & (mask > 0)

    nan_penalty = 1e4 * torch.isnan(pred).float().sum()

    loss = torch.nn.functional.smooth_l1_loss(
        pred, safe_target, reduction="none"
    )
    masked = loss.where(valid, torch.zeros_like(loss))

    num_valid = valid.sum()
    if num_valid.item() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return masked.sum() / num_valid + nan_penalty


def metrics_from_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    valid = mask.astype(bool)
    p = pred[valid].astype(np.float64)
    y = target[valid].astype(np.float64)

    if p.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "n": 0,
        }

    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    if p.size > 1 and np.std(p) > 0 and np.std(y) > 0:
        pearson = float(np.corrcoef(p, y)[0, 1])
    else:
        pearson = float("nan")

    # Rank correlation without scipy dependency.
    pr = p.argsort().argsort().astype(np.float64)
    yr = y.argsort().argsort().astype(np.float64)

    if np.std(pr) > 0 and np.std(yr) > 0:
        spearman = float(np.corrcoef(pr, yr)[0, 1])
    else:
        spearman = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "n": int(p.size),
    }


@torch.no_grad()
def evaluate_tiles(model, tiles: list[str], device: torch.device):
    model.eval()

    pred_chunks = []
    target_chunks = []
    mask_chunks = []

    per_tile = {}

    for idx, tile in enumerate(tiles, start=1):
        depth, rgb, target, mask = load_sample(tile)

        depth = depth.unsqueeze(0).to(device)
        rgb = rgb.unsqueeze(0).to(device)
        target_np = target.numpy().squeeze(0)
        mask_np = mask.numpy().squeeze(0)

        pred = model(depth, rgb).detach().cpu()

        # The official 1024x1024 model returns 1024x1024.
        if tuple(pred.shape[-2:]) != (1024, 1024):
            pred = torch.nn.functional.interpolate(
                pred,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            )

        pred_np = pred.squeeze(0).squeeze(0).numpy()

        m = metrics_from_arrays(pred_np, target_np, mask_np)
        per_tile[tile] = m

        valid = mask_np.astype(bool)
        pred_chunks.append(pred_np[valid])
        target_chunks.append(target_np[valid])
        mask_chunks.append(np.ones(valid.sum(), dtype=np.float32))

        print(
            f"  eval {idx:>3}/{len(tiles)} {tile}: "
            f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}"
        )

    if not pred_chunks:
        raise RuntimeError("No valid evaluation pixels.")

    p = np.concatenate(pred_chunks)
    y = np.concatenate(target_chunks)
    m = np.ones_like(p, dtype=np.float32)

    overall = metrics_from_arrays(p, y, m)
    overall["per_tile"] = per_tile
    return overall


def train_one_epoch(model, tiles, device, optimizer, epoch, total_epochs):
    model.train()

    losses = []

    for idx, tile in enumerate(tiles, start=1):
        depth, rgb, target, mask = load_sample(tile)

        depth = depth.unsqueeze(0).to(device)
        rgb = rgb.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(depth, rgb)

        if pred.shape[-2:] != target.shape[-2:]:
            pred = torch.nn.functional.interpolate(
                pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = masked_smooth_l1(pred, target, mask)
        loss.backward()

        # Keep this first adaptation conservative.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)

        if idx == 1 or idx % 10 == 0 or idx == len(tiles):
            print(
                f"  train epoch {epoch}/{total_epochs} "
                f"{idx:>3}/{len(tiles)} {tile}: loss={loss_value:.6f}"
            )

    return float(np.mean(losses))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    path.write_text(json.dumps(clean(data), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Run only fold 0-3.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Manifest: {MANIFEST}")
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    tiles = load_manifest_tiles()
    tiles = verify_tile_files(tiles)

    print(f"Usable manifest tiles: {len(tiles)}")
    if len(tiles) != 50:
        print(
            "WARNING: manifest/cache coverage is not the expected 50-tile "
            f"benchmark (usable={len(tiles)})."
        )

    folds = make_spatial_folds(tiles)

    print("\nSpatial folds:")
    for fold_id in range(4):
        print(f"  fold {fold_id}: {len(folds[fold_id])} tiles")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_ids = [args.fold] if args.fold is not None else [0, 1, 2, 3]

    all_results = []

    for fold_id in fold_ids:
        if fold_id not in range(4):
            raise ValueError("--fold must be 0, 1, 2 or 3.")

        test_tiles = folds[fold_id]
        train_tiles = [t for f in range(4) if f != fold_id for t in folds[f]]

        fold_dir = OUT_DIR / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"FOLD {fold_id}")
        print(f"train tiles: {len(train_tiles)}")
        print(f"test tiles:  {len(test_tiles)}")
        print("=" * 80)

        model, checkpoint = load_pretrained(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

        history = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_tiles,
                device,
                optimizer,
                epoch,
                args.epochs,
            )

            print(f"Epoch {epoch}: train_loss={train_loss:.6f}")

            val = evaluate_tiles(model, test_tiles, device)

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_mae": val["mae"],
                "test_rmse": val["rmse"],
                "test_pearson": val["pearson"],
                "test_spearman": val["spearman"],
                "test_n": val["n"],
            }
            history.append(epoch_record)

            print(
                f"  test: MAE={val['mae']:.4f} "
                f"RMSE={val['rmse']:.4f} "
                f"Pearson={val['pearson']:.4f} "
                f"Spearman={val['spearman']:.4f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "fold": fold_id,
                    "train_tiles": train_tiles,
                    "test_tiles": test_tiles,
                    "args": vars(args),
                },
                fold_dir / f"epoch{epoch}.pt",
            )

            save_json(fold_dir / "history.json", history)

        final = history[-1]
        fold_result = {
            "fold": fold_id,
            "train_tiles": train_tiles,
            "test_tiles": test_tiles,
            "final": final,
            "checkpoint": str(fold_dir / f"epoch{args.epochs}.pt"),
        }

        save_json(fold_dir / "result.json", fold_result)
        all_results.append(fold_result)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(OUT_DIR / "method_rdah_spatial_cv_results.json", {
        "experiment": "RDAH-FT-1",
        "description": "Tile-level four-quadrant spatial holdout fine-tuning",
        "checkpoint": str(CHECKPOINT),
        "manifest": str(MANIFEST),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "device": str(device),
        "folds": all_results,
    })

    print("\nDONE")
    print(f"Results: {OUT_DIR / 'method_rdah_spatial_cv_results.json'}")


if __name__ == "__main__":
    main()
