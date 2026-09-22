#!/usr/bin/env python3
"""
Pooled 50-tile spatial-CV evaluation for the already-trained RDAH folds.

Checkpoint-selection rule:
    FIXED FINAL EPOCH = 5 for every fold.
This rule is intentionally NOT chosen from test-fold metrics.

The script:
1. Reconstructs the same four spatial folds.
2. Verifies each fold's epoch5 checkpoint exists.
3. Verifies checkpoint test_tiles match the reconstructed fold split.
4. Runs each epoch5 checkpoint on its held-out tiles exactly once.
5. Saves each held-out prediction as <tile>_pred.npy.
6. Computes pooled MAE/RMSE/Pearson over all valid pixels from all 50 tiles.
7. Computes a deterministic pooled Spearman correlation on a bounded
   random sample of valid pixels to avoid multi-GB RAM use on a MacBook Air.
8. Writes per-tile metrics and a final JSON report.

Usage:
    python scripts/evaluate_rdah_pooled_cv.py

Optional:
    python scripts/evaluate_rdah_pooled_cv.py --epoch 5
    python scripts/evaluate_rdah_pooled_cv.py --spearman-sample 5000000
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDAH_DIR = PROJECT_ROOT / "external" / "RDAH-Net"
CHECKPOINT_ROOT = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "rdah"

RGB_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "RGB" / "Track1-RGB"
GT_DIR = PROJECT_ROOT / "data" / "dfc2019" / "raw" / "Truth" / "Track1-Truth"
DEPTH_DIR = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "dav2_baseline"
    / "depth"
)
MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "dav2_baseline"
    / "manifest.csv"
)

DEFAULT_EPOCH = 5
DEFAULT_SPEARMAN_SAMPLE = 2_000_000


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but unavailable.")
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
    return module.HeightPredTransformer


def load_manifest_tiles() -> list[str]:
    with MANIFEST.open("r", newline="") as f:
        reader = csv.DictReader(f)
        candidates = ["tile", "tile_id", "name", "id", "rgb"]
        key = next((c for c in candidates if c in reader.fieldnames), None)
        if key is None:
            raise RuntimeError(
                f"Could not identify tile column. Columns: {reader.fieldnames}"
            )

        tiles = []
        for row in reader:
            value = str(row[key]).strip()
            if not value:
                continue
            value = Path(value).stem
            value = re.sub(r"_RGB$", "", value)
            tiles.append(value)

    tiles = sorted(set(tiles))
    return tiles


def verify_tile_files(tiles: list[str]) -> list[str]:
    usable = []
    for tile in tiles:
        paths = [
            RGB_DIR / f"{tile}_RGB.tif",
            DEPTH_DIR / f"{tile}_depth.npy",
            GT_DIR / f"{tile}_AGL.tif",
        ]
        if all(p.exists() for p in paths):
            usable.append(tile)
    return usable


def tile_coords(tile: str) -> tuple[int, int]:
    m = re.search(r"_(\d+)_(\d+)$", tile)
    if not m:
        raise ValueError(f"Cannot parse DFC tile id: {tile}")
    return int(m.group(1)), int(m.group(2))


def make_spatial_folds(tiles: list[str]) -> dict[int, list[str]]:
    coords = np.array([tile_coords(t) for t in tiles], dtype=np.float64)
    row_mid = float(np.median(coords[:, 0]))
    col_mid = float(np.median(coords[:, 1]))

    folds = {0: [], 1: [], 2: [], 3: []}
    for tile, (row, col) in zip(tiles, coords):
        row_hi = row >= row_mid
        col_hi = col >= col_mid
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
        raise ValueError(f"{path}: NaN/Inf in DAv2 cache")
    if arr.min() < -1e-6 or arr.max() > 1.000001:
        raise ValueError(
            f"{path}: expected [0,1], got min={arr.min()} max={arr.max()}"
        )
    return torch.from_numpy(arr).unsqueeze(0)


def load_target_and_mask(tile: str) -> tuple[np.ndarray, np.ndarray]:
    gt_path = GT_DIR / f"{tile}_AGL.tif"
    rgb_path = RGB_DIR / f"{tile}_RGB.tif"

    with rasterio.open(gt_path) as src:
        target = src.read(1).astype(np.float32)

    if target.shape != (1024, 1024):
        raise ValueError(f"{gt_path}: expected 1024x1024")

    valid = np.isfinite(target) & (target != -9999.0)

    # Preserve signed AGL; NEVER cast to uint16.
    target = np.where(valid, target, 0.0).astype(np.float32)

    # Match training adapter's exclusion of pure-black RGB pixels.
    with rasterio.open(rgb_path) as src:
        raw = src.read([1, 2, 3]).transpose(1, 2, 0)

    non_black = ~np.all(raw == 0, axis=2)
    valid &= non_black

    return target, valid


def load_model(device: torch.device, checkpoint_path: Path):
    HeightPredTransformer = import_rdah_model_class()
    model = HeightPredTransformer()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def infer_tile(model, tile: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    depth = load_depth_tensor(DEPTH_DIR / f"{tile}_depth.npy")
    rgb = load_rgb_tensor(RGB_DIR / f"{tile}_RGB.tif")

    with torch.inference_mode():
        pred = model(
            depth.unsqueeze(0).to(device),
            rgb.unsqueeze(0).to(device),
        )

    if tuple(pred.shape[-2:]) != (1024, 1024):
        pred = F.interpolate(
            pred,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

    pred = pred.squeeze().detach().cpu().numpy().astype(np.float32)
    target, valid = load_target_and_mask(tile)
    return pred, target, valid


def tile_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray):
    p = pred[valid].astype(np.float64)
    y = target[valid].astype(np.float64)

    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))

    if np.std(p) > 0 and np.std(y) > 0:
        pearson = float(np.corrcoef(p, y)[0, 1])
    else:
        pearson = float("nan")

    return {
        "n": int(p.size),
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
    }


def update_streaming_stats(
    p: np.ndarray,
    y: np.ndarray,
    state: dict[str, float | int],
):
    p = p.astype(np.float64)
    y = y.astype(np.float64)

    state["n"] += int(p.size)
    state["sum_abs"] += float(np.abs(p - y).sum())
    state["sum_sq"] += float(np.square(p - y).sum())
    state["sum_p"] += float(p.sum())
    state["sum_y"] += float(y.sum())
    state["sum_p2"] += float(np.square(p).sum())
    state["sum_y2"] += float(np.square(y).sum())
    state["sum_py"] += float((p * y).sum())


def finalize_streaming_stats(state):
    n = float(state["n"])
    if n <= 0:
        raise RuntimeError("No valid pixels.")

    mae = state["sum_abs"] / n
    rmse = math.sqrt(state["sum_sq"] / n)

    cov_num = state["sum_py"] - state["sum_p"] * state["sum_y"] / n
    var_p = state["sum_p2"] - state["sum_p"] ** 2 / n
    var_y = state["sum_y2"] - state["sum_y"] ** 2 / n

    if var_p > 0 and var_y > 0:
        pearson = cov_num / math.sqrt(var_p * var_y)
    else:
        pearson = float("nan")

    return {
        "n_pixels": int(state["n"]),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
    }


def deterministic_reservoir_update(
    sample_p: list[float],
    sample_y: list[float],
    p: np.ndarray,
    y: np.ndarray,
    seen: int,
    limit: int,
    rng: np.random.Generator,
):
    """
    Reservoir sample so the pooled Spearman estimate is not dominated by
    whichever fold/tile happens to be processed last.
    """
    for pv, yv in zip(p, y):
        seen += 1
        if len(sample_p) < limit:
            sample_p.append(float(pv))
            sample_y.append(float(yv))
        else:
            j = int(rng.integers(0, seen))
            if j < limit:
                sample_p[j] = float(pv)
                sample_y[j] = float(yv)
    return seen


def spearman_from_sample(p: np.ndarray, y: np.ndarray) -> float:
    if p.size < 2:
        return float("nan")

    # scipy is used only here; if unavailable, report clearly.
    try:
        from scipy.stats import spearmanr
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for pooled Spearman. "
            "Install it with: pip install scipy"
        ) from exc

    result = spearmanr(p, y)
    return float(result.statistic)


def clean_for_json(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        type=int,
        default=DEFAULT_EPOCH,
        help="Fixed checkpoint epoch used for every fold (default: 5).",
    )
    parser.add_argument(
        "--spearman-sample",
        type=int,
        default=DEFAULT_SPEARMAN_SAMPLE,
        help="Number of pooled valid pixels sampled for Spearman.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Device: {device}")
    print(f"Fixed checkpoint rule: epoch {args.epoch} for EVERY fold")
    print("Checkpoint selection uses NO held-out test metrics.")

    tiles = verify_tile_files(load_manifest_tiles())
    if len(tiles) != 50:
        raise RuntimeError(f"Expected 50 usable benchmark tiles, found {len(tiles)}")

    folds = make_spatial_folds(tiles)
    print("\nSpatial folds:")
    for f in range(4):
        print(f"  fold {f}: {len(folds[f])} tiles")

    pred_root = CHECKPOINT_ROOT / f"pooled_cv_epoch{args.epoch}" / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    pooled = {
        "n": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "sum_p": 0.0,
        "sum_y": 0.0,
        "sum_p2": 0.0,
        "sum_y2": 0.0,
        "sum_py": 0.0,
    }

    per_tile = {}
    sample_p: list[float] = []
    sample_y: list[float] = []
    seen = 0
    rng = np.random.default_rng(12345)

    total_tiles = 0

    for fold in range(4):
        checkpoint_path = CHECKPOINT_ROOT / f"fold{fold}" / f"epoch{args.epoch}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        model, checkpoint = load_model(device, checkpoint_path)

        expected_test = sorted(folds[fold])
        checkpoint_test = sorted(checkpoint.get("test_tiles", []))

        if expected_test != checkpoint_test:
            raise RuntimeError(
                f"Fold {fold} checkpoint test_tiles do not match the current "
                "reconstructed fold split.\n"
                f"Expected: {expected_test}\n"
                f"Checkpoint: {checkpoint_test}"
            )

        print("\n" + "=" * 80)
        print(f"FOLD {fold} | checkpoint epoch {args.epoch}")
        print(f"held-out tiles: {len(expected_test)}")
        print("=" * 80)

        for i, tile in enumerate(expected_test, start=1):
            pred, target, valid = infer_tile(model, tile, device)
            m = tile_metrics(pred, target, valid)

            np.save(pred_root / f"{tile}_pred.npy", pred)

            per_tile[tile] = {
                "fold": fold,
                "n_pixels": m["n"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "pearson": m["pearson"],
            }

            p = pred[valid]
            y = target[valid]

            update_streaming_stats(p, y, pooled)

            remaining = args.spearman_sample - len(sample_p)
            # Reservoir function handles replacement once full.
            seen = deterministic_reservoir_update(
                sample_p,
                sample_y,
                p,
                y,
                seen,
                args.spearman_sample,
                rng,
            )

            total_tiles += 1
            print(
                f"  {i:>2}/{len(expected_test)} {tile}: "
                f"MAE={m['mae']:.4f} "
                f"RMSE={m['rmse']:.4f} "
                f"Pearson={m['pearson']:.4f}"
            )

        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    pooled_metrics = finalize_streaming_stats(pooled)

    sp = spearman_from_sample(
        np.asarray(sample_p, dtype=np.float64),
        np.asarray(sample_y, dtype=np.float64),
    )

    pooled_metrics["spearman"] = sp
    pooled_metrics["spearman_type"] = "deterministic reservoir sample"
    pooled_metrics["spearman_sample_n"] = len(sample_p)

    fold_summary = {}
    for fold in range(4):
        names = folds[fold]
        n = sum(per_tile[t]["n_pixels"] for t in names)
        mae = sum(per_tile[t]["mae"] * per_tile[t]["n_pixels"] for t in names) / n
        # RMSE must be recomputed from squared error; this weighted mean of
        # per-tile RMSE is NOT used.
        fold_summary[str(fold)] = {
            "n_tiles": len(names),
            "n_pixels": n,
            "mean_tile_mae": float(np.mean([per_tile[t]["mae"] for t in names])),
            "mean_tile_rmse": float(np.mean([per_tile[t]["rmse"] for t in names])),
            "pixel_weighted_mae": float(mae),
        }

    report = {
        "experiment": "RDAH-FT-1 pooled 50-tile spatial CV",
        "checkpoint_selection": {
            "rule": "fixed final epoch for every fold",
            "epoch": args.epoch,
            "uses_test_metrics": False,
        },
        "device": str(device),
        "n_tiles": total_tiles,
        "fold_sizes": {str(k): len(v) for k, v in folds.items()},
        "pooled_metrics": pooled_metrics,
        "fold_summary": fold_summary,
        "per_tile": per_tile,
        "prediction_directory": str(pred_root),
        "notes": [
            "Each benchmark tile is evaluated exactly once using the checkpoint "
            "from the fold in which that tile was held out.",
            "MAE, RMSE and Pearson are exact pooled pixel-level statistics.",
            "Spearman is computed on a deterministic reservoir sample to avoid "
            "multi-GB memory use; it is not an exact all-pixel Spearman.",
        ],
    }

    out_path = CHECKPOINT_ROOT / f"pooled_cv_epoch{args.epoch}_report.json"
    out_path.write_text(json.dumps(clean_for_json(report), indent=2))

    print("\n" + "=" * 80)
    print("POOLED 50-TILE SPATIAL-CV RESULT")
    print("=" * 80)
    print(f"Checkpoint rule : fixed epoch {args.epoch}")
    print(f"Tiles           : {total_tiles}")
    print(f"Valid pixels    : {pooled_metrics['n_pixels']:,}")
    print(f"MAE             : {pooled_metrics['mae']:.6f} m")
    print(f"RMSE            : {pooled_metrics['rmse']:.6f} m")
    print(f"Pearson         : {pooled_metrics['pearson']:.6f}")
    print(
        f"Spearman        : {pooled_metrics['spearman']:.6f} "
        f"(sample n={pooled_metrics['spearman_sample_n']:,})"
    )
    print(f"\nPredictions: {pred_root}")
    print(f"Report:      {out_path}")


if __name__ == "__main__":
    main()
