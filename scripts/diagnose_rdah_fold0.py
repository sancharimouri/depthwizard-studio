
#!/usr/bin/env python3
"""
Compare the original pretrained RDAH checkpoint against Fold-0 Epoch-1.

This is diagnostic only. It does NOT train or modify any checkpoints.

Usage:
    python scripts/diagnose_rdah_fold0.py
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDAH_DIR = PROJECT_ROOT / "external" / "RDAH-Net"

PRETRAINED = RDAH_DIR / "104best_model.pth"
FINETUNED = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "rdah"
    / "fold0"
    / "epoch1.pt"
)

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


def import_model_class():
    if str(RDAH_DIR) not in sys.path:
        sys.path.insert(0, str(RDAH_DIR))

    spec = importlib.util.spec_from_file_location(
        "rdah_test_diag", RDAH_DIR / "test.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import RDAH test.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HeightPredTransformer


def load_tiles():
    with MANIFEST.open("r", newline="") as f:
        reader = csv.DictReader(f)
        key = next(
            c for c in ["tile", "tile_id", "name", "id", "rgb"]
            if c in reader.fieldnames
        )

        tiles = []
        for row in reader:
            value = Path(str(row[key]).strip()).stem
            value = re.sub(r"_RGB$", "", value)
            if value:
                tiles.append(value)

    tiles = sorted(set(tiles))

    coords = np.array(
        [tuple(map(int, re.search(r"_(\d+)_(\d+)$", t).groups())) for t in tiles]
    )
    row_mid = float(np.median(coords[:, 0]))
    col_mid = float(np.median(coords[:, 1]))

    fold0 = []
    for tile, (row, col) in zip(tiles, coords):
        row_hi = row >= row_mid
        col_hi = col >= col_mid
        fold = (2 if row_hi else 0) + (1 if col_hi else 0)
        if fold == 0:
            fold0.append(tile)

    return fold0


def load_inputs(tile):
    with rasterio.open(RGB_DIR / f"{tile}_RGB.tif") as src:
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb_tensor = transforms.ToTensor()(Image.fromarray(rgb, mode="RGB"))
    rgb_tensor = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )(rgb_tensor).unsqueeze(0)

    depth = np.load(DEPTH_DIR / f"{tile}_depth.npy").astype(np.float32)
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)

    with rasterio.open(GT_DIR / f"{tile}_AGL.tif") as src:
        gt = src.read(1).astype(np.float32)

    valid = np.isfinite(gt) & (gt != -9999.0)

    # Match the original RDAH black-pixel masking.
    rgb_raw = rgb
    valid &= ~np.all(rgb_raw == 0, axis=2)

    return (
        depth_tensor,
        rgb_tensor,
        gt,
        valid,
    )


def load_models():
    HeightPredTransformer = import_model_class()

    pretrained = HeightPredTransformer()
    checkpoint = torch.load(PRETRAINED, map_location="cpu")
    pretrained.load_state_dict(checkpoint["model_state_dict"], strict=True)
    pretrained.eval()

    finetuned = HeightPredTransformer()
    checkpoint_ft = torch.load(FINETUNED, map_location="cpu")
    finetuned.load_state_dict(checkpoint_ft["model_state_dict"], strict=True)
    finetuned.eval()

    return pretrained, finetuned


def prediction(model, depth, rgb):
    with torch.no_grad():
        out = model(depth, rgb)

    if out.shape[-2:] != (1024, 1024):
        out = torch.nn.functional.interpolate(
            out,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

    return out.squeeze().cpu().numpy()


def metrics(pred, gt, valid):
    p = pred[valid].astype(np.float64)
    y = gt[valid].astype(np.float64)

    err = p - y

    abs_err = np.abs(err)
    sq_err = err * err

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))

    extreme50 = int(np.sum(abs_err > 50))
    extreme100 = int(np.sum(abs_err > 100))
    extreme200 = int(np.sum(abs_err > 200))

    if len(p) > 1 and np.std(p) > 0 and np.std(y) > 0:
        pearson = float(np.corrcoef(p, y)[0, 1])
    else:
        pearson = float("nan")

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
        "pred_min": float(p.min()),
        "pred_max": float(p.max()),
        "pred_mean": float(p.mean()),
        "pred_std": float(p.std()),
        "gt_min": float(y.min()),
        "gt_max": float(y.max()),
        "gt_mean": float(y.mean()),
        "gt_std": float(y.std()),
        "n": int(len(p)),
        "err_gt_50": extreme50,
        "err_gt_100": extreme100,
        "err_gt_200": extreme200,
    }


def print_metrics(label, m):
    print(f"\n{label}")
    print(
        f"  pred range: [{m['pred_min']:.4f}, {m['pred_max']:.4f}]  "
        f"mean={m['pred_mean']:.4f} std={m['pred_std']:.4f}"
    )
    print(
        f"  GT   range: [{m['gt_min']:.4f}, {m['gt_max']:.4f}]  "
        f"mean={m['gt_mean']:.4f} std={m['gt_std']:.4f}"
    )
    print(
        f"  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
        f"Pearson={m['pearson']:.4f}  Spearman={m['spearman']:.4f}"
    )
    print(
        f"  |error|>50m: {m['err_gt_50']}  "
        f">100m: {m['err_gt_100']}  "
        f">200m: {m['err_gt_200']}"
    )


def main():
    if not PRETRAINED.exists():
        raise FileNotFoundError(PRETRAINED)
    if not FINETUNED.exists():
        raise FileNotFoundError(FINETUNED)

    tiles = load_tiles()

    print(f"Fold-0 test tiles: {len(tiles)}")
    print("Tiles:", ", ".join(tiles))
    print(f"\nPretrained: {PRETRAINED}")
    print(f"Fine-tuned: {FINETUNED}")

    pretrained, finetuned = load_models()

    all_pre_p = []
    all_ft_p = []
    all_gt = []

    print("\n" + "=" * 88)

    for i, tile in enumerate(tiles, 1):
        depth, rgb, gt, valid = load_inputs(tile)

        pre = prediction(pretrained, depth, rgb)
        ft = prediction(finetuned, depth, rgb)

        pre_m = metrics(pre, gt, valid)
        ft_m = metrics(ft, gt, valid)

        all_pre_p.append(pre[valid])
        all_ft_p.append(ft[valid])
        all_gt.append(gt[valid])

        print(
            f"{i:02d}/{len(tiles)} {tile:<14} | "
            f"PRE MAE={pre_m['mae']:7.3f} RMSE={pre_m['rmse']:8.3f} "
            f"range=[{pre_m['pred_min']:8.2f},{pre_m['pred_max']:8.2f}] | "
            f"FT MAE={ft_m['mae']:7.3f} RMSE={ft_m['rmse']:8.3f} "
            f"range=[{ft_m['pred_min']:8.2f},{ft_m['pred_max']:8.2f}]"
        )

    pre_all = metrics(
        np.concatenate(all_pre_p),
        np.concatenate(all_gt),
        np.ones(np.concatenate(all_gt).shape, dtype=bool),
    )
    ft_all = metrics(
        np.concatenate(all_ft_p),
        np.concatenate(all_gt),
        np.ones(np.concatenate(all_gt).shape, dtype=bool),
    )

    print("\n" + "=" * 88)
    print("AGGREGATE HELD-OUT RESULTS")

    print_metrics("PRETRAINED CHECKPOINT", pre_all)
    print_metrics("FOLD 0 EPOCH 1", ft_all)


if __name__ == "__main__":
    main()

