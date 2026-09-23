#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "dav2_baseline"
)

MANIFEST_PATH = EXPERIMENT_DIR / "manifest.csv"
DEPTH_DIR = EXPERIMENT_DIR / "depth"
OUTPUT_PATH = EXPERIMENT_DIR / "diagnostic_summary.csv"

GAUSSIAN_SIGMA = 2.0


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)

    if len(x) < 2:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return (
        float(pearsonr(x, y).statistic),
        float(spearmanr(x, y).statistic),
    )


def load_agl(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    rows = []

    for i, row in manifest.iterrows():
        tile_id = str(row["tile_id"])

        agl_path = PROJECT_ROOT / str(row["agl_path"])
        depth_path = DEPTH_DIR / f"{tile_id}_depth.npy"

        if not agl_path.exists():
            print(f"[{i+1}/{len(manifest)}] {tile_id}: missing AGL")
            continue

        if not depth_path.exists():
            print(f"[{i+1}/{len(manifest)}] {tile_id}: missing DAv2")
            continue

        agl = load_agl(agl_path)
        depth = np.load(depth_path).astype(np.float32)

        if agl.shape != depth.shape:
            print(
                f"[{i+1}/{len(manifest)}] {tile_id}: "
                f"shape mismatch {depth.shape} vs {agl.shape}"
            )
            continue

        finite = np.isfinite(agl) & np.isfinite(depth)

        masks = {
            "all": finite,
            "gt0": finite & (agl > 0),
            "gt1": finite & (agl > 1),
            "gt2": finite & (agl > 2),
            "gt5": finite & (agl > 5),
        }

        result = {
            "tile_id": tile_id,
            "city": row["city"],
            "agl_p95": row["agl_p95"],
            "agl_max": row["agl_max"],
        }

        # Standard pixel-level correlations
        for name, mask in masks.items():
            p, s = correlation(depth, agl, mask)

            result[f"pearson_{name}"] = p
            result[f"spearman_{name}"] = s

        # Smoothed diagnostic
        depth_blur = gaussian_filter(depth, sigma=GAUSSIAN_SIGMA)
        agl_blur = gaussian_filter(agl, sigma=GAUSSIAN_SIGMA)

        p_blur, s_blur = correlation(
            depth_blur,
            agl_blur,
            finite,
        )

        result["pearson_smooth"] = p_blur
        result["spearman_smooth"] = s_blur

        rows.append(result)

        print(
            f"[{i+1}/{len(manifest)}] "
            f"{tile_id}: "
            f"P={result['pearson_all']:.3f}, "
            f"S={result['spearman_all']:.3f}"
        )

    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("DFC2019 50-TILE DIAGNOSTIC SUMMARY")
    print("=" * 70)

    metric_columns = [
        "pearson_all",
        "pearson_gt0",
        "pearson_gt1",
        "pearson_gt2",
        "pearson_gt5",
        "pearson_smooth",
        "spearman_all",
        "spearman_gt0",
        "spearman_gt1",
        "spearman_gt2",
        "spearman_gt5",
        "spearman_smooth",
    ]

    print("\nOVERALL MEANS")
    print(df[metric_columns].mean().to_string())

    print("\nOVERALL MEDIANS")
    print(df[metric_columns].median().to_string())

    print("\nBY CITY — MEAN")
    print(
        df.groupby("city")[metric_columns]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nSMOOTHING IMPROVEMENT")
    print(
        f"Mean Pearson change   : "
        f"{(df['pearson_smooth'] - df['pearson_all']).mean():+.4f}"
    )

    print(
        f"Mean Spearman change  : "
        f"{(df['spearman_smooth'] - df['spearman_all']).mean():+.4f}"
    )

    print("\nAGL > 1m EFFECT")
    print(
        f"Mean Pearson change   : "
        f"{(df['pearson_gt1'] - df['pearson_all']).mean():+.4f}"
    )

    print(
        f"Mean Spearman change  : "
        f"{(df['spearman_gt1'] - df['spearman_all']).mean():+.4f}"
    )

    print("\nOUTPUT")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
