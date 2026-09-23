#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "dav2_baseline"
)

MANIFEST_PATH = BASELINE_DIR / "manifest.csv"
DEPTH_DIR = BASELINE_DIR / "depth"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "sparse_anchor"
)

PRED_DIR = OUTPUT_DIR / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


ANCHOR_COUNTS = [2, 3, 5, 10, 20]
RANDOM_SEED = 42


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_agl(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def regression_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    valid = np.isfinite(pred) & np.isfinite(truth)

    x = pred[valid].astype(np.float64)
    y = truth[valid].astype(np.float64)

    errors = x - y

    if len(x) == 0:
        raise ValueError("No valid evaluation pixels.")

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    if np.std(x) == 0 or np.std(y) == 0:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)

    return {
        "valid_pixels": int(valid.sum()),
        "mae_m": mae,
        "rmse_m": rmse,
        "pearson": pearson,
        "spearman": spearman,
    }


def choose_spatially_distributed_anchors(
    depth: np.ndarray,
    agl: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select anchors using a combination of spatial spread and AGL diversity.

    Returns flattened pixel indices.
    """

    h, w = depth.shape

    valid = np.isfinite(depth) & np.isfinite(agl)

    # Ignore tiny negative/no-data-like values for anchor selection.
    valid &= agl >= 0

    ys, xs = np.where(valid)

    if len(xs) < n:
        raise ValueError(
            f"Not enough valid pixels for {n} anchors."
        )

    # --------------------------------------------------------------
    # Create candidates distributed across the image.
    # --------------------------------------------------------------

    candidate_count = min(len(xs), max(5000, n * 500))

    candidate_idx = rng.choice(
        len(xs),
        size=candidate_count,
        replace=False,
    )

    xs_c = xs[candidate_idx]
    ys_c = ys[candidate_idx]

    depth_c = depth[ys_c, xs_c]
    agl_c = agl[ys_c, xs_c]

    # --------------------------------------------------------------
    # Rank candidates by AGL so we can cover the vertical range.
    # --------------------------------------------------------------

    order = np.argsort(agl_c)

    # Pick approximately evenly spaced height quantiles.
    quantile_positions = np.linspace(
        0,
        len(order) - 1,
        n,
        dtype=int,
    )

    selected = []

    for pos in quantile_positions:
        idx = order[pos]
        selected.append(
            (
                ys_c[idx],
                xs_c[idx],
                agl_c[idx],
            )
        )

    # --------------------------------------------------------------
    # If two selected points are too close spatially, replace them
    # using a greedy distance criterion.
    # --------------------------------------------------------------

    final = []

    min_dist = max(8, int(min(h, w) / 50))

    for y, x, z in selected:
        if not final:
            final.append((y, x, z))
            continue

        distances = [
            np.hypot(y - yy, x - xx)
            for yy, xx, _ in final
        ]

        if min(distances) >= min_dist:
            final.append((y, x, z))

    # Fill remaining anchors greedily if needed.
    if len(final) < n:
        remaining = list(
            zip(ys_c, xs_c, agl_c)
        )

        rng.shuffle(remaining)

        for y, x, z in remaining:
            if any(
                y == yy and x == xx
                for yy, xx, _ in final
            ):
                continue

            if not final:
                final.append((y, x, z))
            else:
                distances = [
                    np.hypot(y - yy, x - xx)
                    for yy, xx, _ in final
                ]

                if min(distances) >= min_dist:
                    final.append((y, x, z))

            if len(final) >= n:
                break

    if len(final) < n:
        # Final fallback: random unique valid points.
        all_indices = np.flatnonzero(valid.ravel())

        needed = n - len(final)

        existing = {
            y * w + x
            for y, x, _ in final
        }

        candidates = [
            idx
            for idx in all_indices
            if idx not in existing
        ]

        chosen = rng.choice(
            candidates,
            size=needed,
            replace=False,
        )

        for idx in chosen:
            y, x = np.unravel_index(idx, (h, w))
            final.append(
                (y, x, agl[y, x])
            )

    return np.array(
        [y * w + x for y, x, _ in final],
        dtype=np.int64,
    )


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    rng = np.random.default_rng(RANDOM_SEED)

    all_results = []

    print("=" * 70)
    print("SPARSE METRIC ANCHOR EXPERIMENT")
    print("=" * 70)

    for tile_num, row in manifest.iterrows():
        tile_id = str(row["tile_id"])
        agl_path = resolve_path(str(row["agl_path"]))
        depth_path = DEPTH_DIR / f"{tile_id}_depth.npy"

        print(
            f"\n[{tile_num + 1}/{len(manifest)}] {tile_id}"
        )

        depth = np.load(depth_path).astype(np.float32)
        agl = load_agl(agl_path)

        valid = np.isfinite(depth) & np.isfinite(agl)

        for n_anchors in ANCHOR_COUNTS:

            anchor_indices = choose_spatially_distributed_anchors(
                depth,
                agl,
                n_anchors,
                rng,
            )

            # ------------------------------------------------------
            # Convert flattened indices back to coordinates.
            # ------------------------------------------------------

            h, w = depth.shape

            anchor_y, anchor_x = np.unravel_index(
                anchor_indices,
                (h, w),
            )

            anchor_depth = depth[anchor_y, anchor_x]
            anchor_agl = agl[anchor_y, anchor_x]

            # ------------------------------------------------------
            # Fit scene-specific affine mapping.
            # ------------------------------------------------------

            model = LinearRegression()

            model.fit(
                anchor_depth.reshape(-1, 1),
                anchor_agl,
            )

            predicted = (
                model.predict(
                    depth.reshape(-1, 1)
                )
                .reshape(depth.shape)
                .astype(np.float32)
            )

            # ------------------------------------------------------
            # Exclude anchor pixels from evaluation.
            # ------------------------------------------------------

            evaluation_mask = valid.copy()
            evaluation_mask[
                anchor_y,
                anchor_x,
            ] = False

            result = regression_metrics(
                predicted[evaluation_mask],
                agl[evaluation_mask],
            )

            result.update(
                {
                    "tile_id": tile_id,
                    "city": row["city"],
                    "anchor_count": n_anchors,
                    "slope": float(model.coef_[0]),
                    "intercept": float(model.intercept_),
                    "anchor_agl_min": float(anchor_agl.min()),
                    "anchor_agl_max": float(anchor_agl.max()),
                    "anchor_agl_std": float(anchor_agl.std()),
                }
            )

            all_results.append(result)

            print(
                f"  {n_anchors:2d} anchors | "
                f"MAE={result['mae_m']:.3f} m | "
                f"RMSE={result['rmse_m']:.3f} m | "
                f"P={result['pearson']:.3f} | "
                f"S={result['spearman']:.3f}"
            )

    results_df = pd.DataFrame(all_results)

    per_tile_path = OUTPUT_DIR / "sparse_anchor_metrics.csv"
    results_df.to_csv(per_tile_path, index=False)

    # --------------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------------

    overall = (
        results_df
        .groupby("anchor_count")
        .agg(
            tiles=("tile_id", "count"),
            mae_mean_m=("mae_m", "mean"),
            mae_median_m=("mae_m", "median"),
            rmse_mean_m=("rmse_m", "mean"),
            rmse_median_m=("rmse_m", "median"),
            pearson_mean=("pearson", "mean"),
            pearson_median=("pearson", "median"),
            spearman_mean=("spearman", "mean"),
            spearman_median=("spearman", "median"),
        )
        .reset_index()
    )

    overall_path = (
        OUTPUT_DIR / "sparse_anchor_summary.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    # --------------------------------------------------------------
    # City summary
    # --------------------------------------------------------------

    city_summary = (
        results_df
        .groupby(["anchor_count", "city"])
        .agg(
            tiles=("tile_id", "count"),
            mae_mean_m=("mae_m", "mean"),
            rmse_mean_m=("rmse_m", "mean"),
            pearson_mean=("pearson", "mean"),
            spearman_mean=("spearman", "mean"),
        )
        .reset_index()
    )

    city_path = (
        OUTPUT_DIR / "sparse_anchor_city_summary.csv"
    )

    city_summary.to_csv(
        city_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SPARSE ANCHOR SUMMARY")
    print("=" * 70)

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nFiles:")
    print(per_tile_path)
    print(overall_path)
    print(city_path)


if __name__ == "__main__":
    main()
