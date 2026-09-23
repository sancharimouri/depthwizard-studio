#!/usr/bin/env python3
"""
DepthWizard2 — Stage 3's "repeat quadrant-wise with Huber" branch, finally run.

Stage 3A (evaluate_sparse_anchor_spatial.py) compared tile-wide vs.
quadrant-wise affine calibration, both with OLS, and planned but never ran
a "repeat with Huber if the difference is meaningful" branch (confirmed
absent in docs/method-audit/02-gcp-regression/gaps-and-fixes.md §6). This
script runs exactly that branch, and only that branch:

    quadrant-wise, Huber regressor, EXACT same Stage-1 grid/20 anchors
    Stage 3 originally used.

Tile-wide is NOT re-run here: tile-wide is a single affine fit per tile
using all of that tile's own anchors, which is exactly what Stage 2's
Grid+Huber+20 already computed (MAE 2.929 / RMSE 4.718 / Pearson 0.532 /
Spearman 0.471, from data/dfc2019/experiments/sparse_anchor_regression/) —
re-fitting it here would reproduce the same numbers under a different
script, not answer a new question.

Every other piece of Stage 3A's logic is imported directly from
evaluate_sparse_anchor_spatial.py (region splitting, anchor loading, eval
masks, metric calculation) rather than reimplemented — only fit_ols is
swapped for a Huber fit, matching Stage 2's exact HuberRegressor
hyperparameters (epsilon=1.35, alpha=1e-4, max_iter=2000, tol=1e-8).

Primary comparison (matching every prior stage in this audit trail): the
strict spatial holdout, "quadrant_wise_aggregate" evaluation — the stitched
4-quadrant prediction map evaluated against the whole tile (same shape as
tile-wide's own evaluation, so the two are directly comparable).

No new anchors, no retraining, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import evaluate_sparse_anchor_spatial as s3  # noqa: E402


HUBER_KWARGS = dict(epsilon=1.35, alpha=1e-4, max_iter=2000, tol=1e-8)


def fit_huber(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        raise ValueError("At least 2 anchors are required.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite anchor values.")
    if np.ptp(x) < 1e-10:
        raise ValueError("Anchor DAv2 values have effectively zero spread.")

    model = HuberRegressor(**HUBER_KWARGS)
    model.fit(x.reshape(-1, 1), y)

    slope = float(np.asarray(model.coef_).reshape(-1)[0])
    intercept = float(model.intercept_)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("Huber produced non-finite parameters.")

    return slope, intercept


def run_tile_quadrant_huber(
    tile_id: str,
    depth_path: Path,
    truth_path: Path,
    stage1_dir: Path,
    anchor_count: int,
    placement: str,
    buffer_radius: int,
    min_eval_pixels: int,
) -> dict:
    depth = s3.read_depth(depth_path)
    agl = s3.read_agl(truth_path)

    if depth.shape != agl.shape:
        raise ValueError(f"{tile_id}: DAv2/AGL shape mismatch.")

    valid = s3.valid_mask(depth, agl)

    anchor_path = s3.stage1_anchor_path(stage1_dir, tile_id, placement, anchor_count)
    anchor_data = s3.load_stage1_anchors(anchor_path)

    rows = anchor_data["rows"]
    cols = anchor_data["cols"]
    anchor_agl = anchor_data["agl"]

    if len(rows) != anchor_count:
        raise ValueError(f"{tile_id}: expected {anchor_count} anchors, got {len(rows)}.")
    if not np.all(valid[rows, cols]):
        raise ValueError(f"{tile_id}: invalid Stage-1 anchor.")

    current_dav2 = depth[rows, cols].astype(np.float64)

    if not np.allclose(anchor_data["dav2_saved"], current_dav2, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{tile_id}: Stage-1 DAv2 values changed.")

    # -------------------------------------------------------------
    # Fit 2x2 quadrant models -- Huber instead of OLS. Everything
    # else (region assignment, anchor reuse) is identical to Stage 3A.
    # -------------------------------------------------------------
    anchor_regions = s3.region_id_for_pixel(rows, cols, depth.shape[0], depth.shape[1])

    quadrant_prediction = np.full(depth.shape, np.nan, dtype=np.float64)

    for region in range(4):
        selector = anchor_regions == region
        region_x = current_dav2[selector]
        region_y = anchor_agl[selector]

        if len(region_x) < s3.DEFAULT_MIN_ANCHORS_PER_REGION:
            raise ValueError(f"{tile_id}: {s3.region_name(region)} has only {len(region_x)} anchors.")

        slope, intercept = fit_huber(region_x, region_y)

        qmask = s3.region_mask(depth.shape, region)
        quadrant_prediction[qmask] = slope * depth[qmask].astype(np.float64) + intercept

    if not np.all(np.isfinite(quadrant_prediction[valid])):
        raise ValueError(f"{tile_id}: quadrant model produced non-finite predictions on valid pixels.")

    # -------------------------------------------------------------
    # Evaluation masks -- identical to Stage 3A.
    # -------------------------------------------------------------
    anchor_mask = np.zeros(valid.shape, dtype=bool)
    anchor_mask[rows, cols] = True

    exclusion = s3.make_exclusion_mask(valid.shape, rows, cols, buffer_radius)

    all_eval_mask = valid & ~anchor_mask
    strict_eval_mask = valid & ~exclusion

    if np.count_nonzero(all_eval_mask) < min_eval_pixels:
        raise ValueError(f"{tile_id}: insufficient all-pixel evaluation set.")
    if np.count_nonzero(strict_eval_mask) < min_eval_pixels:
        raise ValueError(f"{tile_id}: insufficient strict evaluation set.")

    region_name_str = "JAX" if tile_id.startswith("JAX_") else "OMA" if tile_id.startswith("OMA_") else "UNKNOWN"

    row = {"tile_id": tile_id, "region": region_name_str}

    for mask_name, mask in [("all", all_eval_mask), ("strict", strict_eval_mask)]:
        metrics_result, count = s3.evaluate_prediction_mask(agl, quadrant_prediction, mask)
        for k, v in metrics_result.items():
            row[f"{mask_name}_{k}"] = v
        row[f"{mask_name}_eval_pixels"] = count

    return row


def main() -> int:
    manifest = pd.read_csv(s3.DEFAULT_MANIFEST)
    tile_ids = manifest["tile_id"].astype(str).tolist()

    anchor_count = s3.DEFAULT_ANCHOR_COUNT
    placement = s3.DEFAULT_PLACEMENT
    buffer_radius = s3.DEFAULT_BUFFER_RADIUS
    min_eval_pixels = s3.DEFAULT_MIN_EVAL_PIXELS
    stage1_dir = s3.DEFAULT_STAGE1_DIR

    print("=" * 78)
    print("Quadrant-wise + Huber (Stage 3's never-run branch)")
    print(f"Anchors: {placement}/{anchor_count} (exact Stage-1 reuse)")
    print(f"Huber params: {HUBER_KWARGS}")
    print("=" * 78)

    rows = []
    for i, tile_id in enumerate(tile_ids, start=1):
        depth_path = s3.DEFAULT_DEPTH_DIR / f"{tile_id}_depth.npy"
        truth_path = s3.DEFAULT_TRUTH_DIR / f"{tile_id}_AGL.tif"

        print(f"{i:3d}/{len(tile_ids)} {tile_id} ...", end=" ", flush=True)
        row = run_tile_quadrant_huber(
            tile_id, depth_path, truth_path, stage1_dir,
            anchor_count, placement, buffer_radius, min_eval_pixels,
        )
        rows.append(row)
        print(f"strict MAE={row['strict_mae_m']:.3f} RMSE={row['strict_rmse_m']:.3f} P={row['strict_pearson']:.4f}")

    df = pd.DataFrame(rows)

    out_dir = PROJECT_ROOT.parent / "data/dfc2019/experiments/sparse_anchor_spatial_huber"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)

    summary = {
        "n_tiles": len(df),
        "strict_mae_m_mean": float(df["strict_mae_m"].mean()),
        "strict_rmse_m_mean": float(df["strict_rmse_m"].mean()),
        "strict_pearson_mean": float(df["strict_pearson"].mean()),
        "strict_spearman_mean": float(df["strict_spearman"].mean()),
        "all_mae_m_mean": float(df["all_mae_m"].mean()),
        "all_rmse_m_mean": float(df["all_rmse_m"].mean()),
        "all_pearson_mean": float(df["all_pearson"].mean()),
        "all_spearman_mean": float(df["all_spearman"].mean()),
    }

    import json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 78)
    print("QUADRANT-WISE + HUBER -- FINAL (strict spatial holdout, n=50 tiles)")
    print("=" * 78)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print(f"Saved: {out_dir / 'results.csv'}")
    print(f"Saved: {out_dir / 'summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
