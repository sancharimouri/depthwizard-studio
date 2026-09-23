#!/usr/bin/env python3
"""
DepthWizard2 — Stage 2: Sparse-Anchor Regression Robustness

Scientific purpose
------------------
Stage 1 showed:

1. More anchors helped:
      5 -> 10 -> 20
2. Grid and random placement were preferable to the current
   spatial+height heuristic.
3. There were substantial catastrophic tails, especially with random
   placement and few anchors.
4. Therefore the next question is NOT "which placement should we try?"
   but:
       "Are OLS failures partly caused by the regression estimator?"

This script isolates that question.

CRITICAL EXPERIMENTAL DESIGN
----------------------------
We DO NOT regenerate anchors.

Instead, we reuse the exact anchor JSON files saved by
Stage 1 (`evaluate_sparse_anchor_robust.py`).

That means:
    same tile
    same anchor locations
    same anchor count
    same random repetition
    same evaluation masks

The ONLY changed variable is the regression estimator:

    OLS
    Huber
    RANSAC

Primary Stage-2 configurations:
    placement: random, grid
    anchor counts: 10, 20
    regression: OLS, Huber, RANSAC

Random placement keeps ALL 20 Stage-1 repetitions.

Therefore:
    50 tiles
    x 2 counts
    x (20 random + 1 grid)
    x 3 regressors
    = 6,300 fitted/evaluated runs

Evaluation:
    A. all valid non-anchor pixels
    B. strict spatial holdout outside the same Stage-1 exclusion buffer

Outputs
-------
data/dfc2019/experiments/sparse_anchor_regression/
    config.json
    results.csv
    summary.csv
    distributions.csv
    paired_deltas.csv
    REPORT.txt
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data/dfc2019/experiments/dav2_baseline/manifest.csv"
)

DEFAULT_DEPTH_DIR = (
    PROJECT_ROOT
    / "data/dfc2019/experiments/dav2_baseline/depth"
)

DEFAULT_TRUTH_DIR = (
    PROJECT_ROOT
    / "data/dfc2019/raw/Truth/Track1-Truth"
)

DEFAULT_STAGE1_DIR = (
    PROJECT_ROOT
    / "data/dfc2019/experiments/sparse_anchor_robust"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/dfc2019/experiments/sparse_anchor_regression"
)

DEFAULT_COUNTS = [10, 20]
DEFAULT_STRATEGIES = ["random", "grid"]
DEFAULT_REGRESSORS = ["ols", "huber", "ransac"]

DEFAULT_BUFFER_RADIUS = 16
DEFAULT_MIN_EVAL_PIXELS = 1000

VALID_AGL_EPS = 1e-6


# ============================================================================
# Helpers
# ============================================================================

def read_depth(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(path, allow_pickle=False)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2-D depth array, got {arr.shape}: {path}"
        )

    return arr.astype(np.float32, copy=False)


def read_agl(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32, copy=False)


def valid_mask(depth: np.ndarray, agl: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(depth)
        & np.isfinite(agl)
        & (agl >= -VALID_AGL_EPS)
    )


def safe_correlations(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float]:
    if len(truth) < 2:
        return math.nan, math.nan

    if (
        np.allclose(truth, truth[0])
        or np.allclose(prediction, prediction[0])
    ):
        return math.nan, math.nan

    return (
        float(pearsonr(truth, prediction).statistic),
        float(spearmanr(truth, prediction).statistic),
    )


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    residual = prediction - truth

    p, s = safe_correlations(truth, prediction)

    return {
        "mae_m": float(np.mean(np.abs(residual))),
        "rmse_m": float(np.sqrt(np.mean(residual ** 2))),
        "bias_m": float(np.mean(residual)),
        "pearson": p,
        "spearman": s,
    }


def make_exclusion_mask(
    shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius < 0:
        raise ValueError("Buffer radius cannot be negative.")

    height, width = shape
    result = np.zeros(shape, dtype=bool)

    r = int(radius)

    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy * yy + xx * xx) <= r * r

    dy, dx = np.where(disk)
    dy -= r
    dx -= r

    for row, col in zip(rows, cols):
        rr = row + dy
        cc = col + dx

        inside = (
            (rr >= 0)
            & (rr < height)
            & (cc >= 0)
            & (cc < width)
        )

        result[rr[inside], cc[inside]] = True

    return result


# ============================================================================
# Stage-1 anchor loading
# ============================================================================

def anchor_file(
    stage1_dir: Path,
    tile_id: str,
    placement: str,
    anchor_count: int,
    repeat: int,
) -> Path:
    return (
        stage1_dir
        / "anchors"
        / (
            f"{tile_id}_{placement}_"
            f"{anchor_count}_{repeat:02d}.json"
        )
    )


def load_anchor_set(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Stage-1 anchor file is missing:\n{path}\n"
            "Run the complete Stage-1 experiment first."
        )

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "anchors" not in payload:
        raise ValueError(
            f"Malformed anchor file; missing 'anchors': {path}"
        )

    anchors = payload["anchors"]

    if not anchors:
        raise ValueError(f"No anchors in {path}")

    rows = np.asarray(
        [int(a["row"]) for a in anchors],
        dtype=np.int32,
    )
    cols = np.asarray(
        [int(a["col"]) for a in anchors],
        dtype=np.int32,
    )
    agl = np.asarray(
        [float(a["agl_m"]) for a in anchors],
        dtype=np.float64,
    )

    dav2_saved = np.asarray(
        [float(a["dav2"]) for a in anchors],
        dtype=np.float64,
    )

    coordinates = np.column_stack([rows, cols])

    if len(np.unique(coordinates, axis=0)) != len(anchors):
        raise ValueError(
            f"Duplicate anchor coordinates in {path}"
        )

    if not np.all(np.isfinite(agl)):
        raise ValueError(
            f"Non-finite anchor AGL in {path}"
        )

    if not np.all(np.isfinite(dav2_saved)):
        raise ValueError(
            f"Non-finite saved DAv2 anchor values in {path}"
        )

    return {
        "rows": rows,
        "cols": cols,
        "agl": agl,
        "dav2_saved": dav2_saved,
        "payload": payload,
    }


# ============================================================================
# Regressors
# ============================================================================

def build_regressor(name: str):
    """
    Fixed estimator definitions for the whole experiment.

    OLS:
        Standard baseline.

    Huber:
        Robust linear regression. epsilon=1.35 is the standard robust default.
        alpha is kept very small so regularization does not become the main
        experimental variable.

    RANSAC:
        Robust consensus estimator.
        residual_threshold=None delegates the scale threshold to scikit-learn's
        documented default based on the target distribution.
        random_state is fixed for reproducibility.
    """
    if name == "ols":
        return LinearRegression()

    if name == "huber":
        return HuberRegressor(
            epsilon=1.35,
            alpha=1e-4,
            max_iter=2000,
            tol=1e-8,
        )

    if name == "ransac":
        return RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=2,
            residual_threshold=None,
            max_trials=100,
            stop_probability=0.999,
            loss="absolute_error",
            random_state=0,
        )

    raise ValueError(f"Unknown regressor: {name}")


def fit_regression(
    name: str,
    depth: np.ndarray,
    anchor_data: dict,
) -> tuple[object, dict[str, float]]:
    rows = anchor_data["rows"]
    cols = anchor_data["cols"]
    y = anchor_data["agl"]

    x = depth[rows, cols].astype(np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "At least one anchor maps to non-finite DAv2."
        )

    if np.ptp(x) < 1e-10:
        raise ValueError(
            "Anchor DAv2 values have essentially zero range."
        )

    model = build_regressor(name)

    model.fit(x.reshape(-1, 1), y)

    # LinearRegression and HuberRegressor expose coef_/intercept_
    # directly. RANSACRegressor is a meta-estimator: the fitted linear
    # estimator lives in model.estimator_ after fitting.
    if name == "ransac":
        fitted_linear = getattr(model, "estimator_", None)

        if fitted_linear is None:
            raise ValueError(
                "RANSAC did not expose a fitted estimator_ after fit()."
            )

        slope = float(
            np.asarray(fitted_linear.coef_).reshape(-1)[0]
        )
        intercept = float(fitted_linear.intercept_)

    else:
        slope = float(
            np.asarray(model.coef_).reshape(-1)[0]
        )
        intercept = float(model.intercept_)

    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError(
            f"{name} produced non-finite slope/intercept."
        )

    extra = {
        "slope_m_per_dav2": slope,
        "intercept_m": intercept,
    }

    # RANSAC diagnostics.
    if name == "ransac":
        inlier_mask = getattr(model, "inlier_mask_", None)

        if inlier_mask is not None:
            inlier_count = int(
                np.count_nonzero(inlier_mask)
            )
            extra["ransac_inlier_count"] = inlier_count
            extra["ransac_outlier_count"] = int(
                len(inlier_mask) - inlier_count
            )
            extra["ransac_inlier_fraction"] = float(
                np.mean(inlier_mask)
            )
        else:
            extra["ransac_inlier_count"] = math.nan
            extra["ransac_outlier_count"] = math.nan
            extra["ransac_inlier_fraction"] = math.nan

        extra["ransac_n_trials"] = int(
            getattr(model, "n_trials_", -1)
        )

    return model, extra


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_mask(
    depth: np.ndarray,
    agl: np.ndarray,
    mask: np.ndarray,
    model,
) -> tuple[dict[str, float], int]:

    use = (
        mask
        & np.isfinite(depth)
        & np.isfinite(agl)
        & (agl >= -VALID_AGL_EPS)
    )

    count = int(np.count_nonzero(use))

    if count < 2:
        return {
            "mae_m": math.nan,
            "rmse_m": math.nan,
            "bias_m": math.nan,
            "pearson": math.nan,
            "spearman": math.nan,
        }, count

    x = depth[use].astype(np.float64)
    y = agl[use].astype(np.float64)

    prediction = model.predict(
        x.reshape(-1, 1)
    ).astype(np.float64)

    return metrics(y, prediction), count


# ============================================================================
# One run
# ============================================================================

def run_one(
    tile_id: str,
    truth_path: Path,
    depth_path: Path,
    stage1_dir: Path,
    output_dir: Path,
    placement: str,
    anchor_count: int,
    repeat: int,
    regressor_name: str,
    buffer_radius: int,
    min_eval_pixels: int,
) -> dict:

    depth = read_depth(depth_path)
    agl = read_agl(truth_path)

    if depth.shape != agl.shape:
        raise ValueError(
            f"{tile_id}: DAv2/AGL shape mismatch: "
            f"{depth.shape} vs {agl.shape}"
        )

    valid = valid_mask(depth, agl)

    # IMPORTANT:
    # Reuse EXACT Stage-1 anchors.
    a_path = anchor_file(
        stage1_dir,
        tile_id,
        placement,
        anchor_count,
        repeat,
    )

    anchor_data = load_anchor_set(a_path)

    rows = anchor_data["rows"]
    cols = anchor_data["cols"]

    if len(rows) != anchor_count:
        raise ValueError(
            f"{a_path}: expected {anchor_count} anchors, "
            f"found {len(rows)}"
        )

    if not np.all(valid[rows, cols]):
        raise ValueError(
            f"{tile_id}: Stage-1 anchor contains invalid "
            "metric pixel(s) under the current validity rule."
        )

    # Check that the saved Stage-1 DAv2 values still match the current files.
    dav2_saved = anchor_data["dav2_saved"]
    dav2_current = depth[rows, cols].astype(np.float64)

    if not np.allclose(
        dav2_saved,
        dav2_current,
        rtol=1e-6,
        atol=1e-6,
    ):
        max_diff = float(
            np.max(
                np.abs(dav2_saved - dav2_current)
            )
        )
        raise ValueError(
            f"{tile_id}: Stage-1 anchor DAv2 values do not match "
            f"current depth file. Max difference={max_diff}"
        )

    model, fit_info = fit_regression(
        regressor_name,
        depth,
        anchor_data,
    )

    # ------------------------------------------------------------
    # Evaluation mask 1: every valid non-anchor pixel.
    # ------------------------------------------------------------

    anchor_mask = np.zeros(
        valid.shape,
        dtype=bool,
    )

    anchor_mask[rows, cols] = True

    all_mask = valid & ~anchor_mask

    all_metrics, all_count = evaluate_mask(
        depth,
        agl,
        all_mask,
        model,
    )

    # ------------------------------------------------------------
    # Evaluation mask 2: spatial holdout outside Stage-1 buffer.
    # ------------------------------------------------------------

    exclusion = make_exclusion_mask(
        valid.shape,
        rows,
        cols,
        buffer_radius,
    )

    strict_mask = valid & ~exclusion

    strict_metrics, strict_count = evaluate_mask(
        depth,
        agl,
        strict_mask,
        model,
    )

    if all_count < min_eval_pixels:
        raise ValueError(
            f"{tile_id}: only {all_count} all-pixel evaluation samples."
        )

    if strict_count < min_eval_pixels:
        raise ValueError(
            f"{tile_id}: only {strict_count} strict evaluation samples."
        )

    region = (
        "JAX"
        if tile_id.startswith("JAX_")
        else "OMA"
        if tile_id.startswith("OMA_")
        else "UNKNOWN"
    )

    result = {
        "tile_id": tile_id,
        "region": region,
        "placement": placement,
        "anchor_count": anchor_count,
        "repeat": repeat,
        "regressor": regressor_name,

        "valid_pixels": int(np.count_nonzero(valid)),
        "eval_pixels_all": all_count,
        "eval_pixels_strict": strict_count,
        "strict_buffer_radius_px": buffer_radius,

        "anchor_agl_min_m": float(np.min(anchor_data["agl"])),
        "anchor_agl_max_m": float(np.max(anchor_data["agl"])),
        "anchor_agl_range_m": float(
            np.ptp(anchor_data["agl"])
        ),
        "anchor_dav2_range": float(
            np.ptp(dav2_current)
        ),

        **fit_info,
    }

    for key, value in all_metrics.items():
        result[f"all_{key}"] = value

    for key, value in strict_metrics.items():
        result[f"strict_{key}"] = value

    result["stage1_anchor_file"] = str(a_path)
    result["truth_path"] = str(truth_path)
    result["depth_path"] = str(depth_path)

    return result


# ============================================================================
# Aggregation
# ============================================================================

METRICS = [
    "all_mae_m",
    "all_rmse_m",
    "all_bias_m",
    "all_pearson",
    "all_spearman",

    "strict_mae_m",
    "strict_rmse_m",
    "strict_bias_m",
    "strict_pearson",
    "strict_spearman",
]


def summarize(group: pd.DataFrame) -> dict:
    output = {}

    for metric in METRICS:
        values = pd.to_numeric(
            group[metric],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_median"] = float(values.median())
        output[f"{metric}_std"] = float(values.std(ddof=0))
        output[f"{metric}_p05"] = float(values.quantile(0.05))
        output[f"{metric}_p25"] = float(values.quantile(0.25))
        output[f"{metric}_p75"] = float(values.quantile(0.75))
        output[f"{metric}_p95"] = float(values.quantile(0.95))
        output[f"{metric}_min"] = float(values.min())
        output[f"{metric}_max"] = float(values.max())

    return output


def create_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = results.groupby(
        [
            "regressor",
            "placement",
            "anchor_count",
        ],
        sort=True,
    )

    for (
        regressor,
        placement,
        anchor_count,
    ), group in grouped:

        row = {
            "regressor": regressor,
            "placement": placement,
            "anchor_count": int(anchor_count),
            "runs": len(group),
            "tiles": group["tile_id"].nunique(),
        }

        row.update(summarize(group))

        row["pearson_lt_0"] = int(
            (group["all_pearson"] < 0).sum()
        )
        row["pearson_lt_0_2"] = int(
            (group["all_pearson"] < 0.2).sum()
        )
        row["rmse_gt_10m"] = int(
            (group["all_rmse_m"] > 10).sum()
        )

        # RANSAC-specific diagnostics.
        if "ransac_inlier_fraction" in group:
            vals = pd.to_numeric(
                group["ransac_inlier_fraction"],
                errors="coerce",
            ).dropna()

            if len(vals):
                row["ransac_inlier_fraction_mean"] = float(
                    vals.mean()
                )
                row["ransac_inlier_fraction_median"] = float(
                    vals.median()
                )

        rows.append(row)

    return pd.DataFrame(rows)


def create_region_distributions(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = results.groupby(
        [
            "regressor",
            "placement",
            "anchor_count",
            "region",
        ],
        sort=True,
    )

    for (
        regressor,
        placement,
        anchor_count,
        region,
    ), group in grouped:

        row = {
            "regressor": regressor,
            "placement": placement,
            "anchor_count": int(anchor_count),
            "region": region,
            "runs": len(group),
            "tiles": group["tile_id"].nunique(),
        }

        row.update(summarize(group))
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Paired deltas versus OLS
# ============================================================================

def create_paired_deltas(
    results: pd.DataFrame,
) -> pd.DataFrame:

    key = [
        "tile_id",
        "placement",
        "anchor_count",
        "repeat",
    ]

    ols = (
        results[
            results["regressor"] == "ols"
        ][
            key
            + [
                "all_mae_m",
                "all_rmse_m",
                "all_pearson",
                "all_spearman",
                "strict_mae_m",
                "strict_rmse_m",
                "strict_pearson",
                "strict_spearman",
            ]
        ]
        .copy()
        .rename(
            columns={
                "all_mae_m": "ols_all_mae_m",
                "all_rmse_m": "ols_all_rmse_m",
                "all_pearson": "ols_all_pearson",
                "all_spearman": "ols_all_spearman",
                "strict_mae_m": "ols_strict_mae_m",
                "strict_rmse_m": "ols_strict_rmse_m",
                "strict_pearson": "ols_strict_pearson",
                "strict_spearman": "ols_strict_spearman",
            }
        )
    )

    robust = results[
        results["regressor"].isin(
            ["huber", "ransac"]
        )
    ][
        key
        + [
            "regressor",
            "all_mae_m",
            "all_rmse_m",
            "all_pearson",
            "all_spearman",
            "strict_mae_m",
            "strict_rmse_m",
            "strict_pearson",
            "strict_spearman",
        ]
    ].copy()

    merged = robust.merge(
        ols,
        on=key,
        how="inner",
        validate="many_to_one",
    )

    # Positive delta = improvement for errors.
    merged["delta_all_mae_m"] = (
        merged["ols_all_mae_m"]
        - merged["all_mae_m"]
    )

    merged["delta_all_rmse_m"] = (
        merged["ols_all_rmse_m"]
        - merged["all_rmse_m"]
    )

    merged["delta_strict_mae_m"] = (
        merged["ols_strict_mae_m"]
        - merged["strict_mae_m"]
    )

    merged["delta_strict_rmse_m"] = (
        merged["ols_strict_rmse_m"]
        - merged["strict_rmse_m"]
    )

    # Positive delta = improvement for correlation.
    merged["delta_all_pearson"] = (
        merged["all_pearson"]
        - merged["ols_all_pearson"]
    )

    merged["delta_all_spearman"] = (
        merged["all_spearman"]
        - merged["ols_all_spearman"]
    )

    merged["delta_strict_pearson"] = (
        merged["strict_pearson"]
        - merged["ols_strict_pearson"]
    )

    merged["delta_strict_spearman"] = (
        merged["strict_spearman"]
        - merged["ols_strict_spearman"]
    )

    return merged


# ============================================================================
# Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 regression robustness experiment "
            "for sparse metric anchors."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=DEFAULT_DEPTH_DIR,
    )

    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=DEFAULT_TRUTH_DIR,
    )

    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=DEFAULT_STAGE1_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=DEFAULT_COUNTS,
        help="Stage-2 anchor counts. Default: 10 20",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["random", "grid"],
        default=DEFAULT_STRATEGIES,
    )

    parser.add_argument(
        "--regressors",
        nargs="+",
        choices=["ols", "huber", "ransac"],
        default=DEFAULT_REGRESSORS,
    )

    parser.add_argument(
        "--buffer-radius",
        type=int,
        default=DEFAULT_BUFFER_RADIUS,
    )

    parser.add_argument(
        "--min-eval-pixels",
        type=int,
        default=DEFAULT_MIN_EVAL_PIXELS,
    )

    parser.add_argument(
        "--tiles",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test tile limit; omit for the full study.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    args = parse_args()

    if any(n <= 0 for n in args.anchors):
        raise ValueError(
            "All anchor counts must be positive."
        )

    if args.buffer_radius < 0:
        raise ValueError(
            "Buffer radius cannot be negative."
        )

    stage1_anchor_dir = (
        args.stage1_dir.resolve()
        / "anchors"
    )

    if not stage1_anchor_dir.exists():
        raise FileNotFoundError(
            f"Stage-1 anchor directory not found:\n"
            f"{stage1_anchor_dir}\n"
            "Run Stage 1 first."
        )

    output_dir = args.output_dir.resolve()

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        args.manifest.resolve()
    )

    if "tile_id" not in manifest.columns:
        raise ValueError(
            "Manifest must contain tile_id."
        )

    tile_ids = (
        manifest["tile_id"]
        .astype(str)
        .tolist()
    )

    if args.tiles:
        requested = set(args.tiles)

        missing = sorted(
            requested - set(tile_ids)
        )

        if missing:
            raise ValueError(
                "Requested tiles are missing from "
                f"manifest: {missing}"
            )

        tile_ids = [
            tile_id
            for tile_id in tile_ids
            if tile_id in requested
        ]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        tile_ids = tile_ids[:args.limit]

    # ------------------------------------------------------------
    # Build a concrete job list before running anything.
    # This catches missing Stage-1 anchor files early.
    # ------------------------------------------------------------

    jobs = []

    for tile_id in tile_ids:
        for placement in args.strategies:
            repeats = (
                20
                if placement == "random"
                else 1
            )

            for anchor_count in args.anchors:

                for repeat in range(repeats):

                    a_path = anchor_file(
                        args.stage1_dir.resolve(),
                        tile_id,
                        placement,
                        anchor_count,
                        repeat,
                    )

                    if not a_path.exists():
                        raise FileNotFoundError(
                            "Stage-1 anchor file missing for "
                            "a required Stage-2 configuration:\n"
                            f"{a_path}\n\n"
                            "Stage 2 intentionally reuses exact "
                            "Stage-1 anchors, so do not regenerate them."
                        )

                    for regressor in args.regressors:
                        jobs.append(
                            (
                                tile_id,
                                placement,
                                anchor_count,
                                repeat,
                                regressor,
                            )
                        )

    config = {
        "experiment": "sparse_anchor_regression_robustness",
        "stage": 2,
        "scientific_question": (
            "Does the regression estimator materially affect "
            "sparse-anchor metric calibration?"
        ),
        "stage1_anchor_reuse": True,
        "anchor_source": str(
            stage1_anchor_dir
        ),
        "tiles": tile_ids,
        "anchor_counts": args.anchors,
        "placements": args.strategies,
        "regressors": args.regressors,
        "random_repetitions": 20,
        "buffer_radius_px": args.buffer_radius,
        "evaluation_masks": [
            "all valid non-anchor pixels",
            "strict spatial holdout outside anchor buffer",
        ],
        "regressor_parameters": {
            "ols": {
                "estimator": "sklearn LinearRegression",
            },
            "huber": {
                "estimator": "sklearn HuberRegressor",
                "epsilon": 1.35,
                "alpha": 1e-4,
                "max_iter": 2000,
                "tol": 1e-8,
            },
            "ransac": {
                "estimator": (
                    "sklearn RANSACRegressor("
                    "estimator=LinearRegression)"
                ),
                "min_samples": 2,
                "residual_threshold": None,
                "max_trials": 100,
                "stop_probability": 0.999,
                "loss": "absolute_error",
                "random_state": 0,
            },
        },
        "fairness_rule": (
            "For every paired comparison, the anchor coordinates "
            "and evaluation masks are identical across regressors."
        ),
        "expected_jobs": len(jobs),
    }

    (
        output_dir / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "DepthWizard2 — Stage 2 "
        "Sparse-Anchor Regression Robustness"
    )
    print("=" * 78)

    print(
        f"Tiles:          {len(tile_ids)}"
    )

    print(
        f"Counts:         {args.anchors}"
    )

    print(
        f"Placements:     {args.strategies}"
    )

    print(
        f"Regressors:     {args.regressors}"
    )

    print(
        "Random repeats: 20"
    )

    print(
        f"Strict buffer:  {args.buffer_radius}px"
    )

    print(
        f"Total runs:     {len(jobs)}"
    )

    print()
    print(
        "IMPORTANT: exact Stage-1 anchor coordinates are reused."
    )
    print(
        "Only the regression estimator changes."
    )
    print("=" * 78)
    print()

    results = []

    total = len(jobs)

    for index, (
        tile_id,
        placement,
        anchor_count,
        repeat,
        regressor,
    ) in enumerate(jobs, start=1):

        truth_path = (
            args.truth_dir.resolve()
            / f"{tile_id}_AGL.tif"
        )

        depth_path = (
            args.depth_dir.resolve()
            / f"{tile_id}_depth.npy"
        )

        print(
            f"{index:5d}/{total} "
            f"{tile_id:14s} "
            f"{placement:6s} "
            f"n={anchor_count:2d} "
            f"rep={repeat:02d} "
            f"{regressor:7s}",
            end=" ... ",
            flush=True,
        )

        try:
            row = run_one(
                tile_id=tile_id,
                truth_path=truth_path,
                depth_path=depth_path,
                stage1_dir=args.stage1_dir.resolve(),
                output_dir=output_dir,
                placement=placement,
                anchor_count=anchor_count,
                repeat=repeat,
                regressor_name=regressor,
                buffer_radius=args.buffer_radius,
                min_eval_pixels=args.min_eval_pixels,
            )

            results.append(row)

            print(
                f"strict MAE="
                f"{row['strict_mae_m']:.3f}m "
                f"RMSE="
                f"{row['strict_rmse_m']:.3f}m "
                f"P="
                f"{row['strict_pearson']:.4f}"
            )

        except Exception as exc:
            print("FAILED")
            raise RuntimeError(
                f"Stage-2 run failed:\n"
                f"tile={tile_id}\n"
                f"placement={placement}\n"
                f"anchors={anchor_count}\n"
                f"repeat={repeat}\n"
                f"regressor={regressor}\n"
                f"error={exc}"
            ) from exc

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "No Stage-2 results were produced."
        )

    results_df = results_df.sort_values(
        [
            "regressor",
            "placement",
            "anchor_count",
            "tile_id",
            "repeat",
        ]
    ).reset_index(drop=True)

    results_path = (
        output_dir / "results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary_df = create_summary(
        results_df
    )

    summary_df = summary_df.sort_values(
        [
            "regressor",
            "placement",
            "anchor_count",
        ]
    ).reset_index(drop=True)

    summary_path = (
        output_dir / "summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    distributions_df = (
        create_region_distributions(
            results_df
        )
    )

    distributions_path = (
        output_dir / "distributions.csv"
    )

    distributions_df.to_csv(
        distributions_path,
        index=False,
    )

    paired_df = create_paired_deltas(
        results_df
    )

    paired_path = (
        output_dir / "paired_deltas.csv"
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Human-readable report.
    # ------------------------------------------------------------

    report_lines = [
        "DepthWizard2 — Stage 2 Sparse-Anchor Regression Study",
        "=" * 68,
        "",
        "Question:",
        "Does the regression estimator materially affect",
        "sparse-anchor metric calibration?",
        "",
        "Design:",
        "- exact Stage-1 anchor coordinates reused",
        "- exact Stage-1 repetitions reused",
        "- identical evaluation masks across regressors",
        "- only OLS / Huber / RANSAC changed",
        "",
        "PRIMARY COMPARISON: strict spatial holdout",
        "",
    ]

    primary_columns = [
        "regressor",
        "placement",
        "anchor_count",
        "runs",
        "tiles",
        "strict_mae_m_mean",
        "strict_rmse_m_mean",
        "strict_pearson_mean",
        "strict_spearman_mean",
        "strict_pearson_p05",
        "strict_pearson_p95",
        "rmse_gt_10m",
    ]

    report_df = summary_df[
        [
            c
            for c in primary_columns
            if c in summary_df.columns
        ]
    ]

    report_lines.append(
        report_df.to_string(
            index=False
        )
    )

    report_lines.extend(
        [
            "",
            "FILES",
            f"results.csv:       {results_path}",
            f"summary.csv:       {summary_path}",
            f"distributions.csv: {distributions_path}",
            f"paired_deltas.csv: {paired_path}",
        ]
    )

    report_path = (
        output_dir / "REPORT.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("STAGE 2 COMPLETE")
    print("=" * 78)

    print(
        f"Results:       {results_path}"
    )
    print(
        f"Summary:       {summary_path}"
    )
    print(
        f"Distributions: {distributions_path}"
    )
    print(
        f"Paired deltas: {paired_path}"
    )
    print(
        f"Report:        {report_path}"
    )

    print()
    print(report_df.to_string(index=False))

    print("=" * 78)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)
