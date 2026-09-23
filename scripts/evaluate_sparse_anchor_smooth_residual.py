#!/usr/bin/env python3
"""
DepthWizard2 — Stage 4B: Regularized Smooth Spatial Residual

Scientific question
-------------------
Stage 4 showed:

    - global sparse-anchor affine calibration is a strong baseline
    - a linear spatial residual gives a modest but repeatable improvement
    - a quadratic polynomial residual overfits / generalizes worse
    - residuals are strongly spatially correlated

Stage 4B therefore tests a LOW-COMPLEXITY SMOOTH residual field.

Baseline:
    AGL_base = a * DAv2 + b

Residual:
    r(x,y) = AGL_truth - AGL_base

Final prediction:
    AGL_final = AGL_base + RBF(x,y)

The smooth residual is fitted ONLY on training spatial blocks and evaluated
on a held-out spatial block.

This remains an OFFLINE ORACLE LEARNABILITY experiment because full DFC2019
AGL truth is used to construct the residual targets for model fitting.

This is deliberately NOT a deployment claim.

Models
------
Three smoothing strengths for a thin-plate-spline RBF interpolator:

    rbf_strong
    rbf_medium
    rbf_weak

The RBF model operates on normalized coordinates and standardized residuals.
This keeps the smoothing parameter numerically stable across tiles.

RBF settings are fixed:
    kernel      = thin_plate_spline
    degree      = 1
    neighbors   = 128

Smoothing values on standardized residuals:
    strong = 1.0
    medium = 0.1
    weak = 0.01

A 4-fold spatial CV is used:
    Q1 | Q2
    ---+---
    Q3 | Q4

Training residual pixels are spatially stratified so the sample does not
collapse into dense regions.

Outputs
-------
data/dfc2019/experiments/sparse_anchor_smooth_residual/
    config.json
    preflight.csv
    results.csv
    summary.csv
    tile_summary.csv
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
from scipy.interpolate import RBFInterpolator
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


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
    / "data/dfc2019/experiments/sparse_anchor_smooth_residual"
)

DEFAULT_ANCHOR_COUNT = 20
DEFAULT_PLACEMENT = "grid"
DEFAULT_BUFFER_RADIUS = 16

# RBF training samples per fold.
DEFAULT_TRAIN_SAMPLES = 5000

# Local RBF neighborhood. This avoids the cubic cost of a global dense RBF.
DEFAULT_NEIGHBORS = 128

DEFAULT_SEED = 20260915

# Smoothing applied after target standardization.
DEFAULT_SMOOTHING = {
    "rbf_strong": 1.0,
    "rbf_medium": 0.1,
    "rbf_weak": 0.01,
}

VALID_AGL_EPS = 1e-6


# ============================================================================
# IO
# ============================================================================

def read_depth(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(path, allow_pickle=False)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D DAv2 array, got {arr.shape}: {path}"
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


# ============================================================================
# METRICS
# ============================================================================

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
    pearson, spearman = safe_correlations(truth, prediction)

    return {
        "mae_m": float(np.mean(np.abs(residual))),
        "rmse_m": float(np.sqrt(np.mean(residual ** 2))),
        "bias_m": float(np.mean(residual)),
        "pearson": pearson,
        "spearman": spearman,
    }


# ============================================================================
# STAGE-1 ANCHORS
# ============================================================================

def stage1_anchor_path(
    stage1_dir: Path,
    tile_id: str,
    placement: str,
    anchor_count: int,
) -> Path:
    return (
        stage1_dir
        / "anchors"
        / (
            f"{tile_id}_{placement}_"
            f"{anchor_count}_00.json"
        )
    )


def load_stage1_anchors(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Stage-1 anchor file is missing:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    anchors = payload.get("anchors")
    if not anchors:
        raise ValueError(
            f"Malformed/empty Stage-1 anchor file: {path}"
        )

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

    coords = np.column_stack([rows, cols])

    if len(np.unique(coords, axis=0)) != len(anchors):
        raise ValueError(
            f"Duplicate anchors in {path}"
        )

    if not np.all(np.isfinite(agl)):
        raise ValueError(
            f"Non-finite anchor AGL in {path}"
        )

    if not np.all(np.isfinite(dav2_saved)):
        raise ValueError(
            f"Non-finite anchor DAv2 in {path}"
        )

    return {
        "rows": rows,
        "cols": cols,
        "agl": agl,
        "dav2_saved": dav2_saved,
        "payload": payload,
    }


# ============================================================================
# EXCLUSION / FOLDS
# ============================================================================

def make_exclusion_mask(
    shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius < 0:
        raise ValueError("Buffer radius cannot be negative.")

    height, width = shape
    mask = np.zeros(shape, dtype=bool)

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

        mask[rr[inside], cc[inside]] = True

    return mask


def spatial_fold_map(
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape

    rows, cols = np.indices(
        shape,
        dtype=np.int32,
    )

    top = rows < (height / 2.0)
    left = cols < (width / 2.0)

    folds = np.empty(
        shape,
        dtype=np.int8,
    )

    folds[top & left] = 0
    folds[top & ~left] = 1
    folds[~top & left] = 2
    folds[~top & ~left] = 3

    return folds


def fold_name(fold: int) -> str:
    return {
        0: "Q1_TL",
        1: "Q2_TR",
        2: "Q3_BL",
        3: "Q4_BR",
    }[int(fold)]


# ============================================================================
# COORDINATES / SAMPLING
# ============================================================================

def normalized_coordinates(
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = (
        cols.astype(np.float64)
        / max(width - 1, 1)
        * 2.0
        - 1.0
    )

    y = (
        rows.astype(np.float64)
        / max(height - 1, 1)
        * 2.0
        - 1.0
    )

    return x, y


def spatial_stratified_sample(
    rows: np.ndarray,
    cols: np.ndarray,
    max_samples: int,
    seed: int,
    image_height: int,
    image_width: int,
    grid_size: int = 16,
) -> np.ndarray:
    """
    Return indices into rows/cols, with broad spatial coverage.
    """
    if len(rows) <= max_samples:
        return np.arange(
            len(rows),
            dtype=np.int64,
        )

    rng = np.random.default_rng(seed)

    rb = np.clip(
        rows * grid_size // max(image_height, 1),
        0,
        grid_size - 1,
    )

    cb = np.clip(
        cols * grid_size // max(image_width, 1),
        0,
        grid_size - 1,
    )

    cell = (
        rb * grid_size + cb
    )

    unique_cells = np.unique(cell)

    # Start with roughly equal representation from cells.
    per_cell = max(
        1,
        max_samples // len(unique_cells),
    )

    chosen = []

    for cid in unique_cells:
        candidates = np.flatnonzero(
            cell == cid
        )

        take = min(
            len(candidates),
            per_cell,
        )

        if len(candidates) > take:
            candidates = rng.choice(
                candidates,
                size=take,
                replace=False,
            )

        chosen.extend(
            candidates.tolist()
        )

    chosen = np.asarray(
        chosen,
        dtype=np.int64,
    )

    if len(chosen) > max_samples:
        chosen = rng.choice(
            chosen,
            size=max_samples,
            replace=False,
        )

    return np.sort(chosen)


# ============================================================================
# RBF
# ============================================================================

def fit_rbf(
    rows: np.ndarray,
    cols: np.ndarray,
    residual: np.ndarray,
    height: int,
    width: int,
    smoothing: float,
    neighbors: int,
) -> tuple[RBFInterpolator, float, float]:
    x, y = normalized_coordinates(
        rows,
        cols,
        height,
        width,
    )

    points = np.column_stack(
        [x, y]
    )

    residual = residual.astype(
        np.float64,
        copy=False,
    )

    mean = float(
        np.mean(residual)
    )

    scale = float(
        np.std(residual)
    )

    if not np.isfinite(mean) or not np.isfinite(scale):
        raise ValueError(
            "Residual mean/std is non-finite."
        )

    if scale < 1e-8:
        # A practically constant residual needs no spatial RBF.
        scale = 1.0

    standardized = (
        residual - mean
    ) / scale

    rbf = RBFInterpolator(
        points,
        standardized,
        neighbors=neighbors,
        smoothing=smoothing,
        kernel="thin_plate_spline",
        degree=1,
    )

    return rbf, mean, scale


def predict_rbf(
    rbf: RBFInterpolator,
    mean: float,
    scale: float,
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    x, y = normalized_coordinates(
        rows,
        cols,
        height,
        width,
    )

    points = np.column_stack(
        [x, y]
    )

    standardized_prediction = (
        rbf(points)
        .reshape(-1)
        .astype(np.float64)
    )

    return (
        standardized_prediction
        * scale
        + mean
    )


# ============================================================================
# ONE TILE
# ============================================================================

def run_tile(
    tile_id: str,
    depth_path: Path,
    truth_path: Path,
    stage1_dir: Path,
    output_dir: Path,
    anchor_count: int,
    placement: str,
    buffer_radius: int,
    train_samples: int,
    neighbors: int,
    seed: int,
    smoothing_map: dict[str, float],
) -> tuple[list[dict], dict]:
    depth = read_depth(
        depth_path
    )

    agl = read_agl(
        truth_path
    )

    if depth.shape != agl.shape:
        raise ValueError(
            f"{tile_id}: shape mismatch "
            f"{depth.shape} vs {agl.shape}"
        )

    height, width = depth.shape

    valid = valid_mask(
        depth,
        agl,
    )

    anchor_path = stage1_anchor_path(
        stage1_dir,
        tile_id,
        placement,
        anchor_count,
    )

    anchors = load_stage1_anchors(
        anchor_path
    )

    anchor_rows = anchors["rows"]
    anchor_cols = anchors["cols"]
    anchor_agl = anchors["agl"]

    if len(anchor_rows) != anchor_count:
        raise ValueError(
            f"{tile_id}: expected {anchor_count} anchors, "
            f"got {len(anchor_rows)}"
        )

    if not np.all(
        valid[
            anchor_rows,
            anchor_cols
        ]
    ):
        raise ValueError(
            f"{tile_id}: invalid Stage-1 anchor."
        )

    current_anchor_dav2 = depth[
        anchor_rows,
        anchor_cols
    ].astype(np.float64)

    if not np.allclose(
        anchors["dav2_saved"],
        current_anchor_dav2,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            f"{tile_id}: Stage-1 DAv2 anchor values changed."
        )

    # ------------------------------------------------------------
    # Global sparse-anchor baseline.
    # ------------------------------------------------------------

    global_model = LinearRegression()

    global_model.fit(
        current_anchor_dav2.reshape(-1, 1),
        anchor_agl,
    )

    global_pred = (
        global_model.predict(
            depth.astype(np.float64).reshape(-1, 1)
        )
        .reshape(depth.shape)
        .astype(np.float64)
    )

    baseline_slope = float(
        np.asarray(
            global_model.coef_
        ).reshape(-1)[0]
    )

    baseline_intercept = float(
        global_model.intercept_
    )

    # ------------------------------------------------------------
    # Exact strict evaluation mask.
    # ------------------------------------------------------------

    anchor_mask = np.zeros(
        valid.shape,
        dtype=bool,
    )

    anchor_mask[
        anchor_rows,
        anchor_cols
    ] = True

    exclusion = make_exclusion_mask(
        valid.shape,
        anchor_rows,
        anchor_cols,
        buffer_radius,
    )

    eligible = (
        valid
        & ~anchor_mask
        & ~exclusion
    )

    eligible_rows, eligible_cols = np.where(
        eligible
    )

    if len(eligible_rows) < 1000:
        raise ValueError(
            f"{tile_id}: too few eligible pixels."
        )

    # ------------------------------------------------------------
    # Global residual truth.
    # ------------------------------------------------------------

    residual = (
        agl.astype(np.float64)
        - global_pred
    )

    residual[
        ~valid
    ] = np.nan

    folds = spatial_fold_map(
        depth.shape
    )

    region_name = (
        "JAX"
        if tile_id.startswith("JAX_")
        else "OMA"
        if tile_id.startswith("OMA_")
        else "UNKNOWN"
    )

    baseline_rows = []

    for fold in range(4):
        test_mask = (
            eligible
            & (folds == fold)
        )

        test_rows, test_cols = np.where(
            test_mask
        )

        truth_test = agl[
            test_rows,
            test_cols
        ].astype(np.float64)

        base_test = global_pred[
            test_rows,
            test_cols
        ].astype(np.float64)

        fold_baseline = metrics(
            truth_test,
            base_test,
        )

        baseline_rows.append(
            (
                fold,
                len(test_rows),
                fold_baseline,
                truth_test,
                base_test,
                test_rows,
                test_cols,
            )
        )

    results = []

    # ------------------------------------------------------------
    # RBF blocked CV.
    # ------------------------------------------------------------

    for model_name, smoothing in smoothing_map.items():

        fold_results = []

        for (
            fold,
            test_count,
            fold_baseline,
            truth_test,
            base_test,
            test_rows,
            test_cols,
        ) in baseline_rows:

            train_mask = (
                eligible
                & (folds != fold)
            )

            train_rows, train_cols = np.where(
                train_mask
            )

            if len(train_rows) < 1000:
                raise ValueError(
                    f"{tile_id} fold {fold}: insufficient "
                    f"training pixels ({len(train_rows)})."
                )

            # Deterministic per tile/model/fold seed.
            local_seed = (
                seed
                + (
                    sum(
                        bytearray(
                            f"{tile_id}:{model_name}:{fold}".encode()
                        )
                    )
                )
            )

            sample_indices = spatial_stratified_sample(
                train_rows,
                train_cols,
                max_samples=train_samples,
                seed=local_seed,
                image_height=height,
                image_width=width,
            )

            sample_rows = train_rows[
                sample_indices
            ]

            sample_cols = train_cols[
                sample_indices
            ]

            sample_residual = residual[
                sample_rows,
                sample_cols
            ].astype(np.float64)

            if not np.all(
                np.isfinite(
                    sample_residual
                )
            ):
                raise ValueError(
                    f"{tile_id} fold {fold}: non-finite residual "
                    "in RBF training sample."
                )

            rbf, residual_mean, residual_scale = fit_rbf(
                rows=sample_rows,
                cols=sample_cols,
                residual=sample_residual,
                height=height,
                width=width,
                smoothing=smoothing,
                neighbors=min(
                    neighbors,
                    len(sample_rows),
                ),
            )

            predicted_residual = predict_rbf(
                rbf=rbf,
                mean=residual_mean,
                scale=residual_scale,
                rows=test_rows,
                cols=test_cols,
                height=height,
                width=width,
            )

            final_prediction = (
                base_test
                + predicted_residual
            )

            result_metrics = metrics(
                truth_test,
                final_prediction,
            )

            row = {
                "tile_id": tile_id,
                "region": region_name,
                "model": model_name,
                "kernel": "thin_plate_spline",
                "smoothing": smoothing,
                "neighbors": min(
                    neighbors,
                    len(sample_rows),
                ),
                "fold": fold,
                "fold_name": fold_name(fold),
                "evaluation": "strict_blocked_cv",
                "anchor_count": anchor_count,
                "placement": placement,
                "train_samples": len(sample_rows),
                "eval_pixels": test_count,
                "baseline_test_mae_m": (
                    fold_baseline["mae_m"]
                ),
                "baseline_test_rmse_m": (
                    fold_baseline["rmse_m"]
                ),
                "baseline_test_pearson": (
                    fold_baseline["pearson"]
                ),
                "baseline_test_spearman": (
                    fold_baseline["spearman"]
                ),
                **result_metrics,
            }

            row["delta_mae_m"] = (
                fold_baseline["mae_m"]
                - result_metrics["mae_m"]
            )

            row["delta_rmse_m"] = (
                fold_baseline["rmse_m"]
                - result_metrics["rmse_m"]
            )

            row["delta_pearson"] = (
                result_metrics["pearson"]
                - fold_baseline["pearson"]
            )

            row["delta_spearman"] = (
                result_metrics["spearman"]
                - fold_baseline["spearman"]
            )

            fold_results.append(row)
            results.append(row)

        # Save a compact per-tile/model diagnostic.
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------
    # Save fit metadata.
    # ------------------------------------------------------------

    fit_metadata = {
        "tile_id": tile_id,
        "region": region_name,
        "anchor_count": anchor_count,
        "placement": placement,
        "global_slope": baseline_slope,
        "global_intercept": baseline_intercept,
        "eligible_pixels": int(
            len(eligible_rows)
        ),
        "residual_mean": float(
            np.mean(
                residual[
                    eligible
                ]
            )
        ),
        "residual_std": float(
            np.std(
                residual[
                    eligible
                ]
            )
        ),
    }

    return results, fit_metadata


# ============================================================================
# AGGREGATION
# ============================================================================

METRICS = [
    "mae_m",
    "rmse_m",
    "pearson",
    "spearman",
    "delta_mae_m",
    "delta_rmse_m",
    "delta_pearson",
    "delta_spearman",
]


def stat_block(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return {}

    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, group in results.groupby(
        "model",
        sort=True,
    ):
        row = {
            "model": model,
            "fold_runs": len(group),
            "tiles": group[
                "tile_id"
            ].nunique(),
        }

        for metric in METRICS:
            for name, value in stat_block(
                group[metric]
            ).items():
                row[
                    f"{metric}_{name}"
                ] = value

        row[
            "improved_mae_folds"
        ] = int(
            (
                group["delta_mae_m"]
                > 0
            ).sum()
        )

        row[
            "improved_rmse_folds"
        ] = int(
            (
                group["delta_rmse_m"]
                > 0
            ).sum()
        )

        row[
            "improved_pearson_folds"
        ] = int(
            (
                group["delta_pearson"]
                > 0
            ).sum()
        )

        row[
            "improved_spearman_folds"
        ] = int(
            (
                group["delta_spearman"]
                > 0
            ).sum()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def create_tile_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        model,
        tile_id,
    ), group in results.groupby(
        [
            "model",
            "tile_id",
        ],
        sort=True,
    ):
        row = {
            "model": model,
            "tile_id": tile_id,
            "region": group[
                "region"
            ].iloc[0],
        }

        for metric in [
            "delta_mae_m",
            "delta_rmse_m",
            "delta_pearson",
            "delta_spearman",
        ]:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            row[
                f"{metric}_mean"
            ] = (
                float(values.mean())
                if len(values)
                else math.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def create_paired_summary(
    tile_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for model, group in tile_summary.groupby(
        "model",
        sort=True,
    ):
        row = {
            "model": model,
            "tiles": group[
                "tile_id"
            ].nunique(),
        }

        for metric in [
            "delta_mae_m_mean",
            "delta_rmse_m_mean",
            "delta_pearson_mean",
            "delta_spearman_mean",
        ]:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            row[
                f"{metric}_mean"
            ] = float(values.mean())

            row[
                f"{metric}_median"
            ] = float(values.median())

            row[
                f"{metric}_std"
            ] = float(values.std(ddof=0))

            row[
                f"{metric}_p05"
            ] = float(values.quantile(0.05))

            row[
                f"{metric}_p95"
            ] = float(values.quantile(0.95))

            row[
                f"{metric}_positive_tiles"
            ] = int(
                (
                    values > 0
                ).sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4B: regularized smooth spatial residual "
            "learnability diagnostic."
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
        default=DEFAULT_ANCHOR_COUNT,
    )

    parser.add_argument(
        "--placement",
        choices=["grid"],
        default=DEFAULT_PLACEMENT,
    )

    parser.add_argument(
        "--buffer-radius",
        type=int,
        default=DEFAULT_BUFFER_RADIUS,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=DEFAULT_NEIGHBORS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
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
        help="Smoke-test tile limit.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    args = parse_args()

    if args.anchors != 20:
        raise ValueError(
            "Stage 4B is intentionally locked to 20 grid anchors."
        )

    if args.placement != "grid":
        raise ValueError(
            "Stage 4B is intentionally locked to grid placement."
        )

    if args.train_samples < 500:
        raise ValueError(
            "Use at least 500 RBF training samples."
        )

    if args.neighbors < 8:
        raise ValueError(
            "Use at least 8 RBF neighbors."
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
            f"{stage1_anchor_dir}"
        )

    output_dir = args.output_dir.resolve()

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(
            output_dir
        )

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
        requested = set(
            args.tiles
        )

        missing = sorted(
            requested - set(tile_ids)
        )

        if missing:
            raise ValueError(
                "Requested tiles missing from manifest:\n"
                + "\n".join(missing)
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

        tile_ids = tile_ids[
            :args.limit
        ]

    if not tile_ids:
        raise ValueError(
            "No tiles selected."
        )

    smoothing_map = dict(
        DEFAULT_SMOOTHING
    )

    # ------------------------------------------------------------
    # Preflight.
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 4B — SMOOTH SPATIAL RESIDUAL")
    print("=" * 78)
    print(
        "OFFLINE ORACLE TEST: full AGL truth supplies residual targets."
    )
    print()
    print(
        f"Tiles:        {len(tile_ids)}"
    )
    print(
        f"Anchors:      {args.anchors} grid"
    )
    print(
        f"Models:       {list(smoothing_map.keys())}"
    )
    print(
        f"Training cap: {args.train_samples}/fold"
    )
    print(
        f"Neighbors:    {args.neighbors}"
    )
    print(
        f"Buffer:       {args.buffer_radius}px"
    )
    print("=" * 78)
    print()

    preflight_rows = []

    for index, tile_id in enumerate(
        tile_ids,
        start=1,
    ):
        depth_path = (
            args.depth_dir.resolve()
            / f"{tile_id}_depth.npy"
        )

        truth_path = (
            args.truth_dir.resolve()
            / f"{tile_id}_AGL.tif"
        )

        depth = read_depth(
            depth_path
        )

        agl = read_agl(
            truth_path
        )

        if depth.shape != agl.shape:
            raise ValueError(
                f"{tile_id}: shape mismatch."
            )

        valid = valid_mask(
            depth,
            agl,
        )

        anchor_path = stage1_anchor_path(
            args.stage1_dir.resolve(),
            tile_id,
            args.placement,
            args.anchors,
        )

        anchors = load_stage1_anchors(
            anchor_path
        )

        rows = anchors["rows"]
        cols = anchors["cols"]

        if len(rows) != args.anchors:
            raise ValueError(
                f"{tile_id}: expected {args.anchors} anchors."
            )

        if not np.all(
            valid[rows, cols]
        ):
            raise ValueError(
                f"{tile_id}: invalid Stage-1 anchor."
            )

        current = depth[
            rows,
            cols
        ].astype(np.float64)

        if not np.allclose(
            current,
            anchors["dav2_saved"],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                f"{tile_id}: Stage-1 DAv2 mismatch."
            )

        preflight_rows.append(
            {
                "tile_id": tile_id,
                "valid_pixels": int(
                    np.count_nonzero(valid)
                ),
                "anchor_dav2_range": float(
                    np.ptp(current)
                ),
                "anchor_agl_range": float(
                    np.ptp(
                        anchors["agl"]
                    )
                ),
                "status": "READY",
            }
        )

        print(
            f"{index:3d}/{len(tile_ids)} "
            f"{tile_id:14s} READY"
        )

    preflight_df = pd.DataFrame(
        preflight_rows
    )

    preflight_path = (
        output_dir / "preflight.csv"
    )

    preflight_df.to_csv(
        preflight_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Config.
    # ------------------------------------------------------------

    config = {
        "experiment": (
            "sparse_anchor_smooth_spatial_residual_stage_4B"
        ),
        "scientific_question": (
            "Can a regularized smooth spatial residual improve "
            "the global sparse-anchor metric baseline on unseen "
            "spatial blocks?"
        ),
        "offline_oracle_warning": (
            "Full AGL truth is used to construct residual targets "
            "for fitting. This is not a deployment simulation."
        ),
        "anchor_count": args.anchors,
        "placement": args.placement,
        "buffer_radius_px": args.buffer_radius,
        "spatial_folds": 4,
        "train_samples_per_fold": args.train_samples,
        "neighbors": args.neighbors,
        "kernel": "thin_plate_spline",
        "degree": 1,
        "smoothing_map": smoothing_map,
        "coordinate_normalization": "[-1,1]",
        "residual_standardization": True,
        "tiles": tile_ids,
        "stage1_anchor_reuse": True,
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

    # ------------------------------------------------------------
    # Run.
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("RUNNING STAGE 4B")
    print("=" * 78)
    print()

    all_results = []
    fit_metadata = []

    for index, tile_id in enumerate(
        tile_ids,
        start=1,
    ):
        depth_path = (
            args.depth_dir.resolve()
            / f"{tile_id}_depth.npy"
        )

        truth_path = (
            args.truth_dir.resolve()
            / f"{tile_id}_AGL.tif"
        )

        print(
            f"{index:3d}/{len(tile_ids)} "
            f"{tile_id:14s}",
            end=" ... ",
            flush=True,
        )

        try:
            results, metadata = run_tile(
                tile_id=tile_id,
                depth_path=depth_path,
                truth_path=truth_path,
                stage1_dir=args.stage1_dir.resolve(),
                output_dir=output_dir,
                anchor_count=args.anchors,
                placement=args.placement,
                buffer_radius=args.buffer_radius,
                train_samples=args.train_samples,
                neighbors=args.neighbors,
                seed=args.seed,
                smoothing_map=smoothing_map,
            )

            all_results.extend(
                results
            )
            fit_metadata.append(
                metadata
            )

            pieces = []

            for model_name in smoothing_map:
                vals = [
                    r
                    for r in results
                    if r["model"]
                    == model_name
                ]

                if vals:
                    mean_dr = float(
                        np.mean(
                            [
                                r[
                                    "delta_rmse_m"
                                ]
                                for r in vals
                            ]
                        )
                    )

                    mean_dp = float(
                        np.mean(
                            [
                                r[
                                    "delta_pearson"
                                ]
                                for r in vals
                            ]
                        )
                    )

                    pieces.append(
                        f"{model_name}: "
                        f"ΔRMSE={mean_dr:+.3f}m "
                        f"ΔP={mean_dp:+.4f}"
                    )

            print(
                " | ".join(
                    pieces
                )
            )

        except Exception as exc:
            print("FAILED")
            raise RuntimeError(
                f"Stage 4B failed for {tile_id}: {exc}"
            ) from exc

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:
        raise RuntimeError(
            "No Stage-4B results generated."
        )

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

    summary_path = (
        output_dir / "summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    tile_summary_df = create_tile_summary(
        results_df
    )

    tile_summary_path = (
        output_dir / "tile_summary.csv"
    )

    tile_summary_df.to_csv(
        tile_summary_path,
        index=False,
    )

    paired_df = create_paired_summary(
        tile_summary_df
    )

    paired_path = (
        output_dir / "paired_deltas.csv"
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    fit_df = pd.DataFrame(
        fit_metadata
    )

    fit_path = (
        output_dir / "fit_metadata.csv"
    )

    fit_df.to_csv(
        fit_path,
        index=False,
    )

    # ------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------

    report_lines = [
        "DepthWizard2 — Stage 4B Smooth Spatial Residual",
        "=" * 70,
        "",
        "OFFLINE ORACLE WARNING:",
        "Full DFC2019 AGL truth is used to construct residual targets.",
        "A positive result demonstrates residual learnability,",
        "not deployment feasibility.",
        "",
        "PRIMARY: 4-fold spatial blocked cross-validation",
        "against the global sparse-anchor affine baseline.",
        "",
    ]

    report_lines.append(
        summary_df.to_string(
            index=False
        )
    )

    report_lines.extend(
        [
            "",
            "TILE-LEVEL PAIRED DELTAS",
            "(positive = smooth residual improvement)",
            "",
            paired_df.to_string(
                index=False
            ),
            "",
            "FILES",
            f"preflight.csv:      {preflight_path}",
            f"results.csv:        {results_path}",
            f"summary.csv:        {summary_path}",
            f"tile_summary.csv:   {tile_summary_path}",
            f"paired_deltas.csv:  {paired_path}",
            f"fit_metadata.csv:   {fit_path}",
        ]
    )

    report_path = (
        output_dir / "REPORT.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # FINAL CONSOLE
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 4B COMPLETE")
    print("=" * 78)
    print(
        f"Preflight:      {preflight_path}"
    )
    print(
        f"Results:        {results_path}"
    )
    print(
        f"Summary:        {summary_path}"
    )
    print(
        f"Tile summary:   {tile_summary_path}"
    )
    print(
        f"Paired deltas:  {paired_path}"
    )
    print(
        f"Fit metadata:   {fit_path}"
    )
    print(
        f"Report:         {report_path}"
    )
    print()
    print(
        summary_df.to_string(
            index=False
        )
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)
