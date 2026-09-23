#!/usr/bin/env python3
"""
DepthWizard2 — Stage 3A: Sparse-Anchor Spatial Calibration

Scientific question
-------------------
Stage 1 established that:
    - more anchors (especially 20) are better than 5
    - grid and random placement are preferable to the tested
      spatial+height heuristic
    - sparse affine calibration can reduce absolute error but has a
      correlation/error ceiling

Stage 2 established that:
    - Huber can improve MAE in some aggregate cases
    - Huber does not solve the large-error / correlation ceiling
    - RANSAC is not competitive enough to justify continued focus

Therefore Stage 3A tests the next hypothesis:

    "Is the DAv2 -> metric-height relationship spatially heterogeneous
     within a tile?"

This script compares, using EXACTLY the same Stage-1 grid/20 anchors:

    1. tile-wide affine calibration
           AGL = a * DAv2 + b

    2. 2x2 quadrant-wise affine calibration
           each quadrant gets its own a,b

No new anchors are generated.

Important fairness rules
------------------------
- Exact Stage-1 grid/20 anchor coordinates are reused.
- The same anchor set is used for both models.
- The same strict evaluation mask is used for both models.
- The same DAv2 and AGL inputs are used.
- OLS is the default/primary estimator.
- Each quadrant must have at least two anchors with non-zero DAv2
  spread; otherwise the tile is rejected rather than silently falling
  back to a global model.
- The quadrant model is evaluated on the corresponding quadrant only.
- This stage deliberately avoids fine grids (4x4, 8x8) until 2x2 has
  demonstrated meaningful benefit.

Outputs
-------
data/dfc2019/experiments/sparse_anchor_spatial/
    config.json
    preflight.csv
    results.csv
    summary.csv
    distributions.csv
    paired_deltas.csv
    REPORT.txt
    fits/
        <tile_id>.json

Primary comparison
------------------
strict spatial holdout:
    valid pixels outside the same 16-pixel anchor exclusion buffer
    used in Stage 1 and Stage 2.

If quadrant-wise calibration is not materially better than tile-wide
calibration, we should NOT jump immediately to finer spatial grids.
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
from sklearn.linear_model import LinearRegression


# ============================================================================
# PATHS
# ============================================================================

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
    / "data/dfc2019/experiments/sparse_anchor_spatial"
)

DEFAULT_ANCHOR_COUNT = 20
DEFAULT_PLACEMENT = "grid"
DEFAULT_BUFFER_RADIUS = 16
DEFAULT_MIN_EVAL_PIXELS = 1000
DEFAULT_MIN_ANCHORS_PER_REGION = 2

VALID_AGL_EPS = 1e-6


# ============================================================================
# HELPERS
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


def calculate_metrics(
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


# ============================================================================
# STAGE-1 ANCHOR REUSE
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
            f"Required Stage-1 anchor file is missing:\n{path}\n"
            "Run the complete Stage-1 experiment first."
        )

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "anchors" not in payload or not payload["anchors"]:
        raise ValueError(
            f"Malformed/empty Stage-1 anchor file: {path}"
        )

    anchors = payload["anchors"]

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
            f"Duplicate anchor coordinates in {path}"
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
# REGIONS
# ============================================================================

def region_id_for_pixel(
    row: np.ndarray,
    col: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Return 2x2 region IDs:

        0 = top-left
        1 = top-right
        2 = bottom-left
        3 = bottom-right

    Boundaries use the exact midpoint split. This matches the same geometric
    rule used for both anchor assignment and evaluation.
    """
    top = row < (height / 2.0)
    left = col < (width / 2.0)

    region = np.empty(
        len(row),
        dtype=np.int8,
    )

    region[top & left] = 0
    region[top & ~left] = 1
    region[~top & left] = 2
    region[~top & ~left] = 3

    return region


def region_name(region: int) -> str:
    return {
        0: "Q1_TL",
        1: "Q2_TR",
        2: "Q3_BL",
        3: "Q4_BR",
    }[int(region)]


def region_mask(
    shape: tuple[int, int],
    region: int,
) -> np.ndarray:
    height, width = shape

    row_mid = height / 2.0
    col_mid = width / 2.0

    rows, cols = np.indices(
        shape,
        dtype=np.int32,
    )

    top = rows < row_mid
    left = cols < col_mid

    if region == 0:
        return top & left

    if region == 1:
        return top & ~left

    if region == 2:
        return ~top & left

    if region == 3:
        return ~top & ~left

    raise ValueError(f"Invalid quadrant ID: {region}")


# ============================================================================
# MODEL FITTING
# ============================================================================

def fit_ols(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    if len(x) < 2:
        raise ValueError("At least 2 anchors are required.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite DAv2 anchor values.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite AGL anchor values.")

    if np.ptp(x) < 1e-10:
        raise ValueError(
            "Anchor DAv2 values have effectively zero spread."
        )

    model = LinearRegression()
    model.fit(
        x.reshape(-1, 1),
        y,
    )

    slope = float(
        np.asarray(model.coef_).reshape(-1)[0]
    )
    intercept = float(model.intercept_)

    if (
        not np.isfinite(slope)
        or not np.isfinite(intercept)
    ):
        raise ValueError(
            "OLS produced non-finite parameters."
        )

    return slope, intercept


# ============================================================================
# PREFLIGHT
# ============================================================================

def preflight_tile(
    tile_id: str,
    depth_path: Path,
    truth_path: Path,
    stage1_dir: Path,
    anchor_count: int,
    placement: str,
    min_anchors_per_region: int,
) -> dict:
    depth = read_depth(depth_path)
    agl = read_agl(truth_path)

    if depth.shape != agl.shape:
        raise ValueError(
            f"{tile_id}: shape mismatch {depth.shape} vs {agl.shape}"
        )

    valid = valid_mask(depth, agl)

    anchor_path = stage1_anchor_path(
        stage1_dir,
        tile_id,
        placement,
        anchor_count,
    )

    anchor_data = load_stage1_anchors(
        anchor_path
    )

    rows = anchor_data["rows"]
    cols = anchor_data["cols"]

    if len(rows) != anchor_count:
        raise ValueError(
            f"{tile_id}: Stage-1 anchor count mismatch. "
            f"Expected {anchor_count}, got {len(rows)}."
        )

    if not np.all(valid[rows, cols]):
        raise ValueError(
            f"{tile_id}: one or more Stage-1 anchors are "
            "invalid under the current validity mask."
        )

    current_dav2 = depth[rows, cols].astype(np.float64)

    if not np.allclose(
        anchor_data["dav2_saved"],
        current_dav2,
        rtol=1e-6,
        atol=1e-6,
    ):
        max_diff = float(
            np.max(
                np.abs(
                    anchor_data["dav2_saved"]
                    - current_dav2
                )
            )
        )
        raise ValueError(
            f"{tile_id}: Stage-1 saved DAv2 does not match "
            f"current depth. Max diff={max_diff}"
        )

    regions = region_id_for_pixel(
        rows,
        cols,
        depth.shape[0],
        depth.shape[1],
    )

    result = {
        "tile_id": tile_id,
        "height": depth.shape[0],
        "width": depth.shape[1],
        "valid_pixels": int(np.count_nonzero(valid)),
        "anchor_count": int(anchor_count),
        "placement": placement,
        "anchor_file": str(anchor_path),
        "all_anchor_dav2_range": float(
            np.ptp(current_dav2)
        ),
        "all_anchor_agl_range": float(
            np.ptp(anchor_data["agl"])
        ),
    }

    total_region_anchor_count = 0

    for region in range(4):
        count = int(
            np.count_nonzero(
                regions == region
            )
        )

        total_region_anchor_count += count

        result[
            f"{region_name(region)}_anchor_count"
        ] = count

        result[
            f"{region_name(region)}_dav2_range"
        ] = (
            float(
                np.ptp(
                    current_dav2[
                        regions == region
                    ]
                )
            )
            if count > 0
            else math.nan
        )

        if count < min_anchors_per_region:
            result[
                f"{region_name(region)}_ready"
            ] = False
        else:
            region_x = current_dav2[
                regions == region
            ]

            result[
                f"{region_name(region)}_ready"
            ] = bool(
                np.ptp(region_x) >= 1e-10
            )

    if total_region_anchor_count != anchor_count:
        raise AssertionError(
            f"{tile_id}: region anchor counts do not sum "
            f"to {anchor_count}."
        )

    bad_regions = [
        region_name(region)
        for region in range(4)
        if (
            result[f"{region_name(region)}_anchor_count"]
            < min_anchors_per_region
            or not result[f"{region_name(region)}_ready"]
        )
    ]

    result["ready_for_quadrant_model"] = (
        len(bad_regions) == 0
    )

    result["bad_regions"] = ",".join(bad_regions)

    return result


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_prediction_mask(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, float], int]:
    use = (
        mask
        & np.isfinite(truth)
        & np.isfinite(prediction)
    )

    count = int(
        np.count_nonzero(use)
    )

    if count < 2:
        return (
            {
                "mae_m": math.nan,
                "rmse_m": math.nan,
                "bias_m": math.nan,
                "pearson": math.nan,
                "spearman": math.nan,
            },
            count,
        )

    return (
        calculate_metrics(
            truth[use].astype(np.float64),
            prediction[use].astype(np.float64),
        ),
        count,
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
    min_eval_pixels: int,
) -> tuple[list[dict], dict]:

    depth = read_depth(depth_path)
    agl = read_agl(truth_path)

    if depth.shape != agl.shape:
        raise ValueError(
            f"{tile_id}: DAv2/AGL shape mismatch."
        )

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

    anchor_data = load_stage1_anchors(
        anchor_path
    )

    rows = anchor_data["rows"]
    cols = anchor_data["cols"]
    anchor_agl = anchor_data["agl"]

    if len(rows) != anchor_count:
        raise ValueError(
            f"{tile_id}: expected {anchor_count} anchors, "
            f"got {len(rows)}."
        )

    if not np.all(valid[rows, cols]):
        raise ValueError(
            f"{tile_id}: invalid Stage-1 anchor."
        )

    current_dav2 = (
        depth[rows, cols]
        .astype(np.float64)
    )

    if not np.allclose(
        anchor_data["dav2_saved"],
        current_dav2,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            f"{tile_id}: Stage-1 DAv2 values changed."
        )

    # --------------------------------------------------------
    # FIT TILE-WIDE MODEL
    # --------------------------------------------------------

    tile_slope, tile_intercept = fit_ols(
        current_dav2,
        anchor_agl,
    )

    tile_prediction = (
        tile_slope
        * depth.astype(np.float64)
        + tile_intercept
    )

    # --------------------------------------------------------
    # FIT 2x2 QUADRANT MODELS
    # --------------------------------------------------------

    anchor_regions = region_id_for_pixel(
        rows,
        cols,
        depth.shape[0],
        depth.shape[1],
    )

    quadrant_models = {}

    for region in range(4):
        selector = anchor_regions == region

        region_x = current_dav2[selector]
        region_y = anchor_agl[selector]

        if len(region_x) < DEFAULT_MIN_ANCHORS_PER_REGION:
            raise ValueError(
                f"{tile_id}: {region_name(region)} has only "
                f"{len(region_x)} anchors."
            )

        slope, intercept = fit_ols(
            region_x,
            region_y,
        )

        quadrant_models[region] = {
            "anchor_count": int(np.count_nonzero(selector)),
            "slope_m_per_dav2": slope,
            "intercept_m": intercept,
            "anchor_rows": rows[selector].tolist(),
            "anchor_cols": cols[selector].tolist(),
        }

    # --------------------------------------------------------
    # Build quadrant prediction map.
    # --------------------------------------------------------

    quadrant_prediction = np.full(
        depth.shape,
        np.nan,
        dtype=np.float64,
    )

    for region in range(4):
        qmask = region_mask(
            depth.shape,
            region,
        )

        params = quadrant_models[region]

        quadrant_prediction[qmask] = (
            params["slope_m_per_dav2"]
            * depth[qmask].astype(np.float64)
            + params["intercept_m"]
        )

    if not np.all(
        np.isfinite(
            quadrant_prediction[valid]
        )
    ):
        raise ValueError(
            f"{tile_id}: quadrant model produced "
            "non-finite predictions on valid pixels."
        )

    # --------------------------------------------------------
    # Evaluation masks
    # --------------------------------------------------------

    anchor_mask = np.zeros(
        valid.shape,
        dtype=bool,
    )

    anchor_mask[rows, cols] = True

    exclusion = make_exclusion_mask(
        valid.shape,
        rows,
        cols,
        buffer_radius,
    )

    all_eval_mask = (
        valid
        & ~anchor_mask
    )

    strict_eval_mask = (
        valid
        & ~exclusion
    )

    if (
        np.count_nonzero(all_eval_mask)
        < min_eval_pixels
    ):
        raise ValueError(
            f"{tile_id}: insufficient all-pixel evaluation set."
        )

    if (
        np.count_nonzero(strict_eval_mask)
        < min_eval_pixels
    ):
        raise ValueError(
            f"{tile_id}: insufficient strict evaluation set."
        )

    results = []

    region = (
        "JAX"
        if tile_id.startswith("JAX_")
        else "OMA"
        if tile_id.startswith("OMA_")
        else "UNKNOWN"
    )

    # --------------------------------------------------------
    # Tile-wide result
    # --------------------------------------------------------

    for mask_name, mask in [
        ("all", all_eval_mask),
        ("strict", strict_eval_mask),
    ]:
        qmetrics, count = evaluate_prediction_mask(
            agl,
            tile_prediction,
            mask,
        )

        if count < 2:
            raise ValueError(
                f"{tile_id}: tile-wide {mask_name} evaluation "
                "has fewer than 2 pixels."
            )

        results.append(
            {
                "tile_id": tile_id,
                "region": region,
                "model": "tile_wide",
                "evaluation": mask_name,
                "anchor_count": anchor_count,
                "placement": placement,
                "eval_pixels": count,
                "slope_m_per_dav2": tile_slope,
                "intercept_m": tile_intercept,
                **qmetrics,
            }
        )

    # --------------------------------------------------------
    # Quadrant-wise result
    # --------------------------------------------------------

    for mask_name, base_mask in [
        ("all", all_eval_mask),
        ("strict", strict_eval_mask),
    ]:
        for quadrant in range(4):
            qmask = region_mask(
                depth.shape,
                quadrant,
            )

            combined = (
                base_mask
                & qmask
            )

            qmetrics, count = evaluate_prediction_mask(
                agl,
                quadrant_prediction,
                combined,
            )

            if count < 2:
                raise ValueError(
                    f"{tile_id}: {region_name(quadrant)} "
                    f"{mask_name} evaluation has fewer than 2 pixels."
                )

            params = quadrant_models[quadrant]

            results.append(
                {
                    "tile_id": tile_id,
                    "region": region,
                    "model": "quadrant_wise",
                    "quadrant": region_name(quadrant),
                    "evaluation": mask_name,
                    "anchor_count": anchor_count,
                    "placement": placement,
                    "eval_pixels": count,
                    "slope_m_per_dav2": params[
                        "slope_m_per_dav2"
                    ],
                    "intercept_m": params[
                        "intercept_m"
                    ],
                    "quadrant_anchor_count": params[
                        "anchor_count"
                    ],
                    **qmetrics,
                }
            )

    # --------------------------------------------------------
    # Aggregate quadrant predictions into a single whole-tile
    # result. This is the MAIN comparison against tile-wide.
    # --------------------------------------------------------

    for mask_name, mask in [
        ("all", all_eval_mask),
        ("strict", strict_eval_mask),
    ]:
        metrics_result, count = evaluate_prediction_mask(
            agl,
            quadrant_prediction,
            mask,
        )

        results.append(
            {
                "tile_id": tile_id,
                "region": region,
                "model": "quadrant_wise_aggregate",
                "evaluation": mask_name,
                "anchor_count": anchor_count,
                "placement": placement,
                "eval_pixels": count,
                **metrics_result,
            }
        )

    # --------------------------------------------------------
    # Save fitted parameters.
    # --------------------------------------------------------

    fit_dir = (
        output_dir / "fits"
    )

    fit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fit_payload = {
        "tile_id": tile_id,
        "anchor_count": anchor_count,
        "placement": placement,
        "buffer_radius_px": buffer_radius,
        "anchor_file": str(anchor_path),
        "synthetic_anchor_warning": (
            "Anchors are synthetic metric anchors from DFC2019 AGL truth; "
            "they are NOT surveyed GCPs."
        ),
        "tile_wide": {
            "slope_m_per_dav2": tile_slope,
            "intercept_m": tile_intercept,
            "anchor_rows": rows.tolist(),
            "anchor_cols": cols.tolist(),
        },
        "quadrants": {
            region_name(region): params
            for region, params in quadrant_models.items()
        },
    }

    (
        fit_dir / f"{tile_id}.json"
    ).write_text(
        json.dumps(
            fit_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    preflight_info = {
        "tile_id": tile_id,
        "region": region,
        "anchor_count": anchor_count,
        "placement": placement,
        "tile_wide_slope": tile_slope,
        "tile_wide_intercept": tile_intercept,
    }

    for q in range(4):
        params = quadrant_models[q]
        preflight_info[
            f"{region_name(q)}_slope"
        ] = params["slope_m_per_dav2"]
        preflight_info[
            f"{region_name(q)}_intercept"
        ] = params["intercept_m"]
        preflight_info[
            f"{region_name(q)}_anchor_count"
        ] = params["anchor_count"]

    return results, preflight_info


# ============================================================================
# AGGREGATION
# ============================================================================

METRIC_COLUMNS = [
    "mae_m",
    "rmse_m",
    "bias_m",
    "pearson",
    "spearman",
]


def summarize_values(
    values: pd.Series,
) -> dict[str, float]:
    values = pd.to_numeric(
        values,
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
    """
    Primary summary uses exactly the whole-tile models:
        tile_wide
        quadrant_wise_aggregate

    Per-quadrant rows remain available in results.csv but are excluded here
    so they cannot be mistaken for independent tiles.
    """
    primary = results[
        results["model"].isin(
            [
                "tile_wide",
                "quadrant_wise_aggregate",
            ]
        )
    ].copy()

    rows = []

    grouped = primary.groupby(
        [
            "model",
            "evaluation",
        ],
        sort=True,
    )

    for (model, evaluation), group in grouped:
        row = {
            "model": model,
            "evaluation": evaluation,
            "runs": len(group),
            "tiles": group["tile_id"].nunique(),
        }

        for metric in METRIC_COLUMNS:
            stats = summarize_values(
                group[metric]
            )

            for name, value in stats.items():
                row[
                    f"{metric}_{name}"
                ] = value

        rows.append(row)

    return pd.DataFrame(rows)


def create_region_distribution(
    results: pd.DataFrame,
) -> pd.DataFrame:
    primary = results[
        results["model"].isin(
            [
                "tile_wide",
                "quadrant_wise_aggregate",
            ]
        )
    ].copy()

    primary["region"] = primary["region"].astype(str)

    rows = []

    grouped = primary.groupby(
        [
            "model",
            "evaluation",
            "region",
        ],
        sort=True,
    )

    for (
        model,
        evaluation,
        region,
    ), group in grouped:
        row = {
            "model": model,
            "evaluation": evaluation,
            "region": region,
            "runs": len(group),
            "tiles": group["tile_id"].nunique(),
        }

        for metric in METRIC_COLUMNS:
            stats = summarize_values(
                group[metric]
            )

            for name, value in stats.items():
                row[
                    f"{metric}_{name}"
                ] = value

        rows.append(row)

    return pd.DataFrame(rows)


def create_paired_deltas(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pair the aggregate quadrant model with the tile-wide model for each
    tile/evaluation regime.

    Positive delta means the quadrant model improved the quantity.

    For errors:
        tile_wide - quadrant_wise

    For correlations:
        quadrant_wise - tile_wide
    """

    tile = results[
        results["model"] == "tile_wide"
    ][
        [
            "tile_id",
            "evaluation",
            "mae_m",
            "rmse_m",
            "pearson",
            "spearman",
        ]
    ].copy()

    tile = tile.rename(
        columns={
            "mae_m": "tile_mae_m",
            "rmse_m": "tile_rmse_m",
            "pearson": "tile_pearson",
            "spearman": "tile_spearman",
        }
    )

    quad = results[
        results["model"]
        == "quadrant_wise_aggregate"
    ][
        [
            "tile_id",
            "evaluation",
            "mae_m",
            "rmse_m",
            "pearson",
            "spearman",
        ]
    ].copy()

    quad = quad.rename(
        columns={
            "mae_m": "quadrant_mae_m",
            "rmse_m": "quadrant_rmse_m",
            "pearson": "quadrant_pearson",
            "spearman": "quadrant_spearman",
        }
    )

    merged = tile.merge(
        quad,
        on=[
            "tile_id",
            "evaluation",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged["delta_mae_m"] = (
        merged["tile_mae_m"]
        - merged["quadrant_mae_m"]
    )

    merged["delta_rmse_m"] = (
        merged["tile_rmse_m"]
        - merged["quadrant_rmse_m"]
    )

    merged["delta_pearson"] = (
        merged["quadrant_pearson"]
        - merged["tile_pearson"]
    )

    merged["delta_spearman"] = (
        merged["quadrant_spearman"]
        - merged["tile_spearman"]
    )

    merged["quadrant_better_mae"] = (
        merged["delta_mae_m"] > 0
    )

    merged["quadrant_better_rmse"] = (
        merged["delta_rmse_m"] > 0
    )

    merged["quadrant_better_pearson"] = (
        merged["delta_pearson"] > 0
    )

    merged["quadrant_better_spearman"] = (
        merged["delta_spearman"] > 0
    )

    return merged


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stage-3A spatial calibration experiment: "
            "tile-wide vs 2x2 quadrant-wise OLS."
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
        help="Must match the Stage-1 anchor count being reused.",
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
        "--min-eval-pixels",
        type=int,
        default=DEFAULT_MIN_EVAL_PIXELS,
    )

    parser.add_argument(
        "--min-anchors-per-region",
        type=int,
        default=DEFAULT_MIN_ANCHORS_PER_REGION,
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
        help="Smoke-test tile limit; omit for full study.",
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
            "Stage 3A is intentionally locked to 20 anchors. "
            "This is the best-supported configuration from Stage 1. "
            "Change the script only if a separate experiment is intended."
        )

    if args.min_anchors_per_region < 2:
        raise ValueError(
            "Each quadrant needs at least 2 anchors for affine OLS."
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

    output_dir = (
        args.output_dir.resolve()
    )

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
                "Requested tile(s) missing from manifest:\n"
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

    # ------------------------------------------------------------
    # Preflight every tile before running the real experiment.
    #
    # This is critical. We don't want tile 37 of 50 to reveal that
    # quadrant-wise fitting is impossible after an hour of processing.
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 3A PREFLIGHT")
    print("=" * 78)
    print(
        f"Tiles:              {len(tile_ids)}"
    )
    print(
        f"Anchors:            {args.anchors}"
    )
    print(
        f"Placement:          {args.placement}"
    )
    print(
        "Spatial model:      2x2 quadrant-wise OLS"
    )
    print(
        f"Minimum anchors/q: {args.min_anchors_per_region}"
    )
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

        row = preflight_tile(
            tile_id=tile_id,
            depth_path=depth_path,
            truth_path=truth_path,
            stage1_dir=args.stage1_dir.resolve(),
            anchor_count=args.anchors,
            placement=args.placement,
            min_anchors_per_region=args.min_anchors_per_region,
        )

        preflight_rows.append(row)

        status = (
            "READY"
            if row["ready_for_quadrant_model"]
            else "NOT READY"
        )

        print(
            f"{index:3d}/{len(tile_ids)} "
            f"{tile_id:14s} "
            f"{status:9s} "
            f"Q anchors = "
            f"{row['Q1_TL_anchor_count']}/"
            f"{row['Q2_TR_anchor_count']}/"
            f"{row['Q3_BL_anchor_count']}/"
            f"{row['Q4_BR_anchor_count']}"
        )

        if not row["ready_for_quadrant_model"]:
            raise RuntimeError(
                f"{tile_id} is not eligible for the 2x2 "
                "quadrant model. "
                f"Bad regions: {row['bad_regions']}"
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

    print()
    print(
        f"Preflight passed for all {len(tile_ids)} tiles."
    )
    print(
        f"Saved: {preflight_path}"
    )

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    config = {
        "experiment": "sparse_anchor_spatial_stage_3A",
        "scientific_question": (
            "Does a spatially varying 2x2 affine calibration "
            "improve upon one tile-wide affine calibration?"
        ),
        "stage1_anchor_reuse": True,
        "anchor_count": args.anchors,
        "placement": args.placement,
        "spatial_model": "2x2 quadrant-wise affine",
        "regression": "OLS",
        "buffer_radius_px": args.buffer_radius,
        "min_anchors_per_region": (
            args.min_anchors_per_region
        ),
        "same_anchor_set_for_both_models": True,
        "same_evaluation_masks": True,
        "tiles": tile_ids,
        "anchor_source": str(
            stage1_anchor_dir
        ),
        "synthetic_anchor_warning": (
            "Anchors are synthetic metric anchors sampled "
            "from DFC2019 AGL truth. They are NOT surveyed GCPs."
        ),
        "primary_evaluation": (
            "strict: valid pixels outside the anchor exclusion buffer"
        ),
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
    # Main experiment
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 3A EXPERIMENT")
    print("=" * 78)
    print(
        "Comparing:"
    )
    print(
        "  1. tile-wide OLS"
    )
    print(
        "  2. 2x2 quadrant-wise OLS"
    )
    print()
    print(
        "The exact same 20 Stage-1 grid anchors are used."
    )
    print(
        "No new anchors are generated."
    )
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
            tile_results, tile_fit = run_tile(
                tile_id=tile_id,
                depth_path=depth_path,
                truth_path=truth_path,
                stage1_dir=args.stage1_dir.resolve(),
                output_dir=output_dir,
                anchor_count=args.anchors,
                placement=args.placement,
                buffer_radius=args.buffer_radius,
                min_eval_pixels=args.min_eval_pixels,
            )

            all_results.extend(
                tile_results
            )

            fit_metadata.append(
                tile_fit
            )

            # Print the strict whole-tile metrics immediately.
            tile_rows = [
                row
                for row in tile_results
                if (
                    row["evaluation"] == "strict"
                    and row["model"]
                    in [
                        "tile_wide",
                        "quadrant_wise_aggregate",
                    ]
                )
            ]

            by_model = {
                row["model"]: row
                for row in tile_rows
            }

            tile_mae = by_model[
                "tile_wide"
            ]["mae_m"]

            quad_mae = by_model[
                "quadrant_wise_aggregate"
            ]["mae_m"]

            print(
                f"tile MAE={tile_mae:.3f}m "
                f"quadrant MAE={quad_mae:.3f}m"
            )

        except Exception as exc:
            print("FAILED")

            raise RuntimeError(
                f"Stage 3A failed for {tile_id}: {exc}"
            ) from exc

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:
        raise RuntimeError(
            "No Stage-3 results were generated."
        )

    # Stable ordering.
    results_df = results_df.sort_values(
        [
            "model",
            "evaluation",
            "tile_id",
        ]
    ).reset_index(
        drop=True
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

    summary_df = summary_df.sort_values(
        [
            "evaluation",
            "model",
        ]
    ).reset_index(
        drop=True
    )

    summary_path = (
        output_dir / "summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    distributions_df = (
        create_region_distribution(
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

    # Save fit metadata separately so it is easy to inspect how much the
    # local slopes/intercepts actually differ across a scene.
    fit_metadata_path = (
        output_dir / "fit_metadata.csv"
    )

    pd.DataFrame(
        fit_metadata
    ).to_csv(
        fit_metadata_path,
        index=False,
    )

    # ------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------

    report_lines = [
        "DepthWizard2 — Stage 3A Spatial Calibration",
        "=" * 68,
        "",
        "Question:",
        "Does 2x2 quadrant-wise affine calibration outperform",
        "a single tile-wide affine calibration?",
        "",
        "Design:",
        "- exact Stage-1 grid/20 anchor coordinates reused",
        "- OLS held fixed",
        "- same exclusion buffer",
        "- same evaluation pixels",
        "- no local fallback model allowed",
        "",
        "PRIMARY RESULTS: STRICT SPATIAL HOLDOUT",
        "",
    ]

    primary_columns = [
        "model",
        "evaluation",
        "runs",
        "tiles",
        "mae_m_mean",
        "rmse_m_mean",
        "pearson_mean",
        "spearman_mean",
        "pearson_p05",
        "pearson_p95",
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
            "PAIRED DELTAS",
            "Positive delta = quadrant-wise improvement.",
            "",
            paired_df[
                paired_df["evaluation"] == "strict"
            ].describe(
                include="all"
            ).to_string()
            if not paired_df.empty
            else "No paired deltas.",
            "",
            "FILES",
            f"preflight.csv:      {preflight_path}",
            f"results.csv:        {results_path}",
            f"summary.csv:        {summary_path}",
            f"distributions.csv: {distributions_path}",
            f"paired_deltas.csv:  {paired_path}",
            f"fit_metadata.csv:   {fit_metadata_path}",
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
    # FINAL OUTPUT
    # ------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 3A COMPLETE")
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
        f"Distributions:  {distributions_path}"
    )
    print(
        f"Paired deltas:  {paired_path}"
    )
    print(
        f"Fit metadata:   {fit_metadata_path}"
    )
    print(
        f"Report:         {report_path}"
    )

    print()
    print(
        "STRICT PRIMARY SUMMARY:"
    )
    print(
        report_df[
            report_df["evaluation"] == "strict"
        ].to_string(
            index=False
        )
    )

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
