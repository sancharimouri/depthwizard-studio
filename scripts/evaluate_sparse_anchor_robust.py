#!/usr/bin/env python3
"""
DepthWizard2 — Robust Sparse Anchor Placement Study

IMPORTANT SCIENTIFIC NOTE
-------------------------
The "anchors" used here are SYNTHETIC METRIC ANCHORS sampled from the
DFC2019 AGL truth raster.

They are NOT surveyed GCPs.

This experiment asks:

    "If a small number of known metric height anchors were available,
     how sensitive would per-tile affine calibration be to where those
     anchors were placed?"

Experimental variable:
    Anchor PLACEMENT only.

Held fixed:
    - DAv2 predictions
    - OLS affine regression
    - evaluation data
    - evaluation masks
    - exclusion-buffer definition

Strategies:
    1. random
    2. grid
    3. spatial_height

Anchor counts:
    5, 10, 20

Random:
    20 independent deterministic repetitions per tile/count.

Evaluation:
    A. all valid non-anchor pixels
    B. strict evaluation excluding a configurable buffer around anchors

Outputs:
    sparse_anchor_robust/
        config.json
        results.csv
        summary.csv
        distributions.csv
        REPORT.txt
        anchors/
            *.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


# ============================================================
# PATHS
# ============================================================

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

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/dfc2019/experiments/sparse_anchor_robust"
)


# ============================================================
# EXPERIMENT DEFAULTS
# ============================================================

DEFAULT_ANCHOR_COUNTS = [5, 10, 20]
DEFAULT_RANDOM_REPEATS = 20

# Pixels around an anchor excluded from strict evaluation.
DEFAULT_BUFFER_RADIUS = 16

# Guard against pathological tiny evaluation sets.
DEFAULT_MIN_EVAL_PIXELS = 1000

VALID_AGL_EPS = 1e-6


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Tile:
    tile_id: str
    truth_path: Path
    depth_path: Path


@dataclass
class AnchorSet:
    rows: np.ndarray
    cols: np.ndarray
    agl: np.ndarray
    strategy: str
    requested_count: int
    seed: int | None
    notes: str


# ============================================================
# REPRODUCIBILITY
# ============================================================

def stable_seed(*parts: object) -> int:
    """
    Stable cross-run seed.

    Python's built-in hash() is intentionally randomized between processes,
    so we use SHA-256 instead.
    """
    payload = "::".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


# ============================================================
# DATA LOADING
# ============================================================

def read_depth(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(path, allow_pickle=False)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D DAv2 array, got {arr.shape}: {path}"
        )

    return arr.astype(np.float32, copy=False)


def read_agl(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)

        metadata = {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "transform": tuple(src.transform),
        }

    return arr, metadata


def valid_pixel_mask(
    depth: np.ndarray,
    agl: np.ndarray,
) -> np.ndarray:
    """
    Valid paired metric pixels.

    DFC AGL can contain tiny negative edge/no-data values, so values below
    approximately zero are excluded.
    """
    return (
        np.isfinite(depth)
        & np.isfinite(agl)
        & (agl >= -VALID_AGL_EPS)
    )


def load_tiles(
    manifest_path: Path,
    depth_dir: Path,
    truth_dir: Path,
    requested_tiles: set[str] | None = None,
) -> list[Tile]:

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_path}"
        )

    if not depth_dir.exists():
        raise FileNotFoundError(
            f"Depth directory not found:\n{depth_dir}"
        )

    if not truth_dir.exists():
        raise FileNotFoundError(
            f"Truth directory not found:\n{truth_dir}"
        )

    manifest = pd.read_csv(manifest_path)

    if "tile_id" not in manifest.columns:
        raise ValueError(
            "Manifest does not contain 'tile_id'. "
            f"Columns are: {list(manifest.columns)}"
        )

    manifest_ids = manifest["tile_id"].astype(str).tolist()

    if requested_tiles is not None:
        missing = sorted(
            requested_tiles - set(manifest_ids)
        )

        if missing:
            raise ValueError(
                "Requested tiles are missing from manifest:\n"
                + "\n".join(missing)
            )

        manifest_ids = [
            tile_id
            for tile_id in manifest_ids
            if tile_id in requested_tiles
        ]

    tiles = []

    for tile_id in manifest_ids:

        depth_path = (
            depth_dir / f"{tile_id}_depth.npy"
        )

        truth_path = (
            truth_dir / f"{tile_id}_AGL.tif"
        )

        if not depth_path.exists():
            raise FileNotFoundError(
                f"Missing DAv2 depth:\n{depth_path}"
            )

        if not truth_path.exists():
            raise FileNotFoundError(
                f"Missing AGL truth:\n{truth_path}"
            )

        tiles.append(
            Tile(
                tile_id=tile_id,
                truth_path=truth_path,
                depth_path=depth_path,
            )
        )

    if not tiles:
        raise ValueError("No tiles selected.")

    return tiles


# ============================================================
# STATISTICS
# ============================================================

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

    pearson = float(
        pearsonr(truth, prediction).statistic
    )

    spearman = float(
        spearmanr(truth, prediction).statistic
    )

    return pearson, spearman


def calculate_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:

    residual = prediction - truth

    mae = float(
        np.mean(np.abs(residual))
    )

    rmse = float(
        np.sqrt(np.mean(residual ** 2))
    )

    bias = float(
        np.mean(residual)
    )

    pearson, spearman = safe_correlations(
        truth,
        prediction,
    )

    return {
        "mae_m": mae,
        "rmse_m": rmse,
        "bias_m": bias,
        "pearson": pearson,
        "spearman": spearman,
    }


# ============================================================
# ANCHOR STATISTICS
# ============================================================

def spatial_statistics(
    rows: np.ndarray,
    cols: np.ndarray,
) -> dict[str, float]:

    if len(rows) < 2:
        return {
            "anchor_row_std": 0.0,
            "anchor_col_std": 0.0,
            "anchor_spread_px": 0.0,
            "anchor_min_pairwise_px": math.nan,
            "anchor_mean_pairwise_px": math.nan,
        }

    points = np.column_stack(
        [
            rows.astype(float),
            cols.astype(float),
        ]
    )

    differences = (
        points[:, None, :]
        - points[None, :, :]
    )

    distances = np.sqrt(
        np.sum(differences ** 2, axis=2)
    )

    upper = distances[
        np.triu_indices(
            len(points),
            k=1,
        )
    ]

    return {
        "anchor_row_std": float(np.std(rows)),
        "anchor_col_std": float(np.std(cols)),
        "anchor_spread_px": float(
            np.sqrt(
                np.var(rows)
                + np.var(cols)
            )
        ),
        "anchor_min_pairwise_px": float(
            np.min(upper)
        ),
        "anchor_mean_pairwise_px": float(
            np.mean(upper)
        ),
    }


def height_statistics(
    agl: np.ndarray,
) -> dict[str, float]:

    return {
        "anchor_agl_min_m": float(np.min(agl)),
        "anchor_agl_max_m": float(np.max(agl)),
        "anchor_agl_mean_m": float(np.mean(agl)),
        "anchor_agl_std_m": float(np.std(agl)),
        "anchor_agl_range_m": float(np.ptp(agl)),
        "anchor_agl_p25_m": float(
            np.percentile(agl, 25)
        ),
        "anchor_agl_median_m": float(
            np.median(agl)
        ),
        "anchor_agl_p75_m": float(
            np.percentile(agl, 75)
        ),
    }


# ============================================================
# RANDOM PLACEMENT
# ============================================================

def select_random(
    valid: np.ndarray,
    agl: np.ndarray,
    n: int,
    seed: int,
) -> AnchorSet:

    rows, cols = np.where(valid)

    values = agl[rows, cols]

    if len(rows) < n:
        raise ValueError(
            f"Only {len(rows)} valid pixels; "
            f"cannot choose {n} anchors."
        )

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(rows),
        size=n,
        replace=False,
    )

    return AnchorSet(
        rows=rows[indices].astype(np.int32),
        cols=cols[indices].astype(np.int32),
        agl=values[indices].astype(np.float32),
        strategy="random",
        requested_count=n,
        seed=seed,
        notes=(
            "Uniform random sampling over all valid "
            "metric pixels."
        ),
    )


# ============================================================
# GRID PLACEMENT
# ============================================================

def nearest_unique_valid_pixels(
    valid: np.ndarray,
    target_rows: np.ndarray,
    target_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    rows, cols = np.where(valid)

    if len(rows) == 0:
        raise ValueError("No valid pixels available.")

    points = np.column_stack(
        [
            rows.astype(float),
            cols.astype(float),
        ]
    )

    targets = np.column_stack(
        [
            target_rows.astype(float),
            target_cols.astype(float),
        ]
    )

    tree = cKDTree(points)

    # Start with several neighbors so collisions can be resolved.
    k = min(
        max(8, len(targets) * 4),
        len(points),
    )

    chosen = []
    used = set()

    while len(chosen) < len(targets):

        _, indices = tree.query(
            targets,
            k=k,
            workers=-1,
        )

        if k == 1:
            indices = indices[:, None]

        for neighbors in indices:

            selected = None

            for candidate in neighbors:

                candidate = int(candidate)

                if candidate not in used:
                    selected = candidate
                    break

            if selected is not None:

                used.add(selected)
                chosen.append(selected)

                if len(chosen) == len(targets):
                    break

        if len(chosen) < len(targets):

            if k >= len(points):
                break

            k = min(
                len(points),
                k * 2,
            )

    if len(chosen) != len(targets):
        raise RuntimeError(
            f"Could not obtain {len(targets)} "
            f"unique valid anchors."
        )

    chosen = np.asarray(
        chosen,
        dtype=np.int64,
    )

    return (
        rows[chosen].astype(np.int32),
        cols[chosen].astype(np.int32),
    )


def select_grid(
    valid: np.ndarray,
    agl: np.ndarray,
    n: int,
) -> AnchorSet:

    height, width = valid.shape

    aspect = width / max(height, 1)

    grid_cols = max(
        1,
        int(round(math.sqrt(n * aspect))),
    )

    grid_rows = max(
        1,
        int(math.ceil(n / grid_cols)),
    )

    while grid_rows * grid_cols < n:
        grid_cols += 1

    targets = []

    for row in range(grid_rows):

        for col in range(grid_cols):

            y0 = row * height / grid_rows
            y1 = (row + 1) * height / grid_rows

            x0 = col * width / grid_cols
            x1 = (col + 1) * width / grid_cols

            targets.append(
                (
                    (y0 + y1) / 2,
                    (x0 + x1) / 2,
                )
            )

    targets = targets[:n]

    target_rows = np.array(
        [x[0] for x in targets],
        dtype=float,
    )

    target_cols = np.array(
        [x[1] for x in targets],
        dtype=float,
    )

    rows, cols = nearest_unique_valid_pixels(
        valid,
        target_rows,
        target_cols,
    )

    return AnchorSet(
        rows=rows,
        cols=cols,
        agl=agl[rows, cols],
        strategy="grid",
        requested_count=n,
        seed=None,
        notes=(
            f"Deterministic {grid_rows}x{grid_cols} "
            "spatial grid; nearest unique valid pixel "
            "to each selected cell center."
        ),
    )


# ============================================================
# SPATIAL + HEIGHT PLACEMENT
# ============================================================

def farthest_point_selection(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    n: int,
    seed: int,
    height_weight: float = 0.20,
) -> np.ndarray:

    if len(rows) < n:
        raise ValueError(
            "Not enough candidates."
        )

    rng = np.random.default_rng(seed)

    start = int(
        rng.integers(
            0,
            len(rows),
        )
    )

    points = np.column_stack(
        [
            rows.astype(float),
            cols.astype(float),
        ]
    )

    point_min = points.min(axis=0)
    point_range = np.maximum(
        points.max(axis=0) - point_min,
        1.0,
    )

    normalized_points = (
        points - point_min
    ) / point_range

    heights = values.astype(float)

    height_range = max(
        float(np.ptp(heights)),
        1e-9,
    )

    normalized_heights = (
        heights - heights.min()
    ) / height_range

    selected = [start]

    min_distance = np.full(
        len(rows),
        np.inf,
        dtype=float,
    )

    for _ in range(1, n):

        last = selected[-1]

        spatial_distance = np.sum(
            (
                normalized_points
                - normalized_points[last]
            ) ** 2,
            axis=1,
        )

        height_distance = (
            normalized_heights
            - normalized_heights[last]
        ) ** 2

        combined = (
            (1.0 - height_weight)
            * spatial_distance
            +
            height_weight
            * height_distance
        )

        min_distance = np.minimum(
            min_distance,
            combined,
        )

        min_distance[selected] = -np.inf

        next_index = int(
            np.argmax(min_distance)
        )

        selected.append(next_index)

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def select_spatial_height(
    valid: np.ndarray,
    agl: np.ndarray,
    n: int,
    seed: int,
) -> AnchorSet:

    rows, cols = np.where(valid)
    values = agl[rows, cols]

    if len(rows) < n:
        raise ValueError(
            f"Only {len(rows)} valid pixels."
        )

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # STEP 1:
    # Divide AGL into quantile strata.
    # --------------------------------------------------------

    number_of_strata = min(
        max(n, 5),
        2 * n,
    )

    quantiles = np.linspace(
        0.0,
        1.0,
        number_of_strata + 1,
    )

    edges = np.quantile(
        values,
        quantiles,
    )

    candidates = []

    candidates_per_stratum = max(
        20,
        int(math.ceil(1200 / n)),
    )

    for i in range(number_of_strata):

        low = edges[i]
        high = edges[i + 1]

        if i == number_of_strata - 1:

            mask = (
                (values >= low)
                &
                (values <= high)
            )

        else:

            mask = (
                (values >= low)
                &
                (values < high)
            )

        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            continue

        take = min(
            len(indices),
            candidates_per_stratum,
        )

        if len(indices) > take:

            indices = rng.choice(
                indices,
                size=take,
                replace=False,
            )

        candidates.extend(
            indices.tolist()
        )

    candidates = np.unique(
        np.asarray(
            candidates,
            dtype=np.int64,
        )
    )

    # --------------------------------------------------------
    # STEP 2:
    # Guarantee a reasonably large candidate pool.
    # --------------------------------------------------------

    desired_pool = min(
        len(rows),
        max(500, 100 * n),
    )

    if len(candidates) < desired_pool:

        all_indices = np.arange(
            len(rows),
            dtype=np.int64,
        )

        remaining = np.setdiff1d(
            all_indices,
            candidates,
            assume_unique=False,
        )

        extra_count = min(
            len(remaining),
            desired_pool - len(candidates),
        )

        if extra_count > 0:

            extra = rng.choice(
                remaining,
                size=extra_count,
                replace=False,
            )

            candidates = np.unique(
                np.concatenate(
                    [
                        candidates,
                        extra,
                    ]
                )
            )

    # --------------------------------------------------------
    # STEP 3:
    # Farthest-point selection.
    #
    # 80% spatial
    # 20% height
    #
    # Height diversity was already injected by quantile
    # stratification, so spatial coverage gets more weight.
    # --------------------------------------------------------

    candidate_rows = rows[candidates]
    candidate_cols = cols[candidates]
    candidate_values = values[candidates]

    chosen_local = farthest_point_selection(
        candidate_rows,
        candidate_cols,
        candidate_values,
        n=n,
        seed=seed,
        height_weight=0.20,
    )

    chosen = candidates[chosen_local]

    return AnchorSet(
        rows=rows[chosen].astype(np.int32),
        cols=cols[chosen].astype(np.int32),
        agl=values[chosen].astype(np.float32),
        strategy="spatial_height",
        requested_count=n,
        seed=seed,
        notes=(
            f"{number_of_strata} AGL quantile strata, "
            "candidate pool, then farthest-point selection "
            "(80% spatial / 20% height)."
        ),
    )


# ============================================================
# ANCHOR VALIDATION
# ============================================================

def validate_anchor_set(
    anchors: AnchorSet,
    valid: np.ndarray,
    requested_count: int,
) -> None:

    if len(anchors.rows) != requested_count:
        raise AssertionError(
            f"Expected {requested_count} anchors, "
            f"got {len(anchors.rows)}."
        )

    coordinates = np.column_stack(
        [
            anchors.rows,
            anchors.cols,
        ]
    )

    if len(
        np.unique(
            coordinates,
            axis=0,
        )
    ) != requested_count:
        raise AssertionError(
            "Duplicate anchor coordinates detected."
        )

    if not np.all(
        valid[
            anchors.rows,
            anchors.cols,
        ]
    ):
        raise AssertionError(
            "At least one anchor is invalid."
        )

    if not np.all(
        np.isfinite(anchors.agl)
    ):
        raise AssertionError(
            "At least one anchor AGL value is non-finite."
        )


# ============================================================
# EXCLUSION BUFFER
# ============================================================

def make_exclusion_mask(
    shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
    radius: int,
) -> np.ndarray:

    if radius < 0:
        raise ValueError(
            "Radius cannot be negative."
        )

    height, width = shape

    mask = np.zeros(
        shape,
        dtype=bool,
    )

    r = int(radius)

    yy, xx = np.mgrid[
        -r:r + 1,
        -r:r + 1,
    ]

    disk = (
        yy * yy
        +
        xx * xx
        <= r * r
    )

    dy, dx = np.where(disk)

    dy -= r
    dx -= r

    for row, col in zip(
        rows,
        cols,
    ):

        rr = row + dy
        cc = col + dx

        inside = (
            (rr >= 0)
            &
            (rr < height)
            &
            (cc >= 0)
            &
            (cc < width)
        )

        mask[
            rr[inside],
            cc[inside],
        ] = True

    return mask


# ============================================================
# OLS CALIBRATION
# ============================================================

def fit_affine_ols(
    depth: np.ndarray,
    agl: np.ndarray,
    anchors: AnchorSet,
) -> tuple[float, float]:

    x = depth[
        anchors.rows,
        anchors.cols,
    ].astype(np.float64)

    y = anchors.agl.astype(np.float64)

    if np.ptp(x) < 1e-10:
        raise ValueError(
            "Anchor DAv2 values have essentially zero "
            "range. Affine calibration is unstable."
        )

    model = LinearRegression()

    model.fit(
        x.reshape(-1, 1),
        y,
    )

    slope = float(
        model.coef_[0]
    )

    intercept = float(
        model.intercept_
    )

    if (
        not np.isfinite(slope)
        or not np.isfinite(intercept)
    ):
        raise ValueError(
            "OLS produced non-finite parameters."
        )

    return slope, intercept


# ============================================================
# EVALUATION
# ============================================================

def evaluate_mask(
    depth: np.ndarray,
    agl: np.ndarray,
    mask: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[dict[str, float], int]:

    usable = (
        mask
        &
        np.isfinite(depth)
        &
        np.isfinite(agl)
        &
        (agl >= -VALID_AGL_EPS)
    )

    count = int(
        np.count_nonzero(usable)
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

    truth = (
        agl[usable]
        .astype(np.float64)
    )

    prediction = (
        slope
        * depth[usable].astype(np.float64)
        + intercept
    )

    return (
        calculate_metrics(
            truth,
            prediction,
        ),
        count,
    )


def evaluate_run(
    depth: np.ndarray,
    agl: np.ndarray,
    anchors: AnchorSet,
    buffer_radius: int,
    min_eval_pixels: int,
) -> dict[str, float]:

    slope, intercept = fit_affine_ols(
        depth,
        agl,
        anchors,
    )

    valid = valid_pixel_mask(
        depth,
        agl,
    )

    # --------------------------------------------------------
    # Evaluation A:
    # Every valid non-anchor pixel.
    # --------------------------------------------------------

    anchor_mask = np.zeros(
        valid.shape,
        dtype=bool,
    )

    anchor_mask[
        anchors.rows,
        anchors.cols,
    ] = True

    all_mask = (
        valid
        &
        ~anchor_mask
    )

    all_metrics, all_count = evaluate_mask(
        depth,
        agl,
        all_mask,
        slope,
        intercept,
    )

    # --------------------------------------------------------
    # Evaluation B:
    # Strictly outside anchor neighborhood.
    # --------------------------------------------------------

    exclusion = make_exclusion_mask(
        valid.shape,
        anchors.rows,
        anchors.cols,
        buffer_radius,
    )

    strict_mask = (
        valid
        &
        ~exclusion
    )

    strict_metrics, strict_count = evaluate_mask(
        depth,
        agl,
        strict_mask,
        slope,
        intercept,
    )

    if all_count < min_eval_pixels:
        raise ValueError(
            f"Only {all_count} all-pixel evaluation samples."
        )

    if strict_count < min_eval_pixels:
        raise ValueError(
            f"Only {strict_count} strict evaluation samples."
        )

    result = {
        "slope_m_per_dav2": slope,
        "intercept_m": intercept,

        "eval_pixels_all": all_count,
        "eval_pixels_strict": strict_count,

        "strict_buffer_radius_px": buffer_radius,
    }

    for key, value in all_metrics.items():
        result[
            f"all_{key}"
        ] = value

    for key, value in strict_metrics.items():
        result[
            f"strict_{key}"
        ] = value

    return result


# ============================================================
# ONE EXPERIMENTAL RUN
# ============================================================

def run_one(
    tile: Tile,
    strategy: str,
    anchor_count: int,
    repeat: int,
    buffer_radius: int,
    min_eval_pixels: int,
    output_dir: Path,
) -> dict:

    depth = read_depth(
        tile.depth_path
    )

    agl, truth_metadata = read_agl(
        tile.truth_path
    )

    # --------------------------------------------------------
    # Hard structural validation.
    # --------------------------------------------------------

    if depth.shape != agl.shape:

        raise ValueError(
            f"{tile.tile_id}: shape mismatch\n"
            f"DAv2: {depth.shape}\n"
            f"AGL:  {agl.shape}"
        )

    valid = valid_pixel_mask(
        depth,
        agl,
    )

    valid_count = int(
        np.count_nonzero(valid)
    )

    if valid_count < anchor_count * 10:
        raise ValueError(
            f"{tile.tile_id}: only "
            f"{valid_count} valid pixels."
        )

    # --------------------------------------------------------
    # Select anchors.
    # --------------------------------------------------------

    if strategy == "random":

        seed = stable_seed(
            "DepthWizard2",
            "sparse-anchor-robust",
            tile.tile_id,
            strategy,
            anchor_count,
            repeat,
        )

        anchors = select_random(
            valid,
            agl,
            anchor_count,
            seed,
        )

    elif strategy == "grid":

        seed = None

        anchors = select_grid(
            valid,
            agl,
            anchor_count,
        )

    elif strategy == "spatial_height":

        seed = stable_seed(
            "DepthWizard2",
            "sparse-anchor-robust",
            tile.tile_id,
            strategy,
            anchor_count,
            0,
        )

        anchors = select_spatial_height(
            valid,
            agl,
            anchor_count,
            seed,
        )

    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    validate_anchor_set(
        anchors,
        valid,
        anchor_count,
    )

    # --------------------------------------------------------
    # Evaluate.
    # --------------------------------------------------------

    metrics = evaluate_run(
        depth=depth,
        agl=agl,
        anchors=anchors,
        buffer_radius=buffer_radius,
        min_eval_pixels=min_eval_pixels,
    )

    # --------------------------------------------------------
    # Collect metadata.
    # --------------------------------------------------------

    region = "UNKNOWN"

    if tile.tile_id.startswith("JAX_"):
        region = "JAX"

    elif tile.tile_id.startswith("OMA_"):
        region = "OMA"

    result = {
        "tile_id": tile.tile_id,
        "region": region,
        "placement": strategy,
        "anchor_count": anchor_count,
        "repeat": repeat,
        "seed": seed,
        "valid_pixels": valid_count,
        **spatial_statistics(
            anchors.rows,
            anchors.cols,
        ),
        **height_statistics(
            anchors.agl,
        ),
        **metrics,
        "truth_path": str(
            tile.truth_path
        ),
        "depth_path": str(
            tile.depth_path
        ),
    }

    # --------------------------------------------------------
    # Save exact anchors.
    # --------------------------------------------------------

    anchor_directory = (
        output_dir / "anchors"
    )

    anchor_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for row, col, height in zip(
        anchors.rows,
        anchors.cols,
        anchors.agl,
    ):

        records.append(
            {
                "row": int(row),
                "col": int(col),
                "agl_m": float(height),
                "dav2": float(
                    depth[row, col]
                ),
            }
        )

    anchor_path = (
        anchor_directory
        /
        (
            f"{tile.tile_id}_"
            f"{strategy}_"
            f"{anchor_count}_"
            f"{repeat:02d}.json"
        )
    )

    anchor_payload = {
        "tile_id": tile.tile_id,
        "placement": strategy,
        "anchor_count": anchor_count,
        "repeat": repeat,
        "seed": seed,
        "selection_notes": anchors.notes,
        "buffer_radius_px": buffer_radius,
        "truth_file": str(
            tile.truth_path
        ),
        "depth_file": str(
            tile.depth_path
        ),
        "synthetic_anchor_warning": (
            "These are synthetic metric anchors sampled "
            "from DFC2019 AGL truth. They are NOT surveyed GCPs."
        ),
        "anchors": records,
    }

    anchor_path.write_text(
        json.dumps(
            anchor_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    result[
        "anchor_file"
    ] = str(anchor_path)

    return result


# ============================================================
# AGGREGATION
# ============================================================

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


def summarize_group(
    group: pd.DataFrame,
) -> dict:

    result = {}

    for metric in METRICS:

        values = pd.to_numeric(
            group[metric],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        result[
            f"{metric}_mean"
        ] = float(values.mean())

        result[
            f"{metric}_median"
        ] = float(values.median())

        result[
            f"{metric}_std"
        ] = float(
            values.std(ddof=0)
        )

        result[
            f"{metric}_p05"
        ] = float(
            values.quantile(0.05)
        )

        result[
            f"{metric}_p25"
        ] = float(
            values.quantile(0.25)
        )

        result[
            f"{metric}_p75"
        ] = float(
            values.quantile(0.75)
        )

        result[
            f"{metric}_p95"
        ] = float(
            values.quantile(0.95)
        )

        result[
            f"{metric}_min"
        ] = float(values.min())

        result[
            f"{metric}_max"
        ] = float(values.max())

    return result


def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = results.groupby(
        [
            "placement",
            "anchor_count",
        ],
        sort=True,
    )

    for (
        placement,
        anchor_count,
    ), group in grouped:

        row = {
            "placement": placement,
            "anchor_count": int(
                anchor_count
            ),
            "runs": len(group),
            "tiles": group[
                "tile_id"
            ].nunique(),
        }

        row.update(
            summarize_group(group)
        )

        row[
            "pearson_lt_0"
        ] = int(
            (
                group[
                    "all_pearson"
                ] < 0
            ).sum()
        )

        row[
            "pearson_lt_0_2"
        ] = int(
            (
                group[
                    "all_pearson"
                ] < 0.2
            ).sum()
        )

        row[
            "rmse_gt_10m"
        ] = int(
            (
                group[
                    "all_rmse_m"
                ] > 10
            ).sum()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def create_region_distributions(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = results.groupby(
        [
            "placement",
            "anchor_count",
            "region",
        ],
        sort=True,
    )

    for (
        placement,
        anchor_count,
        region,
    ), group in grouped:

        row = {
            "placement": placement,
            "anchor_count": int(
                anchor_count
            ),
            "region": region,
            "runs": len(group),
            "tiles": group[
                "tile_id"
            ].nunique(),
        }

        row.update(
            summarize_group(group)
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Robust sparse-anchor placement study "
            "for DepthWizard2 / DFC2019."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=DEFAULT_ANCHOR_COUNTS,
    )

    parser.add_argument(
        "--random-repeats",
        type=int,
        default=DEFAULT_RANDOM_REPEATS,
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
        "--strategies",
        nargs="+",
        choices=[
            "random",
            "grid",
            "spatial_height",
        ],
        default=[
            "random",
            "grid",
            "spatial_height",
        ],
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
        help=(
            "Optional smoke-test tile limit. "
            "Do not use for the final experiment."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    args = parse_arguments()

    if any(
        n <= 0
        for n in args.anchors
    ):
        raise ValueError(
            "Anchor counts must be positive."
        )

    if args.random_repeats <= 0:
        raise ValueError(
            "Random repeats must be positive."
        )

    if args.buffer_radius < 0:
        raise ValueError(
            "Buffer radius cannot be negative."
        )

    output_dir = (
        args.output_dir.resolve()
    )

    if (
        output_dir.exists()
        and args.overwrite
    ):
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested_tiles = (
        set(args.tiles)
        if args.tiles
        else None
    )

    tiles = load_tiles(
        manifest_path=(
            args.manifest.resolve()
        ),
        depth_dir=(
            args.depth_dir.resolve()
        ),
        truth_dir=(
            args.truth_dir.resolve()
        ),
        requested_tiles=requested_tiles,
    )

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        tiles = tiles[
            :args.limit
        ]

    # --------------------------------------------------------
    # Save immutable experiment configuration.
    # --------------------------------------------------------

    config = {
        "experiment": (
            "sparse_anchor_robust_placement"
        ),

        "description": (
            "Robust placement study for synthetic "
            "metric anchors sampled from DFC2019 AGL."
        ),

        "synthetic_anchor_warning": (
            "Anchors are NOT surveyed GCPs."
        ),

        "project_root": str(
            PROJECT_ROOT
        ),

        "manifest": str(
            args.manifest.resolve()
        ),

        "depth_dir": str(
            args.depth_dir.resolve()
        ),

        "truth_dir": str(
            args.truth_dir.resolve()
        ),

        "output_dir": str(
            output_dir
        ),

        "tiles": [
            tile.tile_id
            for tile in tiles
        ],

        "anchor_counts": args.anchors,

        "random_repeats": (
            args.random_repeats
        ),

        "buffer_radius_px": (
            args.buffer_radius
        ),

        "min_eval_pixels": (
            args.min_eval_pixels
        ),

        "strategies": args.strategies,

        "regression": (
            "OLS affine: "
            "AGL = slope * DAv2 + intercept"
        ),

        "evaluation": {
            "all_valid_non_anchor_pixels": True,
            "strict_exclusion_buffer": True,
            "strict_buffer_radius_px": (
                args.buffer_radius
            ),
        },

        "random_seed_scheme": (
            "SHA256 of experiment/tile/"
            "strategy/count/repeat"
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

    # --------------------------------------------------------
    # Print experiment header.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "DepthWizard2 — Robust Sparse Anchor "
        "Placement Study"
    )
    print("=" * 78)

    print(
        f"Tiles:          {len(tiles)}"
    )

    print(
        f"Counts:         {args.anchors}"
    )

    print(
        f"Strategies:     {args.strategies}"
    )

    print(
        f"Random repeats: {args.random_repeats}"
    )

    print(
        f"Buffer:         {args.buffer_radius}px"
    )

    print(
        f"Output:         {output_dir}"
    )

    print()
    print(
        "Synthetic anchors are samples from DFC2019 "
        "AGL truth."
    )
    print(
        "They are NOT surveyed GCPs."
    )

    print("=" * 78)
    print()

    # --------------------------------------------------------
    # Determine total jobs.
    # --------------------------------------------------------

    total_jobs = 0

    for strategy in args.strategies:

        repeats = (
            args.random_repeats
            if strategy == "random"
            else 1
        )

        total_jobs += (
            len(tiles)
            * len(args.anchors)
            * repeats
        )

    completed = 0

    results = []

    # --------------------------------------------------------
    # Run.
    # --------------------------------------------------------

    for tile in tiles:

        print(
            f"\n[{tile.tile_id}]"
        )

        for strategy in args.strategies:

            repeats = (
                args.random_repeats
                if strategy == "random"
                else 1
            )

            for anchor_count in args.anchors:

                for repeat in range(
                    repeats
                ):

                    completed += 1

                    print(
                        f"  {completed:4d}/"
                        f"{total_jobs}  "
                        f"{strategy:16s} "
                        f"n={anchor_count:2d} "
                        f"rep={repeat:02d}",
                        end=" ... ",
                        flush=True,
                    )

                    try:

                        row = run_one(
                            tile=tile,
                            strategy=strategy,
                            anchor_count=(
                                anchor_count
                            ),
                            repeat=repeat,
                            buffer_radius=(
                                args.buffer_radius
                            ),
                            min_eval_pixels=(
                                args.min_eval_pixels
                            ),
                            output_dir=output_dir,
                        )

                        results.append(row)

                        print(
                            f"MAE="
                            f"{row['all_mae_m']:.3f}m "
                            f"RMSE="
                            f"{row['all_rmse_m']:.3f}m "
                            f"r="
                            f"{row['all_pearson']:.4f} "
                            f"rho="
                            f"{row['all_spearman']:.4f}"
                        )

                    except Exception as exc:

                        print(
                            "FAILED"
                        )

                        raise RuntimeError(
                            f"Experiment failed:\n"
                            f"tile={tile.tile_id}\n"
                            f"strategy={strategy}\n"
                            f"anchors={anchor_count}\n"
                            f"repeat={repeat}\n"
                            f"error={exc}"
                        ) from exc

    # --------------------------------------------------------
    # Assemble dataframe.
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise RuntimeError(
            "Experiment produced no results."
        )

    results_df = results_df.sort_values(
        [
            "placement",
            "anchor_count",
            "tile_id",
            "repeat",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save detailed results.
    # --------------------------------------------------------

    results_path = (
        output_dir / "results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Regional distributions.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Human-readable report.
    # --------------------------------------------------------

    report_lines = [
        "DepthWizard2 Robust Sparse Anchor Placement Study",
        "=" * 60,
        "",
        "Synthetic metric anchors sampled from DFC2019 AGL truth.",
        "These are NOT surveyed GCPs.",
        "",
        "OLS affine calibration was held fixed.",
        "Only anchor placement/count/repetition was varied.",
        "",
        "PRIMARY METRICS: strict evaluation outside the",
        "anchor exclusion buffer.",
        "",
    ]

    primary_columns = [
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
            column
            for column in primary_columns
            if column in summary_df.columns
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
            f"Detailed results: {results_path}",
            f"Summary:          {summary_path}",
            f"Distributions:    {distributions_path}",
            f"Anchors:          {output_dir / 'anchors'}",
        ]
    )

    report_path = (
        output_dir / "REPORT.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final console output.
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("EXPERIMENT COMPLETE")
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
        f"Report:        {report_path}"
    )

    print(
        f"Anchors:       {output_dir / 'anchors'}"
    )

    print()
    print(
        report_df.to_string(
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
