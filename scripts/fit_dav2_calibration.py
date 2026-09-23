#!/usr/bin/env python3

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "dav2_calibration"
)

MANIFEST_PATH = EXPERIMENT_DIR / "manifest.csv"
DEPTH_DIR = EXPERIMENT_DIR / "depth"

OUTPUT_DIR = EXPERIMENT_DIR / "models"
SAMPLES_PATH = EXPERIMENT_DIR / "calibration_samples.csv"
SUMMARY_PATH = EXPERIMENT_DIR / "calibration_fit_summary.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PIXELS_PER_TILE = 10_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def resolve_project_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_agl(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def sample_tile(
    depth: np.ndarray,
    agl: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
):
    if depth.shape != agl.shape:
        raise ValueError(
            f"Shape mismatch: depth={depth.shape}, AGL={agl.shape}"
        )

    valid = np.isfinite(depth) & np.isfinite(agl)

    depth_values = depth[valid].astype(np.float64)
    agl_values = agl[valid].astype(np.float64)

    if len(depth_values) == 0:
        return np.array([]), np.array([])

    n = min(n_samples, len(depth_values))

    indices = rng.choice(
        len(depth_values),
        size=n,
        replace=False,
    )

    return (
        depth_values[indices],
        agl_values[indices],
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FIT DAv2 → METRIC HEIGHT CALIBRATION")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    manifest = pd.read_csv(MANIFEST_PATH)

    rng = np.random.default_rng(RANDOM_SEED)

    all_depth = []
    all_agl = []

    tile_rows = []

    for i, row in manifest.iterrows():
        tile_id = str(row["tile_id"])

        agl_path = resolve_project_path(str(row["agl_path"]))
        depth_path = DEPTH_DIR / f"{tile_id}_depth.npy"

        print(
            f"[{i + 1}/{len(manifest)}] {tile_id}"
        )

        if not agl_path.exists():
            print("  ERROR: missing AGL — skipping")
            continue

        if not depth_path.exists():
            print("  ERROR: missing DAv2 output — skipping")
            continue

        depth = np.load(depth_path).astype(np.float32)
        agl = load_agl(agl_path)

        x, y = sample_tile(
            depth,
            agl,
            PIXELS_PER_TILE,
            rng,
        )

        if len(x) == 0:
            print("  ERROR: no valid pixels — skipping")
            continue

        all_depth.append(x)
        all_agl.append(y)

        tile_rows.append(
            {
                "tile_id": tile_id,
                "city": row["city"],
                "samples": len(x),
                "dav2_min": float(x.min()),
                "dav2_max": float(x.max()),
                "dav2_mean": float(x.mean()),
                "agl_min": float(y.min()),
                "agl_max": float(y.max()),
                "agl_mean": float(y.mean()),
            }
        )

        print(f"  sampled {len(x):,} pixels")

    if not all_depth:
        raise RuntimeError("No calibration samples were generated.")

    X = np.concatenate(all_depth)
    y = np.concatenate(all_agl)

    print("\n" + "=" * 70)
    print("CALIBRATION DATASET")
    print("=" * 70)

    print(f"Tiles used       : {len(tile_rows)}")
    print(f"Total samples    : {len(X):,}")
    print(f"DAv2 range       : {X.min():.6f} → {X.max():.6f}")
    print(f"AGL range        : {y.min():.6f} → {y.max():.6f}")
    print(f"DAv2 mean        : {X.mean():.6f}")
    print(f"AGL mean         : {y.mean():.6f}")

    # ---------------------------------------------------------------
    # Method 1A: global linear calibration
    # ---------------------------------------------------------------

    linear = LinearRegression()
    linear.fit(X.reshape(-1, 1), y)

    slope = float(linear.coef_[0])
    intercept = float(linear.intercept_)
    r2 = float(linear.score(X.reshape(-1, 1), y))

    print("\n" + "=" * 70)
    print("METHOD 1A — GLOBAL LINEAR")
    print("=" * 70)

    print(f"Slope     : {slope:.10f}")
    print(f"Intercept : {intercept:.10f}")
    print(f"R²        : {r2:.6f}")

    linear_model = {
        "method": "global_linear",
        "slope": slope,
        "intercept": intercept,
        "r2_on_calibration_samples": r2,
        "pixels_per_tile": PIXELS_PER_TILE,
        "random_seed": RANDOM_SEED,
    }

    with open(
        OUTPUT_DIR / "linear_mapping.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(linear_model, f, indent=2)

    # ---------------------------------------------------------------
    # Method 1B: monotonic nonlinear calibration
    # ---------------------------------------------------------------
    #
    # IsotonicRegression learns:
    #
    #       DAv2 relative value → predicted height
    #
    # while enforcing a monotonically increasing relationship.
    # ---------------------------------------------------------------

    # Sort the samples. IsotonicRegression expects x values in order.
    order = np.argsort(X)

    X_sorted = X[order]
    y_sorted = y[order]

    isotonic = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )

    isotonic.fit(
        X_sorted,
        y_sorted,
    )

    calibration_predictions = isotonic.predict(X_sorted)

    iso_json_path = OUTPUT_DIR / "isotonic_mapping.json"

    # Save the fitted curve explicitly as JSON so it is inspectable
    # without loading a pickle.
    iso_model = {
        "method": "global_monotonic_isotonic",
        "random_seed": RANDOM_SEED,
        "pixels_per_tile": PIXELS_PER_TILE,
        "x_thresholds": isotonic.X_thresholds_.tolist(),
        "y_thresholds": isotonic.y_thresholds_.tolist(),
    }

    with open(
        iso_json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(iso_model, f)

    # Also save the sklearn object for direct reuse.
    joblib.dump(
        isotonic,
        OUTPUT_DIR / "isotonic_mapping.joblib",
    )

    print("\n" + "=" * 70)
    print("METHOD 1B — GLOBAL MONOTONIC ISOTONIC")
    print("=" * 70)

    print(
        f"Number of thresholds: "
        f"{len(isotonic.X_thresholds_)}"
    )

    # ---------------------------------------------------------------
    # Save sampled calibration data
    # ---------------------------------------------------------------

    samples_df = pd.DataFrame(
        {
            "dav2": X.astype(np.float32),
            "agl_m": y.astype(np.float32),
        }
    )

    samples_df.to_csv(
        SAMPLES_PATH,
        index=False,
    )

    tile_df = pd.DataFrame(tile_rows)

    tile_summary_path = (
        EXPERIMENT_DIR / "calibration_tile_summary.csv"
    )

    tile_df.to_csv(
        tile_summary_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------------

    summary = {
        "tiles_requested": int(len(manifest)),
        "tiles_used": int(len(tile_rows)),
        "pixels_per_tile": PIXELS_PER_TILE,
        "total_samples": int(len(X)),
        "random_seed": RANDOM_SEED,
        "linear": {
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
        },
        "isotonic": {
            "threshold_count": int(
                len(isotonic.X_thresholds_)
            ),
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(OUTPUT_DIR / "linear_mapping.json")
    print(OUTPUT_DIR / "isotonic_mapping.json")
    print(OUTPUT_DIR / "isotonic_mapping.joblib")
    print(SAMPLES_PATH)
    print(tile_summary_path)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
