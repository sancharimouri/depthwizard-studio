#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CALIB_DIR = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "dav2_calibration"
)

BASELINE_DIR = (
    PROJECT_ROOT
    / "data"
    / "dfc2019"
    / "experiments"
    / "dav2_baseline"
)

TEST_MANIFEST = BASELINE_DIR / "manifest.csv"
TEST_DEPTH_DIR = BASELINE_DIR / "depth"

MODEL_DIR = CALIB_DIR / "models"

OUTPUT_DIR = CALIB_DIR / "evaluation"
PRED_DIR = OUTPUT_DIR / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


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


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    valid = np.isfinite(pred) & np.isfinite(truth)

    x = pred[valid].astype(np.float64)
    y = truth[valid].astype(np.float64)

    if len(x) == 0:
        raise ValueError("No valid pixels.")

    errors = x - y

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
        "valid_pct": float(100.0 * valid.mean()),
        "mae_m": mae,
        "rmse_m": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "pred_min_m": float(np.min(x)),
        "pred_max_m": float(np.max(x)),
        "pred_mean_m": float(np.mean(x)),
        "pred_std_m": float(np.std(x)),
        "truth_min_m": float(np.min(y)),
        "truth_max_m": float(np.max(y)),
        "truth_mean_m": float(np.mean(y)),
        "truth_std_m": float(np.std(y)),
    }


def evaluate_mapping(
    method_name: str,
    transform,
    manifest: pd.DataFrame,
    predictions_dir: Path,
):
    rows = []

    print("\n" + "=" * 70)
    print(method_name)
    print("=" * 70)

    for i, row in manifest.iterrows():
        tile_id = str(row["tile_id"])
        city = str(row["city"])

        depth_path = TEST_DEPTH_DIR / f"{tile_id}_depth.npy"
        agl_path = resolve_project_path(str(row["agl_path"]))

        if not depth_path.exists():
            print(f"[{i + 1}/{len(manifest)}] {tile_id}: missing DAv2")
            continue

        if not agl_path.exists():
            print(f"[{i + 1}/{len(manifest)}] {tile_id}: missing AGL")
            continue

        depth = np.load(depth_path).astype(np.float32)
        agl = load_agl(agl_path)

        if depth.shape != agl.shape:
            raise ValueError(
                f"{tile_id}: shape mismatch "
                f"{depth.shape} vs {agl.shape}"
            )

        # Apply the already-fitted mapping.
        pred = transform(depth).astype(np.float32)

        out_path = predictions_dir / f"{tile_id}_nDSM.npy"
        np.save(out_path, pred)

        result = metrics(pred, agl)

        result.update(
            {
                "tile_id": tile_id,
                "city": city,
                "prediction_path": str(out_path),
            }
        )

        rows.append(result)

        print(
            f"[{i + 1}/{len(manifest)}] "
            f"{tile_id}: "
            f"MAE={result['mae_m']:.3f} m, "
            f"RMSE={result['rmse_m']:.3f} m, "
            f"P={result['pearson']:.3f}, "
            f"S={result['spearman']:.3f}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(TEST_MANIFEST)

    linear_path = MODEL_DIR / "linear_mapping.json"
    isotonic_path = MODEL_DIR / "isotonic_mapping.joblib"

    if not linear_path.exists():
        raise FileNotFoundError(linear_path)

    if not isotonic_path.exists():
        raise FileNotFoundError(isotonic_path)

    manifest = pd.read_csv(TEST_MANIFEST)

    # ---------------------------------------------------------------
    # Load linear mapping
    # ---------------------------------------------------------------

    with open(linear_path, "r", encoding="utf-8") as f:
        linear = json.load(f)

    slope = float(linear["slope"])
    intercept = float(linear["intercept"])

    def linear_transform(depth):
        return slope * depth + intercept

    # ---------------------------------------------------------------
    # Load isotonic mapping
    # ---------------------------------------------------------------

    isotonic = joblib.load(isotonic_path)

    def isotonic_transform(depth):
        flat = depth.reshape(-1).astype(np.float64)

        predicted = isotonic.predict(flat)

        return predicted.reshape(depth.shape)

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------

    linear_df = evaluate_mapping(
        "METHOD 1A — GLOBAL LINEAR",
        linear_transform,
        manifest,
        PRED_DIR / "linear",
    )

    isotonic_df = evaluate_mapping(
        "METHOD 1B — GLOBAL ISOTONIC",
        isotonic_transform,
        manifest,
        PRED_DIR / "isotonic",
    )

    # ---------------------------------------------------------------
    # Write per-tile results
    # ---------------------------------------------------------------

    linear_csv = OUTPUT_DIR / "linear_metrics.csv"
    isotonic_csv = OUTPUT_DIR / "isotonic_metrics.csv"

    linear_df.to_csv(linear_csv, index=False)
    isotonic_df.to_csv(isotonic_csv, index=False)

    # ---------------------------------------------------------------
    # Overall summary
    # ---------------------------------------------------------------

    summary_rows = []

    for method, df in [
        ("global_linear", linear_df),
        ("global_isotonic", isotonic_df),
    ]:
        summary_rows.append(
            {
                "method": method,
                "tiles": len(df),
                "mae_mean_m": df["mae_m"].mean(),
                "mae_median_m": df["mae_m"].median(),
                "rmse_mean_m": df["rmse_m"].mean(),
                "rmse_median_m": df["rmse_m"].median(),
                "pearson_mean": df["pearson"].mean(),
                "pearson_median": df["pearson"].median(),
                "spearman_mean": df["spearman"].mean(),
                "spearman_median": df["spearman"].median(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "calibration_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ---------------------------------------------------------------
    # City summary
    # ---------------------------------------------------------------

    city_rows = []

    for method, df in [
        ("global_linear", linear_df),
        ("global_isotonic", isotonic_df),
    ]:
        grouped = (
            df.groupby("city")
            .agg(
                tiles=("tile_id", "count"),
                mae_mean_m=("mae_m", "mean"),
                rmse_mean_m=("rmse_m", "mean"),
                pearson_mean=("pearson", "mean"),
                spearman_mean=("spearman", "mean"),
            )
            .reset_index()
        )

        grouped.insert(0, "method", method)
        city_rows.append(grouped)

    city_df = pd.concat(city_rows, ignore_index=True)

    city_path = OUTPUT_DIR / "calibration_evaluation_city_summary.csv"
    city_df.to_csv(city_path, index=False)

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALIBRATION EVALUATION COMPLETE")
    print("=" * 70)

    print("\nOVERALL")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nBY CITY")
    print(
        city_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nFILES")
    print(linear_csv)
    print(isotonic_csv)
    print(summary_path)
    print(city_path)


if __name__ == "__main__":
    main()
