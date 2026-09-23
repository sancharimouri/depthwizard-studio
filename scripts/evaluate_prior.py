#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def valid_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape, dtype=bool)

    for arr in arrays:
        mask &= np.isfinite(arr)

    return mask


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    mask = valid_mask(y_true, y_pred)

    y = y_true[mask].astype(np.float64)
    p = y_pred[mask].astype(np.float64)

    if len(y) < 2:
        raise ValueError("Not enough valid pixels for evaluation.")

    return {
        "mae_m": float(mean_absolute_error(y, p)),
        "rmse_m": float(
            np.sqrt(mean_squared_error(y, p))
        ),
        "pearson": float(
            pearsonr(y, p).statistic
        ),
        "spearman": float(
            spearmanr(y, p).statistic
        ),
        "n_pixels": int(len(y)),
    }


def fit_semantic_model(
    dav2: np.ndarray,
    building: np.ndarray,
    truth: np.ndarray,
):
    """
    Method 3:

        AGL =
            beta0
            + beta1 * DAv2
            + beta2 * P_building
            + beta3 * (DAv2 * P_building)
    """

    mask = (
        np.isfinite(dav2)
        & np.isfinite(building)
        & np.isfinite(truth)
        & (truth >= 0)
        & (building >= 0)
        & (building <= 1)
    )

    x_depth = dav2[mask].astype(np.float64)
    x_building = building[mask].astype(np.float64)
    y = truth[mask].astype(np.float64)

    X = np.column_stack(
        [
            x_depth,
            x_building,
            x_depth * x_building,
        ]
    )

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict(X)

    return model, y, prediction


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Method 3 semantic-conditioned DAv2 calibration."
    )

    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path(
            "data/dfc2019/experiments/dav2_baseline/depth"
        ),
    )

    parser.add_argument(
        "--building-dir",
        type=Path,
        default=Path(
            "data/dfc2019/experiments/semantic/building"
        ),
    )

    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=Path(
            "data/dfc2019/raw/Truth/Track1-Truth"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/dfc2019/experiments/semantic/"
            "method3_results.json"
        ),
    )

    args = parser.parse_args()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # IMPORTANT:
    #
    # DAv2 files are named:
    #
    #     JAX_004_006_depth.npy
    #
    # but the actual DFC tile ID is:
    #
    #     JAX_004_006
    #
    # Therefore we explicitly search for *_depth.npy
    # and remove the "_depth" suffix.

    depth_files = sorted(
        args.depth_dir.glob("*_depth.npy")
    )

    if not depth_files:
        raise RuntimeError(
            f"No DAv2 depth files found in:\n"
            f"{args.depth_dir}"
        )

    print(
        f"Found {len(depth_files)} DAv2 depth files."
    )

    for depth_path in depth_files:

        tile_id = depth_path.stem.removesuffix(
            "_depth"
        )

        building_path = (
            args.building_dir / f"{tile_id}.npy"
        )

        truth_path = (
            args.truth_dir / f"{tile_id}_AGL.tif"
        )

        # --------------------------------------------------
        # Check required files
        # --------------------------------------------------

        if not building_path.exists():
            print(
                f"[SKIP] Missing building prior: "
                f"{building_path}"
            )
            continue

        if not truth_path.exists():
            print(
                f"[SKIP] Missing AGL truth: "
                f"{truth_path}"
            )
            continue

        print(
            f"\n[EVAL] {tile_id}"
        )

        # --------------------------------------------------
        # Load DAv2 depth
        # --------------------------------------------------

        dav2 = np.load(
            depth_path
        ).astype(np.float32)

        # --------------------------------------------------
        # Load building probability
        # --------------------------------------------------

        building = np.load(
            building_path
        ).astype(np.float32)

        # --------------------------------------------------
        # Load DFC AGL truth
        # --------------------------------------------------

        with rasterio.open(truth_path) as src:
            truth = src.read(1).astype(np.float32)

        # --------------------------------------------------
        # Shape validation
        # --------------------------------------------------

        if (
            dav2.shape
            != building.shape
            or dav2.shape
            != truth.shape
        ):
            raise ValueError(
                f"{tile_id}: shape mismatch\n"
                f"    DAv2:     {dav2.shape}\n"
                f"    Building: {building.shape}\n"
                f"    AGL:      {truth.shape}"
            )

        # --------------------------------------------------
        # Validity mask
        # --------------------------------------------------

        mask = (
            np.isfinite(dav2)
            & np.isfinite(building)
            & np.isfinite(truth)
            & (truth >= 0)
            & (building >= 0)
            & (building <= 1)
        )

        if mask.sum() < 2:
            print(
                f"[SKIP] {tile_id}: "
                f"not enough valid pixels."
            )
            continue

        # --------------------------------------------------
        # 1. Baseline
        #
        # AGL = a * DAv2 + b
        # --------------------------------------------------

        x_base = dav2[mask].reshape(-1, 1)
        y_true = truth[mask]

        baseline_model = LinearRegression()
        baseline_model.fit(
            x_base,
            y_true,
        )

        baseline_pred = baseline_model.predict(
            x_base
        )

        baseline_metrics = {
            "mae_m": float(
                mean_absolute_error(
                    y_true,
                    baseline_pred,
                )
            ),
            "rmse_m": float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        baseline_pred,
                    )
                )
            ),
            "pearson": float(
                pearsonr(
                    y_true,
                    baseline_pred,
                ).statistic
            ),
            "spearman": float(
                spearmanr(
                    y_true,
                    baseline_pred,
                ).statistic
            ),
            "n_pixels": int(mask.sum()),
        }

        # --------------------------------------------------
        # 2. Method 3
        # --------------------------------------------------

        semantic_model, y, semantic_pred = (
            fit_semantic_model(
                dav2,
                building,
                truth,
            )
        )

        semantic_metrics = {
            "mae_m": float(
                mean_absolute_error(
                    y,
                    semantic_pred,
                )
            ),
            "rmse_m": float(
                np.sqrt(
                    mean_squared_error(
                        y,
                        semantic_pred,
                    )
                )
            ),
            "pearson": float(
                pearsonr(
                    y,
                    semantic_pred,
                ).statistic
            ),
            "spearman": float(
                spearmanr(
                    y,
                    semantic_pred,
                ).statistic
            ),
            "n_pixels": int(len(y)),
        }

        # --------------------------------------------------
        # Save tile result
        # --------------------------------------------------

        result = {
            "tile": tile_id,

            "files": {
                "dav2": str(depth_path),
                "building": str(building_path),
                "truth": str(truth_path),
            },

            "baseline": {
                "model": "global_affine_per_tile",
                "metrics": baseline_metrics,
                "slope": float(
                    baseline_model.coef_[0]
                ),
                "intercept": float(
                    baseline_model.intercept_
                ),
            },

            "method3_semantic": {
                "formula": (
                    "AGL = beta0 "
                    "+ beta1*DAv2 "
                    "+ beta2*Pbuilding "
                    "+ beta3*(DAv2*Pbuilding)"
                ),

                "metrics": semantic_metrics,

                "coefficients": {
                    "intercept": float(
                        semantic_model.intercept_
                    ),
                    "dav2": float(
                        semantic_model.coef_[0]
                    ),
                    "building": float(
                        semantic_model.coef_[1]
                    ),
                    "interaction": float(
                        semantic_model.coef_[2]
                    ),
                },
            },
        }

        results.append(result)

        print(
            f"    Baseline : "
            f"MAE={baseline_metrics['mae_m']:.3f} m | "
            f"RMSE={baseline_metrics['rmse_m']:.3f} m | "
            f"r={baseline_metrics['pearson']:.3f}"
        )

        print(
            f"    Method 3 : "
            f"MAE={semantic_metrics['mae_m']:.3f} m | "
            f"RMSE={semantic_metrics['rmse_m']:.3f} m | "
            f"r={semantic_metrics['pearson']:.3f}"
        )

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    if not results:
        raise RuntimeError(
            "No tiles were evaluated.\n"
            "Make sure the building probability maps exist in:\n"
            f"{args.building_dir}"
        )

    # Aggregate tile-level means
    baseline_means = {
        key: float(
            np.mean(
                [
                    r["baseline"]["metrics"][key]
                    for r in results
                ]
            )
        )
        for key in [
            "mae_m",
            "rmse_m",
            "pearson",
            "spearman",
        ]
    }

    semantic_means = {
        key: float(
            np.mean(
                [
                    r["method3_semantic"]["metrics"][key]
                    for r in results
                ]
            )
        )
        for key in [
            "mae_m",
            "rmse_m",
            "pearson",
            "spearman",
        ]
    }

    output = {
        "experiment": "DFC2019 Method 3 semantic prior",

        "formula": (
            "AGL = beta0 "
            "+ beta1*DAv2 "
            "+ beta2*Pbuilding "
            "+ beta3*(DAv2*Pbuilding)"
        ),

        "n_tiles": len(results),

        "aggregate_tile_mean": {
            "baseline": baseline_means,
            "method3_semantic": semantic_means,
        },

        "tiles": results,
    }

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Tiles evaluated: {len(results)}"
    )

    print("\nBaseline:")
    print(
        f"  MAE     = "
        f"{baseline_means['mae_m']:.4f} m"
    )
    print(
        f"  RMSE    = "
        f"{baseline_means['rmse_m']:.4f} m"
    )
    print(
        f"  Pearson = "
        f"{baseline_means['pearson']:.4f}"
    )
    print(
        f"  Spearman= "
        f"{baseline_means['spearman']:.4f}"
    )

    print("\nMethod 3:")
    print(
        f"  MAE     = "
        f"{semantic_means['mae_m']:.4f} m"
    )
    print(
        f"  RMSE    = "
        f"{semantic_means['rmse_m']:.4f} m"
    )
    print(
        f"  Pearson = "
        f"{semantic_means['pearson']:.4f}"
    )
    print(
        f"  Spearman= "
        f"{semantic_means['spearman']:.4f}"
    )

    print("\nSaved:")
    print(args.output)


if __name__ == "__main__":
    main()
