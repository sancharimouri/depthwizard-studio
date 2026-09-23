#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(y_true, y_pred):
    if len(y_true) < 2:
        return {
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "n_pixels": int(len(y_true)),
        }

    return {
        "mae_m": float(mean_absolute_error(y_true, y_pred)),
        "rmse_m": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "pearson": float(
            pearsonr(y_true, y_pred).statistic
        ),
        "spearman": float(
            spearmanr(y_true, y_pred).statistic
        ),
        "n_pixels": int(len(y_true)),
    }


def build_features(dav2, building):
    return np.column_stack(
        [
            dav2,
            building,
            dav2 * building,
        ]
    )


def main():
    parser = argparse.ArgumentParser()

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
            "method3_spatial_cv_results.json"
        ),
    )

    args = parser.parse_args()

    results = []

    depth_files = sorted(
        args.depth_dir.glob("*_depth.npy")
    )

    if not depth_files:
        raise RuntimeError("No DAv2 files found.")

    for depth_path in depth_files:

        tile_id = depth_path.stem.removesuffix("_depth")

        building_path = (
            args.building_dir / f"{tile_id}.npy"
        )

        truth_path = (
            args.truth_dir / f"{tile_id}_AGL.tif"
        )

        if not building_path.exists():
            print(f"[SKIP] {tile_id}: building prior missing")
            continue

        if not truth_path.exists():
            print(f"[SKIP] {tile_id}: AGL truth missing")
            continue

        dav2 = np.load(depth_path).astype(np.float32)
        building = np.load(building_path).astype(np.float32)

        import rasterio

        with rasterio.open(truth_path) as src:
            truth = src.read(1).astype(np.float32)

        if not (
            dav2.shape
            == building.shape
            == truth.shape
        ):
            raise ValueError(
                f"{tile_id}: shape mismatch: "
                f"{dav2.shape}, "
                f"{building.shape}, "
                f"{truth.shape}"
            )

        h, w = truth.shape

        # -------------------------------------------------
        # Four spatial quadrants
        # -------------------------------------------------

        folds = [
            ("top_left", 0, h // 2, 0, w // 2),
            ("top_right", 0, h // 2, w // 2, w),
            ("bottom_left", h // 2, h, 0, w // 2),
            ("bottom_right", h // 2, h, w // 2, w),
        ]

        tile_folds = []

        for fold_name, r0, r1, c0, c1 in folds:

            test_mask = np.zeros(
                truth.shape,
                dtype=bool,
            )

            test_mask[r0:r1, c0:c1] = True

            valid = (
                np.isfinite(dav2)
                & np.isfinite(building)
                & np.isfinite(truth)
                & (truth >= 0)
                & (building >= 0)
                & (building <= 1)
            )

            train_mask = valid & ~test_mask
            test_mask = valid & test_mask

            if train_mask.sum() < 100:
                continue

            if test_mask.sum() < 100:
                continue

            # ---------------------------------------------
            # Baseline
            #
            # AGL = a * DAv2 + b
            # ---------------------------------------------

            X_train_base = dav2[train_mask].reshape(-1, 1)
            y_train = truth[train_mask]

            X_test_base = dav2[test_mask].reshape(-1, 1)
            y_test = truth[test_mask]

            base_model = LinearRegression()
            base_model.fit(
                X_train_base,
                y_train,
            )

            base_pred = base_model.predict(
                X_test_base
            )

            baseline_metrics = metrics(
                y_test,
                base_pred,
            )

            # ---------------------------------------------
            # Method 3
            #
            # AGL =
            # beta0
            # + beta1*DAv2
            # + beta2*Pbuilding
            # + beta3*(DAv2*Pbuilding)
            # ---------------------------------------------

            X_train_semantic = build_features(
                dav2[train_mask],
                building[train_mask],
            )

            X_test_semantic = build_features(
                dav2[test_mask],
                building[test_mask],
            )

            semantic_model = LinearRegression()

            semantic_model.fit(
                X_train_semantic,
                y_train,
            )

            semantic_pred = semantic_model.predict(
                X_test_semantic
            )

            semantic_metrics = metrics(
                y_test,
                semantic_pred,
            )

            tile_folds.append(
                {
                    "fold": fold_name,
                    "baseline": baseline_metrics,
                    "method3_semantic": semantic_metrics,
                }
            )

            print(
                f"{tile_id} | {fold_name} | "
                f"Baseline MAE="
                f"{baseline_metrics['mae_m']:.3f} | "
                f"Method3 MAE="
                f"{semantic_metrics['mae_m']:.3f}"
            )

        if tile_folds:
            results.append(
                {
                    "tile": tile_id,
                    "folds": tile_folds,
                }
            )

    if not results:
        raise RuntimeError(
            "No tiles were evaluated."
        )

    # -----------------------------------------------------
    # Aggregate every held-out fold
    # -----------------------------------------------------

    baseline_all = []
    semantic_all = []

    for tile in results:
        for fold in tile["folds"]:
            baseline_all.append(
                fold["baseline"]
            )
            semantic_all.append(
                fold["method3_semantic"]
            )

    def aggregate(records):
        return {
            "mae_m": float(
                np.mean(
                    [r["mae_m"] for r in records]
                )
            ),
            "rmse_m": float(
                np.mean(
                    [r["rmse_m"] for r in records]
                )
            ),
            "pearson": float(
                np.mean(
                    [r["pearson"] for r in records]
                )
            ),
            "spearman": float(
                np.mean(
                    [r["spearman"] for r in records]
                )
            ),
            "n_folds": len(records),
        }

    baseline_summary = aggregate(
        baseline_all
    )

    semantic_summary = aggregate(
        semantic_all
    )

    output = {
        "experiment": (
            "DFC2019 Method 3 spatial 4-fold CV"
        ),
        "protocol": (
            "Each tile is split into 4 spatial quadrants. "
            "Three quadrants train the model and the "
            "remaining quadrant is held out. "
            "All four folds are evaluated."
        ),
        "n_tiles": len(results),
        "baseline": baseline_summary,
        "method3_semantic": semantic_summary,
        "tiles": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(output, indent=2)
    )

    print("\n" + "=" * 70)
    print("SPATIAL HOLDOUT RESULT")
    print("=" * 70)

    print(
        f"Tiles evaluated: {len(results)}"
    )

    print("\nBaseline:")
    print(
        f"  MAE      = "
        f"{baseline_summary['mae_m']:.4f} m"
    )
    print(
        f"  RMSE     = "
        f"{baseline_summary['rmse_m']:.4f} m"
    )
    print(
        f"  Pearson  = "
        f"{baseline_summary['pearson']:.4f}"
    )
    print(
        f"  Spearman = "
        f"{baseline_summary['spearman']:.4f}"
    )

    print("\nMethod 3:")
    print(
        f"  MAE      = "
        f"{semantic_summary['mae_m']:.4f} m"
    )
    print(
        f"  RMSE     = "
        f"{semantic_summary['rmse_m']:.4f} m"
    )
    print(
        f"  Pearson  = "
        f"{semantic_summary['pearson']:.4f}"
    )
    print(
        f"  Spearman = "
        f"{semantic_summary['spearman']:.4f}"
    )

    print("\nSaved:")
    print(args.output)


if __name__ == "__main__":
    main()
