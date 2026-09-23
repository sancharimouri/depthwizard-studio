"""Per-tile DEM-anchored calibration for DAv2, ONLY on tiles the sign-flip
detector (detect_sign_flip.py / evaluate_sign_flip_detector.py) does NOT
flag as inverted. Rejected tiles get "not calibrated, flagged for review"
instead of a number -- calibrating a tile with an inverted raw signal
would just fit a confidently-wrong line.

Method for each accepted tile (same held-out discipline as
backbone-terrain-analysis.md's Follow-up 1 -- fit and evaluate on disjoint
halves, not the same data):
  1. Fit OLS (Copernicus DEM elevation ~ a*dav2_depth + b) on a random 50%
     of that tile's valid DEM pixels (seed 42).
  2. Evaluate the fitted line's HELD-OUT accuracy on the other 50% of DEM
     pixels (RMSE/MAE in meters, plus Pearson for comparability with
     Prompt 1's numbers -- note Pearson is invariant to an affine
     transform of one variable, so it will match sign_flip's raw
     dav2_vs_dem_pearson; RMSE/MAE are the numbers that actually depend on
     the fit).
  3. Where ICESat-2 ground photons exist for that tile (all 32 do), apply
     the SAME DEM-half-fitted (a, b) -- never fit on any ICESat-2 data --
     to DAv2 depth sampled at every photon location, and score against
     real photon height. This is the stronger independent check: ICESat-2
     is a ~decimeter-class per-photon reference (modulo ~6.5m geolocation
     noise, per backbone-comparison.md), vs. Copernicus GLO-30's own
     ~10-16m absolute accuracy.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats

from lib_backbone_correlation import sample_depth_at_photons
from detect_sign_flip import reproject_dem_to_tile_grid

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
DEM_DIR = ROOT / "data/sentinel2_benchmark/copernicus_dem_raw"
PHOTON_DIR = ROOT / "data/icesat2_photons"
VERDICTS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
OUT_CSV = ROOT / "data/sentinel2_benchmark/calibration_results.csv"

SEED = 42


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def calibrate_tile(tile_id: str, rgb_path: Path) -> dict:
    dav2 = np.load(DAV2_DIR / f"{tile_id}_depth.npy")
    dem_on_grid = reproject_dem_to_tile_grid(DEM_DIR / f"{tile_id}_dem.tif", rgb_path)

    valid = ~np.isnan(dem_on_grid)
    idx = np.flatnonzero(valid)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    half = len(idx) // 2
    train_idx, test_idx = idx[:half], idx[half:]

    depth_flat = dav2.ravel()
    dem_flat = dem_on_grid.ravel()

    lr = stats.linregress(depth_flat[train_idx], dem_flat[train_idx])
    a, b = lr.slope, lr.intercept

    pred_test = a * depth_flat[test_idx] + b
    true_test = dem_flat[test_idx]
    dem_pearson, _ = stats.pearsonr(pred_test, true_test)
    dem_rmse = rmse(pred_test, true_test)
    dem_mae = mae(pred_test, true_test)

    result = {
        "tile_id": tile_id,
        "status": "calibrated",
        "a_slope": a,
        "b_intercept": b,
        "n_train_px": len(train_idx),
        "n_test_px": len(test_idx),
        "held_out_dem_pearson": dem_pearson,
        "held_out_dem_rmse_m": dem_rmse,
        "held_out_dem_mae_m": dem_mae,
    }

    photon_path = PHOTON_DIR / f"{tile_id}.csv"
    if photon_path.exists():
        photons = pd.read_csv(photon_path)
        sampled = sample_depth_at_photons(rgb_path, dav2, photons)
        pred_icesat2 = a * sampled["depth_value"].values + b
        true_icesat2 = sampled["height"].values
        ice_pearson, _ = stats.pearsonr(pred_icesat2, true_icesat2)
        result.update({
            "n_icesat2_photons": len(sampled),
            "icesat2_pearson": ice_pearson,
            "icesat2_rmse_m": rmse(pred_icesat2, true_icesat2),
            "icesat2_mae_m": mae(pred_icesat2, true_icesat2),
        })
    else:
        result.update({
            "n_icesat2_photons": 0,
            "icesat2_pearson": float("nan"),
            "icesat2_rmse_m": float("nan"),
            "icesat2_mae_m": float("nan"),
        })

    return result


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")

    rows = []
    for tile_id, row in manifest.iterrows():
        flagged = bool(verdicts.loc[tile_id, "flagged"])
        rgb_path = ROOT / row["rgb_path"]

        if flagged:
            rows.append({
                "tile_id": tile_id, "category": row["category"], "status": "REJECTED",
                "a_slope": float("nan"), "b_intercept": float("nan"),
                "n_train_px": 0, "n_test_px": 0,
                "held_out_dem_pearson": float("nan"), "held_out_dem_rmse_m": float("nan"), "held_out_dem_mae_m": float("nan"),
                "n_icesat2_photons": 0, "icesat2_pearson": float("nan"), "icesat2_rmse_m": float("nan"), "icesat2_mae_m": float("nan"),
            })
            print(f"{tile_id:15s} REJECTED (flagged for review, not calibrated)")
            continue

        result = calibrate_tile(tile_id, rgb_path)
        result["category"] = row["category"]
        rows.append(result)
        print(
            f"{tile_id:15s} CALIBRATED  dem: pearson={result['held_out_dem_pearson']:+.4f} "
            f"rmse={result['held_out_dem_rmse_m']:6.1f}m mae={result['held_out_dem_mae_m']:6.1f}m  |  "
            f"icesat2: pearson={result['icesat2_pearson']:+.4f} rmse={result['icesat2_rmse_m']:6.1f}m mae={result['icesat2_mae_m']:6.1f}m "
            f"(n={result['n_icesat2_photons']})"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
