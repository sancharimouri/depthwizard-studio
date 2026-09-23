"""Task 2: residual correction test. Baseline is the already-calibrated,
geoid-corrected SRTM prediction (Task 1's primary elevation source) for
each of the 25 sign-flip-detector-accepted tiles. Tests whether a SIMPLE
correction (constant offset, or a single linear term) fit on a small
subset of real ICESat-2 points (k=10 or k=20, matching Method 2's
DFC2019 anchor counts for comparability) improves accuracy on the
held-out remainder of that tile's photons. No spatial/quadrant-local
variants -- Method 2 already showed locality makes things worse for this
kind of correction.

Because a k=10/20 random draw is inherently high-variance, each (tile, k,
method) combination is repeated 30 times with different random anchor
draws and the MEDIAN outcome is reported, not a single lucky/unlucky
draw.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import stats

from lib_backbone_correlation import sample_depth_at_photons

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
SRTM_DIR = ROOT / "data/sentinel2_benchmark/srtm_raw"
PHOTON_DIR = ROOT / "data/icesat2_photons"
VERDICTS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
K_VALUES = [10, 20]
N_REPEATS = 30
BASE_SEED = 42

_T_EGM96 = Transformer.from_crs("EPSG:4979", "EPSG:5773", always_xy=True)


def n_egm96(lon, lat):
    _, _, n = _T_EGM96.transform(lon, lat, 0.0)
    return n


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def fit_srtm_baseline(tile_id, rgb_path):
    """Refit the large-sample SRTM calibration exactly as in
    run_srtm_comparison.py (50/50 split, seed 42) -- this IS the Task 1
    primary-source baseline, unmodified."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    dav2 = np.load(DAV2_DIR / f"{tile_id}_depth.npy")
    with rasterio.open(SRTM_DIR / f"{tile_id}_srtm.tif") as src:
        dem = src.read(1).astype(np.float32)
        src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata
    with rasterio.open(rgb_path) as rgb_src:
        dst_crs, dst_transform, dst_shape = rgb_src.crs, rgb_src.transform, (rgb_src.height, rgb_src.width)

    dem_on_grid = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=dem, destination=dem_on_grid,
        src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    valid = ~np.isnan(dem_on_grid)
    idx = np.flatnonzero(valid)
    rng = np.random.RandomState(BASE_SEED)
    rng.shuffle(idx)
    train_idx = idx[: len(idx) // 2]
    depth_flat, dem_flat = dav2.ravel(), dem_on_grid.ravel()
    lr = stats.linregress(depth_flat[train_idx], dem_flat[train_idx])
    return lr.slope, lr.intercept, dav2


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()

    rows = []
    for tile_id in accepted:
        m = manifest.loc[tile_id]
        rgb_path = ROOT / m["rgb_path"]
        n = n_egm96(m["lon"], m["lat"])
        a, b, dav2 = fit_srtm_baseline(tile_id, rgb_path)

        photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
        sampled = sample_depth_at_photons(rgb_path, dav2, photons)
        depth_val = sampled["depth_value"].values
        true_height = sampled["height"].values
        baseline_pred = a * depth_val + b - n
        n_total = len(sampled)

        rec = {"tile_id": tile_id, "category": m["category"], "n_photons": n_total,
               "baseline_rmse_m": rmse(baseline_pred, true_height)}

        for k in K_VALUES:
            const_rmses, linear_rmses, baseline_heldout_rmses = [], [], []
            for rep in range(N_REPEATS):
                rng = np.random.RandomState(BASE_SEED + rep)
                anchor_idx = rng.choice(n_total, size=k, replace=False)
                held_mask = np.ones(n_total, dtype=bool)
                held_mask[anchor_idx] = False
                held_idx = np.flatnonzero(held_mask)

                baseline_heldout_rmses.append(rmse(baseline_pred[held_idx], true_height[held_idx]))

                # constant-offset correction: bias measured on the k anchors, applied to held-out
                offset = np.mean(true_height[anchor_idx] - baseline_pred[anchor_idx])
                const_rmses.append(rmse(baseline_pred[held_idx] + offset, true_height[held_idx]))

                # single linear term: OLS fit on ONLY the k anchors (sparse-anchor regression,
                # Method 2's approach), replacing the DEM-based fit entirely for this test
                lr2 = stats.linregress(depth_val[anchor_idx], true_height[anchor_idx])
                pred_linear = lr2.slope * depth_val[held_idx] + lr2.intercept
                linear_rmses.append(rmse(pred_linear, true_height[held_idx]))

            rec[f"baseline_heldout_rmse_k{k}"] = float(np.median(baseline_heldout_rmses))
            rec[f"const_corrected_rmse_k{k}"] = float(np.median(const_rmses))
            rec[f"linear_corrected_rmse_k{k}"] = float(np.median(linear_rmses))
            rec[f"const_improves_k{k}"] = rec[f"const_corrected_rmse_k{k}"] < rec[f"baseline_heldout_rmse_k{k}"]
            rec[f"linear_improves_k{k}"] = rec[f"linear_corrected_rmse_k{k}"] < rec[f"baseline_heldout_rmse_k{k}"]

        rows.append(rec)
        print(
            f"{tile_id:15s} baseline={rec['baseline_rmse_m']:6.1f}m  "
            f"k=10: const={rec['const_corrected_rmse_k10']:6.1f}m({'better' if rec['const_improves_k10'] else 'worse'})  "
            f"linear={rec['linear_corrected_rmse_k10']:7.1f}m({'better' if rec['linear_improves_k10'] else 'worse'})  |  "
            f"k=20: const={rec['const_corrected_rmse_k20']:6.1f}m({'better' if rec['const_improves_k20'] else 'worse'})  "
            f"linear={rec['linear_corrected_rmse_k20']:7.1f}m({'better' if rec['linear_improves_k20'] else 'worse'})"
        )

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/sentinel2_benchmark/residual_correction_results.csv", index=False)

    print("\n=== Aggregate ===")
    for k in K_VALUES:
        print(f"k={k}: constant-offset improves on {df[f'const_improves_k{k}'].sum()}/{len(df)} tiles; "
              f"linear improves on {df[f'linear_improves_k{k}'].sum()}/{len(df)} tiles")
        print(f"  median baseline RMSE: {df[f'baseline_heldout_rmse_k{k}'].median():.2f}m")
        print(f"  median const-corrected RMSE: {df[f'const_corrected_rmse_k{k}'].median():.2f}m")
        print(f"  median linear-corrected RMSE: {df[f'linear_corrected_rmse_k{k}'].median():.2f}m")

    print(f"\nWrote {ROOT / 'data/sentinel2_benchmark/residual_correction_results.csv'}")


if __name__ == "__main__":
    main()
