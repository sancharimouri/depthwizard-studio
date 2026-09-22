"""Semantic-prior phase 2.3, Step 4 -- building-aware frequency fusion.

Extends run_frequency_fusion_sentinel2.py (the current standing-best,
non-learned Sentinel-2 method: real DEM low-frequency trend + DAv2
high-frequency detail, matched-filter subtraction; 21/25 tiles beat plain
linear calibration, median ICESat-2 error 10.88% -> 3.57%) with two building-
aware integrations of the semantic-prior data prepared in Step 2/3
(data/sentinel2_benchmark/semantic/<tile>/{building_fraction,ob_height,
ob_presence}.tif). Reuses that script's exact helpers (masked_gaussian,
native_resolution_m, n_egm96, rmse, mae) rather than reimplementing them --
this script does NOT modify run_frequency_fusion_sentinel2.py, which stays
the reproducible standing baseline.

NO CNN. Both approaches below are closed-form combinations of already-
computed signals -- nothing is trained, so Method 4's interpolation-
memorization failure mode structurally cannot apply here either.

Approach A -- building-aware confidence weighting:
    final_A = dem_lowpass + (1 + alpha * building_prob) * dav2_highpass
  alpha=1.0 fixed a priori (buildings get up to 2x DAv2 high-frequency
  weight) -- NOT tuned against the ICESat-2 test metric, to avoid the same
  test-set-fitting risk this project's evidence-gating/LOBO work elsewhere
  has deliberately guarded against. building_prob = Step 2's
  building_fraction where available and not gap-excluded, else Step 3's OB
  presence (fractional_count band) as the fallback signal.

Approach B -- direct supplementary height source, at confident building
pixels only (OB presence > CONFIDENT_THRESHOLD=0.5, matching
prepare_method4_openbuildings_data.py's own threshold):
    final_B = 0.5*(dem_lowpass + dav2_highpass) + 0.5*(dem_lowpass + ob_height)
            = dem_lowpass + 0.5*dav2_highpass + 0.5*ob_height
  elsewhere unchanged (= dem_lowpass + dav2_highpass, the existing
  baseline). An equal blend of two independent height estimates (DAv2's
  detail signal and Open Buildings' own measured AGL height) at pixels
  where OB actually has a confident reading; one concrete formula, not
  grid-searched.

Evaluated with the identical held-out-DEM (50% split, seed 42) +
independent-ICESat-2 protocol as the baseline script, on all 25 accepted
tiles.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "ON")

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_frequency_fusion_sentinel2 import (  # noqa: E402
    masked_gaussian, native_resolution_m, n_egm96, rmse, mae,
    MANIFEST, DAV2_DIR, SRTM_DIR, PHOTON_DIR, VERDICTS_CSV, GRID_RES_M, SEED,
)
from lib_backbone_correlation import sample_depth_at_photons  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEMANTIC_DIR = ROOT / "data/sentinel2_benchmark/semantic"
OUT_DIR = ROOT / "data/sentinel2_benchmark/frequency_fusion_semantic_results"
ALPHA = 1.0
CONFIDENT_THRESHOLD = 0.5


def _read_on_grid(path: Path, dst_crs, dst_transform, dst_shape) -> np.ndarray | None:
    """Read a raster and reproject onto the tile's RGB/UTM grid; returns
    None if the file doesn't exist (tile excluded/no-coverage cases)."""
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata
    if arr.shape == dst_shape and src_transform == dst_transform and str(src_crs) == str(dst_crs):
        out = arr
    else:
        out = np.full(dst_shape, np.nan, dtype=np.float32)
        reproject(source=arr, destination=out, src_transform=src_transform, src_crs=src_crs,
                 src_nodata=src_nodata, dst_transform=dst_transform, dst_crs=dst_crs,
                 dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def process_tile(tile_id: str, rgb_path: Path, lat: float, lon: float) -> dict:
    dav2_raw = np.load(DAV2_DIR / f"{tile_id}_depth.npy").astype(np.float32)
    with rasterio.open(SRTM_DIR / f"{tile_id}_srtm.tif") as src:
        dem = src.read(1).astype(np.float32)
        src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata
    with rasterio.open(rgb_path) as rgb_src:
        dst_crs, dst_transform, dst_shape = rgb_src.crs, rgb_src.transform, (rgb_src.height, rgb_src.width)

    dem_on_grid = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(source=dem, destination=dem_on_grid, src_transform=src_transform, src_crs=src_crs,
             src_nodata=src_nodata, dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
             resampling=Resampling.bilinear)
    valid = ~np.isnan(dem_on_grid)

    idx = np.flatnonzero(valid)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    half = len(idx) // 2
    train_idx, test_idx = idx[:half], idx[half:]

    depth_flat, dem_flat = dav2_raw.ravel(), dem_on_grid.ravel()
    lr = stats.linregress(depth_flat[train_idx], dem_flat[train_idx])
    a, b = lr.slope, lr.intercept
    dav2_metric = (a * dav2_raw + b).astype(np.float32)

    lat_m, lon_m, res_m = native_resolution_m(lat, lon)
    sigma_px = res_m / GRID_RES_M

    dem_lowpass = masked_gaussian(dem_on_grid, valid, sigma_px)
    dav2_lowpass = masked_gaussian(dav2_metric, np.ones_like(valid, dtype=bool), sigma_px)
    dav2_highpass = dav2_metric - dav2_lowpass
    final_baseline = dem_lowpass + dav2_highpass

    # ---- semantic inputs
    sem_dir = SEMANTIC_DIR / tile_id
    building_fraction = _read_on_grid(sem_dir / "building_fraction.tif", dst_crs, dst_transform, dst_shape)
    ob_presence = _read_on_grid(sem_dir / "ob_presence.tif", dst_crs, dst_transform, dst_shape)
    ob_height = _read_on_grid(sem_dir / "ob_height.tif", dst_crs, dst_transform, dst_shape)

    if building_fraction is not None:
        building_prob = np.clip(np.nan_to_num(building_fraction, nan=0.0), 0.0, 1.0)
        building_prob_source = "step2_building_fraction"
    elif ob_presence is not None:
        building_prob = np.clip(np.nan_to_num(ob_presence, nan=0.0), 0.0, 1.0)
        building_prob_source = "step3_ob_presence_fallback"
    else:
        building_prob = np.zeros(dst_shape, dtype=np.float32)
        building_prob_source = "none_available"

    # ---- Approach A: building-aware confidence weighting
    final_A = dem_lowpass + (1.0 + ALPHA * building_prob) * dav2_highpass

    # ---- Approach B: direct supplementary height source at confident building pixels
    final_B = final_baseline.copy()
    if ob_height is not None and ob_presence is not None:
        confident = np.isfinite(ob_presence) & (ob_presence > CONFIDENT_THRESHOLD) & np.isfinite(ob_height)
        blended = dem_lowpass + 0.5 * dav2_highpass + 0.5 * ob_height
        final_B = np.where(confident, blended, final_baseline).astype(np.float32)
        confident_pct = 100.0 * confident.sum() / confident.size
    else:
        confident_pct = 0.0

    n = n_egm96(lon, lat)
    photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")

    def eval_variant(final_egm96: np.ndarray) -> dict:
        pred_test, true_test = final_egm96.ravel()[test_idx], dem_flat[test_idx]
        ok = np.isfinite(pred_test) & np.isfinite(true_test)
        dem_pearson, _ = stats.pearsonr(pred_test[ok], true_test[ok])
        dem_rmse_val, dem_mae_val = rmse(pred_test[ok], true_test[ok]), mae(pred_test[ok], true_test[ok])
        final_ellipsoidal = final_egm96 - n
        sampled = sample_depth_at_photons(rgb_path, final_ellipsoidal, photons)
        ice_rmse = rmse(sampled["depth_value"].values, sampled["height"].values)
        elev_range = float(dem_on_grid[valid].max() - dem_on_grid[valid].min())
        return {
            "held_out_dem_rmse_m": dem_rmse_val, "held_out_dem_mae_m": dem_mae_val,
            "held_out_dem_pearson": dem_pearson, "icesat2_rmse_m": ice_rmse, "elev_range_m": elev_range,
            "icesat2_rmse_pct_of_range": 100 * ice_rmse / elev_range, "n_icesat2_photons": len(sampled),
        }

    base_eval = eval_variant(final_baseline)
    a_eval = eval_variant(final_A)
    b_eval = eval_variant(final_B)

    return {
        "tile_id": tile_id, "lat": lat, "lon": lon,
        "building_prob_source": building_prob_source, "ob_confident_pct": confident_pct,
        **{f"base_{k}": v for k, v in base_eval.items()},
        **{f"A_{k}": v for k, v in a_eval.items()},
        **{f"B_{k}": v for k, v in b_eval.items()},
    }


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    print(f"{len(accepted)} accepted tiles")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, tile_id in enumerate(sorted(accepted), 1):
        m = manifest.loc[tile_id]
        r = process_tile(tile_id, ROOT / m["rgb_path"], float(m["lat"]), float(m["lon"]))
        r["category"] = m["category"]
        rows.append(r)
        print(f"[{i}/{len(accepted)}] {tile_id:15s} src={r['building_prob_source']:28s} "
              f"base={r['base_icesat2_rmse_pct_of_range']:.2f}% "
              f"A={r['A_icesat2_rmse_pct_of_range']:.2f}% "
              f"B={r['B_icesat2_rmse_pct_of_range']:.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results.csv", index=False)

    # linear-vs-fusion comparison already lives in sign-flip-detector.md; here we compare A/B to the
    # frequency-fusion baseline this script itself recomputed (base_*), consistent by construction.
    wins_A = (df["A_icesat2_rmse_pct_of_range"] < df["base_icesat2_rmse_pct_of_range"]).sum()
    wins_B = (df["B_icesat2_rmse_pct_of_range"] < df["base_icesat2_rmse_pct_of_range"]).sum()
    print(f"\n{'='*72}\nSUMMARY (n={len(df)}) vs. this run's own frequency-fusion baseline\n{'='*72}")
    print(f"Approach A wins on {wins_A}/{len(df)} tiles vs. baseline; median base={df['base_icesat2_rmse_pct_of_range'].median():.2f}% "
          f"median A={df['A_icesat2_rmse_pct_of_range'].median():.2f}%")
    print(f"Approach B wins on {wins_B}/{len(df)} tiles vs. baseline; median base={df['base_icesat2_rmse_pct_of_range'].median():.2f}% "
          f"median B={df['B_icesat2_rmse_pct_of_range'].median():.2f}%")

    focus = ["bathinda", "amalapuram", "kutch", "chennai"]
    print("\nFocus tiles (currently losing to linear baseline per sign-flip-detector.md):")
    print(df[df.tile_id.isin(focus)][["tile_id", "base_icesat2_rmse_pct_of_range",
                                        "A_icesat2_rmse_pct_of_range", "B_icesat2_rmse_pct_of_range"]].to_string())


if __name__ == "__main__":
    main()
