"""Frequency-fusion elevation correction on the 25 sign-flip-accepted
Sentinel-2 tiles (blakc-coffee/depthwizard's technique from the
competitive-repo audit, adapted onto our own data and evaluation
methodology -- read, not executed, from that repo).

Pure signal processing, nothing learned: real DEM low-frequency trend +
DAv2 high-frequency detail, matched-filter subtraction. No CNN, no
training data, so Method 4's interpolation-memorization failure mode
(a network learning to reproduce the DEM's own resampling artifacts)
cannot apply here by construction -- there is nothing that can memorize.

Pipeline, in the order the task requires:

1. METRIC SCALING FIRST. Reuses the EXACT SAME per-tile calibration this
   project already validated and reports elsewhere (run_srtm_comparison.py
   / calibrate_accepted_tiles.py): a random 50/50 split of the DEM's valid
   pixels (seed 42), OLS fit `dem = a*dav2_raw + b` on the train half.
   `dav2_metric = a*dav2_raw + b` is then in the SAME vertical datum as
   the raw SRTM DEM itself (EGM96 orthometric, uncorrected) -- consistent
   with, not derived independently of, the working baseline. Frequency-
   splitting raw [0,1] DAv2 output directly (the prior mistake) would have
   no metric meaning to preserve; this fixes that.

2. CONFIRMED NATIVE RESOLUTION, not assumed. SRTM here is 1-arcsecond
   (verified: pixel size is exactly 0.0002777... degrees in the raw
   files) -- but 1 arcsecond is NOT a fixed number of metres; it shrinks
   in the east-west direction as cos(latitude), and our 25 tiles span
   roughly 9.9degN (kochi_city) to 30.3degN (bathinda), where cos(lat)
   ranges from 0.985 to 0.863 -- a genuine ~12% swing in real ground
   sampling distance. Computed per-tile via pyproj.Geod geodesic
   distance at each tile's actual centre latitude (not a rule-of-thumb
   30m), for both axes; the scalar "native resolution" used for the
   filter is their geometric mean.

3. MATCHED LOW-PASS. A masked (NaN-aware) Gaussian filter with sigma set
   from step 2's per-tile native resolution (in pixels of the 10m UTM
   grid both signals already share) is applied to BOTH dem_on_grid and
   dav2_metric -- the identical kernel on both, which is the part that is
   not optional. dav2_highpass = dav2_metric - gaussian(dav2_metric).

4. COMBINE. final_egm96 = gaussian(dem_on_grid) + dav2_highpass.

5. CONFIDENCE MAP. Per-pixel: DEM-covered (1) vs. gap/no-data (0), from
   dem_on_grid's own validity mask before any filtering.

6. EVALUATION -- identical to the existing linear-calibration baseline,
   same seed/split/metric, so this is a direct comparison, not approximate:
   - DEM check: RMSE/MAE/Pearson of final_egm96 vs. dem_on_grid on the
     HELD-OUT 50% test pixels (same split as the calibration fit).
     Read this number with real caution (see report) -- the low-frequency
     component is derived from the DEM's own full grid (train+test
     pixels), which is the whole point of "trust the DEM at coarse
     scale," but it means this check is not a clean generalisation test
     the way it is for the pure-linear baseline. Report it anyway, but do
     not treat it as the deciding number.
   - ICESat-2 check: sample final_egm96 (converted to WGS84 ellipsoidal
     via the same EGM96 geoid correction, `- n_geoid`) at real photon
     locations, RMSE and RMSE-as-%-of-elev-range -- this IS the fair,
     independent check, exactly as elsewhere in this project's Sentinel-2
     work, and the one the verdict should rest on.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Must be set before pyproj builds any transformer -- without it, PROJ silently
# falls back to a "ballpark" EGM96 transform that returns N=0 everywhere rather
# than fetching the real geoid grid, which would have quietly reintroduced a
# ~40-100m systematic bias into every ICESat-2 comparison below. Caught by
# smoke-testing on bathinda: N came back 0.0 instead of the real +46.68m this
# project's own srtm_3way_comparison.csv already has on record for that tile.
os.environ.setdefault("PROJ_NETWORK", "ON")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod, Transformer
from rasterio.warp import Resampling, reproject
from scipy import stats
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_backbone_correlation import sample_depth_at_photons  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
SRTM_DIR = ROOT / "data/sentinel2_benchmark/srtm_raw"
PHOTON_DIR = ROOT / "data/icesat2_photons"
VERDICTS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
OUT_DIR = ROOT / "data/sentinel2_benchmark/frequency_fusion_results"
GRID_RES_M = 10.0  # the RGB/UTM grid both signals are reprojected onto
SEED = 42

_T_EGM96 = Transformer.from_crs("EPSG:4979", "EPSG:5773", always_xy=True)
_GEOD = Geod(ellps="WGS84")


def n_egm96(lon, lat):
    _, _, n = _T_EGM96.transform(lon, lat, 0.0)
    return n


def native_resolution_m(lat: float, lon: float) -> tuple[float, float, float]:
    """(lat_spacing_m, lon_spacing_m, geometric_mean_m) for one 1-arcsecond
    SRTM pixel at this tile's actual location -- computed geodesically,
    not assumed. lon spacing shrinks as cos(lat); lat spacing is nearly
    but not exactly constant on the WGS84 ellipsoid."""
    arcsec = 1.0 / 3600.0
    _, _, lat_dist = _GEOD.inv(lon, lat, lon, lat + arcsec)
    _, _, lon_dist = _GEOD.inv(lon, lat, lon + arcsec, lat)
    return lat_dist, lon_dist, float(np.sqrt(lat_dist * lon_dist))


def masked_gaussian(x: np.ndarray, valid: np.ndarray, sigma_px: float) -> np.ndarray:
    """NaN-aware Gaussian blur: convolve(x*mask)/convolve(mask), so
    invalid/gap pixels never contaminate the smoothed neighbourhood."""
    xf = np.where(valid, x, 0.0).astype(np.float64)
    mf = valid.astype(np.float64)
    num = gaussian_filter(xf, sigma=sigma_px, mode="nearest")
    den = gaussian_filter(mf, sigma=sigma_px, mode="nearest")
    out = np.full_like(x, np.nan, dtype=np.float64)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out.astype(np.float32)


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred, true) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


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

    # ---- step 1: metric scaling, same split/fit as the existing baseline
    idx = np.flatnonzero(valid)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    half = len(idx) // 2
    train_idx, test_idx = idx[:half], idx[half:]

    depth_flat, dem_flat = dav2_raw.ravel(), dem_on_grid.ravel()
    lr = stats.linregress(depth_flat[train_idx], dem_flat[train_idx])
    a, b = lr.slope, lr.intercept
    dav2_metric = (a * dav2_raw + b).astype(np.float32)  # EGM96-equivalent, same datum as dem_on_grid

    # ---- step 2: confirmed native resolution (geodesic, per-tile latitude)
    lat_m, lon_m, res_m = native_resolution_m(lat, lon)
    sigma_px = res_m / GRID_RES_M

    # ---- step 3+4: matched low-pass, subtract, combine
    dem_lowpass = masked_gaussian(dem_on_grid, valid, sigma_px)
    dav2_lowpass = masked_gaussian(dav2_metric, np.ones_like(valid, dtype=bool), sigma_px)
    dav2_highpass = dav2_metric - dav2_lowpass
    final_egm96 = dem_lowpass + dav2_highpass

    # ---- step 6: evaluation, identical format to the linear baseline
    pred_test, true_test = final_egm96.ravel()[test_idx], dem_flat[test_idx]
    ok = np.isfinite(pred_test) & np.isfinite(true_test)
    dem_pearson, _ = stats.pearsonr(pred_test[ok], true_test[ok])
    dem_rmse_val, dem_mae_val = rmse(pred_test[ok], true_test[ok]), mae(pred_test[ok], true_test[ok])

    n = n_egm96(lon, lat)
    final_ellipsoidal = final_egm96 - n
    photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
    sampled = sample_depth_at_photons(rgb_path, final_ellipsoidal, photons)
    ice_rmse = rmse(sampled["depth_value"].values, sampled["height"].values)
    elev_range = float(dem_on_grid[valid].max() - dem_on_grid[valid].min())

    coverage_pct = 100.0 * valid.mean()

    return {
        "tile_id": tile_id, "lat": lat, "lon": lon,
        "lat_arcsec_m": lat_m, "lon_arcsec_m": lon_m, "native_res_m": res_m, "sigma_px": sigma_px,
        "a_slope": a, "b_intercept": b,
        "held_out_dem_rmse_m": dem_rmse_val, "held_out_dem_mae_m": dem_mae_val, "held_out_dem_pearson": dem_pearson,
        "icesat2_rmse_m": ice_rmse, "elev_range_m": elev_range,
        "icesat2_rmse_pct_of_range": 100 * ice_rmse / elev_range,
        "n_icesat2_photons": len(sampled), "dem_coverage_pct": coverage_pct,
    }


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    print(f"{len(accepted)} accepted tiles")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, tile_id in enumerate(accepted, 1):
        m = manifest.loc[tile_id]
        r = process_tile(tile_id, ROOT / m["rgb_path"], float(m["lat"]), float(m["lon"]))
        r["category"] = m["category"]
        rows.append(r)
        print(f"[{i}/{len(accepted)}] {tile_id:15s} native_res={r['native_res_m']:.2f}m "
              f"(lat {r['lat_arcsec_m']:.2f}m, lon {r['lon_arcsec_m']:.2f}m) sigma_px={r['sigma_px']:.2f} "
              f"DEM_rmse={r['held_out_dem_rmse_m']:.2f}m ICE_rmse={r['icesat2_rmse_m']:.2f}m "
              f"({r['icesat2_rmse_pct_of_range']:.2f}% of range)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "frequency_fusion_results.csv", index=False)

    print(f"\n{'='*72}\nSUMMARY (n={len(df)})\n{'='*72}")
    print(f"median native_res_m: {df['native_res_m'].median():.2f} (range {df['native_res_m'].min():.2f}-{df['native_res_m'].max():.2f})")
    print(f"median icesat2_rmse_pct_of_range: {df['icesat2_rmse_pct_of_range'].median():.2f}%")
    print(f"mean dem_coverage_pct: {df['dem_coverage_pct'].mean():.2f}%")
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "n_tiles": len(df),
        "median_native_res_m": float(df["native_res_m"].median()),
        "median_icesat2_rmse_pct_of_range": float(df["icesat2_rmse_pct_of_range"].median()),
        "mean_dem_coverage_pct": float(df["dem_coverage_pct"].mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
