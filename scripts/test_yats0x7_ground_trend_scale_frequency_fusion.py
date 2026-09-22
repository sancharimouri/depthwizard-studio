"""yats0x7/DepthWizard vs. blakc-coffee/depthwizard (the source of this
project's frequency fusion): a code-read comparison, plus a targeted test
of the one concretely different, cheap-to-test idea it surfaces for the
flat-terrain losses frequency fusion already has.

Cloned fresh for this task (list entry #7, "interesting" -- flagged much
earlier as a similar hybrid-fusion idea never compared head-to-head).
Read directly (never executed): external/yats0x7-depthwizard/engine/
depthwizard/calibrate/fit.py.

## Architectural comparison (code-read only, the core ask)

Both are structurally the same idea at the top level -- DSM = DEM's own
low-frequency terrain + a scaled high-frequency "structure"/detail band
from the depth model -- but differ in two concrete ways:

1. **What the "structure"/detail band is extracted relative to.** This
   project's frequency fusion (run_frequency_fusion_sentinel2.py):
   `dav2_highpass = dav2_metric - gaussian_blur(dav2_metric)` -- a plain,
   SYMMETRIC Gaussian low-pass. yats0x7's `ground_trend()`: an asymmetric,
   iteratively-reweighted low-pass that explicitly downweights points
   *above* the current trend each iteration (`above_weight=0.08`) so the
   trend converges toward ground rather than being pulled up by
   buildings/trees -- their own docstring names exactly the bias a plain
   Gaussian has ("a plain Gaussian pulls the terrain trend up around
   them"). This is a real, previously-undiscussed critique of this
   project's own detail-extraction step, not just a stylistic difference.

2. **What the scale factor is fit against.** This project fits `a` via
   OLS between the DEM and the model's RAW, unfiltered signal
   (`dem ~ a*dav2_raw + b`) -- exactly the fit this document's
   known-height-scale test found has near-zero correlation with true
   relief at the pixel level on flat terrain (r=-0.005, bathinda).
   yats0x7 fits `a` via RANSAC between the DEM and the model's OWN ROBUST
   GROUND TREND (`dem ~ a*ground_trend(rel) + b`) -- two smooth,
   already-denoised signals, not raw-vs-real. This is the concrete,
   directly testable idea the task points at: fitting scale between two
   trend signals should be far less exposed to the raw-pixel noise floor
   that made the known-height test's single-anchor approach fail
   catastrophically on exactly these same flat tiles.

Both differences point the same direction and are cheap to test together
(no training, pure signal processing, same as frequency fusion itself),
so this test ports both as one combined "yats0x7-style" swap of
frequency fusion's step-1 detail-extraction-and-scale-fit, keeping this
project's own validated DEM low-pass and combination step unchanged (same
"swap one targeted component" pattern as every other test in this
document).

Everything else (native-resolution matched low-pass on the DEM, the
DEM+detail combination, the evaluation protocol) is reused unchanged by
importing directly from run_frequency_fusion_sentinel2.py. Held-out
discipline: yats0x7's own ransac_affine fits on every valid pixel with no
reserved half (RANSAC's inlier detection is its own robustness
mechanism) -- for a fair DEM self-check against this project's own
established practice, this test still reserves the same seed=42 50/50
split and fits RANSAC on the train half only, exactly mirroring the
existing linear-OLS baseline's discipline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "ON")

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import stats
from scipy.ndimage import distance_transform_edt, gaussian_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_backbone_correlation import sample_depth_at_photons  # noqa: E402
from run_frequency_fusion_sentinel2 import (  # noqa: E402
    GRID_RES_M, SEED, MANIFEST, DAV2_DIR, SRTM_DIR, PHOTON_DIR, VERDICTS_CSV,
    masked_gaussian, native_resolution_m, n_egm96, rmse, mae,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/sentinel2_benchmark/yats0x7_ground_trend_results"


# ============================================================================
# Ported from external/yats0x7-depthwizard/engine/depthwizard/calibrate/fit.py
# ============================================================================

def fill_nearest(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    if not valid.any():
        return np.zeros_like(arr)
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def ground_trend(rel: np.ndarray, sigma: float, iters: int = 4, above_weight: float = 0.08) -> np.ndarray:
    h, w = rel.shape
    filled = fill_nearest(rel) if not np.isfinite(rel).all() else rel
    f = max(1, int(sigma / 2))
    ch, cw = max(2, int(round(h / f))), max(2, int(round(w / f)))
    coarse = cv2.resize(filled.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
    s = max(sigma / f, 0.5)
    trend = gaussian_filter(coarse, s, mode="mirror")
    for _ in range(iters):
        wgt = np.where(coarse > trend + 1e-4, above_weight, 1.0).astype(np.float32)
        trend = gaussian_filter(wgt * coarse, s, mode="mirror") / np.maximum(
            gaussian_filter(wgt, s, mode="mirror"), 1e-3
        )
    out = cv2.resize(trend, (w, h), interpolation=cv2.INTER_CUBIC) if f > 1 else trend
    out = out.astype(np.float32)
    out[~np.isfinite(rel)] = np.nan
    return out


def clip_structure(structure: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.8) -> np.ndarray:
    v = structure[np.isfinite(structure)]
    if v.size < 16:
        return structure
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    return np.clip(structure, lo, hi)


def ransac_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    from sklearn.linear_model import LinearRegression, RANSACRegressor

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 32 or np.ptp(x) < 1e-6:
        return 1.0, float(np.mean(y)) if y.size else 0.0, 0.0, int(x.size)
    if x.size > 100_000:
        sel = np.random.default_rng(0).choice(x.size, 100_000, replace=False)
        x, y = x[sel], y[sel]
    resid = max(1.0, 0.5 * float(np.std(y)))
    r = RANSACRegressor(
        LinearRegression(), residual_threshold=resid, random_state=0, min_samples=max(32, int(0.05 * x.size))
    )
    r.fit(x[:, None], y)
    a = float(r.estimator_.coef_[0])
    b = float(r.estimator_.intercept_)
    inl = r.inlier_mask_
    pred = a * x[inl] + b
    ss_res = float(np.sum((y[inl] - pred) ** 2))
    ss_tot = float(np.sum((y[inl] - y[inl].mean()) ** 2)) or 1e-9
    return a, b, max(0.0, 1 - ss_res / ss_tot), int(inl.sum())


# ============================================================================
# Test pipeline
# ============================================================================

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
    dem_flat = dem_on_grid.ravel()

    lat_m, lon_m, res_m = native_resolution_m(lat, lon)
    sigma_px = res_m / GRID_RES_M
    dem_lowpass = masked_gaussian(dem_on_grid, valid, sigma_px)  # UNCHANGED -- this project's own validated terrain baseline

    # ---- yats0x7-style step 1: robust ground-trend detail extraction + trend-vs-trend RANSAC scale
    rel_lp = ground_trend(dav2_raw, sigma_px)
    structure = clip_structure(dav2_raw - rel_lp)

    idx = np.flatnonzero(valid)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    half = len(idx) // 2
    train_idx, test_idx = idx[:half], idx[half:]

    rel_lp_flat, structure_flat = rel_lp.ravel(), structure.ravel()
    a, b, r2, n_inliers = ransac_affine(rel_lp_flat[train_idx], dem_flat[train_idx])

    # diagnostic: how much better is trend-vs-trend correlated than raw-vs-raw was (known-height test)?
    ok_train = np.isfinite(rel_lp_flat[train_idx]) & np.isfinite(dem_flat[train_idx])
    trend_pearson, _ = stats.pearsonr(rel_lp_flat[train_idx][ok_train], dem_flat[train_idx][ok_train])
    raw_pearson, _ = stats.pearsonr(dav2_raw.ravel()[train_idx][ok_train], dem_flat[train_idx][ok_train])

    final_egm96 = dem_lowpass + a * structure

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

    return {
        "tile_id": tile_id, "lat": lat, "lon": lon,
        "native_res_m": res_m, "sigma_px": sigma_px,
        "a_slope": a, "b_intercept": b, "ransac_r2": r2, "ransac_n_inliers": n_inliers,
        "trend_vs_dem_pearson": trend_pearson, "raw_vs_dem_pearson": raw_pearson,
        "held_out_dem_rmse_m": dem_rmse_val, "held_out_dem_mae_m": dem_mae_val, "held_out_dem_pearson": dem_pearson,
        "icesat2_rmse_m": ice_rmse, "elev_range_m": elev_range,
        "icesat2_rmse_pct_of_range": 100 * ice_rmse / elev_range,
        "n_icesat2_photons": len(sampled),
    }


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")

    target = sys.argv[1] if len(sys.argv) > 1 else "losing4"
    if target == "losing4":
        tile_ids = ["bathinda", "amalapuram", "kutch", "chennai"]
    elif target == "all25":
        verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
        tile_ids = verdicts[~verdicts["flagged"]].index.tolist()
    else:
        tile_ids = target.split(",")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, tile_id in enumerate(tile_ids, 1):
        m = manifest.loc[tile_id]
        r = process_tile(tile_id, ROOT / m["rgb_path"], float(m["lat"]), float(m["lon"]))
        r["category"] = m["category"]
        rows.append(r)
        print(f"[{i}/{len(tile_ids)}] {tile_id:15s} a={r['a_slope']:.4f} ransac_r2={r['ransac_r2']:.3f} "
              f"trend_pearson={r['trend_vs_dem_pearson']:.3f} (raw was {r['raw_vs_dem_pearson']:.3f}) "
              f"ICE_rmse={r['icesat2_rmse_m']:.2f}m ({r['icesat2_rmse_pct_of_range']:.2f}% of range)")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"results_{target}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
