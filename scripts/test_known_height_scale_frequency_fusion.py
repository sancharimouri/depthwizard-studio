"""Test arpitparashar06/depthwizard's non-regression, physically-anchored
scale derivation as a drop-in replacement for frequency fusion's step-1
metric scaling ONLY. Everything else in run_frequency_fusion_sentinel2.py
(native-resolution matched low-pass, DEM+detail combination, evaluation
methodology) is reused unchanged, by importing its functions directly
rather than reimplementing them.

Read, not executed, from external/arpitparashar06-depthwizard/mathsandml/
inference.py: `alpha_from_known_height(detail, known_height_m, pct=99)`
computes threshold = p99 of the detail band, top = median(detail values
>= threshold), alpha = known_height_m / top -- a single division, no
least-squares fit, anchored to ONE physical reference (their use case: a
human says "that landmark is ~40m tall"). No regression, unlike their own
alpha_from_gcps (least-squares slope) or this project's existing DEM-OLS
step -- this is the genuinely non-regression option in their file.

Substitution for our domain (their pattern -- swap a competitor's proprietary
ground-truth input for real ICESat-2 points -- already established
elsewhere in this project's Sentinel-2 track, per the task's own framing):
their single manual "known_height_m" becomes the median TRUE relief
(ICESat-2 photon height, converted to the DEM's EGM96 datum, minus the
DEM's own low-pass trend at that pixel) among the top-1% highest-relief
ICESat-2 points in a held-out TRAIN half of the tile's photons; their
"top" (the model's own reading at that same population) becomes DAv2's
RAW (unscaled) high-pass detail sampled at those exact same anchor pixels.
alpha = known_relief_m / model_detail_at_anchors. Their defensive
alpha<=0 rejection is kept as-is.

Order-of-operations change this requires (also read directly out of their
file, not invented here): their alpha is applied to the DETAIL band
itself, not to the raw signal before splitting -- so highpass MUST be
computed on raw dav2 first, then scaled, rather than (as the existing
linear-OLS step does) scaling raw dav2 before splitting. This needs no
intercept term, unlike the OLS step: the detail band is already
zero-centered by construction (raw minus its own low-pass), and the DEM
low-pass supplies the absolute vertical baseline, so only the AMPLITUDE
of the detail band needs fixing, not an additive offset.

Held-out discipline: the anchor-fitting half of each tile's ICESat-2
photons is disjoint from the half used for the ICESat-2 evaluation check,
same seed=42 as the rest of this project's split discipline -- fitting
and evaluating on the same points would be exactly the kind of leakage
this project's audit trail has flagged and fixed elsewhere (the geoid bug,
the interpolation-memorization pattern). This means the new method's
ICESat-2 check sees only half the photons the original linear-OLS
baseline's check does (that baseline never touches ICESat-2 for fitting
at all) -- noted as an asymmetry in the report, not papered over.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "ON")

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_frequency_fusion_sentinel2 import (  # noqa: E402
    GRID_RES_M, SEED, MANIFEST, DAV2_DIR, SRTM_DIR, PHOTON_DIR, VERDICTS_CSV,
    masked_gaussian, native_resolution_m, n_egm96, rmse, mae,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/sentinel2_benchmark/known_height_scale_results"
ANCHOR_PCT = 99.0  # arpitparashar06's HEIGHT_REFERENCE_PCT["tallest"] -- the only setting they measured/tuned


def alpha_from_known_height_icesat2(true_detail: np.ndarray, model_detail: np.ndarray, pct: float = ANCHOR_PCT):
    """Direct port of alpha_from_known_height's math (single division, no
    fit), with the human's one manual known_height_m replaced by the median
    TRUE relief among the top-`pct` ICESat-2 anchor points, and their
    "top" (the model's own reading at that population) computed at those
    SAME anchor pixels rather than spatially independently."""
    ok = np.isfinite(true_detail) & np.isfinite(model_detail)
    true_detail, model_detail = true_detail[ok], model_detail[ok]
    if true_detail.size < 20:
        return None, "too few valid anchor candidates"

    threshold = np.percentile(true_detail, pct)
    top_mask = true_detail >= threshold
    if top_mask.sum() == 0:
        return None, "no points above threshold"

    known_relief_m = float(np.median(true_detail[top_mask]))
    top_model = float(np.median(model_detail[top_mask]))

    if not np.isfinite(top_model) or abs(top_model) < 1e-9:
        return None, f"model reading at anchors is ~0 ({top_model})"

    alpha = known_relief_m / top_model
    if not np.isfinite(alpha) or alpha <= 0:
        return None, f"non-physical alpha={alpha:.4f} (known_relief_m={known_relief_m:.2f}, top_model={top_model:.4f})"

    return alpha, {"known_relief_m": known_relief_m, "top_model": top_model, "n_anchors": int(top_mask.sum())}


def sample_arrays_at_photons(rgb_path: Path, arrays: dict[str, np.ndarray], photons: pd.DataFrame) -> pd.DataFrame:
    """Sample several arrays (same grid as rgb_path) at each photon's pixel
    location in one pass, so every array is indexed at the identical
    row/col -- avoids any risk of two separate sample_depth_at_photons
    calls silently misaligning rows."""
    with rasterio.open(rgb_path) as src:
        crs, transform, height_px, width_px = src.crs, src.transform, src.height, src.width

    to_tile_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_tile_crs.transform(photons["lon"].values, photons["lat"].values)
    inv = ~transform
    cols, rows = inv * (x, y)
    cols, rows = np.floor(cols).astype(int), np.floor(rows).astype(int)
    in_bounds = (cols >= 0) & (cols < width_px) & (rows >= 0) & (rows < height_px)

    out = photons.copy()
    for name, arr in arrays.items():
        vals = np.full(len(photons), np.nan, dtype=np.float64)
        vals[in_bounds] = arr[rows[in_bounds], cols[in_bounds]]
        out[name] = vals
    return out[in_bounds].reset_index(drop=True)


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

    # native resolution / matched low-pass -- UNCHANGED from run_frequency_fusion_sentinel2.py
    lat_m, lon_m, res_m = native_resolution_m(lat, lon)
    sigma_px = res_m / GRID_RES_M
    dem_lowpass = masked_gaussian(dem_on_grid, valid, sigma_px)

    # ---- NEW step 1: highpass RAW dav2 first (no scaling yet), then scale the detail band
    dav2_lowpass_raw = masked_gaussian(dav2_raw, np.ones_like(valid, dtype=bool), sigma_px)
    dav2_highpass_raw = dav2_raw - dav2_lowpass_raw  # unscaled model units

    n = n_egm96(lon, lat)
    photons_all = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
    merged = sample_arrays_at_photons(
        rgb_path,
        {"dem_lowpass_at_px": dem_lowpass, "model_detail_at_px": dav2_highpass_raw},
        photons_all,
    )
    merged = merged.dropna(subset=["dem_lowpass_at_px", "model_detail_at_px"]).reset_index(drop=True)
    merged["photon_egm96"] = merged["height"] + n  # same sign convention as final_egm96 - n = final_ellipsoidal elsewhere
    merged["true_detail"] = merged["photon_egm96"] - merged["dem_lowpass_at_px"]

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(merged))
    half = len(idx) // 2
    train_idx, test_idx = idx[:half], idx[half:]
    train, test = merged.iloc[train_idx], merged.iloc[test_idx]

    alpha, alpha_info = alpha_from_known_height_icesat2(train["true_detail"].values, train["model_detail_at_px"].values)

    result = {
        "tile_id": tile_id, "lat": lat, "lon": lon, "native_res_m": res_m, "sigma_px": sigma_px,
        "n_photons_total": len(merged), "n_photons_train": len(train), "n_photons_test": len(test),
    }

    if alpha is None:
        result.update({"status": f"REJECTED: {alpha_info}", "alpha": None})
        return result

    result.update({"status": "ok", "alpha": alpha, **{f"anchor_{k}": v for k, v in alpha_info.items()}})

    dav2_highpass_scaled = (alpha * dav2_highpass_raw).astype(np.float32)
    final_egm96 = dem_lowpass + dav2_highpass_scaled

    # DEM check -- SAME pixel split/discipline as the original linear-OLS baseline, for comparability
    dem_idx = np.flatnonzero(valid)
    dem_rng = np.random.RandomState(SEED)
    dem_rng.shuffle(dem_idx)
    dem_half = len(dem_idx) // 2
    dem_test_idx = dem_idx[dem_half:]
    pred_test, true_test = final_egm96.ravel()[dem_test_idx], dem_flat[dem_test_idx]
    ok = np.isfinite(pred_test) & np.isfinite(true_test)
    dem_pearson, _ = stats.pearsonr(pred_test[ok], true_test[ok])
    result["held_out_dem_rmse_m"] = rmse(pred_test[ok], true_test[ok])
    result["held_out_dem_mae_m"] = mae(pred_test[ok], true_test[ok])
    result["held_out_dem_pearson"] = dem_pearson

    # ICESat-2 check -- ONLY the held-out photon TEST half (train half was used to fit alpha)
    final_ellipsoidal = final_egm96 - n
    sampled_final_test = sample_arrays_at_photons(rgb_path, {"final_value": final_ellipsoidal}, test[["lat", "lon", "height"]])

    ice_rmse = rmse(sampled_final_test["final_value"].values, sampled_final_test["height"].values)
    elev_range = float(dem_on_grid[valid].max() - dem_on_grid[valid].min())
    result["icesat2_rmse_m"] = ice_rmse
    result["elev_range_m"] = elev_range
    result["icesat2_rmse_pct_of_range"] = 100 * ice_rmse / elev_range
    result["n_icesat2_test_used"] = len(sampled_final_test)
    return result


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
        if r["status"] == "ok":
            print(f"[{i}/{len(tile_ids)}] {tile_id:15s} alpha={r['alpha']:.4f} "
                  f"ICE_rmse={r['icesat2_rmse_m']:.2f}m ({r['icesat2_rmse_pct_of_range']:.2f}% of range) "
                  f"n_anchors={r['anchor_n_anchors']}")
        else:
            print(f"[{i}/{len(tile_ids)}] {tile_id:15s} -> {r['status']}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"results_{target}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
