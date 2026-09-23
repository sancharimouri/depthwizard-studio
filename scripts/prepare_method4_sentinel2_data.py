"""Data prep for porting phase2_building_rank_v2 (Method 4 v2) to
Sentinel-2 / SRTM, on the 25 sign-flip-detector-accepted tiles.

Architecture/loss are reused UNCHANGED from evaluate_method4_v2.py
(ScaleModulationNetV2, huber+smoothness+rank losses, same hyperparameters
as the phase2_building_rank_v2 config: dense patches, 60 epochs,
smoothness_weight=0.01, rank_weight=0.5, rank_pairs_per_patch=2000,
rank_margin=0.25, building channel, ground_plane_weight=0.0). Only the
input DOMAIN changes -- but that requires two real data-prep decisions,
documented here rather than silently assumed:

1. TARGET REFRAMING (necessary, not optional): DFC2019's AGL truth is a
   naturally small quantity (0-50m building/tree height above local
   ground) -- the model's fixed output-scale constants (`scale` in
   [0.25x, 4x], `residual` capped at +-50m in evaluate_method4_v2.py's
   ScaleModulationNetV2.forward) were sized for that range. Our Sentinel-2
   tiles' real elevation spans up to ~4400m (manali) -- feeding that
   directly as the target would make the architecture's own constants
   meaningless by construction (a +-50m residual cap cannot express a
   270m correction that karnal-scale tiles might need), not just
   poorly-tuned. The fix that keeps the architecture LITERALLY unchanged:
   train the model to predict the RESIDUAL on top of the already-
   established per-tile linear SRTM calibration (same a/b/N fit as
   run_srtm_comparison.py / test_residual_correction.py), not raw
   elevation. This residual is naturally small (same order of magnitude
   as DFC2019's AGL), and it's the exact analog of "does the learned
   correction beat the per-tile-OLS baseline" -- the same comparison
   CLAUDE.md already reports for DFC2019 Method 4 vs. its OLS baseline.
2. BUILDING CHANNEL: DFC2019's building-probability channel came from an
   ONNX semantic-segmentation model trained on ~0.3m airborne imagery
   (scripts/generate_dfc_building_prior.py) -- reusing that model on 10m
   Sentinel-2 imagery would be a resolution/domain mismatch, not a valid
   port. Substituted with ESA WorldCover's built-up class (value 50) at
   its native 10m resolution (real categorical land-cover data, not a
   model prediction, but serving the identical semantic role, and at
   the CORRECT native resolution for this domain), fetched fresh via the
   same public S3 access already used in benchmark_content_audit.py.

**Follow-up fix (same day):** the raw linear-fit residual still exceeds
the architecture's +-50m native output range on 8/25 tiles -- checked
empirically (min/max reported per tile below) before training, per
explicit instruction. Breakdown: 7 of the 8 (almora, dehradun,
dharamshala, kohima, manali, mumbai, jaipur) have moderate-to-good
DEM-fit quality (held_out_dem_pearson 0.35-0.83) -- their large residual
comes purely from absolute scale (large true elevation range or a
moderate-but-real amount of unexplained variance), not from a bad fit.
Only pune is a genuinely poor fit (pearson=0.05, this project's existing
LOW-CONFIDENCE tier). Fix applied: **per-tile normalization**, not an
architecture change -- each tile's residual is divided by its own
max-abs value (rescaled so the tile's own extremes map to exactly
+-50, the network's native tanh range) before being used as the training
target, and the model's output is multiplied back by that same per-tile
factor when reconstructing absolute elevation at eval time. This changes
only the data pipeline (a per-tile scalar rescale in and out), not the
architecture or loss functions, which are reused exactly as in
evaluate_method4_v2.py.

Writes, per tile, to data/sentinel2_benchmark/method4_port/:
  {tile}_residual_truth.tif  -- CNN training target, NORMALIZED to the
                                 network's native +-50 range (see above)
  {tile}_building.tif        -- WorldCover built-up fraction, reprojected to tile grid
  {tile}_linear_baseline.tif -- the linear baseline itself (needed to reconstruct
                                 absolute elevation from the model's residual prediction at eval time)
  scale_factors.csv          -- per-tile normalization factor; multiply the
                                 model's raw output by this to get back to true meters
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from pyproj import Transformer
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
SRTM_DIR = ROOT / "data/sentinel2_benchmark/srtm_raw"
VERDICTS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
OUT_DIR = ROOT / "data/sentinel2_benchmark/method4_port"
WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
BASE_SEED = 42

_T_EGM96 = Transformer.from_crs("EPSG:4979", "EPSG:5773", always_xy=True)


def n_egm96(lon, lat):
    _, _, n = _T_EGM96.transform(lon, lat, 0.0)
    return n


def fit_srtm_baseline(tile_id, rgb_path):
    dav2 = np.load(DAV2_DIR / f"{tile_id}_depth.npy")
    with rasterio.open(SRTM_DIR / f"{tile_id}_srtm.tif") as src:
        dem = src.read(1).astype(np.float32)
        src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata
    with rasterio.open(rgb_path) as rgb_src:
        dst_crs, dst_transform, dst_shape = rgb_src.crs, rgb_src.transform, (rgb_src.height, rgb_src.width)
        profile = rgb_src.profile

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
    return lr.slope, lr.intercept, dav2, dem_on_grid, valid, dst_transform, dst_crs, profile


def worldcover_tile_name(lat_c, lon_c):
    import math
    tile_lat = math.floor(lat_c / 3) * 3
    tile_lon = math.floor(lon_c / 3) * 3
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    return f"{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}"


def fetch_building_channel(lat_c, lon_c, dst_crs, dst_transform, dst_shape):
    """ESA WorldCover built-up class (value 50) at native 10m, reprojected
    onto the tile's exact grid. Returns a float32 0/1 array."""
    tile = worldcover_tile_name(lat_c, lon_c)
    url = f"{WORLDCOVER_BASE}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    with rasterio.open(url) as src:
        # window covering the destination tile's bounds (+ small buffer) in the source's CRS
        from pyproj import Transformer as T
        left, bottom, right, top = rasterio.transform.array_bounds(dst_shape[0], dst_shape[1], dst_transform)
        t = T.from_crs(dst_crs, src.crs, always_xy=True)
        w0, s0 = t.transform(left, bottom)
        w1, s1 = t.transform(right, top)
        pad = 0.01
        window = from_bounds(min(w0, w1) - pad, min(s0, s1) - pad, max(w0, w1) + pad, max(s0, s1) + pad, transform=src.transform)
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)
        src_crs = src.crs

    builtup = (data == 50).astype(np.float32)
    out = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=builtup, destination=out,
        src_transform=win_transform, src_crs=src_crs, src_nodata=None,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.average,  # fraction of built-up within each dest pixel
    )
    return np.nan_to_num(out, nan=0.0)


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    print(f"Preparing data for {len(accepted)} accepted tiles.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scale_rows = []

    for i, tile_id in enumerate(accepted, 1):
        m = manifest.loc[tile_id]
        rgb_path = ROOT / m["rgb_path"]
        n = n_egm96(m["lon"], m["lat"])

        a, b, dav2, dem_on_grid, valid, dst_transform, dst_crs, profile = fit_srtm_baseline(tile_id, rgb_path)
        srtm_ellipsoidal = dem_on_grid - n
        linear_baseline = a * dav2 + b - n
        residual_truth = np.where(valid, srtm_ellipsoidal - linear_baseline, np.nan).astype(np.float32)

        raw_max_abs = float(np.nanmax(np.abs(residual_truth)))
        scale_factor = raw_max_abs / 50.0  # maps this tile's own extreme residual to exactly +-50
        residual_truth_normalized = (residual_truth / scale_factor).astype(np.float32)
        scale_rows.append({"tile_id": tile_id, "scale_factor": scale_factor, "raw_max_abs_residual_m": raw_max_abs})

        building = fetch_building_channel(m["lat"], m["lon"], dst_crs, dst_transform, dav2.shape)

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1, nodata=np.nan)

        with rasterio.open(OUT_DIR / f"{tile_id}_residual_truth.tif", "w", **out_profile) as dst:
            dst.write(residual_truth_normalized, 1)
        with rasterio.open(OUT_DIR / f"{tile_id}_building.tif", "w", **out_profile) as dst:
            dst.write(building, 1)
        with rasterio.open(OUT_DIR / f"{tile_id}_linear_baseline.tif", "w", **out_profile) as dst:
            dst.write(linear_baseline.astype(np.float32), 1)

        print(
            f"[{i}/{len(accepted)}] {tile_id:15s} raw_residual_range=[{np.nanmin(residual_truth):.1f},{np.nanmax(residual_truth):.1f}]m "
            f"scale_factor={scale_factor:.3f} (normalized range=[-50,50] by construction) "
            f"builtup_mean={building.mean():.3f} valid_pct={100*valid.mean():.1f}%"
        )

    pd.DataFrame(scale_rows).to_csv(OUT_DIR / "scale_factors.csv", index=False)
    print(f"\nDone. Wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
