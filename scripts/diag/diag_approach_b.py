import sys
from pathlib import Path
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import stats

from run_frequency_fusion_sentinel2 import (
    masked_gaussian, native_resolution_m, n_egm96, rmse, mae,
    MANIFEST, DAV2_DIR, SRTM_DIR, PHOTON_DIR, VERDICTS_CSV, GRID_RES_M, SEED,
)
from lib_backbone_correlation import sample_depth_at_photons

ROOT = Path(".").resolve()
SEMANTIC_DIR = ROOT / "data/sentinel2_benchmark/semantic"
CONFIDENT_THRESHOLD = 0.5

manifest = pd.read_csv(MANIFEST).set_index("tile_id")
tile_id = "chennai"
m = manifest.loc[tile_id]
rgb_path = ROOT / m["rgb_path"]
lat, lon = float(m["lat"]), float(m["lon"])

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

sem_dir = SEMANTIC_DIR / tile_id
with rasterio.open(sem_dir / "ob_height.tif") as src:
    ob_height = src.read(1).astype(np.float32)
with rasterio.open(sem_dir / "ob_presence.tif") as src:
    ob_presence = src.read(1).astype(np.float32)

confident = np.isfinite(ob_presence) & (ob_presence > CONFIDENT_THRESHOLD) & np.isfinite(ob_height)
print("confident pixels:", confident.sum(), "pct:", 100*confident.sum()/confident.size)
print("ob_height at confident px: min/max/mean/std:", np.nanmin(ob_height[confident]), np.nanmax(ob_height[confident]), np.nanmean(ob_height[confident]), np.nanstd(ob_height[confident]))

blended = dem_lowpass + 0.5 * dav2_highpass + 0.5 * ob_height
final_B = np.where(confident, blended, final_baseline).astype(np.float32)

delta_B = (final_B - final_baseline)[confident]
print("\ndelta_B (final_B - baseline) at confident px: min/max/mean/std:", delta_B.min(), delta_B.max(), delta_B.mean(), delta_B.std())

n = n_egm96(lon, lat)
photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")

final_baseline_ellip = final_baseline - n
final_B_ellip = final_B - n
sampled_base = sample_depth_at_photons(rgb_path, final_baseline_ellip, photons)
sampled_B = sample_depth_at_photons(rgb_path, final_B_ellip, photons)
sampled_conf = sample_depth_at_photons(rgb_path, confident.astype(np.float32), photons)

conf_mask = sampled_conf["depth_value"].values > 0.5
print("\nphotons landing on OB-confident building pixels:", conf_mask.sum(), "/", len(sampled_conf))

if conf_mask.sum() > 0:
    true_h = sampled_base.loc[sampled_base.index.isin(sampled_conf.index[conf_mask])]
    # align via index since sample_depth_at_photons drops NaN independently -- use merge on original index instead
    df = photons.copy()
    df["conf"] = confident.astype(np.float32)[
        np.clip((photons.index*0),0,0)  # placeholder, will recompute properly below
    ] if False else None

# Simpler: recompute sampling manually with a single shared pixel index to keep alignment
from lib_backbone_correlation import sample_depth_at_photons as _sdp
with rasterio.open(rgb_path) as src:
    crs = src.crs; transform = src.transform; height_px, width_px = src.height, src.width
from pyproj import Transformer
to_tile_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = to_tile_crs.transform(photons["lon"].values, photons["lat"].values)
inv = ~transform
cols, rows = inv * (x, y)
cols = np.floor(cols).astype(int); rows = np.floor(rows).astype(int)
in_bounds = (cols >= 0) & (cols < width_px) & (rows >= 0) & (rows < height_px)

conf_at_photon = np.zeros(len(photons), dtype=bool)
conf_at_photon[in_bounds] = confident[rows[in_bounds], cols[in_bounds]]
base_at_photon = np.full(len(photons), np.nan)
base_at_photon[in_bounds] = final_baseline_ellip[rows[in_bounds], cols[in_bounds]]
B_at_photon = np.full(len(photons), np.nan)
B_at_photon[in_bounds] = final_B_ellip[rows[in_bounds], cols[in_bounds]]
obh_at_photon = np.full(len(photons), np.nan)
obh_at_photon[in_bounds] = ob_height[rows[in_bounds], cols[in_bounds]]

true_h = photons["height"].values
sel = conf_at_photon & np.isfinite(base_at_photon) & np.isfinite(B_at_photon)
print("\n=== At photons landing on OB-confident building pixels (n=%d) ===" % sel.sum())
print("mean ICESat-2 true height:", np.mean(true_h[sel]))
print("mean baseline pred:", np.mean(base_at_photon[sel]))
print("mean B pred:", np.mean(B_at_photon[sel]))
print("mean ob_height (AGL) there:", np.mean(obh_at_photon[sel]))
print("baseline error (pred-true) mean:", np.mean(base_at_photon[sel]-true_h[sel]), "abs mean:", np.mean(np.abs(base_at_photon[sel]-true_h[sel])))
print("B error (pred-true) mean:", np.mean(B_at_photon[sel]-true_h[sel]), "abs mean:", np.mean(np.abs(B_at_photon[sel]-true_h[sel])))

# Now also test the FULL (unweighted) ground+AGL formula the user suggested / Method4 Test B used
full_blend = dem_lowpass + ob_height
full_at_photon = np.full(len(photons), np.nan)
full_ellip = full_blend - n
full_at_photon[in_bounds] = full_ellip[rows[in_bounds], cols[in_bounds]]
print("\n--- hypothetical FULL dem_lowpass+ob_height (no dav2_highpass, full AGL weight) ---")
print("mean pred:", np.mean(full_at_photon[sel]))
print("error mean:", np.mean(full_at_photon[sel]-true_h[sel]), "abs mean:", np.mean(np.abs(full_at_photon[sel]-true_h[sel])))
