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
ALPHA = 1.0

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

print("=== dav2_highpass stats (whole tile) ===")
print("min/max/mean/std:", np.nanmin(dav2_highpass), np.nanmax(dav2_highpass), np.nanmean(dav2_highpass), np.nanstd(dav2_highpass))
print("abs mean:", np.nanmean(np.abs(dav2_highpass)))
elev_range = float(dem_on_grid[valid].max() - dem_on_grid[valid].min())
print("elev_range_m:", elev_range)

sem_dir = SEMANTIC_DIR / tile_id
with rasterio.open(sem_dir / "building_fraction.tif") as src:
    bf = src.read(1).astype(np.float32)
    bf_nodata = src.nodata
building_prob = np.clip(np.nan_to_num(bf, nan=0.0), 0.0, 1.0)

delta_A = ALPHA * building_prob * dav2_highpass
print("\n=== delta_A stats (whole tile) ===")
print("min/max/mean/std:", np.nanmin(delta_A), np.nanmax(delta_A), np.nanmean(delta_A), np.nanstd(delta_A))
print("abs mean:", np.nanmean(np.abs(delta_A)))
print("abs mean as % of elev range:", 100*np.nanmean(np.abs(delta_A))/elev_range)

# at high building_prob pixels specifically
mask_hi = building_prob > 0.5
print("\n=== delta_A at building_prob>0.5 pixels (n=%d) ===" % mask_hi.sum())
print("abs mean:", np.nanmean(np.abs(delta_A[mask_hi])))
print("dav2_highpass abs mean there:", np.nanmean(np.abs(dav2_highpass[mask_hi])))
print("building_prob mean there:", np.nanmean(building_prob[mask_hi]))

final_A = dem_lowpass + (1.0 + ALPHA * building_prob) * dav2_highpass

n = n_egm96(lon, lat)
photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
print("\ntotal photons:", len(photons))

final_baseline_ellip = final_baseline - n
final_A_ellip = final_A - n
sampled_base = sample_depth_at_photons(rgb_path, final_baseline_ellip, photons)
sampled_A = sample_depth_at_photons(rgb_path, final_A_ellip, photons)
print("sampled photons (in-bounds):", len(sampled_base))

# building_prob sampled at photon pixel locations
sampled_bp = sample_depth_at_photons(rgb_path, building_prob, photons)
print("\n=== building_prob AT ICESat-2 photon locations ===")
print("n:", len(sampled_bp))
print("min/max/mean/std:", sampled_bp['depth_value'].min(), sampled_bp['depth_value'].max(), sampled_bp['depth_value'].mean(), sampled_bp['depth_value'].std())
print("fraction of photons with building_prob>0.5:", (sampled_bp['depth_value']>0.5).mean())
print("fraction of photons with building_prob>0:", (sampled_bp['depth_value']>0).mean())

diff = sampled_A['depth_value'].values - sampled_base['depth_value'].values
print("\n=== A - baseline delta AT ICESat-2 photon locations ===")
print("abs mean:", np.mean(np.abs(diff)), "max abs:", np.max(np.abs(diff)))
print("as % of elev range:", 100*np.mean(np.abs(diff))/elev_range)
