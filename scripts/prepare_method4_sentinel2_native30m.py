"""Follow-up to run_method4_sentinel2.py: retrain the SAME architecture
at SRTM's real native resolution (~30m), not the interpolated 10m grid,
to test whether the diagnosed mechanism (the CNN memorizing bilinear-
interpolation structure introduced by upsampling a 30m DEM onto a 10m
grid) is actually the cause of the ICESat-2-check regression found there.

This directly removes that mechanism rather than working around it: a
NEW 30m UTM grid is defined per tile (same footprint, ~333x333 instead of
1000x1000 pixels), and RGB/DAv2-depth/building are DOWNSAMPLED onto it
(average resampling, real information loss, not interpolation) while
SRTM is reprojected onto it from its own ~30m native grid -- a
same-resolution regrid, not a 3x upsample. The network never sees an
interpolated target pixel finer than the DEM's own real resolution.

Same per-tile linear baseline + normalization approach as before,
recomputed at this new resolution (the linear fit itself must also be
refit at 30m for a fair, consistent comparison -- the 10m-grid linear fit
is not reused here).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds
from pyproj import Transformer
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
SRTM_DIR = ROOT / "data/sentinel2_benchmark/srtm_raw"
VERDICTS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
OUT_DIR = ROOT / "data/sentinel2_benchmark/method4_port_native30m"
WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
BASE_SEED = 42
NATIVE_RES_M = 30.0

_T_EGM96 = Transformer.from_crs("EPSG:4979", "EPSG:5773", always_xy=True)


def n_egm96(lon, lat):
    _, _, n = _T_EGM96.transform(lon, lat, 0.0)
    return n


def native_grid(rgb_path):
    """A new affine transform + shape covering the same bounds as
    rgb_path, but at NATIVE_RES_M resolution instead of the source's own
    (10m)."""
    with rasterio.open(rgb_path) as src:
        crs = src.crs
        left, bottom, right, top = src.bounds
    width = max(1, round((right - left) / NATIVE_RES_M))
    height = max(1, round((top - bottom) / NATIVE_RES_M))
    transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    return crs, transform, (height, width)


def downsample_band(src_path, band_indices, dst_crs, dst_transform, dst_shape, resampling):
    with rasterio.open(src_path) as src:
        src_data = src.read(band_indices).astype(np.float32)
        src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata
    out = np.full((len(band_indices), *dst_shape), np.nan, dtype=np.float32)
    reproject(
        source=src_data, destination=out,
        src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=resampling,
    )
    return out


def downsample_array(arr, src_transform, src_crs, dst_crs, dst_transform, dst_shape, resampling, src_nodata=None):
    out = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=arr.astype(np.float32), destination=out,
        src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=resampling,
    )
    return out


def worldcover_tile_name(lat_c, lon_c):
    import math
    tile_lat = math.floor(lat_c / 3) * 3
    tile_lon = math.floor(lon_c / 3) * 3
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    return f"{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}"


def fetch_building_native(lat_c, lon_c, dst_crs, dst_transform, dst_shape):
    tile = worldcover_tile_name(lat_c, lon_c)
    url = f"{WORLDCOVER_BASE}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    with rasterio.open(url) as src:
        left, bottom, right, top = rasterio.transform.array_bounds(dst_shape[0], dst_shape[1], dst_transform)
        t = Transformer.from_crs(dst_crs, src.crs, always_xy=True)
        w0, s0 = t.transform(left, bottom)
        w1, s1 = t.transform(right, top)
        pad = 0.01
        window = from_bounds(min(w0, w1) - pad, min(s0, s1) - pad, max(w0, w1) + pad, max(s0, s1) + pad, transform=src.transform)
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)
        src_crs = src.crs
    builtup = (data == 50).astype(np.float32)
    return downsample_array(builtup, win_transform, src_crs, dst_crs, dst_transform, dst_shape, Resampling.average)


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    print(f"Preparing native-{NATIVE_RES_M:.0f}m data for {len(accepted)} accepted tiles.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scale_rows = []

    for i, tile_id in enumerate(accepted, 1):
        m = manifest.loc[tile_id]
        rgb_path = ROOT / m["rgb_path"]
        n = n_egm96(m["lon"], m["lat"])

        dst_crs, dst_transform, dst_shape = native_grid(rgb_path)

        rgb_native = downsample_band(rgb_path, [1, 2, 3], dst_crs, dst_transform, dst_shape, Resampling.average)
        rgb_native = np.moveaxis(rgb_native, 0, -1)  # (H, W, 3)

        # DAv2 depth has no georeferencing of its own (.npy) -- it's on the exact
        # same grid as rgb_path (1000x1000, same transform), so reuse rgb_path's
        # source transform/crs to downsample it the same way.
        with rasterio.open(rgb_path) as src:
            src_transform_10m, src_crs_10m = src.transform, src.crs
        dav2 = np.load(DAV2_DIR / f"{tile_id}_depth.npy").astype(np.float32)
        depth_native = downsample_array(dav2, src_transform_10m, src_crs_10m, dst_crs, dst_transform, dst_shape, Resampling.average)

        building_native = fetch_building_native(m["lat"], m["lon"], dst_crs, dst_transform, dst_shape)

        with rasterio.open(SRTM_DIR / f"{tile_id}_srtm.tif") as src:
            srtm = src.read(1).astype(np.float32)
            srtm_transform, srtm_crs, srtm_nodata = src.transform, src.crs, src.nodata
        srtm_native = downsample_array(srtm, srtm_transform, srtm_crs, dst_crs, dst_transform, dst_shape, Resampling.bilinear, src_nodata=srtm_nodata)

        valid = ~np.isnan(depth_native) & ~np.isnan(srtm_native) & ~np.isnan(rgb_native).any(axis=-1)
        srtm_ellipsoidal = srtm_native - n

        idx = np.flatnonzero(valid)
        rng = np.random.RandomState(BASE_SEED)
        rng.shuffle(idx)
        train_idx = idx[: len(idx) // 2]
        depth_flat, srtm_flat = depth_native.ravel(), srtm_ellipsoidal.ravel()
        lr = stats.linregress(depth_flat[train_idx], srtm_flat[train_idx])
        a, b = lr.slope, lr.intercept
        linear_baseline = a * depth_native + b

        residual_truth = np.where(valid, srtm_ellipsoidal - linear_baseline, np.nan).astype(np.float32)
        raw_max_abs = float(np.nanmax(np.abs(residual_truth)))
        scale_factor = raw_max_abs / 50.0
        residual_truth_normalized = (residual_truth / scale_factor).astype(np.float32)
        scale_rows.append({
            "tile_id": tile_id, "scale_factor": scale_factor, "raw_max_abs_residual_m": raw_max_abs,
            "a_slope": a, "b_intercept": b, "n_egm96": n, "shape_h": dst_shape[0], "shape_w": dst_shape[1],
            "n_valid_px": int(valid.sum()),
        })

        out_profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": dst_shape[0], "width": dst_shape[1], "crs": dst_crs, "transform": dst_transform, "nodata": np.nan,
        }
        with rasterio.open(OUT_DIR / f"{tile_id}_residual_truth.tif", "w", **out_profile) as dst:
            dst.write(residual_truth_normalized, 1)
        with rasterio.open(OUT_DIR / f"{tile_id}_building.tif", "w", **out_profile) as dst:
            dst.write(np.nan_to_num(building_native, nan=0.0), 1)
        with rasterio.open(OUT_DIR / f"{tile_id}_linear_baseline.tif", "w", **out_profile) as dst:
            dst.write(linear_baseline.astype(np.float32), 1)
        with rasterio.open(OUT_DIR / f"{tile_id}_depth.tif", "w", **out_profile) as dst:
            dst.write(np.nan_to_num(depth_native, nan=0.0), 1)
        rgb_profile = out_profile.copy()
        rgb_profile.update(count=3, dtype="float32", nodata=None)
        with rasterio.open(OUT_DIR / f"{tile_id}_RGB.tif", "w", **rgb_profile) as dst:
            dst.write(np.moveaxis(np.nan_to_num(rgb_native, nan=0.0), -1, 0))

        print(
            f"[{i}/{len(accepted)}] {tile_id:15s} shape={dst_shape} raw_residual=[{np.nanmin(residual_truth):.1f},{np.nanmax(residual_truth):.1f}]m "
            f"scale_factor={scale_factor:.3f} n_valid_px={valid.sum()} valid_pct={100*valid.mean():.1f}%"
        )

    pd.DataFrame(scale_rows).to_csv(OUT_DIR / "scale_factors.csv", index=False)
    print(f"\nDone. Wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
