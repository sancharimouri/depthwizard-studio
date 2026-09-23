"""Data prep for Test B: retrain Method 4's exact architecture
(phase2_building_rank_v2 / ScaleModulationNetV2) with Google Open
Buildings 2.5D Temporal building heights as the training target on
BUILDING pixels, instead of SRTM everywhere -- on the 8 urban tiles only
(the only category where Open Buildings has real coverage worth using).

Public data access (no auth): the bucket layout and band assignment are
documented in Google's own bucket structure (v1/manifests/*.json ->
v1/geotiffs/*/tile_*.tif; band 1 fractional_count, band 2 building_height,
band 3 building_presence) -- read directly here, independent of any
external repo's script.

Target construction (the actual "instead of SRTM" swap):
  - Ground pixels (Open Buildings has no coverage, or presence <=
    CONFIDENT_THRESHOLD): UNCHANGED from the existing SRTM-based target
    in data/sentinel2_benchmark/method4_port/ -- this isolates the test
    to "does a better BUILDING target help", not "does replacing
    everything with a different, sparser source help".
  - Confident building pixels (presence > CONFIDENT_THRESHOLD, height
    present): target = linear_baseline (the existing per-tile SRTM-fit
    ground estimate, unchanged) + Open Buildings' own building_height
    (AGL). This gives the model a real building-height signal that raw
    30m-native SRTM cannot resolve at all.
  - A NEW per-tile scale_factor is fit for these 8 tiles (Open Buildings
    heights up to 100m can exceed the original SRTM-residual normalization
    range) -- same +-50 native-output-range convention as
    prepare_method4_sentinel2_data.py, just refit on the new residual.

Writes, per tile, to data/sentinel2_benchmark/method4_port_openbuildings/:
  {tile}_residual_truth.tif  -- NEW target, normalized to +-50
  {tile}_building.tif        -- copied unchanged from method4_port/ (WorldCover input channel, untouched)
  {tile}_linear_baseline.tif -- copied unchanged from method4_port/ (still the ground reference)
  {tile}_ob_height.tif       -- raw Open Buildings height, kept for inspection/debugging, not used directly in training
  {tile}_ob_presence.tif     -- raw Open Buildings presence, kept for inspection/debugging
  scale_factors.csv          -- new per-tile normalization factor for this target
  coverage_report.csv        -- % of tile covered by confident-building pixels, and Open Buildings year used
"""
from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
PORT_DIR = ROOT / "data/sentinel2_benchmark/method4_port"
OUT_DIR = ROOT / "data/sentinel2_benchmark/method4_port_openbuildings"

URBAN_TILES = ["bengaluru", "chennai", "delhi", "hyderabad", "jaipur", "kochi_city", "mumbai", "pune"]
OB_YEAR = 2023  # latest year available for EPSG:32643/32644 as of this prep (checked live); tiles are dated late 2025 -- a real ~2-3y temporal gap, reported in coverage_report.csv, not hidden.
CONFIDENT_THRESHOLD = 0.5

BUCKET = "open-buildings-temporal-data"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
OB_NODATA = -99.0
HEIGHT_BAND, PRESENCE_BAND = 2, 3


def _list_manifests(prefix: str) -> list[str]:
    names, tok = [], None
    while True:
        u = f"{API}?prefix={prefix}&maxResults=1000" + (f"&pageToken={tok}" if tok else "")
        d = json.load(urllib.request.urlopen(u, timeout=90))
        names += [i["name"] for i in d.get("items", []) if i["name"].endswith(".json")]
        tok = d.get("nextPageToken")
        if not tok:
            return names


def _tiles_for(epsg: int, year: int):
    tag = f"_EPSG_{epsg}_{year}_"
    out = []
    for name in _list_manifests("v1/manifests/"):
        if tag not in name:
            continue
        m = json.load(urllib.request.urlopen(MEDIA + name, timeout=90))
        pref = m["uriPrefix"].replace(f"gs://{BUCKET}/", "")
        for ts in m.get("tilesets", []):
            for src in ts.get("sources", []):
                a, dim = src["affineTransform"], src["dimensions"]
                x0, y0 = a["translateX"], a["translateY"]
                x1, y1 = x0 + a["scaleX"] * dim["width"], y0 + a["scaleY"] * dim["height"]
                out.append({"url": MEDIA + pref + src["uris"][0],
                           "bounds": (x0, min(y0, y1), x1, max(y0, y1)), "res": abs(a["scaleX"])})
    return out


def fetch_open_buildings(epsg: int, bounds_dst_crs, dst_crs, dst_transform, dst_shape):
    """Height + presence, reprojected onto the tile's own 10m grid. Returns
    (height, presence) float32 arrays, NaN where Open Buildings has no data."""
    cand = _tiles_for(epsg, OB_YEAR)
    left, bottom, right, top = bounds_dst_crs
    hit = [t for t in cand if not (t["bounds"][2] <= left or t["bounds"][0] >= right
                                   or t["bounds"][3] <= bottom or t["bounds"][1] >= top)]
    if not hit:
        raise SystemExit(f"no Open Buildings tile covers this footprint for EPSG:{epsg} year {OB_YEAR}")

    height = np.full(dst_shape, np.nan, dtype=np.float32)
    presence = np.full(dst_shape, np.nan, dtype=np.float32)
    for t in hit:
        with rasterio.open("/vsicurl/" + t["url"]) as src:
            ix = (max(left, t["bounds"][0]), max(bottom, t["bounds"][1]),
                  min(right, t["bounds"][2]), min(top, t["bounds"][3]))
            w = from_bounds(*ix, transform=src.transform).round_offsets().round_lengths()
            if w.width < 1 or w.height < 1:
                continue
            a = src.read([HEIGHT_BAND, PRESENCE_BAND], window=w).astype(np.float32)
            a[a == OB_NODATA] = np.nan
            src_transform = src.window_transform(w)

            h_tmp = np.full(dst_shape, np.nan, dtype=np.float32)
            p_tmp = np.full(dst_shape, np.nan, dtype=np.float32)
            reproject(source=a[0], destination=h_tmp, src_transform=src_transform, src_crs=src.crs,
                     src_nodata=np.nan, dst_transform=dst_transform, dst_crs=dst_crs,
                     dst_nodata=np.nan, resampling=Resampling.average)
            reproject(source=a[1], destination=p_tmp, src_transform=src_transform, src_crs=src.crs,
                     src_nodata=np.nan, dst_transform=dst_transform, dst_crs=dst_crs,
                     dst_nodata=np.nan, resampling=Resampling.average)
            fresh = ~np.isnan(h_tmp) & np.isnan(height)
            height[fresh] = h_tmp[fresh]
            presence[fresh] = p_tmp[fresh]
    return height, presence


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    scale_factors_srtm = pd.read_csv(PORT_DIR / "scale_factors.csv").set_index("tile_id")["scale_factor"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scale_rows, coverage_rows = [], []
    for i, tile_id in enumerate(URBAN_TILES, 1):
        m = manifest.loc[tile_id]
        rgb_path = ROOT / m["rgb_path"]
        epsg = int(m["bbox_utm_epsg"])

        with rasterio.open(rgb_path) as src:
            dst_crs, dst_transform, dst_shape = src.crs, src.transform, (src.height, src.width)
            bounds = tuple(src.bounds)
            profile = src.profile

        with rasterio.open(PORT_DIR / f"{tile_id}_linear_baseline.tif") as src:
            linear_baseline = src.read(1).astype(np.float32)
        with rasterio.open(PORT_DIR / f"{tile_id}_residual_truth.tif") as src:
            residual_norm_srtm = src.read(1).astype(np.float32)
        srtm_scale = float(scale_factors_srtm[tile_id])
        original_abs = linear_baseline + residual_norm_srtm * srtm_scale
        valid = np.isfinite(original_abs)

        ob_height, ob_presence = fetch_open_buildings(epsg, bounds, dst_crs, dst_transform, dst_shape)
        confident = valid & (ob_presence > CONFIDENT_THRESHOLD) & np.isfinite(ob_height)

        new_residual_raw = np.where(confident, ob_height, original_abs - linear_baseline).astype(np.float32)
        new_residual_raw = np.where(valid, new_residual_raw, np.nan).astype(np.float32)

        raw_max_abs = float(np.nanmax(np.abs(new_residual_raw)))
        scale_factor = raw_max_abs / 50.0
        residual_norm_new = (new_residual_raw / scale_factor).astype(np.float32)
        scale_rows.append({"tile_id": tile_id, "scale_factor": scale_factor, "raw_max_abs_residual_m": raw_max_abs})

        out_profile = profile.copy()
        out_profile.update(dtype="float32", count=1, nodata=np.nan)
        with rasterio.open(OUT_DIR / f"{tile_id}_residual_truth.tif", "w", **out_profile) as dst:
            dst.write(residual_norm_new, 1)
        with rasterio.open(OUT_DIR / f"{tile_id}_ob_height.tif", "w", **out_profile) as dst:
            dst.write(np.nan_to_num(ob_height, nan=-1.0), 1)
        with rasterio.open(OUT_DIR / f"{tile_id}_ob_presence.tif", "w", **out_profile) as dst:
            dst.write(np.nan_to_num(ob_presence, nan=-1.0), 1)
        shutil.copy(PORT_DIR / f"{tile_id}_building.tif", OUT_DIR / f"{tile_id}_building.tif")
        shutil.copy(PORT_DIR / f"{tile_id}_linear_baseline.tif", OUT_DIR / f"{tile_id}_linear_baseline.tif")

        cov_pct = 100.0 * confident.sum() / valid.sum() if valid.sum() else 0.0
        coverage_rows.append({"tile_id": tile_id, "ob_year": OB_YEAR, "confident_building_pct": cov_pct,
                              "confident_pixels": int(confident.sum()), "valid_pixels": int(valid.sum())})
        print(f"[{i}/{len(URBAN_TILES)}] {tile_id:12s} confident_building={cov_pct:.2f}% "
              f"raw_residual_range=[{np.nanmin(new_residual_raw):.1f},{np.nanmax(new_residual_raw):.1f}]m "
              f"scale_factor={scale_factor:.3f}")

    pd.DataFrame(scale_rows).to_csv(OUT_DIR / "scale_factors.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT_DIR / "coverage_report.csv", index=False)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
