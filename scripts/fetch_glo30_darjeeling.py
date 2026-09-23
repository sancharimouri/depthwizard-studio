#!/usr/bin/env python3
"""
Re-fetch Copernicus GLO-30 for Darjeeling correctly (2026-09-23).

Diagnosis of the long-standing "GLO-30 for Darjeeling is all-NaN": Darjeeling's DEM
footprint (lat 26.9999-27.0901, lon 88.2099-88.3101) straddles the 27 N tile boundary.
Copernicus GLO-30 1x1 degree tiles are named by their SW corner, so the footprint needs
BOTH N26_00_E088 and N27_00_E088. Choosing a single tile from the footprint's southern
edge (floor(26.9999) = 26 -> N26 only) leaves just one pixel row inside the tile, so a
crop is 0.31% valid, i.e. effectively all-NaN. Reproduced 2026-09-23 against the public
AWS COGs (N26 crop 0.31% valid; N27 crop 100% of its rows).

Fix: mosaic every tile from floor(min) to floor(max) on both axes (the same rule as
scripts/fetch_copernicus_dem_benchmark.py), then crop to the demo's DEM grid
(data/elevation/darjeeling/Darjeeling_OpenTopography_DSM.tif).
Data acquisition only; the demo is NOT changed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
REF = ROOT / "data/elevation/darjeeling/Darjeeling_OpenTopography_DSM.tif"
OUT_DIR = ROOT / "data/elevation/darjeeling"


def tile_url(lat: int, lon: int) -> str:
    name = f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
    return f"{BUCKET}/{name}/{name}.tif"


def main():
    with rasterio.open(REF) as ref:
        w, s, e, n = ref.bounds
        ref_arr, ref_transform, ref_profile = ref.read(1), ref.transform, ref.profile
    tiles = [(la, lo) for la in range(math.floor(s), math.floor(n) + 1)
             for lo in range(math.floor(w), math.floor(e) + 1)]
    print("tiles needed:", tiles)
    srcs = [rasterio.open(tile_url(la, lo)) for la, lo in tiles]
    # Full-tile mosaic (for parity with the other regions' *_Copernicus_GLO30_DSM.tif).
    full, full_tf = merge(srcs)
    prof = srcs[0].profile.copy()
    prof.update(height=full.shape[1], width=full.shape[2], transform=full_tf, driver="GTiff",
                compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(OUT_DIR / "Darjeeling_Copernicus_GLO30_DSM.tif", "w", **prof) as dst:
        dst.write(full)
    # Crop exactly onto the demo DEM's grid.
    crop, crop_tf = merge(srcs, bounds=(w, s, e, n), res=(ref_transform.a, -ref_transform.e))
    for src in srcs:
        src.close()
    crop = crop[:, : ref_arr.shape[0], : ref_arr.shape[1]]
    cprof = ref_profile.copy()
    cprof.update(dtype="float32", transform=crop_tf, compress="deflate")
    with rasterio.open(OUT_DIR / "Darjeeling_Copernicus_GLO30_DSM_cropped.tif", "w", **cprof) as dst:
        dst.write(crop.astype("float32"))
    c = crop[0].astype("float64")
    valid = np.isfinite(c) & (c > -1000)
    diff = c - ref_arr.astype("float64")
    print(f"cropped shape {c.shape}  valid {100 * valid.mean():.2f}%  range {c[valid].min():.3f}..{c[valid].max():.3f} m")
    print(f"vs demo DEM (Darjeeling_OpenTopography_DSM.tif): max|diff| {np.nanmax(np.abs(diff[valid])):.4f} m, "
          f"identical pixels {100 * np.mean(diff[valid] == 0):.2f}%")
    assert valid.all(), "cropped GLO-30 has invalid pixels"


if __name__ == "__main__":
    main()
