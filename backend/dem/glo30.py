"""Copernicus GLO-30 DSM, read live from the public COGs on AWS (no account, no key).

  https://copernicus-dem-30m.s3.amazonaws.com/<tile>/<tile>.tif   (1x1 degree, EPSG:4326)

Only the window covering the requested footprint is fetched (HTTP range reads via GDAL's
/vsicurl/). Used by real generation (backend/generation/pipeline.py) for georeferenced
inputs that have no bundled elevation pack, e.g. uploads and Search Online scenes in the
desktop app (where the Earth Engine FABDEM fetch is not available).
Checked 2026-09-26: the Darjeeling window reads 556.5-2477.5 m, identical to the
OpenTopography DSM the demo uses (which is GLO-30).
"""
from __future__ import annotations

import math
import os

import numpy as np

BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


class Glo30Error(RuntimeError):
    """GLO-30 could not be read for this footprint (offline, or no tile covers it)."""


def _tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def fetch_grid(crs, bounds, shape, transform) -> np.ndarray:
    """GLO-30 elevation (m, NaN = no data) on the target grid: `shape` (h, w), `transform`,
    `crs`; `bounds` (left, bottom, right, top) in that CRS."""
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import from_bounds

    if not os.environ.get("CURL_CA_BUNDLE"):
        try:  # GDAL's curl needs a CA bundle inside the frozen app
            import certifi
            os.environ["CURL_CA_BUNDLE"] = certifi.where()
        except ImportError:
            pass
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    out = np.full(shape, np.nan, np.float32)
    tiles = 0
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_TIMEOUT="60"):
        for lat in range(math.floor(s), math.floor(n) + 1):
            for lon in range(math.floor(w), math.floor(e) + 1):
                name = _tile_name(lat, lon)
                try:
                    src = rasterio.open(f"/vsicurl/{BASE}/{name}/{name}.tif")
                except rasterio.errors.RasterioIOError:
                    continue  # no tile here (open ocean)
                with src:
                    win = from_bounds(max(w, lon) - 0.01, max(s, lat) - 0.01, min(e, lon + 1) + 0.01,
                                      min(n, lat + 1) + 0.01, src.transform).round_offsets().round_lengths()
                    data = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                    part = np.full(shape, np.nan, np.float32)
                    reproject(data, part, src_transform=src.window_transform(win), src_crs=src.crs,
                              dst_transform=transform, dst_crs=crs, resampling=Resampling.bilinear,
                              src_nodata=np.nan, dst_nodata=np.nan)
                    out = np.where(np.isfinite(part), part, out)
                    tiles += 1
    if tiles == 0 or not np.isfinite(out).any():
        raise Glo30Error("No Copernicus GLO-30 data could be read for this footprint (offline, or open ocean).")
    return out
