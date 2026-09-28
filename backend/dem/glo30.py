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


def mosaic_to_grid(sources, crs, bounds_lonlat, shape, transform) -> np.ndarray:
    """Mosaic the GLO-30 tiles `sources` (open rasterio datasets, EPSG:4326) over `bounds_lonlat`
    (w, s, e, n) in their own lon/lat grid FIRST, then reproject once onto the target grid.

    Reprojecting tile by tile (the old way) leaves a seam: bilinear resampling needs pixels
    from both sides of a tile edge, each tile's own read is NaN beyond its edge, so target
    pixels along the edge came out NaN in every part. On Darjeeling (south edge exactly on
    27°N) that was 6 pixels, later filled with the tile minimum and drawn as downward spikes."""
    from rasterio.merge import merge
    from rasterio.warp import Resampling, reproject

    w, s, e, n = bounds_lonlat
    pad = 0.01  # a little beyond the footprint so edge pixels have interpolation neighbours
    mosaic, mosaic_tf = merge(list(sources), bounds=(w - pad, s - pad, e + pad, n + pad), nodata=np.nan,
                              dtype="float32")
    out = np.full(shape, np.nan, np.float32)
    reproject(mosaic[0], out, src_transform=mosaic_tf, src_crs="EPSG:4326", dst_transform=transform,
              dst_crs=crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def fetch_grid(crs, bounds, shape, transform) -> np.ndarray:
    """GLO-30 elevation (m, NaN = no data) on the target grid: `shape` (h, w), `transform`,
    `crs`; `bounds` (left, bottom, right, top) in that CRS."""
    import rasterio
    from rasterio.warp import transform_bounds

    if not os.environ.get("CURL_CA_BUNDLE"):
        try:  # GDAL's curl needs a CA bundle inside the frozen app
            import certifi
            os.environ["CURL_CA_BUNDLE"] = certifi.where()
        except ImportError:
            pass
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_TIMEOUT="60"):
        sources = []
        try:
            for lat in range(math.floor(s - 0.01), math.floor(n + 0.01) + 1):
                for lon in range(math.floor(w - 0.01), math.floor(e + 0.01) + 1):
                    name = _tile_name(lat, lon)
                    try:
                        sources.append(rasterio.open(f"/vsicurl/{BASE}/{name}/{name}.tif"))
                    except rasterio.errors.RasterioIOError:
                        continue  # no tile here (open ocean)
            out = mosaic_to_grid(sources, crs, (w, s, e, n), shape, transform) if sources else None
        finally:
            for src in sources:
                src.close()
    if out is None or not np.isfinite(out).any():
        raise Glo30Error("No Copernicus GLO-30 data could be read for this footprint (offline, or open ocean).")
    return out
