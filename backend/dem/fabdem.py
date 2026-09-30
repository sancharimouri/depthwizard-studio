"""FABDEM (bare-earth DEM) fetch via Google Earth Engine.

The same fetch the research scripts use (scripts/brazil_benchmark.py,
scripts/vhr_dsm_pipeline.py): the community FABDEM collection, mosaicked,
given its NATIVE ~30 m projection before bilinear resampling. (A bare
mosaic() has no native projection, so resample() would interpolate on EE's
default 1° grid and return a smooth ramp instead of the DEM — see
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md.)

Requires an Earth Engine project in EARTHENGINE_PROJECT (.env) and local EE
credentials (`earthengine authenticate`).
"""

from __future__ import annotations

import math
import os
import threading

import numpy as np

FABDEM_ASSET = "projects/sat-io/open-datasets/FABDEM"
FABDEM_RES_M = 30.0
MAX_SIDE_PX = 4096  # computePixels request size guard (~120 km at 30 m)

_initialised = False
_init_lock = threading.Lock()  # concurrent requests (Cloud Run concurrency 4) initialise EE once


class FabdemError(RuntimeError):
    """Earth Engine unavailable, or the request failed / was too large."""


def _ee():
    global _initialised
    try:
        import ee
    except ImportError as exc:  # pragma: no cover - dependency present in this repo
        raise FabdemError("earthengine-api is not installed") from exc
    with _init_lock:
        if not _initialised:
            project = os.environ.get("EARTHENGINE_PROJECT")
            if not project:
                raise FabdemError("EARTHENGINE_PROJECT is not set — FABDEM is fetched through Google Earth Engine")
            try:
                # credentials: ~/.config/earthengine locally; on Cloud Run, Application Default Credentials, i.e.
                # the service's runtime service account (earthengine-api falls back to google.auth.default())
                ee.Initialize(project=project)
            except Exception as exc:  # noqa: BLE001 - EE raises a variety of auth errors
                raise FabdemError(f"Earth Engine initialisation failed: {exc}") from exc
            _initialised = True
    return ee


def fabdem_image(ee):
    col = ee.ImageCollection(FABDEM_ASSET)
    return col.mosaic().setDefaultProjection(col.first().projection()).resample("bilinear")


def fetch_grid(crs: str, left: float, bottom: float, right: float, top: float, res: float) -> dict:
    """FABDEM on a north-up grid in `crs` covering the given bounds at `res`
    (CRS units per pixel). Returns {elevation (float32, NaN = no data), transform
    (a, b, c, d, e, f), crs, width, height}."""
    width = max(1, math.ceil((right - left) / res))
    height = max(1, math.ceil((top - bottom) / res))
    if width > MAX_SIDE_PX or height > MAX_SIDE_PX:
        raise FabdemError(
            f"area too large for one FABDEM request ({width}×{height} px at {res:g}; max {MAX_SIDE_PX} per side)"
        )
    ee = _ee()
    try:
        arr = ee.data.computePixels({
            "expression": fabdem_image(ee),
            "fileFormat": "NUMPY_NDARRAY",
            "grid": {
                "dimensions": {"width": width, "height": height},
                "affineTransform": {"scaleX": res, "shearX": 0, "translateX": left,
                                    "shearY": 0, "scaleY": -res, "translateY": top},
                "crsCode": crs,
            },
        })
    except Exception as exc:  # noqa: BLE001
        raise FabdemError(f"FABDEM request failed: {exc}") from exc
    band = arr[arr.dtype.names[0]] if arr.dtype.names else arr
    elev = np.asarray(band, dtype=np.float32)
    elev[(elev < -1000) | (elev > 9000)] = np.nan
    return {"elevation": elev, "transform": (res, 0.0, left, 0.0, -res, top), "crs": crs,
            "width": width, "height": height}


def native_res_for(crs_is_geographic: bool) -> float:
    """FABDEM's ~30 m native spacing in the request CRS's units."""
    return 1.0 / 3600.0 if crs_is_geographic else FABDEM_RES_M
