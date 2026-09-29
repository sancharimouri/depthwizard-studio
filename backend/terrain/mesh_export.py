# Export terrain arrays/mesh data for Three.js.
"""
Write a surface raster in the exact asset format the frontend viewer loads
(`frontend/src/viewer.js` loadRegion): `terrain.json` with
{width, height, bounds{west,south,east,north}, elevationMin, elevationMax, heights[]}
(heights row-major, normalised to [0,1]) — the same schema scripts/prepare_demo_data.py
writes for the four demo regions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def block_mean(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """NaN-aware area average of `a` onto an (h, w) grid (exact integer blocks after cropping)."""
    h, w = out_hw
    fy, fx = a.shape[0] // h, a.shape[1] // w
    b = a[: h * fy, : w * fx].reshape(h, fy, w, fx)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """NaN cells take the value of the nearest valid cell (no-op without NaN)."""
    bad = ~np.isfinite(grid)
    if not bad.any() or bad.all():
        return grid
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(bad, return_distances=True, return_indices=True)
    return grid[iy, ix]


def write_terrain_json(path: Path, surface: np.ndarray, bounds_lonlat: tuple[float, float, float, float],
                       mesh_hw: tuple[int, int] = (512, 512), display: np.ndarray | None = None,
                       limit_outliers: bool = True) -> dict:
    """Downsample `surface` to the mesh grid, fill any NaN from its nearest valid neighbour,
    normalise, and write. (Filling with the tile minimum drew every hole, e.g. a DEM tile seam,
    as a spike down to the lowest point of the tile.)
    `display` (optional, e.g. a Maxar pack's cosmetic DISPLAY band, scripts/build_maxar_display_packs.py)
    is written as {elevationMin, elevationMax, heights} under "display": frontend/src/terrain.js extrudes
    it instead of `heights`; statistics, readouts and Measure keep using the real `heights`.
    limit_outliers=False writes "limitOutliers": false: the viewer then skips its outlier limiter
    (frontend/src/outlier-relief.js), e.g. for DFC2019 reference-lidar surfaces, where tall cells are buildings."""
    def grid_of(a):
        g = fill_nan_nearest(block_mean(a.astype(np.float64), mesh_hw))
        lo, hi = float(np.nanmin(g)), float(np.nanmax(g))
        return lo, hi, ((g - lo) / (hi - lo) if hi > lo else np.zeros_like(g))

    lo, hi, norm = grid_of(surface)
    west, south, east, north = bounds_lonlat
    terrain = {"width": int(mesh_hw[1]), "height": int(mesh_hw[0]),
               "bounds": {"west": west, "south": south, "east": east, "north": north},
               "elevationMin": lo, "elevationMax": hi,
               "heights": [round(float(v), 6) for v in norm.ravel()]}
    if display is not None:
        dlo, dhi, dnorm = grid_of(np.asarray(display))
        terrain["display"] = {"elevationMin": dlo, "elevationMax": dhi, "heights": [round(float(v), 6) for v in dnorm.ravel()]}
    if not limit_outliers:
        terrain["limitOutliers"] = False
    Path(path).write_text(json.dumps(terrain, separators=(",", ":")))
    return {k: v for k, v in terrain.items() if k not in ("heights", "display")}
