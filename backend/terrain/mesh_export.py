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
                       mesh_hw: tuple[int, int] = (512, 512)) -> dict:
    """Downsample `surface` to the mesh grid, fill any NaN from its nearest valid neighbour,
    normalise, and write. (Filling with the tile minimum drew every hole, e.g. a DEM tile seam,
    as a spike down to the lowest point of the tile.)"""
    grid = fill_nan_nearest(block_mean(surface.astype(np.float64), mesh_hw))
    lo = float(np.nanmin(grid))
    hi = float(np.nanmax(grid))
    norm = (grid - lo) / (hi - lo) if hi > lo else np.zeros_like(grid)
    west, south, east, north = bounds_lonlat
    terrain = {"width": int(mesh_hw[1]), "height": int(mesh_hw[0]),
               "bounds": {"west": west, "south": south, "east": east, "north": north},
               "elevationMin": lo, "elevationMax": hi,
               "heights": [round(float(v), 6) for v in norm.ravel()]}
    Path(path).write_text(json.dumps(terrain, separators=(",", ":")))
    return {k: v for k, v in terrain.items() if k != "heights"}
