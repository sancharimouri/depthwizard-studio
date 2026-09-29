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
    """NaN cells take the value of the nearest valid cell (Euclidean; no-op without NaN).

    numpy only (was scipy.ndimage.distance_transform_edt; parity in docs/container-measurements.md). The nearest
    valid cell of any NaN cell is always a valid cell with a non-valid 8-neighbour: from any other valid cell, the
    step towards the NaN cell lands on a valid cell that is strictly closer. So only those boundary cells are
    searched, in blocks, which keeps it small (the mesh grids are at most 400 x 400)."""
    bad = ~np.isfinite(grid)
    if not bad.any() or bad.all():
        return grid
    good = ~bad
    pad = np.pad(bad, 1, constant_values=False)
    near_bad = np.zeros_like(bad)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                near_bad |= pad[1 + dy: 1 + dy + bad.shape[0], 1 + dx: 1 + dx + bad.shape[1]]
    # column-major order: argmin keeps the first of equal distances, so ties go to the smallest column, then the
    # smallest row, which is scipy's choice (matched on 17,647 of 17,647 tied cells, 2026-09-30)
    bx, by = np.nonzero((good & near_bad).T)
    qy, qx = np.nonzero(bad)
    out = grid.copy()
    step = max(1, 4_000_000 // max(1, len(by)))  # bound the distance block at ~4 M entries
    for i in range(0, len(qy), step):
        y, x = qy[i:i + step, None], qx[i:i + step, None]
        k = np.argmin((y - by) ** 2 + (x - bx) ** 2, axis=1)
        out[qy[i:i + step], qx[i:i + step]] = grid[by[k], bx[k]]
    return out


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
