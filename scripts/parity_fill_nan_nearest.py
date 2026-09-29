#!/usr/bin/env python3
"""Parity: backend/terrain/mesh_export.fill_nan_nearest (numpy) vs scipy.ndimage.distance_transform_edt, on real
grids with gaps (raw FABDEM / GLO-30 of coastal and hilly Sentinel-2 tiles, block-meaned to the app's mesh grid as
write_terrain_json does) and on synthetic masks. Read-only; needs scipy only here (the backend no longer does).
Reports: max |value diff|, cells that differ, and whether every filled cell took a nearest cell (distance equal to
scipy's choice): value differences can then only come from equidistant ties."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.terrain.mesh_export import block_mean, fill_nan_nearest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def scipy_fill(g):
    bad = ~np.isfinite(g)
    _, (iy, ix) = distance_transform_edt(bad, return_distances=True, return_indices=True)
    return g[iy, ix], iy, ix


def ours_with_index(g):
    # same algorithm as fill_nan_nearest, returning the chosen indices (for the distance check)
    bad = ~np.isfinite(g)
    out = fill_nan_nearest(g)
    return out, bad


def check(name, g):
    bad = ~np.isfinite(g)
    if not bad.any():
        return None
    a = fill_nan_nearest(g)
    b, iy, ix = scipy_fill(g)
    qy, qx = np.nonzero(bad)
    # distance of our pick: find, per NaN cell, the valid cell with value a and minimal distance == scipy distance?
    ds = np.hypot(qy - iy[bad], qx - ix[bad])
    gy, gx = np.nonzero(~bad)
    # exact nearest distance per NaN cell (brute force over all valid cells, chunked)
    dmin = np.empty(len(qy))
    for i in range(0, len(qy), 2000):
        d2 = (qy[i:i + 2000, None] - gy) ** 2 + (qx[i:i + 2000, None] - gx) ** 2
        dmin[i:i + 2000] = np.sqrt(d2.min(1))
    # our value must equal the value of SOME valid cell at distance dmin
    ok = 0
    for j in range(len(qy)):
        d2 = (qy[j] - gy) ** 2 + (qx[j] - gx) ** 2
        cand = g[gy[d2 == d2.min()], gx[d2 == d2.min()]]
        ok += np.any(cand == a[qy[j], qx[j]])
    diff = np.abs(a - b)
    return {"grid": name, "shape": list(g.shape), "nan_cells": int(bad.sum()), "max_abs_diff": float(diff.max()),
            "cells_differ": int((diff > 0).sum()), "ours_is_nearest": int(ok) == len(qy),
            "scipy_is_nearest": bool(np.allclose(ds, dmin))}


rows = []
for f in sorted(glob.glob(str(ROOT / "data/sentinel2_benchmark/fabdem/*_fabdem.npy"))):
    a = np.load(f).astype(np.float64)
    if np.isfinite(a).all():
        continue
    r = check(Path(f).stem, block_mean(a, (334, 334)))
    if r:
        rows.append(r)
import rasterio  # noqa: E402

for f in sorted(glob.glob(str(ROOT / "data/elevation/*/*.tif")) + glob.glob(str(ROOT / "data/vhr_dsm/*_margin192/dsm.tif"))
                + glob.glob(str(ROOT / "data/sentinel2_benchmark/copernicus_dem_raw/*_dem.tif"))):
    with rasterio.open(f) as src:
        a = src.read(1, masked=True).astype(np.float64).filled(np.nan)
    if np.isfinite(a).all() or not np.isfinite(a).any():
        continue
    h, w = a.shape
    r = check(str(Path(f).relative_to(ROOT)), block_mean(a, (min(334, h), min(334, w))))
    if r:
        rows.append(r)
        continue
    # gaps vanish at mesh resolution (a block is NaN only when all of it is): test the real gap pattern at native
    # resolution instead, on a 400 x 400 window around the gaps
    ys, xs = np.nonzero(~np.isfinite(a))
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = max(0, min(h - 400, cy - 200)), max(0, min(w - 400, cx - 200))
    r = check(str(Path(f).relative_to(ROOT)) + f" [native window {y0}:{y0 + 400},{x0}:{x0 + 400}]", a[y0:y0 + 400, x0:x0 + 400])
    if r:
        rows.append(r)
rng = np.random.default_rng(0)
for k in range(3):
    g = rng.normal(size=(341, 341)).cumsum(0)
    g[rng.random(g.shape) < [0.01, 0.2, 0.6][k]] = np.nan
    rows.append(check(f"synthetic_{[1, 20, 60][k]}pct_nan", g))
for r in rows:
    print(json.dumps(r))
print("ALL NEAREST:", all(r["ours_is_nearest"] for r in rows), "| worst max_abs_diff:", max(r["max_abs_diff"] for r in rows))
