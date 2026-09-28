#!/usr/bin/env python3
"""Base ground elevation for the 50 DFC2019 tiles (08-dfc2019-terrain-packs, Prompt 3).

  city     city-level fallback: 3DEP DEM median (+ p5/p95) over the union of the city's US3D
           point-cloud footprints (where any DFC2019 tile can lie).
  located  confidently located tiles (Prompt 2 locations.json): finest 3DEP bare-earth DEM
           (1 m where it has valid coverage, else 1/3 arc-second) sampled on the pack grid in the
           tile's own pixel frame (orientation + GSD from the match).
  datum    verification only, never shipped: DEM vs the point cloud's own ground returns over the
           same windows, per city, before and after the geoid (ellipsoid -> NAVD88) correction.
  write    the base layer for all 50 tiles -> data/dfc2019/terrain_packs/base/<tile>.npz

Everything with coordinates stays in the gitignored data/dfc2019/terrain_packs/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "ON")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import httpx
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
SCR = ROOT / "data/dfc2019/terrain_packs"
LOC = SCR / "locate"
BASE = SCR / "base"
EPSG = {"JAX": 32617, "OMA": 32614}           # the US3D clouds' UTM zones (verified on the footprints)
PACK_N = 341                                   # pack grid = the app's mesh grid for a 1024 px tile
TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
DS_1M = "Digital Elevation Model (DEM) 1 meter"
DS_13 = "National Elevation Dataset (NED) 1/3 arc-second"
_http = httpx.Client(timeout=120, follow_redirects=True)


def tnm_products(dataset, lon0, lat0, lon1, lat1):
    j = _http.get(TNM, params={"datasets": dataset, "bbox": f"{lon0},{lat0},{lon1},{lat1}",
                               "prodFormats": "GeoTIFF", "max": 100}).json()
    return j.get("items", [])


def current_13(lon, lat):
    """The newest 1/3 arc-second product covering (lon, lat)."""
    items = tnm_products(DS_13, lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
    items.sort(key=lambda i: i.get("publicationDate") or "", reverse=True)
    return items[0]


def dem_on_grid(url, dst_crs, dst_transform, shape, resampling=Resampling.bilinear):
    out = np.full(shape, np.nan, np.float32)
    with rasterio.open("/vsicurl/" + url) as r:
        reproject(rasterio.band(r, 1), out, src_nodata=r.nodata, dst_transform=dst_transform,
                  dst_crs=dst_crs, dst_nodata=np.nan, resampling=resampling)
    return out


# ----------------------------------------------------------------------------- city level
def cmd_city(a):
    from rasterio.transform import from_origin
    index = json.loads((LOC / "cloud_index.json").read_text())
    out = {}
    for city, epsg in EPSG.items():
        tiles = {k: v for k, v in index.items() if k.startswith(city)}
        e0 = min(v["e0"] for v in tiles.values()); n1 = max(v["n0"] for v in tiles.values()) + 512
        e1 = max(v["e0"] for v in tiles.values()) + 512; n0 = min(v["n0"] for v in tiles.values())
        res = 10.0
        W, H = int(np.ceil((e1 - e0) / res)), int(np.ceil((n1 - n0) / res))
        tf = from_origin(e0, n1, res, res)
        mask = np.zeros((H, W), bool)          # union of the cloud-tile squares
        for v in tiles.values():
            c0, r0 = int((v["e0"] - e0) / res), int((n1 - v["n0"] - 512) / res)
            mask[r0:r0 + 52, c0:c0 + 52] = True
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform((e0 + e1) / 2, (n0 + n1) / 2)
        prod = current_13(lon, lat)
        dem = dem_on_grid(prod["downloadURL"], f"EPSG:{epsg}", tf, (H, W))
        v = dem[mask & np.isfinite(dem)]
        out[city] = {"source": prod["title"], "url": prod["downloadURL"], "grid_m": res,
                     "cells": int(v.size), "valid_frac": float(np.isfinite(dem[mask]).mean()),
                     "median_m": round(float(np.median(v)), 2), "p5_m": round(float(np.percentile(v, 5)), 2),
                     "p95_m": round(float(np.percentile(v, 95)), 2), "vertical": "NAVD88 (3DEP)"}
        print(city, out[city], flush=True)
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "city_level.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- located tiles
K_INV = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}


def dihedral(a, k):
    if k >= 4:
        a = a[:, ::-1]
    return np.rot90(a, k % 4)


def tile_grid_utm(loc, n=PACK_N):
    """UTM (E, N) of the pack-grid cell centres, in the TILE's own pixel frame.
    The match placed dihedral(tile, orient) north-up in UTM, so window coordinates are built
    north-up and mapped back to the tile frame with the inverse transform."""
    k = loc["orient"]
    # the tile is square (1024 px), so the window is square whatever the orientation
    size = 1024 * loc["gsd"]
    step = size / n
    xs = loc["centre_e"] - size / 2 + (np.arange(n) + 0.5) * step
    ys = loc["centre_n"] + size / 2 - (np.arange(n) + 0.5) * step
    E, N = np.meshgrid(xs, ys)
    return dihedral(E, K_INV[k]).copy(), dihedral(N, K_INV[k]).copy()


def finest_dem_at(E, N, epsg):
    """Sample the finest 3DEP DEM with valid coverage at the given UTM points (bilinear)."""
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = tr.transform(E, N)
    bbox = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    tried = []
    for ds in (DS_1M, DS_13):
        items = tnm_products(ds, *bbox)
        items.sort(key=lambda i: i.get("publicationDate") or "", reverse=True)
        out = np.full(E.shape, np.nan, np.float32)
        used = []
        for it in items:
            if "historical" in it["downloadURL"] and ds == DS_13 and used:
                continue
            with rasterio.open("/vsicurl/" + it["downloadURL"]) as r:
                x, y = Transformer.from_crs(epsg, r.crs, always_xy=True).transform(E, N)
                win = rasterio.windows.from_bounds(x.min() - 5, y.min() - 5, x.max() + 5, y.max() + 5, r.transform)
                win = win.round_offsets().round_lengths()
                try:
                    a = r.read(1, window=win, boundless=True, fill_value=r.nodata if r.nodata is not None else np.nan).astype(np.float64)
                except Exception as exc:  # noqa: BLE001
                    tried.append((it["title"], type(exc).__name__)); continue
                if r.nodata is not None:
                    a[a == r.nodata] = np.nan
                a[a < -1000] = np.nan
                wt = rasterio.windows.transform(win, r.transform)
                col = (x - wt.c) / wt.a - 0.5
                row = (y - wt.f) / wt.e - 0.5
                from scipy.ndimage import map_coordinates
                v = map_coordinates(a, [row.ravel(), col.ravel()], order=1, mode="nearest", cval=np.nan).reshape(E.shape)
                fill = ~np.isfinite(out) & np.isfinite(v)
                if fill.any():
                    out[fill] = v[fill]
                    used.append(it["title"])
            if np.isfinite(out).all():
                break
        if np.isfinite(out).mean() > 0.99:
            if not np.isfinite(out).all():
                from scipy.ndimage import distance_transform_edt
                _, (ri, ci) = distance_transform_edt(~np.isfinite(out), return_indices=True)
                out = out[ri, ci]
            return out, {"dataset": "3DEP 1 m" if ds == DS_1M else "3DEP 1/3 arc-second", "products": used}
        tried.append((ds, f"valid {np.isfinite(out).mean():.2f}"))
    return None, {"tried": tried}


def cmd_located(a):
    locs = json.loads((LOC / "locations.json").read_text())
    BASE.mkdir(parents=True, exist_ok=True)
    done = {}
    for tid, loc in sorted(locs["tiles"].items()):
        if a.tiles and tid not in a.tiles:
            continue
        E, N = tile_grid_utm(loc)
        dem, info = finest_dem_at(E, N, EPSG[loc["city"]])
        done[tid] = info
        if dem is None:
            print(tid, "NO 3DEP", info, flush=True)
            continue
        np.savez(BASE / f"located_{tid}.npz", dem=dem.astype(np.float32), E=E.astype(np.float64), N=N.astype(np.float64))
        print(f"{tid}: {info['dataset']} {info['products'][:2]}  {np.min(dem):.1f}..{np.max(dem):.1f} m", flush=True)
    (BASE / "located_sources.json").write_text(json.dumps(done, indent=1))


# ----------------------------------------------------------------------------- datum check
def cmd_datum(a):
    """DEM (NAVD88) vs the cloud's own ground returns at the same points, per city.
    Cloud ground = class-2 min at 4.8 m (the locate mosaic's ground8), sampled where ground
    returns exist nearby. Verification only; nothing here is shipped."""
    from scipy.ndimage import map_coordinates
    locs = json.loads((LOC / "locations.json").read_text())["tiles"]
    res = {}
    for city, epsg in EPSG.items():
        z = np.load(LOC / f"mosaic_{city}.npz", mmap_mode="r")
        g8, E0, N1, cell = z["ground8"], float(z["E0"]), float(z["N1"]), float(z["cell"]) * 8
        diffs, geo = [], []
        for tid, loc in locs.items():
            if loc["city"] != city or not (BASE / f"located_{tid}.npz").exists():
                continue
            b = np.load(BASE / f"located_{tid}.npz")
            E, N, dem = b["E"][::10, ::10], b["N"][::10, ::10], b["dem"][::10, ::10]
            r = (N1 - N) / cell - 0.5
            c = (E - E0) / cell - 0.5
            cg = map_coordinates(g8, [r.ravel(), c.ravel()], order=1).reshape(E.shape)
            d = (dem - cg).ravel()
            diffs.append(np.median(d))
            # geoid: ellipsoidal (WGS84) -> NAVD88 orthometric height of 0 m at these points
            lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(float(E.mean()), float(N.mean()))
            geo.append((tid, lon, lat))
        if not diffs:
            continue
        # geoid undulation N at the city's located tiles (GEOID18 via PROJ network grids)
        t = Transformer.from_crs("EPSG:4979", "EPSG:4269+5703", always_xy=True)
        und = []
        for _, lon, lat in geo:
            _, _, h = t.transform(lon, lat, 0.0)
            und.append(-h)      # h_ortho(ell=0) = -N  ->  N = -h
        dmed = float(np.median(diffs))
        nmed = float(np.median(und))
        res[city] = {"tiles": len(diffs), "dem_minus_cloud_ground_median_m": round(dmed, 2),
                     "per_tile_median_spread_m": [round(float(np.min(diffs)), 2), round(float(np.max(diffs)), 2)],
                     "geoid_N_median_m": round(nmed, 2),
                     "residual_after_geoid_m": round(dmed + nmed, 2)}
        print(city, res[city], flush=True)
    (BASE / "datum_check.json").write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- write base layer
def cmd_write(a):
    city = json.loads((BASE / "city_level.json").read_text())
    locs = json.loads((LOC / "locations.json").read_text())["tiles"]
    tiles = sorted(p.name[:-8] for p in (ROOT / "data/dfc2019/raw/Truth/Track1-Truth").glob("*_AGL.tif"))
    summary = {}
    for tid in tiles:
        lp = BASE / f"located_{tid}.npz"
        if tid in locs and lp.exists():
            dem = np.load(lp)["dem"]
            info = {"base_source": "dem-located", "dem": json.loads((BASE / "located_sources.json").read_text())[tid]}
        else:
            c = city[tid[:3]]
            dem = np.full((PACK_N, PACK_N), c["median_m"], np.float32)
            info = {"base_source": "city-level", "range_m": [c["p5_m"], c["p95_m"]], "dem": c["source"]}
        np.savez(BASE / f"{tid}.npz", base=dem.astype(np.float32), info=json.dumps(info))
        summary[tid] = {**info, "min_m": round(float(dem.min()), 2), "max_m": round(float(dem.max()), 2)}
    (BASE / "base_summary.json").write_text(json.dumps(summary, indent=1))
    from collections import Counter
    print(Counter(v["base_source"] for v in summary.values()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("city")
    p = sub.add_parser("located"); p.add_argument("--tiles", nargs="*")
    sub.add_parser("datum")
    sub.add_parser("write")
    a = ap.parse_args()
    {"city": cmd_city, "located": cmd_located, "datum": cmd_datum, "write": cmd_write}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
