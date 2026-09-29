#!/usr/bin/env python3
"""Parity: the two pyproj uses the backend dropped, against their rasterio replacements (2026-09-30).
  1. backend/generation/pipeline.py: EPSG:4326 lon/lat -> the item's CRS (pyproj Transformer, always_xy) vs
     rasterio.warp.transform, on every catalog item with a georeference (library manifest).
  2. backend/input/store.py: CRS.from_user_input(crs).is_geographic, pyproj vs rasterio, on every CRS in the catalog
     plus EPSG:4326 (CDSE scenes). Read-only; needs pyproj only here."""
import json
from pathlib import Path

from pyproj import CRS as PCRS, Transformer
from rasterio.crs import CRS as RCRS
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
items = [i for i in json.loads((ROOT / "data/library/manifest.json").read_text())["items"] if i.get("geo")]
worst = 0.0
for it in items:
    g = it["geo"]
    px, py = Transformer.from_crs("EPSG:4326", g["crs"], always_xy=True).transform(g["lon"], g["lat"])
    (rx,), (ry,) = transform("EPSG:4326", g["crs"], [g["lon"]], [g["lat"]])
    worst = max(worst, abs(px - rx), abs(py - ry))
crss = sorted({i["geo"]["crs"] for i in items} | {"EPSG:4326"})
geo_ok = all(PCRS.from_user_input(c).is_geographic == RCRS.from_user_input(c).is_geographic for c in crss)
print(json.dumps({"items": len(items), "max_abs_coord_diff_m": worst, "crs_checked": crss, "is_geographic_identical": geo_ok}))
