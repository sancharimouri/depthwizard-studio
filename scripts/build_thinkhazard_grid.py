"""Bundled ThinkHazard! district lookup by coordinates (docs/facts-research.md, v2 Part D).

ThinkHazard! (GFDRR) rates hazards per GAUL 2015 admin-2 division, and its division code IS the GAUL `adm2_code`
(checked 2026-09-30: Darjiling 17948, Kachchh 17643, South Sikkim 17863, Hyderabad 17553, Chennai 70242, Duval 29012).
ThinkHazard has no point lookup, so this builds one:

  1. GAUL 2015 level 2 polygons, paged from FAO's public WFS (gaul:g2015_2014_2, EPSG:4326), are burnt into a global
     0.025° (~2.8 km) grid of ROW INDICES (uint16; 0 = no division). Only this coarse derived grid is shipped, never
     the boundaries.
  2. ThinkHazard's per-hazard division levels (admindiv_hazardsets/<HT>.json) give each row its levels.

Writes backend/facts/data/thinkhazard_adm2.tif (COG) and thinkhazard_levels.json:
  {"hazards": "FL UF CF EQ LS TS VA CY WF", "levels": {"H": "high", ...},
   "rows": [null, [code, name, admin1, admin0, "HMLV-..."], ...]}   (row index = grid value)

  python scripts/build_thinkhazard_grid.py --cache <scratch dir>
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backend/facts/data"
WFS = "https://data.apps.fao.org/map/gsrv/gsrv1/gaul/wfs"
TH = "https://thinkhazard.org/en/admindiv_hazardsets/{}.json"
HAZARDS = ["FL", "UF", "CF", "EQ", "LS", "TS", "VA", "CY", "WF"]
LEVEL_CHAR = {"HIG": "H", "MED": "M", "LOW": "L", "VLO": "V"}
RES = 0.025
WEST, NORTH, SOUTH = -180.0, 84.0, -60.0
UA = {"User-Agent": "DepthWizard2/0.1 (SIH26175 prototype; one-off build of a district lookup grid)"}


def pages(client, cache: Path, count=1000):
    start = 0
    while True:
        f = cache / f"gaul2_{start:06d}.json.gz"
        if f.is_file():
            data = json.loads(gzip.decompress(f.read_bytes()))
        else:
            for attempt in range(4):
                try:
                    r = client.get(WFS, params={
                        "service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": "gaul:g2015_2014_2",
                        "outputFormat": "application/json", "srsName": "EPSG:4326", "sortBy": "FID", "count": count,
                        "startIndex": start, "propertyName": "adm2_code,adm2_name,adm1_name,adm0_name,wkb_geometry"})
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as exc:  # noqa: BLE001 — retry transient failures, then stop loudly
                    print(f"  page {start}: {type(exc).__name__}, retry {attempt + 1}", flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                raise SystemExit(f"GAUL page at {start} failed 4 times")
            f.write_bytes(gzip.compress(json.dumps(data).encode(), 6))
        feats = data.get("features", [])
        yield start, feats
        if len(feats) < count:
            return
        start += count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    width, height = round(360 / RES), round((NORTH - SOUTH) / RES)
    transform = from_origin(WEST, NORTH, RES, RES)
    grid = np.zeros((height, width), np.uint16)
    rows: list = [None]
    row_of: dict[int, int] = {}
    t0 = time.time()
    with httpx.Client(timeout=300, headers=UA, follow_redirects=True) as client:
        for start, feats in pages(client, a.cache):
            shapes = []
            for f in feats:
                p, g = f["properties"], f.get("geometry")
                if not g or p.get("adm2_code") is None:
                    continue
                code = int(p["adm2_code"])
                if code not in row_of:
                    row_of[code] = len(rows)
                    rows.append([code, p.get("adm2_name"), p.get("adm1_name"), p.get("adm0_name"), None])
                shapes.append((shape(g), row_of[code]))
            if shapes:
                # all_touched=False: a cell belongs to the division covering its centre; later pages never
                # overwrite an earlier burn (the mask below), so overlaps resolve deterministically by FID
                burn = rasterize(shapes, out_shape=grid.shape, transform=transform, fill=0, dtype="uint16")
                np.copyto(grid, burn, where=(grid == 0) & (burn > 0))
            print(f"GAUL features {start + len(feats):6d}  divisions {len(rows) - 1:6d}  {time.time() - t0:6.0f}s", flush=True)
        assert len(rows) < 65535
        levels: dict[int, dict[str, str]] = {}
        for hz in HAZARDS:
            f = a.cache / f"th_{hz}.json.gz"
            if not f.is_file():
                r = client.get(TH.format(hz))
                r.raise_for_status()
                f.write_bytes(gzip.compress(r.content, 6))
            for d in json.loads(gzip.decompress(f.read_bytes())):
                lv = LEVEL_CHAR.get(d.get("hazard_level"))
                if lv:
                    levels.setdefault(int(d["code"]), {})[hz] = lv
            print(f"ThinkHazard {hz}: {sum(hz in v for v in levels.values())} divisions rated", flush=True)
    for r in rows[1:]:
        lv = levels.get(r[0], {})
        r[4] = "".join(lv.get(hz, "-") for hz in HAZARDS)
    OUT.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "COG", "dtype": "uint16", "count": 1, "width": width, "height": height, "crs": "EPSG:4326",
            "transform": transform, "nodata": 0, "compress": "DEFLATE", "predictor": 2, "blocksize": 512,
            "overviews": "NONE"}
    tif = OUT / "thinkhazard_adm2.tif"
    with rasterio.open(tif, "w", **prof) as dst:
        dst.write(grid, 1)
        dst.update_tags(SOURCE="GAUL 2015 admin-2 (FAO) rasterised at 0.025 deg; values index thinkhazard_levels.json rows",
                        BUILT=time.strftime("%Y-%m-%d"))
    meta = {"built": time.strftime("%Y-%m-%d"), "grid": tif.name, "resolution_deg": RES,
            "hazards": " ".join(HAZARDS), "levels": {"H": "high", "M": "medium", "L": "low", "V": "very low", "-": None},
            "sources": {"divisions": "GAUL 2015 admin-2 (FAO), via data.apps.fao.org WFS gaul:g2015_2014_2",
                        "levels": "ThinkHazard! (GFDRR), admindiv_hazardsets/<HT>.json, CC BY"},
            "rows": rows}
    (OUT / "thinkhazard_levels.json").write_text(json.dumps(meta, separators=(",", ":"), ensure_ascii=False))
    rated = sum(1 for r in rows[1:] if r[4].strip("-"))
    print(f"{len(rows) - 1} divisions ({rated} rated), grid {width}x{height}, "
          f"{tif.stat().st_size / 1e6:.1f} MB + {(OUT / 'thinkhazard_levels.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
