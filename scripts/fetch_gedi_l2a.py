#!/usr/bin/env python3
"""
Phase 3b (2026-09-23): GEDI L2A footprints for the 25 accepted Sentinel-2
tiles + Darjeeling, via Earth Engine (EARTHENGINE_PROJECT). Data acquisition only.

Source: LARSE/GEDI/GEDI02_A_002_MONTHLY (each L2A shot rasterized at 25 m).
That collection has no lat/lon_lowestmode bands; footprint position is taken
from lat_highestreturn/lon_highestreturn (same shot, geolocated return,
within metres of the lowest-mode position -- negligible vs the ~25 m footprint).
Filter: quality_flag == 1, degrade_flag == 0, sensitivity > 0.95.
Exports per footprint: rh98, elev_lowestmode, lat, lon, date (from
delta_time, seconds since 2018-01-01T00:00:00Z), plus beam/sensitivity.
elev_lowestmode is WGS84 ellipsoidal (GEDI L2A convention); rh98 is metres
above the lowest mode. Surface height = elev_lowestmode + rh98.
Writes data/gedi_l2a/<tile>.csv.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import ee
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import query_icesat2_photons as q  # noqa: E402

OUT = ROOT / "data/gedi_l2a"
BANDS = ["rh98", "elev_lowestmode", "lat_highestreturn", "lon_highestreturn", "delta_time",
         "sensitivity", "quality_flag", "degrade_flag", "beam"]


def tiles():
    m = pd.read_csv(ROOT / "data/sentinel2_benchmark/manifest.csv").set_index("tile_id")
    v = pd.read_csv(ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv").set_index("tile_id")
    for t in v[~v["flagged"]].index:
        yield t, q.utm_bbox_to_wgs84_poly(ast.literal_eval(m.loc[t, "bbox_utm"]), int(m.loc[t, "bbox_utm_epsg"]))
    with rasterio.open(ROOT / "data/diagnostics/darjeeling/Darjeeling_RGB_committed_660ecb6.tif") as s:
        b = s.bounds
        yield "darjeeling", q.utm_bbox_to_wgs84_poly([b.left, b.bottom, b.right, b.top], s.crs.to_epsg())


def fetch(poly) -> pd.DataFrame:
    region = ee.Geometry.Polygon([[p["lon"], p["lat"]] for p in poly])

    def per_image(img):
        img = img.select(BANDS)
        ok = img.select("quality_flag").eq(1).And(img.select("degrade_flag").eq(0)).And(img.select("sensitivity").gt(0.95))
        return img.updateMask(ok).sample(region=region, scale=25, dropNulls=True, geometries=False)

    fc = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY").filterBounds(region).map(per_image).flatten()
    df = ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    if len(df) == 0:
        return pd.DataFrame(columns=["lat", "lon", "rh98", "elev_lowestmode", "date", "sensitivity", "beam"])
    df = df.rename(columns={"lat_highestreturn": "lat", "lon_highestreturn": "lon"})
    df["date"] = (pd.Timestamp("2018-01-01", tz="UTC") + pd.to_timedelta(df["delta_time"], unit="s")).dt.date
    return df[["lat", "lon", "rh98", "elev_lowestmode", "date", "sensitivity", "beam", "delta_time"]]


def main():
    proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
    ee.Initialize(project=proj)
    OUT.mkdir(parents=True, exist_ok=True)
    for t, poly in tiles():
        path = OUT / f"{t}.csv"
        if path.exists():
            print(f"{t}: exists, skip", flush=True)
            continue
        df = fetch(poly)
        df.to_csv(path, index=False)
        print(f"{t}: {len(df)} valid GEDI shots", flush=True)


if __name__ == "__main__":
    main()
