#!/usr/bin/env python3
"""Fetch real per-photon ICESat-2 ATL08 ground-classified heights for every
tile in the Sentinel-2 benchmark manifest.

Unlike scripts/sentinel_benchmark_icesat_coverage.py (which only records a
segment-level ground-photon *count* via atl08p), this queries sliderule's
atl03sp endpoint with atl08_class=['atl08_ground'], which returns individual
ATL03 photons (lat, lon, h_ph) already restricted server-side to photons
ATL08 classified as class 1 ("ground") -- i.e. the ATL08
land_segments/terrain/h_te_best_fit lineage, at native photon resolution
rather than 100 m segment aggregates. "height" in the output is h_ph
(ellipsoidal photon height, ITRF2014 / EPSG:7912), filtered to
atl08_class == 1.

Writes one CSV per tile to data/icesat2_photons/<tile_id>.csv with columns:
lat, lon, height, rgt, cycle, spot, segment_id.
"""

from __future__ import annotations

import ast
import csv
import math
import sys
import time
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from sliderule import icesat2, sliderule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "sentinel2_benchmark" / "manifest.csv"
OUT_DIR = PROJECT_ROOT / "data" / "icesat2_photons"


def utm_bbox_to_wgs84_poly(bbox_utm: list[float], epsg: int) -> list[dict]:
    xmin, ymin, xmax, ymax = bbox_utm
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    corners_xy = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    poly = []
    for x, y in corners_xy:
        lon, lat = to_wgs84.transform(x, y)
        poly.append({"lon": lon, "lat": lat})
    return poly


def query_tile(poly: list[dict]) -> pd.DataFrame:
    parms = {
        "poly": poly,
        "t0": "2019-01-01T00:00:00Z",
        "t1": "2025-12-31T23:59:59Z",
        "srt": icesat2.SRT_LAND,
        "len": 100,
        "res": 100,
        "atl08_class": ["atl08_ground"],
        "phoreal": {
            "binsize": 1.0,
            "geoloc": "center",
            "use_abs_h": False,
            "send_waveform": False,
            "above_classifier": False,
        },
    }
    gdf = icesat2.atl03sp(parms)
    if gdf is None or len(gdf) == 0:
        return pd.DataFrame(columns=["lat", "lon", "height", "rgt", "cycle", "spot", "segment_id"])

    gdf = gdf[gdf["atl08_class"] == 1]

    out = pd.DataFrame(
        {
            "lat": gdf.geometry.y.values,
            "lon": gdf.geometry.x.values,
            "height": gdf["height"].values,
            "rgt": gdf["rgt"].values,
            "cycle": gdf["cycle"].values,
            "spot": gdf["spot"].values,
            "segment_id": gdf["segment_id"].values,
        }
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sliderule.init("slideruleearth.io", verbose=False)

    manifest = pd.read_csv(MANIFEST)
    print(f"Tiles: {len(manifest)}\n")

    summary_rows = []
    for i, row in manifest.iterrows():
        tile_id = row["tile_id"]
        out_path = OUT_DIR / f"{tile_id}.csv"

        if out_path.exists():
            n = sum(1 for _ in open(out_path)) - 1
            print(f"[{i + 1}/{len(manifest)}] {tile_id:15s} already exists ({n} photons) — skipping")
            summary_rows.append({"tile_id": tile_id, "category": row["category"], "photons": n})
            continue

        bbox_utm = ast.literal_eval(row["bbox_utm"])
        poly = utm_bbox_to_wgs84_poly(bbox_utm, int(row["bbox_utm_epsg"]))

        t0 = time.time()
        try:
            df = query_tile(poly)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i + 1}/{len(manifest)}] {tile_id:15s} ERROR: {exc}")
            summary_rows.append({"tile_id": tile_id, "category": row["category"], "photons": 0})
            continue
        elapsed = time.time() - t0

        df.to_csv(out_path, index=False)
        n = len(df)
        flag = "  ⚠ LOW COUNT" if n < 500 else ""
        print(f"[{i + 1}/{len(manifest)}] {tile_id:15s} {n:6d} photons ({elapsed:.1f}s){flag}")
        summary_rows.append({"tile_id": tile_id, "category": row["category"], "photons": n})

    print("\n" + "=" * 60)
    print("PER-TILE PHOTON COUNTS")
    print("=" * 60)
    for r in sorted(summary_rows, key=lambda r: r["photons"]):
        flag = "  ⚠ LOW COUNT" if r["photons"] < 500 else ""
        print(f"{r['tile_id']:15s} {r['category']:14s} {r['photons']:6d}{flag}")


if __name__ == "__main__":
    main()
