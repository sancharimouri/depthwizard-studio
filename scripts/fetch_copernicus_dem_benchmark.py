"""Fetch real Copernicus GLO-30 (30m) elevation for all 32
data/sentinel2_benchmark/ tiles, from the public AWS Open Data bucket
(`copernicus-dem-30m.s3.amazonaws.com`) -- no API key, no rate limit.

Same source/access method already used successfully in this project for
the shimla/nainital/ooty/manali/dharamshala slope-aspect diagnostic
(docs/method-audit/sentinel2/backbone-terrain-analysis.md, Follow-up 2):
OpenTopography's SRTMGL1 API hit its 50-calls/24hr rate limit there too,
so this bypasses that API entirely -- output is Copernicus GLO-30, not
SRTM, hence the renamed output dir/suffix below.

Access method: each 1x1 degree Copernicus DSM cell is itself a
Cloud-Optimized GeoTIFF, served over plain HTTPS from a public bucket, so
GDAL's /vsicurl/ virtual filesystem can windowed-read just the tile's
footprint (+ buffer) without downloading the full ~110km global cell --
no AWS credentials or S3 API calls needed, just an HTTPS range request.

Idempotent: skips any tile whose output file already exists.

Usage:
    source .venv/bin/activate && python3 scripts/fetch_copernicus_dem_benchmark.py [--limit N]
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from pyproj import Transformer

BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
OUT_DIR = "data/sentinel2_benchmark/copernicus_dem_raw"
BUFFER_DEG = 0.02  # ~2km padding so reprojection onto the tile's UTM grid has no edge gaps
NODATA = -32767.0  # Copernicus DEM's documented nodata value


def utm_bbox_to_wgs84(bbox_utm, epsg):
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon1, lat1 = t.transform(bbox_utm[0], bbox_utm[1])
    lon2, lat2 = t.transform(bbox_utm[2], bbox_utm[3])
    return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def cop_tile_name(lat_deg, lon_deg):
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"


def cells_covering_bbox(w, s, e, n):
    lat0, lat1 = math.floor(s), math.floor(n)
    lon0, lon1 = math.floor(w), math.floor(e)
    return [(lat, lon) for lat in range(lat0, lat1 + 1) for lon in range(lon0, lon1 + 1)]


def fetch_tile(bbox_wgs84, out_path):
    w, s, e, n = bbox_wgs84
    w, s, e, n = w - BUFFER_DEG, s - BUFFER_DEG, e + BUFFER_DEG, n + BUFFER_DEG
    cells = cells_covering_bbox(w, s, e, n)

    srcs = []
    for lat, lon in cells:
        name = cop_tile_name(lat, lon)
        url = f"/vsicurl/{BUCKET}/{name}/{name}.tif"
        srcs.append(rasterio.open(url))

    try:
        if len(srcs) == 1:
            src = srcs[0]
            window = from_bounds(w, s, e, n, transform=src.transform)
            data = src.read(1, window=window)
            out_transform = src.window_transform(window)
            crs = src.crs
        else:
            mosaic, out_transform = merge(srcs, bounds=(w, s, e, n))
            data = mosaic[0]
            crs = srcs[0].crs

        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1,
            "height": data.shape[0], "width": data.shape[1],
            "crs": crs, "transform": out_transform, "nodata": NODATA,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)
        return data
    finally:
        for s_ in srcs:
            s_.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only fetch the first N tiles (for a quick sanity check)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(csv.DictReader(open("data/sentinel2_benchmark/manifest.csv")))
    if args.limit:
        rows = rows[: args.limit]

    for i, row in enumerate(rows, 1):
        tile_id = row["tile_id"]
        out_path = os.path.join(OUT_DIR, f"{tile_id}_dem.tif")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{i}/{len(rows)}] {tile_id:16s} SKIP (already downloaded)")
            continue

        bbox_utm = eval(row["bbox_utm"])
        epsg = int(row["bbox_utm_epsg"])
        bbox_wgs84 = utm_bbox_to_wgs84(bbox_utm, epsg)

        t0 = time.time()
        try:
            data = fetch_tile(bbox_wgs84, out_path)
            valid = data[data != NODATA]
            if valid.size == 0:
                print(f"[{i}/{len(rows)}] {tile_id:16s} WARN: all pixels nodata ({time.time()-t0:.1f}s)")
            else:
                print(
                    f"[{i}/{len(rows)}] {tile_id:16s} OK  shape={data.shape} "
                    f"elev=[{valid.min():.0f},{valid.max():.0f}]m mean={valid.mean():.0f}m "
                    f"nodata_pct={100*(1-valid.size/data.size):.1f}% ({time.time()-t0:.1f}s)"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(rows)}] {tile_id:16s} FAIL: {exc}")

    print("\nDone. Re-run to retry any FAIL/WARN rows (existing OK files are skipped).")


if __name__ == "__main__":
    main()
