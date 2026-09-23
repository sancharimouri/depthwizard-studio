"""Fetch real SRTMGL1 (30m) elevation GeoTIFFs for all 32
data/sentinel2_benchmark/ tiles, via the OpenTopography Global DEM API.

Same API/params as scripts/benchmark_content_audit.py's fetch_elevation_stats
(reused unmodified), but saves the raw GeoTIFF per tile instead of just
summary stats, for use as a pixel-level ground-truth signal in the
sign-flip detector (docs/method-audit/sentinel2/sign-flip-detector.md).

Note: this project has previously hit OpenTopography's 50-calls/24hr rate
limit on this same API key (see docs/method-audit/sentinel2/
backbone-terrain-analysis.md, Follow-up 2) -- if this script starts
failing partway through with 429s, that's why. scripts/
fetch_copernicus_dem_benchmark.py (Copernicus GLO-30 via the public AWS
Open Data bucket, no key, no rate limit) is the fallback used for that
reason and covers the same use case.

Idempotent: skips any tile whose output file already exists, so it's safe
to re-run after a partial failure (rate limit, timeout, etc).

Usage:
    source .venv/bin/activate && python3 scripts/fetch_srtm_benchmark.py
"""
import csv
import os
import sys
import time

import httpx
from dotenv import load_dotenv
from pyproj import Transformer

load_dotenv()

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
OUT_DIR = "data/sentinel2_benchmark/srtm_raw"
BUFFER_DEG = 0.02  # ~2km padding so reprojection onto the tile's UTM grid has no edge gaps


def utm_bbox_to_wgs84(bbox_utm, epsg):
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon1, lat1 = t.transform(bbox_utm[0], bbox_utm[1])
    lon2, lat2 = t.transform(bbox_utm[2], bbox_utm[3])
    return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    api_key = os.environ["OPENTOPOGRAPHY_API_KEY"]
    rows = list(csv.DictReader(open("data/sentinel2_benchmark/manifest.csv")))

    for i, row in enumerate(rows, 1):
        tile_id = row["tile_id"]
        out_path = os.path.join(OUT_DIR, f"{tile_id}_srtm.tif")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{i}/{len(rows)}] {tile_id:16s} SKIP (already downloaded)")
            continue

        bbox_utm = eval(row["bbox_utm"])
        epsg = int(row["bbox_utm_epsg"])
        w, s, e, n = utm_bbox_to_wgs84(bbox_utm, epsg)
        params = {
            "demtype": "SRTMGL1",
            "south": s - BUFFER_DEG, "north": n + BUFFER_DEG,
            "west": w - BUFFER_DEG, "east": e + BUFFER_DEG,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }
        t0 = time.time()
        try:
            r = httpx.get(OPENTOPO_URL, params=params, timeout=60.0)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
            print(f"[{i}/{len(rows)}] {tile_id:16s} OK  ({len(r.content)/1024:.0f} KB, {time.time()-t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(rows)}] {tile_id:16s} FAIL: {exc}", file=sys.stderr)
        time.sleep(1.0)  # be polite to the free API tier

    print("\nDone. Re-run this script to retry any FAIL rows (it skips completed ones).")


if __name__ == "__main__":
    main()
