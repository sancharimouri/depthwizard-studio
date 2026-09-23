"""Fetch FABDEM (projects/sat-io/open-datasets/FABDEM, a bare-earth
correction of Copernicus GLO-30 that removes building/tree height --
free, CC BY-NC-SA 4.0) for all 32 data/sentinel2_benchmark/ tiles, via
Google Earth Engine.

One-time setup (this project had no prior GEE usage -- see
docs/method-audit/sentinel2/sign-flip-detector.md's FABDEM section):
    source .venv/bin/activate
    pip install earthengine-api
    python3 -c "import ee; ee.Authenticate()"   # one-time interactive browser login
Then set EARTHENGINE_PROJECT in .env to your registered GEE cloud project
id (created automatically the first time you visit
https://code.earthengine.google.com/ with a Google account, or via
`earthengine authenticate` on the CLI).

Same 32-tile loop, same idempotent skip-if-exists behavior, same output
convention as fetch_copernicus_dem_benchmark.py.

Usage:
    source .venv/bin/activate && python3 scripts/fetch_fabdem_benchmark.py [--limit N]
"""
import argparse
import csv
import os
import time

import ee
import httpx
from dotenv import load_dotenv
from pyproj import Transformer

load_dotenv()

OUT_DIR = "data/sentinel2_benchmark/fabdem_raw"
BUFFER_DEG = 0.02
FABDEM_ASSET = "projects/sat-io/open-datasets/FABDEM"


def utm_bbox_to_wgs84(bbox_utm, epsg):
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon1, lat1 = t.transform(bbox_utm[0], bbox_utm[1])
    lon2, lat2 = t.transform(bbox_utm[2], bbox_utm[3])
    return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def fetch_tile(fabdem_mosaic, bbox_wgs84, out_path):
    w, s, e, n = bbox_wgs84
    w, s, e, n = w - BUFFER_DEG, s - BUFFER_DEG, e + BUFFER_DEG, n + BUFFER_DEG
    region = ee.Geometry.Rectangle([w, s, e, n])

    url = fabdem_mosaic.getDownloadURL({
        "region": region,
        "scale": 30,
        "format": "GEO_TIFF",
        "crs": "EPSG:4326",
    })
    r = httpx.get(url, timeout=120.0)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    project = os.environ.get("EARTHENGINE_PROJECT")
    if not project:
        raise SystemExit("Set EARTHENGINE_PROJECT in .env to your registered GEE cloud project id first.")
    ee.Initialize(project=project)

    fabdem_mosaic = ee.ImageCollection(FABDEM_ASSET).mosaic()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(csv.DictReader(open("data/sentinel2_benchmark/manifest.csv")))
    if args.limit:
        rows = rows[: args.limit]

    for i, row in enumerate(rows, 1):
        tile_id = row["tile_id"]
        out_path = os.path.join(OUT_DIR, f"{tile_id}_fabdem.tif")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[{i}/{len(rows)}] {tile_id:16s} SKIP (already downloaded)")
            continue

        bbox_utm = eval(row["bbox_utm"])
        epsg = int(row["bbox_utm_epsg"])
        bbox_wgs84 = utm_bbox_to_wgs84(bbox_utm, epsg)

        t0 = time.time()
        try:
            fetch_tile(fabdem_mosaic, bbox_wgs84, out_path)
            print(f"[{i}/{len(rows)}] {tile_id:16s} OK  ({time.time()-t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(rows)}] {tile_id:16s} FAIL: {exc}")

    print("\nDone. Re-run to retry any FAIL rows (existing OK files are skipped).")


if __name__ == "__main__":
    main()
