"""Part A: objective content audit for all tiles in the India Sentinel-2
benchmark. For each tile's exact footprint:
  - ESA WorldCover 10m (public AWS COG, windowed read, no download of full tiles)
  - SRTM 30m elevation (OpenTopography Global DEM API, clipped to bbox)
  - Actual cloud+shadow fraction from the tile's own Sentinel-2 SCL band
    (CDSE Process API, low-res 256x256 preview, not full-res)

Writes one row per tile to a CSV. Read-only against the existing manifest;
does not modify it.
"""
import csv
import math
import os
import sys
import time

import httpx
import numpy as np
import rasterio
from dotenv import load_dotenv
from pyproj import Transformer

sys.path.insert(0, ".")
load_dotenv()
from backend.cdse import client as cdse  # noqa: E402

WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

SCL_EVALSCRIPT = """
//VERSION=3
function setup() {
    return { input: [{ bands: ["SCL"] }], output: { bands: 1, sampleType: "UINT8" } };
}
function evaluatePixel(sample) {
    return [sample.SCL];
}
"""


def utm_bbox_to_wgs84(bbox_utm, epsg):
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon1, lat1 = t.transform(bbox_utm[0], bbox_utm[1])
    lon2, lat2 = t.transform(bbox_utm[2], bbox_utm[3])
    return (min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))


def worldcover_tile_name(lat_c, lon_c):
    tile_lat = math.floor(lat_c / 3) * 3
    tile_lon = math.floor(lon_c / 3) * 3
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    return f"{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}"


def fetch_worldcover_pct(bbox):
    lon_c, lat_c = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    tile = worldcover_tile_name(lat_c, lon_c)
    url = f"{WORLDCOVER_BASE}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    with rasterio.open(url) as src:
        from rasterio.windows import from_bounds

        win = from_bounds(*bbox, transform=src.transform)
        data = src.read(1, window=win)
    total = data.size
    if total == 0:
        return None
    vals, counts = np.unique(data, return_counts=True)
    pct = dict(zip(vals.tolist(), (100 * counts / total).tolist()))
    return {
        "tree_pct": pct.get(10, 0.0),
        "cropland_pct": pct.get(40, 0.0),
        "builtup_pct": pct.get(50, 0.0),
        "water_wetland_mangrove_pct": pct.get(80, 0.0) + pct.get(90, 0.0) + pct.get(95, 0.0),
        "worldcover_tile": tile,
    }


def fetch_elevation_stats(bbox, api_key):
    params = {
        "demtype": "SRTMGL1",
        "south": bbox[1], "north": bbox[3], "west": bbox[0], "east": bbox[2],
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    r = httpx.get(OPENTOPO_URL, params=params, timeout=30.0)
    r.raise_for_status()
    with rasterio.MemoryFile(r.content) as memfile:
        with memfile.open() as src:
            arr = src.read(1).astype(np.float64)
    return {
        "elev_min_m": float(arr.min()),
        "elev_max_m": float(arr.max()),
        "elev_std_m": float(arr.std()),
        "elev_range_m": float(arr.max() - arr.min()),
    }


def fetch_actual_cloud_pct(bbox_wgs84, date, token):
    body = {
        "input": {
            "bounds": {"bbox": list(bbox_wgs84), "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": cdse.COLLECTION, "dataFilter": {"timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"}}}],
        },
        "output": {"width": 256, "height": 256, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": SCL_EVALSCRIPT,
    }
    r = httpx.post(cdse.PROCESS_URL, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=60.0)
    r.raise_for_status()
    with rasterio.MemoryFile(r.content) as memfile:
        with memfile.open() as src:
            scl = src.read(1)
    total = scl.size
    vals, counts = np.unique(scl, return_counts=True)
    cd = dict(zip(vals.tolist(), counts.tolist()))
    cloud_only = sum(cd.get(v, 0) for v in (8, 9, 10))
    cloud_shadow = sum(cd.get(v, 0) for v in (3, 8, 9, 10))
    nodata = cd.get(0, 0)
    return {
        "actual_cloud_pct": 100 * cloud_only / total,
        "actual_cloud_shadow_pct": 100 * cloud_shadow / total,
        "actual_nodata_pct": 100 * nodata / total,
    }


def main(out_csv):
    rows = list(csv.DictReader(open("data/sentinel2_benchmark/manifest.csv")))
    token = cdse.get_access_token()
    api_key = os.environ["OPENTOPOGRAPHY_API_KEY"]

    results = []
    for i, row in enumerate(rows, 1):
        tile_id = row["tile_id"]
        bbox_utm = eval(row["bbox_utm"])
        epsg = int(row["bbox_utm_epsg"])
        bbox_wgs84 = utm_bbox_to_wgs84(bbox_utm, epsg)

        t0 = time.time()
        rec = {"tile_id": tile_id, "category": row["category"], "label": row["label"],
               "date_acquired": row["date_acquired"], "reported_cloud_pct": row["cloud_pct"]}
        try:
            rec.update(fetch_worldcover_pct(bbox_wgs84))
        except Exception as exc:  # noqa: BLE001
            rec["worldcover_error"] = str(exc)
        try:
            rec.update(fetch_elevation_stats(bbox_wgs84, api_key))
        except Exception as exc:  # noqa: BLE001
            rec["elevation_error"] = str(exc)
        try:
            rec.update(fetch_actual_cloud_pct(bbox_wgs84, row["date_acquired"], token))
        except Exception as exc:  # noqa: BLE001
            rec["cloud_error"] = str(exc)

        results.append(rec)
        print(f"[{i}/{len(rows)}] {tile_id:16s} ({time.time()-t0:.1f}s) "
              f"crop={rec.get('cropland_pct', -1):.1f}% built={rec.get('builtup_pct', -1):.1f}% "
              f"tree={rec.get('tree_pct', -1):.1f}% water={rec.get('water_wetland_mangrove_pct', -1):.1f}% "
              f"elev_std={rec.get('elev_std_m', -1):.1f}m cloud={rec.get('actual_cloud_pct', -1):.2f}%",
              flush=True)

    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sentinel2_benchmark/content_audit.csv")
