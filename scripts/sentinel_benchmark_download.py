"""Pull real, georeferenced 10x10km Sentinel-2 RGB GeoTIFFs for selected
benchmark tiles via CDSE, reusing the existing OAuth2 + Catalog + Process API
integration in backend/cdse/client.py (auth, catalog search, true-color
evalscript) rather than reimplementing it.

What's genuinely new here (the existing client only ever requested 512x512
preview PNGs for the frontend's Scene Input modal, not analytic tiles):
  - a clean 10x10km bbox in the tile's local UTM zone (not WGS84 degrees),
    matching the production regions' georeferencing style
  - requesting image/tiff output (not image/png) so the result is a real
    GeoTIFF with embedded CRS/geotransform, at 1000x1000 px / 10m resolution
"""

import csv
import math
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from backend.cdse import client as cdse  # noqa: E402

OUT_ROOT = Path("data/sentinel2_benchmark")


def utm_epsg(lat: float, lon: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def utm_bbox(lat: float, lon: float, km: float) -> tuple[list[float], int]:
    epsg = utm_epsg(lat, lon)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    half = (km * 1000) / 2
    return [cx - half, cy - half, cx + half, cy + half], epsg


def find_scene(lat: float, lon: float, max_cloud: float = 20.0) -> dict | None:
    """Search the last ~18 months for the lowest-cloud recent scene, widening
    the cloud threshold once if nothing clears the strict bar."""
    date_to = "2025-12-31"
    date_from = "2024-06-01"
    for cloud_cap in (max_cloud, 40.0, 80.0):
        result = cdse.search_scenes(lat, lon, aoi_km=10.0, date_from=date_from, date_to=date_to, max_cloud=cloud_cap, limit=20)
        scenes = sorted(result["scenes"], key=lambda s: s["cloud"])
        if scenes:
            return scenes[0]
    return None


def fetch_analytic_tiff(bbox_utm: list[float], epsg: int, date: str, width: int = 1000, height: int = 1000) -> bytes:
    token = cdse.get_access_token()
    body = {
        "input": {
            "bounds": {
                "bbox": bbox_utm,
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"},
            },
            "data": [
                {
                    "type": cdse.COLLECTION,
                    "dataFilter": {
                        "timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"},
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": cdse.TRUE_COLOR_EVALSCRIPT,
    }
    for attempt in range(3):
        response = httpx.post(
            cdse.PROCESS_URL,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=120.0,
        )
        if response.status_code == 429 and attempt < 2:
            time.sleep(10 * (attempt + 1))
            continue
        if response.status_code >= 400:
            raise cdse.CDSEUpstreamError(f"Process API failed ({response.status_code}): {response.text}")
        return response.content
    raise cdse.CDSEUpstreamError("Process API rate-limited after retries")


def download_tile(slug: str, label: str, category: str, lat: float, lon: float) -> dict:
    scene = find_scene(lat, lon)
    if scene is None:
        return {"slug": slug, "label": label, "category": category, "status": "no_scene_found"}

    bbox_utm, epsg = utm_bbox(lat, lon, 10.0)
    try:
        tiff_bytes = fetch_analytic_tiff(bbox_utm, epsg, scene["date"])
    except Exception as exc:  # noqa: BLE001
        return {"slug": slug, "label": label, "category": category, "status": f"error: {exc}"}

    out_dir = OUT_ROOT / category / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}_RGB.tif"
    out_path.write_bytes(tiff_bytes)

    return {
        "slug": slug,
        "label": label,
        "category": category,
        "status": "ok",
        "date": scene["date"],
        "cloud": scene["cloud"],
        "epsg": epsg,
        "bbox_utm": bbox_utm,
        "path": str(out_path),
    }


def main(selections_csv: str) -> None:
    rows = []
    with open(selections_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    results = []
    for i, row in enumerate(rows, 1):
        t0 = time.time()
        result = download_tile(row["slug"], row["label"], row["category"], float(row["lat"]), float(row["lon"]))
        elapsed = time.time() - t0
        print(f"[{i}/{len(rows)}] {row['category']:14s} {row['slug']:16s} -> {result['status']} ({elapsed:.1f}s)", flush=True)
        results.append(result)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} tiles downloaded successfully.")

    results_path = OUT_ROOT / "download_results.csv"
    fields = ["slug", "label", "category", "status", "date", "cloud", "epsg", "bbox_utm", "path"]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {len(results)} results to {results_path}")


if __name__ == "__main__":
    main(sys.argv[1])
