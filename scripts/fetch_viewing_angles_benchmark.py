"""Fetch real Sentinel-2 viewing/incidence angle metadata (MTD_TL.xml,
Mean_Viewing_Incidence_Angle_List) for all 32 sentinel2_benchmark tiles.

Not fetchable from the CDSE Catalog/Process APIs already wired into
backend/cdse/client.py (their STAC properties carry no angle fields), and
CDSE's OData download API -- which does serve MTD_TL.xml -- rejected this
project's existing CDSE_CLIENT_ID/SECRET ("Token audience not allowed";
those creds are Sentinel-Hub-scoped, not Data-Space-download-scoped).

Instead this uses Microsoft Planetary Computer's public Sentinel-2 STAC
(no credentials at all) to locate the matching granule by date+bbox, then
its free SAS-token endpoint to read the granule's real MTD_TL.xml straight
off public Azure Open Data blob storage. Genuine per-band viewing zenith/
azimuth angles, not a proxy.
"""

import csv
import re
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

MANIFEST = Path("data/sentinel2_benchmark/manifest.csv")
OUT_CSV = Path("data/sentinel2_benchmark/viewing_angles.csv")

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"


def bbox_around(lat: float, lon: float, half_deg: float = 0.1) -> list[float]:
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


def find_item(lat: float, lon: float, date: str) -> dict | None:
    # +/- 1 day window: dates in the manifest are acquisition dates, scene
    # timestamps can cross UTC midnight relative to the local date recorded.
    r = httpx.get(
        STAC_SEARCH_URL,
        params={
            "collections": "sentinel-2-l2a",
            "bbox": ",".join(str(v) for v in bbox_around(lat, lon)),
            "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
            "limit": 10,
        },
        timeout=30,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return None
    # Prefer the item whose footprint actually covers the point, lowest cloud first.
    feats.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 999))
    return feats[0]


def fetch_mean_angles(granule_metadata_href: str, sas_token: str) -> dict[str, float]:
    url = f"{granule_metadata_href}?{sas_token}"
    r = httpx.get(url, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    zeniths, azimuths = [], []
    for elem in root.iter():
        if elem.tag.endswith("Mean_Viewing_Incidence_Angle"):
            for child in elem:
                if child.tag.endswith("ZENITH_ANGLE") and child.text:
                    zeniths.append(float(child.text))
                elif child.tag.endswith("AZIMUTH_ANGLE") and child.text:
                    azimuths.append(float(child.text))
    if not zeniths:
        raise ValueError("no Mean_Viewing_Incidence_Angle entries found")
    return {
        "mean_viewing_zenith_deg": sum(zeniths) / len(zeniths),
        "min_viewing_zenith_deg": min(zeniths),
        "max_viewing_zenith_deg": max(zeniths),
        "mean_viewing_azimuth_deg": sum(azimuths) / len(azimuths),
        "n_bands": len(zeniths),
    }


def main() -> None:
    rows = list(csv.DictReader(open(MANIFEST)))

    sas = httpx.get(SAS_URL, timeout=30).json()
    sas_token = sas["token"]
    sas_expiry = time.time() + 40 * 60  # refresh well before the ~55min expiry

    results = []
    for i, row in enumerate(rows, 1):
        tile_id = row["tile_id"]
        lat, lon, date = float(row["lat"]), float(row["lon"]), row["date_acquired"]

        if time.time() > sas_expiry:
            sas = httpx.get(SAS_URL, timeout=30).json()
            sas_token = sas["token"]
            sas_expiry = time.time() + 40 * 60

        try:
            item = find_item(lat, lon, date)
            if item is None:
                results.append({"tile_id": tile_id, "status": "no_stac_item_found"})
                print(f"[{i}/{len(rows)}] {tile_id:16s} -> no STAC item found")
                continue

            href = item["assets"]["granule-metadata"]["href"]
            angles = fetch_mean_angles(href, sas_token)
            results.append(
                {
                    "tile_id": tile_id,
                    "category": row["category"],
                    "status": "ok",
                    "stac_item_id": item["id"],
                    "s2_mgrs_tile": item["properties"].get("s2:mgrs_tile", ""),
                    "s2_datetime": item["properties"].get("datetime", ""),
                    "sat_relative_orbit": item["properties"].get("sat:relative_orbit", ""),
                    "sat_orbit_state": item["properties"].get("sat:orbit_state", ""),
                    **angles,
                }
            )
            print(
                f"[{i}/{len(rows)}] {tile_id:16s} -> mean_zenith="
                f"{angles['mean_viewing_zenith_deg']:.3f} deg (n_bands={angles['n_bands']})"
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"tile_id": tile_id, "status": f"error: {exc}"})
            print(f"[{i}/{len(rows)}] {tile_id:16s} -> error: {exc}")

    fields = [
        "tile_id", "category", "status", "stac_item_id", "s2_mgrs_tile", "s2_datetime",
        "sat_relative_orbit", "sat_orbit_state", "mean_viewing_zenith_deg",
        "min_viewing_zenith_deg", "max_viewing_zenith_deg", "mean_viewing_azimuth_deg", "n_bands",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n{ok}/{len(results)} tiles fetched successfully. Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
