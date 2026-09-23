"""Part B remediation for the 11 tiles that failed the Part A content audit.

For 2 tiles (kakinada, ooty) content already passes -- only cloud is wrong,
which a different date can fix (land cover can't). Runs step 1 (new date,
same location) for these.

For the other 9 tiles, the failure is a real land-cover/geography mismatch
(the original candidate point sits in a town center or inland area instead
of the category's actual terrain) -- a different date cannot fix land
cover, so these go straight to step 2 (shift ~15-40km toward real terrain
matching the category, using actual geographic knowledge of each named
place), full check including ICESat-2.

No full-resolution GeoTIFFs downloaded here -- content/cloud checks reuse
the low-res methods from benchmark_content_audit.py; ICESat-2 reuses the
method from sentinel_benchmark_icesat_coverage.py.
"""
import csv
import math
import sys
import time

import httpx
import numpy as np
import rasterio
from dotenv import load_dotenv
from pyproj import Transformer

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
load_dotenv()
from backend.cdse import client as cdse  # noqa: E402
from benchmark_content_audit import (  # noqa: E402
    fetch_worldcover_pct, fetch_elevation_stats, fetch_actual_cloud_pct, OPENTOPO_URL,
)
import os

AOI_KM = 10.0


def bbox_from_point(lat, lon, km=AOI_KM):
    half_lat = (km / 2) / 111.0
    half_lon = (km / 2) / (111.0 * max(0.1, abs(math.cos(math.radians(lat)))))
    return (lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)


def find_best_scene(lat, lon, date_from, date_to, max_cloud=80.0):
    result = cdse.search_scenes(lat, lon, aoi_km=AOI_KM, date_from=date_from, date_to=date_to, max_cloud=max_cloud, limit=50)
    return sorted(result["scenes"], key=lambda s: s["cloud"])


def icesat2_check(lat, lon):
    from sliderule import icesat2, sliderule

    sliderule.init("slideruleearth.io", verbose=False)
    half_lat = (AOI_KM / 2) / 111.0
    half_lon = (AOI_KM / 2) / (111.0 * max(0.1, abs(math.cos(math.radians(lat)))))
    poly = [
        {"lon": lon - half_lon, "lat": lat - half_lat},
        {"lon": lon + half_lon, "lat": lat - half_lat},
        {"lon": lon + half_lon, "lat": lat + half_lat},
        {"lon": lon - half_lon, "lat": lat + half_lat},
        {"lon": lon - half_lon, "lat": lat - half_lat},
    ]
    parms = {
        "poly": poly, "t0": "2019-01-01T00:00:00Z", "t1": "2025-12-31T23:59:59Z",
        "srt": icesat2.SRT_LAND, "len": 100, "res": 100,
        "phoreal": {"binsize": 1.0, "geoloc": "center", "use_abs_h": False, "send_waveform": False, "above_classifier": False},
    }
    try:
        gdf = icesat2.atl08p(parms)
    except Exception as exc:  # noqa: BLE001
        return {"gnd_ph_count": 0, "segments": 0, "error": str(exc)}
    if gdf is None or len(gdf) == 0:
        return {"gnd_ph_count": 0, "segments": 0}
    land = gdf[gdf["landcover"] != 255]
    return {"gnd_ph_count": int(land["gnd_ph_count"].sum()), "segments": int(len(land))}


def check_candidate(lat, lon, date, api_key, token):
    bbox = bbox_from_point(lat, lon)
    rec = {"lat": lat, "lon": lon, "date": date}
    try:
        rec.update(fetch_worldcover_pct(bbox))
    except Exception as exc:  # noqa: BLE001
        rec["worldcover_error"] = str(exc)
    try:
        rec.update(fetch_elevation_stats(bbox, api_key))
    except Exception as exc:  # noqa: BLE001
        rec["elevation_error"] = str(exc)
    try:
        rec.update(fetch_actual_cloud_pct(bbox, date, token))
    except Exception as exc:  # noqa: BLE001
        rec["cloud_error"] = str(exc)
    return rec


MAX_NODATA_PCT = 5.0  # missed in the first remediation pass -- Sangrur slipped through with a
                       # satellite swath edge cutting through the AOI because only cloud%, not
                       # nodata%, was checked.


def passes(rec, category):
    if rec.get("actual_nodata_pct", 0) > MAX_NODATA_PCT:
        return False
    if category == "agricultural":
        return rec.get("cropland_pct", 0) >= 40 and rec.get("builtup_pct", 100) <= 15
    if category == "urban":
        return rec.get("builtup_pct", 0) >= 25
    if category == "hilly":
        return rec.get("elev_std_m", 0) >= 50
    if category == "coastal":
        return rec.get("water_wetland_mangrove_pct", 0) >= 10
    return False


# Step 1 only (same location, new date): content already passes, only cloud is wrong.
STEP1_TILES = {
    "kakinada": ("coastal", 16.9891, 82.2475),
    "ooty": ("hilly", 11.4064, 76.6932),
}

# Step 2 (shift ~15-40km toward real terrain matching the category, using
# actual geographic knowledge of each named place -- reasoning in comments).
STEP2_TILES = {
    # Original points were literal town centers; shifted into surrounding
    # rural farmland in each district.
    "bathinda": ("agricultural", 30.28, 75.15, "rural tract near Rampura Phul, ~22km E of Bathinda city"),
    "hisar": ("agricultural", 29.35, 75.90, "rural tract near Barwala, ~28km NE of Hisar city"),
    "karnal": ("agricultural", 29.85, 77.05, "rural Yamuna floodplain near Indri, ~22km N of Karnal city"),
    "kota": ("agricultural", 25.15, 76.15, "rural canal-irrigated tract near Itawa, ~28km E of Kota city"),
    "nizamabad": ("agricultural", 18.67, 77.85, "rural cotton/paddy belt near Bodhan, ~22km W of Nizamabad city"),
    "sangrur": ("agricultural", 30.38, 75.80, "rural tract near Dhuri, ~15km N of Sangrur city"),
    # Amalapuram's original point landed in tree cover, not the delta's
    # tidal/mangrove creek system -- shifted toward Coringa mangroves.
    "amalapuram": ("coastal", 16.75, 82.28, "Coringa mangrove/tidal creek system, ~30km NE of original point"),
    # Kutch's original point was well inland; shifted to the actual Gulf of
    # Kutch coastal mudflat/creek zone near Koteshwar/Narayan Sarovar.
    "kutch": ("coastal", 23.33, 68.53, "Gulf of Kutch coastal mudflats near Koteshwar, ~40km W of original point"),
    # Dehradun's original point was the urban valley floor; shifted toward
    # the forested Shivalik foothills north of the city.
    "dehradun": ("hilly", 30.47, 78.15, "forested Shivalik foothills near Rajaji NP, ~20km NE of Dehradun city"),
}


def main(out_csv):
    api_key = os.environ["OPENTOPOGRAPHY_API_KEY"]
    token = cdse.get_access_token()
    results = []

    print("=== STEP 1: same location, new date search ===", flush=True)
    for tile_id, (cat, lat, lon) in STEP1_TILES.items():
        scenes = find_best_scene(lat, lon, "2023-01-01", "2025-12-31", max_cloud=80.0)
        found = None
        rec = {"tile_id": tile_id, "category": cat, "stage": "step1_new_date", "date": None}
        for s in scenes[:15]:
            rec = check_candidate(lat, lon, s["date"], api_key, token)
            rec.update({"tile_id": tile_id, "category": cat, "stage": "step1_new_date", "scene_reported_cloud": s["cloud"]})
            if rec.get("actual_cloud_shadow_pct", 100) < 5 and passes(rec, cat):
                found = rec
                break
        chosen = found or rec
        chosen["resolved"] = bool(found)
        results.append(chosen)
        print(f"[step1] {tile_id}: resolved={chosen['resolved']} date={chosen['date']} "
              f"actual_cloud_shadow={chosen.get('actual_cloud_shadow_pct', -1):.1f}%", flush=True)

    print("\n=== STEP 2: shifted location, full check ===", flush=True)
    for tile_id, (cat, lat, lon, reasoning) in STEP2_TILES.items():
        t0 = time.time()
        scenes = find_best_scene(lat, lon, "2024-06-01", "2025-12-31", max_cloud=40.0)
        best = None
        for s in scenes[:10]:
            rec = check_candidate(lat, lon, s["date"], api_key, token)
            if rec.get("actual_cloud_shadow_pct", 100) < 5 and rec.get("actual_nodata_pct", 100) <= MAX_NODATA_PCT:
                best = rec
                break
        chosen = best or (scenes and check_candidate(lat, lon, scenes[0]["date"], api_key, token)) or {}
        chosen.update({"tile_id": tile_id, "category": cat, "stage": "step2_shifted", "shift_reasoning": reasoning})
        content_ok = passes(chosen, cat)
        chosen["content_pass"] = content_ok

        if content_ok:
            ice = icesat2_check(lat, lon)
            chosen.update({f"icesat2_{k}": v for k, v in ice.items()})
            chosen["icesat2_pass"] = ice.get("gnd_ph_count", 0) >= 5000
        else:
            chosen["icesat2_pass"] = None

        chosen["resolved"] = content_ok and chosen["icesat2_pass"] is not False
        results.append(chosen)
        print(f"[step2] {tile_id}: resolved={chosen['resolved']} content_pass={content_ok} "
              f"icesat2_pass={chosen['icesat2_pass']} ({time.time()-t0:.0f}s) -- {reasoning}", flush=True)

    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {out_csv}")


# Round 2: after visual review of round-1 replacements, 4 tiles turned out
# to have problems the automated checks missed -- Sangrur (satellite swath
# edge cutting the AOI: nodata%, which was never checked, not caught by
# cloud% alone), Chilika and Vedaranyam (real haze/thin cloud that SCL
# under-detects over water and over land respectively -- a known SCL
# weakness, not a bug in this script's math), and Shillong (elev_std passed
# numerically but the tile is centered on the city's flat built-up plateau,
# not real hill terrain -- a case the numeric threshold can't see).
# Replaced from the original wider candidate pool, by next-best ICESat-2
# rank among previously-dropped candidates for each category.
ROUND2_REPLACEMENTS = {
    "sangrur": ("agricultural", 16.3067, 80.4365, "guntur", "swap: Guntur, AP -- next-best dropped agricultural candidate by ICESat-2 rank"),
    "chilika": ("coastal", 21.7051, 72.9959, "bharuch", "swap: Bharuch/Narmada estuary, Gujarat -- next-best dropped coastal candidate"),
    "vedaranyam": ("coastal", 20.7167, 86.9167, "bhitarkanika", "swap: Bhitarkanika/Kendrapara delta, Odisha -- 2nd-next-best dropped coastal candidate"),
    "shillong": ("hilly", 31.1048, 77.1734, "shimla", "swap: Shimla, HP -- next-best dropped hilly candidate"),
}

# Round 2b: Guntur and Bharuch both failed content at their raw candidate
# coordinates (both are real towns -- same city-center-vs-rural/estuary
# issue as round 1's agricultural failures). Shifted toward real matching
# terrain, same technique as round 1.
ROUND2B_SHIFTS = {
    "guntur": ("agricultural", 16.15, 80.60, "rural Krishna-delta farmland ~25km SE of Guntur city"),
    "bharuch": ("coastal", 21.62, 72.55, "actual Narmada estuary/tidal mudflats near Dahej, ~40km SW of Bharuch town"),
}


def main_round2b(out_csv):
    api_key = os.environ["OPENTOPOGRAPHY_API_KEY"]
    token = cdse.get_access_token()
    results = []

    print("=== ROUND 2b: shift the round-2 candidates that also failed ===", flush=True)
    for tile_id, (cat, lat, lon, reasoning) in ROUND2B_SHIFTS.items():
        t0 = time.time()
        scenes = find_best_scene(lat, lon, "2024-06-01", "2025-12-31", max_cloud=30.0)
        best = None
        for s in scenes[:10]:
            rec = check_candidate(lat, lon, s["date"], api_key, token)
            if rec.get("actual_cloud_shadow_pct", 100) < 5 and rec.get("actual_nodata_pct", 100) <= MAX_NODATA_PCT:
                best = rec
                break
        chosen = best or (scenes and check_candidate(lat, lon, scenes[0]["date"], api_key, token)) or {}
        chosen.update({"tile_id": tile_id, "category": cat, "stage": "round2b_shifted", "shift_reasoning": reasoning})
        content_ok = passes(chosen, cat)
        chosen["content_pass"] = content_ok

        if content_ok:
            ice = icesat2_check(lat, lon)
            chosen.update({f"icesat2_{k}": v for k, v in ice.items()})
            chosen["icesat2_pass"] = ice.get("gnd_ph_count", 0) >= 5000
        else:
            chosen["icesat2_pass"] = None

        chosen["resolved"] = content_ok and chosen["icesat2_pass"] is not False
        results.append(chosen)
        print(f"[round2b] {tile_id}: resolved={chosen['resolved']} content_pass={content_ok} "
              f"nodata={chosen.get('actual_nodata_pct', -1):.1f}% icesat2_pass={chosen['icesat2_pass']} "
              f"({time.time()-t0:.0f}s) -- {reasoning}", flush=True)

    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {out_csv}")


def main_round2(out_csv):
    api_key = os.environ["OPENTOPOGRAPHY_API_KEY"]
    token = cdse.get_access_token()
    results = []

    print("=== ROUND 2: full replacement from wider candidate pool ===", flush=True)
    for old_tile_id, (cat, lat, lon, new_slug, reasoning) in ROUND2_REPLACEMENTS.items():
        t0 = time.time()
        scenes = find_best_scene(lat, lon, "2024-06-01", "2025-12-31", max_cloud=30.0)
        best = None
        for s in scenes[:10]:
            rec = check_candidate(lat, lon, s["date"], api_key, token)
            if rec.get("actual_cloud_shadow_pct", 100) < 5 and rec.get("actual_nodata_pct", 100) <= MAX_NODATA_PCT:
                best = rec
                break
        chosen = best or (scenes and check_candidate(lat, lon, scenes[0]["date"], api_key, token)) or {}
        chosen.update({
            "tile_id": new_slug, "replaces": old_tile_id, "category": cat,
            "stage": "round2_replacement", "shift_reasoning": reasoning,
        })
        content_ok = passes(chosen, cat)
        chosen["content_pass"] = content_ok

        if content_ok:
            ice = icesat2_check(lat, lon)
            chosen.update({f"icesat2_{k}": v for k, v in ice.items()})
            chosen["icesat2_pass"] = ice.get("gnd_ph_count", 0) >= 5000
        else:
            chosen["icesat2_pass"] = None

        chosen["resolved"] = content_ok and chosen["icesat2_pass"] is not False
        results.append(chosen)
        print(f"[round2] {old_tile_id}->{new_slug}: resolved={chosen['resolved']} content_pass={content_ok} "
              f"nodata={chosen.get('actual_nodata_pct', -1):.1f}% icesat2_pass={chosen['icesat2_pass']} "
              f"({time.time()-t0:.0f}s) -- {reasoning}", flush=True)

    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--round2":
        main_round2(sys.argv[2] if len(sys.argv) > 2 else "data/sentinel2_benchmark/remediation_round2.csv")
    elif len(sys.argv) > 1 and sys.argv[1] == "--round2b":
        main_round2b(sys.argv[2] if len(sys.argv) > 2 else "data/sentinel2_benchmark/remediation_round2b.csv")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "data/sentinel2_benchmark/remediation_results.csv")
