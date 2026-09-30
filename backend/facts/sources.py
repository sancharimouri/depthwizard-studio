"""Facts box + Scenario Analysis cards for a location (docs/facts-research.md, v2 Parts B and D).

Input is a bounding box (georeferenced input: the tile's footprint) or a point (non-georeferenced input with a
user-entered lat/lon: point lookups only). Output is short UI lines (backend/facts/lines.py), grouped:
  facts      ThinkHazard wildfire / cyclone / tsunami / volcano (bundled grid) + Wikidata named features (live)
  scenario   flood: JRC/GloFAS 1-in-100-yr extent + JRC Global Surface Water (live windowed COG reads) + district
             level; landslide: World Bank 1 km class (live window) + district level; earthquake: district level +
             USGS M4.5+ within 100 km (live) + the epicentres for the inset map
Graceful by construction: every source runs in its own thread with a short timeout; a failed, slow or empty source
contributes no lines and is recorded in "status" as a short code (never a raw error, never shown in the UI).
Responses are cached per location in a bounded LRU. Sources, licences and attribution: the Docs page, "Data sources
& credits" (the UI panels carry none). Nominatim, Open-Meteo and GDACS were dropped on 2026-09-30.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import httpx

from backend.facts import lines as L
from backend.facts import thinkhazard

USER_AGENT = "DepthWizard2/0.2 (SIH26175 hackathon prototype; https://github.com/sancharimouri/depthwizard-studio)"
HTTP_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
DEADLINE_S = 9.0  # the whole request; late sources are dropped
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
WIKIDATA = "https://query.wikidata.org/sparql"
JRC_RP100 = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP100/ID{id}_{name}_RP100_depth.tif"
WB_LANDSLIDE = "https://datacatalogfiles.worldbank.org/ddh-published/0037584/DR0045417/LS_TH_COG.tif"
GSW = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{t}v1_4_2021.tif"
GDAL_ENV = {"GDAL_HTTP_TIMEOUT": "5", "GDAL_HTTP_CONNECTTIMEOUT": "3", "GDAL_HTTP_MAX_RETRY": "0",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
            "GDAL_CACHEMAX": 32, "VSI_CACHE": False}
_JRC_TILES = json.loads((Path(__file__).resolve().parent / "data/jrc_flood_tiles.json").read_text())["tiles"]

CACHE_MAX = 128
CACHE_OK_S = 24 * 3600
CACHE_PARTIAL_S = 600
_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()

PEAK_CLASSES = {"Q8502", "Q54050", "Q207326", "Q8072"}  # mountain, hill, summit, volcano
RIVER, GLACIER = "Q4022", "Q35666"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ------------------------------------------------------------------ live sources (each returns a value or None)
def usgs(client, lat, lon, bbox=None, radius=L.EQ_RADIUS_KM) -> dict:
    base = {"format": "geojson", "latitude": lat, "longitude": lon, "maxradiuskm": radius, "eventtype": "earthquake",
            "minmagnitude": 4.5}
    feats = client.get(USGS, params={**base, "starttime": "1973-01-01", "orderby": "magnitude", "limit": 500}).json()["features"]
    top = client.get(USGS, params={**base, "starttime": "1900-01-01", "orderby": "magnitude", "limit": 1}).json()["features"]
    events = []
    for f in feats:
        lo, la = f["geometry"]["coordinates"][:2]
        events.append([round(lo, 3), round(la, 3), f["properties"]["mag"], time.gmtime(f["properties"]["time"] / 1000).tm_year])
    largest = None
    if top:
        lo, la = top[0]["geometry"]["coordinates"][:2]
        p = top[0]["properties"]
        largest = {"mag": p["mag"], "year": time.gmtime(p["time"] / 1000).tm_year,
                   "distance_km": round(haversine_km(lat, lon, la, lo), 1)}
    return {"radius_km": radius, "count_since_1973": len(events), "largest": largest, "events": events,
            "centre": [round(lat, 5), round(lon, 5)], "bbox": [round(v, 5) for v in bbox] if bbox else None}


def wikidata_features(client, lat, lon, radius_km=15) -> dict:
    q = f"""SELECT ?item ?itemLabel ?cls ?dist ?elev ?len WHERE {{
      SERVICE wikibase:around {{ ?item wdt:P625 ?loc . bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
        bd:serviceParam wikibase:radius "{radius_km}" . bd:serviceParam wikibase:distance ?dist . }}
      ?item wdt:P31 ?cls . VALUES ?cls {{ wd:Q8502 wd:Q54050 wd:Q207326 wd:Q8072 wd:{RIVER} wd:{GLACIER} }}
      OPTIONAL {{ ?item wdt:P2044 ?elev }} OPTIONAL {{ ?item wdt:P2043 ?len }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} ORDER BY ?dist LIMIT 100"""
    rows = client.get(WIKIDATA, params={"query": q, "format": "json"}).json()["results"]["bindings"]
    items = []
    for x in rows:
        name = x["itemLabel"]["value"]
        if re.fullmatch(r"Q\d+", name):  # no English label: not a *named* feature
            continue
        num = lambda k: float(x[k]["value"]) if k in x else None  # noqa: E731
        items.append({"name": name, "cls": x["cls"]["value"].rsplit("/", 1)[1], "distance_km": float(x["dist"]["value"]),
                      "elevation_m": num("elev"), "length_km": num("len")})
    out = {}
    peaks = [i for i in items if i["cls"] in PEAK_CLASSES]
    with_h = [i for i in peaks if i["elevation_m"] is not None]
    if with_h or peaks:
        p = min(with_h or peaks, key=lambda i: i["distance_km"])
        out["peak"] = {k: p[k] for k in ("name", "elevation_m", "distance_km")}
    rivers = [i for i in items if i["cls"] == RIVER and i["distance_km"] <= 10]
    if rivers:
        longest = [i for i in rivers if i["length_km"]]
        r = max(longest, key=lambda i: i["length_km"]) if longest else min(rivers, key=lambda i: i["distance_km"])
        out["river"] = {"name": r["name"], "main": bool(longest), "length_km": r["length_km"], "distance_km": r["distance_km"]}
    glaciers = [i for i in items if i["cls"] == GLACIER]
    if glaciers:
        g = min(glaciers, key=lambda i: i["distance_km"])
        out["glacier"] = {"name": g["name"], "distance_km": g["distance_km"]}
    return out


def _window(url, bbox=None, point=None):
    import numpy as np
    import rasterio
    from rasterio.windows import Window, from_bounds

    with rasterio.Env(**GDAL_ENV), rasterio.open("/vsicurl/" + url) as s:
        if point:
            r, c = s.index(point[1], point[0])
            win = Window(c, r, 1, 1)
        else:
            win = from_bounds(*bbox, s.transform)
        a = s.read(1, window=win, boundless=True, fill_value=s.nodata if s.nodata is not None else 0)
        return a.astype(np.float64), s.nodata


def _tile10(lat, lon):
    return int(math.floor(lon / 10) * 10), int(math.ceil(lat / 10) * 10)


def jrc_flood(lat, lon, bbox=None) -> dict | None:
    x, y = _tile10(lat, lon)
    name = f"{'N' if y >= 0 else 'S'}{abs(y)}_{'E' if x >= 0 else 'W'}{abs(x)}"
    if name not in _JRC_TILES:  # ocean / no land tile: nothing modelled
        return None
    a, nd = _window(JRC_RP100.format(id=_JRC_TILES[name], name=name), bbox, None if bbox else (lat, lon))
    wet = (a > 0) & (a != nd)
    if not bbox:
        return {"inside": bool(wet.any())}
    return {"pct": round(float(wet.mean() * 100), 1), "max_depth_m": round(float(a[wet].max()), 1) if wet.any() else 0.0}


def wb_landslide(lat, lon, bbox=None) -> dict | None:
    a, _ = _window(WB_LANDSLIDE, bbox, None if bbox else (lat, lon))
    v = a[(a >= 1) & (a <= 4)]
    if not v.size:
        return None
    if not bbox:
        return {"class": int(v[0])}
    top = int(v.max())
    return {"min": int(v.min()), "max": top, "share_max_pct": round(float((v == top).mean() * 100), 1)}


def gsw(lat, lon, bbox) -> dict | None:
    x, y = _tile10(lat, lon)
    t = f"{abs(x)}{'E' if x >= 0 else 'W'}_{abs(y)}{'N' if y >= 0 else 'S'}"
    a, _ = _window(GSW.format(t=t), bbox)
    ok = a <= 100
    if not ok.any():
        return None
    return {"pct_ever": round(float(((a > 0) & ok).mean() * 100), 1),
            "pct_permanent": round(float(((a >= 90) & ok).mean() * 100), 1)}


# ------------------------------------------------------------------ orchestration
def _key(bbox, lat, lon):
    return ("bbox", *(round(v, 3) for v in bbox)) if bbox else ("point", round(lat, 3), round(lon, 3))


def info(bbox: list[float] | None = None, lat: float | None = None, lon: float | None = None) -> dict:
    if bbox:
        w, s, e, n = bbox
        lat, lon = (s + n) / 2, (w + e) / 2
    key = _key(bbox, lat, lon)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now < hit[0]:
            _cache.move_to_end(key)
            return hit[1]
    status: dict[str, str] = {}
    out_lines: list[dict] = []
    district = thinkhazard.lookup(lat, lon)
    status["thinkhazard"] = "ok" if district else "empty"
    if district:
        out_lines += L.thinkhazard_lines(district["levels"])
    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                      follow_redirects=True) as client:
        jobs = {"wikidata": lambda: L.feature_lines(wikidata_features(client, lat, lon)),
                "usgs": lambda: L.usgs_lines(usgs(client, lat, lon, bbox)),
                "jrc_flood": lambda: L.jrc_flood_lines(jrc_flood(lat, lon, bbox)),
                "wb_landslide": lambda: L.wb_landslide_lines(wb_landslide(lat, lon, bbox))}
        if bbox:  # surface-water share is a tile statistic; a point has none
            jobs["gsw"] = lambda: L.gsw_lines(gsw(lat, lon, bbox))
        pool = ThreadPoolExecutor(max_workers=len(jobs))
        futs = {pool.submit(fn): name for name, fn in jobs.items()}
        done, pending = wait(futs, timeout=DEADLINE_S)
        for f in pending:
            status[futs[f]] = "timeout"
            f.cancel()
        for f in done:
            name = futs[f]
            try:
                got = f.result()
                status[name] = "ok" if got else "empty"
                out_lines += got
            except httpx.TimeoutException:
                status[name] = "timeout"
            except Exception:  # noqa: BLE001 — any failure: that source adds nothing
                status[name] = "failed"
        pool.shutdown(wait=False, cancel_futures=True)
    out = {"where": {"mode": "bbox" if bbox else "point", "bbox": bbox, "centre": [round(lat, 5), round(lon, 5)],
                     "district": {k: district[k] for k in ("name", "admin1", "admin0")} if district else None},
           "queried_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **L.group_lines(out_lines),
           "status": status}
    ttl = CACHE_OK_S if all(v in ("ok", "empty") for v in status.values()) else CACHE_PARTIAL_S
    with _cache_lock:
        _cache[key] = (now + ttl, out)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)
    return out
