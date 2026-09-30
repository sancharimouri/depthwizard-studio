"""Draft geographic facts for every included library tile (docs/facts-research.md, Part E1).

Writes data/library_v2_<date>/facts_drafts.json + facts_drafts.md. Every fact is a retrieved or derived value, status
"draft"; nothing here is shown in the app until a fact is marked "curated" by hand and the static library is re-baked
(scripts/bake_static_library.py --facts).

Reads (read-only): data/library_v2_2026-09-29/tile_manifest.json and dem/<tile>.tif (band TERRAIN = FABDEM bare earth).
Queries, all free and key-less (checked 2026-09-30):
  USGS FDSN event service, ThinkHazard! (GFDRR; district found via one Nominatim reverse call per tile, not shown),
  World Bank Global Landslide Hazard Map (COG, window read), JRC/CEMS GloFAS river flood hazard RP100 (window read),
  JRC Global Surface Water occurrence (window read), Wikidata SPARQL (named natural features).
Bundled files (downloaded once to --cache): IBTrACS v04r01 since1980 CSV, NOAA SPC tornadoes 1950-2024 CSV.

Scope rules:
  Sentinel-2 / Maxar: tile facts from the pack's footprint (bbox); hazard history within 100 km of the tile centre.
  DFC2019: the tile locations are private, so ONLY city-level facts (Jacksonville FL / Omaha NE city centre, and the
  county via ThinkHazard). No DEM-derived facts: the pack surface is ground + Method 6 predicted heights, and slope on
  it would measure building walls, not terrain.

  python scripts/draft_library_facts.py --cache <dir with ibtracs.since1980.csv, spc_tornadoes.csv>
"""

from __future__ import annotations

import argparse
import copy
import csv
import difflib
import json
import math
import re
import time
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data/library_v2_2026-09-29"
TODAY = date.today().isoformat()
UA = "DepthWizard2-research/0.1 (SIH26175 prototype; one-off facts draft; https://github.com/sancharimouri/depthwizard-studio)"

WB_TH = "https://datacatalogfiles.worldbank.org/ddh-published/0037584/DR0045417/LS_TH_COG.tif"
JRC_RP100 = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
GSW = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{t}v1_4_2021.tif"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1"

CITIES = {"JAX": ("Jacksonville, Florida", 30.3322, -81.6557, "Duval", "Florida"),
          "OMA": ("Omaha, Nebraska", 41.2565, -95.9345, "Douglas", "Nebraska")}

SRC = {
    "fabdem": ("FABDEM v1-2 bare-earth DEM (30 m), via this tile's elevation pack",
               "https://doi.org/10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn",
               "CC BY-NC-SA 4.0 (non-commercial); attribution: Hawker et al. (2022), FABDEM, University of Bristol / Fathom"),
    "usgs": ("USGS ANSS Comprehensive Earthquake Catalog (ComCat)", f"{USGS}/query",
             "Public domain (U.S. Government); attribution: U.S. Geological Survey"),
    "thinkhazard": ("ThinkHazard! (GFDRR, World Bank)", "https://thinkhazard.org",
                    "Hazard levels CC BY 4.0; attribution: ThinkHazard! – GFDRR (thinkhazard.org)"),
    "wb_ls": ("Global Landslide Hazard Map, total hazard (World Bank / ARUP, 2021; ~1 km)", WB_TH,
              "World Bank Data Catalog dataset 0037584; licence text not machine-readable today — verify before publishing"),
    "jrc_flood": ("Copernicus EMS / GloFAS global river flood hazard map, 1-in-100-year (v2.1.2, ~90 m)", JRC_RP100,
                  "CC BY 4.0; attribution: © European Union, Copernicus Emergency Management Service (GloFAS)"),
    "gsw": ("JRC Global Surface Water occurrence 1984–2021 (v1.4, 30 m)", "https://global-surface-water.appspot.com",
            "Free of charge, no restriction of use (Copernicus); attribution: EC JRC/Google, Pekel et al. (2016) Nature 540"),
    "ibtracs": ("NOAA NCEI IBTrACS v04r01 (tropical cyclone best tracks since 1980)",
                "https://www.ncei.noaa.gov/products/international-best-track-archive",
                "Public domain (U.S. Government); attribution: Knapp et al. (2010), IBTrACS, NOAA NCEI"),
    "spc": ("NOAA Storm Prediction Center tornado database 1950–2024", "https://www.spc.noaa.gov/wcm/",
            "Public domain (U.S. Government); attribution: NOAA/NWS Storm Prediction Center"),
    "wikidata": ("Wikidata (named natural features, coordinates P625)", "https://query.wikidata.org",
                 "CC0 1.0; attribution: Wikidata (not required)"),
}
TH_NAMES = {"FL": "river flood", "UF": "urban flood", "CF": "coastal flood", "EQ": "earthquake", "LS": "landslide",
            "TS": "tsunami", "VA": "volcano", "CY": "cyclone", "DG": "water scarcity", "EH": "extreme heat", "WF": "wildfire"}
TH_LEVEL = {"HIG": "high", "MED": "medium", "LOW": "low", "VLO": "very low"}
# geographic hazards only (no water scarcity / extreme heat: climate, not terrain)
TH_KEEP = ("EQ", "LS", "FL", "CF", "TS", "CY", "VA", "WF")
WD_CLASSES = {"Q8502": "mountain", "Q54050": "hill", "Q207326": "summit", "Q4022": "river", "Q47521": "stream",
              "Q23397": "lake", "Q131681": "reservoir", "Q35666": "glacier", "Q8072": "volcano", "Q34038": "waterfall",
              "Q40080": "beach", "Q23442": "island", "Q47053": "estuary", "Q170321": "wetland", "Q39594": "bay",
              "Q187223": "lagoon", "Q133056": "mountain pass", "Q39816": "valley"}


# which facts a tile keeps when more than 6 are available (most tile-specific and hazard-relevant first)
PRIORITY = ["relief", "slope", "lowlying", "thinkhazard", "wb_ls", "jrc_flood", "usgs", "ibtracs", "spc", "gsw",
            "wikidata", "tri"]


def hav_km(la1, lo1, la2, lo2):
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def fact(key, text, value, origin, confidence, scope, kind=None):
    name, url, lic = SRC[key]
    return {"kind": kind or key, "text": text, "source": name, "source_url": url, "retrieved": TODAY, "licence": lic, "value": value,
            "origin": origin, "confidence": confidence, "scope": scope, "status": "draft"}


# ------------------------------------------------------------------ C1: derived from the tile's own DEM
def dem_facts(tif: Path):
    with rasterio.open(tif) as s:
        z = s.read(1).astype(np.float64)
        if s.nodata is not None:
            z[z == s.nodata] = np.nan
        rx, ry = abs(s.res[0]), abs(s.res[1])
        bbox = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    # the Maxar TERRAIN band is FABDEM bicubic onto 1.2 m: aggregate to ~30 m so slope/TRI mean the same as Sentinel-2's
    f = max(1, round(30 / rx))
    if f > 1:
        h, w = (z.shape[0] // f) * f, (z.shape[1] // f) * f
        z = np.nanmean(z[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))
        rx, ry = rx * f, ry * f
    v = z[np.isfinite(z)]
    gy, gx = np.gradient(z, ry, rx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    sv = slope[np.isfinite(slope)]
    pad = np.pad(z, 1, mode="edge")
    tri = np.sqrt(sum((pad[1 + dy:1 + dy + z.shape[0], 1 + dx:1 + dx + z.shape[1]] - z) ** 2
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx))
    tv = tri[np.isfinite(tri)]
    p = {k: round(float(np.percentile(v, q)), 1) for k, q in (("p5", 5), ("p50", 50), ("p95", 95))}
    stats = {"min_m": round(float(v.min()), 1), "max_m": round(float(v.max()), 1), **p,
             "relief_m": round(float(v.max() - v.min()), 1), "grid_m": round(rx, 1),
             "slope_mean_deg": round(float(sv.mean()), 1), "slope_p90_deg": round(float(np.percentile(sv, 90)), 1),
             "pct_slope_gt30": round(float((sv > 30).mean() * 100), 1), "tri_mean_m": round(float(tv.mean()), 1),
             "pct_le5m": round(float((v <= 5).mean() * 100), 1), "pct_le10m": round(float((v <= 10).mean() * 100), 1)}
    out = []
    lab = "derived from the DEM"
    out.append(fact("fabdem", f"Bare-earth elevation spans {stats['min_m']:,.0f}–{stats['max_m']:,.0f} m (relief "
                    f"{stats['relief_m']:,.0f} m); half the ground lies below {stats['p50']:,.0f} m ({lab}).",
                    {k: stats[k] for k in ("min_m", "max_m", "relief_m", "p5", "p50", "p95")}, "derived", "high", "tile", "relief"))
    if stats["pct_slope_gt30"] >= 1:
        out.append(fact("fabdem", f"{stats['pct_slope_gt30']:.0f}% of the ground is steeper than 30°, a common "
                        f"landslide-prone threshold; mean slope {stats['slope_mean_deg']:.0f}° ({lab}, 30 m grid).",
                        {k: stats[k] for k in ("pct_slope_gt30", "slope_mean_deg", "slope_p90_deg", "grid_m")},
                        "derived", "medium", "tile", "slope"))
    else:
        out.append(fact("fabdem", f"Gentle ground: mean slope {stats['slope_mean_deg']:.1f}°, with "
                        f"{stats['pct_slope_gt30']:.1f}% steeper than 30° ({lab}, 30 m grid).",
                        {k: stats[k] for k in ("pct_slope_gt30", "slope_mean_deg", "slope_p90_deg", "grid_m")},
                        "derived", "high", "tile", "slope"))
    if stats["pct_le10m"] >= 0.5:
        out.append(fact("fabdem", f"{stats['pct_le10m']:.0f}% of the ground is within 10 m of sea level "
                        f"({stats['pct_le5m']:.0f}% within 5 m), a coastal-flooding exposure proxy ({lab}; heights "
                        "above the EGM2008 geoid, ±2 m).", {k: stats[k] for k in ("pct_le5m", "pct_le10m")},
                        "derived", "medium", "tile", "lowlying"))
    if stats["tri_mean_m"] >= 5:
        out.append(fact("fabdem", f"Mean terrain ruggedness index {stats['tri_mean_m']:.0f} m between 30 m "
                        f"neighbours ({lab}).", {"tri_mean_m": stats["tri_mean_m"], "grid_m": stats["grid_m"]},
                        "derived", "medium", "tile", "tri"))
    return stats, bbox, out


# ------------------------------------------------------------------ C2: sources
def window(url, bbox):
    w_, s_, e_, n_ = bbox
    with rasterio.open("/vsicurl/" + url) as s:
        a = s.read(1, window=from_bounds(w_, s_, e_, n_, s.transform), boundless=True, fill_value=s.nodata or 0)
        return a.astype(np.float64), s.nodata


def wb_landslide(bbox):
    a, _ = window(WB_TH, bbox)
    v = a[(a >= 1) & (a <= 4)]
    if not v.size:
        return []
    vals, cnt = np.unique(v, return_counts=True)
    share = {int(k): round(float(c / v.size * 100), 1) for k, c in zip(vals, cnt)}
    top, low = int(v.max()), int(v.min())
    cls = f"class {top}" if top == low else f"class {low}–{top}"
    return [fact("wb_ls", f"Landslide hazard {cls} on the World Bank global map's 1–4 scale "
                 f"(4 = highest); {share.get(top, 0):.0f}% of the tile's ~1 km cells are class {top}.",
                 {"class_share_pct": share, "cells": int(v.size)}, "source", "medium", "tile")]


def jrc_flood(bbox, lat, lon):
    x = int(math.floor(lon / 10) * 10)
    y = int(math.ceil(lat / 10) * 10)
    name = f"{'N' if y >= 0 else 'S'}{abs(y)}_{'E' if x >= 0 else 'W'}{abs(x)}"
    ext = json.loads(httpx.get(f"{JRC_RP100}/tile_extents.geojson", timeout=60).text)
    tid = next(f["properties"]["id"] for f in ext["features"] if f["properties"]["name"] == name)
    a, nd = window(f"{JRC_RP100}/RP100/ID{tid}_{name}_RP100_depth.tif", bbox)
    wet = (a > 0) & (a != nd)
    pct = round(float(wet.mean() * 100), 1)
    val = {"pct_in_rp100_extent": pct, "max_depth_m": round(float(a[wet].max()), 1) if wet.any() else 0.0,
           "cells": int(a.size), "tile": f"ID{tid}_{name}"}
    if pct == 0:
        return [fact("jrc_flood", "No part of the tile lies in the modelled 1-in-100-year river flood extent "
                     "(~90 m grid; rivers with large catchments only).", val, "source", "medium", "tile")]
    return [fact("jrc_flood", f"{pct:.0f}% of the tile lies in the modelled 1-in-100-year river flood extent, up to "
                 f"{val['max_depth_m']:.1f} m deep (~90 m grid).", val, "source", "medium", "tile")]


def gsw(bbox, lat, lon):
    x = int(math.floor(lon / 10) * 10)
    y = int(math.ceil(lat / 10) * 10)
    t = f"{abs(x)}{'E' if x >= 0 else 'W'}_{abs(y)}{'N' if y >= 0 else 'S'}"
    a, _ = window(GSW.format(t=t), bbox)
    ok = a <= 100
    ever = round(float(((a > 0) & ok).mean() * 100), 1)
    perm = round(float(((a >= 90) & ok).mean() * 100), 1)
    if ever < 0.5:
        return []
    return [fact("gsw", f"Surface water was seen on {ever:.0f}% of the tile at some time in 1984–2021; "
                 f"{perm:.0f}% is water ≥90% of the time (permanent).", {"pct_ever_water": ever,
                 "pct_permanent_ge90": perm, "tile": t}, "source", "high", "tile")]


def usgs(client, lat, lon, radius=100, scope="within 100 km of the tile centre"):
    base = {"format": "geojson", "latitude": lat, "longitude": lon, "maxradiuskm": radius, "eventtype": "earthquake"}
    n = client.get(f"{USGS}/count", params={**base, "minmagnitude": 4.5, "starttime": "1973-01-01"}).json()["count"]
    big = client.get(f"{USGS}/query", params={**base, "minmagnitude": 4.0, "starttime": "1900-01-01",
                                              "orderby": "magnitude", "limit": 1}).json()["features"]
    val = {"count_m45_since_1973": n, "radius_km": radius}
    if big:
        p = big[0]["properties"]
        lo, la = big[0]["geometry"]["coordinates"][:2]
        val["largest"] = {"mag": p["mag"], "place": p["place"], "date": time.strftime("%Y-%m-%d", time.gmtime(p["time"] / 1000)),
                          "distance_km": round(hav_km(lat, lon, la, lo)), "url": p["url"]}
    if n == 0 and not big:
        text = f"No M4+ earthquake is catalogued {scope} since 1900."
    else:
        text = f"{n} earthquakes of M4.5+ {scope} since 1973"
        if big:
            L = val["largest"]
            text += f"; the largest since 1900 is M{L['mag']:.1f} ({L['date'][:4]}, {L['distance_km']} km away)."
        else:
            text += "."
    return [fact("usgs", text, val, "source", "high", f"{radius} km radius")]


def thinkhazard(client, name, admin1, cache):
    key = (name, admin1)
    if key not in cache:
        def search(q):
            return [h for h in client.get("https://thinkhazard.org/en/administrativedivision", params={"q": q}).json()["data"]
                    if h.get("admin2") and h.get("admin1") == admin1]
        hits = search(name)
        hit = next((h for h in hits if h["admin2"].lower() == name.lower()), hits[0] if len(hits) == 1 else None)
        if hit is None and len(name) >= 4:
            cand = search(name[:4])
            score = {h["code"]: difflib.SequenceMatcher(None, h["admin2"].lower(), name.lower()).ratio() for h in cand}
            best = max(cand, key=lambda h: score[h["code"]], default=None)
            hit = best if best and score[best["code"]] >= 0.7 else None
        cache[key] = None
        if hit:
            rep = client.get(f"https://thinkhazard.org/en/report/{hit['code']}.json").json()
            cache[key] = (hit, {h["hazardtype"]["mnemonic"]: h["hazardlevel"]["mnemonic"] for h in rep})
    if not cache[key]:
        return []
    hit, lv = cache[key]
    keep = {TH_NAMES[k]: TH_LEVEL[lv[k]] for k in TH_KEEP if lv.get(k) in TH_LEVEL}
    hi = [k for k, v in keep.items() if v == "high"]
    med = [k for k, v in keep.items() if v == "medium"]
    parts = []
    if hi:
        parts.append("high for " + ", ".join(hi))
    if med:
        parts.append("medium for " + ", ".join(med))
    if not parts:
        parts.append("low or very low for all mapped geographic hazards")
    return [fact("thinkhazard", f"{hit['admin2']}, {hit['admin1']} (district-level rating): hazard " + "; ".join(parts) + ".",
                 {"division_code": hit["code"], "admin": [hit["admin0"], hit["admin1"], hit["admin2"]], "levels": keep,
                  "url": hit.get("url")}, "source", "medium", "district")]


def district(client, lat, lon, last=[0.0]):  # noqa: B006 — Nominatim fair use: <= 1 request/s
    time.sleep(max(0.0, 1.1 - (time.monotonic() - last[0])))
    last[0] = time.monotonic()
    a = client.get("https://nominatim.openstreetmap.org/reverse",
                   params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 8, "addressdetails": 1}).json().get("address", {})
    names = [a.get(k) for k in ("state_district", "county", "city") if a.get(k)]
    names = [re.sub(r"\s+(District|district|Division)$", "", n) for n in names]
    return names, a.get("state")


def cyclones(tracks, lat, lon, radius=100):
    near = {}
    for sid, name, season, la, lo, wind in tracks:
        if abs(la - lat) < 1.5 and hav_km(lat, lon, la, lo) <= radius:
            w = near.setdefault(sid, [name, season, 0])
            w[2] = max(w[2], wind)
    ge64_any = any(v[2] >= 64 for v in near.values())
    if not near or not ge64_any:  # weak remnant tracks only: not a meaningful cyclone fact
        return []
    strongest = max(near.values(), key=lambda v: v[2])
    ge64 = sum(1 for v in near.values() if v[2] >= 64)
    val = {"storms": len(near), "hurricane_strength_ge64kt": ge64, "radius_km": radius, "since": 1980,
           "strongest": {"name": strongest[0].title(), "season": strongest[1], "max_wind_kt": strongest[2]}}
    return [fact("ibtracs", f"{len(near)} tropical cyclone{'s' if len(near) != 1 else ''} passed within {radius} km since 1980, {ge64} of them at "
                 f"hurricane/cyclone strength (≥64 kt); the strongest nearby was {strongest[0].title()} "
                 f"({strongest[1]}, {strongest[2]} kt).", val, "source", "high", f"{radius} km radius")]


def tornadoes(rows, lat, lon, radius=25):
    near = [r for r in rows if hav_km(lat, lon, r[0], r[1]) <= radius]
    if not near:
        return []
    strong = sum(1 for r in near if r[2] >= 3)
    val = {"count": len(near), "ef3_plus": strong, "radius_km": radius, "years": "1950–2024"}
    return [fact("spc", f"{len(near)} tornado touchdowns within {radius} km of the city centre in 1950–2024, "
                 f"{strong} of them (E)F3 or stronger.", val, "source", "high", f"{radius} km of city centre")]


def wikidata(client, bbox):
    w_, s_, e_, n_ = bbox
    vals = " ".join(f"wd:{q}" for q in WD_CLASSES)
    q = f"""SELECT DISTINCT ?item ?itemLabel ?class ?elev WHERE {{
      SERVICE wikibase:box {{ ?item wdt:P625 ?loc . bd:serviceParam wikibase:cornerSouthWest "Point({w_} {s_})"^^geo:wktLiteral .
                               bd:serviceParam wikibase:cornerNorthEast "Point({e_} {n_})"^^geo:wktLiteral . }}
      ?item wdt:P31 ?class . VALUES ?class {{ {vals} }} OPTIONAL {{ ?item wdt:P2044 ?elev }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT 60"""
    b = client.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"}).json()["results"]["bindings"]
    items = {}
    for x in b:
        lab = x["itemLabel"]["value"]
        if re.fullmatch(r"Q\d+", lab):
            continue
        items.setdefault(lab, (WD_CLASSES[x["class"]["value"].rsplit("/", 1)[1]], x.get("elev", {}).get("value")))
    if not items:
        return []
    order = sorted(items.items(), key=lambda kv: (kv[1][0] not in ("mountain", "summit", "glacier", "river"), kv[0]))[:6]
    txt = ", ".join(f"{k} ({c}{', ' + str(round(float(e))) + ' m' if e else ''})" for k, (c, e) in order)
    return [fact("wikidata", f"Named natural features inside the tile: {txt}.",
                 {"features": [{"name": k, "class": c, "elevation_m": e} for k, (c, e) in items.items()]},
                 "source", "medium", "tile")]


def load_ibtracs(path):
    out = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        next(r)  # units row
        ix = {k: head.index(k) for k in ("SID", "NAME", "SEASON", "LAT", "LON", "WMO_WIND", "USA_WIND", "TRACK_TYPE")}
        for row in r:
            if row[ix["TRACK_TYPE"]] != "main":  # spur tracks duplicate storms
                continue
            try:
                wind = row[ix["USA_WIND"]].strip() or row[ix["WMO_WIND"]].strip()
                out.append((row[ix["SID"]], row[ix["NAME"]], int(row[ix["SEASON"]]), float(row[ix["LAT"]]),
                            float(row[ix["LON"]]), int(float(wind)) if wind else 0))
            except ValueError:
                continue
    return out


def load_spc(path):
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append((float(row["slat"]), float(row["slon"]), int(row["mag"])))
            except (ValueError, KeyError):
                continue
    return out


def safe(fn, *a, log=None, name=""):
    try:
        return fn(*a)
    except Exception as exc:  # noqa: BLE001 — a failed source yields no fact, and the failure is recorded
        if log is not None:
            log.append(f"{name}: {type(exc).__name__}: {str(exc)[:160]}")
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / f"data/library_v2_{TODAY}")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    manifest = json.loads((LIB / "tile_manifest.json").read_text())
    tiles = sorted((t for t in manifest["tiles"] if t["include_in_container"]), key=lambda t: t["order_index"])
    if a.only:
        tiles = [t for t in tiles if t["tile_id"] in set(a.only)]
    tracks = load_ibtracs(a.cache / "ibtracs.since1980.csv")
    torn = load_spc(a.cache / "spc_tornadoes.csv")
    th_cache, city_cache = {}, {}
    out = {"generated": TODAY, "generator": "scripts/draft_library_facts.py", "library": str(LIB.relative_to(ROOT)),
           "status_rule": "only facts with status 'curated' are baked (scripts/bake_static_library.py)", "tiles": {}}
    with httpx.Client(timeout=60, headers={"User-Agent": UA}, follow_redirects=True) as client:
        for t in tiles:
            iid, log, facts = t["tile_id"], [], []
            t0 = time.time()
            if t["source"] == "dfc2019":
                code = iid.split("-", 1)[1][:3]
                if code not in city_cache:
                    city, la, lo, county, state = CITIES[code]
                    f = safe(thinkhazard, client, county, state, th_cache, log=log, name="thinkhazard")
                    f += safe(usgs, client, la, lo, 100, "within 100 km of the city centre", log=log, name="usgs")
                    f += safe(cyclones, tracks, la, lo, log=log, name="ibtracs")
                    f += safe(tornadoes, torn, la, lo, log=log, name="spc")
                    for x in f:
                        x["text"] = f"{city} (city-level): {x['text']}"
                    city_cache[code] = (f, log, {"city": city, "lat": la, "lon": lo})
                facts, log, where = copy.deepcopy(city_cache[code])  # own copies: ids are per tile
                entry = {"display_name": t["display_name"], "source": t["source"], "where": where,
                         "note": "DFC2019 tile locations are private: city-level facts only, no DEM-derived facts.",
                         "facts": facts, "log": log}
            else:
                stats, bbox, facts = dem_facts(LIB / "dem" / f"{iid}.tif")
                lat, lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
                names, state = safe(lambda: district(client, lat, lon), log=log, name="nominatim") or ([], None)
                for n in names:
                    th = safe(thinkhazard, client, n, state, th_cache, log=log, name="thinkhazard")
                    if th:
                        facts += th
                        break
                else:
                    log.append(f"thinkhazard: no division matched {names} in {state}")
                facts += safe(wb_landslide, bbox, log=log, name="wb_landslide")
                facts += safe(jrc_flood, bbox, lat, lon, log=log, name="jrc_flood")
                facts += safe(gsw, bbox, lat, lon, log=log, name="gsw")
                facts += safe(usgs, client, lat, lon, log=log, name="usgs")
                facts += safe(cyclones, tracks, lat, lon, log=log, name="ibtracs")
                facts += safe(wikidata, client, bbox, log=log, name="wikidata")
                entry = {"display_name": t["display_name"], "source": t["source"],
                         "where": {"bbox_wgs84": [round(v, 5) for v in bbox], "centre": [round(lat, 5), round(lon, 5)]},
                         "dem_stats": stats, "facts": facts, "log": log}
            entry["facts"] = sorted(entry["facts"], key=lambda f: PRIORITY.index(f["kind"]))[:6]
            for i, f in enumerate(entry["facts"]):
                f["id"] = f"{iid}#{i + 1}"
            out["tiles"][iid] = entry
            print(f"{iid:32s} {len(entry['facts'])} facts  {time.time() - t0:5.1f}s  {'; '.join(log)[:120]}", flush=True)
    write_outputs(out, a.out)


def write_outputs(out: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "facts_drafts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    md = [f"# Library facts: DRAFTS for curation ({TODAY})", "",
          "Every fact below has status **draft**. To publish one, set its `status` to `curated` in `facts_drafts.json`",
          "(optionally edit `text`), then re-run `scripts/bake_static_library.py`. Only curated facts are baked.",
          "Tiles show 3-6 facts where the sources had data; a failed or empty source adds nothing.", ""]
    for iid, e in out["tiles"].items():
        md.append(f"## {e['display_name']} (`{iid}`, {e['source']})")
        if e.get("note"):
            md.append(f"_{e['note']}_")
        md.append("")
        for f in e["facts"]:
            md.append(f"- **{f['id']}** [{f['origin']}, {f['confidence']}, {f['scope']}] {f['text']}  ")
            md.append(f"  _Source:_ {f['source']}. _Licence:_ {f['licence']}. _Value:_ `{json.dumps(f['value'], ensure_ascii=False)[:220]}`")
        if e["log"]:
            md.append(f"- _Log:_ {'; '.join(e['log'])}")
        md.append("")
    (folder / "facts_drafts.md").write_text("\n".join(md))
    n = sum(len(e["facts"]) for e in out["tiles"].values())
    print(f"{len(out['tiles'])} tiles, {n} draft facts -> {folder}")


if __name__ == "__main__":
    main()
