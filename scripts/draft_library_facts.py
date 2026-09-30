"""Draft the Facts box + Scenario Analysis cards for every included library tile (docs/facts-research.md, v2 Part C).

v2 (2026-09-30) replaces scripts/draft_library_facts_v1.py. Per tile, two groups of short UI lines
(backend/facts/lines.py, the same wording as the live /api/facts route):
  facts      wildfire; cyclone / tsunami / volcano where they apply (ThinkHazard!); cyclone-track history (IBTrACS);
             tornadoes (NOAA SPC, US cities); nearest named peak, main or nearest river, glacier (Wikidata)
  scenario   flood: JRC/GloFAS 1-in-100-yr extent, JRC Global Surface Water, ThinkHazard river/coastal flood level;
             landslide: World Bank 1 km class, ThinkHazard level; earthquake: ThinkHazard level, USGS M4.5+ within
             100 km + largest, and the epicentres for the inset map
Every item is a retrieved value with its source, licence, retrieval date and raw value, status "draft". Nothing is
baked until an item is set to "curated" (scripts/bake_static_library.py).

ThinkHazard district: POINT-IN-POLYGON on ThinkHazard's own admin areas (report/<any>/neighbours.geojson with a
~1 m bbox at the tile centre), not name matching (v1 missed 11/31 tiles on GAUL spellings and renamed districts).
DFC2019: tile locations are private, so city-level only (public city centre; county via point-in-polygon), and no
epicentre map (it would imply a tile position).

  python scripts/draft_library_facts.py --cache <dir with ibtracs.since1980.csv, spc_tornadoes.csv>
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.facts import lines as L  # noqa: E402
from backend.facts import sources as S  # noqa: E402

LIB = ROOT / "data/library_v2_2026-09-29"
TODAY = date.today().isoformat()
UA = "DepthWizard2/0.2 (SIH26175 prototype; library facts draft; https://github.com/sancharimouri/depthwizard-studio)"
CITIES = {"JAX": ("Jacksonville, Florida", 30.3322, -81.6557), "OMA": ("Omaha, Nebraska", 41.2565, -95.9345)}
TH_LEVEL = {"HIG": "H", "MED": "M", "LOW": "L", "VLO": "V"}

SRC = {
    "thinkhazard": ("ThinkHazard! (GFDRR, World Bank)", "https://thinkhazard.org",
                    "CC BY 4.0 (hazard levels); attribution: ThinkHazard! – GFDRR", "medium"),
    "wikidata": ("Wikidata", "https://query.wikidata.org", "CC0 1.0; attribution: Wikidata", "medium"),
    "jrc_flood": ("Copernicus EMS / GloFAS global river flood hazard map, 1-in-100-year (v2.1.2, ~90 m)",
                  "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/",
                  "CC BY 4.0; attribution: © European Union, Copernicus Emergency Management Service (GloFAS)", "medium"),
    "gsw": ("JRC Global Surface Water occurrence 1984–2021 (v1.4, 30 m)", "https://global-surface-water.appspot.com",
            "Free, no restriction of use (Copernicus); attribution: EC JRC/Google, Pekel et al. (2016) Nature 540", "high"),
    "wb_ls": ("Global Landslide Hazard Map, total hazard (World Bank / ARUP 2021, ~1 km)", S.WB_LANDSLIDE,
              "World Bank Data Catalog dataset 0037584 (licence text to verify)", "medium"),
    "usgs": ("USGS ANSS Comprehensive Earthquake Catalog (ComCat)", S.USGS,
             "Public domain; attribution: U.S. Geological Survey", "high"),
    "ibtracs": ("NOAA NCEI IBTrACS v04r01", "https://www.ncei.noaa.gov/products/international-best-track-archive",
                "Public domain; attribution: Knapp et al. (2010), IBTrACS, NOAA NCEI", "high"),
    "spc": ("NOAA Storm Prediction Center tornado database 1950–2024", "https://www.spc.noaa.gov/wcm/",
            "Public domain; attribution: NOAA/NWS Storm Prediction Center", "high"),
}
KIND_SRC = {"wildfire": "thinkhazard", "cyclone": "thinkhazard", "tsunami": "thinkhazard", "volcano": "thinkhazard",
            "flood_district": "thinkhazard", "coastal_district": "thinkhazard", "landslide_district": "thinkhazard",
            "earthquake_district": "thinkhazard", "peak": "wikidata", "river": "wikidata", "glacier": "wikidata",
            "rp100": "jrc_flood", "surface_water": "gsw", "wb_class": "wb_ls", "usgs_count": "usgs",
            "usgs_largest": "usgs", "epicentres": "usgs", "cyclone_history": "ibtracs", "tornado": "spc"}


def item(ln, value, scope):
    name, url, lic, conf = SRC[KIND_SRC[ln["kind"]]]
    return {**ln, "source": name, "source_url": url, "licence": lic, "retrieved": TODAY, "value": value,
            "confidence": conf, "scope": scope, "status": "draft"}


def _merc(lon, lat):
    return 6378137.0 * math.radians(lon), 6378137.0 * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _divisions(client, w, s, e, n):
    feats = client.get("https://thinkhazard.org/en/report/17948/neighbours.geojson",
                       params={"bbox": f"{w},{s},{e},{n}"}).json()
    return feats["features"] if isinstance(feats, dict) else feats


def thinkhazard_point(client, lat, lon, bbox=None) -> dict | None:
    """Point-in-polygon on ThinkHazard's admin areas: the admin-2 division under a ~1 m box at the point. If the point
    is in water (no division), the division covering most of the tile bbox instead (geometries are EPSG:3857)."""
    d = 1e-5
    feats = _divisions(client, lon - d, lat - d, lon + d, lat + d)
    how = "tile centre"
    if not feats and bbox:
        from shapely.geometry import box, shape

        tile = box(*_merc(bbox[0], bbox[1]), *_merc(bbox[2], bbox[3]))
        cands = [(shape(f["geometry"]).buffer(0).intersection(tile).area, f) for f in _divisions(client, *bbox) if f.get("geometry")]
        feats = [max(cands, key=lambda c: c[0])[1]] if cands and max(c[0] for c in cands) > 0 else []
        how = "largest share of the tile"
    if not feats:
        return None
    p = feats[0]["properties"]
    rep = client.get(f"https://thinkhazard.org/en/report/{p['code']}.json").json()
    levels = {h["hazardtype"]["mnemonic"]: TH_LEVEL[h["hazardlevel"]["mnemonic"]] for h in rep
              if h["hazardlevel"]["mnemonic"] in TH_LEVEL}
    return {"code": p["code"], "name": p["name"], "levels": levels, "matched_by": how}


def cyclone_history(tracks, lat, lon, radius=100):
    near = {}
    for sid, name, season, la, lo, wind in tracks:
        if abs(la - lat) < 1.5 and S.haversine_km(lat, lon, la, lo) <= radius:
            w = near.setdefault(sid, [name, season, 0])
            w[2] = max(w[2], wind)
    ge64 = [v for v in near.values() if v[2] >= 64]
    if not ge64:  # weak remnant tracks only: not a meaningful cyclone fact
        return [], None
    val = {"storms": len(near), "ge64kt": len(ge64), "radius_km": radius, "since": 1980,
           "strongest": dict(zip(("name", "season", "max_wind_kt"), max(near.values(), key=lambda v: v[2])))}
    return [L.line("facts", "cyclone_history", f"Cyclones within {radius} km",
                   f"{len(near)} since 1980 · {len(ge64)} at ≥64 kt")], val


def tornadoes(rows, lat, lon, radius=25):
    near = [r for r in rows if S.haversine_km(lat, lon, r[0], r[1]) <= radius]
    if not near:
        return [], None
    strong = sum(1 for r in near if r[2] >= 3)
    val = {"count": len(near), "ef3_plus": strong, "radius_km": radius, "years": "1950–2024"}
    return [L.line("facts", "tornado", f"Tornadoes within {radius} km", f"{len(near)} since 1950 · {strong} EF3+")], val


def load_ibtracs(path):
    out = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        next(r)
        ix = {k: head.index(k) for k in ("SID", "NAME", "SEASON", "LAT", "LON", "WMO_WIND", "USA_WIND", "TRACK_TYPE")}
        for row in r:
            if row[ix["TRACK_TYPE"]] != "main":
                continue
            try:
                wind = row[ix["USA_WIND"]].strip() or row[ix["WMO_WIND"]].strip()
                out.append((row[ix["SID"]], row[ix["NAME"]].title(), int(row[ix["SEASON"]]), float(row[ix["LAT"]]),
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


def run(log, name, fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — a failed source adds nothing; the failure is logged per tile
        log.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
        return None


def draft_located(client, bbox, tracks, log):
    lat, lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
    items = []
    th = run(log, "thinkhazard", lambda: thinkhazard_point(client, lat, lon, bbox))
    if th:
        for ln in L.thinkhazard_lines(th["levels"]):
            items.append(item(ln, {"division": th["name"], "code": th["code"], "levels": th["levels"],
                                   "matched_by": th["matched_by"]}, "district"))
    else:
        log.append("thinkhazard: no division at the tile centre")
    feats = run(log, "wikidata", lambda: S.wikidata_features(client, lat, lon))
    for ln in L.feature_lines(feats or {}):
        items.append(item(ln, (feats or {}).get(ln["kind"]), "15 km of tile centre" if ln["kind"] != "river" else "10 km of tile centre"))
    for name, fn, lf in (("jrc_flood", lambda: S.jrc_flood(lat, lon, bbox), L.jrc_flood_lines),
                         ("gsw", lambda: S.gsw(lat, lon, bbox), L.gsw_lines),
                         ("wb_landslide", lambda: S.wb_landslide(lat, lon, bbox), L.wb_landslide_lines)):
        v = run(log, name, fn)
        for ln in lf(v):
            items.append(item(ln, v, "tile"))
    q = run(log, "usgs", lambda: S.usgs(client, lat, lon, bbox))
    for ln in L.usgs_lines(q):
        items.append(item(ln, None if ln["kind"] == "epicentres" else {k: q[k] for k in ("radius_km", "count_since_1973", "largest")},
                          f"{L.EQ_RADIUS_KM} km of tile centre"))
    lines, val = cyclone_history(tracks, lat, lon)
    items += [item(ln, val, "100 km of tile centre") for ln in lines]
    return items


def draft_city(client, code, tracks, torn, log):
    city, lat, lon = CITIES[code]
    items = []
    th = run(log, "thinkhazard", lambda: thinkhazard_point(client, lat, lon))
    if th:
        for ln in L.thinkhazard_lines(th["levels"], scope="county"):
            items.append(item(ln, {"division": th["name"], "code": th["code"], "levels": th["levels"]}, "county"))
    q = run(log, "usgs", lambda: S.usgs(client, lat, lon))
    for ln in L.usgs_lines(q):
        if ln["kind"] == "epicentres":  # an inset map would imply a (private) tile position
            continue
        ln["label"] += " of the city"
        items.append(item(ln, {k: q[k] for k in ("radius_km", "count_since_1973", "largest")}, "city centre"))
    for lines, val in (cyclone_history(tracks, lat, lon), tornadoes(torn, lat, lon)):
        items += [item(ln, val, "city centre") for ln in lines]
    return items, {"city": city, "lat": lat, "lon": lon}


def split(items):
    g = L.group_lines(items)
    return g["facts"], g["scenario"]


def write_outputs(out, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "facts_drafts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    md = [f"# Library Facts + Scenario cards: DRAFTS for curation ({out['generated']})", "",
          "Two groups per tile. **Facts** (the Facts box) and **Scenario** (flood / landslide / earthquake cards).",
          "To publish an item set its `status` to `curated` in `facts_drafts.json` (you may edit `text`), then run",
          "`scripts/bake_static_library.py`. Only curated items are baked; a tile with no curated Facts items hides the",
          "Facts box. Sources are listed here for review only; the app's panels show none (Docs page credits them).", ""]
    for iid, e in out["tiles"].items():
        md += [f"## {e['display_name']} (`{iid}`)", ""]
        if e.get("note"):
            md += [f"_{e['note']}_", ""]
        md.append("**Facts**")
        md += [f"- `{x['id']}` **{x['label']}**: {x['text']}  _({x['source']}; {x['scope']}; {x['confidence']})_" for x in e["facts"]] or ["- _(none)_"]
        for grp in ("flood", "landslide", "earthquake"):
            rows = [x for x in e["scenario"][grp] if x["kind"] != "epicentres"]
            n_ep = sum(len((x.get("data") or {}).get("events") or []) for x in e["scenario"][grp] if x["kind"] == "epicentres")
            md.append(f"**Scenario · {grp}**")
            md += [f"- `{x['id']}` **{x['label']}**: {x['text']}  _({x['source']}; {x['scope']}; {x['confidence']})_" for x in rows] or ["- _(none)_"]
            if n_ep:
                ep = next(x for x in e["scenario"][grp] if x["kind"] == "epicentres")
                md.append(f"- `{ep['id']}` _epicentre map: {n_ep} M4.5+ events within {L.EQ_RADIUS_KM} km since 1973 (USGS)_")
        if e["log"]:
            md.append(f"- _Log:_ {'; '.join(e['log'])}")
        md.append("")
    (folder / "facts_drafts.md").write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data/library_v2_2026-09-30")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    manifest = json.loads((LIB / "tile_manifest.json").read_text())
    tiles = sorted((t for t in manifest["tiles"] if t["include_in_container"]), key=lambda t: t["order_index"])
    if a.only:
        tiles = [t for t in tiles if t["tile_id"] in set(a.only)]
    tracks = load_ibtracs(a.cache / "ibtracs.since1980.csv")
    torn = load_spc(a.cache / "spc_tornadoes.csv")
    out = {"generated": TODAY, "version": 2, "generator": "scripts/draft_library_facts.py",
           "groups": {"facts": "Facts box", "scenario": "Scenario Analysis cards: flood / landslide / earthquake"},
           "status_rule": "only items with status 'curated' are baked (scripts/bake_static_library.py)", "tiles": {}}
    city_cache = {}
    with httpx.Client(timeout=30, headers={"User-Agent": UA, "Accept": "application/json"}, follow_redirects=True) as client:
        for t in tiles:
            iid, log = t["tile_id"], []
            t0 = time.time()
            if t["source"] == "dfc2019":
                code = iid.split("-", 1)[1][:3]
                if code not in city_cache:
                    city_cache[code] = draft_city(client, code, tracks, torn, log) + (list(log),)
                items, where, log = copy.deepcopy(city_cache[code])
                note = "DFC2019 tile locations are private: city-level lines only, and no epicentre map."
            else:
                with rasterio.open(LIB / "dem" / f"{iid}.tif") as s:
                    bbox = [round(v, 5) for v in transform_bounds(s.crs, "EPSG:4326", *s.bounds)]
                items = draft_located(client, bbox, tracks, log)
                where, note = {"bbox_wgs84": bbox}, None
            facts, scenario = split(items)
            n = 0
            for x in facts + [x for g in scenario.values() for x in g]:
                n += 1
                x["id"] = f"{iid}#{n}"
            out["tiles"][iid] = {"display_name": t["display_name"], "source": t["source"], "where": where,
                                 "note": note, "facts": facts, "scenario": scenario, "log": log}
            print(f"{iid:30s} facts {len(facts)}  flood {len(scenario['flood'])}  landslide {len(scenario['landslide'])}  "
                  f"earthquake {len(scenario['earthquake'])}  {time.time() - t0:5.1f}s  {'; '.join(log)[:90]}", flush=True)
    write_outputs(out, a.out)
    n = sum(len(e["facts"]) + sum(len(g) for g in e["scenario"].values()) for e in out["tiles"].values())
    print(f"{len(out['tiles'])} tiles, {n} draft items -> {a.out}")


if __name__ == "__main__":
    main()
