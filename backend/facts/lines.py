"""One short UI line per fact, shared by the live /api/facts route and the library drafts
(scripts/draft_library_facts.py), so a baked library tile and a live scene read the same.

A line is {"group", "kind", "label", "text"} (+ "data" for the earthquake inset map). Groups:
  "facts"      hazards Scenario Analysis doesn't cover (wildfire; cyclone, tsunami, volcano where they apply;
               cyclone/tornado history) + named terrain features (peak, river, glacier)
  "flood" / "landslide" / "earthquake"   the Scenario Analysis cards
Every builder returns [] when its value doesn't exist: the UI never shows "not available".
"""

from __future__ import annotations

LEVEL_WORD = {"H": "HIGH", "M": "MEDIUM", "L": "LOW", "V": "VERY LOW"}
# ThinkHazard! hazard codes in the order of backend/facts/data/thinkhazard_levels.json
TH_HAZARDS = ["FL", "UF", "CF", "EQ", "LS", "TS", "VA", "CY", "WF"]
EQ_RADIUS_KM = 100


def line(group, kind, label, text, data=None):
    out = {"group": group, "kind": kind, "label": label, "text": text}
    if data is not None:
        out["data"] = data
    return out


def km(d):
    return f"{d:.1f} km" if d < 10 else f"{d:.0f} km"


# ------------------------------------------------------------------ ThinkHazard! district levels
def thinkhazard_lines(levels: dict[str, str] | None, scope: str = "district") -> list[dict]:
    """levels: {"WF": "H", ...} (H/M/L/V). Facts get wildfire always, cyclone/tsunami/volcano only when they apply
    (LOW or above); the scenario cards get river/coastal flood, landslide and earthquake."""
    if not levels:
        return []
    out = []
    word = lambda hz: f"{LEVEL_WORD[levels[hz]]} ({scope})"  # noqa: E731
    if levels.get("WF") in LEVEL_WORD:
        out.append(line("facts", "wildfire", "Wildfire hazard", word("WF")))
    for hz, kind, label in (("CY", "cyclone", "Cyclone hazard"), ("TS", "tsunami", "Tsunami hazard"),
                            ("VA", "volcano", "Volcanic hazard")):
        if levels.get(hz) in ("H", "M", "L"):
            out.append(line("facts", kind, label, word(hz)))
    if levels.get("FL") in LEVEL_WORD:
        out.append(line("flood", "flood_district", "River flood", word("FL")))
    if levels.get("CF") in ("H", "M", "L"):
        out.append(line("flood", "coastal_district", "Coastal flood", word("CF")))
    if levels.get("LS") in LEVEL_WORD:
        out.append(line("landslide", "landslide_district", "Landslide", word("LS")))
    if levels.get("EQ") in LEVEL_WORD:
        out.append(line("earthquake", "earthquake_district", "Earthquake", word("EQ")))
    return out


# ------------------------------------------------------------------ named terrain features (Wikidata)
def feature_lines(features: dict) -> list[dict]:
    """features: {"peak": {name, elevation_m, distance_km}, "river": {name, main: bool}, "glacier": {name, distance_km}}"""
    out = []
    p = features.get("peak")
    if p and p.get("name"):
        bits = [p["name"]]
        if p.get("elevation_m") is not None:
            bits.append(f"{round(p['elevation_m']):,} m")
        bits.append(km(p["distance_km"]))
        out.append(line("facts", "peak", "Nearest named peak", " · ".join(bits)))
    r = features.get("river")
    if r and r.get("name"):
        out.append(line("facts", "river", "Main river" if r.get("main") else "Nearest river", r["name"]))
    g = features.get("glacier")
    if g and g.get("name"):
        out.append(line("facts", "glacier", "Glacier", f"{g['name']} · {km(g['distance_km'])}"))
    return out


# ------------------------------------------------------------------ flood / landslide / earthquake cards
def jrc_flood_lines(v: dict | None) -> list[dict]:
    """v: {"pct": %, "max_depth_m": m} for a bbox, or {"inside": bool} for a point."""
    if not v:
        return []
    if "inside" in v:
        return [line("flood", "rp100", "1-in-100-yr river flood", "inside the modelled extent" if v["inside"]
                     else "outside the modelled extent")]
    if v["pct"] <= 0:
        return [line("flood", "rp100", "1-in-100-yr river flood", "none modelled in this tile")]
    return [line("flood", "rp100", "1-in-100-yr river flood", f"{v['pct']:.0f}% of tile · up to {v['max_depth_m']:.1f} m deep")]


def gsw_lines(v: dict | None) -> list[dict]:
    if not v or v["pct_ever"] < 0.5:
        return []
    return [line("flood", "surface_water", "Surface water 1984–2021", f"{v['pct_ever']:.0f}% of tile · {v['pct_permanent']:.0f}% permanent")]


def wb_landslide_lines(v: dict | None) -> list[dict]:
    """v: {"min": c, "max": c, "share_max_pct": %} (bbox) or {"class": c} (point); classes 1-4."""
    if not v:
        return []
    if "class" in v:
        return [line("landslide", "wb_class", "Landslide hazard, 1 km map", f"class {v['class']} of 4")]
    cls = f"class {v['max']}" if v["min"] == v["max"] else f"class {v['min']}–{v['max']}"
    tail = "" if v["min"] == v["max"] else f" · {v['share_max_pct']:.0f}% class {v['max']}"
    return [line("landslide", "wb_class", "Landslide hazard, 1 km map", f"{cls} of 4{tail}")]


def usgs_lines(v: dict | None) -> list[dict]:
    """v: {"radius_km", "count_since_1973", "largest": {mag, year, distance_km} | None, "events": [[lon, lat, mag, year]...],
    "centre": [lat, lon], "bbox": [w, s, e, n] | None}"""
    if not v:
        return []
    r = v["radius_km"]
    out = [line("earthquake", "usgs_count", f"M4.5+ within {r} km", f"{v['count_since_1973']} since 1973")]
    if v.get("largest"):
        L = v["largest"]
        out.append(line("earthquake", "usgs_largest", f"Largest within {r} km",
                        f"M{L['mag']:.1f} · {L['year']} · {round(L['distance_km'])} km away"))
    if v.get("events"):  # the inset map needs at least one epicentre
        out.append(line("earthquake", "epicentres", "", "", data={k: v.get(k) for k in ("radius_km", "centre", "bbox", "events")}))
    return out


# display order within each group (tile-specific values before district ratings)
ORDER = ["wildfire", "cyclone", "tsunami", "volcano", "cyclone_history", "tornado", "peak", "river", "glacier",
         "rp100", "surface_water", "flood_district", "coastal_district", "wb_class", "landslide_district",
         "earthquake_district", "usgs_count", "usgs_largest", "epicentres"]


def group_lines(lines: list[dict]) -> dict:
    """[line] -> {"facts": [...], "scenario": {"flood": [...], "landslide": [...], "earthquake": [...]}}, each group in
    ORDER"""
    out = {"facts": [], "scenario": {"flood": [], "landslide": [], "earthquake": []}}
    lines = sorted(lines, key=lambda ln: ORDER.index(ln["kind"]) if ln["kind"] in ORDER else len(ORDER))
    for ln in lines:
        (out["facts"] if ln["group"] == "facts" else out["scenario"][ln["group"]]).append(ln)
    return out
