"""Bake the cosmetic Bengaluru urban model for the demo page (#/demo).

A ~500 x 500 m block around UB City / Vittal Mallya Road / the Cubbon Park edge:
- building outlines: Google Open Buildings v3 polygons (Earth Engine)
- building heights:  Google Open Buildings 2.5D Temporal v1, 2023, band building_height,
                     median over the outline's building pixels (building_presence > 0.5)
- roads, labels:     OpenStreetMap, one Overpass query for the box (road names, and the LANDMARKS below)
- ground:            FABDEM (backend/dem/fabdem.py), or flat at the median if the relief is negligible

Writes frontend/public/cityscape/bengaluru.json in local metres (x east, y north, origin = box centre).
Needs EARTHENGINE_PROJECT in .env and local Earth Engine credentials.

    .venv/bin/python scripts/bake_cityscape.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiLineString, Polygon, box, mapping, shape
from shapely.ops import linemerge, transform as shp_transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "frontend" / "public" / "cityscape" / "bengaluru.json"

# Box centre: 150 m east and 150 m south of the UB City towers (OSM way 38773290, 12.9723 N 77.5959 E), so UB City
# sits in the north-west quarter; the densest 500 m box around it (394 outlines at confidence >= 0.70, 30% built).
CENTRE_LAT, CENTRE_LON = 12.97095, 77.59728
HALF = 250.0  # m
UTM = "EPSG:32643"

MIN_CONFIDENCE = 0.70  # Open Buildings v3 confidence; keeps 94% of the box's built area (0.75 keeps 79%)
PRESENCE_MIN = 0.5     # temporal building_presence above which a pixel counts as building
HEIGHT_RES = 2.0       # m, grid the 2023 heights are sampled on
MIN_AREA = 25.0        # m², smaller outlines get the fallback height
MIN_PIXELS = 3         # fewer valid building pixels -> fallback height
SIMPLIFY = 0.5         # m
FLAT_RELIEF = 3.0      # m, FABDEM p95-p5 below this -> flat ground
# OSM name -> label. Checked against public listings (UB City's blocks: UB Tower, Concorde, Comet, Canberra).
LANDMARKS = {"UB Tower": "UB Tower", "Concorde": "Concorde", "Comet": "Comet", "Canberra": "Canberra",
             "Prestige Kingfisher Towers": "Kingfisher Towers", "JW Marriott": "JW Marriott",
             "Bishop Cotton Boys School": "Bishop Cotton Boys' School"}
LABEL_ROADS = ("primary", "secondary", "tertiary")  # named roads of these classes get a label
ROAD_WIDTH = {"trunk": 14, "primary": 12, "primary_link": 7, "secondary": 10, "secondary_link": 6,
              "tertiary": 8, "tertiary_link": 5, "residential": 6, "unclassified": 6, "service": 4,
              "living_street": 5, "pedestrian": 4, "footway": 2, "path": 2, "cycleway": 2}


def r1(v):
    return round(float(v), 1)


def flat_coords(geom):
    return [r1(c) for xy in list(geom.coords) for c in xy]


def main():
    load_dotenv(ROOT / ".env")
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    cx, cy = (round(v / 10) * 10 for v in to_utm.transform(CENTRE_LON, CENTRE_LAT))  # on a 10 m grid
    west, south, east, north = cx - HALF, cy - HALF, cx + HALF, cy + HALF
    clip_local = box(-HALF, -HALF, HALF, HALF)
    corners = [to_ll.transform(x, y) for x, y in ((west, south), (east, south), (east, north), (west, north))]
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    bbox_ll = (min(lats), min(lons), max(lats), max(lons))  # S, W, N, E

    def to_local(x, y, z=None):
        ux, uy = to_utm.transform(x, y)
        return np.asarray(ux) - cx, np.asarray(uy) - cy

    def local(geom_ll):
        return shp_transform(to_local, geom_ll)

    # ---------------- OpenStreetMap: roads + landmark outlines, one query ----------------
    s, w, n, e = bbox_ll
    names = "|".join(LANDMARKS)
    q = (f"[out:json][timeout:60];(way[highway]({s},{w},{n},{e});way[name~\"^({names})$\"]({s},{w},{n},{e}););out geom;")
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "depthwizard-cityscape-bake/1.0"})
    for attempt in range(4):  # the public Overpass instance returns 429/504 under load
        try:
            osm = json.load(urllib.request.urlopen(req, timeout=120))["elements"]
            break
        except urllib.error.HTTPError as exc:
            if attempt == 3 or exc.code not in (429, 502, 503, 504):
                raise
            time.sleep(20 * (attempt + 1))
    roads, road_named, landmarks, major_roads = [], {}, {}, set()
    for el in osm:
        tags = el.get("tags", {})
        if el["type"] != "way":
            continue
        if tags.get("highway") in ROAD_WIDTH:
            g = clip_local.intersection(local(LineString([(p["lon"], p["lat"]) for p in el["geometry"]])))
            parts = g.geoms if isinstance(g, MultiLineString) else [g]
            for part in parts:
                if isinstance(part, LineString) and part.length > 1:
                    roads.append({"w": ROAD_WIDTH[tags["highway"]], "p": flat_coords(part.simplify(SIMPLIFY))})
                    if tags.get("name"):
                        road_named.setdefault(tags["name"], []).append(part)
                        if tags["highway"] in LABEL_ROADS:
                            major_roads.add(tags["name"])
        elif tags.get("name") in LANDMARKS and len(el["geometry"]) >= 4:
            poly = local(Polygon([(p["lon"], p["lat"]) for p in el["geometry"]])).buffer(0)
            if clip_local.contains(poly.centroid):
                landmarks[LANDMARKS[tags["name"]]] = poly

    # one label per road, at the middle of its longest piece in the box, oriented along it
    road_labels, road_lengths = [], {}
    for name, parts in road_named.items():
        merged = linemerge(parts)
        line = max(getattr(merged, "geoms", [merged]), key=lambda l: l.length)
        road_lengths[name] = round(line.length)
        if line.length < (60 if name in major_roads else 100):
            continue
        mid, a, b = (line.interpolate(line.length * t) for t in (0.5, 0.45, 0.55))
        road_labels.append([name, r1(mid.x), r1(mid.y), r1(b.x - a.x), r1(b.y - a.y), 1 if name in major_roads else 2])

    # ---------------- Earth Engine: outlines + 2023 heights ----------------
    import ee
    project = os.environ.get("EARTHENGINE_PROJECT")
    if not project:
        sys.exit("EARTHENGINE_PROJECT is not set (.env)")
    ee.Initialize(project=project)
    region = ee.Geometry.Polygon([[list(c) for c in corners]], proj="EPSG:4326", geodesic=False)
    fc = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(region)
    raw = fc.getInfo()["features"]
    kept = [f for f in raw if f["properties"]["confidence"] >= MIN_CONFIDENCE]

    nx = ny = int(2 * HALF / HEIGHT_RES)
    grid = {"dimensions": {"width": nx, "height": ny},
            "affineTransform": {"scaleX": HEIGHT_RES, "shearX": 0, "translateX": west,
                                "shearY": 0, "scaleY": -HEIGHT_RES, "translateY": north},
            "crsCode": UTM}
    col = (ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
           .filterBounds(region).filterDate("2023-01-01", "2024-01-01"))
    n_images = col.size().getInfo()
    img = col.mosaic().setDefaultProjection(col.first().projection()).select(["building_height", "building_presence"])
    arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
    height = np.asarray(arr["building_height"], dtype=np.float32)
    presence = np.asarray(arr["building_presence"], dtype=np.float32)
    valid = presence > PRESENCE_MIN

    affine_local = from_origin(-HALF, HALF, HEIGHT_RES, HEIGHT_RES)

    blds, fallback_idx = [], []
    for f in kept:
        g = clip_local.intersection(local(shape(f["geometry"])).buffer(0))
        for part in getattr(g, "geoms", [g]):
            if not isinstance(part, Polygon) or part.area < 4:
                continue
            mask = features.geometry_mask([mapping(part)], (ny, nx), affine_local, invert=True, all_touched=False)
            vals = height[mask & valid]
            h = float(np.median(vals)) if (vals.size >= MIN_PIXELS and part.area >= MIN_AREA) else None
            if h is None:
                fallback_idx.append(len(blds))
            simple = part.simplify(SIMPLIFY, preserve_topology=True)
            if not isinstance(simple, Polygon) or simple.is_empty:
                simple = part
            ring = Polygon(simple.exterior).exterior  # ponytail: courtyards (holes) dropped
            if ring.is_ccw is False:
                ring = LineString(list(ring.coords)[::-1])
            blds.append({"h": h, "p": flat_coords(ring)[:-2], "area": part.area, "geom": part,
                         "c": (r1(part.centroid.x), r1(part.centroid.y)), "conf": f["properties"]["confidence"]})
    measured = [b["h"] for b in blds if b["h"] is not None]
    fallback_h = float(np.median(measured))
    for i in fallback_idx:
        blds[i]["h"] = fallback_h
    for b in blds:
        b["h"] = r1(max(b["h"], 2.0))

    # landmark labels: anchored over the tallest model building inside the OSM outline
    place_labels = []
    for label, poly in landmarks.items():
        inside = [b for b in blds if b["geom"].intersects(poly)] or \
            [b for b in blds if b["geom"].distance(poly.centroid) < 20]  # outline offsets between OSM and Open Buildings
        top = max((b["h"] for b in inside), default=0.0)
        anchor = max(inside, key=lambda b: b["h"])["geom"].representative_point() if inside else poly.representative_point()
        place_labels.append([label, r1(anchor.x), r1(anchor.y), top])

    # ---------------- Ground: FABDEM ----------------
    from backend.dem import fabdem
    pad = 30.0
    dem = fabdem.fetch_grid(UTM, west - pad, south - pad, east + pad, north + pad, 10.0)
    z = dem["elevation"]
    p5, p50, p95 = np.nanpercentile(z, [5, 50, 95])
    relief = float(p95 - p5)
    if relief < FLAT_RELIEF:
        ground = {"kind": "flat", "median": r1(p50), "relief_p95_p5": r1(relief)}
    else:
        step = 10.0
        ground = {"kind": "fabdem", "median": r1(p50), "relief_p95_p5": r1(relief), "res": step,
                  "x0": -HALF - pad, "y0": HALF + pad, "nx": int(dem["width"]), "ny": int(dem["height"]),
                  "z": [r1(v - p50) for v in np.nan_to_num(z, nan=p50).ravel()]}  # rows north->south, relative to median

    out = {
        "name": "Bengaluru: UB City and Vittal Mallya Road",
        "bbox_wgs84": {"south": round(bbox_ll[0], 5), "west": round(bbox_ll[1], 5),
                       "north": round(bbox_ll[2], 5), "east": round(bbox_ll[3], 5)},
        "size_m": 2 * HALF,
        "sources": {"outlines": "Google Open Buildings v3", "heights": "Google Open Buildings 2.5D Temporal v1, 2023",
"roads_labels": "OpenStreetMap",
                    "ground": "FABDEM" if ground["kind"] == "fabdem" else "flat (FABDEM median)"},
        "buildings": [[b["h"], b["p"]] for b in blds],
        "roads": [[r["w"], r["p"]] for r in roads],
        "labels": {"places": place_labels, "roads": road_labels},  # [name, x, y, top_m] / [name, x, y, dx, dy, rank]
        "ground": ground,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))

    # ---------------- report ----------------
    print(f"bbox S,W,N,E = {bbox_ll[0]:.5f}, {bbox_ll[1]:.5f}, {bbox_ll[2]:.5f}, {bbox_ll[3]:.5f}  ({2*HALF:.0f} m square, {UTM})")
    print(f"OSM: {len(roads)} road pieces; road labels {[r[0] for r in road_labels]}; longest named runs (m) {road_lengths}")
    print(f"     landmarks {[(p[0], p[3]) for p in place_labels]}; missing {set(LANDMARKS.values()) - set(landmarks)}")
    print(f"Open Buildings: {len(raw)} outlines in box, {len(kept)} with confidence >= {MIN_CONFIDENCE} -> {len(blds)} clipped parts")
    print(f"2023 temporal images: {n_images}; fallback (area < {MIN_AREA} m2 or < {MIN_PIXELS} building px) = "
          f"{len(fallback_idx)} buildings set to box median {fallback_h:.1f} m")
    print(f"ground: {ground['kind']}  median {p50:.1f} m, relief p95-p5 {relief:.1f} m")
    print(f"wrote {OUT.relative_to(ROOT)}  {OUT.stat().st_size/1024:.1f} KiB, {len(blds)} buildings")
    print("15 tallest:")
    for b in sorted(blds, key=lambda b: -b["h"])[:15]:
        lon, lat = to_ll.transform(b["c"][0] + cx, b["c"][1] + cy)
        print(f"  {b['h']:6.1f} m  area {b['area']:7.0f} m2  conf {b['conf']:.2f}  {lat:.5f} N {lon:.5f} E")


if __name__ == "__main__":
    main()
