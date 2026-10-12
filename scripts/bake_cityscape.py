"""Bake a cosmetic city model: the Bengaluru urban model of the demo page (#/demo), and the close-up city tiles of
the Studio library. One box per scripts/city_tiles.json entry (centre, half side, UTM zone, sources):
- buildings ("openbuildings", the Indian tiles):
    outlines: Google Open Buildings v3 polygons (Earth Engine)
    heights:  Google Open Buildings 2.5D Temporal v1, 2023, band building_height,
              median over the outline's building pixels (building_presence > 0.5)
- buildings (a 3D city model, e.g. "plateau"): the GLB baked by scripts/bake_city_glb.mjs; its per-building roofs
  give the footprints and heights the tile's 2D layers are painted from (data/cityscape_src/<key>/footprints.json)
- roads, labels:     OpenStreetMap, one Overpass query for the box (road names, and the entry's landmarks)
- ground:            FABDEM (backend/dem/fabdem.py) or the GSI 5 m DEM (Japan, "gsi"),
                     flat at the median if the relief is negligible

Writes frontend/public/cityscape/<key>.json in local metres (x east, y north, origin = box centre).
Needs EARTHENGINE_PROJECT in .env and local Earth Engine credentials.

    .venv/bin/python scripts/bake_cityscape.py [key]      (default: bengaluru)
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
TILES = json.loads((ROOT / "scripts" / "city_tiles.json").read_text())
# Bengaluru's box centre: 150 m east and 150 m south of the UB City towers (OSM way 38773290, 12.9723 N 77.5959 E),
# so UB City sits in the north-west quarter; the densest 500 m box around it (394 outlines at confidence >= 0.70).

MIN_CONFIDENCE = 0.70  # Open Buildings v3 confidence; keeps 94% of the box's built area (0.75 keeps 79%)
PRESENCE_MIN = 0.5     # temporal building_presence above which a pixel counts as building
HEIGHT_RES = 2.0       # m, grid the 2023 heights are sampled on
MIN_AREA = 25.0        # m², smaller outlines get the fallback height
MIN_PIXELS = 3         # fewer valid building pixels -> fallback height
SIMPLIFY = 0.5         # m
FLAT_RELIEF = 3.0      # m, FABDEM p95-p5 below this -> flat ground
# landmarks: OSM name -> label, per tile (Bengaluru's checked against public listings: UB City's blocks are UB Tower,
# Concorde, Comet, Canberra)
LABEL_ROADS = ("primary", "secondary", "tertiary")  # named roads of these classes get a label
ROAD_WIDTH = {"trunk": 14, "primary": 12, "primary_link": 7, "secondary": 10, "secondary_link": 6,
              "tertiary": 8, "tertiary_link": 5, "residential": 6, "unclassified": 6, "service": 4,
              "living_street": 5, "pedestrian": 4, "footway": 2, "path": 2, "cycleway": 2}


def r1(v):
    return round(float(v), 1)


def flat_coords(geom):
    return [r1(c) for xy in list(geom.coords) for c in xy]


# OSM areas drawn on the ground (tiles with "areas": true): water (and the sea, from the coastline) and parks
AREA_TAGS = ('natural=water', 'waterway=riverbank', 'leisure=park', 'leisure=garden', 'landuse=grass',
             'landuse=recreation_ground')
WATER = {("natural", "water"), ("waterway", "riverbank")}


def osm_areas(osm, local, clip_local):
    """(water, parks): lists of shapely polygons in local metres, clipped to the box. Relations are assembled from
    their member ways (outer rings minus inner rings); the sea is the part of the box right of the coastline (OSM
    draws coastlines with the land on the left)."""
    from shapely.ops import polygonize, unary_union
    water, parks, coast = [], [], []
    for el in osm:
        tags = el.get("tags", {})
        kind = "water" if any((k, tags.get(k)) in WATER for k in ("natural", "waterway")) else \
            "park" if tags.get("leisure") in ("park", "garden") or tags.get("landuse") in ("grass", "recreation_ground") else \
            "coast" if tags.get("natural") == "coastline" else None
        if kind is None or "highway" in tags:
            continue
        if el["type"] == "way" and len(el.get("geometry", [])) >= 2:
            line = local(LineString([(p["lon"], p["lat"]) for p in el["geometry"]]))
            if kind == "coast":
                coast.append(line)
                continue
            if len(el["geometry"]) < 4 or not line.is_ring:
                continue
            poly = Polygon(line.coords).buffer(0)
        elif el["type"] == "relation":
            rings = {"outer": [], "inner": []}
            for m in el.get("members", []):
                if m.get("type") == "way" and m.get("geometry") and m.get("role") in rings:
                    rings[m["role"]].append(local(LineString([(p["lon"], p["lat"]) for p in m["geometry"]])))
            outer = unary_union(list(polygonize(linemerge(rings["outer"]) if rings["outer"] else [])))
            inner = unary_union(list(polygonize(linemerge(rings["inner"]) if rings["inner"] else [])))
            poly = outer.difference(inner) if not inner.is_empty else outer
        else:
            continue
        poly = clip_local.intersection(poly)
        (water if kind == "water" else parks).extend(p for p in getattr(poly, "geoms", [poly])
                                                     if isinstance(p, Polygon) and p.area > 20)
    if coast:
        lines = unary_union([clip_local.intersection(c) for c in coast] + [clip_local.exterior])
        for face in polygonize(lines):
            pt = face.representative_point()
            seg = min(coast, key=lambda c: c.distance(pt))
            d = seg.project(pt)
            a, b = seg.interpolate(max(d - 1, 0)), seg.interpolate(min(d + 1, seg.length))
            if (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x) < 0:  # right of the coastline: sea
                water.append(face)
    water_u = unary_union(water) if water else Polygon()
    parks = [p.difference(water_u) for p in parks]
    flat = lambda geoms: [g for x in geoms for g in getattr(x, "geoms", [x]) if isinstance(g, Polygon) and g.area > 20]
    return flat([water_u]) if water else [], flat(parks)


def rings_out(poly):
    """[exterior, holes...] as flat coordinate lists (exterior counter-clockwise), simplified."""
    from shapely.geometry.polygon import orient
    p = orient(poly.simplify(SIMPLIFY, preserve_topology=True), 1.0)
    return [flat_coords(p.exterior)[:-2]] + [flat_coords(r)[:-2] for r in p.interiors]


def gsi_grid(crs, left, bottom, right, top, res):
    """GSI elevation tiles (DEM5A, 5 m; metres above Tokyo Peil, NaN = no data) on a north-up grid in `crs`, the
    same dict as backend.dem.fabdem.fetch_grid. Tiles: https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/{z}/{x}/{y}.png"""
    import io
    import math
    from PIL import Image
    from rasterio.warp import Resampling, reproject
    z = 15
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = to_ll.transform([left, right, right, left], [bottom, bottom, top, top])
    tx = lambda lon: (lon + 180) / 360 * 2 ** z
    ty = lambda lat: (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** z
    x0, x1 = int(tx(min(lons))), int(tx(max(lons)))
    y0, y1 = int(ty(max(lats))), int(ty(min(lats)))
    mosaic = np.full(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.nan, np.float32)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            url = f"https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/{z}/{x}/{y}.png"
            req = urllib.request.Request(url, headers={"User-Agent": "depthwizard-cityscape-bake/1.0"})
            rgb = np.asarray(Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=60).read())).convert("RGB"), np.int64)
            v = rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]  # GSI encoding: 0.01 m steps, 2^23 = no data
            h = np.where(v < 2 ** 23, v, v - 2 ** 24).astype(np.float32) * 0.01
            h[v == 2 ** 23] = np.nan
            mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = h
    size = 2 * math.pi * 6378137 / 2 ** z / 256  # web-mercator metres per pixel at zoom z
    src_tr = from_origin(-math.pi * 6378137 + x0 * 256 * size, math.pi * 6378137 - y0 * 256 * size, size, size)
    width, height = math.ceil((right - left) / res), math.ceil((top - bottom) / res)
    elev = np.full((height, width), np.nan, np.float32)
    reproject(mosaic, elev, src_transform=src_tr, src_crs="EPSG:3857", src_nodata=np.nan,
              dst_transform=from_origin(left, top, res, res), dst_crs=crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    return {"elevation": elev, "transform": (res, 0.0, left, 0.0, -res, top), "crs": crs, "width": width, "height": height}


def main(key="bengaluru"):
    load_dotenv(ROOT / ".env")
    cfg = TILES[key]
    OUT = ROOT / "frontend" / "public" / "cityscape" / f"{key}.json"
    CENTRE_LAT, CENTRE_LON = cfg["centre"]
    HALF = float(cfg["half"])
    UTM = cfg["utm"]
    LANDMARKS = cfg["landmarks"]
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
    names = "|".join(LANDMARKS) or "^$"  # no landmarks: matches nothing
    bb = f"({s},{w},{n},{e})"
    areas = "".join(f"{t}[{k}]{bb};" for t in ("way", "rel") for k in AREA_TAGS) + f"way[natural=coastline]{bb};" \
        if cfg.get("areas") else ""
    q = f"[out:json][timeout:90];(way[highway]{bb};way[name~\"^({names})$\"]{bb};{areas});out geom;"
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
    water, parks = osm_areas(osm, local, clip_local) if cfg.get("areas") else ([], [])
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

    if cfg["buildings"] == "openbuildings":
        # ---------------- Earth Engine: outlines + 2023 heights ----------------
        import ee
        project = os.environ.get("EARTHENGINE_PROJECT")
        if not project:
            sys.exit("EARTHENGINE_PROJECT is not set (.env)")
        ee.Initialize(project=project)
        region = ee.Geometry.Polygon([[list(c) for c in corners]], proj="EPSG:4326", geodesic=False)
        fc = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(region)
        n_raw = fc.size().getInfo()
        k = int(np.ceil(np.sqrt(n_raw / 4000)))  # Earth Engine returns at most 5000 features per request
        if k == 1:
            raw = fc.getInfo()["features"]
            kept = [f for f in raw if f["properties"]["confidence"] >= MIN_CONFIDENCE]
        else:  # dense old cities: k x k cells, confidence filtered server-side, de-duplicated by plus code
            kept = {}
            for i in range(k):
                for j in range(k):
                    cell = ee.Geometry.Rectangle([min(lons) + (max(lons) - min(lons)) * i / k, min(lats) + (max(lats) - min(lats)) * j / k,
                                                  min(lons) + (max(lons) - min(lons)) * (i + 1) / k, min(lats) + (max(lats) - min(lats)) * (j + 1) / k],
                                                 "EPSG:4326", False)
                    sub = fc.filterBounds(cell).filter(ee.Filter.gte("confidence", MIN_CONFIDENCE)).getInfo()["features"]
                    kept.update((f["properties"]["full_plus_code"], f) for f in sub)
            kept, raw = list(kept.values()), range(n_raw)

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
    else:
        # ---------------- a 3D city model (scripts/bake_city_glb.mjs): footprint = union of the roof triangles
        from shapely.ops import unary_union
        src = json.loads((ROOT / "data" / "cityscape_src" / key / "buildings.json").read_text())
        if src["centre_utm"] != [cx, cy]:
            sys.exit(f"{key}: buildings.json was baked for another box; re-run scripts/bake_city_glb.mjs {key}")
        blds, fallback_idx, raw, kept, n_images = [], [], src["buildings"], src["buildings"], 0
        for sb in src["buildings"]:
            tris = [Polygon([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]) for t in sb["tris"]]
            g = unary_union([t.buffer(0.05) for t in tris if t.area > 0.01])
            for part in getattr(g, "geoms", [g]):
                if not isinstance(part, Polygon) or part.area < 4:
                    continue
                simple = part.simplify(SIMPLIFY, preserve_topology=True)
                ring = Polygon(simple.exterior).exterior
                if ring.is_ccw is False:
                    ring = LineString(list(ring.coords)[::-1])
                blds.append({"h": r1(sb["top"] - sb["base"]), "p": flat_coords(ring)[:-2], "area": part.area, "geom": part,
                             "c": (r1(part.centroid.x), r1(part.centroid.y)), "conf": None, "src": sb})

    # landmark labels: anchored over the tallest model building inside the OSM outline
    place_labels = []
    for label, poly in landmarks.items():
        inside = [b for b in blds if b["geom"].intersects(poly)] or \
            [b for b in blds if b["geom"].distance(poly.centroid) < 20]  # outline offsets between OSM and Open Buildings
        top = max((b["h"] for b in inside), default=0.0)
        anchor = max(inside, key=lambda b: b["h"])["geom"].representative_point() if inside else poly.representative_point()
        place_labels.append([label, r1(anchor.x), r1(anchor.y), top])

    # ---------------- Ground: FABDEM, or the GSI 5 m DEM ----------------
    pad = 30.0
    step = 5.0 if cfg["ground"] == "gsi" else 10.0
    if cfg["ground"] == "dataset":  # flat at the 3D model's own ground level: the median building base
        base = float(np.median([b["src"]["base"] for b in blds]))
        dem = {"elevation": np.full((2, 2), base, np.float32), "width": 2, "height": 2}
    elif cfg["ground"] == "model":  # the 3D model's building ground surfaces, interpolated (inverse distance, 8 nearest)
        pts = np.array([b["c"] for b in blds]); zb = np.array([b["src"]["base"] for b in blds])
        gx = np.arange(-HALF - pad + step / 2, HALF + pad, step)
        X, Y = np.meshgrid(gx, gx[::-1])
        d = np.hypot(X.ravel()[:, None] - pts[None, :, 0], Y.ravel()[:, None] - pts[None, :, 1])
        near = np.argsort(d, axis=1)[:, :8]
        w = 1 / np.maximum(np.take_along_axis(d, near, 1), 1.0) ** 2
        z = (w * zb[near]).sum(1) / w.sum(1)
        dem = {"elevation": z.reshape(X.shape).astype(np.float32), "width": X.shape[1], "height": X.shape[0]}
    elif cfg["ground"] == "gsi":
        dem = gsi_grid(UTM, west - pad, south - pad, east + pad, north + pad, step)
    else:
        from backend.dem import fabdem
        dem = fabdem.fetch_grid(UTM, west - pad, south - pad, east + pad, north + pad, step)
    z = dem["elevation"]
    p5, p50, p95 = np.nanpercentile(z, [5, 50, 95])
    relief = float(p95 - p5)
    if relief < FLAT_RELIEF:
        ground = {"kind": "flat", "median": r1(p50), "relief_p95_p5": r1(relief)}
    else:
        ground = {"kind": cfg["ground"], "median": r1(p50), "relief_p95_p5": r1(relief), "res": step,
                  "x0": -HALF - pad, "y0": HALF + pad, "nx": int(dem["width"]), "ny": int(dem["height"]),
                  "z": [r1(v - p50) for v in np.nan_to_num(z, nan=p50).ravel()]}  # rows north->south, relative to median

    ground_name = {"fabdem": "FABDEM", "gsi": "GSI DEM5A (5 m)"}.get(cfg["ground"]) or cfg["ground_name"]
    sources = {"outlines": "Google Open Buildings v3", "heights": "Google Open Buildings 2.5D Temporal v1, 2023",
               "roads_labels": "OpenStreetMap",
               "ground": ground_name if ground["kind"] != "flat" else f"flat ({ground_name} median)"}
    glb = None
    if cfg["buildings"] != "openbuildings":
        # the GLB's y is ellipsoidal height minus its reference; the model's y is height above the ground median. The
        # geoid separation is taken from the data: median over buildings of (base, ellipsoidal) - (ground, orthometric).
        zz = np.nan_to_num(z, nan=p50)
        def dem_at(x, y):
            c = int(np.clip((x - (-HALF - pad)) / step, 0, dem["width"] - 1))
            r = int(np.clip(((HALF + pad) - y) / step, 0, dem["height"] - 1))
            return float(zz[r, c])
        if cfg["ground"] in ("dataset", "model"):  # the model's own ground: no datum to bridge
            geoid = 0.0
            for b in blds:
                b["h"] = r1(max(b["src"]["top"] - p50 - (dem_at(*b["c"]) - p50 if cfg["ground"] == "model" else 0), 2.0))
        else:
            geoid = float(np.median([b["src"]["base"] - dem_at(*b["c"]) for b in blds]))
            for b in blds:  # height above the local ground (the lowest wall vertex can sit in a sunken plaza)
                b["h"] = r1(max(b["src"]["top"] - geoid - dem_at(*b["c"]), 2.0))
        glb = {"url": f"cityscape/{key}.glb", "y_offset": r1(src["ref"] - geoid - p50), "geoid_separation_m": r1(geoid)}
        sources = {"buildings": cfg["sources"]["buildings"], "roads_labels": "OpenStreetMap", "ground": sources["ground"]}
        foot = ROOT / "data" / "cityscape_src" / key / "footprints.json"
        foot.write_text(json.dumps([[b["h"], b["p"]] for b in blds], separators=(",", ":")))

    out = {
        "name": cfg["name"],
        "bbox_wgs84": {"south": round(bbox_ll[0], 5), "west": round(bbox_ll[1], 5),
                       "north": round(bbox_ll[2], 5), "east": round(bbox_ll[3], 5)},
        "size_m": 2 * HALF,
        "sources": sources,
        **({"glb": glb} if glb else {"buildings": [[b["h"], b["p"]] for b in blds]}),
        "roads": [[r["w"], r["p"]] for r in roads],
        **({"water": [rings_out(p) for p in water], "parks": [rings_out(p) for p in parks]} if cfg.get("areas") else {}),
        "labels": {"places": place_labels, "roads": road_labels},  # [name, x, y, top_m] / [name, x, y, dx, dy, rank]
        "ground": ground,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))

    # ---------------- report ----------------
    print(f"bbox S,W,N,E = {bbox_ll[0]:.5f}, {bbox_ll[1]:.5f}, {bbox_ll[2]:.5f}, {bbox_ll[3]:.5f}  ({2*HALF:.0f} m square, {UTM})")
    print(f"OSM: {len(roads)} road pieces; road labels {[r[0] for r in road_labels]}; longest named runs (m) {road_lengths}")
    print(f"     landmarks {[(p[0], p[3]) for p in place_labels]}; missing {set(LANDMARKS.values()) - set(landmarks)}")
    if glb:
        print(f"3D model: {len(raw)} buildings -> {len(blds)} footprint parts; GLB y offset {glb['y_offset']} m, "
              f"geoid separation from the data {glb['geoid_separation_m']} m")
    else:
        print(f"Open Buildings: {len(raw)} outlines in box, {len(kept)} with confidence >= {MIN_CONFIDENCE} -> {len(blds)} clipped parts")
        print(f"2023 temporal images: {n_images}; fallback (area < {MIN_AREA} m2 or < {MIN_PIXELS} building px) = "
              f"{len(fallback_idx)} buildings set to box median {fallback_h:.1f} m")
    print(f"ground: {ground['kind']}  median {p50:.1f} m, relief p95-p5 {relief:.1f} m")
    if cfg.get("areas"):
        print(f"OSM areas: water {len(water)} polygons {sum(p.area for p in water) / 1e4:.1f} ha, "
              f"parks {len(parks)} polygons {sum(p.area for p in parks) / 1e4:.1f} ha")
    print(f"wrote {OUT.relative_to(ROOT)}  {OUT.stat().st_size/1024:.1f} KiB, {len(blds)} buildings")
    print("15 tallest:")
    for b in sorted(blds, key=lambda b: -b["h"])[:15]:
        lon, lat = to_ll.transform(b["c"][0] + cx, b["c"][1] + cy)
        extra = (f"conf {b['conf']:.2f}" if b["conf"] is not None else
                 f"{b['src']['lod']} measuredHeight {b['src'].get('measuredHeight')} name {b['src'].get('name')}"
                 + (f" BIN {b['src']['bin']}" if b['src'].get('bin') else ""))
        print(f"  {b['h']:6.1f} m  area {b['area']:7.0f} m2  {extra}  {lat:.5f} N {lon:.5f} E")


if __name__ == "__main__":
    main(*sys.argv[1:2])
