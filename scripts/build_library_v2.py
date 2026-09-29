#!/usr/bin/env python3
"""Curated tile library v2 (2026-09-29): packs, images, catalog and tile_manifest.json in ONE new folder.

Everything is written under --root (default data/library_v2_2026-09-29). Nothing outside it is written:
data/library/, the Sentinel-2 RGBs, data/display_test_*, the DFC2019 packs and the Maxar packs are
only read. tile_manifest.json is the single source of truth for curation (stars, inclusion, order,
default exaggeration); the backend listing (backend/library/tile_manifest.py), the frontend default
exaggeration (via the generate response) and scripts/stage_container_tiles.py all read it.

Steps (run in this order; `all` runs them all):
  seed     tile_manifest.json from the user's QA ratings (_input_ratings_<date>.json). Refuses to
           overwrite an existing manifest unless --force: after seeding, edit the manifest, not the ratings.
  packs    dem/: Sentinel-2 packs copied byte-identical from data/library/dem, except tiles whose
           calibration_recipe is "offset_only_b1" (rebuilt: h = FABDEM + a, a = median(ICESat-2 - FABDEM));
           DFC2019 packs copied from data/dfc2019/terrain_packs/packs; Maxar packs = the active DISPLAY
           preset (scripts/build_maxar_display_packs.py; built if missing).
  images   previews/ + thumbnails/: Sentinel-2 = the approved B_own_percentiles renders
           (data/display_test_2026-09-29/B_all_32); Darjeeling (not in B_all_32) gets a B render made
           here with the identical stretch, and its active image follows the manifest's texture_variant.
           Maxar and DFC2019 images are copied unchanged.
  catalog  manifest.json (the app's bundle-mode catalog: data/library/manifest.json's items + a `tile`
           entry) and tiles/ (symlinks to the original RGB rasters; never copied, never modified).
  dfc      DFC2019 flat-render rule and the 10x-16x exaggeration mapping (docs/library_v2.md) ->
           include_in_container / default_exaggeration for dfc2019 entries; tables in _analysis/.
  depth    bakes DAv2 relative depth (the app's own Space call, on the exact shipped preview) into each included
           tile's pack, so opening a library tile makes no Space call. Re-run after packs/--activate (cached: no calls).
  exag     bakes the app's automatic exaggeration (frontend/src/terrain.js:336-344, reproduced exactly
           below) where the manifest asks for it; reports every tile's slider range.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-09-29"
LIB = ROOT / "data/library"
B_DIR = ROOT / "data/display_test_2026-09-29/B_all_32"
DFC_PACKS = ROOT / "data/dfc2019/terrain_packs/packs"
SAMPLES = ROOT / "data/sentinel2_benchmark/terrain_rf_residual/samples.parquet"
PROTECTED = [LIB, B_DIR.parent, DFC_PACKS, ROOT / "data/sentinel2_benchmark", ROOT / "data/dfc2019/raw"]

STAR_TIER = {3: 0, 2: 1, 1: 2}
MAXAR_NOTE = "3D shape is a cosmetic display surface; statistics show real elevations."


def _guard(root: Path, path: Path) -> Path:
    """Every write goes through here: inside --root, never inside a protected folder."""
    p = path.parent.resolve() / path.name  # the link itself, never a symlink's target (tiles/ links to the RGBs)
    assert root.resolve() in p.parents or p == root.resolve(), f"refusing to write outside {root}: {p}"
    for q in PROTECTED:
        assert q.resolve() not in p.parents, f"refusing to write into protected {q}: {p}"
    return path


def _catalog() -> list[dict]:
    return json.loads((LIB / "manifest.json").read_text())["items"]


def _load(root: Path) -> dict:
    return json.loads((root / "tile_manifest.json").read_text())


def _save(root: Path, man: dict) -> None:
    _guard(root, root / "tile_manifest.json").write_text(json.dumps(man, indent=1) + "\n")


# ----------------------------------------------------------------------------- seed
def cmd_seed(root: Path, force: bool) -> None:
    if (root / "tile_manifest.json").exists() and not force:
        sys.exit("tile_manifest.json exists: edit it, or pass --force to reseed from the ratings (loses edits)")
    ratings = json.loads((root / f"_input_ratings_{DATE}.json").read_text())
    cat = _catalog()
    by_tile = {i["tile_id"]: i for i in cat if i["collection"] == "sentinel2"}
    rated = [r[0] for r in ratings["sentinel2"]]
    missing = sorted(set(by_tile) - set(rated))
    unknown = sorted(set(rated) - set(by_tile))
    assert not missing and not unknown, f"ratings vs catalog: missing {missing}, unknown {unknown}"
    cat_pos = {i["id"]: n for n, i in enumerate(cat)}
    tiles = []
    for tid, stars, exag, remark in ratings["sentinel2"]:
        it = by_tile[tid]
        recalc = remark == "RECALCULATE"
        include = not (stars == 1 and not recalc)
        if exag is not None:
            origin = "user"
        elif recalc:
            origin = "auto_after_recalc"
        else:
            origin = "baked_from_current_auto"
        recipe = ("offset_only_b1" if recalc else
                  "fabdem_terrain_glo30_surface (no ICESat-2 samples)" if tid == "darjeeling" else "ols_a_plus_b_fabdem")
        notes = []
        if recalc:
            notes.append("RECALCULATE: offset-only calibration (b = 1); needs re-rating")
        if not include:
            notes.append("excluded: 1 star without RECALCULATE (files stay on disk)")
        if tid == "darjeeling":
            notes.append("colour choice pending: original (rated 3 stars) vs generated B_own_percentiles render")
        tiles.append({"tile_id": it["id"], "display_name": it["title"], "source": "sentinel2", "stars": stars,
                      "include_in_container": include, "order_index": None,
                      "default_exaggeration": exag, "exaggeration_origin": origin, "calibration_recipe": recipe,
                      "needs_rerating": recalc, "texture_variant": "original" if tid == "darjeeling" else "B_own_percentiles",
                      "notes": "; ".join(notes)})
    for it in cat:
        if it["collection"] == "vhr":
            tiles.append({"tile_id": it["id"], "display_name": it["title"], "source": "maxar", "stars": None,
                          "include_in_container": True, "order_index": None, "default_exaggeration": None,
                          "exaggeration_origin": "auto_after_recalc",
                          "calibration_recipe": "FABDEM + Method 6 AGL; DISPLAY band (scripts/build_maxar_display_packs.py)",
                          "needs_rerating": False, "texture_variant": "original", "notes": MAXAR_NOTE})
        elif it["collection"] == "dfc2019":
            tiles.append({"tile_id": it["id"], "display_name": it["title"], "source": "dfc2019", "stars": None,
                          "include_in_container": True, "order_index": None, "default_exaggeration": None,
                          "exaggeration_origin": None,
                          "calibration_recipe": "ground (3DEP 1 m or city-level) + DFC2019 lidar AGL",
                          "needs_rerating": False, "texture_variant": "original", "notes": ""})
    _order(tiles, cat_pos)
    man = {"version": f"library_v2_{DATE}", "about": __doc__.splitlines()[0],
           "fields": {"order_index": "global listing order (Sentinel-2 by star tier, then Maxar, then DFC2019)",
                      "default_exaggeration": "slider value applied after generation; null = the app's auto value",
                      "maxar_display_preset": "active DISPLAY preset (scripts/build_maxar_display_packs.py --activate)"},
           "maxar_display_preset": "subtle", "tiles": tiles}
    _save(root, man)
    print(f"seeded {len(tiles)} tiles -> {root / 'tile_manifest.json'}")


def _order(tiles: list[dict], cat_pos: dict) -> None:
    """Sentinel-2: 3 stars, 2 stars, then 1-star tiles pending re-rating; catalog order inside a tier.
    Then Maxar, then DFC2019, each in catalog order. Excluded tiles keep an index (listing skips them)."""
    def key(t):
        src = {"sentinel2": 0, "maxar": 1, "dfc2019": 2}[t["source"]]
        tier = STAR_TIER.get(t["stars"], 3) if t["source"] == "sentinel2" else 0
        return (src, not t["include_in_container"], tier, cat_pos[t["tile_id"]])
    for n, t in enumerate(sorted(tiles, key=key)):
        t["order_index"] = n


# ----------------------------------------------------------------------------- packs
def _offset_only_pack(root: Path, item: dict) -> dict:
    """desktop/tiles/build_dem_pack.py's Sentinel-2 recipe with the OLS fit replaced by an offset:
    a = median(h_ref - FABDEM) over the same ICESat-2 ground samples (h_ref = per-photon EGM2008
    orthometric height, i.e. ellipsoidal height with its own per-point geoid correction), b = 1."""
    import pandas as pd
    import rasterio
    from rasterio.fill import fillnodata
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject
    t = item["tile_id"]
    s = pd.read_parquet(SAMPLES, columns=["tile_id", "height", "h_ref", "fabdem"])
    s = s[s.tile_id == t]
    a = float(np.median(s.h_ref.values - s.fabdem.values))
    b_old, a_old = np.polyfit(s.fabdem.values, s.h_ref.values, 1)
    with rasterio.open(os.path.realpath(ROOT / item["file"])) as r:
        crs, bounds, tile_tf, tile_shape = r.crs, r.bounds, r.transform, r.shape
    size = 334
    tf = from_bounds(*bounds, size, size)
    fab = np.load(ROOT / f"data/sentinel2_benchmark/fabdem/{t}_fabdem.npy").astype(np.float32)
    assert fab.shape == tile_shape
    dst = np.full((size, size), np.nan, np.float32)
    reproject(fab + np.float32(a), dst, src_transform=tile_tf, src_crs=crs, dst_transform=tf, dst_crs=crs,
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    assert np.isfinite(dst).mean() > 0.95
    if not np.isfinite(dst).all():
        dst = fillnodata(dst, mask=np.isfinite(dst).astype(np.uint8), max_search_distance=100)
    out = _guard(root, root / "dem" / f"{item['id']}.tif")
    with rasterio.open(out, "w", driver="GTiff", width=size, height=size, count=2, dtype="float32", crs=crs,
                       transform=tf, nodata=np.nan, compress="deflate", predictor=3, zlevel=9) as w:
        w.write(dst, 1)
        w.write(dst, 2)
        w.set_band_description(1, "TERRAIN")
        w.set_band_description(2, "SURFACE")
        src = "FABDEM v1-2 offset-calibrated to ICESat-2 ground (per-tile median offset, slope fixed at 1)"
        w.update_tags(TERRAIN_SOURCE=src, SURFACE_SOURCE=src,
                      CALIBRATION=f"h = FABDEM + {a:.4f} (median of {len(s)} ICESat-2 ground samples; b = 1)")
    with rasterio.open(LIB / "dem" / f"{item['id']}.tif") as r:
        old = r.read(2)
    geoid = s.h_ref.values - s.height.values
    return {"old_a": float(a_old), "old_b": float(b_old), "new_a": a, "new_b": 1.0, "n_samples": int(len(s)),
            "geoid_term_m": {"median": float(np.median(geoid)), "std": float(np.std(geoid))},
            "relief_before_m": float(np.nanmax(old) - np.nanmin(old)), "relief_after_m": float(np.nanmax(dst) - np.nanmin(dst)),
            "range_before_m": [float(np.nanmin(old)), float(np.nanmax(old))], "range_after_m": [float(np.nanmin(dst)), float(np.nanmax(dst))]}


def cmd_packs(root: Path) -> None:
    man = _load(root)
    cat = {i["id"]: i for i in _catalog()}
    (root / "dem").mkdir(exist_ok=True)
    report = {}
    for t in man["tiles"]:
        iid = t["tile_id"]
        if t["source"] == "sentinel2":
            if t["calibration_recipe"] == "offset_only_b1":
                report[iid] = _offset_only_pack(root, cat[iid])
                print(iid, "offset-only", {k: report[iid][k] for k in ("old_a", "old_b", "new_a", "relief_before_m", "relief_after_m")})
            else:
                shutil.copyfile(LIB / "dem" / f"{iid}.tif", _guard(root, root / "dem" / f"{iid}.tif"))
        elif t["source"] == "dfc2019":
            shutil.copyfile(DFC_PACKS / f"{iid}.tif", _guard(root, root / "dem" / f"{iid}.tif"))
    preset = man.get("maxar_display_preset", "subtle")
    maxar = ROOT / "scripts/build_maxar_display_packs.py"
    if not (root / "dem_maxar" / preset).is_dir():
        os.system(f'"{sys.executable}" "{maxar}" --out "{root}" --all-presets')
    os.system(f'"{sys.executable}" "{maxar}" --out "{root}" --activate {preset}')
    if report:
        _guard(root, root / "_analysis").mkdir(exist_ok=True)
        (root / "_analysis" / "s2_recalc_report.json").write_text(json.dumps(report, indent=1))
    print(f"packs -> {root / 'dem'} ({len(list((root / 'dem').glob('*.tif')))} files)")


# ----------------------------------------------------------------------------- images
def _b_render(raster: Path) -> "Image.Image":
    """The B_own_percentiles render, bit-identical to B_all_32 (checked on bathinda: max |diff| 0):
    per-band 2-98 % bounds on the full-resolution raster, the 1024 px preview read with rasterio's
    out_shape (scripts/library_catalog.py _images), stretched with those bounds."""
    import rasterio
    from PIL import Image
    with rasterio.open(raster) as src:
        full = src.read([1, 2, 3]).astype(np.float32)
        scale = max(src.width, src.height) / 1024
        h, w = max(1, round(src.height / scale)), max(1, round(src.width / scale))
        small = src.read([1, 2, 3], out_shape=(3, h, w)).astype(np.float32)
    out = np.zeros((h, w, 3), np.uint8)
    for i in range(3):
        lo, hi = np.percentile(full[i], [2, 98])
        out[..., i] = np.clip((small[i] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _save_pair(root: Path, img, name: str, sub: str = "") -> None:
    prev = _guard(root, root / "previews" / sub / name)
    prev.parent.mkdir(parents=True, exist_ok=True)
    img.save(prev, quality=88)
    th = img.copy()
    th.thumbnail((256, 256))
    thp = _guard(root, root / "thumbnails" / sub / name)
    thp.parent.mkdir(parents=True, exist_ok=True)
    th.save(thp, quality=82)


def cmd_images(root: Path) -> None:
    from PIL import Image
    man = _load(root)
    cat = {i["id"]: i for i in _catalog()}
    for d in ("previews", "thumbnails"):
        (root / d).mkdir(exist_ok=True)
    # self-check: the re-implemented B stretch must reproduce B_all_32 exactly
    chk = np.asarray(_b_render(Path(os.path.realpath(ROOT / cat["sentinel2-bathinda"]["file"]))), int)
    import io
    buf = io.BytesIO()
    Image.fromarray(chk.astype(np.uint8)).save(buf, "JPEG", quality=88)
    ref = np.asarray(Image.open(B_DIR / "previews/sentinel2-bathinda.jpg"), int)
    assert np.abs(np.asarray(Image.open(buf), int) - ref).max() == 0, "B stretch no longer reproduces B_all_32"
    for t in man["tiles"]:
        iid, it = t["tile_id"], cat[t["tile_id"]]
        name = it["preview"]
        if t["source"] == "sentinel2" and (B_DIR / "previews" / name).is_file():
            for d in ("previews", "thumbnails"):
                shutil.copyfile(B_DIR / d / name, _guard(root, root / d / name))
        elif t["source"] == "sentinel2":  # Darjeeling: no B render exists; make one, keep the original too
            _save_pair(root, _b_render(Path(os.path.realpath(ROOT / it["file"]))), name.replace(".jpg", "__B_own_percentiles.jpg"), "_variants")
            for d in ("previews", "thumbnails"):
                shutil.copyfile(LIB / d / name, _guard(root, root / d / "_variants" / name.replace(".jpg", "__original.jpg")))
            variant = t.get("texture_variant", "original")
            for d in ("previews", "thumbnails"):
                shutil.copyfile(root / d / "_variants" / name.replace(".jpg", f"__{variant}.jpg"), _guard(root, root / d / name))
            print(iid, "texture variant:", variant)
        else:
            for d in ("previews", "thumbnails"):
                shutil.copyfile(LIB / d / name, _guard(root, root / d / name))
    print(f"images -> {root / 'previews'}, {root / 'thumbnails'}")


# ----------------------------------------------------------------------------- catalog
def cmd_catalog(root: Path) -> None:
    src = json.loads((LIB / "manifest.json").read_text())
    (root / "tiles").mkdir(exist_ok=True)
    for it in src["items"]:
        link = _guard(root, root / "tiles" / f"{it['id']}.tif")
        if link.is_symlink():
            link.unlink()
        link.symlink_to(Path(os.path.realpath(ROOT / it["file"])))
        it["tile"] = link.name
        it["bundled"] = True
    src["generator"] = f"{src.get('generator')} | scripts/build_library_v2.py catalog ({DATE})"
    _guard(root, root / "manifest.json").write_text(json.dumps(src, indent=1))
    print(f"catalog -> {root / 'manifest.json'} ({len(src['items'])} items)")


# ----------------------------------------------------------------------------- app terrain reproduction
def _backend(root: Path):
    os.environ.update(DW2_NO_DOTENV="1", DW2_LIBRARY="bundle", DW2_LIBRARY_BUNDLE=str(root))
    sys.path.insert(0, str(ROOT))
    from backend.generation import pipeline
    from backend.library import catalog
    from backend.terrain import mesh_export
    return pipeline, catalog, mesh_export


def app_terrain(root: Path, item_id: str) -> dict:
    """terrain.json exactly as backend/generation/pipeline.py writes it for this item (mesh = DISPLAY
    when the pack has one, else SURFACE), plus the real SURFACE mesh grid."""
    pipeline, catalog, mesh_export = _backend(root)
    from rasterio.warp import transform_bounds
    e = pipeline._library_elevation(catalog.get(item_id))
    s = np.asarray(e["surface"], np.float32)
    ll = transform_bounds(e["crs"], "EPSG:4326", *e["bounds"], densify_pts=21)
    f = tempfile.mktemp(suffix=".json")
    mesh_export.write_terrain_json(f, s, ll, pipeline._mesh_hw(s.shape), display=e.get("display"))
    d = json.loads(Path(f).read_text())
    os.remove(f)
    return d


def auto_exaggeration(d: dict) -> dict:
    """frontend/src/terrain.js createTerrain: exaggerationFactor (the slider's auto value, :336-344) and the
    slider range (:821-827). Uses the mesh source: terrain.json's `display` when present."""
    src = d.get("display") or d
    b = d["bounds"]
    gw, gh = b["east"] - b["west"], b["north"] - b["south"]
    rng = src["elevationMax"] - src["elevationMin"]
    lat = (b["north"] + b["south"]) / 2 * math.pi / 180
    fp = math.sqrt(gw * 111320 * math.cos(lat) * gh * 111320)
    ratio = rng / fp
    floor = (0.15 * 100) / (rng * 0.02) if rng > 0 else 1
    auto = min(60, max(1, min(10, 1 + 4 * math.log10(0.2 / ratio)) if ratio > 0 else 10, floor))
    smax = max(auto * 1.5, min(50, 0.8 * 100 / max(rng * 0.02, 1e-6)))
    return {"auto": auto, "slider_min": min(0.25, auto), "slider_max": smax, "mesh_range_m": rng}


# ----------------------------------------------------------------------------- dfc
def cmd_dfc(root: Path) -> None:
    """Flat-render rule and the 10x-16x mapping. Both are measured on the RENDERED surface: the mesh grid
    of terrain.json (heights above the tile minimum, metres)."""
    from scipy.stats import spearmanr
    man = _load(root)
    ratings = json.loads((root / f"_input_ratings_{DATE}.json").read_text())
    anchors = ratings["dfc2019_anchors"]
    rows = {}
    for t in man["tiles"]:
        if t["source"] != "dfc2019":
            continue
        d = app_terrain(root, t["tile_id"])
        rng = d["elevationMax"] - d["elevationMin"]
        z = np.array(d["heights"]) * rng
        p2, p98 = np.percentile(z, [2, 98])
        q3 = float(np.percentile(z, 75))
        rows[t["tile_id"].split("-", 1)[1]] = {"mesh_range_m": rng, "q3_above_min_m": q3, "p98_p2_m": float(p98 - p2),
                                               "pct_cells_gt2m": float((z > 2).mean() * 100), **auto_exaggeration(d)}
    # Rule (a): the app draws a tile flat when at least 75 % of its mesh cells sit on the ground,
    # i.e. Q3 of the heights above the tile minimum < FLAT_Q3_M (frontend/src/outlier-relief.js: IQR -> cm).
    FLAT_Q3_M = 0.05
    TRUE_FLAT_RANGE_M = 2.0     # and it is genuinely flat data when the whole tile spans < 2 m
    flat = sorted(k for k, r in rows.items() if r["q3_above_min_m"] < FLAT_Q3_M)
    caught_max = max(rows[k]["q3_above_min_m"] for k in flat)
    rest_min = min(r["q3_above_min_m"] for k, r in rows.items() if k not in flat)
    # Rule (b): log(exag) = c - alpha * log(p98 - p2), fitted to the 5 anchors, clamped to [10, 16], 0.5x steps.
    ak = list(anchors)
    x = np.array([rows[k]["p98_p2_m"] for k in ak])
    y = np.array([anchors[k] for k in ak], float)
    slope, c = np.polyfit(np.log(x), np.log(y), 1)
    rho = float(spearmanr(x, y).correlation)
    table = {}
    for k, r in rows.items():
        pred = float(np.exp(c + slope * np.log(r["p98_p2_m"]))) if r["p98_p2_m"] > 0 else 16.0
        final = float(anchors[k]) if k in anchors else float(np.clip(round(pred * 2) / 2, 10, 16))
        true_flat = k in flat and r["mesh_range_m"] < TRUE_FLAT_RANGE_M
        table[k] = {**{kk: round(v, 4) for kk, v in r.items()}, "predicted": round(pred, 3), "final": final,
                    "flat_render": k in flat, "true_flat_data": true_flat, "anchor": k in anchors}
    for t in man["tiles"]:
        if t["source"] != "dfc2019":
            continue
        k = t["tile_id"].split("-", 1)[1]
        r = table[k]
        if t["exaggeration_origin"] == "user" and not r["anchor"]:
            continue  # the owner's own value and notes (set after this rule ran, e.g. 2026-09-30) are kept
        t["default_exaggeration"] = r["final"]
        t["exaggeration_origin"] = "user" if r["anchor"] else "derived_rule"
        notes = []
        if r["true_flat_data"]:
            t["include_in_container"] = False
            notes.append(f"excluded: genuinely flat data (tile spans {r['mesh_range_m']:.2f} m); files stay")
        elif r["flat_render"]:
            notes.append("was drawn flat by the viewer's outlier limiter (Q3 < 5 cm), not flat data; fixed 2026-09-30: "
                         "DFC2019 terrain.json carries limitOutliers: false")
        if r["anchor"]:
            notes.append("exaggeration anchor (user)")
        t["notes"] = "; ".join(notes)
    _order(man["tiles"], {i["id"]: n for n, i in enumerate(_catalog())})
    _save(root, man)
    rep = {"flat_rule": {"statistic": "Q3 of rendered heights above the tile minimum", "threshold_m": FLAT_Q3_M,
                         "largest_caught_q3_m": caught_max, "smallest_not_caught_q3_m": rest_min,
                         "margin_ratio": rest_min / caught_max, "true_flat_range_m": TRUE_FLAT_RANGE_M, "flat": flat},
           "mapping": {"statistic": "p98 - p2 of the rendered surface (m)", "spearman_vs_anchor_exag": rho,
                       "alpha": -slope, "c": c, "clamp": [10, 16], "step": 0.5, "anchors": anchors},
           "tiles": table}
    _guard(root, root / "_analysis").mkdir(exist_ok=True)
    (root / "_analysis" / "dfc_rules.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "tiles"}, indent=1))


# ----------------------------------------------------------------------------- depth
def cmd_depth(root: Path) -> None:
    """Bake DAv2 relative depth into each included tile's pack (and both Darjeeling colour variants), so opening a
    library tile makes no Space call (backend/generation/pipeline.py baked_depth).

    The input is the exact preview image the tile ships with; the call is the app's own
    (backend/api/depth_routes.py _forward, same filename and bytes), so the stored payload is what the app would
    receive. Each response is cached under _analysis/depth_cache/<preview sha256>.json: a rebuild (new packs,
    another Maxar preset) re-bakes without new Space calls. Stored per pack in the DAV2_DEPTH metadata domain as
    SHA256_<preview sha256> = the response JSON (u16+zlib payload) + model, Space id + revision, preview sha, date.
    Maxar: written into dem/ and every dem_maxar/<preset>/ pack, so --activate keeps it."""
    import asyncio
    import datetime
    import hashlib
    import rasterio
    pipeline, catalog, _ = _backend(root)
    from backend.api import depth_routes
    from huggingface_hub import HfApi
    man = _load(root)
    cat = {i["id"]: i for i in _catalog()}
    cache = _guard(root, root / "_analysis" / "depth_cache")
    cache.mkdir(parents=True, exist_ok=True)
    sid = depth_routes.space_id(depth_routes.hosts()[0])
    assert sid, "the primary depth host is not a Space"
    rev = HfApi().space_info(sid, token=depth_routes.hf_token()).sha
    jobs = []
    for t in man["tiles"]:
        if not t["include_in_container"]:
            continue
        name = cat[t["tile_id"]]["preview"]
        jobs.append((t["tile_id"], root / "previews" / name))
        if t["tile_id"] == "sentinel2-darjeeling":
            jobs += [(t["tile_id"], p) for p in sorted((root / "previews" / "_variants").glob(name.replace(".jpg", "__*.jpg")))]
    calls, report = 0, {}
    today = datetime.date.today().isoformat()
    for iid, prev in jobs:
        body = prev.read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        c = cache / f"{sha}.json"
        if c.exists():
            resp = json.loads(c.read_text())
        else:
            resp = asyncio.run(depth_routes._forward(f"{iid}.jpg", body, "image/jpeg"))
            calls += 1
            resp = {**resp, "space": sid, "space_revision": rev, "baked_at": today}
            _guard(root, c).write_text(json.dumps(resp))
        entry = {**resp, "preview_sha256": sha, "preview_file": str(prev.relative_to(root))}
        packs = [root / "dem" / f"{iid}.tif"]
        if iid.startswith("vhr-"):
            packs += sorted((root / "dem_maxar").glob(f"*/{iid}.tif"))
        for p in packs:
            with rasterio.open(_guard(root, p), "r+") as w:
                w.update_tags(ns=pipeline.DEPTH_NS, **{f"SHA256_{sha}": json.dumps(entry)})
        report.setdefault(iid, []).append({"preview": entry["preview_file"], "sha256": sha, "shape": resp.get("shape"),
                                           "encoding": resp.get("encoding"), "model": resp.get("model"),
                                           "packs": [str(p.relative_to(root)) for p in packs]})
    (root / "_analysis" / "depth_bake_report.json").write_text(json.dumps(
        {"space": sid, "space_revision": rev, "space_calls_this_run": calls, "previews": len(jobs), "tiles": report}, indent=1))
    print(f"baked {len(jobs)} previews into {len(report)} tiles; Space calls this run: {calls} ({sid} @ {rev[:12]})")


# ----------------------------------------------------------------------------- exag
def cmd_exag(root: Path) -> None:
    man = _load(root)
    rep = {}
    for t in man["tiles"]:
        a = auto_exaggeration(app_terrain(root, t["tile_id"]))
        if t["exaggeration_origin"] == "baked_from_current_auto":
            t["default_exaggeration"] = round(a["auto"], 4)
        v = t["default_exaggeration"]
        rep[t["tile_id"]] = {**{k: round(x, 4) for k, x in a.items()}, "default": v, "origin": t["exaggeration_origin"],
                             "in_slider_range": None if v is None else a["slider_min"] <= v <= a["slider_max"]}
    _save(root, man)
    _guard(root, root / "_analysis").mkdir(exist_ok=True)
    (root / "_analysis" / "exaggeration_table.json").write_text(json.dumps(rep, indent=1))
    bad = {k: r for k, r in rep.items() if r["in_slider_range"] is False}
    print(f"exaggeration baked; {len(bad)} defaults outside their slider range: {sorted(bad)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["seed", "packs", "images", "catalog", "dfc", "exag", "depth", "all"])
    ap.add_argument("--root", type=Path, default=ROOT / f"data/library_v2_{DATE}")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    assert root != LIB.resolve() and root.name.startswith("library_v2_"), root
    root.mkdir(parents=True, exist_ok=True)
    steps = ["seed", "packs", "images", "catalog", "dfc", "exag", "depth"] if a.step == "all" else [a.step]
    for s in steps:
        {"seed": lambda: cmd_seed(root, a.force), "packs": lambda: cmd_packs(root), "images": lambda: cmd_images(root),
         "catalog": lambda: cmd_catalog(root), "dfc": lambda: cmd_dfc(root), "exag": lambda: cmd_exag(root),
         "depth": lambda: cmd_depth(root)}[s]()


if __name__ == "__main__":
    main()
