#!/usr/bin/env python3
"""Curated "Choose from Library" catalog for the Page 1 input UI.

Two steps, so the manifest is always GENERATED from the curated folders,
never hand-written:

  curate  Materialise data/library/curated/{dfc2019,sentinel2,vhr}/ from the
          project's authoritative lists:
            dfc2019   the 50-tile DFC2019 benchmark (the set defined by
                      data/dfc2019/experiments/dav2_baseline/depth/*.npy),
                      symlinked to data/dfc2019/raw/RGB/Track1-RGB/
            sentinel2 the 32 Sentinel-2 benchmark tiles (manifest.csv), symlinked
            vhr       the 6 Maxar Sikkim/Darjeeling crops of vhr_dsm_pipeline.py,
                      cut from the source scenes with that script's own crop
                      definitions and written as georeferenced GeoTIFFs
          Everything else (Landsat / CBERS / L1C benchmark tiles, resolution
          tests, other research artifacts) is deliberately NOT curated.
  build   Scan ONLY the curated folders; read each file's real metadata; write
          a thumbnail (256 px JPEG) and a preview (1024 px) per item and
          data/library/manifest.json. Removes stale images. Re-run after any
          change to the curated folders.

Routing (the processing tier each item will run), decided here and enforced by
the backend's select endpoint:
  Tier 1 = DEM only (FABDEM terrain; no height prediction)
  Tier 2 = real height prediction (Method 6 above-ground height), on a DEM
           when the image is georeferenced
  sentinel2 -> Tier 1, LOCKED: height prediction was shown not to work at 10 m
               (docs/method-audit, Sentinel-2 track), so no UI toggle can
               request Tier 2 for these.
  dfc2019   -> Tier 2. These tiles carry no georeference, so no DEM can be
               fetched: the output is above-ground height only, not absolute
               elevation.
  vhr       -> Tier 2, on FABDEM (georeferenced 0.3 m Maxar crops).

  .venv/bin/python scripts/library_catalog.py curate
  .venv/bin/python scripts/library_catalog.py build
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data/library"
CURATED = LIB / "curated"
THUMBS = LIB / "thumbnails"
PREVIEWS = LIB / "previews"
MANIFEST = LIB / "manifest.json"

SOURCES = ("dfc2019", "sentinel2", "vhr")
# Paths that must never end up in the demo catalog (research artifacts).
EXCLUDED_MARKERS = ("landsat", "cbers", "l1c", "brazil", "token_grid", "resolution_transfer", "gamus")

DFC_CITY = {"JAX": "Jacksonville, Florida, USA", "OMA": "Omaha, Nebraska, USA"}
# Measured from image content (lane-line cycles, lane widths, tractor-trailers):
# DFC2019 tiles carry no geotransform. See
# docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md (Housekeeping).
DFC_GSD_M = 0.3

ROUTING = {
    "dfc2019": {
        "tier": 2, "locked": False,
        "label": "Tier 2 — height prediction",
        "dem": None,
        "summary": "Method 6 predicts height above ground. These tiles have no georeference, "
                   "so no DEM is added: the result is relative above-ground height, not absolute elevation.",
    },
    "vhr": {
        "tier": 2, "locked": False,
        "label": "Tier 2 — height prediction on FABDEM",
        "dem": "FABDEM",
        "summary": "Method 6 predicts height above ground; DSM = FABDEM bare earth + max(predicted height, 0).",
    },
    "sentinel2": {
        "tier": 1, "locked": True,
        "label": "Tier 1 — DEM only",
        "dem": "FABDEM",
        "summary": "Terrain from FABDEM only. Height prediction was shown not to work on 10 m Sentinel-2, "
                   "so it is never attempted for these tiles.",
    },
}


# --------------------------------------------------------------------------- curate
def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def curate(_args) -> None:
    for src in SOURCES:
        d = CURATED / src
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()

    tiles = sorted(p.stem.removesuffix("_depth")
                   for p in (ROOT / "data/dfc2019/experiments/dav2_baseline/depth").glob("*_depth.npy"))
    assert len(tiles) == 50, f"expected the 50-tile DFC2019 benchmark, found {len(tiles)}"
    for t in tiles:
        _symlink(ROOT / "data/dfc2019/raw/RGB/Track1-RGB" / f"{t}_RGB.tif", CURATED / "dfc2019" / f"{t}_RGB.tif")

    man = pd.read_csv(ROOT / "data/sentinel2_benchmark/manifest.csv")
    assert len(man) == 32, f"expected the 32 Sentinel-2 benchmark tiles, found {len(man)}"
    for _, m in man.iterrows():
        _symlink(ROOT / m.rgb_path, CURATED / "sentinel2" / f"{m.tile_id}_RGB.tif")

    sys.path.insert(0, str(ROOT / "scripts"))
    import vhr_dsm_pipeline as vhr  # reuse the pipeline's own crop definitions
    for name, (scene, cr, cc, cover) in vhr.CROPS.items():
        r0, c0 = cr * vhr.CELL + (vhr.CELL - vhr.SIZE) // 2, cc * vhr.CELL + (vhr.CELL - vhr.SIZE) // 2
        with rasterio.open(ROOT / "data/maxar_sanity" / vhr.SCENES[scene]) as src:
            win = Window(c0, r0, vhr.SIZE, vhr.SIZE)
            rgb = src.read([1, 2, 3], window=win)
            profile = {"driver": "GTiff", "width": vhr.SIZE, "height": vhr.SIZE, "count": 3, "dtype": rgb.dtype,
                       "crs": src.crs, "transform": src.window_transform(win), "compress": "deflate"}
        out = CURATED / "vhr" / f"{name}.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(rgb)
            dst.update_tags(source_scene=vhr.SCENES[scene], land_cover=cover, crop_name=name)
    print(f"curated: dfc2019 {len(tiles)}, sentinel2 {len(man)}, vhr {len(vhr.CROPS)}")


# --------------------------------------------------------------------------- build
def _stretch_rgb(arr: np.ndarray) -> np.ndarray:
    """(3,H,W) → (H,W,3) uint8 for display (uint8 passes through; others 2–98 % stretch)."""
    arr = arr[:3]
    if arr.dtype == np.uint8:
        return np.moveaxis(arr, 0, -1)
    out = np.zeros(arr.shape[1:] + (3,), np.uint8)
    for i in range(3):
        b = arr[i].astype(np.float32)
        lo, hi = np.nanpercentile(b, [2, 98])
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def _images(path: Path, item_id: str) -> tuple[str, str]:
    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / 1024
        h, w = max(1, round(src.height / scale)), max(1, round(src.width / scale))
        arr = src.read([1, 2, 3], out_shape=(3, h, w))
    img = Image.fromarray(_stretch_rgb(arr))
    prev = PREVIEWS / f"{item_id}.jpg"
    img.save(prev, quality=88)
    thumb = img.copy()
    thumb.thumbnail((256, 256))
    th = THUMBS / f"{item_id}.jpg"
    thumb.save(th, quality=82)
    return th.name, prev.name


def _geo(src) -> dict:
    """Centre lon/lat and footprint (km) of a georeferenced raster."""
    t = src.transform
    cx, cy = t * (src.width / 2, src.height / 2)
    lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
    return {"lat": round(lat, 5), "lon": round(lon, 5), "crs": src.crs.to_string(),
            "footprint_km": [round(src.width * abs(t.a) / 1000, 3), round(src.height * abs(t.e) / 1000, 3)]}


def build(_args) -> None:
    for d in (THUMBS, PREVIEWS):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
    s2 = pd.read_csv(ROOT / "data/sentinel2_benchmark/manifest.csv").set_index("tile_id")

    items = []
    for src_name in SOURCES:
        for path in sorted((CURATED / src_name).glob("*.tif")):
            real = path.resolve()
            low = str(real).lower()
            bad = [m for m in EXCLUDED_MARKERS if m in low]
            if bad:
                raise SystemExit(f"refusing to catalog excluded research artifact {real} ({bad})")
            with rasterio.open(path) as src:
                width, height = src.width, src.height
                georef = src.crs is not None and not src.transform.is_identity
                gsd = abs(src.transform.a) if georef else None
                geo = _geo(src) if georef else None
                tags = src.tags()

            if src_name == "dfc2019":
                tile = path.stem.removesuffix("_RGB")
                item_id = f"dfc2019-{tile}"
                meta = {
                    "source": "DFC2019 (IEEE GRSS Data Fusion Contest 2019, Track 1) · WorldView-3",
                    "title": tile,
                    "tile_id": tile,
                    "location": DFC_CITY.get(tile[:3], tile[:3]),
                    "gsd_m": DFC_GSD_M,
                    "gsd_source": "measured from image content (the tiles carry no geotransform)",
                    "georeferenced": False,
                    "geo": None,
                }
            elif src_name == "sentinel2":
                tile = path.stem.removesuffix("_RGB")
                m = s2.loc[tile]
                item_id = f"sentinel2-{tile}"
                meta = {
                    "source": "Sentinel-2 L2A (Copernicus) · true colour",
                    "title": str(m.label),
                    "tile_id": tile,
                    "location": str(m.label),
                    "category": str(m.category),
                    "acquired": str(m.date_acquired),
                    "gsd_m": round(gsd, 3),
                    "gsd_source": "geotransform",
                    "georeferenced": True,
                    "geo": geo,
                }
            else:
                item_id = f"vhr-{path.stem}"
                scene = tags.get("source_scene", "")
                meta = {
                    "source": "Maxar Open Data (VHR) · Sikkim / Darjeeling",
                    "title": path.stem.replace("_", " "),
                    "tile_id": path.stem,
                    "location": f"Sikkim / Darjeeling Himalaya — {tags.get('land_cover', '')}",
                    "land_cover": tags.get("land_cover"),
                    "scene": scene,
                    "acquired": scene.split("_")[2] if scene.count("_") >= 2 else None,
                    "gsd_m": round(gsd, 3),
                    "gsd_source": "geotransform",
                    "georeferenced": True,
                    "geo": geo,
                }
            thumb, preview = _images(path, item_id)
            items.append({
                "id": item_id,
                "collection": src_name,
                **meta,
                "size_px": [width, height],
                "routing": ROUTING[src_name],
                "thumbnail": thumb,
                "preview": preview,
                "file": str(real.relative_to(ROOT)),
            })
            print(f"{item_id:32s} gsd {meta['gsd_m']} m  tier {ROUTING[src_name]['tier']}", flush=True)

    counts = {s: sum(1 for i in items if i["collection"] == s) for s in SOURCES}
    MANIFEST.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/library_catalog.py build",
        "counts": counts,
        "tiers": {"1": "DEM only (FABDEM)", "2": "height prediction (Method 6)"},
        "items": items,
    }, indent=1))
    print(f"manifest: {counts} → {MANIFEST.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curate", "build"])
    a = ap.parse_args()
    {"curate": curate, "build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
