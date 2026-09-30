#!/usr/bin/env python3
"""Pre-bake the curated library as static files (2026-09-30; docs/container-measurements.md, Part 2).

Reads data/library_v2_<date>/tile_manifest.json (read-only) and, for every tile with include_in_container = true,
writes <out>/<tile_id>/ with exactly what the app needs for a library tile, so opening one needs no backend and no
Space call:
  produced by the backend's own backend/generation/pipeline.generate (the same code the live
  POST /api/generate/library/<id> runs), fed with the tile's baked DAv2 depth (pipeline.baked_depth):
    --format compact (default, the web deploy; Vercel Hobby's 100 MB upload limit):
      terrain.u16.gz   generate's terrain.json as uint16 (frontend/src/terrain-data.js decodes it; worst height error
                       = half a step = range / 131070). Maxar: BOTH the real heights (statistics, readouts) and the
                       DISPLAY heights (mesh), as terrain.json carries them.
      preview.jpg      also the 3D satellite texture (generate's satellite.png is this JPEG decoded and re-saved
                       losslessly: same image, 5x the bytes)
      relative_depth.png, elevation.png   as generate writes them
    --format json (exact, for parity and screenshots): terrain.json + satellite.png as generate writes them
  thumbnail.jpg
  tile.json        the generate response the frontend consumes (meta + depth fields, minus the base64 payload, which
                   relative_depth.png renders) + manifest fields (default_exaggeration, stars, source, notes)
and <out>/index.json: the curated listing in manifest order, the same shape as GET /api/library, each item with its
`select` plan (what POST /api/library/<id>/select returns for the UI's request).

Maxar: only ONE DISPLAY preset is baked, the manifest's `maxar_display_preset`, or --maxar-preset (read straight from
data/library_v2_<date>/dem_maxar/<preset>/, nothing in the library is changed). None is chosen yet (2026-09-30):
"medium" is the flagged placeholder.

Facts (2026-09-30, docs/facts-research.md Part E2): --facts points at a facts_drafts.json
(scripts/draft_library_facts.py). ONLY facts whose status is "curated" are baked, into the tile's index.json entry and
tile.json as `facts` (a list; [] when none is curated). The web build shows them with no backend call and hides the
Facts panel for a library tile whose list is empty. The desktop build never reads the static library, so it is unchanged.

Never calls the Space: a tile without a baked depth entry fails the bake (run build_library_v2.py depth first).
Re-runnable: re-rate tiles or pick a Maxar preset in library_v2, then run this again (the output folder is rebuilt).

  python scripts/bake_static_library.py                                   # -> frontend/public/library-static/
  python scripts/bake_static_library.py --out build/slim/static_trial --only sentinel2-almora
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB_V2 = ROOT / "data/library_v2_2026-09-29"
DEFAULT_OUT = ROOT / "frontend/public/library-static"
DEFAULT_FACTS = ROOT / "data/library_v2_2026-09-30/facts_drafts.json"
# what a baked fact carries (the drafts' internal fields, e.g. value/kind/status, stay out of the build)
FACT_FIELDS = ("text", "source", "source_url", "licence", "retrieved", "origin", "confidence", "scope")


def curated_facts(drafts: dict, tile_id: str) -> list[dict]:
    """The tile's facts with status "curated" only, in draft order, reduced to FACT_FIELDS."""
    facts = drafts.get("tiles", {}).get(tile_id, {}).get("facts", [])
    return [{k: f.get(k) for k in FACT_FIELDS} for f in facts if f.get("status") == "curated"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path, default=LIB_V2)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", help="bake just these tile ids (others in --out are kept)")
    ap.add_argument("--format", choices=["compact", "json"], default="compact")
    ap.add_argument("--maxar-preset", help="override the manifest's maxar_display_preset")
    ap.add_argument("--facts", type=Path, default=DEFAULT_FACTS,
                    help="facts_drafts.json; only status 'curated' facts are baked (missing file: no facts)")
    a = ap.parse_args()
    lib, out = a.library.resolve(), a.out.resolve()
    assert lib not in out.parents and out != lib, "never write into the library"
    gen_tmp = Path(tempfile.mkdtemp(prefix="dw2_bake_"))
    os.environ.update(DW2_NO_DOTENV="1", DW2_LIBRARY="bundle", DW2_LIBRARY_BUNDLE=str(lib),
                      DW2_LIBRARY_USER=str(gen_tmp / "_user"), DW2_TILE_MANIFEST=str(lib / "tile_manifest.json"),
                      DW2_GENERATED_DIR=str(gen_tmp / "gen"), DW2_UPLOADS_DIR=str(gen_tmp / "up"),
                      DW2_CACHE_DIR=str(gen_tmp / "cache"))
    sys.path.insert(0, str(ROOT))
    import gzip
    import struct

    import numpy as np

    from backend.api.library_routes import _public
    from backend.generation import pipeline
    from backend.library import catalog
    from backend.storage import library_store

    manifest = json.loads((lib / "tile_manifest.json").read_text())
    drafts = json.loads(a.facts.read_text()) if a.facts and a.facts.is_file() else {}
    preset = a.maxar_preset or manifest.get("maxar_display_preset")
    placeholder = a.maxar_preset is not None and a.maxar_preset != manifest.get("maxar_display_preset")
    local_asset = library_store.local_asset

    def asset_with_preset(item, kind):  # Maxar packs of the chosen preset, read in place
        if kind == "dem" and item.get("collection") == "vhr":
            p = lib / "dem_maxar" / preset / f"{item['id']}.tif"
            assert p.is_file(), p
            return p
        return local_asset(item, kind)

    library_store.local_asset = asset_with_preset
    tiles = sorted((t for t in manifest["tiles"] if t["include_in_container"]), key=lambda t: t["order_index"])
    if a.only:
        tiles = [t for t in tiles if t["tile_id"] in set(a.only)]
    elif out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for t in tiles:
        iid = t["tile_id"]
        item = catalog.get(iid)
        preview = library_store.local_asset(item, "preview").read_bytes()
        depth = pipeline.baked_depth(item, preview)
        if depth is None:
            raise SystemExit(f"{iid}: no baked depth for its preview (run scripts/build_library_v2.py depth); no Space calls here")
        meta = pipeline.generate("library", iid, preview, depth, item=item)
        d = out / iid
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(pipeline.generated_dir() / meta["job"], d)
        (d / "meta.json").unlink(missing_ok=True)  # generate's meta goes into tile.json below
        shutil.copyfile(local_asset(item, "preview"), d / "preview.jpg")
        shutil.copyfile(local_asset(item, "thumbnail"), d / "thumbnail.jpg")
        base = f"library-static/{iid}"
        if a.format == "compact":
            terrain = json.loads((d / "terrain.json").read_text())
            header = {k: v for k, v in terrain.items() if k not in ("heights", "display")}
            arrays = [terrain["heights"]]
            if "display" in terrain:
                header["display"] = {k: v for k, v in terrain["display"].items() if k != "heights"}
                arrays.append(terrain["display"]["heights"])
            hb = json.dumps(header, separators=(",", ":")).encode()
            blob = struct.pack("<I", len(hb)) + hb + b"\0" * ((4 + len(hb)) % 2)
            for arr in arrays:
                blob += np.round(np.clip(np.asarray(arr, np.float64), 0, 1) * 65535).astype("<u2").tobytes()
            (d / "terrain.u16.gz").write_bytes(gzip.compress(blob, 9, mtime=0))
            (d / "terrain.json").unlink()
            (d / "satellite.png").unlink()
            terrain_url, satellite_url = f"{base}/terrain.u16.gz", f"{base}/preview.jpg"
        else:
            terrain_url, satellite_url = f"{base}/terrain.json", f"{base}/satellite.png"
        response = {"meta": meta, "depth": {k: v for k, v in depth.items() if k != "data_b64"},
                    "assets": {"terrain": terrain_url, "satellite": satellite_url,
                               "depth": f"{base}/relative_depth.png", "elevation": f"{base}/elevation.png"}}
        facts = curated_facts(drafts, iid)
        (d / "tile.json").write_text(json.dumps({**response, "facts": facts, "manifest": {k: t.get(k) for k in (
            "display_name", "source", "stars", "order_index", "default_exaggeration", "exaggeration_origin", "notes")}},
            separators=(",", ":")))
        entry = _public(item)
        entry.update(thumbnail_url=f"{base}/thumbnail.jpg", preview_url=f"{base}/preview.jpg", tile_url=None,
                     static=f"{base}/tile.json", bundled=True, available=True,
                     select=catalog.route(item, item["routing"]["tier"]), facts=facts)
        if item.get("collection") == "vhr":
            entry["display_preset"] = {"name": preset, "placeholder": placeholder}
        listing.append(entry)
        print(f"{iid:32s} {sum(f.stat().st_size for f in d.iterdir()) / 1e6:6.2f} MB  {len(facts)} curated facts", flush=True)
    idx = out / "index.json"
    if a.only and idx.exists():
        old = {i["id"]: i for i in json.loads(idx.read_text())["items"]}
        old.update({i["id"]: i for i in listing})
        listing = sorted(old.values(), key=lambda i: i["order_index"])
    full = catalog.load()
    counts = {c: sum(i["collection"] == c for i in listing) for c in full["counts"]}
    idx.write_text(json.dumps({"generated_at": full["generated_at"], "generated_from": str(lib.relative_to(ROOT)),
                               "format": a.format, "maxar_display_preset": {"name": preset, "placeholder": placeholder},
                               "counts": counts, "tiers": full["tiers"], "total": len(listing), "items": listing},
                              separators=(",", ":")))
    shutil.rmtree(gen_tmp, ignore_errors=True)
    files = [f for f in out.rglob("*") if f.is_file()]
    n_facts = sum(len(i.get("facts") or []) for i in listing)
    print(f"{n_facts} curated facts on {sum(bool(i.get('facts')) for i in listing)} tiles (from {a.facts if drafts else 'no facts file'})")
    print(f"{len(listing)} tiles, {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
