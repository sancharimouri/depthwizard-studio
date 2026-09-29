#!/usr/bin/env python3
"""Pre-bake the curated library as static files (2026-09-30; docs/container-measurements.md, Part 2).

Reads data/library_v2_<date>/tile_manifest.json (read-only) and, for every tile with include_in_container = true,
writes <out>/<tile_id>/ with exactly what the app needs for a library tile, so opening one needs no backend and no
Space call:
  satellite.png, relative_depth.png, elevation.png, terrain.json, meta.json
      produced by the backend's own backend/generation/pipeline.generate (the same code the live
      POST /api/generate/library/<id> runs), fed with the tile's baked DAv2 depth (pipeline.baked_depth). Maxar:
      terrain.json carries the DISPLAY surface for the mesh and the real bands for statistics, as generate writes it.
  preview.jpg, thumbnail.jpg      the library_v2 images
  tile.json                       the generate response the frontend consumes (meta + depth fields, minus the base64
                                  payload, which relative_depth.png already renders) + manifest fields
                                  (default_exaggeration, stars, source, credit)
and <out>/index.json: the curated listing, in manifest order.

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", type=Path, default=LIB_V2)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", help="bake just these tile ids (others in --out are kept)")
    a = ap.parse_args()
    lib, out = a.library.resolve(), a.out.resolve()
    assert lib not in out.parents and out != lib, "never write into the library"
    gen_tmp = Path(tempfile.mkdtemp(prefix="dw2_bake_"))
    os.environ.update(DW2_NO_DOTENV="1", DW2_LIBRARY="bundle", DW2_LIBRARY_BUNDLE=str(lib),
                      DW2_LIBRARY_USER=str(gen_tmp / "_user"), DW2_TILE_MANIFEST=str(lib / "tile_manifest.json"),
                      DW2_GENERATED_DIR=str(gen_tmp / "gen"), DW2_UPLOADS_DIR=str(gen_tmp / "up"),
                      DW2_CACHE_DIR=str(gen_tmp / "cache"))
    sys.path.insert(0, str(ROOT))
    from backend.api.library_routes import _public
    from backend.generation import pipeline
    from backend.library import catalog, tile_manifest
    from backend.storage import library_store

    tiles = sorted((t for t in json.loads((lib / "tile_manifest.json").read_text())["tiles"] if t["include_in_container"]),
                   key=lambda t: t["order_index"])
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
        shutil.copyfile(library_store.local_asset(item, "preview"), d / "preview.jpg")
        shutil.copyfile(library_store.local_asset(item, "thumbnail"), d / "thumbnail.jpg")
        base = f"library-static/{iid}"
        response = {"meta": meta, "depth": {k: v for k, v in depth.items() if k != "data_b64"},
                    "assets": {"terrain": f"{base}/terrain.json", "satellite": f"{base}/satellite.png",
                               "depth": f"{base}/relative_depth.png", "elevation": f"{base}/elevation.png"}}
        (d / "tile.json").write_text(json.dumps({**response, "manifest": {k: t.get(k) for k in (
            "display_name", "source", "stars", "order_index", "default_exaggeration", "exaggeration_origin", "notes")}},
            separators=(",", ":")))
        entry = _public(item)
        entry.update(thumbnail_url=f"{base}/thumbnail.jpg", preview_url=f"{base}/preview.jpg", tile_url=None,
                     static=f"{base}/tile.json", bundled=True, available=True)
        listing.append(entry)
        print(f"{iid:32s} {sum(f.stat().st_size for f in d.iterdir()) / 1e6:6.2f} MB", flush=True)
    idx = out / "index.json"
    if a.only and idx.exists():
        old = {i["id"]: i for i in json.loads(idx.read_text())["items"]}
        old.update({i["id"]: i for i in listing})
        listing = sorted(old.values(), key=lambda i: i["order_index"])
    idx.write_text(json.dumps({"generated_from": str(lib.relative_to(ROOT)), "items": listing}, separators=(",", ":")))
    shutil.rmtree(gen_tmp, ignore_errors=True)
    files = [f for f in out.rglob("*") if f.is_file()]
    print(f"{len(listing)} tiles, {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
