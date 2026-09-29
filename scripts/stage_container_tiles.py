#!/usr/bin/env python3
"""Stage the curated tiles for the (private) backend container: reads tile_manifest.json only.

Dry run (the default) lists exactly which files the Docker build would copy: the elevation pack of
every tile with include_in_container = true (dem/<tile_id>.tif), and the preview/thumbnail each
generation needs. --copy DEST really copies them (DEST must be empty or new, outside the library).

The container is private: DFC2019 packs and images may go into it, never into a public image,
repo or release (docs/STORAGE.md).

  python scripts/stage_container_tiles.py                                   # dry run, default manifest
  python scripts/stage_container_tiles.py --root data/library_v2_2026-09-29 --json
  python scripts/stage_container_tiles.py --copy build/container_tiles      # not run as of 2026-09-29
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def plan(root: Path) -> dict:
    man = json.loads((root / "tile_manifest.json").read_text())
    cat = {i["id"]: i for i in json.loads((root / "manifest.json").read_text())["items"]}
    tiles = sorted((t for t in man["tiles"] if t["include_in_container"]), key=lambda t: t["order_index"])
    packs, images, missing = [], [], []
    for t in tiles:
        tid = t["tile_id"]
        p = root / "dem" / f"{tid}.tif"
        (packs if p.is_file() else missing).append(p)
        for d, key in (("previews", "preview"), ("thumbnails", "thumbnail")):
            q = root / d / cat[tid][key]
            (images if q.is_file() else missing).append(q)
    return {"manifest": str(root / "tile_manifest.json"), "tiles": [t["tile_id"] for t in tiles],
            "by_source": {s: sum(t["source"] == s for t in tiles) for s in ("sentinel2", "maxar", "dfc2019")},
            "maxar_display_preset": man.get("maxar_display_preset"),
            "packs": [str(p.relative_to(root)) for p in packs], "images": [str(p.relative_to(root)) for p in images],
            "missing": [str(p) for p in missing],
            "bytes": {"packs": sum(p.stat().st_size for p in packs), "images": sum(p.stat().st_size for p in images)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "data/library_v2_2026-09-29")
    ap.add_argument("--json", action="store_true", help="print the full plan as JSON")
    ap.add_argument("--copy", type=Path, help="really copy into this new folder (default: dry run)")
    a = ap.parse_args()
    root = a.root.resolve()
    p = plan(root)
    if a.json:
        print(json.dumps(p, indent=1))
    else:
        print(f"DRY RUN: {len(p['tiles'])} included tiles {p['by_source']}, Maxar preset {p['maxar_display_preset']}")
        print(f"packs  ({len(p['packs'])}, {p['bytes']['packs'] / 1e6:.1f} MB):")
        for f in p["packs"]:
            print("  ", f)
        print(f"images ({len(p['images'])}, {p['bytes']['images'] / 1e6:.1f} MB): previews/ + thumbnails/ of the same tiles")
        if p["missing"]:
            print("MISSING:", *p["missing"], sep="\n  ")
    if p["missing"]:
        raise SystemExit(1)
    if a.copy:
        dest = a.copy.resolve()
        assert not dest.exists() or not any(dest.iterdir()), f"{dest} is not empty"
        assert root not in dest.parents and dest != root, "copy destination must be outside the library"
        for rel in p["packs"] + p["images"]:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, dest / rel)
        shutil.copy2(root / "tile_manifest.json", dest / "tile_manifest.json")
        print(f"copied {len(p['packs']) + len(p['images'])} files + tile_manifest.json -> {dest}")


if __name__ == "__main__":
    main()
