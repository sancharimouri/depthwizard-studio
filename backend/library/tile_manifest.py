"""Curated tile manifest (library_v2, 2026-09-29; docs/library_v2.md).

The single source of truth for which library tiles are listed, in what order, and with which default
vertical exaggeration: tile_manifest.json, written by scripts/build_library_v2.py and read here, by the
generate response (backend/generation/pipeline.py -> meta.default_exaggeration, applied by the frontend)
and by scripts/stage_container_tiles.py.

Off unless DW2_TILE_MANIFEST points at a manifest file: without it the listing is the catalog, exactly
as before (production config is unchanged).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_cache: dict = {"path": None, "mtime": None, "data": None}


def path() -> Path | None:
    p = os.environ.get("DW2_TILE_MANIFEST", "").strip()
    return Path(p) if p else None


def load() -> dict | None:
    """{tile_id: entry} or None when no manifest is configured."""
    p = path()
    if p is None:
        return None
    mtime = p.stat().st_mtime
    if _cache["path"] != p or _cache["mtime"] != mtime:
        tiles = json.loads(p.read_text())["tiles"]
        _cache.update(path=p, mtime=mtime, data={t["tile_id"]: t for t in tiles})
    return _cache["data"]


def curate(items: list[dict]) -> list[dict]:
    """The listing: included tiles only, in order_index order. Items the manifest doesn't know are
    dropped too, so a stale catalog can't leak a tile into the curated set."""
    man = load()
    if man is None:
        return items
    kept = [i for i in items if man.get(i["id"], {}).get("include_in_container")]
    return sorted(kept, key=lambda i: man[i["id"]]["order_index"])


def entry(item_id: str) -> dict | None:
    man = load()
    return None if man is None else man.get(item_id)


def default_exaggeration(item_id: str) -> float | None:
    e = entry(item_id)
    return None if e is None else e.get("default_exaggeration")
