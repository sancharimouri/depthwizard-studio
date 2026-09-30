"""Where the curated tile library lives (active path; docs/STORAGE.md).

Split by licence, decided 2026-09-26:

  sentinel2, vhr  -> PUBLIC GitHub Release assets
                     github.com/{DW2_ASSETS_REPO}/releases/tag/{DW2_ASSETS_TAG}
                     (Copernicus Sentinel data licence; Maxar Open Data CC BY-NC 4.0,
                     both with attribution in that repo's README and release notes)
  dfc2019         -> PRIVATE Hugging Face dataset {DW2_LIBRARY_DATASET}
                     (the DFC2019 contest terms forbid dissemination of the data)
  manifest.json   -> the private dataset too (it lists every item)

The browser gets direct release URLs for public items. Private DFC2019 thumbnails
and previews are fetched by the backend with HF_TOKEN and served by it; DFC2019 tiles
get no download URL. DW2_LIBRARY=local switches to the local data/library files
(development only).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from backend.storage.hf_checkpoints import token as hf_token

GH_REPO = os.environ.get("DW2_ASSETS_REPO", "sancharimouri/depthwizard2-assets")
GH_TAG = os.environ.get("DW2_ASSETS_TAG", "library-v1")
HF_DATASET = os.environ.get("DW2_LIBRARY_DATASET", "sancharimouri/depthwizard2-library-private")
MANIFEST_FILE = "manifest.json"
PUBLIC_COLLECTIONS = frozenset({"sentinel2", "vhr"})

_memo: dict = {"at": 0.0, "data": None}


def mode() -> str:
    """remote (default), local (dev: data/library), or bundle (the desktop app)."""
    v = os.environ.get("DW2_LIBRARY", "remote").strip().lower()
    return v if v in ("local", "bundle") else "remote"


def is_public(item: dict) -> bool:
    return item["collection"] in PUBLIC_COLLECTIONS


def asset_names(item: dict) -> dict:
    """Public release assets are one flat namespace; the private dataset uses folders."""
    if is_public(item):
        i = item["id"]
        return {"tile": f"{i}.tif", "thumbnail": f"{i}__thumb.jpg", "preview": f"{i}__preview.jpg"}
    i = item["id"]
    return {"tile": f"tiles/{i}.tif", "thumbnail": f"thumbnails/{i}.jpg", "preview": f"previews/{i}.jpg"}


def release_url(name: str) -> str:
    return f"https://github.com/{GH_REPO}/releases/download/{GH_TAG}/{name}"


class PrivateLibraryDisabled(PermissionError):
    """DW2_PRIVATE_LIBRARY=off (the Cloud Run image): the private HF dataset is never read, so nothing is ever written
    to the HF cache. The web app's library is static (frontend/public/library-static) and never calls these routes."""


def private_library_enabled() -> bool:
    return os.environ.get("DW2_PRIVATE_LIBRARY", "on").strip().lower() not in ("off", "0", "false", "no")


def _hub_file(path: str, force: bool = False) -> Path:
    # the ONE place the private dataset is downloaded (manifest, DFC2019 images, packs, on-demand tiles)
    if not private_library_enabled():
        raise PrivateLibraryDisabled("The tile library is served statically by the web app; this backend does not "
                                     "serve the private library.")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_DATASET, filename=path, repo_type="dataset",
                                token=hf_token(), force_download=force))


def load_manifest() -> dict:
    """The catalog from the private dataset, re-checked every DW2_MANIFEST_TTL s (300)."""
    ttl = float(os.environ.get("DW2_MANIFEST_TTL", "300"))
    if _memo["data"] is not None and time.monotonic() - _memo["at"] < ttl:
        return _memo["data"]
    data = json.loads(_hub_file(MANIFEST_FILE).read_text())
    _memo.update(at=time.monotonic(), data=data)
    return data


def private_file(item: dict, kind: str) -> Path:
    """A private (DFC2019) image, via the HF cache (a cache, not the source of truth)."""
    if is_public(item):
        raise ValueError(f"{item['id']} is public; use its release URL")
    return _hub_file(item.get("assets", asset_names(item))[kind])


# --------------------------------------------------------------------------- bundle mode
# The desktop app ships a tiered library (desktop/tiles/build_bundle.py): some items in
# full, the rest thumbnail-only and downloaded on demand into a writable per-user folder.
#   DW2_LIBRARY_BUNDLE  the read-only bundled library (manifest.json, thumbnails/, previews/, tiles/)
#   DW2_LIBRARY_USER    per-user folder for downloaded items (previews/, tiles/)
FOLDERS = {"thumbnail": "thumbnails", "preview": "previews", "tile": "tiles", "dem": "dem"}


class DownloadNeedsToken(PermissionError):
    """A private (DFC2019) item needs HF_TOKEN to download."""


def bundle_dir() -> Path:
    return Path(os.environ["DW2_LIBRARY_BUNDLE"])


def user_dir() -> Path:
    return Path(os.environ.get("DW2_LIBRARY_USER") or bundle_dir() / "_downloads")


def _local_bases() -> list[Path]:
    if mode() == "bundle":
        return [bundle_dir(), user_dir()]
    if mode() == "local":
        return [Path(__file__).resolve().parents[2] / "data" / "library"]
    return []


def local_asset(item: dict, kind: str) -> Path | None:
    """The bundled, already-downloaded (bundle mode) or dev-local (local mode) file, if any.
    kind "dem" is the item's elevation pack (desktop/tiles/build_dem_pack.py)."""
    name = item.get(kind) or (f"{item['id']}.tif" if kind == "dem" else None)
    if not name:
        return None
    for base in _local_bases():
        p = base / FOLDERS[kind] / name
        if p.is_file():
            return p
    return None


def private_pack(item: dict) -> Path | None:
    """A private (DFC2019) item's curated elevation pack from the private HF dataset,
    dem/<item id>.tif (docs/method-audit/08-dfc2019-terrain-packs/), remote mode only. Served to
    generation server-side; the browser never gets the pack. None when the item has no pack or the
    hub is unreachable, so generation falls back exactly as before."""
    if is_public(item) or mode() != "remote":
        return None
    try:
        return _hub_file(f"{FOLDERS['dem']}/{item['id']}.tif")
    except Exception as exc:  # noqa: BLE001 - no pack for this item, no token, or offline
        import logging
        logging.getLogger(__name__).warning("no elevation pack for %s (%s)", item.get("id"), type(exc).__name__)
        return None


def is_available(item: dict) -> bool:
    return local_asset(item, "preview") is not None and local_asset(item, "tile") is not None


def download(item: dict) -> None:
    """Fetch an on-demand item's preview and tile into the per-user folder (atomic per file).
    Public items: the GitHub Release (no token). DFC2019: the private HF dataset (HF_TOKEN)."""
    import shutil

    import httpx

    src = item["download"]
    if src["source"] == "hf-private" and not hf_token():
        raise DownloadNeedsToken("DFC2019 tiles come from a private Hugging Face dataset: set HF_TOKEN to download them.")
    for kind in ("preview", "tile"):
        dst = user_dir() / FOLDERS[kind] / item[kind]
        if dst.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        if src["source"] == "github-release":
            with httpx.stream("GET", src[kind], follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        else:
            shutil.copyfile(_hub_file(src[kind]), part)
        part.replace(dst)
