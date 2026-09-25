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
    return "local" if os.environ.get("DW2_LIBRARY", "remote").strip().lower() == "local" else "remote"


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


def _hub_file(path: str, force: bool = False) -> Path:
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
