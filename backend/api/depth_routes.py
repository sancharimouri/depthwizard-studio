"""Relative depth (DAv2-Small) via a remote inference host (docs/DEPLOY.md).

DAV2_INFERENCE_URL is the host's base URL. It is read on every request, so pointing
it at a new host is an env change, not a code change. Right now that host is the
TEMPORARY Colab + Cloudflare quick tunnel bridge (bridge/dav2_server.py). The response
is passed through unchanged: 518x518 relative depth (larger = nearer), NOT elevation,
in the compact "u16-zlib" encoding (bridge/dav2_server.py encode_depth; older hosts send
raw float32 without an "encoding" field; frontend/src/depth-result.js decodes both).
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.input import store
from backend.library import catalog
from backend.storage import library_store

router = APIRouter(prefix="/api/depth")
TIMEOUT_S = float(os.environ.get("DAV2_TIMEOUT_S", "120"))


def inference_url() -> str:
    url = os.environ.get("DAV2_INFERENCE_URL", "").strip().rstrip("/")
    if not url:
        raise HTTPException(status_code=503, detail="Relative-depth inference host not configured (DAV2_INFERENCE_URL).")
    return url


@router.get("/status")
def status() -> dict:
    url = inference_url()
    try:
        r = httpx.get(f"{url}/health", timeout=15)
        r.raise_for_status()
        return {"reachable": True, "host": url, **r.json()}
    except httpx.HTTPError as exc:
        return {"reachable": False, "host": url, "error": type(exc).__name__}


async def _forward(filename: str, body: bytes, content_type: str) -> dict:
    url = inference_url()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            r = await client.post(f"{url}/predict", files={"file": (filename, body, content_type)})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Inference host unreachable ({type(exc).__name__}); "
                                                    "the temporary Colab bridge may be disconnected.") from exc
    if r.status_code != 200:
        raise HTTPException(status_code=502 if r.status_code >= 500 else r.status_code,
                            detail=f"Inference host returned HTTP {r.status_code}.")
    return r.json()


@router.post("/relative")
async def relative(file: UploadFile = File(...)) -> dict:
    return await _forward(file.filename or "image", await file.read(), file.content_type or "application/octet-stream")


async def _preview_bytes(source: str, item_id: str) -> bytes:
    """The same 1024 px preview the UI shows for a library item or an input (upload / CDSE scene)."""
    if source == "input":
        try:
            return store.preview_path(item_id, "image").read_bytes()
        except store.InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    item = catalog.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Unknown library item.")
    if library_store.mode() == "local":
        return catalog.image_path(item, "preview").read_bytes()
    if not library_store.is_public(item):
        return library_store.private_file(item, "preview").read_bytes()
    names = item.get("assets") or library_store.asset_names(item)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(library_store.release_url(names["preview"]))
    r.raise_for_status()
    return r.content


@router.post("/relative/{source}/{item_id}")
async def relative_for(source: str, item_id: str) -> dict:
    """Relative depth for a selected library item or input, by id (the UI's generation step)."""
    if source not in ("library", "input"):
        raise HTTPException(status_code=404, detail="source must be 'library' or 'input'.")
    inference_url()  # 503 early if unconfigured, before fetching the image
    try:
        body = await _preview_bytes(source, item_id)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch the preview ({type(exc).__name__}).") from exc
    return await _forward(f"{item_id}.jpg", body, "image/jpeg")
