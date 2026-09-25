"""Relative depth (DAv2-Small) via a remote inference host (docs/DEPLOY.md).

DAV2_INFERENCE_URL is the host's base URL. It is read on every request, so pointing
it at a new host is an env change, not a code change. Right now that host is the
TEMPORARY Colab + Cloudflare quick tunnel bridge (bridge/dav2_server.py). The response
is passed through unchanged: raw 518x518 relative depth (larger = nearer), NOT elevation.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

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


@router.post("/relative")
async def relative(file: UploadFile = File(...)) -> dict:
    url = inference_url()
    body = await file.read()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            r = await client.post(f"{url}/predict",
                                  files={"file": (file.filename or "image", body, file.content_type or "application/octet-stream")})
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Inference host unreachable ({type(exc).__name__}); "
                                                    "the temporary Colab bridge may be disconnected.") from exc
    if r.status_code != 200:
        raise HTTPException(status_code=502 if r.status_code >= 500 else r.status_code,
                            detail=f"Inference host returned HTTP {r.status_code}.")
    return r.json()
