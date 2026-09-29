"""Real generation for a selected input (backend/generation/pipeline.py).

  POST /api/generate/{library|input}/{id}   depth (once) + elevation assets for this input
  GET  /api/generated/{job}/{name}          satellite.png | relative_depth.png | elevation.png | terrain.json | meta.json
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.api import depth_routes
from backend.generation import pipeline
from backend.input import store
from backend.library import catalog

router = APIRouter(prefix="/api")
MEDIA = {".png": "image/png", ".json": "application/json"}


@router.post("/generate/{source}/{item_id}")
async def generate(source: str, item_id: str) -> dict:
    if source not in ("library", "input"):
        raise HTTPException(status_code=404, detail="source must be 'library' or 'input'.")
    preview = await depth_routes._preview_bytes(source, item_id)
    # Library tiles: the DAv2 depth baked into the tile's pack for exactly this preview (scripts/build_library_v2.py
    # depth), so opening a library tile makes no Space call. No baked entry -> the Space, exactly as before.
    depth = pipeline.baked_depth(catalog.get(item_id), preview) if source == "library" else None
    if depth is None:
        depth = await depth_routes._forward(f"{item_id}.jpg", preview, "image/jpeg")
    kwargs = {}
    if source == "library":
        kwargs["item"] = catalog.get(item_id)
    else:
        try:
            kwargs["input_meta"] = store.load_meta(item_id)
            kwargs["input_dir"] = store.preview_path(item_id, "image").parent
        except store.InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        meta = await asyncio.to_thread(pipeline.generate, source, item_id, preview, depth, **kwargs)
    except Exception as exc:  # noqa: BLE001 - e.g. GLO-30 unreachable for an input without a DEM
        raise HTTPException(status_code=502, detail=f"Generation failed ({type(exc).__name__}: {str(exc)[:200]}).") from exc
    base = f"/api/generated/{meta['job']}"
    return {"meta": meta, "depth": depth,
            "assets": {"terrain": f"{base}/terrain.json", "satellite": f"{base}/satellite.png",
                       "depth": f"{base}/relative_depth.png", "elevation": f"{base}/elevation.png"}}


@router.get("/generated/{job_id}/{name}")
def generated_asset(job_id: str, name: str):
    try:
        path = pipeline.job_path(job_id, name)
    except pipeline.GenerationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No such generated asset.")
    return FileResponse(path, media_type=MEDIA[path.suffix])
