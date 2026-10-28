"""Real generation for a selected input (backend/generation/pipeline.py).

  POST /api/generate/{library|input}/{id}   depth (once) + elevation assets for this input
  GET  /api/generated/{job}/{name}          satellite.png | relative_depth.png | elevation.png | terrain.json | meta.json
                                            | predicted_height.png | predicted_height.json (Method 6, Tier 2 only)
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import logging

from fastapi import HTTPException as _HTTPException

from backend.api import depth_routes
from backend.generation import pipeline
from backend.input import store
from backend.library import catalog
from backend.storage import tmp_cap

router = APIRouter(prefix="/api")
MEDIA = {".png": "image/png", ".json": "application/json"}


@router.post("/generate/{source}/{item_id}")
async def generate(source: str, item_id: str) -> dict:
    if source not in ("library", "input"):
        raise HTTPException(status_code=404, detail="source must be 'library' or 'input'.")
    hold = None
    if source == "input":  # held for the whole request (a cold Space can take minutes): the /tmp cap keeps it
        try:
            hold = store.input_dir(item_id)
        except store.InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    with tmp_cap.in_use(hold):
        return await _generate(source, item_id)


async def _generate(source: str, item_id: str) -> dict:
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
    if source == "input":
        kwargs["agl"] = await _method6(item_id, preview, kwargs["input_meta"])
    try:
        meta = await asyncio.to_thread(pipeline.generate, source, item_id, preview, depth, **kwargs)
    except Exception as exc:  # noqa: BLE001 - e.g. GLO-30 unreachable for an input without a DEM
        raise HTTPException(status_code=502, detail=f"Generation failed ({type(exc).__name__}: {str(exc)[:200]}).") from exc
    base = f"/api/generated/{meta['job']}"
    assets = {"terrain": f"{base}/terrain.json", "satellite": f"{base}/satellite.png",
              "depth": f"{base}/relative_depth.png", "elevation": f"{base}/elevation.png"}
    if meta.get("agl", {}).get("ran"):
        assets.update(agl=f"{base}/predicted_height.png", agl_grid=f"{base}/predicted_height.json")
    return {"meta": meta, "depth": depth, "assets": assets}


# Method 6 (docker/model/app.py /agl): uploads of TIER2_MAX_GSD_M or finer only. The service resamples to 0.3 m and
# caps that grid at 2048 px a side, so the footprint can be at most 614.4 m; the 1024 px preview then has a GSD of
# 0.6 m or finer, so the preview is what is sent.
AGL_MAX_FOOTPRINT_M = 2048 * 0.3


async def _method6(item_id: str, preview: bytes, meta: dict) -> dict | None:
    gsd = meta.get("gsd_m")
    if not gsd or not depth_routes.model_url():
        return None
    if gsd > store.TIER2_MAX_GSD_M:
        return {"ran": False, "reason": f"{gsd:.2f} m/pixel is coarser than {store.TIER2_MAX_GSD_M} m."}
    from PIL import Image
    import io
    size = meta.get("size_px") or []
    footprint = max(size) * gsd if size else 0
    if not footprint or footprint > AGL_MAX_FOOTPRINT_M:
        return {"ran": False, "reason": f"The image covers {footprint:.0f} m; height prediction is limited to "
                                        f"{AGL_MAX_FOOTPRINT_M:.0f} m a side."}
    preview_gsd = footprint / max(Image.open(io.BytesIO(preview)).size)
    try:
        out = await depth_routes.call_model("/agl", f"{item_id}.jpg", preview, "image/jpeg",
                                            {"gsd_m": f"{preview_gsd:.4f}"})
    except _HTTPException as exc:
        logging.getLogger("uvicorn.error").warning("Method 6 skipped for %s: %s", item_id, exc.detail)
        return {"ran": False, "reason": f"The height model didn't answer ({str(exc.detail)[:120]})."}
    return {"ran": True, **out}


@router.get("/generated/{job_id}/{name}")
def generated_asset(job_id: str, name: str):
    try:
        path = pipeline.job_path(job_id, name)
    except pipeline.GenerationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No such generated asset.")
    return FileResponse(path, media_type=MEDIA[path.suffix])
