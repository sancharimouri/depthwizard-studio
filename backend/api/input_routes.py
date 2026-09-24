"""Page 1 input routes: Upload (image + optional DEM) and Search Online scenes.

Tier routing and DEM sourcing live in backend/input/store.py. FABDEM comes
from Google Earth Engine (backend/dem/fabdem.py).
"""

from fastapi import APIRouter, File, HTTPException, Header, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.cdse.client import CDSEAuthError, CDSEThrottled, CDSEUpstreamError, fetch_true_color_png
from backend.dem.fabdem import FabdemError
from backend.input import store

router = APIRouter(prefix="/api/input")


def _run(fn, *args):
    try:
        return fn(*args)
    except store.InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FabdemError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _public(meta: dict) -> dict:
    out = {k: v for k, v in meta.items() if k not in {"bounds"}}
    out["preview_url"] = f"/api/input/{meta['id']}/preview"
    out["dem_preview_url"] = f"/api/input/{meta['id']}/dem/preview" if meta.get("dem") else None
    return out


@router.post("/upload")
def upload(file: UploadFile = File(...)) -> dict:
    return _public(_run(store.create_upload, file.filename, file.file))


class SceneSelect(BaseModel):
    id: str | None = None
    date: str
    cloud: float | None = None
    bbox: list[float] = Field(min_length=4, max_length=4)


@router.post("/scene")
def select_scene(
    scene: SceneSelect,
    x_cdse_client_id: str | None = Header(None),
    x_cdse_client_secret: str | None = Header(None),
) -> dict:
    """Register a Search Online Sentinel-2 scene as an input (Tier 1, locked)
    and store its true-colour preview."""
    meta = _run(store.create_scene, scene.model_dump())
    creds = (x_cdse_client_id, x_cdse_client_secret) if x_cdse_client_id and x_cdse_client_secret else None
    try:
        png = fetch_true_color_png(scene.bbox, scene.date, 1024, 1024, creds=creds)
        store.save_scene_preview(meta["id"], png)
    except CDSEThrottled as exc:
        raise HTTPException(status_code=429, detail=str(exc),
                            headers={"Retry-After": exc.retry_after} if exc.retry_after else None) from exc
    except (CDSEAuthError, CDSEUpstreamError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _public(meta)


@router.get("/{input_id}")
def get_input(input_id: str) -> dict:
    return _public(_run(store.load_meta, input_id))


@router.get("/{input_id}/preview")
def preview(input_id: str) -> FileResponse:
    return FileResponse(_run(store.preview_path, input_id, "image"), media_type="image/jpeg")


@router.get("/{input_id}/dem/preview")
def dem_preview(input_id: str) -> FileResponse:
    return FileResponse(_run(store.preview_path, input_id, "dem"), media_type="image/png")


@router.post("/{input_id}/fabdem")
def fabdem(input_id: str) -> dict:
    return _public(_run(store.fetch_fabdem, input_id))


@router.post("/{input_id}/dem")
def user_dem(input_id: str, file: UploadFile = File(...)) -> dict:
    return _public(_run(store.attach_user_dem, input_id, file.filename, file.file))
