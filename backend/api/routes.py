"""Production/demo API routes: CDSE Sentinel-2 search, preview and quota.

Every route accepts an optional user-supplied Copernicus key via the
X-CDSE-Client-Id / X-CDSE-Client-Secret headers ("use your own Copernicus API
key"); without them the project's own credentials from .env are used. A user
key is only held in memory for the token exchange and never logged or stored.
"""

from fastapi import APIRouter, Header, HTTPException, Response

from backend.api.schemas import ScenePreviewRequest, SceneSearchRequest, SceneSearchResponse
from backend.cdse.client import (
    CDSEAuthError,
    CDSEThrottled,
    CDSEUpstreamError,
    fetch_true_color_png,
    quota_status,
    search_scenes,
)

router = APIRouter(prefix="/api/cdse")


def _creds(client_id: str | None, client_secret: str | None) -> tuple[str, str] | None:
    if client_id or client_secret:
        if not (client_id and client_secret):
            raise HTTPException(status_code=400, detail="Both a client id and a client secret are needed.")
        return client_id.strip(), client_secret.strip()
    return None


def _call(fn, *args, **kwargs):
    user_key = bool(kwargs.get("creds") or (args and args[0]))
    try:
        return fn(*args, **kwargs)
    except CDSEThrottled as exc:
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        raise HTTPException(status_code=429, detail=str(exc), headers=headers) from exc
    except CDSEAuthError as exc:
        # A rejected user key is the caller's problem (401); missing or
        # rejected project credentials are a server configuration one (503).
        raise HTTPException(status_code=401 if user_key else 503, detail=str(exc)) from exc
    except CDSEUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/search", response_model=SceneSearchResponse)
def cdse_search(
    req: SceneSearchRequest,
    x_cdse_client_id: str | None = Header(None),
    x_cdse_client_secret: str | None = Header(None),
) -> SceneSearchResponse:
    result = _call(
        search_scenes,
        lat=req.lat,
        lon=req.lon,
        aoi_km=req.aoi_km,
        date_from=req.date_from,
        date_to=req.date_to,
        max_cloud=req.max_cloud,
        creds=_creds(x_cdse_client_id, x_cdse_client_secret),
    )
    return SceneSearchResponse(**result)


@router.post("/preview")
def cdse_preview(
    req: ScenePreviewRequest,
    x_cdse_client_id: str | None = Header(None),
    x_cdse_client_secret: str | None = Header(None),
) -> Response:
    png_bytes = _call(
        fetch_true_color_png,
        bbox=req.bbox, date=req.date, width=req.width, height=req.height,
        creds=_creds(x_cdse_client_id, x_cdse_client_secret),
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/quota")
def cdse_quota(
    x_cdse_client_id: str | None = Header(None),
    x_cdse_client_secret: str | None = Header(None),
) -> dict:
    return _call(quota_status, _creds(x_cdse_client_id, x_cdse_client_secret))
