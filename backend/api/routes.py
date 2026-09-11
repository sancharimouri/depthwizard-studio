"""Production/demo API routes."""

from fastapi import APIRouter, HTTPException, Response

from backend.api.schemas import ScenePreviewRequest, SceneSearchRequest, SceneSearchResponse
from backend.cdse.client import CDSEAuthError, CDSEUpstreamError, fetch_true_color_png, search_scenes

router = APIRouter(prefix="/api/cdse")


@router.post("/search", response_model=SceneSearchResponse)
def cdse_search(req: SceneSearchRequest) -> SceneSearchResponse:
    try:
        result = search_scenes(
            lat=req.lat,
            lon=req.lon,
            aoi_km=req.aoi_km,
            date_from=req.date_from,
            date_to=req.date_to,
            max_cloud=req.max_cloud,
        )
    except CDSEAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CDSEUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return SceneSearchResponse(**result)


@router.post("/preview")
def cdse_preview(req: ScenePreviewRequest) -> Response:
    try:
        png_bytes = fetch_true_color_png(bbox=req.bbox, date=req.date)
    except CDSEAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CDSEUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(content=png_bytes, media_type="image/png")
