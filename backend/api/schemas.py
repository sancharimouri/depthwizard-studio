"""Pydantic request/response schemas for the production/demo API."""

from pydantic import BaseModel


class SceneSearchRequest(BaseModel):
    lat: float
    lon: float
    aoi_km: float = 10.0
    date_from: str
    date_to: str
    max_cloud: float = 20.0


class SceneResult(BaseModel):
    id: str
    date: str
    cloud: float
    bbox: list[float]


class SceneSearchResponse(BaseModel):
    bbox: list[float]
    scenes: list[SceneResult]


class ScenePreviewRequest(BaseModel):
    bbox: list[float]
    date: str
    width: int = 512
    height: int = 512
