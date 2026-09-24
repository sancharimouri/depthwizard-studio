"""Curated library ("Choose from Library") endpoints."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.library import catalog

router = APIRouter(prefix="/api/library")


def _public(item: dict) -> dict:
    out = {k: v for k, v in item.items() if k not in ("thumbnail", "preview", "file")}
    out["thumbnail_url"] = f"/api/library/{item['id']}/thumbnail"
    out["preview_url"] = f"/api/library/{item['id']}/preview"
    return out


def _load():
    try:
        return catalog.load()
    except catalog.CatalogUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _item(item_id: str) -> dict:
    _load()
    item = catalog.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"No library item '{item_id}'.")
    return item


@router.get("")
def list_items(
    collection: Literal["dfc2019", "sentinel2", "vhr"] | None = None,
    tier: int | None = Query(None, ge=1, le=2),
):
    data = _load()
    sel = [i for i in data["items"]
           if (collection is None or i["collection"] == collection)
           and (tier is None or i["routing"]["tier"] == tier)]
    return {
        "generated_at": data["generated_at"],
        "counts": data["counts"],
        "tiers": data["tiers"],
        "total": len(sel),
        "items": [_public(i) for i in sel],
    }


@router.get("/{item_id}")
def get_item(item_id: str):
    return _public(_item(item_id))


@router.get("/{item_id}/thumbnail")
def thumbnail(item_id: str):
    return FileResponse(catalog.image_path(_item(item_id), "thumbnail"), media_type="image/jpeg")


@router.get("/{item_id}/preview")
def preview(item_id: str):
    return FileResponse(catalog.image_path(_item(item_id), "preview"), media_type="image/jpeg")


class SelectRequest(BaseModel):
    requested_tier: Literal[1, 2] | None = None


@router.post("/{item_id}/select")
def select(item_id: str, req: SelectRequest | None = None):
    item = _item(item_id)
    plan = catalog.route(item, req.requested_tier if req else None)
    return {"item": _public(item), **plan}
