"""Curated library ("Choose from Library") endpoints."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from backend.library import catalog
from backend.storage import library_store

router = APIRouter(prefix="/api/library")


def _public(item: dict) -> dict:
    """Client view of an item (backend/storage/library_store.py).

    Public items (Sentinel-2, Maxar): direct, stable GitHub Release URLs, incl. the tile.
    Private items (DFC2019): this server's routes, which serve the image from the private
    HF dataset; no tile URL (the DFC2019 terms forbid distributing the data).
    Local mode: this server's local-dev routes.
    """
    out = {k: v for k, v in item.items() if k not in ("thumbnail", "preview", "file", "r2", "assets", "store")}
    if library_store.mode() == "remote" and library_store.is_public(item):
        names = item.get("assets") or library_store.asset_names(item)
        out["thumbnail_url"] = library_store.release_url(names["thumbnail"])
        out["preview_url"] = library_store.release_url(names["preview"])
        out["tile_url"] = library_store.release_url(names["tile"])
    else:
        out["thumbnail_url"] = f"/api/library/{item['id']}/thumbnail"
        out["preview_url"] = f"/api/library/{item['id']}/preview"
    return out


def _image(item: dict, kind: str):
    if library_store.mode() == "remote":
        if library_store.is_public(item):
            names = item.get("assets") or library_store.asset_names(item)
            return RedirectResponse(library_store.release_url(names[kind]), status_code=307)
        try:
            path = library_store.private_file(item, kind)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Private library image unavailable ({type(exc).__name__}).") from exc
        return FileResponse(path, media_type="image/jpeg")
    return FileResponse(catalog.image_path(item, kind), media_type="image/jpeg")


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
    return _image(_item(item_id), "thumbnail")


@router.get("/{item_id}/preview")
def preview(item_id: str):
    return _image(_item(item_id), "preview")


class SelectRequest(BaseModel):
    requested_tier: Literal[1, 2] | None = None


@router.post("/{item_id}/select")
def select(item_id: str, req: SelectRequest | None = None):
    item = _item(item_id)
    plan = catalog.route(item, req.requested_tier if req else None)
    return {"item": _public(item), **plan}
