"""Active library storage (backend/storage/library_store.py): public Sentinel-2/Maxar
items -> GitHub Release URLs; private DFC2019 items -> served by the backend from the
private HF dataset, never a public tile URL. No network: the hub is monkeypatched."""

import json

import pytest
from fastapi.testclient import TestClient

from backend.library import catalog
from backend.main import app
from backend.storage import library_store as ls

client = TestClient(app)
pytestmark = pytest.mark.skipif(not catalog.MANIFEST.exists(), reason="library manifest not generated")


@pytest.fixture
def remote(monkeypatch, tmp_path):
    monkeypatch.setenv("DW2_LIBRARY", "remote")
    ls._memo.update(at=0.0, data=None)
    local = json.loads(catalog.MANIFEST.read_text())
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    served = []

    def fake_hub(path, force=False):
        served.append(path)
        if path == ls.MANIFEST_FILE:
            out = tmp_path / "manifest.json"
            out.write_text(json.dumps(local))
            return out
        return img

    monkeypatch.setattr(ls, "_hub_file", fake_hub)
    yield served
    ls._memo.update(at=0.0, data=None)


def test_asset_layout_splits_by_licence():
    assert ls.is_public({"collection": "sentinel2"}) and ls.is_public({"collection": "vhr"})
    assert not ls.is_public({"collection": "dfc2019"})
    assert ls.asset_names({"id": "vhr-a_forest", "collection": "vhr"})["tile"] == "vhr-a_forest.tif"
    assert ls.asset_names({"id": "dfc2019-JAX_004_006", "collection": "dfc2019"})["preview"] == "previews/dfc2019-JAX_004_006.jpg"


def test_remote_catalog_urls(remote):
    d = client.get("/api/library").json()
    assert remote[0] == ls.MANIFEST_FILE  # the catalog comes from the private dataset
    items = {i["id"]: i for i in d["items"]}
    s2 = items["sentinel2-almora"]
    assert s2["preview_url"] == ls.release_url("sentinel2-almora__preview.jpg")
    assert s2["tile_url"] == ls.release_url("sentinel2-almora.tif")
    dfc = [i for i in d["items"] if i["collection"] == "dfc2019"]
    assert dfc and all(i["preview_url"].startswith("/api/library/") and "tile_url" not in i for i in dfc)
    assert not any("dfc2019" in (i.get("tile_url") or "") for i in d["items"])


def test_remote_image_routes(remote):
    r = client.get("/api/library/vhr-a_forest/thumbnail", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == ls.release_url("vhr-a_forest__thumb.jpg")
    r = client.get("/api/library/dfc2019-JAX_004_006/preview")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert "previews/dfc2019-JAX_004_006.jpg" in remote


def test_remote_unreachable_is_503(monkeypatch):
    monkeypatch.setenv("DW2_LIBRARY", "remote")
    ls._memo.update(at=0.0, data=None)

    def boom(path, force=False):
        raise ConnectionError("offline")

    monkeypatch.setattr(ls, "_hub_file", boom)
    assert client.get("/api/library").status_code == 503
