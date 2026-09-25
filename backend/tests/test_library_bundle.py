"""Desktop bundle mode: bundled vs on-demand items, downloads, and the depth path."""
import contextlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.library import catalog
from backend.main import app
from backend.storage import library_store

client = TestClient(app)


def item(iid, collection, bundled):
    src = ({"source": "github-release", "preview": f"https://x/{iid}__preview.jpg", "tile": f"https://x/{iid}.tif"}
           if collection != "dfc2019" else {"source": "hf-private", "dataset": "o/d", "preview": f"previews/{iid}.jpg",
                                             "tile": f"tiles/{iid}.tif"})
    return {"id": iid, "collection": collection, "title": iid, "routing": {"tier": 1, "locked": True, "label": "T1"},
            "thumbnail": f"{iid}.jpg", "preview": f"{iid}.jpg", "tile": f"{iid}.tif", "bundled": bundled,
            "download": {**src, "bytes": 123}}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    b, u = tmp_path / "bundle", tmp_path / "user"
    for d in ("thumbnails", "previews", "tiles"):
        (b / d).mkdir(parents=True)
    items = [item("sentinel2-a", "sentinel2", True), item("sentinel2-b", "sentinel2", False), item("dfc2019-c", "dfc2019", False)]
    for it in items:
        (b / "thumbnails" / it["thumbnail"]).write_bytes(b"T")
    (b / "previews" / "sentinel2-a.jpg").write_bytes(b"P")
    (b / "tiles" / "sentinel2-a.tif").write_bytes(b"TIF")
    (b / "manifest.json").write_text(json.dumps({"generated_at": "x", "counts": {}, "tiers": {}, "items": items}))
    monkeypatch.setenv("DW2_LIBRARY", "bundle")
    monkeypatch.setenv("DW2_LIBRARY_BUNDLE", str(b))
    monkeypatch.setenv("DW2_LIBRARY_USER", str(u))
    catalog._bundle_cache.update(mtime=None, data=None)
    return b, u


def test_listing_marks_bundled_and_on_demand(bundle):
    items = {i["id"]: i for i in client.get("/api/library").json()["items"]}
    assert items["sentinel2-a"]["available"] is True and items["sentinel2-a"]["bundled"] is True
    assert items["sentinel2-b"]["available"] is False and items["sentinel2-b"]["download_source"] == "github-release"
    assert items["dfc2019-c"]["download_source"] == "hf-private"
    assert all(i["thumbnail_url"].startswith("/api/library/") and i["tile_url"] is None for i in items.values())
    assert client.get("/api/library/sentinel2-b/thumbnail").status_code == 200
    assert client.get("/api/library/sentinel2-b/preview").status_code == 404


def test_public_download_lands_in_the_user_folder(bundle, monkeypatch):
    b, u = bundle

    @contextlib.contextmanager
    def fake_stream(method, url, **kw):
        yield httpx.Response(200, content=b"DATA:" + url.encode(), request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "stream", fake_stream)
    r = client.post("/api/library/sentinel2-b/download")
    assert r.status_code == 200 and r.json()["available"] is True
    assert (u / "tiles" / "sentinel2-b.tif").read_bytes() == b"DATA:https://x/sentinel2-b.tif"
    assert not list(u.rglob("*.part"))
    assert client.get("/api/library/sentinel2-b/preview").status_code == 200


def test_private_download_needs_a_token(bundle, monkeypatch):
    monkeypatch.setattr(library_store, "hf_token", lambda: None)
    r = client.post("/api/library/dfc2019-c/download")
    assert r.status_code == 401 and "HF_TOKEN" in r.json()["detail"]


def test_depth_by_id_reads_local_files_and_refuses_missing_ones(bundle, monkeypatch):
    from backend.api import depth_routes
    seen = {}

    async def fake_forward(filename, body, content_type):
        seen["body"] = body
        return {"ok": True}

    monkeypatch.setattr(depth_routes, "_forward", fake_forward)
    assert client.post("/api/depth/relative/library/sentinel2-a").status_code == 200 and seen["body"] == b"P"
    assert client.post("/api/depth/relative/library/sentinel2-b").status_code == 409
