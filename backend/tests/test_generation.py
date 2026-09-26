"""Real generation: assets come from the input's own elevation, or honestly none."""
import base64
import io
import json
import zlib

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_bounds

from backend.api import depth_routes
from backend.generation import pipeline
from backend.main import app
from backend.storage import library_store

client = TestClient(app)


def depth_resp():
    q = (np.arange(16, dtype="<u2") * 4000).reshape(4, 4)
    return {"encoding": "u16-zlib", "shape": [4, 4], "min": 0.5, "max": 3.5, "infer_s": 0.1, "device": "cpu",
            "data_b64": base64.b64encode(zlib.compress(q.tobytes())).decode()}


def jpeg(w=40, h=30):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 60)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DW2_GENERATED_DIR", str(tmp_path / "gen"))

    async def fake_preview(source, item_id):
        return jpeg()

    async def fake_forward(filename, body, content_type):
        return depth_resp()

    monkeypatch.setattr(depth_routes, "_preview_bytes", fake_preview)
    monkeypatch.setattr(depth_routes, "_forward", fake_forward)
    return tmp_path


def test_library_item_uses_its_elevation_pack(env, monkeypatch):
    pack = env / "pack.tif"
    tf = from_bounds(500000, 3000000, 501000, 3001000, 20, 20)
    terrain = np.linspace(100, 300, 400, dtype=np.float32).reshape(20, 20)
    with rasterio.open(pack, "w", driver="GTiff", width=20, height=20, count=2, dtype="float32",
                       crs="EPSG:32645", transform=tf) as w:
        w.write(terrain, 1)
        w.write(terrain + 10, 2)
        w.update_tags(TERRAIN_SOURCE="FABDEM test", SURFACE_SOURCE="GLO-30 test")
    item = {"id": "sentinel2-x", "collection": "sentinel2", "geo": None}
    from backend.library import catalog
    monkeypatch.setattr(catalog, "get", lambda i: item)
    monkeypatch.setattr(library_store, "local_asset", lambda it, kind: pack if kind == "dem" else None)
    r = client.post("/api/generate/library/sentinel2-x")
    assert r.status_code == 200
    body = r.json()
    m = body["meta"]
    assert m["has_elevation"] and m["terrain_source"] == "FABDEM test" and m["surface_source"] == "GLO-30 test"
    assert m["terrain_range_m"] == [100.0, 300.0] and m["surface_range_m"] == [110.0, 310.0]
    t = client.get(body["assets"]["terrain"]).json()
    assert (t["width"], t["height"]) == (20, 20) and t["elevationMin"] == pytest.approx(110, abs=1)
    for k in ("satellite", "depth", "elevation"):
        assert client.get(body["assets"][k]).headers["content-type"] == "image/png"


def test_no_georeference_means_no_elevation_and_says_so(env, monkeypatch):
    from backend.library import catalog
    monkeypatch.setattr(catalog, "get", lambda i: {"id": "dfc2019-x", "collection": "dfc2019", "geo": None})
    monkeypatch.setattr(library_store, "local_asset", lambda it, kind: None)
    m = client.post("/api/generate/library/dfc2019-x").json()["meta"]
    assert m["has_elevation"] is False and m["terrain_source"] is None and "no DEM" in m["note"]


def test_generated_assets_are_path_safe(env):
    assert client.get("/api/generated/abc/../../etc/passwd").status_code == 404
    assert client.get("/api/generated/abc123/secrets.txt").status_code == 404
