"""Curated library endpoints (requires data/library/manifest.json:
python scripts/library_catalog.py curate && python scripts/library_catalog.py build)."""

import pytest
from fastapi.testclient import TestClient

from backend.library import catalog
from backend.main import app

client = TestClient(app)

pytestmark = pytest.mark.skipif(not catalog.MANIFEST.exists(), reason="library manifest not generated")


@pytest.fixture(autouse=True)
def local_library(monkeypatch):
    """These tests exercise the catalog/routing rules against the local manifest;
    the remote stores are covered in test_library_store.py (no network)."""
    monkeypatch.setenv("DW2_LIBRARY", "local")


def test_list_has_exactly_the_curated_sets():
    r = client.get("/api/library")
    assert r.status_code == 200
    d = r.json()
    assert d["counts"] == {"dfc2019": 50, "sentinel2": 33, "vhr": 6}  # 32 benchmark + Darjeeling (demo scene)
    assert d["total"] == 89
    ids = [i["id"] for i in d["items"]]
    assert len(set(ids)) == 89
    assert ids[:3] == ["sentinel2-darjeeling", "sentinel2-almora", "sentinel2-manali"]
    assert {i["terrain"] for i in d["items"]} == {"hilly", "agricultural", "urban", "coastal"}
    # research artifacts are never exposed
    blob = r.text.lower()
    for marker in ("landsat", "cbers", "l1c", "brazil"):
        assert marker not in blob


def test_filters():
    assert client.get("/api/library?collection=vhr").json()["total"] == 6
    assert client.get("/api/library?tier=1").json()["total"] == 33
    assert client.get("/api/library?tier=2").json()["total"] == 56
    assert client.get("/api/library?collection=landsat").status_code == 422
    assert client.get("/api/library?tier=3").status_code == 422


def test_item_metadata_and_routing():
    s2 = client.get("/api/library/sentinel2-almora").json()
    assert s2["gsd_m"] == 10.0 and s2["gsd_source"] == "geotransform"
    assert s2["routing"]["tier"] == 1 and s2["routing"]["locked"] is True
    assert s2["geo"]["lat"] and s2["geo"]["lon"]
    vhr = client.get("/api/library/vhr-c_town").json()
    assert abs(vhr["gsd_m"] - 0.305) < 0.001 and vhr["routing"]["tier"] == 2 and vhr["routing"]["dem"] == "FABDEM"
    dfc = client.get("/api/library/dfc2019-JAX_004_006").json()
    assert dfc["gsd_m"] == 0.3 and dfc["georeferenced"] is False and dfc["routing"]["tier"] == 2
    assert dfc["location"].startswith("Jacksonville")
    assert "file" not in dfc  # no local paths leak to the client


def test_images_are_real_jpegs():
    for item_id in ("dfc2019-OMA_376_038", "sentinel2-mumbai", "vhr-b_glacier"):
        for kind in ("thumbnail", "preview"):
            r = client.get(f"/api/library/{item_id}/{kind}")
            assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
            assert r.content[:3] == b"\xff\xd8\xff" and len(r.content) > 2000


def test_unknown_and_traversal_ids_404():
    assert client.get("/api/library/nope").status_code == 404
    assert client.get("/api/library/..%2F..%2F.env/thumbnail").status_code == 404


def test_select_enforces_sentinel2_tier1_regardless_of_request():
    r = client.post("/api/library/sentinel2-mumbai/select", json={"requested_tier": 2}).json()
    assert r["tier"] == 1 and r["routing"]["dem"] == "FABDEM"
    assert r["notes"] and "locked" in r["notes"][0]


def test_select_tier2_items():
    assert client.post("/api/library/vhr-c_town/select").json()["tier"] == 2
    down = client.post("/api/library/vhr-c_town/select", json={"requested_tier": 1}).json()
    assert down["tier"] == 1 and down["routing"]["label"].startswith("Tier 1")
    dfc = client.post("/api/library/dfc2019-JAX_004_006/select", json={"requested_tier": 1}).json()
    assert dfc["tier"] == 2 and "no georeference" in dfc["notes"][0]
