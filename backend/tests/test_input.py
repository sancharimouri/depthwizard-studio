"""Page 1 input routes: GSD read from the geotransform, tier routing at 2.4 m,
user-DEM validation. FABDEM (Earth Engine) is exercised separately, live."""

import io

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from backend.input import store
from backend.main import app


@pytest.fixture(autouse=True)
def _tmp_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "UPLOADS", tmp_path)


client = TestClient(app)


def _tif(px, crs="EPSG:32645", origin=(500000, 3000000), bands=3, size=64, dtype="uint8"):
    buf = io.BytesIO()
    data = (np.random.default_rng(0).random((bands, size, size)) * 200).astype(dtype)
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=size, height=size, count=bands, dtype=dtype, crs=crs,
                      transform=from_origin(origin[0], origin[1], px, px)) as dst:
            dst.write(data)
        buf.write(mem.read())
    return buf.getvalue()


def _upload(name, content):
    return client.post("/api/input/upload", files={"file": (name, content)})


def test_vhr_geotiff_routes_to_tier2_with_dem_choice():
    r = _upload("vhr.tif", _tif(0.5))
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["georeferenced"] and m["gsd_m"] == 0.5 and m["gsd_source"] == "geotransform"
    assert m["routing"]["tier"] == 2 and m["routing"]["dem_required"]
    assert client.get(m["preview_url"]).status_code == 200


def test_boundary_and_coarse():
    assert _upload("b.tif", _tif(2.4)).json()["routing"]["tier"] == 2
    coarse = _upload("s2.tif", _tif(10)).json()
    assert coarse["routing"]["tier"] == 1 and "FABDEM" in coarse["routing"]["summary"]


def test_geographic_crs_gsd_in_metres():
    # 1/3600° at 27°N ≈ 27.6 m (x) / 30.7 m (y) -> coarser axis, Tier 1
    m = _upload("geo.tif", _tif(1 / 3600, crs="EPSG:4326", origin=(88.2, 27.1))).json()
    assert 29 < m["gsd_m"] < 32 and m["routing"]["tier"] == 1


def test_png_is_relative_placeholder():
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (90, 120, 60)).save(buf, format="PNG")
    m = _upload("photo.png", buf.getvalue()).json()
    assert not m["georeferenced"] and m["gsd_m"] is None
    assert m["routing"]["tier"] is None and m["routing"]["placeholder"]
    assert client.post(f"/api/input/{m['id']}/fabdem").status_code == 400


def test_rejects_other_types():
    assert _upload("notes.txt", b"hi").status_code == 400
    assert client.get("/api/input/" + "0" * 32).status_code == 400


def test_user_dem_validation():
    m = _upload("vhr.tif", _tif(0.5)).json()  # 32 m footprint
    url = f"/api/input/{m['id']}/dem"
    # multi-band -> rejected
    assert client.post(url, files={"file": ("dem.tif", _tif(0.5))}).status_code == 400
    # single band but elsewhere -> no overlap
    far = _tif(1.0, bands=1, dtype="float32", origin=(600000, 3000000))
    r = client.post(url, files={"file": ("dem.tif", far)})
    assert r.status_code == 400 and "covers only" in r.json()["detail"]
    # covering DEM -> accepted, stats + hillshade preview
    ok = _tif(1.0, bands=1, dtype="float32", origin=(499990, 3000010))
    r = client.post(url, files={"file": ("dem.tif", ok)})
    assert r.status_code == 200, r.text
    dem = r.json()["dem"]
    assert dem["overlap_fraction"] == 1.0 and dem["valid_fraction"] == 1.0
    assert client.get(r.json()["dem_preview_url"]).status_code == 200


def test_user_dem_refused_for_tier1():
    m = _upload("s2.tif", _tif(10)).json()
    r = client.post(f"/api/input/{m['id']}/dem",
                    files={"file": ("dem.tif", _tif(10, bands=1, dtype="float32"))})
    assert r.status_code == 400


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (90, 120, 60)).save(buf, format="PNG")
    return buf.getvalue()


def test_manual_gsd_for_an_image_without_a_geotransform():
    m = _upload("photo.png", _png()).json()
    assert m["gsd_m"] is None and m["gsd_required"] is True
    r = client.post(f"/api/input/{m['id']}/gsd", json={"gsd_m": 0.5})
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["gsd_m"] == 0.5 and g["gsd_manual"] is True and g["gsd_required"] is False
    assert g["gsd_source"].startswith("entered manually")
    # still no location: relative output only, and the summary says so with the GSD
    assert g["routing"]["tier"] is None and "0.50 m/pixel" in g["routing"]["summary"]
    assert client.get(f"/api/input/{m['id']}").json()["gsd_m"] == 0.5


def test_manual_gsd_is_validated_and_refused_when_the_file_has_a_geotransform():
    m = _upload("photo.png", _png()).json()
    for bad in (0, -1, 5000, "abc"):
        assert client.post(f"/api/input/{m['id']}/gsd", json={"gsd_m": bad}).status_code in (400, 422)
    geo = _upload("vhr.tif", _tif(0.5)).json()
    r = client.post(f"/api/input/{geo['id']}/gsd", json={"gsd_m": 1.0})
    assert r.status_code == 400 and "geotransform" in r.json()["detail"]
