"""/api/facts v2: grouped lines, graceful fallback per source, bounded cache (backend/facts/sources.py)."""

import time

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.facts import lines as L
from backend.facts import sources, thinkhazard
from backend.main import app

client = TestClient(app)
BBOX = "88.2,27.0,88.3,27.1"

DISTRICT = {"code": 17948, "name": "Darjiling", "admin1": "West Bengal", "admin0": "India",
            "levels": {"WF": "H", "CY": "V", "TS": "L", "LS": "H", "EQ": "M", "FL": "V"}}
FEATURES = {"peak": {"name": "Observatory Hill", "elevation_m": 2188.0, "distance_km": 0.8},
            "river": {"name": "Bātāsi Jhora", "main": False, "length_km": None, "distance_km": 3.0}}
USGS = {"radius_km": 100, "count_since_1973": 47, "largest": {"mag": 6.9, "year": 2011, "distance_km": 77.2},
        "events": [[88.1, 27.7, 6.9, 2011]], "centre": [27.05, 88.25], "bbox": [88.2, 27.0, 88.3, 27.1]}


@pytest.fixture(autouse=True)
def fake_sources(monkeypatch):
    sources._cache.clear()
    monkeypatch.setattr(thinkhazard, "lookup", lambda lat, lon, bbox=None: DISTRICT)
    monkeypatch.setattr(sources, "wikidata_features", lambda c, lat, lon: FEATURES)
    monkeypatch.setattr(sources, "usgs", lambda c, lat, lon, bbox=None: USGS)
    monkeypatch.setattr(sources, "jrc_flood", lambda lat, lon, bbox=None: {"pct": 0.0, "max_depth_m": 0.0} if bbox else {"inside": False})
    monkeypatch.setattr(sources, "wb_landslide", lambda lat, lon, bbox=None: {"min": 3, "max": 4, "share_max_pct": 75.0} if bbox else {"class": 4})
    monkeypatch.setattr(sources, "gsw", lambda lat, lon, bbox: None)
    yield
    sources._cache.clear()


def kinds(d):
    return [x["kind"] for x in d["facts"]] + [x["kind"] for g in d["scenario"].values() for x in g]


def all_text(d):
    return " ".join(f"{x['label']} {x['text']}" for x in d["facts"] + [x for g in d["scenario"].values() for x in g]).lower()


def test_validation():
    assert client.get("/api/facts").status_code == 422
    assert client.get("/api/facts", params={"lat": 91, "lon": 0}).status_code == 422
    assert client.get("/api/facts", params={"bbox": "1,2,3"}).status_code == 422
    assert client.get("/api/facts", params={"bbox": "10,10,9,11"}).status_code == 422  # w >= e
    assert client.get("/api/facts", params={"bbox": "0,0,5,1"}).status_code == 422  # wider than 2 degrees


def test_success_bbox_groups_and_lines():
    d = client.get("/api/facts", params={"bbox": BBOX}).json()
    assert d["where"]["mode"] == "bbox" and d["where"]["district"]["name"] == "Darjiling"
    facts = {x["kind"]: x for x in d["facts"]}
    assert facts["wildfire"]["text"] == "HIGH (district)"
    assert "cyclone" not in facts  # VERY LOW: does not apply
    assert facts["tsunami"]["text"] == "LOW (district)"
    assert facts["peak"]["text"] == "Observatory Hill · 2,188 m · 0.8 km"
    assert facts["river"]["label"] == "Nearest river"  # chosen by distance, not length: never called "main"
    sc = d["scenario"]
    assert [x["kind"] for x in sc["flood"]] == ["rp100", "flood_district"]
    assert sc["flood"][0]["text"] == "none modelled in this tile"
    assert sc["landslide"][0]["text"] == "class 3–4 of 4 · 75% class 4"
    eq = {x["kind"]: x for x in sc["earthquake"]}
    assert eq["usgs_count"]["label"] == "M4.5+ within 100 km" and eq["usgs_count"]["text"] == "47 since 1973"
    assert eq["usgs_largest"]["text"] == "M6.9 · 2011 · 77 km away"
    assert eq["epicentres"]["data"]["events"] == [[88.1, 27.7, 6.9, 2011]]
    # no elevation/slope, no sources or licences in the lines
    assert not any(w in all_text(d) for w in ("elevation", "slope", "source", "licence", "not available"))
    assert all(v == "ok" or v == "empty" for v in d["status"].values())


def test_point_mode_uses_point_lookups_only():
    d = client.get("/api/facts", params={"lat": 27.05, "lon": 88.25}).json()
    assert d["where"]["mode"] == "point" and "gsw" not in d["status"]
    assert d["scenario"]["flood"][0]["text"] == "outside the modelled extent"
    assert d["scenario"]["landslide"][0]["text"] == "class 4 of 4"


@pytest.mark.parametrize("name", ["wikidata_features", "usgs", "jrc_flood", "wb_landslide"])
def test_one_source_failing_leaves_the_rest(monkeypatch, name):
    def boom(*_a, **_k):
        raise ConnectionError("offline")

    monkeypatch.setattr(sources, name, boom)
    d = client.get("/api/facts", params={"bbox": BBOX}).json()
    status_key = {"wikidata_features": "wikidata"}.get(name, name)
    assert d["status"][status_key] == "failed"
    assert "offline" not in str(d)  # no raw error text anywhere in the response
    assert "wildfire" in kinds(d) and "earthquake_district" in kinds(d)  # the rest still show
    assert len(kinds(d)) > 3


def test_http_timeout_is_reported_as_timeout(monkeypatch):
    def slow(*_a, **_k):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(sources, "usgs", slow)
    d = client.get("/api/facts", params={"bbox": BBOX}).json()
    assert d["status"]["usgs"] == "timeout"
    assert not d["scenario"]["earthquake"] or all(x["kind"] == "earthquake_district" for x in d["scenario"]["earthquake"])


def test_source_past_the_deadline_is_dropped(monkeypatch):
    monkeypatch.setattr(sources, "DEADLINE_S", 0.3)
    monkeypatch.setattr(sources, "wb_landslide", lambda *a, **k: time.sleep(2) or {"class": 1})
    t0 = time.time()
    d = client.get("/api/facts", params={"bbox": BBOX}).json()
    assert time.time() - t0 < 1.5  # never blocks on the slow source
    assert d["status"]["wb_landslide"] == "timeout"
    assert "wb_class" not in kinds(d) and "peak" in kinds(d)


def test_missing_district_bundle_omits_district_lines(monkeypatch):
    monkeypatch.setattr(thinkhazard, "lookup", lambda lat, lon, bbox=None: None)
    d = client.get("/api/facts", params={"bbox": BBOX}).json()
    assert d["where"]["district"] is None and d["status"]["thinkhazard"] == "empty"
    assert not {"wildfire", "landslide_district", "earthquake_district"} & set(kinds(d))
    assert "peak" in kinds(d)


def test_empty_values_give_no_lines():
    assert L.feature_lines({}) == [] and L.usgs_lines(None) == [] and L.gsw_lines({"pct_ever": 0.2, "pct_permanent": 0}) == []
    assert L.thinkhazard_lines(None) == [] and L.wb_landslide_lines(None) == [] and L.jrc_flood_lines(None) == []


def test_cache_is_bounded_and_partial_results_expire_sooner(monkeypatch):
    monkeypatch.setattr(sources, "CACHE_MAX", 3)
    for i in range(5):
        client.get("/api/facts", params={"lat": 10 + i, "lon": 20})
    assert len(sources._cache) == 3
    assert ("point", 10.0, 20.0) not in sources._cache  # oldest evicted

    def boom(*_a, **_k):
        raise ConnectionError("x")

    monkeypatch.setattr(sources, "usgs", boom)
    client.get("/api/facts", params={"lat": 1, "lon": 1})
    expires, _ = sources._cache[("point", 1.0, 1.0)]
    assert expires - time.time() <= sources.CACHE_PARTIAL_S + 1


def test_bundled_thinkhazard_lookup_real_file():
    """The real bundle (skipped if not built): Darjeeling resolves to Darjiling with a landslide level."""
    from importlib import reload

    th = reload(thinkhazard)
    if not (th.DATA / "thinkhazard_levels.json").is_file():
        pytest.skip("bundle not built")
    d = th.lookup(27.045, 88.26)
    assert d and d["code"] == 17948 and d["levels"].get("LS") == "H"
    assert th.lookup(0.0, -30.0) is None  # open Atlantic: no division
    # a coastal tile whose centre is in the sea takes the division covering most of its box (Chennai)
    assert th.lookup(13.05, 80.28) is None
    assert th.lookup(13.05, 80.28, [80.23, 13.005, 80.33, 13.095])["name"] == "Chennai"
