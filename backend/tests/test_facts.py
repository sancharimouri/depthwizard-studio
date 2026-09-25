"""Facts route: validation, and that failures/absent sources are reported, never filled in."""

from fastapi.testclient import TestClient

from backend.facts import sources
from backend.main import app

client = TestClient(app)


def test_rejects_out_of_range_coordinates():
    assert client.get("/api/facts", params={"lat": 91, "lon": 0}).status_code == 422
    assert client.get("/api/facts", params={"lat": 0}).status_code == 422


def test_unreachable_sources_are_errors_not_data(monkeypatch):
    def boom(*_a, **_k):
        raise ConnectionError("offline")

    monkeypatch.setattr(sources, "_get", boom)
    monkeypatch.setattr(sources, "_gdacs_floods", None)
    sources._cache.clear()
    d = client.get("/api/facts", params={"lat": 10.123, "lon": 20.456}).json()
    for k in ("place", "elevation", "seismic", "floods"):
        assert d[k]["status"] == "error", k
        assert set(d[k]) == {"status", "source", "url", "message"}
    assert d["landslides"]["status"] == "unavailable" and d["landslides"]["reason"]
    assert d["volcanoes"]["status"] == "unavailable"
    assert (10.123, 20.456) not in sources._cache  # transient failures aren't cached


def test_haversine():
    assert abs(sources._haversine_km(27.04, 88.26, 26.8, 86.7) - 157) < 5
