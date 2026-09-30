"""DW2_PRIVATE_LIBRARY=off (the Cloud Run image): the private HF dataset is never read, so the HF cache is never written;
every private-library route answers 404 (not 500/503). The desktop sidecar (flag unset) keeps the private library."""

import pytest
from fastapi.testclient import TestClient

from backend.library import catalog
from backend.main import app
from backend.storage import library_store

client = TestClient(app)


@pytest.fixture
def off(monkeypatch, tmp_path):
    monkeypatch.setenv("DW2_PRIVATE_LIBRARY", "off")
    monkeypatch.setenv("DW2_LIBRARY", "remote")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    library_store._memo.update(at=0.0, data=None)
    calls = []

    def spy(*a, **k):  # any download attempt would land here
        calls.append((a, k))
        raise AssertionError("hf_hub_download must not be called")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", spy)
    yield calls, tmp_path / "hf"
    library_store._memo.update(at=0.0, data=None)


def test_hub_file_refuses_and_never_downloads(off):
    calls, hf = off
    with pytest.raises(library_store.PrivateLibraryDisabled):
        library_store._hub_file("manifest.json")
    with pytest.raises(library_store.PrivateLibraryDisabled):
        library_store.private_file({"id": "dfc2019-JAX_004_006", "collection": "dfc2019"}, "preview")
    with pytest.raises(library_store.PrivateLibraryDisabled):
        catalog.load()
    assert calls == [] and not hf.exists()  # nothing downloaded, the HF cache never created


def test_library_routes_answer_404_not_5xx(off):
    calls, hf = off
    for method, url in [("get", "/api/library"), ("get", "/api/library/dfc2019-JAX_004_006/preview"),
                        ("get", "/api/library/dfc2019-JAX_004_006/thumbnail"),
                        ("post", "/api/generate/library/dfc2019-JAX_004_006")]:
        r = getattr(client, method)(url)
        assert r.status_code == 404, (url, r.status_code, r.text)
        assert "statically" in r.json()["detail"]
    assert calls == [] and not hf.exists()


def test_default_is_on_for_the_desktop(monkeypatch):
    monkeypatch.delenv("DW2_PRIVATE_LIBRARY", raising=False)
    assert library_store.private_library_enabled() is True
    for v in ("off", "0", "false", "OFF"):
        monkeypatch.setenv("DW2_PRIVATE_LIBRARY", v)
        assert library_store.private_library_enabled() is False
