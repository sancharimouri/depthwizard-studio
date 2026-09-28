import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.api import depth_routes
from backend.main import app

client = TestClient(app)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
DEPTH = {"encoding": "u16-zlib", "shape": [1, 1], "min": 0.0, "max": 1.0, "data_b64": ""}


def post(path="/api/depth/relative"):
    return client.post(path, files={"file": ("a.png", PNG, "image/png")})


def mock_http(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("DAV2_INFERENCE_URL", raising=False)
    monkeypatch.delenv("DAV2_FALLBACK_URL", raising=False)
    depth_routes._clients.clear()


def test_space_ids_vs_plain_urls():
    assert depth_routes.space_id("sancharimouri/DepthWizard2") == "sancharimouri/DepthWizard2"
    assert depth_routes.space_id("https://huggingface.co/spaces/sancharimouri/DepthWizard2") == "sancharimouri/DepthWizard2"
    assert depth_routes.space_id("https://x.trycloudflare.com") is None
    assert depth_routes.space_id("http://127.0.0.1:8020") is None


def test_default_primary_is_the_space_and_fallback_is_opt_in(monkeypatch):
    assert depth_routes.hosts() == ["sancharimouri/DepthWizard2"]
    monkeypatch.setenv("DAV2_FALLBACK_URL", "https://x.trycloudflare.com/")
    assert depth_routes.hosts() == ["sancharimouri/DepthWizard2", "https://x.trycloudflare.com"]


def test_space_client_always_sends_the_token_as_a_header(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, src, **kw):
            seen.update(src=src, **kw)

        def predict(self, *a, **kw):
            return DEPTH

    import gradio_client
    monkeypatch.setattr(gradio_client, "Client", FakeClient)
    monkeypatch.setattr(depth_routes, "hf_token", lambda: "hf_test")
    r = post()
    assert r.status_code == 200 and r.json()["host"] == "sancharimouri/DepthWizard2" and r.json()["fallback_used"] is False
    assert seen["token"] == "hf_test" and seen["headers"] == {"Authorization": "Bearer hf_test"}


def test_no_token_means_no_anonymous_space_call(monkeypatch):
    monkeypatch.setattr(depth_routes, "hf_token", lambda: None)
    r = post()
    assert r.status_code == 503 and "HF_TOKEN" in r.json()["detail"]


def test_plain_http_host_still_works_colab_bridge(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=DEPTH)

    mock_http(monkeypatch, handler)
    monkeypatch.setenv("DAV2_INFERENCE_URL", "https://a.trycloudflare.com/")
    r = post()
    assert r.status_code == 200 and seen["url"] == "https://a.trycloudflare.com/predict"
    assert r.json()["host"] == "https://a.trycloudflare.com"


def test_quota_on_the_space_falls_back_to_the_explicit_bridge(monkeypatch):
    def quota(*a):
        raise RuntimeError("You have exceeded your GPU quota (30s requested vs. 12s left).")

    monkeypatch.setattr(depth_routes, "_space_predict", quota)
    mock_http(monkeypatch, lambda request: httpx.Response(200, json=DEPTH))
    monkeypatch.setenv("DAV2_FALLBACK_URL", "https://b.trycloudflare.com")
    r = post()
    assert r.status_code == 200 and r.json()["fallback_used"] is True and r.json()["host"] == "https://b.trycloudflare.com"


def test_quota_without_fallback_is_a_429_with_the_real_message(monkeypatch):
    monkeypatch.setattr(depth_routes, "_space_predict",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("You have exceeded your GPU quota")))
    r = post()
    assert r.status_code == 429 and "GPU quota" in r.json()["detail"]


def test_unreachable_bridge_is_502(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("down")

    mock_http(monkeypatch, handler)
    monkeypatch.setenv("DAV2_INFERENCE_URL", "https://gone.trycloudflare.com")
    r = post()
    assert r.status_code == 502 and "Colab bridge" in r.json()["detail"]


def test_by_id_forwards_the_input_preview(monkeypatch, tmp_path):
    from backend.input import store
    jpg = tmp_path / "preview.jpg"
    jpg.write_bytes(b"JPEGBYTES")
    monkeypatch.setattr(store, "preview_path", lambda i, kind="image": jpg)
    seen = {}

    def handler(request):
        seen["body"] = request.content
        return httpx.Response(200, json=DEPTH)

    mock_http(monkeypatch, handler)
    monkeypatch.setenv("DAV2_INFERENCE_URL", "https://x.trycloudflare.com")
    r = client.post("/api/depth/relative/input/abc123")
    assert r.status_code == 200 and b"JPEGBYTES" in seen["body"]
    assert client.post("/api/depth/relative/nope/abc").status_code == 404


def test_audit_counts_auth_on_hf_space_requests_only():
    before = depth_routes.auth_audit()
    depth_routes._record(httpx.Request("GET", "https://x-y.hf.space/config", headers={"Authorization": "Bearer hf_x"}))
    depth_routes._record(httpx.Request("GET", "https://x-y.hf.space/config"))
    depth_routes._record(httpx.Request("GET", "https://example.com/"))
    after = depth_routes.auth_audit()
    assert after["with_auth"] - before["with_auth"] == 1 and after["without_auth"] - before["without_auth"] == 1


def test_transient_space_502_is_retried_with_a_fresh_client(monkeypatch):
    # The Space's Gradio SSR proxy answers 502 while its Python app restarts
    # (seen 2026-09-28): a blip must not fail the job.
    made, calls = [], []

    class FlakyClient:
        def __init__(self, src, **kw):
            made.append(src)

        def predict(self, *a, **kw):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("Server error '502 Bad Gateway' for url 'https://x.hf.space/gradio_api/upload'")
            return DEPTH

    import gradio_client
    monkeypatch.setattr(gradio_client, "Client", FlakyClient)
    monkeypatch.setattr(depth_routes, "hf_token", lambda: "hf_test")
    monkeypatch.setattr(depth_routes, "SPACE_RETRY_DELAYS_S", (0, 0))
    r = post()
    assert r.status_code == 200 and len(calls) == 3
    assert len(made) == 3  # the cached client is dropped before each retry


def test_quota_is_not_retried(monkeypatch):
    calls = []

    def quota(*a):
        calls.append(1)
        raise RuntimeError("You have exceeded your GPU quota")

    monkeypatch.setattr(depth_routes, "_space_predict", quota)
    monkeypatch.setattr(depth_routes, "SPACE_RETRY_DELAYS_S", (0, 0))
    r = post()
    assert r.status_code == 429 and len(calls) == 1
