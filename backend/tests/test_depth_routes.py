import base64

import httpx
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")


def test_unconfigured_is_503(monkeypatch):
    monkeypatch.delenv("DAV2_INFERENCE_URL", raising=False)
    r = client.post("/api/depth/relative", files={"file": ("a.png", PNG, "image/png")})
    assert r.status_code == 503


def test_forwards_to_env_url_read_per_request(monkeypatch):
    seen = {}
    depth = np.arange(4, dtype="<f4").reshape(2, 2)

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"shape": [2, 2], "data_b64": base64.b64encode(depth.tobytes()).decode()})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    for host in ("https://a.trycloudflare.com/", "https://b.example"):
        monkeypatch.setenv("DAV2_INFERENCE_URL", host)
        r = client.post("/api/depth/relative", files={"file": ("a.png", PNG, "image/png")})
        assert r.status_code == 200
        assert seen["url"] == host.rstrip("/") + "/predict"
    assert r.json()["shape"] == [2, 2]


def test_unreachable_host_is_502(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("down")

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setenv("DAV2_INFERENCE_URL", "https://gone.trycloudflare.com")
    r = client.post("/api/depth/relative", files={"file": ("a.png", PNG, "image/png")})
    assert r.status_code == 502 and "Colab bridge" in r.json()["detail"]
