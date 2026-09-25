"""R2 storage: key layout, R2-served catalog URLs, private presigned URLs, and the
local-dev fallback. No network: the R2 manifest fetch is monkeypatched."""

import datetime as dt
import importlib.util
import json
import urllib.parse as up

import pytest
from fastapi.testclient import TestClient

from backend.library import catalog
from backend.main import app
from backend.storage import r2

client = TestClient(app)
needs_manifest = pytest.mark.skipif(not catalog.MANIFEST.exists(), reason="library manifest not generated")

R2_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_PUBLIC_BASE_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in R2_VARS:
        monkeypatch.delenv(k, raising=False)
    catalog._r2_cache.update(at=0.0, data=None)
    yield
    catalog._r2_cache.update(at=0.0, data=None)


def test_key_layout():
    item = {"id": "vhr-a_forest", "collection": "vhr", "tile_id": "a_forest"}
    assert r2.library_keys(item) == {
        "tile": "library/tiles/vhr/a_forest.tif",
        "thumbnail": "library/thumbnails/vhr-a_forest.jpg",
        "preview": "library/previews/vhr-a_forest.jpg",
    }
    assert len(r2.CHECKPOINTS) == 5
    assert all(k.startswith("checkpoints/method6/") for k in r2.CHECKPOINTS)


@needs_manifest
def test_unconfigured_serves_local(monkeypatch):
    assert not r2.configured()
    item = client.get("/api/library").json()["items"][0]
    assert item["preview_url"].startswith("/api/library/")
    assert "tile_url" not in item


@needs_manifest
def test_public_bucket_urls_and_r2_manifest(monkeypatch):
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://assets.example.org/")
    fetched = []
    monkeypatch.setattr(r2, "fetch_bytes", lambda key, timeout=30.0: fetched.append(key) or catalog.MANIFEST.read_bytes())
    d = client.get("/api/library").json()
    assert fetched == [r2.MANIFEST_KEY]  # the catalog comes from R2, not the local file
    item = d["items"][0]
    keys = r2.library_keys(item)
    assert item["preview_url"] == f"https://assets.example.org/{keys['preview']}"
    assert item["thumbnail_url"] == f"https://assets.example.org/{keys['thumbnail']}"
    assert item["tile_url"] == f"https://assets.example.org/{keys['tile']}"
    for k in ("file", "r2", "thumbnail", "preview"):
        assert k not in item
    r = client.get(f"/api/library/{item['id']}/preview", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == item["preview_url"]


@needs_manifest
def test_private_bucket_presigned(monkeypatch):
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(r2, "fetch_bytes", lambda key, timeout=30.0: catalog.MANIFEST.read_bytes())
    item = client.get("/api/library").json()["items"][0]
    # stable route (saved jobs keep it) that redirects to a fresh presigned URL
    assert item["preview_url"] == f"/api/library/{item['id']}/preview"
    r = client.get(item["preview_url"], follow_redirects=False)
    assert r.status_code == 307
    url = r.headers["location"]
    assert up.urlparse(item["tile_url"]).netloc == "acct.r2.cloudflarestorage.com"
    parts = up.urlparse(url)
    q = dict(up.parse_qsl(parts.query))
    assert parts.netloc == "acct.r2.cloudflarestorage.com"
    assert parts.path.startswith("/depthwizard2/library/previews/")
    assert q["X-Amz-Algorithm"] == "AWS4-HMAC-SHA256" and len(q["X-Amz-Signature"]) == 64
    assert "secret" not in url


def test_r2_unreachable_is_an_error_not_a_local_fallback(monkeypatch):
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://assets.example.org")

    def boom(key, timeout=30.0):
        raise ConnectionError("offline")

    monkeypatch.setattr(r2, "fetch_bytes", boom)
    assert client.get("/api/library").status_code == 503


@pytest.mark.skipif(importlib.util.find_spec("boto3") is None, reason="boto3 not installed (cross-check only)")
def test_presign_matches_boto3(monkeypatch):
    import boto3
    from botocore.config import Config

    monkeypatch.setenv("R2_ACCOUNT_ID", "abc123")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY")
    key = "library/previews/dfc2019-JAX_004_006.jpg"
    s3 = boto3.client("s3", endpoint_url="https://abc123.r2.cloudflarestorage.com", region_name="auto",
                      aws_access_key_id="AKIDEXAMPLE", aws_secret_access_key="wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
    theirs = s3.generate_presigned_url("get_object", Params={"Bucket": "depthwizard2", "Key": key}, ExpiresIn=3600)
    tq = dict(up.parse_qsl(up.urlparse(theirs).query))
    now = dt.datetime.strptime(tq["X-Amz-Date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    ours = dict(up.parse_qsl(up.urlparse(r2.presign_get(key, 3600, now)).query))
    assert ours["X-Amz-Signature"] == tq["X-Amz-Signature"]
