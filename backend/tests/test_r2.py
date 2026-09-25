"""Dormant Cloudflare R2 module (backend/storage/r2.py): key layout and the SigV4
presigner. The active library path is backend/storage/library_store.py."""

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
    yield


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
@needs_manifest
@needs_manifest
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
