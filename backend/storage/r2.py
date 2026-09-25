"""Cloudflare R2 storage for the tile library and the Method 6 checkpoints.

Nothing here holds a credential. Configuration comes from the environment
(see docs/R2_SETUP.md and .env.example):

  R2_ACCOUNT_ID          Cloudflare account id (for the S3 endpoint)
  R2_ACCESS_KEY_ID       R2 API token: access key id
  R2_SECRET_ACCESS_KEY   R2 API token: secret access key
  R2_BUCKET              bucket name (default: depthwizard2)
  R2_PUBLIC_BASE_URL     optional: public base URL (r2.dev or custom domain).
                         Leave UNSET to keep the bucket private: the backend then
                         hands out short-lived presigned GET URLs instead.
  R2_URL_TTL             presigned URL lifetime in seconds (default 3600)

Bucket layout (keys):
  library/manifest.json
  library/tiles/{collection}/{tile_id}.tif
  library/thumbnails/{item_id}.jpg
  library/previews/{item_id}.jpg
  checkpoints/method6/height_balanced_seed43/fold{0..3}.pt
  checkpoints/method6/full_dfc2019/method6_full_dfc2019.pt

When R2 is configured, the library is served only from R2 (manifest, images,
tile links); the local data/library files are just the one-time upload
source. When it is not configured, the backend falls back to the local files
so development without an R2 account keeps working.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import os
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_KEY = "library/manifest.json"

# The Method 6 checkpoints that code actually loads: scripts/vhr_dsm_pipeline.py
# (the only Method 6 consumer; the product itself is DEM-only, docs/HANDOFF.md)
# averages the four seed-43 height-balanced fold checkpoints and cross-checks
# against the full-data checkpoint.
CHECKPOINTS = {
    **{f"checkpoints/method6/height_balanced_seed43/fold{q}.pt":
       f"data/dfc2019/experiments/method6_height_balanced_seed43/fold{q}.pt" for q in range(4)},
    "checkpoints/method6/full_dfc2019/method6_full_dfc2019.pt":
        "data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019.pt",
}

CACHE_DIR = Path(os.environ.get("DW2_CACHE_DIR", Path.home() / ".cache" / "depthwizard2"))


# --------------------------------------------------------------------------- config
def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


def bucket() -> str:
    return _env("R2_BUCKET", "depthwizard2")


def public_base() -> str | None:
    base = _env("R2_PUBLIC_BASE_URL")
    return base.rstrip("/") if base else None


def has_credentials() -> bool:
    return all(_env(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"))


def configured() -> bool:
    """R2 is the source of truth once either a public base or API credentials exist."""
    return bool(public_base()) or has_credentials()


def endpoint() -> str:
    return f"https://{_env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"


# --------------------------------------------------------------------------- keys
def library_keys(item: dict) -> dict:
    """R2 keys for one catalog item (deterministic, so the manifest and the
    uploader always agree)."""
    return {
        "tile": f"library/tiles/{item['collection']}/{item['tile_id']}.tif",
        "thumbnail": f"library/thumbnails/{item['id']}.jpg",
        "preview": f"library/previews/{item['id']}.jpg",
    }


# --------------------------------------------------------------------------- URLs
def presign_get(key: str, expires: int | None = None, now: _dt.datetime | None = None) -> str:
    """AWS SigV4 query-string presigned GET for R2's S3 API (region 'auto').
    Implemented directly so the backend needs no boto3."""
    expires = int(expires or _env("R2_URL_TTL", "3600"))
    now = now or _dt.datetime.now(_dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    region, service = "auto", "s3"
    host = f"{_env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
    canonical_uri = "/" + quote(f"{bucket()}/{key}", safe="/~")
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    params = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{_env('R2_ACCESS_KEY_ID')}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    query = "&".join(f"{quote(k, safe='~')}={quote(v, safe='~')}" for k, v in sorted(params.items()))
    canonical_request = "\n".join(["GET", canonical_uri, query, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"])
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])

    def _sign(k: bytes, msg: str) -> bytes:
        return hmac.new(k, msg.encode(), hashlib.sha256).digest()

    k = _sign(("AWS4" + _env("R2_SECRET_ACCESS_KEY")).encode(), datestamp)
    for part in (region, service, "aws4_request"):
        k = _sign(k, part)
    signature = hmac.new(k, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return f"https://{host}{canonical_uri}?{query}&X-Amz-Signature={signature}"


def object_url(key: str) -> str | None:
    """Browser-usable URL for an object: the public URL if the bucket is
    public, else a short-lived presigned URL, else None (R2 not configured)."""
    base = public_base()
    if base:
        return f"{base}/{quote(key, safe='/~')}"
    if has_credentials():
        return presign_get(key)
    return None


# --------------------------------------------------------------------------- reads
def fetch_bytes(key: str, timeout: float = 30.0) -> bytes:
    url = object_url(key)
    if url is None:
        raise RuntimeError("R2 is not configured (see docs/R2_SETUP.md).")
    r = httpx.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.content


def ensure_local(key: str, local: Path | None = None) -> Path:
    """A local file for `key`: the given path if it exists, else a copy
    downloaded once from R2 into the cache dir. Lets large assets (the
    checkpoints) live only in R2 on machines that don't have them."""
    if local is not None and local.exists():
        return local
    dest = CACHE_DIR / key
    if dest.exists():
        return dest
    url = object_url(key)
    if url is None:
        raise FileNotFoundError(f"{local or key} is not on this machine and R2 is not configured (docs/R2_SETUP.md).")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest
