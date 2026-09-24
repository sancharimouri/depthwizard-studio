"""Copernicus Data Space Ecosystem (CDSE) client.

Handles OAuth2 client-credentials auth, Sentinel Hub Catalog (STAC) search,
and Sentinel Hub Process API true-color previews. Credentials are read from
CDSE_CLIENT_ID / CDSE_CLIENT_SECRET env vars and never leave this process —
the frontend only ever talks to our own backend routes.

Endpoint sources (verified against Copernicus documentation, not guessed):
- Token: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html
- Catalog: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html
- Process: https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request_cdse.html
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time

import httpx

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_SEARCH_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

COLLECTION = "sentinel-2-l2a"

TRUE_COLOR_EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{ bands: ["B04", "B03", "B02"] }],
        output: { bands: 3, sampleType: "AUTO" }
    };
}
function evaluatePixel(sample) {
    return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
"""


class CDSEAuthError(RuntimeError):
    """Credentials missing or the token endpoint rejected them."""


class CDSEUpstreamError(RuntimeError):
    """The Catalog or Process API returned an error."""


class CDSEThrottled(RuntimeError):
    """HTTP 429 from CDSE: the account's rate/volume limit was hit."""

    def __init__(self, message: str, retry_after: str | None):
        super().__init__(message)
        self.retry_after = retry_after


# Documented limits (not a live balance: CDSE exposes no "remaining quota"
# endpoint or header). Source, fetched 2026-09-25:
# https://documentation.dataspace.copernicus.eu/Quotas.html — Sentinel Hub
# APIs, "Copernicus General User".
DOCUMENTED_LIMITS = {
    "source": "https://documentation.dataspace.copernicus.eu/Quotas.html",
    "checked": "2026-09-25",
    "typology": "Copernicus General User",
    "requests_per_minute": 300,
    "requests_per_month": 10_000,
    "processing_units_per_minute": 300,
    "processing_units_per_month": 10_000,
    "reset": "monthly quotas reset on the 1st of each month",
    "mechanism": "volume-based throttling: once a limit is reached, requests get HTTP 429 until the window frees up",
}

# Token cache keyed by a hash of the credential pair — the project's own
# (from .env) or a user's own key sent per request. Secrets are never stored
# in plain text or logged, and live only in this process's memory.
_tokens: dict[str, dict] = {}
# What THIS server has spent since it started, per credential key. This is
# local accounting from the x-processingunits-spent response header, not the
# account's balance (other clients of the same account aren't visible).
_usage: dict[str, dict] = {}
_lock = threading.Lock()


def _key(client_id: str, client_secret: str) -> str:
    return hashlib.sha256(f"{client_id}\0{client_secret}".encode()).hexdigest()[:16]


def _get_credentials(creds: tuple[str, str] | None = None) -> tuple[str, str]:
    if creds and creds[0] and creds[1]:
        return creds
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise CDSEAuthError(
            "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not set in the environment"
        )
    return client_id, client_secret


def _usage_for(key: str) -> dict:
    return _usage.setdefault(key, {
        "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "catalog_requests": 0, "process_requests": 0, "processing_units": 0.0,
        "throttled": 0, "last_throttle": None, "retry_after": None,
    })


def _record(key: str, response: httpx.Response, kind: str) -> None:
    with _lock:
        u = _usage_for(key)
        u[f"{kind}_requests"] += 1
        pu = response.headers.get("x-processingunits-spent")
        if pu:
            try:
                u["processing_units"] = round(u["processing_units"] + float(pu), 4)
            except ValueError:
                pass
        if response.status_code == 429:
            u["throttled"] += 1
            u["last_throttle"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            u["retry_after"] = response.headers.get("retry-after")


def _claims(token: str) -> dict:
    try:
        part = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except Exception:  # noqa: BLE001 - claims are informational only
        return {}


def get_access_token(creds: tuple[str, str] | None = None) -> str:
    """Return a cached CDSE access token for these credentials (the project's
    own when `creds` is None), refreshing it if it's expired or missing."""
    client_id, client_secret = _get_credentials(creds)
    key = _key(client_id, client_secret)
    now = time.time()
    cached = _tokens.get(key)
    if cached and now < cached["expires_at"]:
        return cached["access_token"]

    try:
        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise CDSEAuthError(f"CDSE token request failed: {exc}") from exc

    payload = response.json()
    token = payload["access_token"]
    # Refresh a bit early (60s margin) rather than cutting it exactly at expiry.
    expires_in = payload.get("expires_in", 300)
    _tokens[key] = {"access_token": token, "expires_at": now + max(expires_in - 60, 30)}

    return token


def quota_status(creds: tuple[str, str] | None = None) -> dict:
    """Documented limits + account typology (from the token's claims) + what
    this server has spent. Deliberately no 'remaining' figure: CDSE doesn't
    publish one, so any such number would be invented."""
    client_id, client_secret = _get_credentials(creds)
    key = _key(client_id, client_secret)
    claims = _claims(get_access_token((client_id, client_secret)))
    roles = sorted(r for r in claims.get("realm_access", {}).get("roles", []) if "quota" in r)
    groups = [g for v in claims.values() if isinstance(v, list) for g in v
              if isinstance(g, str) and "user_typology" in g]
    typology = groups[0].rstrip("/").rsplit("/", 1)[-1] if groups else None
    with _lock:
        usage = dict(_usage_for(key))
    return {
        "credentials": "user" if creds and creds[0] else "project",
        "account": {"quota_roles": roles, "typology": typology},
        "documented_limits": DOCUMENTED_LIMITS,
        "this_server_usage": usage,
        "remaining": None,
        "remaining_note": "CDSE does not expose a remaining-quota figure; only the documented limits and "
                          "per-request processing-unit cost are known.",
    }


def _raise_for(response: httpx.Response, what: str) -> None:
    if response.status_code == 429:
        raise CDSEThrottled(f"CDSE {what}: rate/volume limit reached (HTTP 429)",
                            response.headers.get("retry-after"))
    if response.is_error:
        raise CDSEUpstreamError(f"CDSE {what} failed: HTTP {response.status_code} {response.text[:500]}")


def _bbox_from_point(lat: float, lon: float, aoi_km: float) -> list[float]:
    """A square bbox of aoi_km x aoi_km centered on (lat, lon), in WGS84 degrees."""
    half_km = aoi_km / 2
    lat_deg = half_km / 111.0
    lon_deg = half_km / (111.0 * max(0.1, abs(_cos_deg(lat))))
    return [lon - lon_deg, lat - lat_deg, lon + lon_deg, lat + lat_deg]


def _cos_deg(deg: float) -> float:
    import math

    return math.cos(math.radians(deg))


def search_scenes(
    lat: float,
    lon: float,
    aoi_km: float,
    date_from: str,
    date_to: str,
    max_cloud: float,
    limit: int = 10,
    creds: tuple[str, str] | None = None,
) -> dict:
    """Query the Sentinel Hub Catalog (STAC) for Sentinel-2 L2A scenes near (lat, lon)."""
    token = get_access_token(creds)
    key = _key(*_get_credentials(creds))
    bbox = _bbox_from_point(lat, lon, aoi_km)

    body = {
        "bbox": bbox,
        "datetime": f"{date_from}T00:00:00Z/{date_to}T23:59:59Z",
        "collections": [COLLECTION],
        "limit": limit,
        # cql2-json's {"op": ..., "args": [...]} form is documented but the
        # catalog rejects it ("Cannot parse parameter `filter`") as of this
        # writing; the cql2-text string form works, so use that instead.
        "filter": f"eo:cloud_cover < {max_cloud}",
    }

    try:
        response = httpx.post(
            CATALOG_SEARCH_URL,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise CDSEUpstreamError(f"CDSE catalog search failed: {exc}") from exc
    _record(key, response, "catalog")
    _raise_for(response, "catalog search")

    data = response.json()
    features = data.get("features", [])

    scenes = []
    for feature in features:
        props = feature.get("properties", {})
        scenes.append(
            {
                "id": feature.get("id"),
                "date": (props.get("datetime") or "")[:10],
                "cloud": round(props.get("eo:cloud_cover", 0), 1),
                "bbox": feature.get("bbox"),
            }
        )

    return {"bbox": bbox, "scenes": scenes}


def fetch_true_color_png(
    bbox: list[float],
    date: str,
    width: int = 512,
    height: int = 512,
    creds: tuple[str, str] | None = None,
) -> bytes:
    """Request a B04/B03/B02 true-color PNG for a scene via the Process API."""
    token = get_access_token(creds)
    key = _key(*_get_credentials(creds))

    body = {
        "input": {
            "bounds": {"bbox": bbox},
            "data": [
                {
                    "type": COLLECTION,
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z",
                        }
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": TRUE_COLOR_EVALSCRIPT,
    }

    try:
        response = httpx.post(
            PROCESS_URL,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=60.0,
        )
    except httpx.HTTPError as exc:
        raise CDSEUpstreamError(f"CDSE process request failed: {exc}") from exc
    _record(key, response, "process")
    _raise_for(response, "process request")

    return response.content
