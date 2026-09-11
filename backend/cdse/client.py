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

import os
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


# Small in-process token cache — one token shared across requests until it
# is close to expiry, refreshed on demand rather than on a schedule.
_token_cache: dict[str, float | str] = {"access_token": "", "expires_at": 0.0}


def _get_credentials() -> tuple[str, str]:
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise CDSEAuthError(
            "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not set in the environment"
        )
    return client_id, client_secret


def get_access_token() -> str:
    """Return a cached CDSE access token, refreshing it if it's expired or missing."""
    now = time.time()
    if _token_cache["access_token"] and now < float(_token_cache["expires_at"]):
        return str(_token_cache["access_token"])

    client_id, client_secret = _get_credentials()

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
    _token_cache["access_token"] = token
    _token_cache["expires_at"] = now + max(expires_in - 60, 30)

    return token


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
) -> dict:
    """Query the Sentinel Hub Catalog (STAC) for Sentinel-2 L2A scenes near (lat, lon)."""
    token = get_access_token()
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
        response.raise_for_status()
    except httpx.HTTPError as exc:
        detail = exc.response.text if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        raise CDSEUpstreamError(f"CDSE catalog search failed: {detail}") from exc

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
) -> bytes:
    """Request a B04/B03/B02 true-color PNG for a scene via the Process API."""
    token = get_access_token()

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
        response.raise_for_status()
    except httpx.HTTPError as exc:
        detail = exc.response.text if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        raise CDSEUpstreamError(f"CDSE process request failed: {detail}") from exc

    return response.content
