"""Location facts + hazard history for the expanded 3D view's Facts box.

Every section is a live query to a named, free source, returned with that
source's name, URL and the exact query parameters; nothing is filled in when
a source fails or does not exist. Sections carry a status:

  "ok"          — real data from the named source
  "unavailable" — no free, global, queryable source was found (reason given)
  "error"       — the source exists but could not be reached just now

Sources (checked live 2026-09-25):
  place       Nominatim reverse geocoding (OpenStreetMap). Fair-use policy:
              <= 1 request/s, identifying User-Agent, results cached.
  elevation   Open-Meteo Elevation API (Copernicus GLO-90 DEM), point value.
  seismic     USGS FDSN event web service (ANSS ComCat), M>=4.5 within 250 km.
  floods      GDACS event list (JRC/UN), flood alerts whose reference point
              lies within 250 km. GDACS lists alert-level events only, and a
              flood's point is a representative location, not its extent.
  landslides  unavailable: NASA's Global Landslide Catalog / COOLR was the
              candidate; its public Socrata API (data.nasa.gov dd9e-wu2v) now
              returns 404, and no queryable service URL was found.
  volcanoes   unavailable: the Smithsonian GVP WFS (webservices.volcano.si.edu)
              did not respond in repeated tries.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import date

import httpx

USER_AGENT = "DepthWizard2/0.1 (SIH26175 hackathon prototype; https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026)"
RADIUS_KM = 250
MIN_MAG = 4.5
TIMEOUT = httpx.Timeout(25.0)

NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
OPEN_METEO_ELEV = "https://api.open-meteo.com/v1/elevation"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1"
GDACS = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"

_nominatim_lock = threading.Lock()
_nominatim_last = 0.0
_cache: dict[tuple, tuple[float, dict]] = {}
_CACHE_S = 24 * 3600
_gdacs_floods: tuple[float, list] | None = None


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _get(client: httpx.Client, url: str, params: dict) -> httpx.Response:
    r = client.get(url, params=params)
    r.raise_for_status()
    return r


def _error(source: str, url: str, exc: Exception) -> dict:
    return {"status": "error", "source": source, "url": url, "message": f"{type(exc).__name__}: {exc}"[:240]}


# ------------------------------------------------------------------ sections
def place(client, lat, lon) -> dict:
    global _nominatim_last
    params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 10, "addressdetails": 1}
    try:
        with _nominatim_lock:  # fair use: at most one request per second
            wait = 1.1 - (time.monotonic() - _nominatim_last)
            if wait > 0:
                time.sleep(wait)
            _nominatim_last = time.monotonic()
            data = _get(client, NOMINATIM, params).json()
    except Exception as exc:  # noqa: BLE001
        return _error("Nominatim (OpenStreetMap)", NOMINATIM, exc)
    if "error" in data:
        return {"status": "ok", "source": "Nominatim (OpenStreetMap)", "url": NOMINATIM, "query": params,
                "name": None, "note": "No named place at these coordinates (e.g. open sea).",
                "licence": data.get("licence")}
    addr = data.get("address", {})
    return {
        "status": "ok",
        "source": "Nominatim (OpenStreetMap)",
        "url": NOMINATIM,
        "query": params,
        "name": data.get("display_name"),
        "locality": addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county"),
        "district": addr.get("state_district") or addr.get("county"),
        "state": addr.get("state"),
        "country": addr.get("country"),
        "licence": data.get("licence"),
    }


def elevation(client, lat, lon) -> dict:
    params = {"latitude": lat, "longitude": lon}
    try:
        value = _get(client, OPEN_METEO_ELEV, params).json()["elevation"][0]
    except Exception as exc:  # noqa: BLE001
        return _error("Open-Meteo Elevation API", OPEN_METEO_ELEV, exc)
    return {"status": "ok", "source": "Open-Meteo Elevation API (Copernicus GLO-90 DEM)", "url": OPEN_METEO_ELEV,
            "query": params, "elevation_m": value}


def seismic(client, lat, lon) -> dict:
    base = {"format": "geojson", "latitude": lat, "longitude": lon, "maxradiuskm": RADIUS_KM,
            "minmagnitude": MIN_MAG, "starttime": "1900-01-01", "eventtype": "earthquake"}
    try:
        count = _get(client, f"{USGS}/count", base).json()["count"]
        largest = _get(client, f"{USGS}/query", {**base, "orderby": "magnitude", "limit": 3}).json()["features"]
        latest = _get(client, f"{USGS}/query", {**base, "orderby": "time", "limit": 1}).json()["features"]
    except Exception as exc:  # noqa: BLE001
        return _error("USGS earthquake catalog (ComCat)", f"{USGS}/query", exc)

    def event(f):
        lon_e, lat_e = f["geometry"]["coordinates"][:2]
        p = f["properties"]
        return {"mag": p["mag"], "place": p["place"], "date": time.strftime("%Y-%m-%d", time.gmtime(p["time"] / 1000)),
                "distance_km": round(_haversine_km(lat, lon, lat_e, lon_e)), "url": p["url"]}

    return {
        "status": "ok",
        "source": "USGS earthquake catalog (ANSS ComCat, FDSN event service)",
        "url": f"{USGS}/query",
        "query": base,
        "radius_km": RADIUS_KM,
        "min_magnitude": MIN_MAG,
        "count": count,
        "largest": [event(f) for f in largest],
        "latest": event(latest[0]) if latest else None,
        "caveat": "Catalogue completeness for M4.5+ is good globally from about 1973; earlier records are sparse.",
    }


def _all_gdacs_floods(client) -> list:
    global _gdacs_floods
    if _gdacs_floods and time.time() - _gdacs_floods[0] < _CACHE_S:
        return _gdacs_floods[1]
    events = []
    for page in range(1, 60):
        params = {"eventlist": "FL", "fromdate": "2000-01-01", "todate": date.today().isoformat(),
                  "pagesize": 100, "pagenumber": page}
        feats = _get(client, GDACS, params).json().get("features", [])
        events += feats
        if len(feats) < 100:
            break
    _gdacs_floods = (time.time(), events)
    return events


def floods(client, lat, lon) -> dict:
    try:
        events = _all_gdacs_floods(client)
    except Exception as exc:  # noqa: BLE001
        return _error("GDACS", GDACS, exc)
    near = []
    for f in events:
        lon_e, lat_e = f["geometry"]["coordinates"][:2]
        d = _haversine_km(lat, lon, lat_e, lon_e)
        if d <= RADIUS_KM:
            p = f["properties"]
            near.append({"name": p.get("name"), "country": p.get("country"), "from": (p.get("fromdate") or "")[:10],
                         "to": (p.get("todate") or "")[:10], "alert": (p.get("alertlevel") or "unknown").title(),
                         "distance_km": round(d), "url": (p.get("url") or {}).get("report")})
    near.sort(key=lambda e: e["from"], reverse=True)
    alerts: dict[str, int] = {}
    for e in near:
        alerts[e["alert"]] = alerts.get(e["alert"], 0) + 1
    return {
        "status": "ok",
        "source": "GDACS (Global Disaster Alert and Coordination System, EC JRC / UN OCHA)",
        "url": GDACS,
        "query": {"eventlist": "FL", "fromdate": "2000-01-01", "radius_km": RADIUS_KM},
        "radius_km": RADIUS_KM,
        "count": len(near),
        "by_alert": alerts,
        "recent": near[:3],
        "caveat": "GDACS lists alert-level flood events only, each at one representative point, not its flooded "
                  "extent. This is a history of alerted events nearby, not a flood-susceptibility rating.",
    }


LANDSLIDES = {
    "status": "unavailable",
    "reason": "No free, global, queryable landslide source was reachable. NASA's Global Landslide Catalog (COOLR) "
              "was the candidate, but its public API (data.nasa.gov, dataset dd9e-wu2v) now returns 404 and no "
              "service endpoint was found.",
}
VOLCANOES = {
    "status": "unavailable",
    "reason": "The Smithsonian Global Volcanism Program web service (webservices.volcano.si.edu) did not respond, "
              "so no volcano data is shown.",
}


def facts(lat: float, lon: float) -> dict:
    key = (round(lat, 3), round(lon, 3))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _CACHE_S:
        return hit[1]
    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                      follow_redirects=True) as client:
        out = {
            "lat": lat,
            "lon": lon,
            "queried_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "place": place(client, lat, lon),
            "elevation": elevation(client, lat, lon),
            "seismic": seismic(client, lat, lon),
            "floods": floods(client, lat, lon),
            "landslides": LANDSLIDES,
            "volcanoes": VOLCANOES,
        }
    # don't cache a response with a transient failure in it
    if not any(isinstance(v, dict) and v.get("status") == "error" for v in out.values()):
        _cache[key] = (time.time(), out)
    return out
