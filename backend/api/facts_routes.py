"""Facts box + Scenario Analysis cards for a job's location (backend/facts/sources.py).

Georeferenced input sends its footprint as bbox=w,s,e,n (WGS84); non-georeferenced input sends the user-entered
lat/lon (point lookups only). This route never infers a location.
"""

from fastapi import APIRouter, HTTPException, Query

from backend.facts.sources import info

router = APIRouter(prefix="/api/facts")


@router.get("")
def get_facts(bbox: str | None = Query(None, description="w,s,e,n in degrees"),
              lat: float | None = Query(None, ge=-90, le=90), lon: float | None = Query(None, ge=-180, le=180)) -> dict:
    if bbox:
        try:
            w, s, e, n = (float(v) for v in bbox.split(","))
        except ValueError:
            raise HTTPException(422, "bbox must be w,s,e,n") from None
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90) or (e - w) > 2 or (n - s) > 2:
            raise HTTPException(422, "bbox must be a valid WGS84 box of at most 2 degrees")
        return info(bbox=[w, s, e, n])
    if lat is None or lon is None:
        raise HTTPException(422, "give bbox=w,s,e,n or lat and lon")
    return info(lat=lat, lon=lon)
