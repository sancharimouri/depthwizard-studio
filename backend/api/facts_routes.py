"""Facts box: location facts + hazard history for a job's coordinates.

Coordinates come from the job's own geotransform, or are typed in by the
user; this route never infers a location. See backend/facts/sources.py.
"""

from fastapi import APIRouter, Query

from backend.facts.sources import facts

router = APIRouter(prefix="/api/facts")


@router.get("")
def get_facts(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)) -> dict:
    return facts(lat, lon)
