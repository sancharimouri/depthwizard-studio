"""Bundled ThinkHazard! district levels by coordinates (built by scripts/build_thinkhazard_grid.py).

A 0.025° global grid maps each cell to the GAUL 2015 admin-2 division covering the cell centre; a table gives that
division's ThinkHazard levels. Near a district boundary the cell (~2.8 km) may name the neighbour: these are
district-level ratings, labelled as such. No network. If the bundle is missing (e.g. an older desktop sidecar),
lookup() returns None and the panels simply omit the district lines.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
_lock = threading.Lock()
_state: dict = {}


def _load():
    with _lock:
        if "rows" not in _state:
            try:
                import rasterio

                meta = json.loads((DATA / "thinkhazard_levels.json").read_text())
                _state["rows"] = meta["rows"]
                _state["hazards"] = meta["hazards"].split()
                _state["ds"] = rasterio.open(DATA / meta["grid"])
            except Exception:  # noqa: BLE001 — a missing/corrupt bundle means "no district lines", never an error
                _state["rows"] = None
    return _state


def lookup(lat: float, lon: float, bbox: list[float] | None = None) -> dict | None:
    """{"code", "name", "admin1", "admin0", "levels": {"WF": "H", ...}} for the division at (lat, lon), or None.
    With a bbox, a centre that falls in water (no division) takes the division covering most of the bbox."""
    st = _load()
    if not st["rows"]:
        return None
    ds = st["ds"]
    try:
        import numpy as np
        from rasterio.windows import Window, from_bounds

        r, c = ds.index(lon, lat)
        if not (0 <= r < ds.height and 0 <= c < ds.width):
            return None
        with _lock:  # one shared dataset handle: GDAL handles are not thread-safe
            idx = int(ds.read(1, window=Window(c, r, 1, 1))[0, 0])
            if idx == 0 and bbox:
                a = ds.read(1, window=from_bounds(*bbox, ds.transform), boundless=True, fill_value=0)
                vals, counts = np.unique(a[a > 0], return_counts=True)
                idx = int(vals[np.argmax(counts)]) if vals.size else 0
    except Exception:  # noqa: BLE001
        return None
    if idx <= 0 or idx >= len(st["rows"]) or not st["rows"][idx]:
        return None
    code, name, a1, a0, lv = st["rows"][idx]
    levels = {hz: ch for hz, ch in zip(st["hazards"], lv or "") if ch != "-"}
    return {"code": code, "name": name, "admin1": a1, "admin0": a0, "levels": levels}
