"""backend/generation/pipeline._grid_from_bounds: cell size in metres, whatever the CRS units (2026-09-30 fix).
A Search Online scene is EPSG:4326; its 30 m grid used to be applied as 30 degrees (and its DEM's 1 m floor as
1 degree), so every scene got the 8 x 8 minimum mesh."""
from backend.generation.pipeline import M_PER_DEG, _grid_from_bounds


def test_geographic_bounds_get_metre_cells():
    # ~10 km x 10 km around Bathinda, in degrees
    (h, w), tf = _grid_from_bounds("EPSG:4326", (74.90, 30.16, 75.00, 30.25), 30.0)
    assert (h, w) == (round(0.09 * M_PER_DEG / 30), round(0.10 * M_PER_DEG / 30))
    assert abs(tf.a * M_PER_DEG - 30.0) < 1.0


def test_projected_bounds_unchanged():
    (h, w), tf = _grid_from_bounds("EPSG:32643", (500000, 3300000, 510000, 3310000), 30.0)
    assert (h, w) == (333, 333)
