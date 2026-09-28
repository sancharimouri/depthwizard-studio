# DEM source/cache/provenance tests.
import numpy as np


def _synthetic_tile(lat0: int, lon0: int, n: int = 120):
    """A 1°x1° EPSG:4326 tile (n x n px) whose value is a smooth function of lat/lon,
    continuous across tile edges, like GLO-30 terrain."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    res = 1.0 / n
    lats = lat0 + 1 - (np.arange(n) + 0.5) * res
    lons = lon0 + (np.arange(n) + 0.5) * res
    data = (1000 + 800 * (lats[:, None] - 26.5) + 300 * np.sin(lons[None, :] * 7)).astype("float32")
    mem = MemoryFile()
    with mem.open(driver="GTiff", width=n, height=n, count=1, dtype="float32", crs="EPSG:4326",
                  transform=from_bounds(lon0, lat0, lon0 + 1, lat0 + 1, n, n), nodata=float("nan")) as dst:
        dst.write(data, 1)
    return mem.open()


def test_glo30_mosaic_has_no_seam_across_a_tile_edge():
    # Darjeeling's footprint: its south edge lies on 27°N, the N26/N27 tile seam (2026-09-28 spikes)
    from pyproj import Transformer
    from backend.dem import glo30
    from backend.generation.pipeline import _grid_from_bounds
    crs = "EPSG:32645"
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(88.26003, 27.04501)
    bounds = (x - 5010, y - 5035, x + 5010, y + 5035)
    shape, tf = _grid_from_bounds(crs, bounds, 30.0)
    from rasterio.warp import transform_bounds
    ll = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    assert ll[1] < 27.0 < ll[3]
    tiles = [_synthetic_tile(26, 88), _synthetic_tile(27, 88)]
    out = glo30.mosaic_to_grid(tiles, crs, ll, shape, tf)
    assert np.isfinite(out).all(), f"{int((~np.isfinite(out)).sum())} NaN pixels at the seam"


def test_terrain_json_fills_holes_from_neighbours_not_the_minimum(tmp_path):
    import json
    from backend.terrain.mesh_export import write_terrain_json
    surf = np.full((40, 40), 2000.0, np.float32)
    surf[0, :] = 500.0           # the tile's real low point
    surf[30:32, 20:22] = np.nan  # a hole in high ground
    meta = write_terrain_json(tmp_path / "t.json", surf, (0, 0, 1, 1), (40, 40))
    d = json.loads((tmp_path / "t.json").read_text())
    h = np.array(d["heights"]).reshape(40, 40) * (meta["elevationMax"] - meta["elevationMin"]) + meta["elevationMin"]
    assert h[30:32, 20:22].min() > 1900, "the hole must take its neighbours' height, not the tile minimum"
