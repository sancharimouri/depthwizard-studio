"""One-off: reuse the 8 urban tiles' Open Buildings height/presence rasters
already fetched by prepare_method4_openbuildings_data.py (same fetch_open_buildings
call, same per-tile UTM grid -- verified bit-identical transform/shape/crs against
each tile's own RGB) instead of re-fetching them, which was taking ~10min/tile for
dense urban tiles and would have made the full 25-tile Step 3 batch impractically
slow. Only the 17 non-urban (+ already-independently-fetched) tiles still need a
real fetch.

That script wrote NaN as -1.0 (nan_to_num) inside float32 rasters; this script
converts back to true NaN on copy so downstream code (run_frequency_fusion_semantic.py)
can treat all semantic/ rasters uniformly regardless of which path produced them.
"""
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data/sentinel2_benchmark/method4_port_openbuildings"
OUT_DIR = ROOT / "data/sentinel2_benchmark/semantic"
URBAN_TILES = ["bengaluru", "chennai", "delhi", "hyderabad", "jaipur", "kochi_city", "mumbai", "pune"]

for tile_id in URBAN_TILES:
    out_dir = OUT_DIR / tile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for band in ["height", "presence"]:
        src_path = SRC_DIR / f"{tile_id}_ob_{band}.tif"
        if not src_path.exists():
            print(f"MISSING {src_path}")
            continue
        with rasterio.open(src_path) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile.copy()
        arr = np.where(arr == -1.0, np.nan, arr)
        profile.update(nodata=np.nan, tiled=False)
        out_path = out_dir / f"ob_{band}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)
        print(f"{tile_id} ob_{band}: wrote {out_path}, valid_pct={100*np.isfinite(arr).mean():.2f}%")

print("done")
