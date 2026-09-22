"""Semantic-prior phase 2.3, Step 3 -- Google Open Buildings 2.5D Temporal
(height + presence) for all 25 sign-flip-accepted Sentinel-2 benchmark
tiles, reprojected onto each tile's own 10m UTM grid.

Extends scripts/prepare_method4_openbuildings_data.py's fetch logic (same
public, unauthenticated GCS bucket, same bands: 1=fractional_count,
2=building_height, 3=building_presence, nodata=-99.0) from its original 8
urban tiles to all 25 accepted tiles -- reuses fetch_open_buildings()
unchanged, does not reimplement it. Coverage will be near-zero outside
urban/built-up areas (agricultural/coastal/hilly tiles); that's expected,
not a bug -- Open Buildings only has real signal where there are buildings.

Output: data/sentinel2_benchmark/semantic/<tile>/ob_height.tif (raw AGL
        height, NaN where no OB data), ob_presence.tif (NaN where no OB
        tile covers the location at all -- distinct from "covered, presence
        near 0", i.e. real building absence)
        data/sentinel2_benchmark/semantic/openbuildings_coverage_report.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_method4_openbuildings_data as _ob_mod  # noqa: E402
from prepare_method4_openbuildings_data import fetch_open_buildings, OB_YEAR, CONFIDENT_THRESHOLD  # noqa: E402

# Memoize _tiles_for(epsg, year) across tiles: the original script called it fresh per
# tile (fine for its original 8-tile, single-EPSG-pair run), but this benchmark's 25
# tiles span 5 EPSGs -- unmemoized, every tile re-lists + re-parses the full GCS
# manifest set for its EPSG, which dominated wall time (~11min/tile) on a first full run.
from functools import lru_cache  # noqa: E402
_ob_mod._tiles_for = lru_cache(maxsize=None)(_ob_mod._tiles_for)

MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
VERDICTS = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
OUT_DIR = ROOT / "data/sentinel2_benchmark/semantic"


def process_tile(tile_id: str, rgb_path: Path, epsg: int) -> dict:
    with rasterio.open(rgb_path) as src:
        dst_crs, dst_transform, dst_shape = src.crs, src.transform, (src.height, src.width)
        bounds = tuple(src.bounds)
        profile = src.profile.copy()

    try:
        height, presence = fetch_open_buildings(epsg, bounds, dst_crs, dst_transform, dst_shape)
    except SystemExit as exc:
        return {"tile_id": tile_id, "status": f"NO_OB_TILE: {exc}"}

    out_dir = OUT_DIR / tile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan, tiled=False)
    with rasterio.open(out_dir / "ob_height.tif", "w", **out_profile) as dst:
        dst.write(height, 1)
    with rasterio.open(out_dir / "ob_presence.tif", "w", **out_profile) as dst:
        dst.write(presence, 1)

    covered = np.isfinite(presence)
    confident = covered & (presence > CONFIDENT_THRESHOLD) & np.isfinite(height)
    total_px = presence.size
    return {
        "tile_id": tile_id,
        "status": "OK",
        "ob_year": OB_YEAR,
        "covered_pct": 100.0 * covered.sum() / total_px,
        "confident_building_pct": 100.0 * confident.sum() / total_px,
        "confident_pixels": int(confident.sum()),
        "mean_height_confident_m": float(np.nanmean(height[confident])) if confident.any() else float("nan"),
    }


URBAN_ALREADY_FETCHED = {"bengaluru", "chennai", "delhi", "hyderabad", "jaipur", "kochi_city", "mumbai", "pune"}
# These 8 were already fetched by prepare_method4_openbuildings_data.py's identical
# fetch_open_buildings call onto the same per-tile grid; reused via
# scripts/_reuse_urban_openbuildings.py instead of re-fetching (dense urban tiles
# took ~10min each to fetch fresh -- see run history/commit message).


def main() -> int:
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS)
    accepted = sorted(set(verdicts.loc[~verdicts["flagged"], "tile_id"].tolist()) - URBAN_ALREADY_FETCHED)
    print(f"{len(accepted)} accepted tiles (excluding {len(URBAN_ALREADY_FETCHED)} urban tiles reused from method4_port_openbuildings)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, tile_id in enumerate(accepted, 1):
        m = manifest.loc[tile_id]
        rgb_path = ROOT / m["rgb_path"]
        epsg = int(m["bbox_utm_epsg"])
        print(f"[{i}/{len(accepted)}] {tile_id}")
        try:
            r = process_tile(tile_id, rgb_path, epsg)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
            r = {"tile_id": tile_id, "status": f"ERROR: {exc}"}
        r["category"] = m["category"]
        rows.append(r)
        print(f"    -> {r.get('status')} confident_building_pct={r.get('confident_building_pct', float('nan')):.3f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "openbuildings_coverage_report.csv", index=False)
    ok = (df["status"] == "OK").sum()
    print(f"\n{ok}/{len(df)} tiles OK.")
    print(df[["tile_id", "category", "status", "confident_building_pct"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
