"""Semantic-prior phase 2.3, Step 2 -- Microsoft GlobalMLBuildingFootprints
building_fraction rasters for the 25 sign-flip-accepted Sentinel-2 benchmark
tiles (data/sentinel2_benchmark/manifest.csv), NOT the old 4/10-city Method3
demo-track tiles under data/sentinel2/. Those are a different AOI set even
where a city name coincides (e.g. the old data/sentinel2/mumbai/ Method3 tile
is not the same bbox/date as this benchmark's mumbai row).

Reuses existing project machinery rather than reimplementing it:
  - quadkey/AOI partition selection, India index loading, download:
    scripts/repair_method3_coverage.py (sentinel_bounds_wgs84,
    load_india_index, select_partitions_for_aoi, download_file)
  - GeoJSONL parsing + supersampled area-fraction rasterization onto the
    exact Sentinel/UTM pixel grid: scripts/prepare_method3_semantic_inputs.py
    (read_geojsonl_gz, ensure_valid_geometry, rasterize_fraction)

Coverage-gap QC (new here): docs/method-audit/03-semantic-prior/
gaps-and-fixes.md documented, but never implemented, "a simple
contiguous-blank-region detector, not just mean/p95" after mean/p95 missed a
real Kolkata download gap. Implemented here (see detect_contiguous_gap) as a
nonzero-pixel spatial-extent test: real content confined to less than 60% of
the tile's width or height is flagged as a likely partition/download cutoff,
distinct from organically thin-but-tile-wide building coverage (a plain
"largest connected zero-blob" test cannot make this distinction -- almost
every low-density rural tile is one big connected zero blob regardless of
whether the data is real). Flagged tiles are excluded from this step's
signal and documented, not filled with a guessed fallback.

Output: data/sentinel2_benchmark/semantic/<tile>/building_fraction.tif
        data/sentinel2_benchmark/semantic/coverage_report.csv
Raw partitions cached at:
  data/sentinel2_benchmark/semantic_sources/globalml_building_footprints/raw/<tile>/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from repair_method3_coverage import (  # noqa: E402
    sentinel_bounds_wgs84,
    load_india_index,
    select_partitions_for_aoi,
    download_file,
)
from prepare_method3_semantic_inputs import (  # noqa: E402
    read_geojsonl_gz,
    ensure_valid_geometry,
)
import geopandas as gpd
from affine import Affine
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import box

MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
VERDICTS = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
INDEX_PATH = ROOT / "data/sentinel2/semantic_sources/globalml_building_footprints/dataset-links.csv"
RAW_DIR = ROOT / "data/sentinel2_benchmark/semantic_sources/globalml_building_footprints/raw"
OUT_DIR = ROOT / "data/sentinel2_benchmark/semantic"
SUPERSAMPLE = 5



def rasterize_fraction_safe(gdf: gpd.GeoDataFrame, rgb_path: Path, output_path: Path,
                             supersample: int = SUPERSAMPLE) -> dict:
    """Same supersampled area-fraction rasterization as
    prepare_method3_semantic_inputs.rasterize_fraction, reimplemented here
    because that function's output profile inherits the RGB source's
    blockxsize/blockysize (e.g. 1000, not a multiple of 16) while forcing
    tiled=True -- fine for the old Method3 tile dimensions, but rasterio
    rejects it on this benchmark's 1000x1000 grids ("TIFF dataset blocks
    must be multiples of 16"). Untiled GTiff output sidesteps this; nothing
    else about the method changes."""
    with rasterio.open(rgb_path) as src:
        width, height, transform, crs = src.width, src.height, src.transform, src.crs

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)

    bounds_geom = box(*rasterio.transform.array_bounds(height, width, transform))
    aoi = gpd.GeoDataFrame({"geometry": [bounds_geom]}, geometry="geometry", crs=crs)
    try:
        gdf = gpd.clip(gdf, aoi)
    except Exception:
        gdf = gdf.loc[gdf.geometry.intersects(bounds_geom)].copy()
    gdf = ensure_valid_geometry(gdf)

    fine_h, fine_w = height * supersample, width * supersample
    fine_transform = transform * Affine.scale(1.0 / supersample, 1.0 / supersample)

    if gdf.empty:
        fraction = np.zeros((height, width), dtype=np.float32)
        n_after_clip = 0
    else:
        fine_mask = np.zeros((fine_h, fine_w), dtype=np.uint8)
        rio_rasterize(shapes=((geom, 1) for geom in gdf.geometry), out=fine_mask,
                      transform=fine_transform, fill=0, all_touched=False, dtype="uint8")
        fraction = fine_mask.reshape(height, supersample, width, supersample).mean(axis=(1, 3)).astype(np.float32)
        n_after_clip = int(len(gdf))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", dtype="float32", count=1, nodata=-9999.0,
                   width=width, height=height, transform=transform, crs=crs,
                   compress="deflate", predictor=2, tiled=False)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(fraction, 1)

    return {
        "footprints_after_clip": n_after_clip,
        "building_fraction_mean": float(fraction.mean()),
        "building_fraction_max": float(fraction.max()),
        "building_pixels_gt_0": int(np.count_nonzero(fraction > 0)),
    }


MIN_FOOTPRINTS_FOR_GAP_TEST = 5
NZ_SPAN_GAP_THRESHOLD = 0.6


def detect_contiguous_gap(fraction: np.ndarray, n_footprints: int) -> dict:
    """Distinguishes a real download/partition gap (gaps-and-fixes.md's
    documented Kolkata/Delhi/Mumbai failure: a large, hard-edged, contiguous
    empty rectangle or band, invisible in mean/p95 alone) from a tile that is
    honestly near-zero-building (e.g. a coastal salt marsh or dense forest,
    where buildings really are scattered thin across the WHOLE tile, not
    confined to one edge).

    A pure "largest connected zero-component" test conflates the two: for
    any genuinely low-density tile almost the whole raster is one connected
    zero blob regardless of whether the data is real. The actual
    discriminator gaps-and-fixes.md's own description implies is *where*
    the real (nonzero) content sits: a download/partition gap leaves real
    content confined to a sub-rectangle not spanning the tile's full width
    or height, whereas organically sparse data is thin but present across
    the tile's full spatial extent.

    Tiles with too few footprints total to say anything about spatial
    pattern (< MIN_FOOTPRINTS_FOR_GAP_TEST) are never flagged -- a
    single-digit footprint count in a 10km AOI is itself the finding (real
    sparse landscape), not evidence of a truncated fetch, and every
    partition intersecting the AOI was requested (see select_partitions_for_aoi),
    so there is no half-downloaded state this pipeline can silently produce.
    """
    nz = fraction > 0.0
    if n_footprints < MIN_FOOTPRINTS_FOR_GAP_TEST or not nz.any():
        return {"gap_flag": False, "nz_width_span": float(nz.any()), "nz_height_span": float(nz.any()),
                "gap_reason": "too_few_footprints_to_test"}

    ys, xs = np.where(nz)
    height, width = fraction.shape
    nz_width_span = (xs.max() - xs.min() + 1) / width
    nz_height_span = (ys.max() - ys.min() + 1) / height

    gap_flag = nz_width_span < NZ_SPAN_GAP_THRESHOLD or nz_height_span < NZ_SPAN_GAP_THRESHOLD
    reason = ""
    if gap_flag:
        reason = (f"real content ({n_footprints} footprints) confined to "
                  f"{nz_width_span:.0%} of width / {nz_height_span:.0%} of height -- "
                  f"consistent with a partition/download cutoff, not natural sparsity")
    return {
        "gap_flag": bool(gap_flag),
        "nz_width_span": float(nz_width_span),
        "nz_height_span": float(nz_height_span),
        "gap_reason": reason,
    }


def process_tile(tile_id: str, rgb_path: Path, india_index: pd.DataFrame) -> dict:
    aoi_bounds = sentinel_bounds_wgs84(rgb_path)
    parts = select_partitions_for_aoi(india_index, aoi_bounds)
    if parts.empty:
        return {"tile_id": tile_id, "status": "NO_PARTITIONS", "n_partitions": 0}

    raw_dir = RAW_DIR / tile_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for _, row in parts.iterrows():
        qk = row["QuadKey"]
        url = row["Url"]
        dest = raw_dir / f"{qk}_{Path(url).name}"
        try:
            download_file(url, dest)
            local_paths.append(dest)
        except Exception as exc:  # noqa: BLE001
            print(f"    [{tile_id}] download failed for quadkey={qk}: {exc}")

    if not local_paths:
        return {"tile_id": tile_id, "status": "DOWNLOAD_FAILED", "n_partitions": len(parts)}

    gdf = read_geojsonl_gz(local_paths)

    out_dir = OUT_DIR / tile_id
    out_path = out_dir / "building_fraction.tif"
    stats = rasterize_fraction_safe(gdf, rgb_path, out_path, supersample=SUPERSAMPLE)

    with rasterio.open(out_path) as src:
        fraction = src.read(1)

    gap = detect_contiguous_gap(fraction, stats["footprints_after_clip"])

    return {
        "tile_id": tile_id,
        "status": "GAP_EXCLUDED" if gap["gap_flag"] else "OK",
        "n_partitions": len(parts),
        "n_footprints_after_clip": stats["footprints_after_clip"],
        "building_fraction_mean": stats["building_fraction_mean"],
        "building_fraction_max": stats["building_fraction_max"],
        "building_pixels_gt_0_pct": 100.0 * stats["building_pixels_gt_0"] / fraction.size,
        **gap,
    }


def main() -> int:
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS)
    accepted = verdicts.loc[~verdicts["flagged"], "tile_id"].tolist()
    print(f"{len(accepted)} accepted tiles")

    india_index = load_india_index(INDEX_PATH)
    print(f"Loaded India partition index: {len(india_index)} quadkey rows")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, tile_id in enumerate(sorted(accepted), 1):
        rgb_path = ROOT / manifest.loc[tile_id, "rgb_path"]
        print(f"[{i}/{len(accepted)}] {tile_id}")
        try:
            r = process_tile(tile_id, rgb_path, india_index)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
            r = {"tile_id": tile_id, "status": f"ERROR: {exc}", "n_partitions": 0}
        r["category"] = manifest.loc[tile_id, "category"]
        rows.append(r)
        print(f"    -> {r.get('status')}  mean={r.get('building_fraction_mean', float('nan')):.4f}  "
              f"nz_width_span={r.get('nz_width_span', float('nan')):.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "coverage_report.csv", index=False)
    ok = (df["status"] == "OK").sum()
    print(f"\n{ok}/{len(df)} tiles OK. Excluded (gap/failed): "
          f"{df.loc[df['status'] != 'OK', 'tile_id'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
