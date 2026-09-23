#!/usr/bin/env python3
"""
DepthWizard2 — Method 3 GlobalMLBuildingFootprints Coverage Repair

Purpose
-------
The first Method-3 rasterization used a hand-selected subset of Microsoft
GlobalMLBuildingFootprints partitions. The QC PDF revealed hard source-coverage
gaps in several Sentinel AOIs (especially Kolkata, Delhi, Mumbai, Kochi).

This script fixes that systematically.

It does NOT guess which CSV belongs to which city.

Instead it:

1. Reads the CURRENT Microsoft India dataset-links.csv.
2. Reads each local Sentinel RGB GeoTIFF to get its exact AOI bounds.
3. Converts the AOI bounds to WGS84.
4. Decodes every India L9 QuadKey in dataset-links.csv to its geographic tile
   bounds.
5. Selects every partition tile whose footprint intersects the Sentinel AOI.
6. Downloads all required .csv.gz partitions that are not already present.
7. Reports exactly which partitions were selected.
8. Rebuilds building_fraction.tif using ALL selected partitions.
9. Derives building_density_50m.tif, building_edge.tif and
   distance_to_building_m.tif.
10. Writes a machine-readable coverage audit.

The .csv.gz files are line-delimited GeoJSON, despite the extension. This is
the format documented by Microsoft.

IMPORTANT
---------
- This uses the 2026-08-13 GlobalMLBuildingFootprints release.
- It does NOT use Microsoft's building-height attribute.
- It keeps the primary Method-3 feature as building_fraction on the exact
  Sentinel grid.
- Supersampling defaults to 5x. Do not move to 10x until the coverage audit
  and regenerated QC PDF look correct.
- Existing raw parts are reused; only missing required parts are downloaded.
- Existing semantic rasters are overwritten only when --overwrite-raster is used.

Expected output
---------------
data/sentinel2/semantic_sources/globalml_building_footprints/
    dataset-links.csv
    coverage_audit/
        coverage_audit.csv
        coverage_audit.json
        selected_partitions.csv
    raw/<city>/
        <all required .csv.gz parts>

data/sentinel2/<city>/semantic/
    building_fraction.tif
    building_density_50m.tif
    building_edge.tif
    distance_to_building_m.tif
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import box, shape


# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(
    "/Users/anweshasaha/projects/DepthWizard2"
)

DOWNLOADS_DIR = Path(
    "/Users/anweshasaha/Downloads"
)

CURRENT_RELEASE = "2026-08-13"

CURRENT_INDEX_URL = (
    "https://bfppub.blob.core.windows.net/$web/"
    f"{CURRENT_RELEASE}/dataset-links.csv"
)

DEFAULT_SENTINEL_ROOT = (
    PROJECT_ROOT / "data" / "sentinel2"
)

DEFAULT_SOURCE_ROOT = (
    DEFAULT_SENTINEL_ROOT
    / "semantic_sources"
    / "globalml_building_footprints"
)

DEFAULT_INDEX = (
    DEFAULT_SOURCE_ROOT / "dataset-links.csv"
)

DEFAULT_RAW_ROOT = (
    DEFAULT_SOURCE_ROOT / "raw"
)

DEFAULT_AUDIT_ROOT = (
    DEFAULT_SOURCE_ROOT / "coverage_audit"
)

INDIA_LOCATION = "India"

CITY_ORDER = [
    "kolkata",
    "bardhaman",
    "delhi",
    "mumbai",
    "bengaluru",
    "hyderabad",
    "jaipur",
    "kochi",
    "darjeeling",
    "sundarbans",
]

CITY_LABELS = {
    "kolkata": "Kolkata",
    "bardhaman": "Bardhaman",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
    "bengaluru": "Bengaluru",
    "hyderabad": "Hyderabad",
    "jaipur": "Jaipur",
    "kochi": "Kochi",
    "darjeeling": "Darjeeling",
    "sundarbans": "Sundarbans",
}


# ============================================================================
# BASIC HELPERS
# ============================================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):
        print(
            f"    exists: {destination.name}"
        )
        return

    tmp = destination.with_suffix(
        destination.suffix + ".part"
    )

    if tmp.exists():
        tmp.unlink()

    print(
        f"    download: {destination.name}"
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DepthWizard2/Method3CoverageRepair"
        },
    )

    try:
        with urllib.request.urlopen(
            req,
            timeout=180,
        ) as response, tmp.open("wb") as out:
            shutil.copyfileobj(
                response,
                out,
                length=1024 * 1024,
            )
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    tmp.replace(destination)


# ============================================================================
# BING QUADKEY DECODING
# ============================================================================

def quadkey_to_tile(
    quadkey: str,
) -> tuple[int, int, int]:
    """
    Decode a Bing Maps quadkey into (x, y, level_of_detail).

    Microsoft documents GlobalMLBuildingFootprints as partitioned by country
    and L9 quadkey.
    """
    quadkey = str(
        quadkey
    ).strip()

    x = 0
    y = 0
    lod = len(quadkey)

    for i, char in enumerate(
        quadkey
    ):
        bit = lod - i - 1

        mask = 1 << bit

        if char == "0":
            pass
        elif char == "1":
            x |= mask
        elif char == "2":
            y |= mask
        elif char == "3":
            x |= mask
            y |= mask
        else:
            raise ValueError(
                f"Invalid quadkey digit: {char}"
            )

    return x, y, lod


def tile_xy_to_lon_lat(
    x: int,
    y: int,
    lod: int,
) -> tuple[float, float]:
    """
    Upper-left corner of a Bing tile in WGS84 degrees.
    """
    n = 2 ** lod

    lon = (
        x / n * 360.0
        - 180.0
    )

    lat_rad = math.atan(
        math.sinh(
            math.pi
            * (
                1
                - 2 * y / n
            )
        )
    )

    lat = (
        lat_rad
        * 180.0
        / math.pi
    )

    return lon, lat


def quadkey_bounds(
    quadkey: str,
) -> tuple[float, float, float, float]:
    """
    Return (min_lon, min_lat, max_lon, max_lat).
    """
    x, y, lod = quadkey_to_tile(
        quadkey
    )

    min_lon, max_lat = tile_xy_to_lon_lat(
        x,
        y,
        lod,
    )

    max_lon, min_lat = tile_xy_to_lon_lat(
        x + 1,
        y + 1,
        lod,
    )

    return (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )


# ============================================================================
# SENTINEL AOI BOUNDS
# ============================================================================

def sentinel_bounds_wgs84(
    rgb_path: Path,
) -> tuple[float, float, float, float]:
    with rasterio.open(
        rgb_path
    ) as src:

        if src.crs is None:
            raise ValueError(
                f"{rgb_path} has no CRS."
            )

        bounds = src.bounds
        source_crs = src.crs

    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True,
    )

    corners = [
        (
            bounds.left,
            bounds.bottom,
        ),
        (
            bounds.left,
            bounds.top,
        ),
        (
            bounds.right,
            bounds.bottom,
        ),
        (
            bounds.right,
            bounds.top,
        ),
    ]

    transformed = [
        transformer.transform(
            x,
            y,
        )
        for x, y in corners
    ]

    lons = [
        p[0]
        for p in transformed
    ]

    lats = [
        p[1]
        for p in transformed
    ]

    return (
        min(lons),
        min(lats),
        max(lons),
        max(lats),
    )


def bbox_intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    aminx, aminy, amaxx, amaxy = a
    bminx, bminy, bmaxx, bmaxy = b

    return not (
        amaxx < bminx
        or amaxx < aminx
        or bmaxx < aminx
        or bmaxy < aminy
        or amaxy < bminy
        or amaxy < aminy
    )


# ============================================================================
# DATASET INDEX
# ============================================================================

def ensure_index(
    index_path: Path,
    download_if_missing: bool,
) -> Path:
    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if index_path.exists():
        print(
            f"Using local index:\n  {index_path}"
        )
        return index_path

    if not download_if_missing:
        raise FileNotFoundError(
            f"Dataset index not found:\n{index_path}"
        )

    print(
        "Downloading current Microsoft dataset-links.csv..."
    )

    download_file(
        CURRENT_INDEX_URL,
        index_path,
    )

    return index_path


def load_india_index(
    index_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(
        index_path
    )

    # Normalize common column naming variations.
    columns = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    required_aliases = {
        "location": [
            "location",
            "regionname",
            "region",
        ],
        "quadkey": [
            "quadkey",
            "quad_key",
        ],
        "url": [
            "url",
            "downloadurl",
            "download_url",
        ],
    }

    resolved = {}

    for target, aliases in required_aliases.items():
        found = None

        for alias in aliases:
            if alias in columns:
                found = columns[alias]
                break

        if found is None:
            raise ValueError(
                f"Could not find a {target} column in "
                f"{list(df.columns)}"
            )

        resolved[target] = found

    out = df.rename(
        columns={
            resolved["location"]: "Location",
            resolved["quadkey"]: "QuadKey",
            resolved["url"]: "Url",
        }
    ).copy()

    out["Location"] = (
        out["Location"]
        .astype(str)
        .str.strip()
    )

    out["QuadKey"] = (
        out["QuadKey"]
        .astype(str)
        .str.strip()
    )

    out["Url"] = (
        out["Url"]
        .astype(str)
        .str.strip()
    )

    india = out[
        out["Location"].str.lower()
        == INDIA_LOCATION.lower()
    ].copy()

    if india.empty:
        raise ValueError(
            "No India rows found in dataset-links.csv."
        )

    # Precompute partition geographic bounds once.
    bounds = []

    for qk in india["QuadKey"]:
        b = quadkey_bounds(
            qk
        )
        bounds.append(b)

    india[
        [
            "min_lon",
            "min_lat",
            "max_lon",
            "max_lat",
        ]
    ] = np.asarray(
        bounds,
        dtype=np.float64,
    )

    return india


# ============================================================================
# SELECT REQUIRED PARTITIONS
# ============================================================================

def select_partitions_for_aoi(
    india_index: pd.DataFrame,
    aoi_bounds: tuple[float, float, float, float],
) -> pd.DataFrame:

    min_lon, min_lat, max_lon, max_lat = (
        aoi_bounds
    )

    mask = (
        (india_index["max_lon"] >= min_lon)
        & (india_index["min_lon"] <= max_lon)
        & (india_index["max_lat"] >= min_lat)
        & (india_index["min_lat"] <= max_lat)
    )

    selected = india_index[
        mask
    ].copy()

    selected = selected.sort_values(
        [
            "QuadKey"
        ]
    ).reset_index(
        drop=True
    )

    return selected


# ============================================================================
# FOOTPRINT PARSING / RASTERIZATION
# ============================================================================

def iter_geojson_features(
    path: Path,
) -> Iterable[dict]:
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                feature = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            if feature.get(
                "geometry"
            ) is None:
                continue

            yield feature


def rasterize_city_fraction(
    rgb_path: Path,
    raw_parts: list[Path],
    output_path: Path,
    supersample: int,
) -> dict:
    """
    Rasterize all selected footprint partitions into one exact Sentinel grid.

    We process part-by-part at the supersampled grid and OR/max the occupied
    subpixels. This avoids merging all polygons into one giant GeoDataFrame.
    """
    if supersample < 1:
        raise ValueError(
            "supersample must be >= 1."
        )

    with rasterio.open(
        rgb_path
    ) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        bounds = src.bounds

    if crs is None:
        raise ValueError(
            f"{rgb_path} has no CRS."
        )

    fine_height = (
        height * supersample
    )
    fine_width = (
        width * supersample
    )

    fine_transform = (
        transform
        * Affine.scale(
            1.0 / supersample,
            1.0 / supersample,
        )
    )

    combined = np.zeros(
        (
            fine_height,
            fine_width,
        ),
        dtype=np.uint8,
    )

    # Exact AOI geometry in Sentinel CRS.
    aoi = box(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
    )

    total_features = 0
    total_after_clip = 0

    # Transform GeoJSON WGS84 polygons to Sentinel CRS per chunk/part.
    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )

    for part_index, path in enumerate(
        raw_parts,
        start=1,
    ):
        print(
            f"      part {part_index}/{len(raw_parts)}: "
            f"{path.name}"
        )

        geometries = []

        for feature in iter_geojson_features(
            path
        ):
            total_features += 1

            geom = shape(
                feature["geometry"]
            )

            if geom.is_empty:
                continue

            # Cheap WGS84 AOI-side filtering using the part's geographic
            # extent is handled by partition selection. The exact AOI clip
            # happens after reprojection.
            try:
                geom = gpd.GeoSeries(
                    [geom],
                    crs="EPSG:4326",
                ).to_crs(
                    crs
                ).iloc[0]
            except Exception as exc:
                raise RuntimeError(
                    f"Could not reproject geometry in {path}"
                ) from exc

            if geom.is_empty:
                continue

            if not geom.intersects(
                aoi
            ):
                continue

            clipped = geom.intersection(
                aoi
            )

            if clipped.is_empty:
                continue

            if not clipped.is_valid:
                try:
                    clipped = clipped.make_valid()
                except AttributeError:
                    clipped = clipped.buffer(0)

            if clipped.is_empty:
                continue

            if clipped.geom_type == "GeometryCollection":
                clipped = (
                    gpd.GeoSeries(
                        [clipped],
                        crs=crs,
                    )
                    .explode(
                        index_parts=False
                    )
                    .unary_union
                )

            total_after_clip += 1
            geometries.append(
                clipped
            )

        if not geometries:
            continue

        # Rasterize this part on the fine grid and combine with previous
        # parts. "max" gives union-like occupancy for subpixels.
        part_mask = rasterize(
            (
                (
                    geom,
                    1,
                )
                for geom in geometries
            ),
            out_shape=(
                fine_height,
                fine_width,
            ),
            transform=fine_transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        )

        combined = np.maximum(
            combined,
            part_mask,
        )

        del part_mask
        del geometries

    # Aggregate fine occupancy into Sentinel-sized area fractions.
    fraction = (
        combined.reshape(
            height,
            supersample,
            width,
            supersample,
        )
        .mean(
            axis=(
                1,
                3,
            )
        )
        .astype(np.float32)
    )

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
        tiled=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            fraction,
            1,
        )

    return {
        "parts_used": len(
            raw_parts
        ),
        "features_seen": int(
            total_features
        ),
        "features_after_clip": int(
            total_after_clip
        ),
        "fraction_mean": float(
            fraction.mean()
        ),
        "fraction_p95": float(
            np.percentile(
                fraction,
                95,
            )
        ),
        "fraction_max": float(
            fraction.max()
        ),
        "pixels_gt_0": int(
            np.count_nonzero(
                fraction > 0
            )
        ),
        "pixels_gt_0_5": int(
            np.count_nonzero(
                fraction > 0.5
            )
        ),
    }


def derive_features(
    fraction_path: Path,
    city_semantic_dir: Path,
) -> dict[str, float]:
    with rasterio.open(
        fraction_path
    ) as src:
        fraction = src.read(
            1
        ).astype(
            np.float32
        )

        profile = src.profile.copy()
        transform = src.transform

    building = fraction >= 0.01

    density = ndimage.uniform_filter(
        fraction,
        size=5,
        mode="nearest",
    ).astype(np.float32)

    dilated = ndimage.maximum_filter(
        fraction,
        size=3,
        mode="nearest",
    )

    eroded = ndimage.minimum_filter(
        fraction,
        size=3,
        mode="nearest",
    )

    edge = (
        dilated - eroded
    ).astype(np.float32)

    if np.any(building):
        distance_px = ndimage.distance_transform_edt(
            ~building
        )

        pixel_size_x = abs(
            transform.a
        )

        pixel_size_y = abs(
            transform.e
        )

        pixel_size_m = float(
            (pixel_size_x + pixel_size_y)
            / 2.0
        )

        distance = (
            distance_px
            * pixel_size_m
        ).astype(
            np.float32
        )

        distance[
            building
        ] = 0.0

    else:
        distance = np.full(
            fraction.shape,
            np.nan,
            dtype=np.float32,
        )

    arrays = {
        "building_density_50m.tif": density,
        "building_edge.tif": edge,
        "distance_to_building_m.tif": distance,
    }

    for filename, array in arrays.items():
        output = (
            city_semantic_dir
            / filename
        )

        output_profile = profile.copy()
        output_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=-9999.0,
            compress="deflate",
            predictor=2,
            tiled=True,
        )

        with rasterio.open(
            output,
            "w",
            **output_profile,
        ) as dst:
            dst.write(
                array,
                1,
            )

    return {
        "density_mean": float(
            density.mean()
        ),
        "density_p95": float(
            np.percentile(
                density,
                95,
            )
        ),
        "edge_mean": float(
            edge.mean()
        ),
        "distance_mean": float(
            np.nanmean(
                distance
            )
            if np.isfinite(distance).any()
            else math.nan
        ),
        "distance_p95": float(
            np.nanpercentile(
                distance,
                95,
            )
            if np.isfinite(distance).any()
            else math.nan
        ),
    }


# ============================================================================
# CITY REPAIR
# ============================================================================

def process_city(
    city: str,
    rgb_path: Path,
    india_index: pd.DataFrame,
    raw_root: Path,
    audit_rows: list[dict],
    supersample: int,
    overwrite_raster: bool,
) -> dict:
    print()
    print("=" * 78)
    print(
        f"CITY: {city} "
        f"({CITY_LABELS[city]})"
    )
    print("=" * 78)

    aoi_bounds = sentinel_bounds_wgs84(
        rgb_path
    )

    print(
        "Sentinel AOI WGS84:"
    )
    print(
        "  "
        f"lon={aoi_bounds[0]:.6f}..{aoi_bounds[2]:.6f}, "
        f"lat={aoi_bounds[1]:.6f}..{aoi_bounds[3]:.6f}"
    )

    selected = select_partitions_for_aoi(
        india_index,
        aoi_bounds,
    )

    if selected.empty:
        raise RuntimeError(
            f"No GlobalMLBuildingFootprints partitions intersect "
            f"{city}."
        )

    city_raw = (
        raw_root / city
    )

    city_raw.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Required partitions intersecting AOI: "
        f"{len(selected)}"
    )

    print(
        "QuadKeys:"
    )

    for qk in selected[
        "QuadKey"
    ].tolist():
        print(
            f"  {qk}"
        )

    local_parts = []

    for _, row in selected.iterrows():
        url = row["Url"]

        filename = url.rsplit(
            "/",
            1,
        )[-1]

        destination = (
            city_raw / filename
        )

        download_file(
            url,
            destination,
        )

        local_parts.append(
            destination
        )

        audit_rows.append(
            {
                "city": city,
                "quadkey": str(
                    row["QuadKey"]
                ),
                "url": url,
                "local_path": str(
                    destination
                ),
                "source_tile_min_lon": row[
                    "min_lon"
                ],
                "source_tile_min_lat": row[
                    "min_lat"
                ],
                "source_tile_max_lon": row[
                    "max_lon"
                ],
                "source_tile_max_lat": row[
                    "max_lat"
                ],
                "status": "selected",
            }
        )

    # Deduplicate paths just in case.
    local_parts = sorted(
        set(local_parts)
    )

    semantic_dir = (
        rgb_path.parent
        / "semantic"
    )

    fraction_path = (
        semantic_dir
        / "building_fraction.tif"
    )

    if (
        fraction_path.exists()
        and not overwrite_raster
    ):
        print(
            f"Existing raster found:\n"
            f"  {fraction_path}\n"
            "Skipping rasterization. Use --overwrite-raster "
            "to rebuild from the complete partition set."
        )

        return {
            "city": city,
            "required_partitions": len(
                selected
            ),
            "local_parts": len(
                local_parts
            ),
            "fraction_path": str(
                fraction_path
            ),
            "status": "raster_exists",
        }

    print()
    print(
        "Rasterizing complete partition set..."
    )

    stats = rasterize_city_fraction(
        rgb_path=rgb_path,
        raw_parts=local_parts,
        output_path=fraction_path,
        supersample=supersample,
    )

    print(
        f"  features after clip: "
        f"{stats['features_after_clip']:,}"
    )

    print(
        f"  mean fraction: "
        f"{stats['fraction_mean']:.6f}"
    )

    print(
        f"  p95 fraction: "
        f"{stats['fraction_p95']:.6f}"
    )

    print(
        f"  >0 pixels: "
        f"{stats['pixels_gt_0']:,}"
    )

    print(
        f"  >0.5 pixels: "
        f"{stats['pixels_gt_0_5']:,}"
    )

    derived_stats = derive_features(
        fraction_path,
        semantic_dir,
    )

    print(
        "Derived:"
    )

    print(
        f"  density mean: "
        f"{derived_stats['density_mean']:.6f}"
    )

    print(
        f"  distance p95: "
        f"{derived_stats['distance_p95']:.2f} m"
    )

    return {
        "city": city,
        "required_partitions": len(
            selected
        ),
        "local_parts": len(
            local_parts
        ),
        "features_after_clip": stats[
            "features_after_clip"
        ],
        "fraction_mean": stats[
            "fraction_mean"
        ],
        "fraction_p95": stats[
            "fraction_p95"
        ],
        "fraction_max": stats[
            "fraction_max"
        ],
        "pixels_gt_0": stats[
            "pixels_gt_0"
        ],
        "pixels_gt_0_5": stats[
            "pixels_gt_0_5"
        ],
        **derived_stats,
        "fraction_path": str(
            fraction_path
        ),
        "status": "rebuilt",
    }


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair GlobalMLBuildingFootprints coverage and rebuild "
            "Method-3 semantic rasters."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )

    parser.add_argument(
        "--sentinel-root",
        type=Path,
        default=DEFAULT_SENTINEL_ROOT,
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=(
            "Current Microsoft dataset-links.csv. "
            "If absent, it is downloaded."
        ),
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
    )

    parser.add_argument(
        "--audit-root",
        type=Path,
        default=DEFAULT_AUDIT_ROOT,
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        choices=CITY_ORDER,
        default=CITY_ORDER,
    )

    parser.add_argument(
        "--supersample",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--overwrite-raster",
        action="store_true",
    )

    parser.add_argument(
        "--index-only",
        action="store_true",
        help=(
            "Only compute/report partition coverage; do not download "
            "parts or rasterize."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    args = parse_args()

    if args.supersample < 1:
        raise ValueError(
            "--supersample must be >= 1."
        )

    sentinel_root = (
        args.sentinel_root.resolve()
    )

    index_path = (
        args.index.resolve()
    )

    raw_root = (
        args.raw_root.resolve()
    )

    audit_root = (
        args.audit_root.resolve()
    )

    audit_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ensure we have the current official index.
    ensure_index(
        index_path,
        download_if_missing=True,
    )

    print()
    print("=" * 78)
    print(
        "DepthWizard2 — Method 3 Coverage Repair"
    )
    print("=" * 78)
    print(
        f"Microsoft release: {CURRENT_RELEASE}"
    )
    print(
        f"Index:             {index_path}"
    )
    print(
        f"Cities:            {', '.join(args.cities)}"
    )
    print(
        f"Supersample:       {args.supersample}x"
    )
    print("=" * 78)

    india_index = load_india_index(
        index_path
    )

    print()
    print(
        f"India partitions in index: "
        f"{len(india_index):,}"
    )

    # ------------------------------------------------------------------------
    # Select partitions for every AOI first, before downloading anything.
    # ------------------------------------------------------------------------

    audit_rows: list[dict] = []

    coverage_summary = []

    for city in args.cities:

        rgb_path = (
            sentinel_root
            / city
            / f"{CITY_LABELS[city]}_RGB.tif"
        )

        if not rgb_path.exists():
            raise FileNotFoundError(
                f"Sentinel RGB missing:\n{rgb_path}"
            )

        aoi_bounds = sentinel_bounds_wgs84(
            rgb_path
        )

        selected = select_partitions_for_aoi(
            india_index,
            aoi_bounds,
        )

        if selected.empty:
            raise RuntimeError(
                f"No Microsoft partitions intersect {city}."
            )

        coverage_summary.append(
            {
                "city": city,
                "rgb_path": str(rgb_path),
                "aoi_min_lon": aoi_bounds[0],
                "aoi_min_lat": aoi_bounds[1],
                "aoi_max_lon": aoi_bounds[2],
                "aoi_max_lat": aoi_bounds[3],
                "required_partition_count": len(
                    selected
                ),
                "quadkeys": ",".join(
                    selected["QuadKey"]
                    .astype(str)
                ),
            }
        )

        for _, row in selected.iterrows():
            audit_rows.append(
                {
                    "city": city,
                    "quadkey": str(
                        row["QuadKey"]
                    ),
                    "url": row["Url"],
                    "source_tile_min_lon": row[
                        "min_lon"
                    ],
                    "source_tile_min_lat": row[
                        "min_lat"
                    ],
                    "source_tile_max_lon": row[
                        "max_lon"
                    ],
                    "source_tile_max_lat": row[
                        "max_lat"
                    ],
                }
            )

    summary_df = pd.DataFrame(
        coverage_summary
    )

    selected_df = pd.DataFrame(
        audit_rows
    )

    summary_path = (
        audit_root
        / "coverage_audit.csv"
    )

    selected_path = (
        audit_root
        / "selected_partitions.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    selected_df.to_csv(
        selected_path,
        index=False,
    )

    coverage_json = {
        "release": CURRENT_RELEASE,
        "index": str(index_path),
        "cities": coverage_summary,
    }

    (
        audit_root
        / "coverage_audit.json"
    ).write_text(
        json.dumps(
            coverage_json,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "COVERAGE AUDIT"
    )
    print(
        summary_df[
            [
                "city",
                "required_partition_count",
                "quadkeys",
            ]
        ].to_string(
            index=False
        )
    )

    if args.index_only:
        print()
        print(
            "Index-only mode complete."
        )
        print(
            f"Audit: {summary_path}"
        )
        print(
            f"Parts: {selected_path}"
        )
        return 0

    # ------------------------------------------------------------------------
    # Download + rasterize.
    # ------------------------------------------------------------------------

    city_results = []

    # Load selected partitions city-wise.
    for city in args.cities:

        rgb_path = (
            sentinel_root
            / city
            / f"{CITY_LABELS[city]}_RGB.tif"
        )

        selected_for_city = selected_df[
            selected_df[
                "city"
            ] == city
        ].copy()

        local_parts = []

        city_raw = (
            raw_root / city
        )

        city_raw.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 78)
        print(
            f"DOWNLOADING / PREPARING {CITY_LABELS[city]}"
        )
        print("=" * 78)

        for _, row in selected_for_city.iterrows():

            url = row["url"]

            filename = url.rsplit(
                "/",
                1,
            )[-1]

            destination = (
                city_raw / filename
            )

            download_file(
                url,
                destination,
            )

            local_parts.append(
                destination
            )

        if not local_parts:
            raise RuntimeError(
                f"No local parts selected for {city}."
            )

        if args.overwrite_raster:
            semantic_dir = (
                rgb_path.parent
                / "semantic"
            )

            fraction_path = (
                semantic_dir
                / "building_fraction.tif"
            )

            if fraction_path.exists():
                print(
                    f"  removing previous semantic raster: "
                    f"{fraction_path.name}"
                )

                # Remove semantic outputs generated by this pipeline.
                for name in [
                    "building_fraction.tif",
                    "building_density_50m.tif",
                    "building_edge.tif",
                    "distance_to_building_m.tif",
                ]:
                    p = semantic_dir / name
                    if p.exists():
                        p.unlink()

        result = process_city(
            city=city,
            rgb_path=rgb_path,
            india_index=india_index,
            raw_root=raw_root,
            audit_rows=[],
            supersample=args.supersample,
            overwrite_raster=True,
        )

        city_results.append(
            result
        )

    results_df = pd.DataFrame(
        city_results
    )

    result_path = (
        audit_root
        / "repair_results.csv"
    )

    results_df.to_csv(
        result_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------------------

    report = {
        "release": CURRENT_RELEASE,
        "index": str(index_path),
        "supersample": args.supersample,
        "cities": coverage_summary,
        "results": city_results,
        "output_root": str(
            raw_root
        ),
        "method3_semantic_product": (
            "building_fraction + derived density/edge/distance"
        ),
    }

    (
        audit_root
        / "repair_results.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "COVERAGE REPAIR COMPLETE"
    )
    print("=" * 78)

    print(
        f"Coverage audit: "
        f"{summary_path}"
    )

    print(
        f"Selected parts: "
        f"{selected_path}"
    )

    print(
        f"Results: "
        f"{result_path}"
    )

    print()
    print(
        results_df[
            [
                "city",
                "required_partitions",
                "features_after_clip",
                "fraction_mean",
                "fraction_p95",
                "pixels_gt_0",
                "pixels_gt_0_5",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Next step: regenerate the Method-3 QC PDF before considering 10x "
        "supersampling."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
