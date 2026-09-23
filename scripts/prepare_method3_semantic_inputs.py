#!/usr/bin/env python3
"""
DepthWizard2 — Method 3 semantic-prior data preparation.

What this script does
---------------------
1. Moves the 10 Sentinel-2 RGB TIFFs from ~/Downloads into the project's
   canonical data/sentinel2/<city>/ location.
2. Downloads the exact Microsoft GlobalMLBuildingFootprints .csv.gz parts
   supplied for each city.
3. Optionally downloads the HOTOSM DINOv3-S building segmenter ONNX model.
   NOTE: the segmenter is for the DFC/VHR branch, NOT for 10 m Sentinel-2.
4. Rasterizes GlobalMLBuildingFootprints onto the EXACT Sentinel pixel grid.
5. Produces a building_fraction raster using supersampled rasterization.
6. Optionally derives building density, edge, and distance-to-building maps.

The Microsoft files are GeoJSONL despite the .csv.gz extension.

The canonical Sentinel structure produced is:
    data/sentinel2/
        kolkata/
            Kolkata_RGB.tif
            semantic/
                building_fraction.tif
                building_density_50m.tif
                building_edge.tif
                distance_to_building_m.tif
        ...

Microsoft raw footprint parts are stored separately:
    data/sentinel2/semantic_sources/globalml_building_footprints/raw/<city>/

Model:
    models/semantic/hotosm_dinov3s_buildings/model.onnx

Design notes
------------
- Sentinel polygons are NOT turned into a hard 0/1 mask.
- building_fraction is estimated by rasterizing the polygons on a fine
  sub-grid and averaging back to the exact Sentinel grid.
- Default supersampling is 5x (2 m subpixels for a 10 m Sentinel pixel).
  Use --supersample 10 for a higher-resolution final run if the machine has
  enough RAM/time.
- No Microsoft building-height attribute is used here. Method 3 is testing
  semantic/building structure as a prior, not importing another height source.
- All cities are processed independently and sequentially to limit memory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box, shape


# ============================================================================
# PROJECT / INPUT DEFINITIONS
# ============================================================================

DEFAULT_PROJECT_ROOT = Path(
    "/Users/anweshasaha/projects/DepthWizard2"
)

DEFAULT_DOWNLOADS = Path(
    "/Users/anweshasaha/Downloads"
)

# Exact user-provided GlobalMLBuildingFootprints parts.
FOOTPRINT_URLS: dict[str, list[str]] = {
    "kolkata": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133321/part-00108-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133323/part-00047-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "bardhaman": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133302/part-00180-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133303/part-00066-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133320/part-00122-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133321/part-00108-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "delhi": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123121303/part-00170-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "mumbai": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123300311/part-00055-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "sundarbans": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133332/part-00057-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "darjeeling": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123131323/part-00139-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123133101/part-00175-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "bengaluru": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123303312/part-00014-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "hyderabad": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123301331/part-00073-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "jaipur": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123123011/part-00047-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
    "kochi": [
        "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=India/quadkey=123321102/part-00113-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz",
    ],
}

# Canonical Downloads filenames -> project city.
RGB_INPUTS: dict[str, str] = {
    "Kolkata_Sentinel2_RGB.tif": "kolkata",
    "Bardhaman_Sentinel2_RGB.tif": "bardhaman",
    "Darjeeling_Sentinel2_RGB.tif": "darjeeling",
    "Sundarbans_Sentinel2_RGB.tif": "sundarbans",
    "Delhi_Sentinel2_RGB.tif": "delhi",
    "Mumbai_Sentinel2_RGB.tif": "mumbai",
    "Bengaluru_Sentinel2_RGB.tif": "bengaluru",
    "Hyderabad_Sentinel2_RGB.tif": "hyderabad",
    "Jaipur_Sentinel2_RGB.tif": "jaipur",
    "Kochi_Sentinel2_RGB.tif": "kochi",
}


MODEL_REPO = "hotosm/dinov3s-buildings"
MODEL_FILENAME = "model.onnx"


# ============================================================================
# UTILITIES
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


def safe_move(
    source: Path,
    destination: Path,
) -> None:
    """
    Move source -> destination.

    The user explicitly requested overwrite semantics for this preparation
    run. Therefore an existing destination is removed first. This is limited
    to the known Sentinel RGB destination generated by this script.
    """
    if not source.exists():
        if destination.exists():
            print(
                f"  source missing but destination exists; "
                f"keeping existing: {destination}"
            )
            return
        raise FileNotFoundError(
            f"Source file not found: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        print(
            f"  overwriting existing: {destination}"
        )
        destination.unlink()

    print(
        f"  moving {source.name} -> {destination}"
    )

    shutil.move(
        str(source),
        str(destination),
    )


def download_file(
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"  already downloaded: {destination.name}"
        )
        return

    tmp = destination.with_suffix(
        destination.suffix + ".part"
    )

    if tmp.exists():
        tmp.unlink()

    print(
        f"  downloading: {destination.name}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "DepthWizard2/semantic-prior-preparation"
            )
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
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
# RGB ORGANIZATION
# ============================================================================

def organize_rgb(
    project_root: Path,
    downloads_dir: Path,
    cities: list[str],
) -> dict[str, Path]:
    sentinel_root = (
        project_root / "data" / "sentinel2"
    )

    resolved: dict[str, Path] = {}

    print()
    print("=" * 78)
    print("1. ORGANIZING SENTINEL-2 RGB INPUTS")
    print("=" * 78)

    for filename, city in RGB_INPUTS.items():
        if city not in cities:
            continue

        source = downloads_dir / filename

        canonical_name = (
            f"{city.title()}_RGB.tif"
        )

        destination = (
            sentinel_root
            / city
            / canonical_name
        )

        safe_move(
            source,
            destination,
        )

        resolved[city] = destination

    # Validate all requested cities exist.
    missing = [
        city
        for city in cities
        if city not in resolved
    ]

    if missing:
        raise RuntimeError(
            "Could not resolve RGB TIFF(s) for:\n"
            + "\n".join(missing)
        )

    print(
        f"\nResolved {len(resolved)} Sentinel RGB inputs."
    )

    return resolved


# ============================================================================
# FOOTPRINT DOWNLOAD
# ============================================================================

def download_footprints(
    project_root: Path,
    cities: list[str],
) -> dict[str, list[Path]]:
    raw_root = (
        project_root
        / "data"
        / "sentinel2"
        / "semantic_sources"
        / "globalml_building_footprints"
        / "raw"
    )

    print()
    print("=" * 78)
    print("2. DOWNLOADING GLOBALMLBUILDINGFOOTPRINTS PARTS")
    print("=" * 78)

    output: dict[str, list[Path]] = {}

    for city in cities:
        city_dir = raw_root / city
        city_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output[city] = []

        urls = FOOTPRINT_URLS.get(city, [])

        if not urls:
            raise KeyError(
                f"No footprint URLs configured for {city}"
            )

        print(
            f"\n--- {city} ({len(urls)} part(s)) ---"
        )

        for url in urls:
            filename = url.rsplit(
                "/",
                1,
            )[-1]

            destination = (
                city_dir / filename
            )

            download_file(
                url,
                destination,
            )

            output[city].append(
                destination
            )

    return output


# ============================================================================
# OPTIONAL DFC BUILDING MODEL
# ============================================================================

def download_hotosm_model(
    project_root: Path,
) -> Path:
    model_dir = (
        project_root
        / "models"
        / "semantic"
        / "hotosm_dinov3s_buildings"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        model_dir / MODEL_FILENAME
    )

    print()
    print("=" * 78)
    print("3. DOWNLOADING HOTOSM DINOv3-S BUILDING MODEL")
    print("=" * 78)

    # Prefer Python huggingface_hub because it handles the HF URL/files cleanly.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed.\n"
            "Install it with:\n"
            "  pip install -U huggingface_hub\n"
            "Then rerun with --download-model."
        )

    if destination.exists() and destination.stat().st_size > 0:
        print(
            f"  already present: {destination}"
        )
        return destination

    cached = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )

    cached_path = Path(cached)

    if cached_path.resolve() != destination.resolve():
        shutil.copy2(
            cached_path,
            destination,
        )

    print(
        f"  model saved to: {destination}"
    )

    return destination


# ============================================================================
# GLOBALMLBUILDINGFOOTPRINT PARSING
# ============================================================================

def read_geojsonl_gz(
    paths: list[Path],
) -> gpd.GeoDataFrame:
    """
    Read Microsoft's .csv.gz files.

    The files are GeoJSON Lines despite the csv.gz extension.
    We parse line-by-line so the workflow does not depend on GDAL recognizing
    the unusual extension.
    """
    geometries = []
    records = []

    for path in paths:
        print(
            f"    parsing {path.name}"
        )

        with gzip.open(
            path,
            mode="rt",
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
                    feature = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}"
                    ) from exc

                geometry = feature.get(
                    "geometry"
                )

                if geometry is None:
                    continue

                try:
                    geom = shape(
                        geometry
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Invalid geometry at "
                        f"{path}:{line_number}"
                    ) from exc

                if geom.is_empty:
                    continue

                properties = feature.get(
                    "properties",
                    {},
                )

                geometries.append(
                    geom
                )
                records.append(
                    properties
                )

    if not geometries:
        return gpd.GeoDataFrame(
            {"geometry": []},
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs="EPSG:4326",
    )

    return gdf


# ============================================================================
# FRACTIONAL RASTERIZATION
# ============================================================================

def ensure_valid_geometry(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        return gdf

    # Shapely >= 2 / GeoPandas exposes make_valid.
    try:
        invalid = ~gdf.geometry.is_valid

        if invalid.any():
            gdf.loc[
                invalid,
                "geometry",
            ] = (
                gdf.loc[
                    invalid,
                    "geometry",
                ].make_valid()
            )

    except AttributeError:
        # Older environments: safer fallback than silently keeping bad shapes.
        invalid = ~gdf.geometry.is_valid

        if invalid.any():
            gdf.loc[
                invalid,
                "geometry",
            ] = gdf.loc[
                invalid,
                "geometry",
            ].buffer(0)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    return gdf


def rasterize_fraction(
    gdf: gpd.GeoDataFrame,
    rgb_path: Path,
    output_path: Path,
    supersample: int = 5,
) -> dict[str, float]:
    """
    Supersampled area-fraction rasterization.

    For every original Sentinel pixel:
        building_fraction =
            mean(fine subpixels covered by buildings)

    This is an approximation to geometric area fraction. With 5x:
        10m pixel -> 25 subpixels at 2m x 2m.
    With 10x:
        10m pixel -> 100 subpixels at 1m x 1m.

    The output remains on EXACTLY the Sentinel raster grid.
    """
    if supersample < 1:
        raise ValueError(
            "supersample must be >= 1"
        )

    with rasterio.open(
        rgb_path
    ) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    if crs is None:
        raise ValueError(
            f"{rgb_path} has no CRS."
        )

    # Reproject footprints onto Sentinel CRS.
    if gdf.crs is None:
        gdf = gdf.set_crs(
            "EPSG:4326"
        )

    if str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(
            crs
        )

    # Clip by Sentinel bounding box before rasterization.
    bounds_geom = box(
        *(
            rasterio.transform.array_bounds(
                height,
                width,
                transform,
            )
        )
    )

    # Make a GeoSeries/GeoDataFrame for spatial filtering.
    aoi = gpd.GeoDataFrame(
        {"geometry": [bounds_geom]},
        geometry="geometry",
        crs=crs,
    )

    try:
        gdf = gpd.clip(
            gdf,
            aoi,
        )
    except Exception:
        # Spatial index is useful but not required for correctness.
        intersects = gdf.geometry.intersects(
            bounds_geom
        )
        gdf = gdf.loc[
            intersects
        ].copy()

    gdf = ensure_valid_geometry(
        gdf
    )

    if gdf.empty:
        # Produce a valid all-zero semantic raster.
        fraction = np.zeros(
            (height, width),
            dtype=np.float32,
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
            "footprints_after_clip": 0,
            "building_fraction_mean": 0.0,
            "building_fraction_max": 0.0,
            "building_pixels_gt_0": 0,
        }

    # Fine sub-grid.
    fine_height = height * supersample
    fine_width = width * supersample

    fine_transform = (
        transform
        * Affine.scale(
            1.0 / supersample,
            1.0 / supersample,
        )
    )

    # uint8 is sufficient because overlapping building polygons are treated
    # as occupied, not accumulated.
    fine_mask = np.zeros(
        (
            fine_height,
            fine_width,
        ),
        dtype=np.uint8,
    )

    shapes = (
        (
            geom,
            1,
        )
        for geom in gdf.geometry
    )

    rasterize(
        shapes=shapes,
        out=fine_mask,
        transform=fine_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )

    # Aggregate each supersample x supersample block.
    #
    # This reshape is exact because the fine grid was constructed as a direct
    # integer refinement of the Sentinel grid.
    reshaped = fine_mask.reshape(
        height,
        supersample,
        width,
        supersample,
    )

    fraction = (
        reshaped.mean(
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
        "footprints_after_clip": int(
            len(gdf)
        ),
        "building_fraction_mean": float(
            fraction.mean()
        ),
        "building_fraction_max": float(
            fraction.max()
        ),
        "building_pixels_gt_0": int(
            np.count_nonzero(
                fraction > 0
            )
        ),
    }


# ============================================================================
# DERIVED SEMANTIC FEATURES
# ============================================================================

def write_derived_features(
    fraction_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Derive simple spatial semantic features from building_fraction.

    1. building_density_50m:
       5x5 mean filter over 10 m pixels.

    2. building_edge:
       morphological gradient of a soft building-occupancy mask.

    3. distance_to_building_m:
       Euclidean distance from each pixel to the nearest building-occupied
       pixel, using the 10 m Sentinel grid.

    The distance map is capped only by the raster extent, not arbitrarily.
    """
    with rasterio.open(
        fraction_path
    ) as src:
        fraction = src.read(1).astype(
            np.float32
        )
        profile = src.profile.copy()
        transform = src.transform

    building = (
        fraction >= 0.01
    )

    # Approximately 50m neighborhood.
    density_50m = ndimage.uniform_filter(
        fraction,
        size=5,
        mode="nearest",
    ).astype(np.float32)

    # Soft/robust edge map.
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

    # Distance from non-building pixels to nearest building pixel.
    # Distance is measured in pixels, then converted to meters.
    #
    # If there are no buildings, distance is NaN rather than an arbitrary
    # sentinel value.
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

        # For our Sentinel product these are both ~10m; use the mean to
        # preserve correctness if the grid is rectangular.
        pixel_size_m = float(
            np.mean(
                [
                    pixel_size_x,
                    pixel_size_y,
                ]
            )
        )

        distance_m = (
            distance_px * pixel_size_m
        ).astype(np.float32)

        # Building pixels themselves should have zero distance.
        distance_m[building] = 0.0

    else:
        distance_m = np.full(
            fraction.shape,
            np.nan,
            dtype=np.float32,
        )

    outputs = {
        "building_density_50m.tif": density_50m,
        "building_edge.tif": edge,
        "distance_to_building_m.tif": distance_m,
    }

    output_paths: dict[str, Path] = {}

    for filename, array in outputs.items():
        path = output_dir / filename

        out_profile = profile.copy()
        out_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=-9999.0,
            compress="deflate",
            predictor=2,
            tiled=True,
        )

        with rasterio.open(
            path,
            "w",
            **out_profile,
        ) as dst:
            dst.write(
                array,
                1,
            )

        output_paths[filename] = path

    return output_paths


# ============================================================================
# CITY PROCESSING
# ============================================================================

def process_city(
    city: str,
    rgb_path: Path,
    footprint_paths: list[Path],
    project_root: Path,
    supersample: int,
    derive_features: bool,
) -> dict:
    print()
    print("-" * 78)
    print(f"CITY: {city}")
    print("-" * 78)

    semantic_dir = (
        rgb_path.parent / "semantic"
    )

    semantic_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"  reading {len(footprint_paths)} footprint part(s)..."
    )

    gdf = read_geojsonl_gz(
        footprint_paths
    )

    print(
        f"  parsed {len(gdf):,} footprint(s)"
    )

    fraction_path = (
        semantic_dir
        / "building_fraction.tif"
    )

    stats = rasterize_fraction(
        gdf=gdf,
        rgb_path=rgb_path,
        output_path=fraction_path,
        supersample=supersample,
    )

    print(
        f"  fraction: {fraction_path}"
    )

    print(
        f"  clipped footprints: "
        f"{stats['footprints_after_clip']:,}"
    )

    print(
        f"  mean building fraction: "
        f"{stats['building_fraction_mean']:.6f}"
    )

    print(
        f"  max building fraction: "
        f"{stats['building_fraction_max']:.6f}"
    )

    print(
        f"  pixels with building fraction > 0: "
        f"{stats['building_pixels_gt_0']:,}"
    )

    derived = {}

    if derive_features:
        print(
            "  deriving density / edge / distance features..."
        )

        derived = write_derived_features(
            fraction_path,
            semantic_dir,
        )

        for name, path in derived.items():
            print(
                f"    {name}: {path}"
            )

    return {
        "city": city,
        "rgb": str(rgb_path),
        "footprint_parts": len(
            footprint_paths
        ),
        "footprints_parsed": int(
            len(gdf)
        ),
        "fraction_path": str(
            fraction_path
        ),
        **stats,
        "derived": {
            name: str(path)
            for name, path in derived.items()
        },
    }


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Method-3 semantic inputs from Sentinel RGB "
            "and GlobalMLBuildingFootprints."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )

    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=DEFAULT_DOWNLOADS,
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        choices=sorted(set(RGB_INPUTS.values())),
        default=sorted(set(RGB_INPUTS.values())),
    )

    parser.add_argument(
        "--supersample",
        type=int,
        default=5,
        help=(
            "Fine-grid multiplier for building_fraction. "
            "5 = 2m subpixels for a 10m Sentinel grid; "
            "10 = 1m subpixels."
        ),
    )

    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Do not download GlobalMLBuildingFootprints parts.",
    )

    parser.add_argument(
        "--skip-rasterize",
        action="store_true",
        help="Download/organize inputs but do not rasterize.",
    )

    parser.add_argument(
        "--no-derived",
        action="store_true",
        help=(
            "Only write building_fraction.tif; skip density/edge/distance."
        ),
    )

    parser.add_argument(
        "--download-model",
        action="store_true",
        help=(
            "Download HOTOSM DINOv3-S model.onnx for the DFC/VHR branch. "
            "Not used on 10m Sentinel."
        ),
    )

    parser.add_argument(
        "--overwrite-raster",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    args = parse_args()

    if args.supersample < 1:
        raise ValueError(
            "--supersample must be >= 1"
        )

    project_root = (
        args.project_root.resolve()
    )

    downloads_dir = (
        args.downloads_dir.resolve()
    )

    cities = list(
        args.cities
    )

    unknown_cities = [
        city
        for city in cities
        if city not in RGB_INPUTS.values()
    ]
    if unknown_cities:
        raise ValueError(
            "Unknown city name(s): "
            + ", ".join(unknown_cities)
        )

    print()
    print("=" * 78)
    print("DepthWizard2 — Method 3 Semantic Input Preparation")
    print("=" * 78)
    print(
        f"Project root: {project_root}"
    )
    print(
        f"Downloads:    {downloads_dir}"
    )
    print(
        f"Cities:       {', '.join(cities)}"
    )
    print(
        f"Supersample:  {args.supersample}x"
    )
    print()

    # 1. Move/organize RGBs.
    rgb_paths = organize_rgb(
        project_root=project_root,
        downloads_dir=downloads_dir,
        cities=cities,
    )

    # 2. Download Microsoft footprints.
    footprint_paths = {}

    if args.skip_downloads:
        print()
        print(
            "Skipping footprint downloads."
        )

        raw_root = (
            project_root
            / "data"
            / "sentinel2"
            / "semantic_sources"
            / "globalml_building_footprints"
            / "raw"
        )

        for city in cities:
            city_dir = raw_root / city

            files = sorted(
                city_dir.glob(
                    "*.csv.gz"
                )
            )

            if not files:
                raise FileNotFoundError(
                    f"No local footprint parts found for {city}:\n"
                    f"{city_dir}"
                )

            footprint_paths[city] = files

    else:
        footprint_paths = download_footprints(
            project_root=project_root,
            cities=cities,
        )

    # 3. Optional DFC/VHR model.
    model_path = None

    if args.download_model:
        model_path = download_hotosm_model(
            project_root
        )

    # 4. Rasterize.
    summaries = []

    if args.skip_rasterize:
        print()
        print(
            "Skipping rasterization."
        )

    else:
        for city in cities:
            rgb_path = rgb_paths[city]

            # If user reruns without --overwrite-raster and the output exists,
            # keep it instead of silently replacing a potentially expensive
            # final raster.
            fraction_path = (
                rgb_path.parent
                / "semantic"
                / "building_fraction.tif"
            )

            if (
                fraction_path.exists()
                and not args.overwrite_raster
            ):
                print()
                print(
                    f"[{city}] building_fraction.tif already exists; "
                    "skipping rasterization. "
                    "Use --overwrite-raster to regenerate."
                )

                summaries.append(
                    {
                        "city": city,
                        "rgb": str(rgb_path),
                        "fraction_path": str(
                            fraction_path
                        ),
                        "status": "existing",
                    }
                )

                continue

            summary = process_city(
                city=city,
                rgb_path=rgb_path,
                footprint_paths=(
                    footprint_paths[city]
                ),
                project_root=project_root,
                supersample=args.supersample,
                derive_features=(
                    not args.no_derived
                ),
            )

            summary["status"] = "created"

            summaries.append(
                summary
            )

    # 5. Save experiment manifest.
    manifest_dir = (
        project_root
        / "data"
        / "sentinel2"
        / "semantic_sources"
        / "globalml_building_footprints"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "dataset": "Microsoft GlobalMLBuildingFootprints",
        "release": "2026-08-13",
        "cities": cities,
        "rgb_paths": {
            city: str(
                rgb_paths[city]
            )
            for city in cities
        },
        "footprint_paths": {
            city: [
                str(p)
                for p in footprint_paths[city]
            ]
            for city in cities
        },
        "supersample": args.supersample,
        "derived_features": (
            not args.no_derived
        ),
        "hotosm_model": (
            str(model_path)
            if model_path is not None
            else None
        ),
        "warnings": [
            (
                "GlobalMLBuildingFootprints is an external "
                "building semantic prior, not ground truth for "
                "the Sentinel acquisition."
            ),
            (
                "building_fraction is a supersampled approximation "
                "of within-pixel coverage."
            ),
            (
                "HOTOSM DINOv3-S is intended for VHR RGB inference "
                "and is not used on 10m Sentinel in this workflow."
            ),
        ],
    }

    manifest_path = (
        manifest_dir
        / "method3_input_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)

    print(
        f"Manifest: {manifest_path}"
    )

    if model_path:
        print(
            f"DINOv3-S ONNX: {model_path}"
        )

    if summaries:
        print()
        print(
            "Raster summary:"
        )

        for item in summaries:
            print(
                f"  {item['city']:12s} "
                f"{item.get('status', '')}"
            )

    print()
    print(
        "Next Method-3 experiment:"
    )
    print(
        "  DFC RGB -> building probability -> "
        "DAv2 + semantic features -> metric regression"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except KeyboardInterrupt:
        print(
            "\nInterrupted.",
            file=sys.stderr,
        )
        raise SystemExit(130)
