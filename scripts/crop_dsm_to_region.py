"""Crop a full-tile Copernicus GLO-30 DSM down to a region's Sentinel-2 footprint.

Darjeeling's OpenTopography DSM already arrived pre-cropped to its Sentinel
tile, so this step wasn't needed for it. The new Kolkata/Bardhaman/Sundarbans
Copernicus GLO-30 DSMs are full 1x1 degree tiles (EPSG:4326), so before
reusing the existing elevation.png/terrain.json generation scripts (which,
like Darjeeling's, read the DSM in its native CRS without reprojecting to
match the Sentinel tile's UTM grid), we crop each DSM to the Sentinel tile's
geographic bounds.
"""

import sys
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]

DSM_FILENAMES = {
    "kolkata": "Kolkata_Copernicus_GLO30_DSM.tif",
    "bardhaman": "Bardhaman_Copernicus_GLO30_DSM.tif",
    "sundarbans": "Sundarbans_Copernicus_GLO30_DSM.tif",
}


def crop(region: str) -> Path:
    sentinel_path = (
        ROOT / "data" / "sentinel2" / region / f"{region.capitalize()}_RGB.tif"
    )
    dsm_path = ROOT / "data" / "elevation" / region / DSM_FILENAMES[region]
    output_path = (
        ROOT
        / "data"
        / "elevation"
        / region
        / f"{region.capitalize()}_Copernicus_GLO30_DSM_cropped.tif"
    )

    with rasterio.open(sentinel_path) as sentinel_src:
        sentinel_bounds = sentinel_src.bounds
        sentinel_crs = sentinel_src.crs

    with rasterio.open(dsm_path) as dsm_src:
        west, south, east, north = transform_bounds(
            sentinel_crs, dsm_src.crs, *sentinel_bounds
        )

        window = from_bounds(west, south, east, north, transform=dsm_src.transform)
        window = window.round_offsets().round_lengths()

        data = dsm_src.read(1, window=window)
        transform = dsm_src.window_transform(window)

        profile = dsm_src.profile.copy()
        profile.update(
            height=data.shape[0],
            width=data.shape[1],
            transform=transform,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    print(f"{region}: cropped {dsm_path.name} -> {output_path.name}")
    print(f"  size: {data.shape[1]} x {data.shape[0]}")
    print(f"  elevation range: {data.min():.2f} - {data.max():.2f} m")

    return output_path


if __name__ == "__main__":
    regions = sys.argv[1:] or list(DSM_FILENAMES.keys())
    for region in regions:
        crop(region)
