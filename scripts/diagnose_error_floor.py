"""Diagnose the absolute-error floor behind the DEM-vs-ICESat2 gap found
in calibration_results_with_range.csv. Two things:

1. Registration spot-check: for the 3 worst-gap tiles (kutch, nagapattinam,
   kakinada), verify each ICESat-2 photon's pixel row/col -- computed by
   lib_backbone_correlation.sample_depth_at_photons's manual affine-invert
   -- against rasterio's own, independently-implemented `.index()` method.
   If they disagree, that's a real pipeline bug. If they agree, the
   pipeline's pixel lookup is not the problem.

2. Reference-vs-reference disagreement, with DAv2 removed from the loop
   entirely: sample the ORIGINAL (native CRS/grid, not reprojected onto
   the tile's 10m UTM grid) Copernicus DEM directly at each ICESat-2
   photon's lon/lat, and compare THAT to the photon's true height. This
   tests whether GLO-30 and ICESat-2 simply disagree with each other on
   these tiles, independent of DAv2's calibration or any reprojection
   step -- if this "floor" is already large, no DAv2 calibration anchored
   to GLO-30 could ever beat it when checked against ICESat-2.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DEM_DIR = ROOT / "data/sentinel2_benchmark/copernicus_dem_raw"
PHOTON_DIR = ROOT / "data/icesat2_photons"
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DEM_NODATA = -32767.0

WORST_GAP_TILES = ["kutch", "nagapattinam", "kakinada"]
CONTROL_TILE = "manali"  # hilly, small gap, for contrast


def registration_spot_check(tile_id: str, rgb_path: Path, n_check: int = 8):
    """Compare the pipeline's manual affine-invert pixel lookup (used by
    lib_backbone_correlation.sample_depth_at_photons) against rasterio's
    own .index() for a sample of real photons."""
    photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv").sample(n=n_check, random_state=1)
    with rasterio.open(rgb_path) as src:
        crs = src.crs
        transform = src.transform

    to_tile_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_tile_crs.transform(photons["lon"].values, photons["lat"].values)

    inv = ~transform
    cols_manual, rows_manual = inv * (x, y)
    cols_manual = np.floor(cols_manual).astype(int)
    rows_manual = np.floor(rows_manual).astype(int)

    print(f"\n--- {tile_id}: registration spot-check (manual affine-invert vs rasterio.index) ---")
    all_match = True
    for i in range(n_check):
        row_r, col_r = rasterio.open(rgb_path).index(x[i], y[i])
        match = (row_r == rows_manual[i]) and (col_r == cols_manual[i])
        all_match &= match
        print(f"  photon {i}: manual=(row={rows_manual[i]}, col={cols_manual[i]})  rasterio.index=(row={row_r}, col={col_r})  {'OK' if match else 'MISMATCH'}")
    print(f"  => {'ALL MATCH -- no pixel-lookup bug found' if all_match else 'MISMATCH FOUND -- pipeline bug'}")
    return all_match


def native_dem_vs_icesat2(tile_id: str):
    """Sample the DEM in its OWN native grid/CRS at each photon's lon/lat
    -- no reprojection onto the tile's UTM grid at all -- and compare
    directly to true photon height. Isolates reference-vs-reference
    disagreement, with DAv2 and the reprojection step both removed."""
    dem_path = DEM_DIR / f"{tile_id}_dem.tif"
    photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        dem_crs = src.crs  # EPSG:4326, same as photon lon/lat -- no reprojection needed
        transform = src.transform
        height_px, width_px = src.height, src.width

    assert dem_crs.to_epsg() == 4326, f"expected native DEM in EPSG:4326, got {dem_crs}"

    inv = ~transform
    cols, rows = inv * (photons["lon"].values, photons["lat"].values)
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)
    in_bounds = (cols >= 0) & (cols < width_px) & (rows >= 0) & (rows < height_px)

    dem_vals = np.full(len(photons), np.nan, dtype=np.float32)
    dem_vals[in_bounds] = dem[rows[in_bounds], cols[in_bounds]]
    valid = in_bounds & (dem_vals != DEM_NODATA) & ~np.isnan(dem_vals)

    dem_valid = dem_vals[valid]
    height_valid = photons["height"].values[valid]

    r, _ = stats.pearsonr(dem_valid, height_valid)
    rmse = float(np.sqrt(np.mean((dem_valid - height_valid) ** 2)))
    mae = float(np.mean(np.abs(dem_valid - height_valid)))
    bias = float(np.mean(dem_valid - height_valid))
    return {
        "tile_id": tile_id, "n": int(valid.sum()),
        "native_dem_vs_icesat2_pearson": r,
        "native_dem_vs_icesat2_rmse_m": rmse,
        "native_dem_vs_icesat2_mae_m": mae,
        "native_dem_vs_icesat2_bias_m": bias,  # + = DEM reads higher than ICESat-2
    }


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")

    print("=== Part 1: registration spot-check (worst-gap tiles) ===")
    for tile_id in WORST_GAP_TILES:
        rgb_path = ROOT / manifest.loc[tile_id, "rgb_path"]
        registration_spot_check(tile_id, rgb_path)

    print("\n=== Part 2: native-grid DEM vs ICESat-2, DAv2 removed entirely ===")
    cal = pd.read_csv(ROOT / "data/sentinel2_benchmark/calibration_results_with_range.csv").set_index("tile_id")
    rows = []
    for tile_id in WORST_GAP_TILES + [CONTROL_TILE]:
        rec = native_dem_vs_icesat2(tile_id)
        rec["pipeline_icesat2_rmse_m"] = float(cal.loc[tile_id, "icesat2_rmse_m"])
        rec["pipeline_held_out_dem_rmse_m"] = float(cal.loc[tile_id, "held_out_dem_rmse_m"])
        rows.append(rec)
        print(
            f"{tile_id:15s} native DEM-vs-ICESat2: r={rec['native_dem_vs_icesat2_pearson']:+.4f} "
            f"rmse={rec['native_dem_vs_icesat2_rmse_m']:6.1f}m mae={rec['native_dem_vs_icesat2_mae_m']:6.1f}m "
            f"bias={rec['native_dem_vs_icesat2_bias_m']:+6.1f}m  (n={rec['n']})  |  "
            f"pipeline icesat2_rmse={rec['pipeline_icesat2_rmse_m']:6.1f}m  "
            f"pipeline dem_rmse={rec['pipeline_held_out_dem_rmse_m']:5.1f}m"
        )

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/sentinel2_benchmark/error_floor_diagnosis.csv", index=False)
    print(f"\nWrote {ROOT / 'data/sentinel2_benchmark/error_floor_diagnosis.csv'}")


if __name__ == "__main__":
    main()
