"""Verify and quantify the vertical-datum hypothesis found by
diagnose_error_floor.py: Copernicus GLO-30 heights are EGM2008 orthometric
(above-geoid), while the ICESat-2 h_ph field used throughout this project
as ground truth is WGS84 ellipsoidal height (explicitly documented as such
in backbone-comparison.md). These are different vertical references, and
the difference between them -- the geoid undulation N -- was measured
directly for kutch/nagapattinam/kakinada/manali via native-grid DEM vs.
ICESat-2 comparison (near-zero scatter, i.e. almost pure bias):

    tile          | observed DEM-vs-ICESat2 bias | independently-computed N (EGM2008 geoid grid, PROJ_NETWORK)
    kutch         | +49.3 m                      | 49.18 m
    nagapattinam  | +95.0 m                      | 95.26 m
    kakinada      | +76.1 m                      | 76.21 m
    manali        | +27.7 m                      | 23.66 m

Match to within a few tens of cm on the three flat tiles -- this IS the
mechanism, not a coincidence. This script applies the correction (predict
in the DEM's orthometric frame, subtract the tile's own geoid undulation N
before comparing to ICESat-2's ellipsoidal height) to all 25 accepted
tiles and reports how much of the earlier "flat terrain calibration looks
unreliable" finding survives.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats

from lib_backbone_correlation import sample_depth_at_photons
from detect_sign_flip import reproject_dem_to_tile_grid

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
DEM_DIR = ROOT / "data/sentinel2_benchmark/copernicus_dem_raw"
PHOTON_DIR = ROOT / "data/icesat2_photons"
CAL_CSV = ROOT / "data/sentinel2_benchmark/calibration_results_with_range.csv"


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def geoid_undulation_n(lon: float, lat: float) -> float:
    """N such that DEM_orthometric ~= ICESat2_ellipsoidal + N (matches the
    empirically observed bias sign/magnitude directly)."""
    t = Transformer.from_crs("EPSG:4979", "EPSG:3855", always_xy=True)
    _, _, n = t.transform(lon, lat, 0.0)
    return n


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    cal = pd.read_csv(CAL_CSV).set_index("tile_id")
    accepted = cal[cal["status"] == "calibrated"]

    rows = []
    for tile_id, row in accepted.iterrows():
        m = manifest.loc[tile_id]
        rgb_path = ROOT / m["rgb_path"]
        n_geoid = geoid_undulation_n(m["lon"], m["lat"])

        a, b = row["a_slope"], row["b_intercept"]
        dav2 = np.load(DAV2_DIR / f"{tile_id}_depth.npy")
        photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
        sampled = sample_depth_at_photons(rgb_path, dav2, photons)

        pred_orthometric = a * sampled["depth_value"].values + b
        pred_ellipsoidal = pred_orthometric - n_geoid  # datum-corrected
        true_icesat2 = sampled["height"].values

        rmse_before = rmse(pred_orthometric, true_icesat2)
        rmse_after = rmse(pred_ellipsoidal, true_icesat2)
        mae_after = mae(pred_ellipsoidal, true_icesat2)

        rows.append({
            "tile_id": tile_id, "category": m["category"],
            "n_geoid_m": n_geoid,
            "icesat2_rmse_m_before": rmse_before,
            "icesat2_rmse_m_after": rmse_after,
            "icesat2_mae_m_after": mae_after,
            "dem_elev_range_m": row["dem_elev_range_m"],
            "icesat2_rmse_pct_of_range_before": row["icesat2_rmse_pct_of_range"],
            "icesat2_rmse_pct_of_range_after": 100 * rmse_after / row["dem_elev_range_m"],
        })
        print(
            f"{tile_id:15s} N={n_geoid:+7.2f}m  "
            f"rmse: {rmse_before:7.1f}m -> {rmse_after:7.1f}m  "
            f"rmse%: {row['icesat2_rmse_pct_of_range']:7.1f}% -> {100*rmse_after/row['dem_elev_range_m']:7.1f}%"
        )

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/sentinel2_benchmark/geoid_correction_results.csv", index=False)

    print(f"\nMedian icesat2_rmse_pct_of_range BEFORE correction: {df['icesat2_rmse_pct_of_range_before'].median():.1f}%")
    print(f"Median icesat2_rmse_pct_of_range AFTER correction:  {df['icesat2_rmse_pct_of_range_after'].median():.1f}%")
    print(f"Tiles with RMSE% > 100% BEFORE: {(df['icesat2_rmse_pct_of_range_before'] > 100).sum()} / {len(df)}")
    print(f"Tiles with RMSE% > 100% AFTER:  {(df['icesat2_rmse_pct_of_range_after'] > 100).sum()} / {len(df)}")
    print(f"\nWrote {ROOT / 'data/sentinel2_benchmark/geoid_correction_results.csv'}")


if __name__ == "__main__":
    main()
