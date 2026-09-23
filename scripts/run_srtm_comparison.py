"""SRTM (EGM96-corrected) calibration + 3-way DEM comparison, on the same
25 tiles the sign-flip detector already accepted (reused, not re-derived
from sign_flip_detector_verdicts.csv).

CRITICAL, verified before use (see the geoid-verification block at the
bottom of this file's __main__ and the printed output when run): SRTM's
vertical datum is EGM96, NOT EGM2008 (which Copernicus GLO-30/FABDEM use)
and NOT WGS84 ellipsoidal. Uses EPSG:5773 (EGM96 height), not EPSG:3855
(EGM2008 height, used for GLO-30/FABDEM in verify_geoid_datum_fix.py /
run_fabdem_comparison.py).

Does three things:
  1. Same held-out methodology as calibrate_accepted_tiles.py (50/50
     split, seed 42, independent ICESat-2 cross-check), against SRTM,
     with the EGM96 correction applied.
  2. 3-way table: SRTM vs GLO-30 vs FABDEM (all corrected to WGS84
     ellipsoidal), same 25 tiles, same metric (ICESat-2 RMSE as % of
     elevation range).
  3. Direct SRTM-vs-GLO-30 agreement check: reproject both onto the same
     tile grid, correct both to ellipsoidal height, and correlate them
     against EACH OTHER (not via ICESat-2) -- independent evidence for
     whether the geoid-correction methodology is genuinely working.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from pyproj import Transformer
from scipy import stats

from lib_backbone_correlation import sample_depth_at_photons
from detect_sign_flip import reproject_dem_to_tile_grid  # generic bilinear reproject-to-tile-grid

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
SRTM_DIR = ROOT / "data/sentinel2_benchmark/srtm_raw"
COP_DIR = ROOT / "data/sentinel2_benchmark/copernicus_dem_raw"
PHOTON_DIR = ROOT / "data/icesat2_photons"
VERDICTS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
COP_NODATA = -32767.0
SEED = 42

_T_EGM96 = Transformer.from_crs("EPSG:4979", "EPSG:5773", always_xy=True)
_T_EGM2008 = Transformer.from_crs("EPSG:4979", "EPSG:3855", always_xy=True)


def n_egm96(lon, lat):
    _, _, n = _T_EGM96.transform(lon, lat, 0.0)
    return n


def n_egm2008(lon, lat):
    _, _, n = _T_EGM2008.transform(lon, lat, 0.0)
    return n


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def calibrate_against_dem(tile_id, rgb_path, dem_path, n_geoid, dem_nodata=None):
    """Held-out fit (50/50, seed 42) against dem_path, ICESat-2 cross-check
    with the geoid correction applied to the fitted prediction."""
    dav2 = np.load(DAV2_DIR / f"{tile_id}_depth.npy")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        src_transform, src_crs, src_nodata = src.transform, src.crs, (dem_nodata if dem_nodata is not None else src.nodata)
    with rasterio.open(rgb_path) as rgb_src:
        dst_crs, dst_transform, dst_shape = rgb_src.crs, rgb_src.transform, (rgb_src.height, rgb_src.width)

    dem_on_grid = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=dem, destination=dem_on_grid,
        src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    valid = ~np.isnan(dem_on_grid)
    idx = np.flatnonzero(valid)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    half = len(idx) // 2
    train_idx = idx[:half]

    depth_flat, dem_flat = dav2.ravel(), dem_on_grid.ravel()
    lr = stats.linregress(depth_flat[train_idx], dem_flat[train_idx])
    a, b = lr.slope, lr.intercept

    photons = pd.read_csv(PHOTON_DIR / f"{tile_id}.csv")
    sampled = sample_depth_at_photons(rgb_path, dav2, photons)
    pred_ellipsoidal = a * sampled["depth_value"].values + b - n_geoid
    true_icesat2 = sampled["height"].values
    icesat2_rmse = rmse(pred_ellipsoidal, true_icesat2)

    dem_range = float(dem_on_grid[valid].max() - dem_on_grid[valid].min())
    return {
        "a": a, "b": b, "n_geoid": n_geoid,
        "icesat2_rmse_m": icesat2_rmse,
        "elev_range_m": dem_range,
        "icesat2_rmse_pct_of_range": 100 * icesat2_rmse / dem_range,
        "dem_on_grid": dem_on_grid,  # returned for the cross-DEM comparison step
    }


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    verdicts = pd.read_csv(VERDICTS_CSV).set_index("tile_id")
    accepted = verdicts[~verdicts["flagged"]].index.tolist()
    print(f"Reusing sign-flip detector's existing verdicts: {len(accepted)} accepted tiles.\n")

    rows = []
    dem_grids = {}  # tile_id -> (srtm_ellipsoidal_grid, glo30_ellipsoidal_grid) for the cross-check
    for tile_id in accepted:
        m = manifest.loc[tile_id]
        rgb_path = ROOT / m["rgb_path"]
        n96 = n_egm96(m["lon"], m["lat"])
        n08 = n_egm2008(m["lon"], m["lat"])

        srtm_res = calibrate_against_dem(tile_id, rgb_path, SRTM_DIR / f"{tile_id}_srtm.tif", n96)
        glo30_res = calibrate_against_dem(tile_id, rgb_path, COP_DIR / f"{tile_id}_dem.tif", n08, dem_nodata=COP_NODATA)

        dem_grids[tile_id] = (srtm_res["dem_on_grid"] - n96, glo30_res["dem_on_grid"] - n08)

        rows.append({
            "tile_id": tile_id, "category": m["category"],
            "n_egm96": n96, "n_egm2008": n08,
            "srtm_icesat2_rmse_pct_of_range": srtm_res["icesat2_rmse_pct_of_range"],
            "glo30_icesat2_rmse_pct_of_range": glo30_res["icesat2_rmse_pct_of_range"],
        })
        print(
            f"{tile_id:15s} N96={n96:+6.1f} N08={n08:+6.1f}  "
            f"SRTM rmse%={srtm_res['icesat2_rmse_pct_of_range']:6.1f}%  "
            f"GLO30 rmse%={glo30_res['icesat2_rmse_pct_of_range']:6.1f}%"
        )

    df = pd.DataFrame(rows).set_index("tile_id")

    # Pull in the already-computed FABDEM-corrected numbers (from the previous session's
    # ad-hoc fabdem_geoid_corrected.csv) for the 3-way table.
    fabdem_corrected = pd.read_csv(ROOT / "data/sentinel2_benchmark/fabdem_geoid_corrected.csv").set_index("tile_id")
    df["fabdem_icesat2_rmse_pct_of_range"] = fabdem_corrected["fabdem_rmse_pct_after"]

    print("\n=== 3-way comparison (corrected ICESat-2 RMSE as % of elevation range) ===")
    cols = ["category", "srtm_icesat2_rmse_pct_of_range", "glo30_icesat2_rmse_pct_of_range", "fabdem_icesat2_rmse_pct_of_range"]
    print(df[cols].sort_values("glo30_icesat2_rmse_pct_of_range").to_string())

    print("\nMedians: SRTM={:.1f}%  GLO-30={:.1f}%  FABDEM={:.1f}%".format(
        df["srtm_icesat2_rmse_pct_of_range"].median(),
        df["glo30_icesat2_rmse_pct_of_range"].median(),
        df["fabdem_icesat2_rmse_pct_of_range"].median(),
    ))

    best = df[["srtm_icesat2_rmse_pct_of_range", "glo30_icesat2_rmse_pct_of_range", "fabdem_icesat2_rmse_pct_of_range"]].idxmin(axis=1)
    print("\nWin counts (which DEM has the lowest RMSE% per tile):")
    print(best.value_counts())

    # Direct DEM-vs-DEM cross-check (both corrected to WGS84 ellipsoidal), same pixels.
    print("\n=== Direct SRTM-vs-GLO-30 agreement (both corrected to ellipsoidal, pixel-for-pixel) ===")
    cross_rows = []
    for tile_id in accepted:
        srtm_ell, glo30_ell = dem_grids[tile_id]
        valid = ~np.isnan(srtm_ell) & ~np.isnan(glo30_ell)
        r, _ = stats.pearsonr(srtm_ell[valid], glo30_ell[valid])
        mean_diff = float(np.mean(srtm_ell[valid] - glo30_ell[valid]))
        std_diff = float(np.std(srtm_ell[valid] - glo30_ell[valid]))
        cross_rows.append({"tile_id": tile_id, "srtm_vs_glo30_pearson": r, "mean_diff_m": mean_diff, "std_diff_m": std_diff})
        print(f"{tile_id:15s} r={r:.4f}  mean_diff={mean_diff:+6.2f}m  std_diff={std_diff:5.2f}m")

    cross_df = pd.DataFrame(cross_rows).set_index("tile_id")
    print(f"\nMedian pearson r: {cross_df['srtm_vs_glo30_pearson'].median():.4f}")
    print(f"Median |mean_diff|: {cross_df['mean_diff_m'].abs().median():.2f}m")

    df = df.join(cross_df)
    df.reset_index().to_csv(ROOT / "data/sentinel2_benchmark/srtm_3way_comparison.csv", index=False)
    print(f"\nWrote {ROOT / 'data/sentinel2_benchmark/srtm_3way_comparison.csv'}")


if __name__ == "__main__":
    main()
