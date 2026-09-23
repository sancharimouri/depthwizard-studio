"""Step 3/4 of the DAv2 calibration-hardening pass: re-run the sign-flip
detector's signal (a) and the calibration methodology against FABDEM
instead of Copernicus GLO-30, to test whether a bare-earth DEM (FABDEM
removes GLO-30's building/tree height) changes (a) which tiles pass the
rejection gate at all, and (b) calibration quality among accepted tiles.

Mirrors detect_sign_flip.py / evaluate_sign_flip_detector.py /
calibrate_accepted_tiles.py's exact methods (same thresholds: a<0.0
rejection, informative>=5% tree_pct gate with b<-0.3; same held-out
discipline: 50/50 split seed 42, then independent ICESat-2 cross-check)
but pointed at data/sentinel2_benchmark/fabdem_raw/ instead of
copernicus_dem_raw/. Signal (b) (DAv2 vs DINOv3) is DEM-independent, so
it's reused unchanged from sign_flip_detector_signals.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import stats

from lib_backbone_correlation import sample_depth_at_photons

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
FABDEM_DIR = ROOT / "data/sentinel2_benchmark/fabdem_raw"
PHOTON_DIR = ROOT / "data/icesat2_photons"
SIGNALS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_signals.csv"  # for signal (b), tree_pct, GLO-30 numbers
STEP1_CSV = ROOT / "data/sentinel2_benchmark/step1_offset_proxy.csv"  # offset_proxy, GLO-30 icesat2_rmse_pct_of_range
SEED = 42
THRESH_A = 0.0
THRESH_B = -0.3
INFORMATIVE_TREE_PCT = 5.0


def reproject_fabdem_to_tile_grid(fabdem_path: Path, rgb_path: Path) -> np.ndarray:
    with rasterio.open(fabdem_path) as src:
        fabdem = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata  # FABDEM's GEE export has no explicit nodata; None is fine for reproject
    with rasterio.open(rgb_path) as rgb_src:
        dst_crs = rgb_src.crs
        dst_transform = rgb_src.transform
        dst_shape = (rgb_src.height, rgb_src.width)

    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=fabdem, destination=dst,
        src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def main():
    manifest = pd.read_csv(MANIFEST).set_index("tile_id")
    signals = pd.read_csv(SIGNALS_CSV).set_index("tile_id")  # has dav2_vs_dinov3_pearson, tree_pct
    step1 = pd.read_csv(STEP1_CSV).set_index("tile_id")  # has offset_proxy, GLO-30 icesat2_rmse_pct_of_range

    rows = []
    for tile_id, row in manifest.iterrows():
        rgb_path = ROOT / row["rgb_path"]
        dav2 = np.load(DAV2_DIR / f"{tile_id}_depth.npy")
        fabdem_on_grid = reproject_fabdem_to_tile_grid(FABDEM_DIR / f"{tile_id}_fabdem.tif", rgb_path)

        valid = ~np.isnan(fabdem_on_grid)
        fabdem_pearson, _ = stats.pearsonr(dav2[valid], fabdem_on_grid[valid])

        dinov3_pearson = float(signals.loc[tile_id, "dav2_vs_dinov3_pearson"])
        tree_pct = float(signals.loc[tile_id, "tree_pct"])
        informative = tree_pct >= INFORMATIVE_TREE_PCT

        flagged = (fabdem_pearson < THRESH_A) or (informative and dinov3_pearson < THRESH_B)

        rec = {
            "tile_id": tile_id, "category": row["category"],
            "dav2_vs_fabdem_pearson": fabdem_pearson,
            "dav2_vs_dinov3_pearson": dinov3_pearson, "tree_pct": tree_pct,
            "fabdem_flagged": flagged,
        }

        if not flagged:
            idx = np.flatnonzero(valid)
            rng = np.random.RandomState(SEED)
            rng.shuffle(idx)
            half = len(idx) // 2
            train_idx, test_idx = idx[:half], idx[half:]

            depth_flat = dav2.ravel()
            fabdem_flat = fabdem_on_grid.ravel()
            lr = stats.linregress(depth_flat[train_idx], fabdem_flat[train_idx])
            a, b = lr.slope, lr.intercept

            pred_test = a * depth_flat[test_idx] + b
            true_test = fabdem_flat[test_idx]
            held_out_pearson, _ = stats.pearsonr(pred_test, true_test)

            photon_path = PHOTON_DIR / f"{tile_id}.csv"
            photons = pd.read_csv(photon_path)
            sampled = sample_depth_at_photons(rgb_path, dav2, photons)
            pred_icesat2 = a * sampled["depth_value"].values + b
            true_icesat2 = sampled["height"].values
            icesat2_pearson, _ = stats.pearsonr(pred_icesat2, true_icesat2)
            icesat2_rmse = rmse(pred_icesat2, true_icesat2)

            fabdem_range = float(fabdem_on_grid[valid].max() - fabdem_on_grid[valid].min())
            rec.update({
                "held_out_fabdem_pearson": held_out_pearson,
                "icesat2_pearson_fabdem": icesat2_pearson,
                "icesat2_rmse_m_fabdem": icesat2_rmse,
                "fabdem_elev_range_m": fabdem_range,
                "icesat2_rmse_pct_of_range_fabdem": 100 * icesat2_rmse / fabdem_range,
            })
        else:
            rec.update({
                "held_out_fabdem_pearson": float("nan"),
                "icesat2_pearson_fabdem": float("nan"),
                "icesat2_rmse_m_fabdem": float("nan"),
                "fabdem_elev_range_m": float("nan"),
                "icesat2_rmse_pct_of_range_fabdem": float("nan"),
            })

        rows.append(rec)
        status = "REJECTED" if flagged else "accepted"
        extra = f"held_out_pearson={rec['held_out_fabdem_pearson']:+.4f} icesat2_rmse%={rec['icesat2_rmse_pct_of_range_fabdem']:.1f}" if not flagged else ""
        print(f"{tile_id:15s} fabdem_a={fabdem_pearson:+.4f}  {status:9s} {extra}")

    df = pd.DataFrame(rows).set_index("tile_id")

    # Tier classification: reuse the FIXED threshold Step 2 already established from
    # GLO-30 (0.114 -- the midpoint of the largest gap in GLO-30's held_out_dem_pearson
    # distribution), rather than re-fitting a fresh gap on FABDEM's own distribution.
    # A per-DEM-source threshold lets the boundary drift for reasons unrelated to any
    # individual tile's reliability -- see the 2026-09-21 root-cause section in
    # sign-flip-detector.md for why this matters (it initially produced a spurious
    # tier disagreement for vidisha/kochi_city that had nothing to do with FABDEM
    # actually being less reliable for those two tiles specifically).
    TIER_THRESHOLD = 0.114

    def fabdem_tier(tile_id):
        if df.loc[tile_id, "fabdem_flagged"]:
            return "REJECTED"
        return "LOW-CONFIDENCE" if df.loc[tile_id, "held_out_fabdem_pearson"] <= TIER_THRESHOLD else "CONFIDENT"

    df["fabdem_tier"] = [fabdem_tier(t) for t in df.index]

    # Join with GLO-30 side for the final consolidated table.
    glo30_tier = pd.read_csv(ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv").set_index("tile_id")
    df["glo30_tier"] = glo30_tier["tier"]
    df["glo30_icesat2_rmse_pct_of_range"] = step1["icesat2_rmse_pct_of_range"].reindex(df.index)
    # for GLO-30-rejected tiles, step1 has no row (they weren't calibrated); fill from nowhere -- leave NaN
    df["offset_proxy"] = step1["offset_proxy"].reindex(df.index)

    print("\n=== Consolidated 32-tile comparison ===")
    cols = ["category", "glo30_tier", "glo30_icesat2_rmse_pct_of_range", "fabdem_tier", "icesat2_rmse_pct_of_range_fabdem", "offset_proxy"]
    print(df[cols].to_string())

    out_path = ROOT / "data/sentinel2_benchmark/fabdem_comparison.csv"
    df.reset_index().to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    return df


if __name__ == "__main__":
    main()
