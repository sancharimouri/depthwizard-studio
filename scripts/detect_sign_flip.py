"""Sign-flip/rejection detector for DAv2's relative-depth output, usable
without ICESat-2 (real deployment tiles won't have it).

Computes two independent, offline-available disagreement signals per tile:
  (a) DAv2 vs. Copernicus GLO-30 DEM (real elevation, reprojected onto the
      tile's own 10m UTM grid) -- coarse (30m native) but real, catches
      flat-terrain false-gradient cases a backbone-vs-backbone check can't.
  (b) DAv2 vs. DINOv3/CHMv2 (both already on the same 1000x1000 grid, no
      reprojection needed) -- flagged per-tile as "informative" or not
      based on tree-cover% (CHMv2 is a canopy-height head; on a flat,
      treeless tile its output is close to constant and this check
      shouldn't be trusted).

Validates against each tile's ALREADY-KNOWN DAv2-vs-ICESat2 correlation
sign (data/sentinel2_benchmark/dav2_depth/dav2_correlation_per_tile.csv)
as ground truth -- that ground-truth column is used ONLY for validation
here, never as an input to the detector itself.

Writes the raw per-tile signal table to
data/sentinel2_benchmark/sign_flip_detector_signals.csv for the threshold
search in evaluate_sign_flip_detector.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/sentinel2_benchmark/manifest.csv"
DAV2_DIR = ROOT / "data/sentinel2_benchmark/dav2_depth"
DINOV3_DIR = ROOT / "data/sentinel2_benchmark/dinov3_depth"
DEM_DIR = ROOT / "data/sentinel2_benchmark/copernicus_dem_raw"
GT_CSV = ROOT / "data/sentinel2_benchmark/dav2_depth/dav2_correlation_per_tile.csv"
TREE_CSV = ROOT / "data/sentinel2_benchmark/content_audit_corrected_32.csv"
OUT_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_signals.csv"

DEM_NODATA = -32767.0


def reproject_dem_to_tile_grid(dem_path: Path, rgb_path: Path) -> np.ndarray:
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs
    with rasterio.open(rgb_path) as rgb_src:
        dst_crs = rgb_src.crs
        dst_transform = rgb_src.transform
        dst_shape = (rgb_src.height, rgb_src.width)

    dst = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=dem,
        destination=dst,
        src_transform=dem_transform,
        src_crs=dem_crs,
        src_nodata=DEM_NODATA,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst


def main():
    manifest = pd.read_csv(MANIFEST)
    gt = pd.read_csv(GT_CSV).set_index("tile_id")
    tree = pd.read_csv(TREE_CSV).set_index("tile_id")

    rows = []
    for _, row in manifest.iterrows():
        tile_id = row["tile_id"]
        rgb_path = ROOT / row["rgb_path"]
        dem_path = DEM_DIR / f"{tile_id}_dem.tif"

        dav2 = np.load(DAV2_DIR / f"{tile_id}_depth.npy")
        dinov3 = np.load(DINOV3_DIR / f"{tile_id}_depth.npy")
        dem_on_grid = reproject_dem_to_tile_grid(dem_path, rgb_path)

        valid = ~np.isnan(dem_on_grid)
        n_valid = int(valid.sum())
        pearson_a, _ = stats.pearsonr(dav2[valid], dem_on_grid[valid])

        pearson_b, _ = stats.pearsonr(dav2.ravel(), dinov3.ravel())
        dinov3_std = float(dinov3.std())
        dinov3_cv = float(dinov3.std() / dinov3.mean()) if dinov3.mean() != 0 else float("nan")

        tree_pct = float(tree.loc[tile_id, "tree_pct"]) if tile_id in tree.index else float("nan")

        true_pearson = float(gt.loc[tile_id, "pearson"])
        true_inverted = true_pearson < 0

        rec = {
            "tile_id": tile_id,
            "category": row["category"],
            "dav2_vs_dem_pearson": pearson_a,
            "n_valid_dem_px": n_valid,
            "dem_coverage_pct": 100 * n_valid / dem_on_grid.size,
            "dav2_vs_dinov3_pearson": pearson_b,
            "dinov3_std": dinov3_std,
            "dinov3_cv": dinov3_cv,
            "tree_pct": tree_pct,
            "true_dav2_icesat2_pearson": true_pearson,
            "true_inverted": true_inverted,
        }
        rows.append(rec)
        print(
            f"{tile_id:15s} a(dem)={pearson_a:+.4f}  b(dinov3)={pearson_b:+.4f}  "
            f"dinov3_std={dinov3_std:.4f}  tree%={tree_pct:5.1f}  "
            f"dem_cov={rec['dem_coverage_pct']:5.1f}%  "
            f"true={true_pearson:+.4f}{'  INVERTED' if true_inverted else ''}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    return df


if __name__ == "__main__":
    main()
