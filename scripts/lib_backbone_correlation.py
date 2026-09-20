"""Shared photon-sampling + correlation procedure for the frozen-backbone
comparison (DAv2 vs. DINOv3). Used identically by both backbones so results
are directly comparable, per the task's methodology requirement.

For each tile: reproject every ICESat-2 ground-photon lat/lon into the
source GeoTIFF's pixel grid (same grid the depth backbone ran on, since
neither backbone resamples geometry), sample the backbone's relative-depth
raster at that pixel, and correlate against the photon's ellipsoidal ground
height (Pearson + Spearman).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHOTON_DIR = PROJECT_ROOT / "data" / "icesat2_photons"
MANIFEST = PROJECT_ROOT / "data" / "sentinel2_benchmark" / "manifest.csv"


def sample_depth_at_photons(rgb_path: Path, depth: np.ndarray, photons: pd.DataFrame) -> pd.DataFrame:
    """Return photons with an added `depth_value` column, sampled at each
    photon's pixel location in the depth raster (same grid as rgb_path)."""
    with rasterio.open(rgb_path) as src:
        crs = src.crs
        transform = src.transform
        height_px, width_px = src.height, src.width

    to_tile_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_tile_crs.transform(photons["lon"].values, photons["lat"].values)

    inv = ~transform
    cols, rows = inv * (x, y)
    cols = np.floor(cols).astype(int)
    rows = np.floor(rows).astype(int)

    in_bounds = (cols >= 0) & (cols < width_px) & (rows >= 0) & (rows < height_px)

    depth_values = np.full(len(photons), np.nan, dtype=np.float32)
    depth_values[in_bounds] = depth[rows[in_bounds], cols[in_bounds]]

    out = photons.copy()
    out["depth_value"] = depth_values
    return out.dropna(subset=["depth_value"])


def correlate_tile(sampled: pd.DataFrame) -> dict:
    if len(sampled) < 3:
        return {"n": len(sampled), "pearson": float("nan"), "spearman": float("nan")}

    pearson_r, _ = stats.pearsonr(sampled["depth_value"], sampled["height"])
    spearman_r, _ = stats.spearmanr(sampled["depth_value"], sampled["height"])
    return {"n": len(sampled), "pearson": float(pearson_r), "spearman": float(spearman_r)}


def run_backbone_correlation(depth_dir: Path, filename_fn) -> pd.DataFrame:
    """depth_dir: directory holding one {tile_id}_depth.npy per tile.
    filename_fn: tile_id -> Path to that tile's .npy depth file."""
    manifest = pd.read_csv(MANIFEST)
    rows = []

    for _, row in manifest.iterrows():
        tile_id = row["tile_id"]
        photon_path = PHOTON_DIR / f"{tile_id}.csv"
        depth_path = filename_fn(tile_id)

        if not photon_path.exists():
            print(f"  {tile_id}: SKIP (no photon CSV yet)")
            continue
        if not depth_path.exists():
            print(f"  {tile_id}: SKIP (no depth output at {depth_path})")
            continue

        photons = pd.read_csv(photon_path)
        depth = np.load(depth_path)
        rgb_path = PROJECT_ROOT / row["rgb_path"]

        sampled = sample_depth_at_photons(rgb_path, depth, photons)
        result = correlate_tile(sampled)
        result.update({"tile_id": tile_id, "category": row["category"]})
        rows.append(result)

        print(
            f"  {tile_id:15s} {row['category']:14s} n={result['n']:7d} "
            f"pearson={result['pearson']:+.4f} spearman={result['spearman']:+.4f}"
        )

    return pd.DataFrame(rows)


def pooled_correlation(depth_dir: Path, filename_fn) -> dict:
    """Pool ALL photons across all tiles, then compute one correlation --
    distinct from averaging per-tile correlations."""
    manifest = pd.read_csv(MANIFEST)
    all_sampled = []

    for _, row in manifest.iterrows():
        tile_id = row["tile_id"]
        photon_path = PHOTON_DIR / f"{tile_id}.csv"
        depth_path = filename_fn(tile_id)
        if not photon_path.exists() or not depth_path.exists():
            continue

        photons = pd.read_csv(photon_path)
        depth = np.load(depth_path)
        rgb_path = PROJECT_ROOT / row["rgb_path"]
        sampled = sample_depth_at_photons(rgb_path, depth, photons)
        sampled["category"] = row["category"]
        all_sampled.append(sampled)

    pooled = pd.concat(all_sampled, ignore_index=True)
    result = correlate_tile(pooled)

    by_category = {}
    for cat, group in pooled.groupby("category"):
        by_category[cat] = correlate_tile(group)

    return {"pooled": result, "by_category": by_category}
