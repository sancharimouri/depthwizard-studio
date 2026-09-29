#!/usr/bin/env python3
"""Curated DFC2019 terrain packs (08-dfc2019-terrain-packs, Prompt 5).

  build    for each of the 50 tiles: base ground (Prompt 3) + above-ground height taken directly from
           the DFC2019 lidar AGL truth (2026-09-29 rebuild; no Method 6 / DAv2 input at all; fallback
           rule below), written as a pack in
           the exact desktop/tiles/build_dem_pack.py format (2-band TERRAIN/SURFACE float32 GeoTIFF),
           at the app's mesh grid (341 x 341), in a synthetic local metric frame at (0, 0) so no
           real location ever ships. -> data/dfc2019/terrain_packs/packs/dfc2019-<tile>.tif
  upload   the packs -> PRIVATE HF dataset (library_store.HF_DATASET) under dem/ (never public).

No error or accuracy number is written into any pack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCR = ROOT / "data/dfc2019/terrain_packs"
PACKS = SCR / "packs"
AGL_DIR = ROOT / "data/dfc2019/raw/Truth/Track1-Truth"
N = 341                     # _mesh_hw(1024 px) = 341: the pack is never finer than the mesh it feeds
LOCAL_CRS = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84"
NEG_FLOOR = -5.0            # AGL in [-5, 0) = ground noise -> 0; below -5 = invalid (none in this data)
# Fallback rule (2026-09-29), per pixel: AGL valid (finite, -5 <= AGL < 1000) -> height = max(AGL, 0), i.e.
# small negative lidar ground noise is ground level; AGL invalid -> height 0 (ground level). Per tile: more than
# LOW_COVERAGE of pixels invalid -> CONFIDENCE=reduced tag + report flag (the heights still follow the pixel rule).
LOW_COVERAGE = 0.05
SURFACE_SOURCE = "Reference lidar above-ground heights (IEEE GRSS DFC2019 AGL)"
CRS_LABEL = "local frame (no georeference)"


def read_agl(tid):
    import rasterio
    with rasterio.open(AGL_DIR / f"{tid}_AGL.tif") as r:
        return r.read(1).astype(np.float32)


def agl_heights(agl):
    """The 2026-09-29 rule: lidar AGL is the height wherever valid; invalid pixels fall back to ground level (0)."""
    valid = np.isfinite(agl) & (agl >= NEG_FLOOR) & (agl < 1000)
    return np.where(valid, np.clip(agl, 0, None), 0.0).astype(np.float32), valid


def block_to(a, n=N):
    from backend.terrain.mesh_export import block_mean
    return block_mean(a.astype(np.float64), (n, n)).astype(np.float32)


def cmd_build(a):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    locs = json.loads((SCR / "locate/locations.json").read_text())["tiles"]
    PACKS.mkdir(parents=True, exist_ok=True)
    # GSD: located tiles use their own matched GSD; city-level tiles the city's median matched GSD
    city_gsd = {c: float(np.median([v["gsd"] for v in locs.values() if v["city"] == c])) if any(
        v["city"] == c for v in locs.values()) else 0.3 for c in ("JAX", "OMA")}
    report = {}
    tiles = sorted(p.name[:-8] for p in AGL_DIR.glob("*_AGL.tif"))
    for tid in tiles:
        agl = read_agl(tid)
        heights, valid = agl_heights(agl)
        valid_frac = float(valid.mean())
        reduced = (1 - valid_frac) > LOW_COVERAGE
        b = np.load(SCR / f"base/{tid}.npz")
        base = b["base"].astype(np.float32)
        info = json.loads(str(b["info"]))
        assert base.shape == (N, N), base.shape
        surface = base + block_to(heights)
        gsd = locs[tid]["gsd"] if tid in locs else city_gsd[tid[:3]]
        size = 1024 * gsd
        tf = from_bounds(0, 0, size, size, N, N)
        located = info["base_source"] == "dem-located"
        terrain_src = ("Public DEM: USGS 3DEP bare earth" if located
                       else "Approximate city-level ground elevation (USGS 3DEP city median)")
        note = ("Ground level from a public DEM; " if located else
                "Elevations are approximate: ground level is one city-level value, not measured at this tile; ") + \
               "heights are reference lidar above-ground heights."
        path = PACKS / f"dfc2019-{tid}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=N, height=N, count=2, dtype="float32",
                           crs=CRS.from_proj4(LOCAL_CRS), transform=tf, nodata=np.nan,
                           compress="deflate", predictor=3, zlevel=9) as wr:
            wr.write(base, 1)
            wr.write(surface.astype(np.float32), 2)
            wr.set_band_description(1, "TERRAIN")
            wr.set_band_description(2, "SURFACE")
            wr.update_tags(TERRAIN_SOURCE=terrain_src, SURFACE_SOURCE=SURFACE_SOURCE, CRS_LABEL=CRS_LABEL,
                           RAMP_BAND="SURFACE", NOTE=note, HOW="curated elevation pack",
                           CONFIDENCE="reduced" if reduced else "normal")
        report[tid] = {"base_source": info["base_source"], "gsd_m": round(gsd, 4), "valid_agl_frac": round(valid_frac, 6),
                       "fallback_pixels_frac": round(1 - valid_frac, 6), "reduced_confidence": reduced,
                       "neg_agl_clamped_frac": round(float((valid & (agl < 0)).mean()), 4),
                       "surface_range_m": [round(float(surface.min()), 1), round(float(surface.max()), 1)],
                       "height_max_m": round(float(heights.max()), 1)}
        print(tid, report[tid], flush=True)
    (PACKS / "build_report.json").write_text(json.dumps({"heights": "lidar AGL primary, no Method 6", "neg_floor_m": NEG_FLOOR,
                                                         "low_coverage_frac": LOW_COVERAGE, "city_gsd_m": city_gsd,
                                                         "tiles": report}, indent=1))


def cmd_upload(a):
    from huggingface_hub import HfApi
    from backend.storage.hf_checkpoints import token
    from backend.storage.library_store import HF_DATASET
    api = HfApi(token=token())
    assert api.dataset_info(HF_DATASET).private, "refusing: the target dataset is not private"
    files = sorted(PACKS.glob("dfc2019-*.tif"))
    assert len(files) == 50, len(files)
    api.upload_folder(repo_id=HF_DATASET, repo_type="dataset", folder_path=str(PACKS), path_in_repo="dem",
                      allow_patterns=["dfc2019-*.tif"],
                      commit_message="DFC2019 curated terrain packs (08-dfc2019-terrain-packs), private")
    listed = [f for f in api.list_repo_files(HF_DATASET, repo_type="dataset") if f.startswith("dem/")]
    print(f"uploaded; dem/ now holds {len(listed)} files in private {HF_DATASET}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("upload")
    a = ap.parse_args()
    {"build": cmd_build, "upload": cmd_upload}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
