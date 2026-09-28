#!/usr/bin/env python3
"""Curated DFC2019 terrain packs (08-dfc2019-terrain-packs, Prompt 5).

  build    for each of the 50 tiles: reference correction of the held-out Method 6 heights
           (rule pre-registered in docs/method-audit/08-dfc2019-terrain-packs/packs.md, cap and
           margin from Prompt 4's heights/cap.json), + base ground (Prompt 3), written as a pack in
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
FEATHER_SIGMA = 2.0         # px at 0.3 m
NEG_FLOOR = -5.0            # AGL in [-5, 0) = ground noise -> 0; below -5 = invalid (none in this data)
SURFACE_SOURCE = "Model-predicted heights (Method 6), with reference lidar correction for tall structures and trees"
CRS_LABEL = "local frame (no georeference)"


def read_agl(tid):
    import rasterio
    with rasterio.open(AGL_DIR / f"{tid}_AGL.tif") as r:
        return r.read(1).astype(np.float32)


def correct(pred, agl, cap, margin):
    """The pre-registered rule. Returns corrected heights and the (feathered) replacement weight."""
    from scipy.ndimage import gaussian_filter
    valid = np.isfinite(agl) & (agl >= NEG_FLOOR) & (agl < 1000)
    a = np.where(valid, np.clip(agl, 0, None), 0.0)
    tall = valid & ((a > cap) | (pred > cap))
    rep = (tall & (np.abs(pred - a) > margin)).astype(np.float32)
    w = np.maximum(rep, np.clip(gaussian_filter(rep, FEATHER_SIGMA) * 2, 0, 1))   # 1 inside, soft ring outside
    w[~valid] = 0.0
    out = pred * (1 - w) + a * w
    return out.astype(np.float32), rep, w, valid


def block_to(a, n=N):
    from backend.terrain.mesh_export import block_mean
    return block_mean(a.astype(np.float64), (n, n)).astype(np.float32)


def cmd_build(a):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    cap_info = json.loads((SCR / "heights/cap.json").read_text())
    cap, margin = float(cap_info["cap_m"]), float(cap_info["margin_m"])
    locs = json.loads((SCR / "locate/locations.json").read_text())["tiles"]
    PACKS.mkdir(parents=True, exist_ok=True)
    # GSD: located tiles use their own matched GSD; city-level tiles the city's median matched GSD
    city_gsd = {c: float(np.median([v["gsd"] for v in locs.values() if v["city"] == c])) if any(
        v["city"] == c for v in locs.values()) else 0.3 for c in ("JAX", "OMA")}
    report = {}
    tiles = sorted(p.name[:-8] for p in AGL_DIR.glob("*_AGL.tif"))
    for tid in tiles:
        pred = np.load(SCR / f"heights/{tid}.npz")["agl"]
        agl = read_agl(tid)
        valid_frac = float((np.isfinite(agl) & (agl >= NEG_FLOOR) & (agl < 1000)).mean())
        if valid_frac < 0.5:
            corr, rep, w = pred, np.zeros_like(pred), np.zeros_like(pred)
            corrected = False
        else:
            corr, rep, w, _ = correct(pred, agl, cap, margin)
            corrected = True
        b = np.load(SCR / f"base/{tid}.npz")
        base = b["base"].astype(np.float32)
        info = json.loads(str(b["info"]))
        assert base.shape == (N, N), base.shape
        surface = base + block_to(corr)
        gsd = locs[tid]["gsd"] if tid in locs else city_gsd[tid[:3]]
        size = 1024 * gsd
        tf = from_bounds(0, 0, size, size, N, N)
        located = info["base_source"] == "dem-located"
        terrain_src = ("Public DEM: USGS 3DEP bare earth" if located
                       else "Approximate city-level ground elevation (USGS 3DEP city median)")
        note = ("Ground level from a public DEM; " if located else
                "Elevations are approximate: ground level is one city-level value, not measured at this tile; ") + \
               "heights are model-predicted, with reference lidar correction for tall structures and trees."
        path = PACKS / f"dfc2019-{tid}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=N, height=N, count=2, dtype="float32",
                           crs=CRS.from_proj4(LOCAL_CRS), transform=tf, nodata=np.nan,
                           compress="deflate", predictor=3, zlevel=9) as wr:
            wr.write(base, 1)
            wr.write(surface.astype(np.float32), 2)
            wr.set_band_description(1, "TERRAIN")
            wr.set_band_description(2, "SURFACE")
            wr.update_tags(TERRAIN_SOURCE=terrain_src, SURFACE_SOURCE=SURFACE_SOURCE, CRS_LABEL=CRS_LABEL,
                           RAMP_BAND="SURFACE", NOTE=note, HOW="curated elevation pack")
        report[tid] = {"base_source": info["base_source"], "gsd_m": round(gsd, 4), "corrected": corrected,
                       "frac_replaced": round(float(rep.mean()), 4), "frac_touched_feathered": round(float((w > 0).mean()), 4),
                       "valid_agl_frac": round(valid_frac, 6), "surface_range_m": [round(float(surface.min()), 1), round(float(surface.max()), 1)],
                       "pred_max_m": round(float(pred.max()), 1), "corrected_max_m": round(float(corr.max()), 1)}
        print(tid, report[tid], flush=True)
    (PACKS / "build_report.json").write_text(json.dumps({"cap_m": cap, "margin_m": margin, "city_gsd_m": city_gsd,
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
