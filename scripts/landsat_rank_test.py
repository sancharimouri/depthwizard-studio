#!/usr/bin/env python3
"""Landsat 8/9 panchromatic (15 m) rank-loss test (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, "Landsat sensor-identity test").

Isolates sensor identity: the identical rank-loss recipe (scripts/s2_rank_loss_test.py) on the SAME 32 benchmark
footprints, with the Sentinel-2 RGB replaced by real Landsat OLI band 8 (pan, 15 m).
  - Imagery: Earth Engine LANDSAT/LC08 + LC09 C02/T1_TOA, band B8, median of scenes with CLOUD_COVER < 20 over 2025
    (the Sentinel-2 benchmark's acquisition year; widened to 2024-2025 / < 40 only if a tile has < 3 such scenes),
    fetched at native 15 m on a grid aligned to each Sentinel-2 tile, then bilinearly warped onto that tile's exact
    10 m grid (adds no information; keeps crops, token geometry, FABDEM target cells and ICESat-2 cells identical).
    8-bit per-tile 2-98 percentile stretch; replicated to 3 channels (DAv2 expects RGB).
  - Model / loss / folds / calibration / oracle: imported unchanged from s2_rank_loss_test (fit, calibrated_eval,
    scores_all, oracle_scores, analyze); the oracle's raw signal is frozen DAv2-Large run on the Landsat tile exactly as
    the Sentinel-2 oracle was made (backend.depth.depth_engine.run_inference, whole tile at 518).
  - Raw Spearman: scripts/s2_rank_raw_spearman.py (cells_for_fold, summarize) on the SAME trained models (no re-fit).

  PROJ_NETWORK=ON .venv/bin/python scripts/landsat_rank_test.py fetch
  .venv/bin/python scripts/landsat_rank_test.py run
  .venv/bin/python scripts/landsat_rank_test.py compare
Output: data/landsat_benchmark/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s2_rank_loss_test as R  # noqa: E402
import s2_rank_raw_spearman as RS  # noqa: E402
import s2_token_grid_phase_c as C  # noqa: E402
from evaluate_method6_finetune_twinhead import get_device  # noqa: E402

BENCH = ROOT / "data/sentinel2_benchmark"
LB = ROOT / "data/landsat_benchmark"
S2_CACHE = BENCH / "token_grid_test/cache"


def fetch(args):
    import ee
    proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
    ee.Initialize(project=proj)
    man = pd.read_csv(BENCH / "manifest.csv").set_index("tile_id")
    (LB / "pan15").mkdir(parents=True, exist_ok=True); (LB / "cache").mkdir(parents=True, exist_ok=True)
    meta = []
    col = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA"))
    for tid in man.index:
        with rasterio.open(ROOT / man.loc[tid, "rgb_path"]) as s:
            T10, crs, H, W = s.transform, s.crs, s.height, s.width
        T15 = T10 * Affine.scale(1.5)
        h15, w15 = int(np.ceil(H / 1.5)), int(np.ceil(W / 1.5))
        left, bottom, right, top = rasterio.transform.array_bounds(H, W, T10)
        region = ee.Geometry.Rectangle([left, bottom, right, top], proj=crs.to_string(), evenOdd=False)
        window = "2025"
        c = col.filterBounds(region).filterDate("2025-01-01", "2026-01-01").filter(ee.Filter.lt("CLOUD_COVER", 20))
        n = c.size().getInfo()
        if n < 3:
            window = "2024-2025, cloud<40"
            c = col.filterBounds(region).filterDate("2024-01-01", "2026-01-01").filter(ee.Filter.lt("CLOUD_COVER", 40))
            n = c.size().getInfo()
        img = c.select("B8").median()
        arr = ee.data.computePixels({"expression": img.unmask(-1), "fileFormat": "NUMPY_NDARRAY",
                                     "grid": {"dimensions": {"width": w15, "height": h15},
                                              "affineTransform": {"scaleX": T15.a, "shearX": T15.b, "translateX": T15.c,
                                                                  "shearY": T15.d, "scaleY": T15.e, "translateY": T15.f},
                                              "crsCode": crs.to_string()}})
        pan15 = np.asarray(arr[arr.dtype.names[0]] if arr.dtype.names else arr, dtype=np.float32)
        pan15[pan15 < 0] = np.nan
        np.save(LB / "pan15" / f"{tid}_B8.npy", pan15)
        pan10 = np.full((H, W), np.nan, np.float32)
        reproject(source=pan15, destination=pan10, src_transform=T15, src_crs=crs, dst_transform=T10, dst_crs=crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        lo, hi = np.nanpercentile(pan10, [2, 98])
        u8 = np.clip(np.nan_to_num((pan10 - lo) / max(hi - lo, 1e-6), nan=0.0) * 255, 0, 255).round().astype(np.uint8)
        z = dict(np.load(S2_CACHE / f"{tid}.npz"))
        z["rgb"] = np.repeat(u8[None], 3, axis=0)
        np.savez_compressed(LB / "cache" / f"{tid}.npz", **z)
        meta.append({"tile": tid, "scenes": n, "window": window, "finite_frac": float(np.isfinite(pan10).mean()),
                     "p2": float(lo), "p98": float(hi)})
        print(f"{tid}: {n} scenes ({window}), finite {meta[-1]['finite_frac']:.3f}", flush=True)
    pd.DataFrame(meta).to_csv(LB / "landsat_fetch_meta.csv", index=False)


def oracle_depth():
    from PIL import Image
    from backend.depth.depth_engine import run_inference
    (LB / "dav2_depth").mkdir(parents=True, exist_ok=True)
    for p in sorted((LB / "cache").glob("*.npz")):
        out = LB / "dav2_depth" / f"{p.stem}_depth.npy"
        if out.exists():
            continue
        rgb = np.load(p)["rgb"].transpose(1, 2, 0)
        np.save(out, run_inference(Image.fromarray(rgb, "RGB")))
        print(f"dav2-L {p.stem}", flush=True)


def run(args):
    oracle_depth()
    C.CACHE, R.OUT, R.DAV2 = LB / "cache", LB / "rank_loss", LB / "dav2_depth"
    R.OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    tids = sorted(p.stem for p in C.CACHE.glob("*.npz"))
    assert len(tids) == 32
    allrows = []
    for fold in range(4):
        t0 = time.time()
        model, hs, run_ = R.fit(fold, tids, 600, 4, device)
        recs = []
        for tid in tids:
            m = R.calibrated_eval(R.scores_all(model, tid, device), tid, fold)
            o = R.calibrated_eval(R.oracle_scores(tid), tid, fold)
            recs.append({"tile": tid, "model": m, "oracle": o})
        (R.OUT / f"eval_rank_fold{fold}.json").write_text(json.dumps(
            {"fold": fold, "steps": 600, "batch": 4, "height_scale": hs, "final_loss100": float(np.mean(run_[-100:])),
             "train_s": time.time() - t0, "records": recs}))
        rows = RS.cells_for_fold(model, tids, fold, device)
        for r in rows:
            r["fold"] = fold
        allrows += rows
        print(f"fold{fold}: done {time.time()-t0:.0f}s, {len(rows)} held-out crops with ICESat-2 cells", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    R.analyze(None)
    (R.OUT / "raw_spearman").mkdir(exist_ok=True)
    RS.summarize(allrows, tids, None, R.OUT / "raw_spearman")


def compare(args):
    """Pre-registered Landsat-vs-Sentinel-2 comparison (paired by tile)."""
    from scipy.stats import wilcoxon
    s2r = json.loads((BENCH / "token_grid_test/rank_loss/raw_spearman/summary.json").read_text())["per_tile"]
    s2c = json.loads((BENCH / "token_grid_test/rank_loss/summary.json").read_text())["checks"]["icesat2"]["per_tile"]
    lr = json.loads((LB / "rank_loss/raw_spearman/summary.json").read_text())["per_tile"]
    lc = json.loads((LB / "rank_loss/summary.json").read_text())["checks"]["icesat2"]["per_tile"]
    tiles = sorted(set(s2r) & set(lr))

    def boot(v, B=10000):
        rng = np.random.default_rng(0)
        m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    def paired(a, b, higher_is_better=True):
        a, b = np.array(a, float), np.array(b, float); d = a - b
        return {"landsat_mean": float(a.mean()), "s2_mean": float(b.mean()), "diff_mean": float(d.mean()),
                "ci95": boot(d), "landsat_better": int(((d > 0) if higher_is_better else (d < 0)).sum()), "n": len(d),
                "wilcoxon_p": float(wilcoxon(a, b).pvalue)}

    S = {"raw_within_crop_model": paired([lr[t]["within_crop_model"] for t in tiles], [s2r[t]["within_crop_model"] for t in tiles]),
         "raw_within_crop_oracle": paired([lr[t]["within_crop_oracle"] for t in tiles], [s2r[t]["within_crop_oracle"] for t in tiles]),
         "calibrated_icesat_rmse_model": paired([lc[t]["model"] for t in tiles], [s2c[t]["model"] for t in tiles], False)}
    r, c = S["raw_within_crop_model"], S["calibrated_icesat_rmse_model"]
    higher = r["diff_mean"] >= 0.10 and r["landsat_better"] > r["n"] / 2 and r["wilcoxon_p"] < 0.05
    raw_dir = np.sign(r["diff_mean"]) if r["wilcoxon_p"] < 0.05 else 0          # +1 Landsat ranks better
    cal_dir = -np.sign(c["diff_mean"]) if c["wilcoxon_p"] < 0.05 else 0         # +1 Landsat calibrated better
    S["divergence_raw_vs_calibrated"] = bool(raw_dir != 0 and cal_dir != 0 and raw_dir != cal_dir)
    S["case"] = ("A: Landsat raw ranking meaningfully higher -> sensor-specific; investigate the Sentinel-2 pipeline"
                 if higher else
                 "B: Landsat raw ranking not meaningfully higher -> no sensor-specific evidence; resolution wall looks general")
    if S["divergence_raw_vs_calibrated"]:
        S["case"] += " | FLAG: raw and calibrated point in opposite directions -> revisit calibration before trusting A/B"
    (LB / "comparison_vs_s2.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "compare"])
    a = ap.parse_args()
    {"fetch": fetch, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
