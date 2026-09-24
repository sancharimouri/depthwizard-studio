#!/usr/bin/env python3
"""Terrain-relief positive control for the rank-loss / raw-Spearman methodology (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, "Terrain-relief positive control").

The DFC2019 control validated the machinery on 18 m crops of object height. This one tests terrain relief on
Sentinel-2's own measurement scale (600 m crops, 30 m target cells) with VHR input (NAIP, 1.0 m), using Part D's
USGS 3DEP + NAIP infrastructure (scripts/forest_mountain_3dep_eval.py: STAC search, SAS signing, site boxes,
forest criteria, leaf-on NAIP rule).

Geometry: the pipeline's crop is defined in pixels, so a 600 m crop at 1.0 m needs the constants rescaled. They are
patched on the imported modules, exactly as landsat_rank_test.py patches C.CACHE; no function body changes:
  N 60 -> 600 px (600 m), K 3 -> 30 (1 m -> 30 m target cells, 20 x 20 per crop as on Sentinel-2),
  P_PAD 70 -> 602 (arm P = true pixel count: 43 x 43 tokens, 14 m / token),
  crop_origins: 6 x 6 crops per quadrant, quadrants start at 0 / 3600 px (tile 7200 px = 7.2 km; a 3DEP item is
  8.2 km, too small for S2's 8 x 8 layout).
Everything else is unchanged: s2_rank_loss_test fit / calibrated_eval / scores_all / oracle_scores / analyze and
s2_rank_raw_spearman cells_for_fold / summarize (rank_pair_loss, 2000 pairs, margin 0.25, 600 x 4 steps, seed 42,
AdamW 5e-6 / 2.5e-4, 4 quadrant folds, per-tile slope calibration, MIN_CELLS 10).

Truth = 3DEP LiDAR DTM (3dep-lidar-dtm, 2 m, NAVD88), bilinear onto the tile's 1 m grid, in every slot:
  fab30 = 30 m block mean (target / calibration), ph cells = the DTM at the centre pixel of every 30 m block
  (20 x 20 = 400 per crop), fab10 = DTM (float16; only the trivial reference slot and the finite mask use it).
Oracle raw signal: frozen DAv2-Large (backend.depth.depth_engine.run_inference) on the whole tile. 7200^2 exceeds its
50 Mpx input budget, so it gets the tile 2x area-averaged (3600^2; it is resized to 518 internally either way) and the
depth is repeated 2x back to 7200.

  PROJ_NETWORK=ON .venv/bin/python scripts/terrain_relief_positive_control.py select|fetch|run
Output: data/terrain_relief_control/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "ON")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import forest_mountain_3dep_eval as F  # noqa: E402
import s2_rank_loss_test as R  # noqa: E402
import s2_rank_raw_spearman as RS  # noqa: E402
import s2_token_grid_phase_c as C  # noqa: E402
from evaluate_method6_finetune_twinhead import get_device  # noqa: E402

OUT = ROOT / "data/terrain_relief_control"
GSD = 1.0
N, K, P_PAD = 600, 30, 602
PER_Q, Q0 = 6, 3600
TILE = 2 * Q0               # 7200 px = 7.2 km
PER_SITE, MAX_TILES = 3, 15
DTM_VALID_MIN, RGB_VALID_MIN = 0.98, 0.99


def crop_origins(q: int, size: int = TILE):
    r0 = 0 if q in (0, 1) else Q0
    c0 = 0 if q in (0, 2) else Q0
    return [(r0 + N * i, c0 + N * j) for i in range(PER_Q) for j in range(PER_Q)]


def patch_geometry():
    C.N, C.K, C.P_PAD, C.crop_origins = N, K, P_PAD, crop_origins
    R.N, R.K = N, K
    RS.N = N


def retry(fn, tries=5):
    for a in range(tries):
        try:
            return fn()
        except (rasterio.errors.RasterioError, OSError):
            if a == tries - 1:
                raise
            time.sleep(20 * (a + 1))


# ----------------------------------------------------------------------------- select
def stat(f, k):
    b = f["properties"]["raster:bands"]
    b = b if isinstance(b, dict) else b[0]
    return float(b["statistics"][k])


def select(args):
    """Part D's per-site forest criteria (HAG valid >= 95%, HAG mean >= 8 m, relaxed to >= 5 m), all eligible items
    ranked by DTM relief (STAC DTM stddev, descending); up to PER_SITE tiles per site, MAX_TILES in total. A tile is
    the centred 7.2 km square of the item; kept if DTM >= 98% valid and leaf-on NAIP covers >= 99% (checked in fetch)."""
    sel = {"sites": {}, "tiles": []}
    for site, (lat, lon) in {**F.SITES, **F.BACKUPS}.items():
        box = [lon - 0.15, lat - 0.15, lon + 0.15, lat + 0.15]
        feats = F.post({"collections": ["3dep-lidar-hag"], "bbox": box, "limit": 500})
        rec = {"n_candidates": len(feats)}
        elig = []
        for thr in (8.0, 5.0):
            elig = [f["id"] for f in feats if stat(f, "valid_percent") >= 95 and stat(f, "mean") >= thr]
            rec[f"eligible_mean_ge_{thr:g}"] = len(elig)
            if elig:
                rec["mean_threshold"] = thr
                break
        ranked = []
        for iid in elig:
            try:
                d = F.item("3dep-lidar-dtm", iid.replace("-hag-", "-dtm-"))
            except Exception:  # noqa: BLE001
                continue
            ranked.append((-stat(d, "stddev"), d["id"], stat(d, "stddev"), d["properties"]["end_datetime"]))
        ranked.sort()
        rec["ranked"] = [{"dtm_item": i, "dtm_stddev": s, "lidar_end": e} for _, i, s, e in ranked]
        sel["sites"][site] = rec
        print(site, {k: v for k, v in rec.items() if k != "ranked"}, [round(r[2]) for r in ranked][:6], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))


# ----------------------------------------------------------------------------- fetch
def tile_grid(dtm_item):
    it = F.item("3dep-lidar-dtm", dtm_item)
    href = F.signed("3dep-lidar-dtm", it["assets"]["data"]["href"])
    with rasterio.open(href) as s:
        from pyproj import CRS
        pc = CRS.from_wkt(s.crs.to_wkt())
        epsg = (pc.sub_crs_list[0] if pc.is_compound else pc).to_epsg()
        W = s.width * s.res[0]
        x0 = round(s.transform.c + (W - TILE * GSD) / 2)
        y0 = round(s.transform.f - (W - TILE * GSD) / 2)
    return href, epsg, from_origin(x0, y0, GSD, GSD), it


def warp(href, epsg, T, bands, resampling, nodata=None):
    def go():
        with rasterio.open(href) as s:
            kw = dict(crs=f"EPSG:{epsg}", transform=T, width=TILE, height=TILE, resampling=resampling)
            if nodata is not None:
                kw.update(src_nodata=nodata, nodata=nodata)
            with WarpedVRT(s, **kw) as v:
                return v.read(bands), v.dataset_mask()
    return retry(go)


def fetch(args):
    from pyproj import Transformer
    sel = json.loads((OUT / "selection.json").read_text())
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    tiles = sel.get("tiles", [])
    done = {t["tile"] for t in tiles}
    for site, rec in sel["sites"].items():
        n_site = sum(t["site"] == site for t in tiles)
        for cand in rec["ranked"]:
            if n_site >= PER_SITE or len(tiles) >= MAX_TILES:
                break
            tid = f"{site}__{cand['dtm_item']}"
            if tid in done or any(x.get("tile") == tid for x in sel.get("rejected", [])):
                continue
            href, epsg, T, it = tile_grid(cand["dtm_item"])
            (dtm,), m = warp(href, epsg, T, [1], Resampling.bilinear)
            dtm = dtm.astype(np.float32)
            dtm[(m == 0) | ~np.isfinite(dtm) | (dtm < -500)] = np.nan
            dv = float(np.isfinite(dtm).mean())
            info = {"tile": tid, "site": site, "dtm_item": cand["dtm_item"], "epsg": epsg, "x0": T.c, "y0": T.f,
                    "lidar_end": cand["lidar_end"], "dtm_valid": dv}
            if dv < DTM_VALID_MIN:
                sel.setdefault("rejected", []).append({**info, "reason": "dtm_valid"}); continue
            # NAIP: leaf-on (Jun-Sep), gsd <= 1.0, the year closest to the LiDAR year; ties -> finer; mosaic that year
            tl = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            xs, ys = [T.c, T.c + TILE], [T.f - TILE, T.f]
            lo, la = tl.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
            nf = F.post({"collections": ["naip"], "bbox": [min(lo), min(la), max(lo), max(la)], "limit": 500})
            cand_n = [f for f in nf if f["properties"]["gsd"] <= 1.0 and 6 <= int(f["properties"]["datetime"][5:7]) <= 9]
            ly = int(cand["lidar_end"][:4])
            years = sorted({int(f["properties"]["datetime"][:4]) for f in cand_n},
                           key=lambda y: (abs(y - ly), min(f["properties"]["gsd"] for f in cand_n
                                                           if int(f["properties"]["datetime"][:4]) == y)))
            ok = False
            for y in years:
                fs = [f for f in cand_n if int(f["properties"]["datetime"][:4]) == y]
                rgb = np.zeros((3, TILE, TILE), np.uint8); valid = np.zeros((TILE, TILE), bool)
                for f in sorted(fs, key=lambda f: f["id"]):
                    h = F.signed("naip", F.item("naip", f["id"])["assets"]["image"]["href"])
                    rs = Resampling.average if f["properties"]["gsd"] < GSD else Resampling.bilinear
                    a, mk = warp(h, epsg, T, [1, 2, 3], rs, nodata=0)
                    new = (mk > 0) & ~valid
                    rgb[:, new] = a[:, new]; valid |= new
                rv = float(valid.mean())
                if rv >= RGB_VALID_MIN:
                    dates = sorted({f["properties"]["datetime"][:10] for f in fs})
                    info.update({"naip_year": y, "naip_dates": dates, "naip_items": sorted(f["id"] for f in fs),
                                 "naip_gsd": sorted({f["properties"]["gsd"] for f in fs}), "rgb_valid": rv,
                                 "lidar_naip_gap_years": abs(y - ly)})
                    ok = True
                    break
                print(f"{tid}: NAIP {y} covers {rv:.3f}, next year", flush=True)
            if not ok:
                sel.setdefault("rejected", []).append({**info, "reason": "naip_coverage"}); continue
            h3 = TILE // K
            fab30 = np.nanmean(dtm.reshape(h3, K, h3, K), axis=(1, 3)).astype(np.float32)
            rr, cc = np.meshgrid(np.arange(h3) * K + K // 2, np.arange(h3) * K + K // 2, indexing="ij")
            okc = np.isfinite(dtm[rr, cc])
            np.savez(OUT / "cache" / f"{tid}.npz", rgb=rgb, fab30=fab30, fab10=dtm.astype(np.float16),
                     ph_row=rr[okc], ph_col=cc[okc], ph_h=dtm[rr, cc][okc].astype(np.float64))
            info["relief_within_crop_std_m"] = float(np.nanmedian(
                [np.nanstd(fab30[r // K:(r + N) // K, c // K:(c + N) // K]) for q in range(4) for r, c in crop_origins(q)]))
            tiles.append(info); n_site += 1
            sel["tiles"] = tiles
            (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
            print(f"{tid}: ok, NAIP {info['naip_year']} {info['naip_gsd']}, gap {info['lidar_naip_gap_years']} y, "
                  f"median crop relief std {info['relief_within_crop_std_m']:.1f} m", flush=True)
    print(f"{len(tiles)} tiles", flush=True)


# ----------------------------------------------------------------------------- run
def oracle_depth(tids):
    from PIL import Image
    from backend.depth.depth_engine import run_inference
    (OUT / "dav2_depth").mkdir(parents=True, exist_ok=True)
    for t in tids:
        o = OUT / "dav2_depth" / f"{t}_depth.npy"
        if o.exists():
            continue
        rgb = np.load(OUT / "cache" / f"{t}.npz")["rgb"].astype(np.float32)
        half = rgb.reshape(3, TILE // 2, 2, TILE // 2, 2).mean((2, 4)).round().astype(np.uint8).transpose(1, 2, 0)
        d = run_inference(Image.fromarray(half, "RGB"))
        np.save(o, np.repeat(np.repeat(d, 2, 0), 2, 1).astype(np.float16))
        print(f"dav2-L {t}", flush=True)


def run(args):
    patch_geometry()
    C.CACHE, R.OUT, R.DAV2 = OUT / "cache", OUT / "rank_loss", OUT / "dav2_depth"
    R.OUT.mkdir(parents=True, exist_ok=True)
    tids = sorted(t["tile"] for t in json.loads((OUT / "selection.json").read_text())["tiles"])
    assert len(tids) == 8, len(tids)  # user-directed scope amendment (was 10-15), see the doc
    oracle_depth(tids)
    device = get_device()
    allrows = []
    for fold in args.folds:
        t0 = time.time()
        model, hs, run_ = R.fit(fold, tids, args.steps, 4, device)
        recs = []
        for tid in tids:
            m = R.calibrated_eval(R.scores_all(model, tid, device), tid, fold)
            o = R.calibrated_eval(R.oracle_scores(tid), tid, fold)
            recs.append({"tile": tid, "model": m, "oracle": o})
        (R.OUT / f"eval_rank_fold{fold}.json").write_text(json.dumps(
            {"fold": fold, "steps": args.steps, "batch": 4, "height_scale": hs,
             "final_loss100": float(np.mean(run_[-100:])), "train_s": time.time() - t0, "records": recs}))
        rows = RS.cells_for_fold(model, tids, fold, device)
        for r in rows:
            r["fold"] = fold
        allrows += rows
        print(f"fold{fold}: done {time.time()-t0:.0f}s, {len(rows)} held-out crops with cells", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    if args.smoke:
        return
    R.analyze(None)
    (R.OUT / "raw_spearman").mkdir(exist_ok=True)
    RS.summarize(allrows, tids, None, R.OUT / "raw_spearman")


def parity(args):
    """Local half of the Kaggle parity check: the ORIGINAL imported pipeline on the CPU, one tile, fold 0, 5 steps."""
    patch_geometry()
    C.CACHE, R.DAV2 = OUT / "cache", OUT / "dav2_depth"
    tid = args.tiles[0]
    oracle_depth([tid])
    dev = torch.device("cpu")
    model, hs, run_ = R.fit(0, [tid], args.steps, 4, dev)
    sm, so = R.scores_all(model, tid, dev), R.oracle_scores(tid)
    m, o = R.calibrated_eval(sm, tid, 0), R.calibrated_eval(so, tid, 0)
    rows = RS.cells_for_fold(model, [tid], 0, dev)
    from scipy.stats import spearmanr
    rho = [None if len(r["h"]) < RS.MIN_CELLS or np.ptp(r["h"]) == 0 or np.ptp(r["model"]) == 0
           else float(spearmanr(r["model"], r["h"]).statistic) for r in rows]
    (OUT / "parity_local.json").write_text(json.dumps({"height_scale": hs, "run": run_, "model": m, "oracle": o,
                                                        "crop_rho_model": rho, "n_rows": len(rows)}, indent=1))
    print("parity_local written", flush=True)


def bundle(args):
    """Kaggle bundle: script, tiles.json, cache (no full-grid reference slot), oracle at 3600^2, DAv2-Small snapshot."""
    import shutil
    tids = sorted(t["tile"] for t in json.loads((OUT / "selection.json").read_text())["tiles"])
    oracle_depth(tids)
    B = OUT / "kaggle_bundle" / "bundle"
    for d in ("cache", "oracle", "dav2_small"):
        (B / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "scripts/terrain_relief_kaggle.py", B)
    (B / "tiles.json").write_text(json.dumps(tids))
    for t in tids:
        z = np.load(OUT / "cache" / f"{t}.npz")
        np.savez(B / "cache" / f"{t}.npz", rgb=z["rgb"], fab30=z["fab30"], ph_row=z["ph_row"], ph_col=z["ph_col"],
                 ph_h=z["ph_h"], ph_fab=z["fab10"][z["ph_row"], z["ph_col"]])
        d = np.load(OUT / "dav2_depth" / f"{t}_depth.npy")
        np.save(B / "oracle" / f"{t}_depth_half.npy", np.ascontiguousarray(d[::2, ::2]))
    snap = next((ROOT / "models/hub/models--depth-anything--Depth-Anything-V2-Small-hf/snapshots").iterdir())
    for f in ("config.json", "model.safetensors"):
        dst = B / "dav2_small" / f
        if not dst.exists():
            os.link((snap / f).resolve(), dst)
    print(f"bundle: {len(tids)} tiles -> {B}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "fetch", "run", "parity", "bundle"])
    ap.add_argument("--tiles", nargs="*")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"select": select, "fetch": fetch, "run": run, "parity": parity, "bundle": bundle}[a.cmd](a)


if __name__ == "__main__":
    main()
