#!/usr/bin/env python3
"""CBERS-4 PAN10M vs Sentinel-2 at 10 m, same Brazilian locations (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, "CBERS-4 vs Sentinel-2 at 10 m ...").

  select  : per category, walk data/brazil_benchmark/candidates.csv in order; keep the first 8 candidates that pass
            the CBERS / FABDEM / Sentinel-2 / ICESat-2 / co-registration gates. Resumable (tiles/<id>.npz + status.json).
  build   : three caches (cbers_fc, s2_fc, s2_rgb) in the s2_token_grid_phase_c cache format (rgb uint8, fab30, fab10,
            ICESat-2 cells in EGM2008) -- identical except for rgb.
  run ARM : the unchanged rank-loss recipe + raw-Spearman on one arm (as scripts/landsat_rank_test.py).
  compare : the pre-registered paired comparison (CBERS-FC vs S2-FC) plus descriptive context.
Network: STAC (stac.scitekno.com.br), S3 COGs (brazil-eosats), Earth Engine, SlideRule, PROJ geoid grids.
"""
from __future__ import annotations

import os

os.environ["PROJ_NETWORK"] = "ON"
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
BB = ROOT / "data/brazil_benchmark"
STAC = "https://stac.scitekno.com.br/v100/search"
S3 = "https://brazil-eosats.s3.amazonaws.com/"
PER_CAT, MAX_READS, MIN_PHOTONS, SHIFT = 8, 12, 2000, 20
T08 = Transformer.from_crs("EPSG:4979", "EPSG:3855", always_xy=True)


# ============================================================================ helpers
def stac(body):
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for k in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=90))["features"]
        except Exception:  # noqa: BLE001
            if k == 3:
                raise
            time.sleep(5 * (k + 1))


def tile_grid(lat, lon):
    zone = int((lon + 180) // 6) + 1
    epsg = (32700 if lat < 0 else 32600) + zone
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    x0, y0 = round((x - 5000) / 10) * 10, round((y + 5000) / 10) * 10
    return f"EPSG:{epsg}", from_origin(x0, y0, 10, 10)


def ee_pixels(img, crs, T, h, w):
    import ee
    arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                 "grid": {"dimensions": {"width": w, "height": h},
                                          "affineTransform": {"scaleX": T.a, "shearX": T.b, "translateX": T.c,
                                                              "shearY": T.d, "scaleY": T.e, "translateY": T.f},
                                          "crsCode": crs}})
    return np.stack([np.asarray(arr[n], dtype=np.float32) for n in arr.dtype.names])


def fabdem_image(ee):
    """FABDEM mosaic, bilinear at its NATIVE ~30 m grid. A bare mosaic() has no native projection, so resample() on it
    interpolates on EE's default 1-degree grid (bug found 2026-09-24: smooth ~200 m ramps instead of the DEM)."""
    col = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    return col.mosaic().setDefaultProjection(col.first().projection()).resample("bilinear")


def refab(args):
    """Re-fetch FABDEM (fab30, fab10) for every passed tile with fabdem_image(); verify against FABDEM point-sampled by
    EE at 300 ICESat-2 photon locations (no gridding involved). Writes data/brazil_benchmark/fab_repair.json."""
    import ee
    proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
    ee.Initialize(project=proj)
    fab = fabdem_image(ee)
    raw = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic()
    rp = BB / "fab_repair.json"
    rep = json.loads(rp.read_text()) if rp.exists() else {}
    status = json.loads((BB / "status.json").read_text())
    for cid, v in status.items():
        if not v["pass"] or cid in rep:
            continue
        p = BB / "tiles" / f"{cid}.npz"
        z = dict(np.load(p))
        crs, T = str(z["crs"]), Affine(*z["T"])
        f30 = ee_pixels(fab, crs, T * Affine.scale(3), 333, 333)[0]
        f10 = ee_pixels(fab, crs, T, 1000, 1000)[0]
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(z["ph_lon"], z["ph_lat"])
        cc, rr = ~T * (x, y)
        row, col = np.floor(rr).astype(int), np.floor(cc).astype(int)
        ok = np.where((row >= 0) & (row < 1000) & (col >= 0) & (col < 1000))[0]
        idx = np.random.default_rng(0).choice(ok, min(300, len(ok)), replace=False)
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(z["ph_lon"][i]), float(z["ph_lat"][i])]), {"i": int(i)})
                                   for i in idx])
        feats = raw.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()["features"]
        band = [k for k in feats[0]["properties"] if k != "i"][0]
        pv = np.array([ft["properties"][band] for ft in feats]); pi = np.array([ft["properties"]["i"] for ft in feats])
        r_grid_vs_point = float(np.corrcoef(f10[row[pi], col[pi]], pv)[0, 1])
        med_abs = float(np.median(np.abs(f10[row[pi], col[pi]] - pv)))
        r_vs_icesat = float(np.corrcoef(f10[row[ok], col[ok]], z["ph_h"][ok])[0, 1])
        good = bool(np.isfinite(f10).all() and np.isfinite(f30).all() and med_abs < 5)
        rep[cid] = {"old_range": [float(z["fab10"].min()), float(z["fab10"].max())],
                    "new_range": [float(f10.min()), float(f10.max())], "grid_vs_point_r": r_grid_vs_point,
                    "grid_vs_point_median_abs_m": med_abs, "grid_vs_icesat_r": r_vs_icesat, "ok": good}
        if good:
            z["fab30"], z["fab10"] = f30, f10
            np.savez_compressed(p, **z)
        rp.write_text(json.dumps(rep, indent=1))
        print(f"{cid}: {'repaired' if good else 'NOT OK'}  range {rep[cid]['old_range'][0]:.0f}-{rep[cid]['old_range'][1]:.0f} -> "
              f"{f10.min():.0f}-{f10.max():.0f} m; grid-vs-point r {r_grid_vs_point:.3f} (median |d| {med_abs:.2f} m); "
              f"vs ICESat-2 r {r_vs_icesat:.3f}", flush=True)


def screen_cbers(item, lat, lon):
    h = item["assets"]["B2"]["href"].replace("s3://brazil-eosats/", S3)
    with rasterio.open("/vsicurl/" + h) as s:
        b = transform_bounds("EPSG:4326", s.crs, lon - .05, lat - .045, lon + .05, lat + .045)
        w = rasterio.windows.from_bounds(*b, s.transform)
        a = s.read(1, window=w, out_shape=(100, 100), boundless=True, fill_value=0)
    return dict(valid=float((a > 0).mean()), sat=float((a >= 250).mean()), bright=float((a >= 200).mean()))


def warp_cbers(item, crs, T):
    out = np.full((3, 1000, 1000), np.nan, np.float32)
    for i, band in enumerate(("B4", "B3", "B2")):  # NIR, red, green
        with rasterio.open("/vsicurl/" + item["assets"][band]["href"].replace("s3://brazil-eosats/", S3)) as s:
            b = transform_bounds(crs, s.crs, *rasterio.transform.array_bounds(1000, 1000, T))
            w = rasterio.windows.from_bounds(*b, s.transform).round_offsets().round_lengths()
            w = rasterio.windows.Window(w.col_off - 20, w.row_off - 20, w.width + 40, w.height + 40)
            a = s.read(1, window=w, boundless=True, fill_value=0).astype(np.float32)
            reproject(source=a, destination=out[i], src_transform=s.window_transform(w), src_crs=s.crs, src_nodata=0,
                      dst_transform=T, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


def xcorr_shift(a, b, m=SHIFT):
    """Integer shift (dy, dx) maximising normalised correlation of a[y, x] with b[y+dy, x+dx] over the central area."""
    c = slice(100, 900)
    A = a[c, c]; A = (A - np.nanmean(A)) / np.nanstd(A)
    best = (-9.0, 0, 0)
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            Bv = b[100 + dy:900 + dy, 100 + dx:900 + dx]
            Bn = (Bv - np.nanmean(Bv)) / np.nanstd(Bv)
            r = float(np.nanmean(A * Bn))
            if r > best[0]:
                best = (r, dy, dx)
    return best


def icesat2_photons(lat, lon, crs, T):
    from query_icesat2_photons import query_tile
    x0, y0 = T.c, T.f
    to = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    poly = [dict(zip(("lon", "lat"), to.transform(x, y))) for x, y in
            ((x0, y0 - 10000), (x0 + 10000, y0 - 10000), (x0 + 10000, y0), (x0, y0), (x0, y0 - 10000))]
    return query_tile(poly)


# ============================================================================ select
def select(args):
    import ee
    from sliderule import sliderule
    proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
    ee.Initialize(project=proj)
    sliderule.init("slideruleearth.io", verbose=False)
    (BB / "tiles").mkdir(parents=True, exist_ok=True)
    stp = BB / "status.json"
    status = json.loads(stp.read_text()) if stp.exists() else {}
    cand = pd.read_csv(BB / "candidates.csv")
    fab = fabdem_image(ee)
    for cat, grp in cand.groupby("category", sort=False):
        for _, c in grp.iterrows():
            if sum(1 for v in status.values() if v.get("category") == cat and v.get("pass")) >= PER_CAT:
                break
            cid = c.candidate_id
            if cid in status:
                continue
            rec = {"category": cat, "pass": False}
            t0 = time.time()
            try:
                crs, T = tile_grid(c.lat, c.lon)
                # gate 1: CBERS L4 scene
                items = [f for f in stac({"collections": ["CBERS4-PAN10M"], "bbox": [c.lon - .05, c.lat - .045, c.lon + .05, c.lat + .045],
                                          "datetime": "2019-01-01T00:00:00Z/2025-12-31T23:59:59Z", "limit": 300})
                         if f["id"].endswith("_L4")]
                items.sort(key=lambda f: f["properties"]["datetime"], reverse=True)
                scene = None
                for f in items[:MAX_READS]:
                    q = screen_cbers(f, c.lat, c.lon)
                    if q["valid"] >= 0.99 and q["sat"] < 0.005 and q["bright"] < 0.02:
                        scene = f; rec["cbers_screen"] = q; break
                rec["cbers_l4_items"] = len(items)
                if scene is None:
                    rec["fail"] = "cbers"; raise StopIteration
                rec["cbers_id"] = scene["id"]; date = scene["properties"]["datetime"][:10]; rec["cbers_date"] = date
                # gate 4 (cheap): FABDEM
                fab30 = ee_pixels(fab, crs, T * Affine.scale(3), 333, 333)[0]
                fab10 = ee_pixels(fab, crs, T, 1000, 1000)[0]
                if not (np.isfinite(fab30).all() and np.isfinite(fab10).all() and (fab30 > -1000).all()):
                    rec["fail"] = "fabdem"; raise StopIteration
                # gate 2: Sentinel-2 median
                d = ee.Date(date); region = ee.Geometry.Point([c.lon, c.lat])
                s2 = None
                for days in (90, 365):
                    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
                           .filterDate(d.advance(-days, "day"), d.advance(days, "day"))
                           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)))
                    n = col.size().getInfo()
                    if n >= 2:
                        s2 = ee_pixels(col.select(["B8", "B4", "B3", "B2"]).median(), crs, T, 1000, 1000)
                        rec["s2_scenes"], rec["s2_window_days"] = n, days
                        break
                if s2 is None or not np.isfinite(s2).all():
                    rec["fail"] = "s2"; raise StopIteration
                # gate 3: ICESat-2
                ph = icesat2_photons(c.lat, c.lon, crs, T)
                rec["photons"] = len(ph)
                if len(ph) < MIN_PHOTONS:
                    rec["fail"] = "icesat2"; raise StopIteration
                # CBERS full read + co-registration on NIR vs S2 B8
                cb = warp_cbers(scene, crs, T)
                if not np.isfinite(cb).all():
                    rec["fail"] = "cbers_coverage"; raise StopIteration
                r0, dy, dx = xcorr_shift(cb[0], s2[0])
                if (dy, dx) != (0, 0):
                    # content at CBERS (y, x) matches S2 (y+dy, x+dx): shift CBERS by re-warping onto a translated grid
                    cb = warp_cbers(scene, crs, T * Affine.translation(-dx, -dy))
                r1, ry, rx = xcorr_shift(cb[0], s2[0], m=3)
                rec["coreg"] = {"initial_shift": [dy, dx], "initial_r": r0, "residual_shift": [ry, rx], "residual_r": r1}
                if abs(ry) > 1 or abs(rx) > 1 or r1 < 0.3 or not np.isfinite(cb).all():
                    rec["fail"] = "coreg"; raise StopIteration
                np.savez_compressed(BB / "tiles" / f"{cid}.npz", cbers=cb, s2=s2, fab30=fab30, fab10=fab10,
                                    ph_lat=ph["lat"].values, ph_lon=ph["lon"].values, ph_h=ph["height"].values,
                                    ph_rgt=ph["rgt"].values, crs=np.array(crs), T=np.array(tuple(T)[:6]))
                rec["pass"] = True
            except StopIteration:
                pass
            except Exception as e:  # noqa: BLE001
                rec["fail"] = f"error: {type(e).__name__}: {str(e)[:200]}"
            rec["seconds"] = round(time.time() - t0, 1)
            status[cid] = rec
            stp.write_text(json.dumps(status, indent=1))
            print(f"{cat:12s} {cid:22s} {'PASS' if rec['pass'] else 'fail: ' + rec.get('fail', '?')}  "
                  f"({rec['seconds']}s) {rec.get('cbers_id', '')} photons={rec.get('photons', '-')} coreg={rec.get('coreg', {}).get('initial_shift', '-')}",
                  flush=True)
    print("passing per category:", {k: sum(1 for v in status.values() if v["category"] == k and v["pass"])
                                    for k in cand.category.unique()})


# ============================================================================ build
def stretch(b):
    lo, hi = np.percentile(b, [2, 98])
    return np.clip((b - lo) / max(hi - lo, 1e-6) * 255, 0, 255).round().astype(np.uint8)


def build(args):
    status = json.loads((BB / "status.json").read_text())
    arms = {"cbers_fc": lambda z: z["cbers"][[0, 1, 2]], "s2_fc": lambda z: z["s2"][[0, 1, 2]],
            "s2_rgb": lambda z: z["s2"][[1, 2, 3]]}
    for a in arms:
        (BB / a / "cache").mkdir(parents=True, exist_ok=True)
    for cid, v in status.items():
        if not v["pass"]:
            continue
        z = np.load(BB / "tiles" / f"{cid}.npz")
        crs, T = str(z["crs"]), Affine(*z["T"])
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(z["ph_lon"], z["ph_lat"])
        cols, rows = ~T * (x, y)
        ph = pd.DataFrame({"row": np.floor(rows).astype(int), "col": np.floor(cols).astype(int),
                           "rgt": z["ph_rgt"], "height": z["ph_h"]})
        ph = ph[(ph.row >= 0) & (ph.row < 1000) & (ph.col >= 0) & (ph.col < 1000)]
        g = ph.groupby(["row", "col", "rgt"])["height"].median().reset_index()
        xc, yc = T * (g["col"].values + 0.5, g["row"].values + 0.5)
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xc, yc)
        ortho = T08.transform(lon, lat, g["height"].values)[2]
        assert np.all(np.abs(g["height"].values - ortho) > 1), f"{cid}: geoid N~0 (PROJ grid not used)"
        for a, f in arms.items():
            rgb = np.stack([stretch(b) for b in f(z)])
            np.savez_compressed(BB / a / "cache" / f"{cid}.npz", rgb=rgb, fab30=z["fab30"], fab10=z["fab10"],
                                ph_row=g["row"].values, ph_col=g["col"].values, ph_h=ortho.astype(np.float64))
        print(f"{cid}: {len(g)} ICESat-2 cells", flush=True)


# ============================================================================ run (unchanged recipe per arm)
def run(args):
    import torch
    from PIL import Image
    import s2_rank_loss_test as R
    import s2_rank_raw_spearman as RS
    import s2_token_grid_phase_c as C
    from evaluate_method6_finetune_twinhead import get_device
    from backend.depth.depth_engine import run_inference
    A = BB / args.arm
    C.CACHE, R.OUT, R.DAV2 = A / "cache", A / "rank_loss", A / "dav2_depth"
    R.OUT.mkdir(parents=True, exist_ok=True); R.DAV2.mkdir(parents=True, exist_ok=True)
    tids = sorted(p.stem for p in C.CACHE.glob("*.npz"))
    for t in tids:
        out = R.DAV2 / f"{t}_depth.npy"
        if not out.exists():
            np.save(out, run_inference(Image.fromarray(np.load(C.CACHE / f"{t}.npz")["rgb"].transpose(1, 2, 0), "RGB")))
    device = get_device()
    allrows = []
    for fold in range(4):
        t0 = time.time()
        model, hs, run_ = R.fit(fold, tids, 600, 4, device)
        recs = [{"tile": t, "model": R.calibrated_eval(R.scores_all(model, t, device), t, fold),
                 "oracle": R.calibrated_eval(R.oracle_scores(t), t, fold)} for t in tids]
        (R.OUT / f"eval_rank_fold{fold}.json").write_text(json.dumps(
            {"fold": fold, "steps": 600, "batch": 4, "height_scale": hs, "final_loss100": float(np.mean(run_[-100:])),
             "train_s": time.time() - t0, "records": recs}))
        rows = RS.cells_for_fold(model, tids, fold, device)
        for r in rows:
            r["fold"] = fold
        allrows += rows
        print(f"[{args.arm}] fold{fold} done {time.time()-t0:.0f}s, {len(rows)} crops with ICESat-2 cells", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    R.analyze(None)
    (R.OUT / "raw_spearman").mkdir(exist_ok=True)
    RS.summarize(allrows, tids, None, R.OUT / "raw_spearman")


# ============================================================================ compare (pre-registered)
def compare(args):
    from scipy.stats import wilcoxon

    def load(a):
        r = json.loads((BB / a / "rank_loss/raw_spearman/summary.json").read_text())
        c = json.loads((BB / a / "rank_loss/summary.json").read_text())
        return r, c

    def boot(v, B=10000):
        rng = np.random.default_rng(0)
        m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    arms = {a: load(a) for a in ("cbers_fc", "s2_fc", "s2_rgb")}
    tiles = sorted(arms["cbers_fc"][0]["per_tile"])

    def paired(a, b, key, raw=True, hib=True):
        if raw:
            x = np.array([arms[a][0]["per_tile"][t][key] for t in tiles], float)
            y = np.array([arms[b][0]["per_tile"][t][key] for t in tiles], float)
        else:
            x = np.array([arms[a][1]["checks"]["icesat2"]["per_tile"][t][key] for t in tiles], float)
            y = np.array([arms[b][1]["checks"]["icesat2"]["per_tile"][t][key] for t in tiles], float)
        d = x - y
        return {f"{a}_mean": float(x.mean()), f"{b}_mean": float(y.mean()), "diff_mean": float(d.mean()), "ci95": boot(d),
                f"{a}_better": int(((d > 0) if hib else (d < 0)).sum()), "n": len(d), "wilcoxon_p": float(wilcoxon(x, y).pvalue)}

    S = {"n_tiles": len(tiles),
         "primary_raw_within_crop_model": paired("cbers_fc", "s2_fc", "within_crop_model"),
         "raw_within_crop_oracle": paired("cbers_fc", "s2_fc", "within_crop_oracle"),
         "calibrated_icesat_rmse_model": paired("cbers_fc", "s2_fc", "model", raw=False, hib=False),
         "descriptive_s2fc_vs_s2rgb_raw_model": paired("s2_fc", "s2_rgb", "within_crop_model"),
         "per_arm": {a: {"raw_within_crop_model": v[0]["means"]["within_crop_model"],
                         "raw_within_crop_oracle": v[0]["means"]["within_crop_oracle"],
                         "raw_within_crop_fab": v[0]["means"]["within_crop_fab"],
                         "calib_implied_r": v[0]["means"]["calib_implied_r"],
                         "calibrated_icesat_rmse": {k: v[1]["checks"]["icesat2"][k] for k in
                                                    ("model_mean_rmse", "oracle_mean_rmse", "model_wins", "wilcoxon_p_holm",
                                                     "flat_mean_rmse", "fabdem_mean_rmse")},
                         "verdict_real_signal": v[1]["verdict_real_signal"], "case_raw": v[0]["case"]}
                     for a, v in arms.items()},
         "india_reference": {"s2_rgb_raw_within_crop_model": 0.134, "landsat_pan_raw_within_crop_model": 0.116}}
    p, c = S["primary_raw_within_crop_model"], S["calibrated_icesat_rmse_model"]
    surprise = p["diff_mean"] >= 0.10 and p["cbers_fc_better"] > p["n"] / 2 and p["wilcoxon_p"] < 0.05
    raw_dir = np.sign(p["diff_mean"]) if p["wilcoxon_p"] < 0.05 else 0
    cal_dir = -np.sign(c["diff_mean"]) if c["wilcoxon_p"] < 0.05 else 0
    S["divergence_raw_vs_calibrated"] = bool(raw_dir != 0 and cal_dir != 0 and raw_dir != cal_dir)
    S["decision"] = ("SURPRISE: CBERS meaningfully higher than Sentinel-2 at the same locations -> investigate before anything else"
                     if surprise else "CONFIRMATION: CBERS not meaningfully higher -> third independent confirmation; close the line")
    if S["divergence_raw_vs_calibrated"]:
        S["decision"] += " | FLAG: raw and calibrated disagree in direction -> revisit calibration first"
    (BB / "comparison.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "refab", "build", "run", "compare"])
    ap.add_argument("--arm", choices=["cbers_fc", "s2_fc", "s2_rgb"])
    a = ap.parse_args()
    {"select": select, "refab": refab, "build": build, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
