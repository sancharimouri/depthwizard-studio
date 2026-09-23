#!/usr/bin/env python3
"""Part D (2026-09-23): Method 6 on forested/mountainous US terrain vs. airborne LiDAR.
Pre-registration: docs/method-audit/07-gamus-generalization/log.md ("Part D pre-registration").
NEON's data API needs a token (403), so the reference is USGS 3DEP LiDAR (Planetary Computer
`3dep-lidar-dsm` / `3dep-lidar-hag`, 2 m, NAD83 UTM + NAVD88) and the RGB is leaf-on NAIP.

Stages (each writes JSON under data/forest_mountain_3dep/):
  select   : items + windows + NAIP per the pre-registered rules -> selection.json
  run      : per window, Method 6 (12-model mean, margin-192 tiling) at 0.3 m (primary) and 0.6 m
             (secondary); AGL -> 2 m (area average); DEMs via Earth Engine on the 2 m grid, converted
             per pixel to NAVD88; metrics -> windows.json (+ small GeoTIFFs per window)
  stats    : Wilcoxon/Holm decision for (b) -> summary.json
  PROJ_NETWORK=ON .venv/bin/python scripts/forest_mountain_3dep_eval.py select|run|stats
"""
from __future__ import annotations

import json
import os
import random
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "ON")  # before any pyproj import (HANDOFF §7)
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/forest_mountain_3dep"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SITES = {"olympic_wa": (47.95, -123.90), "tahoe_ca": (39.10, -120.10),
         "frontrange_co": (40.35, -105.60), "grsm_tn": (35.689, -83.502)}
BACKUPS = {"mlbs_va": (37.378, -80.525), "whitemtns_nh": (44.10, -71.40)}
WIN_M, REF_RES = 600, 2.0
HAG_MAX, HAG_MIN = 100.0, -2.0


def post(body):
    req = urllib.request.Request(f"{STAC}/search", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["features"]


_tok = {}


def signed(col, href):
    if col not in _tok:
        _tok[col] = json.loads(urllib.request.urlopen(
            f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{col}").read())["token"]
    return f"{href}?{_tok[col]}"


def item(col, iid):
    return json.loads(urllib.request.urlopen(f"{STAC}/collections/{col}/items/{iid}").read())


def read_on_grid(src, T, shape):
    """Source band on the grid (T, shape); nearest, NaN outside. Identity when the grids match."""
    if src.transform == T and src.shape == tuple(shape):
        return src.read(1).astype(np.float64)
    out = np.full(shape, np.nan, np.float32)
    reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs, dst_transform=T,
              dst_crs=src.crs, src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.nearest)
    return out.astype(np.float64)


# ----------------------------------------------------------------------------- select
def select():
    rng = random.Random(0)
    sel = {"rule_commit": "7917c79", "sites": {}}
    for site, (lat, lon) in {**SITES, **BACKUPS}.items():
        if len([s for s in sel["sites"].values() if s.get("windows")]) == 4:
            break
        if site in BACKUPS and sum(1 for s in sel["sites"].values() if s.get("windows")) >= 4:
            break
        box = [lon - 0.15, lat - 0.15, lon + 0.15, lat + 0.15]
        feats = post({"collections": ["3dep-lidar-hag"], "bbox": box, "limit": 500})
        def stat(f, k):
            return float(f["properties"]["raster:bands"]["statistics"][k]) if "statistics" in f["properties"]["raster:bands"] \
                else float(f["properties"]["statistics"][k])
        rec = {"n_candidates": len(feats)}
        chosen = None
        for thr in (8.0, 5.0):
            elig = sorted(f["id"] for f in feats if stat(f, "valid_percent") >= 95 and stat(f, "mean") >= thr)
            rec[f"eligible_mean_ge_{thr:g}"] = len(elig)
            while elig and chosen is None:
                iid = elig.pop(rng.randrange(len(elig)))
                try:
                    item("3dep-lidar-dsm", iid.replace("-hag-", "-dsm-"))
                    chosen = iid
                except Exception:  # noqa: BLE001
                    continue
            if chosen:
                rec["mean_threshold"] = thr
                break
        if not chosen:
            rec["result"] = "no eligible item"
            sel["sites"][site] = rec
            continue
        hag_it = item("3dep-lidar-hag", chosen)
        rec.update({"hag_item": chosen, "dsm_item": chosen.replace("-hag-", "-dsm-"),
                    "lidar_start": hag_it["properties"].get("start_datetime"),
                    "lidar_end": hag_it["properties"].get("end_datetime")})
        with rasterio.open(signed("3dep-lidar-hag", hag_it["assets"]["data"]["href"])) as h, \
                rasterio.open(signed("3dep-lidar-dsm", item("3dep-lidar-dsm", rec["dsm_item"])["assets"]["data"]["href"])) as d:
            hag, T, crs = h.read(1), h.transform, h.crs
            dsm = read_on_grid(d, T, h.shape)
        n = int(WIN_M / REF_RES)
        wins, tries = [], 0
        while len(wins) < 2 and tries < 500:
            tries += 1
            r, c = rng.randrange(0, hag.shape[0] - n), rng.randrange(0, hag.shape[1] - n)
            if any(abs(r - r2) < n and abs(c - c2) < n for r2, c2 in [(w["row"], w["col"]) for w in wins]):
                continue
            hw, dw = hag[r:r + n, c:c + n], dsm[r:r + n, c:c + n]
            ok_h = np.isfinite(hw) & (hw >= HAG_MIN) & (hw <= HAG_MAX)
            if ok_h.mean() >= 0.98 and np.isfinite(dw).mean() >= 0.98 and (hw[ok_h] > 5).mean() >= 0.5:
                x0, y0 = T * (c, r)
                wins.append({"row": r, "col": c, "x0": x0, "y0": y0, "draw": tries})
        rec["windows"] = wins
        rec["crs_wkt"] = crs.to_wkt()
        from pyproj import CRS
        pc = CRS.from_wkt(crs.to_wkt())
        horiz = pc.sub_crs_list[0] if pc.is_compound else pc
        rec["horiz_epsg"] = horiz.to_epsg()
        assert rec["horiz_epsg"], "horizontal CRS has no EPSG code"
        # NAIP: leaf-on (Jun-Sep), gsd <= 1.0, date closest to lidar end; ties -> finer
        from pyproj import Transformer
        tl = Transformer.from_crs(horiz, "EPSG:4326", always_xy=True)
        lidar_t = np.datetime64(rec["lidar_end"][:10])
        for w in wins:
            xs = [w["x0"], w["x0"] + WIN_M]; ys = [w["y0"] - WIN_M, w["y0"]]
            lo, la = tl.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
            nf = post({"collections": ["naip"], "bbox": [min(lo), min(la), max(lo), max(la)], "limit": 200})
            cand = [f for f in nf if f["properties"]["gsd"] <= 1.0 and 6 <= int(f["properties"]["datetime"][5:7]) <= 9]
            by_date = {}
            for f in cand:
                by_date.setdefault(f["properties"]["datetime"][:10], []).append(f)
            best = min(by_date, key=lambda d: (abs((np.datetime64(d) - lidar_t).astype(int)),
                                                 min(f["properties"]["gsd"] for f in by_date[d])))
            w["naip_date"] = best
            w["naip_items"] = sorted(f["id"] for f in by_date[best])
            w["naip_gsd"] = min(f["properties"]["gsd"] for f in by_date[best])
            w["lidar_naip_gap_days"] = int(abs((np.datetime64(best) - lidar_t).astype(int)))
        sel["sites"][site] = rec
        print(site, {k: v for k, v in rec.items() if k not in ("crs_wkt",)}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))


# ----------------------------------------------------------------------------- run
def to_navd88(H, lon, lat, vdatum):
    """Per-pixel orthometric DEM height (EGM2008 or EGM96, WGS84) -> NAVD88 via an explicit chain:
    geoid -> WGS84/ITRF2014 ellipsoid -> ITRF2014->NAD83(2011) Helmert @2015.0 -> GEOID (NAVD88).
    PROJ's default EGM->NAVD88 route uses a null NAD83(2011)<->WGS84 step (error up to ~1.3 m here)."""
    from pyproj import Transformer
    src = {"egm2008": "EPSG:4326+3855", "egm96": "EPSG:4326+5773"}[vdatum]
    a = Transformer.from_crs(src, "EPSG:4979", always_xy=True)
    b = Transformer.from_crs("EPSG:7912", "EPSG:6319", always_xy=True)
    c = Transformer.from_crs("EPSG:6319", "EPSG:6318+5703", always_xy=True)
    x, y, h = a.transform(lon, lat, np.zeros_like(lon) + 1000.0)
    x, y, h, _ = b.transform(x, y, h, np.full_like(lon, 2015.0))
    _, _, Hn = c.transform(x, y, h)
    shift = Hn - 1000.0  # H_navd88 - H_src, a smooth field
    assert np.isfinite(shift).all() and np.median(np.abs(shift)) > 0.05, f"datum no-op? {np.median(shift)}"
    return H + shift, shift


def ee_dem(name, transform, crs_epsg, shape):
    import ee
    if name == "srtm":
        img = ee.Image("USGS/SRTMGL1_003").select("elevation")
    else:
        col = ee.ImageCollection("COPERNICUS/DEM/GLO30" if name == "glo30" else "projects/sat-io/open-datasets/FABDEM")
        img = col.mosaic().setDefaultProjection(col.first().projection()).select("DEM" if name == "glo30" else "b1")
    img = img.resample("bilinear")
    T = transform
    arr = ee.data.computePixels({"expression": img.unmask(-9999).toFloat(), "fileFormat": "NUMPY_NDARRAY",
                                 "grid": {"dimensions": {"width": shape[1], "height": shape[0]},
                                          "affineTransform": {"scaleX": T.a, "shearX": T.b, "translateX": T.c,
                                                              "shearY": T.d, "scaleY": T.e, "translateY": T.f},
                                          "crsCode": f"EPSG:{crs_epsg}"}})
    a = np.asarray(arr[arr.dtype.names[0]] if arr.dtype.names else arr, dtype=np.float64)
    a[a <= -9000] = np.nan
    return a


def err_stats(pred, ref):
    m = np.isfinite(pred) & np.isfinite(ref)
    e = pred[m] - ref[m]
    return {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean(e ** 2))), "brmse": float(np.std(e)),
            "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)), "medae": float(np.median(np.abs(e)))}


def run():
    import ee
    import torch
    from scipy.stats import pearsonr, spearmanr
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_method6_gsd_film_height_balanced as hb
    from evaluate_method6_finetune_twinhead import get_device
    from vhr_dsm_pipeline import tiled_predict_margin
    proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
    ee.Initialize(project=proj)
    dev = get_device()
    exp = ROOT / "data/dfc2019/experiments"
    models = []
    for d in ("method6_height_balanced_seed42_ckpt", "method6_height_balanced_seed43", "method6_height_balanced_seed44"):
        for f in range(4):
            ck = torch.load(exp / d / f"fold{f}.pt", map_location="cpu", weights_only=False)
            m = hb.TwinHeadDav2GSD(height_scale=ck["height_scale"])
            m.load_state_dict(ck["state_dict"]); models.append(m.to(dev).eval())
    sel = json.loads((OUT / "selection.json").read_text())
    res_path = OUT / "windows.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    from pyproj import Transformer
    for site, rec in sel["sites"].items():
        for wi, w in enumerate(rec.get("windows", [])):
            key = f"{site}_w{wi}"
            if key in results:
                continue
            epsg = rec["horiz_epsg"]
            n = int(WIN_M / REF_RES)
            T2 = from_origin(w["x0"], w["y0"], REF_RES, REF_RES)
            out = {"site": site, "window": wi, **{k: w[k] for k in ("naip_date", "naip_gsd", "lidar_naip_gap_days")}}
            # reference
            ref = {}
            for col, iid in (("3dep-lidar-hag", rec["hag_item"]), ("3dep-lidar-dsm", rec["dsm_item"])):
                with rasterio.open(signed(col, item(col, iid)["assets"]["data"]["href"])) as s:
                    ref[col] = read_on_grid(s, T2, (n, n))
            hag, ldsm = ref["3dep-lidar-hag"], ref["3dep-lidar-dsm"]
            bad = ~np.isfinite(hag) | (hag < HAG_MIN) | (hag > HAG_MAX)
            out["hag_excluded_frac"] = float(bad.mean())
            hag[bad] = np.nan
            # NAIP -> 0.3 m and 0.6 m grids in the lidar's horizontal CRS
            agl2 = {}
            for gsd in (0.3, 0.6):
                npx = int(round(WIN_M / gsd))
                Tg = from_origin(w["x0"], w["y0"], gsd, gsd)
                rgb = np.full((3, npx, npx), np.nan, np.float32)
                for iid in w["naip_items"]:
                    it = item("naip", iid)
                    with rasterio.open(signed("naip", it["assets"]["image"]["href"])) as s:
                        for b in range(3):
                            tmp = np.full((npx, npx), np.nan, np.float32)
                            reproject(rasterio.band(s, b + 1), tmp, src_transform=s.transform, src_crs=s.crs,
                                      dst_transform=Tg, dst_crs=f"EPSG:{epsg}", dst_nodata=np.nan,
                                      resampling=Resampling.bilinear)
                            rgb[b] = np.where(np.isfinite(tmp), tmp, rgb[b])
                out[f"rgb_valid_frac_{gsd}"] = float(np.isfinite(rgb).all(0).mean())
                rgb = np.nan_to_num(rgb, nan=0.0)
                with torch.no_grad():
                    mu = np.mean([tiled_predict_margin(m, rgb, dev, 192)[0] for m in models], axis=0)
                if gsd == 0.3:
                    np.save(OUT / f"{key}_agl03.npy", mu.astype(np.float16))
                a2 = np.full((n, n), np.nan, np.float32)
                reproject(mu.astype(np.float32), a2, src_transform=Tg, src_crs=f"EPSG:{epsg}",
                          dst_transform=T2, dst_crs=f"EPSG:{epsg}", resampling=Resampling.average)
                agl2[gsd] = a2.astype(np.float64)
                m_ = np.isfinite(a2) & np.isfinite(hag)
                y, p = hag[m_], a2[m_].astype(np.float64)
                out[f"ndsm_{gsd}"] = {**err_stats(p, y), "pearson": float(pearsonr(y, p)[0]),
                                      "spearman": float(spearmanr(y, p)[0]), "var_ratio": float(np.var(p) / np.var(y)),
                                      "p95_pred": float(np.percentile(p, 95)), "p95_ref": float(np.percentile(y, 95)),
                                      "p99_pred": float(np.percentile(p, 99)), "p99_ref": float(np.percentile(y, 99))}
            # DEMs on the 2 m grid, per-pixel datum -> NAVD88
            xs = w["x0"] + REF_RES * (np.arange(n) + 0.5)
            ys = w["y0"] - REF_RES * (np.arange(n) + 0.5)
            X, Y = np.meshgrid(xs, ys)
            lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(X, Y)
            out["dsm_b"] = {}
            for dem, vd in (("srtm", "egm96"), ("glo30", "egm2008"), ("fabdem", "egm2008")):
                H = ee_dem(dem, T2, epsg, (n, n))
                Hn, shift = to_navd88(H, lon, lat, vd)
                comp = Hn + np.maximum(agl2[0.3], 0.0)
                comp06 = Hn + np.maximum(agl2[0.6], 0.0)
                out["dsm_b"][dem] = {"datum_shift_median_m": float(np.median(shift)),
                                     "raw": err_stats(Hn, ldsm), "composed": err_stats(comp, ldsm),
                                     "composed_0.6": err_stats(comp06, ldsm)}
            results[key] = out
            res_path.write_text(json.dumps(results, indent=1))
            print(key, json.dumps({k: out[k] for k in ("ndsm_0.3",)}),
                  {d: (round(v["raw"]["rmse"], 2), round(v["composed"]["rmse"], 2)) for d, v in out["dsm_b"].items()},
                  flush=True)


# ----------------------------------------------------------------------------- stats
def stats():
    from scipy.stats import wilcoxon
    R = json.loads((OUT / "windows.json").read_text())
    keys = sorted(R)
    summ = {"n_windows": len(keys), "tests": {}, "ndsm": {}}
    for gsd in ("0.3", "0.6"):
        summ["ndsm"][gsd] = {k: float(np.mean([R[w][f"ndsm_{gsd}"][k] for w in keys]))
                             for k in ("mae", "rmse", "pearson", "spearman", "bias", "var_ratio",
                                       "p95_pred", "p95_ref", "p99_pred", "p99_ref")}
    for metric in ("rmse", "brmse"):
        ps = {}
        for dem in ("srtm", "glo30", "fabdem"):
            raw = np.array([R[w]["dsm_b"][dem]["raw"][metric] for w in keys])
            comp = np.array([R[w]["dsm_b"][dem]["composed"][metric] for w in keys])
            p = float(wilcoxon(comp, raw, alternative="two-sided", method="exact").pvalue)
            ps[dem] = p
            summ["tests"][f"{dem}|{metric}"] = {"raw_mean": float(raw.mean()), "composed_mean": float(comp.mean()),
                                                 "composed_wins": int((comp < raw).sum()), "p": p}
        order = sorted(ps, key=ps.get)
        running = 0.0
        for i, dem in enumerate(order):  # Holm step-down, monotone
            running = max(running, min(1.0, (len(order) - i) * ps[dem]))
            summ["tests"][f"{dem}|{metric}"]["p_holm"] = running
    for dem in ("srtm", "glo30", "fabdem"):
        t = [summ["tests"][f"{dem}|{m}"] for m in ("rmse", "brmse")]
        summ[f"verdict_{dem}"] = "ADDS VALUE" if all(x["composed_mean"] < x["raw_mean"] and x["p_holm"] < 0.05
                                                     for x in t) else "does not add value"
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"select": select, "run": run, "stats": stats}[sys.argv[1]]()
