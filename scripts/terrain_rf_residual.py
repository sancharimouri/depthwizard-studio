#!/usr/bin/env python3
"""Part F (2026-09-23; OPTIONAL; pre-registered in docs/method-audit/07-gamus-generalization/log.md, commit 3a4cf84).
Sentinel-2 TERRAIN product = FABDEM - r_hat, with r = FABDEM - ICESat-2 ground height, learned by a random forest
on the HRF handcrafted features of Song, Chen & Yokoya (2026, ISPRS J. 232:155-171; arXiv 2505.06905 §5.2.1, §6.1):
64x64-px window; (i) height stats mean/std/min/max/p90/p10, (ii) Sobel magnitude mean/std/p95, (iii) RGB per-channel
mean/std + 3 normalized-difference indices; variant B adds ESA WorldCover v200 fractions (11 classes) + Shannon
entropy + ETH GCH 2020 and DINOv3-CHMv2 window means. RF 100 trees. Held-out ICESat-2 tracks: GroupKFold(5) on RGT.
Baselines: raw FABDEM; per-tile linear residual r = a + b*FABDEM fitted on that tile's training-track samples.
One run, no tuning.  PROJ_NETWORK=ON .venv/bin/python scripts/terrain_rf_residual.py [--from-kaggle=<dir>]
(--from-kaggle: load the RF out-of-fold predictions produced by scripts/terrain_rf_kaggle.py instead of fitting locally.)
Output: data/sentinel2_benchmark/terrain_rf_residual/{samples.parquet,per_tile.csv,summary.json}"""
from __future__ import annotations

import os

os.environ["PROJ_NETWORK"] = "ON"  # before any pyproj import (HANDOFF §7)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BENCH = ROOT / "data/sentinel2_benchmark"
OUT = BENCH / "terrain_rf_residual"
W = 64
T08 = Transformer.from_crs("EPSG:4979", "EPSG:3855", always_xy=True)
WC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def worldcover_on_grid(rgb_path):
    import ee
    with rasterio.open(rgb_path) as s:
        T, crs, h, w = s.transform, s.crs.to_string(), s.height, s.width
    img = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    arr = ee.data.computePixels({"expression": img.unmask(0), "fileFormat": "NUMPY_NDARRAY",
                                 "grid": {"dimensions": {"width": w, "height": h},
                                          "affineTransform": {"scaleX": T.a, "shearX": T.b, "translateX": T.c,
                                                              "shearY": T.d, "scaleY": T.e, "translateY": T.f},
                                          "crsCode": crs}})
    return np.asarray(arr[arr.dtype.names[0]] if arr.dtype.names else arr)


def win_stats(a, r, c):
    h = W // 2
    return a[max(0, r - h):r + h, max(0, c - h):c + h]


def features(tid, m):
    rgb_path = ROOT / m["rgb_path"]
    with rasterio.open(rgb_path) as s:
        rgb = s.read([1, 2, 3]).astype(np.float64); T, crs = s.transform, s.crs
    fab = np.load(BENCH / "fabdem" / f"{tid}_fabdem.npy").astype(np.float64)
    eth = np.nan_to_num(np.load(BENCH / "eth_canopy_2020" / f"{tid}_eth.npy"), nan=0.0)
    chm = np.load(BENCH / "dinov3_depth" / f"{tid}_depth.npy").astype(np.float64)
    wc = worldcover_on_grid(rgb_path)
    sob = np.hypot(ndimage.sobel(np.nan_to_num(fab, nan=np.nanmean(fab)), 0), ndimage.sobel(np.nan_to_num(fab, nan=np.nanmean(fab)), 1)) / 8.0
    ph = pd.read_csv(ROOT / "data/icesat2_photons" / f"{tid}.csv")
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(ph["lon"].values, ph["lat"].values)
    cols, rows = ~T * (x, y)
    ph["row"], ph["col"] = np.floor(rows).astype(int), np.floor(cols).astype(int)
    ph = ph[(ph["row"] >= 0) & (ph["row"] < fab.shape[0]) & (ph["col"] >= 0) & (ph["col"] < fab.shape[1])]
    g = ph.groupby(["row", "col", "rgt"])["height"].median().reset_index()
    xc, yc = T * (g["col"].values + 0.5, g["row"].values + 0.5)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xc, yc)
    ortho = T08.transform(lon, lat, g["height"].values)[2]
    assert np.all(np.abs(g["height"].values - ortho) > 1), f"{tid}: geoid N~0 (PROJ grid not used)"
    g["h_ref"] = ortho
    g["fabdem"] = fab[g["row"], g["col"]]
    rows_out = []
    for r, c in zip(g["row"].values, g["col"].values):
        f = {}
        hw = win_stats(fab, r, c); hv = hw[np.isfinite(hw)]
        f.update({"h_mean": hv.mean(), "h_std": hv.std(), "h_min": hv.min(), "h_max": hv.max(),
                  "h_p90": np.percentile(hv, 90), "h_p10": np.percentile(hv, 10)} if hv.size else {})
        sw = win_stats(sob, r, c).ravel()
        f.update({"sob_mean": sw.mean(), "sob_std": sw.std(), "sob_p95": np.percentile(sw, 95)})
        cm = [win_stats(rgb[b], r, c).ravel() for b in range(3)]
        R, G, B = (v.mean() for v in cm)
        f.update({"r_mean": R, "g_mean": G, "b_mean": B, "r_std": cm[0].std(), "g_std": cm[1].std(), "b_std": cm[2].std(),
                  "ndi_gr": (G - R) / (G + R + 1e-6), "ndi_gb": (G - B) / (G + B + 1e-6), "ndi_rb": (R - B) / (R + B + 1e-6)})
        wv = win_stats(wc, r, c).ravel()
        fr = np.array([(wv == k).mean() for k in WC_CLASSES])
        f.update({f"wc_{k}": v for k, v in zip(WC_CLASSES, fr)})
        p = fr[fr > 0]
        f["wc_entropy"] = float(-(p * np.log(p)).sum())
        f["eth_mean"] = win_stats(eth, r, c).mean()
        f["chmv2_mean"] = np.nanmean(win_stats(chm, r, c))
        rows_out.append(f)
    df = pd.concat([g.reset_index(drop=True), pd.DataFrame(rows_out)], axis=1)
    df["tile_id"], df["category"] = tid, m["category"]
    df["resid"] = df["fabdem"] - df["h_ref"]
    return df


FEAT_A = ["h_mean", "h_std", "h_min", "h_max", "h_p90", "h_p10", "sob_mean", "sob_std", "sob_p95",
          "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "ndi_gr", "ndi_gb", "ndi_rb"]
FEAT_B = FEAT_A + [f"wc_{k}" for k in WC_CLASSES] + ["wc_entropy", "eth_mean", "chmv2_mean"]


def err(e):
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "brmse": float(np.std(e)), "bias": float(np.mean(e)),
            "medae": float(np.median(np.abs(e))), "n": int(len(e))}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


def main():
    import ee
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    proj = [l.split("=", 1)[1].strip().strip('"') for l in open(ROOT / ".env") if l.startswith("EARTHENGINE_PROJECT")][0]
    ee.Initialize(project=proj)
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "samples.parquet"
    if sp.exists():
        D = pd.read_parquet(sp)
    else:
        man = pd.read_csv(BENCH / "manifest.csv").set_index("tile_id")
        D = pd.concat([features(t, man.loc[t]) for t in man.index], ignore_index=True)
        D.to_parquet(sp)
    D = D[np.isfinite(D["resid"]) & D[FEAT_B].notna().all(axis=1)].reset_index(drop=True)
    print(f"samples {len(D)}, tiles {D.tile_id.nunique()}, RGTs {D.rgt.nunique()}", flush=True)
    D["fold"] = -1
    for k, (_, te) in enumerate(GroupKFold(5).split(D, groups=D["rgt"])):
        D.loc[te, "fold"] = k
    for v in ("A", "B"):
        D[f"rhat_{v}"] = np.nan
    D["rhat_lin"] = 0.0
    fallback = 0
    kag = next((Path(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--from-kaggle=")), None)
    # --kaggle-engine=xgb: the OPT-IN GPU approximation (XGBoost RF mode) -- a documented deviation from the
    # pre-registered sklearn RF; its files are prefixed xgbrf_ and results are labelled as such.
    engine = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--kaggle-engine=")), "sklearn")
    pre = "rf" if engine == "sklearn" else "xgbrf"
    if kag is not None:  # RF stage was run on Kaggle (scripts/terrain_rf_kaggle.py): same filter, folds
        assert np.array_equal(np.load(kag / "folds.npy"), D["fold"].values), "Kaggle fold assignment differs"
        S_meta = json.loads((kag / f"{pre}_run_meta.json").read_text())
        print("Kaggle RF run:", S_meta, flush=True)
    for k in range(5):
        tr, te = D["fold"] != k, D["fold"] == k
        for v, F in (("A", FEAT_A), ("B", FEAT_B)):
            if kag is not None:
                D.loc[te, f"rhat_{v}"] = np.load(kag / f"{pre}_oof_{v}_fold{k}.npy")
                continue
            rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
            rf.fit(D.loc[tr, F].values, D.loc[tr, "resid"].values)
            D.loc[te, f"rhat_{v}"] = rf.predict(D.loc[te, F].values)
        for tid in D.loc[te, "tile_id"].unique():
            trt = tr & (D["tile_id"] == tid); tet = te & (D["tile_id"] == tid)
            if trt.sum() < 10:
                fallback += 1
                continue
            b, a = np.polyfit(D.loc[trt, "fabdem"], D.loc[trt, "resid"], 1)
            D.loc[tet, "rhat_lin"] = a + b * D.loc[tet, "fabdem"]
    rows = []
    for tid, x in D.groupby("tile_id"):
        r = {"tile_id": tid, "category": x["category"].iloc[0], "n": len(x), "n_rgt": x["rgt"].nunique()}
        for name, rh in (("raw", 0.0), ("linear", x["rhat_lin"]), ("rf_A", x["rhat_A"]), ("rf_B", x["rhat_B"])):
            e = (x["fabdem"] - rh) - x["h_ref"]
            r.update({f"{name}_{k}": v for k, v in err(e.values).items() if k != "n"})
        rows.append(r)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / ("per_tile.csv" if engine == "sklearn" else f"per_tile_{pre}.csv"), index=False)
    S = {"rf_engine": engine if kag is not None else "sklearn (local)", "n_samples": len(D), "n_tiles": len(P), "n_rgt": int(D.rgt.nunique()), "linear_fallback_tile_folds": fallback,
         "median": {c: float(P[c].median()) for c in P.columns if c.endswith(("_rmse", "_brmse", "_bias", "_medae"))},
         "tests": {}}
    for metric in ("rmse", "brmse"):
        for base in ("raw", "linear"):
            ps = []
            for v in ("A", "B"):
                a, b = P[f"rf_{v}_{metric}"], P[f"{base}_{metric}"]
                p = float(stats.wilcoxon(a, b).pvalue)
                ps.append(p)
                S["tests"][f"rf_{v}|vs_{base}|{metric}"] = {"wins": int((a < b).sum()), "n": len(P), "p": p,
                                                            "median_diff": float((a - b).median())}
            for v, ph in zip(("A", "B"), holm(ps)):
                S["tests"][f"rf_{v}|vs_{base}|{metric}"]["p_holm"] = ph
    for v in ("A", "B"):
        t = [S["tests"][f"rf_{v}|vs_{b}|{m}"] for b in ("raw", "linear") for m in ("rmse", "brmse")]
        S[f"verdict_rf_{v}"] = "ADOPT" if all(x["median_diff"] < 0 and x["p_holm"] < 0.05 and x["wins"] >= 20 for x in t) else "NOT ADOPTED"
    (OUT / ("summary.json" if engine == "sklearn" else f"summary_{pre}.json")).write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
