#!/usr/bin/env python3
"""
DepthWizard2 — Stage 4: Offline spatial-residual learnability diagnostic.

This is an OFFLINE ORACLE experiment. Exact Stage-1 grid/20 anchors provide
an anchor-based global affine calibration. Full DFC2019 AGL truth is then
used offline to fit a low-complexity residual field. Four spatial folds test
whether that residual field generalizes to held-out spatial regions.

Models:
  constant:     r(x,y) = c
  linear_xy:    r(x,y) = c + cx*x + cy*y
  quadratic_xy: r(x,y) = c + cx*x + cy*y + cxx*x² + cxy*xy + cyy*y²

Primary question:
  Is there learnable spatial structure in the residual after the global
  sparse-anchor calibration, even when independent local affine models fail?

This does NOT demonstrate deployment feasibility because residual training
uses full reference AGL truth. It is a diagnostic upper-bound test.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/dfc2019/experiments/dav2_baseline/manifest.csv"
DEPTH_DIR = ROOT / "data/dfc2019/experiments/dav2_baseline/depth"
TRUTH_DIR = ROOT / "data/dfc2019/raw/Truth/Track1-Truth"
STAGE1 = ROOT / "data/dfc2019/experiments/sparse_anchor_robust"
OUT = ROOT / "data/dfc2019/experiments/sparse_anchor_spatial_residual"

EPS = 1e-6
DEFAULT_BUFFER = 16
DEFAULT_TRAIN_SAMPLES = 50000
DEFAULT_SEED = 20260915


def read_depth(p: Path) -> np.ndarray:
    a = np.load(p, allow_pickle=False)
    if a.ndim != 2:
        raise ValueError(f"Depth must be 2-D: {p}: {a.shape}")
    return a.astype(np.float32, copy=False)


def read_agl(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32, copy=False)


def valid_mask(d, a):
    return np.isfinite(d) & np.isfinite(a) & (a >= -EPS)


def corrs(y, p):
    if len(y) < 2 or np.allclose(y, y[0]) or np.allclose(p, p[0]):
        return math.nan, math.nan
    return float(pearsonr(y, p).statistic), float(spearmanr(y, p).statistic)


def metrics(y, p):
    r = p - y
    pr, sr = corrs(y, p)
    return {
        "mae_m": float(np.mean(np.abs(r))),
        "rmse_m": float(np.sqrt(np.mean(r * r))),
        "bias_m": float(np.mean(r)),
        "pearson": pr,
        "spearman": sr,
    }


def exclusion_mask(shape, rows, cols, radius):
    h, w = shape
    if radius < 0:
        raise ValueError("buffer radius must be >= 0")
    out = np.zeros(shape, bool)
    r = int(radius)
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    disk = (yy * yy + xx * xx) <= r * r
    dy, dx = np.where(disk)
    dy -= r; dx -= r
    for row, col in zip(rows, cols):
        rr, cc = row + dy, col + dx
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        out[rr[ok], cc[ok]] = True
    return out


def anchor_path(tile, placement="grid", n=20):
    return STAGE1 / "anchors" / f"{tile}_{placement}_{n}_00.json"


def load_anchors(tile):
    p = anchor_path(tile)
    if not p.exists():
        raise FileNotFoundError(f"Missing Stage-1 anchor file: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    aa = data.get("anchors") or []
    rows = np.array([int(x["row"]) for x in aa], dtype=np.int32)
    cols = np.array([int(x["col"]) for x in aa], dtype=np.int32)
    agl = np.array([float(x["agl_m"]) for x in aa], dtype=np.float64)
    d = np.array([float(x["dav2"]) for x in aa], dtype=np.float64)
    if len(aa) != 20:
        raise ValueError(f"{tile}: expected 20 grid anchors, got {len(aa)}")
    if len(np.unique(np.c_[rows, cols], axis=0)) != 20:
        raise ValueError(f"{tile}: duplicate anchors")
    return rows, cols, agl, d, p


def folds(shape):
    h, w = shape
    rr, cc = np.indices(shape, dtype=np.int32)
    top = rr < h / 2
    left = cc < w / 2
    f = np.empty(shape, dtype=np.int8)
    f[top & left] = 0
    f[top & ~left] = 1
    f[~top & left] = 2
    f[~top & ~left] = 3
    return f


def features(rows, cols, h, w, name):
    x = cols.astype(np.float64) / (w - 1) * 2 - 1
    y = rows.astype(np.float64) / (h - 1) * 2 - 1
    if name == "constant":
        return np.c_[np.ones_like(x)]
    if name == "linear_xy":
        return np.c_[np.ones_like(x), x, y]
    if name == "quadratic_xy":
        return np.c_[np.ones_like(x), x, y, x*x, x*y, y*y]
    raise ValueError(name)


def fit_residual(rows, cols, resid, shape, name):
    X = features(rows, cols, *shape, name)
    m = LinearRegression(fit_intercept=False)
    m.fit(X, resid)
    return m


def spatial_neighbor_corr(resid, valid):
    vals = {}
    for nm, dr, dc in (("horizontal",0,1),("vertical",1,0)):
        r0 = slice(0, resid.shape[0] - dr if dr else resid.shape[0])
        r1 = slice(dr, resid.shape[0])
        c0 = slice(0, resid.shape[1] - dc if dc else resid.shape[1])
        c1 = slice(dc, resid.shape[1])
        m = valid[r0,c0] & valid[r1,c1] & np.isfinite(resid[r0,c0]) & np.isfinite(resid[r1,c1])
        a, b = resid[r0,c0][m].astype(float), resid[r1,c1][m].astype(float)
        vals[f"neighbor_corr_{nm}"] = float(pearsonr(a,b).statistic) if len(a) > 1 else math.nan
    vals["neighbor_corr_mean"] = float(np.nanmean(list(vals.values())))
    return vals


def sample_indices(n, cap, seed):
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, cap, replace=False))


def tile_run(tile, depth_path, truth_path, buffer_radius, train_cap, seed, models, outdir):
    d = read_depth(depth_path)
    a = read_agl(truth_path)
    if d.shape != a.shape:
        raise ValueError(f"{tile}: shape mismatch {d.shape} vs {a.shape}")
    valid = valid_mask(d, a)
    ar, ac, ay, ad, ap = load_anchors(tile)
    if not np.all(valid[ar, ac]):
        raise ValueError(f"{tile}: invalid Stage-1 anchor")
    curd = d[ar, ac].astype(float)
    if not np.allclose(curd, ad, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{tile}: saved/current DAv2 anchor mismatch")

    gm = LinearRegression().fit(curd.reshape(-1,1), ay)
    slope = float(gm.coef_[0]); intercept = float(gm.intercept_)
    gp = gm.predict(d.astype(float).reshape(-1,1)).reshape(d.shape)

    exc = exclusion_mask(d.shape, ar, ac, buffer_radius)
    eligible = valid & ~exc
    er, ec = np.where(eligible)
    if len(er) < 1000:
        raise ValueError(f"{tile}: too few eligible pixels")

    y_all = a[eligible].astype(float)
    p_all = gp[eligible].astype(float)
    baseline = metrics(y_all, p_all)
    resid = a.astype(float) - gp
    resid[~valid] = np.nan

    f = folds(d.shape)
    h, w = d.shape
    rows = []

    # Baseline row: same for all folds, but separately report the baseline on each fold
    # so the paired delta is exactly fold-matched.
    for fold in range(4):
        test = eligible & (f == fold)
        trn = eligible & (f != fold)
        trr, trc = np.where(trn)
        ter, tec = np.where(test)
        if len(ter) < 100 or len(trr) < 100:
            raise ValueError(f"{tile}: insufficient fold {fold}")
        by = a[ter,tec].astype(float)
        bp = gp[ter,tec].astype(float)
        bm = metrics(by,bp)
        rows.append({
            "tile_id":tile,"region":"JAX" if tile.startswith("JAX_") else "OMA" if tile.startswith("OMA_") else "UNKNOWN",
            "model":"global_affine","fold":fold,"fold_name":f"Q{fold+1}",
            "evaluation":"strict_blocked_cv","train_source":"stage1_sparse_anchors",
            "train_samples":20,"eval_pixels":len(ter),"anchor_count":20,"placement":"grid",**bm,
            "delta_mae_m":0.0,"delta_rmse_m":0.0,"delta_pearson":0.0,"delta_spearman":0.0
        })

        sample = sample_indices(len(trr), train_cap, seed + fold * 1009 + sum(tile.encode()))
        sr, sc = trr[sample], trc[sample]
        sy = resid[sr,sc].astype(float)
        for name in models:
            m = fit_residual(sr, sc, sy, d.shape, name)
            corr = m.predict(features(ter, tec, h, w, name))
            final = bp + corr
            met = metrics(by, final)
            rows.append({
                "tile_id":tile,"region":"JAX" if tile.startswith("JAX_") else "OMA" if tile.startswith("OMA_") else "UNKNOWN",
                "model":name,"fold":fold,"fold_name":f"Q{fold+1}",
                "evaluation":"strict_blocked_cv","train_source":"full_AGL_oracle_residual_training",
                "train_samples":len(sample),"eval_pixels":len(ter),"anchor_count":20,"placement":"grid",
                "baseline_test_mae_m":bm["mae_m"],"baseline_test_rmse_m":bm["rmse_m"],
                "baseline_test_pearson":bm["pearson"],"baseline_test_spearman":bm["spearman"],**met,
                "delta_mae_m":bm["mae_m"]-met["mae_m"],"delta_rmse_m":bm["rmse_m"]-met["rmse_m"],
                "delta_pearson":met["pearson"]-bm["pearson"],"delta_spearman":met["spearman"]-bm["spearman"]
            })

    # Offline oracle fit: train on all eligible reference pixels and evaluate on the same eligible set.
    oracle = {}
    all_idx = np.arange(len(er))
    for name in models:
        sel = sample_indices(len(all_idx), train_cap, seed + 7001 + sum(tile.encode()))
        sr, sc = er[sel], ec[sel]
        m = fit_residual(sr, sc, resid[sr,sc].astype(float), d.shape, name)
        final = p_all + m.predict(features(er,ec,h,w,name))
        oracle[name] = {**metrics(y_all,final),"train_samples":len(sel)}

    nbors = spatial_neighbor_corr(resid, eligible)

    diagdir = outdir / "diagnostics"; diagdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(diagdir / f"{tile}.npz", residual=resid.astype(np.float32), valid=valid, eligible=eligible, folds=f)

    fitinfo = {
        "tile_id":tile,"global_slope":slope,"global_intercept":intercept,
        "eligible_pixels":len(er),"residual_mean":float(np.mean(resid[eligible])),
        "residual_std":float(np.std(resid[eligible])),"residual_p95_abs":float(np.percentile(np.abs(resid[eligible]),95)),
        **nbors,"oracle":oracle
    }
    return rows, fitinfo


def summarize(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s): return {}
    return {"mean":float(s.mean()),"median":float(s.median()),"std":float(s.std(ddof=0)),
            "p05":float(s.quantile(.05)),"p25":float(s.quantile(.25)),"p75":float(s.quantile(.75)),"p95":float(s.quantile(.95)),
            "min":float(s.min()),"max":float(s.max())}


def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,default=MANIFEST)
    p.add_argument("--depth-dir",type=Path,default=DEPTH_DIR)
    p.add_argument("--truth-dir",type=Path,default=TRUTH_DIR)
    p.add_argument("--stage1-dir",type=Path,default=STAGE1)
    p.add_argument("--output-dir",type=Path,default=OUT)
    p.add_argument("--anchors",type=int,default=20)
    p.add_argument("--buffer-radius",type=int,default=DEFAULT_BUFFER)
    p.add_argument("--train-samples",type=int,default=DEFAULT_TRAIN_SAMPLES)
    p.add_argument("--seed",type=int,default=DEFAULT_SEED)
    p.add_argument("--spatial-models",nargs="+",choices=["constant","linear_xy","quadratic_xy"],default=["constant","linear_xy","quadratic_xy"])
    p.add_argument("--tiles",nargs="*",default=None)
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


def main():
    args=parse()
    if args.anchors != 20: raise ValueError("Stage 4 is locked to the Stage-1 20-anchor grid baseline.")
    if args.buffer_radius < 0 or args.train_samples < 100: raise ValueError("Invalid buffer/train sample setting.")
    out=args.output_dir.resolve()
    if out.exists() and args.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    man=pd.read_csv(args.manifest.resolve())
    tiles=man["tile_id"].astype(str).tolist()
    if args.tiles:
        req=set(args.tiles); missing=sorted(req-set(tiles))
        if missing: raise ValueError("Missing tiles: "+", ".join(missing))
        tiles=[t for t in tiles if t in req]
    if args.limit is not None: tiles=tiles[:args.limit]
    if not tiles: raise ValueError("No tiles selected.")

    print("="*78); print("STAGE 4 — OFFLINE SPATIAL RESIDUAL DIAGNOSTIC"); print("="*78)
    print("Full AGL truth is used to train the residual model: OFFLINE ORACLE TEST, NOT DEPLOYMENT.")
    print(f"Tiles: {len(tiles)} | anchors: 20 grid | models: {args.spatial_models} | train cap/fold: {args.train_samples}")
    print("="*78)

    pre=[]
    for i,t in enumerate(tiles,1):
        dp=args.depth_dir.resolve()/f"{t}_depth.npy"; tp=args.truth_dir.resolve()/f"{t}_AGL.tif"; d=read_depth(dp); a=read_agl(tp)
        if d.shape!=a.shape: raise ValueError(f"{t}: shape mismatch")
        vr=valid_mask(d,a); ar,ac,aa,ad,_=load_anchors(t)
        if not np.all(vr[ar,ac]): raise ValueError(f"{t}: invalid Stage-1 anchor")
        if not np.allclose(d[ar,ac],ad,rtol=1e-6,atol=1e-6): raise ValueError(f"{t}: DAv2 mismatch at anchors")
        pre.append({"tile_id":t,"valid_pixels":int(vr.sum()),"anchor_dav2_range":float(np.ptp(d[ar,ac])),"anchor_agl_range":float(np.ptp(aa)),"status":"READY"})
        print(f"{i:3d}/{len(tiles)} {t:14s} READY")
    pd.DataFrame(pre).to_csv(out/"preflight.csv",index=False)

    (out/"config.json").write_text(json.dumps({
        "experiment":"stage4_offline_spatial_residual","offline_oracle":True,
        "full_truth_used_for_residual_training":True,"anchors":20,"placement":"grid",
        "buffer_radius_px":args.buffer_radius,"train_samples_per_fold":args.train_samples,
        "spatial_folds":4,"models":args.spatial_models,"seed":args.seed,"tiles":tiles
    },indent=2),encoding="utf-8")

    results=[]; fits=[]
    for i,t in enumerate(tiles,1):
        print(f"{i:3d}/{len(tiles)} {t:14s} ...",end=" ",flush=True)
        rr,fi=tile_run(
            t,
            args.depth_dir.resolve()/f"{t}_depth.npy",
            args.truth_dir.resolve()/f"{t}_AGL.tif",
            args.buffer_radius,
            args.train_samples,
            args.seed,
            args.spatial_models,
            out,
        )
        results.extend(rr); fits.append(fi)
        msg=[]
        for m in args.spatial_models:
            q=[r for r in rr if r["model"]==m and r["evaluation"]=="strict_blocked_cv"]
            msg.append(f"{m} ΔRMSE={np.mean([x['delta_rmse_m'] for x in q]):+.3f}m ΔP={np.mean([x['delta_pearson'] for x in q]):+.4f}")
        print(" | ".join(msg))

    rdf=pd.DataFrame(results); rdf.to_csv(out/"results.csv",index=False)
    cv=rdf[rdf.evaluation=="strict_blocked_cv"].copy()
    srows=[]
    for m,g in cv.groupby("model",sort=True):
        row={"model":m,"fold_runs":len(g),"tiles":g.tile_id.nunique()}
        for col in ["mae_m","rmse_m","pearson","spearman","delta_mae_m","delta_rmse_m","delta_pearson","delta_spearman"]:
            for k,v in summarize(g[col]).items(): row[f"{col}_{k}"]=v
        for col,nm in [("delta_mae_m","improved_mae_folds"),("delta_rmse_m","improved_rmse_folds"),("delta_pearson","improved_pearson_folds"),("delta_spearman","improved_spearman_folds")]:
            row[nm]=int((pd.to_numeric(g[col],errors="coerce")>0).sum())
        srows.append(row)
    sdf=pd.DataFrame(srows); sdf.to_csv(out/"summary.csv",index=False)

    # Tile-level mean deltas are the unit for the robustness decision.
    trows=[]
    for (m,t),g in cv.groupby(["model","tile_id"],sort=True):
        r={"model":m,"tile_id":t,"region":g.region.iloc[0]}
        for c in ["delta_mae_m","delta_rmse_m","delta_pearson","delta_spearman"]: r[c+"_mean"]=float(g[c].mean())
        trows.append(r)
    tdf=pd.DataFrame(trows); tdf.to_csv(out/"tile_summary.csv",index=False)

    # Spatial diagnostics.
    dr=[]
    oracle_rows=[]
    for f in fits:
        dr.append({k:f[k] for k in ["tile_id","global_slope","global_intercept","eligible_pixels","residual_mean","residual_std","residual_p95_abs","neighbor_corr_horizontal","neighbor_corr_vertical","neighbor_corr_mean"]})
        for m,v in f["oracle"].items(): oracle_rows.append({"tile_id":f["tile_id"],"model":m,**v})
    pd.DataFrame(dr).to_csv(out/"spatial_diagnostics.csv",index=False)
    pd.DataFrame(oracle_rows).to_csv(out/"oracle_in_sample.csv",index=False)

    pairs=[]
    for m,g in tdf.groupby("model"):
        for c in ["delta_mae_m_mean","delta_rmse_m_mean","delta_pearson_mean","delta_spearman_mean"]:
            v=pd.to_numeric(g[c],errors="coerce").dropna()
            pairs.append({"model":m,"metric":c,"tiles":len(v),"mean":float(v.mean()),"median":float(v.median()),"std":float(v.std(ddof=0)),"p05":float(v.quantile(.05)),"p95":float(v.quantile(.95)),"positive_tiles":int((v>0).sum())})
    pd.DataFrame(pairs).to_csv(out/"paired_deltas.csv",index=False)

    report=[
        "DepthWizard2 — Stage 4 Offline Spatial Residual Diagnostic", "="*68, "",
        "OFFLINE ORACLE WARNING: full DFC2019 AGL truth is used to train the residual model.",
        "A positive result demonstrates residual learnability, not deployment feasibility.", "",
        "PRIMARY: 4-fold spatial blocked cross-validation against global sparse-anchor affine baseline.", "",
        sdf.to_string(index=False), "",
        "TILE-LEVEL PAIRED DELTAS (positive = spatial residual improvement):", "",
        pd.DataFrame(pairs).to_string(index=False), "",
        "SPATIAL RESIDUAL AUTOCORRELATION DIAGNOSTICS:", "",
        pd.DataFrame(dr)[["neighbor_corr_horizontal","neighbor_corr_vertical","neighbor_corr_mean"]].describe().to_string(), "",
        f"results.csv: {out/'results.csv'}", f"summary.csv: {out/'summary.csv'}", f"paired_deltas.csv: {out/'paired_deltas.csv'}",
        f"oracle_in_sample.csv: {out/'oracle_in_sample.csv'}", f"spatial_diagnostics.csv: {out/'spatial_diagnostics.csv'}"
    ]
    (out/"REPORT.txt").write_text("\n".join(report),encoding="utf-8")
    print("\n"+"="*78); print("STAGE 4 COMPLETE"); print("="*78)
    print(sdf.to_string(index=False)); print(f"Report: {out/'REPORT.txt'}")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: raise SystemExit(130)
