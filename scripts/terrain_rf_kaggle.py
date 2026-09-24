#!/usr/bin/env python3
"""Kaggle port of the RF stage of scripts/terrain_rf_residual.py (Part F, 2026-09-23; pre-registered, commit 3a4cf84).
Self-contained. Input: samples.parquet (features already extracted locally, identical filter below).
Reproduces exactly: sample filter, GroupKFold(5) on RGT (held-out ICESat-2 tracks), and
RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1), otherwise sklearn defaults, per variant.
Writes out-of-fold residual predictions (one .npy per fold, so it resumes after an interruption), plus the
fold assignment for verification at merge. The linear baseline, per-tile metrics and tests are computed locally.

  python terrain_rf_kaggle.py --data <bundle dir> --variant A      (or B, or AB)
Outputs in /kaggle/working/: rf_oof_<V>_fold<k>.npy, folds.npy, rf_run_meta.json

--engine sklearn (DEFAULT, CPU): exactly the pre-registered model. sklearn has no GPU implementation.
--engine xgb (OPT-IN, GPU): XGBoost random-forest mode on CUDA (P100/T4): 100 parallel trees in ONE round,
  learning_rate 1, row subsample 0.632 per tree (~ bootstrap's unique fraction, but WITHOUT replacement),
  all features per split (= sklearn regressor default max_features=1.0), min_child_weight 1 (= min_samples_leaf 1
  for squared error), no L2 (reg_lambda 0), histogram splits (max_bin 256), depth cap 20. This APPROXIMATES the
  pre-registered sklearn RF; results are a documented deviation. Outputs are prefixed xgbrf_ so they can never be
  mixed with sklearn outputs at merge.
--smoke: 20,000-row subset, fold 0 only, to check the environment in seconds (outputs prefixed smoke_).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

WC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
FEAT_A = ["h_mean", "h_std", "h_min", "h_max", "h_p90", "h_p10", "sob_mean", "sob_std", "sob_p95",
          "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std", "ndi_gr", "ndi_gb", "ndi_rb"]
FEAT_B = FEAT_A + [f"wc_{k}" for k in WC_CLASSES] + ["wc_entropy", "eth_mean", "chmv2_mean"]


def load(data):
    D = pd.read_parquet(Path(data) / "samples.parquet")
    D = D[np.isfinite(D["resid"]) & D[FEAT_B].notna().all(axis=1)].reset_index(drop=True)  # identical filter
    return D


def folds(D):
    from sklearn.model_selection import GroupKFold
    f = np.full(len(D), -1)
    for k, (_, te) in enumerate(GroupKFold(5).split(D, groups=D["rgt"])):
        f[te] = k
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="AB", help="A, B or AB")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--engine", default="sklearn", choices=["sklearn", "xgb"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    D = load(args.data)
    F = folds(D)
    prefix = ("smoke_" if args.smoke else "") + ("rf" if args.engine == "sklearn" else "xgbrf")
    if args.smoke:
        keep = np.random.default_rng(0).choice(len(D), 20000, replace=False)
        D, F = D.iloc[keep].reset_index(drop=True), F[keep]
    else:
        np.save(out / "folds.npy", F)
    meta = {"engine": args.engine, "n_samples": len(D), "n_tiles": int(D.tile_id.nunique()), "n_rgt": int(D.rgt.nunique()),
            "sklearn": sklearn.__version__, "cpu_count": os.cpu_count(), "fold_sizes": np.bincount(F).tolist()}
    if args.engine == "xgb":
        import xgboost as xgb
        import subprocess
        meta["xgboost"] = xgb.__version__
        meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
        assert meta["gpu"], "No GPU visible: set the notebook accelerator to GPU"
        XGB = dict(n_estimators=1, num_parallel_tree=100, learning_rate=1.0, subsample=0.632,
                   colsample_bynode=1.0, max_depth=20, min_child_weight=1.0, reg_lambda=0.0,
                   tree_method="hist", max_bin=256, device="cuda", random_state=0, objective="reg:squarederror")
        meta["xgb_params"] = XGB
    print(meta, flush=True)
    (out / f"{prefix}_run_meta.json").write_text(json.dumps(meta, indent=1))
    for v in args.variant:
        feats = FEAT_A if v == "A" else FEAT_B
        for k in ([0] if args.smoke else range(5)):
            p = out / f"{prefix}_oof_{v}_fold{k}.npy"
            if p.exists():
                print(f"variant {v} fold {k}: exists, skip", flush=True)
                continue
            t0 = time.time()
            tr, te = F != k, F == k
            if args.engine == "sklearn":
                rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
            else:
                rf = xgb.XGBRegressor(**XGB)
            rf.fit(D.loc[tr, feats].values, D.loc[tr, "resid"].values)
            pred = rf.predict(D.loc[te, feats].values)
            del rf
            np.save(p, pred.astype(np.float64))
            print(f"variant {v} fold {k}: n_train {tr.sum()}, n_test {te.sum()}, {time.time()-t0:.0f} s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
