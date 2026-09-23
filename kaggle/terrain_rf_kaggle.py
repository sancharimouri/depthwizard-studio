#!/usr/bin/env python3
"""Kaggle port of the RF stage of scripts/terrain_rf_residual.py (Part F, 2026-09-23; pre-registered, commit 3a4cf84).
Self-contained. Input: samples.parquet (features already extracted locally, identical filter below).
Reproduces exactly: sample filter, GroupKFold(5) on RGT (held-out ICESat-2 tracks), and
RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1), otherwise sklearn defaults, per variant.
Writes out-of-fold residual predictions (one .npy per fold, so it resumes after an interruption), plus the
fold assignment for verification at merge. The linear baseline, per-tile metrics and tests are computed locally.

  python terrain_rf_kaggle.py --data <bundle dir> --variant A      (or B, or AB)
Outputs in /kaggle/working/: rf_oof_<V>_fold<k>.npy, folds.npy, rf_run_meta.json
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
    args = ap.parse_args()
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    D = load(args.data)
    F = folds(D)
    np.save(out / "folds.npy", F)
    meta = {"n_samples": len(D), "n_tiles": int(D.tile_id.nunique()), "n_rgt": int(D.rgt.nunique()),
            "sklearn": sklearn.__version__, "cpu_count": os.cpu_count(), "fold_sizes": np.bincount(F).tolist()}
    print(meta, flush=True)
    (out / "rf_run_meta.json").write_text(json.dumps(meta, indent=1))
    for v in args.variant:
        feats = FEAT_A if v == "A" else FEAT_B
        for k in range(5):
            p = out / f"rf_oof_{v}_fold{k}.npy"
            if p.exists():
                print(f"variant {v} fold {k}: exists, skip", flush=True)
                continue
            t0 = time.time()
            tr, te = F != k, F == k
            rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
            rf.fit(D.loc[tr, feats].values, D.loc[tr, "resid"].values)
            pred = rf.predict(D.loc[te, feats].values)
            del rf
            np.save(p, pred.astype(np.float64))
            print(f"variant {v} fold {k}: n_train {tr.sum()}, n_test {te.sum()}, {time.time()-t0:.0f} s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
