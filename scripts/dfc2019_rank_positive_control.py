#!/usr/bin/env python3
"""DFC2019 positive control for the rank-loss / raw-Spearman methodology (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, "DFC2019 positive control").

Question: can the exact pipeline that returned ~0.11-0.13 raw within-crop Spearman on Sentinel-2 / Landsat detect
strong signal where it is known to exist? Same pattern as scripts/landsat_rank_test.py: only the cache contents change.
  - Imagery: the 50 DFC2019 Track-1 RGB tiles at native GSD (~0.3 m; see "Housekeeping" in the doc), uint8, no
    resampling, blurring or degradation of any kind. Tile set = evaluate_method6_finetune_twinhead.tile_ids().
  - Truth: DFC2019 dense LiDAR AGL (*_AGL.tif; invalid = non-finite or < 0 -> NaN), in every slot the pipeline reads:
      fab30 (training target / calibration target, the "30 m grid" slot) = 3x3 block nanmean of AGL (NaN if the whole
            block is invalid); fab10 = AGL; ph_row/ph_col/ph_h ("ICESat-2 cells") = the centre pixel of every 3x3
            block with valid AGL (a stride-3 subsample of the dense truth, ~400 cells per crop, for tractability).
  - Inclusion (pre-registered, data availability only): a tile is used only if every quadrant keeps
    >= MIN_CROPS_PER_QUAD crops under C.load_crops' fully-finite-target rule (OMA tiles have large invalid AGL areas);
    31/50 tiles (26 JAX, 5 OMA). Written to inclusion.csv by prep.
  - Oracle raw signal: frozen DAv2-Large whole-tile depth already used for every DFC2019 oracle
    (data/dfc2019/experiments/dav2_baseline/depth/{tile}_depth.npy).
  - Model / loss / geometry / folds / calibration / analysis: imported unchanged from s2_rank_loss_test (fit,
    calibrated_eval, scores_all, oracle_scores, analyze) and s2_rank_raw_spearman (cells_for_fold, summarize), on the
    same trained models (no re-fit), exactly as the Landsat run.

  .venv/bin/python scripts/dfc2019_rank_positive_control.py prep
  .venv/bin/python scripts/dfc2019_rank_positive_control.py run
Output: data/dfc2019/experiments/rank_positive_control/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s2_rank_loss_test as R  # noqa: E402
import s2_rank_raw_spearman as RS  # noqa: E402
import s2_token_grid_phase_c as C  # noqa: E402
from evaluate_method6_finetune_twinhead import get_device, load_tile_rgb_agl, tile_ids  # noqa: E402

OUT = ROOT / "data/dfc2019/experiments/rank_positive_control"
DAV2 = ROOT / "data/dfc2019/experiments/dav2_baseline/depth"
K = C.K
MIN_CROPS_PER_QUAD = 4  # pre-registered inclusion rule (data availability only, fixed before any training)


def n_valid_crops(fab30, q):
    """Crops of quadrant q that C.load_crops keeps (fully finite target)."""
    return sum(bool(np.isfinite(fab30[r // K:(r + C.N) // K, c // K:(c + C.N) // K]).all()) for r, c in C.crop_origins(q))


def prep(args):
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    for tid in tile_ids():
        out = OUT / "cache" / f"{tid}.npz"
        if out.exists():
            continue
        rgb, agl, valid = load_tile_rgb_agl(tid)
        agl = np.where(valid, agl, np.nan).astype(np.float32)
        H, W = agl.shape
        h3, w3 = H // K, W // K
        blocks = agl[:h3 * K, :w3 * K].reshape(h3, K, w3, K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks -> NaN, intended
            fab30 = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)
        rr, cc = np.meshgrid(np.arange(h3) * K + 1, np.arange(w3) * K + 1, indexing="ij")
        ok = np.isfinite(agl[rr, cc])
        np.savez_compressed(out, rgb=rgb.round().astype(np.uint8), fab30=fab30, fab10=agl,
                            ph_row=rr[ok], ph_col=cc[ok], ph_h=agl[rr, cc][ok].astype(np.float64))
        print(f"{tid}: {H}x{W}, fab30 finite {np.isfinite(fab30).mean():.3f}, cells {ok.sum()}", flush=True)
    rows = []
    for p in sorted((OUT / "cache").glob("*.npz")):
        f = np.load(p)["fab30"]
        n = [n_valid_crops(f, q) for q in range(4)]
        rows.append({"tile": p.stem, "crops_q0": n[0], "crops_q1": n[1], "crops_q2": n[2], "crops_q3": n[3],
                     "included": min(n) >= MIN_CROPS_PER_QUAD})
    import pandas as pd
    D = pd.DataFrame(rows); D.to_csv(OUT / "inclusion.csv", index=False)
    print(f"included {int(D.included.sum())}/{len(D)} tiles (>= {MIN_CROPS_PER_QUAD} valid crops in every quadrant)")


def run(args):
    C.CACHE, R.OUT, R.DAV2 = OUT / "cache", OUT / "rank_loss", DAV2
    R.OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    import pandas as pd
    D = pd.read_csv(OUT / "inclusion.csv")
    tids = sorted(D[D.included].tile)
    assert len(tids) == 31, len(tids)
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
        print(f"fold{fold}: done {time.time()-t0:.0f}s, {len(rows)} held-out crops with cells", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    R.analyze(None)
    (R.OUT / "raw_spearman").mkdir(exist_ok=True)
    RS.summarize(allrows, tids, None, R.OUT / "raw_spearman")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run"])
    a = ap.parse_args()
    {"prep": prep, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
