#!/usr/bin/env python3
"""Raw (pre-calibration) Spearman check on the Sentinel-2 rank-loss run (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, "Raw-Spearman check").

The rank-loss run saved only aggregates (per-tile squared-error sums and slopes), not per-pixel scores or checkpoints,
so each fold is re-fitted with the identical, seeded procedure (s2_rank_loss_test.fit). Reproduction is verified
against the committed eval_rank_fold*.json (per-tile slope and calibrated ICESat-2 RMSE).

Per held-out crop with >= MIN_CELLS ICESat-2 cells: Spearman(score at the photon cell, ICESat-2 height) for
  - the rank model's raw output (10 m grid),
  - frozen DAv2-Large whole-tile depth (the oracle's raw signal),
  - FABDEM 10 m (reference: the terrain product's own within-crop ranking).
Per tile: cell-weighted mean of crop Spearmans (held-out quadrants of all 4 folds). Also: whole-tile raw Spearman
(caveated: the rank model's per-crop offsets are arbitrary) and the linear calibration's implied within-crop
correlation, sqrt(max(0, R^2_relief)), R^2_relief = 1 - SSE(calibrated, crop-demeaned) / SS(h, crop-demeaned).

  .venv/bin/python scripts/s2_rank_raw_spearman.py
Output: data/sentinel2_benchmark/token_grid_test/rank_loss/raw_spearman/{cells_fold*.npz, summary.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s2_rank_loss_test as R  # noqa: E402
import s2_token_grid_phase_c as C  # noqa: E402
from evaluate_method6_finetune_twinhead import get_device  # noqa: E402

OUT = R.OUT / "raw_spearman"
N = C.N
MIN_CELLS = 10


def cells_for_fold(model, tids, fold, device):
    rows = []
    for tid in tids:
        sm, so = R.scores_all(model, tid, device), R.oracle_scores(tid)
        a_m = R.calibrated_eval(sm, tid, fold)["slope"]
        z = np.load(C.CACHE / f"{tid}.npz")
        prow, pcol, ph = z["ph_row"].astype(int), z["ph_col"].astype(int), z["ph_h"]
        pfab = z["fab10"][prow, pcol]
        for ci, ((q, r, c), (s, t)) in enumerate(sm.items()):
            if q != fold:
                continue
            sel = (prow >= r) & (prow < r + N) & (pcol >= c) & (pcol < c + N) & np.isfinite(pfab)
            if not sel.any():
                continue
            s30 = s.reshape(N // 3, 3, N // 3, 3).mean((1, 3))
            rr, cc = prow[sel] - r, pcol[sel] - c
            d = so[(q, r, c)][0]
            rows.append(dict(tile=tid, crop=ci, h=ph[sel], model=s[rr, cc], oracle=d[rr, cc], fab=pfab[sel],
                             calib=float(t.mean()) + a_m * (s[rr, cc] - s30.mean())))
    return rows


def main():
    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    tids = sorted(p.stem for p in C.CACHE.glob("*.npz"))
    allrows, repro = [], []
    for fold in range(4):
        saved = {r["tile"]: r["model"] for r in json.loads((R.OUT / f"eval_rank_fold{fold}.json").read_text())["records"]}
        model, hs, run = R.fit(fold, tids, 600, 4, device)
        for tid in tids:  # reproduction check against the committed run
            m = R.calibrated_eval(R.scores_all(model, tid, device), tid, fold)
            s = saved[tid]
            repro.append(dict(fold=fold, tile=tid, slope_new=m["slope"], slope_saved=s["slope"],
                              rmse_new=float(np.sqrt(m["ice_se"] / m["ice_n"])),
                              rmse_saved=float(np.sqrt(s["ice_se"] / s["ice_n"]))))
        rows = cells_for_fold(model, tids, fold, device)
        for r in rows:
            r["fold"] = fold
        allrows += rows
        print(f"fold{fold}: {len(rows)} held-out crops with ICESat-2 cells", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    rp = np.array([[x["rmse_new"], x["rmse_saved"], x["slope_new"], x["slope_saved"]] for x in repro])
    rmse_rel = np.abs(rp[:, 0] - rp[:, 1]) / rp[:, 1]
    slope_sign_agree = float(np.mean(np.sign(rp[:, 2]) == np.sign(rp[:, 3])))

    def crop_rho(x, h):
        if len(h) < MIN_CELLS or np.ptp(h) == 0 or np.ptp(x) == 0:
            return None
        return float(spearmanr(x, h).statistic)

    per_tile = {}
    for tid in tids:
        rs = [r for r in allrows if r["tile"] == tid]
        acc = {k: [0.0, 0] for k in ("model", "oracle", "fab")}
        sse = sst = 0.0
        for r in rs:
            h = r["h"]
            for k in acc:
                rho = crop_rho(r[k], h)
                if rho is not None:
                    acc[k][0] += rho * len(h); acc[k][1] += len(h)
            if len(h) >= MIN_CELLS and np.ptp(h) > 0:
                e = r["calib"] - h
                sse += float(((e - e.mean()) ** 2).sum()); sst += float(((h - h.mean()) ** 2).sum())
        H = np.concatenate([r["h"] for r in rs])
        rec = {f"within_crop_{k}": (acc[k][0] / acc[k][1] if acc[k][1] else None) for k in acc}
        rec["n_cells_scored"] = acc["model"][1]
        r2 = 1 - sse / sst if sst > 0 else None
        rec["calib_relief_r2"] = r2
        rec["calib_implied_r"] = float(np.sqrt(max(0.0, r2))) if r2 is not None else None
        for k in ("model", "oracle", "fab"):
            rec[f"whole_tile_{k}"] = float(spearmanr(np.concatenate([r[k] for r in rs]), H).statistic)
        per_tile[tid] = rec

    def vec(k):
        return np.array([per_tile[t][k] for t in tids if per_tile[t][k] is not None], float)

    def boot(v, B=10000):
        rng = np.random.default_rng(0)
        m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    S = {"reproduction": {"tile_folds": len(repro), "max_rel_diff_icesat_rmse": float(rmse_rel.max()),
                          "median_rel_diff_icesat_rmse": float(np.median(rmse_rel)),
                          "slope_sign_agreement": slope_sign_agree},
         "min_cells_per_crop": MIN_CELLS, "n_tiles": len(tids), "means": {}, "tests": {}, "per_tile": per_tile}
    for k in ("within_crop_model", "within_crop_oracle", "within_crop_fab", "calib_implied_r",
              "whole_tile_model", "whole_tile_oracle", "whole_tile_fab"):
        v = vec(k); S["means"][k] = {"mean": float(v.mean()), "ci95": boot(v), "n": len(v)}
    # pooled within-crop (all crops of all tiles, cell-weighted)
    for k in ("model", "oracle", "fab"):
        num = den = 0.0
        for r in allrows:
            rho = crop_rho(r[k], r["h"])
            if rho is not None:
                num += rho * len(r["h"]); den += len(r["h"])
        S["means"][f"pooled_within_crop_{k}"] = num / den
    ok = [t for t in tids if per_tile[t]["within_crop_model"] is not None and per_tile[t]["within_crop_oracle"] is not None]
    a = np.array([per_tile[t]["within_crop_model"] for t in ok]); b = np.array([per_tile[t]["within_crop_oracle"] for t in ok])
    g = np.array([per_tile[t]["calib_implied_r"] for t in ok])
    S["tests"]["model_vs_oracle"] = {"diff_mean": float((a - b).mean()), "ci95": boot(a - b), "model_higher": int((a > b).sum()),
                                     "n": len(ok), "wilcoxon_p": float(wilcoxon(a, b).pvalue)}
    S["tests"]["model_spearman_minus_calib_implied"] = {"diff_mean": float((a - g).mean()), "ci95": boot(a - g),
                                                        "spearman_higher": int((a > g).sum()), "n": len(ok),
                                                        "wilcoxon_p": float(wilcoxon(a, g).pvalue)}
    mm, mo = S["means"]["within_crop_model"]["mean"], S["means"]["within_crop_oracle"]["mean"]
    gap = S["tests"]["model_spearman_minus_calib_implied"]
    if abs(mm) < 0.10 and abs(mo) < 0.10:
        case = "1: near-zero raw ranking for both -> calibration was never the bottleneck; line stays closed"
    elif gap["diff_mean"] >= 0.10 and gap["spearman_higher"] > gap["n"] / 2 and gap["wilcoxon_p"] < 0.05:
        case = "2: raw ranking exceeds what linear calibration expresses -> isotonic calibration worth testing"
    else:
        case = "3: non-trivial raw ranking, but linear calibration already expresses it -> isotonic not indicated; line stays closed"
    S["case"] = case
    np.savez_compressed(OUT / "cells.npz", **{f"{i}_{k}": np.asarray(r[k]) for i, r in enumerate(allrows)
                                              for k in ("h", "model", "oracle", "fab", "calib")})
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k != "per_tile"}, indent=1))


if __name__ == "__main__":
    main()
