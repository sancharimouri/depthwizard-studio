#!/usr/bin/env python3
"""Haze-mechanism check (2026-09-24; pre-registered in
docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md, "Haze-mechanism check").

Per hilly tile: Spearman of blue reflectance vs FABDEM elevation over 5,000 elevation-stratified pixels
(10 deciles x 500, seed 0), for L1C blue (primary), L1C - L2A blue (haze component) and L2A blue (surface control);
plus the within-crop (60x60 px) cell-weighted mean of the same; plus a bright-pixel-excluded sensitivity run.
No training, no ICESat-2.   .venv/bin/python scripts/haze_elevation_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
L1C = ROOT / "data/sentinel2_benchmark/l1c_test"
CACHE = ROOT / "data/sentinel2_benchmark/token_grid_test/cache"
HILLY = ("almora", "dehradun", "dharamshala", "kohima")
BRIGHT = 0.30


def measures(elev, l1c, l2a, mask):
    rng = np.random.default_rng(0)
    e = elev[mask]; idx_all = np.flatnonzero(mask.ravel())
    edges = np.quantile(e, np.linspace(0, 1, 11))
    pick = []
    for k in range(10):
        inbin = idx_all[(e >= edges[k]) & ((e < edges[k + 1]) if k < 9 else (e <= edges[k + 1]))]
        pick.append(rng.choice(inbin, min(500, len(inbin)), replace=False))
    pick = np.concatenate(pick)
    E, B1, B2 = elev.ravel()[pick], l1c.ravel()[pick], l2a.ravel()[pick]
    out = {"n": int(len(pick)), "elev_range_m": [float(E.min()), float(E.max())],
           "rho_l1c_blue": float(spearmanr(B1, E).statistic),
           "rho_haze_component": float(spearmanr(B1 - B2, E).statistic),
           "rho_l2a_blue": float(spearmanr(B2, E).statistic)}
    dec = np.digitize(E, edges[1:-1])
    out["haze_offset_by_decile"] = [float(np.median((B1 - B2)[dec == k])) for k in range(10)]
    out["l1c_blue_by_decile"] = [float(np.median(B1[dec == k])) for k in range(10)]
    acc = {k: [0.0, 0] for k in ("l1c", "haze", "l2a")}
    for r in range(0, 960, 60):
        for c in range(0, 960, 60):
            m = mask[r:r + 60, c:c + 60]
            if m.sum() < 100:
                continue
            e = elev[r:r + 60, c:c + 60][m]
            if np.ptp(e) == 0:
                continue
            b1, b2 = l1c[r:r + 60, c:c + 60][m], l2a[r:r + 60, c:c + 60][m]
            for k, v in (("l1c", b1), ("haze", b1 - b2), ("l2a", b2)):
                if np.ptp(v) > 0:
                    acc[k][0] += spearmanr(v, e).statistic * len(e); acc[k][1] += len(e)
    out.update({f"within_crop_rho_{k}": acc[k][0] / acc[k][1] for k in acc})
    return out


def main():
    res = {}
    for t in HILLY:
        raw = np.load(L1C / "raw" / f"{t}.npz")
        l1c, l2a = raw["l1c_float"][2].astype(float), raw["l2a_float"][2].astype(float)
        elev = np.load(CACHE / f"{t}.npz")["fab10"].astype(float)
        ok = np.isfinite(elev) & np.isfinite(l1c) & np.isfinite(l2a) & (l1c > 0)
        res[t] = {"all": measures(elev, l1c, l2a, ok),
                  "bright_excluded": measures(elev, l1c, l2a, ok & (l1c <= BRIGHT)),
                  "bright_frac": float((l1c[ok] > BRIGHT).mean())}
        a, b = res[t]["all"], res[t]["bright_excluded"]
        print(f"{t:12s} elev {a['elev_range_m'][0]:.0f}-{a['elev_range_m'][1]:.0f} m | rho L1C blue {a['rho_l1c_blue']:+.3f} "
              f"(bright-excl {b['rho_l1c_blue']:+.3f}) | haze comp {a['rho_haze_component']:+.3f} | L2A blue {a['rho_l2a_blue']:+.3f} | "
              f"within-crop L1C {a['within_crop_rho_l1c']:+.3f} haze {a['within_crop_rho_haze']:+.3f} L2A {a['within_crop_rho_l2a']:+.3f} "
              f"| bright {res[t]['bright_frac']:.3f}", flush=True)
    n_pass = sum(res[t]["all"]["rho_l1c_blue"] <= -0.30 for t in HILLY)
    res["decision"] = {"tiles_rho_le_-0.30": n_pass,
                       "haze_mechanism_supported": bool(n_pass >= 3),
                       "sensitivity_tiles_rho_le_-0.30": sum(res[t]["bright_excluded"]["rho_l1c_blue"] <= -0.30 for t in HILLY)}
    (L1C / "haze_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"]))


if __name__ == "__main__":
    main()
