#!/usr/bin/env python3
"""
Part B (2026-09-23): tile-level bootstrap (10,000, seed 0, 95% percentile) of the
resolution curves. Arithmetic on saved per-tile records only.
Pooled Pearson / variance ratio are recomputed exactly from per-tile sufficient
statistics on each resample; Spearman is the mean of per-unit values.
Inputs:
  B1 RDAH: data/dfc2019/experiments/rdah_zeroshot/resolution_sweep_50/per_tile.json (depth == dav2_rerun)
  B2 Method 6: data/dfc2019/experiments/method6_resolution_sweep/per_quadrant_seed43.json
     (bootstrap unit = tile, each carrying its 4 held-out quadrants)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B, SEED = 10_000, 0
KEYS = ("n", "sp", "sy", "spp", "syy", "spy")


def pooled(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, sp, sy, spp, syy, spy = (S[..., i] for i in range(6))
    vp, vy = spp / n - (sp / n) ** 2, syy / n - (sy / n) ** 2
    cov = spy / n - (sp / n) * (sy / n)
    return cov / np.sqrt(vp * vy), vp / vy


def curve(recs: list[dict], unit: str) -> dict:
    out = {}
    rng = np.random.default_rng(SEED)
    for f in sorted({r["factor"] for r in recs}):
        rs = [r for r in recs if r["factor"] == f]
        units = sorted({r[unit] for r in rs})
        S = np.array([[sum(r["suff"][k] for r in rs if r[unit] == u) for k in KEYS] for u in units])
        sp = np.array([np.mean([r["spearman"] for r in rs if r[unit] == u]) for u in units])
        pe = np.array([np.mean([r["pearson"] for r in rs if r[unit] == u]) for u in units])
        idx = rng.integers(0, len(units), size=(B, len(units)))
        bp, bv = pooled(S[idx].sum(axis=1))
        p0, v0 = pooled(S.sum(axis=0))
        ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]
        out[str(f)] = {"n_units": len(units), "gsd_m": rs[0].get("gsd_m", 0.3 * f),
                       "pooled_pearson": float(p0), "pooled_pearson_ci95": ci(bp),
                       "pooled_variance_ratio": float(v0), "pooled_variance_ratio_ci95": ci(bv),
                       "mean_unit_pearson": float(pe.mean()), "mean_unit_pearson_ci95": ci(pe[idx].mean(axis=1)),
                       "mean_unit_spearman": float(sp.mean()), "mean_unit_spearman_ci95": ci(sp[idx].mean(axis=1))}
    return out


def main():
    res = {"bootstrap": {"B": B, "seed": SEED}}
    p1 = ROOT / "data/dfc2019/experiments/rdah_zeroshot/resolution_sweep_50/per_tile.json"
    if p1.exists():
        recs = [r for r in json.load(open(p1)) if r["depth"] == "dav2_rerun"]
        res["B1_rdah_swiss_50tiles"] = {"model": "RDAH-Net Swiss checkpoint, zero-shot", "unit": "tile", **curve(recs, "tile")}
    p2 = ROOT / "data/dfc2019/experiments/method6_resolution_sweep/per_quadrant_seed43.json"
    if p2.exists():
        res["B2_method6_seed43"] = {"model": "Method 6 height-balanced, seed 43, held-out quadrants", "unit": "tile (4 quadrants)",
                                    **curve(json.load(open(p2)), "tile")}
    out = ROOT / "data/dfc2019/experiments/resolution_curves_bootstrap.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res.items():
        if k == "bootstrap":
            continue
        print(k)
        for f in ("1", "2", "4", "8"):
            if f in v:
                r = v[f]
                print(f"  x{f} ({r['gsd_m']:.1f} m, n={r['n_units']}): pooled r {r['pooled_pearson']:.3f} {[round(x,3) for x in r['pooled_pearson_ci95']]}"
                      f"  VR {r['pooled_variance_ratio']:.3f} {[round(x,3) for x in r['pooled_variance_ratio_ci95']]}"
                      f"  mean rho {r['mean_unit_spearman']:.3f} {[round(x,3) for x in r['mean_unit_spearman_ci95']]}")


if __name__ == "__main__":
    main()
