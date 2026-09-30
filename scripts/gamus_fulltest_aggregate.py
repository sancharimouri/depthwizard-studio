#!/usr/bin/env python3
"""Applies the §7.8 pre-registered rule (docs/method6-checkpoint-audit.md) to build/gamus_fulltest/tiles.jsonl.

Aggregation is Part B's (scripts/gamus_zeroshot_aggregate.py: tile value = mean of its quadrant metrics, headline =
mean of tiles, tile bootstrap 10,000 / seed 0 / 95%). F = mean of the four seed-42 fold models' metrics, each
computed individually. PASS iff pooled MAE_full <= 1.05 MAE_F, RMSE_full <= 1.05 RMSE_F, Pearson_full >=
Pearson_F - 0.02, AND all three hold in >= 2 of the 3 cities.
  .venv/bin/python scripts/gamus_fulltest_aggregate.py -> build/gamus_fulltest/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gamus_zeroshot_aggregate import boot_ci, tile_table  # noqa: E402  (Part B's own aggregation)

SRC = ROOT / "build/gamus_fulltest/tiles.jsonl"
OUT = SRC.with_name("summary.json")
FOLDS = ["f0", "f1", "f2", "f3"]
PREDS = ["full"] + FOLDS + ["ens", "partB_oracle", "partB_m6_s42"]
METRICS = ("mae", "rmse", "pearson", "spearman", "var_ratio")
RULE_MET = ("mae", "rmse", "pearson")


def pooled_vr(recs, pred):
    M = np.zeros(5)
    for r in recs:
        if pred in r.get("pooled_moments", {}):
            M += np.array(r["pooled_moments"][pred])
    n, sp, sp2, sy, sy2 = M
    return float((sp2 / n - (sp / n) ** 2) / (sy2 / n - (sy / n) ** 2)) if n else None


def rule_check(met_full, met_F):
    c = {"mae": met_full["mae"] <= 1.05 * met_F["mae"], "rmse": met_full["rmse"] <= 1.05 * met_F["rmse"],
         "pearson": met_full["pearson"] >= met_F["pearson"] - 0.02}
    return c, all(c.values())


def summarise(R):
    T = {p: {m: tile_table(R, p, m) for m in METRICS} for p in PREDS}
    out = {"n_tiles": len(R), "metrics": {}}
    for p in PREDS:
        out["metrics"][p] = {m: {"mean": float(np.nanmean(T[p][m])), "ci95": boot_ci(T[p][m])} for m in METRICS
                             if np.isfinite(T[p][m]).any()}
        vr = pooled_vr(R, p)
        if vr is not None:
            out["metrics"][p]["var_ratio_pooled"] = vr
    # F: mean of the four folds' metrics, each fold scored individually (mean of tiles)
    F = {m: float(np.mean([out["metrics"][f][m]["mean"] for f in FOLDS])) for m in METRICS}
    F["var_ratio_pooled"] = float(np.mean([out["metrics"][f]["var_ratio_pooled"] for f in FOLDS]))
    out["F"] = F
    full = {m: out["metrics"]["full"][m]["mean"] for m in METRICS}
    out["rule_checks"], out["meets_all_three"] = rule_check(full, F)
    out["thresholds"] = {"mae_max": 1.05 * F["mae"], "rmse_max": 1.05 * F["rmse"], "pearson_min": F["pearson"] - 0.02}
    # information only: paired per-tile differences full - F_tile, Wilcoxon
    paired = {}
    for m in METRICS:
        a = T["full"][m]
        b = np.nanmean(np.vstack([T[f][m] for f in FOLDS]), axis=0)
        ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]
        paired[m] = {"mean_diff_full_minus_F": float(np.mean(d)), "median_diff": float(np.median(d)),
                     "ci95": boot_ci(d), "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                     "full_better_tiles": int(np.sum(d < 0) if m in ("mae", "rmse") else np.sum(d > 0)),
                     "n": int(ok.sum())}
    out["paired_full_minus_F"] = paired
    return out


def main():
    all_recs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    recs = [r for r in all_recs if r.get("quads")]
    S = {"source": str(SRC.relative_to(ROOT)), "n_tiles_total": len(all_recs), "n_tiles_scored": len(recs),
         "tiles_sharing_dfc_blocks": sum(1 for r in all_recs if r.get("shared_dfc_blocks", 0) > 0),
         "aggregation": "Part B: tile = mean of its quadrant metrics; headline = mean of tiles; bootstrap 10,000, seed 0",
         "subsets": {}}
    # parity with Part B: the recomputed seed-42 ensemble vs Part B's saved m6_s42, per quadrant
    diffs = [abs(q["m"]["ens"][k] - q["m"]["partB_m6_s42"][k]) for r in recs for q in r["quads"] if "partB_m6_s42" in q["m"]
             for k in ("mae", "rmse", "pearson", "spearman") if q["m"]["ens"].get(k) is not None and q["m"]["partB_m6_s42"].get(k) is not None]
    S["parity_ens_vs_partB_m6_s42"] = {"max_abs_diff": float(max(diffs)) if diffs else None, "n_values": len(diffs)}
    subsets = {"pooled": recs}
    for city in sorted({r["city"] for r in recs}):
        subsets[city] = [r for r in recs if r["city"] == city]
    for name, R in subsets.items():
        S["subsets"][name] = summarise(R)
    cities = [k for k in S["subsets"] if k != "pooled"]
    city_pass = {c: S["subsets"][c]["meets_all_three"] for c in cities}
    pooled_ok = S["subsets"]["pooled"]["meets_all_three"]
    S["rule"] = {"pooled_meets_all_three": pooled_ok, "city_meets_all_three": city_pass,
                 "n_cities_meeting": sum(city_pass.values()), "verdict": "PASS" if pooled_ok and sum(city_pass.values()) >= 2 else "FAIL"}
    OUT.write_text(json.dumps(S, indent=1))
    P = S["subsets"]["pooled"]
    print(f"tiles scored {S['n_tiles_scored']} / {S['n_tiles_total']}; parity {S['parity_ens_vs_partB_m6_s42']}")
    for name, sub in S["subsets"].items():
        f, F = sub["metrics"]["full"], sub["F"]
        print(f"{name:7s} n={sub['n_tiles']:4d}  full MAE {f['mae']['mean']:.3f} RMSE {f['rmse']['mean']:.3f} r {f['pearson']['mean']:.3f}"
              f" | F MAE {F['mae']:.3f} RMSE {F['rmse']:.3f} r {F['pearson']:.3f} | checks {sub['rule_checks']}")
    print("RULE:", S["rule"])
    print("pooled paired (full - F):", {m: (round(v['mean_diff_full_minus_F'], 4), f"p={v['wilcoxon_p']:.2g}") for m, v in P["paired_full_minus_F"].items()})


if __name__ == "__main__":
    main()
