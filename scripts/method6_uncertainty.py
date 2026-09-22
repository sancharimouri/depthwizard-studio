#!/usr/bin/env python3
"""
C1-C3 (2026-09-23): Method 6 (height-balanced recipe) uncertainty vs the oracle
per-tile-OLS baseline, plus seed aggregation. Arithmetic on saved JSONs only.

Sources:
  seed 1 (original): data/dfc2019/experiments/method6_height_balanced/m6_heightbal_results.json
  seeds 43/44 (C3):   data/dfc2019/experiments/method6_height_balanced_seed{43,44}/m6_heightbal_seed{S}_results.json
  oracle per-tile-OLS: data/dfc2019/experiments/semantic/method3_spatial_cv_results.json ("baseline";
    per tile, fit on 3 quadrants of that tile, evaluated on the held-out quadrant -- 200 evaluations)
Pairing: Method 6 fold q == held-out quadrant q; quadrant_bounds order 0..3 =
top_left, top_right, bottom_left, bottom_right (evaluate_method4.quadrant_bounds); pixel counts
match between the two files (e.g. JAX_004_006 top_left 260,076).
Aggregation of every headline number: mean over the 200 (tile, quadrant) evaluations
== Method 6's agg() mean-of-folds (each fold has all 50 tiles).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/dfc2019/experiments"
QNAME = {0: "top_left", 1: "top_right", 2: "bottom_left", 3: "bottom_right"}
METRICS = {"mae_m": "lower", "rmse_m": "lower", "pearson": "higher", "spearman": "higher"}
B, SEED = 10_000, 0
M2_GRID_HUBER_20 = {"mae_m": 2.929, "rmse_m": 4.718, "pearson": 0.532, "spearman": 0.471}


def m6_table(path: Path) -> pd.DataFrame:
    d = json.load(open(path))
    rows = []
    for f in d["folds"]:
        for t in f["tiles"]:
            rows.append({"tile": t["tile"], "quadrant": QNAME[f["held_out_quadrant"]], **{m: t[m] for m in METRICS},
                         "n_pixels": t["n_pixels"]})
    return pd.DataFrame(rows), d


def oracle_table() -> pd.DataFrame:
    d = json.load(open(EXP / "semantic/method3_spatial_cv_results.json"))
    return pd.DataFrame([{"tile": t["tile"], "quadrant": f["fold"], **{m: f["baseline"][m] for m in METRICS},
                          "n_pixels": f["baseline"]["n_pixels"]} for t in d["tiles"] for f in t["folds"]])


def boot_mean_by_tile(df: pd.DataFrame, col: str) -> list[float]:
    per_tile = df.groupby("tile")[col].mean().values  # each tile has all 4 quadrants -> equal weights
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_tile), size=(B, len(per_tile)))
    return [float(x) for x in np.percentile(per_tile[idx].mean(axis=1), [2.5, 97.5])]


def compare(m6: pd.DataFrame, orc: pd.DataFrame) -> dict:
    j = m6.merge(orc, on=["tile", "quadrant"], suffixes=("_m6", "_or"))
    assert len(j) == 200, len(j)
    assert (j["n_pixels_m6"] == j["n_pixels_or"]).mean() > 0.95, "pixel counts don't match -- pairing suspect"
    out = {"pixel_count_match_frac": float((j["n_pixels_m6"] == j["n_pixels_or"]).mean())}
    for m, better in METRICS.items():
        j[f"d_{m}"] = j[f"{m}_m6"] - j[f"{m}_or"]
        per_tile = j.groupby("tile")[[f"{m}_m6", f"{m}_or"]].mean()
        dt = per_tile[f"{m}_m6"] - per_tile[f"{m}_or"]
        win = (dt < 0) if better == "lower" else (dt > 0)
        winq = (j[f"d_{m}"] < 0) if better == "lower" else (j[f"d_{m}"] > 0)
        out[m] = {"m6_mean": float(j[f"{m}_m6"].mean()), "m6_ci95": boot_mean_by_tile(j, f"{m}_m6"),
                  "oracle_mean": float(j[f"{m}_or"].mean()), "oracle_ci95": boot_mean_by_tile(j, f"{m}_or"),
                  "diff_mean": float(j[f"d_{m}"].mean()), "diff_ci95": boot_mean_by_tile(j, f"d_{m}"),
                  "tiles_m6_better": int(win.sum()), "n_tiles": int(len(dt)),
                  "quadrants_m6_better": int(winq.sum()), "n_quadrants": int(len(j)),
                  "wilcoxon_p_tiles": float(stats.wilcoxon(per_tile[f"{m}_m6"], per_tile[f"{m}_or"]).pvalue)}
    return out


def main():
    orc = oracle_table()
    runs = {"seed1_original": EXP / "method6_height_balanced/m6_heightbal_results.json"}
    for s in (43, 44):
        p = EXP / f"method6_height_balanced_seed{s}/m6_heightbal_seed{s}_results.json"
        if p.exists() and "overall" in json.load(open(p)):
            runs[f"seed{s}"] = p
    res = {"aggregation": "mean over 200 (tile, held-out quadrant) evaluations == mean of 4 fold means",
           "bootstrap": {"unit": "tile (with its 4 quadrants)", "B": B, "seed": SEED},
           "oracle_source": "data/dfc2019/experiments/semantic/method3_spatial_cv_results.json (baseline)",
           "runs": {}}
    for name, p in runs.items():
        m6, raw = m6_table(p)
        c = compare(m6, orc)
        c["beats_oracle_all4"] = all(
            (c[m]["m6_mean"] < c[m]["oracle_mean"]) if b == "lower" else (c[m]["m6_mean"] > c[m]["oracle_mean"])
            for m, b in METRICS.items())
        c["beats_m2_grid_huber20_all4"] = all(
            (c[m]["m6_mean"] < M2_GRID_HUBER_20[m]) if b == "lower" else (c[m]["m6_mean"] > M2_GRID_HUBER_20[m])
            for m, b in METRICS.items())
        c["overall_from_file"] = raw.get("overall")
        c["fold_diagnostics"] = [f.get("diagnostics") for f in raw["folds"]]
        res["runs"][name] = c
    names = list(res["runs"])
    res["across_seeds"] = {m: {"mean": float(np.mean([res["runs"][n][m]["m6_mean"] for n in names])),
                               "sd": float(np.std([res["runs"][n][m]["m6_mean"] for n in names], ddof=1)) if len(names) > 1 else None,
                               "n_seeds": len(names)} for m in METRICS}
    res["headline_stands"] = all(res["runs"][n]["beats_oracle_all4"] for n in names)
    out = EXP / "method6_uncertainty.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    for n in names:
        r = res["runs"][n]
        print(n, "beats oracle all 4:", r["beats_oracle_all4"],
              {m: (round(r[m]["m6_mean"], 4), [round(x, 3) for x in r[m]["m6_ci95"]], f"{r[m]['tiles_m6_better']}/50",
                   f"{r[m]['quadrants_m6_better']}/200") for m in METRICS})
    print("oracle", {m: (round(res["runs"][names[0]][m]["oracle_mean"], 4), [round(x, 3) for x in res["runs"][names[0]][m]["oracle_ci95"]]) for m in METRICS})
    print("across seeds", res["across_seeds"], "headline stands:", res["headline_stands"])


if __name__ == "__main__":
    main()
