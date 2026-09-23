#!/usr/bin/env python3
"""Part B aggregation (2026-09-23): applies the pre-registered rule to data/gamus_eval/zeroshot_tiles.jsonl.
Tile value = mean of the tile's quadrant metrics; headline = mean of tiles. Tile-level bootstrap
(10,000, seed 0, 95% percentile), paired Wilcoxon + win counts, Holm across the 3 seed candidates.
  .venv/bin/python scripts/gamus_zeroshot_aggregate.py [jsonl] -> data/gamus_eval/zeroshot_summary.json"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/gamus_eval/zeroshot_tiles.jsonl"
OUT = SRC.with_name(SRC.stem.replace("_tiles", "") + "_summary.json")
MET = ("mae", "rmse", "pearson", "spearman")
LOWER_BETTER = {"mae": True, "rmse": True, "pearson": False, "spearman": False}
CAND = ["m6_s42", "m6_s43", "m6_s44"]
SPARSE = ("ground", "low_veg")


def load():
    recs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    return [r for r in recs if "quads" in r and r["quads"]], recs


def tile_table(recs, pred, metric):
    out = []
    for r in recs:
        v = [q["m"][pred].get(metric) for q in r["quads"] if pred in q["m"]]
        v = [x for x in v if x is not None]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out, float)


def boot_ci(x, n=10000, seed=0):
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def pooled_from_sums(recs, pred, key, labels):
    agg = {}
    for r in recs:
        for q in r["quads"]:
            for lab, s in q[key].get(pred, {}).items():
                a = agg.setdefault(lab, np.zeros(6))
                a += s
    out = {}
    for lab in labels:
        if lab in agg and agg[lab][0] > 0:
            n, sa, s2, se, sp, sy = agg[lab]
            out[lab] = {"n_px": int(n), "mae": sa / n, "rmse": float(np.sqrt(s2 / n)), "bias": se / n,
                        "mean_pred": sp / n, "mean_true": sy / n}
    return out


def class_tile_mae(recs, pred, classes):
    """mean-of-tiles per-class MAE over tiles with >= 100 px of the class group."""
    v = []
    for r in recs:
        n = sa = 0.0
        for q in r["quads"]:
            for c in classes:
                s = q["cls"].get(pred, {}).get(c)
                if s:
                    n += s[0]; sa += s[1]
        if n >= 100:
            v.append(sa / n)
    return {"mean_of_tiles_mae": float(np.mean(v)) if v else None, "n_tiles": len(v)}


def main():
    recs, allrecs = load()
    cands = [c for c in CAND if c in recs[0]["quads"][0]["m"]]
    preds = cands + ["oracle"] + [c + "_ols" for c in cands]
    overlapping = [r["tile"] for r in allrecs if r.get("shared_dfc_blocks", 0) > 0]
    S = {"source": str(SRC.relative_to(ROOT)) if SRC.is_relative_to(ROOT) else str(SRC), "n_tiles_total": len(allrecs), "n_tiles_scored": len(recs),
         "n_skipped": len(allrecs) - len(recs), "overlapping_tiles": overlapping,
         "aggregation": "per tile: mean of its 4 quadrant metrics; headline: mean of tiles",
         "valid_frac_mean": float(np.mean([r["valid_frac"] for r in allrecs])),
         "neg_frac_mean": float(np.mean([r["neg_frac"] for r in allrecs])), "subsets": {}}
    subsets = {"full": recs, "non_overlapping": [r for r in recs if r["tile"] not in overlapping]}
    for city in sorted({r["city"] for r in recs}):
        subsets[f"city_{city}"] = [r for r in recs if r["city"] == city]
    for name, R in subsets.items():
        sub = {"n_tiles": len(R), "metrics": {}}
        T = {p: {m: tile_table(R, p, m) for m in MET + ("var_ratio", "bias")} for p in preds}
        T["dav2_raw"] = {m: tile_table(R, "dav2_raw", m) for m in ("pearson", "spearman")}
        for p, d in T.items():
            sub["metrics"][p] = {m: {"mean": float(np.nanmean(v)), "ci95": boot_ci(v) if name in ("full", "non_overlapping") else None}
                                 for m, v in d.items()}
        # pooled variance ratio from moments
        for p in preds:
            M = np.zeros(5)
            for r in R:
                M += np.array(r["pooled_moments"][p])
            n, sp, sp2, sy, sy2 = M
            sub["metrics"][p]["var_ratio_pooled"] = float((sp2 / n - (sp / n) ** 2) / (sy2 / n - (sy / n) ** 2))
        # rule + paired tests vs oracle
        sub["vs_oracle"] = {}
        ps = {m: {} for m in MET}
        for c in cands:
            e = {}
            for m in MET:
                a, b = T[c][m], T["oracle"][m]
                ok = np.isfinite(a) & np.isfinite(b)
                d = (a - b)[ok] if LOWER_BETTER[m] else (b - a)[ok]  # negative = candidate better
                better = bool(np.mean(d) < 0)
                w = wilcoxon(a[ok], b[ok]) if ok.sum() > 10 else None
                ps[m][c] = float(w.pvalue) if w else np.nan
                ca, cb = sub["metrics"][c][m]["ci95"], sub["metrics"]["oracle"][m]["ci95"]
                nonover = None
                if ca and cb:
                    nonover = (ca[1] < cb[0]) if LOWER_BETTER[m] else (ca[0] > cb[1])
                e[m] = {"better_mean": better, "wins": int((d < 0).sum()), "n": int(ok.sum()),
                        "diff_mean_cand_minus_oracle": float(np.mean((a - b)[ok])),
                        "diff_ci95_cand_minus_oracle": boot_ci((a - b)[ok]) if name in ("full", "non_overlapping") else None,
                        "wilcoxon_p": ps[m][c], "ci_nonoverlap": nonover}
            sub["vs_oracle"][c] = e
        for m in MET:
            h = holm({k: v for k, v in ps[m].items() if np.isfinite(v)})
            for c in cands:
                sub["vs_oracle"][c][m]["wilcoxon_p_holm"] = h.get(c)
        # breakdowns (full + cities)
        sub["per_class_pooled"] = {p: pooled_from_sums(R, p, "cls", ["ground", "low_veg", "building", "water", "road", "tree", "unlabelled"])
                                   for p in cands + ["oracle"]}
        sub["landscape_mae"] = {}
        for p in cands + ["oracle"]:
            L = {"urban(building)": class_tile_mae(R, p, ("building",)), "sparse(ground+low_veg)": class_tile_mae(R, p, SPARSE),
                 "forested(tree)": class_tile_mae(R, p, ("tree",))}
            vals = [v["mean_of_tiles_mae"] for v in L.values() if v["mean_of_tiles_mae"] is not None]
            L["spread_max_minus_min"] = float(max(vals) - min(vals)) if vals else None
            sub["landscape_mae"][p] = L
        sub["height_bins_pooled"] = {p: pooled_from_sums(R, p, "bin", ["0-2", "2-5", "5-10", "10-20", "20-30", "30-50", "50-inf"])
                                     for p in cands + ["oracle"]}
        sub["tree_height_bins_pooled"] = {p: pooled_from_sums(R, p, "tree_bin", ["0-2", "2-5", "5-10", "10-20", "20-30", "30-50", "50-inf"])
                                          for p in cands}
        S["subsets"][name] = sub
    # decision rule
    rule = {}
    base = S["subsets"]["non_overlapping"]
    cities = [k for k in S["subsets"] if k.startswith("city_")]
    for c in cands:
        crit1 = all(base["vs_oracle"][c][m]["better_mean"] and base["vs_oracle"][c][m]["ci_nonoverlap"] for m in MET)
        city_wins = {k: all(S["subsets"][k]["vs_oracle"][c][m]["better_mean"] for m in MET) for k in cities}
        rule[c] = {"all4_better_ci_nonoverlap": crit1, "city_wins": city_wins,
                   "n_city_wins": sum(city_wins.values()), "n_cities": len(cities),
                   "pass": bool(crit1 and all(city_wins.values()))}
    S["rule"] = rule
    S["verdict"] = "GENERALIZES" if all(v["pass"] for v in rule.values()) and len(rule) == 3 else \
        ("DOES NOT GENERALIZE" if len(rule) == 3 else "INCOMPLETE (fewer than 3 seeds)")
    OUT.write_text(json.dumps(S, indent=1, default=float))
    print("wrote", OUT)
    for name in ["full"] + cities:
        sub = S["subsets"][name]
        print(f"\n== {name} (n={sub['n_tiles']})")
        for p in cands + ["oracle", cands[0] + "_ols", "dav2_raw"]:
            mm = sub["metrics"][p]
            print(f"  {p:12s}", {k: round(v["mean"], 3) for k, v in mm.items() if isinstance(v, dict)},
                  "vr_pooled", round(mm.get("var_ratio_pooled", np.nan), 3))
    print("\nrule", json.dumps(rule, indent=0)); print("VERDICT", S["verdict"])


if __name__ == "__main__":
    main()
