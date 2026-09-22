#!/usr/bin/env python3
"""
Aggregate RDAH-FT-2's four per-fold results into one JSON. Arithmetic only --
no inference, no retraining.

Source of truth: data/dfc2019/experiments/rdah_quadrant_cv/fold{0..3}/result.json,
written by scripts/train_rdah_quadrant_cv.py. The top-level
method_rdah_quadrant_cv_results.json is NOT a 4-fold aggregate (the training
script overwrites it with whichever --fold ran last); this script records which
folds it actually contains and whether they match the per-fold files.

Reported numbers are each fold's nested-selected epoch evaluated on that
fold's "true test" half (25 of the 50 held-out-quadrant samples). The other 25
held-out-quadrant samples were used for epoch selection -- see
"selection_protocol" in the output.

Aggregations (each labelled in the output):
- pixel_weighted: MAE = sum(n_k*MAE_k)/sum(n_k); RMSE = sqrt(sum(n_k*RMSE_k^2)/sum(n_k)).
  Exact pooled MAE/RMSE over every true-test pixel of all 4 folds.
- mean_of_folds: unweighted mean of the 4 per-fold (pixel-pooled-within-fold) metrics.
- n_weighted_mean_of_folds (correlations only): pixel-count-weighted mean of the
  per-fold Pearson/Spearman. NOT a true pooled correlation -- the JSONs do not
  store per-fold means/variances/covariance or predictions, so a true pooled
  Pearson/Spearman cannot be computed from them.
- mean_of_samples_then_folds: mean of per-(tile,quadrant) metrics within each
  fold, then mean over folds. This is the aggregation Method 6 uses
  (evaluate_method6_finetune_twinhead.py agg(): per-tile mean, then fold mean),
  so it is the like-for-like number for a Method 6 comparison.

Usage:
    python scripts/aggregate_rdah_ft2.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "rdah_quadrant_cv"
OUT_PATH = EXP_DIR / "rdah_ft2_aggregate.json"
METRICS = ("mae", "rmse", "pearson", "spearman")


def rel(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT))


def pixel_weighted(rows: list[dict]) -> dict:
    n = np.array([r["n"] for r in rows], dtype=np.float64)
    mae = np.array([r["mae"] for r in rows])
    rmse = np.array([r["rmse"] for r in rows])
    return {
        "mae": float((n * mae).sum() / n.sum()),
        "rmse": float(math.sqrt((n * rmse ** 2).sum() / n.sum())),
        "n_pixels": int(n.sum()),
    }


def mean_of_folds(rows: list[dict]) -> dict:
    return {m: float(np.mean([r[m] for r in rows])) for m in METRICS}


def n_weighted_corr(rows: list[dict]) -> dict:
    n = np.array([r["n"] for r in rows], dtype=np.float64)
    return {m: float((n * np.array([r[m] for r in rows])).sum() / n.sum())
            for m in ("pearson", "spearman")}


def ranges(rows: list[dict], keys) -> dict:
    return {k: {"min": float(min(r[k] for r in rows)), "max": float(max(r[k] for r in rows))}
            for k in keys}


def sample_mean(per_sample: dict) -> dict:
    out = {}
    for m in METRICS:
        vals = np.array([s[m] for s in per_sample.values()], dtype=np.float64)
        out[m] = float(np.nanmean(vals))
        out[f"{m}_n_nan"] = int(np.isnan(vals).sum())
    out["n_samples"] = len(per_sample)
    return out


def fold_row(ns_block: dict) -> dict:
    return {m: ns_block[m] for m in (*METRICS, "n")}


def ft1_block() -> dict:
    """RDAH-FT-1 re-expressed under the same aggregations, for the comparison
    table. Read-only: its per-fold result.json files and the pooled report
    written by scripts/evaluate_rdah_pooled_cv.py."""
    ft1_dir = PROJECT_ROOT / "data" / "dfc2019" / "experiments" / "rdah"
    rows = []
    for k in range(4):
        fin = json.load(open(ft1_dir / f"fold{k}" / "result.json"))["final"]
        rows.append({"fold": k, "epoch": fin["epoch"], "mae": fin["test_mae"], "rmse": fin["test_rmse"],
                     "pearson": fin["test_pearson"], "spearman": fin["test_spearman"], "n": fin["test_n"]})
    report_path = ft1_dir / "pooled_cv_epoch5_report.json"
    report = json.load(open(report_path))
    by_fold: dict[int, list[dict]] = {}
    for t in report["per_tile"].values():
        by_fold.setdefault(t["fold"], []).append(t)
    tile_fold_means = [{m: float(np.mean([t[m] for t in by_fold[k]])) for m in ("mae", "rmse", "pearson")}
                       for k in sorted(by_fold)]
    return {
        "sources": [rel(ft1_dir / f"fold{k}" / "result.json") for k in range(4)] + [rel(report_path)],
        "protocol": "tile-level 4-fold spatial CV (17/8/7/18 tiles), Track1 checkpoint, unscaled [0,1] depth, epoch 5 fixed",
        "per_fold_epoch5_pooled_within_fold": rows,
        "reported_pooled": report["pooled_metrics"],
        "pixel_weighted": pixel_weighted(rows),
        "mean_of_folds": mean_of_folds(rows),
        "mean_of_tiles_then_folds": {**{m: float(np.mean([r[m] for r in tile_fold_means]))
                                        for m in ("mae", "rmse", "pearson")},
                                     "spearman": "not available per tile"},
        "fold_range": ranges(rows, METRICS),
    }


def main() -> None:
    folds, sources = [], {}
    for path in sorted(EXP_DIR.glob("fold*/result.json")):
        d = json.load(open(path))
        k = d["fold"]  # fold index from contents, not the directory name
        if path.parent.name != f"fold{k}":
            raise ValueError(f"{path}: directory name disagrees with contents fold={k}")
        sources[k] = rel(path)
        ns = d["nested_selection"]
        sel = ns["selected_true_test"]
        folds.append({
            "fold": k,
            "held_out_quadrant": k,
            "train_quadrants": d["train_quadrants"],
            "checkpoint_init": Path(d["checkpoint_init"]).name,
            "input_depth_scale": d["scale"],
            "selected_epoch": ns["selected_epoch"],
            "selection_criterion": "min MAE on inner-validation half of held-out quadrant",
            "inner_val_n_samples": ns["inner_val_n"],
            "true_test_n_samples": ns["true_test_n"],
            "mae": sel["mae"], "rmse": sel["rmse"],
            "pearson": sel["pearson"], "spearman": sel["spearman"],
            "n": sel["n"],
            "variance_ratio": sel["variance_ratio"],
            "ols_slope": sel["ols_slope"],
            "per_sample_mean": sample_mean(sel["per_sample"]),
            "oracle_epoch": ns["oracle_epoch"],
            "oracle_true_test": fold_row(ns["oracle_true_test"]),
            "default_epoch5_true_test": fold_row(ns["default_epoch5_true_test"]),
        })
    folds.sort(key=lambda r: r["fold"])
    if [r["fold"] for r in folds] != [0, 1, 2, 3]:
        raise ValueError(f"expected folds 0-3, found {[r['fold'] for r in folds]}")

    # What the top-level "results" file actually contains
    top_path = EXP_DIR / "method_rdah_quadrant_cv_results.json"
    top = json.load(open(top_path))
    top_folds = [f["fold"] for f in top["folds"]]
    top_matches = {}
    for f in top["folds"]:
        per_fold = json.load(open(EXP_DIR / f"fold{f['fold']}" / "result.json"))
        top_matches[str(f["fold"])] = (f == per_fold)

    sample_rows = [{**r["per_sample_mean"], "n": r["n"]} for r in folds]
    out = {
        "experiment": "RDAH-FT-2",
        "generated_by": "scripts/aggregate_rdah_ft2.py",
        "sources": {str(k): v for k, v in sorted(sources.items())},
        "top_level_results_file": {
            "path": rel(top_path),
            "folds_contained": top_folds,
            "identical_to_per_fold_file": top_matches,
            "note": "Overwritten by whichever --fold ran last; not a 4-fold aggregate.",
        },
        "selection_protocol": {
            "reported_epoch": "nested-selected epoch per fold",
            "selection_data": ("inner-validation half (25 tiles) of the fold's HELD-OUT quadrant; "
                               "metrics reported on the disjoint other half (25 tiles). "
                               "Not inner-train data: selection never sees the reported tiles, "
                               "but it does see the held-out quadrant."),
            "scale_selection_data": "fold's training quadrants only, 4 original probe tiles excluded",
            "source": "scripts/train_rdah_quadrant_cv.py run_fold() and derive_fold_scale()",
        },
        "per_fold": folds,
        "aggregate": {
            "pixel_weighted": pixel_weighted(folds),
            "mean_of_folds": mean_of_folds(folds),
            "n_weighted_mean_of_folds": n_weighted_corr(folds),
            "mean_of_samples_then_folds": mean_of_folds(sample_rows),
            "variance_ratio_mean_of_folds": float(np.mean([r["variance_ratio"] for r in folds])),
            "ols_slope_mean_of_folds": float(np.mean([r["ols_slope"] for r in folds])),
            "true_pooled_correlation": ("not computable: JSONs hold no per-fold means/variances/"
                                        "covariance or predictions"),
        },
        "fold_range": ranges(folds, (*METRICS, "variance_ratio", "ols_slope")),
        "leakage_check": {
            "oracle_epoch_pixel_weighted": pixel_weighted([r["oracle_true_test"] for r in folds]),
            "oracle_epoch_mean_of_folds": mean_of_folds([r["oracle_true_test"] for r in folds]),
            "epoch5_pixel_weighted": pixel_weighted([r["default_epoch5_true_test"] for r in folds]),
            "epoch5_mean_of_folds": mean_of_folds([r["default_epoch5_true_test"] for r in folds]),
        },
        "per_height_bin_error": "not present in the result JSONs",
        "worst_rmse_samples_per_fold": {
            str(r["fold"]): [
                {"sample": s, **{m: v[m] for m in ("mae", "rmse", "pearson")}}
                for s, v in sorted(json.load(open(EXP_DIR / f"fold{r['fold']}" / "result.json"))
                                   ["nested_selection"]["selected_true_test"]["per_sample"].items(),
                                   key=lambda kv: -kv[1]["rmse"])[:3]
            ] for r in folds
        },
        "ft1_comparison": ft1_block(),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {rel(OUT_PATH)}")
    a = out["aggregate"]
    print(json.dumps({"aggregate": a, "fold_range": out["fold_range"],
                      "top_level": out["top_level_results_file"],
                      "leakage_check": out["leakage_check"]}, indent=1))
    for r in folds:
        print(r["fold"], r["selected_epoch"], r["oracle_epoch"], r["input_depth_scale"],
              {m: round(r[m], 4) for m in (*METRICS, "n", "variance_ratio", "ols_slope")},
              {m: round(v, 4) for m, v in r["per_sample_mean"].items()})


if __name__ == "__main__":
    main()
