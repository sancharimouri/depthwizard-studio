#!/usr/bin/env python3
"""
Method 4 v2 — Phase 0: corrected comparison baseline.

Refits the DAv2-only baseline EXACTLY like Method 3's Test 2
(scripts/evaluate_prior_spatial_cv.py): one LinearRegression per tile per
fold, trained only on that tile's 3 training quadrants, evaluated on the
held-out quadrant. This replaces Method 4's original pooled-global-per-fold
baseline (see docs/method-audit/04-learned-scale-modulation/summary.md §1
for why that was a weaker, non-comparable baseline).

No retraining. Method 4's own (already-trained, unmodified) checkpoint
metrics are read directly from the existing
data/dfc2019/experiments/method4/method4_spatial_cv_results.json — this
script only recomputes the baseline side of the comparison.

Reuses evaluate_prior_spatial_cv.metrics() directly (imported, not
reimplemented) for the metric computation, and evaluate_method4.quadrant_bounds
directly for the quadrant split, so both halves of the comparison are
guaranteed to use the exact same logic already used elsewhere in this repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path("/Users/anweshasaha/projects/DepthWizard2")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_method4 as m4  # noqa: E402
from evaluate_prior_spatial_cv import metrics as m3_metrics  # noqa: E402

DEPTH_DIR = PROJECT_ROOT / "data/dfc2019/experiments/dav2_baseline/depth"
RGB_DIR = PROJECT_ROOT / "data/dfc2019/raw/RGB/Track1-RGB"
TRUTH_DIR = PROJECT_ROOT / "data/dfc2019/raw/Truth/Track1-Truth"
OUTDIR = PROJECT_ROOT / "data/dfc2019/experiments/method4_v2_phase0"
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    depth_files = sorted(DEPTH_DIR.glob("*_depth.npy"))
    tiles = []
    for dp in depth_files:
        tile_id = dp.stem.removesuffix("_depth")
        rp = RGB_DIR / f"{tile_id}_RGB.tif"
        tp = TRUTH_DIR / f"{tile_id}_AGL.tif"
        if rp.exists() and tp.exists():
            tiles.append(m4.Tile(tile_id, dp, rp, tp))

    print(f"Tiles: {len(tiles)}")

    # Load depth + truth + valid mask once (matches m4.load_tile's valid
    # definition: isfinite(depth) & isfinite(truth) & truth >= 0).
    cache = []
    for t in tiles:
        depth = np.load(t.depth_path).astype(np.float32)
        with rasterio.open(t.truth_path) as src:
            truth = src.read(1).astype(np.float32)
        valid = np.isfinite(depth) & np.isfinite(truth) & (truth >= 0)
        cache.append((t.tile_id, depth, truth, valid))

    fold_results = []

    for held_out_q in range(4):
        print(f"\nFold {held_out_q}: fitting per-tile baseline (train on 3 quadrants, test on held-out quadrant)")
        tile_results = []

        for tile_id, depth, truth, valid in cache:
            h, w = truth.shape
            r0, r1, c0, c1 = m4.quadrant_bounds(h, w, held_out_q)

            test_mask = np.zeros(truth.shape, dtype=bool)
            test_mask[r0:r1, c0:c1] = True

            train_mask = valid & ~test_mask
            test_mask = valid & test_mask

            if train_mask.sum() < 100 or test_mask.sum() < 100:
                raise RuntimeError(f"{tile_id} fold {held_out_q}: insufficient pixels")

            x_train = depth[train_mask].reshape(-1, 1)
            y_train = truth[train_mask]
            x_test = depth[test_mask].reshape(-1, 1)
            y_test = truth[test_mask]

            model = LinearRegression()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

            m = m3_metrics(y_test, pred)
            tile_results.append({"tile": tile_id, "baseline_per_tile": m})

        # Fold-level aggregate (mean of 50 per-tile metrics, matching
        # evaluate_method4.py's own agg() aggregation style).
        def agg(records, key):
            vals = [r[key] for r in records]
            return {
                "mae_m": float(np.mean([x["mae_m"] for x in vals])),
                "rmse_m": float(np.mean([x["rmse_m"] for x in vals])),
                "pearson": float(np.mean([x["pearson"] for x in vals])),
                "spearman": float(np.mean([x["spearman"] for x in vals])),
                "n_tiles": len(vals),
            }

        fold_agg = agg(tile_results, "baseline_per_tile")
        print("  fold baseline_per_tile:", fold_agg)

        fold_results.append(
            {
                "fold": held_out_q,
                "held_out_quadrant": held_out_q,
                "baseline_per_tile": fold_agg,
                "tiles": tile_results,
            }
        )

    def final_agg(key):
        blocks = [f[key] for f in fold_results]
        return {
            "mae_m": float(np.mean([b["mae_m"] for b in blocks])),
            "rmse_m": float(np.mean([b["rmse_m"] for b in blocks])),
            "pearson": float(np.mean([b["pearson"] for b in blocks])),
            "spearman": float(np.mean([b["spearman"] for b in blocks])),
            "n_folds": 4,
        }

    overall = final_agg("baseline_per_tile")

    output = {
        "experiment": "Method 4 v2 Phase 0 — corrected per-tile-per-fold baseline",
        "method": "One LinearRegression(AGL ~ DAv2) per tile per fold, trained "
        "only on that tile's 3 training quadrants (identical procedure to "
        "evaluate_prior_spatial_cv.py's Method-3 Test-2 baseline). NOT the "
        "pooled-global-per-fold baseline evaluate_method4.py originally used.",
        "baseline_per_tile_overall": overall,
        "folds": fold_results,
    }

    out_path = OUTDIR / "phase0_corrected_baseline.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 78)
    print("PHASE 0 CORRECTED BASELINE (overall, mean of 4 folds)")
    print("=" * 78)
    print(overall)

    # Also print the already-known Method 4 numbers and original weak
    # baseline for direct side-by-side reporting.
    orig = json.loads(
        (PROJECT_ROOT / "data/dfc2019/experiments/method4/method4_spatial_cv_results.json").read_text()
    )
    print("\nOriginal (pooled-global) baseline:", orig["baseline"])
    print("Method 4 (existing checkpoints):   ", orig["method4"])


if __name__ == "__main__":
    main()
