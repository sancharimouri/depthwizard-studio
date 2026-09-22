# GCP regression — Gaps & Fixes

_One issue per subsection. No new computation run except cheap re-aggregation
of existing CSVs — every number cited already existed on disk before this
audit._

## 1. The note collapses a 5-row table into one unlabeled number

**Problem:** Section on `evaluate_sparse_anchors.py` prints "MAE 4.012 m /
RMSE 5.197 m / Pearson 0.492 / Spearman 0.412" twice, with no anchor-count
label, even though the script was explicitly designed to test 5 different
anchor counts (2/3/5/10/20).

**Checked:** Pulled the full table from `sparse_anchor_summary.csv`. The
number is the **20-anchor** row, exact to 3 decimals.

**Outcome:** Not just a labeling slip — the omitted rows are the more
important part of the story. At 2-3 anchors the naive implementation is
wildly unstable (mean MAE 133 m and 16 m respectively, driven by outlier
tiles where a 2-point affine fit's slope blows up). This matters directly
for the Possibility A/B question (see `verdict.md`): a "3 to 5 known points"
test is explicitly what Possibility A/B is asking about, and the real answer
at that anchor count is "still far from fixed," not "one clean 20-anchor
number."

## 2. Stage 1's table only shows the n=20 row; full breakdown shows the effect is anchor-count-independent

**Problem:** The note's Stage 1 table shows Grid/Random/Spatial+height only
at 20 anchors, making Spatial+height's underperformance look like it might
be a large-n artifact.

**Checked:** Pulled 5/10/20 breakdown from `sparse_anchor_robust/summary.csv`.
Spatial+height is worse than Grid and Random at **every** anchor count
tested (5, 10, and 20) — see `summary.md` §2. It's a placement-strategy
effect, not an anchor-count effect.

## 3. Spatial+height underperformance — traced to code, confirmed as a real effect

**Checked (per explicit request):** Read `select_spatial_height()` and
`select_grid()` in `scripts/evaluate_sparse_anchor_robust.py` in full.

**Outcome:** Not a bug. `select_spatial_height` deliberately guarantees
representation from every AGL quantile stratum, including the extreme top
stratum (tallest points in the tile), before farthest-point selection.
`select_grid` and `select_random` have no such bias and end up sampling
mostly typical (low/ground) heights, matching the scene's natural
distribution. With a 2-parameter OLS fit on only 5-20 points, forcing in
extreme-height, high-leverage points measurably destabilizes the fit. This
is a legitimate, explainable statistical cost of the strategy's own design
goal (height diversity), not an implementation defect.

## 4. RANSAC was dropped with a real, evidenced reason — just never stated in prose

**Problem:** Note says "Dropping RANSAC from the pipeline" with nothing to
back it up in the text.

**Checked:** RANSAC rows exist in `sparse_anchor_regression/results.csv`,
`summary.csv`, and `REPORT.txt`. RANSAC underperforms OLS and Huber on every
metric at every placement/anchor-count tested (RMSE ~5.9-6.0 m vs ~4.5-4.7 m;
Pearson collapses to ~0.13-0.25 vs ~0.47-0.53). Its own inlier-fraction
diagnostic (~0.55-0.61) shows why: with only 10-20 anchors, RANSAC's
minimal-subset consensus routinely rejects nearly half of them as
"outliers," fitting on an even smaller, arbitrarily-chosen remainder.

**Outcome:** The decision to drop RANSAC is well-supported by the data that
was already computed — the table is sitting in `REPORT.txt` unused. The gap
is presentation, not methodology: neither `REPORT.txt` nor the note ever
walks through this table before declaring the drop.

## 5. "New baseline" configuration is implicit in code, not stated anywhere

**Problem:** The note's final heading, "New baseline: Global 20-anchor grid
calibration," never specifies OLS vs. Huber vs. which placement, despite
Stage 2 having already shown Huber beats OLS on MAE at the same
placement/anchor-count.

**Checked:** No file contains that exact phrase. Traced forward:
`sparse_anchor_spatial/config.json` explicitly sets `"regression": "OLS"`;
`evaluate_sparse_anchor_spatial_residual.py` and
`evaluate_sparse_anchor_smooth_residual.py` both hardcode
`sklearn.linear_model.LinearRegression` with no Huber path at all.

**Outcome:** This is a real, if quiet, methodological drift: Stage 2 found
Grid+Huber+20 (MAE 2.929 m) beats Grid+OLS+20 (MAE 3.131 m), but every stage
built afterward (3, 4, 4B) silently reverted to OLS as "the" baseline with no
documented decision to do so. Not a correctness bug — OLS is a defensible,
simpler default, and Huber's RMSE was actually *worse* than OLS's (4.718 m vs
4.482 m, see gap #4's sibling finding in `summary.md` §5) — but the choice
should have been stated, not left to be inferred from which regressor a
later script happens to import.

## 6. Stage 3's "repeat with Huber" branch — now run; confirms rather than overturns the OLS finding

**Originally checked (prior audit pass):** no Huber-on-quadrant-wise output
existed anywhere in the repo. Quadrant-wise was clearly *worse* than
tile-wide under OLS (MAE +0.225 m, RMSE +1.03 m, Pearson -0.067, Spearman
-0.075). Judged consistent with "skip the Huber repeat since the spatial
split was already meaningfully worse," not a dropped task.

**Since run** (`scripts/evaluate_sparse_anchor_spatial_huber.py`, exact
Stage-1 grid/20 anchors reused, same region-splitting/eval-mask code
imported directly from `evaluate_sparse_anchor_spatial.py`, only the
per-quadrant fit swapped from OLS to Huber with Stage 2's exact
hyperparameters): quadrant-wise+Huber = MAE 3.314 m / RMSE 5.562 m /
Pearson 0.439 / Spearman 0.377 (`data/dfc2019/experiments/sparse_anchor_spatial_huber/`).

Compared against tile-wide+Huber (Stage 2's Grid+Huber+20, already on
record: MAE 2.929 m / RMSE 4.718 m / Pearson 0.532 / Spearman 0.471),
quadrant-wise remains worse on **every metric**, and by a **wider margin**
on MAE/Pearson/Spearman than the OLS comparison showed (only RMSE's gap
narrows slightly). **Huber does not change the conclusion that spatial
subdivision hurts — it holds regardless of which regressor fits each local
ruler**, which is a stronger, now-verified version of the original
skip-reasoning, not merely an assumption that turned out fine.

## 7. Stage 4's own conclusion sentence understates its own result

**Problem:** "The error... is not well described by a low-order polynomial"
reads as if low-order polynomials failed. The actual results table
(`sparse_anchor_spatial_residual/summary.csv`) shows the opposite: **linear**
(degree-1, the lowest non-trivial polynomial) gave a real improvement (MAE
+0.061 m mean, RMSE +0.132 m mean, majority of CV folds improved), while
**quadratic** (degree-2, higher order) is what overfit and generalized
worse.

**Fix applied:** Restated the conclusion accurately in `summary.md` §7 and
`verdict.md` using the exact per-fold numbers: the residual has real,
strongly spatially-autocorrelated structure (neighbor correlation ≈0.97) that
a *linear* spatial term partially captures — it's specifically the jump to a
*higher-order* polynomial that breaks generalization, not "low order" per se.

## 8. Nothing else found to be missing or fabricated

Cross-checked Stage 4B's specific claims ("rbf improves RMSE on 0/50 tiles";
"MAE improves on only 2/50, 1/50, 1/50 tiles") against the tile-level paired
table in `sparse_anchor_smooth_residual/REPORT.txt` — exact match. (Note:
`REPORT.txt`'s fold-level table shows nonzero "improved_rmse_folds" counts
for the same models — e.g. 17/200 for rbf_medium — which is a different,
coarser unit than the tile-level aggregate the note quotes; both numbers are
real, they're just answering slightly different questions, and the note
correctly used the tile-level one.)
