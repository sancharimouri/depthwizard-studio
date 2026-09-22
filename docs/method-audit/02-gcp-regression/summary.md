# GCP regression (sparse-anchor calibration) — Summary

_Sourced from the actual CSVs/JSON/scripts in `data/dfc2019/experiments/` and
`scripts/`, re-aggregated and cross-checked directly — not taken from
`raw-notes.pdf`'s prose at face value. No refitting or rerunning was needed;
every number below already existed on disk._

## Framing (from the note)

Stage 2 opens with an explicit fork: **Possibility A** — a handful of known
height points "dramatically fixes" the tile, meaning DAv2 just needs a local
scale ("a local ruler"). **Possibility B** — a few known points still leave
massive errors, meaning DAv2 is structurally distorting shape and a heavier
model (RDAH-Net) is required. See `verdict.md` for which one the evidence
supports — the note itself never states a conclusion on this.

## 1. Naive sparse-anchor test (`sparse_anchor/`)

`evaluate_sparse_anchors.py` tested anchor counts 2, 3, 5, 10, 20, fitting
`AGL = a·DAv2 + b` from N pixels of the same tile and evaluating on the rest.
The note quotes one number twice with no anchor-count label
("MAE 4.012 m / RMSE 5.197 m / Pearson 0.492 / Spearman 0.412"). Pulling the
full table from `sparse_anchor_summary.csv`:

| Anchors | MAE mean (m) | MAE median (m) | RMSE mean (m) | Pearson mean | Spearman mean |
|---:|---:|---:|---:|---:|---:|
| 2 | 133.20 | 13.10 | 182.47 | 0.449 | 0.374 |
| 3 | 16.42 | 10.08 | 22.15 | 0.394 | 0.329 |
| 5 | 9.10 | 6.36 | 12.05 | 0.373 | 0.323 |
| 10 | 4.98 | 3.69 | 6.54 | 0.445 | 0.386 |
| **20** | **4.012** | 3.254 | **5.197** | **0.492** | **0.412** |

**The quoted number is the 20-anchor row**, confirmed to 3 decimal places.
The 2-anchor row's huge mean (133 m) is driven by a handful of catastrophic
per-tile blowups (one tile hits MAE **3,711 m**) — an expected consequence of
fitting a 2-parameter affine model to exactly 2 points: if the two anchors'
DAv2 values are close together, the fitted slope explodes. This is a
numerical-instability property of underdetermined 2-point fits, not a script
bug — confirmed by checking the per-tile distribution in
`sparse_anchor_metrics.csv` directly (median at 2 anchors is 13.1 m, still
large, but nowhere near the mean).

## 2. Stage 1 — placement robustness (`sparse_anchor_robust/`)

The note's table only shows the 20-anchor row per placement. Full breakdown
from `summary.csv` (`strict` = spatial-holdout evaluation, which is what the
note's numbers match exactly):

| Placement | n=5 MAE | n=10 MAE | n=20 MAE | n=5 RMSE | n=10 RMSE | n=20 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Grid | 3.78 | 3.26 | **3.13** | 5.85 | 4.85 | **4.48** |
| Random | 3.97 | 3.44 | **3.22** | 5.79 | 4.98 | **4.61** |
| Spatial+height | 5.94 | 4.91 | **4.58** | 7.40 | 6.07 | **5.63** |

Error decreases monotonically with anchor count for all three placements;
Spatial+height is worse than Grid/Random **at every anchor count**, not just
20 (see #3 below). Sparse anchors do cut absolute error relative to the
global-isotonic baseline (3.69 m → 3.13 m at Grid/20, ~15%), but Spearman
barely moves (0.446 → 0.463) — the note's claim that anchors "do not solve
DAv2's underlying scene-dependent ordering errors" holds up.

## 3. Why Spatial+height-stratified underperforms — real effect, not a bug

Read `select_spatial_height()` and `select_grid()` in
`scripts/evaluate_sparse_anchor_robust.py`. `select_grid` just takes the
nearest valid pixel to each cell of a spatial grid — it samples whatever
height happens to be there, which for most urban scenes is overwhelmingly
ground/low-vegetation height (the natural, dominant class). `select_spatial_height`
deliberately partitions the tile into ~N AGL-quantile strata spanning the
**entire** height range (min to max) and guarantees at least one candidate
from every stratum — including the top quantile, i.e. rooftops/treetops —
before doing farthest-point selection. With only 5-20 total points feeding a
2-parameter OLS fit, forcing inclusion of extreme-height (high-leverage,
low-population, often noisier in DAv2's per-tile-normalized output) pixels
makes the fit measurably less stable than anchors that reflect the scene's
natural height distribution. This is the same underlying mechanism as the
2-3 anchor blowups in #1 — small-sample OLS is sensitive to leverage points,
and Spatial+height actively selects for them. No bug found in the selection
code; the strategy does exactly what it's designed to do, and that design
choice is what costs it accuracy.

## 4. Stage 2 — regression robustness (`sparse_anchor_regression/`), RANSAC included

The note's comparison table shows only OLS/Huber, ending with "Dropping
RANSAC from the pipeline" and no evidence. RANSAC numbers **do** exist in
`results.csv`/`summary.csv`/`REPORT.txt` (regressor column includes
`ransac`):

| Regressor | Placement/n | MAE (m) | RMSE (m) | Pearson | Spearman | mean RANSAC inlier fraction | runs with RMSE>10m |
|---|---|---:|---:|---:|---:|---:|---:|
| OLS | Grid/20 | 3.131 | 4.482 | 0.526 | 0.463 | — | 3/50 |
| Huber | Grid/20 | 2.929 | 4.718 | 0.532 | 0.471 | — | 4/50 |
| **RANSAC** | Grid/20 | **3.292** | **5.970** | **0.131** | **0.136** | 0.573 | 4/50 |
| RANSAC | Random/20 | 3.330 | 5.915 | 0.192 | 0.192 | 0.576 | 85/1000 |

RANSAC underperforms both OLS and Huber on every metric, sometimes
dramatically (Pearson collapses to ~0.13-0.25 vs ~0.47-0.53). Its own inlier
fraction (~0.55-0.61) shows it's routinely discarding roughly half of an
already-tiny 10-20-point anchor set as "outliers," fitting on the remainder —
with so few points, its minimal-subset consensus sampling can't reliably tell
signal from noise. This is a well-evidenced, if under-narrated, reason to
drop it: the table exists in `REPORT.txt`, it's just never discussed in
prose there or in the note.

## 5. Huber: central error vs. large-error tail — verified against per-tile paired deltas

Checked directly against `paired_deltas.csv` (per-tile Huber-vs-matched-OLS
deltas; positive = Huber improvement), not just inferred from the summary
MAE/RMSE gap:

- **MAE** (central/typical error): mean delta +0.165 m, median +0.079 m,
  improves in **71%** of individual tile-runs.
- **RMSE** (large-error-sensitive): mean delta **-0.170 m** (worse on
  average), improves in only **34%** of tile-runs.
- Restricting to the tail — the 10% of tile-runs with the largest OLS RMSE
  to begin with — Huber's RMSE degradation gets **worse**, not better (mean
  delta -0.306 m vs -0.170 m overall; only 37% improve).

This directly confirms the note's claim with percentile-level evidence:
Huber's down-weighting of large residuals during fitting helps the typical
case but leaves (and slightly worsens) the worst cases.

## 6. Stage 3 — spatial model (`sparse_anchor_spatial/`)

Only two of the four planned models (Tile-wide, Quadrant-wise, Grid-wise
local affine, Global+spatial-residual) were actually run — confirmed by
`config.json` (`"experiment": "sparse_anchor_spatial_stage_3A"`) and by
`summary.csv`, which contains exactly two rows: `tile_wide` and
`quadrant_wise_aggregate`. No script, config, or output file anywhere in the
repo mentions "grid-wise local affine" — it was never run. "Global model +
spatial residual" was taken up, but as a separately-scoped Stage 4
experiment (`sparse_anchor_spatial_residual/`), not as a third Stage-3 model.

Result: Quadrant-wise (MAE 3.356 m, RMSE 5.515 m) is clearly worse than
Tile-wide (MAE 3.131 m, RMSE 4.482 m) — Pearson drops 0.526→0.459, Spearman
0.463→0.389. The note's planned "repeat with Huber if the difference is
meaningful" branch was never run — no Huber-on-quadrant output exists
anywhere. Given the branch's evident intent (test whether regressor choice
matters *if* the spatial split helps), and the split didn't help, running it
would be moot; its absence is consistent with that logic, not with an
incomplete run.

## 7. Stage 4 — spatial residual, exact magnitude (`sparse_anchor_spatial_residual/`)

The note concludes: *"The error has spatial structure, but it's not well
described by a low-order polynomial."* Pulled exact numbers from
`summary.csv`/`REPORT.txt` (4-fold spatial-blocked CV vs. `global_affine`
baseline, MAE 3.127 m / RMSE 4.346 m):

| Model | Δ MAE mean (m) | Δ RMSE mean (m) | Folds improved (MAE / RMSE) |
|---|---:|---:|---:|
| constant | -0.191 (worse) | -0.138 (worse) | 70/200, 81/200 |
| **linear x,y** | **+0.061** | **+0.132** | **108/200, 120/200** |
| quadratic x,y | -0.355 (worse) | -0.310 (worse) | 83/200, 84/200 |

Linear (degree-1, genuinely the *lowest* non-trivial polynomial) gives a
real, if modest, improvement in a majority of folds (54-60%). Quadratic
(degree-2, *higher* order) is what fails — it overfits and generalizes
worse in a minority of folds (~42%). So the accurate restatement is closer
to the opposite emphasis: **the low-order (linear) term is exactly what
works; it's the higher-order (quadratic) term that breaks.** Residual
spatial autocorrelation is very high (neighbor correlation ≈0.97), which is
consistent with the "spatial structure exists" half of the note's sentence —
just not the "low-order polynomial fails" half. This is explicitly an
oracle experiment (fits the residual model on ground-truth AGL); the report
itself is upfront about that limitation.

## 8. What "New baseline: Global 20-anchor grid calibration" actually specifies

No file anywhere is literally titled or contains that phrase — it's the note
author's own heading, not a quote from any `REPORT.txt`. Traced the
configuration through the code: Stage 3's `config.json` sets
`"regression": "OLS"`; Stage 4's and Stage 4B's residual scripts
(`evaluate_sparse_anchor_spatial_residual.py`,
`evaluate_sparse_anchor_smooth_residual.py`) both hardcode
`sklearn.linear_model.LinearRegression` (OLS) with no Huber option at all.
So **"the baseline" carried through every subsequent stage is Grid
placement, OLS regression, 20 anchors, tile-wide (non-spatial) affine** —
i.e. MAE≈3.13 m / RMSE≈4.48 m — **not** Huber, even though Stage 2 had
already shown Grid+Huber+20 beats it on MAE (2.929 m). No file records an
explicit decision to keep OLS over the empirically better Huber regressor
going forward; it's implicit in what the later scripts hardcode. See
`gaps-and-fixes.md`.

## 9. Possibility A vs. Possibility B

See `verdict.md`.
