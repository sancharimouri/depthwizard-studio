# Learned scale modulation (Method 4) — Summary

_Sourced from `data/dfc2019/experiments/method4/method4_spatial_cv_results.json`,
`scripts/evaluate_method4.py` (read in full), and one new computation this
audit ran itself: reloading `method4_fold0.pt`, reproducing its exact
deterministic train-time normalization stats, and running inference over all
50 tiles' held-out quadrant to get the per-pixel prediction distribution that
no existing file in this repo contains. No model was retrained; the
checkpoint's weights were only loaded and run forward._

## 1. Item 1 (MOST IMPORTANT) — why Method 4's baseline doesn't match Method 3's Test 2 baseline

**Root cause found and confirmed exactly: the two "baselines" are not the
same fitting procedure, despite using an identical quadrant-split protocol
and identical valid-pixel masks.**

Read `scripts/evaluate_prior_spatial_cv.py` (Method 3) and
`scripts/evaluate_method4.py` (Method 4) side by side.

**What's identical** (ruled out as the cause):
- `quadrant_bounds()` in Method 4 produces the exact same four regions, in
  the exact same order, as Method 3's `folds` list (q0=top_left,
  q1=top_right, q2=bottom_left, q3=bottom_right).
- Valid-pixel masks are identical. Checked all 50 tiles' `n_pixels` for
  fold 0 / top_left: **zero mismatches** — Method 3's extra `building>=0 &
  building<=1` condition (present because it also loads the building prior)
  never actually excludes a pixel in practice, since HOTOSM's sigmoid output
  bilinearly resized stays within [0,1] by construction.

**What's different — the actual cause:**
- **Method 3's baseline** (`evaluate_prior_spatial_cv.py`, inside the
  per-tile loop): fits a **separate `LinearRegression` per tile per fold**,
  using only that tile's own 3 training quadrants. 50 tiles × 4 folds =
  **200 independent single-tile affine fits**, each with its own slope and
  intercept tailored to that tile's own DAv2↔AGL relationship.
- **Method 4's baseline** (`evaluate_method4.py`, `collect_baseline_samples`
  + one `LinearRegression.fit()` call *outside* the per-tile loop, right
  after training the CNN): pools up to 5,000 subsampled pixels **per tile
  from all 50 tiles' training quadrants together**, and fits **one single
  global affine model per fold** (4 fits total, not 200), applied uniformly
  to every one of the 50 tiles' held-out quadrant.

**Verified the mechanism directly on real data**, tile `JAX_004_006`,
fold 0 / top_left, pulling both scripts' actual saved per-tile baseline
records:

| | Method 3 (per-tile fit) | Method 4 (pooled global fit) |
|---|---:|---:|
| MAE | 2.6145 m | 2.3126 m |
| RMSE | 3.8561 m | 3.7958 m |
| Pearson | 0.68122369 | 0.68122361 |
| Spearman | 0.54457628 | 0.54457628 |
| n_pixels | 260,076 | 260,076 |

Pearson/Spearman are identical to 7 decimal places, MAE/RMSE differ. This is
not a coincidence — it's a mathematical identity: **Pearson/Spearman
correlation between `a·D+b` and truth is invariant to the choice of `a>0,
b`** for any fixed evaluation set, so it doesn't matter whether `a,b` came
from a per-tile fit or a pooled global fit — only the *magnitude-sensitive*
metrics (MAE, RMSE) can differ, and they do, because a single global
slope/intercept can't adapt to each tile's own AGL range.

**Aggregate confirmation** (fold 0 / top_left, 50 tiles): Method 3's mean
baseline MAE = 3.169 m, Pearson = 0.6015; Method 4's reported fold-0
baseline = MAE 3.648 m, Pearson 0.6050 — MAE differs by ~15%, Pearson by
~0.6%. Same pattern holds at the full-experiment level (baseline MAE 3.39 m
vs 3.88 m; baseline Pearson 0.5824 vs 0.5876).

**Consequence for interpreting Method 4's headline numbers:** Method 4 is
being compared against a strawman for MAE/RMSE purposes specifically. If
compared instead against Method 3's real (per-tile-fit) baseline —
MAE 3.39 m / RMSE 4.58 m / Pearson 0.582 / Spearman 0.509 — then Method 4's
own numbers (MAE 3.27 m / RMSE 5.17 m / Pearson 0.547 / Spearman 0.485) show
a much smaller MAE win (3.39→3.27 m, ~3.5%, not the ~16% implied by
comparing against its own baseline) and an **RMSE that is actually worse**
than Method 3's real baseline (5.17 m vs 4.58 m, ~13% worse) — see
`verdict.md`.

## 2. Item 2 — predicted-height distribution check (new computation, none existed)

No per-pixel prediction array is saved anywhere in this repo — only
aggregate JSON metrics and the raw `.pt` checkpoints. Loaded
`method4_fold0.pt`, reproduced the exact deterministic training-time
normalization stats (`fit_stats()` has no RNG dependency — it's a closed-form
mean/std over train-only pixels, so re-running it from the same files
reproduces the exact same numbers used at training time), and ran the real
`predict_quadrant()` inference function over all 50 tiles' held-out
quadrant 0 (~11.56M pixels).

**Regression-to-the-mean, confirmed dramatically:**

| | True AGL | Method 4 prediction |
|---|---:|---:|
| mean | 3.3849 m | 2.0872 m |
| std | 6.9527 m | 2.5662 m |
| p1–p99 range | 25.636 m | 10.917 m |
| max | 97.447 m | 22.506 m |

- **Variance ratio (pred/true) = 0.136** — predicted variance is only ~14%
  of the true variance.
- **OLS slope of (prediction) ~ (true AGL) = 0.187** — a well-calibrated
  model would have slope ≈ 1.0; 0.187 means for every 1 m of true variation,
  the prediction moves by only ~0.19 m on average. This is the textbook
  signature of regression-to-the-mean, not an open mystery.
- The model never predicts anywhere near the true tall-structure range
  (predicted max 22.5 m vs true max 97.4 m across the dataset).

**Mechanistic corroboration** — spot-checked the model's own internal
scale/residual outputs (not just the final prediction) for tile
`JAX_004_006`: the learned multiplicative **scale is nearly constant**
(mean 1.017, std 0.037 — varies only ~±4% around identity across the whole
tile) and the additive **residual**, while more variable (std 1.20,
range −0.24 to 6.51), never approaches the magnitude needed to reconstruct
AGL's true range. In practice, "the learned spatially-varying correction"
barely deviates from "raw DAv2, unchanged, plus a small smooth bump" — see
`gaps-and-fixes.md` §2 for why this is architecturally unsurprising.

## 3. Item 3 — aggregation arithmetic

Confirmed correct, bit-exact. `d['baseline']`/`d['method4']` (the "Overall"
table) equal the mean of the 4 fold-level blocks, each of which is itself
the mean of that fold's 50 per-tile metrics (`agg()` in
`evaluate_method4.py`). Reproduced this directly from the JSON: mean-of-
4-fold-means matches the reported overall to floating-point precision for
all 8 baseline/method4 metrics.

Note this is architecturally different from Method 3's Test 2, which pools
all 200 tile-fold records into one flat list and takes a single mean (no
intermediate per-fold aggregation stored). In this specific dataset the two
approaches happen to produce numerically identical results, because every
fold has exactly 50 tiles (verified: flat-pool-of-200 vs mean-of-4-fold-
means differ by <1e-9 for every metric) — but that equivalence only holds
because the group sizes are equal; it's not a general property of the two
aggregation schemes.

## 4. Item 4 — training script: input, architecture, loss

Read `scripts/evaluate_method4.py` in full.

- **Input**: 4-channel patch (RGB/255, per-channel normalized, + DAv2
  normalized), 64×64 patches — matches the plan's "DFC RGB + DAv2" diagram
  exactly (`PatchDataset.__getitem__`, `predict_quadrant`).
- **Architecture** (`ScaleModulationNet`): 4 conv layers (4→32→32→64→64
  channels, 3×3, ReLU), then two parallel 1×1-conv heads producing a
  per-pixel scale and residual. Both heads are **zero-initialized**
  (`nn.init.zeros_`), so the model starts training from
  `scale=exp(log(4)·tanh(0))=1.0`, `residual=0` — i.e. training starts
  exactly at "prediction = raw DAv2, unmodified" and only learns small
  deviations from there.
- **Loss**: `huber_masked` (`F.smooth_l1_loss`, beta=1.0) on
  `pred vs. truth`, **plus** `0.01 × smoothness_loss(scale)` — an L1
  total-variation penalty on the *scale map itself* (not the prediction).
  **Zero rank-aware or structure-preserving loss term anywhere** — confirmed
  by reading every loss-related line in the script; only these two terms are
  ever summed into `loss`.
- This combination — a magnitude-only loss (Huber, which like MSE/L1 is
  minimized in expectation by the conditional mean/median when the
  conditioning signal is weak) trained for only 12 epochs on a small,
  patch-sampled budget (12 patches × 64×64 per quadrant per tile), starting
  from an identity-mapping initialization, with an explicit smoothness
  penalty actively discouraging the scale map from varying — is a complete,
  concrete mechanism for the observed regression-to-the-mean, not a mystery
  requiring further investigation. See `gaps-and-fixes.md` §2 and
  `verdict.md` for the direct implication for Method 4's own stated next
  step ("modify the objective to preserve rank").

## 5. Item 5 — leakage check

No leakage found, in either the CNN training or the baseline fit:

- `fit_stats()` (normalization mean/std) is called on `train_stats_subset`,
  which explicitly zeroes the held-out quadrant before stats are computed
  (`train_valid[r0:r1, c0:c1] = False`, in `main()` before `fit_stats()` is
  called) — confirmed by reading the exact lines, for every one of the 4
  folds independently.
- `sample_patch_origins` bounds every sampled patch to a single quadrant's
  row/col range — no patch straddles a train/test boundary.
- `collect_baseline_samples` (the pooled global baseline fit) also only
  reads from `train_mask`-restricted pixels.
- Each of the 4 folds gets a freshly-initialized `ScaleModulationNet()`, a
  freshly-recomputed `rgb_stats`/`depth_stats`, and a freshly-refit baseline
  — no state (weights, stats, or fitted coefficients) carries over between
  folds.

## 6. Item 6 — per-fold pattern, full metric rows

Pulled all 4 folds' full baseline/method4 blocks directly from the JSON
(not just the MAE numbers already quoted in the note):

| Fold (held-out quadrant) | MAE base→m4 | RMSE base→m4 | Pearson base→m4 | Spearman base→m4 |
|---|---|---|---|---|
| 0 (top_left) | 3.6483→2.9028 ↓ | 5.0650→4.8291 ↓ | 0.6050→0.5661 ↓ | 0.5165→0.4877 ↓ |
| 1 (top_right) | 3.6298→3.2616 ↓ | 5.4526→**5.4865 ↑** | 0.5847→0.5264 ↓ | 0.4954→0.4782 ↓ |
| 2 (bottom_left) | 4.5594→3.8493 ↓ | 5.9058→5.6558 ↓ | 0.5319→0.4952 ↓ | 0.4765→0.4384 ↓ |
| 3 (bottom_right) | 3.6892→3.0514 ↓ | 4.9403→4.7232 ↓ | 0.6287→0.5997 ↓ | 0.5565→0.5365 ↓ |

**MAE improves in all 4 folds and Pearson/Spearman worsen in all 4 folds —
exactly as the note states.** But **RMSE does not follow the same uniform
pattern**: it improves in 3 of 4 folds and **gets worse in fold 1**
(5.4526→5.4865 m). The note's overall table shows RMSE improving
(5.3409→5.1736) only because the aggregate mean over 4 folds happens to mask
fold 1's regression. This is consistent with, and further supports, the
regression-to-the-mean finding in §2: RMSE penalizes large errors
disproportionately, so a mean-compressed predictor is specifically exposed
on whichever quadrant/tiles have more extreme (tall) true values — exactly
what a compressed-variance predictor would be expected to do worse on.
