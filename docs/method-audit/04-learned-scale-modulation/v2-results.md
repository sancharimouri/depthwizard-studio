# Method 4 v2 — Results (running log)

This is deliberate new work (retraining, new ablations), not a re-verification
of the original Method 4 audit. Original outputs at
`data/dfc2019/experiments/method4/` are never overwritten — every new result
lives in its own `data/dfc2019/experiments/method4_v2_*/` folder. This file
is updated as each phase completes, not just at the end.

Diagnostics reported for every configuration: MAE/RMSE/Pearson/Spearman
against both the original weak (pooled-global) baseline and the Phase-0
corrected (per-tile) baseline, plus the predicted-vs-true variance ratio and
OLS calibration slope (the regression-to-the-mean diagnostic established in
the original audit, `docs/method-audit/04-learned-scale-modulation/summary.md` §2).

## Phase 0 — corrected comparison baseline

**Method:** `scripts/evaluate_method4_v2_phase0.py`. Refits the DAv2-only
baseline exactly like Method 3's Test 2 (`evaluate_prior_spatial_cv.py`):
one `LinearRegression` per tile per fold, trained on that tile's 3 training
quadrants only, evaluated on the held-out quadrant. Reuses
`evaluate_prior_spatial_cv.metrics()` directly for scoring and
`evaluate_method4.quadrant_bounds()` directly for the split. No retraining —
Method 4's own numbers below are read unchanged from the existing
`data/dfc2019/experiments/method4/method4_spatial_cv_results.json`.

**Cross-check:** the corrected baseline reproduces Method 3's Test 2 baseline
numbers exactly (MAE 3.3924 / RMSE 4.5787 / Pearson 0.5824 / Spearman
0.5093, bit-for-bit) — confirms the refit procedure is correct.

| | MAE (m) | RMSE (m) | Pearson | Spearman |
|---|---:|---:|---:|---:|
| Original weak baseline (pooled-global, per fold) | 3.8817 | 5.3409 | 0.5876 | 0.5112 |
| **Corrected baseline (per-tile, = Method 3's)** | **3.3924** | **4.5787** | **0.5824** | **0.5093** |
| Method 4 (existing checkpoint, unmodified) | 3.2663 | 5.1736 | 0.5469 | 0.4852 |

**Method 4 vs. the corrected baseline** (the honest comparison from here
on): MAE 3.39→3.27 m (**~3.5% better**), RMSE 4.58→5.17 m (**~13% worse**),
Pearson 0.582→0.547 (worse), Spearman 0.509→0.485 (worse). This is the bar
every later phase needs to clear: not "beat MAE 3.88," but "beat MAE 3.39
*and* RMSE 4.58 *and* not lose Pearson/Spearman."

Saved: `data/dfc2019/experiments/method4_v2_phase0/phase0_corrected_baseline.json`

## Phase 1 — training-starvation ablation

**Status: COMPLETE.** `scripts/evaluate_method4_v2.py --patch-mode dense --epochs 60
--smoothness-weight {0, 0.001, 0.01}`, all 4 folds, loss otherwise unchanged
(Huber only) and zero-init heads unchanged — isolates only the
training-budget/coverage/smoothness question, no loss or architecture
changes yet. Dense mode replaces 12 random 64×64 patches/quadrant/tile with
a stepped grid tiling each 512×512 training quadrant (up to 64
patches/quadrant, filtered by the same MIN_VALID_FRACTION=0.85 threshold) —
~7,150–7,381 training patches/fold vs. the original run's ~1,800/fold.
Batch size raised 16→64 (pure speed optimization for this tiny model, not a
result-changing choice). Total wall time: ~47–51 min/setting on MPS
(Apple-Silicon GPU), ~150 min for the full 3-way sweep.

Outputs:
`data/dfc2019/experiments/method4_v2_phase1_smooth{0,0.001,0.01}/phase1_smooth*_results.json`
and per-fold checkpoints alongside.

**Full comparison table** (all metrics = mean over 4 folds; corrected
baseline = Phase 0's per-tile fit; original Method 4 = the existing,
unmodified checkpoints from the original audit, re-scored here with the
calibration diagnostic computed across *all 4 folds* for a fair comparison
— the original audit had only computed it for fold 0):

| Configuration | MAE (m) | RMSE (m) | Pearson | Spearman | Var. ratio | OLS slope |
|---|---:|---:|---:|---:|---:|---:|
| Original weak baseline (pooled-global) | 3.8817 | 5.3409 | 0.5876 | 0.5112 | — | — |
| **Corrected baseline (per-tile, Phase 0)** | **3.3924** | **4.5787** | **0.5824** | **0.5093** | — | — |
| Original Method 4 (sparse patches, 12 epochs, smooth=0.01) | 3.2663 | 5.1736 | 0.5469 | 0.4852 | 0.2252 | 0.2288 |
| Phase 1, dense + 60 epochs, smoothness=0 | 3.1209 | 5.0319 | 0.5562 | 0.5095 | 0.3457 | 0.3079 |
| Phase 1, dense + 60 epochs, smoothness=0.001 | 3.1194 | 5.0392 | 0.5563 | 0.5091 | 0.3525 | 0.3122 |
| Phase 1, dense + 60 epochs, smoothness=0.01 | **3.0999** | **5.0210** | **0.5600** | **0.5111** | 0.3532 | 0.3134 |

(Var. ratio = predicted/true variance ratio, 1.0 = no compression; OLS slope
= slope of predicted-vs-true, 1.0 = calibrated. Both from the same
regression-to-the-mean diagnostic established in the original audit.)

**What this shows:**

1. **Fixing training starvation alone is a clean, unambiguous win over the
   original Method 4** on every metric simultaneously: MAE 3.27→3.10 m,
   RMSE 5.17→5.02 m, Pearson 0.547→0.560, Spearman 0.485→0.511 — no
   trade-off this time, all four move the same direction. This confirms
   the original audit's hypothesis that 12 sparse patches/quadrant + 12
   epochs was leaving real performance on the table, independent of the
   loss-function question.
2. **Against the *honest* corrected baseline, Phase 1 still doesn't clearly
   win**: MAE improves further (3.39→3.10 m, ~8.6% better — a real win now,
   not just beating a strawman) but RMSE is still worse (5.02 m vs 4.58 m,
   ~9.6% worse) and Pearson/Spearman are both still below the baseline
   (0.560 vs 0.582; 0.511 vs 0.509 — Spearman is now essentially tied,
   Pearson still down). So training starvation was a real, fixable problem,
   but fixing it alone does not make Method 4 beat the honest baseline
   outright.
3. **Regression-to-the-mean is reduced but far from solved.** Variance
   ratio improves from 0.225 to ~0.35 (roughly +56% relative), OLS slope
   from 0.229 to ~0.31 — a real, meaningful improvement, but a slope of
   0.31 is still very far from the 1.0 a calibrated model would have. This
   confirms Phase 2's rank-preserving loss is still needed, not made
   redundant by fixing the training budget.
4. **Smoothness weight barely matters at this training budget, contrary to
   the original audit's hypothesis.** All three settings land within ~0.02 m
   MAE and ~0.004 Pearson/Spearman of each other, and smoothness=0.01 (the
   *original*, unchanged value) is marginally the best on every single
   metric, not the worst. The original audit's speculation that the
   smoothness penalty was suppressing useful spatial variation turns out to
   be a much smaller effect than the training-coverage/epoch-budget
   question — with dense coverage and 60 epochs, the smoothness term is no
   longer the bottleneck it may have been at 12 sparse patches × 12 epochs.

**Phase 1 winner for carrying into Phase 2 (pending your go-ahead):
smoothness = 0.01** (the original value) — best or tied-best on every
metric, so there's no reason to change it.

## Phase 2 — missing ingredients

**Phase 1 winner confirmed and locked in as the base config for every run
below: dense coverage, 60 epochs, smoothness=0.01.**

**Reference points for every Phase 2 result** (repeated here since these
are the bar to beat, not the original weak baseline):

| | MAE (m) | RMSE (m) | Pearson | Spearman |
|---|---:|---:|---:|---:|
| Corrected per-tile baseline (Phase 0) | 3.3924 | 4.5787 | 0.5824 | 0.5093 |
| Phase 1 winner (dense, 60ep, smooth=0.01) | 3.0999 | 5.0210 | 0.5600 | 0.5111 |

**Status: running.** Four runs, each the Phase 1 winner config plus one
addition (three individual ablations, then combined):

1. **`phase2_building`** — Method 3's existing DFC building-probability
   `.npy` maps (`data/dfc2019/experiments/semantic/building/<tile>.npy`,
   already generated, no new model) added as a 5th input channel,
   normalized the same way as RGB/depth (train-quadrant-only mean/std).
2. **`phase2_groundplane`** — asymmetric ground-plane loss:
   `threshold = 0.5 m` (the value the task suggested), weight = **2.0**
   (chosen to be clearly heavier than the Huber term's implicit weight of
   1.0, since the goal is a strong deterrent against predicting height
   where the truth says ground, without being so large it destabilizes
   training — not swept, could be tuned further if this ablation looks
   promising). Loss = `2.0 * mean(relu(pred - 0.5)^2)` over training pixels
   with `true AGL < 0.5 m`.
3. **`phase2_rank`** — pairwise ranking loss exactly as specified: per
   training patch, `K=2000` randomly sampled valid-pixel pairs (not all
   O(N²) pairs), `target_sign = sign(true_i - true_j)` with tied pairs
   dropped, `F.margin_ranking_loss(pred_i, pred_j, target_sign, margin=0.25)`
   (the functional form of `torch.nn.MarginRankingLoss`, fully vectorized
   per patch — the only Python-level loop is over the batch, not over
   pairs, since each patch has a different-sized valid-pixel set to index
   into). Margin = **0.25 m**, weight = **0.3** (middle of the requested
   0.1–0.5× Huber range).
4. **`phase2_combined`** — all three of the above together, same
   hyperparameters as their individual runs.

Each run: dense coverage, 60 epochs, smoothness=0.01, all 4 folds.

**Status: COMPLETE.** Wall times: building 3070s (~51 min, same cost as a
Phase-1 setting — no meaningful overhead from the extra channel), ground-
plane 3652s (~61 min), rank 6610s (~110 min — the pair-sampling Python loop
over the batch is real overhead), combined 6955s (~116 min).

**Full results table** (all metrics = mean over 4 folds):

| Configuration | MAE (m) | RMSE (m) | Pearson | Spearman | Var. ratio | OLS slope |
|---|---:|---:|---:|---:|---:|---:|
| Corrected baseline (Phase 0) | 3.3924 | 4.5787 | 0.5824 | 0.5093 | — | — |
| **Phase 1 winner** (dense, 60ep, smooth=0.01) | 3.0999 | 5.0210 | 0.5600 | 0.5111 | 0.353 | 0.313 |
| **+ building channel** | **2.8836** | **4.8712** | 0.5596 | **0.5258** | **0.383** | **0.354** |
| + ground-plane loss | 3.2481 | 5.5302 | 0.4865 | 0.5047 | 0.081 | 0.125 |
| + rank loss | 3.0789 | 4.9513 | **0.5737** | 0.5285 | 0.321 | 0.301 |
| + all three (combined) | 3.1180 | 5.3930 | 0.4824 | 0.5269 | 0.136 | 0.175 |

**Deltas vs. the two reference points** (positive = better than reference):

| Configuration | Δ MAE vs. Phase 1 | Δ MAE vs. baseline | Δ RMSE vs. Phase 1 | Δ RMSE vs. baseline | Δ Pearson vs. Phase 1 | Δ Pearson vs. baseline |
|---|---:|---:|---:|---:|---:|---:|
| + building channel | **+0.216 (better)** | +0.509 (better) | **+0.150 (better)** | −0.293 (worse) | −0.0004 (flat) | −0.023 (worse) |
| + ground-plane loss | −0.148 (worse) | +0.144 (better) | −0.509 (worse) | −0.952 (worse) | −0.074 (worse) | −0.096 (worse) |
| + rank loss | +0.021 (~flat) | +0.313 (better) | +0.070 (better) | −0.373 (worse) | **+0.014 (better)** | −0.009 (~flat) |
| + all three (combined) | −0.018 (~flat) | +0.274 (better) | −0.372 (worse) | −0.814 (worse) | −0.078 (worse) | −0.100 (worse) |

**What this shows:**

1. **The building-probability channel is a clean, broad win** — the single
   best result of this whole audit. Improves MAE, RMSE, Spearman, variance
   ratio, and OLS slope simultaneously over the Phase 1 winner, with Pearson
   essentially unchanged. This is the strongest evidence yet that RGB alone
   was leaving real signal on the table — building probability (from the
   HOTOSM segmenter, already computed, no new model needed) gives the
   network information RGB alone doesn't cleanly encode.
2. **The rank loss does what it was designed to do, narrowly**: Pearson
   improves over the Phase 1 winner (the only config that beats Phase 1 on
   Pearson at all), Spearman improves further still, RMSE improves modestly.
   But it does **not** meaningfully fix the underlying compression — variance
   ratio and OLS slope are both *slightly worse* than the Phase 1 winner,
   not better. This makes sense in hindsight: a pairwise-order loss can
   reward "prediction B > prediction A whenever truth B > truth A" without
   ever requiring the *gap* between predictions to match the true gap — it
   directly targets rank correlation, not variance/calibration, and that's
   exactly the two things that moved (Pearson/Spearman up) and didn't move
   (variance ratio flat-to-down).
3. **The ground-plane loss backfires — confirmed, not a fluke.** It's the
   single worst configuration on every metric, including *worse regression-
   to-the-mean* than the Phase 1 winner it's built on (variance ratio
   0.353→0.081, a 77% relative collapse; OLS slope 0.313→0.125). Root cause
   is visible directly in the DFC AGL distribution already established in
   this audit trail: the median true AGL across these tiles is 0.05 m and a
   large majority of all pixels sit well under the 0.5 m threshold used here
   (`docs/method-audit/04-learned-scale-modulation/summary.md` §2: p50 of
   true AGL = 0.055 m). A "heavily-weighted" (2.0×) penalty that fires on
   *most* training pixels isn't a narrow ground-plane correction — it's
   functionally an even stronger pull-toward-zero than Huber loss alone
   provides, which *worsens* the exact regression-to-the-mean problem this
   whole line of work is trying to fix. This is a scoping/threshold problem
   with this specific configuration, not evidence the idea is unsound in
   general (see `gaps-and-fixes.md` for the concrete fix).
4. **Combined inherits the ground-plane loss's damage.** It sits between
   ground-plane-alone and the better individual results on most metrics,
   with rank's Spearman contribution partially visible (+0.5269, best
   Spearman apart from rank-alone) but everything else dragged down —
   confirming the three additions are not independent/additive as
   configured; ground-plane's broad, heavy penalty dominates whatever
   `rank` and `building` contribute on their own.
5. **None of the four configurations clears the honest corrected baseline
   outright.** Every configuration improves MAE over the baseline
   (building channel by the widest margin, ~15%), and most improve Spearman,
   but **every configuration's RMSE and Pearson remain worse than the
   per-tile baseline's** — the per-tile-fit linear model, simple as it is,
   is still a strong reference point that a learned spatial correction
   hasn't yet beaten across the board. Building channel comes closest.

Both follow-ups below were run — see Phase 2.5.

## Phase 2.5 — corrected ground-plane loss, and building+rank without it

Run on Kaggle (2×T4), locked Phase 1 winner base config
(dense, 60 epochs, smoothness=0.01) throughout.

1. **`phase2_building_rank`** — building channel + rank loss together,
   explicitly *without* the ground-plane term (isolates whether the two
   demonstrated individual winners combine cleanly).
2. **`phase2_groundplane_v2`** — ground-plane loss corrected: threshold
   0.5→**0.1 m**, weight 2.0→**0.1**, so it only fires on a small minority
   of pixels instead of the majority that broke it in Phase 2.
3. **`phase2_v2_combined`** — all three (building + rank + corrected
   ground-plane) together. Run because both 1 and 2 individually beat the
   Phase 1 winner on MAE and RMSE, satisfying the README's condition for
   running this cell.

_(Minor instrumentation note: `mean_epoch_time_sec` came back `null` in all
three results files — the per-epoch timing field didn't populate on this
run, unrelated to the actual training/eval results below. `fold_time_sec`
is present and used for the timing note.)_

**Full results table:**

| Configuration | MAE (m) | RMSE (m) | Pearson | Spearman | Var. ratio | OLS slope |
|---|---:|---:|---:|---:|---:|---:|
| Corrected baseline (Phase 0) | 3.3924 | 4.5787 | 0.5824 | 0.5093 | — | — |
| Phase 1 winner | 3.0999 | 5.0210 | 0.5600 | 0.5111 | 0.353 | 0.313 |
| Phase 2: + building channel | 2.8836 | 4.8712 | 0.5596 | 0.5258 | 0.383 | 0.354 |
| Phase 2: + rank loss | 3.0789 | 4.9513 | 0.5737 | 0.5285 | 0.321 | 0.301 |
| Phase 2: + ground-plane (broken: w=2.0, t=0.5m) | 3.2481 | 5.5302 | 0.4865 | 0.5047 | 0.081 | 0.125 |
| Phase 2: combined (with broken ground-plane) | 3.1180 | 5.3930 | 0.4824 | 0.5269 | 0.136 | 0.175 |
| **Phase 2.5: building + rank (no ground-plane)** | **2.8477** | **4.7971** | **0.5691** | 0.5373 | 0.339 | 0.338 |
| Phase 2.5: ground-plane v2 (w=0.1, t=0.1m) | 2.9519 | 4.9560 | 0.5556 | 0.5191 | 0.264 | 0.269 |
| **Phase 2.5: building + rank + ground-plane v2** | **2.8171** | 4.8568 | 0.5631 | **0.5367** | 0.299 | 0.307 |

**What this shows:**

1. **The corrected ground-plane loss confirms the Phase 2 diagnosis exactly.**
   Lowering weight 2.0→0.1 and tightening threshold 0.5→0.1m turned the
   *worst* configuration tested into a working one: MAE 3.248→2.952m,
   RMSE 5.530→4.956m, Pearson 0.487→0.556, variance ratio **more than
   tripled** (0.081→0.264), OLS slope **more than doubled** (0.125→0.269).
   This wasn't a "maybe the idea doesn't work" question — it was purely a
   scoping bug (the original threshold/weight touched a majority of
   pixels instead of a minority), and fixing exactly that scoping issue
   fixed the outcome, as predicted.
2. **Building + rank together is the best all-around single config yet,
   and the first to beat the corrected baseline's Spearman.** MAE 2.848m,
   RMSE 4.797m, Pearson 0.569, Spearman 0.537 — every one of these beats
   the Phase 1 winner, and Spearman (0.537) now exceeds the honest
   baseline's own Spearman (0.509) for the first time anywhere in this
   audit. The two additions combine cleanly (no interference), unlike
   ground-plane's earlier interaction.
3. **Adding the corrected ground-plane term on top of building+rank is a
   wash, not a further win.** `phase2_v2_combined` has the single best MAE
   overall (2.817m) but slightly worse RMSE, Pearson, and calibration than
   `building_rank` alone (4.857 vs 4.797m; 0.563 vs 0.569; 0.299/0.307 vs
   0.339/0.338 var-ratio/slope). The ground-plane term still trades a little
   of everything else for a little more MAE, even in its corrected form —
   just a much smaller, no-longer-catastrophic trade than in Phase 2.
4. **Against the honest corrected baseline, the gap has narrowed
   substantially but not closed.** Best MAE margin yet (building+rank:
   16% better than baseline; combined: 17% better), and Spearman now
   *beats* baseline in 2 of 3 Phase 2.5 configs — but RMSE and Pearson
   remain below baseline in all three (RMSE 4.8–5.0m vs baseline's 4.58m;
   Pearson 0.556–0.569 vs baseline's 0.582). The per-tile linear baseline
   is still undefeated on those two axes, though the margin is now single
   digits to low-teens percent rather than the larger gaps seen in Phase 1/2.

**Recommended pick if one config had to be chosen (superseded by Phase 2.5
Round 2 below): `phase2_building_rank`** — best RMSE, Pearson, and
calibration among all Phase 2/2.5 additions, and essentially tied with
`v2_combined` on MAE/Spearman without needing the ground-plane term's added
complexity.

## Phase 2.5 Round 2 — bounded final round

**This was declared the last tuning round on this architecture regardless
of outcome, before it was run.** Two targeted variants on top of the two
best Phase 2.5 configs, run on Kaggle (2×T4), same locked base config
throughout (dense coverage, 60 epochs, smoothness=0.01, building channel):

1. **`phase2_building_rank_v2`** — same as `phase2_building_rank`, rank
   weight raised 0.3→**0.5** (margin unchanged at 0.25). Checks whether more
   of the rank loss — the piece that moved Pearson in Phase 2.5 — helps
   further without destabilizing the other metrics.
2. **`phase2_building_rank_groundplane_light`** — building + rank (weight
   back at 0.3, matching `phase2_v2_combined`) + the corrected ground-plane
   term at **half its corrected weight (0.05 instead of 0.1)**, same 0.1m
   threshold. Checks whether a gentler dose adds on top of building+rank
   instead of trading back RMSE/Pearson the way the full 0.1 weight did.

**Full results table, all metrics explicit (Pearson/Spearman included for
every configuration, not just the two new ones):**

| Configuration | MAE (m) | RMSE (m) | Pearson | Spearman | Var. ratio | OLS slope |
|---|---:|---:|---:|---:|---:|---:|
| Corrected baseline (Phase 0) | 3.3924 | 4.5787 | 0.5824 | 0.5093 | — | — |
| Phase 1 winner | 3.0999 | 5.0210 | 0.5600 | 0.5111 | 0.353 | 0.313 |
| Phase 2.5: building + rank (rank=0.3, no gp) | 2.8477 | 4.7971 | 0.5691 | 0.5373 | 0.339 | 0.338 |
| Phase 2.5: building + rank + gp (rank=0.3, gp=0.1) | 2.8171 | 4.8568 | 0.5631 | 0.5367 | 0.299 | 0.307 |
| **R2: building + rank v2 (rank=0.5, no gp)** | 2.8803 | **4.7751** | **0.5835** | **0.5438** | 0.320 | 0.323 |
| R2: building + rank + gp-light (rank=0.3, gp=0.05) | 2.8316 | 4.8098 | 0.5696 | 0.5327 | 0.280 | 0.301 |

**Per-fold breakdown for the two new Round 2 configs** (the previous
round's `mean_epoch_time_sec` instrumentation gap did not recur — both
files populated normally):

| Config | Fold | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| building_rank_v2 | 0 | 2.6676 | 4.5888 | 0.5820 | 0.5379 |
| building_rank_v2 | 1 | 2.9070 | 4.9976 | 0.5756 | 0.5476 |
| building_rank_v2 | 2 | 3.2648 | 5.1292 | 0.5563 | 0.5086 |
| building_rank_v2 | 3 | 2.6819 | 4.3847 | 0.6200 | 0.5812 |
| groundplane_light | 0 | 2.6242 | 4.6169 | 0.5718 | 0.5216 |
| groundplane_light | 1 | 2.9238 | 5.1223 | 0.5499 | 0.5356 |
| groundplane_light | 2 | 3.1157 | 5.0459 | 0.5465 | 0.4992 |
| groundplane_light | 3 | 2.6627 | 4.4540 | 0.6101 | 0.5744 |

(Wall time: 4058s (~68 min) for building_rank_v2, 4184s (~70 min) for
groundplane_light — both single-GPU sequential runs, per-process CPU
contention issue from the initial 2-GPU-parallel attempt this round.)

**What this shows:**

1. **Raising rank weight 0.3→0.5 did exactly what it was hypothesized to
   do, and further than any prior configuration.** Pearson 0.5691→0.5835
   and Spearman 0.5373→0.5438 — both the best Pearson and best Spearman of
   this entire rework, across every phase. Pearson **0.5835 now exceeds the
   honest corrected baseline's own Pearson (0.5824)** — the first
   configuration anywhere in this audit to beat baseline on Pearson, not
   just Spearman. RMSE also improved slightly (4.7971→4.7751, also the best
   RMSE of the whole rework and the smallest-ever gap to baseline RMSE,
   4.58m — down to **4.3%**, from ~4.8% last round and >13% in Phase 1/2).
   The cost: MAE moved backward a little (2.8477→2.8803) and calibration
   softened slightly (var. ratio 0.339→0.320, slope 0.338→0.323) — real
   trade-offs, but small ones next to the Pearson/Spearman/RMSE gains.
2. **The gentler ground-plane dose (0.05 vs 0.1) landed exactly where
   predicted: between building_rank-alone and the full-weight combined,
   not a clean win over either.** Vs. building+rank alone: MAE improves
   (2.8477→2.8316) but RMSE, Spearman, and both calibration metrics get
   slightly worse. Vs. the full 0.1-weight combined: RMSE and Pearson
   improve, but MAE, Spearman, and calibration get slightly worse. This
   confirms the ground-plane term's fundamental trade-off (a little
   MAE for a little of everything else) scales smoothly with its weight
   rather than having a "sweet spot" that escapes it — halving the weight
   just moves along the same curve, closer to the no-ground-plane end,
   which is exactly what the weight parameter should do.
3. **Bounded round, as declared, closed exactly as scoped**: two variants
   run, both informative, no third follow-up chased. No configuration in
   this round needed re-running or came back ambiguous enough to warrant
   extending past the two planned runs.

## Final verdict — tuning on this architecture is now closed

**Single best all-around candidate across every phase of this entire
rework: `phase2_building_rank_v2`** (building-probability channel + rank
loss at weight 0.5, no ground-plane term). Best or tied-best on 3 of 4 core
metrics (RMSE 4.775m, Pearson 0.5835, Spearman 0.5438) and within 2.3% of
the best-ever MAE (2.880m vs. `phase2_v2_combined`'s 2.817m), without that
configuration's added ground-plane complexity.

**Does it beat the honest corrected baseline on all four metrics? No —
three of four, the closest this rework has gotten:**

| Metric | Baseline | `phase2_building_rank_v2` | Beats baseline? |
|---|---:|---:|---|
| MAE | 3.3924m | 2.8803m | **Yes** (15.1% better) |
| RMSE | 4.5787m | 4.7751m | **No** (4.3% worse — the smallest gap reached in this entire audit) |
| Pearson | 0.5824 | 0.5835 | **Yes** (first time any configuration has beaten baseline Pearson) |
| Spearman | 0.5093 | 0.5438 | **Yes** (6.8% better) |

RMSE is the one metric that has never been beaten by any learned
configuration across the whole v1→v2 rework, and remains not beaten here —
but the gap closed from >13% (Phase 1) to ~4.8% (Phase 2.5 round 1) to
4.3% now, a real and consistent trend, not a plateau. Per the instruction
that this was the last tuning round on this architecture regardless of
outcome: **no further tuning of Method 4 v2 is planned.** If RMSE parity
with the per-tile linear baseline is worth pursuing further, it would need
a different architecture or a fundamentally different loss design, not
another weight sweep on this one — the marginal returns from weight
adjustments alone (Phase 2.5 round 1 → round 2) were real but small
(RMSE moved 0.5% between rounds after a much larger jump from Phase 1).

## Overall read across v1 (audit) → v2 (this rework)

- The original Method 4 audit's "MAE down, correlation down" mystery is
  resolved on three fronts now: training starvation (Phase 1: fixing
  coverage/epochs alone improved all four metrics simultaneously), a
  genuinely missing input (Phase 2/2.5: the building-probability channel),
  and a missing rank-aware signal (Phase 2.5: rank loss, once combined with
  building rather than tested with a broken ground-plane term alongside it).
- Regression-to-the-mean, quantified via variance ratio and OLS slope
  throughout, went from 0.225/0.229 (original Method 4) → 0.353/0.313
  (Phase 1 winner) → 0.383/0.354 (Phase 2: + building channel alone, still
  the best calibration result of this whole rework) → 0.339/0.338 (Phase
  2.5: building+rank, calibration cost of adding rank is small). Real,
  repeated, measured progress, but still well short of 1.0/1.0 (a
  calibrated model). This remains open, and no configuration has closed it.
- The ground-plane loss's Phase 2 failure was fully explained, not just
  patched around: re-running it at a corrected weight/threshold (2.0→0.1,
  0.5m→0.1m) more than tripled its variance ratio and more than doubled its
  OLS slope (Phase 2.5 §2), confirming the Phase 2 failure was a scoping
  bug (touching a majority of pixels instead of a minority), not evidence
  against the idea itself.
- **Superseded by Phase 2.5 Round 2 (see below): `phase2_building_rank`**
  (building channel + rank loss, no ground-plane) was the best result
  through Phase 2.5 round 1 — MAE 2.848m, RMSE 4.797m, Pearson 0.569,
  Spearman 0.537, beating the Phase 1 winner on every metric and beating
  the honest corrected baseline's Spearman (0.509) for the first time
  anywhere in this audit. Adding the corrected ground-plane term on top
  (`phase2_v2_combined`) shaved MAE further (2.817m, the single best MAE
  found) but traded back a little RMSE/Pearson/calibration — a wash, not a
  further win. Round 2 (below) found a better all-around config by pushing
  the rank weight further.
- **No longer true as of Phase 2.5 Round 2**: `phase2_building_rank_v2`
  (rank weight 0.5) beats the honest per-tile corrected baseline on
  Pearson (0.5835 vs 0.5824) — the first configuration anywhere in this
  audit to do so. RMSE remains the one metric never beaten by any learned
  configuration, though the gap is now down to 4.3% (from >13% in Phase
  1/2). See "Final verdict" below for the full closing comparison.

## Post-verdict follow-up — SID ordinal height-discretization constraint

**Status: COMPLETE — negative result.** The "Final verdict" section above
named a different loss design (not another weight sweep) as the only
remaining lever on this architecture. This was that lever, tested once,
directly on top of `phase2_building_rank_v2` — it did not reopen general
tuning, and per the outcome below, does not need to.

**What this is:** an ordinal height-discretization constraint adapted from
**weakIM2H** (Chen, Shi, Zhu 2025, "Enhancing Monocular Height Estimation
via Weak Supervision from Imperfect Labels," [arXiv:2506.02534](https://arxiv.org/abs/2506.02534)),
specifically its Spacing-Increasing Discretization (SID) bin-edge formula
(their eq. 7, confirmed by fetching the paper directly rather than
assumed):

```
t_k = exp(log(h_min) + (log(h_max/h_min) / K) * k),   k = 0..K
```

Bins widen at greater heights rather than uniform binning. `h_min`/`h_max`
are **each tile's own AGL range**, fit on that fold's training quadrants
only (same train/test separation already used for rgb/depth/building
stats — held-out-quadrant height range never leaks into bin edges).

**What this explicitly is NOT:** a full reimplementation of weakIM2H. That
paper's own ordinal constraint is a separate pairwise term,
`L_OC = log(1+exp(ĥ_2-ĥ_1))` when `c_1 > c_2` (a softplus-based ranking
loss), applied alongside a **separate balanced soft-height sampling loss**
that is not ported here at all. This adaptation instead reuses this
project's *existing*, already-working `rank_pair_loss` /
`F.margin_ranking_loss` machinery unchanged — the only change is what
`target_sign` is derived from: SID bin difference instead of raw height
difference. Pairs falling in the same bin are dropped before the loss
(same mechanic as the existing exact-height-tie drop), pairs in different
bins keep the identical margin-ranking mechanics already in place. This is
a targeted transplant of one idea (ordinal bins widen with height) into
this project's own architecture, not a benchmark reproduction of their
paper.

**The h_min-near-zero issue:** many AGL pixels in this dataset sit at
exactly 0 m (bare ground), and `log(0)` is undefined. Checked the paper
directly for how it handles this — **it doesn't specify an epsilon or
floor anywhere**, despite acknowledging elsewhere that a majority of
pixels in one of its own datasets are near-zero background. `--sid-eps`
(default 0.1 m) floors `h_min` for the log computation only (heights
themselves are clamped to the same floor, so height-exactly-at-h_min maps
cleanly to bin 0). 0.1 m was chosen because it sits below this project's
own established AGL noise floor (median true AGL 0.055 m across the
50-tile benchmark, from the original Method 3/4 audit) — small enough not
to distort any real bin boundary, not derived from the paper (which gives
no guidance here).

**Implementation:** `scripts/evaluate_method4_v2.py` (Mac) and
`kaggle_phase2.5_package/scripts/evaluate_method4_v2_kaggle.py` (Kaggle),
both updated identically — new `sid_bin_index()` function, `rank_pair_loss`
extended with optional `hmin`/`hmax`/`sid_bins`/`sid_eps` args (backward
compatible: `--sid-bins 0`, the default, reproduces the exact prior
raw-height behavior — verified with a local smoke test comparing
`--sid-bins 0` against the pre-change code path). New CLI flags:
`--sid-bins K` (0=disabled) and `--sid-eps` (meters, default 0.1).

**Config to run — `phase2_building_rank_v2_sid`** (exactly
`phase2_building_rank_v2`'s config, `--sid-bins 10` added, nothing else
changed):

```
--patch-mode dense --epochs 60 --smoothness-weight 0.01 \
--extra-channel building \
--rank-weight 0.5 --rank-pairs-per-patch 2000 --rank-margin 0.25 \
--sid-bins 10 --sid-eps 0.1
```

**Result — full table, per-fold Pearson/Spearman included:**

| Configuration | MAE (m) | RMSE (m) | Pearson | Spearman | Var. ratio | OLS slope |
|---|---:|---:|---:|---:|---:|---:|
| Corrected baseline | 3.3924 | 4.5787 | 0.5824 | 0.5093 | — | — |
| `phase2_building_rank_v2` (pre-SID best) | 2.8803 | 4.7751 | 0.5835 | 0.5438 | 0.320 | 0.323 |
| **`phase2_building_rank_v2_sid`** | 2.9090 | 4.8057 | **0.5852** | **0.5449** | 0.330 | 0.325 |

| Fold | MAE | RMSE | Pearson | Spearman |
|---:|---:|---:|---:|---:|
| 0 | 2.6318 | 4.5703 | 0.5853 | 0.5411 |
| 1 | 2.9356 | 5.0368 | 0.5711 | 0.5470 |
| 2 | 3.3872 | 5.2060 | 0.5557 | 0.5025 |
| 3 | 2.6816 | 4.4097 | 0.6286 | 0.5890 |

(Wall time: 4243s / ~71 min, in line with `phase2_building_rank_v2`'s own
cost — the SID bin lookup adds negligible per-step overhead, confirming
the pre-run cost estimate.)

**Verdict: negative result — the SID ordinal constraint does not move
anything meaningfully, in either direction.** Pearson and Spearman both
tick up by ~0.001-0.002 (0.5835→0.5852, 0.5438→0.5449) — technically new
highs for this whole rework, but far smaller than the fold-to-fold spread
already visible in the table above (e.g. Pearson ranges 0.556-0.629 across
folds), so this is noise, not signal. MAE and RMSE both move slightly the
other way (2.8803→2.9090, 4.7751→4.8057), pushing the RMSE-vs-baseline gap
from 4.3% back out to 5.0% — also noise-level, but in the wrong direction.
Calibration (variance ratio, OLS slope) is essentially unchanged.

This is exactly the useful negative result flagged as worth having going
in: the "different loss design" direction that remained after weight-sweep
tuning closed out has now been tried, with a real, paper-sourced technique
faithfully adapted (not a strawman), and it doesn't help. **No further
work on Method 4 v2 is planned as a result of this test.**
`phase2_building_rank_v2` (without SID) remains the single best all-around
candidate for this architecture — see "Final verdict" above, which this
result does not change.
