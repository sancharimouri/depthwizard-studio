# Method 6 — coarse-to-fine resolution transfer (pre-registered; 2026-09-24)

Question: the fine→coarse direction is known — Method 6 trained at DFC2019's native GSD loses modestly when its input is
degraded (seed-43 sweep, resample-back protocol, pooled Pearson 0.795 → 0.708 from 0.3 → 2.4 m; `final-comparison.md` §4).
Does the **reverse** work — train on coarse imagery (2 / 3 / 5 m), test on fine — and is the transfer symmetric?

Standing rules: R1 dated entry + commit per phase; R2 rules committed before each phase runs, post-hoc labelled;
R3 tile-level bootstrap (10,000, seed 0, 95%), paired Wilcoxon + win counts, Holm across the resolution comparisons;
R4 wall time estimated before training, long jobs nohup'd. This file is both the running log and the final write-up.

---

## 2026-09-24 — Corrections to the prompt's premises (decided before any training; the user was offline)

1. **Native GSD is 0.3 m, not 0.5 m.** This project's DFC2019 convention is `GSD_1X_M = 0.3`
   (`scripts/method6_resolution_sweep.py`); a 512 px quadrant covers **153.6 m**, not 512 m. The prompt's pixel counts
   (256 / 171 / 102 px) assumed 1 m-equivalent geometry. At the true GSD a physical resize gives
   **2 m → 77 px, 3 m → 51 px, 5 m → 31 px** (8 m → 19 px), i.e. after padding to multiples of 14, token grids of
   **6×6, 4×4, 3×3** (8 m: 2×2) — much coarser than the prompt's "~7×7 at 5 m".
2. **The existing 0.795 → 0.708 evidence used a different degradation protocol:** area-average by the factor, then
   **bilinear upsample back to 512 px** (same 37×37 token grid, blurred content), scored on the coarse block grid.
   It never reduced the pixel count.

**Default chosen (documented, not asked):** run **both** protocols, same everything else.
- **P (pixel) — primary, as the prompt specifies:** area-average the quadrant to `n = round(512·0.3/g)` px, reflect-pad
  to the next multiple of 14, predict, crop to n×n, bilinearly upsample the prediction to the native 512² grid.
- **R (resample-back) — secondary, matches the existing sweep:** area-average to n px, bilinear back to 512², pad to 518.
  Removes the tiny-token-grid confound and makes the symmetry comparison comparable with the old evidence.
- **Both score against native-resolution AGL on the native 512² grid** (the prompt's rule), so every number in this file
  is recomputed under one scoring definition; the old block-grid 0.795/0.708 are quoted for reference only.
- Evaluation GSDs: the prompt's "native 0.5 m, 1 m, 2.4 m, own training res" become **0.3 (native), 1.2, 2.4, 2, 3, 5 m**
  (+8 m if Phase 5 runs) — 1.2 m is the sweep's nearest existing level to "1 m"; every model is evaluated at all of them
  (a superset of the requested cells).

## 2026-09-24 — Phase 0: does DA-V2's ViT accept other input sizes? **PASS — natively.**

- The HF `Dinov2` backbone stores one fixed position table (1,370 = 1 + 37² tokens, learned at 518 px) and calls
  `interpolate_pos_encoding` on **every** forward (`transformers/models/dinov2/modeling_dinov2.py`, lines 57/112;
  bicubic resize to the current grid). Verified it is a real interpolation: the table produced for a 6×6 grid is **not**
  equal to a crop of the stored table. **No fix like RDAH's positional-buffer resize is needed.**
- Smoke test, Method 6 seed-42 fold-0 checkpoint: inputs 518 → 14 px (token grids 37×37 → 1×1) all run with correct
  output shapes and finite, non-constant outputs.
- Real-image check (JAX_004_006 quadrant 0, protocol P, zero-shot native model, Pearson vs native AGL on the 512 grid):
  0.3 m 0.801 · 1.2 m 0.561 · 2 m 0.050 · 2.4 m 0.066 · 3 m −0.211 · 5 m 0.002 · 8 m −0.364; prediction std falls from
  3.7 m to 0.2–0.5 m. **Descriptive, one quadrant, not a result** — but it already shows that shrinking the pixel count
  (token grid) is itself destructive for the native model, separate from GSD realism (the resample-back sweep had 0.776
  at 1.2 m). This is why protocol R runs alongside P.
- **Risk flagged (prompt's request):** at 3 m / 5 m the P input is a 4×4 / 3×3 token grid. That is a genuine
  architecture risk — a ViT sees 9–16 tokens for a 154 m scene — independent of whether 5 m imagery "contains" height.
  A P-protocol failure at 3–5 m cannot distinguish "no information" from "too few tokens"; protocol R can.

## 2026-09-24 — Phases 1–4 pre-registration (committed before any training)

**Phase 1 (protocols):** P and R as above; area averaging = `torch.nn.functional.interpolate(mode="area")` (adaptive
average pooling; exact for integer factors, area-weighted for 6.67× etc.). AGL never resampled.

**Phase 2 (training):** for each g ∈ {2, 3, 5} m and each protocol ∈ {P, R}: Method 6's full adopted recipe, unchanged
(`scripts/evaluate_method6_gsd_film_height_balanced.py --enable-height-balanced`: twin head, Gaussian NLL after 30
Huber warm-up steps + 0.35 × capped height-weighted Huber, height-balanced WeightedRandomSampler, 12 epochs, batch 2,
LR 5e-6 / 2.5e-4, warm-up + linear decay, grad clip 1.0, `height_scale` = p95 of the fold's training AGL),
**seed 42**, same 4-fold quadrant split (`evaluate_method4.quadrant_bounds`, bit-identical). The only change: the input
transform, and the prediction is upsampled to 512² before the loss (loss against native AGL on the native grid).
- **Recipe-identity gate:** the new script run at g = 0.3 m (identity transform) for fold 0 must reproduce the adopted
  fold-0 MAE **1.8834942542525481** exactly (bit-identical training was verified 2026-09-23). If not, stop and debug
  before any coarse training.
- Wall time (estimate, measured before committing to the grid): P at 2–5 m is a few minutes per fold (tiny token
  grids); R costs about a native fold (~11–13 min). 3 GSDs × 4 folds × 2 protocols ≈ 2.5–3 h, plus ~15 min evaluation.

**Phase 3 (matrix):** models = native Method 6 (seed-42 folds, adopted) + the 6 new ones (3 GSD × 2 protocols); each
fold model evaluated on its own held-out quadrant (so every tile is scored once per model, out-of-fold) at every
evaluation GSD {0.3, 1.2, 2, 2.4, 3, 5} under **its own protocol** (native model under both). Metrics per quadrant:
MAE, RMSE, Pearson, Spearman, variance ratio; tile value = mean over its 4 quadrants (each held out once);
headline = mean of 50 tiles; 95% tile-bootstrap CI. **Oracle per-tile-OLS recomputed at every evaluation GSD and
protocol:** frozen DAv2-Large via the demo engine on the whole degraded tile, OLS on the tile's 3 other quadrants,
scored on the held-out quadrant (same definition as §1.0 of `final-comparison.md`, only the input degraded).

**Phase 4 — decision rules** (per protocol; **P is the headline**):
1. **"Coarse-to-fine transfer works" for training GSD g** if, evaluated at native 0.3 m, the g-trained model beats the
   0.3 m oracle on **≥ 3 of 4** metrics (MAE, RMSE lower; Pearson, Spearman higher) with **non-overlapping** 95%
   tile-bootstrap CIs. Paired Wilcoxon per metric with Holm across the 3 values of g reported alongside.
2. **Symmetry test:** for each g, paired over 50 tiles, compare
   (g-trained model evaluated at 0.3 m) vs (native model evaluated at g). Primary metric Pearson (all 4 reported);
   Wilcoxon, Holm across the 3 g. **"Asymmetric"** if Holm p < 0.05, with the sign saying which direction transfers
   better; otherwise "no evidence of asymmetry" (not "symmetric").
3. **Dose-response shape** (descriptive, rule fixed now): the native-eval Pearson and MAE of models trained at
   0.3 (native reference) / 2 / 3 / 5 m (/ 8 m). **Monotonic** if every coarser training GSD is worse than the previous
   on Pearson; a **cliff** at step k if that single adjacent drop is ≥ 60% of the total 0.3 → 5 m drop. The full curve
   is reported either way.

**Phase 5 (optional):** 8 m (P: 19 px, 2×2 tokens; R) only after Phases 0–4 are written up and committed; it never
blocks the final step.
