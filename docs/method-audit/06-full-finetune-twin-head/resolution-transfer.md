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

## 2026-09-24 — Phase 2 recipe-identity gate: **PASS (bit-identical)**

`scripts/method6_resolution_transfer.py train --gsd 0.3 --proto P --folds 0` (identity transform) reproduces the adopted
seed-42 fold-0 result exactly: MAE **1.8834942542525481**, RMSE 3.413497563855602, Pearson 0.7572194755073397,
Spearman 0.6599485918985362, height_scale 16.615135192871094; 714 s. The new script's training path is therefore the
adopted recipe; the coarse runs differ only by the input transform. Measured native fold time 714 s ⇒ protocol R
≈ 12 min/fold ⇒ 3 GSD × 4 folds ≈ 2.4 h; protocol P measured on its first fold (logged below). Launched as one
nohup chain: P 2/3/5 m → R 2/3/5 m → Phase 3 eval → Phase 4 analysis
(`data/dfc2019/experiments/resolution_transfer/chain.log`).
- Measured P-protocol fold time (2 m, fold 0): **98 s** ⇒ P grid ≈ 18 min; chain total estimate ≈ 2.8–3 h
  (R dominates), expected completion ≈ 03:30 IST.

## 2026-09-24 — Phase 2 training record

All 24 fold trainings completed (chain 00:45–03:46 IST): P 2/3/5 m ≈ 92–98 s/fold; R 2/3/5 m ≈ 794–814 s/fold.
Own-GSD fold results are in `data/dfc2019/experiments/resolution_transfer/rt_{P,R}_{2,3,5}m_seed42/train_results.json`
(checkpoints not committed, ~100 MB each). One incident: the Phase 4 analysis step crashed on a metric-key name
(`compute_metrics` returns `mae_m`/`rmse_m`); fixed by renaming at load, re-run — no number changed (the 12,000 matrix
records are the raw output of Phase 3).

**Sanity anchors (reproduce known numbers under the new scoring):** native Method 6 at 0.3 m = **1.98 / 3.49 / 0.745 /
0.656** (the adopted headline); recomputed oracle at 0.3 m = **3.39 / 4.58 / 0.582 / 0.509** (exactly §1.0's oracle).

## 2026-09-24 — Phase 3: the full matrix (50 tiles; tile = mean of its 4 held-out quadrants; mean of tiles)

Cell = MAE / RMSE / Pearson / Spearman (m, m, r, ρ). Rows = training GSD (native = 0.3 m); columns = evaluation GSD.
95% tile-bootstrap CIs for every cell: `data/dfc2019/experiments/resolution_transfer/matrix_summary.json`.

**Protocol P (pixel count shrinks — primary):**

| train ↓ / eval → | 0.3 m | 1.2 m | 2 m | 2.4 m | 3 m | 5 m |
|---|---|---|---|---|---|---|
| native 0.3 | **1.98/3.49/0.745/0.656** | 3.79/5.79/0.187/0.174 | 4.08/5.89/0.078/0.063 | 4.19/5.94/0.047/0.037 | 4.24/5.96/0.046/0.051 | 4.29/5.91/0.020/0.022 |
| P 2 m | 3.76/6.12/0.420/0.453 | 3.06/5.04/0.519/0.499 | *3.01/4.87/0.493/0.468* | 3.11/4.98/0.455/0.431 | 3.20/5.07/0.402/0.389 | 3.56/5.45/0.272/0.287 |
| P 3 m | 3.95/6.37/0.203/0.252 | 3.46/5.68/0.392/0.403 | 3.28/5.31/0.422/0.421 | 3.27/5.25/0.415/0.410 | *3.26/5.18/0.406/0.399* | 3.61/5.49/0.292/0.299 |
| P 5 m | 4.19/6.72/0.173/0.228 | 4.02/6.30/0.244/0.288 | 3.80/5.95/0.281/0.308 | 3.73/5.81/0.293/0.310 | 3.67/5.67/0.313/0.319 | *3.62/5.53/0.318/0.314* |
| oracle (DAv2-L, per-tile OLS) | 3.39/4.58/0.582/0.509 | 3.46/4.65/0.566/0.495 | 3.50/4.73/0.538/0.472 | 3.53/4.78/0.521/0.461 | 3.64/4.91/0.485/0.433 | 3.82/5.16/0.387/0.352 |

**Protocol R (resample-back: same 37×37 token grid, degraded content — secondary):**

| train ↓ / eval → | 0.3 m | 1.2 m | 2 m | 2.4 m | 3 m | 5 m |
|---|---|---|---|---|---|---|
| native 0.3 | **1.98/3.49/0.745/0.656** | 2.12/3.76/0.707/0.629 | 2.45/4.26/0.625/0.579 | 2.57/4.41/0.601/0.561 | 2.80/4.72/0.544/0.514 | 3.52/5.54/0.321/0.323 |
| R 2 m | 2.19/3.77/0.721/0.642 | 2.18/3.76/0.718/0.636 | *2.20/3.79/0.706/0.630* | 2.24/3.86/0.693/0.622 | 2.31/3.98/0.675/0.608 | 2.67/4.56/0.580/0.549 |
| R 3 m | 2.26/3.89/0.708/0.630 | 2.23/3.85/0.705/0.626 | 2.25/3.86/0.696/0.622 | 2.26/3.88/0.689/0.617 | *2.29/3.92/0.678/0.609* | 2.51/4.26/0.618/0.572 |
| R 5 m | 2.44/4.10/0.687/0.624 | 2.38/4.06/0.686/0.620 | 2.37/4.04/0.682/0.617 | 2.37/4.03/0.678/0.614 | 2.35/4.00/0.673/0.610 | *2.41/4.07/0.645/0.588* |
| oracle | 3.39/4.58/0.582/0.509 | 3.47/4.67/0.561/0.490 | 3.53/4.77/0.531/0.468 | 3.53/4.79/0.509/0.452 | 3.65/4.94/0.481/0.430 | 3.85/5.21/0.379/0.350 |

(*Italic* = self-consistency cell, the model at its own training GSD.)

**Variance ratio.** The pre-registered mean-of-tiles var(pred)/var(true) is **not interpretable** here (cells of 3–640):
near-flat quadrants (var(true) ≈ 0) blow it up — the same failure noted in 07. **Post-hoc**, the median-of-tiles ratio
(rows as above; columns 0.3 / 1.2 / 2 / 2.4 / 3 / 5 m):
P — native 0.80 0.03 0.02 0.02 0.01 0.00 · P2 1.40 0.87 0.52 0.38 0.26 0.17 · P3 0.75 0.85 0.63 0.62 0.47 0.34 ·
P5 0.99 0.88 0.73 0.71 0.60 0.41 · oracle 0.24 0.21 0.23 0.24 0.20 0.09.
R — native 0.80 0.62 0.48 0.45 0.38 0.29 · R2 0.81 0.75 0.73 0.69 0.62 0.39 · R3 0.81 0.81 0.73 0.72 0.68 0.56 ·
R5 0.78 0.80 0.73 0.71 0.69 0.61 · oracle 0.24 0.20 0.22 0.19 0.15 0.08.
(Under P the native model's output collapses to near-constant once the pixel count shrinks: median ratio ≤ 0.03.)

## 2026-09-24 — Phase 4: pre-registered comparisons

**1. "Coarse-to-fine transfer works" (at 0.3 m, beats the 0.3 m oracle on ≥ 3/4 metrics, non-overlapping CIs).**

| model | MAE | RMSE | Pearson | Spearman | metrics won (CI-separated) | verdict |
|---|---|---|---|---|---|---|
| **P 2 m** | 3.76 vs 3.39 (worse) | 6.12 vs 4.58 (worse) | 0.420 vs 0.582 (worse) | 0.453 vs 0.509 (worse) | 0 | **FAILS** |
| **P 3 m** | 3.95 (worse) | 6.37 (worse) | 0.203 (worse) | 0.252 (worse) | 0 | **FAILS** |
| **P 5 m** | 4.19 (worse) | 6.72 (worse) | 0.173 (worse) | 0.228 (worse) | 0 | **FAILS** |
| R 2 m | **2.19** (49/50 tiles) | 3.77 (CIs overlap; 43/50) | **0.721** (44/50) | **0.642** (44/50) | 3 | **WORKS** |
| R 3 m | **2.26** (48/50) | 3.89 (overlap; 41/50) | **0.708** (45/50) | **0.630** (44/50) | 3 | **WORKS** |
| R 5 m | **2.44** (44/50) | 4.10 (overlap; 38/50) | **0.687** (38/50) | **0.624** (42/50) | 3 | **WORKS** |

All 24 per-metric Wilcoxon tests have Holm p ≤ 0.011 (in the direction shown).

**2. Symmetry (g-trained evaluated at 0.3 m vs native evaluated at g; paired over 50 tiles; primary Pearson, Holm across g).**

| g | P: coarse→fine vs fine→coarse Pearson (Δ [95% CI]) | P verdict | R: c→f vs f→c Pearson (Δ [CI]) | R verdict |
|---|---|---|---|---|
| 2 m | 0.420 vs 0.078 (+0.34 [0.29, 0.40]) | asymmetric, c→f better (p_Holm 9e-13) | 0.721 vs 0.625 (+0.10 [0.08, 0.12]) | asymmetric, c→f better (5e-13) |
| 3 m | 0.203 vs 0.046 (+0.16 [0.11, 0.21]) | asymmetric, c→f better (1e-7) | 0.708 vs 0.544 (+0.17 [0.13, 0.20]) | asymmetric, c→f better (2e-13) |
| 5 m | 0.173 vs 0.020 (+0.15 [0.11, 0.19]) | asymmetric, c→f better (6e-9) | 0.687 vs 0.321 (+0.37 [0.31, 0.43]) | asymmetric, c→f better (5e-14) |

Under R, coarse-trained models are also better on MAE and RMSE in the c→f direction at every g (CIs exclude 0).
Under P, c→f is better on MAE at 2 and 3 m (at 5 m the CI includes 0: [−0.23, +0.02]) but **worse on RMSE** at 3 and 5 m (Δ RMSE +0.42 [0.27, 0.56] and +0.81 [0.66, 0.97]):
both P directions are poor, and neither beats the oracle.

**3. Dose-response (native-eval quality vs training GSD 0.3 → 2 → 3 → 5 m).**
- **P:** Pearson 0.745 → 0.420 → 0.203 → 0.173; MAE 1.98 → 3.76 → 3.95 → 4.19. **Monotonic.** No step reaches 60% of the
  total drop (the 0.3 → 2 m step is 57%), so by the pre-registered rule **no cliff** — but the shape is front-loaded:
  most of the loss happens by 2 m (a 6×6 token grid).
- **R:** Pearson 0.745 → 0.721 → 0.708 → 0.687; MAE 1.98 → 2.19 → 2.26 → 2.44. **Monotonic, gentle, no cliff** — a
  7.7% total Pearson loss from 0.3 to 5 m training.

### Reading (what the pre-registered results say)

- **With the prompt's protocol (P: coarse imagery at its true pixel count), coarse-to-fine transfer fails at every
  GSD tested** — none of the 2/3/5 m models beats the 0.3 m oracle on a single metric. The native model evaluated on
  coarse P input collapses entirely (Pearson ≤ 0.19, near-constant output). So under P both directions fail; c→f is
  the less bad one on correlation.
- **With the same content degradation but the token grid held at 37×37 (R), coarse-to-fine transfer works at
  2, 3 and 5 m** (3/4 metrics vs the oracle; RMSE better on the mean but CI-overlapping), and it is **strongly
  asymmetric in favour of coarse→fine**: a model trained on 5 m-content imagery loses only 0.058 Pearson on sharp
  0.3 m input, while the native model loses 0.424 on 5 m-content input. The R5 model is nearly flat across evaluation
  GSDs (Pearson 0.687 → 0.645 from 0.3 to 5 m).
- **The P-vs-R contrast is the main finding:** at this scene size (153.6 m), what kills the P models is the **pixel
  / token count** (6×6 → 3×3 tokens), not the loss of ground detail — R shows the height signal in 5 m-content
  imagery is still learnable and transfers to fine imagery.
- **Limits:** synthetic degradation of one VHR sensor (WorldView-3, JAX; no real 2–5 m sensor, MTF, atmosphere or
  view-geometry differences); 50 tiles in one city; one seed per cell; scene extent fixed at 153.6 m, so "5 m" here is
  31 px of content. **This does not show that real 10 m Sentinel-2 imagery supports Method 6-style height estimation**;
  it shows the architecture can learn and transfer from coarse-content inputs when given enough tokens.

## 2026-09-24 — Phase 5 (optional) launched after Phases 0–4 were committed

8 m training for both protocols (P: 19 px, 2×2 tokens; R: 37×37 tokens), same recipe/seed/split; the matrix is extended
with an 8 m evaluation column for every model and 8 m rows; the oracle is computed at 8 m. The pre-registered
dose-response rule stays on 0.3 → 5 m; 8 m is reported as an extended curve (descriptive). Estimate ≈ 6 min (P) +
54 min (R) + ≈ 20 min eval ⇒ ≈ 80 min. Chain: `data/dfc2019/experiments/resolution_transfer/run_chain8.sh`.

## 2026-09-24 — Phase 5 result (optional, 8 m; descriptive extension)

Training: P 8 m 92–96 s/fold, R 8 m 778–807 s/fold (chain 04:14–05:23 IST, `chain8.log`). The matrix now also has
an 8 m evaluation column (every model) and 8 m rows; `matrix_summary.json` was regenerated including 8 m.
**Bookkeeping:** the regenerated JSON applies Holm across 4 GSDs (2/3/5/8); the Phase 4 section above keeps the
pre-registered 3-GSD values from the first analysis run. No 2/3/5 m verdict changes (re-checked: P2/P3/P5 fail,
R2/R3/R5 work, all asymmetric c→f better).

| cell | P protocol (MAE/RMSE/r/ρ) | R protocol |
|---|---|---|
| 8 m-trained, eval 0.3 m | 4.64/6.98/0.172/0.209 | 3.06/4.65/**0.631**/0.585 |
| 8 m-trained, eval 8 m (self) | 3.84/5.72/0.219/0.222 | 2.78/4.53/**0.557**/0.520 |
| native, eval 8 m | 4.29/5.85/−0.031/−0.019 | 4.01/5.99/0.169/0.166 |
| 5 m-trained, eval 8 m | 3.76/5.65/0.208/0.227 | 2.75/4.61/0.549/0.518 |
| oracle, eval 0.3 m / 8 m | 3.39/4.58/0.582/0.509 · 3.97/5.35/0.214/0.207 | same at 0.3 m · 4.01/5.40/0.227/0.218 |

- **Rule 1 at 8 m (descriptive):** P8 fails (0/4). **R8 also fails (0/4 CI-separated):** better than the oracle on the
  mean for MAE (3.06 vs 3.39, 33/50 tiles), Pearson (0.631 vs 0.582, 26/50) and Spearman (0.585 vs 0.509, 37/50), but no
  CI separation; RMSE ≈ equal (24/50).
- **Symmetry at 8 m:** asymmetric, c→f better in both protocols (R: 0.631 vs 0.169, Δ +0.46 [0.40, 0.53]).
- **Extended dose-response (native-eval Pearson; training 0.3 / 2 / 3 / 5 / 8 m):**
  P 0.745 / 0.420 / 0.203 / 0.173 / 0.172 (MAE 1.98 / 3.76 / 3.95 / 4.19 / 4.64);
  R 0.745 / 0.721 / 0.708 / 0.687 / **0.631** (MAE 1.98 / 2.19 / 2.26 / 2.44 / **3.06**).
  Under R the 5 → 8 m step (−0.056 Pearson, +0.63 m MAE) is the largest single step — about half of the whole
  0.3 → 8 m Pearson loss — i.e. **the R curve starts to bend between 5 and 8 m** (descriptive; the pre-registered cliff
  rule applies to 0.3 → 5 m only, where there is none).
- **Reading, with the same limits as Phase 4:** under the token-preserving protocol the synthetic 8 m model still
  carries a real height signal (0.557 at its own GSD vs 0.227 for the oracle and 0.169 for the native model), but it no
  longer clears the oracle bar at native resolution, and quality degrades faster beyond 5 m. That is where "the real
  cliff" starts to show in this synthetic setting — still short of Sentinel-2's 10 m, and still WorldView-3 content,
  one city, one seed.
