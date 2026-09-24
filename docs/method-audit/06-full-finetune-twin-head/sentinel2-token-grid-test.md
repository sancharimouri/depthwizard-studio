# Does protocol R's token-grid finding transfer to real Sentinel-2? (pre-registered; 2026-09-24)

Question: `resolution-transfer.md` found that coarse-trained Method 6 transfers to fine imagery when the ViT token grid
is kept (protocol R) and fails when the input has its true pixel count (protocol P). Does that help on real Sentinel-2?
Or was an R-like geometry already tried there and already failed?

Standing rules (as in `resolution-transfer.md`):
- **R1:** a dated entry and a commit per phase.
- **R2:** pass/fail criteria are committed before each phase runs; anything judged after seeing results is labelled
  post-hoc.
- **R3:** tile-level bootstrap (10,000, seed 0, 95%), paired Wilcoxon, Holm correction.
- **R4:** wall time is estimated before training, and long jobs are nohup'd.

This file is both the running log and the final write-up.

---

## 2026-09-24 — Housekeeping: DFC2019 native GSD, re-derived from tile content

**The two prior claims:**
- 0.3 m (`GSD_1X_M = 0.3` in `scripts/method6_resolution_{sweep,transfer}.py`).
- 0.5 m (`stage0-gates/sparse-lidar-feasibility.md` §6: US3D point-cloud extents of 512 m ÷ 1024 px, 37/37 tiles).

**Neither was derived from our tiles.**
- 0.3 m is a code convention.
- 0.5 m divides a point cloud's extent by *our* tile's pixel count. That assumes the point cloud and the tile share a
  footprint, and the same document's §7–§8 retracted exactly that correspondence (0/50 tiles matched in a
  20,176-correlation spatial join).

**No geotransform to read.** Re-checked with rasterio on `JAX_004_006_{RGB,AGL,CLS}.tif`: CRS `None`, identity
transform, TIFF resolution tags `1 (unitless)`, written by `tifffile.py`. The tiles carry no georeferencing, so the GSD
was measured from objects of standardised physical size in the imagery instead. Crops were checked visually at 3–4×
with a pixel ruler.

| Object (tile) | Standard size | Measured | Implied GSD |
|---|---|---|---|
| Interstate lane-line cycle, 10 ft dash + 30 ft gap (JAX_004_006) | 40 ft = 12.19 m | 6 dashes, period 36.6 px along the road (36.8 px corrected for the 6.7° tilt) | **0.33 m/px** |
| Two-lane carriageway, edge line to edge line (JAX_004_006) | 2 × 12 ft = 7.32 m | 21.4 px perpendicular | **0.34 m/px** |
| Tractor + 53 ft trailer, 4 rigs along their axis (OMA_376_038) | ≈ 21–23 m overall | 67, 68, 73, 77 px (mean 71) | **0.30–0.32 m/px** |

**Verdict: the native GSD is ≈ 0.3 m (0.30–0.34). 0.5 m is wrong for our tiles.** At 0.5 m the same objects would be:
- a 10.7 m two-lane carriageway;
- an 18.4 m dash cycle;
- 35 m truck rigs.

None of those is a real standard. The range is consistent with WorldView-3's ~0.31 m nadir GSD plus off-nadir
stretch.

**Corrections:**
1. `GSD_1X_M = 0.3` stays.
2. `sparse-lidar-feasibility.md` gets a dated correction. Its 0.5 m (512 m ÷ 1024 px) is not our tiles' GSD. Its §8
   spatial join also rasterised the point clouds at 0.5 m/px, i.e. ~1.6× the wrong scale. That join's 0/50 negative is
   therefore **not a clean negative**, because a scale mismatch alone could suppress relief correlation. Re-running it
   at 0.3 m needs the point-cloud zips (in `~/Downloads`, not readable from this session). It is not re-run here and is
   left as a flagged open item.

**Effect on resolution-transfer labels: not material.** The labels are 0.3 m × factor. If the true base is 0.30–0.34 m,
the "2 / 3 / 5 / 8 m" rows are really ≈ 2.0–2.3 / 3.0–3.4 / 5.0–5.7 / 8.0–9.1 m. Every comparison in that file is
between factors on one base, so no ordering or verdict changes. All GSD labels below keep the 0.3 m convention.

## 2026-09-24 — Phase A: geometric audit of every prior Sentinel-2 CNN attempt

The token grid is set by the image a ViT receives: `ceil(H/14) × ceil(W/14)` tokens after resizing or padding.

**What "R-like" means.** On DFC2019, protocol R changed two things relative to P at the same time:
1. **Token count:** R has 37×37; P has ceil(n/14)² with n = round(512·0.3/g), i.e. 2×2 at 8–10 m.
2. **Real pixels per token side:** R has 4.2/g, i.e. 0.42 at 10 m, because R *upsamples*; P has 14.

Both protocols cover the same 153.6 m scene. The user's pre-registered classification axis is (1), token count. Axis
(2) is reported alongside, because the two axes disagree for the prior attempts.

### Reference geometry (DFC2019, 0.3 m, confirmed above)

| Protocol | Input to ViT | Token grid | Metres per token | Real px per token side | Scene |
|---|---|---|---|---|---|
| Native | 512 px quadrant → pad 518 | 37×37 = 1,369 | 14 × 0.3 = 4.2 m | 14 | 153.6 m |
| P at g | n = round(153.6/g) px → pad to next ×14 | ceil(n/14)²: 2 m **6×6**, 3 m **4×4**, 5 m **3×3**, 8 m **2×2**, 10 m (n = 15) **2×2** | 14·g m (10 m: 140 m) | 14 | 153.6 m |
| R at g | n px → bilinear back to 512 → pad 518 | **37×37 = 1,369** at every g | 4.2 m | 4.2/g (10 m: **0.42**) | 153.6 m |

### The prior Sentinel-2 attempts (10 m native; tiles 1000 × 1000 px = 10 km)

The user named three attempts: A1–A3. All three are **Method 4**: a frozen-feature CNN correction, not a ViT
fine-tune. A4 is the only prior attempt in which a ViT was *trained* on Sentinel-2, so it is added here. Leaving it out
would hide the most relevant evidence.

| # | Attempt (script) | ViT and what it saw | Token grid (math) | m / token | Real px / token side | What was trained, on what crop | Outcome vs ICESat-2 |
|---|---|---|---|---|---|---|---|
| A1 | Method 4, SRTM target, 10 m grid (`run_method4_sentinel2.py`) | DAv2-**Large**, **frozen**. The whole 1000 px tile was resized to 518 × 518 by `depth_engine.infer_with` (`MODEL_INPUT_SIZE = 518`), and the depth was bicubic'd back to 1000². | 518/14 = 37 → **37×37 = 1,369** | 14 × (1000/518) × 10 = **270.3 m** | 14 × 1000/518 = **27.0** | 4-layer 3×3 CNN (receptive field 9 × 9 px = 90 m), PATCH = 64 px = 640 m. **No tokens in the trained part.** | **Lost**: RMSE 69.79 vs linear 53.40 m (the memorisation signature) |
| A2 | Method 4, native 30 m (`run_method4_sentinel2_native30m.py`) | Same frozen 37×37 map as A1, area-averaged to 333 × 333 at 30 m | **37×37** (inherited) | 270.3 m | 9.0 (30 m px) | Same CNN, PATCH = 64 px at 30 m = 1,920 m, about 300 patches per fold | **Lost**: ICESat-2 wins 16/100, DEM 24/100 |
| A3 | Method 4, Open Buildings target, Test B (`run_method4_openbuildings_test_b.py`) | Same `dav2_depth/*.npy` as A1 | **37×37** | 270.3 m | 27.0 | Same CNN, 64 px = 640 m, 8 urban tiles, 360 patches | **Lost**: 2/8 wins (ICESat-2 RMSE 22.46 vs 18.58 m); DEM 7/8, i.e. target memorisation |
| A4 | Method 6 staged on Sentinel-2 (`evaluate_method6_sentinel2.py`) | DAv2-**Small**, **fully fine-tuned**. Input = one tile quadrant, 500 × 500 px at native 10 m (5 km), reflect-padded to 504 | 504/14 = 36 → **36×36 = 1,296** | 14 × 10 = **140 m** | **14** | The ViT itself, on 75 quadrants per fold. Target: SRTM on the interpolated 10 m grid, absolute elevation. | **Lost**: fold 0 ICESat-2 wins 2/25 (23 losses), DEM 1/25. Stopped per protocol. |

### Classification

**A1–A3: R-like by token count (37×37), but not tests of R.**
- The ViT was frozen in all three: it was never trained, so "train under R" never happened.
- Each token covered 27 real pixels (270 m). That is the *opposite* of R's sub-pixel tokens.
- The only trained model, a 9 × 9 px CNN, has no token grid at all.

Their failures come from the target and the CNN (interpolation memorisation, data starvation, target memorisation),
not from token geometry. They are **uninformative about R**.

**A4: R-like by token count (36×36 ≈ R's 37×37), and it failed against ICESat-2 (2/25).**
- Per the pre-registered rule, this is direct evidence against "a large token grid fixes Sentinel-2" for A4's setup: SRTM
  on an interpolated 10 m grid, absolute elevation, 25 tiles.
- A4 does not clear the confounds: it used the memorisation-prone interpolated SRTM target and 75 training quadrants
  per fold.
- A4 is **P-like on axis (2)**. Its tokens hold 14 native 10 m pixels (140 m per token), exactly P's ratio at 10 m.
  A4 is what "P with a big scene" looks like: native pixel count, large token grid only because the crop is 5 km wide.

**What no prior attempt tested:** R's *other* property, upsampling so each token covers about one real pixel or less,
with the ViT trained. So **the count-only reading of "R" was tried (A4) and failed**. The upsampling reading is
untested on real Sentinel-2. Phase C is designed to test exactly that difference.

## 2026-09-24 — Phase B pre-registration (committed before training)

**Runs.** Protocol R only, g ∈ {10, 12} m, on DFC2019:
- script `scripts/method6_resolution_transfer.py train --gsd {10,12} --proto R`, **unchanged**;
- the adopted Method 6 recipe; 4 quadrant folds; seed 42;
- n = round(153.6/g) = **15 px (10 m), 13 px (12 m)**, bilinear back to 512, pad 518 → 37×37 tokens.

Then `eval --train-gsds 10 12`, which adds R10 and R12 to the matrix plus native and oracle at 10 and 12 m (P-protocol
models are not trained at 10 or 12 m). Then `analyze`.

**Rule (identical to resolution-transfer rule 1).** "R works at g" if, evaluated at native 0.3 m, the g-trained model
beats the 0.3 m oracle (3.39 / 4.58 / 0.582 / 0.509) on **≥ 3 of 4** metrics with **non-overlapping** 95% tile-bootstrap
CIs.

**Stop condition (user's).** If R10 does not work under that rule (CIs overlap or worse than the oracle):
- report that synthetic evidence predicts real Sentinel-2 is unlikely to work under R;
- still run Phase C, with tempered expectations stated.

**Curve.** Native-eval Pearson for R at 5 / 8 / 10 / 12 m is reported as the extended curve. Descriptive only; no
threshold beyond rule 1.

**Wall time (R4).** The measured R 8 m run was 53 min for 4 folds (`chain8.log`). So R10 + R12 ≈ 1.8 h, plus eval
≈ 10–15 min ⇒ **≈ 2 h**. Launched as one nohup chain: `data/dfc2019/experiments/resolution_transfer/run_chain_b.sh`
→ `chainB.log`.

## 2026-09-24 — Phase C pre-registration (committed before the real runs; only 20-step smoke tests run so far)

**Script.** `scripts/s2_token_grid_phase_c.py` (`prep` / `train --arm {P,R}` / `analyze`). Output goes to
`data/sentinel2_benchmark/token_grid_test/`.

**The only manipulated variable is the token-grid geometry.** Held constant across the arms:
- tiles, folds, crops (identical pixels), target, model, loss;
- optimiser, LR schedule, seed, step count, batch, and the crop sampling order (same RNG seed per fold).

**Tiles.** All **32** benchmark tiles in `manifest.csv`. The flagged/unflagged split of A4 was a DAv2 sign-flip
criterion, which is irrelevant here because there is no frozen DAv2. Every tile has a 100%-finite 30 m FABDEM and
6.8k–84k ICESat-2 ground cells.

**Folds.** The 4 tile quadrants, as in every prior Sentinel-2 run. Train on 3 quadrants of every tile, evaluate on the
held-out quadrant.

**Crops.** 60 × 60 px at 10 m (600 m). 64 per quadrant (8 × 8), with origins at multiples of 3 so they align to the
30 m grid. That gives 6,144 training crops per fold.

**Target: FABDEM, interpolation-safe (native-30m construction).**
- Raw FABDEM (`fabdem_raw/*.tif`) is warped directly onto an exact 3× coarsening of the tile's 10 m grid (333 × 333 at
  30 m, bilinear at ~1:1).
- Predictions are 3 × 3-average-pooled to that grid and the loss is computed there. No target value finer than FABDEM's
  own ~30 m resolution exists anywhere.
- Checked: the warp matches the 3 × 3 block mean of the benchmark's 10 m FABDEM product. Mean difference ≈ 0 m; p95
  |d| is 0.1 m on plains and 7 m on steep Almora (bilinear vs block mean).

**What is learned.** Within-crop relief: FABDEM minus its crop mean. Absolute elevation is unidentifiable from a
600 m RGB crop. At evaluation both arms receive the crop's FABDEM mean as offset, so any difference is relief only.

**Model.** Method 6's `TwinHeadDav2`: DAv2-Small, full fine-tune, pretrained head init, `height_scale` = std of the
fold's training relief. The mu head only. Mean-removed Huber loss (β = 1 m). AdamW, LR 5e-6 / 2.5e-4, 5% warm-up +
linear decay, grad clip 1.0.

*Deviation from Method 6, stated:* no Gaussian-NLL stage and no height-balanced sampler. Those were tuned for absolute
AGL. The same loss is used in both arms.

**Arms.**

| Arm | Input to ViT | Token grid | m / token | Real px / token side |
|---|---|---|---|---|
| P | 60 px, reflect-pad → 70 | 70/14 = **5×5 = 25** | 140 m | 14 (A4's and DFC-P's ratio) |
| R | 60 px, bilinear ×9 → 540, reflect-pad → 546; output cropped to 540, 9 × 9 average-pooled to 60 | 546/14 = **39×39 = 1,521** | 15.4 m | 1.54 |

R's 540/546 replaces the 518 first coded, because MPS cannot area-pool 518 → 60. That was an implementation fix found
in the smoke test, before any result was seen.

**Evaluation, per tile, pooled over its 4 held-out quadrants:**
- **ICESat-2 check (primary, independent of the training reference).** ICESat-2 ground photons are grouped per (10 m
  cell, RGT) as the median, then converted to EGM2008 (FABDEM's datum; PROJ grid asserted). Score: RMSE of
  (crop FABDEM mean + predicted relief at 10 m) against those heights.
- **DEM held-out check (secondary).** RMSE on the 30 m grid against FABDEM.
- **Context, not tested:** a flat crop-mean baseline and raw 10 m FABDEM at the same cells. FABDEM is expected to win
  both, since it *is* the terrain product; this phase compares R with P, not with FABDEM.

**Pass rule (user's, made conservative).** "R genuinely helps" if, on the ICESat-2 check, all three hold:
- R has lower RMSE than P on a **majority** of scored tiles;
- R's mean RMSE is lower;
- the paired two-sided Wilcoxon over tiles gives **p < 0.05 after Holm across the 2 checks** (ICESat-2, DEM).

The 95% tile-bootstrap CI of R − P is reported as well.

**Phase D gate.** Only if that rule passes: note that data starvation (32 tiles) remains a separate, untested variable.
Do not test it this session.

## 2026-09-24 — Phase C: Kaggle GPU build and run-selection rule (decided before any Phase C result exists)

**The build.** `scripts/s2_token_grid_phase_c_kaggle.py` follows the CLAUDE.md Kaggle pattern.
- It is self-contained, and the model and Phase C code are verbatim copies.
- The bundle ships the 32-tile cache and the DAv2-Small weights, and runs offline.
- `transformers==5.17.0` is pinned, and CUDA is asserted on `cuda:0`.
- Manual: `data/kaggle_bundles/s2_token_grid_c/read.md`. The zip (278 MB) is gitignored; rebuild it from `bundle_c/`.

**Parity check (CPU):**
- Arm P, 3 steps plus the full fold-0 evaluation: all 288 numeric fields identical (max diff 0.0), same final loss.
- Arm R, forward pass plus loss: max abs diff 0.0.
- The extracted zip runs with `HF_HUB_OFFLINE=1`.

**Run-selection rule** (fixed now, before any Phase C number exists):
1. Both arms of a comparison must come from **one device**. Kaggle and local fold files are never mixed. Kaggle
   output goes to `token_grid_test/kaggle/`.
2. **Primary result:** the Kaggle run, if all 8 folds complete there.
3. **Fallback:** the local MPS run (`run_phase_c.sh`, which starts after Phase B).
4. If both complete, the second one is reported as a **cross-device replication**, and the verdict is the primary's.

## 2026-09-24 — Phase B result: **R fails the rule at 10 m and 12 m. The stop condition fires.**

**Run.** Chain 08:40–10:49 IST (`chainB.log`), no failures: R 10 m and R 12 m, 4 folds each, about 13–18 min per
fold. Then eval and `analyze`, written to `matrix_summary.json`. The analysis reproduces the earlier R 5 m / 8 m
numbers exactly (0.687 / 0.631), so the extended matrix is consistent with the committed one.

**Rule 1: evaluated at native 0.3 m against the 0.3 m oracle.** The oracle is MAE 3.392 [2.963, 3.865], RMSE 4.579
[3.985, 5.229], Pearson 0.582 [0.521, 0.639], Spearman 0.509 [0.456, 0.559]. Each cell shows the mean, its
[95% tile-bootstrap CI], and wins out of 50 tiles.

| Train GSD (R) | MAE | RMSE | Pearson | Spearman | Metrics better with non-overlapping CI | Rule 1 |
|---|---|---|---|---|---|---|
| 5 m (reference) | 2.437 [2.11, 2.80] 44/50 | 4.101 [3.48, 4.82] 38/50 | 0.687 [0.646, 0.724] 38/50 | 0.624 [0.583, 0.660] 42/50 | 3/4 | works |
| 8 m (reference) | 3.062 [2.74, 3.42] 33/50 | 4.647 [4.02, 5.37] 24/50 | 0.631 [0.590, 0.668] 26/50 | 0.585 [0.546, 0.621] 37/50 | 0/4 | fails |
| **10 m** | 3.261 [2.93, 3.63] 29/50 | 4.819 [4.17, 5.58] 21/50 | 0.610 [0.572, 0.645] 26/50 | 0.576 [0.537, 0.610] 37/50 | **0/4** | **fails** |
| **12 m** | 3.500 [3.16, 3.88] 21/50 | 5.028 [4.36, 5.81] 18/50 | 0.551 [0.511, 0.590] 18/50 | 0.523 [0.482, 0.561] 27/50 | **0/4** | **fails** |

Holm-adjusted Wilcoxon p (across the R models):
- R10: MAE 0.62, RMSE 0.43, Pearson 0.40, **Spearman 3e-4** (37/50 tiles; the point estimate is better, but the CIs
  overlap, so this doesn't count under rule 1).
- R12: MAE 0.62, RMSE **0.025 (worse than the oracle)**, Pearson 0.17, Spearman 0.65.

**Extended curve.** Native-eval Pearson against training GSD, under R:

| Training GSD | 0.3 m (native) | 2 m | 3 m | 5 m | 8 m | 10 m | 12 m |
|---|---|---|---|---|---|---|---|
| Pearson | 0.745 | 0.721 | 0.708 | 0.687 | 0.631 | 0.610 | 0.551 |

MAE: 1.98 → 2.19 → 2.26 → 2.44 → 3.06 → 3.26 → 3.50.
- The bend seen between 5 and 8 m continues: it **reaches the oracle around 10–12 m** and falls below it at 12 m on
  Pearson, MAE and RMSE.
- It degrades steadily; it doesn't collapse. At 10 m the model still ranks heights about as well as the per-tile oracle,
  but it no longer beats it.
- Coarse→fine is still better than fine→coarse at every GSD: Pearson c→f 0.610 vs f→c 0.123 at 10 m; 0.551 vs 0.091 at
  12 m.

**Stop condition (pre-registered): fired.**
- Under the most favourable version of R (synthetic degradation of the *same* sensor and city it is tested on, with the
  token grid fully preserved), 10 m-trained Method 6 **does not beat a dense per-tile oracle**.
- The synthetic evidence therefore predicts that real Sentinel-2 under R is **unlikely** to help. The reasons don't
  depend on the architecture: at about 10 m the imagery no longer carries enough height-relevant detail about
  0.3 m-scale structure for this model to beat the oracle.

**Phase C:** still to be run, as pre-registered, with **tempered expectations**. It is a different question: 10 m
terrain relief against ICESat-2, not DFC2019 AGL. Real data can differ from the synthetic prediction, but a clean R win
would now be surprising.

**Caveats:** one seed (42); one WorldView-3 city; synthetic area-average degradation, not a real sensor PSF.

## 2026-09-24 — Phase C result (Kaggle T4, primary per the run-selection rule): **R does not help. Pre-registered bar not met.**

**The run.** `s2_token_grid_phase_c_kaggle.py` on a Tesla T4 (torch 2.10.0+cu128, `cuda:0`), 8 trainings of 600 steps ×
batch 4. P took about 30 s per fold and R about 10 min per fold; peak GPU memory was 2.6 GiB (P) and 6.9 GiB (R).
Outputs are in `data/sentinel2_benchmark/token_grid_test/kaggle/`: `eval_{P,R}_fold{0-3}.json`, `summary.json` and
`phase_c_kaggle_log.txt`.

**Integrity checks, done locally before accepting the output:**
- **Settings:** all 8 folds have steps 600 / batch 4 / 6,144 training crops / 32 tiles. Fold 0's `height_scale` of
  34.47 m equals the local value.
- **Values:** 0 non-finite values.
- **Model-free fields** (the flat and FABDEM baselines' squared errors and the pixel and photon counts) were recomputed
  independently from the local cache for all 256 tile-folds. The counts are exact, and the maximum relative
  difference is **0.0**. The P and R files also agree with each other on these fields.
- **Analysis:** the local `analyze` re-run on the Kaggle fold files reproduces Kaggle's `summary.json` with a maximum
  absolute difference of **0**.

**Result (32 tiles; each tile pooled over its 4 held-out quadrants; RMSE in m):**

| Check | Arm P (5×5 tokens) | Arm R (39×39 tokens) | R − P [95% bootstrap CI] | R wins | Wilcoxon p (Holm) | Flat crop mean | FABDEM (context) |
|---|---|---|---|---|---|---|---|
| **ICESat-2 ground (primary)** | 16.602 | 16.538 | −0.065 [−0.420, +0.220] | **16/32** | 0.705 (**0.821**) | 17.276 | **2.855** |
| DEM held-out, 30 m (secondary) | 17.706 | 17.641 | −0.065 [−0.409, +0.214] | 13/32 | 0.411 (0.821) | 18.301 | — |

Within-crop relief (descriptive):
- Pearson: P 0.256, R 0.280.
- Variance ratio: P 0.73, R 1.48, i.e. R over-disperses.

**The pre-registered verdict is `verdict_R_helps = false`.** All three parts of the rule fail:
- R wins exactly half the tiles, not a majority;
- the mean difference is 0.06 m, with a CI centred on zero;
- the Holm-adjusted p is 0.82.

With identical crops, target, tiles, folds, steps and device, preserving a large token grid by upsampling gives **no
measurable benefit** over the true-pixel-count grid on real Sentinel-2.

**Context (not tested; honest framing):**
- Both arms learn only a little relief: they beat the flat crop mean by about 0.7 m of RMSE, and are **about 6× worse
  than raw FABDEM** (2.85 m).
- Learning 600 m-scale terrain relief from 10 m RGB is weak in either geometry. This agrees with every earlier
  Sentinel-2 result: the product stays DEM-only.

**Post-hoc (labelled; not part of the rule):**
- By landscape, R − P on ICESat-2 is hilly −0.66 m (R wins 5/8), agricultural +0.17 (3/8), coastal +0.17 (4/8) and
  urban +0.06 (4/8).
- The only direction favouring R is hilly terrain, on 8 tiles. It is not significant, and it is not a finding.

**Consistency with Phase B's prediction.** Phase B's stop condition predicted no R benefit at about 10 m, and Phase C
agrees. Synthetic and real evidence point the same way.

**Local run: not executed.** It was put on hold before starting, per the user, and then cancelled once the Kaggle
output was validated. There is therefore no cross-device replication.

## 2026-09-24 — Phase D: **not triggered**

Phase C did not clear its bar, so per the pre-registration Phase D doesn't run. For the record, data starvation
(32 tiles) was held constant in Phase C, not resolved. It remains a separate, untested variable, but there is now no
positive geometry result that would make it the next question to test.

## Conclusions (whole session)

1. **Housekeeping.** DFC2019's native GSD is ≈ 0.3 m (0.30–0.34), measured from lane-line cycles, lane widths and
   truck lengths. The 0.5 m figure is retracted (dated correction in `sparse-lidar-feasibility.md`). The US3D spatial
   join's 0/50 negative is not clean, because it was rasterised at the wrong scale; that is an open item.
   Resolution-transfer labels are not materially affected.
2. **Phase A.** Of the four prior Sentinel-2 attempts:
   - A1–A3 (Method 4) had a frozen 37×37-token DAv2 (270 m per token) and a trained 9 × 9 px CNN. They say nothing
     about R.
   - A4 (Method 6 staged on Sentinel-2) was R-like by token count (36×36) and **failed ICESat-2 2/25**. That is direct
     evidence against "a large token grid fixes Sentinel-2". It was P-like in pixels per token (14).
3. **Phase B.** Synthetic R fails rule 1 at 10 m and 12 m (0/4 metrics each). Native-eval Pearson for 5 / 8 / 10 / 12 m
   is 0.687 / 0.631 / 0.610 / 0.551 against the oracle's 0.582. The stop condition fired.
4. **Phase C.** On real Sentinel-2 with everything held constant except token geometry, R = P: ICESat-2 16/32 tiles,
   Holm p = 0.82, Δ −0.06 m [−0.42, +0.22].
5. **Bottom line.** Resolution-transfer's token-grid finding **does not extend to real Sentinel-2**. Both routes are
   now closed: the count-only reading (A4) and the upsampling reading (Phase C). Open item 0d in HANDOFF is closed
   negative for Sentinel-2.

**Kaggle export cleanup.**
- The bundle folder, zip and manual (`data/kaggle_bundles/s2_token_grid_c/`) were deleted after the output was
  validated, freeing about 360 MB. The tile cache was hard-linked and remains in `token_grid_test/cache/`.
- The script is kept as `scripts/s2_token_grid_phase_c_kaggle.py` (bit-exact CPU parity with
  `scripts/s2_token_grid_phase_c.py`; see the "Phase C: Kaggle GPU build" entry above).
- To re-stage it: `bundle_c/` = that script + `token_grid_test/cache/*.npz` + the DAv2-Small HF snapshot
  (`config.json`, `model.safetensors`) in `dav2_small/`; then `zip -0 -r`.

## 2026-09-24 — Rank-loss test: pre-registration (committed before the real run; only a 5-step smoke test run so far)

**Question.** Does a pure ranking objective find real terrain signal where Phase C's magnitude loss didn't? This is a
**final, cheap check that closes this line**. If it fails, it doesn't open a new direction.

**Script.** `scripts/s2_rank_loss_test.py`, output in `data/sentinel2_benchmark/token_grid_test/rank_loss/`.

**Held fixed from Phase C** (imported from `s2_token_grid_phase_c.py`):
- 32 tiles, 4 quadrant folds, identical 60 px crops;
- the FABDEM-30 m target, with the loss on 3 × 3-pooled output;
- `TwinHeadDav2` (DAv2-Small), 600 steps × batch 4, optimiser, schedule, seed and crop order;
- the ICESat-2 scoring.
- **Geometry: arm P** (5×5 tokens). Phase C found R = P, and P is about 20× cheaper. Chosen before running.

**The only change: the loss.**
- `rank_pair_loss`, imported **unchanged** from `scripts/evaluate_method4_v2.py`: `F.margin_ranking_loss`, 2,000
  random pairs per crop, margin 0.25, exact ties dropped.
- The pairs are ordered by FABDEM-30 m within the crop.
- It is the rank term of Method 4 v2's `phase2_building_rank_v2` on DFC2019. *Correction to the prompt:* Method 6 has no
  rank term; only Method 4's exists.

**Metric scale is recovered post hoc**, with the project's per-tile linear calibration:
- Per tile, a no-intercept OLS slope of FABDEM within-crop relief on score within-crop relief (30 m grid), fitted on
  that tile's **3 training quadrants only**.
- Held-out prediction = crop FABDEM mean + slope × (score − crop-mean score).

**Oracle** (this project's standard definition, calibrated **identically**): frozen DAv2-Large whole-tile depth at 518
(`data/sentinel2_benchmark/dav2_depth/*.npy`). It gets the same per-tile slope fit, on the same training quadrants, and
the same crop-mean-plus-relief prediction. Smoke-test check: its model-free fields are identical to Phase C's (max diff 0).

**Pass rule** (the same bar as every Sentinel-2 test this session). "Real signal" requires all three, on the ICESat-2
check (primary):
- calibrated rank model RMSE below the oracle's on a **majority of the 32 tiles**;
- a lower mean;
- **paired Wilcoxon p < 0.05, Holm-adjusted across the 2 checks** (ICESat-2, DEM held-out).

**Also reported** (context, not part of the rule):
- the 95% tile-bootstrap CI;
- the flat crop-mean and raw-FABDEM baselines;
- the calibration slopes (median, and the fraction negative);
- the rank model against Phase C's arm P (magnitude loss, same geometry, primary Kaggle run). This comparison is
  descriptive: it mixes MPS and CUDA.

**Wall time (R4).** Measured in the smoke test; about 3 min per fold on MPS, ≈ 12 min in total. Run with nohup.

## 2026-09-24 — Rank-loss test result: **no real signal. The pre-registered bar is not met, and this line is closed.**

**The run.** Local MPS, 4 folds of 600 steps (about 70 s each) plus `analyze`. Logs: `rank_loss/{train,analyze}.log`;
results: `rank_loss/summary.json`. Final rank loss per fold 0.19–0.21, down from about 0.66 at initialisation, so the
network did learn within-crop order on its training crops.

**Result (32 tiles, each pooled over its 4 held-out quadrants; RMSE in m):**

| Check | Rank model, calibrated | Oracle (frozen DAv2-L + the same calibration) | Model − oracle [95% CI] | Model wins | Wilcoxon p (Holm) |
|---|---|---|---|---|---|
| **ICESat-2 ground (primary)** | 16.251 | 16.745 | −0.494 [−1.325, +0.280] | 18/32 | 0.270 (**0.540**) |
| DEM held-out (secondary) | 17.275 | 17.420 | −0.145 [−1.168, +1.168] | 18/32 | 0.270 (0.540) |

The two checks have identical p-values by coincidence: the per-tile differences are different arrays (r = 0.90) with
the same signed-rank statistic (204). Checked, not a bug.

**Verdict: `verdict_real_signal = false`.**
- The first two parts hold: 18/32 is a majority, and the mean is lower.
- The significance part fails: Holm p = 0.54, and the CI includes 0.

**Context (not part of the rule):**
- Raw FABDEM is at **2.855 m**, i.e. the calibrated rank model is about 5.7× worse than the terrain product. The flat
  crop mean is at 17.28 m.
- Median calibration slope: model 2.13, oracle 18.3; about 13% negative for both. Per-tile scale recovery is noisy for
  both.
- **Rank loss vs Phase C's magnitude loss** (arm P, same geometry): 16.25 vs 16.60 m, 28/32 tiles, Wilcoxon p = 6e-4,
  Δ −0.35 m [−0.69, −0.04]. This is **descriptive only**: it compares MPS with CUDA runs and was not a pre-registered
  test. Taken at face value, ranking beats magnitude regression as a *loss*, but the difference is a third of a metre on
  a 16 m error. That is far from useful, and it still doesn't beat the oracle.
- **Post-hoc, by landscape** (model − oracle on ICESat-2): hilly −1.94 m (6/8 wins); agricultural, coastal and urban
  about 0.0 (5/8, 3/8, 4/8). As in Phase C, any movement is confined to the 8 hilly tiles. Not a finding.

**Closing the line (as pre-agreed).** Three routes have now been tried on real Sentinel-2 at 10 m, and none adds
terrain signal a DEM user could use:
- token geometry (Phase C: R = P);
- loss design (rank loss: no significant gain over the frozen-prior oracle);
- the earlier CNN and target variants (Phase A).

The **Sentinel-2 learned-terrain-correction line is closed**, and the product stays DEM-only (FABDEM terrain,
GLO-30 surface). No follow-up is opened.
