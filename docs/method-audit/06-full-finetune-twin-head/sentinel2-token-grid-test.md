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
