# Final comparison — Depth Wizard 2 research track (SIH26175)

_Status as of 2026-09-23. Every number below traces to a committed file, and the path is given
next to it. Decisions made after seeing results are labelled **post-hoc**. Every other decision
rule was committed to the log before its test ran (commits prefixed `prereg:` in
`00-audit-log.md`). Chronology and the result/commit gaps: `docs/method-audit/00-audit-log.md`._

## Executive summary

On the DFC2019 benchmark (50 tiles, dense airborne LiDAR, 4-fold held-out spatial quadrants),
**Method 6** is clearly the best result. It is a full fine-tune of Depth-Anything-V2-Small with a
twin mean/variance head and height-balanced loss and sampling, and it scores **MAE 1.980 m, RMSE
3.492 m, Pearson 0.745, Spearman 0.656**. That beats the per-tile-OLS oracle on all four metrics
and on **47–49 of 50 tiles per metric**, with tile-bootstrap intervals that don't overlap on MAE,
Pearson or Spearman. The result holds across three training seeds (MAE 1.990 ± 0.010 m).

On the real deployment domain (10 m Sentinel-2 over India, 32 benchmark tiles, scored against
independent ICESat-2 and GEDI lidar), **none of the depth-model corrections helps**:
- The earlier "frequency fusion" headline turned out to be the DEM's own result.
- Learned CNN corrections failed three separate ways.
- RDAH-Net carries no height-above-ground signal at 10 m.
- The best products are plain DEMs: **FABDEM for bare-earth terrain** (median RMSE 1.78 m vs.
  ICESat-2 ground photons, beating GLO-30 on 32/32 tiles), and **raw Copernicus GLO-30 for the
  surface** (beats SRTM on both RMSE and bias-removed RMSE).
- 10 m canopy-height models do carry a real height signal, mostly between landscapes rather than
  within a scene, but adding them to a DEM doesn't improve on a plain DEM.

Along the way the project overturned several of its own earlier headlines, and those corrections
are listed in §5, because catching them is what the methodology is for.

---

## 1. DFC2019 (proxy benchmark; dense LiDAR AGL truth)

All rows use the 50-tile benchmark. **Aggregation differs by row and is stated.** "Mean of 200" =
mean over (tile, held-out quadrant) evaluations = mean of the 4 fold means.

| method | MAE (m) | RMSE (m) | Pearson | Spearman | aggregation / protocol | source | status |
|---|---:|---:|---:|---:|---|---|---|
| **M6 — full DAv2-Small fine-tune + height-balanced** | **1.980** [1.691, 2.297] | **3.492** [2.964, 4.106] | **0.745** [0.704, 0.779] | **0.656** [0.616, 0.692] | mean of 200; 4-fold quadrant CV; 95% tile-bootstrap CI | `data/dfc2019/experiments/method6_height_balanced/m6_heightbal_results.json`; CIs `data/dfc2019/experiments/method6_uncertainty.json` | **CURRENT BEST** (3 seeds: 1.990 ± 0.010 / 3.504 ± 0.026 / 0.743 ± 0.002 / 0.656 ± 0.0003; §2.3) |
| M6 — original recipe | 2.053 | 3.531 | 0.737 | 0.654 | mean of 200 | `method6_finetune_twinhead/method6_results.json` | superseded by height-balanced |
| **Oracle per-tile-OLS** (DAv2 → AGL fitted on 3 quadrants of the same tile) | 3.392 [2.963, 3.865] | 4.579 [3.985, 5.229] | 0.582 [0.521, 0.639] | 0.509 [0.456, 0.559] | mean of 200; same quadrants as M6 | `semantic/method3_spatial_cv_results.json` ("baseline") | reference |
| M5 — RDAH-Net zero-shot, **Track1** ckpt, ×255 | 2.231 | 4.566 | 0.716 | 0.655 | mean of 4 tile-level folds; per-fold affine | `rdah_zeroshot/rdah_x255_zeroshot_fold_results.csv` | not comparable: 41/50 tiles in the checkpoint's training list |
| M5 — RDAH-Net zero-shot, Swiss ckpt | 3.033 | 6.421 | 0.492 | 0.542 | mean of 4 folds, FT-2's true-test halves (TRANSCRIBED) | `rdah_zeroshot/swiss_zeroshot_fold_results.csv` | reference for FT-2 |
| M5 — RDAH-FT-2 | 2.500 | 4.294 | 0.640 | 0.506 | mean of per-sample, then folds (25 of 50 held-out samples per fold) | `rdah_quadrant_cv/rdah_ft2_aggregate.json` | NOT ADOPTED (loses to M6 on all four) |
| M5 — RDAH-FT-1 | 2.906 | 6.659 | 0.513 | 0.527 | true pixel pooling, tile-level folds | `rdah/pooled_cv_epoch5_report.json` | superseded |
| M4 v2 — learned CNN scale modulation (`phase2_building_rank_v2`) | 2.880 | 4.775 | 0.583 | 0.544 | mean of folds | `method4_v2_phase2.5_r2_building_rank_v2/phase2_building_rank_v2_results.json` | superseded by M6 |
| M3 — semantic prior | 3.395 | 4.907 | 0.554 | 0.486 | mean of 200 | `semantic/method3_spatial_cv_results.json` | CLOSED (overfit) |
| M2 — sparse anchors, Grid + Huber + 20 | 2.929 | 4.718 | 0.532 | 0.471 | mean of 50 whole-tile runs ("strict" columns); no CV | `sparse_anchor_regression/summary.csv` | CLOSED as standalone |
| M1 — global linear calibration | 3.718 | 5.359 | 0.539 | 0.446 | mean of 50 tiles | `dav2_calibration/evaluation/calibration_evaluation_summary.csv` | CLOSED |
| M1 — global isotonic calibration | 3.691 | 5.313 | 0.556 | 0.446 | mean of 50 tiles | same | CLOSED |

All paths are relative to `data/dfc2019/experiments/` unless given in full. CIs were computed only
where stated. Other rows have none, because their per-tile outputs weren't bootstrapped.

**Naming note.** The project's older docs call the Method 2 Grid+Huber+20 row the "oracle
per-tile-OLS baseline". The true per-tile-OLS oracle is the 3.392/4.579 row. Method 6 beats both.

### 1.1 Method 6 vs. the oracle, per tile (C2; `method6_uncertainty.json`)

| metric | M6 better on tiles | on quadrants | mean paired difference [95% tile-bootstrap CI] | Wilcoxon p (50 tiles) |
|---|---:|---:|---|---:|
| MAE | 49/50 | 190/200 | −1.412 m [−1.647, −1.189] | 3.6e-15 |
| RMSE | 47/50 | 174/200 | −1.086 m [−1.326, −0.850] | 1.2e-11 |
| Pearson | 47/50 | 167/200 | +0.162 [+0.122, +0.209] | 1.1e-12 |
| Spearman | 47/50 | 178/200 | +0.147 [+0.115, +0.182] | 1.6e-13 |

## 2. Method 6 hardening (C1–C4)

### 2.1 Variance ratio (C1)

var(pred)/var(gt), pooled within each held-out fold, for the adopted recipe:
- seed 43: 0.481 / 0.554 / 0.521 / 0.657
- seed 44: 0.488 / 0.508 / 0.551 / 0.659
- **range 0.48–0.66**; OLS slope pred ~ gt 0.54–0.63; bias −0.49 to +0.05 m

Source: `data/dfc2019/experiments/method6_uncertainty.json` (`fold_diagnostics`).

That's still underdispersed, but 2–3× better than RDAH-FT-2 (0.19–0.23). Seed 1 saved no
predictions, so its ratio can't be computed.

### 2.2 VHR (Maxar) sanity check (C4)

**What was run.** `scripts/method6_vhr_sanity_check.py` fed three real Maxar Open Data crops
(Sikkim, India-Floods-Oct-2023 event, ~0.305 m GSD, 1,512² px each) through the **full-data**
Method 6 checkpoint `method6_full_checkpoint/method6_full_dfc2019.pt`.
- That checkpoint uses the **original** recipe (not height-balanced), trained on all 50 DFC2019
  tiles with no holdout.
- Inference only.
- Outputs: `data/maxar_sanity/method6_inference/*_{mu,sigma,rgb}.npy`.

**What it found.** No write-up existed. The statistics below were recomputed from the saved outputs
on 2026-09-23, with no new inference: `data/maxar_sanity/method6_inference_summary_recomputed.txt`.
- **Two crops (a: 13° off-nadir; c: 26° off-nadir, finest GSD):** non-degenerate height fields.
  Mean 2.3 / 3.4 m, max 16.0 / 14.1 m, σ ≈ 2.0 / 2.6 m, 99th percentile 8.0 / 11.0 m.
- **Crop b (near-nadir, 2°):** nearly flat. Mean 0.35 m, max 2.95 m, 99th percentile 1.5 m.
- Correlation of μ with grayscale brightness: +0.34, −0.57 and −0.26 across the three.

**What it did NOT test.**
- **Accuracy:** no LiDAR or DEM truth is wired up for these crops.
- The **adopted height-balanced recipe**.
- Held-out-fold generalisation (the checkpoint saw all DFC2019 data).
- Anything at Sentinel-2's 10 m.

It shows only that the DFC2019-trained model does not collapse on real satellite VHR imagery in
2 of 3 crops. The near-flat near-nadir crop is unexplained.

### 2.3 Seeds (C3) — pre-registered: the headline stands if every seed beats the oracle on all 4

| seed | MAE (m) | RMSE (m) | Pearson | Spearman |
|---|---:|---:|---:|---:|
| 1 (original) | 1.980 | 3.492 | 0.745 | 0.656 |
| 43 | 1.989 | 3.486 | 0.744 | 0.656 |
| 44 | 2.000 | 3.535 | 0.741 | 0.656 |
| **mean ± sd** | **1.990 ± 0.010** | **3.504 ± 0.026** | **0.743 ± 0.002** | **0.656 ± 0.0003** |

**Every seed beats the oracle on all four metrics. The headline stands.** Every seed is better
than the oracle on 46–49 of 50 tiles per metric. Sources: the three `m6_heightbal*_results.json`
and `method6_uncertainty.json`.

---

## 3. Sentinel-2 / India (deployment domain; 10 m; no dense truth)

References, all independent lidar:
- **ground:** ICESat-2 ATL08-classified ground photons
- **surface-IS2:** ICESat-2 20 m PhoREAL segments, `h_te_median + h_max_canopy`
- **GEDI:** GEDI L2A `elev_lowestmode + rh98`

**Independence flags:**
- ETH GCH 2020 was trained on GEDI, so ETH vs. GEDI is non-independent.
- FABDEM used GEDI canopy height as a predictor, so FABDEM vs. GEDI is non-independent.
- FABDEM used ICESat-2 only north of 52°N, so it is **independent** here (tiles at 9–33°N).

**Metric conventions:**
- Per-tile RMSE (m), median across tiles.
- "32 tiles" = the full benchmark. "25 tiles" = the DAv2 sign-flip-filtered subset, which is
  irrelevant for DEM-only products.
- R4 offset guard: a "pass" must also hold on bias-removed RMSE.

### 3.1 DEM-only products (A2/A3; 32 tiles; per-point geoid)

Source: `data/sentinel2_benchmark/dem_baselines_32/summary.json`.

| product | ground: median RMSE [95% CI] | surface-IS2 | GEDI | status |
|---|---|---|---|---|
| **FABDEM** (bare earth, EGM2008) | **1.782** [1.068, 3.417] | 5.445 | 11.126 ‡ | **recommended TERRAIN baseline**: beats GLO-30 on ground, 32/32 on RMSE and on bias-removed RMSE, p = 4.7e-10 |
| **Copernicus GLO-30** (DSM, EGM2008) | 3.045 [1.913, 5.223] | **4.484** [3.363, 8.846] | 9.375 | **recommended SURFACE baseline**: beats SRTM on surface-IS2 (RMSE 24/32, p = 6.6e-4; bias-removed 31/32) |
| SRTM 1″ (EGM96) | 3.600 [3.173, 5.023] | 4.552 | 9.317 | superseded |
| FABDEM + ETH canopy | 7.630 | 7.658 | 8.527 ‡ | fails R4 on surface-IS2 (4/32); the GEDI "gain" is an offset (bias-removed 6/32) |
| FABDEM + CHMv2 | 1.787 | 5.404 | 11.101 ‡ | fails on surface-IS2 (0/32); ≈ bare FABDEM |

‡ non-independent reference for that product.

On GEDI, GLO-30 and SRTM don't separate on RMSE (16/32, p = 0.98): both sit ~7–8 m below GEDI
canopy tops. They do separate on bias-removed RMSE (28/32). Per-category tables and the 25-tile
subset are in `sentinel2/sign-flip-detector.md` ("final close-out", A2+A3); the 25-tile subset
reaches the same A2/A3 conclusions.

### 3.2 Everything tried on Sentinel-2

| candidate | ground | surface-IS2 | GEDI | status | source |
|---|---|---|---|---|---|
| Linear-calibrated DAv2 (per tile, fitted to SRTM) | median 10.88% of range | — | — | superseded (weaker than the raw DEM) | `data/sentinel2_benchmark/srtm_3way_comparison.csv` |
| Frequency fusion (SRTM low-pass + DAv2 high-pass) | 3.568% (3.111 m); **= DEM low-pass (10/25, p = 0.853)** | 4.987 vs. 4.997 m control | 9.164 vs. 9.166 | **retired**: the DEM carries it | `frequency_fusion_controls/controls_summary.json`; `detail_source_bakeoff/summary_srtm.json` |
| DAv2 @518 as detail | r_HF −0.037 (8/25 > 0) | r_HF +0.026 | r_HF +0.056 | fails every reference | Phase 1 / Phase 4 summaries |
| DAv2 @1008 as detail | r_HF −0.008 | +0.069 (Test A fragile: fails on the strict subset) | +0.089 | fails (closest near-miss) | `detail_source_bakeoff/summary_srtm{,_strictis2}.json` |
| DINOv3-CHMv2 as detail | r_HF −0.003 | +0.006 | +0.012 | fails; its RMSE "wins" are an offset | same |
| ETH GCH 2020 as detail | worse (0/25) | worse (5/25) | Test A pass ‡, Test B fail | fails; ETH ceiling fires | same |
| RDAH-Net (Swiss, ×255, zero-shot) | — | Darjeeling Spearman +0.026 | Darjeeling Spearman −0.010 | **CLOSED** by pre-registered rule (A5) | `rdah_zeroshot/darjeeling_a5/result.json` |
| Method 4 CNN correction, 10 m grid | DEM RMSE 75.97 vs. linear 117.87 m, but **ICESat-2 69.79 vs. 53.40 m (lost)** | — | — | CLOSED (interpolation memorisation) | `sign-flip-detector.md` 2026-09-21 |
| Method 4, native 30 m | DEM wins 24/100, ICESat-2 wins 16/100 | — | — | CLOSED (data starvation) | same |
| Method 4, Open Buildings target (8 urban) | 6/8 win the training-target check, lose ICESat-2 | — | — | CLOSED (target memorisation) | `sign-flip-detector.md` 2026-09-22 |
| Method 6 staged on Sentinel-2 | fold 0: DEM 1/25, ICESat-2 2/25 tile wins | — | — | stopped at fold 0 | same |

(The "surface-IS2" and "GEDI" entries for the detail sources are Phase 4 r_HF medians; the
pre-registered bar was 0.10.)

### 3.3 Is there any height-above-ground signal at 10 m? (A4, direct test, no DEM)

Source: `data/sentinel2_benchmark/direct_height_test/summary.json`.

| model | pooled Spearman vs. ICESat-2 `h_max_canopy` (25 tiles) | tiles positive | within-tile pooled Spearman (**post-hoc**) | vs. GEDI rh98 | pre-registered verdict |
|---|---:|---:|---:|---:|---|
| DINOv3-CHMv2 | 0.641 | 23/25 | 0.276 | 0.539 | passes |
| ETH GCH 2020 | 0.378 | 21/25 | 0.234 | 0.753 ‡ | passes |
| RDAH-Net (Darjeeling only) | 0.026 | — | — | −0.010 | fails |

---

## 4. The resolution finding (wording fixed in advance)

**What the curves show.** Models **trained on very-high-resolution imagery** lose accuracy as input
resolution coarsens.
- **RDAH-Net (Swiss), 50 DFC2019 tiles:** pooled Pearson vs. LiDAR AGL 0.490 / 0.581 / 0.537 /
  **0.330** at 0.3 / 0.6 / 1.2 / 2.4 m GSD (CIs in `data/dfc2019/experiments/resolution_curves_bootstrap.json`),
  a 43% drop from the 0.6 m peak to 2.4 m.
- **Method 6 (seed-43 checkpoints), 50 tiles, held-out quadrants:** pooled Pearson 0.795 / 0.791
  / 0.776 / **0.708** at the same GSDs, only −11%. Variance ratio 0.55 → 0.37.
- The two curves use different input protocols (RDAH gets the small image directly; Method 6 gets
  it upsampled back to 512²) and aren't directly comparable. Details: `05-rdah-net-fusion/summary.md`
  §13.

That's two models (RDAH-Net Swiss; Method 6, one seed) on 50 DFC2019 tiles, 0.3–2.4 m GSD. **The curves do not show an information limit at 10 m.**

**What the direct test shows (A4, the only evidence on that question):**
**"10 m height signal exists; combining it with a DEM is the failure."** Two 10 m canopy models
pass a pre-registered test against independent ICESat-2. **Post-hoc qualifier:** the signal is
mostly *between landscapes*; within a scene it's weak (Spearman 0.23–0.28). And no DEM + canopy
product beats a plain DEM (A3).

---

## 5. Audit corrections — claims this project overturned in itself

| date | old claim | new claim | why |
|---|---|---|---|
| 2026-09-21 | Sentinel-2 ICESat-2 errors show a DAv2 accuracy ceiling | it was a vertical-datum bug (orthometric vs. ellipsoidal) | root-caused and fixed (`sign-flip-detector.md` 2026-09-21) |
| 2026-09-22 | RDAH-Net zero-shot is broken ("near-constant output") | input-scale bug: depth fed at ~1/255 of training scale | 05 audit (`05-rdah-net-fusion/summary.md` §1) |
| 2026-09-23 | frequency fusion beats linear calibration on 21/25 tiles (10.88% → 3.57%), "deployable baseline" | fusion = its DEM-only control (10/25, p = 0.853); DAv2 detail r_HF −0.037; raw GLO-30 is better (2.32%) | the baseline had been weaker than the raw DEM; a pre-registered DEM-only control exposed it (Phase 1) |
| 2026-09-23 | RDAH rejected on Sentinel-2 *because of checkerboard artifacts* | the checkerboard is intrinsic: it appears on in-domain DFC2019 where RDAH works | Phase 2 resolution sweep |
| 2026-09-23 | RDAH Sentinel-2 correlation test vs. terrain | uninformative: RDAH predicts height *above* ground | construct validity (05 summary §11) |
| 2026-09-23 | RDAH on Sentinel-2 closed via the rule's "default" clause (post-hoc "not graceful") | closed by the pre-registered **rerun branch** (A5: Spearman ≈ 0 against both references) | A5 replaces the post-hoc closure |
| 2026-09-23 | ×255 is RDAH's input scale (clean derivation) | ×255 was picked on Track1, on 3 tiles in Track1's own training list; a clean Swiss re-derivation finds a broad ×200–×1000 plateau | rescued scratchpad (05 summary §11) |
| 2026-09-23 | open question: does fine-tuning damage RDAH? | no: FT-2 beats Swiss zero-shot on every metric in 4/4 folds | rescued Swiss zero-shot (05 verdict §7) |
| 2026-09-23 | DINOv3-CHMv2 is "the frozen prior going forward" | not carried forward: outputs 0.01–0.25 m on 10 m imagery; fails as a detail source | Phase 4 / A2 (R4) / A4 |
| 2026-09-23 | FABDEM "do not adopt" (09-21) | that tested FABDEM as a DAv2 calibration target; raw FABDEM is the best terrain product | A3 |
| 2026-09-23 | Phase 2 8-tile sweep: "cliff" 0.5006 at 8× | on 50 tiles, 8×/1× = 0.672 | the 8-tile sample was Jacksonville-only (B1) |

Two pipeline bugs were caught before their results were read and are recorded in the log: a
FABDEM mosaic defaulting to a 1° projection, and a PROJ "ballpark" geoid returning N = 0. A guard
now stops the second silently recurring.

---

## 6. What remains open, and what is deliberately closed

**Open:**
- **TSE-Net (self-training):** untouched.
- **Track1 seen-vs-unseen split:** Track1 zero-shot on its 9 unseen vs. 41 seen tiles, to separate
  memorisation from in-domain training. Low priority, inference only.
- **Learned Sentinel-2 + GEDI route:** gated. Not recommended now, because the ETH ceiling fired.
  - **Reopen gate:** a canopy/object-height source passing Test B against ICESat-2 surface. Note
    that A3's FABDEM + ETH form, the most natural candidate, has now failed that too.
  - The route *would* address both earlier failure modes: many tiles of direct GEDI labels instead
    of one tile's DEM, and a sparse lidar target instead of an interpolated one.

**Deliberately closed:**
- global/pooled calibration (M1)
- RANSAC and spatially local calibration variants (M2)
- linear semantic prior (M3)
- CNN correction on Sentinel-2 (3 failures)
- RDAH on Sentinel-2
- DAv2, CHMv2 and ETH as detail add-ons
- frequency fusion as a recommendation
- GSD-FiLM
- RS3DAda (contaminated)
- shadow photogrammetry at 10 m
- semantic-prior phase 2.3
