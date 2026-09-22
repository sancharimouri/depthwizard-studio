# SIH DepthWizard — Competitive Repo Audit

Investigation of 22 curated SIH DepthWizard repos (source: `SIH List.pages`), prioritized
per the user's cluster groupings, with all claimed accuracy numbers verified against actual
results files/code rather than README assertions. Each repo was cloned and its actual
calibration/model/eval code read directly.

## Scope note

The source list has 22 line items, but two entries are literal duplicates of the same repo
(`RohitashvaSharma06/DepthWizard` and `amogh-hub/depthwizard` each appear twice in the source
list with different notes attached), and one repo — `lingeshmani23-tech/DepthWizard` — appears
in the source list but was not in the user's categorized groupings. It was included anyway as
a bonus investigation since its note claimed a directly relevant technique (DA-V2 Metric
checkpoint used directly on the elevation problem). Net: **20 unique repos were actually
investigated**, covering all 19 explicitly named repos plus the one bonus repo.

## Security note

One investigating agent (assigned to Cluster 3) executed a cloned repo's own test suite
(`arpitparashar06/depthwizard`, after `pip install`-ing missing dependencies) rather than only
reading code, which tripped a security policy flag for running external/third-party code.
Nothing in the resulting output looks malicious — it confirmed 20+ of the repo's own internal
test assertions pass — but this is disclosed here for transparency since it went beyond
passive code inspection.

---

## Cluster 1 — Robust-regression / RANSAC / Huber DEM-or-GCP calibration

Directly comparable to this project's own Method 2 (sparse-anchor/GCP regression): best
deployable variant **Grid+Huber+20 anchors — MAE 2.929m / RMSE 4.718m on DFC2019**.

### Haresh-kumar28/DepthWizard
- **(a) Technique:** Simple OLS + single-pass MAD-clip (4×1.4826×MAD) affine `z=a·d+b`, plus a
  separate "Aerial-Depth" fine-tuning kit for DAv2. Not RANSAC, not iterative.
- **(b) Numbers:** No real numbers exist for the calibrated pipeline: `docs/benchmark_template.csv`
  is an unfilled template; `AERIAL_DEPTH_INTEGRATION_STATUS.md` explicitly states "READY FOR REAL
  TRAINING / NOT TRAINED... No trained Aerial-Depth weights... No fabricated accuracy." The only
  concrete number found, in `backend/aerial_depth_training/docs/SCIENTIFIC_PROTOCOL.md:59`, is a
  prior baseline: RMSE 35.3m / MAE 27.5m / R² 0.014 / Pearson 0.12 (explicitly called out as bad,
  motivating the aerial-domain work).
- **(c) Implementation status:** Infrastructure built, core calibration result absent — honest
  about it.
- **(d) Comparison:** Far worse than the Method 2 baseline; not a comparable dataset but
  directionally confirms naive global-affine is weak.

### prashant-d4/DepthWizard
- **(a) Technique:** `backend/app/calibration/metric.py` is **byte-for-byte identical** to
  Haresh-kumar28's (confirmed via `diff`), and the whole `backend/app/` tree matches file-for-file
  — these are forks/copies of a shared base, not independent work.
- **(b) Numbers:** Unlike Haresh's, this one has real committed numbers:
  `docs/evidence/DepthWizard_Final_Benchmark_Results.xlsx` (verified by opening with openpyxl) —
  Urban RMSE 35.3m/MAE 27.5m/R² 0.014, Sparse RMSE 5.85m/R² 0.034 (33.8% coverage), Hilly RMSE
  69.5m/R² 0.117, Forested RMSE 13.5m/R² 0.356. Ground truth: USGS 1m DEM, honestly caveated in
  the sheet itself as "bare-earth, not pure DSM-vs-DSM."
- **(c) Implementation status:** Working code, real (bad) results.
- **(d) Comparison:** Dramatically worse than the Method 2 baseline across every landscape —
  confirms this project's own finding that a naive single global affine fit isn't sufficient
  (its own Method 1 verdict).

### Maniteja8883/SIH-Depthwizard
- **(a) Technique:** Far more sophisticated: U-Net building-footprint segmentation,
  ground-restricted RANSAC (excludes building pixels from the scale fit), DTM/nDSM morphological
  decomposition, and a "PeakRecoveryMLP" ensemble structural-height corrector — real, substantial
  code (`depthwizard/calibration/engine.py`).
- **(b) Numbers:** Mixed verification. "Building MAE (NYC zero-shot) 7.63±0.24m" and "skyscraper
  recovery 44.81%" are explicitly tagged `[MEASURED Phase 29]` in `DEPTHWIZARD_STATE.json` — real.
  But the headline "Absolute DSM MAE 8.14m / RMSE 11.29m" in the current README has no equivalent
  measured-tag or traceable output file (the `runs/` dir that would contain it is
  gitignored/uncommitted) — unverifiable as-is, carried over unchanged from an archived
  `docs/archive/README_PHASE98.md`.
- **(c) Implementation status:** Extensive test suite (309 tests), but that verifies code
  correctness, not accuracy.
- **(d) Comparison:** Even taking the DSM numbers at face value, 8.14m/11.29m is worse than
  Method 2's 2.93m/4.72m (different dataset, not apples-to-apples).

### amulyadsouza-tech/DepthWizard
- **(a) Technique:** Real sklearn `RANSACRegressor`/`HuberRegressor` calibration code
  (`backend/scale_engine.py`), multiple modes (SRTM-RANSAC, polynomial, GCP, scene-prior
  fallback) — genuinely implemented.
- **(b) Numbers:** No real numbers anywhere. `sample_data/generate_samples.py` generates its own
  "reference_lidar.tif" from a sinusoidal function — synthetic, not real LiDAR.
  `docs/technical_report.md`'s only accuracy table is explicitly headed "**Expected**
  Performance" (RMSE 3–8m urban / MAE 2–5m as an anticipated range, not a measured result).
- **(c) Implementation status:** Working code, zero real validation.
- **(d) Comparison:** Cannot be compared to Method 2 — no verified number exists.

### SumitRoy-Gh/Depthwizard
- **(a) Technique:** Genuinely well-designed per-semantic-class RANSAC calibration
  (`src/depthwizard/calibration/region_calibration.py`) — separate scale/shift per
  ground/building/vegetation class via `RANSACRegressor`, with an honest docstring distinguishing
  "our adaptation" from literature. `scripts/evaluation/run_real_calibration.py` is built to run
  against a real ISPRS Vaihingen scene with real ground-truth semantic labels.
- **(b) Numbers:** No real numbers found — `data/`, all outputs, and any checkpoints (`best.pt`)
  are gitignored/not committed; README makes only a vague "metric-level accuracy" claim, no
  fabricated figure.
- **(c) Implementation status:** Working, tested code (`tests/unit/calibration/test_region_calibration.py`
  exists) but no verifiable end-to-end result.
- **(d) Comparison:** Directly comparable in spirit to this project's own Method 3 (semantic
  prior, closed — overfit/didn't generalize); can't confirm whether this repo hit the same wall
  since no numbers exist.

### abxijth/Lorem-Ipsum-SIH26175
- **(a) Technique:** Despite the joke repo name, the README confirms this **is** the SIH26175
  DepthWizard problem statement (team of 6 named, explicit problem-statement framing). Technique
  is GAMUS-fine-tuned DAv2 backbone + global RANSAC calibration + GCP refit (closer to Cluster
  2's fine-tuning approach than pure post-hoc calibration, but placed here per the assigned
  grouping).
- **(b) Numbers:** Best-verified in this cluster — real committed JSON artifacts:
  `results/ab_finetuned/summary.json` → RMSE 4.33m/MAE 3.11m/Pearson 0.611 (fine-tuned, matches
  the README's bolded 4.37/3.17/0.61 within rounding); `results/heldout/validation_summary.json`
  → RMSE 5.63m/MAE 4.51m/Pearson 0.255 (frozen baseline reproduction, matches the README's
  parenthetical). GCP refit: `results/gcp_test/refit/PHL_6151_report.json` → RMSE 1.86m (from a
  3.17m baseline, 2-point GCP). Ground truth: GAMUS (real TU Munich/DLR LiDAR nDSM, DC/PHL/NYC),
  held out by city/split. Also real external validation on hilly (Asheville) and forest (GSMNP)
  terrain using USGS 3DEP/Copernicus GLO-30.
- **(c) Implementation status:** Working end-to-end, numbered reproducibility scripts, decision
  gates documented (segmentation rejected via ceiling-probe, piecewise calibration rejected).
- **(d) Comparison:** Numbers (RMSE 4.33/MAE 3.11) are numerically close/slightly better than
  Method 2, but on GAMUS, not DFC2019 — not a like-for-like win.

### amogh-hub/depthwizard *(verification pass — not full investigation)*
- **(a) Technique:** All four deeper ML claims confirmed as real implemented code, not just
  README prose:
  - **Positive-scale robust fitting:** `src/depthwizard/calibration/robust.py`,
    `robust_affine_calibration()` — IRLS with Huber weighting, explicitly rejects non-positive
    scale (`require_positive_scale`, raises `ValueError` if `original_scale <= 0`),
    condition-number gating.
  - **DEM/GCP evidence gating:** `src/depthwizard/calibration/evidence.py`,
    `calibrate_relative_height_with_dem()` (docstring: "Weakly correlated evidence is rejected
    rather than silently fabricating..."), plus `build_anchor_weights()` filtering on
    ground-probability/uncertainty thresholds.
  - **Leave-one-out validation:** `src/depthwizard/calibration/gcp.py`,
    `_affine_leave_one_out_rmse()`, wired into `calibrate_relative_height_with_gcps()` (line
    266) — real, not a stub.
  - **Calibration/eval reference separation:** `src/depthwizard/evaluation/holdout.py`,
    `sparse_anchor_holdout_benchmark()` — calibrates on a sparse anchor mask, evaluates only on
    pixels outside an excluded dilation radius. Separately, real result JSONs live only on
    frozen `qualification/evidence-<SHA>` branches (not `main`), specifically to keep
    calibration evidence and evaluation reference non-identical
    (`domain_generalization_report.json`: `"reference_independence": "passed"`, with calibration
    evidence files physically separate from the reference raster).
- **(b) Numbers:** Fetched from branch
  `qualification/evidence-012301b9c1910ef4ccde5b3da5d4e4d94da60ce6`, file
  `evidence/submission/domain_generalization_report.json`: cross-sensor scene (NEON AOP, CPER
  site) evaluated against real USGS 3DEP 10m NAVD88 DTM: **MAE 4.23m / RMSE 4.76m / Pearson
  0.59**, n=1 scene, 36M valid pixels, checkpoint/prediction bytes frozen before reference
  evaluation. A separate, less carefully-isolated `baseline-comparison.json` on the same branch
  shows worse absolute numbers (MAE ~5.1–5.2m, RMSE ~9.2m) on 5 higher-relief scenes — the
  domain-generalization number above is the more defensible one to cite since it explicitly
  states independent-reference and frozen-checkpoint guarantees.
- **(c) Implementation status:** Most rigorous engineering in this cluster.
- **(d) Comparison vs. Method 2 baseline (MAE 2.929m/RMSE 4.718m, DFC2019):** Does **not** beat
  it — RMSE is essentially tied (4.76 vs 4.72m) but MAE is meaningfully worse (4.23 vs 2.93m).
  Different ground-truth datasets (NEON/USGS 3DEP vs DFC2019), so not strictly apples-to-apples,
  but on the numbers as reported, no improvement over the existing baseline.

### gowthamkrishna27/Elevate3d *(bonus — flagged as technique-matching this cluster by the
investigating agent, though it was assigned to Cluster 4)*
- **(a) Technique:** RANSAC+Huber DEM/GCP calibration ("Phase 7") with unusually thorough
  diagnostics (rejection reasons, inlier ratios, degenerate-geometry detection) plus a separate
  "Phase 11 Scientific Validation Framework" (confidence-weighted error correlation, percentile
  metrics). Substantial FastAPI backend, ~7000 lines across pipeline stages
  (segmentation/refinement/surface3d/uncertainty/validation).
- **(b) Numbers:** None real. `test_phase11_scientific_validation.py` and
  `demo_scientific_validation.py` both use `np.random.seed(42)` synthetic reference elevation
  explicitly labeled `dataset_identifier="Synthetic_Ground_Truth"`. Only real-world asset present
  is one small `satellite_scene_utm.tif` fixture with no paired reference elevation raster found.
- **(c) Implementation status:** Real, well-engineered scaffold, not a stub — the
  calibration/validation code itself is legitimate and more thorough than several repos with
  real numbers. But it has never been run against real ground truth in this repo.
- **(d) Comparison:** Should be credited as "interesting, implemented, unverified" — good
  diagnostic engineering (worth looking at `calibrator.py`'s rejection-reason tracking as a
  pattern) but zero evidence of real-world accuracy.

**Cluster 1 verdict:** Best-implemented are `abxijth/Lorem-Ipsum-SIH26175` (verification rigor,
real fine-tune+RANSAC+GCP refit, honest negative-result documentation) and `amogh-hub/depthwizard`
(most rigorous evidence-gating/LOO-validation design). None of the seven repos in this cluster
beats the Method 2 baseline on a directly comparable dataset.

---

## Cluster 2 — Learned CNN metric-AGL

Comparable to this project's own Method 4 (`phase2_building_rank_v2` on DFC2019: MAE 2.8803m /
RMSE 4.7751m / Pearson 0.5835 / Spearman 0.5438, vs. per-tile-OLS baseline MAE 3.3924m / RMSE
4.5787m / Pearson 0.5824 / Spearman 0.5093 — beats baseline on 3/4 metrics but the RMSE gap
(4.3%) is still **not closed**, an open problem). Also relevant: this project's own CNN
correction attempt on its Sentinel-2 benchmark **failed** because the model memorized
DEM-interpolation artifacts rather than learning real signal.

### devendrakushwah80/DepthWizard-SIH26175 — strongest of the three, real full pipeline
- **(a) Technique:** `M3NetCore` = dual-branch MobileViT encoder (RGB 3ch + frozen DAv2-Small
  relative-depth 1ch) + CBAM/cross-modal attention + transformer bottleneck + GSD FiLM
  conditioning + PixelShuffle decoder, trained end-to-end to regress metric AGL/nDSM directly
  (`src/models/m3_net.py`, confirmed by reading the code — uses DAv2 as an input channel, same
  general idea as this project's approach but far more elaborate).
- **(b) Numbers (verified against actual JSON evidence files, not just docs):** Sealed
  FINAL-HOLDOUT (310 scenes, GAMUS DC/PHL + DFC2019/US3D JAX/OMA, real airborne-LiDAR-derived
  nDSM ground truth) — MAE 3.04m / RMSE 4.96m / R²=0.40, vs. their own frozen-M2 baseline MAE
  4.20m/RMSE 7.32m. `docs/evidence/stage_a2_final_dev_comparison.json` and
  `docs/evidence/nyc_m2_final_metrics.json` contain real per-pixel metrics matching (not
  identically, but consistently with) the model-card tables — genuine run outputs, not asserted
  numbers. Also ran a 3-seed stability check (σ≈0.42 on selection score) and an explicit
  train/val leakage audit (`nyc_split_leakage_check.json`, 0 overlap).
- **Important caveat, self-disclosed:** on the external zero-shot NYC city, R² is still slightly
  negative (-0.02, up from -1.78) — i.e. even this much more sophisticated model still struggles
  to beat a mean-predictor on genuinely unseen geography, echoing this project's own
  spatial-generalization problem. They also ran a dedicated ground-truth audit and found the
  USGS 3DEP natural-terrain benchmark was silently returning bare-earth DTM as if it were DSM
  (fabricated all-zero AGL) — a methodological catch of the same caliber as this project's own
  negative-result audits.
- **Domain mismatch to flag:** trained/evaluated on GAMUS+DFC2019 airborne aerial imagery
  (0.3–1m GSD), not Sentinel-2 (10m). Numbers aren't directly comparable to this project's
  DFC2019-only Method 4 run, and there's no Sentinel-2 evaluation in this repo at all.
- **(c)/(d) Status/comparison:** Working, production-checkpointed, heavily evidenced. Does not
  "beat" this project's Method 4 RMSE on the same benchmark (different benchmark), but is a
  materially more mature CNN metric-AGL implementation than anything else in this cluster —
  worth a closer look, specifically its GSD-FiLM conditioning and height-balanced loss/sampling
  as ideas transferable to closing this project's own RMSE gap.

### Kukyos/DepthWizard — honest, but no metric result exists yet (hint overclaimed)
- **(a) Technique:** DAv2-Base backbone frozen, DPT neck+head retrained on GAMUS `_AGL` for
  metric AGL (`server/depthwizard/head.py`, `train/train_head.py` — real, sensible code: masked
  L1+gradient loss, height-reweighting option).
- **(b) Numbers:** The project's own `docs/12-eval-results.md` (dated 2026-09-15, latest commit)
  explicitly states: "Calibration: none — Phase 2 not built... Absolute accuracy: Not
  measured... Reporting them would mean inventing a scale." Only a zero-shot, scale-free
  correlation is reported: r=+0.23 all-pixel vs. GAMUS LiDAR AGL, 6 DC tiles only.
- **Verdict on the source-list hint** ("real metric AGL predictor trained and evaluated on GAMUS
  LiDAR-derived height"): **not confirmed — this appears to describe planned/in-progress work,
  not a completed result.** The training code exists and is sound, but no trained-checkpoint
  metric evaluation has actually been run and logged.
- **(c)/(d) Status/comparison:** Code-ready, not executed. No numbers to compare against Method
  4.

### jayyyyqwq/resheightnet — real, honestly-run baseline, but different domain
- **(a) Technique:** RGB-only ResNet34 encoder-decoder (no DAv2 prior at all), reimplementation
  of a published 2019 ISPRS paper, trained from scratch on GAMUS.
- **(b) Numbers (verified, real training run):** MAE 2.17m / RMSE 4.33m / Pearson 0.844 on GAMUS
  DC+PHL `stage_a1_val` (200 tiles, 0.5m/px aerial), beating a cited "DepthWizard M1" baseline
  (MAE 3.31/RMSE 6.28/r=0.68) on the same split. Ground truth: real GAMUS LiDAR-derived nDSM.
- **Exemplary rigor:** `docs/limitations.md` discloses that 85/100 DC val tiles are spatially
  adjacent (including diagonals) to training tiles, explicitly warning the headline numbers
  overstate generalization and citing DepthWizard's own worse NYC-holdout number (4.91m vs
  3.31m) as corroborating evidence. Also flags that the M1 baseline number is reported, not
  independently reproducible from that repo.
- **Not a beat of the DEM-interpolation trap, and not comparable to DFC2019/Sentinel-2:** this
  is aerial photography (0.5m GSD) and RGB-only — it doesn't use a coarse DEM as a training
  target at all, so it's not evidence either way on this project's specific memorization
  failure mode.
- **(c)/(d) Status/comparison:** Working, small but genuinely real result, appropriately
  caveated.

**Prominent flags requested by the task:**
- **(a) Working RMSE beating a reasonable baseline:** `resheightnet` beats its own
  DepthWizard-M1 comparison point (real numbers, real run) but on a spatially-contaminated split
  it discloses itself — treat as suggestive, not clean evidence.
  `DepthWizard-SIH26175`'s M3-FINAL beats its own M2 baseline by a wide, well-audited margin, but
  on a different (aerial, not Sentinel-2) benchmark than this project uses.
- **(b) Non-overfitting approach to coarse-DEM-as-target:** **none of the three** uses a coarse
  DEM (SRTM/GLO-30-class) as a CNN training target the way this project's failed Sentinel-2
  attempt did — all three train against fine-resolution, real LiDAR-derived AGL
  (GAMUS/DFC2019), which sidesteps the DEM-interpolation-memorization problem entirely rather
  than solving it. This project's specific failure mode (memorizing coarse-DEM interpolation
  artifacts) is **not addressed by any of these three repos** — none of them attempt metric
  correction of a foundation-model depth output against a coarse global DEM on satellite
  imagery. That problem remains open, industry-wide, not just here.

---

## Cluster 3 — Frequency/multi-scale fusion

Context: this project just diagnosed, via its own DAv2-vs-DINOv3 backbone comparison and
terrain-failure diagnostics, that coarse DEMs (SRTM/GLO-30) carry real usable low-frequency
elevation signal but their fine-scale structure is partly synthetic (an artifact of DEM
interpolation, not real terrain detail). This project has not yet solved genuine
frequency-separated fusion (real DEM low-frequency component + DAv2 high-frequency detail,
properly combined) — this is the cluster where the audit found the most directly useful
results.

### madhu-mitha-e/DepthWizard — flag: state has changed since the earlier check
This repo showed as a near-empty single-commit repo in an earlier check this session. It is
**no longer near-empty** — this has changed. Now 1 commit but ~1800 lines of real backend code
(FastAPI-style `main.py`, `dem_calibrator.py`, `depth_engine.py`, `validator.py`, tests), plus
committed real GeoTIFF sample data (Kodaikanal, Madurai, Periyar, sparse farmland).
- **(a) Technique:** `dem_calibrator.py::calibrate_detail_fusion()` does genuine frequency
  separation: `highpass = rdsm - gaussian_blur(rdsm, sigma=16px)`, robustly normalized to ±1,
  then `elevation = coarse_dem + detail_weight(0.05) * relief * highpass`. Coarse DEM comes from
  `derive_coarse_dem_from_reference()` (downsample→upsample a real elevation raster to strip
  fine detail) — real low-freq/high-freq split, DEM macro shape + capped DAv2 texture.
- **(b) Numbers:** Docstring claims, on a 5-tile GAMUS (Washington DC LiDAR-AGL) benchmark:
  fusing DAv2 detail as sole predictor gave RMSE 8.6m vs 3.75m for coarse-DEM-alone (2.3x
  worse), Pearson r ~0.04–0.19 for DAv2 vs true height — **not independently verifiable**: zero
  results/log/CSV files exist anywhere in the repo (`test_real_world_gamus.py` exists and would
  produce these numbers if run, using real HuggingFace `earthflow/GAMUS` LiDAR data, but nothing
  is committed). Ground truth source (GAMUS LiDAR AGL) is real and named; the specific numbers
  are asserted-in-comment only.
- **(c) Implementation status:** Working code, not a stub — imports resolve, logic is coherent,
  single commit but substantial.
- **(d) Comparison:** A genuine (if small) frequency-fusion implementation, but its own reported
  conclusion is negative — DAv2 high-freq detail made things worse than DEM alone, so weight is
  kept tiny by default. Corroborates, rather than solves, this project's existing finding that
  DAv2 correction doesn't reliably help. Not independently verified.

### arpitparashar06/depthwizard — standout
- **(a) Technique:** By far the most sophisticated in this cluster. `mathsandml/inference.py`:
  4-way rotation-ensembled DAv2 inference → ramp/tilt removal → `detail = p -
  gaussian_blur(p, sigma)` (sigma adaptively set from scene structure scale or DEM resolution) →
  **`height = terrain + alpha * detail`** where `terrain` is a real fetched DEM
  (OpenTopography-style `fetch_dem`/`dem_on_grid`, with rooftop-contamination debiasing) and
  `alpha` is a physically-derived scale factor from shadow geometry, GCPs, or known-height
  priors (never a naive linear regression on the DEM — explicitly designed to avoid the ramp
  leaking through). This is exactly the "real DEM low-freq + DAv2 high-freq, properly separated"
  technique the task asked about.
- **(b) Numbers:** README claims specific, detailed numbers (frequency-split RMSE 3.51m vs
  global-affine 11.05m vs "hybrid" 66.70m on one urban scene; shadow-calibration recovers known
  scale to within 4%) against real named LiDAR sources (AHN4 Netherlands, swisstopo
  swissSURFACE3D, USGS 3DEP). However: `mathsandml/benchmark/results/` does not exist in the
  repo (scenes/results gitignored, too large), and the README itself states "nothing in [the
  stale report] should be quoted" — so these numbers **cannot be independently verified** from
  the repo as-is. The repo's own test suite (`benchmark/tests.py`) was run after installing
  missing deps — 20+ assertions passed cleanly (attenuation-gain correctness, singular-fit
  guards, no-DEM clip behavior, calibration-leak refusal), confirming the logic is real and
  functioning, not vaporware, even though the headline accuracy numbers aren't reproducible from
  the repo as-is. *(Note: running this test suite is the action that triggered the security flag
  noted at the top of this document.)*
- **(c) Implementation status:** Working, extensively tested (65-check regression suite +
  synthetic self-test harness), unusually self-aware about its own limitations and past bugs.
- **(d) Comparison — flag prominently:** This is the closest thing in the batch to solving this
  project's actual unsolved problem — a real DEM-low-frequency + DAv2-high-frequency fusion with
  a non-regression-based scale source (shadows/GCPs/known-height) specifically to avoid the
  ramp-leakage failure mode. Worth reading in full (`mathsandml/inference.py` lines ~1290–1480)
  even though its accuracy claims aren't independently reproducible from the repo.
  *(Dated note, 2026-09-23: this project later tested DEM-low-frequency + DAv2-high-frequency
  fusion on its own benchmark. The DAv2 component adds no measurable value against a DEM-only
  control; see `docs/method-audit/final-comparison.md`.)*

### blakc-coffee/depthwizard — standout, best-verified
- **(a) Technique:** `ml/calibration/dense_fusion.py::fuse_dense_dsm()`: SRTM resampled
  nearest-neighbor to output grid supplies the low-frequency trend; DAv2 relative-depth has its
  own trend removed via downsample-to-SRTM's-native-resolution-then-upsample (a real matched
  low-pass filter) to yield high-frequency-only detail; combined as `srtm_trend + detail` where
  SRTM has coverage, with a per-pixel confidence/provenance map (255=measured, 76=model-inferred)
  recording which regime each pixel is in. This is real frequency-domain separation, explicitly
  motivated by a documented visual-review finding (DAv2 hallucinates smooth gradients on flat
  ground) — not superficial blending.
- **(b) Numbers — the strongest in the batch:** `docs/validation_report.json` (committed,
  generated by `ml/validation/evaluate.py` which runs the actual production pipeline end-to-end,
  not an isolated regressor) reports on 47 real absolute-scale scenes: RMSE 48.72m/median
  10.45m, MAE 17.12m/median 8.08m, Pearson 0.965 — explicitly caveated that the mean is dragged
  up by 3 named outlier scenes (tuscany_patch_14_2 67m, sierra_nevada 257m, scotland 276m, root
  cause undiagnosed and disclosed as such). Ground truth for these "hilly" scenes is Copernicus
  DEM / USGS 3DEP (independent of the SRTM used as the fusion's own low-freq input — not
  circular). `docs/open_decisions.md` documents an earlier scalar-fusion result verified across
  10 real held-out patches: MAE 243.6m (DAv2-regressor-only) → 33.3m (SRTM-fused), an 86.3%
  reduction, with the exact bug found/fixed en route (conflating SRTM elevation range with mean)
  documented transparently.
- **(c) Implementation status:** Full working stack (FastAPI backend, Celery workers, React
  frontend, Alembic migrations, Playwright e2e tests, ~20 unit test files), actively iterated
  with dated, honest engineering logs.
- **(d) Comparison — flag prominently:** The strongest evidence in the entire batch that real
  frequency-separated fusion (real DEM low-freq + DAv2 high-freq) works and is independently
  measurable. Also independently confirms this project's information-ceiling finding: their own
  regressor has "R²≈-0.96 on held-out canopy patches" and DAv2 relative-depth stats "genuinely do
  not encode absolute scale" — the same conclusion this project reached via its own
  correlation-flip/backbone-comparison work, reached completely independently.

### yats0x7/DepthWizard *(reclassified into Cluster 3 by the investigating agent — was assigned
to Cluster 4 but the technique belongs here)*
- **(a) Technique:** Calibration-family repo but with three selectable modes: `affine` (RANSAC
  vs DEM), `prior` (scene-height prior only), and **`hybrid`** — genuine frequency-separated
  fusion (low-freq DEM + high-freq DAv2 structure), plus an opt-in heuristic semantic-prior route
  (`priors.py`, explicitly documented as "not a land-cover classifier").
- **(b) Numbers:** `docs/BENCHMARK.md`, 8 real scenes across 4 terrain classes
  (forested/hilly/sparse/urban), real Sentinel-2 imagery (real S3 URLs) against real USGS 3DEP
  (US) and real AHN4 0.5m DSM (Netherlands) — genuinely independent, non-training reference
  rasters, stated explicitly ("never the DEM consumed by calibration"). Aligned RMSE ranges
  2.08m (sparse-NL) to 6.62m (forested-Portland); r ranges 0.018 (ahn-sparse-flevoland — near-total
  failure) to 0.992 (ahn-hilly-limburg). Per-class aligned RMSE: hilly 2.998m, forested 4.735m,
  sparse 4.471m, urban 6.001m. Semantic-prior A/B table shows it helps on 3/8 scenes, hurts badly
  on 5/8 (e.g. +13.088m RMSE regression on ahn-hilly-limburg) — reported honestly, not
  cherry-picked.
- **(c) Implementation status:** Real, working — full `engine/depthwizard/{calibrate,benchmark}/`
  package, 1 commit visible via shallow clone (large monorepo with desktop/studio apps too —
  feels mature despite thin git history).
- **(d) Comparison:** Genuinely useful cross-validation — its semantic-prior failure pattern
  independently corroborates this project's own closed Method 3 finding (semantic priors
  overfit, don't generalize). Its `hybrid` DEM+model frequency-fusion mode is the concrete
  implementation of the exact idea this project flagged as unsolved. Worth reading
  `fit.py`/`terrarium.py` directly if pursuing that direction.

**Cluster 3 verdict — the most actionable cluster in this audit:** `arpitparashar06` and
`blakc-coffee` both do genuine, non-superficial frequency-separated DEM+DAv2 fusion; `blakc-coffee`'s
numbers are the more rigorously verified (committed JSON artifact, independent ground truth,
disclosed outliers), while `arpitparashar06`'s scale-calibration approach (shadow/GCP/known-height
alpha instead of regression) is architecturally the more novel idea, just unverified in-repo.
Both merit a closer read for this project's Method 4/5 follow-up work.

---

## Cluster 4 — Remaining ungrouped (investigated on their own merits)

### zaidnansari2011/sih2026-depthwizard — ⭐ standout of the entire investigation
- **(a) Technique:** Full fine-tune of the DA-V2-Small backbone (not calibration-only) into a
  twin-head (mean + log-variance) model predicting absolute AGL directly, in metres, via a
  learned output scale (`depthwizard/model.py`). Trained on GAMUS (DC/Philadelphia LiDAR nDSM) +
  DFC2019 (Jacksonville/Omaha LiDAR). Includes uncertainty calibration (ECE), a real held-out
  India validation against Google Open Buildings 2.5D, and terrain-leakage testing on 30°-slope
  Himalayan terrain.
- **(b) Numbers:** `docs/evaluation-protocol.md` reports, on DFC2019, 80 identical val tiles
  (protocol 2): zero-shot DA-V2 raw RMSE 9.998/MAE 3.856; zero-shot+global-affine (deployable
  baseline) RMSE 9.308/MAE 4.528; **fine-tuned run02 (ep11) RMSE 6.454/MAE 1.840** — beats the
  deployable baseline by 30.7% and beats even an oracle per-tile affine fit (RMSE 7.285). Ground
  truth: real airborne LiDAR (DFC2019/US3D), real GAMUS LiDAR nDSM. India cross-check: RMSE
  3.29m/MAE 2.37m/r+0.530 on 89 buildings vs. Google Open Buildings (real, independent,
  non-training reference) — but the doc itself corrects this: filtering to confident building
  pixels reveals the true bias is -6.94m underestimation on tall buildings, and states
  explicitly "do not quote -1.15m as the India result." This is a genuinely rigorous,
  self-correcting protocol (it documents its own past reporting mistake — mixing val/test
  splits — and fixes it).
- **(c) Implementation status:** Real, working, mature. 75 commits (unshallowed), full
  training/eval/data pipeline (`train.py`, `tools/evaluate.py`, `tools/compare_open_buildings.py`,
  `tools/analysis/terrain_leakage.py`, paired bootstrap significance testing, tripwire
  self-tests). No checkpoints committed (correctly gitignored — not fabricated, just not
  redistributed), but code to reproduce every claimed number is present.
- **(d) Verdict:** This is the strongest single result of the entire audit, and directly
  relevant — it fine-tunes the backbone rather than doing post-hoc calibration (like this
  project's Method 4, but going further: full fine-tune, not scale-modulation on frozen
  features), on the same DFC2019 ground truth this project uses, and beats an oracle per-tile
  affine fit, something this project's Method 4 did not achieve (RMSE gap vs. baseline narrowed
  to 4.3% but never closed). It also independently rediscovers this project's own
  building-underestimation finding on a different continent/dataset. Caveat: predicts AGL
  (height above ground), not absolute elevation — not a drop-in replacement, but the fine-tuning
  result is a serious existence proof worth studying. The source list's "seriously seriously
  great" hint is confirmed, not hype.

### RohitashvaSharma06/DepthWizard
- **(a) Technique:** Standard global-affine Huber-regression calibration (`H = s*D + t`), DEM or
  GCP mode, 20% held-out split — architecturally similar to this project's own closed Method 1/2.
- **(b) Numbers:** `MODEL_VALIDATION_RESULTS.md`: DA-V2 ONNX vs SRTM: MAE 50.225m/RMSE
  61.876m/r 0.9481/R² 0.897. **Critical caveat, stated by the repo's own doc:** the "optical
  image" used for this benchmark is synthetically rendered with physical hillshading and
  altitude-correlated luminance matching the reference DEM — i.e., not real satellite imagery,
  but a synthetic image manufactured to correlate with the DEM. This is circular validation,
  honestly disclosed but not a real-world test. Also notable: a hand-crafted heuristic fallback
  (Gaussian blur + luminance + texture energy) beats the actual DA-V2 model (22m vs 50m MAE) on
  this same synthetic benchmark. Unit tests (`test_calibration.py`) are pure synthetic
  linear-affine data (`np.random`, seed 42).
- **(c) Implementation status:** Real, working calibration code, but no real-imagery accuracy
  numbers exist for this repo — everything verified traces to synthetic/self-correlated inputs.
- **(d) Verdict:** Standard/unremarkable technique, and its one headline number is self-disclosed
  as not a real test. Its 50m+ MAE on the synthetic case, and the fallback-beats-ML-model result,
  are mildly consistent with this project's own finding that DAv2 doesn't reliably transfer to
  real nadir satellite correlation — but this repo doesn't actually test that claim on real
  imagery, so treat as "no real numbers found," not a counter-example.

### yats0x7/DepthWizard
See Cluster 3 above — reclassified there by the investigating agent since its `hybrid` mode is
a genuine frequency-fusion implementation.

### gowthamkrishna27/Elevate3d
See Cluster 1 above — reclassified there since its technique (RANSAC+Huber DEM/GCP calibration)
matches that cluster, though it was originally assigned to Cluster 4.

### lingeshmani23-tech/DepthWizard (bonus — metric-checkpoint question)
- **Finding: this is not a satellite/nadir terrain project at all.** It's a single ground-level
  RGB photo tool (`README.md`: "estimates target object heights... from a single outdoor RGB
  image") using DA-V2-Metric-Outdoor-Small-hf with a pinhole camera back-projection model
  (`calibration.py`, assumed 60° FOV) to measure the height of an object (person, pole) in a
  street-level photo, using a known reference object for scale — architecturally unrelated to
  satellite/nadir DEM work.
- **(b) Numbers:** `evaluation/results.csv` (real artifact, only 2 rows): test.jpg — actual
  2.45m, estimated 4.337m, **77.03% error**; sample_street_scene.jpg — actual 2.6m, estimated
  0.312m, **88.0% error**. Both catastrophically wrong.
- **(d) Verdict — answers the task's specific question:** This repo provides no evidence either
  way about whether the DA-V2 Metric-Outdoor checkpoint solves this project's satellite-elevation
  problem, because it's never applied to nadir/satellite imagery — it's tested only on
  ground-level photos, and even there it fails badly (77–88% error). Its own numbers are worse
  evidence against "metric checkpoints just work" than a repo that actually tried and failed on
  satellite imagery would be. Do not credit this as a validated alternative approach.

---

## Verification pass — 3 previously-known repos

Lighter verification (not full re-investigation) confirming specific claims against actual repo
files rather than prior beliefs.

### ArnabTechiee/depthwizard — confirmed, no material change, one new fact surfaced
- Latest commit: `664e92b`, 2026-09-16 (recent, active).
- README.md lines 128–129: "R² = −0.74" confirmed verbatim, measured against held-out shadow
  anchors on a dense urban scene using DAv2.
- README.md line 70 / table (Fort Myers, n=140, 70% USGS 3DEP LiDAR reference): Ridge height
  RMSE 3.88m confirmed verbatim, and separately reproduced in the model-ablation table (line
  142, "DAv2 Base | CPU, offline | 3.88 m").
- USGS 3DEP LiDAR ground truth confirmed (README line 65, cites apps.nationalmap.gov downloader).
- **New fact not previously flagged:** they also fine-tuned DAv2 on 1,000 GAMUS scenes, got
  R²=0.77 on GAMUS's own held-out split, but redeployed on their Fort Myers benchmark it
  "degraded every metric... correlation went negative" (README lines 152–155) — sensor/altitude
  domain gap. Reverted to the shadow-geometry approach. Worth noting as it reinforces
  (independently, on a different real dataset) this project's own DINOv3/GAMUS-style
  domain-transfer finding.

### ParmarManthanrajsinh/DepthWizard — confirmed exactly
- `results/evaluation.json`, `primary_frozen_global_protocol` block: `mae_meters: 4.0652`,
  `rmse_meters: 4.6258`, `pearson_correlation: 0.645` — matches the claimed 4.07/4.63/0.645 to
  the stated rounding.
- Protocol is genuinely frozen: scale/offset (`scale: 9.810477`, `offset_meters: 33.73373`)
  fitted only on the validation split (`results/global_calibration.json`, `val_mae_meters:
  4.3654`) then applied unchanged to test — no test-set leakage.
- Ground truth: ISPRS Potsdam DSM (real published dense-DSM benchmark, not synthetic), 6
  held-out tiles, 150 test crops, 39.3M pixels.
- Also present, not previously known: an oracle upper bound (per-image-fit scale/offset) at MAE
  1.53m/RMSE 2.16m — this is the ceiling if scale drift per scene were correctable, i.e. their
  own honest gap analysis.

### amogh-hub/depthwizard
See full write-up under Cluster 1 above.

---

## Explicit shortlist — what this project should seriously consider

1. **`zaidnansari2011/sih2026-depthwizard`** — top priority. Same DFC2019 benchmark, beats an
   oracle affine fit via full backbone fine-tuning (vs. Method 4's frozen-feature
   scale-modulation). Worth replicating its fine-tuning protocol as a new method attempt — it's
   the only repo in the entire batch that materially outperforms what this project has already
   tried, on directly comparable ground truth.
2. **`blakc-coffee/depthwizard` and `arpitparashar06/depthwizard`** — both tackle the open
   frequency-fusion problem directly. `blakc-coffee` has the more rigorously verified numbers;
   `arpitparashar06`'s shadow/GCP-based scale-recovery (instead of DEM regression) is worth
   studying specifically to avoid the ramp-leakage trap.
3. **`devendrakushwah80/DepthWizard-SIH26175`** — mine its GSD-FiLM conditioning and
   height-balanced loss for ideas to close Method 4's RMSE gap (not directly comparable —
   different imagery domain).
4. **`amogh-hub/depthwizard`** — worth adopting its evidence-gating / leave-one-out /
   calibration-eval-separation code pattern as a rigor template, even though its own accuracy
   doesn't beat Method 2.

Nothing in Cluster 2 solves the coarse-DEM-as-CNN-target memorization failure — that stays an
open problem this project will have to solve itself.

Other notable flags surfaced along the way:
- `Haresh-kumar28/DepthWizard` and `prashant-d4/DepthWizard` share byte-for-byte identical
  calibration code — same underlying codebase, not independent implementations.
- `Kukyos/DepthWizard`'s source-list hint ("real metric AGL predictor... evaluated on GAMUS") is
  overclaimed — the repo's own docs say the metric-accuracy phase isn't built yet.
- `RohitashvaSharma06/DepthWizard` and `gowthamkrishna27/Elevate3d`'s only numbers trace to
  synthetic/self-correlated ground truth, not real imagery.
- `amulyadsouza-tech/DepthWizard` and `SumitRoy-Gh/Depthwizard` have real calibration code but
  zero real accuracy numbers.
- `madhu-mitha-e/DepthWizard` is no longer empty (contrary to the earlier check) but its own
  fusion result is negative/small.
- `lingeshmani23-tech/DepthWizard` is off-topic (ground-level object-height estimation, not
  satellite/nadir) and fails badly (77–88% error) on its own 2-sample eval.
