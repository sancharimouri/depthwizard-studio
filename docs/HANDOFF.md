# DepthWizard2 (SIH26175) — Handoff

Read this once, act on it. This is the organized reference; `docs/method-audit/sentinel2/sign-flip-detector.md`
and the per-method `docs/method-audit/*/` folders are the chronological logs — go there for
blow-by-blow detail, come here for "what's the state and what's next."

## 0. STANDING RULE: the 5 protected links (owner decision, 2026-10-01)

These 5 links must **never break**, whatever the change:

1. https://github.com/sancharimouri/depthwizard2-desktop/releases/latest
2. https://depthwizard-studio.vercel.app/
3. https://depthwizard-studio.vercel.app/#demo-video
4. https://github.com/sancharimouri/depthwizard-studio
5. https://depthwizard-studio.vercel.app/#/demo

**What this forbids, in practice:**
- **Repos:** never rename, delete, transfer or make private either GitHub repo. A history rewrite or force-push is
  allowed only if the repo URL is unchanged.
- **Desktop releases:** every one must be a full, non-draft, non-prerelease release with installers, so
  `/releases/latest` always resolves.
- **Vercel:** never rename the project or change its domain.
- **Frontend changes** (including the coming UI redesign) must keep the element `id="demo-video"` on the home page and
  the hash route `#/demo`, both working. See `frontend/src/routes.js` and `frontend/tests/routes.test.mjs`.
- **Backend cutover:** the web app must keep working throughout. Switch `VITE_API_BASE` only after Cloud Run is
  verified. Shut Render down only after the new setup is confirmed live.

**Enforcement:**
- `scripts/check_protected_links.sh` checks:
  - that all 5 URLs return HTTP 200 after redirects, with `/releases/latest` resolving to a release tag;
  - that the live home page serves `id="demo-video"`;
  - that the local web build keeps `id="demo-video"` and the `#/demo` route.
- **Run it before ANY deploy, push, release or merge, and report the result. If a check fails, stop and tell the
  owner.**
- **Baseline 2026-10-01: ALL PASS.** `/releases/latest` → `v1.0.2`.

## 1. Project overview

Depth Wizard turns satellite RGB into 3D terrain visualization for SIH26175. There are
**two separate tracks** — do not conflate them:

- **DFC2019 ML research track** — offline model-improvement research using the DFC2019
  dataset, which has **dense real LiDAR ground truth**. This is a proxy-validation
  sandbox: if a correction method can't even win here, it's not worth testing on the
  real target domain.
- **Sentinel-2/India track** — the actual deployment domain (Darjeeling, Kolkata,
  Bardhaman, Sundarbans-style 10 m Sentinel-2 imagery). No dense ground truth exists
  here — only sparse ICESat-2 photon footprints and coarse DEMs (SRTM/Copernicus
  GLO-30). A method winning on DFC2019 does **not** automatically transfer (see Method 6
  below — it wins on DFC2019, loses on Sentinel-2).

Neither track is deployed. The live demo (frontend/backend) is a separate, frozen
system — see §5.

## 2. Current state per track

### 2a. DFC2019 track

**Method 6 claim (owner's wording, 2026-10-01):** validated on DFC2019 (US cities, satellite imagery); did not generalize to GAMUS by RMSE. Expected accuracy is the 3-seed cross-validation result on DFC2019: MAE 1.990 ± 0.010 m, RMSE 3.504 ± 0.026 m, Pearson 0.743 ± 0.002, Spearman 0.656 ± 0.0003.

DFC2019 research best: **Method 6**: full DAv2-Small fine-tune, twin (mean, log-variance) head,
height-balanced recipe (`CappedHeightWeightedLoss` + `WeightedRandomSampler`). 50-tile DFC2019,
4-fold spatial-quadrant holdout, mean of fold means, 95% tile-bootstrap CIs.

| | MAE (m) | RMSE (m) | Pearson | Spearman |
|---|---|---|---|---|
| **Method 6** (seed 1) | **1.980** [1.691, 2.297] | **3.492** [2.964, 4.106] | **0.745** [0.704, 0.779] | **0.656** [0.616, 0.692] |
| Method 6, 3 seeds (mean ± sd) | 1.990 ± 0.010 | 3.504 ± 0.026 | 0.743 ± 0.002 | 0.656 ± 0.0003 |
| Oracle per-tile-OLS | 3.392 [2.963, 3.865] | 4.579 [3.985, 5.229] | 0.582 [0.521, 0.639] | 0.509 [0.456, 0.559] |
| Method 2 Grid+Huber+20: sparse-GCP baseline, **not** an oracle (see `final-comparison.md` §1.0) | 2.929 | 4.718 | 0.532 | 0.471 |

- Every seed beats the oracle on all four metrics (pre-registered C3 rule, so **the headline
  stands**).
- Tiles better than the oracle: 47–49/50 per metric.
- Variance ratio 0.48–0.66: still underdispersed.
- The Maxar VHR check shows non-degenerate output on 2 of 3 crops; accuracy untested.
- **VHR → absolute DSM pipeline exists (2026-09-23).** DSM = FABDEM + max(Method 6 AGL, 0) on 6 Maxar
  Sikkim/Darjeeling crops, rendered through the unmodified viewer (`frontend/vhr_preview.html`, unlinked).
  Plausible on textured terrain: buildings and crowns are resolved, no seams, and C4 ρ is +0.45 to +0.68. Two defects: a tile-grid
  artifact on textureless snow, and an AGL ceiling of about 18–23 m (the forest is under-tall compared with GLO-30−FABDEM and GEDI).
  Plausibility only; there is no ground truth. `docs/method-audit/06-full-finetune-twin-head/vhr_dsm_pipeline.md`.
  - **Follow-ups:**
    - **Seam fix.** `--margin=192`, which discards a 192 px ring per window, cuts the glacier seam ratio from
      2.37 to 1.07. Use it for any future full rerun; the five committed crops still use the original tiling.
    - **Height ceiling = training range, confirmed for canopy.** DFC2019 JAX tree pixels reach p99 23.6 m.
      Held-out DFC2019 trees that are 25–50 m are predicted at about 16 m. Buildings still reach 43 m p90,
      so this is not a capacity limit.
    - **GAMUS DC/NYC has about 10× more tree pixels above 25 m**, but tops out around 35 m. It's the candidate
      for a retrain; that retrain is not authorized. Its pre-registered test is in the doc.
    - **Retrain done (DFC2019 + GAMUS-DC, leakage-safe split).**
      - DFC2019 MAE 1.920 (all 4 folds better; single seed) is a candidate, not adopted.
      - **Tall-tree criterion FAILED:** held-out 20–30 m trees are predicted at 14.1 m (was 13.9; target 18).
      - The VHR forest ceiling doesn't rise: the whole distribution shifts up by about 2.6 m. Bare brown terraces gain height.
      - Cause: GAMUS DC/NYC imagery is leaf-off (9% of tree pixels are green, vs. 93% in the Sikkim forest).
        NYC imagery is real (`*_IMG.h5`) but just as leaf-off, so it wasn't added.
- Everything: `docs/method-audit/final-comparison.md` §1–2.

**Resolution transfer (2026-09-24, `06-full-finetune-twin-head/resolution-transfer.md`; pre-registered; seed 42):**
- Native DFC2019 GSD is **0.3 m** (a 512 px quadrant is 153.6 m). Two protocols: P = true pixel count (the prompt's), R = resample-back (37×37 tokens kept).
- DAv2's ViT interpolates position embeddings natively (Phase 0), so no fix was needed. The new training script reproduces the adopted fold 0 bit-for-bit.
- **P: coarse→fine transfer fails at 2/3/5 (and 8) m** (0/4 metrics beat the 0.3 m oracle). Too few tokens: 6×6 → 2×2.
- **R: coarse→fine transfer works at 2/3/5 m** (3/4 metrics; native-eval Pearson 0.721 / 0.708 / 0.687 vs. oracle 0.582). It fails at 8 m (0.631, CI-overlapping); the curve bends between 5 and 8 m. **Boundary sweep (2026-09-24):** the flip is between 5 m (pass, 3/4) and 6 m (fail, 2/4); 7 m is 1/4 (Pearson 0.670 / 0.660). One seed; ±1 step. Stopped for a user decision on next steps.
- **Strongly asymmetric, coarse→fine better** everywhere. E.g. R at 5 m: 0.687 (5 m-trained at 0.3 m) vs. 0.321 (native model at 5 m).
- Synthetic degradation of one WorldView-3 city. This is **not** evidence that real 10 m Sentinel-2 supports this.

**Sentinel-2 token-grid test (2026-09-24, `06-full-finetune-twin-head/sentinel2-token-grid-test.md`; pre-registered). The R finding does NOT extend to real Sentinel-2.**
- **DFC2019 GSD re-derived from tile content (lane cycles, lane widths, trucks): ≈ 0.3 m.** The "0.5 m verified" figure is retracted.
  The US3D spatial join (0/50) was rasterised at the wrong scale, so it is not a clean negative; that is an open item.
- **Phase A.** Method 4's three Sentinel-2 attempts had a frozen DAv2 (37×37 tokens, 270 m per token) and are uninformative about R.
  Method 6 staged on Sentinel-2 had 36×36 tokens (R-like by count) and still failed ICESat-2 (2/25).
- **Phase B (synthetic).** R fails at 10 m and 12 m (0/4 metrics). Pearson for 5 / 8 / 10 / 12 m is 0.687 / 0.631 / 0.610 / 0.551 vs. the oracle's 0.582.
- **Phase C (real Sentinel-2, Kaggle T4).** Identical 600 m crops and FABDEM-30 m target; only the token geometry differs.
  R and P tie: ICESat-2 RMSE 16.54 vs. 16.60 m, 16/32 tiles, Holm p = 0.82. Both are about 6× worse than raw FABDEM (2.85 m).
- **Phase D:** not triggered.
- **Rank-loss test (final check; the line is now closed).** Same setup, loss → Method 4 v2 `rank_pair_loss`, per-tile post-hoc slope calibration.
  Against the identically calibrated frozen-DAv2 oracle: ICESat-2 16.25 vs. 16.74 m, 18/32 tiles, **Holm p = 0.54** (no real signal); still about 5.7× worse than FABDEM.
  Descriptive, not pre-registered: it beats Phase C's magnitude loss by 0.35 m (28/32). **The Sentinel-2 learned-terrain-correction line is CLOSED; don't reopen.**
- **Raw-Spearman check (Case 3):** within-crop Spearman vs. ICESat-2 is rank model 0.134, DAv2-L 0.109, FABDEM 0.631. It is fully expressed by the linear calibration (gap +0.025, 16/32, p = 0.41), so **isotonic is not indicated**.
- **Landsat 8/9 pan (15 m) sensor-identity test: Case B.** Same footprints, grid, recipe and ICESat-2 cells.
  Raw within-crop ρ: rank model 0.116 vs. S2 0.134 (p = 0.09); DAv2-L 0.115 vs. 0.109. Calibrated 16.36 vs. 16.25 m (p = 0.80). No raw/calibrated divergence.
  **Not Sentinel-2-specific.** Any further data should target GSD ≤ ~5 m, regardless of sensor. Nothing started.
- **L1C vs L2A (16 tiles, same acquisitions):** the pre-registered bar (≥ +0.10) is not met, but there is a small, reliable L1C advantage.
  Raw ρ 0.218 vs. 0.175, +0.043, 13/16, p = 0.004; replicated with a stretch (p = 0.008). Calibrated −1.85 m.
  Post-hoc it is **entirely hilly** (Δρ +0.13 to +0.18, 4/4 tiles, −7.3 m). Likely altitude-dependent haze (aerial perspective) that Sen2Cor removes.
  No product relevance (L1C only ties its oracle; ~4.5× worse than FABDEM). A possible follow-up (a hilly-only test) is not started.
- **Haze-mechanism check (no training):** the rule is formally met (L1C blue vs. FABDEM ≤ −0.30 on 3/4 hilly tiles), but it is mostly a land-cover gradient.
  The haze component is +0.003 on Almora and only small elsewhere, and within 600 m crops L1C blue shows no elevation relation (|ρ| ≤ 0.06).
  **So haze is not the main explanation.** The untested alternative is Sen2Cor topographic illumination correction (test: L2A/L1C ratio vs. cos(incidence)). Not started.
- **DFC2019 positive control (2026-09-24): PASS (not strong).** The same unchanged recipe and measure on native 0.3 m DFC2019 against dense LiDAR AGL (31 tiles, `scripts/dfc2019_rank_positive_control.py`).
  Raw within-crop ρ: rank model **0.379** [0.345, 0.413] and DAv2-L 0.441, against 0.13/0.11 on Sentinel-2. Every tile is > 0.134.
  **The raw-Spearman nulls on coarse sensors are real, not a blind spot.** But calibrated RMSE: the rank model loses to frozen DAv2-L even here (3.92 vs. 3.35 m, 5/31).
  So the calibrated "no real signal" rule means "rank training didn't beat DAv2-L", not "no signal". Caveat: 18 m crops and AGL, vs. 600 m and terrain.

- **Terrain-relief positive control (run 2026-09-25 on Kaggle): STRONG PASS.** VHR NAIP at 1 m vs the 3DEP LiDAR DTM, in 600 m crops with 30 m cells;
  8 tiles, as pre-registered (`sentinel2-token-grid-test.md`, final section).
  - Raw within-crop ρ: rank model **0.730** [0.688, 0.768] vs frozen DAv2-L 0.250, 8/8 tiles.
  - Calibrated RMSE: 30.96 vs 44.13 m, 8/8, Holm p = 0.016.
  - **The Sentinel-2 / Landsat / CBERS terrain nulls (ρ ≈ 0.12–0.13) are a genuine absence of signal at 10–15 m, not a test blind spot.**
  - Output is in `data/terrain_relief_control/kaggle_out/`; the checkpoints are local only.
  - Caveats: one seed, 3 US forest sites, within-tile quadrant hold-out.

**Generalization test (2026-09-23, `docs/method-audit/07-gamus-generalization/`): Method 6 does NOT generalize.**
- **GAMUS** (2,861 aerial test tiles, DC/NYC/PHL, 0 leakage). Method 6 vs. the oracle, mean of tiles:
  - MAE 3.130 vs. 3.474; RMSE **4.583 vs. 4.426**; Pearson 0.638 vs. 0.491; Spearman 0.583 vs. 0.425.
  - It wins MAE, Pearson and Spearman but **loses RMSE in all 3 cities for every seed** (tall-object compression; pooled variance ratio 0.30).
- **US forest and mountain vs. airborne LiDAR** (USGS 3DEP, 8 windows):
  - canopy p95 is 10.6 m against 37.7 m;
  - "DEM + predicted height" does not beat the DEM for SRTM, GLO-30 or FABDEM. FABDEM + AGL's RMSE gain is bias-only.
- **GAMUS fine-tune:** a pre-registered stop, because no GAMUS city is leaf-on.
- **Product model: none adopted.** The product recommendation stays **DEM-only** (FABDEM terrain, GLO-30 surface).
  Method 6 remains the DFC2019 research best only.
- Seed-42 fold checkpoints were re-created bit-identically (`method6_height_balanced_seed42_ckpt/`). All 3 seeds × 4 folds now exist.

- Docs: `docs/method-audit/06-full-finetune-twin-head/{summary,verdict}.md`
- Eval/train scripts: `scripts/evaluate_method6_finetune_twinhead.py`,
  `scripts/evaluate_method6_gsd_film_height_balanced.py`,
  `scripts/evaluate_method6_sentinel2.py`, `scripts/train_method6_full_dfc2019.py`,
  `scripts/method6_vhr_sanity_check.py`
- Experiment outputs: `data/dfc2019/experiments/method6*/` (seed runs `method6_height_balanced_seed{43,44}/`, which save per-fold checkpoints; uncertainty `method6_uncertainty.json`)
- **Caveat: DFC2019-only.** Staged on Sentinel-2/SRTM (item 2b) and lost.

Methods 1–5 (1–4 closed/superseded; 5 open on DFC2019 only, see §3.2; doc location in each case):
1. Global DEM-stat calibration — CLOSED, no real improvement. `docs/method-audit/01-dem-stat-anchoring/`
2. Sparse-anchor/GCP regression — CLOSED as standalone. Best variant Grid+Huber+20 anchors
   (2.929/4.718/0.532/0.471): a deployable sparse-GCP baseline, **not** the oracle. The oracle is
   the per-tile-OLS row, 3.392/4.579/0.582/0.509; both are traced in `final-comparison.md` §1.0.
   `docs/method-audit/02-gcp-regression/`
3. Semantic prior (linear building-probability term) — CLOSED, overfit, didn't
   generalize spatially. `docs/method-audit/03-semantic-prior/`
4. Learned CNN scale-modulation — superseded by Method 6, not current best; still the
   best frozen-feature approach (MAE 2.8803m/RMSE 4.7751m/Pearson 0.5835/Spearman
   0.5438). `docs/method-audit/04-learned-scale-modulation/` (full table in `v2-results.md`)
5. RDAH-Net fusion — **open on DFC2019 (low priority), closed on Sentinel-2** (§3.2, §4). `docs/method-audit/05-rdah-net-fusion/`.
   Five separate results:
   - zero-shot on Sentinel-2: closed. The resolution cliff on DFC2019 is the basis; the
     checkerboard is intrinsic and isn't (§2b)
   - zero-shot on DFC2019 with corrected ×255 input: 2.231/4.566/0.716/0.655. Track1 checkpoint
     (41/50 tiles contaminated), tile-level folds; artifact rescued to
     `data/dfc2019/experiments/rdah_zeroshot/`
   - Swiss zero-shot on DFC2019, FT-2's samples: 3.033/6.421/0.492/0.542 (TRANSCRIBED from a
     session log)
   - RDAH-FT-1: 2.906/6.659/0.513/0.527 pooled, unstable across folds
   - **RDAH-FT-2** (2026-09-23): Swiss init, per-fold input scale, quadrant folds, rank loss,
     nested selection. **2.500/4.294/0.640/0.506** (per-sample mean, Method 6's aggregation);
     2.499/5.598 pixel-pooled. **Not adopted**: loses to Method 6 on all four metrics. Fold
     Pearson range 0.503–0.607 (FT-1: 0.254–0.607). Variance ratio 0.19–0.23, still severe
     underdispersion. Aggregate: `data/dfc2019/experiments/rdah_quadrant_cv/rdah_ft2_aggregate.json`
     (`scripts/aggregate_rdah_ft2.py`)

### 2b. Sentinel-2/India track

**Current deployable baselines** (final, 2026-09-23; all 32 benchmark tiles, independent ICESat-2,
per-point geoid, R4 offset-guarded; `data/sentinel2_benchmark/dem_baselines_32/summary.json`):

| use | product | median RMSE vs. ICESat-2 [95% tile-bootstrap CI] | vs. runner-up |
|---|---|---|---|
| **terrain (bare earth)** | **FABDEM** (EGM2008) | ground photons: **1.782 m** [1.068, 3.417] (1.87% of range) | beats GLO-30 on 32/32 tiles, RMSE and bias-removed RMSE, p = 4.7e-10 |
| **surface (DSM)** | **Copernicus GLO-30** (EGM2008) | 20 m canopy-top segments: **4.484 m** [3.363, 8.846] | beats SRTM: RMSE 24/32 (p = 6.6e-4), bias-removed 31/32 |

- **No depth-model or canopy-model add-on beats these.**
  - (2026-09-23, 07 Part F) A Song et al. 2026 HRF random-forest residual on FABDEM (held-out ICESat-2 tracks) is not adopted: 19–21/32 vs. raw, and it loses to a per-tile linear residual. The linear residual's own gain is offset-only.
  - Frequency fusion equals its DEM-only control (10/25, p = 0.853), and DAv2 detail has median
    r_HF −0.037. **The DEM carries the frequency-fusion result.**
  - No detail source passes any reference (Phase 4).
  - No DEM + canopy product passes (A3).
- **Height-above-ground signal at 10 m exists but is weak within a scene.** CHMv2 and ETH pass a
  pre-registered direct test (pooled Spearman 0.641 / 0.378 vs. ICESat-2). Within-tile it's only
  0.28 / 0.23 (post-hoc).
- The earlier "frequency fusion 21/25, 10.88% → 3.57%" headline compared against
  linear-calibrated DAv2, which is weaker than the raw DEM. It's kept only as history.
- Everything, with independence flags: `docs/method-audit/final-comparison.md` §3.

- **Single running log for all Sentinel-2 work**: `docs/method-audit/sentinel2/sign-flip-detector.md`
  (calibration, sign-flip detection, frequency fusion, evidence-gating/LOBO,
  semantic-prior phase 2.3 incl. its closing re-investigation, RDAH zero-shot, DEM-only
  controls, surface references, detail-source bake-off, datum audit, 32-tile DEM baselines,
  FABDEM, direct height test, RDAH rerun — all entries dated, most recent 2026-09-23 "(final
  close-out)")
- Fusion script (superseded as the recommendation, kept as the reference pipeline):
  `scripts/run_frequency_fusion_sentinel2.py`. Result CSVs, now committed:
  `frequency_fusion_results/`, `srtm_3way_comparison.csv`, `frequency_fusion_controls/`.
- Surface references (2026-09-23, all 32 tiles + Darjeeling):
  - ICESat-2 20 m PhoREAL segments `data/icesat2_segments20m/` (committed). Sliderule's `ats` was
    lowered to 5; the A1.3 sensitivity check changes no conclusion. 25 ATL03 granules failed
    server-side (`data/icesat2_segments20m_failed_granules.csv`).
  - GEDI L2A `data/gedi_l2a/` (committed).
  - FABDEM and ETH on each tile's grid: not committed, regeneration in `data/REGENERATION.md`.
- **Datum:** SRTM is EGM96; GLO-30 and FABDEM are EGM2008 (verified A0). Use the per-point geoid
  (`scripts/dem_baselines_32.py`).
  - **PROJ_NETWORK must be set before any pyproj import**, or N silently becomes 0.
    `frequency_fusion_controls.build_tile` now raises if it does.
- **Darjeeling DEM — resolved 2026-09-23.** The "GLO-30 all-NaN for Darjeeling" was a
  **tile-selection fetch bug**, not a data gap.
  - Darjeeling's footprint (lat 26.9999–27.0901) straddles the 27°N boundary of the 1° GLO-30
    tiles. Taking the single tile at the footprint's southern edge (`floor(26.9999)` → N26) leaves
    one pixel row, so the crop is **0.31% valid**.
  - Reproduced against the public AWS COGs. Neither a CRS error nor a swallowed API error.
  - The original fetch script doesn't survive; the claim predates every transcript, so the exact
    code can't be inspected. The failure mode is reproduced exactly.
  - Correct fetch: `scripts/fetch_glo30_darjeeling.py` mosaics N26 + N27 →
    `data/elevation/darjeeling/Darjeeling_Copernicus_GLO30_DSM_cropped.tif` (committed; 100%
    valid, 556.55–2477.54 m). The full 85 MB two-tile mosaic is not committed (see
    `data/REGENERATION.md`).
  - **Consequence for the demo: none needed.** The demo's `Darjeeling_OpenTopography_DSM.tif` is
    **byte-identical to GLO-30 on all 117,325 pixels**, because OpenTopography served COP30. The
    demo already uses GLO-30 and matches the other three regions' source.
  - The only substantive switch still open is to **FABDEM** (bare earth; A3 found it best against
    ICESat-2 ground photons). That is an optional demo decision, **not made**.
- Benchmark data: `data/sentinel2_benchmark/` — `manifest.csv` (25 accepted tiles of
  32 originally selected), `icesat2_coverage.csv`, `srtm_raw/`, `copernicus_dem_raw/`
- **Content-QC history, so it isn't rediscovered as a surprise**: the benchmark started
  at 32 tiles (8 each: agricultural/coastal/hilly/urban) selected via real ICESat-2
  ATL08 coverage. A content-quality audit (`content_audit.csv` →
  `content_audit_corrected_32.csv`) found real problems in 14/32 tiles, which were
  replaced/corrected. Separately, the sign-flip detector then flagged 7 of the
  resulting set as unreliable, leaving **25 tiles as the actual working set**. Two
  different filters, two different reasons — both already resolved.
- Backbone comparison (DAv2 vs DINOv3 SAT493M+CHMv2, closed, DINOv3 wins pooled but
  DAv2 wins on agricultural/hilly specifically): `docs/method-audit/sentinel2/backbone-comparison.md`
- Three independent CNN-correction attempts on top of frequency fusion all failed
  differently (SRTM-interpolation memorization, native-30m data starvation,
  Open-Buildings-target memorization) — **CNN correction is not pursued further
  without new evidence**, this is a deliberate stop, not an open thread. Detail in
  sign-flip-detector.md.
- **RDAH-Net on Sentinel-2: CLOSED by the pre-registered rule (A5).**
  - The Darjeeling rerun with training-exact preprocessing scores Spearman +0.026 vs. ICESat-2
    canopy height and −0.010 vs. GEDI rh98, against a reopen bar of ≥ 0.30 on both.
  - This replaces the earlier post-hoc "default clause" closure.
  - The checkerboard is intrinsic (it appears in-domain) and was never a valid criterion.
  - Scripts: `scripts/rdah_darjeeling_rerun.py`, `scripts/rdah_resolution_sweep.py`.

## 3. Long-term plan — open items, priority order

**New, 2026-09-24 (resolution transfer):**
- ~~**0d.**~~ **CLOSED negative for Sentinel-2 (2026-09-24, `06-full-finetune-twin-head/sentinel2-token-grid-test.md`).** Synthetic R fails at 10 and 12 m. On real Sentinel-2, R (39×39 tokens) = P (5×5): ICESat-2 16/32 tiles, Holm p = 0.82. Don't reopen for 10 m imagery. A real 2–5 m sensor is still untested.

**New, 2026-09-23 (07):**
- **0a. Leaf-on, tall-forest supervision for the canopy ceiling.** GAMUS is leaf-off, so it can't supply it.
  - Candidates: NEON AOP (needs a token), 3DEP + leaf-on NAIP at scale (`scripts/forest_mountain_3dep_eval.py`), or an ExG-selected PHL leaf-on subset (post-hoc idea).
  - Must pass the Sikkim non-regression criterion (`07-.../log.md`, C.0).
- **0b. Part D rerun on NEON AOP** if a NEON API token is added to `.env`.
- **0c. Deliverables gaps** (`docs/deliverables-audit.md`): rDSM and metric-DSM paths, GeoTIFF export, slope/height analysis, in-UI validation, packaging. A user decision; the frontend stays frozen until then.

1. **TSE-Net (self-training)** — untouched, no code or docs. The only substantive open method.
2. **RDAH on DFC2019: Track1 seen-vs-unseen split** — low priority, inference only.
   - Run Track1 zero-shot on its 9 `Track1-test` vs. 41 `Track1-train` tiles, to separate
     memorised tiles from in-domain training.
   - "Fine-tuning damages the model" is resolved (no): Swiss zero-shot 0.492 Pearson, FT-2
     better in 4/4 folds.
   - FT-2 is not adopted.
3. **Learned Sentinel-2 + GEDI route — GATED, not recommended.**
   - Why: the pre-registered ETH ceiling fired. ETH GCH 2020, a global Sentinel-2 + GEDI model,
     fails both surface references as a DEM add-on, and FABDEM + ETH fails too (A3).
   - **Reopen gate:** a canopy/object-height source passing Test B against ICESat-2 surface.
   - What it would fix: many tiles of direct GEDI labels (not one tile's DEM) and a sparse lidar
     target (not an interpolated one), i.e. both diagnosed CNN failure modes.
4. **Sparse-LiDAR 27-feature RF** — untried, low priority. Blocked on DFC2019 (no
   georeferencing, `stage0-gates/sparse-lidar-feasibility.md`). If revisited, prototype on
   Sentinel-2 against the new DEM-only baselines, not against fusion.

**Closed, 2026-09-23:**
- semantic-prior phase 2.3 (clean negative, both mechanism checks clean; log, commit `5cc5e80`)
- RDAH on Sentinel-2 (A5)
- detail add-ons (Phase 4)
- DEM + canopy products (A3)

## 4. Rejected list (don't redo these)

- Global/pooled calibration (Method 1)
- RANSAC for sparse anchors (both rounds — `sparse_anchor_ransac_v2` and predecessor)
- Spatial-local calibration variants: quadrant-wise, smooth RBF residual
  (`sparse_anchor_spatial*`, `sparse_anchor_smooth_residual`)
- Semantic-prior as a **linear interaction term** (Method 3's original DFC2019 form —
  note phase 2.3's closed-form building-aware tests on Sentinel-2 are a *different*,
  also-negative, already-closed result, not a redo of this)
- All CNN-based Sentinel-2 correction — 3 independently diagnosed failures:
  SRTM-interpolation-memorization, native-30m-resolution data-starvation,
  Open-Buildings-target memorization
- GSD-FiLM conditioning on Method 6 (DFC2019 has no real per-tile GSD variation —
  clean wash, not adopted)
- arpitparashar06's non-regression scale derivation
- ArnabTechiee's shadow-geometry photogrammetry (fails at 10m GSD, both physics and
  empirically — `data/sentinel2_benchmark/shadow_photogrammetry_plausibility.csv`)
- RS3DAda (contaminated on DFC2019 — 49/50 benchmark tiles were in its own training
  split; degenerate on Sentinel-2)
- Sparse-LiDAR-Guided-Correction's **DFC2019** feasibility specifically — blocked, no
  recoverable georeferencing (Sentinel-2 domain not equally blocked, see §3.4)
- RDAH-Net **on Sentinel-2**, zero-shot or fine-tuned (2026-09-23). Closed by the pre-registered
  A5 rule: Darjeeling Spearman +0.026 / −0.010 vs. ICESat-2 / GEDI height above ground. The
  checkerboard is *not* the reason; it's intrinsic.
- **DEM + canopy-height products** (FABDEM + ETH, FABDEM + CHMv2; 2026-09-23, A3). Both lose to
  raw GLO-30 on ICESat-2 surface heights under the R4 offset guard (4/32 and 0/32).
- **Frequency fusion as a recommendation** (2026-09-23). It equals its DEM-only control.
- **DAv2 (or any tested no-training source) as a high-frequency detail add-on for Sentinel-2**
  (2026-09-23): DAv2 @518/@1008, DINOv3-CHMv2, ETH canopy height. None passes against ground
  photons, ICESat-2 20 m surface or GEDI (max median r_HF 0.089 < 0.10). Frequency fusion's gain
  was the DEM low-pass.
- **Learned Sentinel-2 + GEDI canopy/surface-height route** (2026-09-23): not started and not
  recommended. The pre-registered ETH ceiling check fired: ETH GCH 2020, itself a global
  Sentinel-2 + GEDI model, doesn't pass either surface reference, and worsens the ICESat-2
  surface product by double-counting with SRTM (8.32 vs. 5.00 m).
  - That route *would* address both earlier failure modes: ~10⁵–10⁶ direct GEDI labels across
    many tiles instead of one tile's DEM, and a sparse direct-lidar target instead of an
    interpolated one.
  - **Gate to reopen:** a canopy/object-height source passing Test B against ICESat-2 surface.
    The `bare-earth DTM + canopy` form (FABDEM + ETH) has now been tested and fails (A3). Also
    listed as gated in §3.3.

**Explicitly excluded from this list: RDAH-Net on DFC2019.** It is open (low priority), see §3.2. Only its Sentinel-2 use is closed (above). The specific
**RDAH-FT-2 recipe** (Swiss init + per-fold scale + quadrant folds + rank loss, 5 epochs) was
not adopted (loses to Method 6 on all four metrics). That closes one recipe, not the method.

## 5. Frontend/backend status (live demo — frozen, separate from all of the above)

The demo stays intentionally separate from every ML research finding above, including
Method 6's DFC2019 win. **A real DSM is the terrain source; DAv2 output is shown only
as a labeled relative-depth *visualization* layer, never as the thing that produced the
terrain.** None of the research-track work is deployed into it.

**Page order (2026-09-24, user-directed; navigation only, no DSM/DEM behaviour changed):**
- **1 Workbench** (the default landing page: scene search, upload, prototype generation), **2 Explore** (the 4-region demo) and
  **3 Docs** (new: what the tool does, how the pages work, the real capabilities and limits).
- Page containers were renamed from `#page-1` / `#page-2` to semantic IDs `#page-workbench` / `#page-explore` / `#page-docs`.
- The Workbench elevation captions were reworded: the stale "RDAH / Prior2DSM correction head … not yet trained" text is gone.
  They now say DEM elevation, no learned correction, and Sentinel-2 corrections tested and not adopted.
- Verified in headless Chrome: all pages and flows work, with 0 JS exceptions (the only console error is the pre-existing favicon 404).

**Curated library catalog — Prompt 1 of prompts.pages (2026-09-25).**
- **Generator:** `scripts/library_catalog.py` (`curate` then `build`) turns the curated folders into `data/library/manifest.json`,
  with real thumbnails and previews. The folders hold 50 DFC2019 + 32 Sentinel-2 + 6 Maxar VHR crops; Landsat, CBERS and L1C are
  excluded by construction and by a build guard.
- **API:** `/api/library` (list with collection/tier filters), `/{id}`, `/{id}/thumbnail`, `/{id}/preview`, and `POST /{id}/select`,
  which enforces routing: Sentinel-2 is always Tier 1 (DEM only); DFC2019 and VHR are Tier 2.
- **Caveat:** DFC2019 has no georeference, so Tier 2 there means above-ground height only, with no DEM.

**Page 1 input view — Prompt 2 of prompts.pages (2026-09-25; `frontend/src/input-view.js`, `backend/input/store.py`,
`backend/api/input_routes.py`, `backend/dem/fabdem.py`).**
- **Layout:** Library / Upload / Search Online tabs on the left half; preview, metadata, tier card, DEM card and START GENERATION
  on the right half. The old scene-input box and its two modals are gone; START hands the input to the existing generation grid.
- **Tier colours (user to confirm):** green = Tier 2, orange = Tier 1, neutral = relative preview only. Tokens `--tier2-rgb` /
  `--tier1-rgb`; same hue in both themes.
- **Upload:** GSD comes from the geotransform (degrees converted to metres; the coarser axis is used).
  - ≤ 2.4 m → Tier 2; the user uploads a DEM (checked: georeferenced, 1 band, ≥ 90% footprint overlap) or has FABDEM fetched.
  - \> 2.4 m → Tier 1; FABDEM is fetched automatically.
  - No georeference → relative preview only (placeholder).
  - Files go to `data/uploads/<id>/` (gitignored, session storage, never cleaned automatically).
- **FABDEM:** Earth Engine `computePixels` at native 30 m, with the projection set before resampling (the 1° bug).
- **Search Online:** CDSE search → `POST /api/input/scene` → FABDEM, always Tier 1 and locked.
  - "Use your own Copernicus API key": headers `X-CDSE-Client-Id/-Secret`; memory only; a bad key → 401.
- **Quota:** `GET /api/cdse/quota`. It shows the documented General-User limits (300 req/min, 10,000 req/month, 300 PU/min,
  10,000 PU/month, monthly reset), the account typology from the token, and this server's own spend (from
  `x-processingunits-spent`). It deliberately shows no "remaining" figure, because CDSE doesn't publish one; it throttles
  with HTTP 429, which is passed through with Retry-After.
- **Placeholder:** the generation stages after START still show the Darjeeling reference outputs (labelled in the UI and log).

**Multi-job sessions — Prompt 3 of prompts.pages (2026-09-25; `frontend/src/jobs.js`, `frontend/src/desktop-close.js`).**
- **Jobs:** each START GENERATION is a job, held in memory only (no server storage). "Save" downloads it as JSON
  (input, GSD, routing, DEM stats, calculation log), and that is what marks it saved.
- **Generate New:** in the jobs panel. It reopens the input view and keeps every job. It is disabled while a job is generating,
  because the staged boxes are shared, so one job generates at a time; switching jobs waits too.
- **Panel order (user to confirm):** newest at the top. The expanded 3D view's tab strip runs oldest → newest, left to right,
  like browser tabs, with a job info line that says the 3D terrain is the Darjeeling reference placeholder.
- **Re-running for a new job:** every box resets to "Awaiting generation" and the stages animate again. Mini/final canvases
  are hidden until their stage ends, and each canvas has one WebGL renderer (no re-creation).
- **Unsaved-work warning:**
  - An in-app modal (`#unsaved-modal`) on Generate New and ✕ Close: job count, a per-job save checkbox, and Save selected / continue-or-discard / Cancel.
  - The old close confirm and its "don't show again" option were removed, as Prompt 3 requires.
  - Web tab close: the native `beforeunload` prompt only. It is suppressed after an in-app confirm (`unloadAllowed`).
  - Tauri v2: `onCloseRequested` → the same modal ("quit" wording). There is no `src-tauri/` in this repo, so it is verified
    only against a simulated `window.__TAURI__` in headless Chrome, not in a real Tauri build.
- **Not per-job:** measurements, notes and the session Library in the 3D view belong to the one shared viewer, not to a job.

**Jobs toolbar + pop-up follow-ups (2026-09-25).**
- **Pop-up right rail:** Visualization Layer, RUN RECONSTRUCTION and FLYTHROUGH are removed; only Scenario Analysis stays. The icon
  toolbar's View ▾ and Flythrough cover them, and `createFlythroughController` no longer needs a button.
- **Grow/shrink animation:** now linear, 470 ms. It holds the speed the old ease-out started at (cubic-bezier(0.2, 0.8, 0.2, 1) over
  1880 ms opens at 4× its average speed). The slow tail is gone; measured ~530 ms headless, with the first and last thirds equally fast.
- **Show/hide icon** (panel-left glyph) next to JOBS: a custom "Hide toolbar" / "Show toolbar" tooltip on hover. The panel collapses to a 44 px strip.
  - It pushes content rather than overlaying it: `--jobs-w` sets the grid/input padding and the expanded view's `left`.
  - The choice is remembered in localStorage.
- **Per job:** Save · Pin · Trash. Trash is two-step ("Delete?" within 3 s) and disabled while generating; deleting the active job switches to the newest remaining job,
  and deleting the last one returns to a fresh input page.
- **Panel order:** Generate New → Saved → PINNED (empty text "No pinned tabs currently") → RECENT. Pinned jobs move to PINNED, and the tab strip marks them.
- **Saved window** (a centred, 60% × 60% window): every save also goes to localStorage (`dw2.savedJobs.v1`: input, routing, DEM, log).
  - The panel stays visible after a reload if saved work exists.
  - Open restores the job as a finished job (and creates the output views if nothing was generated yet). Download re-exports the JSON; Remove is two-step.
  - Per browser only, still no server storage.
- **Tests:** 31/31 unit; 29/29 toolbar headless checks; the Prompt 3 driver still passes 30/30.

**Expanded-view chrome (2026-09-25; `frontend/src/expanded-chrome.js`):**
- **Icon toolbar** (outline icons; names as tooltips only), in `reference.jpg` order: Screenshot, Record (disabled placeholder),
  Measure ▾, View ▾, Flythrough, Save + Library ▾, Mouse pointer, Notes, Trash (clears the current selection).
- **Library:** Points / Lines / Areas / Screenshots / Recordings. Per-item show/hide and delete; per-category and all-items delete,
  both with confirmation. **Session-only (in memory).**
- **Screenshots** are real (canvas capture with the measurement/pins composited, a preview, and a PNG download). **Recording is a placeholder.**
- **Terrain right-click in pointer mode:**
  - Save point (name + tag), Add / Show / Delete note (pins open a square popup).
  - "Delete from selection" or "Start selection here", depending on whether a selection exists.
- **Play/Pause:** a user pause of the idle auto-rotation, via `controls.setAutoRotatePaused`.
- **Theme toggle:** sun while dark, moon while bright. It reuses `applyWorkbenchTheme` (the Workbench bright palette).
- **Scenario analysis:** Flood (existing lowest-30%-elevation shading, now labelled illustrative) plus **Earthquake: an explicit
  placeholder** (a red gradient on the steepest slopes, `terrain.setEarthquakeOverlay`, not a seismic model).
  - **There is no Landslide scenario anywhere in the code** (it's only in `reference.jpg`), so none was built.
- 62/62 real-input headless checks; the measurement suite re-run after the refactor gives 54/54.

**3D measurement tool (2026-09-25; deliverables-audit item 6 → partial):** it lives in the expanded final-demo view.
- **Toolbar:** "Measure A to B ▾" (two points / continuous), "Save ▾" and "Mouse pointer" pills, following `reference.jpg`.
  The other pills and the side panels are the next step.
- **Modules:**
  - `heightfield.js`: an exact ray→terrain hit (DDA over the grid plus Möller–Trumbore). It agrees with `THREE.Raycaster`
    on 3,000/3,000 rays and is ~300–5,000× faster.
  - `measure-metrics.js`: metrics from the raw DEM.
  - `measure-model.js`: modes, click rules, multi-level undo, and a session-only save (a placeholder: no persistence).
  - `measure-tool.js`: the SVG overlay, input handling and menus.
- `controls.setAutoRotateHold()` keeps the idle auto-rotate off while measuring.
- **Tests:** `cd frontend && npm test` (20 unit tests); 55/55 real-input headless checks were run.
**Workbench final-demo expanded view (2026-09-24, user-directed):**
- When the generation pipeline finishes, box 8 (the final 3D demo) pops out to fill the window. It is an in-page
  `position: fixed` overlay, not the native Fullscreen API, because the pop-up isn't triggered by a user gesture.
- Upper-left bar:
  - **← Back** returns it to its grid cell with the state intact. ⛶ re-opens it.
  - **✕ Close** asks "Close without saving?" with Cancel / Close and a "Don't show this message again" checkbox. The checkbox
    is saved only when you confirm (localStorage key `dw2.skipCloseConfirm`, try/catch-guarded).
  - Confirming resets every Workbench field and reloads to a fresh, empty Workbench.
- 28/28 headless checks pass, with 0 JS exceptions.
- **Grow/shrink animation (2026-09-24).** The box animates its fixed-position edges from the grid cell to the full window
  (1.88 s since 2026-09-25, was 380 ms; ease-out; the reverse on Back) using the Web Animations API.
  - A hidden placeholder holds its grid cell.
  - The rails and the Back/Close bar fade in after the box reaches full size.
  - Reduced-motion users get the instant switch.
  - Measured on GPU headless Chrome: ~60 fps (frames 16–17 ms, max 30 ms) across 9 animations; 33/33 checks pass.

- **Labels (2026-09-23, text only):** the header reads "Satellite → 3D Terrain Visualization"
  (was "… Metric Terrain Reconstruction"). The "METRIC ELEVATION" layer/panel labels now read
  "DEM ELEVATION", and the captions read "DEM elevation from <source>". Verified in the
  dev-server-served files and a rebuilt `dist/`; screenshot
  `docs/screenshots/2026-09-23_demo_labels_dem.png`.
  - **Flagged, not changed:** the "DSM" button shows *DAv2 relative depth draped on the real
    DSM relief*. The label is ambiguous but doesn't overclaim.
  - `package.json`'s `build` script runs `vite` (the dev server), not `vite build`.
    `frontend/dist/` is gitignored and was rebuilt manually.
- **Stack**: single-page Three.js + Vite app (`frontend/src/main.js`, `viewer.js`), dev
  server at `localhost:5173`. FastAPI backend (`backend/main.py`) with a CDSE router
  mounted at `/api/cdse` (`backend/api/routes.py`, `backend/cdse/client.py`).
- **Not a multi-page app** — there is no separate "Explore" or "Workbench" page. It's
  one viewer with:
  - A region rail — all 4 regions (Darjeeling, Kolkata, Bardhaman, Sundarbans) are
    interactive, each with satellite/relative-depth/elevation layer switching
    (`activateLayer`/`selectLayer` in `main.js`) and a real per-region stats strip.
  - A mock "RUN RECONSTRUCTION" pipeline (`runReconstruction()`) with staged progress
    and pipeline-step highlighting, labeled inline as a prototype flow.
  - A scene search/upload modal (`openSceneSearchModal`/`openSceneUploadModal`) with
    real Nominatim geocoding (`queryNominatim`) and CDSE scene search/selection
    (`runSceneSearch`, `selectScene`) hitting the backend `/api/cdse` routes — this is
    the "CDSE Scene Input integration."
  - OrbitControls flythrough (damped drag-orbit, idle auto-rotate) and a flood-overlay
    toggle (`setFloodActive`).
- **Assets**: `frontend/public/data/{darjeeling,kolkata,bardhaman,sundarbans}/` all
  populated (satellite/relative-depth/elevation textures + terrain.json per region).
- **Working state**: last verified interactive for all 4 regions per CLAUDE.md Session 2
  outcomes — confirm with a fresh `npm run dev` + visual check before relying on this,
  don't assume it's still green without checking.

### 5z. 2026-09-28 UI/UX sessions (17 prompts): DEPLOYED 2026-09-29 (web + desktop v1.0.2)

Two prompt series ran from `prompts.pages` (series A: 9 prompts, commits `ee6f95c`–`1bfccd1`; series B: 8 prompts,
`cae681d`–`8f7c709`). Every commit message has the detail and the verification; this is the map.

**Frontend (`frontend/`)**
- Jobs: any job can be opened while another generates. The grid paints only the on-screen job (`job.run` stage state,
  `showRunningJob`, `GEN_STAGES` in `src/main.js`). Each job keeps its own viewer edits (`captureViewerMemory` /
  `applyViewerMemory` / `rememberActiveJobView`). Failed generations are marked **Failed** with **↻ RETRY GENERATION**
  (`syncStartButton`, `retryJob`).
- Processing page: box readouts follow the real work, with a 1.3 s minimum each (`src/progress-sync.js`, `trackWork`; was 2 s until 2026-09-29). The
  calculation log is queued and paced, and finishes before the DEM Elevation box (`runCalcLog`, `queueCalcLogLine`,
  `flushCalcLog`). Captions: depth box is 3 lines; no tier or DEM names anywhere in the processing page
  (`generationLogLines`).
- Studio: flat-terrain warning for flat Sentinel-2 tiles (`src/flat-warning.js`). Outliers stop growing with vertical
  exaggeration (`src/outlier-relief.js`, used in `src/terrain.js`).
- Input page: cold-start / loading messages (`LOADING_COPY` in `src/input-view.js`). Floating, pointer-reactive
  background icons: 12. They cruise at a steady speed and never halt (`createCruiser` / `steerCruiser`). They bounce
  off each other and their limits (`reflect`, `bounceImpulse`, contacts in `separate`) and repel so none overlaps. The
  big satellite is locked to the top-left quadrant and the earth + satellite to the bottom-right (`clampToZone`).
  Same-artwork icons keep 5 cm apart (`separate` in `src/bg-float.js`, `src/bg-icons.js`,
  symbols in `index.html`, from `icons/2.svg`). None is ever more than 30% hidden behind the boxes or off the page. The
  left limit is a fixed line at the collapsed sidebar's width, 56 px. Very small icons are 1.4x on the left
  (`hiddenFraction`, `constrainVisible`). They keep 1 cm apart. Their homes are spread evenly on all four sides
  (`layoutHomes`). Very small star dots sit behind them (`starField`, canvas). The green is more vibrant
  (`--bg-green-rgb`). Floating speed: `WANDER_SPEED` 0.85.
- Library filter: the terrain filter is a dropdown on the right under the collection chips (`filterLibrary`,
  `noMatchMessage`, `applyFilter` in `src/input-view.js`). A choice with no tiles keeps the last results and shows a
  notice that closes itself after 4 s.
- Sidebar: rebuilt after shadcn `sidebar-07` (`index.html` `#app-sidebar`, `src/sidebar.js`, CSS "GLOBAL SIDEBAR").
  - D mark `public/brand-d.png` (cropped from `icon.png`) opens Home.
  - Page icons are option 1 of `icons/1.svg`.
  - The rail drags between 208 and 360 px; below 160 px it collapses.
  - Footer links: GitHub, email, LinkedIn; X is a placeholder.
- Page names: **Workbench → DW Studio, Docs → Home**, display text only. Page ids and the permanent routes in
  `src/routes.js` are unchanged (`#/docs`, `#/demo`, `#demo-video`, bare URL all verified).
- Tests: `npm test` in `frontend/` → 91/91.
- Input boxes: a thin green outline glow (2x brighter; the travelling patch was removed). Stars: half the density, big
  ones 6%.
- Bright mode: every green is the site orange (`--neon-rgb` 196, 120, 58; background icons, stars and outlines 223,
  118, 32). The wireframe recolours live. The text base is crisper (40, 24, 10). Only Tier 2's green stays (status
  colour).
- Upload without a geotransform: a card asks for the GSD. It's set with `POST /api/input/{id}/gsd`
  (`store.set_manual_gsd`), and the flat plane is sized from it in `pipeline.generate`.

**Backend (`backend/`)** (must reach the Render copy and the desktop sidecar)
- `backend/api/depth_routes.py`: transient Space errors (502/503/504, ClientDisconnect, resets) are retried twice
  (2 s, 5 s) with a fresh gradio Client (`SPACE_RETRY_DELAYS_S`, `_is_transient`).
- `backend/dem/glo30.py`: `mosaic_to_grid` merges GLO-30 tiles before a single reproject. The old per-tile reproject
  left NaNs on tile seams: Darjeeling's south edge is on 27°N, 6 px.
- `backend/terrain/mesh_export.py`: `fill_nan_nearest`. Holes are filled from neighbours, never the tile minimum. That
  minimum fill drew the 554 m "downward spikes".
- Tests: `uv run python -m pytest -q backend/tests` → 48 passed, 1 skipped.

**Deployed 2026-09-29** (full record: `docs/CHANGES_2026-09-28_29.md` "Deployed"; live checks: `docs/DEPLOY.md` LIVE)
- **Git:** the unpushed commits were rewritten before the first push. `Claude-Session:` lines were stripped, and
  `CLAUDE.md` / `last_session.md` were kept out, since the owner had deleted them on GitHub. The commits were then
  pushed to `origin` (`sancharimouri/depthwizard-studio`, PUBLIC, this whole repo) as fast-forwards, `376c22d..36789c0`
  plus the later docs commits. There's a single identity (Sanchari Mouri) and no Co-Authored-By.
- **Render** builds that repo's `main` with `pip install -r requirements.txt`. That root file didn't exist, so every build
  since 2026-09-27 had failed; it was added in `abbb7b6`. `/api/library` 89, `/gsd` listed, Darjeeling without pits (max
  39.7 m below its 5×5 median), a DFC2019 generation from its private pack, a PNG at 0.5 m → 200 × 150 m, CORS exact.
- **Space:** restarted, then `ssr_mode=False` (Space `f9c671a`). Its README must keep `python_version: '3.12'`: a `3.11`
  push built on 3.10 and failed (ZeroGPU supports 3.12.12 / 3.10.13 only).
- **HF edge 502s** (8–23% of `*.hf.space` requests on 2026-09-29, unrelated Spaces too):
  - `86787df` re-sends edge 502s.
  - `cedb449` retries a Space job whose event stream broke (it used to be a bare 500).
  - After both: 15/15 live generations.
- **Frontend** (Vercel `depthwizard2-62bvvq0xf`, both domains): the whole §5z UI, plus
  - `ceed1ac`, a favicon (the only console error);
  - `36789c0`, the elevation-3D box readout, which paints before its mesh build (it was 1.15–1.24 s; now 1.32–1.39 s).

  The four deck links load with 0 console errors.
- **Desktop v1.0.2** (`c6afa57`, released on `sancharimouri/depthwizard2-desktop`):
  - the backend is re-frozen with every fix, plus scipy (for `fill_nan_nearest`);
  - DMG 355.7 MB, update archive 343.2 MB, installed 492.2 MB;
  - the signature was verified against the built-in key;
  - a published v1.0.1 updated itself to 1.0.2 and relaunched.
- **Still open:**
  - `#demo-video` is a placeholder until the recording exists.
  - Home says "88 curated scenes (32 Sentinel-2)"; the library has 89 (33).
  - The DFC2019 terrain packs are web-only; the desktop app still shows DFC2019 flat.

### 5z+1. Curated tile library v2 (2026-09-29): LOCAL ONLY, branch `curated-tiles-2026-09-29`, not pushed or deployed

Full record: `docs/library_v2.md`. It adds:
- `data/library_v2_2026-09-29/` with `tile_manifest.json`, the single source of truth;
- `scripts/build_library_v2.py`, `scripts/build_maxar_display_packs.py` and `scripts/stage_container_tiles.py`;
- the backend and frontend reading the manifest (`DW2_TILE_MANIFEST`) and a Maxar DISPLAY band.

The protected folders are SHA-256-identical before and after (474 files).

**2026-09-30 update:**
- Re-ratings applied: Bhitarkanika 30x, Kutch 90x, Amalapuram 43x; Maxar a_valley 10x, c_town 4x, c_river 11x; all
  3 stars; b_glacier excluded.
- The DFC2019 limiter bug is fixed (DFC2019 terrain.json has `limitOutliers: false`).
- Darjeeling's A and B are identical, so there is no decision.
- DFC2019: OMA_212_033 (16x) and OMA_258_020 (22x) are kept; OMA_211_039, OMA_211_032, OMA_376_023 and
  OMA_376_038 are excluded (files stay). All included DFC2019 tiles have 3 stars. The container holds 76 tiles.
- Items 1, 2 and 4 below are closed. The DFC2019 anchors may need re-checking, since they now render at full relief.

**Part 4B (2026-09-30):**
- DAv2 depth is baked into the library_v2 packs, so opening a library tile makes 0 Space calls (bake: 76 calls).
- Uploads and CDSE scenes still call the Space.
- Verified (a)–(d); see `docs/library_v2.md`.
- Open: the unchanged pipeline log line still prints the bake-time inference seconds.

**Decisions waiting on the owner (as of 2026-09-29):**
1. **Re-rate** Bhitarkanika, Kutch and Amalapuram (offset-only recalculation, relief 2.4–5.2 → 7.0–12.3 m;
   Bhitarkanika shows a sharp step between mangrove and farmland at 60x).
2. **Darjeeling colours:** original (active) vs the generated B_own_percentiles render.
3. **Maxar preset:** subtle (active), medium or strong, or new f / k / W / T values.
4. **DFC2019 flat tiles:** OMA_211_039, OMA_211_032, OMA_212_033, OMA_376_023, OMA_376_038 and OMA_258_020 render
   flat because of `frontend/src/outlier-relief.js` (IQR collapses when at least 75 % of cells are ground), not their
   data. Fix the limiter, or exclude them. OMA_144_030 is excluded (genuinely flat).
5. **DFC2019 mapping:** p98−p2 orders the anchors except OMA_269_035 (Spearman −0.79). Accept it, or give more anchors.

**Open:**
- No Dockerfile yet (staging is a dry run).
- HF `dem/` and the public release still hold the previous sets.
- The pre-existing uncommitted pack-rebuild edits (`desktop/tiles/build_dem_pack.py`, `scripts/dfc2019_build_packs.py`)
  are still uncommitted.
- `docs/method-audit/08-dfc2019-terrain-packs/packs.md` is only partly updated: a dated note at the top.

### 5z+2. Slim backend container + static web library (2026-09-30): LOCAL ONLY, branch `slim-container-2026-09-30`

Records: `docs/container-measurements.md` (BEFORE / AFTER, dated per part) and `docs/deploy-cloud-run.md` (runbook).

- **§2 build state:**
  - The image is 116.7 MB compressed on amd64 (was 212.3); idle RSS is 80 / 61 MiB (was 117 / 91); all 3 flows'
    outputs are byte-identical.
  - scipy and pyproj are gone; `/health` added.
  - The web library is static on Vercel (52.4 MB), with 0 backend / Space calls to open a tile.
  - The desktop app is unchanged.
  - A CDSE 8 × 8 mesh bug is fixed (5647668).
- **§5 Docker / Cloud Run:**
  - Memory: 512 MiB is feasible (worst peak 287 MiB), with concurrency 4, CPU boost, min-instances 0, and 1 only on
    judging days.
  - Measure the real cold start after the first deploy: keep scale-to-zero if the median is ≤ 5 s.
  - Method 6 would need about 1 GiB and about +120 MB of image.
- **Owner decisions logged 2026-09-30:**
  - compact static format;
  - DFC2019 served publicly at display resolution (**standing rule changed:** "DFC2019 raw data (imagery, AGL, per-pixel research outputs) must never be in the public git repo. Display-resolution tiles derived from it may be served publicly by the web app, with credit to IEEE GRSS DFC2019 and JHU/APL US3D.");
  - keep the `.pyc` files;
  - fix the CDSE bug separately;
  - report-only on the DFC2019 paths already in git.
- **§6 open items / decisions waiting on the owner:**
  - ~~Choose the Maxar DISPLAY preset~~ Done 2026-09-30: subtle (§5z+3).
  - The cleanup of the 3,691 `dfc2019` paths in public git history: all on origin/main, including 47,500 anchor pixel
    samples. It waits for the backup repo.
  - Measure the real Cloud Run cold start after the first deploy.
  - A `/tmp` cap on long-lived instances.
  - `/api/facts` is still called for georeferenced tiles (the Facts panel; not tile data).
  - The Vercel limits page still lists 100 MB for CLI uploads: re-check it before the first deploy.

### 5z+3. Facts research + Maxar preset (2026-09-30): LOCAL ONLY, branch `facts-research-2026-09-30`

Branched from `slim-container-2026-09-30`. Full record: `docs/facts-research.md` (Parts A–F, dated). No deploy, no
push, no GCP, no API sign-ups.

- **Done:**
  - **Maxar preset is SUBTLE.** The static library was re-baked (52.5 MB), and the placeholder flag is cleared. The
    manifest already said `subtle`, so it was not edited. The Vercel upload is 73.0 MB, leaving **27.0 MB of
    headroom**.
  - **Facts diagnosed.** It gives postal-address and point-elevation content, not geography, and a 250 km radius.
    Landslides and volcanoes were permanently "unavailable". On Render: a 64 s cold call and Open-Meteo 429.
  - **Sources researched**, with live calls at 3 points, and costed designs (C3).
  - **Earthquake overlay recommendation:** a hazard card plus USGS epicentres, not a mesh tint. Mock-up in
    `docs/screenshots/`.
  - **347 draft facts** for 76 tiles in the gitignored `data/library_v2_2026-09-30/facts_drafts.{json,md}`.
  - **The curated-facts slot**, built and verified with 0 backend calls.
  - Protected data: SHA-256 identical, 1,864 / 1,864 files.
- **Changed behaviour (web build only):** while nothing is curated, **library tiles show no Facts panel.** CDSE scenes
  and uploads still call `/api/facts`. Desktop is unchanged.
- **Decisions waiting on the owner:**
  1. **Curate facts.** In `facts_drafts.json`, set `status: "curated"` (optionally edit `text`), then run
     `scripts/bake_static_library.py`. Review `facts_drafts.md`.
  2. **Pick a C3 option** for CDSE scenes and uploads. The recommendation is option 3 plus the bundled ThinkHazard
     grid (+~9 MB image, $0). Drop Nominatim and Open-Meteo.
  3. **Licences before any commercial use:**
     - FABDEM is CC BY-NC-SA, and the DEM-derived facts inherit it;
     - GEM is NC;
     - the World Bank landslide licence is unverified;
     - GDACS has no explicit licence.
  4. **Keys not obtained, per the rules:** a NASA FIRMS MAP_KEY and a GeoNames username (both free). Needed only if
     wildfire or GeoNames facts are wanted.
  5. **Earthquake placeholder:** replace it per Part D, or remove it.
  6. `vhr-a_valley` opens at ×10 and looks spiky with subtle: consider a lower `default_exaggeration`.
  7. Still deferred (unchanged): the `/tmp` cap, the DFC2019 git-history cleanup, GCP, and the data/ backup.

### 5z+8. Method 6 production model (2026-10-01): fair GAMUS re-test PASSED; uploaded (private HF)

- **Owner decision:** the §5z+7 agreement rule was flawed (it measured memorisation). It is superseded by a
  pre-registered GAMUS accuracy test; the original FAIL stays on record.
- **Fair re-test:** 2,848 GAMUS test tiles vs LiDAR, the Part B protocol, with exact parity to Part B's saved m6_s42.
  - Full model **MAE 3.156 / RMSE 4.692 / r 0.624** vs F (mean of the seed-42 folds) **3.175 / 4.660 / 0.626**.
  - **PASS** pooled and in 3 of 3 cities.
- **Production Method 6 = `method6_full_dfc2019_hb_seed42.pt`** (SHA-256 `69a29e10…`).
  - On the private HF repo at `full_dfc2019/`; the old model is archived at
    `archive/method6_full_dfc2019_pre_height_balanced.pt` (HF revision `d7bea1c`).
  - Code references are updated; tests and the desktop smoke test pass.
- **ONNX** (scratch, `build/method6_onnx/`, not shipped): parity max |diff| 0.00084 m, r 0.99999999999.
- **Model-service readiness (measured, amd64, 1 vCPU, Rosetta timings indicative):**

  | | Idle | Peak (ORT arena off) | Peak (arena on) |
  |---|---|---|---|
  | Method 6 alone | 214 MiB | 366 MiB | 459 MiB |
  | DAv2 + Method 6 | 320 MiB | 467 MiB | 668 MiB |

  Method 6 takes 3.2 s per pass, 12.5–12.8 s per 1024² image (4 quadrants). **Plan a 1 GiB model service** (512
  MiB fits only with the arena off and ~45 MiB spare). Image ≈ 303 MB unpacked. **Not built or deployed:** it is
  future work.
- **Open items:**
  - **the model service itself** (design, build, deploy): owner decision;
  - an ONNX export into a shipped location: owner decision;
  - **UI wording** (`index.html:1246-1247, 1251`): during the redesign;
  - the Maxar packs: no rebuild needed (they use the seed-43 fold ensemble).
  - Still open from §5z+5/6: the GCP setup, the cold start, the cutover, and the desktop sidecar rebuild.

### 5z+7. Method 6 full-model retrain (2026-10-01): FAILED the pre-registered agreement rule; nothing uploaded

- **Retrain:** `--train-on-all` was added to the adopted script (only the data split changes; tested). Seed 42
  trained in 1,070 s → `data/dfc2019/experiments/method6_full_checkpoint/method6_full_dfc2019_hb_seed42.pt`, SHA-256
  `69a29e10…`.
- **Audit:** every recipe parameter matches, exact load, sane output.
- **Agreement vs the seed-42 folds on their held-out quadrants: median r 0.861 (rule ≥ 0.95), mean |diff| 1.40 m
  (rule ≤ 0.9 m) → FAIL → STOPPED.** No ONNX, no HF changes, no renames.
- **Diagnostic** (in-sample, not an accuracy estimate): the new model memorises its training quadrants harder (MAE
  1.12–1.31 m vs the old recipe's 1.42–1.60 m), so on quadrants it trained on it moves away from the folds' held-out
  predictions. The rule measured memorisation as much as fidelity.
- **Model-service readiness:** **no single-file production model yet.**
  - Production-valid today: the seed-42 or seed-43 4-fold ensembles.
  - The model-service estimate is unchanged (DAv2-Small 214 / 354 MiB measured; + one Method 6 model ≈ 500 MiB, plan
    1 GiB; + a 4-fold ensemble ≈ 800 MiB, plan 2 GiB).
- **HF repo:** PRIVATE, unchanged (still the old full model + the seed-43 folds).
- **Maxar packs:** built from the **seed-43 fold ensemble** (production-valid), not the old full model.
- **Wording updated:** "validated on DFC2019 (US cities, satellite imagery); did not generalize to GAMUS by RMSE"
  (final-comparison, HANDOFF §2a, the VHR doc, the audit).
- **UI overclaims listed** (`index.html:1246-1247, 1251`), not changed; the UI redesign is coming.
- **Decisions waiting on the owner:**
  1. a fair agreement test on unseen data (GAMUS / VHR), **or** ship an ensemble, **or** accept on the recipe audit
     alone;
  2. the UI wording, during the redesign;
  3. whether the Maxar packs should ever be rebuilt (not needed: already on production-valid models).

### 5z+6. Session 2026-10-01: standing links rule, private library off, Method 6 checkpoint audit (branch `release-candidate-2026-09-30`, local only)

- **The 5 protected links** are a standing rule (§0), enforced by `scripts/check_protected_links.sh`. Baseline: ALL
  PASS.
- **Cloud Run image:** `DW2_PRIVATE_LIBRARY=off`. The private HF dataset is never read, `/api/library*` answers 404,
  and `docker diff` confirms no HF cache is ever written. The desktop keeps the private library.
- **Method 6 checkpoint audit:** `docs/method6-checkpoint-audit.md`.
  - All 17 local checkpoints load strictly and give sane held-out output. All 5 private-HF copies are byte-identical
    to the local files.

**Model-service readiness (Method 6):**
- **Production-valid:** the adopted-recipe **4-fold ensembles** for seeds 42, 43 and 44 (12 checkpoints).
  - Expected accuracy = the 3-seed CV result, 1.990 / 3.504 / 0.743 / 0.656.
  - Seed 42 is the headline, reproduced bit-for-bit; seed 43 is the one on HF.
- **Valid for evaluation only:** the GAMUS-DC folds (not adopted; the tall-tree criterion failed).
- **Mismatch:** `method6_full_dfc2019.pt`, the full model (also on HF). It uses the pre-adoption recipe: no
  height-balanced loss or sampler, warmup 40.
  - Its agreement with the adopted fold models (Pearson 0.88–0.93, 1.1–1.3 m) is about twice as loose as seed-to-seed
    agreement (0.98, 0.5–0.7 m).
- **ONNX export:** not done (Part 3 skipped: the full model isn't production-valid).
- **Memory:** DAv2-Small ONNX measured at 214 MiB idle / 354 MiB peak. DAv2 + one Method 6 model ≈ 500 MiB (plan
  1 GiB); + a 4-fold ensemble resident ≈ 800 MiB (plan 2 GiB, or load folds sequentially).

**Open items:**
- **Owner decision:** retrain an adopted-recipe full model (a `--train-all` option, fixed 12 epochs, ≈ 20–25 min per
  seed on MPS; the plan is in the audit §3), **or** ship a 4-fold ensemble.
- **The HF repo's full model is the mismatched one.** If the retrain goes ahead, its upload would replace or sit
  beside it; owner decision, not done.
- **Owner decision:** the HF repo lacks the seed-42 headline folds. Upload them only if the seed-42 ensemble is the
  one to ship.
- Still open from §5z+5: the real Cloud Run cold start, the GCP setup, the cutover, and the desktop sidecar rebuild.

### 5z+5. Release candidate (2026-09-30): LOCAL ONLY, branch `release-candidate-2026-09-30`; not merged into main, not pushed

Full record: `docs/release-candidate-2026-09-30.md` (Parts 1–8). Runbook: `docs/deploy-cloud-run.md`, rewritten.

**Build state:**
- The RC = `facts-v2-2026-09-30`, which already contains curated-tiles, slim-container and facts-research, **plus a
  merge of `pack-rebuild-code`**. There was no textual conflict. `build_dem_pack.py`'s calibrated OLS recipe and
  `build_library_v2.py`'s offset-only override (3 tiles) are consistent with the library.
- **Tests:** backend 84 passed, 1 skipped; frontend 104 passed. Desktop smoke test PASS.
- **Parity:** PNG / GeoTIFF / CDSE / CDSE+facts outputs are **identical** to slim-container
  (`scripts/flow_parity.py`).
- **amd64 image:** **119.1 MB compressed / 335.9 MB unpacked**; idle RSS 79.5 MiB; per-flow peaks 156–251 MiB;
  **3 flows at once (with live facts): 287.6 MiB RSS / 246.5 MiB cgroup: fits 512 MiB with ~225 MiB margin.**
- **New:**
  - the /tmp cap (30 min + 64 MiB, never in-flight or < 2 min-old; image-only);
  - logs to stdout;
  - graceful shutdown 8 s;
  - the CORS `*` guard;
  - the EE init lock;
  - no `.env` fallback when `DW2_NO_DOTENV=1`.

**Deploy readiness checklist (backend only):**
- [x] Image builds for linux/amd64, non-root, `0.0.0.0:$PORT`, `/health`, stdout logs, clean SIGTERM (0.41 s, exit 0).
- [x] Memory fits 512Mi (measured); the timeout for the longest flow is 300 s; concurrency 4 is thread-safe.
- [x] /tmp is capped; the CORS allowlist is exact; secrets are env-only (Secret Manager); no files outside the image.
- [x] Earth Engine works with the runtime service account (ADC); no code change needed.
- [x] The runbook is written, with budget, APIs, EE, secrets, registry cleanup, deploy, smoke tests, cold start,
      min-instances and rollback.
- [ ] **Owner:** GCP project + billing, EE registration decision, the secret values, and running the runbook.
- [ ] **Owner:** measure the real cold start (runbook §11) and decide min-instances for judging days.
- [ ] **Owner:** the Vercel `VITE_API_BASE` cutover after verification; Render stays up until then.

**Open items:**
- **Real Cloud Run cold start:** not measurable locally (0.75 s start-to-health in Docker).
- **4-way concurrency is estimated, not measured:** ≈ 360–380 MiB, ≤ ≈ 450 MiB with a full /tmp.
- ~~`HF_HOME=/tmp/dw2/hf` is not capped~~ **Resolved 2026-10-01:** `DW2_PRIVATE_LIBRARY=off` in the image, so the HF
  cache is never written (§5z+6).
- **The CDSE flow ranged 29–105 s** across runs, from upstream latency.
- **The desktop sidecar binary** still needs a rebuild to get Facts v2 (unchanged from §5z+4).
- **Model service (future):** DAv2-Small ONNX measured at 214 MiB idle and 354 MiB peak (arena off). DAv2 + Method 6
  is estimated at ≈ 500 MiB peak: plan 1 GiB. No Method 6 ONNX export exists yet.
- **Still deferred:** the DFC2019 git-history cleanup and the data/ backup.

**Decisions waiting on the owner:**
1. **Earth Engine project:** reuse the existing `EARTHENGINE_PROJECT` and deploy Cloud Run into it (recommended), or
   use a new project. If new, either register it for EE or grant the service account the roles on the existing one.
2. **Region** (`asia-south1` is proposed) and **max-instances** (3 is proposed, as a cost cap).
3. **When to cut over** `VITE_API_BASE` and **when to decommission Render**.
4. **Merging the RC into `main` and pushing:** not done.

### 5z+4. Facts v2 (2026-09-30): LOCAL ONLY, branch `facts-v2-2026-09-30`

Branched from `facts-research-2026-09-30`. Full record: `docs/facts-research.md` (sections "Facts v2", dated per
part, and "v2 final design"). No deploy, no push, no GCP, no API sign-ups.

- **Done:**
  - **Facts box:** a crisp one-line design, only non-scenario hazards plus named peak / river / glacier, a link to
    Scenario Analysis, and an empty state.
  - **Scenario Analysis:** Flood / Landslide / Earthquake cards with real data. **The fake earthquake slope tint is
    deleted**, replaced by the district level, USGS M4.5+ within 100 km, and an epicentre inset map.
  - **Library drafts** re-split into `facts` / `scenario`: 76 tiles, 732 items, **all draft**. ThinkHazard misses went
    from 11/31 to **0/31** (point-in-polygon).
  - The bake bakes only curated items for both groups, with 0 backend calls for library tiles.
  - **Live route** `/api/facts?bbox=|lat&lon`:
    - a bundled ThinkHazard district grid (+4.3 MB; 2.4 MB compressed);
    - Wikidata, JRC/WB/GSW COG windows and USGS live;
    - a 9 s deadline, a bounded cache, and a peak of 137 MiB with 4 concurrent lookups;
    - Nominatim, Open-Meteo and GDACS removed.
  - The Docs page "Data sources & credits" section.
  - vhr-a_valley ×3 / ×5 / ×7 / ×10 screenshots.
  - The desktop smoke test PASS.
  - Protected data: SHA-256 identical (1,864 / 1,864).
- **Layout pass (owner feedback):** no divider lines, one-line footer, and values wider than half the card are left-aligned under their labels. Screenshots are in `docs/screenshots/facts-v2/`.
- **Decisions waiting on the owner:**
  1. ~~Curate~~ Done 2026-09-30: the owner approved the design and **all 732 items are curated**. The static
     library is re-baked (732 items on 76 tiles). The all-draft file is kept as `facts_drafts_all_draft_backup.json`.
     To un-publish an item, set it back to `draft` and re-run `scripts/bake_static_library.py`.
  2. ~~vhr-a_valley default exaggeration~~ Done 2026-09-30: **×6** (owner's pick; `_input_ratings_2026-09-30c.json`,
     applied to `tile_manifest.json`).
  3. **Rebuild the desktop sidecar** (`desktop/freeze_trial/build_freeze.py`) when the desktop app should get the new
     Facts route. The prebuilt sidecar degrades to "Couldn't look this place up".
  4. **Deploy** (backend + web) when ready; nothing is pushed. `requirements.txt` needs no change.
- **Open items:**
  - The GAUL-derived grid and the World Bank landslide map carry non-commercial or unverified licence terms. No
    action for SIH; they are noted in the docs and the credits.
  - GSW's cold read (~5 s) sits at the GDAL timeout, so it is the first line to drop when cold.
  - Still deferred: the `/tmp` cap, the DFC2019 git-history cleanup, GCP, and the data/ backup.

## 6. Credentials/access inventory (`.env`)

| Key | Status |
|---|---|
| `EARTHENGINE_PROJECT` | Working. |
| `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` | **Sentinel-Hub-scoped only, NOT OData/download-scoped.** This caused a real credentials-audience error earlier — if you hit an audience/scope error against the CDSE OData download endpoint, this is why; the fix is a different credential type, not a retry. |
| `HF_TOKEN` | DINOv3 gated access, granted. |
| `NICFI_API_KEY` | Present, but **account lacks program entitlement** — confirmed via a real HTTP 200 + empty list response, not an auth error. Don't mistake this for a broken key. |
| `OPENTOPOGRAPHY_API_KEY` | 401, unresolved. Copernicus GLO-30 via AWS is the working substitute (see CLAUDE.md's "what actually exists" section for where it's already wired in). |
| `ICESAT_API_KEY` | Present, used for ICESat-2 ATL08 pulls (Sentinel-2 benchmark ground truth). |
| Bhoonidhi | No working API — manual portal fetch only, not automatable. |

## 7. Standing methodology conventions (follow, don't rediscover)

- **4-fold spatial-quadrant holdout** is the standard evaluation protocol for any new
  DFC2019-track method.
- **Always compare against the oracle per-tile-OLS baseline, 3.392/4.579/0.582/0.509**
  (dense OLS of DAv2 on 3 quadrants, scored on the held-out quadrant;
  `data/dfc2019/experiments/semantic/method3_spatial_cv_results.json`). A method that doesn't
  beat it isn't worth adopting. The Method 2 Grid+Huber+20 row (2.929/4.718/0.532/0.471) is a
  *deployable* sparse-GCP baseline on a different pixel set. Report it, but it is not the oracle
  (`final-comparison.md` §1.0).
- **Independent-of-training-reference validation is required.** ICESat-2 for Sentinel-2,
  real held-out LiDAR for DFC2019. A DEM-only or same-source check alone is **not
  trusted** — this project has hit the same memorization-detection pattern three
  separate times on Sentinel-2 CNN attempts (§2b) by trusting a same-source check first.
- **Staged/gated testing with an explicit stop condition** before committing to a full
  run (see Method 6's Sentinel-2 staging: stopped at fold 0 per a pre-agreed protocol
  once it lost decisively, rather than burning budget on folds 1–3).
- **A clean negative result gets the same write-up rigor as a positive one** — never
  silently dropped. Semantic-prior phase 2.3, RDAH-Net zero-shot, and all three CNN
  Sentinel-2 failures are documented this way; keep doing that.
- **Every summary response ends with an explicit list of files created/modified.**
- **Fusion variants must be compared against a DEM-only control** (the same pipeline with the
  added component zeroed, plus the raw DEM(s)). Beating a weaker learned baseline isn't enough:
  the frequency-fusion headline (21/25 vs. linear-calibrated DAv2) turned out to be the DEM's
  own result (2026-09-23).
- **Detail sources are scored against a surface reference, not only ground photons**
  (ICESat-2 20 m canopy-top segments, GEDI rh98). Scoring an above-ground-height product against
  terrain truth is uninformative, as the first RDAH Sentinel-2 test showed.
- **Pre-register each phase's decision rule in the log and commit it before running**, and label
  any post-hoc judgment of an un-operationalized term as such.
- **DEM-only comparisons use all benchmark tiles, not DAv2-filtered subsets.** The 7 sign-flip-
  excluded tiles were removed by a DAv2 criterion that is irrelevant to DEM products (A2 uses 32).
- **R4 offset guard:** a DEM or product "pass" must hold on BOTH RMSE and bias-removed RMSE
  (report bias and median absolute error too). This blocks constant-offset "wins" like CHMv2's
  (Phase 4) and FABDEM + ETH's on GEDI (A3).
- **Set `PROJ_NETWORK=ON` before any pyproj import, and assert |N| > 1 m.** Otherwise the geoid
  silently becomes 0.
- **Commit result files when the result is written up**, not later (see the gaps in
  `00-audit-log.md`).

- **Out-of-domain before adoption (2026-09-23).** A DFC2019 win is not a product claim until it is tested on
  independent data covering the brief's landscapes: urban (GAMUS), sparse, and forested/mountainous (3DEP/NEON LiDAR).
  Method 6's DFC2019 "beats the oracle on all 4" did not survive GAMUS.
- **Check leaf-on/leaf-off before using canopy supervision:** measure the share of tree pixels with ExG > 0.05.
  `scripts/gamus_leafon_precheck.py`.
- **Mean-of-tiles variance ratio is uninformative when some tiles are near-flat** (var(true) ≈ 0 blows it up).
  Report the pixel-pooled variance ratio.
- **Vertical datums on US LiDAR (NAVD88):** convert EGM → WGS84/ITRF2014 ellipsoid → Helmert to NAD83(2011) → GEOID
  explicitly. PROJ's default route uses a null NAD83↔WGS84 step, an error of up to 1.3 m.
  Guard with |N| > 1 m, **not** with the size of the final shift: that can legitimately be ≈ 0, e.g. 0.045 m at MLBS VA.
- **Kaggle GPU offload:** follow the reference pattern in CLAUDE.md (`scripts/gamus_zeroshot_kaggle.py`, `data/kaggle_bundles/gamus_zeroshot/read.md`).

## 8. Skills

`superpowers` (verification-before-completion, systematic-debugging, etc.) and
`claude-scientific-skills` should both be active — confirm they still appear in this
session's skill listing; if either is missing, that's a regression to flag, not
something to work around silently.

## 9. External reference repos (`external/`): read, recorded, and removed on 2026-09-23

**Clone-on-demand policy (standard practice, not a special case).**
- All of these have been read, and what each contributed is recorded in the table below. Don't re-investigate them from scratch.
- On 2026-09-23 the folders were **deleted to reclaim disk**. Only `external/RDAH-Net/` remains; its checkpoints are
  symlinks to `data/checkpoints/rdah/`.
- **If code from one is ever needed again, re-clone it on demand** with the exact pinned command in `data/REGENERATION.md`,
  section "Deleted to reclaim disk, part 2". `external/dinov3` is restored from this repo's history with `git checkout ee30f7f -- external/dinov3`.
- Never run their scripts. Reimplement any logic ourselves. Keep the clone out of commits unless it's vendored on purpose.

| Folder | Source | Contribution |
|---|---|---|
| `sih2026-depthwizard` | `zaidnansari2011/sih2026-depthwizard` | Independently validated the full-DAv2-fine-tune idea (on its own non-comparable split) — this is where **Method 6's approach was sourced from**. |
| `depthwizard` | `blakc-coffee/depthwizard` | Real-DEM-low-frequency + model-high-frequency fusion mechanism — source of frequency fusion, **retired 2026-09-23** (fusion = its DEM-only control; §2b). |
| `arpitparashar06-depthwizard` | `arpitparashar06/depthwizard` | Also does DEM+model frequency fusion (independently flagged alongside blakc-coffee); its non-regression scale-derivation approach was tried and rejected (§4). |
| `DepthWizard-SIH26175` | `devendrakushwah80/DepthWizard-SIH26175` | Source of Method 6's two follow-on ablation ideas: GSD-FiLM conditioning (rejected) and height-balanced loss+sampling (**adopted**, current Method 6 recipe). |
| `amogh-hub-depthwizard` | `amogh-hub/depthwizard` | Source of evidence-gating + leave-one-out validation, adapted onto frequency fusion as LOBO (§2b). |
| `ArnabTechiee-depthwizard` | `ArnabTechiee/depthwizard` | Shadow-geometry photogrammetry approach — tested and rejected, fails at 10m GSD (§4). |
| `RDAH-Net` | (upstream RDAH-Net repo) | Source of the RDAH-Net fusion method itself (§2a Method 5, §3.2 open item). |
| `SynRS3D` | `JTRNEO/SynRS3D` | Synthetic RS 3D dataset/method referenced during the RDAH-Net investigation; not independently adopted. |
| `yats0x7-depthwizard` | `yats0x7/DepthWizard` | Ground-trend scale approach — code-read and compared against blakc-coffee's fusion (`docs/method-audit/` — code-read comparison entry); converges with blakc-coffee, no separate adoption. |
| `madhu-mitha-e-depthwizard` | `madhu-mitha-e/DepthWizard` | Part of the 20-repo competitive audit (`COMPETITIVE_REPO_AUDIT.md`) — reviewed, no method adopted from it directly. |
| `gowthamkrishna27-elevate3d` | `gowthamkrishna27/Elevate3d` | Part of the competitive audit — reviewed, no method adopted from it directly. |
| `dinov3` | (Meta DINOv3) | Backbone used for the DAv2-vs-DINOv3 comparison (§2b, `backbone-comparison.md`); required a HF-checkpoint state-dict conversion to run SAT493M — see that doc for the conversion steps if reused. |

Full 20-repo audit with claimed-numbers verification: `COMPETITIVE_REPO_AUDIT.md`
(repo root). Overall research-track state snapshot: `PROJECT_STATUS_REPORT.md`
(repo root).

---

**Consistency check against CLAUDE.md** (updated 2026-09-23, final close-out): CLAUDE.md's "one
rule that overrides everything else" (quota discipline, act don't ask) still applies and this doc
doesn't change it. CLAUDE.md's research-track items 5–10 were rewritten short. They and this doc
both summarize `docs/method-audit/final-comparison.md`, which is the single consolidated record,
and they agree on:
- Method 6 1.980/3.492/0.745/0.656, holding across 3 seeds
- FT-2 not adopted
- RDAH closed on Sentinel-2 by the pre-registered A5 rule and open on DFC2019 only for the
  Track1 split
- Sentinel-2 baselines: FABDEM (terrain) and GLO-30 (surface)
- frequency fusion retired
- no detail/canopy add-on passes; the learned GEDI route is gated

The "oracle" naming inconsistency is resolved: both figures are traced to code in
`final-comparison.md` §1.0. "Oracle" means only 3.392/4.579/0.582/0.509; 2.929/4.718/0.532/0.471
is Method 2's deployable sparse-GCP result. CLAUDE.md item 2 and HANDOFF §2a/§7 were corrected
to match.

---

**Session footer: 2026-09-23, "07: GAMUS generalization"** (brief: `docs/SIH26175_problem_statement.md`; record: `docs/method-audit/07-gamus-generalization/`)

- **B, GAMUS zero-shot:** Method 6 **does not generalize.** It wins MAE, r and ρ, and loses RMSE in all 3 cities.
- **C, GAMUS fine-tune:** a pre-registered stop (leaf-off).
- **D, 3DEP forest and mountain:** canopy p95 10.6 vs. 37.7 m. DEM + AGL doesn't add value.
- **E:** no deliverable fully met.
- **F:** the RF terrain residual is not adopted.
- **Product:** DEM-only, unchanged.
- **Also this session:**
  - seed-42 checkpoints re-created, bit-identical;
  - the Kaggle GPU offload pattern added to CLAUDE.md;
  - 20 closed-route artifacts and 11 `external/` repos deleted after documenting them in `data/REGENERATION.md` (about 8 GB freed).
- The numbers in this file, CLAUDE.md, `final-comparison.md` §7 and 07 `summary`/`verdict` were cross-checked programmatically against
  `zeroshot_merged_summary.json`, `forest_mountain_3dep/summary.json`, `terrain_rf_residual/summary.json` and `leafon_precheck.json`.
  0 mismatches.

**Session footer: 2026-09-24, "resolution transfer"** (record: `docs/method-audit/06-full-finetune-twin-head/resolution-transfer.md`; summary in `last_session.md`)
- **Phase 0:** native position-embedding interpolation; no fix needed.
- **Protocol P** (true pixel count): coarse→fine fails at 2 / 3 / 5 / 8 m.
- **Protocol R** (resample-back): coarse→fine works at 2 / 3 / 5 m and fails at 8 m.
- Coarse→fine beats fine→coarse at every GSD, in both protocols. Monotonic, no pre-registered cliff at or below 5 m.
- The numbers here were verified against `data/dfc2019/experiments/resolution_transfer/matrix_summary.json`.


**Session footer: 2026-09-24, "Sentinel-2 token-grid test"** (record: `docs/method-audit/06-full-finetune-twin-head/sentinel2-token-grid-test.md`)
- **Housekeeping.** DFC2019 GSD ≈ 0.3 m, measured from image content. The 0.5 m figure is retracted, with a dated correction in `stage0-gates/sparse-lidar-feasibility.md`.
- **Phase A.** Token-grid math for 4 prior Sentinel-2 attempts. A4 (Method 6) was R-like by count and failed.
- **Phase B.** R10 / R12 fail rule 1; the stop condition fired.
- **Phase C** (Kaggle T4; the output was validated locally before acceptance).
  - The model-free fields match an independent recompute exactly, and the analysis re-run reproduces `summary.json` exactly.
  - R does not help: 16/32 tiles, p_Holm = 0.82.
- **Phase D:** not triggered. Open item 0d is closed.
- **Rank-loss test:** no signal vs. the oracle (18/32, Holm p = 0.54). The Sentinel-2 learned-terrain-correction line is closed, as pre-agreed.
- **Repo housekeeping.**
  - `kaggle/`, `kaggle_phase2.5_package/` and `scripts 2/` were consolidated: scripts went to `scripts/` and manuals to `data/kaggle_bundles/` (1.86 GB freed; `data/REGENERATION.md`).
  - The Phase C Kaggle export was deleted after validation. Its script is kept as `scripts/s2_token_grid_phase_c_kaggle.py`.
- The numbers here were checked against `data/dfc2019/experiments/resolution_transfer/matrix_summary.json` and `data/sentinel2_benchmark/token_grid_test/kaggle/summary.json`.

**Session footer: 2026-09-26, "storage migration (licence split)"** (resumed after a usage-limit cut; code: `backend/storage/library_store.py`, `scripts/publish_library.py`; record: `docs/STORAGE.md`)
- **Licensing (primary sources):** the DFC2019 contest terms forbid redistribution, so its 50 tiles and ~100 thumbnails/previews stay private.
  Sentinel-2 (all 2025 acquisitions, Copernicus licence) and Maxar Open Data (3 distinct 2022 scenes, CC BY-NC 4.0) are public with attribution.
- **Checkpoints:** 5 Method 6 files (496 MB) on a PRIVATE HF repo. Byte-identical, load in PyTorch, anonymous access → 401.
  `scripts/vhr_dsm_pipeline.py` and the sanity-check script fetch from it (commit `fa4e22e`).
- **Library store (active):** sentinel2/vhr → public GitHub Release assets (`sancharimouri/depthwizard2-assets`, tag `library-v1`,
  114 files, 138.9 MB, attribution README, anonymous download verified, no DFC2019). dfc2019 + manifest → private HF dataset
  (`sancharimouri/depthwizard2-library-private`, 150 files + manifest, 140.7 MB, checksums match, anonymous → 401), proxied by the backend,
  no public tile URL. `DW2_LIBRARY=local` = local dev files. The R2 module (`backend/storage/r2.py`) is kept but dormant.
- **Verified before the cut:** live backend served all 88 catalog items with correct per-source routing; backend tests 23/23 (an outdated R2 fixture fixed).
- **Closed out after the resume:** all checks passed; details in `docs/STORAGE.md` §Verification.
  - Browser: 88/88 thumbnails; preview and magnifier work on both GitHub and HF-proxied items.
  - Production-mode backend: 176/176 image URLs return 200; it fails closed (503) without a token.
  - Tests: frontend 40/40; backend 22 passed + 1 skipped (the dormant R2 boto3 check).
  - Costs: no card or billing anywhere.
- **Not started, on purpose:** Prompt 2 (Render backend deployment).

**Session footer: 2026-09-26 (night), "deployment + desktop freeze trial"** (records: `docs/DEPLOY.md`, `docs/DESKTOP_FREEZE_TRIAL.md`)
- **Relative depth:**
  - Primary host: the ZeroGPU Space `sancharimouri/DepthWizard2`, through `/api/depth` with HF_TOKEN as an explicit header (audited).
  - Fallback: the Colab bridge, opt-in via `DAV2_FALLBACK_URL`. The UI's Relative Depth box shows the job's real depth.
- **ZeroGPU quota, measured:** 585 calls/day on the free account, then a 429.
  **Spent until about 2026-09-27 02:40 IST.** Use the fallback or wait.
- **Desktop freeze (macOS arm64):** 813 MB installed. Three build fixes: freeze_support, rasterio submodules, rasterio/pyproj data. `ee`/googleapiclient excluded (−103 MB).
- **Next session:**
  1. Push `desktop/freeze_trial/` to the private CI repo over HTTPS and run the Windows/Linux matrix. Report sizes and whether each fix is needed.
  2. Then the tiered tile bundling + update mechanism.
- **Still untouched, on purpose:** the 4 modified Sentinel `.tif` files, the `earthengine-api` line in pyproject/uv.lock, and the deleted `last_session.md`.

**Session footer: 2026-09-26, "desktop v1.0.0 published"** (record: `docs/DESKTOP_APP.md`)
- **Desktop app:** Tauri 2 + frozen backend on ONNX Runtime (no torch) + a tiered tile library (26 bundled, 62 on demand from the public GitHub Release).
  In-app E2E passes. Bundled-tile depth 0.22–0.26 s. Installer DMG 316.2 MB, app 432.4 MB.
- **Updates:** Tauri updater, signed. The key is in `~/.tauri/depthwizard2-updater.key` and its password in the Keychain ("depthwizard2-updater-signing"). **Back them up.**
  The live release is `github.com/sancharimouri/depthwizard2-desktop` v1.0.0.
- **DFC2019 is public temporarily, by owner decision** (see CLAUDE.md). The owner will replace it. Publish nothing more of it without asking.
- **Closed afterwards:**
  - the "Later" button is verified (postponed, nothing downloaded);
  - on-demand card sizes now come from the published release (DFC2019 2.22 MB);
  - downloadable tiles are listed first.
  The last two ship in the next release (v1.0.1).
- **Open:** Windows/Linux CI. Pushing the workflow needs the `workflow` scope on the gh token (`gh auth refresh -h github.com -s workflow`).

**Session footer: 2026-09-28, "DFC2019 terrain packs — Prompt 1 preflight"** (record: `docs/method-audit/08-dfc2019-terrain-packs/plan.md`)
- **Why:** DFC2019 tiles render as a flat plane because they have no georeference, so no DEM pack exists and `generate()` writes a zero mesh.
  The 5-prompt series builds a private curated pack per tile: ground + Method 6 held-out heights + AGL correction of tall objects.
- **Preflight facts:**
  - 50 tiles (26 JAX / 24 OMA); every AGL ≥ 99.9999 % valid (JAX_004_016 has 1 NaN).
  - All 50 RGB tiles are already public: 42 on library-v1, 8 inside the desktop installer. Packs stay private anyway.
  - Fold q holds out quadrant q; all checkpoints match the private HF repo by SHA-256.
- **Point clouds:** `~/Downloads/{JAX,OMA}_PointClouds.zip` are present and intact. They carry **per-point classes** (class 2 = ground), and Up looks ellipsoidal.
- **3DEP 1 m:** reads programmatically in both cities (JAX FL_Peninsular_2018, NAVD88; OMA NE Eastern UA 2016).
- **Web gap:** remote mode has no pack lookup; Prompt 5 adds `private_file(item, "dem")`.
- **Scratch:** `data/dfc2019/terrain_packs/` (self-gitignored).
- **Prompt 2, locate (2026-09-28):** `scripts/dfc2019_locate_tiles.py`; record in `08-dfc2019-terrain-packs/locate.md`.
  - Validity: V1 self-test 16/16, V2 negative control 0/50.
  - **15/50 confident** (JAX 8, OMA 7) under the pre-registered rule.
  - Post-hoc (not used): 49/50 matches fall in the cloud tile their name indexes, and same-index tiles agree within 2–30 m. Most C3 rejections come from a low bias in the cloud-AGL layer.
  - Follow-up: a pre-registered v2 rule (index check + C1/C2) could upgrade about 34 more tiles.
  - Private locations: `data/dfc2019/terrain_packs/locate/locations.json`. The OMA clouds are in UTM 14N.
- **Prompt 3, base ground (2026-09-28):** `scripts/dfc2019_base_ground.py`; record in `08-.../base.md`.
  - 15 tiles are dem-located (USGS 3DEP 1 m, sampled in the tile frame); 35 are city-level (3DEP 1/3″ median over the cloud footprint union): JAX 6.01 m, OMA 299.91 m.
  - Published check: Eppley Airfield 984 ft = 299.9 m; NAS JAX 23 ft = 7.0 m.
  - **US3D Up is ellipsoidal.** DEM − cloud = +28 m; after GEOID18 the residual is −1.5 m (JAX) / −0.8 m (OMA). No datum bug; the NAVD88 DEM is what ships.
- **Prompt 4, heights (2026-09-28):** `scripts/dfc2019_heights.py`; record in `08-.../heights.md`.
  - Held-out fold-q-on-quadrant-q predictions for 50/50 tiles (margin 192, feather ±16 px).
  - Seam ratio: median 0.95; JAX_165_015 and OMA_248_030 are at 1.96 (the reference AGL shows the same, so it is real structure).
  - **Cap 12 m, margin 3 m** (pre-registered rule). The median prediction plateaus at ~18–21 m (slope 0.28).
  - Parity with `vhr_dsm_pipeline`: 0.0 m.
- **Prompt 5, packs + integration (2026-09-28):** `scripts/dfc2019_build_packs.py`, `scripts/dfc2019_packs_qa.py`; record in `08-.../packs.md` (per-tile table).
  - Correction per the pre-registered rule (cap 12 m, margin 3 m): median 5.1 % of pixels replaced, max 30 %.
  - 50 packs are in the **private** HF dataset `dem/`.
  - Web path: `library_store.private_pack` + tag labels in `pipeline.generate`.
  - **QA: 50/50 DFC2019 jobs render 3D;** 6 screenshots in `data/dfc2019/terrain_packs/qa_screens/`. Tests: backend 41 passed + 1 skipped, frontend 53/53.
  - **Not deployed:** the Render backend needs this commit pushed and redeployed before the live site changes.
  - Follow-ups (packs.md list):
    1. OMA_281_002/030 show a ~120 m unclassified reference blob (artifact).
    2. A v2 location rule for 34 tiles.
    3. Stale manifest routing text and "GSD 30 cm".
    4. The elevation caption shows the terrain range.
    5. Desktop bundle mode is still flat.
