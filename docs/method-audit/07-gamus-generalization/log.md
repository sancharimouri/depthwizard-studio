# 07 — GAMUS generalization, NEON forests/mountains, deliverables audit (running log)

Chronological log for this session. Every entry is dated. Decision rules are committed
**before** the part they govern runs; anything judged after seeing results is labelled **post-hoc**.
Brief: `docs/SIH26175_problem_statement.md` (SIH26175 brief, IMG-PROCESS-SAC 2026, commit `ac078d5`).

Statistics convention for this whole folder (R3): tile-level bootstrap, 10,000 resamples, seed 0,
95% percentile CI; paired Wilcoxon signed-rank plus win counts; Holm correction across candidates.
Each number states its aggregation: **mean-of-tiles** (each tile's metric computed on its own
pixels, then averaged over tiles; this is Method 6's DFC2019 aggregation) or **pixel-pooled**.

---

## 2026-09-23 — Part A: GAMUS acquisition and leakage check

### A.1 Size, license, layout (measured via the HF API, `earthflow/GAMUS`)

- **License:** CC-BY-4.0 (dataset card). `XShadow/GAMUS`, the mirror the dataloader repo links, has an identical listing.
- **Total size 80.0 GB.** That's under the 150 GB stop threshold, but only about 20 GB of disk is free on this machine.
  - So tiles are **streamed**: download, reduce to metrics, delete. Nothing bulk is stored.
- **Layout:** `{images,heights,classes}/{train,val,test}/<CITY>_<id>_{RGB|IMG,AGL,CLS}.h5`.
  - One dataset per file, key `image`. There are no attributes and **no georeferencing**.
  - NYC images are named `_IMG`; the others are `_RGB`.

| split | DC | NYC | PHL | total | paper (Xiong et al. 2023) |
|---|---|---|---|---|---|
| train | 1,439 | 1,167 | 2,398 | **5,004** | 6,304 |
| val | 359 | 0 | 500 | **859** | 1,059 |
| test | 361 | 1,000 | 1,500 | **2,861** | 4,144 |

- Size by modality: images 27.5 GB, heights 36.7 GB, classes 15.9 GB.
- Test split alone: 25.3 GB.

**The mirror is not the paper's dataset.**
- The paper (arXiv 2305.14914, §3.2) describes 11,507 tiles from five cities: "Oklahoma, Washington D.C.,
  Philadelphia, Jacksonville, and New York City".
- The official code repo (`EarthNets/RSI-MMSegmentation` README) says:
  - the new version is on HF;
  - "The old version of data containing DFC 2019" is a separate LRZ zip;
  - a TODO is ticked: "Remove the cities (OMA and JAX) from the DFC 2019 dataset to ensure the label quality".
- So the Jacksonville and Omaha tiles in the paper's GAMUS **were DFC2019 tiles**. The HF version the brief
  links has them **removed**: 2,783 tiles fewer, which is exactly DFC2019's published tile count (the paper's Table 1).
- The session prompt's "4,144 test tiles" and "6,304 train" are the paper's numbers. Every GAMUS
  number in this folder is for the 3-city HF version.

### A.2 Properties

- **Sensor:** aerial orthophotos. The nDSM is LiDAR DSM − DTM, from city open-data portals (paper §3.1).
- **GSD:** not stated in the files, the dataset card, or the paper text.
  - DC estimate: DC's grid spans rows 1–77 × cols 2–67 of abutting 1,024 px tiles (abutment verified on pixels
    in `06-.../vhr_dsm_pipeline.md`). 2,159 tiles over DC's ≈177 km² gives about 0.28 m. The ≈22 km N–S extent over
    77 rows gives about 0.28 m.
  - So DC is about 0.3 m (estimate, ±15%). NYC and PHL GSD are **unknown** here.
- **nDSM:** float32 metres.
  - Nodata: DC uses a −5.0 sentinel (previous session).
  - NYC contains small negatives (min −0.44 m on NYC_00735), i.e. ground noise.
  - Kept convention: valid = finite & AGL ≥ 0, the same rule as DFC2019 and the DC retrain. The excluded
    fraction is recorded per tile in Part B.
- **Classes:** `uint8`. 1 ground, 2 low vegetation, 3 building, 4 water, 5 road, 6 tree (paper §3.2).
  - Value 0 also occurs (NYC_00735) and is not in the paper's list. It is treated as "unlabelled" and
    excluded from per-class breakdowns, but kept in whole-tile metrics.
- **Tile naming:** DC is `DC_<row>_<col>` on a grid. NYC and PHL are sequential IDs.
- **Official split geometry:**
  - **DC:** interleaved on the grid, so it leaks spatially (previous session: `DC_20_30` train is directly west of `DC_20_31` test).
  - **NYC:** split by ID block (test IDs 735–22833, train 22835–33931).
  - **PHL:** test is one ID block (3431–5448), with train on both sides (451–6149) and val 6150–6925.
  - NYC and PHL adjacency is unknown without georeferencing. **Matters only for Part C** (Part B never trains on GAMUS).

### A.3 Leakage check vs. Method 6's training tiles (DFC2019 JAX, 50 tiles)

- **By content provenance: zero overlap.** The HF GAMUS has no JAX or OMA tiles (see A.1). DC, NYC and PHL
  are 900–1,200 km from Jacksonville, so geographic overlap is impossible.
  - Georeferenced footprint intersection **cannot be computed**: GAMUS HF tiles carry no coordinates. This is
    reported as a limitation, not glossed.
- **Byte-level confirmation, run during Part B:** every GAMUS test tile's non-flat 32×32 RGB blocks are hashed
  against all 50 DFC2019 JAX RGB tiles. It catches any DFC2019 tile that slipped into the new version.
  - Expected: 0 shared blocks.
  - If any tile shares blocks, it is flagged as "overlapping" and B is reported with and without it.
- The "overlap-excluded" and "full split" results are therefore expected to be identical. Both are printed anyway.

---

## 2026-09-23 — Part B pre-registration (committed before any GAMUS inference)

**Checkpoints.**
- Method 6 = the adopted height-balanced recipe. Seeds 43 and 44 have saved fold checkpoints. Seed 42 (the
  1.980 headline run) never saved its checkpoints.
- Seed 42 is **re-trained with the identical command plus `--save-checkpoints`**
  (`data/dfc2019/experiments/method6_height_balanced_seed42_ckpt/`). Its DFC2019 numbers are reported next
  to the original 1.980 as a reproducibility check. MPS is not bitwise deterministic.
- **Per seed, the 4 fold models are averaged** (mean of μ). Each fold model saw 3/4 of every DFC2019 tile; none saw GAMUS.
- The full-data checkpoint (`method6_full_checkpoint`) is not used: its recipe differs.

**Inference.**
- Each GAMUS 1024² tile is split into its four 512² quadrants. Each quadrant is reflect-padded to 518 and fed
  to the model exactly as in the DFC2019 evaluation. No resampling (GAMUS ≈0.3 m, DFC2019 ≈0.3 m).

**Scale calibration.**
- Method 6's DFC2019 protocol applies **none**: the metric head output μ is scored directly. "Matched protocol"
  and "fully uncalibrated" are therefore the **same numbers**, reported once, and labelled as such.
- Descriptive extra, not a criterion: Method 6 + per-tile OLS, using the same 3-quadrant fit as the oracle.

**Baselines on the same pixels.**
- **Oracle per-tile-OLS on frozen DAv2**, same definition as DFC2019 (`scripts/evaluate_prior_spatial_cv.py`):
  - frozen DAv2-Small is run through the demo engine's exact preprocessing (whole tile → 518², bicubic back,
    min–max normalised; `backend/depth/depth_engine.py`);
  - AGL = a·DAv2 + b is fit on the tile's 3 other quadrants and scored on the held-out quadrant.
- **Frozen DAv2:** relative output, so only Pearson and Spearman are defined; MAE and RMSE are "n/a".

**Metrics.**
- MAE, RMSE, Pearson, Spearman, and the variance ratio var(pred)/var(true).
- Each is computed per (tile, quadrant). A tile's value is the mean of its 4 quadrants, then the mean of tiles.
  This is identical to the DFC2019 oracle's aggregation (every tile-fold weighted equally).
- The variance ratio is also given pixel-pooled.
- Breakdowns:
  - per city;
  - per class: building = urban, ground + low veg = sparse, tree = forested, plus road and water. Reported
    pixel-pooled and as mean-of-tiles over tiles with ≥ 100 px of the class;
  - by true-height bin: 0–2, 2–5, 5–10, 10–20, 20–30, 30–50, ≥ 50 m (pixel-pooled).

**Decision rule** ("Method 6 generalizes"). Applied to each seed's fold-ensemble separately. It holds only if
**all three seeds** pass, the same "every seed" convention as DFC2019 C3.
1. On the non-overlapping test tiles, all 4 metrics are better than the oracle, and the two methods'
   95% tile-bootstrap CIs of the mean-of-tiles do **not** overlap. Direction: lower MAE and RMSE, higher
   Pearson and Spearman.
2. City rule. The prompt's "≥ 4 of 5 cities" was written for 5 cities; the HF version has 3. **80% of 3 ⇒
   all 3 cities are required.** A city "win" is a better mean-of-tiles than the oracle on all 4 metrics in that city.
- Also reported, not part of the rule: the paired difference CI, paired Wilcoxon per metric with Holm across
  the 3 seed candidates, and tile win counts.
- **Per-class stability** is the spread (max − min) of per-class MAE across building / sparse / tree. It is descriptive only.

---

## 2026-09-23 — Part C.0 pre-registration (leaf-on pre-check, before any GAMUS training)

**Why this check exists.** The DC pilot's diagnosis (`06-.../vhr_dsm_pipeline.md`, "Retrain results"):
- GAMUS DC and NYC are leaf-off: 9% of tree pixels are green, against DFC2019 JAX at 40% and Sikkim `a_forest` at 93%;
- the canopy gain did not transfer;
- brown texture was learned as height.

**Measure.**
- The same ExG = (2G − R − B)/(R + G + B) as `scripts/tree_greenness_check.py`.
- Per tile, the share of class-6 (tree) pixels with ExG > 0.05. Tiles need ≥ 1,000 tree pixels to count.
- Per city, the median over **40 random tiles drawn from all splits** (seed 0).
- All three cities are re-measured on this common sample, including DC and NYC.

**Leaf-on threshold.** A city counts as leaf-on if its median share is **≥ 0.40**. That is the DFC2019 JAX level:
the trees Method 6 learned from, whose tall-tree predictions it gets wrong in a different way.

**Rule.**
- **No city ≥ 0.40:** Part C's full fine-tune is **not run**. This is a pre-registered stop. Per the DC diagnosis,
  Part C is not expected to fix the canopy ceiling regardless of data volume.
- **Some city ≥ 0.40:** Part C is scoped to that city's tiles only.
- In either case, before anything is adopted, Part C adds a pass criterion on the six Sikkim VHR crops (below).

**Added Part C criterion** (only used if C runs).
- Rerun the six Sikkim VHR crops with `--margin=192`.
- The non-forest crops (`c_terraces`, `b_glacier`, `c_river`) must not regress beyond their existing pass thresholds
  in `06-.../vhr_dsm_pipeline.md` (lines 62–65):
  - C1: river median AGL < 1 m; glacier p95 < 2 m; every crop max < 80 m;
  - C2: seam ratio ≤ 1.5 and negative AGL < 5%;
  - C4: terraces ρ > 0.
  (Terraces have no C1 magnitude threshold. The old seam-fixed model passes all of these on these crops,
  so any failure is a regression. Under the DC pilot model the river median was 2.25 m, a C1 fail.)
- A model that lifts canopy by degrading these is not adopted.

---

## 2026-09-23 — Part C.0 result: **pre-registered STOP. Part C's fine-tune is not run.**

`scripts/gamus_leafon_precheck.py` → `data/gamus_eval/leafon_precheck.json`. The values are the median over tiles of the per-tile share of tree pixels with ExG > 0.05. The sample is 40 random tiles per city from all splits, seed 0; tiles with < 1,000 tree px are skipped.

| city | tiles used | median share green [p25, p75] | tiles ≥ 0.40 | leaf-on (≥ 0.40)? |
|---|---|---|---|---|
| DC | 40 | **0.064** [0.038, 0.118] | 0 | no |
| NYC | 38 | **0.034** [0.012, 0.102] | 1 | no |
| PHL | 39 | **0.315** [0.225, 0.413] | 12 | no |
| *ref: DFC2019 JAX (Method 6 training)* | 50 | 0.40 | | threshold |
| *ref: Sikkim `a_forest`* | | 0.93 | | |

- **No city clears the leaf-on threshold, so Part C's full fine-tune is not run**, per the rule committed in `b83399a`.
- This agrees with the DC pilot's diagnosis:
  - the tall-canopy supervision GAMUS offers is leaf-off (DC, NYC) or mixed-season (PHL);
  - more of it is not expected to lift the canopy ceiling on leaf-on forest, whatever the data volume.
- DC and NYC come out lower than the earlier 9% because this is a per-tile median, while the earlier figure used a different sample. Same conclusion.
- **Post-hoc observation, not acted on:**
  - PHL imagery is mixed-season: 12 of 39 sampled tiles are individually ≥ 0.40, and the max tile is 0.92.
  - A tile-level "leaf-on PHL" subset (roughly 30% of PHL, about 1,000 tiles) could be selected by ExG.
  - Doing so would re-scope a city-level rule after seeing its result, so it is **not run**. It is listed as an open option.
  - Also note that PHL's canopy is short (tree p99 21.2 m, `06-.../vhr_dsm_pipeline.md`). A leaf-on PHL subset would add greenness, not height.
- **Consequences:**
  - Part C(3) and its adoption rule do not apply.
  - Part D runs Method 6 only ("Part C model, if adopted" is void).
  - The added Sikkim non-regression criterion stays on record for any future GAMUS fine-tune.

---

## 2026-09-23 — Part B pre-registration amendment (before the full run; the only inference so far is a 4-tile smoke test of the script)

- **The oracle's frozen DAv2 is DAv2-Large, not Small.** The DFC2019 oracle depth maps came from the
  demo engine with `Depth-Anything-V2-Large` (`data/dfc2019/experiments/dav2_baseline/config.json`), and
  the engine is still set to Large (`backend/config.py`). To keep the "same definition", Part B uses the
  engine unchanged. So the oracle and `dav2_raw` are **DAv2-Large**. Method 6 is fine-tuned from DAv2-Small.
- **Script:** `scripts/gamus_zeroshot_eval.py` (streamed; output `data/gamus_eval/zeroshot_tiles.jsonl`).
  `h5py` was installed into `.venv` for this; `pyproject.toml` is untouched.
- **Smoke test (4 DC tiles, seeds 43 and 44, not results):** 0 blocks shared with DFC2019. The script runs end to end.

---

## 2026-09-23 — Part E: deliverables audit (read-only) → `docs/deliverables-audit.md`

- **No deliverable is fully met.**
  - Partial: upload, texture drape plus an orbit "flythrough", 4-landscape region coverage.
  - Missing from the app: the rDSM path, the metric-DSM path, GeoTIFF export, slope and height analysis, in-UI
    validation, standalone packaging.
- The elevation layer is **DEM-only** and is already labelled "DEM ELEVATION", so the label is accurate.
- Screenshot: `docs/screenshots/2026-09-23_deliverables_audit_home.png`, headless Chrome, since the extension was not connected.
- Frontend untouched (`git status frontend` shows only the pre-existing untracked `public/data/vhr/`).

---

## 2026-09-23 — Part D pre-registration (committed before any Part D download or inference)

### D.0 Data-source deviation: NEON → USGS 3DEP + NAIP (blocker, default chosen)

- **NEON is blocked.**
  - The NEON **products** API answers without authentication, and all three AOP products exist at the candidate sites:
    DP3.30010.001 camera mosaic 10 cm, DP3.30024.001 LiDAR DSM/DTM 1 m, DP3.30015.001 CHM 1 m.
  - Every **data/file** endpoint returns `403 Access Denied`. NEON now requires an API token for data
    downloads, and that needs a user account. Creating one is not something this session may do, and
    `.env` has no NEON token.
- **Default chosen, per CLAUDE.md's blocker rule:** independent airborne-LiDAR truth plus leaf-on aerial RGB, both served
  anonymously by Microsoft Planetary Computer:
  - **Reference:** USGS 3DEP LiDAR rasters, `3dep-lidar-dsm` (first-return surface) and `3dep-lidar-hag`
    (height above ground).
    - 2 m, NAD83 / UTM + **NAVD88 height** (compound CRS read from the COG).
    - PDAL-reprocessed: SMRF ground, `hag_nn`.
  - **RGB:** USGS NAIP, 0.3–1.0 m, 4-band; RGB bands are used.
- **Rerun offer:** with a NEON token in `.env`, the same protocol can be rerun on NEON AOP. This is an open item.

### D.1 Sites, items, windows (all chosen by rule, before seeing any prediction)

**Sites.** Four forested and mountainous sites with 3DEP coverage on Planetary Computer, from a probe of 18 candidate locations:
1. **Olympic Peninsula, WA** (47.95, −123.90): temperate conifer.
2. **Lake Tahoe / Placer Co., CA** (39.10, −120.10): Sierra mixed conifer.
3. **Colorado Front Range** (40.35, −105.60): subalpine conifer, high relief.
4. **Great Smoky Mountains, TN** (35.689, −83.502): the NEON GRSM site, Appalachian deciduous, steep.
- NEON's SOAP, WREF and NIWO have no 3DEP raster on Planetary Computer. Backups in order: MLBS VA/WV (37.378, −80.525), then White Mtns NH (44.10, −71.40).

**Item per site.**
- Candidates are the `3dep-lidar-hag` items intersecting a 0.3° × 0.3° box centred on the site.
- Eligible if the STAC `valid_percent` ≥ 95 and the STAC HAG `mean` ≥ 8 m. If none qualifies, relax to mean ≥ 5 m;
  if still none, use the next backup site.
- One eligible item is drawn at random (seed 0). Its `-dsm-` twin (same id stem) must exist.

**Windows.**
- Two non-overlapping 600 m × 600 m windows per item, top-left corners drawn at random (seed 0) on the 2 m grid.
- Accepted if DSM and HAG are ≥ 98% valid and ≥ 50% of pixels have HAG > 5 m (forested). Up to 500 draws.
- **8 windows in total.**

**RGB.**
- NAIP items with month June–September (leaf-on) and GSD ≤ 1.0 m. Among those, the acquisition date closest to the
  3DEP item's `end_datetime`; ties go to the finer GSD. Items are mosaicked if a window spans several.
- **Primary input:** RGB bilinearly resampled to **0.3 m** on the window's UTM grid (2000 × 2000 px). This is
  Method 6's training GSD; the real information content is the NAIP GSD (0.6–1.0 m), inside the 0.3–2.4 m range.
- **Secondary, descriptive:** 0.6 m input.
- **Known confound:** the time gap between the LiDAR and NAIP dates (canopy growth, harvest) is reported per window.

**Reference cleaning.** HAG pixels > 100 m or < −2 m are excluded as ground-classification failures on steep
terrain; the GRSM item's STAC max is 949 m. The excluded fraction is reported. No other filtering.

### D.2 Models and evaluation

- **Model:** Method 6 = the mean of all 12 fold models (seeds 42, 43, 44 × 4 folds). There is no Part C model (C.0 stop).
- **Tiling:** `scripts/vhr_dsm_pipeline.py` sliding windows with `--margin=192`, the seam-fixed setting.
- **(a) nDSM:**
  - predicted AGL is block-averaged to the 2 m reference grid and compared with HAG;
  - per window: MAE, RMSE, Pearson, Spearman, bias, variance ratio, and the p95 of AGL vs. the p95 of HAG (the canopy ceiling);
  - **descriptive; no pass rule was set for (a).**
- **(b) Composed DSM:**
  - composed = DEM + max(AGL, 0) on the 2 m grid, with the DEM bilinearly resampled from 30 m (SRTM, GLO-30, FABDEM via Earth Engine);
  - baseline = the DEM alone;
  - reference = 3DEP first-return DSM.
- **Datums, per pixel:**
  - SRTM EGM96 (EPSG:5773), GLO-30 and FABDEM EGM2008 (EPSG:3855);
  - each is converted to NAVD88 at every pixel's lat/lon with pyproj (`PROJ_NETWORK=ON`), source EPSG:4326+{5773|3855} → target EPSG:6318+5703;
  - the script asserts that the operation uses geoid grids and that the median |shift| > 0.05 m, i.e. not a silent no-op.
- **Metrics per window:** RMSE and bias-removed RMSE (the std of the error), plus MAE, bias and median |error|.

**Decision rule (b), per DEM.**
- Composed "adds value" over the raw DEM if it is better on **both** RMSE and bias-removed RMSE.
- Test: a paired two-sided exact Wilcoxon across the 8 windows.
- Threshold: **Holm-adjusted p < 0.05 across the 3 DEMs** (per metric), with win counts reported.
- The prompt's "all tiles won if n < 8" does not apply because n = 8.
- **Prior expectation, stated now:** SRTM and GLO-30 are radar surface models that already contain part of the
  canopy, so adding AGL double-counts. FABDEM (bare earth) is the principled base.

### D.3 India transfer check (descriptive only; no rule)

- The existing composed DSMs `data/vhr_dsm/*_margin192/dsm.tif` = FABDEM + max(Method 6 seed-43 AGL, 0), EGM2008, six Sikkim/Darjeeling crops.
- References:
  - ICESat-2 20 m segments (sliderule PhoREAL, same parameters as `scripts/fetch_icesat2_segments20m.py`), surface = ground + canopy top;
  - GEDI L2A `elev_highestreturn` (WGS84 ellipsoid);
  - each is converted per point to EGM2008.
- Reported per crop: n, bias, MAE and median |error| for the composed DSM and for raw FABDEM and GLO-30 at the same points.
- n is expected to be small; no inference is drawn beyond description.

---

## 2026-09-23 — Part D selection (rule output; no inference yet) → `data/forest_mountain_3dep/selection.json`

- **Olympic WA:** 4 items eligible (mean ≥ 8 m).
- **Tahoe CA:** 0 at ≥ 8 m, 3 at ≥ 5 m. The item drawn is `USGS_LPC_CA_NoCAL_Wildfires_B1_2018`, not the Placer Co. item probed earlier.
- **GRSM TN:** 6 eligible.
- **Front Range CO: no eligible item** (16 candidates, none with mean ≥ 5 m and ≥ 95% valid). Replaced by the first backup, **MLBS VA**: 5 eligible at ≥ 5 m.
- Each site yielded 2 windows within ≤ 4 draws, **8 windows** in total.

| site | window NAIP | GSD | LiDAR–NAIP gap |
|---|---|---|---|
| Olympic WA | 2017-08-26 | 1.0 m | 237 d |
| Tahoe CA | 2018-09-16 | 0.6 m | 258 d |
| GRSM TN | 2016-06-08 | 1.0 m | 207 d |
| MLBS VA | 2012-09-11 | 1.0 m | **1,938 d** |

- **MLBS gap.** The rule picked 2012 for MLBS: no leaf-on NAIP at ≤ 1 m is closer to the 2018 LiDAR there. Its 5.3-year gap is a known confound, reported, not fixed.
- **Caveat.** The 3DEP STAC items carry year-level dates only (YYYY-01-01), so the gaps are approximate.

---

## 2026-09-23 — Part D.3 result (descriptive only): Sikkim composed DSM vs. ICESat-2 20 m and GEDI

`scripts/sikkim_dsm_vs_sparse_lidar.py` → `data/vhr_dsm/_diagnostics/sparse_lidar_dsm_check.json`.

**Setup.**
- Composed = FABDEM + max(Method 6 seed-43 AGL, 0), margin-192, EGM2008.
- Each raster's value = its p98 inside the footprint (GEDI radius 12.5 m, ICESat-2 10 m).
- References converted per point from the WGS84 ellipsoid to EGM2008, asserting |N| > 1 m.

**Deviations.**
- GEDI surface = `elev_lowestmode + rh98`, which is what the existing fetcher exports, not `elev_highestreturn`.
- **Post-hoc:** the median bias and the count of |error| > 50 m were added after seeing gross outliers.
  - ICESat-2 mean biases were +31 to +342 m, driven by 1–8 outlier segments per crop (clouds or false returns).
  - GEDI on `c_river` sits 125–150 m low on 7 of 12 shots: the lowest mode is in a deep gorge, or geolocation fails on the steep valley walls.
- **The medians are the readable numbers.**

**Median error, raster − lidar (m); n = footprints/segments.**

| crop | ref | n | composed | FABDEM | GLO-30 |
|---|---|---|---|---|---|
| c_town | GEDI | 13 | **−1.6** | −8.0 | −3.4 |
| c_town | IS2 20 m | 12 | +5.9 | −1.2 | +0.2 |
| c_terraces | GEDI | 20 | −12.2 | −17.5 | −10.7 |
| c_terraces | IS2 20 m | 24 | +2.6 | −2.6 | +5.6 |
| a_forest | GEDI | 13 | −16.5 | −30.0 | −17.9 |
| a_valley | GEDI | 5 | +6.0 | −5.3 | +2.5 |
| a_valley | IS2 20 m | 10 | +1.3 | −1.7 | +0.7 |
| c_river | IS2 20 m | 84 | +1.0 | −2.2 | −0.3 |
| b_glacier | GEDI | 11 | −0.4 | −0.6 | −0.6 |

- Not usable: c_river GEDI (gorge failures). a_forest and b_glacier have no ICESat-2 20 m segments; one granule had a server-side read failure.

**Reading (descriptive, small n, no test).**
- Against **canopy-top GEDI**, the composed DSM is always closer than bare-earth FABDEM, which it should be, since it adds height.
  - It is about equal to raw GLO-30: better on town and forest, slightly worse on terraces and valley.
  - In the forest, composed and GLO-30 both sit about 17 m below GEDI's rh98. That is the known AGL ceiling plus GEDI's slope inflation.
- Against **ICESat-2 20 m** surface segments, all three rasters sit within ±6 m in median, and none is consistently best.
- **This does not show that composition beats GLO-30 as a surface product on Indian mountain terrain.**
  It shows that composition turns FABDEM into something at GLO-30's level.

---

## 2026-09-23 — Seed-42 fold checkpoints re-created (reproducibility check: exact)

- `data/dfc2019/experiments/method6_height_balanced_seed42_ckpt/` was run with the identical command plus `--save-checkpoints`.
- It reproduces the adopted headline **bit-for-bit**:
  - mean of fold means MAE 1.979971505587631, RMSE 3.4922650091556022, Pearson 0.7445242046106612, Spearman 0.6563199482216565;
  - these are identical to `method6_height_balanced/m6_heightbal_results.json`;
  - per fold: 1.883 / 2.060 / 2.009 / 1.968.
- So training on this machine is deterministic for a fixed seed, and the checkpoints are exactly the headline model.
  Checkpoints are not committed (size); the results JSON and train log are.
- Part B (full GAMUS test, all 3 seeds) and Part D (`run`) launched in the background at 15:44 IST.

## 2026-09-23 — Part F citation (supplied by the user: `Claude_Research.pdf`)

- The "27-feature sparse-LiDAR RF" is **Song, Chen & Yokoya, "Sparse LiDAR-Guided Correction"**:
  - arXiv 2505.06905 (v3, 8 Dec 2025);
  - *ISPRS J. Photogramm. Remote Sens.* 232:155–171 (2026);
  - DOI 10.1016/j.isprsjprs.2025.12.004.
- Per the research note, it regresses the residual (prediction − ICESat-2) with a random forest on about 27 handcrafted
  features over a 64×64 window:
  - predicted-height statistics: mean, std, min, max, p90, p10;
  - Sobel-gradient statistics;
  - optical indices, e.g. (G−R)/(G+R);
  - land-cover fractions and Shannon entropy;
  - or, alternatively, a ViT patch embedding.
- The exact list is to be taken from the paper itself before Part F is pre-registered.
