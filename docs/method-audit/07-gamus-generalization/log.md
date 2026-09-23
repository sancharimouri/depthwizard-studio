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

---

## 2026-09-23 — Part F pre-registration (optional part; committed now, **run only after A–E are documented**)

**Method source.** Song, Chen & Yokoya (2026), *ISPRS J. Photogramm. Remote Sens.* 232:155–171, arXiv 2505.06905v3.
- §5.2: residual r = H_pred − H_ICESat-2; H_corr = H_pred − r̂.
- §5.2.1 HRF: a 64 × 64 window around each photon, about 27 handcrafted features in four groups:
  - (i) predicted-height statistics: mean, std, min, max, p90, p10;
  - (ii) Sobel magnitude: mean, std, p95;
  - (iii) RGB per-channel mean and std, plus simple indices "such as (G−R)/(G+R)";
  - (iv) fractions of 8 land-cover classes plus Shannon entropy.
- §6.1: a random forest with 100 trees.

**Our target (per the prompt): the TERRAIN product.**
- H_pred = FABDEM (EGM2008, the 10 m grid per tile). The reference = ICESat-2 **ground** photons (`data/icesat2_photons/`).
- Photons are aggregated per (10 m pixel, RGT) as the median ellipsoidal height, then converted per point to EGM2008
  (`PROJ_NETWORK=ON`, |N| > 1 m asserted).
- All 32 benchmark tiles.

**Deviations from the paper, stated before running.**
1. The paper works at 0.5 m, so its 64 px window is 32 m. Here 64 px of 10 m pixels is 640 m. The window is kept at
   64 × 64 px (literal) and the scale difference is noted.
2. The "predicted height" is FABDEM (terrain), not a monocular nDSM.
3. Group (iii) indices: (G−R)/(G+R), (G−B)/(G+B), (R−B)/(R+B), computed on window means. The paper names only the first.
4. Group (iv) needs the paper's OpenEarthMap segmentation model (0.5 m), which is unavailable and inapplicable at 10 m.
   - **Variant A** = groups (i)–(iii): **18 features**.
   - **Variant B** = A plus ESA WorldCover v200 (10 m, via Earth Engine) fractions of its 11 classes plus Shannon entropy,
     plus window means of ETH GCH 2020 and DINOv3-CHMv2: **32 features**.
   - B is the prompt's pre-registered variant B.
5. Sentinel-2 RGB comes from each tile's benchmark GeoTIFF.

**Split: held-out tracks.**
- GroupKFold(5) with groups = RGT, global across all tiles. A track held out is absent from training in every tile.
- Every (pixel, RGT) sample gets an out-of-fold prediction. No random photon splits.

**Models.**
- The RF uses `sklearn RandomForestRegressor(n_estimators=100, random_state=0)`, otherwise defaults, trained pooled across
  tiles on the training tracks.
- **Linear residual on FABDEM (recomputed):** per tile, r = a + b·FABDEM fitted by OLS on that tile's training-track
  samples of the same fold. Where a tile has fewer than 10 training samples in a fold, it falls back to r̂ = 0 (raw); the count is reported.
- **Raw FABDEM** = r̂ = 0.

**Metrics.** Per tile, on out-of-fold samples: RMSE and bias-removed RMSE (std) of the corrected terrain vs. ground; bias and median |e| are reported.

**Decision rule (per variant; Holm across A and B).** The RF is adopted only if it beats **both** raw FABDEM and the
linear residual on RMSE **and** bias-removed RMSE, each with:
- paired Wilcoxon (two-sided) Holm-adjusted p < 0.05; and
- ≥ 20 of 32 tiles won.

**Expected outcome: negative.** FABDEM is itself a globally trained ML correction of GLO-30 (random forest), it is already
the best terrain product here (32/32 vs. GLO-30), and the residual left is small. One run, no tuning.

---

## 2026-09-23 — Part D result: forested and mountainous terrain vs. airborne LiDAR (USGS 3DEP)

`scripts/forest_mountain_3dep_eval.py` → `data/forest_mountain_3dep/{windows.json,summary.json}`.
- Model: Method 6 = mean of 12 fold models, margin-192 tiling.
- Input: NAIP resampled to 0.3 m (primary).
- Scale: 8 windows × 600 m, 4 sites.

**Run incidents (neither touches the analysis).**
1. A transient remote-COG read failure ("Chunk and warp failed") after 3 windows. A 5× retry was added and the run resumed.
2. **Guard deviation.** The pre-registered no-op guard (median |EGM→NAVD88 shift| > 0.05 m) fired falsely at MLBS VA, where EGM2008 and NAVD88 nearly coincide (0.045 m).
   - It was replaced by the repo's standard guard: |geoid N| > 1 m at every pixel (N ≈ −30 m there).
   - The datum chain itself is unchanged: EGM → WGS84/ITRF2014 ellipsoid → Helmert to NAD83(2011) @2015.0 → GEOID → NAVD88.
- No HAG pixels were excluded by the > 100 m / < −2 m rule in any window.

### (a) nDSM: predicted AGL vs. 3DEP height above ground

Values are per-window means over 8 windows, on the 2 m grid.

| input | MAE | RMSE | bias | Pearson | Spearman | var ratio | p95 pred / LiDAR | p99 pred / LiDAR |
|---|---|---|---|---|---|---|---|---|
| **0.3 m** | 17.35 | 19.64 | **−17.16** | 0.404 | 0.382 | **0.125** | **10.6 / 37.7** | 12.6 / 42.4 |
| 0.6 m | 17.39 | 19.69 | −17.18 | 0.360 | 0.359 | 0.089 | 9.6 / 37.7 | 11.4 / 42.4 |

| window | LiDAR p95 | pred p95 | bias | Pearson |
|---|---|---|---|---|
| Olympic w0 / w1 | 53.8 / 49.8 | 11.7 / 10.4 | −28.4 / −14.8 | 0.43 / 0.38 |
| Tahoe w0 / w1 | 32.0 / 35.3 | 8.4 / 10.0 | −12.9 / −16.7 | 0.50 / 0.54 |
| GRSM w0 / w1 | 34.4 / 39.2 | 10.2 / 12.8 | −20.6 / −23.0 | 0.14 / 0.03 |
| MLBS w0 / w1 | 27.8 / 29.5 | 9.1 / 12.1 | −14.0 / −6.8 | 0.40 / 0.81 |

**Reading.** The canopy ceiling is **confirmed on independent airborne LiDAR in leaf-on forest.**
- Method 6's AGL p95 is about 10 m where real canopy p95 is 28–54 m.
- Variance ratio 0.12, against 0.48–0.66 on DFC2019.
- Correlation is weak to moderate. It is about 0 on steep GRSM, where the LiDAR ground itself is suspect.
- This is the DFC2019 training-range ceiling (tree p99 23.6 m) seen out of domain, and made worse:
  - NAIP at 0.6–1.0 m is coarser than 0.3 m;
  - these are closed tall forests, not urban trees.

### (b) Composed DSM = DEM + max(AGL, 0) vs. the 3DEP first-return DSM

The pre-registered rule is paired Wilcoxon, n = 8, Holm across 3 DEMs, and **both** RMSE and bias-removed RMSE must pass.

| DEM | RMSE raw → composed (mean) | wins | p (Holm) | bRMSE raw → composed | wins | p (Holm) | **verdict** |
|---|---|---|---|---|---|---|---|
| FABDEM | 11.36 → **8.56** | 7/8 | 0.031 | 7.75 → 7.60 | 5/8 | 1.0 | **does not add value** |
| GLO-30 | 9.16 → 12.84 (worse) | 0/8 | 0.023 | 7.13 → 7.45 | 4/8 | 1.0 | **does not add value** (harms) |
| SRTM | 9.63 → 11.78 (worse) | 2/8 | 0.039 | 8.38 → 8.31 | 4/8 | 1.0 | **does not add value** (harms) |

**Reading.**
- **FABDEM + AGL** lowers RMSE on 7/8 windows, and that part is significant. But the gain is almost all **bias**: FABDEM's mean bias vs. the LiDAR surface goes from −8.2 m to −3.0 m. Bias-removed RMSE is unchanged (5/8, p = 0.46).
  - So the predicted height restores part of the canopy's **level**, not its **shape**.
  - That is exactly what the R4 offset guard exists to block, so the pre-registered verdict is **fail**.
- **GLO-30 and SRTM + AGL are worse.** They are radar surface models that already carry most of the canopy, so adding AGL double-counts.
- **Post-hoc, descriptive:** the best raw surface product on these windows is GLO-30 (RMSE 9.16, bRMSE 7.13). Composed FABDEM has the lower mean RMSE (8.56) but a higher bRMSE (7.60).
- The brief's recommended route (a low-resolution DEM, e.g. SRTM, mapped to absolute heights) **plus Method 6** does not produce a better DSM than the DEM on forested and mountainous terrain.

---

## 2026-09-23 — Part B result: Method 6 zero-shot on GAMUS. **Pre-registered verdict: DOES NOT GENERALIZE.**

**Execution.**
- Split across two runners:
  - locally on MPS, 219 tiles (`zeroshot_tiles.jsonl`);
  - on Kaggle, 2,759 tiles (`zeroshot_tiles_kaggle.jsonl`, script `kaggle/gamus_zeroshot_kaggle.py`).
- Merged file `zeroshot_tiles_merged.jsonl` (not committed, 98 MB; regenerate by concatenation, local wins):
  - all **2,861** test tiles, exactly once;
  - 0 corrupt lines, 0 duplicate keys;
  - 2,848 scored; 13 without a scorable quadrant (too few valid px).
- **Cross-runner agreement** on the 117 tiles scored by both: max |Δ| is MAE 4.3e-5 m, RMSE 7.1e-5 m, Pearson 7e-6, Spearman 2e-6.
  The two hardware paths are interchangeable.
- Kaggle ran on a P100, and the final 162 tiles on a T4, after two session interruptions. Both resumed exactly.
  The run log's dataset path had a slug typo ("games-b-bundle"), corrected to "gamus-b-bundle" in the saved log (text only).
- **Leakage (A.3), confirmed:** 0 of 2,861 GAMUS test tiles share any non-flat 32×32 RGB block with the 50 DFC2019 tiles.
  "Non-overlapping" = "full split", and the two are identical.

**Headline: 2,848 tiles, mean of tiles (each tile's metric is the mean of its 4 held-out quadrants), 95% tile-bootstrap CI.**

| | MAE (m) | RMSE (m) | Pearson | Spearman | var ratio (pixel-pooled) |
|---|---|---|---|---|---|
| **Method 6, seed 42** | **3.130** [3.017, 3.246] | 4.583 [4.435, 4.739] | **0.638** [0.631, 0.645] | **0.583** [0.576, 0.590] | 0.305 |
| Method 6, seed 43 | 3.149 [3.035, 3.268] | 4.610 [4.461, 4.768] | 0.635 [0.627, 0.642] | 0.581 [0.574, 0.588] | 0.300 |
| Method 6, seed 44 | 3.146 [3.032, 3.264] | 4.610 [4.460, 4.766] | 0.634 [0.627, 0.642] | 0.580 [0.573, 0.587] | 0.305 |
| **Oracle per-tile OLS (frozen DAv2-L)** | 3.474 [3.387, 3.564] | **4.426** [4.319, 4.536] | 0.491 [0.480, 0.502] | 0.425 [0.416, 0.434] | 0.603 |
| Frozen DAv2-L (relative) | n/a | n/a | 0.504 [0.494, 0.513] | 0.436 [0.428, 0.445] | n/a |
| *Method 6 s42 + per-tile OLS (descriptive only)* | *2.853* [2.765, 2.942] | *3.908* [3.798, 4.021] | *0.635* | *0.580* | *0.607* |

**Method 6 − oracle, paired.** Tile wins are out of 2,848; Wilcoxon p is Holm-adjusted across the 3 seeds.

| metric | seed 42 Δ [CI] | tile wins | Wilcoxon p (Holm) | CIs non-overlapping? |
|---|---|---|---|---|
| MAE | −0.344 [−0.406, −0.282] | 2,152 | 2e-149 | yes, M6 better |
| **RMSE** | **+0.158 [+0.089, +0.230]** | 1,713 | 1e-13 | **no, M6 worse on the mean** |
| Pearson | +0.147 [+0.140, +0.154] | 2,385 | ≈0 | yes, M6 better |
| Spearman | +0.158 [+0.152, +0.164] | 2,604 | ≈0 | yes, M6 better |

(Seeds 43 and 44 are the same to ±0.03.)

**Per city: Method 6 wins MAE, Pearson and Spearman, and loses RMSE, in every city and for every seed.** So 0 of 3 city wins.

| city (tiles) | M6 s42 MAE / RMSE / r / ρ | oracle MAE / RMSE / r / ρ |
|---|---|---|
| DC (361) | 5.05 / 7.00 / 0.629 / 0.630 | 5.43 / 6.69 / 0.361 / 0.346 |
| NYC (987) | 4.03 / 5.52 / 0.490 / 0.460 | 4.26 / 5.24 / 0.293 / 0.277 |
| PHL (1,500) | 2.08 / 3.39 / 0.737 / 0.653 | 2.49 / 3.35 / 0.653 / 0.541 |

**Rule outcome.**
- Criterion 1 fails: RMSE is worse on the mean and the CIs overlap.
- Criterion 2 fails: 0 of 3 cities.
- This holds for every seed. **Method 6 does not generalize to GAMUS** under the pre-registered rule.

**Why RMSE loses while everything else wins.** Method 6 ranks heights much better than the oracle, and is closer on
most pixels (MAE, 60% of tiles on RMSE too). But it **compresses tall structures**, and RMSE is dominated by that tail.

Pixel-pooled, by true height (mean prediction / mean truth, m):

| true height | 0–2 | 2–5 | 5–10 | 10–20 | 20–30 | 30–50 | ≥ 50 |
|---|---|---|---|---|---|---|---|
| Method 6 s42 | 1.2 / 0.2 | 4.4 / 3.5 | 7.0 / 7.4 | 7.1 / 14.0 | **7.0 / 23.8** | **10.8 / 35.7** | **17.2 / 81.6** |
| Oracle | 2.3 / 0.2 | 4.4 / 3.5 | 6.0 / 7.4 | 8.8 / 14.0 | 14.1 / 23.8 | 24.8 / 35.7 | 53.9 / 81.6 |

- The oracle's per-tile scale fit, which sees each tile's own LiDAR, recovers the tall tail far better.
- Method 6's pooled variance ratio is 0.30, against 0.48–0.66 on DFC2019, so the compression is **worse** out of domain.
- **Trees on GAMUS** (tree class, by true height): 20–30 m trees are predicted at 5.4 m, and 30–50 m trees at 6.0 m.
  - That is the leaf-off imagery again (C.0), on top of the training-range ceiling.
- The mean-of-tiles variance ratio (26–29 for Method 6) is **not interpretable**. It is blown up by near-flat tiles where
  var(true) ≈ 0. This was noticed after seeing results (post-hoc), so the pixel-pooled figure is the one reported.

**Per-landscape stability (descriptive, as pre-registered).** Mean-of-tiles MAE over tiles with ≥ 100 px of the class.

| | urban (building) | sparse (ground + low veg) | forested (tree) | spread |
|---|---|---|---|---|
| Method 6 s42 | 3.72 (2,492 tiles) | **1.29** (2,840) | 5.03 (2,789) | **3.74 m** |
| Oracle | 3.87 | 3.45 | **4.69** | 1.24 m |

Method 6 is much better on sparse ground, slightly better on buildings, and **worse on trees**. It is therefore
**less stable across landscapes** than the oracle: a spread of 3.7 m vs. 1.2 m.

**Context vs. DFC2019 (in-domain, same metric definitions).**
- Method 6: DFC2019 1.980 / 3.492 / 0.745 / 0.656 → GAMUS 3.130 / 4.583 / 0.638 / 0.583.
- The oracle: 3.392 / 4.579 / 0.582 / 0.509 → 3.474 / 4.426 / 0.491 / 0.425.
- The margin over the oracle survives on MAE and on correlation. It disappears on RMSE.
- **Post-hoc observation, not a claim:** most of the DFC2019 "beats the oracle on all four" came from in-domain
  scale. With a per-tile scale fit (M6 + OLS, which needs local LiDAR), Method 6's ranking advantage would win on all four.

---

## 2026-09-23 — Part F execution note (the RF stage is offloaded to Kaggle; the protocol is unchanged)

- The local Part F run finished **feature extraction**: `terrain_rf_residual/samples.parquet`, 1,241,260 (pixel × RGT)
  samples, 32 tiles, 63 RGTs, about 8 min.
- It was then **stopped at the user's request**, 20:21 IST, during the first RF fit (heavy swap on this Mac).
  **No RF result had been produced or seen.**
- The RF stage moves to Kaggle CPU (about 30 GB RAM): `kaggle/terrain_rf_kaggle.py`, manual `kaggle/bundle_f/read.md`.
  - The filter, the `GroupKFold(5)` on RGT and the RF spec (100 trees, `random_state=0`, defaults) are identical.
  - The fold assignment was verified byte-identical to the local code before shipping.
  - Per-variant notebooks (A, B) run in parallel. Out-of-fold predictions are saved per fold, so a run can resume.
- **Merge:** `scripts/terrain_rf_residual.py --from-kaggle=<dir>`. It asserts identical folds, loads the out-of-fold
  predictions, and computes the linear baseline, per-tile metrics and the pre-registered decision locally.
