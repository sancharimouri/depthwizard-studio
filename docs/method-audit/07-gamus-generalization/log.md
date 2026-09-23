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
