# Semantic prior (Method 3) — Summary

_Sourced from the actual JSON/scripts/rasters in `data/dfc2019/experiments/`,
`data/sentinel2/`, and `scripts/`, re-derived and cross-checked directly — not
taken from `raw-notes.pdf`'s prose at face value. Only new computation run:
recomputing raster statistics that already existed as pixels on disk (no
model was refit, no experiment was rerun) and one visual QC render to check a
claim the note asserts but never shows evidence for._

## 1. The model and its two roles

`AGL = β0 + β1·D + β2·Pb + β3·(D×Pb)`, D = DAv2 relative depth, Pb = building
probability/fraction. Two separate uses, confirmed from the actual scripts:

- **DFC2019 benchmark** (`scripts/generate_dfc_building_prior.py`): runs the
  HOTOSM DINOv3-S ONNX model (`models/semantic/hotosm_dinov3s_buildings/model.onnx`)
  directly on DFC RGB tiles, sigmoid(logits) resized back to tile resolution,
  saved to `data/dfc2019/experiments/semantic/building/<tile>.npy`. No DFC CLS
  file is referenced anywhere in this script.
- **Sentinel-2 India demo** (`scripts/prepare_method3_semantic_inputs.py`,
  `scripts/repair_method3_coverage.py`): rasterizes Microsoft
  GlobalMLBuildingFootprints polygons (real `.csv.gz`/GeoJSONL partitions,
  parsed line-by-line and rasterized on a 5x-supersampled sub-grid) onto the
  exact Sentinel pixel grid, producing `building_fraction.tif` plus derived
  density/edge/distance rasters per city.

These are architecturally separate: the DFC branch is the one actually
regression-tested (Tests 1 and 2 below); the Sentinel branch only produced
QC'd rasters — see §5.

## 2. Item 1 — the "mock values" warning vs. the cited Jaipur numbers

**The numbers are real, not mock.** Reproduced independently, twice:

- Read `data/sentinel2/jaipur/semantic/building_fraction.tif` at full
  resolution directly: mean = **0.20584** (rounds to the note's 0.2058).
  Full-resolution p95 = **1.0** (does *not* match the note's 0.8674).
- Replicated exactly what `scripts/make_method3_qc_pdf.py` does for its
  on-page statistics: `rasterio` `Resampling.average` downsample to
  `max_size=1400` before computing `mean`/`p95` (this is a real code path in
  that script, used for both the plotted panel and the info-panel text).
  On that downsampled array: mean = **0.2058**, p95 = **0.8674** — an exact
  match to both numbers the note cites.
- Independently opened the actual PDF at
  `data/sentinel2/method3_semantic_QC.pdf`, page 7 (Jaipur): it prints
  "Building fraction: mean = 0.2058, p95 = 0.8674" verbatim.

So the note's own warning ("[QC PDF] contains mock numerical values, don't
refer to values") is **incorrect as applied to these two numbers** — they are
real, reproducible from the actual raster file via the QC script's own
documented resampling step, and they appear character-for-character in the
real PDF.

Two caveats worth keeping though:
- **p95 is a display-resolution artifact.** At full 10 m Sentinel resolution,
  building_fraction is highly bimodal (near-0 or 1 per pixel), so its true
  p95 is 1.0. The 0.8674 figure only exists because of the QC script's
  `max_size=1400` average-downsampling; it characterizes the downsampled
  preview, not the production-resolution layer that Method 3 would actually
  train against.
- **No dedicated "stats file" exists.** The note says "refer to generated
  stats files in the mentioned folders instead" — but no `.csv`/`.json` stats
  file exists anywhere under `data/sentinel2/semantic_sources/` or any
  city's `semantic/` folder. The only way to get these numbers is to
  recompute them from the raw `.tif`, which is what was done here. The
  note's own redirection points at something that was never generated (see
  `gaps-and-fixes.md` §1).

## 3. Item 2 — what "Baseline" means, and whether Test 1/Test 2 compare the same model

Read `scripts/evaluate_prior.py` (Test 1) and `scripts/evaluate_prior_spatial_cv.py`
(Test 2) in full. Confirmed identical in both:

- **Baseline**: per-tile (Test 1) / per-fold (Test 2) `sklearn.LinearRegression`
  fit of `AGL = a·D + b`, DAv2 only.
- **Method 3**: the identical `LinearRegression` call on features
  `[D, Pb, D×Pb]` — i.e. Method 3 = Baseline + two extra terms, a clean
  ablation, not a different estimator or a different fitting scheme.

The two tests differ only in evaluation protocol:
- **Test 1** (`evaluate_prior.py`): fits and evaluates on the *same* full-tile
  pixel set — in-sample.
- **Test 2** (`evaluate_prior_spatial_cv.py`): splits each tile into 4 spatial
  quadrants, trains on 3, evaluates on the 4th held-out quadrant, all 4 folds
  per tile (50 tiles × 4 = 200 folds) — spatial holdout.

One minor aggregation-granularity difference (not a correctness issue, just
worth naming): Test 1's headline numbers are the mean of 50 per-tile means;
Test 2's are the mean over all 200 folds directly (not tile-then-fold), so a
tile that fails on 1 fold is weighted 1/200 in Test 2 vs. potentially fully
in a per-tile summary. Both are legitimate; they are not literally the same
averaging operation.

**Conclusion: confirmed — Test 1 and Test 2 are the same model (Baseline and
Method 3, identically defined) under two different evaluation protocols, not
two different baselines sharing one label.**

## 4. Item 6 — reproducing the exact numbers

Pulled directly from the two result files (not recomputed/refit):

**Test 1** — `data/dfc2019/experiments/semantic/method3_results.json`,
`aggregate_tile_mean`, n=50 tiles:

| Metric | Baseline | Method 3 |
|---|---:|---:|
| MAE | 3.0349 m | 2.8975 m |
| RMSE | 4.2933 m | 4.1231 m |
| Pearson | 0.5457 | 0.5843 |
| Spearman | 0.4775 | 0.4901 |

**Test 2** — `data/dfc2019/experiments/semantic/method3_spatial_cv_results.json`,
n=50 tiles / 200 folds:

| Metric | Baseline | Method 3 |
|---|---:|---:|
| MAE | 3.3924 m | 3.3948 m |
| RMSE | 4.5787 m | 4.9065 m |
| Pearson | 0.5824 | 0.5542 |
| Spearman | 0.5093 | 0.4856 |

Both tables match the note **exactly**, to the last quoted decimal.

## 5. Item 4 — same 50 tiles across Methods 1/2/3?

Confirmed. `data/dfc2019/experiments/dav2_baseline/manifest.csv` (the
Methods-1/2 benchmark), `method3_results.json`'s tile list, and
`method3_spatial_cv_results.json`'s tile list are the **exact same 50 tile
IDs**, and the DAv2 depth-file directory (`dav2_baseline/depth/`) that all
three Method-3 scripts key off of contains exactly those 50 files. True
cross-method comparability on this axis.

## 6. Item 5 — the "never use DFC CLS as an input" rule

Grepped `generate_dfc_building_prior.py`, `evaluate_prior.py`, and
`evaluate_prior_spatial_cv.py` for any reference to `CLS`: **zero hits in all
three.** The DFC building-probability maps come only from RGB→HOTOSM
inference; the regression scripts only ever load `<tile>_depth.npy`,
`<tile>.npy` (building probability), and `<tile>_AGL.tif`. `*_CLS.tif` files
do exist in `data/dfc2019/raw/Truth/Track1-Truth/` but are not touched by any
Method-3 script. Rule followed in code, confirmed by reading the scripts
directly (not inferred from the note's prose).

## 7. Item 3 — what actually produced the DFC building-probability maps

`scripts/generate_dfc_building_prior.py` loads
`models/semantic/hotosm_dinov3s_buildings/model.onnx` via `onnxruntime`,
runs it directly on `data/dfc2019/raw/RGB/Track1-RGB/<tile>_RGB.tif` (ImageNet-
normalized, resized to 256×256, sigmoid output resized back to original tile
shape), tile list taken from the DAv2 benchmark cache
(`dav2_baseline/depth/*_depth.npy`, 50 files, so exactly the 50-tile
benchmark — not all 2,783 DFC tiles). This **is** the HOTOSM DINOv3-S model
from §2 of the note, run directly on DFC RGB — the note's own manifest
paragraph just describes the Sentinel/GlobalMLBuildingFootprints path
immediately above the HOTOSM model-path line, which reads as if the model
path belongs to that paragraph. It's a documentation ordering/clarity issue,
not a factual error — the actual DFC building-probability maps genuinely
come from HOTOSM DINOv3-S, confirmed from the script itself.

## 8. Item 7 — Sundarbans "useful near-zero-building case"

The QC-page numbers for Sundarbans (page 10 of
`data/sentinel2/method3_semantic_QC.pdf`) are real and dramatic: mean = p95 =
0.0000, **>0 = 0.00%** — genuinely zero building-footprint coverage anywhere
in that 10×10 km AOI. So the *description* is factually accurate.

But: grepped every script in `scripts/` for "sundarbans" (or any other city
name) inside `evaluate_prior.py` / `evaluate_prior_spatial_cv.py` — **zero
hits**. Those two scripts operate only on the DFC2019 50-tile benchmark; there
is no script anywhere that runs the Method-3 regression on Sentinel building
priors, and no per-region breakdown file of any kind exists for the Sentinel
cities. **"Useful near-zero-building case" is asserted, not followed up
anywhere in this repo.** It's a true description of the raw data, not a
tested claim about Method 3's behavior on it.

## 9. Item 8 — Kolkata exclusion is a real, live constraint on the frontend

Kolkata is one of the four regions wired into the DepthWizard2 frontend
(`frontend/public/data/kolkata/...`, per the project's own README/CLAUDE.md
context). Method 3 (or anything else that would reuse the
GlobalMLBuildingFootprints source for a building prior) **cannot currently
be applied to Kolkata** — it was excluded from the Sentinel semantic-prior
track for a real, visually-confirmed coverage gap (see §10). This is
independent of whether Method 3 wins overall: even if Method 3 had been
accepted, it would need a different building-footprint source (or a repaired
one) before it could touch the one live region where the current source has
a confirmed hole.

## 10. Extra finding — the "coverage gap" claim is visually real, but not uniform across the four excluded cities

Rendered `building_fraction.tif` for the four excluded cities
(kolkata/delhi/mumbai/kochi) and the six retained cities at a common
downsample. Result:

- **Mumbai**: ~75% of the frame is a hard, contiguous empty rectangle —
  building-footprint data exists only in the top ~25% of the AOI.
- **Kolkata**: a clear rectangular void covering roughly the bottom-right
  quadrant, with real building texture in the other three quadrants.
- **Delhi**: a full horizontal empty band cuts across the middle of the
  frame, plus a blank region upper-right.
- **Kochi**: only a narrow diagonal strip has any data; most of the frame is
  empty — but Kochi's currently-saved raster is the **original**
  single-partition version (see §11), not the systematically-repaired one,
  so this may partly reflect Kochi's actual estuarine geography (a narrow
  built-up spit between backwaters) rather than a pure data-coverage
  artifact. Unlike Kolkata/Delhi/Mumbai, this one hasn't been confirmed
  against the full intersecting-partition set.

Critically: the aggregate stats printed on each QC page do **not** make this
obvious by themselves — Kolkata's mean (0.2076) and p95 (0.7632) are close to
or better than several *retained* cities (Bardhaman 0.0743/0.60, Sundarbans
0/0). The "coverage gap" the note describes is a spatial/contiguity property
(large contiguous voids), not something visible in the mean/p95 numbers
alone — someone auditing only the info-panel numbers, not the images, would
not find the justification for excluding Kolkata. See `gaps-and-fixes.md` §2.

## 11. Extra finding — `repair_method3_coverage.py` was run, but only partially, and Bengaluru is currently broken

File mtimes on `building_fraction.tif` across all 10 cities split into two
clear groups:

- **Wave 1** (~03:05–03:08, 2026-09-16): darjeeling, hyderabad, jaipur,
  kochi, sundarbans — original `prepare_method3_semantic_inputs.py` run,
  hand-picked single/few partitions per city.
- **Wave 2** (~03:57–04:14, same day): kolkata, bardhaman, delhi, mumbai —
  rebuilt by `repair_method3_coverage.py`, which systematically selects
  *every* Microsoft partition whose real geographic bounds intersect the
  Sentinel AOI (not a hand-picked subset). This is the version visualized in
  §10 for kolkata/delhi/mumbai — i.e. those three cities' gaps are confirmed
  against the maximal correct partition set, not an artifact of an
  incomplete initial download.

**Bengaluru's `semantic/` directory was created at 04:14:12 — inside wave
2 — but `building_fraction.tif` does not currently exist on disk at all.**
`data/sentinel2/bengaluru/semantic/` is empty; the raw partition
(`raw/bengaluru/part-00014-....csv.gz`) is present, so the download step
finished but the rasterization step did not (or its output was later
removed). Yet `data/sentinel2/method3_semantic_QC.pdf` (generated 03:31:06 —
*before* wave 2 started) has a full, real-looking Bengaluru page (mean =
0.2414, p95 = 0.7840, >0 = 61.41%). That means a real `building_fraction.tif`
existed for Bengaluru at QC-generation time (page 5), and was later lost —
almost certainly wave 2's repair run being interrupted partway through
Bengaluru (5th in `CITY_ORDER`, right after kolkata/bardhaman/delhi/mumbai,
right before hyderabad/jaipur/kochi/darjeeling/sundarbans, all of which kept
their untouched wave-1 files).

**Practical consequence: as this repo stands right now, "Retained: ...
Bengaluru ..." cannot actually be reproduced or reused** — the QC PDF is a
historically-accurate snapshot, but the input file needed to rerun or extend
Method 3 for Bengaluru is currently missing. See `gaps-and-fixes.md` §3.
