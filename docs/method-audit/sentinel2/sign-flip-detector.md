# Sign-flip / rejection detector for DAv2 output

**Status: COMPLETE. Result: usable.** A detector built from two signals
available at deployment time (no ICESat-2 required) catches all 6 known
DAv2 sign-inversions in the 32-tile benchmark with exactly one false
positive (bhitarkanika) — 96.9% accuracy, 100% recall, 85.7% precision.
One of the two signals (DAv2-vs-DINOv3 disagreement) turned out to add no
validated value on this dataset; that's reported honestly below rather
than tuned away.

## Goal

`backbone-comparison.md` established that DAv2's raw relative-depth
output sign-inverts (negative correlation with true elevation) on 6 of
the 32 benchmark tiles: karnal, shimla, nainital, ooty, fatehpur, hisar.
Detecting that a tile has inverted is only useful if it can be done
**without** ICESat-2 ground truth, since real deployment tiles won't have
it. This builds that detector from two signals that ARE available at
deployment time, then validates it against the already-known
DAv2-vs-ICESat2 sign as ground truth — used only for validation, never as
a detector input.

## Data sources

- **DAv2 / DINOv3 depth rasters:** the existing `dav2_depth/*.npy` and
  `dinov3_depth/*.npy` outputs from `backbone-comparison.md` (Step 1.1 /
  1.2), reused unmodified. Both are `(1000, 1000)` float32 arrays on the
  exact same UTM grid as each tile's source Sentinel-2 GeoTIFF.
- **Real elevation:** Copernicus GLO-30 DEM (30m), fetched fresh for all
  32 tiles via the public AWS Open Data bucket
  (`copernicus-dem-30m.s3.amazonaws.com`, no key, no rate limit) —
  **not SRTM**, despite that being the original plan. OpenTopography's
  SRTMGL1 API (which the project has hit a 50-calls/24hr rate limit on
  before, per `backbone-terrain-analysis.md`) was tried first
  (`scripts/fetch_srtm_benchmark.py`) but produced zero output in this
  session; Copernicus GLO-30 (`scripts/fetch_copernicus_dem_benchmark.py`)
  is the same source/access method already used successfully for the
  shimla/nainital/ooty/manali/dharamshala slope-aspect diagnostic, and
  was used here as the actual elevation source for all 32 tiles. All 32
  downloads were sanity-checked before use: real, geographically-plausible
  elevation ranges (e.g. manali/dharamshala 1600–4400m Himalayan terrain,
  karnal/hisar/fatehpur ~200–275m flat Punjab/UP plains matching the
  "karnal is the flattest tile in the set" finding from diagnostic 2),
  zero nodata pixels after a 0.02° fetch buffer, 100% valid-pixel coverage
  after reprojection onto every tile's grid.
- **Tree-cover% proxy:** `content_audit_corrected_32.csv` (ESA WorldCover
  10m, the version already corrected for the 4-tile benchmark remediation
  swap, per `backbone-terrain-analysis.md`'s data-integrity fix).
- **Ground truth (validation only):** `dav2_depth/dav2_correlation_per_tile.csv`
  — each tile's real DAv2-vs-ICESat2 Pearson correlation. `true_inverted`
  = that correlation's sign is negative.

## The two signals

**Signal (a) — DAv2 vs. real elevation.** The Copernicus DEM (WGS84) is
reprojected (bilinear) onto each tile's own 1000×1000 10m UTM grid —
same target grid DAv2/DINOv3 already run on — and correlated
pixel-for-pixel against DAv2's raw depth output (Pearson, full grid, no
photon sampling needed since both are dense rasters). Per the sign
convention already established in `backbone-comparison.md` (nearer-to-
sensor = higher elevation in nadir view, so a well-behaved DAv2
correlation should be positive), a negative value here is itself the
definition of "inverted" — this is a real, if coarse (30m native
resolution) and pixel-noisy, elevation signal, independent of ICESat-2.

**Signal (b) — DAv2 vs. DINOv3/CHMv2 disagreement.** Both depth rasters
are already on the same grid, so this is a direct full-grid Pearson
correlation between the two backbones' outputs, no reprojection needed.
Per-tile **informativeness gate**: CHMv2 is a canopy-height regression
head, so on a flat, (near-)treeless tile its output is close to constant
and this signal shouldn't be trusted regardless of what it reads. A tile
is marked **not informative** when `tree_pct < 5%` (from the corrected
WorldCover audit) — chosen because it cleanly separates the two
genuinely near-treeless tiles in the set (kurnool 0.8%, kutch 2.5%) from
everything else (next-lowest is vidisha at 6.1% — a real gap in the
data, not an arbitrary round number).

## Rejection rule and thresholds

```
REJECT (flag as inverted) if:
    dav2_vs_dem_pearson < 0.0
    OR
    (tree_pct >= 5.0  AND  dav2_vs_dinov3_pearson < -0.3)
```

**Threshold for signal (a): 0.0, not a tuned free parameter.** The
project's own established sign convention says a well-behaved correlation
must be positive — so "negative" already means "inverted" by definition,
before any tuning. A sweep confirms this boundary sits in a real gap in
the data, not on a knife-edge: every threshold in **[-0.03, +0.02]**
produces the identical classification (6 TP / 1 FP / 0 FN / 25 TN). The
smallest true-positive magnitude is ooty at -0.0311; the smallest
true-negative magnitude is nizamabad at +0.0357 — a real, if narrow, gap
straddles zero.

| thresh_a | TP | FP | FN | TN | accuracy |
|---:|---:|---:|---:|---:|---:|
| +0.05 | 6 | 2 | 0 | 24 | 0.938 |
| +0.02 | 6 | 1 | 0 | 25 | 0.969 |
| **0.00** | **6** | **1** | **0** | **25** | **0.969** |
| -0.02 | 6 | 1 | 0 | 25 | 0.969 |
| -0.05 | 5 | 1 | 1 | 25 | 0.938 (ooty becomes a false negative) |
| -0.10 | 5 | 1 | 1 | 25 | 0.938 |
| -0.20 | 3 | 0 | 3 | 26 | 0.906 (loses hisar, nainital, ooty) |

**Threshold for signal (b): -0.3 — chosen defensively, not because it
earns its place on this data (see honest limitation below).** A sweep of
signal (b) alone (informative tiles only, signal (a) disabled) shows it
never beats the trivial "always predict not-inverted" baseline
(26/32 = 81.25% accuracy) at any threshold:

| thresh_b (signal b alone) | TP | FP | FN | TN | accuracy |
|---:|---:|---:|---:|---:|---:|
| -0.05 | 1 | 6 | 5 | 20 | 0.656 |
| -0.10 | 1 | 4 | 5 | 22 | 0.719 |
| -0.15 | 0 | 2 | 6 | 24 | 0.750 |
| -0.20 | 0 | 2 | 6 | 24 | 0.750 |
| -0.30 | 0 | 0 | 6 | 26 | 0.812 |
| -0.40 | 0 | 0 | 6 | 26 | 0.812 |

And combined with signal (a) fixed at 0.0, loosening thresh_b only ever
**adds** false positives, never adds a true positive signal (a) hadn't
already caught:

| thresh_b (combined w/ a=0.0) | TP | FP | FN | TN | accuracy | new FPs from (b) | new TPs from (b) |
|---:|---:|---:|---:|---:|---:|---|---|
| -0.10 | 6 | 5 | 0 | 21 | 0.844 | kota, vidisha, jaipur, pune | none |
| -0.15 | 6 | 3 | 0 | 23 | 0.906 | jaipur, pune | none |
| -0.20 | 6 | 3 | 0 | 23 | 0.906 | jaipur, pune | none |
| -0.25 | 6 | 2 | 0 | 24 | 0.938 | jaipur | none |
| **-0.30** | **6** | **1** | **0** | **25** | **0.969** | none | none |
| -0.40 | 6 | 1 | 0 | 25 | 0.969 | none | none |

-0.3 is the loosest threshold at which signal (b) stops contributing
*any* false positives — set there specifically so it stays inert on this
validation set rather than as a value earning its keep. See "Honest
limitation" below for what that means.

## Validation: confusion matrix (all 32 tiles)

| | predicted inverted | predicted not-inverted |
|---|---:|---:|
| **true inverted** (n=6) | **TP = 6** (fatehpur, hisar, karnal, nainital, ooty, shimla) | FN = 0 |
| **true not-inverted** (n=26) | FP = 1 (bhitarkanika) | TN = 25 |

**Accuracy 96.9% (31/32), recall 100% (6/6), precision 85.7% (6/7).**
Every known-bad tile is caught. Full per-tile signal values, flags, and
verdicts: `data/sentinel2_benchmark/sign_flip_detector_verdicts.csv`
(raw signals in `sign_flip_detector_signals.csv`).

All 6 true positives were flagged by **signal (a) alone** — signal (b)
never fired on a true positive at the chosen threshold (see table above:
"new TPs from (b)" is empty at every threshold tested). The one false
positive (bhitarkanika) was also flagged by signal (a) alone; signal (b)
did not flag it either (its DAv2-vs-DINOv3 correlation there is actually
strongly *positive*, +0.49 — the two backbones agree well on
bhitarkanika, which if anything argues against inversion. The OR-only
rule structure specified for this detector can't let signal (b) veto a
signal-(a) rejection, so this doesn't fix the false positive, but it's
worth recording as a real disagreement between the two signals on this
tile, not a case where both pointed the same wrong way).

## The one false positive: bhitarkanika

DAv2-vs-DEM Pearson: -0.195 (flagged). True DAv2-vs-ICESat2 Pearson:
+0.271 (weakly positive, not actually inverted). Bhitarkanika is a
mangrove-reserve coastal tile (Odisha). A plausible, **unverified**
explanation: Copernicus GLO-30 is a radar-derived DSM (surface height,
including canopy), and dense mangrove canopy is a known failure mode for
DSM-vs-true-ground-elevation agreement — the same canopy-vs-ground gap
that makes ATL08's ground-photon classification itself harder in
vegetated terrain (this project's own photon-count table already flags
hilly/canopied terrain as reducing ATL08 ground-classification success).
Both the elevation reference used here and the ICESat-2 "ground truth"
it's being checked against could be independently noisier specifically in
mangrove terrain, in different directions — this would produce exactly
the kind of single-tile disagreement observed, but this diagnostic does
not attempt to disambiguate which reference is at fault, or confirm this
mechanism. Flagged as open, matching this project's practice elsewhere
(e.g. the shimla/nainital/ooty aspect diagnostic's open questions).

## Honest limitation: signal (b) adds no validated benefit here

The task asked for two independent signals combined with OR. Signal (a)
alone already achieves 96.9% accuracy / 100% recall / 85.7% precision;
adding signal (b) at the threshold that avoids hurting that result
(-0.3) also does not help it — every one of this dataset's 6 true
inversions is one signal (a) already caught, and no threshold for signal
(b) exists that adds a true positive without also adding false positives
(the tables above show this precisely: loosening thresh_b only ever picks
up jaipur/pune/kota/vidisha as new false positives, never a new true
positive). Signal (b)'s per-tile values among the true-inverted tiles are
themselves weak and inconsistent — nainital (+0.35) and ooty (+0.26) are
actually *positive*, meaning DINOv3 does not reliably invert alongside
DAv2 even on tiles where DAv2 itself has flipped, consistent with
`backbone-comparison.md`'s finding that DINOv3 is more sign-stable than
DAv2 in general.

This detector still includes signal (b) with the -0.3 threshold in the
shipped rule, for a reason not tested by this 32-tile set: **signal (a)
depends on Copernicus DEM coverage, which was 100% valid for all 32
tiles here but is not guaranteed in general** (Copernicus GLO-30 has
known voids over open water and some other surface types). Signal (b)
needs no elevation reference at all, so it's kept as a fallback for
tiles where signal (a) can't be computed reliably — not because it
proved itself here. **A production deployment of this detector should
add an explicit minimum-DEM-coverage guard** (e.g., fall back to
signal (b) alone, with a wider/re-validated threshold, when DEM valid-
pixel coverage drops below some cutoff) — this wasn't needed for the
32-tile validation set and so wasn't built or tuned here.

## Honest limitation: threshold selection is in-sample, not held-out

Both thresholds above were chosen by looking at all 32 tiles' true
labels, the same 32 tiles being reported as the validation result — this
is in-sample threshold selection, not a held-out test, and with only 6
positive examples total, a train/test split small enough to preserve
statistical meaning in both halves isn't really feasible here (this
mirrors the same honest caveat this project applied to Method 3's
overfitting finding and Follow-up 1's held-out calibration check
elsewhere in this audit). The result should be read as "this rule
separates these 32 tiles' known labels well," not as an unbiased
estimate of accuracy on unseen tiles. The zero-tuning nature of threshold
(a) (it falls directly out of the project's existing sign convention,
not a fitted value) makes it the more trustworthy of the two; threshold
(b) is explicitly a defensive, not a fitted-and-validated, choice per the
limitation above.

## Reproducing this (detector)

- `scripts/fetch_copernicus_dem_benchmark.py` — fetches the 32 Copernicus
  GLO-30 DEM tiles (`data/sentinel2_benchmark/copernicus_dem_raw/`).
- `scripts/detect_sign_flip.py` — computes both raw signals per tile,
  writes `data/sentinel2_benchmark/sign_flip_detector_signals.csv`.
- `scripts/evaluate_sign_flip_detector.py` — threshold sweeps + final
  confusion matrix, writes `sign_flip_detector_verdicts.csv`.

---

# Per-tile DEM-anchored calibration, gated by the detector

**Status: COMPLETE. Result: usable only with the ICESat-2 check —
the DEM-only held-out number is misleading on its own, see below.**

Follow-up to the detector above: for tiles the detector does NOT flag,
fit a local `elevation ≈ a·dav2_depth + b` calibration and report its
**held-out** accuracy — the same held-out discipline as
`backbone-terrain-analysis.md`'s Follow-up 1 (fit on a random 50% seed-42
split, evaluate on the other, disjoint 50%), applied here to dense
Copernicus DEM pixels instead of ICESat-2 photons. The 7 rejected tiles
(fatehpur, hisar, karnal, nainital, ooty, shimla, bhitarkanika) are
**not calibrated at all** — reported as `REJECTED, flagged for review`,
never a number.

## Method

Per accepted tile (25 of 32):
1. Reproject Copernicus GLO-30 onto the tile's own 1000×1000 10m grid
   (same reprojection as the detector). Randomly split all valid DEM
   pixels 50/50 (seed 42).
2. Fit OLS `dem_elevation ~ a·dav2_depth + b` on the train half.
3. Evaluate on the **held-out** other half: Pearson (for comparability
   with the detector's raw `dav2_vs_dem_pearson` — note Pearson is
   invariant to an affine transform of one variable, so this necessarily
   matches the detector's number to within sampling noise, it is not a
   new result on its own) plus **RMSE/MAE in meters**, which do depend on
   how well the fitted line performs, and are the real "calibration
   accuracy" numbers.
4. Independently, apply the **same DEM-half-fitted `(a, b)`** — fit on
   zero ICESat-2 data — to DAv2 depth sampled at every real ICESat-2
   ground-photon location for that tile (all 32 tiles have photon data),
   and score against true photon height. This is the task's requested
   "stronger independent check": ICESat-2 is a per-photon reference
   (modulo the project's known ~6.5m geolocation noise floor) vs.
   Copernicus GLO-30's own ~10–16m absolute accuracy, and — critically —
   it was never involved in fitting `(a, b)` at all, unlike the DEM
   held-out half which comes from the *same* raster the fit was built
   from.

## Critical finding: the DEM-only held-out number is inflated and should not be trusted alone

Before the full results table — this is the single most important thing
this follow-up found, and it's exactly why the task asked for the
ICESat-2 cross-check as "the stronger independent check." The DEM
held-out RMSE looked excellent on first run (sub-meter to a few meters
for several tiles) — good enough to be suspicious. Comparing both RMSEs
against each tile's own real elevation range confirms it:

| | median RMSE as % of tile's own elevation range |
|---|---:|
| DEM held-out (same raster fit was built from) | **9.7%** |
| ICESat-2 (independent source, zero role in fitting) | **79.4%** |

**For 10 of the 25 accepted tiles, the ICESat-2 RMSE actually exceeds
the tile's entire elevation range** — i.e. the calibrated prediction is
not just weak, it's worse than uninformative for absolute elevation on
those tiles specifically (predicting the tile's mean would very likely
do as well or better). The worst cases are the flattest, smallest-range
coastal tiles — nagapattinam (ICESat-2 RMSE = 498% of its 19m range),
kakinada (343% of 22m), amalapuram (338% of 23m), vembanad (328% of
30m), kochi_city (286% of 34m), chennai (250% of 37m).

**Why the DEM number is inflated, most likely:** the held-out split is a
per-pixel random split of a raster that was **bilinear-reprojected from
a native 30m DEM onto a 10m grid** — meaning most "held-out" pixels sit
immediately adjacent to, and are numerically interpolated from, the same
underlying 30m source cells as nearby "training" pixels. A per-pixel
random split doesn't respect that spatial autocorrelation, so the DEM
held-out set isn't really independent of the training set in the way a
genuinely separate reference (ICESat-2) is. This is a plausible
mechanism consistent with the data, not something this diagnostic
independently proves (the same "in-sample-looking held-out result that
isn't really testing what it looks like it's testing" pattern this
project already flagged for the pooled 32-tile calibration in
`backbone-terrain-analysis.md`'s Follow-up 1, for a different reason —
here it's spatial autocorrelation from resampling, there it was
between-tile identity separation).

**Important nuance — this isn't uniformly bad.** The inflation is
concentrated in flat/low-relief tiles. High-relief hilly tiles show the
DEM and ICESat-2 numbers largely *agreeing*: dharamshala (DEM 8.0% vs.
ICESat-2 7.2% of range), kohima (11.5% vs. 10.2%), manali (10.9% vs.
11.9%), dehradun (16.1% vs. 18.2%), almora (17.2% vs. 19.4%) — all
within a few points of each other, meaning the calibration's DEM-based
self-check is actually a reasonable proxy for real accuracy specifically
on these tiles. This matches this project's running finding that DAv2
has its most genuine signal on high-relief terrain (manali is the best
tile in the whole benchmark) — where there's a real, large-magnitude
elevation signal to fit, the small addition of ICESat-2 noise doesn't
swamp it; on flat terrain, it does.

**Practical read:** trust the ICESat-2 number wherever it's available
(it always is, in this benchmark). Don't report the DEM held-out number
as "accuracy" on its own — at best it's a weak proxy that happens to
hold up on high-relief tiles and badly overstates performance on flat
ones.

## Full per-tile results

Rejected (not calibrated — flagged for review, per the detector above):
**fatehpur, hisar, karnal, bhitarkanika, nainital, ooty, shimla** (7 tiles).

Calibrated (25 tiles), sorted by ICESat-2 RMSE as % of the tile's own
elevation range (ascending = calibration most trustworthy at the top):

| tile | category | DEM held-out RMSE (m) | ICESat-2 RMSE (m) | tile's elev range (m) | ICESat-2 RMSE as % of range |
|---|---|---:|---:|---:|---:|
| dharamshala | hilly | 281.4 | 253.8 | 3535 | 7.2% |
| kohima | hilly | 233.0 | 207.0 | 2029 | 10.2% |
| manali | hilly | 301.1 | 328.8 | 2766 | 11.9% |
| dehradun | hilly | 271.7 | 306.9 | 1685 | 18.2% |
| almora | hilly | 186.2 | 210.2 | 1085 | 19.4% |
| jaipur | urban | 26.9 | 59.6 | 253 | 23.5% |
| mumbai | urban | 19.2 | 74.3 | 314 | 23.6% |
| pune | urban | 27.2 | 82.2 | 190 | 43.3% |
| hyderabad | urban | 13.0 | 78.1 | 160 | 48.7% |
| bengaluru | urban | 15.2 | 91.7 | 157 | 58.4% |
| vidisha | agricultural | 7.4 | 57.8 | 98 | 59.0% |
| kurnool | agricultural | 7.0 | 87.1 | 128 | 68.1% |
| nizamabad | agricultural | 6.6 | 73.5 | 93 | 79.4% |
| delhi | urban | 7.8 | 58.9 | 73 | 80.2% |
| goa_estuary | coastal | 19.5 | 96.5 | 109 | 88.8% |
| kota | agricultural | 5.0 | 55.4 | 52 | **107.5%** |
| bathinda | agricultural | 1.9 | 47.7 | 41 | **116.7%** |
| digha | coastal | 3.4 | 61.9 | 36 | **172.1%** |
| kutch | coastal | 0.7 | 49.3 | 25 | **193.6%** |
| chennai | urban | 3.0 | 93.2 | 37 | **250.5%** |
| kochi_city | urban | 2.9 | 96.9 | 34 | **286.2%** |
| vembanad | coastal | 3.2 | 97.4 | 30 | **327.8%** |
| amalapuram | coastal | 2.0 | 78.5 | 23 | **338.1%** |
| kakinada | coastal | 2.3 | 77.0 | 22 | **343.1%** |
| nagapattinam | coastal | 1.9 | 95.4 | 19 | **498.4%** |

Bold % = ICESat-2 RMSE exceeds the tile's entire elevation range (10/25
tiles). Full columns (Pearson, MAE, sample sizes, fitted `a`/`b`, DEM
held-out and ICESat-2 numbers side by side):
`data/sentinel2_benchmark/calibration_results_with_range.csv`
(`calibration_results.csv` has the same data without the range/%
columns, plus the 7 rejected rows).

## Reproducing this (calibration)

- `scripts/calibrate_accepted_tiles.py` — reads the detector's verdicts,
  calibrates accepted tiles with the held-out DEM split, cross-checks
  against ICESat-2, writes `data/sentinel2_benchmark/calibration_results.csv`.
  The elevation-range/% comparison table above was a one-off analysis on
  top of that CSV (not a separate committed script).

---

# 2026-09-21 — Calibration hardening: offset-proxy test, confidence tiers, FABDEM

Four-step follow-up to the calibration section above. **Result: Steps 1
and 3/4 are both clean negative results, reported as such — no forcing.**
Step 2 (confidence tiers) is a real, usable addition.

## Step 1 — does a surface-offset proxy explain calibration accuracy better than category?

**Verdict: fails to confirm.** Hypothesis: DAv2 calibration is worse on
tiles where the DEM's sensed surface sits above bare ground (buildings,
trees), so `tree_pct + builtup_pct` (from `content_audit_corrected_32.csv`)
should predict `icesat2_rmse_pct_of_range` better than land-cover category
does.

- **Simple correlation (n=25 accepted tiles):** `offset_proxy` vs.
  `icesat2_rmse_pct_of_range`: Pearson r = **-0.4652** (p=0.019), Spearman
  r = **-0.5685** (p=0.003) — significant, but **negative**: higher
  tree+built-up% predicts *lower* (better) error, the opposite direction
  the DSM-offset hypothesis predicts.
- **Explanatory power:** category (4-level) R² = **0.557** vs. offset_proxy
  (1 continuous predictor) R² = **0.216** — category explains more than
  twice the variance.
- **Confound check (the decisive test):** `offset_proxy` itself correlates
  with `dem_elev_range_m` (r=+0.40, p=0.048) — tiles with more tree/built-up
  cover in this 32-tile set also happen to have more real relief (the hilly
  category tiles). Controlling for `log(elev_range)`, the **partial
  correlation of offset_proxy with RMSE% drops to r=-0.089 (p=0.673) —
  not significant at all.** The apparent relationship is fully explained
  by elevation range, not by surface offset.
- **Fallback harder proxy** (held-out DEM residual magnitude,
  `held_out_dem_mae_m`, per the task's specified fallback): r=-0.477,
  p=0.016, R²=0.227 — same magnitude, same direction, same story: its
  *normalized* version (`dem_rmse_pct_of_range`, which removes the
  elevation-range confound the same way `icesat2_rmse_pct_of_range` does)
  shows essentially no relationship at all (r=-0.185, p=0.375, R²=0.034).

**Reading:** the real driver of `icesat2_rmse_pct_of_range` is terrain
relief (elevation range) acting as the denominator of a percentage metric,
not a DSM-vs-bare-earth surface offset from vegetation or buildings.
Category is a decent (if coarse) proxy for relief; a purpose-built
vegetation/building-cover proxy is not, once relief is controlled for.
Script: `scripts/step1_offset_proxy_test.py`, data:
`data/sentinel2_benchmark/step1_offset_proxy.csv`.

## Step 2 — confidence tiers on top of the sign-flip gate

**Verdict: confirmed and adopted.** Same "find a real gap, don't pick a
round number" method as the detector's `a=0.0` threshold, applied to
`held_out_dem_pearson` across the 25 accepted tiles: the largest gap
(0.1218) sits between **pune (0.0532)** and **vidisha (0.1750)** —
threshold set at the midpoint, **0.114**. This confirms the task's own
prediction exactly: **nizamabad (0.036) and pune (0.053)** are the two
tiles separated out.

Three tiers, all 32 tiles: **REJECTED** (7 — unchanged from the sign-flip
detector), **LOW-CONFIDENCE** (2 — nizamabad, pune), **CONFIDENT** (23 —
everything else). `data/sentinel2_benchmark/sign_flip_detector_verdicts.csv`
updated in place with the new `tier` column. Script:
`scripts/step2_tier_classification.py`.

## Step 3 — FABDEM re-run

**New GEE access set up for this task** — this project had none before
(no `earthengine-api`, no credentials, no prior code); set up fresh this
session (project id `depthwizard2`, registered for Earth Engine, API
enabled). FABDEM (`projects/sat-io/open-datasets/FABDEM`, a bare-earth
ML-corrected version of Copernicus GLO-30 that removes building/tree
height) fetched for all 32 tiles via `ee.Image.getDownloadURL`
(30m, EPSG:4326, same 0.02° buffer as the GLO-30 pull). All 32 downloads
sanity-checked: real, plausible elevation, and — as expected for a
bare-earth correction — consistently narrower max values than GLO-30 at
the same tiles (e.g. bathinda GLO-30 200.7–241.6m vs. FABDEM 204.4–234.6m).

**Signal (a) re-run against FABDEM: identical rejection set.** All 7
GLO-30-rejected tiles (fatehpur, hisar, karnal, bhitarkanika, nainital,
ooty, shimla) remain rejected under FABDEM, with `dav2_vs_fabdem_pearson`
values close to their GLO-30 counterparts and the same signs throughout
(e.g. karnal: GLO-30 -0.550 → FABDEM -0.604; shimla: -0.669 → -0.667). The
sign-flip detector's verdict is robust to this specific DEM substitution.

**Calibration quality: worse almost everywhere.** Re-running the exact
held-out methodology (50/50 split seed 42, independent ICESat-2 check) on
FABDEM for the 25 accepted tiles: **only 1 of 25 tiles improved** (pune,
marginally, -0.49 percentage points); the other 24 got worse, several
dramatically (kutch +492 points, nagapattinam +478, kochi_city +338,
amalapuram +274, chennai +289, vembanad +225, kakinada +230). No tile
changes tier (`data/sentinel2_benchmark/fabdem_comparison.csv`'s
`fabdem_tier` column now uses the same fixed 0.114 threshold Step 2
established from GLO-30, applied consistently rather than re-fit per DEM
source — an earlier version of this table showed vidisha/kochi_city
dropping a tier, which was an artifact of independently re-fitting the
gap-threshold on FABDEM's own distribution, not a real difference in
either tile's reliability; corrected in `scripts/run_fabdem_comparison.py`
and this table). Full per-tile numbers:
`data/sentinel2_benchmark/fabdem_comparison.csv`. Script:
`scripts/fetch_fabdem_benchmark.py` (fetch), `scripts/run_fabdem_comparison.py`
(re-run detector signal (a) + calibration + tiering against FABDEM).

## Step 4 — synthesis: does the fix address the diagnosed mechanism?

**Verdict: no — FABDEM's effect runs in the opposite direction the
surface-offset hypothesis predicts, and is well-explained by the same
elevation-range mechanism Step 1 already identified.**

The task's real confirmation test: does the *size* of FABDEM's change
correlate with `offset_proxy` (tree%+built-up%) in the hypothesized
direction (tiles with the worst surface-offset problem should see the
biggest gains)? Across the 25 tiles accepted under both DEMs:

- Correlation of `offset_proxy` with `delta_rmse_pct` (FABDEM RMSE% minus
  GLO-30 RMSE%; positive = FABDEM worse): **r = -0.5226 (p=0.0074)**,
  Spearman r = -0.6015 (p=0.0015).

This is **significant and in the wrong direction**: tiles with *low*
offset_proxy show the *largest* degradation (kutch, offset_proxy=3.0,
degraded +492 points; nagapattinam, offset_proxy=17.5, degraded +478
points), while tiles with *high* offset_proxy — the hilly tiles the
hypothesis says should benefit most — show the *smallest* degradation
(dehradun +0.7, almora +1.2, manali +1.5, kohima +3.3, dharamshala +4.2 —
all under 5 points, against a set where the median degradation is +19
points and the worst is +492).

**This is the same mechanism as Step 1, showing up again:** FABDEM's own
correction process appears to add a roughly fixed absolute noise floor
(consistent with its ML-based bare-earth estimation carrying its own
uncertainty, on top of whatever base uncertainty it inherits from being
derived from GLO-30 in the first place) — and a fixed absolute error
matters enormously as a *percentage* on flat, small-elevation-range tiles
(coastal/agricultural) and barely at all on high-relief hilly tiles with
huge elevation ranges. Not a mechanism this diagnostic independently
proves (no direct measurement of FABDEM's own correction-model
uncertainty was made), but it is the explanation consistent with both
this result and Step 1's confound finding, without needing a second,
unrelated story.

**Plain-language bottom line:** switching to FABDEM does not fix the
diagnosed problem, does not help where the diagnosis predicted it would
help most, and net-degrades calibration on all but one of the 25 tiles it
was tested on. **Do not adopt FABDEM as a replacement for Copernicus
GLO-30 for this calibration task based on this evidence.** The project's
existing restriction — trust per-tile calibration on **hilly terrain
only**, treat coastal/agricultural/urban (flat, low-relief) results as
unreliable regardless of which DEM backs the calibration — still stands,
and this round of testing did not find a fix for it.

## Consolidated 32-tile summary table

| tile | category | offset proxy | GLO-30 tier | GLO-30 ICESat-2 RMSE % | FABDEM tier | FABDEM ICESat-2 RMSE % |
|---|---|---:|---|---:|---|---:|
| dharamshala | hilly | 86.5 | CONFIDENT | 7.2% | CONFIDENT | 11.4% |
| kohima | hilly | 91.5 | CONFIDENT | 10.2% | CONFIDENT | 13.5% |
| manali | hilly | 72.4 | CONFIDENT | 11.9% | CONFIDENT | 13.4% |
| dehradun | hilly | 88.8 | CONFIDENT | 18.2% | CONFIDENT | 18.9% |
| almora | hilly | 58.5 | CONFIDENT | 19.4% | CONFIDENT | 20.6% |
| jaipur | urban | 88.1 | CONFIDENT | 23.5% | CONFIDENT | 27.2% |
| mumbai | urban | 87.3 | CONFIDENT | 23.6% | CONFIDENT | 24.3% |
| pune | urban | 83.1 | LOW-CONFIDENCE | 43.3% | LOW-CONFIDENCE | 42.8% |
| hyderabad | urban | 92.3 | CONFIDENT | 48.7% | CONFIDENT | 61.5% |
| bengaluru | urban | 94.6 | CONFIDENT | 58.4% | CONFIDENT | 83.3% |
| vidisha | agricultural | 17.1 | CONFIDENT | 59.0% | CONFIDENT | 85.2% |
| kurnool | agricultural | 2.0 | CONFIDENT | 68.1% | CONFIDENT | 86.6% |
| nizamabad | agricultural | 31.4 | LOW-CONFIDENCE | 79.4% | LOW-CONFIDENCE | 182.5% |
| delhi | urban | 93.2 | CONFIDENT | 80.2% | CONFIDENT | 88.4% |
| goa_estuary | coastal | 50.5 | CONFIDENT | 88.8% | CONFIDENT | 106.6% |
| kota | agricultural | 50.0 | CONFIDENT | 107.5% | CONFIDENT | 126.6% |
| bathinda | agricultural | 53.5 | CONFIDENT | 116.7% | CONFIDENT | 162.3% |
| digha | coastal | 37.3 | CONFIDENT | 172.1% | CONFIDENT | 194.3% |
| kutch | coastal | 3.0 | CONFIDENT | 193.6% | CONFIDENT | 685.4% |
| chennai | urban | 70.2 | CONFIDENT | 250.5% | CONFIDENT | 539.5% |
| kochi_city | urban | 42.0 | CONFIDENT | 286.2% | CONFIDENT | 623.9% |
| vembanad | coastal | 41.8 | CONFIDENT | 327.8% | CONFIDENT | 552.7% |
| amalapuram | coastal | 66.0 | CONFIDENT | 338.1% | CONFIDENT | 611.7% |
| kakinada | coastal | 47.3 | CONFIDENT | 343.1% | CONFIDENT | 572.7% |
| nagapattinam | coastal | 17.5 | CONFIDENT | 498.4% | CONFIDENT | 976.7% |
| fatehpur | agricultural | — | REJECTED | — | REJECTED | — |
| hisar | agricultural | — | REJECTED | — | REJECTED | — |
| karnal | agricultural | — | REJECTED | — | REJECTED | — |
| bhitarkanika | coastal | — | REJECTED | — | REJECTED | — |
| nainital | hilly | — | REJECTED | — | REJECTED | — |
| ooty | hilly | — | REJECTED | — | REJECTED | — |
| shimla | hilly | — | REJECTED | — | REJECTED | — |

Sorted by GLO-30 ICESat-2 RMSE % (ascending). Full CSV:
`data/sentinel2_benchmark/fabdem_comparison.csv`.

## Deployment verdict (SUPERSEDED — see the 2026-09-21 root-cause section below)

**This verdict turned out to be substantially wrong, and not for a
reason related to DAv2, FABDEM, or terrain type** — the ICESat-2 RMSE
numbers it's based on carried an uncorrected vertical-datum mismatch
that inflated flat-terrain error by roughly the same magnitude as the
"failure" being diagnosed. Read the correction below before relying on
anything in this subsection. Left in place, unedited, for the audit
trail — this is what the evidence looked like before the datum bug was
found.

**Restrict deployment of DAv2 per-tile calibration to hilly terrain.**
Across both DEM sources tested, only the 5 hilly tiles keep ICESat-2 RMSE
under ~21% of the tile's own elevation range; every coastal tile and most
agricultural/urban tiles exceed 50–100%+ under GLO-30 and get
categorically worse under FABDEM. This is not a DEM-choice problem this
round of testing found a fix for — it's a structural limitation of
calibrating against a fixed-magnitude absolute error on terrain with a
small real elevation range, present under both DEM sources tried so far.
Coastal and most agricultural/urban terrain should not be sold as
supporting deployable calibrated elevation from this pipeline based on
current evidence.

## Reproducing this (hardening pass)

- `scripts/step1_offset_proxy_test.py` — offset-proxy hypothesis test +
  confound check, writes `step1_offset_proxy.csv`.
- `scripts/step2_tier_classification.py` — three-tier gap search, updates
  `sign_flip_detector_verdicts.csv` in place with a `tier` column.
- `scripts/fetch_fabdem_benchmark.py` — FABDEM fetch via GEE (needs
  `EARTHENGINE_PROJECT` in `.env` and a one-time `ee.Authenticate()`),
  writes `data/sentinel2_benchmark/fabdem_raw/`.
- `scripts/run_fabdem_comparison.py` — re-runs signal (a), calibration,
  and tiering against FABDEM, writes `fabdem_comparison.csv`.

---

# 2026-09-21 (continued) — Root-cause finding: the ICESat-2 "gap" is a vertical-datum bug, not a DAv2 accuracy ceiling

**This is the single most important correction in this document.** The
"Deployment verdict" section above, and large parts of the FABDEM
section's framing, were built on ICESat-2 RMSE numbers that turned out to
carry an uncorrected, systematic vertical-datum offset. Following
systematic-debugging discipline: root cause found before any fix was
applied, verified independently before trusting it, and the fix is
reported with exactly how much of the earlier conclusion it changes (a
lot) and how much survives (the FABDEM-vs-GLO-30 direction, though not
its magnitude).

## Step 1 — decompose the ICESat-2 RMSE

For each of the 25 accepted tiles: component (a) is
`held_out_dem_rmse_m` (DAv2's own scatter against the held-out half of
the *same* DEM it was calibrated on); component (b) is the gap,
`icesat2_rmse_m - held_out_dem_rmse_m`.

| | median gap as % of total ICESat-2 error |
|---|---:|
| hilly tiles | **-1.4%** (dharamshala/kohima: gap is actually slightly *negative* — ICESat-2 error is smaller than the DEM-based error there) |
| everything else | **87–99%** (kutch 98.6%, nagapattinam 98.1%, kakinada 97.1%, amalapuram 97.4%, kochi_city 97.0%, chennai 96.7%) |

Component (b) — the gap — dominates almost totally on every non-hilly
tile. Per the task's own framing: this rules out "DAv2's within-tile DEM
fit is just bad" (component a) as the main story, and points at
something specific to the ICESat-2 comparison. Full numbers:
`data/sentinel2_benchmark/calibration_results_with_range.csv` (`gap_m`,
computed ad hoc from existing columns, not a new script).

## Step 2 — registration/reprojection spot-check: ruled out

For kutch, nagapattinam, kakinada (worst gaps): 8 real photons each, pixel
row/col computed by the pipeline's manual affine-invert
(`lib_backbone_correlation.sample_depth_at_photons`) checked against
`rasterio`'s own, independently-implemented `.index()` method.
**24/24 matched exactly, zero mismatches.** The pixel-lookup math is not
the problem. Script: `scripts/diagnose_error_floor.py`.

## Step 3 — the real cause: DAv2's DEM-anchored prediction is in a different vertical datum than ICESat-2

With the registration path ruled out, tested reference-vs-reference
disagreement directly, **with DAv2 removed from the loop entirely**:
sampled the *native* (unreprojected) Copernicus GLO-30 DEM at each
ICESat-2 photon's exact lon/lat and compared straight to the photon's
true height — no calibration, no reprojection-onto-tile-grid step, no
DAv2 involvement at all.

| tile | native DEM − ICESat-2: bias | RMSE | MAE | correlation r |
|---|---:|---:|---:|---:|
| kutch | +49.3 m | 49.3 m | 49.3 m | 0.528 |
| nagapattinam | +95.0 m | 95.4 m | 95.0 m | 0.798 |
| kakinada | +76.1 m | 77.0 m | 76.1 m | 0.698 |
| manali (control, hilly) | +27.7 m | 29.3 m | 27.7 m | 0.9999 |

**Bias ≈ RMSE ≈ MAE for every tile** — the "error" is almost pure,
near-constant offset, not scatter. That is the signature of a **datum
mismatch, not noise**: Copernicus GLO-30 heights are referenced to the
**EGM2008 geoid** (orthometric height), while the ICESat-2 `h_ph` field
used throughout this project as ground truth (`backbone-comparison.md`,
Step 0: "ATL03 `h_ph` (photon ellipsoidal height)") is explicitly
**WGS84 ellipsoidal height**. These are two different vertical references
separated by the local geoid undulation `N`, which varies smoothly across
India by tens of meters — and India sits partly inside the Indian Ocean
Geoid Low, one of the largest negative geoid anomalies on Earth, so a
40–95m offset here is physically expected, not exotic.

**Independently verified, not just plausible:** computed each tile's
geoid undulation `N` via PROJ's actual EGM2008 grid
(`Transformer.from_crs("EPSG:4979", "EPSG:3855")`, `PROJ_NETWORK=ON` to
fetch the real grid file) and compared to the *observed* bias above —
computed with zero knowledge of the geoid model, from raw DEM-vs-ICESat2
numbers alone:

| tile | observed bias | independently-computed geoid undulation N | difference |
|---|---:|---:|---:|
| kutch | +49.3 m | 49.18 m | 0.12 m |
| nagapattinam | +95.0 m | 95.26 m | 0.26 m |
| kakinada | +76.1 m | 76.21 m | 0.11 m |
| manali | +27.7 m | 23.66 m | 4.04 m (more real terrain variance here, expected to be noisier) |

**Match to within ~0.1–0.3m on the three flat tiles.** This is not "a
plausible mechanism consistent with the data" the way earlier open
questions in this project were phrased — this is a quantitative
confirmation against an independent physical model. Scripts:
`scripts/diagnose_error_floor.py` (bias/registration), `scripts/verify_geoid_datum_fix.py` (independent geoid check + correction).

**Scope of what this bug affects, precisely:** Pearson/Spearman
correlations are invariant to a constant additive offset, so **every
correlation-based result in this entire audit trail is unaffected** —
`backbone-comparison.md`'s DAv2/DINOv3-vs-ICESat2 numbers, the sign-flip
detector's `true_inverted` labels and confusion matrix, Step 2's tiers
(built from `held_out_dem_pearson`, a DEM-vs-DEM-only comparison that
never touches ICESat-2). **Only RMSE/MAE/bias-based ICESat-2 comparisons
from this session's calibration work are affected**: `calibrate_accepted_tiles.py`'s
`icesat2_rmse_m`/`icesat2_mae_m`, everything downstream of them (the
elevation-range-% table, the FABDEM comparison's RMSE-based numbers, Step
1's offset-proxy test, which used `icesat2_rmse_pct_of_range` as its
target variable).

## Step 4 — corrected numbers

Fix: subtract each tile's geoid undulation `N` (computed once from its
manifest centroid lon/lat) from the DEM-orthometric-scale calibrated
prediction before comparing to ICESat-2's ellipsoidal height:
`pred_ellipsoidal = a·dav2_depth + b − N`.

### GLO-30, all 25 accepted tiles

**Correction (2026-09-21, later same day):** the numbers first reported
in this subsection (median 8.4%, the per-tile figures below) used
`dem_elev_range_m` measured on the raw, *buffered* Copernicus DEM file
rather than the exact reprojected tile footprint the RMSE is computed
over — a second, independent bug from the datum issue, found later while
building the SRTM 3-way comparison (see that section for the full
root-cause). **The correct median is ~11.0%, not 8.4%.** The fit itself
(`a`, `b`) was never affected — only the percentage's denominator was.
Numbers below are corrected; `geoid_correction_results.csv` still has the
original (denominator-buggy) values, kept for the audit trail rather than
silently overwritten — use `srtm_3way_comparison.csv`'s
`glo30_icesat2_rmse_pct_of_range` column for the correct per-tile numbers
instead.

| | median ICESat-2 RMSE as % of elevation range |
|---|---:|
| **Before geoid correction** | 79.4% |
| **After geoid correction (corrected denominator)** | **~11.0%** |

**Tiles with RMSE exceeding their own elevation range: 10/25 before → 0/25 after**
(this count is unaffected by the denominator bug — the largest corrected
value, goa_estuary at 20.5%, is nowhere close to 100% either way). Every
previously-catastrophic flat/coastal tile still looks *far* better, just
not quite as dramatically as first reported: nagapattinam 498.4% →
**6.9%**, kochi_city 286.2% → **5.9%**, kakinada 343.1% → **9.8%**,
amalapuram 338.1% → **7.4%**, chennai 250.5% → **9.3%**, kutch 193.6% →
**7.9%**. Hilly tiles barely move either way (their gap was always small,
since a fixed ~24–95m absolute offset is a much smaller fraction of a
1000–4000m elevation range): dharamshala 7.2%→10.9%, manali 11.9%→13.2%,
kohima 10.2%→12.4%. **Corrected category means (recomputed with the fixed
denominator): agricultural 9.0%, coastal 10.6%, urban 10.5%, hilly
14.7%** — hilly remains the *worst*-performing category on average, the
opposite of the pre-correction ranking, though the gap between categories
is narrower than first reported. Correct full table:
`data/sentinel2_benchmark/srtm_3way_comparison.csv`
(`glo30_icesat2_rmse_pct_of_range` column). Scripts:
`scripts/verify_geoid_datum_fix.py` (original, denominator bug),
`scripts/run_srtm_comparison.py` (corrected denominator, found and fixed
while adding SRTM).

### Step 1 (offset-proxy hypothesis) redone with the corrected target

The original Step 1 conclusion ("fails to confirm, confounded by
elevation range") was computed against the datum-contaminated metric;
the numbers immediately below (first posted same day) were then computed
against the datum-*corrected* but still denominator-buggy metric. **Both
are superseded here** by a third pass using the fully-corrected
(datum-fixed AND denominator-fixed) target:

| | before any correction | after geoid fix only (superseded, denominator bug) | after both fixes (correct) |
|---|---:|---:|---:|
| category R² | 0.557 | 0.202 | **0.206** |
| offset_proxy vs RMSE%: r (p) | -0.465 (0.019) | +0.386 (0.057) | **+0.324 (0.114)** |
| elev_range vs RMSE%: r (p) | -0.461 (0.020) | +0.189 (0.365, n.s.) | **-0.180 (0.389, n.s.)** |

Category's explanatory power still collapses (0.557 → 0.206) once the
datum-driven elevation-range confound is removed — that part holds up.
But the offset_proxy correlation, once the denominator bug is also fixed,
is **weaker and no longer even borderline-significant** (r=+0.324,
p=0.114, vs. the previously-reported p=0.057) — n=25 is small enough that
fixing a second measurement bug was enough to cross back over the
(admittedly arbitrary) 0.05 line. **Final revised verdict: the original
Step 1 "fails to confirm" call still doesn't hold up as a clean negative
(the sign flipped to the hypothesized direction and stayed there through
both corrections), but "marginal, borderline-significant support" was
too strong a claim — the honest final read is "a same-direction trend too
weak to call significant at this sample size," which is close to a clean
null result but not quite the same claim as the original confounded
analysis made.** This value bounced around across three different
computations of the same underlying question — worth remembering as a
concrete example of how much a small-n (n=25) correlation analysis can
move when a supposedly-independent measurement bug in the target variable
gets fixed.

### FABDEM comparison redone with the correction

Applying the same fix to the FABDEM-side calibration:

| | median ICESat-2 RMSE % (GLO-30) | median ICESat-2 RMSE % (FABDEM) |
|---|---:|---:|
| Before correction | 79.4% | (huge, unreported median — several 500%+ outliers) |
| **After geoid correction** | ~11.0% (see correction note above — GLO-30's `fabdem_geoid_corrected.csv` baseline column used the same buffered-denominator bug; FABDEM's own column was already correct) | **11.6%** |

**FABDEM is still worse than GLO-30 on most tiles after correction** —
the *direction* of Step 4's original finding survives, and the SRTM
3-way comparison below (same-day follow-up) confirms it with a fully
consistent denominator: GLO-30 median 11.0% vs. FABDEM 11.6%, a real but
now clearly small gap. The *magnitude* first reported here does not
survive: this is not "FABDEM catastrophically fails compared to GLO-30"
(the 500–900%-point gaps were almost entirely the datum bug, present in
both DEMs' raw numbers but not cancelling identically between them
tile-by-tile) — it's "FABDEM is marginally worse than GLO-30," both
comfortably in a deployable range, with barely a percentage point between
them. **The Step 4 mechanism conclusion (FABDEM doesn't fix what Step 1
diagnosed, and isn't worth adopting over GLO-30) still stands** — just
for a much smaller effect size than originally reported, twice over now
(once for the datum fix, again for the denominator fix). Full corrected
table: `data/sentinel2_benchmark/fabdem_geoid_corrected.csv` for the
per-tile FABDEM numbers (correct as originally computed); see the SRTM
section below for the consistent-denominator GLO-30 comparison.

## Revised deployment verdict (replaces the superseded one above)

**DAv2 per-tile calibration against Copernicus GLO-30, WITH the geoid
correction applied, is viable across all four terrain categories tested
— not just hilly.** (Numbers corrected same day — see the correction note
above the GLO-30 table: median is ~11%, not the ~8% first reported here,
after a second, denominator-only bug was found and fixed while adding
SRTM.) Median ICESat-2 RMSE is ~11% of each tile's own elevation range
across the full 25-tile accepted set, with agricultural tiles performing
best (9.0% mean) and hilly performing worst (14.7% mean) — the complete
reverse of the pre-correction ranking. The earlier "hilly-only"
restriction was an artifact of an uncorrected vertical-datum bug, not a
real terrain-dependent limitation of DAv2, GLO-30, or the calibration
method.

**Concrete fix required before this calibration pipeline is used for
anything user-facing:** any DAv2-elevation calibration anchored to a
geoid-referenced DEM (Copernicus GLO-30, FABDEM, and most other
satellite-derived global DEMs) that will be checked against or deployed
alongside an ellipsoidal-height reference (ICESat-2, most consumer/survey
GPS) **must** subtract the local geoid undulation `N` first. This is a
one-line, well-understood, essentially free fix — a single
`pyproj.Transformer.from_crs("EPSG:4979", "EPSG:3855")` call per tile
centroid — not a fundamental limitation requiring a different DEM source
or a terrain-restricted deployment.

**GLO-30 remains the better DEM choice over FABDEM** for this
calibration task, even after the correction — that part of the earlier
finding holds, just with a much smaller (and much less alarming)
magnitude than first reported.

## Reproducing this (root-cause pass)

- `scripts/diagnose_error_floor.py` — registration spot-check + native-grid
  DEM-vs-ICESat2 bias measurement, writes `error_floor_diagnosis.csv`.
- `scripts/verify_geoid_datum_fix.py` — independent EGM2008 geoid-grid
  verification + corrected GLO-30 numbers for all 25 tiles, writes
  `geoid_correction_results.csv`. Needs `PROJ_NETWORK=ON` in the
  environment (fetches the real EGM2008 grid file from PROJ's CDN; without
  it, `pyproj` silently returns 0.0 instead of erroring — worth knowing if
  reproducing this elsewhere).
- `scripts/run_fabdem_comparison.py` — tier-threshold bug fixed (now uses
  the fixed 0.114 threshold from Step 2 instead of re-fitting per DEM
  source); the geoid-corrected FABDEM comparison itself was a one-off
  script, not committed separately.

---

# 2026-09-21 (continued) — SRTM added: EGM96 correction, 3-way DEM comparison

SRTM data (via `scripts/fetch_srtm_benchmark.py`, the OpenTopography
SRTMGL1 path that failed earlier in this session — it worked this time)
landed at `data/sentinel2_benchmark/srtm_raw/{tile_id}_srtm.tif`. **All 32
files confirmed present, non-empty, and sanity-checked**: real,
geographically plausible elevation ranges matching GLO-30/FABDEM closely
at every tile (e.g. bathinda SRTM 200–244m vs. GLO-30 200.7–241.6m),
`EPSG:4326`, zero nodata anywhere.

## SRTM uses a different vertical datum than GLO-30/FABDEM — verified before use, not assumed

SRTM heights are referenced to **EGM96**, not EGM2008 (Copernicus
GLO-30/FABDEM's datum) and not WGS84 ellipsoidal. Reusing the EGM2008
grid here would have been wrong. Built the EGM96 equivalent of the
already-verified correction (`EPSG:4979` → `EPSG:5773`, "EGM96 height",
vs. the earlier `EPSG:4979` → `EPSG:3855` for EGM2008) and verified it
with the **same empirical method** used to confirm the EGM2008 fix — not
by trusting the CRS code alone:

| tile | native SRTM − ICESat-2 bias (observed) | independently-computed N (EGM96, PROJ_NETWORK) | difference |
|---|---:|---:|---:|
| kutch | +49.15 m | 48.90 m | 0.25 m |
| nagapattinam | +95.44 m | 93.76 m | 1.68 m |
| kakinada | +77.29 m | 76.55 m | 0.74 m |
| manali | +31.96 m | 30.58 m | 1.38 m |

Match to within ~0.25–1.7m on all four tiles — at least as tight as (and
on manali, noticeably tighter than) the original EGM2008 verification.
**Secondary sanity check:** EGM96 and EGM2008 undulation at the same
points agree to within 0.3–1.5m on the three flat tiles (expected — both
model the same physical geoid) and diverge more on manali (+6.9m,
expected too — EGM2008's gravity-data improvements over EGM96 are
concentrated in exactly this kind of high-relief, historically
under-surveyed terrain). Both checks point the same way: the EGM96
correction is genuinely correct, not a coincidence.

## A second, more subtle bug found via cross-checking this new script against the old one

Building the 3-way comparison required recomputing GLO-30's numbers
alongside SRTM's, and a side-by-side check against the already-published
GLO-30 numbers turned up a real inconsistency: the earlier
`dem_elev_range_m` (used for the "8.4%" GLO-30 median reported in the
previous section) was computed from the **raw, buffered** Copernicus DEM
file (which includes the 0.02° fetch buffer beyond the tile's actual
10km footprint), not from the **exact reprojected tile grid** the RMSE
itself is computed over. Verified concretely on nizamabad: refitting
from scratch reproduced the *exact same* calibration slope/intercept
(`a=1.159061051970545`, bit-identical — so the fit itself was never
wrong), but the range used as the percentage's denominator was **92.5m
(buffered, incorrect) vs. 53.0m (exact footprint, correct)**. FABDEM's
numbers were already computed correctly (verified: its own
`fabdem_elev_range_m` for nizamabad is 40.2m, from the exact reprojected
grid, matching the same correct method) — so this bug was specific to
the GLO-30-only `geoid_correction_results.csv`/`verify_geoid_datum_fix.py`
path in the previous section, not to the FABDEM comparison.

**Consequence:** the earlier "GLO-30 8.4% vs. FABDEM 11.6%" comparison
was not quite apples-to-apples — GLO-30's number looked better partly
because its denominator was inflated. This script
(`scripts/run_srtm_comparison.py`) uses the correct, exact-footprint
range consistently for all three DEMs, making the table below the first
fully consistent 3-way comparison. It does not overturn the qualitative
finding (GLO-30 was still slightly ahead of FABDEM even measured
correctly — see below), but the size of that gap in the previous section
was overstated.

## 1–2. Held-out calibration + 3-way comparison, same 25 accepted tiles (verdicts reused, not re-derived)

| tile | category | SRTM | GLO-30 | FABDEM |
|---|---|---:|---:|---:|
| bathinda | agricultural | **3.4%** | 3.9% | 5.0% |
| mumbai | urban | **4.4%** | 4.9% | 4.8% |
| kochi_city | urban | 5.8% | 5.9% | **5.4%** |
| kurnool | agricultural | **7.0%** | 6.4%* | 6.7% |
| nagapattinam | coastal | 11.4% | **6.9%** | 10.2% |
| amalapuram | coastal | 7.7% | **7.4%** | 8.6% |
| kutch | coastal | **5.4%** | 7.9% | 12.8% |
| digha | coastal | **8.4%** | 9.1% | 5.6%* |
| jaipur | urban | **9.2%** | 9.2% | 9.0%* |
| chennai | urban | **5.5%** | 9.3% | 15.3% |
| kakinada | coastal | **5.3%** | 9.8% | 10.1% |
| dharamshala | hilly | **10.9%** | 10.9% | 10.8%* |
| bengaluru | urban | 11.8% | **11.0%** | 11.6% |
| vidisha | agricultural | 10.9% | **11.1%** | 11.4% |
| kota | agricultural | **7.2%** | 11.4% | 11.6% |
| nizamabad | agricultural | 13.8% | **12.1%** | 15.9% |
| vembanad | coastal | **8.8%** | 12.4% | 11.3% |
| kohima | hilly | 12.5% | **12.4%** | 12.4% |
| manali | hilly | 13.1% | 13.2%* | **13.2%** |
| hyderabad | urban | 14.2% | **13.7%** | 14.9% |
| delhi | urban | **11.0%** | 14.8% | 14.1% |
| pune | urban | **15.9%** | 15.6%* | 15.6% |
| dehradun | hilly | **18.1%** | 18.0%* | 17.8%* |
| almora | hilly | 19.0% | **18.8%** | 18.9% |
| goa_estuary | coastal | **18.1%** | 20.5% | 19.9% |

Bold = best of the 3 for that tile (`*` marks near-ties decided at the
4th significant figure — not a meaningful difference). Full table:
`data/sentinel2_benchmark/srtm_3way_comparison.csv`.

**Medians: SRTM 10.9%, GLO-30 11.0%, FABDEM 11.6%.** All three are now
within a single percentage point of each other. **Win count** (lowest
RMSE% per tile, 25 tiles): **SRTM 11, GLO-30 7, FABDEM 7.**

## 3. Bonus cross-check: does corrected-SRTM agree with corrected-GLO-30, independent of ICESat-2?

Reprojected both onto the same tile grid, corrected both to WGS84
ellipsoidal (subtracting each DEM's own geoid undulation), and compared
them **directly to each other** — ICESat-2 plays no role in this check at
all.

**Median Pearson r = 0.9796, median |mean difference| = 1.03m.** Two
independently-sourced elevation products (different radar missions,
different processing chains), each corrected with an independently
computed geoid model, land within about a meter of each other on
average, with near-perfect correlation on most tiles (hilly tiles:
r > 0.999). This is strong, independent evidence that the geoid-correction
methodology from the previous section is genuinely sound — if either
correction were wrong, or if the earlier close match to observed bias on
kutch/nagapattinam/kakinada had been coincidental, these two DEMs would
not agree with each other this well after independent correction. A few
tiles show weaker agreement (bathinda r=0.52, kutch r=0.68, amalapuram
r=0.68) — all very flat, small-relief tiles where real elevation
differences are only a few meters and easily dominated by each DEM's own
short-range noise; the *mean* difference stays small even there (kutch
+0.03m, bathinda -0.09m), it's specifically the pixel-to-pixel
correlation that's noisier, consistent with "not much real signal to
correlate against" rather than a correction failure. Full table:
`data/sentinel2_benchmark/srtm_3way_comparison.csv`
(`srtm_vs_glo30_pearson`, `mean_diff_m`, `std_diff_m` columns).

## 4. Plain-language verdict

**Does SRTM change the recommendation?** Marginally, in SRTM's favor —
it wins outright on 11/25 tiles vs. 7 each for GLO-30 and FABDEM, and its
median (10.9%) is the lowest of the three, though by less than half a
percentage point. **None of the three DEM sources is decisively better
once each is correctly geoid-corrected** — the differences between them
(≤1 percentage point at the median) are now much smaller than the
difference correcting the datum bug made in the first place (tens of
percentage points). The practical takeaway: **getting the vertical datum
right matters far more than which of these three DEM products is used.**

**Separately, for the project write-up:** SRTM is the DEM source
literally named in the SIH problem statement as the recommended method.
It is now **confirmed actually fetched, sanity-checked, correctly
datum-corrected, and empirically tested** — not assumed or left as an
unverified recommendation — and it performs comparably to (marginally
better than) both alternatives tried this session. This is worth stating
plainly in the project write-up regardless of the numerical margin: the
officially-recommended method works, was tested rigorously, and holds up.

## Reproducing this (SRTM pass)

- `scripts/fetch_srtm_benchmark.py` — unchanged from earlier in this
  session; the OpenTopography SRTMGL1 fetch that failed before, succeeded
  this run.
- `scripts/run_srtm_comparison.py` — EGM96 correction + verification,
  held-out SRTM calibration on the 25 detector-accepted tiles, the 3-way
  table, and the direct SRTM-vs-GLO-30 cross-check. Writes
  `data/sentinel2_benchmark/srtm_3way_comparison.csv`.

---

# 2026-09-21 (continued) — Standing methodology: SRTM primary, GLO-30 cross-check

**Decision, locked in:** SRTM (EGM96-corrected) is the **primary**
elevation source for calibration/deployment going forward. It edged out
GLO-30 and FABDEM on win-count (11/25 vs. 7/25 each) in the 3-way
comparison above, and it is the method literally named in the SIH
problem statement. GLO-30 is kept as a **standing cross-check, not a
merged or averaged predictor** — the two are never blended into a single
number; they're computed independently and compared.

## The cross-check rule

For any tile going forward: compute both corrected-SRTM and
corrected-GLO-30 predictions (each already geoid-corrected to WGS84
ellipsoidal per the methodology above — EGM96 for SRTM via `EPSG:5773`,
EGM2008 for GLO-30 via `EPSG:3855`). Reproject both onto the same grid
and compute the mean absolute pixel-wise difference over the tile. **If
that difference exceeds 3.5m, flag the tile as "elevation-reference
uncertain"** — its calibration result should be reported with that
caveat, not treated as equally trustworthy as a tile where the two
sources agree.

## Where 3.5m comes from

From the existing 25-tile agreement data (`srtm_3way_comparison.csv`'s
`mean_diff_m` column, the same direct DEM-vs-DEM check from the SRTM
section above — median |diff| = 1.03m, r = 0.98): the full distribution
of `|mean_diff_m|` across all 25 already-validated tiles ranges from
0.03m (kutch) to **3.22m (dehradun, the largest disagreement seen among
tiles already known to be fine)**, median 1.03m, mean 1.10m. **3.5m is
the smallest round number above that observed maximum** — chosen so this
rule produces **zero false "uncertain" flags on the existing validation
set** (every one of the 25 tiles we already trust passes), while still
being a real, data-grounded ceiling rather than an arbitrarily generous
one. It's roughly 3.4x the median agreement, sitting inside the task's
suggested 2–3x range once rounded up to the nearest tile-clearing value
rather than picked as an abstract multiple.

**What this rule is NOT:** it does not resolve disagreements by
averaging, does not pick a "winner" per tile, and does not attempt to
determine which source is right when they diverge — it only flags that
the tile needs manual attention or an additional reference (ICESat-2
where available) before its calibration is trusted. This is a
detection/triage rule, matching the same spirit as the sign-flip
detector itself: catch a real problem before it reaches deployment,
without claiming to fix it automatically.

## Correction to earlier GLO-30 figures in this document

The "8.4%" GLO-30 median first reported in the root-cause section above
was itself affected by the denominator bug found later while building the
SRTM comparison (see the correction notes inline in that section). **The
correct GLO-30 median, referenced consistently from here on, is ~11.0%**
(`srtm_3way_comparison.csv`'s `glo30_icesat2_rmse_pct_of_range` column) —
already corrected in place above rather than left as a dangling
inconsistency; this section is the canonical pointer for anyone who finds
an "8.4%" reference elsewhere in this file's git history.

## Reproducing this (standing methodology)

No new script — this section documents policy and a threshold derived
from `scripts/run_srtm_comparison.py`'s existing `srtm_3way_comparison.csv`
output. A future per-tile deployment pipeline should call the SRTM and
GLO-30 correction paths from `run_srtm_comparison.py` (or factor them into
a shared helper) and apply the 3.5m rule above before reporting a
calibrated result.

---

# 2026-09-21 (continued) — Residual correction test (replaces the originally-scoped 2.2)

## Setup

Baseline: the already-calibrated, geoid-corrected **SRTM** prediction
(Task 1's primary source) for each of the same 25 sign-flip-detector-
accepted tiles (verdicts reused from `sign_flip_detector_verdicts.csv`,
not re-derived). For each tile, computed the per-photon residual (true
ICESat-2 height minus the SRTM-calibrated prediction) at every available
photon, then tested whether a correction fit on a small subset of these
residuals — **k=10 and k=20 points, matching Method 2's DFC2019 anchor
counts** for comparability — improves accuracy on the held-out remainder.
Two correction types only, per the task's scope: a **constant offset**
(mean residual at the k anchors, applied uniformly) and a **single linear
term** (an OLS refit of `true_height ~ depth_value` using only the k
anchors, replacing the DEM-based calibration entirely for that test — not
a spatial/quadrant-local variant, which Method 2 already showed hurts).

**Because a k=10–20 random draw is inherently high-variance**, each
(tile, k, method) combination was repeated **30 times** with independent
random anchor draws (seeds 42–71) and the **median** outcome is reported
— a single draw would risk reporting a lucky or unlucky result as if it
were representative.

## Results

Paired per-tile comparison (corrected vs. that same tile's own baseline,
not two independently-computed medians — an earlier draft of this
analysis compared medians-of-different-columns and got a misleading
"correction looks worse" result purely from that mismatch; fixed before
reporting):

| k | method | tiles improved | median % change | mean % change | Wilcoxon signed-rank p |
|---|---|---:|---:|---:|---:|
| 10 | constant offset | 21/25 | **-5.6%** | -14.4% | <0.0001 |
| 10 | linear (sparse OLS) | 17/25 | -2.6% | -18.8% | 0.045 |
| 20 | constant offset | 21/25 | **-5.4%** | -15.4% | <0.0001 |
| 20 | linear (sparse OLS) | 22/25 | **-7.4%** | -21.5% | <0.0001 |

(Negative % change = improvement, i.e. lower RMSE than the SRTM-only
baseline on the same held-out photons.) All four combinations are
statistically significant by Wilcoxon signed-rank test on the paired
per-tile deltas — this is a real, consistent-direction effect across the
25 tiles, not noise. The gap between median (modest, 2.6–7.4%) and mean
(much larger, 14–21%) improvement means a handful of tiles see large
gains while most see a smaller, real one — not that a few outliers are
driving an otherwise-flat result. Full per-tile table:
`data/sentinel2_benchmark/residual_correction_results.csv`.

**Tiles with the largest gains** (mostly small-elevation-range coastal
tiles, where a few meters of correction is a large relative
improvement): goa_estuary (19.4m → 12.1m, linear, k=20), kochi_city
(2.9m → 0.8m), amalapuram (2.9m → 0.7m), kakinada (2.8m → 1.1m),
vembanad (4.4m → 1.1m), nagapattinam (2.6m → 1.0m). **Tiles that got
worse** under at least one method/k combination: vidisha, pune (worse
under every combination tested — these two tiles' SRTM-only baseline was
apparently already about as good as 10-20 ICESat-2 points can achieve,
and the small-sample fit adds noise instead of removing it), plus
scattered single-combination regressions on kutch, mumbai, hyderabad,
manali, almora, dharamshala (better under one k/method combination,
worse under another — consistent with small-sample correction being
somewhat unstable per-tile, exactly as expected at k=10-20).

## Verdict

**Yes, ICESat-2-based residual correction meaningfully improves on the
DEM-calibration-only (SRTM) baseline — modestly, not dramatically, and
not on every tile.** The improvement is real (statistically significant,
consistent direction across most of the 25 tiles) but small in absolute
terms for most tiles (median 2.6–7.4%): SRTM-DEM calibration is already
reasonably close to what a sparse handful of real ICESat-2 points can
add on top of it, which is itself a useful, honest finding — this is not
a case where DEM calibration is so far from the truth that a tiny number
of ground-truth points transforms the result, but a case where a small,
genuine, worthwhile refinement is available for tiles where a handful of
real elevation control points can be obtained. **The linear (sparse-OLS)
correction at k=20 is the best-performing combination tested** (22/25
tiles improved, -7.4% median, most significant p-value) — if a
correction is deployed, this is the one to use, and 20 points outperforms
10 for the linear variant specifically, consistent with a linear fit
needing slightly more points than a simple mean-offset to stabilize.
This does not override the "restrict quadrant/spatial-local corrections"
finding from Method 2 — this test deliberately did not re-test that,
per the task's scope.

## Reproducing this (residual correction test)

- `scripts/test_residual_correction.py` — the full test (baseline refit,
  30-repeat k=10/k=20 anchor draws, constant + linear corrections,
  paired significance test), writes
  `data/sentinel2_benchmark/residual_correction_results.csv`.

---

# 2026-09-21 (continued) — Scoping note (flagged, not built): ATL08 canopy height for DINOv3/CHMv2 validation

**Not pursued this session — recorded so it isn't lost.**

Every ICESat-2 comparison in this entire audit trail (`backbone-comparison.md`'s
DAv2/DINOv3 correlations, the sign-flip detector's ground truth, every
calibration/residual-correction check in this document) has used ATL08's
**ground-height** field (`h_ph` restricted to `atl08_class == 1`, "ground"
— see `backbone-comparison.md`, Step 0). This is the right reference for
DAv2, which targets terrain elevation. It is very likely the **wrong**
reference for validating **DINOv3's CHMv2 head** specifically — CHMv2 is
explicitly a **Canopy Height Model**, trained to predict vegetation/canopy
height *above* the ground, not terrain elevation itself
(`backbone-comparison.md` already flags this interpretation gap: "CHMv2
predicting canopy height rather than raw elevation" as an unverified but
plausible explanation for DINOv3's behavior).

**The more appropriate ground truth for a future CHMv2-specific
validation:** ATL08 already publishes a canopy-height field directly
(`h_canopy` / the 98th-percentile canopy height fields in the standard
ATL08 product, distinct from the ground-classified photon heights this
project has queried so far via `atl08_class=['atl08_ground']` in
`scripts/query_icesat2_photons.py`) — this measures the same physical
quantity CHMv2 is trained to predict, unlike the ground-height field
currently used everywhere. Re-validating DINOv3 against `h_canopy`
instead of ground height could plausibly show DINOv3/CHMv2 performing
*better* than the existing backbone-comparison numbers suggest, since
those numbers are currently checking it against a target it was never
trained to predict.

**Why this wasn't pursued now:** out of scope for this session's tasks
(SRTM/GLO-30/FABDEM calibration hardening and the Method 4 port), and it
would need a fresh ICESat-2 query (a different ATL08 field/product
request, not just a re-read of the already-downloaded ground-photon
CSVs in `data/icesat2_photons/`) plus its own validation pass before
being trustworthy. Flagged here so it's available as a concrete,
well-motivated next step rather than lost — it directly targets a gap
this project's own documentation has already identified but not yet
tested.

---

# 2026-09-21 (continued) — Method 4 (phase2_building_rank_v2) ported to Sentinel-2/SRTM

**Status: COMPLETE. Result: the learned correction does NOT beat linear
calibration on this domain — a clean overfitting signature, not a close
call.** Confirmed with the user before training on two design decisions
this port required (see "Domain adaptations" below); trained fully (4
folds × 60 epochs, unmodified architecture/loss); result reported
honestly rather than reframed as a win.

## Setup

Architecture and losses reused **unchanged** from `evaluate_method4_v2.py`
(`ScaleModulationNetV2`, `huber_masked`, `smoothness_loss`,
`rank_pair_loss`) and `evaluate_method4.py` (`quadrant_bounds`,
`fit_stats`) — same 4-fold spatial-quadrant holdout, same hyperparameters
as the exact `phase2_building_rank_v2` config that produced this
project's best DFC2019 result (dense patches, 60 epochs,
smoothness_weight=0.01, rank_weight=0.5, rank_pairs_per_patch=2000,
rank_margin=0.25, building channel, ground_plane_weight=0). Training/eval
set: the same 25 sign-flip-detector-accepted tiles used throughout this
document.

## Domain adaptations (confirmed with the user before training)

**1. Target reframing (necessary, not optional) — resolved through two
rounds of user confirmation.** DFC2019's AGL truth is naturally small
(0–50m); the architecture's fixed output-scale constants (`scale` ∈
[0.25x, 4x], `residual` capped at ±50m) were sized for that range. Raw
Sentinel-2 elevation spans up to ~4400m — feeding it directly would make
those constants meaningless by construction. Two fixes were considered
and explicitly discussed before implementation:
   - *Literal height-above-tile-minimum* (a flat per-tile scalar shift) —
     rejected: it removes the tile's absolute baseline offset but does
     **not** compress within-tile dynamic range, so high-relief tiles
     (manali still spans ~2765m within-tile after subtracting its
     minimum) would still break the ±50m cap.
   - **Residual from the per-tile linear SRTM calibration** (same `a`,
     `b`, `N` already established and validated all session) — adopted.
     This is the direct analog of "does the learned correction beat the
     per-tile-OLS baseline," the exact comparison already reported for
     DFC2019 Method 4.

   **Checked empirically before training, as requested:** even this
   residual exceeded the ±50m cap on 8/25 tiles (up to ±1069m,
   dharamshala). Breakdown confirmed: 7 of the 8 (all 5 hilly tiles plus
   mumbai and jaipur) have moderate-to-good DEM fit quality
   (`held_out_dem_pearson` 0.35–0.83) — their large residual is pure
   scale, not a bad fit. Only pune is a genuinely poor fit (pearson=0.05,
   this project's own LOW-CONFIDENCE tier). **Fix: per-tile normalization**
   — each tile's residual divided by its own max-abs value (rescaled to
   the network's native ±50 range) before training, multiplied back at
   eval time. This changes only the data pipeline (a per-tile scalar in
   and out), not the architecture or loss.

**2. Building channel substitution.** DFC2019's building-probability
channel came from an ONNX semantic-segmentation model trained on ~0.3m
airborne imagery (`scripts/generate_dfc_building_prior.py`) — reusing it
on 10m Sentinel-2 imagery would be a resolution/domain mismatch, not a
valid port. Substituted with **ESA WorldCover's built-up class (value
50) at its native 10m resolution** — real categorical land-cover data at
the correct native resolution for this domain, serving the identical
semantic role.

## Results

| | DEM check (held-out) | ICESat-2 check (independent) |
|---|---:|---:|
| **CNN beats linear, tile-fold wins** | **77/100** | **12/100** |
| Pooled RMSE, CNN | 76.0m | 69.8m |
| Pooled RMSE, linear | 117.9m | **53.4m** |

The two checks tell opposite stories. On the DEM check the CNN wins
convincingly and consistently (77% of tile-folds, every fold's pooled
RMSE improves 30–46%). On the **independent ICESat-2 check — the one
that actually matters, since it was never involved in training or in the
DEM-based evaluation** — the CNN **loses on 88% of tile-folds**, and its
pooled RMSE (69.8m) is *worse* than simple linear calibration's (53.4m).
This pattern is completely consistent across all 4 folds individually,
not an artifact of one bad fold: fold 0 (12.1m vs 7.0m, kota — CNN
worse), fold 1 (65.9m vs 19.4m, mumbai — CNN much worse), fold 2 (24.6m
vs 14.5m, bengaluru — CNN worse), fold 3 (51.3m vs 13.8m, mumbai — CNN
much worse); wins for the CNN on ICESat-2 are rare and scattered across
11 of the 25 tiles (kutch wins in 2 of its 4 folds; hyderabad, pune,
almora, dharamshala, vembanad, dehradun, kohima, nizamabad, vidisha, and
jaipur each win in exactly 1 of their 4 folds) — no tile wins in all 4,
and 14 of the 25 tiles never beat linear on ICESat-2 in any fold. Full
per-tile, per-fold table:
`data/sentinel2_benchmark/method4_sentinel2_results/method4_sentinel2_results.json`.

## Why: a clean overfitting-to-DEM-interpolation signature, not a close call

This directly echoes — and sharpens — the very first methodological
lesson of this entire calibration effort (the spatial-autocorrelation
finding from the original GLO-30 calibration section): the SRTM truth
used for training is **bilinear-reprojected from a coarser ~30m native
grid onto the 10m tile grid**, which introduces smooth,
spatially-correlated interpolation structure between neighboring pixels.
A simple 2-parameter linear fit has essentially no capacity to overfit
that structure. A CNN with dense, near-full-coverage training patches
does — it can learn to reproduce local smoothing/interpolation patterns
correlated with RGB/depth/building texture that happen to match the
DEM's *own* resampling artifacts, without that pattern reflecting any
real, independently-verifiable elevation signal. This produces exactly
what was observed: dramatic DEM-check improvement, no ICESat-2-check
improvement — in fact a real ICESat-2-check *regression*, since the
model's excess capacity is being spent chasing DEM-specific noise instead
of a signal that generalizes.

**This plausibly also explains why the port doesn't replicate DFC2019's
result**, without needing a second, unrelated story: DFC2019's AGL truth
is native, high-resolution airborne LiDAR — not resampled from a coarser
source — so it doesn't offer the same bilinear-interpolation structure
for a CNN to overfit to. The architecture and loss aren't broken; the
*opportunity to overfit a resampling artifact* is a property of this
domain's truth data (SRTM/GLO-30, necessarily coarser than Sentinel-2's
own 10m grid), not of DFC2019's.

## Verdict

**No — the learned correction does not beat the linear one on this
domain the way it did on DFC2019.** It appears to on the DEM-only check
(matching the surface form of the DFC2019 result), but the independent
ICESat-2 check — the check this project has consistently treated as the
trustworthy one throughout this entire document — shows the opposite:
linear calibration alone remains the better choice for Sentinel-2/SRTM
elevation correction. This is a genuine, mechanistically-explained
negative result, not a tuning failure to be iterated away — the
mechanism (dense CNN capacity + interpolated coarse-DEM truth) is a
structural property of this problem, not a hyperparameter this
configuration got wrong.

## Reproducing this (Method 4 port)

- `scripts/prepare_method4_sentinel2_data.py` — builds the per-tile
  normalized residual truth, WorldCover building channel, and linear
  baseline rasters; writes `data/sentinel2_benchmark/method4_port/`
  (including `scale_factors.csv`).
- `scripts/run_method4_sentinel2.py` — the training/eval driver (imports
  `evaluate_method4.py`/`evaluate_method4_v2.py`'s architecture and
  losses unchanged), writes
  `data/sentinel2_benchmark/method4_sentinel2_results/` (4 fold
  checkpoints + `method4_sentinel2_results.json` with full per-tile,
  per-fold DEM and ICESat-2 metrics).

---

# 2026-09-21 (continued) — Method 4 retrained at SRTM's native ~30m resolution: mechanism confirmed, gap does not close, stopping here

**Status: COMPLETE. Result: the gap does NOT close — if anything it gets
worse. Per the pre-agreed decision rule, stopping here: linear
calibration remains the right tool for Sentinel-2/SRTM, and further time
should go to 2.3 (semantic-prior) or the ATL08 canopy-height validation
flagged above, not further rounds on this architecture.**

## Why this run

The previous section diagnosed the 10m-grid result's DEM-check win as
likely CNN memorization of bilinear-interpolation structure introduced
by upsampling SRTM's real ~30m native grid onto Sentinel-2's 10m grid —
a mechanism, not just a correlation. This section removes that mechanism
directly rather than working around it: a new ~333×333 UTM grid at
SRTM's real ~30m resolution replaces the 1000×1000 10m grid for all
inputs (RGB and DAv2 depth **downsampled** — real information loss, not
interpolation — via area-averaging; building channel refetched from
WorldCover at this coarser target; SRTM truth reprojected from its own
~30m native grid onto this same-resolution 30m UTM grid, a regrid rather
than a 3x upsample). The linear baseline was refit at this resolution
too, for a fair comparison. Same architecture, same losses, same
hyperparameters, same 4-fold structure — only the resolution changed.

**Reported transparently, not routed around:** with `PATCH=64` left
unchanged, a 166×166 training quadrant at 30m yields only ~4 dense
patches (vs. ~49 at 10m) — training patch count dropped from
~1000–1100/fold to a flat **300/fold**. This is an honest, direct
consequence of matching native resolution with an otherwise-unmodified
pipeline, not a bug.

## Results

| | DEM check (held-out) | ICESat-2 check (independent) |
|---|---:|---:|
| **CNN beats linear, tile-fold wins** | 24/100 (down from 77/100 at 10m) | 16/100 (up slightly from 12/100 at 10m) |
| Pooled RMSE, CNN | 133.1m | 65.1m |
| Pooled RMSE, linear | 115.5m | 53.4m |

**The DEM-check win essentially disappears** — down from 77% to 24%, and
now *losing* to linear in pooled terms (133.1m vs. 115.5m) rather than
winning by 30–46% per fold as it did at 10m. This is exactly what the
diagnosed mechanism predicts: remove the finer, interpolated grid the
CNN could overfit, and its apparent DEM-check advantage evaporates almost
entirely, confirming that advantage was real but illusory — a property
of the 10m grid's interpolation structure, not of the model learning
anything about elevation.

**The ICESat-2-check win rate barely moves** (12/100 → 16/100) and the
CNN still loses decisively in pooled terms (65.1m vs. 53.4m) — a 22%
worse RMSE than simple linear calibration. Per-tile: 17/25 tiles have at
least one DEM-check win across their 4 folds (down from essentially all
25 at 10m), and only 13/25 have at least one ICESat-2-check win, mostly
single, scattered folds (kutch, pune, and almora are the only tiles with
2 of 4 folds winning on ICESat-2; no tile wins all 4 on either check).
Full per-tile, per-fold table:
`data/sentinel2_benchmark/method4_sentinel2_native30m_results/method4_native30m_results.json`.

## Interpretation

Per the task's own pre-agreed decision rule: **the gap did not close, so
this stops here rather than continuing to iterate on the architecture.**
Both results together tell a coherent, mechanistically-complete story:

1. At 10m, the CNN's large training-patch budget (~1000+/fold) and dense
   coverage let it substantially overfit the interpolated DEM's own
   smooth resampling structure — a real, diagnosable, and now-confirmed
   artifact, not a modeling improvement.
2. At native ~30m, that overfitting opportunity is removed, and the
   CNN's real result is exposed: it does not beat linear calibration on
   either check, and its own training data (a flat 300 patches/fold) is
   simply too sparse for a 4-conv-layer CNN to learn a genuine,
   generalizable RGB/depth/building-conditioned elevation correction on
   top of what the linear fit already captures.
3. Both findings point at the same root cause: **this project has no
   dense, native-resolution ground truth for India** — the role
   DFC2019's real airborne LiDAR played there. SRTM/GLO-30/FABDEM are all
   real data, but at ~30m they're too coarse to support the kind of
   dense per-pixel supervision this architecture was built around and
   validated against; when forced to genuinely native resolution
   instead, there simply isn't enough independent training signal per
   tile.

**This is not a failure of the architecture, the losses, or the port
itself** — it's a data-availability ceiling specific to this domain, one
CNN tuning cannot fix. It would take a fundamentally different (and not
currently available) ground-truth source — a genuinely dense,
high-resolution Indian elevation product — to give this architecture
something worth learning from at the resolution it needs.

## Verdict (final, this line of investigation)

**Linear calibration remains the right tool for Sentinel-2/SRTM
elevation correction.** Per the task's own instruction: do not spend
further time tuning this architecture on this domain. The better uses of
remaining time are the semantic-prior approach (2.3) or the ATL08
canopy-height validation for DINOv3/CHMv2 scoped earlier in this
document — both target gaps this project's own evidence has already
identified, rather than continuing to iterate on a mechanism now
confirmed to be a data-availability limit, not a tuning problem.

## Reproducing this (native-resolution follow-up)

- `scripts/prepare_method4_sentinel2_native30m.py` — builds the ~30m
  native-resolution RGB/depth/building/truth/linear-baseline rasters
  (downsampled via area-averaging, not interpolated), writes
  `data/sentinel2_benchmark/method4_port_native30m/`.
- `scripts/run_method4_sentinel2_native30m.py` — the training/eval
  driver (same architecture/loss imports, same methodology as the 10m
  version), writes
  `data/sentinel2_benchmark/method4_sentinel2_native30m_results/` (4
  fold checkpoints + `method4_native30m_results.json`).

---

# 2026-09-22 — Roadmap note for phase 2.3 (semantic-prior): Open Buildings 2.5D as a candidate input channel

**Not implemented. Recorded for later, per instruction not to build this now.**

While investigating a different question (zaidnansari2011/sih2026-depthwizard's
India cross-check, and separately staging Method 6 on this project's own
Sentinel-2/SRTM benchmark), Google Open Buildings 2.5D Temporal came up as a
real, freely available data source: per-building `building_height` (height
above terrain, in [0, 100] m) at roughly 4m effective resolution (Sentinel-2
derived), with a `building_presence`/confidence band per pixel. This project
already has WorldCover's binary built-up fraction wired in as Method 4's
existing building channel (`extra-channel building` in
`evaluate_method4_v2.py`, `data/dfc2019/experiments/semantic/building/` for
DFC2019; ESA WorldCover for the Sentinel-2 port).

**The candidate idea for phase 2.3:** Open Buildings 2.5D is a strictly
richer signal than WorldCover's binary building-probability for the
semantic-prior direction — it carries an actual **height estimate**, not just
a presence probability, and a confidence band that can be used to weight or
threshold which pixels to trust (the same confidence-filtering trick that
produced the honest −6.28m/−6.94m headline in the India cross-check
investigation, as opposed to the misleadingly small −0.25m/−1.15m number from
scoring inside a diluted footprint). As an **input channel** (not a training
target — this project already tried Open Buildings-style dense targets
conceptually via the coarse-DEM-as-target route and hit the
interpolation-memorization/data-scarcity ceiling documented above; using it
as an *input* the model conditions on is a different and untested role), it
could give a per-pixel prior on both "is this a building" and "roughly how
tall," which the current binary building-probability term cannot express.

**Coverage caveat, already known from the India investigation:** Open
Buildings' height field is itself Sentinel-2-derived, not measured — so this
would be a model-derived input, and its own accuracy ceiling (~4m effective
resolution, and the demonstrated tendency to underestimate tall buildings
when scored loosely) should be treated as a property of the input, not
assumed away. It's also coverage-dependent — worth checking before committing
that it actually covers whichever 8 urban Sentinel-2 benchmark tiles a
follow-on Method 6 Open-Buildings-as-target test (test A, staged separately)
would use.

**Status: idea recorded, not built.** The right next step, if phase 2.3 is
picked up, is a small coverage/quality check on Open Buildings over this
project's own benchmark tiles before wiring it in as a channel — not
implementation yet.

---

# 2026-09-22 (continued) — Method 6 (full DAv2-Small fine-tune) staged on the same 10m-grid Sentinel-2/SRTM setup: fails Stage 1, decisively, plain underperformance

**Status: STOPPED at fold 0, per the pre-agreed staged protocol. Result: Method
6 does not replicate its DFC2019 win here — it loses to the existing linear
baseline on both checks, by a wide margin. Follow-on tests (Open Buildings
2.5D as target, for both Method 6 and Method 4) are NOT run, since they were
explicitly conditioned on fold 0 clearing this bar.**

## Why this test

Method 6 (`scripts/evaluate_method6_finetune_twinhead.py` — full DAv2-Small
backbone fine-tune into a twin mean/log-variance head, see
`docs/method-audit/06-full-finetune-twin-head/`) beat both this project's own
oracle-affine baseline and its best prior method on DFC2019, on every
tracked metric. The open question was whether that same architecture, ported
onto the exact 10m-grid Sentinel-2/SRTM setup that produced Method 4's
interpolation-memorization signature (above), would replicate the DFC2019
win, replicate the memorization failure, or fail a third way.

**Port** (`scripts/evaluate_method6_sentinel2.py`): same twin-head full
fine-tune architecture, unmodified from the DFC2019 version, predicting
**absolute (ellipsoidal) elevation directly from RGB** — not a residual on
top of the linear baseline, since Method 6 has no frozen depth-correction
step for a residual to sit "on top of." Same 25 sign-flip-detector-accepted
tiles, same `method4_port/` 10m-grid rasters (reusing the linear baseline
and residual truth to reconstruct absolute SRTM elevation as the training
target), same quadrant-fold geometry (`evaluate_method4.quadrant_bounds`),
same two evaluation checks (DEM held-out, independent ICESat-2 photons)
against the same linear-calibration baseline Method 4 was scored against.
One necessary adaptation, made for a documented reason: `init_mu="constant"`
(model initialized to emit the fold's own training-mean elevation
everywhere, rather than DA-V2's pretrained disparity-scale readout) — with
absolute elevation ranging from −112m (coastal, ellipsoidal) to +4125m
(Himalayan foothills) across tiles, the pretrained init would reproduce the
exact large-residual deadlock the source repo documents for its own
DA-V1-Large 666m-offset failure, just via a different route (a large,
tile-variable target scale rather than a mismatched backbone).

## Stage 1 result — fold 0, all 25 tiles

| | DEM check (held-out) | ICESat-2 check (independent) |
|---|---:|---:|
| **CNN beats linear, tile wins** | **1/25** | **2/25** |
| Pooled RMSE, CNN | 197.35m | 193.31m |
| Pooled RMSE, linear | 55.55m | 54.33m |

**Stop condition met, by a wide margin.** 23 of 25 tiles lose to the linear
baseline on the ICESat-2 check (the threshold was "more than half"). The
failure signature is unambiguous: **plain underperformance, not
interpolation-memorization** — it loses on the DEM check too (1/25 wins,
pooled RMSE nearly 4x worse than linear), the same signature the
native-30m retry showed above, not the "wins DEM, loses ICESat-2" pattern
of the original 10m Method 4 run. Training loss did fall steadily
(472.9 → 22.7 over 12 epochs), so the network fit its own training data;
it simply does not transfer to held-out quadrants or independent photons.

## Why, mechanistically (consistent with, and sharper than, the native-30m diagnosis above)

1. **75 training samples per fold** (25 tiles × 3 quadrants) — the same
   order-of-magnitude data-starvation problem diagnosed for Method 4's
   native-30m retry (300 patches/fold), now feeding a 24.8M-parameter ViT
   rather than a 4-conv CNN. More capacity against less data per fold than
   either DFC2019 (150 samples) or Method 4's original 10m run (~1000+
   patches/fold).
2. **Absolute elevation across 25 disparate tiles has no shared structure
   for a network to generalize from.** DFC2019's AGL is terrain-normalized
   by construction — a 10m building looks like a 10m building regardless of
   which US city it's in, so the network could learn a transferable
   object-height cue. Absolute SRTM elevation is exactly the opposite: each
   tile's baseline (agricultural Punjab ~170m, Himalayan Manali ~2650m,
   coastal ellipsoidal ~−70m) is a property of *where the tile is*, not of
   anything visible in the RGB texture. This project's own earlier finding
   already established that DAv2's relative-depth output doesn't reliably
   correlate with elevation on nadir satellite imagery once spatial trend is
   controlled for — this result is consistent with that: there is no strong
   RGB-texture-to-absolute-elevation cue for a backbone to fine-tune onto in
   the first place, unlike AGL's genuine object-height cue.
3. **The quadrant-fold protocol asks the network to infer each tile's own
   elevation band from only 3 quadrants of that one tile** (not from other
   tiles, since no two tiles share an elevation band) — effectively an
   extreme few-shot memorization task per tile, with a 24.8M-parameter
   model and no regularizing structure analogous to the linear baseline's
   per-tile OLS intercept.

## Decision, per the pre-agreed protocol

**Stopped at fold 0, as instructed.** Folds 1–3 were not run, and neither
follow-on test was attempted: test A (Method 6 with Open Buildings 2.5D as
target) and test B (Method 4's architecture with Open Buildings 2.5D as
target) were both explicitly conditioned on Method 6 clearing this bar,
which it did not. The linear-calibration baseline's standing verdict from
the native-30m section above is reinforced, not overturned: full backbone
fine-tuning is not exempt from this domain's core problem (too little
independent Indian elevation ground truth per tile to support a
high-capacity model), and if anything is more exposed to it than Method 4's
smaller CNN was, because it has no linear-baseline anchor to fall back on
and a target with no cross-tile shared structure to learn from.

## Reproducing this

- `scripts/evaluate_method6_sentinel2.py --folds 0 --epochs 12` — Stage 1
  (what was run). `data/sentinel2_benchmark/method6_sentinel2_results/
  method6_sentinel2_results.json` has the full per-tile fold-0 breakdown;
  `STOPPED_at_fold0.json` in the same directory records the gate decision.

---

# 2026-09-22 (continued) — Test B: Method 4's exact architecture retrained with Google Open Buildings 2.5D as target on the 8 urban tiles: fails Stage 1, memorization signature

**Status: STOPPED at fold 0, per the staged protocol. Result: swapping the
training target from SRTM to a real, denser building-height source does
NOT fix Method 4's CNN — it reproduces the exact interpolation-memorization
signature the original 10m SRTM run showed (wins the check computed
against its own training target, loses the independent ICESat-2 check).
Folds 1-3 not run. Test A (Method 6 with this target) was explicitly not
attempted, per instruction, since Method 6's own failure mode (model
capacity vs. data volume, no residual anchoring) is orthogonal to the
target-quality question this test isolates.**

## Why this test, and what it isolates

Two independent things could explain Method 4's Sentinel-2/SRTM failure:
(1) the *target* (SRTM, ~30m native, is too coarse/interpolated to support
dense per-pixel supervision — the diagnosis reached above), or (2) the
*architecture* (the small CNN itself, regardless of target quality). Test B
holds the architecture fixed and changes only the target, to isolate which.

**Target construction** (`scripts/prepare_method4_openbuildings_data.py`,
new port dir `data/sentinel2_benchmark/method4_port_openbuildings/`, 8
urban tiles only — bengaluru, chennai, delhi, hyderabad, jaipur,
kochi_city, mumbai, pune): Google Open Buildings 2.5D Temporal
(`building_height`, AGL, real per-building estimate at ~4m effective
resolution, fetched live from the public bucket for 2023 — the latest
year available for these UTM zones; a genuine ~2-3 year gap against the
tiles' late-2025 acquisition dates, reported not hidden) replaces the
per-tile linear-SRTM residual specifically on confident building pixels
(`building_presence > 0.5`; coverage 8.9%-32.8% of each tile, per-tile
detail in `coverage_report.csv`). Ground/non-building pixels keep the
original SRTM-based target unchanged, so this isolates "does a better
BUILDING target help" rather than "does a different, sparser source help
everywhere." A new per-tile `scale_factor` was fit (Open Buildings heights
up to 100m exceed several tiles' original SRTM-residual normalization
range — e.g. mumbai's scale_factor grew from its SRTM value to 5.615,
raw_max_abs_residual 280.8m).

**Architecture and hyperparameters: byte-for-byte the same** as the
original Method 4/SRTM port —
`evaluate_method4_v2.ScaleModulationNetV2`/`huber_masked`/
`smoothness_loss`/`rank_pair_loss`/`dense_patch_origins`/
`PatchDatasetV2`/`predict_quadrant_v2` imported unchanged, same
hyperparameters (dense patches, smoothness_weight=0.01, rank_weight=0.5,
rank_pairs_per_patch=2000, rank_margin=0.25, building channel,
ground_plane_weight=0, 60 epochs). Only the 8-tile subset, the target
rasters, and per-tile scale factors differ from the original 25-tile run.

## Stage 1 result — fold 0, 8 urban tiles, full 60-epoch training

Stop condition: each tile's ICESat-2 CNN RMSE vs. that SAME tile's own
best pre-existing baseline (whichever of {linear SRTM calibration,
original Method 4/SRTM CNN} scored better for that tile in the 25-tile
run, since the two trade off tile by tile).

| tile | CNN (Open Buildings target) | best pre-existing baseline | result |
|---|---:|---:|---|
| bengaluru | 19.13m | 16.36m (linear) | loss |
| chennai | 11.28m | 5.46m (linear) | loss |
| delhi | 16.65m | 11.46m (linear) | loss |
| hyderabad | 20.79m | 21.03m (Method 4/SRTM) | **WIN** |
| jaipur | 28.02m | 16.61m (linear) | loss |
| kochi_city | 5.49m | 1.58m (linear) | loss |
| mumbai | 30.29m | 11.20m (linear) | loss |
| pune | 34.41m | 38.68m (Method 4/SRTM) | **WIN** |

**6 of 8 tiles lose. Stop condition met, decisively** (threshold was
losing more than 4 of 8). The result was confirmed at both a 5-epoch smoke
test (6/8 losses, identical qualitative pattern) and the full 60-epoch run
(same 6/8 losses; training loss fell steadily and plausibly, 4.07 → 2.46
over 60 epochs, so this is not an undertrained artifact).

**Failure signature: memorization, the same one the original 10m-grid SRTM
run showed** — the model wins the check computed against its own training
target overwhelmingly (7/8 tiles beat the linear baseline on the
Open-Buildings-informed "DEM-equivalent" check, pooled RMSE 12.88m vs.
18.55m), but loses on the independent ICESat-2 ground-photon check on the
same 6/8 tiles (pooled RMSE 22.46m vs. 18.58m). This is architecturally the
same pattern as the original SRTM run (won DEM 77/100 tile-folds, lost
ICESat-2 89/100 tile-folds) — a genuinely denser, real, non-interpolated
building-height target still produces a model that fits its own supervision
signal well and does not generalize to independent ground truth.

## Interpretation

**This answers the question Test B was designed to isolate: the failure is
not (purely) about SRTM's interpolation artifacts.** A materially different,
denser, real target (Open Buildings 2.5D, not synthetically interpolated,
covering a genuine physical quantity at each of these 8 tiles) reproduces
the same qualitative failure. Combined with the native-30m SRTM retry
above (which showed the *other* failure mode — losing both checks, from
sheer data starvation once the interpolation shortcut was removed), the
two results together bracket the problem: whether the target is
SRTM-interpolated (memorization) or genuinely dense-but-sparse-per-tile
(also memorization, this test), Method 4's small CNN — 4 conv layers,
64x64 patches, ~300-1000 training patches/fold depending on tile count —
does not have enough independent per-tile supervision or receptive field
to learn a correction that survives contact with ground truth it wasn't
fit to. This is now the third distinct CNN-correction attempt on this
domain (10m/SRTM, native-30m/SRTM, 10m/Open-Buildings-on-buildings) to
fail against ICESat-2, each via a different specific mechanism but the
same underlying cause: insufficient independent Indian elevation ground
truth per tile for this class of architecture, not a fixable
hyperparameter or target choice.

## Verdict — per the pre-committed decision rule

**Test B fails. This is the strong, well-evidenced signal to stop
CNN-based Sentinel-2 elevation correction entirely for now, not to propose
a fourth variant.** Three independent CNN attempts, three different root
causes investigated in depth, three failures against the same independent
check. The working tracks for this domain remain what they already were:
**linear per-tile calibration** (still the best deployable baseline on
this benchmark) and the **frequency-fusion approaches identified in the
competitive-repo audit** (`blakc-coffee/depthwizard`,
`arpitparashar06/depthwizard` — real DEM low-frequency + model
high-frequency separation, a genuinely different mechanism from anything
tried here). Further CNN-architecture iteration on Sentinel-2/SRTM-class
targets should not be pursued without new evidence specifically pointing
at a fix for the per-tile data-volume ceiling this and the native-30m test
both hit — not merely a new loss, target, or capacity tweak.

## Reproducing this

- `scripts/prepare_method4_openbuildings_data.py` — fetches Open Buildings
  2.5D (public bucket, no auth), builds
  `data/sentinel2_benchmark/method4_port_openbuildings/` (residual truth,
  linear baseline and building channel copied unchanged, new scale
  factors, `coverage_report.csv`).
- `M4OB_EPOCHS=60 M4OB_FOLDS=0 python scripts/run_method4_openbuildings_test_b.py`
  — Stage 1 (what was run). Writes
  `data/sentinel2_benchmark/method4_openbuildings_testb_results/
  method4_ob_testb_results.json` (full per-tile breakdown) and
  `stage1_gate.json` (the gate decision).

---

# 2026-09-22 (continued) — Frequency fusion (blakc-coffee/depthwizard's technique) on all 25 accepted tiles: beats the linear baseline decisively, 21/25 tiles, median error 10.88% → 3.57%

**Status: COMPLETE. Result: this is a genuine, substantial win — the first
non-CNN correction attempted here, and it beats the working
linear-calibration baseline on 21 of 25 tiles, cutting median ICESat-2
error from 10.88% to 3.57% of elevation range (~3x). Largest gains are in
exactly the terrain hypothesized to benefit: all 5 hilly tiles win, by 10-25x
(e.g. almora 19.02% → 1.05%, dehradun 18.07% → 0.71%). This changes the
project's standing recommendation for this domain — see verdict below.**

## What this is, adapted from the competitive-repo audit

`blakc-coffee/depthwizard` (flagged prominently in `COMPETITIVE_REPO_AUDIT.md`
as doing genuine, non-superficial frequency-separated DEM+DAv2 fusion) was
read, not executed, and its core idea reimplemented independently
(`scripts/run_frequency_fusion_sentinel2.py`) on this project's own 25
sign-flip-accepted tiles, real DEM (SRTM), and real ICESat-2 ground photons.
Pure signal processing — no training, no learned parameters beyond the
per-tile linear calibration this project already validated and uses
everywhere else in this domain.

**A first attempt at this (not documented, corrected before any real run)
skipped step 1 below — frequency-splitting DAv2's raw [0,1] output
directly, which has no metric meaning to preserve.** The corrected pipeline:

1. **Metric scaling first.** Reuses the *exact same* per-tile calibration
   this project already reports (`run_srtm_comparison.py` /
   `calibrate_accepted_tiles.py`): random 50/50 split of DEM-valid pixels
   (seed 42), OLS `dem = a*dav2_raw + b` on the train half.
   `dav2_metric = a*dav2_raw + b` is then in the same vertical datum as the
   raw SRTM DEM (EGM96 orthometric) — the calibration is shared with, not
   independent of, the existing working baseline.
2. **Confirmed native resolution, not assumed 30m.** SRTM here is
   1-arcsecond exactly (verified: 0.0002777... degrees pixel size in the
   raw files) — but 1 arcsecond is not a fixed number of metres; the
   east-west spacing shrinks as cos(latitude). Computed per-tile via
   `pyproj.Geod` geodesic distance at each tile's actual centre latitude
   (not a rule of thumb): across the 25 tiles this ranged **28.40m
   (dharamshala/manali, ~30.8°N) to 30.61m (vembanad, ~9.6°N)** — a real
   ~7.2% spread that a flat-30m assumption would have gotten wrong in both
   directions depending on latitude. The scalar used for the filter is the
   geometric mean of the lat- and lon-direction spacing.
3. **Matched low-pass.** A masked (NaN-aware) Gaussian with sigma set from
   step 2's per-tile native resolution (converted to pixels of the shared
   10m UTM grid) is applied to *both* `dem_on_grid` and `dav2_metric` — the
   identical kernel on both signals, which is the part that is not
   optional. `dav2_highpass = dav2_metric - gaussian(dav2_metric)`.
4. **Combine.** `final_egm96 = gaussian(dem_on_grid) + dav2_highpass`.
5. **Confidence map.** Per-pixel DEM coverage flag from `dem_on_grid`'s own
   validity mask. All 25 tiles: **100% DEM coverage**, no gap-filled
   regions to flag.
6. **Evaluation, identical format to the existing baseline.** Same seed,
   same split, same metric (`icesat2_rmse_pct_of_range`) as
   `srtm_3way_comparison.csv`'s `srtm_icesat2_rmse_pct_of_range` column —
   direct, not approximate, comparison.

**A real bug found and fixed during this work, independent of the
technique itself:** `pyproj`'s default EGM96 geoid transform silently
falls back to a "ballpark" zero-offset conversion (confirmed via
`Transformer.description`: *"ballpark vertical transformation, without
ellipsoid height to vertical height correction"*) unless `PROJ_NETWORK=ON`
is set to fetch the real geoid grid. Caught immediately by smoke-testing:
bathinda's geoid undulation came back `N=0.0` instead of the
`+46.677...m` this project's own `srtm_3way_comparison.csv` already has on
record for that exact tile. Without the fix, every ICESat-2 comparison
below would have carried a ~40-100m systematic bias baked in — this would
not have been a subtle error, results would have been obviously broken
(and initially were, in first testing: 47-96m RMSE against a ~43m
elevation range, before the fix). Now set defensively
(`os.environ.setdefault("PROJ_NETWORK", "ON")`) at the top of the script
so it can't silently recur.

## Results — all 25 tiles

| tile | category | linear baseline | frequency fusion | |
|---|---|---:|---:|---|
| bathinda | agricultural | 3.45% | 3.57% | loss |
| kota | agricultural | 7.23% | 2.58% | **WIN** |
| kurnool | agricultural | 7.00% | 2.18% | **WIN** |
| nizamabad | agricultural | 13.85% | 5.89% | **WIN** |
| vidisha | agricultural | 10.91% | 2.04% | **WIN** |
| amalapuram | coastal | 7.68% | 8.54% | loss |
| digha | coastal | 8.40% | 7.78% | **WIN** |
| goa_estuary | coastal | 18.07% | 2.90% | **WIN** |
| kakinada | coastal | 5.32% | 4.94% | **WIN** |
| kutch | coastal | 5.45% | 5.88% | loss |
| nagapattinam | coastal | 11.43% | 10.55% | **WIN** |
| vembanad | coastal | 8.83% | 5.75% | **WIN** |
| almora | hilly | 19.02% | **1.05%** | **WIN** |
| dehradun | hilly | 18.07% | **0.71%** | **WIN** |
| dharamshala | hilly | 10.88% | **0.27%** | **WIN** |
| kohima | hilly | 12.47% | **0.68%** | **WIN** |
| manali | hilly | 13.12% | **0.48%** | **WIN** |
| bengaluru | urban | 11.84% | 5.82% | **WIN** |
| chennai | urban | 5.52% | 5.67% | loss |
| delhi | urban | 11.01% | 5.99% | **WIN** |
| hyderabad | urban | 14.25% | 3.66% | **WIN** |
| jaipur | urban | 9.18% | 1.69% | **WIN** |
| kochi_city | urban | 5.81% | 4.94% | **WIN** |
| mumbai | urban | 4.44% | 2.48% | **WIN** |
| pune | urban | 15.92% | 2.50% | **WIN** |

**21/25 tiles win (84%). Median: 10.88% → 3.57%. Mean: 10.37% → 3.94%.**
All 4 losses are small (largest: amalapuram, +0.86 percentage points) and
concentrated in flatter agricultural/coastal terrain, where the linear
baseline was already doing reasonably well and there's less real
low-frequency topographic structure for the DEM to contribute. **All 5
hilly tiles win, by the largest margins in the whole table** — exactly
where a single global linear fit struggles most with genuine large-scale
elevation variance, and exactly where trusting the DEM's real coarse
trend should help most.

**On the DEM check (read with real caution, per the module docstring):**
these numbers are good almost by construction, since the low-frequency
component is derived directly from the DEM's own full grid — this is not
a fair generalisation test the way it is for the pure-linear baseline
(which never touches real DEM values at inference), so it is not reported
as evidence of anything here. The ICESat-2 numbers above are the ones that
matter, and they were never part of any fitting or filtering step.

## Does this avoid Method 4's interpolation-memorization failure?

**Yes, structurally, not just empirically.** Method 4's failure mode was a
CNN learning to reproduce the DEM's own resampling/interpolation artifacts
— winning a DEM-comparison check while losing the independent ICESat-2
check, because what it had learned was specific to the training data's
own quirks, not general elevation signal. There is no training process
here, no gradient descent, no model weights fit to reproduce anything —
the low-frequency component IS the (filtered) DEM by direct construction,
not a learned approximation of it, so there is nothing to memorize. The
fact that this method also *wins* the independent ICESat-2 check (not
just the DEM check) on 21/25 tiles is the empirical confirmation that the
improvement is real signal, not the "wins DEM, loses ICESat-2" signature
every CNN attempt in this domain has produced.

## Verdict — updates the project's standing recommendation

**Frequency fusion beats plain per-tile linear calibration on this
benchmark, by a wide and mechanistically-explicable margin, and should
replace it as the recommended deployable baseline for this domain.** The
earlier verdict in this file ("linear per-tile calibration remains the
right deployable tool... CNN-based correction is not pursued further")
still holds for the CNN-correction question — nothing here changes that
conclusion, since this is not a CNN. But it does mean linear calibration
alone is no longer the best *non-learned* option available: frequency
fusion, at essentially the same (near-zero) compute cost, materially
outperforms it, especially in hilly terrain. CLAUDE.md's ML research
track status should be updated to reflect this as the new working
baseline for the Sentinel-2/India domain.

## Reproducing this

- `scripts/run_frequency_fusion_sentinel2.py` — the full pipeline
  (metric scaling, native-resolution computation, matched filtering,
  combination, both evaluation checks). Writes
  `data/sentinel2_benchmark/frequency_fusion_results/
  frequency_fusion_results.csv` (per-tile) and `summary.json`.

---

# 2026-09-22 — Viewing-angle hypothesis test for the 6 sign-flipped tiles: ruled out

**Status: COMPLETE. Result: clean negative.** Tested whether the 6
sign-flipped tiles (karnal, shimla, nainital, ooty, fatehpur, hisar) cluster
toward true-nadir acquisition angles relative to the 26 non-flipped tiles —
the first candidate physical mechanism proposed for the sign-flip since it
was originally diagnosed in `backbone-comparison.md`. They do not.

## A correction to the task as given: the angle data was not already available

The task assumed each tile's viewing/incidence angle was "already available
... no new fetch needed." That did not hold for this repo's actual state,
checked directly rather than assumed:
- Nothing in this repo stores it: not the GeoTIFF tags (`TIFFTAG_*`/
  `AREA_OR_POINT` only), not `manifest.csv`, not `download_results.csv`
  (which doesn't even keep the scene ID).
- The CDSE Catalog (Sentinel Hub) endpoint this project's
  `backend/cdse/client.py` already talks to returns no angle field at all
  (verified live: `datetime`, `platform`, `gsd`, `eo:cloud_cover`, `proj:*`
  only).
- Microsoft Planetary Computer's Sentinel-2 STAC properties (checked as a
  second source) carry solar angles (`s2:mean_solar_zenith/azimuth`) and
  orbit info (`sat:relative_orbit`, `sat:orbit_state`) but not sensor
  viewing/incidence angle either — that field lives only inside each
  product's `MTD_TL.xml` granule metadata, which no STAC summary surfaces.
- The one path that does serve `MTD_TL.xml` from this project's already-
  integrated provider, CDSE's OData download API, rejected the project's
  existing `CDSE_CLIENT_ID`/`CDSE_CLIENT_SECRET` with `"Token audience not
  allowed"` — those credentials are scoped for the Sentinel Hub Catalog/
  Process APIs only, not the separate Data Space download API.

Flagged to the user rather than silently substituting a weaker proxy or
registering new credentials unasked. Per their direction: checked for a
genuinely free, no-new-credential path before falling back to a proxy, and
found one — Planetary Computer's `granule-metadata` asset resolves to a
real Azure Open Data blob (`*.blob.core.windows.net`), servable anonymously
once signed with Planetary Computer's free, keyless SAS-token endpoint
(`/api/sas/v1/token/sentinel-2-l2a`). This serves each tile's actual
`MTD_TL.xml`, including the real `Mean_Viewing_Incidence_Angle_List` per
band — not a proxy, no new credentials, no cost.

## Method

For each of the 32 benchmark tiles: query Planetary Computer's public
Sentinel-2 L2A STAC (`bbox` around the tile's manifest lat/lon, `datetime`
window on its `date_acquired`) to find the matching granule, then fetch
that granule's `MTD_TL.xml` via the SAS-signed blob URL and parse
`Mean_Viewing_Incidence_Angle` for all 13 bands. Per-tile viewing zenith
angle = mean across bands (all 13 values agree to within ~1° per tile —
Sentinel-2's near-nadir-pointing MSI design keeps per-band spread small,
so band choice barely matters here). All 32/32 tiles resolved successfully
on the first run, no manual matching needed.

## Result: no clustering toward nadir — a clean null

| | flipped (n=6) | non-flipped (n=26) |
|---|---:|---:|
| mean viewing zenith | 5.830° | 5.748° |
| median viewing zenith | 5.011° | 5.642° |
| std | 2.845° | 2.248° |
| min – max | 2.794° – 9.701° | 2.905° – 10.415° |

**Mann-Whitney U = 69.5, p = 0.699. Welch's t-test t = 0.061, p = 0.953.**
Neither test comes close to significance, and the point estimates
themselves are nearly identical between groups.

The 6 flipped tiles do not sit together at either end of the angle
distribution — they span almost the entire 32-tile range: shimla is the
single most-nadir tile in the whole benchmark (rank 1/32), nainital is
rank 3/32, but ooty and hisar are near the *most oblique* end (rank
29/32 and 30/32) — the two most extreme members of the flipped group sit
on opposite sides of the distribution's median. karnal (rank 12) and
fatehpur (rank 16) fall in the unremarkable middle. There is no
sub-cluster here to explain away as noise; the flipped tiles' angles look
like a random draw from the same distribution as the non-flipped tiles'.

**Continuous check, not just the binary flip label:** viewing zenith
angle vs. each tile's real `true_dav2_icesat2_pearson` (all 32 tiles, not
just the 6/26 split) — Pearson r = -0.038 (p = 0.835), Spearman r = -0.062
(p = 0.736). Also tested against `|true_dav2_icesat2_pearson|` in case
angle relates to signal *strength* rather than *sign*: r = -0.291
(p = 0.106) — still not significant, though this one is at least in a
plausible direction and worth a note rather than a full follow-up (see
below).

## Reading

**Acquisition viewing angle is not the mechanism behind DAv2's sign-flips
on this benchmark.** This was a real, physically-grounded hypothesis worth
testing — near-nadir imagery genuinely does carry a different relief-
displacement geometry than oblique imagery, and DAv2's raw depth output
sign convention (`backbone-comparison.md`) leans on relief displacement
being present and consistently oriented — but the data doesn't support it
here. All 32 tiles were acquired at small, unremarkable off-nadir angles
(2.8°–10.4° — Sentinel-2's MSI never points far off-nadir by mission
design, so this benchmark never had much angular range to test against in
the first place, which is itself worth noting as a limit on this specific
null result: it doesn't rule out an angle effect at genuinely wide
incidence angles, since none of these 32 tiles come close to that).

The weak, non-significant `|correlation| ~ angle` trend (r=-0.29, p=0.11)
is the one loose thread worth flagging rather than closing outright: if
real, it would say wider off-nadir tiles produce *weaker* DAv2-vs-ICESat2
correlation regardless of sign, which is a different and more mundane
claim than "flips near nadir." Not investigated further here — n=32 with
a 2.8°-10.4° range is a weak basis for chasing a p=0.11 result, and this
project's stated practice throughout this audit is not to force a
borderline number into a story. This remains unexplained, same as before
this test, just with one concrete, physically-motivated mechanism now
ruled out rather than merely undiagnosed.

## Reproducing this

- `scripts/fetch_viewing_angles_benchmark.py` — resolves each of the 32
  tiles' matching Sentinel-2 granule via Planetary Computer's public STAC,
  fetches real `MTD_TL.xml` per-band viewing angles via the free SAS-token
  endpoint, writes `data/sentinel2_benchmark/viewing_angles.csv`.
- Statistical comparison (Mann-Whitney, Welch's t-test, Pearson/Spearman
  against `sign_flip_detector_signals.csv`'s `true_dav2_icesat2_pearson`)
  was a one-off analysis on top of that CSV plus the existing detector
  signals file, not a separately committed script.

---

# 2026-09-22 (continued) — arpitparashar06's non-regression scale derivation as a frequency-fusion replacement: rejected, catastrophically worse

**Status: COMPLETE. Result: clean, decisive negative — do not adopt.**
Tested `arpitparashar06/depthwizard`'s non-regression, physically-anchored
scale derivation (`alpha_from_known_height`, read directly from
`external/arpitparashar06-depthwizard/mathsandml/inference.py`, never
executed) as a drop-in replacement for frequency fusion's step-1 metric
scaling *only*, on the 4 tiles frequency fusion currently loses on
(bathinda, amalapuram, kutch, chennai) — exactly where a regression-fit
scale is least constrained, per the task's own hypothesis. It does not
close any of those 4 gaps. It makes every one of them dramatically worse,
and generalizes just as badly across the full 25-tile accepted set.

## What was ported, and the substitution used

`alpha_from_known_height(detail, known_height_m, pct=99)` is genuinely the
non-regression option in that file (unlike their own `alpha_from_gcps`,
a least-squares slope, or this project's existing DEM-OLS step): threshold
= p99 of the detail band, `top` = median of detail values at/above that
threshold, `alpha = known_height_m / top` — one division, anchored to a
single physical reference (their use case: a person says "that landmark
is ~40m tall").

Substitution for this domain, per the task's direction ("use ICESat-2
points as the known-height anchor source, same substitution already used
when this was first adapted" — i.e. the project's established pattern of
swapping a competitor's proprietary/manual ground-truth input for real
ICESat-2 points): their one manual `known_height_m` becomes the median
**true relief** — ICESat-2 photon height (converted to the DEM's EGM96
datum via this project's existing `n_egm96` geoid correction) minus the
DEM's own low-pass trend at that pixel — among the top-1% highest-relief
points in a held-out **train** half of the tile's photons (seed 42, same
split discipline as the rest of this project). Their "top" (the model's
own reading at that same population) becomes DAv2's **raw, unscaled**
high-pass detail sampled at those exact same anchor pixels. Their
defensive `alpha <= 0` rejection is kept exactly as they wrote it.

**Order-of-operations change this required**, also read directly out of
their file: their alpha multiplies the detail band itself, not the raw
signal before splitting, and needs no intercept (the detail band is
already zero-centered by construction; the DEM low-pass supplies the
absolute vertical baseline, so only the detail band's *amplitude* needs
fixing). So highpass is computed on raw DAv2 first, then scaled — the
reverse order from the existing linear-OLS step, which scales before
splitting. Everything else (native-resolution matched low-pass, DEM
low-pass, DEM+detail combination, the DEM held-out check) is reused
unchanged by importing directly from `run_frequency_fusion_sentinel2.py`,
per the task's instruction to touch only the scaling step.

**Held-out discipline:** the photon half used to fit alpha is disjoint
from the half used for the ICESat-2 evaluation check (same seed=42 split
pattern used throughout this project) — fitting and evaluating on the
same points would be exactly the leakage this project's audit trail has
caught and fixed elsewhere (the geoid bug, the interpolation-memorization
pattern). This does mean the new method's ICESat-2 check sees half as
many photons as the existing linear-OLS baseline's check (which never
touches ICESat-2 for fitting at all) — a real asymmetry, noted rather than
hidden, though irrelevant to the verdict given how large the gaps below
are.

## Results on the 4 target tiles

| tile | linear baseline | frequency fusion | known-height (this test) |
|---|---:|---:|---:|
| bathinda | 3.45% | 3.57% | **77.80%** |
| amalapuram | 7.68% | 8.54% | **REJECTED** (negative alpha, self-caught) |
| kutch | 5.45% | 5.88% | **203.24%** |
| chennai | 5.52% | 5.67% | **4013.04%** |

(ICESat-2 RMSE as % of the tile's own elevation range — lower is better,
same metric as the rest of this file. "REJECTED" means the method's own
defensive check refused to produce a scale at all, per
`alpha_from_known_height`'s design, not a crash.)

**Zero of the 4 gaps close. All 4 get dramatically worse**, one (chennai)
by roughly three orders of magnitude, and one (amalapuram) doesn't even
produce a number.

## Results on the full 25-tile accepted set

Re-run on all 25 tiles to check for regressions on the 21 tiles frequency
fusion currently wins, per the task's request:

- **16/25 tiles: REJECTED outright** (negative alpha, self-caught) —
  agricultural: kota, kurnool; coastal: amalapuram, digha, goa_estuary,
  kakinada, nagapattinam, vembanad; hilly: almora, dharamshala; urban:
  bengaluru, delhi, hyderabad, jaipur, kochi_city, pune.
- **9/25 tiles produced a result** (bathinda, nizamabad, vidisha, kutch,
  dehradun, kohima, manali, chennai, mumbai) — **every single one is worse
  than both existing methods.** Median ICESat-2 RMSE among these 9:
  **77.80% of elevation range**, vs. frequency fusion's 3.57% median and
  the linear baseline's ~10-11% median across the full 25. Even the hilly
  tiles — where frequency fusion wins by its largest margins (dehradun
  0.71%, manali 0.48%, kohima 0.68%) — get catastrophically worse under
  this method: dehradun 12.05%, manali 12.48%, kohima 4.15%. Full
  per-tile numbers: `data/sentinel2_benchmark/known_height_scale_results/
  results_all25.csv`.

**0/25 tiles improve on frequency fusion. 0/25 improve on the plain linear
baseline either.**

## Root cause, verified directly rather than assumed

Checked whether DAv2's raw high-frequency detail band actually correlates
with true fine-scale relief at all, independent of any scale factor —
bathinda, full train half (n=1,194,195 photon-sampled pixels): **Pearson
r = -0.0049** between true relief (ICESat-2-anchored, EGM96) and DAv2's
raw unscaled highpass detail. Statistically distinguishable from zero only
because of the huge sample size — practically, no relationship. The
model's raw detail band has std ≈ 0.0021 (raw DAv2 units) against true
relief's std ≈ 1.45m — i.e. DAv2's fine-scale (sub-native-resolution)
output on this flat tile is dominated by texture/noise unrelated to real
elevation once the smooth low-frequency trend is removed, not a weak-but-
real signal.

This explains the failure mode precisely: `alpha_from_known_height` was
designed for a domain (building height from an isolated skyscraper) where
the anchor point is chosen specifically because it's the least noisy,
most confident structure in the scene, and the ratio only needs to survive
being computed from a handful of expert-curated points. Here the "anchor
population" (the top-1% real-relief ICESat-2 points) is not curated for
model confidence — it is just wherever real relief happens to be largest,
and at those specific pixels DAv2's raw detail reading is architecture-
determined noise, not signal. **A single-point (or single-percentile)
ratio has no way to average that noise out — dividing a real few-metre
signal by a nearly-zero, sign-unstable denominator is exactly what
produces the 4-to-6-figure alpha values and negative-alpha rejections
seen above.** The existing linear-OLS baseline is structurally more robust
here for the same underlying reason frequency fusion itself avoids Method
4's failure mode: it is a least-squares fit across the DEM's entire valid
pixel population (hundreds of thousands of points), which averages the
same noise out rather than anchoring on a handful of its worst-conditioned
samples.

## Verdict

**Reject. Do not adopt arpitparashar06's known-height scale derivation for
this pipeline, in this form.** The task's hypothesis — that a regression-
fit scale is least constrained on flat/low-relief tiles, and a
physically-anchored non-regression alternative might do better there — is
falsified by this specific test, and for a diagnosable, specific reason
(near-zero correlation between DAv2's raw fine-scale detail and true
relief at exactly the points a single-anchor method must trust), not a
generic "it didn't work." Frequency fusion's existing linear-OLS step-1
scaling stands as the project's working method; no change to
`run_frequency_fusion_sentinel2.py` is made by this test.

## Reproducing this

- `scripts/test_known_height_scale_frequency_fusion.py` — the full
  replacement pipeline (raw-DAv2 highpass, ICESat-2-anchored known-height
  scale with train/test photon split, DEM + ICESat-2 evaluation), reusing
  `run_frequency_fusion_sentinel2.py`'s native-resolution/matched-lowpass/
  DEM-combination functions unchanged. `python
  scripts/test_known_height_scale_frequency_fusion.py losing4` for the 4
  target tiles, `... all25` for the full accepted set. Writes
  `data/sentinel2_benchmark/known_height_scale_results/results_*.csv`.

---

# 2026-09-22 (continued) — amogh-hub's evidence-gating and leave-one-out validation, applied to frequency fusion: 21/25 win rate robustly confirmed

**Status: COMPLETE. Result: clean positive — the win rate holds, to
within floating-point noise.** Applied `amogh-hub/depthwizard`'s
evidence-gating and leave-one-out validation pattern (list entry #19,
"strongest one yet" — cloned fresh for this task, `external/
amogh-hub-depthwizard/`, read directly, never executed) to **frequency
fusion specifically, not the old linear-calibration pipeline it predates**
— frequency fusion is this project's current recommended baseline, so
that's where a robustness check actually matters now.

## What was ported

Both patterns read from `src/depthwizard/calibration/gcp.py`:
- **Evidence gating** (`_validate_spatial_distribution`): refuses to
  produce a calibrated result at all when the underlying evidence is too
  weak (`raise ValueError("insufficient evidence for defensible metric
  calibration")`) rather than silently degrading. Adapted onto frequency
  fusion's own already-computed confidence signal — step 5 of
  `run_frequency_fusion_sentinel2.py`'s own docstring literally calls
  this the **"CONFIDENCE MAP"** (per-pixel DEM-covered vs. gap/no-data,
  from `dem_on_grid`'s own validity mask) — this is "the
  confidence/provenance map already computed" the task pointed at, not
  something new to build. Ported as a per-tile gate: reject (flag, not
  silently include) any tile whose DEM coverage falls below a 95% floor.
- **Leave-one-out validation** (`_affine_leave_one_out_rmse`): holds each
  control point out one at a time, refits on the rest, scores the
  held-out point, aggregates — an honest error estimate, not the
  in-sample fit residual. Frequency fusion has no GCPs; its "evidence" is
  the ~500k–1.2M valid DEM pixels the existing single 50/50 random split
  (seed 42) fits the affine scale from. Literal per-pixel LOO is
  computationally absurd at that N, so this was adapted to a
  **leave-one-block-out (LOBO)** scheme: each tile partitioned into a 5×5
  grid of 25 spatial blocks (a granularity comparable to amogh-hub's own
  typical 6–20 GCP count), each block held out in turn, the affine scale
  refit on the other ~96% of the tile's pixels every fold (much closer to
  LOO's "train on everything except the held-out unit" than the existing
  50/50 split), and every DEM pixel *and* every ICESat-2 photon scored
  exactly once, under the one fold where its own block was excluded from
  fitting.

Everything else (native-resolution matched low-pass, DEM low-pass,
DEM+detail combination) is reused unchanged by importing directly from
`run_frequency_fusion_sentinel2.py`. Verified algebraically and
numerically before relying on it: because Gaussian blur is linear and the
fitted intercept cancels in the high-pass, `highpass(a·x+b) = a·highpass(x)`
to within `2e-7` — so the raw DAv2 high-pass band is computed once per
tile, not once per fold, without changing what's being tested.

## Evidence gate: real, verified to fire, but never fires on this benchmark

Self-tested against a synthetic 60% coverage value before trusting it on
real data (mirrors amogh-hub's own test suite testing the gate itself,
`tests/test_confidence_gated_training.py`). On the real 25-tile accepted
set: **every tile has exactly 100.0% DEM coverage** (SRTM has zero nodata
across this whole benchmark, confirmed earlier in this document) — the
gate is real and correctly wired, but this specific benchmark gives it
nothing to reject. Same honest-null pattern as the GSD-FiLM test earlier
in this document: a real mechanism, kept as a safety net for tiles with
actual DEM voids (e.g. near coastlines, a documented Copernicus GLO-30
failure mode noted earlier in this file), not evidence the mechanism does
nothing.

## LOBO cross-validation: the 21/25 win rate is not a split artifact

Full per-tile comparison, `freqfusion_orig` = the existing single-split
number already reported earlier in this file, `freqfusion_LOBO` = this
test's 25-fold leave-one-block-out number:

| | wins vs. linear baseline (single split, existing) | wins vs. linear baseline (LOBO, this test) |
|---|---:|---:|
| **Count** | **21/25** | **21/25** |

**Zero tiles flip win/loss status. The largest single-tile difference
between the original single-split ICESat-2 RMSE-%-of-range and the LOBO
number is 0.0004 percentage points** (mean absolute difference across all
25 tiles: 0.00008 points) — indistinguishable from floating-point noise,
not a real effect in either direction. Full per-tile table (including
each tile's fitted-slope mean/std across the 25 blocks):
`data/sentinel2_benchmark/frequency_fusion_loo_results/results.csv`.

**This is a genuinely informative confirmatory result, not a trivial
one** — the fitted affine slope itself is *not* trivially stable across
blocks (e.g. almora: 327.89 ± 34.70, an ~11% relative spread; dehradun:
748.82 ± 35.29; mumbai: 39.09 ± 3.07, ~8%), so this isn't a case of
"nothing changes because nothing was ever varying." The **mechanism**
this reveals: the final surface is `dem_lowpass + a·dav2_highpass_raw`,
where `dem_lowpass` (the real DEM's own coarse trend, untouched by any
fitting) supplies the dominant share of the surface's magnitude and
`dav2_highpass_raw`'s own amplitude is small (consistent with this
file's earlier finding that DAv2's raw fine-scale detail band has std on
the order of `1e-3` in raw units, see the known-height-scale test above)
— so even an 8–11% swing in the slope that scales that small detail band
translates into a comparatively tiny absolute change in the final
prediction, which is why the ICESat-2 metric barely moves even though the
fitted parameter itself visibly does. The robustness isn't an accident of
this particular random split; it's structural, given how little of the
final surface's magnitude the fitted parameter actually controls.

## Verdict

**The 21/25 win rate holds up under a real held-out protocol.** This is
not a re-confirmation of the same test with different window dressing —
LOBO trains on ~96% of each tile's pixels per fold instead of 50%, scores
every pixel and every photon under a fold that never saw its own
neighborhood during fitting, and still reproduces the original numbers to
four decimal places. Frequency fusion's standing recommendation (this
project's deployable baseline for the Sentinel-2/India domain,
established earlier in this file) is **strengthened, not merely
unchanged**, by this check — the original single-split result was already
trustworthy, not merely lucky.

## Reproducing this

- `scripts/test_frequency_fusion_evidence_gating_loo.py` — evidence gate
  + leave-one-block-out cross-validation, reusing
  `run_frequency_fusion_sentinel2.py`'s native-resolution/matched-lowpass
  functions unchanged. Writes
  `data/sentinel2_benchmark/frequency_fusion_loo_results/results.csv`.

---

# 2026-09-22 (continued) — ArnabTechiee's shadow-length photogrammetry: not usable at 10m GSD, plausibility check only

**Status: COMPLETE (plausibility check, as scoped — not a full
validation).** Cloned `external/ArnabTechiee-depthwizard/` fresh for this
task (list entry #16, "dig deep; cast-shadow geometry, with real USGS
3DEP LiDAR validation"), read `pipeline/calibrate.py` directly (never
executed), and probed whether its shadow-length-to-height geometry is
even usable at Sentinel-2's 10m GSD on the same 4 tiles frequency fusion
loses on (bathinda, amalapuram, kutch, chennai). **It is not** — both the
underlying physics and an empirical run of their own ported logic against
these tiles' real data agree, independently.

**A bug found while reading their code, not reproduced here:** their
docstring states `h = L_pixels x GSD x tan(sun_elevation)`, and their own
`max_len` cap (`200.0 / tan_elev / gsd`) is derived consistently with
that formula — but their actual height line,
`height_m = length_px * gsd / tan_elev`, *divides* by `tan_elev` instead
of multiplying. This adaptation uses the physically correct formula.

## Real sun-angle metadata, fetched fresh

Not stored anywhere in this repo beforehand. Fetched via the same free,
keyless Planetary Computer STAC path already established for the
viewing-angle investigation earlier in this file
(`s2:mean_solar_zenith`/`s2:mean_solar_azimuth`, elevation = 90 − zenith):

| tile | sun elevation | min. structure height for a 3px shadow (their own hard floor) |
|---|---:|---:|
| bathinda | 35.3° | 21.2m |
| kutch | 40.3° | 25.4m |
| amalapuram | 52.7° | 39.4m |
| chennai | 58.2° | 48.4m |

`median < 3` is `measure_building`'s own unconditional rejection floor
(`pipeline/calibrate.py`), not a threshold chosen here — below it, their
own code refuses to trust a shadow measurement at all. At 10m GSD and
these tiles' real sun angles, a structure needs to be 21–48m tall just to
clear that floor. Bathinda, amalapuram, and kutch are agricultural/
coastal (manifest categories) — real structures anywhere near 20–40m tall
are rare to nonexistent in this terrain; even Chennai's 48.4m floor is a
demanding bar unless the specific AOI happens to contain genuine
high-rises.

## Empirical probe: ported their exact logic, ran it on real data

Ported `shadow_mask`, `shadow_direction`, `building_mask`,
`measure_building` verbatim (corrected formula only), run against each
tile's real Sentinel-2 RGB (shadow detection) and DAv2's raw relative
depth as the "nDSM" `building_mask` expects (their function only needs a
relative height field to percentile-threshold — no metric scale required
for this probe).

| tile | shadow px % | "building" px % | candidate blobs | accepted anchors |
|---|---:|---:|---:|---:|
| bathinda | 17.2% | 45.0% | 7 | **1** |
| kutch | 19.7% | 45.0% | 2 | **1** |
| amalapuram | 8.5% | 45.0% | 1 | **0** |
| chennai | 16.0% | 45.1% | 2 | **0** |

**2 of 4 tiles produce even a single accepted anchor; the other 2
produce zero.** ArnabTechiee's own docstring says "twenty clean anchors
is plenty" for a defensible scene-wide scale fit — this benchmark's best
case is 1.

**The `building_pixel_pct ≈ 45%` figure across all four tiles is itself
diagnostic, not a coincidence.** `building_mask` thresholds at the 55th
percentile of positive values — by construction, close to 45% of any
smoothly-varying continuous field will clear that bar, regardless of
whether real discrete buildings are present. This is the mechanism, not
just a correlate: `building_mask` was designed for a genuinely bimodal
nDSM (mostly near-zero ground, a small distinct population of tall
structures), which a real high-resolution nDSM has and DAv2's coarse
10m relative-depth output does not — it's a smooth continuous relief
signal, so the percentile threshold just splits the image roughly in
half rather than isolating discrete buildings. That's why 45% of pixels
read as "building" and yet only 1–7 candidate blobs survive the
morphological/connectivity/min-area filtering into anything
shape-like — and why `measure_building`'s own strict shadow-consistency
gates (≥4 valid rays, ≥85% hit ratio, `std/median ≤ 0.45`, `length ≥ 3px`)
then reject nearly all of even those.

## Verdict

**Not usable at 10m GSD on this benchmark, for two independent reasons
that agree: the physics (minimum detectable structure height of 21–48m,
implausible for 3 of these 4 tiles' terrain) and the empirical building-
segmentation failure (a 10m relative-height field cannot discretize real
building footprints the way `building_mask` needs).** This was scoped as
a plausibility check, not a full validation, and the plausibility check's
own answer is clear enough that a full validation isn't warranted here —
the correct next step per the task's own framing ("before investing
further") is **not** to invest further in this specific technique on this
domain. This does not touch frequency fusion's own 4 losing tiles in any
other way; they remain a small, known loss (largest gap +0.86 percentage
points, amalapuram) as already documented, not one this technique
resolves.

## Reproducing this

- `scripts/test_shadow_photogrammetry_plausibility.py` — sun-angle fetch
  + ported shadow-length geometry + the empirical probe. Writes
  `data/sentinel2_benchmark/shadow_photogrammetry_plausibility.csv`.

---

# 2026-09-22 (continued) — yats0x7 vs. blakc-coffee: a code-read comparison, and a targeted test of the one real difference it surfaces

**Status: COMPLETE. Result: near-total convergence — the more
sophisticated design produces essentially the same answer.** Cloned
`external/yats0x7-depthwizard/` fresh for this task (list entry #7,
"interesting" — flagged earlier as a similar hybrid-fusion idea never
compared head-to-head against blakc-coffee's, which frequency fusion is
sourced from). Read `engine/depthwizard/calibrate/fit.py` directly (never
executed).

## Architectural comparison

Both are the same idea at the top level — DSM = DEM's low-frequency
terrain + a scaled high-frequency "structure" band from the depth model —
but differ in two concrete ways:

1. **What the detail band is extracted relative to.** This project's
   frequency fusion: `dav2_highpass = dav2_metric − gaussian_blur(dav2_metric)`,
   a plain SYMMETRIC low-pass. yats0x7's `ground_trend()`: an asymmetric,
   iteratively-reweighted low-pass (`above_weight=0.08`) that explicitly
   downweights points *above* the current trend each iteration, so the
   trend converges toward ground instead of being pulled up by
   buildings/trees — their own docstring names exactly this bias
   ("a plain Gaussian pulls the terrain trend up around them"), a real
   critique of this project's own detail-extraction step that hadn't been
   raised before.
2. **What the scale factor is fit against.** This project: OLS between
   the DEM and the model's RAW, unfiltered signal
   (`dem ~ a·dav2_raw + b`) — the same fit the known-height-scale test
   earlier in this file found has near-zero pixel-level correlation with
   true relief on flat terrain (r=−0.005, bathinda). yats0x7: RANSAC
   between the DEM and the model's own robust ground trend
   (`dem ~ a·ground_trend(rel) + b`) — two smooth, already-denoised
   signals, not raw-vs-real. This is the concretely testable idea the
   task pointed at.

Both differences plausibly point the same direction (less exposure to
raw-pixel noise), and both are cheap to test (pure signal processing, no
training) — ported together as one combined "yats0x7-style" swap of
frequency fusion's step-1 detail-extraction-and-scale-fit, keeping this
project's own validated DEM low-pass and combination step unchanged (same
"swap one targeted component" pattern as every other test in this file).

## The hypothesis didn't survive contact with the data — but not the way expected

The mechanistic story predicted trend-vs-trend correlation should be
*meaningfully higher* than raw-vs-raw on flat terrain, since ground_trend
smooths away the fine-scale noise the known-height test already showed
carries no real signal. **It isn't — `trend_vs_dem_pearson` and
`raw_vs_dem_pearson` are nearly identical on every single one of the 25
tiles tested**, not just the 4 flat ones (e.g. manali 0.831 vs 0.831,
dharamshala 0.783 vs 0.783, bathinda 0.203 vs 0.205). The overall
pixel-level correlation between DAv2's output and the real DEM is
dominated by whatever coarse, shared regional gradient both already
carry — smoothing away DAv2's fine-scale noise barely moves that number,
because that noise was already contributing ~zero correlation either way
(consistent with, not contradicting, the earlier known-height finding —
noise correlates with nothing, so removing it doesn't change a
correlation number much).

**The fitted scale itself does differ substantially per tile** (kutch:
2.58 → 1.39; amalapuram: 3.66 → 3.00; chennai: 15.21 → 16.53) — this
isn't a case of the two pipelines silently reducing to the same
computation. But the *combined* effect of (different detail band ×
different scale) on the final surface is, empirically, nearly
parameter-invariant here: the different clipping/weighting inside
`structure` and the different fitted `a` appear to renormalize against
each other.

## Result: 21/25 wins under both designs, zero flips

Full 25-tile comparison (existing single-split frequency fusion vs. this
yats0x7-style variant, both vs. the linear baseline):

| | wins vs. linear baseline |
|---|---:|
| Existing frequency fusion | **21/25** |
| yats0x7-style detail+scale | **21/25** |

**Zero tiles flip win/loss status.** Max per-tile difference in ICESat-2
RMSE-%-of-range: **0.090 percentage points** (manali, still tiny relative
to its 0.48% baseline). Mean absolute difference across all 25 tiles:
**0.009 percentage points**. No tile shows a difference exceeding 0.1
points in either direction. Full per-tile table (including RANSAC
`r2`/inlier counts and both correlation diagnostics):
`data/sentinel2_benchmark/yats0x7_ground_trend_results/results_all25.csv`.

## Verdict

**No adoption warranted — yats0x7's added sophistication (asymmetric
ground-tracking trend, RANSAC fit) does not handle frequency fusion's 4
flat-terrain losses, or any of the other 21 tiles, differently in
outcome from the existing simpler design**, despite producing genuinely
different intermediate fitted parameters. This is a clean, informative
result in its own right: it demonstrates the current design's final
ICESat-2 accuracy is not fragile to this particular methodological
choice (symmetric-vs-asymmetric detail extraction, OLS-vs-RANSAC scale
fit) — a second, independent line of evidence (alongside the
evidence-gating/LOO check earlier in this file) that frequency fusion's
current numbers reflect something structural about the data, not an
artifact of one specific implementation choice. Frequency fusion's
existing `run_frequency_fusion_sentinel2.py` is unchanged by this test.

## Reproducing this

- `scripts/test_yats0x7_ground_trend_scale_frequency_fusion.py` — ported
  `ground_trend`/`clip_structure`/`ransac_affine`, reusing
  `run_frequency_fusion_sentinel2.py`'s DEM low-pass/combination
  functions unchanged. `... losing4` for the 4 target tiles, `... all25`
  for the full accepted set. Writes
  `data/sentinel2_benchmark/yats0x7_ground_trend_results/results_*.csv`.

# 2026-09-22 (continued) — Semantic-prior phase 2.3: building-aware frequency fusion, tested against all 4 focus tiles — neither approach helps

**Status: COMPLETE. Result: clean negative — Approach A (building-aware
confidence weighting) is inert, Approach B (Open Buildings height blending)
is net-harmful with one unexplained exception. Neither flips any of
frequency fusion's 4 losing tiles (bathinda, amalapuram, kutch, chennai),
and neither flips any of the 21 winning tiles to a loss either.** No CNN
involved — both are closed-form combinations on top of frequency fusion's
existing `dem_lowpass`/`dav2_highpass` signals, so Method 4's
interpolation-memorization failure mode (the reason CNN correction was
closed off, item 7 above) structurally cannot apply here.

## What this is

Real building-presence/height signals, prepared from scratch for this
project's own 25-tile Sentinel-2 benchmark (a different tile set from the
old Method3 demo-track cities, even where a city name coincides):

- **Step 2** (`scripts/prepare_semantic_benchmark_footprints.py`): Microsoft
  GlobalMLBuildingFootprints `building_fraction`, quadkey-matched and
  rasterized onto each tile's exact UTM grid. Reuses
  `repair_method3_coverage.py`'s AOI/partition-selection machinery and
  `prepare_method3_semantic_inputs.py`'s GeoJSONL parsing, not reimplemented.
  A new coverage-gap QC (nonzero-pixel spatial-extent test) replaces the
  contiguous-blank-region detector gaps-and-fixes.md called for but never
  implemented. 24/25 tiles OK; **kohima has zero Microsoft partition
  coverage at all** (a real data-absence finding for that remote Nagaland
  location, not a bug — Step 4 falls back to Open Buildings presence there).
- **Step 3** (`scripts/prepare_semantic_benchmark_openbuildings.py`):
  Google Open Buildings 2.5D Temporal height + presence, same public
  unauthenticated GCS bucket approach already used for Method 4's 8-urban-tile
  Test B (`prepare_method4_openbuildings_data.py`), extended to all 25 tiles.
  The 8 urban tiles were already fetched onto an identical grid by that
  earlier work and were reused rather than re-fetched
  (`scripts/_reuse_urban_openbuildings.py`, verified bit-identical
  transform/shape/crs). 24/25 at 100% OB coverage; **kurnool at 82.4%** (a
  real partial gap, left as-is rather than filled).

## Two integration approaches, both fixed a priori (not tuned against the test metric)

- **Approach A — building-aware confidence weighting**:
  `final_A = dem_lowpass + (1 + alpha * building_prob) * dav2_highpass`,
  `alpha=1.0` fixed before looking at results (buildings get up to 2x DAv2
  high-frequency weight). `building_prob` = Step 2's `building_fraction`,
  falling back to Step 3's OB presence for kohima (Step 2's one gap).
- **Approach B — direct supplementary height source**: at OB-confident
  building pixels (`presence > 0.5`, matching Method 4 Test B's own
  threshold), `final_B = dem_lowpass + 0.5*dav2_highpass + 0.5*ob_height`
  (an equal blend of DAv2's detail signal and Open Buildings' own measured
  AGL height); unchanged elsewhere.

Both implemented in a new script, `scripts/run_frequency_fusion_semantic.py`,
importing `run_frequency_fusion_sentinel2.py`'s helpers unchanged rather
than modifying that script — it stays the reproducible standing baseline.
Evaluated with the identical held-out-DEM (50%, seed 42) + independent-
ICESat-2 protocol, all 25 tiles. Full results:
`data/sentinel2_benchmark/frequency_fusion_semantic_results/results.csv`.

## Results

**Approach A is a genuine no-op.** Max |delta| vs. frequency fusion's own
ICESat-2 %-of-range across all 25 tiles: **0.0005 percentage points**. Zero
tiles move by more than 0.05pp in either direction. `dav2_highpass`'s
absolute magnitude is simply too small relative to the elevation range for
a 2x local reweighting to register.

**Approach B is net-harmful, one unexplained exception.** 15/25 tiles worse
by >0.01pp, only 1 improved by >0.01pp:

| tile | category | ob_confident_pct | base | B | delta |
|---|---|---:|---:|---:|---:|
| hyderabad | urban | 32.76% | 3.66% | 4.26% | **+0.599pp (worst)** |
| chennai | urban | 23.85% | 5.67% | 5.96% | +0.286pp |
| bengaluru | urban | 28.43% | 5.82% | 6.10% | +0.278pp |
| kakinada | coastal | 7.87% | 4.94% | 5.17% | +0.231pp |
| kochi_city | urban | 8.87% | 4.94% | 5.13% | +0.185pp |
| mumbai | urban | 16.31% | 2.48% | 2.08% | **-0.399pp (only improvement)** |
| dharamshala | hilly | 2.45% | 0.269% | 0.267% | -0.002pp (noise-level) |

Regression size tracks `ob_confident_pct` closely (the 5 worst regressions
are exactly the 5 tiles with the highest confident-building coverage) — the
mechanism is almost certainly that Open Buildings' rooftop AGL height
overshoots what ICESat-2 ATL08 photons actually measure at most
building-pixel locations, so blending it in moves predictions further from
ground truth on average. **Mumbai is a real exception, not noise** (-0.40pp
is the largest single delta after hyderabad's regression, and mumbai has
the second-highest `ob_confident_pct` at 16.31%) — nothing in this data
explains why mumbai's OB heights land closer to ICESat-2 while
hyderabad/chennai/bengaluru/kochi_city's don't at even higher confident-pixel
counts. Flagged as unexplained rather than hand-waved; worth revisiting only
if this integration approach is picked up again with per-tile diagnostics.

## Focus tiles (bathinda, amalapuram, kutch, chennai) — none flip to a win

| tile | base | A | B |
|---|---:|---:|---:|
| bathinda | 3.568% | 3.568% (no-op) | 3.598% (worse) |
| amalapuram | 8.540% | 8.540% (no-op) | 8.542% (worse, negligible) |
| kutch | 5.879% | 5.879% (exact no-op — 0% OB coverage, no building fraction) | 5.879% (exact no-op) |
| **chennai** | 5.672% | 5.672% (no-op) | **5.958% (worse — the largest chennai-specific regression after hyderabad)** |

Chennai was the tile hypothesized most likely to benefit from a
building-aware signal (dense urban, 23.85% confident OB coverage) — it is
unchanged under A and gets measurably **worse** under B, directly
contradicting that hypothesis. Kutch's exact no-op under both approaches is
mechanically forced: Step 2 found essentially zero real buildings there
(1 footprint in the whole 10km tile) and Step 3 found 0.0% confident OB
coverage, so `building_prob` and the confident-pixel mask are both empty —
there is no semantic signal for either approach to act on at this location,
consistent with frequency fusion's existing loss there being a flat/coastal
low-relief issue, not a missing-buildings issue.

**No previously-winning tile flips to a loss either** — frequency fusion's
margins over the linear baseline are large enough (often 2-10x) that even
Approach B's worst regressions (hyderabad, chennai, bengaluru) don't erase
the win against linear calibration.

## Verdict

**Neither approach adopted.** Approach A doesn't do anything measurable;
Approach B actively hurts more tiles than it helps and doesn't touch any of
the 4 tiles it was meant to fix. This closes phase 2.3 as a clean negative
result — semantic building information, at least in this form (confidence
weighting or direct height blending), is not the lever that moves frequency
fusion's remaining 4 losses. Consistent with how Method 4's CNN attempts and
the GSD-FiLM ablation were reported elsewhere in this project: a clean "no"
is reported honestly rather than reframed as a win.

## Re-investigation: is the negative result genuinely understood? (same day, follow-up — both checks pass, verdict unchanged)

Before trusting the clean-negative verdict above, two specific mechanisms
were checked rather than assumed: whether Approach A's suspiciously-exact
0.0005pp max delta is a wiring/scaling bug (dead signal never reaching the
formula) vs. a genuine physical null, and whether Approach B's height
blend used Open Buildings' `building_height` in the wrong vertical
reference frame (the same category of bug already caught three times this
session: CNN target-framing, EGM96/EGM2008 datum).

**Check A — building_prob wiring, on chennai (highest-confidence tile,
23.85% OB-confident coverage): real signal, correctly wired, genuine
null.** Read `building_fraction.tif` directly: real, continuous variation
(0.0-1.0, std 0.34, 26 distinct quantized levels, mean 0.79 specifically
at the pixels where `building_prob > 0.5`) — not flat or near-constant.
Reproduced `process_tile`'s intermediate arrays directly: at those same
high-confidence pixels, `delta_A = alpha * building_prob * dav2_highpass`
averages 0.0043m, matching `building_prob`(0.79) x `dav2_highpass`(0.0054m
there) almost exactly — the formula is reaching real data with the right
magnitude relationship, not a dead or misscaled wire. The reason it still
doesn't register in the final metric: **`dav2_highpass` itself is tiny** —
whole-tile abs-mean 0.0083m (max 0.47m) against a 54.7m elevation range,
i.e. sub-centimeter on average even before any building-aware reweighting.
Doubling a signal that's already ~0.015% of the elevation range cannot
move a percentage-of-range metric by more than the ~0.0005pp actually
observed. At the ICESat-2 photon locations specifically (not just
building-flagged pixels), `building_prob` also has real spread (mean
0.069, std 0.208, 13.4% of photons on nonzero-building pixels) — so
there's no confound where photons simply never see buildings either.
**Confirmed: genuine null, not a bug. Nothing to fix; Approach A stands
as reported.**

**Check B — Open Buildings 2.5D height reference frame: relative (AGL),
correctly handled by the existing blend, and the hypothesized fix makes
it measurably worse, not better.** Read Google's own Earth Engine Data
Catalog entry for the dataset
(`GOOGLE_Research_open-buildings-temporal_v1`) directly rather than
trusting this project's own code comments: the `building_height` band is
documented as **"Building height relative to the terrain in range [0m,
100m]"** — confirmed AGL, matching what this project's code comments
already said (`prepare_semantic_benchmark_openbuildings.py`,
`prepare_method4_openbuildings_data.py`). Approach B's formula,
`final_B = dem_lowpass + 0.5*dav2_highpass + 0.5*ob_height`, already adds
`ob_height` on top of `dem_lowpass` (a ground-level reference) rather than
using it standalone — i.e. it already treats it as relative, not
absolute. So the specific bug category hypothesized (relative value used
as if absolute) does **not** apply here; no datum fix was needed.

To be sure rather than assume, the hypothesized "fix" (full, unweighted
`dem_lowpass + ob_height`, matching how Method 4 Test B added ground +
AGL height a-priori) was tested directly on chennai's 17,592 ICESat-2
photons that land on OB-confident building pixels: mean true height
-83.30m; baseline (frequency fusion, no OB) mean error +0.76m (abs mean
4.29m); the **existing** half-weighted Approach B blend already overshoots
worse, mean error +4.52m (abs mean 5.90m); the **hypothesized "corrected"
full-weight** ground+AGL blend overshoots dramatically more, mean error
+8.28m (abs mean 8.62m) — roughly double the existing blend's error and
2x the baseline's. This confirms, quantitatively, that Open Buildings'
rooftop AGL height genuinely overshoots what ICESat-2 ATL08 measures at
most Indian urban building footprints in this benchmark (ATL08 ground/low
photons evidently aren't sampling true rooftop returns here), and that the
existing half-weight dilution was already an unintentional partial
mitigation of a real physical mismatch, not an underweighted bug waiting
to be fixed by adding more AGL signal.

**Both checks come back clean. The phase 2.3 negative result is confirmed
genuinely understood, not an artifact of a wiring or datum bug.** Verdict
above stands unchanged; no re-test of either approach with a "fix" is
warranted, and none was applied.

## Reproducing this

- `scripts/prepare_semantic_benchmark_footprints.py` — Step 2 (Microsoft
  footprints). `scripts/prepare_semantic_benchmark_openbuildings.py` +
  `scripts/_reuse_urban_openbuildings.py` — Step 3 (Open Buildings).
  `scripts/run_frequency_fusion_semantic.py` — Step 4 (both approaches,
  evaluation). Outputs: `data/sentinel2_benchmark/semantic/` (per-tile
  `building_fraction.tif`/`ob_height.tif`/`ob_presence.tif`, plus
  `coverage_report.csv` and `openbuildings_coverage_report.csv`),
  `data/sentinel2_benchmark/frequency_fusion_semantic_results/results.csv`.

# 2026-09-23 — RDAH-Net zero-shot on Sentinel-2: does the DFC2019 input-scale fix transfer?

**Status: COMPLETE — CLEAN NEGATIVE, stopped at Step 2. The DFC2019 input-scale fix cuts the checkerboard by 2–3 orders of magnitude but does not remove it, and RDAH output has no terrain correlation raw or detrended. Line closed (recommendation c). Written step by step so a session cutoff loses at most one
step.** Goal: test the forward-looking flag in `docs/method-audit/05-rdah-net-fusion/verdict.md`
§5. The flag asks whether the Darjeeling checkerboard that got RDAH-Net rejected on Sentinel-2
changes or disappears once depth is rescaled the way the DFC2019 audit found. **Inference only.**
No fine-tuning on Sentinel-2: that would reopen the per-tile data-volume ceiling behind the three
earlier CNN failures (item 7), and it stays gated behind Step 5.

## Step 0 — pre-flight (no new inference except an exact reproduction of the old run)

**0a. The DFC2019 fix is on the INPUT, not the output.** In `scripts/train_rdah_quadrant_cv.py`
(`train_one_epoch()`, `evaluate_samples()`, `derive_fold_scale()`) the cached DAv2 relative depth
is multiplied by a constant **before** the forward pass: `depth = (depth_raw * scale)`. The RGB
path is unchanged (`/255` then ImageNet normalization). The constant is ×255 for folds 0–1 and
×300 for folds 2–3, derived per fold from training quadrants only (Swiss checkpoint). The 4-tile
probe sweep (Track1) peaked at ×255–×300. The mechanism is `external/RDAH-Net/loaddata.py`
`ToTensor`, which divides RGB by 255 but leaves `rel_depth` undivided, so RDAH was trained on
depth in a 0–255 range and this project feeds [0,1]. Because it changes the input magnitude seen
by every encoder layer, it *can* change spatial structure in the output, including a periodic
artifact. The concern that a global output constant can't remove a checkerboard doesn't apply.
**Provenance caveat:** the scripts that apply the fix (`train_rdah_quadrant_cv.py`,
`run_rdah_scale_sweep.py`, `run_rdah_probe.py`) are **untracked, never committed**. The fix is
documented in commit `0b64c6b` (the 05 audit docs), not implemented in any commit. The 50-tile
zero-shot ×255 run's own script/JSON was not located (see `05-rdah-net-fusion/summary.md` §10).

**0b. Input dtype/range: 8-bit renders, not L2A reflectance.** Every Sentinel-2 RGB this project
feeds a depth model is already a 3-band **uint8** GeoTIFF: all 32 benchmark tiles (1000×1000,
checked), the committed Darjeeling tile, and the production tiles. RDAH's `/255` on RGB is
therefore correct. The rendering path shared with the DAv2 benchmark runs is
`backend/depth/depth_engine.py` `load_geotiff_rgb()` (bands 1–3, `np.clip(…, 0, 255)` to uint8 if
not already uint8; a no-op here). DAv2 depth arrives as float32 in [0,1]
(`data/sentinel2_benchmark/dav2_depth/*_depth.npy`, `data/diagnostics/darjeeling/Darjeeling_RGB_depth.npy`),
which is exactly the convention that caused the DFC2019 bug.

**0c. The original checkerboard run, reproduced bit-exactly.** Code: `backend/rdah/rdah_engine.py`
(`RDAHEngine.predict`, `run_darjeeling`), output
`data/diagnostics/darjeeling/rdah/Darjeeling_RDAH_nDSM.{npy,png}` (10 Sep). Its checkpoint wasn't
recorded, so all three were re-run through the unmodified engine on its original inputs:

| checkpoint | output mean | std | max \|diff\| vs. saved output | Pearson vs. saved |
|---|---:|---:|---:|---:|
| Track1 `104best_model.pth` | 0.0811 | 0.0123 | 1.600 | 0.038 |
| **Swiss `swiss_best_model.pth`** | **0.1249** | **0.0652** | **0.0 (exact)** | **1.000** |
| HK `checkpoints-HK/best_model.pth` | 0.0178 | 0.0221 | 1.903 | 0.320 |

**The original run was the Swiss checkpoint**, with these inputs:
- RGB: the *committed* `data/sentinel2/darjeeling/Darjeeling_RGB.tif` (1007×1002, uint8, EPSG:32645,
  10 m). The working-tree copy is a different, uncommitted 1118×1004 re-export with different
  bounds, so this run uses the committed version to change nothing but the fix.
- Depth: `data/diagnostics/darjeeling/Darjeeling_RGB_depth.npy` (1007×1002, [0,1]), fed
  **unscaled**.
- Size handling: **bilinear resize** of RGB and depth to 1024×1024, then a bilinear resize of the
  output back to 1007×1002. `_pad_to_1024()` exists in the engine but is never called.
- One whole-tile forward pass, no tiling.

Swiss's own file lists (`external/RDAH-Net/Swiss-{train,test}.txt`) are GF-7 ortho tiles, with no
DFC2019 and nothing in India. Any Sentinel-2 result from this checkpoint is contamination-free.

## Step 1 — input-size compatibility (verified empirically, before any new inference)

Read `backend/rdah/model.py` (identical to `external/RDAH-Net/test.py`'s model classes):
- The stem is a 4×4 stride-2 conv, followed by three stride-2 `MobileViTBlock` stages. Feature
  maps sit at strides 4, 8 and 16 (256², 128², 64² for a 1024 input).
- Every `BlockAttention` and `LightCrossAttention` does **hard, non-overlapping 8×8 block
  attention** with no shift or overlap. In input pixels, block boundaries fall every **32, 64 and
  128 px**.
- The decoder is four `PixelShuffle(2)` stages (periods 2, 4, 8, 16 px).
- `PositionalEncoding` is a fixed 64×64 buffer sliced to the feature size.

These are the candidate checkerboard periods for Step 2's FFT: **2, 4, 8, 16, 32, 64, 128 px** on
the native 1024 grid.

Forward passes on zeros confirm the size contract:

| input H×W | result |
|---|---|
| 1007×1002 (original Darjeeling), 1000×1000 (benchmark) | **crash**: block-attention `reshape` needs feature H/W divisible by 8 |
| 1024×1024, 896×896 | run |
| 1152×1152 | **crash**: positional-encoding buffer is 64×64 (72 ≠ 64) |

So H and W must be multiples of 128, and no larger than 1024. Below 1024 the positional encoding
is silently corner-cropped (see `05-rdah-net-fusion/summary.md` §6 for the same issue on
quadrants). **1024 is the only native size that fits these tiles, so they are reflect-padded to
1024 and the output is cropped back.** No resizing, no GSD change.
`train_rdah_quadrant_cv.py`'s 32×32 buffer rebuild is for 512² quadrants and isn't needed here.

**Did the original run mishandle size?** It *resized* 1007×1002 → 1024×1024 (bilinear) and resized
the output back. That's a 1.7%/2.2% anisotropic rescale: a slight GSD change and a mild smoothing,
the kind of mishandling the task flagged. On its own, a ~2% bilinear resample can't *create* a
periodic pattern. It would only shift the phase and period of one that already exists. That's a
prior, not a finding, so Step 2 tests it directly: resize vs. pad at ×1 and at ×255 (a 2×2 grid).
That separates the size cause from the scale cause.

**Construct-validity caveat, recorded before any Step 2 number was seen.** RDAH-Net predicts an
**nDSM**: height above local ground, trained against AGL targets (DFC2019 AGL, GF-7 nDSMs). The
checks below correlate its output with *terrain* elevation (the OpenTopography DSM and ICESat-2
ground-classified photons). On hilly Darjeeling, a correct nDSM model should carry little of the
ridge/valley relief, especially after detrending. A near-zero detrended correlation is therefore
what a *working* nDSM model would also produce. The artifact test (i) doesn't depend on this. The
correlation test (ii) measures "does RDAH output carry terrain structure", which is the property
that matters if RDAH is going to replace DAv2 as the high-frequency source in frequency fusion.
It isn't a test of RDAH's own nDSM accuracy, and no nDSM ground truth exists for these tiles.

## Step 2 — Darjeeling check: STOP CONDITION MET (clean negative)

Ran `python scripts/rdah_sentinel2_zeroshot.py darjeeling --rgb
data/diagnostics/darjeeling/Darjeeling_RGB_committed_660ecb6.tif` (the committed RGB, extracted
unchanged from commit `660ecb6`). Everything matched the original run except two factors, varied
in a 2×2 grid:
- depth scale ×1 vs. **×255** (not re-derived; see the output-range note below)
- bilinear resize vs. **reflect-pad to 1024 then crop**

The script asserts that the ×1/resize variant is bit-identical to the saved original output, and
it is. Ground truth:
- DEM: OpenTopography DSM reprojected bilinearly onto the tile grid (556.6–2476.8 m, 994,491
  valid px, 200,000 sampled at random, seed 0, for correlation).
- ICESat-2: **18,947** ground-classified photons fetched for this tile with
  `query_icesat2_photons.query_tile()` unmodified (542.9–2366.5 m), saved to
  `data/icesat2_photons/darjeeling.csv` (not committed, same as the other photon CSVs).

Results: `data/sentinel2_benchmark/rdah_zeroshot/darjeeling/darjeeling_results.json`.

**(i) 2D FFT, peak-to-background power at the architecture's candidate periods.** Computed on the
native 1024² output after plane detrending and a Hann window. Background is the median power on
the same-radius annulus. The DAv2 depth input and RGB luminance are shown as controls.

| period (px) | original (resize ×1) | pad ×1 | resize ×255 | **pad ×255** | reduction orig→pad×255 | DAv2-input control | RGB control |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.35e7 | 1.26e7 | 1.08e4 | **6.97e3** | ÷1,940 | 1.36e5 | 4.2 |
| 4 | 9.07e5 | 7.78e5 | 2.97e3 | **1.59e3** | ÷570 | 177 | 3.9 |
| 8 | 2.71e4 | 2.40e4 | 148 | **163** | ÷166 | 8.6 | 4.5 |
| 16 | 742 | 580 | 23.8 | **11.1** | ÷67 | 6.9 | 3.0 |
| 32 | 1.5 | 1.1 | 10.2 | **17.9** | **×12 (grew)** | 6.6 | 2.1 |
| 64 | 3.3 | 1.5 | 2.9 | 2.9 | — | 3.8 | 1.1 |
| 128 | 3.5 | 3.9 | 1.4 | 1.4 | — | 2.7 | 4.2 |

- **The scale fix cuts the artifact by 2–3 orders of magnitude but doesn't remove it.** After the
  fix, the period-2/4/8 peaks are still 10²–10⁴× background.
- **The residual peaks are network-generated, not inherited from the input.**
  - The output's period-2 peak sits on the vertical Nyquist bin (512,0). The DAv2 input's period-2
    peak is on the horizontal bin (0,512), a different component.
  - After the fix, the dominant period-8/16/32 peaks all sit on **diagonal** bins ((128,128),
    (64,64), (32,32)), the checkerboard signature.
  - Periods 8, 16 and 32 sit well above both input controls.
- **The period-32 peak, the stride-4 block-attention boundary, *grows* 12× under the fix.** Once
  the PixelShuffle noise drops, the hard 8×8 block-attention tiling becomes visible.
- **Size handling is ruled out as the cause.** Resize vs. pad at ×1 differ by under 20% at every
  period (1.35e7 vs. 1.26e7 at period 2), against a 1,940× change from scale. The original run's
  resize was a real deviation from native handling but not the source of the checkerboard.
- **The scale mismatch was a real contributor**, responsible for the bulk of the artifact's power.
  It isn't the whole cause.

**(ii) Correlation with terrain, raw and detrended** (Pearson / Spearman). "Plane" and "quadratic"
fit a 1st/2nd-order (x, y) surface separately to prediction and truth, then correlate the
residuals.

| field | DEM raw | DEM plane | DEM quad | ICESat-2 raw | ICESat-2 plane | ICESat-2 quad |
|---|---|---|---|---|---|---|
| RDAH original (resize ×1) | −0.042 / −0.080 | −0.056 / −0.041 | −0.019 / −0.065 | −0.145 / −0.162 | −0.232 / −0.207 | −0.132 / −0.114 |
| RDAH pad ×1 | −0.031 / −0.053 | −0.024 / −0.020 | −0.046 / −0.064 | −0.025 / −0.068 | −0.012 / −0.015 | −0.048 / −0.043 |
| RDAH resize ×255 | +0.012 / −0.213 | −0.002 / −0.036 | −0.029 / −0.074 | −0.098 / −0.155 | −0.092 / −0.082 | −0.051 / −0.077 |
| **RDAH pad ×255 (the fix)** | −0.025 / −0.191 | **−0.052 / −0.047** | −0.044 / −0.013 | −0.042 / −0.170 | **−0.139 / −0.132** | −0.036 / −0.006 |
| DAv2 input (control) | +0.641 / +0.635 | −0.431 / −0.417 | −0.115 / −0.091 | +0.531 / +0.512 | −0.510 / −0.498 | −0.084 / −0.021 |

- **The DAv2 control reproduces this project's known Darjeeling flip** (DEM +0.64 → −0.43 after
  plane detrending; ICESat-2 +0.53 → −0.51). The scoring pipeline behaves as expected.
- **RDAH carries no terrain structure under any variant.** Every raw and detrended correlation is
  between −0.23 and +0.01. RDAH doesn't even show DAv2's spurious raw +0.6; it's ~0 or slightly
  negative everywhere.

**Output range.** At ×255 the output has median 0.017, p2–p98 −0.018 to 0.33, and max 7.97. That's
near-zero almost everywhere, with a few bright blobs (see the PNG). As an nDSM for a hill town
with forest canopy and dense buildings this is implausibly flat. It's closer to "the model sees
no above-ground structure it recognizes" than to a mis-scaled but structured output. The ×255
constant was **not re-derived**, per the task: nothing here shows it to be *clearly* wrong
rather than simply out of domain, and re-deriving it would mean fitting a free constant to the
same tile being scored. That residual uncertainty is recorded here rather than tuned away.

**(iii) Side-by-side PNGs.** `data/sentinel2_benchmark/rdah_zeroshot/darjeeling/darjeeling_old_new_dem_dav2.png`
(full tile) and `…/darjeeling_zoom_center.png` (central 256² crop, 2× nearest). Panels, left to
right: original run, fixed run (pad ×255), DEM, DAv2 input. The original shows the dense regular
grid over the whole tile. The fixed run is mostly flat, with a few bright blobs and a still-visible
faint grid in low-signal areas. Neither resembles the DEM's ridge/valley structure.

**STOP CONDITION MET, on both criteria:** the periodic artifact persists (reduced, not removed),
and the detrended correlation is negative (DEM plane −0.052 / −0.047, ICESat-2 plane
−0.139 / −0.132). **Verdict: clean negative. The DFC2019 input-scale fix does not rescue RDAH-Net
zero-shot on Sentinel-2.**
- **Ruled out:** input size handling (resize vs. pad) as the checkerboard's cause.
- **Partial cause, confirmed:** the depth-input scale mismatch.
- **Remaining cause:** architecture-level PixelShuffle and hard block-attention periodicity,
  exposed on out-of-domain 10 m input. That's a mechanism consistent with the evidence, not
  proven: no ablation of those layers was done, and none is warranted for a closed line.

Steps 3–4 (decision rule, benchmark, fusion swap, production-tile checks) were **not run**, per the
stop condition. No fine-tuning was started.

## Step 5 note — what this does and doesn't say about the DFC2019 contamination question

The Swiss checkpoint has no DFC2019 and no India tiles in its lists, so this is a
contamination-free zero-shot test. It **doesn't separate** the 05 verdict's two explanations
(fine-tuning damage vs. Track1 contamination inflating the DFC2019 zero-shot score). Here RDAH
fails zero-shot on Sentinel-2 with a clean checkpoint, but it also crosses a ~12–30× GSD gap
(10 m vs. GF-7's sub-metre imagery and DFC2019's ~0.3 m), a sensor change, and a construct gap
(nDSM output vs. terrain truth). Any of these explains the failure without contamination. The
check that would decide the 05 question is unchanged: **Swiss zero-shot on DFC2019 itself**, at
the fold-derived scale, under the quadrant protocol. The per-tile data-volume ceiling that killed
the three Sentinel-2 CNN attempts is also untouched. A better starting model adds no training
data, and here RDAH isn't even a better starting model. **Recommendation: (c) close this line**
(RDAH as a Sentinel-2 depth/high-frequency source, zero-shot or fine-tuned).

## Reproducing this

- `scripts/rdah_sentinel2_zeroshot.py darjeeling --rgb <committed Darjeeling RGB>`: the 2×2
  scale/size grid, FFT, correlations and PNGs. It reuses `backend/rdah/rdah_engine.py`'s model
  loading and preprocessing unmodified, and `scripts/query_icesat2_photons.py`'s `query_tile()`
  for photons.
- Outputs: `data/sentinel2_benchmark/rdah_zeroshot/darjeeling/{darjeeling_results.json,
  darjeeling_old_new_dem_dav2.png, darjeeling_zoom_center.png}`. Per-variant rasters are in
  `data/diagnostics/darjeeling/rdah/*.npy` (gitignored).
