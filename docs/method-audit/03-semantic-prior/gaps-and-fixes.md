# Semantic prior — Gaps & Fixes

_One issue per subsection. Only new computation run: recomputing raster
statistics from files already on disk (twice, to reproduce the note's exact
numbers two independent ways) and one visualization pass to check a spatial
claim. No model was refit, no experiment was rerun._

## 1. The note's own "refer to generated stats files instead" points at files that don't exist

**Problem:** `raw-notes.pdf` §3 says the QC PDF "contains mock numerical
values, don't refer to values, refer to generated stats files in the
mentioned folders instead." No such stats file (`.csv`/`.json` with per-city
mean/p95/etc.) exists anywhere under
`data/sentinel2/semantic_sources/globalml_building_footprints/` or any
city's `data/sentinel2/<city>/semantic/` folder — only raw `.tif` rasters.

**Checked:** Confirmed via `find` across both locations; only
`coverage_audit.csv/json` and `selected_partitions.csv` exist (partition
selection metadata, not building-fraction statistics).

**Outcome:** The redirection is a dead end as written. Separately, the
warning itself turns out to be wrong for the two numbers actually cited (see
§2/summary.md §2) — reproduced exactly from the real raster via the QC
script's own resampling logic, and confirmed present verbatim in the PDF
itself. **Fix:** either generate the missing per-city stats file (cheap: a
few lines added to `make_method3_qc_pdf.py` to also dump `add_stats_text`'s
inputs to CSV), or drop the "mock values" warning since it doesn't hold for
this PDF as it currently stands.

## 2. "Coverage gap" justification isn't visible in the aggregate numbers the note/QC page show

**Problem:** The note excludes Kolkata/Delhi/Mumbai/Kochi for "large/
inconsistent footprint coverage gaps," but the QC page's own printed
mean/p95/>0% numbers for Kolkata (0.2076 / 0.7632 / 56.65%) are comparable to
or better than several *retained* cities (Bardhaman 0.0743/0.60/17.6%,
Sundarbans 0/0/0%). Reading only the numbers, Kolkata looks fine.

**Checked:** Rendered `building_fraction.tif` for all 10 cities at a common
downsample (see `summary.md` §10). Kolkata, Delhi, and Mumbai each show a
large, hard-edged, contiguous empty region (a rectangular void or band) —
real, visible in the pixels, but invisible in a single mean/p95/percent
summary because those aggregate over the whole tile including both the
covered and uncovered regions.

**Outcome:** The exclusion decision itself checks out for
Kolkata/Delhi/Mumbai (visually confirmed contiguous voids, and for these
three the currently-saved raster is the systematically-repaired, maximal-
partition-set version — see §3 below — so the gap isn't an artifact of an
incomplete initial download either). Kochi is weaker evidence (see §3).
**Fix:** if this audit trail matters going forward, the QC info panel should
report something like "% of tile within N px of a coverage-partition
boundary" or a simple contiguous-blank-region detector, not just mean/p95 —
those numbers alone can't distinguish "sparse but uniform" from "half the
tile is a void."

## 3. Bengaluru's semantic raster is currently missing — a live reproducibility gap, not just a stale note

**Problem:** The note lists Bengaluru under "Retained" with an implied QC
pass. `data/sentinel2/bengaluru/semantic/` exists but is completely empty —
no `building_fraction.tif`, no derived density/edge/distance rasters.

**Checked:** File-mtime forensics across all 10 cities' `building_fraction.tif`
(`summary.md` §11) show two waves: an original run (~03:05–03:08) for
darjeeling/hyderabad/jaipur/kochi/sundarbans, and a `repair_method3_coverage.py`
run (~03:57–04:14) for kolkata/bardhaman/delhi/mumbai. Bengaluru's `semantic/`
directory was created at 04:14:12 — inside the repair wave, right where
`CITY_ORDER` places it (5th, immediately after mumbai) — but no raster was
ever written into it, and the raw footprint partition download did complete.
This is consistent with the repair run being interrupted mid-way through
Bengaluru specifically. Cross-checked against
`data/sentinel2/method3_semantic_QC.pdf` (generated 03:31:06, *before* the
repair wave started): it has a full, real, non-empty Bengaluru page (mean =
0.2414, p95 = 0.7840) — so a real file existed at that point and was later
lost, not merely never generated.

**Outcome:** As the repo stands right now, Method 3 (or anything reusing this
building-footprint source) cannot be rerun or extended for Bengaluru without
first regenerating `building_fraction.tif`. This doesn't affect the DFC
Test-1/Test-2 verdict (Bengaluru was never part of that; see summary.md §8),
but it does mean the Sentinel-side "6 retained cities" claim is currently
only 5-cities-reproducible. **Fix:** rerun
`python scripts/repair_method3_coverage.py --cities bengaluru --overwrite-raster`
(or `prepare_method3_semantic_inputs.py --cities bengaluru --skip-downloads`,
since the raw partition is already present) to regenerate the missing
raster — this is exactly the "genuinely missing from existing output" case
worth new computation for, but it's a data-regeneration step outside this
audit's scope, not something to silently run here.

## 4. Sundarbans "useful near-zero-building case" is asserted with no supporting evaluation

**Problem:** The note frames Sundarbans' 0% building coverage as "useful,"
implying it plays some role in validating Method 3's behavior at the
building-free extreme.

**Checked:** `evaluate_prior.py` and `evaluate_prior_spatial_cv.py` (the only
two scripts that actually run the Method-3 regression) reference only DFC
tiles; grepped both for every Sentinel city name — zero hits. No per-region
breakdown, no Sentinel-side regression run, exists anywhere in this repo.

**Outcome:** The 0%-coverage fact is real (confirmed in the QC PDF), but
"useful" is unearned — nothing in the repo demonstrates usefulness. **Fix:**
either add a one-line caveat to the note ("not yet evaluated") or, if this
matters for the demo narrative, actually run a Method-3-style regression on
Sundarbans once a genuine metric-elevation source exists for it (right now
Sundarbans has no independent AGL ground truth the way DFC tiles do, so this
may not even be testable in the same way).

## 5. Documentation ordering makes the HOTOSM model's target ambiguous, though the code is unambiguous

**Problem:** Note §2 lists the Sentinel/GlobalMLBuildingFootprints
manifest path, then immediately below it lists "HOTOSM DINOv3-S building
model: `models/semantic/hotosm_dinov3s_buildings/model.onnx`" — visually
suggesting the model belongs to the Sentinel pipeline.

**Checked:** `scripts/generate_dfc_building_prior.py` is the only script that
loads that `.onnx` file, and it runs exclusively on
`data/dfc2019/raw/RGB/Track1-RGB/`. `prepare_method3_semantic_inputs.py`
(the Sentinel pipeline) never touches the model except to optionally
download it for the other branch — its own docstring says so explicitly:
"the segmenter is for the DFC/VHR branch, NOT for 10 m Sentinel-2."

**Outcome:** No code inconsistency — the note's plan text (§1) already says
this correctly ("DFC2019 benchmark: derive Pb ... using an off-the-shelf
building segmenter"), it's only §2's paths list that reads ambiguously by
proximity. **Fix:** cosmetic — move the HOTOSM model-path line under a "DFC
building segmenter" sub-heading instead of directly under the
GlobalMLBuildingFootprints paths.
