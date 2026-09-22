# DEM-stat anchoring — Gaps & Fixes

_One issue per subsection: problem, what was checked/run, outcome. No refits
or reruns performed where saved output already existed — only pulled and
re-aggregated existing files, plus reran the two cheap listing scripts
(`inspect_dfc2019_pairs.py`, which just walks the filesystem)._

## 1. The note's own city split doesn't add up to its own total

**Problem:** Section 2 states "RGB/AGL matching pairs: 2783 (JAX: 1014, OMA:
1768)". 1014 + 1768 = 2782, not 2783.

**Checked:** Reran `scripts/inspect_dfc2019_pairs.py` against the actual raw
files. Output: `JAX: 1015, OMA: 1768` — which does sum to 2783.

**Outcome:** The note's JAX count (1014) is off by one; the real value is
1015. Cosmetic transcription error, doesn't affect anything downstream (the
50-tile benchmark and 200-tile calibration set were built from the real file
listing, not from the note's stated number).

## 2. The held-out evaluation the note wanted was actually run, but never reported

**Problem:** `raw-notes.pdf` Section 5 states the goal ("apply the same to
the 50 held-out tiles and calculate MAE/RMSE/Pearson/Spearman") and Section 6
ends still saying it's outstanding ("We must test both models on the
untouched 50-tile benchmark"). The note never says whether this ran, and
lists no script or result file for it.

**Checked:** Found `scripts/evaluate_dav2_calibration.py` (present in the
repo, untracked in git, not mentioned anywhere in the PDF) and its output
directory `data/dfc2019/experiments/dav2_calibration/evaluation/`, which
contains per-tile predictions, per-tile metrics, and summary CSVs, all
timestamped after the calibration models were fit. Read the script to confirm
it evaluates against `dav2_baseline/manifest.csv` — the exact 50-tile
benchmark — and applies each already-fitted mapping without refitting on
those tiles.

**Outcome:** No new computation needed; this result already exists and is
real. It answers the note's own open question — see `verdict.md`. The gap is
purely in the write-up: the PDF is stale relative to the repo.

## 3. File path in the note doesn't match any file that exists

**Problem:** Section 5 says the 200-tile calibration manifest is written to
`data/dfc2019/experiments/dav2_calibration/calibration_manifest.csv`.

**Checked:** `scripts/build_calibration_manifest.py` (the script the note
says produced it) actually hardcodes its output to
`data/dfc2019/experiments/dav2_baseline/calibration_manifest.csv` — a
different directory. Neither path exists in the repo today. What does exist,
and is what `fit_dav2_calibration.py` / `run_dav2_batch.py` actually consumed,
is `data/dfc2019/experiments/dav2_calibration/manifest.csv` (200 rows, matches
the expected schema and tile count).

**Outcome:** Content-wise this checks out (right tile count, right schema, no
overlap with the benchmark — see #4), so the calibration pipeline itself
isn't broken. But as literally described, the audit trail isn't reproducible
from the note alone — the manifest was evidently renamed/moved by hand at
some point outside of what any script does. Documentation gap, not a
methodology gap.

## 4. No leakage between the 200-tile calibration set and the 50-tile benchmark — verified good

**Checked (per explicit request):** Diffed tile IDs between
`dav2_baseline/manifest.csv` (50 tiles) and `dav2_calibration/manifest.csv`
(200 tiles) directly. Result: **0 overlapping tile IDs**, out of 50 unique
and 200 unique respectively. This matches `build_calibration_manifest.py`'s
own explicit filter (`if tile_id in benchmark_ids: continue`) and its
in-script `assert` of zero intersection before writing the file.

**Outcome:** Not a gap — flagging this because it was one of the things
specifically asked to be checked, and it holds up. The held-out evaluation in
gap #2 is a genuine held-out test, not leaked training data re-evaluated.

## 5. Correlation metrics can't discriminate calibration quality, and the note's language obscures this

**Problem:** Section 6 treats "test both models on the untouched 50-tile
benchmark" as if it will reveal which calibration (linear vs isotonic) is
better, using Pearson/Spearman as the implicit yardstick.

**Checked:** A positive-slope affine transform (the linear mapping) is
mathematically guaranteed to leave Pearson and Spearman *unchanged* relative
to raw DAv2 — this is a property of the correlation coefficients themselves,
not an empirical finding. The held-out data confirms it exactly:
`global_linear` Pearson mean = 0.53945, identical (to 4 decimal places) to
the raw baseline Pearson mean of 0.53945 computed in Section 2. Only the
isotonic (genuinely nonlinear) mapping could move Pearson at all, and it only
moved it from 0.539 to 0.556 while Spearman stayed flat (0.4460 → 0.4461).

Notably, `workflow_report.md` §8 — written *before* the calibration stage,
as part of the earlier baseline investigation — already states this exact
property correctly ("A positive affine mapping... does not change Pearson or
Spearman... A strictly monotonic mapping preserves rank ordering, so Spearman
should remain essentially unchanged") and even predicts the correct
interpretation: "If Spearman remains low after calibration, then the problem
cannot be solved by a global monotonic mapping alone."

**Outcome:** Not a bug in any script — the calibration and evaluation code is
consistent with this. The gap is that `raw-notes.pdf` Section 6 doesn't carry
forward its own project's established framework (from an earlier report in
the same experiment lineage) when describing what the held-out test would
prove, making it read as more open-ended than it actually is. See
`verdict.md` — this framework is exactly what makes the held-out MAE/RMSE
numbers (not Pearson/Spearman) the metrics that actually matter here.

## 6. Minor, explained anomalies in the saved CSVs (not gaps in the conclusions)

- `dav2_baseline/metrics.csv`: `JAX_004_016` has empty `agl_min/max/mean/std`
  fields but valid `pearson`/`spearman`/`status=ok`. Doesn't affect any
  reported aggregate (which only uses pearson/spearman), but is a
  data-quality wrinkle worth fixing if this manifest is reused later.
- `dav2_baseline/diagnostic_summary.csv`: `OMA_144_030` (AGL max only 1.14 m)
  has empty `pearson_gt1/gt2/gt5` cells — expected, since almost no pixels in
  that tile clear those thresholds. `JAX_004_016` has empty
  `pearson_smooth`/`spearman_smooth`. `workflow_report.md`'s aggregate means
  for those columns are silently computed over n=49 rather than n=50; I
  reproduced the same n=49 means exactly, so the reported numbers are
  correct, but the report doesn't state that n dropped for those two rows.
