# DEM-stat anchoring — Summary

_Sourced from the actual files in `data/dfc2019/` and `scripts/`, verified by
re-running the lightweight scripts and re-aggregating the saved CSVs/JSONs
myself — not taken from `raw-notes.pdf` at face value._

## 1. Dataset

DFC2019 Track 1 (WorldView-3, JAX + OMA cities), raw files under
`data/dfc2019/raw/{RGB,Truth}/`. Running `scripts/inspect_dfc2019_pairs.py`
today reproduces: **2,783 matched RGB/AGL pairs** — **JAX: 1,015, OMA: 1,768**
(1,015 + 1,768 = 2,783; the note's own city split doesn't sum to its own
total — see gaps-and-fixes.md #1).

## 2. Frozen-DAv2 baseline (50-tile benchmark)

`scripts/build_dfc2019_manifest.py` selected a 50-tile subset
(`dav2_baseline/manifest.csv`), stratified by AGL vertical complexity.
`scripts/evaluate_dfc2019_dav2.py` ran frozen DAv2-Large (`config.json`
confirms `model.frozen: true`) on each tile and wrote `metrics.csv`.

Re-aggregating `metrics.csv` myself reproduces the note's table exactly:

| Metric | Mean | Median | Std |
|---|---:|---:|---:|
| Pearson | 0.5394 | 0.5899 | 0.2188 |
| Spearman | 0.4460 | 0.4824 | 0.2050 |

| City | Pearson | Spearman |
|---|---:|---:|
| JAX (n=26) | 0.5995 | 0.5114 |
| OMA (n=24) | 0.4744 | 0.3750 |

Also confirmed against `reports/baseline_summary.json`: 14/50 tiles ≥0.70
Pearson, 19/50 <0.50.

## 3. `JAX_022_009` diagnostic

`JAX_022_009` (Pearson 0.1675, Spearman 0.1508) is genuinely the lowest-Pearson
**JAX** tile — confirmed against `reports/city_summary.csv`
(`JAX pearson_min == 0.1674550538816038`, exact match). Four OMA tiles score
even lower overall (see `best_worst_tiles.csv`), including `OMA_144_030`
(Pearson **-0.047**, AGL max only 1.14 m — an almost-flat tile).

`scripts/diagnose_dfc_tile.py` output for `JAX_022_009`
(`diagnostics/JAX_022_009/correlation_diagnostics.csv`) matches
`workflow_report.md`'s table exactly: filtering out low-AGL pixels makes
correlation *worse* (down to -0.084 at AGL>5m), and Gaussian smoothing barely
moves it (0.1675 → 0.1675). The strong control tile `JAX_004_014` behaves
oppositely (0.8143 → 0.8320 under smoothing). Scaled to all 50 tiles via
`diagnose_dfc50.py` → `diagnostic_summary.csv`, I reproduced every aggregate
in `workflow_report.md` exactly: mean smoothing gain +0.0136 Pearson/+0.0281
Spearman, mean cost of restricting to AGL>1m is **-0.1053** Pearson.

**`workflow_report.md`'s actual conclusion about `JAX_022_009`:** its low
correlation is *not* a display artifact, not near-ground-pixel contamination,
and not pixel noise (ruled out by the tests above) — it reflects a genuine
scene-dependent mismatch between DAv2's relative-depth prior and true AGL for
that tile. It's used as supporting evidence for the report's overall verdict:
"Frozen DAv2 is a useful relative-height prior... but its relationship to
actual above-ground height is strongly scene-dependent," with the explicit
decision to keep DAv2 frozen and add a metric-reconstruction stage on top
(statistical calibration → sparse anchoring → semantic prior → learned scale
modulation → RDAH-Net fusion).

## 4. Calibration stage (200 tiles)

`scripts/build_calibration_manifest.py` selected 200 tiles (100 JAX/100 OMA),
**explicitly excluding the 50 benchmark tiles** (`if tile_id in benchmark_ids:
continue`, plus an `assert` of zero intersection in the script itself). I
independently verified this by diffing tile IDs between
`dav2_baseline/manifest.csv` (50) and `dav2_calibration/manifest.csv` (200):
**0 overlapping tile IDs**. No leakage.

`scripts/run_dav2_batch.py` ran frozen DAv2 on all 200 tiles;
`scripts/fit_dav2_calibration.py` sampled 10,000 pixels/tile (2,000,000
total) and fit two global mappings, reproduced exactly from
`calibration_fit_summary.json`:

- **1A global linear:** `height = 13.2205 × DAv2 − 1.1372`, **R² = 0.1433**
  (in-sample, on the 200-tile calibration pixels).
- **1B global isotonic:** 460-step monotonic curve.

## 5. Held-out evaluation — actually completed, not just planned

`raw-notes.pdf` Section 5-6 states the plan to "apply the same to the 50
held-out tiles" but never reports a result — it ends with "we cannot judge
whether isotonic calibration is actually better yet. We must test both
models on the untouched 50-tile benchmark." **This step was in fact run** —
`scripts/evaluate_dav2_calibration.py` exists in the repo (untracked in git,
and never named anywhere in the PDF) and its output is on disk at
`data/dfc2019/experiments/dav2_calibration/evaluation/`, timestamped after
the calibration fit. It applies both fitted mappings to
`dav2_baseline/manifest.csv` — the identical 50-tile benchmark used for the
DAv2 baseline — computing MAE/RMSE/Pearson/Spearman per tile. Results
(`calibration_evaluation_summary.csv`):

| Method | Tiles | MAE mean (m) | MAE median (m) | RMSE mean (m) | Pearson mean | Spearman mean |
|---|---:|---:|---:|---:|---:|---:|
| global_linear | 50 | 3.718 | 3.136 | 5.359 | 0.5394 | 0.4460 |
| global_isotonic | 50 | 3.691 | 3.105 | 5.313 | 0.5559 | 0.4461 |

For context: mean AGL across these 50 tiles is 3.43 m, mean within-tile AGL
std is 5.45 m (from `linear_metrics.csv`).

See `verdict.md` for what this means.
