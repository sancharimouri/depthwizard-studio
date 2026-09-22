# DepthWizard / DFC2019 DAv2 Baseline Investigation Report

## Purpose

This report records the investigation performed after visually inspecting `JAX_022_009`, the lowest-Pearson JAX tile in the 50-tile DAv2 benchmark, and noticing that its AGL visualization looked noisy/fragmented.

The goal was to determine whether the low DAv2↔AGL correlation was mainly a display/reference artifact, near-ground-pixel contamination, pixel-scale noise, or a genuine scene-dependent mismatch between relative DAv2 depth and above-ground height.

## 1. Frozen DAv2 baseline

The DFC2019 Track 1 benchmark contained 2,783 matched RGB/AGL pairs.

Core experiment paths:

`data/dfc2019/experiments/dav2_baseline/manifest.csv`

`data/dfc2019/experiments/dav2_baseline/metrics.csv`

`data/dfc2019/experiments/dav2_baseline/config.json`

`data/dfc2019/experiments/dav2_baseline/depth/`

Across the 50 selected tiles:

- Pearson mean: **0.539445**
- Pearson median: **0.589948**
- Pearson std: **0.218819**
- Spearman mean: **0.445957**
- Spearman median: **0.482425**
- Spearman std: **0.204980**

14/50 tiles had Pearson >= 0.70; 19/50 were below 0.50.

By city:

- JAX: Pearson mean **0.599481**, Spearman mean **0.511450**
- OMA: Pearson mean **0.474406**, Spearman mean **0.375008**

Interpretation: frozen DAv2 contains useful height-related information in some WorldView-3 scenes, but the quality is strongly scene-dependent.

## 2. Why `JAX_022_009` was investigated

`JAX_022_009` had:

- Pearson **0.167455**
- Spearman **0.150750**

It was the lowest-Pearson JAX tile in the 50-tile subset.

The RGB and DAv2 maps looked structurally meaningful: buildings, trees, road/canal-like regions and major scene layout were visible. The AGL rendering looked fragmented/noisy.

This raised several hypotheses:

1. The AGL visualization might look bad even if the numeric raster were valid.
2. Near-ground/zero-height pixels might dominate the correlation.
3. Pixel-scale AGL/DAv2 noise or slight registration differences might suppress correlation.
4. DAv2 might capture visual structure but fail to preserve true AGL ordering.
5. The problem might mainly be scale/calibration rather than relative-depth failure.

## 3. Diagnostic of `JAX_022_009`

Script:

`scripts/diagnose_dfc_tile.py`

Command:

`python scripts/diagnose_dfc_tile.py JAX_022_009`

Outputs:

`data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_022_009/correlation_diagnostics.csv`

`data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_022_009/rgb_dav2_agl_comparison.png`

`data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_022_009/dav2_vs_agl_scatter.png`

### AGL numeric statistics

- min **-0.757 m**
- p01 **0.000 m**
- median **1.867556 m**
- p95 **14.017480 m**
- p99 **17.408983 m**
- max **23.702999 m**
- mean **3.958841 m**
- std **4.817303 m**
- 100% of pixels finite
- AGL < 0 m: **0.81%**

### Correlation tests

| Test | Pearson | Spearman |
|---|---:|---:|
| All finite pixels | **0.167455** | **0.150750** |
| AGL > 0 m | 0.163832 | 0.151280 |
| AGL > 1 m | 0.110554 | 0.133367 |
| AGL > 2 m | 0.083691 | 0.106561 |
| AGL > 5 m | **-0.084049** | **-0.119534** |
| Gaussian sigma=2 | **0.167491** | **0.149450** |

### Inference

The low score was **not primarily caused by**:

- how the AGL image was displayed,
- the small amount of negative/near-zero AGL,
- or fine-scale pixel noise.

Removing low AGL pixels made the result worse, and Gaussian smoothing changed Pearson by essentially zero.

The scatter plot showed a broad many-to-many relationship: similar DAv2 values corresponded to widely varying AGL values. This indicates a genuine scene-dependent mismatch between the relative-depth prior and true above-ground height for this tile.

## 4. Strong-tile control: `JAX_004_014`

The identical diagnostic was run on `JAX_004_014`, a strong DAv2 tile.

Command:

`python scripts/diagnose_dfc_tile.py JAX_004_014`

Outputs:

`data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_004_014/correlation_diagnostics.csv`

`data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_004_014/rgb_dav2_agl_comparison.png`

`data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_004_014/dav2_vs_agl_scatter.png`

Results:

| Test | Pearson | Spearman |
|---|---:|---:|
| All finite pixels | **0.814288** | **0.727833** |
| AGL > 0 m | 0.810477 | 0.771826 |
| AGL > 1 m | 0.732279 | 0.711081 |
| AGL > 2 m | 0.722093 | 0.703845 |
| AGL > 5 m | 0.551565 | 0.478791 |
| Gaussian sigma=2 | **0.832043** | **0.801982** |

Inference: this is genuinely a strong relative-depth case. Smoothing gives a modest improvement, but the core DAv2↔AGL relationship is already strong.

This control experiment shows that the low `JAX_022_009` score is not simply a universal failure of the AGL product or our evaluation code.

## 5. Scaling the diagnostic to all 50 tiles

Script:

`scripts/diagnose_dfc50.py`

Output:

`data/dfc2019/experiments/dav2_baseline/diagnostic_summary.csv`

### Overall means

| Diagnostic | Pearson | Spearman |
|---|---:|---:|
| All pixels | **0.539445** | **0.445957** |
| AGL > 0 m | **0.546073** | **0.508319** |
| AGL > 1 m | **0.446101** | **0.442288** |
| AGL > 2 m | **0.415223** | **0.410766** |
| AGL > 5 m | **0.338633** | **0.335048** |
| Gaussian sigma=2 | **0.549915** | **0.472163** |

Overall medians:

- Pearson all **0.589948**
- Pearson smoothed **0.602911**
- Spearman all **0.482425**
- Spearman smoothed **0.539310**

City means:

### JAX
- Pearson all **0.5995**
- Pearson smoothed **0.6069**
- Spearman all **0.5114**
- Spearman smoothed **0.5426**

### OMA
- Pearson all **0.4744**
- Pearson smoothed **0.4906**
- Spearman all **0.3750**
- Spearman smoothed **0.3988**

Mean effect of smoothing:

- Pearson **+0.0136**
- Spearman **+0.0281**

Mean effect of restricting to AGL > 1 m:

- Pearson **-0.1053**
- Spearman **-0.0139**

### Inference

The 50-tile diagnostic confirms that:

1. Near-ground pixels are not the main issue.
2. Simple smoothing has only a small average benefit.
3. The dominant issue is scene-dependent disagreement between DAv2's relative-depth representation and true AGL.
4. DAv2 still contains real predictive structure because many tiles are strongly positively correlated.

## 6. Files created during this investigation

### Core benchmark

- `data/dfc2019/experiments/dav2_baseline/manifest.csv`
- `data/dfc2019/experiments/dav2_baseline/metrics.csv`
- `data/dfc2019/experiments/dav2_baseline/config.json`
- `data/dfc2019/experiments/dav2_baseline/depth/`

### Tile diagnostics

For `JAX_022_009`:

`data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_022_009/`

For `JAX_004_014`:

`data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_004_014/`

Each contains:

- `correlation_diagnostics.csv`
- `rgb_dav2_agl_comparison.png`
- `dav2_vs_agl_scatter.png`

### 50-tile diagnostic

- `data/dfc2019/experiments/dav2_baseline/diagnostic_summary.csv`

### Baseline analysis summary files

Recommended location:

`data/dfc2019/experiments/dav2_baseline/reports/`

- `baseline_summary.json`
- `city_summary.csv`
- `tile_ranking.csv`
- `best_worst_tiles.csv`
- `diagnostic_summary.csv`
- `workflow_report.md`

## 7. Final conclusion and project decision

The investigation does **not** support the idea that the baseline problem is mainly caused by a badly rendered AGL image, near-ground pixels, or small-scale raster noise.

The supported conclusion is:

> **Frozen DAv2 is a useful relative-height prior on high-resolution WorldView-3 imagery, but its relationship to actual above-ground height is strongly scene-dependent.**

Evidence:

- 50-tile mean Pearson **0.539**
- 50-tile mean Spearman **0.446**
- `JAX_004_014`: **0.814 / 0.728**
- `JAX_022_009`: **0.167 / 0.151**
- average smoothing gain only **+0.0136 Pearson / +0.0281 Spearman**
- restricting to AGL > 1 m changes mean Pearson by **-0.1053**

Therefore:

### Keep DAv2 Large frozen as the relative-depth foundation.

Do not fine-tune or replace DAv2 yet.

The next research problem is a metric reconstruction stage that converts an imperfect relative-depth prior into a robust metric nDSM.

Planned progression:

1. statistical DAv2→nDSM calibration baseline,
2. sparse/GCP anchoring,
3. semantic prior,
4. learned scale modulation,
5. RDAH-Net fusion.

The 50 benchmark tiles should remain fixed across these experiments.

## 8. Important clarification for the next experiment

A positive affine mapping,

`nDSM = a * DAv2 + b`

does **not** change Pearson or Spearman when evaluated on the same pixels, assuming positive `a`. It can improve the metric scale, and therefore MAE/RMSE, but correlation is invariant to positive affine changes.

A nonlinear monotonic mapping can change Pearson because it changes the shape of the relationship. A strictly monotonic mapping preserves rank ordering, so Spearman should remain essentially unchanged (ties can cause small implementation-dependent effects).

Therefore, the correct experimental interpretation is:

- If linear calibration leaves Pearson/Spearman unchanged but improves MAE/RMSE, the main issue was metric scale, not rank ordering.
- If a nonlinear calibration improves Pearson while Spearman stays roughly constant, the DAv2→height relationship had nonlinear scaling/shape mismatch but similar rank ordering.
- If Spearman remains low after calibration, then the problem cannot be solved by a global monotonic mapping alone; scene/image/semantic/context-dependent information is needed.

This is why semantic priors, learned scale modulation, and RDAH-Net become justified **only after** establishing how far simple calibration can go.