# Prompt 4 — Method 6 held-out heights for the DFC2019 packs

Script: `scripts/dfc2019_heights.py`. Outputs go to the gitignored `data/dfc2019/terrain_packs/heights/`.

## Method (as built)
- **Models:** the seed-43 height-balanced fold checkpoints. **Fold q holds out quadrant q**
  (0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right; split at 512).
- **Inference:** each fold model runs on the whole 1024 px tile through
  `vhr_dsm_pipeline.tiled_predict_margin` (margin 192, reflect/mirror padding, 128 px cores,
  16 px core ramp) and keeps only its held-out quadrant.
- **No leakage outside the feather band:** no tile pixel is predicted by a model that trained on it,
  except inside the ±16 px feather band across the two quadrant borders (6.2 % of the tile). There
  the adjacent folds' predictions are blended linearly, and some of those folds did train on
  those pixels. This is disclosed; it is the cost of the feathering the brief asks for.
- **Post-processing:** negative predictions → 0. Saved at native 0.3 m (1024²), as inputs to
  Prompt 5 only.
- **Seam ratio:** `vhr_dsm_pipeline.seam_ratio` with lines at the quadrant border (2 px bands at
  511/512, rows and columns) = mean |∇| on the border band / elsewhere. It is reported for the
  feathered result and for a hard-cut composite. Limit to flag: 1.5.
- **Parity:** one tile through this script vs. direct, fresh-model calls of
  `vhr_dsm_pipeline.tiled_predict_margin`; maximum absolute difference reported. The backend has
  no Method 6 inference path (live inference is out of scope), so the pack builder's path is the
  one checked.

## Pre-registered cap rule (committed before the cap is computed)
- **Data:** all 50 tiles, pooled; valid AGL only (finite, not < −100 or > 1000); AGL < 0 counts as
  0. Pixels are subsampled with stride 7.
- **Margin M = 3.0 m** (fixed now). It is about 1.5× Method 6's DFC2019 MAE (1.98 m), so a
  disagreement larger than M is well outside the model's typical error. Prompt 5 uses the same M
  for "prediction and AGL disagree by more than the margin".
- **Bins:** 2 m AGL bins from 0 to 60 m; a bin counts if it has ≥ 2,000 sampled pixels.
- **Cap:** the lowest bin lower edge ≥ 4 m from which **every** counted bin at or above it has
  median(AGL) − median(prediction) > M. That is the level above which the model falls short by
  more than the margin, and stays short.
- **Plateau check (reported, not gating):** the slope of the bin medians (prediction on AGL) at and
  above the cap. A slope well below 1 confirms the plateau.
- **If no bin qualifies,** there is no cap and Prompt 5 corrects nothing on this criterion. That
  would be reported, not re-tuned.

## Results
*(filled in after the run; the sections above are not edited)*

Run 2026-09-28 on MPS: 50/50 tiles, 55–71 s per tile (4 folds × 81 windows), ~50 min in the background.

### Tiles done: 50 / 50
Heights are ≥ 0 and saved at 1024² in `heights/<tile>.npz`. The per-tile maximum prediction is 6.0–52.1 m
(median 19.3 m).

### Seam ratio at the quadrant borders (limit 1.5)
- **Feathered (shipped):** median **0.95**, range 0.47–1.96. **Two tiles are above 1.5: JAX_165_015 (1.96) and
  OMA_248_030 (1.96).**
- **Diagnostic on those two** (after the fact; nothing was changed):
  - The *reference AGL itself* scores 1.85 and 1.89 on the same 512 lines.
  - Control lines at 480/544 in the prediction score 1.21–2.43.
  - The high ratio is real structure lying along row/column 512 (building edges, roads), not a stitching
    seam. They are flagged anyway, as the brief asks.
- **Hard-cut composite (no feather), for comparison:** median 2.65, 48/50 tiles above 1.5. The feather is doing
  its job.

Per tile (feathered):
JAX_004_006 0.79 · 004_014 0.78 · 004_016 0.93 · 018_012 0.93 · 022_009 0.47 · 031_006 0.71 · 072_015 0.95 ·
118_012 1.15 · 118_015 1.01 · 149_006 0.57 · 149_025 0.58 · 161_001 0.96 · 164_008 1.18 · **165_015 1.96** ·
166_006 0.92 · 175_002 0.81 · 204_005 0.65 · 214_015 1.33 · 214_023 1.31 · 224_025 0.72 · 264_013 0.74 ·
269_009 0.89 · 416_009 1.06 · 416_022 1.04 · 505_016 0.91 · 505_018 0.96 ·
OMA_042_011 1.05 · 084_038 1.17 · 134_027 0.82 · 144_030 1.33 · 198_002 0.91 · 211_032 1.01 · 211_039 1.36 ·
212_033 1.10 · 221_034 0.73 · 225_001 1.35 · 230_036 0.65 · 248_029 1.24 · **248_030 1.96** · 258_020 0.88 ·
269_035 0.98 · 281_002 1.17 · 281_030 1.10 · 315_019 1.03 · 315_020 0.71 · 332_037 0.88 · 364_003 1.24 ·
364_043 1.37 · 376_023 0.70 · 376_038 0.55

### Cap and margin (pre-registered rule): **cap = 12.0 m, margin = 3.0 m**
Pooled over 50 tiles: 7,489,850 sampled valid pixels (stride 7); selected bins shown:

| AGL bin (m) | Median AGL | Median prediction | Shortfall |
|---|---|---|---|
| 4–6 | 4.96 | 4.92 | +0.04 |
| 8–10 | 9.00 | 7.67 | +1.33 |
| 10–12 | 10.89 | 8.75 | +2.14 |
| **12–14** | 12.91 | 9.77 | **+3.15** (first bin > M, and every higher bin stays > M) |
| 16–18 | 16.77 | 12.79 | +3.98 |
| 20–22 | 20.91 | 14.15 | +6.76 |
| 28–30 | 29.06 | 17.62 | +11.44 |
| 38–40 | 38.72 | 22.73 | +15.99 |
| 46–48 | 47.08 | 20.74 | +26.34 |
| 58–60 | 58.80 | 20.45 | +38.35 |

- **Plateau:** above the cap the slope of median prediction on median AGL is **0.28**. The median prediction
  levels off at about 18–21 m however tall the object is. This confirms (on held-out predictions, 50 tiles) the earlier
  ~18–23 m ceiling. The model starts falling short already at 12 m.
- Full table: `heights/cap.json`.

### Parity
JAX_004_006, MPS, torch 2.14.0. This script vs direct fresh-model calls of
`vhr_dsm_pipeline.tiled_predict_margin`:

| Comparison | Max abs. difference |
|---|---|
| Per fold | **0.0 m** |
| Composite | **0.0 m** |
| Saved pack input | **0.0 m** |

The backend has no Method 6 inference path (live inference is out of scope), so the pack builder's path is
the checked one.
