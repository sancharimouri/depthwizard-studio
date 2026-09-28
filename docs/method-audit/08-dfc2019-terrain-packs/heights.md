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
