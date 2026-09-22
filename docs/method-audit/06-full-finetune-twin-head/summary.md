# Full DAv2-Small fine-tune, twin (mean, log-variance) head (Method 6) — Summary

**Source of the idea:** read directly from `external/sih2026-depthwizard`
(`zaidnansari2011/sih2026-depthwizard`, cloned full-history into `external/`),
specifically `depthwizard/model.py`, `train.py`, and
`docs/evaluation-protocol.md`. Nothing from that repo was executed or
imported — this is an independent reimplementation of the core idea in
`scripts/evaluate_method6_finetune_twinhead.py`, adapted onto this project's
own data and evaluation protocol. That repo full-fine-tunes DA-V2-Small into
a Kendall & Gal-style twin head (mean + log-variance) predicting AGL in
metres directly, and reports (their own dataset — GAMUS + DFC2019
Jacksonville/Omaha — and their own region/city-holdout split, protocol 2, 80
val tiles): zero-shot+global-affine baseline RMSE 9.308/MAE 4.528, zero-shot+
oracle-affine RMSE 7.285/MAE 3.147, fine-tuned run02 RMSE 6.454/MAE 1.840 —
i.e. full fine-tuning beat even an oracle per-tile affine fit, on their split.

**The question this audit set out to answer:** does the same thing happen on
*our* split? Their number isn't directly comparable pixel-for-pixel to ours
(different tile split, their eval protocol) — the real test is whether full
fine-tuning ALSO beats an oracle fit on OUR 50-tile DFC2019 benchmark, same
4-fold spatial-quadrant holdout already established for Method 4.

## 1. What was adapted vs. simplified

**Adapted (the load-bearing part of the method):**
- Full backbone fine-tune (not frozen-feature scale modulation like this
  project's own Method 4).
- Twin head: reuse the pretrained `conv1`/`conv2`/`conv3`(→ mu) from the
  DA-V2-Small depth head, add a fresh 1×1 `conv_log_var` alongside it
  (shared trunk, same arrangement as the source).
- Output scaled to metres via a per-fold `height_scale` (that fold's
  training-quadrant AGL p95 — never the held-out quadrant's, to avoid
  leakage).
- Two learning rates (backbone low, head 50× higher) and an LR
  warmup+linear-decay schedule (`lr_lambda_factory`-equivalent).
- Gaussian NLL loss (heteroscedastic regression), masked to valid pixels,
  with a short plain-Huber warmup phase before switching to NLL.

**Simplified away** (auxiliary engineering in the source repo, not the core
method-under-test): the gradient-matching loss term, β-NLL reweighting, MSE
warmup tuned per-dataset, the binned/head-tail-cut height head, the thermal
governor (irrelevant on Apple Silicon), and mid-run NaN/sigma-rail
tripwires (a plain non-finite-loss abort is kept). If Method 6 is carried
forward, those are the natural next additions, not blockers to this result.

## 2. Data and split (ours, not theirs)

Same 50-tile DFC2019 benchmark and the same 4-fold spatial-quadrant holdout
as Method 4 (`scripts/evaluate_method4.py::quadrant_bounds`, imported
directly so the fold geometry is bit-identical): for each fold, one
quadrant is held out from every tile; the model trains on the union of the
other 3 quadrants across all 50 tiles, and is scored only on the held-out
quadrant, pooled per-tile-then-per-fold exactly like Method 4's own
`agg()`/`final_agg()` convention. Ground truth is AGL (`*_AGL.tif`), the
same quantity both Method 4 and zaidnansari2011 predict — the quantity
matches even though the split protocol differs (fold-quadrant vs. their
region/city holdout).

Each 512×512 quadrant is reflect-padded to 518×518 (37 × patch size 14) so
DA-V2's ViT patch embedding sees an exact multiple, then the prediction is
cropped back to 512×512 before scoring — no resizing/interpolation of the
input and no loss of quadrant coverage. No precomputed DAv2 depth is used at
all; the backbone runs on raw RGB directly (this is the point of full
fine-tuning vs. Method 4's frozen-feature approach).

## 3. Two real bugs found and fixed while adapting (not user error — see verdict.md)

**Bug 1 — LR schedule dropped as "auxiliary," turned out load-bearing.**
Smoke-testing found the loss going non-finite by step 3 with plain
gradient-norm clipping and a fixed LR. Root cause: AdamW normalizes each
parameter's step by its own running gradient RMS, so clipping the gradient
*direction*'s norm does not bound the actual parameter *step size* — with a
fixed `lr_head=2.5e-4`, `mu` oscillated between −19m and +24m step to step
even under clipping. Restoring the source repo's LR warmup+linear-decay
schedule (dropped in the first simplification pass) fixed the oscillation.

**Bug 2 — compute-then-select instead of select-then-compute in the NLL
loss (a real, previously-undetected data-hygiene issue in this project's
own DFC2019 benchmark).** After fixing Bug 1, training still went non-finite,
now at the exact step where a specific tile, `JAX_004_016`, first entered a
training batch. Investigation with `torch.autograd.set_detect_anomaly(True)`
traced it to a `MulBackward0` producing NaN. Direct inspection confirmed:
**`JAX_004_016`'s raw AGL raster contains genuine `NaN` values at invalid
pixels** (not just an out-of-range sentinel — literal `NaN`), something no
existing script in this repo's Method 4 pipeline happened to surface,
because `evaluate_method4.py`'s Huber loss already masks *before* computing
(`pred[mask]`, `target[mask]` as new, smaller tensors). My first
`gaussian_nll` implementation computed the full elementwise NLL formula over
the *entire* padded tensor and masked only afterward
(`per_pixel[mask].mean()`) — a well-known PyTorch pitfall: even though the
masked-out NaN positions don't appear in the final loss value, autograd's
backward pass for the earlier elementwise ops (`mu - target`) still computes
a local derivative at every position, and `0 (upstream grad at a masked-out
position) × NaN (local derivative there) = NaN`, corrupting the *entire*
batch's gradient, not just that pixel's. Fixed by masking `mu`, `log_var`,
and `target` down to valid pixels *first*, then computing the NLL formula on
the smaller valid-only tensors — mirroring the safe pattern the Huber
warmup loss already used. `nan_to_num` was also added in the dataset as
defense in depth.

## 4. Result: all 4 folds, mean of fold means

| Fold | MAE (m) | RMSE (m) | Pearson | Spearman | height_scale (m) |
|---|---:|---:|---:|---:|---:|
| 0 | 1.8986 | 3.4148 | 0.7533 | 0.6594 | 16.62 |
| 1 | 2.0989 | 3.5834 | 0.7383 | 0.6509 | 16.40 |
| 2 | 2.1783 | 3.6426 | 0.7114 | 0.6249 | 16.25 |
| 3 | 2.0354 | 3.4828 | 0.7441 | 0.6824 | 16.42 |
| **Mean (final, this method's headline number)** | **2.0528** | **3.5309** | **0.7368** | **0.6544** | — |

Saved: `data/dfc2019/experiments/method6_finetune_twinhead/method6_results.json`
(per-tile breakdown for all 50 tiles × 4 folds included). Training/eval run:
12 epochs/fold, batch 2, AdamW, backbone lr 5e-6 / head lr 2.5e-4, 30-step
Huber warmup then NLL, ~13 min/fold on Apple Silicon MPS (52.7 min total, 4
folds), single seed (42), single run — no repeated-seed variance estimate
exists yet (see verdict.md §3 for why this doesn't undermine the headline
finding here).

## 5. Comparison to the three reference points

| Metric | Oracle per-tile-OLS baseline | Method 4 best (`phase2_building_rank_v2`) | **Method 6 (this run)** | Beats both? |
|---|---:|---:|---:|---|
| MAE | 3.3924 m | 2.8803 m | **2.0528 m** | **Yes — 28.7% better than Method 4, 39.5% better than oracle** |
| RMSE | 4.5787 m | 4.7751 m | **3.5309 m** | **Yes — 26.0% better than Method 4, 22.9% better than oracle** |
| Pearson | 0.5824 | 0.5835 | **0.7368** | **Yes, by a wide margin on both** |
| Spearman | 0.5093 | 0.5438 | **0.6544** | **Yes, by a wide margin on both** |

zaidnansari2011's own numbers (RMSE 6.454/MAE 1.840, beating their own
oracle-affine RMSE 7.285) are **not directly comparable** — different tile
set (GAMUS+DFC2019 Jacksonville/Omaha vs. our fixed 50-tile benchmark),
different split protocol (region/city holdout vs. our quadrant holdout),
different evaluation tiling (whole-tile sliding window vs. our
padded-quadrant single pass). The comparable, apples-to-apples question —
does full fine-tuning beat an oracle affine fit *on our own split* — is
answered directly by the table above, not by matching their exact numbers.

**This is the first method in this project's entire audit to beat the
per-tile-OLS oracle baseline on all four metrics simultaneously.** Method 4
only ever beat it on 3/4 (never RMSE, closest gap 4.3%); every calibration
variant in Methods 1–3 and 5 either failed to beat it or wasn't compared on
this same footing.
