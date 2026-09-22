# RDAH-Net — Verdict

**Verdict: works conditionally — the prior investigation's "non-functional zero-shot, adapts via
fine-tuning" narrative is overturned in its most important respect. RDAH-Net is not broken
zero-shot; it was fed at roughly 1/100-1/255 of its likely intended input scale, on every
experiment in the PDF including every fine-tuning run. Fixed at zero training cost, it beats the
PDF's own best fine-tuned result on every metric. It still falls short of this project's current
best (Method 6), and two other real, independently-confirmed problems — Track1 checkpoint
contamination and a fold-protocol mismatch with Methods 4/6 — mean neither the original
zero-shot rejection nor the original fine-tuning acceptance can be taken at face value. RDAH-Net
is a genuinely promising candidate that has not yet been tested correctly, not a rejected one and
not a validated one.**

## 1. Was the near-constant zero-shot output correctly diagnosed by the prior investigation?

**No — not incorrectly observed, but incompletely diagnosed.** The prior investigation's raw
observation (mean≈0.081, std≈0.011-0.014, near-zero correlation) is confirmed, reproduced
exactly. But its scale sweep (§8) tested the one region that mattered — `×100`/`×1000` — found a
large change in output magnitude, and rejected both as "artificial" **without ever checking
correlation against ground truth**, only mean/std. This audit re-ran the identical sweep tracking
Pearson/Spearman and found a broad, real peak (+0.65 to +0.90 across all 4 probe tiles) centered
near `×255` — not noise, not an isolated lucky tile, and directly motivated by a real asymmetry
in `loaddata.py`'s own preprocessing (RGB divided by 255, depth not divided by anything).
Validated at the full 50-tile, 4-fold, tile-level-spatial-holdout scale already established in
this project: **MAE 2.231m / RMSE 4.566m / Pearson 0.716 / Spearman 0.655, with zero epochs of
training** — better than the PDF's own best fine-tuned epoch (MAE 2.862 / RMSE 6.476 / Pearson
0.457 / Spearman 0.493) on every single metric. See `summary.md` §1 and `gaps-and-fixes.md` §1
for the full derivation and the one honest caveat (the scale constant's selection has a mild,
flagged risk of information leakage from the 4 tiles it was originally explored on).

**This is the single most consequential finding of this audit.** It doesn't just add a data
point — it means the PDF's central story ("the model is learning a genuinely new spatially
varying RGB+depth→AGL relationship over 5 epochs") is at minimum incomplete, and quite plausibly
substantially wrong: since every fine-tuning run also used the unscaled `[0,1]` input throughout
(`train_rdah_spatial_cv.py`'s `load_depth_tensor()` enforces this with a hard assertion), the
"near-zero correlation at epoch 1 → 0.43-0.65 by epoch 3-5" progression the PDF treats as its
decisive evidence is at least as consistent with the network spending its first few epochs
learning to internally compensate for a ~100-255× input-scale mismatch as it is with learning a
genuinely new domain relationship. This audit did not fine-tune from the corrected scale to
settle which explanation dominates — that is the clear next experiment, not something this
read-and-verify audit should guess at.

## 2. Is RDAH free of the contamination/methodology problems flagged for investigation?

**No — two real, independently-confirmed problems, neither of which is resolved by the item-1
finding.**

- **Track1 checkpoint contamination, confirmed by checksum, not inference**: `104best_model.pth`
  is byte-identical (verified MD5) to the checkpoint Figshare's own listing files under
  `checkpoints-track1`. 41 of this project's 50 benchmark tiles are in that checkpoint's own
  `Track1-train.txt`; the other 9 are in its `Track1-test.txt`. Every one of this project's 4
  spatial CV folds is 75-89% "seen" tiles. There is no clean, checkpoint-unseen fold anywhere in
  the existing results. (`summary.md` §2, `gaps-and-fixes.md` §2.)
- **Fold protocol mismatch with Methods 4/6, confirmed by reading the fold-construction code**:
  RDAH's 4 folds are whole-tile geographic groups; Methods 4/6 hold out in-tile quadrants across
  all 50 tiles. Section 20's comparison table is not apples-to-apples. A concrete, architecturally
  feasible fix was identified (resize the model's one fixed-size component, a 64×64 positional
  encoding buffer, to 32×32 for 512×512 quadrant inputs) but not executed — it's a new training
  experiment. (`summary.md` §6, `gaps-and-fixes.md` §6.)

**The interaction between these two findings and item 1 is itself informative, not just three
separate problems.** If Track1's contamination alone explained the checkpoint's zero-shot
behavior, it should behave differently from HK/Swiss (zero DFC2019 exposure) — it doesn't; all
three fail identically under the wrong input scale (item 3, confirmed with checksummed downloads
of both other checkpoints). That's independent evidence the input-scale bug (item 1), not
contamination, is the dominant zero-shot failure mode — contamination is a real, separate problem
affecting how any *fine-tuning* result should be trusted, not the explanation for the zero-shot
numbers.

## 3. Does the checkpoint-selection-leakage and variance-compression follow-up change anything about the existing fine-tuned numbers?

**Checkpoint-selection leakage: checked, found not to matter much.** A real nested validation
(inner-validation epoch selection vs. a true held-out half, using the already-saved per-epoch
checkpoints, no retraining) lands within ~0.6% MAE of the PDF's naive "always report epoch 5,"
and in one fold actually does slightly better by correctly avoiding a late-epoch decline the PDF
itself already noted qualitatively. This is a confirmed-clean result, not a forced one.
(`summary.md` §7, `gaps-and-fixes.md` §7.)

**Variance-compression: confirmed, and worse than Method 4's.** Fold 0's epoch 3/5 checkpoints
show variance ratios of 0.02-0.05 and OLS slopes of 0.09-0.14 — more severely compressed toward
the mean than Method 4's simpler CNN (0.136 / 0.187), for the same root-cause reason (a
pure-magnitude SmoothL1 loss with no rank term). This means the *existing* fine-tuned numbers,
even setting aside items 1 and 2, share Method 4's known weakness and would benefit from the same
class of fix (a rank-preserving loss term). (`summary.md` §8, `gaps-and-fixes.md` §8.)

Neither of these findings overturns anything on its own — but neither gives grounds to trust the
existing fine-tuned numbers as a finished result either. They're additional, real, independently-
confirmed reasons the existing fine-tuning run (input-scale-flawed, contamination-exposed,
protocol-mismatched, magnitude-only-loss) is not the number this project should carry forward as
"what RDAH can do."

## 4. Where does this leave RDAH relative to the rest of the project?

| | oracle per-tile-OLS baseline | PDF's best fine-tuned RDAH | **Zero-shot RDAH + scale fix (this audit)** | Method 6 (current best) |
|---|---:|---:|---:|---:|
| MAE | 3.39 m | 2.862 m | **2.231 m** | 1.98 m |
| RMSE | 4.58 m | 6.476 m | **4.566 m** | 3.49 m |
| Pearson | 0.582 | 0.457 | **0.716** | 0.745 |
| Spearman | 0.509 | 0.493 | **0.655** | 0.656 |

RDAH, properly scaled and with zero training, is not this project's new best result — Method 6
still wins clearly on MAE and RMSE. But it is a materially stronger, more interesting candidate
than either the PDF's own conclusion ("validated and promising... final model selection and
comparison remain pending") or this project's prior `PROJECT_STATUS_REPORT.md` summary ("the
worst RMSE of the entire audit") suggested — and its Spearman (0.655) is now within noise of
Method 6's (0.656). **The properly-scoped next step, not attempted in this read-and-verify audit,
is a single combined experiment**: fine-tune RDAH from the item-1-corrected input scale, under
the item-6 quadrant-level protocol (for a genuinely fair comparison with Methods 4/6), with a
rank-preserving loss term added per item 8's finding — three concrete, independently-motivated
fixes, not a vague "try harder."

## 5. Forward-looking: does anything here change the risk assessment for a future RDAH-on-Sentinel-2 attempt?

**Yes, directionally, though not conclusively — flagged as a hypothesis worth testing before
committing further Sentinel-2 effort, not a settled fact.** This project's own
`PROJECT_STATUS_REPORT.md` records an earlier, separate zero-shot test of RDAH on Sentinel-2
(Darjeeling) that found "visually confirmed regular checkerboard-grid artifacts" and was
rejected on that basis. That test isn't part of the PDF this audit reviewed and its own input
pipeline wasn't independently re-inspected here, but this project's DAv2 `[0,1]` convention is
used consistently everywhere else in this codebase, making it likely that test also fed unscaled
`[0,1]` depth into a network whose PixelShuffle-based decoder — a component especially prone to
checkerboard artifacts under out-of-distribution input magnitudes — was expecting a much larger
range. **If RDAH-on-Sentinel-2 is revisited, the very first thing to check, before any
architecture or fine-tuning work, is whether that same checkerboard result changes or disappears
once the input depth is rescaled the way item 1 found for DFC2019.** This would be a cheap,
high-value five-minute check relative to the three independent CNN failures already logged for
the Sentinel-2 track (interpolation-memorization, data starvation, target-memorization) — none of
which apply here, since this is a zero-shot inference check, not a training experiment, and
therefore cannot inherit those specific failure modes.

## What this doesn't change

Methods 1-4's own verdicts are untouched by anything found here. Method 6 remains this project's
current best DFC2019 result by a clear margin on MAE/RMSE, and the Sentinel-2 track's own
standing conclusions (frequency fusion as the deployable baseline, three independent CNN failures
closing that avenue) are unaffected — this audit is entirely about the DFC2019 RDAH-Net
investigation and, per the forward-looking note above, raises a testable question about the
*already-rejected* Sentinel-2 zero-shot test rather than reopening the Sentinel-2 CNN-correction
question itself.
