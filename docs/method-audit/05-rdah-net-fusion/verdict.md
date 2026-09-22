# RDAH-Net — Verdict

> **2026-09-23 update:** the combined fine-tune §4 called for (RDAH-FT-2) has run on all 4 folds.
> **RDAH-FT-2 is NOT ADOPTED.** It loses to Method 6 on all four metrics. The method as a whole
> stays open: whether zero-shot beats fine-tuning because of contamination or because
> fine-tuning damages the model is still unresolved. See §6 below.

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

> **RESOLVED 2026-09-23: tested, clean negative.** The flagged check was run on the original
> Darjeeling tile. The original run turned out to be the Swiss checkpoint, reproduced bit-exactly,
> fed unscaled `[0,1]` depth as suspected below. The ×255 input fix cuts the checkerboard's FFT
> peaks by 2–3 orders of magnitude but **does not remove them**: diagonal peaks persist at periods
> 8/16/32, and the period-32 block-attention peak grows 11.7×. The output has **no terrain
> correlation**, raw or detrended (DEM plane −0.052/−0.047, ICESat-2 plane −0.139/−0.132), while
> the DAv2 control reproduces its known +0.64 → −0.43 flip. Resize vs. pad is ruled out as a
> cause. RDAH-on-Sentinel-2 is **closed**, zero-shot or fine-tuned. This doesn't bear on §6's
> DFC2019 contamination question: the GSD, sensor and construct gaps are confounded with it. Full
> entry: `docs/method-audit/sentinel2/sign-flip-detector.md`, "2026-09-23 — RDAH-Net zero-shot on
> Sentinel-2". The text below is the original, pre-test flag, kept for the record.

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

## 6. 2026-09-23 — RDAH-FT-2 (three fixes stacked): NOT ADOPTED

**Verdict: not adopted. It loses to Method 6 on all four metrics, under every aggregation. It is a
real improvement on FT-1 in both accuracy and fold stability. It still underdisperses severely.
And it doesn't close the question of why zero-shot scores better.** Full numbers, derivations and
sources are in `summary.md` §10. The aggregate comes from
`data/dfc2019/experiments/rdah_quadrant_cv/rdah_ft2_aggregate.json`
(`scripts/aggregate_rdah_ft2.py`, arithmetic only).

- **What ran**: Swiss init, per-fold input scale derived from training quadrants only (×255 for
  folds 0–1, ×300 for folds 2–3), quadrant-level 4-fold holdout matching Methods 4/6, and
  SmoothL1 + 0.5 × rank-pair loss. Epochs were chosen by nested selection. Before training could
  start, an `importlib` registration bug (`@dataclass` crashing on an unregistered module) had to
  be fixed in `import_module()`.
- **Headline, like-for-like with Method 6's aggregation** (mean of per-sample metrics, then mean
  of folds): **MAE 2.500 / RMSE 4.294 / Pearson 0.640 / Spearman 0.506.** Pixel-pooled: MAE 2.499 /
  RMSE 5.598. The mean of fold Pearson/Spearman is 0.571/0.572. A true pooled correlation can't be
  computed from the saved JSONs.
- **vs. Method 6 + height-balanced (1.980 / 3.492 / 0.745 / 0.656): loses on all four.** The
  reason for not adopting FT-2 holds under either aggregation.
- **vs. the HANDOFF §7 bar (Method 2 Grid+Huber+20, 2.929 / 4.718 / 0.532 / 0.471)**: under
  per-sample means, which is how that row was computed, **FT-2 clears it on all four metrics,
  RMSE included.** It fails the RMSE bar only under pixel pooling (5.598). The briefed "fails the
  oracle bar on RMSE" is therefore an aggregation mismatch, and the JSONs don't support it
  like-for-like. Against the per-tile-OLS oracle that Methods 4/6's own audits use
  (3.39 / 4.58 / 0.582 / 0.509), FT-2 wins MAE, RMSE and Pearson, and loses Spearman by 0.003.
- **Fold instability (HANDOFF §3.1)**: Pearson range narrows from FT-1's 0.254–0.607 to
  **0.503–0.607**, both pixel-pooled within fold. FT-1's instability doesn't reproduce here.
- **Nested selection used the held-out quadrant, not inner-train data.** One half of the held-out
  quadrant chooses the epoch and the disjoint other half is reported. The reported tiles never
  influence the choice. The oracle epoch and epoch 5 are within 0.5% MAE of the nested choice, so
  the leakage is negligible. The cost is that every FT-2 number rests on 25 of 50 held-out
  samples per fold.
- **Variance ratio 0.186–0.231** (FT-1: 0.02–0.05), with an OLS slope of 0.19–0.23. That's a real
  improvement, and **still severe underdispersion**: predictions carry about a fifth of the true
  variance. The worst-RMSE samples in every fold are the same few Jacksonville tiles, with high
  Pearson (0.69–0.89) but MAE of 6–12 m. That is consistent with **under-prediction of tall
  structures**. No per-height-bin error was saved, so this is a **hypothesis**, not a finding.

**The open RDAH question, re-scoped.** Zero-shot on DFC2019 (2.231 / 4.566 / 0.716 / 0.655) beats
both fine-tuned runs. Either fine-tuning damages the pretrained model on this small benchmark, or
the zero-shot score is inflated by pretraining overlap. The repo's own file lists make (b)
concrete: the zero-shot run used **Track1**, whose `Track1-train.txt` contains 41/50 benchmark
tiles, while FT-2 used **Swiss**, whose GF-7 lists contain 0/50. The deciding check is Swiss
zero-shot at the fold-derived scale, under the quadrant protocol, with the same per-fold affine
calibration (`summary.md` §10). It was **not run** this session. The zero-shot 50-tile run's own
script and result JSON also weren't found in the repo, so that row is unverified against an
artifact. Until the check runs, RDAH-Net stays **open, not rejected**. Only the FT-2 recipe is
closed.

## 7. 2026-09-23 (later) — rescued artifacts resolve half of the open question

- **The zero-shot row is now backed by a saved file.** `data/dfc2019/experiments/rdah_zeroshot/rdah_x255_zeroshot_fold_results.csv`
  reproduces 2.231/4.566/0.716/0.655. It came from **Track1**, and ×255 was picked on 4 probe
  tiles, 3 of which are in Track1's own training list. The clean Swiss re-derivation is a broad
  plateau (×200–×1000) and picked ×255 or ×300 per fold. The constant is robust; its original
  selection was not clean.
- **"Fine-tuning damages the model": RESOLVED, no.** Swiss zero-shot on FT-2's exact true-test
  samples scores MAE 3.033 / RMSE 6.421 / Pearson 0.492 / Spearman 0.542 (mean of folds). FT-2
  beats it on every metric in 4/4 folds.
- **Contamination: consistent with, not proven.** Track1 0.716 vs. Swiss 0.492 Pearson fits a
  training-data advantage, but protocol differs and "memorised tiles" vs. "in-domain sensor/city"
  are confounded. Track1 on its 9 unseen vs. 41 seen tiles would separate them. Open, low
  priority, inference only.
- **The Sentinel-2 §5 resolution is partly corrected.** Its correlation criterion used terrain
  references for an above-ground-height model, so that half was uninformative. The checkerboard
  half stands. See `summary.md` §11 and the sign-flip-detector 2026-09-23 follow-up entries.

## 8. 2026-09-23 (later) — final status

- **RDAH on Sentinel-2: CLOSED.** Neither zero-shot nor fine-tuning should be attempted. Basis:
  - the pre-registered rule's default clause
  - a resolution cliff on DFC2019 itself (Pearson 0.590 at 1.2 m → 0.242 at 2.4 m; Sentinel-2
    is 10 m)
  - an RGB-intensity mismatch with the training data

  The checkerboard is **not** part of the basis. It's intrinsic to RDAH output and appears
  in-domain. See `summary.md` §12.
- **RDAH on DFC2019: open only for the low-priority contamination split** (Track1 zero-shot on
  its 9 unseen vs. 41 seen tiles; inference only). "Fine-tuning damages the model" is resolved
  (no). The FT-2 recipe stays not adopted.
