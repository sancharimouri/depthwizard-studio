# Full DAv2-Small fine-tune, twin (mean, log-variance) head (Method 6) — Verdict

**Verdict: replicates, cleanly, on our own split. Full fine-tuning of
DA-V2-Small into a twin (mean, log-variance) head beats both the per-tile-OLS
oracle baseline (MAE 3.39m/RMSE 4.58m) and this project's own best prior
method, Method 4's `phase2_building_rank_v2` (MAE 2.88m/RMSE 4.78m), on all
four tracked metrics simultaneously — MAE 2.05m, RMSE 3.53m, Pearson 0.737,
Spearman 0.654, mean over the same 4-fold spatial-quadrant holdout used
throughout this audit. This is the strongest result in the project's ML
research track to date, and the first to clear RMSE against the oracle
baseline at all (every prior method either lost on RMSE or wasn't tested
against it this directly).**

## 1. Does full fine-tuning also beat an oracle affine fit on OUR split?

Yes, unambiguously, and not narrowly. Method 4's best configuration
(`phase2_building_rank_v2`) beat the oracle baseline's MAE by 15.1% but
*lost* to it on RMSE by 4.3% — "the smallest gap reached in this entire
audit," per that method's own writeup, still a loss. Method 6 beats the
oracle baseline's MAE by 39.5% *and* its RMSE by 22.9%, with Pearson and
Spearman both roughly 0.15 higher in absolute terms than either reference.
Every fold individually clears both baselines on every metric — this isn't
one lucky fold dragging a mean; the fold range is MAE 1.90–2.18m, RMSE
3.41–3.64m, tighter than Method 4's own fold-to-fold spread (that method's
audit noted Pearson ranging 0.556–0.629 across folds as a caution against
over-reading small differences; Method 6's Pearson range is 0.711–0.753,
still clearing 0.5835 — the best number anything else in this audit has
posted — by 0.13 to 0.17 in every single fold).

## 2. Did the user give wrong orders anywhere, or is there an adaptation gap?

**No wrong orders. The task's framing was correct throughout** — using
"note their number isn't directly comparable... the real test is whether
full fine-tuning ALSO beats an oracle fit on OUR split" was exactly the
right question to ask, and it has a clean, unambiguous "yes" answer, which
is itself evidence the instruction was well-posed rather than a case where
an honest negative would have been forced by a bad setup.

**Two real implementation gaps did exist, both found and fixed during
adaptation, both documented in full in summary.md §3:**

1. Dropping the source repo's LR warmup/decay schedule as "auxiliary
   engineering" was itself the gap — it is not auxiliary, it is what keeps
   AdamW's per-parameter step size bounded early in training, independent
   of gradient-norm clipping. This was a misjudgment on my part during the
   simplification pass, not something the task asked for; fixed before any
   real numbers were produced, so it never touched the reported result.

2. The masking-order bug in `gaussian_nll` (compute over the whole padded
   tensor, then select, instead of select-then-compute) is arguably the
   more interesting finding: it surfaced a genuine, previously-undocumented
   data-quality issue in this project's own 50-tile DFC2019 benchmark — tile
   `JAX_004_016` has real `NaN` values in its raw AGL raster at invalid
   pixels. This has apparently never caused a problem for any *other*
   method in this audit (Methods 1–5) because every one of them, by
   inspection, masks before computing (e.g. Method 4's
   `huber_masked(pred[mask], target[mask], ...)`). Method 6 is the first
   method whose loss function computed over the raw tensor first, so it's
   the first to have hit this. **Worth flagging forward:** any future
   method that touches raw AGL values before masking (rather than after)
   should be aware of this tile specifically, and a one-line data-hygiene
   pass (`np.nan_to_num` at load time, or an assertion that no raster in the
   benchmark contains NaN outside its own valid mask) would close this off
   for good rather than leaving it as a trap for the next adaptation.

Neither gap reflects an error in the task's instructions — both were
implementation details that surfaced only once actually running the
adaptation, which is exactly the value of doing this experiment rather than
reasoning about it from the source repo's docs alone.

## 3. Caveats — read before treating this as settled

- **Single run, single seed (42), no repeated-seed variance estimate.** The
  4-fold spread (MAE 1.90–2.18m) is much smaller than the margin over both
  baselines (0.7–1.3m), so seed noise is very unlikely to erase the result,
  but this is not the same as having run it twice to confirm.
- **The variance-ratio / OLS-calibration-slope diagnostic that Methods 1–4
  report (regression-to-the-mean check) was not computed for Method 6** —
  the current script saves aggregated metrics, not raw per-pixel
  predictions, so this would need a rerun (or a checkpoint-reload pass, as
  Method 4's own audit did in its summary.md item 1) to add retroactively.
  Given how large Method 6's Pearson/Spearman jump is relative to every
  other method, checking whether that's a genuinely sharper prediction
  distribution or partly a compression artifact that correlation metrics
  are less sensitive to is the natural next check before fully trusting the
  magnitude of the win, not just its direction.
- **Deliberately simplified relative to the source repo**: no
  gradient-matching term, no β-NLL, no binned/head-tail-cut head, only a
  short fixed-length Huber warmup rather than a tuned one, 12 epochs rather
  than their run02's more extensive schedule. That Method 6 already wins by
  this much *without* that additional engineering is grounds for optimism
  that a fuller port would do at least as well, not evidence that the
  simplified version is already the ceiling.
- **DFC2019-only.** This says nothing yet about whether full fine-tuning
  transfers to this project's actual target domain (Sentinel-2, 10m, this
  project's four visualization regions). That is a different, harder
  question than the one this experiment was scoped to answer, and the
  project's own Method 4/5 audits already show DFC2019 gains do not
  automatically survive a domain switch — this is flagged as the natural
  next step, not assumed.
- **Compute/engineering scope**: this ran on Apple Silicon MPS in ~53
  minutes total for all 4 folds — cheap enough that re-running with more
  epochs, a second seed, or the fuller loss (gradient-matching term, β-NLL)
  is a reasonable next increment before calling this method "done" in the
  same sense Methods 1, 2, 3, and 5 are closed.

## 4. Recommendation

**Promote Method 6 to "current best result" in the ML research track**,
ahead of Method 4's `phase2_building_rank_v2`, on the strength of clearing
all four metrics rather than 3 of 4. Do not yet mark it CLOSED — the
caveats above (variance-ratio diagnostic, multi-seed confirmation, and
especially the untested Sentinel-2 domain-transfer question) are concrete,
cheap follow-ups given the run's small compute footprint, not open-ended
scope creep.

---

## 5. 2026-09-22 — Ablation: devendrakushwah80's GSD-FiLM conditioning and height-balanced loss/sampling

**Status: COMPLETE. Result: mixed but genuinely informative — GSD-FiLM is a
clean wash, height-balanced loss/sampling is a real (if modest) win.**
DFC2019-only, same 4-fold spatial-quadrant holdout as the baseline above.
Explicitly **not** a Sentinel-2 experiment — that track is closed per the
three convergent CNN failures documented in
`docs/method-audit/sentinel2/sign-flip-detector.md`.

Both techniques read directly from
`external/DepthWizard-SIH26175/src/{models/m3_net.py,training/{m3_losses.py,
m3_dataset.py}}` (devendrakushwah80/DepthWizard-SIH26175), never executed —
independent reimplementations in
`scripts/evaluate_method6_gsd_film_height_balanced.py`, which imports
everything else (backbone, twin head, height-scale, LR schedule,
Huber-warmup-then-NLL, `quadrant_bounds`) unchanged from the baseline
script above, so only the tested component differs between runs.

### GSD-FiLM conditioning: a real port of a technique that has nothing to condition on here

Ported `GSDFiLMBlock` verbatim (zero-init MLP → per-channel gamma/beta,
`(1+gamma)*feat + beta`), inserted at the closest architectural analogues
of their two injection points: our neck output (`feat`, before `conv1`)
and the post-`conv2`/`activation1` feature map (before `conv_mu`/
`conv_log_var`). **Their model conditions on real per-tile GSD variation;
this benchmark doesn't have any** — DFC2019's 50 tiles are one fixed
sensor product, so `gsd_vec = [1.0, 1.0]` is identical for every sample.
Flagged before running, not discovered after: with a constant input, both
FiLM blocks reduce to a fixed per-channel affine transform (extra learnable
scale/bias parameters at two points), not real scale-adaptive conditioning.

Result confirms exactly that: **MAE +0.63% worse, RMSE +0.82% worse,
Pearson −0.0026, Spearman −0.0024** vs. baseline — a clean wash, all four
metrics move in the same (slightly negative) direction by less than 1%,
consistent with "extra near-identity-initialized capacity with nothing
informative to condition on" rather than a real effect in either direction.
**Not adopted.** This says nothing against the technique itself — it would
need genuine cross-GSD training data (e.g. this project's own Sentinel-2
10m tiles mixed with DFC2019's finer native resolution) to be a fair test,
which is out of scope here per the Sentinel-2 track being closed.

### Height-balanced loss/sampling: a real, modest, consistent win

Two changes, ported together per the task's framing (their own code treats
them as one "height-aware" feature, `height_aware` flag governing both):
- **Loss:** `CappedHeightWeightedLoss` ported verbatim (`weight = clip(1 +
  0.08*target, 1, 4.0)`, their values), added as an auxiliary term with
  their own default weight `lambda_height_weight=0.35`, on top of the
  existing primary loss (Huber warmup → Gaussian NLL) — additive, not a
  replacement, same composite-loss pattern as their `M3CompositeLoss`.
- **Sampling:** their `height_aware_sampling` anchors 65% of *crops* on
  tall (≥10m) or canopy (4–25m) pixels — Method 6 has no cropping step at
  all (one training example is a whole padded 512×512 quadrant), so there
  is no pixel-level anchor point to port literally. Adapted to the
  granularity this architecture actually has: a `WeightedRandomSampler`
  over whole training quadrants, `weight = 1 + 3·frac_tall + 2·frac_canopy`
  using their same 10m/4–25m thresholds — oversampling whole quadrants
  containing more of the underrepresented regime, not sub-quadrant crops.
  A real adaptation to a different granularity, reported as such, not
  claimed as a literal 1:1 port.

Result: **MAE 1.9800m (−3.55%), RMSE 3.4923m (−1.09%), Pearson 0.7445
(+0.0077), Spearman 0.6563 (+0.0019)** vs. baseline — improves on all four
metrics. Per-fold: **MAE and RMSE improve in all 4/4 folds individually**
(not one fold dragging the mean); Pearson/Spearman improve in 3/4 folds
(fold 1 is the exception, both very slightly worse: Pearson 0.7383→0.7366,
Spearman 0.6509→0.6392). Modest in absolute terms, but consistent enough
across folds and metrics to call a real, if small, improvement rather than
noise.

### Does either close the gap to zaidnansari2011's own reported numbers?

Their fine-tuned run02 (their own split, not directly comparable — see
summary.md §5 for why): RMSE 6.454m / MAE 1.840m.

| | MAE (m) | vs. ref MAE gap | RMSE (m) | vs. ref RMSE |
|---|---:|---:|---:|---|
| zaidnansari2011 reference | 1.840 | — | 6.454 | — |
| Method 6 baseline | 2.0528 | +0.2128m (11.6% worse) | 3.5309 | already 45.3% *better* |
| + GSD-FiLM | 2.0657 | +0.2257m (worse than baseline) | 3.5599 | still better than ref, worse than baseline |
| **+ height-balanced loss/sampling** | **1.9800** | **+0.1400m (34.2% of the MAE gap closed)** | **3.4923** | already 45.9% *better*, gap widens further in our favor |

The RMSE "gap" was never really a gap — Method 6's baseline already beats
zaidnansari2011's own reported RMSE by a wide margin (different eval
protocols: their whole-tile sliding window vs. this project's
padded-quadrant single pass, so not literally apples-to-apples, per the
existing non-comparability caveat — but the direction is consistent across
every variant tested here). The number worth tracking is MAE, where a real
gap exists. **Height-balanced loss/sampling closes about a third of it
(34.2%); GSD-FiLM closes none of it and is not adopted.**

### Verdict

**Adopt height-balanced loss/sampling as an improvement to Method 6's
recipe. Do not adopt GSD-FiLM conditioning in this form** — not because
the technique is wrong, but because this benchmark has no real GSD
variation for it to condition on, which was true before running and is
confirmed, not contradicted, by the result. This is DFC2019-only, same
scope limitation as the rest of Method 6 — whether the height-balanced
gain survives a Sentinel-2 domain transfer is exactly the kind of question
the closed Sentinel-2 CNN track (three convergent failures) already
answered "no" for architecturally similar attempts, so this is not
proposed as a reason to reopen that track.

### Reproducing this

- `scripts/evaluate_method6_gsd_film_height_balanced.py` — both variants,
  toggled independently via `--enable-gsd-film` / `--enable-height-balanced`.
  `python scripts/evaluate_method6_gsd_film_height_balanced.py --outdir
  <dir> --tag <tag> --enable-gsd-film --epochs 12` (or
  `--enable-height-balanced`). Writes `data/dfc2019/experiments/
  method6_gsd_film/m6_gsdfilm_results.json` and `data/dfc2019/experiments/
  method6_height_balanced/m6_heightbal_results.json` respectively
  (per-tile breakdown for all 50 tiles × 4 folds included in each).
  ~62 minutes/variant on Apple Silicon MPS, single seed (42), single run
  each — same no-repeated-seed caveat as the baseline.
