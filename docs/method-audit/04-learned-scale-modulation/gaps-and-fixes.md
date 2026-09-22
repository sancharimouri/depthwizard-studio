# Learned scale modulation — Gaps & Fixes

_One issue per subsection. New computation run: reloading the saved fold-0
checkpoint and running forward-pass inference only (no retraining) to get a
per-pixel prediction distribution and internal scale/residual maps that
don't exist as saved output anywhere in this repo._

## 1. Method 4's "baseline" and Method 3's "baseline" are different procedures wearing the same label

**Problem:** Both notes describe "Baseline: AGL = a·DAv2 + b" under "the same
4-fold spatial holdout protocol," implying a like-for-like comparison. They
are not. See `summary.md` §1 for the full mechanism: Method 3 fits 200
independent per-tile affine models; Method 4 fits 4 pooled global affine
models (one per fold, shared across all 50 tiles).

**Checked:** Confirmed identical quadrant boundaries and identical
valid-pixel masks (0 mismatches across 50 tiles), so those are ruled out.
Confirmed the actual fitting-code difference by reading both scripts.
Confirmed the specific numeric signature this produces (MAE/RMSE differ
substantially, Pearson/Spearman nearly identical) both for one tile exactly
and in aggregate.

**Outcome:** This is the load-bearing finding of this audit. It means
Method 4's reported MAE improvement (3.88→3.27 m) is partly an artifact of
a weak baseline, not evidence Method 4 beats the best already-known
DAv2-only correction. Against Method 3's real per-tile baseline
(MAE 3.39 m, RMSE 4.58 m), Method 4 is only marginally better on MAE
(~3.5%) and **worse on RMSE** (5.17 m vs 4.58 m, ~13% worse). **Fix:** rerun
Method 4's evaluation against a per-tile-fit baseline (identical to Method
3's), not a pooled global one, before drawing any "Method 4 beats the
DAv2-only baseline" conclusion. This does not require retraining Method 4's
CNN — only refitting the comparison baseline the same way Method 3 did,
which is cheap (a `LinearRegression` per tile per fold, same as
`evaluate_prior_spatial_cv.py` already does).

## 2. The model's own architecture works against its stated goal of a "spatially varying" correction

**Problem:** The plan explicitly frames Method 4 as learning "a spatially
varying correction ... instead of a fixed affine mapping." In practice, for
the one tile spot-checked, the learned scale map is nearly flat (mean 1.017,
std 0.037 — ±4% around identity) and the residual, while more variable,
never reaches the magnitude needed to reconstruct the true AGL range.

**Checked:** Three compounding causes, all directly visible in
`scripts/evaluate_method4.py`:
1. **Zero-initialized heads** (`nn.init.zeros_` on both `scale_head` and
   `residual_head`) — training starts from the identity mapping
   (`pred = raw_depth`), not from a random spatially-varying function, so
   12 epochs has to move the model *away* from flatness, not just refine it.
2. **`smoothness_loss(scale)`, weighted 0.01** — an explicit L1
   total-variation penalty on the scale map, directly rewarding a *flatter*
   scale field. This isn't a bug (it's presumably there to prevent noisy,
   checkerboard-like per-pixel scale artifacts), but it's in direct tension
   with the plan's stated goal, and the observed result (std 0.037 around
   1.0) is consistent with the smoothness term dominating.
3. **Small training budget**: 12 patches × 64×64 per quadrant per tile
   (≈49,152 px sampled per quadrant out of a ~650K-pixel quadrant, <8%
   coverage) for 12 epochs — a modest budget for learning fine spatial
   structure.

**Outcome:** The correction Method 4 actually learned is much closer to "a
single global scale ≈ 1.0, plus a small smooth additive bump" than to a
genuinely spatially-varying, RGB-conditioned correction. That's a
significant part of why it can't reconstruct AGL's true dynamic range (see
`summary.md` §2) — it isn't different enough from the raw DAv2 baseline to
do much more than nudge values toward whatever minimizes Huber loss given
weak conditioning. **Fix, if pursued:** reduce or schedule down the
smoothness weight, and/or increase the training patch budget, before
concluding the *architecture* (not just the loss) needs to change — right
now it's not possible to tell how much of the flat-scale-map result comes
from the smoothness penalty vs. from the network genuinely having little
learnable spatial signal, and that's a testable, cheap ablation (rerun with
`smoothness weight = 0`) that wasn't run here per this audit's "don't
retrain" instruction.

## 3. The note's per-fold RMSE claim is incomplete

**Problem:** The note states "Fold-wise MAE improved in all 4 folds; Pearson
and Spearman decreased in all 4 folds" and gives only MAE numbers per fold.
This is true as far as it goes, but silently leaves out RMSE, which behaves
differently.

**Checked:** Pulled full per-fold RMSE from the JSON (`summary.md` §6):
RMSE improves in 3 of 4 folds but **worsens in fold 1**
(5.4526→5.4865 m, held-out quadrant = top_right). The overall-table RMSE
improvement (5.3409→5.1736) is a real number, but it's driven by the other
3 folds outweighing fold 1's regression in the average, not a uniform
per-fold pattern the way MAE is.

**Outcome:** Not a contradiction of anything already claimed, but a real
omission — a reader relying on the note's per-fold bullet list would believe
RMSE follows the same clean 4-for-4 pattern as MAE, and it doesn't. This
detail also supports the regression-to-the-mean mechanism (§2 of summary.md):
a compressed-variance predictor is specifically exposed on whichever
fold/quadrant has more high-magnitude true values, which is exactly what a
squared-error metric (RMSE) would surface and MAE would partly average away.
**Fix:** add the RMSE row to the note's per-fold bullet list.
