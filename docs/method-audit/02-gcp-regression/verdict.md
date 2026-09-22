# GCP regression — Verdict

**Verdict: Possibility B. Sparse anchors act as a weak, incomplete patch, not a "local ruler" — the evidence points to structural/scene-dependent distortion that a heavier model is needed to address.**

## The test the note itself proposed

> Possibility A: a few known height points "dramatically fixes" the tile → DAv2 just needs a local ruler.
> Possibility B: a few known points still leave massive errors → DAv2 is structurally distorting shape, and a heavier model (RDAH-Net) is required.

Using only the verified numbers pulled from the repo's own saved output:

## Evidence against Possibility A

1. **At the anchor counts the framing literally names (3-5), the naive
   sparse-anchor test does not come close to "dramatically fixing" anything.**
   3 anchors: mean MAE 16.4 m; 5 anchors: mean MAE 9.1 m — both far larger
   than the scene's own mean AGL (~3.4-3.6 m). Even 2 anchors produces
   catastrophic blowups (one tile hits 3,711 m MAE). A "local ruler" that
   needs 3-5 points and still leaves errors several times the signal's own
   mean is not evidence of a model that already has the shape right.

2. **Rank/shape correlation (Spearman) barely moves across every intervention
   tried**, from the pre-anchor global-isotonic baseline (0.446) through the
   best-tuned deployable configuration found across four stages of
   engineering, Grid+Huber+20 (0.471) — an absolute gain of about 0.025.
   If DAv2 already had the scene's shape right and just needed rescaling, a
   correctly-placed local ruler should recover most of the true ordering.
   It recovers almost none of it.

3. **Making the ruler more local makes things worse, not better.** Stage 3's
   quadrant-wise affine (four independent local rulers instead of one) is
   clearly worse than a single tile-wide ruler (MAE 3.356 m vs 3.131 m, RMSE
   5.515 m vs 4.482 m, Pearson 0.459 vs 0.526). If the problem were "DAv2's
   shape is right, it just needs the right local scale," adding *more*
   locality should help, not hurt.

4. **The residual left after the best global ruler is spatially structured
   but not a smoothly interpolatable field.** Stage 4A (oracle) shows the
   residual is strongly spatially autocorrelated (neighbor correlation
   ≈0.97) and a simple linear spatial term gives a real, if modest,
   improvement (MAE +0.061 m mean, majority of CV folds). But Stage 4B shows
   a flexible smooth interpolator (RBF) of that same residual **fails
   completely to generalize** — RMSE improves on 0/50 tiles for every
   smoothing strength tested, MAE on at most 2/50. A true "local ruler"
   miscalibration (smoothly varying scale/offset across the tile) should be
   exactly the kind of thing a smooth spatial interpolator recovers. It
   doesn't. The structure that's there looks local and content-dependent
   (what's actually at that pixel — building vs. tree vs. road), not a
   smooth geometric drift a ruler could track.

## What sparse anchors do accomplish

Not nothing: going from the global-isotonic baseline to the best deployable
sparse-anchor configuration (Grid + Huber + 20 anchors) does reduce MAE from
3.69 m to 2.93 m (~21% relative) and RMSE-optimal OLS variant gets RMSE down
from 5.31 m to 4.48 m (~16% relative). This is a real, measurable, and
reproducible gain — anchors are not worthless. But it's a modest rescaling
gain layered on top of an already-weak baseline, not the "dramatic fix" that
would flip this toward Possibility A. Absolute errors after all four stages
of engineering remain in the multi-meter range on a scene whose own mean
height is a few meters.

## Conclusion

The evidence, taken end-to-end across the naive test, the placement-strategy
study, the regression-robustness study, and the spatial-residual study, is
consistent and points one direction: sparse metric anchors give a small,
genuine absolute-scale correction but do not fix DAv2's underlying
scene-dependent ordering/shape errors, and attempts to make the correction
more spatially expressive (quadrant-wise, smooth RBF residual) make things
worse rather than better. This is the Possibility B signature the note set
up in advance — DAv2 is not just missing a scale factor on an otherwise
correct shape estimate, and a heavier, scene-aware model (RDAH-Net, or the
semantic-prior direction visible elsewhere in
`data/dfc2019/experiments/semantic/`) is the next step the project's own
evidence justifies, not further tuning of the anchor/regression/placement
recipe within this global-affine family.
