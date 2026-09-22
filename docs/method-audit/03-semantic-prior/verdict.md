# Semantic prior (Method 3) — Verdict

**Verdict: the rejection is confirmed by the real files, and the specific
failure mode lines up exactly with what Method 4 is designed to address.**

## Is the rejection real?

Yes, on every axis checked:

1. **Test 1 and Test 2 numbers reproduce exactly** from
   `method3_results.json` and `method3_spatial_cv_results.json` — to the
   last quoted decimal, no discrepancy.
2. **Baseline and Method 3 are the same fitting scheme in both tests**
   (plain `sklearn.LinearRegression`, Method 3 = Baseline + `Pb` +
   `D×Pb`) — confirmed by reading `evaluate_prior.py` and
   `evaluate_prior_spatial_cv.py` in full. This is a genuine apples-to-apples
   ablation, not two different baselines sharing a label.
3. **Same 50 tiles** across Methods 1/2/3 (dav2_baseline manifest,
   method3_results.json, method3_spatial_cv_results.json all match exactly)
   — the comparison is on equal footing.
4. **The hygiene rule holds in code**: zero references to DFC CLS in any of
   the three Method-3 scripts.
5. **The pattern itself is the textbook overfitting signature**: Method 3
   *helps* in-sample (MAE 3.03→2.90 m, Pearson 0.546→0.584, both real
   improvements under Test 1) and *fails to transfer* under spatial holdout
   (MAE flat at 3.39 m either way, but RMSE *worsens* 4.58→4.91 m, Pearson
   *worsens* 0.582→0.554, Spearman *worsens* 0.509→0.486). A global linear
   interaction term with 2 extra free parameters found signal that
   correlates with `Pb` and `D×Pb` on the training pixels but doesn't
   generalize spatially — exactly what you'd expect from a model with just
   enough extra capacity to fit tile-specific quirks (the exact quadrant
   boundaries where `Pb` happens to correlate with local elevation noise)
   without capturing anything transferable about the D→AGL relationship.

The one thing that is **not** independently confirmed by any file in this
repo is Sundarbans' "useful near-zero-building case" framing — it's a true
description of the raw data (0% building coverage, confirmed in the QC PDF)
but zero evaluation ever ran against it. It doesn't undermine the DFC
verdict above (Sundarbans was never part of Test 1/Test 2), but it shouldn't
be cited as supporting evidence for anything beyond "this AOI happens to
have no buildings."

## Does the failure mode line up with what Method 4/5 address?

**Yes, directly.** Read `scripts/evaluate_method4.py`'s docstring and design:
Method 4 uses the **identical 4-fold spatial-quadrant holdout protocol** as
Method 3's Test 2 (train on 3 quadrants, hold out the 4th, across all 50
tiles) — so it's evaluated on the exact axis Method 3 failed on. But instead
of a single global linear term added to the whole tile, Method 4 trains a
small CNN on 64×64 patches (RGB + normalized DAv2 in, multiplicative scale +
additive residual out), i.e. a **spatially-local, nonlinear, learned**
correction rather than one global affine-plus-interaction coefficient per
tile/fold. That is precisely the axis Method 3's failure exposed: a model
with only enough flexibility to fit global per-tile statistics (2 extra
scalar coefficients) picks up in-sample noise correlated with `Pb`, but has
no way to represent the kind of local, spatially-varying correction that
would actually generalize. Method 4 is architected to have exactly that
local/nonlinear capacity instead.

(Method 4's own verdict is still `TBD` in this repo — `docs/method-audit/04-learned-scale-modulation/verdict.md`
has not been filled in — so this states design *alignment* with Method 3's
failure mode, not a claim that Method 4 actually succeeds. That's a separate
audit.)

## Practical note independent of the DFC verdict

Kolkata — one of the four live regions in the DepthWizard2 frontend — has a
real, visually-confirmed GlobalMLBuildingFootprints coverage gap (a
contiguous rectangular void covering roughly a quarter of the AOI, confirmed
against the systematically-repaired maximal partition set, not an artifact
of incomplete downloading). Bengaluru's semantic raster is currently missing
from disk entirely (regenerable, but not present as of this audit). Neither
fact changes whether Method 3 should be the production correction method —
Method 3 is being rejected regardless, on its own DFC evidence — but any
future method that wants to reuse GlobalMLBuildingFootprints as a Sentinel-
side prior inherits both constraints: it would need a different or repaired
building-footprint source before it could touch Kolkata, and would need to
regenerate Bengaluru's raster before extending to that city.
