# Learned scale modulation (Method 4) — Verdict

**Verdict: the baseline discrepancy with Method 3 is resolved (not a data
error — a genuinely different, weaker baseline-fitting procedure).
Regression-to-the-mean is confirmed, quantitatively and mechanistically, as
the cause of "MAE down, correlation down." And the loss-function finding
points at a specific, concrete fix, not a vague direction — but it's a
narrower fix than "add a rank loss": the smoothness penalty and
zero-init/short-training budget need to be addressed too, or a rank term
alone won't be enough to move a model that currently barely deviates from
the identity mapping.**

## 1. Is the baseline discrepancy resolved?

Yes, completely. Method 3's Test 2 baseline is 200 independent per-tile
affine fits; Method 4's baseline is 4 pooled global affine fits (one per
fold, shared across all 50 tiles) — confirmed by reading both scripts, and
confirmed numerically down to the individual tile (`summary.md` §1:
Pearson/Spearman identical to 7 decimals for `JAX_004_006`/fold 0, because
correlation is invariant to which positive-slope affine coefficients were
used; MAE/RMSE differ because those coefficients do change the actual
predicted magnitude). Quadrant boundaries and valid-pixel masks are
identical between the two scripts — ruled out as contributors.

**This matters beyond just explaining a discrepancy.** It means Method 4's
own reported "baseline" understates how good a DAv2-only correction can
already be. Compared against Method 3's real per-tile baseline:

| | Method 3's real baseline | Method 4 |
|---|---:|---:|
| MAE | 3.39 m | 3.27 m (↓ ~3.5%, not the ~16% implied by Method 4's own baseline) |
| RMSE | 4.58 m | 5.17 m (↑ ~13% — **worse**) |
| Pearson | 0.582 | 0.547 (worse) |
| Spearman | 0.509 | 0.485 (worse) |

Against the correct baseline, Method 4 is not a mixed win (better MAE,
worse correlation) — it's worse on 3 of 4 metrics and only marginally better
on the 4th. "Retain as current best experimental direction" was written
against a weaker comparator than actually exists in this same repo.

## 2. Is regression-to-the-mean confirmed as the mechanism?

Yes, both statistically and mechanistically:

- **Statistically**: predicted variance is 13.6% of true variance
  (std ratio 0.369); the OLS slope of prediction-vs-truth is 0.187 (1.0 =
  calibrated); predicted max caps at 22.5 m against a true max of 97.4 m.
  This is the standard signature of a model whose loss is minimized by
  shrinking toward a central value when its conditioning signal is weak —
  not an open or surprising pattern once measured directly.
- **Mechanistically**: the loss is pure Huber (magnitude-only, no rank/order
  term) trained for 12 epochs from a zero-initialized (`= identity mapping`)
  starting point, with an explicit smoothness penalty on the scale map that
  actively discourages spatial variation. The learned scale map for a
  spot-checked tile is nearly flat (±4% around 1.0). Every piece of this is
  consistent with, and sufficient to produce, the observed compression —
  confirmed directly in the training code, not inferred.

## 3. Does the loss-function finding point at a specific, concrete fix?

**Partially — the note's own proposed next step ("modify the objective to
preserve rank") is the right general direction, but on its own it's
incomplete for this specific model.** Three things need to change together,
not just the loss:

1. **Add a rank/structure-preserving loss term** — e.g. a differentiable
   Spearman surrogate, or a pairwise ranking/margin loss between sampled
   pixel pairs within a patch, added alongside (not necessarily replacing)
   the Huber term. This directly targets what's currently unpenalized:
   nothing in the current loss cares whether pixel A is predicted taller
   than pixel B when the truth says so.
2. **Reduce or anneal the smoothness penalty** (`0.01 × smoothness_loss`) —
   as it stands, it's actively working against the goal of a spatially
   varying correction (`gaps-and-fixes.md` §2), and a rank loss added on top
   of an unchanged strong smoothness penalty may just learn a flat scale map
   that also happens to preserve rank slightly better, without actually
   producing the "spatially varying correction" the plan calls for.
3. **The zero-init / short training budget combination** means the model
   currently has limited opportunity to move far from the identity mapping
   at all; a rank loss added to the same 12-epoch, 12-patches-per-quadrant
   budget may simply not have enough training signal to learn much,
   regardless of loss function.

None of this contradicts the note's "next step" — it confirms rank/structure
preservation is the right target — but a fix that only swaps in a rank loss
without touching the smoothness weight or training budget would likely
under-deliver relative to what the note implies is achievable. This should
be scoped as three small, separately-testable changes (rank loss; smoothness
weight; training budget), not one.

## What this doesn't change

Method 3's Test 1/Test 2 verdict (already rejected — see
`docs/method-audit/03-semantic-prior/verdict.md`) is untouched by anything
found here. And the underlying reason any of these corrections struggle —
DAv2 relative depth's own weak, structurally-unreliable relationship to
absolute AGL, established across Methods 1–3 — is exactly the "weak
conditioning signal" that makes a magnitude-only loss shrink toward the mean
in the first place. Method 4's regression-to-the-mean isn't a bug unrelated
to the rest of this audit trail; it's the same underlying problem
(DAv2 doesn't reliably carry absolute-elevation information) showing up as a
training-dynamics symptom instead of a linear-regression symptom.
