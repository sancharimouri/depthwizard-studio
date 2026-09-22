# DEM-stat anchoring — Verdict

**Verdict: Does not generalize. Confirms the need for scene-conditioned/learned methods, by the project's own predicted criteria.**

## Does DEM-stat anchoring generalize on held-out data?

No, not usefully, on either metric that matters:

- **Absolute scale (MAE/RMSE):** on the untouched 50-tile benchmark, global
  linear calibration gets MAE 3.72 m / RMSE 5.36 m; global isotonic gets MAE
  3.69 m / RMSE 5.31 m. The benchmark's own mean AGL is 3.43 m. A typical
  error that's larger than the mean of the thing being predicted is not a
  usable metric-height product, even though it beats a flat per-tile-mean
  guess (mean within-tile AGL std is 5.45 m, so there is *some* real signal —
  just not enough).
- **Rank/shape (Pearson/Spearman):** unchanged by calibration, as it
  mathematically has to be for the linear model, and barely moved (0.539 →
  0.556 Pearson, Spearman flat at ~0.446) for the more flexible isotonic
  model. In-sample R² on the 200-tile calibration fit was already weak
  (0.143); the held-out numbers show that weakness held, not worsened —
  consistent, and conclusively negative rather than a fluke of overfitting.

## Does this support moving to the learned methods, as the note itself predicted?

Yes — and this can be stated using the project's own stated test rather than
an outside judgment call. `workflow_report.md` §8 (written before the
calibration stage) lays out the exact diagnostic in advance: *"If Spearman
remains low after calibration, then the problem cannot be solved by a global
monotonic mapping alone; scene/image/semantic/context-dependent information
is needed."*

Spearman on the held-out benchmark after the best (isotonic) calibration is
**0.4461** — statistically the same as the **0.4460** it was before any
calibration at all. By the criterion the project itself set in advance, this
is precisely the failure signature that justifies moving past global
statistical anchoring to the next stages already named in
`workflow_report.md`'s planned progression: sparse/GCP anchoring, semantic
prior, learned scale modulation, RDAH-Net fusion (all of which already have
their own experiment folders under `data/dfc2019/experiments/`).

## Scope note

This verdict is specific to *global* (scene-agnostic) statistical
DAv2→height anchoring — the method audited here. It does not by itself say
whether any of the downstream methods succeed; that's outside this audit
(`01-dem-stat-anchoring`).
