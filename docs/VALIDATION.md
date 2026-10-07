# Validation

## How results were validated

- **Independent references.** DFC2019 is scored against dense airborne lidar. Sentinel-2 over India (32 tiles) is
  scored against ICESat-2 ground photons and canopy segments and against GEDI, all independent lidar. Where a product
  used a reference in its own making (FABDEM used GEDI canopy height; ETH canopy height was trained on GEDI), that
  pairing is flagged as non-independent.
- **Held-out data only.** DFC2019 methods are scored on held-out spatial quadrants (4-fold). The comparator is the
  per-tile OLS oracle: height above ground = a · (frozen Depth Anything V2) + b, fitted on all valid lidar pixels of the
  other three quadrants of the same tile and scored on the held-out quadrant. It is an upper-bound-style reference: the
  best a single affine rescale of the depth model can do with dense truth from the same tile.
- **Uncertainty.** Tile-bootstrap 95% confidence intervals, paired per-tile Wilcoxon tests, Holm correction where
  several are made.
- **Pre-registration.** Each decision rule was committed to the audit log before its test ran. Decisions made after
  seeing results are labelled post-hoc.
- **Offset guard.** On Sentinel-2 a "pass" must also hold on bias-removed RMSE, so a constant offset can't pass as an
  improvement.

## Pre-registered rules that decided the outcome

| Test | Rule | Result |
|---|---|---|
| Method 6 seeds | The headline stands only if every seed beats the oracle on all four metrics | Held: 3 of 3 seeds |
| Method 6 on GAMUS (2,861 aerial test tiles, 0 leakage; 2,848 scored, 13 skipped for too few valid lidar pixels) | Must beat the oracle, including on RMSE | **Does not generalize:** better MAE, Pearson and Spearman; worse RMSE (4.583 vs. 4.426 m) in every city and seed, because it compresses tall objects |
| GAMUS fine-tune | Run only if a city is leaf-on | Stopped: no GAMUS city is leaf-on |
| RDAH-Net on Sentinel-2 | Rank correlation against ICESat-2 and GEDI | Closed: Spearman ≈ 0 against both |
| FABDEM residual model (random forest) | Must beat raw FABDEM against ICESat-2 | Not adopted |

## Results

- **DFC2019 (Method 6):** validated on DFC2019 (US cities, satellite imagery); did not generalize to GAMUS by RMSE.
  3-seed result MAE 1.990 ± 0.010 m, RMSE 3.504 ± 0.026 m, Pearson 0.743 ± 0.002, Spearman 0.656 ± 0.0003, against
  the oracle's 3.392 / 4.579 / 0.582 / 0.509.
- **Sentinel-2, 10 m:** no model added value over a plain DEM against independent lidar. FABDEM is the best terrain
  (median RMSE 1.78 m against ICESat-2 ground, better than GLO-30 on 32 of 32 tiles). GLO-30 is the best surface.

## Negatives we kept

These are recorded, not hidden:
- Frequency fusion's early "21 of 25 tiles" win was the DEM's own result. Against its DEM-only control it won 10 of 25.
- Three learned CNN corrections on Sentinel-2 failed, each in a different way (memorising the interpolated DEM, too
  little data at 30 m, memorising the training target).
- Depth Anything V2, CHMv2 and ETH canopy height all fail as detail sources added to a DEM. No DEM + canopy product
  beats a plain DEM.
- Method 6 staged on Sentinel-2 was stopped at the first fold (1 of 25 tile wins against the DEM).
- On leaf-on US forest, "DEM + predicted height" doesn't beat the DEM alone for SRTM, GLO-30 or FABDEM.
- A retrained Method 6 failed its first pre-registered agreement check. That check turned out to measure memorisation;
  the failure stays on record, and a fair re-test on unseen data passed.
- Several of the project's own earlier claims were overturned, among them a vertical-datum bug and an input-scale bug.
  Each correction is logged.

The full audit log is kept in our private research repository and can be shared with the jury on request.
