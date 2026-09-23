# 07 — Does Method 6 generalize? GAMUS, forests/mountains, India, deliverables (2026-09-23)

**Question.** Method 6 (the DFC2019 best: 1.980 / 3.492 / 0.745 / 0.656) had only ever been tested on DFC2019:
2 US cities, urban, one satellite sensor. The SIH26175 brief (`docs/SIH26175_problem_statement.md`) scores 50% on
DSM accuracy and requires stability across **urban, sparse, hilly and forested** terrain. This session tested
Method 6 on independent data, and would have improved it with GAMUS, the brief's recommended dataset.

**Protocol.**
- Every decision rule was committed before its part ran. Post-hoc readings are labelled.
- Full chronological record: `log.md` (this folder).
- Statistics: tile-level bootstrap (10,000 resamples, seed 0, 95%), paired Wilcoxon with win counts, Holm across candidates.
- Every number states its aggregation.

## Parts at a glance

| part | what | outcome |
|---|---|---|
| A | GAMUS acquisition and leakage | The HF GAMUS is the **3-city** version (DC, NYC, PHL; 8,724 tiles; 80 GB; CC-BY-4.0). The paper's JAX and OMA tiles *were DFC2019 tiles* and were removed upstream. **Leakage vs. Method 6's training set: 0** (by provenance, and confirmed by byte hash on all 2,861 test tiles). |
| B | Method 6 zero-shot on GAMUS test (2,861 tiles, 3 seeds × 4 folds) | **Does not generalize** (pre-registered rule). It beats the oracle on MAE, Pearson and Spearman, but **loses on RMSE in all 3 cities**, for every seed. |
| C | Fine-tune on GAMUS | **Pre-registered stop at C.0.** No GAMUS city is leaf-on (DC 0.06, NYC 0.03, PHL 0.32 vs. a 0.40 threshold), so the fine-tune was not run. |
| D | Forested and mountainous terrain vs. airborne LiDAR (USGS 3DEP; NEON's API needed a token) | **Canopy ceiling confirmed:** predicted p95 10.6 m vs. LiDAR 37.7 m. **The composed DSM does not add value** for any DEM. FABDEM + AGL lowers RMSE, but only by fixing bias; GLO-30 and SRTM + AGL are worse. |
| D.3 | Sikkim composed DSM vs. ICESat-2 and GEDI (descriptive) | The composed DSM is about at GLO-30's level and better than FABDEM against canopy-top GEDI. Small n. |
| E | Deliverables audit (read-only) | **No brief deliverable is fully met.** 3 are partial and 6 missing. "DEM ELEVATION" is correctly labelled (DEM-only). `docs/deliverables-audit.md`. |
| F | Sentinel-2 terrain RF residual (Song, Chen & Yokoya 2026; held-out ICESat-2 tracks, 32 tiles) | **Not adopted** (either variant). Both lose to a per-tile linear residual (8–9/32). Median RMSE: raw FABDEM 1.776, linear 1.467, RF A 2.436, RF B 1.425. Post-hoc: the linear gain is offset-only (bRMSE n.s.). |

## Key numbers

**B: GAMUS test.** 2,848 scored tiles; mean of tiles; 95% CI.

| | MAE | RMSE | Pearson | Spearman | var ratio (pooled) |
|---|---|---|---|---|---|
| Method 6 (seed 42; seeds 43 and 44 within 0.03) | **3.130** [3.017, 3.246] | 4.583 [4.435, 4.739] | **0.638** | **0.583** | 0.305 |
| Oracle per-tile OLS (DAv2-L) | 3.474 [3.387, 3.564] | **4.426** [4.319, 4.536] | 0.491 | 0.425 | 0.603 |
| *Method 6 + per-tile OLS (descriptive)* | *2.853* | *3.908* | *0.635* | *0.580* | *0.607* |

- Method 6 compresses tall structures:
  - 20–30 m objects are predicted at 7.0 m, and ≥ 50 m at 17.2 m (pixel-pooled means);
  - GAMUS trees of 20–30 m come out at 5.4 m.
- Stability across landscapes (mean-of-tiles MAE, building / sparse / tree): Method 6 3.72 / 1.29 / 5.03 (spread 3.7 m); oracle 3.87 / 3.45 / 4.69 (spread 1.2 m).

**D: forest and mountain.** 8 windows × 600 m; Olympic WA, Tahoe CA, Great Smoky Mtns TN, MLBS VA; NAIP → 0.3 m.

| | RMSE raw → composed | bias-removed RMSE raw → composed | verdict |
|---|---|---|---|
| FABDEM | 11.36 → 8.56 (7/8, Holm p = 0.031) | 7.75 → 7.60 (5/8, p = 1.0) | does not add value (bias-only gain) |
| GLO-30 | 9.16 → 12.84 (0/8) | 7.13 → 7.45 | does not add value (harms) |
| SRTM | 9.63 → 11.78 (2/8) | 8.38 → 8.31 | does not add value (harms) |

## What this means for the product and the brief

1. **Method 6 is a strong in-domain model, not a general one.**
   - Out of domain it keeps a real advantage in *ranking* heights: Pearson +0.15, Spearman +0.16 over the oracle, on every city and seed.
   - It loses the *scale* of tall objects, and RMSE punishes that.
2. **Canopy is the systematic failure.** It fails on GAMUS (leaf-off imagery) and on US leaf-on forest (3DEP), and it matches Sikkim's under-tall forest.
   - The cause is the training distribution (DFC2019 trees top out at about 24 m).
   - GAMUS cannot fix this: it is leaf-off. Fixing it needs leaf-on, tall-forest supervision.
3. **For a metric DSM, the brief's "DEM + predicted height" route does not beat the DEM alone** on forested and mountainous terrain.
   - FABDEM + AGL recovers some of the missing canopy *level*, but not its shape.
   - On India's 10 m Sentinel-2, the defensible products remain DEM-only (FABDEM for terrain, GLO-30 for surface).
4. **Deliverables:** the app currently meets none of the brief's deliverables in full (`docs/deliverables-audit.md`).
   Wiring anything in is the user's decision.
