# 07 — Verdict (2026-09-23)

All rules were pre-registered in `log.md` before the part they govern ran.

## Pre-registered outcomes

| rule | outcome |
|---|---|
| **B: "Method 6 generalizes"** (all 4 metrics beat the oracle with non-overlapping CIs, and wins in all 3 cities, for every seed) | **FAIL.** RMSE is worse (+0.16 m [+0.09, +0.23]) in every city for every seed. MAE, Pearson and Spearman are clearly better. |
| **C.0: leaf-on pre-check** (some city with median tree-pixel ExG share ≥ 0.40) | **STOP.** DC 0.06, NYC 0.03, PHL 0.32. The GAMUS fine-tune was not run, so the C adoption rule does not apply. |
| **D(b): composed DSM "adds value"** (beats the raw DEM on RMSE **and** bias-removed RMSE; Holm p < 0.05) | **FAIL for all three DEMs.** FABDEM + AGL wins RMSE (7/8, p_Holm 0.031) but not bias-removed RMSE (5/8, p 1.0). GLO-30 and SRTM + AGL are worse on RMSE. |
| **F: RF terrain residual adopted** | See the Part F entry in `log.md`. The expected outcome was negative. |

## Adopted / not adopted

- **No new product model is adopted.** Method 6 stays the DFC2019 research best (1.980 / 3.492 / 0.745 / 0.656).
  It is now explicitly flagged as **not generalizing**: on GAMUS, a different city, sensor and city type, it loses RMSE to the oracle.
- **Product recommendation, unchanged and now tested on forest and mountain:** DEM-only.
  - FABDEM is the terrain product and GLO-30 the surface product.
  - "DEM + Method 6 height" is not adopted for any DEM.
- **Retired as a plan:** "Fine-tune Method 6 on GAMUS to fix the canopy ceiling." GAMUS is leaf-off in 2 cities and
  mixed-season in the third. The canopy fix needs leaf-on, tall-forest supervision.

## What holds up and what doesn't

**Holds:**
- Method 6's advantage in **ranking** heights over a scale-fitted frozen DAv2 transfers out of domain:
  Pearson 0.638 vs. 0.491 and Spearman 0.583 vs. 0.425 on 2,848 GAMUS tiles, in all 3 cities.
- It is also better on MAE in all 3 cities.

**Doesn't hold:**
- **Metric scale for tall objects.** Pooled variance ratio 0.30 (DFC2019 0.48–0.66). On US forest vs. LiDAR, 0.12.
  RMSE is where the brief's accuracy score is most sensitive.
- **Landscape stability.** The per-class MAE spread is 3.7 m for Method 6 against 1.2 m for the oracle. Method 6 is
  best on sparse ground and worst on trees.
- The brief requires stability across urban, sparse, hilly and forested terrain; **Method 6 does not provide it.**

## Caveats

- **GAMUS:** the HF version has 3 cities, not the paper's 5. GSD is only estimated (about 0.3 m for DC). The
  official splits leak spatially in DC; this doesn't matter for zero-shot use.
- **Part D:** USGS 3DEP + NAIP replaced NEON, because NEON's data API now needs a token.
  - NAIP is 0.6–1.0 m, resampled to 0.3 m.
  - LiDAR and imagery dates differ by 207–258 days, and by 5.3 years at MLBS.
  - GRSM's LiDAR ground classification is shaky on steep terrain.
  - n = 8 windows.
- **Part D.3** is descriptive only, with n = 5–84 points per crop, and has gross ICESat-2 and GEDI outliers.
- The oracle uses each tile's own dense LiDAR; it is an upper-bound baseline, not a deployable method.
  Method 6 + per-tile OLS, shown for description only, needs that same LiDAR.

## Open items created by this session

1. **Leaf-on, tall-forest supervision** for the canopy ceiling. Candidates: NEON AOP (needs a NEON API token), 3DEP + leaf-on NAIP at
   scale (the Part D machinery exists), or a PHL leaf-on tile subset selected by ExG (post-hoc idea, not run). Each
   needs its own pre-registration, including the Sikkim non-regression criterion committed in C.0.
2. **The NEON rerun of Part D**, if a token is provided.
3. **Deliverables gaps** (`docs/deliverables-audit.md`): a user decision, frontend frozen.
