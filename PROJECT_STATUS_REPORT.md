> ## Addendum — 2026-09-23 (read this first; the report below is the earlier snapshot, kept as written)
>
> **Current state, in one place: `docs/method-audit/final-comparison.md`** (tables, CIs, independence flags, every number
> traced to a committed file). Chronology and gaps: `docs/method-audit/00-audit-log.md`.
>
> - **DFC2019 (dense-LiDAR proxy benchmark):** Method 6 (full DAv2-Small fine-tune, height-balanced recipe) is the best
>   result — MAE 1.980 m / RMSE 3.492 m / Pearson 0.745 / Spearman 0.656 (mean of 4 spatial-quadrant folds), beating
>   the per-tile-OLS oracle (3.392 / 4.579 / 0.582 / 0.509) on all four metrics and on 47–49 of 50 tiles per metric.
>   Seed robustness (2 extra seeds) and variance ratio: see final-comparison.md.
> - **RDAH-Net (Method 5):** on DFC2019, fine-tuned FT-2 not adopted (loses to Method 6); Swiss zero-shot shows
>   fine-tuning *helps*; Track1 zero-shot (0.716 Pearson) came from a checkpoint trained on 41/50 benchmark tiles.
>   **On Sentinel-2: closed** by a pre-registered rule (Darjeeling height-above-ground Spearman ≈ 0 vs ICESat-2/GEDI).
>   The "checkerboard" reason below turned out not to be diagnostic (the artifact appears in-domain too).
> - **Sentinel-2 / India (the deployment domain):** no model-based correction helps. **Frequency fusion's 21/25 headline
>   was the DEM's own result** (fusion = its DEM-only control, 10/25, p = 0.85). Recommended baselines on all 32
>   benchmark tiles, against independent ICESat-2: **terrain → FABDEM** (median RMSE 1.78 m, beats GLO-30 on 32/32);
>   **surface → raw Copernicus GLO-30** (beats SRTM, R4 offset-guarded). No DEM + canopy-height product and no
>   depth-model detail source passes. 10 m canopy models do carry a height signal (mostly between landscapes).
> - **Closed / not recommended:** CNN correction on Sentinel-2 (3 failures), RDAH on Sentinel-2, DAv2/CHMv2/ETH as
>   detail add-ons, a learned Sentinel-2+GEDI route (ETH ceiling). **Open:** TSE-Net (untouched); Track1 seen-vs-unseen
>   split (low priority).
>
> ---
>
Based on a full re-read of every audit doc, JSON, and CSV in `docs/method-audit/` and `data/dfc2019/experiments/` — here's the complete state.

## Method 1 — Global statistical DAv2→AGL calibration
**Techniques:** global linear fit, global isotonic regression (200-tile fit, evaluated on the untouched 50-tile benchmark).

| | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| Linear | 3.72m | 5.36m | 0.539 | 0.446 |
| Isotonic | 3.69m | 5.31m | 0.556 | 0.446 |

**Verdict: CLOSED, permanently.** Spearman is statistically identical before/after calibration (0.4460→0.4461) — proof by the project's own pre-stated test that no global monotonic mapping can fix this. In-sample R² was already weak (0.143).

## Method 2 — Sparse-anchor (GCP) regression
**Techniques tried:** anchor counts 2/3/5/10/20/30/50/100; placements Grid/Random/Spatial-height-stratified; regressors OLS/Huber/RANSAC (two separate RANSAC studies); spatial models tile-wide vs. quadrant-wise; residual modeling constant/linear/quadratic + smooth RBF.

Best deployable: **Grid+Huber+20** = MAE 2.929m / RMSE 4.718m / Pearson 0.532 / Spearman 0.471.

| Technique | Result |
|---|---|
| Grid placement | **passed** — best placement at every anchor count |
| Random placement | passed, slightly worse than Grid |
| Spatial+height placement | **failed** — worse than Grid/Random at *every* anchor count; real effect (forces high-leverage extreme-height points into a 5-20 point OLS fit), not a bug |
| OLS | passed (baseline-of-the-baselines) |
| Huber | passed — better MAE, worse RMSE than OLS |
| RANSAC (Stage 2, default params) | **failed** — worse than OLS/Huber on every metric, Pearson collapses to ~0.13-0.25 |
| RANSAC v2 (5-10x anchors, tuned threshold, 5000 trials) | **failed again** — confirms structural mismatch, not undertuning; RMSE gets *worse* with more anchors (4.85m→6.02m) |
| Tile-wide affine | passed |
| Quadrant-wise affine (OLS and Huber both tested) | **failed** — worse than tile-wide on every metric under both regressors |
| Linear spatial residual (oracle) | passed — real gain, majority of folds |
| Quadratic spatial residual | **failed** — overfits |
| Smooth RBF residual | **failed** — improves RMSE on 0/50 tiles |
| "Grid-wise local affine" (originally planned) | **never run** — no script/config/output exists anywhere |

**Verdict: CLOSED as a standalone method.** "Possibility B" confirmed — DAv2 has structural, scene-dependent shape distortion; a local ruler cannot fix it, and every attempt to make the ruler *more* local made things worse. This is what justified moving to learned methods.

## Method 3 — Semantic prior (building-probability linear term)
`AGL = β0+β1·D+β2·Pb+β3·(D×Pb)`, Pb = HOTOSM building probability.

| | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| Test 1 (in-sample) baseline→M3 | 3.03→2.90 | — | 0.546→0.584 | — |
| Test 2 (spatial holdout) baseline→M3 | 3.392→3.395 | 4.579→4.907 | 0.582→0.554 | 0.509→0.486 |

**Verdict: CLOSED.** Textbook overfitting — helps in-sample, fails to transfer spatially. A global linear term has just enough capacity to fit tile-specific quirks, nothing transferable. Directly motivated Method 4's local/nonlinear (CNN) design.

## Method 4 — Learned scale modulation (CNN, local per-pixel correction)

**v1 (original):** zero-init heads, Huber-only loss + smoothness penalty, 12 epochs, 12 patches/quadrant. Audit found its self-reported baseline was a strawman (pooled global affine, 4 fits) vs. Method 3's real baseline (200 per-tile fits). Against the *real* baseline: MAE 3.27m (only +3.5%, not the claimed +16%), RMSE 5.17m (**-13%, worse**), Pearson 0.547 (worse), Spearman 0.485 (worse). Root cause confirmed mechanistically: regression-to-the-mean (variance ratio 0.136, OLS slope 0.187, predicted max 22.5m vs true max 97.4m) from zero-init + magnitude-only loss + active smoothness penalty + starved training budget.

**v2 rework (this session, systematic):**

| Phase | Config | MAE | RMSE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| 0 | Corrected baseline (per-tile OLS) | 3.3924 | 4.5787 | 0.5824 | 0.5093 |
| 1 | Dense coverage + 60 epochs (smooth=0.01) | 3.0999 | 5.0210 | 0.5600 | 0.5111 |
| 2 | + building channel alone | 2.8836 | 4.8712 | 0.5596 | 0.5258 |
| 2 | + ground-plane (broken: w=2.0,t=0.5m) | 3.2481 | 5.5302 | 0.4865 | 0.5047 |
| 2 | + rank loss alone | 3.0789 | 4.9513 | 0.5737 | 0.5285 |
| 2.5 | ground-plane corrected (w=0.1,t=0.1m) | 2.9519 | 4.9560 | 0.5556 | 0.5191 |
| 2.5 | building+rank (no gp) | 2.8477 | 4.7971 | 0.5691 | 0.5373 |
| 2.5 | building+rank+gp combined | 2.8171 | 4.8568 | 0.5631 | 0.5367 |
| **2.5 R2** | **building+rank v2 (rank=0.5)** | **2.8803** | **4.7751** | **0.5835** | **0.5438** |
| 2.5 R2 | groundplane-light (gp=0.05) | 2.8316 | 4.8098 | 0.5696 | 0.5327 |
| Post-verdict | + SID ordinal constraint | 2.9090 | 4.8057 | 0.5852 | 0.5449 |

**Passed:** Phase 1's training-budget fix (clean win on all 4 metrics vs. v1). Building channel (broad win, best calibration of the whole rework). Rank loss (moved Pearson/Spearman). Ground-plane term *once corrected* (scoping bug, not a bad idea — corrected version tripled its own variance ratio).
**Failed:** Ground-plane at its original weight/threshold (worst config of the whole audit — a scoping bug, root-caused not just observed). Smoothness-weight sweep (barely mattered, contrary to original hypothesis). SID ordinal-discretization loss (real, paper-faithful adaptation, genuine negative result — movement smaller than fold-to-fold noise).

**Verdict: `phase2_building_rank_v2` is the best result of this entire audit, tuning declared CLOSED.** Beats baseline on 3/4 metrics (MAE +15.1%, Pearson +0.2% — first ever config to beat baseline Pearson, Spearman +6.8%). RMSE gap narrowed >13%→4.8%→4.3% across rounds — real trend, never closed.

## Method 5 — RDAH-Net fusion
**Zero-shot on Sentinel-2** (Darjeeling): visually confirmed regular checkerboard-grid artifacts across the whole output — rejected, matches CLAUDE.md's stated reasoning. *(2026-09-23: still rejected on Sentinel-2, but for a different, pre-registered reason — no height signal at 10 m; the checkerboard is intrinsic to RDAH output and also appears in-domain. See final-comparison.md.)*
**Zero-shot on DFC2019** (as "RS3DAda"): not zero-shot at all — 49/50 benchmark tiles are literally in RS3DAda's own training split (file-level diff confirmed).
**Fine-tuned on DFC2019** (real experiment, `RDAH-FT-1`, 4-fold spatial CV, 5 epochs, lr=1e-5, never before reported): pooled MAE 2.906m / RMSE **6.659m** / Pearson 0.513 / Spearman 0.527, across 52.4M pixels. Per-fold Pearson swings 0.254→0.607 — genuinely unstable, and fold sizes are uneven (7-18 test tiles). Vs. honest baseline: wins MAE (-14.3%) and Spearman (+3.5%), but RMSE is **45% worse** — the worst RMSE of the entire audit.

**Verdict: genuinely unfinished, not rejected or accepted.** *(2026-09-23: superseded — Method 5 docs are now complete; FT-2 not adopted; see `05-rdah-net-fusion/verdict.md` §6–8.)* `verdict.md`/`summary.md`/`gaps-and-fixes.md` for Method 5 are all still empty templates despite this real data existing on disk — this is the clearest "trailed off by mistake" in the project.

## Adjacent tracks (not part of the numbered 1-5 sequence)

**Geometry recovery for DFC2019 (ICESat-2/sparse-LiDAR feasibility):** DFC2019 tiles have zero georeferencing on disk. Attempted recovery via the extended US3D dataset's real point clouds — first pass wrongly concluded "37/37 tiles recoverable" (a filename-coincidence, not verified against pixel content). A rigorous spatial join (20,176 correlation checks) caught this: **0/50 real matches**. **CLOSED as attempted; one avenue never tried** — the original DFC2019 Track1 imagery's own RPC metadata.

**India Sentinel-2 benchmark dataset (32 tiles, separate validation dataset, not a DFC method):** Built via real ICESat-2 coverage filtering, then a content-quality audit (WorldCover/SRTM/actual-SCL-cloud) found 11/32 failing — 6/8 agricultural tiles were literally centered on city coordinates instead of farmland. Two remediation rounds plus visual QC (which caught satellite swath nodata, SCL's known blind spots over water/thin cirrus, and off-season/permanently-turbid replacements the numbers missed) resolved all of it. **14/32 tiles changed, manifest updated and verified — CLOSED, complete.**

## Closed vs. open vs. trailed-off — direct answer

**Permanently CLOSED:** Method 1 (global calibration); Method 2's RANSAC (twice), Spatial+height placement, quadrant-wise/local-affine, smooth-RBF and quadratic residual models; Method 3 (global semantic-prior term); RS3DAda/RDAH-Net zero-shot (both contaminated and degenerate); Method 4 v2's weight-sweep tuning space and the SID loss direction; the extended-US3D point-cloud geometry recovery.

**Genuinely OPEN (not decided):** Method 5/RDAH-Net fine-tuning (only 5 epochs, uneven folds, no formal verdict — could go either way with real investment); DFC2019 Track1 RPC-based georeferencing recovery (never attempted); a genuinely different Method-4 architecture (the only lever left if the RMSE gap still matters, per the v2 rework's own final verdict).

**Trailed off by mistake (real work, never followed through):** *(2026-09-23: the items below are now written up — Method 5 docs, `00-audit-log.md`, `final-comparison.md`.)* Method 5's entire audit writeup (data exists, docs are empty templates); `00-audit-log.md` and `final-comparison.md` (both empty despite 4/5 verdicts already existing elsewhere); Method 2's planned "grid-wise local affine" model (never run, no trace); Sundarbans' "near-zero-building case" (asserted, never actually tested); Bengaluru's semantic building-footprint raster (existed, silently lost mid-repair, never regenerated).

## Best result / benchmark / baseline, and what's left

- **Baseline (the bar everything is measured against):** per-tile OLS, MAE 3.3924m / RMSE 4.5787m / Pearson 0.5824 / Spearman 0.5093.
- **Best result overall: `phase2_building_rank_v2`** (Method 4 v2) — MAE 2.8803m / RMSE 4.7751m / Pearson 0.5835 / Spearman 0.5438. Beats baseline on 3 of 4 metrics; RMSE remains the one never-beaten metric, now within 4.3%.
- **Production reality, unaffected by any of the above:** none of this is deployed. The live DepthWizard2 demo's ML pipeline is frozen by design — real DSM is the terrain source, DAv2 is shown only as a relative-depth visualization layer, and the "absolute elevation from RGB" claim stays explicitly framed as future research, not a shipped capability.
- **What's left, if pursued:** (1) write up Method 5 properly with more epochs/balanced folds before calling it either way, (2) try a genuinely different Method-4 architecture if the RMSE gap still matters, (3) attempt Track1 RPC recovery to finally unblock DFC2019 ICESat-2-guided correction, (4) housekeeping — fill in the empty audit-log/final-comparison docs now that the real verdicts exist to put in them.
