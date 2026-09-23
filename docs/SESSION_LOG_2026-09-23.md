# DepthWizard2 (SIH26175) — Session Log, 2026-09-22 → 2026-09-23

Covers every session run from this account: from absorbing the handoff through
the research-track close-out, final commit `965a2a4`. For each stage it records
what was attempted, why, what worked, and what didn't, then the current state
and the files touched.

**Sources:** this planning conversation (the reasoning, prompts and reviews)
and the final reports Claude Code returned after each stage. Commit hashes are
the ones reported. Where a report didn't name a file, this log says so; run the
verification pass at the end to reconcile against `git log`.

---

## 0. Starting point

State inherited from `docs/HANDOFF.md` and `CLAUDE.md`, as of 2026-09-22.

**DFC2019 track (VHR, dense LiDAR ground truth).**
- Current best was Method 6: full DAv2-Small fine-tune with a twin (mean,
  log-variance) head, plus the height-balanced loss and sampler.
- It beat the oracle per-tile-OLS baseline on all four metrics, but on a
  single seed, with no variance-ratio check.
- Method 5 (RDAH-Net): zero-shot was "rejected (checkerboard)". The first
  fine-tune, RDAH-FT-1, was unstable across folds, and no verdict had been
  written.

**Sentinel-2/India track (10 m, sparse ICESat-2 plus coarse DEMs).**
- Frequency fusion (DEM low-frequency + DAv2 high-frequency) was the
  "deployable baseline", on the strength of 21/25 tile wins.
- CNN correction was deliberately stopped after three independent failures.
- Semantic-prior phase 2.3 had two unverified loose ends.

**Work already done before this account, carried in as context:**
- RDAH-FT-2: a 4-fold, 5-epoch run with a rank term and nested checkpoint
  selection, using `scripts/train_rdah_quadrant_cv.py`.
- An import bug in that script was fixed before a 1-epoch dry run.
- The session hit its usage limit before `HANDOFF.md` or `CLAUDE.md` were
  updated.
- Phase 2.3's follow-up mechanism checks came back clean, so that line was
  closed.

---

## Stage 1 — Absorb the handoff; document RDAH-FT-2

**What:** confirm the project state, then have Claude Code finish the
calculations and documentation FT-2 had left undone.

**Why:** the previous session ended mid-documentation. The standing 90%-usage
rule to update the docs hadn't fired.

**Issues raised before any writing:**
1. The "corrected zero-shot" RDAH result on DFC2019 appeared in neither
   handoff file. It was also anomalous: zero-shot beat both fine-tuned runs.
   That called for a check on where the number came from, and whether
   RDAH's training data overlapped DFC2019. Contamination was a live
   possibility, since RS3DAda had already been found to have 49 of 50
   benchmark tiles in its training data.
2. The FT-2 numbers mixed two ways of combining folds: MAE and RMSE were
   pixel-weighted, while the correlations were averaged across folds. FT-1's
   "pooled" numbers and Method 6's might each have been computed differently
   again. A pooled Spearman can't be rebuilt from per-fold summaries.
3. FT-2 failed the adoption bar in HANDOFF §7 regardless of Method 6: it beat
   the oracle on MAE and both correlations, but lost on RMSE.
4. The variance ratio (predicted variance over true variance) improved from
   about 0.02–0.05 to 0.13–0.22. That's real progress, but predictions were
   still badly compressed toward the mean.
5. The 90% doc-update rule wasn't written in either file.

**Done by Claude Code:** rebuilt the combined result from the four per-fold
JSONs using a script (not transcription), and filled in the 05 summary and
verdict, which had been empty templates.

**Result:**
- FT-2 is **not adopted**. It improves on FT-1, beats the oracle on 3 of 4
  metrics but not RMSE, and loses to Method 6 on all four.
- RDAH as a method stayed open, pending the zero-shot provenance question.
- Phase 2.3 was moved to closed.

**Files** (reconciled against git)
- Commits: `5bb45f7`, `be8ccac`, `9fb0ae2`
- Added: 
  - `data/dfc2019/experiments/rdah_quadrant_cv/rdah_ft2_aggregate.json`
  - `docs/HANDOFF.md`
  - `scripts/aggregate_rdah_ft2.py`
- Modified: 
  - `CLAUDE.md`
  - `docs/method-audit/05-rdah-net-fusion/{summary,verdict}.md` (2)
- Per commit:
  - `5bb45f7`: A `data/dfc2019/experiments/rdah_quadrant_cv/rdah_ft2_aggregate.json`, `scripts/aggregate_rdah_ft2.py`
  - `be8ccac`: M `docs/method-audit/05-rdah-net-fusion/{summary,verdict}.md` (2)
  - `9fb0ae2`: A `docs/HANDOFF.md`; M `CLAUDE.md`
- ⚑ The stage report listed `docs/HANDOFF.md` as *modified*. In git it is **added** (`9fb0ae2`
  is its first commit); it had been untracked.

---

## Stage 2 — Does RDAH's DFC2019 scale fix transfer to Sentinel-2? (inference only)

**What:** rerun RDAH zero-shot on the Darjeeling Sentinel-2 tile, which had
originally shown checkerboard artifacts, with the DFC2019 depth-scale fix
applied. Extend to the benchmark only if structure appeared.

**Why:** the 05 verdict flagged it as the next thing to try. Fine-tuning was
explicitly held back because it would reopen the data-starvation risk that
killed three earlier Sentinel-2 CNN attempts.

**Changes made to the prompt before firing, and why:**
- **Doc updates were made unconditional.** Claude Code can't see plan usage,
  so the 90% rule could never fire. The replacement: write docs after every
  step, and gate completion on `git diff --stat` showing both files changed.
- **Input fix or output fix?** A constant applied to the output can't remove
  a checkerboard, so the test only means something if the fix acts on the
  input. Claude Code was told to establish which, first.
- **Input normalization.** Sentinel-2 imagery might be uint16 reflectance,
  where dividing by 255 would be wrong on its own.
- **Order of steps.** The input-size question (1000 vs 1024 px) was moved
  ahead of the first run, since it could itself be the cause of the artifact.
- **Measurable criteria.** An FFT check for periodic peaks, and correlations
  both raw and after removing the spatial trend (the Darjeeling DAv2 sign
  flip showed raw correlation alone can mislead).
- **A decision rule committed before the benchmark run.**
- **Frequency-fusion swap test** added as a no-training use of RDAH.

**Findings:**
- The fix acts on the input: depth ×255 before the forward pass. It had
  never been committed.
- Inputs were already 8-bit, so RDAH's /255 on RGB was correct.
- The original checkerboard run reproduced exactly (difference 0.0). It had
  resized a 1007×1002 tile to 1024 and fed depth unscaled.
- RDAH only runs when both sides are multiples of 128 and at most 1024.
  Reflect-padding to 1024 was used.
- Claude Code noted that RDAH predicts height above ground, not terrain.
  This turned out to matter a great deal (Stage 3).
- The working-tree Darjeeling tile was an uncommitted 1118×1004 re-export,
  so the committed version was used instead.

**Result:**
- The scale fix cut the checkerboard's FFT peaks by 67× to 1,940×, but they
  stayed far above background. The 32 px peak, which lines up with the
  model's attention blocks, got stronger.
- Correlation with terrain was around zero or negative after removing the
  trend. Output was implausibly flat (median about 0.02 m).
- The stop condition fired, and the line was closed. It was later reopened
  on better grounds.
- Commits: `348b865`, `82689ac`, `b51b46f`, `d90859f`.

**Files** (reconciled against git)
- Commits: `348b865`, `82689ac`, `b51b46f`, `d90859f`
- Added: 
  - `data/sentinel2_benchmark/rdah_zeroshot/darjeeling/darjeeling_results.json`
  - `data/sentinel2_benchmark/rdah_zeroshot/darjeeling/{darjeeling_old_new_dem_dav2,darjeeling_zoom_center}.png` (2)
  - `scripts/rdah_sentinel2_zeroshot.py`
- Modified: 
  - `CLAUDE.md`
  - `docs/HANDOFF.md`
  - `docs/method-audit/05-rdah-net-fusion/verdict.md`
  - `docs/method-audit/sentinel2/sign-flip-detector.md`
- Per commit:
  - `348b865`: M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `82689ac`: A `scripts/rdah_sentinel2_zeroshot.py`; M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `b51b46f`: A `data/sentinel2_benchmark/rdah_zeroshot/darjeeling/darjeeling_results.json`, `data/sentinel2_benchmark/rdah_zeroshot/darjeeling/{darjeeling_old_new_dem_dav2,darjeeling_zoom_center}.png` (2); M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `d90859f`: M `CLAUDE.md`, `docs/HANDOFF.md`, `docs/method-audit/05-rdah-net-fusion/verdict.md`, `docs/method-audit/sentinel2/sign-flip-detector.md`
- Not committed (as the stage report said; confirmed absent from git):
  `data/icesat2_photons/darjeeling.csv` (untracked);
  `data/diagnostics/darjeeling/Darjeeling_RGB_committed_660ecb6.tif`,
  `data/diagnostics/darjeeling/rdah/{orig_resize_x1,pad_x1,resize_x255,pad_x255}.npy`,
  `data/diagnostics/darjeeling/rdah/step2_stdout.txt` (all under the gitignored `data/diagnostics/`).

---

## Stage 3 — Audit the premises ("it has to work")

**What:** rather than move the goalposts until something passed, check which
premises behind the negative result had never actually been verified.

**Why:**
- Changing tests until one passes would have turned a real negative into a
  false positive.
- Some premises were genuinely untested, and one was a flaw in the Stage 2
  prompt itself: the correlation test compared a height-above-ground model
  against *terrain* elevation. A perfect model would score about zero on
  that test.

**Findings from checking where ×255 came from:**
- ×255 was chosen on the **Track1** checkpoint (`104best_model.pth`), using
  3 tiles that were in Track1's own training list.
- The scripts that did it lived only in a `/private/tmp` scratchpad. macOS
  clears that directory, so they were at risk of disappearing.
- FT-2's training script had re-derived the scale on the clean **Swiss**
  checkpoint using training quadrants only. It chose ×255 or ×300, on a
  broad plateau from ×200 to ×1000.
- An undocumented run of Swiss zero-shot on DFC2019 turned up: Pearson about
  0.49. FT-2 beat it in all four folds. So fine-tuning *helps*, and Track1's
  0.72 most likely reflects its training data.
- The Swiss depth convention can't be confirmed from code: the loader never
  rescales depth, and the depth files aren't in the repo. The loader applies
  a per-image min-max stretch to GF-7 RGB, while our pipeline divides 8-bit
  renders by 255. That's a second possible mismatch.

**Analysis of the uploaded files (the turning point):**
1. **Frequency fusion ≈ SRTM, and had never been compared with SRTM alone.**
   The fused output differed from SRTM by only 0.26–1.0 m on non-hilly
   tiles, while its ICESat-2 error was 1–8 m. The "21/25" headline compared
   a DEM against scaled DAv2; it never measured what DAv2 added.
2. **The scoring rewarded smoothness.** All ICESat-2 references were ATL08
   *ground* photons, which measure bare earth. Any real above-ground detail
   (trees, buildings) was scored as error. No surface reference existed.
3. **Nothing had been tested at true 10 m detail.** DAv2 and DINOv3 had both
   been run at 518 px on ~1000 px tiles, an effective resolution of about
   19 m. The upsampling also left a strong period-2 artifact in the depth
   input fed to RDAH.
4. **The same RDAH preprocessing gives realistic heights on DFC2019**
   (variance ratio 0.72–1.02) but flat output on Sentinel-2. That points to
   the resolution gap — sub-metre training imagery vs 10 m input — rather
   than a preprocessing bug.
5. **DAv2's mid-scale detail is inverted in hilly terrain:** +0.64 raw
   correlation becomes −0.43 after removing a planar trend. That's exactly
   the band frequency fusion was adding back.

**Output:** a phased plan with pre-registered decision rules. Rules are
committed before results are seen, and the plan includes a ceiling check.

---

## Stage 4 — Phases 0–5: fix the scoring, give RDAH a fair final test, bake-off (no training)

**What and why, per phase:**
- **P0 Rescue.** Get the scratchpad scripts and outputs into git before macOS
  deletes them. Log the Swiss zero-shot result and the true origin of ×255.
- **P1 DEM-only controls.** Score fusion with the DAv2 detail removed, plus
  raw SRTM and raw GLO-30. Also a direct detail test: correlate the model's
  high-pass component with ICESat-2 height minus the DEM low-pass. This test
  doesn't depend on a fitted scale factor.
- **P2 RDAH final test.** Check whether training used ImageNet
  normalization. Run a resolution sweep on local DFC2019 tiles (1×/2×/4×/8×
  downsampling). Optionally fetch one Swiss depth file.
- **P3 Surface references.** ICESat-2 ATL08 20 m segments (including canopy)
  and GEDI L2A rh98 via Earth Engine.
- **P4 Bake-off.** DAv2 at 518 and 1008 px, DINOv3-CHMv2, the ETH 2020
  canopy map (a validated Sentinel-2 model, used as a ceiling check), and
  RDAH if rescued. Each scored against ground and surface references.
- **P5 Checkpoint.** Report and stop; no training.

**Result:**
- **P0:** done. The Track1 zero-shot CSV reproduced the reported numbers
  exactly. "Fine-tuning damages the model" was resolved as **no**.
- **P1: DAv2 adds nothing.**
  - Fusion lost to DEM-only (10/25 wins, p≈0.85).
  - DAv2's fine detail correlated at a median of −0.04 (positive on 8/25).
  - **Raw GLO-30 beat fusion on 23/25 tiles** (p≈7e-6).
  - Raw SRTM alone was already on par with fusion.
  - The old 21/25 headline was overturned.
- **P2: RDAH closed on Sentinel-2, but only by judgment.**
  - DFC2019 resolution sweep: Pearson 0.48 / 0.59 / 0.59 / 0.24.
  - The pre-registered "halving" test missed by 0.0006, so closing the line
    was a post-hoc call (labelled as such, fixed in Stage 6).
  - The checkerboard also appears on DFC2019 at native resolution, where
    RDAH works. So the original Sentinel-2 checkerboard rejection had the
    wrong reason.
  - Sentinel-2 renders are 2–4× darker than RDAH's training input.
  - The Figshare deposit is one 14.7 GB archive, so the single-file check
    was skipped.
- **P3:** surface references fetched for 25 tiles plus Darjeeling. Sliderule's
  along-track filter (ats) was relaxed to 5 m, because the default dropped
  about 99% of segments. 99 granule reads failed server-side, so the counts
  are a lower bound.
- **P4: no detail source passed any reference.**
  - The best median detail correlation was 0.089, against a 0.10 bar.
  - CHMv2's "win" came only from a constant offset; it outputs 0.01–0.25 m
    on 10 m imagery.
  - DAv2 at 1008 px was the only consistent near-miss.
  - ETH made the surface product worse (8.32 vs 5.00 m), most likely by
    double-counting canopy that SRTM already contains.
- **P5:** baseline changed to **raw GLO-30**. The GEDI learned route was not
  recommended.
- Commit: `c652c7a`.

**Files** (reconciled against git)
- Commits: `37b9122`, `0638569`, `2cc40d4`, `a15b361`, `3aa534c`, `2973692`, `cf83c6d`, `b921b7f`, `da6bde4`, `7501f1f`, `c652c7a`
- Added (100): 
  - `data/{phase3a_stdout,phase3b_stdout}.txt` (2)
  - `data/dfc2019/experiments/rdah_zeroshot/{rdah_x255_zeroshot_fold_results,swiss_zeroshot_fold_results}.csv` (2)
  - `data/dfc2019/experiments/rdah_zeroshot/rdah_nested_variance_out.log`
  - `data/dfc2019/experiments/rdah_zeroshot/README.md`
  - `data/dfc2019/experiments/rdah_zeroshot/{resolution_sweep_stdout,scale_corr_stdout,swiss_zeroshot_per_fold_stdout,x255_calib_stdout,x255_fullcv_stdout}.txt` (5)
  - `data/dfc2019/experiments/rdah_zeroshot/resolution_sweep/{per_tile,summary}.json` (2)
  - `data/gedi_l2a/{almora,amalapuram,bathinda,bengaluru,chennai,darjeeling,dehradun,delhi,dharamshala,digha,goa_estuary,hyderabad,jaipur,kakinada,kochi_city,kohima,kota,kurnool,kutch,manali,mumbai,nagapattinam,nizamabad,pune,vembanad,vidisha}.csv` (26)
  - `data/sentinel2_benchmark/{srtm_3way_comparison,surface_reference_counts}.csv` (2)
  - `data/sentinel2_benchmark/{phase1_stdout,phase4_glo30_stdout,phase4_srtm_stdout}.txt` (3)
  - `data/sentinel2_benchmark/detail_source_bakeoff/{per_tile_glo30,per_tile_srtm}.csv` (2)
  - `data/sentinel2_benchmark/detail_source_bakeoff/{summary_glo30,summary_srtm}.json` (2)
  - `data/sentinel2_benchmark/frequency_fusion_controls/controls_per_tile.csv`
  - `data/sentinel2_benchmark/frequency_fusion_controls/controls_summary.json`
  - `data/sentinel2_benchmark/frequency_fusion_results/frequency_fusion_results.csv`
  - `data/sentinel2_benchmark/frequency_fusion_results/summary.json`
  - `docs/method-audit/{00-audit-log,final-comparison}.md` (2)
  - `docs/method-audit/01-dem-stat-anchoring/{gaps-and-fixes,summary,verdict}.md` (3)
  - `docs/method-audit/01-dem-stat-anchoring/raw-notes.pdf`
  - `docs/method-audit/02-gcp-regression/{gaps-and-fixes,jax_004_006_check,ransac_v2_starvation_check,summary,verdict}.md` (5)
  - `docs/method-audit/02-gcp-regression/raw-notes.pdf`
  - `docs/method-audit/02-gcp-regression/jax_004_006_check.png`
  - `docs/method-audit/03-semantic-prior/{gaps-and-fixes,summary,verdict}.md` (3)
  - `docs/method-audit/03-semantic-prior/raw-notes.pdf`
  - `docs/method-audit/04-learned-scale-modulation/{gaps-and-fixes,summary,v2-results,verdict}.md` (4)
  - `docs/method-audit/04-learned-scale-modulation/raw-notes.pdf`
  - `docs/method-audit/05-rdah-net-fusion/raw-notes.pdf`
  - `docs/method-audit/06-full-finetune-twin-head/summary.md`
  - `docs/method-audit/stage0-gates/{rs3dada-audit,sparse-lidar-feasibility}.md` (2)
  - `scripts/{detail_source_bakeoff,diagnose_rdah_fold0,evaluate_rdah_pooled_cv,fetch_eth_canopy,fetch_gedi_l2a,fetch_icesat2_segments20m,frequency_fusion_controls,rdah_resolution_sweep,run_dav2_1008,run_rdah_probe,run_rdah_scale_sweep,train_rdah_quadrant_cv,train_rdah_spatial_cv}.py` (13)
  - `scripts/diag/{diag_approach_a,diag_approach_b,diag_rdah_3ckpt,diag_rdah_denorm,diag_rdah_nested_variance,diag_rdah_scale_corr,diag_rdah_x255_calib,diag_rdah_x255_fullcv,diag_swiss_zeroshot_per_fold}.py` (9)
- Modified: 
  - `CLAUDE.md`
  - `docs/HANDOFF.md`
  - `docs/method-audit/05-rdah-net-fusion/{summary,verdict}.md` (2)
  - `docs/method-audit/sentinel2/sign-flip-detector.md`
- Per commit:
  - `37b9122`: A `data/dfc2019/experiments/rdah_zeroshot/{rdah_x255_zeroshot_fold_results,swiss_zeroshot_fold_results}.csv` (2), `data/dfc2019/experiments/rdah_zeroshot/rdah_nested_variance_out.log`, `data/dfc2019/experiments/rdah_zeroshot/README.md`, `data/dfc2019/experiments/rdah_zeroshot/{scale_corr_stdout,swiss_zeroshot_per_fold_stdout,x255_calib_stdout,x255_fullcv_stdout}.txt` (4), `scripts/{diagnose_rdah_fold0,evaluate_rdah_pooled_cv,run_rdah_probe,run_rdah_scale_sweep,train_rdah_quadrant_cv,train_rdah_spatial_cv}.py` (6), `scripts/diag/{diag_approach_a,diag_approach_b,diag_rdah_3ckpt,diag_rdah_denorm,diag_rdah_nested_variance,diag_rdah_scale_corr,diag_rdah_x255_calib,diag_rdah_x255_fullcv,diag_swiss_zeroshot_per_fold}.py` (9)
  - `0638569`: A `docs/method-audit/{00-audit-log,final-comparison}.md` (2), `docs/method-audit/01-dem-stat-anchoring/{gaps-and-fixes,summary,verdict}.md` (3), `docs/method-audit/01-dem-stat-anchoring/raw-notes.pdf`, `docs/method-audit/02-gcp-regression/{gaps-and-fixes,jax_004_006_check,ransac_v2_starvation_check,summary,verdict}.md` (5), `docs/method-audit/02-gcp-regression/raw-notes.pdf`, `docs/method-audit/02-gcp-regression/jax_004_006_check.png`, `docs/method-audit/03-semantic-prior/{gaps-and-fixes,summary,verdict}.md` (3), `docs/method-audit/03-semantic-prior/raw-notes.pdf`, `docs/method-audit/04-learned-scale-modulation/{gaps-and-fixes,summary,v2-results,verdict}.md` (4), `docs/method-audit/04-learned-scale-modulation/raw-notes.pdf`, `docs/method-audit/05-rdah-net-fusion/raw-notes.pdf`, `docs/method-audit/06-full-finetune-twin-head/summary.md`, `docs/method-audit/stage0-gates/{rs3dada-audit,sparse-lidar-feasibility}.md` (2); M `docs/method-audit/05-rdah-net-fusion/{summary,verdict}.md` (2), `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `2cc40d4`: M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `a15b361`: A `data/sentinel2_benchmark/srtm_3way_comparison.csv`, `data/sentinel2_benchmark/phase1_stdout.txt`, `data/sentinel2_benchmark/frequency_fusion_controls/controls_per_tile.csv`, `data/sentinel2_benchmark/frequency_fusion_controls/controls_summary.json`, `data/sentinel2_benchmark/frequency_fusion_results/frequency_fusion_results.csv`, `data/sentinel2_benchmark/frequency_fusion_results/summary.json`, `scripts/frequency_fusion_controls.py`; M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `3aa534c`: M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `2973692`: A `data/dfc2019/experiments/rdah_zeroshot/resolution_sweep_stdout.txt`, `data/dfc2019/experiments/rdah_zeroshot/resolution_sweep/{per_tile,summary}.json` (2), `scripts/rdah_resolution_sweep.py`; M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `cf83c6d`: M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `b921b7f`: A `data/{phase3a_stdout,phase3b_stdout}.txt` (2), `data/gedi_l2a/{almora,amalapuram,bathinda,bengaluru,chennai,darjeeling,dehradun,delhi,dharamshala,digha,goa_estuary,hyderabad,jaipur,kakinada,kochi_city,kohima,kota,kurnool,kutch,manali,mumbai,nagapattinam,nizamabad,pune,vembanad,vidisha}.csv` (26), `data/sentinel2_benchmark/surface_reference_counts.csv`, `scripts/{detail_source_bakeoff,fetch_eth_canopy,fetch_gedi_l2a,fetch_icesat2_segments20m,run_dav2_1008}.py` (5); M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `da6bde4`: A `data/sentinel2_benchmark/{phase4_glo30_stdout,phase4_srtm_stdout}.txt` (2), `data/sentinel2_benchmark/detail_source_bakeoff/{per_tile_glo30,per_tile_srtm}.csv` (2), `data/sentinel2_benchmark/detail_source_bakeoff/{summary_glo30,summary_srtm}.json` (2); M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `7501f1f`: M `docs/method-audit/05-rdah-net-fusion/{summary,verdict}.md` (2), `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `c652c7a`: M `CLAUDE.md`, `docs/HANDOFF.md`, `docs/method-audit/05-rdah-net-fusion/{summary,verdict}.md` (2), `docs/method-audit/sentinel2/sign-flip-detector.md`
- Not committed (as the stage report said; confirmed absent from git):
  `data/sentinel2_benchmark/dav2_depth_1008/*.npy`, `data/sentinel2_benchmark/eth_canopy_2020/*.npy`,
  `data/icesat2_segments20m/*.csv` (these were committed later, in Stage 6: `ce861af`, `2710c1d`).
- ⚑ **`scripts/diag/` holds 9 rescued scripts, not 11** as reported.
- ⚑ **`0638569` also added 26 pre-existing, previously untracked docs** that the stage report
  didn't mention:
  - the Methods 1–4 audit docs and `06-full-finetune-twin-head/summary.md`
  - `stage0-gates/` (2 files)
  - five `raw-notes.pdf` files and `02-gcp-regression/jax_004_006_check.png`
  - the **empty stubs** of `00-audit-log.md` and `final-comparison.md`

  They entered git because the commit staged all of `docs/method-audit`.
- ⚑ `phase3a_stdout.txt` and `phase3b_stdout.txt` live in `data/`, not in
  `data/sentinel2_benchmark/` as the report's `phase*_stdout.txt` implied. `phase1_stdout.txt`
  and `phase4_{glo30,srtm}_stdout.txt` are in `data/sentinel2_benchmark/`.

---

## Stage 5 — Review before close-out

Weaknesses found in the Stage 4 results:
1. **The RDAH rule, read literally, said "rerun," not "close."** Neither close
   condition fired, and a preprocessing mismatch (darker renders) *had* been
   found. That's the branch the rule sent to one Darjeeling rerun.
2. **ETH was tested in an unfair combination.** SRTM already includes
   canopy. The fair test is a bare-earth DEM plus canopy height, i.e.
   FABDEM + ETH.
3. **GLO-30's vertical datum** (EGM2008, vs the EGM96 correction used for
   SRTM) needed checking before GLO-30 became the headline baseline.
4. **The ICESat-2 20 m reference wasn't reproducible**, because
   server-side failures vary between runs. It needed committing, or a
   manifest of exactly which granules were used.

The user asked for a full, careful close-out, so the scope expanded:
- **Re-score on all 32 tiles.** The 7 excluded tiles were removed by a
  *DAv2-based* filter, which has no bearing on DEM-only products.
- **Direct height-above-ground test** (no DEM involved) to separate "10 m
  imagery holds no height signal" from "the combination with a DEM fails."
- **Offset guard.** A pass must also hold after removing the average offset.
- **Method 6 hardening:** variance ratio, confidence intervals, extra seeds.
- **Stale-claim sweep** across docs and the frontend.
- **Write the final record:** `final-comparison.md` and `00-audit-log.md`.

---

## Stage 6 — Close-out session (A0–D4)

**Sentinel-2**
- **A0 Datum audit.** GLO-30 was already on EGM2008, so the headline
  stands. Earlier comparisons applied one geoid value per tile, which varies
  by up to 4.56 m across a hilly tile; comparisons now use a per-point
  geoid. A bug where the geoid correction silently returned 0 (import order)
  was caught before any results were read, and a guard now raises an error.
- **A1 Reproducibility.** ICESat-2 20 m segments committed. SHA-256 hashes
  and regeneration commands recorded for large uncommitted artifacts.
  Relaxing the ats filter changes no conclusion.
- **A2 All 32 tiles.** GLO-30 beats SRTM on ICESat-2 surface heights, on
  both plain and offset-removed RMSE.
- **A3 FABDEM.**
  - FABDEM beats GLO-30 on ground photons on **32/32 tiles** (median RMSE
    1.78 vs 3.05 m), so it's the new **terrain baseline**.
  - No DEM + canopy product passes. The offset guard blocked one pass that
    was only a constant offset (FABDEM + ETH against GEDI).
  - The first FABDEM fetch came back averaged over about 100 km (an Earth
    Engine default projection). This was caught and refetched.
- **A4 Direct height-above-ground test.**
  - CHMv2 (pooled Spearman 0.641) and ETH (0.378) pass the pre-registered
    bar. So 10 m canopy models do carry height signal.
  - Within a single tile it's only 0.23–0.28 (labelled post-hoc), so the
    signal is mostly *between* tiles.
  - Conclusion: the signal exists but doesn't improve a DEM.
- **A5 RDAH Darjeeling rerun,** with training-matched preprocessing
  (per-image min-max stretch). Spearman ≈ 0 against both references, so the
  line is **closed by the pre-registered rule**, replacing the earlier
  judgment call.

**DFC2019**
- **B1** RDAH resolution sweep extended to 50 tiles: the drop is milder than
  8 tiles suggested (8× vs 1× ratio 0.672).
- **B2** Method 6 degrades gently: 0.795 → 0.708 at 2.4 m.
- **C1** Predicted variance is 0.48–0.66 of the true variance, so Method 6's
  predictions are still somewhat compressed toward the mean.
- **C2** Method 6 beats the oracle on 47–49 of 50 tiles per metric.
- **C3** 3 seeds: MAE 1.990 ± 0.010, RMSE 3.504 ± 0.026, Pearson
  0.743 ± 0.002, Spearman 0.656 ± 0.0003. **Every seed beats the oracle on all
  four metrics**, so the headline holds under the pre-registered rule.

**Docs and repo hygiene**
- D1–D4 done. `final-comparison.md` and `00-audit-log.md` written.
  `PROJECT_STATUS_REPORT.md` has a dated addendum. Frontend text fixed where
  it stated a now-false conclusion (confirmed in the served `main.js`).
- Git gaps found and closed:
  - 3,615 DFC2019 result files that had never been committed
  - 103 core scripts (7.3 MB), including Method 4/6 evaluation scripts,
    without which a fresh checkout would break
  - `run_frequency_fusion_sentinel2.py` and the libraries it imports
- One rounding error caught during verification (seed 43's MAE is 1.989).
  All 34 cross-checked values match.
- Final commit: `965a2a4`. Both `CLAUDE.md` and `HANDOFF.md` changed. No
  background jobs left running.

**Files** (reconciled against git)
- Commits: `76f9812`, `6bbcd65`, `ce861af`, `d513ee6`, `6ecb1ba`, `59f0285`, `2710c1d`, `169504e`, `ea01358`, `5d91176`, `d3a397f`, `b168dca`, `0af06b3`, `b4c8842`, `2ce0eb4`, `9761c78`, `0607678`, `05fdcfe`, `ad27726`, `fea5b68`, `d48267c`, `93ae9df`, `303f29e`, `264470f`, `965a2a4`
- Added, excluding `b168dca` (87): 
  - `{COMPETITIVE_REPO_AUDIT,PROJECT_STATUS_REPORT}.md` (2)
  - `data/{REGENERATION_sha256,icesat2_segments20m_failed_granules,icesat2_segments20m_tracks_used}.csv` (3)
  - `data/REGENERATION.md`
  - `data/{a2_eth_stdout,a2_gedi_stdout,a2_is2_stdout,a3_fabdem_stdout}.txt` (4)
  - `data/dfc2019/experiments/{method6_uncertainty,resolution_curves_bootstrap}.json` (2)
  - `data/dfc2019/experiments/method6_seeds_nohup.log`
  - `data/dfc2019/experiments/{method6_resolution_sweep_stdout,method6_seeds_status}.txt` (2)
  - `data/dfc2019/experiments/method6_height_balanced_seed43/m6_heightbal_seed43_results.json`
  - `data/dfc2019/experiments/method6_height_balanced_seed43/train.log`
  - `data/dfc2019/experiments/method6_height_balanced_seed44/m6_heightbal_seed44_results.json`
  - `data/dfc2019/experiments/method6_height_balanced_seed44/train.log`
  - `data/dfc2019/experiments/method6_resolution_sweep/per_quadrant_seed43.json`
  - `data/dfc2019/experiments/rdah_zeroshot/resolution_sweep_50_stdout.txt`
  - `data/dfc2019/experiments/rdah_zeroshot/resolution_sweep_50/{per_tile,summary}.json` (2)
  - `data/gedi_l2a/{bhitarkanika,fatehpur,hisar,karnal,nainital,ooty,shimla}.csv` (7)
  - `data/icesat2_segments20m/{almora,amalapuram,bathinda,bengaluru,bhitarkanika,chennai,darjeeling,dehradun,delhi,dharamshala,digha,fatehpur,goa_estuary,hisar,hyderabad,jaipur,kakinada,karnal,kochi_city,kohima,kota,kurnool,kutch,manali,mumbai,nagapattinam,nainital,nizamabad,ooty,pune,shimla,vembanad,vidisha}.csv` (33)
  - `data/maxar_sanity/method6_inference_summary_recomputed.txt`
  - `data/sentinel2_benchmark/{geoid_undulation_audit,geoid_within_tile_range}.csv` (2)
  - `data/sentinel2_benchmark/{a1_strictis2_stdout,a2a3_stdout,a4_posthoc_withintile,a4_stdout}.txt` (4)
  - `data/sentinel2_benchmark/dem_baselines_32/per_tile.csv`
  - `data/sentinel2_benchmark/dem_baselines_32/summary.json`
  - `data/sentinel2_benchmark/detail_source_bakeoff/per_tile_srtm_strictis2.csv`
  - `data/sentinel2_benchmark/detail_source_bakeoff/summary_srtm_strictis2.json`
  - `data/sentinel2_benchmark/direct_height_test/per_tile.csv`
  - `data/sentinel2_benchmark/direct_height_test/summary.json`
  - `data/sentinel2_benchmark/rdah_zeroshot/darjeeling_a5_stdout.txt`
  - `data/sentinel2_benchmark/rdah_zeroshot/darjeeling_a5/result.json`
  - `docs/screenshots/2026-09-23_d1_after_text_fix.png`
  - `scripts/{dem_baselines_32,direct_height_test,fetch_fabdem,method6_resolution_sweep,method6_uncertainty,rdah_darjeeling_rerun,resolution_curve_bootstrap}.py` (7)
  - `scripts/run_method6_seeds.sh`
- Added by `b168dca` (traceability): 3,615 DFC2019 result files under `data/dfc2019/experiments/`.
  Exact paths are in the appendix at the end of this log.
- Modified: 
  - `CLAUDE.md`
  - `docs/HANDOFF.md`
  - `docs/method-audit/{00-audit-log,final-comparison}.md` (2)
  - `docs/method-audit/05-rdah-net-fusion/{summary,verdict}.md` (2)
  - `docs/method-audit/06-full-finetune-twin-head/verdict.md`
  - `docs/method-audit/sentinel2/{backbone-comparison,sign-flip-detector}.md` (2)
  - `frontend/src/main.js`
  - `scripts/{detail_source_bakeoff,evaluate_method6_gsd_film_height_balanced,fetch_eth_canopy,fetch_gedi_l2a,fetch_icesat2_segments20m,frequency_fusion_controls,rdah_resolution_sweep}.py` (7)
- Per commit:
  - `76f9812`: A `data/sentinel2_benchmark/{geoid_undulation_audit,geoid_within_tile_range}.csv` (2); M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `6bbcd65`: M `docs/method-audit/06-full-finetune-twin-head/verdict.md`, `scripts/evaluate_method6_gsd_film_height_balanced.py`
  - `ce861af`: A `data/{REGENERATION_sha256,icesat2_segments20m_failed_granules,icesat2_segments20m_tracks_used}.csv` (3), `data/REGENERATION.md`, `data/icesat2_segments20m/{almora,amalapuram,bathinda,bengaluru,chennai,darjeeling,dehradun,delhi,dharamshala,digha,goa_estuary,hyderabad,jaipur,kakinada,kochi_city,kohima,kota,kurnool,kutch,manali,mumbai,nagapattinam,nizamabad,pune,vembanad,vidisha}.csv` (26)
  - `d513ee6`: M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `6ecb1ba`: A `data/sentinel2_benchmark/a1_strictis2_stdout.txt`, `data/sentinel2_benchmark/detail_source_bakeoff/per_tile_srtm_strictis2.csv`, `data/sentinel2_benchmark/detail_source_bakeoff/summary_srtm_strictis2.json`; M `docs/method-audit/sentinel2/sign-flip-detector.md`, `scripts/detail_source_bakeoff.py`
  - `59f0285`: M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `2710c1d`: A `data/{a2_eth_stdout,a2_gedi_stdout,a2_is2_stdout,a3_fabdem_stdout}.txt` (4), `data/gedi_l2a/{bhitarkanika,fatehpur,hisar,karnal,nainital,ooty,shimla}.csv` (7), `data/icesat2_segments20m/{bhitarkanika,fatehpur,hisar,karnal,nainital,ooty,shimla}.csv` (7), `data/sentinel2_benchmark/a2a3_stdout.txt`, `data/sentinel2_benchmark/dem_baselines_32/per_tile.csv`, `data/sentinel2_benchmark/dem_baselines_32/summary.json`, `scripts/{dem_baselines_32,fetch_fabdem}.py` (2); M `docs/method-audit/sentinel2/sign-flip-detector.md`, `scripts/{fetch_eth_canopy,fetch_gedi_l2a,fetch_icesat2_segments20m,frequency_fusion_controls}.py` (4)
  - `169504e`: A `data/sentinel2_benchmark/{a4_posthoc_withintile,a4_stdout}.txt` (2), `data/sentinel2_benchmark/direct_height_test/per_tile.csv`, `data/sentinel2_benchmark/direct_height_test/summary.json`, `scripts/direct_height_test.py`; M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `ea01358`: A `data/sentinel2_benchmark/rdah_zeroshot/darjeeling_a5_stdout.txt`, `data/sentinel2_benchmark/rdah_zeroshot/darjeeling_a5/result.json`, `scripts/rdah_darjeeling_rerun.py`; M `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `5d91176`: M `docs/method-audit/05-rdah-net-fusion/summary.md`
  - `d3a397f`: A `COMPETITIVE_REPO_AUDIT.md`, `docs/screenshots/2026-09-23_d1_after_text_fix.png`; M `docs/method-audit/sentinel2/{backbone-comparison,sign-flip-detector}.md` (2), `frontend/src/main.js`
  - `b168dca`: A ×3,615 DFC2019 result files (exact paths: appendix below)
  - `0af06b3`: M `docs/method-audit/00-audit-log.md`
  - `b4c8842`: A `PROJECT_STATUS_REPORT.md`
  - `2ce0eb4`: A `data/dfc2019/experiments/resolution_curves_bootstrap.json`, `data/dfc2019/experiments/rdah_zeroshot/resolution_sweep_50_stdout.txt`, `data/dfc2019/experiments/rdah_zeroshot/resolution_sweep_50/{per_tile,summary}.json` (2), `scripts/{method6_resolution_sweep,resolution_curve_bootstrap}.py` (2); M `docs/method-audit/05-rdah-net-fusion/summary.md`, `scripts/rdah_resolution_sweep.py`
  - `9761c78`: A `data/maxar_sanity/method6_inference_summary_recomputed.txt`, `scripts/method6_uncertainty.py`; M `docs/method-audit/final-comparison.md`
  - `0607678`: M `docs/method-audit/05-rdah-net-fusion/verdict.md`
  - `05fdcfe`: M `docs/HANDOFF.md`
  - `ad27726`: M `CLAUDE.md`
  - `fea5b68`: A `data/dfc2019/experiments/method6_uncertainty.json`, `data/dfc2019/experiments/method6_height_balanced_seed43/m6_heightbal_seed43_results.json`, `data/dfc2019/experiments/method6_height_balanced_seed43/train.log`; M `docs/method-audit/06-full-finetune-twin-head/verdict.md`
  - `d48267c`: A `data/dfc2019/experiments/method6_resolution_sweep_stdout.txt`, `data/dfc2019/experiments/method6_resolution_sweep/per_quadrant_seed43.json`; M `data/dfc2019/experiments/resolution_curves_bootstrap.json`, `docs/method-audit/final-comparison.md`, `docs/method-audit/05-rdah-net-fusion/summary.md`
  - `93ae9df`: M `data/REGENERATION_sha256.csv`, `data/REGENERATION.md`
  - `303f29e`: M `docs/method-audit/00-audit-log.md`
  - `264470f`: M `docs/method-audit/00-audit-log.md`
  - `965a2a4`: A `data/dfc2019/experiments/method6_seeds_nohup.log`, `data/dfc2019/experiments/method6_seeds_status.txt`, `data/dfc2019/experiments/method6_height_balanced_seed44/m6_heightbal_seed44_results.json`, `data/dfc2019/experiments/method6_height_balanced_seed44/train.log`, `scripts/run_method6_seeds.sh`; M `CLAUDE.md`, `data/dfc2019/experiments/method6_uncertainty.json`, `docs/HANDOFF.md`, `docs/method-audit/{00-audit-log,final-comparison}.md` (2), `docs/method-audit/06-full-finetune-twin-head/verdict.md`, `docs/method-audit/sentinel2/sign-flip-detector.md`
- Not committed (as reported; confirmed absent from git):
  - `data/icesat2_photons/*.csv`
  - `data/sentinel2_benchmark/{dav2_depth_1008,eth_canopy_2020,fabdem}/*.npy`
  - `data/dfc2019/experiments/method6_height_balanced_seed{43,44}/fold*.pt`
  - `data/dfc2019/experiments/dav2_calibration/calibration_samples.csv`
  - the user's pre-existing changes to the four `data/sentinel2/*/*_RGB.tif`, `pyproject.toml` and
    `uv.lock`
- ⚑ **The 103 core scripts, `run_frequency_fusion_sentinel2.py` and the libraries it imports were
  NOT committed in this stage's range.** They landed in `fd9fd35`, *after* `965a2a4`, followed by
  `c0f0174` (audit-log gap 9). So `965a2a4` is not the last commit of the session.
- ⚑ `PROJECT_STATUS_REPORT.md` is listed as *modified*; git shows it **added** in `b4c8842` (its
  first commit).
- ⚑ `final-comparison.md` and `00-audit-log.md` are listed as *created* here. They were first
  **added as empty stubs in `0638569`** (Stage 4), and this stage filled them (modified).

---

## 7. What worked and what didn't

| Line of work | Outcome | Why |
|---|---|---|
| Method 6 (DFC2019) | **Works.** Beats the oracle, 3 seeds, 47–49/50 tiles | Full backbone fine-tune on VHR imagery |
| RDAH-FT-2 | Not adopted | Loses to Method 6; fails the RMSE bar |
| RDAH zero-shot on Sentinel-2 | Closed by rule | No height signal at 10 m; trained on sub-metre imagery |
| Checkerboard as a rejection reason | Retracted | Also appears where RDAH works |
| Frequency fusion | Overturned as baseline | DAv2 detail adds nothing; the DEM did all the work |
| DAv2 @518 / @1008 as detail | Fails (1008 is a near-miss) | No usable 10 m detail against lidar |
| CHMv2, ETH as DEM add-ons | Fail | Double-count canopy; offset-only wins blocked |
| CHMv2, ETH as height-above-ground models | Pass pooled, weak within tiles | Real signal, mostly between tiles |
| FABDEM (terrain) | **New terrain baseline** | Bare earth matches ground photons |
| GLO-30 (surface) | **Surface baseline** | Beats SRTM on 32 tiles |
| GEDI learned route | Not recommended, gated | ETH is essentially that model, and it doesn't help |

---

## 8. Current state

- **DFC2019 / VHR:** Method 6 is the flagship. It is robust across seeds and
  tiles and degrades gently at coarser resolution. Remaining caveat: its
  predictions are compressed toward the mean.
- **Sentinel-2 / 10 m:** FABDEM for terrain, GLO-30 for surface. No
  RGB-derived model improves on a free DEM against independent lidar.
- **Overall finding:** RGB-to-height works at VHR resolution and doesn't beat
  DEMs at 10 m. This supports the two-tier imagery plan: DEM terrain as the
  default, ML height detail only where VHR imagery exists.
- **Demo:** frozen. Real DSM as the terrain source. Text corrected where it
  stated overturned claims.

## 9. Open decisions and items

1. **Demo labels (your call):** the header "Satellite → Metric Terrain
   Reconstruction" and the "METRIC ELEVATION" button imply RGB-derived
   elevation. They conflict with the framing rule in CLAUDE.md and the
   research findings. Suggested: "Satellite → 3D Terrain Visualization" and
   "DEM ELEVATION".
2. **Oracle baseline numbers:** the Stage 6 table shows 3.392 / 4.579 /
   0.582 / 0.509, not the long-standing 2.929 / 4.718 / 0.532 / 0.471.
   `final-comparison.md` must state which aggregation and tile set each
   figure uses.
3. **Darjeeling DEM:** "GLO-30 all-NaN" was likely a fetch bug. Switching the
   demo DEM is optional.
4. **Still open:** TSE-Net (untouched); Track1 on its seen vs unseen tiles
   (low priority); the GEDI route (gated on a source passing the surface
   detail test); Method 6 on Sentinel-2 was only run for fold 0.

## 10. Methodology conventions added this session

- Decision rules are committed before results. Anything judged after seeing
  results is labelled post-hoc.
- Docs are written after every stage. Completion is gated on `git diff
  --stat`, replacing the unobservable 90% rule.
- Fusion variants must be compared against a DEM-only control.
- Detail sources are scored against a surface reference, not only ground
  photons.
- DEM-only comparisons use all benchmark tiles, not DAv2-filtered subsets.
- Offset guard: a pass must also hold after removing the average offset.
- Every cited number must trace to a committed file.

---

## Verification pass (Claude Code)

Run on 2026-09-23 with `git log --since=2026-09-22 --name-status --oneline` up to `965a2a4`:
**54 commits.** Only the "Files" subsections above were changed. The reasoning and results text
is untouched. Stages were mapped by commit time, commit message and the hashes the stage reports
had named.

### Commit → Stage map

| Stage | commits | files added / modified |
|---|---|---|
| before Stage 1 (§0: work done before this account, 2026-09-22) | `216ddb4`, `5abf8c2`, `39e20f5`, `298c0ac`, `e06df63`, `b5b1339`, `405db5f`, `39e53b7`, `aa27dd5`, `5cc5e80`, `0b64c6b` | 27 / 1 |
| 1 | `5bb45f7`, `be8ccac`, `9fb0ae2` | 3 / 3 |
| 2 | `348b865`, `82689ac`, `b51b46f`, `d90859f` | 4 / 4 |
| 3 | none (read-only investigation) | — |
| 4 | `37b9122`, `0638569`, `2cc40d4`, `a15b361`, `3aa534c`, `2973692`, `cf83c6d`, `b921b7f`, `da6bde4`, `7501f1f`, `c652c7a` | 100 / 5 |
| 5 | none (review) | — |
| 6 | `76f9812`, `6bbcd65`, `ce861af`, `d513ee6`, `6ecb1ba`, `59f0285`, `2710c1d`, `169504e`, `ea01358`, `5d91176`, `d3a397f`, `b168dca`, `0af06b3`, `b4c8842`, `2ce0eb4`, `9761c78`, `0607678`, `05fdcfe`, `ad27726`, `fea5b68`, `d48267c`, `93ae9df`, `303f29e`, `264470f`, `965a2a4` | 3702 / 17 |
| after `965a2a4` (outside the requested range) | `fd9fd35`, `c0f0174` | see flag below |

**Before-Stage-1 commits.** §0 describes this work as context only; its files are listed here so
nothing in range goes unmapped:
- Added:
  - `data/dfc2019/experiments/method6_gsd_film/m6_gsdfilm_results.json`
  - `data/dfc2019/experiments/method6_height_balanced/m6_heightbal_results.json`
  - `data/sentinel2_benchmark/{shadow_photogrammetry_plausibility,viewing_angles}.csv` (2)
  - `data/sentinel2_benchmark/frequency_fusion_loo_results/results.csv`
  - `data/sentinel2_benchmark/frequency_fusion_semantic_results/results.csv`
  - `data/sentinel2_benchmark/known_height_scale_results/{results_all25,results_losing4}.csv` (2)
  - `data/sentinel2_benchmark/semantic/{coverage_report,openbuildings_coverage_report}.csv` (2)
  - `data/sentinel2_benchmark/yats0x7_ground_trend_results/{results_all25,results_losing4}.csv` (2)
  - `docs/method-audit/05-rdah-net-fusion/{gaps-and-fixes,summary,verdict}.md` (3)
  - `docs/method-audit/06-full-finetune-twin-head/verdict.md`
  - `docs/method-audit/sentinel2/sign-flip-detector.md`
  - `scripts/{_reuse_urban_openbuildings,evaluate_method6_gsd_film_height_balanced,fetch_viewing_angles_benchmark,prepare_semantic_benchmark_footprints,prepare_semantic_benchmark_openbuildings,run_frequency_fusion_semantic,test_frequency_fusion_evidence_gating_loo,test_known_height_scale_frequency_fusion,test_shadow_photogrammetry_plausibility,test_yats0x7_ground_trend_scale_frequency_fusion}.py` (10)
- Modified:
  - `CLAUDE.md`

### ⚑ Files in git that the log didn't mention

1. All 28 files of the 11 **before-Stage-1** commits. §0 mentions the work, not the files.
2. **`0638569`'s 26 swept-in pre-existing docs** (Stage 4 Files): Methods 1–4 docs,
   `06/summary.md`, `stage0-gates/`, raw-notes PDFs, a PNG, and the `00-audit-log.md` /
   `final-comparison.md` stubs.
3. **Stage 6 files the report named only generically or not at all:**
   - `COMPETITIVE_REPO_AUDIT.md` (added, `d3a397f`)
   - `docs/method-audit/sentinel2/backbone-comparison.md` (modified, `d3a397f`)
   - `docs/screenshots/2026-09-23_d1_after_text_fix.png`
   - `data/REGENERATION_sha256.csv`
   - `data/icesat2_segments20m_{failed_granules,tracks_used}.csv`
   - `data/sentinel2_benchmark/geoid_{undulation_audit,within_tile_range}.csv`
   - `data/maxar_sanity/method6_inference_summary_recomputed.txt`
   - `data/a2_*_stdout.txt` / `data/a3_fabdem_stdout.txt`
   - `data/dfc2019/experiments/method6_seeds_{nohup.log,status.txt}`
   - `data/dfc2019/experiments/resolution_curves_bootstrap.json`
   - modifications to `scripts/{evaluate_method6_gsd_film_height_balanced,frequency_fusion_controls,fetch_icesat2_segments20m,fetch_gedi_l2a,fetch_eth_canopy,detail_source_bakeoff,rdah_resolution_sweep}.py`

   All are now enumerated in the Stage 6 Files subsection.

### ⚑ Files the log mentions that aren't in git (in range) or are misdescribed

1. **"`scripts/diag/` (11 rescued)"**: git has **9**.
2. **"103 core scripts" and "`run_frequency_fusion_sentinel2.py` and the libraries it imports"**
   (Stage 6): **not in the range**. They're committed in `fd9fd35` (after `965a2a4`), with
   `c0f0174` recording the gap. The same applies to "Final commit: `965a2a4`": two commits follow it.
3. **Added, not modified:** `docs/HANDOFF.md` (Stage 1, `9fb0ae2`) and `PROJECT_STATUS_REPORT.md`
   (Stage 6, `b4c8842`) were added (first commits), not modified.
4. **First added in Stage 4:** `final-comparison.md` and `00-audit-log.md` were first added as
   stubs in Stage 4 (`0638569`), not created in Stage 6.
5. **§0 mentions `scripts/train_rdah_quadrant_cv.py`** as pre-account work. It entered git only
   in Stage 4 (`37b9122`).
6. **Path precision:** `phase3a/3b_stdout.txt` are in `data/`, not `data/sentinel2_benchmark/`.

No file was added to git to make the log match. Everything described as "not committed" was
confirmed absent from git (untracked or gitignored).

### Appendix — exact paths added by `b168dca` (3,615)

<details><summary>expand</summary>

```
data/dfc2019/experiments/dav2_baseline/config.json
data/dfc2019/experiments/dav2_baseline/diagnostic_summary.csv
data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_004_006/correlation_diagnostics.csv
data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_004_014/correlation_diagnostics.csv
data/dfc2019/experiments/dav2_baseline/diagnostics/JAX_022_009/correlation_diagnostics.csv
data/dfc2019/experiments/dav2_baseline/manifest.csv
data/dfc2019/experiments/dav2_baseline/metrics.csv
data/dfc2019/experiments/dav2_baseline/reports/baseline_summary.json
data/dfc2019/experiments/dav2_baseline/reports/best_worst_tiles.csv
data/dfc2019/experiments/dav2_baseline/reports/city_summary.csv
data/dfc2019/experiments/dav2_baseline/reports/diagnostic_summary.csv
data/dfc2019/experiments/dav2_baseline/reports/metrics_source.csv
data/dfc2019/experiments/dav2_baseline/reports/tile_ranking.csv
data/dfc2019/experiments/dav2_baseline/reports/workflow_report.md
data/dfc2019/experiments/dav2_calibration/calibration_fit_summary.json
data/dfc2019/experiments/dav2_calibration/calibration_tile_summary.csv
data/dfc2019/experiments/dav2_calibration/config.json
data/dfc2019/experiments/dav2_calibration/evaluation/calibration_evaluation_city_summary.csv
data/dfc2019/experiments/dav2_calibration/evaluation/calibration_evaluation_summary.csv
data/dfc2019/experiments/dav2_calibration/evaluation/isotonic_metrics.csv
data/dfc2019/experiments/dav2_calibration/evaluation/linear_metrics.csv
data/dfc2019/experiments/dav2_calibration/manifest.csv
data/dfc2019/experiments/dav2_calibration/models/isotonic_mapping.joblib
data/dfc2019/experiments/dav2_calibration/models/isotonic_mapping.json
data/dfc2019/experiments/dav2_calibration/models/linear_mapping.json
data/dfc2019/experiments/method4/method4_spatial_cv_results.json
data/dfc2019/experiments/method4_v2_phase0/phase0_corrected_baseline.json
data/dfc2019/experiments/method4_v2_phase1_smooth0.001/phase1_smooth0.001_results.json
data/dfc2019/experiments/method4_v2_phase1_smooth0.01/phase1_smooth0.01_results.json
data/dfc2019/experiments/method4_v2_phase1_smooth0/phase1_smooth0_results.json
data/dfc2019/experiments/method4_v2_phase2.5_building_rank/phase2_building_rank_results.json
data/dfc2019/experiments/method4_v2_phase2.5_combined/phase2_v2_combined_results.json
data/dfc2019/experiments/method4_v2_phase2.5_groundplane_v2/phase2_groundplane_v2_results.json
data/dfc2019/experiments/method4_v2_phase2.5_r2_building_rank_v2/phase2_building_rank_v2_results.json
data/dfc2019/experiments/method4_v2_phase2.5_r2_groundplane_light/phase2_building_rank_groundplane_light_results.json
data/dfc2019/experiments/method4_v2_phase2_building/phase2_building_results.json
data/dfc2019/experiments/method4_v2_phase2_combined/phase2_combined_results.json
data/dfc2019/experiments/method4_v2_phase2_groundplane/phase2_groundplane_results.json
data/dfc2019/experiments/method4_v2_phase2_rank/phase2_rank_results.json
data/dfc2019/experiments/method4_v2_sid_building_rank_v2/phase2_building_rank_v2_sid_results.json
data/dfc2019/experiments/method6_finetune_twinhead/method6_results.json
data/dfc2019/experiments/method6_full_checkpoint/meta.json
data/dfc2019/experiments/method6_smoketest/smoke_results.json
data/dfc2019/experiments/rdah/fold0/history.json
data/dfc2019/experiments/rdah/fold0/result.json
data/dfc2019/experiments/rdah/fold1/history.json
data/dfc2019/experiments/rdah/fold1/result.json
data/dfc2019/experiments/rdah/fold2/history.json
data/dfc2019/experiments/rdah/fold2/result.json
data/dfc2019/experiments/rdah/fold3/history.json
data/dfc2019/experiments/rdah/fold3/result.json
data/dfc2019/experiments/rdah/method_rdah_spatial_cv_results.json
data/dfc2019/experiments/rdah/pooled_cv_epoch5_report.json
data/dfc2019/experiments/rdah_quadrant_cv/fold0/history.json
data/dfc2019/experiments/rdah_quadrant_cv/fold0/result.json
data/dfc2019/experiments/rdah_quadrant_cv/fold1/history.json
data/dfc2019/experiments/rdah_quadrant_cv/fold1/result.json
data/dfc2019/experiments/rdah_quadrant_cv/fold2/history.json
data/dfc2019/experiments/rdah_quadrant_cv/fold2/result.json
data/dfc2019/experiments/rdah_quadrant_cv/fold3/history.json
data/dfc2019/experiments/rdah_quadrant_cv/fold3/result.json
data/dfc2019/experiments/rdah_quadrant_cv/method_rdah_quadrant_cv_results.json
data/dfc2019/experiments/semantic/method3_results.json
data/dfc2019/experiments/semantic/method3_spatial_cv_results.json
data/dfc2019/experiments/sparse_anchor/sparse_anchor_city_summary.csv
data/dfc2019/experiments/sparse_anchor/sparse_anchor_metrics.csv
data/dfc2019/experiments/sparse_anchor/sparse_anchor_summary.csv
data/dfc2019/experiments/sparse_anchor_ransac_v2/REPORT.txt
data/dfc2019/experiments/sparse_anchor_ransac_v2/calibration.json
data/dfc2019/experiments/sparse_anchor_ransac_v2/config.json
data/dfc2019/experiments/sparse_anchor_ransac_v2/results.csv
data/dfc2019/experiments/sparse_anchor_ransac_v2/summary.csv
data/dfc2019/experiments/sparse_anchor_regression/REPORT.txt
data/dfc2019/experiments/sparse_anchor_regression/config.json
data/dfc2019/experiments/sparse_anchor_regression/distributions.csv
data/dfc2019/experiments/sparse_anchor_regression/paired_deltas.csv
data/dfc2019/experiments/sparse_anchor_regression/results.csv
data/dfc2019/experiments/sparse_anchor_regression/summary.csv
data/dfc2019/experiments/sparse_anchor_robust/REPORT.txt
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_006_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_014_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_004_016_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_018_012_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_022_009_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_031_006_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_072_015_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_012_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_118_015_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_006_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_149_025_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_161_001_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_164_008_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_165_015_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_166_006_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_175_002_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_204_005_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_015_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_214_023_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_224_025_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_264_013_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_269_009_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_009_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_416_022_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_016_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/JAX_505_018_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_042_011_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_084_038_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_134_027_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_144_030_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_198_002_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_032_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_211_039_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_212_033_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_221_034_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_225_001_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_230_036_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_029_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_248_030_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_258_020_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_269_035_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_002_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_281_030_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_019_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_315_020_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_332_037_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_003_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_364_043_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_023_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_grid_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_grid_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_grid_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_10_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_20_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_01.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_02.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_03.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_04.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_05.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_06.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_07.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_08.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_09.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_10.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_11.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_12.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_13.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_14.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_15.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_16.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_17.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_18.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_random_5_19.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_spatial_height_10_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_spatial_height_20_00.json
data/dfc2019/experiments/sparse_anchor_robust/anchors/OMA_376_038_spatial_height_5_00.json
data/dfc2019/experiments/sparse_anchor_robust/config.json
data/dfc2019/experiments/sparse_anchor_robust/distributions.csv
data/dfc2019/experiments/sparse_anchor_robust/results.csv
data/dfc2019/experiments/sparse_anchor_robust/summary.csv
data/dfc2019/experiments/sparse_anchor_robust_ext/REPORT.txt
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_006_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_006_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_006_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_014_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_014_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_014_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_016_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_016_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_004_016_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_018_012_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_018_012_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_018_012_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_022_009_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_022_009_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_022_009_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_031_006_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_031_006_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_031_006_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_072_015_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_072_015_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_072_015_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_118_012_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_118_012_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_118_012_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_118_015_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_118_015_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_118_015_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_149_006_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_149_006_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_149_006_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_149_025_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_149_025_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_149_025_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_161_001_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_161_001_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_161_001_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_164_008_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_164_008_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_164_008_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_165_015_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_165_015_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_165_015_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_166_006_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_166_006_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_166_006_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_175_002_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_175_002_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_175_002_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_204_005_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_204_005_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_204_005_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_214_015_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_214_015_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_214_015_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_214_023_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_214_023_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_214_023_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_224_025_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_224_025_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_224_025_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_264_013_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_264_013_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_264_013_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_269_009_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_269_009_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_269_009_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_416_009_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_416_009_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_416_009_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_416_022_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_416_022_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_416_022_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_505_016_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_505_016_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_505_016_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_505_018_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_505_018_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/JAX_505_018_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_042_011_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_042_011_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_042_011_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_084_038_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_084_038_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_084_038_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_134_027_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_134_027_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_134_027_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_144_030_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_144_030_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_144_030_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_198_002_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_198_002_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_198_002_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_211_032_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_211_032_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_211_032_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_211_039_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_211_039_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_211_039_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_212_033_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_212_033_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_212_033_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_221_034_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_221_034_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_221_034_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_225_001_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_225_001_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_225_001_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_230_036_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_230_036_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_230_036_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_248_029_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_248_029_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_248_029_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_248_030_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_248_030_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_248_030_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_258_020_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_258_020_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_258_020_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_269_035_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_269_035_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_269_035_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_281_002_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_281_002_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_281_002_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_281_030_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_281_030_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_281_030_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_315_019_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_315_019_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_315_019_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_315_020_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_315_020_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_315_020_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_332_037_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_332_037_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_332_037_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_364_003_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_364_003_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_364_003_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_364_043_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_364_043_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_364_043_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_376_023_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_376_023_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_376_023_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_376_038_grid_100_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_376_038_grid_30_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/anchors/OMA_376_038_grid_50_00.json
data/dfc2019/experiments/sparse_anchor_robust_ext/config.json
data/dfc2019/experiments/sparse_anchor_robust_ext/distributions.csv
data/dfc2019/experiments/sparse_anchor_robust_ext/results.csv
data/dfc2019/experiments/sparse_anchor_robust_ext/summary.csv
data/dfc2019/experiments/sparse_anchor_smooth_residual/REPORT.txt
data/dfc2019/experiments/sparse_anchor_smooth_residual/config.json
data/dfc2019/experiments/sparse_anchor_smooth_residual/fit_metadata.csv
data/dfc2019/experiments/sparse_anchor_smooth_residual/paired_deltas.csv
data/dfc2019/experiments/sparse_anchor_smooth_residual/preflight.csv
data/dfc2019/experiments/sparse_anchor_smooth_residual/results.csv
data/dfc2019/experiments/sparse_anchor_smooth_residual/summary.csv
data/dfc2019/experiments/sparse_anchor_smooth_residual/tile_summary.csv
data/dfc2019/experiments/sparse_anchor_spatial/REPORT.txt
data/dfc2019/experiments/sparse_anchor_spatial/config.json
data/dfc2019/experiments/sparse_anchor_spatial/distributions.csv
data/dfc2019/experiments/sparse_anchor_spatial/fit_metadata.csv
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_004_006.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_004_014.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_004_016.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_018_012.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_022_009.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_031_006.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_072_015.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_118_012.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_118_015.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_149_006.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_149_025.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_161_001.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_164_008.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_165_015.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_166_006.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_175_002.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_204_005.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_214_015.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_214_023.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_224_025.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_264_013.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_269_009.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_416_009.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_416_022.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_505_016.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/JAX_505_018.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_042_011.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_084_038.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_134_027.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_144_030.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_198_002.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_211_032.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_211_039.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_212_033.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_221_034.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_225_001.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_230_036.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_248_029.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_248_030.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_258_020.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_269_035.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_281_002.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_281_030.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_315_019.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_315_020.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_332_037.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_364_003.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_364_043.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_376_023.json
data/dfc2019/experiments/sparse_anchor_spatial/fits/OMA_376_038.json
data/dfc2019/experiments/sparse_anchor_spatial/paired_deltas.csv
data/dfc2019/experiments/sparse_anchor_spatial/preflight.csv
data/dfc2019/experiments/sparse_anchor_spatial/results.csv
data/dfc2019/experiments/sparse_anchor_spatial/summary.csv
data/dfc2019/experiments/sparse_anchor_spatial_huber/results.csv
data/dfc2019/experiments/sparse_anchor_spatial_huber/summary.json
data/dfc2019/experiments/sparse_anchor_spatial_residual/REPORT.txt
data/dfc2019/experiments/sparse_anchor_spatial_residual/config.json
data/dfc2019/experiments/sparse_anchor_spatial_residual/oracle_in_sample.csv
data/dfc2019/experiments/sparse_anchor_spatial_residual/paired_deltas.csv
data/dfc2019/experiments/sparse_anchor_spatial_residual/preflight.csv
data/dfc2019/experiments/sparse_anchor_spatial_residual/results.csv
data/dfc2019/experiments/sparse_anchor_spatial_residual/spatial_diagnostics.csv
data/dfc2019/experiments/sparse_anchor_spatial_residual/summary.csv
data/dfc2019/experiments/sparse_anchor_spatial_residual/tile_summary.csv
```

</details>
