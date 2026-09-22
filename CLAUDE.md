# Depth Wizard 2 — Project Context (read this fully before touching anything)

You are working in `DepthWizard2`, a Smart India Hackathon 2026 (SIH26175) project.
This file is your persistent memory for this project. Read it once at the start of
each session, then **inspect the actual repo yourself** (file tree, package.json,
existing components, existing scripts) — this doc gives you the story and the rules,
not a file map.

## The one rule that overrides everything else

**Quota is scarce.** We're on the last stretch of Claude Code sessions for this project.
Do NOT do extensive upfront planning, do NOT ask clarifying questions unless something
is a genuine blocker, do NOT re-read this whole file into your response. Skim, decide,
build, commit in small increments, move on. If you're unsure between two reasonable
approaches, just pick one and note it in the commit message rather than asking.

## What this project actually is

Depth Wizard turns satellite RGB into 3D terrain visualization. There are **two active
tracks** in this repo — don't collapse them into one story:

1. **Frontend/demo track** — a polished demo/prototype UI, not a working metric-elevation
   system. Real data is used throughout (real Sentinel-2 imagery, real DAv2 relative-depth
   output, real DSMs as the terrain source), but the "satellite → absolute elevation" claim
   is honestly framed as a future research direction, not a working capability in the demo.
   Never let the UI imply the current pipeline computes real elevation from RGB — a real
   DSM is the terrain source; DAv2 output is shown as relative-depth *visualization*, not
   as the thing that produced the terrain. This framing rule is not stale — keep it.
   Early smoke tests here found DAv2's raw relative-depth output on nadir satellite imagery
   doesn't reliably correlate with real elevation once you control for spatial trend (on
   Darjeeling, correlation flipped from +0.6 to -0.4 after detrending), and zero-shot
   RDAH-Net produced checkerboard artifacts — those findings motivated the demo framing
   above, and are seeded into the frontend's pipeline log as real research history.
2. **ML research track** — a separate, active, ongoing investigation (`docs/method-audit/`,
   `data/dfc2019/`, `data/sentinel2_benchmark/`, `external/RDAH-Net/`, `external/SynRS3D/`,
   `scripts/evaluate_*.py`, `scripts/fit_dav2_calibration.py`) into whether a real
   metric-elevation correction on top of DAv2 is achievable. **Not frozen, not out of
   scope.** See "ML research track status" below for where it actually stands. When asked
   to work on this track, follow whatever the task specifies — the "don't touch ML
   pipeline" rule below applies only to the frontend/demo track's own asset-generation
   process, not to this track.

**For the frontend/demo track specifically: frontend and asset-generation work only.**
Do not modify DAv2 inference logic, add any correction/calibration/fine-tuning attempt
to the demo's pipeline, or otherwise change the ML pipeline the demo uses. The one
exception, specific to Session 2: you may *run* the existing, already-used
asset-generation process (DAv2 inference + DSM crop/reproject) unmodified, to produce
visualization assets for new regions — that's reusing a frozen process, not changing it.

## ML research track status (as of 2026-09-23, incl. RDAH-on-Sentinel-2)

Full audit trail: `PROJECT_STATUS_REPORT.md` (repo root) and `docs/method-audit/`
(one numbered subfolder per method, each with `summary.md`/`verdict.md`). Methods tried,
in order:

1. **Global DEM-stat calibration** (`01-dem-stat-anchoring`) — CLOSED, no real improvement.
2. **Sparse-anchor / GCP regression** (`02-gcp-regression`, many variants: Grid/Random/
   Spatial × OLS/Huber/RANSAC) — CLOSED as standalone; RANSAC failed twice; best
   deployable was Grid+Huber+20 anchors (MAE 2.929m / RMSE 4.718m / Pearson 0.532 /
   Spearman 0.471, DFC2019). Still the reference "oracle per-tile-OLS baseline" every
   later method is compared against.
3. **Semantic prior** (`03-semantic-prior`, building-probability term) — CLOSED, overfit,
   didn't generalize spatially.
4. **Learned CNN scale-modulation** (`04-learned-scale-modulation`) — **superseded by
   Method 6 (below), not current best.** `phase2_building_rank_v2` (dense coverage +
   building-probability channel + rank loss) on the DFC2019 benchmark: MAE 2.8803m /
   RMSE 4.7751m / Pearson 0.5835 / Spearman 0.5438, vs. the per-tile-OLS baseline above
   — beats it on 3/4 metrics (first config ever to beat baseline Pearson); RMSE gap
   narrowed from >13% to 4.3% but never closed. Full table:
   `docs/method-audit/04-learned-scale-modulation/v2-results.md`.
5. **RDAH-Net fusion** (`05-rdah-net-fusion`) — zero-shot rejected (checkerboard artifacts
   on Sentinel-2; the RS3DAda comparison was separately found contaminated, 49/50 DFC2019
   benchmark tiles were in its own training split — see `stage0-gates/rs3dada-audit.md`).
   A fine-tuned run (`RDAH-FT-1`, 4-fold spatial CV) produced real numbers — pooled MAE
   2.906m / RMSE 6.659m / Pearson 0.513 / Spearman 0.527, unstable across folds (Pearson
   0.254–0.607). A full audit (`summary.md` §1–9) found an input-scale bug: depth was fed at
   about 1/255 of the intended scale. With the scale corrected, **zero-shot** on DFC2019 gives
   MAE 2.231 / RMSE 4.566 / Pearson 0.716 / Spearman 0.655, using the Track1 checkpoint, whose
   training list contains 41/50 benchmark tiles. The run's own artifact isn't in the repo.
   **RDAH-FT-2** (2026-09-23) used the Swiss checkpoint (0/50 overlap), a per-fold input scale,
   quadrant folds, a rank loss, and nested selection. It scored MAE 2.500 / RMSE 4.294 /
   Pearson 0.640 / Spearman 0.506 (per-sample mean, Method 6's aggregation; pixel-pooled RMSE
   5.598). Fold Pearson range is 0.503–0.607, and the variance ratio of 0.19–0.23 is still
   severe underdispersion. **FT-2 is NOT ADOPTED**: it loses to Method 6 on all four metrics.
   **The method itself stays open.** Zero-shot still beats both fine-tuned runs, and the
   unresolved question is contamination vs. fine-tuning damage. Full detail:
   `05-rdah-net-fusion/{verdict.md §6, summary.md §10}`, aggregate in
   `data/dfc2019/experiments/rdah_quadrant_cv/rdah_ft2_aggregate.json`.
   **RDAH on Sentinel-2 is now CLOSED** (2026-09-23; details in item 7). The original
   checkerboard run was the Swiss checkpoint, reproduced exactly. The ×255 input fix reduces the
   artifact but doesn't remove it, and there's no terrain correlation. What stays open is
   DFC2019-only (contamination vs. fine-tuning damage). The Sentinel-2 test doesn't answer it,
   because the GSD, sensor and construct gaps are confounded with it.
6. **Full DAv2-Small fine-tune, twin (mean, log-variance) head**
   (`06-full-finetune-twin-head`) — **current best result, and the first method in this
   project's entire audit to beat the oracle per-tile-OLS baseline on all four tracked
   metrics simultaneously.** Full backbone fine-tune (not frozen-feature scale modulation
   like Method 4), same 50-tile DFC2019 benchmark and same 4-fold spatial-quadrant
   holdout: MAE 2.053m / RMSE 3.531m / Pearson 0.737 / Spearman 0.654, vs. the oracle
   baseline's 2.929m / 4.718m / 0.532 / 0.471 — every fold individually clears both
   references on every metric, not one lucky fold. Idea sourced by reading (not
   executing) `external/sih2026-depthwizard`; adapted independently onto this project's
   own split. Two real implementation bugs found and fixed en route (an LR-warmup
   omission causing NaN, and a mask-after-compute-instead-of-before bug that surfaced a
   genuine NaN-in-raw-AGL data issue on DFC2019 tile `JAX_004_016`) — both documented in
   `verdict.md` with the reasoning, not just the fix. Caveats: single seed/run, no
   variance-ratio calibration diagnostic computed yet, and — important —
   **this result is DFC2019-only.** It has since been staged on this project's own
   Sentinel-2/SRTM benchmark and failed there (item 9 below) — the DFC2019 win does not
   currently transfer to the Sentinel-2/India domain.
   Two follow-on ablations (ideas read from, not executed from,
   `external/DepthWizard-SIH26175`, adapted independently onto this project's own 4-fold
   split) were tested against this baseline: **GSD-FiLM conditioning** — DFC2019 has no
   real per-tile GSD variation to condition on, so the FiLM blocks reduce to a fixed
   per-channel affine; MAE +0.63% / RMSE +0.82% worse, a clean wash, **not adopted**.
   **Height-balanced loss + sampling** — `CappedHeightWeightedLoss` as an auxiliary term
   (lambda=0.35) plus a `WeightedRandomSampler` over whole training quadrants (adapted
   from their crop-anchoring, since Method 6 has no cropping step) weighted toward
   tall/canopy-heavy quadrants; MAE 1.980m (-3.55%), RMSE 3.492m (-1.09%), improving in
   4/4 folds individually — **adopted as the current recipe.** Full numbers and
   reasoning: `docs/method-audit/06-full-finetune-twin-head/verdict.md`.
7. **Sentinel-2 benchmark** (`data/sentinel2_benchmark/`) — separate validation dataset,
   32 real tiles (8 each: agricultural/coastal/hilly/urban), selected via real ICESat-2
   ATL08 ground-photon coverage (`manifest.csv`, `REPORT.md` — "CLOSED, complete").
   **25 of the 32 tiles are actually used going forward** — the sign-flip detector
   (`docs/method-audit/sentinel2/sign-flip-detector.md`) flagged 7 as unreliable and a
   real geoid/datum bug (orthometric-vs-ellipsoidal height mixing) was found and fixed
   during this work. With that fix, **per-tile linear SRTM/GLO-30 calibration is the
   working, deployable baseline for this domain** — around 11% median error on the
   accepted tiles. Three separate CNN-correction attempts on top of that baseline have
   now all failed, each a different way, all against the same independent ICESat-2 check
   (not just the DEM itself, which can be gamed — see below):
   - Method 4 at SRTM's 10m *reprojected* grid: won the DEM check big (RMSE 75.97m vs.
     linear's 117.87m) but **lost the independent ICESat-2 check** (69.79m vs. 53.40m) —
     textbook interpolation-memorization, not real signal.
   - Method 4 retrained at SRTM's true *native* ~30m grid (removing the interpolation
     opportunity): lost on **both** checks (DEM win count 24/100, ICESat-2 16/100) — plain
     underperformance from data starvation (patch count/fold dropped ~1000+ → ~300).
   - Method 4 retrained with **Google Open Buildings 2.5D** (a real, denser building-height
     source, not SRTM-interpolated) as the target on the 8 urban tiles only: still lost —
     6/8 tiles beat the training-target check but lost the ICESat-2 check, the same
     memorization signature as the first attempt, this time with a genuinely different and
     better target.
   **CNN-based correction is not pursued further without new evidence pointing at a fix
   for the per-tile data-volume ceiling all three attempts hit** — this is a stopping
   point reached deliberately, per a pre-agreed decision rule, not an open thread.
   **Current best, non-learned: frequency fusion** (real DEM low-frequency trend + DAv2
   high-frequency detail, matched-filter subtraction, sourced from the competitive-repo
   audit's `blakc-coffee/depthwizard` and adapted independently onto this project's own
   25 tiles/DEM/ICESat-2 data) **beats plain per-tile linear calibration on 21/25 tiles,
   cutting median ICESat-2 error from 10.88% to 3.57% of elevation range** — the largest
   gains are in hilly terrain (all 5 hilly tiles win, 10-25x). No training involved, so
   Method 4's memorization failure mode structurally cannot apply. This is now the
   recommended deployable baseline for this domain, superseding plain linear calibration.
   Full per-tile table and methodology: `docs/method-audit/sentinel2/
   sign-flip-detector.md` (2026-09-22 entry).
   That 21/25 win rate has since been stress-tested rather than taken at face value:
   **evidence-gating and leave-one-out validation** (ported from `amogh-hub/depthwizard`,
   competitive-repo-audit list entry #19, cloned fresh) were adapted onto frequency
   fusion itself, not the superseded linear-calibration pipeline they predate. The
   evidence gate hooks into frequency fusion's own already-computed per-pixel DEM-coverage
   confidence map; self-tested to confirm it actually fires, though on this 25-tile
   benchmark every tile has 100% DEM coverage so it never rejects anything here — a
   verified safety net for tiles with real DEM voids, not evidence it does nothing. Literal
   per-pixel LOO is infeasible at ~1M pixels/tile, so it was adapted to leave-one-
   **block**-out (LOBO: 25 spatial blocks/tile, refit the affine scale on the other ~96%
   per fold, score every DEM pixel and ICESat-2 photon out-of-fold). Result: 21/25 wins
   under both the original single-split protocol and LOBO, zero flips, max per-tile
   difference 0.0004 percentage points — confirms the win rate isn't a split artifact.
   Details: `docs/method-audit/sentinel2/sign-flip-detector.md` (2026-09-22 entry).
   **Semantic-prior phase 2.3** then tested whether real building data (Microsoft
   GlobalMLBuildingFootprints + Google Open Buildings 2.5D, freshly prepared for all 25
   tiles) could fix the 4 remaining losses (bathinda/amalapuram/kutch/chennai) via two
   closed-form integrations (no CNN) — **clean negative**. Building-aware confidence
   weighting is a genuine no-op (max delta 0.0005pp across all 25 tiles); direct Open
   Buildings height blending is net-harmful (15/25 tiles worse, only mumbai improved,
   unexplained) and makes chennai — the tile hypothesized most likely to benefit — worse,
   not better. Neither flips any losing tile to a win or any winning tile to a loss.
   Neither adopted. **CLOSED**: both follow-up mechanism checks came back clean. Approach A
   is a genuine physical no-op (`building_prob` is correctly wired, `dav2_highpass` is
   sub-centimetre), not an integration bug. Approach B has no units/datum mismatch (Open
   Buildings `building_height` is AGL and is already added to `dem_lowpass`; the full-weight
   "fix" doubles the ICESat-2 overshoot). The negative stands. Full writeup:
   `docs/method-audit/sentinel2/sign-flip-detector.md` (2026-09-22, "Semantic-prior phase
   2.3" entry and its "Re-investigation" subsection).
   **RDAH-Net zero-shot on Sentinel-2 (2026-09-23): clean negative, line closed.** On the
   Darjeeling tile, the DFC2019 input-scale fix (×255 depth, plus reflect-pad-to-1024 sizing)
   cuts the checkerboard's FFT peaks by 2–3 orders of magnitude but doesn't remove them
   (diagonal peaks at periods 8/16/32 persist). Resize vs. pad was ruled out as the cause. The
   output has no terrain correlation raw or detrended: DEM plane-detrended −0.052/−0.047,
   ICESat-2 −0.139/−0.132. The DAv2 control reproduces its +0.64→−0.43 flip. The run stopped at
   Step 2, so there was no benchmark scoring, fusion swap or fine-tuning, and frequency fusion
   with DAv2 remains the deployable baseline. Entry: `sign-flip-detector.md`, "2026-09-23 —
   RDAH-Net zero-shot on Sentinel-2"; script `scripts/rdah_sentinel2_zeroshot.py`.
8. **Frozen-backbone comparison: DAv2 vs. DINOv3** (SAT493M+CHMv2) on the Sentinel-2
   benchmark, against real per-photon ICESat-2 ground heights (not the coverage-only
   counts) — CLOSED, DINOv3 won outright per its stated decision rule (pooled Pearson
   +0.3037 / Spearman +0.3490 vs. DAv2's +0.0403 / -0.0484). Important caveat: DAv2 is
   still the better raw correlate specifically in agricultural and hilly terrain —
   DINOv3's pooled win is driven by urban. Full writeup, per-tile/per-category numbers,
   and the HF-checkpoint state-dict conversion needed to get SAT493M running:
   `docs/method-audit/sentinel2/backbone-comparison.md`.
9. **Method 6 staged on Sentinel-2/SRTM** (same section as item 7,
   `docs/method-audit/sentinel2/sign-flip-detector.md`, 2026-09-22 entries) — the DFC2019
   win (item 6) does **not** replicate here: fold 0 alone lost decisively on both the DEM
   and ICESat-2 checks (1/25 and 2/25 tile wins), plain underperformance, not
   memorization. Stopped at fold 0 per the pre-agreed staged protocol; folds 1-3 and the
   planned Open-Buildings-as-target follow-on for Method 6 were not run.
10. **Competitive repo audit** (`COMPETITIVE_REPO_AUDIT.md`, repo root) — 20 external SIH
    DepthWizard repos investigated (cloned, code read directly, claimed numbers verified
    against actual result files). Headline finding:
    `zaidnansari2011/sih2026-depthwizard`'s full DAv2 fine-tune independently validated
    the same core idea Method 6 above uses, on its own (non-comparable) split — this is
    where Method 6's approach was sourced from. Two other repos
    (`blakc-coffee/depthwizard`, `arpitparashar06/depthwizard`) do genuine real-DEM-low-
    frequency + model-high-frequency fusion, a mechanism this project has not tried and
    flagged as worth testing if the Sentinel-2/India track is revisited.

`00-audit-log.md` and `final-comparison.md` in `docs/method-audit/` are still empty stubs
— don't treat their absence of content as "nothing happened," the per-method docs are
where the real record is.

**None of the above touches the live demo.** Every method here — including Method 6's
DFC2019 win — is research-track work, evaluated offline against DFC2019/Sentinel-2
benchmarks. The frontend's ML pipeline (real DSM as terrain source, DAv2 shown only as a
labeled relative-depth visualization) is unchanged and stays that way — see "The one rule
that overrides everything else" framing at the top of this file. Nothing in this section
is deployed, and nothing here should be read as a change to what Session 1/2's frontend
work shipped.

## Tiered imagery strategy (documented product plan — NOT live, nothing to build now)

This is the eventual product's imagery-sourcing strategy, written down so it's not
re-litigated later. It describes a plan, not a change to the current frontend/demo, which
stays exactly as described elsewhere in this file (real DSM terrain, DAv2 shown as a
labeled relative-depth visualization only). Do not wire any of this into the live UI
unless a future session explicitly asks for it.

- **Tier 1 — default, always available.** Sentinel-2, via live self-serve search through
  the existing CDSE (Copernicus Data Space Ecosystem) integration. Free, global, always-on
  — this is the baseline every user gets, regardless of AOI.
- **Tier 2 — opportunistic, sharper when it exists.** VHR (very-high-resolution) imagery —
  Maxar Open Data disaster-response crops (free, CC-BY-4.0, event-based — see the domain-
  transfer sanity check below for how this project already uses it), future ISRO/Cartosat
  access, or any user-supplied high-resolution source. Not searchable by users the way
  Sentinel-2 is (Maxar Open Data in particular is tied to specific disaster events and
  locations, not a general on-demand catalog) — used opportunistically when real VHR
  coverage exists for a given AOI, falling back to Tier 1 otherwise.

The research reason this tiering exists, not just a product preference: this project's
own DFC2019-trained Method 6 model was sanity-tested on real Maxar VHR crops (Sikkim,
~0.305m GSD — essentially the same resolution as its DFC2019 training data) specifically
*because* VHR access is real but opportunistic, and Sentinel-2 access is universal but
10m — see the ML research track status above for what that test found, and update this
note if a fuller VHR domain-transfer investigation follows.

## What actually exists right now

- Frontend: Three.js + Vite, dev server at `localhost:5173`.
- Real Sentinel-2 RGB GeoTIFFs (10x10km, 10m, 3-band, **EPSG:32645**) for **Darjeeling,
  Kolkata, Bardhaman, Sundarbans** under `data/sentinel2/`.
- Elevation sources:
  - Darjeeling: OpenTopography DSM (elevation range ~557–2478 m) — CartoDEM/Copernicus
    for Darjeeling specifically was rejected earlier as unusable/all-NaN, OpenTopography
    is what's actually wired in.
  - **New: Copernicus GLO-30 DSMs for Kolkata, Bardhaman, and Sundarbans**, now at
    `data/elevation/{region}/`, **30 m, EPSG:4326**, each covering its region's Sentinel
    footprint. Note the CRS mismatch vs. the Sentinel tiles (EPSG:32645 vs EPSG:4326) —
    reprojection/alignment is needed when cropping these to match each satellite tile,
    same as whatever step already handles this for Darjeeling's DEM sources. Also note
    these three regions only have this one (coarser, 30 m) elevation source — no
    OpenTopography-equivalent was acquired for them, so their terrain will read visibly
    less detailed than Darjeeling's. That's expected, not a bug to chase.
- Satellite textures: `frontend/public/data/{region}/satellite.png` now exist for **all
  four regions**.
- Still to generate, for Kolkata/Bardhaman/Sundarbans only (Darjeeling already has these):
  DAv2 `relative_depth.png`, a cropped/reprojected DSM-derived `elevation.png`, and a
  matching `terrain.json`. Find and reuse whatever script/process already generated
  Darjeeling's versions of these three files rather than writing a new one from scratch —
  same process, three more regions, no changes to the process itself.
- Once those assets exist: wire Kolkata/Bardhaman/Sundarbans into the existing scene
  switcher/region rail so all four regions are fully interactive (previously only
  Darjeeling was).

## Session 1 outcomes (done — for your own awareness of current repo state, not to redo)

1. Layout: panels restructured into a left rail instead of a hardcoded offset; fixed the
   WASD-hint-hidden-under-run-button bug.
2. Density: a region rail listing all 4 real regions (only Darjeeling was interactive at
   the time), a stats strip (1.17s DAv2 inference, 10 m resolution, 361×325 grid,
   EPSG:32645), and a pipeline log seeded with this project's real research findings
   (the DAv2/RDAH-Net rejections, the correlation flip that froze the elevation pipeline).
3. Mock upload flow: drag-drop/file-picker modal wired into a shared `runReconstruction()`
   with a real progress bar and live pipeline-step highlighting; resolves to the
   Darjeeling result regardless of the uploaded file, labeled inline as a prototype flow.
4. Flythrough: deleted the previous ~941-line broken spline-based implementation,
   replaced with three.js `OrbitControls` (damped drag-orbit, scroll-zoom, gentle idle
   auto-rotate that pauses on user interaction).
5. Also simplified the bottom-left info card: dropped a "TERRAIN RANGE" readout that
   duplicated "ELEVATION," reused that space for the terrain source label instead.

## Visual direction

Reference: `docs/design-reference/muster-reference.png` for density/confidence texture
(big KPI numbers, sparkline rows, activity feed, real map view) — not a literal template.
`docs/design-reference/current-state.png` is now stale (predates Session 1's layout fix);
trust the live `localhost:5173` over that screenshot.

Keep styling **real** numbers as dense telemetry rather than inventing fake readouts —
this project already has plenty of real data (per-region resolution, grid size, CRS,
elevation range, inference timing) to make all 4 regions feel equally "live," even
though Kolkata/Bardhaman/Sundarbans have coarser elevation data than Darjeeling.

Read `.claude/skills/frontend-design/SKILL.md` (auto-loaded) before new visual work —
same rules as before: avoid generic AI-dashboard tells, one hero kept quiet around it,
motion is either one orchestrated moment or a response to user action.

## Session 2 scope (do these, in this order, small commits between each)

1. **Generate the missing assets** for Kolkata, Bardhaman, and Sundarbans: DAv2
   relative-depth output, DSM-derived elevation texture (cropped/reprojected from the
   new Copernicus GLO-30 data to match each region's Sentinel footprint), and
   `terrain.json`. Reuse Darjeeling's exact asset-generation process — find it in the
   repo first, don't rewrite it. Do this per-region and sanity-check each region's
   output (does the relative-depth image look like a plausible depth map, does the
   terrain mesh look like real topography for that region) before moving to the next.
2. **Wire all three new regions into the scene switcher/region rail**, matching how
   Darjeeling already works: satellite/relative-depth/elevation layer switching, the
   RUN RECONSTRUCTION mock pipeline flow, and the per-region stats strip populated with
   that region's real numbers (resolution, grid size, elevation range, CRS, etc.).
3. **If time remains**, in priority order: crossfade transitions between visualization
   layers (currently hard cuts), a lighting/post-processing pass on the terrain, and a
   recorded screen-capture backup of the working demo in case live rendering hiccups
   during judging.

## Workflow rules for this session

- Work in small, buildable increments. Commit after each of the numbered items above,
  and after each individual region in item 1 if that's a natural checkpoint.
- Take a screenshot of the running dev server after wiring each region, and sanity-check
  it before moving on.
- If you hit a genuine blocker (a script that doesn't generalize cleanly to the new
  regions' CRS, a missing dependency), say so briefly and pick the most reasonable
  default rather than stopping to ask.
- Don't refactor unrelated code. Don't touch ML/model logic — only run the existing
  asset-generation process as-is, per the exception noted above.
- When session usage crosses 90%, stop experimental work and update docs/HANDOFF.md and
  CLAUDE.md before anything else.
