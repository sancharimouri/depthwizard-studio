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
   RDAH-Net produced checkerboard artifacts (2026-09-23 note: later shown intrinsic to RDAH,
   not the reason it fails at 10 m; it carries no height signal there) — those findings motivated the demo framing
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

## ML research track status (as of 2026-09-23, final close-out — see `docs/method-audit/final-comparison.md`)

Full audit trail: `PROJECT_STATUS_REPORT.md` (repo root) and `docs/method-audit/`
(one numbered subfolder per method, each with `summary.md`/`verdict.md`). Methods tried,
in order:

1. **Global DEM-stat calibration** (`01-dem-stat-anchoring`) — CLOSED, no real improvement.
2. **Sparse-anchor / GCP regression** (`02-gcp-regression`, many variants: Grid/Random/
   Spatial × OLS/Huber/RANSAC) — CLOSED as standalone; RANSAC failed twice; best
   deployable was Grid+Huber+20 anchors (MAE 2.929m / RMSE 4.718m / Pearson 0.532 /
   Spearman 0.471, DFC2019; mean of 50 whole-tile runs). This is a deployable sparse-GCP
   baseline, **not** the oracle. The oracle per-tile-OLS baseline every later method is compared
   against is 3.392 / 4.579 / 0.582 / 0.509 (dense same-tile OLS, scored on held-out quadrants).
   Both are traced in `docs/method-audit/final-comparison.md` §1.0.
3. **Semantic prior** (`03-semantic-prior`, building-probability term) — CLOSED, overfit,
   didn't generalize spatially.
4. **Learned CNN scale-modulation** (`04-learned-scale-modulation`) — **superseded by
   Method 6 (below), not current best.** `phase2_building_rank_v2` (dense coverage +
   building-probability channel + rank loss) on the DFC2019 benchmark: MAE 2.8803m /
   RMSE 4.7751m / Pearson 0.5835 / Spearman 0.5438, vs. the per-tile-OLS baseline above
   — beats it on 3/4 metrics (first config ever to beat baseline Pearson); RMSE gap
   narrowed from >13% to 4.3% but never closed. Full table:
   `docs/method-audit/04-learned-scale-modulation/v2-results.md`.
5. **RDAH-Net fusion** (`05-rdah-net-fusion`) — **closed on Sentinel-2; low-priority open item on DFC2019.**
   - **DFC2019:**
     - FT-2 (2.500/4.294/0.640/0.506) is not adopted.
     - Swiss zero-shot (Pearson 0.492) shows fine-tuning *helps* (4/4 folds).
     - The strong Track1 zero-shot (0.716) came from a checkpoint trained on 41/50 benchmark
       tiles, with ×255 picked on its own training tiles.
   - **Sentinel-2:** closed by a pre-registered rerun (Darjeeling Spearman ≈ 0 vs. ICESat-2/GEDI
     height above ground). The checkerboard was never a valid reason; it's intrinsic.
   - **Resolution:** on 50 DFC2019 tiles, correlation falls 0.581 → 0.330 from 0.6 m to 2.4 m GSD.
     That's one VHR-trained model, not an information limit.
   - Details: `docs/method-audit/final-comparison.md` §1, §3–5; `05-rdah-net-fusion/verdict.md` §6–9.
6. **Method 6 — full DAv2-Small fine-tune, twin head, height-balanced** (`06-full-finetune-twin-head`)
   — **current DFC2019 best: 1.980/3.492/0.745/0.656** (mean of 4 quadrant folds).
   - Beats the per-tile-OLS oracle (3.392/4.579/0.582/0.509) on all four metrics and on 47–49/50
     tiles per metric, with tile-bootstrap CIs.
   - Holds across 3 seeds (1.990 ± 0.010 / 3.504 ± 0.026 / 0.743 ± 0.002 / 0.656 ± 0.0003).
     Variance ratio 0.48–0.66, still underdispersed.
   - **DFC2019-only:** it failed when staged on Sentinel-2.
   - Details: `final-comparison.md` §1–2; `06-full-finetune-twin-head/verdict.md` §6.
7. **Sentinel-2 / India** (`data/sentinel2_benchmark/`, 32 tiles, ICESat-2 + GEDI references) —
   **deployable baselines are plain DEMs.**
   - **Terrain:** FABDEM, median 1.78 m vs. ICESat-2 ground photons; beats GLO-30 on 32/32 tiles.
   - **Surface:** raw Copernicus GLO-30, which beats SRTM under the R4 offset guard.
   - Frequency fusion's old 21/25 headline was the DEM's own result (fusion = its DEM-only
     control, 10/25).
   - No depth or canopy model adds detail (Phase 4); no DEM + canopy product helps (A3); three CNN
     corrections failed.
   - 10 m canopy models (CHMv2, ETH) do carry a height-above-ground signal, mostly between
     landscapes (within-tile Spearman 0.23–0.28).
   - The learned Sentinel-2 + GEDI route is gated (ETH ceiling fired).
   - Details: `final-comparison.md` §3–4; log `docs/method-audit/sentinel2/sign-flip-detector.md`.
8. **DAv2 vs. DINOv3 frozen-backbone comparison** (`sentinel2/backbone-comparison.md`) —
   **superseded.**
   - Its "DINOv3 is the frozen prior going forward" didn't hold: CHMv2 outputs only 0.01–0.25 m
     on 10 m imagery, and it fails as a detail source.
   - Its pooled correlation was with terrain elevation, a construct it doesn't predict.
   - No backbone is carried forward (dated correction in that doc).
9. **Method 6 staged on Sentinel-2** — failed at fold 0 (DEM 1/25, ICESat-2 2/25 tile wins).
   Stopped per protocol. DFC2019 wins don't transfer to 10 m.
10. **Competitive repo audit** (`COMPETITIVE_REPO_AUDIT.md`) — 20 external repos read. It was the
   source of Method 6's idea and of frequency fusion; the latter is now retired (dated note in the
   audit).

`docs/method-audit/final-comparison.md` is the single consolidated record (tables, CIs,
independence flags, audit corrections). `00-audit-log.md` is the chronological index, with
result/commit gaps flagged.

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
