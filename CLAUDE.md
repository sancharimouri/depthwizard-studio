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

Depth Wizard turns satellite RGB into 3D terrain visualization. The backstory:
we tried to build a real ML pipeline that converts satellite RGB → absolute elevation
using Depth Anything V2 (DAv2). It failed: DAv2's raw relative-depth output on nadir
(top-down) satellite imagery does not reliably correlate with real elevation once you
control for spatial trend (on Darjeeling, correlation flipped from +0.6 to -0.4 after
detrending). We also tried RDAH-Net checkpoints (trained on Swiss/HK building height)
zero-shot on our Sentinel-2 terrain tiles — it produced checkerboard artifacts, rejected.

**Decision made: the ML pipeline is FROZEN. This project is a polished demo/prototype
UI, not a working metric-elevation system.** Real data is used throughout (real Sentinel-2
imagery, real DAv2 relative-depth output, real DSMs as the terrain source), but the
"satellite → absolute elevation" claim is honestly framed as a future research direction,
not a working capability. Never let the UI imply the current pipeline computes real
elevation from RGB — a real DSM is the terrain source; DAv2 output is shown as
relative-depth *visualization*, not as the thing that produced the terrain.

**Frontend and asset-generation work only. Do not modify DAv2 inference logic, add any
correction/calibration/fine-tuning attempt, or otherwise change the ML pipeline itself.**
The one exception, specific to Session 2: you may *run* the existing, already-used
asset-generation process (DAv2 inference + DSM crop/reproject) unmodified, to produce
visualization assets for new regions — that's reusing a frozen process, not changing it.

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
