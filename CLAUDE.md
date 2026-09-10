# Depth Wizard 2 — Project Context (read this fully before touching anything)

You are working in `DepthWizard2`, a Smart India Hackathon 2026 (SIH26175) project.
This file is your persistent memory for this project. Read it once at the start of
each session, then **inspect the actual repo yourself** (file tree, package.json,
existing components) — this doc gives you the story and the rules, not a file map.

## The one rule that overrides everything else

**Quota is scarce.** We have ~2 Claude Code sessions of ~5 hours total left this week.
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

**Decision made: the ML pipeline is FROZEN. This project is now a polished demo/prototype
UI, not a working metric-elevation system.** Real data is used throughout (real Sentinel-2
imagery, real DAv2 relative-depth output, real OpenTopography DSM for the actual 3D
terrain), but the "satellite → absolute elevation" claim is honestly framed as a future
research direction, not a working capability. Never let the UI imply the current pipeline
computes real elevation from RGB — it uses a real DSM as the terrain source; DAv2 output
is shown as relative-depth *visualization*, not as the thing that produced the terrain.

**Your job this session is UI/UX and frontend polish only. Do not touch any ML code,
Python inference scripts, or model checkpoints. Do not attempt to "fix" the elevation
pipeline. Frontend only.**

## What actually exists right now (real, don't re-derive fake versions of this)

- Frontend: Three.js + Vite, dev server at `localhost:5173`.
- Real datasets already in the repo: Sentinel-2 RGB GeoTIFFs (10x10km, 10m, 3-band,
  EPSG:32645) for **Darjeeling, Kolkata, Bardhaman, Sundarbans** under `data/sentinel2/`.
- Real terrain source for the working 3D demo: an OpenTopography DSM for Darjeeling,
  elevation range ~557–2478 m, labeled in-UI as "OpenTopography DSM".
- Precomputed demo assets already generated for Darjeeling: `terrain.json`, satellite
  texture PNG, relative-depth texture PNG, elevation texture PNG, under
  `frontend/public/data/darjeeling/`.
- DAv2 (Depth-Anything-V2-Large-hf) runs on MPS, ~1.17s inference, produces a valid
  normalized 0–1 relative-depth map. Only Darjeeling has a full asset set right now;
  the other 3 regions have satellite imagery but not full processed layers yet.
- Working UI today: dark theme, "DEPTH WIZARD" branding, a green "DEMO MODE" indicator,
  a scene card (Darjeeling / West Bengal), a "Reconstruction Pipeline" panel listing
  `01 Satellite Image / 02 Relative Depth / 03 Metric Alignment / 04 3D Reconstruction`,
  a visualization-layer switch (SATELLITE / RELATIVE DEPTH / ELEVATION), an
  elevation/terrain-range info panel, and a "RUN RECONSTRUCTION" button. See
  `docs/design-reference/current-state.png` — this is genuinely what it looks like now.
  It is honest and clean but sparse/empty, and the panel currently overlaps the terrain.
- Cinematic camera flythrough: attempted multiple times with spline-based paths, always
  came out jarring (erratic rotation, terrain leaving frame, clipping). It's currently
  **disabled**. Do not attempt another spline rewrite this session — see Session 1 scope.

## Visual direction

Reference: `docs/design-reference/muster-reference.png` — screenshots of an existing
Claude Code-built dashboard (muster.vyse.site/ops) that the user wants as the *density
and confidence* reference, not a literal template to copy. Note what it's actually doing:
big single KPI numbers with small labels, per-item small bar/sparkline rows, a running
activity/event feed down one side, compact data tables, a real map view. That's the
texture we're going for: a screen that feels like live operational software, not a
landing page.

**Important: do not invent fake data to fill space.** We already have real numbers —
elevation range, Pearson/Spearman correlation values, inference timing (1.17s), tile
counts, resolution, region names, EPSG code. Style *real* numbers as dense telemetry.
Placeholder-but-meaningless visual texture (subtle grid lines, scanline/radar sweep
accents, a live-looking log of actual pipeline steps as they run) is fine for wow-factor,
but don't fabricate specific numeric readouts that look like real sensor data — that
crosses from "polished demo" into "misleading."

**Also read the embedded design skill at `.claude/skills/frontend-design/SKILL.md`
before making visual decisions** — it's auto-loaded, but skim it explicitly once. Core
things it says that matter most here: avoid the generic AI-dashboard tells (tracked-out
ALL-CAPS eyebrows on everything, numbered 01/02/03 badges unless something really is a
sequence — the pipeline stages ARE a real sequence so numbering those is fine, but don't
add numbering elsewhere reflexively), spend boldness in one place and keep the rest
disciplined, and only add motion that's either one deliberate orchestrated moment or
responds to a user action.

## Session 1 scope (do these, in this order, small commits between each)

1. **Fix the layout bug first.** The pipeline/info panels currently overlap the terrain
   too heavily. Push panels to the edges, terrain stays the visual hero, generous negative
   space. This is the single most visible current flaw — fix it before adding anything.
2. **Add density using real data.** A left-side rail listing the 4 real regions
   (Darjeeling / Kolkata / Bardhaman / Sundarbans) as scene-select cards (Darjeeling is
   the only one fully interactive right now — the others can be present but visually
   marked as additional/lower-detail until Session 2 wires them in). A running
   activity/log feed styled like a pipeline event stream, seeded with real steps and
   real numbers from this project (DAv2 inference time, correlation stats, elevation
   range, resolution). A small stats/metrics strip near the pipeline panel.
3. **Build the mock upload → mock pipeline flow.** A drag-and-drop / file-picker UI to
   "upload a satellite image," which then runs an animated 4-stage pipeline sequence
   (reuse the existing 01→04 stages) with a progress indicator, and resolves to the
   precomputed Darjeeling result regardless of what was uploaded. Label this clearly as
   a prototype/demo flow in the UI copy itself (short, honest, in the interface's voice
   — not a disclaimer paragraph).
4. **Replace the broken flythrough with something simple and stable**: a slow constant
   auto-orbit around the terrain, or just leave manual WASD/mouse control as the primary
   interaction with a gentle idle auto-rotate. Do not attempt a scripted cinematic spline
   path again this session — it has failed multiple times and is a time sink we can't
   afford right now. Simple and reliable beats ambitious and broken.

Not in scope for Session 1 (Session 2, later): wiring Kolkata/Bardhaman/Sundarbans fully,
crossfade transitions between layers, lighting/post-processing pass, recording a backup
demo video.

## Workflow rules for this session

- Work in small, buildable increments. Commit after each of the 4 scope items above
  (or sooner if a commit is a natural checkpoint).
- Take a screenshot of the running dev server after each major visual change if your
  environment supports it, and sanity-check it against `current-state.png` and the
  scope item you just did, before moving to the next item.
- If you hit a genuine blocker (missing asset, ambiguous requirement with no reasonable
  default), say so briefly and pick the most reasonable default rather than stopping to ask.
- Don't refactor unrelated code. Don't touch anything under the ML/data-processing side
  of the repo. Frontend and its assets only.
