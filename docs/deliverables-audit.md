# Deliverables audit against the SIH26175 brief (read-only, 2026-09-23)

**Scope.** The current app (`frontend/`, `backend/`) is checked against each deliverable in the brief
(`docs/SIH26175_problem_statement.md`). **Read-only: nothing in the frontend or backend was changed.**
Evidence is file:line at HEAD plus one screenshot of the running dev server:
`docs/screenshots/2026-09-23_deliverables_audit_home.png`. It was taken with headless Chrome (SwiftShader
WebGL) at 1600×1000, because the Chrome extension was not connected.

Research-track capabilities that exist only as offline scripts are listed separately. **They are not
counted as app features.**

| # | Deliverable (brief wording) | Verdict | Evidence |
|---|---|---|---|
| 1 | Upload of PNG / JPG / TIFF | **partial** | See 1 below. |
| 2 | PNG/JPG → relative DSM (rDSM) | **no** (mock) | See 2 below. |
| 3 | GeoTIFF → absolute metric DSM (DEM + predicted height) | **no** in the app. Research-only offline | See 3 below. |
| 4 | Output DSM in a standard geospatial format (GeoTIFF export) | **no** in the app. Research-only offline | See 4 below. |
| 5 | Texture projection + first-person flythrough | **partial** | See 5 below. |
| 6 | Slope assessment and structural height analysis | **no** | See 6 below. |
| 7 | In-UI validation of estimated heights against reference data | **no** | See 7 below. |
| 8 | Standalone deployment (packaging) | **no** | See 8 below. |
| 9 | Stability across urban / sparse / hilly / forested | **partial**; the app shows DEMs, not estimates | See 9 below. |

**1. Upload of PNG / JPG / TIFF: partial.**
- `frontend/index.html:1044` has `accept="image/*"`. The browser file dialog accepts PNG and JPG, and TIFF only
  where the OS maps it to an image type.
- The file is only previewed (`main.js:851–857`, "prototype pick, not a live reconstruction"). Nothing is uploaded
  to a backend: `backend/api/routes.py` has only `/search` and `/preview`, both for CDSE.

**2. PNG/JPG → relative DSM: no (mock).**
- The upload "resolves to Darjeeling reference data regardless of the file" (`main.js:1142`).
- The previews fetch the fixed `/data/darjeeling/*` files (`main.js:1182–1188`).
- The relative-depth layer is pre-computed DAv2 for the 4 fixed regions.

**3. GeoTIFF → absolute metric DSM: no in the app; research-only offline.**
- There is no GeoTIFF path in the UI.
- Offline, `scripts/vhr_dsm_pipeline.py` composes DSM = FABDEM + max(Method 6 AGL, 0) for Maxar crops.
- That path is **not wired** into the app.

**4. Output DSM in a standard geospatial format: no in the app; research-only offline.**
- The frontend has no export or download code: the only `blob` uses are image previews, at `main.js:825` and `:879`.
- Offline, `vhr_dsm_pipeline.py:202` writes GTiff (`data/vhr_dsm/*/dsm.tif`).

**5. Texture projection + first-person flythrough: partial.**
- Texture projection works: Sentinel-2 RGB, DAv2 and DEM are draped on the mesh (`main.js:30–45`, screenshot).
- Navigation is **orbit only**. `camera-controls` is set to drag-rotate, wheel and pinch (`controls.js:89–102`).
- "FLYTHROUGH" is a scripted 9.5 s dolly to half the distance with faster auto-rotate (`main.js:353–400`).
- No first-person or WASD mode: the only keydown handlers are Escape, at `main.js:517` and `:986`.

**6. Slope assessment and structural height analysis: no → partial (updated 2026-09-25).**
- **Update:** a 3D measurement tool now exists in Workbench's expanded final-demo view
  (`frontend/src/measure-tool.js`, with `heightfield.js`, `measure-metrics.js` and `measure-model.js`).
  - Two-point (A→B) and continuous (polyline / closed polygon) measurement.
  - It reports horizontal and along-surface distance, rise, gradient (% and °), ascent/descent, and polygon area and perimeter.
  - Everything comes from the raw DEM grid, not the exaggerated display mesh.
- **Still missing:** a slope *map*, elevation profiles as charts, and structure (building/tree) height. The DEMs shown
  are GLO-30/OpenTopography surface models, so a measured "rise" is a DEM elevation difference, not an object height.
  Saving is session-only (no persistence).
- The original finding follows.
- No slope, profile, measurement or picking UI. `grep` finds no raycast, measure or profile code.
- `slope` in `terrain.js:125–204` is a mesh **spike-smoothing cap**, not an analysis tool.

**7. In-UI validation of estimated heights against reference data: no.**
- Validation appears only as **text** in the pipeline log: the "Correlation +0.60 → −0.41" string at
  `index.html:665` and `main.js:1819–1821`.
- No reference overlay (ICESat-2 / GEDI / LiDAR) and no metric readout.

**8. Standalone deployment: no.**
- No Electron, Tauri, PyInstaller, Docker or PWA manifest in the repo.
- `frontend/package.json`'s `build` script runs `vite` (the dev server), not `vite build`.
- The app needs the Vite dev server plus a FastAPI backend for CDSE search.

**9. Stability across urban / sparse / hilly / forested: partial.**
- The app has 4 regions: Darjeeling = hilly and forested, Kolkata = urban, Bardhaman = sparse and agricultural,
  Sundarbans = coastal and forest.
- Their terrain is **DEM-only** (GLO-30), so there is no estimate whose stability could be judged.
- Research evidence per landscape: `docs/method-audit/07-gamus-generalization/summary.md`, where GAMUS covers
  urban and NEON covers forest and mountains.

## "METRIC ELEVATION" / "DEM ELEVATION" label check

**Question:** does the elevation layer show a DEM-only surface, or DEM + predicted height?

**Answer: DEM only.**
- The layer is labelled **"DEM ELEVATION"**, not "METRIC ELEVATION" (`index.html:280`, `:644`). Its caption is
  "DEM elevation from <source>" (`main.js:37`, `:41`), and it reads that way in the screenshot.
  - The rename was done on 2026-09-23 (HANDOFF §5).
  - "Metric Elevation" survives only in code **comments** and in the element ID `metric-elevation-3d-box`
    (`main.js:1041`, `:1173`, `:1705`, `:1818`; `terrain.js:12`). None of these are user-visible.
- The mesh heights are the region's DEM: Darjeeling's `terrain.json` spans 556.5–2477.5 m on a 361×325 grid,
  i.e. the GLO-30 range from HANDOFF §2b. No predicted above-ground height is added anywhere.
- **So the current label is accurate.** "METRIC ELEVATION" would be accurate only for DEM + predicted height,
  and that surface does not exist in the app.
- **Minor, flagged only:**
  - The info card's **"TERRAIN SOURCE: OpenTopography DSM"** (screenshot) is correct but could say that it is
    Copernicus GLO-30 (byte-identical, HANDOFF §2b).
  - The **"DSM"** button shows DAv2 *relative depth* draped on DEM relief (`main.js:38–39`). That is the
    ambiguity HANDOFF §5 already flagged.

## Summary for the user (decision is yours; nothing was changed)

- **Fully met:** none.
- **Partial:** upload (1), texture drape plus orbit flythrough (5), landscape coverage (9).
- **Missing from the app:** the rDSM path (2), the metric-DSM path (3), GeoTIFF export (4), slope and height
  analysis (6), in-UI validation (7), standalone packaging (8).
- Items 3 and 4 exist offline in the research track (`scripts/vhr_dsm_pipeline.py`) and would be the natural
  thing to wire in, **if** a model is adopted for the product.
  - Per `07-gamus-generalization`, what is actually defensible on 10 m Sentinel-2 input remains DEM-only.
  - Method 6 is a VHR (≈0.3 m) model.
