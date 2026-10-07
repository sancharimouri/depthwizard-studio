# Architecture

Depth Wizard Studio shows real satellite imagery draped over a real elevation model, in a browser or a desktop app.
Terrain always comes from a DEM, never from the image. A depth model's output is shown only as a labelled
relative-depth layer.

## Imagery tiers

- **Tier 1, always available: Sentinel-2 (10 m).** Live search through the Copernicus Data Space Ecosystem, free and
  global. Every user gets this, for any area.
- **Tier 2, opportunistic: very-high-resolution imagery (~0.3 m).** Used where it exists: Maxar Open Data crops (tied
  to specific disaster events), or a high-resolution image the user uploads. Otherwise the app falls back to Tier 1.

The tiers exist because of a research result: a learned height model works at ~0.3 m and adds nothing at 10 m (see
[VALIDATION.md](VALIDATION.md)).

## Components

| Part | What it does | Code |
|---|---|---|
| Web frontend | Vite + three.js viewer: layer switching, orbit and fly (WASD) controls, measurement, Facts and scenario cards, Docs page | `frontend/` |
| Static tile library | The web build bakes every library tile (listing, meshes, textures, depth) into static files served with the site, so opening a library tile needs no backend call | `frontend/src/library-source.js` |
| Backend | FastAPI container: uploads, live Sentinel-2 search, DEM fetch (GLO-30, FABDEM), per-job generation of the viewer's assets, Facts lookups | `backend/`, `docker/Dockerfile` |
| Depth service | A Hugging Face Space running Depth Anything V2 Small on a shared GPU; the backend calls it for relative depth | `space/app.py` |
| Desktop app | Tauri 2 shell around the same frontend, with the backend frozen (PyInstaller) as a local sidecar and DAv2-Small as ONNX, so it runs offline for local tiles | `desktop/`, [DESKTOP_APP.md](DESKTOP_APP.md) |

## One job, end to end

1. **Input:** a library tile, a Sentinel-2 scene found by live search, or an upload (PNG, JPG or GeoTIFF).
2. **Elevation:** the tile's DEM pack, the DEM attached to the upload, or live Copernicus GLO-30. A plain PNG/JPG
   without georeference has no elevation, and the app says so and shows a flat plane.
3. **Relative depth:** Depth Anything V2 on the image, shown as its own layer.
4. **Assets:** `satellite.png`, `relative_depth.png`, `elevation.png`, `terrain.json` and `meta.json` (which sources
   were used, and why), written per job (`backend/generation/pipeline.py`).
5. **Viewer:** three.js builds the mesh from `terrain.json` and drapes the chosen layer.

The four demo regions (Darjeeling, Kolkata, Bardhaman, Sundarbans) are pre-built assets in `frontend/public/data/`.

## Routes that must keep working

The home page carries `id="demo-video"`, and the app is reached at the hash route `#/demo`
(`frontend/src/routes.js`, guarded by `frontend/tests/routes.test.mjs`).
