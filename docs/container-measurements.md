# Backend container measurements (branch `slim-container-2026-09-30`, local only; nothing deployed)

- **Harness:** `scripts/container_bench.py` (size / start / flows / trace). Raw results are in `build/slim/results/`
  (gitignored).
- **Machine:** Apple Silicon Mac, Docker via Colima 0.10.3 (VZ, Rosetta for amd64).
- **Sizes** are for the Cloud Run target (linux/amd64): compressed = the layer blobs as a registry stores them (pushed
  to a throwaway **local** registry, 127.0.0.1:5055); unpacked = `du` inside the container.
- **Timing and memory are measured on native linux/arm64 and are INDICATIVE.** Cloud Run runs amd64 on other CPUs, and
  its cold start adds image pull and instance start-up.
- **Memory:**
  - RSS = the server process (PID 1, uvicorn); "peak" = its `VmHWM`.
  - cgroup = the container's `memory.current` / `memory.peak`, which is closer to what Cloud Run bills and limits.
  - Cloud Run's filesystem is in memory, so `/tmp` writes (uploads, generated jobs) also count; `/tmp` is listed per
    flow.
- **Flows** (one fresh container each; they call the real HF Space, Earth Engine and CDSE, with credentials passed at
  run time only):
  - **PNG upload:** no georeference → relative depth on a flat plane.
  - **GeoTIFF upload** (Sentinel-2, 10 m, Bathinda): FABDEM fetch → generation with FABDEM terrain + live GLO-30
    surface.
  - **CDSE scene** (Bathinda AOI, clearest Nov–Dec 2025 scene): FABDEM fetch → the same generation.

## Part 0: BEFORE baseline (2026-09-30)

**What the flows really do** (corrections to the brief, from the code):

- **PNG / JPG upload:** no DEM at all. Relative depth only, drawn on a flat plane (`backend/generation/pipeline.py`,
  the `elev is None` branch).
- **GeoTIFF upload:**
  - FABDEM (Earth Engine) is fetched **automatically** when the GSD is > 2.4 m (`frontend/src/input-view.js:988`); at
    ≤ 2.4 m it is a button (`:487`).
  - Generation then uses the attached DEM for the terrain / elevation ramp and **live Copernicus GLO-30** for the
    surface, which the 3D mesh is built from.
  - With no DEM attached, it uses GLO-30 for both.
- **CDSE scene:**
  - FABDEM **is** fetched automatically (`input-view.js:1255`), then it works exactly like a GeoTIFF.
  - So the brief's "TanDEM-X (GLO-30), not FABDEM" is half right: the mesh shape is GLO-30 (derived from TanDEM-X), and
    FABDEM supplies the terrain / elevation layer.
  - The desktop app has no Earth Engine, so it uses GLO-30 only.
- **Found while measuring (a live bug, not fixed here, so the parity checks stay exact):**
  - CDSE scenes get an **8 × 8** mesh.
  - `_input_elevation` sizes the grid with `max(abs(dem.transform.a), 1.0)`. A scene's DEM is in EPSG:4326, so its
    ~0.0003° pixel becomes 1° and the ~0.09° footprint collapses to the 8-cell minimum.

**The image** (`docker/Dockerfile.before`): python:3.11-slim + `requirements.txt` + `earthengine-api==1.7.43` + `backend/`.

- **As first specified it did not start:** the rasterio 1.4.4 wheel's GDAL needs the system `libexpat.so.1`, which
  python:3.11-slim (Debian trixie) lacks (`ImportError: libexpat.so.1`).
- Render never hit this, because it does not use this image. The baseline adds `libexpat1` and nothing else.
- The backend had no `/health`. A 3-line `GET /health` (no I/O) was added first, so BEFORE and AFTER measure the same
  endpoint.

| BEFORE | amd64 (Cloud Run) | arm64 |
|---|---|---|
| compressed (registry) | **212.3 MB** | 208.5 MB |
| unpacked | **703.5 MB** | 741.1 MB |

**Layers** (amd64, compressed):

| layer | size |
|---|---|
| Debian trixie base | 29.8 MB |
| CA certificates etc. | 1.3 MB |
| CPython 3.11.16 | 14.5 MB |
| `pip install -r requirements.txt earthengine-api` | **166.5 MB** |
| `backend/` | 0.1 MB |
| libexpat1 | < 0.1 MB (in the base update) |

**Top installed packages** (amd64, unpacked):

| package | MB | package | MB | package | MB |
|---|---|---|---|---|---|
| scipy | 142.9 | pip | 12.6 | pydantic | 4.0 |
| rasterio | 111.1 | hf-xet | 12.5 | google-cloud-storage | 3.6 |
| google-api-python-client | 105.8 | huggingface_hub | 6.7 | setuptools | 3.6 |
| numpy | 71.4 | earthengine-api | 5.5 | PyYAML | 3.2 |
| pyproj | 32.9 | pydantic_core | 5.2 | protobuf | 2.4 |
| pillow | 21.1 | | | fsspec | 2.0 |
| cryptography | 15.6 | | | google-auth | 1.9 |

The next largest (all under 2 MB): pyasn1_modules, google-api-core, fastapi, anyio, cffi, pyparsing.

**Start-up and memory** (arm64, indicative):

| BEFORE | value |
|---|---|
| container start → first `/health` 200 (median of 5) | **0.40 s** (0.39–0.42) |
| idle RSS / idle cgroup | **91.4 MiB** / 55.2 MiB |

| BEFORE flow | wall time | peak RSS | cgroup peak | /tmp after |
|---|---|---|---|---|
| PNG upload → relative DSM | 9.7 s | **132.8 MiB** | 90.8 MiB | 4.5 MiB |
| GeoTIFF → FABDEM → metric DSM | 19.2 s | **194.4 MiB** | 137.4 MiB | 5.2 MiB |
| CDSE scene → FABDEM → DSM | 25.9 s | **190.1 MiB** | 131.9 MiB | 2.6 MiB |

## Part 1: dependency audit and trimming (2026-09-30)

**Import audit** (static AST scan of `backend/` + runtime trace of all three flows: 1,245 modules):

- Never imported: `pip`, `hf-xet` (12.5 MB), and small Google / rasterio CLI dependencies (`pyasn1*`, `pycparser`,
  `opentelemetry-api`, `proto-plus`, `google-crc32c`, `wheel`, `click-plugins`, `cligj`; under 4 MB together).
- **scipy** is imported in code (1 use) but was never loaded by the flows.
- No second GDAL carrier (pyogrio / fiona / geopandas / shapely) is installed.
- pyproj *is* a second PROJ copy (rasterio ships its own), which the next step removes.

**Changes:**

1. **pyproj → rasterio.**
   - `store.py`: `CRS.from_user_input(...).is_geographic` now uses `rasterio.crs.CRS`. The unused `Transformer`
     import is gone.
   - `pipeline.py`: `Transformer(...).transform` → `rasterio.warp.transform`.
   - Parity (`scripts/parity_pyproj_rasterio.py`): 39 georeferenced catalog items, max coordinate difference
     **0.0 m**; `is_geographic` identical for all 6 CRSs.
2. **scipy → numpy.** The only use was `distance_transform_edt` in `mesh_export.fill_nan_nearest`.
   - The numpy version searches only the valid cells next to a gap. The nearest valid cell is always such a cell.
   - Ties go to the smallest column, then the smallest row: scipy's own rule, identified on 17,647 of 17,647 tied
     cells.
   - Parity (`scripts/parity_fill_nan_nearest.py`, `build/slim/results/parity_fill_nan_nearest.txt`): **max |diff| 0.0,
     0 cells differ**, on 3 real VHR DSM gap windows (native 1.2 m) and synthetic grids with 1 %, 20 % and 60 % gaps.
   - None of the Sentinel-2 FABDEM or GLO-30 rasters keep a gap at mesh resolution.
   - scipy is removed.
3. **Earth Engine client:** added to `requirements.txt`, pinned (`earthengine-api==1.7.43` + google-api-python-client
   2.200.0, google-auth 2.58.0, google-auth-httplib2 0.4.2, httplib2 0.32.0).
   - The image deletes `googleapiclient/discovery_cache/documents`.
   - The real FABDEM fetch still works (GeoTIFF and CDSE flows below).
4. **hf-xet** is uninstalled in the image; the flows are unaffected.
5. **Lazy imports:**
   - rasterio, PIL (in `store.py` and `pipeline.py`) and pyproj no longer load at start-up.
   - The Earth Engine client, gradio_client and huggingface_hub were already imported inside their functions.
   - At start-up only numpy (plus FastAPI / pydantic / httpx) loads now.
6. **Behaviour parity:** the three flows on the trimmed image vs BEFORE.
   - `satellite.png`, `elevation.png`, `relative_depth.png` and `terrain.json` are **byte-identical** in every flow,
     and so are the depth payload, the FABDEM statistics and the response metadata (apart from job ids and timings).
   - This holds for both the no-`.pyc` and the `.pyc` build (`build/slim/results/parity_before_after_flows.txt`).

`pyproject.toml` / `uv.lock` (the full research environment) keep scipy and pyproj: research scripts use them.
`requirements.txt` is the web backend's list.

## Part 2: static library, measured then STOPPED on size (2026-09-30)

**Bake:** `scripts/bake_static_library.py` (re-runnable).

- It runs the backend's own `pipeline.generate` offline on each included tile's **baked** depth, so it makes no Space
  calls; a tile without baked depth fails the bake.
- Each tile gets `satellite.png`, `relative_depth.png`, `elevation.png`, `terrain.json` (Maxar: DISPLAY for the mesh,
  real bands for the statistics), `preview.jpg`, `thumbnail.jpg` and `tile.json` (the generate response minus the
  base64 depth, plus the manifest fields).
- `index.json` is the curated listing.
- The output folder `frontend/public/library-static/` is gitignored and rejected by the pre-commit hook.
- The trial ran into `build/slim/static_trial/` only.

**Parity** (`build/slim/results/parity_static_vs_live.txt`): Almora, Kutch, a_valley (Maxar, DISPLAY), OMA_212_033 and
JAX_416_009, static files vs the live `POST /api/generate/library/<id>`.

- All four assets are **byte-identical**.
- The metadata and the depth fields are identical.

**Size of the full set:** 76 tiles, 533 files, **251.8 MB**. Largest file: 2.64 MB (`sentinel2-pune/satellite.png`).

| part | size |
|---|---|
| `satellite.png` | 139.3 MB |
| `terrain.json` | 74.4 MB |
| `preview.jpg` | 25.1 MB |
| `elevation.png` | 7.6 MB |
| `relative_depth.png` | 3.5 MB |
| thumbnails | 1.6 MB |

| by source | size |
|---|---|
| DFC2019 | 130.9 MB |
| Sentinel-2 | 101.0 MB |
| Maxar | 19.8 MB |

A tile view transfers about 3.6 MB.

**Vercel Hobby limits** (looked up 2026-09-30):

- https://vercel.com/docs/limits (last updated 2026-09-16):
  - "Static File uploads": **100 MB** (Pro 1 GB), the maximum source-file size of a CLI deployment;
  - "Files": **15,000** source files per CLI deployment.
- https://vercel.com/docs/limits/fair-use-guidelines: Hobby includes **100 GB** Fast Data Transfer and 10 GB Fast
  Origin Transfer a month.
- https://vercel.com/changelog/cli-deployment-limits-removed (June 2026) says "CLI-specific deployment limits" were
  removed without naming any; the limits page still lists 100 MB, so it is treated as binding.
- The frontend deploys by CLI upload of `frontend/`, and `frontend/public/` is **already 102 MB**. 84 MB of that is the
  untracked `public/data/vhr/`, which no code references, and there is no `.vercelignore`.

**Verdict:** the set does **not fit** as baked; the file count (533) is fine. Options, and the DFC2019 question, are
in HANDOFF §6. The frontend wiring and `vercel.json` wait for that decision.

### Part 2, continued: compact format and wiring (2026-09-30, owner decisions)

Owner decisions:
- Option 1 (the compact format).
- DFC2019 served publicly by the web app too (see the dated rule change in docs/HANDOFF.md).
- Web build static; desktop build unchanged.
- Maxar: the one selected preset only. None is chosen yet, so **"medium" is a flagged placeholder**: `index.json` and
  each Maxar item carry `display_preset: {name: "medium", placeholder: true}`.

**Compact format** (`scripts/bake_static_library.py`, default):

- `terrain.u16.gz` is generate's `terrain.json` as gzipped uint16.
  - Worst height error = half a step = range / 131070.
  - Maxar keeps both the real heights (statistics) and the DISPLAY heights (mesh).
- `preview.jpg` doubles as the 3D texture. Generate's `satellite.png` is that JPEG decoded and re-saved losslessly.
- The DISPLAY preset is read in place from `dem_maxar/<preset>/`; library_v2 is not changed.

**Result:** 76 tiles, 457 files, **52.4 MB** (it was 251.8 MB).

**The whole Vercel upload** (`frontend/` minus `node_modules/`, `.vercel/`, `dist/` and `.vercelignore`'s
`public/data/vhr/`):

- **535 files, 72.9 MB: 27.1 MB of headroom** under Hobby's 100 MB.
- The 15,000-file limit is far away.
- `public/data/vhr/` stays on disk; `.vercelignore` only keeps it out of the upload.

**Tolerance parity**, 5 tiles, compact vs exact format, same preset (`build/slim/results/parity_compact_*`):

| tile | height range | max height error (= half step) | texture max / mean pixel diff |
|---|---|---|---|
| Almora | 1011.00 m | 7.71 mm (7.71 mm) | 1 / 0.317 |
| Kutch | 7.02 m | 0.054 mm (0.054 mm) | 1 / 0.275 |
| a_valley, real bands | 327.26 m | 2.50 mm (2.50 mm) | 1 / 0.282 |
| a_valley, DISPLAY | 91.69 m | 0.70 mm (0.70 mm) | (same texture) |
| OMA_212_033 | 21.16 m | 0.16 mm (0.16 mm) | 1 / 0.257 |
| JAX_416_009 | 23.27 m | 0.18 mm (0.18 mm) | 1 / 0.270 |

- Every error is at the half-step bound or under it.
- The headers (size, bounds, min/max, `limitOutliers`) are identical.
- The texture is the browser-decoded `preview.jpg` vs the exact `satellite.png`. No channel of any pixel differs by
  more than 1.
- Rendered mesh: `build/slim/qa_static/shots/before_after_sheet.png`, the same viewer at each tile's default
  exaggeration.
  - The elevation min/max, the auto exaggeration and the applied default are identical before and after.
  - Screenshot pixel differences come mostly from the viewer's idle auto-rotate (a different camera angle after a
    timed run); the texture and height numbers above are the authoritative parity.

**Frontend:**

- **Build-time switch** `VITE_LIBRARY_SOURCE` (`frontend/src/library-source.js`):
  - `npm run build:web` (the `vercel.json` `buildCommand`) = `static`.
  - The desktop's Tauri `beforeBuildCommand` (unchanged) and local dev = `backend`: the unchanged `/api/library` +
    `/api/generate/library` routes on the sidecar's local packs.
- **Desktop bundle:** `vite.config.js` drops `library-static/` from every build that isn't `static`. Checked: the
  desktop build has no `library-static/`, and the web build carries all 76 tiles.
- **Static paths:**
  - listing = `library-static/index.json` (the `/api/library` shape, plus each item's baked `select` plan);
  - generation = the tile's `tile.json` (the generate response shape);
  - terrain decoding = `frontend/src/terrain-data.js`, shared by the viewer and the mini previews.
- **End-to-end** (real Workbench, static mode, CDP network log, `build/slim/results/e2e_static_mode.json`): Almora,
  c_town and JAX_416_009 were opened.
  - **0 backend generate calls** and **0 Space calls**.
  - The inference stat and the depth box show "precomputed".
  - Defaults were applied (x2.2 / x4.0 / x16).
  - The Maxar display note and the DFC2019 credit show in Details.
  - The pipeline log line is unchanged, e.g. "Relative depth: DAv2-Small on CUDA (ZEROGPU), 518×518, 0.208s inference
    (u16-zlib) via sancharimouri/DepthWizard2".
  - The only backend request left is `/api/facts` for georeferenced tiles: the Facts panel's live location query, not
    tile data.
- **`vercel.json`:** `X-Robots-Tag: noindex` on `/library-static/(.*)`.

## Part 3: Dockerfile and image hygiene (2026-09-30)

**`docker/Dockerfile`**, multi-stage:

- **Builder:** python:3.11-slim + binutils; a venv; `pip install` with a BuildKit cache mount, which keeps the pip cache
  out of the image. Then, in the venv:
  - uninstall `hf-xet`;
  - delete `googleapiclient/discovery_cache/documents`;
  - delete package `tests/` folders (`numpy/testing` is kept: it is a runtime module);
  - delete the wheels' `__pycache__`;
  - `strip --strip-unneeded` every `.so`: nothing broke, and the 3 flows are byte-identical to BEFORE on arm64 and
    amd64;
  - uninstall pip / setuptools / wheel;
  - compile the bytecode once.
- **Final stage:** python:3.11-slim + `libexpat1` + the venv + `backend/` (compiled). It runs as a non-root user
  (`app`, uid 10001), with `PYTHONDONTWRITEBYTECODE=1`, a `HEALTHCHECK` on `/health` and `PORT` 8080.
- Base python:3.11-slim still carries its own pip (~12 MB unpacked). Removing it in a later layer would not shrink the
  image, so it stays.
- **Deviation from the brief (owner-approved 2026-09-30): the `.pyc` files are KEPT.** Without them every import
  compiles in memory:

  | arm64 | no `.pyc` | `.pyc` |
  |---|---|---|
  | start-up | 0.83 s | 0.45 s |
  | idle RSS | 65.6 MiB | 60.8 MiB |
  | peak RSS per flow | +10–17 MiB | — |
  | compressed image | 102.5 MB | 114.3 MB |

- **`.dockerignore`** is a whitelist: only `requirements.txt` and `backend/` (minus tests and caches) enter the build
  context. So everything on HANDOFF §4's hide list, `data/`, `docs/`, `frontend/`, notebooks, `scripts/` and tests stay
  out. **The image carries zero tile data, no models, no secrets.**
- There is **no duplicate GDAL/PROJ** left: pyproj's PROJ is gone, and only rasterio's own GDAL + PROJ remain. So
  osgeo/gdal was not considered.

## Part 4: AFTER vs BEFORE (2026-09-30)

| | BEFORE | AFTER | change |
|---|---|---|---|
| **image, amd64, compressed** | 212.3 MB | **116.7 MB** | −45 % |
| image, amd64, unpacked | 703.5 MB | **331.5 MB** | −53 % |
| image, arm64, compressed / unpacked | 208.5 / 741.1 MB | 114.3 / 359.3 MB | |
| Python-deps layer, amd64, compressed | 166.5 MB (pip install) | 70.2 MB (venv copy) | −58 % |
| start → first `/health`, arm64 (median of 5, indicative) | 0.40 s | 0.46 s | ≈ |
| start → first `/health`, amd64 under Rosetta | 0.75 s | 0.77 s | ≈ |
| **idle RSS, arm64 / amd64** | 91.4 / 116.7 MiB | **60.8 / 80.0 MiB** | −33 % / −31 % |
| idle cgroup, arm64 / amd64 | 55.2 / 82.4 MiB | 43.7 / 65.2 MiB | |
| peak RSS, PNG flow, arm64 / amd64 | 132.8 / 170.4 MiB | 122.6 / 157.7 MiB | |
| peak RSS, GeoTIFF + FABDEM, arm64 / amd64 | 194.4 / 249.1 MiB | 183.1 / 233.9 MiB | |
| peak RSS, CDSE + FABDEM, arm64 / amd64 | 190.1 / 243.4 MiB | 180.0 / 228.9 MiB | |
| peak, 3 flows concurrently in one instance, amd64 | — | **287.2 MiB RSS / 240.1 MiB cgroup**, /tmp 12.3 MiB | |
| flow outputs vs BEFORE | — | byte-identical (all 3 flows, arm64 and amd64) | |

- **Top packages AFTER** (amd64, unpacked): rasterio 71.7 MB (was 111.1), numpy 52.9 (71.4), pillow 17.5 (21.1),
  cryptography 12.7, huggingface_hub 6.7, pydantic_core 5.2, pydantic 4.0, google-cloud-storage 3.6,
  earthengine-api 2.5.
- **Gone:** scipy (142.9), pyproj (32.9), the google-api-python-client discovery documents (~100), hf-xet (12.5),
  pip / setuptools / wheel.
- amd64 timings and RSS come from Rosetta translation. That inflates RSS by roughly 20–40 MiB against native arm64.

**Recommendation:**

**(a) Memory: 512 MiB is feasible.**
- The worst measured peak is 287 MiB RSS / 240 MiB cgroup, with three heavy flows at once on amd64.
- 512 MiB leaves about 225 MiB of margin, and Cloud Run gen2's minimum *is* 512 MiB.
- **Conditions:**
  - concurrency ≤ 4 per instance (each extra heavy flow adds about 40–60 MiB);
  - `/tmp` is in memory on Cloud Run and **is never cleaned**: each upload/scene generation leaves about 2.5–5 MB. At
    min-instances 0, instances are recycled often. On a long-lived min-instances = 1 day, a few hundred generations
    would eat the margin, so an age/size cap on `DW2_UPLOADS_DIR` / `DW2_GENERATED_DIR` is an open item.
- Library tiles no longer generate on the backend (Part 2), so only uploads and CDSE scenes write `/tmp`.
- Go to 1 GiB only if concurrency is raised or Method 6 is added.

**(b) Cold start.**
- The container is ready in ~0.5 s once running (indicative).
- On Cloud Run, a cold start adds the image pull (116.7 MB compressed; Cloud Run streams it) and the instance boot.
- Expect **roughly 2–5 s** on a gen2 instance with CPU boost. That is an estimate, not a measurement: it must be
  measured after the first real deploy (docs/deploy-cloud-run.md).
- The first request on a fresh instance also imports rasterio / PIL / Earth Engine lazily (≈ +0.3 s).

**(c) What is left to trim, and whether it is worth it:**
- rasterio's bundled GDAL (71.7 MB unpacked) is the floor for GeoTIFF / DEM work.
- numpy 52.9 MB is needed.
- The google-* / grpc-free Earth Engine client is small now.
- Base-image pip (~3 MB compressed) could only go by flattening the image, which loses base-layer sharing: not worth
  it.
- Dropping cryptography (12.7 MB, pulled in by google-auth) would break Earth Engine auth: not worth it.
- **Remaining gains are small (< 10 %)**; memory is dominated by numpy + GDAL buffers during a flow.

**Method 6 (not wired into the web backend; no weights ship) would raise size and memory if added later.**
Measured 2026-09-30 with the desktop app's DAv2-Small ONNX model (99.1 MB, fp32; Method 6 is the same backbone plus a
small head):
- onnxruntime 1.30.0 adds **68.2 MB** unpacked on amd64 (60.2 arm64), estimated ≈ 20–25 MB compressed.
- Each checkpoint adds **~100 MB**, which barely compresses (fp32 weights). The 4-fold ensemble is ~400 MB.
- Memory: a session costs ~150 MiB, and one 518 × 518 inference peaks at **325–339 MiB** in a bare Python process.
- Added to this backend's ~180–230 MiB flow peak, **one instance would need 1 GiB (≈ 2 GiB with the ensemble or
  concurrent inferences)**.
- Image ≈ +120 MB compressed for a single model.
- Per HANDOFF §1, it should only ever run on genuine ≤ ~2.4 m uploads.
