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
