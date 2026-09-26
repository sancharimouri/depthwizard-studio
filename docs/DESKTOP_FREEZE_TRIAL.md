# Desktop freeze trial: measured baseline (2026-09-26)

**Question.** How big is the frozen Python backend (PyTorch + rasterio + DAv2-Small only, CPU, no tile library)
before committing to a full desktop bundle?

**Build:**
- Tools: PyInstaller 6.22.3, one-folder mode, macOS arm64 (Apple M4), Python 3.11.
- An isolated venv pinned to the project's working versions: torch 2.14.0, transformers 5.17.0, rasterio 1.4.4, pyproj 3.7.2,
  fastapi 0.141.1, uvicorn 0.52.4, gradio_client 2.7.1, earthengine-api 1.7.43, …
- The macOS torch wheel has no CUDA (CPU + MPS).
- **What's in it:**
  - The real backend (`backend.main`).
  - `bridge/dav2_server.py`, unchanged, mounted at `/local-dav2`; `DAV2_INFERENCE_URL` points there.
  - The DAv2-Small checkpoint, bundled as an offline HF cache (95 MB).

## Result

| | Size |
|---|---|
| **Installed (one-folder)** | **904 MB** |
| Download, `.tar.gz` | 348 MB |
| Download, DMG (UDZO) | 424 MB |

**Breakdown (MB):**

| Component | Size |
|---|---|
| torch | 460 (`torch/lib` alone 400) |
| googleapiclient | **102** |
| DAv2-Small checkpoint | 95 |
| rasterio + GDAL | 49 |
| transformers | 44 |
| pyproj | 17 |
| libpython | 17 |
| PIL | 12 |
| cryptography | 12 |
| tokenizers | 10 |
| hf_xet | 8 |
| numpy | 7 |

**Works, not just builds.** Launched from an unrelated folder with an empty environment (`env -i`, no `.env`), then `--selftest`:
- One real image went through the backend's own `/api/depth/relative` → HTTP 200, 518×518.
- Inference 0.19–0.35 s on CPU.
- **Cold start (launch + model load + one inference): 3.1 s.**
- Output vs. the local float32 CPU reference: max |diff| 2.64e-5 (the u16-zlib quantisation bound), Pearson 1.00000000.

**Build pitfalls found (both fixed in the trial entry):**
1. **No `multiprocessing.freeze_support()` → infinite respawn.** torch starts multiprocessing's resource tracker, which re-executes the frozen binary with `-c …`.
   The call must come first in `__main__`, before the heavy imports.
2. **`rasterio.serde` missing at runtime.** It is imported from compiled code. Fix: `--collect-submodules rasterio` (and pyproj).
   Also needed: hidden imports `gradio_client`, `huggingface_hub`, `ee`, and the DAv2/DPT transformers modules; `--copy-metadata` for transformers' version checks.

## Follow-ups done the same night (macOS only)

**1. googleapiclient trim: measured.**
- `ee` builds its API client with `discovery.build(static_discovery=False)` (`ee/_cloud_api_utils.py`). It fetches Earth Engine's discovery document live and uses **none** of googleapiclient's 600 bundled discovery JSONs.
- The FABDEM fetch is a live Earth Engine call that needs each user's own EE credentials (`earthengine authenticate`) and a Cloud project. It cannot work offline in a desktop binary regardless.
- **Decision:** exclude `ee` and `googleapiclient` from the frozen binary; FABDEM becomes a thin call to the hosted backend. Not implemented yet, since there is no hosted backend.
- **Size:** 904 → **801 MB** (−103 MB). In the frozen app, `POST /api/input/{id}/fabdem` now returns a clean `503 "earthengine-api is not installed"`.

**2. A third frozen-build bug, found by testing a GeoTIFF (the JPG self-test could not reveal it).**
- Every GeoTIFF upload failed: `Cannot find proj.db` / `GDAL_DATA is not defined`.
- **Fix:** `--collect-data rasterio --collect-data pyproj`; rasterio then finds its bundled data by itself.
- **Size with it:** **813 MB**. The self-test now also uploads a GeoTIFF: `GEOTEST HTTP 200 crs=EPSG:32645 gsd_m=10.0 tier=1`.
- Re-check through the CI harness on macOS (`ci_check.py`, full build): SELFTEST + GEOTEST 200, 48 s wall.
  2 processes (the binary + its multiprocessing resource tracker), no respawn.
- The `--skip-freeze-support` check was stopped before it finished; that result is still open.

**Current macOS baseline: 813 MB installed** (from 904) with depth + GeoTIFF + clean FABDEM degradation.

## What this means for the size picture

- The **runtime alone is about 0.8 GB installed** (about 0.35–0.42 GB compressed download at the 904 MB build; not re-measured at 813 MB).
  That is inside the 500 MB–1 GB estimate.
- Adding the ~875 MB of tiles and models gives about **1.7 GB installed**.
- Torch `_inductor` / `distributed` / `testing` add about 25 MB; not worth the risk yet.
- **Only macOS arm64 is measured.** PyInstaller does not cross-compile.

## Windows / Linux: NOT RUN YET (next session)

- Everything for it is committed in `desktop/freeze_trial/`:
  - `build_freeze.py`: a cross-platform build with a `--variant no-rasterio-submodules` to test whether that fix is still needed;
  - `ci_check.py`: the self-test with a process-count respawn guard and a `--skip-freeze-support` toggle;
  - `ci_prepare.py`, `ci_size.py`, `requirements-freeze.txt`;
  - `freeze-trial.yml`: a GitHub Actions matrix of ubuntu-latest and windows-latest, with CPU torch from the PyTorch `+cpu` index.
- A **private** repo `sancharimouri/depthwizard2-desktop-ci` was created for it. It is still **empty**: the push failed because its remote was SSH and this machine has no GitHub SSH key.
  Push over HTTPS with `git -c credential.helper='!gh auth git-credential' push https://github.com/sancharimouri/depthwizard2-desktop-ci.git main`.
  Its contents are only `backend/` (committed source, no tests), `bridge/dav2_server.py`, and the files above. The model and test inputs are downloaded in CI from public sources.

## Windows / Linux: MEASURED (2026-09-26, GitHub Actions run 36207174181, private repo `depthwizard2-desktop-ci`)

**Build under test:** the ONNX backend, built by `desktop/freeze_trial/build_freeze.py` in a torch-free venv. The model is exported in a throwaway torch venv.
Each check runs the frozen binary's self-test: depth through `/api/depth/relative` + a GeoTIFF upload, with a process-count respawn guard.

| | Linux (ubuntu-latest) | Windows (windows-latest) | macOS arm64 (local, for reference) |
|---|---|---|---|
| **Installed (one-folder)** | **411.6 MB** | **317.0 MB** | 297.3 MB |
| Download: zip / tar.gz | 246.6 / 205.3 MB | 176.7 MB (zip) | — |
| Full build: depth + GeoTIFF | ✅ 518×518, EPSG:32644 10 m | ✅ | ✅ |
| CPU inference on the runner | 1.4–1.7 s | 1.9–2.0 s | 0.2 s (M4) |
| Without `freeze_support()` | works, 1 process | works, 1 process | (see below) |
| Without `--collect-submodules rasterio` | **fails**: `No module named 'rasterio.serde'` | **fails**, same | fails, same |

**Conclusions:**
- **The rasterio fix is required on every platform.**
- **`freeze_support()` is no longer needed.** The macOS respawn loop was caused by torch starting multiprocessing's resource tracker, and the ONNX backend contains no torch. It stays in the entry script (harmless).
- The GDAL/PROJ data fix (`--collect-data rasterio/pyproj`) is part of every build here, and the GeoTIFF check passes with it on all three.
- **Largest parts:**
  - Linux: model 99.1, `rasterio.libs` 61.1, `rasterio` 49.1, onnxruntime 30.5 MB.
  - Windows: model 99.1, `rasterio.libs` 54.3, onnxruntime 37.5 MB.
