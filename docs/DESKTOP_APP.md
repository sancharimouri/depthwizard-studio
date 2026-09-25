# Desktop app (Tauri 2 shell + frozen Python sidecar) — status 2026-09-26

**What it is:**
- The existing Vite/Three.js frontend (`frontend/`, unchanged) in a Tauri 2.11 window (`desktop/tauri/`).
- The frontend is built with `VITE_API_BASE=http://127.0.0.1:8765`.
- The frozen backend (`desktop/freeze_trial/`, PyInstaller one-folder) ships as a bundle resource. The shell starts it as a sidecar process on 127.0.0.1:8765 and kills it on exit.

**How the sidecar is wired:**
- **Spawned directly from the resource dir, not through Tauri's `externalBin`.** `externalBin` only takes single-file binaries, and a one-file PyInstaller build would unpack about 800 MB on every launch.
- **Startup checks:**
  - It refuses to start if port 8765 is already taken, so it never talks to a stale or foreign server. (This happened during testing: a leftover backend answered for the new one and hid a bug.)
  - It fails if the sidecar exits during startup.
  - Then it waits for the port before opening the window.
- **CORS:** the backend gets `CORS_ORIGINS=tauri://localhost,http(s)://tauri.localhost`. The webview origin on macOS is `tauri://localhost`.

**Build** (macOS arm64):
```
cd desktop/tauri && npm install
cp -R <frozen dw2-backend one-folder> src-tauri/sidecar/dw2-backend   # desktop/freeze_trial/build_freeze.py
PATH=$HOME/.cargo/bin:$PATH npx tauri build
./postbundle-macos.sh    # restore the backend's 48 symlinks, rebuild the DMG (see below)
```
Toolchain: Rust 1.98.1 (rustup minimal profile), `@tauri-apps/cli` 2.11.5. First build 2 min 42 s; incremental about 40 s.

## Fixes found on the way (all in this commit)

1. **`HF_HUB_OFFLINE=1` was global in the frozen entry.**
   - It was set so the bundled DAv2 checkpoint loads offline, but it also blocked every other Hub download: the library manifest and the private DFC2019 previews (503 "cannot find the requested files in the local cache").
   - Now offline mode is on only while the bundled model loads. After that it is switched off (`huggingface_hub.constants`, which is read at call time) and the HF cache moves to a writable per-user folder.
2. **Uploads were written inside the install folder** (`<bundle>/_internal/data/uploads`), which is read-only once installed.
   - New optional `DW2_UPLOADS_DIR` in `backend/input/store.py` (dev default unchanged).
   - The desktop entry points it at the per-user cache: macOS `~/Library/Caches/DepthWizard/`, Windows `%LOCALAPPDATA%\DepthWizard`, Linux `$XDG_CACHE_HOME/DepthWizard`.
   - Verified: nothing is written into the bundle.
3. **Tauri's resource copy follows symlinks.**
   - The PyInstaller folder's 48 library symlinks (libtorch, GDAL/PROJ dylibs…, 457 MB behind them) became duplicate files: backend 800 → 1,256 MB inside the .app.
   - `desktop/tauri/postbundle-macos.sh` restores them and rebuilds the DMG. The backend inside the .app is then byte-for-byte its source size.
4. `build_freeze.py` now skips `--copy-metadata` for packages that aren't installed. (`requests` only came in with earthengine-api.)

## End-to-end test in the running desktop shell

**Method:**
- `DW2_E2E=1` makes the shell inject `desktop/tauri/src-tauri/src/e2e.js` into the page; it's a test-only hook.
- The script drives the real UI (Library tab → `sentinel2-almora` → START GENERATION), waits for the Relative Depth box, samples its pixels, and reports through a Tauri command.
- WKWebView can't be driven by Playwright, which is why the hook exists.

**Result on the final .app (after the symlink restore):**
- `ok: true`; page origin `tauri://localhost`; 88 library cards.
- The depth request went to `http://127.0.0.1:8765/api/depth/relative/library/sentinel2-almora`.
- Output: 518×518, gray 2–249.
- Caption: "Depth Anything V2 (ViT-Small) · this input | CPU 0.199s inference · 10.9s round trip".
- Backend listening 1.8 s after launch. 0 backend processes and a free port after the app exits.
- GeoTIFF upload through the bundled backend works (EPSG:32645, 10 m), which exercises the restored GDAL/PROJ symlinks.

**Latency:**
- Inference and the local HTTP hop are 0.19–0.21 s. Depth by id measured directly against the bundled backend is 0.7–1.2 s.
- The ~11 s in-app round trip was the GitHub download of the tile preview at that moment (a standalone download of the same 270 KB file took 75 s right after). Bundling the tiles locally removes that fetch.

**Test-only caveat:** the library ran in remote mode, with `DW2_LIBRARY=remote` and `HF_TOKEN` passed in the launch environment (not built into the app).
A shipped app cannot carry the private HF token. The tiered tile bundling must provide the library locally (and keep DFC2019 out of any public artifact).

## Size (macOS arm64, measured)

| | Size |
|---|---|
| **DepthWizard.app installed** | **854.5 MB** |
| · Tauri shell binary (runtime + the whole frontend embedded, compressed; `frontend/dist` is 130 MB, 129 MB of it `public/data`) | 57.4 MB |
| · frozen backend (sidecar) | 799.7 MB |
| **DMG download** | **446.7 MB** |

The shell adds **about 55 MB** on top of the backend.
The backend here is 800 MB (the earlier 813 MB build also contained `requests` and its dependencies, pulled in by the dropped earthengine-api).

## Torch → ONNX Runtime (2026-09-26): DONE, parity verified first

**Is there a lighter torch?** No:
- Official CPU wheels are already the lightest per platform: macOS arm64 is the only build (127 MB wheel); Windows `+cpu` = PyPI (124 MB); Linux `+cpu` 196 MB (PyPI's default is 555 MB plus CUDA).
- 369 of torch's 460 MB in the app is one file, `libtorch_cpu.dylib`.
- The real saving was dropping torch: ONNX Runtime 1.30.0 is 76 MB installed.

**Export** (`desktop/freeze_trial/export_onnx.py`, loading verbatim from the benchmark):
- opset 17, fixed 1×3×518×518 input, fp32; 99.1 MB; passes `onnx.checker`.

**Parity** (`desktop/freeze_trial/parity_onnx.py`, the three reference images used all session):

| Image | Preprocessing (numpy vs transformers) | Model, same input: max\|diff\| | Pearson |
|---|---|---|---|
| sentinel2-almora | **0 (bit-identical)** | 4.5e-6 | 1.0000000000 |
| dfc2019-JAX_004_006 | 0 | 2.8e-5 | 1.0000000000 |
| vhr-a_forest | 0 | 8.3e-6 | 1.0000000000 |

- The full pipeline (numpy preprocessing + ORT) equals the model-only numbers.
- JAX_004_006 at 2.8e-5 is slightly above the "~2e-5" example. That is the same order as the 2.64e-5 accepted for the compact wire format, and below its half-step.
- **Frozen app vs torch, as the UI receives it (u16):** max |diff| 2.95e-5, Pearson 0.9999999998. The torch app itself was 2.64e-5 against the same wire format.

**The swap:**
- `bridge/dav2_server_onnx.py`: same API and JSON as `bridge/dav2_server.py`, which stays for the Colab bridge and the Space.
- The entry mounts it. `build_freeze.py --onnx` bundles `models/dav2_small.onnx` and excludes torch, transformers, tokenizers and safetensors.
- It is built in a venv **without torch installed** (checked).
- The bundled HF cache and the offline switch are gone. The HF cache is the per-user one.

**Measured on the rebuilt app (after `postbundle-macos.sh`, which now restores 38 symlinks):**

| | torch app | **ONNX app** |
|---|---|---|
| DepthWizard.app installed | 854.5 MB | **352.1 MB** |
| · backend | 799.7 MB | **297.3 MB** |
| · shell binary | 57.4 MB | 57.4 MB |
| DMG | 446.7 MB | **232.7 MB** |
| backend listening after launch | 1.8 s | **0.4 s** |
| DAv2 inference (CPU) | 0.20–0.25 s | 0.23–0.24 s |

- **In-app E2E on the ONNX app:** `ok: true`, 88 cards, the depth request went to the sidecar, 518×518, gray 2–249.
- Caption: "CPU 0.237s inference · 11.3s round trip".
- The round trip is still dominated by the per-request GitHub preview download (see Latency above), which local tile bundling removes.

**CI:** `desktop/freeze_trial/freeze-trial.yml` now exports the ONNX model in a throwaway torch venv and builds without torch. Not run yet.
