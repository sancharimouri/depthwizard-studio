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
