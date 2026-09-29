# Desktop app (Tauri 2 shell + frozen Python sidecar) — status 2026-09-29 (v1.0.2 released)

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

## Tiered tile library (2026-09-26)

**Selection** (`desktop/tiles/selection.json`, rules and numbers inside):

| Collection | Bundled in full | On demand (thumbnail + overlay) | How chosen |
|---|---|---|---|
| DFC2019 | 8 (2026-09-27: OMA_364_043, OMA_315_019 moved to on demand at the owner's request) | 42 | Method 6 per-tile results (`method6_finetune_twinhead/method6_results.json`, mean over the 4 quadrant folds), ranked by the mean of per-metric ranks (MAE, RMSE ↑; Pearson, Spearman ↓). OMA_315_020, OMA_212_033, JAX_161_001, OMA_269_035, OMA_364_043, OMA_315_019, OMA_364_003, OMA_225_001, JAX_072_015, OMA_248_029 |
| Sentinel-2 | 11 (2026-09-27: + Darjeeling, the demo scene; FABDEM + GLO-30 pack) | 22, from the **public** GitHub Release | Final post-QC actual (SCL) cloud + shadow + nodata = 0 (22/32 eligible; 2 with an unmeasured value excluded), real ICESat-2 photon + 20 m segment files on disk, 2 per category + 2 more by ICESat-2 ground-photon count (max 3 per category). Agricultural: bathinda, nizamabad, kota. Coastal: amalapuram, bhitarkanika. Hilly: manali, almora. Urban: jaipur, hyderabad, pune |
| Maxar | 6 | — | all |

**Build:** `desktop/tiles/build_bundle.py OUT`.
- Bundled tiles are re-encoded losslessly (deflate, predictor 2, level 9; each verified pixel- and georeference-identical): 104.8 → 73.3 MB.
- Bundle total: **81.9 MB** (tiles 73.3, previews 7.1, 88 thumbnails 1.5).

**Runtime** (`DW2_LIBRARY=bundle`, `backend/storage/library_store.py`):
- Bundled and already-downloaded files are served from disk.
- `POST /api/library/{id}/download` writes on-demand items (atomic `.part` → rename) into the per-user cache.
- The frontend shows a translucent overlay ("Download · 2.3 MB") on on-demand cards; clicking downloads, then selects.

**DFC2019 on demand:**
- The private HF dataset returns 401 to anonymous requests (checked for manifest, tile, preview and thumbnail).
- So the app first refused ("Needs a Hugging Face token"). Superseded the same day: the 40 tiles are now on the public GitHub Release, see Hosting below.

**Bug fixed:** `load_dotenv()` searches upward from the install path. The packaged backend built inside the repo loaded the repo's `.env`, including `HF_TOKEN`. The desktop entry now sets `DW2_NO_DOTENV=1`.

**Timing** (in-app E2E, shipped conditions: no token, fresh cache, production v1.0.0):
- Depth by id for bundled Sentinel-2 / DFC2019 / Maxar: **0.22–0.26 s**. It was ~11 s when every request re-downloaded the preview from GitHub. The UI caption reads "0.3s round trip".
- On-demand Sentinel-2 (bengaluru, 2.3 MB), download through the overlay: **1.4 s** in the final run, **79 s** in an earlier one. It is the same public GitHub download; the difference is the network. Afterwards its depth is 0.22 s like a bundled tile.

## In-app updates: Tauri updater plugin (2026-09-26)

**Pieces:**
- `tauri-plugin-updater` 2.12 + `tauri-plugin-dialog` (prompt) + `tauri-plugin-process`.
- `check_for_update` in `src-tauri/src/main.rs` runs on every launch (skipped in the E2E hook):
  - fetches the manifest, and if a newer version exists shows **"Update available … Update / Later"**;
  - on Update: download → **signature check against the public key built into the app** → install → stop the sidecar → restart.
- `bundle.createUpdaterArtifacts: true`. Version **1.0.0** is this build (ONNX + tiered tiles).

**Signing key (never in the repo):**
- Private key: `~/.tauri/depthwizard2-updater.key` (minisign, mode 600, `~/.tauri` mode 700).
- Its password: random, in the macOS Keychain, service `depthwizard2-updater-signing` (`security find-generic-password -s depthwizard2-updater-signing -w`).
- Public key: `plugins.updater.pubkey` in `tauri.conf.json` (also `~/.tauri/depthwizard2-updater.key.pub`).
- **Back up the private key and password somewhere safe. If they are lost, installed apps can never accept another update** (they only trust this public key).
- Build signed: `desktop/tauri/build-signed.sh [tauri build args]`. It exports `TAURI_SIGNING_PRIVATE_KEY(_PASSWORD)` from those two places, runs `tauri build` + `postbundle-macos.sh`.
  The post-bundle step rebuilds `DepthWizard.app.tar.gz` from the fixed .app and **re-signs** it, because Tauri signs before the symlink restore.

**Publishing a release** (manifest format as tested):
```json
{"version": "1.0.1", "notes": "…", "pub_date": "2026-…Z",
 "platforms": {"darwin-aarch64": {"signature": "<contents of DepthWizard.app.tar.gz.sig>",
                                  "url": "https://…/DepthWizard_1.0.1.app.tar.gz"}}}
```
- Upload `latest.json` and the `.app.tar.gz` to the release; the configured endpoint is `…/releases/latest/download/latest.json`.
- **`version` must equal the artifact's own version.** The test showed a mismatch re-offers the "new" version after every restart.

**Verified** against a real signed test manifest (`updater-test.conf.json`: local HTTP endpoint, test builds only), installed v1.0.0 → served v1.0.1:
- **Genuine artifact:** "1.0.1 available" → accepted → downloaded → installed → restarted. The relaunched app printed "Depth Wizard 1.0.1 · update check: up to date (1.0.1)", and the installed Info.plist says 1.0.1.
- **Tampered artifact** (1 byte changed, genuine signature): "The signature verification failed", and the install stayed at 1.0.0.
- **Prompt:** launched normally (LaunchServices), the dialog appears and **waits**: no action for 45+ s.
  Clicking **Update** installs (done by hand by the user, three times, during the test).
  **Later** was clicked by hand (2026-09-26) and logged "postponed by the user". The server saw only the manifest request, with no artifact download.
  The version stayed 1.0.0 and the app kept running (depth by id still 0.24 s).
- **Production v1.0.0** checks its real endpoint on launch. Nothing is published yet, so it logs "Could not fetch a valid release JSON" and carries on normally.

**Other findings:**
- Tauri refuses to run from a path containing a symlink (macOS security check; e.g. `/var/…` instead of `/private/var/…`). The shell now reports that clearly instead of panicking with `resource dir: UnknownPath`.
- **Hosting (done 2026-09-26, owner decision: everything public on GitHub):**
  - `sancharimouri/depthwizard2-desktop` (public; releases + README only; no source).
  - Release **v1.0.0**: `DepthWizard_1.0.0_aarch64.dmg` (installer), `DepthWizard.app.tar.gz` + `.sig` (update artifact) and `latest.json`.
  - Published with `desktop/tauri/publish-release.sh`, which creates the repo if needed and writes `latest.json` with the built app's own version.
  - Verified:
    - anonymous `latest.json` at the updater endpoint → HTTP 200, and its signature matches the built artifact;
    - the artifact URL resolves (305.6 MB);
    - the shipped v1.0.0 app logs "update check: up to date (1.0.0)" against the live release.
- **DFC2019 on demand, now token-free.** The 40 on-demand DFC2019 tiles (+ previews) are in the **public** `depthwizard2-assets@library-v1` release (`desktop/tiles/publish_dfc_ondemand.py`; lossless re-encode verified, README marked TEMPORARY with the terms restriction). The bundle downloads every on-demand item from GitHub.
  - In-app E2E with no token: the DFC2019 on-demand card **downloaded**.
  - The owner's HF token was **not** embedded: it has write access to the whole account, so it would have been extractable from a public app.
- **Releasing an update:** bump `version` in `tauri.conf.json` (and `Cargo.toml`), then `desktop/tauri/build-signed.sh`, then `desktop/tauri/publish-release.sh`.
  Installed apps see it on their next launch.

## v1.0.0 size (production build, measured)

| | Size |
|---|---|
| **DepthWizard.app installed** | **432.4 MB** |
| · backend (ONNX Runtime) | 297.4 MB |
| · bundled tile library | 78.4 MB |
| · shell + frontend | 59.3 MB |
| **DMG (installer)** | **316.2 MB** |
| Update artifact (`.app.tar.gz`) | 305.6 MB |

## v1.0.2 (released 2026-09-29)

**What's new:** the 2026-09-28/29 UI (DW Studio / Demo / Home, sidebar, jobs, readouts, bright mode, GSD card) and every
backend fix since v1.0.1: tile-seam mosaic, nearest-neighbour hole fill, manual GSD, Space retries. The desktop depth path
is the local ONNX sidecar, not the Space. The DFC2019 terrain packs are not in the desktop app (still a flat plane).

**Build** (as "Build" and "Torch → ONNX Runtime" above):
- `build_freeze.py` in a fresh torch-free Python 3.11 venv from `requirements-freeze.txt`.
- `dav2_small.onnx` reused byte-for-byte from the v1.0.1 sidecar (sha256 `352d84f3…`), so no re-export.
- **scipy 1.17.1 added to `requirements-freeze.txt`:** `fill_nan_nearest` imports it when a height grid has holes
  (live GLO-30 on a tile seam). Without it, those generations would fail.
- Then `build-signed.sh` and `publish-release.sh`. 41 symlinks restored.

**Measured sizes (2026-09-29, `ls -l` / `du -sk` on the released artifacts and an installed copy):**

| | v1.0.1 | **v1.0.2** |
|---|---|---|
| DMG (installer) | 340.4 MB (340,418,450 B) | **355.7 MB** (355,727,127 B) |
| Update archive `DepthWizard.app.tar.gz` | 328.7 MB | **343.2 MB** (343,215,706 B) |
| `DepthWizard.app` installed | 455.2 MB | **492.2 MB** |
| · frozen backend (ONNX Runtime + scipy) | 297 MB | **335 MB** (scipy ~38 MB) |
| · of which the ONNX model | 95 MB | 95 MB |
| · bundled tile library + elevation packs | 99 MB | 99 MB (unchanged since 2026-09-27) |
| · shell + embedded frontend | 57 MB | 57 MB |
| Per-user cache after 2 on-demand downloads + generating them | — | 10 MB (library 5 MB, generated 5 MB) |

**Verified:**
- **Frozen backend alone** (no token, fresh cache):
  - 89 items, `/gsd` listed;
  - bundled Sentinel-2 / Maxar / DFC2019 generate in 0.3–1.1 s, DAv2 on the CPU in ~0.2 s;
  - Darjeeling from its pack: 554–2,476 m, max 32.6 m below its 5×5 median;
  - a Darjeeling GeoTIFF upload through live GLO-30 across 27°N: 554–2,478 m;
  - a PNG at 0.5 m → 200 × 150 m;
  - listening 0.35 s after launch (the first launch of a new build took ~26 s: macOS scans new binaries once).
- **In-app E2E** (`DW2_E2E=1`, the signed build): `tauri://localhost`, 0 page errors, Almora → Studio in 6.8 s
  (ONNX 0.239 s), Maxar and DFC2019 generation 200.
- **On demand:** Bengaluru 3.0 MB in 1.7 s and DFC2019 `OMA_364_043` 2.2 MB in 2.6 s, both then generated.
- **Updates:**
  - the released `latest.json` says 1.0.2, and its signature verifies against the built-in public key for the artifact
    downloaded from GitHub (a changed byte fails);
  - `…/releases/latest` → v1.0.2;
  - the **published v1.0.1**, installed and launched (auto-accept test flag), logged "1.0.2 available → downloaded →
    installed 1.0.2; restarting", then "Depth Wizard 1.0.2 · update check: up to date (1.0.2)" (~60 s).
- **Licence:** v1.0.2 ships exactly v1.0.1's DFC2019 files (8 tiles, 50 thumbnails); nothing new was published.

## Real generation (2026-09-26): no more Darjeeling placeholders

Every Workbench box and the Depth Wizard Studio now show the **job's own** result. The only request is `POST /api/generate/{library|input}/{id}` (`backend/generation/pipeline.py`).
It runs DAv2-Small relative depth once, then writes the viewer's asset contract (`terrain.json`, `satellite.png`, `relative_depth.png`, `elevation.png`) plus `meta.json` with the sources used.

**Elevation always comes from a real elevation model, never from the image** (CLAUDE.md framing):

| Input | Terrain (DEM) | Surface (mesh) | How |
|---|---|---|---|
| Sentinel-2 library (32) | FABDEM | Copernicus GLO-30 | bundled elevation pack, ~30 m, on the tile's own grid |
| Maxar library (6) | FABDEM | FABDEM + Method 6 above-ground height (research model, labelled; canopy ceiling ~18–23 m) | bundled pack, 1.2 m |
| Upload / Search Online (georeferenced) | the attached DEM (user / FABDEM) | live GLO-30 | the attached DEM; or live GLO-30 for both when none is attached |
| DFC2019, plain PNG/JPG | — | — | no georeference: flat plane, and every box says so |

**Pieces:**
- Elevation packs: `desktop/tiles/build_dem_pack.py` (38 packs, 22.6 MB, built from the benchmark's FABDEM/GLO-30 and the VHR pipeline outputs). They are bundled (`library/dem/`, item field `dem`).
- Live GLO-30: `backend/dem/glo30.py`, public COGs on AWS, ranged reads, no account.
  - Darjeeling across the 27°N tile boundary: 100% valid.
  - Live vs bundled pack: Pearson 0.99998. Kohima upload: 852.0–2,372.2 m vs pack 851.8–2,372.5 m.
- Frontend:
  - the viewer's `loadRegion(key, assets)`; the job is registered as a region with its real sources, CRS, grid and inference time;
  - captions, Studio stats, layer descriptions and the calculation log come from `meta`;
  - no Darjeeling defaults are left in the generation path (Explore's demo regions are unchanged).
- **WebKit quirk:** a CORS texture load can reuse the cached non-CORS `<img>` response of the same URL and then fail, so textures use their own URLs (`?tex=1`). A texture failure is logged instead of hanging the run.
  Before the fix: 1 of 2 in-app runs stalled. After: 4 of 4 passed.

**Verified:**
- **Browser** (Chrome), full Workbench → Studio for Almora, Maxar forest and DFC2019: real per-job boxes and stats, 0 page errors.
- **Desktop app, in-app E2E × 4** (no token, fresh cache):
  - Almora → Studio in ~23 s (mostly the staged animation), showing 1,023–2,042 m · GLO-30 · EPSG:32644, with every image from `/api/generated/`;
  - Maxar and DFC2019 generation 200;
  - 0 page errors.
- **Frozen backend:** a real GeoTIFF upload (Kohima) generated from live GLO-30 in 3.5 s.
- **Sizes (measured):** app **455.2 MB** installed (library with tiles + elevation packs 100.1 MB), DMG **340.9 MB**. Not yet published: that needs a v1.0.1 release.
