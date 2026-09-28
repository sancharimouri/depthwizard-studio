# Next session: deploy everything (web + desktop) and sort out storage

Paste the prompt below into a fresh Claude Code session in the repo root.

---

Deploy all committed-but-undeployed work to the live website and the desktop app, and audit/fix the storage
setup. Work through the phases in order. Commit after each phase.

## Standing rules
- `git config user.name` must be **Sanchari Mouri** (`sancharimouri@gmail.com`), with no Co-Authored-By trailer.
- Use the superpowers `verification-before-completion` and `systematic-debugging` skills. Nothing counts as
  "deployed" until you've hit the live URL and seen it work.
- Stop and say exactly what you need for anything only I can do: logins (`gh auth`, `npx vercel login`, Render
  dashboard, Hugging Face), the Keychain password prompt for the updater key, or approval for an outward action.
- Ask before any outward or destructive action: pushing to a new remote, restarting or redeploying the HF Space,
  publishing a release, deleting remote files.
- Never:
  - force-push;
  - commit `.env` or any token;
  - embed `HF_TOKEN` in the desktop app;
  - publish any more DFC2019 data (licence: `docs/STORAGE.md` "Why DFC2019 is private"; `CLAUDE.md`
    "Tile-library storage").
- Don't change ML/research code.

## Phase 0: read, then check the git situation (no pushing yet)
1. Read:
   - `CLAUDE.md` (sections "2026-09-28 UI/UX sessions" and "Tile-library storage");
   - `docs/HANDOFF.md` §5 and §5z;
   - `docs/DEPLOY.md` (all of it, especially "LIVE" and "PENDING DEPLOY");
   - `docs/DESKTOP_APP.md` ("In-app updates", "Tiered tile library", "Real generation");
   - `docs/STORAGE.md`.
2. `git status`, `git log origin/main..main --oneline` (expect 25 commits, `619d1d7`…`9d0174d`), `git remote -v`.
3. **Resolve a contradiction before pushing.** This repo's `origin` is `sancharimouri/depthwizard-studio`, but
   `docs/DEPLOY.md` says that repo is a private, backend-only copy (`backend/` + `bridge/` + `requirements.txt`) that
   Render auto-deploys from. Check with `gh repo view` / `gh api repos/sancharimouri/depthwizard-studio/contents`
   what that repo really contains:
   - If it is the backend-only copy, do **not** push this whole research repo into it. Sync the three folders as
     `DEPLOY.md` describes.
   - If it mirrors this repo, pushing `main` is fine.

   Report what you found and which route you'll take.
4. Before any push, check the unpushed commits for large or forbidden files: no blob over 90 MB, no `.env`, no
   DFC2019 imagery, and nothing that should be gitignored under `data/`. For example:
   `git rev-list --objects origin/main..main | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sort -k3 -n | tail`.
   `uv.lock` has an uncommitted change from before 2026-09-28: find out why before committing it.

## Phase 1: the DAv2 inference Space (blocker for any generation test)
- Diagnosis from 2026-09-28 (`docs/HANDOFF.md` §5z):
  - `sancharimouri/DepthWizard2` says RUNNING, but every `/gradio_api/upload` returns 502.
  - Gradio runs in SSR mode (Node :7860 → Python :7861), and the Python side stopped answering.
- Check: `curl -X POST localhost:8000/api/generate/library/sentinel2-almora` with the local backend running
  (`npm run dev` in the repo root). Also read the Space run log:
  `curl -N -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/spaces/sancharimouri/DepthWizard2/logs/run`.
- Fix (ask me first):
  - restart the Space (`HfApi().restart_space`);
  - then propose `ssr_mode=False` in `space/app.py` `demo.queue(...).launch()`, and fix `space/README.md`, which says
    "CPU-only" although the hardware is `zero-a10g`;
  - push `space/` to the Space repo only after my OK.
- Verify: 3 consecutive successful generations through the local backend.
- If it's still broken, the backend's `DAV2_FALLBACK_URL` plus `bridge/dav2_server.py` is the documented fallback.

## Phase 2: tests
- `cd frontend && npm test` (expect 86/86) and `npx vite build`.
- `uv run python -m pytest -q backend/tests` from the repo root (expect 45 passed, 1 skipped).
- Stop on any failure.

## Phase 3: backend → Render (`docs/DEPLOY.md` "LIVE")
- Changed since the live deploy:
  - `backend/api/depth_routes.py`: Space retry, `SPACE_RETRY_DELAYS_S`, `_is_transient`;
  - `backend/dem/glo30.py`: `mosaic_to_grid`, the GLO-30 tile-seam fix;
  - `backend/terrain/mesh_export.py`: `fill_nan_nearest`;
  - the DFC2019 terrain-pack web path: `backend/storage/library_store.py` `private_pack`, and
    `backend/generation/pipeline.py` `_library_elevation`;
  - plus anything else in `git diff <last-deployed-commit>..main -- backend bridge requirements.txt`.
  Find the last deployed commit from the Render dashboard or the copy repo's history.
- Sync to the Render source as decided in Phase 0. Push, then wait for the autodeploy.
- Check `requirements.txt` has everything the new code imports (`scipy` for `fill_nan_nearest`, `rasterio.merge`).
- Verify live, against `https://depthwizard2-api.onrender.com`:
  - `/openapi.json` → 200;
  - `/api/library` → 89 items;
  - `POST /api/generate/library/sentinel2-darjeeling` → 200, with `terrain.json` showing no pit deeper than ~50 m
    below its 5×5 neighbourhood (the old bug gave ~1,850 m at the south edge);
  - one DFC2019 generation uses its private pack (`meta.how` = curated elevation pack, not flat);
  - CORS still allows exactly `https://depthwizard-studio.vercel.app` and `https://depthwizard2.vercel.app`.

## Phase 4: frontend → Vercel
- `cd frontend && npx vercel deploy --prod` (project `sherry-c508/depthwizard2`; `VITE_API_BASE` is a production
  env var, don't change it).
- Verify on `https://depthwizard-studio.vercel.app` in a real browser, with 0 console errors:
  - Routes: bare URL → DW Studio; `#/docs` → Home; `#demo-video` → its section; `#/demo` → Demo.
  - Cold start: Library thumbnails show the "server is waking up" message after 2.5 s. Render sleeps after 15 min.
  - A full generation: each box's readout lasts ≥ 1.3 s; the log finishes before the DEM Elevation box; the view
    expands.
  - Open another job mid-generation; switch jobs and back (vertical exaggeration and rotation are kept).
  - Flat-terrain warning on an agricultural tile.
  - Darjeeling south edge has no spikes.
  - Sidebar:
    - drag to resize, and past the minimum to collapse;
    - the D opens Home;
    - the footer links work.
  - Background icons (17) float, react to the pointer, keep 1 cm apart, spread over all four sides, and are never
    more than 60% hidden. Faint star dots appear in the margins, never on the boxes.
  - Library terrain dropdown: pick Hilly, then press DFC2019. A notice should explain there are none, the tiles stay,
    and the notice closes itself after 4 s.
- Update `docs/DEPLOY.md` "LIVE" with the date and what you verified.

## Phase 5: desktop app → v1.0.2 (`docs/DESKTOP_APP.md`)
- Bump `version` in `desktop/tauri/src-tauri/tauri.conf.json` (1.0.1 → 1.0.2) and in `Cargo.toml`.
- Rebuild the frozen Python sidecar with the new backend code. `docs/DESKTOP_APP.md` has the sidecar build steps;
  follow them exactly. The frontend build bundled into the app must be the new one.
- `desktop/tauri/build-signed.sh`. It reads `~/.tauri/depthwizard2-updater.key` and the Keychain item
  `depthwizard2-updater-signing`; ask me if a prompt appears.
- Test the built app locally:
  - library, upload and a generation (the desktop uses the ONNX sidecar, not the Space);
  - Darjeeling and a flat tile;
  - the new UI.
- `desktop/tauri/publish-release.sh`: publishes to `sancharimouri/depthwizard2-desktop`; ask before running.
- Verify:
  - `https://github.com/sancharimouri/depthwizard2-desktop/releases/latest/download/latest.json` says `"version": "1.0.2"`;
  - its signature matches the uploaded `.app.tar.gz`;
  - an installed 1.0.1 offers the update and relaunches as 1.0.2.
- Check every release asset is well under GitHub's 2 GB per-file limit (v1.0.0's artifact was ~306 MB).

## Phase 6: storage audit and fixes (report numbers for each)
1. **Hugging Face (free account):**
   - sizes of the private dataset `sancharimouri/depthwizard2-library-private` (`tiles/`, `previews/`, `thumbnails/`,
     `manifest.json`, and the new `dem/` DFC2019 packs from commit `7c8cbec`);
   - the private model `sancharimouri/depthwizard2-method6`;
   - the Space's own storage.

   Compare against the current free-tier limits (look them up; don't guess). Confirm the backend's `HF_TOKEN` on Render
   can read `dem/`.
2. **GitHub releases:**
   - `sancharimouri/depthwizard2-assets@library-v1`: total size and asset count, and whether the `manifest.json` URLs
     match it.
   - `depthwizard2-desktop` releases: total size; consider deleting superseded test artifacts (ask first).
3. **Render (free):** generated outputs go to `/tmp` (`DW2_{CACHE,GENERATED,UPLOADS}_DIR`), which is ephemeral.
   - Confirm there's no unbounded growth within a single instance lifetime: `data/uploads` is "never cleaned
     automatically" per HANDOFF §5, and the generated job folders build up.
   - If needed, add a small age/size cap for generated and upload folders. Keep it simple and tested.
4. **Local disk and repo:**
   - `du -sh` of `data/`, `external/`, `models/`, `node_modules`, and the HF cache;
   - check that `.gitignore` covers the ~777 untracked entries under `data/` and elsewhere that must never be
     committed.

   List anything big that could be safely removed. Don't delete without my OK.
5. `docs/STORAGE.md`: bring the tables and sizes up to date. Add the `dem/` packs and the desktop release sizes.

## Phase 7: docs and wrap-up
- Update:
  - `docs/HANDOFF.md` §5z: mark deployed, with dates, commit hashes and live verification;
  - `docs/DEPLOY.md`: LIVE and PENDING DEPLOY;
  - `docs/DESKTOP_APP.md`: v1.0.2;
  - `docs/STORAGE.md`;
  - `CLAUDE.md` (keep it short).
- Commit. Push only through the route decided in Phase 0.
- Report per phase: what was done, the evidence (URLs, status codes, versions, sizes), anything skipped, and what
  still needs me.

## File map (2026-09-28 work, for reference)
- Frontend:
  - `frontend/src/main.js`: jobs, viewer memory, retry, generation sequence, calculation log.
  - `frontend/src/progress-sync.js`
  - `frontend/src/flat-warning.js`
  - `frontend/src/outlier-relief.js`
  - `frontend/src/terrain.js`
  - `frontend/src/input-view.js`
  - `frontend/src/bg-float.js`
  - `frontend/src/bg-icons.js`
  - `frontend/src/sidebar.js`
  - `frontend/src/routes.js`: unchanged; permanent routes.
  - `frontend/src/side-panels.js`
  - `frontend/index.html`
  - `frontend/src/styles.css`
  - `frontend/public/brand-d.png`
  - `frontend/tests/*.test.mjs`
- Backend:
  - `backend/api/depth_routes.py`
  - `backend/dem/glo30.py`
  - `backend/terrain/mesh_export.py`
  - `backend/tests/test_depth_routes.py`
  - `backend/tests/test_dem.py`
- Space: `space/app.py`, `space/README.md`, `space/requirements.txt`.
- Desktop:
  - `desktop/tauri/src-tauri/tauri.conf.json`
  - `desktop/tauri/build-signed.sh`
  - `desktop/tauri/publish-release.sh`
  - `desktop/tauri/postbundle-macos.sh`
  - `desktop/tiles/build_dem_pack.py`
- Config: `.mcp.json` (shadcn MCP), root `package.json` (`shadcn` dev dependency), `frontend/vite.config.js`
  (proxy → :8000).
- Icon sources (untracked): `icons/1.svg`, `icons/2.svg`, `icon.png`.
