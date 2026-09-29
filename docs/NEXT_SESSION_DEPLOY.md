# Next session: ship everything to the website and the desktop app, sort out storage/GPU, report sizes and load times

> **Executed 2026-09-29.** Everything below is deployed; see `docs/CHANGES_2026-09-28_29.md` "Deployed" and
> `docs/DEPLOY_REPORT_2026-09-29.md`. Kept as the record of what was asked.

Ship every committed-but-undeployed change to the live website (Vercel frontend + Render backend) and the desktop app,
make storage and GPU work reliably, and finish with a measured report. Work through the phases in order and commit after
each one. The full list of what changed, with file references, is `docs/CHANGES_2026-09-28_29.md`: read it first.

## Standing rules (every phase)
1. **Identity.**
   - `git config user.name` must be `Sanchari Mouri` and `git config user.email` must be `sancharimouri@gmail.com`.
     Check before the first commit.
   - Every commit: author and committer Sanchari Mouri, no Co-Authored-By trailer.
   - Before any push, verify with `git log origin/main..HEAD --format='%an <%ae> | %cn <%ce>' | sort | uniq -c`: exactly
     one identity, no Co-Authored-By.
2. **The SIH deck links must never break.** The submitted deck (`145604_SIH26175.pdf`, links extracted 2026-09-29) links
   exactly these, and they must all work at the end:
   - `https://depthwizard-studio.vercel.app/`, which opens DW Studio (the input page).
   - `https://depthwizard-studio.vercel.app/#/demo`, which opens the Demo page.
   - `https://depthwizard-studio.vercel.app/#demo-video`, which opens Home and scrolls to the demo-video section.
   - `https://github.com/sancharimouri/depthwizard-studio`: must stay reachable at this URL.
   - `https://github.com/sancharimouri/depthwizard2-desktop/releases/latest`: must keep serving a working installer.
   - `https://github.com/sancharimouri`, `https://www.linkedin.com/in/sancharimouri`, `mailto:sancharimouri@gmail.com`.
   - Also permanent: `…/#/docs` (Home).

   What this means in practice:
   - Never rename, delete, transfer or make private the `depthwizard-studio` or `depthwizard2-desktop` repos.
   - Never change the Vercel project's production domain `depthwizard-studio.vercel.app`.
   - Never edit or remove a route in `frontend/src/routes.js` (only add).
   - Never remove `id="demo-video"`.
   - `frontend/tests/routes.test.mjs` "SIH deck links keep resolving" must pass.
3. **Storage and GPU use may go up or down freely**, if it makes the site and app more reliable or faster, as long as no
   link above breaks.
   - Anything that costs money needs my explicit OK first, with the price: a paid Render instance, HF PRO or paid
     hardware, R2 activation, paid Vercel.
   - Nothing may silently switch to a paid plan.
4. **Stop and ask** for anything only I can do, and say exactly what you need:
   - logins: `gh auth login`, `npx vercel login`, the Render dashboard, huggingface.co;
   - the macOS Keychain prompt for the updater signing key;
   - approval of any outward action: push, deploy, Space restart, release publish, deleting remote files.
5. **Licences:** never publish any more DFC2019 data (`docs/STORAGE.md` "Why DFC2019 is private"). Never embed `HF_TOKEN`
   (or any token) in the desktop app or the frontend. Never commit `.env`.
6. **Evidence before claims.**
   - Use the superpowers `verification-before-completion` and `systematic-debugging` skills.
   - "Deployed" means you opened the live URL and saw the change work.
   - Report failures with their actual output.
7. Don't change ML/research code. Don't force-push. Don't rewrite pushed history.

## Local test stack (for checks that don't need the live site)
The HF Space was down on 2026-09-28, so this runs everything on this Mac, next to (not instead of) `npm run dev`:
- **Inference host:** `uv run python -m uvicorn bridge.dav2_server:app --port 8766` (DAv2-Small on CPU, `/predict`).
- **Test backend:** `DAV2_INFERENCE_URL=http://localhost:8766 uv run uvicorn backend.main:app --port 8001`.
- **Test frontend:** a Vite config outside the repo containing only
  `export default { root: "<repo>/frontend", server: { port: 5174, strictPort: true, proxy: { "/api": { target: "http://localhost:8001", changeOrigin: true } } } }`,
  started with `cd frontend && npx vite --config <that file>`. Import nothing from `vite` in it.
- **Headless Chrome with a real GPU** (background tabs throttle timers to 1 s, so don't use them for timing):
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --remote-debugging-port=9333 --user-data-dir=<tmp> --use-angle=metal --enable-gpu --ignore-gpu-blocklist about:blank`.
  Drive it over the DevTools protocol with a small Node script:
  - `Page.navigate`, then `Runtime.evaluate` with `awaitPromise`, then `Page.captureScreenshot`;
  - `DOM.setFileInputFiles` for uploads;
  - `document.getAnimations()` to pause CSS animations.

## Phase 0: read and check (no pushing yet)
1. Read:
   - `CLAUDE.md` (sections "2026-09-28 UI/UX sessions" and "Tile-library storage");
   - `docs/CHANGES_2026-09-28_29.md` (all of it);
   - `docs/HANDOFF.md` §5 and §5z;
   - `docs/DEPLOY.md` (all of it, especially LIVE and PENDING DEPLOY);
   - `docs/DESKTOP_APP.md` (Build, In-app updates, Tiered tile library, Real generation);
   - `docs/STORAGE.md`.
2. `git status`, `git log origin/main..HEAD --oneline` (expect 46 commits: the DFC2019 pack series `9145d44`…
   `0270343`, then these sessions up to the docs commit that added this file), and `git remote -v`.
3. **Resolve the push route before anything else.**
   - This repo's `origin` is `sancharimouri/depthwizard-studio`, the repo linked in the SIH deck. But `docs/DEPLOY.md`
     says that repo is a private, backend-only copy (`backend/` + `bridge/` + `requirements.txt`) that Render
     auto-deploys from `main`.
   - Check with `gh repo view sancharimouri/depthwizard-studio --json visibility,defaultBranchRef,diskUsage` and list its
     top-level contents (`gh api repos/sancharimouri/depthwizard-studio/contents`).
   - Then decide, and tell me before pushing:
     - **If it's the backend-only copy:** don't push this whole research repo into it (Render would redeploy a
       different tree). Sync only `backend/`, `bridge/` and `requirements.txt`, as `DEPLOY.md` describes.
     - **If it mirrors this repo:** pushing `main` is fine.
     - **Either way:** the repo must stay at that URL. If it's private, the deck link 404s for judges, so tell me and
       don't change its visibility yourself.
4. **Scan the unpushed commits** for any blob over 90 MB, any `.env` or token, any DFC2019 imagery, and anything under
   `data/` that should be gitignored:
   `git rev-list --objects origin/main..HEAD | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sort -k3 -n | tail`.
5. **`uv.lock`** has an uncommitted change from before 2026-09-28. Find out what changed (`git diff uv.lock`) and commit
   it only if it's intended.

## Phase 1: the DAv2 inference Space (blocks every live generation)
- **Diagnosis (2026-09-28, `docs/HANDOFF.md` §5z):**
  - `sancharimouri/DepthWizard2` reports RUNNING, but every `/gradio_api/upload` returns 502.
  - Gradio runs in SSR mode (Node proxy :7860 → Python :7861), and the Python app stopped answering; its run log had no
    Python output after its 2026-09-25 start.
  - The backend now retries 502s twice (`backend/api/depth_routes.py`), but that only covers short blips.
- **Check:** `curl -X POST localhost:8000/api/generate/library/sentinel2-almora` with `npm run dev` running, plus the
  run log: `curl -N -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/spaces/sancharimouri/DepthWizard2/logs/run`.
- **Fix, with my OK:**
  1. Restart it (`HfApi().restart_space`).
  2. Propose `ssr_mode=False` in `space/app.py` `demo.queue(max_size=8).launch(...)`, and fix `space/README.md`, which
     says "CPU-only" although the hardware is `zero-a10g`.
  3. Push `space/` to the Space repo only after my OK.
  4. Consider keeping it warm or raising its limits (ZeroGPU quota: `docs/DEPLOY.md` "ZeroGPU daily quota"), within
     rule 3.
- **Done when:** 5 consecutive generations through the local backend succeed. Record their timings.

## Phase 2: tests
- `cd frontend && npm test`: expect **92/92**. Then `npx vite build`.
- `uv run python -m pytest -q backend/tests` from the repo root: expect **48 passed, 1 skipped**.
- Stop on any failure.

## Phase 3: backend → Render (`docs/DEPLOY.md` "LIVE")
- **Everything backend that's undeployed** (`docs/CHANGES_2026-09-28_29.md` §2, plus the DFC2019 pack series):
  - `backend/api/depth_routes.py`: Space retry;
  - `backend/dem/glo30.py`: `mosaic_to_grid`, the tile-seam fix;
  - `backend/terrain/mesh_export.py`: `fill_nan_nearest`;
  - `backend/input/store.py` + `backend/api/input_routes.py`: manual GSD, `POST /api/input/{id}/gsd`;
  - `backend/generation/pipeline.py`: plane sized from the manual GSD, and the DFC2019 pack path via
    `backend/storage/library_store.py` `private_pack`;
  - and anything else in `git diff <last-deployed-commit>..HEAD -- backend bridge requirements.txt`. Find the last
    deployed commit from the Render dashboard or the copy repo's history.
- **Before pushing:** check `requirements.txt` covers the imports (`scipy` is used by `fill_nan_nearest`; `rasterio.merge`).
- **Sync** by the Phase 0 route, push (with my OK), and wait for the autodeploy.
- **Verify on `https://depthwizard2-api.onrender.com`:**
  - `/openapi.json` → 200; `/api/library` → 89 items; `/api/input/{id}/gsd` is listed in `/openapi.json`.
  - `POST /api/generate/library/sentinel2-darjeeling` → 200. Its `terrain.json` has no pit deeper than ~50 m below
    its 5×5 neighbourhood (the old bug gave ~1,850 m at the south edge).
  - One DFC2019 generation uses its private pack (`meta.how` = curated elevation pack).
  - A PNG upload with a manual GSD of 0.5 m gives a 200 × 150 m plane for a 400 × 300 px image.
  - CORS allows exactly `https://depthwizard-studio.vercel.app` and `https://depthwizard2.vercel.app`.

## Phase 4: frontend → Vercel
- `cd frontend && npx vercel deploy --prod` (project `sherry-c508/depthwizard2`). Leave `VITE_API_BASE` alone, and
  don't touch the domains.
- **Verify on `https://depthwizard-studio.vercel.app` in a real browser**, dark and bright mode, 0 console errors,
  following `docs/CHANGES_2026-09-28_29.md` §1:
  - Every SIH deck link from rule 2 lands where it should. Record the result per link.
  - **Sidebar:**
    - D mark → Home;
    - page labels DW Studio / Demo / Home;
    - drag to resize, and past the minimum to collapse;
    - footer links.
  - **Input page:**
    - the Terrain dropdown, and its "no tiles" notice (Hilly + DFC2019);
    - cold-start messages;
    - a PNG upload shows the GSD card, and Start waits for it;
    - the box outline glow;
    - the 12 background icons cruise, bounce, keep apart, the big satellite stays top-left and the earth + satellite
      bottom-right, and they're never more than 30% hidden;
    - sparse stars.
  - **Generation:**
    - each box's readout lasts at least 1.3 s, and the log finishes before the DEM Elevation box;
    - open another job mid-generation (the running one keeps going);
    - switch jobs and back (exaggeration and rotation kept);
    - a failed job shows "Failed" + Retry.
  - **Studio:**
    - flat-terrain warning on an agricultural tile;
    - vertical exaggeration at max without spires;
    - Darjeeling's south edge without spikes;
    - the wireframe is green in dark mode and orange in bright mode, and recolours live.
  - **Bright mode:** no green anywhere except the Tier 2 "T2" badges; outlines crisp.
- **Update `docs/DEPLOY.md` "LIVE"** with the date and what you verified.

## Phase 5: desktop app → v1.0.2 (`docs/DESKTOP_APP.md`)
1. Bump `version` 1.0.1 → 1.0.2 in `desktop/tauri/src-tauri/tauri.conf.json` and `desktop/tauri/src-tauri/Cargo.toml`.
2. **Re-freeze the backend.** `src-tauri/sidecar/dw2-backend` is the OLD frozen backend, so the app won't get the
   backend fixes without this.
   - Run `desktop/freeze_trial/build_freeze.py --onnx`, following `docs/DESKTOP_APP.md` "Build" and "Torch → ONNX
     Runtime" exactly.
   - Replace `src-tauri/sidecar/dw2-backend` with the new one-folder build.
3. The frontend is rebuilt by Tauri's `beforeBuildCommand` (`VITE_API_BASE=http://127.0.0.1:8765 npx vite build`, in
   `tauri.conf.json`). Confirm `frontend/dist` has the new UI.
4. Run `desktop/tauri/build-signed.sh`. It reads `~/.tauri/depthwizard2-updater.key` and the Keychain item
   `depthwizard2-updater-signing`; ask me if a prompt appears. `postbundle-macos.sh` runs inside it and restores the 48
   symlinks.
5. **Test the built app locally:**
   - library (Sentinel-2, Maxar, the bundled DFC2019 tiles, and an on-demand download);
   - upload a GeoTIFF, and a plain PNG with the GSD card;
   - a generation (the desktop uses the ONNX sidecar, not the Space);
   - Darjeeling's edge, a flat tile, bright mode, and the new UI.
6. Ask me, then run `desktop/tauri/publish-release.sh` (publishes to `sancharimouri/depthwizard2-desktop`).
7. **Verify:**
   - `https://github.com/sancharimouri/depthwizard2-desktop/releases/latest/download/latest.json` says
     `"version": "1.0.2"`, and its signature matches the uploaded `.app.tar.gz`;
   - the deck link `…/releases/latest` shows v1.0.2 with the installer;
   - an installed 1.0.1 offers the update and relaunches as 1.0.2.

   Every asset must stay well under GitHub's 2 GB per-file limit.

## Phase 6: storage and GPU (report real numbers, fix what's safe)
1. **Hugging Face:**
   - Sizes of the private dataset `sancharimouri/depthwizard2-library-private` (`tiles/`, `previews/`, `thumbnails/`,
     `manifest.json`, and the `dem/` DFC2019 packs from `0270343`), the private model `sancharimouri/depthwizard2-method6`,
     and the Space.
   - Compare with the current free-tier limits (look them up; don't guess).
   - Confirm Render's `HF_TOKEN` can read `dem/`.
   - ZeroGPU: measured quota use per generation; is the daily quota enough for judging? If not, propose options with costs.
2. **GitHub releases:**
   - `depthwizard2-assets@library-v1`: total size, asset count, and whether the `manifest.json` URLs all resolve.
   - `depthwizard2-desktop`: total release size. Delete superseded test artifacts only with my OK, and never the release
     `latest` points to.
3. **Render (free):**
   - `/tmp` is ephemeral (`DW2_{CACHE,GENERATED,UPLOADS}_DIR`). Uploads and generated job folders are never cleaned
     within an instance's lifetime.
   - If needed, add a small, tested age/size cap.
   - Report the instance's memory headroom during a generation (Render metrics).
4. **Local disk:**
   - `du -sh` of `data/`, `external/`, `models/`, `node_modules`, the HF cache and `desktop/tauri/src-tauri/target`.
   - Make sure `.gitignore` covers the ~777 untracked entries that must never be committed.
   - List safe deletions. Delete nothing without my OK.
5. **Update `docs/STORAGE.md`** (tables, sizes, the `dem/` packs, the desktop release sizes).

## Phase 7: the final measured report (the deliverable of this session)
Report a table for each item below, with dates and methods. Measure; don't estimate.

1. **Desktop app size (v1.0.2):**
   - the `.dmg` installer;
   - the updater `.app.tar.gz`;
   - the installed `DepthWizard.app` on disk (`du -sh`), with its breakdown: frozen backend, bundled library, ONNX model,
     shell;
   - the per-user cache after one on-demand download;
   - compared against v1.0.1's numbers in `docs/DESKTOP_APP.md`.
2. **Website load times, cold start vs normal.**
   - **Cold** = the Render instance spun down. Wait ≥ 16 min with no requests, and confirm "Shutting down" in the Render
     logs.
   - **Normal** = right after a warm request.
   - Measure each **3 times** in a fresh browser profile (headless Chrome with a GPU, or the Chrome extension with the
     tab in front), and report the median and the range:
     - Vercel page: time to first byte and to DOMContentLoaded;
     - `/api/library` → 200 (the Library cards appear);
     - first thumbnail visible; all first-screen thumbnails visible;
     - selecting a tile → preview image loaded;
     - START GENERATION on Almora → all boxes done and the Studio expanded (and the depth request's round trip);
     - the Demo page (`#/demo`) → 3D terrain visible;
     - Home (`#/docs`) → rendered, and `#demo-video` scrolled into view;
     - the cold-start message appearing (it should, after 2.5 s).
   - Say where the time goes (Render spin-up, HF Space cold start, GitHub asset redirects, etc.) and what could reduce it
     within rule 3.
3. **SIH deck links:** each link, the final URL, the HTTP status, and what it shows.

## Phase 8: docs and wrap-up
- **Update:**
  - `docs/DEPLOY.md`: LIVE, and PENDING DEPLOY emptied;
  - `docs/DESKTOP_APP.md`: v1.0.2 and the sizes;
  - `docs/STORAGE.md`;
  - `docs/HANDOFF.md` §5z: mark deployed, with dates, commit hashes and evidence;
  - `docs/CHANGES_2026-09-28_29.md`: add a "Deployed" section;
  - `CLAUDE.md`: keep it short.
- **Commit** (rule 1), and push only by the Phase 0 route, with my OK.
- **Report per phase:** what was done, the evidence (URLs, status codes, versions, sizes, timings), anything skipped,
  and what still needs me.
