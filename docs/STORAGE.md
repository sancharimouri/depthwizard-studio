# Storage: where the non-code assets live (updated 2026-09-29)

Split by licence. Active code: `backend/storage/library_store.py` (tile library) and
`backend/storage/hf_checkpoints.py` (Method 6 checkpoints). Publisher: `scripts/publish_library.py`.
`backend/storage/r2.py` (Cloudflare R2) is kept but **dormant**. Nothing calls it and nothing was uploaded there.

Measured 2026-09-29 (HF API tree listing, `gh release view`, `du`).

| Asset | Where | Access | Files | Size |
|---|---|---|---|---|
| Sentinel-2 + Maxar library, 40 DFC2019 on-demand tiles (+ previews) | GitHub Release `sancharimouri/depthwizard2-assets@library-v1` | **public** | 201 | 228 MB (largest file 11 MB) |
| DFC2019 library: `tiles/` | HF dataset `sancharimouri/depthwizard2-library-private` | **private** | 50 | 127.6 MB |
| DFC2019 terrain packs: `dem/` (series from `7c8cbec`, now `0270343`) | same dataset | private, backend only | 50 | 21.4 MB |
| DFC2019 `previews/`, `thumbnails/` | same dataset | private, proxied by the backend | 50 + 50 | 12.1 + 0.9 MB |
| `manifest.json` (89 items) + README + .gitattributes | same dataset | private | 3 | 0.1 MB |
| **HF library dataset total** | | | **203** | **162.1 MB** (162,102,670 B) |
| Method 6 checkpoints (4 folds, seed 43, + full-DFC2019) | HF model `sancharimouri/depthwizard2-method6` | **private** | 5 + README | **496.2 MB** |
| DAv2 inference Space | HF Space `sancharimouri/DepthWizard2` (ZeroGPU) | public | 4 | 6 KB (code only) |
| Desktop releases | `sancharimouri/depthwizard2-desktop` | public | v1.0.2 (latest): 4 assets, 699 MB; v1.0.1: 669 MB; v1.0.0: 621 MB | 1,989 MB in total |

**Free-tier limits (looked up 2026-09-29, huggingface.co/docs/hub/storage-limits and spaces-zerogpu):**
- HF private storage: **100 GB** per free account. In use: 0.66 GB (dataset 162.1 MB + model 496.2 MB).
- HF public storage: "best-effort". The Space is 6 KB.
- ZeroGPU: a free account may host up to 2 ZeroGPU Spaces (we use 1), and gets **5 min of GPU a day** as a caller
  (PRO: 40 min). Measured 2026-09-26: ~585 depth calls before the first quota refusal, so ~585 generations a day
  (the backend calls the Space with the owner's token). A judging session of a few dozen generations fits easily.
- ZeroGPU supports Python 3.12.12 and 3.10.13 only: the Space's README must keep `python_version: '3.12'`
  (a `3.11` build fell back to 3.10 and failed on `numpy==2.4.6`, 2026-09-29).
- GitHub: every release asset is far below the 2 GB per-file limit (largest: the 1.0.2 DMG, 355.7 MB).
- Verified 2026-09-29: Render's `HF_TOKEN` reads `dem/` (live DFC2019 generation → "curated elevation pack"); all 178
  thumbnail/preview URLs of `/api/library` resolve (78 via 307 to GitHub, 100 from the backend).

**Render (free, 512 MB):** peak backend RSS 192 MB over 15 generations (clean venv from `requirements.txt`,
measured locally; the Render dashboard metrics were not read). Each generation writes ~2.7 MB (max 3.4 MB) under `/tmp`,
which the instance loses on every spin-down (15 min idle) and deploy; at the ZeroGPU ceiling (~585 a day) that is
≤1.6 GB, so no age/size cap was added.

**Local disk (2026-09-29, `du -sh`):** repo 37 GB: `data/` 28 GB (`dfc2019` 19 GB, `sentinel2_benchmark` 3.0 GB,
`vhr_dsm` 1.6 GB), `models/` 1.3 GB, `desktop/tauri/src-tauri/target` 4.5 GB, `frontend/node_modules` 392 MB,
root `node_modules` 86 MB, `external/` 51 MB; HF cache `~/.cache/huggingface` 1.4 GB.
- Rebuildable/downloadable (safe to delete, owner's OK first): `target/` (4.5 GB; `cargo clean`), the HF cache and
  `models/hub` (re-downloaded on demand), `node_modules` (`npm install`).
- Not safe without the owner: `data/` research outputs; most checkpoints there exist nowhere else.
- `.gitignore` (2026-09-29) keeps checkpoints, bulk arrays/rasters, DFC2019-derived files, cloned third-party repos and
  personal documents out of git: untracked went from 3,355 files (~21 GB) to ~300 (~210 MB, the owner's call).

The GitHub repo holds only these release assets and a README. The application code is not published there.

## Curated library v2 (local only, 2026-09-29; `docs/library_v2.md`)

`data/library_v2_2026-09-29/` (190 MB, gitignored; branch `curated-tiles-2026-09-29`, not pushed) holds the curated set
for the private backend container.

- **`tile_manifest.json`:** the single source of truth for inclusion, order and default exaggeration. It is read by
  `backend/library/tile_manifest.py` (only when `DW2_TILE_MANIFEST` is set) and by `scripts/stage_container_tiles.py`.
- **`dem/`:** the elevation packs the app reads (89).
  - Sentinel-2: copies of `data/library/dem/`; 3 are offset-only recalculated.
  - DFC2019: copies of `data/dfc2019/terrain_packs/packs/`.
  - Maxar: 3-band packs with a cosmetic **DISPLAY** band. TERRAIN and SURFACE are unchanged, and DISPLAY shapes the
    mesh only.
- **`dem_maxar/{subtle,medium,strong}/`:** the DISPLAY presets. The active one is copied into `dem/` and named in
  `dem_maxar/ACTIVE`.
- **`previews/`, `thumbnails/`:** Sentinel-2 in the approved B_own_percentiles colours; Darjeeling's variants are in
  `_variants/`. Maxar and DFC2019 are unchanged copies.
- **Other files:**
  - `manifest.json` + `tiles/` (symlinks to the original rasters) are the bundle-mode catalog;
  - `_analysis/` holds tables (the DFC2019 per-tile tables stay here, never in git);
  - `_guard/` holds the SHA-256 baseline.
- **Baked depth (2026-09-30):** each included pack also holds the DAv2 relative depth of its exact preview
  (metadata domain `DAV2_DEPTH`, u16+zlib as the Space returns it; about 0.9 MB per entry). Library tiles need no Space
  call. DFC2019 depth is private, like the rest of those packs.
- **Container staging** (dry run, `scripts/stage_container_tiles.py`): 81 packs (41.0 MB) + 162 images (27.7 MB).
  - DFC2019 content may go into the **private** container only.
  - Nothing here is uploaded to HF or GitHub yet: the HF `dem/` and the public release still hold the previous sets.

## Why DFC2019 is private

The IEEE GRSS DFC2019 contest terms forbid redistributing the data. This was checked against the primary source, not just a search result.
So DFC2019 tiles and derived previews/thumbnails are never public. The browser gets only this backend's
`/api/library/<id>/{thumbnail,preview}` routes. The backend fetches the image with `HF_TOKEN`, and the item gets **no** `tile_url`.
Sentinel-2 (Copernicus licence, all 2025 acquisitions) and Maxar Open Data are public with attribution.

## Runtime behaviour

- Default mode is `remote`. The manifest is read from the private dataset and re-checked every `DW2_MANIFEST_TTL` s (300).
  - Public items get direct GitHub URLs; the backend routes for them 307-redirect to the same URLs.
  - GitHub serves the assets as `application/octet-stream`. `<img>` and CSS backgrounds render them fine, which was verified in Chrome.
  - Private items are served from the HF cache.
- If `HF_TOKEN` is missing or wrong, `/api/library` returns 503. There is **no** silent fallback to local files.
- `DW2_LIBRARY=local` → `data/library/` (development only).
- Environment overrides: `DW2_ASSETS_REPO`, `DW2_ASSETS_TAG`, `DW2_LIBRARY_DATASET`, `DW2_CKPT_REPO`.
- DFC2019 image URLs are relative (`/api/...`); the deployed frontend reaches the backend through `VITE_API_BASE`, and CORS
  allows exactly `https://depthwizard-studio.vercel.app` and `https://depthwizard2.vercel.app` (verified 2026-09-29).

## Cost

No card or billing setup anywhere:
- The GitHub releases are on a public repo, which is free.
- The HF private repos are on a free account (`isPro: false`, `canPay: false`, i.e. no payment method on file).
- R2, the only option that needs a card, was never activated.

## Verification (2026-09-26)

- **Headless Chrome, live UI (Library tab):**
  - 88/88 thumbnails render: Sentinel-2 32, Maxar 6, DFC2019 50.
  - Preview (1024²) and magnifier lens work for a GitHub item (`sentinel2-almora`, `vhr-a_forest`) and a proxied item (`dfc2019-JAX_004_006`).
  - 0 HTTP errors.
- **Production-mode backend:**
  - Setup: an isolated copy of `backend/` with no `data/` and no `.env`, `HF_TOKEN` as an env var only, an empty HF cache, and no `--reload`.
  - All 176 thumbnail + preview URLs return 200. Every public route 307s to GitHub.
  - No DFC2019 item carries a GitHub URL or a tile URL.
  - The cache filled from HF (100 JPEGs + manifest).
  - Without a token: 503, and anonymous HF → 401.
- **Tests:**
  - Frontend 40/40.
  - Backend 22 passed, 1 skipped. The skip is the dormant R2 boto3 cross-check; boto3 isn't installed.
