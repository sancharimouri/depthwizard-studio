# Storage: where the non-code assets live (2026-09-26)

Split by licence. Active code: `backend/storage/library_store.py` (tile library) and
`backend/storage/hf_checkpoints.py` (Method 6 checkpoints). Publisher: `scripts/publish_library.py`.
`backend/storage/r2.py` (Cloudflare R2) is kept but **dormant**. Nothing calls it and nothing was uploaded there.

| Asset | Where | Access | Files | Size |
|---|---|---|---|---|
| Sentinel-2 library (32 scenes: tile + thumb + preview) | GitHub Release `sancharimouri/depthwizard2-assets@library-v1` | **public**, attribution README | 96 | 80.1 MB |
| Maxar Open Data VHR library (6 crops from 3 distinct 2022 scenes) | same release | **public**, CC BY-NC 4.0 attribution | 18 | 58.7 MB |
| **GitHub release total** | | | **114** | **138.9 MB** (138,858,442 B) |
| DFC2019 library (50 tiles) | HF dataset `sancharimouri/depthwizard2-library-private` → `tiles/` | **private** (anonymous → 401) | 50 | 127.6 MB |
| DFC2019 previews / thumbnails | same dataset → `previews/`, `thumbnails/` | private, proxied by the backend | 50 + 50 | 12.1 + 0.9 MB |
| `manifest.json` (all 88 items) + README | same dataset | private | 2 | 0.1 MB |
| **HF library dataset total** | | | **152** | **140.7 MB** (140,707,646 B) |
| Method 6 checkpoints (4 folds, seed 43, + full-DFC2019) | HF model `sancharimouri/depthwizard2-method6` | **private** | 5 + README | **496.2 MB** (496,194,811 B) |

The GitHub repo holds only these release assets and a README. The application code is not published there.

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
- **Deployment note (Prompt 2):** DFC2019 image URLs are relative (`/api/...`). The deployed frontend must reach the backend at the same origin
  (or through a proxy), and CORS in `backend/main.py` currently allows only `http://localhost:5173`.

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
