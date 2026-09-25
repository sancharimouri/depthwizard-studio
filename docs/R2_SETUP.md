# Cloudflare R2 setup (tile library + Method 6 checkpoints)

Everything the app serves from the curated library, and the Method 6 checkpoints the
VHR pipeline loads, live in one R2 bucket. With R2 configured the backend reads the
catalog, thumbnails, previews and tile links **only** from R2; without it, it falls back
to the local `data/library/` files (local development).

Code: `backend/storage/r2.py` (keys, URLs, downloads), `backend/library/catalog.py`,
`backend/api/library_routes.py`, `scripts/r2_sync.py` (upload), `scripts/library_catalog.py`
(writes each item's R2 keys), `scripts/vhr_dsm_pipeline.py` (checkpoint fallback).

## What's in the bucket (measured with `scripts/r2_sync.py --dry-run`, 2026-09-25)

| Prefix | Objects | Bytes |
|---|---|---|
| `library/tiles/dfc2019/` | 50 | 127,617,513 |
| `library/tiles/sentinel2/` | 32 | 71,335,822 |
| `library/tiles/vhr/` | 6 | 56,473,616 |
| `library/thumbnails/` | 88 | 1,495,440 |
| `library/previews/` | 88 | 22,558,440 |
| `checkpoints/method6/` | 5 | 496,193,972 |
| `library/manifest.json` | 1 | ~98,000 |
| **Total** | **270** | **775,772,809 (0.776 GB, 7.8 % of the 10 GB free tier)** |

Checkpoints: the five files code actually loads. The product itself is DEM-only and loads
no Method 6 checkpoint (`docs/HANDOFF.md`: "Product model: none adopted"). The only
Method 6 consumer, `scripts/vhr_dsm_pipeline.py`, uses the mean of the four seed-43
height-balanced fold checkpoints plus the full-data checkpoint as a cross-check:
`checkpoints/method6/height_balanced_seed43/fold{0..3}.pt` and
`checkpoints/method6/full_dfc2019/method6_full_dfc2019.pt`.

## 1. Create the bucket (Cloudflare dashboard)

1. Cloudflare dashboard → **R2 Object Storage** → enable R2 (free tier; a payment method may be
   asked for, the free tier itself is not charged).
2. **Create bucket** → name `depthwizard2` (or set `R2_BUCKET` to your name) → location Automatic.
3. Keep it **private**. Do *not* enable the public `r2.dev` URL unless you have decided the
   imagery may be redistributed publicly: the DFC2019 tiles come with the IEEE GRSS DFC2019
   terms, and Maxar Open Data is CC BY-NC 4.0. Private is the default here — the backend
   hands the browser short-lived presigned URLs (`R2_URL_TTL`, default 3600 s).

## 2. Create an API token

R2 → **Manage R2 API Tokens** → **Create API token**:
- Permissions: **Object Read & Write**, scoped to the `depthwizard2` bucket only.
  (Use **Admin Read & Write** only if you want `r2_sync.py --create-bucket` to create it.)
- Copy the **Access Key ID** and **Secret Access Key** (shown once) and your **Account ID**
  (R2 overview page, right-hand side).

## 3. Put the values in `.env` (never in a committed file)

`.env` is git-ignored. Add (names are listed, empty, in `.env.example`):

```
R2_ACCOUNT_ID=<account id>
R2_ACCESS_KEY_ID=<access key id>
R2_SECRET_ACCESS_KEY=<secret access key>
R2_BUCKET=depthwizard2
# R2_PUBLIC_BASE_URL=   # leave unset for a private bucket
```

The backend loads `.env` at start (`backend/main.py`). For the upload script, export them in
the shell or run it with `uv run --env-file .env`.

## 4. Upload

```
uv run python scripts/r2_sync.py --dry-run                      # sizes only, no network
uv run --env-file .env --with boto3 python scripts/r2_sync.py   # upload (re-runnable)
```

It skips objects whose size and sha256 already match, uploads the manifest last, writes
`data/library/r2_inventory.json` (keys, sizes, sha256; git-ignored), and prints the bucket's
real total from a listing.

## 5. Check

Restart the backend, then `GET /api/library` → `thumbnail_url`, `preview_url`, `tile_url`
point at `<account>.r2.cloudflarestorage.com/...` (presigned) — or your public base URL.
