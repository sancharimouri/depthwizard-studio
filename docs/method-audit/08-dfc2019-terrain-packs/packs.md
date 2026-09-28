# Prompt 5 — reference correction, packs, integration, QA

Script: `scripts/dfc2019_build_packs.py` (build, upload). Backend: `backend/storage/library_store.private_pack`,
`backend/generation/pipeline._library_elevation` / `generate`.

## Pre-registered correction rule (committed before any pack is built)
- **Inputs:**
  - Prompt 4's held-out Method 6 heights P (1024², ≥ 0);
  - the tile's DFC2019 AGL A (1024²);
  - Prompt 4's measured **cap = 12.0 m** and **margin = 3.0 m** (`heights/cap.json`, rule in `heights.md`).
- **Valid AGL:** finite, ≥ −5 m and < 1000 m (no sentinels exist in this data). AGL in [−5, 0) is
  ground noise and is set to 0. Invalid pixels are never used. JAX_004_016 has 1 NaN; it
  keeps P.
- **Tall zone:** valid pixels where A > cap **or** P > cap.
- **Replacement:** inside the tall zone only, where |P − A| > margin, the height becomes A.
  Outside the tall zone, P is untouched.
- **Feather:** replacement weight w = max(mask, clip(2·G_σ=2px(mask), 0, 1)), so w = 1 on
  replaced pixels and falls off over a few pixels around them; w = 0 on invalid AGL. Final height
  = (1 − w)·P + w·A. The feather ring can extend a few pixels outside the tall zone; that is its
  purpose (no cliffs).
- **Tiles with little or no valid AGL** (< 50 % valid) get no correction. Prompt 1 found **none**:
  every tile is ≥ 99.9999 % valid.
- **Reported per tile:** fraction of pixels replaced (mask) and touched (w > 0).

## Pack assembly (decisions)
- **Elevation:** final elevation = base ground (Prompt 3, already on the 341² pack grid in the tile's own
  pixel frame) + corrected height, block-meaned 1024 → 341. `_mesh_hw` gives 341 for a
  1024 px tile. The block mean drops the last pixel row/column (1023 = 3 × 341), a 0.1 % crop.
- **Format:** `build_dem_pack.write` — 2-band float32 GeoTIFF, band 1 `TERRAIN` = base ground,
  band 2 `SURFACE` = final elevation; NaN nodata; deflate.
- **Frame:** synthetic local metric frame `+proj=tmerc +lat_0=0 +lon_0=0 …`, bounds (0, 0, S, S) m with
  S = 1024 × GSD. **GSD:** a located tile uses its own matched GSD. A city-level tile uses the median
  matched GSD of that city's located tiles, or 0.3 m if a city has none. This follows the brief's
  "~0.3 m" with the measured value in place of the nominal one.
- **Tags:**
  - `TERRAIN_SOURCE`: "Public DEM: USGS 3DEP bare earth" or "Approximate city-level ground elevation (USGS 3DEP city median)".
  - `SURFACE_SOURCE`: "Model-predicted heights (Method 6), with reference lidar correction for tall structures and trees".
  - `CRS_LABEL`: "local frame (no georeference)".
  - `RAMP_BAND`: `SURFACE`, because TERRAIN is a constant on city-level tiles and `elevation.png` would be one colour.
  - `NOTE`: provenance; city-level tiles say "Elevations are approximate…".
  - `HOW`: "curated elevation pack".
  - **No error or accuracy number anywhere.**
- **Storage:** `dem/dfc2019-<tile>.tif` in the **private** HF dataset (the upload asserts it is private).
  Nothing goes to a public release or repo, including for the 42 + 8 tiles whose RGB is already public.
- **Integration:** remote mode (web) fetches the pack via `library_store.private_pack`, server-side. Local
  mode reads `data/library/dem/` (gitignored). Bundle mode (the desktop app) is untouched. Uploads
  and other sources are unchanged. No location enters the manifest or the catalog.

## Results
*(filled in after the run; the sections above are not edited)*
