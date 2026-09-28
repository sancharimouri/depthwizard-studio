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

Built, uploaded and QA'd 2026-09-28.

### Correction (pre-registered rule, cap 12 m, margin 3 m)
- **All 50 tiles corrected;** none has little or no valid AGL.
- **Fraction of pixels replaced:** median **5.1 %**, max **30.0 %** (JAX_214_023). Next most corrected: JAX_214_015 27.6 %, JAX_165_015 25.0 %, JAX_166_006 24.5 %, JAX_164_008 23.0 %.
- **Fraction touched including the feather ring:** median 13.5 %, max 44.1 %.

### Packs
- **Format:** 50 packs, 341² two-band float32, 21 MB in total, in the local tmerc frame.
- **GSD:**
  - 15 located tiles use their own matched GSD (0.308–0.353 m).
  - City-level tiles use the city median: JAX 0.325, OMA 0.323.
- **Checks passed by all 50:** finite values, SURFACE ≥ TERRAIN, bounds at the origin, RAMP_BAND=SURFACE, no accuracy wording in the tags.
- **Upload:** to the **private** `sancharimouri/depthwizard2-library-private` under `dem/` (50 files; `private: true` re-checked after upload). Nothing public.

### Integration
- **Remote mode (web):** `library_store.private_pack(item)` fetches `dem/<id>.tif` through the same HF cache and token as the tiles. It returns None for public items, in local/bundle mode, and when there is no pack.
- **`pipeline._library_elevation`:** uses the pack when there is no local one.
- **`generate()`:** honours the optional tags `CRS_LABEL`, `RAMP_BAND`, `NOTE`, `HOW`. Other packs are unchanged: default ramp TERRAIN, CRS from the file.
- **Unchanged:** uploads, Sentinel-2, Maxar, Darjeeling, the desktop app (bundle mode) and the manifest/catalog. No location enters them.
- **Tests:** 2 new tests (DFC2019 pack via `private_pack`; `private_pack` is remote-only and never serves public items). The "no georeference" test now stubs `private_pack`.

### QA
- **Web path:** `scripts/dfc2019_packs_qa.py` ran against a local backend in remote mode (packs from the private HF dataset, DAv2 on the ZeroGPU Space). **50/50 DFC2019 jobs** pass every check:
  - `has_elevation`, the curated-pack provenance, the local-frame CRS label and the note;
  - a 341×341 mesh with relief (6.0–136.7 m);
  - a non-flat elevation.png;
  - no accuracy wording in meta.
- **Screenshots** (browser, dev server; in `data/dfc2019/terrain_packs/qa_screens/`, gitignored because they show DFC2019 imagery):

| # | Tile | Kind | Rendered |
|---|---|---|---|
| 1 | JAX_004_006 | located JAX | ×24 auto-exaggeration |
| 2 | OMA_084_038 | located OMA | ground 299–349 m on a slope, a tower |
| 3 | JAX_214_023 | located, most corrected (30 %) | downtown block, 97 m tower |
| 4 | OMA_281_002 | city-level OMA | shows the artifact below |
| 5 | JAX_166_006 | city-level JAX | downtown, 143 m max |
| 6 | OMA_144_030 | city-level, flat | car park |

- **Test suites:** backend **41 passed, 1 skipped** (the dormant R2 boto3 check). Frontend **53/53**.

### Per-tile table
| Tile | Location | Base ground | Replaced | Touched (feather) | Seam | 3D (web path) | Follow-up |
|---|---|---|---|---|---|---|---|
| JAX_004_006 | confident match | 3DEP 1 m (located) | 6.8 % | 22.9 % | 0.79 | yes, relief 31.8 m |  |
| JAX_004_014 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 9.7 % | 26.1 % | 0.78 | yes, relief 31.2 m | candidate for v2 location rule |
| JAX_004_016 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 8.6 % | 26.0 % | 0.93 | yes, relief 29.4 m | candidate for v2 location rule |
| JAX_018_012 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 4.3 % | 15.1 % | 0.93 | yes, relief 25.0 m | candidate for v2 location rule |
| JAX_022_009 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 6.4 % | 24.1 % | 0.47 | yes, relief 22.9 m | candidate for v2 location rule |
| JAX_031_006 | confident match | 3DEP 1 m (located) | 12.1 % | 31.8 % | 0.71 | yes, relief 27.3 m |  |
| JAX_072_015 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 5.3 % | 15.7 % | 0.95 | yes, relief 23.9 m | candidate for v2 location rule |
| JAX_118_012 | confident match | 3DEP 1 m (located) | 3.3 % | 10.4 % | 1.15 | yes, relief 24.7 m |  |
| JAX_118_015 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 4.5 % | 11.9 % | 1.01 | yes, relief 21.4 m | candidate for v2 location rule |
| JAX_149_006 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 17.8 % | 38.2 % | 0.57 | yes, relief 25.8 m | candidate for v2 location rule |
| JAX_149_025 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 17.9 % | 39.5 % | 0.58 | yes, relief 27.7 m | candidate for v2 location rule |
| JAX_161_001 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 2.3 % | 7.4 % | 0.96 | yes, relief 40.6 m | candidate for v2 location rule |
| JAX_164_008 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 23.0 % | 33.4 % | 1.18 | yes, relief 109.5 m | candidate for v2 location rule |
| JAX_165_015 | confident match | 3DEP 1 m (located) | 25.0 % | 44.0 % | 1.96 | yes, relief 83.6 m | seam 1.96 (real structure on the 512 line) |
| JAX_166_006 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 24.5 % | 35.0 % | 0.92 | yes, relief 136.7 m | candidate for v2 location rule |
| JAX_175_002 | confident match | 3DEP 1 m (located) | 9.9 % | 18.1 % | 0.81 | yes, relief 59.8 m |  |
| JAX_204_005 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 4.8 % | 13.0 % | 0.65 | yes, relief 45.6 m | candidate for v2 location rule |
| JAX_214_015 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 27.6 % | 40.0 % | 1.33 | yes, relief 93.1 m | candidate for v2 location rule |
| JAX_214_023 | confident match | 3DEP 1 m (located) | 30.0 % | 44.0 % | 1.31 | yes, relief 97.8 m |  |
| JAX_224_025 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 6.0 % | 12.2 % | 0.72 | yes, relief 71.8 m | candidate for v2 location rule |
| JAX_264_013 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 7.1 % | 14.0 % | 0.74 | yes, relief 39.9 m | candidate for v2 location rule |
| JAX_269_009 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 7.9 % | 22.4 % | 0.89 | yes, relief 33.4 m | candidate for v2 location rule |
| JAX_416_009 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 2.9 % | 10.6 % | 1.06 | yes, relief 23.3 m | candidate for v2 location rule |
| JAX_416_022 | confident match | 3DEP 1 m (located) | 1.8 % | 7.1 % | 1.04 | yes, relief 24.5 m |  |
| JAX_505_016 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 16.6 % | 38.7 % | 0.91 | yes, relief 34.5 m | candidate for v2 location rule |
| JAX_505_018 | confident match | 3DEP 1 m (located) | 15.0 % | 36.3 % | 0.96 | yes, relief 36.3 m |  |
| OMA_042_011 | unmatched (C1, C3; index-consistent post hoc) | city-level (approx.) | 7.7 % | 28.2 % | 1.05 | yes, relief 22.5 m | candidate for v2 location rule |
| OMA_084_038 | confident match | 3DEP 1 m (located) | 7.0 % | 31.2 % | 1.17 | yes, relief 50.1 m |  |
| OMA_134_027 | confident match | 3DEP 1 m (located) | 1.6 % | 11.9 % | 0.82 | yes, relief 29.2 m |  |
| OMA_144_030 | unmatched (C1, C2, C3) | city-level (approx.) | 0.0 % | 0.0 % | 1.33 | yes, relief 6.0 m | flat car park; one model bump of ~6 m (below cap, left as predicted); nothing to locate on |
| OMA_198_002 | unmatched (C1; index-consistent post hoc) | city-level (approx.) | 5.3 % | 30.7 % | 0.91 | yes, relief 23.6 m | candidate for v2 location rule |
| OMA_211_032 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 1.0 % | 6.3 % | 1.01 | yes, relief 29.2 m | candidate for v2 location rule |
| OMA_211_039 | unmatched (C1, C2, C3; index-consistent post hoc) | city-level (approx.) | 1.9 % | 8.4 % | 1.36 | yes, relief 28.3 m | candidate for v2 location rule |
| OMA_212_033 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 1.1 % | 5.8 % | 1.10 | yes, relief 21.2 m | candidate for v2 location rule |
| OMA_221_034 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 9.0 % | 31.3 % | 0.73 | yes, relief 29.6 m | candidate for v2 location rule |
| OMA_225_001 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 2.0 % | 10.2 % | 1.35 | yes, relief 19.1 m | candidate for v2 location rule |
| OMA_230_036 | confident match | 3DEP 1 m (located) | 0.7 % | 6.1 % | 0.65 | yes, relief 38.4 m |  |
| OMA_248_029 | unmatched (C2; index-consistent post hoc) | city-level (approx.) | 8.9 % | 17.4 % | 1.24 | yes, relief 38.4 m | candidate for v2 location rule |
| OMA_248_030 | unmatched (C2; index-consistent post hoc) | city-level (approx.) | 8.7 % | 16.4 % | 1.96 | yes, relief 38.4 m | seam 1.96 (real structure on the 512 line) |
| OMA_258_020 | unmatched (C1, C2, C3; index-consistent post hoc) | city-level (approx.) | 1.4 % | 5.0 % | 0.88 | yes, relief 33.6 m | candidate for v2 location rule |
| OMA_269_035 | confident match | 3DEP 1 m (located) | 0.8 % | 7.0 % | 0.98 | yes, relief 23.6 m |  |
| OMA_281_002 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 4.7 % | 9.3 % | 1.17 | yes, relief 122.4 m | reference AGL has a ~120 m unclassified (class 65) blob, copied in by the correction: likely lidar artifact |
| OMA_281_030 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 4.3 % | 9.6 % | 1.10 | yes, relief 119.1 m | same unclassified ~120 m blob as OMA_281_002 |
| OMA_315_019 | unmatched (C3; index-consistent post hoc) | city-level (approx.) | 2.7 % | 8.6 % | 1.03 | yes, relief 20.6 m | candidate for v2 location rule |
| OMA_315_020 | confident match | 3DEP 1 m (located) | 1.1 % | 6.2 % | 0.71 | yes, relief 20.3 m |  |
| OMA_332_037 | confident match | 3DEP 1 m (located) | 2.1 % | 12.0 % | 0.88 | yes, relief 59.3 m |  |
| OMA_364_003 | confident match | 3DEP 1 m (located) | 2.0 % | 11.5 % | 1.24 | yes, relief 45.3 m |  |
| OMA_364_043 | unmatched (C2, C3; index-consistent post hoc) | city-level (approx.) | 1.6 % | 11.3 % | 1.37 | yes, relief 25.0 m | candidate for v2 location rule |
| OMA_376_023 | unmatched (C2; index-consistent post hoc) | city-level (approx.) | 1.9 % | 3.4 % | 0.70 | yes, relief 28.9 m | candidate for v2 location rule |
| OMA_376_038 | unmatched (C2; index-consistent post hoc) | city-level (approx.) | 1.7 % | 3.4 % | 0.55 | yes, relief 28.9 m | candidate for v2 location rule |

### What is real, approximate and still placeholder
- **Real:**
  - Heights: held-out Method 6 predictions, with DFC2019 reference lidar (AGL) replacing them in the tall zone, per the pre-registered rule.
  - Ground on the 15 located tiles: USGS 3DEP 1 m, NAVD88, datum-checked.
  - The tile's horizontal scale: matched GSD.
- **Approximate (and labelled so in the UI):**
  - The ground on the 35 city-level tiles is one city-median value (JAX 6.0 m, OMA 299.9 m). Absolute elevations there can be off by the city's spread (p5–p95: JAX −0.0 to 9.4 m, OMA 295.6 to 362.8 m), and there is no ground slope within the tile.
  - The viewer's auto-exaggeration on low-relief tiles shows about 1.6× true relief (JAX_004_006). Scale on flat tiles is up to ×60 (OMA_144_030).
- **Still placeholder or stale (follow-ups; not changed in this series):**
  1. **Unclassified tall blobs in the reference AGL** get copied in. OMA_281_002 and OMA_281_030 show a ~120 m class-65 blob, most likely a lidar artifact.
     - Class 65 is common on real building edges (39 tiles have some among replaced pixels), so blanket exclusion is wrong.
     - A connected-component rule needs its own pre-registration.
  2. **34 unmatched tiles are index-consistent** (`locate.md`). A pre-registered v2 location rule could move them from city-level to 3DEP ground.
  3. **Stale manifest text.** The DFC2019 manifest routing summary still says "no DEM is added: the result is relative above-ground height", and the log line says "GSD: 30 cm". Both come from the manifest, which was deliberately not touched.
  4. **Elevation box caption mismatch.** The caption shows the TERRAIN range, while for DFC2019 the image is ramped from SURFACE.
  5. **Seam ratios:** JAX_165_015 / OMA_248_030 at 1.96, from real structure (`heights.md`).
  6. **Desktop app:** bundled DFC2019 tiles still render flat (bundle mode was out of scope).
