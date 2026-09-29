# Curated tile library v2 (2026-09-29, branch `curated-tiles-2026-09-29`, local only)

> **Update 2026-09-30** (owner re-ratings and the DFC2019 fix):
> - **Sentinel-2 re-rated** (all now 3 stars, `user` defaults):
>   - Bhitarkanika 30x, Kutch 90x (= its slider maximum), Amalapuram 43x.
>   - Amalapuram moves into the 3-star tier.
> - **Maxar ratings:**
>   - a_valley 10x, c_town 4x, c_river 11x; a_forest and c_terraces use auto; all rated 3 stars.
>   - **b_glacier is excluded.**
>   - These values are in the slider units of the active DISPLAY preset (`subtle`). Switching the preset changes what
>     they look like.
> - **The DFC2019 flat-render bug is fixed:**
>   - `backend/generation/pipeline.py` writes `limitOutliers: false` into DFC2019 terrain.json, and
>     `frontend/src/terrain.js` then skips the outlier limiter.
>   - The six tiles show their full relief. Verified in the viewer: OMA_211_039, OMA_211_032, OMA_212_033, OMA_376_023,
>     OMA_376_038, OMA_258_020 (`_qa/screenshots_2026-09-30/`).
>   - It applies to all DFC2019 tiles, so every DFC2019 tile now shows its full relief. The anchors' 10–16x values
>     were chosen while the limiter was capping them (e.g. OMA_269_035 showed 46 % of its relief, JAX_004_006 60 %),
>     so they now look taller at the same value.
>   - Sentinel-2, Maxar and the demo regions keep the limiter, so their rated looks are unchanged.
> - **Darjeeling:** the generated B render is **pixel-identical** to the original. Darjeeling's RGB raster is already
>   stretched to 0–255 (per-band 2nd/98th percentiles = 0/255), so the B stretch is the identity. There is no colour
>   choice to make. The side-by-side is in `_qa/compare/darjeeling_A_original_vs_B.png`.

A curated tile set for the private backend container, built from the owner's visual-QA ratings. **Nothing is pushed or
deployed.** Everything is non-destructive:

- all outputs live in `data/library_v2_2026-09-29/` (gitignored: it holds DFC2019 packs and previews);
- the source folders were hashed before and after.

**Checksum result:** 474 files in `data/library/`, the 34 Sentinel-2 RGBs, `data/display_test_2026-09-29/`, the DFC2019
packs and their backup are **identical** (SHA-256, before vs after).

## The manifest: one source of truth

`data/library_v2_2026-09-29/tile_manifest.json`, one entry per tile (89 entries). Fields:

| field | meaning |
|---|---|
| `tile_id`, `display_name`, `source` | catalog id (e.g. `sentinel2-kochi_city`), title, `sentinel2` / `maxar` / `dfc2019` |
| `stars` | the owner's rating (Sentinel-2 only for now) |
| `include_in_container` | listed and staged; excluded tiles keep their files |
| `order_index` | global listing order |
| `default_exaggeration` | slider value applied after generation; `null` = the app's auto value |
| `exaggeration_origin` | `user` / `baked_from_current_auto` / `auto_after_recalc` / `derived_rule` |
| `calibration_recipe`, `notes` | how the pack was made; flags |
| `needs_rerating`, `texture_variant` | added: re-rating flag; Darjeeling's colour choice (`original` / `B_own_percentiles`) |

Readers of the manifest:

- **Backend listing:** `backend/library/tile_manifest.py`, enabled by `DW2_TILE_MANIFEST`. It is unset in production,
  so the listing is unchanged there.
- **Generate response:** `meta.default_exaggeration`, which `frontend/src/main.js` `applyDefaultExaggeration` applies.
- **Container staging:** `scripts/stage_container_tiles.py`.

To re-rate a tile, edit `stars`, `include_in_container` or `default_exaggeration` in the manifest. The order is
recomputed by `scripts/build_library_v2.py dfc` (which calls `_order`), or you can set `order_index` by hand.

## Build (`scripts/build_library_v2.py`, steps: seed, packs, images, catalog, dfc, exag)

- **seed:** reads `_input_ratings_2026-09-29.json`, the ratings as given, with names mapped.
  - The name mapping had no ambiguity: Kochi → `kochi_city`, Mandovi → `goa_estuary`, Bathindra → `bathinda`,
    Dharmashala → `dharamshala`, OMA_376_23 → `OMA_376_023`.
  - All 33 Sentinel-2 tiles were rated; none were missed.
- **packs:**
  - Sentinel-2 packs are byte-identical copies, except the 3 recalculated ones.
  - DFC2019 packs are copied from `data/dfc2019/terrain_packs/packs/`.
  - Maxar packs come from the active DISPLAY preset.
- **images:** Sentinel-2 previews and thumbnails are the approved B_own_percentiles renders.
  - The stretch is re-implemented and checked bit-exact against B_all_32 (Bathinda, max |diff| 0).
  - It is a per-band 2–98 % stretch on the full-resolution raster, applied to the 1024 px preview.
  - Why the originals look muted: `scripts/library_catalog.py` `_stretch_rgb` passes uint8 rasters through unstretched,
    and every Sentinel-2 RGB is uint8.
- **catalog:** the app's bundle-mode `manifest.json`, plus `tiles/` symlinks to the original rasters.
- **dfc:** the DFC2019 rules below.
- **exag:** bakes the auto values; every default lies inside its tile's slider range.

## Sentinel-2

- **Excluded** (1 star, not RECALCULATE; files stay): Kurnool, Delhi, Bengaluru, Fatehpur, Digha, Nagapattinam, Jaipur.
- **Order:**
  - 3 stars: Darjeeling, Almora, Manali, Kakinada, Bhitarkanika, Kutch, Kochi, Nainital, Dehradun, Kohima, Shimla, Hisar.
  - 2 stars: Vidisha, Kota, Vembanad, Dharamshala, Ooty, Mandovi, Mumbai, Hyderabad, Chennai, Nizamabad, Bathinda, Pune,
    Karnal.
  - Then Amalapuram (1 star, pending re-rating).
- **Offset-only recalculation** (`calibration_recipe = offset_only_b1`):
  - The fit is a = median(h_ICESat-2 − FABDEM), with b = 1, over the same ground samples the OLS fit used.
  - h_ICESat-2 is `h_ref`, the per-photon orthometric height. Its geoid term, `h_ref − height`, varies per point,
    so each photon carries its own geoid correction.

| tile | old a | old b | new a | relief before | relief after | geoid term (median ± sd) |
|---|---|---|---|---|---|---|
| Bhitarkanika | 1.502 | 0.219 | −0.148 | 2.60 m | 11.84 m | 62.03 ± 0.06 m |
| Amalapuram | 1.054 | 0.198 | −0.540 | 2.44 m | 12.31 m | 77.10 ± 0.20 m |
| Kutch | 0.426 | 0.735 | 0.321 | 5.16 m | 7.02 m | 49.42 ± 0.11 m |

The OLS slope had compressed these coastal tiles' relief by 1.4–5×. All three are flagged `needs_rerating`. Their
defaults are `null`, so the app's auto value applies (it is 60x, the ceiling, for all three).

**Default exaggeration.** The auto value is `exaggerationFactor` in `frontend/src/terrain.js:336-344` (with the relief
floor at :325-334). `scripts/build_library_v2.py` `auto_exaggeration` reproduces it from the `terrain.json` the backend
writes.

- The baked values use the slider's own units (terrain.js:817 and :828, `displayExaggeration`), so they reproduce
  today's look exactly.
- Checked live in the viewer: Almora 2.2085 vs 2.20848, Kohima 1.5004 vs 1.50036, Delhi 12.4024 vs 12.40235.

| tile | default | origin | | tile | default | origin |
|---|---|---|---|---|---|---|
| Darjeeling | 1.4 | user | | Vidisha | 17 | user |
| Almora | 2.2085 | auto (baked) | | Kota | 17 | user |
| Manali | 1.0 | auto (baked) | | Vembanad | 60 | auto (baked) |
| Kakinada | 60 | auto (baked) | | Dharamshala | 1.0 | auto (baked) |
| Bhitarkanika | null (auto 60) | after recalc | | Ooty | 1.3 | user |
| Kutch | null (auto 60) | after recalc | | Mandovi | 3.6 | user |
| Kochi | 60 | auto (baked) | | Mumbai | 6.5 | user |
| Nainital | 0.9 | user | | Hyderabad | 6.127 | auto (baked) |
| Dehradun | 1.0 | user | | Chennai | 27 | user |
| Kohima | 1.5004 | auto (baked) | | Nizamabad | 18.4002 | auto (baked) |
| Shimla | 1.5 | user | | Bathinda | 35.5411 | auto (baked) |
| Hisar | 48.6637 | auto (baked) | | Pune | 4 | user |
| Amalapuram | null (auto 60) | after recalc | | Karnal | 38.0032 | auto (baked) |

**Darjeeling colours.** Darjeeling has no B render in B_all_32, so one was generated with the identical stretch:
`previews/_variants/sentinel2-darjeeling__B_own_percentiles.jpg`, with the original kept next to it. The active image
is `texture_variant: original` (it was rated 3 stars on the original colours). **This is the owner's choice.** To
switch, set `texture_variant` and run the `images` step.

## Maxar: DISPLAY band (`scripts/build_maxar_display_packs.py`)

- **Bands:** a 3-band pack.
  - TERRAIN and SURFACE are untouched: statistics, readouts, Measure values and the elevation layer use them.
  - DISPLAY is added and shapes the 3D mesh only.
- **Formula:**
  - obj = max(0, SURFACE − local_ground), and obj < T is set to 0.
  - DISPLAY = mean(TERRAIN) + f·(TERRAIN − mean(TERRAIN)) + k·obj.
- **Changed from the brief:** local_ground is the 10th percentile, over W m, of the **above-ground part**
  (SURFACE − TERRAIN), added back to TERRAIN.
  - Taking the percentile of SURFACE itself marked 77–99 % of pixels as objects on these steep Himalayan crops: a
    30–40 m window on a slope has its 10th percentile far below the centre.
  - With the change, objects cover 37–74 % of pixels on the forest and town crops (trees, buildings) and about 0 % on
    the glacier.
  - `--ground surface` keeps the original recipe.
- **Speed:** the percentile runs on a 4× block-mean grid (~4.9 m cells). A tile takes about 0.25 s; all 3 presets for
  6 tiles take about 5 s.
- **Presets** (unchanged values) are in `dem_maxar/{subtle,medium,strong}/`:
  - subtle: f 0.15, k 3, W 40, T 1.5;
  - medium: f 0.05, k 6, W 40, T 1.5;
  - strong: f 0, k 10, W 30, T 1.
- **Active preset:** `subtle`. **The choice is the owner's.**
  - Switch with `python scripts/build_maxar_display_packs.py --out data/library_v2_2026-09-29 --activate medium`, then
    regenerate the tile. Also set `maxar_display_preset` in the manifest so a rebuild keeps it.
- **Default exaggeration:** `null` (auto). With DISPLAY, the auto value is computed on the DISPLAY range: the relief
  floor rescales it, so a preset's ratio of objects to ground is what shows.
- **App:** `terrain.json` gains an optional `display` field (`backend/terrain/mesh_export.py`), which `terrain.js`
  extrudes.
- **Details panel:** "3D shape is a cosmetic display surface; statistics show real elevations."
- **Caveat:** the flood overlay's water plane is placed from real elevations, so on DISPLAY tiles it does not sit on
  the cosmetic mesh.

## DFC2019

### Flat tiles: a frontend bug, not flat data

OMA_211_039, OMA_211_032 and OMA_376_023 (resolved from `OMA_376_23`) have 28–29 m of real relief in their packs, and
9–13 % of their pixels are over 2 m. That is comparable to JAX_416_009, which renders fine. The backend's
`terrain.json` is correct (e.g. 299.91–328.17 m).

**The cause** is `frontend/src/outlier-relief.js` (`outlierFences` / `limitOutliers`):

- When at least 75 % of the mesh cells are bare ground, Q1 and Q3 both sit within centimetres of the ground.
- The IQR is then 0.7–4 cm, and every building above Q3 + 1.5·IQR is capped at about one IQR above the fence.
- The app then shows 2–15 cm of the 21–34 m. Almost every AGL pixel is valid (fallback fired on one pixel in total), so
  the fallback is not the cause.

**Rule** (measured on the rendered surface): a tile renders flat when Q3 of its heights above the tile minimum is under
0.05 m.

- Largest caught Q3: 0.043 m. Smallest not caught: 0.857 m. Separation margin: **19.8×**.

| tile | mesh relief | Q3 above min | shown relief | verdict |
|---|---|---|---|---|
| OMA_211_039 | 28.26 m | 0.007 m | 0.02 m | limiter bug; kept included, decision pending |
| OMA_211_032 | 29.18 m | 0.008 m | 0.03 m | limiter bug; kept included, decision pending |
| OMA_212_033 | 21.16 m | 0.008 m | 0.03 m | limiter bug; kept included, decision pending |
| OMA_376_023 | 28.92 m | 0.028 m | 0.10 m | limiter bug; kept included, decision pending |
| OMA_376_038 | 28.89 m | 0.029 m | 0.10 m | limiter bug; kept included, decision pending |
| OMA_258_020 | 33.57 m | 0.043 m | 0.15 m | limiter bug; kept included, decision pending |
| OMA_144_030 | 0.75 m | 0.017 m | 0.06 m | **genuinely flat data** (whole tile < 2 m): **excluded** |

- OMA_144_030 is caught by the rule.
- Per the brief, the six bug tiles are **not excluded**, because this is a pipeline bug. The owner decides: fix the
  limiter (e.g. fences from above-ground cells only) or exclude them.
- Weaker hits from the same limiter show only 7–16 % of their relief: OMA_281_002/030, JAX_224_025, OMA_230_036 and
  OMA_225_001.

### Exaggeration, 10x–16x

- **Anchors, exact:** OMA_269_035 12x, JAX_264_013 12x, JAX_022_009 10x, JAX_004_006 10x, JAX_416_009 16x.
- **Candidates** on the rendered surface, Spearman vs the anchor values: p98−p2 **−0.79**, p99 AGL −0.37, AGL std −0.37,
  max AGL 0.00. p98−p2 was chosen.
- **Ordering by p98−p2:** OMA_269_035 10.4 m (12x), JAX_416_009 12.4 m (16x), JAX_264_013 15.4 m (12x), JAX_022_009
  16.1 m (10x), JAX_004_006 18.9 m (10x).
  - It is consistent except OMA_269_035, which has the lowest relief but a 12x anchor.
  - It is sensible enough to fit, but weak: the maximum error at an anchor before override is 3.3x.
- **Model:** log(exag) = 3.856 − 0.521·log(p98−p2), clamped to [10, 16], in 0.5x steps. The anchors keep their exact
  values.
- **Current auto for the anchors:** 31.8 / 18.8 / 32.7 / 23.6 / 32.2x.
- **Resulting values on the 49 included tiles:** 20 at 10x, 6 at 10.5x, 2 at 11x, 4 at 11.5x, 7 at 12x, 3 at 12.5x,
  1 at 13x, 2 at 13.5x, 3 at 14x, 1 at 16x (the anchor). All lie in 10–16x.
- The full 50-tile table (AGL p50/p98/p99/max, % > 2 m, fallback %, nodata, ground source, terrain range, statistic,
  predicted, final) is kept **locally** in `data/library_v2_2026-09-29/_analysis/dfc_tables.md`. It is DFC2019-derived,
  so it is not committed to the public repo.

### Credit

- The Docs page ("Where the heights come from") credits IEEE GRSS DFC2019 and JHU/APL US3D.
- Every DFC2019 tile's Details panel does too: generation `meta.credit`, from `backend/generation/pipeline.py`.

## App changes (branch `curated-tiles-2026-09-29`)

| file | change |
|---|---|
| `backend/library/tile_manifest.py` | new: `DW2_TILE_MANIFEST`, curate / default_exaggeration |
| `backend/api/library_routes.py` | listing through the manifest; items carry order_index, stars, default_exaggeration |
| `backend/generation/pipeline.py` | DISPLAY band; meta `display_note`, `credit`, `default_exaggeration`, `mesh_from` |
| `backend/terrain/mesh_export.py` | optional `display` field in terrain.json |
| `frontend/src/terrain.js` | extrudes `display` when present |
| `frontend/src/main.js` | `applyDefaultExaggeration` (before history baseline and saved view) |
| `frontend/src/input-view.js` | a curated listing's `order_index` is followed |
| `frontend/src/side-panels.js` | Details note: display note and credit |
| `frontend/index.html` | Docs credit |
| tests | `backend/tests/test_tile_manifest.py` (3), `frontend/tests/input-view.test.mjs` (+1): 58 + 93 pass |

## Local QA switch

- `data/library_v2_2026-09-29/qa.env` is untracked.
  - It sets `DW2_LIBRARY=bundle`, `DW2_LIBRARY_BUNDLE`, `DW2_TILE_MANIFEST` and scratch job directories.
  - Start the backend with `_qa/start_backend.sh` (port 8831) and Vite with `_qa/vite.qa.config.mjs` (port 5193).
  - The production config is unchanged.
- **Container staging:** `python scripts/stage_container_tiles.py` is a dry run.
  - It lists 81 packs (26 Sentinel-2, 6 Maxar, 49 DFC2019; 41.0 MB) and their 162 images (27.7 MB).
  - `--copy DEST` exists and has not been run.
  - No Dockerfile was written or built.

## Log

- **2026-09-29, Part 0:** baseline hashed (474 files). Branch created. Ratings recorded; manifest seeded (89 tiles).
- **2026-09-29, Part 1:**
  - 3 packs recalculated (offset only).
  - B images copied; Darjeeling B generated.
  - Auto values baked (verified live against the viewer).
  - 7 tiles excluded.
- **2026-09-29, Part 2:**
  - DISPLAY packs for 6 tiles × 3 presets.
  - The local-ground recipe was changed after a first bake marked 77–99 % of pixels as objects.
- **2026-09-29, Part 3:**
  - The flat-tile cause is found: a frontend limiter bug. Only OMA_144_030 is excluded.
  - The p98−p2 mapping is fitted (Spearman −0.79).
  - Credits added.
- **2026-09-29, Part 4:** app changes, QA switch and staging script. Tests pass.
- **2026-09-29, Part 5:**
  - The preview ran against library_v2 on 192.168.1.5:5193.
  - 19 renders are in `_qa/screenshots/`: 2 per star tier, the 3 recalculated tiles, Darjeeling ×2, DFC2019 10x / 12x /
    16x, and Maxar 2 tiles × 3 presets.
  - The checksums are identical.
