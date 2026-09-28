# 08 — DFC2019 terrain packs: series plan + Prompt 1 preflight (2026-09-28)

**Goal of the series.** Each of the 50 DFC2019 library tiles gets a curated terrain pack
(real ground level + Method 6 heights + reference-corrected tall objects), so the viewer shows
relief instead of today's flat zero plane (`backend/generation/pipeline.py`, `elev is None`
branch). Live inference for uploads and the desktop app are out of scope.

Private scratch area for the whole series: `data/dfc2019/terrain_packs/` (contains its own
`.gitignore` = `*`, so nothing in it is ever committed; verified with `git check-ignore`).

## Prompt 1 preflight findings

### 1. Tile inventory (`terrain_packs/p1_inventory.json`)
- 50 tiles: **26 JAX + 24 OMA**. Every tile has `Track1-RGB/{t}_RGB.tif` (1024×1024×3, no
  geotransform) and `Track1-Truth/{t}_AGL.tif` + `_CLS.tif`. AGL has no nodata tag.
- Valid AGL: **every tile ≥ 99.9999 %**. JAX_004_016 has exactly **1 NaN** pixel. No sentinel
  values (< −100 or > 1000 m). **No tile has little or no valid AGL.**
- Small negatives are universal (ground noise; min −3.16 m on OMA_281_030). Only OMA_281_002/030
  and OMA_376_023/038 have more than ~100 pixels below −1 m.
- OMA_144_030 is essentially flat (AGL max 1.14 m): a valid tile with nothing tall on it.
- Several tiles share an identical AGL range (004_006/004_016, 416_009/022, 505_016/018,
  214_015/023, 315_019/020, 248_029/030): overlapping renders of one footprint, useful as
  consistency checks in Prompt 2.

### 2. Public tile set
- `sancharimouri/depthwizard2-assets@library-v1` holds **42** DFC2019 tiles (tile + preview each).
- The other **8** (JAX_072_015, JAX_161_001, OMA_212_033, OMA_225_001, OMA_248_029,
  OMA_269_035, OMA_315_020, OMA_364_003) are the `desktop/tiles/selection.json` bundled set,
  shipped inside the public desktop installer (`depthwizard2-desktop` v1.0.1).
- So **all 50 RGB tiles are public in some form.** The manifest routes all 50 through
  `store: hf-private` (web backend), `georeferenced: false`, `geo: null`.
- Series rule: packs are **never** published, not even for the public tiles.

### 3. Pack format a DFC2019 pack must match
Consumed by `pipeline.generate()` → `_library_elevation(item)`:
- One **2-band float32 GeoTIFF** per item: band 1 `TERRAIN` (bare earth), band 2 `SURFACE`
  (surface model). NaN = nodata. Tags `TERRAIN_SOURCE`, `SURFACE_SOURCE`
  (`build_dem_pack.write()`). Found via `library_store.local_asset(item, "dem")` =
  `<base>/dem/<item id>.tif`, and only in `local` and `bundle` modes. **In remote (web) mode
  there is no pack lookup today**: `_library_elevation` falls back to `item["geo"]`, which is
  null for DFC2019, so it returns None and the tile renders flat.
- `generate()` uses: `crs`, `transform`, `bounds`, `res_m = |transform.a|`; ramps **TERRAIN**
  into `elevation.png` (`_colour_ramp`, min–max over the grid); meshes **SURFACE** into
  `terrain.json` via `write_terrain_json(surface, lonlat_bounds, _mesh_hw(shape))`.
  `lonlat_bounds = transform_bounds(crs → EPSG:4326)`.
- `terrain.json`: `width`, `height`, `bounds{west,south,east,north}` (degrees),
  `elevationMin`, `elevationMax` (metres), `heights` (normalised 0–1, row-major). The viewer
  turns the bounds into metres with 111 320 m/deg × cos(mid-lat) (`frontend/src/terrain.js`,
  `measure-metrics.js`).
- `meta.json`: `has_elevation`, `terrain_source`, `surface_source`, `how`, `crs`,
  `resolution_m`, `grid`, `bounds_lonlat`, `terrain_range_m`, `surface_range_m`.
- `_mesh_hw`: MESH_MAX 400 → a 1024² grid meshes at **341×341**.

**Decisions committed now (before any pack exists):**
- *Local metric bounds without a location.* Packs carry a synthetic local CRS,
  `+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84`, with bounds
  (0, 0, 307.2, 307.2) m (1024 px × 0.3 m). `transform_bounds` then yields ~0.00276°
  square at (0°, 0°). There, cos(lat) = 1, so the viewer's metric extent is correct, and no
  real coordinate enters the pack, the manifest or the output. Verified: bounds
  (0, 0, 0.0027596, 0.0027782)°, and the viewer's formula gives 307.20 m × 309.27 m. The
  +0.7 % N–S stretch comes from the viewer's constant 111 320 m/deg (the true value at the
  equator is about 110 574). It is cosmetic, and is recorded rather than patched.
- *Grid.* The pack is written at the mesh resolution (**341×341**, block mean), not 1024²,
  so no pack ever carries a finer height raster than the mesh it feeds.
- *Bands.* TERRAIN = base ground (Prompt 3); SURFACE = base ground + corrected heights (Prompt 5).
- *Open item for Prompt 5.* For city-level tiles TERRAIN is a constant, so `_colour_ramp`
  gives a single colour. Prompt 5 must decide whether DFC2019 `elevation.png` ramps SURFACE,
  and record the decision before building.

### 4. Checkpoints
- Local `data/dfc2019/experiments/method6_height_balanced_seed43/fold{0..3}.pt` and
  `method6_full_checkpoint/method6_full_dfc2019.pt`. All 5 **SHA-256-identical** to the private
  HF model repo `sancharimouri/depthwizard2-method6` (private: true).
- **Fold q holds out quadrant q** (`evaluate_method4.quadrant_bounds`, split at 512): 0 = rows
  0–511 × cols 0–511, 1 = rows 0–511 × cols 512–1023, 2 = rows 512–1023 × cols 0–511, 3 = rows
  512–1023 × cols 512–1023. It is the same for all 50 tiles
  (`evaluate_method6_gsd_film_height_balanced.py` L313–335). The full checkpoint trained on all
  50 tiles, so it must not be used for pack heights.
- Inference path to reuse: `scripts/vhr_dsm_pipeline.py` → `load_models`,
  `tiled_predict_margin(model, rgb, device, margin=192)` (reflect/mirror pad, keeps each 512 px
  window's 128 px core, 16 px core ramp). The seam metric is `seam_ratio(a, valid, lines)`
  (mean gradient on seam lines / elsewhere; 1.07 on the glacier crop at margin 192; limit 1.5).

### 5. Point clouds: present, and classified
- `~/Downloads/JAX_PointClouds.zip` (1.70 GB → 8.12 GB, 416 tiles) and
  `~/Downloads/OMA_PointClouds.zip` (2.52 GB → 12.48 GB, 390 tiles). Both pass `unzip -t`.
  `OMA_PointClouds (1).zip` is a byte-identical duplicate. Metadata tars are also present.
- No re-download is needed.
- Format (README, SEMANTIC3D): `*_PC-reduced.txt` = `E,N,Up,intensity,return`, UTM (JAX 17N);
  `*_PC-classification.txt` gives **per-point classes (2 ground, 5 foliage, 6 building, 9 water,
  17 elevated road)**. The Prompt 2 brief assumed no labels; ground returns (class 2) are
  available and give a better AGL layer.
- JAX ground Up ≈ −12 m in a city at +5–20 m NAVD88, so **Up is very likely ellipsoidal
  (WGS84/NAD83) height**. That is exactly the offset Prompt 3's datum check must catch (the geoid
  is about −29 m in Florida and about −27 m in Nebraska).

### 6. Public DEM (USGS 3DEP): programmatic fetch works
- TNM Access API → 1 m DEM product → windowed COG read over `/vsicurl/`:
  - JAX (30.332, −81.656): `USGS 1M 17 x43y336 FL_Peninsular_FDEM_2018`, EPSG:26917, 1 m;
    300 m window median 4.68 m. Vertical **NAVD88** (read from the product XML metadata).
  - OMA (41.257, −95.935): `USGS one meter x25y458 NE Eastern UA 2016`, EPSG:26915, 1 m;
    median 317.3 m. (The `IA_WesternIA_2020` tiles listed there are Iowa-only and NaN over
    Omaha, so products must be filtered by valid coverage, not just by the first hit.) This
    product has no XML sidecar, so NAVD88 comes from the 3DEP specification and still has to
    be confirmed by Prompt 3's check.
  - 1/3 arc-second (EPSG:4269) also reads: OMA median 317.5 m.

### 7. Storage
- Private HF dataset `sancharimouri/depthwizard2-library-private` (private: true; 153 files:
  tiles/previews/thumbnails ×50, manifest, README; 141 MB); our token has write access. Packs go to
  `dem/<item id>.tif` (= `library_store.FOLDERS["dem"]`).
- Generation runs server-side, so the pack never reaches the browser. The needed change is a
  remote-mode `library_store.private_file(item, "dem")` lookup (via `_hub_file`, same token and
  cache as tiles) in `_library_elevation`. The browser only sees `/api/generated/<job>/*`.

## Series plan
| Prompt | Output (all in `data/dfc2019/terrain_packs/`, gitignored, unless noted) |
|---|---|
| 2 Locate | Matcher with self-test + OMA-vs-JAX negative control; pre-registered confidence rule; `locations.json` (private). Ground from **class-2 returns** (post-brief improvement, recorded). |
| 3 Base ground | 3DEP 1 m (fallback 1/3″) per located tile on the pack grid; datum check vs. class-2 ground (ellipsoid → NAVD88 via geoid); city-level median + p5/p95 for the rest, cross-checked with published elevations. |
| 4 Heights | Fold-q model on quadrant q with full-tile context, margin 192, feathered quadrant borders; seam ratios; measured cap + margin committed to a file before Prompt 5. Parity: backend path vs. `vhr_dsm_pipeline`. |
| 5 Packs | Pre-registered correction rule; 50 packs (341², local tmerc); private HF `dem/`; remote-mode pack lookup in `_library_elevation`; QA screenshots + test suites; per-tile report in this folder. |
