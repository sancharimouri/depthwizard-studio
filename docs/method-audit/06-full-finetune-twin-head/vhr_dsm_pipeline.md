# Method 6 → absolute DSM on real India VHR imagery (Maxar, Sikkim/Darjeeling hills)

_2026-09-23. This bridges "Method 6 works on DFC2019" and a renderable elevation surface on real
India imagery. **No ground truth exists for these crops.** Everything below is a visual/statistical
plausibility pass, not an accuracy claim._

## Did this integration exist before?

**No.**
- `backend/terrain/composer.py` (`# DEM terrain + nDSM -> absolute DSM.`) and
  `backend/terrain/mesh_export.py` were one-line stubs from the initial commit `660ecb6`.
- The Method 6 scripts compute metrics only.
- `scripts/method6_vhr_sanity_check.py` saved raw `mu` arrays only.
- No code in the repo composed a terrain DEM with predicted AGL, or wrote a renderable surface.

## Design (fixed before any output was inspected)

- **DSM = FABDEM + max(AGL, 0).**
  - FABDEM is bare earth: the recommended terrain baseline (A3) and the physically correct base
    for height above ground.
  - Adding AGL to GLO-30, which is already a surface model, would double-count canopy and
    buildings.
  - Vertical datum: EGM2008 orthometric, inherited from FABDEM.
- **AGL** = mean of the 4 **seed-43 fold checkpoints** of the adopted height-balanced recipe
  (`data/dfc2019/experiments/method6_height_balanced_seed43/fold*.pt`).
  - Cross-check: the full-data checkpoint `method6_full_checkpoint/method6_full_dfc2019.pt`
    (original recipe, all 50 tiles).
- **Inference** is tiled at 512 px, the training input size (512² quadrants padded to 518), with
  stride 448 (64 px overlap) and feathered blending.
  - The earlier VHR sanity check instead fed a single 1,512² crop, off the training input size.
- **GSD:** the Maxar scenes are 0.305 m, essentially DFC2019's ~0.3 m. No resampling of the
  imagery.
- **Crops:** six 2,048² (625 m) crops from the three already-downloaded Maxar scenes
  (`data/maxar_sanity/*.tif`, EPSG:32645), chosen from overview thumbnails by land cover before
  inference:

| crop | scene | land cover |
|---|---|---|
| `c_town` | `45_120220211230` (27.18°N 88.34°E) | dense hill town |
| `c_river` | same | braided river / sandbars / fields |
| `c_terraces` | same | terraced farmland, scattered houses, greenhouses |
| `a_forest` | `45_120220122200` (27.63°N 88.65°E) | dense conifer forest |
| `a_valley` | same | river valley with settlement/buildings |
| `b_glacier` | `45_120220031201` (28.00°N 88.30°E) | snow/ice (**negative control**, AGL ≈ 0) |

- **Why not more India terrain:** Maxar Open Data's India-relevant events were checked via its
  STAC catalog. `India-Floods-Oct-2023` is entirely Sikkim/Darjeeling hills (27.1–28.0°N), and
  Cyclone Mocha covers Myanmar/Bangladesh. Bhoonidhi has no API and needs a manual portal fetch
  by the owner. So terrain diversity comes from land-cover diversity within the hills.
- **Outputs per crop** (`data/vhr_dsm/<crop>/`):
  - GeoTIFFs: `agl.tif`, `agl_std.tif` (fold spread), `sigma.tif` (predicted aleatoric), `dtm_fabdem.tif`, `dsm.tif`
  - hillshade and colour PNGs
  - the frontend asset set (`terrain.json` in the viewer's schema, `satellite.png`, `elevation.png`, `relative_depth.png` = AGL)
  - `stats.json`

## Plausibility criteria (committed before running)

These are sanity thresholds, not accuracy bars.

| # | check | "plausible" if |
|---|---|---|
| C1 | magnitude, per crop | glacier AGL p95 < 2 m; river-valley AGL median < 1 m; forest AGL median 5–40 m; town AGL p95 5–40 m; every crop max < 80 m |
| C2 | artifacts | seam ratio (mean \|∇AGL\| on tile-seam lines ÷ elsewhere) ≤ 1.5; raw AGL < 0 on < 5% of pixels |
| C3 | internal consistency | Pearson(fold-ensemble AGL, full-data-checkpoint AGL) ≥ 0.7; median fold std < 25% of AGL where AGL > 2 m |
| C4 | independent coarse reference | Spearman(AGL block mean, GLO-30 − FABDEM) at ~30.5 m blocks > 0 on crops with canopy/buildings (town, terraces, forest, valley). GLO-30 − FABDEM is the canopy/building excess FABDEM removed; it's an X-band DSM effect, so partial, and correlation (not level) is the check |
| C5 | sparse lidar (descriptive only) | where GEDI rh98 / ICESat-2 20 m canopy segments fall inside a crop, report AGL vs. lidar; no threshold (few footprints expected in 625 m crops) |
| R | renderability | `terrain.json` + textures load and render through the unmodified `frontend/src/viewer.js` |
