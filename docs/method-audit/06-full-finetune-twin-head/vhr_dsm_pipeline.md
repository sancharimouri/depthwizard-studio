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

## Post-hoc change (after the first crop, before the other five)

**DTM resampling: bilinear → bicubic.**
- **What was seen:** the first crop (`c_town`) showed faint ~30 m square facets in the DSM
  hillshade's flat areas. Bilinear interpolation of FABDEM's 30 m grid gives piecewise-planar
  facets whose slope breaks at every cell edge, and a hillshade exaggerates that.
- **Measured on `c_town`:** bicubic cuts the slope-break spikes (p99.9/median of |Laplacian|)
  from **1,564 to 11**, while moving elevations by a median 0.28 m (max 2.9 m).
- **Scope:** this changes only the DTM (and so the DSM's smooth base). It doesn't touch AGL or any
  AGL-based criterion (C1–C3, C5). C4 uses the DTM, and is computed on the bicubic DTM for all
  crops. **Labelled post-hoc.**

## Results (2026-09-23): all six crops, bicubic DTM

The source of every number here is `data/vhr_dsm/summary.json`, with per-crop `stats.json` files beside it. The visual checks come from `data/vhr_dsm/<crop>/panel_rgb_agl_dsmhs_dtmhs.png`. Each crop is 2048 × 2048 px at 0.305 m, so about 625 m across. The primary AGL is the seed-43 4-fold ensemble. Heights are in metres.

| crop | land cover | AGL p50 / p95 / max | neg. AGL | seam ratio | r(ens, full ckpt) | fold std / AGL | C4 ρ | GLO30−FABDEM vs AGL block, median | DSM range |
|---|---|---|---|---|---|---|---|---|---|
| c_town | dense hill town | 3.64 / 12.56 / 18.4 | 0.01% | 1.03 | 0.979 | 0.14 | **+0.45** | 2.99 vs 4.03 | 1204–1513 |
| c_river | braided river, sandbars | 1.22 / 12.58 / 22.7 | 0.03% | 1.07 | 0.975 | 0.20 | **+0.55** | 2.69 vs 2.12 | 370–622 |
| c_terraces | terraces, houses | 4.00 / 13.09 / 21.2 | 0% | 1.06 | 0.975 | 0.16 | **+0.57** | 8.00 vs 4.60 | 799–1197 |
| a_forest | dense conifer | 7.46 / 13.62 / 18.8 | 0% | 1.03 | 0.897 | 0.19 | −0.05 | 12.06 vs 7.70 | 2627–3164 |
| a_valley | valley settlement | 5.64 / 12.82 / 18.7 | 0.01% | 1.06 | 0.911 | 0.22 | **+0.68** | 5.15 vs 6.10 | 1609–1940 |
| b_glacier | snow/ice (neg. control) | 0.60 / 2.82 / 9.1 | 0% | **2.37** | 0.830 | **0.34** | (n/a) | 0.01 vs 0.66 | 5681–5822 |

### Verdicts against the pre-registered criteria

**C1 (magnitude): 3 of 5 per-crop checks pass, and the global max check passes.**
- **Town** passes: p95 is 12.6 m, inside 5–40.
- **Forest** passes: median is 7.5 m, inside 5–40.
- **Every crop** passes the max check: all are under 80 m, the highest being 22.7 m.
- **River** fails: its median is 1.22 m against a threshold of 1 m. The crop is roughly half riparian trees, buildings and fields, not pure sandbar. With hindsight, the threshold assumed a purer crop than the one chosen. That reading is post-hoc and doesn't change the verdict.
- **Glacier** fails: p95 is 2.8 m against a threshold of 2 m, and the cause is the seam artifact under C2.

**C2 (artifacts): 5 of 6 pass. The glacier fails.**
- Five crops have a seam ratio of 1.03–1.07, and none of those panels shows visible seams.
- The negative-AGL fraction is ≤ 0.03% everywhere.
- The glacier's seam ratio is 2.37. Its panel shows a clear 448 px tile grid: AGL rises to about 1–3 m along tile borders on featureless snow. The DSM hillshade shows the same grid.
- The failure mode is specific: when an image has no texture, the model's output depends on distance to the tile edge. The 64 px feather blends that dependence but doesn't remove it. Textured scenes don't show it.
- A likely fix is to pad or mirror tiles, or overlap them more. That fix is **not** applied here, because it would be a post-hoc change made after seeing the failure.

**C3 (internal consistency): the Pearson check passes on all six; the fold-spread check fails on the glacier.**
- r(ensemble, full-data checkpoint) is 0.83–0.98 on every crop.
- Fold std / AGL is 0.14–0.22 on the five textured crops and 0.34 on the glacier.
- The two checkpoints were trained differently: the full-data checkpoint uses the original recipe, the ensemble uses the height-balanced recipe. So agreement is not guaranteed by construction.

**C4 (independent coarse reference): passes on town, terraces and valley. Fails on forest (ρ = −0.05).**
- The forest failure has a plausible explanation: in a closed canopy, the GLO-30 − FABDEM excess stays high across nearly every block, about 12 m. The within-crop variation is then mostly noise in X-band penetration and in FABDEM's canopy removal. So the rank test there has little signal to find. This explanation is post-hoc.
- On the **river** crop, which was not required, ρ is +0.55.
- On the **level**, not a criterion: median AGL blocks track the GLO-30 − FABDEM excess within about ±1 m on town, river and valley. They fall below it on terraces (4.6 vs 8.0) and forest (7.7 vs 12.1).

**C5 (sparse lidar, descriptive only): GEDI L2A rh98 at 6–23 footprints per crop.**

| crop | AGL p98 median (m) | GEDI rh98 median (m) | Spearman |
|---|---|---|---|
| town | 8.2 | 25.5 | 0.45 |
| terraces | 11.1 | 24.8 | 0.26 |
| valley | 11.1 | 17.7 | 0.20 |
| river | 5.1 | 25.6 | −0.03 |
| forest | 12.2 | 51.1 | 0.40 |
| glacier | 1.0 | 3.0 | — |

- On steep slopes, GEDI rh98 is biased high: the waveform spreads over the slope. The glacier's 3 m reading on bare snow and the forest's 51 m reading both show it. So the absolute gap here overstates the error.
- The **direction is still consistent everywhere**: AGL is below lidar on every crop.

**R (renderability): PASS.**
- All six crops export `terrain.json` plus three textures in the viewer's schema. They load and render through the unmodified `frontend/src/viewer.js` via the standalone, unlinked page `frontend/vhr_preview.html?crop=<crop>`.
- Headless-Chrome (SwiftShader WebGL) screenshots are in `docs/screenshots/vhr_preview_{c_town,a_forest,b_glacier}.png`.
- The relief looks shallow in the viewer. That comes from the viewer's own exaggeration rule: it clamps to 1× at a relief ratio ≥ 0.2, and these 625 m crops have ratios of 0.2–0.9. It is not a property of the DSM.

### Overall reading (post-hoc interpretation, not a criterion)

**The pipeline works end to end and the DSM is physically plausible on textured terrain.**
- Individual buildings appear as flat-topped blocks and tree crowns as blobs, in both AGL and DSM hillshade.
- DSM values match the scale of the FABDEM terrain; AGL is in metres, not tens of metres.
- There are no negative surfaces, and no visible seams except on snow.
- The two checkpoints agree.
- On 3 of 4 required crops, AGL rank-correlates with an independent coarse surface-minus-terrain signal.

**It has two real, specific defects:**
1. **Tile-edge artifacts on textureless scenes** (glacier).
2. **A height ceiling.**
   - AGL tops out at 18–23 m in every crop, and p95 is 12.6–13.6 m even in dense conifer forest.
   - Both independent references (GLO-30 − FABDEM, and GEDI) say forest and terrace canopy is taller than that.
   - The most likely cause is domain, not composition: DFC2019 (Jacksonville) AGL has a p95 of about 16.6 m, which is also the training `height_scale`, and Method 6's DFC2019 outputs were already variance-compressed (0.48–0.66). This is a hypothesis; it hasn't been tested.

**What this does and does not show.** This is a plausibility pass on six crops from one mountainous region of India (Sikkim / Darjeeling hills). It is **not** an accuracy claim, because no ground-truth DSM exists for these crops.
