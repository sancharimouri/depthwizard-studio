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

## Seam fix: margin-discard tiling (design and pass criteria written before the rerun)

**The fix.** `tiled_predict_margin`, run with `--margin=192`:
- Reflect-pad the whole image by 192 px.
- Run the same 512 px windows, but keep only each window's central 128 px core. The outer 192 px ring of every window is discarded.
- Blend overlapping cores with 16 px linear ramps (step 112 px).
- As a result, no kept pixel is closer than 192 px to a window edge, including at the image border, which the reflect padding covers.

**How the margin was chosen.** From the committed glacier AGL: the column-median profile rises from about 0.36 m to 1.0–1.4 m over the last roughly 100–180 px before each window's right edge. The left-edge effect is under 64 px. So the existing 64 px feather cannot absorb it, and 192 px is set to cover the worst case with room to spare. This choice was made from the failed crop's own profile, so it is **post-hoc by construction**; the criteria below are fixed before the rerun.

**Scope.** The rerun covers `b_glacier` only, writing to `data/vhr_dsm/b_glacier_margin192/`. The original outputs are kept for comparison.

**Pass criteria.**
- **S1:** the C2 seam ratio is ≤ 1.5 on the **original** 448-stride window edges, using the same metric as the committed 2.37.
- **S2:** the seam ratio is ≤ 1.5 on the **new** layout's core edges. This checks that the fix doesn't just move the seams.
- **S3:** the panel hillshade shows no visible grid.
- **Reported, not a criterion:** glacier C1 (p95 < 2 m) and C3 (fold std / AGL) against their earlier failures.

**If it passes.** The other five crops are **not** rerun in this pass; they showed no seams, with a seam ratio of 1.03–1.07. Whether to make margin mode the default is a separate decision.

### Seam-fix result (`b_glacier_margin192`, 2026-09-23)

| | before: feather-only, stride 448 | after: margin 192 |
|---|---|---|
| S1: seam ratio on the old 448-stride edges | 2.37 | **1.07** (pass) |
| S2: seam ratio on the new core edges | (n/a) | **1.13** (pass) |
| \|ΔAGL\| between adjacent px, p99 / p99.9 / max | 0.056 / 0.122 / 0.65 m | 0.021 / 0.048 / 0.27 m |
| AGL p50 / p95 / max | 0.60 / 2.82 / 9.1 m | 0.27 / **0.59** / 5.0 m (C1 glacier p95 < 2 m now passes) |
| r(ensemble, full checkpoint) | 0.830 | **0.581** (C3 < 0.7) |
| fold std / AGL, where AGL > 2 m | 0.33 | 0.49 |

**S3 (visual): mostly passes.**
- The regular 448 px grid is gone from both the AGL map and the DSM hillshade.
- A few faint, isolated outlines about the size of one core (~128 px) remain in the hillshade.
- A hillshade exaggerates sub-decimetre steps. The step statistics above put these residuals at under 5 cm per pixel at p99.9.

**The C3 drop is informative, not a regression.** Before the fix, both AGL maps (fold ensemble and full-data checkpoint) were produced with the same tile layout. So they shared the same seam grid, and the grid was most of the glacier's variance. The earlier r = 0.83 was therefore partly inflated by the shared artifact.

After the fix, glacier AGL is nearly constant at about 0.3 m. The correlation then compares two near-zero noise fields, and 0.58 is what that looks like. On a textureless negative control, C3 is not a meaningful check. The fold std / AGL > 2 m ratio is computed over only the handful of pixels still above 2 m. **This interpretation is post-hoc.**

**Decision.** The margin mode is kept as the `--margin=M` option. The committed five-crop results still use the original tiling; they showed no seams (seam ratio 1.03–1.07). Making margin 192 the default costs about 14× more forward passes per crop (361 vs. 25 windows). A future full rerun should use it, **decided now rather than after seeing another result**.

## Height ceiling: is it the training range? (2026-09-23)

**Question.** Every VHR crop tops out at an AGL of about 18–23 m, and the forest is short compared with GLO-30 − FABDEM and with GEDI. Is that because Method 6's training data (DFC2019) doesn't go higher?

### Training AGL distribution

Source: `data/vhr_dsm/_diagnostics/dfc2019_agl_distribution.json`. Class codes are DFC2019 ASPRS (5 = high vegetation, 6 = building).

| population | p50 | p90 | p95 | p99 | p99.9 | > 20 m | > 30 m |
|---|---|---|---|---|---|---|---|
| Method 6 training set, 50 JAX tiles (union of all folds), **all pixels** | 0.1 | 12.2 | 16.4 | 28.7 | 71.4 | 2.71% | 0.85% |
| same, **high vegetation only** | 8.2 | 16.0 | 18.2 | **23.6** | 30.5 | 2.81% | 0.13% |
| same, **buildings only** | 7.8 | 23.6 | 30.5 | 65.2 | 91.2 | 13.6% | 5.2% |
| full Track 1 JAX (1,015 tiles, 1/16 px), high vegetation | 9.9 | 17.9 | 20.3 | 25.9 | 30.6 | 5.5% | 0.14% |
| full Track 1 OMA (1,768 tiles, 1/16 px), high vegetation | 6.4 | 12.9 | 15.0 | 19.3 | 25.4 | 0.78% | 0.02% |

- Method 6 trains on the **50 JAX benchmark tiles only**. Omaha is not in its training set.
- Omaha's trees are shorter still, so adding Omaha would not extend the canopy range.

### Can Method 6 output tall values at all? Held-out DFC2019 test

Setup: each seed-43 fold model predicts its own held-out quadrant of all 50 tiles, and predictions are binned by true height. Source: `data/vhr_dsm/_diagnostics/dfc2019_heldout_pred_by_height.json`.

| true height bin | buildings: median truth → median pred (p90 pred) | high vegetation: median truth → median pred (p90 pred) |
|---|---|---|
| 10–15 m | 12.3 → 10.0 (15.6) | 12.2 → 10.0 (14.2) |
| 15–20 m | 17.5 → 15.1 (23.3) | 16.7 → 12.0 (16.8) |
| 20–25 m | 22.0 → 16.9 (25.5) | 21.5 → 13.4 (19.6) |
| 25–30 m | 27.4 → 18.5 (24.0) | 27.2 → 15.8 (21.2) |
| 30–50 m | 37.9 → 22.1 (33.0) | 31.1 → 16.3 (21.7), n = 10.9k |
| 50 m + | 65.9 → 26.5 (43.0) | (n = 241, too few) |

Across all held-out pixels, prediction p99 is 22.8 m and the max is 55.2 m, against a truth p99 of 28.7 m.

### Verdict on the hypothesis: confirmed for canopy, with one correction

**Confirmed: the canopy ceiling is a training-range limitation.**
1. Trees in Method 6's training data top out right where the VHR ceiling sits: p95 is 18.2 m and p99 is 23.6 m, and only 0.13% of tree pixels exceed 30 m.
2. The same ceiling appears **in-domain on held-out DFC2019 trees**. Trees that are truly 25–50 m are predicted at a median of 16 m, with a p90 of about 21 m. This matches the 18–23 m maxima seen on all six VHR crops.

So the VHR forest result is not a new transfer failure. It is Method 6's known in-domain behaviour, carried over to terrain where tall trees are common. Himalayan conifers routinely exceed 30 m.

**Correction: it is not a hard output-range or capacity limit.**
- The same models predict 30–55 m on tall held-out *buildings*: p90 is 43 m in the 50 m + bin. Building supervision is much denser at those heights (5.2% of building pixels exceed 30 m).
- Buildings are still strongly shrunk (true 66 m → 26.5 m median). That is the variance compression already on record (variance ratio 0.48–0.66), and it follows how scarce tall examples are.

**The accurate statement:** Method 6 learned heights in proportion to how often it saw them. Its training trees almost never exceed about 24 m, so its canopy predictions don't either.

## Does GAMUS have the tall-canopy tail that DFC2019 lacks? (2026-09-23, data check only, no training)

**Method.** Own code, written independently. It reads nothing from any audited repo except the dataset location (`earthflow/GAMUS` on Hugging Face).
- 60 random tiles per city (seed 0, all splits), heights and classes. Each tile is 1024², at about 0.33 m.
- Tiles are downloaded to scratch, reduced to statistics and deleted.
- Void = −5 sentinel.
- Class 6 = tree and 3 = building. This matches the GAMUS class order and is consistent with the data: class 3 has the heavy 50 m + tail, class 6 a median of 12 m and p99.9 ≤ 37 m.
- Source: `data/vhr_dsm/_diagnostics/gamus_agl_distribution_sample.json`.

| tree pixels | p50 | p95 | p99 | p99.9 | > 20 m | > 25 m | > 30 m |
|---|---|---|---|---|---|---|---|
| DFC2019, Method 6's 50 JAX tiles | 8.2 | 18.2 | 23.6 | 30.5 | 2.8% | 0.72% | 0.13% |
| GAMUS **DC** | 11.9 | 27.1 | 32.7 | 36.8 | 18.2% | 7.7% | 2.5% |
| GAMUS **NYC** | 11.7 | 26.0 | 29.5 | 34.6 | 19.0% | 6.8% | 0.78% |
| GAMUS PHL | 4.1 | 15.0 | 21.2 | 29.7 | 1.4% | 0.36% | 0.09% |

### Verdict: **not a clean negative.** GAMUS has a real, longer canopy tail than DFC2019.

**What GAMUS adds.** Per tree pixel, trees above 25 m are about 10× more frequent in DC and NYC than in Method 6's training set. Above 30 m, DC has about 19× the frequency.

**Limits on what it can fix.**
1. **It extends the ceiling; it doesn't remove it.** GAMUS canopy still ends at about 35–37 m (p99.9). Tall Himalayan conifers can exceed that. Whether the `a_forest` canopy really does is unknown: GEDI's 51 m is slope-inflated, and GLO-30 − FABDEM (12 m median excess) is a partial X-band signal.
2. **Domain.** These are US urban and suburban, mostly deciduous street and park trees in aerial imagery. Sikkim hill conifers look different from above.
3. **Imagery for NYC.** The competitor repo's audit says NYC has no imagery. The Hugging Face listing does contain NYC `images/*.h5` files, and one checked is 3.1 MB, the size of a 1024² RGB tile. Their content is **not** verified here.
4. **Tall-building supervision** from GAMUS (DC building p95 33.5 m) would also help the buildings' shrinkage. That is a separate benefit.

**Not done: no retraining.** Retraining is not authorized in this session.

**The concrete next step, if authorized.** Fine-tune Method 6 on DFC2019 JAX plus GAMUS DC (+ NYC if its imagery checks out), keeping the height-balanced sampler. Pre-register the evaluation before running:
- **(i) DFC2019 regression check.** On the same 4-fold quadrant holdout, it must not lose more than 2% MAE against 1.980 m.
- **(ii) Held-out tall-tree test.** On DFC2019 high-vegetation pixels truly 20–30 m, the median prediction must rise from 13.4–15.8 m to at least 18 m.
- **(iii) VHR `a_forest` crop.** Report AGL p95 and C4 ρ. Descriptive only, since there is no ground truth.

Because this uses GAMUS DC, the split must be cut by city or grid block. GAMUS's own splits leak spatially: adjacent DC grid tiles land in different splits.
