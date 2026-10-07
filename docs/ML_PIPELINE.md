# ML pipeline

## What the shipped product uses

| Surface | Source | Model involved |
|---|---|---|
| Sentinel-2 library tiles, terrain and surface | FABDEM with a per-tile linear calibration to ICESat-2 ground photons (both bands) | none |
| Darjeeling library tile | Terrain FABDEM, surface Copernicus GLO-30 (no ICESat-2 samples to calibrate with) | none |
| The four demo regions | Copernicus GLO-30 (Darjeeling via OpenTopography) | none |
| Searched Sentinel-2 scene, georeferenced upload | Terrain: FABDEM (fetched automatically) or the user's own DEM; surface: live Copernicus GLO-30. Without an attached DEM, GLO-30 for both | none |
| The 6 Maxar VHR library crops | Terrain FABDEM; surface FABDEM + max(predicted height above ground, 0), baked offline | Method 6 (research model) |
| Relative-depth layer | Depth Anything V2 Small (the Hugging Face Space for the web app, ONNX in the desktop app) | frozen, not fine-tuned |

The Maxar crops' 3D mesh uses a cosmetic display surface (flattened ground, boosted objects) so that buildings and trees
are visible; statistics and readouts show the real elevations.

**Relative depth is not elevation.** Depth Anything V2 orders pixels near-to-far and has no metres. On nadir satellite
imagery it does not reliably track elevation, so it is shown only as a labelled visualisation.

**At 10 m (Sentinel-2) no model added value over a plain DEM against independent lidar.** That is why every
Sentinel-2 surface in the product is a plain DEM (FABDEM or GLO-30), as in the table above. The evidence is summarised in [VALIDATION.md](VALIDATION.md).

## Method 6 (research model, used only for the Maxar crops)

- **What it is:** a full fine-tune of Depth Anything V2 Small with a twin mean/variance head, a height-balanced loss
  and height-balanced sampling, predicting height above ground in metres.
- **Training:** 512 × 512 quadrants of the 50 DFC2019 tiles, reflect-padded to 518, ImageNet normalisation, no
  augmentation, a fixed 12 epochs with the final weights kept (no selection on validation data).
- **Claim:** validated on DFC2019 (US cities, satellite imagery); did not generalize to GAMUS by RMSE.
- **Expected accuracy** (3-seed cross-validation on DFC2019, 4 held-out spatial quadrants): MAE 1.990 ± 0.010 m,
  RMSE 3.504 ± 0.026 m, Pearson 0.743 ± 0.002, Spearman 0.656 ± 0.0003.
- **Known limit:** it compresses tall objects. On leaf-on US forest its canopy p95 was 10.6 m against 37.7 m from
  airborne lidar. The app labels the Maxar surface as a research model that under-states tall canopy.
- **Production weights:** one model trained on all 50 DFC2019 tiles with the adopted recipe. A pre-registered test on
  2,848 unseen GAMUS tiles found it behaves like the cross-validation fold models (it passed on all three cities).
  The weights are not published.

## Other code in `backend/`

`backend/rdah/`, `backend/lora/`, `backend/calibration/` and `backend/training/` hold earlier research approaches or
stubs. None of them is on the product's path.
