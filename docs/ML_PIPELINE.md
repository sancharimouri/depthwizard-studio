# ML pipeline

## What the shipped product uses

| Surface | Source | Model involved |
|---|---|---|
| Terrain, Sentinel-2 library tiles | FABDEM bare earth, with a per-tile linear calibration to ICESat-2 ground photons | none |
| Surface, Sentinel-2 and searched scenes | Copernicus GLO-30 | none |
| Relative-depth layer | Depth Anything V2 (Large in the backend, Small in the Space and desktop app) | frozen, not fine-tuned |
| Surface, the 6 Maxar VHR library crops | FABDEM + max(predicted height above ground, 0) | Method 6 (research model) |

**Relative depth is not elevation.** Depth Anything V2 orders pixels near-to-far and has no metres. On nadir satellite
imagery it does not reliably track elevation, so it is shown only as a labelled visualisation.

**At 10 m (Sentinel-2) no model added value over a plain DEM against independent lidar.** That is why the product's
terrain is calibrated FABDEM and its surface is GLO-30. The evidence is summarised in [VALIDATION.md](VALIDATION.md).

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
  The weights are not published: DFC2019's contest terms restrict redistribution.

## Other code in `backend/`

`backend/rdah/`, `backend/lora/`, `backend/calibration/` and `backend/training/` hold earlier research approaches or
stubs. None of them is on the product's path.
