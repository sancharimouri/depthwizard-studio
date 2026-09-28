# Prompt 2 — locating DFC2019 tiles in the US3D point clouds

Script: `scripts/dfc2019_locate_tiles.py` (parse → mosaic → selftest → match). All outputs,
including every coordinate, go to the gitignored `data/dfc2019/terrain_packs/locate/`. No
coordinates are written in this doc, the manifest, the catalog or any committed file.

## Method (as built)
- **Cloud rasters.** Each US3D cloud tile (512 m square, ~1.8 pts/m²) is binned at 0.6 m:
  DSM = max Up of all returns; ground = min Up of **class-2 (ground) returns**.
  - The clouds carry per-point classes (US3D README). The brief assumed none, so ground comes from
    labelled returns rather than a high-pass filter (judgment call).
  - 0.6 m, not 0.3 m: at this point density a 0.3 m raster is ~85 % empty cells. The GSD scan is done by
    resampling the DFC2019 tile, so 0.20–0.35 m/px is still searched.
- **City mosaic.** Every cloud tile of a city is placed by its own UTM extent, so tiles straddling
  a cloud-tile boundary can match. Cloud AGL = DSM − ground. Ground is the min of ground returns at
  4.8 m, nearest-filled, min-filtered and smoothed. DSM holes are nearest-filled inside coverage,
  followed by a 3×3 median.
- **Search.** FFT normalised cross-correlation of the tile's AGL (clipped at 0) against the whole
  same-city mosaic:
  - coarse at 2.4 m/px;
  - GSD 0.20–0.35 in 4 % geometric steps (15 values);
  - all 8 dihedral orientations;
  - 5 separated peaks kept per map; windows with < 90 % cloud coverage or near-zero variance cannot score.
- **Refinement.** At 0.6 m/px, ±6 % GSD in 1 % steps, best orientation, within the coarse peak's neighbourhood.
- **Runner-up.** The best candidate anywhere in the search (any GSD, any orientation) whose centre is
  > 100 m from the best candidate's centre.
- **Height scale.** Through-origin least-squares slope of cloud AGL on tile AGL over the refined
  window, using pixels where either exceeds 2 m.

## Pre-registered rules (committed before any real DFC2019 tile is matched)
**Matcher validity. Both must pass, or no real result is trusted:**
- **V1 self-test.** Synthetic crops from each city's mosaic: random GSD 0.20–0.35, random dihedral
  transform, height ×0.9–1.1, σ = 2 px blur, 0.5 m noise, ten 100 px dropout blocks.
  - A case passes with position error ≤ 30 m, GSD within 3 % and the exact orientation.
  - **V1 passes if ≥ 90 % of cases pass** (≥ 12 cases, both cities).
- **V2 negative control.** All 26 JAX tiles are matched against the OMA pool, and all 24 OMA tiles
  against the JAX pool. **V2 passes if 0 of 50 are "confident"** under the rule below.

**Confident match. All three are required:**
- **C1 peak.** Refined NCC ≥ 0.50.
- **C2 separation.** Coarse best − runner-up ≥ 0.15 **and** runner-up ≤ 0.8 × coarse best.
- **C3 height scale.** Fitted height scale in [0.80, 1.25].

Anything else is reported as unmatched, with the failing criterion. No best guess is forced.

**Reported, not gating:**
- Whether the matched centre falls inside the cloud tile whose index equals the tile name's first
  number (e.g. `JAX_004_*` → `JAX_Tile_004`). US3D names tiles `<city>_<tile index>_<image index>`,
  so a match there is an independent check.
- Agreement between tiles that share an index.

## Results
*(filled in after the runs; the sections above are not edited)*
