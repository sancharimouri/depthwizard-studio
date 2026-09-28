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

Run 2026-09-28. Wall time: parse 36 s (806 cloud tiles), mosaics 64 s, then 40–180 s per tile match
(slower while Prompt 4 inference shared the machine).

### Validity
- **V1 self-test: PASS, 16/16.** Position error ≤ 0.4 m, GSD within 0.7 %, orientation exact, height
  scale recovered within ~3 %. Synthetic peaks 0.79–0.94 NCC; runner-ups 0.27–0.65.
- **V2 negative control: PASS, 0/50 confident.** The best wrong-city scores are high (fine NCC up to 0.706,
  coarse up to 0.735), so **C1 alone would not have separated them**; the joint rule did (C2 and/or C3
  failed on every one).

### Confident matches (pre-registered rule): **15 / 50: JAX 8 / 26, OMA 7 / 24**
Confident: JAX_004_006, JAX_031_006, JAX_118_012, JAX_165_015, JAX_175_002, JAX_214_023, JAX_416_022,
JAX_505_018; OMA_084_038, OMA_134_027, OMA_230_036, OMA_269_035, OMA_315_020, OMA_332_037, OMA_364_003.
Their locations are in the private `locate/locations.json` only.

Why each of the other 35 is unmatched (failing criteria):
| Failing criteria | Tiles |
|---|---|
| C3 only (height scale 0.58–0.80) | JAX_004_014, 004_016, 018_012, 022_009, 072_015, 118_015, 149_006, 161_001, 166_006, 204_005, 214_015, 269_009, 416_009, 505_016; OMA_211_032, 225_001, 315_019 |
| C2 only (runner-up too close) | OMA_248_029, 248_030, 376_023, 376_038 |
| C2 + C3 | JAX_149_025, 164_008, 224_025, 264_013; OMA_212_033, 221_034, 281_002, 281_030, 364_043 |
| C1 only | OMA_198_002 (fine NCC 0.397) |
| C1 + C3 | OMA_042_011 |
| C1 + C2 + C3 | OMA_211_039, OMA_258_020, OMA_144_030 (AGL max 1.1 m: nothing to match on) |

### Post-hoc observations (after seeing results; NOT used for any pack)
- **The index check is nearly perfect.** 49/50 matched centres (all but the flat OMA_144_030) fall in
  the cloud tile named by the tile's own index. Each city has ~400 cloud tiles, so chance agreement is
  ~1/400 per tile. The 14 groups of tiles sharing an index land 2–30 m apart. Most unmatched tiles
  are therefore very probably located correctly; **the 0/50 of the earlier study was the scale error.**
- **C3 fails for a systematic reason, not because the locations are wrong.** The fitted height scale is biased low
  (median 0.77 over all 50; 0.80–0.91 on the confident ones). This points to this matcher's cloud-AGL
  layer: the reduced cloud (~1.8 pts/m²), per-cell max, and a smoothed 4.8 m ground surface that
  rides up under dense blocks. The DFC2019 AGL is built from the full-density lidar. The pre-registered
  [0.80, 1.25] did not anticipate that bias.
- **Measured GSD:** 0.308–0.367 m (median ≈ 0.336) on the index-consistent matches. That is consistent
  with the 0.30–0.34 hand measurement.
- **Orientation:** 0 (north-up, as delivered) on 46/49 index-consistent tiles; 1 on JAX_161_001 and
  JAX_175_002, 4 on OMA_212_033.
- **Suggested follow-up (needs its own pre-registration; not done here):** a revised rule that gates on
  the index check plus C1/C2 and drops or re-centres C3. Validate it on the same negative control, then
  upgrade tiles from city-level to DEM-located with `dfc2019_base_ground.py located`.
