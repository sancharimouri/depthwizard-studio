# Frozen-backbone comparison: DAv2 vs DINOv3 (SAT493M + CHMv2)

**Status: IN PROGRESS — DAv2 side complete, DINOv3 side blocked on gated model access.**

Goal: decide which frozen backbone (no calibration/fine-tuning) correlates better
with real elevation, on real Sentinel-2 + real ICESat-2 ground truth, across the
32-tile `data/sentinel2_benchmark/` set (8 tiles each: agricultural, coastal,
hilly, urban). This mirrors Method 1's original DFC2019 baseline test, but on a
different (real satellite-photon, not DFC2019 airborne-lidar) ground truth source.
Winner becomes the frozen prior for every method going forward.

## Step 0 — real per-photon ICESat-2 ground truth

Re-queried ICESat-2 ATL08 with sliderule (same PhoREAL block as the earlier
coverage-only script), but this time via `icesat2.atl03sp` with
`atl08_class=['atl08_ground']`, which returns **individual ATL03 photons**
already filtered server-side to `atl08_class == 1` ("ground") — i.e. the
ATL08 ground-classification lineage, at native photon resolution, not the
100m PhoREAL segment aggregates the coverage script used.

- **Field used for height:** `height` column from sliderule's `atl03sp` output
  — this is ATL03 `h_ph` (photon ellipsoidal height), restricted to photons
  ATL08 classified as ground (`atl08_class == 1`). CRS: **EPSG:7912**
  (ITRF2014, geographic 3D) — treated as WGS84 for reprojection purposes;
  the ITRF2014/WGS84 divergence is centimeter-level and negligible against
  10m Sentinel-2 pixels.
- Query window: 2019-01-01 to 2025-12-31, land surface type only, tile bboxes
  taken from `manifest.csv`'s `bbox_utm` (reprojected to WGS84).
- Script: `scripts/query_icesat2_photons.py`. Output: one CSV per tile under
  `data/icesat2_photons/<tile_id>.csv` (not committed — regenerable, and the
  full set is several GB; largest single tile is ~1.2M photons / 74MB).

### Per-tile usable photon counts

No tile fell below the 500-photon low-count flag threshold. Lowest counts
cluster in **hilly** terrain (canopy/steep-slope obstruction reduces ATL08
ground-classification success) — smallest is `dehradun` at 24,295, still
enough for a stable per-tile correlation. All others range from ~40K to
~1.2M photons.

| tile | category | photons |
|---|---|---|
| dehradun | hilly | 24,295 |
| shimla | hilly | 41,431 |
| nainital | hilly | 43,424 |
| dharamshala | hilly | 56,895 |
| delhi | urban | 69,005 |
| kohima | hilly | 75,202 |
| ooty | hilly | 102,110 |
| almora | hilly | 122,645 |
| mumbai | urban | 137,231 |
| bengaluru | urban | 141,799 |
| pune | urban | 158,255 |
| manali | hilly | 169,513 |
| kurnool | agricultural | 220,520 |
| hyderabad | urban | 236,177 |
| chennai | urban | 281,071 |
| vembanad | coastal | 325,409 |
| jaipur | urban | 331,373 |
| karnal | agricultural | 353,137 |
| kochi_city | urban | 374,118 |
| bhitarkanika | coastal | 434,455 |
| vidisha | agricultural | 503,839 |
| hisar | agricultural | 552,653 |
| fatehpur | agricultural | 560,011 |
| kakinada | coastal | 586,656 |
| nagapattinam | coastal | 587,696 |
| kota | agricultural | 624,926 |
| nizamabad | agricultural | 721,743 |
| digha | coastal | 727,241 |
| goa_estuary | coastal | 728,750 |
| amalapuram | coastal | 1,060,531 |
| kutch | coastal | 1,062,057 |
| bathinda | agricultural | 1,194,809 |

**Important caveat, applies to every correlation below:** ICESat-2's absolute
geolocation accuracy is ~6.5m against Sentinel-2's 10m pixels. This injects
real positional noise into every photon-to-pixel sample here, independent of
backbone quality — a modest correlation is not necessarily a weak prior, it
may partly reflect this ground-truth positional uncertainty, and differences
smaller than that noise floor should not be over-read.

## Step 1.1 — DAv2 (complete)

Reused the existing, already-validated DAv2 batch pipeline
(`backend/depth/depth_engine.py` via `scripts/run_dav2_batch.py`) unmodified
— same code path used for the DFC2019 benchmark and the earlier RS3DAda
smoke test. Ran frozen DAv2 on all 32 tiles (mean 0.71s/tile on this
machine's DAv2 device).

Photon-sampling procedure (`scripts/lib_backbone_correlation.py`, shared
with the DINOv3 run for methodological parity): reproject each tile's
ICESat-2 ground photons from WGS84 into the tile's UTM CRS, then into pixel
space via the source GeoTIFF's affine transform (same pixel grid DAv2 ran
on — DAv2 output is per-pixel same-resolution as the source), sample the
depth raster at each photon's pixel, compute per-tile and pooled Pearson +
Spearman against photon height.

**Sign convention:** DAv2's relative-depth output is larger-value=nearer
(matches "inverse depth"/disparity convention). In nadir satellite view,
nearer-to-sensor = higher elevation, so a well-behaved correlation should be
*positive*. Correlations reported below are raw (not sign-corrected) so a
negative value is a real methodological signal, not a labeling artifact.

### Per-tile results

| tile | category | n | pearson | spearman |
|---|---|---:|---:|---:|
| bathinda | agricultural | 1,194,195 | +0.2118 | +0.3065 |
| fatehpur | agricultural | 559,803 | -0.1544 | -0.1578 |
| hisar | agricultural | 552,384 | -0.2123 | -0.3785 |
| karnal | agricultural | 352,982 | -0.6176 | -0.6024 |
| kota | agricultural | 624,582 | +0.5305 | +0.3813 |
| kurnool | agricultural | 220,429 | +0.7042 | +0.7440 |
| nizamabad | agricultural | 721,319 | +0.1825 | +0.2022 |
| vidisha | agricultural | 503,498 | +0.1408 | +0.1313 |
| amalapuram | coastal | 1,060,205 | +0.4303 | +0.4351 |
| bhitarkanika | coastal | 434,280 | +0.2708 | +0.1553 |
| digha | coastal | 727,088 | +0.6286 | +0.7285 |
| goa_estuary | coastal | 728,473 | +0.2462 | +0.1782 |
| kakinada | coastal | 586,309 | +0.7611 | +0.8511 |
| kutch | coastal | 1,061,324 | +0.1979 | +0.2072 |
| nagapattinam | coastal | 587,493 | +0.6475 | +0.7666 |
| vembanad | coastal | 325,335 | +0.1224 | +0.0641 |
| almora | hilly | 122,628 | +0.3012 | +0.1804 |
| dehradun | hilly | 24,273 | +0.6096 | +0.6002 |
| dharamshala | hilly | 56,878 | +0.7295 | +0.7654 |
| kohima | hilly | 75,182 | +0.2474 | +0.1277 |
| manali | hilly | 169,397 | +0.9051 | +0.8867 |
| nainital | hilly | 43,410 | -0.2424 | -0.2398 |
| ooty | hilly | 102,095 | -0.0962 | -0.0093 |
| shimla | hilly | 41,423 | -0.6093 | -0.6168 |
| bengaluru | urban | 141,770 | +0.5220 | +0.4927 |
| chennai | urban | 280,990 | +0.6673 | +0.7105 |
| delhi | urban | 68,961 | +0.7728 | +0.8837 |
| hyderabad | urban | 236,072 | +0.1420 | +0.1256 |
| jaipur | urban | 331,252 | +0.4867 | +0.5199 |
| kochi_city | urban | 373,853 | +0.0534 | +0.1154 |
| mumbai | urban | 137,172 | +0.4528 | +0.4839 |
| pune | urban | 158,228 | +0.1580 | +0.2599 |

Full CSV: `data/sentinel2_benchmark/dav2_depth/dav2_correlation_per_tile.csv`.

### Pooled results (all photons across all tiles, not an average of per-tile r)

| scope | n | pearson | spearman |
|---|---:|---:|---:|
| **pooled (all 32 tiles)** | 12,603,283 | **+0.0403** | **-0.0484** |
| agricultural | 4,729,192 | +0.3367 | +0.4448 |
| coastal | 5,510,507 | -0.1856 | -0.1454 |
| hilly | 635,286 | +0.3422 | +0.2215 |
| urban | 1,728,298 | +0.1054 | +0.2081 |

### Reading these numbers

- Per-tile correlations swing from strongly positive (manali +0.91, kakinada
  +0.76, delhi +0.77) to clearly negative (karnal -0.62, shimla -0.61) —
  **sign instability across tiles is the dominant pattern**, consistent with
  the project's earlier finding that DAv2's raw relative-depth signal is not
  a stable elevation proxy once you look beyond a single scene (the
  Darjeeling correlation-flip finding this project's demo framing is built
  on).
- The pooled Pearson/Spearman are both near zero (+0.04 / -0.05) — pooling
  across tiles washes out the per-tile signal because the sign itself isn't
  consistent, not just the magnitude.
- **coastal is the only category with a negative pooled correlation**
  (-0.19 / -0.15) despite several individual coastal tiles scoring strongly
  positive (kakinada +0.76, nagapattinam +0.65, digha +0.63) — this is
  itself informative: coastal tiles mix very-low-relief terrain (where
  photon-sampling noise dominates any real signal) with a few tiles that
  have real bathymetry/topography contrast, and pooling amplifies the
  disagreement between them rather than averaging it out.

## Step 1.2 — DINOv3 (blocked)

Identified Meta's actual intended tool for a satellite-domain monocular
height probe: `dinov3_vitl16_chmv2` (from `facebookresearch/dinov3`,
vendored unmodified at `external/dinov3/`) — the **SAT493M** (MAXAR-pretrained)
backbone paired with Meta's own official **CHMv2** (Canopy Height Model v2)
DPT decoder head. This is a better fit than the generic SYNTHMIX depth head
(which is only released for the general-purpose LVD1689M backbone): CHMv2 is
itself a satellite-imagery absolute-height regression head, pretrained by
Meta, loaded and run frozen with no training and no custom readout, per the
task's methodology (`scripts/run_dinov3_batch.py`).

**Blocker:** the SAT493M backbone weights are Meta-license-gated —
`facebook/dinov3-vitl16-pretrain-sat493m` on Hugging Face shows
`gated: manual` (requires Meta's manual approval, not instant), and the
direct `fbaipublicfiles.com` CDN URL 403s without prior access. (The CHMv2
head weights themselves, 135MB, are public — only the backbone is gated.)
Confirmed the rest of the pipeline runs cleanly up to that download step.

Waiting on backbone access (HF approval or a signed download URL) before
Step 1.2 can run. `scripts/run_dinov3_batch.py --backbone-weights <path-or-url>`
is ready to go once access is available; `scripts/run_dinov3_correlation.py`
reuses the exact same photon-sampling/correlation module as DAv2 for a
directly comparable result.

## Step 1.3 — comparison and decision (pending)

Cannot be completed until Step 1.2 runs. Decision rule (stated in advance,
per the task spec): the backbone with higher **pooled** Pearson AND Spearman
across all 32 tiles wins outright; if split, report both explicitly and
break the tie by whichever backbone wins in more of the 4 terrain
categories.

DAv2's pooled numbers to beat: **Pearson +0.0403, Spearman -0.0484** — note
these already disagree in sign, so DAv2 may itself force a "split" outcome
under the stated decision rule regardless of what DINOv3 scores; if DINOv3
also splits, or if DINOv3 is unambiguously positive on both while DAv2 is
mixed, resolve per the category tie-break as specified.
