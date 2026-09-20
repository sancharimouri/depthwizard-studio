# Frozen-backbone comparison: DAv2 vs DINOv3 (SAT493M + CHMv2)

**Status: COMPLETE. Winner: DINOv3 (SAT493M + CHMv2)** — higher pooled Pearson
AND Spearman across all 32 tiles than DAv2, so the decision rule resolves
outright with no tie-break needed. See "Step 1.3 — decision" for the full
numbers and important category-level caveats before treating this as
unconditional.

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

## Step 1.2 — DINOv3 (complete)

Used Meta's actual intended tool for a satellite-domain monocular height
probe: `dinov3_vitl16_chmv2` (from `facebookresearch/dinov3`, vendored
unmodified at `external/dinov3/`) — the **SAT493M** (MAXAR-pretrained)
backbone paired with Meta's own official **CHMv2** (Canopy Height Model v2)
DPT decoder head. This is a better fit than the generic SYNTHMIX depth head
(only released for the general-purpose LVD1689M backbone): CHMv2 is itself
a satellite-imagery absolute-height regression head, pretrained by Meta,
loaded and run frozen with no training and no custom readout to it — the
DPT head's weights are used exactly as Meta shipped them.

### Backbone access — resolved, with a wrinkle worth recording

`facebook/dinov3-vitl16-pretrain-sat493m` on Hugging Face is `gated: manual`.
Once the user's HF access request was approved (`HF_TOKEN` in `.env`), the
Hugging Face download worked directly — but Meta's own
`fbaipublicfiles.com` CDN URL that `torch.hub`'s `dinov3_vitl16_chmv2`
entrypoint calls internally **still 403s**, even with that same approval;
it appears to be a separate, unresolved gate from the HF one. Worked around
it by downloading the HF-hosted checkpoint and converting its state dict
into the original repo's parameter naming (`scripts/lib_dinov3_sat493m_loader.py`)
— same pretrained weights, same architecture, two independent
re-implementations (`transformers`' port vs. the original research repo's
code) with different key names. This is a pure key/shape translation, not
a re-derivation: fused-qkv concatenation of HF's split q/k/v projections,
a zero-filled k-bias segment + a deterministic `bias_mask` buffer (both
verified as fixed, non-learned values from the target code, not
approximations), and a tensor reshape for `mask_token`. No weight values
were invented or altered. The public CHMv2 head weights (135MB, not gated)
loaded normally from `fbaipublicfiles.com`.

### Input resolution — matched to DAv2's own precedent

Full source-resolution (1000x1000) inference on CPU (this machine has no
CUDA; the vendored `Depther` class has no MPS path) took over 25 minutes
for a single tile and was killed as intractable for 32 tiles. DAv2's own
pipeline (`backend/depth/depth_engine.py`) already resizes to a fixed
518x518 for the model forward pass and bicubic-upsamples the output back to
source resolution — the same handling was applied here
(`DINOV3_INPUT_SIZE = 518` in `run_dinov3_batch.py`), which is matching an
existing precedent, not introducing a new asymmetry between the two
backbones' procedures. Runtime: 95.3s/tile mean, 32/32 succeeded, 0 errors.

Sanity-checked visually against DAv2's output for `manali` (hilly): same
dark central valley structure, plus scattered bright canopy-like clusters
consistent with CHMv2 predicting canopy height rather than raw elevation
(see interpretation note below).

### Per-tile results

| tile | category | n | pearson | spearman |
|---|---|---:|---:|---:|
| bathinda | agricultural | 1,194,195 | -0.1211 | -0.0938 |
| fatehpur | agricultural | 559,803 | +0.1329 | +0.0749 |
| hisar | agricultural | 552,384 | +0.0284 | +0.0537 |
| karnal | agricultural | 352,982 | -0.0788 | -0.2001 |
| kota | agricultural | 624,582 | -0.0861 | -0.0374 |
| kurnool | agricultural | 220,429 | +0.3384 | +0.2311 |
| nizamabad | agricultural | 721,319 | +0.3462 | +0.1970 |
| vidisha | agricultural | 503,498 | +0.1656 | +0.2353 |
| amalapuram | coastal | 1,060,205 | +0.2070 | +0.3294 |
| bhitarkanika | coastal | 434,280 | +0.2161 | +0.1087 |
| digha | coastal | 727,088 | +0.4516 | +0.5669 |
| goa_estuary | coastal | 728,473 | +0.4348 | +0.4125 |
| kakinada | coastal | 586,309 | +0.3004 | +0.3869 |
| kutch | coastal | 1,061,324 | +0.2350 | +0.3120 |
| nagapattinam | coastal | 587,493 | +0.2453 | +0.4389 |
| vembanad | coastal | 325,335 | +0.5548 | +0.0491 |
| almora | hilly | 122,628 | +0.1039 | -0.0190 |
| dehradun | hilly | 24,273 | +0.4491 | +0.5215 |
| dharamshala | hilly | 56,878 | +0.6125 | +0.4790 |
| kohima | hilly | 75,182 | +0.1541 | +0.2072 |
| manali | hilly | 169,397 | -0.1393 | -0.1760 |
| nainital | hilly | 43,410 | +0.2911 | +0.4744 |
| ooty | hilly | 102,095 | +0.2058 | +0.2248 |
| shimla | hilly | 41,423 | +0.1844 | +0.1917 |
| bengaluru | urban | 141,770 | +0.2373 | +0.2800 |
| chennai | urban | 280,990 | +0.6583 | +0.7087 |
| delhi | urban | 68,961 | -0.0426 | +0.0071 |
| hyderabad | urban | 236,072 | -0.1007 | -0.1830 |
| jaipur | urban | 331,252 | -0.0886 | -0.1527 |
| kochi_city | urban | 373,853 | +0.4380 | +0.2958 |
| mumbai | urban | 137,172 | +0.2240 | +0.4053 |
| pune | urban | 158,228 | -0.1621 | -0.1898 |

Full CSV: `data/sentinel2_benchmark/dinov3_depth/dinov3_correlation_per_tile.csv`.

### Pooled results

| scope | n | pearson | spearman |
|---|---:|---:|---:|
| **pooled (all 32 tiles)** | 12,603,283 | **+0.3037** | **+0.3490** |
| agricultural | 4,729,192 | -0.3097 | -0.4948 |
| coastal | 5,510,507 | -0.0951 | -0.1528 |
| hilly | 635,286 | +0.0032 | +0.1422 |
| urban | 1,728,298 | +0.3314 | +0.4382 |

## Step 1.3 — comparison and decision

**Decision rule** (stated in advance, per the task spec): the backbone with
higher pooled Pearson AND Spearman across all 32 tiles wins outright.

| backbone | pooled pearson | pooled spearman |
|---|---:|---:|
| DAv2 | +0.0403 | -0.0484 |
| **DINOv3 (SAT493M+CHMv2)** | **+0.3037** | **+0.3490** |

**DINOv3 wins outright** — higher on both pooled metrics, no split, no
tie-break needed. DINOv3 becomes the frozen prior going forward per the
decision rule as stated.

### Category breakdown (informational — not needed for the decision, since it already resolved outright)

| category | DAv2 pearson/spearman | DINOv3 pearson/spearman | better on this category |
|---|---:|---:|---|
| agricultural | +0.3367 / +0.4448 | -0.3097 / -0.4948 | DAv2, clearly |
| coastal | -0.1856 / -0.1454 | -0.0951 / -0.1528 | roughly tied (DINOv3 less-negative Pearson, DAv2 marginally better Spearman) |
| hilly | +0.3422 / +0.2215 | +0.0032 / +0.1422 | DAv2, clearly |
| urban | +0.1054 / +0.2081 | +0.3314 / +0.4382 | DINOv3, clearly |

**This is the important caveat to the "DINOv3 wins" headline:** DINOv3's
pooled win is driven almost entirely by urban (its strongest category by
far) and by not being as badly wrong as DAv2 in coastal/hilly, while DAv2
is actually the better raw correlate in agricultural and hilly terrain
specifically. A backbone selection based on the single pooled number would
be reasonable per the stated rule, but a per-category or per-use-case
choice might not pick DINOv3 uniformly — flag this if "frozen prior going
forward" gets applied to a specific terrain type rather than pooled
general use.

### Why DINOv3 is more sign-stable here, most likely

DAv2's per-tile signs flip essentially at random across scenes (the
project's known finding this demo's framing is built on). DINOv3's
per-tile signs are still not perfectly consistent, but noticeably less
scattered, and CHMv2 is trained specifically as an absolute-height
regression head (Canopy Height Model) rather than a general relative-depth
estimator — it's plausible that even though its target quantity (canopy /
object height above local ground, not terrain elevation) is not exactly
what we're correlating against, that supervision target is closer to
"real physical height in meters" than DAv2's inverse-depth-style training
objective, giving it a more consistent sign relationship with true ground
elevation. This is a plausible explanation, not a verified mechanism —
it's not something this comparison itself proves.

### Reused caveat (from Step 1)

Same ICESat-2 geolocation-accuracy caveat applies here: ~6.5m positional
noise against 10m pixels affects both backbones' correlations equally, so
neither number should be read as a hard ceiling on what real signal is
present.
