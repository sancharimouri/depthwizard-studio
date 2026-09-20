# Backbone terrain-failure diagnostics (DAv2 vs DINOv3)

Follow-up to `backbone-comparison.md`. Four targeted diagnostics on the
existing per-tile correlation results (`dav2_correlation_per_tile.csv`,
`dinov3_correlation_per_tile.csv`) — no new inference run. Reuses land-cover
and elevation stats from the earlier Sentinel-2 content-quality audit.

**Data-integrity fix applied first:** `content_audit.csv` is stale — it
predates two remediation rounds that swapped 4 of the 32 tiles (shimla
replaced shillong, bhitarkanika replaced vedaranyam, kurnool replaced
sangrur, vembanad replaced chilika — see `remediation_round2.csv` /
`remediation_round2c.csv`). Joining directly against `content_audit.csv`
silently drops those 4 tiles or mismatches them against their *predecessor*
tiles' stats. Built a corrected 32-tile table
(`data/sentinel2_benchmark/content_audit_corrected_32.csv`) pulling the
replacement tiles' real cropland/tree/builtup values from the remediation
CSVs before running any of the diagnostics below. (Two of the four replaced
tiles — kurnool, vembanad — don't have elevation stats in `remediation_round2c.csv`;
irrelevant to diagnostic 2 since neither is in the tiles being examined there.)

## Summary — what actually held up vs. what didn't

| Hypothesis | Result |
|---|---|
| DINOv3 correlates negatively with cropland%, positively with tree% | **Not supported.** Direction matches weakly but neither is remotely significant (n=32) |
| DAv2 shows no vegetation relationship | **True, but so does DINOv3** — this isn't a point of difference between the two backbones |
| DAv2's worst tiles cluster at either flatness extreme | **Not supported as a clean relationship** (overall r≈0.10, p=0.60). karnal is genuinely flat; shimla/nainital/ooty are high-relief but still fail, while equally-high-relief manali/dharamshala succeed strongly |
| karnal = false-gradient (flat cloud), shimla = topology inversion (clean negative slope) — two different mechanisms | **Not supported.** Both show a clean, coherent *negative*-sloped trend. Same mechanism (genuine sign inversion), not two different ones, and it isn't gated by whether the terrain is flat |
| Cross-tile scale-inconsistency (not noise) explains why pooling collapses correlation in DAv2-urban / DINOv3-coastal | **Strongly supported**, with a mechanism caveat (see Diagnostic 4) |
| Held-out per-tile calibration confirms diagnostic 4 wasn't an overfitting artifact | **Confirmed** — held-out numbers match non-held-out to 3 decimals (Follow-up 1) |
| Per-tile calibration "fixes" all 32 tiles pooled, both backbones, even DAv2's sign-inverted ones | **True, but this measures tile-identity separation, not signal quality** — DAv2's simple per-tile average (+0.41) actually beats DINOv3's (+0.25) despite DAv2's worse pooled/raw numbers (Follow-up 1) |
| Slope-aspect (solar exposure) or built-up% separates shimla/nainital/ooty (fail) from manali/dharamshala (succeed) | **Not supported** — no separation on aspect, slope, south-facing%, or built-up% (Follow-up 2) |

## Diagnostic 1 — vegetation cover vs. per-tile correlation

| | vs. cropland% | vs. tree% |
|---|---:|---:|
| DINOv3 pearson | r=-0.0447, p=0.808 | r=+0.0542, p=0.768 |
| DAv2 pearson | r=-0.1464, p=0.424 | r=-0.0405, p=0.826 |

Neither relationship is statistically significant at n=32 for either
backbone. DINOv3's signs point the hypothesized direction (negative with
cropland, positive with tree cover) but the effect is indistinguishable
from noise. DAv2 shows no relationship either, in either direction. The
original framing — "DINOv3 fails on vegetation, DAv2 fails on
relief/flatness instead" — is not something this data can support as a
clean split; if anything, neither backbone's per-tile sign is well
explained by land-cover composition alone at this sample size.

## Diagnostic 2 — elevation variance vs. DAv2 per-tile correlation

Overall: r=+0.0992, p=0.602 (n=30 tiles with elevation stats) — no
significant linear relationship between terrain flatness/relief and DAv2's
correlation sign or strength.

| tile | pearson | elev_std_m | elev_range_m |
|---|---:|---:|---:|
| karnal | -0.618 | 3.4 (flattest tile in the set) | 36 |
| shimla | -0.609 | 210.0 | 1,170 |
| nainital | -0.242 | 335.0 | 1,735 |
| ooty | -0.096 | 122.3 | 737 |

karnal is genuinely the flattest tile in the whole 32-tile set (lowest
elev_std of any tile with data) — consistent with a flatness-driven
failure *for that one tile specifically*. But shimla, nainital, and ooty
all have substantial real relief and still fail, while two *other*
comparably- or more-rugged hilly tiles succeed strongly: dharamshala
(elev_std=452.8, pearson=+0.729) and manali (elev_std=541.8, pearson=+0.905,
the best tile in the entire benchmark). Elevation variance alone does not
separate DAv2's successes from its failures — something more specific to
shimla/nainital/ooty (possibly ridge-town development pattern, aspect,
illumination/shadow geometry, or something else not captured by elev_std)
differentiates them from manali/dharamshala despite similar terrain
roughness. This is flagged as an open question, not resolved here.

## Diagnostic 3 — karnal vs. shimla scatter plots

`diagnostics/karnal_dav2_vs_icesat2_scatter.png`,
`diagnostics/shimla_dav2_vs_icesat2_scatter.png` (same style as this
project's earlier DFC2019 `dav2_vs_agl_scatter.png` diagnostic: x=DAv2
relative depth, y=true ground height, 100K-point random sample, alpha=0.15).

**Both tiles show a clean, coherent negative-sloped trend — neither shows
a flat, structureless scatter cloud.** karnal's trend spans a narrow real
range (~198–222m, consistent with its flat terrain) but the slope is
unambiguous and consistent across the whole depth range, not a shapeless
blob. shimla's trend spans a much larger range (~1,250–2,400m) with more
visible along-track streak structure (individual ICESat-2 passes), but the
same clean negative slope. This directly contradicts the
"false-gradient-on-flat-terrain vs. genuine-topology-inversion" dichotomy
as two *different* failure modes — visually, karnal and shimla look like
the *same* mechanism (a consistent inverse relationship between DAv2's
depth output and true elevation) operating at two very different absolute
elevation scales, not two distinguishable patterns. Whatever causes DAv2 to
invert sign on these tiles isn't something the flat-vs-relief framing
predicts correctly.

## Diagnostic 4 — does per-tile rescaling recover the pooled signal?

Tested on DAv2's urban tiles (all 8 individually positive, pearson
+0.05 to +0.77, yet pooled only +0.105) and DINOv3's coastal tiles (all 8
individually positive, +0.21 to +0.55, yet pooled *negative*, -0.095).
For each tile, fit an OLS line (true height ~ a·depth_value + b) using
that tile alone, replace each photon's depth value with the fitted
prediction, then pool the rescaled values across the 8 tiles and
recompute the pooled correlation against true height.

| subset | raw pooled pearson | simple avg of per-tile pearson | **rescaled pooled pearson** |
|---|---:|---:|---:|
| DAv2 urban (8 tiles) | +0.1054 | +0.4069 | **+0.9989** |
| DINOv3 coastal (8 tiles) | -0.0951 | +0.3306 | **+0.9670** |

The rescaled pooled correlation doesn't just recover the per-tile
average — it goes well past it, in both cases. **This confirms the
cross-tile scale-inconsistency explanation, not "urban/coastal are just
noisy tiles"**: every individual tile has real, positive within-tile
signal; raw pooling destroys it because each tile's depth-to-height
mapping has its own slope and offset (e.g. DAv2 urban OLS intercepts range
from -95 (kochi_city) to +802 (bengaluru) — i.e. very different baseline
elevations map to the same raw [0,1] depth range across tiles), so pooling
raw values mixes physically incomparable numbers.

**Caveat on the magnitude, so this isn't over-read as a cleaner result
than it is:** the rescaled pooled correlation exceeding even the per-tile
average (rather than landing "close to" it) is expected, not a sign the
fix did better than hoped — fitting a per-tile intercept `b` necessarily
snaps each tile's rescaled cluster onto roughly its own true mean
elevation, so once pooled, a large share of that near-1.0 correlation is
being driven by correctly separating tiles' *between-tile* elevation
differences (e.g. hyderabad's ~400m+ baseline vs. kochi_city's near-sea-
level baseline), not purely by within-tile predictive accuracy. It's still
the right diagnostic for the question asked (raw pooling fails from
scale/offset mismatch, not from an absence of real signal), but the exact
number shouldn't be read as "DAv2/DINOv3 would achieve ~0.98 pooled
correlation with a trivial per-tile calibration" — that would require
holding out the fit from the evaluation set, which this diagnostic
intentionally does not do (it's testing the *mechanism*, not proposing a
deployable calibration).

---

## Follow-up 1 — held-out per-tile calibration (an honest number, not just a mechanism test)

Diagnostic 4 above deliberately fit and evaluated on the same photons —
useful for confirming the *mechanism*, but not a real estimate of what
per-tile calibration buys you. Repeated properly: for each tile, fit the
OLS `a, b` on a random 50% of its photons (seed 42), evaluate the pooled
correlation on the **other**, held-out 50%.

### DAv2-urban and DINOv3-coastal (the two confirmed cases)

| subset | held-out pooled pearson | held-out pooled spearman | (diagnostic 4's non-held-out pearson, for comparison) |
|---|---:|---:|---:|
| DAv2 urban (8 tiles) | +0.9989 | +0.9814 | +0.9989 |
| DINOv3 coastal (8 tiles) | +0.9668 | +0.9683 | +0.9670 |

**Essentially identical to the non-held-out version in both cases** (to 3
decimal places). This matters: it confirms diagnostic 4's near-perfect
result was not an overfitting artifact — with 68K–1.06M photons per tile,
fitting 2 parameters (`a, b`) leaves no meaningful room to overfit. The
mechanism (cross-tile scale/offset inconsistency, not within-tile noise)
holds up under honest evaluation.

### All 32 tiles, both backbones

| backbone | held-out pooled pearson | held-out pooled spearman | simple avg of held-out per-tile pearson | (original, uncalibrated pooled pearson/spearman, for reference) |
|---|---:|---:|---:|---:|
| DAv2 | +0.9937 | +0.9944 | +0.4083 | +0.0403 / -0.0484 |
| DINOv3 | +0.9797 | +0.9942 | +0.2511 | +0.3037 / +0.3490 |

By category (held-out, all 32):

| category | DAv2 pearson/spearman | DINOv3 pearson/spearman |
|---|---:|---:|
| agricultural | +0.9987 / +0.9717 | +0.9985 / +0.9635 |
| coastal | +0.9624 / +0.9630 | +0.9671 / +0.9684 |
| hilly | +0.9525 / +0.9071 | +0.8358 / +0.8515 |
| urban | +0.9989 / +0.9813 | +0.9987 / +0.9802 |

**This result needs the same magnitude caveat as diagnostic 4, more
sharply.** Across all 32 tiles the between-tile elevation spread is huge
— from near-sea-level coastal tiles to Himalayan tiles above 4,000m — so
the per-tile intercept alone does almost all of the "work" once pooled;
this is why even **DAv2**, which includes 3 genuinely sign-inverted tiles
(karnal -0.62, shimla -0.61, nainital -0.24 raw pearson), still reaches a
+0.9937 held-out pooled correlation: an OLS fit correctly anchors each
tile — including the inverted ones — near its own true elevation, and
that placement dominates pooled variance regardless of whether the
within-tile relationship was good, weak, or backwards. **This number
answers "can a per-tile fit correctly separate which tile a photon came
from," not "how good is each backbone's within-tile signal."**

The more informative number here is the **simple average of held-out
per-tile pearson**: DAv2 +0.4083 vs. DINOv3 +0.2511 — DAv2 is actually
*higher* on this metric, despite having the worse raw pooled correlation
and the sign-instability problem the whole comparison is centered on. This
complicates the "DINOv3 wins" headline from `backbone-comparison.md`
further, in a different way than the category breakdown already did: by
per-tile-average quality (ignoring pooling entirely), DAv2's good tiles are
apparently *better* than DINOv3's good tiles, and DAv2's few very-bad tiles
don't drag the per-tile average down as much as DINOv3's broader spread of
mediocre-to-negative tiles does. Also notable: DINOv3's `manali` per-tile
correlation is *negative* (-0.139) — the single tile DAv2 does best on
(+0.905, the best tile in the whole benchmark) — reinforcing that neither
backbone's within-tile strengths and weaknesses line up with the other's.

**Bottom line on calibration:** per-tile calibration is a genuinely real,
held-out-validated fix for the *pooling* problem specifically (both
backbones), but it is not evidence that either backbone's raw relative-
depth signal is more reliable than the pooled numbers suggested — the
near-1.0 pooled numbers here are answering a different, easier question
(tile identity) than the original comparison was asking (elevation
signal quality).

## Follow-up 2 — does slope-aspect or built-up% separate shimla/nainital/ooty (fail) from manali/dharamshala (succeed)?

Elevation variance (diagnostic 2) didn't separate these two groups.
Tested whether real terrain aspect (slope direction, hence exposure to
solar illumination) or built-up% does instead.

**Data:** OpenTopography's SRTMGL1 API (used for the original elev_std_m
audit) had hit its 50-calls/24hr rate limit, so this used Copernicus
GLO-30 DEM directly from its public AWS Open Data bucket
(`copernicus-dem-30m.s3.amazonaws.com`, no auth/key needed, no rate
limit) — same underlying global 30m DEM product family already used
elsewhere in this project for Kolkata/Bardhaman/Sundarbans. Each of the 5
tiles' bbox fell within a single 1°×1° Copernicus tile (no mosaicking
needed). Slope/aspect computed via `np.gradient` on the elevation array
using real per-pixel ground spacing (~26–31m depending on tile latitude),
standard GIS aspect convention (0°=N, 90°=E, 180°=S, 270°=W). Aspect
statistics computed only over pixels with slope > 2° (near-flat pixels
have essentially undefined/noisy aspect). India's latitude band here
(11–32°N, all north of the 23.5° solar-declination limit) means "south-
facing" is a safe, date-independent proxy for "more sun-exposed" at these
5 tiles. built-up% is from the corrected WorldCover audit above.

| tile | DAv2 result | mean slope (°) | slope-weighted circular mean aspect | south-facing % (135–225°) | north-facing % (315–45°) | built-up % |
|---|---|---:|---:|---:|---:|---:|
| shimla | FAIL (-0.609) | 29.6 | 104.1° (ESE) | 26.0 | 22.3 | 6.22 |
| nainital | FAIL (-0.242) | 29.4 | 179.8° (S) | 27.9 | 23.8 | 1.92 |
| ooty | FAIL (-0.096) | 14.7 | 110.7° (ESE) | 28.0 | 25.2 | 8.46 |
| manali | SUCCEED (+0.905) | 29.3 | 170.4° (S) | 22.5 | 20.7 | 2.58 |
| dharamshala | SUCCEED (+0.729) | 19.7 | 144.1° (SE) | 42.3 | 7.6 | 7.49 |

**Not supported — no separation on any of the four measures.** Circular
mean aspect: nainital (FAIL, 179.8° due-south) is nearly identical to
manali (SUCCEED, 170.4° south) — the two most-different-outcome tiles have
nearly the same dominant aspect. Mean slope: shimla (FAIL, 29.6°) and
manali (SUCCEED, 29.3°) are essentially the same steepness. South-facing%:
dharamshala (SUCCEED) has the *highest* value (42.3%) of all 5 tiles, but
manali (SUCCEED) has the *lowest* (22.5%), lower than all three FAIL
tiles. Built-up%: ooty (FAIL, 8.46%) and dharamshala (SUCCEED, 7.49%) are
nearly tied and both well above nainital (FAIL, 1.92%) and manali
(SUCCEED, 2.58%), which are themselves nearly tied with each other.

This is a second clean negative result (after diagnostics 1 and 2):
vegetation cover, elevation variance, slope-aspect/solar-exposure, and
built-up% each fail to separate DAv2's per-tile successes from its
failures among these 5 hilly tiles. Whatever actually drives the sign
flips on shimla/nainital/ooty (vs. success on manali/dharamshala) is not
one of the terrain/land-cover covariates checked across this whole
diagnostic pass. Candidates not yet tested: acquisition-date-specific
shadow/illumination conditions (not captured by a static aspect
calculation), seasonal snow cover in these Himalayan-belt tiles, or
scene content DAv2's training distribution doesn't generalize to well —
flagged as open, not resolved.
