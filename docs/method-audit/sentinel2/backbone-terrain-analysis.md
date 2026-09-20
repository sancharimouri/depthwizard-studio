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
