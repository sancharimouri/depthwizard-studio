# India Sentinel-2 Benchmark Dataset — Build Report

Real Sentinel-2 L2A RGB tiles (10x10km, 10m, 3-band uint8) across four terrain
categories, acquired via the CDSE Catalog + Process API, filtered for
ICESat-2 land-photon coverage. Separate from the four production regions
(Darjeeling, Kolkata, Bardhaman, Sundarbans) — this is a validation
benchmark, not a change to the live site.

## Result

**32/32 target tiles acquired — 8 per category, the top of the requested
6-8 range in all four.**

| Category | Candidates tried | Cleared ICESat-2 threshold | Downloaded | Kept |
|---|---|---|---|---|
| Hilly | 14 | 8 | 8/8 | 8 |
| Urban | 8 | 8 | 8/8 | 8 |
| Agricultural | 15 | 8 | 8/8 | 8 |
| Coastal | 15 | 8 | 8/8 | 8 |
| **Total** | **52** | **32** | **32/32** | **32** |

No category came up short — every category had enough candidates clear the
bar to hit 8. Nothing here needed forcing weak tiles in to hit a count: the
6 hilly, 7 agricultural, and 7 coastal candidates that were dropped all had
real (nonzero) ICESat-2 coverage, just less than the ones kept — this was a
genuine ranking, not a pass/fail cliff with borderline near-misses.

## Method

**1. Candidate selection (52 total, 44 new + reusing urban's original
scope).** Candidates were chosen by geographic reasoning, not arbitrarily:
Himalayan/Western-Ghats hill towns for "hilly" (Gangtok, Shimla, Manali,
Dehradun, Ooty, Munnar, etc.), other major built-up cities for "urban"
(Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, etc.), known agricultural
belts for "agricultural" (Punjab/Haryana wheat-rice belt, Andhra
cotton/rice, Malwa soybean plateau, Chambal command area), and delta/
estuary/backwater stretches for "coastal" (Mahanadi/Godavari/Krishna/Cauvery
deltas, Kerala backwaters, Gulf of Kutch, Konkan coast, Goa estuary). Full
list in `scripts/sentinel_benchmark_candidates.py`.

**2. ICESat-2 coverage scoring.** No prior sliderule/ICESat-2 integration
existed in this repo to reuse (checked first — `backend/icesat/*.py` were
empty stubs, and the only related doc,
`docs/method-audit/stage0-gates/sparse-lidar-feasibility.md`, explicitly
says a sliderule query was "not attempted" for this project before). Built
fresh: for each candidate, ran sliderule's ATL08 PhoREAL processing
(`icesat2.atl08p`) over a ~10x10km WGS84 polygon, 2019-01-01 to present,
land surface type only, with the `phoreal` block set (its absence silently
returns fill-sentinel values for the classified-photon fields — confirmed
by inspecting field ranges directly, not assumed). Yield metric: sum of
`gnd_ph_count` (ATL08-classified ground-photon returns) across all segments
with a valid (non-fill) `landcover` classification. All 52 candidates
returned real, valid coverage (script + full per-candidate numbers in
`data/sentinel2_benchmark/icesat2_coverage.csv`).

**Threshold chosen:** `gnd_ph_count >= 5,000` across the AOI. This is a low
bar (every one of the 52 candidates cleared it by a wide margin — the lowest
was ~29,000) chosen deliberately loose: ICESat-2's ground tracks are narrow
and widely spaced, so the real risk for a small AOI is near-zero coverage,
not "borderline" coverage. It turned out not to bind at all here — India's
candidate density meant no location came close to failing it — so category
ranking was effectively "top 8 by yield," not "however many clear a bar."

**3. Full-resolution acquisition.** No existing CDSE integration in this
repo pulled analytic GeoTIFFs either — `backend/cdse/client.py` only ever
requested 512x512 8-bit preview PNGs for the frontend's Scene Input modal.
Extended (not rewrote) that client: reused its OAuth2 token caching,
Catalog search, and true-color evalscript, and added a new Process API call
requesting `image/tiff` output in the tile's local UTM zone at exactly
1000x1000px/10m, matching the production regions' georeferencing style.
Scene selection matches the existing low-cloud/recent-date preference
(searches cloud <20% first, widening to 40%/80% only if nothing clears the
strict bar; none of the 32 selected tiles needed the wider bar — all found
sub-3% cloud scenes from 2025, one from April and two from October/November
2025).

## Data quality notes (reported honestly, not smoothed over)

- **Sangrur (agricultural) has ~21% nodata** at the tile edge — its 10x10km
  AOI lands near an MGRS granule boundary, so the Process API filled part of
  the requested extent from outside the available scene footprint. The
  other 31 tiles are 0-7% nodata (typically resampling edge effects). Worth
  a re-pull with a different reference date if this tile is used for
  anything precision-sensitive; left as-is here since 79% real coverage is
  still usable for a visual/qualitative benchmark.
- **Coarser three of the original four production regions' elevation caveat
  doesn't apply here** — this benchmark is Sentinel-2 RGB only, no DSM/
  elevation component was acquired or requested.
- All 32 tiles are genuinely different Sentinel-2 acquisitions (dates range
  Apr-Dec 2025), not one scene reused across tiles.

## Where things live

- `data/sentinel2_benchmark/{category}/{tile_id}/{tile_id}_RGB.tif` — the 32 tiles
- `data/sentinel2_benchmark/manifest.csv` — tile_id, category, lat/lon, UTM
  bbox/EPSG, ICESat-2 ground-photon count + segment count, date acquired,
  cloud %, resolution, path
- `data/sentinel2_benchmark/icesat2_coverage.csv` — all 52 candidates scored
  (including the 20 dropped)
- `data/sentinel2_benchmark/selected_tiles.csv` — the 32 that were selected
  for download
- `data/sentinel2_benchmark/download_results.csv` — raw download outcomes
- `scripts/sentinel_benchmark_candidates.py`,
  `scripts/sentinel_benchmark_icesat_coverage.py`,
  `scripts/sentinel_benchmark_select.py`,
  `scripts/sentinel_benchmark_download.py`,
  `scripts/sentinel_benchmark_manifest.py` — the original pipeline, in run order

## Content-quality remediation (post-build)

A follow-up objective audit found the initial selection had real content
problems the ICESat-2/cloud-metadata screen alone couldn't catch: **11 of
32 tiles failed** category-specific content thresholds (ESA WorldCover %
+ SRTM elevation std + actual Sentinel-2 SCL cloud fraction, not just
scene-level cloud metadata) — most strikingly **6 of 8 agricultural
tiles**, because the original candidate coordinates were literal city/town
centroids rather than rural farmland. A second pass, after visual review of
the resulting thumbnails, caught 4 more problems the numeric checks missed
entirely: a satellite-swath nodata edge, two cases of Sentinel-2's SCL
cloud mask under-detecting real haze (a known weakness over water and for
thin cirrus), and one tile numerically passing its threshold while still
looking wrong (a city's flat plateau, not real hill terrain). Two of the
remediated tiles' first replacement attempt also failed on inspection
(off-season/brown farmland; a permanently silt-choked river mouth) and
needed a second swap.

**Net result: 14 of the 32 tiles changed** (10 shifted to a better location
in the same named region, 4 replaced outright with a different named region
from the original wider candidate pool). Every replacement was
re-validated against the full original bar: content thresholds, actual
(not scene-metadata) cloud fraction, nodata fraction, and — for any tile
that moved location — a fresh ICESat-2 coverage check. All still land at
8/8 per category. Full before/after numbers, reasoning per tile, and the
scripts used:

- `scripts/benchmark_content_audit.py` — the objective audit (WorldCover +
  SRTM + actual SCL cloud fraction) against every one of the original 32 tiles
- `scripts/benchmark_remediation.py` — the remediation cascade (same
  location/new date → shift ~15-40km toward real matching terrain → full
  replacement from the wider candidate pool), each stage re-checking
  content + cloud + nodata + ICESat-2
- `data/sentinel2_benchmark/content_audit.csv`,
  `remediation_results.csv`, `remediation_round2.csv`,
  `remediation_round2b.csv`, `remediation_round2c.csv` — full numeric
  trail for every candidate tried, not just the ones kept
- `data/sentinel2_benchmark/audit_thumbnails/` — a 512×512 true-color
  preview for every one of the final 32 tiles, used for the visual review
  that caught what the numeric thresholds missed
