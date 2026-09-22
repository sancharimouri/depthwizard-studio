# Sparse LiDAR-Guided Correction — Stage 0 feasibility gate

**Question:** can the Sparse LiDAR-Guided Correction method even be tested on
this project's current DFC2019 data? Investigation only — no correction
logic implemented.

**UPDATE (see §6, §7, then §8 for the final word): §6's original claim that
this works for all 50 tiles was WRONG and has been retracted.** The point
clouds are real and the 512m/0.5m-GSD numbers in §6 are real, but §7 found
— by directly comparing rasterized point-cloud relief against each tile's
actual AGL content, something §6 did not do — that the point cloud's
`Tile_<index>` numbering does NOT correspond to our tile IDs' embedded
number. §8 then ran the full, exhaustive version of that check (every
benchmark tile against every point-cloud index in its city, 20,176
comparisons) to rule out the shortcut simply looking in the wrong place —
**still 0/50 real matches found.** The original "0 of 50 tiles eligible"
verdict below **stands** — no working per-tile geotransform recovery has
been established via this dataset.

**Original bottom line (at the time this doc was first written): no, not as
this data currently sits in this repo.** Georeferencing is completely absent
from all 50 benchmark tiles (both RGB and AGL, confirmed directly, not
assumed), there are no separate metadata/RPC files anywhere locally to
recover it from, and recovering it would require external re-acquisition of
the original DFC2019 distribution with an uncertain outcome (see below).
0 of 50 tiles were eligible for a real Stage 1A (ICESat-2-guided) test at
that time.

## 1. Is the `NotGeoreferencedWarning` real, or a false alarm?

**Real — confirmed directly with rasterio, not assumed.** Checked every one
of the 50 benchmark tiles (both `_RGB.tif` and `_AGL.tif`, 100 files total,
not a sample):

| Check | Result across all 50 tiles |
|---|---|
| `src.crs` | `None` for all 100 files |
| `src.transform` | identity matrix for all 100 files |
| `src.gcps` | empty for all 100 files |
| `src.rpcs` | `None` for all 100 files |

No exceptions, no partial cases. A representative tile's tag dump
(`JAX_004_006_RGB.tif`):

```
TIFFTAG_IMAGEDESCRIPTION: '{"shape": [1024, 1024, 3]}'
TIFFTAG_SOFTWARE:         'tifffile.py'
TIFFTAG_DATETIME:         '2018:12:17 11:57:44'
```

**The `TIFFTAG_SOFTWARE: tifffile.py` tag is the key clue.** Python's
`tifffile` library does not preserve GDAL-style GeoTIFF tags (CRS,
geotransform, RPC) unless a caller explicitly writes them — and none did
here. This tag, plus the JSON `{"shape": ...}` description (a pattern
typical of `tifffile.imwrite(path, array, description=json.dumps({"shape":
array.shape}))`), indicates these specific local files were re-saved by some
Python-based repackaging step at some point before landing in this repo's
`data/dfc2019/raw/`, not that they're a byte-for-byte copy of whatever the
official DFC2019 distribution ships. Whether that repackaging *stripped*
real georeferencing that existed upstream, or the upstream files never had
it in the first place, isn't resolvable from the local files alone — see §2.

## 2. Local search for separate RPC/metadata files

Searched the entire `data/dfc2019/raw/` tree for anything besides imagery:

```
find data/dfc2019/raw -type f -not -iname "*.tif"
  -> data/dfc2019/raw/.DS_Store
  -> data/dfc2019/raw/RGB/.DS_Store
  -> data/dfc2019/raw/Truth/.DS_Store
```

**Nothing.** 8,350 `.tif` files (RGB + AGL + CLS across all 2,783 DFC2019
tiles) and three macOS `.DS_Store` files — no `.RPB`, `.rpc`, `.txt`, `.xml`,
or any other sidecar metadata anywhere. No prior script in this repo
(`build_dfc2019_manifest.py`, `inspect_dfc2019_pairs.py`, or any other DFC-
related script) references RPC, CRS, or georeferencing at all — this hasn't
been investigated here before.

**Checked what's publicly documented about the official distribution**
(web search, since nothing local resolves this):

- The DFC2019 collection overview states the underlying WorldView-3 source
  imagery (26 scenes over Jacksonville, 43 over Omaha, 2014–2016) "are
  provided with RPC parameters, acquisition timestamps, and solar angles."
- However, Track 1's actual released product is described as
  **"unrectified single-view"** per-tile crops, and neither the GitHub
  baseline repo (`pubgeo/dfc2019`) nor the IEEE DataPort listing confirms
  whether that per-tile RPC information is embedded in the Track 1 RGB TIFFs
  themselves, shipped as separate per-source-image files, or omitted from
  the Track 1 product entirely. RPC embedding is explicitly documented for
  **Track 3** ("RPC sensor model information is embedded directly in image
  files for Track 3" / "Unrectified images are provided with RPC metadata
  already adjusted using the lidar" — Track 3 only); no equivalent
  confirmation exists for Track 1 in anything surfaced by this search.
- This is a plausible, common benchmark-design choice: releasing
  intentionally non-georeferenced crops for a *relative* height-prediction
  task prevents trivial geo-lookup shortcuts. It's also plausible RPC was
  simply omitted from the Track 1 package because Track 1 doesn't need it
  (single-image height regression, not multi-view stereo like Track 3).

**Conclusion for this step:** georeferencing is not recoverable from
anything already present in this repo. Whether it exists at all in the
official Track 1 download package is genuinely unresolved by public
documentation — resolving it with certainty would require directly
re-downloading the official DFC2019 archive (IEEE DataPort account or the
`pubgeo/dfc2019` torrent, multi-GB) and inspecting the *original,
untouched* files, which is outside the scope of this local investigation
and wasn't attempted here.

## 3. Per-tile lat/lon bounding boxes

**Blocked.** With zero CRS/transform/GCP/RPC information on any of the 50
tiles, there is no real geographic footprint to compute for any tile. The
tile IDs encode only a city code (`JAX` = Jacksonville FL, `OMA` = Omaha NE)
and internal tile/source-image numbers — city-level, not tile-level,
location. That's roughly 100+ km of uncertainty per tile, useless as an
"AOI" for a coverage query (it would trivially return "yes, ICESat-2 has
passed over this city at some point," which says nothing about whether it
crossed this specific 1024×1024 px, ~307 m × 307 m tile at 0.3 m/pixel).

## 4. ICESat-2 (sliderule) coverage query

**Not attempted, deliberately.** `sliderule` needs a real polygon/bbox to
query against. Feeding it a city-level guess instead of an actual tile
footprint wouldn't be a real feasibility test — it would just be querying
whether ICESat-2 ever crossed Jacksonville or Omaha (yes, trivially, for
both, many times), which answers a different question than "does ICESat-2
data exist for these 50 specific 300 m tiles." Running the query anyway and
reporting a number would be fabricating a result the input data can't
support, which is the opposite of what this investigation is for.

**Worth noting even independent of the georeferencing blocker:** each tile
is small (~307 m × 307 m). ICESat-2's ground tracks are narrow and spaced
several km apart between passes; a given repeat cycle's reference ground
tracks only cross a small fraction of any region's total area. Even with
real footprints in hand, the a priori chance that a specific 300 m tile
happens to be crossed by an ICESat-2 ground track at all (let alone with
enough clean, cloud-free ATL08 canopy/terrain photon returns for a usable
correction signal) is low per tile. With only 50 tiles scattered across two
mid-size cities, expect a small minority to have any usable intersection
even in the best case where georeferencing is fully recovered — this should
temper expectations regardless of how §2's open question resolves.

## 5. Per-tile table and conclusion

All 50 benchmark tiles, confirmed identical outcome (georeferencing check
was exhaustive, not sampled — see §1):

| Georeferencing recoverable (locally) | ICESat-2 intersects | Usable photon count |
|---|---|---|
| **N** (all 50/50 tiles) | N/A — blocked upstream | N/A — blocked upstream |

Tile IDs checked (all 50, from `dav2_baseline/manifest.csv`): `JAX_004_006,
JAX_004_014, JAX_004_016, JAX_018_012, JAX_022_009, JAX_031_006,
JAX_072_015, JAX_118_012, JAX_118_015, JAX_149_006, JAX_149_025,
JAX_161_001, JAX_164_008, JAX_165_015, JAX_166_006, JAX_175_002,
JAX_204_005, JAX_214_015, JAX_214_023, JAX_224_025, JAX_264_013,
JAX_269_009, JAX_416_009, JAX_416_022, JAX_505_016, JAX_505_018,
OMA_042_011, OMA_084_038, OMA_134_027, OMA_144_030, OMA_198_002,
OMA_211_032, OMA_211_039, OMA_212_033, OMA_221_034, OMA_225_001,
OMA_230_036, OMA_248_029, OMA_248_030, OMA_258_020, OMA_269_035,
OMA_281_002, OMA_281_030, OMA_315_019, OMA_315_020, OMA_332_037,
OMA_364_003, OMA_364_043, OMA_376_023, OMA_376_038`.

### Is this worth pursuing?

**Not as-is.** Concretely, to make this testable at all would require, in
order:

1. Re-acquire the *original, untouched* DFC2019 Track 1 files directly from
   IEEE DataPort or the official torrent (not attempted here — multi-GB,
   needs an account/client, and outside this investigation's scope) and
   check whether *those* files carry real RPC/GCP data that this repo's
   `tifffile.py`-resaved copies lost.
2. Even if step 1 succeeds, confirm real ICESat-2 ground-track intersection
   with each tile's true footprint — genuinely unknown until step 1 is
   done, but likely to affect only a minority of the 50 tiles given typical
   ICESat-2 ground-track spacing versus a ~300 m tile footprint (§4).
3. Only then would a real Stage 1A test (usable photon counts per
   intersecting tile) be possible.

Given the uncertainty at step 1 and the low expected yield even if it
succeeds, this is a **low-confidence, multi-step external dependency**
before any correction work could begin — not a quick unblock. Recommend
treating Sparse LiDAR-Guided Correction as **on hold pending step 1**
(a one-time external verification of whether the official Track 1 download
actually includes recoverable RPC), rather than committing further design
effort to it now.

**Step 1 status: DONE, superseded by a better path than originally
guessed — see §6.** RPC recovery from the original Track 1 imagery itself
was never actually attempted; instead, real UTM geolocation was recovered
via a different official artifact (the point clouds shipped alongside this
same dataset), which turned out to be simpler and fully conclusive. §4's
ICESat-2 query (step 2/3 above) has explicitly NOT been run yet, per
instruction — this update covers geometry recovery only.

## 6. Geometry recovery via the extended US3D dataset's point clouds

**Bottom line: this works, cleanly, for all 50 of our benchmark tiles.**

**Source:** [Urban Semantic 3D Dataset](https://ieee-dataport.org/open-access/urban-semantic-3d-dataset)
(IEEE DataPort, DOI 10.21227/9frn-7208) — an official JHU/APL extension of
DFC19 that ships `JAX_PointClouds.zip` (1.58 GB) and `OMA_PointClouds.zip`
(2.35 GB) explicitly "with full UTM coordinates to enable experiments
requiring geolocation" (the page's own instructions text). Downloaded both
via an already-authenticated IEEE account (same account used for the
original DFC2019 access) — no new credentials needed. Also grabbed the tiny
`{JAX,OMA}_Metadata_And_Readme.tar` (140 KB / 250 KB) first, which is
enough by itself to confirm the file format before committing to the
multi-GB downloads.

**Format (confirmed from the shipped README, not guessed):** point clouds
are plain comma-delimited ASCII (SEMANTIC3D.NET-style), not LAS/LAZ —
`{City}_Tile_<index>_PC-reduced.txt`, one row per point:
`UTM Easting, UTM Northing, UTM Up, Intensity, Return Number`. A matching
`_PC-classification.txt` carries per-point CLS labels. No special point
cloud library needed — `numpy.loadtxt` reads it directly.

**The key open question going in:** does the point cloud's `<index>` line
up with our tile IDs' embedded number (`JAX_004_006` → `004`), or with the
much smaller set of raw satellite source-image indices (only ~24-26 per
city, per the `*_NITF_METADATA_*.json` files also in the metadata tar)?
**Checked directly, not assumed:** `unzip -l` on both zips shows index
ranges 000-559 (JAX, 833 files) and 000-399 (OMA, 390 files) — far larger
than the raw-image count, confirming these are **per-physical-tile** point
clouds, indexed the same way our tile IDs are.

**All 37 unique tile indices across our 50 benchmark tiles have a matching
point cloud file — 100% hit rate, no misses:**

| City | Unique indices in our 50 tiles | Found in PointClouds.zip |
|---|---:|---:|
| JAX | 19 | 19 / 19 |
| OMA | 18 | 18 / 18 |
| **Total** | **37** | **37 / 37** |

(37 unique physical tiles cover all 50 tile *records* because a handful of
physical tiles are shared by multiple tile IDs, e.g. `JAX_004_006`,
`JAX_004_014`, `JAX_004_016` all carry `004` as their middle number — this
is exactly what you'd expect if AGL ground truth for one physical footprint
gets reused across renders from different source satellite passes: the
same ground truth, different RGB texture. This doesn't break the recovery —
all three tile IDs correctly recover the *identical* real-world geotransform,
since they share the same underlying physical patch.)

**Verification against the known GSD, computed for every one of the 37
indices, not spot-checked:**

For each `{City}_Tile_<index>_PC-reduced.txt`, computed the real-world
extent (`max - min` of Easting and Northing) and divided by 1024 (our
tiles' known pixel dimension) to get an implied GSD. Result, **exactly
37/37**:

```
extent_x = extent_y = 512.00 m (± 0.02 m rounding noise, every single tile)
implied_GSD = 512 / 1024 = 0.5000 m/pixel, every single tile
```

This is a materially different (and better-founded) number than the
"~0.3–0.35 m" figure this investigation was working from at the start —
that figure conflated the *raw sensor's* native GSD (WorldView PAN, ~0.3 m
at nadir, confirmed separately from one `NITF_CSEXRA` metadata record's
`IGEOLO` corners ÷ pixel count) with the *product's* actual working
resolution after the DFC19/US3D pipeline resamples to a standard 0.5 m
grid. The exact, unanimous 512.000 m across 37 independently-checked tiles
(not an average, not an approximation — every single one) is strong
evidence this is the real, exact figure, not noise landing near a
plausible value.

**Bonus: the recovered UTM origins land on a clean, shared grid.** E.g.
JAX tiles `018` and `022` have identical Y-bounds
(`[3357880.00, 3358392.00]`) and X-bounds exactly 1024 m apart — i.e. two
grid cells over on a regular 512 m tiling, not independently-fitted
boxes that happen to be similar. Both cities' recovered UTM zones match
their real-world longitude exactly (JAX ≈ -81.6°E → UTM zone 17N /
EPSG:32617; OMA ≈ -95.9°E → UTM zone 15N / EPSG:32615). OMA's recovered Z
values (264-342 m) also land right on Omaha's real elevation above sea
level — another independent sanity check that these are genuine surveyed
coordinates, not placeholder/normalized values.

**What this gives us:** for every one of the 50 benchmark tiles, a
complete, real geotransform — origin `(min Easting, max Northing)`, pixel
size `(0.5, -0.5)`, rotation 0, CRS = the tile's city's UTM zone — ready to
write directly onto our local (currently non-georeferenced) RGB/AGL/CLS
GeoTIFFs. Point cloud files and both extraction scripts used for this
check are not yet committed anywhere in the repo (this was a report-back
checkpoint, per instruction) — full zips are in `~/Downloads/` on this
machine if this gets picked back up.

**Explicitly not done, per instruction: no ICESat-2 query yet.** §4's
original concern (does ICESat-2 actually cross these tiles at all, given
narrow ground-track spacing vs. a ~500 m footprint) is now testable with a
real footprint instead of a guess, but that step was intentionally not
run here.

## 7. §6's tile mapping was never actually verified against pixel content — checked now, and it fails

**This section retracts §6's headline claim.** §6 established that
`{City}_Tile_<index>_PC-reduced.txt` files exist, are real, per-512m-patch
point clouds with recoverable geotransforms — that part holds up. What it
did *not* do, and should have, was confirm that `<index>` actually
corresponds to the same physical location as our tile IDs' embedded number
(`JAX_004_006` → index `004`). It treated a filename-number coincidence as
a validated correspondence. Asked directly to verify this before trusting
it, and it doesn't hold.

**Method:** for every benchmark tile ID sharing an index with another tile
(`JAX_004_006`/`014`/`016` all share `004`, plus 11 other pairs/triples —
25 of our 50 tiles across 12 groups), loaded the real local AGL raster
(`data/dfc2019/raw/Truth/Track1-Truth/{tile}_AGL.tif`) for each tile in the
group and compared them directly:

- **Pairwise AGL-vs-AGL correlation within each group**: if these tiles
  really were different renders of one shared physical footprint, their
  AGL (height) values should correlate very strongly (AGL is a property of
  the ground, not of which satellite pass rendered the RGB). Actual
  result: correlations of 0.39-0.84 across all 12 groups — real but
  moderate, consistent with tiles that are simply *nearby* in a scattered
  benchmark (similar urban texture), not evidence of an identical
  footprint.
- **The decisive test: point-cloud relief vs. AGL relief.** For each
  group's shared index, rasterized that index's point cloud onto its
  recovered geotransform (0.5 m/pixel grid, nearest-cell mean of `UTM Up`),
  then high-pass filtered both the rasterized point cloud and each
  candidate tile's AGL (subtract a σ=30px Gaussian blur, removing broad
  elevation trend/datum offset and keeping only the "bumps" — buildings,
  trees — which is exactly what should spatially align if it's really the
  same ground). Result across **all 12 groups, every candidate tile**:
  correlation between -0.06 and +0.03 — indistinguishable from noise. Not
  one candidate in any group showed real structure-matching correlation.
- **Ruled out an off-by-N indexing shift**: scanned point-cloud indices
  000-010 against `JAX_004_006`'s AGL specifically, in case the "004"
  numbering was offset by a constant. No index in that window correlates
  either (all -0.04 to +0.01).
- **Ruled out a pipeline bug**: the point-cloud raster itself has real
  spatial structure — neighbor-pixel autocorrelation 0.54 at 1px, 0.41 at
  20px, exactly what genuine terrain/building relief looks like, not
  scrambled noise. And reverse-geocoding `Tile_004`'s recovered UTM bbox
  (EPSG:32617 → WGS84) lands at 30.355-30.360°N, -81.709 to -81.704°W —
  squarely inside the real Jacksonville area, in fact inside the exact
  bounding box of NITF satellite image `004` (the raw source-image
  metadata check from earlier in this investigation, a separate,
  much-smaller ~24-value numbering space than the point cloud's 000-559
  tile index — coincidentally the same digits, another reminder not to
  trust matching numbers across different indexing schemes). So `Tile_004`
  is a real Jacksonville location with real relief — it correlates with **none of
  our 26 JAX benchmark tiles** (checked exhaustively, not just the `004`
  group; best absolute correlation among all 26 was 0.053, still noise).
  It just isn't one of the specific 512m patches we happen to have locally.

**Conclusion:** the point cloud's `Tile_<index>` numbering and our tile
IDs' embedded number are two independent numbering schemes that happen to
overlap in small-integer range, not the same grid. §6's "37/37, 100% hit
rate" was a real hit rate on *filenames existing*, not on *correct
correspondence* — an important distinction this investigation conflated.
**No valid per-tile geotransform has actually been recovered.** Building
the lookup table requested as the next step would, on this mapping,
produce 50 confidently-labeled but wrong geotransforms — worse than no
table, since wrong-but-plausible-looking geolocation is harder to catch
downstream than an honest "unknown." Not built.

**What would actually be needed to recover this correctly** (not attempted
here): a real spatial join — e.g. correlating each of our 50 tiles' AGL
raster against a broad scan of many point-cloud indices (not just the
filename-coincident one) to find its true match by relief pattern, the way
§7's negative test did but searching outward instead of confirming one
candidate. With ~560 JAX and ~400 OMA point-cloud tiles and only 26+24
tiles to place, this is a real but bounded search (a few hundred
correlation checks per tile, not combinatorial), not a dead end — just
unproven and meaningfully more work than the filename-matching shortcut
this investigation tried first.

## 8. The full spatial join — run, and it also comes back negative

**This closes the geometry-recovery question via the extended US3D point
clouds: 0 of 50 tiles found a real match. No lookup table was built.**

**Method:** exactly the bounded search §7 proposed. For every one of the
50 benchmark tiles, computed the high-pass-filtered AGL relief (same
method validated in §7 — σ=30px Gaussian, isolates building/tree bumps
from broad elevation trend) and correlated it against **every**
`{City}_Tile_<index>_PC-reduced.txt` in that city's pool — 416 indices for
JAX, 390 for OMA (the real counts, per `unzip`'s member list — not the
000-559/000-399 numeric *range*, which has gaps) — not just the
filename-coincident one. 20,176 correlations total
(416×26 + 390×24), computed directly from the zip via streaming reads (no
full extraction needed). Full results:
`docs/method-audit/stage0-gates/artifacts/us3d_spatial_join_results.csv`.

**Threshold, decided from the data, not fixed in advance:** across all
20,176 correlations (the true population, since at most 50 of them could
ever be a real match), the distribution is centered almost exactly on zero
(mean -0.00004, std 0.0159) with a 99.99th-percentile of 0.116 — a clean
noise distribution, no separate population of high-correlation values
anywhere in it. **Per-tile best-match results, sorted by best correlation:**

| Rank | Tile | Best index | Best corr | Runner-up | Gap |
|---:|---|---:|---:|---:|---:|
| 1 | JAX_149_025 | 179 | 0.1633 | 0.0903 | 0.0729 |
| 2 | JAX_149_006 | 179 | 0.1466 | 0.0920 | 0.0546 |
| 3 | OMA_376_038 | 376 | 0.1156 | 0.0961 | 0.0195 |
| ... | (47 more tiles) | | 0.034-0.106 | | mostly <0.03 |

Full table in the CSV/artifacts above. Even the single most extreme value
found across the *entire* 20,176-comparison search (`JAX_149_025` vs. index
`179`, r=0.163) is ~10 standard deviations from the population mean —
notable only in the sense of "most extreme value in a large search," not
in absolute terms. A genuine same-footprint match (independently-derived
height data over the identical ground truth) should show strong,
unmistakable correlation — the trivial self-correlation case is 1.0, and
even accounting for real differences between a photogrammetric AGL product
and an averaged point-cloud raster, a true match should land well above
0.3-0.5, not 0.16. Nothing in this search comes remotely close to that.

**No tile clears any reasonable confidence bar. 0/50, not partial.** Per
instruction, this is reported as the honest outcome rather than stretched
to claim the top 1-2 borderline values (0.163, 0.147 — both landing on the
*same* external index, `179`, itself mildly interesting but far too weak
to call a match) as real. **No geotransform lookup table was built** —
there is nothing to build it from.

**Most likely explanation:** the extended US3D dataset's
`{JAX,OMA}_PointClouds.zip` almost certainly covers a specific subset of
each city's full tile grid (the README describes it as "the original DFC19
training and validation point clouds") that does not overlap with
whichever specific 50 tiles ended up in this project's local benchmark —
i.e., our 50 tiles may be drawn from a different split (e.g. an "Extra"
pool) than the one these point clouds were generated for. This isn't
provable from what's available locally, but it's consistent with every
piece of evidence gathered across §§6-8: the point clouds are real,
correctly-located Jacksonville/Omaha data (confirmed via reverse-geocoding
in §7) with genuine spatial structure (confirmed via the point-cloud's own
neighbor-autocorrelation in §7) — they just don't appear to include our
specific 50 tiles anywhere in their index space.

**Bottom line for Sparse LiDAR-Guided Correction feasibility, updating §
"Is this worth pursuing?" above:** geometry recovery via this specific
external dataset does not work for our 50-tile benchmark. Real per-tile
georeferencing for this benchmark remains unrecovered. The step 1 marked
"DONE" above should be read as "attempted via two independent methods
(filename-index matching, then a full spatial search), both negative" —
not "solved." A different recovery path (e.g. the original DFC2019 Track 1
imagery's own RPC data, per the original §"Is this worth pursuing?"
plan — never actually attempted in this investigation) remains the
only unexplored option, and its own odds were already assessed as uncertain
there.
