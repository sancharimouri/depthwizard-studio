# Prompt 3 — base ground elevation for the 50 DFC2019 tiles (2026-09-28)

Script: `scripts/dfc2019_base_ground.py` (city, located, datum, write). Outputs go to the gitignored
`data/dfc2019/terrain_packs/base/` (`<tile>.npz` holds the 341² base plus a source record).

## Counts by source
| City | dem-located (USGS 3DEP 1 m) | city-level (approximate) |
|---|---|---|
| JAX | 8 | 18 |
| OMA | 7 | 17 |
| **All** | **15** | **35** |

## Located tiles (the 15 confident matches from Prompt 2)
- **DEM:** the finest 3DEP bare-earth product with valid coverage over the tile's window. **All 15 got 1 m**;
  1/3″ and GLO-30/FABDEM were not needed.
  - JAX: `FL_Peninsular_FDEM_2018`.
  - OMA: `NE Eastern UA 2016` (Nebraska side) or `IA_WesternIA_2020` (Iowa side of the Missouri; one tile uses
    two zone-14/15 products). Products are chosen by valid coverage, not by the first hit; the Iowa
    products are NaN over Nebraska.
- **Sampling:** the DEM is sampled bilinearly at the 341² pack-grid cell centres in the **tile's own pixel
  frame**, using the matched orientation and GSD. The orientation mapping is unit-checked:
  tile → window reproduces a north-up grid for all 8 dihedral transforms. One of the 15 is not north-up (JAX_175_002, orient 1).
- **Ground range:** JAX −0.6 to 16.1 m, OMA 287.9 to 348.9 m.

## Datum check (verification only, never shipped)
This compares the DEM with the cloud's own class-2 ground returns (min at 4.8 m) at the same points.

| City | Tiles | DEM − cloud ground (median) | Per-tile spread | GEOID18 undulation N | Residual after geoid |
|---|---|---|---|---|---|
| JAX | 8 | +28.28 m | 28.25–28.40 | −29.81 m | **−1.53 m** |
| OMA | 7 | +27.76 m | 27.57–27.99 | −28.56 m | **−0.80 m** |

- **The US3D cloud's "UTM Up" is ellipsoidal height,** as Prompt 1 suspected. The raw 28 m offset is
  entirely the geoid.
- **After the geoid correction the residual is −1.5 / −0.8 m, "a few metres", with no datum bug.** The
  sub-2 m remainder matches the known NAD83 vs WGS84/ITRF ellipsoid-height difference in CONUS
  (roughly 1–1.5 m), plus the smoothing of our 4.8 m cloud ground.
- The shipped base is the NAVD88 3DEP DEM itself. The cloud is not shipped.

## City-level fallback (35 tiles)
- **Base:** the median of the 3DEP 1/3″ DEM (current products `n31w082` 2022-11-03 and `n42w096` 2022-12-18,
  resampled to 10 m) over the **union of the city's US3D cloud-tile footprints**, where any DFC2019
  tile can lie. The 5th–95th percentiles are recorded as the range.

| City | Base (median) | p5 – p95 | Cells |
|---|---|---|---|
| JAX | **6.01 m** | −0.01 – 9.43 m | 1.09 M |
| OMA | **299.91 m** | 295.55 – 362.81 m | 1.02 M |

- **Independent published checks:**
  - **Omaha, Eppley Airfield (KOMA), inside the OMA footprint:** FAA field elevation **984 ft = 299.9 m**
    ([FAA NFDC](https://nfdc.faa.gov/nfdcApps/services/ajv5/airportDisplay.jsp?airportId=OMA),
    [SkyVector](https://skyvector.com/airport/OMA/Eppley-Airfield-Airport)).
    - City median: 299.9 m, a 0.0 m difference.
    - 3DEP 1/3″ at the airport reference point: 297.4 m (−2.5 m). FAA field elevation is the highest runway point.
  - **Jacksonville, NAS Jacksonville (KNIP), 1.5 km south of the JAX footprint:** **23 ft = 7.0 m**
    ([SkyVector](https://skyvector.com/airport/NIP/Jacksonville-NAS-Towers-Field--Airport),
    [SKYbrary](https://skybrary.aero/airports/knip)).
    - City median: 6.0 m (−1.0 m).
    - 3DEP at the reference point: 4.8 m (−2.2 m).
  - Agreement is within 1 m for both city medians. The source values are recorded here.
- **City-level tiles are approximate and must not read as measured.** Each pack stores
  `base_source: "city-level"` and the p5–p95 range. Prompt 5 labels them "Approximate city-level ground
  elevation" and adds the note "Elevations are approximate…".
- These tiles are flat at the city median; the relief comes only from the heights.

## Upgrade path
When more locations are accepted (see the follow-up in `locate.md`), run
`dfc2019_base_ground.py located` then `write`, and rebuild the affected packs. Nothing else changes.
