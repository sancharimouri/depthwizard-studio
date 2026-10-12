# Attributions

The MIT licence (`LICENSE`) covers this repository's code only. The data and models below keep their own
licences. Only sources the shipped app uses are listed. The app's own Docs page ("Data sources & credits") shows the
same credits.

## Imagery

| Source | Used for | Attribution | Licence |
|---|---|---|---|
| Copernicus Sentinel-2 (via the Copernicus Data Space Ecosystem) | Demo regions (Darjeeling, Gulf of Kutch; Bangalore below); library scenes; live "Search Online" | Contains modified Copernicus Sentinel data (2024–2026), processed by ESA | Copernicus Sentinel data licence (free, full and open) |
| Maxar (now Vantor) Open Data Program | 6 very-high-resolution library crops (Sikkim, Darjeeling) | Maxar (now Vantor) Open Data Program | CC BY-NC 4.0 |
| IEEE GRSS Data Fusion Contest 2019, Track 1 | Library tiles, display resolution only | IEEE GRSS Data Fusion Contest 2019; JHU/APL Urban Semantic 3D (US3D) dataset | Contest terms; credited as required |

## Elevation

| Source | Used for | Attribution | Licence |
|---|---|---|---|
| Copernicus DEM GLO-30 | Surface for the Darjeeling demo region, the Darjeeling library tile, searched scenes and uploads | © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA | Copernicus DEM licence (GLO-30: free) |
| OpenTopography | Serves the Darjeeling GLO-30 DSM | OpenTopography (opentopography.org), NSF | As the underlying GLO-30 |
| FABDEM v1-2 (via the awesome-gee-community-catalog, `projects/sat-io/open-datasets/FABDEM`) | Terrain (calibrated to ICESat-2 for the Sentinel-2 library tiles) for library tiles, searched scenes and uploads; base of the Maxar surface | Hawker et al. (2022), Environ. Res. Lett. 17 024016; University of Bristol / Fathom | CC BY-NC-SA 4.0 |

## Licence of the demo data files

The MIT licence covers this repository's code. The data files under `frontend/public/data/` and
`frontend/public/cityscape/` are not MIT: they keep their source licences, as credited in this file. FABDEM-derived
data is CC BY-NC-SA 4.0; OpenStreetMap-derived data is ODbL 1.0; Google Open Buildings data is CC BY 4.0; swisstopo
data is Swiss federal open government data (free use with source credit); Copernicus DEM and Sentinel-2 data are under
their Copernicus licences.

## Cosmetic urban model (Bengaluru block: Studio library tile since 2026-10-12, formerly the demo page)

| Source | Used for | Attribution | Licence |
|---|---|---|---|
| Google Open Buildings v3 (polygons) and Open Buildings 2.5D Temporal v1 (2023 `building_height`), via Earth Engine | Building footprints and heights of the 500 × 500 m UB City / Vittal Mallya Road block (ML estimates from satellite imagery) | Google Research, Open Buildings (sites.research.google/open-buildings) | CC BY 4.0 |
| OpenStreetMap (one Overpass query) | Roads, road names and landmark names in the same block | © OpenStreetMap contributors | ODbL 1.0 |

Terrain under the block is FABDEM (listed under Elevation). The same block is the Bengaluru library tile ("UB City, Bengaluru"): its
imagery is a synthetic top view rendered from these buildings and roads, with trees placed where Copernicus Sentinel-2
(listed above; Earth Engine `COPERNICUS/S2_SR_HARMONIZED`, a 2023–24 dry-season median NDVI) sees vegetation; its DSM
is FABDEM plus these building heights, and its true-colour layer is this model.

## Showcase tiles (demo: Lavaux, Darjeeling 20 km, Gulf of Kutch; Studio library: Nanda Devi Sanctuary, Kangchenjunga)

| Source | Used for | Attribution | Licence |
|---|---|---|---|
| swisstopo swissALTI3D 0.5 m (2021), swissSURFACE3D Raster 0.5 m (2025), SWISSIMAGE 10 cm (2023), via the STAC API at data.geo.admin.ch | Lavaux: elevation layer, surface (mesh), imagery (shown at 0.39 m) | © swisstopo | Swiss federal open government data (free use with source credit) |
| Copernicus DEM GLO-30 (2024 release, Earth Engine `COPERNICUS/DEM/GLO30_2024_1`) | Darjeeling (demo, 20 × 20 km), Nanda Devi Sanctuary and Kangchenjunga (library): surface (mesh) and elevation layer; it is a surface model, there is no separate terrain model | © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA | Copernicus DEM licence (GLO-30: free) |
| Copernicus Sentinel-2 L2A (Earth Engine `COPERNICUS/S2_SR_HARMONIZED`), Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`) cloud mask | Darjeeling, Nanda Devi Sanctuary and Kangchenjunga imagery: 2025-10-01 to 2025-11-30 median (Darjeeling: blue haze / shadow tones recoloured to greens) | Contains modified Copernicus Sentinel data 2025 | Copernicus Sentinel data licence (free, full and open) |

The Gulf of Kutch demo region (14 × 14 km containing the library tile's Koteshwar mudflats, extended south-west) uses FABDEM v1-2 with the
library tile's ICESat-2 offset (+0.3214 m) and the Copernicus Sentinel-2 scene of 2025-12-28 (both listed above;
"Contains modified Copernicus Sentinel data 2025").

The Nanda Devi and Kangchenjunga squares were placed from OpenStreetMap peak and glacier positions
(© OpenStreetMap contributors, ODbL 1.0); no map data is drawn on them.

## Close-up city tiles (Studio library, "City 3D": Connaught Place, Nariman Point & Marine Drive, Dalhousie Square & Hooghly, Charminar, Nishi-Shinjuku, Canal ring Amsterdam, Financial District New York)

DRAFT (2026-10-12). Connaught Place is now a demo region (#/demo, in place of the Bengaluru block, which moved to the
Studio library with the same sources). The app shows no credit line on the 3D view; its Docs page ("Data sources &
credits") credits these sources. The "required wording" column quotes each source's own terms.

| Source | Used for | Required wording | Licence |
|---|---|---|---|
| Google Open Buildings v3 (polygons) and Open Buildings 2.5D Temporal v1 (2023 `building_height`, median per building), via Earth Engine | Connaught Place, Mumbai, Kolkata, Hyderabad: building footprints and heights (ML estimates from satellite imagery) | Google Research, Open Buildings (sites.research.google/open-buildings). The Temporal dataset's page adds: "We have leveraged Copernicus Sentinel data (2016-2023) processed by the European Space Agency." | CC BY 4.0 (dual-licensed with ODbL 1.0; used under CC BY 4.0) |
| Project PLATEAU (MLIT) 3D city models, FY2025 data, 3D Tiles: 新宿区, 渋谷区, 中野区 (LoD2 with textures; LoD1 where a building has no LoD2) | Nishi-Shinjuku: buildings (2,044 from Shinjuku, 334 from Shibuya, 21 from Nakano); clipped to the box, re-projected, textures downscaled | 「3D都市モデル（Project PLATEAU）新宿区（2025年度）」「3D都市モデル（Project PLATEAU）渋谷区（2025年度）」「3D都市モデル（Project PLATEAU）中野区（2025年度）」（国土交通省）（https://www.geospatial.jp/ckan/dataset/plateau-13104-shinjuku-ku-2025, …/plateau-13113-shibuya-ku-2025, …/plateau-13114-nakano-ku-2025）を加工して作成 | PLATEAU Site Policy §3 (Public Data License 1.0, compatible with CC BY 4.0) |
| GSI elevation tiles (地理院タイル 標高タイル DEM5A, 5 m), cyberjapandata.gsi.go.jp | Nishi-Shinjuku: ground | 地理院タイル（標高タイル（基盤地図情報数値標高モデル））を加工して作成 (出典：国土地理院ウェブサイト, https://maps.gsi.go.jp/development/ichiran.html) | GSI terms of use (Public Data License 1.0); attribution only, no application needed for these tiles |
| FABDEM v1-2 | Connaught Place (flat at its median: relief under 3 m), Mumbai, Kolkata, Hyderabad: ground | Hawker et al. (2022), Environ. Res. Lett. 17 024016; University of Bristol / Fathom | CC BY-NC-SA 4.0 |
| OpenStreetMap (one Overpass query per tile) | Roads in every tile; water (and the sea, from the coastline) and parks in the five phase-2 tiles; the boxes were placed from OSM positions | © OpenStreetMap contributors (openstreetmap.org/copyright) | ODbL 1.0 |
| 3D BAG v2023.10.08 (OGC API, LoD 2.2 solids), TU Delft 3D geoinformation group and 3DGI | Canal ring, Amsterdam: buildings (2,724; clipped to the box, re-projected); flat ground at the median 3D BAG ground level (NAP) | © 3DBAG by tudelft3d and 3DGI, with a link to https://docs.3dbag.nl/en/copyright/ | CC BY 4.0 |
| NYC 3-D Building Model, NYC Office of Technology and Innovation (CityGML LoD2, delivery area 12; DoITT 2014 aerial survey) | Financial District, New York: buildings (626; clipped, re-projected); ground interpolated from the model's building ground surfaces | NYC 3-D Building Model, NYC Office of Technology and Innovation (NYC Open Data, data.cityofnewyork.us/d/tnru-abg2). No wording is prescribed | No licence is stated; used on the owner's decision (2026-10-12) |
| Copernicus Sentinel-2 (Earth Engine `COPERNICUS/S2_SR_HARMONIZED`, Nov 2023 – Mar 2024 median NDVI) | Where trees are drawn in the synthetic top view | Contains modified Copernicus Sentinel data 2023–24 | Copernicus Sentinel data licence |

Each tile's image is a synthetic top view rendered from these sources (as for the Bengaluru block), not a photograph.
Its relative depth is Depth Anything V2 on that image (listed under Models).


## Models

| Model | Used for | Attribution | Licence |
|---|---|---|---|
| Depth Anything V2 Large (`depth-anything/Depth-Anything-V2-Large-hf`) | Offline only: the four demo regions' precomputed relative-depth images (not run by the app) | Yang et al. (2024), *Depth Anything V2* | CC BY-NC 4.0 |
| Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) | The relative-depth layer: Hugging Face Space (web), ONNX (desktop app) | Yang et al. (2024), *Depth Anything V2* | Apache-2.0 |

## Facts and scenario cards

| Source | Used for | Attribution | Licence |
|---|---|---|---|
| ThinkHazard! (GFDRR, World Bank) | Hazard levels: wildfire, cyclone, tsunami, volcano; district flood, landslide, earthquake | ThinkHazard! – GFDRR (thinkhazard.org) | Hazard levels CC BY 4.0 |
| FAO GAUL 2015, admin level 2 | District lookup (a coarse 0.025° grid; no boundaries shipped) | FAO Global Administrative Unit Layers (GAUL) 2015 | FAO terms of use; non-commercial |
| Wikidata | Nearest named peak, river, glacier | Wikidata | CC0 1.0 |
| Copernicus EMS / GloFAS river flood hazard maps v2.1.2 | Modelled 1-in-100-year river flood extent | © European Union, Copernicus Emergency Management Service (GloFAS) | CC BY 4.0 |
| JRC Global Surface Water v1.4 | Surface water seen 1984–2021 | EC JRC / Google; Pekel et al. (2016), Nature 540, 418–422 | Free of charge, no restriction of use |
| Global Landslide Hazard Map (World Bank / ARUP, 2021) | Landslide hazard class | World Bank / ARUP (2021) | World Bank Data Catalog terms (dataset 0037584) |
| USGS ANSS ComCat | M4.5+ earthquakes within 100 km | U.S. Geological Survey | Public domain |
| NOAA NCEI IBTrACS v04r01 | Tropical cyclone tracks within 100 km since 1980 | Knapp et al. (2010), NOAA NCEI | Public domain |
| NOAA Storm Prediction Center tornado database | Tornadoes within 25 km since 1950 (US tiles) | NOAA / NWS Storm Prediction Center | Public domain |

## Map tiles, libraries and fonts

| Item | Licence |
|---|---|
| OpenStreetMap tiles (location picker) — © OpenStreetMap contributors | ODbL 1.0 (data); OSMF tile usage policy |
| three.js, camera-controls | MIT |
| Leaflet | BSD-2-Clause |
| DM Sans, Outfit, IBM Plex Mono, Geist Mono | SIL Open Font License 1.1 |

The non-commercial terms above (Maxar Open Data, FABDEM, Depth Anything V2 Large, FAO GAUL) apply to the app as shipped. The
project is a non-commercial Smart India Hackathon entry.
