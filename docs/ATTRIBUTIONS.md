# Attributions

The MIT licence (`LICENSE`) covers this repository's code only. The data and models below keep their own
licences. Only sources the shipped app uses are listed. The app's own Docs page ("Data sources & credits") shows the
same credits.

## Imagery

| Source | Used for | Attribution | Licence |
|---|---|---|---|
| Copernicus Sentinel-2 (via the Copernicus Data Space Ecosystem) | The four demo regions; library scenes; live "Search Online" | Contains modified Copernicus Sentinel data (2024–2026), processed by ESA | Copernicus Sentinel data licence (free, full and open) |
| Maxar (now Vantor) Open Data Program | 6 very-high-resolution library crops (Sikkim, Darjeeling) | Maxar (now Vantor) Open Data Program | CC BY-NC 4.0 |
| IEEE GRSS Data Fusion Contest 2019, Track 1 | Library tiles, display resolution only | IEEE GRSS Data Fusion Contest 2019; JHU/APL Urban Semantic 3D (US3D) dataset | Contest terms; credited as required |

## Elevation

| Source | Used for | Attribution | Licence |
|---|---|---|---|
| Copernicus DEM GLO-30 | Terrain surface for every region and live searches | © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA | Copernicus DEM licence (GLO-30: free) |
| OpenTopography | Serves the Darjeeling GLO-30 DSM | OpenTopography (opentopography.org), NSF | As the underlying GLO-30 |
| FABDEM v1-2 (via the awesome-gee-community-catalog, `projects/sat-io/open-datasets/FABDEM`) | Bare-earth terrain for library tiles and the Maxar surface model | Hawker et al. (2022), Environ. Res. Lett. 17 024016; University of Bristol / Fathom | CC BY-NC-SA 4.0 |

## Models

| Model | Used for | Attribution | Licence |
|---|---|---|---|
| Depth Anything V2 Large (`depth-anything/Depth-Anything-V2-Large-hf`) | Backend relative-depth layer | Yang et al. (2024), *Depth Anything V2* | CC BY-NC 4.0 |
| Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) | Hugging Face Space, desktop ONNX build, Colab bridge | Yang et al. (2024), *Depth Anything V2* | Apache-2.0 |

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
