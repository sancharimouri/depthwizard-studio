# Depth Wizard Studio

Real satellite imagery draped over real elevation models, explored in 3D in the browser or on the desktop.
Built for Smart India Hackathon 2026 (problem SIH26175).

- **Live demo:** https://depthwizard-studio.vercel.app (open the app at https://depthwizard-studio.vercel.app/#/demo)
- **Demo video:** https://depthwizard-studio.vercel.app/#demo-video
- **Desktop app (macOS, Apple Silicon):** https://github.com/sancharimouri/depthwizard2-desktop/releases/latest.
  Linux and Windows builds are not available yet.

## What it is

Pick a library tile, search Sentinel-2 imagery live, or upload a PNG, JPG or GeoTIFF. The app builds a 3D terrain from
a real elevation model (Copernicus GLO-30, FABDEM) and drapes the imagery over it, with layer switching, flythrough,
measurement and hazard facts for the area.

**What it is not:** it does not compute elevation from a single image at 10 m. The terrain always comes from a DEM.
Depth Anything V2's output is shown as a labelled *relative-depth* layer, not as height. Our research found that at
10 m Sentinel-2 no model added value over a plain DEM against independent lidar, so the product uses plain DEMs:
FABDEM (calibrated to ICESat-2 for the Sentinel-2 library tiles) and Copernicus GLO-30. A learned height model (Method 6) is used only for very-high-resolution crops; it is validated on DFC2019
(US cities, satellite imagery) and did not generalize to GAMUS by RMSE.

## Imagery tiers

- **Tier 1, always:** Sentinel-2 (10 m), searched live through the Copernicus Data Space Ecosystem.
- **Tier 2, where available:** very-high-resolution imagery (~0.3 m): Maxar Open Data crops (height prediction baked
  offline). Your own uploads get the DEM terrain and a labelled relative-depth layer.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): tiers, web app, static tile library, backend, depth service, desktop app
- [ML pipeline](docs/ML_PIPELINE.md): which models and DEMs the product uses
- [Validation](docs/VALIDATION.md): how results were tested, pre-registered rules, and the negatives
- [Desktop app](docs/DESKTOP_APP.md)
- [Attributions](docs/ATTRIBUTIONS.md)

## Running locally

One-time setup: `npm install` (root) and `npm install --prefix frontend`, plus a
`.env` at the repo root (copy `.env.example`) with real `CDSE_CLIENT_ID` /
`CDSE_CLIENT_SECRET` values.

Then, from the repo root:

```
npm run dev
```

This starts the backend (FastAPI/uvicorn, port 8000) and frontend (Vite, port
5173) together, with each process's output labeled `[BACKEND]` / `[FRONTEND]`
in one terminal. It runs a preflight check first and exits with a clear error
— missing `.env` values, an occupied port, missing `node_modules` — instead of
starting halfway or failing silently. Open http://localhost:5173.

## Licences

Code: MIT. Datasets and models keep their own licences, several are non-commercial; see docs/ATTRIBUTIONS.md.

## Data and model credits

Full table: `docs/ATTRIBUTIONS.md`.

- **Imagery:** Copernicus Sentinel-2 (contains modified Copernicus Sentinel data, via the Copernicus Data Space
  Ecosystem); Maxar (now Vantor) Open Data Program (CC BY-NC 4.0); IEEE GRSS Data Fusion Contest 2019 Track 1 / JHU/APL US3D (library tiles,
  display resolution).
- **Elevation:** Copernicus DEM GLO-30 (© DLR e.V. / © Airbus Defence and Space GmbH, provided under COPERNICUS by the EU
  and ESA), served for Darjeeling by OpenTopography; FABDEM (Hawker et al. 2022, CC BY-NC-SA 4.0).
- **Models:** Depth Anything V2 Small (Apache-2.0), run by the app; Depth Anything V2 Large (CC BY-NC 4.0), used offline
  only for the four demo regions' precomputed relative-depth images. Yang et al. 2024.
- **Facts and scenario cards:** ThinkHazard! (GFDRR), FAO GAUL 2015, Wikidata, Copernicus EMS / GloFAS, JRC Global
  Surface Water, World Bank / ARUP landslide map, USGS ComCat, NOAA IBTrACS, NOAA SPC.
- **Map tiles:** © OpenStreetMap contributors.
