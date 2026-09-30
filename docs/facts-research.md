# Facts panel research + Maxar preset (2026-09-30, branch `facts-research-2026-09-30`)

Owner decisions this session:
- Maxar look: SUBTLE.
- Facts panel: diagnose, research and draft only. No live service is built; the owner decides after this report.
- /tmp cap, DFC2019 history cleanup, GCP and the data/ backup are deferred.

No deploys, no push, no API sign-ups.

## Part A: Maxar preset = subtle (2026-09-30)

- **Command:** `.venv/bin/python scripts/bake_static_library.py --maxar-preset subtle`, which ran in 27 s. Output: 76 tiles,
  457 files, **52.5 MB**, written to `frontend/public/library-static/` (gitignored).
- **Manifest:** `data/library_v2_2026-09-29/tile_manifest.json` already had `maxar_display_preset: "subtle"`, and so
  did `dem_maxar/ACTIVE`.
  - The placeholder flag was never stored in the manifest. The bake computes it (`scripts/bake_static_library.py:76`)
    as "`--maxar-preset` given and different from the manifest's value".
  - The previous bake ran with `--maxar-preset medium`, so it wrote `placeholder: true`.
  - With `subtle`, `index.json` and all 5 included Maxar entries now say `{"name": "subtle", "placeholder": false}`.
  - The manifest file was **not edited** (it stays byte-identical).
- **Vercel upload**, measured as before (`frontend/` minus `node_modules/`, `.vercel/`, `dist/` and the
  `.vercelignore`d `public/data/vhr/`): **539 files, 73.0 MB, so 27.0 MB of headroom** under Hobby's 100 MB. It was
  72.9 MB.
  - The local `dist/` is 160.7 MB only because Vite copies the 84 MB `public/data/vhr/`, which the upload excludes.
    Vercel builds from source.
- **Visual check:** headless Chrome (Metal, DevTools protocol) against `vite preview` of `npm run build:web`, clicking
  the library card, then Generate. The preview server was stopped afterwards.
  - `docs/screenshots/2026-09-30_maxar_subtle_a_valley.png`: relief 327 m, rendered at the manifest's default ×10.
    The canopy reads spiky at ×10, so a lower default may suit subtle (owner call).
  - `docs/screenshots/2026-09-30_maxar_subtle_c_town.png`: relief 305 m, at ×4.
  - Both render with the static source: no backend, no Space call.

## Part B: why the Facts panel fell short (read-only diagnosis, 2026-09-30)

### How it works today

- **Frontend.** The panel is `frontend/index.html:838-855`: a `data-box="facts"` section with a lat/lon `<form>` and a
  body. The logic is in `frontend/src/side-panels.js:306-463`.
  - `renderFacts(job)` (`:409`) runs on every jobs re-render (`main.js:1436`, `:2593`) and on opening the box
    (`initFacts`, `:445`).
  - It only fetches while the box is open (`:436`), once per job and location (`:380`).
  - Coordinates come from `factsGeo` (`:315`): `job.input.geo` first, then `job.userGeo` (typed in).
- **Request.** `GET {VITE_API_BASE}/api/facts?lat=<lat>&lon=<lon>` (`side-panels.js:390`). It sends **one point, the
  tile or footprint centre**, never the bounding box, and nothing else.
- **Backend.**
  - `backend/api/facts_routes.py:14-16` validates lat/lon and calls `facts()` in `backend/facts/sources.py:201-222`.
  - The result is cached in process memory for 24 h, keyed by lat/lon rounded to 3 decimals (`:202`). A response
    with any error is not cached (`:220`).
  - Every sub-request uses `httpx` with a 25 s timeout.

| Section | Service and parameters (`backend/facts/sources.py`) | Returns |
|---|---|---|
| place | Nominatim reverse (`:69-97`): `format=jsonv2, zoom=10, addressdetails=1`; global lock at 1 req/s (`:73`) | `display_name` + address parts |
| elevation | Open-Meteo Elevation (`:100-107`): `latitude, longitude` | one GLO-90 value at the centre point |
| seismic | USGS FDSN (`:110-137`): `count` + 2× `query`, `maxradiuskm=250, minmagnitude=4.5, starttime=1900-01-01` | count, 3 largest, latest |
| floods | GDACS event list (`:140-185`): `eventlist=FL, fromdate=2000-01-01, pagesize=100`, **paginates up to 59 pages**, filters to 250 km locally | alert-level flood events at representative points |
| landslides | hard-coded "unavailable" (`:188-193`) | reason text |
| volcanoes | hard-coded "unavailable" (`:194-198`) | reason text |

### What each service does today (re-tested 2026-09-30)

Test points: Darjeeling 27.045/88.260, Chennai coast 13.05/80.28, Jacksonville FL 30.33/-81.66.
- **From this Mac:** all four live sections return "ok".
- **From the deployed Render backend** (`https://depthwizard2-api.onrender.com/api/facts`):
  - The first call took **64.3 s** (cold start plus the GDACS pagination); the second took 2.3 s.
  - **The elevation section failed with `429 Too Many Requests` from Open-Meteo** at both Darjeeling and Chennai.
    Render's egress IPs are shared, so Open-Meteo's per-IP free quota is already spent by other tenants.
  - Because an error disables the cache (`:220`), every later call re-queries all four sources.

Why it fell short, by cause:

1. **Content is not geographic.**
   - PLACE is a postal address. Darjeeling returns "Darjeeling, Darjeeling Pulbazar, Darjeeling, West Bengal, 734101,
     India". That is administrative and postcode text, not terrain.
   - POINT ELEVATION is a single GLO-90 value (Darjeeling 1906 m) at the centre. It duplicates the tile's own DEM,
     which the Terrain Statistics box already shows at 30 m or better.
   - Neither section is a hazard.
2. **Scale mismatch.** The 250 km radius is 25–400× the tile size (10 km for Sentinel-2, 0.6 km for Maxar).
   - Darjeeling gets "276 earthquakes M4.5+ within 250 km", led by the 1934 Bihar–Nepal M8.0. That describes the
     region, not the tile.
   - GDACS floods (Darjeeling 10, Chennai 5, Jacksonville 1) are alerts at one representative point per event, not
     flooded extents.
3. **No hazard levels, only event lists.** Nothing says how prone the tile is. Two of the six sections are permanent
   "Not available" rows. Re-checked today:
   - The NASA Global Landslide Catalog API (`data.nasa.gov/resource/dd9e-wu2v`) still returns 404, and the
     data.nasa.gov CKAN API timed out.
   - The Smithsonian GVP WFS closes the connection (`RemoteProtocolError`).
   - Its Excel export returns 403 "Request Rejected" to a non-browser User-Agent.
4. **Deployment and rate limits.**
   - Open-Meteo returns 429 from shared cloud IPs, and its free API is non-commercial only.
   - Nominatim's usage policy allows at most 1 req/s with no heavy use. The global lock at `:73` serialises all users
     on an instance.
   - GDACS's first call per process downloads the whole flood list since 2000 (19.1 s measured). Its cache lives in
     process memory, so it is lost on every scale-to-zero.
5. **The web build depends on a backend for this one panel.** The static library removed every other backend call
   (HANDOFF §5, `container-measurements.md:258`). Facts is the only remaining one, so a cold or sleeping backend
   shows "Facts service unreachable" or a long "Querying…".
6. **CORS.** The work can't simply move to the browser.
   - GDACS's WAF rejects any request that carries an `Origin` header (403 "Request Rejected", reproduced today) and
     sends no `Access-Control-Allow-Origin`.
   - Open-Meteo and Nominatim would work from the browser, but their usage policies make that risky too.
7. **Licensing.**
   - Open-Meteo's free tier is non-commercial.
   - Nominatim/OSM data is ODbL, which needs attribution; the panel shows the licence string.
   - GDACS has no explicit open licence for reuse.

### What the panel shows now, per input type

| Input | `job.input.geo` | Lat/lon form | Panel |
|---|---|---|---|
| Library tile: Sentinel-2 / Maxar | tile centre from the pack's geotransform (`input-view.js:865`) | hidden (`side-panels.js:425`) | live `/api/facts` for the centre point. Web build: the Render backend (cold start, Open-Meteo 429). Desktop: the sidecar, calling from the user's IP |
| Library tile: DFC2019 | `null` (no georeference; the locations are private) | **shown** | "Location not available: no coordinates in this image's metadata, and none entered." (`:428`); the user may type any lat/lon |
| CDSE scene | footprint centre (`inputGeo`, `input-view.js:939-949`, origin "the searched Sentinel-2 scene's area (centre)") | hidden | live point query |
| Georeferenced upload | footprint centre (same function, origin "the file's geotransform") | hidden | live point query |
| Non-georeferenced upload | `null` | **shown** | empty message until coordinates are entered, then a live point query |

### Lat/long entry box

- **It exists:** `index.html:845-851` (`#xp-facts-lat`, `#xp-facts-lon`, "Look up").
- **It is wired:** the `submit` handler at `side-panels.js:452-462` validates the range, sets `job.userGeo`
  (`:460`) and re-renders. That triggers the same `/api/facts` point query, and the entered values are shown as
  "from coordinates you entered".
- It is shown whenever the input has no geo, which includes DFC2019 library tiles.

### Earthquake placeholder

- **Where it is:**
  - Button: `frontend/index.html:873-876` (`#final-demo-earthquake-button`, tagged "PLACEHOLDER").
  - Handler: `frontend/src/main.js:2667-2675`.
  - State: `main.js:2297-2304`.
  - Note text: `main.js:2275-2276`.
  - Overlay: `frontend/src/terrain.js:764-815` (`computePlaceholderSlopeDanger`, `setEarthquakeOverlay`).
  - The old demo page has a disabled "EARTHQUAKE SOON" button at `index.html:447-454`.
- **What it shows:** a red vertex tint on the terrain, alongside the note "Earthquake — PLACEHOLDER, not a seismic
  hazard model…".
- **What it is faked from:** the tile's own DEM slope only.
  - Slope in degrees comes from central differences on the mesh heights (`terrain.js:773-787`).
  - It is normalised within the tile: no tint below the median slope, full red from the 95th percentile
    (`:790-795`).
  - There is **no seismic data**: a flat tile in a high-hazard zone looks safe, and a steep tile in a stable craton
    looks dangerous.

## Part C: sources for geographic facts (researched and tested 2026-09-30)

**Scope:** geographic facts only: terrain, natural hazards, disasters, dangerous zones, hydrology. No cultural,
historical or tourism content.
- The real calls were run from this Mac at three points: Darjeeling 27.045/88.260, Chennai coast 13.05/80.28, and
  outside India, Jacksonville FL 30.33/-81.66.
- "Box" means a ~10 × 10 km window around the point, the Sentinel-2 tile scale.

### C1. Facts computed from the tile's own DEM (no service)

- **Implementation:** `scripts/draft_library_facts.py` (`dem_facts`). Every one of these facts is labelled
  "derived from the DEM" in its text, with `origin: "derived"`.
- **Input:** the TERRAIN band of the tile's pack (FABDEM v1-2 bare earth). Maxar's 1.2 m TERRAIN is FABDEM bicubic,
  so it is block-averaged back to ~30 m first. That way slope and TRI mean the same thing as for Sentinel-2.
- **What is computed:**
  - relief (max − min);
  - elevation percentiles p5 / p50 / p95;
  - mean and p90 slope;
  - **% steeper than 30°** (a landslide-prone proxy);
  - terrain ruggedness index (Riley, mean |Δz| to the 8 neighbours);
  - **% of the ground within 5 m / 10 m of sea level** (a coastal-flooding proxy; FABDEM heights are above the
    EGM2008 geoid, ±2 m).
- **Cost:** 0 MB extra (the packs are already bundled), about 50 ms per tile, and no network.
- **Values at the test tiles:**
  - Darjeeling: relief 1,919 m; 37% of the ground > 30°; TRI 40 m.
  - Chennai: relief 12 m, mean slope 0.3°; **99.6% within 10 m of sea level**, 48% within 5 m.
- **Caveats:**
  - A 30 m grid under-counts steep ground, which is why the slope fact is "medium" confidence.
  - FABDEM is **CC BY-NC-SA 4.0**, so derived facts inherit "non-commercial" (see licences).
- **DFC2019:** no C1 facts. Its pack surface is ground plus Method 6 predicted heights, so slope there measures
  building walls, not terrain. Also, 35 of its 50 grounds are city-level approximations.

### C2. External sources

Measured from this Mac. "Live" means it is callable at request time from the backend; "browser" means it is
CORS-enabled and callable from the web page itself. Memory is for a 512 MiB Cloud Run instance.

| Source | Facts it yields | Coverage / resolution | Licence (commercial? attribution) | Access: key / limits / CORS | Observed today (3 points) | Live / bundled | Bundled size, cheapest form | Memory on 512 MiB |
|---|---|---|---|---|---|---|---|---|
| **USGS ComCat** (FDSN event API) | earthquake counts, largest and latest events near a point or box | global; M4.5+ complete from ~1973 | public domain, commercial OK. "U.S. Geological Survey, ANSS ComCat" | no key; 20,000 events max per query; CORS `*` | 200 at all 3; `count` 0.5–1.0 s, `query` 2–8 s. Darjeeling 47 × M4.5+ within 100 km since 1973 (largest M6.9, 2011, 77 km); Chennai 0; Jacksonville 0 | **live + browser** | not needed (a catalogue extract would be ~2–5 MB for M4.5+) | ~0 (JSON of ≤ 100 events) |
| **GEM Global Seismic Hazard Map** (PGA, 475-year) | PGA class for the tile | global raster (~0.05°, not downloaded) | **CC BY-NC-SA 4.0: non-commercial only**; commercial use needs a GEM licence request | download via a request form, no API | page reachable 200; licence text confirmed; raster **not fetched** (it needs the form) | bundled only | not measured (a clipped uint8 class raster would be ≤ 5 MB) | tiny |
| **ThinkHazard! (GFDRR)** | hazard LEVEL (high / medium / low / very low) per district for river, urban and coastal flood, earthquake, landslide, tsunami, volcano, cyclone, wildfire, water scarcity, extreme heat | global, admin-2 (GAUL) | hazard levels **CC BY**, text CC BY-SA, code GPL-3. "ThinkHazard! – GFDRR" | no key; **no CORS**; **no point lookup** (name search only; `fromlonlat` is 404) | search 0.3–0.7 s, report 0.2–0.3 s. Chennai: cyclone HIGH, earthquake / river / coastal flood / tsunami MED. Duval FL: river flood HIGH, cyclone HIGH. "Darjeeling" → no hit; GAUL spells it "Darjiling" (then landslide HIGH, earthquake MED). Sikkim's renamed "Namchi" → no match | live (server) or bundled | levels for every district: `admindiv_hazardsets/{HT}.json` is 5.0 MB × 11 hazards (JSON). Cheapest: rasterise the levels to a 0.05° grid at 2 bits × 8 hazards, ~2–5 MB PNG. A point lookup needs the geometry, so a pre-rasterised grid avoids shipping GAUL polygons | a 0.05° grid of 8 hazards ≈ 7200 × 3600 × 2 B = 52 MB if decoded whole: **read windows from a COG instead** (≈ 0) |
| **World Bank Global Landslide Hazard Map** (ARUP 2021) | landslide hazard class 1–4 (total / rainfall / earthquake-triggered) | global, 30″ (~1 km) | Data Catalog dataset 0037584; licence text **not machine-readable today, verify** (WB data is usually CC BY 4.0) | **COGs** on datacatalogfiles.worldbank.org, no key | window read 0.1–7 s. Darjeeling class 3–4 (75% class 4); Chennai 1–2; Jacksonville 1 | **live (COG window)** or bundled | LS_TH 58 MB, rainfall mean 152 MB, EQ-triggered 59 MB. Clipped to India plus the US cities: < 3 MB | ≈ 0 (one 256² block) |
| **NASA Global Landslide Catalog** (COOLR) | historical landslide points | global points, 2007+ | NASA open data | the old Socrata API | **404** (`data.nasa.gov/resource/dd9e-wu2v`); data.nasa.gov CKAN **timed out** | none today | — | — |
| **NASA LHASA susceptibility** | 1 km landslide susceptibility | global | NASA open data | — | `gpm.nasa.gov/landslides/*` pages **404** today; not verified | none today | — | — |
| **JRC / CEMS GloFAS river flood hazard** v2.1.2 | % of the tile in the 1-in-10…500-year flood extent, max depth | global land, 3″ (~90 m), large rivers only | **CC BY 4.0**, commercial OK. "© European Union, Copernicus Emergency Management Service" | no key; tiled 256 GeoTIFF over HTTPS (range reads work); no CORS | window 0.6–2.6 s. Darjeeling 0%; Chennai box 20% (max 7.6 m); Jacksonville box 10% (max 4.8 m) | **live (window)** or bundled | RP100 depth 3.44 GB (271 tiles), reclass 516 MB. Cheapest: RP100 extent as a 1-bit mask clipped to the library tiles, KB-scale; global uint8 at 0.01° ~10–20 MB | ≈ 0 |
| **JRC Global Surface Water** v1.4 (Pekel 2016) | % of the tile ever water / permanent water (1984–2021) | global, 30 m | free, no restriction of use (Copernicus). "EC JRC/Google, Pekel et al. 2016" | no key; tiles and PNG tiles CORS `*` | window 4.4–5.1 s. Darjeeling 0%; Chennai box 46% ever / 43% permanent (includes sea); Jacksonville 23 / 21% | **live (window)** | global occurrence ~15 GB+; not worth bundling | ≈ 0 |
| **WRI Aqueduct Floods** (riverine + coastal inundation) | coastal flood depth by return period | global, 30″ | reported CC BY 4.0; **not verified today** (the old S3 index is 404) | — | download location not found today | bundled (if found) | — | — |
| **NOAA IBTrACS** v04r01 | cyclone tracks near the tile: count, strongest | global, 3-hourly points since 1842 | public domain. "Knapp et al. (2010), IBTrACS, NOAA NCEI" | CSV download, no API | since-1980 CSV 144 MB; ALL 332 MB. Chennai 23 storms within 100 km since 1980, 2 at ≥ 64 kt; Jacksonville 31, strongest Matthew 2016 (98 kt) | **bundled** | points as int16 (lat, lon, year, wind, storm index), since 1980: ~700k rows × 10 B ≈ 7 MB, ~3 MB gzipped | ~7 MB if loaded; or precompute per tile at bake time (0) |
| **NOAA NCEI Hazards** (hazel API: significant earthquakes, tsunamis, volcano locations and eruptions) | tsunami events and run-ups, significant (damaging) earthquakes, volcanoes near the tile | global, historical | public domain. "NCEI/WDS Global Significant Earthquake / Tsunami / Volcano Database" | no key; CORS `*` | 200 at all, 1.0–1.9 s. Tsunami events in a 10–16°N, 78–83°E box around Chennai: returned (e.g. Nagapattinam, year 900); Darjeeling significant EQ: 2006 Sikkim M5.3; volcano locations ±5°: Darjeeling 0, Chennai 1, Jacksonville 0 | **live + browser** | not needed | ≈ 0 |
| **Smithsonian GVP** (Volcanoes of the World) | Holocene volcanoes | global, ~1,300 volcanoes | GVP terms; cite "Global Volcanism Program, Smithsonian Institution" | WFS **broken** (`RemoteProtocolError`); Excel export 403 to non-browser UAs, 1.1 MB with a browser UA | see access | bundled (use NCEI volcano locations live instead) | ~100 KB CSV | ≈ 0 |
| **NASA FIRMS** (active fire) | fire detections near the tile | global, 375 m / 1 km, daily | NASA open data | **needs a MAP_KEY** (free registration): `Invalid MAP_KEY` without it. **Not signed up**, per the rules | blocked on the key | live (with key) | — | — |
| **GDACS** (already used) | alert-level flood, cyclone, earthquake, volcano, drought and wildfire events | global since 2000 | no explicit open licence | no key; the WAF **rejects requests carrying an `Origin` header**, so no browser use | 200 server-side in 0.7 s; the full flood list takes 19 s | live (server) | — | a few MB for the list |
| **OpenStreetMap Overpass** | named peaks, rivers, glaciers, coastline, wetlands | global | **ODbL**, commercial OK with share-alike on the database. "© OpenStreetMap contributors" | no key; public instance fair use (< 10k queries/day); CORS `*` | Darjeeling 3.8 s (Observatory Hill 2188 m, 16 rivers); Chennai 3.5 s, **mostly temple tanks** ("…Koil Kulam", cultural water bodies, need filtering); Jacksonville **504** (overloaded) | live (flaky) | — | ≈ 0 |
| **GeoNames** | named terrain features | global | CC BY 4.0 | **needs a username** (free account); the demo account is exhausted; the free API is HTTP only (HTTPS cert mismatch) | blocked on the account | dumps are bundle-able (`allCountries.zip` ~400 MB; the feature-class T/H subset would be ~30 MB) | — | — |
| **Wikidata SPARQL** | named natural features inside the tile (class-filtered: mountain, hill, river, glacier, lake, beach, waterfall, valley…) with elevation P2044 | global, crowd-sourced | **CC0** | no key; CORS `*`; 60 s query limit | 0.5–1.3 s. Darjeeling: Observatory Hill 2188 m, Tiger Hill, Senchal Lake; Chennai: Cooum River, Adyar River, Marina Beach; Jacksonville: Arlington River, Ribault River | **live + browser** | — | ≈ 0 |
| Nominatim / Open-Meteo (current) | address, point elevation | global | ODbL / CC BY 4.0 data, **non-commercial free API** | 1 req/s / per-IP quota | Open-Meteo **429 from Render** | drop both: not geographic facts (Part B) | — | — |

**Better sources than the starting list:**
- The **World Bank landslide COGs** give a real 1 km landslide hazard class, window-readable. They replace the dead
  NASA catalogue for the "landslide" row.
- **NCEI hazel** covers tsunamis, significant earthquakes and volcanoes with CORS in one API. It replaces the broken
  GVP WFS.
- **Wikidata** returns named features more cleanly than Overpass.
- **ThinkHazard!** is the one source that gives hazard *levels* for flood, cyclone, tsunami and wildfire.

### C3. Recommended design per input type

The principle: **tile-scale facts come from rasters read over the tile's bbox; point facts come from event
catalogues within a radius.** A district rating is labelled as district-level. Every section fails independently,
and a failed or empty source adds nothing (as `safe()` does in the draft script).

| Input | Facts | How |
|---|---|---|
| Library tile (web build) | pre-baked, curated only (Part E2) | zero calls, 0 MB runtime; box hidden if none are curated |
| CDSE scene or georeferenced upload | C1 on the job's own DEM, plus bbox facts: JRC RP100 %, WB landslide class, GSW %, Wikidata features; plus radius facts: USGS, NCEI tsunami / volcano; district: ThinkHazard via a bundled level grid | C1 in the browser from the already-loaded terrain (like the slope placeholder does today); the rasters by backend COG window reads; the catalogues from the browser (CORS `*`) or the backend |
| Non-georeferenced upload + typed lat/lon | **point-based only**: USGS within radius, NCEI, ThinkHazard district level, WB landslide class of the one ~1 km cell | no bbox facts (the footprint is unknown); no C1 geography claims (the relative-depth mesh is not a DEM) |
| Any source unreachable | the other sections still render; that row says "Source unreachable" (the existing `factSection` "error" state) | per-source timeouts ≤ 8 s; results cached per tile |

**Cost of each option:**

| Option | Image MB | Runtime memory | Money | Notes |
|---|---|---|---|---|
| **1. Curated library only** (E2, built) | +~0.05 MB in `library-static` (text only) | 0 | $0 | no backend call; the facts are only as fresh as the last bake |
| **2. Browser-only live** (USGS, NCEI hazel, Wikidata) + C1 in the browser | 0 | browser only | $0 | removes the backend dependency entirely; no hazard levels (ThinkHazard has no CORS, the rasters would need range reads from the browser: GSW has CORS, JRC and WB don't) |
| **3. Backend live** (option 2 + COG window reads of JRC / WB / GSW + ThinkHazard by name) | 0 (rasterio is already in the container) | ~10–30 MB per request (GDAL block cache; set `GDAL_CACHEMAX=32`) | Cloud Run requests plus egress: ~1–3 MB per call, pennies per 1,000 calls | 3–8 s cold per tile, cache per bbox; ThinkHazard name matching is fragile |
| **4. Backend + bundled grids** (option 3, with ThinkHazard levels as a 0.05° COG ~3 MB, IBTrACS points ~3 MB gz, WB landslide clip ~3 MB) | **+~9 MB** | + ~10 MB | $0 extra | robust, with no fragile name lookups; needs a one-off build script, and the licences must be carried into the UI |

**Recommendation:**
- Keep option 1 for the library now.
- Build option 3 minus ThinkHazard, plus the bundled ThinkHazard grid from option 4, for CDSE scenes and uploads.
- Drop Nominatim and Open-Meteo from the panel.
- **Licence watch-outs before any commercial use:**
  - FABDEM (NC), so the derived facts inherit it;
  - GEM (NC);
  - the World Bank landslide licence (unverified);
  - GDACS (no explicit licence);
  - OSM (ODbL share-alike).

## Part D: a real earthquake-prone overlay (research only, 2026-09-30)

**Scale check:**
- GEM's GSHM is ~0.05° (~5 km); ThinkHazard's earthquake level is per district.
- A 10 km Sentinel-2 tile covers **~2 × 2 GEM cells**. A Maxar tile (0.6 km) or a DFC2019 tile (~0.15 km) sits
  **inside one cell**.
- A per-vertex colour overlay would therefore be uniform (one colour over the whole mesh) on almost every library
  tile. It would look like a rendering bug, and it adds nothing over one number.

**Recommendation:** replace the slope-based placeholder (`terrain.js:764-815`) with a **hazard card plus epicentres**,
not a mesh tint:
1. **Hazard class** for the tile:
   - ThinkHazard earthquake level (CC BY) for the district;
   - optionally GEM PGA (475-year) as a number, only while the app is non-commercial (CC BY-NC-SA).
2. **Nearby epicentres** from USGS ComCat (M4.5+ since 1973, 150 km), in a small 2D inset map around the tile
   outline, sized by magnitude. They can also be drawn as markers on the 3D ground plane, projected to the mesh edges
   when outside the tile.
3. **If a tile ever spans more than one hazard cell** (large uploads), tint by the GEM class raster clipped to the
   tile. Otherwise don't tint.
4. **Text:** "Earthquake hazard: MEDIUM (district level). 47 M4.5+ earthquakes within 100 km since 1973; largest
   M6.9 (2011, 77 km)." Every value is retrieved and dated.

**Mock-up:** `docs/screenshots/2026-09-30_earthquake_overlay_mockup.png`. It uses real data:
- the Darjeeling FABDEM hillshade;
- 73 real USGS M4.5+ epicentres within 150 km since 1973 (queried today);
- the ThinkHazard Darjiling earthquake level MEDIUM.

It is a static matplotlib sketch; nothing is wired.

**Attribution text:**
- "Earthquakes: U.S. Geological Survey, ANSS Comprehensive Catalog (public domain)."
- "Hazard level: ThinkHazard! – GFDRR (CC BY 4.0)."
- If GEM is used: "Seismic hazard: GEM Global Seismic Hazard Map v2023.1, Global Earthquake Model Foundation (CC
  BY-NC-SA 4.0)."
