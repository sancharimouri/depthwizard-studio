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
