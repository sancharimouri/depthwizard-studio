import * as THREE from "three";
import { createTerrain } from "./terrain.js";
import { createTerrainViewer } from "./viewer.js";

const canvas = document.getElementById("terrain-canvas");

// Explore's main viewer. Vertical-only rig offset so the structure clears
// the panels occupying the top of the page — no horizontal offset, kept
// centered on X so it lines up with controls.target's X, which is also the
// flythrough's zoom pivot; an X mismatch between the two is what drifted
// the structure sideways as the flythrough zoomed in.
const exploreViewer = createTerrainViewer(canvas, { rigOffsetY: -10 });
exploreViewer.resize(window.innerWidth, window.innerHeight);

const { scene, camera, renderer, controls, terrainRig } = exploreViewer;


// ============================================================
// HELPERS
// ============================================================

function sleep(ms) {
    return new Promise(resolve => {
        setTimeout(resolve, ms);
    });
}

function layerDescriptionText(layer, regionKey = currentRegionKey) {
    const source = REGIONS[regionKey]?.elevationSource ?? "DSM";

    switch (layer) {
        case "satellite-flat":
            return "Sentinel-2 RGB imagery — flat";
        case "depth-flat":
            return "DAv2 relative depth — flat, not absolute elevation";
        case "elevation-flat":
            return `Metric elevation from ${source} — flat`;
        case "dsm-3d":
            return "DAv2 relative depth draped on extruded terrain";
        case "elevation-3d":
            return `Metric elevation from ${source} — color-ramped by elevation`;
        case "satellite-3d":
            return "Sentinel-2 RGB draped on extruded terrain";
        default:
            return "Sentinel-2 RGB imagery";
    }
}

function setLayerDescription(layer) {
    const description = document.getElementById("layer-description");

    if (!description) {
        return;
    }

    description.textContent = layerDescriptionText(layer);
}


// ============================================================
// REGION RAIL
// ============================================================

const REGIONS = {
    darjeeling: {
        label: "Darjeeling",
        elevationSource: "OpenTopography DSM",
        crsEpsg: "32645",
        resolution: "10m",
        inferenceTime: "1.17s",
    },
    kolkata: {
        label: "Kolkata",
        elevationSource: "Copernicus GLO-30 DSM",
        crsEpsg: "32645",
        resolution: "10m",
        inferenceTime: "1.67s",
    },
    bardhaman: {
        label: "Bardhaman",
        elevationSource: "Copernicus GLO-30 DSM",
        crsEpsg: "32645",
        resolution: "10m",
        inferenceTime: "0.66s",
    },
    sundarbans: {
        label: "Sundarbans",
        elevationSource: "Copernicus GLO-30 DSM",
        crsEpsg: "32645",
        resolution: "10m",
        inferenceTime: "0.66s",
    },
};


// ============================================================
// LOAD TERRAIN (per region — region rail switches between these)
// ============================================================

let currentTerrain = null;
let currentRegionKey = null;
let running = false;

function activateLayer(layer) {
    if (!currentTerrain) {
        return;
    }

    currentTerrain.setLayer(layer);

    // Scoped to #page-1 — Final Demo's box 8 reuses the same .layer-button
    // class for visual parity with Explore, and has its own independent
    // activateFinalDemoLayer() below; an unscoped selector here would
    // cross-wire the two viewers' active states.
    document.querySelectorAll("#page-1 .layer-button").forEach(button => {
        button.classList.toggle("active", button.dataset.layer === layer);
    });

    setLayerDescription(layer);
}

// Selecting a layer by hand (as opposed to the flood toggle switching to
// it itself) always clears the flood overlay — it only makes sense on
// top of the DSM layer.
function selectLayer(layer) {
    if (floodActive) {
        setFloodActive(false);
    }

    activateLayer(layer);
}

function updateStatsAndLabels(regionKey, terrain, terrainData) {
    const region = REGIONS[regionKey];

    const elevationElement = document.getElementById("elevation-value");
    if (elevationElement) {
        elevationElement.textContent = `${Math.round(terrain.elevationMin)}–${Math.round(terrain.elevationMax)} m`;
    }

    const sourceElement = document.getElementById("terrain-source-value");
    if (sourceElement) {
        sourceElement.textContent = region.elevationSource;
    }

    const inferenceStat = document.getElementById("stat-inference");
    if (inferenceStat) {
        inferenceStat.textContent = region.inferenceTime;
    }

    const resolutionStat = document.getElementById("stat-resolution");
    if (resolutionStat) {
        resolutionStat.textContent = region.resolution;
    }

    const gridStat = document.getElementById("stat-grid");
    if (gridStat) {
        gridStat.textContent = `${terrainData.width}×${terrainData.height}`;
    }

    const epsgStat = document.getElementById("stat-epsg");
    if (epsgStat) {
        epsgStat.textContent = region.crsEpsg;
    }
}

async function loadRegion(regionKey) {
    const { terrain, terrainData } = await exploreViewer.loadRegion(regionKey);

    currentTerrain = terrain;
    currentRegionKey = regionKey;


    // --------------------------------------------------------
    // Stats, labels, region rail, default layer
    // --------------------------------------------------------

    updateStatsAndLabels(regionKey, terrain, terrainData);

    document.querySelectorAll("#page-1 .region-card").forEach(card => {
        card.classList.toggle("active", card.dataset.region === regionKey);
    });

    activateLayer("satellite-3d");
    resetRunState();

    return terrain;
}


// ============================================================
// LAYER BUTTONS
// ============================================================

document.querySelectorAll("#page-1 .layer-button").forEach(button => {
    button.addEventListener("click", () => { selectLayer(button.dataset.layer); });
});


// ============================================================
// REGION RAIL — switching
// ============================================================

// [data-region] excludes the "Build Your Own" card — it reuses
// .region-card for consistent styling but isn't a region to load.
document.querySelectorAll("#page-1 .region-card[data-region]").forEach(card => {
    card.addEventListener("click", async () => {
        const region = card.dataset.region;

        if (running || region === currentRegionKey) {
            return;
        }

        await loadRegion(region);
    });
});


// ============================================================
// RECONSTRUCTION SEQUENCE
// (shared by the RUN RECONSTRUCTION button and the mock upload
// flow — both step through all 6 visualization states in order,
// flat row first, then the extruded 3D row)
// ============================================================

const STAGES = [
    { layer: "satellite-flat", caption: "Satellite", duration: 800 },
    { layer: "depth-flat", caption: "Relative Depth", duration: 900 },
    { layer: "elevation-flat", caption: "Elevation", duration: 900 },
    { layer: "dsm-3d", caption: "DSM", duration: 1000 },
    { layer: "elevation-3d", caption: "Metric Elevation", duration: 1000 },
    { layer: "satellite-3d", caption: "True Color", duration: 1300 },
];

const STAGE_TOTAL_MS = STAGES.reduce((sum, s) => sum + s.duration, 0);

const progressBar = document.getElementById("pipeline-progress");
const progressFill = document.getElementById("pipeline-progress-fill");

const runPanel = document.getElementById("run-panel");
const runButton = document.getElementById("run-button");
const postRunPanel = document.getElementById("post-run-panel");
const runAgainButton = document.getElementById("run-again-button");
const flythroughButton = document.getElementById("flythrough-button");
const floodButton = document.getElementById("flood-button");

function resetRunState() {
    running = false;

    if (runPanel) {
        runPanel.hidden = false;
    }
    if (postRunPanel) {
        postRunPanel.hidden = true;
    }
    if (runButton) {
        runButton.disabled = false;
        runButton.textContent = "▶ RUN RECONSTRUCTION";
    }
    if (progressBar) {
        progressBar.hidden = true;
    }
    if (progressFill) {
        progressFill.style.width = "0%";
    }

    setFloodActive(false);
    resetFlythrough();
}

async function runReconstruction() {
    if (running) {
        return;
    }

    running = true;

    setFloodActive(false);
    resetFlythrough();

    if (postRunPanel) {
        postRunPanel.hidden = true;
    }
    if (runPanel) {
        runPanel.hidden = false;
    }
    if (runButton) {
        runButton.disabled = true;
        runButton.textContent = "PROCESSING…";
    }
    if (progressBar) {
        progressBar.hidden = false;
    }

    const description = document.getElementById("layer-description");

    let elapsed = 0;

    for (const stage of STAGES) {
        activateLayer(stage.layer);

        if (description) {
            description.textContent = stage.caption;
        }

        await sleep(stage.duration);

        elapsed += stage.duration;

        if (progressFill) {
            progressFill.style.width = `${Math.round((elapsed / STAGE_TOTAL_MS) * 100)}%`;
        }
    }

    if (description) {
        description.textContent = "Reconstruction complete";
    }

    setTimeout(() => {
        if (progressBar) {
            progressBar.hidden = true;
        }
        if (progressFill) {
            progressFill.style.width = "0%";
        }

        running = false;

        if (runPanel) {
            runPanel.hidden = true;
        }
        if (postRunPanel) {
            postRunPanel.hidden = false;
        }
    }, 800);
}

runButton?.addEventListener("click", runReconstruction);
runAgainButton?.addEventListener("click", runReconstruction);


// ============================================================
// FLYTHROUGH
// (one-shot camera dolly-in — no loop back out — the terrain rig
// keeps auto-rotating underneath it via its own render-loop update)
//
// Factored into a reusable controller so Workbench's Final Demo box
// (box 8) can offer the same feature on its own independent viewer,
// camera, and terrain rig without sharing state with Explore's.
// ============================================================

const FLYTHROUGH_ROTATE_SPEED_MULTIPLIER = 2;
const FLYTHROUGH_DURATION_MS = 9500;

function createFlythroughController({ camera: flCamera, controls: flControls, terrainRig: flRig, button }) {
    let flythrough = null;

    function reset() {
        flythrough = null;
        flRig.setSpeedMultiplier(1);

        if (button) {
            button.disabled = false;
            button.textContent = "◎ FLYTHROUGH";
        }
    }

    function start() {
        if (flythrough || !button || button.disabled) {
            return;
        }

        const startDistance = flCamera.position.distanceTo(flControls.target);
        const endDistance = Math.max(flControls.minDistance, startDistance * 0.5);

        flythrough = {
            startDistance,
            endDistance,
            startTime: performance.now(),
            duration: FLYTHROUGH_DURATION_MS,
        };

        // 2x idle rotation speed for the zoom-in and for the continued
        // rotation afterward — stays elevated until the next reset.
        flRig.setSpeedMultiplier(FLYTHROUGH_ROTATE_SPEED_MULTIPLIER);

        button.disabled = true;
        button.textContent = "FLYING THROUGH…";
    }

    function update() {
        if (!flythrough) {
            return;
        }

        const t = Math.min(1, (performance.now() - flythrough.startTime) / flythrough.duration);
        const eased = 1 - Math.pow(1 - t, 3);

        const distance = flythrough.startDistance + (flythrough.endDistance - flythrough.startDistance) * eased;

        const offset = flCamera.position.clone().sub(flControls.target);
        offset.setLength(distance);
        flCamera.position.copy(flControls.target).add(offset);

        if (t >= 1) {
            flythrough = null;

            if (button) {
                button.textContent = "✓ FLYTHROUGH";
            }
        }
    }

    button?.addEventListener("click", start);

    return { start, update, reset };
}

const exploreFlythrough = createFlythroughController({
    camera,
    controls,
    terrainRig,
    button: flythroughButton,
});

function resetFlythrough() {
    exploreFlythrough.reset();
}

function updateFlythrough() {
    exploreFlythrough.update();
}


// ============================================================
// DANGER ZONES — FLOOD (toggle) / EARTHQUAKE (coming soon)
// ============================================================

let floodActive = false;

function setFloodActive(active) {
    floodActive = active;

    floodButton?.classList.toggle("active", floodActive);
    currentTerrain?.setFloodOverlay(floodActive);
}

floodButton?.addEventListener("click", () => {
    if (!floodActive) {
        activateLayer("dsm-3d");
    }

    setFloodActive(!floodActive);
});


// ============================================================
// WORKBENCH — DIRECT UPLOAD (Scene Input box 1, alternative to scene
// search). Was previously wired to a "upload-trigger" button that got
// deleted from Page 1's markup in an earlier cleanup, leaving this
// modal/logic as dead code with nothing to open it — rebuilt here as
// Workbench's actual upload entry point instead.
// ============================================================

const sceneUploadTrigger = document.getElementById("scene-upload-trigger");
const sceneUploadModal = document.getElementById("scene-upload-modal");
const sceneUploadBackdrop = document.getElementById("scene-upload-backdrop");
const sceneUploadCancel = document.getElementById("scene-upload-cancel");
const sceneUploadDropzone = document.getElementById("scene-upload-dropzone");
const sceneUploadInput = document.getElementById("scene-upload-input");

function openSceneUploadModal() {
    if (sceneUploadModal) {
        sceneUploadModal.hidden = false;
    }
}

function closeSceneUploadModal() {
    if (sceneUploadModal) {
        sceneUploadModal.hidden = true;
    }
    sceneUploadDropzone?.classList.remove("drag-over");
}

function handleUploadedFile(file) {
    if (!file) {
        return;
    }

    closeSceneUploadModal();
    selectUploadedScene(file);
}

sceneUploadTrigger?.addEventListener("click", openSceneUploadModal);
sceneUploadCancel?.addEventListener("click", closeSceneUploadModal);
sceneUploadBackdrop?.addEventListener("click", closeSceneUploadModal);

sceneUploadInput?.addEventListener("change", () => {
    handleUploadedFile(sceneUploadInput.files?.[0]);
});

sceneUploadDropzone?.addEventListener("dragover", event => {
    event.preventDefault();
    sceneUploadDropzone.classList.add("drag-over");
});

sceneUploadDropzone?.addEventListener("dragleave", () => {
    sceneUploadDropzone.classList.remove("drag-over");
});

sceneUploadDropzone?.addEventListener("drop", event => {
    event.preventDefault();
    sceneUploadDropzone.classList.remove("drag-over");
    handleUploadedFile(event.dataTransfer?.files?.[0]);
});

document.addEventListener("keydown", event => {
    if (event.key === "Escape" && sceneUploadModal && !sceneUploadModal.hidden) {
        closeSceneUploadModal();
    }
});


// ============================================================
// LIVE LOCATION AUTOCOMPLETE (OpenStreetMap Nominatim, no API key)
// ============================================================

const GEOCODE_DEBOUNCE_MS = 400;
const GEOCODE_MIN_QUERY_LENGTH = 3;

const geocodeInput = document.getElementById("geocode-location-input");
const geocodeDropdown = document.getElementById("geocode-dropdown");
const geocodeDetails = document.getElementById("geocode-details");

let selectedGeocodeResult = null;
let geocodeDebounceTimer = null;
let geocodeRequestId = 0;

async function queryNominatim(query) {
    const url =
        `https://nominatim.openstreetmap.org/search?format=jsonv2&addressdetails=1&limit=5&q=` +
        encodeURIComponent(query);

    const response = await fetch(url, {
        headers: { "Accept": "application/json" },
    });

    if (!response.ok) {
        throw new Error(`Nominatim request failed: ${response.status}`);
    }

    return response.json();
}

function renderGeocodeStatus(text) {
    if (!geocodeDropdown) {
        return;
    }
    geocodeDropdown.innerHTML = `<div class="geocode-status">${text}</div>`;
    geocodeDropdown.hidden = false;
}

function renderGeocodeResults(results) {
    if (!geocodeDropdown) {
        return;
    }

    if (!results || results.length === 0) {
        renderGeocodeStatus("No matches");
        return;
    }

    geocodeDropdown.innerHTML = "";

    results.forEach(result => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "geocode-result";
        item.textContent = result.display_name;
        item.addEventListener("click", () => selectGeocodeResult(result));
        geocodeDropdown.appendChild(item);
    });

    geocodeDropdown.hidden = false;
}

function renderGeocodeDetails(result) {
    if (!geocodeDetails) {
        return;
    }

    const address = result.address ?? {};
    const region = address.state || address.region || address.county || "";
    const regionCountry = [region, address.country].filter(Boolean).join(", ") || "—";

    geocodeDetails.innerHTML = `
        <div class="geocode-detail-row"><span>Place</span><span>${result.display_name}</span></div>
        <div class="geocode-detail-row"><span>Lat, Lng</span><span class="numeric-mono">${Number(result.lat).toFixed(4)}, ${Number(result.lon).toFixed(4)}</span></div>
        <div class="geocode-detail-row"><span>Region</span><span>${regionCountry}</span></div>
    `;
    geocodeDetails.hidden = false;
}

function selectGeocodeResult(result) {
    selectedGeocodeResult = result;

    if (geocodeInput) {
        geocodeInput.value = result.display_name;
    }
    if (geocodeDropdown) {
        geocodeDropdown.hidden = true;
        geocodeDropdown.innerHTML = "";
    }

    renderGeocodeDetails(result);

    if (sceneSearchTrigger) {
        sceneSearchTrigger.disabled = false;
    }
}

geocodeInput?.addEventListener("input", () => {
    const query = geocodeInput.value.trim();
    const requestId = ++geocodeRequestId;

    selectedGeocodeResult = null;
    if (geocodeDetails) {
        geocodeDetails.hidden = true;
    }
    if (sceneSearchTrigger) {
        sceneSearchTrigger.disabled = true;
    }

    if (geocodeDebounceTimer) {
        clearTimeout(geocodeDebounceTimer);
    }

    if (query.length < GEOCODE_MIN_QUERY_LENGTH) {
        if (geocodeDropdown) {
            geocodeDropdown.hidden = true;
        }
        return;
    }

    geocodeDebounceTimer = setTimeout(async () => {
        renderGeocodeStatus("Searching…");

        try {
            const results = await queryNominatim(query);
            if (requestId !== geocodeRequestId) {
                return; // a newer keystroke has already superseded this lookup
            }
            renderGeocodeResults(results);
        } catch (error) {
            if (requestId !== geocodeRequestId) {
                return;
            }
            console.error("Nominatim lookup failed:", error);
            renderGeocodeStatus("Lookup failed — try again");
        }
    }, GEOCODE_DEBOUNCE_MS);
});

document.addEventListener("click", event => {
    if (
        geocodeDropdown && !geocodeDropdown.hidden &&
        event.target !== geocodeInput && !geocodeDropdown.contains(event.target)
    ) {
        geocodeDropdown.hidden = true;
    }
});


// ============================================================
// MOCK SCENE SEARCH (Workbench "Scene Input" prototype flow) —
// AOI / date range / cloud cover stay mocked (no live Copernicus STAC
// query yet, that needs OAuth credentials this project doesn't have),
// but now searches around the real coordinates resolved above.
// ============================================================

const MOCK_SCENES = [
    {
        id: "S2A_MSIL2A_KOLKATA_20251118",
        thumb: "/data/kolkata/satellite.png",
        date: "2025-11-18",
        cloud: 8,
    },
    {
        id: "S2B_MSIL2A_BARDHAMAN_20251103",
        thumb: "/data/bardhaman/satellite.png",
        date: "2025-11-03",
        cloud: 14,
    },
];

const sceneSearchTrigger = document.getElementById("scene-search-trigger");
const sceneSearchModal = document.getElementById("scene-search-modal");
const sceneSearchBackdrop = document.getElementById("scene-search-backdrop");
const sceneSearchCancel = document.getElementById("scene-search-cancel");
const sceneSearchButton = document.getElementById("scene-search-button");
const sceneSearchStatus = document.getElementById("scene-search-status");
const sceneSearchResults = document.getElementById("scene-search-results");
const sceneSearchLocationReadout = document.getElementById("scene-search-location-readout");

const previewEmpty = document.getElementById("preview-empty");
const previewContent = document.getElementById("preview-content");
const previewImage = document.getElementById("preview-image");
const previewMeta = document.getElementById("preview-meta");

const startGenerationButton = document.getElementById("start-generation-button");

let sceneSelected = false;

// Populated on selection (search pick or direct upload), but not shown
// until the box 2 (Preview) step of the START GENERATION sequence reveals
// it — selecting a scene only enables the button, boxes 2-8 stay idle
// until it's clicked.
let pendingScenePreview = null;

// What box 1 actually did, for Calculation Logs (box 7) to open with —
// it starts from this project's own process rather than jumping straight
// into the generic reconstruction trace.
let sceneSelectionSummary = null;

function markSceneSelected() {
    sceneSelected = true;
    if (startGenerationButton) {
        startGenerationButton.disabled = false;
    }
}

function openSceneSearchModal() {
    if (sceneSearchModal) {
        sceneSearchModal.hidden = false;
    }
    if (sceneSearchStatus) {
        sceneSearchStatus.hidden = true;
    }
    if (sceneSearchResults) {
        sceneSearchResults.hidden = true;
        sceneSearchResults.innerHTML = "";
    }
    if (sceneSearchButton) {
        sceneSearchButton.disabled = false;
        sceneSearchButton.textContent = "SEARCH";
    }
    if (sceneSearchLocationReadout && selectedGeocodeResult) {
        const coords = document.createElement("span");
        coords.className = "numeric-mono";
        coords.textContent =
            `${Number(selectedGeocodeResult.lat).toFixed(4)}, ${Number(selectedGeocodeResult.lon).toFixed(4)}`;

        sceneSearchLocationReadout.replaceChildren(
            `📍 ${selectedGeocodeResult.display_name} · `,
            coords
        );
    }
}

function closeSceneSearchModal() {
    if (sceneSearchModal) {
        sceneSearchModal.hidden = true;
    }
}

function selectScene(scene) {
    pendingScenePreview = {
        src: scene.thumb,
        meta: `${scene.id} · ${scene.date} · ${scene.cloud}% cloud · prototype pick, not a live STAC query`,
    };
    sceneSelectionSummary = { type: "search", geocode: selectedGeocodeResult, scene };

    markSceneSelected();
    closeSceneSearchModal();
}

function selectUploadedScene(file) {
    pendingScenePreview = {
        src: URL.createObjectURL(file),
        meta: `${file.name} · uploaded directly · prototype pick, not a live reconstruction`,
    };
    sceneSelectionSummary = { type: "upload", file };

    markSceneSelected();
}

function renderSceneResults(scenes) {
    if (!sceneSearchResults) {
        return;
    }

    sceneSearchResults.innerHTML = "";

    scenes.forEach(scene => {
        const card = document.createElement("button");
        card.className = "scene-result-card";
        card.innerHTML = `
            <img class="scene-result-thumb" src="${scene.thumb}" alt="${scene.id} thumbnail" />
            <div>
                <div class="scene-result-name">${scene.id}</div>
                <div class="scene-result-sub">${scene.date} · ${scene.cloud}% cloud · 10m</div>
            </div>
        `;
        card.addEventListener("click", () => selectScene(scene));
        sceneSearchResults.appendChild(card);
    });

    sceneSearchResults.hidden = false;
}

async function runSceneSearch() {
    if (sceneSearchButton) {
        sceneSearchButton.disabled = true;
        sceneSearchButton.textContent = "SEARCHING…";
    }
    if (sceneSearchResults) {
        sceneSearchResults.hidden = true;
    }
    if (sceneSearchStatus) {
        sceneSearchStatus.hidden = false;

        const center = selectedGeocodeResult
            ? `${Number(selectedGeocodeResult.lat).toFixed(4)}, ${Number(selectedGeocodeResult.lon).toFixed(4)}`
            : "the selected location";

        sceneSearchStatus.innerHTML =
            `<span class="scene-search-spinner"></span> Querying Copernicus STAC near ${center} (mock)…`;
    }

    await sleep(1100);

    if (sceneSearchStatus) {
        sceneSearchStatus.hidden = true;
    }
    if (sceneSearchButton) {
        sceneSearchButton.disabled = false;
        sceneSearchButton.textContent = "SEARCH";
    }

    renderSceneResults(MOCK_SCENES);
}

sceneSearchTrigger?.addEventListener("click", openSceneSearchModal);
sceneSearchCancel?.addEventListener("click", closeSceneSearchModal);
sceneSearchBackdrop?.addEventListener("click", closeSceneSearchModal);
sceneSearchButton?.addEventListener("click", runSceneSearch);

document.addEventListener("keydown", event => {
    if (event.key === "Escape" && sceneSearchModal && !sceneSearchModal.hidden) {
        closeSceneSearchModal();
    }
});

startGenerationButton?.addEventListener("click", runGenerationSequence);


// ============================================================
// PAGE NAVIGATION (collapsible left icon rail)
// ============================================================

const pageNav = document.getElementById("page-nav");
const pageNavToggle = document.getElementById("page-nav-toggle");
const pageNavTabs = document.querySelectorAll(".page-nav-tab");
const pages = document.querySelectorAll(".page");

let activePageId = "page-1";

function setActivePage(pageId) {
    activePageId = pageId;

    pages.forEach(page => {
        page.classList.toggle("active", page.id === pageId);
    });

    pageNavTabs.forEach(tab => {
        tab.classList.toggle("active", tab.dataset.page === pageId);
    });

    if (pageId === "page-2") {
        requestAnimationFrame(initWorkbenchGrid);
    }
}

pageNavTabs.forEach(tab => {
    tab.addEventListener("click", () => {
        setActivePage(tab.dataset.page);
    });
});

pageNavToggle?.addEventListener("click", () => {
    pageNav?.classList.toggle("expanded");
});

document.getElementById("build-your-own-card")?.addEventListener("click", () => {
    setActivePage("page-2");
});


// ============================================================
// WORKBENCH RECONSTRUCTION GRID (columns 2-4, static/placeholder)
// ============================================================

// ---- Staged caption reveal: boxes 3-6 (Relative Depth, Elevation,
// DSM, Metric Elevation) show a couple of lines that fade in one at a
// time rather than appearing all at once. ----

function revealStagedCaption(container) {
    if (!container || container.dataset.revealed === "true") {
        return;
    }
    container.dataset.revealed = "true";

    let lines = [];
    try {
        lines = JSON.parse(container.dataset.lines ?? "[]");
    } catch {
        lines = [];
    }

    lines.forEach((text, index) => {
        const line = document.createElement("div");
        line.className = "staged-line";
        line.textContent = text;
        container.appendChild(line);

        setTimeout(() => {
            requestAnimationFrame(() => line.classList.add("visible"));
        }, 350 + index * 450);
    });
}

// ---- Calculation log: a running technical trace pulled from real
// facts elsewhere in this project's code/data that aren't already used
// as a caption above (CRS reprojection, per-region mesh sizes, the
// vertical-exaggeration formula and its real per-region outputs, edge-
// erosion / spike-smoothing parameters). Timestamps are the actual
// elapsed time since the log started, not staged numbers. It opens with
// what box 1 actually did (the location lookup/scene pick or the direct
// upload), not the generic trace. ----

const CALC_LOG_LINES = [
    "Sentinel-2 RGB tiles loaded — 4 regions, 10 m, EPSG:32645",
    "Darjeeling elevation source: OpenTopography DSM (native CRS)",
    "Kolkata / Bardhaman / Sundarbans source: Copernicus GLO-30 (EPSG:4326)",
    "Reprojecting Copernicus GLO-30 DEM → EPSG:32645 to align with Sentinel-2 grid",
    "Mesh grid — Darjeeling 361×325, Kolkata 367×330",
    "Mesh grid — Bardhaman 365×328, Sundarbans 368×332",
    "verticalExaggeration = 0.02 × clamp(1 + 4·log10(0.2 / reliefRatio), 1, 10)",
    "Darjeeling relief ratio 0.192 → exaggeration 1.07x",
    "Kolkata relief ratio 0.0041 → exaggeration 7.74x",
    "Bardhaman relief ratio 0.0031 → exaggeration 8.25x",
    "Sundarbans relief ratio 0.0010 → exaggeration 10.00x (ceiling clamp)",
    "Edge erosion — Kolkata: seed 1337, noiseScale 5, amount 0.06",
    "Edge erosion — Sundarbans: seed 2701, noiseScale 7, amount 0.08",
    "Edge erosion — Bardhaman: seed 8161, noiseScale 4, amount 0.06",
    "Spike smoothing — Kolkata / Bardhaman: median k=3, slope cap 0.6, 4 iterations",
    "Spike smoothing — Sundarbans: median k=3, slope cap 0.5, 4 iterations",
    "Darjeeling mesh uses real DSM coverage — no edge erosion / spike smoothing applied",
    "RDAH-Net (Swiss/HK building-height checkpoint), zero-shot on Sentinel-2 terrain: rejected — checkerboard artifacts",
    "── end of trace ──",
];

let calcLogStartTime = 0;

function appendCalcLogLine(scrollEl, text) {
    const elapsedSeconds = (Date.now() - calcLogStartTime) / 1000;
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = (elapsedSeconds % 60).toFixed(2).padStart(5, "0");
    const timestamp = `${String(minutes).padStart(2, "0")}:${seconds}`;

    const line = document.createElement("div");
    line.className = "calc-log-line";

    const timestampSpan = document.createElement("span");
    timestampSpan.className = "calc-log-timestamp";
    timestampSpan.textContent = `[${timestamp}]`;

    line.appendChild(timestampSpan);
    line.appendChild(document.createTextNode(text));
    scrollEl.appendChild(line);
    scrollEl.scrollTop = scrollEl.scrollHeight;
}

function buildCalcLogLines() {
    const opening = [];

    if (sceneSelectionSummary?.type === "search") {
        const geocode = sceneSelectionSummary.geocode;
        const scene = sceneSelectionSummary.scene;

        if (geocode) {
            opening.push(
                `Location resolved via Nominatim: ${geocode.display_name}`,
                `Coordinates: ${Number(geocode.lat).toFixed(4)}, ${Number(geocode.lon).toFixed(4)}`
            );
        }
        if (scene) {
            opening.push(`Selected scene ${scene.id} · ${scene.date} · ${scene.cloud}% cloud`);
        }
    } else if (sceneSelectionSummary?.type === "upload") {
        const file = sceneSelectionSummary.file;
        opening.push(
            `Direct upload received: ${file.name} (${Math.max(1, Math.round(file.size / 1024))} KB)`,
            `No STAC lookup for a direct upload — prototype pipeline defaults to Darjeeling reference data`
        );
    }

    opening.push("Handing off to reconstruction pipeline…");

    return [...opening, ...CALC_LOG_LINES];
}

// totalDurationMs paces the log to span roughly the same window as the
// box-by-box sequence driving it, so it reads as running alongside that
// work rather than being dumped out ahead of or behind it.
function runCalcLog(totalDurationMs) {
    const scrollEl = document.getElementById("calc-log-scroll");

    if (!scrollEl || scrollEl.dataset.started === "true") {
        return;
    }
    scrollEl.dataset.started = "true";

    calcLogStartTime = Date.now();

    const lines = buildCalcLogLines();
    const interval = totalDurationMs / lines.length;

    lines.forEach((text, index) => {
        setTimeout(() => appendCalcLogLine(scrollEl, text), 200 + index * interval);
    });
}

// ---- Mini 3D previews: DSM / Metric Elevation. Gently auto-rotating,
// no OrbitControls attached (no drag/zoom), sharing one fetch of
// Darjeeling's terrain data + textures across both. ----

const miniPreviews = [];
let miniPreviewAssetsPromise = null;

function loadMiniPreviewAssets() {
    if (!miniPreviewAssetsPromise) {
        miniPreviewAssetsPromise = (async () => {
            const terrainResponse = await fetch("/data/darjeeling/terrain.json");
            const terrainData = await terrainResponse.json();

            const textureLoader = new THREE.TextureLoader();
            const satelliteTexture = await textureLoader.loadAsync("/data/darjeeling/satellite.png");
            const depthTexture = await textureLoader.loadAsync("/data/darjeeling/relative_depth.png");
            const elevationTexture = await textureLoader.loadAsync("/data/darjeeling/elevation.png");

            return { terrainData, satelliteTexture, depthTexture, elevationTexture };
        })();
    }

    return miniPreviewAssetsPromise;
}

function resizeMiniPreview(preview) {
    const width = preview.canvas.clientWidth;
    const height = preview.canvas.clientHeight;

    if (width === 0 || height === 0 || (width === preview.lastWidth && height === preview.lastHeight)) {
        return;
    }

    preview.lastWidth = width;
    preview.lastHeight = height;

    preview.camera.aspect = width / height;
    preview.camera.updateProjectionMatrix();
    preview.renderer.setSize(width, height, false);
}

function createMiniPreview(canvas, layer) {
    const miniScene = new THREE.Scene();

    // Same fixed 100-unit terrain height (and tuned camera framing) as
    // the main viewer — createTerrain() always builds at that scale
    // regardless of region aspect ratio.
    const miniCamera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
    miniCamera.position.set(0, 58, 143);
    miniCamera.lookAt(0, 12, 0);

    const miniRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    miniRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    miniScene.add(new THREE.HemisphereLight(0xddebd8, 0x172018, 2.0));

    const miniSun = new THREE.DirectionalLight(0xffffff, 3.0);
    miniSun.position.set(-40, 100, 50);
    miniScene.add(miniSun);

    const group = new THREE.Group();
    miniScene.add(group);

    const preview = {
        canvas,
        scene: miniScene,
        camera: miniCamera,
        renderer: miniRenderer,
        group,
        ready: false,
        lastWidth: 0,
        lastHeight: 0,
    };

    loadMiniPreviewAssets().then(({ terrainData, satelliteTexture, depthTexture, elevationTexture }) => {
        const terrain = createTerrain(group, "darjeeling", terrainData, satelliteTexture, depthTexture, elevationTexture);
        terrain.setLayer(layer);
        preview.ready = true;
        resizeMiniPreview(preview);
    });

    miniPreviews.push(preview);
}

function updateMiniPreviews() {
    miniPreviews.forEach(preview => {
        if (!preview.ready) {
            return;
        }

        resizeMiniPreview(preview);

        preview.group.rotation.y += 0.0025;
        preview.renderer.render(preview.scene, preview.camera);
    });
}

// ---- Final Demo (box 8): the same shared, fully-interactive viewer
// used on Explore (drag/pinch/scroll, idle auto-rotate), now with full
// parity with Explore's own controls too — region rail, stats, 6-view
// layer switching, RUN RECONSTRUCTION, FLYTHROUGH, DANGER ZONES. All
// lookups below are scoped to #final-demo-box: the markup reuses
// Explore's exact classes (.region-card, .layer-button, etc.) for
// visual parity, and Explore's own equivalents are scoped to #page-1
// (see activateLayer / the region-rail bindings above) so the two
// viewers' independent state never cross-wires. Deferred until box 8's
// turn in the START GENERATION sequence — initFinalDemoViewer() below
// only runs once, the first time that happens. ----

let finalDemoViewer = null;
let finalDemoCurrentTerrain = null;
let finalDemoCurrentRegionKey = null;
let finalDemoRunning = false;
let finalDemoFloodActive = false;
let finalDemoFlythrough = null;
let finalDemoInitialized = false;

// Workbench's Final Demo box stays locked to whatever scene the box 1-7
// sequence generated — no region switcher, unlike Explore's own viewer.
// Gated with this flag (rather than a forked copy of initFinalDemoViewer)
// since the two instances share the same createTerrainViewer component.
const FINAL_DEMO_REGION_SWITCHER_ENABLED = false;

function updateFinalDemoStats(regionKey, terrain, terrainData) {
    const region = REGIONS[regionKey];

    const elevationElement = document.getElementById("final-demo-elevation-value");
    if (elevationElement) {
        elevationElement.textContent = `${Math.round(terrain.elevationMin)}–${Math.round(terrain.elevationMax)} m`;
    }

    const sourceElement = document.getElementById("final-demo-terrain-source-value");
    if (sourceElement) {
        sourceElement.textContent = region.elevationSource;
    }

    const inferenceStat = document.getElementById("final-demo-stat-inference");
    if (inferenceStat) {
        inferenceStat.textContent = region.inferenceTime;
    }

    const resolutionStat = document.getElementById("final-demo-stat-resolution");
    if (resolutionStat) {
        resolutionStat.textContent = region.resolution;
    }

    const gridStat = document.getElementById("final-demo-stat-grid");
    if (gridStat) {
        gridStat.textContent = `${terrainData.width}×${terrainData.height}`;
    }

    const epsgStat = document.getElementById("final-demo-stat-epsg");
    if (epsgStat) {
        epsgStat.textContent = region.crsEpsg;
    }
}

function activateFinalDemoLayer(layer) {
    if (!finalDemoCurrentTerrain) {
        return;
    }

    finalDemoCurrentTerrain.setLayer(layer);

    document.querySelectorAll("#final-demo-box .layer-button").forEach(button => {
        button.classList.toggle("active", button.dataset.layer === layer);
    });

    const description = document.getElementById("final-demo-layer-description");
    if (description) {
        description.textContent = layerDescriptionText(layer, finalDemoCurrentRegionKey);
    }
}

function setFinalDemoFloodActive(active) {
    finalDemoFloodActive = active;

    document.getElementById("final-demo-flood-button")?.classList.toggle("active", finalDemoFloodActive);
    finalDemoCurrentTerrain?.setFloodOverlay(finalDemoFloodActive);
}

function selectFinalDemoLayer(layer) {
    if (finalDemoFloodActive) {
        setFinalDemoFloodActive(false);
    }
    activateFinalDemoLayer(layer);
}

function resetFinalDemoRunState() {
    finalDemoRunning = false;

    const runPanelEl = document.getElementById("final-demo-run-panel");
    const postRunPanelEl = document.getElementById("final-demo-post-run-panel");
    const runButtonEl = document.getElementById("final-demo-run-button");
    const progressBarEl = document.getElementById("final-demo-pipeline-progress");
    const progressFillEl = document.getElementById("final-demo-pipeline-progress-fill");

    if (runPanelEl) runPanelEl.hidden = false;
    if (postRunPanelEl) postRunPanelEl.hidden = true;
    if (runButtonEl) {
        runButtonEl.disabled = false;
        runButtonEl.textContent = "▶ RUN RECONSTRUCTION";
    }
    if (progressBarEl) progressBarEl.hidden = true;
    if (progressFillEl) progressFillEl.style.width = "0%";

    setFinalDemoFloodActive(false);
    finalDemoFlythrough?.reset();
}

async function loadFinalDemoRegion(regionKey) {
    const { terrain, terrainData } = await finalDemoViewer.loadRegion(regionKey);

    finalDemoCurrentTerrain = terrain;
    finalDemoCurrentRegionKey = regionKey;

    updateFinalDemoStats(regionKey, terrain, terrainData);

    document.querySelectorAll("#final-demo-box .region-card").forEach(card => {
        card.classList.toggle("active", card.dataset.region === regionKey);
    });

    activateFinalDemoLayer("satellite-3d");
    resetFinalDemoRunState();

    return terrain;
}

// Mirrors Explore's runReconstruction() exactly (same STAGES/STAGE_TOTAL_MS),
// just targeting box 8's own elements.
async function runFinalDemoReconstruction() {
    if (finalDemoRunning) {
        return;
    }
    finalDemoRunning = true;

    setFinalDemoFloodActive(false);
    finalDemoFlythrough?.reset();

    const runPanelEl = document.getElementById("final-demo-run-panel");
    const postRunPanelEl = document.getElementById("final-demo-post-run-panel");
    const runButtonEl = document.getElementById("final-demo-run-button");
    const progressBarEl = document.getElementById("final-demo-pipeline-progress");
    const progressFillEl = document.getElementById("final-demo-pipeline-progress-fill");
    const description = document.getElementById("final-demo-layer-description");

    if (postRunPanelEl) postRunPanelEl.hidden = true;
    if (runPanelEl) runPanelEl.hidden = false;
    if (runButtonEl) {
        runButtonEl.disabled = true;
        runButtonEl.textContent = "PROCESSING…";
    }
    if (progressBarEl) progressBarEl.hidden = false;

    let elapsed = 0;

    for (const stage of STAGES) {
        activateFinalDemoLayer(stage.layer);

        if (description) {
            description.textContent = stage.caption;
        }

        await sleep(stage.duration);

        elapsed += stage.duration;

        if (progressFillEl) {
            progressFillEl.style.width = `${Math.round((elapsed / STAGE_TOTAL_MS) * 100)}%`;
        }
    }

    if (description) {
        description.textContent = "Reconstruction complete";
    }

    setTimeout(() => {
        if (progressBarEl) progressBarEl.hidden = true;
        if (progressFillEl) progressFillEl.style.width = "0%";

        finalDemoRunning = false;

        if (runPanelEl) runPanelEl.hidden = true;
        if (postRunPanelEl) postRunPanelEl.hidden = false;
    }, 800);
}

// ============================================================
// WORKBENCH THEME TOGGLE (light/dark — Workbench only, Explore and the
// shared page-nav rail are untouched since they never receive a
// [data-theme] attribute). Every color in styles.css reads from the
// custom properties #page-2[data-theme="light"] overrides, so flipping
// this one attribute repaints the whole page; the only piece CSS can't
// reach is the Final Demo viewer's THREE.js scene background (UI
// chrome, not imagery), updated here via viewer.setBackground().
// ============================================================

const WORKBENCH_THEME_BG_HEX = {
    dark: 0x110f0e,
    light: 0xf4e9d2,
};

let workbenchTheme = "dark";

function applyWorkbenchTheme(theme) {
    workbenchTheme = theme;

    const page2 = document.getElementById("page-2");
    if (page2) {
        page2.dataset.theme = theme;
    }

    const toggle = document.getElementById("workbench-theme-toggle");
    if (toggle) {
        const isLight = theme === "light";
        toggle.setAttribute("aria-checked", String(isLight));
        toggle.setAttribute("aria-label", `Switch Workbench to ${isLight ? "dark" : "light"} mode`);
        toggle.title = `Switch to ${isLight ? "dark" : "light"} mode`;
    }

    finalDemoViewer?.setBackground(WORKBENCH_THEME_BG_HEX[theme]);
}

document.getElementById("workbench-theme-toggle")?.addEventListener("click", () => {
    applyWorkbenchTheme(workbenchTheme === "dark" ? "light" : "dark");
});

// Creates the viewer, wires every control, and loads Darjeeling. Returns
// the loadFinalDemoRegion() promise so callers can await real asset load
// alongside the box's minimum "generating" duration.
function initFinalDemoViewer() {
    if (finalDemoInitialized) {
        return Promise.resolve();
    }
    finalDemoInitialized = true;

    const canvas = document.getElementById("final-demo-canvas");

    if (!canvas) {
        return Promise.resolve();
    }

    finalDemoViewer = createTerrainViewer(canvas, {
        rigOffsetY: 0,
        backgroundColor: WORKBENCH_THEME_BG_HEX[workbenchTheme],
    });

    finalDemoFlythrough = createFlythroughController({
        camera: finalDemoViewer.camera,
        controls: finalDemoViewer.controls,
        terrainRig: finalDemoViewer.terrainRig,
        button: document.getElementById("final-demo-flythrough-button"),
    });

    document.querySelectorAll("#final-demo-box .layer-button").forEach(button => {
        button.addEventListener("click", () => { selectFinalDemoLayer(button.dataset.layer); });
    });

    if (FINAL_DEMO_REGION_SWITCHER_ENABLED) {
        // [data-region] excludes nothing here (no "Build Your Own" card in
        // this copy), but kept for parity with Explore's own selector.
        document.querySelectorAll("#final-demo-box .region-card[data-region]").forEach(card => {
            card.addEventListener("click", async () => {
                const region = card.dataset.region;

                if (finalDemoRunning || region === finalDemoCurrentRegionKey) {
                    return;
                }

                await loadFinalDemoRegion(region);
            });
        });
    } else {
        document.querySelector("#final-demo-box .region-panel")?.remove();
    }

    document.getElementById("final-demo-run-button")?.addEventListener("click", runFinalDemoReconstruction);
    document.getElementById("final-demo-run-again-button")?.addEventListener("click", runFinalDemoReconstruction);

    document.getElementById("final-demo-flood-button")?.addEventListener("click", () => {
        if (!finalDemoFloodActive) {
            activateFinalDemoLayer("dsm-3d");
        }
        setFinalDemoFloodActive(!finalDemoFloodActive);
    });

    const fullscreenToggle = document.getElementById("final-demo-fullscreen");
    const finalDemoBox = document.getElementById("final-demo-box");

    fullscreenToggle?.addEventListener("click", () => {
        if (document.fullscreenElement) {
            document.exitFullscreen().catch(() => {});
        } else {
            finalDemoBox?.requestFullscreen().catch(() => {});
        }
    });

    document.addEventListener("fullscreenchange", () => {
        const isFullscreen = document.fullscreenElement === finalDemoBox;

        if (fullscreenToggle) {
            fullscreenToggle.textContent = isFullscreen ? "⤡" : "⛶";
            fullscreenToggle.title = isFullscreen ? "Exit fullscreen" : "Expand to fullscreen";
        }

        // The box's own size changes (viewport-filling vs. its grid
        // cell), not just the window — resizeToCanvas() picks that up
        // on the next animate() tick regardless, this just avoids a
        // one-frame stretch while the fullscreen transition settles.
        finalDemoViewer?.resizeToCanvas();
    });

    return loadFinalDemoRegion("darjeeling");
}

function updateFinalDemoViewer() {
    if (!finalDemoViewer) {
        return;
    }

    finalDemoViewer.resizeToCanvas();
    finalDemoFlythrough?.update();
    finalDemoViewer.update();
    finalDemoViewer.render();
}

// ============================================================
// START GENERATION SEQUENCE
// Drives boxes 2-8 one at a time (grid/box-number order) after the
// button in box 1 is clicked — nothing in them renders before that.
// Each box gets a real-feeling "generating" animation (a live percent
// counter and a status line stepping through what's actually happening,
// grounded in this project's real facts) for a minimum of ~4.5s before
// settling into its final state. Box 7 (Calculation Logs) isn't part of
// this one-at-a-time queue — it starts filling immediately alongside it
// and keeps appending lines, paced to span the same total window.
// ============================================================

const BOX_GENERATE_MS = 4500;

// Sequential box count (excludes box 7, which runs continuously in
// parallel instead) — used to pace Calculation Logs to the same span.
const SEQUENTIAL_BOX_COUNT = 6;
const TOTAL_PIPELINE_MS = BOX_GENERATE_MS * SEQUENTIAL_BOX_COUNT;

function generatingOverlayMarkup() {
    return `
        <div class="generating-overlay">
            <div class="generating-percent">0%</div>
            <div class="generating-status"></div>
            <div class="generating-scanbar"></div>
        </div>
    `;
}

// Ticks a percent counter and steps a status line through `steps`,
// evenly spaced across durationMs — the actual visual of "something is
// being calculated." Resolves once durationMs has elapsed.
async function animateGenerating({ percentEl, statusEl, steps, durationMs }) {
    const tickMs = 120;
    const stepIntervalMs = durationMs / Math.max(1, steps.length);

    // Tracks real elapsed time (performance.now()) rather than assuming
    // each tick took exactly tickMs — a backgrounded/throttled tab can
    // delay individual setTimeout ticks well past 120ms, and accumulating
    // a fixed increment per tick would make the whole animation run far
    // longer in wall-clock time than durationMs in that case.
    const startTime = performance.now();
    let elapsed = 0;
    let stepIndex = -1;

    function setStep(index) {
        if (index === stepIndex) {
            return;
        }
        stepIndex = index;
        if (statusEl && steps[index] !== undefined) {
            statusEl.textContent = steps[index];
        }
    }

    setStep(0);

    while (elapsed < durationMs) {
        await sleep(tickMs);
        elapsed = performance.now() - startTime;

        if (percentEl) {
            percentEl.textContent = `${Math.min(100, Math.round((elapsed / durationMs) * 100))}%`;
        }

        setStep(Math.min(steps.length - 1, Math.floor(elapsed / stepIntervalMs)));
    }
}

// ---- Box 2 (Preview): reveals whatever was selected/uploaded in box 1,
// which was held back until now. ----

async function generatePreviewBox() {
    const emptyEl = document.getElementById("preview-empty");
    const contentEl = document.getElementById("preview-content");

    if (!emptyEl || !contentEl || !pendingScenePreview) {
        return;
    }

    emptyEl.innerHTML = generatingOverlayMarkup();

    await animateGenerating({
        percentEl: emptyEl.querySelector(".generating-percent"),
        statusEl: emptyEl.querySelector(".generating-status"),
        steps: [
            "Registering scene footprint…",
            "Validating against Sentinel-2 tiling grid…",
            "Preparing reconstruction inputs…",
        ],
        durationMs: BOX_GENERATE_MS,
    });

    if (previewImage) {
        previewImage.src = pendingScenePreview.src;
    }
    if (previewMeta) {
        previewMeta.textContent = pendingScenePreview.meta;
    }

    emptyEl.hidden = true;
    contentEl.hidden = false;
}

// ---- Boxes 3 (Relative Depth) / 4 (Elevation): the image + caption
// were already in the DOM, just held behind .preview-empty — swap that
// for a real-feeling generating readout, then reveal both. ----

async function generateCaptionPreviewBox(boxId, steps) {
    const box = document.getElementById(boxId);

    if (!box) {
        return;
    }

    const emptyEl = box.querySelector(".preview-empty");
    const contentEl = box.querySelector(".preview-content");
    const captionEl = box.querySelector(".staged-caption");

    if (emptyEl) {
        emptyEl.innerHTML = generatingOverlayMarkup();
    }

    await animateGenerating({
        percentEl: emptyEl?.querySelector(".generating-percent"),
        statusEl: emptyEl?.querySelector(".generating-status"),
        steps,
        durationMs: BOX_GENERATE_MS,
    });

    if (emptyEl) {
        emptyEl.hidden = true;
    }
    if (contentEl) {
        contentEl.hidden = false;
    }
    if (captionEl) {
        revealStagedCaption(captionEl);
    }
}

// ---- Boxes 5 (DSM) / 6 (Metric Elevation): the generating overlay runs
// for at least BOX_GENERATE_MS *and* until the real (shared, memoized)
// terrain asset fetch actually resolves, whichever is longer. ----

async function generateMiniPreviewBox(boxId, canvasId, layer, steps) {
    const box = document.getElementById(boxId);
    const canvas = document.getElementById(canvasId);

    if (!box || !canvas) {
        return;
    }

    const overlay = box.querySelector(".mini3d-generating");
    const captionEl = box.querySelector(".staged-caption");

    if (overlay) {
        overlay.hidden = false;
    }

    createMiniPreview(canvas, layer);

    await Promise.all([
        animateGenerating({
            percentEl: overlay?.querySelector(".generating-percent"),
            statusEl: overlay?.querySelector(".generating-status"),
            steps,
            durationMs: BOX_GENERATE_MS,
        }),
        loadMiniPreviewAssets(),
    ]);

    if (overlay) {
        overlay.hidden = true;
    }
    if (captionEl) {
        revealStagedCaption(captionEl);
    }
}

// ---- Box 8 (Final Demo): builds the full interactive viewer + controls
// (only once — initFinalDemoViewer() is itself idempotent) alongside the
// generating animation, then reveals both. ----

async function generateFinalDemoBox() {
    const overlay = document.getElementById("final-demo-generating");
    const controlsEl = document.getElementById("final-demo-controls");
    const fullscreenToggle = document.getElementById("final-demo-fullscreen");

    if (overlay) {
        overlay.hidden = false;
    }

    await Promise.all([
        animateGenerating({
            percentEl: overlay?.querySelector(".generating-percent"),
            statusEl: overlay?.querySelector(".generating-status"),
            steps: [
                "Compiling interactive viewer…",
                "Wiring OrbitControls + layer shaders…",
                "Loading RUN RECONSTRUCTION, FLYTHROUGH, DANGER ZONES…",
            ],
            durationMs: BOX_GENERATE_MS,
        }),
        initFinalDemoViewer(),
    ]);

    if (overlay) {
        overlay.hidden = true;
    }
    if (controlsEl) {
        controlsEl.hidden = false;
    }
    if (fullscreenToggle) {
        fullscreenToggle.hidden = false;
    }
}

let generationStarted = false;

async function runGenerationSequence() {
    if (generationStarted) {
        return;
    }
    generationStarted = true;

    if (startGenerationButton) {
        startGenerationButton.disabled = true;
        startGenerationButton.textContent = "▶ GENERATING…";
    }

    // Starts filling immediately and keeps appending in parallel with
    // whichever box below is currently generating.
    runCalcLog(TOTAL_PIPELINE_MS);

    await generatePreviewBox();

    await generateCaptionPreviewBox("depth-preview-box", [
        "Loading Sentinel-2 RGB tiles…",
        "Running Depth Anything V2 (ViT-Large)…",
        "Inference complete — 1.17s, frozen weights…",
    ]);

    await generateCaptionPreviewBox("elevation-preview-box", [
        "Evaluating terrain DEM + correction head…",
        "Checking sparse ICESat-2 anchors…",
        "Correction head not yet trained — flat DEM elevation shown…",
    ]);

    await generateMiniPreviewBox("dsm-3d-box", "dsm-3d-canvas", "dsm-3d", [
        "Reprojecting DSM → EPSG:32645…",
        "Extruding 361×325 vertex grid…",
        "Applying vertical exaggeration 1.07x…",
    ]);

    await generateMiniPreviewBox("metric-elevation-3d-box", "metric-elevation-3d-canvas", "elevation-3d", [
        "Running spatial-trend validation…",
        "Detrending elevation vs. position…",
        "Correlation +0.60 → −0.41 after detrending…",
    ]);

    await generateFinalDemoBox();

    if (startGenerationButton) {
        startGenerationButton.textContent = "✓ GENERATION COMPLETE";
    }
}


// ---- Staged grid entrance: the 8 boxes fade/slide in left-to-right,
// top-to-bottom, one subtle cascade rather than 8 independent panels. ----

const GRID_ENTER_ORDER = [
    "scene-input-box",
    "depth-preview-box",
    "dsm-3d-box",
    "calc-logs-box",
    "preview-box",
    "elevation-preview-box",
    "metric-elevation-3d-box",
    "final-demo-box",
];

function staggerGridEntrance() {
    GRID_ENTER_ORDER.forEach((id, index) => {
        const el = document.getElementById(id);

        if (!el) {
            return;
        }

        setTimeout(() => el.classList.add("grid-enter-visible"), index * 80);
    });
}

// ---- Init (first Workbench visit only — avoids re-creating WebGL
// contexts or re-running the staged reveals on every tab switch). ----

let workbenchGridInitialized = false;

function initWorkbenchGrid() {
    if (workbenchGridInitialized) {
        return;
    }
    workbenchGridInitialized = true;

    // Only the box chrome fades in on arrival — boxes 2-8's actual content
    // stays idle until the START GENERATION sequence above drives it.
    staggerGridEntrance();
}


// ============================================================
// RENDER LOOP
// ============================================================

function animate() {
    requestAnimationFrame(animate);

    if (activePageId === "page-2") {
        updateMiniPreviews();
        updateFinalDemoViewer();
        return;
    }

    if (activePageId !== "page-1") {
        return;
    }

    updateFlythrough();
    terrainRig.update();
    controls.update();
    renderer.render(scene, camera);
}


// ============================================================
// RESIZE
// ============================================================

window.addEventListener("resize", () => {
    exploreViewer.resize(window.innerWidth, window.innerHeight);
});


// ============================================================
// START
// ============================================================

loadRegion("darjeeling")
    .then(() => {
        console.log("DepthWizard2 viewer ready.");
        animate();
    })
    .catch(error => {
        console.error("DepthWizard2 startup error:", error);
    });