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

function layerDescriptionText(layer) {
    const source = REGIONS[currentRegionKey]?.elevationSource ?? "DSM";

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

    document.querySelectorAll(".layer-button").forEach(button => {
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

    document.querySelectorAll(".region-card").forEach(card => {
        card.classList.toggle("active", card.dataset.region === regionKey);
    });

    activateLayer("satellite-3d");
    resetRunState();

    return terrain;
}


// ============================================================
// LAYER BUTTONS
// ============================================================

document.querySelectorAll(".layer-button").forEach(button => {
    button.addEventListener("click", () => { selectLayer(button.dataset.layer); });
});


// ============================================================
// REGION RAIL — switching
// ============================================================

// [data-region] excludes the "Build Your Own" card — it reuses
// .region-card for consistent styling but isn't a region to load.
document.querySelectorAll(".region-card[data-region]").forEach(card => {
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
// ============================================================

let flythrough = null;

const FLYTHROUGH_ROTATE_SPEED_MULTIPLIER = 2;

function resetFlythrough() {
    flythrough = null;
    terrainRig.setSpeedMultiplier(1);

    if (flythroughButton) {
        flythroughButton.disabled = false;
        flythroughButton.textContent = "◎ FLYTHROUGH";
    }
}

function startFlythrough() {
    if (flythrough || !flythroughButton || flythroughButton.disabled) {
        return;
    }

    const startDistance = camera.position.distanceTo(controls.target);
    const endDistance = Math.max(controls.minDistance, startDistance * 0.5);

    flythrough = {
        startDistance,
        endDistance,
        startTime: performance.now(),
        duration: 9500,
    };

    // 2x idle rotation speed for the zoom-in and for the continued
    // rotation afterward — stays elevated until the next reset.
    terrainRig.setSpeedMultiplier(FLYTHROUGH_ROTATE_SPEED_MULTIPLIER);

    flythroughButton.disabled = true;
    flythroughButton.textContent = "FLYING THROUGH…";
}

function updateFlythrough() {
    if (!flythrough) {
        return;
    }

    const t = Math.min(1, (performance.now() - flythrough.startTime) / flythrough.duration);
    const eased = 1 - Math.pow(1 - t, 3);

    const distance = flythrough.startDistance + (flythrough.endDistance - flythrough.startDistance) * eased;

    const offset = camera.position.clone().sub(controls.target);
    offset.setLength(distance);
    camera.position.copy(controls.target).add(offset);

    if (t >= 1) {
        flythrough = null;

        if (flythroughButton) {
            flythroughButton.textContent = "✓ FLYTHROUGH";
        }
    }
}

flythroughButton?.addEventListener("click", startFlythrough);


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
        <div class="geocode-detail-row"><span>Lat, Lng</span><span>${Number(result.lat).toFixed(4)}, ${Number(result.lon).toFixed(4)}</span></div>
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
        sceneSearchLocationReadout.textContent =
            `📍 ${selectedGeocodeResult.display_name} · ` +
            `${Number(selectedGeocodeResult.lat).toFixed(4)}, ${Number(selectedGeocodeResult.lon).toFixed(4)}`;
    }
}

function closeSceneSearchModal() {
    if (sceneSearchModal) {
        sceneSearchModal.hidden = true;
    }
}

function selectScene(scene) {
    if (previewImage) {
        previewImage.src = scene.thumb;
    }
    if (previewMeta) {
        previewMeta.textContent =
            `${scene.id} · ${scene.date} · ${scene.cloud}% cloud · ` +
            `prototype pick, not a live STAC query`;
    }
    if (previewEmpty) {
        previewEmpty.hidden = true;
    }
    if (previewContent) {
        previewContent.hidden = false;
    }

    markSceneSelected();
    closeSceneSearchModal();
}

function selectUploadedScene(file) {
    const objectUrl = URL.createObjectURL(file);

    if (previewImage) {
        previewImage.src = objectUrl;
    }
    if (previewMeta) {
        previewMeta.textContent =
            `${file.name} · uploaded directly · prototype pick, not a live reconstruction`;
    }
    if (previewEmpty) {
        previewEmpty.hidden = true;
    }
    if (previewContent) {
        previewContent.hidden = false;
    }

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

// Placeholder — enabled once a scene is selected via search or upload.
// Intentionally a no-op for now; wiring it to actually run the box 2-8
// reconstruction sequence is next-prompt scope.
startGenerationButton?.addEventListener("click", () => {
    console.log("START GENERATION clicked — pipeline wiring not yet implemented.");
});


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

function revealAllStagedCaptions() {
    document.querySelectorAll(".staged-caption").forEach(revealStagedCaption);
}

// ---- Calculation log: a running technical trace pulled from real
// facts elsewhere in this project's code/data that aren't already used
// as a caption above (CRS reprojection, per-region mesh sizes, the
// vertical-exaggeration formula and its real per-region outputs, edge-
// erosion / spike-smoothing parameters). Timestamps are the actual
// elapsed time since the log started, not staged numbers. ----

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

function runCalcLog() {
    const scrollEl = document.getElementById("calc-log-scroll");

    if (!scrollEl || scrollEl.dataset.started === "true") {
        return;
    }
    scrollEl.dataset.started = "true";

    calcLogStartTime = Date.now();

    CALC_LOG_LINES.forEach((text, index) => {
        setTimeout(() => appendCalcLogLine(scrollEl, text), 200 + index * 420);
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

// ---- Final Demo: the same shared, fully-interactive viewer used on
// Explore (drag/pinch/scroll, idle auto-rotate), defaulting to
// Darjeeling, with a fullscreen toggle. ----

let finalDemoViewer = null;

function initFinalDemoViewer() {
    const canvas = document.getElementById("final-demo-canvas");

    if (!canvas) {
        return;
    }

    finalDemoViewer = createTerrainViewer(canvas, { rigOffsetY: 0 });

    finalDemoViewer.loadRegion("darjeeling").then(() => {
        finalDemoViewer.setLayer("satellite-3d");
        finalDemoViewer.resizeToCanvas();
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
}

function updateFinalDemoViewer() {
    if (!finalDemoViewer) {
        return;
    }

    finalDemoViewer.resizeToCanvas();
    finalDemoViewer.update();
    finalDemoViewer.render();
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
    // (captions, calc log, mini 3D previews, Final Demo viewer) stays idle
    // until START GENERATION drives it. That wiring is next-prompt scope;
    // revealAllStagedCaptions() / runCalcLog() / initFinalDemoViewer() /
    // createMiniPreview() below are left defined for it to call.
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