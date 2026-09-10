import * as THREE from "three";
import { createTerrain } from "./terrain.js";
import { createControls } from "./controls.js";
import { createTerrainRig } from "./terrainRig.js";

const canvas = document.getElementById("terrain-canvas");

const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
});

renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07100d);

const camera = new THREE.PerspectiveCamera(
    55,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);

// No base yaw correction — the earlier "tilt it a few degrees" tweak was
// actually compensating for the camera's steep default angle, not a mesh
// orientation problem (see the lowered camera elevation below).
const BASE_YAW = 0;

// terrainRig's wheel listener must attach to domElement before
// OrbitControls' own (created next) so it can claim plain two-finger
// scroll for rotation ahead of OrbitControls treating it as a zoom.
const terrainRig = createTerrainRig(renderer.domElement, BASE_YAW);

// Vertical-only screen offset so the structure clears the panels occupying
// the top of the page. No horizontal offset — kept centered on X so it
// lines up with controls.target's X, which is also the flythrough's zoom
// pivot; an X mismatch between the two is what drifted the structure
// sideways as the flythrough zoomed in.
terrainRig.rig.position.set(0, -10, 0);
scene.add(terrainRig.rig);

const controls = createControls(
    camera,
    renderer.domElement,
    new THREE.Vector3(0, 12, 0)
);

const ambient = new THREE.HemisphereLight(
    0xddebd8,
    0x172018,
    2.0
);
scene.add(ambient);

const sun = new THREE.DirectionalLight(
    0xffffff,
    3.0
);
sun.position.set(-40, 100, 50);
sun.castShadow = true;
scene.add(sun);


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

function disposeTerrain(terrain) {
    if (!terrain) {
        return;
    }

    terrainRig.rig.remove(terrain.mesh);
    terrain.mesh.geometry.dispose();
    terrain.material.dispose();
}

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
    console.log(`Loading ${regionKey} terrain...`);

    const terrainResponse = await fetch(`/data/${regionKey}/terrain.json`);

    if (!terrainResponse.ok) {
        throw new Error(`Failed to load terrain: ${terrainResponse.status}`);
    }

    const terrainData = await terrainResponse.json();


    // --------------------------------------------------------
    // Load visual layers
    // --------------------------------------------------------

    const textureLoader = new THREE.TextureLoader();

    const satelliteTexture = await textureLoader.loadAsync(`/data/${regionKey}/satellite.png`);
    const depthTexture = await textureLoader.loadAsync(`/data/${regionKey}/relative_depth.png`);
    const elevationTexture = await textureLoader.loadAsync(`/data/${regionKey}/elevation.png`);


    // --------------------------------------------------------
    // Swap in the new terrain
    // --------------------------------------------------------

    disposeTerrain(currentTerrain);

    const terrain = createTerrain(
        terrainRig.rig,
        regionKey,
        terrainData,
        satelliteTexture,
        depthTexture,
        elevationTexture
    );

    currentTerrain = terrain;
    currentRegionKey = regionKey;

    console.log("Terrain created:", terrain);


    // --------------------------------------------------------
    // Camera
    // --------------------------------------------------------

    // ~18 degrees above the horizon (was ~34 degrees / a steep bird's-eye
    // angle) — same viewing distance, just lower and more oblique.
    camera.position.set(0, 58, 143);
    controls.target.set(0, 12, 0);
    controls.update();
    terrainRig.reset();


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

document.querySelectorAll(".region-card").forEach(card => {
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
// MOCK UPLOAD FLOW
// ============================================================

const uploadTrigger = document.getElementById("upload-trigger");
const uploadModal = document.getElementById("upload-modal");
const uploadBackdrop = document.getElementById("upload-backdrop");
const uploadCancel = document.getElementById("upload-cancel");
const uploadDropzone = document.getElementById("upload-dropzone");
const uploadInput = document.getElementById("upload-input");

function openUploadModal() {
    if (uploadModal) {
        uploadModal.hidden = false;
    }
}

function closeUploadModal() {
    if (uploadModal) {
        uploadModal.hidden = true;
    }
    uploadDropzone?.classList.remove("drag-over");
}

async function handleUploadedFile(file) {
    if (!file) {
        return;
    }

    closeUploadModal();

    await runReconstruction();
}

uploadTrigger?.addEventListener("click", openUploadModal);
uploadCancel?.addEventListener("click", closeUploadModal);
uploadBackdrop?.addEventListener("click", closeUploadModal);

uploadInput?.addEventListener("change", () => {
    handleUploadedFile(uploadInput.files?.[0]);
});

uploadDropzone?.addEventListener("dragover", event => {
    event.preventDefault();
    uploadDropzone.classList.add("drag-over");
});

uploadDropzone?.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("drag-over");
});

uploadDropzone?.addEventListener("drop", event => {
    event.preventDefault();
    uploadDropzone.classList.remove("drag-over");
    handleUploadedFile(event.dataTransfer?.files?.[0]);
});

document.addEventListener("keydown", event => {
    if (event.key === "Escape" && uploadModal && !uploadModal.hidden) {
        closeUploadModal();
    }
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
// RENDER LOOP
// ============================================================

function animate() {
    requestAnimationFrame(animate);

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
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
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