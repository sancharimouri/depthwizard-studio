import * as THREE from "three";
import { createTerrain } from "./terrain.js";
import { createTerrainViewer } from "./viewer.js";
import { createMeasureTool } from "./measure-tool.js";
import { createExpandedChrome } from "./expanded-chrome.js";
import { createFloodSim } from "./flood-sim.js";
import { initCollapsibleBoxes, initFacts, initTour, renderFacts, renderSource, renderTerrainStats } from "./side-panels.js";
import { createInputView } from "./input-view.js";
import {
    STORAGE_NOTE, createJobStore, createSavedStore, exportFilename, jobLabel, jobsExport, savedRecord, unsavedCopy,
} from "./jobs.js";
import { installCloseGuard } from "./desktop-close.js";

const canvas = document.getElementById("terrain-canvas");

// Explore's main viewer. Vertical-only rig offset so the structure clears
// the panels occupying the top of the page — no horizontal offset, kept
// centered on X so it lines up with controls.target's X, which is also the
// flythrough's zoom pivot; an X mismatch between the two is what drifted
// the structure sideways as the flythrough zoomed in.
const exploreViewer = createTerrainViewer(canvas, { rigOffsetY: -10 });
exploreViewer.resize(window.innerWidth, window.innerHeight);

const { scene, camera, renderer, controls } = exploreViewer;


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
            return `DEM elevation from ${source} — flat`;
        case "dsm-3d":
            return "DAv2 relative depth draped on extruded terrain";
        case "elevation-3d":
            return `DEM elevation from ${source} — color-ramped by elevation`;
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

    // Scoped to #page-explore — Final Demo's box 8 reuses the same .layer-button
    // class for visual parity with Explore, and has its own independent
    // activateFinalDemoLayer() below; an unscoped selector here would
    // cross-wire the two viewers' active states.
    document.querySelectorAll("#page-explore .layer-button").forEach(button => {
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

    document.querySelectorAll("#page-explore .region-card").forEach(card => {
        card.classList.toggle("active", card.dataset.region === regionKey);
    });

    activateLayer("satellite-3d");
    resetRunState();

    return terrain;
}


// ============================================================
// LAYER BUTTONS
// ============================================================

document.querySelectorAll("#page-explore .layer-button").forEach(button => {
    button.addEventListener("click", () => { selectLayer(button.dataset.layer); });
});


// ============================================================
// REGION RAIL — switching
// ============================================================

// [data-region] excludes the "Build Your Own" card — it reuses
// .region-card for consistent styling but isn't a region to load.
document.querySelectorAll("#page-explore .region-card[data-region]").forEach(card => {
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
    { layer: "elevation-3d", caption: "DEM Elevation", duration: 1000 },
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
// (one-shot camera dolly-in — no loop back out — auto-rotate keeps
// spinning the camera around the terrain underneath it via controls' own
// render-loop update)
//
// Factored into a reusable controller so Workbench's Final Demo box
// (box 8) can offer the same feature on its own independent viewer and
// controls without sharing state with Explore's.
// ============================================================

const FLYTHROUGH_ROTATE_SPEED_MULTIPLIER = 2;
const FLYTHROUGH_DURATION_MS = 9500;

// The flythrough drives its OWN rotation + zoom, independent of base
// auto-rotate: it runs the same whether auto-rotate is on, user-paused
// (the ⏸ button), held (measuring) or idle-paused after a drag. While it
// runs, base auto-rotate is suspended (setExternallyDriven) so the two never
// add up; afterwards the camera returns to whatever base state was set.
// Grabbing the camera ends it.
function createFlythroughController({ controls: flControls, button }) {
    let flythrough = null;
    const rotateRadPerSec = flControls.AUTO_ROTATE_RADIANS_PER_SEC * FLYTHROUGH_ROTATE_SPEED_MULTIPLIER;

    function finish(label = "✓ FLYTHROUGH") {
        flythrough = null;
        flControls.setExternallyDriven(false);
        if (button) {
            button.textContent = label;
        }
    }

    function reset() {
        flythrough = null;
        flControls.setExternallyDriven(false);
        flControls.setSpeedMultiplier(1);

        if (button) {
            button.disabled = false;
            button.textContent = "◎ FLYTHROUGH";
        }
    }

    function start() {
        // The button is optional: the final demo's toolbar icon starts it too.
        if (flythrough || button?.disabled) {
            return;
        }

        const startDistance = flControls.distance;
        const endDistance = Math.max(flControls.minDistance, startDistance * 0.5);

        flythrough = {
            startDistance,
            endDistance,
            startTime: performance.now(),
            lastTime: performance.now(),
            duration: FLYTHROUGH_DURATION_MS,
        };
        flControls.setExternallyDriven(true);

        if (button) {
            button.disabled = true;
            button.textContent = "FLYING THROUGH…";
        }
    }

    function update() {
        if (!flythrough) {
            return;
        }

        const now = performance.now();
        const t = Math.min(1, (now - flythrough.startTime) / flythrough.duration);
        const dt = Math.min((now - flythrough.lastTime) / 1000, 0.1);
        flythrough.lastTime = now;
        const eased = 1 - Math.pow(1 - t, 3);

        const distance = flythrough.startDistance + (flythrough.endDistance - flythrough.startDistance) * eased;

        // Immediate (no transition): this loop already supplies its own
        // cubic ease, and letting controls' own damping smooth it too
        // would double up and lag behind the intended curve.
        flControls.dollyTo(distance, false);
        // Own rotation, same sign as base auto-rotate.
        flControls.azimuthAngle += rotateRadPerSec * dt;

        if (t >= 1) {
            finish();
        }
    }

    // User grabbed the camera: stop driving it.
    flControls.addEventListener("control", () => {
        if (flythrough) {
            finish(button ? "◎ FLYTHROUGH" : undefined);
            if (button) {
                button.disabled = false;
            }
        }
    });

    button?.addEventListener("click", start);

    return { start, update, reset, isRunning: () => Boolean(flythrough) };
}

const exploreFlythrough = createFlythroughController({
    controls,
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


const previewEmpty = document.getElementById("preview-empty");
const previewContent = document.getElementById("preview-content");
const previewImage = document.getElementById("preview-image");
const previewMeta = document.getElementById("preview-meta");

const startGenerationButton = document.getElementById("start-generation-button");

let sceneSelected = false;

// Populated on selection (search pick or direct upload). Box 2 (Preview)
// reveals it immediately — only boxes 3-8 stay idle until START GENERATION.
let pendingScenePreview = null;

// What box 1 actually did, for Calculation Logs (box 7) to open with —
// it starts from this project's own process rather than jumping straight
// into the generic reconstruction trace.
let sceneSelectionSummary = null;

function revealScenePreview() {
    if (!pendingScenePreview) {
        return;
    }
    if (previewImage) {
        previewImage.src = pendingScenePreview.src;
    }
    if (previewMeta) {
        previewMeta.textContent = pendingScenePreview.meta;
    }
    if (previewEmpty) {
        previewEmpty.hidden = true;
    }
    if (previewContent) {
        previewContent.hidden = false;
    }
}

function markSceneSelected() {
    sceneSelected = true;
    if (startGenerationButton) {
        startGenerationButton.disabled = false;
    }
    revealScenePreview();
}


// ============================================================
// PAGE 1 INPUT VIEW → GENERATION GRID, AS JOBS. The input view
// (src/input-view.js) owns selection; each START GENERATION creates a job
// (src/jobs.js — in-memory, session only) and runs the staged pipeline for
// it. "Generate New" reopens the input view without discarding anything;
// the jobs panel (and, in the expanded 3D view, a tab strip) switches
// between finished jobs.
// ============================================================

const inputViewEl = document.getElementById("input-view");
const workbenchGridEl = document.querySelector("#page-workbench .workbench-grid");
const workbenchPageEl = document.getElementById("page-workbench");
const jobsPanelEl = document.getElementById("jobs-panel");
const jobsListEl = document.getElementById("jobs-list");
const jobsCountEl = document.getElementById("jobs-count");
const generateNewButton = document.getElementById("generate-new-button");
const jobTabsEl = document.getElementById("job-tabs");
const jobTabInfoEl = document.getElementById("job-tab-info");
const jobsPinnedEl = document.getElementById("jobs-pinned");
const jobsPinnedEmptyEl = document.getElementById("jobs-pinned-empty");
const savedCountEl = document.getElementById("saved-count");
const jobsPanelToggle = document.getElementById("jobs-panel-toggle");

const jobStore = createJobStore();
let inputView = null;
let unloadAllowed = false;

const CAPTION_BOX_IDS = ["depth-preview-box", "elevation-preview-box"];
const MINI_BOX_IDS = ["dsm-3d-box", "metric-elevation-3d-box"];

function choosingNewInput() {
    return Boolean(inputViewEl) && !inputViewEl.hidden;
}

// Both re-render the jobs panel: its "choosing input" row and the Generate
// New button depend on which view is showing.
function showInputView() {
    inputViewEl.hidden = false;
    workbenchGridEl?.classList.add("is-hidden");
    renderJobs();
}

function showGrid() {
    inputViewEl.hidden = true;
    workbenchGridEl?.classList.remove("is-hidden");
    renderJobs();
}

function applyJobInput(job) {
    pendingScenePreview = { src: job.input.previewUrl, meta: `${jobLabel(job)} · ${job.input.metaLine}` };
    sceneSelectionSummary = { type: "input", ...job.input };
    markSceneSelected();
}

function resetCaption(captionEl) {
    if (captionEl) {
        captionEl.dataset.revealed = "false";
        captionEl.replaceChildren();
    }
}

// Pop the expanded view back into its cell with no animation (used when the
// grid is about to be hidden or reset underneath it).
function collapseFinalDemoInstantly() {
    if (!finalDemoBoxEl?.classList.contains("is-expanded")) {
        return;
    }
    finalDemoMeasureTool?.setMode("normal");
    finalDemoChrome?.closeAll();
    finalDemoBoxEl.getAnimations?.().forEach(animation => animation.cancel());
    finalDemoBoxEl.classList.remove("is-expanded", "is-animating");
    finalDemoPlaceholder?.remove();
    finalDemoPlaceholder = null;
    finalDemoAnimating = false;
    syncNavDefault();
    if (finalDemoExpandedBar) {
        finalDemoExpandedBar.hidden = true;
    }
    requestAnimationFrame(() => finalDemoViewer?.resizeToCanvas());
}

// Every box back to "Awaiting generation" so the next job's stages animate
// in one by one, exactly like the first run.
function resetGridForGeneration() {
    collapseFinalDemoInstantly();

    CAPTION_BOX_IDS.forEach(id => {
        const box = document.getElementById(id);
        const emptyEl = box?.querySelector(".preview-empty");
        if (emptyEl) {
            emptyEl.hidden = false;
            emptyEl.textContent = "Awaiting generation";
        }
        const contentEl = box?.querySelector(".preview-content");
        if (contentEl) {
            contentEl.hidden = true;
        }
        resetCaption(box?.querySelector(".staged-caption"));
    });

    MINI_BOX_IDS.forEach(id => {
        const box = document.getElementById(id);
        box?.classList.add("is-pending");
        const overlay = box?.querySelector(".mini3d-generating");
        if (overlay) {
            overlay.hidden = true;
        }
        resetCaption(box?.querySelector(".staged-caption"));
    });

    finalDemoBoxEl?.classList.add("is-pending");
    ["final-demo-generating", "final-demo-controls", "final-demo-fullscreen"].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.hidden = true;
        }
    });

    document.getElementById("calc-log-scroll")?.replaceChildren();
}

// A finished job, shown again: every box in its generated state, the
// job's own input in the preview box and its own calculation log.
function showCompletedJob(job) {
    applyJobInput(job);

    CAPTION_BOX_IDS.forEach(id => {
        const box = document.getElementById(id);
        const emptyEl = box?.querySelector(".preview-empty");
        if (emptyEl) {
            emptyEl.hidden = true;
        }
        const contentEl = box?.querySelector(".preview-content");
        if (contentEl) {
            contentEl.hidden = false;
        }
        const captionEl = box?.querySelector(".staged-caption");
        resetCaption(captionEl);
        revealStagedCaption(captionEl);
    });

    MINI_BOX_IDS.forEach(id => {
        const box = document.getElementById(id);
        box?.classList.remove("is-pending");
        const captionEl = box?.querySelector(".staged-caption");
        resetCaption(captionEl);
        revealStagedCaption(captionEl);
    });

    finalDemoBoxEl?.classList.remove("is-pending");
    const controlsEl = document.getElementById("final-demo-controls");
    if (controlsEl) {
        controlsEl.hidden = false;
    }
    const fullscreenToggle = document.getElementById("final-demo-fullscreen");
    if (fullscreenToggle) {
        fullscreenToggle.hidden = false;
    }

    const scrollEl = document.getElementById("calc-log-scroll");
    if (scrollEl) {
        scrollEl.replaceChildren();
        job.log.forEach(line => renderCalcLogLine(scrollEl, line.t, line.text));
    }

    if (startGenerationButton) {
        startGenerationButton.disabled = true;
        startGenerationButton.textContent = "✓ GENERATION COMPLETE";
    }

    showGrid();
    requestAnimationFrame(() => finalDemoViewer?.resizeToCanvas());
}

function startFromInput(selection) {
    if (jobStore.generating()) {
        return;
    }
    const job = jobStore.add(selection);
    applyJobInput(job);
    resetGridForGeneration();
    showGrid();
    staggerGridEntrance();
    runGenerationSequence(job);
}

function selectJob(id) {
    const job = jobStore.get(id);
    if (!job || jobStore.generating()) {
        return;
    }
    jobStore.setActive(id);
    showCompletedJob(job);
    renderJobs();
}

async function generateNew() {
    if (jobStore.generating() || choosingNewInput()) {
        return;
    }
    if (jobStore.unsaved().length) {
        const choice = await confirmUnsaved("new");
        if (choice === "cancel") {
            return;
        }
    }
    collapseFinalDemoInstantly();
    inputView?.reset();
    showInputView();
    renderJobs();
}

// ---- Save = download the job(s) as JSON and keep a copy in this browser's
// Saved list (there is no server storage) ----

const savedStore = createSavedStore();

function downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function saveJobs(ids) {
    const jobs = ids.map(id => jobStore.get(id)).filter(Boolean);
    if (!jobs.length) {
        return;
    }
    downloadJson(jobsExport(jobs), exportFilename(jobs));
    jobs.forEach(job => savedStore.put(savedRecord(job)));
    jobStore.markSaved(jobs.map(job => job.id));
}

// ---- Jobs panel: Generate New, Saved, PINNED, RECENT (newest at the top);
// plus the expanded-view tab strip ----

function tierDotClass(routing) {
    if (!routing || routing.tier == null) {
        return "tier-none";
    }
    return routing.tier === 2 ? "tier-2" : "tier-1";
}

const ICONS = {
    pin: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/></svg>',
    trash: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
};

// Delete is two-step (click, then "Delete?" within 3 s), so a stray click
// never throws a job away. Survives re-renders via this id.
let pendingDelete = null;

function armDelete(key) {
    clearTimeout(pendingDelete?.timer);
    pendingDelete = {
        key,
        timer: setTimeout(() => {
            pendingDelete = null;
            renderJobs();
            if (!savedModal?.hidden) {
                renderSaved();
            }
        }, 3000),
    };
}

function deleteJob(id) {
    const job = jobStore.get(id);
    if (!job || job.status === "generating") {
        return;
    }
    clearTimeout(pendingDelete?.timer);
    pendingDelete = null;
    const wasActive = jobStore.active() === job;
    const next = jobStore.remove(id);
    if (!next) {
        // last job gone: back to a fresh input page
        collapseFinalDemoInstantly();
        inputView?.reset();
        showInputView();
        return;
    }
    if (wasActive && !choosingNewInput()) {
        showCompletedJob(next);
    }
    renderJobs();
}

function iconAction(className, label, iconSvg) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `job-icon ${className}`;
    button.setAttribute("aria-label", label);
    button.title = label;
    button.innerHTML = iconSvg;
    return button;
}

function jobItem(job, { running, active, choosing }) {
    const li = document.createElement("li");
    li.className = "job-item";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "job-select";
    button.dataset.jobId = job.id;
    if (job === active && !choosing) {
        button.setAttribute("aria-current", "true");
    }
    button.disabled = Boolean(running) && job !== running;
    if (button.disabled) {
        button.title = "Available when the current generation finishes";
    }
    button.addEventListener("click", () => selectJob(job.id));

    const head = document.createElement("div");
    head.className = "job-head";
    const num = document.createElement("span");
    num.className = "job-num";
    num.textContent = jobLabel(job);
    const status = document.createElement("span");
    status.className = `job-status${job.status === "generating" ? " is-running" : ""}`;
    status.textContent = job.status === "generating" ? `Generating ${job.progress}%` : "Complete";
    head.append(num, status);

    const title = document.createElement("div");
    title.className = "job-title";
    title.textContent = job.input.title;

    const tier = document.createElement("div");
    tier.className = `job-tier ${tierDotClass(job.input.routing)}`;
    tier.textContent = job.input.routing.label;

    button.append(head, title, tier);

    const foot = document.createElement("div");
    foot.className = "job-foot";
    const saved = document.createElement("span");
    saved.className = `job-saved${job.saved ? " is-saved" : ""}`;
    saved.textContent = job.saved ? "Saved" : "Unsaved";

    const actions = document.createElement("div");
    actions.className = "job-actions";

    const save = document.createElement("button");
    save.type = "button";
    save.className = "job-save";
    save.textContent = "Save";
    save.setAttribute("aria-label", `Save ${jobLabel(job)} (downloads a JSON file and adds it to Saved)`);
    save.disabled = job.status !== "complete";
    save.addEventListener("click", () => saveJobs([job.id]));

    const pin = iconAction(`job-pin${job.pinned ? " is-pinned" : ""}`,
        job.pinned ? `Unpin ${jobLabel(job)}` : `Pin ${jobLabel(job)}`, ICONS.pin);
    pin.setAttribute("aria-pressed", String(job.pinned));
    pin.addEventListener("click", () => jobStore.togglePin(job.id));

    const trash = iconAction("job-trash", `Delete ${jobLabel(job)}`, ICONS.trash);
    trash.disabled = job.status === "generating";
    trash.addEventListener("click", async () => {
        if (await confirmDeleteJob(job)) {
            deleteJob(job.id);
        }
    });

    actions.append(save, pin, trash);
    foot.append(saved, actions);
    li.append(button, foot);
    return li;
}

function renderJobs() {
    // Pages-panel Jobs list first: it must also refresh (to its empty state)
    // when the jobs panel below bails out with no jobs and no saved work.
    renderNavJobs();
    const hasJobs = jobStore.count() > 0;
    const savedCount = savedStore.list().length;
    // Shown once there's a job, or earlier saved work to reopen (after a reload).
    const showPanel = hasJobs || savedCount > 0;
    workbenchPageEl?.classList.toggle("has-jobs", showPanel);
    if (jobsPanelEl) {
        jobsPanelEl.hidden = !showPanel;
    }
    if (savedCountEl) {
        savedCountEl.textContent = savedCount ? String(savedCount) : "";
    }
    if (!showPanel || !jobsListEl) {
        return;
    }

    const running = jobStore.generating();
    const active = jobStore.active();
    const choosing = choosingNewInput();
    const unsavedCount = jobStore.unsaved().length;
    const ctx = { running, active, choosing };

    if (jobsCountEl) {
        jobsCountEl.textContent = `${jobStore.count()} · ${unsavedCount} unsaved`;
    }
    if (generateNewButton) {
        generateNewButton.disabled = Boolean(running) || choosing;
        generateNewButton.title = running
            ? "Available when the current generation finishes"
            : choosing ? "Already choosing a new input" : "Choose a new input; current jobs stay open";
    }

    const pinned = jobStore.pinnedOrder();
    jobsPinnedEl?.replaceChildren(...pinned.map(job => jobItem(job, ctx)));
    if (jobsPinnedEmptyEl) {
        jobsPinnedEmptyEl.hidden = pinned.length > 0;
    }

    const items = [];
    if (!hasJobs) {
        const none = document.createElement("li");
        none.className = "jobs-empty";
        none.textContent = "No jobs in this tab yet";
        items.push(none);
    } else if (choosing) {
        const pending = document.createElement("li");
        pending.className = "job-item job-item-pending";
        pending.textContent = "New job — choosing input…";
        items.push(pending);
    }
    jobStore.panelOrder().forEach(job => items.push(jobItem(job, ctx)));
    if (!items.length) {
        const none = document.createElement("li");
        none.className = "jobs-empty";
        none.textContent = "No other jobs (all pinned)";
        items.push(none);
    }
    jobsListEl.replaceChildren(...items);

    renderJobTabs();
}

// ---- Inline job rename, shared by the pages panel's Jobs list and the 3D
// view's tab strip. Both render from jobStore, so a rename in either place
// shows in both (and in the jobs panel, logs and exports via jobLabel()).
// One persistent <input> survives re-renders (progress ticks re-render the
// lists) so typing isn't interrupted.
let renaming = null; // { id, where: "nav" | "tab" }
const renameInput = document.createElement("input");
renameInput.type = "text";
renameInput.maxLength = 60;
renameInput.className = "job-rename-input";
renameInput.setAttribute("aria-label", "Job name");

function startRename(id, where) {
    const job = jobStore.get(id);
    if (!job) {
        return;
    }
    renaming = { id, where };
    renameInput.value = jobLabel(job);
    renderJobs();
    renameInput.focus();
    renameInput.select();
}

function endRename(commit) {
    if (!renaming) {
        return;
    }
    const { id } = renaming;
    renaming = null;
    if (commit) {
        jobStore.rename(id, renameInput.value); // emits → renderJobs
    }
    renderJobs();
}

renameInput.addEventListener("keydown", event => {
    if (event.key === "Enter") {
        event.preventDefault();
        endRename(true);
    } else if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        endRename(false);
    }
});
renameInput.addEventListener("blur", () => {
    // re-renders move the input (blur fires); only a real focus loss commits
    setTimeout(() => {
        if (renaming && document.activeElement !== renameInput) {
            endRename(true);
        }
    }, 0);
});

function renameField(job, className) {
    const wrap = document.createElement("div");
    wrap.className = className;
    const hadFocus = document.activeElement === renameInput;
    const [a, b] = [renameInput.selectionStart, renameInput.selectionEnd];
    wrap.append(renameInput);
    if (hadFocus) {
        requestAnimationFrame(() => {
            renameInput.focus({ preventScroll: true });
            renameInput.setSelectionRange(a, b);
        });
    }
    return wrap;
}

// ---- Jobs section of the pages panel
const navJobsList = document.getElementById("page-nav-jobs-list");
const navJobsCount = document.getElementById("page-nav-jobs-count");
document.getElementById("page-nav-jobs")?.addEventListener("click", () => {
    if (!pageNav?.classList.contains("expanded")) {
        setNavExpanded(true);
    }
    navJobsList?.querySelector("button")?.focus({ preventScroll: true });
});

function renderNavJobs() {
    if (!navJobsList) {
        return;
    }
    const jobs = jobStore.panelOrder().concat(jobStore.pinnedOrder());
    if (navJobsCount) {
        navJobsCount.textContent = String(jobStore.count());
    }
    if (!jobs.length) {
        const empty = document.createElement("p");
        empty.className = "page-nav-jobs-empty";
        empty.textContent = "No jobs running. Choose an input on Workbench and press START GENERATION.";
        navJobsList.replaceChildren(empty);
        return;
    }
    const active = jobStore.active();
    const running = jobStore.generating();
    navJobsList.replaceChildren(...jobs.map(job => {
        if (renaming?.id === job.id && renaming.where === "nav") {
            return renameField(job, "page-nav-job is-editing");
        }
        const row = document.createElement("div");
        row.className = `page-nav-job${job === active ? " is-active" : ""}`;
        const open = document.createElement("button");
        open.type = "button";
        open.className = "page-nav-job-open";
        open.disabled = Boolean(running) && job !== running;
        open.title = `${jobLabel(job)} · ${job.input.title} · double-click to rename`;
        const name = document.createElement("span");
        name.className = "page-nav-job-name";
        name.textContent = jobLabel(job);
        const meta = document.createElement("span");
        meta.className = "page-nav-job-meta";
        meta.textContent = job.status === "generating"
            ? `Generating ${job.progress}%`
            : `${job.saved ? "Saved" : "Unsaved"} · ${job.input.title}`;
        open.append(name, meta);
        open.addEventListener("click", () => {
            setActivePage("page-workbench");
            selectJob(job.id);
        });
        open.addEventListener("dblclick", event => {
            event.preventDefault();
            startRename(job.id, "nav");
        });
        const edit = document.createElement("button");
        edit.type = "button";
        edit.className = "page-nav-job-edit";
        edit.setAttribute("aria-label", `Rename ${jobLabel(job)}`);
        edit.title = "Rename";
        edit.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/></svg>';
        edit.addEventListener("click", () => startRename(job.id, "nav"));
        row.append(open, edit);
        return row;
    }));
}

// ---- Job delete confirmation (trash icon): a real modal, no "don't show again"
const jobDeleteModal = document.getElementById("job-delete-modal");
let jobDeleteResolve = null;
let jobDeleteReturnFocus = null;

function confirmDeleteJob(job) {
    if (!jobDeleteModal) {
        return Promise.resolve(window.confirm(`Delete ${jobLabel(job)}?`));
    }
    jobDeleteResolve?.(false);
    document.getElementById("job-delete-title").textContent = `Delete ${jobLabel(job)}?`;
    document.getElementById("job-delete-body").textContent = `${jobLabel(job)} · ${job.input.title} will be removed from this session`
        + (job.saved
            ? ". Its saved copy stays in Saved."
            : ". It isn't saved, so its input, DEM and calculation log will be lost. This can't be undone.");
    jobDeleteReturnFocus = document.activeElement;
    jobDeleteModal.hidden = false;
    document.getElementById("job-delete-cancel").focus();
    return new Promise(resolve => {
        jobDeleteResolve = resolve;
    });
}

function settleJobDelete(ok) {
    if (!jobDeleteResolve) {
        return;
    }
    jobDeleteModal.hidden = true;
    const resolve = jobDeleteResolve;
    jobDeleteResolve = null;
    jobDeleteReturnFocus?.focus?.({ preventScroll: true });
    resolve(ok);
}

document.getElementById("job-delete-cancel")?.addEventListener("click", () => settleJobDelete(false));
document.getElementById("job-delete-backdrop")?.addEventListener("click", () => settleJobDelete(false));
document.getElementById("job-delete-ok")?.addEventListener("click", () => settleJobDelete(true));
document.addEventListener("keydown", event => {
    if (event.key === "Escape" && jobDeleteModal && !jobDeleteModal.hidden) {
        settleJobDelete(false);
    }
});

function renderJobTabs() {
    if (!jobTabsEl) {
        return;
    }
    const active = jobStore.active();
    const running = jobStore.generating();

    jobTabsEl.replaceChildren(...jobStore.creationOrder().map(job => {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = `job-tab ${tierDotClass(job.input.routing)}`;
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-selected", String(job === active));
        tab.dataset.jobId = job.id;
        tab.disabled = Boolean(running) && job !== running;
        tab.title = `${jobLabel(job)} · ${job.input.title} · ${job.input.routing.label}${job.saved ? "" : " · unsaved"}`;

        const dot = document.createElement("span");
        dot.className = "job-tab-dot";
        dot.setAttribute("aria-hidden", "true");
        const label = document.createElement("span");
        label.className = "job-tab-label";
        label.textContent = `${jobLabel(job)} · ${job.input.title}`;
        tab.append(dot, label);
        if (job.pinned) {
            const pinMark = document.createElement("span");
            pinMark.className = "job-tab-pin";
            pinMark.setAttribute("aria-label", "pinned");
            pinMark.innerHTML = ICONS.pin;
            tab.append(pinMark);
        }
        if (!job.saved) {
            const mark = document.createElement("span");
            mark.className = "job-tab-unsaved";
            mark.setAttribute("aria-label", "unsaved");
            mark.textContent = "●";
            tab.append(mark);
        }
        tab.addEventListener("click", () => selectJob(job.id));
        tab.addEventListener("dblclick", event => {
            event.preventDefault();
            startRename(job.id, "tab");
        });
        tab.title += " · double-click to rename";
        if (renaming?.id === job.id && renaming.where === "tab") {
            return renameField(job, "job-tab job-tab-editing");
        }
        return tab;
    }));

    renderSource(active);
    renderFacts(active);

    if (jobTabInfoEl && active) {
        const dem = active.input.dem;
        jobTabInfoEl.textContent = [
            `${jobLabel(active)}: ${active.input.title}`,
            active.input.routing.label,
            dem ? `DEM ${dem.min_m}–${dem.max_m} m` : null,
            "3D terrain shown: Darjeeling reference (placeholder)",
        ].filter(Boolean).join(" · ");
    }
}

jobStore.onChange(renderJobs);
renderNavJobs(); // empty state before the first job
generateNewButton?.addEventListener("click", generateNew);

// ---- Toolbar show/hide (the panel collapses to a narrow strip and the page
// content, including the expanded 3D view, moves over — never an overlay) ----

const JOBS_COLLAPSED_KEY = "dw2.jobsPanelCollapsed";

function setJobsPanelCollapsed(collapsed) {
    workbenchPageEl?.classList.toggle("jobs-collapsed", collapsed);
    const label = collapsed ? "Show toolbar" : "Hide toolbar";
    jobsPanelToggle?.setAttribute("aria-expanded", String(!collapsed));
    jobsPanelToggle?.setAttribute("aria-label", label);
    const tip = jobsPanelToggle?.querySelector(".jobs-tip");
    if (tip) {
        tip.textContent = label;
    }
    try {
        localStorage.setItem(JOBS_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
        // storage blocked: the choice just isn't remembered
    }
}

jobsPanelToggle?.addEventListener("click", () => {
    setJobsPanelCollapsed(!workbenchPageEl.classList.contains("jobs-collapsed"));
});

try {
    setJobsPanelCollapsed(localStorage.getItem(JOBS_COLLAPSED_KEY) === "1");
} catch {
    setJobsPanelCollapsed(false);
}

// ---- Saved window: earlier saved work, reopened as a (finished) job ----

const savedModal = document.getElementById("saved-modal");
const savedListEl = document.getElementById("saved-list");
const savedOpenButton = document.getElementById("saved-open-button");
let savedReturnFocus = null;

function formatSavedAt(iso) {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, {
        year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
}

// Opening a saved job before anything was generated this session: the shared
// output views (mini previews, final 3D viewer) must exist first.
async function ensureGeneratedViews() {
    [["dsm-3d-canvas", "dsm-3d"], ["metric-elevation-3d-canvas", "elevation-3d"]].forEach(([canvasId, layer]) => {
        const canvas = document.getElementById(canvasId);
        if (canvas && !miniPreviewCanvases.has(canvas)) {
            miniPreviewCanvases.add(canvas);
            createMiniPreview(canvas, layer);
        }
    });
    await Promise.all([initFinalDemoViewer(), loadMiniPreviewAssets()]);
    staggerGridEntrance();
}

async function openSavedWork(uid) {
    if (jobStore.generating()) {
        return;
    }
    const record = savedStore.get(uid);
    if (!record) {
        return;
    }
    closeSaved();
    const existing = jobStore.findByUid(uid);
    if (existing) {
        selectJob(existing.id);
        return;
    }
    await ensureGeneratedViews();
    const job = jobStore.add(record.input, record);
    showCompletedJob(job);
    renderJobs();
}

function renderSaved() {
    if (!savedListEl) {
        return;
    }
    const records = savedStore.list();
    if (!records.length) {
        const empty = document.createElement("div");
        empty.className = "saved-empty";
        empty.textContent = "No saved work yet. Use Save on a job in the jobs panel.";
        savedListEl.replaceChildren(empty);
        return;
    }
    const running = jobStore.generating();

    savedListEl.replaceChildren(...records.map(record => {
        const card = document.createElement("article");
        card.className = "saved-card";

        const thumb = document.createElement("div");
        thumb.className = "saved-thumb";
        if (record.input.previewUrl) {
            const img = document.createElement("img");
            img.src = record.input.previewUrl;
            img.alt = "";
            img.loading = "lazy";
            img.addEventListener("error", () => img.remove()); // the input file may be gone
            thumb.append(img);
        }

        const body = document.createElement("div");
        body.className = "saved-body";
        const title = document.createElement("div");
        title.className = "saved-card-title";
        title.textContent = record.input.title;
        const tier = document.createElement("div");
        tier.className = `job-tier ${tierDotClass(record.input.routing)}`;
        tier.textContent = record.input.routing?.label ?? "";
        const meta = document.createElement("div");
        meta.className = "saved-meta";
        const dem = record.input.dem;
        meta.textContent = [
            `Saved ${formatSavedAt(record.savedAt)}`,
            dem ? `DEM ${dem.min_m}–${dem.max_m} m` : null,
            jobStore.findByUid(record.uid) ? "open now" : null,
        ].filter(Boolean).join(" · ");
        body.append(title, tier, meta);

        const actions = document.createElement("div");
        actions.className = "saved-actions";
        const open = document.createElement("button");
        open.type = "button";
        open.className = "confirm-button confirm-primary";
        open.textContent = jobStore.findByUid(record.uid) ? "Go to job" : "Open";
        open.disabled = Boolean(running);
        if (running) {
            open.title = "Available when the current generation finishes";
        }
        open.addEventListener("click", () => openSavedWork(record.uid));

        const download = document.createElement("button");
        download.type = "button";
        download.className = "confirm-button";
        download.textContent = "Download";
        download.addEventListener("click", () => {
            const pseudo = { ...record, n: 1, status: "complete" };
            downloadJson(jobsExport([pseudo]), `depthwizard-saved-${record.savedAt.slice(0, 19).replace(/[:T]/g, "-")}.json`);
        });

        const confirming = pendingDelete?.key === `saved:${record.uid}`;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = `confirm-button${confirming ? " confirm-danger" : ""}`;
        remove.textContent = confirming ? "Remove?" : "Remove";
        remove.setAttribute("aria-label", confirming ? `Confirm: remove ${record.input.title} from Saved` : `Remove ${record.input.title} from Saved`);
        remove.addEventListener("click", () => {
            if (pendingDelete?.key === `saved:${record.uid}`) {
                clearTimeout(pendingDelete.timer);
                pendingDelete = null;
                savedStore.remove(record.uid);
                const openJob = jobStore.findByUid(record.uid);
                if (openJob) {
                    jobStore.update(openJob.id, { saved: false });
                }
                renderSaved();
                renderJobs();
            } else {
                armDelete(`saved:${record.uid}`);
                renderSaved();
            }
        });

        actions.append(open, download, remove);
        card.append(thumb, body, actions);
        return card;
    }));
}

function openSaved() {
    if (!savedModal) {
        return;
    }
    renderSaved();
    savedReturnFocus = document.activeElement;
    savedModal.hidden = false;
    document.getElementById("saved-close")?.focus();
}

function closeSaved() {
    if (!savedModal || savedModal.hidden) {
        return;
    }
    savedModal.hidden = true;
    savedReturnFocus?.focus?.({ preventScroll: true });
}

savedOpenButton?.addEventListener("click", openSaved);
document.getElementById("saved-close")?.addEventListener("click", closeSaved);
document.getElementById("saved-backdrop")?.addEventListener("click", closeSaved);
document.addEventListener("keydown", event => {
    if (event.key === "Escape" && savedModal && !savedModal.hidden) {
        closeSaved();
    }
});

// ---- Unsaved-work modal (in-app navigation, Close, and the desktop app's
// window close). Job count + per-job save choice; no "don't show again". ----

const unsavedModal = document.getElementById("unsaved-modal");
const unsavedTitle = document.getElementById("unsaved-title");
const unsavedBody = document.getElementById("unsaved-body");
const unsavedList = document.getElementById("unsaved-list");
const unsavedNote = document.getElementById("unsaved-note");
const unsavedCancel = document.getElementById("unsaved-cancel");
const unsavedDiscard = document.getElementById("unsaved-discard");
const unsavedSave = document.getElementById("unsaved-save");
let unsavedResolve = null;
let unsavedSaveLabel = "";
let unsavedReturnFocus = null;

function checkedUnsavedIds() {
    return [...(unsavedList?.querySelectorAll("input[type=checkbox]:checked") ?? [])].map(box => box.value);
}

function updateUnsavedSaveButton() {
    const count = checkedUnsavedIds().length;
    if (unsavedSave) {
        unsavedSave.disabled = count === 0;
        unsavedSave.textContent = unsavedSaveLabel.replace("selected", `selected (${count})`);
    }
}

function confirmUnsaved(action) {
    if (!unsavedModal) {
        return Promise.resolve(window.confirm("You have unsaved jobs. Continue?") ? "discard" : "cancel");
    }
    unsavedResolve?.("cancel");

    const jobs = jobStore.unsaved();
    const copy = unsavedCopy(action, jobs.length);
    unsavedTitle.textContent = copy.title;
    unsavedBody.textContent = copy.body;
    unsavedNote.textContent = STORAGE_NOTE;
    unsavedDiscard.textContent = copy.discard;
    unsavedSaveLabel = copy.save;

    unsavedList.replaceChildren(...jobs.map(job => {
        const label = document.createElement("label");
        label.className = "unsaved-job";
        const box = document.createElement("input");
        box.type = "checkbox";
        box.value = job.id;
        box.checked = job.status === "complete";
        box.disabled = job.status !== "complete";
        box.addEventListener("change", updateUnsavedSaveButton);
        const text = document.createElement("span");
        text.textContent = `${jobLabel(job)} · ${job.input.title} · ${job.input.routing.label}`
            + (job.status === "complete" ? "" : " · still generating (can't be saved yet)");
        label.append(box, text);
        return label;
    }));
    updateUnsavedSaveButton();

    unsavedReturnFocus = document.activeElement;
    unsavedModal.hidden = false;
    unsavedCancel.focus();
    return new Promise(resolve => {
        unsavedResolve = resolve;
    });
}

function settleUnsaved(choice) {
    if (!unsavedResolve) {
        return;
    }
    if (choice === "save") {
        const ids = checkedUnsavedIds();
        if (!ids.length) {
            return;
        }
        saveJobs(ids);
    }
    unsavedModal.hidden = true;
    const resolve = unsavedResolve;
    unsavedResolve = null;
    unsavedReturnFocus?.focus?.({ preventScroll: true });
    resolve(choice);
}

unsavedCancel?.addEventListener("click", () => settleUnsaved("cancel"));
document.getElementById("unsaved-backdrop")?.addEventListener("click", () => settleUnsaved("cancel"));
unsavedDiscard?.addEventListener("click", () => settleUnsaved("discard"));
unsavedSave?.addEventListener("click", () => settleUnsaved("save"));
document.addEventListener("keydown", event => {
    if (event.key === "Escape" && unsavedModal && !unsavedModal.hidden) {
        settleUnsaved("cancel");
    }
});

// Tab/window close: Tauri desktop gets the same custom modal; the web build
// can only use the browser's own generic beforeunload prompt.
const closeGuardReady = installCloseGuard({
    hasUnsaved: () => jobStore.unsaved().length > 0,
    confirmUnsaved,
    isUnloadAllowed: () => unloadAllowed,
});

if (import.meta.env.DEV) {
    window.__dwJobs = { store: jobStore, closeGuardReady, viewer: () => finalDemoViewer };
}

if (inputViewEl) {
    inputView = createInputView(inputViewEl, { onStart: startFromInput });
}

renderJobs(); // shows the panel on load if there is saved work to reopen


// ============================================================
// PAGE NAVIGATION (collapsible left icon rail)
// ============================================================

const pageNav = document.getElementById("page-nav");
const pageNavToggle = document.getElementById("page-nav-toggle");
const pageNavTabs = document.querySelectorAll(".page-nav-tab");
const pages = document.querySelectorAll(".page");

// Page order (2026-09-24): 1 Workbench (default), 2 Explore, 3 Docs.
let activePageId = "page-workbench";

function setActivePage(pageId) {
    activePageId = pageId;

    pages.forEach(page => {
        page.classList.toggle("active", page.id === pageId);
    });

    pageNavTabs.forEach(tab => {
        tab.classList.toggle("active", tab.dataset.page === pageId);
    });

    if (pageId === "page-workbench") {
        requestAnimationFrame(initWorkbenchGrid);
    }
    syncNavDefault();
}

pageNavTabs.forEach(tab => {
    tab.addEventListener("click", () => {
        setActivePage(tab.dataset.page);
    });
});

// The panel's expanded state also drives --nav-w (body.nav-expanded), so the
// Workbench jobs panel and the expanded 3D view move over instead of being covered.
function setNavExpanded(expanded) {
    pageNav?.classList.toggle("expanded", expanded);
    document.body.classList.toggle("nav-expanded", expanded);
    pageNavToggle?.setAttribute("aria-expanded", String(expanded));
}

pageNavToggle?.addEventListener("click", () => {
    setNavExpanded(!pageNav?.classList.contains("expanded"));
});

// Default visibility: expanded while the post-generation 3D window is open,
// collapsed everywhere else. Applied on each transition; manual toggles in
// between are respected.
function syncNavDefault() {
    const viewing3d = activePageId === "page-workbench"
        && document.getElementById("final-demo-box")?.classList.contains("is-expanded");
    setNavExpanded(Boolean(viewing3d));
}

// Workbench is the landing page, so run its first-show init (grid entrance) on load too.
setActivePage(activePageId);

document.getElementById("build-your-own-card")?.addEventListener("click", () => {
    setActivePage("page-workbench");
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
    "RDAH-Net (Swiss building-height checkpoint), zero-shot on Sentinel-2: rejected — no height-above-ground signal at 10 m vs ICESat-2/GEDI (checkerboard artifacts observed, later shown intrinsic to the model)",
    "── end of trace ──",
];

let calcLogStartTime = 0;

// Appends a line to the on-screen log and to the job's own record (so a
// finished job's log can be shown again when the user switches back to it).
function appendCalcLogLine(scrollEl, text, job) {
    const elapsedSeconds = (Date.now() - calcLogStartTime) / 1000;
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = (elapsedSeconds % 60).toFixed(2).padStart(5, "0");
    const timestamp = `${String(minutes).padStart(2, "0")}:${seconds}`;

    job?.log.push({ t: timestamp, text });
    if (jobStore.active() === job) {
        renderCalcLogLine(scrollEl, timestamp, text);
    }
}

function renderCalcLogLine(scrollEl, timestamp, text) {
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

    if (sceneSelectionSummary?.type === "input") {
        opening.push(...sceneSelectionSummary.logLines);
        opening.push("Generation stages below are a placeholder: they show the Darjeeling reference outputs");
    }

    opening.push("Handing off to reconstruction pipeline…");

    return [...opening, ...CALC_LOG_LINES];
}

// totalDurationMs paces the log to span roughly the same window as the
// box-by-box sequence driving it, so it reads as running alongside that
// work rather than being dumped out ahead of or behind it.
function runCalcLog(totalDurationMs, job) {
    const scrollEl = document.getElementById("calc-log-scroll");

    if (!scrollEl) {
        return;
    }

    calcLogStartTime = Date.now();

    const lines = buildCalcLogLines();
    const interval = totalDurationMs / lines.length;

    lines.forEach((text, index) => {
        setTimeout(() => appendCalcLogLine(scrollEl, text, job), 200 + index * interval);
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
// visual parity, and Explore's own equivalents are scoped to #page-explore
// (see activateLayer / the region-rail bindings above) so the two
// viewers' independent state never cross-wires. Deferred until box 8's
// turn in the START GENERATION sequence — initFinalDemoViewer() below
// only runs once, the first time that happens. ----

let finalDemoViewer = null;
let finalDemoMeasureTool = null;
let finalDemoChrome = null;
let finalDemoCurrentLayer = "satellite-3d";
let finalDemoEarthquakeActive = false;
let finalDemoCurrentTerrain = null;
let finalDemoCurrentRegionKey = null;
let finalDemoRunning = false;
let finalDemoFloodActive = false;
let finalDemoFlythrough = null;
let finalDemoInitialized = false;
let finalDemoFloodSim = null;

// Workbench's Final Demo box stays locked to whatever scene the box 1-7
// sequence generated — no region switcher, unlike Explore's own viewer.
// Gated with this flag (rather than a forked copy of initFinalDemoViewer)
// since the two instances share the same createTerrainViewer component.
const FINAL_DEMO_REGION_SWITCHER_ENABLED = false;

function updateFinalDemoStats(regionKey, terrain, terrainData) {
    const region = REGIONS[regionKey];

    renderTerrainStats(terrainData);
    syncExaggerationSlider(terrain);

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

// Vertical exaggeration slider (Details box): display-only mesh scale, range
// computed per terrain in terrain.js; measurements/stats read the raw grid.
function formatExaggeration(value) {
    return `x${value < 10 ? value.toFixed(value < 1 ? 2 : 1) : Math.round(value)}`;
}

function syncExaggerationSlider(terrain) {
    const slider = document.getElementById("xp-vex-slider");
    if (!slider || !terrain?.setDisplayExaggeration) {
        return;
    }
    slider.min = String(terrain.minDisplayExaggeration);
    slider.max = String(terrain.maxDisplayExaggeration);
    slider.step = String((terrain.maxDisplayExaggeration - terrain.minDisplayExaggeration) / 400);
    slider.value = String(terrain.displayExaggeration());
    document.getElementById("xp-vex-min").textContent = formatExaggeration(terrain.minDisplayExaggeration);
    document.getElementById("xp-vex-max").textContent = formatExaggeration(terrain.maxDisplayExaggeration);
    document.getElementById("xp-vex-value").textContent = formatExaggeration(terrain.displayExaggeration());
}

function initExaggerationSlider() {
    const slider = document.getElementById("xp-vex-slider");
    slider?.addEventListener("input", () => {
        if (!finalDemoCurrentTerrain) {
            return;
        }
        const value = finalDemoCurrentTerrain.setDisplayExaggeration(Number(slider.value));
        document.getElementById("xp-vex-value").textContent = formatExaggeration(value);
        finalDemoMeasureTool?.invalidate(); // markers/pins re-project onto the rescaled surface
    });
}

function activateFinalDemoLayer(layer) {
    if (!finalDemoCurrentTerrain) {
        return;
    }

    finalDemoCurrentTerrain.setLayer(layer);
    finalDemoCurrentLayer = layer;
    finalDemoFloodSim?.refresh();
    // Wireframe: neon lines on pure black; every other layer uses the theme's background.
    finalDemoViewer.setBackground(layer === "wireframe-3d" ? 0x000000 : WORKBENCH_THEME_BG_HEX[workbenchTheme]);

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
    // Expanded view: a water plane at a chosen level (flood-sim.js) instead of the old low-ground tint.
    finalDemoFloodSim?.setActive(finalDemoFloodActive);
    syncScenarioNote();
}

// Scenario overlays share the terrain's vertex colours, so at most one is on.
const SCENARIO_NOTES = {
    flood: "Flood (illustrative): a flat water plane at the chosen level, filling every DEM cell below it (a 'bathtub' fill). "
        + "It ignores flow and connectivity. It is not a hydrological flood model.",
    earthquake: "Earthquake — PLACEHOLDER, not a seismic hazard model. The red gradient only marks the steepest slopes " +
        "(a slope-based heuristic on the DEM); there is no earthquake model anywhere in this project.",
};

function syncScenarioNote() {
    const note = document.getElementById("final-demo-scenario-note");
    if (!note) {
        return;
    }
    const key = finalDemoEarthquakeActive ? "earthquake" : finalDemoFloodActive ? "flood" : null;
    note.hidden = !key;
    note.textContent = key ? SCENARIO_NOTES[key] : "";
    note.classList.toggle("is-placeholder", key === "earthquake");
}

function setFinalDemoEarthquakeActive(active) {
    finalDemoEarthquakeActive = active;
    document.getElementById("final-demo-earthquake-button")?.classList.toggle("active", active);
    finalDemoCurrentTerrain?.setEarthquakeOverlay(active);
    syncScenarioNote();
}

function selectFinalDemoLayer(layer) {
    if (finalDemoFloodActive) {
        setFinalDemoFloodActive(false);
    }
    if (finalDemoEarthquakeActive) {
        setFinalDemoEarthquakeActive(false);
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
// custom properties #page-workbench[data-theme="light"] overrides, so flipping
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

    const page2 = document.getElementById("page-workbench");
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

    finalDemoViewer?.setBackground(finalDemoCurrentLayer === "wireframe-3d" ? 0x000000 : WORKBENCH_THEME_BG_HEX[theme]);
    finalDemoChrome?.syncTheme();
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

    // Measurement tool (expanded view only; deliverables-audit item 6).
    finalDemoMeasureTool = createMeasureTool({
        viewer: finalDemoViewer,
        box: document.getElementById("final-demo-box"),
        canvas,
    });
    // Toolbar, library, terrain context menu, notes, screenshots, play/pause, theme.
    finalDemoChrome = createExpandedChrome({
        box: document.getElementById("final-demo-box"),
        canvas,
        viewer: finalDemoViewer,
        tool: finalDemoMeasureTool,
        getRegionKey: () => finalDemoCurrentRegionKey,
        setLayer: selectFinalDemoLayer,
        getLayer: () => finalDemoCurrentLayer,
        startFlythrough: () => finalDemoFlythrough?.start(),
        getTheme: () => workbenchTheme,
        setTheme: applyWorkbenchTheme,
    });
    if (import.meta.env?.DEV) {
        window.__dwMeasureTool = finalDemoMeasureTool; // dev-only hooks for the headless tests
        window.__dwChrome = finalDemoChrome;
    }
    // Side-panel boxes (collapsible) + the Explore Options product tour.
    initCollapsibleBoxes(document.getElementById("final-demo-box"));
    initTour(document.getElementById("final-demo-box"));
    initFacts(() => jobStore.active());
    initExaggerationSlider();
    finalDemoFloodSim = createFloodSim({
        getTerrain: () => finalDemoCurrentTerrain,
        onChange: () => finalDemoMeasureTool?.invalidate(),
        els: {
            panel: document.getElementById("xp-flood-panel"),
            slider: document.getElementById("xp-flood-slider"),
            levelLabel: document.getElementById("xp-flood-level"),
            playBtn: document.getElementById("xp-flood-play"),
            resetBtn: document.getElementById("xp-flood-reset"),
            pctEl: document.getElementById("xp-flood-pct"),
            areaEl: document.getElementById("xp-flood-area"),
        },
    });
    if (import.meta.env?.DEV) {
        window.__dwFloodSim = finalDemoFloodSim; // dev-only, headless checks
        window.__dwTerrain = () => finalDemoCurrentTerrain;
        window.__dwCamera = () => finalDemoViewer.camera;
        window.__dwTHREE = THREE;
        window.__dwViewer = finalDemoViewer;
    }
    renderSource(jobStore.active());
    renderFacts(jobStore.active());

    finalDemoFlythrough = createFlythroughController({
        controls: finalDemoViewer.controls,
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
            if (finalDemoEarthquakeActive) {
                setFinalDemoEarthquakeActive(false);
            }
            activateFinalDemoLayer("dsm-3d");
        }
        setFinalDemoFloodActive(!finalDemoFloodActive);
    });

    document.getElementById("final-demo-earthquake-button")?.addEventListener("click", () => {
        if (!finalDemoEarthquakeActive) {
            if (finalDemoFloodActive) {
                setFinalDemoFloodActive(false);
            }
            activateFinalDemoLayer("dsm-3d");
        }
        setFinalDemoEarthquakeActive(!finalDemoEarthquakeActive);
    });

    const fullscreenToggle = document.getElementById("final-demo-fullscreen");
    const finalDemoBox = document.getElementById("final-demo-box");

    // ⛶ re-opens the same expanded view the pipeline pops up on completion
    // (one consistent "big view"; see FINAL DEMO — EXPANDED VIEW below).
    fullscreenToggle?.addEventListener("click", () => {
        expandFinalDemo();
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
    finalDemoMeasureTool?.update();
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

// Sequential box count (excludes box 2, which now reveals at selection
// time, and box 7, which runs continuously in parallel instead) — used
// to pace Calculation Logs to the same span.
const SEQUENTIAL_BOX_COUNT = 5;
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

// ---- Box 2 (Preview) reveals at selection time now (see
// revealScenePreview() above), not as part of the generation sequence. ----

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

    // One renderer per canvas: later jobs re-run the stage over the same view.
    if (!miniPreviewCanvases.has(canvas)) {
        miniPreviewCanvases.add(canvas);
        createMiniPreview(canvas, layer);
    }

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
    box.classList.remove("is-pending");
    if (captionEl) {
        revealStagedCaption(captionEl);
    }
}

const miniPreviewCanvases = new Set();

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
    document.getElementById("final-demo-box")?.classList.remove("is-pending");
    if (controlsEl) {
        controlsEl.hidden = false;
    }
    if (fullscreenToggle) {
        fullscreenToggle.hidden = false;
    }
}

// One job's run through the staged boxes. Only one job generates at a time
// (the boxes are shared); switching jobs and Generate New wait until it ends.
async function runGenerationSequence(job) {
    if (startGenerationButton) {
        startGenerationButton.disabled = true;
        startGenerationButton.textContent = "▶ GENERATING…";
    }

    // Starts filling immediately and keeps appending in parallel with
    // whichever box below is currently generating.
    runCalcLog(TOTAL_PIPELINE_MS, job);
    const stageDone = progress => jobStore.update(job.id, { progress });

    await generateCaptionPreviewBox("depth-preview-box", [
        "Loading Sentinel-2 RGB tiles…",
        "Running Depth Anything V2 (ViT-Large)…",
        "Inference complete — 1.17s, frozen weights…",
    ]);
    stageDone(20);

    await generateCaptionPreviewBox("elevation-preview-box", [
        "Loading terrain DEM (reference elevation)…",
        "Learned Sentinel-2 corrections: tested, not adopted…",
        "DEM elevation shown — no model correction applied…",
    ]);
    stageDone(40);

    await generateMiniPreviewBox("dsm-3d-box", "dsm-3d-canvas", "dsm-3d", [
        "Reprojecting DSM → EPSG:32645…",
        "Extruding 361×325 vertex grid…",
        "Applying vertical exaggeration 1.07x…",
    ]);
    stageDone(60);

    await generateMiniPreviewBox("metric-elevation-3d-box", "metric-elevation-3d-canvas", "elevation-3d", [
        "Running spatial-trend validation…",
        "Detrending elevation vs. position…",
        "Correlation +0.60 → −0.41 after detrending…",
    ]);
    stageDone(80);

    await generateFinalDemoBox();
    jobStore.update(job.id, { status: "complete", progress: 100 });

    if (startGenerationButton) {
        startGenerationButton.textContent = "✓ GENERATION COMPLETE";
    }

    // Whole pipeline done: pop the final 3D view out to fill the window.
    expandFinalDemo();
}


// ============================================================
// FINAL DEMO — EXPANDED VIEW (Back / Close) + CLOSE CONFIRMATION
// ============================================================

const finalDemoBoxEl = document.getElementById("final-demo-box");
const finalDemoExpandedBar = document.getElementById("final-demo-expanded-bar");
const finalDemoBackButton = document.getElementById("final-demo-back");
const finalDemoCloseButton = document.getElementById("final-demo-close");
// Constant speed, equal to the speed the earlier ease-out opened with (user
// request, 2026-09-25: "starts at the right speed, then drags"). That curve,
// cubic-bezier(0.2, 0.8, 0.2, 1) over 1880 ms, starts at 0.8 / 0.2 = 4x its
// average speed, so holding that speed covers the distance in 1880 / 4 ms.
const FINAL_DEMO_ANIM_MS = 470;
const FINAL_DEMO_EASING = "linear";
let finalDemoPlaceholder = null;
let finalDemoAnimating = false;

function prefersReducedMotion() {
    return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

// Edge offsets (top/right/bottom/left, px) of a viewport rect, for animating
// the fixed box between its grid cell and the full window.
function edgesOf(rect) {
    return {
        top: `${rect.top}px`,
        right: `${window.innerWidth - rect.right}px`,
        bottom: `${window.innerHeight - rect.bottom}px`,
        left: `${rect.left}px`,
    };
}

// "Full window" = everything right of the jobs panel once jobs exist
// (matches the .has-jobs .is-expanded rule in styles.css).
// Same arithmetic as the CSS rule (left: nav-w + 16px + jobs-w), not the
// panel's live rect, which may still be mid-transition when the nav opens.
function fullWindowEdges() {
    const panel = document.getElementById("jobs-panel");
    if (!panel || panel.hidden) {
        return { top: "0px", right: "0px", bottom: "0px", left: "0px" };
    }
    const px = (el, name) => parseFloat(getComputedStyle(el).getPropertyValue(name)) || 0;
    const left = px(document.body, "--nav-w") + 16 + px(workbenchPageEl, "--jobs-w");
    return { top: "0px", right: "0px", bottom: "0px", left: `${Math.round(left)}px` };
}

// The canvas resizes every frame via animate() → resizeToCanvas(), so the 3D
// view stays undistorted while the box grows or shrinks.
function animateFinalDemoEdges(fromEdges, toEdges, fromRadius, toRadius) {
    if (prefersReducedMotion() || !finalDemoBoxEl.animate) {
        return Promise.resolve();
    }
    const animation = finalDemoBoxEl.animate(
        [
            { ...fromEdges, borderRadius: fromRadius },
            { ...toEdges, borderRadius: toRadius },
        ],
        { duration: FINAL_DEMO_ANIM_MS, easing: FINAL_DEMO_EASING },
    );
    return animation.finished.catch(() => {});
}

// Pop out: the box grows from its grid cell to fill the window.
async function expandFinalDemo() {
    if (!finalDemoBoxEl || finalDemoAnimating || finalDemoBoxEl.classList.contains("is-expanded")) {
        return;
    }
    finalDemoAnimating = true;

    const from = finalDemoBoxEl.getBoundingClientRect();
    finalDemoPlaceholder = document.createElement("div");
    finalDemoPlaceholder.className = "workbench-box final-demo-placeholder";
    finalDemoPlaceholder.setAttribute("aria-hidden", "true");
    finalDemoBoxEl.before(finalDemoPlaceholder);

    finalDemoBoxEl.classList.add("is-expanded", "is-animating");
    syncNavDefault(); // pages panel opens with the 3D window (before the edges are computed)
    if (finalDemoExpandedBar) {
        finalDemoExpandedBar.hidden = false;
    }

    await animateFinalDemoEdges(edgesOf(from), fullWindowEdges(), "12px", "0px");

    finalDemoBoxEl.classList.remove("is-animating");
    finalDemoAnimating = false;
    finalDemoBackButton?.focus({ preventScroll: true });
}

// Back: the box shrinks back into its grid cell with its state intact.
async function collapseFinalDemo() {
    if (!finalDemoBoxEl?.classList.contains("is-expanded") || finalDemoAnimating) {
        return;
    }
    finalDemoAnimating = true;
    // Measuring is an expanded-view tool: back in the small grid cell the
    // selection stays visible but inert.
    finalDemoMeasureTool?.setMode("normal");
    finalDemoChrome?.closeAll();
    finalDemoBoxEl.classList.add("is-animating");

    const to = finalDemoPlaceholder?.getBoundingClientRect();
    if (to) {
        await animateFinalDemoEdges(fullWindowEdges(), edgesOf(to), "0px", "12px");
    }

    finalDemoBoxEl.classList.remove("is-expanded", "is-animating");
    finalDemoPlaceholder?.remove();
    finalDemoPlaceholder = null;
    if (finalDemoExpandedBar) {
        finalDemoExpandedBar.hidden = true;
    }
    finalDemoAnimating = false;
    syncNavDefault();
    requestAnimationFrame(() => finalDemoViewer?.resizeToCanvas());
    document.getElementById("final-demo-fullscreen")?.focus({ preventScroll: true });
}

// Close: discard everything and come back to a fresh, input-less Workbench.
// A reload is the one reset that can't leave stale state behind (search
// results, previews, generated boxes, logs, timers, 3D viewers); the form
// fields are reset first so the browser doesn't restore them on reload.
function discardWorkbenchAndReload() {
    unloadAllowed = true; // the in-app modal already decided; no second (native) prompt
    document.querySelectorAll("#page-workbench input, #page-workbench select, #page-workbench textarea").forEach(el => {
        if (el.type === "checkbox" || el.type === "radio") {
            el.checked = el.defaultChecked;
        } else if (el.type === "file") {
            el.value = "";
        } else {
            el.value = el.defaultValue;
        }
    });
    window.location.reload();
}

finalDemoBackButton?.addEventListener("click", collapseFinalDemo);

finalDemoCloseButton?.addEventListener("click", async () => {
    if (jobStore.unsaved().length) {
        const choice = await confirmUnsaved("close");
        if (choice === "cancel") {
            return;
        }
    }
    discardWorkbenchAndReload();
});


// ---- Staged grid entrance: the 8 boxes fade/slide in left-to-right,
// top-to-bottom, one subtle cascade rather than 8 independent panels. ----

const GRID_ENTER_ORDER = [
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

    if (activePageId === "page-workbench") {
        updateMiniPreviews();
        updateFinalDemoViewer();
        return;
    }

    if (activePageId !== "page-explore") {
        return;
    }

    updateFlythrough();
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