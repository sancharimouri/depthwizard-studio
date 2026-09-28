import * as THREE from "three";
import { createTerrain } from "./terrain.js";
import { createTerrainViewer } from "./viewer.js";
import { createMeasureTool } from "./measure-tool.js";
import { createExpandedChrome } from "./expanded-chrome.js";
import { createFloodSim } from "./flood-sim.js";
import { createSidebar } from "./sidebar.js";
import { createViewerHistory } from "./viewer-history.js";
import { createFlythrough } from "./flythrough.js";
import { initCollapsibleBoxes, initFacts, initTour, renderFacts, renderSource, renderTerrainStats, setInspectionHandlers, setInspectionSelected } from "./side-panels.js";
import { createSurfacePoints } from "./surface-point.js";
import { createInputView } from "./input-view.js";
import {
    STORAGE_NOTE, createJobStore, createSavedStore, exportFilename, jobLabel, jobsExport, savedRecord, unsavedCopy,
} from "./jobs.js";
import { installCloseGuard } from "./desktop-close.js";
import { apiUrl } from "./api-base.js";
import { PAGE_ROUTES, resolveHash, hashMatchesPage } from "./routes.js";
import { flatTerrainWarning } from "./flat-warning.js";

const canvas = document.getElementById("terrain-canvas");

// Explore's main viewer. Vertical-only rig offset so the structure clears
// the panels occupying the top of the page — no horizontal offset, kept
// centered on X so it lines up with controls.target's X, which is also the
// flythrough's zoom pivot; an X mismatch between the two is what drifted
// the structure sideways as the flythrough zoomed in.
const exploreViewer = createTerrainViewer(canvas, { rigOffsetY: -10 });
// Sized by its container (#app-main minus the sidebar), not the window: a
// ResizeObserver keeps the renderer and camera aspect in step while the
// sidebar's width animates, so the view never stretches.
const exploreViewerEl = canvas.parentElement;
exploreViewer.resize(exploreViewerEl.clientWidth || window.innerWidth, exploreViewerEl.clientHeight || window.innerHeight);
new ResizeObserver(() => {
    if (exploreViewerEl.clientWidth && exploreViewerEl.clientHeight) {
        exploreViewer.resize(exploreViewerEl.clientWidth, exploreViewerEl.clientHeight);
    }
}).observe(exploreViewerEl);

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
    const region = REGIONS[regionKey] ?? {};
    const surface = region.elevationSource ?? "DSM";
    const terrain = region.terrainSource ?? surface;
    const imagery = region.imagery ?? "Sentinel-2 RGB imagery";

    switch (layer) {
        case "satellite-flat":
            return `${imagery} — flat`;
        case "depth-flat":
            return "DAv2 relative depth — flat, not absolute elevation";
        case "elevation-flat":
            return `Terrain elevation from ${terrain} — flat`;
        case "dsm-3d":
            return `DAv2 relative depth draped on the ${surface} surface`;
        case "elevation-3d":
            return `Terrain elevation from ${terrain} — colour-ramped, on the ${surface} surface`;
        case "satellite-3d":
            return `${imagery} draped on the ${surface} surface`;
        default:
            return imagery;
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
const jobsListEl = document.getElementById("jobs-list");
const jobsCountEl = document.getElementById("jobs-count");
const generateNewButton = document.getElementById("generate-new-button");
const jobTabsEl = document.getElementById("job-tabs");
const jobTabInfoEl = document.getElementById("job-tab-info");
const jobsPinnedEl = document.getElementById("jobs-pinned");
const jobsPinnedEmptyEl = document.getElementById("jobs-pinned-empty");
const savedCountEl = document.getElementById("saved-count");

const jobStore = createJobStore();

// The one global sidebar (src/sidebar.js). Created before the first
// renderJobs() so the jobs-running line and the JOBS tab exist from the start.
const sidebar = createSidebar({ onNavigate: pageId => navigateToPage(pageId) });

// Image Inspection ↔ 3D surface markers (see initPointSelection). Declared up
// here because the first renderJobs() below already clears the selection.
let finalDemoSurfacePoints = null;
let inputView = null;
let unloadAllowed = false;

const CAPTION_BOX_IDS = ["depth-preview-box", "elevation-preview-box"];

// ---- Boxes 3-8: REAL generation for the job's own input (POST /api/generate/...;
// backend/generation/pipeline.py). Relative depth is DAv2-Small on this input.
// Elevation always comes from a real elevation model (the item's bundled FABDEM /
// GLO-30 / VHR pack, the DEM attached to an upload, or live Copernicus GLO-30), never
// from the image. An input without a georeference has no elevation, and the boxes say
// so instead of showing someone else's terrain. ----

const EMPTY_IMAGE = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";

function jobRegionKey(job) {
    return `job-${job.uid}`;
}

function shortSource(source) {
    return String(source ?? "").replace(/\s*\(.*$/, "") || "—";
}

function rangeText(r) {
    return r ? `${Math.round(r[0])}–${Math.round(r[1])} m` : "—";
}

// A generated job becomes a "region" of the shared viewer (stats, layer captions).
function registerJobRegion(job) {
    const m = job.gen.meta;
    const imagery = job.input.source === "library"
        ? (job.input.meta?.find(([k]) => k === "Source")?.[1] ?? "Input imagery")
        : (job.input.sourceLabel === "upload" ? "Uploaded imagery" : "Sentinel-2 (Search Online)");
    REGIONS[job.gen.key] = {
        label: job.input.title,
        hasElevation: m.has_elevation,
        imagery: shortSource(imagery),
        elevationSource: m.has_elevation ? shortSource(m.surface_source) : "No elevation model (no georeference)",
        terrainSource: m.has_elevation ? shortSource(m.terrain_source) : null,
        crsEpsg: m.crs ? m.crs.replace(/^EPSG:/, "") : "—",
        resolution: m.resolution_m ? `${m.resolution_m >= 10 ? Math.round(m.resolution_m) : m.resolution_m} m` : "—",
        inferenceTime: m.depth?.infer_s != null ? `${m.depth.infer_s}s` : "—",
    };
}

function setBoxView(boxId, view) {
    const box = document.getElementById(boxId);
    const img = box?.querySelector(".preview-image");
    const captionEl = box?.querySelector(".staged-caption");
    if (img && view.src !== undefined && img.getAttribute("src") !== view.src) {
        img.src = view.src;
    }
    if (img && view.alt !== undefined) {
        img.alt = view.alt;
    }
    if (captionEl) {
        captionEl.dataset.lines = JSON.stringify(view.lines);
    }
}

function applyDepthBox(job) {
    const gen = job?.gen;
    if (gen?.status === "ok") {
        const d = gen.depth ?? {};
        const where = String(d.device ?? "?").toUpperCase();
        setBoxView("depth-preview-box", {
            src: gen.assets.depth,
            alt: `Relative depth of ${job.input.title}`,
            lines: [
                d.fallback_used ? "Depth Anything V2 (ViT-Small) · this input · fallback host" : "Depth Anything V2 (ViT-Small) · this input",
                `${where} ${d.infer_s}s inference · ${gen.roundTripS.toFixed(1)}s generation`,
                "Relative depth (brighter = nearer), not elevation",
            ],
        });
    } else {
        setBoxView("depth-preview-box", {
            src: EMPTY_IMAGE,
            alt: "",
            lines: gen?.status === "failed" ? ["Generation failed", gen.error] : ["Awaiting generation"],
        });
    }
}

function applyElevationBox(job) {
    const gen = job?.gen;
    const m = gen?.meta;
    if (gen?.status === "ok" && m.has_elevation) {
        setBoxView("elevation-preview-box", {
            src: gen.assets.elevation,
            alt: `Terrain elevation of ${job.input.title}`,
            lines: [`Terrain: ${m.terrain_source}`, `${rangeText(m.terrain_range_m)} · ${m.crs} · ${m.resolution_m} m`],
        });
    } else if (gen?.status === "ok") {
        setBoxView("elevation-preview-box", { src: EMPTY_IMAGE, alt: "", lines: [m.note] });
    } else {
        setBoxView("elevation-preview-box", {
            src: EMPTY_IMAGE, alt: "",
            lines: gen?.status === "failed" ? ["Generation failed", gen.error] : ["Awaiting generation"],
        });
    }
}

// The 3D boxes' captions come from the job's own metadata.
function applyMiniBoxCaptions(job) {
    const m = job?.gen?.meta;
    const set = (id, lines) => {
        const captionEl = document.getElementById(id)?.querySelector(".staged-caption");
        if (captionEl) {
            captionEl.dataset.lines = JSON.stringify(lines);
        }
    };
    if (job?.gen?.status !== "ok") {
        const lines = job?.gen?.status === "failed" ? ["Generation failed", job.gen.error] : ["Awaiting generation"];
        set("dsm-3d-box", lines);
        set("metric-elevation-3d-box", lines);
    } else if (!m.has_elevation) {
        set("dsm-3d-box", ["No surface model: no georeference", "Relative depth on a flat plane"]);
        set("metric-elevation-3d-box", ["No DEM: no georeference", m.note]);
    } else {
        set("dsm-3d-box", [`${m.grid[0]}×${m.grid[1]} mesh · ${m.surface_source}`, `Surface ${rangeText(m.surface_range_m)} · relative depth draped`]);
        set("metric-elevation-3d-box", [`Terrain: ${m.terrain_source}`, `${rangeText(m.terrain_range_m)} · ${m.crs}`]);
    }
}

// Never throws: the outcome (real result or failure) is recorded on the job.
async function computeJobGeneration(job) {
    const scrollEl = document.getElementById("calc-log-scroll");
    const ref = job.input.inputRef;
    const t0 = performance.now();
    try {
        if (!ref?.id) {
            throw new Error("this input has no id to send");
        }
        const kind = ref.source === "library" ? "library" : "input";
        const response = await fetch(apiUrl(`/api/generate/${kind}/${encodeURIComponent(ref.id)}`), { method: "POST" });
        if (!response.ok) {
            let detail = `HTTP ${response.status}`;
            try {
                detail = (await response.json()).detail || detail;
            } catch {
                // not JSON
            }
            throw new Error(detail);
        }
        const g = await response.json();
        const assets = Object.fromEntries(Object.entries(g.assets).map(([k, v]) => [k, apiUrl(v)]));
        // WebGL textures load as CORS images. WebKit (the desktop shell) can reuse the cached
        // non-CORS response of the same URL shown in an <img> box and then fail the CORS check,
        // so textures get their own URLs.
        const textures = Object.fromEntries(Object.entries(assets).map(([k, v]) => [k, k === "terrain" ? v : `${v}?tex=1`]));
        job.gen = { status: "ok", key: jobRegionKey(job), assets, textures, meta: g.meta, depth: g.depth,
                    roundTripS: (performance.now() - t0) / 1000 };
        registerJobRegion(job);
        const m = g.meta;
        const d = g.depth ?? {};
        appendCalcLogLine(scrollEl, `Relative depth: DAv2-Small on ${String(d.device ?? "?").toUpperCase()}, `
            + `${(d.shape ?? []).join("×")}, ${d.infer_s}s inference (${d.encoding ?? "float32"}) via ${d.host ?? "inference host"}`
            + `${d.fallback_used ? " (fallback: primary host failed)" : ""}`, job);
        if (m.has_elevation) {
            appendCalcLogLine(scrollEl, `Terrain: ${m.terrain_source}, ${rangeText(m.terrain_range_m)}`, job);
            appendCalcLogLine(scrollEl, `Surface: ${m.surface_source}, ${rangeText(m.surface_range_m)}`, job);
            appendCalcLogLine(scrollEl, `Elevation source: ${m.how} · ${m.crs} · ${m.resolution_m} m grid`, job);
            appendCalcLogLine(scrollEl, `Mesh ${m.grid[0]}×${m.grid[1]} from the surface model · generated in ${job.gen.roundTripS.toFixed(1)}s`, job);
        } else {
            appendCalcLogLine(scrollEl, m.note, job);
        }
    } catch (error) {
        job.gen = { status: "failed", error: String(error.message ?? error).slice(0, 160) };
        appendCalcLogLine(scrollEl, `Generation failed (${job.gen.error})`, job);
    }
}
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

// The flat-terrain warning follows the on-screen job: shown from the moment its
// generation starts (library tiles know their terrain class up front; a searched
// scene only once its relief is known).
function renderFlatWarning(job) {
    const range = job?.gen?.meta?.surface_range_m;
    const warning = flatTerrainWarning(job?.input.landscape, range ? range[1] - range[0] : null);
    document.querySelectorAll(".flat-warning").forEach(el => {
        el.hidden = !warning;
        if (warning) {
            el.querySelector(".flat-warning-title").textContent = warning.title;
            el.querySelector(".flat-warning-body").textContent = warning.body;
        }
    });
}

function applyJobInput(job) {
    renderFlatWarning(job);
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
    syncSidebarDefault();
    if (finalDemoExpandedBar) {
        finalDemoExpandedBar.hidden = true;
    }
    requestAnimationFrame(() => finalDemoViewer?.resizeToCanvas());
}

let finalDemoTour = null;

// A new (or another) job opens the 3D viewer in its defaults, not in whatever
// state the previous job left it: auto-rotation on, input unlocked, camera and
// structure yaw reset, side panels back to their default open/collapsed state,
// no scenario, flythrough, tour, measurement mode or selection, popovers closed.
// (The layer and vertical exaggeration are reset by the terrain load itself.)
function resetViewerForNewJob() {
    finalDemoTour?.stop();
    finalDemoFlythrough?.stop();
    if (finalDemoFloodActive) {
        setFinalDemoFloodActive(false);
    }
    if (finalDemoEarthquakeActive) {
        setFinalDemoEarthquakeActive(false);
    }
    finalDemoMeasureTool?.setMode("normal");
    finalDemoMeasureTool?.model.clear();
    finalDemoChrome?.closeAll();
    document.querySelectorAll("#final-demo-box .xp-box").forEach(box => {
        box.xpSetOpen?.(box.hasAttribute("data-default-open"));
    });
    const controls = finalDemoViewer?.controls;
    if (controls) {
        controls.setAutoRotatePaused?.(false);
        controls.setInputLocked?.(false);
        controls.resetView?.();
        finalDemoViewer.setStructureYaw?.(0, false);
    }
    finalDemoChrome?.syncPlay?.();
    finalDemoHistory?.reset();
}

// Every box back to "Awaiting generation" so the next job's stages animate
// in one by one, exactly like the first run.
function resetGridForGeneration() {
    collapseFinalDemoInstantly();
    resetViewerForNewJob();
    applyDepthBox(null);
    applyElevationBox(null);
    applyMiniBoxCaptions(null);

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
    applyDepthBox(job);
    applyElevationBox(job);
    applyMiniBoxCaptions(job);
    showJobInMiniPreviews(job);
    showJobInFinalDemo(job);

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
        const overlay = box?.querySelector(".mini3d-generating");
        if (overlay) {
            overlay.hidden = true; // another job may be mid-generation behind it
        }
        const captionEl = box?.querySelector(".staged-caption");
        resetCaption(captionEl);
        revealStagedCaption(captionEl);
    });

    finalDemoBoxEl?.classList.remove("is-pending");
    const finalOverlay = document.getElementById("final-demo-generating");
    if (finalOverlay) {
        finalOverlay.hidden = true;
    }
    const controlsEl = document.getElementById("final-demo-controls");
    if (controlsEl) {
        controlsEl.hidden = false;
    }
    syncStudioButton();
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

// Any job can be opened at any time, including while another one generates
// (that run carries on off screen; opening it again shows where it has got to).
function selectJob(id) {
    const job = jobStore.get(id);
    if (!job) {
        return;
    }
    if (jobStore.active() !== job) {
        resetViewerForNewJob();
    }
    jobStore.setActive(id);
    if (job.status === "generating") {
        showRunningJob(job);
    } else {
        showCompletedJob(job);
    }
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
    save: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/></svg>',
    pencil: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/></svg>',
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
        if (next.status === "generating") {
            showRunningJob(next);
        } else {
            showCompletedJob(next);
        }
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

function jobItem(job, { running, active, choosing }, section = "recent") {
    const li = document.createElement("li");
    li.className = "job-item";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "job-select";
    button.dataset.jobId = job.id;
    if (job === active && !choosing) {
        button.setAttribute("aria-current", "true");
    }
    // open-in-viewer: from any page, a finished job opens in the 3D viewer window
    button.addEventListener("click", () => {
        if (activePageId !== "page-workbench") {
            setActivePage("page-workbench");
        }
        selectJob(job.id);
        if (job.status === "complete" && jobStore.active() === job) {
            expandFinalDemo();
        }
    });
    // rename: double-click the job (the only way; no pencil)
    button.title = "Double-click to rename";
    button.addEventListener("dblclick", event => {
        event.preventDefault();
        startRename(job.id, "list");
    });

    const head = document.createElement("div");
    head.className = "job-head";
    const num = document.createElement("span");
    num.className = "job-num";
    num.textContent = jobLabel(job);
    const status = document.createElement("span");
    status.className = `job-status${job.status === "generating" ? " is-running" : ""}`;
    status.textContent = job.status === "generating" ? `Generating ${job.progress}%` : "Complete";
    head.append(num, status);

    // only the (editable) job name identifies the job: no tile id, no tier line
    button.append(head);

    const foot = document.createElement("div");
    foot.className = "job-foot";
    const saved = document.createElement("span");
    saved.className = `job-saved${job.saved ? " is-saved" : ""}`;
    saved.textContent = job.saved ? "Saved" : "Unsaved";

    const actions = document.createElement("div");
    actions.className = "job-actions";

    const save = iconAction("job-save-icon", `Save ${jobLabel(job)} (downloads a JSON file and adds it to Saved)`, ICONS.save);
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
    // inline rename: the field sits above the (still clickable) job button, never inside it
    if (renaming?.id === job.id && renaming.where === "list" && section !== "pinned") {
        li.prepend(renameField(job, "job-rename-row"));
    }
    return li;
}

function renderJobs() {
    syncStudioButton();
    const hasJobs = jobStore.count() > 0;
    const savedCount = savedStore.list().length;
    sidebar.setJobsRunning(jobStore.count());
    if (savedCountEl) {
        savedCountEl.textContent = savedCount ? String(savedCount) : "";
    }
    // The JOBS tab always exists now, so it always renders (incl. its empty state).
    if (!jobsListEl) {
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
    jobsPinnedEl?.replaceChildren(...pinned.map(job => jobItem(job, ctx, "pinned")));
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
    // Enter/blur save; Esc cancels; an empty input keeps the previous name
    if (commit && renameInput.value.trim()) {
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

let lastInspectedJobId = null;

function renderJobTabs() {
    if (!jobTabsEl) {
        return;
    }
    const active = jobStore.active();

    jobTabsEl.replaceChildren(...jobStore.creationOrder().map(job => {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = "job-tab";
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-selected", String(job === active));
        tab.dataset.jobId = job.id;
        tab.title = jobLabel(job);

        const dot = document.createElement("span");
        dot.className = "job-tab-dot";
        dot.setAttribute("aria-hidden", "true");
        const label = document.createElement("span");
        label.className = "job-tab-label";
        label.textContent = jobLabel(job);
        tab.append(dot, label); // header tile: the dot and the (editable) name only
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
    if (active?.id !== lastInspectedJobId) {
        lastInspectedJobId = active?.id;
        clearPointSelection();
    }

    if (jobTabInfoEl) {
        jobTabInfoEl.hidden = true; // the header shows only the job tab (dot + name)
    }
}

// DEPTH WIZARD STUDIO: an extra way into the same 3D window that also opens
// automatically when generation completes. Usable once the 3D structure
// exists and the active job is complete.
const studioOpenButton = document.getElementById("studio-open-button");
function syncStudioButton() {
    if (!studioOpenButton) {
        return;
    }
    // the 3D view's controls are revealed once the structure has been built
    // (a DOM check: this runs before the final-demo state variables exist)
    const built = document.getElementById("final-demo-controls")?.hidden === false;
    const ready = built && jobStore.active()?.status === "complete";
    studioOpenButton.disabled = !ready;
    studioOpenButton.title = ready ? "Open the 3D structure in Depth Wizard Studio" : "Available when the 3D structure is ready";
}
studioOpenButton?.addEventListener("click", () => {
    if (!studioOpenButton.disabled) {
        expandFinalDemo();
    }
});

jobStore.onChange(renderJobs);
generateNewButton?.addEventListener("click", generateNew);

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
    // The Saved window belongs to the Workbench page; the sidebar is on every page.
    if (activePageId !== "page-workbench") {
        setActivePage("page-workbench");
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
// PAGE NAVIGATION — the global sidebar's PAGES tab (src/sidebar.js)
// ============================================================

const pages = document.querySelectorAll(".page");

// Page order (2026-09-24): 1 Workbench (default), 2 Explore, 3 Docs.
let activePageId = "page-workbench";

// The .page element that contains element `id`, or null (src/routes.js).
function pageOfElement(id) {
    const el = document.getElementById(id);
    return el?.closest(".page")?.id ?? null;
}

// Keep the URL on the page being shown, without adding history entries for
// programmatic switches (e.g. opening a job jumps to Workbench).
function syncUrlToPage(pageId) {
    if (hashMatchesPage(location.hash, pageId, pageOfElement)) {
        return;
    }
    history.replaceState(null, "", PAGE_ROUTES[pageId] || location.pathname + location.search);
}

// User navigation (sidebar): a real history entry, so Back returns to the previous page.
function navigateToPage(pageId) {
    if (location.hash !== PAGE_ROUTES[pageId]) {
        history.pushState(null, "", PAGE_ROUTES[pageId] || location.pathname + location.search);
    }
    setActivePage(pageId);
}

// URL -> page (+ scroll to an in-page anchor such as #demo-video).
function applyUrlHash() {
    const { pageId, anchorId } = resolveHash(location.hash, pageOfElement);
    setActivePage(pageId);
    if (anchorId) {
        // the page was display:none until now; scroll once it has layout
        requestAnimationFrame(() => document.getElementById(anchorId)?.scrollIntoView({ block: "start" }));
    }
}

window.addEventListener("hashchange", applyUrlHash);
window.addEventListener("popstate", applyUrlHash);

function setActivePage(pageId) {
    activePageId = pageId;
    syncUrlToPage(pageId);

    pages.forEach(page => {
        page.classList.toggle("active", page.id === pageId);
    });
    sidebar.setActivePage(pageId);

    if (pageId === "page-workbench") {
        requestAnimationFrame(initWorkbenchGrid);
    }
    syncSidebarDefault();
}

// Default sidebar tab: JOBS inside the 3D viewer window, PAGES everywhere
// else. Applied on each transition (viewer opened/closed, page changed);
// the user's own tab choice in between is left alone.
function syncSidebarDefault() {
    const viewing3d = activePageId === "page-workbench"
        && document.getElementById("final-demo-box")?.classList.contains("is-expanded");
    sidebar.setTab(viewing3d ? "jobs" : "pages");
    if (viewing3d) {
        // open by default in the 3D view, even if it was collapsed earlier this session
        sidebar.setCollapsed(false);
    }
}

// Open the page the URL names (#/docs, #/demo, #demo-video); Workbench otherwise.
// Workbench is the landing page, so its first-show init (grid entrance) runs on load too.
applyUrlHash();

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
    "verticalExaggeration = 0.02 × min(60, max(1, min(10, 1 + 4·log10(0.2 / reliefRatio)), 0.15 × 100 / (0.02 × relief m)))",
    "Darjeeling relief ratio 0.192 → exaggeration 1.07x (steep: floor not needed)",
    "Kolkata relief 42.8 m → exaggeration 17.5x (low-relief floor: 15% of the tile)",
    "Bardhaman relief 31.7 m → exaggeration 23.7x (low-relief floor)",
    "Sundarbans relief 10.1 m → exaggeration 60.00x (ceiling clamp)",
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
        opening.push("Generating from this input's own data: relative depth, its elevation model and the 3D terrain");
        opening.push("Handing off to the generation pipeline…");
        // the real steps (sources, ranges, grid, timings) are appended by computeJobGeneration
        return opening;
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
// no OrbitControls attached (no drag/zoom). They show the generated job's own
// terrain (one fetch per job, shared by both boxes); nothing before that. ----

const miniPreviews = [];
const miniAssetCache = new Map();

function loadMiniPreviewAssets(job) {
    const gen = job?.gen;
    if (gen?.status !== "ok") {
        return Promise.resolve(null);
    }
    if (!miniAssetCache.has(gen.key)) {
        miniAssetCache.set(gen.key, (async () => {
            const terrainData = await (await fetch(gen.textures.terrain)).json();
            const textureLoader = new THREE.TextureLoader();
            const [satelliteTexture, depthTexture, elevationTexture] = await Promise.all([
                textureLoader.loadAsync(gen.textures.satellite),
                textureLoader.loadAsync(gen.textures.depth),
                textureLoader.loadAsync(gen.textures.elevation),
            ]);
            return { key: gen.key, terrainData, satelliteTexture, depthTexture, elevationTexture };
        })());
    }
    return miniAssetCache.get(gen.key);
}

async function showJobInMiniPreviews(job) {
    let assets = null;
    try {
        assets = await loadMiniPreviewAssets(job);
    } catch (error) {
        miniAssetCache.delete(job?.gen?.key);
        appendCalcLogLine(document.getElementById("calc-log-scroll"),
            `3D preview textures failed to load (${error?.message ?? error?.type ?? error})`, job);
    }
    // the user switched jobs while these loaded: the on-screen job owns the previews
    if (job && jobStore.active() !== job) {
        return;
    }
    miniPreviews.forEach(preview => {
        preview.terrain?.mesh?.geometry?.dispose();
        preview.group.clear();
        preview.terrain = null;
        preview.ready = false;
        if (!assets) {
            return;
        }
        const { key, terrainData, satelliteTexture, depthTexture, elevationTexture } = assets;
        preview.terrain = createTerrain(preview.group, key, terrainData, satelliteTexture, depthTexture, elevationTexture);
        preview.terrain.setLayer(preview.layer);
        preview.ready = true;
        preview.lastWidth = 0;
        resizeMiniPreview(preview);
    });
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
    // The framing (0,58,143 → 0,12,0) was tuned for wide boxes. In a
    // narrower pane (the processing page's black screen) back the camera
    // off along the same line so the terrain's full width stays in view.
    // distance at which the terrain's half-width (~56 units, + margin for the
    // near edge's perspective) fits the horizontal FOV, vs the tuned 150
    const tanHalfH = Math.tan(THREE.MathUtils.degToRad(preview.camera.fov) / 2) * preview.camera.aspect;
    const fit = Math.max(1, (64 / tanHalfH) / 150);
    preview.camera.position.set(0, 12 + 46 * fit, 143 * fit);
    preview.camera.lookAt(0, 12, 0);
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
        layer,
        terrain: null,
        ready: false,
        lastWidth: 0,
        lastHeight: 0,
    };

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
let finalDemoHistory = null;

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
        elevationElement.textContent = region?.hasElevation === false
            ? "—"
            : `${Math.round(terrain.elevationMin)}–${Math.round(terrain.elevationMax)} m`;
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
    finalDemoSurfacePoints?.refresh();
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
    if (active) {
        finalDemoLandslideInfo = false;
    }

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

// Landslide: a visible, clickable placeholder with a "Coming soon" state.
// There is no landslide logic, model or data behind it.
let finalDemoLandslideInfo = false;
SCENARIO_NOTES.landslide = "Landslide — coming soon. There is no landslide susceptibility model or data in this project yet, so this option does nothing for now.";

function syncScenarioNote() {
    const note = document.getElementById("final-demo-scenario-note");
    if (!note) {
        return;
    }
    const key = finalDemoEarthquakeActive ? "earthquake" : finalDemoFloodActive ? "flood" : finalDemoLandslideInfo ? "landslide" : null;
    document.getElementById("final-demo-landslide-button")?.classList.toggle("is-soon-open", key === "landslide");
    document.getElementById("final-demo-landslide-button")?.setAttribute("aria-pressed", String(key === "landslide"));
    note.hidden = !key;
    note.textContent = key ? SCENARIO_NOTES[key] : "";
    note.classList.toggle("is-placeholder", key === "earthquake" || key === "landslide");
}

function setFinalDemoEarthquakeActive(active) {
    finalDemoEarthquakeActive = active;
    if (active) {
        finalDemoLandslideInfo = false;
    }
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
    finalDemoFlythrough?.stop();
}

// The Studio shows the generated job's own terrain and textures (never a demo region).
async function showJobInFinalDemo(job) {
    if (!finalDemoViewer || job?.gen?.status !== "ok") {
        return null;
    }
    let terrain;
    try {
        terrain = await loadFinalDemoRegion(job.gen.key, job.gen.textures);
    } catch (error) {
        appendCalcLogLine(document.getElementById("calc-log-scroll"),
            `Studio terrain failed to load (${error?.message ?? error?.type ?? error})`, job);
        return null;
    }
    // the user switched jobs while this loaded: the on-screen job owns the Studio
    const active = jobStore.active();
    if (active && active !== job) {
        return active.gen?.status === "ok" && active.status === "complete" ? showJobInFinalDemo(active) : null;
    }
    finalDemoHistory?.reset();
    renderSource(job);
    return terrain;
}

async function loadFinalDemoRegion(regionKey, assets = null) {
    const { terrain, terrainData } = await finalDemoViewer.loadRegion(regionKey, assets);

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
    finalDemoFlythrough?.stop();

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
// GLOBAL THEME (dark / bright): one sun/moon button for every page, the
// sidebar and the 3D viewer. It sets [data-theme] on <html>; every colour in
// styles.css reads the custom properties :root[data-theme="light"]
// overrides. The only parts CSS can't reach are the two THREE.js scene
// backgrounds (UI chrome, not imagery), set here. Remembered per browser.
// ============================================================

const WORKBENCH_THEME_BG_HEX = {
    dark: 0x110f0e,
    light: 0xf4e9d2,
};
const THEME_KEY = "dw2.theme";

let workbenchTheme = "dark";

const THEME_ICONS = {
    sun: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8 6 18M18 6l1.8-1.8"/></svg>',
    moon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.5 13.2A8.5 8.5 0 1 1 10.8 3.5a6.8 6.8 0 0 0 9.7 9.7z"/></svg>',
};

function applyWorkbenchTheme(theme) {
    workbenchTheme = theme;
    document.documentElement.dataset.theme = theme;

    // the icon is the mode you'd switch TO: sun while dark, moon while bright
    const toggle = document.getElementById("global-theme-toggle");
    if (toggle) {
        const dark = theme === "dark";
        toggle.innerHTML = dark ? THEME_ICONS.sun : THEME_ICONS.moon;
        const name = dark ? "Switch to bright mode" : "Switch to dark mode";
        toggle.setAttribute("aria-label", name);
        toggle.dataset.tip = name;
    }

    exploreViewer?.setBackground(WORKBENCH_THEME_BG_HEX[theme]);
    finalDemoViewer?.setBackground(finalDemoCurrentLayer === "wireframe-3d" ? 0x000000 : WORKBENCH_THEME_BG_HEX[theme]);
    try {
        localStorage.setItem(THEME_KEY, theme);
    } catch {
        // storage blocked: the choice just isn't remembered
    }
}

document.getElementById("global-theme-toggle")?.addEventListener("click", () => {
    applyWorkbenchTheme(workbenchTheme === "dark" ? "light" : "dark");
});

try {
    applyWorkbenchTheme(localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark");
} catch {
    applyWorkbenchTheme("dark");
}

// Creates the viewer, wires every control, and loads Darjeeling. Returns
// the loadFinalDemoRegion() promise so callers can await real asset load
// alongside the box's minimum "generating" duration.
function initFinalDemoViewer(job) {
    if (finalDemoInitialized) {
        return showJobInFinalDemo(job);
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
    // Renderer + camera aspect follow the canvas's container as the sidebar
    // pushes it (the animate loop's resizeToCanvas() also catches this).
    new ResizeObserver(() => finalDemoViewer.resizeToCanvas()).observe(canvas.parentElement);

    // Measurement tool (expanded view only; deliverables-audit item 6).
    finalDemoMeasureTool = createMeasureTool({
        viewer: finalDemoViewer,
        box: document.getElementById("final-demo-box"),
        canvas,
        // measure mode: a click on the Image Inspection's selected point starts the measurement there
        snapTarget: {
            get: () => finalDemoSurfacePoints?.selectedGrid() ?? null,
            onUsed: () => clearPointSelection(),
        },
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
        // flythrough moved out of the edit toolbar (Reset View took its place)
        nav: createNavActions(),
        getTheme: () => workbenchTheme,
        setTheme: applyWorkbenchTheme,
    });
    if (import.meta.env?.DEV) {
        window.__dwMeasureTool = finalDemoMeasureTool; // dev-only hooks for the headless tests
        window.__dwChrome = finalDemoChrome;
    }
    // Side-panel boxes (collapsible) + the Explore Options product tour.
    initCollapsibleBoxes(document.getElementById("final-demo-box"));
    finalDemoTour = initTour(document.getElementById("final-demo-box"));
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
    initViewerHistory(canvas);
    initPointSelection(canvas);
    if (import.meta.env?.DEV) {
        window.__dwHistory = finalDemoHistory;
        window.__dwSurfacePoints = finalDemoSurfacePoints;
        Object.defineProperty(window, "__dwFly", { configurable: true, get: () => finalDemoFlythrough });
        window.__dwFloodSim = finalDemoFloodSim; // dev-only, headless checks
        window.__dwTerrain = () => finalDemoCurrentTerrain;
        window.__dwCamera = () => finalDemoViewer.camera;
        window.__dwTHREE = THREE;
        window.__dwViewer = finalDemoViewer;
    }
    renderSource(jobStore.active());
    renderFacts(jobStore.active());

    // Fly-through box (src/flythrough.js). Obstacles = what covers the canvas:
    // the two panel columns, the job tabs at the top, the toolbars at the bottom.
    finalDemoFlythrough = createFlythrough({
        viewer: finalDemoViewer,
        canvas,
        getTerrain: () => finalDemoCurrentTerrain,
        getObstacles: () => {
            const box = document.getElementById("final-demo-box");
            const rectOf = sel => {
                const n = box.querySelector(sel);
                const r = n?.getBoundingClientRect();
                return r && r.width > 0 && r.height > 0 ? r : null;
            };
            const left = rectOf(".final-demo-left-rail");
            const right = rectOf(".final-demo-right-rail");
            const top = rectOf("#job-tab-info") ?? rectOf(".job-tabs-wrap");
            const bottom = rectOf(".xv-navbar") ?? rectOf(".xv-toolbar");
            return {
                left: left ? left.right + 12 : null,
                right: right ? right.left - 12 : null,
                top: top ? top.bottom + 12 : null,
                bottom: bottom ? bottom.top - 12 : null,
            };
        },
        onReset: () => createNavActions().resetView(),
        els: {
            toggle: document.getElementById("xp-fly-toggle"),
            again: document.getElementById("xp-fly-again"),
            reset: document.getElementById("xp-fly-reset"),
        },
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

    document.getElementById("final-demo-landslide-button")?.addEventListener("click", () => {
        finalDemoLandslideInfo = !finalDemoLandslideInfo;
        syncScenarioNote();
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

    return showJobInFinalDemo(job);
}

// ---- Image Inspection ↔ 3D surface (src/surface-point.js) ----
// Hovering the magnifier shows a yellow disc at the matching surface spot;
// clicking selects that point (one at a time, persistent marker). Selecting
// never starts the measure tool. Measure ON: clicking exactly on the marker
// starts a measurement there (measure-tool snapTarget). Measure OFF: a click
// anywhere else on the 3D canvas clears it; clicking another pixel in the
// image replaces it.
function selectPoint(u, v) {
    finalDemoSurfacePoints?.select(u, v);
    setInspectionSelected({ u, v });
}

function clearPointSelection() {
    finalDemoSurfacePoints?.clear();
    setInspectionSelected(null);
}

function initPointSelection(canvas) {
    finalDemoSurfacePoints = createSurfacePoints({ getTerrain: () => finalDemoCurrentTerrain });
    setInspectionHandlers({
        onHover: (u, v) => finalDemoSurfacePoints.hover(u, v),
        onLeave: () => finalDemoSurfacePoints.clearHover(),
        onPick: (u, v) => selectPoint(u, v),
    });

    // Is the pointer over the selected marker? Same 14 px radius as the
    // click that keeps it (measure off) or snaps a measurement to it (measure on).
    function overMarker(event) {
        const grid = finalDemoSurfacePoints.selectedGrid();
        if (!grid) {
            return false;
        }
        const sp = finalDemoMeasureTool.project(grid.x, grid.y);
        const rect = canvas.getBoundingClientRect();
        return Boolean(sp) && Math.hypot(sp.sx - (event.clientX - rect.left), sp.sy - (event.clientY - rect.top)) <= 14;
    }

    // Hover highlight, like a measure point's: the marker turns #ffd23f with
    // a stronger glow and the cursor becomes a pointer.
    let hoverFrame = 0;
    let lastMove = null;
    canvas.addEventListener("pointermove", event => {
        lastMove = event;
        if (hoverFrame) {
            return;
        }
        hoverFrame = requestAnimationFrame(() => {
            hoverFrame = 0;
            const on = overMarker(lastMove);
            finalDemoSurfacePoints.setHighlight(on);
            canvas.classList.toggle("is-over-selected-point", on);
        });
    });
    canvas.addEventListener("pointerleave", () => {
        finalDemoSurfacePoints.setHighlight(false);
        canvas.classList.remove("is-over-selected-point");
    });

    // a plain click (not an orbit drag) on the canvas, measure tool off
    let down = null;
    canvas.addEventListener("pointerdown", event => {
        down = event.button === 0 ? { x: event.clientX, y: event.clientY } : null;
    });
    canvas.addEventListener("pointerup", event => {
        const start = down;
        down = null;
        if (!start || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) {
            return;
        }
        if (!finalDemoSurfacePoints.selectedGrid() || finalDemoMeasureTool.model.mode !== "normal") {
            return;
        }
        if (!overMarker(event)) {
            clearPointSelection();
        }
    });
}

// ---- Navigation bar actions (expanded-chrome.js draws the bar) ----
// Every camera/structure change runs through recordCameraChange, so it is one
// undo/redo step. The ground grid is never moved: Up/Down/Top/Side orbit the
// camera; the rotate buttons turn the structure about the world vertical axis.
let recordCameraChange = action => action(); // replaced once the history exists
const NAV_ORBIT_STEP = THREE.MathUtils.degToRad(15);
const NAV_TOP_POLAR = 0.001;
const NAV_SIDE_POLAR = THREE.MathUtils.degToRad(80);

function createNavActions() {
    const c = () => finalDemoViewer.controls;
    return {
        // where the camera is heading (end value), not its mid-animation angle
        isTopView: () => (c()._sphericalEnd?.phi ?? c().polarAngle) < THREE.MathUtils.degToRad(12),
        toggleTopSide() {
            const toTop = !this.isTopView();
            recordCameraChange(() => c().rotatePolarTo(toTop ? NAV_TOP_POLAR : NAV_SIDE_POLAR, true));
        },
        // dir -1 = Up (camera higher), +1 = Down; clamped by the polar limits
        orbitStep(dir) {
            const ctl = c();
            const next = THREE.MathUtils.clamp(ctl.polarAngle + dir * NAV_ORBIT_STEP, Math.max(ctl.minPolarAngle, NAV_TOP_POLAR), ctl.maxPolarAngle);
            recordCameraChange(() => ctl.rotatePolarTo(next, true));
        },
        // + = anticlockwise seen from above
        rotateBy(degrees) {
            recordCameraChange(() => finalDemoViewer.setStructureYaw(
                finalDemoViewer.getStructureYaw() + THREE.MathUtils.degToRad(degrees), true));
        },
        resetView() {
            recordCameraChange(() => {
                // the first frame after creation, immediately: no animated unwinding
                // of the rotation accumulated this session
                c().resetView();
                finalDemoViewer.setStructureYaw(0, false);
            });
        },
        isPaused: () => c().isAutoRotatePaused?.() ?? false,
        togglePlay: () => c().setAutoRotatePaused(!(c().isAutoRotatePaused?.() ?? false)),
        isLocked: () => c().isInputLocked?.() ?? false,
        toggleLock: () => c().setInputLocked(!(c().isInputLocked?.() ?? false)),
    };
}

// ---- Viewer-wide undo / redo (src/viewer-history.js) ----
// State entries cover measurements/selections, notes, the view layer,
// vertical exaggeration and the scenario overlay; camera entries cover one
// user rotate/zoom gesture each. Committed after each user action in the box.
function initViewerHistory(canvas) {
    const box = document.getElementById("final-demo-box");
    const undoBtn = document.getElementById("final-demo-undo");
    const redoBtn = document.getElementById("final-demo-redo");
    const controlsRef = finalDemoViewer.controls;
    const v1 = new THREE.Vector3();
    const v2 = new THREE.Vector3();
    // end values (where damping is heading), not the mid-glide camera
    const cam = () => {
        controlsRef.getPosition(v1, true);
        controlsRef.getTarget(v2, true);
        return { p: v1.toArray(), t: v2.toArray() };
    };
    // Nav-bar actions record the structure's yaw too (drag gestures don't, so
    // undoing a drag never rewinds rotation that auto-rotation added meanwhile).
    recordCameraChange = action => {
        const before = { ...cam(), yaw: finalDemoViewer.getStructureYaw() };
        action();
        const after = { ...cam(), yaw: finalDemoViewer.getStructureYaw() };
        finalDemoHistory.commitCamera(before, after);
    };

    finalDemoHistory = createViewerHistory({
        capture: () => ({
            // selection only: the model's internal "which sub-mode made it" (owner)
            // changes on a plain mode switch and must not become an undo step
            measure: (({ chains, active }) => ({ chains, active }))(finalDemoMeasureTool.model.snapshot()),
            notes: finalDemoChrome.notes.map(n => ({
                id: n.id, x: n.x, y: n.y, text: n.text, visible: n.visible, createdAt: +n.createdAt,
            })),
            layer: finalDemoCurrentLayer,
            vex: Number((finalDemoCurrentTerrain?.displayExaggeration?.() ?? 1).toFixed(4)),
            scenario: finalDemoEarthquakeActive ? "earthquake" : finalDemoFloodActive ? "flood" : null,
        }),
        apply: st => {
            finalDemoMeasureTool.model.restore(st.measure);
            const notes = finalDemoChrome.notes;
            notes.length = 0;
            st.notes.forEach(n => notes.push({ ...n, createdAt: new Date(n.createdAt) }));
            if (st.layer !== finalDemoCurrentLayer) {
                activateFinalDemoLayer(st.layer);
            }
            if (finalDemoFloodActive !== (st.scenario === "flood")) {
                setFinalDemoFloodActive(st.scenario === "flood");
            }
            if (finalDemoEarthquakeActive !== (st.scenario === "earthquake")) {
                setFinalDemoEarthquakeActive(st.scenario === "earthquake");
            }
            if (finalDemoCurrentTerrain?.setDisplayExaggeration) {
                finalDemoCurrentTerrain.setDisplayExaggeration(st.vex);
                syncExaggerationSlider(finalDemoCurrentTerrain);
            }
            finalDemoMeasureTool.invalidate();
        },
        applyCamera: c => {
            controlsRef.setLookAt(...c.p, ...c.t, true);
            if (c.yaw !== undefined) {
                finalDemoViewer.setStructureYaw(c.yaw, true, { shortest: true });
            }
            finalDemoChrome?.syncPlay?.(); // keep the Top/Side icon honest
        },
        sameCamera: (a, b) => a.p.every((x, i) => Math.abs(x - b.p[i]) < 1e-3)
            && a.t.every((x, i) => Math.abs(x - b.t[i]) < 1e-3)
            && Math.abs((a.yaw ?? 0) - (b.yaw ?? 0)) < 1e-4,
        onChange: () => {
            undoBtn.disabled = !finalDemoHistory.canUndo;
            redoBtn.disabled = !finalDemoHistory.canRedo;
        },
    });

    // State: commit once after each user action inside the viewer (after its
    // handlers ran), so a whole drag / menu action becomes one entry.
    let commitTimer = 0;
    const scheduleCommit = () => {
        clearTimeout(commitTimer);
        commitTimer = setTimeout(() => finalDemoHistory.commit(), 0);
    };
    ["pointerup", "click", "keyup", "change"].forEach(type => box.addEventListener(type, event => {
        if (event.target.closest?.("#final-demo-undo, #final-demo-redo")) {
            return;
        }
        scheduleCommit();
    }, true));

    // Camera: one entry per drag / touch gesture…
    let dragBefore = null;
    let wheelBefore = null;
    let wheelTimer = 0;
    controlsRef.addEventListener("controlstart", () => {
        dragBefore = cam();
    });
    controlsRef.addEventListener("controlend", () => {
        const before = dragBefore;
        dragBefore = null;
        // a wheel gesture is recorded by the wheel handler below instead
        setTimeout(() => {
            if (before && !wheelBefore) {
                finalDemoHistory.commitCamera(before, cam());
            }
        }, 0);
    });
    // …and one per scroll / pinch burst (settles 350 ms after the last event).
    canvas.addEventListener("wheel", () => {
        if (!wheelBefore) {
            wheelBefore = dragBefore ?? cam();
        }
        clearTimeout(wheelTimer);
        wheelTimer = setTimeout(() => {
            finalDemoHistory.commitCamera(wheelBefore, cam());
            wheelBefore = null;
        }, 350);
    }, { passive: true });

    // The change itself is the feedback; only an empty stack gets a toast
    // (the buttons are disabled then, so that's keyboard-only).
    const undo = () => {
        if (!finalDemoHistory.undo()) {
            finalDemoMeasureTool.showToast?.("Nothing to undo.", "info");
        }
    };
    const redo = () => {
        if (!finalDemoHistory.redo()) {
            finalDemoMeasureTool.showToast?.("Nothing to redo.", "info");
        }
    };
    undoBtn.addEventListener("click", undo);
    redoBtn.addEventListener("click", redo);

    document.addEventListener("keydown", event => {
        if (!box.classList.contains("is-expanded") || !(event.metaKey || event.ctrlKey)) {
            return;
        }
        const key = event.key.toLowerCase();
        const isUndo = key === "z" && !event.shiftKey;
        const isRedo = (key === "z" && event.shiftKey) || (key === "y" && event.ctrlKey && !event.metaKey);
        if (!isUndo && !isRedo) {
            return;
        }
        const t = event.target;
        if (t instanceof HTMLInputElement && t.type === "text" || t instanceof HTMLTextAreaElement || t?.isContentEditable) {
            return; // text fields keep their own undo
        }
        if (document.querySelector(".confirm-modal:not([hidden])")) {
            return;
        }
        event.preventDefault();
        finalDemoHistory.commit(); // fold any unrecorded change in first
        if (isUndo) {
            undo();
        } else {
            redo();
        }
    });
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
// being calculated." Resolves once durationMs has elapsed. It never touches
// the DOM itself: onTick(percent, stepText) decides whether (and where) to
// paint, because the job it belongs to may not be the one on screen.
async function animateGenerating({ steps, durationMs, onTick }) {
    const tickMs = 120;
    const stepIntervalMs = durationMs / Math.max(1, steps.length);

    // Tracks real elapsed time (performance.now()) rather than assuming
    // each tick took exactly tickMs — a backgrounded/throttled tab can
    // delay individual setTimeout ticks well past 120ms, and accumulating
    // a fixed increment per tick would make the whole animation run far
    // longer in wall-clock time than durationMs in that case.
    const startTime = performance.now();
    let elapsed = 0;

    onTick(0, steps[0] ?? "");

    while (elapsed < durationMs) {
        await sleep(tickMs);
        elapsed = performance.now() - startTime;
        const step = steps[Math.min(steps.length - 1, Math.floor(elapsed / stepIntervalMs))] ?? "";
        onTick(Math.min(100, Math.round((elapsed / durationMs) * 100)), step);
    }
}

// ---- Box 2 (Preview) reveals at selection time now (see
// revealScenePreview() above), not as part of the generation sequence. ----

// ---- The staged boxes, in run order. A generating job keeps its own
// progress (job.run: current stage, percent, status line), and the shared grid
// only paints the job that is on screen (jobStore.active()). So while one job
// generates, any other job can be opened from the jobs panel, and coming back
// repaints the running job exactly where it has got to. ----

const GEN_STAGES = [
    {
        id: "depth", kind: "caption", boxId: "depth-preview-box",
        steps: [
            "Sending this input's preview to the inference host…",
            "Running Depth Anything V2 (ViT-Small)…",
            "Decoding 518×518 relative depth…",
        ],
    },
    {
        id: "elevation", kind: "caption", boxId: "elevation-preview-box",
        steps: [
            "Finding this input's elevation model…",
            "Reprojecting the DEM onto the input's footprint…",
            "Colour-ramping terrain elevation…",
        ],
    },
    {
        id: "dsm", kind: "mini", boxId: "dsm-3d-box", canvasId: "dsm-3d-canvas", layer: "dsm-3d",
        steps: [
            "Loading the surface model (DSM)…",
            "Building the terrain mesh…",
            "Draping relative depth…",
        ],
    },
    {
        id: "metric", kind: "mini", boxId: "metric-elevation-3d-box", canvasId: "metric-elevation-3d-canvas", layer: "elevation-3d",
        steps: [
            "Loading the terrain DEM…",
            "Colour-ramping by elevation…",
            "Scaling vertical relief…",
        ],
    },
    {
        id: "final", kind: "final", boxId: "final-demo-box",
        steps: [
            "Compiling interactive viewer…",
            "Wiring OrbitControls + layer shaders…",
            "Loading RUN RECONSTRUCTION, FLYTHROUGH, DANGER ZONES…",
        ],
    },
];

const miniPreviewCanvases = new Set();

function isOnScreen(job) {
    return jobStore.active() === job;
}

// One renderer per canvas: later jobs re-run the stage over the same view.
function ensureMiniPreview(stage) {
    const canvas = document.getElementById(stage.canvasId);
    if (canvas && !miniPreviewCanvases.has(canvas)) {
        miniPreviewCanvases.add(canvas);
        createMiniPreview(canvas, stage.layer);
    }
}

// The element holding a stage's generating readout (percent + status line).
function stageOverlay(stage) {
    if (stage.kind === "caption") {
        const emptyEl = document.getElementById(stage.boxId)?.querySelector(".preview-empty");
        if (emptyEl && !emptyEl.querySelector(".generating-overlay")) {
            emptyEl.innerHTML = generatingOverlayMarkup();
        }
        return emptyEl;
    }
    if (stage.kind === "mini") {
        return document.getElementById(stage.boxId)?.querySelector(".mini3d-generating");
    }
    return document.getElementById("final-demo-generating");
}

function paintStageRunning(stage, percent, text) {
    const overlay = stageOverlay(stage);
    if (!overlay) {
        return;
    }
    overlay.hidden = false;
    const percentEl = overlay.querySelector(".generating-percent");
    const statusEl = overlay.querySelector(".generating-status");
    if (percentEl) {
        percentEl.textContent = `${percent}%`;
    }
    if (statusEl && statusEl.textContent !== text) {
        statusEl.textContent = text;
    }
}

// A finished stage, painted for `job`. The 3D content itself (mini previews,
// Studio terrain) is loaded by the caller, which knows whether it's needed.
function paintStageDone(stage, job) {
    const box = document.getElementById(stage.boxId);
    if (stage.kind === "caption") {
        (stage.id === "depth" ? applyDepthBox : applyElevationBox)(job);
        const emptyEl = box?.querySelector(".preview-empty");
        const contentEl = box?.querySelector(".preview-content");
        if (emptyEl) {
            emptyEl.hidden = true;
        }
        if (contentEl) {
            contentEl.hidden = false;
        }
        revealStagedCaption(box?.querySelector(".staged-caption"));
        return;
    }
    if (stage.kind === "mini") {
        const overlay = box?.querySelector(".mini3d-generating");
        if (overlay) {
            overlay.hidden = true;
        }
        box?.classList.remove("is-pending");
        revealStagedCaption(box?.querySelector(".staged-caption"));
        return;
    }
    const overlay = document.getElementById("final-demo-generating");
    if (overlay) {
        overlay.hidden = true;
    }
    box?.classList.remove("is-pending");
    ["final-demo-controls", "final-demo-fullscreen"].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.hidden = false;
        }
    });
    syncStudioButton();
}

// Repaints the grid for a job that is still generating (the user came back to
// it from another job): finished stages done, the current one mid-readout,
// the rest awaiting generation, and its own calculation log so far.
function showRunningJob(job) {
    applyJobInput(job);
    resetGridForGeneration();
    const scrollEl = document.getElementById("calc-log-scroll");
    job.log.forEach(line => renderCalcLogLine(scrollEl, line.t, line.text));
    if (startGenerationButton) {
        startGenerationButton.disabled = true;
        startGenerationButton.textContent = "▶ GENERATING…";
    }
    const run = job.run ?? { stage: 0, percent: 0, text: GEN_STAGES[0].steps[0] };
    if (run.stage >= 2) {
        applyMiniBoxCaptions(job);
    }
    GEN_STAGES.forEach((stage, index) => {
        if (index < run.stage) {
            paintStageDone(stage, job);
        } else if (index === run.stage) {
            paintStageRunning(stage, run.percent, run.text);
        }
    });
    if (run.stage > 2) {
        showJobInMiniPreviews(job);
    }
    if (run.stage >= 4) {
        initFinalDemoViewer(job);
    }
    showGrid();
    requestAnimationFrame(() => finalDemoViewer?.resizeToCanvas());
}

// One stage of one job: the readout runs for at least BOX_GENERATE_MS and until
// the stage's real work resolves, painting only while the job is on screen.
async function runStage(job, index, work) {
    const stage = GEN_STAGES[index];
    job.run = { stage: index, percent: 0, text: stage.steps[0] };
    const [, result] = await Promise.all([
        animateGenerating({
            steps: stage.steps,
            durationMs: BOX_GENERATE_MS,
            onTick: (percent, text) => {
                job.run = { stage: index, percent, text };
                if (isOnScreen(job)) {
                    paintStageRunning(stage, percent, text);
                }
            },
        }),
        work,
    ]);
    return result;
}

// One job's run through the staged boxes. Only one job generates at a time,
// but the others stay viewable meanwhile (see GEN_STAGES above).
async function runGenerationSequence(job) {
    if (startGenerationButton) {
        startGenerationButton.disabled = true;
        startGenerationButton.textContent = "▶ GENERATING…";
    }

    // Starts filling immediately and keeps appending in parallel with
    // whichever box below is currently generating.
    runCalcLog(TOTAL_PIPELINE_MS, job);
    const finishStage = (index, progress) => {
        job.run = { stage: index + 1, percent: 0, text: GEN_STAGES[index + 1]?.steps[0] ?? "" };
        if (isOnScreen(job)) {
            paintStageDone(GEN_STAGES[index], job);
            renderFlatWarning(job); // a searched scene's relief is known from here on
        }
        jobStore.update(job.id, { progress });
    };
    // One real generation request for this input; every box below waits on it.
    const genWork = computeJobGeneration(job);

    await runStage(job, 0, genWork);
    finishStage(0, 20);

    await runStage(job, 1, genWork);
    finishStage(1, 40);
    if (isOnScreen(job)) {
        applyMiniBoxCaptions(job);
    }

    for (const index of [2, 3]) {
        ensureMiniPreview(GEN_STAGES[index]);
        await runStage(job, index, loadMiniPreviewAssets(job).catch(() => null));
        if (isOnScreen(job)) {
            await showJobInMiniPreviews(job);
        }
        finishStage(index, index === 2 ? 60 : 80);
    }

    // The Studio viewer is shared: load this job's terrain into it only while
    // the job is on screen (showRunningJob / showCompletedJob load it otherwise).
    await runStage(job, 4, isOnScreen(job) ? initFinalDemoViewer(job) : null);
    job.run = null;
    const onScreen = isOnScreen(job);
    if (onScreen) {
        paintStageDone(GEN_STAGES[4], job);
    }
    jobStore.update(job.id, { status: "complete", progress: 100 });

    if (onScreen) {
        if (startGenerationButton) {
            startGenerationButton.textContent = "✓ GENERATION COMPLETE";
        }
        // Whole pipeline done: pop the final 3D view out to fill the window.
        expandFinalDemo();
    }
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
// A viewport rect as inset edges of #app-main (the fixed box's containing block).
function edgesOf(rect) {
    const main = document.getElementById("app-main").getBoundingClientRect();
    return {
        top: `${rect.top - main.top}px`,
        right: `${main.right - rect.right}px`,
        bottom: `${main.bottom - rect.bottom}px`,
        left: `${rect.left - main.left}px`,
    };
}

// The expanded box is position: fixed inside #app-main, whose contain: layout
// makes it the containing block, so "full window" is inset 0 of the main area.
function fullWindowEdges() {
    return { top: "0px", right: "0px", bottom: "0px", left: "0px" };
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
    syncSidebarDefault(); // JOBS tab is the default inside the 3D viewer
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
    syncSidebarDefault();
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

// Reading order of the processing page: the top row (preview, relative
// depth, elevation), the second row (logs, DSM, DEM), then the 3D result and
// the Studio button.
const GRID_ENTER_ORDER = [
    "preview-box",
    "depth-preview-box",
    "elevation-preview-box",
    "calc-logs-box",
    "dsm-3d-box",
    "metric-elevation-3d-box",
    "final-demo-box",
    "wb-studio",
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