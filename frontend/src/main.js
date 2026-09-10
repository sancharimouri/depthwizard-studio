import * as THREE from "three";
import { createTerrain } from "./terrain.js";
import { createControls } from "./controls.js";

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

function setLayerDescription(layer) {
    const description = document.getElementById("layer-description");

    if (!description) {
        return;
    }

    if (layer === "depth") {
        description.textContent = "DAv2 relative depth — not absolute elevation";
    } else if (layer === "elevation") {
        description.textContent = "Metric elevation from OpenTopography DSM";
    } else {
        description.textContent = "Sentinel-2 RGB imagery";
    }
}


// ============================================================
// REGION RAIL
// ============================================================

const REGION_LABELS = {
    darjeeling: "Darjeeling",
    kolkata: "Kolkata",
    bardhaman: "Bardhaman",
    sundarbans: "Sundarbans",
};

document.querySelectorAll(".region-card").forEach(card => {
    card.addEventListener("click", () => {
        const region = card.dataset.region;
        const note = document.getElementById("region-note");

        if (!note) {
            return;
        }

        if (card.classList.contains("locked")) {
            note.textContent = `${REGION_LABELS[region]} — imagery indexed, full pipeline lands in Session 2.`;
            setTimeout(() => {
                note.textContent = "";
            }, 3500);
            return;
        }

        note.textContent = "";
    });
});


// ============================================================
// ACTIVITY / PIPELINE LOG
// ============================================================

const ACTIVITY_LOG = [
    { tag: "TERRAIN", time: "now", text: "OpenTopography DSM loaded — 557–2478 m range, Darjeeling" },
    { tag: "MESH", time: "2s ago", text: "Terrain mesh built — 361 × 325 vertices at 10 m/px" },
    { tag: "VALIDATION", time: "1m ago", warn: true, text: "Spatial-trend correlation check — Pearson r +0.60 → −0.41 after detrending. Elevation pipeline frozen." },
    { tag: "DEPTH", time: "1m ago", text: "DAv2 (Depth-Anything-V2-Large) relative-depth inference — 1.17s on MPS" },
    { tag: "VALIDATION", time: "2m ago", warn: true, text: "RDAH-Net zero-shot check (Swiss/HK weights) — checkerboard artifacts, rejected" },
    { tag: "INGEST", time: "3m ago", text: "Sentinel-2 RGB tile indexed — 10×10 km, EPSG:32645" },
    { tag: "INGEST", time: "6m ago", text: "Kolkata / Bardhaman / Sundarbans imagery indexed — full pipeline pending" },
];

function renderActivityFeed() {
    const feed = document.getElementById("activity-feed");

    if (!feed) {
        return;
    }

    feed.innerHTML = "";

    for (const entry of ACTIVITY_LOG) {
        const item = document.createElement("div");
        item.className = entry.warn ? "activity-item warn" : "activity-item";

        item.innerHTML = `
            <div class="activity-item-head">
                <span>${entry.tag}</span>
                <span class="activity-item-time">${entry.time}</span>
            </div>
            <div class="activity-item-text">${entry.text}</div>
        `;

        feed.appendChild(item);
    }
}

renderActivityFeed();

function prependActivity(entry) {
    ACTIVITY_LOG.unshift(entry);
    renderActivityFeed();
}


// ============================================================
// LOAD TERRAIN
// ============================================================

async function loadTerrain() {
    console.log("Loading Darjeeling terrain...");

    const terrainResponse = await fetch("/data/darjeeling/terrain.json");

    if (!terrainResponse.ok) {
        throw new Error(`Failed to load terrain: ${terrainResponse.status}`);
    }

    const terrainData = await terrainResponse.json();

    console.log("Terrain loaded:", terrainData);


    // --------------------------------------------------------
    // Load visual layers
    // --------------------------------------------------------

    const textureLoader = new THREE.TextureLoader();

    console.log("Loading imagery and depth maps...");

    const satelliteTexture = await textureLoader.loadAsync("/data/darjeeling/satellite.png");
    const depthTexture = await textureLoader.loadAsync("/data/darjeeling/relative_depth.png");
    const elevationTexture = await textureLoader.loadAsync("/data/darjeeling/elevation.png");


    // --------------------------------------------------------
    // Create terrain
    // --------------------------------------------------------

    const terrain = createTerrain(
        scene,
        terrainData,
        satelliteTexture,
        depthTexture,
        elevationTexture
    );

    console.log("Terrain created:", terrain);


    // --------------------------------------------------------
    // Initial camera
    // --------------------------------------------------------

    camera.position.set(0, 95, 125);
    controls.target.set(0, 12, 0);
    controls.update();


    // --------------------------------------------------------
    // Elevation information
    // --------------------------------------------------------

    const elevationElement = document.getElementById("elevation-value");

    if (elevationElement) {
        elevationElement.textContent = `${Math.round(terrain.elevationMin)}–${Math.round(terrain.elevationMax)} m`;
    }


    // --------------------------------------------------------
    // Layer buttons
    // --------------------------------------------------------

    const satelliteButton = document.getElementById("satellite-button");
    const depthButton = document.getElementById("depth-button");
    const elevationButton = document.getElementById("elevation-button");

    function activateLayer(layer) {
        terrain.setLayer(layer);

        if (satelliteButton) {
            satelliteButton.classList.toggle("active", layer === "satellite");
        }

        if (depthButton) {
            depthButton.classList.toggle("active", layer === "depth");
        }

        if (elevationButton) {
            elevationButton.classList.toggle("active", layer === "elevation");
        }

        setLayerDescription(layer);
    }

    satelliteButton?.addEventListener("click", () => { activateLayer("satellite"); });
    depthButton?.addEventListener("click", () => { activateLayer("depth"); });
    elevationButton?.addEventListener("click", () => { activateLayer("elevation"); });


    // ========================================================
    // RECONSTRUCTION SEQUENCE
    // (shared by the RUN RECONSTRUCTION button and the mock
    // upload flow — both resolve to the same Darjeeling result)
    // ========================================================

    const STAGES = [
        { step: "satellite", layer: "satellite", duration: 900 },
        { step: "depth", layer: "depth", duration: 1350 },
        { step: "alignment", layer: "elevation", duration: 1200 },
        { step: "terrain", layer: "elevation", duration: 1450 },
    ];

    const STAGE_TOTAL_MS = STAGES.reduce((sum, s) => sum + s.duration, 0);

    const progressBar = document.getElementById("pipeline-progress");
    const progressFill = document.getElementById("pipeline-progress-fill");
    const pipelineSteps = document.querySelectorAll(".pipeline-step");

    let running = false;

    async function runReconstruction(sourceLabel) {
        if (running) {
            return;
        }

        running = true;

        if (progressBar) {
            progressBar.hidden = false;
        }

        let elapsed = 0;

        for (const stage of STAGES) {
            pipelineSteps.forEach(el => {
                el.classList.toggle("active", el.dataset.step === stage.step);
            });

            activateLayer(stage.layer);

            await sleep(stage.duration);

            elapsed += stage.duration;

            if (progressFill) {
                progressFill.style.width = `${Math.round((elapsed / STAGE_TOTAL_MS) * 100)}%`;
            }
        }

        const description = document.getElementById("layer-description");
        if (description) {
            description.textContent = "Reconstruction complete — metric terrain ready";
        }

        prependActivity({
            tag: "RECONSTRUCT",
            time: "now",
            text: `${sourceLabel} — resolved to the Darjeeling demo reconstruction`,
        });

        setTimeout(() => {
            if (progressBar) {
                progressBar.hidden = true;
            }
            if (progressFill) {
                progressFill.style.width = "0%";
            }
            running = false;
        }, 1500);
    }


    // ========================================================
    // RECONSTRUCTION BUTTON
    // ========================================================

    const runButton = document.createElement("button");
    runButton.textContent = "▶ RUN RECONSTRUCTION";
    runButton.className = "run-reconstruction-button";
    document.body.appendChild(runButton);

    runButton.addEventListener("click", async () => {
        if (running) {
            return;
        }

        runButton.disabled = true;
        runButton.textContent = "PROCESSING…";

        await runReconstruction("Darjeeling scene");

        runButton.textContent = "✓ RECONSTRUCTION COMPLETE";

        setTimeout(() => {
            runButton.textContent = "↻ RUN AGAIN";
            runButton.disabled = false;
        }, 1500);
    });


    // ========================================================
    // MOCK UPLOAD FLOW
    // ========================================================

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

        prependActivity({
            tag: "UPLOAD",
            time: "now",
            text: `"${file.name}" received — running mock reconstruction`,
        });

        await runReconstruction(`"${file.name}"`);
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
}


// ============================================================
// RENDER LOOP
// ============================================================

function animate() {
    requestAnimationFrame(animate);
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

loadTerrain()
    .then(() => {
        console.log("DepthWizard2 viewer ready.");
        animate();
    })
    .catch(error => {
        console.error("DepthWizard2 startup error:", error);
    });