import * as THREE from "three";
import { createTerrain } from "./terrain.js";
import { createFlythrough } from "./flythrough.js";

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

const flythrough = createFlythrough(camera, scene);

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
    camera.lookAt(0, 12, 0);


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
    // RECONSTRUCTION BUTTON
    // ========================================================

    const runButton = document.createElement("button");
    runButton.textContent = "▶ RUN RECONSTRUCTION";
    runButton.className = "run-reconstruction-button";
    document.body.appendChild(runButton);

    let running = false;

    runButton.addEventListener("click", async () => {
        if (running) {
            return;
        }

        running = true;
        runButton.disabled = true;
        runButton.textContent = "PROCESSING…";

        // ------------------------------------------------
        // Start from satellite
        // ------------------------------------------------
        activateLayer("satellite");
        await sleep(550);
        await sleep(600);
        await sleep(350);

        // ------------------------------------------------
        // Relative depth
        // ------------------------------------------------
        activateLayer("depth");
        await sleep(650);
        await sleep(700);
        await sleep(350);

        // ------------------------------------------------
        // Metric alignment
        // ------------------------------------------------
        activateLayer("elevation");
        await sleep(550);
        await sleep(750);
        await sleep(350);

        // ------------------------------------------------
        // 3D reconstruction
        // ------------------------------------------------
        activateLayer("elevation");
        await sleep(650);
        await sleep(650);
        await sleep(700);

        // ------------------------------------------------
        // Finished
        // ------------------------------------------------
        activateLayer("elevation");
        
        const description = document.getElementById("layer-description");
        if (description) {
            description.textContent = "Reconstruction complete — metric terrain ready";
        }
        
        flythrough.start();
        
        runButton.textContent = "✓ RECONSTRUCTION COMPLETE";
        
        setTimeout(() => {
            runButton.textContent = "↻ RUN AGAIN";
            runButton.disabled = false;
            running = false;
        }, 2500);
    });
}


// ============================================================
// RENDER LOOP
// ============================================================

function animate() {
    requestAnimationFrame(animate);
    flythrough.update();
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