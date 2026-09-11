import * as THREE from "three";
import { createTerrain } from "./terrain.js";
import { createControls } from "./controls.js";
import { createTerrainRig } from "./terrainRig.js";

// Shared interactive 3D terrain viewer — one instance drives Explore's
// main canvas, a second drives Workbench's Final Demo box. Each instance
// owns its own scene/camera/renderer/controls/terrainRig, so multiple
// viewers can run side by side with fully independent camera state and
// gesture handling.
export function createTerrainViewer(canvas, options = {}) {
    const {
        // Explore's canvas sits under fixed screen panels, so its rig is
        // nudged down to clear them. Workbench's Final Demo box has no
        // such overlap, so it defaults to centered.
        rigOffsetY = 0,
        cameraTarget = new THREE.Vector3(0, 12, 0),
    } = options;

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
    });

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x08160e);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);

    // terrainRig's wheel listener must attach to domElement before
    // OrbitControls' own (created next) so it can claim plain two-finger
    // scroll for rotation ahead of OrbitControls treating it as a zoom.
    const terrainRig = createTerrainRig(renderer.domElement, 0);
    terrainRig.rig.position.set(0, rigOffsetY, 0);
    scene.add(terrainRig.rig);

    const controls = createControls(camera, renderer.domElement, cameraTarget);

    scene.add(new THREE.HemisphereLight(0xddebd8, 0x172018, 2.0));

    const sun = new THREE.DirectionalLight(0xffffff, 3.0);
    sun.position.set(-40, 100, 50);
    sun.castShadow = true;
    scene.add(sun);

    let currentTerrain = null;
    let currentRegionKey = null;
    let lastWidth = 0;
    let lastHeight = 0;

    function disposeTerrain(terrain) {
        if (!terrain) {
            return;
        }

        terrainRig.rig.remove(terrain.mesh);
        terrain.mesh.geometry.dispose();
        terrain.material.dispose();
    }

    async function loadRegion(regionKey) {
        const terrainResponse = await fetch(`/data/${regionKey}/terrain.json`);

        if (!terrainResponse.ok) {
            throw new Error(`Failed to load terrain: ${terrainResponse.status}`);
        }

        const terrainData = await terrainResponse.json();

        const textureLoader = new THREE.TextureLoader();

        const satelliteTexture = await textureLoader.loadAsync(`/data/${regionKey}/satellite.png`);
        const depthTexture = await textureLoader.loadAsync(`/data/${regionKey}/relative_depth.png`);
        const elevationTexture = await textureLoader.loadAsync(`/data/${regionKey}/elevation.png`);

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

        // ~18 degrees above the horizon, same distance regardless of
        // region — createTerrain() always builds at a fixed 100-unit
        // terrain height, so this framing holds across all regions.
        camera.position.set(0, 58, 143);
        controls.target.copy(cameraTarget);
        controls.update();
        terrainRig.reset();

        return { terrain, terrainData };
    }

    function setLayer(layer) {
        currentTerrain?.setLayer(layer);
    }

    function setFloodOverlay(active) {
        currentTerrain?.setFloodOverlay(active);
    }

    function resize(width, height) {
        if (width === 0 || height === 0) {
            return;
        }

        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
    }

    // Convenience for viewers sized by their container (Final Demo) rather
    // than by the window (Explore) — only touches the renderer/camera when
    // the box's actual pixel size has changed.
    function resizeToCanvas() {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        if (width === 0 || height === 0 || (width === lastWidth && height === lastHeight)) {
            return;
        }

        lastWidth = width;
        lastHeight = height;
        resize(width, height);
    }

    function update() {
        terrainRig.update();
        controls.update();
    }

    function render() {
        renderer.render(scene, camera);
    }

    return {
        scene,
        camera,
        renderer,
        controls,
        terrainRig,
        get currentTerrain() {
            return currentTerrain;
        },
        get currentRegionKey() {
            return currentRegionKey;
        },
        loadRegion,
        setLayer,
        setFloodOverlay,
        resize,
        resizeToCanvas,
        update,
        render,
    };
}
