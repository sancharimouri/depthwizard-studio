import * as THREE from "three";
import { createTerrain } from "./terrain.js";
import { createControls } from "./controls.js";

// Shared interactive 3D terrain viewer — one instance drives Explore's
// main canvas, a second drives Workbench's Final Demo box. Each instance
// owns its own scene/camera/renderer/controls, so multiple viewers can run
// side by side with fully independent camera state and gesture handling.
export function createTerrainViewer(canvas, options = {}) {
    const {
        // Explore's canvas sits under fixed screen panels, so its rig is
        // nudged down to clear them. Workbench's Final Demo box has no
        // such overlap, so it defaults to centered.
        rigOffsetY = 0,
        cameraTarget = new THREE.Vector3(0, 12, 0),
        // The void behind the terrain mesh — UI chrome, not imagery, so
        // Workbench's light/dark toggle is allowed to swap it (via
        // setBackground() below) while Explore's instance never calls
        // that and stays on this dark default.
        backgroundColor = 0x110f0e,
    } = options;

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
    });

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(backgroundColor);

    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);

    // Holds the terrain mesh — camera-controls orbits the camera around it
    // instead of this group rotating under a fixed camera, so it only ever
    // carries the static per-instance layout offset, never a rotation.
    const terrainGroup = new THREE.Group();
    terrainGroup.position.set(0, rigOffsetY, 0);
    scene.add(terrainGroup);

    const controls = createControls(camera, renderer.domElement, cameraTarget);

    scene.add(new THREE.HemisphereLight(0xddebd8, 0x172018, 2.0));

    const sun = new THREE.DirectionalLight(0xffffff, 3.0);
    sun.position.set(-40, 100, 50);
    sun.castShadow = true;
    scene.add(sun);

    // Persistent base reference grid under the terrain in every view mode
    // (true colour, DSM, DEM, flat, wireframe), as in the reference repo
    // (gridHelper 80 units / 40 divisions under a 60-unit terrain, i.e.
    // 1.33× the terrain, shown by default). Rebuilt per terrain for its size.
    let grid = null;
    let gridDark = true;
    const GRID_COLORS = { dark: [0x3f4a40, 0x252b26], light: [0xb4a17c, 0xd6c6a3] };

    function buildGrid(terrain) {
        if (grid) {
            scene.remove(grid);
            grid.geometry.dispose();
            grid.material.dispose();
        }
        const size = 1.33 * Math.max(terrain.terrainWidth, terrain.terrainHeight);
        const [center, line] = GRID_COLORS[gridDark ? "dark" : "light"];
        grid = new THREE.GridHelper(size, 40, center, line);
        grid.position.set(0, rigOffsetY - 0.3, 0);
        grid.material.transparent = true;
        grid.material.opacity = 0.9;
        scene.add(grid);
    }

    let currentTerrain = null;
    let currentRegionKey = null;
    let lastWidth = 0;
    let lastHeight = 0;

    function disposeTerrain(terrain) {
        if (!terrain) {
            return;
        }

        terrainGroup.remove(terrain.mesh);
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
            terrainGroup,
            regionKey,
            terrainData,
            satelliteTexture,
            depthTexture,
            elevationTexture
        );

        currentTerrain = terrain;
        currentRegionKey = regionKey;
        buildGrid(terrain);

        // ~18 degrees above the horizon, same distance regardless of
        // region — createTerrain() always builds at a fixed 100-unit
        // terrain height, so this framing holds across all regions.
        controls.reset();

        return { terrain, terrainData };
    }

    function setLayer(layer) {
        currentTerrain?.setLayer(layer);
    }

    function setFloodOverlay(active) {
        currentTerrain?.setFloodOverlay(active);
    }

    function setBackground(color) {
        scene.background = new THREE.Color(color);
        const c = scene.background;
        const dark = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b < 0.5;
        if (dark !== gridDark) {
            gridDark = dark;
            if (currentTerrain) {
                buildGrid(currentTerrain);
            }
        }
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
        get currentTerrain() {
            return currentTerrain;
        },
        get grid() {
            return grid;
        },
        get currentRegionKey() {
            return currentRegionKey;
        },
        loadRegion,
        setLayer,
        setFloodOverlay,
        setBackground,
        resize,
        resizeToCanvas,
        update,
        render,
    };
}
