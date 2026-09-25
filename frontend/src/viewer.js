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

    // Structure yaw: the terrain group turns about the world vertical axis
    // (rotate buttons, auto-rotation). The camera and the ground grid stay
    // put, so the grid never rotates or tilts. Everything that reads the
    // terrain (picking, measurements, markers, water) goes through its
    // matrixWorld, so it follows.
    const YAW_ANIM_MS = 450;
    let yawTarget = 0;
    let yawAnim = null;
    function setStructureYaw(yaw, animate = true) {
        yawTarget = yaw;
        if (!animate) {
            yawAnim = null;
            terrainGroup.rotation.y = yaw;
            return;
        }
        yawAnim = { from: terrainGroup.rotation.y, to: yaw, start: performance.now() };
    }
    function stepYaw() {
        if (!yawAnim) {
            return;
        }
        const t = Math.min(1, (performance.now() - yawAnim.start) / YAW_ANIM_MS);
        const e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
        terrainGroup.rotation.y = yawAnim.from + (yawAnim.to - yawAnim.from) * e;
        if (t >= 1) {
            yawAnim = null;
        }
    }
    // idle auto-rotation spins the structure, not the camera
    controls.autoRotateHandler = radians => {
        if (!yawAnim) {
            yawTarget -= radians;
            terrainGroup.rotation.y = yawTarget;
        }
    };

    scene.add(new THREE.HemisphereLight(0xddebd8, 0x172018, 2.0));

    const sun = new THREE.DirectionalLight(0xffffff, 3.0);
    sun.position.set(-40, 100, 50);
    sun.castShadow = true;
    scene.add(sun);

    // Persistent base reference grid under the terrain in every view mode —
    // now INFINITE: a shader grid on a horizontal plane that follows the
    // camera in X/Z (so it never ends) while its lines are computed from
    // world coordinates (so they never slide). It is a direct child of the
    // scene, at a fixed height, and never rotated: orbiting / tilting /
    // zooming moves only the camera, so the grid always stays flat.
    // Anti-aliased with fwidth; fades out with distance from the camera.
    const GRID_COLORS = { dark: [0x3a443c, 0x2a312c], light: [0xd9ccae, 0xe2d5b8] }; // [line, unused] — sRGB, shown as-is now
    let gridDark = true;
    const gridMaterial = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        uniforms: {
            // uniform small squares: 2.5 world units (the terrain is 100 units N-S), no major lines
            uMinor: { value: 2.5 },
            uMinorColor: { value: new THREE.Color(GRID_COLORS.dark[1]) },
            uMajorColor: { value: new THREE.Color(GRID_COLORS.dark[0]) },
            uFadeStart: { value: 120 },
            uFadeEnd: { value: 520 },
        },
        vertexShader: /* glsl */ `
            varying vec3 vWorld;
            void main() {
                vec4 world = modelMatrix * vec4(position, 1.0);
                vWorld = world.xyz;
                gl_Position = projectionMatrix * viewMatrix * world;
            }
        `,
        fragmentShader: /* glsl */ `
            uniform float uMinor;
            uniform vec3 uMinorColor;
            uniform vec3 uMajorColor;
            uniform float uFadeStart;
            uniform float uFadeEnd;
            varying vec3 vWorld;

            float gridLine(vec2 p, float size) {
                vec2 c = p / size;
                vec2 w = fwidth(c);
                vec2 d = abs(fract(c - 0.5) - 0.5) / w;
                float line = 1.0 - min(min(d.x, d.y), 1.0);
                // squares smaller than ~3 px would shimmer: fade those lines out
                float cellPx = 1.0 / max(w.x, w.y);
                return line * smoothstep(2.0, 5.0, cellPx);
            }

            void main() {
                float minor = gridLine(vWorld.xz, uMinor);
                float dist = distance(vWorld.xz, cameraPosition.xz);
                float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, dist);
                vec3 color = uMajorColor;
                float alpha = minor * fade;
                if (alpha < 0.01) discard;
                gl_FragColor = vec4(color, alpha);
                // uniforms hold linear colours: convert to the output colour
                // space, or every line renders darker than its hex value
                #include <colorspace_fragment>
            }
        `,
    });
    const grid = new THREE.Mesh(new THREE.PlaneGeometry(2400, 2400), gridMaterial);
    grid.rotation.x = -Math.PI / 2; // lies in the world XZ plane — set once, never changed
    grid.position.y = rigOffsetY - 0.3; // just under the terrain's base
    grid.renderOrder = -1;
    grid.frustumCulled = false;
    grid.onBeforeRender = (_renderer, _scene, cam) => {
        // follow the camera sideways only: the plane never ends, the lines stay put
        grid.position.x = cam.position.x;
        grid.position.z = cam.position.z;
        grid.updateMatrixWorld();
    };
    scene.add(grid);

    function setGridTheme(dark) {
        const [major, minor] = GRID_COLORS[dark ? "dark" : "light"];
        gridMaterial.uniforms.uMajorColor.value.setHex(major);
        gridMaterial.uniforms.uMinorColor.value.setHex(minor);
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
            setGridTheme(dark);
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
        stepYaw();
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
        setStructureYaw,
        getStructureYaw: () => yawTarget,
        setLayer,
        setFloodOverlay,
        setBackground,
        resize,
        resizeToCanvas,
        update,
        render,
    };
}
