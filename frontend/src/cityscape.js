// City models: the TRUE COLOR layer of the demo page's Bangalore region and of the Studio library's close-up city
// tiles. Lazy-loaded by main.js (attachCity) when a terrain carries a `structure`. Data: a model JSON baked by
// scripts/bake_cityscape.py (local metres, x east / y north, origin = box centre): ground, roads and either
// extruded footprints (`buildings`) or a meshopt-compressed GLB of a 3D city model (`glb`, scripts/bake_city_glb.mjs;
// x east, y up, z = -north, `y_offset` to the ground median).
// Look: a dusk grade. Warm peach blocks deepening to muted mauve, a dusky plum ground, a low golden sun with
// lavender sky fill, light haze; textured models keep their textures. While the model shows, the viewer's own lights
// are off, the frame is tone-mapped, and ambient occlusion runs as a post pass (stepped down when frames get slow) on
// top of a baked ground AO map that is always on. OSM water and parks (tiles that have them) lie under the roads.
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const TOP = new THREE.Color("#fff0e8");   // building colour at the roof: warm peach...
const FOOT = new THREE.Color("#897386");  // ...deepening to plum at street level
const EDGE = "#ffd2c0";       // soft warm edge lines
const GROUND = "#716274";     // dusk plum
const STREET = "#ebbdb0";     // glowing rose-peach streets
const PLINTH = "#20171d";
const HAZE = "#c8b4c6";       // light lavender haze, thickening with distance
const WATER = "#56557b";      // dusk water: blue-violet, a little glossy
const PARK = "#7b8064";       // muted sage parks
const GROUND_STEP = 5; // m, ground mesh spacing (FABDEM is 10 m; finer so roads follow it)
const MIN_FPS = 30;    // below this for a while: drop ambient occlusion, then shadows, then render at 3/4 resolution
const STEPS = ["0.75x resolution", "shadows off"]; // the step taken on reaching each level
const GRACE_MS = 2000; // frames right after a model shows (uploads, first draws) are not judged
let deviceLevel = 2;   // the quality this device settled on at rest: the next model starts there, so switching never re-steps
const SPIN_MOVING = 0.2;   // rad/s: a step taken at or above this camera rotation speed was caused by the motion
const SPIN_RECOVER = 0.25; // ...and is undone once the rotation slows to this fraction of its speed at the step
const RECOVER_MS = 300;    // for this long
// a turn of the event loop, so a long build doesn't freeze the page's animations
const yieldToPage = () => new Promise(resolve => setTimeout(resolve));

// Box-plane UVs (u east, v north across the box) on a ground-level mesh in model coordinates, so the baked ground AO
// map lines up on the ground, roads, water and parks alike (the ground plane's own UVs already run this way).
function boxUv(geo, size) {
    const p = geo.attributes.position, uv = new Float32Array(p.count * 2);
    for (let i = 0; i < p.count; i++) {
        uv[2 * i] = p.getX(i) / size + 0.5;
        uv[2 * i + 1] = -p.getZ(i) / size + 0.5;
    }
    geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
}

// Bilinear ground height (m, relative to the box median) at local x, y.
function groundSampler(g) {
    if (!g.z) return () => 0; // flat
    return (x, y) => {
        const c = Math.min(Math.max((x - g.x0) / g.res, 0), g.nx - 1.001);
        const r = Math.min(Math.max((g.y0 - y) / g.res, 0), g.ny - 1.001);
        const c0 = Math.floor(c), r0 = Math.floor(r), fc = c - c0, fr = r - r0;
        const z = (rr, cc) => g.z[rr * g.nx + cc];
        return (z(r0, c0) * (1 - fc) + z(r0, c0 + 1) * fc) * (1 - fr)
            + (z(r0 + 1, c0) * (1 - fc) + z(r0 + 1, c0 + 1) * fc) * fr;
    };
}

function buildGround(data, groundAt) {
    const n = Math.round(data.size_m / GROUND_STEP);
    const geo = new THREE.PlaneGeometry(data.size_m, data.size_m, n, n);
    geo.rotateX(-Math.PI / 2); // (x, y) plane -> (x, -z): three z = -north
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) pos.setY(i, groundAt(pos.getX(i), -pos.getZ(i)));
    geo.computeVertexNormals(); // PlaneGeometry's UVs already span the box, north up
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: GROUND, roughness: 0.95 }));
    mesh.receiveShadow = true;

    // edge walls down to a common floor, following the ground profile, so the block reads as a cut-out slab
    const half = data.size_m / 2, floor = Math.min(...(data.ground.z ?? [0])) - 8, wall = [];
    const corners = [[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]];
    for (let c = 0; c < 4; c++) {
        const [x0, y0] = corners[c], [x1, y1] = corners[c + 1];
        for (let i = 0; i < n; i++) {
            const xa = x0 + ((x1 - x0) * i) / n, ya = y0 + ((y1 - y0) * i) / n;
            const xb = x0 + ((x1 - x0) * (i + 1)) / n, yb = y0 + ((y1 - y0) * (i + 1)) / n;
            const ta = [xa, groundAt(xa, ya), -ya], tb = [xb, groundAt(xb, yb), -yb];
            wall.push(...ta, xa, floor, -ya, ...tb, ...tb, xa, floor, -ya, xb, floor, -yb);
        }
    }
    const wallGeo = new THREE.BufferGeometry();
    wallGeo.setAttribute("position", new THREE.Float32BufferAttribute(wall, 3));
    wallGeo.computeVertexNormals();
    const skirt = new THREE.Mesh(wallGeo, new THREE.MeshStandardMaterial({ color: PLINTH, roughness: 1, side: THREE.DoubleSide }));
    const group = new THREE.Group();
    group.add(mesh, skirt);
    return group;
}

// thin warm lines along the creases (walls meeting roofs, corners)
function edgeLines(geometry) {
    return new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry, 30),
        new THREE.LineBasicMaterial({ color: EDGE, transparent: true, opacity: 0.12 }),
    );
}

async function buildBuildings(data, groundAt) {
    const geos = [];
    for (const [h, p] of data.buildings) {
        if (geos.length % 400 === 399) await yieldToPage();
        const shape = new THREE.Shape();
        let cx = 0, cy = 0;
        for (let i = 0; i < p.length; i += 2) {
            (i ? shape.lineTo : shape.moveTo).call(shape, p[i], p[i + 1]);
            cx += p[i]; cy += p[i + 1];
        }
        const base = groundAt(cx / (p.length / 2), cy / (p.length / 2)) - 1; // sunk 1 m so slopes never show a gap
        const geo = new THREE.ExtrudeGeometry(shape, { depth: h + 1, bevelEnabled: false });
        geo.rotateX(-Math.PI / 2); // extrusion (z) -> up, north (y) -> -z
        geo.translate(0, base, 0);
        // vertical gradient: deep at the foot, white at the roof
        const pos = geo.attributes.position, colors = new Float32Array(pos.count * 3), c = new THREE.Color();
        for (let i = 0; i < pos.count; i++) {
            const t = Math.min(Math.max((pos.getY(i) - base) / (h + 1), 0), 1);
            c.lerpColors(FOOT, TOP, t).toArray(colors, i * 3);
        }
        geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geos.push(geo);
    }
    const merged = mergeGeometries(geos, false);
    geos.forEach(g => g.dispose());
    const mesh = new THREE.Mesh(merged, new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.85, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    }));
    mesh.castShadow = mesh.receiveShadow = true;
    const group = new THREE.Group();
    group.add(mesh, edgeLines(merged));
    return group;
}

// A 3D city model: textured parts keep their textures; untextured parts carry COLOR_0 = height within their building
// (0 at the foot, 1 at the roof), mapped to the same gradient as the extruded blocks.
async function loadModelBuildings(data) {
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(`${import.meta.env.BASE_URL}${data.glb.url}`);
    const root = gltf.scene;
    root.position.y = data.glb.y_offset;
    const meshes = [];
    root.traverse(o => { if (o.isMesh) meshes.push(o); });
    const c = new THREE.Color();
    for (const mesh of meshes) {
        const shared = { roughness: 0.85, metalness: 0, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 };
        if (mesh.material.map) {
            // photo facades (dark blue glass) read purple-black under the dusk light: their brightness is mapped onto the
            // blocks' warm peach (a fifth of the original colour kept), so they sit with the palette; the detail stays
            mesh.material = new THREE.MeshStandardMaterial({ ...shared, map: mesh.material.map, roughness: 0.9 });
            mesh.material.onBeforeCompile = shader => {
                shader.fragmentShader = shader.fragmentShader.replace("#include <map_fragment>", `#include <map_fragment>
                    float lum = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));
                    diffuseColor.rgb = mix(vec3(1.0, 0.87, 0.80) * (0.55 + 0.9 * lum), diffuseColor.rgb, 0.2);`);
            };
        } else {
            const t = mesh.geometry.attributes.color, colors = new Float32Array(t.count * 3);
            for (let i = 0; i < t.count; i++) c.lerpColors(FOOT, TOP, t.getX(i)).toArray(colors, i * 3);
            mesh.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
            mesh.material = new THREE.MeshStandardMaterial({ ...shared, vertexColors: true, flatShading: !mesh.geometry.attributes.normal });
        }
        mesh.castShadow = mesh.receiveShadow = true;
        mesh.add(edgeLines(mesh.geometry));
    }
    return root;
}

// Each road: a flat ribbon of its width, resampled every <= 5 m so it follows the ground.
function buildRoads(data, groundAt) {
    const pos = [];
    for (const [w, p] of data.roads) {
        const pts = [];
        for (let i = 0; i + 3 < p.length; i += 2) {
            const [x0, y0, x1, y1] = [p[i], p[i + 1], p[i + 2], p[i + 3]];
            const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 5));
            for (let s = 0; s < steps; s++) pts.push([x0 + ((x1 - x0) * s) / steps, y0 + ((y1 - y0) * s) / steps]);
        }
        pts.push([p[p.length - 2], p[p.length - 1]]);
        for (let i = 0; i + 1 < pts.length; i++) {
            const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
            const len = Math.hypot(bx - ax, by - ay) || 1;
            const nx = (-(by - ay) / len) * (w * 0.28), ny = ((bx - ax) / len) * (w * 0.28); // 56% width: crisp lines
            const v = (x, y) => [x, groundAt(x, y) + 0.3, -y];
            const a1 = v(ax + nx, ay + ny), a2 = v(ax - nx, ay - ny), b1 = v(bx + nx, by + ny), b2 = v(bx - nx, by - ny);
            pos.push(...a1, ...a2, ...b1, ...b1, ...a2, ...b2);
        }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.computeVertexNormals();
    boxUv(geo, data.size_m);
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        color: STREET, roughness: 0.9, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: -2,
    }));
    mesh.receiveShadow = true;
    return mesh;
}

// OSM areas (water, parks: [outer ring, holes...] flat coordinate lists), laid on the ground under the roads.
function buildAreas(data, groundAt, key, color, extra = {}) {
    if (!data[key]?.length) return null;
    const pts = r => Array.from({ length: r.length / 2 }, (_, i) => new THREE.Vector2(r[2 * i], r[2 * i + 1]));
    const shapes = data[key].map(([outer, ...holes]) => {
        const shape = new THREE.Shape(pts(outer));
        shape.holes = holes.map(h => new THREE.Path(pts(h)));
        return shape;
    });
    const geo = new THREE.ShapeGeometry(shapes);
    geo.rotateX(-Math.PI / 2); // (x, y) -> (x, -z)
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) p.setY(i, groundAt(p.getX(i), -p.getZ(i)) + 0.15);
    geo.computeVertexNormals();
    boxUv(geo, data.size_m);
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        color, roughness: 0.9, polygonOffset: true, polygonOffsetFactor: -1, ...extra,
    }));
    mesh.receiveShadow = true;
    return mesh;
}

// Separable box blur, 3 passes (~Gaussian), radius r px, edges clamped; n x n floats.
function blur(src, n, r) {
    let a = Float32Array.from(src), b = new Float32Array(n * n);
    const pass = (from, to, step, line) => {
        for (let l = 0; l < n; l++) {
            const o = l * line;
            let sum = 0;
            for (let k = -r; k <= r; k++) sum += from[o + Math.min(Math.max(k, 0), n - 1) * step];
            for (let i = 0; i < n; i++) {
                to[o + i * step] = sum / (2 * r + 1);
                sum += from[o + Math.min(i + r + 1, n - 1) * step] - from[o + Math.max(i - r, 0) * step];
            }
        }
    };
    for (let it = 0; it < 3; it++) {
        pass(a, b, 1, n); // rows
        pass(b, a, n, 1); // columns
    }
    return a;
}

// Baked ground ambient occlusion: the buildings drawn once from straight above (height in the red channel), then
// blurred into a contact darkening at their feet and a wider one round tall blocks. Applied as a map on the
// ground-level meshes, so it costs nothing per frame and stays on at every quality step.
function bakeGroundAO(renderer, buildings, sizeM, groundMeshes) {
    const N = 512, half = sizeM / 2, mPerPx = sizeM / N; // 1-3 m per pixel: plenty for a blurred darkening
    buildings.updateMatrixWorld(true);
    const top = new THREE.Box3().setFromObject(buildings).max.y;
    const camera = new THREE.OrthographicCamera(-half, half, half, -half, 1, top + 2000);
    camera.position.set(0, top + 1000, 0);
    camera.up.set(0, 0, -1); // north up: image row 0 (bottom) is the south edge, as the maps' v = 0
    camera.lookAt(0, 0, 0);
    const scene = new THREE.Scene();
    scene.overrideMaterial = new THREE.ShaderMaterial({
        uniforms: { uTop: { value: Math.max(top, 1) } },
        vertexShader: "varying float vH; void main() { vec4 w = modelMatrix * vec4(position, 1.0); vH = w.y;"
            + " gl_Position = projectionMatrix * viewMatrix * w; }",
        fragmentShader: "uniform float uTop; varying float vH; void main() { gl_FragColor = vec4(clamp(vH / uTop, 0.0, 1.0), 1.0, 0.0, 1.0); }",
    });
    const parent = buildings.parent;
    scene.add(buildings);
    const target = new THREE.WebGLRenderTarget(N, N);
    const clear = renderer.getClearColor(new THREE.Color()), clearAlpha = renderer.getClearAlpha();
    renderer.setRenderTarget(target);
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    renderer.render(scene, camera);
    const px = new Uint8Array(N * N * 4);
    renderer.readRenderTargetPixels(target, 0, 0, N, N, px);
    renderer.setRenderTarget(null);
    renderer.setClearColor(clear, clearAlpha);
    target.dispose();
    scene.overrideMaterial.dispose();
    parent.add(buildings);

    const occ = new Float32Array(N * N), tall = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
        occ[i] = px[4 * i + 1] > 127 ? 1 : 0;
        tall[i] = occ[i] * Math.min((px[4 * i] / 255) * top / 40, 1); // full weight from 40 m up
    }
    const near = blur(occ, N, Math.max(1, Math.round(3 / mPerPx)));
    const far = blur(tall, N, Math.max(2, Math.round(14 / mPerPx)));
    const data = new Uint8Array(N * N * 4);
    for (let i = 0; i < N * N; i++) {
        const ao = Math.min(Math.max(1 - 0.55 * near[i] - 0.4 * far[i], 0.35), 1) * 255;
        data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = ao;
        data[4 * i + 3] = 255;
    }
    const map = new THREE.DataTexture(data, N, N);
    map.magFilter = THREE.LinearFilter;
    map.minFilter = THREE.LinearMipmapLinearFilter;
    map.generateMipmaps = true;
    map.needsUpdate = true;
    for (const mesh of groundMeshes) {
        mesh.material.map = map; // linear multiply on the albedo
        mesh.material.needsUpdate = true;
    }
}

// Sunset: lavender sky fill from above, plum bounce from below, a low gold-orange sun from the west, soft shadows
// sized to the box. `unitsPerMetre`: world units per metre of the parent's frame, for the shadow camera.
function addLights(parent, unitsPerMetre, sizeM) {
    const sky = new THREE.HemisphereLight("#cdbadb", "#4f4052", 1.3);
    sky.position.set(0, 1e6, 0); // its "up" is its world position: far up, so any parent offset can't tilt it
    parent.add(sky);
    const sun = new THREE.DirectionalLight("#ffbf78", 2.8);
    const s = sizeM / 500; // Bengaluru's 500 m box = 1
    sun.position.set(-520 * s, 340 * s, 240 * s);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const k = unitsPerMetre * s;
    Object.assign(sun.shadow.camera, { left: -300 * k, right: 300 * k, top: 300 * k, bottom: -300 * k, near: 50 * k, far: 1400 * k });
    sun.shadow.radius = 4;
    sun.shadow.blurSamples = 8;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 1.2 * unitsPerMetre; // no shadow speckle on the walls
    parent.add(sun, sun.target); // target in the graph, so it follows the parent's frame
    return sun;
}

// One Object3D in local metres (x east, y up, z = -north, origin = box centre) carrying its own lights, to be placed
// on the terrain by the caller. No text is drawn on it (no landmark or road names).
export async function buildCityStructure(data, unitsPerMetre, renderer) {
    const groundAt = groundSampler(data.ground);
    const object = new THREE.Group();
    const ground = buildGround(data, groundAt);
    await yieldToPage();
    const roads = buildRoads(data, groundAt);
    const water = buildAreas(data, groundAt, "water", WATER, { roughness: 0.35, metalness: 0.15 });
    const parks = buildAreas(data, groundAt, "parks", PARK);
    await yieldToPage();
    const buildings = data.glb ? await loadModelBuildings(data) : await buildBuildings(data, groundAt);
    await yieldToPage();
    object.add(...[ground, water, parks, roads, buildings].filter(Boolean));
    bakeGroundAO(renderer, buildings, data.size_m, [ground.children[0], roads, water, parks].filter(Boolean));
    await yieldToPage();
    const sun = addLights(object, unitsPerMetre, data.size_m);
    return { object, sun };
}

// While `frame` is visible: the viewer's own lights off, tone mapping and haze (no screen-space AO: the GTAO pass
// drew the model near-black until the frame-rate check dropped it, 2026-10-12; the ground keeps its baked AO).
// Quality steps down when the frame rate stays under MIN_FPS: first the sun's shadows,
// then the render resolution (3/4). A step taken while the camera rotates (fast orbiting) is temporary: quality
// comes back once the rotation slows to a quarter of its speed at the step, or stops. The hysteresis is that ratio
// plus the 1.5 s slow-frame window before any step. A step taken at rest (a slow device) stays, as before.
function createCityRenderer(viewer, frame, sun) {
    const { renderer, scene, camera } = viewer;
    const viewerLights = scene.children.filter(o => o.isLight);
    const fog = new THREE.FogExp2(HAZE, 0.0024);
    const saved = {};
    let active = false;
    let level = deviceLevel; // 2: shadows, 1: none, 0: none at 3/4 resolution
    let restLevel = level;   // the level to come back to when a motion-caused step is undone
    sun.castShadow = level >= 2;
    let activeSince = 0;
    let stepPending = false;
    let restorePending = false;
    // the view's rotation relative to the model: camera orbit and the structure's own yaw (rotate buttons, auto-rotate)
    const viewQ = new THREE.Quaternion(), lastQ = new THREE.Quaternion();
    let spin = 0, peakSpin = 0, stepSpin = 0, calmSince = null; // camera rotation speed (rad/s, smoothed); its peak over the slow window; that peak at the step

    // the quality step, crossfaded from the frame just drawn at the old quality (no visible pop)
    function stepDown() {
        stepPending = false;
        viewer.holdFrame({ fresh: false });
        level -= 1;
        stepSpin = peakSpin;
        peakSpin = 0;
        if (stepSpin < SPIN_MOVING) {
            restLevel = deviceLevel = level;
        }
        if (level === 1) {
            sun.castShadow = false;
        } else if (level === 0) {
            renderer.setPixelRatio(saved.pixelRatio * 0.75);
        }
        viewer.releaseFrame(600);
    }
    // back to restLevel, crossfaded the same way
    function restore() {
        restorePending = false;
        viewer.holdFrame({ fresh: false });
        if (level === 0) {
            renderer.setPixelRatio(saved.pixelRatio);
        }
        level = restLevel;
        sun.castShadow = level >= 2;
        viewer.releaseFrame(600);
        console.info("city model: rotation slowed, quality restored");
    }
    let last = 0, avg = 1000 / 60, slowSince = null;

    function setActive(on) {
        if (on === active) return;
        active = on;
        if (on) {
            Object.assign(saved, { toneMapping: renderer.toneMapping, fog: scene.fog, pixelRatio: renderer.getPixelRatio() });
            viewerLights.forEach(l => { saved[l.uuid] = l.visible; l.visible = false; });
            renderer.toneMapping = THREE.NeutralToneMapping;
            scene.fog = fog;
            if (level === 0) renderer.setPixelRatio(saved.pixelRatio * 0.75);
            activeSince = performance.now();
        } else {
            viewerLights.forEach(l => { l.visible = saved[l.uuid] ?? true; });
            renderer.toneMapping = saved.toneMapping;
            scene.fog = saved.fog;
            renderer.setPixelRatio(saved.pixelRatio);
            last = 0;
        }
    }

    // frame-time average; time spent off-screen (hidden tab, other page) is skipped
    function watchFrameRate() {
        const now = performance.now();
        const dt = last ? now - last : 0;
        last = now;
        frame.getWorldQuaternion(viewQ).invert().multiply(camera.quaternion);
        const turned = viewQ.angleTo(lastQ);
        lastQ.copy(viewQ);
        if (dt && dt <= 250) {
            spin = spin * 0.85 + (turned / (dt / 1000)) * 0.15;
            peakSpin = Math.max(peakSpin, spin);
        }
        if (level < restLevel && stepSpin >= SPIN_MOVING && spin < stepSpin * SPIN_RECOVER) {
            calmSince ??= now;
            if (now - calmSince > RECOVER_MS) {
                restorePending = true;
                calmSince = null;
                slowSince = null;
                avg = 1000 / 60;
                return;
            }
        } else {
            calmSince = null;
        }
        if (!dt || dt > 250 || level === 0 || now - activeSince < GRACE_MS) {
            slowSince = null;
            avg = 1000 / 60;
            return;
        }
        avg = avg * 0.92 + dt * 0.08;
        if (avg <= 1000 / MIN_FPS) {
            slowSince = null;
            peakSpin = 0;
            return;
        }
        slowSince ??= now;
        if (now - slowSince > 1500) {
            stepPending = true; // applied after this frame is drawn, under a crossfade from it
            console.info(`city model: ${Math.round(1000 / avg)} fps, rotating ${peakSpin.toFixed(2)} rad/s, quality step: ${STEPS[level - 1]}`);
            slowSince = null;
            avg = 1000 / 60;
        }
    }

    return {
        // Builds everything the first visible frame needs (shadow map, shaders, AO targets) without drawing to the
        // screen: one off-screen render in the model's state, then compile().
        warm() {
            setActive(true);
            const target = new THREE.WebGLRenderTarget(16, 16);
            renderer.setRenderTarget(target);
            renderer.render(scene, camera);
            renderer.setRenderTarget(null);
            target.dispose();
            renderer.compile(scene, camera);
            setActive(false);
        },
        // true when it drew the frame (the model is showing)
        render() {
            const on = frame.visible && frame.parent !== null;
            setActive(on);
            if (!on) return false;
            watchFrameRate();
            renderer.render(scene, camera);
            if (stepPending) {
                stepDown();
            } else if (restorePending) {
                restore();
            }
            return true;
        },
        dispose() {
            setActive(false);
        },
        get level() { return level; },
        get fps() { return Math.round(1000 / avg); },
    };
}

// Places the model on the terrain (its `structure`: { url, affine, groundMedian }) and hands the viewer its renderer.
// model metres (x east, y up, z = -north) -> the mesh's local frame (x east, y north, z up, mesh units): horizontally
// the baked affine to degrees from the tile centre, vertically the terrain's own metres-to-units.
export async function attachCity(viewer, terrain, { url, affine, groundMedian }) {
    const res = await fetch(`${import.meta.env.BASE_URL}${url}`);
    if (!res.ok) {
        throw new Error(`city model: HTTP ${res.status}`);
    }
    const [a, b, c, d, e, f] = affine;
    const kx = terrain.terrainWidth / terrain.geographicWidth;
    const ky = terrain.terrainHeight / terrain.geographicHeight;
    const v = terrain.localZForElevation(1) - terrain.localZForElevation(0);
    const unitsPerMetre = Math.hypot(kx * a, ky * d);
    const city = await buildCityStructure(await res.json(), unitsPerMetre, viewer.renderer);
    const frame = new THREE.Group();
    frame.matrixAutoUpdate = false;
    frame.matrix.set(
        kx * a, 0, -kx * b, kx * c,
        ky * d, 0, -ky * e, ky * f,
        0, v, 0, terrain.localZForElevation(groundMedian),
        0, 0, 0, 1,
    );
    frame.add(city.object);
    terrain.attachStructure(frame);
    const cityRenderer = createCityRenderer(viewer, frame, city.sun);
    viewer.setCityRenderer(cityRenderer);
    frame.visible = true; // warm up in the exact TRUE COLOR state, so the first switch to it doesn't stall
    cityRenderer.warm();
    frame.visible = false;
    return cityRenderer;
}
