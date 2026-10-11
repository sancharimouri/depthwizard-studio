// Cosmetic Bengaluru urban model: the TRUE COLOR layer of the demo page's Bangalore region. Lazy-loaded by
// main.js (attachCityStructure) when that region loads. Data: public/cityscape/bengaluru.json, baked by
// scripts/bake_cityscape.py (local metres, x east / y north, origin = box centre).
// Look: a dusk grade. Warm peach blocks deepening to muted mauve, a dusky plum ground, a low golden sun with
// lavender sky fill (the model brings its own lights; main.js turns the scene's off and tone-maps it).
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

const TOP = new THREE.Color("#fff0e8");   // building colour at the roof: warm peach...
const FOOT = new THREE.Color("#897386");  // ...deepening to plum at street level
const EDGE = "#ffd2c0";       // soft warm edge lines
const GROUND = "#716274";     // dusk plum
const STREET = "#ebbdb0";     // glowing rose-peach streets
const PLINTH = "#20171d";
const GROUND_STEP = 5; // m, ground mesh spacing (FABDEM is 10 m; finer so roads follow it)

// Bilinear ground height (m, relative to the box median) at local x, y.
function groundSampler(g) {
    if (g.kind !== "fabdem") return () => 0;
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

function buildBuildings(data, groundAt) {
    const geos = [];
    for (const [h, p] of data.buildings) {
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
    const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(merged, 30),
        new THREE.LineBasicMaterial({ color: EDGE, transparent: true, opacity: 0.14 }),
    );
    const group = new THREE.Group();
    group.add(mesh, edges);
    return group;
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
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        color: STREET, roughness: 0.9, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: -2,
    }));
    mesh.receiveShadow = true;
    return mesh;
}

// Sunset: lavender sky fill from above, plum bounce from below, a low gold-orange sun from the west.
// `unitsPerMetre`: world units per metre of the parent's frame, for the shadow camera (it works in world units).
function addLights(parent, unitsPerMetre) {
    const sky = new THREE.HemisphereLight("#cdbadb", "#4f4052", 1.3);
    sky.position.set(0, 1e6, 0); // its "up" is its world position: far up, so any parent offset can't tilt it
    parent.add(sky);
    const sun = new THREE.DirectionalLight("#ffbf78", 2.8);
    sun.position.set(-520, 340, 240);
    sun.castShadow = true;
    sun.shadow.mapSize.set(4096, 4096);
    const k = unitsPerMetre;
    Object.assign(sun.shadow.camera, { left: -300 * k, right: 300 * k, top: 300 * k, bottom: -300 * k, near: 50 * k, far: 1400 * k });
    sun.shadow.radius = 3;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 1.2 * k; // no shadow speckle on the walls
    parent.add(sun, sun.target); // target in the graph, so it follows the parent's frame
}

// The same model as a demo-region layer (Bangalore's TRUE COLOR on the #/demo region rail): one Object3D in local
// metres (x east, y up, z = -north, origin = box centre) carrying its own lights, to be placed on the terrain by
// the caller. No text is drawn on it (no landmark or road names).
export function buildCityStructure(data, unitsPerMetre) {
    const groundAt = groundSampler(data.ground);
    const object = new THREE.Group();
    object.add(buildGround(data, groundAt), buildRoads(data, groundAt), buildBuildings(data, groundAt));
    addLights(object, unitsPerMetre);
    return { object };
}
