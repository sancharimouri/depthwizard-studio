// Unit tests for the 3D measurement tool's pure modules.
//   cd frontend && npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";

import { createTerrain } from "../src/terrain.js";
import { raycastHeightfield, surfaceAt, gridToLocalXY } from "../src/heightfield.js";
import { createGeoGrid, measureSegment, measureChain } from "../src/measure-metrics.js";
import { createMeasureModel, MODES } from "../src/measure-model.js";

// ---------------------------------------------------------------- helpers

function loadRegion(region) {
    return JSON.parse(readFileSync(new URL(`../public/data/${region}/terrain.json`, import.meta.url)));
}

function buildTerrain(region) {
    const data = loadRegion(region);
    const group = new THREE.Group();
    const tex = () => new THREE.Texture();
    const terrain = createTerrain(group, region, data, tex(), tex(), tex());
    group.updateMatrixWorld(true);
    return { terrain, data };
}

// Same field adapter measure-tool.js builds from a live terrain.
function fieldOf(terrain) {
    const pos = terrain.mesh.geometry.attributes.position;
    let zMin = Infinity;
    let zMax = -Infinity;
    for (let i = 0; i < pos.count; i++) {
        zMin = Math.min(zMin, pos.getZ(i));
        zMax = Math.max(zMax, pos.getZ(i));
    }
    const mask = terrain.erodedVertexMask();
    return {
        width: terrain.grid.width,
        height: terrain.grid.height,
        terrainWidth: terrain.terrainWidth,
        terrainHeight: terrain.terrainHeight,
        getZ: i => pos.getZ(i),
        zMin,
        zMax,
        triangleValid: mask ? (a, b, c) => !mask[a] && !mask[b] && !mask[c] : null,
    };
}

function seeded(seed) {
    let s = seed >>> 0;
    return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// Rays from camera-like positions (around and above the tile) toward random
// points on/near it, compared against THREE.Raycaster on the real mesh.
function compareWithThree(terrain, n, seed) {
    const field = fieldOf(terrain);
    const rand = seeded(seed);
    const raycaster = new THREE.Raycaster();
    const inv = new THREE.Matrix4().copy(terrain.mesh.matrixWorld).invert();
    let hits = 0;
    let misses = 0;
    let maxErr = 0;
    for (let k = 0; k < n; k++) {
        const angle = rand() * Math.PI * 2;
        const elev = 0.15 + rand() * 1.2;
        const dist = 90 + rand() * 120;
        const origin = new THREE.Vector3(Math.cos(angle) * dist, Math.sin(elev) * dist, Math.sin(angle) * dist);
        const target = new THREE.Vector3((rand() - 0.5) * 130, rand() * 20, (rand() - 0.5) * 120);
        const dir = target.clone().sub(origin).normalize();

        raycaster.set(origin, dir);
        const threeHit = raycaster.intersectObject(terrain.mesh, false)[0] ?? null;

        const lo = origin.clone().applyMatrix4(inv);
        const ld = origin.clone().add(dir).applyMatrix4(inv).sub(lo);
        const mine = raycastHeightfield(field, lo, ld);

        assert.equal(!!mine, !!threeHit, `hit/miss disagreement on ray ${k}`);
        if (threeHit) {
            hits += 1;
            const world = new THREE.Vector3(mine.x, mine.y, mine.z).applyMatrix4(terrain.mesh.matrixWorld);
            maxErr = Math.max(maxErr, world.distanceTo(threeHit.point));
        } else {
            misses += 1;
        }
    }
    return { hits, misses, maxErr };
}

// Synthetic grid for exact metric checks: 11×11 vertices, 0.001° spacing.
function syntheticGeo(heightFn) {
    const width = 11;
    const height = 11;
    const heights = new Float32Array(width * height);
    for (let r = 0; r < height; r++) {
        for (let c = 0; c < width; c++) {
            heights[r * width + c] = heightFn(c, r);
        }
    }
    return createGeoGrid({
        width, height, heights,
        bounds: { west: 88.0, east: 88.01, south: 27.0, north: 27.01 },
        elevationMin: 0, elevationMax: 100,
    });
}

// ---------------------------------------------------------------- heightfield

test("heightfield raycast matches THREE.Raycaster on the real Darjeeling mesh (3D)", () => {
    const { terrain } = buildTerrain("darjeeling");
    const r = compareWithThree(terrain, 400, 1);
    assert.ok(r.hits > 200 && r.misses > 20, JSON.stringify(r));
    assert.ok(r.maxErr < 1e-3, `max world error ${r.maxErr}`);
});

test("heightfield raycast matches THREE on the flat layer", () => {
    const { terrain } = buildTerrain("darjeeling");
    terrain.setLayer("satellite-flat");
    terrain.mesh.geometry.computeBoundingSphere();
    const r = compareWithThree(terrain, 300, 2);
    assert.ok(r.hits > 100, JSON.stringify(r));
    assert.ok(r.maxErr < 1e-3, `max world error ${r.maxErr}`);
});

test("heightfield raycast respects eroded mesh edges (Kolkata)", () => {
    const { terrain } = buildTerrain("kolkata");
    assert.equal(terrain.erodedVertexMask(), null, "full index until a 3D layer is applied");
    terrain.setLayer("satellite-3d");
    terrain.mesh.geometry.computeBoundingSphere();
    assert.ok(terrain.erodedVertexMask(), "Kolkata's 3D mesh has an eroded boundary");
    const r = compareWithThree(terrain, 400, 3);
    assert.ok(r.hits > 150 && r.misses > 20, JSON.stringify(r));
    assert.ok(r.maxErr < 1e-3, `max world error ${r.maxErr}`);
});

test("surfaceAt returns vertex heights exactly at grid vertices, null outside", () => {
    const { terrain } = buildTerrain("darjeeling");
    const field = fieldOf(terrain);
    const pos = terrain.mesh.geometry.attributes.position;
    for (const [c, r] of [[0, 0], [100, 50], [360, 324], [17, 300]]) {
        assert.ok(Math.abs(surfaceAt(field, c, r).z - pos.getZ(r * 361 + c)) < 1e-9);
        const xy = gridToLocalXY(field, c, r);
        assert.ok(Math.abs(xy.x - pos.getX(r * 361 + c)) < 1e-4 && Math.abs(xy.y - pos.getY(r * 361 + c)) < 1e-4);
    }
    assert.equal(surfaceAt(field, -0.1, 5), null);
    assert.equal(surfaceAt(field, 5, 324.1), null);
});

// ---------------------------------------------------------------- metrics

test("metrics: flat terrain has zero rise, and distances follow the bounds", () => {
    const geo = syntheticGeo(() => 0.5);
    const east = measureSegment(geo, { x: 0, y: 5 }, { x: 10, y: 5 });
    const north = measureSegment(geo, { x: 5, y: 10 }, { x: 5, y: 0 });
    const expectEast = 0.01 * 111320 * Math.cos(27.005 * Math.PI / 180);
    assert.ok(Math.abs(east.horizontal - expectEast) < 1e-6);
    assert.ok(Math.abs(north.horizontal - 0.01 * 111320) < 1e-6);
    assert.equal(east.rise, 0);
    assert.equal(east.gradient, 0);
    assert.ok(Math.abs(east.surface - east.horizontal) < 1e-6);
});

test("metrics: uniform slope gives the exact rise, gradient and surface length", () => {
    const geo = syntheticGeo(c => c / 10); // 0 → 100 m west → east
    const s = measureSegment(geo, { x: 0, y: 3 }, { x: 10, y: 3 });
    assert.ok(Math.abs(s.rise - 100) < 1e-9);
    assert.ok(Math.abs(s.gradient - (100 / s.horizontal) * 100) < 1e-9);
    assert.ok(Math.abs(s.slopeDeg - Math.atan2(100, s.horizontal) * 180 / Math.PI) < 1e-9);
    assert.ok(Math.abs(s.surface - Math.hypot(s.horizontal, 100)) < 1e-6);
    assert.ok(Math.abs(s.chord - s.surface) < 1e-6);
    const back = measureSegment(geo, { x: 10, y: 3 }, { x: 0, y: 3 });
    assert.ok(Math.abs(back.rise + 100) < 1e-9, "descent is negative rise");
});

test("metrics: surface length ≥ chord ≥ horizontal over real Darjeeling relief", () => {
    const geo = createGeoGrid(loadRegion("darjeeling"));
    assert.ok(Math.abs(geo.pixelSizeX - 27.6) < 0.5 && Math.abs(geo.pixelSizeY - 31.3) < 0.5,
        `pixel size ${geo.pixelSizeX} × ${geo.pixelSizeY} m`);
    const s = measureSegment(geo, { x: 20, y: 30 }, { x: 300, y: 260 });
    assert.ok(s.surface >= s.chord - 1e-9 && s.chord >= s.horizontal - 1e-9);
    assert.ok(s.startElevation >= 556 && s.startElevation <= 2478);
});

test("metrics: polygon area and perimeter of a grid-aligned rectangle", () => {
    const geo = syntheticGeo(() => 0);
    const pts = [{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 8, y: 6 }, { x: 2, y: 6 }];
    const m = measureChain(geo, pts, true);
    const w = 6 * geo.pixelSizeX;
    const h = 4 * geo.pixelSizeY;
    assert.equal(m.type, "polygon");
    assert.ok(Math.abs(m.area - w * h) < 1e-6);
    assert.ok(Math.abs(m.perimeter - 2 * (w + h)) < 1e-6);
    const open = measureChain(geo, pts, false);
    assert.equal(open.type, "polyline");
    assert.equal(open.area, undefined);
    assert.equal(measureChain(geo, pts.slice(0, 2), false).type, "segment");
});

// ---------------------------------------------------------------- model: two-point

const P = (x, y) => ({ x, y });

test("two-point: place A, place B, 3rd click clears; off-terrain with 1 point errors", () => {
    const m = createMeasureModel();
    assert.equal(m.click(P(1, 1)).type, "ignored", "normal mode ignores clicks");
    m.setMode(MODES.TWO_POINT);
    const off0 = m.click(null);
    assert.equal(off0.type, "error");
    assert.equal(m.click(P(1, 1)).type, "placed");
    const off1 = m.click(null);
    assert.equal(off1.type, "error");
    assert.match(off1.message, /second point/);
    assert.equal(m.pointCount(), 1, "error did not change the selection");
    assert.equal(m.click(P(5, 5)).complete, true);
    assert.equal(m.click(null).type, "cleared", "3rd click clears even off-terrain");
    assert.equal(m.pointCount(), 0);
    m.click(P(1, 1));
    m.click(P(2, 2));
    assert.equal(m.click(P(9, 9)).type, "cleared", "3rd click on terrain clears too");
});

test("two-point: clicking the placed point again is a duplicate, not a zero-length line", () => {
    const m = createMeasureModel();
    m.setMode(MODES.TWO_POINT);
    m.click(P(1, 1));
    assert.equal(m.click(P(1, 1), { chain: 0, index: 0 }).type, "duplicate");
    assert.equal(m.pointCount(), 1);
});

test("drag: live moves, one undo entry per drag, untouched drag adds none", () => {
    const m = createMeasureModel();
    m.setMode(MODES.TWO_POINT);
    m.click(P(1, 1));
    m.click(P(5, 5));
    const depth = m.undoDepth;
    m.beginDrag({ chain: 0, index: 1 });
    for (let i = 0; i < 10; i++) {
        m.moveDrag({ chain: 0, index: 1 }, 5 + i, 5);
    }
    assert.equal(m.endDrag({ chain: 0, index: 1 }).type, "moved");
    assert.equal(m.undoDepth, depth + 1);
    assert.deepEqual([m.chains[0].points[1].x, m.chains[0].points[1].y], [14, 5]);
    m.beginDrag({ chain: 0, index: 0 });
    assert.equal(m.endDrag({ chain: 0, index: 0 }).type, "unchanged");
    assert.equal(m.undoDepth, depth + 1);
    m.undo();
    assert.deepEqual([m.chains[0].points[1].x, m.chains[0].points[1].y], [5, 5]);
});

test("right-click edits: numeric X/Y, delete point, delete line (two-point)", () => {
    const m = createMeasureModel();
    m.setMode(MODES.TWO_POINT);
    m.click(P(1, 1));
    m.click(P(5, 5));
    assert.equal(m.setPointXY({ chain: 0, index: 0 }, 2, 3).type, "moved");
    assert.deepEqual([m.chains[0].points[0].x, m.chains[0].points[0].y], [2, 3]);
    assert.equal(m.deletePoint({ chain: 0, index: 0 }).type, "deleted-point");
    assert.equal(m.pointCount(), 1);
    assert.equal(m.isComplete(), false, "back to needing a second point");
    m.click(P(7, 7));
    assert.equal(m.deleteSegment({ chain: 0, index: 0 }).type, "deleted-segment");
    assert.equal(m.pointCount(), 0, "the only line of a two-point measurement is the measurement");
    assert.equal(m.deleteSegment({ chain: 0, index: 0 }).type, "error");
});

// ---------------------------------------------------------------- model: continuous

test("continuous: growing chain, close on first point needs ≥3 points, then next click clears", () => {
    const m = createMeasureModel();
    m.setMode(MODES.CONTINUOUS);
    m.click(P(1, 1));
    m.click(P(5, 1));
    assert.equal(m.click(P(1, 1), { chain: 0, index: 0 }).type, "duplicate", "2 points can't close");
    m.click(P(5, 5));
    assert.equal(m.click(P(1, 1), { chain: 0, index: 0 }).type, "closed");
    assert.equal(m.chains[0].closed, true);
    assert.equal(m.isComplete(), true);
    assert.equal(m.click(P(9, 9), { chain: 0, index: 1 }).type, "cleared");
    assert.equal(m.pointCount(), 0);
});

test("continuous: clicking a non-first point is a duplicate; off-terrain errors", () => {
    const m = createMeasureModel();
    m.setMode(MODES.CONTINUOUS);
    m.click(P(1, 1));
    m.click(P(5, 1));
    m.click(P(5, 5));
    assert.equal(m.click(P(5, 1), { chain: 0, index: 1 }).type, "duplicate");
    const e = m.click(null);
    assert.equal(e.type, "error");
    assert.match(e.message, /next point/);
    assert.equal(m.pointCount(), 3);
});

test("continuous: delete a middle segment splits the chain; new points extend the tail", () => {
    const m = createMeasureModel();
    m.setMode(MODES.CONTINUOUS);
    for (const x of [0, 1, 2, 3, 4]) {
        m.click(P(x, 0));
    }
    m.deleteSegment({ chain: 0, index: 1 }); // between points 1 and 2
    assert.equal(m.chains.length, 2);
    assert.deepEqual(m.chains[0].points.map(p => p.x), [0, 1]);
    assert.deepEqual(m.chains[1].points.map(p => p.x), [2, 3, 4]);
    assert.equal(m.activeChain, 1);
    m.click(P(5, 0));
    assert.deepEqual(m.chains[1].points.map(p => p.x), [2, 3, 4, 5]);
});

test("continuous: delete a polygon edge opens it; delete a point in a triangle reopens it", () => {
    const m = createMeasureModel();
    m.setMode(MODES.CONTINUOUS);
    for (const [x, y] of [[0, 0], [4, 0], [4, 4], [0, 4]]) {
        m.click(P(x, y));
    }
    m.click(P(0, 0), { chain: 0, index: 0 });
    assert.equal(m.chains[0].closed, true);
    m.deleteSegment({ chain: 0, index: 3 }); // closing edge (0,4)→(0,0)
    assert.equal(m.chains[0].closed, false);
    assert.deepEqual(m.chains[0].points.map(p => [p.x, p.y]), [[0, 0], [4, 0], [4, 4], [0, 4]]);
    m.undo();
    assert.equal(m.chains[0].closed, true);
    m.deleteSegment({ chain: 0, index: 1 }); // (4,0)→(4,4): open chain starts at (4,4)
    assert.deepEqual(m.chains[0].points.map(p => [p.x, p.y]), [[4, 4], [0, 4], [0, 0], [4, 0]]);

    const t = createMeasureModel();
    t.setMode(MODES.CONTINUOUS);
    for (const [x, y] of [[0, 0], [4, 0], [4, 4]]) {
        t.click(P(x, y));
    }
    t.click(P(0, 0), { chain: 0, index: 0 });
    t.deletePoint({ chain: 0, index: 1 });
    assert.equal(t.chains[0].closed, false, "2 points can't stay a polygon");
    assert.equal(t.activeChain, 0, "and it becomes the growing chain again");
});

test("continuous: deleting an interior point reconnects its neighbours", () => {
    const m = createMeasureModel();
    m.setMode(MODES.CONTINUOUS);
    for (const x of [0, 1, 2]) {
        m.click(P(x, 0));
    }
    m.deletePoint({ chain: 0, index: 1 });
    assert.deepEqual(m.chains[0].points.map(p => p.x), [0, 2]);
});

// ---------------------------------------------------------------- model: esc, undo, modes, save

test("Esc clears everything; a multi-step undo stack restores each state exactly", () => {
    const m = createMeasureModel();
    m.setMode(MODES.CONTINUOUS);
    const states = [];
    const snap = () => JSON.stringify({ chains: m.chains, active: m.activeChain });
    states.push(snap());
    for (const x of [0, 3, 6]) {
        m.click(P(x, x));
        states.push(snap());
    }
    m.beginDrag({ chain: 0, index: 1 });
    m.moveDrag({ chain: 0, index: 1 }, 9, 1);
    m.endDrag({ chain: 0, index: 1 });
    states.push(snap());
    m.deletePoint({ chain: 0, index: 0 });
    states.push(snap());
    assert.equal(m.clear().type, "cleared");
    assert.equal(m.pointCount(), 0);
    for (let i = states.length - 1; i >= 0; i--) {
        assert.equal(m.undo().type, i === states.length - 1 ? "undone" : "undone");
        assert.equal(snap(), states[i], `undo step back to state ${i}`);
    }
    assert.equal(m.undo().type, "empty");
    assert.equal(m.clear().type, "unchanged", "Esc on an empty selection is a no-op");
});

test("modes: pointer keeps the selection; switching sub-mode clears it, and undo brings it back in its own mode", () => {
    const m = createMeasureModel();
    m.setMode(MODES.CONTINUOUS);
    for (const [x, y] of [[0, 0], [4, 0], [4, 4]]) {
        m.click(P(x, y));
    }
    m.click(P(0, 0), { chain: 0, index: 0 });
    m.setMode(MODES.NORMAL);
    assert.equal(m.pointCount(), 3, "pointer mode keeps the polygon on screen");
    assert.equal(m.click(P(9, 9)).type, "ignored");
    m.setMode(MODES.CONTINUOUS);
    assert.equal(m.pointCount(), 3, "same sub-mode: still there");
    m.setMode(MODES.TWO_POINT);
    assert.equal(m.pointCount(), 0, "other sub-mode: cleared");
    const u = m.undo();
    assert.equal(u.mode, MODES.CONTINUOUS);
    assert.equal(m.mode, MODES.CONTINUOUS);
    assert.equal(m.chains[0].closed, true);
});

test("save: needs a line; stores chains with metrics (session-only)", () => {
    const geo = syntheticGeo(c => c / 10);
    const m = createMeasureModel();
    m.setMode(MODES.TWO_POINT);
    assert.equal(m.save((pts, closed) => measureChain(geo, pts, closed)).type, "error");
    m.click(P(0, 3));
    assert.equal(m.save().type, "error", "one point is not a measurement");
    m.click(P(10, 3));
    const r = m.save((pts, closed) => measureChain(geo, pts, closed));
    assert.equal(r.type, "saved");
    assert.equal(m.saved.length, 1);
    assert.ok(Math.abs(r.items[0].metrics.netRise - 100) < 1e-9);
    assert.equal(m.pointCount(), 2, "saving does not clear the selection");
    assert.equal(m.undoDepth, 2, "save is not an undoable edit");
});
