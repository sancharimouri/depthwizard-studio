import * as THREE from "three";


// ============================================================
// EXPERIMENT: irregular 3D-mesh edge erosion (Kolkata, Sundarbans,
// Bardhaman)
//
// Darjeeling's tile boundary is naturally irregular (real DSM data
// coverage); the flat-region tiles' hard rectangular edges look
// artificial next to it. This carves a low-frequency noise-based bite
// out of the boundary of each listed region's *extruded* (3D) meshes
// only — DSM / Metric Elevation / True Color. Flat 2D layers, and any
// region not listed here (Darjeeling), are untouched.
//
// Each region gets its own entry so it can be tweaked or reverted
// (just delete its entry) independently. Reload after editing to see
// the change:
//   seed       — any integer; changes the exact shape of the bites.
//   noiseScale — low-frequency noise lattice resolution across the
//                tile. Lower = broader, gentler bays. Higher = more,
//                smaller ones.
//   amount     — max fraction of the tile's span eaten away from any
//                edge. Keep this small — it's meant to be a
//                restrained nibble, not a starburst or a
//                rounded-rectangle blob.
//
// Kolkata's values are the tuned baseline; Sundarbans and Bardhaman
// started from the same numbers and were nudged from there.
// ============================================================

const EDGE_EROSION_PARAMS = {
    kolkata: {
        seed: 1337,
        noiseScale: 5,
        amount: 0.06,
    },
    sundarbans: {
        // Real Sundarbans coastline is a much busier tangle of tidal
        // creeks than Kolkata's riverbank — a slightly higher noiseScale
        // (more, smaller bays) and a touch more amount reads closer to
        // that than Kolkata's gentler nibble.
        seed: 2701,
        noiseScale: 7,
        amount: 0.08,
    },
    bardhaman: {
        // Rural/agricultural coverage gaps tend to be broader and
        // blockier (field-sized) than a riverbank or tidal coastline —
        // a lower noiseScale gives fewer, broader bays.
        seed: 8161,
        noiseScale: 4,
        amount: 0.06,
    },
};


function createSeededRandom(seed) {
    let state = seed >>> 0;

    return function random() {
        state |= 0;
        state = (state + 0x6D2B79F5) | 0;

        let t = Math.imul(state ^ (state >>> 15), 1 | state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;

        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Smooth, low-frequency 2D value noise: a coarse lattice of random
// values, bilinearly interpolated with a smoothstep easing so the
// result undulates gently instead of looking like static.
function createValueNoise2D(seed, gridSize) {
    const random = createSeededRandom(seed);

    const lattice = [];
    for (let row = 0; row <= gridSize; row++) {
        const values = [];
        for (let col = 0; col <= gridSize; col++) {
            values.push(random());
        }
        lattice.push(values);
    }

    function smoothstep(t) {
        return t * t * (3 - 2 * t);
    }

    return function sample(u, v) {
        const x = u * gridSize;
        const y = v * gridSize;

        const x0 = Math.floor(x);
        const y0 = Math.floor(y);
        const x1 = Math.min(x0 + 1, gridSize);
        const y1 = Math.min(y0 + 1, gridSize);

        const tx = smoothstep(x - x0);
        const ty = smoothstep(y - y0);

        const top = lattice[y0][x0] + (lattice[y0][x1] - lattice[y0][x0]) * tx;
        const bottom = lattice[y1][x0] + (lattice[y1][x1] - lattice[y1][x0]) * tx;

        return top + (bottom - top) * ty;
    };
}


// ============================================================
// EXPERIMENT: spike smoothing (Kolkata, Sundarbans, Bardhaman)
//
// Each of these regions' real relief is subtle relative to its 10km
// footprint, so the automatic vertical-exaggeration factor is high to
// make that relief read as terrain at all. That amplifies single/
// few-vertex elevation outliers in the source Copernicus GLO-30 DSM —
// plausibly structures, which a surface-model DEM bakes into the
// height value at 30m resolution — into sharp needles. Two
// independent, additive passes per region, applied only to that
// region's height field, only before/for the extruded (3D) meshes:
//
//   1. A small-kernel median filter on the raw normalized heights,
//      before exaggeration — removes isolated single-vertex noise/
//      outliers without blurring genuine broader relief.
//   2. A slope cap on the exaggerated (world-space) heights — pulls
//      any vertex down toward "lowest neighbor + cap", iteratively, so
//      a spike that survives the median filter (e.g. a structure a few
//      vertices wide) flattens into a small plateau — reading as a
//      flat-topped block — instead of a point.
//
// Each region gets its own entry (tweak or delete independently).
// Reload after editing to see the change:
//   medianKernelSize   — window width in vertices (must be odd). 3 =
//                        3x3. Larger softens more but starts eating
//                        real relief detail.
//   slopeCap           — max world-space height delta allowed between
//                        adjacent vertices, per iteration. Scale this
//                        against the region's own exaggerated
//                        elevation range, not Kolkata's.
//   slopeCapIterations — how many relaxation passes propagate that
//                        cap outward from a spike's base.
//
// Kolkata's values are the tuned baseline; Sundarbans and Bardhaman
// started from the same numbers and were nudged from there.
// ============================================================

const SPIKE_SMOOTHING_PARAMS = {
    kolkata: {
        medianKernelSize: 3,
        slopeCap: 0.6,
        slopeCapIterations: 4,
    },
    sundarbans: {
        // Waterlogged delta: DSM noise over water/mudflats tends to be
        // noisier than Kolkata's built-up riverbank, so a slightly wider
        // median window helps before the slope cap does its work.
        medianKernelSize: 3,
        slopeCap: 0.5,
        slopeCapIterations: 4,
    },
    bardhaman: {
        // Flat agricultural land with occasional isolated structures
        // (silos, water towers) — same shape of problem as Kolkata's
        // buildings, at lower density, so Kolkata's values as-is.
        medianKernelSize: 3,
        slopeCap: 0.6,
        slopeCapIterations: 4,
    },
};


// Out-of-place median filter over a WxH scalar field (border-clamped).
function medianFilter2D(field, width, height, kernelSize) {
    const radius = Math.floor(kernelSize / 2);
    const output = new Float32Array(field.length);
    const neighborhood = [];

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {

            neighborhood.length = 0;

            for (let ky = -radius; ky <= radius; ky++) {
                const ny = Math.min(height - 1, Math.max(0, y + ky));

                for (let kx = -radius; kx <= radius; kx++) {
                    const nx = Math.min(width - 1, Math.max(0, x + kx));
                    neighborhood.push(field[ny * width + nx]);
                }
            }

            neighborhood.sort((a, b) => a - b);
            output[y * width + x] = neighborhood[Math.floor(neighborhood.length / 2)];
        }
    }

    return output;
}

// Iteratively pulls each vertex down to at most "lowest 4-neighbor +
// maxDelta", in place. Peaks taller than their surroundings by more than
// maxDelta flatten into a plateau instead of keeping a sharp apex; genuine
// gradual slopes (delta already under the cap) are left alone.
function capSlope2D(field, width, height, maxDelta, iterations) {
    for (let iteration = 0; iteration < iterations; iteration++) {

        let changed = false;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {

                const index = y * width + x;
                let lowestNeighbor = Infinity;

                if (x > 0) lowestNeighbor = Math.min(lowestNeighbor, field[index - 1]);
                if (x < width - 1) lowestNeighbor = Math.min(lowestNeighbor, field[index + 1]);
                if (y > 0) lowestNeighbor = Math.min(lowestNeighbor, field[index - width]);
                if (y < height - 1) lowestNeighbor = Math.min(lowestNeighbor, field[index + width]);

                const cap = lowestNeighbor + maxDelta;

                if (field[index] > cap) {
                    field[index] = cap;
                    changed = true;
                }
            }
        }

        if (!changed) {
            break;
        }
    }
}


export function createTerrain(
    parent,
    regionKey,
    terrainData,
    satelliteTexture,
    depthTexture,
    elevationTexture
) {

    const {
        width,
        height,
        heights,
        elevationMin,
        elevationMax,
        bounds
    } = terrainData;


    const geographicWidth =
        bounds.east - bounds.west;

    const geographicHeight =
        bounds.north - bounds.south;

    const aspect =
        geographicWidth / geographicHeight;


    const terrainHeight = 100;

    const terrainWidth =
        terrainHeight * aspect;


    const geometry =
        new THREE.PlaneGeometry(
            terrainWidth,
            terrainHeight,
            width - 1,
            height - 1
        );


    const positions =
        geometry.attributes.position;


    const elevationRange =
        elevationMax - elevationMin;


    // Vertical exaggeration is derived per-region from the real relief
    // ratio (elevation range vs. horizontal footprint in meters), not
    // hardcoded per region. A relief ratio around 0.2 — roughly what
    // Darjeeling's real terrain has — reads as natural mountainous
    // relief at ~1x; flatter regions (Kolkata/Bardhaman/Sundarbans)
    // scale up so their real-but-subtle relief still reads as terrain,
    // clamped to a 1x-10x range so nothing goes flat or comically spiky.

    const metersPerDegLat = 111320;

    const latMidRad =
        ((bounds.north + bounds.south) / 2) *
        (Math.PI / 180);

    const metersPerDegLon =
        metersPerDegLat * Math.cos(latMidRad);

    const footprintWidthMeters =
        geographicWidth * metersPerDegLon;

    const footprintHeightMeters =
        geographicHeight * metersPerDegLat;

    const footprintMeters =
        Math.sqrt(
            footprintWidthMeters *
            footprintHeightMeters
        );

    const reliefRatio =
        elevationRange / footprintMeters;

    const referenceReliefRatio = 0.2;

    const baseVerticalScale = 0.02;

    // Low-relief scenes (cities, deltas, farmland: tens of metres over the
    // tile) would still read as flat under the 10x ceiling, so the factor is
    // also floored to give at least MIN_RELIEF_FRACTION of the tile's
    // north-south extent as displayed relief, up to MAX_AUTO_EXAGGERATION.
    // Steep scenes (Darjeeling ~1900 m) already exceed the floor: unchanged.
    const MIN_RELIEF_FRACTION = 0.15;
    const MAX_AUTO_EXAGGERATION = 60;
    const reliefFloorFactor = elevationRange > 0
        ? (MIN_RELIEF_FRACTION * terrainHeight) / (elevationRange * baseVerticalScale)
        : 1;

    const exaggerationFactor =
        Math.min(
            MAX_AUTO_EXAGGERATION,
            Math.max(
                1,
                Math.min(10, 1 + 4 * Math.log10(referenceReliefRatio / reliefRatio)),
                reliefFloorFactor
            )
        );

    const verticalExaggeration =
        baseVerticalScale * exaggerationFactor;


    // Extruded (3D) height per vertex, kept separately from the plane's
    // flat z=0 rest state so layers can toggle between the two without
    // reloading geometry.

    // Per-region: median-filter the raw height field before exaggeration
    // (see the block comment above) to remove single/few-vertex outliers.
    const spikeSmoothingParams = SPIKE_SMOOTHING_PARAMS[regionKey];

    const heightsForMesh = spikeSmoothingParams
        ? medianFilter2D(heights, width, height, spikeSmoothingParams.medianKernelSize)
        : heights;

    const extrudedZ = new Float32Array(positions.count);

    for (
        let y = 0;
        y < height;
        y++
    ) {

        for (
            let x = 0;
            x < width;
            x++
        ) {

            const index =
                y * width + x;

            const elevation =
                elevationMin +
                heightsForMesh[index] *
                elevationRange;

            extrudedZ[index] =
                (
                    elevation -
                    elevationMin
                ) *
                verticalExaggeration;
        }
    }

    // Per-region: cap the world-space slope between adjacent vertices so
    // any spike the median filter didn't fully absorb flattens into a
    // small plateau instead of keeping a sharp apex.
    if (spikeSmoothingParams) {
        capSlope2D(
            extrudedZ,
            width,
            height,
            spikeSmoothingParams.slopeCap,
            spikeSmoothingParams.slopeCapIterations
        );
    }

    for (
        let index = 0;
        index < positions.count;
        index++
    ) {
        positions.setZ(index, extrudedZ[index]);
    }


    geometry.computeVertexNormals();


    // Per-region: erode the 3D-mesh boundary (see the block comment at
    // the top of this file). Everywhere else, erodedIndexArray stays null
    // and setExtruded() never touches the index buffer.

    const fullIndexArray = geometry.index.array.slice();
    let erodedIndexArray = null;
    // Kept (not block-scoped) so the measurement tool can tell which cells
    // the eroded 3D mesh actually has triangles for.
    let erodedVertexMask = null;
    let isExtrudedNow = true;

    const edgeErosionParams = EDGE_EROSION_PARAMS[regionKey];

    if (edgeErosionParams) {

        const edgeNoise = createValueNoise2D(edgeErosionParams.seed, edgeErosionParams.noiseScale);
        const erodedVertex = new Uint8Array(positions.count);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {

                const index = y * width + x;

                const u = x / (width - 1);
                const v = y / (height - 1);

                const edgeDistance = Math.min(u, 1 - u, v, 1 - v);
                const biteDepth = edgeErosionParams.amount * edgeNoise(u, v);

                if (edgeDistance < biteDepth) {
                    erodedVertex[index] = 1;
                }
            }
        }

        const keptIndices = [];
        for (let i = 0; i < fullIndexArray.length; i += 3) {
            const a = fullIndexArray[i];
            const b = fullIndexArray[i + 1];
            const c = fullIndexArray[i + 2];

            if (!erodedVertex[a] && !erodedVertex[b] && !erodedVertex[c]) {
                keptIndices.push(a, b, c);
            }
        }

        erodedIndexArray = new fullIndexArray.constructor(keptIndices);
        erodedVertexMask = erodedVertex;
    }


    // Flood-overlay vertex colors. Default is neutral white (no tint);
    // setFloodOverlay() reddens low-lying vertices on top of whatever
    // texture is currently mapped.

    const vertexColors = new Float32Array(positions.count * 3).fill(1);
    geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(vertexColors, 3)
    );


    // Texture setup

    for (
        const texture of [
            satelliteTexture,
            depthTexture,
            elevationTexture
        ]
    ) {

        texture.minFilter =
            THREE.LinearFilter;

        texture.magFilter =
            THREE.LinearFilter;

        texture.flipY = true;
    }


    satelliteTexture.colorSpace =
        THREE.SRGBColorSpace;

    elevationTexture.colorSpace =
        THREE.SRGBColorSpace;


    const material =
        new THREE.MeshStandardMaterial({

            map: satelliteTexture,

            vertexColors: true,

            roughness: 0.95,

            metalness: 0,

            side: THREE.DoubleSide
        });


    const terrain =
        new THREE.Mesh(
            geometry,
            material
        );


    terrain.rotation.x =
        -Math.PI / 2;


    terrain.receiveShadow = true;

    terrain.castShadow = true;


    parent.add(terrain);


    // Six visualization states: the top row is the same three textures
    // shown flat (no extrusion), the bottom row is those textures draped
    // over the real extruded terrain.

    const LAYER_TEXTURES = {
        "satellite-flat": satelliteTexture,
        "depth-flat": depthTexture,
        "elevation-flat": elevationTexture,
        "dsm-3d": depthTexture,
        "elevation-3d": elevationTexture,
        "satellite-3d": satelliteTexture,
    };

    const EXTRUDED_LAYERS = new Set([
        "dsm-3d",
        "elevation-3d",
        "satellite-3d",
        "wireframe-3d",
    ]);

    // Wireframe view — replicates the reference SIH26175 repo
    // (external/SIH26175-DepthWizard-notvishuuuu: utils/terrainMesh.js +
    // components/TerrainMesh.jsx; read, not run): the heightfield is
    // bilinearly resampled to 256 samples on its long side ("medium"), each
    // cell is split into two triangles (a,c,b) + (b,c,d) — so the diagonals
    // show — and drawn with MeshBasicMaterial({ wireframe: true }) at 0.85
    // opacity (1 px WebGL lines). Colour: the app's green token (--neon-rgb)
    // instead of their purple. No solid fill: the textured mesh is hidden.
    // Built lazily from the displayed surface (extrudedZ); a child of the
    // mesh, so it follows the display exaggeration. Picking is unaffected
    // (it uses the heightfield, not materials).
    const WIREFRAME_RESOLUTION = 256;
    let wireframe = null;

    function neonColor() {
        const raw = typeof document !== "undefined"
            ? getComputedStyle(document.documentElement).getPropertyValue("--neon-rgb")
            : "";
        const [r, g, b] = raw.split(",").map(Number);
        // CSS rgb() strings are parsed as sRGB (raw 0-1 components would be linear and wash out)
        return new THREE.Color(Number.isFinite(b) ? `rgb(${r}, ${g}, ${b})` : "#5e9872");
    }

    function buildWireframe() {
        const longSide = Math.max(width, height);
        const meshW = Math.max(2, Math.round(WIREFRAME_RESOLUTION * (width / longSide)));
        const meshH = Math.max(2, Math.round(WIREFRAME_RESOLUTION * (height / longSide)));
        const xRatio = (width - 1) / (meshW - 1);
        const yRatio = (height - 1) / (meshH - 1);
        const positionsOut = new Float32Array(meshW * meshH * 3);
        for (let y = 0; y < meshH; y++) {
            const sy = y * yRatio;
            const y0 = Math.floor(sy);
            const y1 = Math.min(y0 + 1, height - 1);
            const fy = sy - y0;
            for (let x = 0; x < meshW; x++) {
                const sx = x * xRatio;
                const x0 = Math.floor(sx);
                const x1 = Math.min(x0 + 1, width - 1);
                const fx = sx - x0;
                const top = extrudedZ[y0 * width + x0] + (extrudedZ[y0 * width + x1] - extrudedZ[y0 * width + x0]) * fx;
                const bottom = extrudedZ[y1 * width + x0] + (extrudedZ[y1 * width + x1] - extrudedZ[y1 * width + x0]) * fx;
                const i = (y * meshW + x) * 3;
                positionsOut[i] = (x / (meshW - 1) - 0.5) * terrainWidth; // same local frame as the PlaneGeometry
                positionsOut[i + 1] = (0.5 - y / (meshH - 1)) * terrainHeight;
                positionsOut[i + 2] = top + (bottom - top) * fy;
            }
        }
        const indices = new Uint32Array((meshW - 1) * (meshH - 1) * 6);
        let k = 0;
        for (let y = 0; y < meshH - 1; y++) {
            for (let x = 0; x < meshW - 1; x++) {
                const a = y * meshW + x;
                const b = a + 1;
                const c = a + meshW;
                const d = c + 1;
                indices[k++] = a; indices[k++] = c; indices[k++] = b;
                indices[k++] = b; indices[k++] = c; indices[k++] = d;
            }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(positionsOut, 3));
        g.setIndex(new THREE.BufferAttribute(indices, 1));
        const wire = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
            color: neonColor(),
            wireframe: true,
            transparent: true,
            opacity: 0.85,
        }));
        wire.visible = false;
        terrain.add(wire);
        return wire;
    }


    function setExtruded(extruded) {

        isExtrudedNow = extruded;

        for (
            let index = 0;
            index < positions.count;
            index++
        ) {

            positions.setZ(
                index,
                extruded ? extrudedZ[index] : 0
            );
        }

        positions.needsUpdate = true;

        if (erodedIndexArray) {
            geometry.setIndex(
                new THREE.BufferAttribute(
                    extruded ? erodedIndexArray : fullIndexArray,
                    1
                )
            );
        }

        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
    }


    function setLayer(layer) {

        material.map =
            LAYER_TEXTURES[layer] ?? satelliteTexture;

        material.color.set(0xffffff);
        material.needsUpdate = true;

        const isWire = layer === "wireframe-3d";
        if (isWire && !wireframe) {
            wireframe = buildWireframe();
        }
        if (wireframe) {
            wireframe.visible = isWire;
        }
        material.visible = !isWire;

        setExtruded(EXTRUDED_LAYERS.has(layer));
    }


    // Flood overlay: reddens genuinely low-lying vertices via a smooth
    // (non-hard-cutoff) gradient, multiplied on top of whatever texture
    // is currently mapped. The threshold is an elevation percentile
    // computed from this region's actual distribution — not a fixed
    // normalized-elevation number — so coverage stays ~25-35% of the
    // surface regardless of how relief is distributed in a given region.

    function computePercentile(values, percentile) {
        const sorted = [...values].sort((a, b) => a - b);
        const index = Math.min(
            sorted.length - 1,
            Math.max(0, Math.floor(percentile * (sorted.length - 1)))
        );
        return sorted[index];
    }

    const FLOOD_COVERAGE_PERCENTILE = 0.3;
    // Guard against a degenerate zero-width ramp (e.g. a region where the
    // lowest 30% of vertices all sit at the same minimum elevation).
    const floodLevel = Math.max(computePercentile(heights, FLOOD_COVERAGE_PERCENTILE), 1e-6);

    function smoothstep(x, edge0, edge1) {
        const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
        return t * t * (3 - 2 * t);
    }

    function setFloodOverlay(active) {

        const colorAttribute = geometry.attributes.color;

        for (
            let index = 0;
            index < positions.count;
            index++
        ) {

            if (!active) {
                colorAttribute.setXYZ(index, 1, 1, 1);
                continue;
            }

            const normalizedElevation = heights[index];
            const aboveFloodLevel = smoothstep(normalizedElevation, 0, floodLevel);
            const redAmount = 1 - aboveFloodLevel;

            colorAttribute.setXYZ(
                index,
                1,
                1 - redAmount * 0.75,
                1 - redAmount * 0.75
            );
        }

        colorAttribute.needsUpdate = true;
    }


    // EARTHQUAKE — ILLUSTRATIVE PLACEHOLDER, NOT A SEISMIC MODEL.
    // There is no earthquake/seismic hazard model anywhere in this project.
    // This paints a red gradient on the steepest ground (slope from the raw
    // DEM, in degrees) purely so the scenario UI has something to show —
    // steep slopes are a common *input* to landslide/shaking susceptibility,
    // but this is a mocked visual, not a hazard result. The UI labels it so.

    let placeholderSlopeDanger = null;

    function computePlaceholderSlopeDanger() {
        const cellX = footprintWidthMeters / (width - 1);
        const cellY = footprintHeightMeters / (height - 1);
        const slopeDeg = new Float32Array(width * height);
        const at = (x, y) => elevationMin + heights[
            Math.min(height - 1, Math.max(0, y)) * width + Math.min(width - 1, Math.max(0, x))
        ] * elevationRange;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const gx = (at(x + 1, y) - at(x - 1, y)) / (2 * cellX);
                const gy = (at(x, y + 1) - at(x, y - 1)) / (2 * cellY);
                slopeDeg[y * width + x] = Math.atan(Math.hypot(gx, gy)) * 180 / Math.PI;
            }
        }
        // Relative within the tile: gentle below the median slope, full red
        // from the 95th percentile up.
        const lo = computePercentile(slopeDeg, 0.5);
        const hi = Math.max(computePercentile(slopeDeg, 0.95), lo + 1e-3);
        const danger = new Float32Array(slopeDeg.length);
        for (let i = 0; i < danger.length; i++) {
            danger[i] = smoothstep(slopeDeg[i], lo, hi);
        }
        return danger;
    }

    function setEarthquakeOverlay(active) {

        const colorAttribute = geometry.attributes.color;

        if (active && !placeholderSlopeDanger) {
            placeholderSlopeDanger = computePlaceholderSlopeDanger();
        }

        for (let index = 0; index < positions.count; index++) {
            if (!active) {
                colorAttribute.setXYZ(index, 1, 1, 1);
                continue;
            }
            const d = placeholderSlopeDanger[index];
            colorAttribute.setXYZ(index, 1, 1 - d * 0.85, 1 - d * 0.85);
        }

        colorAttribute.needsUpdate = true;
    }


    // DISPLAY-ONLY vertical exaggeration (expanded view's slider). It is a
    // render-time scale on the mesh's local Z (world Y after the -90° X
    // rotation): vertex positions, `heights`, and everything measured from
    // `grid` are untouched, so Measure / terrain stats / exports keep real,
    // un-exaggerated values. Picking still works because the measure tool
    // maps rays through mesh.matrixWorld, which includes this scale.
    //
    // Units match the existing "exaggeration 1.07x" readout (the auto factor
    // above). Range is per-terrain: the max keeps the displayed relief within
    // ~80% of the tile's north-south extent, so a flat scene (tens of metres
    // of relief) can go far higher than an already-steep one.
    const DISPLAY_RELIEF_CAP = 0.8 * terrainHeight;
    const reliefUnitsPerFactor = Math.max(elevationRange * baseVerticalScale, 1e-6);
    const maxDisplayExaggeration = Math.max(
        exaggerationFactor * 1.5,
        Math.min(50, DISPLAY_RELIEF_CAP / reliefUnitsPerFactor)
    );
    const minDisplayExaggeration = Math.min(0.25, exaggerationFactor);
    let displayExaggeration = exaggerationFactor;

    function setDisplayExaggeration(factor) {
        displayExaggeration = Math.min(maxDisplayExaggeration, Math.max(minDisplayExaggeration, factor));
        terrain.scale.z = displayExaggeration / exaggerationFactor;
        terrain.updateMatrixWorld(true);
        return displayExaggeration;
    }

    return {

        mesh: terrain,

        setDisplayExaggeration,

        displayExaggeration: () => displayExaggeration,

        minDisplayExaggeration,

        maxDisplayExaggeration,

        // Local mesh Z of a real elevation (metres), before the display scale.
        localZForElevation: elevationM => (elevationM - elevationMin) * verticalExaggeration,

        footprintWidthMeters,

        footprintHeightMeters,

        material,

        setLayer,

        setFloodOverlay,

        setEarthquakeOverlay,

        elevationMin,

        elevationMax,

        elevationRange,

        exaggerationFactor,

        geographicWidth,

        geographicHeight,

        terrainWidth,

        terrainHeight,

        // Read-only access for the measurement tool (measure-tool.js): the
        // raw DEM grid (unsmoothed, un-exaggerated) and the mesh's current
        // state (flat vs. extruded, which vertices the eroded mesh dropped).
        grid: { width, height, heights, bounds, elevationMin, elevationMax },

        isExtruded: () => isExtrudedNow,

        // Mask only while the eroded index buffer is the one actually drawn
        // (it's swapped in by setExtruded(true), not at construction).
        erodedVertexMask: () =>
            (erodedIndexArray && geometry.index?.array === erodedIndexArray ? erodedVertexMask : null)
    };
}