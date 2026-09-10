import * as THREE from "three";


// ============================================================
// EXPERIMENT: Kolkata-only irregular 3D-mesh edge erosion
//
// Darjeeling's tile boundary is naturally irregular (real DSM data
// coverage); Kolkata's hard rectangular edge looks artificial next to
// it. This carves a low-frequency noise-based bite out of the boundary
// of Kolkata's *extruded* (3D) meshes only — DSM / Metric Elevation /
// True Color. Flat 2D layers and every other region are untouched.
//
// Tweak and reload (Vite will pick these up) to adjust the look:
//   SEED   — any integer; changes the exact shape of the bites.
//   SCALE  — low-frequency noise lattice resolution across the tile.
//            Lower = broader, gentler bays. Higher = more, smaller ones.
//   AMOUNT — max fraction of the tile's span eaten away from any edge.
//            Keep this small — it's meant to be a restrained nibble,
//            not a starburst or a rounded-rectangle blob.
// ============================================================

const EDGE_EROSION_REGION = "kolkata";
const EDGE_EROSION_SEED = 1337;
const EDGE_EROSION_NOISE_SCALE = 5;
const EDGE_EROSION_AMOUNT = 0.06;


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

    const exaggerationFactor =
        Math.min(
            10,
            Math.max(
                1,
                1 + 4 * Math.log10(
                    referenceReliefRatio / reliefRatio
                )
            )
        );

    const baseVerticalScale = 0.02;

    const verticalExaggeration =
        baseVerticalScale * exaggerationFactor;


    // Extruded (3D) height per vertex, kept separately from the plane's
    // flat z=0 rest state so layers can toggle between the two without
    // reloading geometry.

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
                heights[index] *
                elevationRange;

            extrudedZ[index] =
                (
                    elevation -
                    elevationMin
                ) *
                verticalExaggeration;

            positions.setZ(
                index,
                extrudedZ[index]
            );
        }
    }


    geometry.computeVertexNormals();


    // Kolkata-only: erode the 3D-mesh boundary (see the block comment at
    // the top of this file). Everywhere else, erodedIndexArray stays null
    // and setExtruded() never touches the index buffer.

    const fullIndexArray = geometry.index.array.slice();
    let erodedIndexArray = null;

    if (regionKey === EDGE_EROSION_REGION) {

        const edgeNoise = createValueNoise2D(EDGE_EROSION_SEED, EDGE_EROSION_NOISE_SCALE);
        const erodedVertex = new Uint8Array(positions.count);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {

                const index = y * width + x;

                const u = x / (width - 1);
                const v = y / (height - 1);

                const edgeDistance = Math.min(u, 1 - u, v, 1 - v);
                const biteDepth = EDGE_EROSION_AMOUNT * edgeNoise(u, v);

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
    ]);


    function setExtruded(extruded) {

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


    return {

        mesh: terrain,

        material,

        setLayer,

        setFloodOverlay,

        elevationMin,

        elevationMax,

        elevationRange,

        exaggerationFactor,

        geographicWidth,

        geographicHeight,

        terrainWidth,

        terrainHeight
    };
}