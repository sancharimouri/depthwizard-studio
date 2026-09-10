import * as THREE from "three";


export function createTerrain(
    scene,
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


    scene.add(terrain);


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
    // is currently mapped.

    const FLOOD_LEVEL = 0.35;

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
            const aboveFloodLevel = smoothstep(normalizedElevation, 0, FLOOD_LEVEL);
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