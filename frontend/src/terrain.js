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


    const verticalExaggeration =
        0.02;


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

            positions.setZ(
                index,
                (
                    elevation -
                    elevationMin
                ) *
                verticalExaggeration
            );
        }
    }


    geometry.computeVertexNormals();


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


    function setLayer(layer) {

        if (layer === "depth") {

            material.map =
                depthTexture;

            material.color.set(
                0xffffff
            );

        }

        else if (layer === "elevation") {

            material.map =
                elevationTexture;

            material.color.set(
                0xffffff
            );

        }

        else {

            material.map =
                satelliteTexture;

            material.color.set(
                0xffffff
            );
        }


        material.needsUpdate = true;
    }


    return {

        mesh: terrain,

        material,

        setLayer,

        elevationMin,

        elevationMax,

        elevationRange,

        geographicWidth,

        geographicHeight,

        terrainWidth,

        terrainHeight
    };
}