import * as THREE from "three";

/**
 * DepthWizard2 cinematic terrain flythrough.
 *
 * IMPORTANT:
 * - This module ONLY moves the camera. It never rotates/scales/translates terrain.
 * - The first shot is framed from the terrain's actual world-space bounds.
 * - The animation is a real camera flight:
 *
 *      establishing orbit
 *            ↓
 *      smooth inward transition
 *            ↓
 *      low terrain flight
 *            ↓
 *      pull-up / wide reveal
 *
 * - Terrain height is sampled once while constructing the flight path.
 * - update() only evaluates the already-built camera path.
 */
export function createFlythrough(camera, scene) {
    let active = false;
    let startTime = 0;

    // ------------------------------------------------------------
    // Timing
    // ------------------------------------------------------------

    const ORBIT_MS = 3000;
    const TRANSITION_MS = 1500;
    const FLIGHT_MS = 6500;
    const REVEAL_MS = 1800;

    const TOTAL_MS =
        ORBIT_MS +
        TRANSITION_MS +
        FLIGHT_MS +
        REVEAL_MS;

    // ------------------------------------------------------------
    // Terrain state
    // ------------------------------------------------------------

    const raycaster = new THREE.Raycaster();

    let terrainMesh = null;
    let terrainBox = new THREE.Box3();
    let terrainCenter = new THREE.Vector3();

    let path = null;

    let segments = [];

    let finalPosition = new THREE.Vector3();
    let finalTarget = new THREE.Vector3();

    // Temporary vectors reused to reduce allocations.
    const tmpA = new THREE.Vector3();
    const tmpB = new THREE.Vector3();
    const tmpC = new THREE.Vector3();

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    function clamp01(t) {
        return Math.min(1, Math.max(0, t));
    }

    function smoothstep(t) {
        t = clamp01(t);
        return t * t * (3 - 2 * t);
    }

    function smootherstep(t) {
        t = clamp01(t);
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    function easeInOutCubic(t) {
        t = clamp01(t);

        return t < 0.5
            ? 4 * t * t * t
            : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    // ------------------------------------------------------------
    // Find actual terrain
    // ------------------------------------------------------------

    function discoverTerrain() {
        const meshes = [];

        scene.traverse((object) => {
            if (object.isMesh && object.visible) {
                meshes.push(object);
            }
        });

        if (!meshes.length) {
            console.warn("Flythrough: no visible mesh found.");
            return false;
        }

        // Pick the mesh with the largest horizontal footprint.
        // This is more reliable than bounding-box volume when helper
        // objects have unusual vertical dimensions.
        let best = null;
        let bestArea = -Infinity;

        const box = new THREE.Box3();
        const size = new THREE.Vector3();

        for (const mesh of meshes) {
            box.setFromObject(mesh);
            box.getSize(size);

            const area = Math.abs(size.x * size.z);

            if (area > bestArea) {
                bestArea = area;
                best = mesh;
            }
        }

        terrainMesh = best;

        terrainBox = new THREE.Box3().setFromObject(terrainMesh);
        terrainBox.getCenter(terrainCenter);

        return true;
    }

    // ------------------------------------------------------------
    // Terrain height sampling
    // ------------------------------------------------------------

    function sampleTerrainHeight(x, z) {
        const size = terrainBox.getSize(tmpA);

        const startY =
            terrainBox.max.y +
            Math.max(100, size.y + 100);

        raycaster.set(
            new THREE.Vector3(x, startY, z),
            new THREE.Vector3(0, -1, 0)
        );

        raycaster.near = 0;

        raycaster.far = Math.max(
            1000,
            startY - terrainBox.min.y + 100
        );

        const hits = raycaster.intersectObject(
            terrainMesh,
            true
        );

        if (hits.length > 0) {
            return hits[0].point.y;
        }

        // Safe fallback.
        return terrainBox.min.y;
    }

    function insideFootprint(x, z, margin = 0) {
        return (
            x >= terrainBox.min.x + margin &&
            x <= terrainBox.max.x - margin &&
            z >= terrainBox.min.z + margin &&
            z <= terrainBox.max.z - margin
        );
    }

    // ------------------------------------------------------------
    // Build terrain-following flight points
    // nx/nz are normalized coordinates in [-1, 1].
    // ------------------------------------------------------------

    function makeFlightPoint(nx, nz, clearance) {
        const width =
            terrainBox.max.x - terrainBox.min.x;

        const depth =
            terrainBox.max.z - terrainBox.min.z;

        const x =
            terrainCenter.x +
            nx * width * 0.5;

        const z =
            terrainCenter.z +
            nz * depth * 0.5;

        const surfaceY =
            sampleTerrainHeight(x, z);

        return new THREE.Vector3(
            x,
            surfaceY + clearance,
            z
        );
    }

    // ------------------------------------------------------------
    // Calculate camera distance that keeps terrain framed
    // ------------------------------------------------------------

    function getFitDistance() {
        const size = terrainBox.getSize(tmpA);

        const halfW = size.x * 0.5;
        const halfD = size.z * 0.5;
        const halfH = size.y * 0.5;

        const radius = Math.sqrt(
            halfW * halfW +
            halfD * halfD +
            halfH * halfH
        );

        const verticalFov =
            THREE.MathUtils.degToRad(camera.fov);

        const distance =
            radius /
            Math.tan(verticalFov * 0.5);

        // Extra margin prevents cropping at laptop aspect ratios.
        return Math.max(
            distance * 1.18,
            Math.max(size.x, size.z) * 1.15
        );
    }

    // ------------------------------------------------------------
    // Phase 1:
    // Establishing aerial orbit.
    //
    // This is intentionally NOT a giant 210° / 360° turn.
    // The viewer should understand the terrain immediately.
    // ------------------------------------------------------------

    function makeOrbitCurve() {
        const size = terrainBox.getSize(tmpA);

        const fit = getFitDistance();

        // Keep the camera elevated enough to see the whole terrain.
        const orbitY =
            terrainCenter.y +
            Math.max(
                size.y * 0.85,
                fit * 0.20
            );

        const radius =
            Math.max(
                fit * 0.96,
                Math.max(size.x, size.z) * 0.95
            );

        // Roughly 112 degrees.
        const startAngle =
            Math.PI * 0.5;

        const sweep =
            THREE.MathUtils.degToRad(112);

        const points = [];

        for (let i = 0; i <= 18; i++) {
            const t = i / 18;

            const angle =
                startAngle -
                sweep * t;

            // Very small vertical movement.
            // This keeps the shot cinematic without looking like
            // the entire mountain is rotating.
            const y =
                orbitY -
                fit *
                    0.035 *
                    smootherstep(t);

            points.push(
                new THREE.Vector3(
                    terrainCenter.x +
                        Math.cos(angle) * radius,

                    y,

                    terrainCenter.z +
                        Math.sin(angle) * radius
                )
            );
        }

        const curve =
            new THREE.CatmullRomCurve3(
                points,
                false,
                "centripetal",
                0.35
            );

        curve.arcLengthDivisions = 300;
        curve.updateArcLengths();

        return curve;
    }

    // ------------------------------------------------------------
    // Phase 3:
    // Actual flight through the landscape.
    //
    // The path stays inside the terrain footprint.
    // Heights are sampled from the actual terrain surface.
    // ------------------------------------------------------------

    function makeFlightCurve() {
        const size = terrainBox.getSize(tmpA);

        const verticalRange =
            Math.max(1, size.y);

        const horizontalSpan =
            Math.max(size.x, size.z);

        // Cinematic clearance.
        //
        // We don't want a tiny clearance because the camera would
        // feel like it is scraping the ground.
        //
        // We also don't want huge clearance because then it stops
        // feeling like a terrain flight.
        const base = Math.max(
            14,
            Math.min(
                horizontalSpan * 0.10,
                verticalRange * 0.28 + 12
            )
        );

        const points = [
            // Entry.
            makeFlightPoint(
                0.68,
                0.62,
                base * 1.25
            ),

            // Descend toward first valley.
            makeFlightPoint(
                0.43,
                0.30,
                base * 0.78
            ),

            // Low central pass.
            makeFlightPoint(
                0.10,
                0.02,
                base * 0.92
            ),

            // Rise over central ridge.
            makeFlightPoint(
                -0.25,
                -0.20,
                base * 1.18
            ),

            // Lower again over far terrain.
            makeFlightPoint(
                -0.56,
                -0.46,
                base * 0.82
            ),

            // Begin climbing toward final reveal.
            makeFlightPoint(
                -0.28,
                -0.68,
                base * 1.35
            ),
        ];

        let curve =
            new THREE.CatmullRomCurve3(
                points,
                false,
                "centripetal",
                0.25
            );

        // --------------------------------------------------------
        // Safety pass.
        //
        // Catmull-Rom curves can overshoot between points.
        // Sample the entire path and lift anything that becomes
        // too close to the real terrain surface.
        // --------------------------------------------------------

        const safePoints = [];

        const samples = 180;

        const minimumClearance =
            Math.max(
                10,
                base * 0.58
            );

        for (let i = 0; i <= samples; i++) {
            const t = i / samples;

            const p =
                curve
                    .getPointAt(t)
                    .clone();

            if (
                insideFootprint(
                    p.x,
                    p.z,
                    0
                )
            ) {
                const floor =
                    sampleTerrainHeight(
                        p.x,
                        p.z
                    ) +
                    minimumClearance;

                if (p.y < floor) {
                    p.y = floor;
                }
            }

            safePoints.push(p);
        }

        curve =
            new THREE.CatmullRomCurve3(
                safePoints,
                false,
                "centripetal",
                0.15
            );

        curve.arcLengthDivisions = 400;
        curve.updateArcLengths();

        return curve;
    }

    // ------------------------------------------------------------
    // Phase 2:
    // Transition from orbit into flight.
    //
    // Cubic Bezier lets us explicitly control the tangent so there
    // is no ugly direction change at the handoff.
    // ------------------------------------------------------------

    function makeBridgeCurve(
        orbitCurve,
        flightCurve
    ) {
        const start =
            orbitCurve.getPointAt(1);

        const orbitTangent =
            orbitCurve
                .getTangentAt(1)
                .normalize();

        const end =
            flightCurve.getPointAt(0);

        const flightTangent =
            flightCurve
                .getTangentAt(0)
                .normalize();

        const distance =
            start.distanceTo(end);

        const handle =
            Math.max(
                distance * 0.38,
                1
            );

        const c1 =
            start
                .clone()
                .addScaledVector(
                    orbitTangent,
                    handle
                );

        const c2 =
            end
                .clone()
                .addScaledVector(
                    flightTangent,
                    -handle
                );

        return new THREE.CubicBezierCurve3(
            start,
            c1,
            c2,
            end
        );
    }

    // ------------------------------------------------------------
    // Phase 4:
    // Pull up and reveal the entire terrain again.
    // ------------------------------------------------------------

    function makeRevealCurve(
        flightCurve
    ) {
        const start =
            flightCurve.getPointAt(1);

        const tangent =
            flightCurve
                .getTangentAt(1)
                .normalize();

        const fit =
            getFitDistance();

        const size =
            terrainBox.getSize(tmpA);

        // Final camera position.
        const end =
            new THREE.Vector3(
                terrainCenter.x -
                    size.x * 0.78,

                terrainCenter.y +
                    fit * 0.48,

                terrainCenter.z -
                    size.z * 0.98
            );

        const distance =
            start.distanceTo(end);

        const c1 =
            start
                .clone()
                .addScaledVector(
                    tangent,
                    Math.max(
                        distance * 0.42,
                        fit * 0.18
                    )
                );

        // Point back toward the terrain.
        const toCenter =
            terrainCenter
                .clone()
                .sub(end)
                .normalize();

        const c2 =
            end
                .clone()
                .addScaledVector(
                    toCenter,
                    Math.max(
                        distance * 0.34,
                        fit * 0.15
                    )
                );

        finalPosition.copy(end);
        finalTarget.copy(terrainCenter);

        return new THREE.CubicBezierCurve3(
            start,
            c1,
            c2,
            end
        );
    }

    // ------------------------------------------------------------
    // Build the complete camera path.
    // ------------------------------------------------------------

    function buildPath() {
        if (!discoverTerrain()) {
            return false;
        }

        const orbit =
            makeOrbitCurve();

        const flight =
            makeFlightCurve();

        const bridge =
            makeBridgeCurve(
                orbit,
                flight
            );

        const reveal =
            makeRevealCurve(
                flight
            );

        path =
            new THREE.CurvePath();

        path.add(orbit);
        path.add(bridge);
        path.add(flight);
        path.add(reveal);

        path.arcLengthDivisions = 800;
        path.updateArcLengths();

        // --------------------------------------------------------
        // Calculate arc-length boundaries for each curve.
        //
        // This is NOT used to move the camera.
        // It is only used to know which phase we're in so that
        // lookAt behaviour can transition correctly.
        // --------------------------------------------------------

        const lengths =
            path.getCurveLengths();

        const total =
            lengths[lengths.length - 1];

        segments =
            lengths.map(
                (length, index) => ({
                    index,
                    u: length / total,
                })
            );

        return true;
    }

    // ------------------------------------------------------------
    // Find current path segment.
    // ------------------------------------------------------------

    function segmentAt(u) {
        for (
            let i = 1;
            i < segments.length;
            i++
        ) {
            if (
                u <= segments[i].u
            ) {
                return {
                    index: i - 1,

                    local:
                        (u -
                            segments[i - 1].u) /
                        Math.max(
                            segments[i].u -
                                segments[i - 1].u,
                            1e-6
                        ),
                };
            }
        }

        return {
            index:
                segments.length - 2,
            local: 1,
        };
    }

    // ------------------------------------------------------------
    // Look target during the actual flight.
    //
    // We don't simply look at the terrain center because that makes
    // a moving camera feel like it's orbiting instead of travelling.
    // ------------------------------------------------------------

    function getFlightLookTarget(
        localU
    ) {
        const curves =
            path.curves;

        // Curve index:
        //
        // 0 = orbit
        // 1 = bridge
        // 2 = terrain flight
        // 3 = reveal
        //
        const flightCurve =
            curves[2];

        const ahead =
            Math.min(
                1,
                localU + 0.055
            );

        const p =
            flightCurve.getPointAt(
                ahead,
                tmpB
            );

        const tangent =
            flightCurve
                .getTangentAt(
                    ahead,
                    tmpC
                )
                .normalize();

        const target =
            p.clone();

        // Look slightly ahead of the camera.
        target.addScaledVector(
            tangent,
            12
        );

        // Slight downward bias.
        target.y -= 8;

        return target;
    }

    // ------------------------------------------------------------
    // Compute look target based on cinematic phase.
    // ------------------------------------------------------------

    function getLookTarget(u) {
        const info =
            segmentAt(u);

        // --------------------------------------------------------
        // Segment 0: Establishing orbit
        // --------------------------------------------------------

        if (info.index === 0) {
            return terrainCenter.clone();
        }

        // --------------------------------------------------------
        // Segment 1: Transition
        //
        // Gradually change from looking at the terrain center to
        // looking in the flight direction.
        // --------------------------------------------------------

        if (info.index === 1) {
            const flightTarget =
                getFlightLookTarget(0);

            const t =
                smootherstep(
                    info.local
                );

            return terrainCenter
                .clone()
                .lerp(
                    flightTarget,
                    t
                );
        }

        // --------------------------------------------------------
        // Segment 2: Flight
        // --------------------------------------------------------

        if (info.index === 2) {
            return getFlightLookTarget(
                info.local
            );
        }

        // --------------------------------------------------------
        // Segment 3: Final reveal
        //
        // Gradually return gaze toward the whole terrain.
        // --------------------------------------------------------

        const flightTarget =
            getFlightLookTarget(1);

        const t =
            smoothstep(
                info.local
            );

        return flightTarget
            .clone()
            .lerp(
                terrainCenter,
                t
            );
    }

    // ------------------------------------------------------------
    // Start / replay
    // ------------------------------------------------------------

    function start() {
        if (!buildPath()) {
            return;
        }

        active = false;

        // Reset camera to the exact beginning of the cinematic.
        const startPoint =
            path.getPointAt(0);

        camera.position.copy(
            startPoint
        );

        camera.lookAt(
            terrainCenter
        );

        startTime =
            performance.now();

        active = true;
    }

    // ------------------------------------------------------------
    // Per-frame update
    // ------------------------------------------------------------

    function update() {
        if (
            !active ||
            !path
        ) {
            return false;
        }

        const elapsed =
            performance.now() -
            startTime;

        const raw =
            clamp01(
                elapsed /
                    TOTAL_MS
            );

        // Global ease makes the entire cinematic feel deliberate
        // rather than mechanically linear.
        const t =
            easeInOutCubic(
                raw
            );

        // --------------------------------------------------------
        // Position
        // --------------------------------------------------------

        const point =
            path.getPointAt(
                t,
                tmpA
            );

        camera.position.copy(
            point
        );

        // --------------------------------------------------------
        // Orientation
        // --------------------------------------------------------

        const target =
            getLookTarget(t);

        camera.lookAt(
            target
        );

        // --------------------------------------------------------
        // Finish
        // --------------------------------------------------------

        if (raw >= 1) {
            camera.position.copy(
                finalPosition
            );

            camera.lookAt(
                finalTarget
            );

            active = false;

            return false;
        }

        return true;
    }

    return {
        start,
        update,
        isActive: () => active,
    };
}