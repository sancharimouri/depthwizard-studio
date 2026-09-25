// Fly-through for the expanded 3D viewer (the Fly-through box).
//
// One run: a short eased lead-in from wherever the camera is to the start
// pose, then a 300° orbit while zooming in. Start and end are 1.5× closer
// than the fitted distances (start 1.1× the worst-case box fit, end the
// final azimuth's fit, both ÷ 1.5), and the run may leave the screen edges:
// there is no per-frame fit clamp, so the motion is one smooth ease with no
// kinks. Every phase starts and ends at rest, and pause/resume ramp the
// playback rate instead of snapping, so nothing is jarring.
//
// The fit maths (freeHalfExtents, fitDistance, boxFitDistance) still sets
// the start and end distances from the structure's bounding box, the camera
// FOV/aspect and the free area between the panels.
//
// Buttons: Fly-through (start / pause at the current position / resume),
// Run again (restart), Reset (stop and return to the default view).

import * as THREE from "three";

const LEAD_MS = 900; // eased move from the current view to the start pose
const ORBIT_MS = 5000; // the 270° orbit + zoom-in
const ORBIT = THREE.MathUtils.degToRad(300);
const START_FACTOR = 1.1; // × worst-case fit (was 2.2: the structure now starts twice as big)
const MARGIN = 0.04;
const ZOOM = 1.5; // start and end 1.5× closer than the fitted distances
const RATE_TAU_MS = 150; // pause/resume ease

const easeInOutCubic = t => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const easeInOutSine = t => -(Math.cos(Math.PI * t) - 1) / 2;

// Usable half-extents (px, from the canvas centre) left free by the panels.
export function freeHalfExtents(canvasRect, obstacles) {
    const cx = canvasRect.left + canvasRect.width / 2;
    const cy = canvasRect.top + canvasRect.height / 2;
    let halfW = canvasRect.width / 2;
    let halfH = canvasRect.height / 2;
    if (obstacles.left != null) {
        halfW = Math.min(halfW, cx - obstacles.left);
    }
    if (obstacles.right != null) {
        halfW = Math.min(halfW, obstacles.right - cx);
    }
    if (obstacles.top != null) {
        halfH = Math.min(halfH, cy - obstacles.top);
    }
    if (obstacles.bottom != null) {
        halfH = Math.min(halfH, obstacles.bottom - cy);
    }
    return { halfW: Math.max(halfW, 60), halfH: Math.max(halfH, 60) };
}

// Camera distance at which a sphere of radius R fits the free region.
export function fitDistance(radius, vfovDeg, canvasW, canvasH, halfW, halfH) {
    const tanV = Math.tan(THREE.MathUtils.degToRad(vfovDeg) / 2);
    const tanHalfV = tanV * (halfH / (canvasH / 2));
    const tanHalfH = tanV * (canvasW / canvasH) * (halfW / (canvasW / 2));
    const theta = Math.atan(Math.min(tanHalfV, tanHalfH));
    return radius / Math.sin(theta);
}

// Tighter fit than the sphere: the smallest distance at which all 8 corners
// of the structure's bounding box stay inside the free half-extents, taken as
// the worst case over the whole 360° orbit (sampled every 10°). Beyond that
// distance every azimuth fits too (moving the camera back along its view
// ray only shrinks the projection), so a run whose distance only decreases
// towards this value keeps the structure in view on every frame.
export function boxFitDistance({ box, center, polar, camera, canvasW, canvasH, halfW, halfH, upper, samples = 0 }) {
    const corners = [];
    for (const x of [box.min.x, box.max.x]) {
        for (const y of [box.min.y, box.max.y]) {
            for (const z of [box.min.z, box.max.z]) {
                corners.push(new THREE.Vector3(x, y, z));
            }
        }
    }
    const cam = camera.clone();
    cam.aspect = canvasW / canvasH;
    cam.updateProjectionMatrix();
    const p = new THREE.Vector3();
    const fits = (az, d) => {
        cam.position.set(
            center.x + d * Math.sin(polar) * Math.sin(az),
            center.y + d * Math.cos(polar),
            center.z + d * Math.sin(polar) * Math.cos(az),
        );
        cam.lookAt(center);
        cam.updateMatrixWorld(true);
        for (const c of corners) {
            p.copy(c).project(cam);
            if (p.z > 1 || Math.abs(p.x) * canvasW / 2 > halfW || Math.abs(p.y) * canvasH / 2 > halfH) {
                return false;
            }
        }
        return true;
    };
    const fitAt = az => {
        let lo = 0;
        let hi = upper;
        if (!fits(az, hi)) {
            return upper; // the sphere bound always fits; keep it
        }
        for (let k = 0; k < 24; k++) {
            const mid = (lo + hi) / 2;
            if (fits(az, mid)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return hi;
    };
    if (samples) {
        return Array.from({ length: samples }, (_, i) => fitAt((i / samples) * 2 * Math.PI));
    }
    let worst = 0;
    for (let i = 0; i < 36; i++) {
        worst = Math.max(worst, fitAt((i / 36) * 2 * Math.PI));
    }
    return worst;
}

// Per-azimuth fit profile (PROFILE_N samples over 360°), read with a
// conservative lookup: the larger of the two neighbouring samples.
const PROFILE_N = 72;
function profileAt(profile, az) {
    const n = profile.length;
    const x = ((((az / (2 * Math.PI)) % 1) + 1) % 1) * n;
    const i = Math.floor(x) % n;
    return Math.max(profile[i], profile[(i + 1) % n]);
}

export function createFlythrough({ viewer, getTerrain, getObstacles, canvas, onReset, els }) {
    const controls = viewer.controls;
    const { toggle, again, reset } = els;
    let state = "idle"; // idle | running | paused | done
    let plan = null;
    let elapsed = 0;
    let last = 0;
    let everPressed = false;
    let rate = 0; // playback rate, eased towards 1 (running) or 0 (paused)
    let savedMaxDistance = null;

    function computePlan() {
        const terrain = getTerrain();
        if (!terrain) {
            return null;
        }
        terrain.mesh.updateWorldMatrix(true, true);
        const box = new THREE.Box3().setFromObject(terrain.mesh);
        const sphere = box.getBoundingSphere(new THREE.Sphere());
        const rect = canvas.getBoundingClientRect();
        const { halfW, halfH } = freeHalfExtents(rect, getObstacles());
        // keep the current elevation, within a range that reads as a flight
        const polar = THREE.MathUtils.clamp(controls.polarAngle, THREE.MathUtils.degToRad(40), THREE.MathUtils.degToRad(72));
        const sphereFit = fitDistance(sphere.radius, viewer.camera.fov, rect.width, rect.height, halfW, halfH);
        // fit per azimuth, in the camera-controls azimuth convention (same as the Spherical above)
        const profile = boxFitDistance({
            box, center: sphere.center, polar, camera: viewer.camera,
            canvasW: rect.width, canvasH: rect.height, halfW, halfH, upper: sphereFit, samples: PROFILE_N,
        });
        const az0 = controls.azimuthAngle;
        const worst = Math.max(...profile);
        // ends on the tight fit for the final azimuth (270° after the start)
        const endDist = profileAt(profile, az0 + ORBIT) * (1 + MARGIN) / ZOOM;
        const from = controls.getTarget(new THREE.Vector3(), true);
        return {
            center: sphere.center.clone(),
            endDist,
            startDist: worst * (1 + MARGIN) * START_FACTOR / ZOOM,
            az0,
            polar,
            // where the camera is now: the lead-in starts here
            from: { target: from, polar: controls.polarAngle, dist: controls.distance },
        };
    }

    const target = new THREE.Vector3();
    const position = new THREE.Vector3();
    function pose(tgt, polar, az, dist) {
        position.set(
            tgt.x + dist * Math.sin(polar) * Math.sin(az),
            tgt.y + dist * Math.cos(polar),
            tgt.z + dist * Math.sin(polar) * Math.cos(az),
        );
        controls.setLookAt(position.x, position.y, position.z, tgt.x, tgt.y, tgt.z, false);
    }

    // time (ms since the run began) → camera: lead-in, then the orbit.
    // Both phases ease in and out, so the joins are at rest (no jolt).
    function applyAt(ms) {
        if (ms < LEAD_MS) {
            const e = easeInOutCubic(ms / LEAD_MS);
            target.lerpVectors(plan.from.target, plan.center, e);
            const polar = plan.from.polar + (plan.polar - plan.from.polar) * e;
            const dist = plan.from.dist + (plan.startDist - plan.from.dist) * e;
            pose(target, polar, plan.az0, dist);
            return false;
        }
        const k = Math.min(1, (ms - LEAD_MS) / ORBIT_MS);
        const az = plan.az0 + ORBIT * easeInOutSine(k);
        const dist = plan.startDist + (plan.endDist - plan.startDist) * easeInOutCubic(k);
        pose(plan.center, plan.polar, az, dist);
        return k >= 1;
    }

    function releaseCamera() {
        controls.setExternallyDriven(false);
        if (savedMaxDistance !== null) {
            controls.maxDistance = Math.max(savedMaxDistance, plan?.endDist ?? 0);
            savedMaxDistance = null;
        }
    }

    function begin() {
        plan = computePlan();
        if (!plan) {
            return;
        }
        if (savedMaxDistance === null) {
            savedMaxDistance = controls.maxDistance;
        }
        controls.maxDistance = Math.max(controls.maxDistance, plan.startDist * 1.01);
        controls.setExternallyDriven(true); // base auto-rotation stays out of it
        elapsed = 0;
        rate = 1;
        last = performance.now();
        state = "running";
        sync();
    }

    function pause() {
        state = "paused";
        sync();
    }

    function resume() {
        last = performance.now();
        rate = Math.max(rate, 0); // ramps back up from wherever it is
        controls.setExternallyDriven(true);
        state = "running";
        sync();
    }

    function stop() {
        const wasActive = state === "running" || state === "paused";
        state = "idle";
        rate = 0;
        if (wasActive || savedMaxDistance !== null) {
            releaseCamera();
        }
        sync();
    }

    function sync() {
        toggle.textContent = state === "running" ? "❚❚ Pause" : state === "paused" ? "▶ Resume" : "◎ Fly-through"; // ◎ as on the Demo page
        toggle.setAttribute("aria-pressed", String(state === "running"));
        again.disabled = !everPressed;
        reset.disabled = !everPressed;
    }

    toggle.addEventListener("click", () => {
        everPressed = true;
        if (state === "running") {
            pause();
        } else if (state === "paused") {
            resume();
        } else {
            begin();
        }
    });
    again.addEventListener("click", () => {
        stop();
        begin();
    });
    reset.addEventListener("click", () => {
        stop();
        onReset?.();
    });
    // grabbing the camera halts the flight where it is (Fly-through resumes it)
    controls.addEventListener("controlstart", () => {
        if (state === "running") {
            pause();
        }
    });

    sync();

    return {
        update() {
            if (!plan || (state !== "running" && !(state === "paused" && rate > 0.002))) {
                return;
            }
            const now = performance.now();
            const dt = Math.min(now - last, 100);
            last = now;
            const goal = state === "running" ? 1 : 0;
            rate += (goal - rate) * (1 - Math.exp(-dt / RATE_TAU_MS));
            elapsed += dt * rate;
            const finished = applyAt(elapsed);
            globalThis.__flyTrace?.push([elapsed, position.x, position.y, position.z]); // dev/test trace only
            if (finished && state === "running") {
                state = "done";
                rate = 0;
                releaseCamera();
                sync();
            }
        },
        stop,
        get state() {
            return state;
        },
        get plan() {
            return plan;
        },
    };
}
