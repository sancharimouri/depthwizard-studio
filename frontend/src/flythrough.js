// Fly-through for the expanded 3D viewer (the Fly-through box).
//
// One run: start zoomed out (structure small), then zoom in while orbiting
// exactly one full 360° turn, in FLY_MS (faster than the old 9.5 s flight).
//
// Fit: the structure's bounding sphere (radius R, world space) must stay
// inside the view for the whole run, including the last frame. The camera
// looks at the sphere centre, so the centre projects to the canvas centre;
// the usable half-extent on each axis is the smaller distance from that
// centre to the nearest obstruction (the left / right panel columns, the job
// tabs at the top, the toolbars at the bottom). With the tighter of the
// vertical and horizontal half-angles θ, a sphere is fully inside the view
// cone at distance ≥ R / sin θ. The distance only decreases, from
// START_FACTOR × that to (1 + MARGIN) × that, so every frame fits.
//
// Buttons: Fly-through (start / pause at the current position / resume),
// Run again (restart), Reset (stop and return to the default view).

import * as THREE from "three";

const FLY_MS = 6000;
const START_FACTOR = 2.2;
const MARGIN = 0.04;

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
        // ends on the tight fit for the final azimuth (= the start azimuth, one full turn later)
        const endDist = profileAt(profile, az0) * (1 + MARGIN);
        return {
            center: sphere.center.clone(),
            endDist,
            startDist: worst * (1 + MARGIN) * START_FACTOR,
            profile,
            az0,
            polar,
        };
    }

    // Zoom eases in, but never closer than this azimuth's fit, so every frame
    // keeps the whole structure inside the free area.
    function applyAt(k) {
        const az = plan.az0 + 2 * Math.PI * easeInOutSine(k);
        const eased = plan.startDist + (plan.endDist - plan.startDist) * easeInOutCubic(k);
        const d = Math.max(eased, profileAt(plan.profile, az) * (1 + MARGIN));
        controls.rotateTo(az, plan.polar, false);
        controls.dollyTo(d, false);
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
        controls.setTarget(plan.center.x, plan.center.y, plan.center.z, false);
        elapsed = 0;
        last = performance.now();
        applyAt(0);
        state = "running";
        sync();
    }

    function pause() {
        state = "paused";
        sync();
    }

    function resume() {
        last = performance.now();
        controls.setExternallyDriven(true);
        state = "running";
        sync();
    }

    function stop() {
        const wasActive = state === "running" || state === "paused";
        state = "idle";
        if (wasActive || savedMaxDistance !== null) {
            releaseCamera();
        }
        sync();
    }

    function sync() {
        toggle.textContent = state === "running" ? "❚❚ Pause" : state === "paused" ? "▶ Resume" : "▶ Fly-through";
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
            if (state !== "running" || !plan) {
                return;
            }
            const now = performance.now();
            elapsed += Math.min(now - last, 100);
            last = now;
            const k = Math.min(1, elapsed / FLY_MS);
            applyAt(k);
            if (k >= 1) {
                state = "done";
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
