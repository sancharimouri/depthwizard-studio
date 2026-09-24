import * as THREE from "three";
import CameraControls from "camera-controls";

CameraControls.install({ THREE });

const IDLE_RESUME_MS = 2500;

// Was 0.0018 rad/frame at ~60fps under the old rig's manual per-frame
// increment. Converted to a per-second rate so it's frame-rate independent
// under camera-controls' delta-based update().
const AUTO_ROTATE_RADIANS_PER_SEC = 0.0018 * 60;

// Was ±72° (could tip past horizontal toward upside-down). Restricted to
// ~30% of that range so the camera tilts noticeably but can never approach
// upside-down: ±72° * 0.3 = ±21.6°, applied around the initial framing's
// elevation angle (not the literal horizon).
const PITCH_LIMIT = THREE.MathUtils.degToRad(72 * 0.3);

// Matches the old OrbitControls-based zoom: a fixed ~5% distance change
// per wheel event regardless of the event's raw deltaY magnitude (that
// magnitude varies wildly across trackpads/browsers — scaling by it made
// zoom crawl, ~1% per pinch, on trackpads that report small deltaY).
// OrbitControls' own default (zoomSpeed=1) applies this same fixed
// per-event 0.95 either direction, which is why "like before" means
// sign-only here, not magnitude-scaled.
const PINCH_DOLLY_SCALE = 0.95;

// Camera orbits a fixed terrain (NASA-Eyes-style), replacing the old
// terrainRig's approach of spinning the terrain group under a fixed
// camera. Visually equivalent, but lets camera-controls own the full
// gesture set (drag, wheel, pinch) with built-in weighted/inertial
// damping instead of the old instant, undamped rig rotation.
export function createControls(camera, domElement, target, homePosition = new THREE.Vector3(0, 58, 143)) {
    // Constructed without domElement so our own wheel listener (registered
    // below) attaches to domElement *before* camera-controls' internal one
    // — required for stopImmediatePropagation to actually pre-empt it for
    // the ctrlKey pinch case.
    const controls = new CameraControls(camera, undefined);

    controls.setLookAt(
        homePosition.x, homePosition.y, homePosition.z,
        target.x, target.y, target.z,
        false
    );
    // Captures the framing above as what native reset()/reset() restores to
    // — without this, reset() would snap back to the (0,0,0)-ish state
    // captured at construction, before this setLookAt ever ran.
    controls.saveState();

    controls.minDistance = 40;
    controls.maxDistance = 260;

    // Negative: camera-controls' default azimuth sign is a "camera orbit"
    // feel (drag right → camera swings right → the terrain's near face
    // slides left on screen). The old rig instead rotated the terrain
    // itself, a direct-manipulation feel (drag right → the surface under
    // the cursor follows the cursor right) — confirmed by dispatching
    // synthetic drags and comparing the rendered result before/after.
    // Negating azimuthRotateSpeed restores that same feel here, for both
    // drag and wheel-rotate (both route through the same internal call).
    controls.azimuthRotateSpeed = -1;

    // Same reasoning, vertical axis: camera-controls' default polar sign
    // tilts the opposite way from the old rig's pitch for a given
    // drag/scroll-down — confirmed backwards in actual use. Negating
    // polarRotateSpeed fixes drag and wheel-rotate together (both use it).
    controls.polarRotateSpeed = -1;

    // Weighted coast: tighter follow while actively dragging so the camera
    // doesn't feel laggy mid-gesture, a longer glide once released so a
    // flick keeps drifting briefly before friction settles it — tuned down
    // from the library's defaults (0.125 / 0.25) so release-and-coast
    // reads as weighted without feeling loose or slow to settle.
    controls.draggingSmoothTime = 0.09;
    controls.smoothTime = 0.22;

    // Pitch: clamp polar angle to ±21.6° around the initial framing's
    // elevation, matching the old rig's pitch clamp.
    const basePolarAngle = controls.polarAngle;
    controls.minPolarAngle = basePolarAngle - PITCH_LIMIT;
    controls.maxPolarAngle = basePolarAngle + PITCH_LIMIT;

    // Roll (the old rig's "third axis"): camera-controls' orbit keeps the
    // camera's up vector fixed to world Y at all times, so roll never
    // occurs at all — stricter than, and a direct replacement for, the old
    // rig's near-locked third axis.

    // Drag = rotate (yaw + pitch), zoom disabled on drag/right/middle.
    controls.mouseButtons.left = CameraControls.ACTION.ROTATE;
    controls.mouseButtons.right = CameraControls.ACTION.NONE;
    controls.mouseButtons.middle = CameraControls.ACTION.NONE;
    // Plain two-finger trackpad scroll = rotate. camera-controls hardcodes
    // ctrlKey wheel events (trackpad pinch) to its own zoom action
    // regardless of this setting, so it never conflicts with the pinch
    // handling below.
    controls.mouseButtons.wheel = CameraControls.ACTION.ROTATE;

    // Real touch: one finger rotates, two-finger pinch dollies (distance),
    // not the lens-style zoom.
    controls.touches.one = CameraControls.ACTION.TOUCH_ROTATE;
    controls.touches.two = CameraControls.ACTION.TOUCH_DOLLY;
    controls.touches.three = CameraControls.ACTION.NONE;

    let autoRotating = true;
    // Held off regardless of the idle timer (e.g. while the measurement tool
    // is active — placing points on a slowly turning scene is fiddly).
    let autoRotateHeld = false;
    let speedMultiplier = 1;
    let idleTimer = null;

    function pauseAutoRotate() {
        autoRotating = false;
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
            autoRotating = true;
        }, IDLE_RESUME_MS);
    }

    controls.addEventListener("control", pauseAutoRotate);

    // camera-controls always treats a ctrlKey wheel event (trackpad pinch,
    // or the ctrl-scroll a mouse+keyboard user substitutes for it) as its
    // own ACTION.ZOOM — a camera.zoom lens scale — no matter how
    // mouseButtons.wheel is configured. That's a different mechanism than
    // the old distance-based zoom (min/maxDistance), so claim the gesture
    // ourselves ahead of the library's listener and dolly instead, keeping
    // the same zoom range and feel as before. Plain two-finger scroll (no
    // ctrlKey) is left alone for the library's own ACTION.ROTATE handling.
    function onWheel(event) {
        if (!event.ctrlKey) {
            return;
        }

        event.preventDefault();
        event.stopImmediatePropagation();
        pauseAutoRotate();

        // Sign only, not scaled by |deltaY| — see PINCH_DOLLY_SCALE above.
        // Pinch-out / scroll-up (deltaY < 0) zooms in (closer), matching
        // the standard ctrl-scroll page-zoom convention and the old
        // OrbitControls-based zoom it replaces.
        if (event.deltaY === 0) {
            return;
        }
        // Based on the pending target radius (_sphericalEnd), not the
        // public `distance` getter (the damped, currently-rendered value,
        // which lags behind during a fast multi-event pinch/scroll under
        // smoothTime damping) — basing it on the lagging value meant
        // rapid-fire wheel events each recomputed off the same stale
        // number instead of compounding, which is what made zoom crawl.
        const dollyScale = event.deltaY < 0 ? PINCH_DOLLY_SCALE : 1 / PINCH_DOLLY_SCALE;
        const nextDistance = THREE.MathUtils.clamp(controls._sphericalEnd.radius * dollyScale, controls.minDistance, controls.maxDistance);
        controls.dollyTo(nextDistance, true);
    }

    domElement.addEventListener("wheel", onWheel, { passive: false });
    controls.connect(domElement);

    let lastFrameTime = performance.now();
    const nativeUpdate = CameraControls.prototype.update.bind(controls);
    const nativeReset = CameraControls.prototype.reset.bind(controls);

    // Shadows the instance's own update()/reset() so every existing call
    // site (`controls.update()`, `controls.reset()`) keeps working
    // unchanged while gaining idle auto-rotate and speed-multiplier
    // support that used to live on the separate terrainRig.
    controls.update = function update() {
        const now = performance.now();
        // Clamped: a backgrounded tab resuming (or a long GC pause) would
        // otherwise hand smoothDamp a huge one-off delta, which produced
        // wild multi-second rotation jumps when reproduced during testing.
        const delta = Math.min((now - lastFrameTime) / 1000, 0.1);
        lastFrameTime = now;

        if (autoRotating && !autoRotateHeld) {
            // Positive, matching a rightward drag under the
            // azimuthRotateSpeed=-1 fix above (both increase azimuthAngle)
            // — old code's auto-rotate and drag shared the same sign too.
            controls.azimuthAngle += AUTO_ROTATE_RADIANS_PER_SEC * speedMultiplier * delta;
        }

        return nativeUpdate(delta);
    };

    controls.reset = function reset() {
        speedMultiplier = 1;
        autoRotating = true;
        clearTimeout(idleTimer);
        return nativeReset(false);
    };

    controls.setAutoRotateHold = function setAutoRotateHold(held) {
        autoRotateHeld = !!held;
    };

    controls.setSpeedMultiplier = function setSpeedMultiplier(multiplier) {
        speedMultiplier = multiplier;
    };

    return controls;
}
