import * as THREE from "three";
import CameraControls from "camera-controls";

CameraControls.install({ THREE });

const IDLE_RESUME_MS = 2500;

// Was 0.0018 rad/frame at ~60fps under the old rig's manual per-frame
// increment. Converted to a per-second rate so it's frame-rate independent
// under camera-controls' delta-based update().
const AUTO_ROTATE_RADIANS_PER_SEC = 0.0018 * 60;

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

    // Movement envelope matched (2026-09-25) to the reference SIH26175 repo
    // (external/SIH26175-DepthWizard-notvishuuuu, CameraControls.jsx:
    // OrbitControls minDistance 5 / maxDistance 140 on a 60-unit terrain,
    // maxPolarAngle 0.495π, dampingFactor 0.08). Distances are scaled by
    // 100/60 to our 100-unit terrain.
    controls.minDistance = 5 * (100 / 60);
    controls.maxDistance = 140 * (100 / 60);

    // Direct manipulation (user report, 2026-10-05: the negated speeds moved
    // the structure opposite to the drag): camera-controls' own signs, so the
    // surface under the cursor follows a drag, and a sideways scroll, in the
    // direction the hand moves.
    controls.azimuthRotateSpeed = 1;
    controls.polarRotateSpeed = 1;

    // Weighted coast: tighter follow while actively dragging so the camera
    // doesn't feel laggy mid-gesture, a longer glide once released so a
    // flick keeps drifting briefly before friction settles it — tuned down
    // from the library's defaults (0.125 / 0.25) so release-and-coast
    // reads as weighted without feeling loose or slow to settle.
    // OrbitControls dampingFactor 0.08 at 60 fps is an exponential lag with a
    // ~0.21 s time constant, applied to drags and releases alike. smoothDamp
    // with smoothTime 0.2 reaches 63% of a step in ~0.21 s, so both match.
    controls.draggingSmoothTime = 0.2;
    controls.smoothTime = 0.2;

    // Pitch: clamp polar angle to ±21.6° around the initial framing's
    // elevation, matching the old rig's pitch clamp.
    // Reference repo: from straight down (0) to just above the horizon
    // (0.495π), so the camera can never dip under the terrain.
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI * 0.495;

    // Roll (the old rig's "third axis"): camera-controls' orbit keeps the
    // camera's up vector fixed to world Y at all times, so roll never
    // occurs at all — stricter than, and a direct replacement for, the old
    // rig's near-locked third axis.

    // Drag = rotate (yaw + pitch), zoom disabled on drag/right/middle.
    controls.mouseButtons.left = CameraControls.ACTION.ROTATE;
    controls.mouseButtons.right = CameraControls.ACTION.NONE;
    controls.mouseButtons.middle = CameraControls.ACTION.NONE;
    // Every wheel event is handled by onWheel below (scroll up/down and pinch
    // zoom, sideways scroll rotates), so the library's own wheel action is off.
    controls.mouseButtons.wheel = CameraControls.ACTION.NONE;

    // Real touch: one finger rotates, two-finger pinch dollies (distance),
    // not the lens-style zoom.
    controls.touches.one = CameraControls.ACTION.TOUCH_ROTATE;
    controls.touches.two = CameraControls.ACTION.TOUCH_DOLLY;
    controls.touches.three = CameraControls.ACTION.NONE;

    let autoRotating = true;
    // Held off regardless of the idle timer (e.g. while the measurement tool
    // is active — placing points on a slowly turning scene is fiddly).
    let autoRotateHeld = false;
    // The user's Play/Pause choice (expanded view's playback button).
    let autoRotateUserPaused = false;
    // Set while something else (the flythrough) drives the camera itself:
    // base auto-rotate stays out of the way so the two never add up.
    let externallyDriven = false;
    let autoRotateRamp = 0; // 0 → 1 while auto-rotation eases in
    const AUTO_ROTATE_RAMP_SEC = 1;
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

    // Lock View: while locked, mouse/touch orbit/pan/zoom are off whatever
    // else sets `enabled` (e.g. the measure tool re-enables it after a point
    // drag). Programmatic camera moves (the nav bar buttons) still work.
    let inputLocked = false;
    let enabledWanted = true;
    const enabledDesc = Object.getOwnPropertyDescriptor(CameraControls.prototype, "enabled");
    Object.defineProperty(controls, "enabled", {
        configurable: true,
        get() {
            return enabledDesc.get.call(this);
        },
        set(value) {
            enabledWanted = Boolean(value);
            enabledDesc.set.call(this, enabledWanted && !inputLocked);
        },
    });
    controls.setInputLocked = locked => {
        inputLocked = Boolean(locked);
        enabledDesc.set.call(controls, enabledWanted && !inputLocked);
    };
    controls.isInputLocked = () => inputLocked;

    // Auto-rotation target: by default the camera orbits; a viewer can hand in
    // its own handler (radians this frame) to spin the structure instead.
    controls.autoRotateHandler = null;

    // Wheel gestures (2026-10-05): scroll up/down and pinch (ctrlKey) zoom by
    // dollying the camera (distance, min/maxDistance); a mostly sideways
    // scroll rotates like a sideways drag. Handled here rather than by
    // camera-controls, whose wheel zoom is a camera.zoom lens scale.
    function onWheel(event) {
        event.preventDefault();
        if (inputLocked) {
            return; // Lock View: no zoom or rotation either
        }
        // a user gesture, like a drag: pauses auto-rotation and ends a fly-through
        controls.dispatchEvent({ type: "control" });

        if (!event.ctrlKey && Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
            // same formula and sign as camera-controls' drag/wheel rotate
            const height = domElement.clientHeight || 1;
            const px = event.deltaMode === 1 ? event.deltaX * 33 : event.deltaX;
            controls.rotate((2 * Math.PI * controls.azimuthRotateSpeed * px) / height, 0, true);
            return;
        }
        if (event.deltaY === 0) {
            return;
        }
        if (!event.ctrlKey) {
            // scroll: up (deltaY < 0) zooms in; a mouse notch (~100 px) ≈ 10 %, a
            // trackpad's small deltas a fraction of that, so neither crawls nor jumps
            const px = Math.min(Math.abs(event.deltaMode === 1 ? event.deltaY * 33 : event.deltaY), 150);
            const scale = PINCH_DOLLY_SCALE ** (px / 50);
            const next = controls._sphericalEnd.radius * (event.deltaY < 0 ? scale : 1 / scale);
            controls.dollyTo(THREE.MathUtils.clamp(next, controls.minDistance, controls.maxDistance), true);
            return;
        }

        // Pinch: sign only, not scaled by |deltaY| — see PINCH_DOLLY_SCALE above.
        // Pinch-out (deltaY < 0) zooms in (closer).
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

        if (autoRotating && !autoRotateHeld && !autoRotateUserPaused && !externallyDriven) {
            // Positive, matching a rightward drag under the
            // azimuthRotateSpeed=-1 fix above (both increase azimuthAngle)
            // — old code's auto-rotate and drag shared the same sign too.
            // Eased back in (~1 s) whenever it (re)starts, so resuming after a
            // fly-through, the idle timer or Play never jolts.
            autoRotateRamp = Math.min(1, autoRotateRamp + delta / AUTO_ROTATE_RAMP_SEC);
            const ramp = autoRotateRamp * autoRotateRamp * (3 - 2 * autoRotateRamp);
            const step = AUTO_ROTATE_RADIANS_PER_SEC * speedMultiplier * delta * ramp;
            if (controls.autoRotateHandler) {
                controls.autoRotateHandler(step);
            } else {
                controls.azimuthAngle += step;
            }
        } else {
            autoRotateRamp = 0;
        }

        return nativeUpdate(delta);
    };

    controls.reset = function reset() {
        speedMultiplier = 1;
        autoRotating = true;
        clearTimeout(idleTimer);
        return nativeReset(false);
    };

    // Reset View: straight back to the saved default framing (the first
    // frame after creation), instantly. Auto-rotation's on/off state is left
    // exactly as the user set it.
    controls.resetView = () => nativeReset(false);

    controls.setAutoRotatePaused = function setAutoRotatePaused(paused) {
        autoRotateUserPaused = !!paused;
    };

    controls.isAutoRotatePaused = () => autoRotateUserPaused;

    controls.setAutoRotateHold = function setAutoRotateHold(held) {
        autoRotateHeld = !!held;
    };

    controls.setExternallyDriven = function setExternallyDriven(driven) {
        externallyDriven = !!driven;
    };

    controls.AUTO_ROTATE_RADIANS_PER_SEC = AUTO_ROTATE_RADIANS_PER_SEC;

    controls.setSpeedMultiplier = function setSpeedMultiplier(multiplier) {
        speedMultiplier = multiplier;
    };

    return controls;
}
