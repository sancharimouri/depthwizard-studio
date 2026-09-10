import * as THREE from "three";

const IDLE_RESUME_MS = 2500;
const AUTO_ROTATE_RADIANS_PER_FRAME = 0.0018;
const YAW_SENSITIVITY = 0.0065;
const PITCH_SENSITIVITY = 0.0065;
const WHEEL_YAW_SENSITIVITY = 0.0065;
const WHEEL_PITCH_SENSITIVITY = 0.0065;

// Was ±72° (could tip past horizontal toward upside-down). Restricted to
// ~30% of that range so the hinge tilts noticeably but can never approach
// upside-down: ±72° * 0.3 = ±21.6°.
const PITCH_LIMIT = THREE.MathUtils.degToRad(72 * 0.3);

// Direct trackball-style control of the terrain itself (not the camera):
// single-pointer drag, and two-finger trackpad SCROLL, yaw/pitch the rig
// around its own center. Two-finger PINCH (real touch pinch, or a trackpad
// pinch gesture — reported as a wheel event with ctrlKey true) is left
// alone so OrbitControls' zoom-only handling still owns it; this module's
// wheel listener must be registered on domElement before OrbitControls'
// own listener so it can claim (and stopImmediatePropagation) plain
// two-finger scroll before OrbitControls treats it as a zoom. Idle
// auto-rotate resumes shortly after the last interaction, but any new
// drag/scroll/pinch preempts it immediately — no waiting for a cycle to
// finish.
export function createTerrainRig(domElement, baseYaw = 0) {
    const rig = new THREE.Group();

    let yaw = baseYaw;
    let pitch = 0;
    let autoRotating = true;
    let idleTimer = null;
    let speedMultiplier = 1;

    const activePointers = new Map();
    let dragPointerId = null;
    let lastX = 0;
    let lastY = 0;

    // Order matters here, and it was the source of the "third axis": with
    // "YXZ", yaw is applied as the outer (world-fixed Y) rotation and pitch
    // as the inner one — meaning pitch's own effective axis is whatever
    // world direction yaw has rotated the local X axis to. At yaw=90° that
    // axis lands exactly on the camera's view/depth axis, so a pitch drag
    // at that point reads as pure roll instead of a tilt — up to the full
    // pitch range (was ±72°) could show up as unintended roll. Swapping to
    // "XYZ" makes pitch the outer rotation instead, always applied around
    // the fixed world X axis regardless of yaw, so it reads as a clean
    // up/down tilt from the camera's fixed viewpoint at any yaw — the
    // emergent-roll range this eliminates goes from as much as ±72° down
    // to ~0°.
    function applyRotation() {
        rig.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, "XYZ"));
    }

    function pauseAutoRotate() {
        autoRotating = false;

        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
            autoRotating = true;
        }, IDLE_RESUME_MS);
    }

    function onPointerDown(event) {
        activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        pauseAutoRotate();

        if (activePointers.size === 1) {
            dragPointerId = event.pointerId;
            lastX = event.clientX;
            lastY = event.clientY;
        } else {
            // A second pointer joined — this is a pinch, not a drag.
            dragPointerId = null;
        }
    }

    function onPointerMove(event) {
        if (!activePointers.has(event.pointerId)) {
            return;
        }

        activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

        if (activePointers.size !== 1 || event.pointerId !== dragPointerId) {
            return;
        }

        pauseAutoRotate();

        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        lastX = event.clientX;
        lastY = event.clientY;

        // Rotation follows the swipe direction: drag right → yaw right,
        // drag down → pitch down.
        yaw += dx * YAW_SENSITIVITY;
        pitch = THREE.MathUtils.clamp(pitch + dy * PITCH_SENSITIVITY, -PITCH_LIMIT, PITCH_LIMIT);

        applyRotation();
    }

    function onPointerUp(event) {
        activePointers.delete(event.pointerId);

        if (event.pointerId === dragPointerId) {
            dragPointerId = null;
        }

        // If a pinch just ended and one finger remains, resume dragging from it.
        if (activePointers.size === 1) {
            const [[id, pos]] = activePointers;
            dragPointerId = id;
            lastX = pos.x;
            lastY = pos.y;
        }
    }

    function onWheel(event) {
        pauseAutoRotate();

        // A pinch gesture (real touch pinch, or a trackpad pinch reported
        // as wheel+ctrlKey) is zoom-only — leave it for OrbitControls.
        if (event.ctrlKey) {
            return;
        }

        // Plain two-finger scroll: claim it for rotation before
        // OrbitControls' own wheel listener treats it as a zoom.
        event.preventDefault();
        event.stopImmediatePropagation();

        yaw -= event.deltaX * WHEEL_YAW_SENSITIVITY;
        pitch = THREE.MathUtils.clamp(pitch + event.deltaY * WHEEL_PITCH_SENSITIVITY, -PITCH_LIMIT, PITCH_LIMIT);

        applyRotation();
    }

    domElement.addEventListener("pointerdown", onPointerDown);
    domElement.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    // Registered before OrbitControls' own wheel listener (createTerrainRig
    // is called before createControls in main.js) so stopImmediatePropagation
    // above can actually pre-empt it for plain two-finger scroll.
    domElement.addEventListener("wheel", onWheel, { passive: false });

    applyRotation();

    function update() {
        if (autoRotating) {
            yaw += AUTO_ROTATE_RADIANS_PER_FRAME * speedMultiplier;
            applyRotation();
        }
    }

    function reset() {
        yaw = baseYaw;
        pitch = 0;
        speedMultiplier = 1;
        applyRotation();
    }

    function setSpeedMultiplier(multiplier) {
        speedMultiplier = multiplier;
    }

    return { rig, update, reset, setSpeedMultiplier };
}
