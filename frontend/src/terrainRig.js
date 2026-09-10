import * as THREE from "three";

const IDLE_RESUME_MS = 2500;
const AUTO_ROTATE_RADIANS_PER_FRAME = 0.0018;
const YAW_SENSITIVITY = 0.0065;
const PITCH_SENSITIVITY = 0.0065;
const PITCH_LIMIT = THREE.MathUtils.degToRad(72);

// Direct trackball-style control of the terrain itself (not the camera):
// single-pointer drag yaw/pitches the rig around its own center, two-finger
// touch is left alone so OrbitControls' zoom-only pinch handling still owns
// it. Idle auto-rotate resumes shortly after the last interaction, but any
// new drag/pinch preempts it immediately — no waiting for a cycle to finish.
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

    function applyRotation() {
        rig.quaternion.setFromEuler(new THREE.Euler(pitch, yaw, 0, "YXZ"));
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

        yaw -= dx * YAW_SENSITIVITY;
        pitch = THREE.MathUtils.clamp(pitch - dy * PITCH_SENSITIVITY, -PITCH_LIMIT, PITCH_LIMIT);

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

    function onWheel() {
        pauseAutoRotate();
    }

    domElement.addEventListener("pointerdown", onPointerDown);
    domElement.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    domElement.addEventListener("wheel", onWheel, { passive: true });

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
