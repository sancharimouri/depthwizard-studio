import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const IDLE_RESUME_MS = 2500;

export function createControls(camera, domElement, target) {
    const controls = new OrbitControls(camera, domElement);

    controls.target.copy(target);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;

    controls.minDistance = 40;
    controls.maxDistance = 260;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;

    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    let idleTimer = null;

    function pauseAutoRotate() {
        controls.autoRotate = false;

        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
            controls.autoRotate = true;
        }, IDLE_RESUME_MS);
    }

    domElement.addEventListener("pointerdown", pauseAutoRotate);
    domElement.addEventListener("wheel", pauseAutoRotate, { passive: true });

    controls.update();

    return controls;
}
