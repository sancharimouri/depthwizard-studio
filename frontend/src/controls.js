import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Zoom only. Rotation (yaw/pitch of the terrain itself) is handled
// separately by terrainRig.js — this only dollies the camera in/out
// along its fixed view axis, via wheel or two-finger pinch.
export function createControls(camera, domElement, target) {
    const controls = new OrbitControls(camera, domElement);

    controls.target.copy(target);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;

    controls.enableRotate = false;
    controls.enablePan = false;
    controls.enableZoom = true;

    controls.minDistance = 40;
    controls.maxDistance = 260;

    controls.update();

    return controls;
}
