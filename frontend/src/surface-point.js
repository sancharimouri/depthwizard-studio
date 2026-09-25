// Linked-hover disc and selected-point marker on the 3D terrain surface, for
// the Image Inspection magnifier.
//
//   hover(u, v)   translucent yellow radial-gradient disc at image UV (u, v)
//   select(u, v)  persistent marker (one at a time); clear() removes it
//
// Image UV → grid (col = u·(W−1), row = v·(H−1)): the terrain texture is the
// image drawn over the same grid (row 0 = image top = north). Each marker is
// a small patch built on the heightfield around that point (z sampled from
// the mesh's current vertices via heightfield.surfaceAt, so it follows the
// terrain in 3D and lies flat on the flat layers), a child of the terrain
// mesh so it also follows the display exaggeration.

import * as THREE from "three";
import { gridToLocalXY, surfaceAt } from "./heightfield.js";

const STEPS = 24; // patch resolution across its diameter
const LIFT = 0.12; // local units above the surface

const vertexShader = /* glsl */ `
    attribute vec2 aOff;
    varying vec2 vOff;
    void main() {
        vOff = aOff;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const hoverFragment = /* glsl */ `
    uniform float uOpacity;
    varying vec2 vOff;
    void main() {
        float d = length(vOff);
        if (d > 1.0) discard;
        float a = (1.0 - smoothstep(0.0, 1.0, d)) * 0.75;
        gl_FragColor = vec4(1.0, 0.84, 0.25, a * uOpacity);
    }
`;

// uHighlight = 1 while the pointer is over the marker: the same hover
// treatment as a measure point (#ffd23f with a stronger glow), plus a
// thicker ring and a larger core.
const selectedFragment = /* glsl */ `
    uniform float uOpacity;
    uniform float uHighlight;
    varying vec2 vOff;
    void main() {
        float d = length(vOff);
        if (d > 1.0) discard;
        float inner = mix(0.62, 0.52, uHighlight);
        float ring = smoothstep(inner, inner + 0.1, d) * (1.0 - smoothstep(0.88, 1.0, d));
        float coreR = mix(0.16, 0.24, uHighlight);
        float core = 1.0 - smoothstep(coreR, coreR + 0.1, d);
        float glow = (1.0 - smoothstep(0.0, 1.0, d)) * mix(0.25, 0.55, uHighlight);
        float a = max(max(ring, core), glow);
        vec3 color = mix(vec3(1.0, 0.82, 0.2), vec3(1.0, 0.824, 0.247), uHighlight);
        color = mix(color, vec3(1.0, 0.93, 0.6), 0.35 * uHighlight * core);
        gl_FragColor = vec4(color, a * uOpacity);
    }
`;

// Draped pass (depth-tested, full strength) + a faint "ghost" pass without
// depth test, so a marker behind a ridge still shows through the terrain.
function material(fragmentShader, ghost = false) {
    return new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: { uOpacity: { value: ghost ? 0.45 : 1 }, uHighlight: { value: 0 } },
        // double-sided: a point on a slope facing away from the camera must still show
        side: THREE.DoubleSide,
        transparent: true,
        depthWrite: false,
        depthTest: !ghost,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        polygonOffsetUnits: -4,
    });
}

export function createSurfacePoints({ getTerrain, onChange = () => {} }) {
    let terrain = null;
    let hover = null;
    let selected = null;
    let selectedUV = null;

    // Selected point: a short upright pin (stem + head) above the surface
    // point, so the marker can be found from any angle, not only top-down.
    let pin = null;
    const PIN_HEIGHT = 6; // local units (~600 m at the 100-unit terrain scale)
    function placePin(u, v) {
        const f = field();
        const col = u * (f.width - 1);
        const row = v * (f.height - 1);
        const xy = gridToLocalXY(f, col, row);
        const z = surfaceAt(f, col, row)?.z ?? 0;
        if (!pin) {
            pin = new THREE.Group();
            const stem = new THREE.Mesh(
                new THREE.CylinderGeometry(0.12, 0.12, PIN_HEIGHT, 8),
                new THREE.MeshBasicMaterial({ color: 0xf5d042 }),
            );
            stem.rotation.x = Math.PI / 2; // cylinder along local Z (up)
            stem.position.z = PIN_HEIGHT / 2;
            const head = new THREE.Mesh(
                new THREE.SphereGeometry(0.75, 16, 12),
                new THREE.MeshBasicMaterial({ color: 0xf5d042 }),
            );
            head.position.z = PIN_HEIGHT;
            pin.add(stem, head);
            pin.renderOrder = 5;
            terrain.mesh.add(pin);
        }
        pin.position.set(xy.x, xy.y, z);
        pin.visible = true;
    }

    function field() {
        const pos = terrain.mesh.geometry.attributes.position;
        return {
            width: terrain.grid.width,
            height: terrain.grid.height,
            terrainWidth: terrain.terrainWidth,
            terrainHeight: terrain.terrainHeight,
            getZ: i => pos.getZ(i),
        };
    }

    function sync() {
        const t = getTerrain();
        if (t === terrain) {
            return Boolean(t);
        }
        [hover, selected].forEach(m => {
            m?.parent?.remove(m);
            m?.geometry.dispose();
            m?.material.dispose();
            m?.children[0]?.material.dispose();
        });
        pin?.parent?.remove(pin);
        pin = null;
        terrain = t;
        hover = null;
        selected = null;
        return Boolean(t);
    }

    // A disc patch of `radius` grid cells centred on (col, row), draped on the surface.
    function buildPatch(col, row, radius) {
        const f = field();
        const n = STEPS + 1;
        const pos = new Float32Array(n * n * 3);
        const off = new Float32Array(n * n * 2);
        for (let j = 0; j < n; j++) {
            for (let i = 0; i < n; i++) {
                const ox = (i / STEPS) * 2 - 1;
                const oy = (j / STEPS) * 2 - 1;
                const c = Math.min(Math.max(col + ox * radius, 0), f.width - 1);
                const r = Math.min(Math.max(row + oy * radius, 0), f.height - 1);
                const xy = gridToLocalXY(f, c, r);
                const s = surfaceAt(f, c, r);
                const k = j * n + i;
                pos[k * 3] = xy.x;
                pos[k * 3 + 1] = xy.y;
                pos[k * 3 + 2] = (s ? s.z : 0) + LIFT;
                off[k * 2] = ox;
                off[k * 2 + 1] = oy;
            }
        }
        const idx = [];
        for (let j = 0; j < STEPS; j++) {
            for (let i = 0; i < STEPS; i++) {
                const a = j * n + i;
                idx.push(a, a + n, a + 1, a + 1, a + n, a + n + 1);
            }
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        g.setAttribute("aOff", new THREE.BufferAttribute(off, 2));
        g.setIndex(idx);
        return g;
    }

    function place(mesh, u, v, radius, fragment, order) {
        const col = u * (terrain.grid.width - 1);
        const row = v * (terrain.grid.height - 1);
        const geometry = buildPatch(col, row, radius);
        if (!mesh) {
            mesh = new THREE.Mesh(geometry, material(fragment));
            mesh.renderOrder = order;
            mesh.frustumCulled = false;
            const ghost = new THREE.Mesh(geometry, material(fragment, true));
            ghost.renderOrder = order - 1;
            ghost.frustumCulled = false;
            mesh.add(ghost);
            terrain.mesh.add(mesh);
        } else {
            mesh.geometry.dispose();
            mesh.geometry = geometry;
            mesh.children[0].geometry = geometry;
        }
        mesh.visible = true;
        return mesh;
    }

    return {
        hover(u, v) {
            if (!sync()) {
                return;
            }
            hover = place(hover, u, v, 16, hoverFragment, 3);
            onChange();
        },
        clearHover() {
            if (hover) {
                hover.visible = false;
                onChange();
            }
        },
        select(u, v) {
            if (!sync()) {
                return;
            }
            selectedUV = { u, v };
            selected = place(selected, u, v, 9, selectedFragment, 4);
            placePin(u, v);
            onChange();
        },
        // hover highlight of the selected marker (like a measure point's :hover)
        setHighlight(on) {
            const v = on && selectedUV ? 1 : 0;
            if (!selected || selected.material.uniforms.uHighlight.value === v) {
                return;
            }
            selected.material.uniforms.uHighlight.value = v;
            selected.children[0].material.uniforms.uHighlight.value = v;
            if (pin) {
                pin.children[1].scale.setScalar(v ? 1.35 : 1);
                pin.children.forEach(m => m.material.color.setHex(v ? 0xffd23f : 0xf5d042));
            }
            onChange();
        },
        clear() {
            this.setHighlight(false);
            selectedUV = null;
            if (selected) {
                selected.visible = false;
            }
            if (pin) {
                pin.visible = false;
            }
            onChange();
        },
        // re-drape after the surface changed (flat ↔ 3D layer switch)
        refresh() {
            if (sync() && selectedUV) {
                selected = place(selected, selectedUV.u, selectedUV.v, 9, selectedFragment, 4);
                placePin(selectedUV.u, selectedUV.v);
            }
        },
        // the selected point in grid coordinates, or null
        selectedGrid() {
            if (!selectedUV || !terrain) {
                return null;
            }
            return { x: selectedUV.u * (terrain.grid.width - 1), y: selectedUV.v * (terrain.grid.height - 1) };
        },
        get selectedUV() {
            return selectedUV;
        },
    };
}
