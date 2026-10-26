// Connaught Place, 3D: the DepthWizard demo's TRUE COLOR city model, exactly as the demo page draws it (colours,
// materials, sunset lights, soft shadows, haze, baked ground AO and tone mapping copied from depthwizard-studio
// frontend/src/cityscape.js), at the demo's 26x vertical exaggeration, turning slowly on a transparent background.
// Data: src/demo/connaught-city.json (the demo's cityscape/connaught_place.json: Google Open Buildings outlines and
// heights, OpenStreetMap roads, flat ground). Renders only while on screen; reduced motion shows it still.
// Three.js r186, same as DepthWizard2 and the studio.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import city from './demo/connaught-city.json';

// The demo's exaggeration: its tile is 100 units across the box's north-south extent, its vertical scale 0.02 units per
// metre times the factor, so factor 26 draws heights 0.52 / (100 / extent) times their real proportion (~5.3x here).
const EXAGGERATION = 26;
const EXTENT_M = (city.bbox_wgs84.north - city.bbox_wgs84.south) * 111320;
const HEIGHT_SCALE = (0.02 * EXAGGERATION) / (100 / EXTENT_M);
const TURN = 0.12; // radians per second
const SETTLE_AFTER = 800, SETTLE_TAU = 1.2;   // once let go: wait (ms), then ease back home (time constant, s: unhurried)
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ── from depthwizard-studio frontend/src/cityscape.js (unchanged values) ──
const TOP = new THREE.Color('#fff0e8');   // building colour at the roof: warm peach...
const FOOT = new THREE.Color('#897386');  // ...deepening to plum at street level
const EDGE = '#ffd2c0';       // soft warm edge lines
const GROUND = '#716274';     // dusk plum
const STREET = '#ebbdb0';     // glowing rose-peach streets
const PLINTH = '#20171d';
const HAZE = '#c8b4c6';       // light lavender haze, thickening with distance
const GROUND_STEP = 5;        // m, ground mesh spacing

function boxUv(geo, size) {
  const p = geo.attributes.position, uv = new Float32Array(p.count * 2);
  for (let i = 0; i < p.count; i++) { uv[2 * i] = p.getX(i) / size + 0.5; uv[2 * i + 1] = -p.getZ(i) / size + 0.5; }
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
}
function groundSampler(g) {
  if (!g.z) return () => 0; // flat
  return (x, y) => {
    const c = Math.min(Math.max((x - g.x0) / g.res, 0), g.nx - 1.001), r = Math.min(Math.max((g.y0 - y) / g.res, 0), g.ny - 1.001);
    const c0 = Math.floor(c), r0 = Math.floor(r), fc = c - c0, fr = r - r0, z = (rr, cc) => g.z[rr * g.nx + cc];
    return (z(r0, c0) * (1 - fc) + z(r0, c0 + 1) * fc) * (1 - fr) + (z(r0 + 1, c0) * (1 - fc) + z(r0 + 1, c0 + 1) * fc) * fr;
  };
}
function buildGround(data, groundAt) {
  const n = Math.round(data.size_m / GROUND_STEP), geo = new THREE.PlaneGeometry(data.size_m, data.size_m, n, n);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, groundAt(pos.getX(i), -pos.getZ(i)));
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: GROUND, roughness: 0.95 }));
  mesh.receiveShadow = true;
  const half = data.size_m / 2, floor = Math.min(...(data.ground.z ?? [0])) - 8, wall = [];
  const corners = [[-half, -half], [half, -half], [half, half], [-half, half], [-half, -half]];
  for (let c = 0; c < 4; c++) {
    const [x0, y0] = corners[c], [x1, y1] = corners[c + 1];
    for (let i = 0; i < n; i++) {
      const xa = x0 + ((x1 - x0) * i) / n, ya = y0 + ((y1 - y0) * i) / n, xb = x0 + ((x1 - x0) * (i + 1)) / n, yb = y0 + ((y1 - y0) * (i + 1)) / n;
      const ta = [xa, groundAt(xa, ya), -ya], tb = [xb, groundAt(xb, yb), -yb];
      wall.push(...ta, xa, floor, -ya, ...tb, ...tb, xa, floor, -ya, xb, floor, -yb);
    }
  }
  const wallGeo = new THREE.BufferGeometry(); wallGeo.setAttribute('position', new THREE.Float32BufferAttribute(wall, 3)); wallGeo.computeVertexNormals();
  const skirt = new THREE.Mesh(wallGeo, new THREE.MeshStandardMaterial({ color: PLINTH, roughness: 1, side: THREE.DoubleSide }));
  const group = new THREE.Group(); group.add(mesh, skirt); return group;
}
const edgeLines = (geometry) => new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 30), new THREE.LineBasicMaterial({ color: EDGE, transparent: true, opacity: 0.12 }));
function buildBuildings(data, groundAt) {
  const geos = [];
  for (const [h, p] of data.buildings) {
    const shape = new THREE.Shape(); let cx = 0, cy = 0;
    for (let i = 0; i < p.length; i += 2) { (i ? shape.lineTo : shape.moveTo).call(shape, p[i], p[i + 1]); cx += p[i]; cy += p[i + 1]; }
    const base = groundAt(cx / (p.length / 2), cy / (p.length / 2)) - 1;
    const geo = new THREE.ExtrudeGeometry(shape, { depth: h + 1, bevelEnabled: false });
    geo.rotateX(-Math.PI / 2); geo.translate(0, base, 0);
    const pos = geo.attributes.position, colors = new Float32Array(pos.count * 3), c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) c.lerpColors(FOOT, TOP, Math.min(Math.max((pos.getY(i) - base) / (h + 1), 0), 1)).toArray(colors, i * 3);
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3)); geos.push(geo);
  }
  const merged = mergeGeometries(geos, false); geos.forEach((g) => g.dispose());
  const mesh = new THREE.Mesh(merged, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
  mesh.castShadow = mesh.receiveShadow = true;
  const group = new THREE.Group(); group.add(mesh, edgeLines(merged)); return group;
}
function buildRoads(data, groundAt) {
  const pos = [];
  for (const [w, p] of data.roads) {
    const pts = [];
    for (let i = 0; i + 3 < p.length; i += 2) {
      const [x0, y0, x1, y1] = [p[i], p[i + 1], p[i + 2], p[i + 3]], steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 5));
      for (let s = 0; s < steps; s++) pts.push([x0 + ((x1 - x0) * s) / steps, y0 + ((y1 - y0) * s) / steps]);
    }
    pts.push([p[p.length - 2], p[p.length - 1]]);
    for (let i = 0; i + 1 < pts.length; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[i + 1], len = Math.hypot(bx - ax, by - ay) || 1;
      const nx = (-(by - ay) / len) * (w * 0.28), ny = ((bx - ax) / len) * (w * 0.28), v = (x, y) => [x, groundAt(x, y) + 0.3, -y];
      const a1 = v(ax + nx, ay + ny), a2 = v(ax - nx, ay - ny), b1 = v(bx + nx, by + ny), b2 = v(bx - nx, by - ny);
      pos.push(...a1, ...a2, ...b1, ...b1, ...a2, ...b2);
    }
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.computeVertexNormals(); boxUv(geo, data.size_m);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: STREET, roughness: 0.9, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2 }));
  mesh.receiveShadow = true; return mesh;
}
function blur(src, n, r) {
  let a = Float32Array.from(src), b = new Float32Array(n * n);
  const pass = (from, to, step, line) => {
    for (let l = 0; l < n; l++) {
      const o = l * line; let sum = 0;
      for (let k = -r; k <= r; k++) sum += from[o + Math.min(Math.max(k, 0), n - 1) * step];
      for (let i = 0; i < n; i++) { to[o + i * step] = sum / (2 * r + 1); sum += from[o + Math.min(i + r + 1, n - 1) * step] - from[o + Math.max(i - r, 0) * step]; }
    }
  };
  for (let it = 0; it < 3; it++) { pass(a, b, 1, n); pass(b, a, n, 1); }
  return a;
}
function bakeGroundAO(renderer, buildings, sizeM, groundMeshes) {
  const N = 512, half = sizeM / 2, mPerPx = sizeM / N;
  buildings.updateMatrixWorld(true);
  const top = new THREE.Box3().setFromObject(buildings).max.y;
  const camera = new THREE.OrthographicCamera(-half, half, half, -half, 1, top + 2000);
  camera.position.set(0, top + 1000, 0); camera.up.set(0, 0, -1); camera.lookAt(0, 0, 0);
  const scene = new THREE.Scene();
  scene.overrideMaterial = new THREE.ShaderMaterial({
    uniforms: { uTop: { value: Math.max(top, 1) } },
    vertexShader: 'varying float vH; void main() { vec4 w = modelMatrix * vec4(position, 1.0); vH = w.y; gl_Position = projectionMatrix * viewMatrix * w; }',
    fragmentShader: 'uniform float uTop; varying float vH; void main() { gl_FragColor = vec4(clamp(vH / uTop, 0.0, 1.0), 1.0, 0.0, 1.0); }',
  });
  const parent = buildings.parent; scene.add(buildings);
  const target = new THREE.WebGLRenderTarget(N, N), clear = renderer.getClearColor(new THREE.Color()), clearAlpha = renderer.getClearAlpha();
  renderer.setRenderTarget(target); renderer.setClearColor(0x000000, 1); renderer.clear(); renderer.render(scene, camera);
  const px = new Uint8Array(N * N * 4); renderer.readRenderTargetPixels(target, 0, 0, N, N, px);
  renderer.setRenderTarget(null); renderer.setClearColor(clear, clearAlpha); target.dispose(); scene.overrideMaterial.dispose(); parent.add(buildings);
  const occ = new Float32Array(N * N), tall = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) { occ[i] = px[4 * i + 1] > 127 ? 1 : 0; tall[i] = occ[i] * Math.min((px[4 * i] / 255) * top / 40, 1); }
  const near = blur(occ, N, Math.max(1, Math.round(3 / mPerPx))), far = blur(tall, N, Math.max(2, Math.round(14 / mPerPx)));
  const data = new Uint8Array(N * N * 4);
  for (let i = 0; i < N * N; i++) { const ao = Math.min(Math.max(1 - 0.55 * near[i] - 0.4 * far[i], 0.35), 1) * 255; data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = ao; data[4 * i + 3] = 255; }
  const map = new THREE.DataTexture(data, N, N);
  map.magFilter = THREE.LinearFilter; map.minFilter = THREE.LinearMipmapLinearFilter; map.generateMipmaps = true; map.needsUpdate = true;
  for (const mesh of groundMeshes) { mesh.material.map = map; mesh.material.needsUpdate = true; }
}
function addLights(parent, unitsPerMetre, sizeM) {
  const sky = new THREE.HemisphereLight('#cdbadb', '#4f4052', 1.3); sky.position.set(0, 1e6, 0); parent.add(sky);
  const sun = new THREE.DirectionalLight('#ffbf78', 2.8), s = sizeM / 500;
  sun.position.set(-520 * s, 340 * s, 240 * s); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
  const k = unitsPerMetre * s;
  Object.assign(sun.shadow.camera, { left: -300 * k, right: 300 * k, top: 300 * k, bottom: -300 * k, near: 50 * k, far: 1400 * k });
  sun.shadow.radius = 4; sun.shadow.blurSamples = 8; sun.shadow.bias = -0.0004; sun.shadow.normalBias = 1.2 * unitsPerMetre;
  parent.add(sun, sun.target); return sun;
}
// ── end of the studio code ──

export function initDemoCity(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;   // r186 dropped PCFSoft (it fell back to this anyway)
  renderer.toneMapping = THREE.NeutralToneMapping; // as the demo while the city model shows
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  // the demo's haze colour at about its on-screen strength (~13% at this camera's 3 km): its 0.0024 per world unit
  // converted to metres would read 40% at this distance
  scene.fog = new THREE.FogExp2(HAZE, 0.000125);
  const camera = new THREE.PerspectiveCamera(30, 1, 10, 9000);
  camera.position.set(0, 1500, 2600); camera.lookAt(0, -30, 0);   // 3 km off: the 1 km box turning (707 m to a corner) fits

  // the model in metres (x east, y up, z = -north), raised by the demo's exaggeration; turning about its centre
  const groundAt = groundSampler(city.ground), model = new THREE.Group(), turn = new THREE.Group();
  const ground = buildGround(city, groundAt), roads = buildRoads(city, groundAt), buildings = buildBuildings(city, groundAt);
  model.add(ground, roads, buildings); model.scale.y = HEIGHT_SCALE;
  bakeGroundAO(renderer, buildings, city.size_m, [ground.children[0], roads]);
  addLights(model, 1, city.size_m);
  turn.add(model); scene.add(turn); turn.rotation.y = -0.5;

  const resize = () => {
    const w = container.clientWidth, h = container.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  resize(); new ResizeObserver(resize).observe(container);
  renderer.render(scene, camera);   // first draw now (shaders, shadow map): done when the stage came on screen, it stalled the teal piece's drop

  // drag to orbit all the way round (right drag or shift: pan), as in the studio, no zoom (the wheel scrolls the page);
  // one finger: sideways orbits, up and down scrolls. Let go and the view eases back to where it was (owner 2026-10-17).
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, -30, 0); controls.enableZoom = false; controls.enableDamping = !reduceMotion; controls.update(); controls.saveState();
  renderer.domElement.style.touchAction = 'pan-y';
  const home = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target)), s = new THREE.Spherical(), v = new THREE.Vector3();
  let held = false, letGo = 0;
  controls.addEventListener('start', () => { held = true; });
  controls.addEventListener('end', () => { held = false; letGo = performance.now(); if (reduceMotion) setTimeout(() => held || controls.reset(), 1500); });
  const settle = (t, dt) => {
    const since = (t - letGo - SETTLE_AFTER) / 1000; if (held || since < 0) return;
    const k = (1 - Math.exp(-dt / SETTLE_TAU)) * Math.min(1, since / 1.2) ** 2;   // eased in over 1.2 s: no sudden start
    s.setFromVector3(v.copy(camera.position).sub(controls.target));
    s.theta += (Math.atan2(Math.sin(home.theta - s.theta), Math.cos(home.theta - s.theta))) * k;   // the short way round
    s.phi += (home.phi - s.phi) * k; s.radius += (home.radius - s.radius) * k;
    controls.target.lerp(controls.target0, k); camera.position.copy(controls.target).add(v.setFromSpherical(s));
  };

  let onScreen = false, last = 0, raf = 0;
  const frame = (t) => {
    raf = 0; if (!onScreen) return;
    if (last && !reduceMotion) { const dt = Math.min(0.1, (t - last) / 1000); turn.rotation.y += dt * TURN; settle(t, dt); }
    controls.update(); last = t; renderer.render(scene, camera);
    if (!reduceMotion) raf = requestAnimationFrame(frame);
  };
  if (reduceMotion) controls.addEventListener('change', () => { if (onScreen && !raf) raf = requestAnimationFrame(frame); });
  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; last = 0; if (onScreen && !raf) raf = requestAnimationFrame(frame); }).observe(container);
}
