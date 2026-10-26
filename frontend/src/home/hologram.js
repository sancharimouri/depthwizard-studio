// Hero hologram: a neon wireframe mountain tile with two echo slabs below it, floating in low
// gravity. Grid lines are drawn by a shader on the real 3D surface (clean quads, depth-correct
// hidden lines, soft fade at the tile edges). Hover tilts the stack and lights the mesh under the
// cursor; a click sends a glowing ripple across the terrain and down through the layers. A band of
// light slowly scans up and down through the stack.
// Three.js r186, same as DepthWizard2, so this module can move across as-is.
import * as THREE from 'three';

const GRID = 200; // vertices per side
const FLOAT_PERIOD = 6.5; // seconds per float cycle
const RIPPLE_SPEED = 1.1; // world units / second
const LAYER_DELAY = 0.18; // seconds the ripple takes to reach each layer below
// Scan timing, as a repeating pair: a top-to-bottom pass, a SCAN_GAP-second gap, a bottom-to-top
// pass, then a SCAN_REST-second rest. Gaps are measured as seen: the band leaves the bottom of the
// stack, dips just far enough below to take SCAN_GAP seconds out and back, and returns.
const SCAN_SPEED = 0.97; // world units / second
const SCAN_GAP = 0.5;
const SCAN_HIGH = 0.65; // just above the summit (~0.62)
const SCAN_REST = 4.5 - (2 * (SCAN_HIGH - 0.62)) / SCAN_SPEED; // 4.5s as seen, between leaving and re-entering the summit
const SCAN_LOW = -0.95 - (SCAN_SPEED * SCAN_GAP) / 2; // bottom of the stack minus half the gap
const SCAN_PASS = (SCAN_HIGH - SCAN_LOW) / SCAN_SPEED; // ~1.9s
// Entrance: when the intro cross-fades to the page, the terrain fades in exactly where the intro's scan cloud
// drew it (no rise, so the points become the mesh); the slabs below float up into place, starting at once.
const APPEAR_RISE = 0.9; // world units below the resting position (slabs only)
const PEAK = 0.82; // terrain height scale (uPeak)
const APPEAR_MS = 2800;
const APPEAR_LAG = 80; // ms each slab trails the one above it (the first starts at once; was 320 for every layer)
const BASE_YAW = -0.62; // tile turned to the right, a wide diamond

// Each layer: relief scale, vertical drop, grid density, line colours (low -> peak), line
// brightness, fill colour/alpha, float amplitude and phase, and size (each slab a little larger so its
// rim shows around the layer above) and a soft edge glow on the slabs (no crisp border line). Teal throughout, brighter on high ridges
// (light teal at the very peaks); each slab below is dimmer and darker.
const LAYERS = [
  { relief: 1, drop: 0.3, lines: 120, low: '#59d9c0', high: '#a8fbe9', bright: 1.0, fill: '#04100e', fillA: 0.8, amp: 0.085, phase: 0, size: 1, rim: 0.18, shade: 0.6, fade: 0.025, scanW: 0.03, scanGain: 1 },
  { relief: 0.2, drop: 0.6, lines: 48, low: '#3fc7ad', high: '#59d9c0', bright: 0.75, fill: '#0a2f29', fillA: 0.42, amp: 0.075, phase: 0.55, size: 1.06, rim: 0.322, shade: 0, fade: 0.05, scanW: 0.3, scanGain: 0.45 },
  { relief: 0.09, drop: 0.875, lines: 36, low: '#2a9d89', high: '#3fc7ad', bright: 0.6, fill: '#06211d', fillA: 0.5, amp: 0.065, phase: 1.1, size: 1.12, rim: 0.263, shade: 0, fade: 0.05, scanW: 0.3, scanGain: 0.38 },
];

// ── Terrain ────────────────────────────────────────────────────────────────
function hash2(x, y) {
  const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return n - Math.floor(n);
}
function noise(x, y) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y, oct = 5) {
  let s = 0, a = 0.5, f = 1;
  for (let i = 0; i < oct; i += 1) { s += a * noise(x * f, y * f); f *= 2.03; a *= 0.5; }
  return s;
}
// Ridged multifractal: sharp crests that get craggier where the previous octave was high.
function ridged(x, y) {
  let s = 0, a = 0.5, f = 1, w = 1;
  for (let i = 0; i < 6; i += 1) {
    let n = 1 - Math.abs(noise(x * f, y * f) * 2 - 1);
    n *= n * w;
    w = Math.min(1, n * 1.8);
    s += n * a; f *= 2.07; a *= 0.5;
  }
  return s;
}
const g = (x, z, cx, cz, sx, sz) => Math.exp(-((x - cx) ** 2) / sx - ((z - cz) ** 2) / sz);

// Shaped after the reference: broad massif, a cluster of jagged spikes along the back crest,
// a shoulder on the left, a stepped ridge falling to the right-front, gullies on the faces.
// x, z in [-1, 1] (z < 0 is the back of the tile); returns height in [0, 1].
function terrainHeight(x, z) {
  const wx = x + 0.2 * (fbm(x * 1.6 + 4, z * 1.6 + 9) - 0.5);
  const wz = z + 0.2 * (fbm(x * 1.6 + 13, z * 1.6 + 2) - 0.5);
  const massif = g(wx, wz, 0, -0.04, 0.3, 0.24);
  // Summit masses (broad, irregular), not finger-like spikes: the crags come from the noise.
  const summit = Math.max(
    1.0 * g(wx, wz, -0.1, -0.18, 0.05, 0.04),
    0.82 * g(wx, wz, 0.16, -0.1, 0.04, 0.05),
    0.7 * g(wx, wz, -0.34, -0.1, 0.05, 0.05),
  );
  const shoulder = 0.34 * g(wx, wz, -0.58, 0.14, 0.05, 0.06);
  const spur = 0.3 * g(wx * 0.8 + wz * 0.6, wz * 0.8 - wx * 0.6, 0.4, 0.22, 0.025, 0.1);
  const crag = ridged(wx * 2.2 + 1.3, wz * 2.2 + 5.1);
  const gully = 1 - ridged(wx * 3.4 + 8, wz * 3.4 + 3);
  let h = massif * 0.42 + summit * 0.42 + shoulder + spur;
  h += crag * 0.55 * massif ** 1.15 - gully * 0.09 * massif;
  h += 0.035 * fbm(x * 6 + 21, z * 6 + 17, 3); // low ground texture
  // Foothills: gentle ridged undulations around the base so the lower mesh has form too.
  const base = 1 - Math.min(1, massif * 1.6);
  h += base * (0.042 * ridged(x * 2.1 + 40, z * 2.1 + 9) + 0.03 * (fbm(x * 3.3 + 31, z * 3.3 + 12) - 0.4));
  const edge = Math.min(1, (1 - Math.max(Math.abs(x), Math.abs(z))) / 0.25);
  return Math.max(0, h * edge * edge * (3 - 2 * edge));
}

function buildHeights() {
  const n = GRID + 1, heights = new Float32Array(n * n);
  let max = 0;
  for (let j = 0; j < n; j += 1) {
    for (let i = 0; i < n; i += 1) {
      const v = terrainHeight((i / GRID) * 2 - 1, (j / GRID) * 2 - 1);
      heights[j * n + i] = v;
      max = Math.max(max, v);
    }
  }
  for (let k = 0; k < heights.length; k += 1) heights[k] /= max;
  return heights;
}

// ── Shaders ────────────────────────────────────────────────────────────────
const vertexShader = /* glsl */ `
  attribute float aHeight;
  uniform float uRelief, uPeak, uClickT;
  uniform vec2 uClick;
  varying vec2 vUv;
  varying float vH;
  varying vec2 vXZ;
  varying float vNear;
  varying vec3 vPos;
  varying float vWorldY;
  float ring(vec2 xz) {
    if (uClickT < 0.0) return 0.0;
    float r = distance(xz, uClick);
    return exp(-pow((r - uClickT * ${RIPPLE_SPEED.toFixed(2)}) / 0.07, 2.0)) * exp(-uClickT * 1.4);
  }
  void main() {
    vUv = uv;
    vH = aHeight;
    vXZ = position.xz;
    vec3 p = position;
    p.y = aHeight * uPeak * uRelief + ring(p.xz) * 0.05;
    vPos = p;
    vWorldY = (modelMatrix * vec4(p, 1.0)).y;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vNear = smoothstep(6.2, 3.6, -mv.z); // 0 far side of the tile, 1 near side
    gl_Position = projectionMatrix * mv;
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3 uLow, uHigh, uFill;
  uniform float uLines, uBright, uFillA, uHoverOn, uClickT, uRim, uFade;
  uniform vec2 uHover, uClick;
  varying vec2 vUv;
  varying float vH;
  varying vec2 vXZ;
  varying float vNear;
  varying vec3 vPos;
  varying float vWorldY;
  uniform float uShade, uScanOn, uScanY, uScanW, uScanGain, uAppear;
  // Anti-aliased grid line strength, w = line half-width in pixels.
  float grid(float w) {
    vec2 c = vUv * uLines;
    vec2 d = abs(fract(c - 0.5) - 0.5) / fwidth(c);
    return 1.0 - smoothstep(0.0, w, min(d.x, d.y));
  }
  void main() {
    float core = grid(0.9);
    float halo = grid(3.2) * 0.3 + grid(9.0) * 0.1; // tight + wide glow, like a neon bloom
    // Distance to the nearest tile edge (uv space).
    float e = min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y));
    // Slabs fade softly at their edges; the terrain tile fades only over its last ~10px.
    float edge = uFade > 0.0 ? smoothstep(0.0, uFade, e) : 1.0;
    // A soft glow that rises toward the edge, so each sheet reads as its own layer.
    float rim = uRim * smoothstep(0.0, 0.012, e) * (1.0 - smoothstep(0.012, 0.09, e));
    float hover = uHoverOn * exp(-dot(vXZ - uHover, vXZ - uHover) / 0.035);
    float ring = 0.0;
    if (uClickT >= 0.0) {
      float r = distance(vXZ, uClick);
      ring = exp(-pow((r - uClickT * ${RIPPLE_SPEED.toFixed(2)}) / 0.06, 2.0)) * exp(-uClickT * 1.2);
    }
    vec3 lineCol = mix(uLow, uHigh, smoothstep(0.2, 1.0, vH)); // gradual from the lower slopes up; peak colour unchanged
    float lit = (0.55 + 0.45 * min(1.0, vH * 1.4)) * (0.72 + 0.28 * vNear); // ridges and near slopes brighter
    // Light from the upper-left: creases and slopes facing away fall into shadow, lit faces glow a
    // little more, so ridges and gullies read as form (terrain only).
    vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
    if (n.y < 0.0) n = -n;
    float diffuse = max(dot(n, normalize(vec3(-0.6, 0.75, 0.3))), 0.0);
    lit *= mix(1.0, 0.3 + 0.9 * diffuse, uShade);
    float glow = (core + halo) * uBright * lit * (1.0 + hover * 1.6 + ring * 3.0);
    // Scan: a horizontal band of light (world y) sweeps through the stack, lighting the mesh where it
    // crosses. Thin on the mountain; wide and faint on the flat slabs, which it would otherwise cross
    // in a single flash.
    // On the mountain layer, flat ground gets a wider, softer band (the thin band would race across
    // it in an instant); steep slopes keep the thin band unchanged.
    float flatness = uShade > 0.0 ? smoothstep(0.88, 0.99, n.y) : 0.0;
    float bandW = mix(uScanW, 0.065, flatness);
    float scan = uScanOn * uScanGain * mix(1.0, 0.55, flatness) * exp(-pow((vWorldY - uScanY) / bandW, 2.0)) * edge;
    vec3 col = uFill * (1.0 + ring * 1.5) + lineCol * (glow + rim + (core + halo) * scan * 2.2 + scan * 0.06);
    float a = max(max(max(uFillA, clamp(glow, 0.0, 1.0)) * edge, rim), scan * (core + halo));
    gl_FragColor = vec4(col, a * uAppear);
  }
`;

// ── Scene ──────────────────────────────────────────────────────────────────
export function initHologram(container) {
  const canvas = container.querySelector('canvas');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  camera.position.set(0, 2.3, 5.05); // ~29 degrees down, straight on: the tile reads as a wide diamond
  camera.lookAt(0, -0.48, 0);

  const stack = new THREE.Group(); // tilts toward the cursor
  stack.rotation.y = BASE_YAW;
  scene.add(stack);

  const heights = buildHeights();
  const geometry = new THREE.PlaneGeometry(2, 2, GRID, GRID);
  geometry.rotateX(-Math.PI / 2); // plane in x/z; uv.y runs back -> front
  geometry.setAttribute('aHeight', new THREE.BufferAttribute(heights, 1));

  const shared = {
    uHover: { value: new THREE.Vector2(9, 9) }, uHoverOn: { value: 0 }, uClick: { value: new THREE.Vector2() },
    uScanOn: { value: reduceMotion ? 0 : 1 }, uScanY: { value: -2 },
  };
  const layers = LAYERS.map((cfg, k) => {
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: true,
      side: THREE.DoubleSide,
      uniforms: {
        ...shared,
        uRelief: { value: cfg.relief },
        uPeak: { value: PEAK },
        uLines: { value: cfg.lines },
        uLow: { value: new THREE.Color(cfg.low) },
        uHigh: { value: new THREE.Color(cfg.high) },
        uFill: { value: new THREE.Color(cfg.fill) },
        uFillA: { value: cfg.fillA },
        uBright: { value: cfg.bright },
        uClickT: { value: -1 },
        uRim: { value: cfg.rim },
        uShade: { value: cfg.shade },
        uFade: { value: cfg.fade },
        uScanW: { value: cfg.scanW },
        uScanGain: { value: cfg.scanGain },
        uAppear: { value: 0 },
      },
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.set(cfg.size, 1, cfg.size);
    mesh.renderOrder = LAYERS.length - k; // bottom slab first, terrain last
    stack.add(mesh);
    return { mesh, material, cfg };
  });

  // ── Pointer: tilt + hover glow + click ripple ──
  const pointer = { x: 0, y: 0, inside: false };
  const target = { yaw: 0, pitch: 0 };
  const raycaster = new THREE.Raycaster();
  const hoverPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.18);
  const hit = new THREE.Vector3();
  let clickStart = -1;

  function updatePointer(event) {
    const r = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - r.left) / r.width) * 2 - 1;
    pointer.y = -((event.clientY - r.top) / r.height) * 2 + 1;
    pointer.inside = Math.abs(pointer.x) <= 1 && Math.abs(pointer.y) <= 1;
  }
  // Pointer on the terrain, in the stack's local x/z (null when off the tile).
  function terrainPoint() {
    raycaster.setFromCamera(pointer, camera);
    const local = raycaster.ray.clone().applyMatrix4(layers[0].mesh.matrixWorld.clone().invert());
    if (!local.intersectPlane(hoverPlane, hit) || Math.abs(hit.x) > 1.05 || Math.abs(hit.z) > 1.05) return null;
    return hit;
  }
  window.addEventListener('pointermove', updatePointer, { passive: true });
  window.addEventListener('touchmove', (e) => updatePointer(e.touches[0]), { passive: true });   // iOS: a finger drag scrolls, which cancels pointer events
  window.addEventListener('pointerdown', (event) => {
    updatePointer(event);
    const p = pointer.inside && terrainPoint();
    if (!p) return;
    shared.uClick.value.set(p.x, p.z);
    clickStart = performance.now();
  });

  // ── Size, visibility and loop ──
  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize, { passive: true });

  let running = false, onScreen = true, raf = 0, appearStart = -1, raiseStart = -1; // terrain / slabs entrance
  const t0 = performance.now();
  function frame(now) {
    const t = (now - t0) / 1000;
    // Hover tilt and glow ease toward their targets.
    // No tilt until the entrance is done, so the terrain meets the scan cloud at its rest pose.
    const steer = pointer.inside && (appearStart === -Infinity || (appearStart >= 0 && now - appearStart > APPEAR_MS));
    target.yaw = steer ? pointer.x * 0.16 : 0;
    target.pitch = steer ? -pointer.y * 0.07 : 0;
    stack.rotation.y += (BASE_YAW + target.yaw - stack.rotation.y) * 0.05;
    stack.rotation.x += (target.pitch - stack.rotation.x) * 0.05;
    stack.updateMatrixWorld();
    const p = pointer.inside && terrainPoint();
    if (p) shared.uHover.value.lerp(new THREE.Vector2(p.x, p.z), 0.25);
    shared.uHoverOn.value += ((p ? 1 : 0) - shared.uHoverOn.value) * 0.08;

    const since = clickStart < 0 ? -1 : (now - clickStart) / 1000;
    layers.forEach(({ mesh, material, cfg }, k) => {
      // Low-gravity float: each layer bobs vertically, slightly out of phase.
      const bob = reduceMotion ? 0 : Math.sin((t / FLOAT_PERIOD) * Math.PI * 2 - cfg.phase) * cfg.amp;
      // Entrance: rise from below with an ease-out, fading in; each lower layer trails.
      // Two tracks: the terrain from appear(), the slabs from raise() (or appear(), whichever comes first).
      const from = k ? raiseStart : appearStart, lag = k ? (k - 1) * APPEAR_LAG : 0;
      const ap = from === -Infinity ? 1 : from < 0 ? 0 : Math.min(1, Math.max(0, (now - from - lag) / APPEAR_MS));
      const rise = 1 - (1 - ap) ** 3;
      // The float eases in slowly (ap^2), so the terrain holds the scan cloud's rest pose through the cross-fade.
      mesh.position.y = -cfg.drop + bob * ap * ap - (k ? (1 - rise) * APPEAR_RISE : 0);
      material.uniforms.uAppear.value = Math.min(1, ap * 1.6);
      const local = since < 0 ? -1 : since - k * LAYER_DELAY;
      material.uniforms.uClickT.value = local < 0 || local > 3 || reduceMotion ? -1 : local;
    });
    // Scan: starts 1.5s before the stack settles, then repeats the pass / gap / pass / rest pair.
    const scanT = appearStart === -Infinity ? t : appearStart < 0 ? -1
      : (now - appearStart - APPEAR_MS - (LAYERS.length - 1) * APPEAR_LAG) / 1000 + 1.5; // starts 1.5s before the stack settles
    const c = scanT < 0 ? -1 : scanT % (2 * SCAN_PASS + SCAN_REST);
    const span = SCAN_HIGH - SCAN_LOW;
    shared.uScanY.value = c < 0 ? SCAN_HIGH + 2 // not started: parked out of sight
      : c < SCAN_PASS ? SCAN_HIGH - span * (c / SCAN_PASS) // top -> bottom (then the short gap below)
      : c < 2 * SCAN_PASS ? SCAN_LOW + span * ((c - SCAN_PASS) / SCAN_PASS) // bottom -> top
      : SCAN_HIGH + 2; // long rest, parked above
    renderer.render(scene, camera);
    raf = running && onScreen && !document.hidden ? requestAnimationFrame(frame) : 0;
  }
  const kick = () => { if (running && onScreen && !document.hidden && !raf) raf = requestAnimationFrame(frame); };
  new IntersectionObserver(([entry]) => { onScreen = entry.isIntersecting; kick(); }).observe(container);
  document.addEventListener('visibilitychange', kick);

  renderer.render(scene, camera); // first frame ready before it fades in
  return {
    // The terrain's vertices at rest (base yaw, no tilt, no float), every `step`-th per side, projected into the
    // container: [x, y] in CSS px, height 0..1 and the shader's near-side factor, for the intro's point cloud.
    cloud(step = 2) {
      const w = container.clientWidth, h = container.clientHeight;
      const model = new THREE.Matrix4().makeRotationY(BASE_YAW)
        .multiply(new THREE.Matrix4().makeTranslation(0, -LAYERS[0].drop, 0));
      resize(); // the camera's aspect must match the box being measured
      camera.updateMatrixWorld();
      const pos = geometry.attributes.position, n = GRID + 1, v = new THREE.Vector3(), out = [];
      for (let j = 0; j <= GRID; j += step) {
        for (let i = 0; i <= GRID; i += step) {
          const k = j * n + i;
          v.set(pos.getX(k), heights[k] * PEAK * LAYERS[0].relief, pos.getZ(k)).applyMatrix4(model);
          const depth = -v.clone().applyMatrix4(camera.matrixWorldInverse).z;
          const near = Math.min(1, Math.max(0, (6.2 - depth) / 2.6));
          v.project(camera);
          out.push([(v.x + 1) / 2 * w, (1 - v.y) / 2 * h, heights[k], near * near * (3 - 2 * near)]);
        }
      }
      return out;
    },
    start() { running = true; kick(); },
    // Float the stack up into existence (instant under reduced motion): the terrain fades in, and the slabs rise unless
    // raise() already started them.
    appear() {
      if (appearStart >= 0) return;
      appearStart = reduceMotion ? -Infinity : performance.now();
      if (raiseStart < 0) raiseStart = appearStart;
      container.classList.add('is-in');
      kick();
    },
    // Start the slabs' rise on their own, ahead of the terrain.
    raise() {
      if (raiseStart >= 0 || raiseStart === -Infinity) return;
      raiseStart = reduceMotion ? -Infinity : performance.now();
      kick();
    },
  };
}
