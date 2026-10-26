// Landmarks row in 3D. Cells 1-3: their live three.js scenes, pre-built by scripts/bake_landmarks3d.mjs into
// public/landmarks3d/*.binz (fetched during the intro, unpacked off the critical path), drawn the way their images are made
// (card size 960 x 1280, a warm-only pass, glow and golden-glow layers screened over it, edge fade), scaled into the cell.
// Cell 4: the owner's villa image, unchanged, given depth (Depth Anything V2 Large) and an extended background.
// One WebGL canvas over the whole row draws every cell in its own viewport: one context (browsers cap and drop extra
// ones), shared render and blur targets, each cell keeping only its composed 8-bit image. If the browser drops the
// context, the cells fall back to their still images.
// Behaviour (owner): the structure turns toward the mouse, up to 35 deg. While the row is at least 30% on screen and a
// cell is idle, it zooms in a little, very slowly, then back, over and over; the mouse over a cell (or a tap) eases the
// zoom to a stop where it is and brightens the cell a little, and the idle cycle resumes when it leaves. Below 30%
// everything holds still. Three.js r186, same as DepthWizard2.
import * as THREE from 'three';
import villaImg from './landmarks3d/villa-ext.webp';
import villaDepth from './landmarks3d/villa-ext-depth.png';

const MAX_DEG = 35;      // tilt toward the mouse
const CYCLE_S = 7;       // seconds for an idle zoom in (and again out)
const EASE_S = 0.9;      // time constant of the zoom restarting after a hover (it stops in 0.35)
const BRIGHT = 1.12;     // hover brightness
const FADE_S = 1.4, STAGGER_S = 0.18;   // each structure fades in (smoothstep) once the row is on screen, left to right (owner 2026-10-25: they popped in)
const VISIBLE = 0.3;     // the row animates only while at least this much of it is on screen
const ZMAX = [1.12, 1.08, 1.14, 1.10];   // idle zoom: visible but small (owner), cell 2 the least
// per live cell: structure size and downward shift in its box (fractions of the cell), teal glow strength (owner: cell 1
// 5% smaller with its glow a little dimmer, cell 2 17% smaller and a bit lower; cell 4's 12% is in its camera zoom below)
const FIT = [[0.95, 0, 0.8], [0.83, 0.05, 1], [1, 0, 1]];
const SCENES = ['curv', 'bas', 'antilia'].map((n) => `/home/landmarks3d/${n}.binz`);

// ── loading: fetched now (during the intro), unpacked as it arrives ──
async function loadScene(url) {
  const res = await fetch(url, { priority: 'low' });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = await new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const hl = new DataView(buf).getUint32(0, true), header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const base = (4 + hl + 3) & ~3;
  const MAT = { MeshBasicMaterial: THREE.MeshBasicMaterial, LineBasicMaterial: THREE.LineBasicMaterial, PointsMaterial: THREE.PointsMaterial };
  const scene = new THREE.Scene(), curtains = new Set();
  for (const o of header.objects) {
    const g = new THREE.BufferGeometry();
    for (const [k, a] of Object.entries(o.attrs)) {
      if (k === 'range') continue;
      if (k === 'index') g.setIndex(new THREE.BufferAttribute(a.t === 'u32' ? new Uint32Array(buf, base + a.at, a.n) : new Uint16Array(buf, base + a.at, a.n), 1));
      else g.setAttribute(k, a.t === 'u8' ? new THREE.BufferAttribute(new Uint8Array(buf, base + a.at, a.n), a.size, true) : new THREE.BufferAttribute(new Float32Array(buf, base + a.at, a.n), a.size));
    }
    if (o.attrs.range) g.setDrawRange(...o.attrs.range);
    const { type, color, ...props } = o.mat, m = new MAT[type](props);
    if (color !== undefined) m.color.setHex(color);
    const obj = new (o.type === 'Mesh' ? THREE.Mesh : o.type === 'Points' ? THREE.Points : THREE.LineSegments)(g, m);
    obj.matrixAutoUpdate = false; obj.matrix.fromArray(o.matrix); obj.visible = o.visible; obj.renderOrder = o.renderOrder; obj.userData.warm = o.warm;
    if (o.curtain) curtains.add(m);
    scene.add(obj);
  }
  const cam = new THREE.PerspectiveCamera(); const c = header.camera;
  new THREE.Matrix4().fromArray(c.matrix).decompose(cam.position, cam.quaternion, cam.scale);
  cam.near = c.near; cam.far = c.far; cam.projectionMatrix.fromArray(c.projection); cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
  return { scene, cam, curtains, P: header.params };
}

// ── full-screen passes ──
const FULL = 'attribute vec3 position; attribute vec2 uv; varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }';
const raw = (frag, uniforms) => new THREE.RawShaderMaterial({ vertexShader: FULL, fragmentShader: 'precision highp float; varying vec2 vUv; ' + frag, uniforms, depthTest: false, depthWrite: false });
const blurMat = raw(`uniform sampler2D t; uniform vec2 d; uniform float s;
  void main() { vec4 a = vec4(0.0); float w = 0.0; for (int i = -9; i <= 9; i++) { float x = float(i); if (x * x > 9.0 * s * s) continue; float g = exp(-0.5 * x * x / (s * s)); a += texture2D(t, vUv + d * x) * g; w += g; } gl_FragColor = a / w; }`,
  { t: { value: null }, d: { value: new THREE.Vector2() }, s: { value: 1 } });
// once per tilt: the base with the six blurred layers screened over it, edge fade
const composeMat = raw(`uniform sampler2D base, g0, g1, g2, w0, w1, w2; uniform vec3 kg, kw, f;
  vec4 scr(vec4 a, vec4 b) { return a + b - a * b; }
  void main() {
    vec4 c = texture2D(base, vUv);
    c = scr(c, texture2D(g0, vUv) * kg.x); c = scr(c, texture2D(g1, vUv) * kg.y); c = scr(c, texture2D(g2, vUv) * kg.z);
    c = scr(c, texture2D(w0, vUv) * kw.x); c = scr(c, texture2D(w1, vUv) * kw.y); c = scr(c, texture2D(w2, vUv) * kw.z);
    float m = smoothstep(0.0, f.x, vUv.x) * smoothstep(1.0, 1.0 - f.x, vUv.x) * smoothstep(1.0, 1.0 - f.y, vUv.y) * smoothstep(0.0, 1.0 - f.z, vUv.y);
    gl_FragColor = c * m; }`,
  Object.fromEntries(['base', 'g0', 'g1', 'g2', 'w0', 'w1', 'w2', 'kg', 'kw', 'f'].map((k) => [k, { value: ['kg', 'kw', 'f'].includes(k) ? new THREE.Vector3() : null }])));
// every frame, per cell: zoom about the centre (mip-biased: crisp lines), brightness, fade-in (premultiplied output)
const showMat = raw(`uniform sampler2D a; uniform float z, bright, op; uniform vec3 fit;
  void main() { vec4 c = texture2D(a, 0.5 + (vUv + vec2(0.0, fit.y) - 0.5) / (z * fit.x), -1.0); c.rgb = min(c.rgb * bright, vec3(c.a)); gl_FragColor = c * op; }`,
  { a: { value: null }, z: { value: 1 }, bright: { value: 1 }, op: { value: 0 }, fit: { value: new THREE.Vector3(1, 0, 0) } });

export function initLandmarks3d(row) {
  const cells = [...row.querySelectorAll('.cells__cell')];
  const loads = SCENES.map((u) => loadScene(u));                                       // start downloading now
  const canvas = document.createElement('canvas'); canvas.className = 'cells__3d'; canvas.setAttribute('aria-hidden', 'true');
  row.append(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, premultipliedAlpha: true });
  renderer.setClearColor(0, 0); renderer.autoClear = false;
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2)), qs = new THREE.Scene(), qc = new THREE.Camera(); qs.add(quad);
  const pass = (mat, target) => { quad.material = mat; renderer.setRenderTarget(target); renderer.render(qs, qc); };

  // shared working targets (half float, in the images' sRGB values), made on first use
  const pool = new Map();
  const shared = (key, w, h) => {
    let r = pool.get(key);
    if (!r) { r = new THREE.WebGLRenderTarget(Math.ceil(w), Math.ceil(h), { type: THREE.HalfFloatType }); r.texture.colorSpace = THREE.SRGBColorSpace; pool.set(key, r); }
    return r;
  };

  const live = ({ scene, cam, curtains, P }, [scale, down, glow]) => {
    const W = P.w, H = P.h;
    const composed = new THREE.WebGLRenderTarget(W, H, { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter });
    const layers = [...P.glow.map(([sg]) => ['g', sg]), ...P.warmGlow.map(([sg]) => ['w', sg])].map(([src, sigma], i) => {
      const sc = Math.max(src === 'w' ? 2 : 1, Math.ceil(sigma / 3)); return { src, sg: sigma / sc, sc, key: `${W}x${H}/${i}` };
    });
    const saved = [];
    const warmState = (on) => {
      if (on) scene.traverse((o) => {
        saved.push([o, o.visible, o.material, o.renderOrder]);
        if (o.userData.warm) o.visible = true;
        else if (o.isMesh && !curtains.has(o.material)) { const m = o.material.clone(); Object.assign(m, { colorWrite: false, transparent: false, depthWrite: true }); o.material = m; o.renderOrder = -1; }
        else if (o.isMesh || o.isLine || o.isPoints) o.visible = false;
      });
      else { for (const [o, v, m, r] of saved) { if (o.material !== m) o.material.dispose(); o.visible = v; o.material = m; o.renderOrder = r; } saved.length = 0; }
    };
    const dir = cam.getWorldDirection(new THREE.Vector3()), along = -cam.position.dot(dir);
    const target = cam.position.clone().addScaledVector(dir, Math.max(along, 1)), offset = cam.position.clone().sub(target), up = new THREE.Vector3(0, 1, 0);
    return {
      compose(yaw, pitch) {                                   // the expensive part: only when the tilt changes
        const o = offset.clone().applyAxisAngle(up, THREE.MathUtils.degToRad(-yaw));
        o.applyAxisAngle(new THREE.Vector3().crossVectors(up, o).normalize(), THREE.MathUtils.degToRad(pitch));
        cam.position.copy(target).add(o); cam.lookAt(target);
        const base = shared(`${W}x${H}/base`, W, H), warm = shared(`${W}x${H}/warm`, W / 2, H / 2), u = composeMat.uniforms;
        renderer.setRenderTarget(base); renderer.clear(); renderer.render(scene, cam);
        warmState(true); renderer.setRenderTarget(warm); renderer.clear(); renderer.render(scene, cam); warmState(false);
        u.base.value = base.texture;
        for (const [i, L] of layers.entries()) {
          const a = shared(L.key + 'a', W / L.sc, H / L.sc), b = shared(L.key + 'b', W / L.sc, H / L.sc);
          blurMat.uniforms.t.value = (L.src === 'g' ? base : warm).texture; blurMat.uniforms.s.value = L.sg; blurMat.uniforms.d.value.set(1 / a.width, 0); pass(blurMat, a);
          blurMat.uniforms.t.value = a.texture; blurMat.uniforms.d.value.set(0, 1 / a.height); pass(blurMat, b);
          u[['g0', 'g1', 'g2', 'w0', 'w1', 'w2'][i]].value = b.texture;
        }
        u.kg.value.set(...P.glow.map(([, k]) => k * glow)); u.kw.value.set(...P.warmGlow.map(([, k]) => k)); u.f.value.set(...P.edgeFade);
        pass(composeMat, composed);                                                        // three makes its mipmaps
      },
      draw(z, bright, op) { const u = showMat.uniforms; u.a.value = composed.texture; u.z.value = z; u.bright.value = bright; u.op.value = op; u.fit.value.set(scale, down, 0); quad.material = showMat; renderer.render(qs, qc); },
    };
  };
  const parallax = async () => {
    const load = (url) => new THREE.TextureLoader().loadAsync(url);
    const [map, dep] = await Promise.all([load(villaImg), load(villaDepth)]);
    map.colorSpace = THREE.NoColorSpace; dep.colorSpace = THREE.NoColorSpace;          // the picture's own values, unchanged
    const RELIEF = 0.22, aspect = 4 / 3, ex = 1.6, ey = 1.5;                            // background extended 30% / 25% a side
    const mat = new THREE.ShaderMaterial({
      transparent: true, side: THREE.DoubleSide,
      uniforms: { map: { value: map }, dep: { value: dep }, relief: { value: RELIEF }, bright: { value: 1 }, op: { value: 0 } },
      vertexShader: 'uniform sampler2D dep; uniform float relief; varying vec2 vUv; void main() { vUv = uv; vec3 p = position; p.z += texture2D(dep, uv).r * relief; gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0); }',
      fragmentShader: `uniform sampler2D map; uniform float bright, op; varying vec2 vUv;
        void main() { vec4 c = texture2D(map, vUv);
          c.a *= op * smoothstep(0.08, 0.24, max(c.r, max(c.g, c.b)))   // its dark grey ground fades out: page black like the other cells
            * smoothstep(0.0, 0.07, vUv.x) * smoothstep(1.0, 0.93, vUv.x) * smoothstep(1.0, 0.94, vUv.y) * smoothstep(0.0, 0.08, vUv.y);
          float hi = max(c.r, max(c.g, c.b)), line = smoothstep(0.3, 0.55, hi) * smoothstep(0.08, 0.25, hi - min(c.r, min(c.g, c.b)));   // bright, coloured: a line
          gl_FragColor = vec4(min(mix(c.rgb, c.rgb * vec3(0.9, 1.3, 1.25), line) * bright, vec3(1.0)), c.a); }`,   // only the lines pushed to a brighter teal (owner)
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(ex, aspect * ey, Math.round(256 * ex), Math.round(256 * aspect * ey)), mat);
    const scene = new THREE.Scene(); scene.add(mesh);
    const cam = new THREE.PerspectiveCamera(30, 3 / 4, 0.1, 10);
    cam.position.z = (aspect / 2) / Math.tan(THREE.MathUtils.degToRad(15)) + RELIEF / 2;
    return {
      compose(yaw, pitch) { mesh.rotation.set(THREE.MathUtils.degToRad(pitch), THREE.MathUtils.degToRad(yaw), 0); },
      draw(z, bright, op) { cam.zoom = z * 0.88;   // owner: the villa 12% smaller (its extended background fills the rest)
         cam.updateProjectionMatrix(); mat.uniforms.bright.value = bright; mat.uniforms.op.value = op; renderer.clearDepth(); renderer.render(scene, cam); },
    };
  };

  // ── behaviour ──
  const state = cells.map((cell, i) => ({ cell, i, age: 0, zmax: ZMAX[i], z: 1, ph: 0, k: 1, hover: false, held: false, b: 1, op: 0, yaw: 0, pitch: 0, ty: 0, tp: 0, view: null, dirty: true }));
  let active = false, raf = 0, last = 0, lost = false;
  const ease = (dt, tau) => 1 - Math.exp(-dt / tau);
  const step = (s, dt) => {
    const go = s.hover || s.held ? 0 : 1;                                               // hovered / tapped: the zoom eases to a stop
    s.k += (go - s.k) * ease(dt, go ? EASE_S : 0.35);
    s.ph += s.k * dt * Math.PI / CYCLE_S;                                               // a cosine cycle: slow, smooth turns at both ends
    s.z = 1 + (s.zmax - 1) * (1 - Math.cos(s.ph)) / 2;
    s.b += ((s.hover ? BRIGHT : 1) - s.b) * ease(dt, 0.35);
    s.age += dt; const a = Math.min(1, Math.max(0, (s.age - s.i * STAGGER_S) / FADE_S)); s.op = a * a * (3 - 2 * a);
    const k = ease(dt, 0.12), ny = s.yaw + (s.ty - s.yaw) * k, np = s.pitch + (s.tp - s.pitch) * k;
    if (Math.abs(ny - s.yaw) > 0.005 || Math.abs(np - s.pitch) > 0.005) { s.yaw = ny; s.pitch = np; s.dirty = true; }
  };
  const frame = (t) => {
    raf = 0; if (lost) return;
    const dt = last ? Math.min(0.05, (t - last) / 1000) : 0; last = t;
    for (const s of state) if (s.view) { step(s, dt); if (s.dirty) { s.view.compose(s.yaw, s.pitch); s.dirty = false; } }
    const rb = row.getBoundingClientRect(), ox = rb.left + row.clientLeft, oy = rb.top + row.clientTop, H = row.clientHeight;
    renderer.setRenderTarget(null); renderer.setScissorTest(false); renderer.clear();
    renderer.setScissorTest(true);
    for (const s of state) {
      if (!s.view) continue;
      const r = s.cell.getBoundingClientRect(), x = r.left - ox, y = H - (r.top - oy) - r.height;   // GL origin: bottom left
      renderer.setViewport(x, y, r.width, r.height); renderer.setScissor(x, y, r.width, r.height);
      s.view.draw(s.z, s.b, s.op);
    }
    renderer.setScissorTest(false); renderer.setViewport(0, 0, row.clientWidth, H);
    if (active) raf = requestAnimationFrame(frame);
  };
  const start = () => { if (!raf && active && !lost) { last = 0; raf = requestAnimationFrame(frame); } };
  new IntersectionObserver(([e]) => { active = e.intersectionRatio >= VISIBLE; start(); }, { threshold: [0, VISIBLE, 0.6, 1] }).observe(row);
  new ResizeObserver(() => { renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); renderer.setSize(row.clientWidth, row.clientHeight, false); if (!raf && !lost) frame(performance.now()); }).observe(row);

  for (const s of state) {
    const aim = (p) => {
      const r = s.cell.getBoundingClientRect();
      s.ty = Math.max(-1, Math.min(1, ((p.clientX - r.left) / r.width) * 2 - 1)) * MAX_DEG;
      s.tp = Math.max(-1, Math.min(1, ((p.clientY - r.top) / r.height) * 2 - 1)) * MAX_DEG;
    };
    const free = () => { s.hover = false; s.ty = 0; s.tp = 0; };
    s.cell.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') s.hover = true; });
    s.cell.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') aim(e); });
    s.cell.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') free(); });
    // touch: a finger on the cell acts as the hover, tilting toward it (touch events, as iOS cancels pointer events once the page scrolls)
    s.cell.addEventListener('touchstart', (e) => { s.hover = true; aim(e.touches[0]); }, { passive: true });
    s.cell.addEventListener('touchmove', (e) => aim(e.touches[0]), { passive: true });
    for (const type of ['touchend', 'touchcancel']) s.cell.addEventListener(type, free, { passive: true });
    s.cell.addEventListener('pointerup', (e) => { if (e.pointerType !== 'mouse') s.held = !s.held; });   // touch: a tap holds or frees the zoom
  }

  // the browser dropped the 3D (GPU memory, too many contexts): back to the still images
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault(); lost = true; cancelAnimationFrame(raf); raf = 0;
    canvas.hidden = true; cells.forEach((c) => c.classList.remove('is-3d'));
    console.warn('landmarks 3D: WebGL context lost, showing the still images');
  });

  const ready = (s, view) => { s.view = view; s.cell.classList.add('is-3d'); start(); };
  loads.forEach((p, i) => p.then((data) => ready(state[i], live(data, FIT[i]))).catch((err) => console.warn(`landmarks 3D: cell ${i + 1} stays a still image:`, err)));
  parallax().then((view) => ready(state[3], view)).catch((err) => console.warn('landmarks 3D: cell 4 stays a still image:', err));
}
