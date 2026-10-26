import { splitChars, setupSections } from './sections.js';
import '@fontsource-variable/geist';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
import '@fontsource/sora/latin-300.css';   // the opening statement and the hero lead (owner 2026-10-16)
import '@fontsource/sora/latin-500.css';   // card titles (H3)
import '@fontsource/sora/latin-600.css';   // section titles (H2)
import '@fontsource/jetbrains-mono/latin-500.css';   // tall digits for the hero's live coordinates
import './styles.css';

// Every load starts at the top of the page; the intro hands off to the hero heading.
// A hash naming an element on the page (e.g. #demo-video) skips the intro and opens at that element instead.
let deepLink = null;
try { deepLink = location.hash.length > 1 ? document.getElementById(decodeURIComponent(location.hash.slice(1))) : null; } catch { /* malformed hash: no deep link */ }
history.scrollRestoration = 'manual';
if (!deepLink) window.scrollTo(0, 0);

const intro = document.querySelector('#intro');
const introCanvas = document.querySelector('#intro-canvas');
const introCtx = introCanvas.getContext('2d');
const contoursCanvas = document.querySelector('#intro-contours');
const contoursCtx = contoursCanvas.getContext('2d');
const cloudLayer = document.querySelector('#intro-cloud-layer'); // the points' canvas, moved and faded on the compositor
const cloudCanvas = document.querySelector('#intro-cloud');
const cloudCtx = cloudCanvas.getContext('2d');
const heroTitle = document.querySelector('#hero-title');
const site = document.querySelector('#site');
const cursorArrow = document.querySelector('#cursor-arrow');
const cursorSquare = document.querySelector('#cursor-square');
const trailCanvas = document.querySelector('#cursor-trail');
const trailCtx = trailCanvas.getContext('2d');
const scanHorizontal = document.querySelector('#scan-horizontal');
const scanVertical = document.querySelector('#scan-vertical');
const introCrosshair = document.querySelector('#intro-crosshair');
const introShapes = document.querySelector('#intro-shapes');
const readoutLat = document.querySelector('#readout-lat');
const readoutLon = document.querySelector('#readout-lon');

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;

let introW = 0;
let introH = 0;
let dpr = Math.min(window.devicePixelRatio || 1, 2);
let introStart = performance.now();
let rafId = 0;
let axisRaf = 0;
let introSequenceTimer = 0;
let introFinished = false;

const VERTICAL_X = 0.765;
const HORIZONTAL_START_Y = 0.505;
const HORIZONTAL_END_Y = 0.315;
// t: the horizontal axis reaches END here, eases into a slow crawl, and the wordmark starts.
const HORIZONTAL_CLIMB_MS = 2600;
// The whole intro timeline (axes, crosshair, status, letters, shapes, float) starts this far in.
// The background grid keeps its own clock.
const INTRO_OFFSET_MS = 1500;
const HORIZONTAL_CRAWL = 0.12;
// Loader: the hero terrain's points rise from below the screen haphazardly, all together, in the centre of the screen,
// each flying straight to its own place on the mountain, so the arriving points build its body. Once all have landed
// and the page has loaded, the returns at once settle (brightening, and staying bright), the mountain slides to where the hologram
// sits on the page, and the
// hologram fades in on it (hologram.js draws the terrain at the same rest pose).
const SCAN_MS = 3600; // loading progress, paced (held at 90% until the page is ready); it gates the hand-off
// Corner contours: a faint light front flows out from each corner, ring by ring, while the particles rise (finishing as
// the last lands); they fade out during the settle.
const CONTOUR_IN_MS = 700; // the contours fade in with the intro
const RING_REST = 0.04, RING_LIT = 0.2, RING_CREST = 0.18; // unlit, lit, the front's crest (before the corner falloff)
const FRONT_SOFT = 0.22; // the front's half-width, as a fraction of the ring stack
const CLOUD_STEP = 2; // every 2nd terrain vertex per side: ~10k points
const CLOUD_NOISE = 2.6; // px, spread of the returns' vertical noise before they settle
const SETTLE_MS = 700; // the noise resolving onto the surface, starting the moment every point has landed
const FLOW_SPAN_MS = 1000; // every point sets off at its own random moment within this window (shorter = more rising at once)
const FLOW_MS = 1500; // each point's rise from below the screen (cubic ease-out, so it decelerates into place)
// Point colour by brightness bucket: deep teal -> teal -> pale teal (the hologram's own low/high line colours).
const CLOUD_RAMP = Array.from({ length: 10 }, (_, i) => {
  const t = i / 9, mix = (a, b, u) => Math.round(a + (b - a) * u);
  const [r, g, b] = t < 0.6 ? [42, 157, 137].map((c, k) => mix(c, [89, 217, 192][k], t / 0.6))
    : [89, 217, 192].map((c, k) => mix(c, [200, 255, 244][k], (t - 0.6) / 0.4));
  return `rgba(${r},${g},${b},${(0.22 + 0.73 * t).toFixed(2)})`;
});
// The status line starts this long before the letters (its CSS timeline ends before the float).
 // crawl speed as a fraction of the climb speed
const HORIZONTAL_EASE_MS = 280; // how quickly the climb decays into the crawl
// Axes animate until the intro has faded out (~8.7s of axis time); ticks cover this long.
// ponytail: fixed coverage window; if the intro grows past it the scales simply stop at the end.
const AXIS_RUN_MS = 9000;
const INTRO_FADE_MS = 800; // keep in sync with .intro.is-done opacity transition
// Both scales travel at this one speed (viewport heights / second); tune here.
const AXIS_SPEED = 0.28;
const MAJOR_STEP = 120; // px between labelled ticks
const MINOR_PER_MAJOR = 4;

const axisState = {
  start: 0,
  horizontalY: HORIZONTAL_START_Y,
  intersectionX: 0,
  intersectionY: 0,
};
let scales = null;

function resizeIntroCanvas() {
  introW = window.innerWidth;
  introH = window.innerHeight;
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  introCanvas.width = Math.floor(introW * dpr);
  introCanvas.height = Math.floor(introH * dpr);
  introCanvas.style.width = `${introW}px`;
  introCanvas.style.height = `${introH}px`;
  Object.assign(cloudCanvas, { width: introCanvas.width, height: introCanvas.height });
  cloudCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  Object.assign(contoursCanvas, { width: introCanvas.width, height: introCanvas.height });
  introCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  buildContours();
  buildCloud();
  renderIntroBackground(0);
  if (!introFinished) setupAxes(); // tick coverage depends on viewport size
}

// Topographic contours around two off-screen "hills", at the top-right and bottom-left corners.
// Rings share one wobble shape and drift only slightly per ring, so they nest without crossing.
// Each ring is one continuous stroke (no per-segment caps, so no beading), faded toward the screen centre by a cached
// radial mask and lit by the scan through a vertical mask. Every 4th is an index contour (heavier, as on survey maps).
let contourMask = null, contourScratch = null, contourHills = [];
let scanP = 0, scanDoneAt = 0, scanLast = 0;
let cloudSlideAt = 0; // the hand-off: the points' canvas is frozen from here
let cloudMoving = false; // the slide has started (it leads the hand-off by SLIDE_LEAD_MS)
let cloud = null; // { x, y, b, n }: the hologram's terrain at rest, in viewport px (set once hologram.js has loaded)
let pageLoaded = document.readyState === 'complete';
addEventListener('load', () => { pageLoaded = true; }, { once: true });

function ringPoint({ cx, cy, r0, gap, seed }, k, th) {
  const f = 1 + 0.09 * Math.sin(3 * th + seed) + 0.05 * Math.sin(5 * th + 2 * seed + k * 0.15)
    + 0.03 * Math.sin(9 * th + 3 * seed);
  const radius = (r0 + k * gap) * f;
  return [cx + radius * Math.cos(th), cy + radius * 0.85 * Math.sin(th)];
}

// Corner falloff: smoothstep of the normalised elliptical radius squared (0 at r^2 = .55, 1 at 1.15).
function cornerFalloff(x, y) {
  const nx = (2 * x) / introW - 1, ny = (2 * y) / introH - 1;
  return smoothstep((nx * nx + ny * ny - 0.55) / 0.6);
}
const smoothstep = (u) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

function buildContours() {
  const make = () => Object.assign(document.createElement('canvas'), { width: introCanvas.width, height: introCanvas.height });
  contourMask = make(); contourScratch = make();
  contourHills = [
    { cx: introW + 40, cy: -60, rings: 12, r0: 46, gap: 32, seed: 1.3 },
    { cx: -50, cy: introH + 40, rings: 12, r0: 46, gap: 32, seed: 4.1 },
  ];
  for (const hill of contourHills) {
    hill.paths = [];
    for (let k = 0; k < hill.rings; k += 1) {
      const n = Math.max(96, Math.round((hill.r0 + k * hill.gap) / 3));
      const path = new Path2D();
      let shows = false;
      for (let i = 0; i <= n; i += 1) {
        const [x, y] = ringPoint(hill, k, (i / n) * Math.PI * 2);
        path.lineTo(x, y);
        shows ||= x > 0 && x < introW && y > 0 && y < introH && cornerFalloff(x, y) > 0.15;
      }
      if (shows) hill.paths.push({ path, index: k % 4 === 3 });
    }
    hill.paths.forEach((ring, i) => { ring.u = (i + 0.5) / hill.paths.length; }); // place in the stack: 0 corner .. 1 centre
  }
  // Corner falloff: smoothstep of the normalised elliptical radius squared (0 at r^2 = .55, 1 at 1.15).
  const c = contourMask.getContext('2d');
  c.setTransform(dpr * introW / 2, 0, 0, dpr * introH / 2, dpr * introW / 2, dpr * introH / 2);
  const g = c.createRadialGradient(0, 0, 0, 0, 0, 1.2);
  for (let i = 0; i <= 24; i++) {
    const r = 0.7 + (i / 24) * 0.4, t = Math.min(1, Math.max(0, (r * r - 0.55) / 0.6));
    g.addColorStop(r / 1.2, `rgba(0,0,0,${t * t * (3 - 2 * t)})`);
  }
  g.addColorStop(0, 'rgba(0,0,0,0)');
  c.fillStyle = g;
  c.fillRect(-2, -2, 4, 4);
}

// The cloud: the hologram's terrain vertices at rest, placed over the hero's hologram box, each with a bit of vertical
// noise (summed uniforms, roughly normal) that settles once all have landed, and its flow: a random delay, a start below the screen and a sideways sway. Drawn centred on the
// screen until the hand-off slides it to the hologram's place.
function buildCloud() {
  const box = holo?.getBoundingClientRect();
  if (!hologram || !box?.width) { cloud = null; return; }
  const pts = hologram.cloud(CLOUD_STEP);
  const n = pts.length, x = new Float32Array(n), y = new Float32Array(n), b = new Float32Array(n), noise = new Float32Array(n);
  pts.forEach(([px, py, h, near], i) => {
    x[i] = box.left + px; // where the hologram sits on the page
    y[i] = box.top + py;
    // The hologram's own line brightness: ridges and the near side brighter.
    b[i] = (0.55 + 0.45 * Math.min(1, h * 1.4)) * (0.72 + 0.28 * near);
    noise[i] = (Math.random() + Math.random() + Math.random() - 1.5) * 2 * CLOUD_NOISE;
  });
  const delay = new Float32Array(n), fromY = new Float32Array(n), sway = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    delay[i] = Math.random() * FLOW_SPAN_MS;
    fromY[i] = innerHeight + 20 + Math.random() * 160;
    sway[i] = (Math.random() - 0.5) * 36;
  }
  const flowEnd = FLOW_SPAN_MS + FLOW_MS;
  cloud = {
    at: performance.now() - introStart, x, y, b, noise, n, delay, fromY, sway, flowEnd,
    dx: innerWidth / 2 - (box.left + box.width / 2), dy: innerHeight / 2 - (box.top + box.height / 2), // centring offset
  };
}

// Loading progress: the eased intro clock, held at 90% until the page has loaded and the cloud is ready (narrow
// screens hide the hologram, so there the page load alone counts). Reduced motion: no sweep, the scan is complete.
// ponytail: window load + the hologram chunk only; feed per-asset progress here if the loader ever needs to be exact.
function scanProgress(time) {
  const paced = reduceMotion ? 1 : 0.5 - Math.cos(Math.PI * Math.min(1, time / SCAN_MS)) / 2;
  const ready = pageLoaded && (cloud || !holo?.offsetWidth);
  return Math.min(paced, ready ? 1 : 0.9);
}

function renderContours(time) {
  // A soft front lights the rings one by one, out from each corner, over the particles' rise; then a second front with
  // the same shape and easing slowly dims them again, from the centre back to the corners. When the particles land the
  // whole layer fades out on the compositor (CSS-smooth, whatever the main thread is doing), gone at the slide start, so
  // the dimming may be cut off part-way. Before the particles are ready the rings wait unlit; with no cloud (narrow
  // screens) the lighting runs over the loading progress and the layer leaves with the intro.
  if (cloud?.landedAt && !contoursCanvas.dataset.out) {
    contoursCanvas.dataset.out = '1';
    contoursCanvas.animate({ opacity: [1, 0] }, { duration: SETTLE_MS, easing: 'cubic-bezier(.45, 0, .55, 1)', fill: 'forwards' });
  }
  if (cloud?.landedAt && time - cloud.landedAt > SETTLE_MS) return; // faded out: stop drawing
  const start = cloud ? cloud.at : 0, span = cloud ? cloud.flowEnd : SCAN_MS; // one front's duration
  const waiting = !cloud && holo?.offsetWidth; // a cloud is coming (wide screens): stay unlit until it appears
  const q = waiting ? 0 : (time - start) / span; // 0..1 lighting, 1..2 dimming
  const across = (v) => -FRONT_SOFT + v * (1 + 2 * FRONT_SOFT); // from inside the corner to past the last ring
  const front = across(smoothstep(q));
  const dim = 1 - across(smoothstep(q - 1)); // from past the last ring back to inside the corner
  const alpha = Math.min(1, time / CONTOUR_IN_MS);
  const sc = contourScratch.getContext('2d');
  sc.setTransform(1, 0, 0, 1, 0, 0);
  sc.globalCompositeOperation = 'source-over';
  sc.clearRect(0, 0, contourScratch.width, contourScratch.height);
  sc.setTransform(dpr, 0, 0, dpr, 0, 0);
  sc.lineJoin = 'round';
  sc.strokeStyle = 'rgb(117,240,214)';
  for (const hill of contourHills) {
    for (const { path, index, u } of hill.paths) {
      const d = (front - u) / FRONT_SOFT; // -1 .. 1 across the front
      const lit = RING_REST + (RING_LIT - RING_REST) * smoothstep((d + 1) / 2) + RING_CREST * Math.exp(-((d * 1.6) ** 2));
      const a = lit * (1 - smoothstep(((u - dim) / FRONT_SOFT + 1) / 2)); // the dimming front, the same shape, reversed
      sc.globalAlpha = Math.min(1, index ? a * 1.2 : a);
      sc.lineWidth = index ? 1.9 : 1.3;
      sc.stroke(path);
    }
  }
  sc.globalAlpha = 1;
  sc.globalCompositeOperation = 'destination-in';
  sc.setTransform(1, 0, 0, 1, 0, 0);
  sc.drawImage(contourMask, 0, 0);
  contoursCtx.clearRect(0, 0, contoursCanvas.width, contoursCanvas.height);
  contoursCtx.globalAlpha = alpha;
  contoursCtx.drawImage(contourScratch, 0, 0);
}

// Returns above the line, bucketed by brightness into one path per colour (10 fills a frame, not 10k).
// The points, on their own canvas. From the slide start they are finished and still, so they stop being redrawn: the
// whole layer is then moved and faded by the compositor (slideCloud, formHologram), smooth even while the page loads in.
function renderCloud(time) {
  if (!cloud || cloudSlideAt) return;
  const t = time - cloud.at;
  // Every point has landed and the page has really loaded (not the paced progress clock): the returns settle at once.
  if (!cloud.landedAt && (reduceMotion || (pageLoaded && t >= cloud.flowEnd))) cloud.landedAt = time;
  const settle = reduceMotion ? 1 : cloud.landedAt ? Math.min(1, (time - cloud.landedAt) / SETTLE_MS) : 0;
  const ease = 1 - (1 - settle) ** 3;
  const ox = cloud.dx, oy = cloud.dy; // drawn centred on the screen
  const paths = CLOUD_RAMP.map(() => new Path2D());
  const { x, y, b, noise, n, delay, fromY, sway } = cloud;
  for (let i = 0; i < n; i += 1) {
    // Rise from below the screen straight to its place on the mountain, decelerating, swaying a little sideways.
    const u = reduceMotion ? 1 : Math.min(1, (t - delay[i]) / FLOW_MS);
    if (u <= 0) continue;
    const e = 1 - (1 - u) ** 3;
    const tx = x[i] + ox, ty = y[i] + oy + noise[i] * (1 - ease);
    const px = tx + sway[i] * Math.sin(Math.PI * e), py = fromY[i] + (ty - fromY[i]) * e;
    const a = Math.min(1, b[i] * 0.62 * Math.min(1, u * 4)); // fades in as it sets off
    paths[Math.min(9, Math.floor(a * 9.99))].rect(px - 0.6, py - 0.6, 1.2, 1.2);
  }
  cloudCtx.clearRect(0, 0, introW, introH);
  paths.forEach((path, j) => { cloudCtx.fillStyle = CLOUD_RAMP[j]; cloudCtx.fill(path); });
}

function renderIntroBackground(time) {
  introCtx.clearRect(0, 0, introW, introH);
  // The backdrop: the page's own colour (from the slide start the whole canvas dissolves, styles.css).
  introCtx.fillStyle = '#050707';
  introCtx.fillRect(0, 0, introW, introH);

  // Loading progress (no longer drawn): paced by the intro clock, held until the page and the cloud are ready; glides
  // toward its target (~.25s lag) so a late load never jumps it.
  const target = scanProgress(time);
  const dt = Math.max(0, time - scanLast);
  scanLast = time;
  scanP = reduceMotion ? target : Math.max(scanP, scanP + (target - scanP) * (1 - Math.exp(-dt / 250)));
  if (scanP > 0.995 && !scanDoneAt) scanDoneAt = time || 1;

  if (cloud && !cloudMoving && !reduceMotion && pageLoaded) {
    const handoffAt = (cloud.landedAt || cloud.at + cloud.flowEnd) + SETTLE_MS;
    if (time >= handoffAt - SLIDE_LEAD_MS) moveCloud();
  }
  // Hand off once the landed returns have settled (no cloud on narrow screens: once loaded).
  const ready = cloud ? cloud.landedAt && time - cloud.landedAt >= SETTLE_MS : scanDoneAt;
  if (!reduceMotion && ready) finishIntro();
  renderContours(time);
  renderCloud(time);
}

function introLoop(now) {
  renderIntroBackground(now - introStart);
  rafId = requestAnimationFrame(introLoop);
}

const axisTravel = (ms) => AXIS_SPEED * introH * ms / 1000;

function createAxisScale(el, orientation) {
  const horizontal = orientation === 'horizontal';
  const travel = axisTravel(AXIS_RUN_MS);
  // Ticks cover the viewport plus the full travel (and a step of slack each side),
  // so the scale reaches both screen edges for the whole move.
  const length = (horizontal ? introW : introH) + travel + 2 * MAJOR_STEP;
  const start = horizontal ? -(travel + MAJOR_STEP) : -MAJOR_STEP;
  const pos = horizontal ? 'left' : 'top';
  const step = MAJOR_STEP / (MINOR_PER_MAJOR + 1);

  const scale = document.createElement('div');
  scale.className = `scan__scale scan__scale--${orientation}`;
  scale.style[pos] = `${start}px`;
  scale.style[horizontal ? 'width' : 'height'] = `${length}px`;

  for (let i = 0; i * step <= length; i += 1) {
    const isMajor = i % (MINOR_PER_MAJOR + 1) === 0;
    const tick = document.createElement('span');
    tick.className = `scan__mark ${isMajor ? 'is-major' : 'is-minor'}`;
    tick.style[pos] = `${i * step}px`;
    scale.appendChild(tick);
  }

  el.replaceChildren(scale);
  return scale;
}

function setupAxes() {
  scales = {
    h: createAxisScale(scanHorizontal, 'horizontal'),
    v: createAxisScale(scanVertical, 'vertical'),
  };
}

function placeCrosshair(x, y) {
  axisState.intersectionX = x;
  axisState.intersectionY = y;
  introCrosshair.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
  introCrosshair.classList.add('is-placed'); // hidden until first placed, so it never flashes at 0,0
}

// Horizontal axis height (viewport fraction) at elapsed ms: linear climb until t, then the
// velocity decays exponentially into a slow crawl. Position and velocity stay continuous at t.
function horizontalYAt(ms) {
  const v = (HORIZONTAL_END_Y - HORIZONTAL_START_Y) / HORIZONTAL_CLIMB_MS;
  if (ms <= HORIZONTAL_CLIMB_MS) return HORIZONTAL_START_Y + v * ms;
  const d = ms - HORIZONTAL_CLIMB_MS;
  const crawl = v * HORIZONTAL_CRAWL;
  return HORIZONTAL_END_Y + crawl * d + (v - crawl) * HORIZONTAL_EASE_MS * (1 - Math.exp(-d / HORIZONTAL_EASE_MS));
}

function animateAxes(now) {
  if (!axisState.start) axisState.start = now - INTRO_OFFSET_MS;
  let axisTime = now - axisState.start;
  if (axisState.decelAt) {
    // During the float, axis time (scales, crawl, readout) eases to a stop: speed (1 - p)^2, so it
    // drops to ~30% by the time the lines start dissolving and reaches 0 as the float lands.
    const p = Math.min(1, (now - axisState.decelAt) / FLOAT_MS);
    axisTime = axisState.decelAt - axisState.start + (FLOAT_MS * (1 - (1 - p) ** 3)) / 3;
  }
  const elapsed = Math.min(AXIS_RUN_MS, axisTime);

  axisState.horizontalY = horizontalYAt(elapsed);
  const x = VERTICAL_X * introW;
  const y = axisState.horizontalY * introH;
  const travel = axisTravel(elapsed);

  // Lines stay full-bleed; only the scales move: horizontal right, vertical up, same speed.
  scanHorizontal.style.transform = `translate3d(0, ${y}px, 0)`;
  scanVertical.style.transform = `translate3d(${x}px, 0, 0)`;
  scales.h.style.transform = `translate3d(${travel}px, 0, 0)`;
  scales.v.style.transform = `translate3d(0, ${-travel}px, 0)`;
  placeCrosshair(x, y);
  // Readout drifts with the axes: latitude climbs with the horizontal axis and the vertical
  // scale, longitude with the horizontal scale. Mock coordinates around the hero's 24.79N 85.94E.
  readoutLat.textContent = `${(24.7862 + (HORIZONTAL_START_Y - axisState.horizontalY) * 0.6 + travel * 0.00002).toFixed(4)}° N`;
  readoutLon.textContent = `${(85.9412 + travel * 0.00004).toFixed(4)}° E`;

  if (elapsed < AXIS_RUN_MS) axisRaf = requestAnimationFrame(animateAxes);
}

const FLOAT_MS = 1800;
// The settled mountain slides from the centre to the hologram's place: quick off the mark, then a long soft landing.
// Keep the backdrop dissolve and the .hero .reveal delay in styles.css (both 1.6s) in step.
const SLIDE_MS = 1800;
// A gentle start (so the last rising points curve into the sideways motion) and a long soft glide into place.
const SLIDE_EASE = 'cubic-bezier(.42, 0, .2, 1)';
const CLOUD_OUT_MS = 1400; // after the slide the points fade out as the hologram fades in on them
const SLAB_LEAD_MS = 500; // the hologram's slabs start rising this long before the points finish sliding in

// The hand-off, as one continuous page. At the slide start the intro's backdrop begins to dissolve into the page's
// identical one and the hero assembles itself (styles.css: its backdrop art fades in, the heading floats in from the
// left, the other text floats up with the slabs); meanwhile the mountain slides to the hologram's place. On arrival
// the hologram fades in on the points as they fade out, and its slabs rise. No cloud (narrow screens): no slide.
// The slide starts SLIDE_LEAD_MS before the hand-off (the rise and settle carry on inside the moving layer: its canvas
// keeps being redrawn until the hand-off, while the compositor moves it).
const SLIDE_LEAD_MS = 1000;
const COLOUR_AHEAD_MS = 40;
function moveCloud() {
  cloudMoving = true;
  // The real colour comes in as the structure reaches its spot: triggered COLOUR_AHEAD_MS (about two frames) before the
  // slide ends, so its first drawn frame lands on the arrival, never after. The slabs start rising SLAB_LEAD_MS before
  // the arrival (the hologram starts rendering then too, so it is already drawing when the colour comes in).
  cloudLayer.animate({ transform: ['none', `translate3d(${-cloud.dx}px, ${-cloud.dy}px, 0)`] },
    { duration: SLIDE_MS, easing: SLIDE_EASE, fill: 'forwards' });
  window.setTimeout(formHologram, SLIDE_MS - COLOUR_AHEAD_MS);
  window.setTimeout(() => { hologram?.start(); hologram?.raise(); }, SLIDE_MS - SLAB_LEAD_MS);
}

function slideCloud() {
  cloudSlideAt = performance.now(); // also freezes the points' canvas
  heroTitle.style.visibility = 'visible';
  site.classList.add('is-visible');
  intro.classList.add('is-exiting');
  if (!cloud) formHologram(); // narrow screens: no slide
}

function formHologram() {
  cloudLayer.animate({ opacity: [1, 0] }, { duration: CLOUD_OUT_MS, easing: 'cubic-bezier(.45, 0, .55, 1)', fill: 'forwards' });
  hologramAppear = true;
  ensureHologram();
  window.setTimeout(fadeOutIntro, cloud ? CLOUD_OUT_MS : SLIDE_MS);
}

// The intro cross-fades to the main page; loops end once it's invisible.
function fadeOutIntro() {
  intro.classList.add('is-done');
  hologramAppear = true;
  hologram?.appear(); // the hologram floats up as the page cross-fades in
  window.setTimeout(() => {
    cancelAnimationFrame(axisRaf);
    cancelAnimationFrame(rafId);
  }, INTRO_FADE_MS);
}

function finishIntro(skip = false) {
  if (introFinished) return;
  introFinished = true;
  removeEventListener('pointerdown', skipIntro);
  removeEventListener('keydown', skipIntro);
  window.clearTimeout(introSequenceTimer);

  if (reduceMotion || skip) {
    fadeOutIntro();
    site.classList.add('is-visible');
    ensureHologram();
    heroTitle.style.visibility = 'visible';
    enableCursor();
    return;
  }
  slideCloud();
  window.setTimeout(enableCursor, cloud ? SLIDE_MS + CLOUD_OUT_MS : SLIDE_MS);
}

/* ─────────────────────────────────────────────
   Cursor: native-looking white arrow + teal square + canvas trail.
   ───────────────────────────────────────────── */
const cursor = {
  targetX: window.innerWidth / 2,
  targetY: window.innerHeight / 2,
  x: window.innerWidth / 2,
  y: window.innerHeight / 2,
  trail: [],
  maxTrail: 34,
  raf: 0,
  speed: 0,
  previousX: window.innerWidth / 2,
  previousY: window.innerHeight / 2,
  enabled: false,
};

function resizeTrailCanvas() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  trailCanvas.width = Math.floor(width * scale);
  trailCanvas.height = Math.floor(height * scale);
  trailCanvas.style.width = `${width}px`;
  trailCanvas.style.height = `${height}px`;
  trailCtx.setTransform(scale, 0, 0, scale, 0, 0);
}

function drawCursorTrail() {
  const ctx = trailCtx;
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

  if (cursor.trail.length < 2) return;

  const motion = Math.min(1, cursor.speed / 7);
  if (motion < 0.02) return;

  ctx.lineWidth = 1.4;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgb(89,217,192)';

  for (let i = 1; i < cursor.trail.length; i += 1) {
    const ratio = i / cursor.trail.length;
    const opacity = Math.min(1, Math.max(0, ratio * ratio * 0.72)) * motion;
    ctx.globalAlpha = opacity;
    ctx.beginPath();
    ctx.moveTo(cursor.trail[i - 1].x, cursor.trail[i - 1].y);
    ctx.lineTo(cursor.trail[i].x, cursor.trail[i].y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function cursorLoop() {
  const dx = cursor.targetX - cursor.x;
  const dy = cursor.targetY - cursor.y;
  cursor.x += dx * 0.18;
  cursor.y += dy * 0.18;

  const movedX = cursor.x - cursor.previousX;
  const movedY = cursor.y - cursor.previousY;
  const dist = Math.hypot(movedX, movedY);
  cursor.speed += (dist - cursor.speed) * 0.2;

  if (dist > 0.05) {
    cursor.trail.push({ x: cursor.x, y: cursor.y });
    if (cursor.trail.length > cursor.maxTrail) cursor.trail.shift();
  }

  cursor.previousX = cursor.x;
  cursor.previousY = cursor.y;

  cursorSquare.style.transform = `translate3d(${cursor.x}px, ${cursor.y}px, 0) translate(-50%, -50%)`;
  cursorArrow.style.transform = `translate3d(${cursor.targetX}px, ${cursor.targetY}px, 0)`;
  drawCursorTrail();

  if (dist < 0.05 && cursor.speed < 0.05) {
    cursor.trail = [{ x: cursor.x, y: cursor.y }];
  }

  cursor.raf = requestAnimationFrame(cursorLoop);
}

function enableCursor() {
  if (!finePointer || reduceMotion || cursor.enabled) return;
  cursor.enabled = true;
  document.body.classList.add('cursor-enabled');
  resizeTrailCanvas();
  cursorArrow.style.opacity = '1';
  cursorSquare.style.opacity = '1';
  document.addEventListener('mousemove', onMouseMove, { passive: true });
  window.addEventListener('resize', resizeTrailCanvas, { passive: true });
  cursor.raf = requestAnimationFrame(cursorLoop);
}

function onMouseMove(event) {
  cursor.targetX = event.clientX;
  cursor.targetY = event.clientY;
  cursorArrow.style.transform = `translate3d(${event.clientX}px, ${event.clientY}px, 0)`;
  cursorSquare.classList.toggle('is-hover', !!event.target.closest?.('button, a'));
}

function startIntro() {
  resizeIntroCanvas(); // also builds the axis scales
  introStart = performance.now();
  rafId = requestAnimationFrame(introLoop);

  if (reduceMotion) {
    // Park the axes at their end state in one frame; the complete, still cloud shows for 6s, then the page.
    axisState.start = introStart - AXIS_RUN_MS;
    animateAxes(introStart);
    window.setTimeout(finishIntro, 6000);
    return;
  }

  axisRaf = requestAnimationFrame(animateAxes);
}

// Any click or key skips the intro straight to the hero.
function skipIntro(event) {
  event.preventDefault(); // Space/arrow skip must not also scroll the page
  finishIntro(true);
}

// Hero dust: on hover, black dust is pulled toward the pointer and white dust gathers around the black,
// moving inside a box around the word (four walls) and seen only through the letters, over a grey haze. Each grain chases the
// pointer's past path, so a moving pointer leaves a comet tail. Crisp round grains, no glow.
const DUST_BLACK = 520;
const DUST_WHITE = 520;
const DUST_BOUNCE = 0.95; // restitution off the word box's walls
const DUST_KICK = 1.6; // minimum rebound speed (px per 60fps frame at 70px caps): even a slow touch bounces
function setupHeroDust() {
  if (!finePointer || reduceMotion) return; // the layer is display:none there (styles.css)
  const layer = document.querySelector('.hero__dust');
  const canvas = layer.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const letters = document.createElement('canvas'); // the glyphs: the windows the dust is seen through
  const lctx = letters.getContext('2d');
  const sprite = (rgb) => {
    const c = document.createElement('canvas');
    c.width = c.height = 32;
    const g = c.getContext('2d');
    const fill = g.createRadialGradient(16, 16, 0, 16, 16, 16);
    // A crisp disc with a short anti-aliased rim: each grain keeps a defined shape.
    fill.addColorStop(0, `rgba(${rgb},1)`);
    fill.addColorStop(0.7, `rgba(${rgb},1)`);
    fill.addColorStop(1, `rgba(${rgb},0)`);
    g.fillStyle = fill;
    g.fillRect(0, 0, 32, 32);
    return c;
  };
  const dotBlack = sprite('14,16,16');
  const dotWhite = sprite('255,255,255');
  const bell = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5; // -1..1, centre-heavy
  const black = Array.from({ length: DUST_BLACK }, () => ({ life: 0 }));
  const white = Array.from({ length: DUST_WHITE }, (_, i) => ({ life: 0, anchor: black[i % DUST_BLACK] }));
  // The dust's only walls: start of the E to the end of the last D, cap height to baseline.
  let bx0 = 0, bx1 = 0, by0 = 0, by1 = 0;
  let w = 0, h = 0, cap = 70, px = 0, py = 0, hx = 0, hy = 0, haze = 0, inside = false, raf = 0, last = 0;
  // The pointer's recent path (~1s). Grains chase where the pointer *was*, each with its own
  // delay, so the dust flows along the path the cursor took instead of cutting straight to it.
  let trail = [];
  let sx = 0, sy = 0;
  const sample = (t) => {
    let lo = 0, hi = trail.length - 1;
    if (t >= trail[hi].t) { sx = trail[hi].x; sy = trail[hi].y; return; }
    if (t <= trail[0].t) { sx = trail[0].x; sy = trail[0].y; return; }
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (trail[mid].t <= t) lo = mid; else hi = mid; }
    const a = trail[lo], b = trail[hi], u = (t - a.t) / (b.t - a.t);
    sx = a.x + (b.x - a.x) * u;
    sy = a.y + (b.y - a.y) * u;
  };

  // Black grains spawn around the pointer and are drawn in; white grains spawn around their black one.
  const spawnBlack = (p, delay) => {
    const angle = Math.random() * Math.PI * 2;
    const dist = cap * (0.15 + 0.35 * Math.random());
    Object.assign(p, {
      x: px + Math.cos(angle) * dist, y: py + Math.sin(angle) * dist, vx: 0, vy: 0,
      ox: bell() * cap * 0.3, oy: bell() * cap * 0.3, // where it settles around the pointer
      lag: 60 + 690 * Math.random() ** 1.6, // ms behind the pointer: most close, a long tail
      k: 0.04 + 0.05 * Math.random(),
      swirl: (Math.random() - 0.5) * 0.15,
      r: (1 + 2.2 * Math.random()) * cap / 70,
      age: -delay, life: 1.8 + 1.4 * Math.random(), alpha: 0.9,
    });
    bounce(p);
  };
  const spawnWhite = (p, delay) => {
    const a = p.anchor.life ? p.anchor : { x: px, y: py };
    Object.assign(p, {
      x: a.x + bell() * cap * 0.35, y: a.y + bell() * cap * 0.35, vx: 0, vy: 0,
      ox: bell() * cap * 0.15, oy: bell() * cap * 0.15,
      push: cap * (0.12 + 0.2 * Math.random()), // settles on the outer side of its black grain
      k: 0.02 + 0.03 * Math.random(),
      swirl: (Math.random() - 0.5) * 0.2,
      r: (0.8 + 1.6 * Math.random()) * cap / 70,
      age: -delay, life: 2 + 1.4 * Math.random(), alpha: 0.75,
    });
    bounce(p);
  };

  const step = (p, tx, ty, f, spawn) => {
    if (!p.life) return false;
    p.age += f / 60;
    if (p.age >= p.life) {
      if (inside) spawn(p, 0);
      else { p.life = 0; return false; }
    }
    // Rest points past a wall are folded back inside (mirrored off a margin), so no grain ever
    // settles against a wall: it only reaches one with momentum, and bounces off.
    const m = cap * 0.1;
    const dx = fold(tx + p.ox, bx0 + m, bx1 - m) - p.x;
    const dy = fold(ty + p.oy, by0 + m, by1 - m) - p.y;
    const drag = Math.pow(0.9, f);
    // Spring toward the target, a slight sideways swirl, and a little jitter so the dust stays alive.
    p.vx = (p.vx + (dx * p.k - dy * p.k * p.swirl + (Math.random() - 0.5) * 0.06) * f) * drag;
    p.vy = (p.vy + (dy * p.k + dx * p.k * p.swirl + (Math.random() - 0.5) * 0.06) * f) * drag;
    p.x += p.vx * f;
    p.y += p.vy * f;
    bounce(p);
    return true;
  };
  const fold = (v, lo, hi) => (v < lo ? Math.min(hi, 2 * lo - v) : v > hi ? Math.max(lo, 2 * hi - v) : v);
  // Grains stay inside the word box and bounce off its walls.
  const bounce = (p) => {
    const kick = DUST_KICK * cap / 70;
    if (p.x < bx0 + p.r) { p.x = bx0 + p.r; p.vx = Math.max(Math.abs(p.vx) * DUST_BOUNCE, kick); }
    else if (p.x > bx1 - p.r) { p.x = bx1 - p.r; p.vx = -Math.max(Math.abs(p.vx) * DUST_BOUNCE, kick); }
    if (p.y < by0 + p.r) { p.y = by0 + p.r; p.vy = Math.max(Math.abs(p.vy) * DUST_BOUNCE, kick); }
    else if (p.y > by1 - p.r) { p.y = by1 - p.r; p.vy = -Math.max(Math.abs(p.vy) * DUST_BOUNCE, kick); }
  };
  const draw = (p, dot) => {
    if (!p.life || p.age < 0) return;
    ctx.globalAlpha = p.alpha * Math.sin(Math.PI * p.age / p.life) ** 0.4; // long plateau, soft ends
    ctx.drawImage(dot, p.x - p.r, p.y - p.r, p.r * 2, p.r * 2);
  };

  const tick = (now) => {
    const f = Math.min(Math.max(now - last, 0), 50) / 16.7; // 1 = one 60fps frame
    last = now;
    let alive = false;
    trail.push({ t: now, x: px, y: py });
    while (trail.length > 2 && trail[0].t < now - 1000) trail.shift();
    for (const p of black) {
      sample(now - p.lag);
      p.tx = sx;
      p.ty = sy;
      alive = step(p, sx, sy, f, spawnBlack) || alive;
    }
    for (const p of white) {
      // Its black grain, pushed outward (away from that grain's own target), so at rest the white
      // rings the black core instead of covering it; in motion it trails the black.
      const a = p.anchor;
      const ax = a.x ?? px, ay = a.y ?? py;
      const dx = ax - (a.tx ?? px), dy = ay - (a.ty ?? py);
      const len = Math.hypot(dx, dy) || 1;
      alive = step(p, ax + dx / len * p.push, ay + dy / len * p.push, f, spawnWhite) || alive;
    }
    // A faint grey haze under the dust turns the grains into a gradient (and lets the white show).
    sample(now - 180);
    hx += (sx - hx) * 0.12 * f;
    hy += (sy - hy) * 0.12 * f;
    haze += ((inside ? 1 : 0) - haze) * 0.06 * f;

    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, w, h);
    const glow = ctx.createRadialGradient(hx, hy, 0, hx, hy, cap * 1.1);
    glow.addColorStop(0, `rgba(40,44,43,${0.55 * haze})`);
    glow.addColorStop(1, 'rgba(40,44,43,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);
    for (const p of black) draw(p, dotBlack);
    for (const p of white) draw(p, dotWhite);
    // The letters are windows: grains move anywhere inside the four walls (across the gaps too) but
    // are only seen through the white letters; the black background hides them.
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'destination-in';
    ctx.drawImage(letters, 0, 0, w, h);

    if (alive || haze > 0.01) raf = requestAnimationFrame(tick);
    else { raf = 0; ctx.clearRect(0, 0, w, h); }
  };

  // Redraw the letter clip from the live layout: each glyph at its laid-out position, on the
  // baseline (the bottom of the logo D box), so it matches the HTML text exactly.
  const fit = () => {
    const box = layer.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = box.width;
    h = box.height;
    for (const c of [canvas, letters]) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const style = getComputedStyle(layer);
    cap = parseFloat(style.fontSize) * 0.71;
    lctx.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
    lctx.fontKerning = 'none';
    const logo = layer.querySelector('.logo-d').getBoundingClientRect();
    const baseline = logo.bottom - box.top;
    const text = layer.childNodes[1];
    const range = document.createRange();
    bx1 = 0;
    by0 = baseline - cap;
    by1 = baseline;
    for (let i = 0; i < text.length; i++) {
      range.setStart(text, i);
      range.setEnd(text, i + 1);
      const x = range.getBoundingClientRect().left - box.left;
      lctx.fillText(text.data[i], x, baseline);
      const ink = lctx.measureText(text.data[i]);
      if (i === 0) bx0 = x - ink.actualBoundingBoxLeft; // left ink edge of the E
      bx1 = Math.max(bx1, x + ink.actualBoundingBoxRight);
    }
  };
  const aim = (e) => {
    const box = layer.getBoundingClientRect();
    px = e.clientX - box.left;
    py = e.clientY - box.top;
  };

  layer.addEventListener('pointerenter', (e) => {
    aim(e);
    if (!raf) { fit(); hx = px; hy = py; trail = [{ t: performance.now(), x: px, y: py }]; }
    inside = true;
    // Grains still alive keep going; dead ones stream in over the first ~0.6s.
    black.forEach((p) => p.life || spawnBlack(p, Math.random() * 0.6));
    white.forEach((p) => p.life || spawnWhite(p, 0.15 + Math.random() * 0.6));
    if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); }
  });
  layer.addEventListener('pointermove', aim, { passive: true });
  layer.addEventListener('pointerleave', () => { inside = false; });
}
setupHeroDust();

// Hero hologram (Three.js, ~500KB): loaded as its own chunk so the intro paints without it,
// skipped while hidden (narrow screens), and started once the hero is revealed.
const holo = document.querySelector('.hero__holo');
let hologram = null;
let hologramLoading = false;
let hologramAppear = false; // set when the page is revealed; applied once the module has loaded
function ensureHologram() {
  if (hologram) {
    if (site.classList.contains('is-visible')) hologram.start();
    if (hologramAppear) hologram.appear();
    return;
  }
  if (hologramLoading || !holo || getComputedStyle(holo).display === 'none') return;
  hologramLoading = true;
  import('./hologram.js').then(({ initHologram }) => {
    hologram = initHologram(holo);
    buildCloud(); // the intro's scan cloud
    ensureHologram();
  });
}
ensureHologram();
window.addEventListener('resize', ensureHologram, { passive: true });

// The ridgeline band fades right down behind the hologram (mountain and slabs), so its lines don't show through them:
// an elliptical hole in its mask on the hologram's footprint (styles.css .hero__ridges).
function fadeRidgesUnderHologram() {
  const band = document.querySelector('.hero__ridges'), h = holo?.getBoundingClientRect();
  if (!band || !h?.width) return; // no hologram (narrow screens): the hole stays off-screen
  const b = band.getBoundingClientRect();
  band.style.setProperty('--hx', `${h.left + h.width * 0.5 - b.left}px`);
  band.style.setProperty('--hy', `${h.top + h.height * 0.52 - b.top}px`);
  band.style.setProperty('--rx', `${h.width * 0.6}px`);
  band.style.setProperty('--ry', `${h.height * 0.52}px`);
}
fadeRidgesUnderHologram();
window.addEventListener('resize', fadeRidgesUnderHologram, { passive: true });

// Scroll jigsaw, after assembly: as the black tail scrolls in, --pz-release goes 0 -> 1 and the grey pieces slide up
// with it at scroll speed (CSS). The teal piece, pinned at the screen centre until then, drifts down the screen
// (ease-out) and lands on the Connaught Place stage in the tail, then rises with it. Over the last part of the drop the
// RGB view and the 3D model float up either side of it and the waves grow from the equator (.is-shown; setupDemoStage).
// Reduced motion: no drop; both sides simply follow the puzzle (CSS).
const PUZZLE_LAND_Y = 0.72;   // where on screen (fraction of the viewport height) the teal piece lands
const PUZZLE_SHRINK = 0.31;   // the teal piece eases to 69% of its size over the drop (smootherstep: subtle)
const CP_RISE_AT = 0.9;       // the Connaught Place sides start floating up once the teal piece is this far into its drop (about to settle)
const PUZZLE_CATCH = 0.7;     // from here on the piece is handed over to the rising stage: its path blends into the stage's (smoothstep), so it
                              // meets it moving with it; it used to stop dead at the landing line, then jump to scroll speed (owner 2026-10-25)
function setupPuzzleRelease() {
  const puzzle = document.querySelector('.puzzle'), tail = puzzle?.querySelector('.puzzle__tail');
  const engine = puzzle?.querySelector('.puzzle__piece--engine'), stage = puzzle?.querySelector('.cp-stage');   // the layer: never transformed
  if (!tail || !engine || !stage) return;
  const flip = setupDemoStage(stage);
  if (reduceMotion) { flip.show(true); return; }
  // once landed, a copy of the board inside the stage takes over and moves with it natively: the piece itself, placed by
  // JS a frame after each scroll, would jitter against the stage while they ride together (owner 2026-10-17)
  const twin = engine.querySelector('.puzzle__board').cloneNode(true);
  twin.classList.add('cp-piece'); twin.setAttribute('aria-hidden', 'true'); stage.append(twin);
  let queued = false;
  const update = () => {
    queued = false;
    const vh = window.innerHeight, t = tail.getBoundingClientRect(), d = vh - t.top;      // d: how far the tail has come in
    puzzle.style.setProperty('--pz-release', Math.min(1, Math.max(0, d / vh)).toFixed(4));
    const e = engine.getBoundingClientRect(), base = e.top + e.height / 2;                // the piece's centre without the drop
    const sr = stage.getBoundingClientRect(), stageMid = sr.top + sr.height / 2, landY = PUZZLE_LAND_Y * vh, start = vh / 2;
    let target = base, k = d > 0 ? 1 : 0;
    if (d > 0 && stageMid > landY) {                                                       // drifting down towards the stage
      const dLand = vh + (stageMid - t.top) - landY;                                       // scroll at which the stage reaches landY
      k = Math.min(1, d / dLand);
      const drop = start + (landY - start) * (1 - (1 - k) ** 3), c = Math.max(0, (k - PUZZLE_CATCH) / (1 - PUZZLE_CATCH));
      target = drop + (stageMid - drop) * c * c * (3 - 2 * c);
    } else if (d > 0) target = stageMid;                                                   // landed: ride with the stage
    puzzle.classList.toggle('is-landed', d > 0 && stageMid <= landY);
    puzzle.style.setProperty('--pz-scale', (1 - PUZZLE_SHRINK * k * k * k * (k * (6 * k - 15) + 10)).toFixed(4));
    puzzle.style.setProperty('--pz-drop', (target - base).toFixed(1) + 'px');
    if (k >= CP_RISE_AT) flip.show(true); else if (k < CP_RISE_AT / 2) flip.show(false);  // timed CSS rise (not scroll-linked: smooth); hysteresis
  };
  const queue = () => { if (!queued) { queued = true; requestAnimationFrame(update); } };
  addEventListener('scroll', queue, { passive: true });
  addEventListener('resize', queue, { passive: true });
  update();
}

// Connaught Place sides: shown when the stage pins; the 3D city model loads (three.js, own chunk) when the stage comes
// near, and the RGB side gets the depth magnifier.
function setupDemoStage(stage) {
  let shown = false;
  const city = stage.querySelector('.cp-3d');
  new IntersectionObserver(([entry], obs) => {
    if (!entry.isIntersecting) return;
    obs.disconnect();
    import('./demo-city.js').then(({ initDemoCity }) => initDemoCity(city));
  }, { rootMargin: '200% 0px' }).observe(stage);   // two screens ahead: its build and first draw run while the jigsaw is still assembling
  setupDepthLens(stage.querySelector('.cp-rgb'));
  return { show(on) { if (on !== shown) { shown = on; stage.classList.toggle('is-shown', on); } } };
}

// Depth lens: over the RGB image the pointer becomes a round window showing the black-and-white depth map of the same
// spot, at the same scale (the depth map is the same area as the RGB image). Mouse and pen: on hover;
// touch: while the finger is down. The site's own cursor hides while the lens is up.
const DEPTH_LENS_ZOOM = 1;
function setupDepthLens(view) {
  const lens = view?.querySelector('.cp-lens');
  if (!lens) return;
  const place = (e) => {
    const r = view.getBoundingClientRect(), d = lens.offsetWidth, x = e.clientX - r.left, y = e.clientY - r.top;
    lens.style.transform = `translate(${x - d / 2}px, ${y - d / 2}px)`;
    lens.style.backgroundSize = `${r.width * DEPTH_LENS_ZOOM}px ${r.height * DEPTH_LENS_ZOOM}px`;
    lens.style.backgroundPosition = `${d / 2 - x * DEPTH_LENS_ZOOM}px ${d / 2 - y * DEPTH_LENS_ZOOM}px`;
  };
  const on = (e) => { place(e); view.classList.add('is-lens'); document.body.classList.add('is-lens'); };
  const off = () => { view.classList.remove('is-lens'); document.body.classList.remove('is-lens'); };
  view.addEventListener('pointerenter', (e) => { if (e.pointerType !== 'touch') on(e); });
  view.addEventListener('pointerdown', (e) => { if (e.pointerType === 'touch') on(e); });
  view.addEventListener('pointermove', (e) => { if (view.classList.contains('is-lens')) place(e); });
  for (const type of ['pointerleave', 'pointerup', 'pointercancel']) view.addEventListener(type, (e) => { if (type === 'pointerleave' || e.pointerType === 'touch') off(); });
}

setupPuzzleRelease();

// Landmarks row in 3D (src/landmarks3d.js, its own chunk): loaded when the row comes within a screen of view (content
// script: 3D loads only near the screen). Reduced motion: the cells stay as their still images.
const landmarkRow = document.querySelector('.cells');
if (landmarkRow && !reduceMotion) new IntersectionObserver(([entry], obs) => {
  if (!entry.isIntersecting) return;
  obs.disconnect();
  import('./landmarks3d.js').then(({ initLandmarks3d }) => initLandmarks3d(landmarkRow));
}, { rootMargin: '100% 0px' }).observe(landmarkRow);

// Opening statement (owner 2026-10-16): animated like the intro line on heronaiapp.com (read from its GSAP calls,
// 2026-10-16), without GSAP. Split into words and characters.
// - First appearance (screens over 767px, once): when the block's top + 45% of its height reaches the bottom of the screen,
//   the characters rise from one character height below while fading in, 0.6s power2.out, 4ms apart (CSS, .is-in).
// - Scroll (both ways): a colour sweep in reading order, each character from faint to its colour (--k 0 -> 1) over 0.5 of the
//   timeline, 0.03 apart, power1.out, from when the block's centre is 10% below the screen to when it is 40% down from the top.
// - While the mouse is over a line, that line turns teal (one line at a time; --hl eases in CSS).
const SWEEP_STAGGER = 0.03, SWEEP_EACH = 0.5;
function setupStatement() {
  const text = document.querySelector('.statement__text');
  if (!text) return;
  const chars = splitChars(text), words = [...text.querySelectorAll('.w')];
  const total = (chars.length - 1) * SWEEP_STAGGER + SWEEP_EACH;
  if (reduceMotion) text.classList.add('is-in');
  let lines = [], lit = -1, queued = false;
  const lineOf = new Map(), ks = new Float32Array(chars.length).fill(-1);
  const measure = () => {
    lines = []; let top = null;
    for (const w of words) { if (w.offsetTop !== top) { top = w.offsetTop; lines.push([]); } lines.at(-1).push(w); lineOf.set(w, lines.length - 1); }
  };
  const update = () => {
    queued = false;
    const vh = innerHeight, r = text.getBoundingClientRect();
    // the rise replays every time it comes back into view (owner, like the titles): in at 45% visible, reset once fully off
    if (!reduceMotion) {
      if (Math.min(r.bottom, vh) - Math.max(r.top, 0) >= r.height * 0.45) text.classList.add('is-in');
      else if (r.bottom < 0 || r.top > vh) text.classList.remove('is-in');
    }
    const t = Math.min(1, Math.max(0, (vh * 1.1 - (r.top + r.height / 2)) / (vh * 0.7))) * total;
    chars.forEach((c, i) => {
      const u = Math.min(1, Math.max(0, (t - i * SWEEP_STAGGER) / SWEEP_EACH)), k = Math.round((1 - (1 - u) ** 2) * 1000) / 1000;
      if (k !== ks[i]) { ks[i] = k; c.style.setProperty('--k', k); }
    });
  };
  const light = (i) => {
    if (i === lit) return;
    lines[lit]?.forEach((w) => w.classList.remove('is-lit'));
    lines[i]?.forEach((w) => w.classList.add('is-lit'));
    lit = i;
  };
  const queue = () => { if (!queued) { queued = true; requestAnimationFrame(update); } };
  measure(); update();
  addEventListener('scroll', queue, { passive: true });
  addEventListener('resize', () => { measure(); queue(); }, { passive: true });
  text.addEventListener('pointermove', (e) => {
    const w = e.pointerType === 'mouse' && e.target.closest('.w');
    if (w) light(lineOf.get(w));   // between words: keep the line
  });
  text.addEventListener('pointerleave', () => light(-1));
  document.fonts?.ready.then(() => { measure(); queue(); });   // the web font can rewrap the lines
}
setupStatement();
setupSections();   // titles, fade-ups, count-ups, charts (src/sections.js)






// Hero corner coordinates, live: the mouse across the screen spans 10 deg (whole degrees roll too; always 2 digits) around 24.7862 N 85.9412 E (north up, east
// right). Each digit is a 0-9 strip in a one-line window, rolled by a CSS transition (odometer).
const odo = [...document.querySelectorAll('[data-odo]')].map((el) => {
  const digits = [];
  el.replaceChildren(...[...el.textContent].map((ch) => {
    if (!/\d/.test(ch)) return ch;
    const win = document.createElement('span'), strip = document.createElement('span');
    win.className = 'odo'; strip.className = 'odo__strip'; strip.textContent = '0123456789'; strip.style.setProperty('--d', ch);
    win.append(strip); digits.push(strip); return win;
  }));
  el.setAttribute('aria-hidden', 'true');
  return { el, digits };
});
const odoLabel = document.querySelector('.hero__meta');
let odoRaf = 0, odoX = 0.5, odoY = 0.5;
const odoSet = ({ digits }, text) => { let i = 0; for (const ch of text) if (/\d/.test(ch)) digits[i++].style.setProperty('--d', ch); };
const odoMove = (e) => {
  odoX = e.clientX / innerWidth; odoY = e.clientY / innerHeight;
  odoRaf ||= requestAnimationFrame(() => {
    odoRaf = 0;
    const lat = (24.7862 + (0.5 - odoY) * 10).toFixed(4), lon = (85.9412 + (odoX - 0.5) * 10).toFixed(4);
    odoSet(odo[0], lat); odoSet(odo[1], lon);
    odoLabel.setAttribute('aria-label', `${lat}° N, ${lon}° E`);
  });
};
addEventListener('pointermove', odoMove, { passive: true });
addEventListener('touchmove', (e) => odoMove(e.touches[0]), { passive: true });   // iOS: a finger drag scrolls, which cancels pointer events

window.addEventListener('resize', resizeIntroCanvas, { passive: true });
addEventListener('pointerdown', skipIntro);
addEventListener('keydown', skipIntro);
startIntro();
if (deepLink) {
  finishIntro(true);
  requestAnimationFrame(() => deepLink.scrollIntoView({ block: 'start', behavior: 'instant' }));
}

// ENTER STUDIO opens the Studio.
document.querySelectorAll('button.hero__cta').forEach((button) => button.addEventListener('click', () => { location.href = '/#/studio'; }));

// Layout edit mode for the owner (dev server only, open the page with ?edit): src/edit-mode.js; never in the build.
if (import.meta.env.DEV && new URLSearchParams(location.search).has('edit')) import('./edit-mode.js');
