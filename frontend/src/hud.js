// Input-page HUD (2026-10-02): the live coordinate readout (top left), the full-height latitude
// ruler (right edge) and the WGS84 footnote under the Scene Input box. The static markup is in
// index.html (.hud); this file drives it.
//
// What the readout shows, in priority order:
//   magnify  the pointer is over the selected tile's preview (the magnifier): the coordinates under it;
//            the ruler rescales to 0.01 deg per major tick
//   tile     nothing is selected and the pointer is over a library card: that tile's centre
//   screen   anywhere else: the coordinates under the pointer on a virtual map centred on the selected
//            tile (or the default centre), 0.1 deg per major tick
//   idle     the pointer has not moved for IDLE_MS: every 6-10 s a new plausible value (Indian
//            subcontinent) with a short cross-fade, and every 12-20 s the ruler glides 40-120 px
//            (slow, then fast, then slow) and may change its scale. Never continuous.
// prefers-reduced-motion: no glides, fades or idle changes; values change instantly on pointer input.

const DEFAULT_CENTRE = { lat: 24.7863, lon: 85.9412 };
const MAJOR_PX = 104;               // pitch of a major tick at the default scale (claude_ui: 0.1 deg per 104 px)
const IDLE_MS = 5000;
const INDIA = { lat: [8.5, 34], lon: [69, 96] };
// Ruler motion: the view (centre latitude, scale) follows its target through three cascaded first-order lags.
// That gives a speed that ramps up smoothly from rest, peaks about 2*TAU in, and eases off in a long visible
// deceleration; because only the first lag's target changes, a new target in mid-glide bends the motion smoothly
// (speed and acceleration stay continuous), so the ruler never stops dead. Peak speed = 0.27 x distance / TAU.
const TAU_MS = 1500;                // idle glides: slow
const USER_TAU_MS = 800;            // changes the user causes (selecting, hovering, waking): brisker
const USER_LIMIT_MS = 3000;         // ...and always settled within 3 s of the click: bigger moves go faster
const SWEEP_MS = 1300;              // idle readout change: digits sweep through values like a fast pointer move
const SCALES = [0.05, 0.1, 0.2, 0.5];
// decorative footnote values (name, EPSG code); they are not read from any real data
const FOOT_CRS = [
    ["WGS84", 4326],
    ["WGS84 / UTM 43N", 32643],
    ["WGS84 / UTM 44N", 32644],
    ["WGS84 / UTM 45N", 32645],
    ["WGS84 / UTM 46N", 32646],
    ["NAD83", 4269],
    ["ETRS89", 4258],
    ["GDA94", 4283],
    ["Pseudo-Mercator", 3857],
    ["WGS84 / UTM 32N", 32632],
]; 


export function formatCoord(lat, lon) {
    return `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? "N" : "S"} · ${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? "E" : "W"}`;
}

export function screenCoord(centre, x, y, w, h, degPerPx) {
    const lat = centre.lat + (h / 2 - y) * degPerPx;
    const lon = centre.lon + (x - w / 2) * degPerPx / Math.max(Math.cos(centre.lat * Math.PI / 180), 0.2);
    return { lat, lon };
}

// One step of three cascaded lags toward `target`: st = [y1, y2, y3] (y3 is the output).
export function lagCascade(st, target, dtMs, tauMs = TAU_MS) {
    const k = 1 - Math.exp(-dtMs / tauMs);
    st[0] += (target - st[0]) * k;
    st[1] += (st[0] - st[1]) * k;
    st[2] += (st[1] - st[2]) * k;
    return st[2];
}

// How many time constants the 3-lag cascade needs before its remaining error is below eps (fraction of the
// distance): 1 - S(x) = e^-x (1 + x + x^2/2) <= eps, found by bisection.
export function settleFactor(eps) {
    const e = Math.min(Math.max(eps, 1e-12), 0.5);
    let lo = 0;
    let hi = 60;
    for (let i = 0; i < 60; i++) {
        const x = (lo + hi) / 2;
        const rest = Math.exp(-x) * (1 + x + x * x / 2);
        if (rest > e) lo = x; else hi = x;
    }
    return hi;
}

// The time constant to use so a move of the given size is settled within limitMs (never slower than tauMs).
export function tauWithin(limitMs, tauMs, eps) {
    return Math.max(Math.min(tauMs, limitMs / settleFactor(eps)), 120);
}

export function smoothstep(x, a, b) {
    const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
    return t * t * (3 - 2 * t);
}

export function easeOutCubic(t) {
    const c = Math.min(Math.max(t, 0), 1);
    return 1 - (1 - c) ** 3;
}

const rand = (a, b) => a + Math.random() * (b - a);

const impl = {
    hoverTile() {},
    magnify() {},
    setAnchor() {},
};

// public API (safe to call before init and when the HUD is absent)
export const hud = {
    hoverTile: (geo) => impl.hoverTile(geo),
    magnify: (coord) => impl.magnify(coord),
    setAnchor: (geo, selected) => impl.setAnchor(geo, selected),
};

export function initHud(root) {
    const readout = root.querySelector(".hud-readout");
    const latEl = root.querySelector(".hud-lat");
    const lonEl = root.querySelector(".hud-lon");
    const canvas = root.querySelector(".hud-ruler");
    if (!readout || !canvas) {
        return;
    }
    const ctx = canvas.getContext("2d");
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

    // ---- state
    let anchor = { ...DEFAULT_CENTRE };      // centre of the virtual map (selected tile, else default)
    let tileHover = null;                    // { lat, lon } of the hovered card (nothing selected)
    let magnifyCoord = null;                 // { lat, lon } under the magnifier
    let hasSelection = false;
    let pointer = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    let lastMove = performance.now();
    let idle = false;
    let idleRead = null;                     // { lat, lon } shown while idle

    // ruler view: centre latitude and scale (degrees per pixel), tweened
    const view = { cLat: anchor.lat, dpp: 0.1 / MAJOR_PX };
    const motion = { lat: [anchor.lat, anchor.lat, anchor.lat], logDpp: [0, 0, 0].map(() => Math.log(view.dpp)) };
    let target = { cLat: view.cLat, dpp: view.dpp };
    let moving = false;
    let lastFrame = 0;
    let motionTau = TAU_MS;


    function visible() {
        return root.offsetParent !== null;
    }

    function currentMode() {
        if (idle) return "idle";
        if (magnifyCoord) return "magnify";
        if (tileHover && !hasSelection) return "tile";
        return "screen";
    }

    // With a georeferenced tile selected, the ruler and the pointer's coordinates use that tile's own scale:
    // its latitude extent fills about 60% of the window height.
    let tileExtentDeg = null;
    function liveScale() {
        const h = canvas.clientHeight || window.innerHeight;
        return hasSelection && tileExtentDeg ? tileExtentDeg / (0.6 * h) : 0.1 / MAJOR_PX;
    }

    function currentRead() {
        switch (currentMode()) {
        case "idle": return idleRead;
        case "magnify": return magnifyCoord;
        case "tile": return tileHover;
        default: return screenCoord(anchor, pointer.x, pointer.y, window.innerWidth, window.innerHeight, liveScale());
        }
    }

    function targetView() {
        switch (currentMode()) {
        case "magnify": return { cLat: anchor.lat, dpp: liveScale() };
        case "tile": return { cLat: tileHover.lat, dpp: 0.1 / MAJOR_PX };
        case "idle": return null;               // idle drives the view itself
        default: return { cLat: anchor.lat, dpp: liveScale() };
        }
    }

    // ---- ruler drawing
    let raf = 0;
    function requestDraw() {
        if (!raf) {
            raf = requestAnimationFrame(frame);
        }
    }

    function frame(now) {
        raf = 0;
        if (moving) {
            const dt = lastFrame ? Math.min(now - lastFrame, 50) : 16;
            lastFrame = now;
            view.cLat = lagCascade(motion.lat, target.cLat, dt, motionTau);
            view.dpp = Math.exp(lagCascade(motion.logDpp, Math.log(target.dpp), dt, motionTau));
            const dpp = view.dpp;
            const settled = motion.lat.every(v => Math.abs(v - target.cLat) / dpp < 0.02)
                && motion.logDpp.every(v => Math.abs(v - Math.log(target.dpp)) < 2e-4);
            if (settled) {
                view.cLat = target.cLat;
                view.dpp = target.dpp;
                motion.lat.fill(target.cLat);
                motion.logDpp.fill(Math.log(target.dpp));
                moving = false;
                lastFrame = 0;
            } else {
                requestDraw();
            }
        }
        draw();
    }

    // Move the ruler to a new view. Safe to call mid-glide: the motion bends smoothly towards the new target.
    function glideTo(to, tau = USER_TAU_MS, limitMs = tau === USER_TAU_MS ? USER_LIMIT_MS : Infinity) {
        // speed up a long move so it settles within the limit (a short one keeps its natural pace)
        if (Number.isFinite(limitMs)) {
            const dpp = Math.min(view.dpp, to.dpp);
            const px = Math.max(Math.abs(to.cLat - view.cLat) / dpp, 0.02);
            const dLog = Math.max(Math.abs(Math.log(to.dpp) - Math.log(view.dpp)), 2e-4);
            const eps = Math.min(0.02 / px, 2e-4 / dLog, 0.5);
            tau = tauWithin(limitMs * 0.95, tau, eps);
        }
        target = { ...to };
        motionTau = tau;
        if (reduced || !visible()) {
            view.cLat = to.cLat;
            view.dpp = to.dpp;
            motion.lat.fill(to.cLat);
            motion.logDpp.fill(Math.log(to.dpp));
            moving = false;
            requestDraw();
            return;
        }
        moving = true;
        requestDraw();
    }

    function draw() {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (!w || !h) {
            return;
        }
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
            canvas.width = Math.round(w * dpr);
            canvas.height = Math.round(h * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        // theme colours ("r, g, b") from CSS: mint/grey-green in dark mode, orange/brown in bright mode
        const css = getComputedStyle(root);
        const C = name => css.getPropertyValue(name).trim() || "143, 179, 162";
        const lineRgb = C("--hud-line-rgb");
        const labelRgb = C("--hud-label-rgb");
        const markRgb = C("--hud-mark-rgb");

        const axisX = w - 31;
        const latAt = y => view.cLat + (h / 2 - y) * view.dpp;
        const yAt = lat => h / 2 + (view.cLat - lat) / view.dpp;
        // sub-pixel positions while moving (smooth), pixel-snapped when still (crisp)
        const place = y => (moving ? y : Math.round(y) + 0.5);
        const top = latAt(0);
        const bottom = latAt(h);

        // axis, end to end
        ctx.strokeStyle = `rgba(${lineRgb}, 0.55)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(axisX + 0.5, 0);
        ctx.lineTo(axisX + 0.5, h);
        ctx.stroke();

        // Decade levels (..., 0.01, 0.1, 1, ... degrees). Each level fades in as its pitch grows, so changing the
        // scale never pops ticks or labels in and out: ticks and labels cross-fade continuously.
        const nMax = Math.ceil(Math.log10(Math.max(h * view.dpp, 1e-9)));
        const nMin = Math.floor(Math.log10(view.dpp * 7));
        const labels = new Map();
        for (let n = nMax; n >= nMin; n--) {
            const stepDeg = 10 ** n;
            const pitch = stepDeg / view.dpp;
            const tickFade = smoothstep(pitch, 7, 30);
            const len = 4 + 10 * smoothstep(pitch, 20, 100);
            const baseAlpha = (0.3 + 0.55 * smoothstep(pitch, 30, 90)) * tickFade;
            const labelAlpha = smoothstep(pitch, 56, 96) * (1 - smoothstep(pitch, 520, 1000));
            const decimals = Math.max(1, -n);
            const lo = Math.ceil(bottom / stepDeg - 1e-9);
            const hi = Math.floor(top / stepDeg + 1e-9);
            for (let i = lo; i <= hi; i++) {
                const lat = i * stepDeg;
                const key = Math.round(lat * 1e6);
                if (labelAlpha > 0.01) {
                    const prev = labels.get(key);
                    if (!prev || prev.alpha < labelAlpha) {
                        labels.set(key, { lat, alpha: labelAlpha, decimals });
                    }
                }
                if (n < nMax && i % 10 === 0) {
                    continue;                                   // drawn by the next coarser level
                }
                const half = Math.abs(i) % 10 === 5;
                const y = place(yAt(lat));
                ctx.strokeStyle = `rgba(${lineRgb}, ${baseAlpha * (half ? 0.85 : 1)})`;
                ctx.beginPath();
                ctx.moveTo(axisX - len * (half ? 1.5 : 1), y);
                ctx.lineTo(axisX, y);
                ctx.stroke();
            }
        }
        ctx.font = '400 11.5px "IBM Plex Mono", ui-monospace, monospace';
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        for (const { lat, alpha, decimals } of labels.values()) {
            ctx.fillStyle = `rgba(${labelRgb}, ${alpha})`;
            ctx.fillText(`${lat.toFixed(decimals)}\u00B0 ${lat >= 0 ? "N" : "S"}`, axisX - 22, place(yAt(lat)));
        }

        // marker at the readout's latitude
        const read = currentRead();
        if (read) {
            const rawY = yAt(read.lat);
            const y = Math.min(Math.max(rawY, 8), h - 8);      // off-screen while the view is still gliding: parked at the edge, dimmed
            {
                const a = rawY === y ? 1 : 0.4;
                ctx.strokeStyle = `rgba(${markRgb}, ${0.9 * a})`;
                ctx.fillStyle = `rgba(${markRgb}, ${0.95 * a})`;
                ctx.beginPath();
                ctx.moveTo(axisX + 1, y);
                ctx.lineTo(axisX + 13, y);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(axisX + 13, y - 3.5);
                ctx.lineTo(axisX + 13, y + 3.5);
                ctx.lineTo(axisX + 7, y);
                ctx.closePath();
                ctx.fill();
            }
        }
    }

    // ---- readout text: two lines (latitude over longitude). A discrete change (hover, selection, idle value)
    // rolls the digits one at a time from the right-hand end (the decimals) to the left, the latitude first and
    // then the longitude; live pointer movement just updates the text.
    const DIGIT_MS = 55;      // gap between neighbouring digits
    const DIGIT_ANIM_MS = 220;
    let timers = [];
    const shown = { lat: "", lon: "" };

    function buildChars(el, text) {
        if (el.children.length !== text.length) {
            el.replaceChildren(...[...text].map(ch => {
                const span = document.createElement("span");
                span.textContent = ch;
                return span;
            }));
        }
    }

    function setLine(el, key, text, animate, startDelay) {
        const old = shown[key];
        shown[key] = text;
        if (old === text) {
            return startDelay;
        }
        if (!animate || reduced || !visible() || old.length !== text.length) {
            buildChars(el, text);
            [...text].forEach((ch, i) => { el.children[i].textContent = ch; });
            return startDelay;
        }
        buildChars(el, old);
        let delay = startDelay;
        for (let i = text.length - 1; i >= 0; i--) {          // right to left
            if (old[i] === text[i]) {
                continue;
            }
            const span = el.children[i];
            const ch = text[i];
            timers.push(setTimeout(() => {
                span.textContent = ch;
                span.animate([{ opacity: 0.1, transform: "translateY(-4px)" }, { opacity: 1, transform: "none" }],
                    { duration: DIGIT_ANIM_MS, easing: "ease-out" });
            }, delay));
            delay += DIGIT_MS;
        }
        return delay + 60;                                     // the next line starts after this one has settled
    }

    function setReadout(lat, lon, animate) {
        const latText = `${Math.abs(lat).toFixed(4)}\u00B0 ${lat >= 0 ? "N" : "S"}`;
        const lonText = `${Math.abs(lon).toFixed(4)}\u00B0 ${lon >= 0 ? "E" : "W"}`;
        if (latText === shown.lat && lonText === shown.lon) {
            return;
        }
        timers.forEach(clearTimeout);
        timers = [];
        // finish whatever was still rolling so the comparison below starts from what is on screen
        if (shown.lat) buildChars(latEl, shown.lat);
        if (shown.lon) buildChars(lonEl, shown.lon);
        [...shown.lat].forEach((ch, i) => { if (latEl.children[i]) latEl.children[i].textContent = ch; });
        [...shown.lon].forEach((ch, i) => { if (lonEl.children[i]) lonEl.children[i].textContent = ch; });
        const next = setLine(latEl, "lat", latText, animate, 0);
        setLine(lonEl, "lon", lonText, animate, next);
    }

    // idle value change: sweep the numbers from what is shown to the new value, so every digit churns the way it
    // does when the pointer is dragged quickly across the page, the decimals fastest and the left digits slowest
    let sweep = null;
    let displayed = { lat: DEFAULT_CENTRE.lat, lon: DEFAULT_CENTRE.lon };
    function sweepTo(target) {
        cancelSweep();
        if (reduced || !visible()) {
            displayed = { ...target };
            setReadout(target.lat, target.lon, false);
            return;
        }
        const from = { ...displayed };
        const t0 = performance.now();
        const step = now => {
            const k = easeOutCubic((now - t0) / SWEEP_MS);
            displayed = { lat: from.lat + (target.lat - from.lat) * k, lon: from.lon + (target.lon - from.lon) * k };
            setReadout(displayed.lat, displayed.lon, false);
            sweep = k < 1 ? requestAnimationFrame(step) : null;
        };
        sweep = requestAnimationFrame(step);
    }
    function cancelSweep() {
        if (sweep) {
            cancelAnimationFrame(sweep);
            sweep = null;
        }
    }

    function refresh({ fade = false, glide = true } = {}) {
        const read = currentRead();
        if (read && !sweep) {
            displayed = { lat: read.lat, lon: read.lon };
            setReadout(read.lat, read.lon, fade);
        }
        if (glide) {
            const to = targetView();
            if (to) {
                if (Math.abs(to.cLat - target.cLat) > 1e-9 || Math.abs(to.dpp - target.dpp) > 1e-12) {
                    glideTo(to);
                }
            }
        }
        requestDraw();
    }

    // The next idle scale: 2-6x larger or smaller than now, inside 0.01-1 degree per major tick, alternating
    // between expanding and contracting so the ruler breathes.
    let lastZoomDir = 1;
    function nextScale(dpp) {
        const lo = 0.01 / MAJOR_PX;
        const hi = 1 / MAJOR_PX;
        let dir = -lastZoomDir;
        const factor = rand(2, 6);
        if (dpp * factor > hi) dir = -1;
        if (dpp / factor < lo) dir = 1;
        lastZoomDir = dir;
        return Math.min(Math.max(dir > 0 ? dpp * factor : dpp / factor, lo), hi);
    }

    // ---- idle behaviour: one thing at a time, then a rest (never continuous). Off while a tile is selected.
    let idleTimer = 0;
    function scheduleIdleStep(delay) {
        clearTimeout(idleTimer);
        if (reduced || !idle || hasSelection) {
            return;
        }
        idleTimer = setTimeout(idleStep, delay);
    }

    function idleStep() {
        if (!idle || hasSelection || !visible()) {
            return;
        }
        if (moving || sweep) {
            scheduleIdleStep(1000);
            return;
        }
        const h = canvas.clientHeight || window.innerHeight;
        if (Math.random() < 0.4) {
            // a new place: the digits sweep, the ruler glides to it, usually at another scale
            idleRead = { lat: rand(...INDIA.lat), lon: rand(...INDIA.lon) };
            const dpp = nextScale(view.dpp);
            glideTo({ cLat: idleRead.lat + rand(-0.25, 0.25) * h * dpp, dpp }, TAU_MS);
            sweepTo(idleRead);
        } else {
            // the ruler alone: it breathes, expanding or contracting 2-6x (the ticks cross-fade), with a small pan
            const dpp = nextScale(view.dpp);
            glideTo({ cLat: view.cLat + rand(-90, 90) * view.dpp, dpp }, TAU_MS);
        }
        requestDraw();
        restAfterMotion();
    }

    // wait for everything to stop, then rest 7-14 s before the next step
    function restAfterMotion() {
        clearTimeout(idleTimer);
        idleTimer = setTimeout(function poll() {
            if (moving || sweep) {
                idleTimer = setTimeout(poll, 500);
            } else {
                scheduleIdleStep(rand(7000, 14000));
            }
        }, 500);
    }

    function enterIdle() {
        if (idle || reduced || hasSelection) {
            return;
        }
        idle = true;
        idleRead = { ...(tileHover ?? magnifyCoord ?? currentRead() ?? anchor) };
        scheduleIdleStep(rand(2500, 5000));
        refresh({ fade: true, glide: false });
    }

    function wake() {
        lastMove = performance.now();
        if (idle) {
            idle = false;
            cancelSweep();
            clearTimeout(idleTimer);
            refresh({ fade: true });
        }
    }

    setInterval(() => {
        if (!idle && visible() && performance.now() - lastMove > IDLE_MS) {
            enterIdle();
        }
    }, 1000);

    window.addEventListener("pointermove", e => {
        pointer = { x: e.clientX, y: e.clientY };
        wake();
        if (visible()) {
            refresh();
        }
    }, { passive: true });
    window.addEventListener("resize", requestDraw);
    new ResizeObserver(requestDraw).observe(canvas);

    // ---- inputs from the input view
    impl.hoverTile = geo => {
        const next = geo && Number.isFinite(geo.lat) && Number.isFinite(geo.lon) ? { lat: geo.lat, lon: geo.lon } : null;
        tileHover = next;
        wake();
        refresh({ fade: true });
    };
    impl.magnify = coord => {
        const before = currentMode();
        magnifyCoord = coord && Number.isFinite(coord.lat) ? coord : null;
        wake();
        refresh({ fade: before !== currentMode() });
    };
    impl.setAnchor = (geo, selected = Boolean(geo)) => {
        hasSelection = selected;
        anchor = geo && Number.isFinite(geo.lat) ? { lat: geo.lat, lon: geo.lon } : { ...DEFAULT_CENTRE };
        tileExtentDeg = geo?.bbox ? Math.max(Math.abs(geo.bbox[3] - geo.bbox[1]), 1e-4) : null;
        if (hasSelection && idle) {
            idle = false;                 // a selected tile ends the idle animation
            cancelSweep();
            clearTimeout(idleTimer);
        }
        refresh({ fade: true });
    };

    // ---- the footnote under the Scene Input box: purely decorative, changes from time to time with the same
    // right-to-left digit roll; deliberately not tied to any real data, the pointer or the selection
    const footEl = root.querySelector(".hud-foot");
    if (footEl && !reduced) {
        let footText = footEl.textContent.trim();
        let footTimers = [];
        const draw_ = text => footEl.replaceChildren(...[...text].map(ch => {
            const span = document.createElement("span");
            span.textContent = ch;
            return span;
        }));
        draw_(footText);
        const roll = next => {
            if (!visible()) {
                return;
            }
            const len = Math.max(footText.length, next.length);
            const from = footText.padEnd(len);
            const to = next.padEnd(len);
            draw_(from);
            footTimers.forEach(t => { clearTimeout(t); clearInterval(t); });
            footTimers = [];
            // every changed character spins through several random characters of its own kind (digits through
            // digits, letters through letters) before landing; the spinning starts at the right-hand end and the
            // rest follow one after another
            const pool = ch => (/[0-9]/.test(ch) ? "0123456789" : /[A-Z]/.test(ch) ? "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                : /[a-z]/.test(ch) ? "abcdefghijklmnopqrstuvwxyz" : null);
            let delay = 0;
            for (let i = len - 1; i >= 0; i--) {
                if (from[i] === to[i]) {
                    continue;
                }
                const span = footEl.children[i];
                const letters = pool(to[i]) ?? pool(from[i]);
                const spinMs = 520 + (len - 1 - i) * 40;
                footTimers.push(setTimeout(() => {
                    if (!letters) {
                        span.textContent = to[i];
                        return;
                    }
                    const t0 = performance.now();
                    const spin = setInterval(() => {
                        if (performance.now() - t0 >= spinMs) {
                            clearInterval(spin);
                            span.textContent = to[i];
                            span.animate([{ opacity: 0.4 }, { opacity: 1 }], { duration: 160 });
                        } else {
                            span.textContent = letters[Math.floor(Math.random() * letters.length)];
                        }
                    }, 55);
                    footTimers.push(spin);
                }, delay));
                delay += 70;
            }
            delay += 520 + len * 40;
            footTimers.push(setTimeout(() => draw_(next), delay + 300));
            footText = next;
        };
        const nextFoot = () => {
            let next = footText;
            while (next === footText) {
                const [name, code] = FOOT_CRS[Math.floor(Math.random() * FOOT_CRS.length)];
                next = `${name} \u00B7 EPSG:${code}`;
            }
            roll(next);
            setTimeout(nextFoot, rand(7000, 14000));
        };
        setTimeout(nextFoot, rand(5000, 9000));
    }

    new MutationObserver(requestDraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    document.fonts?.ready.then(requestDraw);
    refresh();
    draw();
}
