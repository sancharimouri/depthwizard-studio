// Side-panel boxes of the expanded 3D view (#final-demo-box): Details, Image
// Inspection, Scenario Analysis, Learn Gestures, Explore Options.
//
// Every .xp-box collapses to a thin heading-only tab via its chevron. Only the
// boxes marked [data-default-open] in index.html start expanded.
//
// Details shows only what is actually known for the job's source (the input
// view's `meta` list already omits unknown fields, e.g. DFC2019 tiles have no
// acquisition date); nothing is filled in by guesswork here.

import { countLines, factsBodyHtml, fitFactRows, infoQuery, loadGeoInfo, normaliseInfo, queryKey, scenarioCardHtml } from "./geo-info.js";
import { attachMagnifier } from "./magnifier.js";

const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// ------------------------------------------------------------------ boxes
export function initCollapsibleBoxes(root) {
    root.querySelectorAll(".xp-box").forEach(box => {
        const head = box.querySelector(".xp-head");
        const set = open => {
            box.classList.toggle("is-collapsed", !open);
            head.setAttribute("aria-expanded", String(open));
        };
        set(box.hasAttribute("data-default-open"));
        head.addEventListener("click", () => set(box.classList.contains("is-collapsed")));
        box.xpSetOpen = set;
    });
}

// ------------------------------------------------------------------ details
// Source metadata, in the order the user asked for: sensor, tile/scene ID,
// acquisition date, GSD — then whatever else the source really carries.
const SOURCE_ORDER = ["Sensor", "Tile ID", "Scene ID", "File", "Acquired", "GSD"];

export function sourceRows(input) {
    if (!input) {
        return [];
    }
    const rows = [];
    for (const [key, value] of input.meta ?? []) {
        if (value == null || value === "" || value === "—") {
            continue;
        }
        if (key === "Source") {
            rows.push(["Sensor", value]);
        } else if (key === "Tile") {
            rows.push(["Tile ID", value]);
        } else if (key === "File" && input.source === "search") {
            rows.push(["Scene ID", value]);
        } else {
            rows.push([key, value]);
        }
    }
    // A CDSE search result is Sentinel-2 by construction; an upload's sensor is unknown (omitted).
    if (input.source === "search" && !rows.some(([k]) => k === "Sensor")) {
        rows.unshift(["Sensor", "Sentinel-2 (CDSE search)"]);
    }
    if (input.dem) {
        rows.push(["Job DEM", `${input.dem.source} · ${input.dem.min_m}–${input.dem.max_m} m`]);
    }
    const rank = k => {
        const i = SOURCE_ORDER.indexOf(k);
        return i < 0 ? SOURCE_ORDER.length : i;
    };
    return rows.map((row, i) => [row, i]).sort((a, b) => rank(a[0][0]) - rank(b[0][0]) || a[1] - b[1]).map(([row]) => row);
}

export function renderSource(job) {
    const dl = document.getElementById("xp-source-meta");
    if (!dl) {
        return;
    }
    const rows = sourceRows(job?.input);
    dl.innerHTML = rows.length
        ? rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")
        : `<dt>Source</dt><dd>Reference scene (no job input)</dd>`;
    const note = document.getElementById("xp-terrain-note");
    if (note) {
        const m = job?.gen?.status === "ok" ? job.gen.meta : null;
        note.textContent = !job ? ""
            : !m ? "Terrain not generated for this job."
            : [m.has_elevation
                ? `The 3D terrain and its statistics are this job's own: surface from ${m.surface_source}, `
                    + `terrain from ${m.terrain_source}.`
                : m.note,
               m.display_note, // Maxar DISPLAY band: the mesh shape is cosmetic, statistics are real
               m.credit]       // DFC2019 attribution
                .filter(Boolean).join(" ");
    }
    renderInspection(job?.input);
}

export function terrainStats(terrainData) {
    const { heights, elevationMin: lo, elevationMax: hi } = terrainData;
    let sum = 0;
    let n = 0;
    for (let i = 0; i < heights.length; i += 1) {
        const h = heights[i];
        if (Number.isFinite(h)) {
            sum += h;
            n += 1;
        }
    }
    const mean = n ? lo + (sum / n) * (hi - lo) : null;
    return { min: lo, max: hi, mean, relief: hi - lo };
}

export function renderTerrainStats(terrainData) {
    const s = terrainStats(terrainData);
    const fmt = v => (v == null ? "—" : `${Math.round(v).toLocaleString()} m`);
    [["xp-elev-min", s.min], ["xp-elev-max", s.max], ["xp-elev-mean", s.mean], ["xp-elev-relief", s.relief]]
        .forEach(([id, v]) => {
            const node = document.getElementById(id);
            if (node) {
                node.textContent = fmt(v);
            }
        });
}

// ------------------------------------------------------------------ inspection
// The shared magnifier (src/magnifier.js). Hover / pick are forwarded to the
// 3D viewer (main.js) through setInspectionHandlers().
const LENS_ZOOM = 3;
let inspectMag = null;
let inspectHandlers = {};

export function setInspectionHandlers(handlers) {
    inspectHandlers = handlers;
}

// the selected pixel's marker on the inspection image (null hides it)
export function setInspectionSelected(uv) {
    inspectMag?.setSelected(uv);
}

function renderInspection(input) {
    const img = document.getElementById("xp-inspect-img");
    const empty = document.getElementById("xp-inspect-empty");
    if (!img) {
        return;
    }
    if (!inspectMag) {
        inspectMag = attachMagnifier({
            stage: document.getElementById("xp-inspect-stage"),
            img,
            readout: document.getElementById("xp-inspect-readout"),
            zoom: LENS_ZOOM,
            onHover: (u, v) => inspectHandlers.onHover?.(u, v),
            onLeave: () => inspectHandlers.onLeave?.(),
            onPick: (u, v) => inspectHandlers.onPick?.(u, v),
        });
    }
    const url = input?.previewUrl ?? "data/darjeeling/satellite.png";
    if (img.getAttribute("src") !== url) {
        img.hidden = false;
        empty.hidden = true;
        img.src = url;
    }
    img.onerror = () => {
        img.hidden = true;
        empty.hidden = false;
    };
}

// ------------------------------------------------------------------ tour
// Standard product tour: one control highlighted at a time, a short
// description, Next and Close. Started only from the Explore Options box.
const TOUR_STEPS = [
    { sel: "#job-tabs", title: "Job tabs", text: "Every generation is a separate job. Switch between open jobs here." },
    { sel: "#final-demo-back", title: "Back", text: "Shrinks this view back into its DW Studio grid cell. Nothing is lost." },
    { sel: "#final-demo-close", title: "Close", text: "Closes the view and clears the jobs on this page (you are warned about unsaved work first)." },
    { sel: "#final-demo-undo", title: "Undo / Redo", text: "Steps back and forward through your rotations, measurements, notes and view changes (also Cmd/Ctrl+Z and Cmd/Ctrl+Shift+Z)." },
    { sel: ".xv-theme", title: "Theme", text: "Switches between dark and bright mode." },
    { sel: ".xv-navbar", title: "Navigation", text: "Top/side view, camera up/down, play/pause the rotation, rotate the structure clockwise/anticlockwise (hover to set the step), and lock the view." },
    { sel: '.xv-toolbar [aria-label="Pointer"]', title: "Pointer", text: "Back to plain navigation (no measuring)." },
    { sel: '.xv-toolbar [aria-label="Screenshot"]', title: "Screenshot", text: "Captures the current view, including measurements and pins, as a PNG." },
    { sel: '.xv-toolbar [aria-label="Measure"]', title: "Measure", text: "Point-to-point distance or a continuous line/area, read from the DEM." },
    { sel: '.xv-toolbar [aria-label="View"]', title: "View", text: "Switch the surface between true colour, DSM, DEM and wireframe." },
    { sel: '.xv-toolbar [aria-label="Reset view"]', title: "Reset view", text: "Returns the camera and the structure to the default framing." },
    { sel: '.xv-toolbar [aria-label="Notes"]', title: "Notes", text: "Lists every note you placed (right-click → Add note), each with its own show/hide switch and delete." },
    { sel: '.xv-toolbar [aria-label="Save"]', title: "Save", text: "Saves the current measurement; the arrow next to it opens everything saved this session." },
    { sel: '.xv-toolbar [aria-label="Trash"]', title: "Trash", text: "Clears the measurement currently being drawn." },
    { sel: '[data-box="terrain"]', title: "Terrain statistics", text: "Elevation range, mean, relief and mesh grid of the terrain shown, plus the vertical exaggeration control." },
    { sel: '[data-box="details"]', title: "Details", text: "Where the image and the terrain come from: sensor, tile, date, GSD, CRS." },
    { sel: '[data-box="inspect"]', title: "Image inspection", text: "The job's source image. Hover it to magnify; click to select that point on the 3D surface." },
    { sel: '[data-box="scenario"]', title: "Scenario analysis", text: "Illustrative flood and (placeholder) slope overlays, and a Landslide option that is coming soon. Not hazard models." },
    { sel: '[data-box="flythrough"]', title: "Fly-through", text: "One full orbit while zooming in, with pause/resume, Run again and Reset." },
    { sel: '[data-box="facts"]', title: "Facts", text: "Hazards not covered by Scenario Analysis, and named peaks, rivers and glaciers near this place." },
    { sel: '[data-box="gestures"]', title: "Learn gestures", text: "How to rotate, zoom and use the terrain menu with mouse, trackpad or touch." },
];

export function initTour(box) {
    const startBtn = box.querySelector("#xp-tour-start");
    if (!startBtn) {
        return { stop() {} };
    }
    const ring = document.createElement("div");
    ring.className = "xp-tour-ring";
    ring.hidden = true;
    ring.setAttribute("data-xv-ui", "");
    const card = document.createElement("div");
    card.className = "xp-tour-card";
    card.hidden = true;
    card.setAttribute("role", "dialog");
    card.setAttribute("aria-live", "polite");
    card.setAttribute("data-xv-ui", "");
    card.setAttribute("data-xv-popover", "");
    card.innerHTML = `
        <div class="xp-tour-count numeric-mono"></div>
        <div class="xp-tour-title"></div>
        <p class="xp-tour-text"></p>
        <div class="xp-tour-actions">
            <button type="button" class="xp-tour-next">Next</button>
            <button type="button" class="xp-tour-close">Close</button>
        </div>`;
    box.append(ring, card);

    let steps = [];
    let index = -1;

    const visible = node => node && node.getClientRects().length > 0 && getComputedStyle(node).visibility !== "hidden";

    function place() {
        const step = steps[index];
        const target = step && box.querySelector(step.sel);
        if (!visible(target)) {
            return;
        }
        const b = box.getBoundingClientRect();
        const r = target.getBoundingClientRect();
        const pad = 4;
        Object.assign(ring.style, {
            left: `${r.left - b.left - pad}px`,
            top: `${r.top - b.top - pad}px`,
            width: `${r.width + pad * 2}px`,
            height: `${r.height + pad * 2}px`,
        });
        const cw = card.offsetWidth;
        const ch = card.offsetHeight;
        // below the target if it fits, else above; clamped inside the view
        let top = r.bottom - b.top + 12;
        if (top + ch > b.height - 8) {
            top = r.top - b.top - ch - 12;
        }
        let left = r.left - b.left + r.width / 2 - cw / 2;
        // side boxes: put the card beside them instead of on top
        if (target.closest(".final-demo-left-rail")) {
            left = r.right - b.left + 14;
            top = r.top - b.top;
        } else if (target.closest(".final-demo-right-rail")) {
            left = r.left - b.left - cw - 14;
            top = r.top - b.top;
        }
        card.style.left = `${Math.min(Math.max(left, 8), b.width - cw - 8)}px`;
        card.style.top = `${Math.min(Math.max(top, 8), b.height - ch - 8)}px`;
    }

    function show(i) {
        index = i;
        const step = steps[i];
        card.querySelector(".xp-tour-count").textContent = `${i + 1} / ${steps.length}`;
        card.querySelector(".xp-tour-title").textContent = step.title;
        card.querySelector(".xp-tour-text").textContent = step.text;
        const last = i === steps.length - 1;
        const next = card.querySelector(".xp-tour-next");
        next.textContent = last ? "Finish" : "Next";
        ring.hidden = false;
        card.hidden = false;
        place();
        next.focus({ preventScroll: true });
    }

    function stop() {
        index = -1;
        ring.hidden = true;
        card.hidden = true;
    }

    startBtn.addEventListener("click", () => {
        steps = TOUR_STEPS.filter(s => visible(box.querySelector(s.sel)));
        if (steps.length) {
            show(0);
        }
    });
    card.querySelector(".xp-tour-next").addEventListener("click", () => {
        if (index < steps.length - 1) {
            show(index + 1);
        } else {
            stop();
        }
    });
    card.querySelector(".xp-tour-close").addEventListener("click", stop);
    document.addEventListener("keydown", e => {
        if (e.key === "Escape" && index >= 0) {
            stop();
        }
    });
    window.addEventListener("resize", () => {
        if (index >= 0) {
            place();
        }
    });
    return { stop };
}

// ------------------------------------------------------------------ facts + scenario cards
// Facts box: hazards Scenario Analysis doesn't cover + named terrain features, one short line each
// (src/geo-info.js). Scenario Analysis cards: flood / landslide / earthquake lines + the epicentre map.
// Library tiles in the web build use the curated, baked lines (no request); everything else asks
// /api/facts once per job and location, only while a box that needs it is open.

let factsJob = null;
let scenarioKey = null;

function factsBox() {
    return document.querySelector('[data-box="facts"]');
}

function isOpen(box) {
    return box && !box.hidden && !box.classList.contains("is-collapsed");
}

export function renderFacts(job) {
    factsJob = job ?? null;
    const box = factsBox();
    const where = document.getElementById("xp-facts-where");
    const form = document.getElementById("xp-facts-form");
    const body = document.getElementById("xp-facts-body");
    if (!box || !where || !body) {
        return;
    }
    where.textContent = "";
    const baked = job?.input?.staticInfo;
    if (baked) {
        // web build, library tile: hidden when nothing is curated for it
        const info = normaliseInfo(baked);
        box.hidden = countLines(info) === 0;
        form.hidden = true;
        body.innerHTML = box.hidden ? "" : factsBodyHtml(info);
        fitFactRows(body);
        renderScenarioCard(job, scenarioKey);
        return;
    }
    box.hidden = false;
    if (!job) {
        form.hidden = true;
        body.innerHTML = `<p class="xp-fact-empty">No job is open.</p>`;
        return;
    }
    const q = infoQuery(job);
    form.hidden = Boolean(job.input?.geo);
    if (!q) {
        body.innerHTML = `<p class="xp-fact-empty">Enter this image's coordinates to look up its hazards and named features.</p>`;
        return;
    }
    if (job.userGeo && !job.input?.geo) {
        document.getElementById("xp-facts-lat").value = job.userGeo.lat;
        document.getElementById("xp-facts-lon").value = job.userGeo.lon;
    }
    if (!isOpen(box) && !scenarioKey) {
        return; // external sources only while a box that shows them is open
    }
    body.innerHTML = `<p class="xp-fact-empty">Looking up this place…</p>`;
    loadGeoInfo(job).then(res => {
        if (factsJob !== job || res.key !== queryKey(infoQuery(job))) {
            return;
        }
        // the whole lookup failed (offline, backend asleep): say so plainly, never an error text, never "no hazards"
        body.innerHTML = res.failed
            ? `<p class="xp-fact-empty">Couldn't look this place up just now.</p>`
                + `<button class="xp-tour-close" type="button" data-facts-retry>Retry</button>`
            : factsBodyHtml(res.info);
        fitFactRows(body);
        renderScenarioCard(job, scenarioKey);
    });
}

// The card under the Scenario Analysis buttons for the active option ("flood" / "landslide" / "earthquake"
// / null). Hidden when that option has no lines.
export function renderScenarioCard(job, key) {
    scenarioKey = key ?? null;
    const card = document.getElementById("xp-scenario-card");
    if (!card) {
        return;
    }
    if (!job || !key || (!job.input?.staticInfo && !infoQuery(job))) {
        card.hidden = true;
        card.innerHTML = "";
        return;
    }
    loadGeoInfo(job).then(res => {
        if (scenarioKey !== key || factsJob !== job) {
            return;
        }
        const html = scenarioCardHtml(res.info, key);
        card.hidden = !html;
        card.innerHTML = html;
        fitFactRows(card);
    });
}

export function initFacts(getActiveJob) {
    const box = factsBox();
    const form = document.getElementById("xp-facts-form");
    if (!box || !form) {
        return;
    }
    box.querySelector(".xp-head").addEventListener("click", () => renderFacts(getActiveJob()));
    // rows are fitted by measuring, which needs a visible box: re-fit the card when Scenario Analysis opens
    document.querySelector('[data-box="scenario"] .xp-head')?.addEventListener("click", () =>
        requestAnimationFrame(() => fitFactRows(document.getElementById("xp-scenario-card"))));
    // "Flood · Earthquake · Landslide → Scenario Analysis" opens that box
    box.addEventListener("click", event => {
        if (event.target.closest("[data-facts-retry]")) {
            renderFacts(getActiveJob());
            return;
        }
        if (!event.target.closest("[data-open-scenario]")) {
            return;
        }
        const scenario = document.querySelector('[data-box="scenario"]');
        scenario?.xpSetOpen?.(true);
        requestAnimationFrame(() => fitFactRows(document.getElementById("xp-scenario-card")));
        scenario?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
    form.addEventListener("submit", event => {
        event.preventDefault();
        const job = getActiveJob();
        const lat = Number(document.getElementById("xp-facts-lat").value);
        const lon = Number(document.getElementById("xp-facts-lon").value);
        if (!job || !Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
            return;
        }
        job.userGeo = { lat, lon };
        renderFacts(job);
    });
}
