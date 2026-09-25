// Side-panel boxes of the expanded 3D view (#final-demo-box): Details, Image
// Inspection, Scenario Analysis, Learn Gestures, Explore Options.
//
// Every .xp-box collapses to a thin heading-only tab via its chevron. Only the
// boxes marked [data-default-open] in index.html start expanded.
//
// Details shows only what is actually known for the job's source (the input
// view's `meta` list already omits unknown fields, e.g. DFC2019 tiles have no
// acquisition date); nothing is filled in by guesswork here.

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
        note.textContent = job
            ? "The 3D terrain and the statistics above are the Darjeeling reference DSM, shown for every job (placeholder). "
                + (job.input.dem ? "The job's own DEM range is listed under Image source." : "")
            : "";
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
const LENS_ZOOM = 3;
let inspectWired = false;

function renderInspection(input) {
    const img = document.getElementById("xp-inspect-img");
    const empty = document.getElementById("xp-inspect-empty");
    const readout = document.getElementById("xp-inspect-readout");
    if (!img) {
        return;
    }
    wireInspection();
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
    img.onload = () => {
        readout.textContent = `${img.naturalWidth} × ${img.naturalHeight} px preview · hover to magnify (${LENS_ZOOM}×)`;
    };
}

function wireInspection() {
    if (inspectWired) {
        return;
    }
    inspectWired = true;
    const stage = document.getElementById("xp-inspect-stage");
    const img = document.getElementById("xp-inspect-img");
    const lens = document.getElementById("xp-inspect-lens");
    const readout = document.getElementById("xp-inspect-readout");
    stage.addEventListener("pointermove", e => {
        const r = img.getBoundingClientRect();
        if (img.hidden || !img.naturalWidth || !r.width) {
            return;
        }
        const fx = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
        const fy = Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1);
        const size = lens.offsetWidth || 96;
        lens.hidden = false;
        lens.style.left = `${e.clientX - stage.getBoundingClientRect().left - size / 2}px`;
        lens.style.top = `${e.clientY - stage.getBoundingClientRect().top - size / 2}px`;
        lens.style.backgroundImage = `url("${img.src}")`;
        lens.style.backgroundSize = `${r.width * LENS_ZOOM}px ${r.height * LENS_ZOOM}px`;
        lens.style.backgroundPosition = `${size / 2 - fx * r.width * LENS_ZOOM}px ${size / 2 - fy * r.height * LENS_ZOOM}px`;
        readout.textContent = `px ${Math.floor(fx * (img.naturalWidth - 1))}, ${Math.floor(fy * (img.naturalHeight - 1))}`
            + ` of ${img.naturalWidth} × ${img.naturalHeight} (preview)`;
    });
    stage.addEventListener("pointerleave", () => {
        lens.hidden = true;
        if (img.naturalWidth) {
            readout.textContent = `${img.naturalWidth} × ${img.naturalHeight} px preview · hover to magnify (${LENS_ZOOM}×)`;
        }
    });
}

// ------------------------------------------------------------------ tour
// Standard product tour: one control highlighted at a time, a short
// description, Next and Close. Started only from the Explore Options box.
const TOUR_STEPS = [
    { sel: "#job-tabs", title: "Job tabs", text: "Every generation is a separate job. Switch between open jobs here." },
    { sel: "#final-demo-back", title: "Back", text: "Shrinks this view back into its Workbench grid cell. Nothing is lost." },
    { sel: "#final-demo-close", title: "Close", text: "Closes the view and clears the jobs on this page (you are warned about unsaved work first)." },
    { sel: ".xv-play", title: "Auto-rotation", text: "Pauses or resumes the slow idle rotation of the terrain." },
    { sel: ".xv-theme", title: "Theme", text: "Switches between dark and bright mode." },
    { sel: '.xv-toolbar [aria-label="Screenshot"]', title: "Screenshot", text: "Captures the current view, including measurements and pins, as a PNG." },
    { sel: '.xv-toolbar [aria-label^="Measure"]', title: "Measure", text: "Point-to-point distance or a continuous line/area, read from the DEM." },
    { sel: '.xv-toolbar [aria-label^="View"]', title: "View", text: "Switch the surface between true colour, DSM and DEM layers." },
    { sel: '.xv-toolbar [aria-label="Flythrough"]', title: "Flythrough", text: "Plays a scripted camera flight over the terrain." },
    { sel: '.xv-toolbar [aria-label="Save current measurement"]', title: "Save", text: "Saves the current measurement; the arrow next to it opens everything saved this session." },
    { sel: '.xv-toolbar [aria-label="Show notes"]', title: "Notes", text: "Shows or hides the note pins you placed with right-click." },
    { sel: '.xv-toolbar [aria-label="Clear current selection"]', title: "Clear", text: "Clears the measurement currently being drawn." },
    { sel: '[data-box="details"]', title: "Details", text: "Image-source metadata for this job and statistics for the terrain shown." },
    { sel: '[data-box="inspect"]', title: "Image inspection", text: "The job's source image. Hover it to magnify and read pixel coordinates." },
    { sel: '[data-box="scenario"]', title: "Scenario analysis", text: "Illustrative flood and (placeholder) slope overlays. Not hazard models." },
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
