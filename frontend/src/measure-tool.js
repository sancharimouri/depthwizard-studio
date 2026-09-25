// 3D measurement tool for Workbench's expanded Final Demo viewer
// (deliverables-audit item 6: slope assessment / height analysis).
//
// Glue between the tested pure modules and the page:
//   heightfield.js     exact, fast ray → terrain picking
//   measure-metrics.js distances / rise / gradient / area from the RAW DEM
//   measure-model.js   modes, click rules, undo stack, session-only save
// This file draws the selection (an SVG overlay above the WebGL canvas:
// crisp glowing lines, native hover/right-click/cursors) and handles pointer
// and keyboard input. The toolbar, library, terrain context menu and notes
// around it live in expanded-chrome.js, which drives this tool through the
// API returned below.

import * as THREE from "three";
import { raycastHeightfield, surfaceAt, gridToLocalXY } from "./heightfield.js";
import { createGeoGrid, measureChain } from "./measure-metrics.js";
import { createMeasureModel, MODES } from "./measure-model.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const CLICK_SLOP_PX = 5;      // pointer travel below this = a click, not an orbit/drag
const LINE_LIFT = 0.12;       // local units above the surface, so lines don't z-fight
const TOAST_MS = 3200;

const fmtM = m => (Math.abs(m) >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(0)} m`);
const fmtSigned = m => `${m >= 0 ? "+" : "−"}${fmtM(Math.abs(m))}`;
const fmtArea = a => (a >= 1e6 ? `${(a / 1e6).toFixed(3)} km²` : `${Math.round(a).toLocaleString()} m²`);
const fmtPct = p => (p === null || p === undefined ? "—" : `${p >= 0 ? "+" : "−"}${Math.abs(p).toFixed(1)}%`);

function el(tag, attrs = {}, parent = null, ns = null) {
    const node = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "text") {
            node.textContent = v;
        } else {
            node.setAttribute(k, v);
        }
    }
    parent?.appendChild(node);
    return node;
}

export function createMeasureTool({ viewer, box, canvas }) {
    const model = createMeasureModel({ onChange: () => { dirty = true; } });
    let dirty = true;

    // ---------------------------------------------------------------- terrain adapters
    let terrainRef = null;
    let geo = null;
    let field = null;
    let fieldVersion = -1;
    let fieldMask = undefined;

    function syncTerrain() {
        const terrain = viewer.currentTerrain;
        if (!terrain) {
            return null;
        }
        if (terrain !== terrainRef) {
            // A different region's grid: coordinates from the old one are meaningless.
            terrainRef = terrain;
            geo = createGeoGrid(terrain.grid);
            fieldVersion = -1;
            if (model.pointCount() > 0) {
                model.clear();
            }
        }
        const pos = terrain.mesh.geometry.attributes.position;
        const mask = terrain.erodedVertexMask();
        if (pos.version !== fieldVersion || mask !== fieldMask) {
            let zMin = Infinity;
            let zMax = -Infinity;
            for (let i = 0; i < pos.count; i++) {
                const z = pos.getZ(i);
                zMin = Math.min(zMin, z);
                zMax = Math.max(zMax, z);
            }
            field = {
                width: terrain.grid.width,
                height: terrain.grid.height,
                terrainWidth: terrain.terrainWidth,
                terrainHeight: terrain.terrainHeight,
                getZ: i => pos.getZ(i),
                zMin,
                zMax,
                triangleValid: mask ? (a, b, c) => !mask[a] && !mask[b] && !mask[c] : null,
            };
            fieldVersion = pos.version;
            fieldMask = mask;
            dirty = true;
        }
        return terrain;
    }

    const ndc = new THREE.Vector2();
    const raycaster = new THREE.Raycaster();
    const invMatrix = new THREE.Matrix4();
    const tmpA = new THREE.Vector3();
    const tmpB = new THREE.Vector3();

    // Grid position under a client (screen) point, or null if off the terrain.
    function pick(clientX, clientY) {
        const terrain = syncTerrain();
        if (!terrain) {
            return null;
        }
        const rect = canvas.getBoundingClientRect();
        ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
        raycaster.setFromCamera(ndc, viewer.camera);
        terrain.mesh.updateWorldMatrix(true, false);
        invMatrix.copy(terrain.mesh.matrixWorld).invert();
        const o = tmpA.copy(raycaster.ray.origin).applyMatrix4(invMatrix);
        const d = tmpB.copy(raycaster.ray.origin).add(raycaster.ray.direction).applyMatrix4(invMatrix).sub(o);
        const hit = raycastHeightfield(field, o, d);
        return hit ? { x: hit.col, y: hit.row } : null;
    }

    function onTerrain(x, y) {
        return !!(field && surfaceAt(field, x, y));
    }

    const world = new THREE.Vector3();
    // Screen position (px, relative to the canvas) of a grid point on the
    // CURRENT display surface (3D or flat), or null if behind the camera.
    function project(x, y, lift = 0) {
        const s = surfaceAt(field, x, y);
        if (!s) {
            return null;
        }
        const xy = gridToLocalXY(field, x, y);
        world.set(xy.x, xy.y, s.z + lift).applyMatrix4(terrainRef.mesh.matrixWorld);
        world.project(viewer.camera);
        if (world.z > 1 || world.z < -1) {
            return null;
        }
        return { sx: ((world.x + 1) / 2) * canvas.clientWidth, sy: ((1 - world.y) / 2) * canvas.clientHeight };
    }

    // ---------------------------------------------------------------- DOM
    const overlay = el("svg", { class: "measure-overlay", "aria-hidden": "true" }, box, SVG_NS);
    const gFill = el("g", { class: "m-fills" }, overlay, SVG_NS);
    const gSegs = el("g", { class: "m-segs" }, overlay, SVG_NS);
    const gPts = el("g", { class: "m-pts" }, overlay, SVG_NS);

    const readout = el("div", { class: "measure-readout", hidden: "" }, box);
    const toast = el("div", { class: "measure-toast", role: "alert", hidden: "" }, box);
    const menu = el("div", { class: "measure-menu", role: "menu", hidden: "" }, box);
    const status = el("div", { class: "measure-status", hidden: "" }, box);

    // ---------------------------------------------------------------- feedback
    let toastTimer = null;
    function showToast(message, kind = "error") {
        toast.textContent = message;
        toast.dataset.kind = kind;
        toast.hidden = false;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { toast.hidden = true; }, TOAST_MS);
    }

    function handleOutcome(r) {
        if (!r) {
            return;
        }
        if (r.type === "error") {
            showToast(r.message, "error");
        } else if (r.type === "duplicate") {
            showToast(r.message, "info");
        } else if (r.type === "closed") {
            showToast("Shape closed — measuring it as a polygon (area and perimeter).", "info");
        } else if (r.type === "saved") {
            showToast(`Saved ${r.items.length === 1 ? "measurement" : `${r.items.length} measurements`} for this session only — not stored yet.`, "info");
        } else if (r.type === "undone" && r.mode) {
            showToast(`Undo restored the ${r.mode === MODES.TWO_POINT ? "two-point" : "continuous"} measurement.`, "info");
        }
    }

    // ---------------------------------------------------------------- drawing
    function drapedPath(p0, p1) {
        // Follows the displayed surface between the points (sampled every
        // half grid pixel) so the line reads as lying on the terrain.
        const n = Math.max(1, Math.ceil(Math.hypot(p1.x - p0.x, p1.y - p0.y) * 2));
        let d = "";
        let pen = false;
        for (let i = 0; i <= n; i++) {
            const t = i / n;
            const s = project(p0.x + (p1.x - p0.x) * t, p0.y + (p1.y - p0.y) * t, LINE_LIFT);
            if (!s) {
                pen = false; // gap over an eroded hole / behind camera
                continue;
            }
            d += `${pen ? "L" : "M"}${s.sx.toFixed(1)},${s.sy.toFixed(1)}`;
            pen = true;
        }
        return d;
    }

    let dragRef = null;

    // Persistent SVG elements, reused and updated in place (keyed by
    // chain:index). Rebuilding them every frame would drop :hover whenever
    // the camera moves under a stationary cursor.
    const segEls = new Map();
    const ptEls = new Map();
    const fillEls = new Map();

    function segEl(key) {
        let g = segEls.get(key);
        if (!g) {
            g = el("g", { class: "m-seg" }, gSegs, SVG_NS);
            g._glow = el("path", { class: "m-seg-glow" }, g, SVG_NS);
            g._line = el("path", { class: "m-seg-line" }, g, SVG_NS);
            g._hit = el("path", { class: "m-seg-hit" }, g, SVG_NS);
            segEls.set(key, g);
        }
        return g;
    }

    function ptEl(key) {
        let g = ptEls.get(key);
        if (!g) {
            g = el("g", { class: "m-pt" }, gPts, SVG_NS);
            el("circle", { class: "m-pt-hit", r: 11 }, g, SVG_NS);
            el("circle", { class: "m-pt-dot", r: 5 }, g, SVG_NS);
            g._title = el("title", {}, g, SVG_NS);
            ptEls.set(key, g);
        }
        return g;
    }

    function prune(map, live) {
        for (const [key, node] of map) {
            if (!live.has(key)) {
                node.remove();
                map.delete(key);
            }
        }
    }

    function render() {
        if (!syncTerrain()) {
            return;
        }
        const editable = model.mode !== MODES.NORMAL;
        overlay.classList.toggle("is-editable", editable);
        viewer.controls.setAutoRotateHold?.(editable); // also covers undo restoring a mode
        const liveSeg = new Set();
        const livePt = new Set();
        const liveFill = new Set();

        model.chains.forEach((chain, ci) => {
            const pts = chain.points;
            if (chain.closed && pts.length >= 3) {
                const ring = pts.map(p => project(p.x, p.y, LINE_LIFT)).filter(Boolean);
                if (ring.length >= 3) {
                    let f = fillEls.get(ci);
                    if (!f) {
                        f = el("path", { class: "m-fill" }, gFill, SVG_NS);
                        fillEls.set(ci, f);
                    }
                    f.setAttribute("d", `M${ring.map(q => `${q.sx.toFixed(1)},${q.sy.toFixed(1)}`).join("L")}Z`);
                    liveFill.add(ci);
                }
            }
            const segCount = chain.closed ? pts.length : pts.length - 1;
            for (let si = 0; si < segCount; si++) {
                const key = `${ci}:${si}`;
                const d = drapedPath(pts[si], pts[(si + 1) % pts.length]);
                const g = segEl(key);
                g.setAttribute("class", `m-seg${chain.closed ? " is-closed" : ""}`);
                g.setAttribute("data-chain", ci);
                g.setAttribute("data-index", si);
                g._glow.setAttribute("d", d);
                g._line.setAttribute("d", d);
                g._hit.setAttribute("d", d);
                liveSeg.add(key);
            }
            pts.forEach((p, pi) => {
                const q = project(p.x, p.y, LINE_LIFT);
                if (!q) {
                    return;
                }
                const key = `${ci}:${pi}`;
                const canClose = model.mode === MODES.CONTINUOUS && ci === model.activeChain && pi === 0 && pts.length >= 3;
                const dragging = dragRef && dragRef.chain === ci && dragRef.index === pi;
                const g = ptEl(key);
                g.setAttribute("class", `m-pt${dragging ? " is-dragging" : ""}${canClose ? " can-close" : ""}`);
                g.setAttribute("data-chain", ci);
                g.setAttribute("data-index", pi);
                g.setAttribute("transform", `translate(${q.sx.toFixed(1)},${q.sy.toFixed(1)})`);
                g._title.textContent = canClose ? "Click to close the shape" : `Point ${pi + 1} — drag to move, right-click to edit`;
                livePt.add(key);
            });
        });
        prune(segEls, liveSeg);
        prune(ptEls, livePt);
        prune(fillEls, liveFill);
        renderStatus();
        renderToolbar();
        dirty = false;
    }

    function selectionMetrics() {
        const withLine = model.chains
            .map((c, ci) => ({ c, ci }))
            .filter(({ c }) => c.points.length >= 2);
        return withLine.map(({ c, ci }) => ({ ci, m: measureChain(geo, c.points, c.closed) }));
    }

    function renderStatus() {
        const show = model.mode !== MODES.NORMAL || model.pointCount() > 0;
        status.hidden = !show || !box.classList.contains("is-expanded");
        if (status.hidden) {
            return;
        }
        const parts = [];
        const modeLabel = { [MODES.TWO_POINT]: "Two points", [MODES.CONTINUOUS]: "Continuous", [MODES.NORMAL]: "Pointer" }[model.mode];
        parts.push(`<span class="ms-mode">${modeLabel}</span>`);
        const metrics = selectionMetrics();
        if (!metrics.length) {
            const n = model.pointCount();
            const hint = model.mode === MODES.NORMAL
                ? "Selection shown — choose Measure to edit it."
                : n === 0
                    ? (model.mode === MODES.TWO_POINT ? "Click the terrain to place point A." : "Click the terrain to start a chain.")
                    : (model.mode === MODES.TWO_POINT ? "Click the terrain to place point B." : "Click the terrain to add the next point.");
            parts.push(`<span class="ms-hint">${hint}</span>`);
        }
        for (const { m } of metrics.slice(0, 2)) {
            if (m.type === "polygon") {
                parts.push(`<span><b>Polygon</b> area ${fmtArea(m.area)} · perimeter ${fmtM(m.perimeter)} · ${m.points.length} pts · elev ${Math.round(m.minElevation)}–${Math.round(m.maxElevation)} m</span>`);
            } else {
                const s = m.segments;
                parts.push(`<span><b>${m.type === "segment" ? "A → B" : `Chain (${m.points.length} pts)`}</b> ` +
                    `${fmtM(m.horizontal)} horizontal · ${fmtM(m.surface)} along surface · rise ${fmtSigned(m.netRise)} · gradient ${fmtPct(m.gradient)}` +
                    `${m.type === "segment" ? ` (${s[0].slopeDeg.toFixed(1)}°)` : ` · ↑${fmtM(m.ascent)} ↓${fmtM(m.descent)}`}</span>`);
            }
        }
        if (metrics.length > 2) {
            parts.push(`<span>+${metrics.length - 2} more</span>`);
        }
        status.innerHTML = parts.join("") +
            `<button type="button" class="ms-undo" ${model.canUndo ? "" : "disabled"} title="Undo (Ctrl/⌘+Z)">↶ Undo</button>`;
        status.querySelector(".ms-undo").addEventListener("click", () => { handleOutcome(model.undo()); dirty = true; });
    }

    const renderListeners = [];
    function renderToolbar() {
        canvas.classList.toggle("is-measuring", model.mode !== MODES.NORMAL);
        for (const fn of renderListeners) {
            fn();
        }
    }

    function setMode(mode) {
        closeMenu();
        const had = model.pointCount();
        const r = model.setMode(mode);
        if (mode !== MODES.NORMAL && had > 0 && model.pointCount() === 0) {
            showToast("Switched measurement type — the previous selection was cleared (Undo brings it back).", "info");
        }
        if (mode === MODES.NORMAL) {
            readout.hidden = true;
        }
        viewer.controls.setAutoRotateHold?.(model.mode !== MODES.NORMAL);
        dirty = true;
        return r;
    }

    // ---------------------------------------------------------------- context menu
    let menuRef = null;
    function closeMenu() {
        menu.hidden = true;
        menu.replaceChildren();
        menuRef = null;
    }

    function openMenu(kind, ref, clientX, clientY) {
        closeMenu();
        menuRef = { kind, ...ref };
        const boxRect = box.getBoundingClientRect();
        const chain = model.chains[ref.chain];
        if (kind === "point") {
            const p = chain.points[ref.index];
            el("div", { class: "mm-title", text: `Point ${ref.index + 1}${model.chains.length > 1 ? ` · chain ${ref.chain + 1}` : ""}` }, menu);
            const form = el("form", { class: "mm-form", novalidate: "" }, menu);
            const row = el("div", { class: "mm-row" }, form);
            const lx = el("label", { text: "X" }, row);
            const ix = el("input", { type: "number", step: "any", min: "0", max: String(geo.width - 1), value: p.x.toFixed(1), "aria-label": "X (pixel column)" }, lx);
            const ly = el("label", { text: "Y" }, row);
            const iy = el("input", { type: "number", step: "any", min: "0", max: String(geo.height - 1), value: p.y.toFixed(1), "aria-label": "Y (pixel row)" }, ly);
            el("div", { class: "mm-help", text: `Grid pixels · X 0–${geo.width - 1}, Y 0–${geo.height - 1}` }, form);
            const err = el("div", { class: "mm-error", role: "alert", hidden: "" }, form);
            const actions = el("div", { class: "mm-actions" }, form);
            el("button", { type: "submit", class: "mm-apply", text: "Apply" }, actions);
            const del = el("button", { type: "button", class: "mm-danger", text: "Delete point" }, actions);
            form.addEventListener("submit", e => {
                e.preventDefault();
                const x = Number(ix.value);
                const y = Number(iy.value);
                if (!Number.isFinite(x) || !Number.isFinite(y) || ix.value === "" || iy.value === "") {
                    err.textContent = "Enter numbers for both X and Y.";
                } else if (!geo.inGrid(x, y)) {
                    err.textContent = `Out of range: X must be 0–${geo.width - 1} and Y 0–${geo.height - 1}.`;
                } else if (!onTerrain(x, y)) {
                    err.textContent = "That position isn't on the terrain surface.";
                } else {
                    handleOutcome(model.setPointXY(ref, x, y));
                    closeMenu();
                    return;
                }
                err.hidden = false;
            });
            del.addEventListener("click", () => { handleOutcome(model.deletePoint(ref)); closeMenu(); });
            setTimeout(() => ix.focus(), 0);
        } else {
            el("div", { class: "mm-title", text: model.mode === MODES.TWO_POINT ? "Line A → B" : `Line ${ref.index + 1}` }, menu);
            if (model.mode === MODES.TWO_POINT) {
                el("div", { class: "mm-help", text: "Deletes the line and both of its points." }, menu);
            }
            const del = el("button", { type: "button", class: "mm-danger", text: "Delete line" }, menu);
            del.addEventListener("click", () => { handleOutcome(model.deleteSegment(ref)); closeMenu(); });
            setTimeout(() => del.focus(), 0);
        }
        menu.hidden = false;
        const mw = menu.offsetWidth;
        const mh = menu.offsetHeight;
        menu.style.left = `${Math.min(clientX - boxRect.left + 6, boxRect.width - mw - 8)}px`;
        menu.style.top = `${Math.min(clientY - boxRect.top + 6, boxRect.height - mh - 8)}px`;
    }

    // ---------------------------------------------------------------- pointer input
    const refOf = node => {
        const g = node?.closest?.("[data-chain]");
        return g ? { chain: Number(g.dataset.chain), index: Number(g.dataset.index), kind: g.classList.contains("m-pt") ? "point" : "segment" } : null;
    };
    const editable = () => model.mode !== MODES.NORMAL && box.classList.contains("is-expanded");

    let down = null;

    box.addEventListener("pointerdown", e => {
        if (e.button !== 0 || !editable()) {
            return;
        }
        if (!menu.hidden && !menu.contains(e.target)) {
            closeMenu();
        }
        if (e.target.closest?.("[data-xv-ui]") || menu.contains(e.target) || status.contains(e.target)) {
            return; // the surrounding chrome (toolbar, library, notes…) owns these
        }
        const ref = refOf(e.target);
        if (ref && ref.kind === "point") {
            // Point drag: owns the gesture, so the camera must not orbit.
            e.preventDefault();
            e.stopPropagation();
            model.beginDrag(ref);
            dragRef = ref;
            viewer.controls.enabled = false;
            box.setPointerCapture?.(e.pointerId);
            down = { x: e.clientX, y: e.clientY, ref, moved: false };
            dirty = true;
            return;
        }
        if (e.target === canvas || (ref && ref.kind === "segment")) {
            down = { x: e.clientX, y: e.clientY, ref: null, moved: false };
        }
    }, { capture: true });

    let hoverFrame = 0;
    let lastMove = null;
    box.addEventListener("pointermove", e => {
        lastMove = e;
        if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > CLICK_SLOP_PX) {
            down.moved = true;
        }
        if (dragRef) {
            e.preventDefault();
            const hit = pick(e.clientX, e.clientY);
            if (hit) {
                model.moveDrag(dragRef, hit.x, hit.y);
            }
            updateReadout(e, hit);
            return;
        }
        if (!hoverFrame) {
            hoverFrame = requestAnimationFrame(() => {
                hoverFrame = 0;
                if (!lastMove || !editable()) {
                    readout.hidden = true;
                    return;
                }
                const onUi = !(lastMove.target === canvas || overlay.contains(lastMove.target));
                updateReadout(lastMove, onUi ? null : pick(lastMove.clientX, lastMove.clientY), onUi);
            });
        }
    });

    box.addEventListener("pointerleave", () => { readout.hidden = true; });

    function updateReadout(e, hit, hide = false) {
        if (hide || !editable()) {
            readout.hidden = true;
            return;
        }
        const r = box.getBoundingClientRect();
        readout.hidden = false;
        readout.classList.toggle("is-off", !hit);
        readout.textContent = hit
            ? `X ${hit.x.toFixed(1)} · Y ${hit.y.toFixed(1)} px · ${Math.round(geo.elevation(hit.x, hit.y)).toLocaleString()} m`
            : "Off terrain";
        readout.style.left = `${e.clientX - r.left + 16}px`;
        readout.style.top = `${e.clientY - r.top + 16}px`;
    }

    window.addEventListener("pointerup", e => {
        if (!down) {
            return;
        }
        const d = down;
        down = null;
        if (dragRef) {
            const ref = dragRef;
            dragRef = null;
            viewer.controls.enabled = true;
            box.releasePointerCapture?.(e.pointerId);
            const r = model.endDrag(ref);
            dirty = true;
            if (!d.moved && r.type === "unchanged") {
                // A click (no drag) on an existing point.
                handleOutcome(model.click(pick(e.clientX, e.clientY), ref));
            }
            return;
        }
        if (d.moved || e.button !== 0 || !editable()) {
            return; // an orbit, not a click
        }
        handleOutcome(model.click(pick(e.clientX, e.clientY), null));
    });

    box.addEventListener("contextmenu", e => {
        if (!box.classList.contains("is-expanded")) {
            return;
        }
        const ref = refOf(e.target);
        if (ref && editable()) {
            e.preventDefault();
            openMenu(ref.kind, ref, e.clientX, e.clientY);
        } else if (editable() && (e.target === canvas || overlay.contains(e.target))) {
            e.preventDefault(); // no browser menu over the terrain while measuring
        }
    });

    document.addEventListener("pointerdown", e => {
        if (!menu.hidden && !menu.contains(e.target) && !box.contains(e.target)) {
            closeMenu();
        }
    });

    // ---------------------------------------------------------------- keyboard
    document.addEventListener("keydown", e => {
        if (!box.classList.contains("is-expanded")) {
            return;
        }
        if (document.querySelector(".confirm-modal:not([hidden])")) {
            return; // the Close confirmation owns the keyboard
        }
        const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
        if (e.key === "Escape") {
            if (document.querySelector("[data-xv-popover]:not([hidden])")) {
                return; // an open chrome popover/dialog closes first (handled there)
            }
            if (!menu.hidden) {
                closeMenu();
            } else if (model.mode !== MODES.NORMAL && !typing) {
                if (dragRef) {
                    return;
                }
                const r = model.clear();
                if (r.type === "cleared") {
                    showToast("Selection cleared (Undo brings it back).", "info");
                }
            }
            e.preventDefault();
            return;
        }
        // Cmd/Ctrl+Z is the viewer-wide undo now (src/viewer-history.js, wired in main.js).
    });

    // ---------------------------------------------------------------- per frame
    let lastCam = "";
    function update() {
        if (!syncTerrain()) {
            return;
        }
        // Redraw when the selection, the camera, the canvas size, or the
        // displayed surface (flat/3D) changed.
        const e = viewer.camera.matrixWorld.elements;
        const cam = `${e[12].toFixed(3)},${e[13].toFixed(3)},${e[14].toFixed(3)},${e[8].toFixed(4)},${e[9].toFixed(4)},${e[10].toFixed(4)},${canvas.clientWidth}x${canvas.clientHeight},${viewer.camera.zoom}`;
        if (cam !== lastCam) {
            lastCam = cam;
            dirty = true;
        }
        if (dirty) {
            overlay.setAttribute("viewBox", `0 0 ${canvas.clientWidth} ${canvas.clientHeight}`);
            render();
        }
    }

    renderToolbar();

    // Chrome hooks (expanded-view toolbar, library, context menu, notes).
    function screenPath(p0, p1, lift = LINE_LIFT) {
        // Draped screen polyline between two grid points; null entries = gaps.
        const n = Math.max(1, Math.ceil(Math.hypot(p1.x - p0.x, p1.y - p0.y) * 2));
        const out = [];
        for (let i = 0; i <= n; i++) {
            const t = i / n;
            out.push(syncTerrain() ? project(p0.x + (p1.x - p0.x) * t, p0.y + (p1.y - p0.y) * t, lift) : null);
        }
        return out;
    }

    function saveSelection() {
        return model.save((pts, closed) => measureChain(geo, pts, closed));
    }

    function clearSelection() {
        const r = model.clear();
        if (r.type === "cleared") {
            showToast("Selection cleared (Undo brings it back).", "info");
        }
        return r;
    }

    // Removes the selection point nearest to a grid position (undoable).
    function deleteNearestPoint(x, y) {
        let best = null;
        model.chains.forEach((c, ci) => c.points.forEach((p, pi) => {
            const d = Math.hypot(p.x - x, p.y - y);
            if (!best || d < best.d) {
                best = { chain: ci, index: pi, d };
            }
        }));
        if (!best) {
            return null;
        }
        const r = model.deletePoint({ chain: best.chain, index: best.index });
        showToast(`Removed point ${best.index + 1} from the selection (Undo brings it back).`, "info");
        return r;
    }

    // "Start selection" from the terrain context menu: enter a measure mode
    // and place the first point where the user right-clicked.
    function startSelectionAt(mode, hit) {
        setMode(mode);
        handleOutcome(model.click(hit, null));
    }

    return {
        model,
        update,
        setMode,
        pick,
        showToast,
        screenPath,
        saveSelection,
        clearSelection,
        deleteNearestPoint,
        startSelectionAt,
        invalidate: () => { dirty = true; },
        onRender: fn => renderListeners.push(fn),
        get geo() {
            return geo;
        },
        get lineLift() {
            return LINE_LIFT;
        },
        // Test/diagnostic hooks (read-only views).
        get saved() {
            return model.saved;
        },
        metrics: selectionMetrics,
        project: (x, y) => (syncTerrain() ? project(x, y, LINE_LIFT) : null),
    };
}
