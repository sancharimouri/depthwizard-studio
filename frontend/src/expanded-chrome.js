// Chrome around Workbench's expanded 3D view: icon toolbar, Save + Library
// (points / lines / areas / screenshots / recordings), terrain right-click menu
// (named points, notes, selection shortcuts), note pins, screenshot capture,
// play/pause of the idle auto-rotation, and the dark/bright theme toggle.
//
// ALL saved state here is SESSION-ONLY (in memory): there is no backend
// storage, so everything is gone after a reload or Close. The UI says so.
// Video recording is an explicit placeholder (disabled), not implemented.
//
// Every element this module adds carries [data-xv-ui] so the measurement
// tool ignores clicks on it; open popovers/dialogs carry [data-xv-popover]
// so they get Esc before the measurement tool does.

import { MODES } from "./measure-model.js";

const SVG_NS = "http://www.w3.org/2000/svg";

// Outline icons (24×24, stroke = currentColor).
const ICONS = {
    camera: '<path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/><circle cx="12" cy="13.5" r="3.5"/>',
    video: '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10.5l5-3v9l-5-3z"/>',
    ruler: '<path d="M3.5 16.5 16.5 3.5l4 4-13 13z"/><path d="M7.5 12.5l2 2M10.5 9.5l1.5 1.5M13.5 6.5l2 2"/>',
    layers: '<path d="M12 3 2 8l10 5 10-5z"/><path d="M2 12.5l10 5 10-5"/><path d="M2 17l10 5 10-5"/>',
    plane: '<path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/>',
    save: '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/>',
    chevronDown: '<path d="M6 9l6 6 6-6"/>',
    chevronRight: '<path d="M9 6l6 6-6 6"/>',
    cursor: '<path d="M4 3l7.5 17 2.3-7.2L21 10.5z"/>',
    notes: '<path d="M4 4h16v10l-6 6H4z"/><path d="M14 20v-6h6"/><path d="M8 9h8M8 13h4"/>',
    trash: '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M6 6l1 14h10l1-14"/><path d="M10 11v6M14 11v6"/>',
    play: '<path d="M7 4.5v15l12.5-7.5z"/>',
    pause: '<rect x="6" y="4.5" width="4" height="15" rx="1"/><rect x="14" y="4.5" width="4" height="15" rx="1"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8 6 18M18 6l1.8-1.8"/>',
    moon: '<path d="M20.5 13.2A8.5 8.5 0 1 1 10.8 3.5a6.8 6.8 0 0 0 9.7 9.7z"/>',
    pin: '<path d="M12 21s-7-6.3-7-11a7 7 0 0 1 14 0c0 4.7-7 11-7 11z"/><circle cx="12" cy="10" r="2.5"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    download: '<path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M5 21h14"/>',
    point: '<path d="M12 21s-6-5.6-6-10a6 6 0 0 1 12 0c0 4.4-6 10-6 10z"/><circle cx="12" cy="11" r="2"/>',
    line: '<path d="M5 19 19 5"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="5" r="2"/>',
    area: '<path d="M4 8l8-5 8 6-3 11H7z"/>',
    image: '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-8 8"/>',
};

const icon = (name, size = 22) =>
    `<svg xmlns="${SVG_NS}" viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;

const CATEGORIES = [
    { key: "points", label: "Points", icon: "point", single: "point" },
    { key: "lines", label: "Lines", icon: "line", single: "line" },
    { key: "areas", label: "Areas", icon: "area", single: "area" },
    { key: "screenshots", label: "Screenshots", icon: "image", single: "screenshot" },
    { key: "recordings", label: "Recordings", icon: "video", single: "recording" },
];

const fmtM = m => (Math.abs(m) >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(0)} m`);
const fmtArea = a => (a >= 1e6 ? `${(a / 1e6).toFixed(3)} km²` : `${Math.round(a).toLocaleString()} m²`);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function el(tag, attrs = {}, parent = null, ns = null) {
    const node = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "text") {
            node.textContent = v;
        } else if (k === "html") {
            node.innerHTML = v;
        } else {
            node.setAttribute(k, v);
        }
    }
    parent?.appendChild(node);
    return node;
}

// An icon-only button whose name shows as a hover/focus tooltip.
function iconButton(parent, { name, iconName, cls = "", tipBelow = false, disabled = false }) {
    const b = el("button", {
        type: "button",
        class: `xv-icon-btn ${cls}${tipBelow ? " tip-below" : ""}`,
        "aria-label": name,
        "data-tip": name,
        html: icon(iconName),
    }, parent);
    if (disabled) {
        b.setAttribute("aria-disabled", "true");
        b.classList.add("is-disabled");
    }
    return b;
}

export function createExpandedChrome({
    box, canvas, viewer, tool, getRegionKey, setLayer, getLayer, startFlythrough, getTheme, setTheme,
}) {
    // ------------------------------------------------------------------ state (session-only)
    const library = { points: [], lines: [], areas: [], screenshots: [], recordings: [] };
    const notes = [];
    let nextId = 1;
    let openCategory = null;

    // ------------------------------------------------------------------ overlay (saved items + pins)
    const overlay = el("svg", { class: "xv-overlay", "data-xv-ui": "" }, box, SVG_NS);
    const gSaved = el("g", {}, overlay, SVG_NS);
    const gPins = el("g", {}, overlay, SVG_NS);

    function pathD(points) {
        let d = "";
        let pen = false;
        for (const q of points) {
            if (!q) {
                pen = false;
                continue;
            }
            d += `${pen ? "L" : "M"}${q.sx.toFixed(1)},${q.sy.toFixed(1)}`;
            pen = true;
        }
        return d;
    }

    function drawOverlay() {
        if (!tool.geo) {
            return;
        }
        overlay.setAttribute("viewBox", `0 0 ${canvas.clientWidth} ${canvas.clientHeight}`);
        gSaved.replaceChildren();
        for (const cat of ["lines", "areas"]) {
            for (const item of library[cat]) {
                if (!item.visible) {
                    continue;
                }
                const pts = item.points;
                const n = item.closed ? pts.length : pts.length - 1;
                let d = "";
                for (let i = 0; i < n; i++) {
                    d += pathD(tool.screenPath(pts[i], pts[(i + 1) % pts.length]));
                }
                if (item.closed) {
                    const ring = pts.map(p => tool.screenPath(p, p)[0]).filter(Boolean);
                    if (ring.length >= 3) {
                        el("path", { class: "xv-saved-fill", d: `M${ring.map(q => `${q.sx},${q.sy}`).join("L")}Z` }, gSaved, SVG_NS);
                    }
                }
                el("path", { class: "xv-saved-line", d }, gSaved, SVG_NS);
                pts.forEach(p => {
                    const q = tool.screenPath(p, p)[0];
                    if (q) {
                        el("circle", { class: "xv-saved-dot", cx: q.sx, cy: q.sy, r: 3.5 }, gSaved, SVG_NS);
                    }
                });
            }
        }
        gPins.replaceChildren();
        for (const p of library.points) {
            if (!p.visible) {
                continue;
            }
            const q = tool.screenPath(p, p)[0];
            if (!q) {
                continue;
            }
            const g = el("g", { class: "xv-point-marker", transform: `translate(${q.sx.toFixed(1)},${q.sy.toFixed(1)})` }, gPins, SVG_NS);
            el("path", { d: "M0,-9 L7,0 L0,9 L-7,0 Z" }, g, SVG_NS);
            el("text", { x: 11, y: 4, text: p.name }, g, SVG_NS);
        }
        for (const note of notes) {
            if (!note.visible) {
                continue;
            }
            const q = tool.screenPath(note, note)[0];
            if (!q) {
                continue;
            }
            const g = el("g", {
                class: "xv-note-pin", "data-note": note.id, transform: `translate(${q.sx.toFixed(1)},${q.sy.toFixed(1)})`,
                role: "button", tabindex: "0", "aria-label": "Open note",
            }, gPins, SVG_NS);
            el("title", { text: "Open note" }, g, SVG_NS);
            el("circle", { class: "xv-pin-hit", cx: 0, cy: -14, r: 14 }, g, SVG_NS);
            el("path", { class: "xv-pin-shape", d: "M0,0 C-7,-8 -9,-12 -9,-17 A9,9 0 1 1 9,-17 C9,-12 7,-8 0,0 Z" }, g, SVG_NS);
            el("circle", { class: "xv-pin-eye", cx: 0, cy: -17, r: 3 }, g, SVG_NS);
        }
        if (notePopup.dataset.note) {
            placeNotePopup();
        }
    }

    tool.onRender(drawOverlay);
    const invalidate = () => tool.invalidate();

    // ------------------------------------------------------------------ toast (reuse the tool's)
    const toast = (msg, kind = "info") => tool.showToast(msg, kind);

    // ------------------------------------------------------------------ top controls
    const playBtn = iconButton(box, { name: "Pause rotation", iconName: "pause", cls: "xv-play", tipBelow: true });
    playBtn.dataset.xvUi = "";
    const themeBtn = iconButton(box, { name: "Switch to bright mode", iconName: "sun", cls: "xv-theme", tipBelow: true });
    themeBtn.dataset.xvUi = "";

    function syncPlay() {
        const paused = viewer.controls.isAutoRotatePaused?.() ?? false;
        playBtn.innerHTML = icon(paused ? "play" : "pause");
        const name = paused ? "Play rotation" : "Pause rotation";
        playBtn.setAttribute("aria-label", name);
        playBtn.dataset.tip = name;
        playBtn.setAttribute("aria-pressed", String(paused));
    }
    playBtn.addEventListener("click", () => {
        viewer.controls.setAutoRotatePaused?.(!(viewer.controls.isAutoRotatePaused?.() ?? false));
        syncPlay();
    });

    // The icon is the mode you'd switch TO: sun while dark, moon while bright.
    function syncTheme() {
        const dark = getTheme() === "dark";
        themeBtn.innerHTML = icon(dark ? "sun" : "moon");
        const name = dark ? "Switch to bright mode" : "Switch to dark mode";
        themeBtn.setAttribute("aria-label", name);
        themeBtn.dataset.tip = name;
    }
    themeBtn.addEventListener("click", () => {
        setTheme(getTheme() === "dark" ? "light" : "dark");
        syncTheme();
    });

    // ------------------------------------------------------------------ toolbar (reference order)
    const toolbar = el("div", { class: "xv-toolbar", role: "toolbar", "aria-label": "3D view tools", "data-xv-ui": "" }, box);
    const shotBtn = iconButton(toolbar, { name: "Screenshot", iconName: "camera" });
    const recBtn = iconButton(toolbar, { name: "Record video — not yet available", iconName: "video", disabled: true });
    const measureWrap = el("div", { class: "xv-wrap" }, toolbar);
    const measureBtn = iconButton(measureWrap, { name: "Measure (A to B · 2 points / continuous)", iconName: "ruler" });
    const viewWrap = el("div", { class: "xv-wrap" }, toolbar);
    const viewBtn = iconButton(viewWrap, { name: "View (true colour / DSM / DEM)", iconName: "layers" });
    const flyBtn = iconButton(toolbar, { name: "Flythrough", iconName: "plane" });
    const saveWrap = el("div", { class: "xv-wrap xv-split" }, toolbar);
    const saveBtn = iconButton(saveWrap, { name: "Save current measurement", iconName: "save" });
    const libBtn = iconButton(saveWrap, { name: "Saved items (library)", iconName: "chevronDown", cls: "xv-caret" });
    const pointerBtn = iconButton(toolbar, { name: "Mouse pointer (navigate)", iconName: "cursor" });
    const notesWrap = el("div", { class: "xv-wrap" }, toolbar);
    const notesBtn = iconButton(notesWrap, { name: "Notes", iconName: "notes", cls: "xv-notes-count" });
    const trashBtn = iconButton(toolbar, { name: "Clear current selection", iconName: "trash" });

    // popovers
    const popovers = [];
    function popover(parent, cls) {
        const p = el("div", { class: `xv-popover ${cls}`, "data-xv-popover": "", "data-xv-ui": "", hidden: "" }, parent);
        popovers.push(p);
        return p;
    }
    function closeAll(except = null) {
        for (const p of popovers) {
            if (p !== except && !p.contains(except)) {
                p.hidden = true;
            }
        }
        for (const b of [measureBtn, viewBtn, libBtn, notesBtn]) {
            b.setAttribute("aria-expanded", "false");
        }
        if (!except || !libraryPanel.contains(except)) {
            openCategory = null;
        }
    }
    function toggle(p, btn) {
        const open = p.hidden;
        closeAll();
        p.hidden = !open;
        btn?.setAttribute("aria-expanded", String(open));
        return open;
    }

    const measureMenu = popover(measureWrap, "xv-menu xv-above");
    const optTwo = el("button", { type: "button", role: "menuitemradio", text: "Two points (A → B)" }, measureMenu);
    const optCont = el("button", { type: "button", role: "menuitemradio", text: "Continuous (chain / shape)" }, measureMenu);
    measureBtn.addEventListener("click", () => toggle(measureMenu, measureBtn));
    optTwo.addEventListener("click", () => { closeAll(); tool.setMode(MODES.TWO_POINT); });
    optCont.addEventListener("click", () => { closeAll(); tool.setMode(MODES.CONTINUOUS); });

    const viewMenu = popover(viewWrap, "xv-menu xv-above");
    for (const [layer, label] of [["satellite-3d", "True colour"], ["dsm-3d", "DSM (relative depth)"], ["elevation-3d", "DEM elevation"], ["wireframe-3d", "Wireframe"]]) {
        const b = el("button", { type: "button", role: "menuitemradio", "data-layer": layer, text: label }, viewMenu);
        b.addEventListener("click", () => { closeAll(); setLayer(layer); syncToolbar(); });
    }
    viewBtn.addEventListener("click", () => { syncToolbar(); toggle(viewMenu, viewBtn); });

    flyBtn.addEventListener("click", () => { closeAll(); startFlythrough(); });
    pointerBtn.addEventListener("click", () => { closeAll(); tool.setMode(MODES.NORMAL); });
    trashBtn.addEventListener("click", () => {
        closeAll();
        if (tool.model.pointCount() === 0) {
            toast("Nothing selected to clear. Saved items are deleted from the library (▾ beside Save).");
            return;
        }
        tool.clearSelection();
    });

    // ------------------------------------------------------------------ notes (▾ list, same pattern as Save's library)
    const notesPanel = popover(notesWrap, "xv-library xv-above xv-notes-panel");
    notesBtn.setAttribute("aria-haspopup", "true");
    notesBtn.addEventListener("click", () => {
        if (toggle(notesPanel, notesBtn)) {
            renderNotesList();
        }
    });

    async function deleteAllNotes() {
        const n = notes.length;
        if (!n) {
            return;
        }
        const ok = await confirmAction({
            title: "Delete all notes?",
            body: `This deletes ${n} note${n === 1 ? "" : "s"} from this session. It can't be undone.`,
            okLabel: `Delete ${n}`,
        });
        if (!ok) {
            return;
        }
        notes.length = 0;
        closeNotePopup();
        toast("Deleted all notes.");
        renderNotesList();
        syncToolbar();
        invalidate();
    }

    function renderNotesList() {
        syncToolbar();
        notesPanel.replaceChildren();
        const head = el("div", { class: "xv-lib-head" }, notesPanel);
        el("div", { class: "xv-lib-title", text: "Notes" }, head);
        const allShown = notes.length && notes.every(n => n.visible);
        const eye = el("button", {
            type: "button", class: "xv-text-btn xv-sm-text",
            text: allShown ? "Hide all" : "Show all",
        }, head);
        eye.disabled = !notes.length;
        eye.addEventListener("click", () => {
            const show = !(notes.length && notes.every(n => n.visible));
            notes.forEach(n => { n.visible = show; });
            if (!show) {
                closeNotePopup();
            }
            renderNotesList();
            invalidate();
        });
        const allTrash = el("button", {
            type: "button", class: "xv-icon-btn xv-sm xv-danger-hover", "aria-label": "Delete all notes", "data-tip": "Delete all notes",
            html: icon("trash", 18),
        }, head);
        allTrash.disabled = !notes.length;
        allTrash.addEventListener("click", deleteAllNotes);

        if (!notes.length) {
            el("div", { class: "xv-empty", text: "No notes yet. Right-click the terrain in pointer mode and choose “Add note”." }, notesPanel);
        } else {
            const ul = el("ul", { class: "xv-fly-list" }, notesPanel);
            notes.forEach((note, i) => {
                const li = el("li", { class: "xv-fly-item" }, ul);
                const lab = el("label", { class: "xv-switch", "data-tip": note.visible ? "Hide on terrain" : "Show on terrain" }, li);
                const cb = el("input", { type: "checkbox", "aria-label": `Show note ${i + 1} on the terrain` }, lab);
                cb.checked = note.visible;
                el("span", { class: "xv-switch-track" }, lab);
                cb.addEventListener("change", () => {
                    note.visible = cb.checked;
                    lab.dataset.tip = note.visible ? "Hide on terrain" : "Show on terrain";
                    if (!note.visible && notePopup.dataset.note === String(note.id)) {
                        closeNotePopup();
                    }
                    syncToolbar();
                    invalidate();
                });
                const text = el("div", { class: "xv-fly-text" }, li);
                el("div", { class: "xv-fly-name", text: note.text.length > 48 ? `${note.text.slice(0, 47)}…` : note.text }, text);
                el("div", {
                    class: "xv-fly-meta",
                    text: `Note ${i + 1} · X ${note.x.toFixed(1)} · Y ${note.y.toFixed(1)} · ${note.createdAt.toLocaleTimeString()}`,
                }, text);
                const del = el("button", {
                    type: "button", class: "xv-icon-btn xv-sm xv-danger-hover", "aria-label": `Delete note ${i + 1}`,
                    "data-tip": "Delete", html: icon("trash", 16),
                }, li);
                del.addEventListener("click", () => {
                    const at = notes.indexOf(note);
                    if (at >= 0) {
                        notes.splice(at, 1);
                    }
                    if (notePopup.dataset.note === String(note.id)) {
                        closeNotePopup();
                    }
                    toast("Note deleted.");
                    renderNotesList();
                    syncToolbar();
                    invalidate();
                });
            });
        }
        el("div", {
            class: "xv-lib-note",
            text: "Session only: notes are kept in this browser tab and are lost on reload or Close.",
        }, notesPanel);
    }

    saveBtn.addEventListener("click", () => {
        closeAll();
        const r = tool.saveSelection();
        if (r.type !== "saved") {
            toast(r.message, "error");
            return;
        }
        for (const item of r.items) {
            const cat = item.closed ? "areas" : "lines";
            library[cat].push({
                id: nextId++,
                name: `${cat === "areas" ? "Area" : "Line"} ${library[cat].length + 1}`,
                points: item.points.map(p => ({ x: p.x, y: p.y })),
                closed: item.closed,
                metrics: item.metrics,
                region: getRegionKey(),
                createdAt: new Date(),
                visible: true,
            });
        }
        toast(`Saved to Library → ${r.items.every(i => i.closed) ? "Areas" : r.items.some(i => i.closed) ? "Lines & Areas" : "Lines"} (session only — not kept after reload).`);
        renderLibrary();
        invalidate();
    });

    function syncToolbar() {
        const mode = tool.model.mode;
        measureBtn.classList.toggle("is-active", mode !== MODES.NORMAL);
        pointerBtn.classList.toggle("is-active", mode === MODES.NORMAL);
        pointerBtn.setAttribute("aria-pressed", String(mode === MODES.NORMAL));
        optTwo.setAttribute("aria-checked", String(mode === MODES.TWO_POINT));
        optCont.setAttribute("aria-checked", String(mode === MODES.CONTINUOUS));
        const layer = getLayer();
        viewMenu.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.layer === layer)));
        const canSave = tool.model.chains.some(c => c.points.length >= 2);
        saveBtn.classList.toggle("is-muted", !canSave);
        const total = CATEGORIES.reduce((n, c) => n + library[c.key].length, 0);
        libBtn.dataset.count = total ? String(total) : "";
        const shown = notes.length && notes.every(n => n.visible);
        notesBtn.classList.toggle("is-active", !!shown);
        notesBtn.dataset.count = notes.length ? String(notes.length) : "";
        const nName = notes.length ? `Notes (${notes.length})` : "Notes (none yet)";
        notesBtn.setAttribute("aria-label", nName);
        notesBtn.dataset.tip = nName;
    }
    tool.onRender(syncToolbar);

    // ------------------------------------------------------------------ confirmation dialog
    const confirmModal = el("div", {
        class: "upload-modal confirm-modal xv-confirm", role: "alertdialog", "aria-modal": "true",
        "aria-labelledby": "xv-confirm-title", "aria-describedby": "xv-confirm-body",
        "data-xv-popover": "", "data-xv-ui": "", hidden: "",
    }, box);
    const confirmBackdrop = el("div", { class: "upload-backdrop" }, confirmModal);
    const confirmPanel = el("div", { class: "upload-panel confirm-panel" }, confirmModal);
    const confirmTitle = el("div", { id: "xv-confirm-title", class: "confirm-title" }, confirmPanel);
    const confirmBody = el("p", { id: "xv-confirm-body", class: "confirm-body" }, confirmPanel);
    const confirmActions = el("div", { class: "confirm-actions" }, confirmPanel);
    const confirmCancel = el("button", { type: "button", class: "confirm-button", text: "Cancel" }, confirmActions);
    const confirmOk = el("button", { type: "button", class: "confirm-button confirm-danger", text: "Delete" }, confirmActions);
    let confirmResolve = null;

    function confirmAction({ title, body, okLabel = "Delete" }) {
        confirmTitle.textContent = title;
        confirmBody.textContent = body;
        confirmOk.textContent = okLabel;
        confirmModal.hidden = false;
        confirmCancel.focus();
        return new Promise(resolve => { confirmResolve = resolve; });
    }
    function settleConfirm(value) {
        confirmModal.hidden = true;
        confirmResolve?.(value);
        confirmResolve = null;
    }
    confirmCancel.addEventListener("click", () => settleConfirm(false));
    confirmBackdrop.addEventListener("click", () => settleConfirm(false));
    confirmOk.addEventListener("click", () => settleConfirm(true));

    // ------------------------------------------------------------------ library (Save ▾)
    const libraryPanel = popover(saveWrap, "xv-library xv-above");
    const flyout = el("div", { class: "xv-flyout", hidden: "" }, libraryPanel);
    libBtn.addEventListener("click", () => {
        if (toggle(libraryPanel, libBtn)) {
            renderLibrary();
        }
    });

    function itemMeta(cat, item) {
        const m = item.metrics;
        if (cat === "points") {
            return `${item.tag ? `#${esc(item.tag)} · ` : ""}X ${item.x.toFixed(1)} · Y ${item.y.toFixed(1)} · ${Math.round(item.elevation).toLocaleString()} m`;
        }
        if (cat === "lines" && m) {
            return `${fmtM(m.horizontal)} · rise ${m.netRise >= 0 ? "+" : "−"}${fmtM(Math.abs(m.netRise))} · ${m.points.length} pts`;
        }
        if (cat === "areas" && m) {
            return `${fmtArea(m.area)} · perimeter ${fmtM(m.perimeter)}`;
        }
        if (cat === "screenshots") {
            return `${item.width}×${item.height} px · ${item.region} · ${item.createdAt.toLocaleTimeString()}`;
        }
        return "";
    }

    async function deleteCategory(cat) {
        const c = CATEGORIES.find(x => x.key === cat);
        const n = library[cat].length;
        if (!n) {
            return;
        }
        const ok = await confirmAction({
            title: `Delete all ${c.label.toLowerCase()}?`,
            body: `This deletes ${n} saved ${n === 1 ? c.single : c.label.toLowerCase()} from this session. It can't be undone.`,
            okLabel: `Delete ${n}`,
        });
        if (!ok) {
            return;
        }
        if (cat === "screenshots") {
            library.screenshots.forEach(s => URL.revokeObjectURL(s.url));
        }
        library[cat].length = 0;
        toast(`Deleted all ${c.label.toLowerCase()}.`);
        renderLibrary();
        invalidate();
    }

    async function deleteAll() {
        const total = CATEGORIES.reduce((n, c) => n + library[c.key].length, 0);
        if (!total) {
            return;
        }
        const parts = CATEGORIES.filter(c => library[c.key].length).map(c => `${library[c.key].length} ${c.label.toLowerCase()}`);
        const ok = await confirmAction({
            title: "Delete every saved item?",
            body: `This deletes all ${total} saved items across every category (${parts.join(", ")}). It can't be undone.`,
            okLabel: "Delete everything",
        });
        if (!ok) {
            return;
        }
        library.screenshots.forEach(s => URL.revokeObjectURL(s.url));
        for (const c of CATEGORIES) {
            library[c.key].length = 0;
        }
        toast("Deleted all saved items.");
        renderLibrary();
        invalidate();
    }

    function renderLibrary() {
        syncToolbar();
        const keepFlyout = openCategory;
        libraryPanel.querySelectorAll(":scope > :not(.xv-flyout)").forEach(n => n.remove());
        const head = el("div", { class: "xv-lib-head" });
        libraryPanel.prepend(head);
        el("div", { class: "xv-lib-title", text: "Saved items" }, head);
        const total = CATEGORIES.reduce((n, c) => n + library[c.key].length, 0);
        const allTrash = el("button", {
            type: "button", class: "xv-icon-btn xv-sm xv-danger-hover", "aria-label": "Delete all saved items", "data-tip": "Delete all saved items",
            html: icon("trash", 18),
        }, head);
        if (!total) {
            allTrash.disabled = true;
        }
        allTrash.addEventListener("click", deleteAll);
        const list = el("div", { class: "xv-lib-list", role: "list" });
        head.after(list);
        for (const c of CATEGORIES) {
            const n = library[c.key].length;
            const row = el("div", { class: `xv-lib-row${keepFlyout === c.key ? " is-open" : ""}`, role: "listitem" }, list);
            el("span", { class: "xv-lib-icon", html: icon(c.icon, 18) }, row);
            el("span", { class: "xv-lib-label", text: c.label }, row);
            el("span", { class: "xv-lib-count", text: String(n) }, row);
            const del = el("button", {
                type: "button", class: "xv-icon-btn xv-sm xv-danger-hover", "aria-label": `Delete all ${c.label.toLowerCase()}`,
                "data-tip": `Delete all ${c.label.toLowerCase()}`, html: icon("trash", 16),
            }, row);
            if (!n) {
                del.disabled = true;
            }
            del.addEventListener("click", () => deleteCategory(c.key));
            const open = el("button", {
                type: "button", class: "xv-icon-btn xv-sm", "aria-label": `Show ${c.label.toLowerCase()}`,
                "data-tip": `Open ${c.label.toLowerCase()}`, "aria-expanded": String(keepFlyout === c.key), html: icon("chevronRight", 16),
            }, row);
            open.addEventListener("click", () => {
                openCategory = openCategory === c.key ? null : c.key;
                renderLibrary();
            });
        }
        el("div", {
            class: "xv-lib-note",
            text: "Session only: saved items are kept in this browser tab and are lost on reload or Close. Permanent storage isn't built yet.",
        }, libraryPanel).before(flyout);
        renderFlyout();
    }

    function renderFlyout() {
        flyout.replaceChildren();
        if (!openCategory) {
            flyout.hidden = true;
            return;
        }
        flyout.hidden = false;
        const c = CATEGORIES.find(x => x.key === openCategory);
        const head = el("div", { class: "xv-fly-head" }, flyout);
        el("span", { html: icon(c.icon, 16) }, head);
        el("span", { text: c.label }, head);
        const items = library[c.key];
        if (c.key === "recordings") {
            el("div", {
                class: "xv-empty",
                text: "Video recording isn't available yet — it needs capture and encoding support that doesn't exist in this project. Use Screenshot for now.",
            }, flyout);
            const b = el("button", { type: "button", class: "xv-text-btn", disabled: "", text: "Start recording (not yet available)" }, flyout);
            b.setAttribute("aria-disabled", "true");
            return;
        }
        if (!items.length) {
            const how = {
                points: "Right-click the terrain in pointer mode and choose “Save point”.",
                lines: "Measure two points or an open chain, then click Save.",
                areas: "Close a continuous chain into a shape, then click Save.",
                screenshots: "Click the camera button to capture the view.",
            }[c.key];
            el("div", { class: "xv-empty", text: `Nothing saved yet. ${how}` }, flyout);
            return;
        }
        const ul = el("ul", { class: "xv-fly-list" }, flyout);
        for (const item of items) {
            const li = el("li", { class: "xv-fly-item" }, ul);
            if (c.key === "screenshots") {
                const thumb = el("button", { type: "button", class: "xv-thumb", "aria-label": `Preview ${item.name}` }, li);
                el("img", { src: item.url, alt: item.name }, thumb);
                thumb.addEventListener("click", () => openPreview(item));
            } else {
                const lab = el("label", { class: "xv-switch", "data-tip": item.visible ? "Hide on terrain" : "Show on terrain" }, li);
                const cb = el("input", { type: "checkbox", "aria-label": `Show ${item.name} on the terrain` }, lab);
                cb.checked = item.visible;
                el("span", { class: "xv-switch-track" }, lab);
                cb.addEventListener("change", () => {
                    item.visible = cb.checked;
                    lab.dataset.tip = item.visible ? "Hide on terrain" : "Show on terrain";
                    invalidate();
                });
            }
            const text = el("div", { class: "xv-fly-text" }, li);
            el("div", { class: "xv-fly-name", text: item.name }, text);
            el("div", { class: "xv-fly-meta", html: itemMeta(c.key, item) }, text);
            if (c.key === "screenshots") {
                const dl = el("a", {
                    class: "xv-icon-btn xv-sm", href: item.url, download: item.fileName,
                    "aria-label": `Download ${item.name}`, "data-tip": "Download PNG", html: icon("download", 16),
                }, li);
                dl.addEventListener("click", e => e.stopPropagation());
            }
            const del = el("button", {
                type: "button", class: "xv-icon-btn xv-sm xv-danger-hover", "aria-label": `Delete ${item.name}`,
                "data-tip": "Delete", html: icon("trash", 16),
            }, li);
            del.addEventListener("click", () => {
                const i = library[c.key].indexOf(item);
                if (i >= 0) {
                    library[c.key].splice(i, 1);
                    if (c.key === "screenshots") {
                        URL.revokeObjectURL(item.url);
                    }
                }
                toast(`Deleted ${item.name}.`);
                renderLibrary();
                invalidate();
            });
        }
    }

    // ------------------------------------------------------------------ screenshot (real, client-side)
    const preview = el("div", { class: "upload-modal xv-preview", "data-xv-popover": "", "data-xv-ui": "", hidden: "", role: "dialog", "aria-modal": "true", "aria-label": "Screenshot preview" }, box);
    const previewBackdrop = el("div", { class: "upload-backdrop" }, preview);
    const previewPanel = el("div", { class: "xv-preview-panel" }, preview);
    function openPreview(item) {
        previewPanel.replaceChildren();
        const head = el("div", { class: "xv-preview-head" }, previewPanel);
        el("div", { class: "xv-fly-name", text: `${item.name} · ${item.width}×${item.height} px` }, head);
        el("a", { class: "xv-text-btn", href: item.url, download: item.fileName, html: `${icon("download", 16)} Download PNG` }, head);
        const close = el("button", { type: "button", class: "xv-icon-btn xv-sm", "aria-label": "Close preview", "data-tip": "Close", html: icon("x", 18) }, head);
        close.addEventListener("click", () => { preview.hidden = true; });
        el("img", { src: item.url, alt: item.name }, previewPanel);
        preview.hidden = false;
        close.focus();
    }
    previewBackdrop.addEventListener("click", () => { preview.hidden = true; });

    function drawSelectionOnto(ctx, scale) {
        // Same visuals as the live overlay: measurement selection (red),
        // visible saved lines/areas (cyan), saved points, visible note pins.
        const stroke = (pts, color, width, glow) => {
            ctx.save();
            ctx.strokeStyle = color;
            ctx.lineWidth = width * scale;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            if (glow) {
                ctx.shadowColor = glow;
                ctx.shadowBlur = 6 * scale;
            }
            ctx.beginPath();
            let pen = false;
            for (const q of pts) {
                if (!q) {
                    pen = false;
                    continue;
                }
                pen ? ctx.lineTo(q.sx * scale, q.sy * scale) : ctx.moveTo(q.sx * scale, q.sy * scale);
                pen = true;
            }
            ctx.stroke();
            ctx.restore();
        };
        const dot = (q, fill, r) => {
            if (!q) {
                return;
            }
            ctx.save();
            ctx.fillStyle = fill;
            ctx.strokeStyle = "rgba(255,255,255,0.9)";
            ctx.lineWidth = 1.5 * scale;
            ctx.beginPath();
            ctx.arc(q.sx * scale, q.sy * scale, r * scale, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        };
        const chainsOf = src => src.map(c => ({ points: c.points, closed: c.closed }));
        const draw = (chains, color, glow, dotColor) => {
            for (const c of chains) {
                const n = c.closed ? c.points.length : c.points.length - 1;
                for (let i = 0; i < n; i++) {
                    stroke(tool.screenPath(c.points[i], c.points[(i + 1) % c.points.length]), color, 1.75, glow);
                }
                c.points.forEach(p => dot(tool.screenPath(p, p)[0], dotColor, 4.5));
            }
        };
        draw([...library.lines, ...library.areas].filter(i => i.visible), "#45d4ff", "rgba(69,212,255,0.7)", "#45d4ff");
        draw(chainsOf(tool.model.chains), "#ff4545", "rgba(255,64,64,0.8)", "#ff3b3b");
        for (const p of library.points.filter(i => i.visible)) {
            dot(tool.screenPath(p, p)[0], "#45d4ff", 5);
        }
        for (const nt of notes.filter(i => i.visible)) {
            dot(tool.screenPath(nt, nt)[0], "#ffb020", 5);
        }
    }

    shotBtn.addEventListener("click", () => {
        closeAll();
        // Read the WebGL buffer right after a fresh render (same task), so
        // preserveDrawingBuffer isn't needed.
        viewer.render();
        const w = canvas.width;
        const h = canvas.height;
        const out = document.createElement("canvas");
        out.width = w;
        out.height = h;
        const ctx = out.getContext("2d");
        ctx.drawImage(canvas, 0, 0);
        drawSelectionOnto(ctx, w / canvas.clientWidth);
        out.toBlob(blob => {
            if (!blob) {
                toast("Screenshot failed — the browser couldn't encode the image.", "error");
                return;
            }
            const n = library.screenshots.length + 1;
            const stamp = new Date();
            const region = getRegionKey() || "terrain";
            library.screenshots.push({
                id: nextId++,
                name: `Screenshot ${n}`,
                url: URL.createObjectURL(blob),
                blob,
                width: w,
                height: h,
                region,
                createdAt: stamp,
                fileName: `depthwizard-${region}-${stamp.toISOString().replace(/[:.]/g, "-").slice(0, 19)}.png`,
                visible: false,
            });
            box.classList.add("xv-flash");
            setTimeout(() => box.classList.remove("xv-flash"), 180);
            toast(`Screenshot saved to Library → Screenshots (session only). ${w}×${h} px.`);
            renderLibrary();
        }, "image/png");
    });
    recBtn.addEventListener("click", () => toast("Video recording isn't available yet — use Screenshot for now."));

    // ------------------------------------------------------------------ terrain right-click menu (pointer mode)
    const ctxMenu = popover(box, "xv-ctx");
    let ctxHit = null;

    function noteNear(clientX, clientY) {
        const r = canvas.getBoundingClientRect();
        let best = null;
        for (const n of notes) {
            const q = tool.screenPath(n, n)[0];
            if (!q) {
                continue;
            }
            const d = Math.hypot(q.sx + r.left - clientX, q.sy + r.top - clientY);
            if (d <= 22 && (!best || d < best.d)) {
                best = { note: n, d };
            }
        }
        return best?.note ?? null;
    }

    function place(pop, clientX, clientY) {
        const br = box.getBoundingClientRect();
        pop.hidden = false;
        pop.style.left = `${Math.min(clientX - br.left + 6, br.width - pop.offsetWidth - 8)}px`;
        pop.style.top = `${Math.min(clientY - br.top + 6, br.height - pop.offsetHeight - 8)}px`;
    }

    function openCtx(clientX, clientY, hit) {
        closeAll();
        ctxHit = { ...hit, clientX, clientY };
        ctxMenu.replaceChildren();
        const geo = tool.geo;
        el("div", {
            class: "xv-ctx-title",
            text: `X ${hit.x.toFixed(1)} · Y ${hit.y.toFixed(1)} px · ${Math.round(geo.elevation(hit.x, hit.y)).toLocaleString()} m`,
        }, ctxMenu);
        const item = (label, fn, cls = "") => {
            const b = el("button", { type: "button", role: "menuitem", class: cls, text: label }, ctxMenu);
            b.addEventListener("click", fn);
            return b;
        };
        item("Save point…", () => openPointForm());
        item("Add note…", () => openNoteForm());
        const near = noteNear(clientX, clientY);
        if (near && !near.visible) {
            item("Show note", () => { near.visible = true; closeAll(); invalidate(); openNotePopup(near); });
        }
        if (near) {
            item("Delete note", () => {
                notes.splice(notes.indexOf(near), 1);
                closeAll();
                closeNotePopup();
                toast("Note deleted.");
                syncToolbar();
                invalidate();
            }, "xv-danger");
        }
        if (tool.model.pointCount() > 0) {
            item("Delete from selection", () => {
                closeAll();
                tool.deleteNearestPoint(hit.x, hit.y);
            }, "xv-danger");
        } else {
            el("div", { class: "xv-ctx-sep", text: "Start selection here" }, ctxMenu);
            item("Two points (A → B)", () => { closeAll(); tool.startSelectionAt(MODES.TWO_POINT, hit); });
            item("Continuous (chain / shape)", () => { closeAll(); tool.startSelectionAt(MODES.CONTINUOUS, hit); });
        }
        place(ctxMenu, clientX, clientY);
        ctxMenu.querySelector("button")?.focus();
    }

    function openForm({ title, fields, submitLabel, onSubmit }) {
        const { clientX, clientY } = ctxHit;
        ctxMenu.replaceChildren();
        el("div", { class: "xv-ctx-title", text: title }, ctxMenu);
        const form = el("form", { class: "xv-form", novalidate: "" }, ctxMenu);
        const inputs = {};
        for (const f of fields) {
            const lab = el("label", {}, form);
            el("span", { text: f.label }, lab);
            inputs[f.key] = f.multiline
                ? el("textarea", { rows: "4", maxlength: "500", placeholder: f.placeholder ?? "" }, lab)
                : el("input", { type: "text", maxlength: "60", placeholder: f.placeholder ?? "", value: f.value ?? "" }, lab);
        }
        const err = el("div", { class: "mm-error", role: "alert", hidden: "" }, form);
        const actions = el("div", { class: "xv-form-actions" }, form);
        const cancel = el("button", { type: "button", text: "Cancel" }, actions);
        el("button", { type: "submit", class: "xv-primary", text: submitLabel }, actions);
        cancel.addEventListener("click", () => closeAll());
        form.addEventListener("submit", e => {
            e.preventDefault();
            const problem = onSubmit(Object.fromEntries(Object.entries(inputs).map(([k, v]) => [k, v.value.trim()])));
            if (problem) {
                err.textContent = problem;
                err.hidden = false;
            } else {
                closeAll();
            }
        });
        place(ctxMenu, clientX, clientY);
        inputs[fields[0].key].focus();
    }

    function openPointForm() {
        const hit = ctxHit;
        openForm({
            title: "Save point",
            fields: [
                { key: "name", label: "Name", value: `Point ${library.points.length + 1}` },
                { key: "tag", label: "Tag (optional)", placeholder: "e.g. viewpoint, survey" },
            ],
            submitLabel: "Save point",
            onSubmit: ({ name, tag }) => {
                if (!name) {
                    return "Give the point a name.";
                }
                library.points.push({
                    id: nextId++, name, tag, x: hit.x, y: hit.y, elevation: tool.geo.elevation(hit.x, hit.y),
                    region: getRegionKey(), createdAt: new Date(), visible: true,
                });
                toast(`Saved “${name}” to Library → Points (session only).`);
                renderLibrary();
                invalidate();
                return null;
            },
        });
    }

    function openNoteForm() {
        const hit = ctxHit;
        openForm({
            title: "Add note",
            fields: [{ key: "text", label: "Note", multiline: true, placeholder: "What's here?" }],
            submitLabel: "Add note",
            onSubmit: ({ text }) => {
                if (!text) {
                    return "Write something for the note.";
                }
                notes.push({ id: nextId++, x: hit.x, y: hit.y, text, createdAt: new Date(), visible: true });
                toast("Note added — click its pin to read it (session only).");
                syncToolbar();
                invalidate();
                return null;
            },
        });
    }

    box.addEventListener("contextmenu", e => {
        if (!box.classList.contains("is-expanded") || tool.model.mode !== MODES.NORMAL) {
            return; // measure modes have their own point/line menus
        }
        if (e.target !== canvas && !e.target.closest?.(".xv-overlay, .measure-overlay")) {
            return;
        }
        e.preventDefault();
        const hit = tool.pick(e.clientX, e.clientY);
        if (!hit) {
            closeAll();
            toast("Right-click on the terrain surface for point, note and selection options.");
            return;
        }
        openCtx(e.clientX, e.clientY, hit);
    });

    // ------------------------------------------------------------------ note popup (small square)
    const notePopup = el("div", { class: "xv-note-popup", role: "dialog", "aria-label": "Note", "data-xv-popover": "", "data-xv-ui": "", hidden: "" }, box);
    function placeNotePopup() {
        const n = notes.find(x => String(x.id) === notePopup.dataset.note);
        const q = n && tool.screenPath(n, n)[0];
        if (!n || !q || !n.visible) {
            closeNotePopup();
            return;
        }
        notePopup.style.left = `${Math.min(q.sx + 14, canvas.clientWidth - notePopup.offsetWidth - 10)}px`;
        notePopup.style.top = `${Math.max(10, q.sy - notePopup.offsetHeight - 30)}px`;
    }
    function openNotePopup(n) {
        notePopup.replaceChildren();
        notePopup.dataset.note = String(n.id);
        const head = el("div", { class: "xv-note-head" }, notePopup);
        el("span", { text: "Note" }, head);
        const close = el("button", { type: "button", class: "xv-icon-btn xv-sm", "aria-label": "Close note", "data-tip": "Close", html: icon("x", 16) }, head);
        close.addEventListener("click", closeNotePopup);
        el("div", { class: "xv-note-text", text: n.text }, notePopup);
        el("div", { class: "xv-note-meta", text: `X ${n.x.toFixed(1)} · Y ${n.y.toFixed(1)} px` }, notePopup);
        notePopup.hidden = false;
        placeNotePopup();
    }
    function closeNotePopup() {
        notePopup.hidden = true;
        delete notePopup.dataset.note;
    }
    overlay.addEventListener("click", e => {
        const pin = e.target.closest?.(".xv-note-pin");
        if (pin) {
            const n = notes.find(x => String(x.id) === pin.dataset.note);
            if (n) {
                closeAll();
                openNotePopup(n);
            }
        }
    });
    overlay.addEventListener("keydown", e => {
        const pin = e.target.closest?.(".xv-note-pin");
        if (pin && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            openNotePopup(notes.find(x => String(x.id) === pin.dataset.note));
        }
    });

    // ------------------------------------------------------------------ outside click / Esc
    document.addEventListener("pointerdown", e => {
        if (!confirmModal.hidden) {
            return;
        }
        const inside = popovers.some(p => !p.hidden && p.contains(e.target)) ||
            [measureBtn, viewBtn, libBtn].some(b => b.contains(e.target));
        if (!inside) {
            closeAll();
        }
    });
    document.addEventListener("keydown", e => {
        if (e.key !== "Escape" || !box.classList.contains("is-expanded")) {
            return;
        }
        if (!confirmModal.hidden) {
            settleConfirm(false);
        } else if (!preview.hidden) {
            preview.hidden = true;
        } else if (popovers.some(p => !p.hidden)) {
            closeAll();
        } else if (!notePopup.hidden) {
            closeNotePopup();
        } else {
            return;
        }
        e.preventDefault();
    });

    syncPlay();
    syncTheme();
    syncToolbar();
    renderLibrary();
    libraryPanel.hidden = true;

    return {
        closeAll: () => { closeAll(); closeNotePopup(); preview.hidden = true; if (!confirmModal.hidden) settleConfirm(false); },
        syncTheme,
        syncPlay,
        // read-only views for tests/diagnostics
        get library() {
            return library;
        },
        get notes() {
            return notes;
        },
    };
}
