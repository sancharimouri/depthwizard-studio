// ============================================================
// PAGE 1 INPUT VIEW — "Choose from Library" / "Upload" / "Search Online"
// on the left half, live preview + tier routing + START GENERATION on the
// right half. Everything shown here is real (catalog, GSD read from the
// file's geotransform, FABDEM fetched from Earth Engine, CDSE search and
// quota); what is NOT yet real — the generation stages that follow — is
// labelled as a placeholder next to the button.
//
// Tier colours (flagged for the user to confirm): green = Tier 2 (height
// prediction), orange = Tier 1 (DEM only), neutral = relative preview only.
// ============================================================

import { LIBRARY_SOURCE, staticLibraryListing } from "./library-source.js";
import { bboxFromCentre } from "./geo-info.js";
import { apiUrl } from "./api-base.js";
import { attachMagnifier } from "./magnifier.js";
import { hud } from "./hud.js";
import { footprintKmFromBbox } from "./flat-warning.js";

const TIER2_MAX_GSD_M = 2.4;

const COLLECTIONS = [
    { key: "all", label: "All" },
    { key: "vhr", label: "Maxar VHR" },
    { key: "dfc2019", label: "DFC2019" },
    { key: "sentinel2", label: "Sentinel-2" },
];

// The terrain filter's dropdown, in menu order.
const TERRAINS = [
    { key: "all", label: "Any terrain" },
    { key: "hilly", label: "Hilly" },
    { key: "coastal", label: "Coastal" },
    { key: "agricultural", label: "Agricultural" },
    { key: "urban", label: "Urban" },
];

// Library items for a collection ("all" or a key) and a terrain ("all" or a class).
export function filterLibrary(items, collection, terrain) {
    return items
        .filter(item => collection === "all" || item.collection === collection)
        .filter(item => terrain === "all" || itemTerrain(item) === terrain);
}

// The message shown when a collection + terrain choice has no tiles (null when it has some):
// it says so and names the imagery that does have that terrain.
export function noMatchMessage(items, collection, terrain) {
    if (filterLibrary(items, collection, terrain).length) {
        return null;
    }
    const t = TERRAINS.find(x => x.key === terrain)?.label ?? terrain;
    const c = COLLECTIONS.find(x => x.key === collection)?.label ?? collection;
    const others = COLLECTIONS
        .filter(x => x.key !== "all" && x.key !== collection && filterLibrary(items, x.key, terrain).length)
        .map(x => x.label);
    const where = collection === "all" ? "in the library" : `in ${c} imagery`;
    const tail = others.length
        ? `Try the ${t} filter with ${others.length > 1 ? `${others.slice(0, -1).join(", ")} or ${others.at(-1)}` : others[0]} imagery.`
        : "Try another terrain.";
    return `No ${t.toLowerCase()} tiles ${where}. ${tail}`;
}

// Always first in the library, in this order; everything else keeps the catalog's
// (deliberately mixed) order from scripts/library_catalog.py.
const PINNED_FIRST = ["sentinel2-darjeeling", "sentinel2-almora", "sentinel2-manali"];

// Loading copy. A hosted backend that has been idle is frozen, and its first
// requests (catalog, thumbnails, previews) take much longer than usual. After
// COLD_AFTER_MS without an answer the UI says why, instead of sitting empty.
export const COLD_AFTER_MS = 2500;
export const LOADING_COPY = {
    coldStart: "The server is waking up after being idle, so this first load takes a few seconds longer. "
        + "It only happens once. Please don't close this window.",
    thumb: "Loading…",
    thumbCold: "Waking the server…",
    thumbError: "Preview unavailable",
};

// The catalog's terrain class; older manifests carry only Sentinel-2's `category`.
export function itemTerrain(item) {
    return item.terrain ?? item.category ?? (item.collection === "dfc2019" ? "urban" : "hilly");
}

// Tile card text. The heading is the title without its state ("…, Odisha", which moves to the
// subheading); if that still does not fit one line it also loses any bracketed part
// ("(Godavari delta)"). The subheading is the
// full location (+ ", India" for Indian scenes) minus whatever the heading already shows.
export function titleCandidates(title) {
    const noState = title.replace(/,\s*[^,()]+$/, "").trim();
    const noBracket = noState.replace(/\s*\([^)]*\)/g, "").replace(/\s{2,}/g, " ").trim();
    // the state always moves to the subheading; brackets go only if the heading still won't fit
    return [...new Set([noState, noBracket].filter(Boolean))];
}

export function cardSubtitle(item, shownTitle) {
    const base = item.collection === "dfc2019" || /india$/i.test(item.location) ? item.location : `${item.location}, India`;
    let rest = base.startsWith(shownTitle) ? base.slice(shownTitle.length).replace(/^[\s,;\u2014-]+/, "") : base;
    if (rest.startsWith("(")) {
        rest = rest.replace(/^\(([^)]*)\)/, "$1").replace(/^[\s,;]+/, "");
    }
    return rest.charAt(0).toUpperCase() + rest.slice(1);
}

const TERRAIN_SHORT = { agricultural: "Argi", coastal: "Coast" };

export function cardTerrain(item) {
    const terrain = itemTerrain(item);
    const short = TERRAIN_SHORT[terrain] ?? terrain;
    return short.charAt(0).toUpperCase() + short.slice(1);
}

// Display names for the Maxar VHR crops (the catalog's raw titles are the crop codes, "a forest", "c town"...).
export const VHR_DISPLAY_TITLES = {
    "vhr-a_valley": "Sikkim Valley",
    "vhr-c_town": "Sikkim Town",
    "vhr-c_terraces": "Sikkim Terraces",
    "vhr-a_forest": "Sikkim Forest",
    "vhr-c_river": "Sikkim River",
    "vhr-b_glacier": "Sikkim Glacier",
};

export function withDisplayTitles(listing) {
    return { ...listing, items: listing.items.map(item => (VHR_DISPLAY_TITLES[item.id] ? { ...item, title: VHR_DISPLAY_TITLES[item.id] } : item)) };
}

export function orderLibraryItems(items) {
    // A curated listing (backend tile_manifest.json) carries its own order: follow it as served.
    if (items.length && items.every(item => Number.isFinite(item.order_index))) {
        return [...items].sort((a, b) => a.order_index - b.order_index);
    }
    const rank = item => {
        const i = PINNED_FIRST.indexOf(item.id);
        return i < 0 ? PINNED_FIRST.length : i;
    };
    return items.map((item, i) => ({ item, i }))
        .sort((a, b) => rank(a.item) - rank(b.item) || a.i - b.i)
        .map(x => x.item);
}

// ---------------------------------------------------------------- pure helpers (unit-tested)

export function tierClass(routing) {
    if (!routing || routing.tier == null) {
        return "tier-none";
    }
    return routing.tier === 2 ? "tier-2" : "tier-1";
}

// No tier on the button (2026-09-25): the routing card below it says it.
export function startLabel(_routing) {
    return "▶ START GENERATION";
}

export function formatGsd(gsd) {
    if (gsd == null || !Number.isFinite(gsd)) {
        return "unknown";
    }
    if (gsd < 1) {
        return `${Math.round(gsd * 100)} cm`;
    }
    return `${Number(gsd.toFixed(gsd < 10 ? 2 : 1))} m`;
}

// Is the selection ready to generate? Georeferenced inputs need their DEM
// (user upload or FABDEM) first; non-georeferenced ones can only run the
// relative preview, which needs only their GSD, typed in by the user.
export function isReady(selection) {
    if (!selection) {
        return false;
    }
    if (selection.source === "library") {
        return true;
    }
    if (selection.gsdRequired) {
        return false;
    }
    const routing = selection.routing;
    if (!routing || routing.tier == null) {
        return true;
    }
    return Boolean(selection.dem);
}

// Quota panel text — the documented limits plus this server's own spend.
// Never a "remaining" count: CDSE doesn't publish one.
export function quotaLines(q) {
    if (!q) {
        return [];
    }
    const lim = q.documented_limits;
    const use = q.this_server_usage;
    const lines = [
        `Account: ${q.account?.typology ?? "unknown"} (${q.credentials === "user" ? "your key" : "project key"})`,
        `Documented limits: ${lim.requests_per_minute} req/min · ${lim.requests_per_month.toLocaleString("en-US")} req/month · `
            + `${lim.processing_units_per_minute} PU/min · ${lim.processing_units_per_month.toLocaleString("en-US")} PU/month`,
        `This server since ${use.since.replace("T", " ").replace("Z", " UTC")}: ${use.catalog_requests} searches, `
            + `${use.process_requests} image requests, ${use.processing_units} PU`,
    ];
    lines.push(use.throttled
        ? `Throttled ${use.throttled}× (last ${use.last_throttle}${use.retry_after ? `, retry after ${use.retry_after}s` : ""})`
        : "Not throttled");
    return lines;
}

const DEM_UNAVAILABLE = "The elevation service isn't available right now, so terrain can't be fetched for this image. "
    + "Please try again later, or pick a scene from the library.";

// ---------------------------------------------------------------- DOM helpers

function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (value == null || value === false) {
            continue;
        }
        if (key === "class") {
            node.className = value;
        } else if (key === "text") {
            node.textContent = value;
        } else if (key.startsWith("on")) {
            node.addEventListener(key.slice(2), value);
        } else {
            node.setAttribute(key, value === true ? "" : value);
        }
    }
    node.append(...children.filter(child => child != null));
    return node;
}

// Small line icons for the pane headers, tabs and the preview's empty state (stroke = currentColor).
const ICONS = {
    grid: '<rect x="3.5" y="3.5" width="17" height="17" rx="2.5"/><path d="M3.5 9.2h17M3.5 14.8h17M9.2 3.5v17M14.8 3.5v17"/>',
    eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"/><circle cx="12" cy="12" r="3"/>',
    image: '<rect x="3.5" y="3.5" width="17" height="17" rx="3"/><circle cx="9" cy="9" r="1.8"/><path d="m4 17.5 5-5 3.5 3.5 3-3 4 4"/>',
    cloud: '<path d="M7 18.5a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 17.9 9.2 4.7 4.7 0 0 1 17.5 18.5Z"/><path d="M12 15.5V10m0 0-2.2 2.2M12 10l2.2 2.2"/>',
    search: '<circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/>',
};

function icon(name, size = 18) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("iv-icon");
    svg.innerHTML = ICONS[name];
    return svg;
}

function paneHeader(iconName, title, subtitle) {
    return el("header", { class: "iv-pane-head" },
        el("span", { class: "iv-pane-icon" }, icon(iconName, 20)),
        el("div", { class: "iv-pane-heading" },
            el("div", { class: "panel-title", text: title }),
            el("div", { class: "iv-pane-sub", text: subtitle })));
}

async function api(url, options = {}) {
    const response = await fetch(apiUrl(url), options);
    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            detail = (await response.json()).detail || detail;
        } catch {
            // not JSON
        }
        const error = new Error(detail);
        error.status = response.status;
        error.retryAfter = response.headers.get("Retry-After");
        throw error;
    }
    return response;
}

// ---------------------------------------------------------------- view

export function createInputView(root, { onStart }) {
    const state = {
        tab: "library",
        library: null,
        filter: "all",
        terrain: "all",
        selection: null,
        userKey: null, // { id, secret } — memory only, never persisted
        searchAoi: null,
        geocode: null,
    };

    // ---------- layout
    const tabDefs = [
        { key: "library", label: "Choose from Library", icon: "image" },
        { key: "upload", label: "Upload", icon: "cloud" },
        { key: "search", label: "Search Online", icon: "search" },
    ];

    const tabBar = el("div", { class: "iv-tabs", role: "tablist", "aria-label": "Input source" });
    const panels = {};
    const tabButtons = {};

    tabDefs.forEach(def => {
        const button = el("button", {
            class: "iv-tab",
            type: "button",
            role: "tab",
            id: `iv-tab-${def.key}`,
            "aria-controls": `iv-panel-${def.key}`,
            onclick: () => setTab(def.key),
        }, icon(def.icon, 16), el("span", { text: def.label }));
        tabButtons[def.key] = button;
        tabBar.append(button);
        panels[def.key] = el("div", {
            class: "iv-panel",
            role: "tabpanel",
            id: `iv-panel-${def.key}`,
            "aria-labelledby": `iv-tab-${def.key}`,
        });
    });

    tabBar.addEventListener("keydown", event => {
        const order = tabDefs.map(def => def.key);
        const index = order.indexOf(state.tab);
        let next = null;
        if (event.key === "ArrowRight") {
            next = order[(index + 1) % order.length];
        } else if (event.key === "ArrowLeft") {
            next = order[(index - 1 + order.length) % order.length];
        }
        if (next) {
            event.preventDefault();
            setTab(next);
            tabButtons[next].focus();
        }
    });

    const inputPane = el("section", { class: "iv-pane iv-input-pane" },
        paneHeader("grid", "SCENE INPUT", "Choose a scene to generate relative depth and terrain"),
        tabBar,
        ...Object.values(panels));

    // preview pane
    const previewImg = el("img", { class: "iv-preview-img", alt: "", hidden: true });
    const previewEmpty = el("div", { class: "iv-preview-empty" },
        el("span", { class: "iv-viewfinder" }, icon("image", 44)),
        el("div", { class: "iv-empty-title", text: "Nothing selected yet" }),
        el("div", { class: "iv-empty-sub", text: "Pick a scene on the left to load its imagery and details." }));
    const previewStage = el("div", { class: "iv-preview-stage" }, previewEmpty, previewImg);
    // A newly selected image is loaded off-screen; the current frame (the empty state or the previous image)
    // stays until it has fully loaded, then the new one fades in. No loading text, no flash.
    let previewWanted = null;
    function showPreviewWhenLoaded(url) {
        previewWanted = url;
        const loader = new Image();
        loader.onload = () => {
            if (previewWanted !== url) {
                return;                                     // another selection came in meanwhile
            }
            previewImg.src = url;
            previewImg.hidden = false;
            previewEmpty.hidden = true;
            previewImg.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: 150, easing: "ease-out" });
        };
        loader.src = url;
    }
    // same magnifying-glass inspector as the 3D viewer's Image Inspection box
    const previewReadout = el("div", { class: "xp-inspect-readout iv-preview-readout numeric-mono" });
    // the magnifier also drives the HUD readout/ruler: the coordinates under the pointer, from the tile's footprint
    function coordAt(u, v) {
        const bbox = state.selection?.geo?.bbox;
        if (!bbox) {
            return null;
        }
        const [w, s, e, n] = bbox;
        return { lat: n - v * (n - s), lon: w + u * (e - w) };
    }
    attachMagnifier({
        stage: previewStage, img: previewImg, readout: previewReadout, zoom: 3,
        onHover: (u, v) => hud.magnify(coordAt(u, v)),
        onLeave: () => hud.magnify(null),
    });
    const metaList = el("dl", { class: "iv-meta" });
    const routingCard = el("div", { class: "iv-routing", hidden: true });
    const demCard = el("div", { class: "iv-dem", hidden: true });
    // An upload with no geotransform: say so and ask for its GSD (metres per pixel).
    const gsdCard = el("div", { class: "iv-gsd", hidden: true });
    const startButton = el("button", {
        class: "run-reconstruction-button iv-start tier-none",
        type: "button",
        disabled: true,
        text: "▶ START GENERATION",
        onclick: () => {
            if (isReady(state.selection)) {
                onStart(buildHandoff(state.selection));
            }
        },
    });
    const startNote = el("div", {
        class: "iv-placeholder-note",
        text: "Everything generated is this input's own: relative depth from the image, elevation and the 3D terrain "
            + "from a real elevation model for its footprint (none if it has no georeference).",
    });

    const previewPane = el("section", { class: "iv-pane iv-preview-pane" },
        paneHeader("eye", "PREVIEW", "Selected scene will appear here"),
        // image first, then START GENERATION, then everything else
        previewStage, previewReadout, gsdCard, startButton, metaList, routingCard, demCard, startNote);

    root.replaceChildren(inputPane, previewPane);

    // ---------- tabs
    function setTab(key) {
        state.tab = key;
        tabDefs.forEach(def => {
            const active = def.key === key;
            tabButtons[def.key].setAttribute("aria-selected", String(active));
            tabButtons[def.key].tabIndex = active ? 0 : -1;
            panels[def.key].hidden = !active;
        });
        if (key === "library" && !state.library) {
            loadLibrary();
        }
        if (key === "search") {
            refreshQuota();
        }
    }

    // ---------- preview / routing rendering
    function renderSelection() {
        const sel = state.selection;
        hud.setAnchor(sel?.geo ?? null, Boolean(sel));
        if (!sel) {
            previewWanted = null;
            previewEmpty.hidden = false;
            previewImg.hidden = true;
        }
        metaList.replaceChildren();
        routingCard.hidden = !sel;
        demCard.hidden = true;
        gsdCard.hidden = true;

        if (!sel) {
            startButton.disabled = true;
            startButton.className = "run-reconstruction-button iv-start tier-none";
            startButton.textContent = "▶ START GENERATION";
            return;
        }

        if (previewImg.getAttribute("src") !== sel.previewUrl && previewWanted !== sel.previewUrl) {
            showPreviewWhenLoaded(sel.previewUrl);
        }
        previewImg.alt = `${sel.title} preview`;

        sel.meta.forEach(([key, value]) => {
            metaList.append(el("dt", { text: key }), el("dd", { text: value }));
        });

        const routing = sel.routing;
        routingCard.className = `iv-routing ${tierClass(routing)}`;
        routingCard.replaceChildren(
            el("div", { class: "iv-routing-label" },
                el("span", { class: "iv-tier-dot", "aria-hidden": "true" }),
                routing.label + (routing.locked ? " · locked" : "")),
            el("p", { class: "iv-routing-summary", text: routing.summary }),
            ...(sel.notes || []).map(note => el("p", { class: "iv-routing-note", text: note })),
        );

        renderDem(sel);
        renderGsd(sel);

        startButton.className = `run-reconstruction-button iv-start ${tierClass(routing)}`;
        startButton.textContent = startLabel(routing);
        startButton.disabled = !isReady(sel);
        startButton.title = sel.gsdRequired ? "Enter the image's GSD first" : "";
    }

    function renderGsd(sel) {
        if (sel.source !== "upload" || !(sel.gsdRequired || sel.gsdManual)) {
            return;
        }
        gsdCard.hidden = false;
        gsdCard.classList.toggle("is-set", !sel.gsdRequired);
        const input = el("input", {
            class: "iv-gsd-input numeric-mono", type: "number", inputmode: "decimal", min: "0.01", max: "1000", step: "any",
            placeholder: "e.g. 0.5", "aria-label": "Ground sample distance in metres per pixel",
            value: sel.gsdManual && sel.gsdM != null ? String(sel.gsdM) : null,
        });
        const apply = el("button", { class: "secondary-button iv-gsd-apply", type: "button", text: sel.gsdManual ? "Update" : "Set GSD" });
        const submit = () => applyManualGsd(sel, input.value);
        apply.addEventListener("click", submit);
        input.addEventListener("keydown", event => {
            if (event.key === "Enter") {
                event.preventDefault();
                submit();
            }
        });
        gsdCard.replaceChildren(...[
            el("div", { class: "iv-gsd-title" },
                el("span", { class: "iv-gsd-icon", "aria-hidden": "true", text: sel.gsdRequired ? "!" : "✓" }),
                sel.gsdRequired ? "No geotransform in this image" : "GSD entered manually"),
            el("p", { class: "iv-gsd-text", text: sel.gsdRequired
                ? "Its ground resolution can't be read from the file. Enter the GSD (ground distance per pixel) manually to continue."
                : `Set to ${formatGsd(sel.gsdM)} per pixel. You can change it below.` }),
            el("div", { class: "iv-gsd-row" }, input, el("span", { class: "iv-gsd-unit", text: "m / pixel" }), apply),
            sel.gsdError ? el("div", { class: "iv-status is-error", text: sel.gsdError }) : null,
        ].filter(Boolean)); // (replaceChildren would print a null as "null")
        if (sel.gsdRequired && !sel.gsdError) {
            requestAnimationFrame(() => input.focus({ preventScroll: true }));
        }
    }

    async function applyManualGsd(sel, raw) {
        const value = Number(String(raw).trim());
        if (!(value >= 0.01 && value <= 1000)) {
            state.selection = { ...sel, gsdError: "Enter a number of metres per pixel between 0.01 and 1000." };
            renderSelection();
            return;
        }
        try {
            const meta = await (await api(`/api/input/${sel.id}/gsd`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ gsd_m: value }),
            })).json();
            if (state.selection?.id === sel.id) {
                select({ ...selectionFromInput(meta, "upload"), gsdError: null });
            }
        } catch (error) {
            state.selection = { ...sel, gsdError: `Couldn't set the GSD: ${error.message}` };
            renderSelection();
        }
    }

    function renderDem(sel) {
        if (sel.source === "library" || !sel.routing || sel.routing.tier == null) {
            return;
        }
        demCard.hidden = false;
        const children = [el("div", { class: "panel-title", text: "DEM" })];

        if (sel.demBusy) {
            children.push(el("div", { class: "iv-status" },
                el("span", { class: "scene-search-spinner" }), sel.demBusy));
        } else if (sel.dem) {
            const d = sel.dem;
            children.push(
                el("div", { class: "iv-dem-row" },
                    sel.demPreviewUrl ? el("img", { class: "iv-dem-img", src: sel.demPreviewUrl, alt: "DEM hillshade" }) : null,
                    el("div", { class: "iv-dem-text" },
                        el("div", { text: d.source }),
                        el("div", { class: "numeric-mono", text: `${d.min_m} – ${d.max_m} m · mean ${d.mean_m} m` }),
                        el("div", { class: "numeric-mono", text: `${d.size_px[0]}×${d.size_px[1]} px · ${d.crs} · ${Math.round(d.valid_fraction * 100)}% valid` }),
                    )),
            );
        } else if (sel.routing.tier === 2) {
            // GSD <= 2.4 m: the user chooses — their own DEM, or FABDEM fetched for them.
            const demInput = el("input", { type: "file", accept: ".tif,.tiff", hidden: true });
            demInput.addEventListener("change", () => {
                if (demInput.files?.[0]) {
                    attachUserDem(sel, demInput.files[0]);
                }
            });
            children.push(
                el("p", { class: "iv-hint", text: "Choose the terrain source for this image:" }),
                el("div", { class: "iv-dem-choice" },
                    el("label", { class: "secondary-button iv-choice" }, "Upload my DEM (GeoTIFF)", demInput),
                    el("button", { class: "secondary-button iv-choice", type: "button", text: "Fetch FABDEM for me",
                        onclick: () => fetchFabdem(sel) }),
                ),
            );
        }
        if (sel.demNote) {
            children.push(el("div", { class: "iv-status", text: sel.demNote }));
        }
        if (sel.demError) {
            children.push(el("div", { class: "iv-status is-error", text: sel.demError }));
        }
        demCard.replaceChildren(...children);
    }

    function select(selection) {
        state.selection = selection;
        renderSelection();
    }

    // ---------- hand-off to the generation page
    function buildHandoff(sel) {
        // the processing page's calculation log: no tier or elevation-model names
        const log = [`Input: ${sel.title} (${sel.sourceLabel})`];
        sel.meta.forEach(([key, value]) => log.push(`${key}: ${value}`));
        if (sel.dem) {
            log.push(`Attached DEM: ${sel.dem.min_m}–${sel.dem.max_m} m`);
        }
        return {
            source: sel.source,
            title: sel.title,
            previewUrl: sel.previewUrl,
            metaLine: `${sel.title} · ${sel.sourceLabel} · ${sel.routing.label}`,
            sourceLabel: sel.sourceLabel,
            meta: sel.meta,
            routing: sel.routing,
            dem: sel.dem ?? null,
            geo: sel.geo ?? null,
            staticInfo: sel.staticInfo ?? null,
            landscape: sel.landscape ?? null,
            // what the backend needs to run relative depth on this exact input (src/depth-result.js)
            inputRef: sel.id ? { source: sel.source, id: sel.id } : null,
            logLines: log,
        };
    }

    // ================================================================ LIBRARY
    const chipRow = el("div", { class: "iv-chips", role: "group", "aria-label": "Filter by collection" });
    const cardGrid = el("div", { class: "iv-cards" });
    const libraryStatus = el("div", { class: "iv-status", hidden: true });

    // Terrain filter: a dropdown on the right of a small bar between the collection chips and
    // the tiles (the collection chips stay as they are). A choice with no tiles for the current
    // collection is not applied: the last results stay, and a notice says why and where that
    // terrain can be found. It closes itself after NOTICE_MS, or with its x.
    const NOTICE_MS = 4000;
    const filterCount = el("span", { class: "iv-filter-count" });
    const filterLabel = el("span", { class: "iv-filter-label" });
    const filterButton = el("button", {
        class: "iv-filter-btn", type: "button", "aria-haspopup": "listbox", "aria-expanded": "false",
        "aria-controls": "iv-terrain-menu",
    }, filterIcon(), filterLabel, el("span", { class: "iv-filter-chevron", "aria-hidden": "true" }));
    const filterMenu = el("div", {
        id: "iv-terrain-menu", class: "iv-filter-menu", role: "listbox", "aria-label": "Terrain", hidden: true,
    });
    const filterOptions = TERRAINS.map(t => el("button", {
        class: "iv-filter-option", type: "button", role: "option", "data-key": t.key,
        onclick: () => {
            closeFilterMenu(true);
            applyFilter(state.filter, t.key);
        },
    }, el("span", { class: "iv-filter-option-label", text: t.label }), el("span", { class: "iv-filter-option-count" })));
    filterMenu.append(...filterOptions);
    const filterBar = el("div", { class: "iv-filter-bar" },
        filterCount, el("div", { class: "iv-filter" }, filterButton, filterMenu));

    const noticeText = el("span", { class: "iv-filter-notice-text" });
    const notice = el("div", { class: "iv-filter-notice", role: "status", "aria-live": "polite", hidden: true },
        noticeText,
        el("button", {
            class: "iv-filter-notice-close", type: "button", "aria-label": "Close message", text: "×",
            onclick: () => hideNotice(),
        }));
    const cardsWrap = el("div", { class: "iv-cards-wrap" },
        el("div", { class: "iv-filter-notice-anchor" }, notice), cardGrid);

    panels.library.append(
        el("p", { class: "iv-hint", text: "Curated scenes with real imagery." }),
        chipRow, filterBar, libraryStatus, cardsWrap);

    function filterIcon() {
        const ns = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(ns, "svg");
        svg.setAttribute("viewBox", "0 0 24 24");
        svg.setAttribute("aria-hidden", "true");
        svg.classList.add("iv-filter-icon");
        const path = document.createElementNS(ns, "path");
        path.setAttribute("d", "M22 3H2l8 9.46V19l4 2v-8.54L22 3z");
        svg.append(path);
        return svg;
    }

    let noticeTimer = 0;
    function showNotice(message) {
        noticeText.textContent = message;
        notice.hidden = false;
        clearTimeout(noticeTimer);
        noticeTimer = setTimeout(hideNotice, NOTICE_MS);
    }
    function hideNotice() {
        clearTimeout(noticeTimer);
        notice.hidden = true;
    }

    function openFilterMenu() {
        filterMenu.hidden = false;
        filterButton.setAttribute("aria-expanded", "true");
        (filterOptions.find(o => o.dataset.key === state.terrain) ?? filterOptions[0]).focus();
    }
    function closeFilterMenu(returnFocus = false) {
        if (filterMenu.hidden) {
            return;
        }
        filterMenu.hidden = true;
        filterButton.setAttribute("aria-expanded", "false");
        if (returnFocus) {
            filterButton.focus();
        }
    }
    filterButton.addEventListener("click", () => (filterMenu.hidden ? openFilterMenu() : closeFilterMenu()));
    filterMenu.addEventListener("keydown", event => {
        const i = filterOptions.indexOf(document.activeElement);
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            const step = event.key === "ArrowDown" ? 1 : -1;
            filterOptions[(i + step + filterOptions.length) % filterOptions.length].focus();
        } else if (event.key === "Escape") {
            event.preventDefault();
            closeFilterMenu(true);
        } else if (event.key === "Tab") {
            closeFilterMenu();
        }
    });
    document.addEventListener("pointerdown", event => {
        if (!filterMenu.hidden && !event.target.closest?.(".iv-filter")) {
            closeFilterMenu();
        }
    });

    // Button label, per-terrain counts for the current collection, selection marks.
    function syncFilterUi() {
        const items = state.library?.items ?? [];
        const t = TERRAINS.find(x => x.key === state.terrain);
        filterLabel.textContent = state.terrain === "all" ? "Terrain" : `Terrain: ${t.label}`;
        filterButton.classList.toggle("is-active", state.terrain !== "all");
        filterOptions.forEach(option => {
            const key = option.dataset.key;
            const n = filterLibrary(items, state.filter, key).length;
            option.setAttribute("aria-selected", String(key === state.terrain));
            option.classList.toggle("is-empty", state.library != null && n === 0);
            option.querySelector(".iv-filter-option-count").textContent = state.library ? String(n) : "";
        });
        chipRow.querySelectorAll(".iv-chip").forEach(chip => {
            chip.setAttribute("aria-pressed", String(chip.dataset.key === state.filter));
        });
        const shown = state.library ? filterLibrary(items, state.filter, state.terrain).length : null;
        filterCount.textContent = shown == null ? "" : `${shown} scene${shown === 1 ? "" : "s"}`;
    }

    // Applies a collection + terrain choice, unless it has no tiles: then the current results
    // stay on screen and the notice explains. Returns whether it was applied.
    function applyFilter(collection, terrain) {
        const message = state.library ? noMatchMessage(state.library.items, collection, terrain) : null;
        if (message) {
            showNotice(message);
            syncFilterUi();
            return false;
        }
        state.filter = collection;
        state.terrain = terrain;
        hideNotice();
        syncFilterUi();
        renderCards();
        return true;
    }

    COLLECTIONS.forEach(col => {
        chipRow.append(el("button", {
            class: "iv-chip",
            type: "button",
            "data-key": col.key,
            "aria-pressed": String(col.key === state.filter),
            text: col.label,
            onclick: () => applyFilter(col.key, state.terrain),
        }));
    });
    syncFilterUi();

    async function loadLibrary() {
        libraryStatus.hidden = false;
        libraryStatus.className = "iv-status";
        libraryStatus.replaceChildren(el("span", { class: "scene-search-spinner" }), "Loading catalog…");
        const coldTimer = setTimeout(() => {
            libraryStatus.replaceChildren(el("span", { class: "scene-search-spinner" }), LOADING_COPY.coldStart);
        }, COLD_AFTER_MS);
        try {
            const listing = LIBRARY_SOURCE === "static" ? await staticLibraryListing() : await (await api("/api/library")).json();
            state.library = withDisplayTitles(listing);
            clearTimeout(coldTimer);
            libraryStatus.hidden = true;
            chipRow.querySelectorAll(".iv-chip").forEach(chip => {
                const key = chip.dataset.key;
                const count = key === "all" ? state.library.total : state.library.counts[key];
                chip.textContent = `${COLLECTIONS.find(c => c.key === key).label} ${count}`;
            });
            syncFilterUi();
            renderCards();
        } catch (error) {
            clearTimeout(coldTimer);
            libraryStatus.className = "iv-status is-error";
            libraryStatus.textContent = `Catalog unavailable: ${error.message}`;
        }
    }

    function renderCards() {
        if (!state.library) {
            return;
        }
        // Same order in the web and desktop apps (pinned hill scenes, then the catalog's mixed
        // order); on-demand desktop tiles stay in place with their download overlay.
        const items = orderLibraryItems(filterLibrary(state.library.items, state.filter, state.terrain));
        cardGrid.replaceChildren(...items.map(item => el("button", {
            class: `iv-card ${tierClass(item.routing)}${item.available === false ? " is-remote" : ""}${item.collection === "dfc2019" ? "" : " is-named"}`,
            type: "button",
            "data-id": item.id,
            "aria-pressed": String(state.selection?.id === item.id),
            "aria-label": item.available === false ? `${item.title}: not downloaded yet, download` : null,
            onclick: event => (item.available === false ? downloadLibraryItem(item, event.currentTarget) : selectLibraryItem(item)),
            onpointerenter: () => hud.hoverTile(item.geo),
            onpointerleave: () => hud.hoverTile(null),
        },
        thumbMedia(item),
        el("div", { class: "iv-card-body" },
            el("div", { class: "iv-card-title", text: item.title }),
            el("div", { class: "iv-card-sub", text: cardSubtitle(item, item.title) }),
            el("div", { class: "iv-card-tags" },
                el("span", {
                    class: "iv-card-details",
                    text: [
                        formatGsd(item.gsd_m),
                        item.collection === "vhr" ? "Maxar" : item.collection === "dfc2019" ? "DFC2019" : "Sentinel-2",
                        cardTerrain(item),
                    ].join(" \u2022 "),
                }),
                el("span", { class: "iv-card-tier", text: `T${item.routing.tier}` })),
        ))));
        cardItems = items;
        fitCardText();
        watchForColdThumbs();
    }

    // One-line headings and detail lines: shorten the heading step by step, then (only if the
    // detail line still overflows) nudge its size down. Needs layout, so it waits for a visible grid.
    let cardItems = [];
    function fitCardText() {
        if (!cardGrid.clientWidth) {
            return;
        }
        cardGrid.querySelectorAll(".iv-card").forEach(card => {
            const item = cardItems.find(candidate => candidate.id === card.dataset.id);
            const titleEl = card.querySelector(".iv-card-title");
            const subEl = card.querySelector(".iv-card-sub");
            const detailsEl = card.querySelector(".iv-card-details");
            if (!item || !titleEl) {
                return;
            }
            const candidates = titleCandidates(item.title);
            let shown = candidates[candidates.length - 1];
            for (const candidate of candidates) {
                titleEl.textContent = candidate;
                if (titleEl.scrollWidth <= titleEl.clientWidth + 0.5) {
                    shown = candidate;
                    break;
                }
            }
            titleEl.textContent = shown;
            titleEl.classList.toggle("is-wrap", titleEl.scrollWidth > titleEl.clientWidth + 0.5);
            subEl.textContent = cardSubtitle(item, shown);
            if (detailsEl) {
                detailsEl.style.fontSize = "";
                let size = parseFloat(getComputedStyle(detailsEl).fontSize);
                while (detailsEl.scrollWidth > detailsEl.clientWidth + 0.5 && size > 6) {
                    size -= 0.5;
                    detailsEl.style.fontSize = `${size}px`;
                }
            }
        });
    }
    // Refit only once the grid's width has settled. The sidebar's 0.2 s open/close animation resizes the
    // grid every frame, and measuring ~76 cards per frame (forced layout) made it stutter.
    let fittedWidth = 0;
    let fitTimer = null;
    new ResizeObserver(() => {
        clearTimeout(fitTimer);
        fitTimer = setTimeout(() => {
            if (cardGrid.clientWidth !== fittedWidth) {
                fittedWidth = cardGrid.clientWidth;
                fitCardText();
            }
        }, 150);
    }).observe(cardGrid);
    document.fonts?.ready.then(fitCardText);

    // A card's thumbnail with a message in its place until it arrives. If no
    // thumbnail has arrived COLD_AFTER_MS after the cards first render, the
    // server is cold: the cards say so and the status line explains.
    let thumbsWarm = false;
    let thumbColdTimer = null;

    function thumbMedia(item) {
        const media = el("div", { class: `iv-card-media${thumbsWarm ? "" : " is-loading"}` });
        const label = el("div", { class: "iv-card-loading", text: LOADING_COPY.thumb });
        const img = el("img", { class: "iv-card-thumb", src: apiUrl(item.thumbnail_url), alt: "", loading: "lazy" });
        img.addEventListener("load", () => {
            media.classList.remove("is-loading");
            label.remove();
            markThumbsWarm();
        });
        img.addEventListener("error", () => {
            label.textContent = LOADING_COPY.thumbError;
            img.style.visibility = "hidden"; // no broken-image icon under the label
        });
        media.append(label, img);
        if (item.available === false) {
            media.append(downloadOverlay(item));
        }
        if (img.complete && img.naturalWidth) {
            media.classList.remove("is-loading");
            label.remove();
        }
        return media;
    }

    function markThumbsWarm() {
        if (thumbsWarm) {
            return;
        }
        thumbsWarm = true;
        clearTimeout(thumbColdTimer);
        if (libraryStatus.dataset.cold === "thumbs") {
            libraryStatus.hidden = true;
            delete libraryStatus.dataset.cold;
        }
    }

    function watchForColdThumbs() {
        if (thumbsWarm || thumbColdTimer) {
            return;
        }
        thumbColdTimer = setTimeout(() => {
            if (thumbsWarm) {
                return;
            }
            libraryStatus.hidden = false;
            libraryStatus.className = "iv-status";
            libraryStatus.dataset.cold = "thumbs";
            libraryStatus.replaceChildren(el("span", { class: "scene-search-spinner" }), LOADING_COPY.coldStart);
            cardGrid.querySelectorAll(".iv-card-loading").forEach(label => {
                if (label.textContent === LOADING_COPY.thumb) {
                    label.textContent = LOADING_COPY.thumbCold;
                }
            });
        }, COLD_AFTER_MS);
    }

    // Desktop app only (the backend's bundle mode): an item that isn't bundled shows its
    // thumbnail under a translucent overlay; clicking downloads its preview + tile into the
    // per-user library, then selects it. The web app never sets `available`.
    function downloadOverlay(item) {
        const mb = item.download_bytes ? ` · ${(item.download_bytes / 1e6).toFixed(1)} MB` : "";
        const overlay = el("div", { class: "iv-card-download" }, el("span", { class: "iv-card-download-label", text: `Download${mb}` }));
        overlay.prepend(downloadIcon());
        return overlay;
    }

    function downloadIcon() {
        const ns = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(ns, "svg");
        svg.setAttribute("viewBox", "0 0 24 24");
        svg.setAttribute("aria-hidden", "true");
        svg.classList.add("iv-card-download-icon");
        const path = document.createElementNS(ns, "path");
        path.setAttribute("d", "M12 4v11m0 0-4.5-4.5M12 15l4.5-4.5M5 19h14");
        svg.append(path);
        return svg;
    }

    async function downloadLibraryItem(item, card) {
        if (card.classList.contains("is-downloading")) {
            return;
        }
        const label = card.querySelector(".iv-card-download-label");
        card.classList.remove("is-error");
        card.classList.add("is-downloading");
        label.textContent = "Downloading…";
        try {
            const updated = await (await api(`/api/library/${item.id}/download`, { method: "POST" })).json();
            const i = state.library.items.findIndex(x => x.id === item.id);
            state.library.items[i] = { ...state.library.items[i], ...updated };
            renderCards();
            selectLibraryItem(state.library.items[i]);
        } catch (error) {
            card.classList.remove("is-downloading");
            card.classList.add("is-error");
            label.textContent = error.status === 401 ? "Needs a Hugging Face token" : "Download failed: retry";
            card.title = error.message;
        }
    }

    async function selectLibraryItem(item) {
        cardGrid.querySelectorAll(".iv-card").forEach(card => {
            card.setAttribute("aria-pressed", String(card.dataset.id === item.id));
        });
        const meta = [
            ["Source", item.source],
            ["Location", item.location],
            ["Tile", item.tile_id],
            ["GSD", `${formatGsd(item.gsd_m)} (${item.gsd_source})`],
            ["Size", `${item.size_px[0]} × ${item.size_px[1]} px`],
        ];
        if (item.geo) {
            meta.push(["CRS", item.geo.crs], ["Centre", `${item.geo.lat.toFixed(4)}, ${item.geo.lon.toFixed(4)}`]);
        }
        if (item.acquired) {
            meta.push(["Acquired", item.acquired]);
        }
        select({
            source: "library",
            id: item.id,
            title: item.title,
            sourceLabel: "curated library",
            previewUrl: apiUrl(item.preview_url),
            meta,
            routing: item.routing,
            // centre from the tile's own geotransform (scripts/library_catalog.py _geo); none for DFC2019
            geo: item.geo ? {
                lat: item.geo.lat, lon: item.geo.lon, origin: "the file's geotransform (tile centre)",
                bbox: item.geo.footprint_km ? bboxFromCentre(item.geo.lat, item.geo.lon, item.geo.footprint_km) : null,
            } : null,
            // web build only: the curated Facts + Scenario lines baked into the static library (none curated: the
            // Facts panel hides); null elsewhere, so the desktop build asks its own backend
            staticInfo: LIBRARY_SOURCE === "static" ? { facts: item.facts ?? [], scenario: item.scenario ?? {} } : null,
            // for the flat-terrain warning (src/flat-warning.js)
            landscape: {
                collection: item.collection,
                terrain: itemTerrain(item),
                footprintKm: item.geo?.footprint_km ?? null,
                gsdM: item.gsd_m,
            },
        });
        // Server-side routing is authoritative (Sentinel-2 is locked to Tier 1 there).
        try {
            // static web build: the plan the backend returns for this request, baked with the listing
            const routed = LIBRARY_SOURCE === "static" ? item.select : await (await api(`/api/library/${item.id}/select`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ requested_tier: item.routing.tier }),
            })).json();
            if (state.selection?.id === item.id) {
                state.selection.routing = routed.routing;
                state.selection.notes = routed.notes;
                renderSelection();
            }
        } catch (error) {
            console.error("Library routing failed:", error);
        }
    }

    // ================================================================ UPLOAD
    const uploadInput = el("input", { type: "file", accept: ".tif,.tiff,.png,.jpg,.jpeg", hidden: true });
    const dropzone = el("label", { class: "upload-dropzone iv-dropzone" },
        el("div", { class: "upload-dropzone-glyph", text: "⇧" }),
        el("div", { class: "upload-dropzone-text", text: "Drop a GeoTIFF, PNG or JPG — or click to browse" }),
        uploadInput);
    const uploadStatus = el("div", { class: "iv-status", hidden: true });
    panels.upload.append(
        dropzone,
        uploadStatus,
        el("ul", { class: "iv-rules" },
            el("li", { text: `GeoTIFF ≤ ${TIER2_MAX_GSD_M} m/pixel → Tier 2 (height prediction). You upload a DEM, or FABDEM is fetched for you.` }),
            el("li", { text: `GeoTIFF > ${TIER2_MAX_GSD_M} m/pixel → Tier 1 (DEM only). FABDEM is fetched automatically.` }),
            el("li", { text: "PNG / JPG (no georeference) → relative preview only. There is no location to fetch a DEM for." })),
        el("p", { class: "iv-hint", text: "GSD is read from the file's geotransform, not assumed." }),
    );

    uploadInput.addEventListener("change", () => uploadFile(uploadInput.files?.[0]));
    dropzone.addEventListener("dragover", event => {
        event.preventDefault();
        dropzone.classList.add("drag-over");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
    dropzone.addEventListener("drop", event => {
        event.preventDefault();
        dropzone.classList.remove("drag-over");
        uploadFile(event.dataTransfer?.files?.[0]);
    });

    function inputMeta(m) {
        const meta = [["File", m.filename ?? m.scene_id ?? "—"]];
        if (m.kind === "sentinel2-scene") {
            meta.push(["Acquired", m.date], ["Cloud", `${m.cloud}%`]);
        } else {
            meta.push(["Format", `${m.format.toUpperCase()} · ${m.bands} band${m.bands === 1 ? "" : "s"} · ${m.dtype}`],
                ["Size", `${m.size_px[0]} × ${m.size_px[1]} px`]);
        }
        meta.push(["GSD", m.gsd_m == null ? `unknown (${m.gsd_source})` : `${formatGsd(m.gsd_m)} (${m.gsd_source})`]);
        if (m.georeferenced) {
            const [w, s, e, n] = m.footprint_wgs84;
            meta.push(["CRS", m.crs], ["Footprint", `${s.toFixed(4)}–${n.toFixed(4)} N, ${w.toFixed(4)}–${e.toFixed(4)} E`]);
        }
        return meta;
    }

    // Coordinates only from real geo-metadata: the upload's geotransform, or the
    // area a searched Sentinel-2 scene was cropped to. Never inferred.
    function inputGeo(m, source) {
        if (!m.georeferenced || !m.footprint_wgs84) {
            return null;
        }
        const [w, s, e, n] = m.footprint_wgs84;
        return {
            lat: Number(((s + n) / 2).toFixed(5)),
            lon: Number(((w + e) / 2).toFixed(5)),
            bbox: [w, s, e, n].map(v => Number(Number(v).toFixed(5))),
            origin: source === "search" ? "the searched Sentinel-2 scene's area (centre)" : "the file's geotransform (footprint centre)",
        };
    }

    function selectionFromInput(m, source) {
        return {
            source,
            id: m.id,
            title: m.filename ?? `Sentinel-2 ${m.date}`,
            sourceLabel: source === "upload" ? "upload" : "Sentinel-2 via CDSE search",
            previewUrl: `${apiUrl(m.preview_url)}?v=${m.id}`,
            meta: inputMeta(m),
            routing: m.routing,
            geo: inputGeo(m, source),
            // a searched scene has no terrain class: the warning judges it by its generated relief
            landscape: m.kind === "sentinel2-scene" && m.footprint_wgs84
                ? { collection: "sentinel2", terrain: null, footprintKm: footprintKmFromBbox(m.footprint_wgs84), gsdM: m.gsd_m }
                : null,
            dem: m.dem,
            demPreviewUrl: apiUrl(m.dem_preview_url),
            // no geotransform: the GSD is asked for (gsdCard) and set with POST /api/input/<id>/gsd
            gsdRequired: Boolean(m.gsd_required),
            gsdManual: Boolean(m.gsd_manual),
            gsdM: m.gsd_m,
        };
    }

    async function uploadFile(file) {
        if (!file) {
            return;
        }
        uploadStatus.hidden = false;
        uploadStatus.className = "iv-status";
        uploadStatus.replaceChildren(el("span", { class: "scene-search-spinner" }),
            `Uploading ${file.name} (${(file.size / 1024 ** 2).toFixed(1)} MB) and reading its geotransform…`);
        const body = new FormData();
        body.append("file", file);
        try {
            const meta = await (await api("/api/input/upload", { method: "POST", body })).json();
            uploadStatus.hidden = true;
            const sel = selectionFromInput(meta, "upload");
            select(sel);
            if (sel.routing.tier === 1) {
                fetchFabdem(sel); // > 2.4 m: automatic, no choice offered
            }
        } catch (error) {
            uploadStatus.className = "iv-status is-error";
            uploadStatus.textContent = `Upload failed: ${error.message}`;
        } finally {
            uploadInput.value = "";
        }
    }

    function updateIfCurrent(sel, patch) {
        Object.assign(sel, patch);
        if (state.selection === sel) {
            renderSelection();
        }
    }

    async function fetchFabdem(sel) {
        updateIfCurrent(sel, { demBusy: "Fetching FABDEM (30 m bare-earth) for this footprint from Google Earth Engine…", demError: null, demNote: null });
        try {
            const meta = await (await api(`/api/input/${sel.id}/fabdem`, { method: "POST" })).json();
            updateIfCurrent(sel, { demBusy: null, dem: meta.dem, demPreviewUrl: `${apiUrl(meta.dem_preview_url)}?t=${Date.now()}` });
        } catch (error) {
            // a server-side outage (5xx, e.g. the elevation service missing on the host) is not the user's doing:
            // a calm note instead of the raw error text
            updateIfCurrent(sel, error.status >= 500
                ? { demBusy: null, demError: null, demNote: DEM_UNAVAILABLE }
                : { demBusy: null, demError: `FABDEM fetch failed: ${error.message}` });
        }
    }

    async function attachUserDem(sel, file) {
        updateIfCurrent(sel, { demBusy: `Checking ${file.name}: georeference, single band, footprint overlap…`, demError: null });
        const body = new FormData();
        body.append("file", file);
        try {
            const meta = await (await api(`/api/input/${sel.id}/dem`, { method: "POST", body })).json();
            updateIfCurrent(sel, { demBusy: null, dem: meta.dem, demPreviewUrl: `${apiUrl(meta.dem_preview_url)}?t=${Date.now()}` });
        } catch (error) {
            updateIfCurrent(sel, { demBusy: null, demError: `DEM rejected: ${error.message}` });
        }
    }

    // ================================================================ SEARCH ONLINE
    const locInput = el("input", { class: "scene-input", type: "text", placeholder: "Search a place name…", autocomplete: "off",
        "aria-label": "Location" });
    const locDropdown = el("div", { class: "geocode-dropdown", hidden: true });
    const locReadout = el("div", { class: "iv-hint numeric-mono", hidden: true });
    const aoiSelect = el("select", { class: "scene-input", "aria-label": "Area of interest" },
        el("option", { value: "5", text: "5 × 5 km" }),
        el("option", { value: "10", selected: true, text: "10 × 10 km" }),
        el("option", { value: "20", text: "20 × 20 km" }));
    const dateFrom = el("input", { class: "scene-input", type: "date", value: "2025-11-01", "aria-label": "From" });
    const dateTo = el("input", { class: "scene-input", type: "date", value: "2026-01-31", "aria-label": "To" });
    const cloudInput = el("input", { class: "scene-input", type: "number", min: "0", max: "100", value: "20", "aria-label": "Max cloud %" });
    const searchButton = el("button", { class: "secondary-button iv-search-button", type: "button", text: "SEARCH SENTINEL-2", disabled: true });
    const searchStatus = el("div", { class: "iv-status", hidden: true });
    const results = el("div", { class: "iv-results" });

    const keyToggle = el("input", { type: "checkbox" });
    const keyId = el("input", { class: "scene-input", type: "text", placeholder: "Client ID", autocomplete: "off", "aria-label": "CDSE client ID" });
    const keySecret = el("input", { class: "scene-input", type: "password", placeholder: "Client secret", autocomplete: "off",
        "aria-label": "CDSE client secret" });
    const keyApply = el("button", { class: "secondary-button", type: "button", text: "Use this key" });
    const keyStatus = el("div", { class: "iv-hint" });
    const keyFields = el("div", { class: "iv-key-fields", hidden: true },
        keyId, keySecret, keyApply, keyStatus,
        el("p", { class: "iv-hint",
            text: "Create an OAuth client under your Copernicus Data Space account (Sentinel Hub dashboard → User settings → OAuth clients). "
                + "The key stays in this tab's memory only: it's sent with each request and never saved." }));
    const quotaBox = el("div", { class: "iv-quota" });

    panels.search.append(
        el("p", { class: "iv-hint", text: "Live Sentinel-2 L2A search (Copernicus Data Space Ecosystem). Sentinel-2 is 10 m, so it always runs as Tier 1: FABDEM terrain, fetched automatically." }),
        el("div", { class: "geocode-field iv-field" }, el("span", { class: "scene-field-label", text: "Location" }), locInput, locDropdown),
        locReadout,
        el("div", { class: "iv-field-row" },
            el("label", { class: "iv-field" }, el("span", { class: "scene-field-label", text: "AOI" }), aoiSelect),
            el("label", { class: "iv-field" }, el("span", { class: "scene-field-label", text: "From" }), dateFrom),
            el("label", { class: "iv-field" }, el("span", { class: "scene-field-label", text: "To" }), dateTo),
            el("label", { class: "iv-field" }, el("span", { class: "scene-field-label", text: "Max cloud %" }), cloudInput)),
        searchButton, searchStatus, results,
        el("label", { class: "iv-key-toggle" }, keyToggle, "Use your own Copernicus API key"),
        keyFields,
        el("div", { class: "panel-title iv-quota-title", text: "CDSE QUOTA" }),
        quotaBox,
    );

    function cdseHeaders(extra = {}) {
        return state.userKey
            ? { ...extra, "X-CDSE-Client-Id": state.userKey.id, "X-CDSE-Client-Secret": state.userKey.secret }
            : extra;
    }

    keyToggle.addEventListener("change", () => {
        keyFields.hidden = !keyToggle.checked;
        if (!keyToggle.checked) {
            state.userKey = null;
            keyId.value = "";
            keySecret.value = "";
            keyStatus.textContent = "";
            refreshQuota();
        }
    });
    keyApply.addEventListener("click", async () => {
        const id = keyId.value.trim();
        const secret = keySecret.value.trim();
        if (!id || !secret) {
            keyStatus.textContent = "Both the client ID and the secret are needed.";
            return;
        }
        state.userKey = { id, secret };
        keyStatus.textContent = "Checking key…";
        const ok = await refreshQuota();
        keyStatus.textContent = ok ? "Key accepted — searches now use your account's quota." : "Key rejected — still using it would fail; clear it or fix it.";
        if (!ok) {
            state.userKey = null;
        }
    });

    async function refreshQuota() {
        try {
            const q = await (await api("/api/cdse/quota", { headers: cdseHeaders() })).json();
            quotaBox.replaceChildren(
                ...quotaLines(q).map(line => el("div", { text: line })),
                el("div", { class: "iv-hint", text: "No 'remaining' count is shown: CDSE doesn't publish one. It throttles by volume (HTTP 429) once a limit is hit, and monthly limits reset on the 1st." }),
                el("a", { class: "iv-link", href: q.documented_limits.source, target: "_blank", rel: "noopener", text: "Copernicus quota documentation" }),
            );
            return true;
        } catch (error) {
            quotaBox.replaceChildren(el("div", { class: "iv-status is-error", text: `Quota unavailable: ${error.message}` }));
            return false;
        }
    }

    // Location autocomplete (OpenStreetMap Nominatim, no key)
    let geoTimer = null;
    let geoRequest = 0;
    locInput.addEventListener("input", () => {
        const query = locInput.value.trim();
        const requestId = ++geoRequest;
        state.geocode = null;
        searchButton.disabled = true;
        locReadout.hidden = true;
        clearTimeout(geoTimer);
        if (query.length < 3) {
            locDropdown.hidden = true;
            return;
        }
        geoTimer = setTimeout(async () => {
            locDropdown.hidden = false;
            locDropdown.replaceChildren(el("div", { class: "geocode-status", text: "Searching…" }));
            try {
                const response = await fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&q=${encodeURIComponent(query)}`,
                    { headers: { Accept: "application/json" } });
                const places = await response.json();
                if (requestId !== geoRequest) {
                    return;
                }
                if (!places.length) {
                    locDropdown.replaceChildren(el("div", { class: "geocode-status", text: "No matches" }));
                    return;
                }
                locDropdown.replaceChildren(...places.map(place => el("button", {
                    class: "geocode-result",
                    type: "button",
                    text: place.display_name,
                    onclick: () => {
                        state.geocode = place;
                        locInput.value = place.display_name;
                        locDropdown.hidden = true;
                        locReadout.hidden = false;
                        locReadout.textContent = `📍 ${Number(place.lat).toFixed(4)}, ${Number(place.lon).toFixed(4)}`;
                        searchButton.disabled = false;
                    },
                })));
            } catch (error) {
                if (requestId === geoRequest) {
                    locDropdown.replaceChildren(el("div", { class: "geocode-status", text: `Lookup failed: ${error.message}` }));
                }
            }
        }, 400);
    });

    function showSearchStatus(text, isError = false, spinner = false) {
        searchStatus.hidden = false;
        searchStatus.className = `iv-status${isError ? " is-error" : ""}`;
        searchStatus.replaceChildren(...(spinner ? [el("span", { class: "scene-search-spinner" })] : []), text);
    }

    function cdseErrorText(error) {
        if (error.status === 429) {
            return `CDSE throttled this request (volume limit reached)${error.retryAfter ? ` — retry in ${error.retryAfter}s` : ""}.`;
        }
        return error.message;
    }

    searchButton.addEventListener("click", async () => {
        if (!state.geocode) {
            return;
        }
        searchButton.disabled = true;
        results.replaceChildren();
        showSearchStatus("Querying the Sentinel Hub Catalog…", false, true);
        try {
            const data = await (await api("/api/cdse/search", {
                method: "POST",
                headers: cdseHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({
                    lat: Number(state.geocode.lat),
                    lon: Number(state.geocode.lon),
                    aoi_km: Number(aoiSelect.value),
                    date_from: dateFrom.value,
                    date_to: dateTo.value,
                    max_cloud: Number(cloudInput.value),
                }),
            })).json();
            state.searchAoi = data.bbox;
            if (!data.scenes.length) {
                showSearchStatus("No Sentinel-2 scenes matched — widen the dates or raise the cloud limit.", true);
            } else {
                searchStatus.hidden = true;
                renderResults(data.scenes);
            }
        } catch (error) {
            showSearchStatus(`Search failed: ${cdseErrorText(error)}`, true);
        } finally {
            searchButton.disabled = false;
            refreshQuota();
        }
    });

    function renderResults(scenes) {
        results.replaceChildren(...scenes.map(scene => {
            const thumb = el("div", { class: "scene-result-thumb scene-result-thumb-placeholder", text: "S2" });
            const card = el("button", { class: "scene-result-card", type: "button", onclick: () => selectScene(scene, card) },
                thumb,
                el("div", {},
                    el("div", { class: "scene-result-name", text: scene.id }),
                    el("div", { class: "scene-result-sub", text: `${scene.date} · ${scene.cloud}% cloud · 10 m` })));
            loadThumb(scene, thumb);
            return card;
        }));
    }

    // Real 96 px Process API crops (the catalog has no quicklook asset).
    async function loadThumb(scene, thumbEl) {
        try {
            const blob = await (await api("/api/cdse/preview", {
                method: "POST",
                headers: cdseHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ bbox: state.searchAoi, date: scene.date, width: 96, height: 96 }),
            })).blob();
            thumbEl.replaceWith(el("img", { class: "scene-result-thumb", src: URL.createObjectURL(blob), alt: "" }));
        } catch {
            // leave the placeholder tile — no fabricated imagery
        }
    }

    async function selectScene(scene, card) {
        results.querySelectorAll(".scene-result-card").forEach(c => c.classList.toggle("is-loading", c === card));
        showSearchStatus("Fetching the true-colour scene…", false, true);
        try {
            const meta = await (await api("/api/input/scene", {
                method: "POST",
                headers: cdseHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ id: scene.id, date: scene.date, cloud: scene.cloud, bbox: state.searchAoi }),
            })).json();
            searchStatus.hidden = true;
            const sel = selectionFromInput(meta, "search");
            select(sel);
            fetchFabdem(sel); // Tier 1: automatic
        } catch (error) {
            showSearchStatus(`Couldn't load that scene: ${cdseErrorText(error)}`, true);
        } finally {
            card.classList.remove("is-loading");
            refreshQuota();
        }
    }

    setTab("library");

    return {
        reset() {
            select(null);
            cardGrid.querySelectorAll(".iv-card").forEach(card => card.setAttribute("aria-pressed", "false"));
            setTab("library");
        },
    };
}
