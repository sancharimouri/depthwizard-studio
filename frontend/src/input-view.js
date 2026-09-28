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

import { apiUrl } from "./api-base.js";
import { attachMagnifier } from "./magnifier.js";
import { footprintKmFromBbox } from "./flat-warning.js";

const TIER2_MAX_GSD_M = 2.4;

const COLLECTIONS = [
    { key: "all", label: "All" },
    { key: "vhr", label: "Maxar VHR" },
    { key: "dfc2019", label: "DFC2019" },
    { key: "sentinel2", label: "Sentinel-2" },
];

const TERRAINS = [
    { key: "all", label: "Any terrain" },
    { key: "hilly", label: "Hilly" },
    { key: "agricultural", label: "Agricultural" },
    { key: "urban", label: "Urban" },
    { key: "coastal", label: "Coastal" },
];

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
    preview: {
        library: "Retrieving the image from the library…",
        upload: "Loading your image…",
        search: "Loading the Sentinel-2 scene…",
    },
    previewCold: "The server is waking up, so this takes a little longer the first time.",
    keepOpen: "Don't close this window.",
};

// The catalog's terrain class; older manifests carry only Sentinel-2's `category`.
export function itemTerrain(item) {
    return item.terrain ?? item.category ?? (item.collection === "dfc2019" ? "urban" : "hilly");
}

export function orderLibraryItems(items) {
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
// relative preview, which needs nothing more.
export function isReady(selection) {
    if (!selection) {
        return false;
    }
    if (selection.source === "library") {
        return true;
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
        { key: "library", label: "Choose from Library" },
        { key: "upload", label: "Upload" },
        { key: "search", label: "Search Online" },
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
            text: def.label,
            onclick: () => setTab(def.key),
        });
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
        el("div", { class: "panel-title", text: "SCENE INPUT" }),
        tabBar,
        ...Object.values(panels));

    // preview pane
    const previewImg = el("img", { class: "iv-preview-img", alt: "" });
    const previewEmpty = el("div", { class: "iv-preview-empty", text: "Nothing selected yet — pick a scene on the left." });
    const previewLoadingText = el("div", { class: "iv-preview-loading-text" });
    const previewLoadingNote = el("div", { class: "iv-preview-loading-note" });
    const previewLoading = el("div", { class: "iv-preview-loading", role: "status", hidden: true },
        el("span", { class: "scene-search-spinner" }), previewLoadingText, previewLoadingNote);
    const previewStage = el("div", { class: "iv-preview-stage" }, previewEmpty, previewImg, previewLoading);
    let previewColdTimer = null;
    function hidePreviewLoading() {
        clearTimeout(previewColdTimer);
        previewLoading.hidden = true;
    }
    previewImg.addEventListener("load", hidePreviewLoading);
    previewImg.addEventListener("error", hidePreviewLoading);
    function showPreviewLoading(sel) {
        previewLoadingText.textContent = LOADING_COPY.preview[sel.source] ?? LOADING_COPY.preview.library;
        previewLoadingNote.textContent = LOADING_COPY.keepOpen;
        previewLoading.hidden = false;
        clearTimeout(previewColdTimer);
        previewColdTimer = setTimeout(() => {
            previewLoadingNote.textContent = `${LOADING_COPY.previewCold} ${LOADING_COPY.keepOpen}`;
        }, COLD_AFTER_MS);
    }
    // same magnifying-glass inspector as the 3D viewer's Image Inspection box
    const previewReadout = el("div", { class: "xp-inspect-readout iv-preview-readout numeric-mono" });
    attachMagnifier({ stage: previewStage, img: previewImg, readout: previewReadout, zoom: 3 });
    const metaList = el("dl", { class: "iv-meta" });
    const routingCard = el("div", { class: "iv-routing", hidden: true });
    const demCard = el("div", { class: "iv-dem", hidden: true });
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
        el("div", { class: "panel-title", text: "PREVIEW" }),
        // image first, then START GENERATION, then everything else
        previewStage, previewReadout, startButton, metaList, routingCard, demCard, startNote);

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
        previewEmpty.hidden = Boolean(sel);
        previewImg.hidden = !sel;
        if (!sel) {
            hidePreviewLoading();
        }
        metaList.replaceChildren();
        routingCard.hidden = !sel;
        demCard.hidden = true;

        if (!sel) {
            startButton.disabled = true;
            startButton.className = "run-reconstruction-button iv-start tier-none";
            startButton.textContent = "▶ START GENERATION";
            return;
        }

        if (previewImg.getAttribute("src") !== sel.previewUrl) {
            previewImg.src = sel.previewUrl;
            if (!previewImg.complete) {
                showPreviewLoading(sel);
            }
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

        startButton.className = `run-reconstruction-button iv-start ${tierClass(routing)}`;
        startButton.textContent = startLabel(routing);
        startButton.disabled = !isReady(sel);
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
        const log = [`Input: ${sel.title} (${sel.sourceLabel})`];
        sel.meta.forEach(([key, value]) => log.push(`${key}: ${value}`));
        log.push(`Routing: ${sel.routing.label}${sel.routing.locked ? " (locked)" : ""}`);
        if (sel.dem) {
            log.push(`DEM: ${sel.dem.source} · ${sel.dem.min_m}–${sel.dem.max_m} m`);
        }
        return {
            source: sel.source,
            title: sel.title,
            previewUrl: sel.previewUrl,
            metaLine: `${sel.title} · ${sel.sourceLabel} · ${sel.routing.label}`,
            meta: sel.meta,
            routing: sel.routing,
            dem: sel.dem ?? null,
            geo: sel.geo ?? null,
            landscape: sel.landscape ?? null,
            // what the backend needs to run relative depth on this exact input (src/depth-result.js)
            inputRef: sel.id ? { source: sel.source, id: sel.id } : null,
            logLines: log,
        };
    }

    // ================================================================ LIBRARY
    const chipRow = el("div", { class: "iv-chips", role: "group", "aria-label": "Filter by collection" });
    const terrainRow = el("div", { class: "iv-chips iv-chips-terrain", role: "group", "aria-label": "Filter by terrain" });
    const cardGrid = el("div", { class: "iv-cards" });
    const libraryStatus = el("div", { class: "iv-status", hidden: true });
    panels.library.append(
        el("p", { class: "iv-hint", text: "Curated scenes with real imagery. Selecting one loads it into the preview." }),
        chipRow, terrainRow, libraryStatus, cardGrid);

    TERRAINS.forEach(t => {
        terrainRow.append(el("button", {
            class: "iv-chip",
            type: "button",
            "data-key": t.key,
            "aria-pressed": String(t.key === state.terrain),
            text: t.label,
            onclick: () => {
                state.terrain = t.key;
                terrainRow.querySelectorAll(".iv-chip").forEach(chip => {
                    chip.setAttribute("aria-pressed", String(chip.dataset.key === t.key));
                });
                renderCards();
            },
        }));
    });

    COLLECTIONS.forEach(col => {
        chipRow.append(el("button", {
            class: "iv-chip",
            type: "button",
            "data-key": col.key,
            "aria-pressed": String(col.key === state.filter),
            text: col.label,
            onclick: () => {
                state.filter = col.key;
                chipRow.querySelectorAll(".iv-chip").forEach(chip => {
                    chip.setAttribute("aria-pressed", String(chip.dataset.key === col.key));
                });
                renderCards();
            },
        }));
    });

    async function loadLibrary() {
        libraryStatus.hidden = false;
        libraryStatus.className = "iv-status";
        libraryStatus.replaceChildren(el("span", { class: "scene-search-spinner" }), "Loading catalog…");
        const coldTimer = setTimeout(() => {
            libraryStatus.replaceChildren(el("span", { class: "scene-search-spinner" }), LOADING_COPY.coldStart);
        }, COLD_AFTER_MS);
        try {
            state.library = await (await api("/api/library")).json();
            clearTimeout(coldTimer);
            libraryStatus.hidden = true;
            chipRow.querySelectorAll(".iv-chip").forEach(chip => {
                const key = chip.dataset.key;
                const count = key === "all" ? state.library.total : state.library.counts[key];
                chip.textContent = `${COLLECTIONS.find(c => c.key === key).label} ${count}`;
            });
            terrainRow.querySelectorAll(".iv-chip").forEach(chip => {
                const key = chip.dataset.key;
                const count = key === "all" ? state.library.items.length
                    : state.library.items.filter(item => itemTerrain(item) === key).length;
                chip.textContent = `${TERRAINS.find(t => t.key === key).label} ${count}`;
            });
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
        const items = orderLibraryItems(state.library.items
            .filter(item => state.filter === "all" || item.collection === state.filter)
            .filter(item => state.terrain === "all" || itemTerrain(item) === state.terrain));
        cardGrid.replaceChildren(...items.map(item => el("button", {
            class: `iv-card ${tierClass(item.routing)}${item.available === false ? " is-remote" : ""}`,
            type: "button",
            "data-id": item.id,
            "aria-pressed": String(state.selection?.id === item.id),
            "aria-label": item.available === false ? `${item.title}: not downloaded yet, download` : null,
            onclick: event => (item.available === false ? downloadLibraryItem(item, event.currentTarget) : selectLibraryItem(item)),
        },
        thumbMedia(item),
        el("div", { class: "iv-card-body" },
            el("div", { class: "iv-card-title", text: item.title }),
            el("div", { class: "iv-card-sub", text: item.location }),
            el("div", { class: "iv-card-tags" },
                el("span", { class: "numeric-mono", text: formatGsd(item.gsd_m) }),
                el("span", { text: item.collection === "vhr" ? "Maxar" : item.collection === "dfc2019" ? "DFC2019" : "Sentinel-2" }),
                el("span", { class: "iv-card-terrain", text: itemTerrain(item) }),
                el("span", { class: "iv-card-tier", text: `T${item.routing.tier}` })),
        ))));
        watchForColdThumbs();
    }

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
            geo: item.geo ? { lat: item.geo.lat, lon: item.geo.lon, origin: "the file's geotransform (tile centre)" } : null,
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
            const routed = await (await api(`/api/library/${item.id}/select`, {
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
        updateIfCurrent(sel, { demBusy: "Fetching FABDEM (30 m bare-earth) for this footprint from Google Earth Engine…", demError: null });
        try {
            const meta = await (await api(`/api/input/${sel.id}/fabdem`, { method: "POST" })).json();
            updateIfCurrent(sel, { demBusy: null, dem: meta.dem, demPreviewUrl: `${apiUrl(meta.dem_preview_url)}?t=${Date.now()}` });
        } catch (error) {
            updateIfCurrent(sel, { demBusy: null, demError: `FABDEM fetch failed: ${error.message}` });
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
