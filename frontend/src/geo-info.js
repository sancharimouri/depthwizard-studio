// Facts box + Scenario Analysis cards: where the lines come from, and how they render
// (docs/facts-research.md, facts v2). One object per job:
//   { facts: [line], scenario: { flood: [line], landslide: [line], earthquake: [line] } }
// where a line is { kind, label, text[, data] } (backend/facts/lines.py writes the same shape).
//   - web build, library tile: the curated lines baked into the static library (job.input.staticInfo); no request
//   - anything else: GET /api/facts with the input's footprint (bbox) or the user-entered point
// Lines carry no sources or licences: the Docs page credits them ("Data sources & credits").

import { apiUrl } from "./api-base.js";

export const SCENARIOS = ["flood", "landslide", "earthquake"];
const FETCH_TIMEOUT_MS = 15000;

const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

export function emptyInfo() {
    return { facts: [], scenario: { flood: [], landslide: [], earthquake: [] } };
}

// Normalises anything shaped like an info object (baked, fetched, or missing parts).
export function normaliseInfo(raw) {
    const out = emptyInfo();
    if (!raw) {
        return out;
    }
    out.facts = Array.isArray(raw.facts) ? raw.facts.filter(isLine) : [];
    SCENARIOS.forEach(k => {
        out.scenario[k] = Array.isArray(raw.scenario?.[k]) ? raw.scenario[k].filter(isLine) : [];
    });
    return out;
}

// The facts payload this frontend reads: { facts: [...], scenario: {...} }.
export function isFactsPayload(data) {
    return Boolean(data) && (Array.isArray(data.facts) || (data.scenario && typeof data.scenario === "object"));
}

function isLine(x) {
    return x && typeof x === "object" && typeof x.kind === "string";
}

export function countLines(info) {
    return info.facts.length + SCENARIOS.reduce((n, k) => n + info.scenario[k].length, 0);
}

// Box around a centre with a footprint in km (library items carry centre + footprint_km).
export function bboxFromCentre(lat, lon, footprintKm) {
    const [fx, fy] = footprintKm;
    const dLat = fy / 2 / 110.574;
    const dLon = fx / 2 / (111.32 * Math.cos(lat * Math.PI / 180));
    return [lon - dLon, lat - dLat, lon + dLon, lat + dLat].map(v => Number(v.toFixed(5)));
}

// What to ask for: the georeferenced footprint (bbox), else the user-entered point, else nothing.
export function infoQuery(job) {
    const geo = job?.input?.geo;
    if (geo?.bbox) {
        return { bbox: geo.bbox };
    }
    if (geo && Number.isFinite(geo.lat) && Number.isFinite(geo.lon)) {
        return { lat: geo.lat, lon: geo.lon };
    }
    if (job?.userGeo) {
        return { lat: job.userGeo.lat, lon: job.userGeo.lon };
    }
    return null;
}

export function queryKey(q) {
    return q ? (q.bbox ? `bbox:${q.bbox.join(",")}` : `pt:${q.lat},${q.lon}`) : "";
}

// Resolves to { key, info, source: "static" | "live" | "none", failed }. Cached on the job per location;
// a failed or slow request resolves to an empty info object (never throws, never surfaces an error text).
export function loadGeoInfo(job, fetchImpl = globalThis.fetch) {
    if (!job) {
        return Promise.resolve({ key: "", info: emptyInfo(), source: "none", failed: false });
    }
    const baked = job.input?.staticInfo;
    if (baked) {
        return Promise.resolve({ key: "static", info: normaliseInfo(baked), source: "static", failed: false });
    }
    const q = infoQuery(job);
    const key = queryKey(q);
    if (!q) {
        return Promise.resolve({ key, info: emptyInfo(), source: "none", failed: false });
    }
    if (job.geoInfo?.key === key) {
        return job.geoInfo.promise;
    }
    const params = q.bbox ? `bbox=${q.bbox.join(",")}` : `lat=${encodeURIComponent(q.lat)}&lon=${encodeURIComponent(q.lon)}`;
    const ctrl = typeof AbortController === "function" ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS) : null;
    const promise = fetchImpl(apiUrl(`/api/facts?${params}`), ctrl ? { signal: ctrl.signal } : undefined)
        .then(r => (r.ok ? r.json() : Promise.reject(Object.assign(new Error(`HTTP ${r.status}`), { status: r.status }))))
        .then(data => (isFactsPayload(data)
            ? { key, info: normaliseInfo(data), source: "live", failed: false }
            // a backend that doesn't speak this facts format (an older deployment): no box, no error
            : { key, info: emptyInfo(), source: "live", failed: false, unsupported: true }))
        .catch(error => {
            if (job.geoInfo?.key === key) {
                job.geoInfo = null; // let the next open retry
            }
            // 4xx = the backend doesn't accept this query (e.g. an older one without bbox support): hide quietly
            const unsupported = error?.status >= 400 && error?.status < 500;
            return { key, info: emptyInfo(), source: "live", failed: !unsupported, unsupported };
        })
        .finally(() => timer && clearTimeout(timer));
    job.geoInfo = { key, promise };
    return promise;
}

// A value's parts ("Observatory Hill · 2,188 m · 0.8 km") render as separate spans with a gap between them.
function valueParts(text) {
    return String(text).split(" · ").map(p => `<span class="xp-fact-part">${esc(p)}</span>`).join("");
}

function rows(lines) {
    return lines.filter(l => l.kind !== "epicentres" && l.text)
        .map(l => `<div class="xp-fact-row"><span class="xp-fact-label">${esc(l.label)}</span>`
            + `<span class="xp-fact-value">${valueParts(l.text)}</span></div>`).join("");
}

// The closing "Flood · Earthquake · Landslide → Scenario Analysis" line is sized to span the box's width
// exactly (one line). Re-fitted when the box changes width.
const fitObserved = new WeakSet();
export function fitFactsMore(root) {
    const line = root?.querySelector(".xp-facts-more");
    if (!line) {
        return;
    }
    const fit = () => {
        const parent = line.parentElement;
        const cs = parent ? getComputedStyle(parent) : null;
        const width = parent ? parent.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight) : 0;
        if (!width) {
            return;
        }
        line.style.fontSize = "10px";
        const natural = line.scrollWidth;
        if (natural) {
            line.style.fontSize = `${Math.min(10 * width / natural, 15).toFixed(2)}px`;
        }
    };
    fit();
    if (root && !fitObserved.has(root) && typeof ResizeObserver !== "undefined") {
        fitObserved.add(root);
        new ResizeObserver(() => fitFactsMore(root)).observe(root);
    }
}

// Value left-aligned under its label when, on one line, it would be wider than half the row. Needs layout,
// so it runs on visible content (again whenever a box opens).
export function fitFactRows(root) {
    fitFactsMore(root);
    root?.querySelectorAll(".xp-fact-row").forEach(row => {
        const value = row.querySelector(".xp-fact-value");
        if (!value || !row.clientWidth) {
            return;
        }
        row.classList.remove("is-long");
        row.classList.add("is-measuring");
        const natural = value.getBoundingClientRect().width;
        row.classList.remove("is-measuring");
        row.classList.toggle("is-long", natural > row.clientWidth * 0.5);
    });
}

export const SCENARIO_LINK = `<p class="xp-facts-more">Flood · Earthquake · Landslide → `
    + `<button type="button" class="xp-link" data-open-scenario>Scenario Analysis</button></p>`;

// The Facts box body: one row per fact, or the empty-state message; always the Scenario Analysis link.
export function factsBodyHtml(info) {
    const body = rows(info.facts);
    return (body || `<p class="xp-fact-empty">No other terrain hazards on record here.</p>`) + SCENARIO_LINK;
}

// One Scenario Analysis card ("" when the option has no lines).
export function scenarioCardHtml(info, key) {
    const lines = info.scenario[key] ?? [];
    const body = rows(lines);
    const epi = key === "earthquake" ? lines.find(l => l.kind === "epicentres")?.data : null;
    if (!body && !epi) {
        return "";
    }
    return body + (epi ? epicentreSvg(epi) : "");
}

// Inset map for the earthquake card: the tile outline, 50/100 km rings and every recorded M4.5+ epicentre
// within the stated radius, sized by magnitude. At tile scale a hazard tint would be one colour, so none is drawn.
export function epicentreSvg(data, size = 220) {
    const events = data?.events ?? [];
    const [lat0, lon0] = data?.centre ?? [];
    if (!events.length || !Number.isFinite(lat0) || !Number.isFinite(lon0)) {
        return "";
    }
    const radius = data.radius_km || 100;
    const half = size / 2;
    const scale = (half - 8) / radius; // px per km
    const kx = 111.32 * Math.cos(lat0 * Math.PI / 180);
    const px = (lat, lon) => [half + (lon - lon0) * kx * scale, half - (lat - lat0) * 110.574 * scale];
    const ring = r => `<circle cx="${half}" cy="${half}" r="${(r * scale).toFixed(1)}" class="xp-epi-ring"/>`
        + `<text x="${half}" y="${(half - r * scale - 2).toFixed(1)}" class="xp-epi-label">${r} km</text>`;
    let tile = "";
    if (Array.isArray(data.bbox)) {
        const [w, s, e, n] = data.bbox;
        const [x0, y0] = px(n, w);
        const [x1, y1] = px(s, e);
        const wpx = Math.max(3, x1 - x0);
        const hpx = Math.max(3, y1 - y0);
        tile = `<rect x="${(x0 + (x1 - x0 - wpx) / 2).toFixed(1)}" y="${(y0 + (y1 - y0 - hpx) / 2).toFixed(1)}" `
            + `width="${wpx.toFixed(1)}" height="${hpx.toFixed(1)}" class="xp-epi-tile"/>`;
    } else {
        tile = `<circle cx="${half}" cy="${half}" r="2.5" class="xp-epi-tile"/>`;
    }
    const dots = [...events].sort((a, b) => a[2] - b[2]).map(([lon, lat, mag, year]) => {
        const [x, y] = px(lat, lon);
        const r = 1.6 + Math.max(0, mag - 4.5) * 2.2;
        // great-circle distance from the centre, as the card's "largest" line (backend/facts haversine_km)
        const rad = Math.PI / 180;
        const h = Math.sin((lat - lat0) * rad / 2) ** 2 + Math.cos(lat0 * rad) * Math.cos(lat * rad) * Math.sin((lon - lon0) * rad / 2) ** 2;
        const km = 6371 * 2 * Math.asin(Math.sqrt(h));
        const info = `M${Number(mag).toFixed(1)} · ${year} · ${Math.round(km)} km away`;
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" data-info="${esc(info)}" class="xp-epi-dot${mag >= 6 ? " is-strong" : ""}">`
            + `<title>M${Number(mag).toFixed(1)} · ${esc(year)}</title></circle>`;
    }).join("");
    return `<figure class="xp-epi"><svg viewBox="0 0 ${size} ${size}" width="100%" role="img" `
        + `aria-label="${events.length} M4.5+ epicentres within ${radius} km since 1973">`
        + `<clipPath id="xp-epi-clip"><circle cx="${half}" cy="${half}" r="${half - 1}"/></clipPath>`
        + `<g clip-path="url(#xp-epi-clip)">${ring(radius / 2)}${ring(radius)}${dots}${tile}</g></svg>`
        + `<figcaption>Epicentres M4.5+ since 1973 · tile outlined · ${radius} km radius</figcaption></figure>`;
}
