import test from "node:test";
import assert from "node:assert/strict";

import {
    bboxFromCentre, countLines, emptyInfo, epicentreSvg, factsBodyHtml, infoQuery, loadGeoInfo, normaliseInfo,
    queryKey, scenarioCardHtml,
} from "../src/geo-info.js";

const DARJ = {
    facts: [
        { kind: "wildfire", label: "Wildfire hazard", text: "HIGH (district)" },
        { kind: "peak", label: "Nearest named peak", text: "Observatory Hill · 2,188 m · 0.8 km" },
    ],
    scenario: {
        flood: [{ kind: "rp100", label: "1-in-100-yr river flood", text: "none modelled in this tile" }],
        landslide: [],
        earthquake: [
            { kind: "earthquake_district", label: "Earthquake", text: "MEDIUM (district)" },
            { kind: "epicentres", label: "", text: "", data: { radius_km: 100, centre: [27.05, 88.26], bbox: [88.2, 27.0, 88.3, 27.1], events: [[88.1, 27.7, 6.9, 2011], [88.4, 27.2, 4.8, 1999]] } },
        ],
    },
};

test("Facts box: one row per fact, the Scenario Analysis link, no sources or badges", () => {
    const html = factsBodyHtml(normaliseInfo(DARJ));
    assert.match(html, /Wildfire hazard<\/span><span class="xp-fact-value">HIGH \(district\)/);
    assert.match(html, /Observatory Hill · 2,188 m · 0.8 km/);
    assert.match(html, /data-open-scenario>Scenario Analysis</);
    assert.doesNotMatch(html, /Source|licen|Derived from the DEM|not available|curated/i);
});

test("Facts box empty state when only scenario lines exist", () => {
    const html = factsBodyHtml(normaliseInfo({ facts: [], scenario: DARJ.scenario }));
    assert.match(html, /No other terrain hazards on record here\./);
    assert.match(html, /Scenario Analysis/);
});

test("Scenario cards: rows per option, epicentre map for earthquake, nothing for an empty option", () => {
    const info = normaliseInfo(DARJ);
    assert.match(scenarioCardHtml(info, "flood"), /none modelled in this tile/);
    assert.equal(scenarioCardHtml(info, "landslide"), "");
    const eq = scenarioCardHtml(info, "earthquake");
    assert.match(eq, /MEDIUM \(district\)/);
    assert.match(eq, /<svg[^>]+aria-label="2 M4\.5\+ epicentres within 100 km since 1973"/);
    assert.equal((eq.match(/class="xp-epi-dot/g) || []).length, 2);
    assert.match(eq, /xp-epi-dot is-strong/); // the M6.9
    assert.match(eq, /<rect [^>]*class="xp-epi-tile"/); // the tile outline
});

test("epicentre map needs events and a centre", () => {
    assert.equal(epicentreSvg({ radius_km: 100, centre: [1, 2], events: [] }), "");
    assert.equal(epicentreSvg({ radius_km: 100, events: [[1, 2, 5, 2000]] }), "");
    // no bbox (point input): a centre marker instead of a tile outline
    assert.match(epicentreSvg({ radius_km: 100, centre: [0, 0], events: [[0.1, 0.1, 5, 2000]] }), /<circle cx="110" cy="110" r="2.5" class="xp-epi-tile"/);
});

test("normalise and count ignore junk", () => {
    const info = normaliseInfo({ facts: [null, { kind: "peak", label: "P", text: "x" }, 3], scenario: { flood: "x" } });
    assert.equal(info.facts.length, 1);
    assert.deepEqual(info.scenario.flood, []);
    assert.equal(countLines(info), 1);
    assert.equal(countLines(emptyInfo()), 0);
});

test("query: bbox for georeferenced input, point for typed coordinates, none otherwise", () => {
    assert.deepEqual(infoQuery({ input: { geo: { lat: 1, lon: 2, bbox: [1, 1, 3, 3] } } }), { bbox: [1, 1, 3, 3] });
    assert.deepEqual(infoQuery({ input: { geo: { lat: 1, lon: 2 } } }), { lat: 1, lon: 2 });
    assert.deepEqual(infoQuery({ input: { geo: null }, userGeo: { lat: 5, lon: 6 } }), { lat: 5, lon: 6 });
    assert.equal(infoQuery({ input: { geo: null } }), null);
    assert.equal(queryKey({ lat: 5, lon: 6 }), "pt:5,6");
    const b = bboxFromCentre(27, 88, [10, 10]);
    assert.ok(Math.abs((b[3] - b[1]) * 110.574 - 10) < 0.01);
});

test("static library info resolves with no request", async () => {
    let calls = 0;
    const res = await loadGeoInfo({ input: { staticInfo: DARJ, geo: { lat: 1, lon: 2 } } }, () => { calls += 1; });
    assert.equal(res.source, "static");
    assert.equal(calls, 0);
    assert.equal(res.info.facts.length, 2);
});

test("live: one request per job and location; failures resolve empty and retry next time", async () => {
    const job = { input: { geo: { lat: 1, lon: 2, bbox: [1, 1, 3, 3] } } };
    const urls = [];
    const ok = url => { urls.push(url); return Promise.resolve({ ok: true, json: () => Promise.resolve(DARJ) }); };
    const a = await loadGeoInfo(job, ok);
    const b = await loadGeoInfo(job, ok);
    assert.equal(urls.length, 1);
    assert.match(urls[0], /\/api\/facts\?bbox=1,1,3,3$/);
    assert.equal(a, b);
    assert.equal(a.info.scenario.earthquake.length, 2);

    const job2 = { input: { geo: null }, userGeo: { lat: 5, lon: 6 } };
    const down = () => Promise.reject(new TypeError("Failed to fetch"));
    const failed = await loadGeoInfo(job2, down);
    assert.equal(failed.failed, true);
    assert.equal(countLines(failed.info), 0);
    assert.equal(job2.geoInfo, null); // not cached: the next open retries
    const http500 = () => Promise.resolve({ ok: false, status: 500 });
    assert.equal((await loadGeoInfo(job2, http500)).failed, true);
});
