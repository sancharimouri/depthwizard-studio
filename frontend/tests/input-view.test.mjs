import test from "node:test";
import assert from "node:assert/strict";

import { formatGsd, isReady, itemTerrain, orderLibraryItems, quotaLines, startLabel, tierClass } from "../src/input-view.js";

test("tier colour classes: green T2, orange T1, neutral otherwise", () => {
    assert.equal(tierClass({ tier: 2 }), "tier-2");
    assert.equal(tierClass({ tier: 1 }), "tier-1");
    assert.equal(tierClass({ tier: null }), "tier-none");
    assert.equal(tierClass(null), "tier-none");
    // the button carries no tier (the routing card below it does)
    assert.equal(startLabel({ tier: 1 }), "▶ START GENERATION");
    assert.equal(startLabel({ tier: null }), "▶ START GENERATION");
});

test("GSD formatting", () => {
    assert.equal(formatGsd(0.305), "31 cm");
    assert.equal(formatGsd(2.4), "2.4 m");
    assert.equal(formatGsd(10), "10 m");
    assert.equal(formatGsd(30.71), "30.7 m");
    assert.equal(formatGsd(null), "unknown");
});

test("readiness: georeferenced inputs wait for their DEM", () => {
    assert.equal(isReady(null), false);
    assert.equal(isReady({ source: "library", routing: { tier: 1 } }), true);
    assert.equal(isReady({ source: "upload", routing: { tier: 2 }, dem: null }), false);
    assert.equal(isReady({ source: "upload", routing: { tier: 2 }, dem: { min_m: 1 } }), true);
    assert.equal(isReady({ source: "search", routing: { tier: 1 } }), false);
    assert.equal(isReady({ source: "upload", routing: { tier: null } }), true);
});

test("quota lines never invent a remaining count", () => {
    const lines = quotaLines({
        credentials: "project",
        account: { typology: "copernicus_general" },
        documented_limits: { requests_per_minute: 300, requests_per_month: 10000, processing_units_per_minute: 300,
            processing_units_per_month: 10000 },
        this_server_usage: { since: "2026-09-24T20:03:15Z", catalog_requests: 1, process_requests: 1,
            processing_units: 4.04, throttled: 0 },
        remaining: null,
    });
    assert.equal(lines.length, 4);
    assert.match(lines[1], /10,000 req\/month/);
    assert.match(lines[2], /4.04 PU/);
    assert.ok(lines.every(line => !/remaining/i.test(line)));
});

test("library order: Darjeeling, Almora, Manali first, then catalog order", () => {
    const items = ["dfc2019-A", "sentinel2-manali", "sentinel2-kota", "sentinel2-almora", "sentinel2-darjeeling"]
        .map(id => ({ id }));
    assert.deepEqual(orderLibraryItems(items).map(i => i.id),
        ["sentinel2-darjeeling", "sentinel2-almora", "sentinel2-manali", "dfc2019-A", "sentinel2-kota"]);
});

test("library terrain: manifest field, Sentinel-2 category, collection fallback", () => {
    assert.equal(itemTerrain({ terrain: "coastal", collection: "sentinel2" }), "coastal");
    assert.equal(itemTerrain({ category: "agricultural", collection: "sentinel2" }), "agricultural");
    assert.equal(itemTerrain({ collection: "dfc2019" }), "urban");
    assert.equal(itemTerrain({ collection: "vhr" }), "hilly");
});

import { filterLibrary, noMatchMessage } from "../src/input-view.js";

const LIB = [
    { collection: "sentinel2", terrain: "hilly" }, { collection: "sentinel2", terrain: "coastal" },
    { collection: "sentinel2", terrain: "agricultural" }, { collection: "sentinel2", terrain: "urban" },
    { collection: "vhr", terrain: "hilly" }, { collection: "dfc2019", terrain: "urban" },
];

test("library filter: collection and terrain combine", () => {
    assert.equal(filterLibrary(LIB, "all", "all").length, 6);
    assert.equal(filterLibrary(LIB, "sentinel2", "all").length, 4);
    assert.equal(filterLibrary(LIB, "all", "hilly").length, 2);
    assert.equal(filterLibrary(LIB, "dfc2019", "hilly").length, 0);
});

test("no-match message names the imagery that has that terrain", () => {
    assert.equal(noMatchMessage(LIB, "sentinel2", "hilly"), null);
    assert.equal(noMatchMessage(LIB, "dfc2019", "hilly"),
        "No hilly tiles in DFC2019 imagery. Try the Hilly filter with Maxar VHR or Sentinel-2 imagery.");
    assert.equal(noMatchMessage(LIB, "vhr", "coastal"),
        "No coastal tiles in Maxar VHR imagery. Try the Coastal filter with Sentinel-2 imagery.");
});

test("readiness: an upload with no geotransform waits for its GSD", () => {
    const upload = { source: "upload", routing: { tier: null }, gsdRequired: true };
    assert.equal(isReady(upload), false);
    assert.equal(isReady({ ...upload, gsdRequired: false }), true);
});
