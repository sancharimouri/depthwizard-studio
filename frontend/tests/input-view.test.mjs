import test from "node:test";
import assert from "node:assert/strict";

import { formatGsd, isReady, quotaLines, startLabel, tierClass } from "../src/input-view.js";

test("tier colour classes: green T2, orange T1, neutral otherwise", () => {
    assert.equal(tierClass({ tier: 2 }), "tier-2");
    assert.equal(tierClass({ tier: 1 }), "tier-1");
    assert.equal(tierClass({ tier: null }), "tier-none");
    assert.equal(tierClass(null), "tier-none");
    assert.match(startLabel({ tier: 1 }), /TIER 1/);
    assert.match(startLabel({ tier: null }), /RELATIVE PREVIEW/);
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
