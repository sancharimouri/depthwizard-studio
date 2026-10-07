import test from "node:test";
import assert from "node:assert/strict";

import { formatGsd, isReady, itemTerrain, orderLibraryItems, startLabel, tierClass } from "../src/input-view.js";

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

test("library order: Darjeeling, Almora, Manali first, then catalog order", () => {
    const items = ["dfc2019-A", "sentinel2-manali", "sentinel2-kota", "sentinel2-almora", "sentinel2-darjeeling"]
        .map(id => ({ id }));
    assert.deepEqual(orderLibraryItems(items).map(i => i.id),
        ["sentinel2-darjeeling", "sentinel2-almora", "sentinel2-manali", "dfc2019-A", "sentinel2-kota"]);
});

test("library order: a curated listing (order_index on every item) is followed as served", () => {
    const items = [
        { id: "sentinel2-almora", order_index: 2 },
        { id: "sentinel2-kota", order_index: 0 },
        { id: "sentinel2-darjeeling", order_index: 1 },
    ];
    assert.deepEqual(orderLibraryItems(items).map(i => i.id), ["sentinel2-kota", "sentinel2-darjeeling", "sentinel2-almora"]);
    // one item without an index: the pinned order applies, as before
    assert.deepEqual(orderLibraryItems([{ id: "sentinel2-kota", order_index: 0 }, { id: "sentinel2-darjeeling" }]).map(i => i.id),
        ["sentinel2-darjeeling", "sentinel2-kota"]);
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

test("card heading shortens: state first, then brackets", async () => {
    const { titleCandidates } = await import("../src/input-view.js");
    assert.deepEqual(titleCandidates("Kakinada (Godavari delta), AP"), ["Kakinada (Godavari delta)", "Kakinada"]);
    assert.deepEqual(titleCandidates("Darjeeling, West Bengal"), ["Darjeeling"]);
    assert.deepEqual(titleCandidates("OMA_269_035"), ["OMA_269_035"]);
});

test("card subheading drops what the heading shows, unbrackets a leading bracket", async () => {
    const { cardSubtitle, cardTerrain } = await import("../src/input-view.js");
    const k = { collection: "sentinel2", location: "Kakinada (Godavari delta), AP" };
    assert.equal(cardSubtitle(k, "Kakinada (Godavari delta), AP"), "India");
    assert.equal(cardSubtitle(k, "Kakinada (Godavari delta)"), "AP, India");
    assert.equal(cardSubtitle(k, "Kakinada"), "Godavari delta, AP, India");
    assert.equal(cardSubtitle({ collection: "dfc2019", location: "Omaha, Nebraska, USA" }, "OMA_269_035"), "Omaha, Nebraska, USA");
    assert.equal(cardTerrain({ terrain: "agricultural" }), "Argi");
    assert.equal(cardTerrain({ terrain: "coastal" }), "Coast");
    assert.equal(cardTerrain({ terrain: "hilly" }), "Hilly");
});

test("Maxar crops get display names; others untouched", async () => {
    const { withDisplayTitles, VHR_DISPLAY_TITLES } = await import("../src/input-view.js");
    const ids = Object.keys(VHR_DISPLAY_TITLES);
    assert.equal(ids.length, 6);
    const out = withDisplayTitles({ items: [...ids.map(id => ({ id, title: "x" })), { id: "sentinel2-kochi", title: "Kochi, Kerala" }] });
    assert.deepEqual(out.items.slice(0, 6).map(i => i.title), ["Sikkim Valley", "Sikkim Town", "Sikkim Terraces", "Sikkim Forest", "Sikkim River", "Sikkim Glacier"]);
    assert.equal(out.items[6].title, "Kochi, Kerala");
});

const STATIC_INDEX = new URL("../public/library-static/index.json", import.meta.url);
const { existsSync } = await import("node:fs");
test("every card text helper handles every catalog item", {
    skip: !existsSync(STATIC_INDEX) && "public/library-static/index.json is private (not in this checkout)",
}, async () => {
    const { titleCandidates, cardSubtitle, cardTerrain } = await import("../src/input-view.js");
    const { readFileSync } = await import("node:fs");
    const { items } = JSON.parse(readFileSync(STATIC_INDEX, "utf8"));
    assert.ok(items.length > 50);
    for (const item of items) {
        const [first] = titleCandidates(item.title);
        assert.ok(first, item.id);
        assert.ok(cardSubtitle(item, first).length > 0, item.id);
        assert.ok(cardTerrain(item).length > 0, item.id);
    }
});
