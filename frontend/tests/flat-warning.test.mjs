import test from "node:test";
import assert from "node:assert/strict";
import { flatTerrainWarning, footprintKmFromBbox } from "../src/flat-warning.js";

const s2 = terrain => ({ collection: "sentinel2", terrain, footprintKm: [10.02, 10.07], gsdM: 10 });

test("flat Sentinel-2 classes get a class-specific warning with the exact footprint", () => {
    assert.match(flatTerrainWarning(s2("agricultural")).body, /10 m Sentinel-2 over 10\.0 × 10\.1 km of mostly flat farmland/);
    assert.match(flatTerrainWarning(s2("coastal")).body, /coastal land/);
    assert.match(flatTerrainWarning(s2("urban")).body, /urban area/);
});

test("hilly, VHR, DFC2019 and unknown inputs get no warning", () => {
    assert.equal(flatTerrainWarning(s2("hilly")), null);
    assert.equal(flatTerrainWarning({ ...s2("urban"), collection: "dfc2019" }), null);
    assert.equal(flatTerrainWarning({ ...s2("hilly"), collection: "vhr" }), null);
    assert.equal(flatTerrainWarning(null), null);
});

test("a searched scene (no class) is judged by its generated relief", () => {
    assert.equal(flatTerrainWarning(s2(null)), null);
    assert.equal(flatTerrainWarning(s2(null), 400), null);
    assert.match(flatTerrainWarning(s2(null), 40).body, /low-relief land/);
});

test("bbox footprint is in km", () => {
    const [x, y] = footprintKmFromBbox([88.0, 22.0, 88.1, 22.1]);
    assert.ok(Math.abs(x - 10.32) < 0.05 && Math.abs(y - 11.06) < 0.05, `${x} ${y}`);
});
