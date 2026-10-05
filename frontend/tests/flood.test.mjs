import { test } from "node:test";
import assert from "node:assert/strict";
import { inundationIndex, floodRange } from "../src/flood-sim.js";

// 4 cells at 100, 110, 120, 200 m (heights normalized to [100, 200])
const grid = { heights: [0, 0.1, 0.2, 1], elevationMin: 100, elevationMax: 200 };

test("fraction of cells strictly below the water level, from real metres", () => {
    const idx = inundationIndex(grid);
    assert.equal(idx.fractionBelow(100), 0);
    assert.equal(idx.fractionBelow(100.01), 0.25);
    assert.equal(idx.fractionBelow(115), 0.5);
    assert.equal(idx.fractionBelow(200), 0.75);
    assert.equal(idx.fractionBelow(1e9), 1);
});

test("percentile reads real elevations", () => {
    const idx = inundationIndex(grid);
    assert.equal(idx.percentile(0), 100);
    assert.equal(idx.percentile(1), 200);
});

test("does not mutate the grid", () => {
    const heights = [0.5, 0, 1];
    inundationIndex({ heights, elevationMin: 0, elevationMax: 10 });
    assert.deepEqual(heights, [0.5, 0, 1]);
});

test("flood slider range: lowest ground to 2 m above the highest, default half way", () => {
    assert.deepEqual(floodRange(554.3, 2477.6), { min: 554, max: 2480, half: 1517 });
});
