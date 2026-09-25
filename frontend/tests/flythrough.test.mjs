import { test } from "node:test";
import assert from "node:assert/strict";
import { fitDistance, freeHalfExtents } from "../src/flythrough.js";

test("free half-extents take the nearest obstruction on each axis", () => {
    const rect = { left: 0, top: 0, width: 1000, height: 800 };
    const h = freeHalfExtents(rect, { left: 300, right: 650, top: 100, bottom: 640 });
    assert.equal(h.halfW, 150); // right side is closer to the centre (500)
    assert.equal(h.halfH, 240); // bottom (640) is closer to the centre (400)
});

test("fit distance: a sphere just fits the tighter half-angle", () => {
    // unobstructed square canvas, 60° vertical FOV → half-angle 30°, sin = 0.5
    const d = fitDistance(10, 60, 800, 800, 400, 400);
    assert.ok(Math.abs(d - 20) < 1e-9);
    // halving the free width must push the camera further out
    assert.ok(fitDistance(10, 60, 800, 800, 200, 400) > d);
});
