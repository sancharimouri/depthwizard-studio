import test from "node:test";
import assert from "node:assert/strict";
import { railDragTarget, SIDEBAR_MIN_W, SIDEBAR_MAX_W, SIDEBAR_COLLAPSE_BELOW_W } from "../src/sidebar.js";

test("dragging the rail resizes within the limits", () => {
    assert.deepEqual(railDragTarget(300), { collapsed: false, width: 300 });
    assert.deepEqual(railDragTarget(900), { collapsed: false, width: SIDEBAR_MAX_W });
    // just under the minimum: held at the minimum, not collapsed
    assert.deepEqual(railDragTarget(SIDEBAR_MIN_W - 10), { collapsed: false, width: SIDEBAR_MIN_W });
});

test("dragging well below the minimum collapses the sidebar", () => {
    assert.deepEqual(railDragTarget(SIDEBAR_COLLAPSE_BELOW_W - 1), { collapsed: true });
    assert.deepEqual(railDragTarget(40), { collapsed: true });
});
