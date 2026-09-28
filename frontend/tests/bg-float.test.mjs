import test from "node:test";
import assert from "node:assert/strict";
import { createFloater, wander, stepPush, kick, CM_PX } from "../src/bg-float.js";

test("the wander path stays inside its circle", () => {
    const f = createFloater(3, 5 * CM_PX);
    for (let t = 0; t < 600; t += 0.37) {
        const w = wander(f, t);
        assert.ok(Math.hypot(w.x, w.y) <= f.radius + 1e-9);
        assert.ok(Math.abs(w.rot) <= 7);
    }
});

test("the wander path actually covers its neighbourhood (not a tiny jitter)", () => {
    const f = createFloater(1, 4 * CM_PX);
    let far = 0;
    for (let t = 0; t < 300; t += 0.5) {
        far = Math.max(far, Math.hypot(wander(f, t).x, wander(f, t).y));
    }
    assert.ok(far > 0.6 * f.radius, `max reach ${far}`);
});

test("a nearby cursor pushes the icon away, and it springs back when the cursor leaves", () => {
    const f = createFloater(2, 150);
    const at = { x: 500, y: 300 };
    for (let i = 0; i < 60; i++) {
        stepPush(f, 1 / 60, { x: at.x + f.push.x, y: at.y + f.push.y }, { x: 450, y: 300 });
    }
    assert.ok(f.push.x > 10, `pushed right by ${f.push.x}`);
    for (let i = 0; i < 60 * 8; i++) {
        stepPush(f, 1 / 60, at, null);
    }
    assert.ok(Math.hypot(f.push.x, f.push.y) < 2, "settles back");
});

test("a click kicks nearby icons outward and leaves far ones alone", () => {
    const near = createFloater(4, 150);
    const far = createFloater(5, 150);
    kick(near, { x: 100, y: 0 }, { x: 0, y: 0 });
    kick(far, { x: 2000, y: 0 }, { x: 0, y: 0 });
    assert.ok(near.push.vx > 0 && Math.abs(near.push.vy) < 1e-9);
    assert.equal(far.push.vx, 0);
});

test("the push never exceeds its limit", () => {
    const f = createFloater(6, 150);
    for (let i = 0; i < 30; i++) {
        kick(f, { x: 1, y: 0 }, { x: 0, y: 0 });
        stepPush(f, 1 / 60, { x: 1, y: 0 }, { x: 0, y: 0 });
    }
    assert.ok(Math.hypot(f.push.x, f.push.y) <= 170 + 1e-6);
});
