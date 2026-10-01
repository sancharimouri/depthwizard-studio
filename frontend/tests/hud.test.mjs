import test from "node:test";
import assert from "node:assert/strict";

import { easeOutCubic, formatCoord, lagCascade, screenCoord, settleFactor, smoothstep, tauWithin } from "../src/hud.js";

test("coordinates are formatted like the design: 4 decimals, hemisphere letters", () => {
    assert.equal(formatCoord(24.7863, 85.9412), "24.7863° N · 85.9412° E");
    assert.equal(formatCoord(-12.5, -70.25), "12.5000° S · 70.2500° W");
});

test("screen coordinates: the window centre is the anchor, up is north", () => {
    const c = { lat: 24.7863, lon: 85.9412 };
    const mid = screenCoord(c, 500, 400, 1000, 800, 0.001);
    assert.ok(Math.abs(mid.lat - c.lat) < 1e-9 && Math.abs(mid.lon - c.lon) < 1e-9);
    assert.ok(screenCoord(c, 500, 100, 1000, 800, 0.001).lat > c.lat);
    assert.ok(screenCoord(c, 900, 400, 1000, 800, 0.001).lon > c.lon);
});

function run(st, target, ms, dt = 16) {
    const out = [];
    for (let t = 0; t < ms; t += dt) {
        out.push(lagCascade(st, target, dt));
    }
    return out;
}

test("ruler motion: starts from rest, accelerates, then decelerates visibly and settles", () => {
    const st = [0, 0, 0];
    const path = run(st, 1, 22000);
    const speed = path.map((p, i) => (i ? p - path[i - 1] : p));
    assert.ok(speed[0] < speed[10] && speed[10] < speed[100], "speeds up from rest");
    const peakAt = speed.indexOf(Math.max(...speed));
    assert.ok(peakAt * 16 > 2000 && peakAt * 16 < 4000, `peak near 2 x tau, got ${peakAt * 16} ms`);
    // a long, gradual slow-down: speed at +3 s after the peak is a fraction of the peak, still decreasing steadily
    const later = speed[peakAt + Math.round(3000 / 16)];
    assert.ok(later < speed[peakAt] * 0.6 && later > speed[peakAt] * 0.05);
    for (let i = peakAt + 1; i < speed.length; i++) {
        assert.ok(speed[i] <= speed[i - 1] + 1e-12, "no re-acceleration after the peak");
    }
    assert.ok(Math.abs(path[path.length - 1] - 1) < 1e-3);
    // peak speed = 0.27 x distance / tau (about 0.18 per second for tau = 1.5 s)
    assert.ok(Math.abs(speed[peakAt] * 1000 / 16 - 0.18) < 0.02);
});

test("ruler motion: a new target mid-glide never makes the speed jump", () => {
    const st = [0, 0, 0];
    const path = run(st, 1, 3000);
    const before = path[path.length - 1] - path[path.length - 2];
    const next = lagCascade(st, -1, 16);          // retarget the other way
    const after = next - path[path.length - 1];
    assert.ok(Math.abs(after - before) < 0.002, `${before} -> ${after}`);
});

test("smoothstep and idle sweep easing", () => {
    assert.equal(smoothstep(0, 1, 2), 0);
    assert.equal(smoothstep(3, 1, 2), 1);
    assert.ok(Math.abs(smoothstep(1.5, 1, 2) - 0.5) < 1e-9);
    assert.equal(easeOutCubic(0), 0);
    assert.equal(easeOutCubic(1), 1);
    assert.ok(easeOutCubic(0.2) > 0.4);
});

test("a long move is sped up to settle within the limit; a short one keeps its pace", () => {
    // the cascade really does settle within the limit with the sped-up tau
    for (const eps of [1e-1, 1e-3, 1e-5, 1e-8]) {
        const tau = tauWithin(2850, 800, eps);
        const st = [0, 0, 0];
        let t = 0;
        let err = 1;
        while (err > eps && t < 20000) {
            err = 1 - lagCascade(st, 1, 4, tau);
            t += 4;
        }
        assert.ok(t <= 2900, `eps ${eps}: settled in ${t} ms with tau ${tau.toFixed(0)}`);
    }
    // small move: natural pace (800 ms) is already fast enough
    assert.equal(tauWithin(2850, 800, 0.5), 800);
    // a huge move is faster than the natural pace
    assert.ok(tauWithin(2850, 800, 1e-8) < 800);
    assert.ok(settleFactor(1e-3) > settleFactor(1e-1));
});
